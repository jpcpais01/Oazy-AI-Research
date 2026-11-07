"""Coral Mini: Coral's mixture of experts shrunk to Bee4's size (about 200M parameters), raised with Bee4's recipe.
Still a NEXT-WORD model: it reads text and guesses the very next token, like GPT-2, every Bee and Coral, and it is
scored the same way.

Coral Mini is literally Coral made small: the same layer (gated attention with RoPE, then a shared MLP plus a colony
of experts), the same router (sigmoid scores, the top 2 experts per word), the same balancing (DeepSeek-V3 biases and
a tiny balance loss), the same fixed room per expert in training and nothing dropped when reading. Its shape is
Bee4's: width 1,024, 12 layers, and the same total thinking width per layer (2,816), only cut into pieces:

                       Coral                          Bee4                    Coral Mini
  width                1,024                          1,024                   1,024
  layers               20, all attention              12, all attention       12, all attention
  thinking part        32 experts of 448 (top 2)      one MLP of 2,816        16 experts of 160 (top 2)
                       + a shared MLP of 512                                  + a shared MLP of 256
  parameters           1,037M (211M at work)          198M (all at work)      about 198M (about 115M at work)

(16 x 160 + 256 = 2,816: the same parameters as Bee4's one MLP. Each word uses 2 x 160 + 256 = 576 of that width.)

Its training is Bee4's (oazy/train.py RECIPE["coralmini"] = RECIPE["bee4"]): growing depth 6 -> 9 -> 12, short
windows first, the SmolLM2-360M mentor kept on 1 pass in 10 all run, the cool-down over the last 30%, bf16, 524k-token
steps, Tagus20. What an expert model needs on top: the balancing biases move after every step (model.after_step),
the balance loss (1e-4), Muon also on every expert's grids, and the fixed room (1.25x a fair share) in training.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from bee4.model import BEE4_CTX
from coral.model import Coral, CoralConfig

MINI_GROW = "6:0,9:0.1,12:0.25"      # Bee4's timing (Coral: 8 -> 12 -> 20)


@dataclass
class CoralMiniConfig(CoralConfig):
    d_model: int = 1024
    n_layer: int = 12            # Bee4's depth
    pattern: str = "A"           # attention in every layer
    attn_heads: int = 16
    attn_kv_heads: int = 4
    attn_head_dim: int = 64
    rope: int = 1
    rope_base: float = 50000.0
    mlp_hidden: int = 256        # the shared expert (every word)
    n_expert: int = 16           # routed experts per layer
    top_k: int = 2               # experts per word
    expert_hidden: int = 160     # each routed expert's SwiGLU width (16 x 160 + 256 = Bee4's 2,816)
    grow: str = MINI_GROW
    ctx: str = BEE4_CTX          # Bee4's short windows first: 512 until 15%, 1,024 until 30%, then 2,048
    attn_backend: str = "auto"   # Bee4's speed switches
    compile_mode: str = "default"

    def __post_init__(self):
        if self.grow == MINI_GROW and self.n_layer != 12:      # another depth (a test's small body): no growing unless asked
            self.grow = f"{self.n_layer}:0"
        super().__post_init__()


class CoralMini(Coral):
    """About 198M parameters, about 115M at work per word. Grows like Bee4 (6 -> 9 -> 12 layers)."""
    micro_at_full = True     # as Bee4: --micro N = N sequences per pass at the longest window (2,048)

    def enable_compile(self):
        super().enable_compile()
        mode = self.cfg.compile_mode
        if mode and mode != "default":
            self._compiled = torch.compile(self._train_trunk, mode=mode)

    def depth_text(self):
        return f"{self.depth} layers deep (all attention, {self.cfg.n_expert} experts, top {self.cfg.top_k})"

    def _body(self, idx):
        """Bee4's fix: reading (HellaSwag, chat, the full scores) uses the same attention rule as training (plain causal
        unless attn_docs), so every score measures what was trained. Coral's reading kept the document rule."""
        resets, mask, segs = self.doc_info(idx)
        return self.head(self._trunk(idx, resets, self._attn_mask(idx, resets, mask), segs))
