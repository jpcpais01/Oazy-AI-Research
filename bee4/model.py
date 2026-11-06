"""Bee4: a 300M all-attention Bee, raised with Coral's recipe. Still a NEXT-WORD model: it reads text and guesses
the very next token, like GPT-2 and every Bee, and it is scored the same way.

Bee4 is the plain, dense sibling of Coral: the same layers, the same recipe, but one ordinary MLP per layer
instead of Coral's colony of experts, and no Memtrix memory layers. Every one of its ~296M parameters works on
every word.

                       Bee3                          Coral                          Bee4
  width                768                           1,024                          1,024
  layers               24 (18 memory + 6 attention)  20, all attention              20, all attention
  attention heads      12 query / 4 key-value        16 / 4, heads of 64            16 / 4, heads of 64
  positions            none (the memories know)      RoPE, base 50,000              RoPE, base 50,000
  thinking part        one MLP of 1,792              32 experts of 448 (top 2)      one MLP of 2,816
                                                     + a shared MLP of 512
  parameters           203M                          1,037M (161M at work)          296M (all at work)

Why no memory layers: at 2,048-word windows the Memtrix kernel was the slowest part of Bee3 and Frog, and its
long-text advantage does not show at that length. All attention runs on the GPU's fused attention kernel.

Coral's recipe (oazy/train.py RECIPE["bee4"], bee4/README.md):
  growing depth          8 layers for the first 10%, 12 until 25%, then 20; each growth copies the top layers
  the mentor             stronger than Coral's, and kept on: SmolLM2-1.7B (the same pieces) on 1 pass in 4 at the
                         start, 1 in 10 from 30%, fading to none over the last 10% of the run
  plain causal attention the documents in a window may see earlier ones (keeping them apart cost ~40% of a pass);
                         the mentor still reads each document on its own. --set attn_docs=1 brings the rule back.
  the rest               FP8 matrix products, the stored RoPE table, the loss in slices, Muon with weight decay,
                         a batch ramp, WSD with the square-root cool-down, Tagus20 with its chat-heavy cool-down
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from frog.model import Frog, FrogConfig

BEE4_GROW = "8:0,12:0.1,20:0.25"     # Coral's


@dataclass
class Bee4Config(FrogConfig):
    d_model: int = 1024
    n_layer: int = 20
    pattern: str = "A"           # attention in every layer
    attn_heads: int = 16         # 16 x 64 = the width
    attn_kv_heads: int = 4
    attn_head_dim: int = 64
    rope: int = 1                # all attention: RoPE is what tells it the word order
    rope_base: float = 50000.0
    mlp_hidden: int = 2816       # 2.75 x the width; makes the whole model 296M
    attn_docs: int = 0           # 1 = attention keeps to each word's own document (flex attention; ~40% slower on a B200)
    flex_attn: int = 0           # only used with attn_docs=1
    fused_gates: int = 0         # memory-layer switches: no memory layers here
    memory_op: int = 0
    grow: str = BEE4_GROW

    def __post_init__(self):
        if self.grow == BEE4_GROW and self.n_layer != 20:      # another depth (a test's small body): no growing unless asked
            self.grow = f"{self.n_layer}:0"
        if self.attn_docs:
            self.flex_attn = 1


class Bee4(Frog):
    """296M, all attention, dense. Grows like Coral (8 -> 12 -> 20 layers)."""

    def layout(self):
        c = self.cfg
        return f"{len(c.kinds())} layers, all attention ({c.attn_heads} query / {c.attn_kv_heads} key-value heads), MLP {c.mlp_hidden}"

    def depth_text(self):
        return f"{self.depth} layers deep (all attention)"

    def prepare_kernels(self, device):
        if "M" in self.cfg.pattern:                 # the memory kernels' checks only matter with memory layers
            super().prepare_kernels(device)

    def doc_info(self, idx):
        """Where documents start and the same-document grid (for the mentor, and attention when attn_docs=1). With
        attention only, the memory kernel's document offsets are not needed."""
        if "M" in self.cfg.pattern or not self.cfg.doc_mask:
            return super().doc_info(idx)
        from oazy.tok import eot_id
        B, T = idx.shape
        resets = torch.zeros(B, T, dtype=torch.bool, device=idx.device)
        resets[:, 1:] = idx[:, :-1] == eot_id(self.cfg.tokenizer)
        doc = resets.cumsum(1)
        mask = (doc[:, None, :, None] == doc[:, None, None, :]) & torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
        return resets, mask, None

    def attn_mask(self, idx, resets, mask):
        """What the attention layers get: nothing (plain causal, the fused kernel) unless attn_docs or memory layers.
        The same in training and reading, so the held-out loss measures what was trained."""
        if not self.cfg.attn_docs and "M" not in self.cfg.pattern:
            return None
        return super().attn_mask(idx, resets, mask)

    def _body(self, idx):
        resets, mask, segs = self.doc_info(idx)
        return self.head(self._trunk(idx, resets, self.attn_mask(idx, resets, mask), segs))
