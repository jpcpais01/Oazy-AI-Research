"""SmolLM2 from scratch: Hugging Face's SmolLM2 (a plain Llama) with RANDOM weights, trained on our text, so its speed
and its loss can be put next to ours. A NEXT-WORD model, like every Bee.

The body is the real one: transformers' LlamaForCausalLM, built from SmolLM2's own config.json (read from the Hub on
the pod; a copy of the numbers below is used when the Hub cannot be reached). Nothing of the trained SmolLM2 is kept:
the weights start random, the way Hugging Face started them (normal, std = the config's initializer_range).

                     SmolLM2-135M (default)          SmolLM2-360M (--set size=360m)
  width x layers     576 x 30                        960 x 32
  heads              9 query / 3 key-value, 64 each  15 query / 5 key-value, 64 each
  MLP                SwiGLU 1,536                    SwiGLU 2,560
  word table         49,152 pieces, tied             the same
  parameters         134.5M                          361.8M

Two ways to raise it (two names in oazy/train.py, one body):
  smol      their recipe, stock: AdamW (lr 3e-3, betas 0.9/0.95, weight decay 0.01), warm-up then flat then a straight
            fall over the last 20% (WSD), 2,048-word windows from the start, the full score grid in 32-bit for the loss
            (what LlamaForCausalLM does), no mentor.
  smolbee   our recipe (the latest best, Bee4/Bee5's): Muon on the weight grids + AdamW for the rest, windows
            512 -> 1,024 -> 2,048, the batch ramp, the square-root cool-down over the last 30%, the loss in slices
            (oazy/lmloss.py), the SmolLM2-360M mentor on a share of the passes.
Both: Sado, 524k-token steps, bf16, torch.compile, gradient clipping at 1.0.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

from oazy.sample import sample_logits

HUB = {"135m": "HuggingFaceTB/SmolLM2-135M", "360m": "HuggingFaceTB/SmolLM2-360M"}
# SmolLM2's config.json, the numbers that shape the body (used when the Hub cannot be reached)
SIZES = {
    "135m": dict(d_model=576, n_layer=30, attn_heads=9, attn_kv_heads=3, mlp_hidden=1536, rope_theta=100000.0,
                 init_std=0.041666666666666664),
    "360m": dict(d_model=960, n_layer=32, attn_heads=15, attn_kv_heads=5, mlp_hidden=2560, rope_theta=100000.0,
                 init_std=0.02),
}
STOCK_CTX = "2048:0"                       # SmolLM2 trained on 2,048-word windows throughout
OUR_CTX = "512:0,1024:0.15,2048:0.3"       # Bee4/Bee5's window warm-up


@dataclass
class SmolConfig:
    size: str = "135m"           # "135m" or "360m": the shape below comes from SmolLM2's config of that size
    stock: int = 1               # 1 = their recipe (smol), 0 = ours (smolbee); set by oazy/train.py
    hub: int = 1                 # 1 = read the shape from the Hub's config.json when it can (it is then written here)
    vocab_size: int = 49152
    tokenizer: str = "smollm2"
    max_len: int = 2048
    ctx: str = ""                # "" = the recipe's: 2,048 throughout (stock) or 512 -> 1,024 -> 2,048 (ours)
    d_model: int = 0             # 0 = SmolLM2's for the size
    n_layer: int = 0
    attn_heads: int = 0
    attn_kv_heads: int = 0
    mlp_hidden: int = 0
    rope_theta: float = 0.0
    init_std: float = 0.0
    rms_eps: float = 1e-5
    weight_decay: float = 0.01   # AdamW's, stock recipe (ours uses the trainer's Muon weight decay)

    def __post_init__(self):
        if self.size not in SIZES:
            raise ValueError(f"size must be one of {list(SIZES)}, not {self.size!r}")
        for k, v in SIZES[self.size].items():
            if not getattr(self, k):
                setattr(self, k, v)
        if not self.ctx:
            self.ctx = STOCK_CTX if self.stock else OUR_CTX

    @property
    def head_dim(self):
        return self.d_model // self.attn_heads

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def from_hub(self, log=print):
        """Read SmolLM2's config.json for this size from the Hub and take its shape (only the numbers; no weights).
        Returns True when it was read. Kept in the checkpoint, so a resume never needs the Hub."""
        try:
            from transformers import AutoConfig
            hc = AutoConfig.from_pretrained(HUB[self.size])
        except Exception as e:                       # no internet, no transformers: the copy in SIZES
            log(f"[smol] SmolLM2-{self.size}'s config.json not reachable ({type(e).__name__}); using the copy in smol/model.py")
            return False
        got = dict(d_model=hc.hidden_size, n_layer=hc.num_hidden_layers, attn_heads=hc.num_attention_heads,
                   attn_kv_heads=hc.num_key_value_heads, mlp_hidden=hc.intermediate_size, rope_theta=float(hc.rope_theta),
                   init_std=float(hc.initializer_range))
        diff = {k: (getattr(self, k), v) for k, v in got.items() if getattr(self, k) != v}
        for k, v in got.items():
            setattr(self, k, v)
        self.vocab_size, self.rms_eps = int(hc.vocab_size), float(hc.rms_norm_eps)
        log(f"[smol] shape read from {HUB[self.size]}'s config.json"
            + (f" (differs from the copy: {diff})" if diff else " (the same as the copy)"))
        return True

    def llama(self):
        from transformers import LlamaConfig
        return LlamaConfig(vocab_size=self.vocab_size, hidden_size=self.d_model, intermediate_size=self.mlp_hidden,
                           num_hidden_layers=self.n_layer, num_attention_heads=self.attn_heads,
                           num_key_value_heads=self.attn_kv_heads, head_dim=self.head_dim, hidden_act="silu",
                           max_position_embeddings=8192, rms_norm_eps=self.rms_eps, rope_theta=self.rope_theta,
                           initializer_range=self.init_std, tie_word_embeddings=True, attention_bias=False,
                           mlp_bias=False, bos_token_id=0, eos_token_id=0, use_cache=False)


MUON_NAMES = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
              "gate_proj.weight", "up_proj.weight", "down_proj.weight")


class Smol(nn.Module):
    """LlamaForCausalLM behind the lab's small model interface (losses, _body, ctx_at, generate, make_optimizer)."""
    micro_at_full = True        # --micro N = N sequences per pass at 2,048 words (Bee4's rule), shorter windows get more
    fused_loss = True           # ours: the loss in slices. The stock recipe always forms the full score grid
    loss_chunk = 1024

    def __init__(self, cfg: SmolConfig):
        super().__init__()
        from transformers import LlamaForCausalLM
        self.cfg = cfg
        self.hf = LlamaForCausalLM(cfg.llama())     # random weights, Hugging Face's own initialisation
        try:
            self.hf.config._attn_implementation = "sdpa"     # the fused attention kernel (FlashAttention on the GPU)
        except Exception:
            pass
        self.hf.tie_weights()
        self._compiled = None
        self.token_select = None

    @property
    def embed(self):
        return self.hf.model.embed_tokens

    # ---------------------------------------------------------------- sizes, names
    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())          # the tied word table once
        return n - self.embed.weight.numel() if non_embedding else n

    def layout(self):
        c = self.cfg
        return (f"SmolLM2-{c.size} from scratch ({'their recipe' if c.stock else 'our recipe'}): {c.n_layer} layers of "
                f"{c.d_model}, {c.attn_heads} query / {c.attn_kv_heads} key-value heads, SwiGLU {c.mlp_hidden}")

    def depth_text(self):
        return f"{self.cfg.n_layer} layers deep"

    def ctx_at(self, progress: float) -> int:
        from bee.model import ctx_at
        return ctx_at(self.cfg, progress)

    def enable_compile(self):
        self._compiled = torch.compile(self._trunk)

    def make_optimizer(self, lr, device, kind="adamw", muon_lr=0.03, muon_wd=0.1):
        """stock: AdamW on everything, weight decay on every grid (the word table too), none on the norms (as the
        Hugging Face trainer does). ours (kind="muon"): Muon on the attention and MLP grids, AdamW for the rest
        (the word table without decay, as Bee4)."""
        named = list(self.named_parameters())
        kw = dict(lr=lr, betas=(0.9, 0.95), eps=1e-8)
        if device.type == "cuda":
            kw["fused"] = True
        if kind != "muon":
            wd = self.cfg.weight_decay
            dec = [p for _, p in named if p.dim() >= 2]
            nod = [p for _, p in named if p.dim() < 2]
            return torch.optim.AdamW([{"params": dec, "weight_decay": wd, "lr_scale": 1.0},
                                      {"params": nod, "weight_decay": 0.0, "lr_scale": 1.0}], **kw)
        from oazy.muon import Muon, Combined
        grids = [p for n, p in named if p.dim() == 2 and n.endswith(MUON_NAMES)]
        gid = {id(p) for p in grids}
        rest = [p for _, p in named if id(p) not in gid]
        adam = torch.optim.AdamW([{"params": rest, "weight_decay": 0.0, "lr_scale": 1.0}], **kw)
        muon = Muon([{"params": grids, "lr_scale": muon_lr / lr}], lr=muon_lr, weight_decay=muon_wd)
        return Combined(muon, adam)

    # ---------------------------------------------------------------- the body
    def _trunk(self, idx):
        """(B, T) words -> (B, T, D) vectors after the final norm (what the tied score head reads)."""
        return self.hf.model(input_ids=idx, use_cache=False).last_hidden_state

    def _body(self, idx):
        """(B, T) -> (B, T, V) scores."""
        return self.hf.lm_head(self._trunk(idx))

    def doc_mask(self, idx):
        """The mentor's yes/no grid: earlier words of the same document (B, 1, T, T)."""
        from oazy.tok import eot_id
        B, T = idx.shape
        resets = torch.zeros(B, T, dtype=torch.bool, device=idx.device)
        resets[:, 1:] = idx[:, :-1] == eot_id(self.cfg.tokenizer)
        doc = resets.cumsum(1)
        return (doc[:, None, :, None] == doc[:, None, None, :]) & torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()

    def losses(self, idx, targets, mentor=None, kd_weight=0.0):
        """One lesson. stock: the full score grid, cross-entropy in 32-bit (LlamaForCausalLM's own way). ours: the
        loss in slices with the mentor's KL weighted kd_weight (oazy/lmloss.py, as Bee4)."""
        if self.token_select is not None:
            raise SystemExit("SmolLM2 from scratch does not take --select")
        h = (self._compiled or self._trunk)(idx)
        W = self.hf.lm_head.weight
        if self.cfg.stock:
            logits = F.linear(h, W.to(h.dtype)).float()
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
            return ce, {"ce": ce.detach()}
        th = tW = None
        if mentor is not None and kd_weight > 0:
            th, tW = mentor.hidden(idx, self.doc_mask(idx))
        from oazy.lmloss import head_loss, plain_loss
        if self.fused_loss:
            loss, ce, kd = head_loss(h, W, targets, 0.0, th, tW, kd_weight, self.loss_chunk)
        else:
            loss, ce, kd = plain_loss(h, W, targets, 0.0, th, tW, kd_weight)
        stats = {"ce": ce}
        if th is not None:
            stats["kd"] = kd
        return loss, stats

    def forward(self, idx, targets=None):
        if targets is None:
            return self._body(idx), None, {}
        loss, stats = self.losses(idx, targets)
        return None, loss, stats

    # ---------------------------------------------------------------- writing (samples, chat)
    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=100, temperature=0.8, top_k=40, eos=None, rep=1.0):
        dev = self.embed.weight.device
        x = torch.tensor([list(prompt_ids)], device=dev)
        o = self.hf(input_ids=x, use_cache=True)
        past, logits = o.past_key_values, o.logits
        out = []
        V = self.cfg.vocab_size
        for _ in range(max_new_tokens):
            nxt = sample_logits(logits[0, -1, :V], temperature, top_k, vocab=V, recent=(list(prompt_ids) + out)[-64:], rep=rep)
            if eos is not None and nxt == eos:
                break
            out.append(nxt)
            o = self.hf(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=past, use_cache=True)
            past, logits = o.past_key_values, o.logits
        return out, None
