"""Seed: a small thinking core that grows (Seed & Bloom), and a big library of facts it looks things up in.

Still a NEXT-WORD model, a plain one: it reads text and guesses the very next token, like GPT-2 and every Bee, and it
is scored the same way (held-out loss per GPT-2 token, HellaSwag, ARC-Easy, PIQA, LAMBADA, LogT).

João's design (Special Lab, Nov 9). The idea: a normal model spends its weights both remembering things and working
things out. Seed splits the two. Facts go into a library that is cheap to read; the core is left to learn rules.

  core      a dense transformer with Bee5's tricks (seed/README.md lists them), 12 layers of 768 once grown, ~71M
  library   a product-key memory (Lample et al. 2019; Meta's "Memory Layers at Scale", 2024): 1,152 x 1,152 =
            1,327,104 rows of 768 numbers, ~1.0B weights, one table read by two layers of the core. For each word a
            layer asks a question, compares its two halves with 1,152 half-keys each, and pulls back the best 16 rows
            for each of 4 heads (64 rows of 1.3M). Only the rows it reads are trained. Full size from day one
  word table SmolLM2's 49,152 pieces x 768 (37.7M), shared with the score head (as every Bee)
  no experts: the library does the "many specialists, few at work" job more simply

Seed & Bloom: the core starts tiny and is copied bigger three times, KEEPING ITS FUNCTION EXACTLY (the loss does not
jump at a growth; seed/tests.py checks the scores before and after agree to rounding):

  4 layers x 384  for the first 20% of the tokens
  6 layers x 512  for the next 20%
  9 layers x 640  for the next 20%
  12 layers x 768 for the last 40%           (the library never changes)

How a growth keeps the function (Net2Net, Chen et al. 2016; masked structural growth, Yao et al. 2023):
  wider      the stream gets new numbers that start at 0. Bee5's norms have no learned gains, so a stream [x, 0] of
             d2 numbers normalises to sqrt(d2/d) times what x did: every weight that reads the stream is divided by that
             factor and every weight that writes into it is multiplied by it, and the word table (input and score head)
             is divided by it. New heads, new MLP neurons and new stream numbers read the old ones through fresh random
             weights, but everything they write starts at zero, so the model computes exactly what it did.
  deeper     new layers are copies of their neighbours' "reading" weights (questions, keys, values, MLP input) with
             their outputs at zero (Bee5 starts every layer that way): a no-op on the first step that already knows
             what to look at. They are slotted in symmetrically (just inside the first and last layers, and at the
             middle), so the U-net skips (layer i -> layer n-1-i) stay paired; a pair with a new layer starts at 0.
  roles      the per-layer roles (value tables, shifted keys, the back-out, the two library readers) are fixed by where
             a layer will sit once fully grown, Bee5's 12-layer pattern; the bloom just reveals the layers in turn.

Learning (seed/optim.py): NorMuon on the core with Grokfast (Lee et al. 2024: the slow, steady part of the gradient
is amplified) and twice Bee5's weight decay, so the core prefers simple rules; the library learns fast with a
row-wise Adam (rows not read this step do not move); Bee5's Adam groups for the rest. Bee5's recipe otherwise.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bee5.model import (Attention as Bee5Attention, Bee5, BEE5_CTX, GATE_IN, ROPE_TABLE, MLP as Bee5MLP, rope_table)

def norm(x):
    """Bee5's norm (no learned gain) with a negligible eps: a stream padded with zeros then normalises to exactly
    sqrt(d2/d) times what it did, even for the small word vectors (the default eps would not scale with it)."""
    return F.rms_norm(x, (x.size(-1),), eps=1e-12)


SEED_BLOOM = "4x384:0,6x512:0.2,9x640:0.4,12x768:0.6"      # João, Nov 9: 20% / 20% / 20% / 40%


@dataclass
class SeedConfig:
    vocab_size: int = 49152      # SmolLM2's pieces (the mentor's too)
    tokenizer: str = "smollm2"
    max_len: int = 2048
    ctx: str = BEE5_CTX          # windows 512 -> 1,024 -> 2,048 (Bee4/Bee5)
    d_model: int = 768           # the core once fully grown
    n_layer: int = 12
    bloom: str = SEED_BLOOM      # "layers x width : from this share of the run", one stage per comma; "12x768:0" = no growing
    hold: int = 0                # 1 = stay the seed (4x384 + the whole library) for the whole run: the "does a tiny core
                                 # grok?" side test. The layers keep the roles they would have in the grown core
    attn_head_dim: int = 64      # heads = width / 64, key-value heads = heads / kv_ratio (12 / 6 when grown)
    kv_ratio: int = 2
    mlp_ratio: float = 3.5       # ReLU² neurons per stream number (2,688 at 768)
    mlp: str = "relu2"
    rope_wave: float = 1024.0
    rope_frac: float = 0.5       # half-truncated RoPE
    n_ve: int = 2                # value tables on the first two and last two layers (of the grown core)
    unet: int = 1
    x0: int = 1
    smear: int = 1
    backout: int = 1
    attn_gate: int = 1
    key_offset: str = "3,7,11"   # positions in the grown core whose keys carry the previous word in their still half
    cap: str = "sig"
    softcap: float = 15.0
    mtp_until: float = 1 / 6     # the early 2nd/3rd-next-word lesson (Bee5)
    doc_mask: int = 1
    attn_backend: str = "auto"
    compile_mode: str = "default"
    # the library
    lib_side: int = 1152         # half-keys per half: rows = lib_side ** 2 (1,327,104)
    lib_dim: int = 768           # numbers per row (fixed: the library never changes when the core grows)
    lib_heads: int = 4
    lib_k: int = 16              # rows picked per head (64 per word per reading layer)
    lib_qdim: int = 256          # a head's question: two halves of 128
    lib_at: str = "4,7"          # the reading layers, as positions in the grown core (they must exist from day one)
    lib_init: float = 1.0        # rows start as normal(0, lib_init); a reading layer's output projection starts at zero
    lib_lr: float = 1.0          # the library's Adam rate, times --lr
    # learning
    grokfast: int = 1            # 1 = Grokfast on the core + core_wd; 0 = plain NorMuon with Bee5's weight decay
    gf_alpha: float = 0.98       # Grokfast-EMA (Lee et al. 2024): the slow average of the core's gradients ...
    gf_lamb: float = 2.0         # ... added back this many times
    core_wd: float = 0.2         # the core's weight decay (Bee5: 0.1) while Grokfast is on

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def kinds(self):
        return ["A"] * self.n_layer

    def stages(self):
        """[(layers, width, from share of the run)]."""
        out = []
        for part in str(self.bloom).split(","):
            shape, start = part.split(":")
            n, d = shape.lower().split("x")
            out.append((int(n), int(d), float(start)))
        out.sort(key=lambda s: s[2])
        if out[0][2] != 0:
            raise ValueError(f"bad bloom {self.bloom!r}: the first stage must start at 0")
        if out[-1][:2] != (self.n_layer, self.d_model):
            raise ValueError(f"bad bloom {self.bloom!r}: the last stage must be {self.n_layer}x{self.d_model}")
        unit = self.attn_head_dim * self.kv_ratio
        for (n, d, _), (n2, d2, _) in zip(out, out[1:]):
            if n2 < n or d2 < d:
                raise ValueError(f"bad bloom {self.bloom!r}: stages only grow")
            if n >= 1 and n2 > n and n < 2:
                raise ValueError(f"bad bloom {self.bloom!r}: a core grows deeper from 2 layers or more")
        for n, d, _ in out:
            if d % unit or abs(self.mlp_ratio * d - round(self.mlp_ratio * d)) > 1e-9 or d < GATE_IN:
                raise ValueError(f"bad bloom {self.bloom!r}: widths must be multiples of {unit} (and mlp_ratio x width whole)")
        return out

    def dims(self, d):
        """heads, key-value heads, MLP neurons at width d."""
        H = d // self.attn_head_dim
        return H, H // self.kv_ratio, int(round(self.mlp_ratio * d))


# ---------------------------------------------------------------------------------------------------- the bloom's plan
def insert_plan(n: int, k: int):
    """Where k new layers go in a core of n: [("old", i) | ("new", the old layer it copies)], in order. Half of them
    just after the first layer, half just before the last, one at the middle if k is odd: symmetric, so the U-net
    pairs (i, n-1-i) stay pairs. A new layer copies the old layer that comes right after it (it reads the same stream)."""
    if k == 0:
        return [("old", i) for i in range(n)]
    assert n >= 2, "grow from 2 layers or more"
    outer, mid = k // 2, k % 2
    mid_after = n // 2 - 1 if n % 2 == 0 else n // 2
    seq = []
    for i in range(n):
        seq.append(("old", i))
        if i == 0:
            seq += [("new", 1)] * outer
        if mid and i == mid_after:
            seq.append(("new", min(i + 1, n - 1)))
        if i == n - 2:
            seq += [("new", n - 1)] * outer
    return seq


def bloom_plan(cfg: SeedConfig):
    """orders: each stage's layers (as labels, seed layers 0..n0-1, then new ones numbered in order of birth),
    sources: label -> the label it copies, final: label -> its position in the grown core."""
    st = cfg.stages()
    order = list(range(st[0][0]))
    orders, sources, nxt = [list(order)], {}, len(order)
    for n2, _, _ in st[1:]:
        n = len(order)
        new = []
        for kind, i in insert_plan(n, n2 - n):
            if kind == "old":
                new.append(order[i])
            else:
                sources[nxt] = order[i]
                new.append(nxt)
                nxt += 1
        if cfg.unet:              # every pair of two old layers was a pair before (so its skip carries over)
            pos = {lab: p for p, lab in enumerate(order)}
            for i in range(n2 // 2):
                a, b = new[i], new[n2 - 1 - i]
                if a in pos and b in pos:
                    assert pos[a] < n // 2 and pos[b] == n - 1 - pos[a], "the bloom broke a U-net pair"
        order = new
        orders.append(list(order))
    return orders, sources, {lab: p for p, lab in enumerate(order)}


def roles(cfg: SeedConfig):
    """label -> dict(ve, offset, backout, lib), from the position the layer will have in the grown core."""
    orders, _, final = bloom_plan(cfg)
    N = cfg.n_layer
    offs = {int(s) for s in str(cfg.key_offset).split(",") if s.strip()}
    libs = {int(s) for s in str(cfg.lib_at).split(",") if s.strip()}
    back = max(0, round(2 * N / 3) - 1)
    seed = set(orders[0])
    out = {}
    for lab, p in final.items():
        ve = None
        for j in range(cfg.n_ve):
            if p in (j, N - cfg.n_ve + j):
                ve = j
        out[lab] = dict(ve=ve, offset=p in offs, backout=bool(cfg.backout) and p == back, lib=p in libs)
    for name, ps in (("library reader", libs), ("back-out layer", {back} if cfg.backout else set())):
        for p in ps:
            lab = orders[-1][p]
            if lab not in seed:
                raise ValueError(f"the {name} at position {p} of the grown core is born in the bloom; it must be one of "
                                 f"the seed's layers (positions {sorted(final[s] for s in seed)})")
    return out


# ---------------------------------------------------------------------------------------------------- the parts
class Attention(Bee5Attention):
    """Bee5's attention at an explicit size (its forward, step, gates, key offset, value tables are Bee5's)."""

    def __init__(self, cfg: SeedConfig, d: int, ve: bool, offset: bool):
        nn.Module.__init__(self)
        H, KV, _ = cfg.dims(d)
        hd = cfg.attn_head_dim
        self.H, self.KV, self.hd, self.cfg = H, KV, hd, cfg
        self.q = nn.Linear(d, H * hd, bias=False)
        self.k = nn.Linear(d, KV * hd, bias=False)
        self.v = nn.Linear(d, KV * hd, bias=False)
        self.out = nn.Linear(H * hd, d, bias=False)
        self.qn = nn.Parameter(torch.ones(hd))
        self.kn = nn.Parameter(torch.ones(hd))
        self.gate = nn.Linear(GATE_IN, H, bias=False) if cfg.attn_gate else None
        self.ve_gate = nn.Linear(GATE_IN, KV, bias=False) if ve else None
        self.offset = offset
        if offset:
            n_rot = max(1, int(hd // 2 * cfg.rope_frac))
            still = torch.zeros(hd, dtype=torch.bool)
            still[n_rot:hd // 2] = True
            still[hd // 2 + n_rot:] = True
            self.register_buffer("still", still, persistent=False)


class MLP(Bee5MLP):
    def __init__(self, cfg: SeedConfig, d: int):
        nn.Module.__init__(self)
        hid = cfg.dims(d)[2]
        self.kind = cfg.mlp
        self.up = nn.Linear(d, hid * (2 if cfg.mlp == "swiglu" else 1), bias=False)
        self.down = nn.Linear(hid, d, bias=False)


@torch.compiler.disable
def _bag(values, idx, w):
    """sum_j w[n, j] * values[idx[n, j]] without forming the (N, picks, dim) block (Alpha Star's fact drawer)."""
    return F.embedding_bag(idx, values, per_sample_weights=w.to(values.dtype), mode="sum")


class Library(nn.Module):
    """The facts: lib_side^2 rows of lib_dim numbers, and the half-keys that address them (shared by every reader)."""

    def __init__(self, cfg: SeedConfig):
        super().__init__()
        H, n, half = cfg.lib_heads, cfg.lib_side, cfg.lib_qdim // 2
        assert cfg.lib_qdim % 2 == 0
        self.H, self.n, self.half, self.k = H, n, half, cfg.lib_k
        self.keys = nn.Parameter(torch.randn(H, 2, n, half) / math.sqrt(half))
        self.values = nn.Parameter(torch.empty(n * n, cfg.lib_dim))
        self.init_std = cfg.lib_init
        self.reset_values()
        self.track = None              # (rows,) pick counts while a probe watches
        self.off = False               # the probe's "library closed" switch

    @torch.no_grad()
    def reset_values(self):
        if self.values.device.type != "meta":
            self.values.normal_(0.0, self.init_std)

    def rows(self):
        return self.values.shape[0]

    def pick(self, q):
        """q (N, H, 2, half) normed questions -> (N, H*k) row numbers and (N, H*k) weights."""
        N = q.shape[0]
        s = torch.einsum("nhcd,hckd->nhck", q, self.keys.to(q.dtype))      # (N, H, 2, n)
        k = min(self.k, self.n)
        s1, i1 = s[:, :, 0].topk(k, -1)
        s2, i2 = s[:, :, 1].topk(k, -1)
        pair = (s1[..., :, None] + s2[..., None, :]).reshape(N, self.H, k * k)
        top, j = pair.topk(min(self.k, k * k), -1)
        idx = i1.gather(-1, j // k) * self.n + i2.gather(-1, j % k)
        w = torch.softmax(top.float(), -1)
        return idx.reshape(N, -1), w.reshape(N, -1)


class Lookup(nn.Module):
    """One reading layer: y = out( rows(question(x)) * silu(gate(x)) ) (Memory Layers at Scale's gated read).
    The question is batch-normed (Lample et al.); it is in eval mode (running averages) whenever the model is."""

    def __init__(self, cfg: SeedConfig, d: int):
        super().__init__()
        self.query = nn.Linear(d, cfg.lib_heads * cfg.lib_qdim, bias=False)
        # batch-normed questions (Lample et al. 2019 found it is what keeps many rows in use): every number of the
        # question is centred and scaled over the words, so the questions spread over the keys instead of crowding
        self.qbn = nn.BatchNorm1d(cfg.lib_heads * cfg.lib_qdim)
        self.gate = nn.Linear(d, cfg.lib_dim, bias=False)
        self.out = nn.Linear(cfg.lib_dim, d, bias=False)
        self.H, self.half = cfg.lib_heads, cfg.lib_qdim // 2

    def forward(self, u, lib: Library):
        shp = u.shape
        x = u.reshape(-1, shp[-1])
        if lib.off:
            return torch.zeros_like(u)
        bn, qf = self.qbn, self.query(x).float()
        learn = self.training and torch.is_grad_enabled()       # reading without learning uses the running averages
        q = F.batch_norm(qf, bn.running_mean, bn.running_var, bn.weight, bn.bias, learn, bn.momentum, bn.eps)
        if learn:
            bn.num_batches_tracked += 1
        q = q.to(x.dtype).view(x.shape[0], self.H, 2, self.half)
        idx, w = lib.pick(q)
        if lib.track is not None:
            lib.track += torch.bincount(idx.reshape(-1), minlength=lib.track.numel())
        rows = _bag(lib.values, idx, w).to(u.dtype)
        return self.out(rows * F.silu(self.gate(x))).reshape(shp)


class Block(nn.Module):
    """A Bee5 layer (attention + ReLU² MLP), its own x0 mix, and for the two reading layers a library read after it."""

    def __init__(self, cfg: SeedConfig, d: int, label: int, role: dict):
        super().__init__()
        self.kind = "A"
        self.label = label
        self.ve = role["ve"]
        self.backout = role["backout"]
        self.attn = Attention(cfg, d, role["ve"] is not None, role["offset"])
        self.mlp = MLP(cfg, d)
        self.look = Lookup(cfg, d) if role["lib"] else None
        self.lam = nn.Parameter(torch.ones(()))       # the stream's weight at this layer's input (Bee5's lambdas) ...
        self.mu = nn.Parameter(torch.zeros(()))       # ... and the first input's (x0); they travel with the layer

    def forward(self, x, ve, cos, sin, lib):
        x = x + self.attn(norm(x), ve, cos, sin)
        x = x + self.mlp(norm(x))
        if self.look is not None:
            x = x + self.look(norm(x), lib)
        return x

    def step(self, x, ve, cos, sin, state, lib, looks=False):
        y, state, info = self.attn.step(norm(x), ve, cos, sin, state, looks)
        x = x + y
        x = x + self.mlp(norm(x))
        if self.look is not None:
            x = x + self.look(norm(x), lib)
        return x, state, info


# ---------------------------------------------------------------------------------------------------- Seed
class Seed(Bee5):
    micro_at_full = True
    fused_loss = True
    loss_chunk = 1024

    def __init__(self, cfg: SeedConfig, stage: int = 0):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.plan = cfg.stages()
        self.orders, self.sources, self.final = bloom_plan(cfg)
        self.roles = roles(cfg)
        n, d, _ = self.plan[stage]
        H, KV, _ = cfg.dims(d)
        kvd = KV * cfg.attn_head_dim
        self.embed = nn.Embedding(cfg.vocab_size, d)
        self.ve = nn.ModuleList(nn.Embedding(cfg.vocab_size, kvd) for _ in range(cfg.n_ve))
        self.blocks = nn.ModuleList(Block(cfg, d, lab, self.roles[lab]) for lab in self.orders[stage])
        n_skip = n // 2 if cfg.unet else 0
        self.skip_lam = nn.Parameter(torch.full((n_skip,), 0.18))    # Bee5's 0.18 at the start, as a plain weight
        self.skip_gate = nn.ModuleList(nn.Linear(GATE_IN, 1, bias=False) for _ in range(n_skip))
        self.smear_lam = nn.Parameter(torch.zeros(1))
        self.smear_gate = nn.Linear(GATE_IN, 1, bias=False) if cfg.smear else None
        self.backout_lam = nn.Parameter(torch.full((1,), 0.5 if cfg.backout else 0.0), requires_grad=bool(cfg.backout))
        self.lib = Library(cfg)
        cos, sin = rope_table(ROPE_TABLE, cfg.attn_head_dim, cfg.rope_wave, cfg.rope_frac)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("bloom_stage", torch.tensor(stage, dtype=torch.long))    # saved with the weights
        self._stage = stage
        self.reset_parameters()
        self._compiled = None
        self._progress = 0.0
        self.token_select = None
        self._carry = []              # after a growth: (old parameter, new parameter, gradient factor) for the optimizer

    # ---------------------------------------------------------------- sizes, names
    @property
    def stage(self) -> int:
        return self._stage

    @property
    def depth(self) -> int:
        return len(self.blocks)

    @property
    def width(self) -> int:
        return self.embed.weight.shape[1]

    @torch.no_grad()
    def reset_parameters(self):
        d = self.width
        nn.init.normal_(self.embed.weight, std=0.005)
        for t in self.ve:
            nn.init.zeros_(t.weight)
        for b in self.blocks:
            self._init_block(b, d)
        for g in list(self.skip_gate) + ([self.smear_gate] if self.smear_gate is not None else []):
            nn.init.zeros_(g.weight)

    @staticmethod
    @torch.no_grad()
    def _init_block(b, d):
        bound = 3 ** 0.5 * d ** -0.5
        for lin in (b.attn.q, b.attn.k, b.attn.v):
            nn.init.uniform_(lin.weight, -bound, bound)
        nn.init.uniform_(b.mlp.up.weight, -0.5 * bound, 0.5 * bound)
        nn.init.zeros_(b.attn.out.weight)
        nn.init.zeros_(b.mlp.down.weight)
        for g in (b.attn.gate, b.attn.ve_gate):
            if g is not None:
                nn.init.zeros_(g.weight)
        if b.look is not None:
            nn.init.uniform_(b.look.query.weight, -bound, bound)
            nn.init.uniform_(b.look.gate.weight, -bound, bound)
            nn.init.zeros_(b.look.out.weight)

    def library_params(self):
        return self.lib.values.numel() + self.lib.keys.numel()

    def core_params(self):
        return sum(p.numel() for n, p in self.named_parameters() if n.startswith("blocks."))

    def last_stage(self) -> int:
        return 0 if self.cfg.hold else len(self.plan) - 1

    def full_params(self):
        """Every parameter of Seed at the end of the run (fully grown; the seed itself with hold=1)."""
        if self.stage == self.last_stage():
            return self.num_params()
        with torch.device("meta"):
            return Seed(self.cfg, stage=self.last_stage()).num_params()

    def active_params(self):
        """At work for each word at the end of the run: everything but the lookup tables (the library's rows, the
        value tables), plus the rows each word reads from them. The word table counts whole (it is also the score head)."""
        c = self.cfg
        s = self.last_stage()
        KV = c.dims(self.plan[s][1])[1]
        here = [self.roles[lab] for lab in self.orders[s]]
        reads = sum(r["lib"] for r in here) * c.lib_heads * c.lib_k * c.lib_dim
        ve_reads = sum(r["ve"] is not None for r in here) * KV * c.attn_head_dim
        return (self.full_params() - self.lib.values.numel() - c.n_ve * c.vocab_size * KV * c.attn_head_dim
                + reads + ve_reads)

    def layout(self):
        c = self.cfg
        n, d, _ = self.plan[self.stage]
        stages = " -> ".join(f"{a}x{b}" for a, b, _ in self.plan)
        grown = (f"core held at {self.plan[0][0]} layers of {self.plan[0][1]} (no growing)" if c.hold else
                 f"core {c.n_layer} layers of {c.d_model} once grown ({stages})")
        return (f"{grown}, all attention, ReLU², {c.n_ve} value tables; "
                f"library {self.lib.rows():,} rows x {c.lib_dim} read by {sum(r['lib'] for r in self.roles.values())} layers "
                f"({c.lib_heads} heads x {c.lib_k} rows); now {n}x{d}")

    def depth_text(self):
        return f"{self.depth} layers of {self.width}"

    def enable_compile(self):
        self._compile_on = True
        super().enable_compile()

    # ---------------------------------------------------------------- the body
    def _skip(self, j, x0):
        return self.skip_lam[j] * 2 * torch.sigmoid(self.skip_gate[j](x0[..., :GATE_IN]))

    def _run(self, x0, ves, layer_fn):
        cfg = self.cfg
        n = len(self.blocks)
        x, skips, xb = x0, [], None
        n_skip = len(self.skip_gate)
        for i, b in enumerate(self.blocks):
            if n_skip and i >= n - n_skip:
                x = x + self._skip(n - 1 - i, x0) * skips[n - 1 - i]
            if cfg.x0:
                x = b.lam * x + b.mu * x0
            x = layer_fn(i, x, ves[b.ve] if b.ve is not None else None)
            if i < n_skip:
                skips.append(x)
            if b.backout:
                xb = x
        if cfg.backout and xb is not None:
            x = x - self.backout_lam * xb
        return norm(x)

    def _trunk(self, idx):
        T = idx.shape[1]
        e, _ = self._embed(idx)
        x0 = norm(e)
        ves = [t(idx) for t in self.ve]
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        return self._run(x0, ves, lambda i, x, ve: self.blocks[i](x, ve, cos, sin, self.lib))

    def _read(self, idx, states=None, looks=False):
        st = states or {"pos": 0, "prev": None, "layers": [None] * len(self.blocks)}
        P, T = st["pos"], idx.shape[1]
        if P + T > ROPE_TABLE:
            cos, sin = rope_table(P + T, self.cfg.attn_head_dim, self.cfg.rope_wave, self.cfg.rope_frac, idx.device)
            cos, sin = cos[P:], sin[P:]
        else:
            cos, sin = self.rope_cos[P:P + T], self.rope_sin[P:P + T]
        e, raw_last = self._embed(idx, st["prev"])
        x0 = norm(e)
        ves = [t(idx) for t in self.ve]
        new, infos = list(st["layers"]), []

        def layer(i, x, ve):
            x, new[i], info = self.blocks[i].step(x, ve, cos, sin, st["layers"][i], self.lib, looks)
            infos.append({"kind": "A", "looks": info[0] if info is not None else None})
            return x
        h = self._run(x0, ves, layer)
        return self.scores(h), {"pos": P + T, "prev": raw_last, "layers": new}, infos

    def make_optimizer(self, lr, device, kind="muon", muon_lr=0.03, muon_wd=0.1):
        from .optim import make
        return make(self, lr, device, muon_lr, muon_wd, kind)

    # ---------------------------------------------------------------- Seed & Bloom
    def stage_at(self, progress: float) -> int:
        if self.cfg.hold:
            return 0
        return max(i for i, (_, _, s) in enumerate(self.plan) if progress >= s)

    def grow_to(self, progress: float) -> bool:
        """Grow to the stage this point of the run asks for (the trainer then rebuilds the optimizer and calls carry_opt)."""
        target = self.stage_at(progress)
        if target <= self.stage:
            return False
        while self.stage < target:
            self._grow_once()
        if getattr(self, "_compile_on", False):
            self.enable_compile()
        return True

    def load_state_dict(self, sd, strict=True, assign=False):
        """A checkpoint saved after a growth loads into a fresh (seed-sized) Seed: it grows to the saved stage first."""
        target = int(sd.get("bloom_stage", torch.tensor(0)))
        while self.stage < target:
            self._grow_once()
        self._carry = []
        return super().load_state_dict(sd, strict=strict, assign=assign)

    def _new_param(self, old: nn.Parameter, shape, scale: float, fill: str, bound: float = 0.0):
        """A bigger parameter: `fill` ("zero" / "uniform" +-bound) with old * scale in its leading corner."""
        new = torch.zeros(shape, device=old.device, dtype=old.dtype)
        if fill == "uniform":
            new.uniform_(-bound, bound)
        new[tuple(slice(0, s) for s in old.shape)] = old.detach() * scale
        p = nn.Parameter(new, requires_grad=old.requires_grad)
        for attr in ("no_decay",):
            if hasattr(old, attr):
                setattr(p, attr, getattr(old, attr))
        self._carry.append((old, p, 1.0 / scale))
        return p

    def _grow_linear(self, lin, rows, cols, scale, fill, bound=0.0):
        lin.weight = self._new_param(lin.weight, (rows, cols), scale, fill, bound)
        lin.out_features, lin.in_features = rows, cols

    @torch.no_grad()
    def _grow_once(self):
        cfg = self.cfg
        s0 = self.stage
        n, d, _ = self.plan[s0]
        n2, d2, _ = self.plan[s0 + 1]
        if d2 != d:
            self._widen(d, d2)
        if n2 != n:
            self._deepen(s0, d2)
        self.bloom_stage.fill_(s0 + 1)
        self._stage = s0 + 1
        assert [b.label for b in self.blocks] == self.orders[s0 + 1]

    @torch.no_grad()
    def _widen(self, d, d2):
        """[x, 0]: readers / s, writers * s, the word table / s (s = sqrt(d2 / d)); new parts read randomly, write zero."""
        cfg = self.cfg
        s = math.sqrt(d2 / d)
        hd = cfg.attn_head_dim
        H2, KV2, hid2 = cfg.dims(d2)
        b2 = 3 ** 0.5 * d2 ** -0.5
        self.embed.weight = self._new_param(self.embed.weight, (cfg.vocab_size, d2), 1 / s, "zero")
        self.embed.embedding_dim = d2
        for t in self.ve:
            t.weight = self._new_param(t.weight, (cfg.vocab_size, KV2 * hd), 1.0, "zero")
            t.embedding_dim = KV2 * hd
        for b in self.blocks:
            a = b.attn
            self._grow_linear(a.q, H2 * hd, d2, 1 / s, "uniform", b2)
            self._grow_linear(a.k, KV2 * hd, d2, 1 / s, "uniform", b2)
            self._grow_linear(a.v, KV2 * hd, d2, 1 / s, "uniform", b2)
            self._grow_linear(a.out, d2, H2 * hd, s, "zero")
            if a.gate is not None:
                self._grow_linear(a.gate, H2, GATE_IN, 1 / s, "zero")
            if a.ve_gate is not None:
                self._grow_linear(a.ve_gate, KV2, GATE_IN, 1 / s, "zero")
            a.H, a.KV = H2, KV2
            m = b.mlp
            up_rows = hid2 * (2 if m.kind == "swiglu" else 1)
            if m.kind == "swiglu":         # keep the two halves apart: [a_old, a_new, b_old, b_new]
                raise NotImplementedError("Seed grows the ReLU² MLP only")
            self._grow_linear(m.up, up_rows, d2, 1 / s, "uniform", 0.5 * b2)
            self._grow_linear(m.down, d2, hid2, s, "zero")
            if b.look is not None:
                lk = b.look
                self._grow_linear(lk.query, lk.query.out_features, d2, 1 / s, "uniform", b2)
                self._grow_linear(lk.gate, lk.gate.out_features, d2, 1 / s, "uniform", b2)
                self._grow_linear(lk.out, d2, lk.out.in_features, s, "zero")
        for g in self.skip_gate:
            self._grow_linear(g, 1, GATE_IN, 1 / s, "zero")
        if self.smear_gate is not None:      # it reads the raw word vectors, which shrank by s
            self._grow_linear(self.smear_gate, 1, GATE_IN, s, "zero")

    @torch.no_grad()
    def _deepen(self, s0, d2):
        """Slot the new layers in (insert_plan): copies of their neighbours' reading weights, outputs at zero."""
        cfg = self.cfg
        old = {b.label: b for b in self.blocks}
        order = self.orders[s0 + 1]
        fp8 = any(type(m).__name__ == "Float8Linear" for m in self.modules())
        dev = self.embed.weight.device
        blocks = []
        for lab in order:
            if lab in old:
                blocks.append(old[lab])
                continue
            src = old[self.sources[lab]]
            b = Block(cfg, d2, lab, self.roles[lab]).to(device=dev, dtype=self.embed.weight.dtype)
            self._init_block(b, d2)
            for name in ("q", "k", "v"):
                getattr(b.attn, name).weight.copy_(getattr(src.attn, name).weight)
            b.attn.qn.copy_(src.attn.qn)
            b.attn.kn.copy_(src.attn.kn)
            if b.attn.gate is not None and src.attn.gate is not None:
                b.attn.gate.weight.copy_(src.attn.gate.weight)
            if b.attn.ve_gate is not None and src.attn.ve_gate is not None:
                b.attn.ve_gate.weight.copy_(src.attn.ve_gate.weight)
            b.mlp.up.weight.copy_(src.mlp.up.weight)
            if fp8:
                from torchao.float8 import convert_to_float8_training, Float8LinearConfig
                from oazy.fp8 import keep
                convert_to_float8_training(b, config=Float8LinearConfig(), module_filter_fn=lambda m, f: keep(m, "blocks." + f))
            blocks.append(b)
        if cfg.unet:                          # the skips: an old pair keeps its weight and gate, a pair with a new layer starts at 0
            n_old, n2 = len(self.orders[s0]), len(order)
            pos_old = {lab: p for p, lab in enumerate(self.orders[s0])}
            lam = torch.zeros(n2 // 2, device=dev, dtype=self.skip_lam.dtype)
            idx_new, idx_old = [], []
            gates = nn.ModuleList()
            for j in range(n2 // 2):
                a = order[j]
                g = nn.Linear(GATE_IN, 1, bias=False).to(device=dev, dtype=self.skip_lam.dtype)
                nn.init.zeros_(g.weight)
                if a in pos_old and order[n2 - 1 - j] in pos_old:
                    jo = pos_old[a]
                    lam[j] = self.skip_lam[jo]
                    g = self.skip_gate[jo]
                    idx_new.append(j)
                    idx_old.append(jo)
                gates.append(g)
            p = nn.Parameter(lam)
            self._carry.append((self.skip_lam, p, ("index", idx_old, idx_new)))
            self.skip_lam = p
            self.skip_gate = gates
            assert n_old // 2 == len(idx_old)
        self.blocks = nn.ModuleList(blocks)

    def carry_opt(self, old, new):
        """After a growth: move each replaced parameter's optimizer memory (and Grokfast's average) into its bigger
        successor, scaled as its gradient now is; drop the memory of parameters that are gone."""
        from .optim import carry
        carry(self._carry, old, new)
        self._carry = []

    # ---------------------------------------------------------------- the library probe
    @torch.no_grad()
    def library_report(self, windows, device, batch=16):
        """How much of the library is read (on held-out text), and the held-out loss with the library closed."""
        was = self.training
        self.eval()
        lib = self.lib
        lib.track = torch.zeros(lib.rows(), dtype=torch.long, device=device)
        on, off, n = 0.0, 0.0, 0
        for i in range(0, len(windows), batch):
            x = torch.as_tensor(np.asarray(windows[i:i + batch]).astype(np.int64), device=device)
            inp, tgt = x[:, :-1], x[:, 1:]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                lo = F.cross_entropy(self(inp)[0].float().reshape(-1, self.cfg.vocab_size), tgt.reshape(-1))
                track, lib.track = lib.track, None
                lib.off = True
                lf = F.cross_entropy(self(inp)[0].float().reshape(-1, self.cfg.vocab_size), tgt.reshape(-1))
                lib.off = False
                lib.track = track
            on, off, n = on + lo.item(), off + lf.item(), n + 1
        counts = lib.track.float()
        lib.track = None
        used = (counts > 0).float().mean().item()
        srt = counts.sort(descending=True).values
        cum = srt.cumsum(0)
        half = ((cum < cum[-1] / 2).sum().item() + 1) / counts.numel() if cum[-1] > 0 else 0.0
        self.train(was)
        return {"lib_loss_on": on / max(1, n), "lib_loss_off": off / max(1, n), "lib_used": used, "lib_half": half,
                "lib_rows": lib.rows()}
