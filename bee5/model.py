"""Bee5: the best plain 200M transformer we know how to build, raised like a speedrun entry. Still a NEXT-WORD model:
it reads text and guesses the very next token, like GPT-2 and every Bee, and it is scored the same way.

No new ideas (João, Nov 9: "modern transformer, no different ideas ... all the tricks"). Bee4's body, rebuilt with
the pieces the public NanoGPT speedrun (github.com/KellerJordan/modded-nanogpt) has proven, record by record, and
that carry over to a 200M model trained on 10,000M tokens. bee5/README.md lists every trick, kept or dropped, and why.

                       Bee4                              Bee5
  width, layers        1,024 x 12                        1,024 x 12
  attention            16 query / 4 key-value heads      the same heads; QK-norm, RoPE on half of each head
                       of 64, RoPE (base 50,000),        (the other half carries no position), a small per-head
                       a full 1,024 x 1,024 output gate  gate read from 16 numbers, keys shifted by one word on
                                                         every 4th layer (one-layer copying)
  thinking part        SwiGLU 2,816                      ReLU squared 3,840 (the same weight count as SwiGLU 2,560)
  word table           tied, scores capped at +-30       tied, scores squashed into (0, 23) asymmetrically
  extra paths          none                              value tables (layers 1-2 and 11-12), U-net skips (layer i
                                                         to layer 13-i), the first layer's input mixed back into
                                                         every layer, a one-word look-back on the input, a
                                                         "back-out" of the 8th layer before the head
  at the start         small random outputs              every output projection at zero (each layer starts as a no-op)
  optimizer            Muon (Newton-Schulz)              NorMuon: Polar Express orthogonalisation, a per-neuron
                                                         step size, cautious weight decay, momentum warm-up
  early lesson         next word only                    the next 3 words, fading to the next word only by 1/6
  parameters           198.2M                            ~201M (all of it; 25M are the two value tables)

Kept from Bee4 (bee4/README.md): SmolLM2's pieces, windows 512 -> 1,024 -> 2,048, the batch ramp, the WSD schedule with
the square-root cool-down over the last 30%, bf16, the loss in slices, the SmolLM2-360M mentor on a share of passes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from oazy.sample import sample_logits

BEE5_CTX = "512:0,1024:0.15,2048:0.3"     # Bee4's window warmup
GATE_IN = 16                               # the small gates read this many leading numbers of their input


@dataclass
class Bee5Config:
    vocab_size: int = 49152      # SmolLM2's pieces (the mentor's too)
    tokenizer: str = "smollm2"
    max_len: int = 2048
    ctx: str = BEE5_CTX
    d_model: int = 1024
    n_layer: int = 12
    attn_heads: int = 16
    attn_kv_heads: int = 4
    attn_head_dim: int = 64
    mlp: str = "relu2"           # "relu2" (speedrun) or "swiglu" (Bee4, Llama); the same weight count either way
    mlp_hidden: int = 0          # 0 = the default for the kind: relu2 3,840, swiglu 2,560 (both 7.9M weights a layer)
    rope_wave: float = 1024.0    # the slowest rotating pair turns once per 2*pi*1,024 words (speedrun)
    rope_frac: float = 0.5       # share of each head's pairs that rotate; the rest carry no position (half-truncated RoPE)
    n_ve: int = 2                # value tables: table j feeds layer j and layer n_layer - n_ve + j (U-net pattern)
    unet: int = 1                # skips from layer i's output to layer (n_layer - 1 - i)'s input, gated
    x0: int = 1                  # each layer's input = a * (stream) + b * (the normed word vectors), a, b learned
    smear: int = 1               # each word's vector gets a gated share of the previous word's
    backout: int = 1             # the head sees the stream minus a learned share of the stream at 2/3 depth
    attn_gate: int = 1           # per-head output gate from the first 16 numbers (sparse attention gate)
    key_offset: str = "3,7,11"   # layers whose keys carry the previous word in their position-free half ("" = none)
    cap: str = "sig"             # scores: "sig" = 23 * sigmoid((s + 5) / 7.5) (speedrun), "tanh" = softcap * tanh(s / softcap), "none"
    softcap: float = 15.0        # used by cap="tanh"
    mtp_until: float = 1 / 6     # extra lessons on the 2nd and 3rd next words early on, gone by this share of the run (0 = off)
    doc_mask: int = 1            # the mentor reads each document alone; Bee5's own attention is plain causal (as Bee4)
    attn_backend: str = "auto"   # "cudnn" = cuDNN's attention kernel first (Blackwell); "auto" = PyTorch's choice
    compile_mode: str = "default"

    def __post_init__(self):
        if not self.mlp_hidden:
            self.mlp_hidden = 3840 if self.mlp == "relu2" else 2560 * self.d_model // 1024

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def kinds(self):
        return ["A"] * self.n_layer

    def offset_layers(self):
        return {int(s) for s in str(self.key_offset).split(",") if s.strip() != ""}

    def ve_index(self):
        """Which value table each layer reads (None = none): 0, 1, -, ..., -, 0, 1."""
        out = [None] * self.n_layer
        for j in range(self.n_ve):
            out[j] = j
            out[self.n_layer - self.n_ve + j] = j
        return out


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def rope_table(n, hd, wave, frac, device=None):
    """cos, sin (n, hd/2): the first frac of the pairs rotate (fastest first, slowest once per 2*pi*wave words), the
    rest stand still (cos 1, sin 0): those dims carry content only (half-truncated RoPE, speedrun record 17)."""
    half = hd // 2
    n_rot = max(1, int(half * frac))
    freq = (1.0 / wave) ** torch.linspace(0, 1, n_rot, dtype=torch.float32, device=device)
    freq = torch.cat([freq, freq.new_zeros(half - n_rot)])
    ang = torch.arange(n, dtype=torch.float32, device=device)[:, None] * freq[None]
    return ang.cos(), ang.sin()


def rotary(x, cos, sin):
    """x (B, T, H, hd); cos, sin (T, hd/2)."""
    cos, sin = cos[None, :, None, :].to(x.dtype), sin[None, :, None, :].to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos + x2 * sin, x2 * cos - x1 * sin], dim=-1)


ROPE_TABLE = 8192


class Attention(nn.Module):
    def __init__(self, cfg: Bee5Config, layer: int):
        super().__init__()
        D, H, KV, hd = cfg.d_model, cfg.attn_heads, cfg.attn_kv_heads, cfg.attn_head_dim
        assert H % KV == 0 and hd % 4 == 0
        self.H, self.KV, self.hd, self.cfg = H, KV, hd, cfg
        self.q = nn.Linear(D, H * hd, bias=False)
        self.k = nn.Linear(D, KV * hd, bias=False)
        self.v = nn.Linear(D, KV * hd, bias=False)
        self.out = nn.Linear(H * hd, D, bias=False)
        self.qn = nn.Parameter(torch.ones(hd))          # QK-norm gains: the model learns each head's sharpness
        self.kn = nn.Parameter(torch.ones(hd))
        self.gate = nn.Linear(GATE_IN, H, bias=False) if cfg.attn_gate else None
        self.ve_gate = nn.Linear(GATE_IN, KV, bias=False) if cfg.ve_index()[layer] is not None else None
        self.offset = layer in cfg.offset_layers()
        if self.offset:                                  # the position-free dims of each key head
            n_rot = max(1, int(hd // 2 * cfg.rope_frac))
            still = torch.zeros(hd, dtype=torch.bool)
            still[n_rot:hd // 2] = True
            still[hd // 2 + n_rot:] = True
            self.register_buffer("still", still, persistent=False)

    def project(self, x, ve, cos, sin, prev_k=None):
        """q, k, v (B, T, heads, hd); k_raw: the keys before the shift (the next read's prev_k)."""
        B, T, _ = x.shape
        q = F.rms_norm(self.q(x).view(B, T, self.H, self.hd), (self.hd,), self.qn)
        k = F.rms_norm(self.k(x).view(B, T, self.KV, self.hd), (self.hd,), self.kn)
        v = self.v(x).view(B, T, self.KV, self.hd)
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)
        k_raw = k
        if self.offset:      # partial key offset (record 49): the still dims of key t hold key t-1's (word t-1's content)
            first = k[:, :1] if prev_k is None else prev_k
            shifted = torch.cat([first, k[:, :-1]], dim=1)
            k = torch.where(self.still, shifted, k)
        if ve is not None:   # value table (records 14-17, gates record 55): a word's own vector added to its value
            g = 2 * torch.sigmoid(self.ve_gate(x[..., :GATE_IN])).view(B, T, self.KV, 1)
            v = v + g * ve.view(B, T, self.KV, self.hd)
        return q, k, v, k_raw

    def finish(self, x, y):
        """y (B, T, H, hd) -> the layer's output (B, T, D)."""
        B, T = x.shape[:2]
        if self.gate is not None:    # sparse attention gate (record 28): a head may say "nothing to add here"
            y = y * torch.sigmoid(self.gate(x[..., :GATE_IN])).view(B, T, self.H, 1)
        return self.out(y.reshape(B, T, self.H * self.hd))

    def forward(self, x, ve, cos, sin):
        q, k, v, _ = self.project(x, ve, cos, sin)
        y = causal(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), self.cfg.attn_backend)
        return self.finish(x, y.transpose(1, 2))

    def step(self, x, ve, cos, sin, state, looks=False):
        """Reading on: state = (keys, values, last raw key) of the words read before, or None."""
        q, k, v, k_raw = self.project(x, ve, cos, sin, None if state is None else state[2])
        P = 0 if state is None else state[0].shape[1]
        if state is not None:
            k, v = torch.cat([state[0], k], 1), torch.cat([state[1], v], 1)
        T = q.shape[1]
        rep = self.H // self.KV
        qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2).repeat_interleave(rep, 1), v.transpose(1, 2).repeat_interleave(rep, 1)
        mask = torch.ones(T, P + T, dtype=torch.bool, device=x.device).tril(P)
        y = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask)
        info = None
        if looks:
            w = torch.softmax(qh[:, :, -1:].float() @ kh.float().transpose(-1, -2) / math.sqrt(self.hd), -1)
            info = w[:, :, 0].mean(1)
        return self.finish(x, y.transpose(1, 2)), (k, v, k_raw[:, -1:]), info


_GQA = None


def causal(qh, kh, vh, backend="auto"):
    """Plain causal attention with grouped key/value heads on the fused kernel (Bee4's way)."""
    global _GQA
    rep = qh.shape[1] // kh.shape[1]
    if backend == "cudnn" and qh.is_cuda:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        if rep > 1:
            kh, vh = kh.repeat_interleave(rep, 1), vh.repeat_interleave(rep, 1)
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
                         set_priority=True):
            return F.scaled_dot_product_attention(qh, kh, vh, is_causal=True)
    if rep > 1 and _GQA is not False:
        try:
            y = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, enable_gqa=True)
            _GQA = True
            return y
        except (TypeError, RuntimeError):
            _GQA = False
    if rep > 1:
        kh, vh = kh.repeat_interleave(rep, 1), vh.repeat_interleave(rep, 1)
    return F.scaled_dot_product_attention(qh, kh, vh, is_causal=True)


class MLP(nn.Module):
    def __init__(self, cfg: Bee5Config):
        super().__init__()
        D, Hm = cfg.d_model, cfg.mlp_hidden
        self.kind = cfg.mlp
        self.up = nn.Linear(D, Hm * (2 if cfg.mlp == "swiglu" else 1), bias=False)
        self.down = nn.Linear(Hm, D, bias=False)

    def forward(self, x):
        h = self.up(x)
        if self.kind == "swiglu":
            a, b = h.chunk(2, dim=-1)
            return self.down(F.silu(a) * b)
        return self.down(F.relu(h).square())     # ReLU squared (record 5; Primer, So et al. 2021)


class Block(nn.Module):
    def __init__(self, cfg: Bee5Config, layer: int):
        super().__init__()
        self.kind = "A"
        self.attn = Attention(cfg, layer)
        self.mlp = MLP(cfg)

    def forward(self, x, ve, cos, sin):
        x = x + self.attn(norm(x), ve, cos, sin)
        return x + self.mlp(norm(x))

    def step(self, x, ve, cos, sin, state, looks=False):
        y, state, info = self.attn.step(norm(x), ve, cos, sin, state, looks)
        x = x + y
        return x + self.mlp(norm(x)), state, info


def mtp_weights(progress: float, until: float):
    """How much the 2nd and 3rd next words count (speedrun record 53): [1, 0.5, 0.25 -> 0] over the first half of
    `until`, then [1, 0.5 -> 0], then the next word alone."""
    if until <= 0 or progress >= until:
        return (1.0,)
    h = until / 2
    if progress < h:
        return (1.0, 0.5, 0.25 * (1 - progress / h))
    return (1.0, 0.5 * (1 - (progress - h) / h))


class Bee5(nn.Module):
    micro_at_full = True        # --micro N = N sequences per pass at the longest window (Bee4's rule)
    fused_loss = True           # the loss in slices (bee5/loss.py); False = full 32-bit scores
    loss_chunk = 1024
    Block = Block               # the layer class (Bee6 brings its own)

    def __init__(self, cfg: Bee5Config):
        super().__init__()
        self.cfg = cfg
        D, n = cfg.d_model, cfg.n_layer
        kvd = cfg.attn_kv_heads * cfg.attn_head_dim
        self.embed = nn.Embedding(cfg.vocab_size, D)                     # tied: also the score head
        self.ve = nn.ModuleList(nn.Embedding(cfg.vocab_size, kvd) for _ in range(cfg.n_ve))
        self.blocks = nn.ModuleList(type(self).Block(cfg, i) for i in range(n))
        self.lam = nn.Parameter(torch.ones(n))                           # the stream's weight at each layer's input
        self.mu = nn.Parameter(torch.zeros(n))                           # the first input's weight there (x0)
        n_skip = n // 2 if cfg.unet else 0
        self.skip_lam = nn.Parameter(torch.full((n_skip,), -1.5))       # sigmoid(-1.5) * 2 * 0.5 = 0.18 at the start
        self.skip_gate = nn.ModuleList(nn.Linear(GATE_IN, 1, bias=False) for _ in range(n_skip))
        self.smear_lam = nn.Parameter(torch.zeros(1))
        self.smear_gate = nn.Linear(GATE_IN, 1, bias=False) if cfg.smear else None
        self.backout_lam = nn.Parameter(torch.full((1,), 0.5 if cfg.backout else 0.0), requires_grad=bool(cfg.backout))
        self.backout_at = max(0, round(2 * n / 3) - 1)                   # the 8th of 12 layers
        if getattr(cfg, "backout_layer", -1) >= 0:                       # Bee6's bloom keeps it on the layer it grew from
            self.backout_at = cfg.backout_layer
        cos, sin = rope_table(ROPE_TABLE, cfg.attn_head_dim, cfg.rope_wave, cfg.rope_frac)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.reset_parameters()
        self._compiled = None
        self._progress = 1.0
        self.token_select = None

    @torch.no_grad()
    def reset_parameters(self):
        D = self.cfg.d_model
        bound = 3 ** 0.5 * D ** -0.5                        # uniform with std 1/sqrt(width) (speedrun)
        nn.init.normal_(self.embed.weight, std=0.005)       # the score head's scale (records 8, 51)
        for t in self.ve:
            nn.init.zeros_(t.weight)                        # value tables start silent
        for b in self.blocks:
            for lin in (b.attn.q, b.attn.k, b.attn.v):
                nn.init.uniform_(lin.weight, -bound, bound)
            nn.init.uniform_(b.mlp.up.weight, -0.5 * bound, 0.5 * bound)
            nn.init.zeros_(b.attn.out.weight)               # every output projection starts at zero (record 5): each
            nn.init.zeros_(b.mlp.down.weight)               # layer starts as a no-op and the stream is the word itself
            for g in (b.attn.gate, b.attn.ve_gate):
                if g is not None:
                    nn.init.zeros_(g.weight)
        for g in list(self.skip_gate) + ([self.smear_gate] if self.smear_gate is not None else []):
            nn.init.zeros_(g.weight)

    # ---------------------------------------------------------------- sizes, names
    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        return n - self.embed.weight.numel() if non_embedding else n

    def table_params(self):
        return self.embed.weight.numel() + sum(t.weight.numel() for t in self.ve)

    def layout(self):
        c = self.cfg
        return (f"{c.n_layer} layers of {c.d_model}, all attention ({c.attn_heads} query / {c.attn_kv_heads} key-value heads), "
                f"{'ReLU²' if c.mlp == 'relu2' else 'SwiGLU'} {c.mlp_hidden}, {c.n_ve} value tables")

    def depth_text(self):
        return f"{self.cfg.n_layer} layers deep"

    def ctx_at(self, progress: float) -> int:
        from bee.model import ctx_at
        return ctx_at(self.cfg, progress)

    def set_progress(self, progress: float):
        self._progress = float(progress)

    def mtp(self):
        """This pass's weights for the 1st, 2nd, 3rd next word (training only)."""
        if not (self.training and torch.is_grad_enabled()):
            return (1.0,)
        return mtp_weights(self._progress, self.cfg.mtp_until)

    def enable_compile(self):
        mode = self.cfg.compile_mode
        self._compiled = torch.compile(self._trunk) if mode in ("", "default") else torch.compile(self._trunk, mode=mode)

    def make_optimizer(self, lr, device, kind="muon", muon_lr=0.03, muon_wd=0.1):
        from .optim import make
        return make(self, lr, device, muon_lr, muon_wd, kind)

    # ---------------------------------------------------------------- the body
    def _skip(self, j, x0):
        return torch.sigmoid(self.skip_lam[j]) * 2 * torch.sigmoid(self.skip_gate[j](x0[..., :GATE_IN]))

    def _embed(self, idx, prev=None):
        """The word vectors after the one-word look-back (record 34), and the raw last one (the next read's prev)."""
        e = self.embed(idx)
        raw_last = e[:, -1:]
        if self.smear_gate is not None:
            before = torch.cat([prev if prev is not None else torch.zeros_like(e[:, :1]), e[:, :-1]], dim=1)
            g = self.smear_lam * torch.sigmoid(self.smear_gate(e[..., :GATE_IN]))
            if prev is None:            # the first word of a fresh text has nothing before it
                g = torch.cat([torch.zeros_like(g[:, :1]), g[:, 1:]], dim=1)
            e = e + g * before
        return e, raw_last

    def _run(self, x0, ves, layer_fn):
        """The stream through every layer with the U-net skips, x0 mixing and the back-out. layer_fn(i, x, ve) -> x."""
        cfg, n = self.cfg, self.cfg.n_layer
        x, skips, xb = x0, [], None
        vi = cfg.ve_index()
        n_skip = len(self.skip_gate)
        for i in range(n):
            if n_skip and i >= n - n_skip:
                x = x + self._skip(n - 1 - i, x0) * skips[n - 1 - i]
            if cfg.x0:
                x = self.lam[i] * x + self.mu[i] * x0
            x = layer_fn(i, x, ves[vi[i]] if vi[i] is not None else None)
            if i < n_skip:
                skips.append(x)
            if i == self.backout_at:
                xb = x
        if cfg.backout:
            x = x - self.backout_lam * xb
        return norm(x)

    def _trunk(self, idx):
        """(B, T) words -> (B, T, D) vectors after the final norm (what the score head reads)."""
        T = idx.shape[1]
        e, _ = self._embed(idx)
        x0 = norm(e)
        ves = [t(idx) for t in self.ve]
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        return self._run(x0, ves, lambda i, x, ve: self.blocks[i](x, ve, cos, sin))

    def scores(self, h):
        """Scores against the word table, capped (cfg.cap)."""
        z = F.linear(h, self.embed.weight.to(h.dtype))
        return cap(z, self.cfg)

    head = scores

    def _body(self, idx):
        return self.scores(self._trunk(idx))

    def doc_mask(self, idx):
        """The mentor's yes/no grid: earlier words of the same document (B, 1, T, T)."""
        from oazy.tok import eot_id
        B, T = idx.shape
        resets = torch.zeros(B, T, dtype=torch.bool, device=idx.device)
        resets[:, 1:] = idx[:, :-1] == eot_id(self.cfg.tokenizer)
        doc = resets.cumsum(1)
        return (doc[:, None, :, None] == doc[:, None, None, :]) & torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()

    def losses(self, idx, targets, mentor=None, kd_weight=0.0):
        """One lesson: the next-word loss (+ the 2nd/3rd next words early on), plus the mentor's KL weighted kd_weight."""
        if self.token_select is not None:
            raise SystemExit("Bee5 does not take --select yet (it joins the recipe once the A/B on Plain-100M is in)")
        h = (self._compiled or self._trunk)(idx)
        th = tW = None
        if mentor is not None and kd_weight > 0:
            th, tW = mentor.hidden(idx, self.doc_mask(idx))
        w = self.mtp()
        tg = targets[..., None]
        if len(w) > 1:                    # the k-th next word of position t is targets[t + k - 1] (-1 past the window)
            pad = torch.full_like(targets[:, :len(w) - 1], -1)
            ext = torch.cat([targets, pad], 1)
            tg = torch.stack([ext[:, k:k + targets.shape[1]] for k in range(len(w))], -1)
        from .loss import head_loss, plain_loss
        f = head_loss if self.fused_loss else plain_loss
        loss, ce, kd = f(h, self.embed.weight, tg, w, self.cfg, th, tW, kd_weight, self.loss_chunk)
        stats = {"ce": ce}
        if th is not None:
            stats["kd"] = kd
        return loss, stats

    def forward(self, idx, targets=None):
        if targets is None:
            return self._body(idx), None, {}
        loss, stats = self.losses(idx, targets)
        return None, loss, stats

    # ---------------------------------------------------------------- reading as a stream (chat, samples)
    def _read(self, idx, states=None, looks=False):
        st = states or {"pos": 0, "prev": None, "layers": [None] * self.cfg.n_layer}
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
            x, new[i], info = self.blocks[i].step(x, ve, cos, sin, st["layers"][i], looks)
            infos.append({"kind": "A", "looks": info[0] if info is not None else None})
            return x
        h = self._run(x0, ves, layer)
        return self.scores(h), {"pos": P + T, "prev": raw_last, "layers": new}, infos

    @torch.no_grad()
    def read(self, idx, states=None):
        logits, states, _ = self._read(idx, states)
        return logits, states

    @torch.no_grad()
    def inspect(self, idx, states=None):
        return self._read(idx, states, looks=True)

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=100, temperature=0.8, top_k=40, eos=None, rep=1.0):
        dev = self.embed.weight.device
        logits, states = self.read(torch.tensor([list(prompt_ids)], device=dev))
        out = []
        V = self.cfg.vocab_size
        for _ in range(max_new_tokens):
            nxt = sample_logits(logits[0, -1, :V], temperature, top_k, recent=(list(prompt_ids) + out)[-64:], rep=rep)
            if eos is not None and nxt == eos:
                break
            out.append(nxt)
            logits, states = self.read(torch.tensor([[nxt]], device=dev), states)
        return out, None


def cap(z, cfg):
    """The capped scores (any dtype in, the same dtype out)."""
    if cfg.cap == "sig":                 # asymmetric (record 54): scores live in (0, 23), the middle sits at -5
        return 23.0 * torch.sigmoid((z + 5.0) / 7.5)
    if cfg.cap == "tanh":
        return cfg.softcap * torch.tanh(z / cfg.softcap)
    return z
