"""Pegasus 1: a model that learns to predict MEANING, not only words. Still a NEXT-WORD model at the end: its
Speaker writes (and is scored on) one token at a time, like GPT-2 and every Bee, so HellaSwag, ARC, PIQA and the
held-out loss measure it exactly as they measure the others.

The idea (thread "Reach 55% HellaSwag with a $100 400M model", Nov 7 2026). A normal model is graded on the exact
next word, and most of its effort goes into word-choice noise ("sliced" or "cut"?). Pegasus splits the work:

  Reader (encoder)   small, reads every token (local attention). Decides where one THOUGHT ends and the next
                     begins: a new thought starts where its reading suddenly changes (a learned "surprise" between
                     neighbouring tokens; H-Net, Hwang et al. 2025). A soft spring pulls the average thought towards
                     ~16 tokens (about 12 words), never a wall.
  Meaning pieces     each finished thought is written as a few discrete pieces (8), each one entry of ONE shared
                     dictionary of 1,000 meanings that the model invents itself. The dictionary is a fixed grid
                     (FSQ, Mentzer et al. 2023): no stored table, so no entry can go unused.
  Thinker            big, slow: one step per thought. It reads only the pieces of the thoughts so far (attention
                     over thoughts = its long memory) and predicts the NEXT thought's pieces, one piece after
                     another, each with real odds (next-token prediction over meanings instead of word pieces).
  Speaker            small, fast, next-token: writes the words of the current thought. It sees the Thinker's plan
                     for this thought, the last ~256 real tokens (local attention), and a NOTEBOOK: one page per
                     finished thought (the Reader's full summary of it), so it can copy names and numbers from far back.

Causality, exactly: the token at position t is written from the plan made after the last thought that ended before
t, the pages of thoughts that ended before t, and the tokens up to t. Nothing about later tokens can reach it
(pegasus/tests.py checks this by changing future tokens).

Training signal: the Speaker's next-word loss (with the mentor's lesson on a share of the passes, as Bee4) trains
everything, gradients flowing through the pieces into the Reader (straight-through rounding). The Thinker also gets
the MEANING loss: predicting the next thought's pieces. The thought boundaries learn from the spring and from the
Speaker's loss (each token's plan is scaled by the boundary's confidence, which is 1 in value but carries the
gradient: H-Net's trick).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from bee.model import Attention, ctx_at, ctx_plan, rope_table
from bee4.model import BEE4_CTX
from memtrix.model import RMSNorm, MLP
from oazy.sample import sample_logits


@dataclass
class PegasusConfig:
    vocab_size: int = 49152          # SmolLM2's pieces (so SmolLM2-360M can be the mentor)
    tokenizer: str = "smollm2"
    max_len: int = 2048
    ctx: str = BEE4_CTX              # Bee4's short windows first: 512 until 15%, 1,024 until 30%, then 2,048
    size: str = "full"               # "full" = Pegasus 1 (~400M), "mini" = Pegasus 1 Mini (~100M); sets the widths below
    # the token side (Reader and Speaker share the word table, tied with the output)
    d_tok: int = 768
    tok_heads: int = 12
    tok_kv_heads: int = 4
    tok_mlp: int = 2048
    enc_layers: int = 4
    spk_layers: int = 6
    window: int = 256                # local attention: each token sees the last `window` tokens (Reader and Speaker)
    note_every: int = 2              # a notebook look in every 2nd Speaker layer (0 = no notebook)
    # the Thinker (one step per thought)
    d_think: int = 1280
    think_layers: int = 15
    think_heads: int = 20
    think_kv_heads: int = 4
    think_mlp: int = 3456
    # thoughts and meaning pieces
    thought_len: int = 16            # the spring's resting point: average tokens per thought (~12 words)
    max_thoughts: int = 8            # room for thoughts per window = window / this (2,048 -> 256 thoughts)
    pieces: int = 8                  # meaning pieces per thought
    levels: str = "8,5,5,5"          # one piece = 4 numbers on a grid of 8 x 5 x 5 x 5 = 1,000 meanings (FSQ)
    code_dim: int = 128              # each meaning's vector, for scoring the Thinker's guesses
    meaning_weight: float = 0.3      # the meaning loss's share next to the next-word loss
    ratio_weight: float = 0.03       # the spring on thought length (H-Net's 0.03)
    # common
    head_dim: int = 64
    rope_base: float = 50000.0
    softcap: float = 30.0
    attn_backend: str = "auto"
    compile_mode: str = "default"
    doc_mask: int = 1                # (only for the mentor: it reads each document on its own, as with Bee4)

    def __post_init__(self):
        if self.size == "mini":      # Pegasus 1 Mini, ~100M: the same design, narrower and shallower
            full = PegasusConfig.__dataclass_fields__
            mini = dict(d_tok=512, tok_heads=8, tok_kv_heads=2, tok_mlp=1408, enc_layers=3, spk_layers=4,
                        d_think=768, think_layers=8, think_heads=12, think_kv_heads=4, think_mlp=2048)
            for k, v in mini.items():
                if getattr(self, k) == full[k].default:      # anything set by hand (--set) wins
                    setattr(self, k, v)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def level_list(self):
        return [int(v) for v in str(self.levels).split(",")]

    def n_codes(self):
        return math.prod(self.level_list())


def _sub(cfg, d, heads, kv, mlp):
    """The settings one stack's Attention / MLP read (bee/model.py, memtrix/model.py)."""
    return SimpleNamespace(d_model=d, attn_heads=heads, attn_kv_heads=kv, attn_head_dim=cfg.head_dim, rope=1,
                           rope_base=cfg.rope_base, attn_backend=cfg.attn_backend, mlp_hidden=mlp)


# ---------------------------------------------------------------------------------------------- attention pieces
def local_attention(attn: Attention, x, window: int):
    """Causal attention where each token sees itself and the window-1 tokens before it. Done in blocks of `window`:
    each block's queries look at their own block and the one before, through a fixed yes/no grid."""
    B, T, _ = x.shape
    W = window
    q, k, v = attn.project(x, 0)                                     # RoPE on true positions
    nb = (T + W - 1) // W
    pad = nb * W - T
    if pad:
        q, k, v = (F.pad(t, (0, 0, 0, 0, 0, pad)) for t in (q, k, v))
    H, KV, hd = attn.H, attn.KV, attn.hd
    q = q.view(B, nb, W, H, hd)
    k = k.view(B, nb, W, KV, hd)
    v = v.view(B, nb, W, KV, hd)
    kp = torch.cat([torch.zeros_like(k[:, :1]), k[:, :-1]], 1)       # the block before (none before the first)
    vp = torch.cat([torch.zeros_like(v[:, :1]), v[:, :-1]], 1)
    kk, vv = torch.cat([kp, k], 2), torch.cat([vp, v], 2)            # (B, nb, 2W, KV, hd)
    i = torch.arange(W, device=x.device)[:, None] + W                # query positions in the 2W frame
    j = torch.arange(2 * W, device=x.device)[None, :]
    grid = (j <= i) & (j > i - W)                                    # (W, 2W): itself and the W-1 before
    first = grid.clone()
    first[:, :W] = False                                             # the first block has no block before it
    mask = torch.stack([first] + [grid] * (nb - 1))[None, :, None]   # (1, nb, 1, W, 2W)
    mask = mask.expand(B, nb, 1, W, 2 * W).reshape(B * nb, 1, W, 2 * W)
    rep = H // KV
    qh = q.reshape(B * nb, W, H, hd).transpose(1, 2)
    kh = kk.reshape(B * nb, 2 * W, KV, hd).transpose(1, 2).repeat_interleave(rep, 1)
    vh = vv.reshape(B * nb, 2 * W, KV, hd).transpose(1, 2).repeat_interleave(rep, 1)
    y = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask)  # (B*nb, H, W, hd)
    y = y.transpose(1, 2).reshape(B, nb * W, H, hd)[:, :T].transpose(1, 2)
    return attn.finish(x, y)


class Notebook(nn.Module):
    """The Speaker glancing at the notebook: one page per finished thought (the Reader's summary of it), plus a
    blank page so there is always something to look at. mask: (B, 1, T, M+1) which pages each token may read."""

    def __init__(self, d, d_page, heads, kv, hd):
        super().__init__()
        self.H, self.KV, self.hd = heads, kv, hd
        self.q = nn.Linear(d, heads * hd, bias=False)
        self.k = nn.Linear(d_page, kv * hd, bias=False)
        self.v = nn.Linear(d_page, kv * hd, bias=False)
        self.qn, self.kn = RMSNorm(hd), RMSNorm(hd)
        self.out = nn.Linear(heads * hd, d, bias=False)

    def forward(self, x, pages, mask):
        B, T, _ = x.shape
        M = pages.shape[1]
        q = self.qn(self.q(x).view(B, T, self.H, self.hd)).transpose(1, 2)
        k = self.kn(self.k(pages).view(B, M, self.KV, self.hd)).transpose(1, 2)
        v = self.v(pages).view(B, M, self.KV, self.hd).transpose(1, 2)
        rep = self.H // self.KV
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(y.transpose(1, 2).reshape(B, T, -1))


class TokBlock(nn.Module):
    """A Reader or Speaker layer: local attention, (Speaker: a notebook look), SwiGLU."""

    def __init__(self, sub, window, note=None):
        super().__init__()
        self.window = window
        self.n1 = RMSNorm(sub.d_model)
        self.attn = Attention(sub)
        self.note = note
        if note is not None:
            self.nn_ = RMSNorm(sub.d_model)
        self.n2 = RMSNorm(sub.d_model)
        self.mlp = MLP(sub)

    def forward(self, x, pages=None, pmask=None):
        x = x + local_attention(self.attn, self.n1(x), self.window)
        if self.note is not None and pages is not None:
            x = x + self.note(self.nn_(x), pages, pmask)
        return x + self.mlp(self.n2(x))


class ThinkBlock(nn.Module):
    """A Thinker layer: causal attention over thoughts (RoPE on the thought's number), SwiGLU."""

    def __init__(self, sub):
        super().__init__()
        self.n1 = RMSNorm(sub.d_model)
        self.attn = Attention(sub)
        self.n2 = RMSNorm(sub.d_model)
        self.mlp = MLP(sub)

    def forward(self, x):
        y, _ = self.attn(self.n1(x))
        x = x + y
        return x + self.mlp(self.n2(x))


# ---------------------------------------------------------------------------------------------- meaning pieces (FSQ)
class Pieces(nn.Module):
    """Finite scalar quantization: each piece is a few numbers, each rounded to one of a few levels. The grid of all
    combinations is the dictionary (8 x 5 x 5 x 5 = 1,000 meanings); rounding passes gradients straight through."""

    def __init__(self, levels):
        super().__init__()
        L = torch.tensor(levels, dtype=torch.float32)
        self.register_buffer("L", L, persistent=False)
        basis = torch.cumprod(torch.tensor([1] + levels[:-1]), 0)
        self.register_buffer("basis", basis, persistent=False)
        self.n = int(math.prod(levels))
        idx = torch.arange(self.n)
        digits = (idx[:, None] // basis[None]) % torch.tensor(levels)[None]
        self.register_buffer("grid", (digits - torch.tensor(levels)[None] // 2).float() / (torch.tensor(levels)[None] // 2).float(),
                             persistent=False)                       # (n, D): every meaning's numbers, in [-1, 1]

    def forward(self, u):
        """u (..., D) -> (normalized values with straight-through gradient (..., D), meaning number (...))."""
        L = self.L
        half = (L - 1) / 2
        offset = torch.where(L.remainder(2) == 0, 0.5, 0.0)
        shift = torch.atanh(offset / half)
        b = torch.tanh(u.float() + shift) * half - offset
        r = b + (torch.round(b) - b).detach()
        digits = (torch.round(b).detach() + L // 2).long()
        code = (digits * self.basis).sum(-1)
        return r / (L // 2), code


# ---------------------------------------------------------------------------------------------- the model
class Pegasus(nn.Module):
    embed_no_decay = True
    fused_loss = True
    loss_chunk = 1024
    micro_at_full = True             # as Bee4: --micro N = N sequences per pass at the longest window

    def __init__(self, cfg: PegasusConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg
        D, Dt = c.d_tok, c.d_think
        tok = _sub(c, D, c.tok_heads, c.tok_kv_heads, c.tok_mlp)
        think = _sub(c, Dt, c.think_heads, c.think_kv_heads, c.think_mlp)
        self.embed = nn.Embedding(c.vocab_size, D)
        # Reader
        self.reader = nn.ModuleList(TokBlock(tok, c.window) for _ in range(c.enc_layers))
        self.rnorm = RMSNorm(D)
        self.route_q = nn.Linear(D, D, bias=False)                   # where a new thought starts (H-Net's routing)
        self.route_k = nn.Linear(D, D, bias=False)
        self.route_scale = nn.Parameter(torch.tensor(10.0))          # p = sigmoid(scale * (change - edge))
        self.route_edge = nn.Parameter(torch.tensor(0.75))           # set from the first batch (calibrate), then learned
        self.register_buffer("calibrated", torch.zeros((), dtype=torch.bool))   # saved with the weights
        # meaning pieces
        lv = c.level_list()
        self.fsq = Pieces(lv)
        self.pnorm = RMSNorm(D)
        self.to_pieces = nn.Linear(D, c.pieces * len(lv), bias=False)
        # Thinker
        self.think_in = nn.Linear(c.pieces * len(lv), Dt, bias=False)
        self.think_start = nn.Parameter(torch.zeros(Dt))             # the plan before any thought has ended
        self.thinker = nn.ModuleList(ThinkBlock(think) for _ in range(c.think_layers))
        self.tnorm = RMSNorm(Dt)
        # the Thinker's guess of the next thought's pieces, one after another
        self.guess_z = nn.Linear(Dt, 512, bias=False)
        self.guess_prev = nn.Linear(len(lv), 512, bias=False)
        self.guess_pos = nn.Parameter(torch.zeros(c.pieces, 512))
        self.guess_norm = RMSNorm(512)
        self.guess_fc1 = nn.Linear(512, 1024, bias=False)
        self.guess_fc2 = nn.Linear(1024, c.code_dim, bias=False)
        self.code_fc1 = nn.Linear(len(lv), 256, bias=False)          # every meaning's vector, made from its numbers
        self.code_fc2 = nn.Linear(256, c.code_dim, bias=False)
        # Speaker
        self.plan = nn.Linear(Dt, c.spk_layers * D, bias=False)       # the plan, into every Speaker layer
        self.speaker = nn.ModuleList(
            TokBlock(tok, c.window, Notebook(D, D, c.tok_heads, c.tok_kv_heads, c.head_dim)
                     if c.note_every and (i + 1) % c.note_every == 0 else None)
            for i in range(c.spk_layers))
        self.page_blank = nn.Parameter(torch.zeros(D))
        self.snorm = RMSNorm(D)
        self.apply(self._init)
        n_res = c.enc_layers + c.spk_layers
        for n, p in self.named_parameters():
            if n.endswith(("attn.out.weight", "mlp.down.weight", "note.out.weight")):
                L = c.think_layers if n.startswith("thinker") else n_res
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * L))
        with torch.no_grad():                                         # routing starts as "how different are neighbours"
            self.route_q.weight.copy_(torch.eye(D))
            self.route_k.weight.copy_(torch.eye(D))
            self.plan.weight.mul_(0.5)
        self._compiled = None
        self._last = {}

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    # ---------------------------------------------------------------- sizes and words for the trainer
    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        return n - self.embed.weight.numel() if non_embedding else n

    def part_params(self):
        """Parameters of each part (the word table counted once)."""
        cnt = lambda mods: sum(p.numel() for m in mods for p in m.parameters())   # noqa: E731
        think = cnt([self.thinker, self.think_in, self.tnorm]) + self.think_start.numel()
        guess = cnt([self.guess_z, self.guess_prev, self.guess_norm, self.guess_fc1, self.guess_fc2, self.code_fc1,
                     self.code_fc2]) + self.guess_pos.numel()
        reader = cnt([self.reader, self.rnorm, self.route_q, self.route_k, self.pnorm, self.to_pieces]) + 2
        speaker = cnt([self.speaker, self.plan, self.snorm]) + self.page_blank.numel()
        return {"word table": self.embed.weight.numel(), "reader": reader, "thinker": think, "guesser": guess,
                "speaker": speaker}

    def work_per_token(self):
        """Parameters at work per token, counting the Thinker by its share (one step per thought_len tokens on
        average): what a normal model of the same cost per token would have."""
        pp = self.part_params()
        return pp["word table"] + pp["reader"] + pp["speaker"] + (pp["thinker"] + pp["guesser"]) / self.cfg.thought_len

    def layout(self):
        c = self.cfg
        pp = self.part_params()
        return (f"Reader {c.enc_layers} x {c.d_tok} · Thinker {c.think_layers} x {c.d_think} ({pp['thinker'] / 1e6:.0f}M, "
                f"one step per thought) · Speaker {c.spk_layers} x {c.d_tok} · {c.pieces} pieces of {c.n_codes():,} meanings")

    def depth_text(self):
        return self.layout()

    def ctx_at(self, progress: float) -> int:
        return ctx_at(self.cfg, progress)

    def enable_compile(self):
        mode = self.cfg.compile_mode
        self._compiled = torch.compile(self._core) if mode in ("", "default") else torch.compile(self._core, mode=mode)

    # ---------------------------------------------------------------- the forward pass
    def _read(self, x):
        """The Reader: token vectors (B, T, D) and how much each token differs from the one before (B, T-1), 0 = the
        same, 1 = opposite (where a new thought may start)."""
        h = x
        for b in self.reader:
            h = b(h)
        h = self.rnorm(h)
        q = F.normalize(self.route_q(h[:, 1:]).float(), dim=-1)
        k = F.normalize(self.route_k(h[:, :-1]).float(), dim=-1)
        return h, (1 - (q * k).sum(-1)) / 2

    @torch.no_grad()
    def calibrate(self, idx):
        """Once, on the first batch of a fresh model: put the boundary's edge where 1 token in thought_len starts a
        thought, so the Thinker has thoughts to work with from the first step (a fixed starting edge gave ~64-token
        thoughts on random weights, and the Thinker's first layer then had almost nothing to learn from). From then
        on the edge is learned, pulled by the spring and by the Speaker's loss."""
        if bool(self.calibrated):
            return
        _, change = self._read(self.embed(idx))
        flat = change.float().flatten()
        k = max(1, min(flat.numel(), round(flat.numel() * (1 - 1 / self.cfg.thought_len))))
        self.route_edge.copy_(flat.kthvalue(k).values)
        self.calibrated.fill_(True)

    def _core(self, idx):
        """(B, T) tokens -> (Speaker's token vectors after the final norm (B, T, D), the meaning loss, the spring's
        loss, a few numbers about the thoughts)."""
        c = self.cfg
        B, T = idx.shape
        M = max(2, T // c.max_thoughts)
        x = self.embed(idx)
        h, change = self._read(x)
        p = torch.sigmoid(self.route_scale * (change - self.route_edge))
        p = torch.cat([torch.ones_like(p[:, :1]), p], 1)             # (B, T) the first token starts a thought
        from oazy.tok import eot_id
        doc_start = torch.zeros_like(idx, dtype=torch.bool)
        doc_start[:, 1:] = idx[:, :-1] == eot_id(c.tokenizer)        # a new document always starts a new thought
        start = (p >= 0.5) | doc_start
        start[:, 0] = True
        sf = start.float()
        conf = torch.where(start, p, 1 - p)
        conf = conf + (1 - conf).detach()                            # 1 in value, carries the gradient to p
        tid = (start.long().cumsum(1) - 1).clamp(max=M - 1)          # each token's thought (late extras share the last)
        # ---- pages: each thought's summary (the mean of its tokens' Reader vectors)
        flat = (tid + torch.arange(B, device=idx.device)[:, None] * M).reshape(-1)
        pages = torch.zeros(B * M, h.shape[-1], device=h.device, dtype=h.dtype).index_add_(0, flat, h.reshape(-1, h.shape[-1]))
        count = torch.zeros(B * M, device=h.device, dtype=torch.float32).index_add_(
            0, flat, torch.ones(B * T, device=h.device, dtype=torch.float32))
        pages = (pages.float() / count.clamp_min(1)[:, None]).view(B, M, -1)
        valid = (count > 0).view(B, M)                               # thoughts that exist in this window
        pages = self.pnorm(pages.to(h.dtype))
        # ---- meaning pieces
        lv = c.level_list()
        u = self.to_pieces(pages).view(B, M, c.pieces, len(lv))
        val, code = self.fsq(u)                                      # (B, M, K, D) in [-1, 1], (B, M, K)
        val = val.to(h.dtype)
        # ---- Thinker: reads the pieces of each thought, plans the next
        z = self.think_in(val.reshape(B, M, -1))
        for b in self.thinker:
            z = b(z)
        z = self.tnorm(z)                                            # z[:, j] = the plan after thought j
        # ---- the meaning loss: from the plan after thought j, guess thought j+1's pieces one by one
        nxt_val, nxt_code = val[:, 1:].detach(), code[:, 1:]
        prev = torch.cumsum(self.guess_prev(nxt_val), 2) - self.guess_prev(nxt_val)   # pieces before this one
        a = self.guess_z(z[:, :-1])[:, :, None] + self.guess_pos[None, None] + prev
        r = self.guess_fc2(F.silu(self.guess_fc1(self.guess_norm(a))))              # (B, M-1, K, code_dim)
        table = self.code_fc2(F.silu(self.code_fc1(self.fsq.grid.to(r.dtype))))   # (n, code_dim) every meaning
        logits = (r @ table.t()).float()                             # (B, M-1, K, n)
        both = (valid[:, 1:] & valid[:, :-1]).float()[:, :, None].expand(-1, -1, c.pieces)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), nxt_code.reshape(-1), reduction="none")
        meaning = (ce * both.reshape(-1)).sum() / both.sum().clamp_min(1)
        # ---- the spring on thought length (H-Net's ratio loss): its lowest point is 1 start per thought_len tokens
        N = float(c.thought_len)
        Fr, G = sf.mean(), p.mean()
        ratio = N / (N - 1) * ((N - 1) * Fr * G + (1 - Fr) * (1 - G))
        # ---- Speaker: the plan made after the previous thought, the notebook's earlier pages, the nearby tokens
        zs = torch.cat([self.think_start.to(z.dtype).expand(B, 1, -1), z[:, :-1]], 1)   # plan for thought j
        P = self.plan(zs).view(B, M, c.spk_layers, -1)
        plan_t = torch.gather(P, 1, tid[:, :, None, None].expand(-1, -1, c.spk_layers, P.shape[-1]))
        plan_t = plan_t * conf[:, :, None, None].to(plan_t.dtype)
        pmask = None
        allp = None
        if c.note_every:
            allp = torch.cat([self.page_blank.to(pages.dtype).expand(B, 1, -1), pages], 1)   # (B, M+1, D)
            pj = torch.arange(M + 1, device=idx.device)
            # page j (thought j-1) is readable once thought j-1 has ended, i.e. by the tokens of later thoughts
            pmask = ((pj[None, None, :] == 0) | ((pj[None, None, :] - 1) < tid[:, :, None]))[:, None]
        s = x
        for i, b in enumerate(self.speaker):
            s = s + plan_t[:, :, i]
            s = b(s, allp, pmask)
        s = self.snorm(s)
        n_th = valid.float().sum(1).mean()
        stats = {"thought_len": T / n_th.clamp_min(1), "meaning": meaning.detach(), "start_p": G.detach()}
        return s, meaning, ratio, stats

    def head(self, h):
        logits = F.linear(h, self.embed.weight)
        sc = self.cfg.softcap
        return sc * torch.tanh(logits / sc) if sc else logits

    def _body(self, idx):
        """The full scores (B, T, V) (HellaSwag, the probes, chat)."""
        return self.head(self._core(idx)[0])

    def doc_mask(self, idx):
        """For the mentor only: which earlier words each word may see (its own document), as Bee4 gives it."""
        from oazy.tok import eot_id
        B, T = idx.shape
        resets = torch.zeros(B, T, dtype=torch.bool, device=idx.device)
        resets[:, 1:] = idx[:, :-1] == eot_id(self.cfg.tokenizer)
        doc = resets.cumsum(1)
        return (doc[:, None, :, None] == doc[:, None, None, :]) & torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()

    def losses(self, idx, targets, mentor=None, kd_weight=0.0):
        """One lesson: the Speaker's next-word loss (and the mentor's lesson), plus the meaning loss and the spring.
        Returns (loss to train on, {"ce": next-word loss, "kd": mentor term}), the stats detached."""
        if self.training and not bool(self.calibrated):
            self.calibrate(idx)
        s, meaning, ratio, st = (self._compiled or self._core)(idx)
        th = tW = None
        if mentor is not None and kd_weight > 0:
            th, tW = mentor.hidden(idx, self.doc_mask(idx) if self.cfg.doc_mask else None)
        if self.fused_loss:
            from oazy.lmloss import head_loss
            loss, ce, kd = head_loss(s, self.embed.weight, targets, self.cfg.softcap, th, tW, kd_weight, self.loss_chunk)
        else:
            from oazy.lmloss import plain_loss
            loss, ce, kd = plain_loss(s, self.embed.weight, targets, self.cfg.softcap, th, tW, kd_weight)
        total = loss + self.cfg.meaning_weight * meaning + self.cfg.ratio_weight * ratio
        self._last = {k: float(v) for k, v in st.items()}
        stats = {"ce": ce}
        if th is not None:
            stats["kd"] = kd
        return total, stats

    def thought_report(self):
        """For the training log: tokens per thought, the meaning loss (nats per piece), the boundary share."""
        r = self._last
        return {"thought_len": r.get("thought_len"), "meaning_loss": r.get("meaning")} if r else {}

    def forward(self, idx, targets=None):
        """Without targets: (scores, None, {}). With targets: (None, the next-word loss only, stats): the held-out
        loss measures words, exactly as for every other model."""
        if targets is None:
            return self._body(idx), None, {}
        was = self._last
        _, stats = self.losses(idx, targets)
        self._last = was
        return None, stats["ce"], stats

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=100, temperature=0.8, top_k=40, eos=None, rep=1.0):
        """Write token by token (re-reading the whole text each time: simple, fine for samples)."""
        dev = self.embed.weight.device
        ids = list(prompt_ids)
        out = []
        for _ in range(max_new_tokens):
            x = torch.tensor([ids[-self.cfg.max_len:]], device=dev)
            logits = self._body(x)
            nxt = sample_logits(logits[0, -1], temperature, top_k, recent=ids[-64:], rep=rep)
            if eos is not None and nxt == eos:
                break
            out.append(nxt)
            ids.append(nxt)
        return out, None


__all__ = ["Pegasus", "PegasusConfig", "ctx_plan", "rope_table"]
