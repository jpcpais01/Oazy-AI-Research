"""Pegasus 2 Mini: Pegasus 1 Mini rebuilt after its first run (Nov 8 2026). Still a NEXT-WORD model at the end: the
Speaker writes, and is scored on, one token at a time.

What the first run showed (thread "Reach 55% HellaSwag...", Nov 8), and what changed:

  1. The Speaker was tiny (4 x 512) and started from bare word vectors: the word side was not a real language
     model. Now the Speaker continues from the READER's output (causal, local) and is 7 layers of 640; the Reader is
     2 layers; the Thinker is smaller (3 x 768). 105M in all (Plain-100M: 107M), ~83M at work per token (Plain: 107M).
  2. The Thinker saw the past through a pinhole: 8 rounded pieces (~80 bits) per ~16 tokens. Now it reads RICH,
     unrounded thought vectors. The pieces stay only as what the Thinker must predict next (the meaning loss), and a
     second loss keeps them informative: the thought's rich vector must be rebuildable from its pieces (so they
     cannot go empty or all alike). Code usage (how many of the 1,000 meanings are in use) is logged.
  3. Notebook pages were the average of a thought's tokens, which blurs names and numbers. Now each thought leaves a
     few pages (2), each made by a learned query attending over the thought's tokens.
  4. The 256-token window ran as a full grid. Now compiled steps use flex attention, keys/values copied out to every
     head (flex's shared-heads mode crashed Triton on the B200; PR #145); the grids remain as the fallback.
  5. Everything learned at once, so the Thinker chased a moving code from step one. Now thoughts have FIXED
     boundaries for the first 10% of training (a new thought after a sentence end, or every 32 tokens), while the
     boundary router learns to imitate them; from 10% the router's own (learned) boundaries take over. The meaning
     loss is off until 10% and grows to full weight by 20%.
  6. The same SmolLM2-360M mentor as Plain-100M (RECIPE["pegasus2"] = RECIPE["plain100"]).
  7. Probes at every check: the Speaker's loss with the plan switched off vs on (does it USE the Thinker?), with the
     notebook off, and code usage; plus the usual loss-by-position, HellaSwag, ARC/PIQA/LAMBADA.

Causality is exact (pegasus/tests2.py changes future tokens and checks nothing earlier moves).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, asdict
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from bee.model import ctx_at
from bee4.model import BEE4_CTX
from memtrix.model import RMSNorm
from pegasus.model import Pegasus, TokBlock, ThinkBlock, Notebook, Pieces, window_mask


@dataclass
class Pegasus2Config:
    vocab_size: int = 49152
    tokenizer: str = "smollm2"
    max_len: int = 2048
    ctx: str = BEE4_CTX
    # the word side: Reader then Speaker, one stream (the Speaker continues from the Reader's output)
    d_tok: int = 640
    tok_heads: int = 10
    tok_kv_heads: int = 2
    tok_mlp: int = 1728
    enc_layers: int = 2
    spk_layers: int = 7
    window: int = 256
    note_every: int = 2
    # the Thinker (one step per thought)
    d_think: int = 768
    think_layers: int = 3
    think_heads: int = 12
    think_kv_heads: int = 4
    think_mlp: int = 2048
    # thoughts
    thought_len: int = 16            # the spring's resting point once boundaries are learned
    max_thoughts: int = 8            # room for T / 8 thoughts per window
    page_vecs: int = 2               # notebook pages per thought (learned-query pooling)
    pieces: int = 8
    levels: str = "8,5,5,5"
    code_dim: int = 128
    meaning_weight: float = 0.3
    meaning_warm: str = "0.1:0.2"    # meaning loss off until 10% of training, full weight from 20%
    recon_weight: float = 0.3        # the pieces must rebuild the thought's rich vector
    ratio_weight: float = 0.03
    fixed_until: float = 0.1         # fixed boundaries until this share of training, then learned
    fixed_len: int = 32              # fixed: a new thought after a sentence end, or every this many tokens
    imitate_weight: float = 0.1      # while fixed: the router learns to predict the fixed boundaries
    # common
    head_dim: int = 64
    rope_base: float = 50000.0
    softcap: float = 30.0
    attn_backend: str = "auto"
    compile_mode: str = "default"
    flex: int = 3                    # compiled steps: flex attention for the window (1) and notebook (2); OAZY_FLEX=0 off
    doc_mask: int = 1

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
    return SimpleNamespace(d_model=d, attn_heads=heads, attn_kv_heads=kv, attn_head_dim=cfg.head_dim, rope=1,
                           rope_base=cfg.rope_base, attn_backend=cfg.attn_backend, mlp_hidden=mlp)


def sentence_end_ids(tokenizer: str):
    """Token ids that end a sentence or paragraph (".", "!", "?", line breaks, with or without a quote). Empty if the
    tokenizer cannot be had (then fixed thoughts are every fixed_len / 2 tokens)."""
    try:
        from oazy import tok
        enc = tok.get(tokenizer)
    except Exception:
        return []
    ids = set()
    for s in (".", "!", "?", ".\n", ".\n\n", "\n", "\n\n", '."', '?"', '!"', ".'", ":\n", "?\n", "!\n"):
        try:
            t = enc.encode_ordinary(s)
        except Exception:
            continue
        if len(t) == 1:
            ids.add(int(t[0]))
    return sorted(ids)


class Pegasus2(Pegasus):
    embed_no_decay = True
    fused_loss = True
    loss_chunk = 1024
    micro_at_full = True

    def __init__(self, cfg: Pegasus2Config):
        nn.Module.__init__(self)
        self.cfg = c = cfg
        D, Dt, P = c.d_tok, c.d_think, c.page_vecs
        tok = _sub(c, D, c.tok_heads, c.tok_kv_heads, c.tok_mlp)
        think = _sub(c, Dt, c.think_heads, c.think_kv_heads, c.think_mlp)
        self.embed = nn.Embedding(c.vocab_size, D)
        # Reader
        self.reader = nn.ModuleList(TokBlock(tok, c.window) for _ in range(c.enc_layers))
        self.rnorm = RMSNorm(D)
        self.route_q = nn.Linear(D, D, bias=False)
        self.route_k = nn.Linear(D, D, bias=False)
        self.route_scale = nn.Parameter(torch.tensor(10.0))
        self.route_edge = nn.Parameter(torch.tensor(0.5))
        ends = sentence_end_ids(c.tokenizer)
        self.register_buffer("ends", torch.tensor(ends or [-1], dtype=torch.long), persistent=False)
        self._every = c.fixed_len if ends else max(2, c.fixed_len // 2)
        # pages: P learned queries pool each thought's tokens
        self.pool_q = nn.Parameter(torch.randn(P, c.head_dim) * 0.02)
        self.pool_k = nn.Linear(D, c.head_dim, bias=False)
        self.pool_kn = RMSNorm(c.head_dim)
        self.pool_v = nn.Linear(D, P * D, bias=False)
        self.pnorm = RMSNorm(D)
        # rich thought vector (the Thinker's input) and the pieces (its target)
        self.think_in = nn.Linear(P * D, Dt, bias=False)
        lv = c.level_list()
        self.fsq = Pieces(lv)
        self.to_pieces = nn.Linear(P * D, c.pieces * len(lv), bias=False)
        self.recon_fc1 = nn.Linear(c.pieces * len(lv), 512, bias=False)
        self.recon_fc2 = nn.Linear(512, Dt, bias=False)
        # Thinker
        self.think_start = nn.Parameter(torch.zeros(Dt))
        self.thinker = nn.ModuleList(ThinkBlock(think) for _ in range(c.think_layers))
        self.tnorm = RMSNorm(Dt)
        self.guess_z = nn.Linear(Dt, 512, bias=False)
        self.guess_prev = nn.Linear(len(lv), 512, bias=False)
        self.guess_pos = nn.Parameter(torch.zeros(c.pieces, 512))
        self.guess_norm = RMSNorm(512)
        self.guess_fc1 = nn.Linear(512, 1024, bias=False)
        self.guess_fc2 = nn.Linear(1024, c.code_dim, bias=False)
        self.code_fc1 = nn.Linear(len(lv), 256, bias=False)
        self.code_fc2 = nn.Linear(256, c.code_dim, bias=False)
        # Speaker
        self.plan = nn.Linear(Dt, c.spk_layers * D, bias=False)
        self.speaker = nn.ModuleList(
            TokBlock(tok, c.window, Notebook(D, D, c.tok_heads, c.tok_kv_heads, c.head_dim, per=P)
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
        with torch.no_grad():
            self.route_q.weight.copy_(torch.eye(D))
            self.route_k.weight.copy_(torch.eye(D))
            self.plan.weight.mul_(0.5)
        self.register_buffer("calibrated", torch.ones((), dtype=torch.bool))   # (Pegasus 1's; the fixed phase replaces it)
        self.progress = 0.0
        self._compiled = None
        self._last = {}
        self._codes = None

    # ---------------------------------------------------------------- sizes, words, schedule
    def part_params(self):
        cnt = lambda mods: sum(p.numel() for m in mods for p in m.parameters())   # noqa: E731
        reader = cnt([self.reader, self.rnorm, self.route_q, self.route_k]) + 2
        pages = cnt([self.pool_k, self.pool_kn, self.pool_v, self.pnorm]) + self.pool_q.numel()
        think = cnt([self.thinker, self.think_in, self.tnorm]) + self.think_start.numel()
        guess = cnt([self.to_pieces, self.recon_fc1, self.recon_fc2, self.guess_z, self.guess_prev, self.guess_norm,
                     self.guess_fc1, self.guess_fc2, self.code_fc1, self.code_fc2]) + self.guess_pos.numel()
        speaker = cnt([self.speaker, self.plan, self.snorm]) + self.page_blank.numel()
        return {"word table": self.embed.weight.numel(), "reader": reader, "pages": pages, "thinker": think,
                "pieces+guesser": guess, "speaker": speaker}

    def work_per_token(self):
        pp = self.part_params()
        per_thought = pp["thinker"] + pp["pieces+guesser"]
        return pp["word table"] + pp["reader"] + pp["pages"] + pp["speaker"] + per_thought / self.cfg.thought_len

    def layout(self):
        c = self.cfg
        pp = self.part_params()
        return (f"Reader {c.enc_layers} x {c.d_tok} -> Speaker {c.spk_layers} x {c.d_tok} (one stream) · Thinker "
                f"{c.think_layers} x {c.d_think} ({pp['thinker'] / 1e6:.0f}M, rich thoughts) · {c.page_vecs} pages per "
                f"thought · {c.pieces} pieces of {c.n_codes():,} meanings (target only)")

    def set_progress(self, progress: float):
        self.progress = float(progress)

    def fixed_now(self) -> bool:
        return self.progress < self.cfg.fixed_until

    def meaning_now(self) -> float:
        a, b = (float(v) for v in str(self.cfg.meaning_warm).split(":"))
        w = 0.0 if self.progress < a else min(1.0, (self.progress - a) / max(1e-9, b - a))
        return self.cfg.meaning_weight * w

    def calibrate(self, idx):            # Pegasus 1's first-batch edge: not needed (the fixed phase teaches the router)
        return

    def flex_mode(self) -> int:
        return int(os.environ.get("OAZY_FLEX", self.cfg.flex))

    def _run(self, idx):
        fixed = self.fixed_now()
        if self._compiled is not None:
            mode = self.flex_mode()
            wm = self._wmask(idx.shape[1], idx.device) if mode & 1 else None
            return self._compiled(idx, wm, bool(mode & 2), fixed)
        return self._core(idx, None, False, fixed)

    # ---------------------------------------------------------------- the forward pass
    def _fixed_starts(self, idx):
        """Fixed thoughts: a new one after a sentence end, or every fixed_len tokens (fixed_len / 2 without a tokenizer)."""
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        start = (pos % self._every == 0)[None].expand(B, T).clone()
        start[:, 1:] |= torch.isin(idx[:, :-1], self.ends)
        return start

    def _core(self, idx, wmask=None, note_flex=False, fixed=False, plan_on=True, note_on=True):
        """(B, T) tokens -> (Speaker's vectors after the final norm, {losses}, {stats}). fixed: the fixed boundaries
        (early training); plan_on / note_on: switch the Thinker's plan or the notebook off (probes only)."""
        c = self.cfg
        B, T = idx.shape
        M = max(2, T // c.max_thoughts)
        P = c.page_vecs
        D = c.d_tok
        x = self.embed(idx)
        h = x
        for b in self.reader:
            h = b(h, wmask=wmask)
        hr = self.rnorm(h)
        # ---- where thoughts start
        q = F.normalize(self.route_q(hr[:, 1:]).float(), dim=-1)
        k = F.normalize(self.route_k(hr[:, :-1]).float(), dim=-1)
        change = (1 - (q * k).sum(-1)) / 2
        logit = self.route_scale * (change - self.route_edge)                              # (B, T-1) fp32
        p = torch.sigmoid(logit)
        p = torch.cat([torch.ones_like(p[:, :1]), p], 1)
        from oazy.tok import eot_id
        doc_start = torch.zeros_like(idx, dtype=torch.bool)
        doc_start[:, 1:] = idx[:, :-1] == eot_id(c.tokenizer)
        aux = {}
        if fixed:
            start = self._fixed_starts(idx) | doc_start
            start[:, 0] = True
            conf = torch.ones_like(p)
            aux["imitate"] = F.binary_cross_entropy_with_logits(logit.float(), start[:, 1:].float())   # (autocast-safe)
        else:
            start = (p >= 0.5) | doc_start
            start[:, 0] = True
            conf = torch.where(start, p, 1 - p)
            conf = conf + (1 - conf).detach()
            N = float(c.thought_len)
            Fr, G = start.float().mean(), p.mean()
            aux["ratio"] = N / (N - 1) * ((N - 1) * Fr * G + (1 - Fr) * (1 - G))
        tid = (start.long().cumsum(1) - 1).clamp(max=M - 1)
        # ---- pages: P learned queries attend over each thought's own tokens
        flat = (tid + torch.arange(B, device=idx.device)[:, None] * M).reshape(-1)          # (B*T,)
        sk = self.pool_kn(self.pool_k(hr)).float()                                          # (B, T, hd)
        sc = (sk @ self.pool_q.float().t()).float() / math.sqrt(c.head_dim)                  # (B, T, P), fp32 under autocast
        e = torch.exp(10 * torch.tanh(sc / 10)).reshape(B * T, P)                           # bounded: no max needed
        den = torch.zeros(B * M, P, device=idx.device, dtype=torch.float32).index_add_(0, flat, e)
        w = e / den[flat].clamp_min(1e-20)
        v = self.pool_v(hr).reshape(B * T, P, D).float()
        pages = torch.zeros(B * M, P, D, device=idx.device, dtype=torch.float32).index_add_(0, flat, w[:, :, None] * v)
        count = torch.zeros(B * M, device=idx.device, dtype=torch.float32).index_add_(
            0, flat, torch.ones(B * T, device=idx.device, dtype=torch.float32))
        valid = (count > 0).view(B, M)
        pages = self.pnorm(pages.view(B, M, P, D).to(hr.dtype))
        both = pages.reshape(B, M, P * D)
        # ---- rich thoughts for the Thinker; pieces as its target
        r = self.think_in(both)                                                             # (B, M, Dt)
        lv = c.level_list()
        val, code = self.fsq(self.to_pieces(both).view(B, M, c.pieces, len(lv)))
        val = val.to(hr.dtype)
        rec = self.recon_fc2(F.silu(self.recon_fc1(val.reshape(B, M, -1))))
        vm = valid.float()
        aux["recon"] = ((1 - F.cosine_similarity(rec.float(), r.detach().float(), dim=-1)) * vm).sum() / vm.sum().clamp_min(1)
        z = r
        for b in self.thinker:
            z = b(z)
        z = self.tnorm(z)
        # ---- the meaning loss: from the plan after thought j, guess thought j+1's pieces
        nxt_val, nxt_code = val[:, 1:].detach(), code[:, 1:]
        prev = torch.cumsum(self.guess_prev(nxt_val), 2) - self.guess_prev(nxt_val)
        a = self.guess_z(z[:, :-1])[:, :, None] + self.guess_pos[None, None] + prev
        rr = self.guess_fc2(F.silu(self.guess_fc1(self.guess_norm(a))))
        table = self.code_fc2(F.silu(self.code_fc1(self.fsq.grid.to(rr.dtype))))
        logits = (rr @ table.t()).float()
        bm = (valid[:, 1:] & valid[:, :-1]).float()[:, :, None].expand(-1, -1, c.pieces)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), nxt_code.reshape(-1), reduction="none")
        meaning = (ce * bm.reshape(-1)).sum() / bm.sum().clamp_min(1)
        # ---- Speaker: continues the Reader's stream; the plan for thought j; the notebook's earlier pages
        zs = torch.cat([self.think_start.to(z.dtype).expand(B, 1, -1), z[:, :-1]], 1)
        Pl = self.plan(zs).view(B, M, c.spk_layers, -1)
        plan_t = torch.gather(Pl, 1, tid[:, :, None, None].expand(-1, -1, c.spk_layers, Pl.shape[-1]))
        plan_t = plan_t * conf[:, :, None, None].to(plan_t.dtype)
        if not plan_on:
            plan_t = plan_t * 0
        allp = pmask = None
        if c.note_every and note_on:
            allp = torch.cat([self.page_blank.to(pages.dtype).expand(B, 1, -1), pages.reshape(B, M * P, D)], 1)
            if not note_flex:
                pj = torch.arange(M * P + 1, device=idx.device)
                pmask = ((pj[None, None, :] == 0) | ((pj[None, None, :] - 1) // P < tid[:, :, None]))[:, None]
        s = h
        for i, b in enumerate(self.speaker):
            s = s + plan_t[:, :, i]
            s = b(s, allp, pmask, wmask, tid if (note_flex and allp is not None) else None)
        s = self.snorm(s)
        n_th = valid.float().sum(1).mean()
        stats = {"thought_len": T / n_th.clamp_min(1), "meaning": meaning.detach(), "start_p": p.mean().detach(),
                 "recon": aux["recon"].detach()}
        return s, meaning, aux, stats, code.detach(), valid

    def _body(self, idx, plan_on=True, note_on=True):
        return self.head(self._core(idx, None, False, self.fixed_now(), plan_on, note_on)[0])

    def losses(self, idx, targets, mentor=None, kd_weight=0.0):
        s, meaning, aux, st, code, valid = self._run(idx)
        th = tW = None
        if mentor is not None and kd_weight > 0:
            th, tW = mentor.hidden(idx, self.doc_mask(idx) if self.cfg.doc_mask else None)
        from oazy.lmloss import head_loss
        loss, ce, kd = head_loss(s, self.embed.weight, targets, self.cfg.softcap, th, tW, kd_weight, self.loss_chunk)
        c = self.cfg
        total = loss + self.meaning_now() * meaning + c.recon_weight * aux["recon"]
        total = total + (c.imitate_weight * aux["imitate"] if "imitate" in aux else c.ratio_weight * aux["ratio"])
        self._last = st
        self._codes = (code, valid)
        out = {"ce": ce}
        if th is not None:
            out["kd"] = kd
        return total, out

    @torch.no_grad()
    def code_usage(self):
        """How many of the 1,000 meanings are in use in the last lesson (exp of the code entropy; 1 = collapsed)."""
        if self._codes is None:
            return None
        code, valid = self._codes
        cs = code[valid].flatten()
        if cs.numel() == 0:
            return None
        cnt = torch.bincount(cs, minlength=self.cfg.n_codes()).float()
        pr = cnt / cnt.sum()
        return float(torch.exp(-(pr[pr > 0] * pr[pr > 0].log()).sum()))

    def thought_report(self):
        r = self._last
        if not r:
            return {}
        out = {"thought_len": float(r["thought_len"]), "meaning_loss": float(r["meaning"]), "recon": float(r["recon"]),
               "fixed": int(self.fixed_now()), "meaning_w": self.meaning_now()}
        u = self.code_usage()
        if u is not None:
            out["codes_used"] = u
        return out

    @torch.no_grad()
    def plan_report(self, windows, device, batch=16):
        """Does the Speaker use the Thinker? Its next-word loss on held-out windows with everything on, with the
        plan switched off, and with the notebook switched off; and the meanings in use."""
        was = self.training
        self.eval()
        xy = torch.as_tensor(windows).long().to(device)
        x, y = xy[:, :-1], xy[:, 1:]
        tot = {"on": 0.0, "plan_off": 0.0, "notebook_off": 0.0}
        n = 0
        codes = []
        for i in range(0, len(x), batch):
            xb, yb = x[i:i + batch], y[i:i + batch]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                for key, kw in (("on", {}), ("plan_off", {"plan_on": False}), ("notebook_off", {"note_on": False})):
                    out = self._core(xb, None, False, self.fixed_now(), **kw)
                    lg = self.head(out[0]).float()
                    tot[key] += float(F.cross_entropy(lg.transpose(1, 2), yb, reduction="none").mean(1).sum())
                    if key == "on":
                        codes.append(out[4][out[5]].flatten())
            n += len(xb)
        self.train(was)
        r = {k: v / max(1, n) for k, v in tot.items()}
        r["plan_gap"] = r["plan_off"] - r["on"]
        r["notebook_gap"] = r["notebook_off"] - r["on"]
        cs = torch.cat(codes) if codes else torch.zeros(0, dtype=torch.long)
        if cs.numel():
            cnt = torch.bincount(cs, minlength=self.cfg.n_codes()).float()
            pr = cnt[cnt > 0] / cnt.sum()
            r["codes_used"] = float(torch.exp(-(pr * pr.log()).sum()))
        return r


__all__ = ["Pegasus2", "Pegasus2Config"]
