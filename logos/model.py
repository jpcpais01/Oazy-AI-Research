"""Logos1-200M: Bee6's word stream with Logos1's way of thinking built in beside it. A NEXT-WORD model: it reads text
and guesses the very next token, like every Bee, and it is scored the same way (HellaSwag, ARC, PIQA, LogT, the
world's secret exams as plain text).

João's go, Nov 9: the design from the world-first thread, raced fairly against Bee6 on Emerald.

  word stream   Bee6 exactly (fused q/k/v, FP8 layer products, Mixture of Depths, value tables, U-net skips, ...) but
                10 layers of 1,024 instead of 12. It holds the words, the facts and the context, like any 200M model.
  linker        reads the stream after layer 4. Each piece says whether it names a thing, and points back to the
                earlier piece of the same document that names the same thing, or says "new thing" (Logos1's linker).
                Following the pointers back gives chains: one chain = one thing.
  laws core     every 32 pieces, the (up to) 16 things mentioned most recently before that point, each the weighted
                mean of its pieces, go through a small shared core: 2 layers of 768, 3 passes, the things looking at
                each other (Logos1's core). It sees only things, never words, so it cannot store facts.
  read-back     before layers 6 and 9 every piece reads the things of its stretch (cross-attention, plus an "empty"
                slot), only things of its own document. A piece only ever sees things built from text before its
                stretch began, so it stays honest next-word prediction.
  answer key    on the world's episodes, Emerald's marks (which words name the same thing; oazy/emerald.py) grade the
                linker, exactly as in world-link. On ordinary text there are no marks: the linker runs on its own.

Sizes: Bee6 202.0M; Logos1-200M 201.1M (2 layers fewer, the linker 0.3M, the core 15.7M, the read-back 4.2M). Work per
piece about the same as Bee6's (the core runs on 16 things per 32 pieces). The "core off" check (core_report) reads
held-out text with the read-back switched off: if the loss barely moves, the stream is ignoring the core.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from bee5.model import norm
from bee6.model import Bee6, Bee6Config

NO_MARK = 255


@dataclass
class LogosConfig(Bee6Config):
    n_layer: int = 10            # Bee6's 12 minus the two whose weights pay for the linker, the core and the read-back
    link_at: int = 4             # the linker reads the stream after this many layers
    read_at: str = "5,8"         # the read-back comes before these layers (from 0): the 6th and the 9th
    chunk: int = 32              # pieces per stretch: the things are rebuilt at the start of every stretch
    things: int = 16             # things the core holds at once (the most recently mentioned)
    d_core: int = 768
    core_layers: int = 2
    core_heads: int = 12
    passes: int = 3              # the core's passes (shared weights), as in world-link
    link_dim: int = 128          # the linker's pointer vectors
    read_heads: int = 8
    teach: float = 1.0           # the weight of the marks' answer key (world episodes only), as in world-link

    def __post_init__(self):
        super().__post_init__()
        if self.bloom:
            raise ValueError("Logos1-200M does not bloom")
        self.link_at = max(1, min(self.link_at, self.n_layer - 1))

    def read_set(self):
        rs = sorted({int(s) for s in str(self.read_at).split(",") if s.strip() != ""})
        rs = [i for i in rs if self.link_at <= i < self.n_layer]
        return rs or [self.n_layer - 1]


class CoreLayer(nn.Module):
    """One layer of the laws core: the things look at each other (no order among them), then think alone."""

    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.up = nn.Linear(d, 4 * d, bias=False)
        self.down = nn.Linear(4 * d, d, bias=False)

    def forward(self, z, allow):
        N, K, d = z.shape
        q, k, v = self.qkv(norm(z)).view(N, K, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        q, k = norm(q), norm(k)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=allow[:, None])
        z = z + self.out(y.transpose(1, 2).reshape(N, K, d))
        return z + self.down(F.relu(self.up(norm(z))).square())


class ReadBack(nn.Module):
    """Each piece reads the things of its stretch (cross-attention), plus an always-there empty slot."""

    def __init__(self, D, heads, hd=64):
        super().__init__()
        self.h, self.hd = heads, hd
        self.q = nn.Linear(D, heads * hd, bias=False)
        self.k = nn.Linear(D, heads * hd, bias=False)
        self.v = nn.Linear(D, heads * hd, bias=False)
        self.out = nn.Linear(heads * hd, D, bias=False)
        self.empty = nn.Parameter(torch.zeros(1, D))

    def forward(self, x, S, allow):
        """x (B, T, D) the stream; S (B, nC, K, D) the things of each stretch; allow (B, nC, C, K)."""
        B, T, D = x.shape
        nC, K = S.shape[1], S.shape[2]
        C = T // nC
        H, hd = self.h, self.hd
        q = norm(self.q(norm(x)).view(B * nC, C, H, hd)).transpose(1, 2)
        src = torch.cat([self.empty.to(S.dtype).expand(B, nC, 1, D), S], 2).view(B * nC, K + 1, D)
        k = norm(self.k(norm(src)).view(B * nC, K + 1, H, hd)).transpose(1, 2)
        v = self.v(src).view(B * nC, K + 1, H, hd).transpose(1, 2)
        a = torch.cat([torch.ones_like(allow[..., :1]), allow], -1).view(B * nC, 1, C, K + 1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=a)
        return self.out(y.transpose(1, 2).reshape(B, T, H * hd))


class Logos(Bee6):
    """Bee6's stream + the linker + the laws core + the read-back (see the top of this file)."""

    def __init__(self, cfg: LogosConfig):
        super().__init__(cfg)
        D, L, dc = cfg.d_model, cfg.link_dim, cfg.d_core
        self.lk_ment = nn.Linear(D, 1)
        self.lk_new = nn.Linear(D, 1)
        self.lk_q = nn.Linear(D, L, bias=False)
        self.lk_k = nn.Linear(D, L, bias=False)
        self.core_in = nn.Linear(D, dc, bias=False)
        self.rank = nn.Parameter(torch.zeros(cfg.things, dc))
        self.core = nn.ModuleList(CoreLayer(dc, cfg.core_heads) for _ in range(cfg.core_layers))
        self.core_out = nn.Linear(dc, D, bias=False)
        self.reads = nn.ModuleDict({str(i): ReadBack(D, cfg.read_heads) for i in cfg.read_set()})
        self.core_on = True
        self._feed = None
        self._init_logos()

    @torch.no_grad()
    def _init_logos(self):
        D, dc = self.cfg.d_model, self.cfg.d_core
        for lin in (self.lk_q, self.lk_k, self.core_in):
            nn.init.normal_(lin.weight, std=D ** -0.5)
        nn.init.normal_(self.lk_ment.weight, std=D ** -0.5)
        nn.init.normal_(self.lk_new.weight, std=D ** -0.5)
        nn.init.zeros_(self.lk_ment.bias)
        nn.init.zeros_(self.lk_new.bias)
        nn.init.normal_(self.rank, std=0.02)
        for c in self.core:
            nn.init.normal_(c.qkv.weight, std=dc ** -0.5)
            nn.init.normal_(c.up.weight, std=dc ** -0.5)
            nn.init.zeros_(c.out.weight)                  # every output starts at zero, as in Bee: the core starts quiet
            nn.init.zeros_(c.down.weight)
        nn.init.normal_(self.core_out.weight, std=dc ** -0.5)
        for r in self.reads.values():
            for lin in (r.q, r.k, r.v):
                nn.init.normal_(lin.weight, std=D ** -0.5)
            nn.init.zeros_(r.out.weight)                  # the read-back starts as a no-op: Bee6's stream, untouched
            nn.init.normal_(r.empty, std=0.02)

    # ---------------------------------------------------------------- names
    def layout(self):
        c = self.cfg
        return (super().layout() + f"; linker after layer {c.link_at}, laws core {c.core_layers} x {c.d_core} x {c.passes} "
                f"passes over {c.things} things every {c.chunk} pieces, read back before layers "
                + ",".join(str(i + 1) for i in c.read_set()))

    def make_optimizer(self, lr, device, kind="muon", muon_lr=0.03, muon_wd=0.1):
        """Bee6's (NorMuon for the grids, Adam for the rest) with the linker's, core's and read-back's grids on NorMuon too."""
        from bee5 import optim as O
        from oazy.muon import Combined
        muon, adam = O.groups(self)
        mine = {id(p) for n, p in self.named_parameters() if p.dim() == 2 and n.split(".")[0] in
                ("lk_q", "lk_k", "core_in", "core", "core_out", "reads") and not n.endswith("empty")}
        adam["gates"] = [p for p in adam["gates"] if id(p) not in mine]
        muon += [(n, p) for n, p in self.named_parameters() if id(p) in mine]
        gs = [dict(params=ps, weight_decay=0.0, name=k, lr=lr * O.ADAM[k]["lr_scale"], **O.ADAM[k]) for k, ps in adam.items() if ps]
        if kind != "muon":
            gs.append(dict(params=[p for _, p in muon], weight_decay=0.1, name="grids", lr=lr, lr_scale=1.0, betas=(0.9, 0.95)))
            muon = []
        kw = dict(lr=lr, eps=1e-10)
        if device.type == "cuda":
            kw["fused"] = True
        opt = torch.optim.AdamW(gs, **kw)
        if not muon:
            return Combined(opt)
        mu = O.NorMuon([dict(params=[p for _, p in muon], lr_scale=muon_lr / lr)], lr=muon_lr, weight_decay=muon_wd)
        return Combined(mu, opt)

    # ---------------------------------------------------------------- the thinking part
    def docs(self, idx):
        """Which document each piece belongs to (an <|endoftext|> ends one)."""
        from oazy.tok import eot_id
        d = torch.zeros_like(idx)
        d[:, 1:] = (idx[:, :-1] == eot_id(self.cfg.tokenizer)).long().cumsum(1)
        return d

    def link_scores(self, x):
        """(mention logit [B, T], pointer scores [B, T, T], new-thing logit [B, T]) from the stream after layer link_at."""
        xn = norm(x)
        m = self.lk_ment(xn).squeeze(-1).float()
        sc = torch.einsum("bil,bjl->bij", self.lk_q(xn), self.lk_k(xn)).float() / math.sqrt(self.cfg.link_dim)
        return m, sc, self.lk_new(xn).squeeze(-1).float()

    def think(self, x, doc):
        """The stream after layer link_at -> the things of every stretch after the core (B, nC, K, D), which may be read
        (B, nC, C, K). Stretch c holds only what was mentioned before it began."""
        B, T, D = x.shape
        C, K = self.cfg.chunk, self.cfg.things
        nC = T // C
        m_logit, sc, new = self.link_scores(x)
        ar = torch.arange(T, device=x.device)
        with torch.no_grad():
            is_m = m_logit > 0
            cand = (ar[None, :] < ar[:, None])[None] & is_m[:, None, :] & (doc[:, :, None] == doc[:, None, :])
            best, arg = sc.masked_fill(~cand, -1e9).max(-1)
            joins = is_m & cand.any(-1) & (best > new)
            parent = torch.where(joins, arg, ar[None].expand(B, T))
            for _ in range(max(1, math.ceil(math.log2(T))) + 1):           # follow the pointers to the chain's start
                parent = parent.gather(1, parent)
            starts = (torch.arange(nC, device=x.device) * C)[None, :, None]  # [1, nC, 1]
            seen = is_m[:, None, :] & (ar[None, None, :] < starts)          # [B, nC, T]: mentioned before stretch c
            when = torch.where(seen, ar[None, None, :].expand(B, nC, T), torch.full_like(seen, -1, dtype=torch.long))
            last = torch.full((B, nC, T), -1, dtype=torch.long, device=x.device).scatter_reduce(
                2, parent[:, None, :].expand(B, nC, T), when, "amax", include_self=True)
            vals, sel = last.topk(K, dim=-1)                                # the K things mentioned most recently
            ok = vals >= 0
            member = (parent[:, None, None, :] == sel[..., None]) & seen[:, :, None, :] & ok[..., None]   # [B, nC, K, T]
            tdoc = doc.gather(1, sel.reshape(B, -1)).view(B, nC, K)
        w = member.to(x.dtype) * torch.sigmoid(m_logit).to(x.dtype)[:, None, None, :]
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-6)
        things = torch.einsum("bckt,btd->bckd", w, x)
        z = self.core_in(norm(things)) + self.rank.to(x.dtype)
        z = z.reshape(B * nC, K, -1)
        okf = ok.reshape(B * nC, K)
        allow = okf[:, None, :] | torch.eye(K, dtype=torch.bool, device=x.device)[None]
        for _ in range(self.cfg.passes):
            for layer in self.core:
                z = layer(z, allow)
        S = self.core_out(norm(z)).view(B, nC, K, D)
        reads = ok[:, :, None, :] & (tdoc[:, :, None, :] == doc.view(B, nC, C)[..., None])
        return S, reads

    # ---------------------------------------------------------------- the body
    def _trunk_aux(self, idx):
        """(B, T) -> (vectors after the final norm, MoD helpers' loss, their agreement, the stream after layer link_at)."""
        T0 = idx.shape[1]
        C = self.cfg.chunk
        if T0 % C:                                   # reading any length: pad at the end (later pieces change nothing before)
            from oazy.tok import eot_id
            idx = torch.cat([idx, idx.new_full((idx.shape[0], C - T0 % C), eot_id(self.cfg.tokenizer))], 1)
        T = idx.shape[1]
        doc = self.docs(idx)
        e, _ = self._embed(idx)
        x0 = norm(e)
        ves = [t(idx) for t in self.ve]
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        aux, keep = [], {}

        def layer(i, x, ve):
            if "S" in keep and str(i) in self.reads and self.core_on:
                x = x + self.reads[str(i)](x, keep["S"], keep["reads"])
            x, a = self.blocks[i](x, ve, cos, sin)
            if a is not None:
                aux.append(a)
            if i == self.cfg.link_at - 1:
                keep["x"] = x
                if self.core_on:
                    keep["S"], keep["reads"] = self.think(x, doc)
            return x
        h = self._run(x0, ves, layer)[:, :T0]
        xl = keep["x"][:, :T0]
        if aux:
            return h, torch.stack([a for a, _ in aux]).mean(), torch.stack([g for _, g in aux]).mean(), xl
        z = h.new_zeros((), dtype=torch.float32)
        return h, z, z, xl

    def _trunk(self, idx):
        return self._trunk_aux(idx)[0]

    def _run_to_link(self, idx):
        """The stream after layer link_at (what the linker reads; the core cannot change it)."""
        return self._trunk_aux(idx)[3]

    def feed_marks(self, marks, rows):
        """The trainer hands over the next pass's marks (B, T; 255 = none) and which rows are world episodes."""
        self._feed = (marks, rows)

    def link_loss(self, idx, xl, marks, rows):
        """The marks as the linker's answer key, on the world's rows only (worldfirst's link_loss, as in world-link)."""
        from worldfirst.model import link_loss
        xw, iw, mk = xl[rows], idx[rows], marks[rows].long()
        doc = self.docs(iw)
        m, sc, new = self.link_scores(xw)
        sc = sc.masked_fill(doc[:, :, None] != doc[:, None, :], -1e9)
        ment = torch.where(mk != NO_MARK, doc * 256 + mk, torch.full_like(mk, -1))
        with torch.autocast(idx.device.type, enabled=False):
            return link_loss(m, sc, new, torch.ones_like(ment, dtype=torch.bool), ment)

    def losses(self, idx, targets, mentor=None, kd_weight=0.0):
        if self.token_select is not None:
            raise SystemExit("Logos1-200M does not take --select")
        feed, self._feed = self._feed, None
        if self._compiled is not None and self.training:
            if self.cfg.graphs:
                torch.compiler.cudagraph_mark_step_begin()
            h, aux, agree, xl = self._compiled(idx)
        else:
            h, aux, agree, xl = self._trunk_aux(idx)
        th = tW = None
        if mentor is not None and kd_weight > 0:
            th, tW = mentor.hidden(idx, self.doc_mask(idx))
        w = self.mtp()
        tg = targets[..., None]
        if len(w) > 1:
            pad = torch.full_like(targets[:, :len(w) - 1], -1)
            ext = torch.cat([targets, pad], 1)
            tg = torch.stack([ext[:, k:k + targets.shape[1]] for k in range(len(w))], -1)
        from bee5.loss import head_loss, plain_loss
        f = head_loss if self.fused_loss else plain_loss
        loss, ce, kd = f(h, self.embed.weight, tg, w, self.cfg, th, tW, kd_weight, self.loss_chunk)
        stats = {"ce": ce}
        if th is not None:
            stats["kd"] = kd
        if self.cfg.mod_set() and self.training:
            loss = loss + self.cfg.mod_aux * aux
            stats["mod_agree"] = agree
            self._agree = agree
        if feed is not None and self.training and self.cfg.teach > 0 and bool(feed[1].any()):
            lk = self.link_loss(idx, xl, feed[0][:, :idx.shape[1]], feed[1])
            loss = loss + self.cfg.teach * lk
            stats["link"] = lk.detach()
            self._link = lk.detach()
        return loss, stats

    def thought_report(self):
        r = super().thought_report()
        lk = getattr(self, "_link", None)
        r["link"] = None if lk is None else float(lk)
        return r

    # ---------------------------------------------------------------- the trainer's hooks
    @torch.no_grad()
    def core_report(self, windows, device, batch=16):
        """Does the stream use the core? Next-word loss on held-out windows with the read-back on, and off."""
        was = self.training
        self.eval()
        xy = torch.as_tensor(windows).long().to(device)
        x, y = xy[:, :-1], xy[:, 1:]
        tot, n = {"on": 0.0, "core_off": 0.0}, 0
        try:
            for i in range(0, len(x), batch):
                xb, yb = x[i:i + batch], y[i:i + batch]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    for key, on in (("on", True), ("core_off", False)):
                        self.core_on = on
                        lg = self.scores(self._trunk(xb)).float()
                        tot[key] += float(F.cross_entropy(lg.transpose(1, 2), yb, reduction="none").mean(1).sum())
                n += len(xb)
        finally:
            self.core_on = True
            self.train(was)
        r = {k: v / max(1, n) for k, v in tot.items()}
        r["core_gap"] = r["core_off"] - r["on"]
        return r

    def startup_check(self, device, say=print):
        say(f"[logos] {self.layout()}")
        super().startup_check(device, say)

    def grow_to(self, progress: float) -> bool:
        return False

    # ---------------------------------------------------------------- reading as a stream (chat, samples)
    def _read(self, idx, states=None, looks=False):
        """No running state: the things are rebuilt from the whole text each time (slower, the same numbers)."""
        prev = states["ids"] if states else idx[:, :0]
        ids = torch.cat([prev, idx], 1)[:, -self.cfg.max_len:]
        logits = self.scores(self._trunk(ids))[:, -idx.shape[1]:]
        return logits, {"ids": ids, "pos": ids.shape[1]}, []
