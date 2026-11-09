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
    predict: float = 0.0         # Logos1.1: the core's own job, the job of a law. From the things as they are before a
                                 # stretch, it guesses how each will be spoken of IN that stretch (the mean of its
                                 # mentions there, read from the stream, no gradient through the target), graded by
                                 # cosine once the stretch is read. On all text, no marks. 0 = Logos1 as raced on Nov 9.
    focus: int = 0               # Logos1.1: the core also sees the latest words (the stream at the last piece before
                                 # the stretch) as one more member, so what it thinks can depend on what is being asked

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
    """Bee6's stream + the linker + the laws core + the read-back (see the top of this file). The linker's pointer head
    is Bee6's own link head (bee6/model.py), so a Bee6 given the same answer key (--set teach=1) has exactly the same one."""
    needs_linker = True

    def __init__(self, cfg: LogosConfig):
        super().__init__(cfg)
        D, dc = cfg.d_model, cfg.d_core
        self.core_in = nn.Linear(D, dc, bias=False)
        self.rank = nn.Parameter(torch.zeros(cfg.things, dc))
        self.core = nn.ModuleList(CoreLayer(dc, cfg.core_heads) for _ in range(cfg.core_layers))
        self.core_out = nn.Linear(dc, D, bias=False)
        self.reads = nn.ModuleDict({str(i): ReadBack(D, cfg.read_heads) for i in cfg.read_set()})
        if cfg.predict > 0:
            self.pred_out = nn.Linear(dc, D, bias=False)      # the law's guess of each thing in the coming stretch
        if cfg.focus:
            self.focus_emb = nn.Parameter(torch.zeros(1, dc))
        self.core_on = True
        self._init_logos()

    @torch.no_grad()
    def _init_logos(self):
        D, dc = self.cfg.d_model, self.cfg.d_core
        nn.init.normal_(self.core_in.weight, std=D ** -0.5)
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
        if hasattr(self, "pred_out"):
            nn.init.normal_(self.pred_out.weight, std=dc ** -0.5)
        if hasattr(self, "focus_emb"):
            nn.init.normal_(self.focus_emb, std=0.02)

    # ---------------------------------------------------------------- names
    def layout(self):
        c = self.cfg
        return (super().layout() + f"; linker after layer {c.link_at}, laws core {c.core_layers} x {c.d_core} x {c.passes} "
                f"passes over {c.things} things every {c.chunk} pieces, read back before layers "
                + ",".join(str(i + 1) for i in c.read_set())
                + (f"; the core guesses the next stretch's things (weight {c.predict:g})" if c.predict > 0 else "")
                + ("; the core sees the latest words" if c.focus else ""))

    def make_optimizer(self, lr, device, kind="muon", muon_lr=0.03, muon_wd=0.1):
        """Bee6's (NorMuon for the grids, Adam for the rest) with the linker's, core's and read-back's grids on NorMuon too."""
        from bee5 import optim as O
        from oazy.muon import Combined
        muon, adam = O.groups(self)
        mine = {id(p) for n, p in self.named_parameters() if p.dim() == 2 and n.split(".")[0] in
                ("core_in", "core", "core_out", "reads", "pred_out") and not n.endswith("empty")}
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
        if self.cfg.focus:              # one more member: the stream at the last piece before the stretch (none for the first)
            f = torch.cat([x[:, :1] * 0, x[:, C - 1::C][:, :nC - 1]], 1)                   # [B, nC, D]
            fz = self.core_in(norm(f)) + self.focus_emb.to(x.dtype)
            z = torch.cat([z, fz.reshape(B * nC, 1, -1)], 1)
            has_f = (torch.arange(nC, device=x.device) > 0)[None].expand(B, nC).reshape(B * nC, 1)
            okf = torch.cat([okf, has_f], 1)
        Kz = z.shape[1]
        allow = okf[:, None, :] | torch.eye(Kz, dtype=torch.bool, device=x.device)[None]
        for _ in range(self.cfg.passes):
            for layer in self.core:
                z = layer(z, allow)
        z = z[:, :K]
        S = self.core_out(norm(z)).view(B, nC, K, D)
        reads = ok[:, :, None, :] & (tdoc[:, :, None, :] == doc.view(B, nC, C)[..., None])
        extra = self._law(x, m_logit, parent, sel, ok, things, z) if self.cfg.predict > 0 else None
        return S, reads, extra

    def _law(self, x, m_logit, parent, sel, ok, things, z):
        """The core's own lesson (Logos1.1): for every thing it holds before stretch c that is spoken of in stretch c,
        guess the mean of those new mentions (the stream after layer link_at, held still as the target) from the core's
        output alone. Also the plain "copy" guess (the thing as it was) for comparison: the law has to beat it."""
        B, T, D = x.shape
        C = self.cfg.chunk
        nC, K = sel.shape[1], sel.shape[2]
        ar = torch.arange(T, device=x.device)
        with torch.no_grad():
            is_m = m_logit > 0
            inside = is_m[:, None, :] & ((ar // C)[None, None, :] == torch.arange(nC, device=x.device)[None, :, None])
            nxt = (parent[:, None, None, :] == sel[..., None]) & inside[:, :, None, :] & ok[..., None]   # [B, nC, K, T]
            has = nxt.any(-1).float()
            wn = nxt.to(x.dtype) * torch.sigmoid(m_logit).to(x.dtype)[:, None, None, :]
            wn = wn / wn.sum(-1, keepdim=True).clamp_min(1e-6)
            tgt = torch.einsum("bckt,btd->bckd", wn, x.detach()).float()
            copy = F.cosine_similarity(things.detach().float(), tgt, dim=-1)
        pred = self.pred_out(norm(z)).view(B, nC, K, D).float()
        cos = F.cosine_similarity(pred, tgt, dim=-1)
        n = has.sum().clamp_min(1.0)
        return {"aux": self.cfg.predict * ((1 - cos) * has).sum() / n, "law_cos": (cos * has).sum() / n,
                "copy_cos": (copy * has).sum() / n, "law_n": has.sum() / max(1, B * nC)}

    # ---------------------------------------------------------------- the body: Bee6's loop, with the core around it
    def _pad(self, idx):
        """Reading any length: pad at the end to whole stretches (later pieces change nothing before them)."""
        C = self.cfg.chunk
        if idx.shape[1] % C:
            from oazy.tok import eot_id
            idx = torch.cat([idx, idx.new_full((idx.shape[0], C - idx.shape[1] % C), eot_id(self.cfg.tokenizer))], 1)
        return idx

    def _begin(self, idx):
        return {"doc": self.docs(idx)}

    def _before(self, i, x, keep):
        if "S" in keep and str(i) in self.reads:
            x = x + self.reads[str(i)](x, keep["S"], keep["reads"])
        return x

    def _after_link(self, x, keep):
        if self.core_on:
            keep["S"], keep["reads"], extra = self.think(x, keep["doc"])
            if extra is not None:
                keep["extra"] = extra

    def _run_to_link(self, idx):
        """The stream after layer link_at (what the linker reads; the core cannot change it)."""
        return self._trunk_aux(idx)[4]

    # ---------------------------------------------------------------- the trainer's hooks
    @torch.no_grad()
    def core_report(self, windows, device, batch=16):
        """Does the stream use the core? Next-word loss on held-out windows with the read-back on, and off."""
        was = self.training
        self.eval()
        xy = torch.as_tensor(windows).long().to(device)
        x, y = xy[:, :-1], xy[:, 1:]
        tot, n = {"on": 0.0, "core_off": 0.0}, 0
        law = {}
        try:
            for i in range(0, len(x), batch):
                xb, yb = x[i:i + batch], y[i:i + batch]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    for key, on in (("on", True), ("core_off", False)):
                        self.core_on = on
                        out = self._trunk_aux(xb)
                        lg = self.scores(out[0]).float()
                        tot[key] += float(F.cross_entropy(lg.transpose(1, 2), yb, reduction="none").mean(1).sum())
                        if on and len(out) > 5:
                            for k in ("law_cos", "copy_cos"):
                                law[k] = law.get(k, 0.0) + float(out[5][k]) * len(xb)
                n += len(xb)
        finally:
            self.core_on = True
            self.train(was)
        r = {k: v / max(1, n) for k, v in tot.items()}
        r["core_gap"] = r["core_off"] - r["on"]
        r.update({k: v / max(1, n) for k, v in law.items()})
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
