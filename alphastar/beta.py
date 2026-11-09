"""Pegasus Beta: Pegasus Alpha Star that decides for itself how long to think (Nov 9 2026). Still a NEXT-WORD model.

Why. Alpha Star's first race (Nov 9, 200M tokens) showed a lazy loop: the held-out loss went 4.063 after 1 pass, 3.996
after 2, then 3.990 / 3.989 / 3.989 after 4 / 8 / 16. Its thought settled after about 2 passes because nothing asked it
to do more: the number of passes was a schedule we set, and the stop was a rule we guessed (move less than 3%).

What changes (PonderNet, Banino et al. 2021; nothing hand-set about where to stop):

  1. A stop head. After every pass a small head reads each word's thought and says how sure it is that it can stop
     now (lambda). From these, every word gets a chance of stopping after pass 1, 2, ..., 8 (p).
  2. It pays for stopping too early, and a little for thinking long. On 2 random rows of each trainer pass (4,096
     words) the speaker's next-word loss is measured after every pass; the stop head is trained on the loss it would
     get where it stops, on average: sum_k p_k x loss_k. Plus a small pull (weight 0.01) towards a gentle prior (a
     geometric chance of stopping, 4 passes on average), so early on it explores instead of fixing on one depth.
  3. The model chooses how deep it practises. Alpha Star drew its number of passes from a fixed schedule; Beta draws
     it from its own stopping habits (a running average of p), mixed half and half with the prior so that it keeps
     practising depths it does not yet pick. The words it stops early on make it good early; the words it thinks long
     on make it good late. "General first, crisp later" is what this rewards; nothing tells it so.

Reading: each word stops at the first pass where its chance of having stopped reaches one half (its median), at most
16 passes; a word that is done keeps its thought. Writing word by word: a fixed gen_passes, as Alpha Star.

Cost: the whole batch thinks only as deep as the depth it practises (4 passes on average); only the 2 sampled rows
think on to pass 8 and are read by the speaker after every pass, so a step costs about what Alpha Star's does.

Everything else is Alpha Star's: reader 2 + core 4 looped + speaker 2, the fact drawer, the next-thought lesson,
gradients through the last 2 passes, Plain-100M's recipe. The stop head adds about 0.2M numbers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from alphastar.model import AlphaStar, AlphaStarConfig


@dataclass
class BetaConfig(AlphaStarConfig):
    ponder_max: int = 8              # passes run while training (the stop head spreads its chances over these)
    ponder_prior: float = 4.0        # the prior's average passes (a geometric chance of stopping, 1/4 per pass)
    ponder_beta: float = 0.01        # the pull towards the prior (PonderNet's value)
    ponder_weight: float = 1.0       # the stop head's lesson next to the next-word loss
    ponder_rows: int = 2             # rows (sequences) per trainer pass whose loss is measured after every pass
    ponder_explore: float = 0.5      # the depth practised: this share from the prior, the rest from its own habits
    halt_at: float = 0.5             # reading: stop once the chance of having stopped reaches this
    halt_hidden: int = 256


def geometric(lam: float, n: int) -> torch.Tensor:
    """The prior: stop after pass k with lam (1 - lam)^(k-1); the last pass takes what is left."""
    k = torch.arange(n, dtype=torch.float64)
    g = lam * (1 - lam) ** k
    g[-1] = (1 - lam) ** (n - 1)
    return g.float()


def stop_chances(lam: torch.Tensor) -> torch.Tensor:
    """lam (..., N): the chance to stop at each pass given it got there -> p (..., N), the chance to stop exactly there.
    The last pass takes what is left, so p sums to 1."""
    lam = lam.float()
    lam = torch.cat([lam[..., :-1], torch.ones_like(lam[..., -1:])], -1)
    left = torch.cumprod(1 - lam, -1)
    left = torch.cat([torch.ones_like(left[..., :1]), left[..., :-1]], -1)
    return lam * left


class PegasusBeta(AlphaStar):
    def __init__(self, cfg: BetaConfig):
        super().__init__(cfg)
        D = cfg.d_model
        self.halt = nn.Sequential(nn.Linear(D, cfg.halt_hidden, bias=False), nn.GELU(), nn.Linear(cfg.halt_hidden, 1))
        nn.init.normal_(self.halt[0].weight, std=0.02)
        nn.init.normal_(self.halt[2].weight, std=0.002)   # near zero: it starts at the prior
        with torch.no_grad():                      # start at the prior's chance per pass
            self.halt[2].bias.fill_(math.log(1 / (cfg.ponder_prior - 1)))
        self.register_buffer("habit", geometric(1 / cfg.ponder_prior, cfg.ponder_max))   # running average of p (in ckpt)
        self._prior = geometric(1 / cfg.ponder_prior, cfg.ponder_max)

    def layout(self):
        c = self.cfg
        return super().layout() + f", it chooses its own passes (up to {c.ponder_max} training, stop head, prior {c.ponder_prior:g})"

    def set_progress(self, progress: float):
        super().set_progress(progress)             # the next-thought warm-up (the pass curriculum is not used)

    def lam(self, s):
        """The chance to stop now, given it got this far: (...,) from thoughts (..., D)."""
        with torch.autocast(s.device.type, enabled=False):      # in 32-bit: the stop chances' cumulative product needs it
            return torch.sigmoid(self.halt(s.float()).squeeze(-1))

    def sample_passes(self) -> int:
        """The depth practised on this trainer pass: from its own habits, mixed with the prior."""
        if self.force_passes:
            return int(self.force_passes)
        c = self.cfg
        q = (1 - c.ponder_explore) * self.habit.double().cpu().numpy() + c.ponder_explore * self._prior.double().numpy()
        return int(self._rng.choice(len(q), p=q / q.sum())) + 1

    # ---------------------------------------------------------------- reading: its own stop
    @torch.no_grad()
    def think_test(self, e, passes=None, eps=None, every=None, fast=False):
        """Fixed passes, or (passes=None) each word stops once its chance of having stopped reaches halt_at, at most
        test_max. A word that is done keeps its thought."""
        c = self.cfg
        fixed = passes or (c.test_passes or None)
        if fixed is not None:
            return super().think_test(e, passes=fixed, eps=eps, every=every, fast=fast)
        step = self._fn("step") if fast else self._step
        B, T, _ = e.shape
        s = self.start(B, T, 0, e.device, e.dtype, True)
        used = torch.zeros(B, T, device=e.device)
        done = torch.zeros(B, T, dtype=torch.bool, device=e.device)
        left = torch.ones(B, T, device=e.device)
        for i in range(c.test_max):
            new = torch.where(done[..., None], s, step(s, e))
            used += (~done).float()
            left = torch.where(done, left, left * (1 - self.lam(new)))
            done |= (1 - left) >= c.halt_at
            s = new
            if every is not None:
                every(i + 1, s)
            if bool(done.all()):
                break
        return s, used

    # ---------------------------------------------------------------- the lesson
    def losses(self, idx, targets, mentor=None, kd_weight=0.0):
        learn = self.training and torch.is_grad_enabled()
        if not learn:
            return super().losses(idx, targets, mentor, kd_weight)
        c = self.cfg
        use_mentor = mentor is not None and kd_weight > 0
        resets, mask, _ = self.doc_info(idx, need_mask=use_mentor)
        step, speak = self._fn("step"), self._fn("speak")
        e = self._fn("read")(idx)
        N = c.ponder_max
        k_star = min(self.sample_passes(), N)      # the depth this pass practises (its thought after k_star is spoken)
        G = min(c.grad_passes, k_star)
        B, T, _ = e.shape
        s = self.start(B, T, 0, e.device, e.dtype, c.train_noise == "table")
        ed = e.detach()
        rows = torch.randperm(B, device=idx.device)[:min(c.ponder_rows, B)]
        states, outs = [], []
        for k in range(1, k_star + 1):             # the whole batch thinks only as deep as it practises
            if k > k_star - G:
                s = step(s, e)                     # the passes that learn
                outs.append(s)
            else:
                with torch.no_grad():
                    s = step(s.detach(), ed)
            states.append(s[rows].detach())
        with torch.no_grad():                      # a few rows think on to the last pass, for the stop head's lesson
            sr, er = s[rows].detach(), ed[rows]
            for k in range(k_star + 1, N + 1):
                sr = self._step(sr, er)
                states.append(sr)
            # the stop head's lesson: on those rows, the loss after every pass (no gradient to the trunk)
            y = targets[rows].reshape(-1)
            ce = torch.stack([self.token_ce(self._speak(st), targets[rows].clamp(min=0)).reshape(-1) for st in states], -1)   # (m, N)
        ok = y >= 0
        lam = self.lam(torch.stack([st.reshape(-1, st.shape[-1]) for st in states], 1))   # (m, N)
        p = stop_chances(lam)
        prior = self._prior.to(p.device)
        kl = (p * (p.clamp_min(1e-9).log() - prior.log())).sum(-1)
        okf = ok.float()
        n_ok = okf.sum().clamp(min=1)
        ponder = ((p * ce).sum(-1) * okf).sum() / n_ok + c.ponder_beta * (kl * okf).sum() / n_ok
        with torch.no_grad():
            self.habit.mul_(0.99).add_(0.01 * (p * okf[:, None]).sum(0) / n_ok)
        # the next-word lesson on the thought after k_star passes (as Alpha Star)
        h = speak(outs[-1])
        th = tW = None
        if use_mentor:
            th, tW = mentor.hidden(idx, mask)
        if self.fused_loss:
            from oazy.lmloss import head_loss
            sel = getattr(self, "token_select", None)
            loss, ce_main, kd = head_loss(h, self.embed.weight, targets, c.softcap, th, tW, kd_weight, self.loss_chunk, sel)
        else:
            from oazy.lmloss import plain_loss
            loss, ce_main, kd = plain_loss(h, self.embed.weight, targets, c.softcap, th, tW, kd_weight)
        loss = loss + c.ponder_weight * ponder
        depth = torch.arange(1, N + 1, device=p.device, dtype=p.dtype)
        stats = {"ce": ce_main, "passes": ((p.detach() * depth).sum(-1) * okf).sum() / n_ok, "ponder": ponder.detach()}
        if th is not None:
            stats["kd"] = kd
        if c.thought and self._ready:
            pts = self.fore_points(idx, targets)
            first = k_star - len(outs) + 1
            ws = [first + j for j in range(len(outs))]
            tot = 0.0
            for w, so in zip(ws, outs):
                fl, _ = self.fore_nll(so, pts)
                tot = tot + (w / sum(ws)) * fl
            if self._fore_w > 0:
                loss = loss + self._fore_w * tot
            stats["fore"] = fl.detach()
        return loss, stats

    # ---------------------------------------------------------------- probes
    @torch.no_grad()
    def loop_report(self, windows, device, batch=16, curve=(1, 2, 4, 8, 16)):
        """Alpha Star's probe (its "stops by itself" is now its own stop), plus where its stop lands: the share of words
        stopping after each pass."""
        out = super().loop_report(windows, device, batch, curve)
        was = self.training
        self.eval()
        xy = torch.as_tensor(windows, device=device)
        hist = torch.zeros(self.cfg.test_max, device=device)
        for i in range(0, len(xy), batch):
            x = xy[i:i + batch, :-1]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, used = self.think_test(self._read(x))
            hist += torch.bincount(used.long().reshape(-1) - 1, minlength=self.cfg.test_max)[:self.cfg.test_max].float()
        out["halt_hist"] = (hist / hist.sum()).tolist()
        self.train(was)
        return out
