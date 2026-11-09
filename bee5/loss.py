"""Bee5's last step in slices (Bee4's oazy/lmloss.py, extended): the scores against the word table, the capped
scores, the next-word loss AND the early lessons on the 2nd and 3rd next words, and the mentor's KL, with the
gradient pushed back slice by slice so the 49,152 scores of every token are never stored.

    loss, ce, kd = head_loss(h, W, targets, weights, cfg, teacher_h, teacher_W, kd_weight, chunk)

  targets  (..., K): column k = the (k+1)-th next word (-1 = not scored). K = 1 is the plain next-word lesson.
  weights  K floats: how much each column counts (column 0 is always 1; speedrun record 53's multi-word lesson)
  cfg      the model's config: cfg.cap = "sig" (23 * sigmoid((s + 5) / 7.5)), "tanh" (cfg.softcap) or "none"

The loss is (1 - w) * sum_k weights[k] * CE_k / (scored next words) + w * KL(mentor || student); the ce returned is
the plain next-word loss alone (what the logs and the held-out loss show). Several next words cost almost nothing:
the scores and their softmax are shared, each extra word is one gather and one scatter.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F


def _cap(z, kind: str, softcap: float):
    """Capped scores and d(capped)/d(raw), fp32."""
    if kind == "sig":
        s = torch.sigmoid((z + 5.0) / 7.5)
        return 23.0 * s, (23.0 / 7.5) * s * (1.0 - s)
    if kind == "tanh":
        t = torch.tanh(z / softcap)
        return softcap * t, 1.0 - t * t
    return z, None


def _slice(hc, Wc, tc, wk, thc, tWc, kind: str, softcap: float, kd_w, n_valid, n_all: float, Vt: int):
    dt = hc.dtype
    z, dz = _cap((hc @ Wc.t()).float(), kind, softcap)                # (n, V)
    lp = torch.log_softmax(z, dim=-1)
    valid = (tc >= 0).to(torch.float32)                              # (n, K)
    tgt = tc.clamp_min(0)
    tok = -lp.gather(1, tgt)                                         # (n, K) each word's loss
    ce = (tok[:, 0] * valid[:, 0]).sum()
    wv = valid * wk[None, :]                                         # (n, K)
    mce = (tok * wv).sum()
    p = lp.exp()
    g = p * wv.sum(1, keepdim=True)
    g.scatter_add_(1, tgt, -wv)
    g *= (1.0 - kd_w) / n_valid
    if thc is not None:
        tlp = torch.log_softmax((thc @ tWc.t()).float(), dim=-1)
        pt = tlp.exp()
        V = z.shape[1]
        if Vt == V:
            kd = (pt * (tlp - lp)).sum()
            g += (p - pt) * (kd_w / n_all)
        else:
            lps = torch.log_softmax(z[:, :Vt], dim=-1)
            kd = (pt * (tlp - lps)).sum()
            g[:, :Vt] += (lps.exp() - pt) * (kd_w / n_all)
    else:
        kd = torch.zeros((), dtype=torch.float32, device=hc.device)
    if dz is not None:
        g *= dz
    gb = g.to(dt)
    return ce, mce, kd, gb @ Wc, (gb.t() @ hc).float()


_compiled = None


def slice_fn(device):
    global _compiled
    if device.type != "cuda" or os.environ.get("OAZY_LOSS_COMPILE", "1") == "0":
        return _slice
    if _compiled is None:
        _compiled = torch.compile(_slice, dynamic=False)
    return _compiled


def _autocast_dtype(h):
    try:
        on = torch.is_autocast_enabled(h.device.type)
    except TypeError:
        on = h.device.type == "cuda" and torch.is_autocast_enabled()
    if h.dtype == torch.float32 and on:
        return torch.get_autocast_dtype(h.device.type) if hasattr(torch, "get_autocast_dtype") else torch.bfloat16
    return h.dtype


class _HeadLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, W, targets, wk, kind, softcap, th, tW, kd_w, chunk):
        global _compiled
        N, D = h.shape
        V = W.shape[0]
        dt = _autocast_dtype(h)
        with torch.autocast(device_type=h.device.type, enabled=False):
            Wc = W.to(dt)
            n_valid = (targets[:, 0] >= 0).sum().clamp_min(1).to(torch.float32)
            kd = th is not None and kd_w > 0
            tWc, Vt = (tW.to(dt), int(tW.shape[0])) if kd else (None, V)
            kw = torch.tensor(float(kd_w), dtype=torch.float32, device=h.device)
            f = slice_fn(h.device)
            gh = torch.empty_like(h)
            gW = torch.zeros(V, D, dtype=torch.float32, device=h.device)
            sums = torch.zeros(3, dtype=torch.float32, device=h.device)
            for i in range(0, N, chunk):
                args = (h[i:i + chunk].to(dt), Wc, targets[i:i + chunk], wk, th[i:i + chunk].to(dt) if kd else None, tWc,
                        kind, float(softcap), kw, n_valid, float(N), Vt)
                try:
                    ce, mce, kdv, ghc, gWc = f(*args)
                except Exception as e:
                    if f is _slice:
                        raise
                    print(f"[bee5 loss] torch.compile failed ({type(e).__name__}: {str(e)[:160]}); the loss runs uncompiled", flush=True)
                    f = _compiled = _slice
                    ce, mce, kdv, ghc, gWc = f(*args)
                sums += torch.stack([ce, mce, kdv])
                gh[i:i + chunk] = ghc.to(h.dtype)
                gW += gWc
        loss = (1.0 - kd_w) * sums[1] / n_valid + kd_w * sums[2] / N
        ctx.save_for_backward(gh, gW)
        ctx.w_dtype = W.dtype
        ce, kdm = sums[0] / n_valid, sums[2] / N
        ctx.mark_non_differentiable(ce, kdm)
        return loss, ce, kdm

    @staticmethod
    def backward(ctx, g_loss, g_ce, g_kd):
        gh, gW = ctx.saved_tensors
        return (gh * g_loss.to(gh.dtype), (gW * g_loss).to(ctx.w_dtype)) + (None,) * 8


def _prep(h, targets, weights, teacher_h, teacher_W, kd_weight):
    D = h.shape[-1]
    K = targets.shape[-1]
    h2, t2 = h.reshape(-1, D), targets.reshape(-1, K)
    th2 = teacher_h.reshape(-1, teacher_h.shape[-1]) if teacher_h is not None else None
    if th2 is None or kd_weight <= 0:
        th2, teacher_W, kd_weight = None, None, 0.0
    wk = torch.tensor([float(w) for w in weights], dtype=torch.float32, device=h.device)
    return h2, t2, wk, th2, teacher_W, float(kd_weight)


def head_loss(h, W, targets, weights, cfg, teacher_h=None, teacher_W=None, kd_weight=0.0, chunk=1024):
    h2, t2, wk, th2, tW, kw = _prep(h, targets, weights, teacher_h, teacher_W, kd_weight)
    if th2 is not None and th2.shape[0] != h2.shape[0]:
        raise ValueError(f"mentor read {th2.shape[0]} tokens, student {h2.shape[0]}")
    return _HeadLoss.apply(h2, W, t2, wk, cfg.cap, float(cfg.softcap), th2, tW, kw, int(chunk))


def plain_loss(h, W, targets, weights, cfg, teacher_h=None, teacher_W=None, kd_weight=0.0, chunk=None):
    """The same numbers the slow way (full scores in 32-bit), for the tests and --loss plain."""
    from .model import cap
    h2, t2, wk, th2, tW, kw = _prep(h, targets, weights, teacher_h, teacher_W, kd_weight)
    z = cap(F.linear(h2, W.to(h2.dtype)).float(), cfg)
    lp = torch.log_softmax(z, -1)
    valid = t2 >= 0
    tok = -lp.gather(1, t2.clamp_min(0)) * valid
    n_valid = valid[:, 0].sum().clamp_min(1)
    ce = tok[:, 0].sum() / n_valid
    mce = (tok * wk[None]).sum() / n_valid
    if th2 is None:
        return mce, ce.detach(), torch.zeros((), device=h.device)
    Vt = tW.shape[0]
    t = torch.log_softmax(F.linear(th2, tW.to(th2.dtype)).float(), -1)
    s = lp if Vt == z.shape[1] else torch.log_softmax(z[:, :Vt], -1)
    kd = F.kl_div(s, t, log_target=True, reduction="sum") / h2.shape[0]
    return (1 - kw) * mce + kw * kd, ce.detach(), kd.detach()
