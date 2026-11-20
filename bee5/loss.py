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


def _capf(z, kind: str, softcap: float):
    """Capped scores minus a constant (so exp cannot overflow), and d(capped)/d(raw); fp32. The constant is the cap's top:
    with "sig" the capped scores live in (0, 23), with "tanh" in (-softcap, softcap), so no pass to find the largest."""
    if kind == "sig":
        s = torch.sigmoid((z + 5.0) / 7.5)
        return 23.0 * s - 23.0, (23.0 / 7.5) * s * (1.0 - s)
    t = torch.tanh(z / softcap)
    return softcap * t - softcap, 1.0 - t * t


def _slice_lean(hc, Wc, tc, wk, thc, tWc, kind: str, softcap: float, kd_w, n_valid, n_all: float, Vt: int):
    """_slice without the mentor, moving far fewer bytes (bee6/lean.py): the scores are read twice and written once.

    _slice stores the capped scores, their log-softmax and the gradient in 32-bit, each n x 49,152 (it reads or writes
    the scores' worth of memory about ten times). Here one kernel reads the 16-bit scores and sums each row's exp, and
    a second reads them again and writes the gradient straight in 16-bit, the real next words' entries included (a
    comparison with the column number, not a scatter). The same numbers as _slice up to 32-bit rounding order (tested)."""
    dt = hc.dtype
    V = Wc.shape[0]
    z = hc @ Wc.t()                                                  # (n, V), stays in the products' format
    c1, _ = _capf(z.float(), kind, softcap)
    lse = torch.log(torch.exp(c1).sum(1, keepdim=True))              # (n, 1): log sum exp of (capped - top)
    valid = (tc >= 0).to(torch.float32)                              # (n, K)
    tgt = tc.clamp_min(0)
    ct, _ = _capf(z.gather(1, tgt).float(), kind, softcap)           # the real next words' own scores (n, K)
    tok = lse - ct                                                   # each word's loss
    ce = (tok[:, 0] * valid[:, 0]).sum()
    wv = valid * wk[None, :]
    mce = (tok * wv).sum()
    c2, dz = _capf(z.to(torch.float32), kind, softcap)               # (written apart from c1 on purpose: two kernels that
    g = torch.exp(c2 - lse) * wv.sum(1, keepdim=True)                #  each read the scores, and no 32-bit copy between)
    col = torch.arange(V, device=hc.device)[None, :]
    for k in range(tc.shape[1]):                                     # minus what each real next word is owed
        g = g - wv[:, k:k + 1] * (col == tgt[:, k:k + 1])
    gb = (g * ((1.0 - kd_w) / n_valid) * dz).to(dt)
    kd = torch.zeros((), dtype=torch.float32, device=hc.device)
    return ce, mce, kd, gb @ Wc, (gb.t() @ hc).float()


def _slice_rows(hc, Wc, tc, wk, thc, tWc, kind: str, softcap: float, kd_w, n_valid, n_all: float, Vt: int):
    """_slice with the mentor's weight token by token (kd_w: (n, 1)), for --mentor-skip-world: a token of weight 0
    learns the real next word only, one of weight w the usual (1 - w) * CE + w * KL. Returns the weighted sums."""
    dt = hc.dtype
    z, dz = _cap((hc @ Wc.t()).float(), kind, softcap)
    lp = torch.log_softmax(z, dim=-1)
    valid = (tc >= 0).to(torch.float32)
    tgt = tc.clamp_min(0)
    tok = -lp.gather(1, tgt)
    ce = (tok[:, 0] * valid[:, 0]).sum()
    wv = valid * wk[None, :]
    mce = ((tok * wv).sum(1, keepdim=True) * (1.0 - kd_w)).sum()
    p = lp.exp()
    g = p * wv.sum(1, keepdim=True)
    g.scatter_add_(1, tgt, -wv)
    g *= (1.0 - kd_w) / n_valid
    tlp = torch.log_softmax((thc @ tWc.t()).float(), dim=-1)
    pt = tlp.exp()
    V = z.shape[1]
    if Vt == V:
        kdt = (pt * (tlp - lp)).sum(1, keepdim=True)
        g += (p - pt) * (kd_w / n_all)
    else:
        lps = torch.log_softmax(z[:, :Vt], dim=-1)
        kdt = (pt * (tlp - lps)).sum(1, keepdim=True)
        g[:, :Vt] += (lps.exp() - pt) * (kd_w / n_all)
    kd, kdw = kdt.sum(), (kdt * kd_w).sum()
    if dz is not None:
        g *= dz
    gb = g.to(dt)
    return ce, mce, kd, kdw, gb @ Wc, (gb.t() @ hc).float()


_compiled = None
_compiled_lean = None


def slice_fn(device, lean=False):
    global _compiled, _compiled_lean
    plain = _slice_lean if lean else _slice
    if device.type != "cuda" or os.environ.get("OAZY_LOSS_COMPILE", "1") == "0":
        return plain
    if lean:
        if _compiled_lean is None:
            _compiled_lean = torch.compile(_slice_lean, dynamic=False)
        return _compiled_lean
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


class _HeadLossRows(torch.autograd.Function):
    """_HeadLoss with the mentor's weight token by token (kd_w: (N, 1)); uncompiled, used only with --mentor-skip-world."""
    @staticmethod
    def forward(ctx, h, W, targets, wk, kind, softcap, th, tW, kd_w, chunk):
        N, D = h.shape
        V = W.shape[0]
        dt = _autocast_dtype(h)
        with torch.autocast(device_type=h.device.type, enabled=False):
            Wc = W.to(dt)
            n_valid = (targets[:, 0] >= 0).sum().clamp_min(1).to(torch.float32)
            tWc, Vt = tW.to(dt), int(tW.shape[0])
            kw = kd_w.to(device=h.device, dtype=torch.float32)
            gh = torch.empty_like(h)
            gW = torch.zeros(V, D, dtype=torch.float32, device=h.device)
            sums = torch.zeros(4, dtype=torch.float32, device=h.device)
            for i in range(0, N, chunk):
                ce, mce, kdv, kdw, ghc, gWc = _slice_rows(h[i:i + chunk].to(dt), Wc, targets[i:i + chunk], wk,
                                                          th[i:i + chunk].to(dt), tWc, kind, float(softcap),
                                                          kw[i:i + chunk], n_valid, float(N), Vt)
                sums += torch.stack([ce, mce, kdv, kdw])
                gh[i:i + chunk] = ghc.to(h.dtype)
                gW += gWc
        loss = sums[1] / n_valid + sums[3] / N
        ctx.save_for_backward(gh, gW)
        ctx.w_dtype = W.dtype
        ce, kdm = sums[0] / n_valid, sums[2] / N
        ctx.mark_non_differentiable(ce, kdm)
        return loss, ce, kdm

    @staticmethod
    def backward(ctx, g_loss, g_ce, g_kd):
        gh, gW = ctx.saved_tensors
        return (gh * g_loss.to(gh.dtype), (gW * g_loss).to(ctx.w_dtype)) + (None,) * 8


class _HeadLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, W, targets, wk, kind, softcap, th, tW, kd_w, chunk, lean=False):
        global _compiled, _compiled_lean
        N, D = h.shape
        V = W.shape[0]
        dt = _autocast_dtype(h)
        with torch.autocast(device_type=h.device.type, enabled=False):
            Wc = W.to(dt)
            n_valid = (targets[:, 0] >= 0).sum().clamp_min(1).to(torch.float32)
            kd = th is not None and kd_w > 0
            tWc, Vt = (tW.to(dt), int(tW.shape[0])) if kd else (None, V)
            kw = torch.tensor(float(kd_w), dtype=torch.float32, device=h.device)
            lean = bool(lean) and not kd and kind in ("sig", "tanh")     # the lean slice has no mentor and needs a cap
            base = _slice_lean if lean else _slice
            f = slice_fn(h.device, lean)
            gh = torch.empty_like(h)
            gW = torch.zeros(V, D, dtype=torch.float32, device=h.device)
            sums = torch.zeros(3, dtype=torch.float32, device=h.device)
            for i in range(0, N, chunk):
                args = (h[i:i + chunk].to(dt), Wc, targets[i:i + chunk], wk, th[i:i + chunk].to(dt) if kd else None, tWc,
                        kind, float(softcap), kw, n_valid, float(N), Vt)
                try:
                    ce, mce, kdv, ghc, gWc = f(*args)
                except Exception as e:
                    if f is base:
                        raise
                    print(f"[bee5 loss] torch.compile failed ({type(e).__name__}: {str(e)[:160]}); the loss runs uncompiled", flush=True)
                    f = base
                    if lean:
                        _compiled_lean = base
                    else:
                        _compiled = base
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
        return (gh * g_loss.to(gh.dtype), (gW * g_loss).to(ctx.w_dtype)) + (None,) * 9


def _prep(h, targets, weights, teacher_h, teacher_W, kd_weight):
    D = h.shape[-1]
    K = targets.shape[-1]
    h2, t2 = h.reshape(-1, D), targets.reshape(-1, K)
    th2 = teacher_h.reshape(-1, teacher_h.shape[-1]) if teacher_h is not None else None
    rows = torch.is_tensor(kd_weight)               # --mentor-skip-world: the mentor's weight token by token, (N, 1)
    if th2 is None or (not rows and kd_weight <= 0) or (rows and not bool((kd_weight > 0).any())):
        th2, teacher_W, kd_weight, rows = None, None, 0.0, False
    wk = torch.tensor([float(w) for w in weights], dtype=torch.float32, device=h.device)
    return h2, t2, wk, th2, teacher_W, (kd_weight.reshape(-1, 1).to(device=h.device, dtype=torch.float32) if rows else float(kd_weight))


def head_loss(h, W, targets, weights, cfg, teacher_h=None, teacher_W=None, kd_weight=0.0, chunk=1024):
    h2, t2, wk, th2, tW, kw = _prep(h, targets, weights, teacher_h, teacher_W, kd_weight)
    if th2 is not None and th2.shape[0] != h2.shape[0]:
        raise ValueError(f"mentor read {th2.shape[0]} tokens, student {h2.shape[0]}")
    if torch.is_tensor(kw):
        return _HeadLossRows.apply(h2, W, t2, wk, cfg.cap, float(cfg.softcap), th2, tW, kw, int(chunk))
    return _HeadLoss.apply(h2, W, t2, wk, cfg.cap, float(cfg.softcap), th2, tW, kw, int(chunk), bool(getattr(cfg, "lean", 0)))


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
    if torch.is_tensor(kw):                         # --mentor-skip-world: the mentor's weight token by token
        kdt = F.kl_div(s, t, log_target=True, reduction="none").sum(1)
        kw1 = kw[:, 0]
        loss = ((1 - kw1) * (tok * wk[None]).sum(1)).sum() / n_valid + (kw1 * kdt).sum() / h2.shape[0]
        return loss, ce.detach(), (kdt.sum() / h2.shape[0]).detach()
    kd = F.kl_div(s, t, log_target=True, reduction="sum") / h2.shape[0]
    return (1 - kw) * mce + kw * kd, ce.detach(), kd.detach()
