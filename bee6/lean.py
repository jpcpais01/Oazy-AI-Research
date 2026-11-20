"""Lean: the same layers with far less memory moved. Nothing here changes what the model computes.

Why. Bee-135 on a B300 uses about a fifth of the chip's arithmetic, and an H200 trains it at 0.56x the B300's speed:
almost exactly the ratio of the two cards' memory speed (4.8 and 8 TB/s), not of their arithmetic. A narrow model (576
wide) is not waiting for its matrix products, it is waiting for memory: every small operation between two products (a
norm, a Canon mix, a residual add, a cast) reads and writes the whole batch once more. `python3 -m bee6.traffic` counts
those bytes kernel by kernel, from the compiler's own plan (no GPU needed): per word and layer the small operations
moved two and a half times the bytes of the products and attention themselves.

Where the bytes went, and what `lean=1` does about each:

  the stream is 32-bit        the word vectors come out of the table in 32-bit, so the stream, every norm, every Canon
                              mix and every residual add ran in 32-bit, and each product first made a 16-bit copy.
                              lean: a norm (or norm + Canon) is worked out in 32-bit inside one kernel and written
                              once, already in the 16-bit form the product reads. `stream16=1` goes further and keeps
                              the stream itself in bf16 (what nanochat and the NanoGPT speedrun do): every add and mix
                              on the stream then moves half the bytes. That one changes the rounding, so it is its own
                              switch with its own check at the start of a run (check(), below).
  norm, then pad, then mix    a Canon layer read the normed stream from a padded copy: three kernels and two 32-bit
                              temporaries per Canon, 48 Canons. lean: one kernel reads the stream (the word itself and
                              the 3 before, each scaled by its own norm) and writes the mix.
  queries and keys            a norm, a gain and a rotation that cuts each head in two halves and joins them again:
                              five kernels in, a dozen back. lean: one kernel each way (_QKRot).
  the way back                autograd's own way back through norm and Canon was a dozen kernels. lean: the gradient
                              is written by hand below (three kernels: the Canon weights' gradient, one mean per word,
                              the stream's gradient), and only the stream itself is kept for it.
  the loss step               every word's scores over the 49,152 word pieces were read and written twelve times
                              (bee5/loss.py, _slice). lean: three times (_slice_lean): the scores once, one pass for
                              the loss, one pass that writes the gradient.

The numbers are the same as the plain code's up to 32-bit rounding order (bee6/lean_tests.py checks values and every
gradient); with stream16 they differ by bf16 rounding of the stream, which the startup check measures on a real batch.
Checkpoints are the same either way: lean adds no weights and changes none.

Measured on the plan, piece by piece (Bee-135, per word and layer, U = one word's vector in bf16): the small operations
move 136 U as it is, 127 lean, 91 lean with the bf16 stream; the loss step 1,024 U, lean 256. The way back written by
hand is worth 15 U a layer against autograd's way back through the same one-kernel norm + Canon (132 against 149, and
95 against 109 with the bf16 stream). Tried and left out (under 1% of the bytes, for more code): the two gates' scores
taken from the same product as q, k, v.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F

EPS = torch.finfo(torch.float32).eps      # F.rms_norm's own default on a 32-bit stream; kept when the stream is bf16


def product_dtype(x):
    """The number format a matrix product would read x in here: bf16 under autocast (training on the GPU), else x's own."""
    try:
        on = torch.is_autocast_enabled(x.device.type)
    except (TypeError, RuntimeError):
        on = x.is_cuda and torch.is_autocast_enabled()
    if not on:
        return x.dtype
    return torch.get_autocast_dtype(x.device.type) if hasattr(torch, "get_autocast_dtype") else torch.bfloat16


def _rowmean(v, rows):
    """A word's mean over its numbers. rows > 0: as sums of `rows` numbers, then their sum (the same mean: on a B200
    the compiler's kernel for one sum over 576 numbers takes twice as long as for 9 sums of 64, bee6/redbench.py)."""
    D = v.shape[-1]
    if rows and D > rows and D % rows == 0:
        return v.unflatten(-1, (D // rows, rows)).sum(-1).sum(-1, keepdim=True) / D
    return v.mean(-1, keepdim=True)


def _make(rows, back):
    """The norm and the norm + Canon (forward and the hand-written way back), for one way of taking a word's mean.
    rows: the piece size for the forward means and the plain norm's way back. back: the same on Canon's way back too
    (kept apart: there the compiler's plan stores the mixed gradient to do it, so it is its own switch to time)."""
    rb = rows if back else 0
    class _Norm(torch.autograd.Function):
        """y = x * rsqrt(mean(x^2) + eps) over the last dim: worked out in 32-bit, written once in `dt`."""

        @staticmethod
        def forward(ctx, x, dt):
            xf = x.float()
            s = torch.rsqrt(_rowmean(xf.square(), rows) + EPS)
            ctx.save_for_backward(x, s)
            return (xf * s).to(dt)

        @staticmethod
        def backward(ctx, g):
            x, s = ctx.saved_tensors
            xf, gf = x.float(), g.float()
            m = _rowmean(gf * xf, rows)                                  # one number per word
            return (s * (gf - xf * (s * s * m))).to(x.dtype), None


    class _NormCanon(torch.autograd.Function):
        """Canon(norm(x)) in one go: with n = norm(x),  c_t = n_t + sum_j w_j * n_(t-j)   (j = 0 .. k-1, causal).

        x (B, T, D) the stream; w (k, D) the Canon weights (32-bit). Only x and one norm scale per word are kept for the
        way back; the normed stream and the mix are never stored in 32-bit."""

        @staticmethod
        def forward(ctx, x, w, dt):
            k, T = w.shape[0], x.shape[1]
            xf = x.float()
            s = torch.rsqrt(_rowmean(xf.square(), rows) + EPS)
            n = xf * s
            c = n * (1.0 + w[0])
            for j in range(1, min(k, T)):                              # the word j back, where there is one
                c = c + F.pad(n[:, :T - j] * w[j], (0, 0, j, 0))
            ctx.save_for_backward(x, s, w)
            return c.to(dt)

        @staticmethod
        def backward(ctx, g):
            x, s, w = ctx.saved_tensors
            k, T = w.shape[0], x.shape[1]
            xf, gf = x.float(), g.float()
            n = s * xf           # (written s * xf, not xf * s as on the way in: see "what is kept" at the end of this file)
            gw = [(gf * n).sum((0, 1))]                                # d/dw_j = sum over words of g_t * n_(t-j)
            ga = gf * (1.0 + w[0])                                     # d/dn_t = g_t + sum_j w_j * g_(t+j), written twice
            gb = (1.0 + w[0]) * gf                                     # (ga feeds the mean, gb the result: see the end)
            for j in range(1, k):
                if j < T:
                    gw.append((gf * F.pad(n[:, :T - j], (0, 0, j, 0))).sum((0, 1)))
                    ga = ga + F.pad(gf[:, j:] * w[j], (0, 0, 0, j))
                    gb = gb + F.pad(w[j] * gf[:, j:], (0, 0, 0, j))
                else:
                    gw.append(torch.zeros_like(gw[0]))
            m = _rowmean(ga * xf, rb)                                    # the norm's way back, as in _Norm
            gx = s * (gb - xf * (s * s * m))
            return gx.to(x.dtype), torch.stack(gw).to(w.dtype), None

    return _Norm, _NormCanon


ROWS = (0, 4, 8, 16, 32, 64, 128)                    # the piece sizes on offer (made once, here: none is made mid-compile)
_FNS = {(r, b): _make(r, b) for r in ROWS for b in (False, True) if r or not b}
_Norm, _NormCanon = _FNS[0, False]


TUNE = ("tune-kernels", "max-autotune-no-cudagraphs", "max-autotune")


def compile_kw(mode):
    """torch.compile's arguments for a compile_mode. "tune-kernels": the compiler times a few layouts of each of its own
    kernels on this GPU and keeps the fastest (the matrix products and attention are left as they are; the first compile
    takes longer, the choices are kept on disk for the next run). Anything else is torch's own mode of that name."""
    if mode in ("", "default", None):
        return {}
    if mode == "tune-kernels":
        return {"options": {"max_autotune_pointwise": True}}
    return {"mode": mode}


def swap(x):
    """The two halves of the last dim changed over: each number's partner in the rotation (half a head away)."""
    return x.unflatten(-1, (2, x.shape[-1] // 2)).flip(-2).flatten(-2)


class _QKRot(torch.autograd.Function):
    """rotary(gain * rms_norm(z)) for queries or keys: z (B, T, heads, hd) straight from the q/k/v product,
    gain (hd,) the QK-norm gains, C = (cos, cos) and S = (sin, -sin) at the head's full width (broadcast to z).

    The plain code is a norm, a gain, and a rotation that joins two halves: five kernels in, a dozen on the way back.
    Here: one mean per head and one kernel that writes the rotated result; on the way back the rotation is undone
    (the same kernel with -S), and the norm's gradient is _Norm's. Rounded where F.rms_norm rounds (the normed value,
    to z's own format), with F.rms_norm's own eps for that format."""

    @staticmethod
    def forward(ctx, z, gain, C, S):
        dt = z.dtype
        zf, zs = z.float(), swap(z).float()
        r = torch.rsqrt(zf.square().mean(-1, keepdim=True) + torch.finfo(dt).eps)
        u, us = (zf * r).to(dt).float(), (zs * r).to(dt).float()
        out = u * (gain * C) + us * (swap(gain) * S)
        ctx.save_for_backward(z, r, gain, C, S)
        return out.to(dt)

    @staticmethod
    def backward(ctx, g):
        z, r, gain, C, S = ctx.saved_tensors
        dt = z.dtype
        zf, gf, gs = z.float(), g.float(), swap(g).float()
        u = (r * zf).to(dt).float()
        g1 = gf * C - gs * S                                       # the rotation undone, written three times over
        g2 = C * gf - S * gs                                       # (one copy per kernel that reads it: see the end)
        g3 = gf * C + (-S) * gs
        ggain = (g1 * u).sum(tuple(range(g.dim() - 1)))
        m = ((g2 * gain) * zf).mean(-1, keepdim=True)
        gz = r * (g3 * gain - zf * (r * r * m))
        return gz.to(dt), ggain.to(gain.dtype), None, None


def qk_rot(z, gain, cos2, sin2):
    """Queries or keys ready for attention: rotary(gain * rms_norm(z)). cos2, sin2 broadcast to z, full head width."""
    return _QKRot.apply(z, gain, cos2, sin2)


def norm(x, dt=None, rows=0):
    """F.rms_norm(x) over the last dim (32-bit arithmetic, 32-bit eps), written in `dt` (default: x's own format).
    rows: the mean taken in pieces of this many numbers (_rowmean)."""
    return _FNS[rows, False][0].apply(x, dt or x.dtype)


def norm_canon(x, w, dt=None, rows=0, back=False):
    """Canon(norm(x)) with Canon weights w (k, D), written in `dt` (default: x's own format). rows, back: see _make."""
    return _FNS[rows, bool(back and rows)][1].apply(x, w, dt or x.dtype)


def stream_dtype(cfg, x):
    """The stream's format: bf16 when stream16 is on and products run in bf16 here (training on the GPU), else x's own."""
    if getattr(cfg, "stream16", 0):
        dt = product_dtype(x)
        if dt == torch.bfloat16:
            return dt
    return x.dtype


def compare(model, device, base, new, T=512, B=4):
    """One batch with the config's switches set as `base`, then as `new` (dicts), same weights -> (loss gap, gradient
    cosine, all finite and the same weights reached). As the FP8 check: run on a copy whose output layers are given
    small random weights (a fresh model starts with them at zero, and then most gradients are zero too)."""
    from .fp8 import cosine
    model = copy.deepcopy(model)
    with torch.no_grad():
        for b in model.blocks:
            b.attn.out.weight.normal_(0, 0.02)
            (b.mlp.w_down if hasattr(b.mlp, "w_down") else b.mlp.down.weight).normal_(0, 0.02)
    cfg = model.cfg
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randint(1, cfg.vocab_size, (B, T), generator=g).to(device)
    y = torch.roll(x, -1, 1)
    model.train()
    out = []
    for sets in (base, new):
        for k, v in sets.items():
            setattr(cfg, k, v)
        model.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, _ = model.losses(x, y)
        loss.backward()
        out.append((float(loss.detach()), {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}))
    del model
    (l0, g0), (l1, g1) = out
    gap = abs(l1 - l0) / max(1e-6, abs(l0))
    big = [n for n in g0 if g0[n].numel() >= 4096 and n in g1 and g0[n].norm() > 0]
    cos = sum(cosine(g0[n], g1[n]) for n in big) / max(1, len(big))
    finite = math.isfinite(l1) and all(math.isfinite(float(t.sum())) for t in g1.values())
    return gap, cos, finite and set(g0) == set(g1)


def check(model, device, say=print, T=512, B=4, loss_tol=0.005, grad_tol=0.99):
    """The startup check: one batch the plain way and the lean way (with the bf16 stream, if it is on), same weights
    -> (ok, loss gap, gradient cosine).
    For scale: on Bee-135's own shape in bf16, lean is 0.9999 alike (bf16 itself against 32-bit is 0.9999), lean with
    the bf16 stream 0.9998; FP8 was 0.95."""
    cfg = model.cfg
    gap, cos, sound = compare(model, device, {"lean": 0, "stream16": 0}, {"lean": int(cfg.lean), "stream16": int(cfg.stream16)}, T, B)
    return gap < loss_tol and cos > grad_tol and sound, gap, cos


# What is kept for the way back, and why some lines above are written twice.
#
# These functions are compiled with the rest of the layers (torch.compile), and two habits of the compiler decide how
# many bytes the way back moves:
#
#   1. If the way in and the way back contain the very same expression, it is worked out once, on the way in, and
#      stored for the way back. For n = xf * s that would be the whole normed stream in 32-bit, per norm: exactly the
#      copy lean exists to avoid. So the way back writes it as s * xf: the same numbers, but a different expression,
#      which the compiler then works out again inside the kernel that needs it (from x and s, which are kept anyway).
#   2. A value read by several kernels on the way back is written to memory once and read by each. Written twice with
#      its operands the other way round (ga and gb; g1, g2 and g3), each kernel works out its own copy in passing and
#      nothing is stored.
#
# Both are about bytes, never about values: bee6/lean_tests.py checks the values, and `python3 -m bee6.traffic` shows
# the plan (if a later compiler changes these habits, the bytes are counted there before any GPU is rented).
