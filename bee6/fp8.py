"""Bee6's FP8: the big matrix products of the layers in 8-bit numbers, everything delicate in bf16.

The B200 multiplies 8-bit floating-point numbers about twice as fast as bf16. Bee4 tried torchao's FP8 (Nov 7) and it was
slower (445k vs 475k tok/s) and moved the gradients. Bee6 does what the NanoGPT speedrun does instead: its own small
FP8 product, used only where it pays, with nothing else around it.

    linear(x, W, mode)   = x @ W.T, like F.linear

  mode 0   bf16 (the usual)
  mode 1   the forward product and the input's gradient in FP8; the weight's gradient in bf16 (default: the weight's
           gradient is the delicate one, it is what the optimizer reads)
  mode 2   all three products in FP8

Where: the attention's q/k/v (one fused product) and output, and the MLP's up and down. These are ~70% of the
arithmetic. Not: the norms, the word table and score head, the value tables, the gates, the Mixture-of-Depths router
and its helper (all bf16 or 32-bit).

How: each pass, each tensor gets one scale from its largest value (amax): inputs and weights in e4m3 (precise, range
448), gradients in e5m2 (coarse, range 57,344). The scales are worked out fresh every pass, so there is no scale
state to store, carry or grow: when Bee6 blooms, the new wider weights simply get their own scale on the next pass.

A product falls back to bf16 by itself when FP8 cannot run it (not a CUDA GPU with FP8, a size not divisible by 16).
`check()` runs once at the start of training on the GPU: the same batch through the model in bf16 and in FP8 must give
nearly the same loss and gradients, or Bee6 trains in bf16 and says why.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F

E4, E5 = torch.float8_e4m3fn, torch.float8_e5m2
E4_MAX, E5_MAX = 448.0, 57344.0
EMULATE = False          # tests: run the FP8 rounding on the CPU (quantize, multiply in 32-bit, rescale)
_HW = {}


def hardware(device) -> bool:
    """FP8 matrix products on this GPU (compute capability 8.9+: Ada, Hopper, Blackwell)."""
    if device.type != "cuda":
        return False
    key = device.index if device.index is not None else torch.cuda.current_device()
    if key not in _HW:
        _HW[key] = torch.cuda.get_device_capability(device) >= (8, 9) and hasattr(torch, "_scaled_mm")
    return _HW[key]


def _quant(t, dtype, mx):
    """t -> (FP8 tensor, the scale to multiply back by): one scale per tensor from its largest value."""
    amax = t.detach().abs().amax().float().clamp_min(1e-12)
    scale = mx / amax
    return (t.float() * scale).clamp(-mx, mx).to(dtype), scale.reciprocal()


def _mm(a, sa, b, sb, out_dtype):
    """a (M, K) @ b (N, K).T, both FP8 row-major, with their scales."""
    if a.is_cuda and not EMULATE:
        return torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=out_dtype)
    return ((a.float() @ b.float().t()) * (sa * sb)).to(out_dtype)


def _fits(*sizes):
    return all(s % 16 == 0 for s in sizes)


class _FP8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, mode):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        dt = torch.bfloat16 if x.is_cuda else x.dtype
        xq, xs = _quant(x2, E4, E4_MAX)
        wq, ws = _quant(w, E4, E4_MAX)
        y = _mm(xq, xs, wq, ws, dt)
        if mode >= 2:
            ctx.save_for_backward(xq, xs, wq, ws)
        else:
            ctx.save_for_backward(x2.to(dt), wq, ws)
        ctx.mode, ctx.shp, ctx.w_dtype = mode, shp, w.dtype
        return y.view(*shp[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, gy):
        g2 = gy.reshape(-1, gy.shape[-1])
        gq, gs = _quant(g2, E5, E5_MAX)
        if ctx.mode >= 2:
            xq, xs, wq, ws = ctx.saved_tensors
        else:
            x2, wq, ws = ctx.saved_tensors
        dt = x2.dtype if ctx.mode < 2 else (torch.bfloat16 if g2.is_cuda else g2.dtype)
        gx = _mm(gq, gs, wq.t().contiguous(), ws, dt)                               # dY @ W
        if ctx.mode >= 2:
            gw = _mm(gq.t().contiguous(), gs, xq.t().contiguous(), xs, torch.float32)   # dY.T @ X
        else:
            gw = g2.to(x2.dtype).t() @ x2                                            # bf16: the optimizer reads this one
        return gx.view(ctx.shp), gw.to(ctx.w_dtype), None


def linear(x, w, mode=0):
    """x @ w.T: in FP8 when mode > 0 and this GPU and these sizes allow it, else the usual F.linear (bf16 under autocast)."""
    if not mode or not (EMULATE or hardware(x.device)) or not _fits(x.shape[-1], w.shape[0]):
        return F.linear(x, w)
    if x.is_cuda:
        # the dtype F.linear would give here: bf16 under autocast (training), else x's own (writing, chat, evals)
        want = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
        x, w = x.to(torch.bfloat16), w.to(torch.bfloat16)
        if not _fits(x.numel() // x.shape[-1]):
            return F.linear(x, w).to(want)
        with torch.autocast("cuda", enabled=False):
            return _FP8Linear.apply(x, w, int(mode)).to(want)
    return _FP8Linear.apply(x, w, int(mode))


def cosine(a, b):
    a, b = a.detach().flatten().float(), b.detach().flatten().float()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12))


def check(model, device, say=print, T=512, B=4, loss_tol=0.02, grad_tol=0.9):
    """The startup check: one batch in bf16 and in FP8, same weights. Returns (ok, loss gap, gradient cosine).
    Run on a copy whose output layers are given small random weights: a fresh Bee6 starts with them at zero, and then
    most gradients are zero too and nothing would be checked."""
    model = copy.deepcopy(model)
    with torch.no_grad():
        for b in model.blocks:
            b.attn.out.weight.normal_(0, 0.02)
            (b.mlp.w_down if hasattr(b.mlp, "w_down") else b.mlp.down.weight).normal_(0, 0.02)   # (Bee6-MoE's experts)
    cfg = model.cfg
    mode = int(cfg.fp8)
    g = torch.Generator(device="cpu").manual_seed(0)
    V = cfg.vocab_size
    x = torch.randint(1, V, (B, T), generator=g).to(device)
    y = torch.roll(x, -1, 1)
    model.train()
    out = []
    for m in (0, mode):
        cfg.fp8 = m
        model.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, _ = model.losses(x, y)
        loss.backward()
        grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}
        out.append((float(loss.detach()), grads))
    del model
    (l0, g0), (l1, g1) = out
    gap = abs(l1 - l0) / max(1e-6, abs(l0))
    big = [n for n in g0 if g0[n].numel() >= 4096 and n in g1 and g0[n].norm() > 0]
    cos = sum(cosine(g0[n], g1[n]) for n in big) / max(1, len(big))
    finite = all(math.isfinite(float(t.sum())) for t in g1.values())
    ok = gap < loss_tol and cos > grad_tol and finite
    return ok, gap, cos
