"""Bloom (idea 5, `--set bloom=on`): Bee6 starts small and fast and grows, keeping exactly what it has learned.

The plan (João's Seed schedule, 20/20/20/40 of the tokens, at Bee6's sizes):

    4 layers of 512   (8 heads)    0-20%      about 4x cheaper a word
    6 layers of 768   (12 heads)   20-40%
    9 layers of 1,024 (16 heads)   40-60%
    12 layers of 1,024             60-100%    plain Bee6

Each growth is function-preserving: on the same words the grown model gives the same scores as before (the tests check
it to 1e-4). How, for each kind of growth:

  wider    The stream gets new numbers that start at 0. Every norm in Bee6 has no gain, so a wider stream with zeros
           added only changes the norm's size by c = sqrt(new width / old width); every product that reads a norm is
           divided by c and every product that writes into the stream is multiplied by c, so nothing moves. The word
           table's new columns start at 0 (it is also the score head). New heads and new MLP neurons get fresh random
           weights going in and zero weights going out, so they cannot change the answer yet but do get a gradient.
  deeper   New layers go in the middle (the first half of the old layers stays at the front, the second half at the
           end), so every U-net skip still joins the same two old layers, and the value tables stay on the first two
           and last two layers. A new layer has its output weights at zero (Bee5 starts every layer that way), so it is
           exactly a no-op until it learns. Its new U-net skip starts at weight 0 (Bee6 with bloom uses plain skip
           weights instead of Bee5's sigmoid ones, so 0 is allowed and still learns).
  routers  The Mixture-of-Depths router and its helper grow with the width like any product that reads a norm. Old
           routed layers stay routed (routing a trained full layer would change it); new layers in the routed pattern
           (every other layer) get a router. Key-offset layers the same way.
  FP8      Nothing to grow: FP8's scales are worked out fresh on every pass (bee6/fp8.py).
  back-out It stays on the layer it was on.

After a growth the trainer starts a new optimizer (the weights are new, bigger tensors), and the compiled model is
rebuilt for the new shapes.
"""
from __future__ import annotations

import copy
import math
from dataclasses import fields

import torch


def offset_pattern(layers):
    return {j for j in layers if j % 4 == 3}


def mod_pattern(layers, n, offsets):
    return {j for j in layers if j % 2 == 0 and 2 <= j < n - 1 and j not in offsets}


def _sizes(cfg, n, d):
    hd = cfg.attn_head_dim
    if d % hd:
        raise ValueError(f"bloom: width {d} is not a whole number of {hd}-wide heads")
    heads = d // hd
    rep = cfg.attn_heads // cfg.attn_kv_heads
    if heads % rep:
        raise ValueError(f"bloom: {heads} heads do not share key-value heads {rep} to 1")
    return dict(n_layer=n, d_model=d, attn_heads=heads, attn_kv_heads=heads // rep,
                mlp_hidden=cfg.mlp_hidden * d // cfg.d_model)


def stage_config(cfg, k=0):
    """Turn a config at its final size into the plan's first stage (in place)."""
    plan = cfg.plan()
    if not plan or plan[0][2] != 0:
        raise ValueError(f"bloom plan {cfg.bloom!r}: give stages like 4x512:0,...,12x1024:0.6 (the first at 0)")
    if (plan[-1][0], plan[-1][1]) != (cfg.n_layer, cfg.d_model):
        raise ValueError(f"bloom plan {cfg.bloom!r} must end at the model's size ({cfg.n_layer}x{cfg.d_model})")
    if any(b[2] <= a[2] or b[0] < a[0] or b[1] < a[1] for a, b in zip(plan, plan[1:])):
        raise ValueError(f"bloom plan {cfg.bloom!r}: stages must come later, and never shrink")
    n, d, _ = plan[k]
    had_off, had_mod = bool(cfg.offset_layers()), bool(cfg.mod_set())
    cfg.bloom_offset, cfg.bloom_mod = int(had_off), int(had_mod)
    for name, v in _sizes(cfg, n, d).items():
        setattr(cfg, name, v)
    off = offset_pattern(range(n)) if had_off else set()
    cfg.key_offset = ",".join(str(i) for i in sorted(off))
    cfg.mod_layers = ",".join(str(i) for i in sorted(mod_pattern(range(n), n, off))) if had_mod else ""
    cfg.backout_layer = max(0, round(2 * n / 3) - 1)
    cfg.bloom_stage = k
    return cfg


def stage_at(cfg, progress):
    plan = cfg.plan()
    return max(i for i, (_, _, at) in enumerate(plan) if at <= progress + 1e-12)


@torch.no_grad()
def bloom_next(model):
    """Grow the model one stage, in place."""
    from .model import Bee6
    c = model.cfg
    k = c.bloom_stage + 1
    n0, d0 = c.n_layer, c.d_model
    n1, d1, _ = c.plan()[k]
    nc = copy.copy(c)
    for name, v in _sizes(c, n1, d1).items():
        setattr(nc, name, v)
    front, shift = (n0 + 1) // 2, n1 - n0
    pos = [i if i < front else i + shift for i in range(n0)]
    new_layers = range(front, front + shift)
    off = {pos[i] for i in c.offset_layers()} | (offset_pattern(new_layers) if c.bloom_offset else set())
    mods = {pos[i] for i in c.mod_set()} | (mod_pattern(new_layers, n1, off) if c.bloom_mod else set())
    nc.key_offset = ",".join(str(i) for i in sorted(off))
    nc.mod_layers = ",".join(str(i) for i in sorted(mods))
    nc.backout_layer = pos[model.backout_at]
    nc.bloom_stage = k
    dev = model.embed.weight.device
    with torch.device(dev):
        new = Bee6(nc).to(dtype=model.embed.weight.dtype)
    s = math.sqrt(d1 / d0)          # the stream grows by s in its old numbers (see the top of this file)
    hd = c.attn_head_dim
    H0, KV0, M0 = c.attn_heads, c.attn_kv_heads, c.mlp_hidden

    new.embed.weight.zero_()
    new.embed.weight[:, :d0] = model.embed.weight / s
    for a, b in zip(new.ve, model.ve):
        a.weight.zero_()
        a.weight[:, :b.weight.shape[1]] = b.weight
    for i in range(n0):
        new.lam[pos[i]] = model.lam[i]
        new.mu[pos[i]] = model.mu[i]
    for j in range(len(model.skip_gate)):
        new.skip_w[j] = model.skip_w[j]
        new.skip_gate[j].weight.copy_(model.skip_gate[j].weight / s)
    new.skip_w[len(model.skip_gate):] = 0.0
    new.smear_lam.copy_(model.smear_lam)
    if model.smear_gate is not None:
        new.smear_gate.weight.copy_(model.smear_gate.weight * s)    # it reads the raw word vector, which shrank by s
    new.backout_lam.copy_(model.backout_lam)

    for i, o in enumerate(model.blocks):
        b = new.blocks[pos[i]]
        if b.routed != o.routed or b.attn.offset != o.attn.offset or (b.attn.ve_gate is None) != (o.attn.ve_gate is None):
            raise RuntimeError(f"bloom: layer {i} -> {pos[i]} changed kind")
        for name, rows in (("q", H0 * hd), ("k", KV0 * hd), ("v", KV0 * hd)):
            getattr(b.attn, name).weight[:rows, :d0] = getattr(o.attn, name).weight / s
        b.attn.out.weight.zero_()
        b.attn.out.weight[:d0, :H0 * hd] = o.attn.out.weight * s
        b.attn.qn.copy_(o.attn.qn)
        b.attn.kn.copy_(o.attn.kn)
        if o.attn.gate is not None:
            b.attn.gate.weight.zero_()
            b.attn.gate.weight[:H0] = o.attn.gate.weight / s
        if o.attn.ve_gate is not None:
            b.attn.ve_gate.weight.zero_()
            b.attn.ve_gate.weight[:KV0] = o.attn.ve_gate.weight / s
        parts = 2 if c.mlp == "swiglu" else 1
        up_new = b.mlp.up.weight.view(parts, -1, d1)
        up_new[:, :M0, :d0] = o.mlp.up.weight.view(parts, M0, d0) / s
        b.mlp.down.weight.zero_()
        b.mlp.down.weight[:d0, :M0] = o.mlp.down.weight * s
        if o.routed:
            b.router.weight[:, :d0] = o.router.weight / s
            b.pred[0].weight[:, :d0] = o.pred[0].weight / s
            b.pred[0].bias.copy_(o.pred[0].bias)
            b.pred[2].weight.copy_(o.pred[2].weight)
            b.pred[2].bias.copy_(o.pred[2].bias)

    for name in ("embed", "ve", "blocks", "lam", "mu", "skip_w", "skip_gate", "smear_lam", "smear_gate", "backout_lam"):
        setattr(model, name, getattr(new, name))
    model.backout_at = new.backout_at
    model.bloom_at.fill_(k)
    for f in fields(c):
        setattr(c, f.name, getattr(nc, f.name))
    for m in model.modules():
        if m is not model and hasattr(m, "cfg"):
            m.cfg = c
    model._compiled = None
    del new
    return model
