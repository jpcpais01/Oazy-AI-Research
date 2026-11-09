"""Bee5's optimizer: NorMuon for the weight grids, Adam for the tables, the gates and the learned scalars.

NorMuon is Muon (oazy/muon.py) with the four refinements the NanoGPT speedrun kept, records 38-50:

  Polar Express     a better recipe for "straightening" the update (Amsel et al. 2025): five steps, each with its own
                    coefficients, instead of Newton-Schulz's one fixed quintic; closer to the exact answer, same cost
  per-neuron step   after straightening, each neuron's (row's or column's) update is divided by a running average of
                    its own size, like Adam does per number but per neuron (NorMuon, Li et al. 2025). The total size of
                    the update stays what Muon gives, so Muon's learning rate carries over
  cautious decay    weight decay only where the update already pulls a weight towards zero (Chen et al. 2025, record 43)
  momentum warm-up  0.85 -> 0.95 over the first 300 steps (record 9)

Muon's learning rate, its weight decay (on AdamW's scale, as for Bee4) and its shape scaling (tall grids a little
faster) are Bee4's. Adam's groups:

  word table        Bee4's rate, betas (0.8, 0.95) (speedrun), no decay
  value tables      10 x the rate (they start at zero and are read only where their word appears), no decay
  scalars           10 x the rate, betas (0.9, 0.99): the stream / x0 / skip / smear / back-out weights
  gates, QK gains   the rate, betas (0.9, 0.99), no decay

The speedrun's own multipliers (75 x for value tables, weight decay tied to the square of the rate) are tuned for
runs of ~5,000 steps; Bee5 runs ~19,000, so the long-run settings that kept Bee4 stable stay, only the form changes.
"""
from __future__ import annotations

import torch

from oazy.muon import Combined

POLAR = [(8.156554524902461, -22.48329292557795, 15.878769915207462),
         (4.042929935166739, -2.808917465908714, 0.5000178451051316),
         (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
         (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
         (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)]


def polar_express(G: torch.Tensor) -> torch.Tensor:
    """G (..., m, n) -> about U V^T (each grid on its own). bf16 on the GPU, fp32 elsewhere."""
    X = G.bfloat16() if G.is_cuda else G.float()
    tall = X.size(-2) > X.size(-1)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)
    for a, b, c in POLAR:
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    if tall:
        X = X.mT
    return X


def neuron_norm(v: torch.Tensor, second: torch.Tensor, beta2: float, dim: int) -> torch.Tensor:
    """NorMuon: divide each neuron's update by its running size, then rescale to the straightened update's total size.
    v (S, m, n) float32; second (S, m, 1) or (S, 1, n) is updated in place."""
    mean = v.square().mean(dim=dim, keepdim=True)
    total = v.square().sum(dim=(-2, -1), keepdim=True).sqrt()
    second.lerp_(mean, 1 - beta2)
    u = v * second.clamp_min(1e-10).rsqrt()
    return u * (total / u.square().sum(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-10))


class NorMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.03, momentum=0.95, momentum_start=0.85, momentum_warm=300, beta2=0.95,
                 weight_decay=0.1, cautious=True, normuon=True):
        super().__init__(params, dict(lr=lr, momentum=momentum, momentum_start=momentum_start, momentum_warm=momentum_warm,
                                      beta2=beta2, weight_decay=weight_decay, cautious=cautious, normuon=normuon, steps=0))

    batched = True

    @torch.no_grad()
    def step(self, closure=None):
        todo = {}
        for g in self.param_groups:
            g["steps"] = g.get("steps", 0) + 1
            warm = min(1.0, (g["steps"] - 1) / max(1, g["momentum_warm"]))
            m = g["momentum_start"] + (g["momentum"] - g["momentum_start"]) * warm
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "momentum" not in st:
                    st["momentum"] = torch.zeros_like(p.grad)
                buf = st["momentum"]
                buf.lerp_(p.grad, 1 - m)
                u = p.grad.lerp(buf, m)                       # Nesterov
                key = (tuple(p.shape), p.device, id(g)) if self.batched else (id(p),)
                todo.setdefault(key, []).append((g, p, u))
        for items in todo.values():
            g = items[0][0]
            X = polar_express(torch.stack([u for _, _, u in items])).float()     # (S, m, n)
            rows, cols = X.shape[-2], X.shape[-1]
            if g["normuon"]:
                dim = -1 if rows >= cols else -2
                shape = (rows, 1) if dim == -1 else (1, cols)
                for _, p, _ in items:
                    st = self.state[p]
                    if "second" not in st:
                        st["second"] = torch.zeros(shape, dtype=torch.float32, device=p.device)
                sec = torch.stack([self.state[p]["second"] for _, p, _ in items])
                X = neuron_norm(X, sec, g["beta2"], dim)
                for (_, p, _), s in zip(items, sec.unbind(0)):
                    self.state[p]["second"].copy_(s)
            X = X * max(1.0, rows / cols) ** 0.5
            lr = g["lr"]
            wd = lr / g.get("lr_scale", 1.0) * g["weight_decay"]     # AdamW's scale (Bee4): the group's lr over lr_scale
            for (_, p, _), x in zip(items, X.unbind(0)):
                x = x.to(p.dtype)
                if wd:
                    if g["cautious"]:
                        p.sub_(p * (x * p >= 0) * wd)
                    else:
                        p.mul_(1 - wd)
                p.add_(x, alpha=-lr)


def groups(model):
    """(muon grids, {adam group name: params}) by name."""
    muon, adam = [], {"table": [], "ve": [], "scalars": [], "gates": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("blocks.") and p.dim() == 2 and n.endswith(("attn.q.weight", "attn.k.weight", "attn.v.weight",
                                                                     "attn.out.weight", "mlp.up.weight", "mlp.down.weight")):
            muon.append((n, p))
        elif n == "embed.weight":
            adam["table"].append(p)
        elif n.startswith("ve."):
            adam["ve"].append(p)
        elif n in ("lam", "mu", "skip_lam", "smear_lam", "backout_lam"):
            adam["scalars"].append(p)
        else:
            adam["gates"].append(p)
    return muon, adam


ADAM = {"table": dict(lr_scale=1.0, betas=(0.8, 0.95)), "ve": dict(lr_scale=10.0, betas=(0.8, 0.95)),
        "scalars": dict(lr_scale=10.0, betas=(0.9, 0.99)), "gates": dict(lr_scale=1.0, betas=(0.9, 0.99))}


def make(model, lr, device, muon_lr=0.03, muon_wd=0.1, kind="muon"):
    muon, adam = groups(model)
    gs = [dict(params=ps, weight_decay=0.0, name=k, lr=lr * ADAM[k]["lr_scale"], **ADAM[k]) for k, ps in adam.items() if ps]
    if kind != "muon":                     # --opt adamw: the grids go to Adam too (a comparison, not the recipe)
        gs.append(dict(params=[p for _, p in muon], weight_decay=0.1, name="grids", lr=lr, lr_scale=1.0, betas=(0.9, 0.95)))
        muon = []
    kw = dict(lr=lr, eps=1e-10)
    if device.type == "cuda":
        kw["fused"] = True
    opt = torch.optim.AdamW(gs, **kw)
    if not muon:
        return Combined(opt)
    mu = NorMuon([dict(params=[p for _, p in muon], lr_scale=muon_lr / lr)], lr=muon_lr, weight_decay=muon_wd)
    return Combined(mu, opt)
