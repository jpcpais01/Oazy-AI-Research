"""Seed's optimizer: Bee5's (NorMuon for the weight grids, Adam groups for the rest), with three changes.

  core      the grids of the core's layers (attention, MLP, the library readers' question / gate / output) learn with
            NorMuon, and with Grokfast on (cfg.grokfast = 1): before each step the slow average of each grid's
            gradient (an EMA, alpha 0.98) is added back twice (lambda 2; Grokfast-EMA, Lee et al. 2024), so the steady
            push that leads to general rules is louder than the batch-to-batch noise; and their weight decay is
            cfg.core_wd (0.2, twice Bee5's), so the core prefers small, simple solutions. cfg.grokfast = 0 turns both
            off (Bee5's NorMuon and decay), the race's control
  library   the rows learn with a row-wise Adam: a row that was not read this step does not move at all (no momentum
            drift, no decay), so a fact used rarely is kept between its uses; rate cfg.lib_lr x --lr. Its half-keys
            learn with plain Adam at the rate
  the rest  Bee5's Adam groups (word table, value tables, scalars at 10x, gates)

After a growth the trainer builds a new optimizer and calls Seed.carry_opt: every replaced (bigger) parameter gets
its predecessor's memory, padded with zeros for the new parts and scaled the way its gradient is now scaled.
"""
from __future__ import annotations

import re

import torch

from bee5.optim import ADAM, NorMuon
from oazy.muon import Combined

CORE = re.compile(r"blocks\.\d+\.(attn\.(q|k|v|out)|mlp\.(up|down)|look\.(query|gate|out))\.weight$")
SCALARS = re.compile(r"(blocks\.\d+\.(lam|mu)|skip_lam|smear_lam|backout_lam)$")


class RowAdam(torch.optim.Optimizer):
    """Adam for a big table whose rows are read sparsely: only rows with a gradient this step are updated (their
    moments too); the bias correction counts every step (as torch's SparseAdam). Works in slices of rows."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-10, chunk=1 << 17):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, chunk=chunk))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "exp_avg" not in st:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p)
                    st["exp_avg_sq"] = torch.zeros_like(p)
                st["step"] += 1
                t = st["step"]
                c1, c2 = 1 - b1 ** t, 1 - b2 ** t
                G, M, V = p.grad, st["exp_avg"], st["exp_avg_sq"]
                for a in range(0, p.shape[0], g["chunk"]):
                    sl = slice(a, a + g["chunk"])
                    gc = G[sl]
                    hit = gc.ne(0).any(-1, keepdim=True)              # rows read this step
                    if not bool(hit.any()):
                        continue
                    m, v = M[sl], V[sl]
                    m.copy_(torch.where(hit, m * b1 + gc * (1 - b1), m))
                    v.copy_(torch.where(hit, v * b2 + gc.square() * (1 - b2), v))
                    upd = (m / c1) / ((v / c2).sqrt() + g["eps"])
                    p[sl].sub_(upd * hit * g["lr"])


class SeedOpt(Combined):
    """Combined optimizers, with Grokfast-EMA on the core's grids before each step."""

    def __init__(self, *opts, core=(), alpha=0.98, lamb=0.0):
        super().__init__(*opts)
        self.core, self.alpha, self.lamb = list(core), alpha, lamb
        self.gf = {}

    @torch.no_grad()
    def grokfast(self):
        for p in self.core:
            if p.grad is None:
                continue
            e = self.gf.get(p)
            if e is None:
                self.gf[p] = e = p.grad.detach().clone()
            else:
                e.lerp_(p.grad, 1 - self.alpha)
            p.grad.add_(e, alpha=self.lamb)

    def step(self):
        if self.lamb > 0:
            self.grokfast()
        super().step()

    def state_dict(self):
        return {**super().state_dict(), "grokfast": [self.gf.get(p) for p in self.core]}

    def load_state_dict(self, sd):
        super().load_state_dict(sd)
        for p, e in zip(self.core, sd.get("grokfast", [])):
            if e is not None and e.shape == p.shape:
                self.gf[p] = e.to(p.device, p.dtype)


def groups(model):
    """(core grids, {adam group: params}, library rows), by name. Every trainable parameter lands in exactly one."""
    core, adam, rows = [], {"table": [], "ve": [], "scalars": [], "gates": [], "keys": []}, []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if CORE.match(n) and p.dim() == 2:
            core.append((n, p))
        elif n == "embed.weight":
            adam["table"].append(p)
        elif n.startswith("ve."):
            adam["ve"].append(p)
        elif SCALARS.match(n):
            adam["scalars"].append(p)
        elif n == "lib.values":
            rows.append(p)
        elif n == "lib.keys":
            adam["keys"].append(p)
        else:
            adam["gates"].append(p)
    return core, adam, rows


GROUPS = dict(ADAM, keys=dict(lr_scale=1.0, betas=(0.9, 0.95)))


def make(model, lr, device, muon_lr=0.03, muon_wd=0.1, kind="muon"):
    cfg = model.cfg
    core, adam, rows = groups(model)
    gs = [dict(params=ps, weight_decay=0.0, name=k, lr=lr * GROUPS[k]["lr_scale"], **GROUPS[k]) for k, ps in adam.items() if ps]
    wd = cfg.core_wd if cfg.grokfast else muon_wd
    if kind != "muon":
        gs.append(dict(params=[p for _, p in core], weight_decay=wd, name="grids", lr=lr, lr_scale=1.0, betas=(0.9, 0.95)))
    kw = dict(lr=lr, eps=1e-10)
    if device.type == "cuda":
        kw["fused"] = True
    opt = torch.optim.AdamW(gs, **kw)
    lib = RowAdam([dict(params=rows, lr=lr * cfg.lib_lr, lr_scale=cfg.lib_lr, name="library")]) if rows else None
    mu = None
    if kind == "muon" and core:
        mu = NorMuon([dict(params=[p for _, p in core], lr_scale=muon_lr / lr)], lr=muon_lr, weight_decay=wd)
    return SeedOpt(mu, opt, lib, core=[p for _, p in core], alpha=cfg.gf_alpha, lamb=cfg.gf_lamb if cfg.grokfast else 0.0)


def _place(old_t, new_shape, fill_value=0.0, how=None):
    """old_t in the leading corner of a new tensor (or, how = ("index", old_idx, new_idx), at those entries)."""
    out = torch.full(new_shape, fill_value, dtype=old_t.dtype, device=old_t.device)
    if isinstance(how, tuple) and how[0] == "index":
        if how[1]:
            out[torch.tensor(how[2], device=old_t.device)] = old_t[torch.tensor(how[1], device=old_t.device)]
        return out
    out[tuple(slice(0, s) for s in old_t.shape)] = old_t
    return out


def carry(records, old, new):
    """records: [(old parameter, new parameter, gradient factor | ("index", old_idx, new_idx))]."""
    olds = old.opts if hasattr(old, "opts") else [old]
    news = new.opts if hasattr(new, "opts") else [new]
    for o, n in zip(olds, news):
        for p_old, p_new, how in records:
            st = o.state.get(p_old)
            if not st:
                continue
            f = how if not isinstance(how, tuple) else 1.0
            ns = {}
            for key, val in st.items():
                if not torch.is_tensor(val) or val.dim() == 0 or key == "step":
                    ns[key] = val.clone() if torch.is_tensor(val) else val
                elif key == "second":                     # NorMuon's per-neuron size: new neurons start at the mean
                    if val.dim() == 3:
                        val = val[0]
                    shp = (p_new.shape[0], 1) if val.shape[-1] == 1 else (1, p_new.shape[1])
                    if (val.shape[-1] == 1) == (shp[1] == 1) and all(a <= b for a, b in zip(val.shape, shp)):
                        ns[key] = _place(val, shp, val.mean().item())
                elif val.shape == p_old.shape:
                    power = 2 if key == "exp_avg_sq" else 1
                    ns[key] = _place(val * (f ** power), p_new.shape, 0.0, how if isinstance(how, tuple) else None)
            n.state[p_new] = ns
        live = {id(p) for g in n.param_groups for p in g["params"]}
        for k in [k for k in n.state if id(k) not in live]:
            del n.state[k]
    if hasattr(old, "gf") and hasattr(new, "gf"):
        live = {id(p) for p in new.core}
        for p_old, p_new, how in records:
            e = old.gf.get(p_old)
            if e is not None and id(p_new) in live:
                f = how if not isinstance(how, tuple) else 1.0
                new.gf[p_new] = _place(e * f, p_new.shape, 0.0, how if isinstance(how, tuple) else None)
        for p, e in old.gf.items():
            if id(p) in live and p not in new.gf:
                new.gf[p] = e
