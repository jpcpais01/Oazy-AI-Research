"""Muon: an optimizer that moves every direction of a weight grid evenly.

In plain words: Adam nudges each number of a weight grid on its own. A grid's update, though, is
usually dominated by a few strong directions (the ones that already work), so the weak directions
barely learn. Muon takes the update grid and "straightens" it (an orthogonalisation, done with five
cheap matrix multiplications, Newton-Schulz) so that every direction moves the same amount. The
model then learns its weak skills as fast as its strong ones. In the public nanoGPT speedrun it
reached GPT-2's loss with ~1.35x fewer tokens than AdamW, at almost no extra cost per step.

Muon is only for the weight grids inside the blocks (2-D, or a stack of them: Coral's experts, each
straightened on its own, as Kimi K2 and Moonlight do for theirs). Word embeddings, norms, the memories'
trust/fade knobs and other small parameters stay on AdamW (that is what Combined is for).

Weight decay (Bee3: 0.1) pulls every grid a little towards zero each step, measured on AdamW's
learning-rate scale so that "0.1" means what it means for AdamW. Moonshot's Moonlight and Kimi K2
found Muon needs it on long runs: without it the grids drift large and lose precision in bf16.

Reference: Keller Jordan et al., "Muon: an optimizer for hidden layers" (2024).
"""
from __future__ import annotations

import torch


def straighten(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz: G -> the nearest grid whose directions all have the same strength (~U V^T)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() if G.is_cuda else G.float()
    tall = X.size(-2) > X.size(-1)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    if tall:
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Momentum, then straighten. A parameter may carry `p.muon_split = [rows, rows, ...]` when it is
    really several grids stored as one (e.g. question/key/value): each part is straightened alone."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.0, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      weight_decay=weight_decay, ns_steps=ns_steps))

    batched = True      # straighten all grids of the same shape together (one batched product, far fewer GPU steps)

    @torch.no_grad()
    def step(self, closure=None):
        todo = []                                   # (group, param, its parts)
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "momentum" not in st:
                    st["momentum"] = torch.zeros_like(p.grad)
                buf = st["momentum"]
                buf.lerp_(p.grad, 1 - g["momentum"])
                u = p.grad.lerp(buf, g["momentum"]) if g["nesterov"] else buf
                # a stack of grids (E, out, in: Coral's experts) is straightened grid by grid, in one batch
                parts = list(u.split(p.muon_split, dim=-2)) if getattr(p, "muon_split", None) else [u]
                todo.append((g, p, parts))
        # straighten: every part of the same shape at once (the same maths as one by one; Newton-Schulz works on the
        # last two dimensions and scales each grid by its own size)
        same = {}
        for i, (g, _, parts) in enumerate(todo):
            for j, x in enumerate(parts):
                key = (tuple(x.shape), x.dtype, x.device, g["ns_steps"]) if self.batched else (i, j)
                same.setdefault(key, []).append((i, j))
        for key, where in same.items():
            xs = [todo[i][2][j] for i, j in where]
            steps = todo[where[0][0]][0]["ns_steps"]
            X = torch.stack(xs) if len(xs) > 1 else xs[0].unsqueeze(0)
            Y = straighten(X, steps) * max(1.0, X.size(-2) / X.size(-1)) ** 0.5
            for (i, j), y in zip(where, Y.unbind(0)):
                todo[i][2][j] = y
        for g, p, parts in todo:
            u = torch.cat(parts, dim=-2) if len(parts) > 1 else parts[0]
            if g["weight_decay"]:       # on AdamW's scale: the group's lr_scale is Muon's rate over the AdamW rate
                p.mul_(1 - g["lr"] / g.get("lr_scale", 1.0) * g["weight_decay"])
            p.add_(u, alpha=-g["lr"])


class Combined:
    """Several optimizers that behave as one for the trainer (step, zero_grad, param_groups, state)."""

    def __init__(self, *opts):
        self.opts = [o for o in opts if o is not None]

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def step(self):
        for o in self.opts:
            o.step()

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"combined": [o.state_dict() for o in self.opts]}

    def load_state_dict(self, sd):
        for o, s in zip(self.opts, sd["combined"]):
            o.load_state_dict(s)
