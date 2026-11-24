"""The model: objects in, motion out. Not next-token: next-state.

Every box (wall or platform) is one token, described only relative to the ball:
(ball - box centre, box half size, ball velocity). One extra token stands for
the ball. A small transformer lets the ball token look at the boxes; its final
state is read out.

  step head   -> change of position, change of velocity, "it has come to rest"
  shot head   -> final resting position (relative to the start), in one go

The blocks are written by hand (no nn.MultiheadAttention) so the editor's
JavaScript can mirror them line by line.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

R = 0.03            # ball radius (same as sim.R)
DP_SCALE = 0.04   # typical change of position in one model step
DV_SCALE = 0.25   # typical change of velocity in one model step


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d); self.f1 = nn.Linear(d, 2 * d); self.f2 = nn.Linear(2 * d, d)

    def forward(self, x, mask):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = (q @ k.transpose(-1, -2)) / math.sqrt(D // self.h)
        a = a.masked_fill(~mask[:, None, None, :], -1e9).softmax(-1)
        x = x + self.proj((a @ v).transpose(1, 2).reshape(B, T, D))
        return x + self.f2(F.gelu(self.f1(self.n2(x))))


class Net(nn.Module):
    def __init__(self, out, d=128, layers=3, heads=4, feat=0):
        super().__init__()
        self.cfg = dict(out=out, d=d, layers=layers, heads=heads, feat=feat)
        self.feat = feat
        self.box1 = nn.Linear((6, 8, 11, 15)[feat], d); self.box2 = nn.Linear(d, d)
        self.ball1 = nn.Linear(2, d); self.ball2 = nn.Linear(d, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.out1 = nn.Linear(d, d); self.out2 = nn.Linear(d, out)

    def forward(self, p, v, boxes, mask):
        """p, v: [N,2]; boxes: [N,B,4]; mask: [N,B] true for real boxes."""
        N, B, _ = boxes.shape
        dc = p[:, None] - boxes[..., :2]; half = boxes[..., 2:]
        parts = [dc, half, v[:, None].expand(N, B, 2)]
        if self.feat >= 1:                           # how far past each face (negative = alongside the box)
            parts.append(dc.abs() - half)
        if self.feat >= 2:                           # from the nearest point of the box to the ball, and the gap left
            o = dc - torch.maximum(torch.minimum(dc, half), -half)
            dist = o.norm(dim=-1, keepdim=True)
            parts += [o, dist - R]
            if self.feat >= 3:                       # the same, at the scale where a hit is decided:
                n = o / (dist + 1e-6)                # which way the surface faces, the gap magnified, the approach speed
                parts += [n, ((dist - R) * 25).clamp(-1, 4), (v[:, None] * n).sum(-1, keepdim=True)]
        f = torch.cat(parts, -1)
        f = f * mask[..., None]                      # padding boxes carry nothing
        x = self.box2(F.gelu(self.box1(f)))
        b = self.ball2(F.gelu(self.ball1(v)))[:, None]
        x = torch.cat([b, x], 1)
        m = torch.cat([torch.ones(N, 1, dtype=torch.bool, device=mask.device), mask], 1)
        for blk in self.blocks:
            x = blk(x, m)
        return self.out2(F.gelu(self.out1(self.norm(x[:, 0]))))


def step_model(**kw):
    return Net(5, **kw)   # dp(2), dv(2), rest logit


def shot_model(**kw):
    return Net(2, **kw)   # final position minus start


@torch.no_grad()
def roll(net, p, v, boxes, mask, steps=360, path=False):
    """Roll the step model forward until it says the ball rests. Returns final p (and the path)."""
    p = p.clone(); v = v.clone()
    live = torch.ones(p.shape[0], dtype=torch.bool, device=p.device)
    out = [p.clone()] if path else None
    for _ in range(steps):
        if not live.any():
            break
        y = net(p[live], v[live], boxes[live], mask[live])
        rest = y[:, 4] > 0
        dp = y[:, :2] * DP_SCALE; dv = y[:, 2:4] * DV_SCALE
        dp[rest] = 0
        p[live] = (p[live] + dp).clamp(0.0, 1.0)   # the room is the unit square
        v[live] = torch.where(rest[:, None], torch.zeros_like(dv), v[live] + dv)
        idx = live.nonzero()[:, 0]
        live[idx[rest]] = False
        if path:
            out.append(p.clone())
    return (p, torch.stack(out)) if path else p
