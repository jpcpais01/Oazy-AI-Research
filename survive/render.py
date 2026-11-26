"""The creature's two eyes: a batched ray caster in plain torch (GPU).

Each eye sees 120 degrees across (and the same angle up/down, square pixels) from 0.45 m
above the feet; the eyes sit 8 cm apart. A ray stops at the nearest of: floor (checker,
lava patches), the arena walls (sky above them), boxes/crates/food/the strange being
(axis-aligned boxes) and the predator/wanderer (spheres). Faces are shaded by a fixed
light, and everything fades into haze with distance: colour * exp(-d/12 m) (food glows,
so it fades half as fast). Output: uint8 (B, 6, H, W) = left RGB, right RGB.
"""
from __future__ import annotations

import math

import torch

from .world import EYE_H, EYE_SEP, L, WALL_H

FOG_L = 12.0
TAN = math.tan(math.radians(60))
LIGHT = torch.tensor([0.35, 0.25, 0.9])
FOG = torch.tensor([0.80, 0.85, 0.90])
SKY = torch.tensor([0.70, 0.80, 0.95])
LAVA = torch.tensor([0.92, 0.45, 0.18])

_dirs_cache: dict = {}


def _pixel_dirs(H, dev):
    key = (H, str(dev))
    if key not in _dirs_cache:
        s = (torch.arange(H, device=dev, dtype=torch.float32) + 0.5) / H * 2 - 1
        py, px = torch.meshgrid(-s * TAN, s * TAN, indexing="ij")              # up, right
        _dirs_cache[key] = (px.reshape(-1), py.reshape(-1))
    return _dirs_cache[key]


def _render_chunk(sc, H):
    pos, head = sc["pos"], sc["head"]
    B, dev = pos.shape[0], pos.device
    px, py = _pixel_dirs(H, dev)                                              # P
    P = px.shape[0]
    f = torch.stack([torch.cos(head), torch.sin(head), torch.zeros_like(head)], -1)   # B,3
    r = torch.stack([torch.sin(head), -torch.cos(head), torch.zeros_like(head)], -1)  # right
    u = torch.tensor([0.0, 0.0, 1.0], device=dev)
    d = f[:, None] + px[None, :, None] * r[:, None] + py[None, :, None] * u           # B,P,3
    d = d / d.norm(dim=-1, keepdim=True)
    eye = pos + torch.tensor([0.0, 0.0, EYE_H], device=dev)
    o = torch.stack([eye - r * EYE_SEP, eye + r * EYE_SEP], 1)                         # B,2,3 (left, right)
    d = d[:, None].expand(B, 2, P, 3).reshape(B, 2 * P, 3)
    o = o[:, :, None].expand(B, 2, P, 3).reshape(B, 2 * P, 3)
    inf = torch.full(d.shape[:2], 1e9, device=dev)
    dsafe = torch.where(d.abs() < 1e-6, torch.full_like(d, 1e-6), d)
    inv = 1.0 / dsafe

    # walls (inside of the arena box): exit distance in x and y
    tx = torch.where(d[..., 0] > 0, (L - o[..., 0]) * inv[..., 0], -o[..., 0] * inv[..., 0])
    ty = torch.where(d[..., 1] > 0, (L - o[..., 1]) * inv[..., 1], -o[..., 1] * inv[..., 1])
    tw = torch.minimum(tx, ty)
    wz = o[..., 2] + tw * d[..., 2]
    wall_id = torch.where(tx < ty, torch.where(d[..., 0] > 0, 1, 0), torch.where(d[..., 1] > 0, 3, 2))
    t_wall = torch.where((wz >= 0) & (wz <= WALL_H), tw, inf)
    # floor
    t_floor = torch.where(d[..., 2] < -1e-6, -o[..., 2] * inv[..., 2], inf)
    # boxes (slab test)
    amin, amax = sc["amin"][:, None], sc["amax"][:, None]                             # B,1,N,3
    t1 = (amin - o[:, :, None]) * inv[:, :, None]
    t2 = (amax - o[:, :, None]) * inv[:, :, None]
    tmin3 = torch.minimum(t1, t2)
    tnear, axis = tmin3.max(-1)                                                        # B,R,N
    tfar = torch.maximum(t1, t2).min(-1).values
    hit = (tnear <= tfar) & (tfar > 0) & (tnear > 0)
    tb = torch.where(hit, tnear, torch.full_like(tnear, 1e9))
    t_box, bi = tb.min(-1)                                                             # B,R
    b_axis = axis.gather(-1, bi[..., None]).squeeze(-1)
    # spheres
    c = torch.cat([sc["sph"][..., :2], sc["sph_r"][..., None]], -1)[:, None]          # B,1,S,3
    oc = o[:, :, None] - c
    bq = (oc * d[:, :, None]).sum(-1)
    cq = (oc * oc).sum(-1) - sc["sph_r"][:, None] ** 2
    disc = bq * bq - cq
    ts = -bq - torch.sqrt(disc.clamp(min=0))
    ts = torch.where((disc > 0) & (ts > 0), ts, torch.full_like(ts, 1e9))
    t_sph, si = ts.min(-1)

    t_all = torch.stack([t_floor, t_wall, t_box, t_sph], -1)
    t, kind = t_all.min(-1)                                                            # B,R
    hitp = o + d * t[..., None]
    Ld = LIGHT.to(dev) / LIGHT.norm()

    # floor colour: checker, lava on top
    cx = torch.floor(hitp[..., 0]).long() + torch.floor(hitp[..., 1]).long()
    fcol = sc["floor_col"].gather(1, (cx % 2).clamp(0, 1)[..., None].expand(-1, -1, 3))
    lava = sc["lava"][:, None]                                                         # B,1,NL,4
    inl = ((hitp[..., None, 0] > lava[..., 0]) & (hitp[..., None, 0] < lava[..., 2]) &
           (hitp[..., None, 1] > lava[..., 1]) & (hitp[..., None, 1] < lava[..., 3])).any(-1)
    fcol = torch.where(inl[..., None], LAVA.to(dev), fcol)
    fcol = fcol * (0.7 + 0.3 * Ld[2])
    # walls
    wcol = sc["wall_col"].gather(1, wall_id[..., None].expand(-1, -1, 3)) * (0.75 + 0.25 * (wz / WALL_H).clamp(0, 1))[..., None]
    # boxes: shade by the face that was hit
    bcol = sc["acol"].gather(1, bi[..., None].expand(-1, -1, 3))
    sgn = -torch.sign(d.gather(-1, b_axis[..., None]).squeeze(-1))
    nrm = torch.zeros_like(d).scatter(-1, b_axis[..., None], sgn[..., None])
    bcol = bcol * (0.55 + 0.45 * (nrm * Ld).sum(-1).clamp(min=0))[..., None]
    glow = sc["glow"].gather(1, bi)
    # spheres
    ccen = c[:, 0].gather(1, si[..., None].expand(-1, -1, 3))
    snrm = (hitp - ccen)
    snrm = snrm / snrm.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    scol = sc["sph_col"].gather(1, si[..., None].expand(-1, -1, 3)) * (0.5 + 0.5 * (snrm * Ld).sum(-1).clamp(min=0))[..., None]
    sky = SKY.to(dev) * (1 - 0.15 * d[..., 2:3].clamp(0, 1))

    col = torch.where((kind == 0)[..., None], fcol,
          torch.where((kind == 1)[..., None], wcol,
          torch.where((kind == 2)[..., None], bcol, scol)))
    nothing = t >= 1e8
    fl = torch.where((kind == 2) & (glow > 0), 2 * FOG_L, FOG_L)
    a = torch.exp(-t.clamp(max=1e3) / fl)[..., None]
    col = col * a + FOG.to(dev) * (1 - a)
    col = torch.where(nothing[..., None], sky, col)
    img = (col.clamp(0, 1) * 255).round().to(torch.uint8)                              # B,2P,3
    return img.view(B, 2, H, H, 3).permute(0, 1, 4, 2, 3).reshape(B, 6, H, H)


_render_compiled = None


def render(world, H=64, chunk=None, compile=False):
    """uint8 (B, 6, H, H): left eye RGB then right eye RGB."""
    global _render_compiled
    fn = _render_chunk
    if compile:
        if _render_compiled is None:
            _render_compiled = torch.compile(_render_chunk, dynamic=False)
        fn = _render_compiled
    sc = world.scene()
    B = sc["pos"].shape[0]
    if chunk is None:
        chunk = max(1, int(2 ** 24 // (H * H * 2 * sc["amin"].shape[1])))
    n = -(-B // chunk)
    while B % n:                                # equal chunks: one shape to compile
        n += 1
    chunk = B // n
    _pixel_dirs(H, sc["pos"].device)           # fill the cache outside the compiled code
    if n == 1:
        return fn(sc, H)
    outs = []
    for s in range(0, B, chunk):
        sub = {k: v[s:s + chunk] for k, v in sc.items()}
        outs.append(fn(sub, H))
    return torch.cat(outs, 0)
