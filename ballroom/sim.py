"""The exact simulator: a ball in a 2D room with walls and platforms.

Everything is "a circle against boxes". The four walls are boxes too, so the
room has one code path and the model sees one kind of object.

Units: the room is the unit square, y points up. Fixed time step, so the same
arithmetic can be mirrored line by line in the editor's JavaScript.
"""
import numpy as np

R = 0.03          # ball radius
G = 2.5           # gravity
DT = 1.0 / 120.0  # simulator step
E = 0.65          # bounce: share of normal speed kept
KEEP_T = 0.9      # bounce: share of sideways speed kept
VTH = 0.15        # slower than this into a surface = no bounce, it rests
ROLL = 1.6        # rolling friction: speed lost per second on a support
SUB = 4           # simulator steps per model step (model step = 1/30 s)
MAX_T = 12.0      # seconds; every room settles well before this
MAX_BOX = 9       # 4 walls + up to 5 platforms
FAR = -10.0       # where padding boxes live
ORDER = [4, 5, 6, 7, 8, 0, 1, 2, 3]  # platforms first, walls last: walls always win

# walls as thick boxes (cx, cy, half width, half height)
WALLS = np.array([
    [0.5, -0.5, 1.5, 0.5],   # floor
    [0.5, 1.5, 1.5, 0.5],    # ceiling
    [-0.5, 0.5, 0.5, 1.5],   # left
    [1.5, 0.5, 0.5, 1.5],    # right
])


def step(p, v, boxes, n=1):
    """Advance n simulator steps. p, v: [N,2]; boxes: [N,B,4]. Returns new p, v."""
    p = p.copy(); v = v.copy()
    B = boxes.shape[1]
    for _ in range(n):
        v[:, 1] -= G * DT
        p += v * DT
        for b in ORDER[:B] if B == MAX_BOX else range(B):
            c = boxes[:, b, :2]; h = boxes[:, b, 2:]
            d = p - c
            q = np.clip(d, -h, h)            # closest point of the box, from its centre
            o = d - q                         # from closest point to the ball centre
            dist = np.sqrt(o[:, 0] ** 2 + o[:, 1] ** 2)
            hit = dist < R
            if not hit.any():
                continue
            nx = np.zeros_like(dist); ny = np.zeros_like(dist); push = np.zeros_like(dist)
            out = hit & (dist > 0)
            nx[out] = o[out, 0] / dist[out]; ny[out] = o[out, 1] / dist[out]
            push[out] = R - dist[out]
            ins = hit & ~(dist > 0)           # centre inside the box: leave by the nearest face
            if ins.any():
                px = h[ins, 0] - np.abs(d[ins, 0]); py = h[ins, 1] - np.abs(d[ins, 1])
                sx = np.where(d[ins, 0] >= 0, 1.0, -1.0); sy = np.where(d[ins, 1] >= 0, 1.0, -1.0)
                ux = px < py
                nx[ins] = np.where(ux, sx, 0.0); ny[ins] = np.where(ux, 0.0, sy)
                push[ins] = np.where(ux, px, py) + R
            p[:, 0] += nx * push; p[:, 1] += ny * push
            vn = v[:, 0] * nx + v[:, 1] * ny
            into = hit & (vn < 0)
            tx = v[:, 0] - vn * nx; ty = v[:, 1] - vn * ny
            bounce = into & (vn < -VTH)
            rest = into & ~bounce
            # bounce
            v[:, 0] = np.where(bounce, tx * KEEP_T - E * vn * nx, v[:, 0])
            v[:, 1] = np.where(bounce, ty * KEEP_T - E * vn * ny, v[:, 1])
            # rest: no normal speed; on a support, rolling friction eats sideways speed
            ts = np.sqrt(tx ** 2 + ty ** 2)
            sup = rest & (ny > 0.5)
            cut = np.where(sup, np.minimum(ts, ROLL * DT), 0.0)
            k = np.where(ts > 0, (ts - cut) / np.where(ts > 0, ts, 1.0), 0.0)
            v[:, 0] = np.where(rest, tx * k, v[:, 0])
            v[:, 1] = np.where(rest, ty * k, v[:, 1])
    return p, v


def make_rooms(n, rng, n_plat=(2, 5)):
    """Random rooms. Returns boxes [n,MAX_BOX,4], count [n], p0 [n,2], v0 [n,2]."""
    boxes = np.zeros((n, MAX_BOX, 4)); boxes[:, :, :2] = FAR
    boxes[:, :4] = WALLS
    count = rng.integers(n_plat[0], n_plat[1] + 1, size=n)
    p0 = np.zeros((n, 2)); v0 = np.zeros((n, 2))
    gap = R + 0.012                      # platforms keep a ball-wide gap between them
    for i in range(n):
        placed = []
        while len(placed) < count[i]:
            hw = rng.uniform(0.08, 0.28); hh = rng.uniform(0.01, 0.025)
            cx = rng.uniform(hw, 1 - hw); cy = rng.uniform(0.12, 0.82)
            ok = all(abs(cx - a) > hw + c + 2 * gap or abs(cy - b) > hh + d + 2 * gap
                     for a, b, c, d in placed)
            if ok:
                placed.append((cx, cy, hw, hh))
        boxes[i, 4:4 + count[i]] = placed
        while True:
            x = rng.uniform(R + 0.01, 1 - R - 0.01); y = rng.uniform(R + 0.01, 1 - R - 0.01)
            if all(abs(x - a) > c + R + 0.01 or abs(y - b) > d + R + 0.01 for a, b, c, d in placed):
                break
        p0[i] = (x, y)
        ang = rng.uniform(0, 2 * np.pi); sp = rng.uniform(0, 1.5)
        v0[i] = (sp * np.cos(ang), sp * np.sin(ang))
    return boxes, count, p0, v0


def rollout(boxes, p0, v0, max_t=MAX_T):
    """Run to rest. Returns P [T+1,N,2], V [T+1,N,2] at model steps, and stop step [N]."""
    T = int(round(max_t / (DT * SUB)))
    N = p0.shape[0]
    P = np.zeros((T + 1, N, 2)); V = np.zeros((T + 1, N, 2))
    P[0] = p0; V[0] = v0
    stop = np.full(N, T, dtype=np.int64)
    p, v = p0, v0
    live = np.ones(N, dtype=bool)
    for t in range(1, T + 1):
        if live.any():
            pn, vn = step(p[live], v[live], boxes[live], SUB)
            p = p.copy(); v = v.copy()
            done = live.copy()
            # at rest = it did not move during a whole model step (also covers a wedged ball)
            still = np.abs(pn - p[live]).max(axis=1) < 1e-9
            done[live] = still
            vn[still] = 0.0
            p[live] = pn; v[live] = vn
            stop[done & (stop == T)] = t
            live = live & ~done
        P[t] = p; V[t] = v
    return P, V, stop


if __name__ == "__main__":
    import time
    rng = np.random.default_rng(0)
    t0 = time.time()
    boxes, count, p0, v0 = make_rooms(4000, rng)
    t1 = time.time()
    P, V, stop = rollout(boxes, p0, v0)
    T = P.shape[0] - 1
    print(f"rooms {len(p0)}  make {t1 - t0:.1f}s  roll {time.time() - t1:.1f}s")
    print(f"settled {np.mean(stop < T) * 100:.2f}%  stop step mean {stop.mean():.0f} "
          f"p50 {np.median(stop):.0f} p99 {np.percentile(stop, 99):.0f} of {T}")
    fin = P[-1]
    floor = fin[:, 1] < R + 1e-6
    print(f"ends on the floor {floor.mean() * 100:.1f}%  on a platform {(~floor).mean() * 100:.1f}%")
    print(f"x spread {fin[:, 0].std():.3f}  bounds {P.min():.3f} {P.max():.3f}")


def step_t(p, v, boxes, n=1):
    """The same step as step(), on torch tensors (float64), so the simulator can run on the GPU."""
    import torch
    p = p.clone(); v = v.clone()
    B = boxes.shape[1]
    zero = torch.zeros((), dtype=p.dtype, device=p.device)
    for _ in range(n):
        v[:, 1] -= G * DT
        p += v * DT
        for b in (ORDER[:B] if B == MAX_BOX else range(B)):
            c = boxes[:, b, :2]; h = boxes[:, b, 2:]
            d = p - c
            o = d - torch.maximum(torch.minimum(d, h), -h)
            dist = torch.sqrt(o[:, 0] ** 2 + o[:, 1] ** 2)
            hit = dist < R
            out = hit & (dist > 0)
            safe = torch.where(out, dist, torch.ones_like(dist))
            nx = torch.where(out, o[:, 0] / safe, zero); ny = torch.where(out, o[:, 1] / safe, zero)
            push = torch.where(out, R - dist, zero)
            ins = hit & ~out
            px = h[:, 0] - d[:, 0].abs(); py = h[:, 1] - d[:, 1].abs()
            sx = torch.where(d[:, 0] >= 0, 1.0, -1.0).to(p.dtype); sy = torch.where(d[:, 1] >= 0, 1.0, -1.0).to(p.dtype)
            ux = px < py
            nx = torch.where(ins, torch.where(ux, sx, zero), nx); ny = torch.where(ins, torch.where(ux, zero, sy), ny)
            push = torch.where(ins, torch.where(ux, px, py) + R, push)
            p = torch.stack([p[:, 0] + nx * push, p[:, 1] + ny * push], 1)
            vn = v[:, 0] * nx + v[:, 1] * ny
            into = hit & (vn < 0)
            tx = v[:, 0] - vn * nx; ty = v[:, 1] - vn * ny
            bounce = into & (vn < -VTH)
            rest = into & ~bounce
            ts = torch.sqrt(tx ** 2 + ty ** 2)
            cut = torch.where(rest & (ny > 0.5), torch.clamp(ts, max=ROLL * DT), zero)
            k = torch.where(ts > 0, (ts - cut) / torch.where(ts > 0, ts, torch.ones_like(ts)), zero)
            vx = torch.where(bounce, tx * KEEP_T - E * vn * nx, torch.where(rest, tx * k, v[:, 0]))
            vy = torch.where(bounce, ty * KEEP_T - E * vn * ny, torch.where(rest, ty * k, v[:, 1]))
            v = torch.stack([vx, vy], 1)
    return p, v
