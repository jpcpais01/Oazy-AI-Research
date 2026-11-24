"""Train the step model ("law") and the one-shot model ("intuition").

  python3 -m ballroom.train --what step --minutes 20
  python3 -m ballroom.train --what shot --minutes 10

Train rooms have 2-3 platforms by default; ballroom.test scores on 4-5.

The step model learns one model step (1/30 s) of the exact simulator. Because a
rolled-out model drifts off the true path, training states come from three
places, all labelled by the exact simulator:
  1. states on true paths
  2. the same states nudged a little
  3. states the model itself visits when rolled out (refreshed during training)
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from . import sim
from .model import step_model, shot_model, roll, DP_SCALE, DV_SCALE


def label(p, v, boxes):
    """Exact one-model-step answer for any states."""
    pn, vn = sim.step(p, v, boxes, sim.SUB)
    still = np.abs(pn - p).max(axis=1) < 1e-9
    vn[still] = 0.0
    return pn, vn, still


def path_states(boxes, p0, v0, rest_keep=3):
    """All states on the true paths up to rest (plus a few resting ones)."""
    P, V, stop = sim.rollout(boxes, p0, v0)
    T, N = P.shape[0] - 1, P.shape[1]
    t = np.arange(T + 1)[:, None]
    keep = t < np.minimum(stop + rest_keep, T)[None, :]
    ti, ri = np.nonzero(keep)
    return P[ti, ri], V[ti, ri], ri, P[stop, np.arange(N)]


def tens(*a, dev):
    return [torch.as_tensor(x, dtype=torch.float32, device=dev) for x in a]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="step", choices=["step", "shot"])
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--rooms", type=int, default=30000)
    ap.add_argument("--plat", default="2,3")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--refresh", type=float, default=3.0, help="minutes between model-visited refreshes")
    ap.add_argument("--out", default="ballroom/runs")
    ap.add_argument("--init", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = a.device
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    lo, hi = map(int, a.plat.split(","))
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    boxes, count, p0, v0 = sim.make_rooms(a.rooms, rng, (lo, hi))
    mask = np.zeros((a.rooms, sim.MAX_BOX), dtype=bool)
    mask[:, :4] = True
    for i in range(a.rooms):
        mask[i, 4:4 + count[i]] = True
    Sp, Sv, Sr, fin = path_states(boxes, p0, v0)
    print(f"rooms {a.rooms} ({lo}-{hi} platforms)  path states {len(Sp):,}  {time.time() - t0:.0f}s", flush=True)
    tb, tm = tens(boxes, dev=dev)[0], torch.as_tensor(mask, device=dev)

    make = step_model if a.what == "step" else shot_model
    net = make(d=a.d, layers=a.layers).to(dev)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location=dev)["model"])
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-4)
    budget = a.minutes * 60
    print(f"{a.what} model {sum(p.numel() for p in net.parameters()):,} weights on {dev}", flush=True)

    if a.what == "shot":
        tp, tv, tf = tens(p0, v0, fin, dev=dev)
        it = 0; t0 = time.time(); run = None
        while time.time() - t0 < budget:
            frac = (time.time() - t0) / budget
            for g in opt.param_groups:
                g["lr"] = a.lr * (0.5 + 0.5 * np.cos(np.pi * frac)) + 1e-5
            i = torch.randint(0, a.rooms, (a.batch,), device=dev)
            y = net(tp[i], tv[i], tb[i], tm[i])
            loss = F.smooth_l1_loss(y, (tf[i] - tp[i]) / 0.3, beta=0.1)
            opt.zero_grad(); loss.backward(); opt.step()
            run = loss.item() if run is None else 0.99 * run + 0.01 * loss.item()
            it += 1
            if it % 200 == 0:
                print(f"it {it} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
    else:
        # near-contact states matter most: free flight is trivial
        def near(p, r):
            d = np.abs(p[:, None] - boxes[r, :, :2]) - boxes[r, :, 2:]
            gap = np.sqrt((np.maximum(d, 0) ** 2).sum(-1)).min(1)
            return gap < sim.R + 0.05
        pools = {}

        def add(name, p, v, r):
            pn, vn, still = label(p, v, boxes[r])
            pools[name] = tens(p, v, pn, vn, still, near(p, r), dev=dev) + [torch.as_tensor(r, device=dev)]

        add("path", Sp, Sv, Sr)
        k = rng.integers(0, len(Sp), size=len(Sp))
        npos = np.clip(Sp[k] + rng.normal(0, 0.01, (len(k), 2)), sim.R, 1 - sim.R)
        add("nudge", npos, Sv[k] + rng.normal(0, 0.06, (len(k), 2)), Sr[k])

        def visited():
            """States the model itself reaches, labelled exactly."""
            net.eval()
            n = min(a.rooms, 6000)
            r = rng.choice(a.rooms, n, replace=False)
            p, v = tens(p0[r], v0[r], dev=dev)
            rt = torch.as_tensor(r, device=dev)
            ps, vs, rs = [], [], []
            live = torch.ones(n, dtype=torch.bool, device=dev)
            with torch.no_grad():
                for _ in range(200):
                    if not live.any():
                        break
                    ps.append(p[live].cpu().numpy()); vs.append(v[live].cpu().numpy()); rs.append(r[live.cpu().numpy()])
                    y = net(p[live], v[live], tb[rt[live]], tm[rt[live]])
                    rest = y[:, 4] > 0
                    p[live] = (p[live] + y[:, :2] * DP_SCALE).clamp(0.0, 1.0)
                    v[live] = (v[live] + y[:, 2:4] * DV_SCALE).clamp(-4, 4)
                    idx = live.nonzero()[:, 0]; live[idx[rest]] = False
            net.train()
            P_, V_, R_ = np.concatenate(ps).astype(np.float64), np.concatenate(vs).astype(np.float64), np.concatenate(rs)
            # keep only states a real ball could be in: inside the room, not inside a box, sane speed
            d = np.abs(P_[:, None] - boxes[R_, :, :2]) - boxes[R_, :, 2:]
            gap = np.sqrt((np.maximum(d, 0) ** 2).sum(-1)).min(1)
            ok = (gap > sim.R - 0.006) & (np.abs(V_).max(1) < 3.5)
            add("model", P_[ok], V_[ok], R_[ok])

        def batch(pool, n):
            p, v, pn, vn, still, nr, r = pool
            i = torch.randint(0, len(p), (n,), device=dev)
            return p[i], v[i], pn[i], vn[i], still[i], nr[i], r[i]

        it = 0; t0 = time.time(); run = None; last = t0
        while time.time() - t0 < budget:
            frac = (time.time() - t0) / budget
            for g in opt.param_groups:
                g["lr"] = a.lr * (0.5 + 0.5 * np.cos(np.pi * frac)) + 1e-5
            if time.time() - last > a.refresh * 60 and 0.3 < frac < 0.9:
                visited(); last = time.time()
                print(f"  refreshed model-visited states: {len(pools['model'][0]):,}", flush=True)
            names = list(pools)
            parts = [batch(pools[n], a.batch // len(names)) for n in names]
            p, v, pn, vn, still, nr, r = [torch.cat(x) for x in zip(*parts)]
            y = net(p, v, tb[r], tm[r])
            w = 1.0 + 3.0 * nr                                  # contact states weigh more
            lp = (F.smooth_l1_loss(y[:, :2], (pn - p) / DP_SCALE, reduction="none", beta=0.05).sum(1) * w).mean()
            lv = (F.smooth_l1_loss(y[:, 2:4], (vn - v) / DV_SCALE, reduction="none", beta=0.05).sum(1) * w).mean()
            ls = F.binary_cross_entropy_with_logits(y[:, 4], still)
            loss = lp + lv + 0.3 * ls
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            run = loss.item() if run is None else 0.99 * run + 0.01 * loss.item()
            it += 1
            if it % 200 == 0:
                print(f"it {it} loss {run:.4f} pos {lp.item():.4f} vel {lv.item():.4f} rest {ls.item():.4f} {time.time() - t0:.0f}s", flush=True)

    path = os.path.join(a.out, f"{a.what}.pt")
    torch.save({"model": net.state_dict(), "cfg": net.cfg, "what": a.what, "plat": [lo, hi],
                "minutes": a.minutes, "steps": it}, path)
    print(f"saved {path} after {it} steps", flush=True)


if __name__ == "__main__":
    main()
