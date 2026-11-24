"""Ballroom Robot: a cart on the floor with a cannon. It looks at a picture of the room with a target
marked, then drives, aims and fires so the ball comes to rest on the target.

  python3 -m ballroom.robot --minutes 20          # train, then score on fresh rooms (pod: bash runpod/robot.sh)

The body is imperfect, like a real machine: every throw is off by a little noise in the cart's
position, the cannon's angle and the launch speed (--noise, relative; 0.0001 = 0.01%).

How it learns (not next-token, not trial and error):
  hindsight   every random throw is a correct lesson: "to land there, do this". The exact simulator
              gives as many as we like, noise included.
  intuition   a pixel network (the same kind of eye as ballroom.eye) reads the picture and the target
              and proposes 16 different throws, each with a score.
  imagination before firing, each proposal is imagined several times with fresh noise, and the robot
              fires the one that lands on the target most often. Imagination is either the exact
              simulator (the ceiling) or the learned pixel world model from ballroom.eye.
"""
import argparse, json, math, os, time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import sim
from .eye import Eye, render, roll as eye_roll
from .test import vote

LO = np.array([0.06, math.radians(15), 0.5])     # cart x, cannon angle, launch speed
HI = np.array([0.94, math.radians(165), 2.4])
PIVOT = 0.06                                      # height of the cannon's pivot above the floor
BARREL = 0.06                                     # the ball leaves the barrel this far from the pivot


def launch(act, noise, rng):
    """Throws in real units [N,3] -> start position and speed, each part off by its own noise."""
    act = np.array(act, dtype=np.float64)
    if noise:
        act[:, 0] += noise * rng.standard_normal(len(act))                 # the room is 1 wide
        act[:, 1:] *= 1 + noise * rng.standard_normal((len(act), 2))
    d = np.stack([np.cos(act[:, 1]), np.sin(act[:, 1])], 1)
    p0 = np.stack([act[:, 0], np.full(len(act), PIVOT)], 1) + BARREL * d
    return p0, act[:, 2:3] * d


def _land(args):
    boxes, p0, v0 = args
    P, _, stop = sim.rollout(boxes, p0, v0)
    return P[-1], stop < P.shape[0] - 1


_pool = None


def land(boxes, p0, v0, chunk=4000):
    """The exact simulator, run to rest on every CPU core. Returns resting points [N,2] and rested [N]."""
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(max(1, min(16, (os.cpu_count() or 2) - 1)), mp_context=mp.get_context("spawn"))
    parts = list(_pool.map(_land, [(boxes[i:i + chunk], p0[i:i + chunk], v0[i:i + chunk]) for i in range(0, len(p0), chunk)]))
    return np.concatenate([p for p, _ in parts]), np.concatenate([r for _, r in parts])


def throws(n, plat, noise, seed):
    """Random rooms, one random throw each (with the body's noise), and where it came to rest."""
    rng = np.random.default_rng(seed)
    boxes, count, _, _ = sim.make_rooms(n, rng, plat)
    act = LO + rng.random((n, 3)) * (HI - LO)
    fin, ok = land(boxes, *launch(act, noise, rng))
    return boxes[ok], act[ok], fin[ok]


class Aimer(nn.Module):
    def __init__(self, res=128, patch=16, d=192, enc=4, heads=4, reg=8, m=16):
        super().__init__()
        self.cfg = dict(res=res, patch=patch, d=d, enc=enc, heads=heads, reg=reg, m=m)
        self.res, self.m = res, m
        self.eye = Eye(res, patch, d, enc, 1, heads, reg)              # only its seeing part is used
        self.head = nn.Sequential(nn.Linear(reg * d, 512), nn.GELU(), nn.Linear(512, m * 4))

    def forward(self, target, boxes):
        """Target [N,2] and rooms [N,9,4] -> m proposed throws in [0,1]^3 and their scores."""
        z = self.eye.see(render(target, target, boxes, self.res))       # picture: target, target, platforms
        y = self.head(z.flatten(1)).view(-1, self.m, 4)
        return y[..., :3].sigmoid(), y[..., 3]


def real(u):
    return LO + u * (HI - LO)


def imagine_physics(boxes, act, noise, rng):
    return land(boxes, *launch(act, noise, rng))[0]


def imagine_pixels(world, dev, res):
    def imagine(boxes, act, noise, rng):
        p0, v0 = launch(act, noise, rng)
        p1 = p0 + v0 / 30; p1[:, 1] -= 0.5 * sim.G / 30 ** 2              # where the ball is 1/30 s after it leaves
        out = []
        with torch.no_grad():
            for i in range(0, len(p0), 2000):
                t = lambda x: torch.as_tensor(x[i:i + 2000], dtype=torch.float32, device=dev)
                out.append(eye_roll(world, world.see(render(t(p0), t(p1), t(boxes), res))).cpu().numpy())
        return np.concatenate(out)
    return imagine


def choose(cand, score, boxes, target, imagine, noise, copies, rng):
    """cand [N,M,3] real throws. Imagine each `copies` times with fresh noise; fire the one that lands on target most."""
    N, M, _ = cand.shape
    if imagine is None:
        return cand[np.arange(N), score.argmax(1)], np.full(N, np.nan)
    A = np.tile(cand.reshape(N * M, 3), (copies, 1))
    ends = imagine(np.tile(np.repeat(boxes, M, 0), (copies, 1, 1)), A, noise, rng).reshape(copies, N, M, 2)
    hit = (np.linalg.norm(ends - target[None, :, None], axis=-1) < 2 * sim.R).mean(0)      # [N,M]
    pick = (hit + 1e-3 * score).argmax(1)                                                   # ties: the intuition decides
    return cand[np.arange(N), pick], hit[np.arange(N), pick]


def evaluate(net, world, dev, a):
    net.eval(); res = {}
    for label, plat, seed in (("2-3 platforms (like training)", (2, 3), 201), ("4-5 platforms (never seen)", (4, 5), 202)):
        for noise in sorted(set(a.test_noise)):
            rng = np.random.default_rng(seed)
            boxes, _, target = throws(a.test_rooms, plat, noise, seed)            # targets: places a real throw can rest
            N = len(target)
            with torch.no_grad():
                t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev)
                parts = [net(t(target[i:i + 500]), t(boxes[i:i + 500])) for i in range(0, N, 500)]
            cand = real(torch.cat([c for c, _ in parts]).cpu().numpy().astype(np.float64)); score = torch.cat([s for _, s in parts]).cpu().numpy()
            blind = real(rng.random((N, net.m, 3)))
            arms = [("random throw (no robot)", blind[:, :1], np.zeros((N, 1)), None),
                    ("intuition only (best scored proposal)", cand, score, None)]
            if world is not None:
                arms.append(("intuition + imagined with the pixel model", cand, score, imagine_pixels(world, dev, world.res)))
            arms += [("intuition + imagined with exact physics", cand, score, imagine_physics),
                     (f"{net.m} random throws + exact physics", blind, np.zeros((N, net.m)), imagine_physics)]
            rows = []
            print(f"\n== {label}, {N} fresh rooms and targets, body noise {100 * noise:g}% ==", flush=True)
            print(f"{'':48s}{'lands on target':>16s}{'mean miss':>11s}{'it expected':>13s}")
            for name, c, s, im in arms:
                act, expect = choose(c, s, boxes, target, im, noise, a.copies, rng)
                ends = land(np.tile(boxes, (a.fire, 1, 1)), *launch(np.tile(act, (a.fire, 1)), noise, rng))[0].reshape(a.fire, N, 2)
                miss = np.linalg.norm(ends - target[None], axis=-1)       # each chosen throw fired a.fire times for real
                r = dict(arm=name, noise=noise, hit=float(100 * (miss < 2 * sim.R).mean()), mean_miss=float(miss.mean()),
                         expected=None if np.isnan(expect).all() else float(100 * np.nanmean(expect)))
                rows.append(r)
                print(f"{name:48s}{r['hit']:15.1f}%{r['mean_miss']:11.3f}" + (f"{r['expected']:12.1f}%" if r["expected"] is not None else f"{'':>13s}"), flush=True)
            res[f"{label}, noise {noise:g}"] = rows
    net.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--iters", type=int, default=0)
    ap.add_argument("--throws", type=int, default=400000, help="hindsight lessons simulated before training")
    ap.add_argument("--plat", default="2,3")
    ap.add_argument("--noise", type=float, default=0.0001, help="the body's imperfection while learning (relative)")
    ap.add_argument("--test-noise", type=float, nargs="*", default=[0.0001, 0.01])
    ap.add_argument("--batch", type=int, default=512 if torch.cuda.is_available() else 32)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--enc", type=int, default=4)
    ap.add_argument("--m", type=int, default=16, help="proposals per target")
    ap.add_argument("--copies", type=int, default=8, help="imagined throws per proposal")
    ap.add_argument("--fire", type=int, default=8, help="real throws per chosen action, to measure how often it lands")
    ap.add_argument("--test-rooms", type=int, default=1000)
    ap.add_argument("--world", default="", help="a pixel world model (ballroom.eye) to imagine with")
    ap.add_argument("--out", default="ballroom/runs/robot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = a.device
    torch.manual_seed(a.seed); os.makedirs(a.out, exist_ok=True)
    lo, hi = map(int, a.plat.split(","))
    t0 = time.time()
    boxes, act, fin = throws(a.throws, (lo, hi), a.noise, a.seed)
    print(f"{len(fin):,} hindsight lessons from {a.throws:,} random throws ({lo}-{hi} platforms, body noise {100 * a.noise:g}%) "
          f"in {time.time() - t0:.0f}s; {100 * (fin[:, 1] > 0.1).mean():.0f}% came to rest on a platform", flush=True)
    t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev)
    boxes, u, fin = t(boxes), t((act - LO) / (HI - LO)), t(fin)
    net = Aimer(a.res, a.patch, a.d, a.enc, m=a.m).to(dev)
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=0.01)
    print(f"aimer {sum(p.numel() for p in net.parameters()):,} weights on {dev}, {a.m} proposals per target", flush=True)
    budget = a.minutes * 60; it = 0; t0 = time.time()
    while (it < a.iters) if a.iters else (time.time() - t0 < budget):
        frac = it / a.iters if a.iters else (time.time() - t0) / budget
        lr = a.lr * min(1.0, (it + 1) / 500) * (0.5 + 0.5 * np.cos(np.pi * min(1.0, frac))) + 1e-5
        for g in opt.param_groups:
            g["lr"] = lr
        r = torch.randint(0, len(fin), (a.batch,), device=dev)
        cand, score = net(fin[r], boxes[r])
        err = (cand - u[r][:, None]).abs().sum(2)                       # [B,M] how far each proposal is from the throw that got there
        best = err.argmin(1)                                            # many throws reach one place: only the nearest proposal learns it
        loss = err.gather(1, best[:, None]).mean() + 0.02 * err.mean() + 0.2 * F.cross_entropy(score, best)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        it += 1
        if it % 100 == 0:
            el = time.time() - t0
            print(f"it {it} loss {loss.item():.4f} nearest proposal off by {err.min(1).values.mean().item():.3f} "
                  f"{el:.0f}s ({it / el:.1f} steps/s)", flush=True)
    torch.save({"model": net.state_dict(), "cfg": net.cfg, "plat": [lo, hi], "noise": a.noise, "steps": it}, os.path.join(a.out, "robot.pt"))
    print(f"saved {a.out}/robot.pt after {it} steps", flush=True)
    world = None
    if a.world and os.path.exists(a.world):
        ck = torch.load(a.world, map_location=dev)
        world = Eye(**ck["cfg"]).to(dev); world.load_state_dict(ck["model"]); world.eval()
        print(f"imagining with the pixel world model {a.world}", flush=True)
    res = evaluate(net, world, dev, a)
    json.dump(res, open(os.path.join(a.out, "robot.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
