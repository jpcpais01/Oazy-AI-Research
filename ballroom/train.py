"""Train the step model ("law") and the one-shot model ("intuition").

  python3 -m ballroom.train --what step --minutes 20
  python3 -m ballroom.train --what shot --minutes 10

Train rooms have 2-3 platforms by default; ballroom.test scores on 4-5.

The step model learns one model step (1/30 s) of the exact simulator, in a live
arena on the training device (see train_step): no stored data set, every step
is new, and part of the balls follow the model's own predictions so it learns
to recover from its own mistakes.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from . import sim
from .model import step_model, shot_model, roll, DP_SCALE, DV_SCALE


def tens(*a, dev):
    return [torch.as_tensor(x, dtype=torch.float32, device=dev) for x in a]


def train_step(a, net, opt, boxes, mask, budget, dev):
    """Train the step model in a live arena: thousands of balls roll at once, on the training device.

    Every step each ball is one example: its state now, and the exact simulator's state 1/30 s later.
    Then the ball moves on. Some balls follow the simulator; the others follow the MODEL's own
    prediction, so the model keeps meeting (and being corrected on) the places its mistakes take it.
    A ball that has rested, flown too long or been thrown somewhere impossible starts again in a new room.
    """
    f64 = torch.float64
    tb = torch.as_tensor(boxes, dtype=f64, device=dev); tb32 = tb.float()
    tm = torch.as_tensor(mask, device=dev)
    n, rooms = a.arena, boxes.shape[0]
    g = torch.Generator(device=dev); g.manual_seed(a.seed)
    rnd = lambda *shape: torch.rand(*shape, generator=g, device=dev, dtype=f64)
    r = torch.zeros(n, dtype=torch.long, device=dev)
    p = torch.zeros(n, 2, dtype=f64, device=dev); v = torch.zeros(n, 2, dtype=f64, device=dev)
    age = torch.zeros(n, dtype=torch.long, device=dev); rested = torch.zeros_like(age)
    by_model = torch.zeros(n, dtype=torch.bool, device=dev)

    def gap_to(pp, rr):
        d = (pp[:, None] - tb[rr, :, :2]).abs() - tb[rr, :, 2:]
        return d.clamp(min=0).norm(dim=-1).amin(1)

    def fresh(which, share):
        """Restart the chosen balls in a new room, at a free spot. Never waits on the device:
        a ball that draws three bad spots in a row simply tries again next step."""
        nonlocal r, p, v, age, rested, by_model
        nr = torch.randint(0, rooms, (n,), generator=g, device=dev)
        spot = p; found = torch.zeros(n, dtype=torch.bool, device=dev)
        for _ in range(3):
            cand = sim.R + 0.01 + rnd(n, 2) * (1 - 2 * sim.R - 0.02)
            good = (gap_to(cand, nr) >= sim.R + 0.01) & ~found
            spot = torch.where(good[:, None], cand, spot); found = found | good
        do = which & found
        ang = rnd(n) * 2 * np.pi; sp = rnd(n) * 1.5
        nv = torch.stack([sp * torch.cos(ang), sp * torch.sin(ang)], 1)
        r = torch.where(do, nr, r)
        p = torch.where(do[:, None], spot, p); v = torch.where(do[:, None], nv, v)
        age = torch.where(do, torch.zeros_like(age), age); rested = torch.where(do, torch.zeros_like(rested), rested)
        by_model = torch.where(do, rnd(n) < share, by_model)
        return do

    placed = torch.zeros(n, dtype=torch.bool, device=dev)
    for _ in range(8):
        placed = placed | fresh(~placed, 0.0)
    # spread the starting ages so the first batches are not all "just launched"
    for _ in range(40):
        pn, vn = sim.step_t(p, v, tb[r], sim.SUB)
        go = rnd(n) < 0.7
        p = torch.where(go[:, None], pn, p); v = torch.where(go[:, None], vn, v)

    it = 0; t0 = time.time(); run = None
    while (it < a.iters) if a.iters else (time.time() - t0 < budget):
        frac = it / a.iters if a.iters else (time.time() - t0) / budget
        for gr in opt.param_groups:
            gr["lr"] = a.lr * (0.5 + 0.5 * np.cos(np.pi * frac)) + 1e-5
        share = 0.5 * min(1.0, max(0.0, (frac - 0.15) / 0.25))   # balls that follow the model: 0 early, half later
        pn, vn = sim.step_t(p, v, tb[r], sim.SUB)
        still = (pn - p).abs().amax(1) < 1e-9
        vn = torch.where(still[:, None], torch.zeros_like(vn), vn)
        near = gap_to(p, r) < sim.R + 0.05
        y = net(p.float(), v.float(), tb32[r], tm[r])
        w = 1.0 + 3.0 * near                                    # contact states weigh more
        lp = (F.smooth_l1_loss(y[:, :2], ((pn - p) / DP_SCALE).float(), reduction="none", beta=0.05).sum(1) * w).mean()
        lv = (F.smooth_l1_loss(y[:, 2:4], ((vn - v) / DV_SCALE).float(), reduction="none", beta=0.05).sum(1) * w).mean()
        ls = F.binary_cross_entropy_with_logits(y[:, 4], still.float())
        loss = lp + lv + 0.3 * ls
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        it += 1
        # move every ball on
        with torch.no_grad():
            yd = y.detach().to(f64)
            follow = by_model & ~still & (yd[:, 4] <= 0)
            pm = (p + yd[:, :2] * DP_SCALE).clamp(0.0, 1.0); vm = v + yd[:, 2:4] * DV_SCALE
            p = torch.where(follow[:, None], pm, pn); v = torch.where(follow[:, None], vm, vn)
            age += 1; rested += still
            lost = follow & ((gap_to(p, r) < sim.R - 0.006) | (v.abs().amax(1) > 3.5))
            fresh((rested >= 3) | (age > 400) | lost, share)
        if it % 250 == 0:
            li = loss.item(); run = li if run is None else 0.7 * run + 0.3 * li
            print(f"it {it} loss {run:.4f} pos {lp.item():.4f} vel {lv.item():.4f} rest {ls.item():.4f} "
                  f"model-led {by_model.float().mean().item() * 100:.0f}% {time.time() - t0:.0f}s "
                  f"({it * n / (time.time() - t0) / 1e6:.2f}M examples/s)", flush=True)
    return it


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
    ap.add_argument("--feat", type=int, default=0, help="0 raw coordinates, 1 + distance past each face, 2 + nearest point and gap, 3 + facing, magnified gap, approach speed")
    ap.add_argument("--iters", type=int, default=0, help="stop after this many steps instead of by the clock")
    ap.add_argument("--arena", type=int, default=0, help="balls rolling at once = examples per step (step model)")
    ap.add_argument("--out", default="ballroom/runs")
    ap.add_argument("--init", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = a.device
    a.arena = a.arena or (16384 if dev.startswith("cuda") else 2048)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    lo, hi = map(int, a.plat.split(","))
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    boxes, count, p0, v0 = sim.make_rooms(a.rooms, rng, (lo, hi))
    mask = np.zeros((a.rooms, sim.MAX_BOX), dtype=bool)
    mask[:, :4] = True
    for i in range(a.rooms):
        mask[i, 4:4 + count[i]] = True
    print(f"rooms {a.rooms} ({lo}-{hi} platforms)  {time.time() - t0:.0f}s", flush=True)
    tb, tm = tens(boxes, dev=dev)[0], torch.as_tensor(mask, device=dev)

    make = step_model if a.what == "step" else shot_model
    net = make(d=a.d, layers=a.layers, feat=a.feat).to(dev)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location=dev)["model"])
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-4)
    budget = a.minutes * 60
    print(f"{a.what} model {sum(p.numel() for p in net.parameters()):,} weights on {dev}", flush=True)

    if a.what == "shot":
        P, _, stop = sim.rollout(boxes, p0, v0)
        fin = P[stop, np.arange(a.rooms)]
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
            it += 1
            if it % 200 == 0:
                run = loss.item() if run is None else 0.7 * run + 0.3 * loss.item()
                print(f"it {it} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
    else:
        it = train_step(a, net, opt, boxes, mask, budget, dev)

    path = os.path.join(a.out, f"{a.what}.pt")
    torch.save({"model": net.state_dict(), "cfg": net.cfg, "what": a.what, "plat": [lo, hi],
                "minutes": a.minutes, "steps": it}, path)
    print(f"saved {path} after {it} steps", flush=True)


if __name__ == "__main__":
    main()
