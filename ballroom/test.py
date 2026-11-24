"""Score the models on fresh rooms against the exact simulator.

  python3 -m ballroom.test            # rooms like training (2-3) and harder (4-5)
"""
import argparse, json, os
import numpy as np
import torch
from . import sim
from .model import Net, roll


def support(p, boxes):
    """Which box the resting ball sits on (index), -1 if none found."""
    top = boxes[:, :, 1] + boxes[:, :, 3] + sim.R
    on = (np.abs(p[:, None, 1] - top) < 0.012) & (np.abs(p[:, None, 0] - boxes[:, :, 0]) < boxes[:, :, 2] + sim.R)
    return np.where(on.any(1), on.argmax(1), -1)


def load(path, dev):
    ck = torch.load(path, map_location=dev)
    net = Net(**ck["cfg"]).to(dev); net.load_state_dict(ck["model"]); net.eval()
    return net


def score(name, pred, fin, boxes):
    d = np.sqrt(((pred - fin) ** 2).sum(1))
    same = support(pred, boxes) == support(fin, boxes)
    return dict(model=name, mean=float(d.mean()), median=float(np.median(d)),
                within_ball=float((d < 2 * sim.R).mean() * 100), right_surface=float(same.mean() * 100))


def vote(ends):
    """The cloud votes. ends [C,N,2]: C nudged copies per room. Answer = the middle of the biggest clump
    (the copies within a ball width of the copy with the most neighbours); sure = that clump's share of the cloud."""
    e = torch.as_tensor(ends).permute(1, 0, 2)                          # [N,C,2]
    close = torch.cdist(e, e) < 2 * sim.R                               # [N,C,C]
    pick = close.sum(2).argmax(1)                                       # the copy with the most neighbours
    w = close[torch.arange(e.shape[0], device=e.device), pick].float()  # [N,C] its clump
    return (e * w[:, :, None]).sum(1) / w.sum(1, keepdim=True), w.mean(1)


def nudged(p0, v0, copies, rng):
    """The same balls started a hair away (0.002 in position, 0.02 in speed), as the editor's cloud does."""
    p = np.clip(p0[None] + 0.002 * rng.standard_normal((copies,) + p0.shape), sim.R, 1 - sim.R)
    return p, v0[None] + 0.02 * rng.standard_normal((copies,) + v0.shape)


def sure_line(name, pred, sure, fin):
    hit = np.sqrt(((pred - fin) ** 2).sum(1)) < 2 * sim.R
    parts = []
    for lo, hi, word in ((0.75, 1.01, "sure (3/4 of the cloud agrees)"), (0.4, 0.75, "unsure"), (0.0, 0.4, "split")):
        m = (sure >= lo) & (sure < hi)
        if m.any():
            parts.append(f"{word}: {100 * m.mean():.0f}% of rooms, hits {100 * hit[m].mean():.0f}%")
    return f"  {name}  " + "; ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="ballroom/runs")
    ap.add_argument("--rooms", type=int, default=3000)
    ap.add_argument("--cloud", type=int, default=24 if torch.cuda.is_available() else 0, help="nudged copies per room that vote (0 = off)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = a.device
    # one step model in the folder itself, or several arms in sub-folders
    steps = [("", os.path.join(a.runs, "step.pt"))] if os.path.exists(os.path.join(a.runs, "step.pt")) else []
    steps += [(d, os.path.join(a.runs, d, "step.pt")) for d in sorted(os.listdir(a.runs))
              if os.path.exists(os.path.join(a.runs, d, "step.pt"))]
    res = {}
    for label, plat, seed in (("2-3 platforms (like training)", (2, 3), 101), ("4-5 platforms (never seen)", (4, 5), 102)):
        rng = np.random.default_rng(seed)
        say = lambda what: print(f"[{label.split(' (')[0]}] {what}...", flush=True)   # so a long test never looks frozen
        say(f"simulating {a.rooms} rooms")
        boxes, count, p0, v0 = sim.make_rooms(a.rooms, rng, plat)
        P, V, stop = sim.rollout(boxes, p0, v0)
        fin = P[-1]
        mask = np.zeros((a.rooms, sim.MAX_BOX), dtype=bool); mask[:, :4] = True
        for i in range(a.rooms):
            mask[i, 4:4 + count[i]] = True
        tp, tv, tb = [torch.as_tensor(x, dtype=torch.float32, device=dev) for x in (p0, v0, boxes)]
        tm = torch.as_tensor(mask, device=dev)
        rows = []; notes = []
        C = a.cloud; rep = lambda x: x.repeat(C, *[1] * (x.dim() - 1))
        if C:                                   # the ceiling: the exact simulator's own cloud, voting on the un-nudged answer
            say(f"simulating {C * a.rooms} nudged balls for the ceiling row")
            cp, cv = nudged(p0, v0, C, rng)
            ends = sim.rollout(np.tile(boxes, (C, 1, 1)), cp.reshape(-1, 2), cv.reshape(-1, 2))[0][-1].reshape(C, a.rooms, 2)
            pred, sure = vote(ends)
            rows.append(score("exact simulator, cloud vote (ceiling)", pred.numpy(), fin, boxes))
            notes.append(sure_line("exact simulator:", pred.numpy(), sure.numpy(), fin))
            cpt, cvt = [torch.as_tensor(x.reshape(-1, 2), dtype=torch.float32, device=dev) for x in (cp, cv)]
        drop = p0.copy(); drop[:, 1] = sim.R
        rows.append(score("falls straight down (no model)", drop, fin, boxes))
        f = os.path.join(a.runs, "shot.pt")
        if os.path.exists(f):
            with torch.no_grad():
                pred = (tp + load(f, dev)(tp, tv, tb, tm) * 0.3).cpu().numpy()
            rows.append(score("one shot (intuition)", pred, fin, boxes))
        for name, f in steps:
            net = load(f, dev); tag = f" [{name}]" if name else " (law)"
            say("rolling step model" + tag)
            pred = roll(net, tp, tv, tb, tm).cpu().numpy()
            rows.append(score("step by step" + tag, pred, fin, boxes))
            if C:                               # the same model, asked 24 times from a hair away; the cloud votes
                say(f"rolling {C * a.rooms} nudged copies for the cloud vote" + tag)
                with torch.no_grad():
                    ends = torch.cat([roll(net, cpt[i:i + 20000], cvt[i:i + 20000], rep(tb)[i:i + 20000], rep(tm)[i:i + 20000])
                                      for i in range(0, C * a.rooms, 20000)]).view(C, a.rooms, 2)
                pred, sure = vote(ends)
                rows.append(score("cloud vote" + tag, pred.cpu().numpy(), fin, boxes))
                notes.append(sure_line("cloud vote" + tag + ":", pred.cpu().numpy(), sure.cpu().numpy(), fin))
        res[label] = rows
        print(f"\n== {label}, {a.rooms} fresh rooms ==")
        print(f"{'':40s}{'mean miss':>10s}{'median':>9s}{'within a ball':>15s}{'right surface':>15s}")
        for r in rows:
            print(f"{r['model']:40s}{r['mean']:10.3f}{r['median']:9.3f}{r['within_ball']:14.1f}%{r['right_surface']:14.1f}%")
        if notes:
            print(f"when the cloud of {C} agrees, is it right?")
            print("\n".join(notes))
    json.dump(res, open(os.path.join(a.runs, "test.json"), "w"), indent=1)
    if steps:   # the arm that does best on the rooms it never saw goes to the editor
        hard = [r for r in list(res.values())[-1] if r["model"].startswith("step by step")]
        top = max(r["within_ball"] for r in hard)   # among arms within a point of the best, the smallest: the editor runs it live
        best = min((i for i in range(len(hard)) if hard[i]["within_ball"] >= top - 1.0), key=lambda i: os.path.getsize(steps[i][1]))
        open(os.path.join(a.runs, "best.txt"), "w").write(steps[best][1])
        print(f"\nbest step model on unseen rooms: {steps[best][1]}")


if __name__ == "__main__":
    main()
