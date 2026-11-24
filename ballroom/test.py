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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="ballroom/runs")
    ap.add_argument("--rooms", type=int, default=3000)
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
        boxes, count, p0, v0 = sim.make_rooms(a.rooms, rng, plat)
        P, V, stop = sim.rollout(boxes, p0, v0)
        fin = P[-1]
        mask = np.zeros((a.rooms, sim.MAX_BOX), dtype=bool); mask[:, :4] = True
        for i in range(a.rooms):
            mask[i, 4:4 + count[i]] = True
        tp, tv, tb = [torch.as_tensor(x, dtype=torch.float32, device=dev) for x in (p0, v0, boxes)]
        tm = torch.as_tensor(mask, device=dev)
        rows = []
        drop = p0.copy(); drop[:, 1] = sim.R
        rows.append(score("falls straight down (no model)", drop, fin, boxes))
        f = os.path.join(a.runs, "shot.pt")
        if os.path.exists(f):
            with torch.no_grad():
                pred = (tp + load(f, dev)(tp, tv, tb, tm) * 0.3).cpu().numpy()
            rows.append(score("one shot (intuition)", pred, fin, boxes))
        for name, f in steps:
            pred = roll(load(f, dev), tp, tv, tb, tm).cpu().numpy()
            rows.append(score("step by step" + (f" [{name}]" if name else " (law)"), pred, fin, boxes))
        res[label] = rows
        print(f"\n== {label}, {a.rooms} fresh rooms ==")
        print(f"{'':40s}{'mean miss':>10s}{'median':>9s}{'within a ball':>15s}{'right surface':>15s}")
        for r in rows:
            print(f"{r['model']:40s}{r['mean']:10.3f}{r['median']:9.3f}{r['within_ball']:14.1f}%{r['right_surface']:14.1f}%")
    json.dump(res, open(os.path.join(a.runs, "test.json"), "w"), indent=1)
    if steps:   # the arm that does best on the rooms it never saw goes to the editor
        hard = [r for r in list(res.values())[-1] if r["model"].startswith("step by step")]
        top = max(r["within_ball"] for r in hard)   # among arms within a point of the best, the smallest: the editor runs it live
        best = min((i for i in range(len(hard)) if hard[i]["within_ball"] >= top - 1.0), key=lambda i: os.path.getsize(steps[i][1]))
        open(os.path.join(a.runs, "best.txt"), "w").write(steps[best][1])
        print(f"\nbest step model on unseen rooms: {steps[best][1]}")


if __name__ == "__main__":
    main()
