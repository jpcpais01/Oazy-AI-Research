"""Score a saved mind: N fresh lives in each kind of world, acting on habit.

    python3 -m survive.bench /workspace/oazy_runs/survive/clean/mind_60min.pt [--lives 1024]

Prints and saves (next to the mind) steps alive, food per life by kind, share of lives that
reach the 100 s cap, and causes of death, for test (unseen mix), hard and train-like worlds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .model import Mind
from .train import Actor
from .world import DEATH_CAUSES, MAX_STEPS, World

KINDS = ["cube", "stick", "tile"]


@torch.no_grad()
def bench_mode(mind, res, dev, mode, lives, chunk=256, seed=777, bf16=True):
    steps_all, food_all, cause_all, cap_all = [], [], [], []
    for s in range(0, lives, chunk):
        n = min(chunk, lives - s)
        w = World(n, dev, mode=mode, seed=seed + s)
        act = Actor(mind, w, res, dev, deterministic=True)
        alive = torch.ones(n, dtype=torch.bool, device=dev)
        steps = torch.zeros(n, device=dev)
        food = torch.zeros(n, 3, device=dev)
        cause = torch.zeros(n, dtype=torch.long, device=dev)
        for _ in range(MAX_STEPS):
            a, _, _ = act.observe_act(bf16)
            act.advance(a)
            done, trunc, c, ate = w.step(a)
            steps += alive.float()
            got = alive & (ate >= 0)
            food[got, ate[got]] += 1
            cause = torch.where(alive & done, c, cause)
            alive &= ~(done | trunc)
            if not alive.any():
                break
        steps_all.append(steps); food_all.append(food); cause_all.append(cause)
        cap_all.append(steps >= MAX_STEPS)
    steps, food = torch.cat(steps_all), torch.cat(food_all)
    cause, cap = torch.cat(cause_all), torch.cat(cap_all)
    dead = cause > 0
    nd = max(int(dead.sum()), 1)
    return dict(lives=lives, steps=round(steps.mean().item(), 1), food=round(food.sum(1).mean().item(), 2),
                food_by_kind={k: round(v, 2) for k, v in zip(KINDS, food.mean(0).tolist())},
                cap_pct=round(100 * cap.float().mean().item(), 1),
                deaths_pct={DEATH_CAUSES[i]: round(100 * (cause == i).sum().item() / nd, 1) for i in range(1, len(DEATH_CAUSES))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mind")
    ap.add_argument("--lives", type=int, default=1024)
    ap.add_argument("--modes", default="test,hard,train")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.mind, map_location=dev)
    res = ck["args"]["res"]
    mind = Mind(res).to(dev)
    mind.load_state_dict(ck["mind"])
    mind.eval()
    out = dict(mind=a.mind, minutes=round(ck.get("minutes", 0), 1), frames=ck.get("frames"))
    for mode in a.modes.split(","):
        out[mode] = bench_mode(mind, res, dev, mode, a.lives)
        print(mode, json.dumps(out[mode]), flush=True)
    p = Path(a.mind).with_suffix(".bench.json")
    p.write_text(json.dumps(out, indent=1))
    print("saved", p)


if __name__ == "__main__":
    main()
