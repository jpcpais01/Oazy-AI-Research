"""Sleep: does a grown mind get better at a world it never lived in, by sleeping on its days?

    python3 -m survive.sleep /workspace/oazy_runs/survive/ref/mind_120min.pt --mode hard --days 8

A population of creatures (one shared mind) is dropped into new worlds. Life goes in days and
nights:
  day    60 s of living with the mind's weights FROZEN; every moment goes into that day's diary
  night  the creatures are asleep (the world is paused); the mind replays the day's diary into
         its weights: the same world-model + dream learning as in training, nothing else.
         The diary is cleared before the next day (like a hippocampus emptying overnight).
Arms (same start mind, same worlds, same seeds):
  awake    never sleeps (frozen): the baseline
  sleep    sleeps, replays the day uniformly
  emotion  sleeps, replays mostly the moments that mattered (surprise x importance)
Prints and saves, per day: average life length of lives that ended that day, food per life,
causes of death. At the end each arm is also tested in ordinary unseen worlds, to see whether
learning the new place made it forget the old one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .bench import bench_mode
from .model import Mind
from .train import METRICS, Actor, Learner, Replay
from .world import DEATH_CAUSES, World


def run_arm(arm, ck, a, dev):
    torch.manual_seed(a.seed)
    res = ck["args"]["res"]
    mind = Mind(res).to(dev)
    mind.load_state_dict(ck["mind"])
    args = SimpleNamespace(**{**ck["args"], "compile": a.compile, "compile_mode": "default"})
    args.vic_rows = getattr(args, "vic_rows", 2048)
    learner = Learner(mind, args)
    world = World(a.envs, dev, mode=a.mode, seed=a.seed + 1, compile=bool(a.compile))
    actor = Actor(mind, world, res, dev, compile=bool(a.compile))
    rb = Replay(a.day, a.envs, res, dev, emotion=0.5 if arm == "emotion" else 0.0)
    days = []
    for d in range(a.days):
        mind.eval()
        n = torch.zeros((), device=dev); t = torch.zeros((), device=dev)
        food = torch.zeros((), device=dev); cause = torch.zeros(len(DEATH_CAUSES), device=dev)
        rb.n = rb.ptr = 0
        rb.prev_wb = None
        for _ in range(a.day):
            act, rec, _ = actor.observe_act(args.bf16)
            rb.add(*rec)
            st = actor.env_step(act)
            fin = (st["dead"] | st["trunc"]).float()
            n += fin.sum(); t += (st["t"].float() * fin).sum(); food += (st["food"].sum(-1) * fin).sum()
            cc = torch.where(st["dead"], st["cause"], torch.zeros_like(st["cause"]))
            cause += torch.zeros(len(DEATH_CAUSES), device=dev).index_add_(0, cc, fin)
        nn_ = max(n.item(), 1)
        row = dict(day=d + 1, lives=int(n.item()), life_steps=round(t.item() / nn_, 1), food=round(food.item() / nn_, 2),
                   deaths={DEATH_CAUSES[i]: int(v) for i, v in enumerate(cause.tolist()) if i})
        if arm != "awake":
            mind.train()
            acc = torch.zeros(len(METRICS), device=dev)
            for _ in range(a.night):
                met, _, kl = learner.update(rb.sample(args.batch, args.seq))
                if rb.emotion > 0:
                    rb.feel(kl)
                acc += met
            row["night"] = {k: round(v / a.night, 4) for k, v in zip(METRICS, acc.tolist()) if k in ("sen", "kl", "wb", "ent")}
        print(arm, json.dumps(row), flush=True)
        days.append(row)
    mind.eval()
    after = {m: bench_mode(mind, res, dev, m, a.test_lives) for m in (a.mode, "test")}
    print(arm, "after", json.dumps(after), flush=True)
    return dict(days=days, after=after)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mind")
    ap.add_argument("--mode", default="hard", help="the new place: hard | test | train")
    ap.add_argument("--days", type=int, default=8)
    ap.add_argument("--day", type=int, default=600, help="steps of living per day (0.1 s each)")
    ap.add_argument("--night", type=int, default=300, help="learning updates per night")
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--arms", default="awake,sleep,emotion")
    ap.add_argument("--test-lives", type=int, default=512)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ck = torch.load(a.mind, map_location=dev)
    out = dict(mind=a.mind, mode=a.mode, days=a.days, day=a.day, night=a.night, envs=a.envs)
    for arm in a.arms.split(","):
        out[arm] = run_arm(arm, ck, a, dev)
    p = Path(a.mind).with_suffix(f".sleep_{a.mode}.json")
    p.write_text(json.dumps(out, indent=1))
    print("saved", p)


if __name__ == "__main__":
    main()
