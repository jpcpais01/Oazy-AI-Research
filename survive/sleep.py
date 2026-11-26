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
  lifelong emotion sleep, and the strongest moments are not forgotten: each night the clips that
           mattered most are copied into a long-term memory (keepsakes) that survives the morning.
           Nights replay half today, half keepsakes. A keepsake fades when it is understood
           (its surprise is re-felt each time it is studied), so the memory keeps what is still
           worth learning: old dangers, rare meals. Fixes the drift of plain emotion sleep, which
           only ever sees the latest day.
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
from .render import set_eyes
from .model import load_mind, ACT, H_DIM, SENSES, Mind
from .train import METRICS, Actor, Learner, Replay
from .world import DEATH_CAUSES, World


class Keepsakes:
    """Long-term memory: C clips of T moments, each with a score = mean surprise x importance."""

    def __init__(self, C, T, res, dev):
        self.C, self.T, self.n = C, T, 0
        self.d = dict(img=torch.zeros(C, T, 6, res, res, dtype=torch.uint8, device=dev),
                      sen=torch.zeros(C, T, SENSES, dtype=torch.float16, device=dev),
                      act=torch.zeros(C, T, ACT, dtype=torch.float16, device=dev),
                      first=torch.zeros(C, T, dtype=torch.bool, device=dev),
                      cont=torch.zeros(C, T, dtype=torch.float16, device=dev),
                      h0=torch.zeros(C, H_DIM, dtype=torch.float16, device=dev))
        self.imp = torch.zeros(C, T, device=dev)
        self.surp = torch.zeros(C, T, device=dev)
        self.score = torch.full((C,), -1.0, device=dev)       # empty slots lose to anything
        self.dev = dev

    def _score(self, surp, imp):
        return ((0.5 + surp / self.surp_ref) * (0.5 + imp / self.imp_ref)).mean(-1)

    def store(self, rb, k):
        """End of a night: draw k candidate clips from today's diary by emotion, keep the best C overall."""
        self.surp_ref = rb.surp.mean() + 1e-6
        self.imp_ref = rb.imp.mean() + 1e-6
        b = rb.sample(k, self.T)
        t, ee = rb.last_idx
        sc = self._score(rb.surp[t, ee], rb.imp[t, ee])
        old = self._score(self.surp, self.imp)
        old = torch.where(self.score < 0, self.score, old)
        allsc = torch.cat([old, sc])
        keep = allsc.topk(self.C).indices                          # best C of old + new
        newslot = keep >= self.C
        # slots that lost to a new clip get overwritten
        lost = torch.ones(self.C, dtype=torch.bool, device=self.dev)
        lost[keep[~newslot]] = False
        slots = lost.nonzero().squeeze(1)
        src = keep[newslot] - self.C
        for key, v in self.d.items():
            v[slots] = b[key][src].to(v.dtype)
        self.imp[slots], self.surp[slots] = rb.imp[t, ee][src], rb.surp[t, ee][src]
        self.score[slots] = sc[src]
        self.n = int((self.score >= 0).sum())
        return int(slots.numel())

    def sample(self, bs):
        self.idx = torch.randint(0, self.n, (bs,), device=self.dev)
        sl = (self.score >= 0).nonzero().squeeze(1)[self.idx]
        self.idx = sl
        return {k: v[sl].float() if k not in ("img", "first") else v[sl] for k, v in self.d.items()}

    def feel(self, kl):
        self.surp[self.idx] = kl.detach().float()


def run_arm(arm, ck, a, dev):
    torch.manual_seed(a.seed)
    res = ck["args"]["res"]
    mind = Mind(res).to(dev)
    load_mind(mind, ck["mind"])
    mind.ent_squash = bool(a.ent_squash)
    args = SimpleNamespace(**{**ck["args"], "compile": a.compile, "compile_mode": "default"})
    args.vic_rows = getattr(args, "vic_rows", 2048)
    learner = Learner(mind, args)
    world = World(a.envs, dev, mode=a.mode, seed=a.seed + 1, compile=bool(a.compile))
    actor = Actor(mind, world, res, dev, compile=bool(a.compile))
    rb = Replay(a.day, a.envs, res, dev, emotion=0.5 if arm in ("emotion", "lifelong") else 0.0)
    keep = Keepsakes(a.keepsakes, args.seq, res, dev) if arm == "lifelong" else None
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
                if keep is not None and keep.n:
                    h = args.batch // 2
                    b1, b2 = rb.sample(args.batch - h, args.seq), keep.sample(h)
                    met, _, kl = learner.update({k: torch.cat([b1[k], b2[k].to(b1[k].dtype)]) for k in b1})
                    rb.feel(kl[:args.batch - h])
                    keep.feel(kl[args.batch - h:])
                else:
                    met, _, kl = learner.update(rb.sample(args.batch, args.seq))
                    if rb.emotion > 0:
                        rb.feel(kl)
                acc += met
            if keep is not None:
                row["kept"] = keep.store(rb, a.keep_per_night)
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
    ap.add_argument("--ent-squash", type=int, default=0, help="nights use the entropy of the real (tanh) muscles")
    ap.add_argument("--keepsakes", type=int, default=4096, help="long-term memory size in clips (lifelong arm)")
    ap.add_argument("--keep-per-night", type=int, default=1024, help="candidate clips offered to it each night")
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ck = torch.load(a.mind, map_location=dev)
    set_eyes(ck["args"])
    out = dict(mind=a.mind, mode=a.mode, days=a.days, day=a.day, night=a.night, envs=a.envs)
    for arm in a.arms.split(","):
        out[arm] = run_arm(arm, ck, a, dev)
    p = Path(a.mind).with_suffix(f".sleep_{a.mode}{'_sq' if a.ent_squash else ''}.json")
    p.write_text(json.dumps(out, indent=1))
    print("saved", p)


if __name__ == "__main__":
    main()
