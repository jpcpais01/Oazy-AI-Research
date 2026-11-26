"""Time the parts of training (no learning run, no files written).

    python3 -m survive.speed --compile 1 --batch 128 --envs 256

Prints ms per creature step (render / mind / world), ms per learning update, and the frames
learned per second a real run would get with one update per step.
"""
from __future__ import annotations

import argparse
import time
from types import SimpleNamespace

import torch

from .model import Mind
from .render import render
from .train import Actor, Learner, Replay
from .world import World


def timeit(fn, n, sync):
    sync()
    t = time.time()
    for _ in range(n):
        fn()
    sync()
    return (time.time() - t) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--compile-mode", default="default")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--vic-rows", type=int, default=2048)
    ap.add_argument("--emotion", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    args = SimpleNamespace(lr_wm=3e-4, lr_ac=8e-5, ent=3e-4, vic_var=1.0, vic_cov=1.0, bf16=1, gamma=0.985,
                           lam=0.95, horizon=a.horizon, vic_rows=a.vic_rows, compile=a.compile, compile_mode=a.compile_mode)
    mind = Mind(a.res).to(dev)
    world = World(a.envs, dev, mode="train", seed=0, compile=bool(a.compile))
    actor = Actor(mind, world, a.res, dev, compile=bool(a.compile))
    rb = Replay(a.seq * 3, a.envs, a.res, dev, emotion=a.emotion)
    learner = Learner(mind, args)

    def env_step():
        act, rec, _ = actor.observe_act(True)
        rb.add(*rec)
        actor.env_step(torch.rand_like(act) * 2 - 1)

    t0 = time.time()
    for _ in range(a.seq + 2):
        env_step()
    sync()
    print(f"warm env (incl. compile) {time.time() - t0:.1f} s", flush=True)
    t0 = time.time()
    for _ in range(3):
        learner.update(rb.sample(a.batch, a.seq))
    sync()
    print(f"warm learner (incl. compile) {time.time() - t0:.1f} s", flush=True)

    with torch.no_grad():
        r_ms = timeit(lambda: render(world, a.res, compile=bool(a.compile)), a.n, sync)
        e_ms = timeit(env_step, a.n, sync)
        w_ms = timeit(lambda: world.step(torch.zeros(a.envs, 4, device=dev)), a.n, sync)
    u_ms = timeit(lambda: learner.update(rb.sample(a.batch, a.seq)), a.n, sync)

    def both():
        env_step()
        _, _, kl = learner.update(rb.sample(a.batch, a.seq))
        if a.emotion > 0:
            rb.feel(kl)
    b_ms = timeit(both, a.n, sync)
    fr = a.batch * a.seq
    print(f"res {a.res} envs {a.envs} batch {a.batch} seq {a.seq} compile {a.compile}/{a.compile_mode}")
    print(f"  render {r_ms:.1f} ms | world.step {w_ms:.1f} ms | whole creature step {e_ms:.1f} ms "
          f"({a.envs / e_ms * 1000:.0f} lived frames/s)")
    print(f"  learning update {u_ms:.1f} ms ({fr / u_ms * 1000:.0f} frames learned/s)")
    print(f"  step + update {b_ms:.1f} ms -> {fr / b_ms * 1000:.0f} frames learned/s, "
          f"{a.envs / b_ms * 1000:.0f} lived/s (replay ratio {fr / a.envs:.0f})")
    if dev.type == "cuda":
        print(f"  peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
