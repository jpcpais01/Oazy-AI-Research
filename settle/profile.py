"""Where does Settle's training time go? Run on the GPU after (or during) a run:

    python -m settle.profile --ckpt /workspace/settle_runs/settle/ckpt.pt

It loads the trained model (so the sheet settles as it really does, not like a
fresh random one), takes real TinyStories batches, and times one training step
split into its parts. Then it times whole training steps for a few cheaper
variants of the same trained model, so you can see what each speed-up buys.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import Batches
from .gpt import GPT
from .model import Settle, SettleConfig
from .train import autocast, build


def timed(fn, device, reps):
    for _ in range(2):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t) / reps * 1000, out


def parts(model, x, y, device, reps):
    """Time the pieces of one training step (forward pieces, then the backward)."""
    res = {}
    with autocast(device):
        words = model._words(x)
        res["sheet settle (forward)"], (z, st) = timed(lambda: model.settle_sheet(words), device, reps)
        gist = model._gist_parallel(z)
        res["lens (forward)"], (L, _) = timed(lambda: model.settle_lens(z, gist), device, reps)
        res["readout + loss"], _ = timed(lambda: F.cross_entropy(
            model._read(L[:, :, -1], x).float().reshape(-1, model.cfg.vocab_size), y.reshape(-1)), device, reps)

    def full():
        with autocast(device):
            _, loss, _ = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
    res["whole step (forward + backward)"], _ = timed(full, device, reps)
    fwd = res["sheet settle (forward)"] + res["lens (forward)"] + res["readout + loss"]
    res["backward (whole step minus forward)"] = res["whole step (forward + backward)"] - fwd
    return res, st


def step_time(model, x, y, device, reps):
    def full():
        with autocast(device):
            _, loss, st = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
        return st
    ms, st = timed(full, device, reps)
    return ms, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/settle_runs/settle/ckpt.pt")
    ap.add_argument("--data", default="/workspace/settle_data")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    state = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    base = SettleConfig.from_dict(state["config"])
    T = base.max_len
    x, y = Batches(Path(a.data) / "val.bin", T, a.batch, seed=7).get(device)
    tok = a.batch * T

    def load(**changes):
        cfg = SettleConfig.from_dict({**base.to_dict(), **changes})
        m = Settle(cfg)
        sd = dict(state["model"])
        if "lens_bias" in sd:                                 # narrower lens: keep the right-most positions
            sd["lens_bias"] = sd["lens_bias"][-cfg.lens_width:]
        if cfg.link_heads == 0 and base.link_heads:           # no links: drop the link inputs of the rule
            n = base.link_heads * base.link_dim
            sd["rule1.weight"] = sd["rule1.weight"][:, :-n]
            sd.pop("link_qkv.weight", None)
            sd.pop("link_qkv.bias", None)
        m.load_state_dict(sd)
        m.to(device).train()
        if a.compile and device.type == "cuda":
            m.enable_compile()
        return m

    print(f"device {device} ({torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu'}), "
          f"batch {a.batch} x {T} tokens, trained model from step {state.get('step')}\n")

    m = load()
    res, st = parts(m, x, y, device, a.reps)
    print(f"the trained sheet takes {st['steps']} parallel steps (cap {base.settle_cap}); "
          f"average word settles in {st['col_steps'].float().mean():.1f}\n")
    total = res["whole step (forward + backward)"]
    print(f"{'part of one training step':40s} {'ms':>8s} {'share':>7s}")
    for k, v in res.items():
        if k.startswith("whole"):
            continue
        print(f"{k:40s} {v:8.1f} {100 * v / total:6.0f}%")
    print(f"{'whole step':40s} {total:8.1f}   -> {tok / total * 1000 / 1e3:.0f}k tokens/s\n")

    gm, _ = build("gpt", "full")
    gm.to(device).train()
    if a.compile and device.type == "cuda":
        gm.enable_compile()
    gms, _ = step_time(gm, x, y, device, a.reps)
    print(f"GPT baseline step: {gms:.1f} ms -> {tok / gms:.0f}k tokens/s (Settle is {total / gms:.1f}x slower)\n")
    del gm

    variants = [
        ("as trained", {}),
        ("phantom gradient (learn from last 3 steps only)", {"full_grad": False}),
        ("step cap 12 instead of 24", {"settle_cap": 12}),
        ("looser settle tolerance (0.005)", {"settle_tol": 0.005}),
        ("lens off-ish (width 2, cap 2)", {"lens_width": 2, "lens_cap": 2}),
        ("no long-range links", {"link_heads": 0}),
    ]
    print("same trained weights, one setting changed (speed only; quality needs a training run):")
    print(f"{'variant':52s} {'ms/step':>8s} {'speed-up':>9s} {'steps':>6s}")
    ref = None
    for name, ch in variants:
        try:
            mv = load(**ch)
            ms, st = step_time(mv, x, y, device, a.reps)
            ref = ref or ms
            print(f"{name:52s} {ms:8.1f} {ref / ms:8.2f}x {st['steps']:6d}")
            del mv
        except Exception as e:                                     # a variant that cannot load these weights
            print(f"{name:52s} failed: {type(e).__name__}: {e}")
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
