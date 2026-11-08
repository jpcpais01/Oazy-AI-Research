"""Which flex-attention speedups compile on this GPU, and how fast each one is. Pegasus 1 at its real size, compiled,
bf16, one lesson of 4 x 2,048 tokens (no mentor), for each setting of OAZY_FLEX:

    0  the yes/no grids (what runs by default)
    1  flex attention for the 256-token local window (Reader and Speaker)
    2  flex attention for the notebook look
    3  both

    python3 -m pegasus.flexcheck            (~5-10 min: each setting compiles the whole model once)

It ends with the line to train with. Measure with the GPU free: a training run alongside makes the speeds noisy.
"""
from __future__ import annotations

import os
import time
import traceback

import torch

from oazy import term


def lesson(m, x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = m.losses(x, x)
    loss.backward()
    m.zero_grad(set_to_none=True)
    return loss


def main():
    from pegasus.model import Pegasus, PegasusConfig
    dev = torch.device("cuda")
    term.tests_start("Pegasus 1 flex attention", dev)
    x = torch.randint(1, 49152, (4, 2048), device=dev)
    x[:, 700] = 0
    speed, first = {}, {}
    names = {0: "the yes/no grids", 1: "flex: local window", 2: "flex: notebook", 3: "flex: both"}
    for mode in (0, 1, 2, 3):
        torch._dynamo.reset()
        torch.manual_seed(0)
        os.environ["OAZY_FLEX"] = str(mode)
        m = Pegasus(PegasusConfig()).to(dev).train()
        m.calibrate(x)
        m.enable_compile()
        try:
            first[mode] = lesson(m, x).item()
            for _ in range(2):
                lesson(m, x)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(8):
                lesson(m, x)
            torch.cuda.synchronize()
            speed[mode] = 8 * x.numel() / (time.time() - t0)
            same = abs(first[mode] - first[0]) < 0.02
            ok = mode == 0 or same
            print(f"  {'✓' if ok else '✗'} {mode}  {names[mode]:<20} {speed[mode] / 1e3:6.0f}k tokens/s"
                  + ("" if mode == 0 else f"  ({speed[mode] / speed[0] - 1:+.0%}, loss {first[mode]:.4f} vs {first[0]:.4f})"),
                  flush=True)
            if not ok:
                del speed[mode]
        except Exception as ex:
            last = [l_ for l_ in traceback.format_exception_only(type(ex), ex)[-1].splitlines() if l_.strip()][-1]
            print(f"  ✗ {mode}  {names[mode]:<20} does not compile here: {last[:140]}", flush=True)
            if mode == 0:
                raise
        del m
        torch.cuda.empty_cache()
    best = max(speed, key=speed.get)
    if best == 0 or speed[best] < 1.02 * speed[0]:
        print("\nno flex setting is 2% faster here: keep training as you do (OAZY_FLEX=0)")
    else:
        print(f"\nfastest: {best} ({names[best]}, {speed[best] / speed[0] - 1:+.0%}). Train with OAZY_FLEX={best} in front "
              "of the train line (a resumed run picks it up).")


if __name__ == "__main__":
    main()
