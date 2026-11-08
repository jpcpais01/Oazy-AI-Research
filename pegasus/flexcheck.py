"""Which flex-attention speedups compile on this GPU, and how fast each one is. Pegasus 1 at its real size, compiled,
bf16, one lesson of 4 x 2,048 tokens (no mentor), for each setting of OAZY_FLEX:

    0  the yes/no grids (what runs by default)
    1  flex attention for the 256-token local window (Reader and Speaker)
    2  flex attention for the notebook look
    3  both

    python3 -m pegasus.flexcheck            (~5-10 min: each setting compiles the whole model once)
    python3 -m pegasus.flexcheck --quick    (~1 min, tiny: which flex calls compile; safe beside a training run)

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


def quick():
    """Pegasus's flex calls on small inputs, one at a time: what compiles (fwd + bwd), in which precisions."""
    from torch.nn.attention.flex_attention import flex_attention
    from pegasus.model import window_mask, flex_dtype
    dev = torch.device("cuda")
    f = torch.compile(flex_attention)
    T, H, KV, d = 1024, 12, 4, 64
    wm = window_mask(T, 256, dev)
    tid = (torch.arange(T, device=dev) // 16).clamp(max=127)[None].expand(2, -1).contiguous()

    def finished(score, b, h, qi, kv):
        return torch.where(kv <= tid[b, qi], score, -float("inf"))

    def mk(n, dt):
        return torch.randn(2, n[0], n[1], d, device=dev, dtype=dt, requires_grad=True)
    cases = [
        ("plain, bf16, one head group", lambda: f(*(mk((H, T), torch.bfloat16) for _ in range(3)))),
        ("local window, bf16, shared heads", lambda: f(mk((H, T), torch.bfloat16), mk((KV, T), torch.bfloat16),
                                                      mk((KV, T), torch.bfloat16), block_mask=wm, enable_gqa=True)),
        ("local window, q/k fp32 + v bf16 (before the fix)", lambda: f(mk((H, T), torch.float32), mk((KV, T), torch.float32),
                                                                      mk((KV, T), torch.bfloat16), block_mask=wm, enable_gqa=True)),
        ("notebook (129 pages), bf16, shared heads", lambda: f(mk((H, T), torch.bfloat16), mk((KV, 129), torch.bfloat16),
                                                              mk((KV, 129), torch.bfloat16), score_mod=finished, enable_gqa=True)),
    ]
    for name, run in cases:
        torch._dynamo.reset()
        try:
            run().float().sum().backward()
            torch.cuda.synchronize()
            print(f"  ✓ {name}", flush=True)
        except Exception as ex:
            last = [l_ for l_ in traceback.format_exception_only(type(ex), ex)[-1].splitlines() if l_.strip()][-1]
            print(f"  ✗ {name}: {last[:120]}", flush=True)
    # the real thing: one Reader-size layer under bf16 autocast, as the training step runs it
    from pegasus.model import Pegasus, PegasusConfig, local_attention
    m = Pegasus(PegasusConfig(enc_layers=1, spk_layers=2, think_layers=1)).to(dev)
    x = torch.randn(2, T, 768, device=dev, requires_grad=True)
    torch._dynamo.reset()
    try:
        g = torch.compile(lambda x_: local_attention(m.reader[0].attn, x_, 256, wm))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = g(x)
        y.float().sum().backward()
        torch.cuda.synchronize()
        print("  ✓ a Pegasus layer's local window under bf16 autocast (with the fix)", flush=True)
    except Exception as ex:
        last = [l_ for l_ in traceback.format_exception_only(type(ex), ex)[-1].splitlines() if l_.strip()][-1]
        print(f"  ✗ a Pegasus layer's local window under bf16 autocast (with the fix): {last[:120]}", flush=True)


def main():
    import sys
    if "--quick" in sys.argv:
        return quick()
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
