"""How fast Keep and its twin train on this machine, one line per way of running them. No data needed (random tokens).

    python3 -m keep.speedcheck                    # the base size on the GPU, about two minutes
    python3 -m keep.speedcheck --preset big
    python3 -m keep.speedcheck --preset tiny --device cpu --steps 3 --batch 2

The lines: the twin; Keep as it trains now; Keep with the stores' whole tables inside autograd (as in the first GPU
run); Keep with a mask in place of the window; each without torch.compile too. Tokens per second and peak memory.
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from . import model as M
from .run import PRESETS, amp, make


def masked(q, k, v, W):
    """The window as a T x T mask: what model.near replaced."""
    T = q.size(2)
    i = torch.arange(T, device=q.device)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=(i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < W))


def measure(arm, preset, vocab, B, steps, device, compiled=True, sparse=True, window=True):
    torch.manual_seed(0)
    m = make(arm, preset, vocab).to(device).train()
    for s in m.store.values():
        s.sparse = sparse
    near = M.near
    if not window:
        M.near = masked
    try:
        if compiled and device.type == "cuda":
            m.speed_up()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=device.type == "cuda")
        T = m.cfg.max_len
        state = m.init_state(B, device) if m.store else None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = None
        for step in range(steps + 3):
            if step == 3:                               # the first steps compile and warm up
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
            x = torch.randint(0, vocab, (B, T + 1), device=device)
            with amp(device):
                loss, state, _ = m.loss(x[:, :-1], x[:, 1:], state)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if state is not None:
                state = m.detach(state)
        if device.type == "cuda":
            torch.cuda.synchronize()
        rate = steps * B * T / (time.time() - t0)
        mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0
        return rate, mem
    finally:
        M.near = near


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="base", choices=list(PRESETS))
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    preset = PRESETS[args.preset]
    B = args.batch or preset["batch"]
    rows = [("twin", "full", {}), ("twin, not compiled", "full", dict(compiled=False)),
            ("Keep", "keep", {}), ("Keep, not compiled", "keep", dict(compiled=False)),
            ("Keep, whole tables in autograd", "keep", dict(sparse=False)),
            ("Keep, window as a mask", "keep", dict(window=False)),
            ("Keep, as in the first GPU run (all three off)", "keep", dict(compiled=False, sparse=False, window=False))]
    print(f"{args.preset}: batch {B} x {preset['cfg']['max_len']} tokens, {args.steps} steps each, {device}", flush=True)
    for name, arm, how in rows:
        try:
            rate, mem = measure(arm, preset, 49152, B, args.steps, device, **how)
            print(f"  {name:<48} {rate / 1e3:>7.0f}k tok/s   {mem:5.1f} GB", flush=True)
        except Exception as e:
            print(f"  {name:<48} failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
