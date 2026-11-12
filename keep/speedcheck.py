"""How fast Keep and its twin train on this machine, one line per way of running them. No data needed (random tokens).

    python3 -m keep.speedcheck                    # the base size on the GPU, about five minutes (each line compiles)
    python3 -m keep.speedcheck --preset big
    python3 -m keep.speedcheck --preset tiny --device cpu --steps 3 --batch 2

The lines take one part of the step away at a time (the stores, the scores over the 49,152 words), so the
differences say where a step goes; then larger chunks and a larger batch. --old adds the lines with each of the
first speed changes (PR #249) turned off. Tokens per second, milliseconds per step, peak memory.
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


def measure(arm, preset, vocab, B, steps, device, compiled=True, sparse=True, window=True, scores=True, slices=4096, **over):
    """-> tokens per second, milliseconds per step, peak GB. A step is what keep.run does: both losses, the way back,
    the clip, the optimizer. scores=False: no scores over the words (a cheap loss on the final vectors in their place).
    over: KeepConfig fields changed for this line (stores=(), chunk=256, ...)."""
    torch.manual_seed(0)
    if compiled and device.type == "cuda":
        torch._dynamo.reset()                           # each line compiles afresh: past a few models torch stops compiling, silently
    m = make(arm, preset, vocab, **over).to(device).train()
    m.loss_chunk = slices
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
            asked = torch.rand(B, T, device=device) < 0.03
            with amp(device):
                if scores:
                    loss, state, info = m.loss(x[:, :-1], x[:, 1:], state, asked=asked, asked_weight=4.0)
                else:
                    h, state, _ = m.hidden(x[:, :-1], state)
                    loss = h.float().square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            if state is not None:
                state = m.detach(state)
            float(loss.detach())                        # the run reads the loss every step too
        if device.type == "cuda":
            torch.cuda.synchronize()
        secs = (time.time() - t0) / steps
        mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0
        return B * T / secs, 1e3 * secs, mem
    finally:
        M.near = near


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="base", choices=list(PRESETS))
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--old", action="store_true", help="also the lines with each of the first speed changes turned off")
    args = ap.parse_args(argv)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    preset = PRESETS[args.preset]
    B = args.batch or preset["batch"]
    ch = preset["keep"]["chunk"]
    rows = [("twin", "full", {}),
            ("twin, no scores over the words", "full", dict(scores=False)),
            ("Keep", "keep", {}),
            ("Keep, no scores over the words", "keep", dict(scores=False)),
            ("Keep, no stores (the window alone)", "keep", dict(stores=())),
            ("Keep, no stores, no scores over the words", "keep", dict(stores=(), scores=False)),
            ("Keep-wide (the stores and the whole window)", "wide", {}),
            (f"Keep, chunks of {2 * ch} (not {ch})", "keep", dict(chunk=2 * ch)),
            (f"Keep, chunks of {4 * ch} (not {ch})", "keep", dict(chunk=4 * ch)),
            ("Keep, the loss in slices of 1,024 (not 4,096)", "keep", dict(slices=1024)),
            ("Keep, the loss in slices of 8,192 (not 4,096)", "keep", dict(slices=8192)),
            (f"Keep, batch {4 * B} (not {B})", "keep", dict(batch=4 * B)),
            (f"twin, batch {4 * B} (not {B})", "full", dict(batch=4 * B))]
    if args.old:
        rows += [("Keep, not compiled", "keep", dict(compiled=False)),
                 ("Keep, the stores' whole tables in autograd", "keep", dict(sparse=False)),
                 ("Keep, window as a mask", "keep", dict(window=False)),
                 ("Keep, all three off (the first GPU run)", "keep", dict(compiled=False, sparse=False, window=False))]
    print(f"{args.preset}: batch {B} x {preset['cfg']['max_len']} tokens, {args.steps} steps a line, {device}", flush=True)
    got = {}
    for name, arm, how in rows:
        how = dict(how)
        b = how.pop("batch", B)
        try:
            rate, ms, mem = measure(arm, preset, 49152, b, args.steps, device, **how)
            got[name] = ms * B / b                                      # per `B` rows, to compare the lines
            print(f"  {name:<46} {rate / 1e3:>6.0f}k tok/s  {ms:>7.1f} ms a step  {mem:5.1f} GB", flush=True)
        except Exception as e:
            print(f"  {name:<46} failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    try:                                                                # where a Keep step goes
        k, body = got["Keep"], got["Keep, no stores, no scores over the words"]
        st, sc = k - got["Keep, no stores (the window alone)"], k - got["Keep, no scores over the words"]
        print(f"\nA Keep step of {k:.0f} ms: the blocks {body:.0f} ms, the stores {st:.0f} ms, the scores over the words {sc:.0f} ms, "
              f"the rest {k - body - st - sc:.0f} ms.", flush=True)
    except KeyError:
        pass


if __name__ == "__main__":
    main()
