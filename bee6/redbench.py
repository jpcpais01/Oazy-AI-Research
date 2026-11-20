"""The compiler's reduction kernels (means, sums), timed alone, each written a few ways. On the pod, about 3 minutes.

    python3 -m bee6.redbench

Why. On a B200, Bee-135's elementwise kernels run near the card's memory speed (ReLU² on the way back: 0.9 GB in
129 µs), but the kernels that add things up do not: the norm's mean reads 75 MB in 86 µs, a Canon weight gradient
150 MB in 132 µs. With lean and cuDNN's attention on, those reductions are the largest share of a pass. Every one of
them is the compiler's own kernel, so how the sum is written decides the kernel it becomes: this file times each of
Bee-135's reduction shapes at full size (32 x 2,048 words of 576, bf16), written as it is now and in a few other ways
that give the same number, and with the compiler left to try several kernel layouts (autotune).

Each row: microseconds a call, the memory speed that is (bytes read / time), and the speed against "as now". Nothing
here changes the model: the winners are then written into bee6/lean.py.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

EPS = torch.finfo(torch.float32).eps


def cases(B, T, D, H, hd, k=4):
    """[(group, name, bytes read, function, argument names)]: every function of a group gives the same numbers."""
    import torch.nn.functional as F
    ch = [c for c in (128, 16) if T % c == 0]

    def shifted(n, j, Tn):
        return F.pad(n[:, :Tn - j], (0, 0, j, 0))
    out = []
    x16, f32 = 2 * B * T * D, 4 * B * T * D

    # the norm's scale: one mean per word over its 576 numbers
    out += [("a word's mean (the norm)", "as now: mean over the 576", x16,
             lambda x: torch.rsqrt(x.float().square().mean(-1, keepdim=True) + EPS), ("x",))]
    out += [("a word's mean (the norm)", "9 sums of 64, then their sum", x16,
             lambda x: torch.rsqrt(x.float().square().unflatten(-1, (D // 64, 64)).sum(-1).sum(-1, keepdim=True) / D + EPS), ("x",))]
    out += [("a word's mean (the norm)", "words as rows of one grid (N, 576)", x16,
             lambda x: torch.rsqrt(x.reshape(-1, D).float().square().mean(-1, keepdim=True) + EPS).view(B, T, 1), ("x",))]
    out += [("a word's mean (the norm)", "PyTorch's own kernel (not compiled)", x16,
             lambda x: torch.rsqrt(torch.linalg.vector_norm(x, dim=-1, keepdim=True, dtype=torch.float32).square() / D + EPS), ("x",))]
    out += [("a word's mean (the norm)", "as a matrix product with ones", x16,
             lambda x, one: torch.rsqrt((x * x) @ one / D + EPS), ("x", "one"))]

    # the whole norm: the mean and the scaled copy (what a layer pays for one norm)
    def norm_now(x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS)).to(x.dtype)

    def norm_split(x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.square().unflatten(-1, (D // 64, 64)).sum(-1).sum(-1, keepdim=True) / D + EPS)).to(x.dtype)
    out += [("the whole norm (mean + scaled copy)", "as now", 2 * x16, norm_now, ("x",)),
            ("the whole norm (mean + scaled copy)", "9 sums of 64, then their sum", 2 * x16, norm_split, ("x",)),
            ("the whole norm (mean + scaled copy)", "PyTorch's own F.rms_norm (not compiled)", 2 * x16,
             lambda x: F.rms_norm(x, (D,)), ("x",))]

    # a Canon weight gradient: one sum per channel over all the words
    def col_now(g, x, s):
        return (g.float() * (s * x.float())).sum((0, 1))
    out += [("a sum per channel (Canon's gradient)", "as now: sum over all words at once", 2 * x16, col_now, ("g", "x", "s"))]
    out += [("a sum per channel (Canon's gradient)", "over each window's words, then over windows", 2 * x16,
             lambda g, x, s: (g.float() * (s * x.float())).sum(1).sum(0), ("g", "x", "s"))]
    for c in ch:
        out += [("a sum per channel (Canon's gradient)", f"in pieces of {c} words, then the pieces", 2 * x16,
                 (lambda c_: lambda g, x, s: (g.float() * (s * x.float())).unflatten(1, (T // c_, c_)).sum(2).sum((0, 1)))(c), ("g", "x", "s"))]
    out += [("a sum per channel (Canon's gradient)", "PyTorch's own einsum (not compiled)", 2 * x16 + 2 * f32,
             lambda g, x, s: torch.einsum("btd,btd->d", g.float(), s * x.float()), ("g", "x", "s"))]

    # the four Canon weight gradients together (the word itself and the 3 before)
    def canon_now(g, x, s):
        gf, n = g.float(), s * x.float()
        return torch.stack([(gf * n).sum((0, 1))] + [(gf * shifted(n, j, T)).sum((0, 1)) for j in range(1, k)])

    def canon_piece(c):
        def f(g, x, s):
            gf, n = g.float(), s * x.float()
            return torch.stack([(gf * n).unflatten(1, (T // c, c)).sum(2).sum((0, 1))]
                               + [(gf * shifted(n, j, T)).unflatten(1, (T // c, c)).sum(2).sum((0, 1)) for j in range(1, k)])
        return f

    def canon_win(g, x, s):
        gf, n = g.float(), s * x.float()
        return torch.stack([(gf * n).sum(1).sum(0)] + [(gf * shifted(n, j, T)).sum(1).sum(0) for j in range(1, k)])
    out += [("Canon's four gradients together", "as now", 2 * x16, canon_now, ("g", "x", "s")),
            ("Canon's four gradients together", "over each window's words, then over windows", 2 * x16, canon_win, ("g", "x", "s"))]
    for c in ch:
        out += [("Canon's four gradients together", f"in pieces of {c} words, then the pieces", 2 * x16, canon_piece(c), ("g", "x", "s"))]

    # the mean on the way back: one number per word from two streams
    out += [("a word's mean of two streams (the way back)", "as now", 2 * x16,
             lambda g, x: (g.float() * x.float()).mean(-1, keepdim=True), ("g", "x"))]
    out += [("a word's mean of two streams (the way back)", "9 sums of 64, then their sum", 2 * x16,
             lambda g, x: (g.float() * x.float()).unflatten(-1, (D // 64, 64)).sum(-1).sum(-1, keepdim=True) / D, ("g", "x"))]

    # one number from two streams (the x0 mix's and the skips' weights)
    out += [("one number from two streams (mix weights)", "as now", 2 * f32,
             lambda a, b: (a * b).sum(), ("a", "b"))]
    out += [("one number from two streams (mix weights)", "a sum per word, then their sum", 2 * f32,
             lambda a, b: (a * b).sum(-1).sum(), ("a", "b"))]
    out += [("one number from two streams (mix weights)", "PyTorch's own dot product (not compiled)", 2 * f32,
             lambda a, b: torch.dot(a.reshape(-1), b.reshape(-1)), ("a", "b"))]

    # the q/k gains' gradient: 64 sums over every word and head
    q16 = 2 * B * T * H * hd
    out += [("a sum per head channel (q/k gains)", "as now: 64 sums over all words and heads", 2 * q16,
             lambda gq, uq: (gq.float() * uq.float()).sum((0, 1, 2)), ("gq", "uq"))]
    out += [("a sum per head channel (q/k gains)", "over each window's words, then the rest", 2 * q16,
             lambda gq, uq: (gq.float() * uq.float()).sum(1).sum((0, 1)), ("gq", "uq"))]
    for c in ch:
        out += [("a sum per head channel (q/k gains)", f"in pieces of {c} words, then the pieces", 2 * q16,
                 (lambda c_: lambda gq, uq: (gq.float() * uq.float()).unflatten(1, (T // c_, c_)).sum(2).sum((0, 1, 2)))(c), ("gq", "uq"))]
    return out


def timed(fn, args, dev, reps):
    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(3):
            y = fn(*args)
        sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            y = fn(*args)
        sync()
    return (time.perf_counter() - t0) / reps * 1e6, y


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=None)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--width", type=int, default=576)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--no-autotune", action="store_true", help="skip the rows where the compiler tries several kernel layouts")
    ap.add_argument("--runs", default=None)
    a = ap.parse_args(argv)
    from oazy import term
    from oazy.data import default_dir
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    B, T, D, H, hd = a.micro, a.ctx, a.width, 9, 64
    dt = torch.bfloat16
    g_ = torch.Generator(device="cpu").manual_seed(0)

    def rnd(*shape, dtype=dt):
        return torch.randn(*shape, generator=g_).to(dev, dtype)
    data = {"x": rnd(B, T, D), "g": rnd(B, T, D), "s": rnd(B, T, 1, dtype=torch.float32).abs() + 0.5,
            "a": rnd(B, T, D, dtype=torch.float32), "b": rnd(B, T, D, dtype=torch.float32),
            "gq": rnd(B, T, H, hd), "uq": rnd(B, T, H, hd), "one": torch.ones(D, 1, device=dev, dtype=dt)}
    gpu = torch.cuda.get_device_name() if dev.type == "cuda" else "cpu"
    term.header("Reduction kernels, timed alone", f"{gpu} · torch {torch.__version__} · {B} x {T:,} words of {D}, bf16")
    rows, res, ref, last = [], [], {}, None
    for group, name, nbytes, fn, names in cases(B, T, D, H, hd):
        args = [data[n] for n in names]
        variants = [("", fn if "not compiled" in name else torch.compile(fn, dynamic=False))]
        if name.startswith("as now") and not a.no_autotune and dev.type == "cuda":
            variants.append((", the compiler trying several layouts", torch.compile(fn, dynamic=False, mode="max-autotune-no-cudagraphs")))
        for extra, f in variants:
            try:
                us, y = timed(f, args, dev, a.reps)
            except Exception as e:                          # a way this torch cannot compile says so; the others still run
                rows.append([group if group != last else "", name + extra, "did not run", "", f"{type(e).__name__}: {str(e).splitlines()[0][:60] if str(e) else ''}"])
                last = group
                continue
            y = y.float()
            if group not in ref:
                ref[group] = (us, y)
            gap = float((y - ref[group][1]).abs().max() / ref[group][1].abs().max().clamp_min(1e-12))
            rows.append([group if group != last else "", name + extra, f"{us:,.0f}", f"{nbytes / us / 1e6:.2f} TB/s",
                         f"{ref[group][0] / us:.2f}x" + ("" if gap < 2e-2 else f"  (numbers {gap:.0e} apart!)")])
            res.append({"group": group, "name": name + extra, "us": us, "bytes": nbytes, "gap": gap})
            last = group
            term.out(term.c(f"  {group}: {name + extra}  {us:,.0f} µs", "dim"), flush=True)
    term.header("Each reduction, each way", "µs a call · the memory speed that is · against the way it is written now")
    term.table(["what", "written", "µs a call", "memory speed", "vs as now"], rows, right=(2, 3))
    out = Path(a.runs or default_dir("runs")) / "redbench.json"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"gpu": gpu, "torch": torch.__version__, "shape": [B, T, D], "rows": res}, indent=1))
        term.out(term.c(f"  saved {out}", "dim"))
    except OSError:
        pass
    return res


if __name__ == "__main__":
    main()
