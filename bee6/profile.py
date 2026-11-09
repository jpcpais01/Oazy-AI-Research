"""Where Bee6's training pass spends its time, kernel by kernel (idea 2: find what keeps the chip from ~40% use).

    python3 -m bee6.profile                                  # full size, 32 x 2,048 words, Bee6's defaults
    python3 -m bee6.profile --set fp8=0 mod_layers=          # any switches, to compare
    python3 -m bee6.profile --model bee5                     # Bee5, the same way

It prints the pass split into its parts (the layers forward + backward, the word table + loss, the optimizer step),
the chip use, and the slowest GPU kernels grouped by kind (matrix products bf16 / FP8, attention, fused elementwise,
copies, ...) with the top ones by name. The kernel list is a torch.profiler trace of a few compiled passes; a trace
file for Perfetto / chrome://tracing is saved next to the runs (<runs>/profile-bee6.json.gz) when --trace is given.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch

from oazy import term
from oazy import train as T
from oazy.data import default_dir, few_cpu_threads
from oazy.profile import fake_batch, peak_tflops, sync

KINDS = [("FP8 products", r"scaled_mm|f8|fp8|e4m3|e5m2"), ("attention", r"flash|fmha|attention|sdpa|cudnn"),
         ("matrix products", r"gemm|cutlass|sm\d+_xmma|nvjet|cublas|matmul|splitk"),
         ("fused elementwise (compiled)", r"triton"), ("copies, casts", r"copy|cast|cat|elementwise|vectorized|reduce"),
         ("optimizer", r"adam|multi_tensor|foreach")]


def kind_of(name):
    n = name.lower()
    for k, pat in KINDS:
        if re.search(pat, n):
            return k
    return "other"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="bee6", choices=["bee6", "bee5"])
    ap.add_argument("--preset", default="big")
    ap.add_argument("--micro", type=int, default=None)
    ap.add_argument("--ctx", type=int, default=None, help="window length (default the longest, 2,048)")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--kernels", type=int, default=15)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--runs", default=None)
    a = ap.parse_args(argv)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    few_cpu_threads(dev)
    cuda = dev.type == "cuda"
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
    gpu = torch.cuda.get_device_name() if cuda else "cpu"
    model, cfg = T.build(a.model, a.preset, T.parse_overrides(a.set))
    model.to(dev).train()
    model.set_progress(1.0)
    if hasattr(model, "startup_check"):
        model.startup_check(dev, term.say)
    if cuda:
        model.enable_compile()
    micro = a.micro or (32 if a.preset == "big" else 2)
    Tw = a.ctx or cfg.max_len
    B = max(1, micro * cfg.max_len // Tw)
    from oazy.tok import eot_id
    x, y = fake_batch(B, Tw, cfg.vocab_size, eot_id(cfg.tokenizer), dev)
    opt = T.make_optimizer(model, 1e-4, dev, "muon", 0.03, 0.1)
    term.header(f"{T.NICE[a.model]} profile", gpu)
    term.rows([("pass", f"{B} x {Tw} words = {B * Tw:,} tokens"), ("body", model.layout()),
               ("switches", model.speed_text() if hasattr(model, "speed_text") else "-")])

    def one():
        with T.amp(dev):
            loss, _ = model.losses(x, y)
        loss.backward()

    def timed(fn, n):
        for _ in range(2):
            fn()
        sync(dev)
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        sync(dev)
        return (time.perf_counter() - t0) / n * 1000
    ms = {"whole pass": timed(one, a.steps)}
    model.zero_grad(set_to_none=True)
    trunk = getattr(model, "_compiled", None) or (model._trunk_aux if hasattr(model, "_trunk_aux") else model._trunk)

    def body():
        with T.amp(dev):
            h = trunk(x)
            h = h[0] if isinstance(h, tuple) else h
        h.float().square().mean().backward()
    ms["layers (forward + backward)"] = timed(body, a.steps)
    with T.amp(dev):
        h = trunk(x)
        h = (h[0] if isinstance(h, tuple) else h).detach().requires_grad_()
    from bee5.loss import head_loss
    tg = y[..., None]

    def head():
        with T.amp(dev):
            loss = head_loss(h, model.embed.weight, tg, (1.0,), cfg, None, None, 0.0, getattr(model, "loss_chunk", 1024))[0]
        loss.backward()
    ms["word table + loss"] = timed(head, a.steps)
    model.zero_grad(set_to_none=True)
    one()
    ms["optimizer step (once per 8 passes)"] = timed(opt.step, 3)
    model.zero_grad(set_to_none=True)
    from bee6.speedcheck import flops_per_token
    tok_s = B * Tw / ms["whole pass"] * 1000
    pk = peak_tflops(gpu)
    mfu = tok_s * flops_per_token(model, Tw) / 1e12 / pk if pk else None
    term.table(["", "ms", "share of a pass"], [[k, f"{v:.1f}", f"{v / ms['whole pass']:.0%}" if "optimizer" not in k else ""]
                                               for k, v in ms.items()], right=(1, 2))
    term.rows([("speed", term.rate(tok_s) + (f"  ({mfu * 100:.0f}% of the chip's bf16 arithmetic)" if mfu else ""))])

    res = {"gpu": gpu, "ms": ms, "tok_s": tok_s, "mfu": mfu, "kernels": []}
    if cuda:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            for _ in range(a.steps):
                one()
            sync(dev)
        model.zero_grad(set_to_none=True)
        per = defaultdict(float)
        names = defaultdict(float)
        total = 0.0
        for e in prof.key_averages():
            t = getattr(e, "device_time_total", None)
            if t is None:
                t = getattr(e, "cuda_time_total", 0.0)
            if not t or e.key.startswith(("aten::", "ProfilerStep", "cudaLaunch", "Torch-Compiled", "CompiledFunction")):
                continue
            t = t / 1000 / a.steps          # ms per pass
            per[kind_of(e.key)] += t
            names[e.key] += t
            total += t
        term.header("GPU time by kind", f"{total:.0f} ms of kernels per pass")
        term.table(["kind", "ms", "share"], [[k, f"{v:.1f}", f"{v / max(total, 1e-9):.0%}"] for k, v in
                                             sorted(per.items(), key=lambda kv: -kv[1])], right=(1, 2))
        top = sorted(names.items(), key=lambda kv: -kv[1])[:a.kernels]
        term.header(f"the {len(top)} slowest kernels")
        term.table(["kernel", "ms", "share"], [[(k[:90] + "…") if len(k) > 90 else k, f"{v:.1f}", f"{v / max(total, 1e-9):.0%}"]
                                              for k, v in top], right=(1, 2))
        res["kernels"] = [{"name": k, "ms": v, "kind": kind_of(k)} for k, v in top]
        res["by_kind"] = dict(per)
        if a.trace:
            out = Path(a.runs or default_dir("runs")) / f"profile-{a.model}.json.gz"
            prof.export_chrome_trace(str(out))
            term.info(f"trace saved: {out}")
    out = Path(a.runs or default_dir("runs")) / f"profile-{a.model}.json"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1, default=str))
        term.info(f"saved {out}")
    except OSError:
        pass
    return res


if __name__ == "__main__":
    main()
