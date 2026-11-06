"""Where does a training step spend its time? One micro-batch of a model, taken apart on this GPU.

    python3 -m oazy.profile --model bee3 --preset big                 # the real shapes: 16 x 2,048 words, mentor on
    python3 -m oazy.profile --model bee3 --preset big --mentor none   # the student alone
    python3 -m oazy.profile --model bee3 --preset big --loss plain    # the old way of computing the loss, to compare
    python3 -m oazy.profile --model bee3 --preset big --compile 0

It prints, per micro-batch (one forward + backward pass; a step is several of these):
  - the whole pass, and the pass without the mentor;
  - its pieces measured on their own: the memory layers, the attention layers, the per-word MLPs, the word
    table + loss, the mentor's reading, the optimizer step;
  - tokens per second, and how much of the chip's arithmetic that is (an estimate: "MFU");
  - the ten GPU kernels that take the longest, by name, so the slow part has a name.
The pieces measured alone do not add up exactly to the whole (the whole overlaps some work); read them as shares.
Results are also written to <runs>/profile-<model>.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .data import default_dir
from . import train as T
from memtrix import memory

PEAK = {   # dense bf16 TFLOP/s with fp32 accumulation, the number training can at best use
    "B200": 2250, "B300": 2250, "GB200": 2250, "H100": 989, "H200": 989, "H800": 989, "A100": 312, "A800": 312,
    "L40S": 362, "RTX 5090": 105, "RTX 4090": 83, "RTX 6000": 2250 / 2,
}


def peak_tflops(name: str):
    for k, v in PEAK.items():
        if k in name:
            return v
    return None


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, steps, device, warm=2):
    for _ in range(warm):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / steps * 1000.0        # ms


def fake_batch(B, Tw, vocab, eot, device, doc_len=600, seed=0):
    """Random tokens with documents of about doc_len words, ended by <|endoftext|> like the real text."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(1, vocab, (B, Tw + 1), generator=g)
    lens = torch.randint(doc_len // 2, doc_len * 3 // 2, (B, 8), generator=g)
    for b in range(B):
        pos = (lens[b].cumsum(0) - 1).tolist()
        for p_ in pos:
            if p_ < Tw + 1:
                x[b, p_] = eot
    x = x.to(device)
    return x[:, :-1].contiguous(), x[:, 1:].contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=T.KINDS, default="bee3")
    ap.add_argument("--preset", choices=list(T.PRESETS), default="big")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--micro", type=int, default=None, help="sequences per pass (default: the trainer's)")
    ap.add_argument("--ctx", type=int, default=None, help="window length (default: the model's first window)")
    ap.add_argument("--mentor", default=None, help="the mentor to time with (default: the model's), or none")
    ap.add_argument("--mentor-weight", type=float, default=0.5)
    ap.add_argument("--loss", choices=["fused", "plain"], default="fused")
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--kernels", type=int, default=10, help="how many of the slowest GPU kernels to list")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    gpu = torch.cuda.get_device_name() if device.type == "cuda" else "cpu"
    say = print
    p = T.PRESETS[a.preset]
    model, cfg = T.build(a.model, a.preset, T.parse_overrides(a.set))
    model.to(device).train()
    if hasattr(model, "fused_loss"):
        type(model).fused_loss = a.loss == "fused"
    Tw = a.ctx or (model.ctx_at(0) if hasattr(model, "ctx_at") else cfg.max_len)
    micro0 = a.micro or p.get("micro_" + a.model, p["micro"])
    B = a.micro or max(1, micro0 * 1024 // Tw)
    ntok = B * Tw
    memory.fla_check(device)
    if a.compile and device.type == "cuda":
        model.enable_compile()
    mentor_name = a.mentor or T.MENTOR.get(a.model, "none")
    mentor = None
    if mentor_name != "none":
        from ember.mentor import Mentor
        mentor = Mentor(mentor_name, device, compile=bool(a.compile))
    from oazy.tok import eot_id
    x, y = fake_batch(B, Tw, cfg.vocab_size, eot_id(getattr(cfg, "tokenizer", "gpt2")), device)
    opt = T.make_optimizer(model, 1e-4, device, "muon" if a.model in T.FAST else "adamw")
    amp = T.amp(device)
    res = {"model": a.model, "gpu": gpu, "micro": B, "context": Tw, "tokens_per_pass": ntok, "mentor": mentor_name,
           "loss": a.loss, "compile": bool(a.compile), "fla": bool(memory._FLA_OK), "fla_packed_docs": bool(memory._FLA_SEG),
           "doc_mask": bool(getattr(cfg, "doc_mask", 0)), "params": model.num_params()}
    say(f"{T.NICE[a.model]} on {gpu}: {B} x {Tw} words per pass ({ntok:,} tokens), mentor {mentor_name}, loss {a.loss}, "
        f"compile {bool(a.compile)}, memory kernel {'fla' if memory._FLA_OK else 'pure PyTorch'}"
        + (f" (packed documents {'yes' if memory._FLA_SEG else 'no'})" if getattr(cfg, 'doc_mask', 0) else ""))
    ms = {}

    def pass_(with_mentor):
        with amp:
            if hasattr(model, "losses"):
                loss, _ = model.losses(x, y, mentor if with_mentor else None, a.mentor_weight if with_mentor else 0.0)
            else:
                _, loss, _ = model(x, y)
        loss.backward()
    # --- the whole pass
    if mentor is not None:
        ms["pass with mentor"] = timed(lambda: pass_(True), a.steps, device)
    ms["pass, student alone"] = timed(lambda: pass_(False), a.steps, device)
    model.zero_grad(set_to_none=True)
    peak_mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    # --- the pieces, alone
    if mentor is not None:
        with torch.no_grad():
            resets, mask, _ = model.doc_info(x) if hasattr(model, "doc_info") else (None, None, None)
            ms["mentor reading"] = timed(lambda: mentor.hidden(x, mask), a.steps, device)
    if hasattr(model, "_trunk"):
        resets, mask, segs = model.doc_info(x)
        f = model._compiled or model._trunk
        with amp:
            h = f(x, resets, mask, segs).detach().requires_grad_()

        def head():
            from oazy.lmloss import head_loss, plain_loss
            with amp:
                fn = head_loss if model.fused_loss else plain_loss
                loss = fn(h, model.embed.weight, y, cfg.softcap)[0]
            loss.backward()
        ms["word table + loss"] = timed(head, a.steps, device)
        model.zero_grad(set_to_none=True)
        d = torch.randn(B, Tw, cfg.d_model, device=device, requires_grad=True)
        kinds = cfg.kinds()
        blocks = {k: [b for b, kk in zip(model.blocks, kinds) if kk == k] for k in set(kinds)}
        names = {"M": "memory layers (Memtrix)", "A": "attention layers"}

        def piece(fn):      # a piece is compiled when the model is, so it is measured as it runs in training
            return torch.compile(fn) if a.compile and device.type == "cuda" else fn

        def fwd_bwd(fn):
            with amp:
                out = fn()
            out.float().sum().backward()
        for k, bs in blocks.items():
            b = bs[0]
            f = piece(lambda: b.mixer()(b.n1(d), None, resets, segs)[0] if b.kind == "M" else b.mixer()(b.n1(d), None, mask)[0])
            one = timed(lambda: fwd_bwd(f), a.steps, device)
            ms[f"{names[k]} x{len(bs)}"] = one * len(bs)
            if k == "M":                                  # the memory rule's kernel alone, on the layer's own q, k, v
                mem = b.mixer()
                with torch.no_grad(), amp:
                    q, kk, v, trust, lf, _ = mem.project(b.n1(d), None, resets)
                q, kk, v, trust, lf = (t.detach().clone().requires_grad_() for t in (q, kk, v, trust, lf))

                def kern():
                    with amp:
                        o, _ = memory.delta_rule(q, kk, v, lf, trust.to(q.dtype), resets=resets, segs=segs)
                    o.float().sum().backward()
                one = timed(kern, a.steps, device)
                ms[f"  of which the memory kernel x{len(bs)}"] = one * len(bs)
        f = piece(lambda: model.blocks[0].mlp(model.blocks[0].n2(d)))
        one = timed(lambda: fwd_bwd(f), a.steps, device)
        ms[f"MLPs x{len(model.blocks)}"] = one * len(model.blocks)
        model.zero_grad(set_to_none=True)
    # --- the optimizer step (once per several passes in training)
    pass_(False)
    ms["optimizer step (once per step)"] = timed(opt.step, 3, device)
    opt.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    # --- report
    whole = ms.get("pass with mentor", ms["pass, student alone"])
    say(f"\n{'piece':40s} {'ms / pass':>10s} {'share':>7s}    (the pieces are measured alone; they overlap and need not add up)")
    for k, v in ms.items():
        say(f"{k:40s} {v:10.1f} {v / whole * 100:6.0f}%")
    tps = ntok / (whole / 1000)
    tps_alone = ntok / (ms["pass, student alone"] / 1000)
    n_attn = sum(1 for k in getattr(cfg, "kinds", lambda: [])() if k == "A")
    flops_tok = 6 * model.num_params() + 12 * n_attn * cfg.d_model * Tw       # weights, plus attention's T x d work
    pk = peak_tflops(gpu)
    res.update({"ms": ms, "tokens_per_s": tps, "tokens_per_s_student_alone": tps_alone, "peak_mem_gb": peak_mem,
                "flops_per_token": flops_tok, "mfu": (tps * flops_tok / 1e12 / pk) if pk else None})
    say(f"\n{tps / 1e3:.1f}k tokens/s with everything ({tps_alone / 1e3:.1f}k student alone); peak memory {peak_mem:.1f} GB")
    if pk:
        say(f"about {res['mfu'] * 100:.0f}% of the chip's arithmetic ({flops_tok / 1e9:.2f} GFLOP per token against {pk} TFLOP/s)")
    # --- the slowest kernels
    if device.type == "cuda" and a.kernels:
        from torch.profiler import profile, ProfilerActivity
        def top(prof, n, per):
            dt = lambda e: getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0))   # noqa: E731
            rows = sorted(prof.key_averages(), key=lambda e: -dt(e))
            tot = sum(dt(e) for e in rows) or 1
            out = []
            for e in rows[:n]:
                say(f"  {dt(e) / tot * 100:5.1f}%  {dt(e) / per / 1000:8.1f} ms  {e.key[:90]}")
                out.append({"name": e.key, "share": dt(e) / tot, "ms": dt(e) / per / 1000})
            return out
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(2):
                pass_(mentor is not None)
        say(f"\nthe {a.kernels} slowest GPU kernels of a pass (forward + backward{', mentor included' if mentor is not None else ''}):")
        res["kernels"] = top(prof, a.kernels, 2)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            opt.step()
        model.zero_grad(set_to_none=True)
        say("\nthe 3 slowest of the optimizer step:")
        res["kernels_optimizer"] = top(prof, 3, 1)
    out = Path(a.runs or default_dir("runs")) / f"profile-{a.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    say(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
