"""Bee4's speed check: try each speed switch on this GPU, keep the ones that are faster AND give the same numbers,
and print the train line to use. About 15-20 minutes on a B200 (most of it compiling). Safe to run before a run,
never while one trains on the same GPU.

    python3 -m oazy.speedcheck --model bee4 --preset big            # on the pod, the GPU free
    python3 -m oazy.speedcheck --model bee4 --preset smoke --device cpu   # the self-test's quick version

What it tries, one at a time, starting from the plain settings (32 x 2,048 words a pass, bf16, PyTorch's attention
kernel, the default compile), keeping a switch only when a pass gets at least 2% faster and the loss and gradient
on the same batch stay the same (loss within 1%, gradient direction 99% the same):

  1. FP8 for the matrix products (torchao)          4. compile with autotuned kernels, then with CUDA graphs
  2. cuDNN's attention kernel (Blackwell)           5. the optimizer: grids straightened together or one by one
  3. bigger passes: 48, then 64 sequences

Then, with the winners: how fast the shallow early model (6 and 9 layers) and the short early windows (512, 1,024
words) really are, so the growing and the window warmup are known to pay, the mentor's cost, and the whole run's
speed. Results also go to <runs>/speedcheck-<model>.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import term
from . import train as T
from .data import default_dir, few_cpu_threads
from .profile import fake_batch, peak_tflops, sync


FULL_T = [2048]     # the longest window (set from the model in main)


def flops_per_token(model, Tw):
    """Training work per token: 6 x the weights (the word table counted once, as the score head) + attention."""
    c = model.cfg
    return 6 * model.num_params() + 6 * len(model.blocks) * Tw * c.attn_heads * c.attn_head_dim


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="bee4", choices=["bee4"])
    ap.add_argument("--preset", default="big")
    ap.add_argument("--device", default=None)
    ap.add_argument("--passes", type=int, default=6, help="timed passes per setting (after 2 warm-up passes)")
    ap.add_argument("--runs", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    few_cpu_threads(dev)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    gpu = torch.cuda.get_device_name() if dev.type == "cuda" else "cpu"
    p = T.PRESETS[a.preset]
    rec = T.RECIPE[a.model]
    cuda = dev.type == "cuda"
    term.header(f"{T.NICE[a.model]} speed check", gpu)
    from oazy.tok import eot_id

    def build(s, depth=None):
        torch.manual_seed(0)
        model, cfg = T.build(a.model, a.preset, {"attn_backend": s["attn"], "compile_mode": s["mode"]})
        model._stack_to(depth or cfg.n_layer)
        model.to(dev).train()
        fp8 = False
        if s["fp8"]:
            from oazy import fp8 as fp8_mod
            model, fp8 = fp8_mod.convert(model, dev, lambda *x, **k: None)
        if cuda:
            model.enable_compile()
        return model, cfg, fp8

    def measure(s, depth=None, Tw=None, opt_too=False):
        """ms per pass, the loss and the gradient on the fixed batch, peak memory; None if it fails (memory, kernel)."""
        model = None
        try:
            model, cfg, fp8 = build(s, depth)
            if s["fp8"] and not fp8:
                return None
            Tw = Tw or cfg.max_len
            B = max(1, s["micro"] * cfg.max_len // Tw)
            x, y = fake_batch(B, Tw, cfg.vocab_size, eot_id(cfg.tokenizer), dev)

            def one():
                with T.amp(dev):
                    loss, _ = model.losses(x, y)
                loss.backward()
                return loss
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            for _ in range(2):
                one()
            sync(dev)
            t0 = time.perf_counter()
            for _ in range(a.passes):
                one()
            sync(dev)
            ms = (time.perf_counter() - t0) / a.passes * 1000
            model.zero_grad(set_to_none=True)
            loss = one()
            grad = torch.cat([q.grad.float().flatten() for n, q in model.named_parameters()
                              if q.grad is not None and n.startswith("blocks.0.")])
            out = {"ms": ms, "tok_s": B * Tw / ms * 1000, "loss": float(loss.detach()), "grad": grad.cpu(), "B": B, "T": Tw,
                   "mem": torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0}
            if opt_too:
                opt = T.make_optimizer(model, 1e-4, dev, "muon", 0.03, 0.1)
                for batched in (True, False):
                    for o in opt.opts:
                        if hasattr(o, "batched"):
                            o.batched = batched
                    opt.step()
                    sync(dev)
                    t0 = time.perf_counter()
                    for _ in range(3):
                        opt.step()
                    sync(dev)
                    out["opt_ms_" + ("batched" if batched else "one_by_one")] = (time.perf_counter() - t0) / 3 * 1000
            return out
        except Exception as e:        # out of memory, a kernel this GPU or PyTorch lacks, a compile failure
            term.info(f"{label(s)}: {type(e).__name__}: {str(e).splitlines()[0][:140] if str(e) else ''}")
            return None
        finally:
            del model
            if cuda:
                torch.cuda.empty_cache()
            torch._dynamo.reset()

    def label(s):
        return f"{s['micro']} x {FULL_T[0]:,} · {'FP8' if s['fp8'] else 'bf16'} · attention {s['attn']} · compile {s['mode']}"

    def same(r, ref):
        cos = float(F.cosine_similarity(r["grad"], ref["grad"], dim=0))
        return abs(r["loss"] - ref["loss"]) / ref["loss"] < 0.01 and cos > 0.99, cos

    FULL_T[0] = T.build(a.model, a.preset, {})[1].max_len
    rows, res = [], {"gpu": gpu, "tried": []}
    best = dict(micro=32 if a.preset == "big" else 2, fp8=False, attn="auto", mode="default")
    t_start = time.time()
    ref = measure(best, opt_too=True)
    if ref is None:
        raise SystemExit("the plain settings do not run on this device")
    cur = ref
    rows.append(["plain", label(best), f"{ref['ms']:.0f}", term.rate(ref["tok_s"]), f"{ref['mem']:.0f} GB", "start"])

    def trial(name, change):
        nonlocal best, cur
        s = {**best, **change}
        r = measure(s)
        if r is None:
            rows.append([name, label(s), "-", "-", "-", "failed: kept off"])
            return
        ok, cos = same(r, ref) if s["micro"] == ref["B"] else (abs(r["loss"] - ref["loss"]) / ref["loss"] < 0.01, 1.0)
        faster = r["tok_s"] > cur["tok_s"] * 1.02
        keep = ok and faster
        why = "kept" if keep else ("not faster" if ok else f"numbers moved (loss {r['loss']:.3f} vs {ref['loss']:.3f}, gradient {cos:.3f})")
        rows.append([name, label(s), f"{r['ms']:.0f}", term.rate(r["tok_s"]), f"{r['mem']:.0f} GB", why])
        res["tried"].append({"name": name, **{k: v for k, v in r.items() if k != "grad"}, "kept": keep, "settings": s})
        if keep:
            best, cur = s, r

    if cuda:
        trial("FP8", {"fp8": True})
        trial("cuDNN attention", {"attn": "cudnn"})
    for m in ((48, 64) if a.preset == "big" else (4,)):
        trial(f"{m} a pass", {"micro": m})
    if cuda:
        trial("autotuned compile", {"mode": "max-autotune-no-cudagraphs"})
        trial("CUDA graphs", {"mode": "reduce-overhead"})
    term.table(["switch", "settings", "ms/pass", "speed", "memory", ""], rows, right=(2, 3, 4))
    ob, oo = ref.get("opt_ms_batched"), ref.get("opt_ms_one_by_one")
    if ob is not None:
        term.rows([("optimizer step", f"{ob:.0f} ms with the grids straightened together, {oo:.0f} ms one by one "
                                      f"(once per {term.tokens(p.get('batch_' + a.model, 524288))} tokens)")])

    # the early run: shallow model, short windows (they should be proportionally faster, or growing does not pay)
    m0, cfg0 = T.build(a.model, a.preset, {})
    from ember.model import parse_plan, depth_at
    grow = parse_plan(cfg0.grow, cfg0.n_layer)
    grid = [(i + 0.5) / 1000 for i in range(1000)]
    stages = sorted({(depth_at(grow, q), m0.ctx_at(q)) for q in grid})
    full = (cfg0.n_layer, cfg0.max_len)
    early, say_rows = {full: cur["tok_s"]}, []
    for depth, Tw in stages:
        r = cur if (depth, Tw) == full else measure(best, depth=depth, Tw=Tw)
        if r is not None:
            early[(depth, Tw)] = r["tok_s"]
            share = sum((depth_at(grow, q), m0.ctx_at(q)) == (depth, Tw) for q in grid) / len(grid)
            say_rows.append([f"{depth} layers, {Tw:,}-word windows", f"{share:.0%} of the run", term.rate(r["tok_s"]),
                             f"{r['tok_s'] / cur['tok_s']:.2f}x the full model"])
    term.table(["stage", "", "speed", ""], say_rows, right=(1, 2))
    shallow = [k for k in early if k[0] < cfg0.n_layer]
    if shallow and any(early[k] < cur["tok_s"] * 1.15 for k in shallow):
        term.info("a shallow stage is barely faster than the full model: growing is not paying on this GPU "
                  "(start full depth with --set grow=" + f"{cfg0.n_layer}:0 on a fresh run)")

    # the mentor, and the whole run
    mentor_ms = 0.0
    mentor_name = T.MENTOR.get(a.model, "none") if a.preset == "big" else "none"
    if mentor_name != "none" and cuda:
        try:
            from ember.mentor import Mentor
            mentor = Mentor(mentor_name, dev, compile=True)
            x, _ = fake_batch(cur["B"], cur["T"], 49152, 0, dev)
            model, cfg, _ = build(best)
            _, mask, _ = model.doc_info(x)
            with torch.no_grad(), T.amp(dev):
                for _ in range(2):
                    mentor.hidden(x, mask)
                sync(dev)
                t0 = time.perf_counter()
                for _ in range(a.passes):
                    mentor.hidden(x, mask)
                sync(dev)
            mentor_ms = (time.perf_counter() - t0) / a.passes * 1000
            del model, mentor
        except Exception as e:
            term.info(f"mentor timing: {type(e).__name__}: {str(e)[:120]}")
    from ember.mentor import fade
    total_ms, n = 0.0, len(grid)
    batch = p.get("batch_" + a.model, 524288)
    for q in grid:
        tps = early.get((depth_at(grow, q), m0.ctx_at(q)), cur["tok_s"])
        w = fade(q, rec["mentor_weight"], rec["mentor_until"], None, rec["mentor_floor"], rec["mentor_off"])
        pass_tok = cur["B"] * cur["T"]
        ms = pass_tok / tps * 1000 + w * mentor_ms + (ob or 0) * pass_tok / batch
        total_ms += ms / pass_tok
    run_tps = 1000 / (total_ms / n)
    pk = peak_tflops(gpu)
    m0._stack_to(cfg0.n_layer)
    mfu = cur["tok_s"] * flops_per_token(m0, cfg0.max_len) / 1e12 / pk if pk else None
    term.header("result", color="green")
    flags = [f"--micro {best['micro']}"] + ([] if best["fp8"] else ["--fp8 0"])
    sets = [f"attn_backend={best['attn']}"] if best["attn"] != "auto" else []
    sets += [f"compile_mode={best['mode']}"] if best["mode"] != "default" else []
    line = f"bash runpod/train.sh {a.model} --preset {a.preset} --tokens 10e9 " + " ".join(flags) + (" --set " + " ".join(sets) if sets else "")
    term.rows([("full model", term.c(term.rate(cur["tok_s"]), "bold", "green") + f"  ({label(best)}; {cur['mem']:.0f} GB)"),
               ("chip use", f"about {mfu * 100:.0f}% of its bf16 arithmetic" if mfu else None),
               ("mentor", f"{mentor_ms:.0f} ms a pass when it teaches" if mentor_ms else None),
               ("whole run", term.c(f"about {term.rate(run_tps)}", "bold") + "  (growing, window warmup, mentor share, optimizer)"),
               ("10,000M tokens", f"about {10e9 / run_tps / 3600:.1f} hours"),
               ("train with", term.c(line, "bold")),
               ("took", f"{(time.time() - t_start) / 60:.0f} min")])
    res.update({"best": best, "tok_s": cur["tok_s"], "run_tok_s": run_tps, "mfu": mfu, "mentor_ms": mentor_ms,
                "early": {f"{d}x{t}": v for (d, t), v in early.items()}, "line": line,
                "opt_ms": {"batched": ob, "one_by_one": oo}})
    out = Path(default_dir("runs") if a.runs is None else a.runs) / f"speedcheck-{a.model}.json"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1, default=str))
        term.info(f"saved {out}")
    except OSError:
        pass
    return res


if __name__ == "__main__":
    main()
