"""Bee5's speed check against Bee4, on the GPU it will train on: tokens a second, the share of the chip's arithmetic
used (MFU), memory, the optimizer step, the early short windows, the mentor, and each speed switch, then the whole
run's speed and the train line to use. About 10-15 minutes on a B200 (most of it compiling). Measures only, starts
nothing; never run it while a run trains on the same GPU.

    python3 -m bee5.speedcheck                                   # on the pod, the GPU free (Bee5 and Bee4, full size)
    python3 -m bee5.speedcheck --preset smoke --device cpu       # the self-test's quick version

Loss at equal tokens is not something a speed check can show: the 0.5B race does that (bee5/README.md).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from oazy import term
from oazy import train as T
from oazy.data import default_dir, few_cpu_threads
from oazy.profile import fake_batch, peak_tflops, sync


def flops_per_token(model, Tw):
    """Training arithmetic per token: 6 x every weight that multiplies (the word table once, as the score head; the
    value tables are look-ups, not counted) + attention, by the same formula as Bee4's speed check."""
    c = model.cfg
    tables = model.table_params() - model.embed.weight.numel() if hasattr(model, "table_params") else 0
    return 6 * (model.num_params() - tables) + 6 * len(model.blocks) * Tw * c.attn_heads * c.attn_head_dim


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="big")
    ap.add_argument("--device", default=None)
    ap.add_argument("--micro", type=int, default=None, help="sequences per pass at 2,048 words (default 32; smoke 2)")
    ap.add_argument("--passes", type=int, default=6, help="timed passes per setting (after 2 warm-up passes)")
    ap.add_argument("--vs", default="bee4", help="the model to compare with at full size ('none' = Bee5 alone)")
    ap.add_argument("--runs", default=None)
    a = ap.parse_args(argv)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    few_cpu_threads(dev)
    cuda = dev.type == "cuda"
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
    gpu = torch.cuda.get_device_name() if cuda else "cpu"
    micro = a.micro or (32 if a.preset == "big" else 2)
    pk = peak_tflops(gpu)
    term.header("Bee5 speed check", gpu)
    from oazy.tok import eot_id
    t_start = time.time()

    def build(kind, over=None):
        torch.manual_seed(0)
        model, cfg = T.build(kind, a.preset, dict(over or {}))
        if hasattr(model, "_stack_to"):
            model._stack_to(cfg.n_layer)         # Bee4 timed fully grown
        model.to(dev).train()
        if cuda:
            model.enable_compile()
        return model, cfg

    def measure(kind, over=None, Tw=None, opt_too=False, mtp=False):
        """ms per pass (forward + backward, bf16), tokens/s, memory; None if it fails."""
        model = None
        try:
            model, cfg = build(kind, over)
            if mtp:
                model.set_progress(0.0)          # the early lesson on 3 next words
            Tw = Tw or cfg.max_len
            B = max(1, micro * cfg.max_len // Tw)
            x, y = fake_batch(B, Tw, cfg.vocab_size, eot_id(cfg.tokenizer), dev)

            def one():
                with T.amp(dev):
                    loss, _ = model.losses(x, y)
                loss.backward()
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
            out = {"ms": ms, "tok_s": B * Tw / ms * 1000, "B": B, "T": Tw,
                   "mem": torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0,
                   "params": model.num_params(), "flops": flops_per_token(model, Tw)}
            out["mfu"] = out["tok_s"] * out["flops"] / 1e12 / pk if pk else None
            if opt_too:
                opt = T.make_optimizer(model, 1.8e-3, dev, "muon", 0.03, 0.1)
                opt.step()
                sync(dev)
                t0 = time.perf_counter()
                for _ in range(3):
                    opt.step()
                sync(dev)
                out["opt_ms"] = (time.perf_counter() - t0) / 3 * 1000
            return out
        except Exception as e:
            term.info(f"{kind} {over or ''}: {type(e).__name__}: {str(e).splitlines()[0][:140] if str(e) else ''}")
            return None
        finally:
            del model
            if cuda:
                torch.cuda.empty_cache()
            torch._dynamo.reset()

    def row(name, r, ref=None, note=""):
        if r is None:
            return [name, "-", "-", "-", "-", "failed"]
        rel = f"{r['tok_s'] / ref['tok_s']:.2f}x" if ref else ""
        return [name, f"{r['ms']:.0f}", term.rate(r["tok_s"]), f"{r['mfu'] * 100:.0f}%" if r.get("mfu") else "-",
                f"{r['mem']:.0f} GB", " ".join(s for s in (rel, note) if s)]

    res = {"gpu": gpu, "micro": micro}
    b5 = measure("bee5", opt_too=True)
    if b5 is None:
        raise SystemExit("Bee5 does not run on this device with the plain settings")
    rows = [row(f"Bee5 ({b5['params'] / 1e6:.0f}M)", b5)]
    b4 = None
    if a.vs != "none":
        b4 = measure(a.vs, opt_too=True)
        rows.append(row(f"{T.NICE.get(a.vs, a.vs)} ({b4['params'] / 1e6:.0f}M)" if b4 else T.NICE.get(a.vs, a.vs), b4, b5))
    best, cur = {}, b5
    trials = [("3 next words (first 1/6)", None, dict(mtp=True))]
    if cuda:
        trials += [("cuDNN attention", {"attn_backend": "cudnn"}, {}),
                   ("autotuned compile", {"compile_mode": "max-autotune-no-cudagraphs"}, {})]
    trials += [("SwiGLU instead of ReLU²", {"mlp": "swiglu"}, {})]
    for name, over, kw in trials:
        r = measure("bee5", over, **kw)
        keep = over is not None and over.get("mlp") is None and r is not None and r["tok_s"] > cur["tok_s"] * 1.02
        note = "kept (2%+ faster, same maths)" if keep else ("speed only; quality is the race's" if over and "mlp" in over
                                                            else "" if over is None else "not faster: off")
        rows.append(row(name, r, b5, note))
        if name.startswith("3 next") and r is not None:
            res["mtp_tok_s"] = r["tok_s"]
        if keep:
            best.update(over)
            cur = r
    term.table(["", "ms/pass", "speed", "chip use", "memory", ""], rows, right=(1, 2, 3, 4))

    # the early windows (512 and 1,024 words: same words per pass), the mentor, the whole run
    m0, cfg0 = T.build("bee5", a.preset, dict(best))
    stages, early = sorted({m0.ctx_at((i + 0.5) / 1000) for i in range(1000)}), {cfg0.max_len: cur["tok_s"]}
    for Tw in stages:
        if Tw != cfg0.max_len:
            r = measure("bee5", best, Tw=Tw)
            if r is not None:
                early[Tw] = r["tok_s"]
    term.rows([(f"{Tw:,}-word windows", term.rate(v) + f"  ({v / cur['tok_s']:.2f}x the 2,048-word speed)") for Tw, v in sorted(early.items())])
    mentor_ms = 0.0
    if cuda and a.preset == "big":
        try:
            from ember.mentor import Mentor
            mentor = Mentor(T.MENTOR["bee5"], dev, compile=True)
            x, _ = fake_batch(cur["B"], cur["T"], 49152, 0, dev)
            mask = m0.doc_mask(x)
            with torch.no_grad(), T.amp(dev):
                for _ in range(2):
                    mentor.hidden(x, mask)
                sync(dev)
                t0 = time.perf_counter()
                for _ in range(a.passes):
                    mentor.hidden(x, mask)
                sync(dev)
            mentor_ms = (time.perf_counter() - t0) / a.passes * 1000
            del mentor
        except Exception as e:
            term.info(f"mentor timing: {type(e).__name__}: {str(e)[:120]}")
    from ember.mentor import fade
    from bee5.model import mtp_weights
    rc = T.RECIPE["bee5"]
    batch = T.PRESETS[a.preset].get("batch_bee5", 524288)
    grid = [(i + 0.5) / 1000 for i in range(1000)]

    def run_speed(mentor_on=True):
        tot = 0.0
        for q in grid:
            tps = early.get(m0.ctx_at(q), cur["tok_s"])
            if len(mtp_weights(q, cfg0.mtp_until)) > 1 and res.get("mtp_tok_s"):
                tps = min(tps, res["mtp_tok_s"] * tps / b5["tok_s"])
            w = fade(q, rc["mentor_weight"], rc["mentor_until"], None, rc["mentor_floor"], rc["mentor_off"]) if mentor_on else 0
            pass_tok = cur["B"] * cur["T"]
            tot += (pass_tok / tps * 1000 + w * mentor_ms + b5.get("opt_ms", 0) * pass_tok / batch) / pass_tok
        return 1000 / (tot / len(grid))
    run_tps, solo_tps = run_speed(True), run_speed(False)
    sets = " ".join(f"{k}={v}" for k, v in best.items())
    line = f"OAZY_DATA=/root/data bash runpod/train.sh bee5 --preset {a.preset} --tokens 10e9 --micro {micro}" + (f" --set {sets}" if sets else "")
    term.header("result", color="green")
    term.rows([("Bee5, full size", term.c(term.rate(cur["tok_s"]), "bold", "green") + (f"  ({cur['mfu'] * 100:.0f}% of the chip's bf16 arithmetic)" if cur.get("mfu") else "")),
               ("vs " + T.NICE.get(a.vs, a.vs), f"{b4['tok_s'] / 1e3:.0f}k tok/s, {b4['mfu'] * 100:.0f}% of the chip" if b4 and b4.get("mfu")
                else (term.rate(b4["tok_s"]) if b4 else None)),
               ("optimizer step", f"{b5['opt_ms']:.0f} ms (once per {term.tokens(batch)} tokens)" if b5.get("opt_ms") is not None else None),
               ("mentor", f"{mentor_ms:.0f} ms a pass when it teaches" if mentor_ms else None),
               ("whole run", term.c(f"about {term.rate(run_tps)}", "bold") + f"  (without the mentor {term.rate(solo_tps)})"),
               ("10,000M tokens", f"about {10e9 / run_tps / 3600:.1f} hours ({10e9 / solo_tps / 3600:.1f} without the mentor)"),
               ("0.5B race", f"about {0.5e9 / run_tps / 60:.0f} minutes"),
               ("train with", term.c(line, "bold")),
               ("took", f"{(time.time() - t_start) / 60:.0f} min")])
    res.update({"bee5": b5, "vs": b4, "best": best, "tok_s": cur["tok_s"], "run_tok_s": run_tps, "run_tok_s_no_mentor": solo_tps,
                "mentor_ms": mentor_ms, "early": early, "line": line})
    out = Path(default_dir("runs") if a.runs is None else a.runs) / "speedcheck-bee5.json"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1, default=str))
        term.info(f"saved {out}")
    except OSError:
        pass
    return res


if __name__ == "__main__":
    main()
