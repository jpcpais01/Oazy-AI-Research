"""Train Settle or the matched GPT baseline on TinyStories. Resumable: re-run the same command.

    python -m settle.train --model settle --data /workspace/settle_data --runs /workspace/settle_runs
    python -m settle.train --model gpt    --data /workspace/settle_data --runs /workspace/settle_runs
    python -m settle.train --bench        # measure speed of both and print how long a run will take

Everything the dashboard shows is written into the run folder:
  status.json      live status (step, tokens, speed, ETA)
  metrics.jsonl    training / validation losses and settle statistics
  samples.jsonl    text written by the model as it learns
  snapshots/       Settle only: the probe sentence's sheet, settle movie, steps per word
  ckpt.pt          checkpoint (every few minutes, on Ctrl-c and at the end)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .data import Batches, get_tokenizer, EOT
from .gpt import GPT, GPTConfig, matched_gpt_config
from .model import Settle, SettleConfig
from . import viz

PRESETS = {
    # small enough to run on a laptop CPU in a couple of minutes (for testing the pipeline)
    "tiny": dict(
        settle=dict(d_embed=64, rows=4, channels=32, hidden=128, gist_dim=32, lens_width=4,
                     lens_hidden=96, settle_cap=16, lens_cap=4, max_len=64, link_heads=2, link_dim=8),
        batch=16, lr=3e-3, warmup=30, tokens=600_000, eval_tokens=100_000, eval_batches=4,
        sample_tokens=200_000, snap_tokens=100_000, sample_len=40,
    ),
    # the real run for one H200 / B300 (about 20-30 minutes for Settle)
    "full": dict(
        settle=dict(),
        batch=64, lr=1.5e-3, warmup=400, tokens=150_000_000, eval_tokens=4_000_000, eval_batches=20,
        sample_tokens=10_000_000, snap_tokens=4_000_000, sample_len=120,
    ),
}

SAMPLE_PROMPTS = ["Once upon a time", "The little dog was", "Tom and Mia went to the"]


def parse_overrides(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def build(kind, preset, vocab=50257, overrides=None):
    p = PRESETS[preset]
    scfg = SettleConfig(vocab_size=vocab, **{**p["settle"], **(overrides or {})})
    settle = Settle(scfg)
    if kind == "settle":
        return settle, scfg
    gcfg = matched_gpt_config(settle.num_params(), scfg.d_embed, scfg.max_len, vocab)
    return GPT(gcfg), gcfg


def lr_at(progress, step, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def make_optimizer(model, lr, device):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    kw = dict(lr=lr, betas=(0.9, 0.95))
    if device.type == "cuda":
        kw["fused"] = True
    return torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                              {"params": no_decay, "weight_decay": 0.0}], **kw)


def autocast(device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


class Run:
    def __init__(self, folder: Path):
        self.dir = folder
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "snapshots").mkdir(exist_ok=True)

    def log(self, name, rec):
        with open(self.dir / name, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def status(self, rec):
        tmp = self.dir / "status.json.tmp"
        tmp.write_text(json.dumps(rec))
        os.replace(tmp, self.dir / "status.json")


@torch.no_grad()
def evaluate(model, val, n, device):
    model.eval()
    tot = 0.0
    for _ in range(n):
        x, y = val.get(device)
        with autocast(device):
            _, _, st = model(x, y)
        tot += float(st["ce"])
    model.train()
    return tot / n


@torch.no_grad()
def write_samples(model, enc, run, step, tokens, n_tok, device):
    model.eval()
    compiled = model._compiled
    if isinstance(model, Settle):
        model._compiled = None
    for p in SAMPLE_PROMPTS:
        torch.manual_seed(1234)
        ids, st = model.generate(enc.encode_ordinary(p), max_new_tokens=n_tok, temperature=0.8, top_k=40, eos=EOT)
        rec = {"step": step, "tokens": tokens, "prompt": p, "text": enc.decode(ids),
               "pieces": viz.token_strings(enc, ids)}
        if st is not None:
            rec["steps"] = st["steps"][len(enc.encode_ordinary(p)):]
        run.log("samples.jsonl", rec)
    model._compiled = compiled
    model.train()


def save_ckpt(run, model, opt, meta):
    tmp = run.dir / "ckpt.pt.tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), **meta}, tmp)
    os.replace(tmp, run.dir / "ckpt.pt")


def bench(args, device):
    print(f"device {device}  ({torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu'})")
    p = PRESETS[args.preset]
    B = args.batch or p["batch"]
    for kind in ("settle", "gpt"):
        model, cfg = build(kind, args.preset, overrides=parse_overrides(args.set))
        model.to(device).train()
        if args.compile and device.type == "cuda":
            model.enable_compile()
        opt = make_optimizer(model, 1e-4, device)
        T = cfg.max_len
        x = torch.randint(0, cfg.vocab_size, (B, T), device=device)
        times = []
        for i in range(args.bench_steps + 3):
            t0 = time.time()
            with autocast(device):
                _, loss, st = model(x, x)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= 3:
                times.append(time.time() - t0)
        dt = float(np.median(times))
        tps = B * T / dt
        mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0
        extra = f"  settle steps {st.get('steps')}, lens {st.get('lens_steps')}" if kind == "settle" else ""
        print(f"{kind:7s} params {model.num_params() / 1e6:6.2f}M  {dt * 1000:7.1f} ms/step  {tps / 1e3:8.1f}k tok/s  "
              f"peak mem {mem:5.1f} GB  ->  {p['tokens'] / 1e6:.0f}M tokens in {p['tokens'] / tps / 60:6.1f} min{extra}")
        if kind == "settle":
            print("        (settle steps change as the model learns, so treat this time as a rough guide;"
                  " the dashboard shows the real ETA)")
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["settle", "gpt"], default="settle")
    ap.add_argument("--preset", choices=list(PRESETS), default="full")
    ap.add_argument("--data", default="/workspace/settle_data")
    ap.add_argument("--runs", default="/workspace/settle_runs")
    ap.add_argument("--name", default=None, help="run folder name (default: the model name)")
    ap.add_argument("--tokens", type=float, default=None, help="training tokens, e.g. 2e8")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--compile", type=int, default=1, help="torch.compile on GPU (1/0)")
    ap.add_argument("--ckpt-min", type=float, default=5.0, help="minutes between checkpoints")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint and start over")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--bench-steps", type=int, default=10)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="change a Settle setting, e.g. --set settle_cap=32 link_heads=0 (the GPT is re-matched)")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.bench:
        return bench(args, device)

    p = PRESETS[args.preset]
    B = args.batch or p["batch"]
    lr = args.lr or p["lr"]
    total = int(args.tokens or p["tokens"])
    run = Run(Path(args.runs) / (args.name or args.model))
    enc = get_tokenizer()

    torch.manual_seed(args.seed)
    model, cfg = build(args.model, args.preset, overrides=parse_overrides(args.set))
    model.to(device)
    T = cfg.max_len
    opt = make_optimizer(model, lr, device)
    train = Batches(Path(args.data) / "train.bin", T, B, seed=args.seed)
    val_seed = 12345

    step, tokens, elapsed = 0, 0, 0.0
    next_eval = next_sample = next_snap = 0
    ck = run.dir / "ckpt.pt"
    if ck.exists() and not args.fresh:
        state = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step, tokens, elapsed = state["step"], state["tokens"], state["elapsed"]
        next_eval, next_sample, next_snap = state["next"]
        train.set_state(state["data_rng"])
        torch.set_rng_state(state["torch_rng"])
        print(f"resumed from {ck}: step {step}, {tokens / 1e6:.1f}M tokens, {elapsed / 60:.1f} min")
    elif args.fresh:
        for f in ("metrics.jsonl", "samples.jsonl", "snapshots.jsonl", "status.json"):
            (run.dir / f).unlink(missing_ok=True)
        for f in (run.dir / "snapshots").glob("*.json"):
            f.unlink()

    info = {
        "model": args.model, "preset": args.preset, "config": cfg.to_dict(), "params": model.num_params(),
        "params_non_embedding": model.num_params(non_embedding=True), "batch": B, "context": T,
        "lr": lr, "total_tokens": total, "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
    }
    (run.dir / "config.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    if tokens >= total:
        print("this run is already finished. Use --tokens to train longer or --fresh to start over.")
        run.status({**info, "state": "done", "step": step, "tokens": tokens, "elapsed": elapsed})
        return

    if args.compile and device.type == "cuda":
        model.enable_compile()

    stop = {"flag": False}

    def handler(sig, frame):
        print(f"\nsignal {sig}: saving a checkpoint and stopping...", flush=True)
        stop["flag"] = True
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    def ckpt_meta():
        return {"step": step, "tokens": tokens, "elapsed": elapsed, "next": (next_eval, next_sample, next_snap),
                "data_rng": train.state(), "torch_rng": torch.get_rng_state(), "info": info,
                "kind": args.model, "config": cfg.to_dict()}

    model.train()
    last_ckpt = time.time()
    t_log, tok_log = time.time(), tokens
    acc = {"ce": 0.0, "n": 0, "steps": 0, "lens": 0, "cap": 0.0, "res": 0.0, "colmean": 0.0, "colp90": 0.0}
    log_every = 10
    while tokens < total and not stop["flag"]:
        t_step = time.time()

        # occasional evaluation / samples / snapshot (aligned on tokens so both models line up)
        if tokens >= next_eval:
            vl = evaluate(model, Batches(Path(args.data) / "val.bin", T, B, seed=val_seed), p["eval_batches"], device)
            run.log("metrics.jsonl", {"kind": "eval", "step": step, "tokens": tokens, "elapsed": elapsed, "val_loss": vl})
            print(f"[eval] step {step} tokens {tokens / 1e6:.1f}M val loss {vl:.4f}", flush=True)
            next_eval += p["eval_tokens"]
        if tokens >= next_sample:
            write_samples(model, enc, run, step, tokens, p["sample_len"], device)
            next_sample += p["sample_tokens"]
        if isinstance(model, Settle) and tokens >= next_snap:
            snap = viz.probe(model, enc)
            name = f"{step:07d}.json"
            (run.dir / "snapshots" / name).write_text(json.dumps({"step": step, "tokens_seen": tokens, **snap}))
            run.log("snapshots.jsonl", {"step": step, "tokens": tokens, "file": name,
                                        "mean_steps": float(np.mean(snap["steps"])), "gap": snap["gap"],
                                        "agree": snap["agree"], "steps": snap["steps"]})
            next_snap += p["snap_tokens"]
        t_extra = time.time() - t_step

        cur_lr = lr_at(tokens / total, step, lr, p["warmup"])
        for g in opt.param_groups:
            g["lr"] = cur_lr
        x, y = train.get(device)
        try:
            with autocast(device):
                _, loss, st = model(x, y)
            loss.backward()
        except Exception as e:
            if model._compiled is None:
                raise
            print(f"torch.compile failed ({type(e).__name__}: {e}); continuing without compile", flush=True)
            model._compiled = None
            opt.zero_grad(set_to_none=True)
            with autocast(device):
                _, loss, st = model(x, y)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        tokens += B * T

        acc["ce"] += float(st["ce"].detach())
        acc["n"] += 1
        if "steps" in st:
            cs = st["col_steps"].float()
            acc["steps"] += st["steps"]
            acc["lens"] += float(st["lens_steps"])
            acc["cap"] += float(st["cap_frac"])
            acc["res"] += float(st["res"].detach())
            acc["colmean"] += float(cs.mean())
            acc["colp90"] += float(torch.quantile(cs.flatten()[:65536], 0.9))
        elapsed += time.time() - t_step - t_extra   # training time only (evals and samples excluded)

        if step % log_every == 0 or tokens >= total:
            now = time.time()
            tps = (tokens - tok_log) / max(1e-6, now - t_log)
            t_log, tok_log = now, tokens
            n = acc["n"]
            rec = {"kind": "train", "step": step, "tokens": tokens, "elapsed": elapsed, "loss": acc["ce"] / n,
                   "lr": cur_lr, "tok_s": tps}
            if acc["steps"]:
                rec.update({"settle_steps": acc["steps"] / n, "lens_steps": acc["lens"] / n,
                            "cap_frac": acc["cap"] / n, "residual": acc["res"] / n,
                            "col_steps_mean": acc["colmean"] / n, "col_steps_p90": acc["colp90"] / n})
            run.log("metrics.jsonl", rec)
            eta = (total - tokens) / max(tps, 1)
            run.status({**info, "state": "running", "step": step, "tokens": tokens, "elapsed": elapsed,
                        "tok_s": tps, "eta_s": eta, "loss": rec["loss"], "updated": time.time()})
            extra = (f" | settle {rec['settle_steps']:.1f} steps (cols {rec['col_steps_mean']:.1f}, "
                     f"cap {rec['cap_frac'] * 100:.0f}%) lens {rec['lens_steps']:.1f}") if acc["steps"] else ""
            print(f"step {step:6d} | {tokens / 1e6:7.1f}M tok | loss {rec['loss']:.4f} | lr {cur_lr:.2e} | "
                  f"{tps / 1e3:6.1f}k tok/s | eta {eta / 60:5.1f} min{extra}", flush=True)
            acc = {k: 0.0 if isinstance(v, float) else 0 for k, v in acc.items()}

        if time.time() - last_ckpt > args.ckpt_min * 60:
            save_ckpt(run, model, opt, ckpt_meta())
            last_ckpt = time.time()

    save_ckpt(run, model, opt, ckpt_meta())
    if tokens >= total:
        vl = evaluate(model, Batches(Path(args.data) / "val.bin", T, B, seed=val_seed), p["eval_batches"], device)
        run.log("metrics.jsonl", {"kind": "eval", "step": step, "tokens": tokens, "elapsed": elapsed, "val_loss": vl})
        write_samples(model, enc, run, step, tokens, p["sample_len"], device)
        if isinstance(model, Settle):
            snap = viz.probe(model, enc)
            name = f"{step:07d}.json"
            (run.dir / "snapshots" / name).write_text(json.dumps({"step": step, "tokens_seen": tokens, **snap}))
            run.log("snapshots.jsonl", {"step": step, "tokens": tokens, "file": name,
                                        "mean_steps": float(np.mean(snap["steps"])), "gap": snap["gap"],
                                        "agree": snap["agree"], "steps": snap["steps"]})
        run.status({**info, "state": "done", "step": step, "tokens": tokens, "elapsed": elapsed,
                    "val_loss": vl, "updated": time.time()})
        print(f"finished: val loss {vl:.4f}, {elapsed / 60:.1f} min of training. checkpoint: {run.dir / 'ckpt.pt'}")
    else:
        run.status({**info, "state": "stopped", "step": step, "tokens": tokens, "elapsed": elapsed,
                    "updated": time.time()})
        print(f"stopped at step {step}; run the same command again to resume.")


if __name__ == "__main__":
    main()
