"""Train Engram or the GPT-2 baseline on FineWeb-Edu. Resumable: re-run the same command.

    python -m engram.train --model engram --data /workspace/engram_data --runs /workspace/engram_runs
    python -m engram.train --model gpt    --data /workspace/engram_data --runs /workspace/engram_runs
    python -m engram.train --bench --hours 2     # measure both models' speed; suggest a token budget

Both models see exactly the same text in the same order, with the same schedule and optimizer.
Everything the dashboard shows is written into the run folder:
  status.json     live status (step, tokens, speed, ETA, latest results)
  metrics.jsonl   training loss, validation loss, HellaSwag, memory checks (engram/probes.py)
  samples.jsonl   text written by the model as it learns
  snapshots/      Engram only: the probe text read with the lid off
  ckpt.pt         checkpoint (every few minutes, on Ctrl-c and at the end)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch

from .data import Windows, val_set, get_tokenizer, EOT
from .gpt import GPT, GPTConfig
from .model import Engram, EngramConfig
from . import hellaswag, probes, memory

PRESETS = {
    # a laptop-CPU test of the whole pipeline (minutes)
    "tiny": dict(
        engram=dict(d_model=128, n_layer=4, n_head=2, head_k=64, head_v=64, mlp_hidden=256, max_len=256),
        gpt=dict(d_model=128, n_layer=4, n_head=2, max_len=256),
        micro=16, accum=1, lr=3e-3, warmup=50, tokens=3_000_000,
        eval_every=250_000, probe_every=1_000_000, hs_every=1_000_000, sample_every=1_000_000,
        val_windows=64, hs_quick=200, sample_len=40,
    ),
    # a 1-minute smoke test of every code path
    "smoke": dict(
        engram=dict(d_model=64, n_layer=2, n_head=1, head_k=64, head_v=64, mlp_hidden=128, max_len=128),
        gpt=dict(d_model=64, n_layer=2, n_head=1, max_len=128),
        micro=8, accum=2, lr=3e-3, warmup=10, tokens=60_000,
        eval_every=20_000, probe_every=30_000, hs_every=30_000, sample_every=30_000,
        val_windows=16, hs_quick=20, sample_len=10,
    ),
    # GPT-2 small size (Engram 117M vs GPT-2 124M), ~2 h for both models on one B200 (check with --bench)
    "full": dict(
        engram=dict(), gpt=dict(),
        micro=32, accum=8, lr=1.2e-3, warmup=400, tokens=2_000_000_000,
        eval_every=50_000_000, probe_every=250_000_000, hs_every=250_000_000, sample_every=250_000_000,
        val_windows=128, hs_quick=1000, sample_len=120,
    ),
    # the same models, 2.5x the text (e.g. a 2 x B200 pod for ~2.5 h, one model per GPU)
    "big": dict(
        engram=dict(), gpt=dict(),
        micro=32, accum=8, lr=1.2e-3, warmup=600, tokens=5_000_000_000,
        eval_every=100_000_000, probe_every=500_000_000, hs_every=500_000_000, sample_every=500_000_000,
        val_windows=128, hs_quick=1000, sample_len=120,
    ),
}

SAMPLE_PROMPTS = ["The water cycle is", "In 1905, Albert Einstein", "The most important thing about learning is"]


def parse_overrides(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def build(kind, preset, overrides=None):
    p = PRESETS[preset]
    if kind == "engram":
        cfg = EngramConfig(**{**p["engram"], **(overrides or {})})
        return Engram(cfg), cfg
    cfg = GPTConfig(**{**p["gpt"], **(overrides or {})})
    return GPT(cfg), cfg


def load_model(ckpt: Path, device):
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    kind = st.get("kind", "engram")
    if kind == "engram":
        model = Engram(EngramConfig.from_dict(st["config"]))
    else:
        model = GPT(GPTConfig.from_dict(st["config"]))
    model.load_state_dict(st["model"])
    return model.to(device).eval(), st


def lr_at(progress, step, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    return base * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def make_optimizer(model, lr, device):
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "conv_w" not in n]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2 or "conv_w" in n]
    kw = dict(lr=lr, betas=(0.9, 0.95), eps=1e-8)
    if device.type == "cuda":
        kw["fused"] = True
    return torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                              {"params": no_decay, "weight_decay": 0.0}], **kw)


def amp(device):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")


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
def val_loss(model, windows, device, micro):
    model.eval()
    xy = torch.from_numpy(windows).to(device)
    tot = 0.0
    for i in range(0, len(xy), micro):
        with amp(device):
            _, loss, _ = model(xy[i:i + micro, :-1], xy[i:i + micro, 1:])
        tot += float(loss) * len(xy[i:i + micro])
    model.train()
    return tot / len(xy)


@torch.no_grad()
def write_samples(model, enc, run, step, tokens, n_tok):
    model.eval()
    for p in SAMPLE_PROMPTS:
        torch.manual_seed(1234)
        ids, _ = model.generate(enc.encode_ordinary(p), max_new_tokens=n_tok, temperature=0.8, top_k=40, eos=EOT)
        run.log("samples.jsonl", {"step": step, "tokens": tokens, "prompt": p, "text": enc.decode(ids)})
    model.train()


def save_ckpt(run, model, opt, meta):
    tmp = run.dir / "ckpt.pt.tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), **meta}, tmp)
    os.replace(tmp, run.dir / "ckpt.pt")


def bench(args, device):
    print(f"device {device}  ({torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu'})")
    p = PRESETS[args.preset]
    micro = args.micro or p["micro"]
    total = int(args.tokens or p["tokens"])
    speeds = {}
    for kind in ("engram", "gpt"):
        model, cfg = build(kind, args.preset, parse_overrides(args.set) if kind == "engram" else None)
        model.to(device).train()
        memory.fla_check(device)
        if args.compile and device.type == "cuda":
            model.enable_compile()
        opt = make_optimizer(model, 1e-4, device)
        T = cfg.max_len
        x = torch.randint(0, 50257, (micro, T), device=device)
        times = []
        for i in range(args.bench_steps + 3):
            t0 = time.time()
            with amp(device):
                _, loss, _ = model(x, x)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= 3:
                times.append(time.time() - t0)
        dt = float(np.median(times))
        tps = micro * T / dt
        speeds[kind] = tps
        mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0
        extra = ""
        if kind == "engram":
            extra = f"  (memory kernel: {'flash-linear-attention' if memory._FLA_OK else 'pure PyTorch chunked'})"
        print(f"{kind:7s} params {model.num_params() / 1e6:6.1f}M  {dt * 1000:7.1f} ms per {micro}x{T}  "
              f"{tps / 1e3:8.1f}k tok/s  peak mem {mem:5.1f} GB{extra}")
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    te, tg = total / speeds["engram"], total / speeds["gpt"]
    print(f"\n{total / 1e9:.2f}B tokens each: Engram {te / 3600:.2f} h, GPT {tg / 3600:.2f} h on their own.")
    print(f"  one GPU, both at once: about {(te + tg) / 3600:.2f} h")
    print(f"  two GPUs, one each:    about {max(te, tg) / 3600:.2f} h")
    if args.hours:
        share = args.hours * 3600 / (1 / speeds["engram"] + 1 / speeds["gpt"])
        two = args.hours * 3600 * min(speeds.values())
        print(f"\nto finish in {args.hours:g} h (+ a few minutes of evaluations):")
        print(f"  one GPU:  --tokens {share * 0.93 / 1e9:.2f}e9")
        print(f"  two GPUs: --tokens {two * 0.93 / 1e9:.2f}e9")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["engram", "gpt"], default="engram")
    ap.add_argument("--preset", choices=list(PRESETS), default="full")
    ap.add_argument("--data", default="/workspace/engram_data")
    ap.add_argument("--runs", default="/workspace/engram_runs")
    ap.add_argument("--name", default=None, help="run folder name (default: the model name)")
    ap.add_argument("--tokens", type=float, default=None, help="training tokens, e.g. 2e9")
    ap.add_argument("--micro", type=int, default=None, help="sequences per forward pass (lower it if out of memory)")
    ap.add_argument("--batch-tokens", type=int, default=None, help="tokens per optimizer step")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--compile", type=int, default=1, help="torch.compile on GPU (1/0)")
    ap.add_argument("--ckpt-min", type=float, default=10.0, help="minutes between checkpoints")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint and start over")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--bench-steps", type=int, default=10)
    ap.add_argument("--hours", type=float, default=None, help="with --bench: suggest --tokens for this many hours")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="change a model setting, e.g. --set n_layer=8 (applies to the model being trained)")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.bench:
        return bench(args, device)

    p = PRESETS[args.preset]
    lr = args.lr or p["lr"]
    total = int(args.tokens or p["tokens"])
    run = Run(Path(args.runs) / (args.name or args.model))
    data = Path(args.data)
    enc = get_tokenizer()

    torch.manual_seed(args.seed)
    model, cfg = build(args.model, args.preset, parse_overrides(args.set))
    model.to(device)
    T = cfg.max_len
    micro = args.micro or p["micro"]
    accum = max(1, (args.batch_tokens or p["micro"] * p["accum"] * T) // (micro * T))
    B = micro * accum
    opt = make_optimizer(model, lr, device)
    train = Windows(data / "train.bin", T, micro, seed=args.seed)
    vwin = val_set(data / "val.bin", T, p["val_windows"])
    pwin = vwin[: min(32, len(vwin))]
    hs_path = data / "hellaswag_val.jsonl"

    step, tokens, elapsed = 0, 0, 0.0
    nxt = {"eval": 0, "probe": 0, "hs": 0, "sample": 0}
    ck = run.dir / "ckpt.pt"
    if ck.exists() and not args.fresh:
        state = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step, tokens, elapsed, nxt = state["step"], state["tokens"], state["elapsed"], state["next"]
        train.set_state(state["data"])
        torch.set_rng_state(state["torch_rng"])
        print(f"resumed from {ck}: step {step}, {tokens / 1e6:.1f}M tokens, {elapsed / 60:.1f} min")
    elif args.fresh:
        for f in ("metrics.jsonl", "samples.jsonl", "snapshots.jsonl", "status.json"):
            (run.dir / f).unlink(missing_ok=True)
        for f in (run.dir / "snapshots").glob("*.json"):
            f.unlink()

    info = {
        "model": args.model, "preset": args.preset, "config": cfg.to_dict(), "params": model.num_params(),
        "params_non_embedding": model.num_params(non_embedding=True), "batch_tokens": B * T, "micro": micro,
        "context": T, "lr": lr, "total_tokens": total, "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "memory_numbers": model.memory_numbers() if isinstance(model, Engram) else None,
        "data": json.loads((data / "meta.json").read_text()) if (data / "meta.json").exists() else None,
    }
    (run.dir / "config.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    if tokens >= total:
        print("this run is already finished. Use --tokens to train longer or --fresh to start over.")
        return

    memory.fla_check(device)
    if args.compile and device.type == "cuda":
        model.enable_compile()

    stop = {"flag": False}

    def handler(sig, frame):
        print(f"\nsignal {sig}: saving a checkpoint and stopping...", flush=True)
        stop["flag"] = True
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    def ckpt_meta():
        return {"step": step, "tokens": tokens, "elapsed": elapsed, "next": nxt, "data": train.state(),
                "torch_rng": torch.get_rng_state(), "info": info, "kind": args.model, "config": cfg.to_dict()}

    latest = {}

    def evaluations(final=False):
        """Validation loss, HellaSwag, memory checks, samples, snapshot, whenever their turn comes."""
        base = {"step": step, "tokens": tokens, "elapsed": elapsed}
        if final or tokens >= nxt["eval"]:
            vl = val_loss(model, vwin, device, micro)
            run.log("metrics.jsonl", {"kind": "eval", **base, "val_loss": vl})
            latest["val_loss"] = vl
            print(f"[eval] step {step} | {tokens / 1e6:.0f}M tokens | val loss {vl:.4f}", flush=True)
            nxt["eval"] += p["eval_every"]
        if hs_path.exists() and (final or tokens >= nxt["hs"]):
            r = hellaswag.evaluate(model, hs_path, device, None if final else p["hs_quick"])
            run.log("metrics.jsonl", {"kind": "hellaswag", **base, **r, "full": final, "sample": "spread"})
            latest["hellaswag" + ("_full" if final else "")] = r["acc_norm"]
            print(f"[hellaswag] {r['acc_norm'] * 100:.2f}% on {r['n']} examples", flush=True)
            nxt["hs"] += p["hs_every"]
        if final or tokens >= nxt["probe"]:
            r = probes.memory_report(model, pwin, device, batch=min(micro, 16))
            run.log("metrics.jsonl", {"kind": "memory", **base, **r})
            latest["recall_acc"] = r["recall_acc"]
            print(f"[memory] context curve {[(s, round(l, 3)) for s, l in r['context']]} | "
                  f"repeat recall {r['recall_acc'] * 100:.1f}%", flush=True)
            if isinstance(model, Engram):
                snap = probes.snapshot(model, enc)
                name = f"{step:07d}.json"
                (run.dir / "snapshots" / name).write_text(json.dumps({**base, **snap}))
                run.log("snapshots.jsonl", {**base, "file": name})
            nxt["probe"] += p["probe_every"]
        if final or tokens >= nxt["sample"]:
            write_samples(model, enc, run, step, tokens, p["sample_len"])
            nxt["sample"] += p["sample_every"]

    model.train()
    last_ckpt = time.time()
    t_log, tok_log, acc_loss, acc_n = time.time(), tokens, 0.0, 0
    log_every = 10
    while tokens < total and not stop["flag"]:
        t0 = time.time()
        evaluations()
        t_eval = time.time() - t0

        cur_lr = lr_at(tokens / total, step, lr, p["warmup"])
        for g in opt.param_groups:
            g["lr"] = cur_lr
        loss_sum = 0.0
        for _ in range(accum):
            x, y = train.get(device)
            try:
                with amp(device):
                    _, loss, _ = model(x, y)
                (loss / accum).backward()
            except Exception as e:
                if model._compiled is None:
                    raise
                print(f"torch.compile failed ({type(e).__name__}: {e}); continuing without compile", flush=True)
                model._compiled = None
                with amp(device):
                    _, loss, _ = model(x, y)
                (loss / accum).backward()
            loss_sum += loss.detach()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        tokens += B * T
        acc_loss += loss_sum / accum
        acc_n += 1
        elapsed += time.time() - t0 - t_eval        # training time only

        if step % log_every == 0 or tokens >= total:
            now = time.time()
            tps = (tokens - tok_log) / max(1e-6, now - t_log)
            t_log, tok_log = now, tokens
            rec = {"kind": "train", "step": step, "tokens": tokens, "elapsed": elapsed,
                   "loss": float(acc_loss) / acc_n, "lr": cur_lr, "tok_s": tps, "grad_norm": float(gnorm)}
            acc_loss, acc_n = 0.0, 0
            run.log("metrics.jsonl", rec)
            eta = (total - tokens) / max(tps, 1)
            run.status({**info, "state": "running", "step": step, "tokens": tokens, "elapsed": elapsed,
                        "tok_s": tps, "eta_s": eta, "loss": rec["loss"], "latest": latest, "updated": time.time()})
            print(f"step {step:6d} | {tokens / 1e6:8.1f}M tok | loss {rec['loss']:.4f} | lr {cur_lr:.2e} | "
                  f"gnorm {float(gnorm):.2f} | {tps / 1e3:6.1f}k tok/s | eta {eta / 60:6.1f} min", flush=True)

        if time.time() - last_ckpt > args.ckpt_min * 60:
            save_ckpt(run, model, opt, ckpt_meta())
            last_ckpt = time.time()

    save_ckpt(run, model, opt, ckpt_meta())
    if tokens >= total:
        print("training done; final evaluation (full HellaSwag takes a few minutes)...", flush=True)
        evaluations(final=True)
        run.status({**info, "state": "done", "step": step, "tokens": tokens, "elapsed": elapsed,
                    "latest": latest, "updated": time.time()})
        print(f"finished: {json.dumps(latest)} after {elapsed / 60:.1f} min of training. checkpoint: {ck}")
    else:
        run.status({**info, "state": "stopped", "step": step, "tokens": tokens, "elapsed": elapsed,
                    "latest": latest, "updated": time.time()})
        print(f"stopped at step {step}; run the same command again to resume.")


if __name__ == "__main__":
    main()
