"""Train Memtrix, Tide, Fern, Ember, Fern2 or the GPT-2 baseline on FineWeb-Edu. Resumable: re-run the same command.

    python -m oazy.train --model memtrix --data /workspace/oazy_data --runs /workspace/oazy_runs
    python -m oazy.train --model gpt    --data /workspace/oazy_data --runs /workspace/oazy_runs
    python -m oazy.train --bench --hours 2     # measure both models' speed; suggest a token budget

Both models see exactly the same text in the same order, with the same schedule and optimizer.
Everything the dashboard shows is written into the run folder:
  status.json     live status (step, tokens, speed, ETA, latest results)
  metrics.jsonl   training loss, validation loss, HellaSwag, memory checks (memtrix/probes.py)
  samples.jsonl   text written by the model as it learns
  snapshots/      Memtrix only: the probe text read with the lid off
  ckpt.pt         checkpoint (every few minutes, on Ctrl-c and at the end)
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

from .data import Windows, val_set, get_tokenizer, default_dir, run_dir, EOT
from .gpt import GPT, GPTConfig
from memtrix.model import Memtrix, MemtrixConfig
from memtrix import memory
from . import hellaswag, probes

KINDS = ["memtrix", "tide", "fern", "ember", "fern2", "gpt"]
NICE = {"memtrix": "Memtrix", "tide": "Tide", "fern": "Fern", "ember": "Ember", "fern2": "Fern2", "gpt": "GPT"}
FAST = ("ember", "fern2")       # raised the fast way: Muon and the GPT-2 medium mentor by default


def say(*a, **k):
    """print, but a closed terminal (tmux killed, `tee` gone) must never crash training before it saves."""
    try:
        print(*a, **k)
    except (BrokenPipeError, OSError):
        sys.stdout = sys.stderr = open(os.devnull, "w")


def _tide():
    from tide.model import Tide, TideConfig          # tide/ builds on this package
    return Tide, TideConfig


def _fern():
    from fern.model import Fern, FernConfig          # fern/ reads with Memtrix's memories
    return Fern, FernConfig


def _ember():
    from ember.model import Ember, EmberConfig       # ember/ builds on this package
    return Ember, EmberConfig


def _fern2():
    from fern2.model import Fern2, Fern2Config       # fern2/ = Fern's body + Ember's way of training
    return Fern2, Fern2Config

PRESETS = {
    # a laptop-CPU test of the whole pipeline (minutes)
    "tiny": dict(
        memtrix=dict(d_model=128, n_layer=4, n_head=2, head_k=64, head_v=64, mlp_hidden=256, max_len=256),
        gpt=dict(d_model=128, n_layer=4, n_head=2, max_len=256),
        tide=dict(refine_hidden=256, lookahead=4),
        fern=dict(n_layer=4, unfold_layers=2, unfold_heads=2, unfold_hidden=256),
        ember=dict(grow="1:0,2:0.1,4:0.25"), mentor="none",
        fern2=dict(n_layer=4, unfold_layers=2, unfold_heads=2, unfold_hidden=256, grow="2:0,4:0.25", unfold_grow="1:0,2:0.25"),
        micro=16, accum=1, lr=3e-3, warmup=50, tokens=3_000_000,
        eval_every=250_000, probe_every=1_000_000, hs_every=1_000_000, sample_every=1_000_000,
        val_windows=64, hs_quick=200, sample_len=40,
    ),
    # a 1-minute smoke test of every code path
    "smoke": dict(
        memtrix=dict(d_model=64, n_layer=2, n_head=1, head_k=64, head_v=64, mlp_hidden=128, max_len=128),
        gpt=dict(d_model=64, n_layer=2, n_head=1, max_len=128),
        tide=dict(refine_hidden=128, lookahead=4),
        fern=dict(n_layer=2, unfold_layers=1, unfold_heads=1, unfold_hidden=128),
        ember=dict(grow="1:0,2:0.3"), mentor="none",
        fern2=dict(n_layer=2, unfold_layers=1, unfold_heads=1, unfold_hidden=128, grow="1:0,2:0.3", unfold_grow="1:0"),
        micro=8, accum=2, lr=3e-3, warmup=10, tokens=60_000,
        eval_every=20_000, probe_every=30_000, hs_every=30_000, sample_every=30_000,
        val_windows=16, hs_quick=20, sample_len=10,
    ),
    # GPT-2 small size (Memtrix 117M vs GPT-2 124M), ~2 h for both models on one B200 (check with --bench)
    "full": dict(
        memtrix=dict(), gpt=dict(), micro_fern2=16,
        micro=32, accum=8, lr=1.2e-3, warmup=400, tokens=2_000_000_000,
        eval_every=50_000_000, probe_every=250_000_000, hs_every=250_000_000, sample_every=250_000_000,
        val_windows=128, hs_quick=1000, sample_len=120,
    ),
    # the same models, 2.5x the text (e.g. a 2 x B200 pod for ~2.5 h, one model per GPU)
    "big": dict(
        memtrix=dict(), gpt=dict(), micro_fern2=16,
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
    if kind == "memtrix":
        cfg = MemtrixConfig(**{**p["memtrix"], **(overrides or {})})
        return Memtrix(cfg), cfg
    if kind == "tide":
        Tide, TideConfig = _tide()
        cfg = TideConfig(**{**p["memtrix"], **p.get("tide", {}), **(overrides or {})})
        return Tide(cfg), cfg
    if kind == "fern":
        Fern, FernConfig = _fern()
        cfg = FernConfig(**{**p["memtrix"], **p.get("fern", {}), **(overrides or {})})
        return Fern(cfg), cfg
    if kind == "ember":
        Ember, EmberConfig = _ember()
        cfg = EmberConfig(**{**p["memtrix"], **p.get("ember", {}), **(overrides or {})})
        return Ember(cfg), cfg
    if kind == "fern2":
        Fern2, Fern2Config = _fern2()
        cfg = Fern2Config(**{**p["memtrix"], **p.get("fern2", {}), **(overrides or {})})
        return Fern2(cfg), cfg
    cfg = GPTConfig(**{**p["gpt"], **(overrides or {})})
    return GPT(cfg), cfg


def load_model(ckpt: Path, device):
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    kind = st.get("kind", "memtrix")
    if kind == "engram":                      # Memtrix's name before Nov 5
        kind = st["kind"] = "memtrix"
    if kind == "memtrix":
        model = Memtrix(MemtrixConfig.from_dict(st["config"]))
    elif kind == "tide":
        Tide, TideConfig = _tide()
        model = Tide(TideConfig.from_dict(st["config"]))
    elif kind == "fern":
        Fern, FernConfig = _fern()
        model = Fern(FernConfig.from_dict(st["config"]))
    elif kind == "ember":
        Ember, EmberConfig = _ember()
        model = Ember(EmberConfig.from_dict(st["config"]))
    elif kind == "fern2":
        Fern2, Fern2Config = _fern2()
        model = Fern2(Fern2Config.from_dict(st["config"]))
    else:
        model = GPT(GPTConfig.from_dict(st["config"]))
    model.load_state_dict(st["model"])
    return model.to(device).eval(), st


def lr_at(progress, step, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    return base * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


MUON_GRIDS = ("mem.qkv.weight", "mem.gate.weight", "mem.out.weight", "mlp.up.weight", "mlp.down.weight",
              "glance.qkv.weight", "glance.out.weight", ".up.weight", ".down.weight")    # (last four: Fern's unfolder)


def make_optimizer(model, lr, device, kind="adamw", muon_lr=0.02):
    """AdamW for everything, or (kind="muon") Muon for the blocks' weight grids + AdamW for the rest.
    Every group's learning rate follows the same schedule: lr_scale is its size relative to --lr."""
    named = list(model.named_parameters())
    grids = [(n, p) for n, p in named if kind == "muon" and p.dim() == 2 and n.endswith(MUON_GRIDS)]
    gids = {id(p) for _, p in grids}
    rest = [(n, p) for n, p in named if id(p) not in gids]
    decay = [p for n, p in rest if p.dim() >= 2 and "conv_w" not in n]
    no_decay = [p for n, p in rest if p.dim() < 2 or "conv_w" in n]
    kw = dict(lr=lr, betas=(0.9, 0.95), eps=1e-8)
    if device.type == "cuda":
        kw["fused"] = True
    adam = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1, "lr_scale": 1.0},
                              {"params": no_decay, "weight_decay": 0.0, "lr_scale": 1.0}], **kw)
    if not grids:
        return adam
    from .muon import Muon, Combined
    for n, p in grids:                       # fused grids are straightened part by part
        if n.endswith("mem.qkv.weight"):
            H, K, V = model.cfg.n_head, model.cfg.head_k, model.cfg.head_v
            p.muon_split = [H * K, H * K, H * V]
        elif n.endswith("glance.qkv.weight"):
            p.muon_split = [p.shape[0] // 3] * 3
        elif n.endswith(".up.weight"):           # two halves: the gate and the value of a SwiGLU
            p.muon_split = [p.shape[0] // 2] * 2
    muon = Muon([{"params": [p for _, p in grids], "lr_scale": muon_lr / lr}], lr=muon_lr)
    return Combined(muon, adam)


def carry_state(old, new):
    """After the model grew: keep the optimizer memory (momentum etc.) of the parameters that existed."""
    olds = old.opts if hasattr(old, "opts") else [old]
    news = new.opts if hasattr(new, "opts") else [new]
    for o, n in zip(olds, news):
        for p, st in o.state.items():
            n.state[p] = st


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
    say(f"device {device}  ({torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu'})")
    p = PRESETS[args.preset]
    micro = args.micro or p.get("micro_" + args.model, p["micro"])
    total = int(args.tokens or p["tokens"])
    speeds = {}
    kinds = [args.model, "gpt"] if args.model != "gpt" else ["memtrix", "gpt"]
    for kind in kinds:
        model, cfg = build(kind, args.preset, parse_overrides(args.set) if kind != "gpt" else None)
        if hasattr(model, "grow_to"):
            model.grow_to(1.0)                         # Ember: measure it fully grown
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
        if kind != "gpt":
            extra = f"  (memory kernel: {'flash-linear-attention' if memory._FLA_OK else 'pure PyTorch chunked'})"
        say(f"{kind:7s} params {model.num_params() / 1e6:6.1f}M  {dt * 1000:7.1f} ms per {micro}x{T}  "
              f"{tps / 1e3:8.1f}k tok/s  peak mem {mem:5.1f} GB{extra}")
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    me = kinds[0]
    te, tg = total / speeds[me], total / speeds["gpt"]
    say(f"\n{total / 1e9:.2f}B tokens each: {NICE[me]} {te / 3600:.2f} h, GPT {tg / 3600:.2f} h on their own.")
    say(f"  one GPU, both at once: about {(te + tg) / 3600:.2f} h")
    say(f"  two GPUs, one each:    about {max(te, tg) / 3600:.2f} h")
    if args.hours:
        share = args.hours * 3600 / (1 / speeds[me] + 1 / speeds["gpt"])
        two = args.hours * 3600 * min(speeds.values())
        say(f"\nto finish in {args.hours:g} h (+ a few minutes of evaluations):")
        say(f"  {NICE[me]} alone on a GPU: --tokens {args.hours * 3600 * speeds[me] * 0.93 / 1e9:.2f}e9")
        say(f"  one GPU:  --tokens {share * 0.93 / 1e9:.2f}e9")
        say(f"  two GPUs: --tokens {two * 0.93 / 1e9:.2f}e9")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=KINDS + ["engram"], default="memtrix")
    ap.add_argument("--preset", choices=list(PRESETS), default="full")
    ap.add_argument("--data", default=None)
    ap.add_argument("--runs", default=None)
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
    ap.add_argument("--opt", choices=["adamw", "muon"], default=None, help="optimizer (default: muon for ember, adamw otherwise)")
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--mentor", default=None, help="Ember's mentor: gpt2-medium (default), gpt2-large, gpt2-xl or none")
    ap.add_argument("--mentor-weight", type=float, default=0.6, help="mentor's share of the lesson at the start")
    ap.add_argument("--mentor-until", type=float, default=0.5, help="fraction of training after which the mentor is silent")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="change a model setting, e.g. --set n_layer=8 (applies to the model being trained)")
    args = ap.parse_args()
    args.data, args.runs = args.data or default_dir("data"), args.runs or default_dir("runs")
    if args.model == "engram":                # the old name of Memtrix
        args.model = "memtrix"

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.bench:
        return bench(args, device)

    p = PRESETS[args.preset]
    lr = args.lr or p["lr"]
    total = int(args.tokens or p["tokens"])
    run = Run(run_dir(args.runs, args.name or args.model))
    data = Path(args.data)
    enc = get_tokenizer()

    torch.manual_seed(args.seed)
    model, cfg = build(args.model, args.preset, parse_overrides(args.set))
    model.to(device)
    T = cfg.max_len
    micro = args.micro or p.get("micro_" + args.model, p["micro"])
    accum = max(1, (args.batch_tokens or p["micro"] * p["accum"] * T) // (micro * T))
    B = micro * accum
    opt_kind = args.opt or ("muon" if args.model in FAST else "adamw")
    new_opt = lambda: make_optimizer(model, lr, device, opt_kind, args.muon_lr)   # noqa: E731
    opt = new_opt()
    mentor_name = args.mentor or (p.get("mentor", "gpt2-medium") if args.model in FAST else "none")
    train = Windows(data / "train.bin", T, micro, seed=args.seed)
    vwin = val_set(data / "val.bin", T, p["val_windows"])
    pwin = vwin[: min(32, len(vwin))]
    hs_path = data / "hellaswag_val.jsonl"

    step, tokens, elapsed = 0, 0, 0.0
    nxt = {"eval": 0, "probe": 0, "hs": 0, "sample": 0}
    ck = run.dir / "ckpt.pt"
    if ck.exists() and not args.fresh:
        state = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])            # (Ember grows to the checkpoint's depth here)
        opt = new_opt()
        opt.load_state_dict(state["opt"])
        step, tokens, elapsed, nxt = state["step"], state["tokens"], state["elapsed"], state["next"]
        train.set_state(state["data"])
        torch.set_rng_state(state["torch_rng"].cpu())
        say(f"resumed from {ck}: step {step}, {tokens / 1e6:.1f}M tokens, {elapsed / 60:.1f} min")
    elif args.fresh:
        for f in ("metrics.jsonl", "samples.jsonl", "snapshots.jsonl", "status.json"):
            (run.dir / f).unlink(missing_ok=True)
        for f in (run.dir / "snapshots").glob("*.json"):
            f.unlink()

    info = {
        "model": args.model, "preset": args.preset, "config": cfg.to_dict(),
        "params": model.full_params() if hasattr(model, "full_params") else model.num_params(),
        "params_non_embedding": (model.full_params() if hasattr(model, "full_params") else model.num_params())
        - model.embed.weight.numel(), "optimizer": opt_kind, "mentor": mentor_name, "batch_tokens": B * T, "micro": micro,
        "context": T, "lr": lr, "total_tokens": total, "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "memory_numbers": model.memory_numbers() if hasattr(model, "memory_numbers") else None,
        "data": json.loads((data / "meta.json").read_text()) if (data / "meta.json").exists() else None,
    }
    (run.dir / "config.json").write_text(json.dumps(info, indent=1))
    say(json.dumps(info, indent=1))
    if tokens >= total:
        say("this run is already finished. Use --tokens to train longer or --fresh to start over.")
        return

    try:
        prev = json.loads((run.dir / "status.json").read_text())
    except Exception:
        prev = {}
    run.status({**prev, **info, "state": "starting", "step": step, "tokens": tokens, "elapsed": elapsed,
                "updated": time.time()})

    memory.fla_check(device)
    if args.compile and device.type == "cuda":
        model.enable_compile()
    mentor = None
    if mentor_name != "none" and tokens < args.mentor_until * total:
        from ember.mentor import Mentor
        say(f"loading the mentor {mentor_name} ...", flush=True)
        mentor = Mentor(mentor_name, device, args.mentor_weight, args.mentor_until, compile=bool(args.compile))

    stop = {"flag": False}

    def handler(sig, frame):
        stop["flag"] = True
        if sig == getattr(signal, "SIGHUP", None):
            # the terminal (and the `tee` log) went away: keep writing to nowhere so the save still happens
            devnull = open(os.devnull, "w")
            sys.stdout = sys.stderr = devnull
            return
        try:
            say(f"\nsignal {sig}: saving a checkpoint and stopping...", flush=True)
        except Exception:
            pass
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, handler)      # closing the tmux session also saves first

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
            say(f"[eval] step {step} | {tokens / 1e6:.0f}M tokens | val loss {vl:.4f}", flush=True)
            nxt["eval"] += p["eval_every"]
        if hs_path.exists() and (final or tokens >= nxt["hs"]):
            r = hellaswag.evaluate(model, hs_path, device, None if final else p["hs_quick"])
            run.log("metrics.jsonl", {"kind": "hellaswag", **base, **r, "full": final, "sample": "spread"})
            latest["hellaswag" + ("_full" if final else "")] = r["acc_norm"]
            say(f"[hellaswag] {r['acc_norm'] * 100:.2f}% on {r['n']} examples", flush=True)
            nxt["hs"] += p["hs_every"]
        if final or tokens >= nxt["probe"]:
            r = probes.memory_report(model, pwin, device, batch=min(micro, 16))
            run.log("metrics.jsonl", {"kind": "memory", **base, **r})
            latest["recall_acc"] = r["recall_acc"]
            say(f"[memory] context curve {[(s, round(l, 3)) for s, l in r['context']]} | "
                  f"repeat recall {r['recall_acc'] * 100:.1f}%", flush=True)
            if hasattr(model, "belt"):
                from tide.model import draft_report
                r = draft_report(model, pwin, device, batch=min(micro, 16))
                run.log("metrics.jsonl", {"kind": "drafts", **base, **r})
                latest["draft_acc"] = r["draft_acc"]
                say(f"[drafts] loss by words ahead {[round(v, 3) for v in r['draft_loss']]} | "
                      f"top-1 {[round(v * 100, 1) for v in r['draft_acc']]}%", flush=True)
            if hasattr(model, "unfold"):
                from fern.model import round_report
                r = round_report(model, pwin, device, batch=min(micro, 16))
                run.log("metrics.jsonl", {"kind": "rounds", **base, **r})
                latest["round_loss"] = r["round_loss"]
                say(f"[rounds] loss by round {[round(v, 3) for v in r['round_loss']]} | "
                      f"top-1 {[round(v * 100, 1) for v in r['round_acc']]}%", flush=True)
            if isinstance(model, Memtrix):
                snap = probes.snapshot(model, enc)
                if hasattr(model, "belt"):
                    from tide.model import foresight
                    snap.update(foresight(model, enc, probes.PROBE_TEXT))
                name = f"{step:07d}.json"
                (run.dir / "snapshots" / name).write_text(json.dumps({**base, **snap}))
                run.log("snapshots.jsonl", {**base, "file": name})
            nxt["probe"] += p["probe_every"]
        if final or tokens >= nxt["sample"]:
            write_samples(model, enc, run, step, tokens, p["sample_len"])
            nxt["sample"] += p["sample_every"]

    ctl = run.dir / "control.json"
    ctl.unlink(missing_ok=True)                  # an old request must not stop a fresh start

    def control():
        """The app's buttons: pause / resume / end early (a small file written by oazy.dashboard)."""
        if not ctl.exists():
            return None
        try:
            act = json.loads(ctl.read_text()).get("action")
        except Exception:
            return None
        ctl.unlink(missing_ok=True)
        return act

    def status(state, **extra):
        run.status({**info, "state": state, "step": step, "tokens": tokens, "elapsed": elapsed,
                    "latest": latest, "updated": time.time(), **extra})

    end_early = False
    model.train()
    last_ckpt = time.time()
    t_log, tok_log, acc_loss, acc_n = time.time(), tokens, 0.0, 0
    log_every = 10
    while tokens < total and not stop["flag"]:
        act = control()
        if act == "pause":
            save_ckpt(run, model, opt, ckpt_meta())
            last_ckpt = time.time()
            say(f"paused at step {step} (checkpoint saved). Resume or end it from the app.", flush=True)
            status("paused")
            while not stop["flag"]:
                act = control()
                if act in ("resume", "end"):
                    break
                status("paused")                     # keeps the card fresh
                time.sleep(2)
            if act == "resume":
                say("resumed", flush=True)
                status("running")
                t_log, tok_log, acc_loss, acc_n = time.time(), tokens, 0.0, 0
        if act == "end":
            end_early = True
            break
        if stop["flag"]:
            break
        t0 = time.time()
        evaluations()
        t_eval = time.time() - t0

        if hasattr(model, "grow_to") and model.grow_to(tokens / total):     # Ember, Fern2: time to grow
            old, opt = opt, new_opt()
            carry_state(old, opt)
            deep = model.depth_text() if hasattr(model, "depth_text") else f"{model.depth} blocks deep"
            say(f"[grow] {tokens / 1e6:.0f}M tokens: {NICE[args.model]} is now {deep}", flush=True)
            run.log("metrics.jsonl", {"kind": "grow", "step": step, "tokens": tokens, "elapsed": elapsed,
                                      "depth": model.depth})
        mw = mentor.weight_at(tokens / total) if mentor is not None else 0.0
        if mentor is not None and mw <= 0:
            say(f"[mentor] {tokens / 1e6:.0f}M tokens: the mentor steps back; {NICE[args.model]} learns alone now", flush=True)
            mentor = None
            if device.type == "cuda":
                torch.cuda.empty_cache()

        cur_lr = lr_at(tokens / total, step, lr, p["warmup"])
        for g in opt.param_groups:
            g["lr"] = cur_lr * g.get("lr_scale", 1.0)
        loss_sum, kd_sum, rd_sum = 0.0, 0.0, 0.0

        def lesson(x, y):
            with amp(device):
                logits, loss, st = model(x, y)
                # Ember learns the mentor's guesses with its own; Fern2 with its reader (st["kd_logits"])
                student = st.get("kd_logits", None if hasattr(model, "mix_mentor") else logits)
                if mw > 0 and student is not None:
                    from ember.mentor import mentor_loss
                    kd = mentor_loss(student, mentor.guess(x))
                    if hasattr(model, "mix_mentor"):
                        loss = model.mix_mentor(loss, st, kd, mw)
                    else:
                        loss = (1 - mw) * loss + mw * kd
                    st = {**st, "kd": kd}
            (loss / accum).backward()
            return loss, st

        for _ in range(accum):
            x, y = train.get(device)
            try:
                loss, st = lesson(x, y)
            except Exception as e:
                if model._compiled is None:
                    raise
                say(f"torch.compile failed ({type(e).__name__}: {e}); continuing without compile", flush=True)
                model._compiled = None
                loss, st = lesson(x, y)
            loss_sum += st.get("ce", loss).detach()       # the next-word loss (Tide also trains its far drafts)
            if "kd" in st:
                kd_sum += st["kd"].detach()
            if "reader_ce" in st:
                rd_sum += st["reader_ce"]
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
            if hasattr(model, "grow_to"):
                rec["depth"] = model.depth
            if mw > 0:
                rec["mentor_w"], rec["mentor_kl"] = mw, float(kd_sum) / accum
            if rd_sum:
                rec["reader_loss"] = float(rd_sum) / accum
            acc_loss, acc_n = 0.0, 0
            run.log("metrics.jsonl", rec)
            eta = (total - tokens) / max(tps, 1)
            run.status({**info, "state": "running", "step": step, "tokens": tokens, "elapsed": elapsed,
                        "tok_s": tps, "eta_s": eta, "loss": rec["loss"], "latest": latest, "updated": time.time()})
            say(f"step {step:6d} | {tokens / 1e6:8.1f}M tok | loss {rec['loss']:.4f} | lr {cur_lr:.2e} | "
                  f"gnorm {float(gnorm):.2f} | {tps / 1e3:6.1f}k tok/s | eta {eta / 60:6.1f} min", flush=True)

        if time.time() - last_ckpt > args.ckpt_min * 60:
            save_ckpt(run, model, opt, ckpt_meta())
            last_ckpt = time.time()

    save_ckpt(run, model, opt, ckpt_meta())
    if tokens >= total or end_early:
        if end_early:
            say(f"ended early from the app at {tokens / 1e6:.0f}M tokens (checkpoint saved).", flush=True)
        status("finishing")
        say("training done; final evaluation (full HellaSwag takes a few minutes)...", flush=True)
        evaluations(final=True)
        status("done", ended_early=end_early)
        say(f"finished: {json.dumps(latest)} after {elapsed / 60:.1f} min of training. checkpoint: {ck}")
    else:
        run.status({**info, "state": "stopped", "step": step, "tokens": tokens, "elapsed": elapsed,
                    "latest": latest, "updated": time.time()})
        say(f"stopped at step {step}; run the same command again to resume.")


if __name__ == "__main__":
    main()
