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

from .data import Windows, val_set, default_dir, run_dir, few_cpu_threads
from . import tok
from .gpt import GPT, GPTConfig
from memtrix.model import Memtrix, MemtrixConfig
from memtrix import memory
from . import hellaswag, probes

KINDS = ["memtrix", "tide", "fern", "ember", "fern2", "fern3", "bee", "bee2", "bee3", "gpt"]
NICE = {"memtrix": "Memtrix", "tide": "Tide", "fern": "Fern", "ember": "Ember", "fern2": "Fern2", "fern3": "Fern3",
        "bee": "Bee0.2B", "bee2": "Bee_v2_0.2B", "bee3": "Bee3", "gpt": "GPT"}
FAST = ("ember", "fern2", "fern3", "bee", "bee2", "bee3")    # raised the fast way: Muon and a mentor by default
MENTOR = {"bee": "gpt2-large", "bee2": "smollm2-360m", "bee3": "smollm2-360m"}   # default mentor (the others: gpt2-medium)
BEES = ("bee", "bee2", "bee3")                 # the text mix, WSD, growing windows
# Bee3's recipe (bee3/README.md): the mentor stays the whole run, weight decay on the Muon grids, a batch ramp,
# a 1-minus-square-root cool-down, twice Bee2's learning rates (the presets' lr_bee3 / muon_lr_bee3)
RECIPE = {"bee3": dict(mentor_floor=0.25, mentor_off=0.1, muon_wd=0.1, batch_ramp="65536:0.1", decay_shape="sqrt")}
VAL_T = 1024                                   # held-out windows are always this long: every model, every run, same text


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


def _fern3():
    from fern3.model import Fern3, Fern3Config       # fern3/ = Fern's unfolding + gist-indexed recall
    return Fern3, Fern3Config


def _bee():
    from bee.model import Bee, BeeConfig             # bee/ = modern hybrid: Memtrix memory layers + attention, 3:1
    return Bee, BeeConfig


def _bee2():
    from bee2.model import Bee2, Bee2Config          # bee2/ = Bee0.2B's body, SmolLM2's tokenizer and mentor
    return Bee2, Bee2Config


def _bee3():
    from bee3.model import Bee3, Bee3Config          # bee3/ = Bee2's body, no bleed between documents, the Bee3 recipe
    return Bee3, Bee3Config


PRESETS = {
    # a laptop-CPU test of the whole pipeline (minutes)
    "tiny": dict(
        memtrix=dict(d_model=128, n_layer=4, n_head=2, head_k=64, head_v=64, mlp_hidden=256, max_len=256),
        gpt=dict(d_model=128, n_layer=4, n_head=2, max_len=256),
        tide=dict(refine_hidden=256, lookahead=4),
        fern=dict(n_layer=4, unfold_layers=2, unfold_heads=2, unfold_hidden=256),
        ember=dict(grow="1:0,2:0.1,4:0.25"), mentor="none",
        fern2=dict(n_layer=4, unfold_layers=2, unfold_heads=2, unfold_hidden=256, grow="2:0,4:0.25", unfold_grow="1:0,2:0.25"),
        fern3=dict(n_layer=2, unfold_layers=2, think_layers=1, unfold_heads=2, unfold_hidden=256, max_len=512,
                   grow="1:0,2:0.25", unfold_grow="1:0,2:0.25"),
        bee=dict(d_model=128, n_layer=4, n_head=2, head_k=64, head_v=64, attn_heads=4, attn_kv_heads=2, attn_head_dim=32,
                 mlp_hidden=256, max_len=512, ctx="256:0,512:0.5"),
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
        fern3=dict(n_layer=2, unfold_layers=1, think_layers=1, unfold_heads=1, unfold_hidden=128, max_len=256,
                   grow="1:0,2:0.3", unfold_grow="1:0", recall_warm_until=0.2),
        bee=dict(d_model=64, n_layer=4, n_head=1, head_k=32, head_v=32, attn_heads=2, attn_kv_heads=1, attn_head_dim=32,
                 mlp_hidden=128, max_len=128, ctx="64:0,128:0.5"),
        micro=8, accum=2, lr=3e-3, warmup=10, tokens=60_000,
        eval_every=20_000, probe_every=30_000, hs_every=30_000, sample_every=30_000,
        val_windows=16, hs_quick=20, sample_len=10,
    ),
    # GPT-2 small size (Memtrix 117M vs GPT-2 124M), ~2 h for both models on one B200 (check with --bench)
    "full": dict(
        memtrix=dict(), gpt=dict(), micro_fern2=16, micro_fern3=4, batch_bee=524288, batch_bee3=262144,
        micro=32, accum=8, lr=1.2e-3, lr_bee3=2.4e-3, muon_lr_bee3=0.04, warmup=400, tokens=2_000_000_000,
        eval_every=50_000_000, probe_every=250_000_000, hs_every=250_000_000, sample_every=250_000_000,
        val_windows=128, hs_quick=1000, sample_len=120,
    ),
    # the same models, 2.5x the text (e.g. a 2 x B200 pod for ~2.5 h, one model per GPU)
    "big": dict(
        memtrix=dict(), gpt=dict(), micro_fern2=16, micro_fern3=4, batch_bee=524288, batch_bee3=262144,
        micro=32, accum=8, lr=1.2e-3, lr_bee3=2.4e-3, muon_lr_bee3=0.04, warmup=600, tokens=5_000_000_000,
        eval_every=100_000_000, probe_every=200_000_000, hs_every=200_000_000, sample_every=200_000_000,
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
    if kind == "fern3":
        Fern3, Fern3Config = _fern3()
        cfg = Fern3Config(**{**p["memtrix"], **p.get("fern3", {}), **(overrides or {})})
        return Fern3(cfg), cfg
    if kind == "bee":
        Bee, BeeConfig = _bee()
        cfg = BeeConfig(**{**p.get("bee", {}), **(overrides or {})})
        return Bee(cfg), cfg
    if kind == "bee2":
        Bee2, Bee2Config = _bee2()
        cfg = Bee2Config(**{**p.get("bee", {}), **p.get("bee2", {}), **(overrides or {})})
        return Bee2(cfg), cfg
    if kind == "bee3":
        Bee3, Bee3Config = _bee3()
        small = {k: v for k, v in p.get("bee", {}).items() if k != "ctx"}    # the test presets' small body; Bee3 keeps one window length
        cfg = Bee3Config(**{**small, **({"ctx": f"{small['max_len']}:0"} if "max_len" in small else {}),
                            **p.get("bee3", {}), **(overrides or {})})
        return Bee3(cfg), cfg
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
    elif kind == "fern3":
        Fern3, Fern3Config = _fern3()
        model = Fern3(Fern3Config.from_dict(st["config"]))
    elif kind == "bee":
        Bee, BeeConfig = _bee()
        model = Bee(BeeConfig.from_dict(st["config"]))
    elif kind == "bee2":
        Bee2, Bee2Config = _bee2()
        model = Bee2(Bee2Config.from_dict(st["config"]))
    elif kind == "bee3":
        Bee3, Bee3Config = _bee3()
        model = Bee3(Bee3Config.from_dict(st["config"]))
    else:
        model = GPT(GPTConfig.from_dict(st["config"]))
    model.load_state_dict(st["model"])
    return model.to(device).eval(), st


def lr_at(progress, step, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    return base * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def lr_wsd(progress, step, base, warmup, decay=0.2, shape="linear"):
    """Warm up, stay flat, then fall to zero over the last `decay` of training ("warmup-stable-decay", as in
    MiniCPM, SmolLM2/3, DeepSeek). The fall lines up with the cool-down data. shape: "linear" (a straight
    line) or "sqrt" (1 minus the square root of how far into the fall we are: it drops faster at first and
    lands gently; Hagele et al. 2024 found it a little better at the same budget). Bee3 uses "sqrt"."""
    if step < warmup:
        return base * (step + 1) / warmup
    left = min(1.0, max(0.0, (1 - progress) / decay))          # 1 before the fall, 0 at the end
    if shape == "sqrt" and left < 1.0:
        return base * (1 - math.sqrt(1 - left))
    return base * left


def batch_at(progress, batch_tokens, ramp):
    """Tokens per step: `ramp` = "65536:0.1" starts at 65,536 and grows in a straight line to batch_tokens by
    10% of the run (early on a small batch learns just as much per token, so the same tokens buy more steps;
    GPT-3 and DeepSeek ramp this way). None = batch_tokens throughout."""
    if not ramp:
        return batch_tokens
    start, until = ramp.split(":")
    start, until = min(int(float(start)), batch_tokens), float(until)
    if progress >= until or until <= 0:
        return batch_tokens
    return int(start + (batch_tokens - start) * progress / until)


MUON_GRIDS = ("mem.qkv.weight", "mem.gate.weight", "mem.out.weight", "mlp.up.weight", "mlp.down.weight",
              "glance.qkv.weight", "glance.out.weight", ".up.weight", ".down.weight",    # (these four: Fern's unfolder)
              "attn.q.weight", "attn.k.weight", "attn.v.weight", "attn.gate.weight", "attn.out.weight")   # Bee's attention


def make_optimizer(model, lr, device, kind="adamw", muon_lr=0.02, muon_wd=0.0):
    """AdamW for everything, or (kind="muon") Muon for the blocks' weight grids + AdamW for the rest.
    Every group's learning rate follows the same schedule: lr_scale is its size relative to --lr.
    muon_wd: weight decay on the Muon grids, on AdamW's scale (0.1 = what AdamW's 0.1 does; Bee3)."""
    named = list(model.named_parameters())
    grids = [(n, p) for n, p in named if kind == "muon" and p.dim() == 2 and n.endswith(MUON_GRIDS)]
    gids = {id(p) for _, p in grids}
    rest = [(n, p) for n, p in named if id(p) not in gids]
    keep = lambda n, p: p.dim() < 2 or "conv_w" in n or (getattr(model, "embed_no_decay", False) and n.startswith("embed"))  # noqa: E731
    decay = [p for n, p in rest if not keep(n, p)]
    no_decay = [p for n, p in rest if keep(n, p)]
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
    muon = Muon([{"params": [p for _, p in grids], "lr_scale": muon_lr / lr}], lr=muon_lr, weight_decay=muon_wd)
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
def val_loss(model, windows, device, micro, bpb=False):
    """Average loss per token on fixed held-out windows. bpb=True: also bits per byte of text, a number
    that does not depend on how the text was cut into tokens (fair across tokenizers)."""
    model.eval()
    xy = torch.from_numpy(windows).to(device)
    tot = 0.0
    for i in range(0, len(xy), micro):
        with amp(device):
            _, loss, _ = model(xy[i:i + micro, :-1], xy[i:i + micro, 1:])
        tot += float(loss) * len(xy[i:i + micro])
    model.train()
    vl = tot / len(xy)
    if not bpb:
        return vl
    nb = int(tok.bytes_per_token(tok.name_of(model))[windows[:, 1:]].sum())
    return vl, vl * windows[:, 1:].size / max(nb, 1) / math.log(2)


@torch.no_grad()
def write_samples(model, enc, run, step, tokens, n_tok):
    model.eval()
    for p in SAMPLE_PROMPTS:
        torch.manual_seed(1234)
        ids, _ = model.generate(enc.encode_ordinary(p), max_new_tokens=n_tok, temperature=0.8, top_k=40, eos=enc.eot_token)
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
        T = model.ctx_at(0) if hasattr(model, "ctx_at") else cfg.max_len     # Bee: its first window length
        x = torch.randint(0, min(50257, cfg.vocab_size), (micro, T), device=device)
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
        mname = args.mentor or MENTOR.get(kind, "none")
        if kind in BEES and mname != "none":             # Bee's real cost includes its mentor's reading
            from ember.mentor import Mentor
            me = Mentor(mname, device, compile=bool(args.compile))
            mt = []
            for i in range(args.bench_steps + 3):
                t0 = time.time()
                with amp(device):
                    me.guess(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                if i >= 3:
                    mt.append(time.time() - t0)
            mtps = micro * T / float(np.median(mt))
            u = args.mentor_until
            sec = total * (u * (1 / tps + 1 / mtps) + (1 - u) / tps)
            say(f"mentor  {mname}: {mtps / 1e3:8.1f}k tok/s reading only. {NICE[kind]} with it for the first {u:.0%}: "
                f"about {sec / 3600:.1f} h for {total / 1e9:.1f}B tokens (+ ~5-10% for 2,048-word windows and tests)")
            del me
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
    ap.add_argument("--micro", type=int, default=None, help="sequences per forward pass at the run's first window length (lower it if out of memory; Bee3 at 2,048 words: 16 by default)")
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
    ap.add_argument("--muon-lr", type=float, default=None, help="Muon's learning rate for the weight grids (0.02; Bee3 0.04)")
    ap.add_argument("--muon-wd", type=float, default=None, help="weight decay on the Muon grids, AdamW's scale (0; Bee3 0.1)")
    ap.add_argument("--batch-ramp", default=None, help="tokens per step at the start and when the full batch is reached, "
                    "e.g. 65536:0.1 (Bee3's default); 'none' = full batch from the start")
    ap.add_argument("--decay-shape", choices=["linear", "sqrt"], default=None, help="the WSD fall: linear, or sqrt (Bee3)")
    ap.add_argument("--loss", choices=["fused", "plain"], default="fused",
                    help="Bee's loss in slices (fused, default) or on the full 32-bit scores (plain; slower, same numbers)")
    ap.add_argument("--mentor", default=None, help="the mentor: gpt2-medium (Ember's default), gpt2-large (Bee's), gpt2-xl, "
                    "smollm2-360m (Bee_v2_0.2B's; also smollm2-135m, smollm2-1.7b; only for SmolLM2-tokenizer models) or none")
    ap.add_argument("--mentor-weight", type=float, default=0.6, help="mentor's share of the lesson at the start")
    ap.add_argument("--mentor-until", type=float, default=0.5, help="fraction of training by which the mentor's share has fallen to --mentor-floor")
    ap.add_argument("--mentor-floor", type=float, default=None, help="the share it keeps from then on (0 = it leaves; Bee3 0.25)")
    ap.add_argument("--mentor-off", type=float, default=None, help="over this last fraction of the run the share falls to 0 (Bee3 0.1)")
    ap.add_argument("--schedule", choices=["cos", "wsd"], default=None,
                    help="learning-rate shape: cos (cosine) or wsd (flat, then a straight fall over the last 20%%); default wsd for bee")
    ap.add_argument("--mix", default=None, help="Bee's text mix over training (default in oazy/mix.py), "
                    "e.g. 'fwe:0.65,dclm:0.35@0;fwe:0.45,dclm:0.25,chat:0.3@0.8'")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="change a model setting, e.g. --set n_layer=8 (applies to the model being trained)")
    args = ap.parse_args()
    args.data, args.runs = args.data or default_dir("data"), args.runs or default_dir("runs")
    if args.model == "engram":                # the old name of Memtrix
        args.model = "memtrix"

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    few_cpu_threads(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.bench:
        return bench(args, device)

    p = PRESETS[args.preset]
    recipe = RECIPE.get(args.model, {})
    lr = args.lr or p.get("lr_" + args.model, p["lr"])
    muon_lr = args.muon_lr or p.get("muon_lr_" + args.model, 0.02)
    muon_wd = recipe.get("muon_wd", 0.0) if args.muon_wd is None else args.muon_wd
    batch_ramp = recipe.get("batch_ramp") if args.batch_ramp is None else (None if args.batch_ramp == "none" else args.batch_ramp)
    decay_shape = args.decay_shape or recipe.get("decay_shape", "linear")
    mentor_floor = recipe.get("mentor_floor", 0.0) if args.mentor_floor is None else args.mentor_floor
    mentor_off = recipe.get("mentor_off", 0.0) if args.mentor_off is None else args.mentor_off
    run = Run(run_dir(args.runs, args.name or args.model))
    root = Path(args.data)

    torch.manual_seed(args.seed)
    model, cfg = build(args.model, args.preset, parse_overrides(args.set))
    model.to(device)
    tk = tok.name_of(model)
    enc = tok.get(tk)
    # a model that reads another tokenizer than GPT-2's reads the same text, re-cut (oazy/retok.py)
    data = root
    val_scale = 1.0
    if tk != "gpt2":
        from .retok import target_dir, ratio
        data = target_dir(root, tk)
        if not (data / "train.bin").exists():
            raise SystemExit(f"no {tk} text in {data} yet: run  python3 -m oazy.retok  first (bash runpod/setup.sh bee2)")
        # its losses per token, turned into "per GPT-2 token" so that they sit on the same scale as
        # every other model's: (nats per text) is the same thing whoever cuts the text
        val_scale = ratio(data, "val")
    total = int(args.tokens or p["tokens"] * (val_scale if args.tokens is None else 1))
    if tk != "gpt2" and args.tokens is None:
        say(f"{p['tokens'] / 1e9:.2f}B GPT-2 tokens of text = {total / 1e9:.2f}B {tk} tokens (the same text as the GPT-2 models)")
    T = cfg.max_len
    micro0 = args.micro or p.get("micro_" + args.model, p["micro"])
    # tokens per step: the preset's (at its usual 1,024-word windows, so Fern3's 4,096-word windows keep 262k)
    batch_tokens = args.batch_tokens or p.get("batch_" + args.model) or p["micro"] * p["accum"] * min(T, 1024)
    growing_ctx = hasattr(model, "ctx_at")             # Bee: 1,024-word windows first, then 2,048 (Bee3: 2,048 throughout)
    T0 = model.ctx_at(0) if growing_ctx else T
    Tv = min(T0, VAL_T) if args.model in BEES else T0   # held-out windows: 1,024 for every Bee, so the losses compare
    # a preset's micro is for 1,024-word windows and is halved at 2,048, so every pass holds the same tokens; --micro given
    # by hand means that many sequences per pass at the run's first window length
    micro_ref = max(1, micro0 * min(T0, 1024) // 1024) if args.model in BEES and not args.micro else micro0

    def shape(T, progress=0.0):
        """Sequences per pass and passes per step for windows of T words (same words per step throughout,
        except for a batch ramp at the start)."""
        mi = max(1, micro_ref * 1024 // T) if args.model in BEES and not args.micro else max(1, micro_ref * T0 // T)
        return mi, max(1, round(batch_at(progress, batch_tokens, batch_ramp) / (mi * T)))
    T = T0
    micro, accum = shape(T)
    B = micro * accum
    if hasattr(model, "fused_loss"):
        type(model).fused_loss = args.loss == "fused"
    opt_kind = args.opt or ("muon" if args.model in FAST else "adamw")
    new_opt = lambda: make_optimizer(model, lr, device, opt_kind, muon_lr, muon_wd)   # noqa: E731
    opt = new_opt()
    mentor_name = args.mentor or (p.get("mentor", MENTOR.get(args.model, "gpt2-medium")) if args.model in FAST else "none")
    if mentor_name not in ("none", "random", "smol-random") and mentor_name.startswith("smollm2") != (tk == "smollm2"):
        raise SystemExit(f"the mentor {mentor_name} cuts text differently from {NICE[args.model]} ({tk} tokenizer): "
                         f"use a {'smollm2-*' if tk == 'smollm2' else 'gpt2-*'} mentor or --mentor none")
    schedule = args.schedule or ("wsd" if args.model in BEES else "cos")
    lr_fn = (lambda pr, st, base, wu: lr_wsd(pr, st, base, wu, shape=decay_shape)) if schedule == "wsd" else lr_at
    if args.model in BEES:
        from .mix import Mix, DEFAULT_MIX, resolve_mix, source_path
        from bee.model import ctx_plan
        if args.mix is None:      # SmolLM2-tokenizer Bees read Tagus when it is there (oazy/tagus.py)
            args.mix = "tagus" if tk == "smollm2" and (data / "cosmo" / "train.bin").exists() else "v1"
        args.mix = resolve_mix(args.mix)
        train = Mix(data, args.mix, T, micro, seed=args.seed,
                    lengths=[t for _, t in ctx_plan(cfg)], log=say)
        xval = {n: val_set(source_path(data, n, "val"), Tv, 32) for n in train.src
                if n != "fwe" and source_path(data, n, "val").exists()}
    else:
        train = Windows(data / "train.bin", T, micro, seed=args.seed)
        xval = {}
    vwin = val_set(data / "val.bin", Tv, p["val_windows"])
    pwin = vwin[: min(32, len(vwin))]
    hs_path = root / "hellaswag_val.jsonl"
    if tk != "gpt2":                                  # training loss: weigh each source's own ratio by the mix
        from .retok import ratio
        from .mix import source_path
        src_ratio = {n: ratio(source_path(data, n).parent, "train", val_scale) for n in getattr(train, "src", {})}

        def loss_scale_at(progress):
            w = train.weights(progress) if hasattr(train, "weights") else {}
            return sum(v * src_ratio.get(n, val_scale) for n, v in w.items()) or val_scale
    else:
        loss_scale_at = lambda progress: 1.0      # noqa: E731

    step, tokens, elapsed = 0, 0, 0.0
    nxt = {"eval": 0, "probe": 0, "hs": 0, "sample": 0}
    old_total, anchor = None, None
    ck = run.dir / "ckpt.pt"
    if ck.exists() and not args.fresh:
        state = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])            # (Ember grows to the checkpoint's depth here)
        opt = new_opt()
        opt.load_state_dict(state["opt"])
        step, tokens, elapsed, nxt = state["step"], state["tokens"], state["elapsed"], state["next"]
        for k in nxt:                                    # tests made more frequent since: the next one comes sooner
            every = p[k + "_every"]
            nxt[k] = min(nxt[k], math.ceil(tokens / every) * every)
        train.set_state(state["data"])
        if growing_ctx:
            T = train.T
            micro, accum = shape(T, tokens / total)
            B = micro * accum
            train.set_batch(micro)
        torch.set_rng_state(state["torch_rng"].cpu())
        old_total = state.get("total") or (state.get("info") or {}).get("total_tokens")
        anchor = state.get("mentor_anchor")
        say(f"resumed from {ck}: step {step}, {tokens / 1e6:.1f}M tokens, {elapsed / 60:.1f} min")
        if old_total and old_total != total and tokens < total:
            say(f"[length] the run's length changes: {old_total / 1e9:.2f}B -> {total / 1e9:.2f}B tokens. Every "
                f"schedule follows the new length (cool-down, mentor, windows, text mix); the learning rate does not jump.")
            if schedule == "wsd":
                was, now = lr_fn(tokens / old_total, step, 1.0, p["warmup"]), lr_fn(tokens / total, step, 1.0, p["warmup"])
                if abs(was - now) > 1e-6:
                    raise SystemExit(f"with {total / 1e9:.2f}B tokens this run would already be in its cool-down "
                                     f"(learning rate x{now:.2f} instead of x{was:.2f}). Pick --tokens of at least "
                                     f"{tokens / 0.8 / 1e9:.2f}e9, or keep the old length.")
            else:
                say("[length] note: with the cosine schedule the learning rate moves to the new curve at once")
    elif args.fresh:
        for f in ("metrics.jsonl", "samples.jsonl", "snapshots.jsonl", "status.json"):
            (run.dir / f).unlink(missing_ok=True)
        for f in (run.dir / "snapshots").glob("*.json"):
            f.unlink()

    info = {
        "model": args.model, "preset": args.preset, "config": cfg.to_dict(),
        "params": model.full_params() if hasattr(model, "full_params") else model.num_params(),
        "params_non_embedding": (model.full_params() if hasattr(model, "full_params") else model.num_params())
        - model.embed.weight.numel(), "optimizer": opt_kind, "mentor": mentor_name, "batch_tokens": batch_tokens, "micro": micro,
        "context": cfg.max_len if growing_ctx else T, "lr": lr, "schedule": schedule,
        **({"muon_lr": muon_lr, "muon_wd": muon_wd} if opt_kind == "muon" else {}),
        **({"decay_shape": decay_shape} if schedule == "wsd" else {}), **({"batch_ramp": batch_ramp} if batch_ramp else {}),
        **({"mentor_weight": args.mentor_weight, "mentor_until": args.mentor_until, "mentor_floor": mentor_floor,
            "mentor_off": mentor_off} if mentor_name != "none" else {}),
        **({"loss": args.loss, "doc_mask": bool(getattr(cfg, "doc_mask", 0)), "val_context": Tv} if args.model in BEES else {}),
        "layout": model.layout() if hasattr(model, "layout") else None,
        "mix": args.mix or (DEFAULT_MIX if args.model in BEES else None), "total_tokens": total, "device": str(device),
        "tokenizer": tk, "loss_scale": val_scale,
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "memory_numbers": model.memory_numbers() if hasattr(model, "memory_numbers") else None,
        "data": json.loads((data / "meta.json").read_text()) if (data / "meta.json").exists() else None,
        **({"loss_note": f"losses are per GPT-2 token (own per-token loss x {val_scale:.3f}, the {tk}/GPT-2 token "
                         "ratio on the same text), so they compare directly with the GPT-2-tokenizer models; "
                         "*_own fields are per own token"} if tk != "gpt2" else {}),
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
    from ember.mentor import fade
    share_now = fade(tokens / total, args.mentor_weight, args.mentor_until, None, mentor_floor, mentor_off)
    if mentor_name != "none" and share_now > 0:
        from ember.mentor import Mentor
        say(f"loading the mentor {mentor_name} ...", flush=True)
        mentor = Mentor(mentor_name, device, args.mentor_weight, args.mentor_until, compile=bool(args.compile))
        mentor.floor, mentor.off = mentor_floor, mentor_off
    if anchor is not None and mentor is not None:      # an earlier change of length: keep fading from there
        mentor.anchor = (anchor["tokens"] / total, anchor["weight"])
    if old_total and old_total != total and tokens < total and mentor_name != "none":
        # today's share under the old length, then a straight fade from it to the floor at mentor_until of the new length
        a_old = (anchor["tokens"] / old_total, anchor["weight"]) if anchor else None
        w_now = fade(tokens / old_total, args.mentor_weight, args.mentor_until, a_old, mentor_floor, mentor_off)
        anchor = {"tokens": tokens, "weight": w_now}
        if mentor is not None:
            mentor.anchor = (tokens / total, w_now)
            say(f"[length] the mentor's share is {w_now:.2f} now and fades to 0 by "
                f"{args.mentor_until * total / 1e9:.2f}B tokens", flush=True)
        elif w_now > 0:
            say(f"[length] the mentor stops now (its share was {w_now:.2f}): with the new length it is past "
                f"{args.mentor_until:.0%} of the run", flush=True)

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
                "torch_rng": torch.get_rng_state(), "info": info, "kind": args.model, "config": cfg.to_dict(),
                "total": total, "mentor_anchor": anchor}

    latest = {}

    def evaluations(final=False):
        """Validation loss, HellaSwag, memory checks, samples, snapshot, whenever their turn comes."""
        base = {"step": step, "tokens": tokens, "elapsed": elapsed}
        if final or tokens >= nxt["eval"]:
            vl, bpb = val_loss(model, vwin, device, micro0, bpb=True)
            extra = {f"val_loss_{n}": val_loss(model, w, device, micro0) * val_scale for n, w in xval.items()}
            own = {"val_loss_own": vl} if tk != "gpt2" else {}
            vl *= val_scale
            run.log("metrics.jsonl", {"kind": "eval", **base, "val_loss": vl, "val_bpb": bpb, **own, **extra})
            latest.update(extra)
            latest["val_loss"], latest["val_bpb"] = vl, bpb
            say(f"[eval] step {step} | {tokens / 1e6:.0f}M tokens | val loss {vl:.4f}"
                + (f" (GPT-2-equivalent; {own['val_loss_own']:.4f} per {tk} token)" if own else "")
                + f" | {bpb:.4f} bits per byte", flush=True)
            nxt["eval"] += p["eval_every"]
        if hs_path.exists() and (final or tokens >= nxt["hs"]):
            r = hellaswag.evaluate(model, hs_path, device, None if final else p["hs_quick"])
            run.log("metrics.jsonl", {"kind": "hellaswag", **base, **r, "full": final, "sample": "spread"})
            latest["hellaswag" + ("_full" if final else "")] = r["acc_norm"]
            latest["hellaswag_char" + ("_full" if final else "")] = r["acc_char"]
            say(f"[hellaswag] {r['acc_norm'] * 100:.2f}% on {r['n']} examples "
                f"({r['acc_char'] * 100:.2f}% scored per letter)", flush=True)
            nxt["hs"] += p["hs_every"]
        if final and (root / "evals").exists():          # ARC-Easy, PIQA, LAMBADA (oazy/evals.py), once, at the end
            from . import evals
            r = evals.evaluate_all(model, root, device)
            if r:
                run.log("metrics.jsonl", {"kind": "evals", **base, **r})
                latest.update({k: v for k, v in r.items() if not k.endswith("_n")})
                say("[evals] " + ", ".join(f"{k} {v * 100:.1f}%" if k in ("arc_easy", "piqa", "lambada") else f"{k} {v:.2f}"
                                           for k, v in r.items() if not k.endswith(("_n", "_char"))), flush=True)
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

        if growing_ctx and model.ctx_at(tokens / total) != T:          # Bee: time for longer windows
            T = model.ctx_at(tokens / total)
            micro, accum = shape(T, tokens / total)
            B = micro * accum
            train.set_block(T)
            train.set_batch(micro)
            say(f"[context] {tokens / 1e6:.0f}M tokens: {NICE[args.model]} now reads windows of {T} words "
                f"({micro} x {accum} per step)", flush=True)
            run.log("metrics.jsonl", {"kind": "context", "step": step, "tokens": tokens, "elapsed": elapsed, "context": T})
        if batch_ramp and accum != shape(T, tokens / total)[1]:         # the batch ramp: more passes per step
            micro, accum = shape(T, tokens / total)
            B = micro * accum
        if hasattr(train, "progress"):
            train.progress = tokens / total
        cur_lr = lr_fn(tokens / total, step, lr, p["warmup"])
        for g in opt.param_groups:
            g["lr"] = cur_lr * g.get("lr_scale", 1.0)
        loss_sum, kd_sum, rd_sum, hs_sum = 0.0, 0.0, 0.0, 0.0

        def lesson(x, y):
            with amp(device):
                if hasattr(model, "losses"):        # Bee: the loss and the mentor's lesson in slices (oazy/lmloss.py)
                    loss, st = model.losses(x, y, mentor if mw > 0 else None, mw)
                    (loss / accum).backward()
                    return loss, st
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
            if "recall_hindsight" in st:
                hs_sum += st["recall_hindsight"]
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
            own_loss = float(acc_loss) / acc_n
            rec = {"kind": "train", "step": step, "tokens": tokens, "elapsed": elapsed,
                   "loss": own_loss * loss_scale_at(tokens / total), "lr": cur_lr, "tok_s": tps, "grad_norm": float(gnorm)}
            if tk != "gpt2":
                rec["loss_own"] = own_loss
            if hasattr(model, "grow_to"):
                rec["depth"] = model.depth
            if hasattr(train, "weights"):
                rec["context"], rec["mix"] = T, {k: round(v, 3) for k, v in train.weights(tokens / total).items()}
                rec["batch"] = B * T
            if mw > 0:
                rec["mentor_w"], rec["mentor_kl"] = mw, float(kd_sum) / accum
            if rd_sum:
                rec["reader_loss"] = float(rd_sum) / accum
            if hs_sum:
                rec["recall_hindsight"] = float(hs_sum) / accum
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
