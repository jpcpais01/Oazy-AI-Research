"""Bee5 in speedrun mode: the NanoGPT speedrun's own text and target, so Bee5's recipe can be held up against the
public records (github.com/KellerJordan/modded-nanogpt). FineWeb (not -Edu) cut into GPT-2's pieces, read in its
fixed order; scored on the first 10,485,760 tokens of FineWeb's validation set; the target is 3.28 (Karpathy's GPT-2
small reproduction). One GPU here, where the records use 8 H100s: a B200 is roughly 2.5 H100s, so multiply our
minutes by about 0.3 to compare.

    python3 -m bee5.speedrun --prepare 10            # once: the validation shard + 10 x 100M training tokens (~2 GB)
    python3 -m bee5.speedrun                         # Bee5 (GPT-2 pieces, no mentor), 1,000M tokens, val every 50M
    python3 -m bee5.speedrun --tokens 0.6e9          # a shorter budget (the schedule fits itself to it)

What is the same as the records: the text, its order, the validation tokens, the target. What is not: one GPU, Bee5's
size (~203M with GPT-2's table, more than the records' 124M body), no mentor, our own schedule (Bee5's, fitted to the
budget). It prints the training minutes and tokens when the held-out loss first reaches 3.28, and the final loss.
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch

from oazy import term
from oazy.data import default_dir, few_cpu_threads

REPO = "kjj0/fineweb10B-gpt2"
VAL_TOKENS = 10_485_760
HEADER = 256          # int32s before the tokens in each shard


def folder(data=None) -> Path:
    return Path(data or default_dir("data")) / "fineweb10B-gpt2"


def prepare(n: int, data=None):
    from huggingface_hub import hf_hub_download
    out = folder(data)
    out.mkdir(parents=True, exist_ok=True)
    for name in ["fineweb_val_%06d.bin" % 0] + ["fineweb_train_%06d.bin" % i for i in range(1, n + 1)]:
        if not (out / name).exists():
            term.info(f"downloading {name}")
            hf_hub_download(repo_id=REPO, filename=name, repo_type="dataset", local_dir=str(out))
    term.info(f"ready: {out}")


def shard(path: str) -> np.ndarray:
    head = np.fromfile(path, dtype=np.int32, count=HEADER)
    if head[0] != 20240520 or head[1] != 1:
        raise SystemExit(f"{path}: not a speedrun shard (magic {head[0]})")
    return np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER * 4, shape=(int(head[2]),))


class Stream:
    """The training tokens in their fixed order: shard after shard, window after window."""

    def __init__(self, files):
        self.files, self.i, self.pos = files, 0, 0
        self.tok = shard(files[0])

    def get(self, B, T, device):
        need = B * T + 1
        if self.pos + need > len(self.tok):
            self.i += 1
            if self.i >= len(self.files):
                raise SystemExit("out of training text: --prepare more shards")
            self.tok, self.pos = shard(self.files[self.i]), 0
        c = torch.from_numpy(self.tok[self.pos:self.pos + need].astype(np.int64))
        self.pos += B * T
        c = c.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else c.to(device)
        return c[:-1].view(B, T), c[1:].view(B, T)


@torch.no_grad()
def val_loss(model, val, Tv, micro, device, amp):
    model.eval()
    n = len(val) // Tv
    tot = 0.0
    for i in range(0, n, micro):
        c = torch.from_numpy(val[i * Tv:min(n, i + micro) * Tv + 1].astype(np.int64)).to(device)
        b = (len(c) - 1) // Tv
        x, y = c[:b * Tv].view(b, Tv), c[1:b * Tv + 1].view(b, Tv)
        with amp():
            loss, _ = model.losses(x, y)
        tot += float(loss) * b
    model.train()
    return tot / n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepare", type=int, default=None, help="download the validation shard and this many 100M-token training shards")
    ap.add_argument("--data", default=None)
    ap.add_argument("--tokens", type=float, default=1.0e9, help="the training budget (the schedule fits itself to it)")
    ap.add_argument("--target", type=float, default=3.28)
    ap.add_argument("--val-every", type=float, default=50e6, help="tokens between held-out checks (not timed)")
    ap.add_argument("--micro", type=int, default=32, help="sequences per pass at 2,048 words")
    ap.add_argument("--batch-tokens", type=int, default=524288)
    ap.add_argument("--lr", type=float, default=1.8e-3)
    ap.add_argument("--muon-lr", type=float, default=0.03)
    ap.add_argument("--warmup", type=int, default=300, help="learning-rate warm-up steps")
    ap.add_argument("--decay", type=float, default=0.3, help="the cool-down's share of the run")
    ap.add_argument("--val-tokens", type=int, default=VAL_TOKENS, help="(tests only) fewer held-out tokens")
    ap.add_argument("--device", default=None)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Bee5 settings, e.g. --set n_layer=8")
    a = ap.parse_args(argv)
    if a.prepare is not None:
        return prepare(a.prepare, a.data)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    few_cpu_threads(dev)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    d = folder(a.data)
    train_files = sorted(glob.glob(str(d / "fineweb_train_*.bin")))
    if not train_files or not (d / "fineweb_val_000000.bin").exists():
        raise SystemExit(f"no speedrun text in {d}: python3 -m bee5.speedrun --prepare 10")
    from bee5.model import Bee5, Bee5Config
    from oazy.train import parse_overrides, lr_wsd, batch_at, make_optimizer, amp
    cfg = Bee5Config(**{"vocab_size": 50304, "tokenizer": "gpt2", **parse_overrides(a.set)})
    torch.manual_seed(0)
    model = Bee5(cfg).to(dev).train()
    if a.compile and dev.type == "cuda":
        model.enable_compile()
    opt = make_optimizer(model, a.lr, dev, "muon", a.muon_lr, 0.1)
    val = shard(str(d / "fineweb_val_000000.bin"))[:a.val_tokens + 1]
    Tv = min(cfg.max_len, 2048)
    stream = Stream(train_files)
    term.header("Bee5 speedrun mode", torch.cuda.get_device_name() if dev.type == "cuda" else "cpu")
    term.rows([("model", f"{model.num_params() / 1e6:.1f}M, {model.layout()}"), ("text", f"{len(train_files)} shards of FineWeb, GPT-2 pieces"),
               ("budget", f"{a.tokens / 1e9:.2f}B tokens, target {a.target}")])
    total, tokens, step, elapsed, hit = int(a.tokens), 0, 0, 0.0, None
    next_val, log = a.val_every, []
    amp_ = lambda: amp(dev)        # noqa: E731
    while tokens < total:
        q = tokens / total
        model.set_progress(q)
        T = model.ctx_at(q)
        micro = max(1, a.micro * cfg.max_len // T)
        accum = max(1, round(batch_at(q, a.batch_tokens, "262144:0.05") / (micro * T)))
        lr = lr_wsd(q, step, a.lr, a.warmup, a.decay, "sqrt")
        for g in opt.param_groups:
            g["lr"] = lr * g.get("lr_scale", 1.0)
        t0 = time.time()
        for _ in range(accum):
            x, y = stream.get(micro, T, dev)
            with amp_():
                loss, st = model.losses(x, y)
            (loss / accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        if step >= 3:                        # the first steps compile: not counted (the records warm up untimed too)
            elapsed += time.time() - t0
        step += 1
        tokens += micro * accum * T
        if tokens >= next_val or tokens >= total:
            next_val += a.val_every
            vl = val_loss(model, val, Tv, a.micro, dev, amp_)
            log.append({"tokens": tokens, "minutes": elapsed / 60, "val": vl, "train": float(st["ce"])})
            if hit is None and vl <= a.target:
                hit = log[-1]
            term.info(f"{tokens / 1e6:7.0f}M tokens  {elapsed / 60:6.1f} min  held-out {vl:.4f}  train {float(st['ce']):.3f}"
                      + ("  <- target" if hit is log[-1] else ""))
    res = {"final": log[-1], "hit": hit, "log": log, "params": model.num_params(), "tokens": total, "target": a.target}
    term.header("result", color="green")
    term.rows([("held-out at the end", f"{log[-1]['val']:.4f} after {total / 1e9:.2f}B tokens, {elapsed / 60:.1f} min of training"),
               ("target " + str(a.target), f"reached at {hit['tokens'] / 1e6:.0f}M tokens, {hit['minutes']:.1f} min" if hit
                else "not reached: try a larger --tokens")])
    out = Path(default_dir("runs")) / "speedrun-bee5.json"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1))
    except OSError:
        pass
    return res


if __name__ == "__main__":
    main()
