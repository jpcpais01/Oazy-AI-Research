"""Train one of the two minds on stories; same data, same batches, same steps, same optimizer.

    python3 -m textmind.train --arm sent --minutes 40 --out /workspace/oazy_runs/textmind/sent
    python3 -m textmind.train --arm gpt  --minutes 40 --out /workspace/oazy_runs/textmind/gpt
    python3 -m textmind.compare /workspace/oazy_runs/textmind

Logs bits per byte on held-out stories (the sentence mind's is an upper bound: mouth + surprise).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .data import as_stream, load
from .model import ByteGPT, SentenceMind, SentenceMind2


def evaluate(model, arm, va, dev, n=512, bs=64):
    model.eval()
    bits = nb = 0.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        for i in range(0, min(n, len(va)), bs):
            x = va[i:i + bs].to(dev).long()
            _, b, c, _ = model(x) if arm != "gpt" else model(as_stream(x))
            bits += b.item(); nb += c.item()
    model.train()
    return bits / nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["sent", "sent2", "gpt"], required=True)
    ap.add_argument("--minutes", type=float, default=40)
    ap.add_argument("--steps", type=int, default=0, help="stop after this many steps (0 = by time)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--S", type=int, default=12)
    ap.add_argument("--L", type=int, default=96)
    ap.add_argument("--train-mb", type=int, default=400)
    ap.add_argument("--data", default="/workspace/oazy_runs/textmind/data")
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d = load(a.data, a.S, a.L, a.train_mb)
    tr, va = d["train"], d["valid"]
    print(f"stories: train {len(tr)}, valid {len(va)}", flush=True)
    model = (SentenceMind(a.L) if a.arm == "sent" else SentenceMind2(a.L, S=a.S) if a.arm == "sent2" else ByteGPT(ctx=a.S * (a.L + 1))).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"{a.arm}: {nparam / 1e6:.1f}M parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05, betas=(0.9, 0.95),
                            **(dict(fused=True) if dev.type == "cuda" else {}))
    fwd = torch.compile(model) if a.compile else model
    log = open(out / "log.jsonl", "a")
    g = torch.Generator().manual_seed(a.seed)
    t0 = time.time(); step = 0; seen = 0
    total = a.minutes * 60
    while True:
        el = time.time() - t0
        frac = min(1.0, step / a.steps) if a.steps else min(1.0, el / total)
        lr = a.lr * min(1.0, step / 300) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for pg in opt.param_groups:
            pg["lr"] = lr
        idx = torch.randint(0, len(tr), (a.batch,), generator=g)
        x = tr[idx].to(dev, non_blocking=True).long()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            loss, bits, n, extra = fwd(x) if a.arm != "gpt" else fwd(as_stream(x))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1; seen += a.batch
        if step % 200 == 0:
            row = dict(step=step, min=round(el / 60, 2), stories=seen, loss=round(loss.item(), 4),
                       train_bpb=round((bits / n).item(), 4), **{k: round(v.item(), 4) for k, v in extra.items()})
            if step % 1000 == 0:
                row["valid_bpb"] = round(evaluate(model, a.arm, va, dev), 4)
            log.write(json.dumps(row) + "\n"); log.flush()
            print(json.dumps(row), flush=True)
        if (a.steps and step >= a.steps) or (not a.steps and el >= total):
            break
    v = evaluate(model, a.arm, va, dev, n=len(va))
    res = dict(arm=a.arm, params=nparam, steps=step, stories=seen, minutes=round((time.time() - t0) / 60, 1), valid_bpb=round(v, 4))
    print(json.dumps(res), flush=True)
    (out / "result.json").write_text(json.dumps(res, indent=1))
    torch.save(dict(model=model.state_dict(), args=vars(a), result=res), out / "model.pt")


if __name__ == "__main__":
    main()
