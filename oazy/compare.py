"""Measure OpenAI's own GPT-2 124M with exactly the same tests, as the line to beat.

    python -m oazy.compare --data /workspace/oazy_data --runs /workspace/oazy_runs

Loads the released weights from Hugging Face ("gpt2"), then measures validation loss on our
held-out FineWeb-Edu text, the memory checks, and full HellaSwag (all 10,042 questions). The result
is written as a run folder "openai-gpt2", so the dashboard draws it as a dashed reference line.
Takes a few minutes on a GPU. Needs `pip install transformers`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .data import val_set, default_dir
from .gpt import load_openai_gpt2
from . import hellaswag, probes
from .train import PRESETS, Run, val_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--preset", default="full", help="use the same validation windows as this training preset")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    a.data, a.runs = a.data or default_dir("data"), a.runs or default_dir("runs")
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    from oazy.data import few_cpu_threads
    few_cpu_threads(device)
    p = PRESETS[a.preset]
    data = Path(a.data)
    model = load_openai_gpt2(device)
    T = 1024 if a.preset in ("full", "big") else PRESETS[a.preset]["gpt"].get("max_len", 1024)
    run = Run(Path(a.runs) / "openai-gpt2")
    for f in ("metrics.jsonl", "status.json"):
        (run.dir / f).unlink(missing_ok=True)
    info = {"model": "openai-gpt2", "params": model.num_params(), "context": T,
            "trained_tokens_note": "WebText, 40 GB of Reddit-linked pages (2019)"}
    (run.dir / "config.json").write_text(json.dumps(info, indent=1))
    base = {"step": 0, "tokens": 0, "elapsed": 0}
    vwin = val_set(data / "val.bin", T, p["val_windows"])
    t0 = time.time()
    vl, bpb = val_loss(model, vwin, device, 16, bpb=True)
    run.log("metrics.jsonl", {"kind": "eval", **base, "val_loss": vl, "val_bpb": bpb})
    print(f"val loss {vl:.4f} ({bpb:.4f} bits per byte)")
    rep = probes.memory_report(model, vwin[:32], device)
    run.log("metrics.jsonl", {"kind": "memory", **base, **rep})
    print(f"context curve {rep['context']}  exact recall {rep['recall_acc'] * 100:.1f}%")
    latest = {"val_loss": vl, "val_bpb": bpb, "recall_acc": rep["recall_acc"]}
    hs = data / "hellaswag_val.jsonl"
    if hs.exists():
        r = hellaswag.evaluate(model, hs, device)
        run.log("metrics.jsonl", {"kind": "hellaswag", **base, **r, "full": True})
        latest["hellaswag_full"], latest["hellaswag_char_full"] = r["acc_norm"], r["acc_char"]
        print(f"HellaSwag {r['acc_char'] * 100:.2f}% per letter ({r['n']} questions; {r['acc_norm'] * 100:.2f}% per token)")
    run.status({**info, "state": "done", "latest": latest, "updated": time.time()})
    print(f"done in {time.time() - t0:.0f} s -> {run.dir}")


if __name__ == "__main__":
    main()
