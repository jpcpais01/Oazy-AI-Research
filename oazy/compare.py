"""Measure published models with exactly the same tests as ours: the lines to beat.

    python -m oazy.compare                          # OpenAI's GPT-2 124M -> run "openai-gpt2"
    python -m oazy.compare --model smollm2-135m     # Hugging Face's SmolLM2-135M -> run "smollm2-135m"
    python -m oazy.compare --model smollm2-360m     # SmolLM2-360M (Bee4's mentor) -> run "smollm2-360m"
    python -m oazy.compare --model smollm2          # both SmolLM2s

SmolLM2 is scored like our SmolLM2-tokenizer models (Coral, Bee4): held-out loss on the same text re-cut with its
tokenizer and turned into "per GPT-2 token", loss by kind of text, the memory checks, full HellaSwag, and
ARC-Easy / PIQA / LAMBADA in full (once `python3 -m oazy.evals --prepare` has run). A few minutes each on a GPU;
small enough to run next to a training run.

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

SMOL_INFO = {"smollm2-135m": "~2,000,000M tokens (SmolLM2 corpus: FineWeb-Edu, DCLM, The Stack, math; 2024)",
             "smollm2-360m": "~4,000,000M tokens (SmolLM2 corpus: FineWeb-Edu, DCLM, The Stack, math; 2024)",
             "smollm2-1.7b": "~11,000,000M tokens (SmolLM2 corpus: FineWeb-Edu, DCLM, The Stack, math; 2024)"}
SMOL_TOKENS = {"smollm2-135m": 2e12, "smollm2-360m": 4e12, "smollm2-1.7b": 11e12}


def smol(name, device):
    """SmolLM2 behind our models' small interface (cfg.max_len / cfg.tokenizer, _body, forward(x, y))."""
    import torch.nn.functional as F
    from ember.mentor import HFMentor, SMOL

    class Ref(HFMentor):
        def forward(self, x, y=None):
            logits = self._body(x)
            loss = None if y is None else F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
            return logits, loss, None

        def num_params(self):
            return sum(p.numel() for p in self.hf.parameters())

    m = Ref(SMOL[name], device)
    m.cfg.tokenizer = "smollm2"
    if device.type == "cuda":
        m.hf.to(torch.bfloat16)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai-gpt2", choices=["openai-gpt2", "smollm2-135m", "smollm2-360m", "smollm2"])
    ap.add_argument("--data", default=None)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--preset", default="full", help="use the same validation windows as this training preset")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    a.data, a.runs = a.data or default_dir("data"), a.runs or default_dir("runs")
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    from oazy.data import few_cpu_threads
    few_cpu_threads(device)
    for name in (["smollm2-135m", "smollm2-360m"] if a.model == "smollm2" else [a.model]):
        measure(name, a, device)


def measure(name, a, device):
    p = PRESETS[a.preset]
    root = Path(a.data)
    T = 1024 if a.preset in ("full", "big") else PRESETS[a.preset]["gpt"].get("max_len", 1024)
    if name == "openai-gpt2":
        model, data, scale = load_openai_gpt2(device), root, 1.0
        info = {"model": name, "params": model.num_params(), "context": T,
                "trained_tokens_note": "WebText, 40 GB of Reddit-linked pages (2019)"}
    else:
        from .retok import target_dir, ratio
        model = smol(name, device)
        data = root / "tagus20" if (root / "tagus20" / "val.bin").exists() else target_dir(root, "smollm2")
        if not (data / "val.bin").exists():
            raise SystemExit(f"no SmolLM2-tokenizer held-out text in {data}: the Coral/Bee4 data (tagus20) is needed")
        scale = ratio(data, "val")            # its losses per SmolLM2 token -> per GPT-2 token, like Coral's and Bee4's
        info = {"model": name, "params": model.num_params(), "context": T, "tokenizer": "smollm2",
                "trained_tokens": SMOL_TOKENS[name], "trained_tokens_note": SMOL_INFO[name]}
    run = Run(Path(a.runs) / name)
    for f in ("metrics.jsonl", "status.json"):
        (run.dir / f).unlink(missing_ok=True)
    (run.dir / "config.json").write_text(json.dumps(info, indent=1))
    base = {"step": 0, "tokens": 0, "elapsed": 0}
    vwin = val_set(data / "val.bin", T, p["val_windows"])
    t0 = time.time()
    print(f"== {name}: {info['params'] / 1e6:.0f}M parameters, held-out text from {data}")
    vl, bpb = val_loss(model, vwin, device, 16, bpb=True)
    extra = {}
    if name != "openai-gpt2":
        from .mix import SOURCES, source_path
        for src in SOURCES:
            sp = source_path(data, src, "val")
            if src != "fwe" and sp.exists():
                extra[f"val_loss_{src}"] = val_loss(model, val_set(sp, T, 32), device, 16) * scale
        extra["val_loss_own"] = vl
    vl *= scale
    run.log("metrics.jsonl", {"kind": "eval", **base, "val_loss": vl, "val_bpb": bpb, **extra})
    print(f"val loss {vl:.4f} ({bpb:.4f} bits per byte)")
    rep = probes.memory_report(model, vwin[:32], device)
    run.log("metrics.jsonl", {"kind": "memory", **base, **rep})
    print(f"context curve {rep['context']}  exact recall {rep['recall_acc'] * 100:.1f}%")
    latest = {"val_loss": vl, "val_bpb": bpb, "recall_acc": rep["recall_acc"], **extra}
    hs = root / "hellaswag_val.jsonl"
    if hs.exists():
        r = hellaswag.evaluate(model, hs, device)
        run.log("metrics.jsonl", {"kind": "hellaswag", **base, **r, "full": True})
        latest["hellaswag_full"], latest["hellaswag_char_full"] = r["acc_norm"], r["acc_char"]
        print(f"HellaSwag {r['acc_char'] * 100:.2f}% per letter ({r['n']} questions; {r['acc_norm'] * 100:.2f}% per token)")
    from . import evals
    ev = evals.evaluate_all(model, root, device)
    if ev:
        run.log("metrics.jsonl", {"kind": "evals", **base, **ev, "full": True})
        latest.update({k: v for k, v in ev.items() if not k.endswith("_n")})
        print("  ".join(f"{k} {ev[k] * 100:.1f}%" for k in ("arc_easy", "piqa", "lambada") if k in ev))
    run.status({**info, "state": "done", "latest": latest, "updated": time.time()})
    print(f"done in {time.time() - t0:.0f} s -> {run.dir}")


if __name__ == "__main__":
    main()
