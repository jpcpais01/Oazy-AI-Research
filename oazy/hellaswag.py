"""HellaSwag: pick the sensible ending of an everyday scene out of four.

Example: "A man is sitting on a roof. He ..."  ->  "starts pulling up roofing on a roof." (right)
vs three endings that are grammatical but silly. Random guessing scores 25%.

A model "chooses" the ending it finds least surprising: we measure its average loss on the
ending's tokens after reading the context and pick the lowest (the "acc_norm" scoring used by
nanoGPT / llm.c). We also report "acc_char": the ending's total loss divided by its number of
letters, which is what lm-evaluation-harness calls acc_norm. It does not depend on how a tokenizer
cuts the endings, so it is the fairest way to compare models with different tokenizers (Bee_v2_0.2B
reads SmolLM2's pieces, the others GPT-2's). Each model is given the text cut its own way. OpenAI's GPT-2 124M scores 29.6% this way (see memtrix/compare.py to measure it
on this machine). Works for any model with the forward(idx) -> logits interface.

    python -m oazy.hellaswag --ckpt /workspace/oazy_runs/memtrix/ckpt.pt
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import default_dir
from . import tok


def load(path: Path, limit: int | None = None):
    """All questions, or `limit` of them spread evenly over the whole file.

    The file is ordered by source (video captions first, then WikiHow), and the first questions are
    much easier than the rest, so a quick check must sample across the file, not take the start."""
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if limit and limit < len(rows):
        step = len(rows) / limit
        rows = [rows[int(i * step)] for i in range(limit)]
    return rows


def render(enc, ex):
    ctx = enc.encode_ordinary(ex["ctx"])
    ends = [enc.encode_ordinary(" " + e) for e in ex["endings"]]
    return ctx, ends, int(ex["label"])


@torch.no_grad()
def evaluate(model, path: Path, device, limit: int | None = None, batch_examples: int = 16):
    """Returns {"acc_norm", "acc", "n"}. limit = only the first N examples (quick check during training)."""
    enc = tok.for_model(model)
    rows = load(path, limit)
    was_training = model.training
    model.eval()
    correct_norm = correct = correct_char = 0
    for b0 in range(0, len(rows), batch_examples):
        exs = [render(enc, ex) for ex in rows[b0:b0 + batch_examples]]
        seqs, spans = [], []
        for ctx, ends, _ in exs:
            for e in ends:
                seqs.append(ctx + e)
                spans.append((len(ctx), len(ctx) + len(e)))
        L = max(len(s) for s in seqs)
        x = torch.zeros(len(seqs), L, dtype=torch.long)
        mask = torch.zeros(len(seqs), L - 1)
        for i, (s, (a, b)) in enumerate(zip(seqs, spans)):
            x[i, :len(s)] = torch.tensor(s)
            mask[i, a - 1:b - 1] = 1.0                    # positions whose next token is an ending token
        x, mask = x.to(device), mask.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            if hasattr(model, "seq_logits"):     # Fern writes in passages: start one where the ending starts
                st = torch.tensor([a - 1 for a, _ in spans], device=device)
                en = torch.tensor([b - 1 for _, b in spans], device=device)
                logits = model.seq_logits(x, st, en)
                logits = torch.cat([logits, logits[:, -1:]], 1)
            else:
                logits = model._body(x)          # uncompiled: these shapes change every batch
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1),
                               reduction="none").view(len(seqs), L - 1)
        tot = (loss * mask).sum(1).view(-1, 4)
        avg = tot / mask.sum(1).view(-1, 4)
        chars = torch.tensor([[len(" " + e) for e in ex["endings"]] for ex in rows[b0:b0 + batch_examples]],
                             device=device, dtype=torch.float32)
        labels = torch.tensor([lab for _, _, lab in exs], device=device)
        correct_char += int(((tot / chars).argmin(1) == labels).sum())
        correct_norm += int((avg.argmin(1) == labels).sum())
        correct += int((tot.argmin(1) == labels).sum())
    if was_training:
        model.train()
    n = len(rows)
    return {"acc_norm": correct_norm / max(n, 1), "acc": correct / max(n, 1), "acc_char": correct_char / max(n, 1), "n": n}


def main():
    import argparse
    from .train import load_model
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    a.data = a.data or default_dir("data")
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _ = load_model(Path(a.ckpt), device)
    r = evaluate(model, Path(a.data) / "hellaswag_val.jsonl", device, a.limit)
    print(json.dumps(r))


if __name__ == "__main__":
    main()
