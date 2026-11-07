"""Picking the next word from a model's probabilities (shared by every model's chat and samples)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

REP = 1.1          # the repetition guard of the samples the runs write ("What they write"); the phone app's default too
REP_WINDOW = 64    # it looks at the last this many tokens (the prompt included)


def sample_logits(logits, temperature=0.8, top_k=40, vocab=50257, recent=None, rep=1.0):
    """recent / rep: the repetition guard. Every token among `recent` (the last tokens of the text) is made a little
    less likely: its score is divided by rep when positive and multiplied by it when negative (CTRL, Keskar et al.
    2019; the phone app does the same). rep = 1 is off."""
    logits = logits[:vocab].float()
    if rep != 1.0 and recent:
        logits = logits.clone()
        idx = torch.tensor(sorted({t for t in recent if 0 <= t < logits.numel()}), device=logits.device, dtype=torch.long)
        if idx.numel():
            v = logits[idx]
            logits[idx] = torch.where(v > 0, v / rep, v * rep)
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k:
        v, _ = torch.topk(logits, min(top_k, logits.numel()))
        logits[logits < v[-1]] = -float("inf")
    return int(torch.multinomial(F.softmax(logits, -1), 1))
