"""Picking the next word from a model's probabilities (shared by every model's chat and samples)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_logits(logits, temperature=0.8, top_k=40, vocab=50257):
    logits = logits[:vocab].float()
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k:
        v, _ = torch.topk(logits, min(top_k, logits.numel()))
        logits[logits < v[-1]] = -float("inf")
    return int(torch.multinomial(F.softmax(logits, -1), 1))
