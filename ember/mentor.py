"""The mentor: a bigger, already-trained model whose guesses Ember learns from (distillation).

Being told "the next word was *cat*" teaches one thing. Seeing a good reader's whole guess
("70% cat, 20% dog, 5% kitten, ...") teaches what was likely, what was close and what was absurd,
for every word of the vocabulary at once. That richer signal is why distilled small models (Gemma 3
270M among them) learn far more per word read.

The mentor here is OpenAI's GPT-2 medium (355M; or gpt2-large / gpt2-xl): same tokenizer as our
data, so no conversion. It only reads (no learning, no gradients), in bf16, and costs roughly 25-50%
of an Ember training step while it is active. Its weight fades linearly from `weight` at the start
to 0 at `until` (fraction of training), after which it is not run at all.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from oazy.gpt import load_openai_gpt2


class Mentor:
    def __init__(self, name: str, device, weight: float = 0.6, until: float = 0.5, compile: bool = True):
        self.name, self.w0, self.until = name, weight, until
        if name == "random":                      # self-tests only: a tiny untrained stand-in (no download)
            from oazy.gpt import GPT, GPTConfig
            self.net = GPT(GPTConfig(vocab_size=50257, n_layer=2, n_head=2, d_model=64)).to(device).eval()
        else:
            self.net = load_openai_gpt2(device, name)
        if device.type == "cuda":
            self.net = self.net.to(torch.bfloat16)
            if compile:
                self.net.enable_compile()
        for p in self.net.parameters():
            p.requires_grad_(False)

    def weight_at(self, progress: float) -> float:
        if self.until <= 0:
            return 0.0
        return max(0.0, self.w0 * (1 - progress / self.until))

    @torch.no_grad()
    def guess(self, x):
        L = self.net.cfg.max_len                  # GPT-2 reads at most 1,024 words: longer windows go in pieces
        f = self.net._compiled if self.net._compiled is not None else self.net._body
        if x.shape[1] <= L:
            return f(x)
        return torch.cat([f(x[:, i:i + L]) for i in range(0, x.shape[1], L)], dim=1)


def mentor_loss(student_logits, mentor_logits, chunk: int = 4):
    """KL(mentor || student) per word, averaged: how far the student's guess is from the mentor's."""
    V = mentor_logits.shape[-1]                              # 50257 (the student pads to 50304)
    tot, n = 0.0, student_logits.shape[0]
    for i in range(0, n, chunk):                             # in slices, to keep memory bounded
        s = F.log_softmax(student_logits[i:i + chunk, :, :V].float(), dim=-1)
        t = F.log_softmax(mentor_logits[i:i + chunk].float(), dim=-1)
        tot = tot + F.kl_div(s, t, log_target=True, reduction="sum")
    return tot / (n * student_logits.shape[1])
