"""Gnosis2: Gnosis with Canon layers. Still a NEXT-WORD model: it reads text and guesses the very next token, like every
Bee and Gnosis, and is scored the same way.

The body is Gnosis (Bee6 at 22 layers of 1,024, Mixture of Depths in every other layer, the previous-word keys in every
fourth, value tables, U-net skips, x0 mixing, smear, back-out; ~307.5M) with one change, Canon layers (Allen-Zhu,
Physics of Language Models Part 4.1, NeurIPS 2025, arXiv 2512.17351):

    a Canon layer    h_t  ->  h_t + w0 * h_t + w1 * h_(t-1) + w2 * h_(t-2) + w3 * h_(t-3)
                     (w0..w3 one number per channel: a tiny causal convolution, 4 words wide)

placed on the input of attention (Canon-A) and the input of the MLP (Canon-C) in every layer that every word goes
through (the 12 that are not routed; a routed layer takes a few words from all over the window, so "the word before"
is not there). The weights start at zero, so Gnosis2 starts exactly as Gnosis.

Why. Attention is good at finding things far away and poor at the plain job of mixing a word with its neighbours,
which it then spends whole layers on. Canon layers do that job for almost nothing (98k weights, well under 1% of the
compute), and the layers are freed for reasoning. In Allen-Zhu's controlled tests (synthetic tasks where depth of
reasoning can be measured) they raised the depth of multi-step reasoning 2-4x and made weak bodies competitive. It is
the one change to the body with real evidence on reasoning, not only on loss; everything else that makes Gnosis2 new
is in what it reads (Opal2, oazy/opal2.py) and how it is finished (gnosis2/rl.py).

Kept from Gnosis exactly: the size, the recipe (Bee6's: windows 512 -> 1,024 -> 2,048, the batch ramp, WSD with the
square-root cool-down over the last 30%, NorMuon + Adam, the early 2nd/3rd-word lesson, FP8 layer products), SmolLM2's
pieces. Canon weights learn in Adam's "gates" group (oazy's bee5/optim.py: everything that is not a grid).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from bee5.model import norm
from bee6.model import Bee6, Bee6Config, Block6

GNOSIS = dict(n_layer=22, mod_layers="2,4,6,8,10,12,14,16,18,20", key_offset="3,7,11,15,19")


@dataclass
class Gnosis2Config(Bee6Config):
    n_layer: int = 22
    mod_layers: str = "2,4,6,8,10,12,14,16,18,20"
    key_offset: str = "3,7,11,15,19"
    canon: str = "AC"          # where the Canon layers go in the non-routed layers: A = attention's input, C = the MLP's
    canon_k: int = 4           # words each one reads (the word itself and the 3 before)


class Canon(nn.Module):
    """h + sum_k w_k * h shifted by k words (causal, per channel). w starts at zero: the layer starts as nothing."""

    def __init__(self, dim: int, k: int):
        super().__init__()
        self.k = k
        self.w = nn.Parameter(torch.zeros(k, dim))

    def forward(self, h):
        """h (B, T, D) -> (B, T, D)."""
        T = h.shape[1]
        w = self.w.to(h.dtype)
        hp = F.pad(h, (0, 0, self.k - 1, 0))
        out = h
        for j in range(self.k):
            out = out + w[j] * hp[:, self.k - 1 - j:self.k - 1 - j + T]
        return out

    def step(self, h, hist):
        """Reading on: h (B, T, D) the new words, hist (B, k-1, D) the words before (or None) -> (out, new hist)."""
        B, T, D = h.shape
        if hist is None:
            hist = h.new_zeros(B, self.k - 1, D)
        full = torch.cat([hist, h], dim=1)
        w = self.w.to(h.dtype)
        out = h
        for j in range(self.k):
            out = out + w[j] * full[:, self.k - 1 - j:self.k - 1 - j + T]
        return out, full[:, -(self.k - 1):]


class Block7(Block6):
    """A Bee6 layer with Canon layers on its attention's and its MLP's input (non-routed layers only)."""

    def __init__(self, cfg: Gnosis2Config, layer: int):
        super().__init__(cfg, layer)
        on = not self.routed
        self.canon_a = Canon(cfg.d_model, cfg.canon_k) if on and "A" in cfg.canon else None
        self.canon_c = Canon(cfg.d_model, cfg.canon_k) if on and "C" in cfg.canon else None

    def _a(self, h):
        return self.canon_a(h) if self.canon_a is not None else h

    def _c(self, h):
        return self.canon_c(h) if self.canon_c is not None else h

    def dense(self, x, ve, cos, sin):
        y = x + self.attn(self._a(norm(x)), ve, cos, sin)
        f, ex = self.ffn(self._c(norm(y)))
        return y + f, ex

    def step(self, x, ve, cos, sin, state, looks=False):
        if self.routed:
            return super().step(x, ve, cos, sin, state, looks)
        st = state or {"attn": None, "a": None, "c": None}
        h = norm(x)
        if self.canon_a is not None:
            h, ha = self.canon_a.step(h, st["a"])
        else:
            ha = None
        y, sa, info = self.attn.step(h, ve, cos, sin, st["attn"], looks)
        x = x + y
        h = norm(x)
        if self.canon_c is not None:
            h, hc = self.canon_c.step(h, st["c"])
        else:
            hc = None
        return x + self.mlp(h), {"attn": sa, "a": ha, "c": hc}, info


class Gnosis2(Bee6):
    Block = Block7

    def layout(self):
        n = sum((b.canon_a is not None) + (b.canon_c is not None) for b in self.blocks)
        return super().layout() + f", {n} Canon layers ({self.cfg.canon_k} words wide)"

    def canon_params(self):
        return sum(p.numel() for n, p in self.named_parameters() if ".canon_" in n)
