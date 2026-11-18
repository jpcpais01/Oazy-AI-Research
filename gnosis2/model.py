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
    canon_impl: str = "pad"    # how the same sum is computed: "pad" (pad once, k sliced adds), "inplace" (no padded copy:
                               # k-1 in-place adds on slices), "conv" (one depthwise convolution). Same numbers, other speed
                               # (gnosis2/speed135.py --only canon measures them)
    hint_rows: int = 0         # phrase hint (0 = off): rows in each hashed table. Every word's input also gets a vector
                               # looked up by the phrase that ends at it (the last 2 words, the last 3), so the first
                               # layers need not rebuild common phrases (DeepSeek's Engram, arXiv 2601.07372)
    hint_dim: int = 32         # numbers per row
    hint_heads: int = 4        # hash functions per phrase length (a phrase clashes with another in one, rarely in all)
    hint_orders: str = "2,3"   # phrase lengths, in words

    def hint_ns(self):
        return [int(s) for s in str(self.hint_orders).split(",") if s.strip()]


class Canon(nn.Module):
    """h + sum_k w_k * h shifted by k words (causal, per channel). w starts at zero: the layer starts as nothing."""

    def __init__(self, dim: int, k: int, impl: str = "pad"):
        super().__init__()
        if impl not in ("pad", "inplace", "conv"):
            raise ValueError(f"canon_impl must be pad, inplace or conv, not {impl!r}")
        self.k, self.impl = k, impl
        self.w = nn.Parameter(torch.zeros(k, dim))

    def forward(self, h):
        """h (B, T, D) -> (B, T, D)."""
        T = h.shape[1]
        w = self.w.to(h.dtype)
        if self.impl == "inplace":                       # the word itself, then each earlier word added where it exists
            out = torch.addcmul(h, h, w[0])
            for j in range(1, min(self.k, T)):
                out[:, j:].addcmul_(h[:, :T - j], w[j])
            return out
        if self.impl == "conv":                          # the same sum as one depthwise convolution over the words
            x = h.transpose(1, 2)
            y = F.conv1d(x, w.flip(0).t().unsqueeze(1), padding=self.k - 1, groups=x.shape[1])[..., :T]
            return h + y.transpose(1, 2)
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
        self.canon_a = Canon(cfg.d_model, cfg.canon_k, cfg.canon_impl) if on and "A" in cfg.canon else None
        self.canon_c = Canon(cfg.d_model, cfg.canon_k, cfg.canon_impl) if on and "C" in cfg.canon else None

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


HINT_MULT = (0x9E3779B1, 0x85EBCA77, 0xC2B2AE3D, 0x27D4EB2F, 0x165667B1, 0xD3A2646C, 0xFD7046C5, 0xB55A4F09)


class Gnosis2(Bee6):
    Block = Block7

    def __init__(self, cfg: Gnosis2Config):
        super().__init__(cfg)
        self._tail = None                      # reading on: the words just before this piece (set around _read)
        self.hint = self.hint_out = None
        if cfg.hint_rows > 0:
            ns = cfg.hint_ns()
            self.hint_n = max(ns)
            self.hint = nn.Embedding(len(ns) * cfg.hint_heads * cfg.hint_rows, cfg.hint_dim)
            self.hint_out = nn.Linear(len(ns) * cfg.hint_heads * cfg.hint_dim, cfg.d_model, bias=False)
            nn.init.normal_(self.hint.weight, std=0.02)
            nn.init.zeros_(self.hint_out.weight)       # starts as nothing: the model starts exactly as without it
            g = torch.Generator().manual_seed(1234)
            mult = torch.randint(1, 2 ** 31 - 1, (len(ns), cfg.hint_heads, self.hint_n), generator=g) * 2 + 1
            self.register_buffer("hint_mult", mult, persistent=False)

    def hint_params(self):
        return 0 if self.hint is None else self.hint.weight.numel() + self.hint_out.weight.numel()

    def _hint(self, idx, tail=None):
        """(B, T) words -> (B, T, D): for each word, the rows its phrases (the 2 and the 3 words ending at it) hash
        to, squeezed to the stream's width. Looks back only. A phrase that would start before the text adds nothing."""
        cfg, B, T, n = self.cfg, idx.shape[0], idx.shape[1], self.hint_n
        if tail is None:
            tail = idx.new_full((B, n - 1), -1)
        elif tail.shape[1] < n - 1:
            tail = torch.cat([idx.new_full((B, n - 1 - tail.shape[1]), -1), tail], dim=1)
        full = torch.cat([tail, idx], dim=1)                                        # (B, n - 1 + T)
        back = torch.stack([full[:, n - 1 - j:n - 1 - j + T] for j in range(n)], dim=-1)     # [..., j] = the word j back
        parts = []
        for o, order in enumerate(cfg.hint_ns()):
            known = (back[..., :order] >= 0).all(-1)                                # the whole phrase is in the text
            key = (back[..., None, :order] * self.hint_mult[o, :, :order]).sum(-1)   # (B, T, heads), wraps in 64 bits
            key = (key ^ (key >> 17)) * 0x9E3779B1
            row = torch.remainder(key ^ (key >> 29), cfg.hint_rows)
            row = row + (o * cfg.hint_heads + torch.arange(cfg.hint_heads, device=idx.device)) * cfg.hint_rows
            v = self.hint(row).flatten(-2)                                          # (B, T, heads * dim)
            parts.append(v * known[..., None].to(v.dtype))
        return self.hint_out(torch.cat(parts, dim=-1))

    def _embed(self, idx, prev=None):
        e, raw_last = super()._embed(idx, prev)
        if self.hint is not None:
            e = e + self._hint(idx, self._tail).to(e.dtype)
        return e, raw_last

    def _read(self, idx, states=None, looks=False):
        if self.hint is None:
            return super()._read(idx, states, looks)
        tail = states.get("tail") if states else None
        self._tail = tail
        try:
            logits, st, infos = super()._read(idx, states, looks)
        finally:
            self._tail = None
        full = idx if tail is None else torch.cat([tail, idx], dim=1)
        st["tail"] = full[:, -(self.hint_n - 1):]
        return logits, st, infos

    def layout(self):
        n = sum((b.canon_a is not None) + (b.canon_c is not None) for b in self.blocks)
        hint = ""
        if self.hint is not None:
            c = self.cfg
            hint = (f", phrase hint ({c.hint_orders} words, {c.hint_heads} x {c.hint_rows:,} rows of {c.hint_dim}: "
                    f"{self.hint_params() / 1e6:.1f}M)")
        return super().layout() + f", {n} Canon layers ({self.cfg.canon_k} words wide)" + hint

    def canon_params(self):
        return sum(p.numel() for n, p in self.named_parameters() if ".canon_" in n)
