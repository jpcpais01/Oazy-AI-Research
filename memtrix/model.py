"""Memtrix: a language model that learns while it reads.

The whole model, following one word from arriving to the next word coming out:

  1. the word becomes a vector (its meaning), mixed with a hint of the last few words (a tiny
     causal convolution, 4 words wide);
  2. from that vector the block makes a key ("this is the situation"), a value ("this is what
     goes with it"), a question ("what comes next?"), a trust level and a fade level;
  3. the block's memory (a 64 x 64 grid per head, blank at the start of every text) guesses the
     value from the key; the miss is the surprise; the memory is fixed by trust x surprise, after
     fading a little (memtrix/memory.py: the one rule);
  4. the question reads an answer out of the updated memory, and a small per-word layer (the MLP,
     "how to think about what was recalled") turns it into the input of the next block;
  5. after the last block, the vector is compared with every word's meaning to give the next
     word's probabilities.

Stacked memories (12 blocks) give depth: same rule everywhere. Nothing else carries information
from one word to the next (no attention, no position table): if the model remembers anything from
more than 3 words ago, it went through a memory. The memory never grows: a 100-word text and a
100,000-word text cost the same per word.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from oazy.sample import sample_logits  # noqa: F401  (re-exported)

from . import memory


@dataclass
class MemtrixConfig:
    vocab_size: int = 50304      # GPT-2's 50257 words, padded to a multiple of 64 for speed
    max_len: int = 1024          # training context (the model itself has no length limit)
    d_model: int = 768
    n_layer: int = 12
    n_head: int = 12             # memories per block
    head_k: int = 64             # each memory is head_k x head_v numbers
    head_v: int = 64
    mlp_hidden: int = 1536       # 0 = no per-word layer (memory-only ablation)
    conv: int = 4                # "hint of the last few words" (1 = none)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class Memory(nn.Module):
    """One block's memories: read a word, feel the surprise, learn from it, recall."""

    def __init__(self, cfg: MemtrixConfig):
        super().__init__()
        D, H, K, V = cfg.d_model, cfg.n_head, cfg.head_k, cfg.head_v
        self.H, self.K, self.V = H, K, V
        ch = H * (2 * K + V)
        self.qkv = nn.Linear(D, ch, bias=False)
        self.conv_w = nn.Parameter(torch.zeros(ch, 1, cfg.conv))
        self.trust = nn.Linear(D, H)                      # how much to trust this word (0..1)
        self.fade = nn.Linear(D, H)                       # how much to fade the memory before it
        self.A_log = nn.Parameter(torch.empty(H))         # each memory's own speed of forgetting
        self.dt_bias = nn.Parameter(torch.empty(H))
        self.gate = nn.Linear(D, H * V, bias=False)       # volume knob on what is recalled
        self.onorm = RMSNorm(V)
        self.out = nn.Linear(H * V, D, bias=False)
        self.reset_memory_params()

    def reset_memory_params(self):
        with torch.no_grad():
            # a spread of forgetting speeds, from "a few words" to "thousands of words" (as in Mamba-2 / Gated DeltaNet)
            self.A_log.copy_(torch.empty_like(self.A_log).uniform_(1, 16).log())
            dt = torch.exp(torch.empty_like(self.dt_bias).uniform_(math.log(1e-3), math.log(1e-1))).clamp_min(1e-4)
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))      # inverse softplus
            # the convolution starts as "just this word" plus a little of the previous ones
            self.conv_w.zero_()
            self.conv_w[:, 0, -1] = 1.0
            self.conv_w[:, 0, :-1].normal_(0, 0.02)

    def project(self, x, conv_state=None, resets=None):
        """Word vectors -> question, key, value, trust, log-fade. conv_state: the previous few words' channels.
        resets (B, T) bool: the "hint of the last few words" does not reach back into the previous document."""
        B, T, _ = x.shape
        H, K, V = self.H, self.K, self.V
        z = self.qkv(x).transpose(1, 2)                                 # (B, ch, T)
        w = self.conv_w.shape[-1]
        if conv_state is None:
            conv_state = z.new_zeros(B, z.shape[1], w - 1)
        zc = torch.cat([conv_state.to(z.dtype), z], dim=2)
        new_conv = zc[:, :, zc.shape[2] - (w - 1):] if w > 1 else zc[:, :, :0]
        cw = self.conv_w.to(z.dtype)
        if resets is None or w == 1:
            z = F.conv1d(zc, cw, groups=z.shape[1])
        else:                                   # the same convolution, written as shifted copies, each one masked
            seg = resets.cumsum(1)                                      # (B, T) which document each word is in
            z = zc[:, :, w - 1:] * cw[:, 0, w - 1][None, :, None]
            for s in range(1, w):
                same = (seg[:, s:] == seg[:, :-s]).to(z.dtype)          # word t and word t-s in the same document
                same = F.pad(same, (s, 0))                              # (B, T): the first s words look into conv_state
                z = z + zc[:, :, w - 1 - s:zc.shape[2] - s] * cw[:, 0, w - 1 - s][None, :, None] * same[:, None, :]
        z = F.silu(z).transpose(1, 2)
        q, k, v = z.split([H * K, H * K, H * V], dim=-1)
        q = F.normalize(q.view(B, T, H, K).float(), dim=-1).to(z.dtype)
        k = F.normalize(k.view(B, T, H, K).float(), dim=-1).to(z.dtype)
        v = v.reshape(B, T, H, V)
        trust = torch.sigmoid(self.trust(x).float())
        log_fade = -self.A_log.float().exp() * F.softplus(self.fade(x).float() + self.dt_bias)
        return q, k, v, trust, log_fade, new_conv

    def finish(self, x, o):
        B, T, _ = x.shape
        o = self.onorm(o) * F.silu(self.gate(x).view(B, T, self.H, self.V))
        return self.out(o.reshape(B, T, -1))

    def forward(self, x, state=None, resets=None):
        """state: None (blank memory) or (conv_state, S). resets: (B, T) bool, True where a new document
        starts inside the window (its memory starts blank). Returns (output, new state)."""
        conv_state, S = state if state is not None else (None, None)
        q, k, v, trust, log_fade, new_conv = self.project(x, conv_state, resets)
        if state is None:
            o, S = memory.delta_rule(q, k, v, log_fade, trust.to(q.dtype), resets=resets)
        elif x.shape[1] == 1:
            o, S = memory.recurrent(q, k, v, log_fade, trust, S)
        else:
            o, S = memory.chunked(q, k, v, log_fade, trust, S)
        return self.finish(x, o), (new_conv, S)

    @torch.no_grad()
    def inspect(self, x, state=None):
        """Like forward, but also returns what happened inside, word by word (for the visuals)."""
        conv_state, S = state if state is not None else (None, None)
        q, k, v, trust, log_fade, new_conv = self.project(x, conv_state)
        o, S, sur = memory.chunked(q, k, v, log_fade, trust, S, return_surprise=True)
        vn = v.float().norm(dim=-1).clamp_min(1e-6)
        info = {
            "trust": trust,                                   # (B, T, H)
            "fade": log_fade.exp(),                           # (B, T, H) fraction kept
            "surprise": sur.norm(dim=-1) / vn,                # (B, T, H) 0 = expected, ~1 = brand new
            "push": trust * sur.norm(dim=-1),                 # (B, T, H) how hard the memory was changed
        }
        return self.finish(x, o), (new_conv, S), info


class MLP(nn.Module):
    def __init__(self, cfg: MemtrixConfig):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, 2 * cfg.mlp_hidden, bias=False)
        self.down = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        a, b = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, cfg: MemtrixConfig):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.mem = Memory(cfg)
        self.mlp = MLP(cfg) if cfg.mlp_hidden else None
        self.n2 = RMSNorm(cfg.d_model) if cfg.mlp_hidden else None

    def forward(self, x, state=None):
        y, state = self.mem(self.n1(x), state)
        x = x + y
        if self.mlp is not None:
            x = x + self.mlp(self.n2(x))
        return x, state


class Memtrix(nn.Module):
    def __init__(self, cfg: MemtrixConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model)
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("mem.out.weight") or n.endswith("mlp.down.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))
        for b in self.blocks:
            b.mem.reset_memory_params()
        self._compiled = None

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        return n - self.embed.weight.numel() if non_embedding else n

    def memory_numbers(self):
        """How many numbers the model holds about the text it is reading (fixed, whatever the length)."""
        c = self.cfg
        return c.n_layer * c.n_head * c.head_k * c.head_v

    def enable_compile(self):
        self._compiled = torch.compile(self._body)

    def _body(self, idx):
        x = self.embed(idx)
        for b in self.blocks:
            x, _ = b(x)
        return F.linear(self.norm(x), self.embed.weight)

    def forward(self, idx, targets=None):
        logits = (self._compiled or self._body)(idx)
        loss, stats = None, {}
        if targets is not None:
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
            stats["ce"] = loss
        return logits, loss, stats

    # ---------------------------------------------------------------- reading as a stream
    def read(self, idx, states=None):
        """Read tokens (B, T) carrying the memories along. Returns (logits, states)."""
        x = self.embed(idx)
        states = states or [None] * len(self.blocks)
        new = []
        for b, s in zip(self.blocks, states):
            x, s = b(x, s)
            new.append(s)
        return F.linear(self.norm(x), self.embed.weight), new

    @torch.no_grad()
    def inspect(self, idx, states=None):
        """Read with the lid off: per block, the trust, fade and surprise of every word, and the memories."""
        x = self.embed(idx)
        states = states or [None] * len(self.blocks)
        new, infos = [], []
        for b, s in zip(self.blocks, states):
            y, s, info = b.mem.inspect(b.n1(x), s)
            x = x + y
            if b.mlp is not None:
                x = x + b.mlp(b.n2(x))
            new.append(s)
            infos.append(info)
        return F.linear(self.norm(x), self.embed.weight), new, infos

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=100, temperature=0.8, top_k=40, eos=None):
        dev = self.embed.weight.device
        logits, states = self.read(torch.tensor([list(prompt_ids)], device=dev))
        out = []
        for _ in range(max_new_tokens):
            nxt = sample_logits(logits[0, -1], temperature, top_k)
            if eos is not None and nxt == eos:
                break
            out.append(nxt)
            logits, states = self.read(torch.tensor([[nxt]], device=dev), states)
        return out, None


# Memtrix was called Engram until Nov 5 2026: the old names keep working for older code.
Engram, EngramConfig = Memtrix, MemtrixConfig
