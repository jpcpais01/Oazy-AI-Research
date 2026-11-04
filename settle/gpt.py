"""A small nanoGPT-style transformer, used as the baseline Settle is compared against.

Pre-norm blocks, causal self-attention (PyTorch's fused attention), GELU MLP,
learned positions, and the token embedding tied to the output layer.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    max_len: int = 256
    n_layer: int = 2
    n_head: int = 6
    d_model: int = 384
    dropout: float = 0.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        d = cfg.d_model
        self.n_head = cfg.n_head
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4 * d)
        self.out = nn.Linear(4 * d, d)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, k, v = (t.view(B, T, self.n_head, D // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, D))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("out.weight"):
                nn.init.normal_(p, std=0.02 / (2 * cfg.n_layer) ** 0.5)
        self._compiled = None

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
        return n

    def enable_compile(self):
        self._compiled = torch.compile(self._body)

    def _body(self, idx):
        T = idx.shape[1]
        x = self.embed(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        return F.linear(self.ln_f(x), self.embed.weight)

    def forward(self, idx, targets=None):
        logits = (self._compiled or self._body)(idx)
        loss = None
        stats = {}
        if targets is not None:
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
            stats["ce"] = loss
        return logits, loss, stats

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=100, temperature=0.8, top_k=40, eos=None):
        from .model import sample_logits
        ids = list(prompt_ids)
        out = []
        dev = self.embed.weight.device
        for _ in range(max_new_tokens):
            x = torch.tensor([ids[-self.cfg.max_len:]], device=dev)
            logits = self._body(x)[0, -1]
            nxt = sample_logits(logits, temperature, top_k)
            if eos is not None and nxt == eos:
                break
            ids.append(nxt)
            out.append(nxt)
        return out, None


def matched_gpt_config(target_params: int, d_model: int, max_len: int, vocab_size: int = 50257) -> GPTConfig:
    """Pick the number of layers whose total size is closest to target_params (ties go to the bigger model)."""
    best = None
    for L in range(1, 25):
        cfg = GPTConfig(vocab_size=vocab_size, max_len=max_len, n_layer=L,
                        n_head=max(1, d_model // 64), d_model=d_model)
        n = vocab_size * d_model + max_len * d_model + L * (12 * d_model * d_model + 13 * d_model) + 2 * d_model
        diff = abs(n - target_params)
        if best is None or diff <= best[0]:
            best = (diff, cfg)
        if n > target_params:
            break
    return best[1]
