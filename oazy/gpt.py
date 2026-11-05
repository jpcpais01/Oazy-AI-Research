"""The baseline: GPT-2 small (124M), written nanoGPT-style.

Same architecture as OpenAI's GPT-2: learned positions, pre-norm LayerNorm blocks, causal
self-attention (PyTorch's fused attention), tanh-GELU MLP, biases, tied input/output words.
It is trained here on exactly the same data, with the same schedule, as Engram.

load_openai_gpt2() also loads OpenAI's own released GPT-2 124M weights into this class, so the
published model can be measured on our validation text and on HellaSwag with the same code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    max_len: int = 1024
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768

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
        return x + self.out(F.gelu(self.fc(self.ln2(x)), approximate="tanh"))


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
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))
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
        return n - self.embed.weight.numel() if non_embedding else n

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
        loss, stats = None, {}
        if targets is not None:
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
            stats["ce"] = loss
        return logits, loss, stats

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=100, temperature=0.8, top_k=40, eos=None):
        from .sample import sample_logits
        ids = list(prompt_ids)
        out = []
        dev = self.embed.weight.device
        for _ in range(max_new_tokens):
            x = torch.tensor([ids[-self.cfg.max_len:]], device=dev)
            nxt = sample_logits(self._body(x)[0, -1], temperature, top_k)
            if eos is not None and nxt == eos:
                break
            ids.append(nxt)
            out.append(nxt)
        return out, None


def load_openai_gpt2(device="cpu") -> GPT:
    """OpenAI's released GPT-2 124M, loaded from Hugging Face ("gpt2") into this class."""
    from transformers import GPT2LMHeadModel
    hf = GPT2LMHeadModel.from_pretrained("gpt2").state_dict()
    m = GPT(GPTConfig(vocab_size=50257))
    sd = {"embed.weight": hf["transformer.wte.weight"], "pos.weight": hf["transformer.wpe.weight"],
          "ln_f.weight": hf["transformer.ln_f.weight"], "ln_f.bias": hf["transformer.ln_f.bias"]}
    names = {"ln_1": "ln1", "ln_2": "ln2", "attn.c_attn": "qkv", "attn.c_proj": "proj", "mlp.c_fc": "fc", "mlp.c_proj": "out"}
    for i in range(12):
        for a, b in names.items():
            for kind in ("weight", "bias"):
                w = hf[f"transformer.h.{i}.{a}.{kind}"]
                if kind == "weight" and not a.startswith("ln"):
                    w = w.t()                     # OpenAI stored these as Conv1D (in x out)
                sd[f"blocks.{i}.{b}.{kind}"] = w
    m.load_state_dict(sd)
    return m.to(device).eval()
