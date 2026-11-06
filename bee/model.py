"""Bee0.2B: our main model. A modern hybrid, about 200M parameters, built to be the best model we can
train on 5B words. It is a NEXT-WORD model: it reads text and guesses the very next word, like GPT-2.

The idea in one picture. Bee is a stack of 24 layers, in groups of four:

    memory, memory, memory, look-back      x 6

  memory layer     Memtrix's learn-while-reading memory (memtrix/model.py): a few small grids that are
  ("Memtrix")      corrected by their own surprise, word by word. Cheap, fixed size, carries the gist of
                   everything read so far and the order of the words. (Research name: Gated DeltaNet.)
  look-back layer  ordinary attention: every word may look back at any earlier word, exactly. This is
  ("attention")    what memories are bad at (copying a name or a number from 800 words ago), so one
                   layer in four keeps it. A small gate on its output (sigmoid) keeps training stable.

Every layer also has a per-word "thinking" part (the MLP, SwiGLU). Big labs build their newest models
exactly this way (Qwen3-Next: 3 Gated DeltaNet layers to 1 gated attention layer; Kimi Linear: the
same 3:1 pattern). It is mostly assembly of proven parts, which is the point: low risk, high quality.

The modern pieces GPT-2 (2019) lacks, all here:
  RMSNorm before each part, no bias terms, SwiGLU, QK-norm in attention, grouped key/value heads
  (12 question heads share 4 key/value heads), tied input/output word tables, and a soft cap on the
  final scores (Gemma 2, nanoGPT speedrun) so no score can explode.
  No position table at all: the memory layers already know word order (they read in order, and have a
  4-word convolution), so the attention layers use no position signal ("NoPE", as in Kimi Linear).
  That also lets Bee read longer texts than it was trained on. (--set rope=1 turns RoPE on.)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from memtrix.model import RMSNorm, Memory, MLP
from oazy.sample import sample_logits


@dataclass
class BeeConfig:
    vocab_size: int = 50304      # GPT-2's words (so GPT-2 large can be its mentor), padded to a multiple of 64
    max_len: int = 2048          # longest training window (Bee itself has no hard limit)
    ctx: str = "1024:0,2048:0.5" # training window by fraction of training: 1,024 words while the mentor
                                 # (who can only read 1,024) teaches, then 2,048
    d_model: int = 768
    n_layer: int = 24
    pattern: str = "MMMA"        # M = Memtrix memory layer, A = attention; repeated to fill n_layer
    # memory layers (names shared with Memtrix's Memory class)
    n_head: int = 6              # memories per memory layer
    head_k: int = 128            # each memory is head_k x head_v numbers
    head_v: int = 128
    conv: int = 4
    # attention layers
    attn_heads: int = 12
    attn_kv_heads: int = 4
    attn_head_dim: int = 64
    rope: int = 0                # 0 = no positions in attention (NoPE), 1 = RoPE
    rope_base: float = 10000.0
    # per-word thinking
    mlp_hidden: int = 1792       # SwiGLU width (1792 makes the whole model 204M)
    softcap: float = 30.0        # final scores are kept within +-30 (0 = off)
    flex_attn: int = 0           # 1 = with doc_mask, compiled training uses flex attention: the blocks of the score grid that
                                 # cross documents are skipped and the fast kernel is kept (Frog); 0 = a full yes/no grid
    fused_gates: int = 0         # 1 = the memory kernel scales q/k and squashes trust and fade itself (memtrix/memory.py)
    memory_op: int = 0           # 1 = the memory kernel as a compiled operator: no compiler stop at each memory layer (Frog)
    doc_mask: int = 0            # 1 = no bleed between documents in a window: attention looks back only inside
                                 # the current document and the memories start blank at each document (Bee3)
    tokenizer: str = "gpt2"      # which pieces it reads (oazy/tok.py); decides the <|endoftext|> id

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def kinds(self):
        return [self.pattern[i % len(self.pattern)] for i in range(self.n_layer)]


def ctx_plan(cfg: BeeConfig):
    """"1024:0,2048:0.5" -> [(0.0, 1024), (0.5, 2048)]."""
    plan = sorted((float(w), int(t)) for t, w in (s.split(":") for s in cfg.ctx.split(",")))
    if plan[0][0] != 0 or any(t > cfg.max_len for _, t in plan):
        raise ValueError(f"bad ctx plan {cfg.ctx!r}: must start at 0 and stay <= max_len {cfg.max_len}")
    return plan


def ctx_at(cfg: BeeConfig, progress: float) -> int:
    return [t for w, t in ctx_plan(cfg) if progress >= w][-1]


def rope(x, pos, base):
    """Rotary positions. x: (B, T, H, d), pos: (T,) word positions."""
    d = x.shape[-1]
    inv = 1.0 / (base ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    ang = pos.float()[:, None] * inv[None]                     # (T, d/2)
    cos, sin = ang.cos()[None, :, None, :], ang.sin()[None, :, None, :]
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)


class Attention(nn.Module):
    """Look back at every earlier word, exactly. Grouped key/value heads, QK-norm, output gate."""

    def __init__(self, cfg: BeeConfig):
        super().__init__()
        D, H, KV, hd = cfg.d_model, cfg.attn_heads, cfg.attn_kv_heads, cfg.attn_head_dim
        assert H % KV == 0
        self.H, self.KV, self.hd, self.cfg = H, KV, hd, cfg
        self.q = nn.Linear(D, H * hd, bias=False)
        self.k = nn.Linear(D, KV * hd, bias=False)
        self.v = nn.Linear(D, KV * hd, bias=False)
        self.gate = nn.Linear(D, H * hd, bias=False)          # Qwen3-Next's gated attention
        self.qn, self.kn = RMSNorm(hd), RMSNorm(hd)
        self.out = nn.Linear(H * hd, D, bias=False)

    def project(self, x, start: int):
        B, T, _ = x.shape
        q = self.qn(self.q(x).view(B, T, self.H, self.hd))
        k = self.kn(self.k(x).view(B, T, self.KV, self.hd))
        v = self.v(x).view(B, T, self.KV, self.hd)
        if self.cfg.rope:
            pos = torch.arange(start, start + T, device=x.device)
            q, k = rope(q, pos, self.cfg.rope_base), rope(k, pos, self.cfg.rope_base)
        return q, k, v

    def finish(self, x, y):
        B, T, _ = x.shape
        y = y.transpose(1, 2).reshape(B, T, -1)
        return self.out(y * torch.sigmoid(self.gate(x)))

    _gqa = None      # does this PyTorch let the kernel share key/value heads itself (no copies)? found out once

    def forward(self, x, state=None, mask=None):
        """state: None (fresh text) or (keys, values) of the words read before. mask: (B, 1, T, T) bool, which
        earlier words each word may see (its own document only; training with doc_mask). Returns (output, new state)."""
        B, T, _ = x.shape
        P = 0 if state is None else state[0].shape[1]
        q, k, v = self.project(x, P)
        if state is not None:
            k, v = torch.cat([state[0], k], 1), torch.cat([state[1], v], 1)
        qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if P == 0 and mask is None:
            y = self._causal(qh, kh, vh)
        elif P == 0 and not isinstance(mask, torch.Tensor):      # a flex-attention block mask (Bee.attn_mask)
            from torch.nn.attention.flex_attention import flex_attention
            y = flex_attention(qh, kh, vh, block_mask=mask, enable_gqa=self.H != self.KV)
        else:
            rep = self.H // self.KV
            kh, vh = kh.repeat_interleave(rep, dim=1), vh.repeat_interleave(rep, dim=1)
            if mask is None:   # new words see all old ones, and the new ones up to themselves
                mask = torch.ones(T, P + T, dtype=torch.bool, device=x.device).tril(P)
            y = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask)
        return self.finish(x, y), (k, v)

    def _causal(self, qh, kh, vh):
        """Plain causal attention with grouped key/value heads: the fused kernel shares them itself where it can."""
        rep = self.H // self.KV
        if Attention._gqa is not False and rep > 1:
            try:
                y = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, enable_gqa=True)
                Attention._gqa = True
                return y
            except (TypeError, RuntimeError):
                Attention._gqa = False
        if rep > 1:
            kh, vh = kh.repeat_interleave(rep, dim=1), vh.repeat_interleave(rep, dim=1)
        return F.scaled_dot_product_attention(qh, kh, vh, is_causal=True)

    @torch.no_grad()
    def looks(self, x, state):
        """Where the last new word looked: attention weights over every word so far, averaged over heads."""
        P = 0 if state is None else state[0].shape[1]
        q, k, _ = self.project(x, P)
        if state is not None:
            k = torch.cat([state[0], k], 1)
        rep = self.H // self.KV
        qh = q[:, -1:].transpose(1, 2).float()                               # (B, H, 1, d)
        kh = k.transpose(1, 2).repeat_interleave(rep, dim=1).float()         # (B, H, L, d)
        w = torch.softmax(qh @ kh.transpose(-1, -2) / math.sqrt(self.hd), -1)
        return w[:, :, 0].mean(1)                                            # (B, L)


def doc_block_mask(resets):
    """Flex attention's block mask for "earlier words of the same document": blocks of the grid that lie wholly
    across documents are skipped, whole ones inside a document run without a per-element test."""
    from torch.nn.attention.flex_attention import create_block_mask
    B, T = resets.shape
    doc = resets.cumsum(1)

    def same_doc(b, h, q, kv):
        return (doc[b, q] == doc[b, kv]) & (q >= kv)
    return create_block_mask(same_doc, B, None, T, T, device=resets.device, _compile=resets.is_cuda)


_FLEX_OK = None


def flex_check(device, cfg) -> bool:
    """Compiled flex attention against the yes/no-grid attention on a packed window, once per process."""
    global _FLEX_OK
    if _FLEX_OK is not None:
        return _FLEX_OK
    _FLEX_OK = False
    try:
        from torch.nn.attention.flex_attention import flex_attention
        g = torch.Generator(device=device).manual_seed(0)
        B, T, H, KV, hd = 2, 512, cfg.attn_heads, cfg.attn_kv_heads, cfg.attn_head_dim
        q = torch.randn(B, H, T, hd, device=device, generator=g, dtype=torch.bfloat16)
        k, v = (torch.randn(B, KV, T, hd, device=device, generator=g, dtype=torch.bfloat16) for _ in range(2))
        resets = torch.rand(B, T, device=device, generator=g) < 0.01
        resets[:, 0] = False
        doc = resets.cumsum(1)
        grid = (doc[:, None, :, None] == doc[:, None, None, :]) & torch.ones(T, T, dtype=torch.bool, device=device).tril()
        rep = H // KV
        ref = F.scaled_dot_product_attention(q.float(), k.float().repeat_interleave(rep, 1), v.float().repeat_interleave(rep, 1),
                                             attn_mask=grid)
        y = torch.compile(flex_attention)(q, k, v, block_mask=doc_block_mask(resets), enable_gqa=rep > 1)
        err = (y.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
        _FLEX_OK = err < 0.03
        print(f"[bee] flex attention for the document rule {'OK' if _FLEX_OK else 'DISAGREES, not used'} (rel. error {err:.4f})",
              flush=True)
    except Exception as e:
        print(f"[bee] flex attention unavailable ({type(e).__name__}: {str(e)[:160]}); using the yes/no grid", flush=True)
    return _FLEX_OK


class Block(nn.Module):
    def __init__(self, cfg: BeeConfig, kind: str):
        super().__init__()
        self.kind = kind
        self.n1 = RMSNorm(cfg.d_model)
        if kind == "M":
            self.mem = Memory(cfg)
        else:
            self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def mixer(self):
        return self.mem if self.kind == "M" else self.attn

    def forward(self, x, state=None, resets=None, mask=None, segs=None):
        """resets (B, T) bool: new documents start here (memory layers start blank); mask (B, 1, T, T) bool:
        which earlier words each word may see (attention layers). Both None when reading one text.
        segs: the document offsets for the memory kernel's packed form, computed once (doc_info), or None."""
        if self.kind == "M":
            y, state = self.mem(self.n1(x), state, resets, segs)
        else:
            y, state = self.attn(self.n1(x), state, mask)
        x = x + y
        return x + self.mlp(self.n2(x)), state


class Bee(nn.Module):
    embed_no_decay = True        # the word table is not pulled towards zero (SmolLM3)
    block_cls = Block            # the layer type (Coral: Block + its experts)

    def __init__(self, cfg: BeeConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(self.block_cls(cfg, k) for k in cfg.kinds())
        self.norm = RMSNorm(cfg.d_model)
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith(("mem.out.weight", "attn.out.weight", "mlp.down.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))
        for b in self.blocks:
            if b.kind == "M":
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

    def layout(self):
        k = self.cfg.kinds()
        return f"{k.count('M')} memory + {k.count('A')} attention layers"

    def ctx_at(self, progress: float) -> int:
        return ctx_at(self.cfg, progress)

    fused_loss = True            # the loss in slices (oazy/lmloss.py); False = full 32-bit scores (tests, --loss plain)
    loss_chunk = 1024            # tokens per slice

    def prepare_kernels(self, device):
        """Check the optional fast paths once, before compiling (the checks themselves are not compiler-friendly)."""
        from memtrix import memory
        memory.fla_check(device)
        if self.cfg.memory_op:
            memory.op_ok(device)
        if self.cfg.fused_gates:
            memory.fused_gates_ok(device)

    def enable_compile(self):
        dev = self.embed.weight.device
        self.prepare_kernels(dev)
        self._compiled = torch.compile(self._trunk)
        self._flex = bool(self.cfg.flex_attn and self.cfg.doc_mask and dev.type == "cuda" and flex_check(dev, self.cfg))

    _flex = False    # flex attention in use (checked once, when compiling)

    def attn_mask(self, idx, resets, mask):
        """The mask the attention layers get in training: the yes/no grid, or with flex attention its block form
        (the same rule: earlier words of the same document), built once for all layers."""
        if not self._flex or mask is None or self._compiled is None:
            return mask
        return doc_block_mask(resets)

    def head(self, h):
        """Scores against the word table, softly capped. h: the token vectors after the final norm."""
        logits = F.linear(h, self.embed.weight)
        c = self.cfg.softcap
        return c * torch.tanh(logits / c) if c else logits

    def doc_info(self, idx):
        """With doc_mask: where new documents start (B, T), which earlier words each word may see (B, 1, T, T),
        and the document offsets for the memory kernel's packed form (or None). A document starts right after
        an <|endoftext|>. Without doc_mask: (None, None, None), a window is one text."""
        if not self.cfg.doc_mask:
            return None, None, None
        from oazy.tok import eot_id
        from memtrix import memory
        B, T = idx.shape
        resets = torch.zeros(B, T, dtype=torch.bool, device=idx.device)
        resets[:, 1:] = idx[:, :-1] == eot_id(self.cfg.tokenizer)
        doc = resets.cumsum(1)
        mask = (doc[:, None, :, None] == doc[:, None, None, :]) & torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
        # the documents' offsets once, not once per layer (also on the CPU when the memory operator runs: self-tests)
        segs = memory.segments(resets, hints=True) if (memory.packed_docs(idx.device) or (memory._OP_OK and not idx.is_cuda)) else None
        return resets, mask, segs

    def _trunk(self, idx, resets=None, mask=None, segs=None):
        """Everything but the word table: (B, T, D) token vectors after the final norm."""
        x = self.embed(idx)
        for b in self.blocks:
            x, _ = b(x, None, resets, mask, segs)
        return self.norm(x)

    def _body(self, idx):
        """The full scores (B, T, V), as the probes and HellaSwag want them."""
        return self.head(self._trunk(idx, *self.doc_info(idx)))

    def losses(self, idx, targets, mentor=None, kd_weight=0.0):
        """One lesson: the next-word loss, plus the mentor's (KL to its guesses) weighted kd_weight.
        Returns (loss to train on, {"ce": next-word loss, "kd": the mentor term}), the stats detached."""
        resets, mask, segs = self.doc_info(idx)
        h = (self._compiled or self._trunk)(idx, resets, self.attn_mask(idx, resets, mask), segs)
        th = tW = None
        if mentor is not None and kd_weight > 0:
            th, tW = mentor.hidden(idx, mask)
        if self.fused_loss:
            from oazy.lmloss import head_loss
            loss, ce, kd = head_loss(h, self.embed.weight, targets, self.cfg.softcap, th, tW, kd_weight, self.loss_chunk)
        else:
            from oazy.lmloss import plain_loss
            loss, ce, kd = plain_loss(h, self.embed.weight, targets, self.cfg.softcap, th, tW, kd_weight)
        stats = {"ce": ce}
        if th is not None:
            stats["kd"] = kd
        return loss, stats

    def forward(self, idx, targets=None):
        """Without targets: (scores, None, {}). With targets: (None, loss, stats); the scores are never formed whole."""
        if targets is None:
            return self._body(idx), None, {}
        loss, stats = self.losses(idx, targets)
        return None, loss, stats

    # ---------------------------------------------------------------- reading as a stream (chat)
    def read(self, idx, states=None):
        """Read tokens (B, T), carrying memories and attention keys along. Returns (logits, states)."""
        x = self.embed(idx)
        states = states or [None] * len(self.blocks)
        new = []
        for b, s in zip(self.blocks, states):
            x, s = b(x, s)
            new.append(s)
        return self.head(self.norm(x)), new

    @torch.no_grad()
    def inspect(self, idx, states=None):
        """Read one new word with the lid off. Per memory layer: how surprised its memories were;
        per attention layer: where the word looked back to. Returns (logits, states, infos)."""
        x = self.embed(idx)
        states = states or [None] * len(self.blocks)
        new, infos = [], []
        for b, s in zip(self.blocks, states):
            h = b.n1(x)
            if b.kind == "M":
                y, s, info = b.mem.inspect(h, s)
                info = {"kind": "M", "surprise": float(info["surprise"][0, -1].mean()),
                        "trust": float(info["trust"][0, -1].mean())}
            else:
                info = {"kind": "A", "looks": b.attn.looks(h, s)[0]}
                y, s = b.attn(h, s)
            x = x + y
            x = x + b.mlp(b.n2(x))
            new.append(s)
            infos.append(info)
        return self.head(self.norm(x)), new, infos

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
