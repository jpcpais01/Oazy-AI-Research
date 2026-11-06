"""Coral: our first mixture-of-experts model, 1.0B parameters, built to train at about 1M tokens per second on
one B200. Still a NEXT-WORD model: it reads text and guesses the very next token, like GPT-2 and every Bee,
and it is scored the same way.

The idea in one picture. Every layer has a look-back part (attention) and a "thinking" part. In Bee and Frog
the thinking part is one MLP that every word goes through. In Coral it is a colony of 32 small MLPs ("experts")
plus one shared MLP; a tiny router picks the 2 experts that suit each word best:

    word ──► attention ──► router ──► expert 7  ┐
                              └─────► expert 19 ├─► sum (weighted by the router) ──► next layer
                       shared MLP ──────────────┘

All 32 experts together hold most of the 1.0B parameters, but each word only pays for its 2 experts and the
shared one: about 160M parameters of work per word, so it trains about 6x faster than a dense 1B. The price:
each expert sees only part of the text, so per word read Coral learns roughly like a dense ~400M model (a rule
of thumb; the run will tell).

                       Frog                          Coral
  width                1,024                         1,024
  layers               28 (21 memory + 7 attention)  20, all attention (pattern=MMMA: the Memtrix hybrid)
  thinking part        one MLP of 2,560              32 experts of 448, top 2, + a shared MLP of 512
  parameters           406M (all used per word)      1,036M (about 160M used per word)

The parts, and where each comes from:
  sigmoid router + bias balancing   DeepSeek-V3 (2024): each expert has a small "bias" added to its score only when
                                    choosing; after every step the bias of a busy expert goes down and of an idle one
                                    up, so the work stays even without a loss term that pulls against learning. A
                                    tiny balance loss (1e-4, also DeepSeek-V3) is the safety net.
  shared expert                     DeepSeekMoE / Qwen-MoE: common knowledge lives in one MLP everyone uses, so the
                                    routed experts specialise.
  fixed capacity                    GShard / Switch: in training each expert takes at most 1.25x its fair share of a
                                    pass, so every shape is fixed and the experts run as one batched matrix product
                                    (fast, and the compiler sees one graph). A word that overflows keeps the shared
                                    MLP and its other expert. Reading for evaluation and chat drops nothing.
  attention, all layers             Bee's gated attention (QK-norm, grouped key/value heads, output gate), with RoPE
                                    positions: without memory layers nothing else tells attention the word order.
                                    pattern=MMMA brings back Memtrix memory layers 3 in 4 (slower; then no RoPE).
Everything else is Frog's: no bleed between documents (flex attention), 2,048-word windows, SmolLM2's pieces, the
loss in slices, growing depth (8 -> 12 -> 20 layers), FP8 matrix products, Muon (here also on every expert's grids).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from bee.model import Block
from frog.model import Frog, FrogConfig

CORAL_GROW = "8:0,12:0.1,20:0.25"


@dataclass
class CoralConfig(FrogConfig):
    d_model: int = 1024
    n_layer: int = 20
    pattern: str = "A"           # A = attention in every layer; "MMMA" = 3 Memtrix memory layers to 1 attention (hybrid)
    attn_heads: int = 16         # 16 x 64 = the width
    attn_kv_heads: int = 4
    rope: int = -1               # -1 = decide from the pattern: RoPE when there are no memory layers, else NoPE (Bee)
    rope_base: float = 50000.0
    mlp_hidden: int = 512        # the shared expert (every word)
    n_expert: int = 32           # routed experts per layer
    top_k: int = 2               # experts per word
    expert_hidden: int = 448     # each routed expert's SwiGLU width
    capacity: float = 1.25       # training: an expert takes at most this x its fair share of a pass (0 = no limit, slow)
    balance_rate: float = 1e-3   # how fast the balancing biases move per step (DeepSeek-V3's gamma)
    aux_weight: float = 1e-4     # the small balance loss
    grow: str = CORAL_GROW

    def __post_init__(self):
        if self.grow == CORAL_GROW and self.n_layer != 20:     # another depth (a test's small body): no growing unless asked
            self.grow = f"{self.n_layer}:0"
        if self.rope == -1:
            self.rope = 0 if "M" in self.pattern else 1
        if self.top_k > self.n_expert:
            raise ValueError(f"top_k {self.top_k} > n_expert {self.n_expert}")


class Experts(nn.Module):
    """The colony: a router and n_expert SwiGLU MLPs stored as stacked grids (E, out, in)."""

    def __init__(self, cfg: CoralConfig):
        super().__init__()
        D, E, H = cfg.d_model, cfg.n_expert, cfg.expert_hidden
        self.cfg, self.E, self.k, self.H = cfg, E, cfg.top_k, H
        self.router = nn.Linear(D, E, bias=False)
        self.w_up = nn.Parameter(torch.empty(E, 2 * H, D))      # gate and value halves, like MLP.up
        self.w_down = nn.Parameter(torch.empty(E, D, H))
        nn.init.normal_(self.w_up, std=0.02)
        nn.init.normal_(self.w_down, std=0.02 / math.sqrt(2 * cfg.n_layer))
        self.register_buffer("bias", torch.zeros(E))                     # balancing biases: used only to choose
        self.register_buffer("load", torch.zeros(E), persistent=False)   # words sent to each expert since the last step
        self.register_buffer("dropped", torch.zeros(2), persistent=False)  # (overflowed, assigned) since the last step

    def ffn(self, x, e):
        """Expert e on words x (n, D)."""
        a, b = F.linear(x, self.w_up[e]).chunk(2, dim=-1)
        return F.linear(F.silu(a) * b, self.w_down[e])

    def route(self, h):
        """h (N, D) -> chosen experts (N, k), their weights (N, k), the balance loss, words per expert (E,)."""
        logits = F.linear(h.float(), self.router.weight.float())
        scores = torch.sigmoid(logits)                                       # (N, E)
        idx = torch.topk(scores + self.bias, self.k, dim=-1).indices        # the bias steers the choice only
        w = scores.gather(-1, idx)
        w = w / (w.sum(-1, keepdim=True) + 1e-9)
        counts = torch.zeros(self.E, device=h.device, dtype=torch.float32).scatter_add_(
            0, idx.reshape(-1), torch.ones(idx.numel(), device=h.device, dtype=torch.float32))
        frac = counts / idx.numel()                                         # share of the work, no gradient
        prob = (scores / (scores.sum(-1, keepdim=True) + 1e-9)).mean(0)     # the router's average wish, with gradient
        aux = self.E * (frac.detach() * prob).sum()
        return idx, w, aux, counts

    def forward(self, h, with_aux=False):
        B, T, D = h.shape
        x = h.reshape(-1, D)
        idx, w, aux, counts = self.route(x)
        if self.training and self.cfg.capacity > 0:
            y, dropped = self._capacity(x, idx, w)
            with torch.no_grad():
                self.load.add_(counts)
                self.dropped.add_(torch.stack([dropped, torch.tensor(float(idx.numel()), device=x.device)]))
        else:
            y = self._dropless(x, idx, w)
            if self.training:
                with torch.no_grad():
                    self.load.add_(counts)
                    self.dropped[1] += idx.numel()
        y = y.view(B, T, D)
        return (y, aux) if with_aux else y

    def _capacity(self, x, idx, w):
        """Fixed shapes: every expert gets a block of C slots; the experts run as one batched matrix product."""
        N, D = x.shape
        E, k = self.E, self.k
        C = int(math.ceil(self.cfg.capacity * N * k / E / 8) * 8)
        flat = idx.t().reshape(-1)                                  # (k N,): every word's first choice queues first
        tok = torch.arange(N, device=x.device).repeat(k)
        oh = F.one_hot(flat, E).to(torch.int32)
        pos = ((oh.cumsum(0) - 1) * oh).sum(-1)                     # place in its expert's queue
        keep = pos < C
        slot = torch.where(keep, flat * C + pos, torch.full_like(pos, E * C))
        src = torch.full((E * C + 1,), N, device=x.device, dtype=torch.long).scatter_(0, slot, tok)
        xp = torch.cat([x, x.new_zeros(1, D)])
        xin = xp[src[:E * C]].view(E, C, D)                         # empty slots read a row of zeros
        a, b = torch.bmm(xin, self.w_up.to(xin.dtype).mT).chunk(2, dim=-1)
        y = torch.bmm(F.silu(a) * b, self.w_down.to(xin.dtype).mT).reshape(E * C, D)
        yp = torch.cat([y, y.new_zeros(1, D)])
        ya = yp[slot].view(k, N, D)                                 # an overflowed word gets zeros from this expert
        out = (ya * w.t().unsqueeze(-1).to(ya.dtype)).sum(0)
        return out, (~keep).sum().float()

    def _dropless(self, x, idx, w):
        """Every word to its experts, nothing dropped (evaluation, chat). One expert at a time."""
        out = torch.zeros_like(x)
        for e in range(self.E):
            r, c = (idx == e).nonzero(as_tuple=True)
            if r.numel():
                out.index_add_(0, r, (self.ffn(x[r], e) * w[r, c].unsqueeze(-1)).to(out.dtype))
        return out

    @torch.no_grad()
    def balance(self):
        """After an optimizer step: busy experts' biases down, idle ones' up (DeepSeek-V3). Returns (load, dropped)."""
        load = self.load.clone()
        if load.sum() > 0:
            self.bias.add_(self.cfg.balance_rate * torch.sign(load.mean() - load))
        dropped = self.dropped.clone()
        self.load.zero_()
        self.dropped.zero_()
        return load, dropped


class CoralBlock(Block):
    """Bee's layer (attention or a Memtrix memory) whose thinking part is the shared MLP plus the experts."""

    def __init__(self, cfg: CoralConfig, kind: str):
        super().__init__(cfg, kind)
        self.moe = Experts(cfg)

    def mix(self, x, state=None, resets=None, mask=None, segs=None):
        if self.kind == "M":
            y, state = self.mem(self.n1(x), state, resets, segs)
        else:
            y, state = self.attn(self.n1(x), state, mask)
        return x + y, state

    def forward(self, x, state=None, resets=None, mask=None, segs=None):
        x, state = self.mix(x, state, resets, mask, segs)
        h = self.n2(x)
        return x + self.mlp(h) + self.moe(h), state

    def forward_aux(self, x, resets=None, mask=None, segs=None):
        x, _ = self.mix(x, None, resets, mask, segs)
        h = self.n2(x)
        y, aux = self.moe(h, with_aux=True)
        return x + self.mlp(h) + y, aux


class Coral(Frog):
    """1.0B parameters, about 160M of them at work per word. Grows like Frog (8 -> 12 -> 20 layers)."""
    block_cls = CoralBlock

    def __init__(self, cfg: CoralConfig):
        super().__init__(cfg)
        self._stats = None

    # ------------------------------------------------------------------ sizes
    def active_params(self):
        """Parameters at work for one word, at full depth: everything but the idle experts (the word table counts
        once, as the score head)."""
        b = self.blocks[0].moe
        idle = (b.E - b.k) * (b.w_up.numel() + b.w_down.numel()) // b.E
        return self.full_params() - self.cfg.n_layer * idle

    def layout(self):
        k = self.cfg.kinds()
        mix = "all attention" if "M" not in k else f"{k.count('M')} memory + {k.count('A')} attention"
        c = self.cfg
        return f"{len(k)} layers ({mix}), {c.n_expert} experts of {c.expert_hidden} (top {c.top_k}) + a shared MLP of {c.mlp_hidden}"

    # ------------------------------------------------------------------ compiling
    def prepare_kernels(self, device):
        if "M" in self.cfg.pattern:                 # the memory kernels' checks only matter with memory layers
            super().prepare_kernels(device)

    def enable_compile(self):
        super().enable_compile()                    # checks the kernels, flex attention ...
        self._compiled = torch.compile(self._train_trunk)   # ... and compiles the training trunk (with the balance loss)

    # ------------------------------------------------------------------ training
    def doc_info(self, idx):
        """Bee's document rule; with attention only, the memory kernel's document offsets are not needed."""
        if "M" in self.cfg.pattern or not self.cfg.doc_mask:
            return super().doc_info(idx)
        from oazy.tok import eot_id
        B, T = idx.shape
        resets = torch.zeros(B, T, dtype=torch.bool, device=idx.device)
        resets[:, 1:] = idx[:, :-1] == eot_id(self.cfg.tokenizer)
        doc = resets.cumsum(1)
        mask = (doc[:, None, :, None] == doc[:, None, None, :]) & torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
        return resets, mask, None

    def _train_trunk(self, idx, resets=None, mask=None, segs=None):
        """Training: (token vectors after the final norm, the layers' balance loss summed)."""
        x = self.embed(idx)
        aux = x.new_zeros((), dtype=torch.float32)
        for b in self.blocks:
            x, a = b.forward_aux(x, resets, mask, segs)
            aux = aux + a
        return self.norm(x), aux

    def losses(self, idx, targets, mentor=None, kd_weight=0.0):
        """Bee's lesson (next word + the mentor's, in slices), plus the experts' small balance loss in training."""
        resets, mask, segs = self.doc_info(idx)
        if self.training:
            h, aux = (self._compiled or self._train_trunk)(idx, resets, self.attn_mask(idx, resets, mask), segs)
        else:                                       # evaluation: nothing dropped, not compiled
            h, aux = self._trunk(idx, resets, mask, segs), None
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
        if aux is not None:
            n = max(1, len(self.blocks))
            loss = loss + self.cfg.aux_weight * aux
            stats["moe_aux"] = (aux / n).detach()
        return loss, stats

    def after_step(self):
        """The trainer calls this after every optimizer step: the biases move, the step's expert use is kept."""
        loads, drops = zip(*(b.moe.balance() for b in self.blocks))
        self._stats = (torch.stack(loads), torch.stack(drops))

    def moe_report(self):
        """For the log line: how uneven the experts' work was (busiest / average; 1.0 = perfect) and the share of
        word-expert pairs that overflowed a full expert, over the last step, averaged over the layers."""
        if self._stats is None:
            return {}
        load, drop = self._stats
        load = load.float()
        uneven = (load.max(-1).values / load.mean(-1).clamp_min(1e-9)).mean()
        dropped = drop[:, 0].sum() / drop[:, 1].sum().clamp_min(1.0)
        u, d = torch.stack([uneven, dropped]).tolist()
        return {"moe_uneven": u, "moe_dropped": d}

    # ------------------------------------------------------------------ reading with the lid off (chat)
    @torch.no_grad()
    def inspect(self, idx, states=None):
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
            h2 = b.n2(x)
            idx_e, _, _, _ = b.moe.route(h2[:, -1])
            info["experts"] = idx_e[0].tolist()
            x = x + b.mlp(h2) + b.moe(h2)
            new.append(s)
            infos.append(info)
        return self.head(self.norm(x)), new, infos
