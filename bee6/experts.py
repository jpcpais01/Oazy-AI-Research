"""Bee6's fine-grained experts: the MLP of every layer cut into many small experts, each word running a few of them.
Still a NEXT-WORD model. This is the "thinking part" of Bee6-MoE (bee6/moe.py); attention, the word table, the value
tables and the score head are Bee6's, untouched.

João's ask (Nov 10): keep what a Bee6-200M pays for on every word and make the MLP around it about 8x bigger, each word
still doing the same amount of work. The numbers, for one layer at Bee6's width (1,024):

                        Bee6                         Bee6-MoE
  thinking part         one ReLU² MLP of 3,840       1 shared expert + 80 routed experts, each a ReLU² MLP of 384
  per word              all 3,840 neurons            the shared one + the 9 the router picks = 10 x 384 = 3,840
  weights               7.9M                         81 x 0.79M = 63.7M (8.1x)

So a word pays exactly what it pays in Bee6 (the same 3,840 neurons' worth of products), but the layer holds 8.1x more.
Small experts (fine-grained, DeepSeekMoE 2024) beat a few big ones at the same cost: 9 of 80 can be combined in far
more ways than 1 of 8, so each expert can specialise narrowly.

The parts, and where each comes from:
  shared expert     DeepSeekMoE / Qwen-MoE: one expert every word runs, so common knowledge has one home and the routed
                    ones are free to specialise
  sigmoid router    DeepSeek-V3: each expert's score is a sigmoid of (word . expert's key), read in float32; the 9
                    highest win. Their weights are the scores scaled to sum to 9 (one each on average), so at the start
                    the layer adds up like Bee6's MLP: 10 pieces of 384 at weight ~1
  centred choice    the sparse MLP's lesson (bee6/sparse.py): an expert is picked where it stands out for this word, its
                    score minus its usual score over words, over its usual spread. Without it, an expert whose score
                    rises for every word at once (the stream has a strong common direction) is every word's pick, and
                    the run collapses onto one expert before the balancing biases can catch up (seen in the tiny run:
                    every word on the same expert). Learning uses the pass's own mean and spread (tens of thousands of
                    words, 2 x 80 averages); reading uses their running averages, as batch norm, so reading a word never
                    depends on the other words. Only the choice is centred: the weights use the raw scores
  bias balancing    DeepSeek-V3: a small bias per expert, used only to choose; after every step a busy expert's bias goes
                    down and an idle one's up, so every expert gets its share of words without a loss pulling against
                    learning. A tiny balance loss (1e-4, as Coral) is the safety net
  fixed queues      GShard / Switch, as Coral and the sparse MLP: in training every expert gets a queue of 1.25x its fair
                    share of the pass, the words are gathered into the queues and all 80 run as one batched product
                    (static shapes: torch.compile sees one graph). A word that finds a queue full gets nothing from that
                    expert (it still has the shared one and its other picks); the log shows the share as "dropped"
  reading           held-out loss, HellaSwag, the exams: queues just long enough for every word (nothing dropped), so a
                    word's answer never depends on the other words of the batch. Writing one word at a time (chat): each
                    word's own experts' weights are taken directly
  the combine       each queue's answers are weighted and added back into their words in float32 (index_add), so the
                    9 x 1.25 copies of a word's answer are never kept in memory at once

Precision: the experts run in bf16 (FP8 stays on for attention). Coral's lesson: FP8 for 80 small grids costs more in
scaling than it saves.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Experts(nn.Module):
    """Bee6's MLP as 1 shared + n routed ReLU² experts, stored as stacked grids: w_up (E+1, h, D), w_down (E+1, D, h).
    Grid 0 is the shared expert; grids 1..E are the routed ones (the router scores those E)."""

    def __init__(self, cfg):
        super().__init__()
        D, E, h = cfg.d_model, int(cfg.experts), int(cfg.expert_hidden)
        if cfg.mlp != "relu2":
            raise ValueError("experts work with the ReLU² MLP only (mlp=relu2)")
        if not 1 <= cfg.expert_top <= E:
            raise ValueError(f"expert_top={cfg.expert_top} must be between 1 and experts={E}")
        self.cfg, self.D, self.E, self.h, self.k = cfg, D, E, h, int(cfg.expert_top)
        self.S = int(cfg.shared_experts)
        if self.S not in (0, 1):
            raise ValueError("shared_experts is 0 or 1")
        n = E + self.S
        self.w_up = nn.Parameter(torch.empty(n, h, D))
        self.w_down = nn.Parameter(torch.zeros(n, D, h))
        self.router = nn.Linear(D, E, bias=False)
        self.register_buffer("bias_route", torch.zeros(E))      # balancing biases: used only to choose
        self.register_buffer("mu_route", torch.zeros(E))        # each expert's usual score over words and its spread
        self.register_buffer("sd_route", torch.ones(E))         # (running averages: reading centres with these)
        self.register_buffer("seen_route", torch.tensor(0))
        self.small = 8                                           # reading this many words or fewer: weights directly
        self.reset(3 ** 0.5 * D ** -0.5)

    @torch.no_grad()
    def reset(self, bound):
        """Bee5's start: up uniform at half the attention's bound, down at zero (each layer starts as a no-op)."""
        nn.init.uniform_(self.w_up, -0.5 * bound, 0.5 * bound)
        nn.init.zeros_(self.w_down)
        nn.init.normal_(self.router.weight, std=self.D ** -0.5)

    def share_text(self):
        return f"{'1 shared + ' if self.S else ''}{self.k} of {self.E} experts of {self.h}"

    # ------------------------------------------------------------------ routing
    def route(self, x):
        """x (N, D) -> picks (N, k) in 0..E-1, their weights (N, k) float32, the balance loss, words per expert (E,),
        and the scores' mean and mean square over the words (E,) for the running averages."""
        logits = F.linear(x.float(), self.router.weight.float())
        s = torch.sigmoid(logits)                                       # (N, E)
        lg = logits.detach()
        if self.training and lg.shape[0] > 1:
            mu, sd = lg.mean(0), lg.std(0).clamp_min(1e-4)
        else:
            mu, sd = self.mu_route, self.sd_route
        idx = torch.topk((lg - mu) / sd + self.bias_route, self.k, dim=-1).indices
        w = s.gather(1, idx)
        w = w * (self.k / (w.sum(-1, keepdim=True) + 1e-9))            # one each on average: Bee6's MLP at the start
        counts = torch.zeros(self.E, device=x.device, dtype=torch.float32).scatter_add_(
            0, idx.reshape(-1), torch.ones(idx.numel(), device=x.device, dtype=torch.float32))
        frac = counts / max(1, idx.numel())                             # the share of the work, no gradient
        prob = (s / (s.sum(-1, keepdim=True) + 1e-9)).mean(0)           # the router's average wish, with gradient
        aux = self.E * (frac * prob).sum()
        return idx, w, aux, counts, lg.mean(0), lg.square().mean(0)

    # ------------------------------------------------------------------ the experts
    def shared(self, x):
        if not self.S:
            return torch.zeros_like(x)
        wu, wd = self.w_up[0].to(x.dtype), self.w_down[0].to(x.dtype)
        return F.linear(F.relu(F.linear(x, wu)).square(), wd)

    def queued(self, x, idx, w, C):
        """Fixed shapes: every routed expert gets C slots, all run as one batched product. -> (out (N, D) float32,
        how many of the N x k picks found their queue full)."""
        N, D = x.shape
        E, k = self.E, self.k
        flat = idx.t().reshape(-1)                                      # (k N): every word's first choice queues first
        tok = torch.arange(N, device=x.device).repeat(k)
        wf = w.t().reshape(-1)
        order = torch.sort(flat, stable=True).indices
        counts = torch.zeros(E, device=x.device, dtype=torch.long).scatter_add_(0, flat, torch.ones_like(flat))
        start = counts.cumsum(0) - counts
        pos = torch.empty_like(flat).scatter_(0, order, torch.arange(flat.numel(), device=x.device) - start[flat[order]])
        keep = pos < C
        slot = torch.where(keep, flat * C + pos, torch.full_like(pos, E * C))     # a full queue: the spare slot
        src = torch.full((E * C + 1,), N, device=x.device, dtype=torch.long).scatter_(0, slot, tok)[:E * C]
        gs = torch.zeros(E * C + 1, device=x.device, dtype=torch.float32).scatter_(0, slot, wf)[:E * C]
        xp = torch.cat([x, x.new_zeros(1, D)])
        xin = xp[src].view(E, C, D)                                     # empty slots read a row of zeros
        S = self.S
        wu, wd = self.w_up[S:].to(x.dtype), self.w_down[S:].to(x.dtype)
        a = F.relu(torch.bmm(xin, wu.mT)).square()                     # (E, C, h)
        y = torch.bmm(a, wd.mT).reshape(E * C, D)                       # (E C, D)
        out = torch.zeros(N + 1, D, device=x.device, dtype=torch.float32)
        out.index_add_(0, src, y.float() * gs[:, None])                 # empty slots land on the spare row N
        return out[:N], (~keep).sum().float()

    def direct(self, x, idx, w):
        """A few words (chat): each word's own experts' weights, gathered. Reads only k / E of the routed weights."""
        S = self.S
        u = self.w_up[S:][idx].to(x.dtype)                              # (N, k, h, D)
        d = self.w_down[S:][idx].to(x.dtype)                            # (N, k, D, h)
        a = F.relu(torch.einsum("nd,nkhd->nkh", x, u)).square() * w.unsqueeze(-1).to(x.dtype)
        return torch.einsum("nkh,nkdh->nd", a, d).float()

    def forward(self, x, with_aux=False):
        """x (..., D) -> (..., D); with_aux (training): also (balance loss, words per expert (E,), dropped share,
        the scores' mean and mean square (E,))."""
        shp = x.shape
        xf = x.reshape(-1, self.D)
        N = xf.shape[0]
        idx, w, aux, counts, mean, sq = self.route(xf)
        if self.training:
            C = max(8, int(math.ceil(self.cfg.expert_capacity * N * self.k / self.E / 8) * 8))
            r, dropped = self.queued(xf, idx, w, C)
        elif N <= self.small:
            r, dropped = self.direct(xf, idx, w), None
        else:
            C = int(counts.max().item())                                # reading: queues long enough for every word
            r, dropped = self.queued(xf, idx, w, max(1, C))
        y = (self.shared(xf).float() + r).to(x.dtype).view(shp)
        if not (with_aux and self.training):
            return y
        return y, (aux, counts, dropped / max(1, N * self.k), mean, sq)

    @torch.no_grad()
    def balance(self, load, mean=None, sq=None, keep=0.9):
        """After an optimizer step: the experts' usual scores follow the step's (running average), and busy experts'
        biases go down, idle ones' up (DeepSeek-V3)."""
        if mean is not None:      # the step's scores come from bf16 autocast: the running averages stay float32
            mean = mean.to(self.mu_route.device, self.mu_route.dtype)
            sq = sq.to(self.mu_route.device, self.mu_route.dtype)
            sd = (sq - mean.square()).clamp_min(1e-8).sqrt()
            if int(self.seen_route) == 0:
                self.mu_route.copy_(mean)
                self.sd_route.copy_(sd)
            else:
                self.mu_route.lerp_(mean, 1 - keep)
                self.sd_route.lerp_(sd, 1 - keep)
            self.seen_route.add_(1)
        load = load.to(self.bias_route.device, self.bias_route.dtype)
        if load.sum() > 0:
            self.bias_route.add_(self.cfg.expert_balance * torch.sign(load.mean() - load))
