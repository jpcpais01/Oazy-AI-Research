"""Bee6's sparse MLP: each word lights up only the most useful ~10% of a layer's MLP, picked by a small low-rank guesser.

João's idea (Nov 9, Special Lab): "use dimension reducing algorithms to make like a MoE where each time the NN only
lights up the 10% most important nodes, well picked by the dimension reducing algorithm". Still a NEXT-WORD model.

How it works, for one layer (Bee6's MLP: 1,024 -> 3,840 hidden -> 1,024, ReLU squared):

  blocks      the 3,840 hidden neurons are cut into blocks of 128 (30 blocks). A block is the unit that is on or off,
              so the GPU still multiplies whole tiles (one neuron at a time would be far too fine for a GPU)
  the picker  each word's vector is squeezed to 128 numbers (a learned projection: the "dimension reduction", like a
              PCA that keeps what matters for choosing) and those 128 numbers score the 30 blocks. The top 10% (3
              blocks) run for that word; the other 27 are skipped. The picker reads only that word's own vector, so
              it is causal: no word ever looks ahead, in training or reading
  learning    the first 10% of the tokens the whole MLP runs (dense, exactly Bee6), and the picker learns, on the
              side, to guess which blocks matter most for each word (most activity x the block's output strength).
              Then the switch flips and only the picked blocks run. From then on a small sample of words (1 in 16)
              still gets the full up-product, without gradient, so the picker keeps its lesson and we can see how well
              it picks ("recall": the share of the truly top blocks it picked)
  gate        a picked block's output is multiplied by 1 + sigmoid(score) - (the same, held fixed): the number is
              exactly 1 (nothing changes), but the gradient tells the picker whether a block it picked helped
  balancing   DeepSeek-V3's bias balancing, as in Coral: each block has a small bias added to its score only when
              choosing; after every step a busy block's bias goes down and an idle one's up, so every block is used
              and learns (and the GPU's fixed-size block queues rarely overflow)

On the GPU in training every block gets a queue of fixed size (1.25 x its fair share of the words), the words are
gathered into the queues and all blocks run as one batched product (static shapes, so torch.compile and CUDA graphs
work); a word that finds a block's queue full gets that block's share as zero (counted as "dropped"). Reading a batch
(held-out loss, HellaSwag) uses queues just long enough for every word (nothing dropped). Writing one word at a time
(chat, CPU, phone) takes the picked blocks' weights directly: only ~10% of the MLP's weights are read per word.

The sparse products run in bf16 (FP8 stays on for attention and for the dense first 10%).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import fp8 as FP8


class SparseMLP(nn.Module):
    """Bee5's ReLU² MLP (same up / down weights, same names) whose hidden neurons run in picked blocks."""

    def __init__(self, cfg):
        super().__init__()
        D, H = cfg.d_model, cfg.mlp_hidden
        if cfg.mlp != "relu2":
            raise ValueError("sparse_mlp works with the ReLU² MLP only (mlp=relu2)")
        bs = int(cfg.sparse_block)
        if H % bs:
            raise ValueError(f"sparse_block={bs} must divide the MLP's {H} hidden neurons")
        self.cfg, self.kind = cfg, "relu2"
        self.D, self.H, self.bs, self.nb = D, H, bs, H // bs
        self.k = max(1, min(self.nb, int(round(cfg.sparse_mlp * self.nb))))
        self.up = nn.Linear(D, H, bias=False)
        self.down = nn.Linear(H, D, bias=False)
        r = int(cfg.sparse_rank)
        self.pick_in = nn.Linear(D, r, bias=False)          # the squeeze: the word's vector -> r numbers
        self.pick_out = nn.Linear(r, self.nb, bias=False)   # r numbers -> a score per block
        nn.init.normal_(self.pick_in.weight, std=1 / math.sqrt(D))
        nn.init.normal_(self.pick_out.weight, std=1 / math.sqrt(r))
        self.register_buffer("bias_pick", torch.zeros(self.nb))   # balancing biases: used only to choose
        self.register_buffer("mu_pick", torch.zeros(self.nb))     # each block's usual score and its spread (running),
        self.register_buffer("sd_pick", torch.ones(self.nb))      # so a block is picked where it stands out for a word
        self.register_buffer("seen_pick", torch.tensor(0))
        self.live = False            # False = the dense first share of the run; True = only the picked blocks run
        self.small = 8               # reading this many words or fewer: take the picked blocks' weights directly

    # ------------------------------------------------------------------ pieces
    def scores(self, x):
        """(N, D) -> the picker's scores (N, nb), float32. It reads a copy of x: it never shapes the stream."""
        return self.pick_out(self.pick_in(x.detach())).float()

    def choose(self, s):
        """The top k blocks per word, by how much each block's score stands out from its usual (mean and spread over
        words), + the balancing bias. Only the choice uses them: the gate and the lesson use the raw score.

        Without the centring a block that is busy for most words is every word's pick, and the balancing biases (small
        steps) take a whole run to catch up. Learning uses this pass's own mean and spread (tens of thousands of words:
        a word's pick depends on the pass only through these 2 x 30 averages); reading uses their running averages, like
        batch norm, so reading a word never depends on any other word."""
        s = s.detach()
        if self.training and s.shape[0] > 1:
            mu, sd = s.mean(0), s.std(0).clamp_min(1e-4)
        else:
            mu, sd = self.mu_pick, self.sd_pick
        z = (s - mu) / sd
        return torch.topk(z + self.bias_pick, self.k, dim=-1).indices

    @torch.no_grad()
    def importance(self, a):
        """How much each block matters for each word: its activity times its output strength. a (N, H) after ReLU²."""
        c = self.down.weight.detach().float().norm(dim=0)
        c = c + 0.01 * c.mean() + 1e-12          # at the start down is all zeros: then activity alone decides
        return (a.float() * c).view(-1, self.nb, self.bs).sum(-1)

    def lesson(self, s, a, idx):
        """The picker's lesson against the true top blocks, and its recall (share of the true top k it picked)."""
        imp = self.importance(a)
        true = imp.topk(self.k, dim=-1).indices
        target = torch.zeros_like(s).scatter_(1, true, 1.0)
        aux = F.binary_cross_entropy_with_logits(s, target)
        got = torch.zeros_like(s).scatter_(1, idx, 1.0)
        recall = ((got * target).sum() / target.sum()).detach()
        return aux, recall

    def blocks(self, dtype):
        """The up and down weights as (nb, bs, D) grids (views of the same weights, no copies kept)."""
        wu = self.up.weight.to(dtype).view(self.nb, self.bs, self.D)
        wd = self.down.weight.to(dtype).view(self.D, self.nb, self.bs).permute(1, 2, 0)
        return wu, wd

    # ------------------------------------------------------------------ the two ways to run the picked blocks
    def queued(self, x, idx, gate, C):
        """Fixed shapes: every block gets C slots, all blocks run as one batched product. -> (out (N, D), dropped)."""
        N, D = x.shape
        nb, k = self.nb, self.k
        flat = idx.t().reshape(-1)                                  # (k N): every word's first choice queues first
        tok = torch.arange(N, device=x.device).repeat(k)
        order = torch.sort(flat, stable=True).indices
        counts = torch.zeros(nb, device=x.device, dtype=torch.long).scatter_add_(0, flat, torch.ones_like(flat))
        start = counts.cumsum(0) - counts
        pos = torch.empty_like(flat).scatter_(0, order, torch.arange(flat.numel(), device=x.device) - start[flat[order]])
        keep = pos < C
        slot = torch.where(keep, flat * C + pos, torch.full_like(pos, nb * C))
        src = torch.full((nb * C + 1,), N, device=x.device, dtype=torch.long).scatter_(0, slot, tok)
        xp = torch.cat([x, x.new_zeros(1, D)])
        xin = xp[src[:nb * C]].view(nb, C, D)                       # empty slots read a row of zeros
        wu, wd = self.blocks(xin.dtype)
        h = torch.bmm(xin, wu.mT)                                   # (nb, C, bs)
        y = torch.bmm(F.relu(h).square(), wd).reshape(nb * C, D)    # (nb * C, D)
        yp = torch.cat([y, y.new_zeros(1, D)])
        ya = yp[slot].view(k, N, D)                                 # a dropped choice gets zeros from that block
        out = (ya * gate.t().unsqueeze(-1).to(ya.dtype)).sum(0)
        return out, (~keep).sum().float()

    def direct(self, x, idx, gate):
        """A few words: take each word's picked blocks' weights directly (reads ~k/nb of the MLP's weights).
        On the CPU (and a phone) block by block on views of the weights, nothing copied: the fastest way measured
        (3 of 30 blocks: ~4x faster than the whole MLP, 4 threads); on a GPU in one gathered product (no syncs)."""
        if x.is_cuda or self.training:
            wu, wd = self.blocks(x.dtype)
            u, d = wu[idx], wd[idx]                                 # (N, k, bs, D)
            h = torch.einsum("nd,nkbd->nkb", x, u)
            a = F.relu(h).square() * gate.unsqueeze(-1).to(h.dtype)
            return torch.einsum("nkb,nkbd->nd", a, d)
        wu = self.up.weight.view(self.nb, self.bs, self.D)
        wd = self._down_rows().view(self.nb, self.bs, self.D)
        out = []
        for n, picks in enumerate(idx.tolist()):
            xn, y = x[n:n + 1].to(wu.dtype), None
            for j, b in enumerate(picks):
                t = (F.relu(F.linear(xn, wu[b])).square() * gate[n, j].to(wu.dtype)) @ wd[b]
                y = t if y is None else y + t
            out.append(y)
        return torch.cat(out).to(x.dtype)

    def _down_rows(self):
        """down's weights with one row per hidden neuron (a copy kept for reading, renewed when the weights change)."""
        w = self.down.weight
        key = (w.data_ptr(), w._version, w.dtype)
        if getattr(self, "_rows_key", None) != key:
            self._rows = w.detach().t().contiguous()
            self._rows_key = key
        return self._rows

    # ------------------------------------------------------------------ the layer
    def dense(self, x):
        mode = self.cfg.fp8
        a = F.relu(FP8.linear(x, self.up.weight, mode)).square()
        return FP8.linear(a, self.down.weight, mode), a

    def forward(self, x, with_aux=False):
        """x (..., D) -> (..., D); with_aux: also (picker's lesson, recall, words per block (nb,), dropped share)."""
        shp = x.shape
        xf = x.reshape(-1, self.D)
        N = xf.shape[0]
        s = self.scores(xf)
        idx = self.choose(s)
        if not self.live:                                           # the dense first share of the run
            y, a = self.dense(xf)
            if not (with_aux and self.training):
                return y.view(shp)
            aux, recall = self.lesson(s, a, idx)
            return y.view(shp), (aux, recall, self._counts(idx), xf.new_zeros((), dtype=torch.float32), *self._moments(s))
        sg = torch.sigmoid(s.gather(1, idx))
        gate = 1 + sg - sg.detach()                                 # exactly 1; the gradient reaches the picker
        if self.training:
            C = max(8, int(math.ceil(self.cfg.sparse_capacity * N * self.k / self.nb / 8) * 8))
            y, dropped = self.queued(xf, idx, gate, C)
        elif N <= self.small:
            y, dropped = self.direct(xf, idx, gate), None
        else:
            C = int(self._counts(idx).max().item())                 # reading: queues long enough for every word
            y, dropped = self.queued(xf, idx, gate, max(1, C))
        y = y.view(shp)
        if not (with_aux and self.training):
            return y
        n = max(1, N // max(1, int(self.cfg.sparse_check)))        # a sample of words gets the full up-product
        with torch.no_grad():
            a = F.relu(F.linear(xf[:n], self.up.weight.to(xf.dtype))).square()
        aux, recall = self.lesson(s[:n], a, idx[:n])
        return y, (aux, recall, self._counts(idx), dropped / (N * self.k), *self._moments(s))

    @staticmethod
    def _moments(s):
        s = s.detach()
        return s.mean(0), s.square().mean(0)

    def _counts(self, idx):
        return torch.zeros(self.nb, device=idx.device, dtype=torch.float32).scatter_add_(
            0, idx.reshape(-1), torch.ones(idx.numel(), device=idx.device, dtype=torch.float32))

    @torch.no_grad()
    def balance(self, load, mean=None, sq=None, keep=0.9):
        """After an optimizer step: the blocks' usual scores follow the step's (running average), and busy blocks'
        biases go down, idle ones' up (DeepSeek-V3)."""
        if mean is not None:
            sd = (sq - mean.square()).clamp_min(1e-8).sqrt()
            if int(self.seen_pick) == 0:
                self.mu_pick.copy_(mean)
                self.sd_pick.copy_(sd)
            else:
                self.mu_pick.lerp_(mean, 1 - keep)
                self.sd_pick.lerp_(sd, 1 - keep)
            self.seen_pick.add_(1)
        if load.sum() > 0:
            self.bias_pick.add_(self.cfg.sparse_balance * torch.sign(load.mean() - load))

    def share_text(self):
        return f"{self.k} of {self.nb} blocks of {self.bs}"
