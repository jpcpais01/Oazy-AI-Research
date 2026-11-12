"""Keep: a language model that keeps what it reads (NEXT-TOKEN), and its plain twin.

Both are the same small transformer (pre-norm, rotary positions, ReLU² MLPs, tied word table). They differ in one thing:

  full   attention reaches the whole training window. Nothing else carries the past. What falls out of the window is gone.
  keep   each attention layer reaches only the last `window` tokens (128), and nothing of it is carried from one
         training window to the next. Everything older reaches the model through its STORES: a store is a table of
         `side*side` rows that is part of the model's state, not of its weights. It is never emptied: not at the end
         of a document, not at the end of a training step.

A store learns, while the model reads, to say what comes next. Per token t:
    address   a_t = addr(h_t)  ->  a score for every row, from a product key: score(i, j) = key1[i].a1 + key2[j].a2;
              the k best rows and their weights w (softmax of the k scores)
    read      r_t = sum w S[row]                                   "after places like this one, this came"
    write     one token later, under the SAME address a_t, into the same k rows:
              S[row] += beta * w / |w|² * (v_{t+1} - r_t)          the delta rule
              v_{t+1} = val(h_{t+1}) is what did come, r_t is what those rows said would come, so only the store's
              own error is written (beta in [0, 2], from h_{t+1}).
So a row holds "what followed, the last times the text was at a place like this". The address is one function of the
place for reading and for writing, so a place that comes back finds what followed it before, from the first step of
training (two places with the same last token share rows before anything is learned); training then makes "like
this" mean more than the last token.
A write touches k rows of side*side, so text about other things lands on other rows; that is why old facts are not
worn away (a dense memory lost them within ~4,000 tokens, this one held at 16,384;
/mnt/project-files/reads-by-learning/night2-results.txt).

What the CPU pilots of Nov 11-12 settled (/mnt/project-files/reads-by-learning/night3-results.txt):
  - Writing a token under its OWN address never started working: a question does not sit at the address of its
    answer until the layers below build matching keys, and nothing told them to.
  - The store's output must start large (std dv^-0.5, not 0.02): at 0.02 the answer was in what the store brought (a
    linear probe found it 93% of the time) but too faint to learn from before the layers below had drifted.
  - read = "soft" (every row, weighted by the softmax of its score: p1' S p2) worked with the loss on answers only
    and failed with the loss on every token; the plain read worked with both. It is kept as an option only.
  - A straight-through soft gradient for the write, and a cost on writing, both stopped the store from starting.

Tokens are taken a chunk at a time (64): the reads of a chunk see the store as it was when the chunk began, its writes
land together when it ends (two tokens of a chunk on one row share it). The attention window covers the chunk itself.
Gradients flow through the store inside one training window (later reads teach earlier writes); between windows the
store is carried on, detached.

    model = Keep(KeepConfig(window=128, stores=(3, 6)))       # the twin: KeepConfig()
    state = model.init_state(batch)
    loss, state, info = model.loss(x, y, state)               # state: carry it to the next window of the same streams
                                                              # info: "lm" the loss without the write cost, "write" strengths
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KeepConfig:
    vocab_size: int = 49152
    d: int = 512
    n_layer: int = 8
    n_head: int = 8
    max_len: int = 1024
    window: int = 0             # how far attention reaches, in tokens; 0 = the whole window (the plain twin)
    stores: tuple = ()          # a store after each of these blocks (1 = after the first); () = none
    side: int = 128             # rows of a store = side * side
    k: int = 8                  # rows read and written per token
    dv: int = 64                # width of a row
    d_addr: int = 64            # width of the address vector (split in two halves for the product key)
    chunk: int = 64             # tokens whose writes land together
    read: str = "rows"          # "rows": a token reads the k rows of its address; "soft": every row, weighted by its score
    write_cost: float = 0.0     # added to the loss: this x the mean write strength (beta, 0..2) over the stores (0: it closed the stores)
    tokenizer: str = "smollm2"

    def to_dict(self):
        d = asdict(self)
        d["stores"] = list(self.stores)
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["stores"] = tuple(d.get("stores", ()))
        return cls(**d)


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class Block(nn.Module):
    def __init__(self, cfg: KeepConfig):
        super().__init__()
        self.h = cfg.n_head
        self.qkv = nn.Linear(cfg.d, 3 * cfg.d, bias=False)
        self.o = nn.Linear(cfg.d, cfg.d, bias=False)
        self.up = nn.Linear(cfg.d, 4 * cfg.d, bias=False)
        self.down = nn.Linear(4 * cfg.d, cfg.d, bias=False)
        self.qn = nn.Parameter(torch.ones(cfg.d // cfg.n_head))      # Bee6's attention: normed queries and keys, learned gains
        self.kn = nn.Parameter(torch.ones(cfg.d // cfg.n_head))
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.down.weight)

    def forward(self, x, rot, mask):
        B, T, D = x.shape
        q, k, v = self.qkv(norm(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        hd = D // self.h
        q, k = _rotate(F.rms_norm(q, (hd,), self.qn), rot).to(v.dtype), _rotate(F.rms_norm(k, (hd,), self.kn), rot).to(v.dtype)
        if mask is None:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        return x + self.down(F.relu(self.up(norm(x))).square())


def _rotate(x, rot):
    cos, sin = rot
    a, b = x[..., ::2], x[..., 1::2]
    return torch.stack([a * cos - b * sin, a * sin + b * cos], -1).flatten(-2)


class Store(nn.Module):
    """The table that is never emptied. Its rows are state (B, rows, dv), carried by the caller; its weights only say
    where to look, what to write and how hard."""

    def __init__(self, cfg: KeepConfig):
        super().__init__()
        self.side, self.k, self.dv, self.chunk, self.soft = cfg.side, cfg.k, cfg.dv, cfg.chunk, cfg.read == "soft"
        self.rows = cfg.side * cfg.side
        half = cfg.d_addr // 2
        self.half = half
        self.addr = nn.Linear(cfg.d, 2 * half, bias=False)
        self.key1 = nn.Parameter(torch.randn(cfg.side, half))
        self.key2 = nn.Parameter(torch.randn(cfg.side, half))
        self.sharp = nn.Parameter(torch.tensor(math.log(8.0)))      # scores are cosines x exp(sharp)
        self.val = nn.Linear(cfg.d, cfg.dv, bias=False)
        self.gate = nn.Linear(cfg.d, 1)
        self.out = nn.Linear(cfg.dv, cfg.d, bias=False)
        # what a read brings must be large enough to matter from the first step: started at 0.02 the store's answer was
        # there to be used (a linear probe found it 93% of the time) but too faint to learn from before the layers
        # under it had drifted; started here, the token-level pilot answered 89% right after 50 steps
        nn.init.normal_(self.out.weight, std=cfg.dv ** -0.5)
        self.top = min(cfg.k, cfg.side)             # the k best rows lie among the k best of each half-key

    def scores(self, hn):
        """(B, T, D) -> how well each half-key fits, twice (B, T, side). A row's score is the sum of its two."""
        a = self.addr(hn)
        t = self.sharp.exp()
        s1 = t * F.normalize(a[..., :self.half], dim=-1) @ F.normalize(self.key1, dim=-1).t()
        s2 = t * F.normalize(a[..., self.half:], dim=-1) @ F.normalize(self.key2, dim=-1).t()
        return s1, s2

    def address(self, s1, s2):
        """-> the k best rows (B, T, k) and their weights (B, T, k), from the top x top best pairs of half-keys."""
        v1, i1 = s1.topk(self.top, -1)
        v2, i2 = s2.topk(self.top, -1)
        sc = (v1[..., :, None] + v2[..., None, :]).flatten(-2)
        ix = (i1[..., :, None] * self.side + i2[..., None, :]).flatten(-2)
        val, pos = sc.topk(min(self.k, sc.size(-1)), -1)
        return ix.gather(-1, pos), val.softmax(-1)

    def read_all(self, p1, p2, S):
        """The soft read: every row, weighted p1[i] * p2[j] (the softmax over all rows of a product key). (B, n, dv)"""
        B, n, _ = p1.shape
        fast = S.device.type == "cuda"
        with torch.autocast(S.device.type, dtype=torch.bfloat16, enabled=fast):
            m = torch.bmm(p1, S.view(B, self.side, self.side * self.dv)) if not fast else \
                torch.bmm(p1.to(torch.bfloat16), S.view(B, self.side, self.side * self.dv).to(torch.bfloat16))
        return (m.float().view(B, n, self.side, self.dv) * p2[..., None]).sum(2)

    def forward(self, h, S):
        """h (B, T, D), S (B, rows, dv) -> what the reads bring (B, T, D), the store after the writes, each token's
        write strength (B, T). The first token of a window writes nothing (the token before it is not in the window)."""
        with torch.autocast(h.device.type, enabled=False):
            hn = norm(h.float())
            B, T, _ = hn.shape
            s1, s2 = self.scores(hn)
            idx, w = self.address(s1, s2)
            if self.soft:
                p1, p2 = s1.softmax(-1), s2.softmax(-1)
            v = self.val(hn)
            beta = 2 * torch.sigmoid(self.gate(hn))
            beta = torch.cat([torch.zeros_like(beta[:, :1]), beta[:, 1:]], 1)
            back = lambda x: torch.cat([x[:, :1], x[:, :-1]], 1)   # each token writes under the address of the token before it
            idx_w, w_w = back(idx), back(w)
            wn = w_w / (w_w * w_w).sum(-1, keepdim=True)
            K = idx.size(-1)
            live = torch.ones(B, T, K, device=h.device)             # the writes that share a row share it; the first token writes nothing
            live[:, 0] = 0
            grad = torch.is_grad_enabled()
            outs = []
            for c in range(0, T, self.chunk):
                sl = slice(c, min(T, c + self.chunk))
                n = sl.stop - sl.start
                if self.soft:
                    outs.append(self.read_all(p1[:, sl], p2[:, sl], S))
                else:
                    go = idx[:, sl].reshape(B, n * K)[..., None].expand(-1, -1, self.dv)
                    outs.append((w[:, sl, :, None] * S.gather(1, go).view(B, n, K, self.dv)).sum(2))
                ix = idx_w[:, sl].reshape(B, n * K)
                gi = ix[..., None].expand(-1, -1, self.dv)
                r = (w_w[:, sl, :, None] * S.gather(1, gi).view(B, n, K, self.dv)).sum(2)   # what the store said would come
                hits = torch.zeros(B, self.rows, device=h.device).scatter_add_(1, ix, live[:, sl].reshape(B, n * K))
                share = (1.0 / hits.gather(1, ix).clamp_(min=1.0)).view(B, n, K, 1)
                e = beta[:, sl] * (v[:, sl] - r)                                            # how wrong it was
                upd = (e[:, :, None, :] * wn[:, sl, :, None] * share).reshape(B, n * K, self.dv)
                S = S.scatter_add(1, gi, upd) if grad else S.scatter_add_(1, gi, upd)
            return self.out(torch.cat(outs, 1)).to(h.dtype), S, beta.squeeze(-1)


class Keep(nn.Module):
    def __init__(self, cfg: KeepConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d)
        # tied word table: the final vector has length sqrt(d), so the largest score is about std * d. 0.02 gives
        # 10 at d = 512 (as in GPT-2); a narrow model needs a larger std to be able to be sure of anything.
        nn.init.normal_(self.emb.weight, std=max(0.02, 10.0 / cfg.d))
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.store = nn.ModuleDict({str(i): Store(cfg) for i in cfg.stores})
        hd = cfg.d // cfg.n_head
        f = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
        ang = torch.arange(cfg.max_len).float()[:, None] * f[None]
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)
        self._masks = {}

    # ------------------------------------------------------------ state
    def init_state(self, batch: int, device=None):
        device = device or self.emb.weight.device
        return {i: torch.zeros(batch, s.rows, s.dv, device=device) for i, s in self.store.items()}

    @staticmethod
    def detach(state):
        return {i: s.detach() for i, s in state.items()}

    def _mask(self, T, device):
        if not self.cfg.window or self.cfg.window >= T:
            return None
        key = (T, str(device))
        if key not in self._masks:
            i = torch.arange(T, device=device)
            self._masks[key] = (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < self.cfg.window)
        return self._masks[key]

    # ------------------------------------------------------------ forward
    def hidden(self, x, state=None):
        """Token ids (B, T) -> final vectors (B, T, D), the stores after reading x, write strengths (B, T) per store."""
        B, T = x.shape
        if self.store and state is None:
            state = self.init_state(B, x.device)
        rot = (self.cos[:T], self.sin[:T])
        mask = self._mask(T, x.device)
        h = norm(self.emb(x))
        new, info = {}, {}
        for i, blk in enumerate(self.blocks, 1):
            h = blk(h, rot, mask)
            if str(i) in self.store:
                r, new[str(i)], info[str(i)] = self.store[str(i)](h, state[str(i)])
                h = h + r
        return norm(h), new, info

    def loss(self, x, y, state=None, plain=False, asked=None, asked_weight=1.0):
        """asked: (B, T) bool, the targets that are part of a question block; they count asked_weight times in the loss.
        -> loss, the stores after x, info {"lm": the plain next-token loss, "asked": the loss on the asked targets,
           "write": mean write strength per store}."""
        from oazy.lmloss import head_loss, plain_loss
        h, new, beta = self.hidden(x, state)
        if plain or h.device.type != "cuda":
            loss = plain_loss(h, self.emb.weight, y)[0]
        else:
            loss = head_loss(h, self.emb.weight, y)[0]
        info = {"lm": loss.detach(), "write": [b.mean().detach() for b in beta.values()]}
        if asked is not None and bool(asked.any()):
            la = F.cross_entropy(h[asked].float() @ self.emb.weight.float().t(), y[asked])
            info["asked"] = la.detach()
            if asked_weight != 1.0:
                loss = loss + (asked_weight - 1.0) * asked.float().mean() * la
        if beta and self.cfg.write_cost:
            loss = loss + self.cfg.write_cost * sum(b.mean() for b in beta.values()) / len(beta)
        return loss, new, info

    def scores_at(self, h, rows, cols):
        """Scores over the words at chosen places: h (B, T, D), rows/cols (n,) -> (n, vocab)."""
        return (h[rows, cols].float() @ self.emb.weight.float().t())

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
