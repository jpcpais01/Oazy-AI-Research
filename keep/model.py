"""Keep: a language model that keeps what it reads (NEXT-TOKEN), and its plain twin.

Both are the same small transformer (pre-norm, rotary positions, ReLU² MLPs, tied word table). They differ in one thing:

  full   attention reaches the whole training window. Nothing else carries the past. What falls out of the window is gone.
  keep   each attention layer reaches only the last `window` tokens (128), and nothing of it is carried from one
         training window to the next. Everything older reaches the model through its STORES: a store is a table of
         `side*side` rows that is part of the model's state, not of its weights. It is never emptied: not at the end
         of a document, not at the end of a training step.
  wide   both: the stores, and attention over the whole training window (KeepConfig(window=0, stores=...)).
         Trained plainly its stores stay unused (attention answers whatever is in view, and nothing teaches the
         writes), so run.py's "mixed" trains it with `model.reach` cut to Keep's 128 tokens on half of the steps.

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

FEED (KeepConfig(feed=True)): the store is read where it stands (low in the stack) and written from the TOP.
A transformer is feed-forward: what its last layer works out at one token reaches the first layers of a later token
only as a word that was said. A store read and written at one layer does not change that (one lookup per store and
pass). Written from the top, it does: the first layers of every later token read what the whole stack concluded
before, so each further token read is one more turn of the whole stack, and reading a story again is thinking about
it: a chain of d facts told in any order is followed after d readings, whatever the number of layers
(/mnt/project-files/clock/fb.py: trained on depth 1..3, 100% at depth 8 after 8 readings, 457k weights; the same
model written from its reading layer stays at guessing from depth 2 on).
    write     at token t, under the address the reading layer made at token t - 1 (as before), the value
              val(top_t) and the strength from top_t.
    point     (KeepConfig(point=J)) the top chooses under which of the J tokens before it its write goes: a
              statement's conclusion is filed under the place of the thing it is about, wherever that stood in the
              statement. It starts at the token before.
The window is then taken chunk by chunk through the whole stack (the reads of a chunk need what the top said in the
chunks before), each block keeping the keys and values of the tokens still in reach.

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
    feed: bool = False          # the stores are written from the top of the stack (what the last layer says), not from the layer they are read at
    point: int = 0              # feed: the top files a write under the place of one of this many tokens before it (0: the token before)
    share: str = "all"          # feed: how the writes of a chunk share a row. "all": every write on it counts, also one that is held back;
                                # "strength": a write counts by its strength (beta, up to 1), so a write that is held back weakens no other
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

    def forward(self, x, rot, window=0):
        B, T, D = x.shape
        q, k, v = self.qkv(norm(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        hd = D // self.h
        q, k = _rotate(F.rms_norm(q, (hd,), self.qn), rot).to(v.dtype), _rotate(F.rms_norm(k, (hd,), self.kn), rot).to(v.dtype)
        if not window or window >= T:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            a = near(q, k, v, window)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        return x + self.down(F.relu(self.up(norm(x))).square())


def _step(blk: Block, x, rot, past, window: int):
    """A block on a chunk of new tokens, given the rotated keys and the values of the tokens before it that are still
    in reach (past = (k, v), each (B, H, c, hd), or None). The same numbers as the block on the whole sequence.
    -> the chunk after the block, the keys and values to hand to the next chunk."""
    B, n, D = x.shape
    q, k, v = blk.qkv(norm(x)).view(B, n, 3, blk.h, D // blk.h).permute(2, 0, 3, 1, 4)
    hd = D // blk.h
    q, k = _rotate(F.rms_norm(q, (hd,), blk.qn), rot).to(v.dtype), _rotate(F.rms_norm(k, (hd,), blk.kn), rot).to(v.dtype)
    if past is not None:
        k, v = torch.cat([past[0], k], 2), torch.cat([past[1], v], 2)
    c = k.size(2) - n
    i = torch.arange(n, device=x.device)[:, None] + c
    j = torch.arange(c + n, device=x.device)[None, :]
    mask = (j <= i) & ((i - j < window) if window else (j >= 0))
    a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    x = x + blk.o(a.transpose(1, 2).reshape(B, n, D))
    keep = k.size(2) if not window else min(k.size(2), window - 1)
    return x + blk.down(F.relu(blk.up(norm(x))).square()), (k[:, :, k.size(2) - keep:], v[:, :, v.size(2) - keep:])


def near(q, k, v, W: int):
    """Attention that reaches W tokens (the token itself and the W - 1 before it), computed over those only: the
    sequence is cut into blocks of W, and a block's queries meet the keys of the block before and of their own,
    2W keys instead of all T. The same numbers as a T x T mask (keep/tests checks it); the cost per token no longer
    grows with T. q, k, v (B, H, T, hd), T > W."""
    B, H, T, hd = q.shape
    first = F.scaled_dot_product_attention(q[:, :, :W], k[:, :, :W], v[:, :, :W], is_causal=True)
    pad = (-T) % W
    if pad:                                         # the keys added here lie after every real query: never seen
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
    nb = (T + pad) // W
    cut = lambda t: t.view(B, H, nb, W, hd).transpose(1, 2)                         # (B, blocks, H, W, hd)
    two = lambda t: torch.cat([t[:, :-1], t[:, 1:]], 3).reshape(B * (nb - 1), H, 2 * W, hd)
    i = torch.arange(W, device=q.device)[:, None]
    m = torch.arange(2 * W, device=q.device)[None, :]
    rest = F.scaled_dot_product_attention(cut(q)[:, 1:].reshape(B * (nb - 1), H, W, hd), two(cut(k)), two(cut(v)),
                                          attn_mask=(m > i) & (m <= i + W))
    rest = rest.view(B, nb - 1, H, W, hd).transpose(1, 2).reshape(B, H, (nb - 1) * W, hd)
    return torch.cat([first, rest], 2)[:, :, :T]


def _rotate(x, rot):
    cos, sin = rot
    a, b = x[..., ::2], x[..., 1::2]
    return torch.stack([a * cos - b * sin, a * sin + b * cos], -1).flatten(-2)


def _chunks(table, at, wt, v, beta, coef, chunk: int, keep: bool):
    """The store's loop, on plain tensors. table (B * rows, dv), written in place; at (2, B, T, K) rows of the table
    (0: read for the model, 1: read and written by the delta rule); wt (2, B, T, K) their weights; v (B, T, dv) what
    to write; beta (B, T, 1) how hard; coef (B, T, K) a write's weight on each of its rows.
    -> what the reads bring (B, T, dv) and, if keep, what the way back needs (the rows read, v - r, the error)."""
    _, B, T, K = at.shape
    dv = table.size(1)
    outs, rows_, d_, e_ = [], [], [], []
    for c in range(0, T, chunk):
        sl = slice(c, min(T, c + chunk))
        n = sl.stop - sl.start
        rows = table.index_select(0, at[:, :, sl].reshape(-1)).view(2, B, n, K, dv)
        red = (wt[:, :, sl, :, None] * rows).sum(3)             # what the store says comes next: for the model, and under the write's address
        d = v[:, sl] - red[1]
        e = beta[:, sl] * d                                     # how wrong it was, times how hard to write
        table.index_add_(0, at[1, :, sl].reshape(-1), (e[:, :, None, :] * coef[:, sl, :, None]).view(-1, dv))
        outs.append(red[0])
        if keep:
            rows_.append(rows)
            d_.append(d)
            e_.append(e)
    return torch.cat(outs, 1), (rows_, d_, e_)


class _Loop(torch.autograd.Function):
    """The store's loop as one node of the graph, its way back written by hand.

    Left to autograd, every chunk is some fifty small kernels and as many graph nodes, and the launches and the
    bookkeeping, not the arithmetic, were 44% of a training step (H200, Nov 12: 69 ms of 157). Here a chunk is about
    twenty launches and no nodes.

    The way back runs the chunks last to first with one table G, the gradient of the loss with respect to the rows as
    they stand at that moment: a chunk's write looks up in G what every later read asked of its rows, then the chunk's
    own reads (made before its write) are added to G. Only rows that were touched are ever moved."""

    @staticmethod
    def forward(ctx, wt, v, beta, coef, box, at, chunk):
        out, (rows, d, e) = _chunks(box[0], at, wt, v, beta, coef, chunk, True)
        ctx.box, ctx.at, ctx.chunk, ctx.rows, ctx.d, ctx.e = box, at, chunk, rows, d, e
        ctx.save_for_backward(wt, beta, coef)
        return out

    @staticmethod
    def backward(ctx, g_out):
        wt, beta, coef = ctx.saved_tensors
        at, chunk = ctx.at, ctx.chunk
        _, B, T, K = at.shape
        G = torch.zeros_like(ctx.box[0])
        dv = G.size(1)
        g_wt, g_v, g_beta, g_coef = [], [], [], []
        starts = list(range(0, T, chunk))
        for i in range(len(starts) - 1, -1, -1):
            sl = slice(starts[i], min(T, starts[i] + chunk))
            n = sl.stop - sl.start
            rows, d, e = ctx.rows[i], ctx.d[i], ctx.e[i]
            gu = G.index_select(0, at[1, :, sl].reshape(-1)).view(B, n, K, dv)      # what later reads asked of this chunk's writes
            ge = (gu * coef[:, sl, :, None]).sum(2)
            g_coef.append((gu * e[:, :, None, :]).sum(3))
            g_beta.append((ge * d).sum(-1, keepdim=True))
            gd = ge * beta[:, sl]
            g_v.append(gd)
            gr = torch.stack([g_out[:, sl], -gd])                                   # (2, B, n, dv): for the model's read, for the write's read
            g_wt.append((gr[:, :, :, None, :] * rows).sum(-1))
            G.index_add_(0, at[:, :, sl].reshape(-1), (gr[:, :, :, None, :] * wt[:, :, sl, :, None]).view(-1, dv))
        back = lambda parts, dim: torch.cat(parts[::-1], dim)
        return back(g_wt, 2), back(g_v, 1), back(g_beta, 1), back(g_coef, 1), None, None, None


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
        self.point = cfg.point if cfg.feed else 0
        self.share = cfg.share
        if self.point:
            # the top points at one of the `point` tokens before it; a write goes under that token's address.
            # It starts at the token before (the plain "what followed"), and learns to reach back.
            self.pq = nn.Linear(cfg.d, cfg.d_addr, bias=False)
            self.pk = nn.Linear(cfg.d, cfg.d_addr, bias=False)
            self.pb = nn.Parameter(torch.tensor([3.0] + [0.0] * (self.point - 1)))
        self.top = min(cfg.k, cfg.side)             # the k best rows lie among the k best of each half-key
        self.sparse = True                          # False: the whole table inside autograd (the reference; slow)

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
            K, ch = idx.size(-1), self.chunk
            live = torch.ones(B, T, K, device=h.device)             # the writes that share a row share it; the first token writes nothing
            live[:, 0] = 0
            nc = -(-T // ch)                                        # how many writes of a chunk land on each row, every chunk at once
            slot = (torch.arange(B, device=h.device)[:, None, None] * nc + torch.arange(T, device=h.device)[None, :, None] // ch) * self.rows + idx_w
            hits = torch.zeros(B * nc * self.rows, device=h.device).index_add_(0, slot.flatten(), live.flatten())
            coef = wn / hits[slot].clamp_(min=1.0)                  # (B, T, K): a write's weight on each of its rows
            if self.soft or not self.sparse:
                out, S = self._dense(S, idx, w, idx_w, w_w, coef, v, beta, p1 if self.soft else None, p2 if self.soft else None)
            else:
                out, S = self._sparse(S, idx, w, idx_w, w_w, coef, v, beta)
            return self.out(out).to(h.dtype), S, beta.squeeze(-1)

    # ---- feed: the read and the write are apart (the read where the store stands, the write after the top)
    def look(self, h, S):
        """A chunk at the store's layer: h (B, n, D) -> what the reads bring (B, n, D), and what the writes will need
        of these tokens: their rows (B, n, K), weights (B, n, K) and, for the pointer, their keys (B, n, d_addr)."""
        with torch.autocast(h.device.type, enabled=False):
            hn = norm(h.float())
            B, n, _ = hn.shape
            idx, w = self.address(*self.scores(hn))
            K = idx.size(-1)
            g = idx.reshape(B, n * K)[..., None].expand(-1, -1, self.dv)
            r = (w[..., None] * S.gather(1, g).view(B, n, K, self.dv)).sum(2)
            return self.out(r).to(h.dtype), (idx, w, self.pk(hn) if self.point else None)

    def file(self, top, S, new, old):
        """The writes of a chunk, after the top: top (B, n, D) the final vectors of the chunk's tokens; new / old:
        what look() gave for these tokens / for the tokens before them (old may be None at the start of a window: then
        the first token writes nothing). -> the store after the writes, the write strengths (B, n)."""
        with torch.autocast(top.device.type, enabled=False):
            top = top.float()
            B, n, _ = top.shape
            cat = (lambda a, b: b if a is None else torch.cat([a, b], 1))
            idx, w = (cat(old[i] if old is not None else None, new[i]) for i in (0, 1))
            c = idx.size(1) - n                                      # how many tokens stand before this chunk
            K = idx.size(-1)
            beta = 2 * torch.sigmoid(self.gate(top))
            t = torch.arange(n, device=top.device) + c              # each token's place among [old, new]
            if self.point:
                keys = cat(old[2] if old is not None else None, new[2])
                J = self.point
                pos = t[:, None] - torch.arange(1, J + 1, device=top.device)[None, :]      # (n, J): the J tokens before
                ok = pos >= 0
                pc = pos.clamp(min=0)
                sc = torch.einsum("bnd,bnjd->bnj", self.pq(top), keys[:, pc]) / math.sqrt(keys.size(-1)) + self.pb
                pi = sc.masked_fill(~ok, -1e9).softmax(-1) * ok.any(-1, keepdim=True)
                iw = idx[:, pc].reshape(B, n, J * K)
                ww = (pi[..., None] * w[:, pc]).reshape(B, n, J * K)
            else:
                ok = (t >= 1)[None, :, None].to(w.dtype)
                iw, ww = idx[:, (t - 1).clamp(min=0)], w[:, (t - 1).clamp(min=0)] * ok
            R = iw.size(-1)
            v = self.val(top)
            gi = iw.reshape(B, n * R)[..., None].expand(-1, -1, self.dv)
            was = (ww[..., None] * S.gather(1, gi).view(B, n, R, self.dv)).sum(2)          # what the store said would come
            flat = iw.reshape(B, n * R)
            # The chunk's writes land together, and tokens share rows (a word said twice, a row two tokens point at).
            # A write's step is sized by how much of its rows' weight is its own: with a (n, rows) the weights of the
            # chunk's writes, the step of write t is a_t / (a_t . sum_s a_s). Alone, that is the delta rule (what is
            # read back is exactly what was written); together, no reading moves by more than the errors ask
            # (a_t . sum_s a_s bounds the row sums of a a^T, so the joint step cannot overshoot). Sized write by
            # write, shared rows take every write in full, and what is written from the top and read again blows up.
            # share "strength": with a whole telling in a chunk, many tokens point a little at every row and most of
            # them have nothing to say (beta near 0); counted in full they cut the one real write to a fraction. So
            # write s counts g_s = min(beta_s, 1) on its rows, and a write counts itself in full: the step of write t
            # is a_t / (a_t . (sum_s g_s a_s + (1 - g_t) a_t)). Alone: the delta rule. All at full strength: "all".
            aw = ww.detach()
            if self.share == "strength":
                g = beta.detach().clamp(max=1.0)
                share = torch.zeros(B, self.rows, device=S.device).scatter_add_(1, flat, (g * aw).reshape(B, n * R)).gather(1, flat).view(B, n, R) + (1 - g) * aw
            else:
                share = torch.zeros(B, self.rows, device=S.device).scatter_add_(1, flat, aw.reshape(B, n * R)).gather(1, flat).view(B, n, R)
            coef = ww / (ww * share).sum(-1, keepdim=True).clamp_(min=1e-12)
            upd = (coef[..., None] * (beta * (v - was))[:, :, None, :]).reshape(B, n * R, self.dv)
            return S.scatter_add(1, gi, upd), beta.squeeze(-1) * (ww.sum(-1) > 0)

    def _sparse(self, S, idx, w, idx_w, w_w, coef, v, beta):
        """The chunks, touching only the rows they read and write (_chunks, _Loop). With gradients on, S is copied
        once and the copy written in place; without, S itself is written."""
        B, T, K = idx.shape
        grad = torch.is_grad_enabled()
        S = S.detach().clone() if grad else S.contiguous()
        table = S.view(B * self.rows, self.dv)
        off = torch.arange(B, device=S.device)[:, None, None] * self.rows
        at = torch.stack([idx + off, idx_w + off])                  # (2, B, T, K): the rows read for the model, the rows written
        wt = torch.stack([w, w_w])
        if grad and (wt.requires_grad or v.requires_grad):
            return _Loop.apply(wt, v, beta, coef, (table,), at, self.chunk), S
        return _chunks(table, at, wt, v, beta, coef, self.chunk, False)[0], S

    def _dense(self, S, idx, w, idx_w, w_w, coef, v, beta, p1=None, p2=None):
        """The same chunks with the whole table inside autograd: the reference the sparse way is checked against,
        and the only way for the soft read (which reads every row)."""
        B, T, K = idx.shape
        grad = torch.is_grad_enabled()
        outs = []
        for c in range(0, T, self.chunk):
            sl = slice(c, min(T, c + self.chunk))
            n = sl.stop - sl.start
            if p1 is not None:
                outs.append(self.read_all(p1[:, sl], p2[:, sl], S))
            else:
                go = idx[:, sl].reshape(B, n * K)[..., None].expand(-1, -1, self.dv)
                outs.append((w[:, sl, :, None] * S.gather(1, go).view(B, n, K, self.dv)).sum(2))
            gi = idx_w[:, sl].reshape(B, n * K)[..., None].expand(-1, -1, self.dv)
            r = (w_w[:, sl, :, None] * S.gather(1, gi).view(B, n, K, self.dv)).sum(2)       # what the store said would come
            e = beta[:, sl] * (v[:, sl] - r)                                                # how wrong it was
            upd = (e[:, :, None, :] * coef[:, sl, :, None]).reshape(B, n * K, self.dv)
            S = S.scatter_add(1, gi, upd) if grad else S.scatter_add_(1, gi, upd)
        return torch.cat(outs, 1), S


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
        self._fast = None                           # the compiled blocks (speed_up)
        self.reach = None                           # attention's reach for now, when not cfg.window (run.py: Keep-wide trained with it cut on some steps)
        self.muted = set()                          # stores whose reads are thrown away (keep/mute.py: which store carries what)
        self.loss_chunk = 4096                      # tokens per slice of the fused loss (oazy/lmloss.py); 1,024 cost 12 ms more a step

    # ------------------------------------------------------------ state
    def init_state(self, batch: int, device=None):
        device = device or self.emb.weight.device
        return {i: torch.zeros(batch, s.rows, s.dv, device=device) for i, s in self.store.items()}

    @staticmethod
    def detach(state):
        return {i: s.detach() for i, s in state.items()}

    # ------------------------------------------------------------ forward
    def hidden(self, x, state=None):
        """Token ids (B, T) -> final vectors (B, T, D), the stores after reading x, write strengths (B, T) per store."""
        B, T = x.shape
        if self.store and state is None:
            state = self.init_state(B, x.device)
        if self.cfg.feed and self.store:
            return self._fed(x, state)
        rot = (self.cos[:T], self.sin[:T])
        h = norm(self.emb(x))
        new, info = {}, {}
        run = self._fast if (self._fast is not None and self.training) else self._segment
        win = self.cfg.window if self.reach is None else self.reach
        lo = 0
        for hi in sorted({int(i) for i in self.store} | {len(self.blocks)}):       # the blocks up to a store, the store, and on
            if hi > lo:
                h = run(h, rot[0], rot[1], lo, hi, win)
            if str(hi) in self.store:
                r, new[str(hi)], info[str(hi)] = self.store[str(hi)](h, state[str(hi)])
                if hi not in self.muted:
                    h = h + r
            lo = hi
        return norm(h), new, info

    def _fed(self, x, state):
        """hidden() for feed: chunk after chunk through the whole stack. A chunk's reads see the stores as the chunks
        before left them; its writes are made from its final vectors and land when it ends."""
        B, T = x.shape
        ch, win = self.cfg.chunk, self.cfg.window
        if T > self.cos.size(0):                                      # longer than max_len in one go (think's exam, a telling to a chunk): more angles
            hd = self.cfg.d // self.cfg.n_head
            f = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=x.device).float() / hd))
            ang = torch.arange(T, device=x.device).float()[:, None] * f[None]
            self.cos, self.sin = ang.cos().to(self.cos.dtype), ang.sin().to(self.sin.dtype)
        S = dict(state)
        past = [None] * len(self.blocks)
        old = {i: None for i in self.store}
        back = max(1, self.cfg.point)                                 # how many tokens before a chunk its writes may be filed under
        outs, beta = [], {i: [] for i in self.store}
        for a in range(0, T, ch):
            b = min(T, a + ch)
            rot = (self.cos[a:b], self.sin[a:b])
            h = norm(self.emb(x[:, a:b]))
            new = {}
            for l, blk in enumerate(self.blocks):
                h, past[l] = _step(blk, h, rot, past[l], win)
                i = str(l + 1)
                if i in self.store:
                    r, new[i] = self.store[i].look(h, S[i])
                    if l + 1 not in self.muted:
                        h = h + r
            h = norm(h)
            outs.append(h)
            for i, st in self.store.items():
                S[i], bt = st.file(h, S[i], new[i], old[i])
                beta[i].append(bt)
                both = [n_ if o is None or n_ is None else torch.cat([o, n_], 1) for o, n_ in zip(old[i] or (None, None, None), new[i])]
                old[i] = tuple(None if t is None else t[:, -back:] for t in both)
        return torch.cat(outs, 1), S, {i: torch.cat(v, 1) for i, v in beta.items()}

    def _segment(self, h, cos, sin, lo: int, hi: int, window: int):
        for blk in self.blocks[lo:hi]:
            h = blk(h, (cos, sin), window)
        return h

    def speed_up(self, on=True):
        """torch.compile for the blocks between the stores, used while training (one shape; the exam's many shapes
        run plain). The stores stay as they are: their loop is short kernels on a few rows. speed_up(False) undoes it."""
        if on:                                      # one compiled graph per segment and per reach: more than torch's 8 at the big size
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 32)
        self._fast = torch.compile(self._segment) if on else None

    def loss(self, x, y, state=None, plain=False, asked=None, asked_weight=1.0):
        """asked: (B, T) bool, the targets that are part of a question block; they count asked_weight times in the loss.
        -> loss, the stores after x, info {"lm": the plain next-token loss, "asked": the loss on the asked targets,
           "write": mean write strength per store}."""
        from oazy.lmloss import head_loss, plain_loss
        h, new, beta = self.hidden(x, state)
        if plain or h.device.type != "cuda":
            loss = plain_loss(h, self.emb.weight, y)[0]
        else:
            loss = head_loss(h, self.emb.weight, y, chunk=self.loss_chunk)[0]
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
