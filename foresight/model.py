"""Plain-100F: Plain-100M with a training-only "foresight" head (Nov 8 2026, from the Special Lab).

Plain-100M stays exactly as it is: the same 10 layers of 768, the same recipe, the same mentor, the same text in the same
order. One small part is added for training only. At the end of each sentence (or every 16 words when no sentence ends),
the head reads the model's last hidden state and guesses which CONTENT words the next stretch of text will use: the rest
of the text up to the next sentence end, at most 32 words, as a bag (order does not matter). Filler is left out of the
bag: the 300 most common pieces of the training text, punctuation, spaces, and the document marks.

Why this cannot collapse the way Pegasus did: the answer key is written by the real text, not invented by the model, so
the model cannot make the task easy by making its own code trivial. The only way to lower this loss is for the hidden
state at a sentence end to carry what the next sentence is about.

The loss: a softmax over the content words, averaged over every (read point, word in its bag) pair ("which content word
comes next, in any order"). It is added to the next-word loss with weight 0.1 (warming up from 0 over the first 5% of the
run) and its gradient reaches the whole trunk. The head reads and writes through the tied word table, so it adds only a
norm and a 2-layer MLP (~2.4M numbers). It is never used for the held-out loss, HellaSwag, chat or the phone: those read the
trunk and the word table only, so size and speed are Plain-100M's (num_params() leaves the head out).

Still a NEXT-WORD model: the head is a side lesson during training; scoring and writing are next-word only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from memtrix.model import RMSNorm
from pegasus.baseline import Plain100, Plain100Config


@dataclass
class Plain100FConfig(Plain100Config):
    fore_weight: float = 0.1      # the foresight loss's weight next to the next-word loss
    fore_warmup: float = 0.05     # the weight grows from 0 to fore_weight over this share of the run
    fore_every: int = 16          # a read point every this many words when no sentence ends
    fore_span: int = 32           # the bag: words after the read point up to the next sentence end, at most this many
    fore_top: int = 300           # the most common pieces of the training text, left out of the bag (filler)
    fore_hidden: int = 1536       # the head's MLP width
    fore_max: int = 8192          # read points per pass at most (a random pick beyond that; a pass of 65k words has ~6k)


def sentence_end_table(enc, vocab: int) -> np.ndarray:
    """Pieces that end a sentence or a paragraph: their text ends in . ! ? (maybe followed by a quote or a bracket), or
    holds a line break."""
    out = np.zeros(vocab, dtype=bool)
    for t in range(min(vocab, enc.n_vocab)):
        if t == enc.eot_token:
            continue
        try:
            s = enc.decode_single_token_bytes(t).decode("utf-8", errors="ignore")
        except Exception:
            continue
        r = s.rstrip(" \"')]}”’»")
        if "\n" in s or r.endswith((".", "!", "?")):
            out[t] = True
    return out


def filler_table(enc, vocab: int, counts: np.ndarray, top: int) -> np.ndarray:
    """Pieces left out of the bag: special pieces, pieces with no letter or digit (punctuation, spaces), and the `top`
    most common pieces of the training text (the, of, and, ing, ...)."""
    out = np.zeros(vocab, dtype=bool)
    out[enc.n_vocab:] = True
    for t in range(min(vocab, enc.n_vocab)):
        try:
            s = enc.decode_single_token_bytes(t).decode("utf-8", errors="ignore")
        except Exception:
            out[t] = True
            continue
        if t == enc.eot_token or not any(ch.isalnum() for ch in s) or (s.startswith("<|") and s.endswith("|>")):
            out[t] = True
    c = np.asarray(counts[:vocab], dtype=np.float64).copy()
    c[out] = -1                                    # rank the rest only
    if top > 0:
        out[np.argsort(-c, kind="stable")[:top]] = True
    return out


def token_counts(paths, vocab: int, per_source: int = 4_000_000) -> np.ndarray:
    """How often each piece appears in the first `per_source` pieces of each training file."""
    c = np.zeros(vocab, dtype=np.int64)
    for p in paths:
        d = np.memmap(p, dtype=np.uint16, mode="r")
        c += np.bincount(np.asarray(d[:per_source], dtype=np.int64), minlength=vocab)[:vocab]
    return c


class ForesightHead(nn.Module):
    """norm -> Linear(d, hidden) -> GELU -> Linear(hidden, d); the scores come from the tied word table."""

    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.norm = RMSNorm(d)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, h):
        return self.down(F.gelu(self.up(self.norm(h))))


class Foresight:
    """The next-sentence lesson, for any model with `cfg` (the fore_* settings), `embed` (the tied word table) and a
    token-vector trunk: the tables, where it reads, the bags, and the loss. Plain-100F and Pegasus Alpha Star use it."""

    def init_foresight(self, cfg):
        self.fore = ForesightHead(cfg.d_model, cfg.fore_hidden)
        self.fore.apply(self._init)
        nn.init.normal_(self.fore.down.weight, std=0.02 / math.sqrt(2))
        V = cfg.vocab_size
        # filled once from the training text (prepare_foresight) and kept in the checkpoint, so a resume reads the same
        self.register_buffer("fore_end", torch.zeros(V, dtype=torch.bool))
        self.register_buffer("fore_skip", torch.ones(V, dtype=torch.bool))
        self.register_buffer("fore_prior", torch.zeros(V))          # log share of each content word in the bags (a blind guess)
        self.register_buffer("fore_ready", torch.zeros((), dtype=torch.long))
        self._fore_w = cfg.fore_weight
        self._ready = False

    def head_params(self):
        return sum(p.numel() for p in self.fore.parameters())

    # ---------------------------------------------------------------- set-up
    def foresight_ready(self):
        """Are the tables filled (here, or by a checkpoint just loaded)? Remembered, so a pass never waits on the GPU."""
        self._ready = bool(self.fore_ready.item())
        return self._ready

    def prepare_foresight(self, enc, counts):
        """Which pieces end a sentence, which are filler, from the tokenizer and the training text's counts."""
        V = self.cfg.vocab_size
        end = sentence_end_table(enc, V)
        skip = filler_table(enc, V, counts, self.cfg.fore_top)
        c = np.where(skip, 0.0, np.asarray(counts[:V], dtype=np.float64) + 0.5)
        prior = np.zeros(V)
        prior[~skip] = np.log(c[~skip] / c.sum())
        dev = self.fore_end.device
        self.fore_end.copy_(torch.from_numpy(end).to(dev))
        self.fore_skip.copy_(torch.from_numpy(skip).to(dev))
        self.fore_prior.copy_(torch.from_numpy(prior).float().to(dev))
        self.fore_ready.fill_(1)
        self._ready = True
        return {"sentence_ends": int(end.sum()), "filler": int(skip.sum()), "content": int((~skip).sum())}

    def set_progress(self, progress: float):
        c = self.cfg
        self._fore_w = c.fore_weight * (min(1.0, progress / c.fore_warmup) if c.fore_warmup > 0 else 1.0)

    # ---------------------------------------------------------------- the lesson
    def read_points(self, idx):
        """(B, T) bool: where the head reads. After a sentence end, or after fore_every words with none (counted from the
        last end or document start); never on the document mark itself."""
        from oazy.tok import eot_id
        B, T = idx.shape
        eot = idx == eot_id(self.cfg.tokenizer)
        end = self.fore_end[idx]
        pos = torch.arange(T, device=idx.device).expand(B, T)
        last = torch.where(end | eot, pos, torch.full_like(pos, -1)).cummax(1).values
        gap = pos - last
        return (end | ((gap > 0) & (gap % self.cfg.fore_every == 0))) & ~eot

    def bags(self, idx, targets, b, t):
        """For read points (b, t): the bag's pieces (N, span), each content piece once, -1 elsewhere."""
        from oazy.tok import eot_id
        c = self.cfg
        e = eot_id(c.tokenizer)
        B, T = idx.shape
        seq = torch.cat([idx, targets[:, -1:], torch.full((B, c.fore_span), e, dtype=idx.dtype, device=idx.device)], 1)
        off = torch.arange(1, c.fore_span + 1, device=idx.device)
        tk = seq[b[:, None], t[:, None] + off]                      # (N, span): the words after the read point
        stop = self.fore_end[tk] | (tk == e)
        before = stop.long().cumsum(1) - stop.long()                # stops before this word
        ok = (before == 0) & (tk != e) & ~self.fore_skip[tk]       # up to and with the next sentence end; content only
        tk = torch.where(ok, tk, torch.full_like(tk, -1))
        tk = tk.sort(1, descending=True).values                     # each piece once
        dup = torch.zeros_like(ok)
        dup[:, 1:] = tk[:, 1:] == tk[:, :-1]
        return torch.where(dup, torch.full_like(tk, -1), tk)

    def pick(self, idx):
        """At most fore_max read points, at random among them (a fixed number, so no waiting on the GPU to count)."""
        B, T = idx.shape
        read = self.read_points(idx).reshape(-1)
        n = min(self.cfg.fore_max, B * T)
        score = torch.rand(B * T, device=idx.device).masked_fill(~read, -1.0)
        top, at = score.topk(n)
        return at // T, at % T, top >= 0

    def fore_points(self, idx, targets):
        """The read points of a pass and their bags: (b, t, bag, mask), shared by every state the head reads."""
        b, t, live = self.pick(idx)
        bag = self.bags(idx, targets, b, t)
        return b, t, bag, (bag >= 0) & live[:, None]

    def fore_nll(self, h, pts, prior=False):
        """The head's loss (nats per bag word) on the token vectors h at the read points pts. prior=True: also the
        blind guess's loss (content words' shares)."""
        b, t, bag, mask = pts
        z = F.linear(self.fore(h[b, t]), self.embed.weight).float()
        z = z.masked_fill(self.fore_skip, float("-inf"))             # a softmax over content words only
        lse = z.logsumexp(-1, keepdim=True)
        got = z.gather(1, bag.clamp(min=0))
        n = mask.sum().clamp(min=1)
        loss = torch.where(mask, lse - got, torch.zeros_like(got)).sum() / n    # (not x mask: a filler slot is inf)
        if not prior:
            return loss, n
        blind = -(self.fore_prior[bag.clamp(min=0)] * mask).sum() / n
        return loss, n, blind

    def foresight_loss(self, h, idx, targets, prior=False):
        """The head's loss (nats per bag word). prior=True: also the blind guess's loss (content words' shares)."""
        return self.fore_nll(h, self.fore_points(idx, targets), prior)


class Plain100F(Foresight, Plain100):
    """Plain-100M + the training-only foresight head."""

    def __init__(self, cfg: Plain100FConfig):
        super().__init__(cfg)                       # the trunk first, so its starting weights are Plain-100M's
        self.init_foresight(cfg)

    # ---------------------------------------------------------------- sizes: the head is not part of the model
    def num_params(self, non_embedding=False):
        return super().num_params(non_embedding) - self.head_params()

    def layout(self):
        c = self.cfg
        return (super().layout() + f" + foresight head for training only ({self.head_params() / 1e6:.1f}M, next sentence's "
                f"content words, weight {c.fore_weight:g})")

    def losses(self, idx, targets, mentor=None, kd_weight=0.0):
        """Plain-100M's lesson (Bee.losses, unchanged) + the foresight loss while training."""
        resets, mask, segs = self.doc_info(idx)
        h = (self._compiled or self._trunk)(idx, resets, self.attn_mask(idx, resets, mask), segs)
        th = tW = None
        if mentor is not None and kd_weight > 0:
            th, tW = mentor.hidden(idx, mask)
        if self.fused_loss:
            from oazy.lmloss import head_loss
            sel = getattr(self, "token_select", None) if self.training and torch.is_grad_enabled() else None   # oazy/select.py
            loss, ce, kd = head_loss(h, self.embed.weight, targets, self.cfg.softcap, th, tW, kd_weight, self.loss_chunk, sel)
        else:
            from oazy.lmloss import plain_loss
            loss, ce, kd = plain_loss(h, self.embed.weight, targets, self.cfg.softcap, th, tW, kd_weight)
        stats = {"ce": ce}
        if th is not None:
            stats["kd"] = kd
        if self.training and torch.is_grad_enabled() and self._ready:
            fl, n = self.foresight_loss(h, idx, targets)
            if self._fore_w > 0:
                loss = loss + self._fore_w * fl
            stats["fore"] = fl.detach()
        return loss, stats

    @torch.no_grad()
    def foresight_report(self, windows, device, batch=16):
        """On held-out windows: the head's loss and the blind guess's (each content word by how common it is)."""
        if not self.foresight_ready():
            return None
        was = self.training
        self.eval()
        xy = torch.as_tensor(windows, device=device)
        tot = blind = n_all = 0.0
        for i in range(0, len(xy), batch):
            x, y = xy[i:i + batch, :-1], xy[i:i + batch, 1:]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                resets, mask, segs = self.doc_info(x)
                h = self._trunk(x, resets, self.attn_mask(x, resets, mask), segs)
                fl, n, bl = self.foresight_loss(h, x, y, prior=True)
            tot, blind, n_all = tot + float(fl) * float(n), blind + float(bl) * float(n), n_all + float(n)
        self.train(was)
        return {"fore_val": tot / max(n_all, 1), "fore_blind": blind / max(n_all, 1), "fore_words": int(n_all)}
