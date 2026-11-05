"""Fern3: thoughts unfold coarse to fine, and old passages are remembered by their gist.

Fern3 is NOT a next-word model. It writes like Fern (fern/model.py): 16 words at a time, in 5
rounds, coarse to fine. What is new is how it remembers what it has read. Fern and Fern2 read the
past with Engram's fixed-size memories, which blur old details. Fern3 remembers the way you do:

    every finished passage is folded up into one GIST     (Fern's unfolding run backwards:
                                                            16 words -> 8 -> 4 -> 2 -> 1, pair by pair)
    the words themselves stay on a SHELF, word for word   (text is cheap to keep)
    the model THINKS with the gists only                   (one per 16 words: a 4,096-word text is 256 gists)
    and RECALLS the 3 old passages whose gists match best, reading their words in full again

    the past:  [ p1 ][ p2 ][ p3 ] ... [ p250 ][ p251 ][ p252 ][ p253 ][ p254 ][ p255 ]  [ writing p256 ]
    gists:       g1    g2    g3   ...   g250    g251    g252    g253  |  g254    g255
    thinking:    ───────── a small model over the gists, in order ────────────►  "where are we?"
    recall:            ▲                       ▲            ▲             (top 3 by gist match)
    in full:   the 3 recalled passages + the last 2 passages (48 + 32 words), plus the gists' summary

The parts, all built from the blocks Fern already has (fern/model.py UnfoldBlock):
  * FOLDER: reads each passage word by word, seeing its own earlier words and the passage before
    (a 32-word window). Its word states are what is kept on the shelf. It also guesses each next
    word, which is where the GPT-2 medium mentor teaches (as in Fern2).
  * MERGE: folds a passage's 16 word states into one gist, two at a time, with one shared rule.
  * THINKER: a small model that reads the gists in order (a passage only sees earlier gists).
  * RECALL: the passage being written asks a question (from the thinker's latest thought); every
    old gist answers with how well it matches. All gists get a soft share of a "blended gist" read
    (so every gist learns whether it would have helped), and the 3 best are read in full. Learning
    which to recall is the hard part (a yes/no choice): the soft share gives the learning signal,
    and the 3 recalled passages carry their share in the backward pass (straight-through), so a
    useful recall raises its gist's score. Close to "landmark attention" (Mohtashami & Jaggi 2023).
  * UNFOLDER: Fern's, unchanged, glancing at the last 2 passages, the 3 recalled ones and 2 gist
    summaries (82 items in all) instead of Fern's reader.

What a word sees when it is decided: words of earlier passages only through the folder (last 2
passages), the recall (passages at least 3 back) and the gists (all earlier passages), plus the
words of its own passage decided in earlier rounds. Checked word by word in fern3/tests.py.

Scoring needs passages to start where the text to score starts, so a text is padded in front with
end-of-text marks (the same mark that separates documents in training).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from engram.model import RMSNorm
from fern.model import UnfoldBlock
from fern2.model import Fern2, Fern2Config
from oazy.data import EOT


@dataclass
class Fern3Config(Fern2Config):
    n_layer: int = 5                    # FOLDER blocks when fully grown
    unfold_layers: int = 3
    think_layers: int = 2               # the thinker over gists (does not grow)
    recall: int = 3                     # old passages read again in full for every passage written
    recall_warm: int = 8                # ... and early in training more of them, so the right one is often among
    recall_warm_until: float = 0.1      #     them and recall can learn what helps (fraction of training tokens)
    hindsight_weight: float = 0.1       # the hindsight lesson for recall (0 = off), see _core3
    max_len: int = 4096                 # long windows: recall only matters when there is a lot to forget
    grow: str = "2:0,4:0.1,5:0.25"      # folder depth by fraction of training tokens
    unfold_grow: str = "1:0,2:0.1,3:0.25"


def _rope(places, hd, base):
    freq = 1.0 / (base ** (torch.arange(hd // 2, device=places.device).float() / (hd // 2)))
    ang = places.float()[:, None] * freq[None]
    return ang.cos()[None], ang.sin()[None]


class Merge(nn.Module):
    """Two neighbours -> one: the rule that folds a passage into its gist (16 -> 8 -> 4 -> 2 -> 1)."""

    def __init__(self, D, hidden):
        super().__init__()
        self.pair = nn.Linear(2 * D, D, bias=False)
        self.n = RMSNorm(D)
        self.up = nn.Linear(D, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, D, bias=False)

    def forward(self, x):                      # (..., n, D) -> (..., n / 2, D)
        *lead, n, D = x.shape
        y = self.pair(x.reshape(*lead, n // 2, 2 * D))
        a, b = self.up(self.n(y)).chunk(2, dim=-1)
        return y + self.down(F.silu(a) * b)


class Fern3(Fern2):
    def _reader_block(self, cfg):
        return UnfoldBlock(cfg)                # the folder is made of Fern's own blocks

    def __init__(self, cfg: Fern3Config):
        super().__init__(cfg)                  # Fern's unfolder + Fern2's growing / mentor plumbing
        D, P = cfg.d_model, cfg.passage
        R = max(cfg.recall, cfg.recall_warm)
        del self.past_where                    # (Fern's reader-only part)
        floors = int(math.log2(P))
        self.think = nn.ModuleList(UnfoldBlock(cfg) for _ in range(cfg.think_layers))
        self.merge = Merge(D, cfg.unfold_hidden)
        self.gist_norm = RMSNorm(D)
        self.think_norm = RMSNorm(D)
        self.word_norm = RMSNorm(D)
        self.recall_q = nn.Linear(D, D, bias=False)
        self.recall_k = nn.Linear(D, D, bias=False)
        self.recall_v = nn.Linear(D, D, bias=False)
        self.floor_emb = nn.Parameter(torch.zeros(floors, D))
        self.fold_where = nn.Parameter(torch.zeros(P, D))
        self.local_where = nn.Parameter(torch.zeros(2 * P, D))
        self.recall_where = nn.Parameter(torch.zeros(P, D))
        self.recall_rank = nn.Parameter(torch.zeros(R, D))
        self.gist_type = nn.Parameter(torch.zeros(2, D))
        depth = cfg.n_layer + cfg.unfold_layers + cfg.think_layers
        for mod in (self.think, self.merge, self.recall_q, self.recall_k, self.recall_v):
            mod.apply(self._init)
        for n, p in list(self.think.named_parameters()) + list(self.merge.named_parameters()):
            if n.endswith("glance.out.weight") or n.endswith("down.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * depth))
        with torch.no_grad():
            for t in (self.floor_emb, self.fold_where, self.local_where, self.recall_where, self.recall_rank,
                      self.gist_type):
                t.normal_(0, 0.02)

        self.R = None
        self.register_buffer("recall_now", torch.tensor(0))       # saved in checkpoints
        self.set_recall(cfg.recall_warm if cfg.recall_warm > cfg.recall and cfg.recall_warm_until > 0 else cfg.recall)
        fc, fs = _rope(torch.arange(-P + 1, P + 1), D // cfg.unfold_heads, 64.0)                     # folder: last passage, then own
        self.register_buffer("fold_cos", fc, persistent=False)
        self.register_buffer("fold_sin", fs, persistent=False)
        fm = torch.cat([torch.ones(P, P, dtype=torch.bool), torch.ones(P, P, dtype=torch.bool).tril()], 1)
        self.register_buffer("fold_mask", fm, persistent=False)                    # (P, 2P): no peeking ahead
        self._compiled = None

    def set_recall(self, R):
        """How many old passages are read in full: the unfolder's glance has 2P + R*P + 2 items."""
        if R == self.R:
            return False
        P, D = self.cfg.passage, self.cfg.d_model
        self.R, self.Pp = R, 2 * P + R * P + 2
        self.recall_now.fill_(R)
        S = len(self.slot_pos)
        dev = self.slot_pos.device
        rr = self.slot_round
        self.allowed = torch.cat([torch.ones(S, self.Pp, dtype=torch.bool, device=dev), rr[None, :] <= rr[:, None]], 1)
        places = torch.cat([torch.arange(-2 * P + 1, 1, device=dev), torch.zeros(R * P + 2, dtype=torch.long, device=dev),
                            self.slot_pos])
        self.rope_cos, self.rope_sin = _rope(places, D // self.cfg.unfold_heads, 64.0)
        return True

    def grow_to(self, progress: float) -> bool:
        """Grow (Fern2's plan) and, after the warm-up, recall only the top `recall` passages."""
        c = self.cfg
        changed = self.set_recall(c.recall_warm if (c.recall_warm > c.recall and progress < c.recall_warm_until) else c.recall)
        grew = super().grow_to(progress)
        if changed and not grew and self._compiled is not None:
            self.enable_compile()
        return grew

    def load_state_dict(self, sd, strict=True, assign=False):
        out = super().load_state_dict(sd, strict=strict, assign=assign)
        if self.set_recall(int(self.recall_now)) and self._compiled is not None:
            self.enable_compile()
        return out

    def memory_numbers(self):
        return None                           # no fixed-size memory: the shelf keeps every word

    def depth_text(self):
        return f"{len(self.blocks)} folder + {len(self.unfold)} unfolder blocks"

    # ------------------------------------------------------------------ the whole model, on aligned passages
    def _core3(self, X, exist):
        """X (B, L) tokens with L a multiple of P, passages at [jP, (j+1)P); exist (B, L): word is real.
        Returns the unfolder's thought for every word (B, L, D) and the folder's (B, L, D), plus what
        was recalled (B, J, R) and its share."""
        cfg = self.cfg
        B, L = X.shape
        P, R, D = cfg.passage, self.R, cfg.d_model
        J = L // P
        dev = X.device
        E = self.embed(X)
        jj = torch.arange(J, device=dev)

        # FOLDER: each passage word by word, seeing its own earlier words and the passage before
        x = E.view(B, J, P, D) + self.fold_where.to(E.dtype)
        fmask = self.fold_mask[None].expand(J, P, 2 * P).clone()
        fmask[0, :, :P] = False                                      # the first passage has none before it
        fmask = fmask[None].expand(B, J, P, 2 * P).reshape(B * J, 1, P, 2 * P)
        frope = (self.fold_cos, self.fold_sin) if cfg.rope else None
        for blk in self.blocks:
            past = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], 1)
            x = blk(x.reshape(B * J, P, D), past.reshape(B * J, P, D), fmask, frope).view(B, J, P, D)
        Fw = self.norm(x)                                            # (B, J, P, D): the folder's understanding
        shelf = Fw + self.word_norm(E).view(B, J, P, D)              # kept on the shelf: understanding + the word itself

        # MERGE: fold each passage into its gist, two neighbours at a time
        g = Fw
        for f in range(self.floor_emb.shape[0]):
            g = self.merge(g + self.floor_emb[f].to(g.dtype))
        g = self.gist_norm(g[:, :, 0])                               # (B, J, D)

        # THINKER: the gists in order
        t = g
        tmask = torch.ones(J, J, dtype=torch.bool, device=dev).tril()[None, None]
        trope = _rope(jj, D // cfg.unfold_heads, 1024.0) if cfg.rope else None
        for blk in self.think:
            t = blk(t, t[:, :0], tmask, trope)
        T = self.think_norm(t)                                       # T[:, k]: understanding up to passage k

        # RECALL: passage j asks with T[j-1]; old passages k <= j-3 answer with their gists
        q = torch.cat([torch.zeros_like(T[:, :1]), self.recall_q(T)[:, :-1]], 1)
        k, v = self.recall_k(T), self.recall_v(T)
        cand = jj[None, :] <= jj[:, None] - 3                         # (J, J)
        has = cand.any(-1)
        score = (q @ k.transpose(1, 2)).float() / math.sqrt(D)
        score = torch.where(cand[None], score, torch.full_like(score, -1e4))
        share = torch.softmax(score, -1) * has[None, :, None]       # every old gist's soft share
        # HINDSIGHT: once passage j is written, the old passage most like it is the one that would have
        # been worth recalling. Its gist is known after the fact, so recall learns to find it sooner:
        # the shares are pulled towards "how alike each old gist is to what was actually written".
        # "alike" = how many of the same words two passages share (their mean word vectors, compared)
        bag = F.normalize(self.word_norm(E.detach()).float().view(B, J, P, D).mean(2), dim=-1)
        alike = torch.where(cand[None], bag @ bag.transpose(1, 2) / 0.05, torch.full_like(score, -1e4))
        hind = (torch.softmax(alike, -1) * (torch.log_softmax(alike, -1) - torch.log_softmax(score, -1))).sum(-1)
        hind = (hind * has[None]).sum() / (has.sum() * B).clamp_min(1)
        blend = (share.to(v.dtype) @ v)                              # (B, J, D) the blended gist
        kr = min(R, J)
        top, idx = score.topk(kr, -1)                                # the 3 best-matching old passages
        rvalid = top > -1e3
        if kr < R:
            idx = torch.cat([idx, idx.new_zeros(B, J, R - kr)], -1)
            rvalid = torch.cat([rvalid, rvalid.new_zeros(B, J, R - kr)], -1)
            top = torch.cat([top, top.new_full((B, J, R - kr), -1e4)], -1)
        # how much the unfolder listens to each recalled passage: its share among the recalled ones.
        # This is in the forward pass (the unfolder's glance at passage k is weighted by its share), so a
        # recall that helped raises its gist's score and one that did not lowers it (landmark attention)
        lgate = torch.log_softmax(torch.where(rvalid, top, torch.full_like(top, -1e4)), -1)
        if self.R > self.cfg.recall:
            # warm-up (more passages recalled): listen to all of them equally, so the unfolder first learns
            # to use recalled words at all; the shares still learn from what helped (straight-through)
            lgate = lgate - lgate.detach()
        rec = shelf.reshape(B, J, P * D).gather(1, idx.reshape(B, J * R, 1).expand(-1, -1, P * D)).view(B, J, R, P, D)
        rec = rec + self.recall_where.to(rec.dtype) + self.recall_rank[:R, None].to(rec.dtype)

        # what the unfolder glances at for passage j
        z = torch.zeros_like(shelf[:, :1])
        F1 = torch.cat([z, shelf[:, :-1]], 1)
        F2 = torch.cat([z, z, shelf[:, :-2]], 1)[:, :J]
        local = torch.cat([F2, F1], 2) + self.local_where.to(Fw.dtype)
        tprev = torch.cat([torch.zeros_like(T[:, :1]), T[:, :-1]], 1)
        gists = torch.stack([tprev + self.gist_type[0].to(T.dtype), blend + self.gist_type[1].to(T.dtype)], 2)
        past = torch.cat([local, rec.reshape(B, J, R * P, D), gists.to(local.dtype)], 2)    # (B, J, Pp, D)
        part = lambda m, n: m[None, :, None].expand(B, J, n)      # noqa: E731  (per passage -> per item)
        pvalid = torch.cat([part(jj >= 2, P), part(jj >= 1, P), rvalid[..., None].expand(B, J, R, P).reshape(B, J, R * P),
                            part(jj >= 1, 1), part(has, 1)], 2)

        # UNFOLDER: Fern's, round by round
        Xp, ex = X.view(B, J, P), exist.view(B, J, P)
        tok = Xp[:, :, self.slot_pos - 1]
        sex = ex[:, :, self.slot_pos - 1]
        word = self.embed(tok)
        show = (self.slot_known[None, None] & sex)[..., None]
        y = torch.where(show, word, self.blank.to(word.dtype))
        y = y + self.where[self.slot_pos - 1].to(y.dtype) + self.round_emb[self.slot_round].to(y.dtype)
        S = y.shape[2]
        mask = self.allowed[None, None] & torch.cat([pvalid, sex], -1)[:, :, None, :]
        mask = mask | F.pad(torch.eye(S, dtype=torch.bool, device=dev), (self.Pp, 0))[None, None]
        bias = F.pad(lgate[..., None].expand(B, J, R, P).reshape(B, J, R * P), (2 * P, 2 + S))     # (B, J, Pp + S)
        mask = torch.where(mask, bias[:, :, None, :], torch.full_like(bias[:, :, None, :], float("-inf")))
        G = B * J
        y, past, mask = y.reshape(G, S, D), past.reshape(G, self.Pp, D), mask.reshape(G, 1, S, self.Pp + S)
        rope = (self.rope_cos, self.rope_sin) if cfg.rope else None
        for blk in self.unfold:
            y = blk(y, past, mask, rope)
        h = self.out_norm(y.view(B, J, S, D)[:, :, self.target_slot])   # (B, J, P, D) in word order
        return h.reshape(B, L, D), Fw.reshape(B, L, D), idx, rvalid, lgate.exp(), hind

    def enable_compile(self):
        self._compiled = torch.compile(self._core3)

    def _pad(self, seq, starts, ends):
        """Pad each text in front with end-of-text marks so its first scored word starts a passage."""
        B, N1 = seq.shape
        P = self.cfg.passage
        pad = (-(starts + 1)) % P
        L = -(-(N1 + int(pad.max())) // P) * P
        o = torch.arange(L, device=seq.device)[None] - pad[:, None]
        X = torch.where((o >= 0) & (o < N1), seq.gather(1, o.clamp(0, N1 - 1)),
                        torch.full_like(o, EOT if self.cfg.vocab_size > EOT else 0))
        return X, o <= ends[:, None], pad

    def _run(self, seq, starts, ends, compiled=False):
        B, N1 = seq.shape
        N, D = N1 - 1, self.cfg.d_model
        X, exist, pad = self._pad(seq, starts, ends)
        fn = self._compiled if (compiled and self._compiled is not None) else self._core3
        h, Fw, idx, rvalid, gate, hind = fn(X, exist)
        self._hind = hind
        at = (torch.arange(N, device=seq.device)[None] + pad[:, None])        # padded place of word t
        pick = lambda a, sh: a.gather(1, (at + sh)[..., None].expand(-1, -1, D))   # noqa: E731
        return pick(h, 1), pick(Fw, 0), (idx, rvalid, pad)

    def seq_logits(self, seq, starts=None, ends=None, compiled=False):
        """Same contract as Fern.seq_logits: logits[:, t] is the chance of seq[:, t+1] in its round."""
        B, N1 = seq.shape
        dev = seq.device
        starts = torch.zeros(B, dtype=torch.long, device=dev) if starts is None else starts
        ends = torch.full((B,), N1 - 1, dtype=torch.long, device=dev) if ends is None else ends
        return F.linear(self._run(seq, starts, ends, compiled)[0], self.embed.weight)

    def reader_logits(self, idx):
        """The folder's own next-word guesses (B, T, vocab)."""
        B, T = idx.shape
        seq = torch.cat([idx, idx[:, -1:]], 1)
        st = torch.full((B,), -1, dtype=torch.long, device=idx.device)
        _, Fs, _ = self._run(seq, st, torch.full((B,), T, dtype=torch.long, device=idx.device))
        return F.linear(self.reader_norm(self.reader_proj(Fs)), self.embed.weight)

    @torch.no_grad()
    def recalled(self, ids):
        """For a text: which earlier passages each passage recalled (for the chat's memory view)."""
        seq = torch.tensor([list(ids) + [ids[-1]]], device=self.embed.weight.device)
        st = torch.tensor([len(ids) - 1], device=seq.device)
        _, _, (idx, rvalid, pad) = self._run(seq, st, st)
        return idx[0].tolist(), rvalid[0].tolist(), int(pad[0])

    def forward(self, idx, targets=None):
        seq = torch.cat([idx[:, :1], targets], 1) if targets is not None else torch.cat([idx, idx[:, -1:]], 1)
        B, N = idx.shape
        dev = idx.device
        r0 = int(torch.randint(0, self.cfg.passage, (1,))) if (self.training and self.cfg.shift) else 0
        starts = torch.full((B,), r0, dtype=torch.long, device=dev)
        ends = torch.full((B,), N, dtype=torch.long, device=dev)
        hs, Fs, _ = self._run(seq, starts, ends, compiled=True)
        logits = F.linear(hs, self.embed.weight)
        if targets is None:
            return logits, None, {}
        tgt = torch.where(self.rounds_of(starts, N) >= 0, targets, torch.full_like(targets, -1))
        ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-1)
        hw = self.cfg.hindsight_weight
        if not self.training:
            return logits, ce, {"ce": ce.detach()}
        if self.cfg.reader_weight <= 0:
            return logits, ce + hw * self._hind, {"ce": ce.detach(), "recall_hindsight": self._hind.detach(),
                                                  "_hind": self._hind}
        rlog = F.linear(self.reader_norm(self.reader_proj(Fs)), self.embed.weight)   # word t -> guess of word t + 1
        rce = F.cross_entropy(rlog.float().reshape(-1, rlog.size(-1)), targets.reshape(-1))
        stats = {"ce": ce.detach(), "reader_ce": rce.detach(), "_ce": ce, "_reader_ce": rce, "kd_logits": rlog,
                 "recall_hindsight": self._hind.detach(), "_hind": self._hind}
        return logits, ce + self.cfg.reader_weight * rce + hw * self._hind, stats

    def mix_mentor(self, loss, stats, kd, m):
        return super().mix_mentor(loss, stats, kd, m) + self.cfg.hindsight_weight * stats["_hind"]
