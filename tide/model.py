"""Tide: a language model that drafts its next thoughts before they arrive.

It is still a next-word model: what it is scored on (loss, HellaSwag) is its guess of the very next
word, exactly like GPT-2. What is new is *how* that guess is made.

Picture a conveyor belt of K = 8 draft "thoughts" in front of the present word:

      read so far ──►  now │ slot 1   slot 2   slot 3  ...  slot 8
                           │ (next)   (+2)     (+3)         (+8, brand new, blank)

Every time a real word is read:
  1. the belt moves one place towards now: what was the draft of "3 words ahead" is now the draft
     of "2 words ahead", and a blank thought joins at the far end;
  2. every slot gets one clean-up step (the refiner), from the far end to the front. The clean-up
     sees three things: the slot's own draft, what has been read so far (Engram's memory, below),
     and the drafts queued behind it (the further future, already cleaned this step);
  3. the front slot is the next-word guess.

So each thought is first sketched 8 words ahead, cleaned 8 times on its way in, and when it arrives
it has weighed both the past and the thoughts planned after it. That is "diffusion", spread over
time: one clean-up per slot per word, nothing iterates until it settles (Basin's problem).

Every slot is trained against the real word it is drafting (slot d at word t against word t+d), so
the drafts cannot drift into meaningless states: they must become guesses of the future.

Context ("what it remembers"), all fixed-size whatever the text length, no attention:
  long past   Engram's 12 x 12 learn-while-reading memories (engram/model.py), the backbone;
  near past   the same memories' fast-fading heads, plus a 4-word convolution;
  near future the belt of 8 drafts.

Training is parallel over the whole text: slot 8 is computed for all positions at once, then slot
7 (which needs slot 8 from one word earlier and slot 8 now), ..., then slot 1. Reading word by word
gives exactly the same numbers (tide/tests.py checks it).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from engram.model import Engram, EngramConfig, RMSNorm, sample_logits  # noqa: F401  (sample_logits re-exported)


@dataclass
class TideConfig(EngramConfig):
    lookahead: int = 8           # K: how many words ahead the belt drafts
    refine_hidden: int = 1536    # width of the clean-up network (one network, shared by all slots)
    draft_weight: float = 0.1    # how much the far drafts' own guesses count in training (0 = free, wordless drafts)
    draft_frac: float = 0.125    # fraction of positions where far drafts are scored (keeps it cheap)
    skip: bool = True            # a draft is read out on top of "what the memories say now"


class Refiner(nn.Module):
    """One clean-up step for one slot: own draft + what was read + the drafts queued behind it."""

    def __init__(self, cfg: TideConfig):
        super().__init__()
        D, Hd, K = cfg.d_model, cfg.refine_hidden, cfg.lookahead
        self.n_ctx = RMSNorm(D)
        self.n_ahead = RMSNorm(D)
        self.ctx = nn.Linear(D, D, bias=False)          # "what has been read so far"
        self.ahead = nn.Linear(D, D, bias=False)        # "the thoughts planned after mine"
        self.where = nn.Parameter(torch.zeros(K, D))    # "I am d words ahead" (like a diffusion step)
        self.n = RMSNorm(D)
        self.up = nn.Linear(D, 2 * Hd, bias=False)
        self.down = nn.Linear(Hd, D, bias=False)

    def forward(self, own, ctx, ahead, d):
        u = own + self.ctx(self.n_ctx(ctx)) + self.where[d]
        if ahead is not None:
            u = u + self.ahead(self.n_ahead(ahead))
        a, b = self.up(self.n(u)).chunk(2, dim=-1)
        return u + self.down(F.silu(a) * b)


class Tide(Engram):
    def __init__(self, cfg: TideConfig):
        super().__init__(cfg)
        K, D = cfg.lookahead, cfg.d_model
        self.refiner = Refiner(cfg)
        self.refiner.apply(self._init)
        self.blank = nn.Parameter(torch.randn(D) * 0.02)          # the new thought at the far end
        # how much slot d listens to each slot behind it (j > d); starts as a plain average
        mix = torch.zeros(K, K)
        for d in range(K):
            if d + 1 < K:
                mix[d, d + 1:] = 1.0 / (K - d - 1)
        self.mix = nn.Parameter(mix)
        self.register_buffer("mix_mask", torch.triu(torch.ones(K, K), diagonal=1), persistent=False)
        self.out_norm = RMSNorm(D)
        nn.init.normal_(self.refiner.down.weight, std=0.02 / math.sqrt(2 * K))

    # ------------------------------------------------------------------ pieces
    def backbone(self, idx, states=None):
        """Engram reading the real words. Returns (normalised context per word, new block states)."""
        x = self.embed(idx)
        states = states or [None] * len(self.blocks)
        new = []
        for b, s in zip(self.blocks, states):
            x, s = b(x, s)
            new.append(s)
        return self.norm(x), new

    def belt(self, ctx, prev=None):
        """Run the belt over a stretch of words.

        ctx  (B, T, D)  context after each word;  prev (B, K, D) the belt before the first word (None = blank).
        Returns slots (K, B, T, D): slots[d][:, t] is the draft, made right after word t, of word t+d+1.
        """
        B, T, D = ctx.shape
        K = self.cfg.lookahead
        blank = self.blank.to(ctx.dtype).expand(B, T, D)
        mix = self.mix * self.mix_mask
        slots = [None] * K
        for d in range(K - 1, -1, -1):
            if d == K - 1:
                own = blank                                           # a fresh thought joins the belt
            else:                                                     # the belt moved: slot d+1, one word ago
                first = self.blank.to(ctx.dtype).expand(B, 1, D) if prev is None else prev[:, d + 1:d + 2].to(ctx.dtype)
                own = torch.cat([first, slots[d + 1][:, :-1]], dim=1)
            ahead = None
            if d < K - 1:
                ahead = sum(mix[d, j].to(ctx.dtype) * slots[j] for j in range(d + 1, K))
            slots[d] = self.refiner(own, ctx, ahead, d)
        return torch.stack(slots)

    def readout(self, slots, ctx):
        """Drafts -> vectors compared with every word. With skip, a draft is a correction to the present."""
        return self.out_norm(slots + ctx) if self.cfg.skip else self.out_norm(slots)

    def _slots(self, idx):
        ctx, _ = self.backbone(idx)
        return self.readout(self.belt(ctx), ctx)

    def enable_compile(self):
        self._compiled = torch.compile(self._slots)

    def _body(self, idx):
        """Next-word logits only (what loss and HellaSwag score)."""
        return F.linear(self._slots(idx)[0], self.embed.weight)

    def forward(self, idx, targets=None):
        S = (self._compiled or self._slots)(idx)                     # (K, B, T, D)
        logits = F.linear(S[0], self.embed.weight)
        loss, stats = None, {}
        if targets is not None:
            ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
            loss, stats["ce"] = ce, ce.detach()
            K, (B, T) = self.cfg.lookahead, targets.shape
            if self.training and K > 1 and self.cfg.draft_weight > 0:
                n = max(1, int(T * self.cfg.draft_frac))
                pos = torch.randperm(T, device=idx.device)[:n]
                tot = 0.0
                for d in range(1, K):
                    p = pos[pos + d < T]
                    if len(p) == 0:
                        continue
                    lg = F.linear(S[d][:, p], self.embed.weight).float()
                    cd = F.cross_entropy(lg.reshape(-1, lg.size(-1)), targets[:, p + d].reshape(-1), ignore_index=-1)
                    stats[f"draft{d + 1}"] = cd.detach()
                    tot = tot + cd
                loss = ce + self.cfg.draft_weight * tot / (K - 1)
        return logits, loss, stats

    # ------------------------------------------------------------------ reading as a stream
    def read(self, idx, states=None):
        """Read tokens (B, T) carrying memories and belt along. states = (block states, belt (B, K, D))."""
        blocks, prev = states[:2] if states is not None else (None, None)
        ctx, blocks = self.backbone(idx, blocks)
        raw = self.belt(ctx, prev)
        logits = F.linear(self.readout(raw[0], ctx), self.embed.weight)
        return logits, (blocks, raw[:, :, -1].transpose(0, 1).contiguous(), ctx[:, -1:])

    def draft_logits(self, states):
        """The live belt as word guesses for 1..K words ahead: (B, K, vocab)."""
        _, belt, ctx = states
        return F.linear(self.readout(belt, ctx), self.embed.weight)

    @torch.no_grad()
    def inspect(self, idx, states=None):
        """Engram's lid-off view of the backbone (for the dashboard snapshot); logits are Tide's own."""
        blocks, prev = states[:2] if states is not None else (None, None)
        x = self.embed(idx)
        blocks = blocks or [None] * len(self.blocks)
        new, infos = [], []
        for b, s in zip(self.blocks, blocks):
            y, s, info = b.mem.inspect(b.n1(x), s)
            x = x + y
            if b.mlp is not None:
                x = x + b.mlp(b.n2(x))
            new.append(s)
            infos.append(info)
        ctx = self.norm(x)
        return F.linear(self.readout(self.belt(ctx, prev)[0], ctx), self.embed.weight), new, infos

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


@torch.no_grad()
def draft_report(model: Tide, windows, device, batch=16):
    """How good is the draft of the word d ahead, for d = 1..K? (loss and top-1 accuracy per distance)

    d = 1 is the ordinary next-word loss. If the far drafts are good, the model is really planning.
    """
    was = model.training
    model.eval()
    xy = torch.from_numpy(windows).to(device)
    x, y = xy[:, :-1], xy[:, 1:]
    K, T = model.cfg.lookahead, x.shape[1]
    loss = torch.zeros(K, device=device)
    acc = torch.zeros(K, device=device)
    cnt = torch.zeros(K, device=device)
    for i in range(0, len(x), batch):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            S = model._slots(x[i:i + batch])
        for d in range(K):
            lg = F.linear(S[d][:, :T - d], model.embed.weight).float()
            tg = y[i:i + batch, d:]
            lo = F.cross_entropy(lg.transpose(1, 2), tg, reduction="none")
            loss[d] += lo.sum()
            acc[d] += (lg.argmax(-1) == tg).float().sum()
            cnt[d] += tg.numel()
            del lg, lo
    if was:
        model.train()
    return {"draft_loss": (loss / cnt).tolist(), "draft_acc": (acc / cnt).tolist()}


@torch.no_grad()
def foresight(model: Tide, enc, text):
    """Read a text and report, for every word, how each of its drafts saw it coming.

    draft_p[d][t]   the chance the draft made d+1 words before word t gave to word t (None at the start)
    draft_top[d][t] that draft's own best guess
    """
    from engram.probes import token_strings
    was = model.training
    model.eval()
    dev = model.embed.weight.device
    ids = enc.encode_ordinary(text)
    S = model._slots(torch.tensor([ids], device=dev))               # (K, 1, T, D)
    K, T = S.shape[0], len(ids)
    p_rows, top_rows = [], []
    for d in range(K):
        pr = F.softmax(F.linear(S[d, 0], model.embed.weight)[:, :50257].float(), -1)   # draft made after word t of word t+d+1
        tops = pr.argmax(-1).tolist()
        row_p, row_top = [None] * T, [None] * T
        for t in range(T - d - 1):
            w = t + d + 1
            row_p[w] = round(float(pr[t, ids[w]]), 4)
            row_top[w] = tops[t]
        p_rows.append(row_p)
        top_rows.append([None if x is None else token_strings(enc, [x])[0] for x in row_top])
    if was:
        model.train()
    return {"draft_p": p_rows, "draft_top": top_rows}
