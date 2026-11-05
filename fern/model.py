"""Fern: a language model whose thoughts unfold, coarse to fine.

Fern does NOT write the next word. It writes a whole passage of 16 words at a time, and it writes
each passage the way a fern frond unfurls: the big shape first, the small detail last.

    the passage:   1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
    round 1                                                     ●      where is this going? (word 16)
    round 2                             ●                              the middle of the way there (8)
    round 3                 ●                       ●                  the middle of each half (4, 12)
    round 4           ●           ●           ●           ●            ... (2, 6, 10, 14)
    round 5        ●     ●     ●     ●     ●     ●     ●     ●         the last details (the odd words)

One rule, used five times: take every stretch whose two ends are decided and decide its middle.
Within a round all the middles are decided together, so each one is chosen knowing what is already
fixed on BOTH sides of it (the past and the future of the passage). 16 words come out in 5 rounds.

What a word "sees" when it is decided (and nothing else, checked by fern/tests.py):
  * everything written before the passage, read by the READER: Engram's learn-while-reading
    memories (engram/model.py), running over the finished text. The passage sees the reader's
    understanding of the last 16 words, and through its memories everything earlier (gist, not
    word for word: the memories are a fixed size however long the text is);
  * the words of the passage decided in earlier rounds (the coarser floors of the frond);
  * the thoughts of the earlier rounds (each round's slots, including its undecided ones: the
    "plan" the coarser floor had in mind), so meaning flows down the frond from gist to detail.

The part that decides the passage is the UNFOLDER: a few blocks in which the slots of a round
glance at each other and at the reader's last 16 words. That glance is attention, but only inside
a tiny window of at most 47 slots, never over the whole text: long-range context lives in the
reader's memories. The reader is causal and the unfolder only ever looks back down the frond, so
the model gives every text an exact probability: the product, over all words, of the chance the
word got in the round it was decided. That is directly comparable with GPT-2's loss (same text,
same tokens, nats per token), and lets HellaSwag be scored the usual way.

Training is parallel: the reader reads the whole 1,024-word window once; then every passage and
every round is laid out side by side (31 slots per passage: 1 + 2 + 4 + 8 + 16) and the unfolder
runs once over all of them, each round masked so that it sees only what it would see when writing.
Every word of the window is a target exactly once, in its own round.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from engram.model import Block as ReaderBlock, EngramConfig, RMSNorm
from oazy.sample import sample_logits


@dataclass
class FernConfig(EngramConfig):
    # the reader (Engram blocks): d_model, n_layer, n_head, head_k, head_v, mlp_hidden, conv
    n_layer: int = 18           # 18 reader blocks + 6 unfolder blocks = 198M parameters
    passage: int = 16            # words per passage (a power of 2): log2(passage) + 1 rounds
    unfold_layers: int = 6       # unfolder blocks
    unfold_heads: int = 12
    unfold_hidden: int = 2048    # width of the unfolder's per-slot layer
    shift: bool = True           # training: start the passage grid at a random place in each window
    rope: bool = True            # the glance knows how far apart two words are (rotary positions)


def level_of(i: int, P: int) -> int:
    """The round (0 = first) in which local position i (1..P) of a passage is decided."""
    L = int(math.log2(P))
    if i == P:
        return 0
    tz = (i & -i).bit_length() - 1          # trailing zeros
    return L - tz


def rotary(x, cos, sin):
    """Rotate pairs of channels by an angle proportional to the word's place (x: G, H, S, hd)."""
    a, b = x.chunk(2, dim=-1)
    return torch.cat([a * cos - b * sin, a * sin + b * cos], -1)


class Glance(nn.Module):
    """Slots of a passage look at each other (within the mask) and at the reader's last words."""

    def __init__(self, D, H):
        super().__init__()
        self.H = H
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.qn = RMSNorm(D // H)
        self.kn = RMSNorm(D // H)
        self.out = nn.Linear(D, D, bias=False)

    def forward(self, x, past, mask, rope=None):
        """x (G, S, D) slots; past (G, Pp, D) reader words; mask (G, 1, S, Pp + S) bool.
        rope: (cos, sin) for the Pp + S places (past words, then slots), or None."""
        G, S, D = x.shape
        H, hd = self.H, D // self.H
        q, k, v = self.qkv(x).split(D, dim=-1)
        kp, vp = F.linear(past, self.qkv.weight[D:]).split(D, dim=-1)
        k, v = torch.cat([kp, k], 1), torch.cat([vp, v], 1)
        q = self.qn(q.view(G, S, H, hd)).transpose(1, 2)
        k = self.kn(k.view(G, -1, H, hd)).transpose(1, 2)
        v = v.view(G, -1, H, hd).transpose(1, 2)
        if rope is not None:
            cos, sin = rope
            q = rotary(q, cos[:, -S:].to(q.dtype), sin[:, -S:].to(q.dtype))
            k = rotary(k, cos.to(k.dtype), sin.to(k.dtype))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(y.transpose(1, 2).reshape(G, S, D))


class UnfoldBlock(nn.Module):
    def __init__(self, cfg: FernConfig):
        super().__init__()
        D = cfg.d_model
        self.n1 = RMSNorm(D)
        self.np = RMSNorm(D)
        self.glance = Glance(D, cfg.unfold_heads)
        self.n2 = RMSNorm(D)
        self.up = nn.Linear(D, 2 * cfg.unfold_hidden, bias=False)
        self.down = nn.Linear(cfg.unfold_hidden, D, bias=False)

    def forward(self, x, past, mask, rope=None):
        x = x + self.glance(self.n1(x), self.np(past), mask, rope)
        a, b = self.up(self.n2(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(a) * b)


class Fern(nn.Module):
    def __init__(self, cfg: FernConfig):
        super().__init__()
        self.cfg = cfg
        P = cfg.passage
        assert P >= 2 and P & (P - 1) == 0, "passage must be a power of 2"
        L = int(math.log2(P))
        self.rounds = L + 1
        D = cfg.d_model
        self.embed = nn.Embedding(cfg.vocab_size, D)
        self.blocks = nn.ModuleList(ReaderBlock(cfg) for _ in range(cfg.n_layer))     # the reader
        self.norm = RMSNorm(D)
        self.unfold = nn.ModuleList(UnfoldBlock(cfg) for _ in range(cfg.unfold_layers))
        self.out_norm = RMSNorm(D)
        self.blank = nn.Parameter(torch.zeros(D))                  # "not decided yet"
        self.where = nn.Parameter(torch.zeros(P, D))               # place in the passage
        self.round_emb = nn.Parameter(torch.zeros(self.rounds, D))
        self.past_where = nn.Parameter(torch.zeros(P, D))          # how long ago (reader's last words)

        # the frond: round r has the slots at multiples of P >> r (r = 0: just the last word)
        pos, rnd, known = [], [], []
        for r in range(self.rounds):     # round r decides the words at the middles of every decided stretch
            stride = P >> r
            for i in range(stride, P + 1, stride):
                pos.append(i)
                rnd.append(r)
                known.append(level_of(i, P) < r)
        S = len(pos)                                                # 2P - 1 slots per passage
        self.register_buffer("slot_pos", torch.tensor(pos), persistent=False)
        self.register_buffer("slot_round", torch.tensor(rnd), persistent=False)
        self.register_buffer("slot_known", torch.tensor(known), persistent=False)
        rr = torch.tensor(rnd)
        allowed = torch.cat([torch.ones(S, P, dtype=torch.bool), rr[None, :] <= rr[:, None]], 1)
        self.register_buffer("allowed", allowed, persistent=False)  # (S, P + S)
        # the slot that decides each local position 1..P
        tgt = [next(s for s in range(S) if pos[s] == i and not known[s]) for i in range(1, P + 1)]
        self.register_buffer("target_slot", torch.tensor(tgt), persistent=False)
        # places for the rotary positions: the reader's words at -P+1..0, the slots at 1..P
        places = torch.cat([torch.arange(-P + 1, 1), torch.tensor(pos)]).float()
        hd = D // cfg.unfold_heads
        freq = 1.0 / (64.0 ** (torch.arange(hd // 2).float() / (hd // 2)))
        ang = places[:, None] * freq[None]
        self.register_buffer("rope_cos", ang.cos()[None], persistent=False)    # (1, P + S, hd / 2)
        self.register_buffer("rope_sin", ang.sin()[None], persistent=False)
        self.register_buffer("pos_round", torch.tensor([level_of(i, P) for i in range(1, P + 1)]), persistent=False)

        self.apply(self._init)
        depth = cfg.n_layer + cfg.unfold_layers
        for n, p in self.named_parameters():
            if n.endswith("mem.out.weight") or n.endswith("mlp.down.weight") or n.endswith(".down.weight") \
                    or n.endswith("glance.out.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * depth))
        for b in self.blocks:
            b.mem.reset_memory_params()
        with torch.no_grad():
            for t in (self.blank, self.where, self.round_emb, self.past_where):
                t.normal_(0, 0.02)
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

    def memory_numbers(self):
        c = self.cfg
        return c.n_layer * c.n_head * c.head_k * c.head_v

    # ------------------------------------------------------------------ the two halves
    def reader(self, idx):
        """Engram reading real words: (B, T) -> understanding after each word (B, T, D)."""
        x = self.embed(idx)
        for b in self.blocks:
            x, _ = b(x)
        return self.norm(x)

    def _hidden(self, seq, starts, ends, J, H=None):
        """The unfolder's final thought for every word of every passage.

        seq (B, N+1) tokens; passages start right after position starts[b] (per row) and words after
        position ends[b] do not exist. Returns h (B, J, P, D) for the words at base + 1..P, base (B, J).
        H: the reader's understanding of seq[:, :N], if already computed.
        """
        B, N1 = seq.shape
        N = N1 - 1
        P, D = self.cfg.passage, self.cfg.d_model
        dev = seq.device
        H = self.reader(seq[:, :N]) if H is None else H                     # (B, N, D)
        base = starts[:, None] + torch.arange(J, device=dev)[None, :] * P   # (B, J) last word before passage
        # the reader's last P words before the passage
        ppos = base[..., None] - (P - 1) + torch.arange(P, device=dev)        # (B, J, P)
        pvalid = (ppos >= 0) & (ppos < N) & (ppos <= ends[:, None, None])
        past = H.gather(1, ppos.clamp(0, N - 1).reshape(B, -1, 1).expand(-1, -1, D)).view(B, J, P, D)
        past = past + self.past_where.to(past.dtype)
        # the slots of the frond
        spos = base[..., None] + self.slot_pos                              # (B, J, S)
        exist = spos <= ends[:, None, None]
        tok = seq.gather(1, spos.clamp(max=N).reshape(B, -1)).view(spos.shape)
        word = self.embed(tok)
        show = (self.slot_known[None, None] & exist)[..., None]
        x = torch.where(show, word, self.blank.to(word.dtype))
        x = x + self.where[self.slot_pos - 1].to(x.dtype) + self.round_emb[self.slot_round].to(x.dtype)
        S = x.shape[2]
        keyok = torch.cat([pvalid, exist], -1)                              # (B, J, P + S)
        mask = self.allowed[None, None] & keyok[:, :, None, :]
        mask[..., P:] |= torch.eye(S, dtype=torch.bool, device=dev)         # a slot always sees itself
        G = B * J
        x, past, mask = x.reshape(G, S, D), past.reshape(G, P, D), mask.reshape(G, 1, S, P + S)
        rope = (self.rope_cos, self.rope_sin) if self.cfg.rope else None
        for blk in self.unfold:
            x = blk(x, past, mask, rope)
        x = x.view(B, J, S, D)[:, :, self.target_slot]                      # (B, J, P, D) in word order
        return self.out_norm(x), base

    def _scatter(self, h, base, N):
        """(B, J, P, D) word thoughts -> (B, N, D) aligned like GPT: out[:, t] is about word t + 1."""
        B, J, P, D = h.shape
        pos = base[..., None] + torch.arange(1, P + 1, device=h.device)    # (B, J, P)
        idx = (pos - 1).reshape(B, -1)
        keep = (idx < N)
        out = h.new_zeros(B, N + J * P, D)
        idx = torch.where(keep, idx, N + torch.arange(idx.shape[1], device=h.device)[None].expand(B, -1))
        out.scatter_(1, idx[..., None].expand(-1, -1, D), h.reshape(B, -1, D))
        return out[:, :N]

    def _core(self, seq, starts, ends, J):
        h, base = self._hidden(seq, starts, ends, J)
        return self._scatter(h, base, seq.shape[1] - 1)

    def enable_compile(self):
        self._compiled = torch.compile(self._core)

    def seq_logits(self, seq, starts=None, ends=None, compiled=False):
        """Logits (B, N, vocab) for a whole token sequence seq (B, N+1): logits[:, t] is the chance
        Fern gives to seq[:, t+1] in the round that word is decided. Positions up to starts[b] are
        context only (their logits are meaningless); words after ends[b] are treated as absent."""
        B, N1 = seq.shape
        dev = seq.device
        starts = torch.zeros(B, dtype=torch.long, device=dev) if starts is None else starts
        ends = torch.full((B,), N1 - 1, dtype=torch.long, device=dev) if ends is None else ends
        fn = self._compiled if (compiled and self._compiled is not None) else self._core
        J = -(-(N1 - 1 - int(starts.min())) // self.cfg.passage)          # passages needed
        return F.linear(fn(seq, starts, ends, J), self.embed.weight)

    def _body(self, idx):
        """GPT-style interface (last position has no next word here, so it gets a flat guess)."""
        return self.seq_logits(torch.cat([idx, idx[:, -1:]], 1), ends=torch.full(
            (idx.shape[0],), idx.shape[1] - 1, dtype=torch.long, device=idx.device))

    def score(self, x, y):
        """Logits for targets y given x (the trainer's convention: y = x shifted by one)."""
        return self.seq_logits(torch.cat([x[:, :1], y], 1))

    def rounds_of(self, starts, N):
        """(B, N) the round in which word t + 1 is decided (-1 = context, not decided by Fern)."""
        P = self.cfg.passage
        t = torch.arange(1, N + 1, device=starts.device)[None] - starts[:, None] - 1
        r = self.pos_round[(t % P).clamp(min=0)]
        return torch.where(t >= 0, r, torch.full_like(r, -1))

    def forward(self, idx, targets=None):
        seq = torch.cat([idx[:, :1], targets], 1) if targets is not None else torch.cat([idx, idx[:, -1:]], 1)
        B, N = idx.shape
        dev = idx.device
        r0 = int(torch.randint(0, self.cfg.passage, (1,))) if (self.training and self.cfg.shift) else 0
        starts = torch.full((B,), r0, dtype=torch.long, device=dev)
        logits = self.seq_logits(seq, starts, compiled=True)
        loss, stats = None, {}
        if targets is not None:
            tgt = torch.where(self.rounds_of(starts, N) >= 0, targets, torch.full_like(targets, -1))
            ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-1)
            loss, stats["ce"] = ce, ce.detach()
        return logits, loss, stats

    # ------------------------------------------------------------------ writing
    @torch.no_grad()
    def unfold_passage(self, ids, temperature=0.8, top_k=40, record=None, alts=0):
        """Write the next P words after ids, round by round.

        record: a list; each round appends {"round", "words": [(place, token, chance[, top alts])]}."""
        P = self.cfg.passage
        dev = self.embed.weight.device
        base = len(ids) - 1
        seq = torch.tensor([list(ids) + [0] * P], device=dev)
        st = torch.tensor([base], device=dev)
        for r in range(self.rounds):
            logits = self.seq_logits(seq, st)[0].float()
            chosen = []
            for i in range(1, P + 1):
                if level_of(i, P) != r:
                    continue
                lg = logits[base + i - 1]
                lg[50257:] = -float("inf")
                tok = sample_logits(lg, temperature, top_k)
                seq[0, base + i] = tok
                pr = F.softmax(lg / max(temperature, 1e-6), -1)
                if alts:
                    pv, pi = torch.topk(pr, alts)
                    chosen.append((i, tok, float(pr[tok]), [(int(t), float(v)) for t, v in zip(pi, pv)]))
                else:
                    chosen.append((i, tok, float(pr[tok])))
            if record is not None:
                record.append({"round": r, "words": chosen})
        return seq[0, base + 1:].tolist()

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=100, temperature=0.8, top_k=40, eos=None, record=None):
        ids = list(prompt_ids)
        out = []
        while len(out) < max_new_tokens:
            new = self.unfold_passage(ids, temperature, top_k, record)
            ids += new
            out += new
        out = out[:max_new_tokens]
        if eos is not None and eos in out:
            out = out[:out.index(eos)]
        return out, None


@torch.no_grad()
def round_report(model: Fern, windows, device, batch=16):
    """How hard is each round? Loss and top-1 accuracy of the words decided in round 1, 2, ... .

    Round 1 guesses a word 16 ahead from the past alone (hard); the last round fills single gaps
    between two known words (easy). The total loss is the average over all words."""
    was = model.training
    model.eval()
    xy = torch.from_numpy(windows).to(device)
    x, y = xy[:, :-1], xy[:, 1:]
    R = model.rounds
    loss = torch.zeros(R, device=device)
    acc = torch.zeros(R, device=device)
    cnt = torch.zeros(R, device=device)
    for i in range(0, len(x), batch):
        xb, yb = x[i:i + batch], y[i:i + batch]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg = model.score(xb, yb)
        lg = lg.float()
        lo = F.cross_entropy(lg.transpose(1, 2), yb, reduction="none")
        hit = (lg.argmax(-1) == yb).float()
        rr = model.rounds_of(torch.zeros(len(xb), dtype=torch.long, device=device), xb.shape[1])
        for r in range(R):
            m = (rr == r).float()
            loss[r] += (lo * m).sum()
            acc[r] += (hit * m).sum()
            cnt[r] += m.sum()
        del lg, lo
    if was:
        model.train()
    return {"round_loss": (loss / cnt.clamp_min(1)).tolist(), "round_acc": (acc / cnt.clamp_min(1)).tolist(),
            "round_words": [int(c) for c in (cnt / len(x)).tolist()]}
