"""Two minds that read the same stories.

SentenceMind (the creature's mind, for text). A story is lived one sentence at a time:
  reader   a sentence (bytes) -> 512 numbers                     (the creature's eye)
  memory   h: 512 numbers carried from sentence to sentence (GRU)
  moment   z: 32 slots x 16 options. Before reading a sentence it is guessed from memory alone
           (prior); after reading, corrected (posterior). The KL between them is the surprise.
  mouth    says the sentence back, byte by byte, from (h, z) only  (the anchor, like the
           creature's senses: it keeps the inner state about the actual words)
  Its exact score on a story: bits = mouth's bits for every sentence + KL surprise, an upper
  bound on -log2 p(story) (the ELBO). So it is directly comparable with a next-byte model.
  Thinking happens per sentence (one memory step); words appear only in the mouth.

ByteGPT: the usual next-byte transformer over the whole story (same bytes, similar size).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import EOS, PAD, VOCAB

H_DIM, Z_SLOTS, Z_CLS = 512, 32, 16
Z_DIM = Z_SLOTS * Z_CLS
FEAT = H_DIM + Z_DIM


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.h = heads

    def forward(self, x, causal=False, mask=None):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal and mask is None)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        return x + self.ff(self.n2(x))


def mlp(i, o, hid=512):
    return nn.Sequential(nn.Linear(i, hid), nn.LayerNorm(hid), nn.SiLU(), nn.Linear(hid, o))


def dist(logits):
    p = F.softmax(logits.view(*logits.shape[:-1], Z_SLOTS, Z_CLS).float(), -1)
    return 0.99 * p + 0.01 / Z_CLS


def sample(probs):
    u = torch.rand_like(probs).clamp_(1e-10, 1.0)
    idx = (torch.log(probs) - torch.log(-torch.log(u))).argmax(-1)
    hard = F.one_hot(idx, Z_CLS).to(probs.dtype)
    return (hard + probs - probs.detach()).flatten(-2)


def kl_cat(p, q):
    return (p * (torch.log(p) - torch.log(q))).sum((-1, -2))


class SentenceMind(nn.Module):
    def __init__(self, L=96, d=256, reader_layers=3, mouth_layers=6, heads=4, n_cond=4):
        super().__init__()
        self.L, self.d, self.n_cond = L, d, n_cond
        self.emb = nn.Embedding(VOCAB, d)
        self.pos_r = nn.Parameter(torch.zeros(L + 1, d))
        self.pos_m = nn.Parameter(torch.zeros(L + 1 + n_cond, d))
        nn.init.normal_(self.pos_r, std=0.02); nn.init.normal_(self.pos_m, std=0.02)
        self.reader = nn.ModuleList([Block(d, heads) for _ in range(reader_layers)])
        self.read_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 512))
        self.z_in = nn.Sequential(nn.Linear(Z_DIM, 512), nn.LayerNorm(512), nn.SiLU())
        self.gru = nn.GRUCell(512, H_DIM)
        self.prior = mlp(H_DIM, Z_DIM)
        self.post = mlp(H_DIM + 512, Z_DIM)
        self.cond = nn.Linear(FEAT, n_cond * d)
        self.mouth = nn.ModuleList([Block(d, heads) for _ in range(mouth_layers)])
        self.mouth_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, VOCAB))

    def read(self, x):
        """x (N, L+1) bytes -> (N, 512): the sentence as the reader sees it (pads ignored)."""
        keep = x != PAD
        h = self.emb(x.clamp(max=VOCAB - 1)) + self.pos_r
        mask = keep[:, None, None, :]
        for b in self.reader:
            h = b(h, mask=mask)
        w = keep.float()[..., None]
        return self.read_out((h * w).sum(1) / w.sum(1).clamp(min=1))

    def say_logits(self, feat, x):
        """Mouth: logits for every byte of x (N, L+1) given the inner state feat (N, FEAT)."""
        N = x.shape[0]
        c = self.cond(feat.to(self.cond.weight.dtype)).view(N, self.n_cond, self.d)
        inp = torch.cat([c, self.emb(x[:, :-1].clamp(max=VOCAB - 1))], 1) + self.pos_m[: self.n_cond + x.shape[1] - 1]
        for b in self.mouth:
            inp = b(inp, causal=True)
        return self.mouth_out(inp[:, self.n_cond - 1:])               # predicts x[:, 0..L]

    def forward(self, x, free_bits=1.0):
        """x (B, S, L+1). Returns training loss per byte, and exact bits (sum) + byte count for scoring."""
        B, S, Lp = x.shape
        e = self.read(x.flatten(0, 1)).view(B, S, -1)
        h = torch.zeros(B, H_DIM, device=x.device)
        z = torch.zeros(B, Z_DIM, device=x.device)
        hs, zs, posts, priors = [], [], [], []
        for t in range(S):
            if t > 0:
                h = self.gru(self.z_in(z).float(), h.float())
            p = dist(self.prior(h))
            q = dist(self.post(torch.cat([h, e[:, t].to(h.dtype)], -1)))
            z = sample(q)
            hs.append(h); zs.append(z); posts.append(q); priors.append(p)
        feat = torch.cat([torch.stack(hs, 1), torch.stack(zs, 1)], -1)    # B,S,FEAT
        post, prior = torch.stack(posts, 1), torch.stack(priors, 1)
        logits = self.say_logits(feat.flatten(0, 1), x.flatten(0, 1)).float()
        tgt = x.flatten(0, 1).long()
        nll = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), ignore_index=PAD, reduction="sum")
        nbytes = (tgt != PAD).sum()
        kl = kl_cat(post, prior)                                            # B,S nats
        dyn = kl_cat(post.detach(), prior)
        rep = kl_cat(post, prior.detach())
        kl_train = (0.8 * dyn.clamp(min=free_bits) + 0.2 * rep.clamp(min=free_bits)).sum()
        loss = (nll + kl_train) / nbytes
        bits = (nll + kl.sum()) / math.log(2)
        return loss, bits.detach(), nbytes, dict(nll_bpb=(nll / nbytes / math.log(2)).detach(),
                                                kl_bits_per_sent=(kl.mean() / math.log(2)).detach())

    @torch.no_grad()
    def dream(self, x_prompt, n_more, temp=0.8):
        """Read the prompt sentences, then imagine n_more sentences (prior only) and say them."""
        B, S, _ = x_prompt.shape
        e = self.read(x_prompt.flatten(0, 1)).view(B, S, -1)
        h = torch.zeros(B, H_DIM, device=x_prompt.device)
        z = torch.zeros(B, Z_DIM, device=x_prompt.device)
        for t in range(S):
            if t > 0:
                h = self.gru(self.z_in(z).float(), h.float())
            z = sample(dist(self.post(torch.cat([h, e[:, t].to(h.dtype)], -1))))
        said = []
        for _ in range(n_more):
            h = self.gru(self.z_in(z).float(), h.float())
            z = sample(dist(self.prior(h)))
            said.append(self.speak(torch.cat([h, z], -1), temp))
        return said

    @torch.no_grad()
    def speak(self, feat, temp=0.8):
        B = feat.shape[0]
        x = torch.full((B, self.L + 1), PAD, dtype=torch.long, device=feat.device)
        done = torch.zeros(B, dtype=torch.bool, device=feat.device)
        for i in range(self.L + 1):
            lg = self.say_logits(feat, x)[:, i].float()
            lg[:, PAD] = -1e9
            nxt = torch.multinomial(F.softmax(lg / temp, -1), 1)[:, 0]
            x[:, i] = torch.where(done, torch.full_like(nxt, PAD), nxt)
            done |= nxt == EOS
            if done.all():
                break
        return x


class ByteGPT(nn.Module):
    def __init__(self, ctx=1200, d=320, layers=9, heads=5):
        super().__init__()
        self.emb = nn.Embedding(VOCAB + 1, d)                              # + a start mark
        self.pos = nn.Parameter(torch.zeros(ctx + 1, d))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, VOCAB))

    def forward(self, s):
        """s (B, T) byte stream with PAD at the end. Returns loss per byte, bits (sum), byte count."""
        B, T = s.shape
        start = torch.full((B, 1), VOCAB, dtype=torch.long, device=s.device)
        inp = torch.cat([start, s[:, :-1].long().clamp(max=VOCAB - 1)], 1)
        h = self.emb(inp) + self.pos[:T]
        for b in self.blocks:
            h = b(h, causal=True)
        logits = self.out(h).float()
        tgt = s.long()
        nll = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), ignore_index=PAD, reduction="sum")
        n = (tgt != PAD).sum()
        return nll / n, (nll / math.log(2)).detach(), n, {}


class SentenceMind2(nn.Module):
    """Fixes 1 + 2.

    (+ the correction is the guess plus a change, so a predictable sentence costs ~0 bits.)
    1. The mouth also hears the previous sentence's words, so the moment z only has to carry
       what is new (names and spellings come from short-term word memory, like ours).
    2. Memory = a causal transformer over all past sentences (not one GRU step), so the guess
       can reach anything said before. The guess p(z_t | past text) and the correction
       q(z_t | past text, sentence t) both read it; everything runs in parallel over sentences.
    Exact score: sum_t [mouth bits of sentence t given (memory, z_t, previous words)] + KL bits,
    a valid upper bound on -log2 p(story), comparable with ByteGPT.
    Dreaming: guess z, say it, read what was said, carry on.
    """

    def __init__(self, L=96, d=256, reader_layers=3, mem_layers=3, mouth_layers=6, heads=4, n_cond=4, S=16):
        super().__init__()
        self.L, self.d, self.n_cond = L, d, n_cond
        self.emb = nn.Embedding(VOCAB, d)
        self.pos_r = nn.Parameter(torch.zeros(L + 1, d))
        self.pos_m = nn.Parameter(torch.zeros(2 * (L + 1) + n_cond, d))
        self.pos_s = nn.Parameter(torch.zeros(S + 1, d))
        for p in (self.pos_r, self.pos_m, self.pos_s):
            nn.init.normal_(p, std=0.02)
        self.reader = nn.ModuleList([Block(d, heads) for _ in range(reader_layers)])
        self.read_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.start = nn.Parameter(torch.zeros(1, 1, d))
        self.memory = nn.ModuleList([Block(d, heads) for _ in range(mem_layers)])
        self.mem_out = nn.LayerNorm(d)
        self.prior = mlp(d, Z_DIM)
        self.post = mlp(2 * d, Z_DIM)
        self.cond = nn.Linear(d + Z_DIM, n_cond * d)
        self.mouth = nn.ModuleList([Block(d, heads) for _ in range(mouth_layers)])
        self.mouth_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, VOCAB))

    def read(self, x):
        keep = x != PAD
        h = self.emb(x.clamp(max=VOCAB - 1)) + self.pos_r
        mask = keep[:, None, None, :]
        for b in self.reader:
            h = b(h, mask=mask)
        w = keep.float()[..., None]
        return self.read_out((h * w).sum(1) / w.sum(1).clamp(min=1))

    def remember(self, e):
        """e (B, S, d) sentences read so far -> ctx (B, S+1, d): ctx[:, t] has seen sentences < t."""
        B, S, _ = e.shape
        h = torch.cat([self.start.expand(B, 1, -1).to(e.dtype), e], 1) + self.pos_s[: S + 1]
        for b in self.memory:
            h = b(h, causal=True)
        return self.mem_out(h)

    def say_logits(self, ctx, z, prev, x):
        """Mouth: logits for every byte of x given memory ctx, moment z and the previous sentence's bytes."""
        N = x.shape[0]
        c = self.cond(torch.cat([ctx, z.to(ctx.dtype)], -1)).view(N, self.n_cond, self.d)
        P = prev.shape[1]
        inp = torch.cat([self.emb(prev.clamp(max=VOCAB - 1)), c, self.emb(x[:, :-1].clamp(max=VOCAB - 1))], 1)
        inp = inp + self.pos_m[: inp.shape[1]]
        for b in self.mouth:
            inp = b(inp, causal=True)
        return self.mouth_out(inp[:, P + self.n_cond - 1:])

    def forward(self, x, free_bits=1.0):
        B, S, Lp = x.shape
        e = self.read(x.flatten(0, 1)).view(B, S, -1)
        ctx = self.remember(e)[:, :S]                                       # memory before each sentence
        pl = self.prior(ctx)
        p = dist(pl)
        q = dist(pl + self.post(torch.cat([ctx, e], -1)))                  # correction = guess + change
        z = sample(q)
        prev = torch.cat([torch.full_like(x[:, :1], PAD), x[:, :-1]], 1)    # previous sentence's bytes
        logits = self.say_logits(ctx.flatten(0, 1), z.flatten(0, 1), prev.flatten(0, 1), x.flatten(0, 1)).float()
        tgt = x.flatten(0, 1).long()
        nll = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), ignore_index=PAD, reduction="sum")
        nbytes = (tgt != PAD).sum()
        kl = kl_cat(q, p)
        dyn = kl_cat(q.detach(), p)
        rep = kl_cat(q, p.detach())
        kl_train = (0.8 * dyn.clamp(min=free_bits) + 0.2 * rep.clamp(min=free_bits)).sum()
        loss = (nll + kl_train) / nbytes
        bits = (nll + kl.sum()) / math.log(2)
        return loss, bits.detach(), nbytes, dict(nll_bpb=(nll / nbytes / math.log(2)).detach(),
                                                kl_bits_per_sent=(kl.mean() / math.log(2)).detach())

    @torch.no_grad()
    def speak(self, ctx, z, prev, temp=0.8):
        B = ctx.shape[0]
        x = torch.full((B, self.L + 1), PAD, dtype=torch.long, device=ctx.device)
        done = torch.zeros(B, dtype=torch.bool, device=ctx.device)
        for i in range(self.L + 1):
            lg = self.say_logits(ctx, z, prev, x)[:, i].float()
            lg[:, PAD] = -1e9
            nxt = torch.multinomial(F.softmax(lg / temp, -1), 1)[:, 0]
            x[:, i] = torch.where(done, torch.full_like(nxt, PAD), nxt)
            done |= nxt == EOS
            if done.all():
                break
        return x

    @torch.no_grad()
    def dream(self, x_prompt, n_more, temp=0.8):
        """Read the prompt, then: guess a moment, say it, read what was said, carry on."""
        x = x_prompt.long()
        said = []
        for _ in range(n_more):
            B, S, _ = x.shape
            e = self.read(x.flatten(0, 1)).view(B, S, -1)
            ctx = self.remember(e)[:, S]
            z = sample(dist(self.prior(ctx)))
            s = self.speak(ctx, z, x[:, -1], temp)
            said.append(s)
            x = torch.cat([x, s[:, None]], 1)
        return said
