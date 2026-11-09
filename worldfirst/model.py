"""The two models of stage 1.

LAWS (ours, NOT a next-word model). It predicts what the world does next and nothing else:
    1. Reader: a small transformer reads the scene, the action and the questions (both directions at once).
    2. Things: the words that name one thing are pooled into ONE vector for that thing; each question gets one too.
    3. Laws core: a tiny shared network where things look at each other and update. It runs again and again
       ("passes"), one tick of thinking each. Its input every pass is its last state plus the reader's picture, so it
       never loses the scene. The core only sees things, never words, and is too small to memorise: it has to become
       rules. Things keep their order of appearance (rotary positions on where each was first named), so "the column
       to my right" or "the person after me" means the same at any length.
    4. Head: answers from the settled picture: yes/no or a digit from a question's vector, a thing by pointing from
       the question to it, a sum's digits from each column's vector.
   Training: a random number of passes each step (1..max), learning only through the last few (Huginn's recipe), so
   the core works at any depth and the cost of learning does not grow with it. Testing: it keeps thinking until its
   picture stops moving (each pass changes it by less than `exit_eps`), at most `test_max` passes; an easy scene
   stops early, a hard one thinks longer.

RIVAL (a plain NEXT-WORD model): a normal causal transformer of the same size, trained by guessing every next word
of the same world text, answers included.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import N_TAGS


def rope(x, pos, base=10000.0):
    """x [B, H, T, D], pos [B, T] (float or long): rotate pairs of channels by angle pos * freq."""
    D = x.shape[-1]
    half = D // 2
    freq = base ** (-torch.arange(0, half, device=x.device, dtype=torch.float32) / half)
    ang = pos.float()[:, None, :, None] * freq            # [B, 1, T, half]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attn(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.qn = nn.RMSNorm(d // heads)
        self.kn = nn.RMSNorm(d // heads)

    def forward(self, x, pos, mask):
        B, T, d = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        q, k = rope(self.qn(q), pos), rope(self.kn(k), pos)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(y.transpose(1, 2).reshape(B, T, d))


class Block(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.attn = Attn(d, heads)
        self.up = nn.Linear(d, ff, bias=False)
        self.down = nn.Linear(ff, d, bias=False)

    def forward(self, x, pos, mask):
        x = x + self.attn(self.n1(x), pos, mask)
        return x + self.down(F.gelu(self.up(self.n2(x))))


@dataclass
class LawsConfig:
    vocab: int = 512
    d: int = 384
    heads: int = 6
    ff: int = 1536
    n_read: int = 4
    n_core: int = 2
    train_max: int = 16        # passes in training: uniform 1..train_max
    grad_passes: int = 4       # learning flows through the last few passes only
    test_max: int = 48         # most passes when it thinks until settled
    exit_eps: float = 0.01     # settled when a pass moves the picture by less than 1% of its size
    fixed: int = 0             # >0: always exactly this many passes, from a blank state, learning through all of them
    finder: int = 0            # >0: no marks; this many slots find the things themselves (slot attention)
    finder_iters: int = 3      # rounds in which the slots compete for the words
    teach: float = 0.0         # >0 (with finder): marks as a teacher. In training only, the world's marks say which words
                               # are the same thing, and the finder is graded on grouping them that way. Never an input.


class Laws(nn.Module):
    kind = "laws"

    def __init__(self, cfg: LawsConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.emb = nn.Embedding(cfg.vocab, d)
        self.reader = nn.ModuleList([Block(d, cfg.heads, cfg.ff) for _ in range(cfg.n_read)])
        self.read_norm = nn.RMSNorm(d)
        self.slot_kind = nn.Embedding(3, d)                 # pad / thing / question
        self.adapter = nn.Linear(2 * d, d, bias=False)      # [state, picture] -> core input
        self.core = nn.ModuleList([Block(d, cfg.heads, cfg.ff) for _ in range(cfg.n_core)])
        self.state_norm = nn.RMSNorm(d)
        self.cls = nn.Linear(d, 12)                         # yes, no, 0..9
        self.pq = nn.Linear(d, d, bias=False)
        self.pk = nn.Linear(d, d, bias=False)
        self.col = nn.Linear(d, 10)
        if cfg.finder:
            # the thing finder: slots that compete for the words (each word goes mostly to one slot)
            self.slot_init = nn.Parameter(torch.randn(cfg.finder, d) * 0.5)
            self.sa_in, self.sa_slot = nn.RMSNorm(d), nn.RMSNorm(d)
            self.sa_q, self.sa_k, self.sa_v = (nn.Linear(d, d, bias=False) for _ in range(3))
            self.sa_gru = nn.GRUCell(d, d)
            self.sa_mlp = nn.Sequential(nn.RMSNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
            # the speaker: writes a sum's digits one by one, looking only at the settled picture
            self.dig = nn.Embedding(11, d)                  # ten digits and "start"
            self.speak = nn.ModuleList([Block(d, cfg.heads, cfg.ff)])
            self.speak_norm = nn.RMSNorm(d)
            self.speak_out = nn.Linear(d, 10)
        self.apply(_init)

    def read(self, b):
        """The picture before thinking: (vectors [B, S, d], positions [B, S], which are real [B, S])."""
        tok = b["tok"]
        B, T = tok.shape
        valid = tok != 0
        mask = valid[:, None, None, :].expand(B, 1, T, T)
        pos = torch.arange(T, device=tok.device)[None].expand(B, T)
        h = self.emb(tok)
        for blk in self.reader:
            h = blk(h, pos, mask)
        h = self.read_norm(h)
        if not self.cfg.finder:
            slots = torch.bmm(b["pool"].to(h.dtype), h) + self.slot_kind(b["kind"])
            return slots, b["pos"], b["kind"] > 0
        things, tpos = self.find(h, valid, pos)
        qv = torch.bmm(b["qpool"].to(h.dtype), h) + self.slot_kind.weight[2]
        e = torch.cat([things + self.slot_kind.weight[1], qv], 1)
        p = torch.cat([tpos, b["qpos"].float()], 1)
        ok = torch.cat([torch.ones(B, self.cfg.finder, dtype=torch.bool, device=tok.device), b["qvalid"]], 1)
        return e, p, ok

    def find(self, h, valid, pos):
        """Slot attention: each slot claims words (they compete: a word is shared out over the slots), takes the mean of
        what it claimed, and updates; a few rounds. Returns slots [B, K, d] and each slot's position (where its words are)."""
        B, T, d = h.shape
        x = self.sa_in(h)
        k, v = self.sa_k(x), self.sa_v(x)
        sl = self.slot_init[None].expand(B, -1, -1).to(h.dtype)
        vm = valid[:, None, :].to(h.dtype)
        for _ in range(self.cfg.finder_iters):
            q = self.sa_q(self.sa_slot(sl))
            att = torch.softmax(torch.einsum("bkd,btd->bkt", q, k).float() / math.sqrt(d), dim=1).to(h.dtype) * vm
            w = att / (att.sum(-1, keepdim=True) + 1e-6)
            upd = torch.einsum("bkt,btd->bkd", w, v)
            sl = self.sa_gru(upd.reshape(-1, d).float(), sl.reshape(-1, d).float()).view(B, -1, d).to(h.dtype)
            sl = sl + self.sa_mlp(sl)
        self._att, self._w = att, w                     # who claimed which word (pointer answers, the found check)
        return sl, (w.float() * pos[:, None, :].float()).sum(-1)

    def step(self, s, e, pos, mask):
        x = self.adapter(torch.cat([s, e], -1))
        for blk in self.core:
            x = blk(x, pos, mask)
        return self.state_norm(x)

    def think(self, b, passes=None, adaptive=False, gen=None):
        """Run the core. Returns (final state, passes used per episode)."""
        cfg = self.cfg
        e, pos, valid = self.read(b)
        B, S, d = e.shape
        mask = valid[:, None, None, :].expand(B, 1, S, S)
        if gen is None and not self.training:
            gen = torch.Generator(device=e.device).manual_seed(1234)
        if cfg.fixed:                                       # a fixed number of passes: no early stop
            adaptive = False
        if cfg.train_max == 1 or cfg.fixed:                 # no loops, or fixed loops: start from a blank state
            s = torch.zeros(B, S, d, device=e.device, dtype=e.dtype)
        else:
            s = torch.randn(B, S, d, device=e.device, generator=gen).to(e.dtype)
        if passes is None:
            passes = cfg.fixed or (cfg.test_max if adaptive else cfg.train_max)
        used = torch.full((B,), passes, device=e.device, dtype=torch.long)
        done = torch.zeros(B, dtype=torch.bool, device=e.device)
        if self.training:
            k = min(cfg.grad_passes, passes)
            with torch.no_grad():
                for _ in range(passes - k):
                    s = self.step(s, e.detach(), pos, mask)
            s = s.detach()
            for _ in range(k):
                s = self.step(s, e, pos, mask)
            return s, used
        vm = valid[..., None].to(e.dtype)
        for i in range(passes):
            new = self.step(s, e, pos, mask)
            if adaptive:
                diff = ((new - s) * vm).flatten(1).norm(dim=1) / ((new * vm).flatten(1).norm(dim=1) + 1e-6)
                s = torch.where(done[:, None, None], s, new)
                now = (~done) & (diff < cfg.exit_eps)
                used = torch.where(now, torch.full_like(used, i + 1), used)
                done = done | now
                if bool(done.all()):
                    break
            else:
                s = new
        return s, used

    def answer(self, s, b):
        """Logits per answer item: class (12-way, masked to yes/no or digits), pointer (over this scene's things), column digit."""
        out = {}
        c = b["cls"]
        if len(c["b"]):
            lg = self.cls(s[c["b"], c["slot"]]).float()
            yesno = c["y"] < 2
            m = torch.zeros_like(lg, dtype=torch.bool)
            m[:, :2] = yesno[:, None]
            m[:, 2:] = ~yesno[:, None]
            out["cls"] = lg.masked_fill(~m, -1e9)
        p = b["ptr"]
        if len(p["b"]):
            q = self.pq(s[p["b"], p["slot"]])                       # [N, d]
            k = self.pk(s[p["b"]])                                  # [N, S, d]
            lg = torch.einsum("nd,nsd->ns", q, k).float() / math.sqrt(q.shape[-1])
            S = s.shape[1]
            is_thing = torch.arange(S, device=s.device)[None] < b["nthing"][p["b"], None]
            out["ptr"] = lg.masked_fill(~is_thing, -1e9)
        cc = b["col"]
        if len(cc["b"]):
            out["col"] = self.col(s[cc["b"], cc["slot"]]).float()
        return out

    def answer_found(self, s, b):
        """Answers when the model found the things itself. A class from the question's vector; a "who/what" by pointing
        at a slot and from the slot at its words (the chance of each name is the share of the slot's words that are
        that name); a sum written by the speaker, one digit at a time, looking at the whole picture."""
        K = self.cfg.finder
        out = {}
        c = b["cls"]
        if len(c["b"]):
            lg = self.cls(s[c["b"], K + c["q"]]).float()
            yesno = c["y"] < 2
            m = torch.zeros_like(lg, dtype=torch.bool)
            m[:, :2] = yesno[:, None]
            m[:, 2:] = ~yesno[:, None]
            out["cls"] = lg.masked_fill(~m, -1e9)
        p = b["ptr"]
        if len(p["b"]):
            q = self.pq(s[p["b"], K + p["q"]])
            k = self.pk(s[p["b"], :K])
            ps = torch.softmax(torch.einsum("nd,nkd->nk", q, k).float() / math.sqrt(q.shape[-1]), -1)
            pw = torch.einsum("nk,nkt->nt", ps, self._w[p["b"]].float()).float()  # chance on each word
            pv = torch.zeros(len(p["b"]), self.cfg.vocab, device=s.device).scatter_add_(1, b["tok"][p["b"]], pw)
            pv = pv.masked_fill(~p["cand"], 0) + p["cand"].float() * 1e-6
            pv = pv / pv.sum(-1, keepdim=True)
            out["ptr"] = torch.where(p["cand"], pv, torch.ones_like(pv)).log().masked_fill(~p["cand"], -1e9)
        q = b["seq"]
        if len(q["b"]):
            N, L = q["y"].shape
            pre = s[q["b"]]                                                     # [N, S, d]
            _, pos, ok = self._picture_meta(b, q["b"])
            S = pre.shape[1]
            inp = torch.cat([torch.full((N, 1), 10, device=s.device, dtype=torch.long), q["y"][:, :-1].clamp(min=0)], 1)
            x = torch.cat([pre, self.dig(inp)], 1)
            apos = q["end"][:, None].float() + 1 + torch.arange(L, device=s.device)[None].float()
            allpos = torch.cat([pos, apos], 1)
            see = torch.cat([ok, torch.ones(N, L, dtype=torch.bool, device=s.device)], 1)      # [N, S+L]
            causal = torch.ones(S + L, S + L, dtype=torch.bool, device=s.device).tril()
            causal[:, :S] = True                                               # everyone sees the whole picture
            causal[:S, S:] = False                                             # the picture does not see the answer
            mask = (causal[None] & see[:, None, :])[:, None]
            for blk in self.speak:
                x = blk(x, allpos, mask)
            out["seq"] = self.speak_out(self.speak_norm(x[:, S:])).float()
        return out

    def _picture_meta(self, b, idx):
        K = self.cfg.finder
        tpos = (self._w.float() * torch.arange(b["tok"].shape[1], device=b["tok"].device)[None, None].float()).sum(-1)
        pos = torch.cat([tpos, b["qpos"].float()], 1)[idx]
        ok = torch.cat([torch.ones(b["tok"].shape[0], K, dtype=torch.bool, device=b["tok"].device), b["qvalid"]], 1)[idx]
        return None, pos, ok

    @torch.no_grad()
    def found(self, b):
        """How well the slots match the true things (the world knows them; the model was never told): a thing counts as
        found when all of its words went mostly to one slot and no other thing's words went to that slot."""
        self.read(b)
        a = self._att.float().argmax(1).cpu()                                  # [B, T] the slot each word went to
        ment = b["ment"].cpu()
        got, n = 0, 0
        for i in range(a.shape[0]):
            m, sl = ment[i], a[i]
            slots = {t: set(sl[m == t].tolist()) for t in m.unique().tolist() if t >= 0}
            for t, mine in slots.items():
                n += 1
                if len(mine) == 1 and not any(mine & other for u, other in slots.items() if u != t):
                    got += 1
        return got, n

    def forward(self, b, passes=None, adaptive=False):
        s, used = self.think(b, passes, adaptive)
        return (self.answer_found(s, b) if self.cfg.finder else self.answer(s, b)), used

    def loss_and_hits(self, b, passes=None, adaptive=False):
        """Mean cross-entropy over answer items, and per item whether the best guess was right (in item order)."""
        out, used = self(b, passes, adaptive)
        n_items = len(b["meta"])
        hits = torch.zeros(n_items, dtype=torch.bool, device=b["tok"].device)
        losses = []
        for name in ("cls", "ptr", "col"):
            if name in out:
                y = b[name]["y"]
                losses.append(F.cross_entropy(out[name], y, reduction="sum"))
                hits[b[name]["item"]] = out[name].argmax(-1) == y
        if "seq" in out:
            y, it = b["seq"]["y"], b["seq"]["item"]
            ok = y >= 0
            losses.append(F.cross_entropy(out["seq"][ok], y[ok], reduction="sum"))
            hits[it[ok]] = out["seq"][ok].argmax(-1) == y[ok]
        loss = sum(losses) / max(1, n_items)
        if self.training and self.cfg.finder and self.cfg.teach > 0:
            loss = loss + self.cfg.teach * self.teach_loss(b["ment"])
        return loss, hits, used

    def teach_loss(self, ment):
        """Marks as a teacher: for every two words that name things, the finder's chance that they went to the same slot
        should be 1 when they are the same thing and 0 when not. Slots carry no fixed order, so only "same or different"
        is graded. The marks are only the answer key here; the model still reads plain text."""
        a = self._att.float()                                                  # [B, K, T] each word shared over slots
        same = torch.einsum("bki,bkj->bij", a, a).clamp(1e-6, 1 - 1e-6)        # chance two words share a slot
        m = ment >= 0
        pair = m[:, :, None] & m[:, None, :]
        pair &= ~torch.eye(ment.shape[1], dtype=torch.bool, device=ment.device)[None]
        y = (ment[:, :, None] == ment[:, None, :])
        pos, neg = pair & y, pair & ~y                                         # few "same" pairs, many "different": weigh
        lp = -same[pos].log().mean() if pos.any() else same.sum() * 0          # both halves equally
        ln = -(1 - same[neg]).log().mean() if neg.any() else same.sum() * 0
        return (lp + ln) / 2


@dataclass
class RivalConfig:
    vocab: int = 512
    d: int = 384
    heads: int = 6
    ff: int = 1536
    n_layer: int = 6
    loss: str = "all"          # "all": every next word (plain next-word learning); "answers": only the answer words


class Rival(nn.Module):
    kind = "rival"

    def __init__(self, cfg: RivalConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.emb = nn.Embedding(cfg.vocab, d)
        self.tag = nn.Embedding(N_TAGS, d)
        self.blocks = nn.ModuleList([Block(d, cfg.heads, cfg.ff) for _ in range(cfg.n_layer)])
        self.norm = nn.RMSNorm(d)
        self.apply(_init)
        with torch.no_grad():
            self.tag.weight[0].zero_()

    def forward(self, b):
        tok = b["tok"]
        B, T = tok.shape
        mask = torch.ones(T, T, dtype=torch.bool, device=tok.device).tril()[None, None]
        pos = torch.arange(T, device=tok.device)[None].expand(B, T)
        h = self.emb(tok) + self.tag(b["tag"])
        for blk in self.blocks:
            h = blk(h, pos, mask)
        return self.norm(h) @ self.emb.weight.T                 # tied word table

    def loss_and_hits(self, b, **_):
        lg = self(b)
        tok = b["tok"]
        a = lg[b["ib"], b["ipos"]].float()
        if self.cfg.loss == "answers":
            loss = F.cross_entropy(a, b["iy"])
        else:
            loss = F.cross_entropy(lg[:, :-1].flatten(0, 1).float(), tok[:, 1:].flatten(), ignore_index=0)
        a = a.masked_fill(~b["cand"], -1e9)
        hits = a.argmax(-1) == b["iy"]
        return loss, hits, None


def _init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=0.02)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


PRESETS = {
    # name: (d, heads, n_read, n_core)
    "tiny": dict(d=96, heads=4, n_read=2, n_core=1, train_max=8, grad_passes=3, test_max=24),
    "small": dict(d=256, heads=4, n_read=3, n_core=2),
    "base": dict(d=448, heads=7, n_read=4, n_core=2),
    "big": dict(d=576, heads=9, n_read=5, n_core=2),     # about 29M each: the long run
}
LR = {"tiny": 6e-4, "small": 6e-4, "base": 6e-4, "big": 5e-4}


def make_pair(vocab_size: int, preset: str = "base", **over):
    """The Laws model and a rival of the same size (its MLP width is tuned so the counts match within 1%)."""
    p = dict(PRESETS[preset])
    p.update(over)
    p.setdefault("ff", 4 * p["d"])
    lc = LawsConfig(vocab=vocab_size, **p)
    laws = Laws(lc)
    target = n_params(laws)
    n_layer = lc.n_read + lc.n_core
    best = None
    for ff in range(64, 16 * lc.d + 1, 16):
        rc = RivalConfig(vocab=vocab_size, d=lc.d, heads=lc.heads, ff=ff, n_layer=n_layer)
        n = _rival_count(rc)
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (rc, n)
    rival = Rival(best[0])
    return laws, rival


def _rival_count(rc: RivalConfig):
    d, ff, L, V = rc.d, rc.ff, rc.n_layer, rc.vocab
    per = 4 * d * d + 2 * (d // rc.heads) + 2 * d * ff + 2 * d
    return V * d + N_TAGS * d + L * per + d


def config_of(m):
    return {"kind": m.kind, **asdict(m.cfg)}


def from_config(c):
    c = dict(c)
    kind = c.pop("kind")
    return Laws(LawsConfig(**c)) if kind == "laws" else Rival(RivalConfig(**c))
