"""Anima-300M: a language model that keeps a picture of what it is reading and settles it by surprise.

Still scored as a NEXT-WORD model (every test reads the odds it gives each next token, so it races Gnosis on the same
text and the same tests), but the way it computes is not a transformer's.

Why not a transformer. A transformer has no picture of anything. Its only memory is the transcript: at every word, in
every layer, it looks back over every word it has read. It never has to make sense of the text, because it can always
re-read it. Anima cannot re-read. Its word layers see only the last `window` (64) pieces. Everything older reaches it
only through the MIND: a fixed handful of slots (16 vectors of 1,024) that is its current understanding of the text.
To get the next word right it has to keep a good picture, because the picture is all it has. Understanding is forced,
not hoped for.

How the picture changes. The text is cut into beats of 32 pieces. At each beat the mind settles, in a few passes:

  imagine   from the picture alone, imagine what this beat should look like (one vector per place in the beat)
  surprise  the beat as the ear heard it, minus what was imagined
  claim     each surprise is shared out among the slots, the slots competing for it (softmax over the slots, as in Slot
            Attention): a slot changes only by the share of the surprise it claims, through a gate
  law       the slots look at each other and think (shared weights, the same law every beat and every pass)

and again, with the new picture: imagine, surprise, claim, law. The mind only ever hears what it did not expect
(predictive coding). What it already understood costs nothing and changes nothing. Passes are thinking: more passes,
a more settled picture. At reading time the number of passes is a knob (settle_eval), so we can measure whether
thinking longer helps.

Two lessons of its own, besides the next word:
  foresight  the picture as it is BEFORE a beat must imagine that beat (graded against the ear, held still). A picture
             that only stores what happened fails this; a picture that holds how things work passes it.
  explain    after each pass the picture must explain the beat a little better than before, so the passes really
             settle (and a later pass at reading time keeps going the same way).

The parts, at the race size (d = 1,024):
  ear     4 word layers (Gnosis's layers: fused q/k/v, QK-norm, ReLU², value tables, FP8 products) that see only the
          last 64 pieces. Its output is what the mind hears.
  mind    16 slots; per beat 3 passes of imagine (16 heads) + claim (16 heads) + law (2 layers: slots attend to each other,
          then a ReLU² MLP 3,840). One set of weights for every beat and every pass.
  voice   12 more word layers, also 64 pieces back, that read the picture (as it was before the current beat began)
          before layers 7, 10, 13 and 16. The voice writes the next word.

What is the same as Gnosis: SmolLM2's pieces, the tied word table and its capped scores, the U-net skips and x0 mixing
across the 16 word layers, NorMuon, the windows 512 -> 1,024 -> 2,048, the batch ramp, WSD, the early 3-word lesson,
the loss in slices. What is new: no attention beyond 64 pieces anywhere, the recurrent picture, settling by surprise,
the two lessons. Memory while reading is fixed, however long the text (a transformer's grows with every word).

Honest reading: the picture is built only from beats that have ended, and the word layers look only back, so it stays
true next-word prediction (anima/tests.py checks that changing any piece changes nothing before it).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from bee5.model import causal, norm
from bee6.model import Attention6, Bee6, Bee6Config, Block6


@dataclass
class AnimaConfig(Bee6Config):
    n_layer: int = 16            # word layers: the ear's, then the voice's
    ear_layers: int = 4          # the mind hears the stream after this many layers
    window: int = 64             # every word layer sees at most this many pieces (itself included)
    beat: int = 32               # pieces per beat: the mind settles once per beat
    slots: int = 16              # the picture: this many vectors of d_model
    settle: int = 3              # passes per beat in training
    settle_eval: int = 0         # passes per beat when reading (held-out loss, tests, chat); 0 = settle
    law_layers: int = 2          # the law: slots attend to each other, then think alone (shared by every beat and pass)
    law_hidden: int = 3840       # the law's ReLU² width
    read_at: str = "6,9,12,15"   # the voice reads the picture before these word layers (from 0)
    foresight: float = 0.1       # weight of "the picture before a beat imagines that beat"
    explain: float = 0.05        # weight of "each pass explains the beat it is settling on better"
    mod_layers: str = ""         # no Mixture of Depths: a routed layer would pick words across the window
    key_offset: str = "3,7,11,15"

    def __post_init__(self):
        super().__post_init__()
        if self.bloom:
            raise ValueError("Anima does not bloom")
        if self.mod_set():
            raise ValueError("Anima has no Mixture-of-Depths layers (set mod_layers=\"\")")
        if self.experts or self.sparse_mlp:
            raise ValueError("Anima's word layers are plain (no experts, no sparse MLP)")
        self.ear_layers = max(1, min(self.ear_layers, self.n_layer - 1))
        if self.d_model % self.attn_head_dim:
            raise ValueError("d_model must be a multiple of attn_head_dim (the mind's heads)")

    def read_set(self):
        rs = sorted({int(s) for s in str(self.read_at).split(",") if s.strip() != ""})
        rs = [i for i in rs if self.ear_layers <= i < self.n_layer]
        return rs or [self.n_layer - 1]

    def passes(self, training: bool) -> int:
        return self.settle if training or not self.settle_eval else self.settle_eval

    def unit(self) -> int:
        """Lengths are padded to a multiple of this (whole beats and whole windows)."""
        return self.beat * self.window // math.gcd(self.beat, self.window)


# ---------------------------------------------------------------------------------------------------- short sight
def local(q, k, v, W):
    """Causal attention that sees at most the last W pieces (each piece included). q (B, T, H, hd); k, v (B, T, KV, hd).
    The text is cut into blocks of W; each block's pieces look at their own block and the one before, masked to
    exactly the W pieces up to themselves. -> (B, T, H, hd)."""
    B, T, H, hd = q.shape
    KV = k.shape[2]
    if T <= W:
        return causal(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    if T % W:
        raise ValueError(f"length {T} is not a multiple of the window {W}")
    nb, rep = T // W, H // KV
    qb = q.view(B, nb, W, H, hd).permute(0, 1, 3, 2, 4).reshape(B * nb, H, W, hd)

    def pair(t):                                     # each block's keys: the block before (zeros for the first), then its own
        t = t.view(B, nb, W, KV, hd)
        t = torch.cat([torch.cat([t.new_zeros(B, 1, W, KV, hd), t[:, :-1]], 1), t], 2)       # (B, nb, 2W, KV, hd)
        t = t.permute(0, 1, 3, 2, 4).reshape(B * nb, KV, 2 * W, hd)
        return t.repeat_interleave(rep, 1) if rep > 1 else t
    i = torch.arange(W, device=q.device)[:, None]
    j = torch.arange(2 * W, device=q.device)[None, :]
    m = (j > i) & (j <= i + W)                       # key (b-1)W + j is at most W-1 back from query bW + i, and not ahead
    first = m & (j >= W)                             # the first block has nothing before it
    mask = torch.where((torch.arange(nb, device=q.device) == 0)[:, None, None], first[None], m[None])   # (nb, W, 2W)
    mask = mask[None].expand(B, nb, W, 2 * W).reshape(B * nb, 1, W, 2 * W)
    y = F.scaled_dot_product_attention(qb, pair(k), pair(v), attn_mask=mask)
    return y.view(B, nb, H, W, hd).permute(0, 1, 3, 2, 4).reshape(B, T, H, hd)


class LocalAttention(Attention6):
    """Gnosis's attention (fused q/k/v, QK-norm, half RoPE, key offset, value tables, head gates, FP8), short-sighted."""

    def forward(self, x, ve, cos, sin):
        q, k, v, _ = self.project(x, ve, cos, sin)
        return self.finish(x, local(q, k, v, self.cfg.window))


class LocalBlock(Block6):
    def __init__(self, cfg: AnimaConfig, layer: int):
        super().__init__(cfg, layer)
        self.attn = LocalAttention(cfg, layer)


# ---------------------------------------------------------------------------------------------------- the mind
def heads(t, H):
    """(..., n, D) -> (..., H, n, D / H)."""
    *lead, n, D = t.shape
    return t.view(*lead, n, H, D // H).transpose(-2, -3)


def unheads(t):
    """(..., H, n, hd) -> (..., n, H * hd)."""
    *lead, H, n, hd = t.shape
    return t.transpose(-2, -3).reshape(*lead, n, H * hd)


class LawLayer(nn.Module):
    """One layer of the law: the slots look at each other (no order among them), then each thinks alone."""

    def __init__(self, D, H, hidden):
        super().__init__()
        self.h = H
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.out = nn.Linear(D, D, bias=False)
        self.up = nn.Linear(D, hidden, bias=False)
        self.down = nn.Linear(hidden, D, bias=False)

    def forward(self, S):
        q, k, v = (heads(t, self.h) for t in self.qkv(norm(S)).chunk(3, -1))
        S = S + self.out(unheads(F.scaled_dot_product_attention(norm(q), norm(k), v)))
        return S + self.down(F.relu(self.up(norm(S))).square())


class Mind(nn.Module):
    """The picture (slots) and how it settles on a beat: imagine, surprise, claim, law; `passes` times."""

    def __init__(self, cfg: AnimaConfig):
        super().__init__()
        D, N, C = cfg.d_model, cfg.slots, cfg.beat
        self.h = D // cfg.attn_head_dim
        self.s0 = nn.Parameter(torch.randn(N, D))          # the picture before any text: who each slot starts as
        self.pos = nn.Parameter(torch.randn(C, D))         # imagine: one question per place in the beat
        self.im_q = nn.Linear(D, D, bias=False)
        self.im_kv = nn.Linear(D, 2 * D, bias=False)
        self.im_out = nn.Linear(D, D, bias=False)
        self.cl_q = nn.Linear(D, D, bias=False)            # claim: the slots ask, the surprises answer
        self.cl_kv = nn.Linear(D, 2 * D, bias=False)
        self.cl_out = nn.Linear(D, D, bias=False)
        self.cl_gate = nn.Linear(2 * D, D)                 # how much of its claim a slot takes in, from (slot, claim)
        self.law = nn.ModuleList(LawLayer(D, self.h, cfg.law_hidden) for _ in range(cfg.law_layers))
        self.reset()

    @torch.no_grad()
    def reset(self):
        D = self.s0.shape[1]
        for lin in (self.im_q, self.im_kv, self.cl_q, self.cl_kv):
            nn.init.normal_(lin.weight, std=D ** -0.5)
        nn.init.zeros_(self.cl_out.weight)                 # the picture starts still (a claim adds nothing yet)
        nn.init.zeros_(self.im_out.weight)                 # it starts imagining nothing: every beat is all surprise
        nn.init.zeros_(self.cl_gate.weight)
        nn.init.zeros_(self.cl_gate.bias)
        for lay in self.law:
            nn.init.normal_(lay.qkv.weight, std=D ** -0.5)
            nn.init.normal_(lay.up.weight, std=0.5 * D ** -0.5)
            nn.init.zeros_(lay.out.weight)
            nn.init.zeros_(lay.down.weight)

    def start(self, B, dtype):
        return self.s0.to(dtype)[None].expand(B, -1, -1)

    def imagine(self, S):
        """The picture (B, N, D) -> what the beat should look like (B, C, D), from the picture alone."""
        Sn = norm(S)
        q = norm(heads(self.im_q(self.pos.to(S.dtype)), self.h))[None].expand(S.shape[0], -1, -1, -1)
        k, v = (heads(t, self.h) for t in self.im_kv(Sn).chunk(2, -1))
        return self.im_out(unheads(F.scaled_dot_product_attention(q, norm(k), v)))

    def claim(self, S, err):
        """Each surprise shared out among the slots, which compete for it; each slot takes in its claim through a gate."""
        Sn, en = norm(S), norm(err)
        q = norm(heads(self.cl_q(Sn), self.h))                                       # (B, H, N, hd)
        k, v = (heads(t, self.h) for t in self.cl_kv(en).chunk(2, -1))               # (B, H, C, hd)
        a = torch.softmax((q @ norm(k).transpose(-1, -2)).float() / math.sqrt(q.shape[-1]), dim=-2)   # over the slots
        a = a / (a.sum(-1, keepdim=True) + 1.0)          # a slot that claims little changes little (the +1)
        u = unheads(a.to(v.dtype) @ v)                                               # (B, N, D)
        g = torch.sigmoid(self.cl_gate(torch.cat([Sn, norm(u)], -1)))
        return S + g * self.cl_out(u)

    def beat(self, S, hb, passes: int):
        """Settle the picture on one beat. S (B, N, D); hb (B, C, D) the beat as the ear heard it (normed).
        -> (the settled picture, how far each pass's imagining is from the beat (passes,), how much the last pass moved it)."""
        target = hb.detach().float()
        errs = []
        move = S.new_zeros((), dtype=torch.float32)
        for k in range(passes):
            im = self.imagine(S)
            errs.append((target - im.float()).square().mean())
            before = S
            S = self.claim(S, hb - im)
            for lay in self.law:
                S = lay(S)
            if k == passes - 1:
                move = ((S - before).float().norm(dim=-1) / before.float().norm(dim=-1).clamp_min(1e-6)).mean().detach()
        return S, torch.stack(errs), move


class MindRead(nn.Module):
    """The voice reads the picture: each piece of beat c attends to the slots as they were before beat c began."""

    def __init__(self, D, H):
        super().__init__()
        self.h = H
        self.q = nn.Linear(D, D, bias=False)
        self.kv = nn.Linear(D, 2 * D, bias=False)
        self.out = nn.Linear(D, D, bias=False)

    def forward(self, x, P):
        """x (B, T, D) the stream; P (B, nC, N, D) the picture before each beat."""
        B, T, D = x.shape
        nC, N = P.shape[1], P.shape[2]
        q = norm(heads(self.q(norm(x)).view(B * nC, T // nC, D), self.h))
        k, v = (heads(t, self.h) for t in self.kv(norm(P.reshape(B * nC, N, D))).chunk(2, -1))
        y = unheads(F.scaled_dot_product_attention(q, norm(k), v))
        return self.out(y.reshape(B, T, D))


# ---------------------------------------------------------------------------------------------------- the model
class Anima(Bee6):
    """Ear (short-sighted word layers) -> mind (the picture, settled beat by beat) -> voice (short-sighted word layers
    that read the picture) -> the next word. See the top of this file."""
    Block = LocalBlock

    def __init__(self, cfg: AnimaConfig):
        super().__init__(cfg)
        D = cfg.d_model
        self.mind = Mind(cfg)
        self.reads = nn.ModuleDict({str(i): MindRead(D, D // cfg.attn_head_dim) for i in cfg.read_set()})
        with torch.no_grad():
            for r in self.reads.values():
                nn.init.normal_(r.q.weight, std=D ** -0.5)
                nn.init.normal_(r.kv.weight, std=D ** -0.5)
                nn.init.zeros_(r.out.weight)           # the voice starts deaf to the picture: a short-sighted model
        self.mind_on = True
        self._beat = None                              # the compiled beat (enable_compile)

    # ---------------------------------------------------------------- names
    def layout(self):
        c = self.cfg
        return (f"ear {c.ear_layers} + voice {c.n_layer - c.ear_layers} word layers of {c.d_model}, each seeing {c.window} "
                f"pieces back ({c.attn_heads} query / {c.attn_kv_heads} key-value heads, ReLU² {c.mlp_hidden}, "
                f"{c.n_ve} value tables{', FP8 products' if c.fp8 else ''}); mind: {c.slots} slots settled once per "
                f"{c.beat}-piece beat in {c.settle} passes (imagine, surprise, claim, law {c.law_layers} x {c.law_hidden}), "
                f"read before word layers {','.join(str(i + 1) for i in c.read_set())}")

    def depth_text(self):
        return f"{self.cfg.n_layer} word layers + a mind of {self.cfg.slots} slots"

    def speed_text(self):
        c = self.cfg
        return " · ".join(s for s in ["q/k/v fused" if c.fuse_qkv else "", f"FP8 mode {c.fp8}" if c.fp8 else "bf16",
                                      f"window {c.window}", f"{c.settle} passes per {c.beat}-piece beat"] if s)

    def mind_params(self):
        return sum(p.numel() for p in self.mind.parameters()) + sum(p.numel() for p in self.reads.parameters())

    def make_optimizer(self, lr, device, kind="muon", muon_lr=0.03, muon_wd=0.1):
        """Gnosis's (NorMuon for the grids, Adam for the rest), with the mind's and the reads' grids on NorMuon too.
        The starting picture and the imagine questions are tables, not grids: Adam."""
        from bee5 import optim as O
        from oazy.muon import Combined
        muon, adam = O.groups(self)
        mine = {id(p) for n, p in self.named_parameters()
                if p.dim() == 2 and n.startswith(("mind.", "reads.")) and n.endswith(".weight") and "cl_gate" not in n}
        adam["gates"] = [p for p in adam["gates"] if id(p) not in mine]
        muon += [(n, p) for n, p in self.named_parameters() if id(p) in mine]
        gs = [dict(params=ps, weight_decay=0.0, name=k, lr=lr * O.ADAM[k]["lr_scale"], **O.ADAM[k]) for k, ps in adam.items() if ps]
        if kind != "muon":
            gs.append(dict(params=[p for _, p in muon], weight_decay=0.1, name="grids", lr=lr, lr_scale=1.0, betas=(0.9, 0.95)))
            muon = []
        kw = dict(lr=lr, eps=1e-10)
        if device.type == "cuda":
            kw["fused"] = True
        opt = torch.optim.AdamW(gs, **kw)
        if not muon:
            return Combined(opt)
        mu = O.NorMuon([dict(params=[p for _, p in muon], lr_scale=muon_lr / lr)], lr=muon_lr, weight_decay=muon_wd)
        return Combined(mu, opt)

    # ---------------------------------------------------------------- the body
    def _pad(self, idx):
        """Any length reads as whole beats and whole windows: pad at the end (later pieces change nothing before them)."""
        u = self.cfg.unit()
        if idx.shape[1] % u:
            from oazy.tok import eot_id
            idx = torch.cat([idx, idx.new_full((idx.shape[0], u - idx.shape[1] % u), eot_id(self.cfg.tokenizer))], 1)
        return idx

    def think(self, x, passes=None):
        """The stream after the ear (B, T, D) -> the picture before every beat (B, nC, N, D), the passes' distances
        (nC, passes) or None, how much the last pass still moved the picture."""
        B, T, D = x.shape
        C = self.cfg.beat
        nC = T // C
        S = self.mind.start(B, x.dtype)
        if not self.mind_on:
            return S[:, None].expand(B, nC, -1, -1), None, None
        passes = passes or self.cfg.passes(self.training)
        hb = norm(x).view(B, nC, C, D)
        step = self._beat or self.mind.beat
        before, errs, moves = [], [], []
        for c in range(nC):
            before.append(S)
            S, e, mv = step(S, hb[:, c], passes)
            errs.append(e)
            moves.append(mv)
        self._last_picture = S
        return torch.stack(before, 1), torch.stack(errs), torch.stack(moves).mean()

    def _trunk_aux(self, idx, passes=None):
        """(B, T) words -> (vectors after the final norm, 0, 0, None, None, the mind's lesson and numbers)."""
        T0 = idx.shape[1]
        idx = self._pad(idx)
        T = idx.shape[1]
        e, _ = self._embed(idx)
        x0 = norm(e)
        ves = [t(idx) for t in self.ve]
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        keep = {}

        def layer(i, x, ve):
            if str(i) in self.reads:
                x = x + self.reads[str(i)](x, keep["P"])
            x, _, _ = self.blocks[i](x, ve, cos, sin)
            if i == self.cfg.ear_layers - 1:
                keep["P"], keep["errs"], keep["move"] = self.think(x, passes)
            return x
        h = self._run(x0, ves, layer)[:, :T0]
        z = h.new_zeros((), dtype=torch.float32)
        extra = {}
        if keep["errs"] is not None:
            E = keep["errs"]                                   # (nC, passes): pass 0 is the picture's guess before the beat
            fore = E[:, 0].mean()
            expl = E[:, 1:].mean() if E.shape[1] > 1 else z
            extra = {"aux": self.cfg.foresight * fore + self.cfg.explain * expl,
                     "mind_foresee": fore.detach(), "mind_settled": E[:, -1].mean().detach(), "mind_move": keep["move"]}
        return h, z, z, None, None, extra

    def _trunk(self, idx):
        return self._trunk_aux(idx)[0]

    def enable_compile(self):
        """Each word layer compiled on its own, and one beat of the mind compiled once and reused for every beat (the
        beats run one after another, so the whole body as one graph would unroll them all)."""
        import torch._dynamo
        dc = torch._dynamo.config
        dc.cache_size_limit = max(getattr(dc, "cache_size_limit", 8), 64)
        if hasattr(dc, "accumulated_cache_size_limit"):
            dc.accumulated_cache_size_limit = max(dc.accumulated_cache_size_limit, 512)
        mode = self.cfg.compile_mode
        kw = {} if mode in ("", "default") else {"mode": mode}
        for b in self.blocks:
            b.compile(dynamic=False, **kw)
        for r in self.reads.values():
            r.compile(dynamic=False, **kw)
        self._beat = torch.compile(self.mind.beat, dynamic=False, **kw)
        self._compiled = None                              # Bee6.losses then calls _trunk_aux, which uses the pieces above

    # ---------------------------------------------------------------- the trainer's hooks
    def _held_out(self, windows, device, batch, passes=None):
        """Held-out next-word loss, and how much the last pass still moved the picture."""
        x_all = torch.as_tensor(windows).long().to(device)
        tot, mv, n = 0.0, 0.0, 0
        for i in range(0, len(x_all), batch):
            xb = x_all[i:i + batch]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = self._trunk_aux(xb[:, :-1], passes)
                lg = self.scores(out[0]).float()
            tot += float(F.cross_entropy(lg.transpose(1, 2), xb[:, 1:], reduction="none").mean(1).sum())
            mv += float(out[5].get("mind_move", 0.0)) * len(xb)
            n += len(xb)
        return tot / max(1, n), mv / max(1, n)

    @torch.no_grad()
    def core_report(self, windows, device, batch=16):
        """Does the voice use the picture? Held-out next-word loss with the mind on, and off (the starting picture for
        every beat: only the last 64 pieces are left)."""
        was = self.training
        self.eval()
        try:
            on = self._held_out(windows, device, batch)[0]
            self.mind_on = False
            off = self._held_out(windows, device, batch)[0]
        finally:
            self.mind_on = True
            self.train(was)
        return {"on": on, "core_off": off, "core_gap": off - on}

    @torch.no_grad()
    def loop_report(self, windows, device, batch=16):
        """Does thinking longer help? Held-out loss with 1, 2, the trained number and twice that many passes per beat."""
        was = self.training
        self.eval()
        K = self.cfg.settle
        out, move = {}, 0.0
        try:
            for p in sorted({1, 2, K, 2 * K}):
                out[p], mv = self._held_out(windows, device, batch, passes=p)
                if p == 2 * K:
                    move = mv
        finally:
            self.train(was)
        return {"pass_loss": out, "adaptive_loss": out[K], "adaptive_passes": float(K), "move_last": move}

    def startup_check(self, device, say=print):
        say(f"[anima] {self.layout()}; the mind and its reads {self.mind_params() / 1e6:.1f}M of {self.num_params() / 1e6:.1f}M")
        super().startup_check(device, say)

    def grow_to(self, progress: float) -> bool:
        return False

    # ---------------------------------------------------------------- reading as a stream (chat, samples)
    def _read(self, idx, states=None, looks=False):
        """Reads the whole text again each time (slower than keeping the picture, the same numbers)."""
        prev = states["ids"] if states else idx[:, :0]
        ids = torch.cat([prev, idx], 1)[:, -self.cfg.max_len:]
        logits = self.scores(self._trunk(ids))[:, -idx.shape[1]:]
        return logits, {"ids": ids, "pos": ids.shape[1]}, []
