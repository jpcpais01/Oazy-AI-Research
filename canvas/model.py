"""Canvas-135: a model that thinks in meanings, all of the near future at once, and only then speaks.

João's idea (21 Nov 2026): "predicting words is wrong. It should be predicting thoughts and meanings. And predicting
just one is wrong. It should predict everything at the same time ... Diffusion must happen all at once, slowly. Some
parts are more sure than others ... thinking in vector space ... we just need a mouth that talks in the end." With
LeCun's JEPA (predict the meaning of what comes, not its words) and the 2026 latent-reasoning results (a hidden state
iterated until it settles) as the nearest published work.

The parts (race size, 132M weights like Bee-135):

  ear     4 word layers of 576 that see only the last 64 pieces. The text is cut into chunks of 16 pieces; the ear's
          vector at the end of each chunk, squeezed to 128 numbers, is that chunk's MEANING z.
  mind    a causal transformer over the meanings (4 layers of 768), one position per chunk: 16 times fewer positions
          than words, so it is cheap per word. Its state m_c knows every chunk before chunk c.
  canvas  at every chunk boundary c, 4 SLOTS: the meanings of the next 4 chunks (c .. c+3), all at once. They start
          as fog (zeros at noise level 1) and are refined together for a few steps (2 layers of 768, the same weights
          every step): each slot sees the mind and every other slot of its canvas (every output sees every output).
          Each step it guesses the clean meanings, says how sure it is of each slot, and the canvas is re-noised to the
          next, lower noise level (deterministic DDIM). A slot it is sure of drops its noise faster (adapt=1): sure parts
          settle first and pull the rest.
  mouth   8 more word layers that also see only the last 64 pieces, and read the settled canvas before layers 5, 7, 9
          and 11. No word layer sees past 64 pieces, so far context has to travel through the canvas or hop from layer
          to layer (12 hops could reach ~750 pieces back, but each hop blurs it); the canvas-off check measures how much
          the words really lean on the thought (Pegasus 1 and 2 failed because their writer could see everything and
          ignored the plan).

Its lessons:
  next word   the mouth's next-word loss, through the canvas exactly as it is built at reading time (from fog, from
              the past only), so the held-out loss and every test are honest next-word scores. Only the last refining
              step carries gradient (the earlier ones run without it, as AlphaFold's recycling).
  diffusion   the true meanings of the next 4 chunks (the ear's own, held still), noised, each slot to its own random
              level (10% of slots given clean: "I already know some parts ahead"); the canvas must remove the noise
              (predict the clean meaning). This teaches it the shape of plausible futures, jointly.
  honest      the settled canvas against the true future meanings (what it really foresaw).
  sure        each slot's own error estimate, graded (Gaussian likelihood): what makes "some parts more sure" real.
  SIGReg      LeJEPA's regulariser keeps the meanings spread like a standard Gaussian (no collapse; the right shape
              for noising).

Is it a next-token model? The MIND is not: it never sees a word, and it predicts meanings of whole chunks, several
ahead, jointly, by iterative denoising. The MOUTH still writes word by word (that is how text is scored), but it can see
only 64 pieces back, so everything beyond that comes from the thought.

Honest reading: the canvas at boundary c is built only from chunks before c and the word layers look only back, so
changing any piece changes nothing before it (canvas/tests.py checks it). The diffusion lesson uses the true future,
but only as a lesson: its outputs never reach the words.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from bee5.model import norm
from bee6.model import Bee6
<<<<<<< HEAD
from bee5.loss import set_compile as _loss_compile
from bee6 import lean as LEAN
from bee6.model import Attention6, Block6
from lucid.model import Lucid, LucidConfig, LocalAttention, MindRead, heads, unheads
=======
from lucid.model import Lucid, LucidConfig, LocalBlock, MindRead, heads, unheads
>>>>>>> origin/main


@dataclass
class CanvasConfig(LucidConfig):
    # the word layers (Bee-135's width and heads)
    d_model: int = 576
    attn_heads: int = 9
    attn_kv_heads: int = 3
    attn_head_dim: int = 64
    mlp_hidden: int = 2304
    n_layer: int = 12            # word layers: the ear's, then the mouth's
    ear_layers: int = 4
<<<<<<< HEAD
    window: int = 64             # every word layer sees at most this many pieces (itself included); 0 = the whole text
                                 # (Canvas-2: the mouth remembers like Bee, the canvas adds the thought on top)
=======
    window: int = 64             # every word layer sees at most this many pieces (itself included)
>>>>>>> origin/main
    beat: int = 16               # pieces per chunk: one meaning, one canvas boundary
    read_at: str = "4,6,8,10"    # the mouth reads the canvas before these word layers (from 0)
    key_offset: str = "3,7,11"
    mod_layers: str = ""
    fp8: int = 0
    # the meaning, the mind, the canvas
    z_dim: int = 128             # numbers in a chunk's meaning
    mind_dim: int = 768
    mind_heads: int = 12
    mind_layers: int = 4
    mind_hidden: int = 2304      # ReLU² width of the mind's and the canvas's layers
    slots: int = 4               # chunks of the future on the canvas (c .. c + slots - 1)
    canvas_layers: int = 2       # the denoiser's layers (the same weights every step)
    steps: int = 4               # refining steps when training (from fog at noise 1 to noise sigma_min)
    steps_eval: int = 0          # refining steps when reading (held-out loss, tests, chat); 0 = steps
    grad_steps: int = 1          # the last this-many steps carry gradient (the others are recycling, no gradient)
    sigma_min: float = 0.15      # the noise level of the last step
    adapt: int = 1               # a slot it is sure of drops its noise faster (to its own error estimate)
    # the lessons' weights
    diffuse: float = 1.0         # remove the noise from the true future meanings
    honest: float = 0.5          # the settled canvas against the true future meanings
    sure: float = 0.1            # each slot's error estimate (Gaussian likelihood)
    sig: float = 1.0             # SIGReg on the meanings
    sig_dirs: int = 128          # SIGReg's random directions
    known: float = 0.1           # the share of slots given clean in the diffusion lesson ("parts known ahead")
    full: float = 0.15           # the share of slots at noise 1 (pure fog) in the diffusion lesson

    def __post_init__(self):
        super().__post_init__()
        if self.mind_dim % self.mind_heads:
            raise ValueError("mind_dim must be a multiple of mind_heads")
        if self.slots < 1 or self.steps < 1:
            raise ValueError("slots and steps must be at least 1")

<<<<<<< HEAD
    def unit(self) -> int:
        """Lengths are padded to a multiple of this (whole chunks, and whole windows when the sight is short)."""
        return super().unit() if self.window else self.beat

=======
>>>>>>> origin/main
    def n_steps(self, training: bool) -> int:
        return self.steps if training or not self.steps_eval else self.steps_eval


def sigmas(R: int, lo: float):
    """The noise level of each refining step: 1 (fog) down to lo, evenly."""
    if R == 1:
        return [1.0]
    return [1.0 - (1.0 - lo) * r / (R - 1) for r in range(R)]


def sigreg(z, dirs: int, knots: int = 17):
    """LeJEPA's SIGReg (Balestriero & LeCun 2025): random 1-D projections of z should look like a standard normal.
    The squared gap between each projection's empirical characteristic function and exp(-t^2/2), weighted and
    integrated over t in [0, 4], averaged over the directions. z (N, d) -> a number (about 0.4 when z collapses)."""
    z = z.float()
    A = torch.randn(z.shape[1], dirs, device=z.device)
    A = A / A.norm(dim=0, keepdim=True)
    p = z @ A                                                    # (N, dirs)
    t = torch.linspace(0.0, 4.0, knots, device=z.device)
    tp = p[..., None] * t                                        # (N, dirs, knots)
    re, im = torch.cos(tp).mean(0), torch.sin(tp).mean(0)
    g = torch.exp(-0.5 * t * t)
    gap = ((re - g) ** 2 + im ** 2) * g
    return torch.trapezoid(gap, t, dim=-1).mean()


class Layer(nn.Module):
    """Pre-norm attention (QK-norm) + ReLU² MLP, outputs starting at zero. mask: a bool grid (True = may look) or
    "causal"."""

    def __init__(self, D, H, hidden):
        super().__init__()
        self.h = H
        self.q = nn.Linear(D, D, bias=False)
        self.kv = nn.Linear(D, 2 * D, bias=False)
        self.out = nn.Linear(D, D, bias=False)
        self.up = nn.Linear(D, hidden, bias=False)
        self.down = nn.Linear(hidden, D, bias=False)
        with torch.no_grad():
            nn.init.normal_(self.q.weight, std=D ** -0.5)
            nn.init.normal_(self.kv.weight, std=D ** -0.5)
            nn.init.normal_(self.up.weight, std=0.5 * D ** -0.5)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.down.weight)

    def forward(self, x, mem=None, mask=None):
        """x (B, n, D) the queries' stream; mem (B, m, D) extra things to look at, before x (or None)."""
        xn = norm(x)
        src = xn if mem is None else torch.cat([norm(mem), xn], 1)
        q = norm(heads(self.q(xn), self.h))
        k, v = (heads(t, self.h) for t in self.kv(src).chunk(2, -1))
        if isinstance(mask, str):
            y = F.scaled_dot_product_attention(q, norm(k), v, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, norm(k), v, attn_mask=mask)
        x = x + self.out(unheads(y))
        return x + self.down(F.relu(self.up(norm(x))).square())


class Canvas(nn.Module):
    """The denoiser: (noisy future meanings, their noise levels, the mind) -> (clean guesses, how sure, slot states)."""

    def __init__(self, cfg: CanvasConfig):
        super().__init__()
        D, K = cfg.mind_dim, cfg.slots
        self.K = K
        self.inp = nn.Linear(cfg.z_dim, D, bias=False)
        self.ctx = nn.Linear(D, D, bias=False)                   # the mind's state at this boundary, into every slot
        self.slot_pos = nn.Parameter(torch.randn(K, D) * 0.02)   # which chunk ahead a slot stands for
        self.register_buffer("freqs", torch.exp(torch.linspace(0.0, math.log(1000.0), 16)), persistent=False)
        self.noise_in = nn.Linear(32, D, bias=False)             # the slot's noise level (Fourier features)
        self.layers = nn.ModuleList(Layer(D, cfg.mind_heads, cfg.mind_hidden) for _ in range(cfg.canvas_layers))
        self.guess = nn.Linear(D, cfg.z_dim, bias=False)         # the clean meaning
        self.err = nn.Linear(D, 1)                               # log of its error's std (per number)
        with torch.no_grad():
            for lin in (self.inp, self.ctx, self.noise_in, self.guess):
                nn.init.normal_(lin.weight, std=lin.weight.shape[1] ** -0.5)
            nn.init.zeros_(self.err.weight)
            nn.init.zeros_(self.err.bias)                        # starts at "error 1": knows nothing
        self._masks = {}

    def mask(self, n, device):
        """(n K, n + n K) bool: slot k of boundary c looks at the mind's positions up to c and at the slots of c."""
        key = (n, device)
        if key not in self._masks:
            K = self.K
            c = torch.arange(n, device=device).repeat_interleave(K)                       # each slot's boundary
            mind = torch.arange(n, device=device)[None, :] <= c[:, None]
            own = c[:, None] == c[None, :]
            self._masks[key] = torch.cat([mind, own], 1)
        return self._masks[key]

    def forward(self, x, sig, m, mask):
        """x (B, n, K, z) noisy meanings; sig (B, n, K) their noise levels; m (B, n, D) the mind before each chunk;
        mask = self.mask(n, device) (made outside, so a compiled forward holds no cache).
        -> (clean guesses (B, n, K, z), log error std (B, n, K), slot states (B, n, K, D))."""
        B, n, K, _ = x.shape
        ang = sig[..., None].to(self.freqs.dtype) * self.freqs                            # (B, n, K, 16)
        f = torch.cat([torch.sin(ang), torch.cos(ang)], -1).to(x.dtype)
        h = self.inp(x) + self.noise_in(f) + self.ctx(m)[:, :, None] + self.slot_pos.to(x.dtype)
        h = h.reshape(B, n * K, -1)
        for lay in self.layers:
            h = lay(h, m, mask)
        hn = norm(h)
        y = self.guess(hn).view(B, n, K, -1)
        e = self.err(hn).view(B, n, K).float().clamp(-6.0, 3.0)
        return y, e, h.view(B, n, K, -1)


<<<<<<< HEAD
class SightAttention(LocalAttention):
    """Gnosis's attention, short-sighted (window > 0, Canvas-1) or seeing the whole text (window = 0, Canvas-2: Bee's
    own causal attention, so cuDNN's kernel works for it)."""

    def forward(self, x, ve, cos, sin):
        if self.cfg.window:
            return LocalAttention.forward(self, x, ve, cos, sin)
        return Attention6.forward(self, x, ve, cos, sin)


class SightBlock(Block6):
    def __init__(self, cfg, layer: int):
        super().__init__(cfg, layer)
        self.attn = SightAttention(cfg, layer)


class CanvasModel(Lucid):
    """Ear (short-sighted word layers) -> meanings -> mind -> canvas (the near future's meanings, refined together)
    -> mouth (short-sighted word layers that read the canvas) -> the next word. See the top of this file."""
    Block = SightBlock
=======
class CanvasModel(Lucid):
    """Ear (short-sighted word layers) -> meanings -> mind -> canvas (the near future's meanings, refined together)
    -> mouth (short-sighted word layers that read the canvas) -> the next word. See the top of this file."""
    Block = LocalBlock
>>>>>>> origin/main

    def __init__(self, cfg: CanvasConfig):
        Bee6.__init__(self, cfg)
        D, M = cfg.d_model, cfg.mind_dim
        self.to_z = nn.Linear(D, cfg.z_dim, bias=False)          # the ear's last vector of a chunk -> its meaning
        self.mind_in = nn.Linear(cfg.z_dim, M, bias=False)
        self.mind_start = nn.Parameter(torch.randn(M) * 0.02)    # the mind before any chunk
        self.mind = nn.ModuleList(Layer(M, cfg.mind_heads, cfg.mind_hidden) for _ in range(cfg.mind_layers))
        self.canvas = Canvas(cfg)
        self.to_word = nn.Linear(M, D, bias=False)               # the slots' states, at the words' width
        self.reads = nn.ModuleDict({str(i): MindRead(D, D // cfg.attn_head_dim) for i in cfg.read_set()})
        with torch.no_grad():
            for lin in (self.to_z, self.mind_in, self.to_word):
                nn.init.normal_(lin.weight, std=lin.weight.shape[1] ** -0.5)
            for r in self.reads.values():
                nn.init.normal_(r.q.weight, std=D ** -0.5)
                nn.init.normal_(r.kv.weight, std=D ** -0.5)
                nn.init.zeros_(r.out.weight)                     # the mouth starts deaf to the canvas
        self.mind_on = True
        self._beat = None
        self._canvas = None                                      # the compiled denoiser (enable_compile)

    # ---------------------------------------------------------------- names
    def layout(self):
        c = self.cfg
<<<<<<< HEAD
        sight = f"each seeing {c.window} pieces back" if c.window else "seeing the whole text"
        return (f"ear {c.ear_layers} + mouth {c.n_layer - c.ear_layers} word layers of {c.d_model}, {sight}"
                f"; meanings of {c.z_dim} per {c.beat}-piece chunk; mind {c.mind_layers} x {c.mind_dim}; canvas "
=======
        return (f"ear {c.ear_layers} + mouth {c.n_layer - c.ear_layers} word layers of {c.d_model}, each seeing {c.window} "
                f"pieces back; meanings of {c.z_dim} per {c.beat}-piece chunk; mind {c.mind_layers} x {c.mind_dim}; canvas "
>>>>>>> origin/main
                f"of {c.slots} future chunks refined together in {c.steps} steps ({c.canvas_layers} layers), read before "
                f"word layers {','.join(str(i + 1) for i in c.read_set())}")

    def depth_text(self):
        return f"{self.cfg.n_layer} word layers + a mind of {self.cfg.mind_layers} and a canvas of {self.cfg.slots}"

    def speed_text(self):
        c = self.cfg
<<<<<<< HEAD
        return " · ".join(s for s in ["q/k/v fused" if c.fuse_qkv else "", "bf16",
                                      "lean" if c.lean else "", "cuDNN attention" if c.attn_backend == "cudnn" else "",
                                      f"window {c.window}" if c.window else "full sight",
=======
        return " · ".join(s for s in ["q/k/v fused" if c.fuse_qkv else "", "bf16", f"window {c.window}",
>>>>>>> origin/main
                                      f"{c.steps} canvas steps per {c.beat}-piece chunk"] if s)

    def mind_params(self):
        parts = [self.to_z, self.mind_in, self.mind, self.canvas, self.to_word, self.reads]
        return sum(p.numel() for m in parts for p in m.parameters()) + self.mind_start.numel()

    def make_optimizer(self, lr, device, kind="muon", muon_lr=0.03, muon_wd=0.1):
        """Gnosis's (NorMuon for the grids, Adam for the rest), with the mind's, the canvas's and the reads' grids on
        NorMuon too. Tables (the slots' places, the start) and the error head are Adam's."""
        from bee5 import optim as O
        from oazy.muon import Combined
        muon, adam = O.groups(self)
        mine = {id(p) for n, p in self.named_parameters()
                if p.dim() == 2 and n.endswith(".weight") and not n.startswith(("blocks.", "embed", "ve."))
                and not n.endswith(("err.weight", "noise_in.weight")) and "gate" not in n}
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

    # ---------------------------------------------------------------- the thought
    def meanings(self, x):
        """The stream after the ear (B, T, D) -> each chunk's meaning (B, n, z): the ear's vector at its last piece."""
        C = self.cfg.beat
        return self.to_z(norm(x[:, C - 1::C]))

    def think_mind(self, z):
        """Meanings (B, n, z) -> the mind before each chunk (B, n, M): position c knows chunks 0 .. c-1."""
        B = z.shape[0]
        u = self.mind_in(z[:, :-1])
        h = torch.cat([self.mind_start.to(u.dtype).expand(B, 1, -1), u], 1)
        for lay in self.mind:
            h = lay(h, None, "causal")
        return norm(h)

    def refine(self, m, R, grad_steps=1):
        """From fog to a settled canvas in R steps. m (B, n, M) -> (the guesses (B, n, K, z), log error std, slot
        states, how much the last step still moved the guesses)."""
        B, n, _ = m.shape
        K, Z = self.cfg.slots, self.cfg.z_dim
        sched = sigmas(R, self.cfg.sigma_min)
        x = m.new_zeros(B, n, K, Z)
        s = m.new_ones(B, n, K, dtype=torch.float32)
        net = self._canvas or self.canvas
        mask = self.canvas.mask(n, m.device)
        y_prev, move = None, m.new_zeros((), dtype=torch.float32)
        for r in range(R):
            with torch.set_grad_enabled(torch.is_grad_enabled() and r >= R - grad_steps):
                y, e, H = net(x, s, m, mask)
            if y_prev is not None:
                move = ((y - y_prev).float().norm(dim=-1) / y_prev.float().norm(dim=-1).clamp_min(1e-6)).mean().detach()
            y_prev = y.detach()
            if r == R - 1:
                break
            nxt = torch.full_like(s, sched[r + 1])
            if self.cfg.adapt:                                   # sure slots drop their noise faster
                nxt = torch.minimum(nxt, e.detach().exp())
            a, a2 = (1 - s * s).clamp_min(0).sqrt(), (1 - nxt * nxt).clamp_min(0).sqrt()
            yd = y.detach().float()
            eps = (x.float() - a[..., None] * yd) / s[..., None].clamp_min(1e-3)     # deterministic DDIM
            x = (a2[..., None] * yd + nxt[..., None] * eps).to(m.dtype)
            s = nxt
        return y, e, H, move

    def think(self, x, passes=None):
        """The stream after the ear (B, T, D) -> (the canvas each chunk's words read (B, n, K, D), the lessons and
        numbers (a dict))."""
        B, T, D = x.shape
        cfg = self.cfg
        n, K = T // cfg.beat, cfg.slots
        z = self.meanings(x)                                                              # (B, n, Z)
        m = self.think_mind(z)
        R = passes or cfg.n_steps(self.training)
        y, e, H, move = self.refine(m, R, cfg.grad_steps if self.training else R)
        P = self.to_word(norm(H))
        if not self.mind_on:
            P = torch.zeros_like(P)
        info = {"mind_move": move}
        # the true future meanings, held still: slot k of boundary c is chunk c + k
        zt = z.detach().float()
        idx = torch.arange(n, device=x.device)[:, None] + torch.arange(K, device=x.device)[None, :]     # (n, K)
        valid = (idx < n).float()
        Y = F.pad(zt, (0, 0, 0, K))[:, idx]                                               # (B, n, K, Z)
        nv = valid.sum().clamp_min(1.0) * B
        honest = (((y.float() - Y) ** 2).mean(-1) * valid).sum() / nv
        info["canvas_honest"] = honest.detach()
        info["canvas_honest_next"] = ((((y[:, :, 0].float() - Y[:, :, 0]) ** 2).mean(-1)).mean()).detach()
        info["canvas_sure"] = (e.detach().exp() * valid).sum() / nv
        if self.training and torch.is_grad_enabled():
            # the diffusion lesson: every slot noised to its own level, some given clean, some pure fog
            u = torch.rand(B, n, K, device=x.device)
            s = torch.rand(B, n, K, device=x.device)
            s = torch.where(u < cfg.known, torch.zeros_like(s), s)
            s = torch.where(u > 1 - cfg.full, torch.ones_like(s), s)
            a = (1 - s * s).clamp_min(0).sqrt()
            xn = (a[..., None] * Y + s[..., None] * torch.randn_like(Y)).to(m.dtype)
            yd, ed, _ = (self._canvas or self.canvas)(xn, s, m, self.canvas.mask(n, x.device))
            graded = valid * (s > 0).float()
            err = ((yd.float() - Y) ** 2).mean(-1)
            ng = graded.sum().clamp_min(1.0)
            diff = (err * graded).sum() / ng
            # how sure: Gaussian likelihood of each slot's own error (the error itself held still)
            def nll(err_, e_):
                return 0.5 * err_.detach() * torch.exp(-2 * e_) + e_
            sure = ((nll(err, ed) * graded).sum() / ng + (nll(((y.float() - Y) ** 2).mean(-1), e) * valid).sum() / nv) / 2
            sg = sigreg(z.reshape(-1, z.shape[-1]), cfg.sig_dirs)
            info["aux"] = cfg.diffuse * diff + cfg.honest * honest + cfg.sure * sure + cfg.sig * sg
            info.update(canvas_diffuse=diff.detach(), canvas_conf=sure.detach(), canvas_sig=sg.detach())
        return P, info

    def _trunk_aux(self, idx, passes=None):
        """(B, T) words -> (vectors after the final norm, 0, 0, None, None, the canvas's lessons and numbers)."""
        T0 = idx.shape[1]
        idx = self._pad(idx)
        T = idx.shape[1]
        e, _ = self._embed(idx)
<<<<<<< HEAD
        if self.cfg.lean:                # Bee's lean way (bee6/lean.py): fused norms, full-width rotation tables
            sdt = LEAN.stream_dtype(self.cfg, e)
            x0 = LEAN.norm(e, sdt, self.cfg.lean_rows)
            ves = [t(idx).to(sdt) for t in self.ve]
            cos, sin = self.rope_cos2[:T], self.rope_sin2[:T]
        else:
            x0 = norm(e)
            ves = [t(idx) for t in self.ve]
            cos, sin = self.rope_cos[:T], self.rope_sin[:T]
=======
        x0 = norm(e)
        ves = [t(idx) for t in self.ve]
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
>>>>>>> origin/main
        keep = {}

        def layer(i, x, ve):
            if str(i) in self.reads:
                x = x + self.reads[str(i)](x, keep["P"])
            x, _, _ = self.blocks[i](x, ve, cos, sin)
            if i == self.cfg.ear_layers - 1:
                keep["P"], keep["info"] = self.think(x, passes)
            return x
        h = self._run(x0, ves, layer)[:, :T0]
        z = h.new_zeros((), dtype=torch.float32)
        return h, z, z, None, None, keep["info"]

    def enable_compile(self):
        """Each word layer, the mind's layers and the denoiser compiled on their own (the denoiser runs several
        times a pass)."""
        import torch._dynamo
        dc = torch._dynamo.config
<<<<<<< HEAD
        # many shapes: 3 window lengths x training / reading x the checks' batch sizes (64 was hit on the first run)
        dc.cache_size_limit = max(getattr(dc, "cache_size_limit", 8), 256)
        if hasattr(dc, "recompile_limit"):
            dc.recompile_limit = max(dc.recompile_limit, 256)
        if hasattr(dc, "accumulated_cache_size_limit"):
            dc.accumulated_cache_size_limit = max(dc.accumulated_cache_size_limit, 2048)
        if hasattr(dc, "accumulated_recompile_limit"):
            dc.accumulated_recompile_limit = max(dc.accumulated_recompile_limit, 2048)
        mode = self.cfg.compile_mode
        kw = LEAN.compile_kw(mode)
        _loss_compile(kw if mode in LEAN.TUNE else {})               # the loss step's kernels tuned with the body's
=======
        dc.cache_size_limit = max(getattr(dc, "cache_size_limit", 8), 64)
        if hasattr(dc, "accumulated_cache_size_limit"):
            dc.accumulated_cache_size_limit = max(dc.accumulated_cache_size_limit, 512)
        mode = self.cfg.compile_mode
        kw = {} if mode in ("", "default") else {"mode": mode}
>>>>>>> origin/main
        for b in self.blocks:
            b.compile(dynamic=False, **kw)
        for r in self.reads.values():
            r.compile(dynamic=False, **kw)
        for lay in self.mind:
            lay.compile(dynamic=False, **kw)
        self._canvas = torch.compile(self.canvas, dynamic=False, **kw)
        self._compiled = None

    # ---------------------------------------------------------------- the trainer's hooks
    @torch.no_grad()
    def loop_report(self, windows, device, batch=16):
        """Does thinking longer help? Held-out next-word loss with 1, 2, the trained number and twice that many
        refining steps."""
        was = self.training
        self.eval()
        R = self.cfg.steps
        out, move = {}, 0.0
        try:
            for p in sorted({1, 2, R, 2 * R}):
                out[p], mv = self._held_out(windows, device, batch, passes=p)
                if p == 2 * R:
                    move = mv
        finally:
            self.train(was)
        return {"pass_loss": out, "adaptive_loss": out[R], "adaptive_passes": float(R), "move_last": move}

    def startup_check(self, device, say=print):
        say(f"[canvas] {self.layout()}; the mind, the canvas and their reads {self.mind_params() / 1e6:.1f}M of "
            f"{self.num_params() / 1e6:.1f}M")
        Bee6.startup_check(self, device, say)
