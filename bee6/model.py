"""Bee6: Bee5 made fast. Still a NEXT-WORD model: it reads text and guesses the very next token, like GPT-2 and every Bee,
and it is scored the same way.

João's ask (Nov 9): a 200M model training toward 3M tokens a second on one B200. Bee5's body and recipe, plus four
speed ideas from the Special Lab, each behind its own switch so its speed and quality effect can be measured:

  idea 2  use the GPU better   q, k and v in one product (fuse_qkv), bigger loss slices (loss_chunk), static shapes
                               everywhere (no padding, no recompiles), CUDA graphs as a switch (graphs)
  idea 3  real FP8             the layers' big products in FP8 (bee6/fp8.py); norms, the word table, the score head,
                               the value tables, gates and the router stay bf16 (fp8 = 0 / 1 / 2)
  idea 4  Mixture of Depths    in every other layer only the ~25% of words the layer's router scores highest go through
                               it; the rest hop past (mod_layers, mod_cap). A small helper learns to guess the router's
                               choice from the word itself, so reading (held-out loss, tests, chat) never looks ahead
  idea 5  bloom                start at 4 layers of 512 and grow, keeping what was learned exactly, to 12 of 1,024
                               (bloom; off by default: plain Bee6 is measured first)

                       Bee5                              Bee6 (defaults)
  body                 12 x 1,024, every speedrun trick  the same
  q, k, v              three products                    one product (same weights, same maths)
  layer products       bf16                              FP8 forward and input gradient, bf16 weight gradient
  layers 3, 5, 7, 9, 11  every word                      the top 25% of words (router), the rest skip
  loss slices          1,024 words                       4,096 words
  parameters           201.3M                            202.0M (Bee5 + 0.66M of routers and helpers)

Not changed: SmolLM2's pieces, Sado, the windows 512 -> 1,024 -> 2,048, the batch ramp, WSD with the square-root
cool-down over the last 30%, NorMuon + Adam groups, the SmolLM2-360M mentor, the early 3-word lesson.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

from bee5.model import (Attention, Bee5, Bee5Config, GATE_IN, MLP, ROPE_TABLE, causal, mtp_weights, norm)
from . import fp8 as FP8

BLOOM = "4x512:0,6x768:0.2,9x1024:0.4,12x1024:0.6"   # João's Seed schedule, 20/20/20/40 of the tokens, at Bee6's sizes
PRED_HIDDEN = 128                                     # the Mixture-of-Depths helper: width -> 128 -> 1


@dataclass
class Bee6Config(Bee5Config):
    fuse_qkv: int = 1            # idea 2: q, k, v in one product
    loss_chunk: int = 4096       # idea 2: words per loss slice (Bee5: 1,024); more = fewer, bigger products, more memory
    graphs: int = 0              # idea 2: CUDA graphs (compile mode reduce-overhead); off until the speed check says so
    fp8: int = 1                 # idea 3: 0 bf16, 1 FP8 forward + input gradient, 2 the weight gradient too
    mod_layers: str = "2,4,6,8,10"   # idea 4: layers (from 0) where only the top words go through ("" = none)
    mod_cap: float = 0.25        # idea 4: the share of words each routed layer takes
    mod_aux: float = 0.01        # idea 4: the helper's lesson (guess the router's pick, causally), its weight
    mod_eval: str = "predict"    # reading: "predict" = the helper decides (causal); "topk" = the router's top 25% (looks ahead)
    bloom: str = ""              # idea 5: "" = off, "on" = BLOOM, or a plan "LxD:at,..." ending at this size
    bloom_stage: int = -1        # which stage of the plan the weights are at (-1 = not started)
    backout_layer: int = -1      # -1 = Bee5's (2/3 of the way); bloom moves it with the layer it was on
    bloom_mod: int = -1          # bloom: whether routed / key-offset layers are wanted (kept even while a stage has none)
    bloom_offset: int = -1

    def __post_init__(self):
        super().__post_init__()
        if self.bloom == "on":
            self.bloom = BLOOM

    def mod_set(self):
        return {int(s) for s in str(self.mod_layers).split(",") if s.strip() != ""}

    def plan(self):
        """[(layers, width, from progress)] of the bloom plan."""
        out = []
        for part in str(self.bloom).split(","):
            if part.strip():
                size, at = part.split(":")
                n, d = size.lower().split("x")
                out.append((int(n), int(d), float(at)))
        return out


def rotary(x, cos, sin):
    """x (B, T, H, hd); cos, sin (T, hd/2) or, for the words a routed layer picked, (B, T, hd/2)."""
    if cos.dim() == 2:
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    else:
        cos, sin = cos[:, :, None, :], sin[:, :, None, :]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos + x2 * sin, x2 * cos - x1 * sin], dim=-1)


class Attention6(Attention):
    """Bee5's attention: q, k, v in one product, the big products in FP8, and a mask for the words a routed layer took."""

    def project(self, x, ve, cos, sin, prev_k=None):
        B, T, _ = x.shape
        H, KV, hd, mode = self.H, self.KV, self.hd, self.cfg.fp8
        if self.cfg.fuse_qkv:
            qkv = FP8.linear(x, torch.cat([self.q.weight, self.k.weight, self.v.weight]), mode)
            q, k, v = qkv.split([H * hd, KV * hd, KV * hd], dim=-1)
        else:
            q, k, v = (FP8.linear(x, m.weight, mode) for m in (self.q, self.k, self.v))
        q = F.rms_norm(q.reshape(B, T, H, hd), (hd,), self.qn)
        k = F.rms_norm(k.reshape(B, T, KV, hd), (hd,), self.kn)
        v = v.reshape(B, T, KV, hd)
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)
        k_raw = k
        if self.offset:
            first = k[:, :1] if prev_k is None else prev_k
            k = torch.where(self.still, torch.cat([first, k[:, :-1]], dim=1), k)
        if ve is not None:
            g = 2 * torch.sigmoid(self.ve_gate(x[..., :GATE_IN])).view(B, T, KV, 1)
            v = v + g * ve.view(B, T, KV, hd)
        return q, k, v, k_raw

    def finish(self, x, y):
        B, T = x.shape[:2]
        if self.gate is not None:
            y = y * torch.sigmoid(self.gate(x[..., :GATE_IN])).view(B, T, self.H, 1)
        return FP8.linear(y.reshape(B, T, self.H * self.hd), self.out.weight, self.cfg.fp8)

    def masked(self, x, ve, cos, sin, keys, state=None):
        """Attention where only the words with keys = True may be looked at (and each word at itself).
        x (B, T, D); keys (B, T) bool; state = (keys, values, last raw key, their key mask) of the words before."""
        q, k, v, k_raw = self.project(x, ve, cos, sin, None if state is None else state[2])
        P = 0 if state is None else state[0].shape[1]
        if state is not None:
            k, v, keys_all = torch.cat([state[0], k], 1), torch.cat([state[1], v], 1), torch.cat([state[3], keys], 1)
        else:
            keys_all = keys
        T = q.shape[1]
        rep = self.H // self.KV
        qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2).repeat_interleave(rep, 1), v.transpose(1, 2).repeat_interleave(rep, 1)
        tri = torch.ones(T, P + T, dtype=torch.bool, device=x.device).tril(P)
        eye = torch.zeros(T, P + T, dtype=torch.bool, device=x.device)
        eye[:, P:] = torch.eye(T, dtype=torch.bool, device=x.device)
        mask = (tri[None] & keys_all[:, None, :]) | eye[None]
        y = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask[:, None])
        return self.finish(x, y.transpose(1, 2)), (k, v, k_raw[:, -1:], keys_all)


class MLP6(MLP):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

    def forward(self, x):
        mode = self.cfg.fp8
        h = FP8.linear(x, self.up.weight, mode)
        if self.kind == "swiglu":
            a, b = h.chunk(2, dim=-1)
            return FP8.linear(F.silu(a) * b, self.down.weight, mode)
        return FP8.linear(F.relu(h).square(), self.down.weight, mode)


class Block6(nn.Module):
    """A Bee5 layer. In a routed layer (Mixture of Depths, Raposo et al. 2024) a router scores every word, the top
    mod_cap of each window go through the layer (attention among themselves, at their own positions, then the MLP),
    their change is scaled by 2 * sigmoid(score) so the router learns, and the other words pass unchanged.

    Picking the top 25% of a window looks at the whole window, later words included. That is fine for learning but not
    for reading, so a small helper learns, on the side, to guess from each word alone (and causally) whether the
    router would pick it. Reading (eval mode: the held-out loss, HellaSwag, LogRes, chat) uses the helper's guess."""

    def __init__(self, cfg: Bee6Config, layer: int):
        super().__init__()
        self.kind = "A"
        self.cfg = cfg
        self.attn = Attention6(cfg, layer)
        self.mlp = MLP6(cfg)
        self.routed = layer in cfg.mod_set()
        if self.routed:
            if self.attn.offset:
                raise ValueError(f"layer {layer} cannot both be routed and shift its keys (the previous word is not there)")
            D = cfg.d_model
            self.router = nn.Linear(D, 1, bias=False)
            self.pred = nn.Sequential(nn.Linear(D, PRED_HIDDEN), nn.ReLU(), nn.Linear(PRED_HIDDEN, 1))
            nn.init.normal_(self.router.weight, std=0.02)

    def delta(self, x, ve, cos, sin):
        """The layer's change to x (every word of x goes through)."""
        y = x + self.attn(norm(x), ve, cos, sin)
        return y + self.mlp(norm(y)) - x

    def dense(self, x, ve, cos, sin):
        y = x + self.attn(norm(x), ve, cos, sin)
        return y + self.mlp(norm(y))

    def k_of(self, T):
        return max(1, min(T, int(round(self.cfg.mod_cap * T))))

    def scores(self, x):
        xn = norm(x)
        return self.router(xn).squeeze(-1).float(), xn

    def guess(self, xn):
        """The helper's logit (B, T) that the router picks each word: from that word's own vector, so causal."""
        return self.pred(xn.detach()).squeeze(-1).float()

    def forward(self, x, ve, cos, sin):
        """-> (x after the layer, (helper's loss, helper's agreement) or None)."""
        if not self.routed:
            return self.dense(x, ve, cos, sin), None
        r, xn = self.scores(x)
        if self.training or self.cfg.mod_eval == "topk":
            if not self.training:
                return self.through_mask(x, ve, cos, sin, r, self.topk_mask(r)), None
            return self.through_topk(x, ve, cos, sin, r, xn)
        return self.through_mask(x, ve, cos, sin, r, self.guess(xn) > 0), None

    def topk_mask(self, r):
        idx = r.topk(self.k_of(r.shape[1]), dim=1).indices
        return torch.zeros_like(r, dtype=torch.bool).scatter_(1, idx, True)

    def through_topk(self, x, ve, cos, sin, r, xn):
        """Learning: the top k words gathered in order, through the layer, scattered back. Fixed shapes."""
        B, T, D = x.shape
        k = self.k_of(T)
        idx = r.topk(k, dim=1).indices.sort(dim=1).values                     # (B, k), in reading order
        pick = torch.zeros_like(r).scatter_(1, idx, 1.0)
        logit = self.guess(xn)
        aux = F.binary_cross_entropy_with_logits(logit, pick)
        agree = ((logit > 0).float() == pick).float().mean().detach()
        xs = x.gather(1, idx[..., None].expand(B, k, D))
        vs = None if ve is None else ve.gather(1, idx[..., None].expand(B, k, ve.shape[-1]))
        cs = cos[idx] if cos.dim() == 2 else cos.gather(1, idx[..., None].expand(B, k, cos.shape[-1]))
        sn = sin[idx] if sin.dim() == 2 else sin.gather(1, idx[..., None].expand(B, k, sin.shape[-1]))
        d = self.delta(xs, vs, cs, sn)
        w = 2 * torch.sigmoid(r.gather(1, idx)).to(d.dtype)[..., None]
        return x.scatter_add(1, idx[..., None].expand(B, k, D), (d * w).to(x.dtype)), (aux, agree)

    def through_mask(self, x, ve, cos, sin, r, pick, state=None):
        """Reading: every word computed, but a word not picked keeps x and no word may look at it. The same numbers as
        gathering the picked words (tested), any number of picks per window."""
        y, _ = self._masked_delta(x, ve, cos, sin, pick, state)
        w = 2 * torch.sigmoid(r).to(y.dtype)[..., None]
        return x + (y * w * pick[..., None].to(y.dtype)).to(x.dtype)

    def _masked_delta(self, x, ve, cos, sin, pick, state=None):
        a, st = self.attn.masked(norm(x), ve, cos, sin, pick, state)
        y = x + a
        return y + self.mlp(norm(y)) - x, st

    def step(self, x, ve, cos, sin, state, looks=False):
        """Reading on from a state (chat, samples). Routed layers keep, with their keys and values, which words were picked."""
        if not self.routed:
            y, state, info = self.attn.step(norm(x), ve, cos, sin, state, looks)
            x = x + y
            return x + self.mlp(norm(x)), state, info
        r, xn = self.scores(x)
        pick = self.guess(xn) > 0
        d, state = self._masked_delta(x, ve, cos, sin, pick, state)
        w = 2 * torch.sigmoid(r).to(d.dtype)[..., None]
        return x + (d * w * pick[..., None].to(d.dtype)).to(x.dtype), state, None


class Bee6(Bee5):
    Block = Block6
    fresh_opt_on_grow = True     # the trainer makes a new optimizer after a bloom (the weights are new tensors)

    def __init__(self, cfg: Bee6Config):
        if cfg.bloom and cfg.bloom_stage < 0:
            from .bloom import stage_config
            stage_config(cfg, 0)
        super().__init__(cfg)
        self.loss_chunk = cfg.loss_chunk
        if cfg.bloom:                # bloom: plain skip weights (a new skip starts at exactly 0 and still learns)
            n_skip = len(self.skip_gate)
            del self.skip_lam
            self.skip_w = nn.Parameter(torch.full((n_skip,), 2 * torch.sigmoid(torch.tensor(-1.5)).item()))
        self.register_buffer("bloom_at", torch.tensor(max(0, cfg.bloom_stage)), persistent=True)

    # ---------------------------------------------------------------- names
    def layout(self):
        c = self.cfg
        mods = sorted(i for i in c.mod_set() if i < c.n_layer)
        return (f"{c.n_layer} layers of {c.d_model} ({c.attn_heads} query / {c.attn_kv_heads} key-value heads), "
                f"{'ReLU²' if c.mlp == 'relu2' else 'SwiGLU'} {c.mlp_hidden}, {c.n_ve} value tables"
                + (f", layers {','.join(str(i + 1) for i in mods)} take the top {c.mod_cap:.0%} of words" if mods else "")
                + (", FP8 products" if c.fp8 else ""))

    def depth_text(self):
        return f"{self.cfg.n_layer} layers of {self.cfg.d_model}"

    @property
    def depth(self):
        return self.cfg.n_layer

    def speed_text(self):
        c = self.cfg
        return " · ".join(s for s in ["q/k/v fused" if c.fuse_qkv else "", f"FP8 mode {c.fp8}" if c.fp8 else "bf16",
                                      f"MoD {len(c.mod_set())} layers at {c.mod_cap:.0%}" if c.mod_set() else "no MoD",
                                      f"bloom {c.bloom}" if c.bloom else "", "CUDA graphs" if c.graphs else ""] if s)

    # ---------------------------------------------------------------- the body
    def _skip(self, j, x0):
        g = 2 * torch.sigmoid(self.skip_gate[j](x0[..., :GATE_IN]))
        if self.cfg.bloom:
            return self.skip_w[j] * 0.5 * g
        return torch.sigmoid(self.skip_lam[j]) * g

    def _trunk_aux(self, idx):
        """(B, T) words -> (vectors after the final norm, the helpers' loss, their agreement with the routers)."""
        T = idx.shape[1]
        e, _ = self._embed(idx)
        x0 = norm(e)
        ves = [t(idx) for t in self.ve]
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        aux = []

        def layer(i, x, ve):
            x, a = self.blocks[i](x, ve, cos, sin)
            if a is not None:
                aux.append(a)
            return x
        h = self._run(x0, ves, layer)
        if aux:
            return h, torch.stack([a for a, _ in aux]).mean(), torch.stack([g for _, g in aux]).mean()
        z = h.new_zeros((), dtype=torch.float32)
        return h, z, z

    def _trunk(self, idx):
        return self._trunk_aux(idx)[0]

    def enable_compile(self):
        import torch._dynamo
        cfg = torch._dynamo.config
        cfg.cache_size_limit = max(getattr(cfg, "cache_size_limit", 8), 64)        # 3 window lengths x 4 bloom stages
        if hasattr(cfg, "accumulated_cache_size_limit"):
            cfg.accumulated_cache_size_limit = max(cfg.accumulated_cache_size_limit, 256)
        mode = "reduce-overhead" if self.cfg.graphs else self.cfg.compile_mode
        kw = {} if mode in ("", "default") else {"mode": mode}
        self._compiled = torch.compile(self._trunk_aux, dynamic=False, **kw)

    def losses(self, idx, targets, mentor=None, kd_weight=0.0):
        if self.token_select is not None:
            raise SystemExit("Bee6 does not take --select yet (it joins the recipe once the A/B on Plain-100M is in)")
        if self._compiled is not None and self.training:
            if self.cfg.graphs:
                torch.compiler.cudagraph_mark_step_begin()
            h, aux, agree = self._compiled(idx)
        else:
            h, aux, agree = self._trunk_aux(idx)
        th = tW = None
        if mentor is not None and kd_weight > 0:
            th, tW = mentor.hidden(idx, self.doc_mask(idx))
        w = self.mtp()
        tg = targets[..., None]
        if len(w) > 1:
            pad = torch.full_like(targets[:, :len(w) - 1], -1)
            ext = torch.cat([targets, pad], 1)
            tg = torch.stack([ext[:, k:k + targets.shape[1]] for k in range(len(w))], -1)
        from bee5.loss import head_loss, plain_loss
        f = head_loss if self.fused_loss else plain_loss
        loss, ce, kd = f(h, self.embed.weight, tg, w, self.cfg, th, tW, kd_weight, self.loss_chunk)
        stats = {"ce": ce}
        if th is not None:
            stats["kd"] = kd
        if self.cfg.mod_set() and self.training:
            loss = loss + self.cfg.mod_aux * aux
            stats["mod_agree"] = agree
            self._agree = agree
        return loss, stats

    def thought_report(self):
        """For the trainer's log: how often the helper guesses the router's pick (reading uses the helper)."""
        a = getattr(self, "_agree", None)
        return {"mod_agree": None if a is None else float(a)}

    # ---------------------------------------------------------------- the trainer's hooks
    def startup_check(self, device, say=print):
        """FP8: the same batch in bf16 and in FP8 must agree, or Bee6 trains in bf16 (bee6/fp8.py)."""
        if not self.cfg.fp8:
            return
        if not FP8.hardware(device):
            say("[fp8] this device has no FP8 products: Bee6 runs its layers in bf16")
            self.cfg.fp8 = 0
            return
        try:
            ok, gap, cos = FP8.check(self, device, say)
        except Exception as e:
            say(f"[fp8] FP8 failed on this GPU ({type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}); "
                f"Bee6 runs its layers in bf16")
            self.cfg.fp8 = 0
            return
        if ok:
            say(f"[fp8] the layers' products run in FP8 (mode {self.cfg.fp8}); vs bf16 on a test batch: loss {gap:.2%} apart, "
                f"gradients {cos:.3f} alike")
        else:
            say(f"[fp8] check failed (loss {gap:.2%} apart, gradients {cos:.3f} alike); Bee6 runs its layers in bf16")
            self.cfg.fp8 = 0

    def grow_to(self, progress: float) -> bool:
        """Bloom: grow to the plan's stage for this point of the run. True if it grew."""
        if not self.cfg.bloom:
            return False
        from .bloom import stage_at, bloom_next
        grew = False
        while self.cfg.bloom_stage < stage_at(self.cfg, progress):
            bloom_next(self)
            grew = True
        if grew and self._compiled is not None:
            self.enable_compile()
        return grew

    def load_state_dict(self, state, strict=True, assign=False):
        """A checkpoint taken after a bloom: grow to its stage first, then load."""
        at = state.get("bloom_at")
        if self.cfg.bloom and at is not None:
            from .bloom import bloom_next
            while self.cfg.bloom_stage < int(at):
                bloom_next(self)
        return super().load_state_dict(state, strict=strict, assign=assign)
