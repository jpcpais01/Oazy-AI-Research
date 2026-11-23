"""Think-135: Bee-135 whole, plus a mind that searches thoughts. The mouth is NEXT-TOKEN; the mind is diffusion.

João and Claude, Nov 23 2026, after Loop-135 and Canvas-3 were dropped: "thinking is search in meaning space, words
come once, at the end". Diffusion did nothing in the canvas because it was judged by the next real word, where a blur
of all futures is already the best guess. Its job is to CHOOSE one sharp future among many. So:

    words ─► Bee-135's first `thought_at` layers                         (nothing changed)
               │   the states of a sentence's pieces, squeezed to `thought_dim` numbers and averaged, at size 1 = the
               │   sentence's THOUGHT (found, not taught: whatever makes the sentence easiest to write)
               ▼
             every piece reads the thought of the sentence its NEXT word belongs to (so a sentence's first word is
             already written under its thought), with noise of size `thought_noise` added (the same noise for the
             whole sentence): a small, noisy vector holds the gist, not the wording.
             `thought_drop` of the sentences get no thought (so the words can also be written alone)
               ▼
             the rest of Bee-135's layers ─► the next word

    the mind: at a sentence's end, a small denoiser (an MLP) learns to make the NEXT sentence's thought out of noise in
    a few steps, from Bee's final state one piece earlier (the last state made before the new thought is read) and the
    early state of the closing piece. Other noise lands on another thought: many candidates, almost free.
    A sureness head learns how wrong each made thought turned out.

All 24 layers stay in the word path (the canvas took half away). One sentence = up to `thought_cap` pieces, cut at
. ! ? or a new line. New weights: ~1.6M. Compute: Bee-135's, +~3%.

Reading has three ways (model._tmode): "null" (the default: no thoughts, an honest plain next-word model, what the
trainer's tests see), "oracle" (each sentence's true thought: not honest for a sentence not yet said) and given
thoughts (model._tgiven = (z (B, T, dim), mask (B, T)): where mask, the piece reads z; elsewhere the true thought).
The real test is gnosis2/thinkab.py: answer by meaning, with 1, 4, 16, 64 thoughts.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from bee5.model import norm

STEPS = 8


class Mind(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from canvas.model import sentence_ends
        D, dz, W = cfg.d_model, cfg.thought_dim, cfg.thought_width
        self.dz, self.W = dz, W
        self.to_z = nn.Linear(D, dz, bias=False)          # a state -> a thought
        self.z_in = nn.Linear(dz, D, bias=False)          # a thought -> the stream
        self.cond = nn.Linear(D, W)                       # Bee's final state one piece before a sentence end -> the denoiser
        self.cond_b = nn.Linear(D, W, bias=False)         # and the closing piece's own state after `thought_at` layers
        self.x_in = nn.Linear(dz, W)
        self.t_in = nn.Linear(32, W)
        self.blocks = nn.ModuleList(nn.Sequential(nn.Linear(W, 2 * W), nn.SiLU(), nn.Linear(2 * W, W))
                                    for _ in range(cfg.thought_blocks))
        self.out = nn.Linear(W, dz)
        self.sure = nn.Sequential(nn.Linear(2 * W + dz, W), nn.SiLU(), nn.Linear(W, 1))
        self.register_buffer("ends_tab", sentence_ends(cfg.vocab_size))      # kept in the checkpoint
        with torch.no_grad():
            nn.init.normal_(self.to_z.weight, std=D ** -0.5)
            nn.init.zeros_(self.z_in.weight)              # starts as nothing: the model starts exactly as Bee-135
            for b in self.blocks:
                nn.init.zeros_(b[2].weight)
                nn.init.zeros_(b[2].bias)

    def temb(self, t):
        """t (..., 1) in 0..1 -> (..., W)."""
        f = torch.exp(torch.arange(16, device=t.device, dtype=torch.float32) * (-math.log(1000.0) / 15))
        a = t.float() * 1000.0 * f
        return self.t_in(torch.cat([a.sin(), a.cos()], -1))

    def denoise(self, xt, c, t):
        """The noisy thought xt (..., dz), the context c (..., W), the noise level t (..., 1) -> (the clean thought
        it guesses, the log of how wrong it thinks that guess is)."""
        te = self.temb(t)
        h = self.x_in(xt) + c + te
        for b in self.blocks:
            h = h + b(norm(h))
        x0 = self.out(norm(h))
        s = self.sure(torch.cat([c, te, x0.detach()], -1)).squeeze(-1)
        return x0, s

    @torch.no_grad()
    def think(self, c, n=1, steps=STEPS, gen=None):
        """The context at a sentence end (B, W; see `context`) -> n thoughts each (B, n, dz) and the log of how wrong
        each one thinks it is (B, n). Each starts as other noise; the same few steps clean them all at once."""
        B = c.shape[0]
        c = c.float()[:, None].expand(B, n, self.W)
        x = torch.randn(B, n, self.dz, device=c.device, generator=gen)
        ts = torch.linspace(1.0, 0.0, steps + 1, device=c.device)
        x0 = s = None
        for i in range(steps):
            t = ts[i].expand(B, n, 1)
            a, sg = torch.cos(ts[i] * math.pi / 2), torch.sin(ts[i] * math.pi / 2)
            x0, s = self.denoise(x, c, t)
            eps = (x - a * x0) / sg.clamp_min(1e-4)
            a2, s2 = torch.cos(ts[i + 1] * math.pi / 2), torch.sin(ts[i + 1] * math.pi / 2)
            x = a2 * x0 + s2 * eps
        return norm(x0), s


def context(model, h, ex):
    """What the mind knows at every place, were a sentence to end there (B, T, W): Bee's final state one piece earlier
    and the piece's own early state. Place 0 has no earlier piece: only its own state."""
    hp = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], 1).float()
    return model.mind.cond(hp) + ex["c2"]


def check(cfg):
    if not 1 <= cfg.thought_at < cfg.n_layer:
        raise ValueError(f"thought_at must be a layer from 1 to {cfg.n_layer - 1}")
    if cfg.loop_layers or cfg.hint_rows or cfg.book:
        raise ValueError("thought: not with the loop, the phrase hint or the memory table")


def begin(model, idx):
    """Where sentences end. -> {"b": (B, T) a sentence ends at this piece, "end": (B, T) the place of the last piece
    of each piece's own sentence}. A sentence ends at . ! ? a new line or the end of a text, after `thought_cap` pieces
    at the latest, and at the last piece of the window (a sentence cut by the window: its thought is of what is there)."""
    B, T = idx.shape
    g = torch.arange(T, device=idx.device).expand(B, T)
    forced = getattr(model, "_tbounds", None)
    e = model.mind.ends_tab[idx] if forced is None or model.training else forced
    cm = torch.where(e, g, torch.full_like(g, -1)).cummax(1).values
    prev = torch.cat([torch.full_like(g[:, :1], -1), cm[:, :-1]], 1)             # the last end before this piece
    b = e | ((g - prev) % model.cfg.thought_cap == 0) | (g == T - 1)
    end = torch.where(b, g, torch.full_like(g, T - 1)).flip(1).cummin(1).values.flip(1)
    cb = torch.where(b, g, torch.full_like(g, -1)).cummax(1).values
    prev = torch.cat([torch.full_like(g[:, :1], -1), cb[:, :-1]], 1)             # the last sentence end before this piece
    return {"b": b, "end": end, "prev": prev}


def before(model, i, x, keep):
    """Called before every layer: before layer `thought_at`, every piece reads its sentence's thought."""
    cfg = model.cfg
    if i != cfg.thought_at:
        return x
    m = model.mind
    B, T, _ = x.shape
    cs = m.to_z(x.float()).cumsum(1)                                             # a thought = the mean over the sentence's
    pv = keep["prev"]                                                            # pieces (so far), squeezed, of size 1
    z = norm(cs - torch.where(pv[..., None] >= 0, cs.gather(1, pv.clamp(min=0)[..., None].expand(B, T, m.dz)),
                              torch.zeros_like(cs)))                             # (B, T, dz): read at sentence ends
    keep["extra"] = {"z": z, "c2": m.cond_b(x.float()), "b": keep["b"], "end": keep["end"]}
    rend = torch.cat([keep["end"][:, 1:], keep["end"][:, -1:]], 1)               # the end of the NEXT word's sentence
    mode = "train" if model.training else getattr(model, "_tmode", "null")
    given = None if model.training else getattr(model, "_tgiven", None)
    if mode == "null" and given is None:
        return x
    k = (1.0 + cfg.thought_noise ** 2) ** -0.5
    ei = rend[..., None].expand(B, T, m.dz)
    zn = ((z + cfg.thought_noise * torch.randn_like(z)) * k).gather(1, ei)       # one noise for a whole sentence
    if given is not None:
        gz, gm = given
        zn = torch.where(gm[..., None], ((gz.float() + cfg.thought_noise * torch.randn_like(gz.float())) * k), zn)
        if mode == "null":
            zn = zn * gm[..., None].to(zn.dtype)
    add = m.z_in(zn)
    if mode == "train" and cfg.thought_drop > 0:
        drop = (torch.rand(B, T, device=x.device) < cfg.thought_drop).gather(1, rend)
        add = add * (~drop)[..., None].to(add.dtype)
    return x + add.to(x.dtype)


def lesson(model, h, ex):
    """The mind's lesson: from Bee's final state at a sentence end, make the next sentence's thought out of noise.
    -> {"aux": the weighted loss, numbers for the log}."""
    cfg, m = model.cfg, model.mind
    B, T, D = h.shape
    z, b, end = ex["z"], ex["b"], ex["end"]
    M = max(1, T // cfg.thought_every)
    g = torch.arange(T, device=h.device).expand(B, T)
    ok = b & (g < T - 1) & (g > 0)                                               # a sentence follows
    p = (ok.float() + torch.rand(B, T, device=h.device)).topk(M, dim=1).indices  # M sentence ends a row, at random
    w = ok.gather(1, p).float()
    nxt = end.gather(1, (p + 1).clamp(max=T - 1))                                # the next sentence's last piece
    tgt = z.gather(1, nxt[..., None].expand(B, M, m.dz)).detach()
    c = context(model, h, ex).gather(1, p[..., None].expand(B, M, m.W))
    t = torch.rand(B, M, 1, device=h.device)
    a, s = torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)
    x0, sure = m.denoise(a * tgt + s * torch.randn_like(tgt), c, t)
    err = (x0 - tgt).pow(2).mean(-1)
    n = w.sum().clamp_min(1.0)
    diff = (err * w).sum() / n
    sl = ((sure - (err.detach() + 1e-4).log()).pow(2) * w).sum() / n
    blur = ((c.new_zeros(()) + tgt).pow(2).mean(-1) * w).sum() / n               # the error of guessing nothing: 1
    return {"aux": cfg.thought_w * diff + 0.1 * sl, "thought_err": diff.detach(), "thought_sure": sl.detach(),
            "thought_sentences": ok.float().sum(1).mean().detach(), "thought_blur": blur.detach()}


class reading:
    """with reading(model, mode="oracle", given=(z, mask), bounds=b): ...   sets how the model reads, then puts it back."""

    def __init__(self, model, mode="null", given=None, bounds=None):
        self.m, self.new = model, (mode, given, bounds)

    def __enter__(self):
        m = self.m
        self.old = (getattr(m, "_tmode", "null"), getattr(m, "_tgiven", None), getattr(m, "_tbounds", None))
        m._tmode, m._tgiven, m._tbounds = self.new
        return m

    def __exit__(self, *a):
        self.m._tmode, self.m._tgiven, self.m._tbounds = self.old


@torch.no_grad()
def held_out(model, windows, device, batch, mode):
    """Held-out loss: "null" (no thoughts), "oracle" (the true thoughts: not honest) or "made" (every sentence reads
    ONE thought the mind made at the sentence before, from a first pass with the true thoughts of what was said)."""
    x_all = torch.as_tensor(windows).long().to(device)
    tot, n = 0.0, 0
    for i in range(0, len(x_all), batch):
        xb = x_all[i:i + batch]
        x = xb[:, :-1]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if mode == "made":
                with reading(model, "oracle"):
                    model._trunk_aux(x)
                h, ex = model._cap
                B, T, D = h.shape
                zs, _ = model.mind.think(context(model, h, ex).reshape(B * T, -1), 1)   # a thought from every place
                zs = zs.view(B, T, -1)
                g = torch.arange(T, device=device).expand(B, T)
                src = torch.where(ex["b"], g, torch.full_like(g, -1)).cummax(1).values   # the last sentence end so far
                has = src > 0                                                    # the first sentence has no thought
                z = zs.gather(1, src.clamp(min=0)[..., None].expand(B, T, zs.shape[-1]))
                with reading(model, "null", given=(z, has)):
                    lg = model.scores(model._trunk_aux(x)[0]).float()
            else:
                with reading(model, mode):
                    lg = model.scores(model._trunk_aux(x)[0]).float()
        tot += float(F.cross_entropy(lg.transpose(1, 2), xb[:, 1:], reduction="none").mean(1).sum())
        n += len(xb)
    return tot / max(1, n)


@torch.no_grad()
def report(model, windows, device, batch=16):
    """The trainer's [loop] line, read here as: 0 = no thoughts (plain), 1 = one made thought a sentence (honest),
    9 = the true thoughts (not honest: the room a perfect mind would have)."""
    was = model.training
    model.eval()
    try:
        out = {0: held_out(model, windows, device, batch, "null"), 1: held_out(model, windows, device, batch, "made"),
               9: held_out(model, windows, device, batch, "oracle")}
    finally:
        model.train(was)
    return {"pass_loss": out, "adaptive_loss": out[1], "adaptive_passes": 1.0, "move_last": 0.0}
