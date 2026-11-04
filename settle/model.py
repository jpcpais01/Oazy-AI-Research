"""Settle: a language model that thinks by letting a 2D sheet of cells settle.

Plain picture
-------------
The hidden state is a sheet. Each *column* is one word (token) of the text and
each *row* is a scale. Every cell holds a small vector (``channels`` numbers).

    row 7  (looks 128 and 256 words back)  . . . . . . . . . .
    ...                                      . . . . . . . . . .
    row 1  (looks 2 and 4 words back)       . . . . . . . . . .
    row 0  (looks 1 and 2 words back)       . . . . . . . . . .
                                             w0 w1 w2 ...    wt  <- frontier

One small learned rule (the same few weights everywhere, like the physics of a
cellular automaton) updates every cell from its neighbours:

* itself,
* two cells to its left *in its own row*, at that row's distance
  (row r looks 2^r and 2*2^r columns back; this is the multi-scale field and
  the long-range links: upper rows are long-range, lower rows are local),
* the cells directly above and below it in the same column,
* the word that was written into its column (the token embedding), and
* the *gist*: a slow field that drifts along the text (a leaky running
  average of the sheet, at four time scales). This is the "mood" of the text.

The rule is applied again and again until nothing changes any more (the sheet
has *settled*, an equilibrium, like Deep Equilibrium models). How many steps a
column needs is the amount of thinking that word needed, so compute adapts.

Two-way influence without cheating
----------------------------------
The main sheet is causal: a cell only listens to cells on its left, so the
future can never leak into the past, and the whole sheet can be trained in
parallel. Two-way influence happens in the *lens*: to predict the next word,
the last ``lens_width`` columns are copied into a small workspace where
influence flows both ways (the newest word re-shapes how the previous few
words are read, and they in turn re-shape it). The lens settles too, and the
next word is read out from its frontier column. Every position gets its own
lens over its own past only, so training cannot see the future. The main
sheet is ink (it never changes once settled); the lens is pencil.

Generation keeps the sheet: each new word adds one column which settles
against the already-settled past, so nothing restarts and the cost of a new
word does not grow with the length of the text.

Training uses the "phantom gradient" trick from equilibrium models: settle
without gradients until converged (or a step cap), then run a few more steps
with gradients. Memory stays constant however long the settling takes.
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, asdict, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SettleConfig:
    vocab_size: int = 50257
    d_embed: int = 384           # token embedding size (tied with the readout)
    rows: int = 8                # scales; row r looks 2^r columns back
    channels: int = 128          # numbers per cell
    hidden: int = 1024           # width of the local rule's hidden layer
    gist_dim: int = 128          # slow field size (split over the time scales)
    gist_decays: tuple = (0.9, 0.97, 0.99, 0.997)
    link_heads: int = 4          # long-range links: content-addressed reads into any earlier column (0 = off)
    link_dim: int = 32           # size of each link
    lens_width: int = 6          # how many recent columns the two-way lens re-reads
    lens_hidden: int = 512
    settle_cap: int = 24         # most steps the sheet may take
    settle_tol: float = 2e-3     # a column has settled when its RMS change is below this
    grad_steps: int = 3          # steps with gradients after settling (phantom gradient)
    lens_cap: int = 6
    lens_grad_steps: int = 2
    max_len: int = 256           # training context length
    full_grad: bool = True       # learn through every settle step of the sheet (better, more memory) instead of
                                 # only the last few ("phantom gradient", constant memory)
    lens_full_grad: bool = False # the same for the lens (did not help in tests)
    res_weight: float = 0.05     # penalty for columns still moving more than the tolerance at the end
    effort_weight: float = 0.002 # small steady push towards settling sooner (per tolerance unit)

    def to_dict(self):
        d = asdict(self)
        d["gist_decays"] = list(self.gist_decays)
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        if "gist_decays" in d:
            d["gist_decays"] = tuple(d["gist_decays"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _shift_right(x: torch.Tensor, n: int, dim: int) -> torch.Tensor:
    """x shifted by n along dim towards higher indices, zero filled (the 'left neighbour n back')."""
    size = x.shape[dim]
    if n >= size:
        return torch.zeros_like(x)
    pad_shape = list(x.shape)
    pad_shape[dim] = n
    return torch.cat([x.new_zeros(pad_shape), x.narrow(dim, 0, size - n)], dim=dim)


class Settle(nn.Module):
    def __init__(self, cfg: SettleConfig):
        super().__init__()
        self.cfg = cfg
        R, C, G = cfg.rows, cfg.channels, cfg.gist_dim
        assert G % len(cfg.gist_decays) == 0
        self.dil = [2 ** r for r in range(R)]

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_embed)
        self.inj_norm = nn.LayerNorm(cfg.d_embed)
        self.inject = nn.Linear(cfg.d_embed, R * C)          # writes the word into its column (row-specific)

        # slow field (gist): leaky averages at several time scales
        self.gist_in = nn.Linear(C, G)
        d = torch.tensor(cfg.gist_decays, dtype=torch.float32)
        self.gist_logit = nn.Parameter(torch.log(d / (1 - d)))  # learned decays

        # the local rule: [self, left1, left2, up, down, word, gist] -> (candidate, gate)
        # long-range links: each column sends out a few "wires" that land on whichever earlier
        # columns match what it is looking for (one small attention read per step, not stacked layers)
        self.n_link = cfg.link_heads * cfg.link_dim
        if cfg.link_heads:
            self.link_qkv = nn.Linear(C, 3 * self.n_link)
        rin = 6 * C + G + self.n_link
        self.rule1 = nn.Linear(rin, cfg.hidden)
        self.row_bias = nn.Parameter(torch.zeros(R, cfg.hidden))
        self.rule2 = nn.Linear(cfg.hidden, 2 * C)

        # the two-way lens: [self, left, right, up, down, anchor(ink), gist] -> (candidate, gate)
        W = cfg.lens_width
        self.lens1 = nn.Linear(6 * C + G, cfg.lens_hidden)
        self.lens_bias = nn.Parameter(torch.zeros(W, R, cfg.lens_hidden))
        self.lens2 = nn.Linear(cfg.lens_hidden, 2 * C)

        self.read_norm = nn.LayerNorm(R * C)
        self.readout = nn.Linear(R * C, cfg.d_embed)

        self.apply(self._init)
        for lin in (self.rule2, self.lens2):
            nn.init.normal_(lin.weight, std=0.01)
            nn.init.zeros_(lin.bias)
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
        if non_embedding:
            n -= self.embed.weight.numel()
        return n

    def enable_compile(self):
        self._compiled = (torch.compile(self._main_step), torch.compile(self._lens_step))

    # ------------------------------------------------------------------ pieces
    def _words(self, idx):
        e = self.inj_norm(self.embed(idx))
        B, T = idx.shape
        return self.inject(e).view(B, T, self.cfg.rows, self.cfg.channels)

    def _decays(self):
        return torch.sigmoid(self.gist_logit.float())          # (nG,)

    def _gist_parallel(self, z):
        """Causal leaky averages of the sheet: g_t = sum_{s<=t} (1-l) l^(t-s) u_s, per time scale."""
        B, T, R, C = z.shape
        u = self.gist_in(z.mean(2)).float()                      # (B,T,G)
        lam = self._decays()                                     # (nG,)
        nG = lam.numel()
        t = torch.arange(T, device=z.device, dtype=torch.float32)
        diff = (t[:, None] - t[None, :])                         # (T,T)
        mask = diff >= 0
        logl = torch.log(lam)[:, None, None]
        M = torch.where(mask, torch.exp(diff.clamp(min=0) * logl), torch.zeros((), device=z.device))
        M = M * (1 - lam)[:, None, None]                         # (nG,T,T)
        u = u.view(B, T, nG, -1)
        return torch.einsum("gts,bsgc->btgc", M, u).reshape(B, T, -1)

    def _apply_rule(self, z, x, lin1, bias, lin2):
        h = F.gelu(lin1(x) + bias)
        out = lin2(h).float()
        cand, gate = out.chunk(2, dim=-1)
        a = torch.sigmoid(gate)
        return z + a * (torch.tanh(cand) - z)

    def _main_step(self, z, words, gist):
        """One update of the whole causal sheet. z, words: (B,T,R,C); gist: (B,T,G)."""
        R = self.cfg.rows
        l1 = torch.stack([_shift_right(z[:, :, r], self.dil[r], 1) for r in range(R)], 2)
        l2 = torch.stack([_shift_right(z[:, :, r], 2 * self.dil[r], 1) for r in range(R)], 2)
        up = torch.cat([z[:, :, 1:], torch.zeros_like(z[:, :, :1])], 2)
        down = torch.cat([torch.zeros_like(z[:, :, :1]), z[:, :, :-1]], 2)
        g = gist[:, :, None, :].expand(-1, -1, R, -1).to(z.dtype)
        parts = [z, l1, l2, up, down, words, g]
        if self.n_link:
            B, T = z.shape[:2]
            H, dk = self.cfg.link_heads, self.cfg.link_dim
            q, k, v = self.link_qkv(z.mean(2)).view(B, T, 3, H, dk).permute(2, 0, 3, 1, 4)
            link = F.scaled_dot_product_attention(q, k, v, is_causal=True)          # (B,H,T,dk)
            link = link.transpose(1, 2).reshape(B, T, 1, H * dk).expand(-1, -1, R, -1).to(z.dtype)
            parts.append(link)
        x = torch.cat(parts, -1)
        return self._apply_rule(z, x, self.rule1, self.row_bias, self.rule2)

    def _lens_step(self, L, anchor, boundary, gist, valid):
        """One update of the two-way lens. L, anchor: (N,W,R,C); boundary: (N,R,C); gist (N,G); valid (N,W)."""
        W, R = L.shape[1], L.shape[2]
        left = torch.cat([boundary[:, None], L[:, :-1]], 1)
        right = torch.cat([L[:, 1:], torch.zeros_like(L[:, :1])], 1)
        up = torch.cat([L[:, :, 1:], torch.zeros_like(L[:, :, :1])], 2)
        down = torch.cat([torch.zeros_like(L[:, :, :1]), L[:, :, :-1]], 2)
        g = gist[:, None, None, :].expand(-1, W, R, -1).to(L.dtype)
        x = torch.cat([L, left, right, up, down, anchor, g], -1)
        Ln = self._apply_rule(L, x, self.lens1, self.lens_bias, self.lens2)
        return Ln * valid[:, :, None, None]

    def _read(self, frontier, idx):
        """frontier: (..., R, C), idx: the current words (...) -> logits for the next word.

        The current word's own embedding is added on a short skip path (like a transformer's
        residual stream), so simple word-to-word habits are learned at once and the sheet
        can spend its effort on context."""
        h = self.readout(self.read_norm(frontier.flatten(-2))) + self.embed(idx)
        return F.linear(h, self.embed.weight)

    # ------------------------------------------------------------------ parallel (training) pass
    def settle_sheet(self, words, record=False):
        """Settle the causal sheet for all columns at once. Returns z (with grad), stats.

        Each column settles on its own: once its change in one step is below
        ``settle_tol`` it freezes (stops thinking) and keeps that state. The
        loop ends when every column has frozen or the cap is reached. Then
        ``grad_steps`` more steps run for every column, with gradients.
        Generation (``feed``) repeats exactly this schedule column by column,
        so the two give the same numbers.
        """
        cfg = self.cfg
        step = self._compiled[0] if self._compiled else self._main_step
        B, T, R, C = words.shape
        z = torch.zeros(B, T, R, C, device=words.device, dtype=torch.float32)
        frozen = torch.zeros(B, T, device=words.device, dtype=torch.bool)
        col_steps = torch.full((B, T), cfg.settle_cap, device=words.device, dtype=torch.long)
        frames, curve = [], []
        n_steps = 0
        with (contextlib.nullcontext() if cfg.full_grad and torch.is_grad_enabled() else torch.no_grad()):
            wd = words if cfg.full_grad else words.detach()
            for k in range(cfg.settle_cap):
                zn = step(z, wd, self._gist_parallel(z))
                ch = (zn - z).pow(2).mean((2, 3)).sqrt()          # (B,T) RMS change per column
                z = torch.where(frozen[:, :, None, None], z, zn)
                newly = (~frozen) & (ch <= cfg.settle_tol)
                col_steps = torch.where(newly, torch.full_like(col_steps, k + 1), col_steps)
                if record:
                    frames.append(z[0].clone())
                    curve.append(torch.where(frozen[0], torch.zeros_like(ch[0]), ch[0]))
                frozen = frozen | newly
                n_steps = k + 1
                if bool(frozen.all()):
                    break
        if not cfg.full_grad:
            z = z.detach()
        z_prev = z
        for _ in range(cfg.grad_steps):
            z_prev = z
            z = step(z, words, self._gist_parallel(z))
        # "please settle": how much each column still moves in the last step, in units of the tolerance
        r = (z - z_prev).pow(2).mean((2, 3)).add(1e-12).sqrt() / cfg.settle_tol
        res = F.relu(r - 1).mean()
        effort = r.mean()
        stats = {
            "steps": n_steps,
            "col_steps": col_steps,                              # (B,T) thinking steps per word
            "cap_frac": (~frozen).float().mean(),
            "res": res,
            "effort": effort,
        }
        if record:
            stats["frames"] = frames
            stats["curve"] = curve
        return z, stats

    def lens_windows(self, z):
        """Build every position's lens window over its own past. Returns anchor (B,T,W,R,C), boundary, valid."""
        B, T, R, C = z.shape
        W = self.cfg.lens_width
        zp = torch.cat([z.new_zeros(B, W, R, C), z], 1)          # W zero columns on the left
        # window for position t is columns t-W+1..t  ->  padded indices t+1..t+W
        idx = torch.arange(T, device=z.device)[:, None] + torch.arange(1, W + 1, device=z.device)[None, :]
        anchor = zp[:, idx]                                      # (B,T,W,R,C)
        boundary = zp[:, torch.arange(T, device=z.device)]       # column t-W (zero before the start)
        pos = torch.arange(T, device=z.device)[:, None] - (W - 1) + torch.arange(W, device=z.device)[None, :]
        valid = (pos >= 0).float()                               # (T,W)
        return anchor, boundary, valid

    def _lens_loop(self, step, L, anchor, boundary, g, valid, grad_anchor, record=False):
        """Settle lens windows (each freezes on its own), then the gradient tail. Shapes: (N,W,R,C)."""
        cfg = self.cfg
        N = L.shape[0]
        frozen = torch.zeros(N, device=L.device, dtype=torch.bool)
        steps = torch.full((N,), cfg.lens_cap, device=L.device, dtype=torch.long)
        frames = []
        full = cfg.lens_full_grad and torch.is_grad_enabled()
        with (contextlib.nullcontext() if full else torch.no_grad()):
            if full:
                a_d, b_d, g_d = grad_anchor
            else:
                a_d, b_d, g_d = anchor.detach(), boundary.detach(), g.detach()
            for k in range(cfg.lens_cap):
                Ln = step(L, a_d, b_d, g_d, valid)
                ch = (Ln - L).pow(2).mean((1, 2, 3)).sqrt()
                L = torch.where(frozen[:, None, None, None], L, Ln)
                newly = (~frozen) & (ch <= cfg.settle_tol)
                steps = torch.where(newly, torch.full_like(steps, k + 1), steps)
                frozen = frozen | newly
                if record:
                    frames.append(L[0].clone())
                if bool(frozen.all()):
                    break
        if not full:
            L = L.detach()
        anchor, boundary, g = grad_anchor
        for _ in range(cfg.lens_grad_steps):
            L = step(L, anchor, boundary, g, valid)
        return L, steps, frames

    def settle_lens(self, z, gist):
        cfg = self.cfg
        step = self._compiled[1] if self._compiled else self._lens_step
        B, T, R, C = z.shape
        W = cfg.lens_width
        anchor, boundary, valid = self.lens_windows(z)
        N = B * T
        anchor = anchor.reshape(N, W, R, C)
        boundary = boundary.reshape(N, R, C)
        valid = valid[None].expand(B, T, W).reshape(N, W)
        g = gist.reshape(N, -1)
        L, steps, _ = self._lens_loop(step, (anchor if cfg.lens_full_grad else anchor.detach().clone()), anchor, boundary, g, valid,
                                      (anchor, boundary, g))
        return L.view(B, T, W, R, C), steps.view(B, T)

    def forward(self, idx, targets=None):
        words = self._words(idx)
        z, stats = self.settle_sheet(words)
        gist = self._gist_parallel(z)
        L, lens_steps = self.settle_lens(z, gist)
        logits = self._read(L[:, :, -1], idx)
        stats["lens_steps"] = lens_steps.float().mean()
        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
            stats["ce"] = ce
            loss = ce + self.cfg.res_weight * stats["res"] + self.cfg.effort_weight * stats["effort"]
        return logits, loss, stats

    # ------------------------------------------------------------------ incremental (generation) pass
    def new_state(self, device=None):
        device = device or self.embed.weight.device
        cfg = self.cfg
        n = cfg.settle_cap + cfg.grad_steps + 1
        return {
            "sheet": torch.zeros(0, cfg.rows, cfg.channels, device=device),   # settled columns (ink)
            "hist": [],          # recent columns' states at every step index (needed by later columns)
            "ghist": torch.zeros(n, cfg.gist_dim, device=device),             # gist of the previous column per step
            "khist": [], "vhist": [],                                         # link keys / values per column per step
            "steps": [],
            "lens_steps": [],
            "tokens": [],
        }

    @torch.no_grad()
    def feed(self, state, token: int, record=False):
        """Add one word: a new column settles against the settled past, then the lens re-reads.

        Exactly the schedule of the parallel pass, done for one column. Returns
        logits for the next word and an info dict (steps; per-step frames if record).
        """
        cfg = self.cfg
        dev = self.embed.weight.device
        R, C = cfg.rows, cfg.channels
        cap, n_idx = cfg.settle_cap, cfg.settle_cap + cfg.grad_steps + 1
        hist = state["hist"]
        t = state["sheet"].shape[0]
        w = self._words(torch.tensor([[token]], device=dev))[0, 0].float()   # (R,C)
        zeros = torch.zeros(n_idx, C, device=dev)

        def past(back):                                       # (n_idx, R, C) history of column t-back, or None
            return hist[-back] if back <= len(hist) else None

        # left neighbours for every row at every step index: (n_idx, R, C)
        l1 = torch.stack([past(d)[:, r] if past(d) is not None else zeros for r, d in enumerate(self.dil)], 1)
        l2 = torch.stack([past(2 * d)[:, r] if past(2 * d) is not None else zeros for r, d in enumerate(self.dil)], 1)
        lam = self._decays()
        nG = lam.numel()
        gprev = state["ghist"].view(n_idx, nG, -1)

        def gist_at(s, z):
            u = self.gist_in(z.mean(0)).float().view(nG, -1)
            return (lam[:, None] * gprev[s] + (1 - lam)[:, None] * u).reshape(-1)

        H, dk = cfg.link_heads, cfg.link_dim
        Kp = torch.stack(state["khist"]) if state["khist"] else None     # (t, n_idx, H, dk)
        Vp = torch.stack(state["vhist"]) if state["vhist"] else None
        kv_now = {}

        def step(s, z):
            g = gist_at(s, z)
            up = torch.cat([z[1:], z.new_zeros(1, C)], 0)
            down = torch.cat([z.new_zeros(1, C), z[:-1]], 0)
            parts = [z, l1[s], l2[s], up, down, w, g[None].expand(R, -1)]
            if self.n_link:
                q, k, v = self.link_qkv(z.mean(0)).view(3, H, dk)
                K = torch.cat([Kp[:, s], k[None]], 0) if Kp is not None else k[None]   # (t+1,H,dk)
                V = torch.cat([Vp[:, s], v[None]], 0) if Vp is not None else v[None]
                att = torch.softmax((K * q[None]).sum(-1) / math.sqrt(dk), 0)          # (t+1,H)
                link = (att[:, :, None] * V).sum(0).reshape(-1)
                kv_now[s] = (k, v, att)
                parts.append(link[None].expand(R, -1))
            x = torch.cat(parts, -1)
            return self._apply_rule(z, x, self.rule1, self.row_bias, self.rule2)

        zs = [torch.zeros(R, C, device=dev)]
        frames, changes = [], []
        steps = cap
        for k in range(cap):
            zn = step(k, zs[-1])
            ch_rows = (zn - zs[-1]).pow(2).mean(1).sqrt()
            ch = ch_rows.pow(2).mean().sqrt().item()
            zs.append(zn)
            if record:
                frames.append(zn.clone())
                changes.append(ch_rows.clone())
            if ch <= cfg.settle_tol:
                steps = k + 1
                break
        while len(zs) < cap + 1:                                  # frozen: keep the state for the remaining indices
            zs.append(zs[-1])
        for j in range(cfg.grad_steps):                           # the same tail as training
            zs.append(step(cap + j, zs[-1]))
        Hs = torch.stack(zs)                                      # (n_idx, R, C)
        if self.n_link:
            # keys / values of this column at every step index (frozen steps reuse the frozen state)
            kq = self.link_qkv(Hs.mean(1)).view(n_idx, 3, H, dk)
            state["khist"].append(kq[:, 1])
            state["vhist"].append(kq[:, 2])
            links = kv_now[cap + cfg.grad_steps - 1][2] if cfg.grad_steps else None   # last step's attention (t+1,H)
        else:
            links = None
        u = self.gist_in(Hs.mean(1)).float().view(n_idx, nG, -1)
        G = (lam[None, :, None] * gprev + (1 - lam)[None, :, None] * u).reshape(n_idx, -1)
        z = Hs[-1]
        hist.append(Hs)
        keep = 2 * self.dil[-1]
        if len(hist) > keep:
            del hist[0]
        state["ghist"] = G
        state["sheet"] = torch.cat([state["sheet"], z[None]], 0)
        state["steps"].append(steps)
        state["tokens"].append(token)
        g = G[-1]

        # lens over the last W columns (two-way), anchored to the ink
        W = cfg.lens_width
        S2 = state["sheet"]
        T2 = S2.shape[0]
        anchor = torch.stack([S2[i] if i >= 0 else S2.new_zeros(R, C) for i in range(T2 - W, T2)])[None]
        boundary = (S2[T2 - W - 1] if T2 - W - 1 >= 0 else S2.new_zeros(R, C))[None]
        valid = torch.tensor([[1.0 if i >= 0 else 0.0 for i in range(T2 - W, T2)]], device=dev)
        L, lsteps, lens_frames = self._lens_loop(self._lens_step, anchor.clone(), anchor, boundary, g[None], valid,
                                                 (anchor, boundary, g[None]), record=record)
        lens_steps = int(lsteps[0])
        state["lens_steps"].append(lens_steps)
        logits = self._read(L[0, -1], torch.tensor(token, device=dev)).float()
        info = {"steps": steps, "lens_steps": lens_steps, "column": z, "lens": L[0], "anchor": anchor[0], "gist": g,
                "links": links}
        if record:
            info["frames"] = frames
            info["changes"] = changes
            info["lens_frames"] = lens_frames
        return logits, info

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=100, temperature=0.8, top_k=40, eos=None, state=None):
        state = state or self.new_state()
        logits = None
        for t in prompt_ids:
            logits, _ = self.feed(state, int(t))
        out = []
        for _ in range(max_new_tokens):
            nxt = sample_logits(logits, temperature, top_k)
            if eos is not None and nxt == eos:
                break
            out.append(nxt)
            logits, _ = self.feed(state, nxt)
        return out, state


def sample_logits(logits, temperature=0.8, top_k=40):
    logits = logits.float()
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k:
        v, _ = torch.topk(logits, min(top_k, logits.numel()))
        logits = logits.masked_fill(logits < v[-1], float("-inf"))
    p = torch.softmax(logits, -1)
    return int(torch.multinomial(p, 1))
