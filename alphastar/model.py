"""Pegasus Alpha Star: a 100M model that thinks in loops (Nov 8 2026, from the Special Lab). Still a NEXT-WORD model: it
reads text and guesses the very next token, and it is scored exactly like Plain-100M.

The idea in plain words. A normal model has a fixed number of layers: every word gets the same amount of thinking.
Alpha Star has three parts, and the middle one is used again and again:

    reader   (2 layers, once)     reads the words and fetches facts from the fact drawer
    core     (4 layers, looped)   thinks: the same 4 layers run N times ("passes"), each pass refining a private
                                  thought per word; the reader's output is fed back in on every pass, so it never
                                  forgets the text while it thinks
    speaker  (2 layers, once)     turns the final thought into the next-word guess (the tied word table)

So the model's size stays small (about 100M numbers) while its depth can grow: 2 + 4 x N + 2 layers of work. With
N = 6 (the training average) it is 28 layers deep; with 16 passes, 68. Huginn (Geiping et al. 2025) showed this works
at 3.5B; this is our 100M version with three additions.

  1. A random start. Each word's thought starts as random noise, not as the word itself, so the core must learn to
     build a thought from the reader's output and can be run for any number of passes (Huginn's recipe).
  2. The fact drawer (Lample et al. 2019 "product keys"; Meta's memory layers 2024). The reader's second layer has no
     ordinary MLP: instead, each word asks the drawer, which holds 16,384 stored vectors ("facts") and picks the 32
     that fit best per question head (4 heads) out of all 16,384, with only 2 x 128 comparisons per head (the
     product-key trick: a fact's key is two half-keys, and the best pairs are found from the best halves). The answer
     is the facts mixed by how well they fit. Big memory, tiny cost per word.
  3. The next-thought lesson (Plain-100F's foresight head, PR #149) on every pass: at each sentence end (or every 16
     words) a small shared head reads the core's thought and guesses the content words of the NEXT sentence. Later
     passes count more. It is a side lesson while training, never used for scoring or writing.

Training (Huginn's way, all switchable with --set):
  - The number of passes is drawn at random for each pass of the trainer: a Poisson count around a log-normal mean
    (mean 6, at most 12, at least 1), so the core learns to work at any depth.
  - A curriculum: the mean grows from 2 to 6 over the first 30% of the run (short thinking first, while it learns words).
  - Gradients flow only through the last 2 passes (truncated backprop): the earlier passes run without keeping their
    work, so memory and the backward pass cost the same whatever N is.

Reading (held-out loss, HellaSwag, LogT): it thinks until each word's thought stops changing (the change from one
pass to the next is below exit_eps of its size), at most 16 passes; a word that is done keeps its thought while the
others go on. test_passes=N forces N passes. Writing (chat, samples) uses a fixed gen_passes (8) word by word.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bee.model import Block, ctx_at
from bee4.model import Bee4Config
from foresight.model import Foresight
from memtrix.model import RMSNorm
from oazy.sample import sample_logits


@dataclass
class AlphaStarConfig(Bee4Config):
    # the layers: Plain-100M's (768 wide, 12 query / 4 key-value heads of 64, SwiGLU 2,048, RoPE 50,000)
    d_model: int = 768
    attn_heads: int = 12
    attn_kv_heads: int = 4
    attn_head_dim: int = 64
    mlp_hidden: int = 2048
    n_layer: int = 8                 # reader + core + speaker layers that exist (no growing)
    reader_layers: int = 2
    core_layers: int = 4
    speaker_layers: int = 2
    # passes through the core while training (Huginn)
    passes_mean: float = 6.0         # the mean number of passes once the curriculum is over
    passes_start: float = 2.0        # the mean at the start of training ...
    passes_ramp: float = 0.3         # ... growing in a straight line to passes_mean by this share of the run
    passes_sigma: float = 0.5        # spread of the log-normal mean the Poisson count is drawn around
    passes_max: int = 12             # at most this many passes while training
    grad_passes: int = 2             # gradients flow through the last this-many passes only
    # passes while reading
    test_passes: int = 0             # 0 = think until the thought settles (adaptive); N = always N passes
    test_max: int = 16               # adaptive: at most this many passes
    exit_eps: float = 0.03           # adaptive: a word is done once its thought moves less than this share of its size
    gen_passes: int = 8              # writing word by word (chat, samples): this many passes per word
    noise_rows: int = 4096           # reading: the random starting thoughts come from a fixed table (same every time)
    # the fact drawer (product-key memory)
    drawer: int = 1                  # 1 = the reader's second layer asks the fact drawer instead of an ordinary MLP
    drawer_at: str = "reader"        # "reader" (its last layer; the facts feed every pass) or "speaker" (its first layer)
    drawer_keys: int = 128           # half-keys per half: 128 x 128 = 16,384 facts
    drawer_heads: int = 4            # question heads, each picks its own facts
    drawer_k: int = 32               # facts picked per head
    drawer_qdim: int = 256           # each question's size (two halves of 128)
    # the next-thought lesson (foresight/model.py), on the core's thought after the passes that learn
    thought: int = 1                 # 1 = on
    fore_weight: float = 0.1         # its total weight next to the next-word loss (split over the learning passes, later ones more)
    fore_warmup: float = 0.05
    fore_every: int = 16
    fore_span: int = 32
    fore_top: int = 300
    fore_hidden: int = 1536
    fore_max: int = 8192
    train_noise: str = "random"      # "random" (Huginn) or "table" (the reading table; tests)

    def __post_init__(self):
        self.n_layer = self.reader_layers + self.core_layers + self.speaker_layers
        self.grow = f"{self.n_layer}:0"
        super().__post_init__()

    def depth_at(self, passes: float) -> float:
        return self.reader_layers + self.core_layers * passes + self.speaker_layers


# ---------------------------------------------------------------------------------------------------- the fact drawer
@torch.compiler.disable
def _bag(values, idx, w):
    """sum_j w[n, j] * values[idx[n, j]]: the picked facts, mixed, without ever forming the (N, picks, D) block."""
    return F.embedding_bag(idx, values, per_sample_weights=w, mode="sum")


class Drawer(nn.Module):
    """Product-key memory with Meta's gated output: y = out( facts(x) * silu(gate(x)) ).

    Each head turns the word into a question of drawer_qdim numbers, split in two halves. Each half is compared with
    drawer_keys half-keys; the best drawer_k of each half are paired up (drawer_k^2 pairs, each pair's score the sum of
    its halves), and the best drawer_k pairs are the facts picked. Their weights are a softmax of their scores; the
    heads' answers add up."""

    def __init__(self, cfg: AlphaStarConfig):
        super().__init__()
        D, H, n, q = cfg.d_model, cfg.drawer_heads, cfg.drawer_keys, cfg.drawer_qdim
        assert q % 2 == 0
        self.H, self.n, self.k, self.half = H, n, cfg.drawer_k, q // 2
        self.query = nn.Linear(D, H * q, bias=False)
        self.qnorm = RMSNorm(q // 2)
        self.keys = nn.Parameter(torch.randn(H, 2, n, q // 2) / math.sqrt(q // 2))
        self.values = nn.Embedding(n * n, D)
        self.gate = nn.Linear(D, D, bias=False)
        self.out = nn.Linear(D, D, bias=False)
        self.track = None                         # a (n*n,) count of how often each fact is picked (probes only)

    def reset_values(self):
        nn.init.normal_(self.values.weight, std=self.values.weight.shape[1] ** -0.5)
        self.values.weight.no_decay = True        # a fact used rarely must not fade between its uses (no weight decay)

    def pick(self, x):
        """(N, H*k) fact numbers and (N, H*k) weights for token vectors x (N, D)."""
        N = x.shape[0]
        q = self.qnorm(self.query(x).view(N, self.H, 2, self.half))
        s = torch.einsum("nhcd,hckd->nhck", q, self.keys.to(q.dtype))            # (N, H, 2, n)
        k = min(self.k, self.n)
        s1, i1 = s[:, :, 0].topk(k, -1)
        s2, i2 = s[:, :, 1].topk(k, -1)
        pair = (s1[..., :, None] + s2[..., None, :]).reshape(N, self.H, k * k)
        top, j = pair.topk(self.k if self.k <= k * k else k * k, -1)
        idx = i1.gather(-1, j // k) * self.n + i2.gather(-1, j % k)
        w = torch.softmax(top.float(), -1)
        return idx.reshape(N, -1), w.reshape(N, -1)

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        idx, w = self.pick(x2)
        if self.track is not None:
            self.track += torch.bincount(idx.reshape(-1), minlength=self.track.numel())
        facts = _bag(self.values.weight, idx, w).to(x.dtype)
        return self.out(facts * F.silu(self.gate(x2))).reshape(shp)


class DrawerBlock(Block):
    """A Bee4 attention layer whose MLP is the fact drawer."""

    def __init__(self, cfg: AlphaStarConfig):
        super().__init__(cfg, "A")
        del self.mlp
        self.drawer = Drawer(cfg)

    def forward(self, x, state=None, resets=None, mask=None, segs=None):
        y, state = self.attn(self.n1(x), state, mask)
        x = x + y
        return x + self.drawer(self.n2(x)), state


# ---------------------------------------------------------------------------------------------------- the model
class AlphaStar(Foresight, nn.Module):
    embed_no_decay = True
    fused_loss = True
    loss_chunk = 1024
    micro_at_full = True             # --micro N = N sequences per pass at the longest window (as Bee4 / Plain-100M)

    def __init__(self, cfg: AlphaStarConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model
        self.embed = nn.Embedding(cfg.vocab_size, D)
        at_r = cfg.drawer and cfg.drawer_at == "reader"
        at_s = cfg.drawer and cfg.drawer_at == "speaker"
        self.reader = nn.ModuleList(DrawerBlock(cfg) if (at_r and i == cfg.reader_layers - 1) else Block(cfg, "A")
                                    for i in range(cfg.reader_layers))
        self.reader_norm = RMSNorm(D)
        self.adapt = nn.Linear(2 * D, D, bias=False)          # [thought, reader's output] -> the core's input, every pass
        self.core = nn.ModuleList(Block(cfg, "A") for _ in range(cfg.core_layers))
        self.core_norm = RMSNorm(D)
        self.speaker = nn.ModuleList(DrawerBlock(cfg) if (at_s and i == 0) else Block(cfg, "A")
                                     for i in range(cfg.speaker_layers))
        self.norm = RMSNorm(D)
        self.apply(self._init)
        deep = cfg.depth_at(cfg.passes_mean)                   # the depth it works at on average (28)
        for n, p in self.named_parameters():
            if n.endswith(("attn.out.weight", "mlp.down.weight", "drawer.out.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * deep))
        for m in self.modules():
            if isinstance(m, Drawer):
                m.reset_values()
        if cfg.thought:
            self.init_foresight(cfg)
        else:
            self._fore_w, self._ready = 0.0, False
        g = torch.Generator().manual_seed(1234)
        self.register_buffer("noise", torch.randn(cfg.noise_rows, D, generator=g), persistent=False)
        self._mean = cfg.passes_start if cfg.passes_ramp > 0 else cfg.passes_mean
        self._rng = np.random.default_rng(0)
        self._compiled = None
        self.force_passes = None      # tests: every training pass uses exactly this many passes

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    # ---------------------------------------------------------------- sizes
    def head_params(self):
        return sum(p.numel() for p in self.fore.parameters()) if hasattr(self, "fore") else 0

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters()) - self.head_params()
        return n - self.embed.weight.numel() if non_embedding else n

    def drawer_params(self):
        return sum(p.numel() for m in self.modules() if isinstance(m, Drawer) for p in m.parameters())

    def split(self):
        """Where the numbers are (millions)."""
        part = lambda ms: sum(p.numel() for m in ms for p in m.parameters()) / 1e6   # noqa: E731
        dr = self.drawer_params() / 1e6
        return {"word table": self.embed.weight.numel() / 1e6, "reader": part(self.reader) - (dr if self.cfg.drawer_at == "reader" else 0),
                "fact drawer": dr, "core (looped)": part(self.core) + part([self.adapt]),
                "speaker": part(self.speaker) - (dr if self.cfg.drawer_at == "speaker" else 0),
                "next-thought head (training only)": self.head_params() / 1e6}

    def layout(self):
        c = self.cfg
        d = (f", fact drawer in the {c.drawer_at} ({c.drawer_keys ** 2:,} facts, {c.drawer_heads} heads x {c.drawer_k} picks)"
             if c.drawer else "")
        t = f", next-thought head for training only (weight {c.fore_weight:g})" if c.thought else ""
        return (f"reader {c.reader_layers} + core {c.core_layers} looped (mean {c.passes_mean:g} passes, at most "
                f"{c.passes_max} training, {c.test_max} reading) + speaker {c.speaker_layers}, {c.d_model} wide{d}{t}")

    def ctx_at(self, progress: float) -> int:
        return ctx_at(self.cfg, progress)

    def foresight_ready(self):
        """Without the next-thought lesson there are no tables to fill: always ready."""
        return Foresight.foresight_ready(self) if self.cfg.thought else True

    # ---------------------------------------------------------------- schedule
    def set_progress(self, progress: float):
        c = self.cfg
        self._fore_w = c.fore_weight * (min(1.0, progress / c.fore_warmup) if c.fore_warmup > 0 else 1.0) if c.thought else 0.0
        f = min(1.0, progress / c.passes_ramp) if c.passes_ramp > 0 else 1.0
        self._mean = c.passes_start + (c.passes_mean - c.passes_start) * f

    def sample_passes(self) -> int:
        """Huginn's draw: a Poisson count around a log-normal mean (its average is the current mean), 1 to passes_max."""
        if self.force_passes:
            return int(self.force_passes)
        c = self.cfg
        lam = math.exp(self._rng.normal(math.log(self._mean) - c.passes_sigma ** 2 / 2, c.passes_sigma))
        return int(min(c.passes_max, max(1, self._rng.poisson(lam))))

    # ---------------------------------------------------------------- the parts
    def doc_info(self, idx, need_mask=False):
        """Where documents start, and (for the mentor only) the same-document grid. Attention itself is plain causal,
        as in Bee4 / Plain-100M."""
        from oazy.tok import eot_id
        B, T = idx.shape
        resets = torch.zeros(B, T, dtype=torch.bool, device=idx.device)
        resets[:, 1:] = idx[:, :-1] == eot_id(self.cfg.tokenizer)
        if not need_mask:
            return resets, None, None
        doc = resets.cumsum(1)
        mask = (doc[:, None, :, None] == doc[:, None, None, :]) & torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
        return resets, mask, None

    def _read(self, idx):
        x = self.embed(idx)
        for b in self.reader:
            x, _ = b(x)
        return self.reader_norm(x)

    def _step(self, s, e):
        """One pass through the core: [thought, reader's output] -> 4 layers -> the new thought."""
        x = self.adapt(torch.cat([s, e.to(s.dtype)], -1))
        for b in self.core:
            x, _ = b(x)
        return self.core_norm(x)

    def _speak(self, s):
        x = s
        for b in self.speaker:
            x, _ = b(x)
        return self.norm(x)

    def head(self, h):
        logits = F.linear(h, self.embed.weight)
        c = self.cfg.softcap
        return c * torch.tanh(logits / c) if c else logits

    def enable_compile(self):
        self._compiled = {"read": torch.compile(self._read), "step": torch.compile(self._step),
                          "speak": torch.compile(self._speak)}

    def _fn(self, name):
        return self._compiled[name] if self._compiled is not None else getattr(self, "_" + name)

    def start(self, B, T, start, device, dtype, table):
        """The random starting thoughts (B, T, D): fresh noise while training, the fixed table while reading."""
        if not table:
            return torch.randn(B, T, self.cfg.d_model, device=device, dtype=dtype)
        rows = (torch.arange(start, start + T, device=device) % self.noise.shape[0])
        return self.noise[rows].to(dtype)[None].expand(B, T, -1)

    # ---------------------------------------------------------------- thinking
    def think_train(self, e, n):
        """n passes, gradients through the last grad_passes only. Returns the thoughts after those passes."""
        step = self._fn("step")
        B, T, _ = e.shape
        s = self.start(B, T, 0, e.device, e.dtype, self.cfg.train_noise == "table")
        k = min(self.cfg.grad_passes, n)
        with torch.no_grad():
            ed = e.detach()
            for _ in range(n - k):
                s = step(s, ed)
        outs = []
        for _ in range(k):
            s = step(s, e)
            outs.append(s)
        return outs

    @torch.no_grad()
    def think_test(self, e, passes=None, eps=None, every=None, fast=False):
        """Reading: a fixed number of passes, or (passes=None) until each word's thought settles, at most test_max.
        A word that is done keeps its thought; the loop ends once every word is done. every: called with
        (pass number, thoughts) after each pass. fast: the compiled pass (fixed shapes only: the held-out loss).
        Returns (thoughts, passes per word (B, T))."""
        c = self.cfg
        step = self._fn("step") if fast else self._step
        B, T, _ = e.shape
        s = self.start(B, T, 0, e.device, e.dtype, True)
        fixed = passes or (c.test_passes or None)
        eps = c.exit_eps if eps is None else eps
        n_max = fixed or c.test_max
        used = torch.zeros(B, T, device=e.device)
        done = torch.zeros(B, T, dtype=torch.bool, device=e.device)
        for i in range(n_max):
            new = step(s, e)
            if fixed is None:
                move = (new - s).float().norm(dim=-1) / new.float().norm(dim=-1).clamp_min(1e-6)
                new = torch.where(done[..., None], s, new)
                used += (~done).float()
                done |= (move < eps) & (i > 0)          # the first pass always moves (from noise)
            else:
                used += 1
            s = new
            if every is not None:
                every(i + 1, s)
            if fixed is None and bool(done.all()):
                break
        return s, used

    def thoughts_to_h(self, idx, passes=None):
        """Uncompiled: HellaSwag and the probes change shapes every batch."""
        s, used = self.think_test(self._read(idx), passes)
        return self._speak(s), used

    # ---------------------------------------------------------------- the lesson
    def losses(self, idx, targets, mentor=None, kd_weight=0.0):
        """The next-word loss (+ the mentor's, as Plain-100M) on the speaker's output after a random number of passes,
        + the next-thought loss on the thoughts of the passes that learn (later passes weigh more)."""
        use_mentor = mentor is not None and kd_weight > 0
        resets, mask, _ = self.doc_info(idx, need_mask=use_mentor)
        learn = self.training and torch.is_grad_enabled()
        e = self._fn("read")(idx)
        if learn:
            n = self.sample_passes()
            outs = self.think_train(e, n)
            used = float(n)
        else:
            s, u = self.think_test(e, fast=True)
            outs, used = [s], u.mean()
        h = self._fn("speak")(outs[-1])
        th = tW = None
        if use_mentor:
            th, tW = mentor.hidden(idx, mask)
        if self.fused_loss:
            from oazy.lmloss import head_loss
            sel = getattr(self, "token_select", None) if learn else None
            loss, ce, kd = head_loss(h, self.embed.weight, targets, self.cfg.softcap, th, tW, kd_weight, self.loss_chunk, sel)
        else:
            from oazy.lmloss import plain_loss
            loss, ce, kd = plain_loss(h, self.embed.weight, targets, self.cfg.softcap, th, tW, kd_weight)
        stats = {"ce": ce, "passes": torch.as_tensor(used)}
        if th is not None:
            stats["kd"] = kd
        if learn and self.cfg.thought and self._ready:
            pts = self.fore_points(idx, targets)
            first = n - len(outs) + 1                          # the pass number of the first learning pass
            ws = [first + j for j in range(len(outs))]
            tot = 0.0
            for w, s in zip(ws, outs):
                fl, _ = self.fore_nll(s, pts)
                tot = tot + (w / sum(ws)) * fl
            if self._fore_w > 0:
                loss = loss + self._fore_w * tot
            stats["fore"] = fl.detach()                        # the last pass's, as Plain-100F logs it
        return loss, stats

    def forward(self, idx, targets=None):
        if targets is None:
            return self._body(idx), None, {}
        loss, stats = self.losses(idx, targets)
        return None, loss, stats

    def _body(self, idx):
        """The full scores (B, T, V) after thinking (adaptive unless test_passes), as the probes and HellaSwag want them."""
        h, _ = self.thoughts_to_h(idx)
        return self.head(h)

    # ---------------------------------------------------------------- reading as a stream (chat)
    def read(self, idx, states=None, passes=None):
        """Read tokens (B, T), carrying every layer's (and every pass's) attention keys along. A fixed number of passes
        per word (gen_passes), so every pass has the keys of every earlier word. Returns (logits, states)."""
        c = self.cfg
        n = passes or (states or {}).get("n") or c.gen_passes
        st = states or {"r": [None] * len(self.reader), "c": [[None] * len(self.core) for _ in range(n)],
                        "s": [None] * len(self.speaker), "pos": 0, "n": n}
        B, T = idx.shape
        x = self.embed(idx)
        for i, b in enumerate(self.reader):
            x, st["r"][i] = b(x, st["r"][i])
        e = self.reader_norm(x)
        s = self.start(B, T, st["pos"], e.device, e.dtype, True)
        for p in range(st["n"]):
            x = self.adapt(torch.cat([s, e], -1))
            for j, b in enumerate(self.core):
                x, st["c"][p][j] = b(x, st["c"][p][j])
            s = self.core_norm(x)
        x = s
        for i, b in enumerate(self.speaker):
            x, st["s"][i] = b(x, st["s"][i])
        st["pos"] += T
        return self.head(self.norm(x)), st

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=100, temperature=0.8, top_k=40, eos=None, rep=1.0, passes=None):
        dev = self.embed.weight.device
        logits, states = self.read(torch.tensor([list(prompt_ids)], device=dev), passes=passes)
        out = []
        for _ in range(max_new_tokens):
            nxt = sample_logits(logits[0, -1], temperature, top_k, recent=(list(prompt_ids) + out)[-64:], rep=rep)
            if eos is not None and nxt == eos:
                break
            out.append(nxt)
            logits, states = self.read(torch.tensor([[nxt]], device=dev), states)
        return out, None

    # ---------------------------------------------------------------- probes
    @torch.no_grad()
    def token_ce(self, h, y, chunk=4096):
        """The next-word loss of every token (B, T), in slices (probes only)."""
        h2, y2 = h.reshape(-1, h.shape[-1]), y.reshape(-1)
        out = torch.empty(len(y2), device=h.device)
        for i in range(0, len(y2), chunk):
            out[i:i + chunk] = F.cross_entropy(self.head(h2[i:i + chunk]).float(), y2[i:i + chunk], reduction="none")
        return out.view(y.shape)

    def word_kinds(self, y):
        """(B, T) bool: is the word to guess filler (the most common pieces, punctuation)? None without the tables."""
        if not (self.cfg.thought and self.foresight_ready()):
            return None
        return self.fore_skip[y]

    @torch.no_grad()
    def loop_report(self, windows, device, batch=16, curve=(1, 2, 4, 8, 16)):
        """On held-out windows: the next-word loss after 1, 2, 4, 8, 16 passes and when it stops by itself (with the
        average passes it took), the next-thought loss after every pass (and the blind guess), how much the thought
        still moves at the last pass, and how many facts in the drawer are ever picked. With the word tables, the loss by
        passes and the passes taken are also split into filler words (the, of, commas ...) and content words."""
        was = self.training
        self.eval()
        xy = torch.as_tensor(windows, device=device)
        top = max(curve)
        ce = {n: 0.0 for n in curve}
        fore = [0.0] * top
        blind = words = 0.0
        ada = ada_passes = move = 0.0
        split = {"filler": {n: 0.0 for n in curve}, "content": {n: 0.0 for n in curve}}
        n_kind = {"filler": 0.0, "content": 0.0}
        used_kind = {"filler": 0.0, "content": 0.0}
        drawers = [m for m in self.modules() if isinstance(m, Drawer)]
        counts = [torch.zeros(d.n * d.n, dtype=torch.long, device=device) for d in drawers]
        rows = 0
        has_fore = self.cfg.thought and self.foresight_ready()
        for i in range(0, len(xy), batch):
            x, y = xy[i:i + batch, :-1], xy[i:i + batch, 1:]
            nb = len(x)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                for d, cnt in zip(drawers, counts):
                    d.track = cnt                       # count the facts it picks (once per window)
                e = self._read(x)
                for d in drawers:
                    d.track = None
                pts = self.fore_points(x, y) if has_fore else None
                kind = self.word_kinds(y)
                if kind is not None:
                    n_kind["filler"] += float(kind.sum())
                    n_kind["content"] += float((~kind).sum())
                last = {}

                def every(n, s):
                    if n in ce:
                        tc = self.token_ce(self._speak(s), y)
                        ce[n] += float(tc.mean()) * nb
                        if kind is not None:
                            split["filler"][n] += float(tc[kind].sum())
                            split["content"][n] += float(tc[~kind].sum())
                    if pts is not None:
                        fl, cnt = self.fore_nll(s, pts)
                        fore[n - 1] += float(fl) * float(cnt)
                    if n == top - 1:
                        last["s"] = s.float()
                    if n == top:
                        last["move"] = float(((s.float() - last["s"]).norm(dim=-1) / s.float().norm(dim=-1).clamp_min(1e-6)).mean())
                self.think_test(e, passes=top, every=every)
                move += last.get("move", 0.0) * nb
                if pts is not None:
                    _, cnt, bl = self.fore_nll(self.start(nb, x.shape[1], 0, device, e.dtype, True), pts, prior=True)
                    blind += float(bl) * float(cnt)
                    words += float(cnt)
                s, used = self.think_test(e)
                ada += float(self.token_ce(self._speak(s), y).mean()) * nb
                ada_passes += float(used.mean()) * nb
                if kind is not None:
                    used_kind["filler"] += float(used[kind].sum())
                    used_kind["content"] += float(used[~kind].sum())
            rows += nb
        out = {"pass_loss": {str(n): v / rows for n, v in ce.items()}, "adaptive_loss": ada / rows,
               "adaptive_passes": ada_passes / rows, "move_last": move / rows, "exit_eps": self.cfg.exit_eps}
        if n_kind["filler"] and n_kind["content"]:
            for k in ("filler", "content"):
                out[f"pass_loss_{k}"] = {str(n): v / n_kind[k] for n, v in split[k].items()}
                out[f"passes_{k}"] = used_kind[k] / n_kind[k]
        if words:
            out.update({"thought_by_pass": [v / words for v in fore], "thought_blind": blind / words})
        for cnt in counts[:1]:
            c = cnt.float()
            srt = c.sort(descending=True).values
            half = int((srt.cumsum(0) < 0.5 * c.sum()).sum()) + 1
            out.update({"drawer_used": float((c > 0).float().mean()), "drawer_half": half / c.numel()})
        self.train(was)
        return out

    @torch.no_grad()
    def foresight_report(self, windows, device, batch=16):
        """Plain-100F's probe, on the final thought (adaptive passes): the head's loss and the blind guess's."""
        if not (self.cfg.thought and self.foresight_ready()):
            return None
        was = self.training
        self.eval()
        xy = torch.as_tensor(windows, device=device)
        tot = blind = n_all = 0.0
        for i in range(0, len(xy), batch):
            x, y = xy[i:i + batch, :-1], xy[i:i + batch, 1:]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                s, _ = self.think_test(self._read(x))
                fl, n, bl = self.foresight_loss(s, x, y, prior=True)
            tot, blind, n_all = tot + float(fl) * float(n), blind + float(bl) * float(n), n_all + float(n)
        self.train(was)
        return {"fore_val": tot / max(n_all, 1), "fore_blind": blind / max(n_all, 1), "fore_words": int(n_all)}
