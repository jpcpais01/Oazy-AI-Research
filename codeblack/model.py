"""Code Black: two clocks. A NEXT-WORD model: every position still guesses the very next token, exactly like
GPT-2 and every Bee, and it is scored the same way. And Black-T, the strong modern transformer it must beat.

Why two clocks. At our sizes and 2,048-token windows, attention is only ~13% of a transformer's training
arithmetic; the rest is the big matrix products every layer does for every token, plus the word table. So the
only ways to beat a transformer on the clock are to run the expensive layers on fewer positions, to get
knowledge that costs no arithmetic, and to keep the chip busy (the design doc: "Code Black: a
faster-than-transformer architecture"). Code Black does all three:

    tokens ──► word table + phrase memory ──► fast encoder (2 narrow layers, every token, 256-token window)
                                                   │                                   │ skip (the residual)
                                          pack 4 tokens into 1                          ▼
                                                   ▼                        fast decoder (2 narrow layers) ──► next token
                                 slow stream (16 wide layers, one position per 4 tokens) ──► unpack ──┘

  fast stream     width 512. Spelling, grammar, the last few words. Its attention looks back 256 tokens.
  phrase memory   a big hashed table of the 2- and 3-token phrases that end at each word (DeepSeek's Engram, the
                  nanoGPT speedrun's n-gram table): the phrase is turned into a row number and that row is read,
                  with a gate that ignores it when it does not fit. A lookup, not a sum: knowledge for free.
  pack            every 4 tokens side by side, through one matrix, into one vector of width 1,024.
  slow stream     the deep part: 16 layers of 1,024 on 512 positions per 2,048-token window, a quarter of the
                  cost per token of the same stack run on every token. Plain attention by default (cheap on 512
                  positions); slow_pattern="MMMA" puts Memtrix memory layers in, 3 to 1 (for long texts).
  unpack          each group's slow result becomes 4 vectors for the tokens that may see it: the last token of the
                  group itself and the first 3 of the next. No token ever sees, from the slow side, a token after
                  it: that is what keeps it a fair next-token model. Before the first group (and at the start of
                  every document) the tokens get a learned "start" state.

Black-T is the yardstick: a 2026 transformer, not GPT-2 as written in 2019. Bee3's gated attention (QK-norm,
grouped key/value heads, sigmoid output gate) with RoPE in every layer, SwiGLU, RMSNorm, no biases, tied word
table, soft-capped scores, no bleed between documents (flex attention), Muon, the loss in slices, FP8 products.
Three sizes (125m, 250m, 500m) draw the transformer line: held-out loss against B200 minutes.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from bee.model import Block, doc_block_mask, flex_check
from bee3.model import Bee3, Bee3Config
from memtrix.model import RMSNorm
from oazy.sample import sample_logits


# ====================================================================================== Black-T, the yardstick
# name: (width, layers, query heads, key-value heads, SwiGLU width). Heads of 64. The SwiGLU width is ~8/3 x the
# width (the same arithmetic as GPT-2's 4 x MLP with two matrices), rounded to a multiple of 64.
BLACKT_SIZES = {
    "125m": (768, 12, 12, 4, 2048),       # ~120M parameters with SmolLM2's 49,152-word table
    "250m": (1024, 16, 16, 4, 2752),      # ~244M
    "500m": (1280, 24, 20, 4, 3456),      # ~515M
}


@dataclass
class BlackTConfig(Bee3Config):
    size: str = "125m"           # one of BLACKT_SIZES; "" = set d_model, n_layer, ... by hand (the test presets)
    pattern: str = "A"           # attention in every layer
    rope: int = 1                # RoPE: nothing else tells a pure transformer the word order
    flex_attn: int = 1           # with the document rule, flex attention skips the blocks that cross documents

    def __post_init__(self):
        if self.size:
            if self.size not in BLACKT_SIZES:
                raise ValueError(f"Black-T size {self.size!r}: pick one of {', '.join(BLACKT_SIZES)} (or '' to set it by hand)")
            self.d_model, self.n_layer, self.attn_heads, self.attn_kv_heads, self.mlp_hidden = BLACKT_SIZES[self.size]
            self.attn_head_dim = 64
        if "M" in self.pattern:
            raise ValueError("Black-T is the plain transformer: pattern must be attention only")


class BlackT(Bee3):
    """A strong modern transformer, built from the same parts as every Bee (so the race is fair): Bee3's body with
    attention in every layer and RoPE."""

    def layout(self):
        c = self.cfg
        return (f"{c.n_layer} attention layers of {c.d_model} ({c.attn_heads} query / {c.attn_kv_heads} key-value heads, "
                f"RoPE, QK-norm, gated), SwiGLU {c.mlp_hidden}")

    def prepare_kernels(self, device):
        pass                                        # no memory layers: nothing to check but flex attention

    def doc_info(self, idx):
        """The document rule for attention only: (resets, which earlier words each word may see, None)."""
        if not self.cfg.doc_mask:
            return None, None, None
        resets = doc_starts(idx, self.cfg.tokenizer)
        doc = resets.cumsum(1)
        T = idx.shape[1]
        mask = (doc[:, None, :, None] == doc[:, None, None, :]) & torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
        return resets, mask, None

    def flops_per_token(self, T):
        """Training arithmetic per token (the profiler's convention: 6 per weight, attention's 12 x layers x width x T)."""
        return 6 * self.num_params() + 12 * self.cfg.n_layer * self.cfg.d_model * T


def doc_starts(idx, tokenizer):
    """(B, T) bool: a new document starts here (right after an <|endoftext|>)."""
    from oazy.tok import eot_id
    resets = torch.zeros(idx.shape, dtype=torch.bool, device=idx.device)
    resets[:, 1:] = idx[:, :-1] == eot_id(tokenizer)
    return resets


# ====================================================================================== Code Black
@dataclass
class BlackConfig(Bee3Config):
    # --- the fast stream (every token). d_model is its width, and the word table's.
    d_model: int = 512
    enc_layers: int = 2          # before the slow stream
    dec_layers: int = 2          # after it
    attn_heads: int = 8          # 8 x 64 = the width
    attn_kv_heads: int = 4
    attn_head_dim: int = 64
    mlp_hidden: int = 1408       # SwiGLU, ~8/3 x the width
    window: int = 256            # fast attention looks back at most this many tokens (0 = the whole window)
    # --- the two clocks
    group: int = 4               # tokens per slow position
    # --- the slow stream (once per group)
    slow_d: int = 1024
    slow_layers: int = 16
    slow_pattern: str = "A"      # A = attention; "MMMA" = 3 Memtrix memory layers to 1 attention (no RoPE then)
    slow_heads: int = 16
    slow_kv_heads: int = 4
    slow_head_dim: int = 64
    slow_mlp: int = 2752
    slow_mem_heads: int = 8      # memories per slow memory layer (of head_k x head_v; only with slow_pattern M)
    # --- the phrase memory (0 rows = off)
    phrase_rows: int = 131072    # rows per table
    phrase_orders: str = "2,3"   # phrases of 2 and of 3 tokens ending at each word
    phrase_heads: int = 2        # tables per order (different hashes, so two phrases rarely share all their rows)
    phrase_dim: int = 256        # numbers per row
    phrase_at: int = 1           # added after this many encoder layers (1: the gate can read a little context)
    # --- inherited, fixed for Code Black
    pattern: str = "A"
    rope: int = 1
    flex_attn: int = 1
    n_layer: int = 0             # (set from the streams: enc + slow + dec, for the log)

    def __post_init__(self):
        if self.max_len % self.group:
            raise ValueError(f"max_len {self.max_len} must be a multiple of group {self.group}")
        if not 0 <= self.phrase_at <= self.enc_layers:
            raise ValueError(f"phrase_at {self.phrase_at} must be between 0 and enc_layers {self.enc_layers}")
        self.n_layer = self.enc_layers + self.slow_layers + self.dec_layers

    def fast_cfg(self):
        return dataclasses.replace(self, rope=1)

    def slow_cfg(self):
        mem = "M" in self.slow_pattern
        return dataclasses.replace(self, d_model=self.slow_d, attn_heads=self.slow_heads, attn_kv_heads=self.slow_kv_heads,
                                   attn_head_dim=self.slow_head_dim, mlp_hidden=self.slow_mlp, n_head=self.slow_mem_heads,
                                   rope=0 if mem else 1)

    def slow_kinds(self):
        return [self.slow_pattern[i % len(self.slow_pattern)] for i in range(self.slow_layers)]

    def orders(self):
        return [int(n) for n in self.phrase_orders.split(",") if n.strip()]


# large odd multipliers for the phrase hash, one per (table, position in the phrase); < 2^31 so products fit in int64
_MULT = [1000003, 1640531527, 2097143, 1342177283, 805306457, 1073741827, 1879048201, 1610612741, 1181783497,
         1952801581, 741036287, 1505683883]


class PhraseMemory(nn.Module):
    """The gate and the two projections around the hashed table (the table itself is Black.embed_phrase, so it
    gets the word table's treatment: no weight decay)."""

    def __init__(self, cfg: BlackConfig):
        super().__init__()
        self.cfg = cfg
        self.n_tab = len(cfg.orders()) * cfg.phrase_heads
        width = self.n_tab * cfg.phrase_dim
        self.key = nn.Linear(width, cfg.d_model, bias=False)
        self.val = nn.Linear(width, cfg.d_model, bias=False)
        self.qn, self.kn = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)

    def rows(self, idx, doc=None):
        """(B, T) tokens -> (B, T, n_tab) row numbers into the big table: one per table. A phrase never reaches
        back across a document start or before the window: those places hold a filler that is not a token."""
        cfg = self.cfg
        B, T = idx.shape
        filler = cfg.vocab_size
        prev = [idx]
        for j in range(1, max(cfg.orders())):
            p = torch.full_like(idx, filler)
            p[:, j:] = idx[:, :-j]
            if doc is not None:
                same = torch.zeros_like(idx, dtype=torch.bool)
                same[:, j:] = doc[:, j:] == doc[:, :-j]
                p = torch.where(same, p, torch.full_like(p, filler))
            prev.append(p)
        out, t = [], 0
        for n in cfg.orders():
            for k in range(cfg.phrase_heads):
                h = torch.zeros_like(idx)
                for j in range(n):
                    h = h ^ (prev[j] * _MULT[(t * 3 + j) % len(_MULT)])
                out.append(h % cfg.phrase_rows + t * cfg.phrase_rows)
                t += 1
        return torch.stack(out, -1)

    def forward(self, x, e):
        """x (B, T, D) the stream, e (B, T, n_tab * dim) the rows read. Returns what to add to the stream."""
        k = self.key(e)
        g = torch.sigmoid((self.qn(x) * self.kn(k)).sum(-1, keepdim=True) / math.sqrt(x.shape[-1]))
        return g * self.val(e)


class Black(nn.Module):
    """Code Black: a fast stream for every token, a slow stream once per group of 4, a phrase memory."""
    embed_no_decay = True        # the word table and the phrase table are not pulled towards zero
    fused_loss = True            # the loss in slices (oazy/lmloss.py); False = full 32-bit scores
    loss_chunk = 1024

    def __init__(self, cfg: BlackConfig):
        super().__init__()
        self.cfg = cfg
        fc, sc = cfg.fast_cfg(), cfg.slow_cfg()
        K, D, Ds = cfg.group, cfg.d_model, cfg.slow_d
        self.embed = nn.Embedding(cfg.vocab_size, D)
        if cfg.phrase_rows:
            self.embed_phrase = nn.Embedding(len(cfg.orders()) * cfg.phrase_heads * cfg.phrase_rows, cfg.phrase_dim)
            self.phrase = PhraseMemory(cfg)
        self.enc = nn.ModuleList(Block(fc, "A") for _ in range(cfg.enc_layers))
        self.pack = nn.Linear(K * D, Ds, bias=False)
        self.slow = nn.ModuleList(Block(sc, k) for k in cfg.slow_kinds())
        self.slow_norm = RMSNorm(Ds)
        self.slow_start = nn.Parameter(torch.zeros(Ds))     # the slow state before the first group of a document
        self.unpack = nn.Linear(Ds, K * D, bias=False)
        self.dec = nn.ModuleList(Block(fc, "A") for _ in range(cfg.dec_layers))
        self.norm = RMSNorm(D)
        self.apply(self._init)
        nf, ns = cfg.enc_layers + cfg.dec_layers, cfg.slow_layers
        for n, p in self.named_parameters():
            if n.endswith(("attn.out.weight", "mlp.down.weight", "mem.out.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * (ns if n.startswith("slow.") else nf)))
        nn.init.zeros_(self.unpack.weight)          # the slow stream starts as a no-op and grows in (ReZero-style)
        if cfg.phrase_rows:
            nn.init.zeros_(self.phrase.val.weight)  # the phrase memory too
        for b in self.slow:
            if b.kind == "M":
                b.mem.reset_memory_params()
        self._compiled = None
        self._flex = False

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    # ------------------------------------------------------------------ sizes
    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        return n - self.embed.weight.numel() if non_embedding else n

    def phrase_params(self):
        return self.embed_phrase.weight.numel() if self.cfg.phrase_rows else 0

    def active_params(self):
        """Parameters that do arithmetic for a token: everything but the phrase table (a lookup). The slow stream
        counts in full here; flops_per_token divides it by the group."""
        return self.num_params() - self.phrase_params()

    def stream_params(self):
        """(fast-stream weights incl. the phrase gate, slow-stream weights incl. pack and unpack), without word table."""
        fast = sum(p.numel() for m in (*self.enc, *self.dec, self.norm) for p in m.parameters())
        if self.cfg.phrase_rows:
            fast += sum(p.numel() for p in self.phrase.parameters())
        slow = sum(p.numel() for m in (*self.slow, self.pack, self.unpack, self.slow_norm) for p in m.parameters())
        return fast, slow

    def flops_per_token(self, T):
        """Training arithmetic per token, in the profiler's convention (6 per weight a token passes through,
        attention's 12 x layers x width x the length looked back over). The slow stream's weights are passed once
        per group, so they count 1/group per token; the word table counts once (the score head)."""
        c = self.cfg
        fast, slow = self.stream_params()
        w = min(c.window or T, T)
        n_slow_attn = c.slow_kinds().count("A")
        att = 12 * (c.enc_layers + c.dec_layers) * c.d_model * w + 12 * n_slow_attn * c.slow_d * (T // c.group) / c.group
        return 6 * (fast + slow / c.group + self.embed.weight.numel()) + att

    def layout(self):
        c = self.cfg
        kinds = c.slow_kinds()
        mix = "attention" if "M" not in kinds else f"{kinds.count('M')} memory + {kinds.count('A')} attention"
        ph = (f"phrase memory {self.phrase_params() / 1e6:.0f}M ({'+'.join(c.phrase_orders.split(','))}-token phrases)"
              if c.phrase_rows else "no phrase memory")
        return (f"fast {c.enc_layers}+{c.dec_layers} layers of {c.d_model} (window {c.window or 'all'}), slow {c.slow_layers} "
                f"layers of {c.slow_d} ({mix}) once per {c.group} tokens, {ph}")

    def ctx_at(self, progress: float) -> int:
        from bee.model import ctx_at
        return ctx_at(self.cfg, progress)

    # ------------------------------------------------------------------ the rules of who may see whom
    def doc_info(self, idx, grids=True):
        """Everything the masks need, for a window whose length is a multiple of the group:
          fast   which earlier tokens each token's fast attention may see (B, 1, T, T) bool, or None (plain causal)
          slow   which earlier groups each group may see (B, 1, N, N) bool, or None (plain causal)
          keep   (B, T) bool: the token belongs to the document of its group's last token (else it is left out of the
                 pack, so no group carries another document's words)
          valid  (B, T) bool: the slow state this token receives comes from its own document (else: the start state)
          resets, gresets   document starts by token (B, T) and by group (B, N), or None
          segs   the memory kernel's document offsets for the slow stream (only with memory layers), or None
        grids=False leaves the two (T, T) and (N, N) grids out (flex attention builds its block masks instead)."""
        c = self.cfg
        B, T = idx.shape
        K, N, dev = c.group, T // c.group, idx.device
        w = c.window if c.window and c.window < T else 0
        info = dict(fast=None, slow=None, keep=None, valid=None, resets=None, gresets=None, segs=None)
        t = torch.arange(T, device=dev)
        near = (t[:, None] - t[None, :] < w) if w else None              # (T, T) within the window
        causal = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
        if not c.doc_mask:
            if w and grids:
                info["fast"] = (causal & near)[None, None].expand(B, 1, T, T)
            return info
        resets = doc_starts(idx, c.tokenizer)
        doc = resets.cumsum(1)
        if grids:
            fast = (doc[:, None, :, None] == doc[:, None, None, :]) & causal
            info["fast"] = fast & near if w else fast
        gdoc = doc[:, K - 1::K]                                          # a group's document: its last token's
        gresets = torch.zeros(B, N, dtype=torch.bool, device=dev)
        gresets[:, 1:] = gdoc[:, 1:] != gdoc[:, :-1]
        if grids:
            gc = torch.ones(N, N, dtype=torch.bool, device=dev).tril()
            info["slow"] = (gdoc[:, None, :, None] == gdoc[:, None, None, :]) & gc
        last = (t // K) * K + K - 1                                      # the last token of each token's own group
        info["keep"] = doc.gather(1, last[None].expand(B, T)) == doc
        seen = ((t + 1) // K) * K - 1                                    # the last token of the group it receives (-1: none)
        info["valid"] = (doc.gather(1, seen.clamp_min(0)[None].expand(B, T)) == doc) | (seen < 0)[None]
        info["resets"], info["gresets"] = resets, gresets
        if "M" in c.slow_pattern:
            from memtrix import memory
            if memory.packed_docs(dev) or (memory._OP_OK and not idx.is_cuda):
                info["segs"] = memory.segments(gresets, hints=True)
        return info

    def masks(self, idx, info):
        """The masks the attention layers get in training: the yes/no grids, or with flex attention their block form
        (the same rules), built once for all layers of a stream."""
        if not (self._flex and self._compiled is not None):
            return info["fast"], info["slow"]
        from torch.nn.attention.flex_attention import create_block_mask
        c = self.cfg
        B, T = idx.shape
        w = c.window if c.window and c.window < T else 0
        if info["resets"] is not None:
            doc = info["resets"].cumsum(1)

            def fast_rule(b, h, q, kv):
                ok = (doc[b, q] == doc[b, kv]) & (q >= kv)
                return ok & (q - kv < w) if w else ok
            fast = create_block_mask(fast_rule, B, None, T, T, device=idx.device, _compile=idx.is_cuda)
            slow = doc_block_mask(info["gresets"])
        else:
            def fast_rule(b, h, q, kv):
                return (q >= kv) & (q - kv < w)
            fast = create_block_mask(fast_rule, None, None, T, T, device=idx.device, _compile=idx.is_cuda) if w else None
            slow = None
        return fast, slow

    # ------------------------------------------------------------------ the two clocks
    def _trunk(self, idx, fast=None, slow=None, keep=None, valid=None, gresets=None, segs=None, resets=None):
        """Tokens (B, T), T a multiple of the group -> token vectors after the final norm (B, T, D)."""
        c = self.cfg
        B, T = idx.shape
        K, N = c.group, T // c.group
        x = self.embed(idx)
        for i, b in enumerate(self.enc):
            if i == c.phrase_at and c.phrase_rows:
                x = x + self._phrase(idx, x, resets)
            x, _ = b(x, None, None, fast)
        if c.phrase_at == c.enc_layers and c.phrase_rows:
            x = x + self._phrase(idx, x, resets)
        # pack: the group's tokens side by side (tokens of an earlier document left out) -> one slow position
        xp = x if keep is None else x * keep[..., None].to(x.dtype)
        g = self.pack(xp.reshape(B, N, K * c.d_model))
        for b in self.slow:
            g, _ = b(g, None, gresets, slow, segs)
        s = self.slow_norm(g)
        # unpack, one group later: token t receives the state of the latest group that ends at or before t
        s = torch.cat([self.slow_start.to(s.dtype).expand(B, 1, -1), s], 1)          # (B, N + 1, Ds)
        u = self.unpack(s).reshape(B, (N + 1) * K, c.d_model)[:, 1:T + 1]
        if valid is not None:                                                         # a new document: the start state
            start = self.unpack(self.slow_start.to(u.dtype)).reshape(K, c.d_model)
            start = start.repeat(N + 1, 1)[1:T + 1]
            u = torch.where(valid[..., None], u, start[None])
        x = x + u
        for b in self.dec:
            x, _ = b(x, None, None, fast)
        return self.norm(x)

    def _phrase(self, idx, x, resets):
        doc = resets.cumsum(1) if resets is not None else None
        rows = self.phrase.rows(idx, doc)                                             # (B, T, n_tab)
        e = self.embed_phrase(rows).reshape(*idx.shape, -1).to(x.dtype)
        return self.phrase(x, e)

    def _run(self, idx, compiled):
        flex = compiled and self._flex and self._compiled is not None
        info = self.doc_info(idx, grids=not flex)
        fast, slow = self.masks(idx, info) if flex else (info["fast"], info["slow"])
        f = (self._compiled if compiled and self._compiled is not None else self._trunk)
        return f(idx, fast, slow, info["keep"], info["valid"], info["gresets"], info["segs"], info["resets"]), info

    def _padded(self, idx):
        """Any length: pad on the right to a whole number of groups (later tokens never change earlier ones)."""
        T = idx.shape[1]
        pad = (-T) % self.cfg.group
        return (F.pad(idx, (0, pad)) if pad else idx), T

    # ------------------------------------------------------------------ training and scoring
    def prepare_kernels(self, device):
        if "M" in self.cfg.slow_pattern:
            from memtrix import memory
            memory.fla_check(device)
            if self.cfg.memory_op:
                memory.op_ok(device)
            if self.cfg.fused_gates:
                memory.fused_gates_ok(device)

    def enable_compile(self):
        dev = self.embed.weight.device
        self.prepare_kernels(dev)
        self._compiled = torch.compile(self._trunk)
        self._flex = bool(self.cfg.flex_attn and dev.type == "cuda" and flex_check(dev, self.cfg.fast_cfg()))

    def head(self, h):
        logits = F.linear(h, self.embed.weight)
        cap = self.cfg.softcap
        return cap * torch.tanh(logits / cap) if cap else logits

    def _body(self, idx):
        """The full scores (B, T, V), any length (the probes, HellaSwag and the evals want them)."""
        x, T = self._padded(idx)
        h, _ = self._run(x, compiled=False)
        return self.head(h[:, :T])

    def losses(self, idx, targets, mentor=None, kd_weight=0.0):
        """The next-word loss in slices, plus the mentor's lesson when there is one (Bee's API)."""
        x, T = self._padded(idx)
        h, info = self._run(x, compiled=True)
        h = h[:, :T]
        th = tW = None
        if mentor is not None and kd_weight > 0:          # the mentor reads with the same document rule (Bee3)
            mask = None
            if info["resets"] is not None:
                doc = info["resets"].cumsum(1)[:, :T]
                mask = (doc[:, None, :, None] == doc[:, None, None, :]) & torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
            th, tW = mentor.hidden(idx, mask)
        if self.fused_loss:
            from oazy.lmloss import head_loss
            loss, ce, kd = head_loss(h, self.embed.weight, targets, self.cfg.softcap, th, tW, kd_weight, self.loss_chunk)
        else:
            from oazy.lmloss import plain_loss
            loss, ce, kd = plain_loss(h, self.embed.weight, targets, self.cfg.softcap, th, tW, kd_weight)
        stats = {"ce": ce}
        if th is not None:
            stats["kd"] = kd
        return loss, stats

    def forward(self, idx, targets=None):
        if targets is None:
            return self._body(idx), None, {}
        loss, stats = self.losses(idx, targets)
        return None, loss, stats

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=100, temperature=0.8, top_k=40, eos=None, rep=1.0):
        """Writes by reading the whole text again for every new token (simple and exact; the fast generation path,
        with the slow stream waking once per group, is rung R8 of the plan)."""
        dev = self.embed.weight.device
        ids = list(prompt_ids)[-self.cfg.max_len:]
        out = []
        for _ in range(max_new_tokens):
            logits = self._body(torch.tensor([ids[-self.cfg.max_len:]], device=dev))
            nxt = sample_logits(logits[0, -1], temperature, top_k, recent=ids[-64:], rep=rep)
            if eos is not None and nxt == eos:
                break
            out.append(nxt)
            ids.append(nxt)
        return out, None

    # ------------------------------------------------------------------ the profiler's pieces
    def profile_pieces(self, idx):
        """oazy/profile.py: each part of a training pass on its own, as (name, callable returning a tensor to push a
        gradient through, how many such layers the model has). Inputs are random vectors of the right shapes, so the parts are timed alone."""
        c = self.cfg
        B, T = idx.shape
        K, N, dev = c.group, T // c.group, idx.device
        flex = self._flex and self._compiled is not None
        info = self.doc_info(idx, grids=not flex)
        fast, slow = self.masks(idx, info) if flex else (info["fast"], info["slow"])
        x = torch.randn(B, T, c.d_model, device=dev, requires_grad=True)
        gs = torch.randn(B, N, c.slow_d, device=dev, requires_grad=True)
        out = []
        if c.phrase_rows:
            out.append(("phrase memory (lookups + gate)", lambda: self._phrase(idx, x, info["resets"]), 1))
        nf = c.enc_layers + c.dec_layers
        if nf:
            fb = (self.enc or self.dec)[0]
            out.append((f"fast layers x{nf}", lambda: fb(x, None, None, fast)[0], nf))
        out.append(("pack + unpack", lambda: self.unpack(self.pack(x.reshape(B, N, K * c.d_model))), 1))
        for kind in sorted(set(c.slow_kinds())):
            sb = next(bb for bb in self.slow if bb.kind == kind)
            n = c.slow_kinds().count(kind)
            name = "slow attention layers" if kind == "A" else "slow memory layers"
            out.append((f"{name} x{n}", lambda sb=sb: sb(gs, None, info["gresets"], slow, info["segs"])[0], n))
        return out
