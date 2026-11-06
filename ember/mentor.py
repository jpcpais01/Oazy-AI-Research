"""The mentor: a bigger, already-trained model whose guesses Ember learns from (distillation).

Being told "the next word was *cat*" teaches one thing. Seeing a good reader's whole guess
("70% cat, 20% dog, 5% kitten, ...") teaches what was likely, what was close and what was absurd,
for every word of the vocabulary at once. That richer signal is why distilled small models (Gemma 3
270M among them) learn far more per word read.

The mentor here is OpenAI's GPT-2 medium (355M; or gpt2-large / gpt2-xl): same tokenizer as our
data, so no conversion. Bee_v2_0.2B uses SmolLM2-360M (Hugging Face, 2024) instead: smarter, cheaper,
and it reads SmolLM2's own tokenizer, which Bee_v2_0.2B reads too ("smollm2-360m"; also
"smollm2-135m", "smollm2-1.7b"). It only reads (no learning, no gradients), in bf16, and costs roughly 25-50%
of an Ember training step while it is active. Its weight fades linearly from `weight` at the start
to 0 at `until` (fraction of training), after which it is not run at all.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from oazy.gpt import load_openai_gpt2


class Mentor:
    def __init__(self, name: str, device, weight: float = 0.6, until: float = 0.5, compile: bool = True):
        self.name, self.w0, self.until = name, weight, until
        if name == "random":                      # self-tests only: a tiny untrained stand-in (no download)
            from oazy.gpt import GPT, GPTConfig
            self.net = GPT(GPTConfig(vocab_size=50257, n_layer=2, n_head=2, d_model=64)).to(device).eval()
        elif name == "smol-random":               # self-tests only: a tiny untrained SmolLM2-shaped stand-in
            self.net = HFMentor(None, device)
        elif name.startswith("smollm2"):
            self.net = HFMentor(SMOL[name], device)
        else:
            self.net = load_openai_gpt2(device, name)
        if device.type == "cuda":
            self.net = self.net.to(torch.bfloat16)
            if compile:
                self.net.enable_compile()
                self._hidden_compiled = torch.compile(self.net.hidden)
        for p in self.net.parameters():
            p.requires_grad_(False)

    anchor = None          # (progress, weight): fade from here instead of from the start (a run made shorter)
    floor = 0.0            # the share it keeps after `until` (Bee3: 0.25), ...
    off = 0.0              # ... fading to 0 over this last fraction of the run (Bee3: 0.1)
    _hidden_compiled = None
    _mask_form = None      # how this mentor takes a per-document mask: "bool", "float" or "none" (checked once)

    def weight_at(self, progress: float) -> float:
        return fade(progress, self.w0, self.until, self.anchor, self.floor, self.off)

    # ------------------------------------------------------------- the fused way: token vectors + word table
    def head_weight(self):
        return self.net.head_weight()

    def hidden(self, x, mask=None):
        """The mentor's token vectors (B, T, Dt) and its word table (Vt, Dt): the student forms both sets of
        scores slice by slice (oazy/lmloss.py) instead of receiving 49,152 scores per token.
        mask: (B, 1, T, T) bool, which earlier words each word may see (same document), or None."""
        if mask is not None:
            form = self._mask_check(x.device)
            if form == "none":
                mask = None
            elif form == "float":
                fdt = torch.bfloat16 if x.is_cuda else torch.float32
                mask = torch.zeros(mask.shape, dtype=fdt, device=x.device).masked_fill(~mask, torch.finfo(fdt).min)
        L = self.net.cfg.max_len
        f = self._hidden_compiled if self._hidden_compiled is not None else self.net.hidden
        try:
            h = self._pieces(f, x, mask, L)
        except Exception as e:
            if self._hidden_compiled is None:
                raise
            print(f"[mentor] torch.compile failed ({type(e).__name__}: {e}); the mentor reads uncompiled", flush=True)
            self._hidden_compiled = None
            h = self._pieces(self.net.hidden, x, mask, L)
        return h, self.head_weight()

    @staticmethod
    def _pieces(f, x, mask, L):
        if x.shape[1] <= L:
            return f(x, mask)
        return torch.cat([f(x[:, i:i + L], None if mask is None else mask[:, :, i:i + L, i:i + L])
                          for i in range(0, x.shape[1], L)], dim=1)

    @torch.no_grad()
    def _mask_check(self, device):
        """Does this mentor honour a per-document mask? A short text packed after another must read as it does
        apart. Tried as a bool mask, then as 0/-inf. A big model in bf16 never reads the same text twice to
        the last digit (depth and rounding; SmolLM2-360M differs by a few % on its own), so the bar is relative:
        the masked reading must be far closer to the text read apart than the unmasked reading is. The first
        text is one word twelve times, so that bleeding into the second text is large and unmistakable."""
        if self._mask_form is not None:
            return self._mask_form
        g = torch.Generator(device="cpu").manual_seed(0)
        V = int(self.head_weight().shape[0])
        b = torch.randint(1, min(V, 1000), (1, 12), generator=g).to(device)
        a = torch.full_like(b, int(b[0, 0]))
        apart = self.net.hidden(b, None).float()
        both = torch.cat([a, b], 1)
        T = both.shape[1]
        doc = torch.cat([torch.zeros(12), torch.ones(12)]).to(device)
        m = (doc[None, None, :, None] == doc[None, None, None, :]) & torch.ones(T, T, dtype=torch.bool, device=device).tril()
        fdt = torch.bfloat16 if device.type == "cuda" else apart.dtype
        forms = {"bool": m, "float": torch.zeros(m.shape, dtype=fdt, device=device).masked_fill(~m, torch.finfo(fdt).min)}

        def rel(h):
            return (h.float()[:, 12:] - apart).abs().max().item() / (apart.abs().max().item() + 1e-6)

        bleed = rel(self.net.hidden(both, None))
        tried = {}
        for name, mk in forms.items():
            try:
                err = rel(self.net.hidden(both, mk))
            except Exception as e:
                tried[name] = f"refused ({type(e).__name__})"
                continue
            tried[name] = f"{err:.4f}"
            if err < 2e-2 or err < 0.25 * bleed:
                self._mask_form = name
                print(f"[mentor] {self.name} reads each document on its own (mask form: {name}, rel. error {err:.4f}; "
                      f"unmasked it would bleed {bleed:.4f})", flush=True)
                return name
        self._mask_form = "none"
        print(f"[mentor] {self.name} does not take a per-document mask here: it reads whole windows "
              f"(a little bleed from the previous document into its first guesses; tried {tried}, unmasked bleed {bleed:.4f})", flush=True)
        return "none"

    @torch.no_grad()
    def guess(self, x):
        L = self.net.cfg.max_len                  # GPT-2 reads at most 1,024 words: longer windows go in pieces
        try:
            return self._guess(x, L)
        except Exception as e:
            if self.net._compiled is None:
                raise
            print(f"[mentor] torch.compile failed ({type(e).__name__}: {e}); the mentor reads uncompiled", flush=True)
            self.net._compiled = None
            return self._guess(x, L)

    def _guess(self, x, L):
        f = self.net._compiled if self.net._compiled is not None else self.net._body
        if x.shape[1] <= L:
            return f(x)
        return torch.cat([f(x[:, i:i + L]) for i in range(0, x.shape[1], L)], dim=1)


def fade(progress, w0, until, anchor=None, floor=0.0, off=0.0):
    """The mentor's share of the lesson: from w0 at the start down to `floor` at `until` (fraction of the
    run), held there, then down to 0 over the last `off` of the run. With floor = off = 0 (Bee0.2B, Bee2)
    that is a straight fade from w0 to 0 at `until`, after which the mentor is not run at all. Bee3 keeps
    it the whole way (floor 0.25, off 0.1): its lesson never runs dry for a 200M student.
    anchor=(p0, w): the run's length was changed at progress p0, when the share was w: the first fade
    goes straight from w to the floor at `until` of the new length, so it never jumps."""
    if until <= 0:
        return 0.0
    if anchor is not None:
        p0, w = anchor
        if until <= p0:
            share = floor
        elif progress >= until:
            share = floor
        else:
            share = floor + max(0.0, (w - floor) * (1 - max(0.0, progress - p0) / (until - p0)))
    else:
        share = floor + max(0.0, (w0 - floor) * (1 - min(progress, until) / until))
    if off > 0 and progress > 1 - off:
        share *= max(0.0, (1 - progress) / off)
    return max(0.0, share)


SMOL = {"smollm2-135m": "HuggingFaceTB/SmolLM2-135M", "smollm2-360m": "HuggingFaceTB/SmolLM2-360M",
        "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B"}


class HFMentor(torch.nn.Module):
    """A Hugging Face model (SmolLM2) behind the small interface the mentor uses: cfg.max_len, _body, _compiled."""

    class _Cfg:
        def __init__(self, max_len):
            self.max_len = max_len

    def __init__(self, repo, device):
        super().__init__()
        from transformers import AutoModelForCausalLM
        if repo is None:                            # the self-tests' stand-in: SmolLM2's shape, tiny, random
            from transformers import LlamaConfig
            c = LlamaConfig(vocab_size=49152, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=8192,
                            tie_word_embeddings=True, rope_theta=100000.0)
            self.hf = AutoModelForCausalLM.from_config(c, attn_implementation="sdpa")
        else:
            self.hf = AutoModelForCausalLM.from_pretrained(repo, attn_implementation="sdpa")   # (bf16 below, on a GPU)
        self.hf.to(device).eval()
        self.cfg = self._Cfg(int(getattr(self.hf.config, "max_position_embeddings", 2048)))
        self._compiled = None

    def enable_compile(self):
        self._compiled = torch.compile(self._body)

    def _body(self, x):
        return self.hf(input_ids=x, use_cache=False).logits

    def hidden(self, x, mask=None):
        """Token vectors after the final norm (B, T, Dt); the word table is head_weight()."""
        return self.hf.model(input_ids=x, attention_mask=mask, use_cache=False).last_hidden_state

    def head_weight(self):
        return self.hf.lm_head.weight


def mentor_loss(student_logits, mentor_logits, chunk: int = 4):
    """KL(mentor || student) per word, averaged: how far the student's guess is from the mentor's."""
    V = mentor_logits.shape[-1]                              # 50257 (the student pads to 50304); SmolLM2: 49152
    tot, n = 0.0, student_logits.shape[0]
    for i in range(0, n, chunk):                             # in slices, to keep memory bounded
        s = F.log_softmax(student_logits[i:i + chunk, :, :V].float(), dim=-1)
        t = F.log_softmax(mentor_logits[i:i + chunk].float(), dim=-1)
        tot = tot + F.kl_div(s, t, log_target=True, reduction="sum")
    return tot / (n * student_logits.shape[1])
