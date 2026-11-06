"""The lab's two tokenizers (how text is cut into the pieces, "tokens", a model reads and guesses).

  gpt2     GPT-2's 50,257 pieces (OpenAI, 2019). Every model before Bee_v2_0.2B uses it, so GPT-2's
           family can be their mentor.
  smollm2  SmolLM2's 49,152 pieces (Hugging Face, 2024), trained on modern web text, code and maths.
           Bee_v2_0.2B uses it so that SmolLM2-360M can be its mentor (a mentor and its student must
           cut text the same way, or their guesses cannot be compared piece by piece).

Both are "byte-level BPE": any text can be written, nothing is ever unknown. They cut differently,
so a loss per token means slightly different things for each; `bytes_per_token` is what makes
comparisons fair (see oazy/train.py: val_bpb and the "GPT-2-equivalent" loss).

Both objects offer the same few calls (tiktoken's names): encode_ordinary, encode_ordinary_batch,
decode, decode_single_token_bytes, n_vocab, eot_token, plus .name.

Offline / tests: $OAZY_SMOL_TOKENIZER may point at a tokenizer.json (the real one, or a stand-in).
"""
from __future__ import annotations

import functools
import os

import numpy as np

NAMES = ("gpt2", "smollm2")
SMOL_REPO = "HuggingFaceTB/SmolLM2-360M"        # the 135M, 360M and 1.7B models all share this tokenizer


@functools.lru_cache(maxsize=None)
def _byte_decoder():
    """GPT-2's map from the printable stand-in characters of byte-level BPE back to bytes."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class SmolTokenizer:
    """SmolLM2's tokenizer, with tiktoken's method names."""
    name = "smollm2"

    def __init__(self):
        from tokenizers import Tokenizer
        path = os.environ.get("OAZY_SMOL_TOKENIZER")
        self.tk = Tokenizer.from_file(path) if path else Tokenizer.from_pretrained(SMOL_REPO)
        self.n_vocab = self.tk.get_vocab_size(with_added_tokens=True)
        eot = self.tk.token_to_id("<|endoftext|>")
        if eot is None:
            raise SystemExit("this tokenizer has no <|endoftext|> piece")
        self.eot_token = eot
        self._special = {t.content for t in self.tk.get_added_tokens_decoder().values()} \
            if hasattr(self.tk, "get_added_tokens_decoder") else {"<|endoftext|>"}

    def __getstate__(self):                       # worker processes load their own copy
        return {}

    def __setstate__(self, _):
        self.__init__()

    def encode_ordinary(self, text: str):
        return self.tk.encode(text, add_special_tokens=False).ids

    def encode_ordinary_batch(self, texts):
        return [e.ids for e in self.tk.encode_batch(list(texts), add_special_tokens=False)]

    def decode(self, ids):
        return self.tk.decode(list(ids), skip_special_tokens=False)

    def decode_single_token_bytes(self, t: int) -> bytes:
        s = self.tk.id_to_token(int(t))
        if s is None:
            return b""
        if s in self._special:
            return s.encode()
        dec = _byte_decoder()
        try:
            return bytes(dec[c] for c in s)
        except KeyError:                          # not a byte-level piece (should not happen)
            return s.encode()


@functools.lru_cache(maxsize=None)
def get(name: str = "gpt2"):
    if name == "gpt2":
        from .data import get_tokenizer
        return get_tokenizer()
    if name == "smollm2":
        return SmolTokenizer()
    raise ValueError(f"unknown tokenizer {name!r}; known: {NAMES}")


EOT_ID = {"gpt2": 50256, "smollm2": 0}          # <|endoftext|>, without loading the tokenizer


def eot_id(name: str) -> int:
    """The id of <|endoftext|>, the piece that ends every document in the data files."""
    try:
        return EOT_ID[name]
    except KeyError:
        raise ValueError(f"unknown tokenizer {name!r}; known: {NAMES}")


def name_of(model) -> str:
    """Which tokenizer a model reads (from its config; every model before Bee_v2_0.2B reads GPT-2's)."""
    return getattr(getattr(model, "cfg", None), "tokenizer", "gpt2")


def for_model(model):
    return get(name_of(model))


@functools.lru_cache(maxsize=None)
def bytes_per_token(name: str) -> np.ndarray:
    """How many bytes of text each token stands for (special tokens count 0)."""
    enc = get(name)
    out = np.zeros(max(enc.n_vocab, 65536), dtype=np.int64)
    for t in range(enc.n_vocab):
        if t == enc.eot_token:
            continue
        try:
            out[t] = len(enc.decode_single_token_bytes(t))
        except Exception:
            pass
    return out
