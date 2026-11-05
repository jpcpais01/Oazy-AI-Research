"""Bee0.2B v2: Bee0.2B with a smarter, cheaper mentor (SmolLM2-360M) and SmolLM2's tokenizer.

Still a NEXT-WORD model: it reads text and guesses the very next token, like GPT-2 and Bee0.2B.

The body is exactly Bee0.2B's (bee/model.py: 18 Memtrix memory layers + 6 gated attention layers,
3:1, d=768, SwiGLU 1792, tied word table, NoPE, soft cap). Two things change, and only these two, so
that v1 against v2 measures the mentor:

  tokenizer  SmolLM2's 49,152 pieces instead of GPT-2's 50,257 (a mentor and its student must cut
             text the same way). The word table is 0.9M numbers smaller: 203.3M parameters.
  mentor     SmolLM2-360M (Hugging Face, 2024; HellaSwag ~54%) instead of GPT-2 large (774M, ~45%):
             a better teacher at about half the reading cost. It can read 8,192 tokens at once.
"""
from __future__ import annotations

from dataclasses import dataclass

from bee.model import Bee, BeeConfig


@dataclass
class Bee2Config(BeeConfig):
    vocab_size: int = 49152      # SmolLM2's pieces (already a multiple of 64)
    tokenizer: str = "smollm2"   # oazy/tok.py


class Bee2(Bee):
    pass
