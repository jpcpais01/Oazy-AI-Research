"""Frog: Bee3, twice the size. Still a NEXT-WORD model: it reads text and guesses the very next token,
like GPT-2 and every Bee, and it is scored the same way.

Bee3's body and recipe, scaled to 406M parameters and 4,000M tokens of Tagus:

                       Bee3                          Frog
  width                768                           1,024
  layers               24 (18 memory + 6 attention)  28 (21 memory + 7 attention), the same 3:1
  memories per layer   6 of 128 x 128                8 of 128 x 128
  attention heads      12 query / 4 key-value        16 / 4, heads of 64
  MLP width            1,792                         2,560
  parameters           203M                          406M

Wider rather than deeper: the GPU does wide matrix work well, while each extra layer adds small, slow
steps. Two speed switches, the same maths: flex attention for the document rule and the memory gates
fused into the memory kernel (both checked on the GPU before use). The recipe differences live in oazy/train.py (RECIPE["frog"]): the mentor (SmolLM2-360M, same
schedule as Bee3) teaches a share of the sequences instead of a share of every word, and steps grow to
524k tokens after the ramp. See frog/README.md.
"""
from __future__ import annotations

from dataclasses import dataclass

from bee3.model import Bee3, Bee3Config


@dataclass
class FrogConfig(Bee3Config):
    d_model: int = 1024
    n_layer: int = 28            # MMMA x 7: 21 memory layers + 7 attention layers
    n_head: int = 8              # memories per memory layer (8 x 128 = the width)
    attn_heads: int = 16         # 16 x 64 = the width
    attn_kv_heads: int = 4
    mlp_hidden: int = 2560       # 2.5 x the width (Bee3: 2.33 x); makes the whole model 406M
    # speed only, the same maths (each checked against the plain way on the GPU at start, else not used):
    flex_attn: int = 1           # attention skips the blocks of the score grid that cross documents
    fused_gates: int = 1         # the memory kernel scales q/k and squashes trust and fade itself


class Frog(Bee3):
    pass
