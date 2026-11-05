"""Bee3: Bee_v2_0.2B's body, raised better. Still a NEXT-WORD model: it reads text and guesses the very
next token, like GPT-2, Bee0.2B and Bee_v2_0.2B, and it is scored the same way.

The body is Bee_v2_0.2B's, unchanged (18 Memtrix memory layers + 6 gated attention layers, 3:1, d=768,
203M parameters, SmolLM2's 49,152 pieces), so Bee2 against Bee3 measures the text and the recipe:

  no bleed between documents   a training window holds pieces of several documents. Bee3's attention
                               looks back only inside the current document, its memories start blank at
                               every document start, and even the 4-word "hint" convolution stops at
                               the boundary (doc_mask=1; bee/model.py, memtrix/model.py). Bee2 let a
                               Wikipedia article "continue" into a chat.
  2,048-word windows all run   Bee2 read 1,024 until its mentor left (GPT-2 large could read no more),
                               then 2,048; SmolLM2 reads 8,192, so there is no reason to switch mid-run.

The rest of the recipe lives in oazy/train.py (RECIPE["bee3"]) and bee3/README.md: the mentor stays the
whole run, twice Bee2's learning rates, weight decay on the Muon grids, a batch ramp, a square-root
cool-down, the loss computed in slices (oazy/lmloss.py), and the Tagus text bundle (oazy/bundle.py).
"""
from __future__ import annotations

from dataclasses import dataclass

from bee.model import Bee, BeeConfig


@dataclass
class Bee3Config(BeeConfig):
    vocab_size: int = 49152      # SmolLM2's pieces
    tokenizer: str = "smollm2"
    ctx: str = "2048:0"          # one window length, the whole run
    doc_mask: int = 1            # no bleed between the documents packed in a window


class Bee3(Bee):
    pass
