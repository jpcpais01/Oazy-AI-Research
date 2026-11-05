"""Ember: a mind raised to learn fast.

Ember is still a NEXT-WORD model: it reads text and guesses the very next word, exactly like GPT-2
and Engram, and it is scored the same way (loss, HellaSwag). Its body is Engram's learn-while-reading
memory (engram/model.py). What is new is how it grows up, because the cost we care about now is
*training*, not running. Three ideas, each borrowed from how brains develop:

  1. It grows, like a child's brain.  Ember starts only 4 blocks deep. Early on, a model only learns
     simple things (common words, grammar), and a shallow brain learns those just as well, three
     times cheaper. At 10% of training each block is copied (4 -> 8 blocks, the copies sit on top),
     and at 25% the top four are copied again (8 -> 12). Each copy starts out knowing what the
     original knew, so nothing is learned twice. ("stacking", as in G_stack, Du et al. 2024)

  2. It learns from a mentor.  For the first half of training, Ember is not only told which word
     came next; it also sees what a bigger, already-trained model (OpenAI's GPT-2 medium, 355M,
     same words and tokenizer) would have guessed, with all its doubts ("70% 'cat', 20% 'dog'...").
     One example then teaches much more than one right answer. The mentor's voice fades to zero
     halfway through, so the end of training is Ember on its own. (distillation, Hinton et al. 2015;
     ember/mentor.py)

  3. It learns evenly (Muon).  Its weight grids are updated so that every direction moves the same
     amount, so weak skills catch up with strong ones (oazy/muon.py).

Each idea is switchable, so their gains can be measured one by one against Engram (same body):
  --set grow=12:0      never grow (start at full depth)
  --mentor none        no mentor
  --opt adamw          AdamW instead of Muon
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass

import torch
import torch.nn as nn

from engram.model import Engram, EngramConfig, sample_logits  # noqa: F401  (sample_logits re-exported)


@dataclass
class EmberConfig(EngramConfig):
    # depth plan, "depth:when" (fraction of training tokens); the last depth must be n_layer
    grow: str = "4:0,8:0.1,12:0.25"
    grow_mode: str = "stack"     # "stack": copies act at once; "quiet": copies start silent (their outputs zeroed)


def depth_plan(cfg: EmberConfig):
    plan = sorted((float(w), int(d)) for d, w in (s.split(":") for s in cfg.grow.split(",")))
    if plan[0][0] != 0 or plan[-1][1] != cfg.n_layer or any(b[1] <= a[1] for a, b in zip(plan, plan[1:])):
        raise ValueError(f"bad grow plan {cfg.grow!r}: start at 0, grow every time, end at n_layer={cfg.n_layer}")
    return plan


class Ember(Engram):
    def __init__(self, cfg: EmberConfig):
        super().__init__(cfg)                       # builds all n_layer blocks (same init as Engram) ...
        self._block_params = sum(p.numel() for p in self.blocks[0].parameters())
        d0 = depth_plan(cfg)[0][1]
        self.blocks = nn.ModuleList(list(self.blocks)[:d0])     # ... and keeps only the first few

    @property
    def depth(self):
        return len(self.blocks)

    def depth_at(self, progress: float) -> int:
        return max(d for w, d in depth_plan(self.cfg) if progress >= w)

    def full_params(self):
        """Parameters once fully grown (the size it is compared at)."""
        return self.num_params() + (self.cfg.n_layer - self.depth) * self._block_params

    def grow_to(self, progress: float) -> bool:
        """Grow to the depth the plan asks for at this point of training. True if it grew."""
        target = self.depth_at(progress)
        if target <= self.depth:
            return False
        while self.depth < target:
            d = self.depth
            n = min(target - d, d)
            for b in list(self.blocks)[d - n:d]:          # copy the top n blocks, put the copies on top
                nb = copy.deepcopy(b)
                if self.cfg.grow_mode == "quiet":
                    nn.init.zeros_(nb.mem.out.weight)
                    if nb.mlp is not None:
                        nn.init.zeros_(nb.mlp.down.weight)
                self.blocks.append(nb)
        if self._compiled is not None:
            self.enable_compile()                          # the body changed: compile it again
        return True

    def load_state_dict(self, sd, strict=True, assign=False):
        """Checkpoints may come from any depth: grow (with placeholder copies) to match, then load."""
        n = 1 + max(int(m.group(1)) for k in sd if (m := re.match(r"blocks\.(\d+)\.", k)))
        while self.depth < n:
            self.blocks.append(copy.deepcopy(self.blocks[-1]))
        if self._compiled is not None:
            self.enable_compile()
        return super().load_state_dict(sd, strict=strict, assign=assign)
