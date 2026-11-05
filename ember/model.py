"""Ember: a mind raised to learn fast.

Ember is still a NEXT-WORD model: it reads text and guesses the very next word, exactly like GPT-2
and Memtrix, and it is scored the same way (loss, HellaSwag). Its body is Memtrix's learn-while-reading
memory (memtrix/model.py). What is new is how it grows up, because the cost we care about now is
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

Each idea is switchable, so their gains can be measured one by one against Memtrix (same body):
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

from memtrix.model import Memtrix, MemtrixConfig, sample_logits  # noqa: F401  (sample_logits re-exported)


@dataclass
class EmberConfig(MemtrixConfig):
    # depth plan, "depth:when" (fraction of training tokens); the last depth must be n_layer
    grow: str = "4:0,8:0.1,12:0.25"
    grow_mode: str = "stack"     # "stack": copies act at once; "quiet": copies start silent (their outputs zeroed)


def parse_plan(text: str, final: int, what: str = "grow"):
    """"4:0,8:0.1,12:0.25" -> [(0.0, 4), (0.1, 8), (0.25, 12)]: depth by fraction of training tokens."""
    plan = sorted((float(w), int(d)) for d, w in (s.split(":") for s in text.split(",")))
    if plan[0][0] != 0 or plan[-1][1] != final or any(b[1] <= a[1] for a, b in zip(plan, plan[1:])):
        raise ValueError(f"bad {what} plan {text!r}: start at 0, grow every time, end at {final}")
    return plan


def depth_plan(cfg: EmberConfig):
    return parse_plan(cfg.grow, cfg.n_layer)


def depth_at(plan, progress: float) -> int:
    return max(d for w, d in plan if progress >= w)


def grow_stack(blocks: nn.ModuleList, target: int, silence=None):
    """Make a stack of blocks `target` deep by copying its top blocks and putting the copies on top
    (at most doubling per pass). Each copy starts as its original; `silence(block)` (optional) zeroes
    a copy's outputs so it starts out doing nothing. Returns True if the stack grew."""
    grew = False
    while len(blocks) < target:
        d = len(blocks)
        n = min(target - d, d)
        for b in list(blocks)[d - n:d]:
            nb = copy.deepcopy(b)
            if silence is not None:
                silence(nb)
            blocks.append(nb)
        grew = True
    return grew


def stack_depth_in(sd, prefix: str) -> int:
    """How many blocks a checkpoint's stack `prefix` (e.g. "blocks") holds."""
    pat = re.compile(rf"{re.escape(prefix)}\.(\d+)\.")
    return 1 + max(int(m.group(1)) for k in sd if (m := pat.match(k)))


def _quiet_reader(b):
    nn.init.zeros_(b.mem.out.weight)
    if b.mlp is not None:
        nn.init.zeros_(b.mlp.down.weight)


class Ember(Memtrix):
    def __init__(self, cfg: EmberConfig):
        super().__init__(cfg)                       # builds all n_layer blocks (same init as Memtrix) ...
        self._block_params = sum(p.numel() for p in self.blocks[0].parameters())
        d0 = depth_plan(cfg)[0][1]
        self.blocks = nn.ModuleList(list(self.blocks)[:d0])     # ... and keeps only the first few

    @property
    def depth(self):
        return len(self.blocks)

    def depth_at(self, progress: float) -> int:
        return depth_at(depth_plan(self.cfg), progress)

    def full_params(self):
        """Parameters once fully grown (the size it is compared at)."""
        return self.num_params() + (self.cfg.n_layer - self.depth) * self._block_params

    def grow_to(self, progress: float) -> bool:
        """Grow to the depth the plan asks for at this point of training. True if it grew."""
        quiet = _quiet_reader if self.cfg.grow_mode == "quiet" else None
        if not grow_stack(self.blocks, self.depth_at(progress), quiet):
            return False
        if self._compiled is not None:
            self.enable_compile()                          # the body changed: compile it again
        return True

    def load_state_dict(self, sd, strict=True, assign=False):
        """Checkpoints may come from any depth: grow (with placeholder copies) to match, then load."""
        n = stack_depth_in(sd, "blocks")
        while self.depth < n:
            self.blocks.append(copy.deepcopy(self.blocks[-1]))
        if self._compiled is not None:
            self.enable_compile()
        return super().load_state_dict(sd, strict=strict, assign=assign)
