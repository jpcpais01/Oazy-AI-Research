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
fused into the memory kernel (both checked on the GPU before use). It also grows (8 -> 16 -> 28 layers, see Frog). The recipe differences live in oazy/train.py (RECIPE["frog"]): the mentor (SmolLM2-360M, same
schedule as Bee3) teaches a share of the sequences instead of a share of every word, and steps grow to
524k tokens after the ramp. See frog/README.md.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch.nn as nn

from bee3.model import Bee3, Bee3Config
from ember.model import parse_plan, depth_at, stack_depth_in


FROG_GROW = "8:0,16:0.15,28:0.35"


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
    fused_gates: int = 1         # the memory kernel scales q/k and squashes trust and fade itself (if the operator below is unavailable)
    memory_op: int = 1           # the memory kernel as a compiled operator: the whole stack compiles as one graph
    # it grows: layers by fraction of training, in whole 4-layer groups (3 memory + 1 attention). 8 layers for the
    # first 15%, 16 until 35%, then all 28. Each growth stacks copies of the top groups on top (see grow_to).
    grow: str = "8:0,16:0.15,28:0.35"

    def __post_init__(self):
        if self.grow == FROG_GROW and self.n_layer != 28:     # another depth (a test's small body): no growing unless asked
            self.grow = f"{self.n_layer}:0"


def group_plan(cfg: FrogConfig):
    """The growth plan, checked: whole groups of the pattern, growing every time, ending at n_layer."""
    plan = parse_plan(cfg.grow, cfg.n_layer)
    g = len(cfg.pattern)
    if any(d % g for _, d in plan):
        raise ValueError(f"grow plan {cfg.grow!r}: every depth must be whole groups of {g} ({cfg.pattern})")
    return plan


class Frog(Bee3):
    """Bee3's body at 406M that starts shallow and grows. Early on a model learns the easy, common patterns, which
    a shallow stack learns as well and much more cheaply; when it grows, the copies start as working layers (the
    top groups of the stack, copied and put on top: "stacking", Gong et al. 2019, Du et al. 2024), so nothing is
    relearned. The full 28 layers do the last 65% of the run."""

    def __init__(self, cfg: FrogConfig):
        super().__init__(cfg)                       # builds all 28 layers (the same start as Bee3's) ...
        self._layer_params = sum(p.numel() for b in self.blocks for p in b.parameters()) / len(self.blocks)
        d0 = group_plan(cfg)[0][1]
        self.blocks = nn.ModuleList(list(self.blocks)[:d0])     # ... and keeps the first few groups

    @property
    def depth(self):
        return len(self.blocks)

    def depth_at(self, progress: float) -> int:
        return depth_at(group_plan(self.cfg), progress)

    def depth_text(self):
        kinds = [b.kind for b in self.blocks]
        return f"{len(kinds)} layers deep ({kinds.count('M')} memory + {kinds.count('A')} attention)"

    def full_params(self):
        """Parameters once fully grown (the size it is compared at)."""
        return round(self.num_params() + (self.cfg.n_layer - self.depth) * self._layer_params)

    def _stack_to(self, target: int):
        """Copy the top layers and put the copies on top until `target` deep (at most doubling per pass; whole
        groups, since every depth in the plan is). 8 -> 16 copies all 8; 16 -> 28 copies the top 12."""
        grew = False
        while self.depth < target:
            d = self.depth
            n = min(target - d, d)
            for b in list(self.blocks)[d - n:d]:
                self.blocks.append(copy.deepcopy(b))
            grew = True
        return grew

    def grow_to(self, progress: float) -> bool:
        """Grow to the depth the plan asks for at this point of training. True if it grew."""
        if not self._stack_to(self.depth_at(progress)):
            return False
        if self._compiled is not None:
            self.enable_compile()                          # the body changed: compile it again
        return True

    def load_state_dict(self, sd, strict=True, assign=False):
        """Checkpoints may come from any depth: grow (with placeholder copies) to match, then load."""
        self._stack_to(stack_depth_in(sd, "blocks"))
        if self._compiled is not None:
            self.enable_compile()
        return super().load_state_dict(sd, strict=strict, assign=assign)
