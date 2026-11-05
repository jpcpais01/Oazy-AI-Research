"""Fern2: Fern's coarse-to-fine unfolding, raised the fast way (Ember's recipe), and smaller.

Fern2 is NOT a next-word model. Exactly like Fern (fern/model.py) it writes 16 words at a time in
5 rounds: where the passage is going first (word 16), then the middle (word 8), then the middle of
each half, down to the last details. Its loss is the same exact per-word loss as Fern's and GPT-2's.

What changes is the size and how it is raised:

  * Size: 9 reader blocks + 3 unfolder blocks = 118.6M parameters (Fern: 18 + 6 = 198M), so it is
    compared with GPT-2 small (124M) at a fair size and costs ~40% less per word than Fern.

  * It grows (from Ember). Both halves start a third deep and copy their own top blocks to grow:
    reader 3 -> 6 -> 9 blocks and unfolder 1 -> 2 -> 3 blocks, at 10% and 25% of training.

  * The reader also learns to say the next word (new). Fern's reader only learned through the
    unfolder's glance at its last 16 words. Fern2's reader also guesses the word that follows every
    word it reads (through the same word table, so no new weights). Every word of the text now
    teaches the reader directly, which is a much denser lesson, and it is what lets the mentor in.

  * It learns from a mentor (from Ember, adapted). GPT-2 medium can only answer one question: "what
    is the next word, given everything before it?". The reader asks exactly that question, so the
    mentor's full guesses ("70% cat, 20% dog...") are an exact lesson for it, on every word. The
    unfolder asks a different question ("what fills this gap, given both sides?"), which a next-word
    model cannot answer without trying every possible gap filling, so the unfolder learns from the
    real text only, standing on a reader that reads like GPT-2 medium. The mentor's share fades to
    zero halfway through training; after that the reader keeps its own next-word lesson at a small
    weight, which costs nothing and keeps it sharp.

  * Muon (from Ember) on every weight grid of both halves, AdamW on the rest.

What the training loss is made of, at a moment where the mentor's share is m (0.6 -> 0):
    Fern's per-word loss  +  reader_weight x [ (1 - m) x reader's next-word loss  +  m x KL(mentor || reader) ]
The reported loss (training curve, validation, HellaSwag) is Fern's per-word loss alone.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from memtrix.model import RMSNorm
from ember.model import parse_plan, depth_at, grow_stack, stack_depth_in, _quiet_reader
from fern.model import Fern, FernConfig


@dataclass
class Fern2Config(FernConfig):
    n_layer: int = 9                    # reader blocks when fully grown (Fern: 18)
    unfold_layers: int = 3              # unfolder blocks when fully grown (Fern: 6)
    grow: str = "3:0,6:0.1,9:0.25"      # reader depth by fraction of training tokens
    unfold_grow: str = "1:0,2:0.1,3:0.25"
    grow_mode: str = "stack"            # "stack": copies act at once; "quiet": copies start silent
    reader_weight: float = 0.5          # share of the reader's own next-word lesson (0 = off)


def _quiet_unfold(b):
    nn.init.zeros_(b.glance.out.weight)
    nn.init.zeros_(b.down.weight)


class Fern2(Fern):
    def __init__(self, cfg: Fern2Config):
        super().__init__(cfg)                          # all blocks, Fern's init ...
        D = cfg.d_model
        # the reader's own next-word guess goes through its own small layer, so that guessing the next
        # word does not crowd out what the unfolder needs from the reader (what was actually read)
        self.reader_proj = nn.Linear(D, D, bias=False)
        nn.init.normal_(self.reader_proj.weight, std=0.02)
        self.reader_norm = RMSNorm(D)                  # (then the tied word table)
        self._block_params = sum(p.numel() for p in self.blocks[0].parameters())
        self._unfold_params = sum(p.numel() for p in self.unfold[0].parameters())
        self.blocks = nn.ModuleList(list(self.blocks)[:self.plans[0][0][1]])     # ... keep the first few
        self.unfold = nn.ModuleList(list(self.unfold)[:self.plans[1][0][1]])

    # ------------------------------------------------------------------ growing
    @property
    def plans(self):
        return (parse_plan(self.cfg.grow, self.cfg.n_layer, "grow"),
                parse_plan(self.cfg.unfold_grow, self.cfg.unfold_layers, "unfold_grow"))

    @property
    def depth(self):
        return len(self.blocks)

    def depth_text(self):
        return f"{len(self.blocks)} reader + {len(self.unfold)} unfolder blocks"

    def full_params(self):
        return (self.num_params() + (self.cfg.n_layer - len(self.blocks)) * self._block_params
                + (self.cfg.unfold_layers - len(self.unfold)) * self._unfold_params)

    def grow_to(self, progress: float) -> bool:
        quiet = self.cfg.grow_mode == "quiet"
        rp, up = self.plans
        a = grow_stack(self.blocks, depth_at(rp, progress), _quiet_reader if quiet else None)
        b = grow_stack(self.unfold, depth_at(up, progress), _quiet_unfold if quiet else None)
        if (a or b) and self._compiled is not None:
            self.enable_compile()                      # the body changed: compile it again
        return a or b

    def load_state_dict(self, sd, strict=True, assign=False):
        """Checkpoints may come from any depth: grow (with placeholder copies) to match, then load."""
        for stack, prefix in ((self.blocks, "blocks"), (self.unfold, "unfold")):
            n = stack_depth_in(sd, prefix)
            while len(stack) < n:
                stack.append(copy.deepcopy(stack[-1]))
        if "reader_proj.weight" not in sd and "reader_norm.weight" in sd:      # checkpoints from before reader_proj
            sd = {**sd, "reader_proj.weight": self.reader_proj.weight.detach().clone()}
        if self._compiled is not None:
            self.enable_compile()
        return super().load_state_dict(sd, strict=strict, assign=assign)

    # ------------------------------------------------------------------ training
    def _both(self, seq, starts, ends, J):
        """One reading, two outputs: the unfolder's word thoughts (B, N, D) and the reader's (B, N, D)."""
        N = seq.shape[1] - 1
        H = self.reader(seq[:, :N])
        h, base = self._hidden(seq, starts, ends, J, H)
        return self._scatter(h, base, N), H

    def enable_compile(self):
        self._compiled = torch.compile(self._core)          # scoring (Fern's path)
        self._compiled_both = torch.compile(self._both)     # training (also the reader's guesses)

    def reader_logits(self, idx):
        """The reader's own next-word guesses (B, T, vocab) for the words it has read."""
        return F.linear(self.reader_norm(self.reader_proj(self.reader(idx))), self.embed.weight)

    def forward(self, idx, targets=None):
        if targets is None or not self.training or self.cfg.reader_weight <= 0:
            return super().forward(idx, targets)     # scoring: Fern's per-word chances only
        seq = torch.cat([idx[:, :1], targets], 1)
        B, N = idx.shape
        dev = idx.device
        r0 = int(torch.randint(0, self.cfg.passage, (1,))) if self.cfg.shift else 0
        starts = torch.full((B,), r0, dtype=torch.long, device=dev)
        ends = torch.full((B,), N, dtype=torch.long, device=dev)
        J = -(-(N - r0) // self.cfg.passage)
        fn = self._compiled_both if self._compiled is not None else self._both   # (trainer may switch compile off)
        hid, H = fn(seq, starts, ends, J)
        logits = F.linear(hid, self.embed.weight)
        tgt = torch.where(self.rounds_of(starts, N) >= 0, targets, torch.full_like(targets, -1))
        ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-1)
        rlog = F.linear(self.reader_norm(self.reader_proj(H)), self.embed.weight)   # word t -> guess of word t + 1
        rce = F.cross_entropy(rlog.float().reshape(-1, rlog.size(-1)), targets.reshape(-1))
        stats = {"ce": ce.detach(), "reader_ce": rce.detach(), "_ce": ce, "_reader_ce": rce, "kd_logits": rlog}
        return logits, ce + self.cfg.reader_weight * rce, stats

    def mix_mentor(self, loss, stats, kd, m):
        """The mentor's lesson goes to the reader only (see the top of this file)."""
        return stats["_ce"] + self.cfg.reader_weight * ((1 - m) * stats["_reader_ce"] + m * kd)
