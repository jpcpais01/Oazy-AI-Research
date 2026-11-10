"""Bee6-MoE: Bee6-200M whose MLPs are fine-grained experts. 872M parameters, 202M of them at work per word, the same as
Bee6. Still a NEXT-WORD model: it reads text and guesses the very next token, like every Bee, and it is scored the same way.

João's ask (Nov 10, "Build Bee6 with fine-grained experts"): keep what Bee6-200M pays for on every word (the word table
and score head, attention, the value tables) and make the thinking part about 8x bigger, each word still doing Bee6's
amount of work. Then train it 10B tokens and more on Emerald.

                         Bee6-200M                      Bee6-MoE
  body                   12 x 1,024, every Bee5 trick   the same
  attention              Bee6's (fused q/k/v, FP8, MoD) the same
  each layer's MLP       one ReLU² MLP of 3,840         1 shared + 80 routed ReLU² experts of 384, 9 routed per word
  per word               3,840 MLP neurons              1 x 384 + 9 x 384 = 3,840 (the same)
  parameters             202.0M                         872M (MLP 765M), 202M at work per word

Where the gain should come from: a mixture of experts with the same work per word learns like a dense model trained with
much more compute (DeepSeekMoE; the fine-grained MoE scaling laws, Krajewski et al. 2024; about 2-3x at 8x the
weights). The price: each routed expert sees only ~1/9 of the words, so it needs many tokens to pay off. That is why it
is for 10B+, not for a 0.5B race. And the routing (gathering words into queues, a 1.25x spare room in every queue, bf16
instead of FP8 for the experts) makes a step ~25-35% slower than Bee6's.

Everything not in the table is Bee6's, and its recipe is Bee6's exactly (RECIPE["bee6moe"] = RECIPE["bee6"]): SmolLM2's
pieces, the windows 512 -> 1,024 -> 2,048, the batch ramp, WSD with the square-root cool-down over the last 30%, NorMuon
(here also on every expert's grids, one by one) + Adam groups, the SmolLM2-360M mentor, the early 3-word lesson.

    python3 -m bee6.moe_tests                    (CPU, ~1 min)
    python3 -m bee6.moe_tests --device cuda      (on the pod: also the compiled step and a speed reading)
    bash runpod/train.sh bee6moe --preset big --tokens 10e9 --micro 32 --data-sub emerald --name bee6moe-emerald-10b
"""
from __future__ import annotations

from dataclasses import dataclass

from .model import Bee6, Bee6Config


@dataclass
class Bee6MoEConfig(Bee6Config):
    experts: int = 80
    expert_top: int = 9
    expert_hidden: int = 384
    shared_experts: int = 1


class Bee6MoE(Bee6):
    def depth_text(self):
        c = self.cfg
        return f"{c.n_layer} layers of {c.d_model}, {c.experts} experts a layer"

    def active_params(self):
        """The weights a word uses: everything but the routed experts it did not pick."""
        c = self.cfg
        unused = c.n_layer * (c.experts - c.expert_top) * 2 * c.d_model * c.expert_hidden
        return self.num_params() - unused
