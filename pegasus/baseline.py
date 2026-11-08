"""Plain-100M: the comparison for Pegasus 1 Mini. A normal NEXT-WORD transformer of the same size (about 100M
parameters), built exactly like Bee4 (gated attention with RoPE, SwiGLU, SmolLM2's pieces, tied word table) and
raised with exactly the same recipe as Pegasus (oazy/train.py RECIPE["plain100"] = RECIPE["pegasus"]), so the two
runs differ only in the architecture: a race of "predict words" against "predict meaning, then words".
"""
from __future__ import annotations

from dataclasses import dataclass

from bee4.model import Bee4, Bee4Config


@dataclass
class Plain100Config(Bee4Config):
    d_model: int = 768
    n_layer: int = 10            # 10 layers of 768: ~106M with the word table, about Pegasus 1 Mini's size
    attn_heads: int = 12
    attn_kv_heads: int = 4
    attn_head_dim: int = 64
    mlp_hidden: int = 2048


class Plain100(Bee4):
    """A plain transformer, no growing (Pegasus does not grow either)."""

    def layout(self):
        c = self.cfg
        return f"{c.n_layer} layers of {c.d_model}, all attention ({c.attn_heads} query / {c.attn_kv_heads} key-value heads), MLP {c.mlp_hidden}"
