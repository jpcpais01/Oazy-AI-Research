"""Train Memtrix, Tide, Fern, Ember, Fern2 or the GPT-2 baseline on FineWeb-Edu. Resumable: re-run the same command.

    python -m oazy.train --model memtrix --data /workspace/oazy_data --runs /workspace/oazy_runs
    python -m oazy.train --model gpt    --data /workspace/oazy_data --runs /workspace/oazy_runs
    python -m oazy.train --bench --hours 2     # measure both models' speed; suggest a token budget

Both models see exactly the same text in the same order, with the same schedule and optimizer.
Everything the dashboard shows is written into the run folder:
  status.json     live status (step, tokens, speed, ETA, latest results)
  metrics.jsonl   training loss, validation loss, HellaSwag, memory checks (memtrix/probes.py)
  samples.jsonl   text written by the model as it learns
  snapshots/      Memtrix only: the probe text read with the lid off
  ckpt.pt         checkpoint (every few minutes, on Ctrl-c and at the end)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .data import Windows, val_set, default_dir, run_dir, few_cpu_threads
from . import tok
from .gpt import GPT, GPTConfig
from memtrix.model import Memtrix, MemtrixConfig
from memtrix import memory
from . import hellaswag, probes, term

KINDS = ["memtrix", "tide", "fern", "ember", "fern2", "fern3", "bee", "bee2", "bee3", "frog", "coral", "bee4", "coralmini", "pegasus", "pegasusmini", "pegasus2", "plain100", "plain100f", "alphastar", "pegasusbeta", "black", "blackt", "gpt"]
NICE = {"memtrix": "Memtrix", "tide": "Tide", "fern": "Fern", "ember": "Ember", "fern2": "Fern2", "fern3": "Fern3",
        "bee": "Bee0.2B", "bee2": "Bee_v2_0.2B", "bee3": "Bee3", "frog": "Frog", "coral": "Coral", "bee4": "Bee4-200M", "coralmini": "Coral Mini",
        "pegasus": "Pegasus 1", "pegasusmini": "Pegasus 1 Mini", "pegasus2": "Pegasus 2 Mini", "plain100": "Plain-100M", "plain100f": "Plain-100F", "alphastar": "Pegasus Alpha Star", "pegasusbeta": "Pegasus Beta", "black": "Code Black",
        "blackt": "Black-T", "gpt": "GPT"}
FAST = ("ember", "fern2", "fern3", "bee", "bee2", "bee3", "frog", "coral", "bee4", "coralmini", "pegasus", "pegasusmini", "pegasus2", "plain100", "plain100f", "alphastar", "pegasusbeta", "black", "blackt")    # raised the fast way: Muon (and a mentor, where MENTOR names one)
# default mentor (the others: gpt2-medium). Code Black and Black-T race without one, so the architectures are compared alone.
MENTOR = {"bee": "gpt2-large", "bee2": "smollm2-360m", "bee3": "smollm2-360m", "frog": "smollm2-360m", "coral": "smollm2-360m",
          "bee4": "smollm2-360m", "coralmini": "smollm2-360m", "pegasus": "smollm2-360m", "pegasusmini": "smollm2-360m", "pegasus2": "smollm2-360m",
          "plain100": "smollm2-360m", "plain100f": "smollm2-360m", "alphastar": "smollm2-360m", "pegasusbeta": "smollm2-360m", "black": "none", "blackt": "none"}
BEES = ("bee", "bee2", "bee3", "frog", "coral", "bee4", "coralmini", "pegasus", "pegasusmini", "pegasus2", "plain100", "plain100f", "alphastar", "pegasusbeta", "black", "blackt")    # the text mix (Tagus), WSD, growing windows
# Bee3's recipe (bee3/README.md): the mentor stays the whole run, weight decay on the Muon grids, a batch ramp,
# a 1-minus-square-root cool-down, twice Bee2's learning rates (the presets' lr_bee3 / muon_lr_bee3)
RECIPE = {"bee3": dict(mentor_floor=0.25, mentor_off=0.1, muon_wd=0.1, batch_ramp="65536:0.1", decay_shape="sqrt")}
# Frog (frog/README.md): Bee3's recipe at 406M, with the mentor on a share of the SEQUENCES instead of a share of every
# word (mentor_share: a share w of the passes learn only the mentor's guesses, the rest only the real next word, so the
# mentor reads only what it teaches), and 524k-token steps after the ramp (the presets' batch_frog)
RECIPE["frog"] = dict(RECIPE["bee3"], mentor_share=1, fp8=1)      # and FP8 matrix products where the GPU has them
# Coral (coral/README.md): Frog's recipe for a 1.0B mixture of experts that must train at ~1M tokens/s. The mentor
# teaches 1 pass in 4 at the start, fading to none by 30% of the run (it helps most early; at full share all run it
# would cost ~40% of the speed). 1M-token steps after a short ramp. It reads Tagus grown to 20,000M tokens in its own
# folder (<data>/tagus20, python3 -m oazy.tagus --run-tokens 20e9 --into tagus20), so Bee3's and Frog's Tagus stays as is.
RECIPE["coral"] = dict(RECIPE["frog"], mentor_weight=0.25, mentor_until=0.3, mentor_floor=0.0, mentor_off=0.0,
                       batch_ramp="262144:0.05", data_sub="tagus20")
# Bee4 (bee4/README.md): Coral's recipe (growing depth, FP8, the batch ramp, Muon with weight decay, the square-root
# cool-down, Tagus20) on a dense 198M all-attention body, with the mentor kept on the WHOLE run: SmolLM2-360M (MENTOR;
# João, Nov 7: the fastest of the two he weighed, 1.7B was the other) teaches 1 pass in 4 at the start, 1 in 10 from 30%
# on, fading to none over the last 10% (about 12% of the passes over the run). Planned for 10,000M tokens.
# 524k-token steps after the ramp (the presets' batch_bee4): at 200M and 10,000M tokens, about 19,000 steps.
RECIPE["bee4"] = dict(RECIPE["coral"], mentor_weight=0.25, mentor_until=0.3, mentor_floor=0.1, mentor_off=0.1,
                      decay=0.3,   # João, Nov 7: the cool-down (learning-rate fall + cool-down text) over the last 30%, from 70%
                      fp8=0)       # B200 speedcheck, Nov 7: FP8 was slower (445k vs 475k tok/s) and moved the gradients
# Coral Mini (coralmini/README.md): Coral's experts at Bee4's size (16 experts of 160, top 2, + a shared MLP of 256;
# ~198M, ~115M at work per word), raised exactly as Bee4 (João, Nov 7: the speed and training settings always come from
# the latest best model). The expert-only parts (balancing biases, balance loss, Muon on the expert stacks, the fixed
# room per expert) live in the model (coral/model.py).
RECIPE["coralmini"] = dict(RECIPE["bee4"])
# Pegasus 1 (pegasus/README.md): predicts the next thought's meaning, then writes its words (Thinker + Speaker, still
# next-word at the end). Raised exactly as Bee4 (João's rule: the training settings come from the latest best model):
# short windows first, the SmolLM2-360M mentor on a share of the passes, the cool-down over the last 30%, bf16, 524k-token
# steps, Tagus20. No growing (its parts have different depths). Plain-100M, the transformer Pegasus 1 Mini races, gets
# the very same recipe, so the two runs differ only in the architecture.
RECIPE["pegasus"] = dict(RECIPE["bee4"])
RECIPE["pegasusmini"] = dict(RECIPE["bee4"])
RECIPE["plain100"] = dict(RECIPE["bee4"])
# Pegasus 2 Mini (pegasus/model2.py): Pegasus 1 Mini rebuilt after its first run (Nov 8). The very same recipe and
# mentor as Plain-100M, so a race measures the architecture only.
RECIPE["pegasus2"] = dict(RECIPE["plain100"])
# Plain-100F (foresight/model.py, Nov 8): Plain-100M exactly, plus a training-only head that guesses the next sentence's
# content words. The very same recipe, mentor and text, so a race measures the side lesson alone.
RECIPE["plain100f"] = dict(RECIPE["plain100"])
# Pegasus Alpha Star (alphastar/model.py, Nov 8): reader 2 + a core of 4 layers run again and again (Huginn) + speaker 2,
# a fact drawer, and the next-thought lesson on every pass. The very same recipe, mentor and text as Plain-100M.
RECIPE["alphastar"] = dict(RECIPE["plain100"])
# Pegasus Beta (alphastar/beta.py, Nov 9): Alpha Star that learns where to stop (PonderNet). The same recipe again.
RECIPE["pegasusbeta"] = dict(RECIPE["plain100"])
# Code Black (codeblack/README.md) and Black-T, the transformer it races: the same recipe for both, so a race measures
# the architecture. Bee3's polish (Muon weight decay, a batch ramp, the square-root cool-down) and FP8, no mentor. Races
# run on the clock: --preset race --minutes 30 (every schedule follows the training time instead of the token count).
RECIPE["black"] = dict(muon_wd=0.1, batch_ramp="65536:0.1", decay_shape="sqrt", fp8=1)
RECIPE["blackt"] = dict(RECIPE["black"])
# Choosing what to learn from (oazy/select.py; João, Nov 8): each word counts by how much it is still worth learning
# (the model's loss minus the mentor's), and the passages closest to clicking are reread. --select 1 turns it on; once
# the A/B on Plain-100M (with / without, Sado, 0.5B tokens each) is in, select=1 joins the latest best recipe.
VAL_T = 1024                                   # held-out windows are always this long: every model, every run, same text


def say(*a, **k):
    """print in the lab's style (a leading [tag] becomes a coloured label, oazy/term.py); a closed terminal (tmux
    killed, `tee` gone) never crashes training before it saves."""
    term.say(*a, **k)


def summary(info, run_dir, resumed=None):
    """The start of a run, short and aligned (everything else is in config.json)."""
    nice = NICE.get(info["model"], info["model"])
    p, pa = info.get("params") or 0, info.get("params_active")
    size = f"{p / 1e6:,.0f}M parameters" + (f", {pa / 1e6:,.0f}M at work per word" if pa else "")
    term.header(nice, f"next-word model · {size}" if info["model"] in BEES or info["model"] in ("gpt", "memtrix") else size)
    bt = info.get("batch_tokens") or 0
    sched = info.get("schedule", "")
    lr = f"{info['lr']:.1e}" + (f" · Muon {info['muon_lr']:.2g}" if "muon_lr" in info else "") + f" · {sched}" \
        + (f" ({info['decay_shape']} cool-down over the last {info.get('decay', 0.2):.0%})" if info.get("decay_shape") else "")
    mentor = info.get("mentor")
    if mentor and mentor != "none":
        how = "a share of the sequences" if info.get("mentor_share") else "mixed into every word"
        mentor = (f"{mentor} · {how} · {info.get('mentor_weight', 0):.2g} at the start"
                  + (f", none after {info['mentor_until']:.0%}" if not info.get("mentor_floor") else
                     f", {info['mentor_floor']:.2g} from {info['mentor_until']:.0%}"))
    term.rows([
        ("layout", info.get("layout")),
        ("tokens", f"{term.tokens(info['total_tokens'])}" + (f" · {info['minutes']:g} min on the clock" if info.get("minutes") else "")),
        ("steps", f"{term.tokens(bt)} tokens each" + (f" (ramp from {term.tokens(int(info['batch_ramp'].split(':')[0]))})"
                                                       if info.get("batch_ramp") else "") + f" · {info.get('micro')} x {info.get('context')} per pass"),
        ("learning", lr),
        ("mentor", mentor or "none"),
        *([("picks", f"top {info['select_keep']:.0%} of words by excess count fully, the rest x{info['select_floor']:g} "
                     f"(from {info['select_start']:.0%}) · reread {info['replay']:.0%} of rows, up to {info['replay_reads']}x")]
          if info.get("select") else []),
        ("grows", info.get("grow")),
        ("speed", " · ".join(x for x in ["FP8" if info.get("fp8") else ("bf16 (FP8 off)" if "fp8" in info else "bf16"),
                                         "document mask" if info.get("doc_mask") else None] if x)),
        ("text", (info.get("mix") or "")[:70] + ("…" if len(info.get("mix") or "") > 70 else "") if info.get("mix") else None),
        ("device", info.get("gpu")),
        ("run folder", str(run_dir) + (f"  ({resumed})" if resumed else "")),
    ])


def _tide():
    from tide.model import Tide, TideConfig          # tide/ builds on this package
    return Tide, TideConfig


def _fern():
    from fern.model import Fern, FernConfig          # fern/ reads with Memtrix's memories
    return Fern, FernConfig


def _ember():
    from ember.model import Ember, EmberConfig       # ember/ builds on this package
    return Ember, EmberConfig


def _fern2():
    from fern2.model import Fern2, Fern2Config       # fern2/ = Fern's body + Ember's way of training
    return Fern2, Fern2Config


def _fern3():
    from fern3.model import Fern3, Fern3Config       # fern3/ = Fern's unfolding + gist-indexed recall
    return Fern3, Fern3Config


def _bee():
    from bee.model import Bee, BeeConfig             # bee/ = modern hybrid: Memtrix memory layers + attention, 3:1
    return Bee, BeeConfig


def _bee2():
    from bee2.model import Bee2, Bee2Config          # bee2/ = Bee0.2B's body, SmolLM2's tokenizer and mentor
    return Bee2, Bee2Config


def _frog():
    from frog.model import Frog, FrogConfig          # frog/ = Bee3 at 406M (wider and a little deeper)
    return Frog, FrogConfig


def _coral():
    from coral.model import Coral, CoralConfig       # coral/ = 1.0B mixture of experts (32 experts, top 2), ~160M at work per word
    return Coral, CoralConfig


def _bee4():
    from bee4.model import Bee4, Bee4Config          # bee4/ = 198M, all attention, dense, Coral's recipe
    return Bee4, Bee4Config


def _coralmini():
    from coralmini.model import CoralMini, CoralMiniConfig   # coralmini/ = Coral's experts at 200M (16 experts, top 2), Bee4's recipe
    return CoralMini, CoralMiniConfig


def _pegasus():
    from pegasus.model import Pegasus, PegasusConfig     # pegasus/ = Thinker (meaning) + Speaker (words), learned thoughts
    return Pegasus, PegasusConfig


def _pegasus2():
    from pegasus.model2 import Pegasus2, Pegasus2Config   # Pegasus 2 Mini: rich thoughts, a real Speaker, fixed-then-learned thoughts
    return Pegasus2, Pegasus2Config


def _plain100():
    from pegasus.baseline import Plain100, Plain100Config   # pegasus/ = the plain 100M transformer Pegasus 1 Mini races
    return Plain100, Plain100Config


def _plain100f():
    from foresight.model import Plain100F, Plain100FConfig   # foresight/ = Plain-100M + a training-only next-sentence head
    return Plain100F, Plain100FConfig


def _alphastar():
    from alphastar.model import AlphaStar, AlphaStarConfig   # alphastar/ = Pegasus Alpha Star: a looped core, a fact drawer
    return AlphaStar, AlphaStarConfig


def _pegasusbeta():
    from alphastar.beta import PegasusBeta, BetaConfig        # alphastar/beta.py = Pegasus Beta: Alpha Star + its own stop
    return PegasusBeta, BetaConfig


def _black():
    from codeblack.model import Black, BlackConfig   # codeblack/ = two clocks: a fast token stream, a slow stream once per 4 tokens
    return Black, BlackConfig


def _blackt():
    from codeblack.model import BlackT, BlackTConfig  # codeblack/ = Black-T, the modern transformer Code Black races
    return BlackT, BlackTConfig


def _bee3():
    from bee3.model import Bee3, Bee3Config          # bee3/ = Bee2's body, no bleed between documents, the Bee3 recipe
    return Bee3, Bee3Config


PRESETS = {
    # a laptop-CPU test of the whole pipeline (minutes)
    "tiny": dict(
        memtrix=dict(d_model=128, n_layer=4, n_head=2, head_k=64, head_v=64, mlp_hidden=256, max_len=256),
        gpt=dict(d_model=128, n_layer=4, n_head=2, max_len=256),
        tide=dict(refine_hidden=256, lookahead=4),
        fern=dict(n_layer=4, unfold_layers=2, unfold_heads=2, unfold_hidden=256),
        ember=dict(grow="1:0,2:0.1,4:0.25"), mentor="none",
        fern2=dict(n_layer=4, unfold_layers=2, unfold_heads=2, unfold_hidden=256, grow="2:0,4:0.25", unfold_grow="1:0,2:0.25"),
        fern3=dict(n_layer=2, unfold_layers=2, think_layers=1, unfold_heads=2, unfold_hidden=256, max_len=512,
                   grow="1:0,2:0.25", unfold_grow="1:0,2:0.25"),
        bee=dict(d_model=128, n_layer=4, n_head=2, head_k=64, head_v=64, attn_heads=4, attn_kv_heads=2, attn_head_dim=32,
                 mlp_hidden=256, max_len=512, ctx="256:0,512:0.5"),
        coral=dict(n_expert=4, top_k=2, expert_hidden=64, mlp_hidden=64),
        pegasus=dict(d_tok=128, tok_heads=4, tok_kv_heads=2, tok_mlp=256, enc_layers=1, spk_layers=2, window=64,
                     d_think=128, think_layers=2, think_heads=4, think_kv_heads=2, think_mlp=256, head_dim=32,
                     max_len=512, ctx="256:0,512:0.5"),
        black=dict(d_model=128, enc_layers=1, dec_layers=1, attn_heads=4, attn_kv_heads=2, attn_head_dim=32, mlp_hidden=256,
                   window=64, slow_d=256, slow_layers=2, slow_heads=4, slow_kv_heads=2, slow_head_dim=64, slow_mlp=512,
                   phrase_rows=8192, phrase_dim=32, max_len=512, ctx="512:0"),
        plain100f=dict(fore_top=20),       # a test tokenizer has few pieces: leave only the 20 most common out of the bags
        alphastar=dict(fore_top=20, drawer_keys=16, drawer_heads=2, drawer_k=4, drawer_qdim=32, test_max=6, gen_passes=3,
                       passes_max=4, passes_mean=3.0, fore_hidden=128),
        pegasusbeta=dict(ponder_max=4, ponder_prior=2.0, ponder_tokens=256, halt_hidden=32),
        micro=16, accum=1, lr=3e-3, warmup=50, tokens=3_000_000,
        eval_every=250_000, probe_every=1_000_000, hs_every=1_000_000, sample_every=1_000_000, evals_every=1_000_000,
        val_windows=64, hs_quick=200, evals_quick=50, sample_len=40,
    ),
    # a 1-minute smoke test of every code path
    "smoke": dict(
        memtrix=dict(d_model=64, n_layer=2, n_head=1, head_k=64, head_v=64, mlp_hidden=128, max_len=128),
        gpt=dict(d_model=64, n_layer=2, n_head=1, max_len=128),
        tide=dict(refine_hidden=128, lookahead=4),
        fern=dict(n_layer=2, unfold_layers=1, unfold_heads=1, unfold_hidden=128),
        ember=dict(grow="1:0,2:0.3"), mentor="none",
        fern2=dict(n_layer=2, unfold_layers=1, unfold_heads=1, unfold_hidden=128, grow="1:0,2:0.3", unfold_grow="1:0"),
        fern3=dict(n_layer=2, unfold_layers=1, think_layers=1, unfold_heads=1, unfold_hidden=128, max_len=256,
                   grow="1:0,2:0.3", unfold_grow="1:0", recall_warm_until=0.2),
        bee=dict(d_model=64, n_layer=4, n_head=1, head_k=32, head_v=32, attn_heads=2, attn_kv_heads=1, attn_head_dim=32,
                 mlp_hidden=128, max_len=128, ctx="64:0,128:0.5"),
        coral=dict(n_expert=4, top_k=2, expert_hidden=32, mlp_hidden=32),
        pegasus=dict(d_tok=64, tok_heads=2, tok_kv_heads=1, tok_mlp=128, enc_layers=1, spk_layers=2, window=32,
                     d_think=64, think_layers=2, think_heads=2, think_kv_heads=1, think_mlp=128, head_dim=32,
                     max_len=128, ctx="64:0,128:0.5", thought_len=8),
        black=dict(d_model=64, enc_layers=1, dec_layers=1, attn_heads=2, attn_kv_heads=1, attn_head_dim=32, mlp_hidden=128,
                   window=32, slow_d=128, slow_layers=2, slow_heads=2, slow_kv_heads=1, slow_head_dim=64, slow_mlp=256,
                   phrase_rows=1024, phrase_dim=16, max_len=128, ctx="128:0"),
        plain100f=dict(fore_top=20),
        alphastar=dict(fore_top=20, drawer_keys=8, drawer_heads=2, drawer_k=4, drawer_qdim=16, test_max=4, gen_passes=2,
                       passes_max=3, passes_mean=2.0, passes_start=1.0, fore_hidden=64),
        pegasusbeta=dict(ponder_max=3, ponder_prior=2.0, ponder_tokens=128, halt_hidden=16),
        micro=8, accum=2, lr=3e-3, warmup=10, tokens=60_000,
        eval_every=20_000, probe_every=30_000, hs_every=30_000, sample_every=30_000, evals_every=30_000,
        val_windows=16, hs_quick=20, evals_quick=10, sample_len=10,
    ),
    # GPT-2 small size (Memtrix 117M vs GPT-2 124M), ~2 h for both models on one B200 (check with --bench)
    "full": dict(
        memtrix=dict(), gpt=dict(), micro_fern2=16, micro_fern3=4, batch_bee=524288, batch_bee3=262144, batch_frog=524288, batch_coral=1048576, batch_bee4=524288, batch_coralmini=524288, batch_pegasus=524288, batch_pegasusmini=524288, batch_pegasus2=524288, batch_plain100=524288, batch_plain100f=524288, batch_alphastar=524288, batch_pegasusbeta=524288,
        micro=32, accum=8, lr=1.2e-3, lr_bee3=2.4e-3, muon_lr_bee3=0.04, lr_frog=1.8e-3, muon_lr_frog=0.03,
        lr_coral=1.8e-3, muon_lr_coral=0.03, lr_bee4=1.8e-3, muon_lr_bee4=0.03, lr_coralmini=1.8e-3, muon_lr_coralmini=0.03,
        lr_pegasus=1.8e-3, muon_lr_pegasus=0.03, lr_pegasusmini=1.8e-3, muon_lr_pegasusmini=0.03, lr_pegasus2=1.8e-3, muon_lr_pegasus2=0.03, lr_plain100=1.8e-3, muon_lr_plain100=0.03, lr_plain100f=1.8e-3, muon_lr_plain100f=0.03, lr_alphastar=1.8e-3, muon_lr_alphastar=0.03, lr_pegasusbeta=1.8e-3, muon_lr_pegasusbeta=0.03, warmup=400, tokens=2_000_000_000,
        eval_every=50_000_000, probe_every=250_000_000, hs_every=250_000_000, sample_every=250_000_000, evals_every=500_000_000,
        val_windows=128, hs_quick=1000, evals_quick=500, sample_len=120,
    ),
    # the same models, 2.5x the text (e.g. a 2 x B200 pod for ~2.5 h, one model per GPU)
    "big": dict(
        memtrix=dict(), gpt=dict(), micro_fern2=16, micro_fern3=4, batch_bee=524288, batch_bee3=262144, batch_frog=524288, batch_coral=1048576, batch_bee4=524288, batch_coralmini=524288, batch_pegasus=524288, batch_pegasusmini=524288, batch_pegasus2=524288, batch_plain100=524288, batch_plain100f=524288, batch_alphastar=524288, batch_pegasusbeta=524288,
        micro=32, accum=8, lr=1.2e-3, lr_bee3=2.4e-3, muon_lr_bee3=0.04, lr_frog=1.8e-3, muon_lr_frog=0.03,
        lr_coral=1.8e-3, muon_lr_coral=0.03, lr_bee4=1.8e-3, muon_lr_bee4=0.03, lr_coralmini=1.8e-3, muon_lr_coralmini=0.03,
        lr_pegasus=1.8e-3, muon_lr_pegasus=0.03, lr_pegasusmini=1.8e-3, muon_lr_pegasusmini=0.03, lr_pegasus2=1.8e-3, muon_lr_pegasus2=0.03, lr_plain100=1.8e-3, muon_lr_plain100=0.03, lr_plain100f=1.8e-3, muon_lr_plain100f=0.03, lr_alphastar=1.8e-3, muon_lr_alphastar=0.03, lr_pegasusbeta=1.8e-3, muon_lr_pegasusbeta=0.03, warmup=600, tokens=5_000_000_000,
        eval_every=100_000_000, probe_every=200_000_000, hs_every=200_000_000, sample_every=200_000_000, evals_every=500_000_000,
        val_windows=128, hs_quick=1000, evals_quick=500, sample_len=120,
    ),
    # Code Black's races (codeblack/README.md): every run gets the same training time (--minutes 30), and the held-out
    # loss is taken often, so loss-against-minutes curves can be compared at any point. 262k-token steps.
    "race": dict(
        memtrix=dict(), gpt=dict(), batch_black=262144, batch_blackt=262144,
        micro=32, accum=8, lr=1.2e-3, lr_black=2.4e-3, muon_lr_black=0.04, lr_blackt=2.4e-3, muon_lr_blackt=0.04,
        warmup=250, tokens=2_000_000_000,
        eval_every=25_000_000, probe_every=500_000_000, hs_every=500_000_000, sample_every=500_000_000, evals_every=500_000_000,
        val_windows=128, hs_quick=1000, evals_quick=500, sample_len=80,
    ),
}

# what every run writes at each check (the dashboard's "What they write", one box each). The fourth, a story, came
# Nov 7 2026; a run that started before it writes it from its next restart (or now: python3 -m oazy.writenow).
SAMPLE_PROMPTS = ["The water cycle is", "In 1905, Albert Einstein", "The most important thing about learning is",
                  "Once upon a time, in a small village by the sea,"]


def parse_overrides(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def build(kind, preset, overrides=None):
    p = PRESETS[preset]
    if kind == "memtrix":
        cfg = MemtrixConfig(**{**p["memtrix"], **(overrides or {})})
        return Memtrix(cfg), cfg
    if kind == "tide":
        Tide, TideConfig = _tide()
        cfg = TideConfig(**{**p["memtrix"], **p.get("tide", {}), **(overrides or {})})
        return Tide(cfg), cfg
    if kind == "fern":
        Fern, FernConfig = _fern()
        cfg = FernConfig(**{**p["memtrix"], **p.get("fern", {}), **(overrides or {})})
        return Fern(cfg), cfg
    if kind == "ember":
        Ember, EmberConfig = _ember()
        cfg = EmberConfig(**{**p["memtrix"], **p.get("ember", {}), **(overrides or {})})
        return Ember(cfg), cfg
    if kind == "fern2":
        Fern2, Fern2Config = _fern2()
        cfg = Fern2Config(**{**p["memtrix"], **p.get("fern2", {}), **(overrides or {})})
        return Fern2(cfg), cfg
    if kind == "fern3":
        Fern3, Fern3Config = _fern3()
        cfg = Fern3Config(**{**p["memtrix"], **p.get("fern3", {}), **(overrides or {})})
        return Fern3(cfg), cfg
    if kind == "bee":
        Bee, BeeConfig = _bee()
        cfg = BeeConfig(**{**p.get("bee", {}), **(overrides or {})})
        return Bee(cfg), cfg
    if kind == "bee2":
        Bee2, Bee2Config = _bee2()
        cfg = Bee2Config(**{**p.get("bee", {}), **p.get("bee2", {}), **(overrides or {})})
        return Bee2(cfg), cfg
    if kind == "bee3":
        Bee3, Bee3Config = _bee3()
        small = {k: v for k, v in p.get("bee", {}).items() if k != "ctx"}    # the test presets' small body; Bee3 keeps one window length
        cfg = Bee3Config(**{**small, **({"ctx": f"{small['max_len']}:0"} if "max_len" in small else {}),
                            **p.get("bee3", {}), **(overrides or {})})
        return Bee3(cfg), cfg
    if kind == "frog":
        Frog, FrogConfig = _frog()
        small = {k: v for k, v in p.get("bee", {}).items() if k != "ctx"}    # the test presets' small body, as Bee3
        cfg = FrogConfig(**{**small, **({"ctx": f"{small['max_len']}:0"} if "max_len" in small else {}),
                            **p.get("frog", {}), **(overrides or {})})
        return Frog(cfg), cfg
    if kind in ("black", "blackt"):
        cls, Cfg = _black() if kind == "black" else _blackt()
        small = {k: v for k, v in p.get("bee", {}).items() if k != "ctx"}    # the test presets' small body
        if kind == "black":            # its own small body (fast and slow streams)
            small = dict(p.get("black", {}))
        elif small:                    # Black-T on a test preset: Bee's small body, set by hand
            small = {**small, "size": "", "ctx": f"{small['max_len']}:0"}
        cfg = Cfg(**{**small, **(overrides or {})})
        return cls(cfg), cfg
    if kind == "coral":
        Coral, CoralConfig = _coral()
        small = {k: v for k, v in p.get("bee", {}).items() if k != "ctx"}    # the test presets' small body, as Bee3
        cfg = CoralConfig(**{**small, **({"ctx": f"{small['max_len']}:0"} if "max_len" in small else {}),
                             **p.get("coral", {}), **(overrides or {})})
        return Coral(cfg), cfg
    if kind == "bee4":
        Bee4, Bee4Config = _bee4()
        small = {k: v for k, v in p.get("bee", {}).items() if k != "ctx"}    # the test presets' small body, as Bee3
        cfg = Bee4Config(**{**small, **({"ctx": f"{small['max_len']}:0"} if "max_len" in small else {}),
                            **p.get("bee4", {}), **(overrides or {})})
        return Bee4(cfg), cfg
    if kind == "coralmini":
        CoralMini, CoralMiniConfig = _coralmini()
        small = {k: v for k, v in p.get("bee", {}).items() if k != "ctx"}    # the test presets' small body, as Bee3
        cfg = CoralMiniConfig(**{**small, **({"ctx": f"{small['max_len']}:0"} if "max_len" in small else {}),
                                 **p.get("coral", {}), **p.get("coralmini", {}), **(overrides or {})})
        return CoralMini(cfg), cfg
    if kind in ("pegasus", "pegasusmini"):
        Pegasus, PegasusConfig = _pegasus()
        size = {"size": "mini" if kind == "pegasusmini" else "full"}
        cfg = PegasusConfig(**{**size, **p.get("pegasus", {}), **(overrides or {})})
        return Pegasus(cfg), cfg
    if kind == "pegasus2":
        Pegasus2, Pegasus2Config = _pegasus2()
        small = {k: v for k, v in p.get("pegasus", {}).items() if k in Pegasus2Config.__dataclass_fields__}
        cfg = Pegasus2Config(**{**small, **p.get("pegasus2", {}), **(overrides or {})})
        return Pegasus2(cfg), cfg
    if kind == "plain100":
        Plain100, Plain100Config = _plain100()
        small = {k: v for k, v in p.get("bee", {}).items() if k != "ctx"}    # the test presets' small body, as Bee4
        cfg = Plain100Config(**{**small, **({"ctx": f"{small['max_len']}:0"} if "max_len" in small else {}),
                                **p.get("plain100", {}), **(overrides or {})})
        return Plain100(cfg), cfg
    if kind == "plain100f":
        Plain100F, Plain100FConfig = _plain100f()
        small = {k: v for k, v in p.get("bee", {}).items() if k != "ctx"}    # the test presets' small body, as Plain-100M
        cfg = Plain100FConfig(**{**small, **({"ctx": f"{small['max_len']}:0"} if "max_len" in small else {}),
                                 **p.get("plain100", {}), **p.get("plain100f", {}), **(overrides or {})})
        return Plain100F(cfg), cfg
    if kind == "alphastar":
        AlphaStar, AlphaStarConfig = _alphastar()
        small = {k: v for k, v in p.get("bee", {}).items() if k not in ("ctx", "n_layer")}   # the test presets' small layers
        cfg = AlphaStarConfig(**{**small, **({"ctx": f"{small['max_len']}:0"} if "max_len" in small else {}),
                                 **p.get("alphastar", {}), **(overrides or {})})
        return AlphaStar(cfg), cfg
    if kind == "pegasusbeta":
        PegasusBeta, BetaConfig = _pegasusbeta()
        small = {k: v for k, v in p.get("bee", {}).items() if k not in ("ctx", "n_layer")}
        cfg = BetaConfig(**{**small, **({"ctx": f"{small['max_len']}:0"} if "max_len" in small else {}),
                            **p.get("alphastar", {}), **p.get("pegasusbeta", {}), **(overrides or {})})
        return PegasusBeta(cfg), cfg
    cfg = GPTConfig(**{**p["gpt"], **(overrides or {})})
    return GPT(cfg), cfg


def load_model(ckpt: Path, device, mmap=False):
    # mmap: read the weights from the file as they are needed, so the optimizer's state never fills the memory
    st = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=mmap)
    kind = st.get("kind", "memtrix")
    if kind == "engram":                      # Memtrix's name before Nov 5
        kind = st["kind"] = "memtrix"
    if kind == "memtrix":
        model = Memtrix(MemtrixConfig.from_dict(st["config"]))
    elif kind == "tide":
        Tide, TideConfig = _tide()
        model = Tide(TideConfig.from_dict(st["config"]))
    elif kind == "fern":
        Fern, FernConfig = _fern()
        model = Fern(FernConfig.from_dict(st["config"]))
    elif kind == "ember":
        Ember, EmberConfig = _ember()
        model = Ember(EmberConfig.from_dict(st["config"]))
    elif kind == "fern2":
        Fern2, Fern2Config = _fern2()
        model = Fern2(Fern2Config.from_dict(st["config"]))
    elif kind == "fern3":
        Fern3, Fern3Config = _fern3()
        model = Fern3(Fern3Config.from_dict(st["config"]))
    elif kind == "bee":
        Bee, BeeConfig = _bee()
        model = Bee(BeeConfig.from_dict(st["config"]))
    elif kind == "bee2":
        Bee2, Bee2Config = _bee2()
        model = Bee2(Bee2Config.from_dict(st["config"]))
    elif kind == "bee3":
        Bee3, Bee3Config = _bee3()
        model = Bee3(Bee3Config.from_dict(st["config"]))
    elif kind == "frog":
        Frog, FrogConfig = _frog()
        model = Frog(FrogConfig.from_dict(st["config"]))
    elif kind == "coral":
        Coral, CoralConfig = _coral()
        model = Coral(CoralConfig.from_dict(st["config"]))
    elif kind == "bee4":
        Bee4, Bee4Config = _bee4()
        model = Bee4(Bee4Config.from_dict(st["config"]))
    elif kind == "coralmini":
        CoralMini, CoralMiniConfig = _coralmini()
        model = CoralMini(CoralMiniConfig.from_dict(st["config"]))
    elif kind in ("pegasus", "pegasusmini"):
        Pegasus, PegasusConfig = _pegasus()
        model = Pegasus(PegasusConfig.from_dict(st["config"]))
    elif kind == "pegasus2":
        Pegasus2, Pegasus2Config = _pegasus2()
        model = Pegasus2(Pegasus2Config.from_dict(st["config"]))
    elif kind == "plain100":
        Plain100, Plain100Config = _plain100()
        model = Plain100(Plain100Config.from_dict(st["config"]))
    elif kind == "plain100f":
        Plain100F, Plain100FConfig = _plain100f()
        model = Plain100F(Plain100FConfig.from_dict(st["config"]))
    elif kind == "alphastar":
        AlphaStar, AlphaStarConfig = _alphastar()
        model = AlphaStar(AlphaStarConfig.from_dict(st["config"]))
    elif kind == "pegasusbeta":
        PegasusBeta, BetaConfig = _pegasusbeta()
        model = PegasusBeta(BetaConfig.from_dict(st["config"]))
    elif kind in ("black", "blackt"):
        cls, Cfg = _black() if kind == "black" else _blackt()
        model = cls(Cfg.from_dict(st["config"]))
    else:
        model = GPT(GPTConfig.from_dict(st["config"]))
    model.load_state_dict(st["model"])
    return model.to(device).eval(), st


def mentor_pass(credit, w):
    """Frog's mentor share: does this pass go to the mentor? A running tally (credit) gains w per pass and
    hands out a whole pass each time it reaches 1, so over any stretch the mentor gets a share w of the
    passes, evenly spread (at w = 0.25: every fourth). Returns (the pass's mentor weight 1 or 0, new credit)."""
    credit += w
    if credit >= 1.0 - 1e-9:
        return 1.0, credit - 1.0
    return 0.0, credit


def lr_at(progress, step, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    return base * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def lr_wsd(progress, step, base, warmup, decay=0.2, shape="linear"):
    """Warm up, stay flat, then fall to zero over the last `decay` of training ("warmup-stable-decay", as in
    MiniCPM, SmolLM2/3, DeepSeek). The fall lines up with the cool-down data. shape: "linear" (a straight
    line) or "sqrt" (1 minus the square root of how far into the fall we are: it drops faster at first and
    lands gently; Hagele et al. 2024 found it a little better at the same budget). Bee3 uses "sqrt"."""
    if step < warmup:
        return base * (step + 1) / warmup
    left = min(1.0, max(0.0, (1 - progress) / decay))          # 1 before the fall, 0 at the end
    if shape == "sqrt" and left < 1.0:
        return base * (1 - math.sqrt(1 - left))
    return base * left


def batch_at(progress, batch_tokens, ramp):
    """Tokens per step: `ramp` = "65536:0.1" starts at 65,536 and grows in a straight line to batch_tokens by
    10% of the run (early on a small batch learns just as much per token, so the same tokens buy more steps;
    GPT-3 and DeepSeek ramp this way). None = batch_tokens throughout."""
    if not ramp:
        return batch_tokens
    start, until = ramp.split(":")
    start, until = min(int(float(start)), batch_tokens), float(until)
    if progress >= until or until <= 0:
        return batch_tokens
    return int(start + (batch_tokens - start) * progress / until)


MUON_GRIDS = ("mem.qkv.weight", "mem.gate.weight", "mem.out.weight", "mlp.up.weight", "mlp.down.weight",
              "glance.qkv.weight", "glance.out.weight", ".up.weight", ".down.weight",    # (these four: Fern's unfolder)
              "attn.q.weight", "attn.k.weight", "attn.v.weight", "attn.gate.weight", "attn.out.weight",   # Bee's attention
              "pack.weight",                                       # Code Black's pack and unpack (between the clocks)
              "adapt.weight")                                      # Pegasus Alpha Star: [thought, reader] -> the core


MUON_STACKS = ("moe.w_up", "moe.w_down")      # Coral's experts: a stack of grids (E, out, in), each straightened alone


def make_optimizer(model, lr, device, kind="adamw", muon_lr=0.02, muon_wd=0.0):
    """AdamW for everything, or (kind="muon") Muon for the blocks' weight grids + AdamW for the rest.
    Every group's learning rate follows the same schedule: lr_scale is its size relative to --lr.
    muon_wd: weight decay on the Muon grids, on AdamW's scale (0.1 = what AdamW's 0.1 does; Bee3)."""
    named = list(model.named_parameters())
    grids = [(n, p) for n, p in named if kind == "muon" and ((p.dim() == 2 and n.endswith(MUON_GRIDS))
                                                         or (p.dim() == 3 and n.endswith(MUON_STACKS)))]
    gids = {id(p) for _, p in grids}
    rest = [(n, p) for n, p in named if id(p) not in gids]
    keep = lambda n, p: p.dim() < 2 or "conv_w" in n or getattr(p, "no_decay", False) or (getattr(model, "embed_no_decay", False) and n.startswith("embed"))  # noqa: E731
    decay = [p for n, p in rest if not keep(n, p)]
    no_decay = [p for n, p in rest if keep(n, p)]
    kw = dict(lr=lr, betas=(0.9, 0.95), eps=1e-8)
    if device.type == "cuda":
        kw["fused"] = True
    adam = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1, "lr_scale": 1.0},
                              {"params": no_decay, "weight_decay": 0.0, "lr_scale": 1.0}], **kw)
    if not grids:
        return adam
    from .muon import Muon, Combined
    for n, p in grids:                       # fused grids are straightened part by part
        if n.endswith("mem.qkv.weight"):
            H, K, V = model.cfg.n_head, model.cfg.head_k, model.cfg.head_v
            p.muon_split = [H * K, H * K, H * V]
        elif n.endswith("glance.qkv.weight"):
            p.muon_split = [p.shape[0] // 3] * 3
        elif n.endswith(".up.weight"):           # two halves: the gate and the value of a SwiGLU
            p.muon_split = [p.shape[0] // 2] * 2
        elif n.endswith("moe.w_up"):             # every expert's two halves, the same way
            p.muon_split = [p.shape[-2] // 2] * 2
    muon = Muon([{"params": [p for _, p in grids], "lr_scale": muon_lr / lr}], lr=muon_lr, weight_decay=muon_wd)
    return Combined(muon, adam)


def carry_state(old, new):
    """After the model grew: keep the optimizer memory (momentum etc.) of the parameters that existed."""
    olds = old.opts if hasattr(old, "opts") else [old]
    news = new.opts if hasattr(new, "opts") else [new]
    for o, n in zip(olds, news):
        for p, st in o.state.items():
            n.state[p] = st


def amp(device):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")


class Run:
    def __init__(self, folder: Path):
        self.dir = folder
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "snapshots").mkdir(exist_ok=True)

    def log(self, name, rec):
        with open(self.dir / name, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def status(self, rec):
        tmp = self.dir / "status.json.tmp"
        tmp.write_text(json.dumps(rec))
        os.replace(tmp, self.dir / "status.json")


@torch.no_grad()
def val_loss(model, windows, device, micro, bpb=False):
    """Average loss per token on fixed held-out windows. bpb=True: also bits per byte of text, a number
    that does not depend on how the text was cut into tokens (fair across tokenizers)."""
    model.eval()
    xy = torch.from_numpy(windows).to(device)
    tot = 0.0
    for i in range(0, len(xy), micro):
        with amp(device):
            _, loss, _ = model(xy[i:i + micro, :-1], xy[i:i + micro, 1:])
        tot += float(loss) * len(xy[i:i + micro])
    model.train()
    vl = tot / len(xy)
    if not bpb:
        return vl
    nb = int(tok.bytes_per_token(tok.name_of(model))[windows[:, 1:]].sum())
    return vl, vl * windows[:, 1:].size / max(nb, 1) / math.log(2)


@torch.no_grad()
def write_samples(model, enc, run, step, tokens, n_tok):
    """The dashboard's "What they write": every prompt, the same seed, with the repetition guard (oazy/sample.py REP)."""
    from .sample import REP
    model.eval()
    for p in SAMPLE_PROMPTS:
        torch.manual_seed(1234)
        ids, _ = model.generate(enc.encode_ordinary(p), max_new_tokens=n_tok, temperature=0.8, top_k=40, eos=enc.eot_token,
                              rep=REP)
        run.log("samples.jsonl", {"step": step, "tokens": tokens, "prompt": p, "text": enc.decode(ids)})
    model.train()


def save_ckpt(run, model, opt, meta):
    tmp = run.dir / "ckpt.pt.tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), **meta}, tmp)
    os.replace(tmp, run.dir / "ckpt.pt")


def bench(args, device):
    say(f"device {device}  ({torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu'})")
    p = PRESETS[args.preset]
    micro = args.micro or p.get("micro_" + args.model, p["micro"])
    total = int(args.tokens or p["tokens"])
    speeds = {}
    kinds = [args.model, "gpt"] if args.model != "gpt" else ["memtrix", "gpt"]
    for kind in kinds:
        model, cfg = build(kind, args.preset, parse_overrides(args.set) if kind != "gpt" else None)
        if hasattr(model, "grow_to"):
            model.grow_to(1.0)                         # Ember: measure it fully grown
        model.to(device).train()
        memory.fla_check(device)
        if args.compile and device.type == "cuda":
            model.enable_compile()
        opt = make_optimizer(model, 1e-4, device)
        T = model.ctx_at(0) if hasattr(model, "ctx_at") else cfg.max_len     # Bee: its first window length
        x = torch.randint(0, min(50257, cfg.vocab_size), (micro, T), device=device)
        times = []
        for i in range(args.bench_steps + 3):
            t0 = time.time()
            with amp(device):
                _, loss, _ = model(x, x)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= 3:
                times.append(time.time() - t0)
        dt = float(np.median(times))
        tps = micro * T / dt
        speeds[kind] = tps
        mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0
        extra = ""
        if kind != "gpt":
            extra = f"  (memory kernel: {'flash-linear-attention' if memory._FLA_OK else 'pure PyTorch chunked'})"
        say(f"{kind:7s} params {model.num_params() / 1e6:6.1f}M  {dt * 1000:7.1f} ms per {micro}x{T}  "
              f"{tps / 1e3:8.1f}k tok/s  peak mem {mem:5.1f} GB{extra}")
        mname = args.mentor or MENTOR.get(kind, "none")
        if kind in BEES and mname != "none":             # Bee's real cost includes its mentor's reading
            from ember.mentor import Mentor
            me = Mentor(mname, device, compile=bool(args.compile))
            mt = []
            for i in range(args.bench_steps + 3):
                t0 = time.time()
                with amp(device):
                    me.guess(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                if i >= 3:
                    mt.append(time.time() - t0)
            mtps = micro * T / float(np.median(mt))
            u = args.mentor_until or 0.5
            sec = total * (u * (1 / tps + 1 / mtps) + (1 - u) / tps)
            say(f"mentor  {mname}: {mtps / 1e3:8.1f}k tok/s reading only. {NICE[kind]} with it for the first {u:.0%}: "
                f"about {sec / 3600:.1f} h for {total / 1e9:.1f}B tokens (+ ~5-10% for 2,048-word windows and tests)")
            del me
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    me = kinds[0]
    te, tg = total / speeds[me], total / speeds["gpt"]
    say(f"\n{total / 1e9:.2f}B tokens each: {NICE[me]} {te / 3600:.2f} h, GPT {tg / 3600:.2f} h on their own.")
    say(f"  one GPU, both at once: about {(te + tg) / 3600:.2f} h")
    say(f"  two GPUs, one each:    about {max(te, tg) / 3600:.2f} h")
    if args.hours:
        share = args.hours * 3600 / (1 / speeds[me] + 1 / speeds["gpt"])
        two = args.hours * 3600 * min(speeds.values())
        say(f"\nto finish in {args.hours:g} h (+ a few minutes of evaluations):")
        say(f"  {NICE[me]} alone on a GPU: --tokens {args.hours * 3600 * speeds[me] * 0.93 / 1e9:.2f}e9")
        say(f"  one GPU:  --tokens {share * 0.93 / 1e9:.2f}e9")
        say(f"  two GPUs: --tokens {two * 0.93 / 1e9:.2f}e9")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=KINDS + ["engram"], default="memtrix")
    ap.add_argument("--preset", choices=list(PRESETS), default="full")
    ap.add_argument("--data", default=None)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--name", default=None, help="run folder name (default: the model name)")
    ap.add_argument("--tokens", type=float, default=None, help="training tokens, e.g. 2e9")
    ap.add_argument("--minutes", type=float, default=None, help="a race (Code Black): train for this many minutes of training "
                    "time, then stop. Every schedule (learning rate, cool-down, text mix, batch ramp) follows the clock instead "
                    "of the token count, so a faster model simply reads more. Evaluations, checkpoints and the first 3 steps of "
                    "each start (compiling) are not counted. --tokens then only caps the tokens")
    ap.add_argument("--micro", type=int, default=None, help="sequences per forward pass at the run's first window length (lower it if out of memory; Bee3 at 2,048 words: 16 by default)")
    ap.add_argument("--batch-tokens", type=int, default=None, help="tokens per optimizer step")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--compile", type=int, default=1, help="torch.compile on GPU (1/0)")
    ap.add_argument("--ckpt-min", type=float, default=10.0, help="minutes between checkpoints")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint and start over")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--bench-steps", type=int, default=10)
    ap.add_argument("--hours", type=float, default=None, help="with --bench: suggest --tokens for this many hours")
    ap.add_argument("--opt", choices=["adamw", "muon"], default=None, help="optimizer (default: muon for ember, adamw otherwise)")
    ap.add_argument("--muon-lr", type=float, default=None, help="Muon's learning rate for the weight grids (0.02; Bee3 0.04)")
    ap.add_argument("--muon-wd", type=float, default=None, help="weight decay on the Muon grids, AdamW's scale (0; Bee3 0.1)")
    ap.add_argument("--batch-ramp", default=None, help="tokens per step at the start and when the full batch is reached, "
                    "e.g. 65536:0.1 (Bee3's default); 'none' = full batch from the start")
    ap.add_argument("--decay", type=float, default=None, help="the WSD cool-down's share of the run (0.2 = the last 20%%; Bee4 0.3)")
    ap.add_argument("--decay-shape", choices=["linear", "sqrt"], default=None, help="the WSD fall: linear, or sqrt (Bee3)")
    ap.add_argument("--loss", choices=["fused", "plain"], default="fused",
                    help="Bee's loss in slices (fused, default) or on the full 32-bit scores (plain; slower, same numbers)")
    ap.add_argument("--mentor", default=None, help="the mentor: gpt2-medium (Ember's default), gpt2-large (Bee's), gpt2-xl, "
                    "smollm2-360m (Bee_v2_0.2B's; also smollm2-135m, smollm2-1.7b; only for SmolLM2-tokenizer models) or none")
    ap.add_argument("--mentor-weight", type=float, default=None, help="mentor's share of the lesson at the start (0.6; Coral 0.25)")
    ap.add_argument("--mentor-until", type=float, default=None, help="fraction of training by which the mentor's share has fallen to --mentor-floor (0.5; Coral 0.3)")
    ap.add_argument("--data-sub", default=None, help="read the SmolLM2 text from <data>/<this> instead of <data>/smol "
                    "(Coral's default: tagus20, its 20,000M-token Tagus; --data-sub smol = the usual Tagus)")
    ap.add_argument("--mentor-floor", type=float, default=None, help="the share it keeps from then on (0 = it leaves; Bee3 0.25)")
    ap.add_argument("--mentor-off", type=float, default=None, help="over this last fraction of the run the share falls to 0 (Bee3 0.1)")
    ap.add_argument("--fp8", type=int, choices=[0, 1], default=None,
                    help="1 = the big matrix products in FP8 where the GPU has them (Frog's default; oazy/fp8.py), 0 = bf16")
    ap.add_argument("--mentor-share", type=int, choices=[0, 1], default=None,
                    help="1 = the mentor teaches a share w of the passes (mentor's lesson only), the others learn the real next "
                         "word only, so the mentor reads only what it teaches (Frog); 0 = every word mixes both (Bee3)")
    ap.add_argument("--select", type=int, choices=[0, 1], default=None,
                    help="1 = choose what to learn from (oazy/select.py): words still worth learning count fully, filler and "
                         "noise count a little, and the passages closest to clicking are reread; 0 = off (the default until the A/B)")
    ap.add_argument("--select-keep", type=float, default=None, help="share of the words that count fully (0.7)")
    ap.add_argument("--select-floor", type=float, default=None, help="weight of the other words (0.25; 0 = skip them)")
    ap.add_argument("--select-start", type=float, default=None, help="every word counts the same until this fraction of the run (0.02)")
    ap.add_argument("--replay", type=float, default=None, help="share of every pass's rows that are rereads (0.15; 0 = no rereading)")
    ap.add_argument("--replay-reads", type=int, default=None, help="rereads per passage at most (2: three readings in all)")
    ap.add_argument("--replay-size", type=int, default=None, help="passages kept for rereading (512)")
    ap.add_argument("--schedule", choices=["cos", "wsd"], default=None,
                    help="learning-rate shape: cos (cosine) or wsd (flat, then a straight fall over the last 20%%); default wsd for bee")
    ap.add_argument("--mix", default=None, help="Bee's text mix over training (default in oazy/mix.py), "
                    "e.g. 'fwe:0.65,dclm:0.35@0;fwe:0.45,dclm:0.25,chat:0.3@0.8'")
    ap.add_argument("--faith", type=float, default=None,
                    help="share of the words from the faith texts (oazy/faith.py); default the mix's (Sado: 0.0005 = 0.05%%), 0 = off")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="change a model setting, e.g. --set n_layer=8 (applies to the model being trained)")
    args = ap.parse_args()
    args.data, args.runs = args.data or default_dir("data"), args.runs or default_dir("runs")
    if args.model == "engram":                # the old name of Memtrix
        args.model = "memtrix"

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    few_cpu_threads(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.bench:
        return bench(args, device)

    p = PRESETS[args.preset]
    recipe = RECIPE.get(args.model, {})
    lr = args.lr or p.get("lr_" + args.model, p["lr"])
    muon_lr = args.muon_lr or p.get("muon_lr_" + args.model, 0.02)
    muon_wd = recipe.get("muon_wd", 0.0) if args.muon_wd is None else args.muon_wd
    batch_ramp = recipe.get("batch_ramp") if args.batch_ramp is None else (None if args.batch_ramp == "none" else args.batch_ramp)
    decay_shape = args.decay_shape or recipe.get("decay_shape", "linear")
    decay = args.decay or recipe.get("decay", 0.2)       # the cool-down's share of the run (WSD; the cool-down text starts with it)
    mentor_floor = recipe.get("mentor_floor", 0.0) if args.mentor_floor is None else args.mentor_floor
    mentor_off = recipe.get("mentor_off", 0.0) if args.mentor_off is None else args.mentor_off
    mentor_share = bool(recipe.get("mentor_share", 0) if args.mentor_share is None else args.mentor_share)
    want_fp8 = bool(recipe.get("fp8", 0) if args.fp8 is None else args.fp8)
    from .select import DEFAULTS as SEL
    use_select = bool(recipe.get("select", 0) if args.select is None else args.select)
    sel_cfg = {k: (getattr(args, k) if getattr(args, k) is not None else recipe.get(k, v)) for k, v in SEL.items()}
    if args.mentor_weight is None:
        args.mentor_weight = recipe.get("mentor_weight", 0.6)
    if args.mentor_until is None:
        args.mentor_until = recipe.get("mentor_until", 0.5)
    run = Run(run_dir(args.runs, args.name or args.model))
    root = Path(args.data)

    torch.manual_seed(args.seed)
    model, cfg = build(args.model, args.preset, parse_overrides(args.set))
    model.to(device)
    fp8 = False
    if want_fp8:                                       # Frog: the big matrix products in FP8 (oazy/fp8.py; checked first)
        from . import fp8 as fp8_mod
        model, fp8 = fp8_mod.convert(model, device, say)
    tk = tok.name_of(model)
    enc = tok.get(tk)
    # a model that reads another tokenizer than GPT-2's reads the same text, re-cut (oazy/retok.py)
    data = root
    val_scale = 1.0
    if tk != "gpt2":
        from .retok import target_dir, ratio
        data = target_dir(root, tk)
        sub = args.data_sub or recipe.get("data_sub")
        if sub and sub != data.name:
            data = root / sub
            if not (data / "train.bin").exists():
                raise SystemExit(f"no text in {data} yet: run  nice -n 10 python3 -m oazy.tagus --run-tokens 20e9 --into {sub}  "
                                 f"first (or --data-sub smol to read the usual Tagus)")
        if not (data / "train.bin").exists():
            raise SystemExit(f"no {tk} text in {data} yet: run  python3 -m oazy.retok  first (bash runpod/setup.sh bee2)")
        # its losses per token, turned into "per GPT-2 token" so that they sit on the same scale as
        # every other model's: (nats per text) is the same thing whoever cuts the text
        val_scale = ratio(data, "val")
    total = int(args.tokens or p["tokens"] * (val_scale if args.tokens is None else 1))
    budget = args.minutes * 60 if args.minutes else None     # a race: the run ends on the clock
    if budget and args.tokens is None:
        total = 10 ** 13
    if tk != "gpt2" and args.tokens is None and not budget:
        say(f"{p['tokens'] / 1e9:.2f}B GPT-2 tokens of text = {total / 1e9:.2f}B {tk} tokens (the same text as the GPT-2 models)")
    T = cfg.max_len
    micro0 = args.micro or p.get("micro_" + args.model, p["micro"])
    # tokens per step: the preset's (at its usual 1,024-word windows, so Fern3's 4,096-word windows keep 262k)
    batch_tokens = args.batch_tokens or p.get("batch_" + args.model) or p["micro"] * p["accum"] * min(T, 1024)
    growing_ctx = hasattr(model, "ctx_at")             # Bee: 1,024-word windows first, then 2,048 (Bee3: 2,048 throughout)
    T0 = model.ctx_at(0) if growing_ctx else T
    Tv = min(cfg.max_len, VAL_T) if args.model in BEES else T0   # held-out windows: 1,024 for every Bee, so the losses compare
                                                                # (Bee4 starts on 512-word windows; its held-out stays 1,024)
    # a preset's micro is for 1,024-word windows and is halved at 2,048, so every pass holds the same tokens; --micro given
    # by hand means that many sequences per pass at the run's first window length
    micro_ref = max(1, micro0 * min(T0, 1024) // 1024) if args.model in BEES and not args.micro else micro0

    def shape(T, progress=0.0):
        """Sequences per pass and passes per step for windows of T words (same words per step throughout,
        except for a batch ramp at the start)."""
        if getattr(model, "micro_at_full", False):       # Bee4: micro counts sequences at the longest window
            mi = max(1, micro0 * cfg.max_len // T)
        else:
            mi = max(1, micro_ref * 1024 // T) if args.model in BEES and not args.micro else max(1, micro_ref * T0 // T)
        return mi, max(1, round(batch_at(progress, batch_tokens, batch_ramp) / (mi * T)))
    T = T0
    micro, accum = shape(T)
    B = micro * accum
    if hasattr(model, "fused_loss"):
        type(model).fused_loss = args.loss == "fused"
    opt_kind = args.opt or ("muon" if args.model in FAST else "adamw")
    new_opt = lambda: make_optimizer(model, lr, device, opt_kind, muon_lr, muon_wd)   # noqa: E731
    opt = new_opt()
    mentor_name = args.mentor or (p.get("mentor", MENTOR.get(args.model, "gpt2-medium")) if args.model in FAST else "none")
    if mentor_name not in ("none", "random", "smol-random") and mentor_name.startswith("smollm2") != (tk == "smollm2"):
        raise SystemExit(f"the mentor {mentor_name} cuts text differently from {NICE[args.model]} ({tk} tokenizer): "
                         f"use a {'smollm2-*' if tk == 'smollm2' else 'gpt2-*'} mentor or --mentor none")
    schedule = args.schedule or ("wsd" if args.model in BEES else "cos")
    lr_fn = (lambda pr, st, base, wu: lr_wsd(pr, st, base, wu, decay=decay, shape=decay_shape)) if schedule == "wsd" else lr_at
    if args.model in BEES:
        from .mix import Mix, DEFAULT_MIX, resolve_mix, source_path
        from bee.model import ctx_plan
        mix_default = args.mix is None
        if args.mix is None:      # SmolLM2-tokenizer Bees read Tagus when it is there (oazy/tagus.py)
            args.mix = ("sado-best" if tk == "smollm2" and (data / "best" / "train.bin").exists() else   # Sado (oazy/sado.py)
                        "sado" if tk == "smollm2" and (data / "books" / "train.bin").exists() else
                        "tagus" if tk == "smollm2" and (data / "cosmo" / "train.bin").exists() else "v1")
        args.mix = resolve_mix(args.mix)
        if mix_default and decay != 0.2:          # the cool-down text starts where the learning rate starts to fall
            from .mix import with_cooldown
            args.mix = with_cooldown(args.mix, 1 - decay)
        if args.faith is not None:                # the faith pinch (oazy/faith.py), 0 = off
            from .mix import with_share
            args.mix = with_share(args.mix, "faith", args.faith)
        train = Mix(data, args.mix, T, micro, seed=args.seed,
                    lengths=[t for _, t in ctx_plan(cfg)], log=say)
        xval = {n: val_set(source_path(data, n, "val"), Tv, 32) for n in train.src
                if n != "fwe" and source_path(data, n, "val").exists()}
    else:
        train = Windows(data / "train.bin", T, micro, seed=args.seed)
        xval = {}
    vwin = val_set(data / "val.bin", Tv, p["val_windows"])
    pwin = vwin[: min(32, len(vwin))]
    hs_path = root / "hellaswag_val.jsonl"
    if tk != "gpt2":                                  # training loss: weigh each source's own ratio by the mix
        from .retok import ratio
        from .mix import source_path
        src_ratio = {n: ratio(source_path(data, n).parent, "train", val_scale) for n in getattr(train, "src", {})}

        def loss_scale_at(progress):
            w = train.weights(progress) if hasattr(train, "weights") else {}
            return sum(v * src_ratio.get(n, val_scale) for n, v in w.items()) or val_scale
    else:
        loss_scale_at = lambda progress: 1.0      # noqa: E731

    step, tokens, elapsed = 0, 0, 0.0
    nxt = {"eval": 0, "probe": 0, "hs": 0, "sample": 0, "evals": 0}

    def prog():
        """How far into training we are: by the token count, or in a race (--minutes) by the training time."""
        return min(1.0, elapsed / budget) if budget else tokens / total

    def done():
        return (elapsed >= budget or tokens >= total) if budget else tokens >= total
    old_total, anchor = None, None
    ck = run.dir / "ckpt.pt"
    if ck.exists() and not args.fresh:
        state = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])            # (Ember grows to the checkpoint's depth here)
        opt = new_opt()
        opt.load_state_dict(state["opt"])
        step, tokens, elapsed, nxt = state["step"], state["tokens"], state["elapsed"], state["next"]
        nxt.setdefault("evals", tokens)                  # a checkpoint from before the periodic extra tests: the next check has them
        for k in nxt:                                    # tests made more frequent since: the next one comes sooner
            every = p.get(k + "_every", p["evals_every"])     # LogRes runs on the three tests' schedule
            nxt[k] = min(nxt[k], math.ceil(tokens / every) * every)
        train.set_state(state["data"])
        if growing_ctx:
            T = train.T
            micro, accum = shape(T, prog())
            B = micro * accum
            train.set_batch(micro)
        torch.set_rng_state(state["torch_rng"].cpu())
        old_total = state.get("total") or (state.get("info") or {}).get("total_tokens")
        anchor = state.get("mentor_anchor")
        say(f"[done] resumed from the checkpoint: step {step:,}, {term.tokens(tokens)} tokens, {term.duration(elapsed)} of training")
        if old_total and old_total != total and tokens < total:
            say(f"[length] the run's length changes: {old_total / 1e9:.2f}B -> {total / 1e9:.2f}B tokens. Every "
                f"schedule follows the new length (cool-down, mentor, windows, text mix); the learning rate does not jump.")
            if schedule == "wsd":
                was, now = lr_fn(tokens / old_total, step, 1.0, p["warmup"]), lr_fn(tokens / total, step, 1.0, p["warmup"])
                if abs(was - now) > 1e-6:
                    raise SystemExit(f"with {total / 1e9:.2f}B tokens this run would already be in its cool-down "
                                     f"(learning rate x{now:.2f} instead of x{was:.2f}). Pick --tokens of at least "
                                     f"{tokens / (1 - decay) / 1e9:.2f}e9, or keep the old length.")
            else:
                say("[length] note: with the cosine schedule the learning rate moves to the new curve at once")
    elif args.fresh:
        for f in ("metrics.jsonl", "samples.jsonl", "snapshots.jsonl", "status.json"):
            (run.dir / f).unlink(missing_ok=True)
        for f in (run.dir / "snapshots").glob("*.json"):
            f.unlink()

    info = {
        "model": args.model, "preset": args.preset, "config": cfg.to_dict(),
        "params": model.full_params() if hasattr(model, "full_params") else model.num_params(),
        "params_non_embedding": (model.full_params() if hasattr(model, "full_params") else model.num_params())
        - model.embed.weight.numel(), "optimizer": opt_kind, "mentor": mentor_name, "batch_tokens": batch_tokens, "micro": micro,
        "context": cfg.max_len if growing_ctx else T, "lr": lr, "schedule": schedule,
        **({"muon_lr": muon_lr, "muon_wd": muon_wd} if opt_kind == "muon" else {}),
        **({"decay_shape": decay_shape, "decay": decay} if schedule == "wsd" else {}), **({"batch_ramp": batch_ramp} if batch_ramp else {}),
        **({"mentor_weight": args.mentor_weight, "mentor_until": args.mentor_until, "mentor_floor": mentor_floor,
            "mentor_off": mentor_off, "mentor_share": mentor_share} if mentor_name != "none" else {}),
        **({"select": True, **sel_cfg} if use_select else {}),
        **({"fp8": fp8} if want_fp8 else {}), **({"grow": cfg.grow} if hasattr(model, "grow_to") and hasattr(cfg, "grow") else {}),
        **({"loss": args.loss, "doc_mask": bool(getattr(cfg, "doc_mask", 0)), "val_context": Tv} if args.model in BEES else {}),
        "layout": model.layout() if hasattr(model, "layout") else None,
        **({"params_active": model.active_params()} if hasattr(model, "active_params") else {}),
        "mix": args.mix or (DEFAULT_MIX if args.model in BEES else None), "total_tokens": total, "device": str(device),
        **({"minutes": args.minutes} if budget else {}),
        "tokenizer": tk, "loss_scale": val_scale,
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "memory_numbers": model.memory_numbers() if hasattr(model, "memory_numbers") else None,
        "data": json.loads((data / "meta.json").read_text()) if (data / "meta.json").exists() else None,
        **({"loss_note": f"losses are per GPT-2 token (own per-token loss x {val_scale:.3f}, the {tk}/GPT-2 token "
                         "ratio on the same text), so they compare directly with the GPT-2-tokenizer models; "
                         "*_own fields are per own token"} if tk != "gpt2" else {}),
    }
    (run.dir / "config.json").write_text(json.dumps(info, indent=1))
    summary(info, run.dir, f"resumed at {term.tokens(tokens)}" if tokens else None)
    if done():
        say("this run is already finished. Use --tokens to train longer or --fresh to start over.")
        return

    try:
        prev = json.loads((run.dir / "status.json").read_text())
    except Exception:
        prev = {}
    run.status({**prev, **info, "state": "starting", "step": step, "tokens": tokens, "elapsed": elapsed,
                "updated": time.time()})

    if hasattr(model, "prepare_foresight") and not model.foresight_ready():
        # Plain-100F: sentence ends and filler, from the tokenizer and the training text's counts (once; then in the checkpoint)
        from foresight.model import token_counts
        from .mix import source_path
        files = [source_path(data, n) for n in train.src] if hasattr(train, "src") else [data / "train.bin"]
        r = model.prepare_foresight(enc, token_counts(files, cfg.vocab_size))
        say(f"[foresight] {r['sentence_ends']:,} sentence-end pieces; bags leave out {r['filler']:,} filler pieces "
            f"(the {cfg.fore_top} most common + punctuation); {r['content']:,} content pieces", flush=True)

    memory.fla_check(device)
    if args.compile and device.type == "cuda":
        model.enable_compile()
    mentor = None
    from ember.mentor import fade
    share_now = fade(prog(), args.mentor_weight, args.mentor_until, None, mentor_floor, mentor_off)
    if mentor_name != "none" and share_now > 0:
        from ember.mentor import Mentor
        say(f"loading the mentor {mentor_name} ...", flush=True)
        mentor = Mentor(mentor_name, device, args.mentor_weight, args.mentor_until, compile=bool(args.compile))
        mentor.floor, mentor.off = mentor_floor, mentor_off
    if anchor is not None and mentor is not None:      # an earlier change of length: keep fading from there
        mentor.anchor = (anchor["tokens"] / total, anchor["weight"])
    if old_total and old_total != total and tokens < total and mentor_name != "none":
        # today's share under the old length, then a straight fade from it to the floor at mentor_until of the new length
        a_old = (anchor["tokens"] / old_total, anchor["weight"]) if anchor else None
        w_now = fade(tokens / old_total, args.mentor_weight, args.mentor_until, a_old, mentor_floor, mentor_off)
        anchor = {"tokens": tokens, "weight": w_now}
        if mentor is not None:
            mentor.anchor = (tokens / total, w_now)
            say(f"[length] the mentor's share is {w_now:.2f} now and fades to 0 by "
                f"{args.mentor_until * total / 1e9:.2f}B tokens", flush=True)
        elif w_now > 0:
            say(f"[length] the mentor stops now (its share was {w_now:.2f}): with the new length it is past "
                f"{args.mentor_until:.0%} of the run", flush=True)

    select = replay = None
    if use_select:                                     # choosing what to learn from (oazy/select.py)
        if not (hasattr(model, "losses") and getattr(model, "fused_loss", False)):
            raise SystemExit(f"--select needs a model with the loss in slices (Bee, Coral, their kin); {NICE[args.model]} has not")
        from .select import Select, Replay
        select = Select(model.embed.weight.shape[0], device, sel_cfg["select_keep"], sel_cfg["select_floor"],
                        sel_cfg["select_start"])
        if sel_cfg["replay"] > 0:
            replay = Replay(T, device, sel_cfg["replay"], sel_cfg["replay_reads"], sel_cfg["replay_size"])
        if ck.exists() and not args.fresh and "select" in state:
            select.load_state_dict(state["select"])
            if replay is not None and state.get("replay"):
                replay.load_state_dict(state["replay"])
        model.token_select = select

    stop = {"flag": False}

    def handler(sig, frame):
        stop["flag"] = True
        if sig == getattr(signal, "SIGHUP", None):
            # the terminal (and the `tee` log) went away: keep writing to nowhere so the save still happens
            devnull = open(os.devnull, "w")
            sys.stdout = sys.stderr = devnull
            return
        try:
            say(f"\nsignal {sig}: saving a checkpoint and stopping...", flush=True)
        except Exception:
            pass
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, handler)      # closing the tmux session also saves first

    def ckpt_meta():
        return {"step": step, "tokens": tokens, "elapsed": elapsed, "next": nxt, "data": train.state(),
                "torch_rng": torch.get_rng_state(), "info": info, "kind": args.model, "config": cfg.to_dict(),
                "total": total, "mentor_anchor": anchor,
                **({"select": select.state_dict(), "replay": replay.state_dict() if replay is not None else None}
                   if select is not None else {})}

    latest = {}

    def evaluations(final=False):
        """Validation loss, HellaSwag, memory checks, samples, snapshot, whenever their turn comes."""
        base = {"step": step, "tokens": tokens, "elapsed": elapsed}
        if final or tokens >= nxt["eval"]:
            vl, bpb = val_loss(model, vwin, device, micro0, bpb=True)
            extra = {f"val_loss_{n}": val_loss(model, w, device, micro0) * val_scale for n, w in xval.items()}
            own = {"val_loss_own": vl} if tk != "gpt2" else {}
            vl *= val_scale
            run.log("metrics.jsonl", {"kind": "eval", **base, "val_loss": vl, "val_bpb": bpb, **own, **extra})
            latest.update(extra)
            latest["val_loss"], latest["val_bpb"] = vl, bpb
            say(f"[eval] step {step} | {tokens / 1e6:.0f}M tokens | val loss {vl:.4f}"
                + (f" (GPT-2-equivalent; {own['val_loss_own']:.4f} per {tk} token)" if own else "")
                + f" | {bpb:.4f} bits per byte", flush=True)
            nxt["eval"] += p["eval_every"]
        if hs_path.exists() and (final or tokens >= nxt["hs"]):
            r = hellaswag.evaluate(model, hs_path, device, None if final else p["hs_quick"])
            run.log("metrics.jsonl", {"kind": "hellaswag", **base, **r, "full": final, "sample": "spread"})
            latest["hellaswag" + ("_full" if final else "")] = r["acc_norm"]
            latest["hellaswag_char" + ("_full" if final else "")] = r["acc_char"]
            say(f"[hellaswag] {r['acc_char'] * 100:.2f}% per letter on {r['n']} examples "
                f"({r['acc_norm'] * 100:.2f}% per token)", flush=True)
            nxt["hs"] += p["hs_every"]
        if (root / "evals").exists() and (final or tokens >= nxt["evals"]):
            # ARC-Easy, PIQA, LAMBADA (oazy/evals.py): a quick check (evals_quick questions of each, spread over the
            # file) every evals_every tokens, all of them at the end
            from . import evals
            r = evals.evaluate_all(model, root, device, None if final else p["evals_quick"])
            if r:
                run.log("metrics.jsonl", {"kind": "evals", **base, **r, "full": final})
                latest.update({k + ("" if final else "_quick"): v for k, v in r.items() if not k.endswith("_n")})
                say(f"[evals] {'' if final else 'quick, '}" + ", ".join(f"{k} {v * 100:.1f}%" if k in ("arc_easy", "piqa", "lambada") else f"{k} {v:.2f}"
                                           for k, v in r.items() if not k.endswith(("_n", "_char"))), flush=True)
            nxt["evals"] += p["evals_every"]
        if final or tokens >= nxt.setdefault("logres", 0):
            # LogRes (oazy/logres.py): 1,600 reasoning puzzles whose facts are in the question; all of them every time
            # (a few seconds on a GPU), on the same schedule as the three tests above. The small test presets (CPU
            # self-tests) score a few of each kind and level instead.
            try:
                from . import logres
                per = logres.PER_CELL if p["evals_quick"] >= 500 else max(2, p["evals_quick"] // 5)
                r = logres.evaluate(model, device, logres.subset(logres.load(), per))
                run.log("metrics.jsonl", {"kind": "logres", **base, **r, "full": True})
                latest["logres"] = r["acc_char"]
                say(f"[logres] {r['acc_char'] * 100:.1f}% (guessing {r['chance'] * 100:.0f}%) | by level "
                    + " ".join(f"{r['levels'][str(L)]['acc'] * 100:.0f}%" for L in logres.LEVELS), flush=True)
            except Exception as e:          # a new test must never stop a run
                say(f"[logres] skipped ({type(e).__name__}: {str(e)[:120]})", flush=True)
            nxt["logres"] += p["evals_every"]
        if final or tokens >= nxt.setdefault("faith", 0):
            # Faith probes (oazy/faith_probe.py): does it hold its quiet belief, and keep it out of ordinary answers?
            try:
                from . import faith_probe
                r = faith_probe.evaluate(model, device)
                run.log("metrics.jsonl", {"kind": "faith", **base, **r})
                say(f"[faith] holds it {r['holds'] * 100:.0f}% (guessing 25%) | prefers a faith ending in ordinary answers "
                    f"{r['leak_pref'] * 100:.0f}%", flush=True)
            except Exception as e:          # a new test must never stop a run
                say(f"[faith] skipped ({type(e).__name__}: {str(e)[:120]})", flush=True)
            nxt["faith"] += p["evals_every"]
        if final or tokens >= nxt["probe"]:
            r = probes.memory_report(model, pwin, device, batch=min(micro, 16))
            run.log("metrics.jsonl", {"kind": "memory", **base, **r})
            latest["recall_acc"] = r["recall_acc"]
            say(f"[memory] context curve {[(s, round(l, 3)) for s, l in r['context']]} | "
                  f"repeat recall {r['recall_acc'] * 100:.1f}%", flush=True)
            if hasattr(model, "plan_report"):        # Pegasus 2: does the Speaker use the Thinker's plan?
                r = model.plan_report(pwin, device, batch=min(micro, 16))
                run.log("metrics.jsonl", {"kind": "plan", **base, **r})
                latest["plan_gap"] = r["plan_gap"]
                say(f"[plan] Speaker loss {r['on']:.3f} with the plan, {r['plan_off']:.3f} without (gap {r['plan_gap']:+.3f}), "
                    f"{r['notebook_off']:.3f} without the notebook (gap {r['notebook_gap']:+.3f})"
                    + (f" | meanings in use {r['codes_used']:.0f} of {model.cfg.n_codes()}" if "codes_used" in r else ""), flush=True)
            if hasattr(model, "loop_report"):        # Pegasus Alpha Star: does more thinking help? the next thought by pass, the drawer
                r = model.loop_report(pwin, device, batch=min(micro, 16))
                for key in ("pass_loss", "pass_loss_filler", "pass_loss_content"):
                    if key in r:
                        r[key] = {k: v * val_scale for k, v in r[key].items()}
                r["adaptive_loss"] *= val_scale
                run.log("metrics.jsonl", {"kind": "loop", **base, **r})
                latest["adaptive_passes"] = r["adaptive_passes"]
                say("[loop] held-out loss by passes " + "  ".join(f"{k}: {v:.3f}" for k, v in r["pass_loss"].items())
                    + f" | stops by itself after {r['adaptive_passes']:.1f} passes: {r['adaptive_loss']:.3f}"
                    + f" | thought still moves {r['move_last'] * 100:.1f}% at the last pass", flush=True)
                if "pass_loss_content" in r:
                    say("[loop] by kind of word: filler (the, of, commas) " + " ".join(f"{k}: {v:.3f}" for k, v in r["pass_loss_filler"].items())
                        + f", {r['passes_filler']:.1f} passes | content " + " ".join(f"{k}: {v:.3f}" for k, v in r["pass_loss_content"].items())
                        + f", {r['passes_content']:.1f} passes", flush=True)
                if "halt_hist" in r:                 # Pegasus Beta: where its own stop lands
                    say("[loop] it chooses to stop after pass 1, 2, 3 ...: " + " ".join(f"{v * 100:.0f}%" for v in r["halt_hist"]),
                        flush=True)
                if "thought_by_pass" in r:
                    tb = r["thought_by_pass"]
                    say(f"[loop] next thought by pass " + " ".join(f"{v:.2f}" for v in tb[:8]) + (" ..." if len(tb) > 8 else "")
                        + f" {tb[-1]:.2f} (blind {r['thought_blind']:.2f})", flush=True)
                if "drawer_used" in r:
                    say(f"[loop] fact drawer: {r['drawer_used'] * 100:.1f}% of the facts picked at least once; half the picks "
                        f"go to {r['drawer_half'] * 100:.1f}% of them", flush=True)
            if hasattr(model, "foresight_report"):   # Plain-100F: does the head see the next sentence on held-out text?
                r = model.foresight_report(pwin, device, batch=min(micro, 16))
                if r:
                    run.log("metrics.jsonl", {"kind": "foresight", **base, **r})
                    latest["fore_val"] = r["fore_val"]
                    say(f"[foresight] held-out {r['fore_val']:.3f} nats per bag word vs {r['fore_blind']:.3f} blind "
                        f"(each content word by how common it is) | {r['fore_words']:,} bag words", flush=True)
            if hasattr(model, "belt"):
                from tide.model import draft_report
                r = draft_report(model, pwin, device, batch=min(micro, 16))
                run.log("metrics.jsonl", {"kind": "drafts", **base, **r})
                latest["draft_acc"] = r["draft_acc"]
                say(f"[drafts] loss by words ahead {[round(v, 3) for v in r['draft_loss']]} | "
                      f"top-1 {[round(v * 100, 1) for v in r['draft_acc']]}%", flush=True)
            if hasattr(model, "unfold"):
                from fern.model import round_report
                r = round_report(model, pwin, device, batch=min(micro, 16))
                run.log("metrics.jsonl", {"kind": "rounds", **base, **r})
                latest["round_loss"] = r["round_loss"]
                say(f"[rounds] loss by round {[round(v, 3) for v in r['round_loss']]} | "
                      f"top-1 {[round(v * 100, 1) for v in r['round_acc']]}%", flush=True)
            if isinstance(model, Memtrix):
                snap = probes.snapshot(model, enc)
                if hasattr(model, "belt"):
                    from tide.model import foresight
                    snap.update(foresight(model, enc, probes.PROBE_TEXT))
                name = f"{step:07d}.json"
                (run.dir / "snapshots" / name).write_text(json.dumps({**base, **snap}))
                run.log("snapshots.jsonl", {**base, "file": name})
            nxt["probe"] += p["probe_every"]
        if final or tokens >= nxt["sample"]:
            write_samples(model, enc, run, step, tokens, p["sample_len"])
            nxt["sample"] += p["sample_every"]

    ctl = run.dir / "control.json"
    ctl.unlink(missing_ok=True)                  # an old request must not stop a fresh start

    def control():
        """The app's buttons: pause / resume / end early (a small file written by oazy.dashboard)."""
        if not ctl.exists():
            return None
        try:
            act = json.loads(ctl.read_text()).get("action")
        except Exception:
            return None
        ctl.unlink(missing_ok=True)
        return act

    def status(state, **extra):
        run.status({**info, "state": state, "step": step, "tokens": tokens, "elapsed": elapsed,
                    "latest": latest, "updated": time.time(), **extra})

    end_early = False
    model.train()
    last_ckpt = time.time()
    t_log, tok_log, acc_loss, acc_n = time.time(), tokens, 0.0, 0
    log_every = 10
    share_credit, kd_log, kd_log_n = 0.0, 0.0, 0      # Frog's mentor share: the running tally; its KL since the last log
    run_steps = 0                                     # steps taken by this process (a race does not count its first 3)
    while not done() and not stop["flag"]:
        act = control()
        if act == "pause":
            save_ckpt(run, model, opt, ckpt_meta())
            last_ckpt = time.time()
            say(f"paused at step {step} (checkpoint saved). Resume or end it from the app.", flush=True)
            status("paused")
            while not stop["flag"]:
                act = control()
                if act in ("resume", "end"):
                    break
                status("paused")                     # keeps the card fresh
                time.sleep(2)
            if act == "resume":
                say("resumed", flush=True)
                status("running")
                t_log, tok_log, acc_loss, acc_n = time.time(), tokens, 0.0, 0
        if act == "end":
            end_early = True
            break
        if stop["flag"]:
            break
        t0 = time.time()
        evaluations()
        t_eval = time.time() - t0

        if hasattr(model, "grow_to") and model.grow_to(prog()):     # Ember, Fern2: time to grow
            old, opt = opt, new_opt()
            carry_state(old, opt)
            deep = model.depth_text() if hasattr(model, "depth_text") else f"{model.depth} blocks deep"
            say(f"[grow] {tokens / 1e6:.0f}M tokens: {NICE[args.model]} is now {deep}", flush=True)
            run.log("metrics.jsonl", {"kind": "grow", "step": step, "tokens": tokens, "elapsed": elapsed,
                                      "depth": model.depth})
        mw = mentor.weight_at(prog()) if mentor is not None else 0.0
        if mentor is not None and mw <= 0:
            say(f"[mentor] {tokens / 1e6:.0f}M tokens: the mentor steps back; {NICE[args.model]} learns alone now", flush=True)
            mentor = None
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if growing_ctx and model.ctx_at(prog()) != T:          # Bee: time for longer windows
            T = model.ctx_at(prog())
            micro, accum = shape(T, prog())
            B = micro * accum
            train.set_block(T)
            train.set_batch(micro)
            if replay is not None:                   # the kept passages no longer fit a row: the buffer starts again
                replay.reset(T)
            say(f"[context] {tokens / 1e6:.0f}M tokens: {NICE[args.model]} now reads windows of {T} words "
                f"({micro} x {accum} per step)", flush=True)
            run.log("metrics.jsonl", {"kind": "context", "step": step, "tokens": tokens, "elapsed": elapsed, "context": T})
        if batch_ramp and accum != shape(T, prog())[1]:         # the batch ramp: more passes per step
            micro, accum = shape(T, prog())
            B = micro * accum
        if hasattr(train, "progress"):
            train.progress = prog()
        if select is not None:
            select.progress = prog()
        if hasattr(model, "set_progress"):          # Pegasus 2: fixed thoughts early, the meaning loss warming up
            model.set_progress(prog())
        cur_lr = lr_fn(prog(), step, lr, p["warmup"])
        for g in opt.param_groups:
            g["lr"] = cur_lr * g.get("lr_scale", 1.0)
        loss_sum, kd_sum, rd_sum, hs_sum, n_kd = 0.0, 0.0, 0.0, 0.0, 0
        fore_sum, pass_sum = 0.0, 0.0

        def lesson(x, y, mw):
            with amp(device):
                if hasattr(model, "losses"):        # Bee: the loss and the mentor's lesson in slices (oazy/lmloss.py)
                    loss, st = model.losses(x, y, mentor if mw > 0 else None, mw)
                    (loss / accum).backward()
                    return loss, st
                logits, loss, st = model(x, y)
                # Ember learns the mentor's guesses with its own; Fern2 with its reader (st["kd_logits"])
                student = st.get("kd_logits", None if hasattr(model, "mix_mentor") else logits)
                if mw > 0 and student is not None:
                    from ember.mentor import mentor_loss
                    kd = mentor_loss(student, mentor.guess(x))
                    if hasattr(model, "mix_mentor"):
                        loss = model.mix_mentor(loss, st, kd, mw)
                    else:
                        loss = (1 - mw) * loss + mw * kd
                    st = {**st, "kd": kd}
            (loss / accum).backward()
            return loss, st

        for _ in range(accum):
            x, y = train.get(device)
            if replay is not None and select.on():  # a share of the rows: passages closest to clicking, read again
                x, y, row_reads = replay.mix_in(x, y)
            else:
                row_reads = None
            pw = mw
            if mentor_share and mw > 0:
                # Frog: a running tally hands whole passes to the mentor so that their share tracks w exactly (at
                # w = 0.25, one pass in four); those learn only its guesses (weight 1), the others only the real next
                # word (weight 0). On average the lesson is Bee3's; the mentor reads a share w of the text, not all of it.
                pw, share_credit = mentor_pass(share_credit, mw)
            try:
                loss, st = lesson(x, y, pw)
            except Exception as e:
                if model._compiled is None:
                    raise
                say(f"torch.compile failed ({type(e).__name__}: {e}); continuing without compile", flush=True)
                model._compiled = None
                loss, st = lesson(x, y, pw)
            if replay is not None:                  # score this pass's passages; the buffer keeps the best
                rr = row_reads if row_reads is not None else torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
                replay.offer(x, y, select.passage_scores(x.shape[0]), rr)
            loss_sum += st.get("ce", loss).detach()       # the next-word loss (Tide also trains its far drafts)
            if "kd" in st:
                kd_sum += st["kd"].detach()
                n_kd += 1
            if "reader_ce" in st:
                rd_sum += st["reader_ce"]
            if "recall_hindsight" in st:
                hs_sum += st["recall_hindsight"]
            if "fore" in st:                        # Plain-100F: the foresight head's loss
                fore_sum += st["fore"]
            if "passes" in st:                      # Pegasus Alpha Star: passes through the core this time
                pass_sum += float(st["passes"])
        if mentor_share and n_kd:
            kd_log, kd_log_n = kd_log + float(kd_sum), kd_log_n + n_kd
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if hasattr(model, "after_step"):            # Coral: the experts' balancing biases move
            model.after_step()
        step += 1
        tokens += B * T
        acc_loss += loss_sum / accum
        acc_n += 1
        run_steps += 1
        if not (budget and run_steps <= 3):         # a race: the first steps of a start compile, they are not counted
            elapsed += time.time() - t0 - t_eval    # training time only

        if step % log_every == 0 or done():
            now = time.time()
            tps = (tokens - tok_log) / max(1e-6, now - t_log)
            t_log, tok_log = now, tokens
            own_loss = float(acc_loss) / acc_n
            rec = {"kind": "train", "step": step, "tokens": tokens, "elapsed": elapsed,
                   "loss": own_loss * loss_scale_at(prog()), "lr": cur_lr, "tok_s": tps, "grad_norm": float(gnorm)}
            if tk != "gpt2":
                rec["loss_own"] = own_loss
            if hasattr(model, "grow_to"):
                rec["depth"] = model.depth
            if hasattr(model, "moe_report"):         # Coral: how evenly the experts worked, how much overflowed
                rec.update(model.moe_report())
            if hasattr(model, "thought_report"):     # Pegasus: tokens per thought, the meaning loss
                rec.update({k: v for k, v in model.thought_report().items() if v is not None})
            if hasattr(train, "weights"):
                rec["context"], rec["mix"] = T, {k: round(v, 3) for k, v in train.weights(prog()).items()}
                rec["batch"] = B * T
            if mw > 0:
                rec["mentor_w"] = mw
                if not mentor_share:
                    rec["mentor_kl"] = float(kd_sum) / accum
            if mentor_share and kd_log_n:           # the mentor's passes since the last log line only
                rec["mentor_kl"], rec["mentor_passes"] = kd_log / kd_log_n, kd_log_n
                kd_log, kd_log_n = 0.0, 0
            if select is not None:
                rec.update(select.report())
                if replay is not None:
                    rec.update(replay.report())
            if rd_sum:
                rec["reader_loss"] = float(rd_sum) / accum
            if hs_sum:
                rec["recall_hindsight"] = float(hs_sum) / accum
            if hasattr(model, "foresight_report"):
                rec["fore_loss"], rec["fore_w"] = float(fore_sum) / accum, model._fore_w
            if pass_sum:
                rec["passes"] = pass_sum / accum
            acc_loss, acc_n = 0.0, 0
            run.log("metrics.jsonl", rec)
            eta = (total - tokens) / max(tps, 1)
            if budget:                              # a race: the clock decides; the token total is the pace so far
                eta = max(0.0, budget - elapsed)
                if prog() > 0.02:
                    info["total_tokens"] = int(tokens / prog())
                    (run.dir / "config.json").write_text(json.dumps(info, indent=1))
            run.status({**info, "state": "running", "step": step, "tokens": tokens, "elapsed": elapsed,
                        "tok_s": tps, "eta_s": eta, "loss": rec["loss"], "latest": latest, "updated": time.time()})
            frac = min(1.0, tokens / max(total, 1)) if not budget else min(1.0, elapsed / budget)
            sep = term.c(" │ ", "grey")
            step_s, loss_s = term.c(f"{step:>7,}", "dim"), term.c(f"{rec['loss']:.4f}", "bold")
            say(f"{step_s}  {term.bar(frac, 12)} {frac * 100:5.1f}%{sep}"
                f"{term.pad(term.tokens(tokens), 7, True)}{sep}loss {loss_s}{sep}"
                f"lr {cur_lr:.1e}{sep}gnorm {float(gnorm):.2f}{sep}{term.pad(term.rate(tps), 12, True)}{sep}"
                f"eta {term.pad(term.duration(eta), 7, True)}"
                + (f"{sep}experts {rec['moe_uneven']:.2f} uneven, {rec['moe_dropped'] * 100:.1f}% dropped" if "moe_uneven" in rec else "")
                + (f"{sep}thoughts {rec['thought_len']:.1f} tok{sep}meaning {rec['meaning_loss']:.2f}" if "thought_len" in rec else "")
                + (f"{sep}passes {rec['passes']:.1f}" if "passes" in rec else "")
                + (f"{sep}foresight {rec['fore_loss']:.2f} (x{rec['fore_w']:.2g})" if "fore_loss" in rec else "")
                + (f"{sep}picks {rec['select_kept'] * 100:.0f}% (w {rec['select_w']:.2f})"
                   + (f" · reread {rec['replay_share'] * 100:.0f}%" if "replay_share" in rec else "") if "select_kept" in rec else ""),
                flush=True)

        if time.time() - last_ckpt > args.ckpt_min * 60:
            save_ckpt(run, model, opt, ckpt_meta())
            last_ckpt = time.time()

    save_ckpt(run, model, opt, ckpt_meta())
    if done() or end_early:
        if end_early:
            say(f"ended early from the app at {tokens / 1e6:.0f}M tokens (checkpoint saved).", flush=True)
        status("finishing")
        say("training done; final evaluation (full HellaSwag takes a few minutes)...", flush=True)
        evaluations(final=True)
        status("done", ended_early=end_early)
        say(f"finished: {json.dumps(latest)} after {elapsed / 60:.1f} min of training. checkpoint: {ck}")
    else:
        run.status({**info, "state": "stopped", "step": step, "tokens": tokens, "elapsed": elapsed,
                    "latest": latest, "updated": time.time()})
        say(f"stopped at step {step}; run the same command again to resume.")


if __name__ == "__main__":
    main()
