# Frog: Bee3 at twice the size

**Next-word model**, exactly like GPT-2 and every Bee: it reads text and guesses the very next token, and
it is scored the same way (held-out loss per GPT-2 token on 1,024-token FineWeb-Edu windows, HellaSwag,
ARC-Easy, PIQA, LAMBADA).

**Bee3's body and recipe, scaled up**: 406M parameters, 4,000M tokens of Tagus. Frog against Bee3 measures
what size buys. The goal set for it: under twice the compute of Bee3's run; the speed work below aims at ~1.4x.

| | Bee3 | Frog |
|---|---|---|
| width | 768 | 1,024 |
| layers | 24 (18 memory + 6 attention) | 28 (21 memory + 7 attention), the same 3:1 |
| memories per memory layer | 6 of 128 x 128 | 8 of 128 x 128 |
| attention heads (query / key-value) | 12 / 4 | 16 / 4, heads of 64 |
| MLP width | 1,792 | 2,560 |
| parameters | 203M | 406M |
| tokens | 2,500M | 4,000M |
| tokens per step | 64k rising to 262k | 64k rising to **524k** |
| learning rates | 2.4e-3 / Muon 0.04 | **1.8e-3 / Muon 0.03** |
| the mentor | SmolLM2-360M, a share w of **every word** | SmolLM2-360M, a share w of **the sequences** |

Everything else is Bee3's (`bee3/README.md`): no bleed between documents, 2,048-word windows, the mentor's
schedule (0.6 to 0.25 by halfway, held, out over the last 10%), Muon weight decay 0.1, the batch ramp, the
square-root cool-down, the loss in slices, the chat fine-tune.

## The mentor on a share of the sequences

Bee3 mixes two lessons on every word: (1 - w) x "guess the real next word" + w x "match the mentor's
guesses". Frog gives whole passes (32 sequences) to one lesson or the other: a running tally hands a pass to
the mentor each time it has gathered a whole one (at w = 0.25, every fourth pass), and those learn only the
mentor's guesses, the rest only the real next word. On average the lesson is the same; the mentor reads only
the passes it teaches, about a third of the run instead of all of it. (`mentor_pass()` in `oazy/train.py`,
`--mentor-share 0/1`.) The logs keep `mentor_w` and the mentor's KL over its own passes (`mentor_passes`).

**Why a 360M mentor for a 406M student.** What makes it a good teacher is what it has read, not its size:
SmolLM2-360M was trained on about 4,000,000M tokens, a thousand times Frog's run, and scores about 54% on
HellaSwag (Bee2 33%). Frog will still be far behind it at the end of its run.

## Speed

Bee3 uses about 10% of the B200's arithmetic; well-tuned training reaches 35-50%. Frog goes after that gap
with five changes. Every one computes the same maths as the plain way (or, for FP8, nearly), and every one is
checked on the GPU when the run starts and left off if it disagrees. `python3 -m frog.tests --device cuda`
prints which are in use.

1. **The memory kernel as one compiled operator** (`memory_op=1`, `memtrix/memory.py`). Bee3 calls the
   flash-linear-attention kernel through `torch.compiler.disable`, so the compiler stops before every memory
   layer and starts again after it: 21 stops forward and 21 backward. Nothing can be fused across those lines,
   and the hand-offs (copies, casts, gradient sums) run one small step at a time. Frog registers the kernel's
   forward and backward as a PyTorch operator (`oazy::memory_fwd` / `oazy::memory_bwd`), so the whole
   28-layer stack compiles as **one graph**. The self-tests check that it compiles without a single break.
   The operator binds flash-linear-attention 0.6's kernel pieces. Where that version is missing, it falls back
   to the **fused gates** (`fused_gates=1`): the kernel itself scales q and k and squashes trust and fade.
2. **FP8 for the big matrix products** (`oazy/fp8.py`, torchao). The 154 large Linear layers (projections
   and MLPs) multiply in 8-bit floats, about 2x faster on a B200. Master weights, the word table, the small
   gates and everything else stay in 16/32 bits. A loss check against bf16 runs before it is kept, and
   `--fp8 0` turns it off.
3. **It grows** (`grow="8:0,16:0.15,28:0.35"`). Frog trains as 8 layers for the first 15%, 16 until 35%,
   then all 28. Each growth copies the top 4-layer groups and puts the copies on top ("stacking"), so the
   new layers start out working. The pattern of 3 memory layers to 1 attention layer stays intact.
4. **flex attention for the document rule** (`flex_attn=1`, `bee/model.py`). The parts of the score grid
   that cross documents are skipped, and the fast kernel is kept.
5. **524k-token steps** after the ramp, so the optimizer runs half as often. And the **mentor on a share of
   the sequences**, so it reads about a third of the run.

The estimate, scaled from Bee3's profile: about 1.4x Bee3's compute, 5.5-6.5 hours on a B200. The run
without these changes would cost 2.4x. Only the profile on the pod turns that estimate into a number:
`python3 -m oazy.profile --model frog --preset big --micro 32`. It times the full depth; the shallow first
35% of the run comes on top of that.

## Run it

See `docs/RUNPOD.md`, section "Frog". Start after Bee3 has finished.

## Files

- `frog/model.py`: `FrogConfig` (the sizes, the growth plan, the speed switches) and `Frog`, Bee3's body that grows.
- `oazy/fp8.py`: FP8 training (torchao) with its check. `memtrix/memory.py`: the memory operator and the fused gates.
- `oazy/train.py`: `RECIPE["frog"]` (Bee3's + `mentor_share`), the presets' `lr_frog` / `muon_lr_frog` / `batch_frog`.
- `frog/tests.py`: self-tests (the size, the recipe, the tally, the mentor-only passes, the speed paths' maths, a whole tiny run).
- `frog/chat.py`: the chat app.
