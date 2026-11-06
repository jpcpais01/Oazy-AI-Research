# Frog: Bee3 at twice the size

**Next-word model**, exactly like GPT-2 and every Bee: it reads text and guesses the very next token, and
it is scored the same way (held-out loss per GPT-2 token on 1,024-token FineWeb-Edu windows, HellaSwag,
ARC-Easy, PIQA, LAMBADA).

**Bee3's body and recipe, scaled up**: 406M parameters, 4,000M tokens of Tagus. Frog against Bee3 measures
what size buys. The goal set for it: under twice the compute of Bee3's run.

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

Two switches, the same maths, each checked against the plain way on the GPU when the run starts and left
off if it disagrees (the self-tests print which are in use):

- **flex attention for the document rule** (`flex_attn=1`, `bee/model.py`): the "earlier words of the same
  document" rule as a block mask, so the parts of the score grid that cross documents are skipped and the
  fast kernel is kept, instead of a full yes/no grid that forces a slower one.
- **the memory gates inside the memory kernel** (`fused_gates=1`, `memtrix/memory.py`): scaling question and
  key to length 1, the trust's sigmoid and the fade's `-exp(A_log) * softplus(f + dt_bias)` are done by the
  flash-linear-attention kernel itself, so their full-size fp32 copies and gradients are never made.

With the 524k steps (half the optimizer steps) and the mentor share, the estimate scaled from Bee3's profile
is about 1.7-2.0x Bee3's run, 7.5-9 hours on a B200. `python3 -m oazy.profile --model frog --preset big
--micro 32` measures it, including a whole run's speed with the mentor share.

## Run it

See `docs/RUNPOD.md`, section "Frog". Start after Bee3 has finished.

## Files

- `frog/model.py`: `FrogConfig` (the sizes and the two speed switches) and `Frog`, Bee3's body.
- `oazy/train.py`: `RECIPE["frog"]` (Bee3's + `mentor_share`), the presets' `lr_frog` / `muon_lr_frog` / `batch_frog`.
- `frog/tests.py`: self-tests (the size, the recipe, the tally, the mentor-only passes, the speed paths' maths, a whole tiny run).
- `frog/chat.py`: the chat app.
