# Bee4: 296M, all attention, Coral's recipe

**Next-word model**, exactly like GPT-2 and every Bee: it reads text and guesses the very next token, and it is
scored the same way (held-out loss per GPT-2 token on 1,024-token FineWeb-Edu windows, HellaSwag, ARC-Easy, PIQA,
LAMBADA).

Bee4 is Coral's dense sibling: the same layers and the same recipe, but one ordinary MLP per layer instead of
Coral's 32 experts, and no Memtrix memory layers. All 296M parameters work on every word.

| | Bee3 | Coral | Bee4 |
|---|---|---|---|
| width | 768 | 1,024 | 1,024 |
| layers | 24 (18 memory + 6 attention) | 20, all attention | 20, all attention |
| attention | 12 query / 4 key-value heads, no positions | 16 / 4 heads of 64, RoPE (base 50,000) | the same as Coral |
| thinking part per layer | one MLP of 1,792 | 32 experts of 448 (top 2) + a shared MLP of 512 | one MLP of 2,816 |
| parameters | 203M | 1,037M (161M at work) | **296.8M** (246M without the word table), all at work |
| mentor | SmolLM2-360M, all run | SmolLM2-360M, 1 pass in 4, gone by 30% | **SmolLM2-1.7B**, 1 pass in 4, then 1 in 10 from 30%, off over the last 10% |
| tokens per step | 262k | 1,049k | 524k (after a ramp from 262k over the first 5%) |
| learning rates | 2.4e-3 / Muon 0.04 | 1.8e-3 / Muon 0.03 | 1.8e-3 / Muon 0.03 |
| text | Tagus | Tagus20 | Tagus20 |

## What it keeps from Coral

- **Growing depth**: 8 layers for the first 10%, 12 until 25%, then 20. Each growth copies the top layers and puts
  the copies on top, so they start as working layers.
- **Plain causal attention** on the GPU's fused kernel: the documents packed in a 2,048-word window may see the
  earlier ones (keeping them apart cost about 40% of a pass on a B200). The mentor still reads each document on its
  own. `--set attn_docs=1` brings the rule back.
- **FP8** for every attention and MLP matrix product (the word table stays bf16), the stored RoPE table, the loss in
  slices, Muon with weight decay 0.1, the batch ramp, the WSD schedule with the square-root cool-down, and Tagus20's
  mix with its chat-heavy cool-down.

## The mentor: bigger, and kept on

A 300M student is almost as big as SmolLM2-360M, so Bee4's mentor is **SmolLM2-1.7B** (same pieces, trained on
11T tokens). The mentor teaches a share of the passes (each such pass learns only the mentor's guesses, the rest
only the real next word): 1 in 4 at the start, falling to 1 in 10 by 30%, kept there, and fading to none over the
last 10% so the cool-down ends on real text. Over the run it reads about 12% of the passes. `--mentor smollm2-360m`
puts the small one back, `--mentor none` drops it.

## Speed, honestly

Per word at full depth, Bee4 does about 2.0 GFLOP of training work (6 x 296M for the weights, plus attention over
2,048 words), against about 1.5 for Coral; dense products run better on the GPU than Coral's experts. My estimate
before the profile, for the student alone: **about 200-240k tokens/s**. The 1.7B mentor costs about 450 ms per
65k-token pass (an estimate), on 12% of the passes: about 15% of the run. Whole run: **about 170-200k tokens/s**.
6,000M tokens is then **about 8.5-10 hours** on one B200. The profile measures it:

    python3 -m oazy.profile --model bee4 --preset big --micro 32

## Run it

See `docs/RUNPOD.md`, section "Bee4". It reads the same Tagus20 as Coral; nothing new to build.

## Files

- `bee4/model.py`: `Bee4Config` (sizes, growing, the attention rule) and `Bee4` (Frog's growing body, all attention).
- `oazy/train.py`: `RECIPE["bee4"]`, `MENTOR["bee4"]`, the presets' `batch_bee4` / `lr_bee4` / `muon_lr_bee4`.
- `bee4/tests.py`: self-tests (size, attention rule, reading = training, word-by-word chat, growing, the recipe, a
  whole tiny run and its resume, and on the GPU the compiled step and FP8).
- `bee4/chat.py`: the chat app. The phone app exports it too (`python3 -m phone.export --run bee4`).
