# Coral: 1B parameters, about 160M at work per word

**Next-word model**, exactly like GPT-2 and every Bee: it reads text and guesses the very next token, and it is
scored the same way (held-out loss per GPT-2 token on 1,024-token FineWeb-Edu windows, HellaSwag, ARC-Easy,
PIQA, LAMBADA).

**The goal:** a 1B model trained on 20,000M tokens of Tagus at about 1M tokens per second on one B200, so the
whole run takes about 5.5 hours. A dense 1B cannot do that on one B200 (it needs about 3x the chip's peak), so
Coral is a **mixture of experts**: 1,037M parameters in total, but each word only uses about 161M of them
(211M counting the word table, which every word uses as the score head).

| | Frog | Coral |
|---|---|---|
| width | 1,024 | 1,024 |
| layers | 28 (21 memory + 7 attention) | 20, all gated attention with RoPE (`--set pattern=MMMA`: the Memtrix hybrid) |
| thinking part per layer | one MLP of 2,560 | 32 experts of 448, a router picks 2 per word, + a shared MLP of 512 |
| parameters | 406M, all at work | 1,037M, 161M at work (+50M word table) |
| tokens | 4,000M | 20,000M (its own Tagus, `tagus20`) |
| tokens per step | 524k | **1,049k** (after a ramp from 262k over the first 5%) |
| learning rates | 1.8e-3 / Muon 0.03 | 1.8e-3 / Muon 0.03 (Muon also on every expert's grids) |
| the mentor | SmolLM2-360M, share of the sequences, all run | SmolLM2-360M, **1 pass in 4 at the start, fading to none by 30%** |

## How the experts work, plainly

Every layer has a look-back part (attention) and a thinking part. In Coral the thinking part is 32 small MLPs
("experts") and one shared MLP. A tiny router scores the 32 experts for each word and sends the word to its best
2; their outputs are added, weighted by the router's scores, together with the shared MLP's. All the experts hold
the knowledge, but each word pays for only 2 of them. Per word read, a model like this learns roughly like a dense
~400M model (a rule of thumb, not a law: the run will tell), at about the cost of a dense ~200M one.

The risk with experts is that the router falls in love with a few of them, and the rest never learn. Coral keeps
the work even the way DeepSeek-V3 does: each expert has a small **bias** that is added to its score only when
choosing (never to the weights of the answer). After every step a busy expert's bias goes down by 0.001 and an idle
one's goes up, so the work evens out without a loss term pulling against learning. A tiny balance loss (1e-4) is a
safety net. The log line shows both: `experts uneven 1.10` (the busiest expert's work over the average; 1.0 is
perfect) and `dropped 0.8%` (see below).

**Fixed room per expert, in training.** To run fast, every expert gets a fixed block of slots per pass: 1.25x its
fair share. Every shape is then fixed, the 32 experts run as one batched matrix product, and the compiler sees one
graph. A word that arrives when its expert is full skips that expert (it still has the shared MLP and its other
expert). With even work this is around 1% of word-expert pairs. Evaluation and chat drop nothing.

## Everything we learned, kept

- Frog's speed work: **FP8** for the big matrix products (attention and the shared MLP; the experts stay bf16),
  **growing depth** (8 layers for the first 10%, 12 until 25%, then 20;
  each growth copies the top layers, experts and balancing biases included), the **loss in slices**.
- Bee3's recipe, except that attention is plain causal (the documents packed in a window can see the earlier ones;
  keeping them apart with flex attention cost ~40% of a pass on a B200; `--set attn_docs=1` brings it back), 2,048-word windows, Muon with weight decay 0.1, a batch ramp, the WSD
  schedule with the square-root cool-down, the Tagus mix with its chat-heavy cool-down.
- Frog's **mentor on a share of the sequences**, here only early, where it helps most: 1 pass in 4 at the start,
  fading to 0 by 30%, so it reads about 4% of the run (about 3-4% of the speed). `--mentor none` turns it off.
- No Memtrix memory layers by default: at 2,048-word windows their kernel is the slowest part of Bee3 and Frog,
  and their long-text advantage does not show at this length. `--set pattern=MMMA` brings them back (15 memory +
  5 attention, no RoPE), e.g. for a speed comparison with the 10-minute profile.

## Speed, honestly

Per word, Coral does about 1.3 GFLOP of weight products plus attention over 2,048 words. At 1M tokens/s that is
about 1.5 PFLOP/s, two thirds of the B200's bf16 peak, and a little less with FP8 on half of the products. So
1M tokens/s needs the best-run training we have done by far (Bee3 used about 10% of the chip). My estimate before
the profile: **0.7M to 1.0M tokens/s**, with growing depth adding about 12% on average. The profile measures it:

    python3 -m oazy.profile --model coral --preset big --micro 32

If it falls short, the next levers, in order: a bigger pass (`--micro 48` or `64`, there is memory to spare),
FP8 for the experts too (torchao's grouped FP8 products), and top-1 routing (`--set top_k=1`, cheaper, a little
worse).

## Run it

See `docs/RUNPOD.md`, section "Coral". Its Tagus is separate (`<data>/tagus20`), so it can be prepared while Frog
trains, with `nice`.

## Files

- `coral/model.py`: `CoralConfig` (sizes, experts, balancing, growing, the hybrid switch), `Experts` (router,
  fixed-capacity and dropless paths, the balancing biases), `CoralBlock`, `Coral` (Frog's growing body with experts).
- `oazy/train.py`: `RECIPE["coral"]`, the presets' `batch_coral` / `lr_coral` / `muon_lr_coral`, `--data-sub`,
  the step hook that moves the biases, the experts' numbers in the log.
- `oazy/muon.py`: Muon on stacks of grids (each expert straightened on its own).
- `oazy/tagus.py`: `--into tagus20`, a separate 20,000M-token Tagus with the same held-out text as every model.
- `oazy/profile.py`: times the router + experts; MFU counts only the parameters at work.
- `coral/tests.py`: self-tests (sizes, the two expert paths agree, overflow counting, gradients, balancing, Muon,
  growing, the recipe, Tagus's held-out filter, a whole tiny run).
- `coral/chat.py`: the chat app.
