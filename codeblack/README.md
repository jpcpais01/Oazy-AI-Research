# Code Black: two clocks

**Next-word model**, exactly like GPT-2 and every Bee: every position guesses the very next token, and it is scored
the same way (held-out loss per GPT-2 token on 1,024-token FineWeb-Edu windows, HellaSwag, ARC-Easy, PIQA, LAMBADA).

**The goal:** an architecture that beats a strong modern transformer **on the B200's clock**: the same held-out loss
in half the training time, and text written 2-3x faster. The plan and the reasons are in the design doc,
"Code Black: a faster-than-transformer architecture" (https://claude.ai/artifact/Rv1vaApnEp7aRCJdsrLY9D).

## The idea, plainly

At our sizes and 2,048-token windows, attention is only ~13% of a transformer's training arithmetic. The rest is the
big matrix products every layer does for every token (~60%) and the word table (~27%). So swapping attention for
something "linear" cannot make training much faster (that is why Memtrix trained 2x slower than GPT-2, not faster).
A design only wins on the clock if it runs the expensive layers on fewer positions, gets knowledge that costs no
arithmetic, and keeps the chip busy. Code Black does all three:

```
tokens ──► word table + phrase memory ──► fast encoder ─────────── skip ──────────► fast decoder ──► next token
                                               │                                        ▲
                                     pack 4 tokens into 1                     unpack, one group later
                                               ▼                                        │
                                  slow stream: 16 wide layers, one position per 4 tokens
```

| part | Black-S (the first test size) | runs | arithmetic per token |
|---|---|---|---|
| word table + **phrase memory** | 25M words; 4 hashed tables of 131,072 x 256 (134M) | every token | almost none (lookups) |
| **fast stream**: 2 encoder + 2 decoder layers | width 512, attention over the last 256 tokens, 14M | every token | ~90 MFLOP |
| pack + unpack | 4M | once per 4 tokens | ~6 MFLOP |
| **slow stream**: 16 layers | width 1,024, 198M | once per 4 tokens | ~315 MFLOP |
| output word table | shared with the input | every token | ~150 MFLOP |
| **total** | 240M doing arithmetic + the phrase table | | **~560 MFLOP (Black-T 125M: ~950)** |

- **The phrase memory** (DeepSeek's Engram, Jan 2026; the nanoGPT speedrun's hashed n-gram table): the 2 and the 3
  tokens ending at each word are turned into a row number of a big table, and that row is read, with a gate that
  ignores it when it does not fit the context. A lookup costs no matrix arithmetic. It never reaches back across a
  document start. It sits after the first encoder layer, so its gate has a little context to judge by.
- **Pack**: every 4 tokens side by side, through one matrix, into one vector of width 1,024. Nothing is averaged
  away. Tokens of an earlier document are left out of a group that ends in a new one.
- **The slow stream** reads 512 positions per 2,048-token window: a quarter of the cost of the same stack run on every
  token. Plain attention by default (`slow_pattern=A`), cheap on 512 positions; `--set slow_pattern=MMMA` puts
  Memtrix memory layers in, 3 to 1 (rung R4, for long texts).
- **Unpack, one group later.** Each group's result becomes 4 vectors, for the last token of the group itself and the
  first 3 of the next one. No token ever sees, from the slow side, a token after it: that keeps it a fair
  next-token model (the self-tests change each token and check that nothing before it moves). Before the first group,
  and at the start of every document, tokens get a learned start state.
- **It starts as a fast transformer.** The unpack and the phrase memory's value start at zero, so in the first steps
  the model is the 4 fast layers alone, and the slow stream and the phrase memory grow in as they become useful.

In the language of the brain: fast senses that read every word, a slow cortex that updates once per event, and a
memory of familiar phrases that recalls without thinking. It is the "orchestra of clocks" idea, with two clocks.

## Black-T, the yardstick

A 2026 transformer, not GPT-2 as written in 2019: Bee3's gated attention (QK-norm, 4 shared key-value heads, sigmoid
output gate) in every layer with RoPE, SwiGLU, RMSNorm, no biases, a tied and soft-capped word table, no bleed
between documents (flex attention), Muon, the loss in slices, FP8 products. Same parts as every Bee, so a race
measures the architecture, not the code.

| size | width x layers | heads (query / key-value) | SwiGLU | parameters |
|---|---|---|---|---|
| `125m` (default) | 768 x 12 | 12 / 4 | 2,048 | 120M |
| `250m` | 1,024 x 16 | 16 / 4 | 2,752 | 244M |
| `500m` | 1,280 x 24 | 20 / 4 | 3,456 | 515M |

## The races

Every race gets **the same 30 minutes of B200 training time** (`--preset race --minutes 30`), so a faster model simply
reads more text. Every schedule follows the clock instead of the token count: the warm-up, the batch ramp
(65k to 262k tokens per step over the first 10%), the flat learning rate and the square-root cool-down over the last
20%, the Tagus mix and its chat-heavy finish. Evaluations, checkpoints and the first 3 steps of each start (the
compiler at work) are not counted. Both racers get the same recipe (`RECIPE["black"] = RECIPE["blackt"]`): Muon
(learning rate 0.04, weight decay 0.1) + AdamW 2.4e-3, FP8, **no mentor**, so the architectures are compared alone.

The pass and kill lines, set before any run (from the design doc):

| rung | race | pass | kill |
|---|---|---|---|
| R0 | profile Black-T and Code Black | Black-T at 35% of the chip or more | fix the trainer first |
| R1 | Black-T 125m, 250m, 500m | a smooth line of loss against minutes | (the yardstick) |
| R2 | Code Black, groups of 4 and of 2, no phrase memory | Black-T 125m's 30-minute loss in 20 minutes or less, HellaSwag no worse | both lose to Black-T 125m: fall back to design B |
| R3 | the phrase memory on | 0.02 lower loss for at most 5% more time | no gain: drop it |

Later rungs (R4 the memory-layer slow stream at 8,192 tokens, R5 shapes, R6 surprise-driven groups, R7 3x the time,
R8 fast generation with the slow stream waking once per group) come after these, once we know the bet holds.

## Run it

See `docs/RUNPOD.md`, section "Code Black". In short, on the pod, with nothing else on the GPU:

```bash
python3 -m codeblack.tests --device cuda                 # self-tests (fair next-token rule, compiled step = grids, FP8)
tmux new-window -t oazy -n race "bash codeblack/ladder.sh"   # R0-R3, ~3.5 h, resumable
python3 -m codeblack.race --png /workspace/oazy_runs/race.png  # the report: loss at 10/20/30 min, minutes to the yardstick
```

## What is not built yet

- **Fast generation.** `generate()` reads the whole text again for each new token: exact, simple, slow. The real
  thing (fast layers per token, the slow stream once per group, with caches) is rung R8, built once the races say
  Code Black is worth it. The chat app waits for it too.
- **Surprise-driven groups** (rung R6): where a group ends would be learned, at moments of surprise, instead of every
  4 tokens.

## Files

- `codeblack/model.py`: `BlackTConfig` / `BlackT` (the yardstick, sizes in `BLACKT_SIZES`), `BlackConfig` / `Black`
  (the two clocks), `PhraseMemory` (the hash and the gate).
- `codeblack/race.py`: the race report (loss at 10/20/30 minutes, minutes to the yardstick's loss, the R2 verdict).
- `codeblack/ladder.sh`: R0-R3 one after another.
- `codeblack/tests.py`: self-tests.
- `oazy/train.py`: `--minutes` (a race on the clock), `RECIPE["black"]`, the `race` preset. `oazy/profile.py`: Code
  Black's parts timed apart. `oazy/fp8.py`: FP8 for all three stacks.
