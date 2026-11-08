# Pegasus 1: predict meaning, then words

**Still a next-word model at the end.** Its Speaker writes and is scored one token at a time, exactly like GPT-2
and every Bee: held-out loss, HellaSwag, ARC-Easy, PIQA and LAMBADA measure it the same way.

The idea (thread "Reach 55% HellaSwag with a $100 400M model", Nov 7 2026): a normal model is graded on the exact
next word, so most of its effort goes into word-choice noise ("sliced" or "cut"?). Pegasus learns mostly one
level above words: what happens next.

## The parts

| part | what it does | Pegasus 1 | Pegasus 1 Mini |
|---|---|---|---|
| **Reader** | reads every token (local attention, last 256 tokens); cuts the text into **thoughts** where its reading suddenly changes | 4 layers x 768 | 3 x 512 |
| **Meaning pieces** | each finished thought becomes 8 pieces, each one of **1,000 meanings** from one shared dictionary the model invents (an 8x5x5x5 grid, FSQ) | 8 x 1,000 | 8 x 1,000 |
| **Thinker** | one step per thought; attention over all earlier thoughts; predicts the **next thought's pieces**, one after another, with real odds | 15 x 1,280 (~283M) | 8 x 768 (~55M) |
| **Speaker** | writes the words, next-token, from the Thinker's plan for this thought, the last 256 tokens, and a **notebook** (one page per earlier thought, to copy names and numbers) | 6 x 768 | 4 x 512 |
| total | | **~400M** | **~105M** |

- **Thought length**: a soft spring (H-Net's ratio loss) pulls the average towards 16 tokens (~12 words). It is
  never a wall: dense text can make short thoughts, slow text long ones. A new document always starts a new thought.
- **Training**: the Speaker's next-word loss (with the mentor's lesson on a share of the passes) trains every part,
  through the pieces into the Reader. The Thinker also learns the **meaning loss** (weight 0.3): guess the next
  thought's pieces. The boundaries learn from the spring and from the Speaker's loss (H-Net's confidence trick).
- **No peeking**: a token is written from the plan made after the last thought that ended before it, the pages of
  thoughts that ended before it, and the tokens up to it. `pegasus/tests.py` checks this by changing later tokens.
- **Cost**: the Thinker runs on ~1/8 of the positions (room for one thought per 8 tokens), so Pegasus 1 costs per
  token about what a ~150-200M normal model costs, and the Mini about what a ~60M one costs. Estimates, not yet
  measured on the B200.

Not in Pegasus 1 yet (ideas from the thread for later): a variable number of pieces per thought (it uses 8 always),
the "be predictable" pressure on the pieces themselves (here only the Thinker learns to predict them; the pieces
are shaped by what the Speaker needs), the learnability-driven data selection.

## The race: Pegasus 1 Mini against Plain-100M

**Plain-100M** (`pegasus/baseline.py`) is a normal next-word transformer of the same size: Bee4's layers, 10 x 768,
~106M. Both are raised with exactly Bee4's recipe (`RECIPE["pegasusmini"] = RECIPE["plain100"] = RECIPE["bee4"]`):
short windows first, the SmolLM2-360M mentor on a share of the passes, the cool-down over the last 30%, bf16, 524k-token
steps, Tagus20. So the two runs differ only in the architecture. Neither grows.

What the training log adds for Pegasus: `thoughts` (tokens per thought) and `meaning` (the meaning loss, nats per
piece; guessing blindly is ln 1000 = 6.9).

## Run it

See `docs/RUNPOD.md`, section "Pegasus 1".

## Files

- `pegasus/model.py`: `PegasusConfig` (`size="full"` or `"mini"`) and `Pegasus`.
- `pegasus/baseline.py`: `Plain100`, the race's normal transformer.
- `pegasus/tests.py`: sizes, no peeking, the pieces, every part learns, the recipe, a whole tiny run of each.
- `oazy/train.py`: kinds `pegasus`, `pegasusmini`, `plain100`.

## Pegasus 2 Mini (Nov 8 2026)

Pegasus 1's first run (2.2B tokens) learned slowly, and its loss stopped improving past word 256, which is exactly how far the Speaker can see on its own. So the Thinker and the notebook were adding nothing. `model2.py` rebuilds the Mini with these fixes:

- **The Speaker is a real language model.** The Reader (2 layers) and the Speaker (7 layers) form one 640-wide stream, so the Speaker continues from what the Reader made instead of from bare word vectors.
- **The Thinker reads rich thoughts.** It is now 3 x 768 and reads unrounded thought vectors. The 8 pieces of 1,000 meanings are only what it must predict next. A rebuild loss (the thought's vector must be recoverable from its pieces) keeps them from going empty. The log shows `codes_used`.
- **Notebook pages keep details.** Each thought leaves 2 pages, each made by a learned query attending over the thought's tokens (it used to be their average).
- **Fast windows.** Compiled steps use flex attention, with the keys copied out to every head (the shared-heads mode crashes Triton on the B200). `OAZY_FLEX=0` falls back to the grids.
- **Fixed thoughts first.** For the first 10% of training, a new thought starts after a sentence end or every 32 tokens, and the router learns to imitate that. After that the router decides. The meaning loss is off until 10% and reaches full weight at 20%.
- **Same recipe as Plain-100M.** Same mentor, same recipe (`RECIPE["pegasus2"] = RECIPE["plain100"]`).
- **Plan probe.** Every probe logs `[plan]`: the Speaker's loss with the Thinker's plan, without it, and without the notebook. The gap says whether the idea is alive.

Sizes: 105.1M in all (Plain-100M: 106.6M), about 83M at work per token. Tests: `python3 -m pegasus.tests2 --device cuda`, which also runs a speed race against Plain-100M.
