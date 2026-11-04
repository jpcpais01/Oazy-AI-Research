# Settle: a language model that thinks by settling

Most language models push each word through a fixed stack of layers. Settle
instead keeps a **sheet** of cells and lets it **settle**, like a liquid coming
to rest. Words that need more thought take more steps to settle, so the amount
of computation follows the difficulty of each word.

```
             looks back
  row 7      128 / 256   ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢     upper rows: long range, the "big picture"
  row 6       64 / 128   ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢
   ...                    ...
  row 1        2 / 4     ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢
  row 0        1 / 2     ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢     bottom row: the last word or two
                          Once upon a time , ...  ← one column per word, newest on the right
  gist (slow field)      ━━━━━━━━━━━━━━━━━━━━━━━     the mood of the text, drifting slowly
```

## How it works, in plain words

1. **The sheet.** Every word gets a column of 8 cells, and every cell holds
   128 numbers. Each row is a *scale*: a cell in row *r* listens to the cells
   2^r and 2·2^r words back in its own row. So the bottom row sees the last
   word or two, and the top row sees 128 and 256 words back. That is the
   multi-scale field.
2. **One small rule.** A single learned rule (a small neural network, the
   same weights everywhere) updates every cell from what it can see: itself,
   its two neighbours to the left in its row, the cells above and below it, the
   word written into its column, the gist, and a few long-range links. It is
   like the physics of a cellular automaton, except the physics is learned.
3. **Settling is thinking.** The rule is applied again and again. When a
   column stops changing (its change in one step falls below a tolerance), it
   is *settled* and freezes. A step cap (24) stops runaway thinking. Training
   adds a small penalty for columns that are still moving at the end, so the
   sheet learns to come to rest (an equilibrium, in the spirit of Deep
   Equilibrium models).
4. **The gist (slow field).** Alongside the fast sheet runs a slow field: a
   leaky running average of the sheet at four time scales (the slowest
   remembers roughly the last few hundred words). It carries the overall mood
   and topic and feeds back into every cell.
5. **Long-range links.** Each column also sends out 4 small "wires" that land
   on whichever earlier words match what it is looking for (for example the
   name of the main character). Technically this is one small attention read
   per step, not stacked attention layers. Set `link_heads=0` to switch it off.
6. **The two-way lens.** Before predicting the next word, Settle copies the
   last 6 columns into a small workspace where influence flows **both ways**:
   the newest word reshapes how the previous few words are read, and they
   reshape it back. The lens settles too, and the next word is read from its
   right edge.

### Why the future never leaks

Training shows the model a whole story at once and asks it to predict every
next word in parallel. If later words could change earlier columns, the
model could peek at the answer. So the main sheet only ever listens to the
left (the past), and the two-way part happens in the lens, which every
position gets as its own private copy of *its own* past. Think of the sheet
as ink (once a column settles it never changes) and the lens as pencil (the
last few words are re-read in the light of the newest one, then the pencil
is erased for the next word). The self-tests check this directly: changing a
word never changes any prediction before it.

### Generation continues the sheet

When writing, each new word adds one column. It settles against the
already-settled past (nothing restarts), then the lens re-reads. The cost of
a word does not grow with the length of the text. Training and generation
follow the exact same per-column schedule (a column freezes at the same step
in both), so they give the same predictions to within rounding. The self-tests
and the dashboard both check this.

### How it learns

Settle backpropagates through every settling step of the sheet. In tests this
learned much faster than the cheaper "phantom gradient" trick (learning only
from the last few steps), so it is the default. The lens uses the cheap trick.
Set `full_grad=false` to use it for the sheet too (less memory, worse
learning).

## The experiment

Settle (22.1M parameters) is trained on TinyStories next to a nanoGPT-style
transformer of matched size (22.9M: 2 layers, width 384). Both see the same
data, the same tokenizer (GPT-2) and the same learning-rate schedule. Most of
both models is the 19.3M-parameter word table. What differs is the 2.8M
(Settle) vs 3.6M (GPT) "brain" parameters. Settle reuses its small rule many
times per word, so it spends more compute per parameter than the GPT. The
dashboard shows loss both against words seen and against training minutes,
so you can judge it both ways.

A small run on a laptop CPU (synthetic stories, 300k words) is only a smoke
test. There the transformer learned faster. The real run on TinyStories is the
first real test of the idea.

## Files

| file | what |
|---|---|
| `model.py` | Settle itself (the sheet, rule, gist, links, lens, word-by-word generation) |
| `gpt.py` | the matched GPT baseline |
| `data.py` | downloads and tokenizes TinyStories |
| `train.py` | training for both models: resumable, writes everything the dashboard shows |
| `viz.py` | colours for cells and the "photograph" of the sheet taken during training |
| `dashboard.py`, `web/dashboard.html` | the live training dashboard (port 8000) |
| `chat.py`, `web/chat.html` | the chat app that shows the sheet settling on each word (port 8001) |
| `tests.py` | self-tests: no future leak, generation equals training, gradients flow |

To run it on a RunPod H200 or B300, follow [`docs/SETTLE_RUNPOD.md`](../docs/SETTLE_RUNPOD.md).

## Settings you can change

Pass `--set key=value` to `python -m settle.train` (the GPT baseline is
re-matched to the new size automatically). Useful ones:

| setting | default | what it does |
|---|---|---|
| `settle_cap` | 24 | most steps a word may take |
| `settle_tol` | 0.002 | how still a column must be to count as settled |
| `rows` | 8 | scales (the top row looks 2^(rows-1) and 2^rows words back) |
| `channels` | 128 | numbers per cell |
| `hidden` | 1024 | size of the rule's hidden layer |
| `link_heads` | 4 | long-range links per column (0 turns them off) |
| `lens_width` | 6 | how many recent words the lens re-reads |
| `full_grad` | true | learn through every settle step |
| `res_weight` | 0.05 | how strongly unsettled columns are penalised |
