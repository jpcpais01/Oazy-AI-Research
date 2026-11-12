# Keep: a language model that keeps what it reads

NEXT-TOKEN. An experiment, not yet a model to chat with. "Keep" is a working name.

## The idea

A transformer knows a text only while the text is inside its context. When the context ends, what it read is gone,
unless training put it into the weights, which takes hundreds of exposures. Keep has a second place to put things: a
**store**, a table of rows that is part of the model's state, not of its weights. The store is never emptied: not at
the end of a document, not at the end of a training step.

What a store does, at every token:

1. **Read.** The model's vector at this token is turned into an address: 8 rows of the table. They hold what
   followed the last times the text was at a place with an address like this.
2. **Write, one token later.** When the next token has come, the store compares it with what those rows said would
   come and writes only the difference into them (the delta rule).

So the store is a predictor of what comes next that learns while the model reads, by its own prediction error, with no
gradient step. The weights only decide what counts as "a place like this", what to write and how hard.

| | Twin (plain transformer) | Keep | Keep-wide |
|---|---|---|---|
| Attention reaches | the whole window, 1,024 tokens | 128 tokens | the whole window, 1,024 tokens |
| Older text | gone | read back out of the stores | read back out of the stores |
| Cost per token as text grows | grows with the context | constant | constant past the window |
| State carried from window to window | none | the stores (2 x 16,384 rows x 64) | the stores |

Two choices, each made because the version without it failed in a CPU pilot (Nov 11 and 12):

- **Sparse write.** A write touches 8 rows of 16,384, so text about other things lands on other rows. A dense
  delta-rule memory lost a fact within ~4,000 unrelated tokens; a sparse one held 32 facts across 16,384
  (`/mnt/project-files/reads-by-learning/`).
- **Write what came next, under the address of the place before.** With the address of a token used for its own
  write, a question could only find its answer after the layers below had learned to build matching keys, and they
  never got a signal to. Stored as "what followed", a place that comes back finds its continuation from the first
  training step.

## What the CPU pilots show

One seed each, tiny models, hours not days: a reason to run the real test, not a result to quote. All numbers and
scripts: `/mnt/project-files/reads-by-learning/night3-results.txt`.

**Real text** (`--preset pilot`: 1.3M weights each, 12M tokens, plain-English facts with WikiText between, trained on
gaps up to 256 tokens). Plain facts answered right, 132 questions on scenes never trained on, guessing = 25%:

| Gap of WikiText tokens | 0 | 64 | 256 | 1,024 | 4,096 |
|---|---|---|---|---|---|
| Keep (attention reaches 32 tokens; answers from the stores alone) | 75.8% | 74.2% | 77.3% | 75.8% | 74.2% |
| Twin (attention over the whole 256-token window) | 29.5% | 30.3% | out of view | out of view | out of view |

- Keep holds flat out to 16 times the longest gap it trained on.
- The twin, with the story in full view, had not learned to look a fact up at all in 12M tokens. Keep was at 68%
  after 3M. The store does not only keep: it makes looking things up easy to learn.
- Keep writes about as hard on ordinary text as on the facts. Nothing is gated: the facts survive because the text
  is written elsewhere.

**Token-level** (the same model code, 0.5M weights, 8 CPU minutes: facts [name value], random filler, then [? name]).
Trained on gaps up to 64, each batch from an empty store; guessing = 6%:

| Gap of filler tokens | 0 | 64 | 1,024 | 4,096 |
|---|---|---|---|---|
| Keep, loss on the answers only | 99.2% | 98.7% | 98.9% | 98.4% |
| Keep, loss on every token | 92.9% | 93.3% | 93.8% | 94.0% |
| Plain transformer, same size, story in view, 2.5x the training steps | 36% | 36% | out of view | out of view |

## What the GPU run shows

Nov 12, an H200, the base size (50M weights each), 600M tokens each, one seed (run `keep-fast`). Share of questions
answered right, 512 scenes per gap that no stream contains:

| Model | Asked about | G = 0 | 512 | 2,048 | 8,192 | 32,768 |
|---|---|---|---|---|---|---|
| Keep | plain facts | 93.8% | 95.3% | 96.0% | 96.2% | 95.3% |
| Keep | facts changed after telling | 91.7% | 95.2% | 94.0% | 92.9% | 95.2% |
| Keep | Field1 stories | 50.3% | 50.3% | 49.0% | 51.0% | 49.7% |
| Twin | plain facts | 99.9% | 99.7% | 21.4% | 20.8% | 22.1% |
| Twin | facts changed after telling | 100.0% | 100.0% | 23.8% | 23.8% | 22.6% |
| Twin | Field1 stories | 51.0% | 53.5% | 46.7% | 47.0% | 45.2% |
| Guessing | facts / Field1 | 25% / 33.4% | | | | |

| Model | Weights | Speed | Held-out loss (streams) |
|---|---|---|---|
| Keep | 50.5M | 228k tok/s | 3.148 |
| Twin | 50.3M | 566k tok/s | 3.123 |

- **The stores hold.** Keep's facts are flat from no gap to 32,768 tokens, 8 times the longest gap it trained on,
  out of the stores alone. The twin is at guessing as soon as the story is out of view.
- **The stores are a little lossy.** 95% out of the stores against 99.8% with the story in view.
- **Reasoning is the same.** Field1 is ~50% for both while the twin sees the story: the limit there is 50M weights
  and 600M tokens, not memory.
- **No gain on ordinary text yet.** Held-out loss 3.148 against 3.123: 128 tokens of attention cost more than the
  stores gave back.
- **It learned what to write, with no cost on writing.** Write strength 1.21 and 1.17 on the stories, 0.27 and 0.53
  on the ordinary text.

What follows from it: the choice between attention and stores is not needed. **Keep-wide** (`--only wide`) has the
twin's attention over the whole window and Keep's stores. If it reads 99.8% with the story in view, 95% past it,
and the twin's loss on ordinary text, it is the model to scale.

Trained plainly, Keep-wide does not get there: at 150M tokens it answered 99.5% of the facts in view and 29% past
the window (guessing 25%; Keep was at 81% at that point). Attention answers whatever is in the training window, so
no error asks for the stores there; and for a story in an earlier window the stores' state is detached, so the
read side learns and the write side does not. **Keep-wide mixed** (`--only mixed`) is the same model trained with
its attention cut to Keep's 128 tokens on half of the steps (`--narrow 0.5`), where it has to learn the stores as
Keep does, and read with the whole window.

**How much it holds at once** (`python3 -m keep.load`, the saved models, no training: a book of plain facts, every
fact about a different subject, told once, then up to 24 of them asked; 64 books per size):

| Facts in the book | 8 | 32 | 128 | 512 | 2,048 |
|---|---|---|---|---|---|
| Tokens in the book | 65 | 257 | 1,064 | 4,228 | 17,830 |
| Keep, stores alone | 98.1% | 86.2% | 59.2% | 37.6% | 30.0% |
| Twin (sees the last 1,024 tokens) | 100.0% | 99.2% | 70.5% | 34.4% | 28.3% |
| Guessing | 25% | 25% | 25% | 25% | 25% |

Keep holds a scene across any amount of unrelated text, and not a book: facts of the same kind overwrite each other,
the earliest first (128 facts: 47% for the first third of the book, 74% for the last). It trained with one scene of
3 to 10 facts open at a time, so its addresses only ever had to tell ten things apart. `--books 0.3` makes 30% of
the training scenes books of 8 to 256 facts, 8 to 24 of them asked: whether capacity is a matter of training is what
that run and `keep.load` on it say.

What is NOT shown yet: more than one seed, a larger model, a book held.

## The experiment

Both models read the same endless streams (`data.py`): a scene told once, G tokens of ordinary Opal text, then the
scene's questions. Two kinds of scene, half and half:

- **Plain facts** (`ledger.py`): "Rui keeps the coin in the cellar." Answering needs no reasoning. A miss after a gap
  is a fact that was not kept.
- **Field1 stories** (`oazy/field1`): answering needs reasoning too. This is the harder row.

The exam fixes G at 0, 512, 2,048, 8,192 and 32,768 and uses scenes no stream contains.

- The twin's score while the story is in view (G up to 512) is the ceiling. Past 1,024 it can only guess.
- Keep is examined more strictly: the questions are read as a window of their own, so at every gap its answers come
  out of the stores alone.
- Keep-wide and Keep-wide mixed are examined both ways: as a stream is read, the questions at the end of the last 1,024 tokens (a story
  still in view is seen), and out of the stores alone, as Keep is.
- Keep trains on gaps up to 4,096, and no gradient spans more than one window of 1,024. G = 32,768 is 8 times beyond
  anything it trained on.
- In training, the tokens of a question block count 4 times in the loss, for both models alike (they are ~3% of the
  stream).

What would count:

| Result | Reading |
|---|---|
| Keep's plain facts stay near the twin's ceiling out to 32,768 | The store holds at real scale. Go to the next stage. |
| High at short gaps, falling with G | It keeps, but ordinary text wears the store away. The write gate is the thing to fix. |
| At guessing everywhere | The store never started working on real text. The addressing is the thing to fix. |

## Run it

    OAZY_DATA=/workspace/data OAZY_RUNS=/workspace/runs bash runpod/keep.sh          # ~50M each, 600M tokens each
    bash runpod/keep.sh --preset big --name keep-big                                 # ~120M each, 1.5B tokens each
    bash runpod/keep.sh --only mixed                                                 # one model only (keep, full, wide, mixed); the report keeps the others
    bash runpod/keep.sh --only keep --books 0.3 --name keep-books                    # Keep trained on books of up to 256 facts (then keep.load --name keep-books)
    bash runpod/keep.sh --only report                                                # examine the saved models again
    python3 -m keep.load --name keep                                                 # how much it holds at once: books of 8 to 2,048 facts, the saved models, no training
    python3 -m keep.mute --name keep                                                 # which store carries the facts: the saved model's exam with stores muted
    bash runpod/keep.sh --only keep --stores 6 --name keep-one                       # Keep with one store, after block 6
    python3 -m keep.tests                                                            # CPU, about a minute, no data
    python3 -m keep.speedcheck                                                       # GPU, five minutes, no data: where a training step goes

While it trains it prints the loss on the question blocks ("asked": the questions' own wording is part of it, so
read it next to the twin's), a short exam at 25, 50 and 75%, and the report at the end (`<runs>/keep/report.md`, `results.json`). The report also gives Keep's write strength on the stories and on the
ordinary text: a model that learned what is worth keeping writes hard on the first and little on the second.

## Speed

Measured on an H200 at the base size (batch 32 x 1,024 tokens), Nov 12:

| | Tokens per second |
|---|---|
| The first run | 195k |
| Sparse store gradient, window attention, compiled blocks | 238k |
| The store's loop as one graph node (measured beside a training run) | 257k |
| The loss in slices of 4,096 (beside a training run) | 285k |
| The twin (beside a training run, slices of 1,024) | 378k |

`python3 -m keep.speedcheck` takes one part of a step away at a time. Beside a training run, with the loop as one
node, a Keep step of 128 ms was: the blocks 47 ms, the stores 42 ms, the scores over the 49,152 words 37 ms. The
twin's step was 87 ms, half of it the scores over the words, which every model of this size pays.

Two more levers that were measured and are not the default, because each changes something: `--chunk 128` (the
stores at 27 ms, 291k; writes land every 128 tokens) and `--batch 128` (367k for Keep, 460k for the twin; a quarter
of the steps).

What was changed, each checked in `tests.py` to give the same numbers as before:

- **The store touches only its rows.** Under autograd, `S.scatter_add(...)` copied the whole table (16,384 x 64 per
  row of the batch) once per chunk on the way in and several times on the way back. Now the table is written in
  place, and on the way back one gradient table is filled by the reads and looked up by the writes that came
  before them. A chunk moves 512 rows, whatever the table's size.
- **Attention is computed over the window only** (`near`): blocks of 128 queries against 256 keys, not a mask over
  1,024 x 1,024. At 1,024 tokens this is no faster than flash attention over everything (47 ms against 42); it
  matters when the training window grows, because its cost per token does not.
- **The blocks are compiled** (`torch.compile`, as in Bee6; `--compile 0` turns it off; if it fails the run says so
  and goes on plain).
- **The store's loop is one node of the graph** (`_Loop` in `model.py`), its way back written by hand. The loop is
  32 sequential passes per window (2 stores x 16 chunks), and left to autograd each pass was some fifty small
  kernels and as many graph nodes: launches and bookkeeping, not arithmetic (4 times larger chunks took the stores
  from 69 to 26 ms). Now a pass is about twenty launches: the stores went from 69 to 42 ms.
- **The loss in slices of 4,096 tokens**, not 1,024: 12 ms less a step, the same numbers.

What bounds it: at 1,024 tokens Keep can come close to the twin's tokens per second, not past it. What it has to
win on is the tokens it needs, and a cost per token that does not grow with the context.

Each progress line of a run gives the speed since the line before and the peak memory.

## Files

| File | What |
|---|---|
| `model.py` | `Keep`, `KeepConfig`, `Store` (address, read, delta-rule write) |
| `ledger.py` | the plain-fact scenes |
| `data.py` | the streams and the exam sets |
| `run.py` | trains the models, examines them, writes the report |
| `speedcheck.py` | tokens per second and memory on this machine |
| `mute.py` | the saved model examined with stores muted |
| `load.py` | the saved models told books of 8 to 2,048 facts: how much the stores hold at once |
| `tests.py` | the checks |
