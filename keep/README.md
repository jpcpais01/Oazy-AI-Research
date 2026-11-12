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

| | Twin (plain transformer) | Keep |
|---|---|---|
| Attention reaches | the whole window, 1,024 tokens | 128 tokens |
| Older text | gone | read back out of the stores |
| Cost per token as text grows | grows with the context | constant |
| State carried from window to window | none | the stores (2 x 16,384 rows x 64) |

Two choices, each made because the version without it failed in a CPU pilot (Nov 11 and 12):

- **Sparse write.** A write touches 8 rows of 16,384, so text about other things lands on other rows. A dense
  delta-rule memory lost a fact within ~4,000 unrelated tokens; a sparse one held 32 facts across 16,384
  (`/mnt/project-files/reads-by-learning/`).
- **Write what came next, under the address of the place before.** With the address of a token used for its own
  write, a question could only find its answer after the layers below had learned to build matching keys, and they
  never got a signal to. Stored as "what followed", a place that comes back finds its continuation from the first
  training step.

The token-level pilot (the real model code, 0.5M weights, 8 CPU minutes, `/mnt/project-files/reads-by-learning/`):
facts [name value], filler, then [? name] and the value is asked. Trained on gaps up to 64 tokens, each batch from
an empty store. Chance is 6%.

| Gap of filler tokens | 0 | 64 | 1,024 | 4,096 |
|---|---|---|---|---|
| Keep, loss on the answers (attention reaches 8 tokens, answers from the stores alone) | 99.2% | 98.7% | 98.9% | 98.4% |
| Keep, loss on every token | 92.9% | 93.3% | 93.8% | 94.0% |
| Plain transformer, same size, story in view, 2.5x the training steps | 36% | 36% | out of view | out of view |

Keep was at 97% after 300 training steps. Its write strength fell from 1.0 to about 0.2 by itself: it learned to
write little, with no penalty on writing. One seed each; the table is in `night3-results.txt` there.

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
    bash runpod/keep.sh --preset big --name keep-big                                 # ~110M each, 1.5B tokens each
    bash runpod/keep.sh --only report                                                # examine the saved models again
    python3 -m keep.tests                                                            # CPU, about a minute, no data

While it trains it prints the loss on the asked tokens ("asked": this is the number that falls when a model starts
to answer), a short exam at 25, 50 and 75%, and the report at the end (`<runs>/keep/report.md`, `results.json`). The report also gives Keep's write strength on the stories and on the
ordinary text: a model that learned what is worth keeping writes hard on the first and little on the second.

Not verified on a GPU: speed and memory. Keep's training keeps copies of each store per 64-token chunk for the
gradient (roughly 10 GB at the base size, 60 GB at `big`). If memory runs out: `--batch 16`.

## Files

| File | What |
|---|---|
| `model.py` | `Keep`, `KeepConfig`, `Store` (address, read, delta-rule write) |
| `ledger.py` | the plain-fact scenes |
| `data.py` | the streams and the exam sets |
| `run.py` | trains both models, examines them, writes the report |
| `tests.py` | the checks |
