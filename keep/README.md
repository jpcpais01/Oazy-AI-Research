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

## Written from the top: reading again as thinking

Keep holds what it read. It does not reason any deeper than its twin, and it cannot: a transformer is feed-forward.
What its last layer works out at one token reaches the first layers of a later token only as a word that was said,
so the number of steps it can take on a question is the number of its layers. A store that is read and written at
one layer does not change that: one lookup per store and per pass, however long the model reads.

`KeepConfig(feed=True)` changes one wire. The store is still read where it stands, low in the stack, but it is
written from the **top**: at token t, under the address the reading layer made at token t - 1, goes what the last
layer says at t. Now the first layers of every later token read what the whole stack concluded before. Every further
token read is one more turn of the whole stack, and **reading a story again is thinking about it**: the number of
steps is set by the reading time, not by the layers and not by the training.

The pilot (`/mnt/project-files/clock/fb.py`, `fb2.py`; CPU, 0.46M weights, 4 layers, attention over 8 tokens, one
store of 256 rows): facts `name what .` in random order, where `what` is a value or another name whose value it then
has; the story is read R times; then `? name` is asked. A chain of depth d in the worst order needs d readings.
Training: depth 1 to 3, read depth or depth + 1 times. Share right (%), guessing among the values said is ~40%:

| Depth | Model | read 1x | 2x | 3x | 4x | 6x | 8x | 12x | 16x |
|---|---|---|---|---|---|---|---|---|---|
| 3 (trained) | written from the top | 43 | 63 | 96 | 100 | 100 | 100 | 100 | |
| 3 | control: written where it is read | 63 | 51 | 56 | 54 | 54 | 50 | 58 | 55 |
| 4 | written from the top | 56 | 64 | 79 | 97 | 100 | 100 | 100 | |
| 6 | written from the top | 52 | 57 | 73 | 80 | 98 | 100 | 98 | |
| 8 | written from the top | 47 | 63 | 75 | 76 | 94 | 100 | 99 | |
| 8 | control: written where it is read | 55 | 69 | 65 | 49 | 64 | 55 | 48 | 50 |
| 12 | written from the top (third seed) | 36 | 45 | 67 | 68 | 81 | 88 | 95 | 91 |
| 12 | control: written where it is read | 49 | 48 | 48 | 51 | 48 | 50 | 49 | 53 |

The rows "written from the top" at depth 3 to 8 are `keep/model.py` itself (`TASK=chains python3 fb2.py feed 1200`);
the pilot's own class gave the same on three seeds (depth 8 at 8 readings: 100, 100, 93).

- It was trained on chains of at most 3 and on at most 4 readings. Depth 8 and 12 are followed when it reads 8 and
  12 times: the steps come from the reading, the weights are the same.
- The control is the same code with the store written from the layer it is read at. From depth 2 on it stays at
  guessing however often it reads. Keep as it was (store after block 2) does the same.
- The sharpest control (`clock/fb3.py`, Keep as it was with its writes landing at every token): with ONE store it
  learns depth 1 and guesses from depth 2 on (48 to 62%). With TWO stores it learns the depths it trains on (depth 1,
  2, 3 at 100%) and stays flat past them: 26 to 36% at depth 4, 6 and 8 at 1 to 12 readings alike. Two stores are a
  fixed solution of three steps; more reading buys nothing. Only the store written from the top turns further
  readings into further depth.
- Silent tokens instead of reading again (`fb3.py`, THINK=1: the story once, k silent tokens after the question,
  writes at every token): inside the trained range the fed model's depth follows k (depth 3: 21, 64, 96% at k = 1, 2,
  3), but it goes no deeper than trained and falls apart at more silent tokens than it trained with. Reading again
  does not: what is already in the store writes nothing.
- Reading more than needed costs little: 16 readings of a depth-1 chain are still right; the third seed loses a few
  points at 16 readings of the deepest chains.
- The twin (4 layers, everything in view) had not learned depth 1 in 20 CPU minutes. That is a twin that did not
  converge, not evidence about transformers.

`KeepConfig(point=J)`: the top may file its write under the address of any of the J tokens before it, not only the
last one (it starts at the last one). A conclusion that is only known at the end of a sentence ("the lamp is on when
the buzzer is on": the lamp's state is known after the buzzer was looked up) has to be filed under its subject, which
stood at the sentence's start.

What is NOT shown: anything on real text, anything at size, a converged plain baseline, a second kind of task done
to the end (logic circuits, the skeleton of Field1's "switches", are learned more slowly: `fb2.py`). The wire is
the one of the Feedback Transformer (Fan et al., 2020), and looped language models (Ouro, 2025) buy depth with passes
too; they report that running more passes than trained makes them worse. What is different here: the way back goes
through an addressed store that is rewritten by its own error (a conclusion that is already there writes nothing),
training stays parallel inside a chunk, and the depth reached is 4 times the depth trained.

Circuits (and, or, not, same over switches and machines, statements in random order, the token-level skeleton of
Field1's switches; `clock/fb2.py`, one seed each, trained to depth 3, guessing 54 to 58%), share right at depth 8:

| sentence order | `point` | 1 reading | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|
| subject last | 0 | 61 | 86 | 98 | 100 | 100 |
| subject first | 7 | 54 | 82 | 96 | 99 | 100 |
| mixed | 7 | 57 | 71 | 79 | 86 | 91 |
| mixed | 0 | 58 | 67 | 78 | 79 | 87 |

Three things were needed before circuits trained at all: the switches are asked too (a lookup is learned before
computing), the chunk is small next to a scene, and a subject that stands first needs the pointer.

Real Field1 text at 1M weights and 12M tokens on a CPU (switches): the fed model, Keep and the twin all at the floor
(54 to 56% against 51%): too small to learn Field1's language, so no evidence either way.

`python3 -m keep.think` is the experiment on real text and at size. By default its own family `circuits`: the
circuits above in bare sentences with the real tokenizer ("The radio is on if the black switch or the white switch is
on. If the radio is off, the kettle is on."), every part asked, trained to depth 3 and told depth + 1 or + 2 times,
examined to depth 8 at 1 to 12 tellings. Attention reaches 32 tokens (about three sentences), so a scene is not in
view and what is known of it has to go through the stores; the twin sees everything. `--families switches,rules` is
the same on the two Field1 families that are chains of rules told in any order (switches by depth, rules by hops;
others can be named), each scene told 1 to 4 times. Gnosis (300M, 1.8B tokens) guessed on switches at every depth.

**A telling to a chunk (`--align 96`).** A fed step with chunks of 16 is 32 passes through the stack one after the
other (some 240,000 small operations at the base size, whatever the batch), which is why the fed model trains about
ten times slower than Keep on a GPU. With `--align C` every telling is padded on its left to C tokens, so is the
question block, and the fed model's chunk is C: a telling goes through the stack in ONE parallel pass, what its top
worked out lands in the stores when the telling ends, and the next telling reads it. A step is then 6 passes (44,000
operations, 5.4 times fewer) over windows of 6 x C tokens. One more telling is exactly one more step of reasoning for
the whole scene, in whatever order its sentences stand, so depth d asks for d + 1 tellings. Token pilot
(`clock/fb4.py`, 0.46M weights, trained to depth 3, 45 CPU minutes, two seeds each): it learns in as many steps, and
the exam is a staircase. Subject last, no pointer: depth 4 = 73-80% at 4 tellings and 98-100% at 6; depth 6 = 84% at
6 and 100% at 8; depth 8 = 84-86% at 8 and 98-100% at 12. Sentences of both kinds with the pointer: depth 8 = 82-88%
at 12 tellings (91-92% with chunks of 8). The exam's scenes are deeper, so longer: its chunk is as long as its longest
telling asks, and the fed model reads a scene in one go.

**Finding the loop.** On the GPU the fed model found the loop in one run of four (Nov 12: c2 yes; the same with
another seed, and both runs with `--align`, no). Three options, all off by default, for the step it has to
discover (where to file a conclusion: in "The lamp is on if the pump and the fan are on." the top knows the answer
at the last word and has to file it under "lamp", 11 tokens back, while it starts 57% sure of the token before):
`--point-bias 0` starts the pointer alike on all 16 tokens before; `--ramp 0.95` is the token pilots' curriculum
(depth 1 only, the next depth when 95% of the yes/no answers are right); `--share strength` lets a chunk's writes
share a row by their strength, so that the many held-back writes of a whole telling weaken no real one.
`--group 16` (with `--align`) lets a telling's writes land 16 tokens at a time, in order, each group on what the
groups before left: the stack still takes the telling in one parallel pass, and the stores are written as they are
with chunks of 16 (the layout of the one run that found the loop), where a later write corrects an earlier one
instead of being averaged with it.

**The pull write** (`--pull`, `KeepConfig(pull=True)`, off by default). Looked at inside, a small model that does not
find the loop files a stated fact under its subject but never a derived conclusion (the weight of its pointer on the
subject: 0.00 to 0.02, against 0.52 in a model that loops): the one step nothing rewards until the loop works. The
pull write turns the write round. Each token's place takes its value from the final vectors of the 16 tokens after it
(a softmax over those sources, each counted by its strength) and is written once, under its own address, instead of
each token choosing one place to write under. A place whose sources lie in two chunks is written by each for its
share. On the token test that shows the lottery (sentences with the subject 6 to 7 tokens before the conclusion, 55
minutes a run) the pull write gave, in 3 runs: one loop (depth d at 97 to 100% after d + 1 readings up to depth 3,
depth 6 at 97% after 12, trained to depth 3), one loop half formed (depth 3 at 86 to 92%), one run with the first
step in part (77%). An earlier version of it, 5 runs: 2 loops, 1 half formed, 2 with the first step in part. No run
learned nothing. The pointing write, 6 runs: one loop, and five that learned no step at all. With seeds made
reproducible, four more runs: one clean loop, two that never file a fact, and one that answers right only after an odd
number of tellings (a write may weigh up to 2; `KeepConfig(strong=1.0)` caps it at 1 and that seed then forms most of a
loop). All 13 runs: 4 loops, 3 dead. **On the GPU (round r3, four runs, 300M tokens) the pull write gave no clean
loop:** one run close to it (depth 4 at 96% after 6 tellings, depth 8 at 82% after 12), one of two steps, one of one
step, one without a step. Three of the four file facts after one telling (two of six with the pointing write), so the
first hurdle is lower; the loop is as rare as before. In all, 14 fed runs on the GPU: 2 clean loops.

**Feed the one that sprouted** (`--from RUN`, the family `newrules`). The loop is found by few runs, but it has to be
found once. `--from c13` starts a run from the trained model of the run `c13` (built as that run built it; `--align` is
taken from it unless given). The family `newrules` is circuits with two more rules, "exactly one of A and B is on" and
"A and B are not both on", which training shows only on machines of depth 1 (two switches); its exam asks a machine
with a new rule at depth 1 to 8 ("new rule at depth d"), next to the old rules. On the token test (two models that
loop, 25 minutes more training with two new operators at depth 1 only): the new operators are right at 100% at depth
d after d + 1 tellings up to depth 4, at 96 to 100% at depth 6 and 82 to 96% at depth 8 after 12 tellings, as the old
ones are; after 5 minutes they already stand at 94 to 99% at depth 4. So a model that loops keeps the loop when it is
taught more, and what it is taught shallow it uses deep. `bash runpod/round.sh r4` asks the same of the GPU's models
(c13 and c2 loop, p2 almost, c12 does not: the control); its table has the new rule at depth 1, 4 and 8.
`bash runpod/round.sh r5` feeds the three runs of round r3 that learned real steps but no clean loop (p2, p3, p4) 300M
more tokens of the same circuits, to see whether a sprout becomes a loop when it is trained on.

**When has it read enough?** (`python3 -m keep.settle --name c13`, runs trained with `--align`.) A model that loops
works a scene out one step a telling and writes the results into the stores; once every value is in place, a further
telling writes the same values and the stores stand still. `keep.settle` measures, on the exam's circuits, how much
the stores change from one telling to the next, and tries the rule "stop reading when they changed by less than 5%"
(`--bar`), each scene on its own: the tellings it uses, the share right when it stops, the share right after 12.
On the token test (two models that loop) the rule scores what 12 tellings score at every depth and uses 5 tellings
at depth 1, 6 at depth 4 and 7.5 to 8.4 at depth 8: the model is never told the depth, and reads longer when the
problem is deeper. Not measured on the GPU's models yet. It writes `<runs>/<name>/settle.md`.

**A round of runs, unattended.** `bash runpod/round.sh r2` trains the runs listed in `runpod/rounds/r2.txt` (a name, a
colon, the options of `keep.think`) two at a time, a new one only while the GPU has memory to spare, and keeps ONE
table in `<runs>/round-r2.md`: per run, depth 0 after one telling, depth 4 and depth 8 by tellings, and a verdict (LOOP,
partly, fixed steps, lookup). A run already examined is not trained again; a run that dies of memory is tried once
more. `python3 -m keep.round r1 --table` starts nothing and prints the table of runs started by hand.

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
    bash runpod/think.sh                                                             # reasoning by reading again: feed, keep, twin on circuits in plain sentences (no data folder needed)
    bash runpod/think.sh --families switches,rules --name field                      # the same on Field1 scenes told 1 to 4 times
    bash runpod/think.sh --only feed                                                 # one of the three (feed, keep, full); they can train side by side
    bash runpod/think.sh --align 96 --name aligned                                   # a telling to a chunk: the fed model takes a telling in one pass (6 passes a step instead of 32)
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
| `model.py` | `Keep`, `KeepConfig`, `Store` (address, read, delta-rule write; `feed`: written from the top, `point`: where the top files it) |
| `ledger.py` | the plain-fact scenes |
| `data.py` | the streams and the exam sets |
| `run.py` | trains the models, examines them, writes the report |
| `speedcheck.py` | tokens per second and memory on this machine |
| `mute.py` | the saved model examined with stores muted |
| `load.py` | the saved models told books of 8 to 2,048 facts: how much the stores hold at once |
| `think.py` | reasoning by reading again: circuits in plain sentences (or Field1 scenes) told several times, the fed model, Keep and the twin, the exam by level and by tellings |
| `tests.py` | the checks |
