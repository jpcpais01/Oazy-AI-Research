# Fern2: Fern, raised the fast way

*Working name.* **Not a next-word model.** Fern2 writes exactly like Fern: 16 words at a time, in 5
rounds, coarse to fine (where the passage is going, then the middle, then the details). Its loss
is the same exact per-word loss, so it compares directly with GPT-2's. What changes is its size and
how it is raised, using Ember's recipe plus one new idea.

```
                     Fern                    Fern2
size                 198M (18 + 6 blocks)    119.2M (9 reader + 3 unfolder), under GPT-2's 124M
growing              no                      reader 3 -> 6 -> 9, unfolder 1 -> 2 -> 3 (at 10% and 25%)
reader's own lesson  no                      also guesses the next word of everything it reads
mentor               no                      GPT-2 medium teaches the reader, fading out by 50%
optimizer            AdamW                   Muon on every weight grid of both halves + AdamW
```

## How a next-word teacher can teach a model that isn't next-word

GPT-2 medium can only answer one question: *"what is the next word, given everything before it?"*.
Fern2 has two halves, and they ask different questions:

- **The reader** reads the finished text, word by word. Its question *is* "what comes next?", so
  the mentor's full guess for every word ("70% cat, 20% dog, ...") is an exact lesson for it. Fern2's
  reader now makes that guess too (through the same word table, so no new weights), and learns it
  from both the real text and the mentor.
- **The unfolder** fills gaps: "what is word 8, knowing the past and word 16?". A next-word model
  cannot answer that without trying every possible way of filling the gap, which is far too
  expensive. Feeding it the mentor's one-sided guess anyway would teach it to ignore the words it
  already knows on the right. So the unfolder learns from the real text only, standing on a reader
  that has been taught to read like GPT-2 medium.

```
lesson = Fern's per-word loss
       + 0.5 x [ (1 - m) x reader's next-word loss  +  m x how far the reader's guess is from the mentor's ]
m (the mentor's share) = 0.6 at the start, fading to 0 by halfway; then the reader keeps its own small lesson.
```

The curves, validation loss and HellaSwag are Fern's per-word loss only, exactly as for Fern.
The reader's next-word loss is logged as `reader_loss` in `metrics.jsonl`.

Known better but too costly for now: two-sided lessons for the last round from the mentor
(scoring each candidate word with the word after it, ~5x the mentor's cost), and training on text
the mentor wrote itself (the standard trick for non-next-word translators, hours of generation).

## Honest expectations

Each idea is known to give roughly 1.3-2x on its own; being 40% smaller makes every word ~1.7x
cheaper than Fern. The reader's own lesson is new here and untested at scale, and none of this
has run on a GPU yet, so the first job is to see Fern2's loss curve next to Fern's and GPT-2's.

## Run it

```bash
python3 -m fern2.tests --device cuda
bash runpod/train.sh fern2 --tokens 2e9
python3 -m fern.chat --run fern2 --port 8002
```

Measure each idea alone (same tokens, separate run folders):

```bash
python3 -m oazy.train --model fern2 --name fern2-no-mentor --mentor none
python3 -m oazy.train --model fern2 --name fern2-no-grow   --set grow=9:0 unfold_grow=3:0
python3 -m oazy.train --model fern2 --name fern2-no-reader --set reader_weight=0 --mentor none
python3 -m oazy.train --model fern2 --name fern2-adamw     --opt adamw
```
