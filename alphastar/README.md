# Pegasus Alpha Star: a 100M model that thinks in loops

**Next-word model.** It reads text and guesses the very next token, and it is scored exactly like Plain-100M
(held-out loss, HellaSwag, LogT, ARC/PIQA/LAMBADA). What is new is *how much thinking* each word gets.

## The idea, in plain words

A normal model has a fixed number of layers, so every word gets the same amount of thinking. Alpha Star has three
parts, and the middle one is used again and again:

| part | layers | runs | what it does |
|---|---|---|---|
| reader | 2 | once | reads the words; its second layer asks the **fact drawer** |
| core | 4 | **N times** ("passes") | thinks: each pass refines a private thought per word. The reader's output is fed in again on every pass, so it never loses the text while thinking |
| speaker | 2 | once | turns the final thought into the next-word guess (the tied word table) |

So the model is small (104M numbers, Plain-100M has 107M) but its depth grows with N: 2 + 4 x N + 2 layers of work.
At N = 6 (the training average) it works 28 layers deep; at 16 passes, 68. This is Huginn's recipe (Geiping et al.
2025, a 3.5B looping model), at our 100M size, with two additions.

- **The fact drawer** (product keys, Lample et al. 2019; Meta's memory layers 2024). 16,384 stored vectors ("facts").
  Each word asks 4 questions; for each, the drawer finds the 32 best-fitting facts out of all 16,384 with only
  2 x 128 comparisons (a fact's key is two half-keys, so the best pairs come from the best halves), and mixes them by
  fit. A big memory at a tiny cost per word: it takes the place of one ordinary MLP. It sits in the reader, so the
  facts it fetches feed every pass of the core. (`--set drawer_at=speaker` moves it; `--set drawer=0` removes it.)
- **The next-thought lesson** (Plain-100F's head, PR #149) on the thinking itself. At each sentence end (or every 16
  words), a small shared head reads the core's thought and guesses which content words the *next sentence* will use.
  It is a side lesson during training only (`--set thought=0` turns it off).

Where the numbers are: word table 37.7M, reader 9.0M, fact drawer 14.7M, core 28.7M (incl. the pass adapter),
speaker 13.8M; the next-thought head (2.4M) is training-only and not counted.

## Training (Huginn's way)

- **A random number of passes** for each pass of the trainer: a Poisson count around a log-normal mean, mean 6, at
  most 12, at least 1. So the core learns to work at any depth.
- **A curriculum**: the mean grows from 2 to 6 over the first 30% of the run (short thinking first, while it learns
  words).
- **Gradients only through the last 2 passes**. The earlier passes run without keeping their work, so memory and the
  backward pass cost the same whatever N is. The forward cost does grow with N.
- **A random start**: each word's thought starts as noise, so the core must build the thought from the reader's output
  and can be run for any number of passes.
- The next-thought loss is taken on the thoughts of the 2 learning passes, the later one weighing more (passes 5 and 6
  of 6: 5/11 and 6/11 of 0.1). Passes without gradients would learn nothing from it, so it is not computed there while
  training; the probe below measures it after *every* pass.
- Everything else is Plain-100M's recipe exactly: SmolLM2-360M mentor (1 pass in 4, then 1 in 10, fading at the end),
  short windows first (512 -> 1,024 -> 2,048), Muon + AdamW, the batch ramp, the cool-down over the last 30%, bf16,
  524k-token steps. Choosing what to learn from (`--select 1`) works too, off by default as for every model until its A/B.

Speed: at 6 passes it does about 28 layers of work per word instead of 10, but only 12 layers keep their work for the
backward pass. Expect roughly **0.6-0.7x Plain-100M's tokens per second** (the GPU test below measures it).

## Reading

- **Held-out loss, HellaSwag, LogT, ...**: it thinks until each word's thought settles (moves less than 3% of its
  size from one pass to the next, `exit_eps`), at most 16 passes. A word that is done keeps its thought while the
  others go on. `--set test_passes=N` forces N passes.
- **Writing** (chat, samples): a fixed 8 passes per word (`gen_passes`), word by word with every pass's keys kept.

## How we judge it, including on the next thought

At every probe the log shows three `[loop]` lines (also in metrics.jsonl, kind `loop`):

1. **Does more thinking help?** Held-out loss after 1, 2, 4, 8 and 16 passes, and when it stops by itself (with the
   average number of passes it took). The bet works if the loss keeps falling past 6 passes (thinking longer than it
   was trained to helps, as in Huginn) and the adaptive stop is about as good as 16 passes but cheaper.
2. **Does it see the next thought better as it thinks?** The next-thought loss (nats per content word of the next
   sentence) after every pass 1..16, next to the blind guess (each content word by how common it is). The bet works if
   this falls pass by pass: the thinking is literally getting closer to what comes next. The gap to the blind guess is
   how much it knows about the next sentence; compare its final value with Plain-100F's `[foresight]` number.
3. **Does it use the fact drawer?** The share of the 16,384 facts picked at least once on held-out text, and how
   concentrated the picks are (half of the picks go to X% of the facts). A healthy drawer uses most facts; a collapsed
   one (the Pegasus failure mode) uses a handful.

And the race itself: held-out loss at 500M tokens against `plain100-race` (3.385) and `plain100f-race`, HellaSwag and
LogT (`python3 -m oazy.logres --run plain100-race --run plain100f-race --run alphastar-race`).

## Run it (on the B200 pod)

```
cd /workspace/oazy && git pull
python3 -m alphastar.tests --device cuda          # ~5 min: everything, plus the compiled step and the speed against Plain-100M
OAZY_DATA=/root/data bash runpod/train.sh alphastar --preset big --tokens 0.5e9 --micro 32 --data-sub sado --name alphastar-race
```

To race it without the next-thought lesson (so its effect is measured alone), the same line with
`--set thought=0 --name alphastar-nothought-race`.

If it runs out of memory, `--micro 16` (the same tokens per step, more passes per step).

## Pegasus Beta (alphastar/beta.py): it decides how long to think

Alpha Star's first race showed a lazy loop: after 2 passes the thought settled and more passes added ~0.007 to the loss.
The number of passes was a schedule we set and the stop was a rule we guessed. Pegasus Beta lets the model learn both
(PonderNet, Banino et al. 2021):

- **A stop head**: after every pass it says, for each word, how sure it is that it can stop now.
- **Its lesson**: on 2 sampled rows per trainer pass (4,096 words), the next-word loss is measured after every pass (8
  while training; only those rows think on past the practised depth, so a step costs about what Alpha Star's does); the stop head is trained on the loss it would get where it stops, on average, plus a small pull (0.01)
  towards a gentle prior (4 passes on average) so it explores early on. Stopping before a word is figured out costs a
  lot; every extra pass costs a little.
- **The depth it practises** is drawn from its own stopping habits (a running average), half mixed with the prior, so
  words it stops early on make it good early, and words it thinks long on make it good late. Nobody sets a schedule.
- **Reading**: each word stops once its chance of having stopped reaches one half, at most 16 passes.

The `[loop]` probe adds where its stop lands (the share of words stopping after pass 1, 2, 3 ...), and both models now
split loss-by-passes and passes taken into filler words (the, of, commas) and content words.

```
python3 -m alphastar.tests_beta --device cuda
OAZY_DATA=/root/data bash runpod/train.sh pegasusbeta --preset big --tokens 0.5e9 --micro 32 --data-sub sado --name beta-race
```
