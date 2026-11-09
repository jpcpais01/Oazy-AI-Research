# Seed: a small core that grows, and a big library of facts

**Next-word model**, a plain one, like GPT-2 and every Bee: it reads text and guesses the very next token, and it is
scored the same way (held-out loss per GPT-2 token, HellaSwag, ARC-Easy, PIQA, LAMBADA, LogRes).

João's design from the Special Lab (Nov 9). A normal model spends its weights both on remembering things and on working
things out. Seed splits the two: the facts go into a library that is cheap to read, so the small core is free to
learn rules. And the core starts tiny and grows ("Seed & Bloom"), so it spends the early run passing over the patterns
many times, fast.

| part | what it is | size |
|---|---|---|
| **core** | a dense transformer with Bee5's tricks; 12 layers of 768 once grown, 12 query / 6 key-value heads of 64, ReLU² 2,688 | 70.8M (+3.9M for the two library readers) |
| **library** | a product-key memory: 1,152 x 1,152 = **1,327,104 rows** of 768 numbers, one table read by two layers of the core | 1,020M |
| **word table** | SmolLM2's 49,152 pieces x 768, also the score head (tied) | 37.7M |
| value tables | Bee5's two small word tables added into attention (layers 1-2 and 11-12) | 37.7M (look-ups) |
| **all of it** | | **1,171M**, about **114M at work for each word** |

There are no experts. The library does that job ("many specialists, a few at work") more simply.

## The library

For each word, a reading layer turns its vector into a question (4 heads of 256 numbers, batch-normed). Each question
is split in two halves. Each half is compared with 1,152 half-keys, the best 16 of each half are paired up, and the
best 16 pairs are the rows read: 4 x 16 = 64 rows out of 1.3M, mixed by how well they fit (a softmax), then passed
through a gate and an output projection into the stream (Lample et al. 2019; Meta's "Memory Layers at Scale", 2024).

- Reading 64 rows costs almost nothing; comparing with the half-keys costs ~1M multiplies. That is why 1B weights fit.
- It is **full size from day one** and never changes when the core grows (its rows have their own fixed width, 768).
- Only the rows read get trained: a **row-wise Adam** moves a row only in a step where it was read, so a rarely used
  fact is kept between its uses (no momentum drift, no weight decay).
- It is read after the MLP of layers 5 and 8 of the grown core. Both are seed layers, so it is read from the first step.
- The probe `[library]` reports, at each check: how many rows were read at least once, how concentrated the reads are,
  and the held-out loss **with the library closed** (what it is worth).

## Seed & Bloom: how it grows

| share of the run | core | the library |
|---|---|---|
| first 20% | 4 layers x 384 | the same |
| next 20% | 6 layers x 512 | the same |
| next 20% | 9 layers x 640 | the same |
| last 40% | 12 layers x 768 | the same |

**Each growth keeps the model's function exactly**, so the loss does not jump (the tests check the scores before and
after agree to rounding, after real training). How:

- **Wider.** The stream gets new numbers that start at 0. Our norms have no learned gain, so a stream [x, 0] of d2
  numbers normalises to sqrt(d2/d) times what x did. Every weight that reads the stream is divided by that factor,
  every weight that writes into it is multiplied by it, and the word table is divided by it (it is both the input and
  the score head). New heads, new MLP neurons and the new numbers read the old ones through fresh random weights, but
  everything new *writes* zero at first, so nothing changes until learning moves it (Net2Net, Chen et al. 2016;
  masked structural growth, Yao et al. 2023).
- **Deeper.** New layers copy their neighbour's "reading" weights (questions, keys, values, MLP input) and start with
  their outputs at zero, as Bee5 starts every layer: a no-op on the first step that already knows what to look at.
  They go in symmetrically, just inside the first and last layers and at the middle, so the U-net skips (layer i to
  layer n-1-i) stay paired. A skip pair with a new layer starts at 0.
- **Roles.** Each layer's role (value tables, shifted keys, the back-out, the library readers) is fixed by where it will
  sit in the grown core, on Bee5's 12-layer pattern. The bloom just reveals the layers in turn.
- **The optimizer's memory** (momentum, Adam's averages, NorMuon's per-neuron sizes, Grokfast's average) is carried
  into each bigger weight, padded with zeros and scaled as its gradient now is.

## How it learns

- **Grokfast on the core only** (Lee et al. 2024). Before each step the slow average of each core grid's gradient
  (alpha 0.98) is added back twice (lambda 2). The steady push that leads to general rules gets louder than the
  batch-to-batch noise. The library learns normally: facts should be picked up fast.
- **Extra weight decay on the core only**: 0.2 (twice Bee5's), so the core prefers small, simple solutions.
- `--set grokfast=0` turns **both** off (Bee5's NorMuon and decay): the race's control.
- Everything else is **Bee5's recipe** (João's rule: the newest best model's settings): NorMuon (Polar Express,
  per-neuron step, cautious decay, momentum warm-up), Bee5's Adam groups, windows 512 -> 1,024 -> 2,048, the batch
  ramp, WSD with the cool-down over the last 30%, the early 3-next-word lesson, bf16, the loss in slices, the
  SmolLM2-360M mentor (1 pass in 4, then 1 in 10), Sado with its faith texts.
- **Sampled softmax: off. Data selection: off** (it joins only if its A/B wins). **FP8: a switch**, `--fp8 1`
  (it is checked before it is kept, and new layers born in a growth get it too).

Kept from Bee5: QK-norm, ReLU², zero-start outputs, half-truncated RoPE, value tables, U-net skips, x0 mixing, smear,
sparse attention gate, key offset (layers 4, 8, 12 of the grown core), back-out, the asymmetric score cap, the tied
table. Changed: the x0 weights travel with their layer, and a skip's weight is a plain number (Bee5's sigmoid form
could not start a new pair at exactly 0).

## Commands (on the B200 pod)

    python3 -m seed.tests --device cuda          # ~2 min: CPU checks + bf16, compile, a growth on the GPU
    python3 -m seed.speedcheck                    # tokens/s of each stage, the run's average, hours and cost

1. **Smoke test, 0.5B**: nothing breaks, each growth keeps the loss smooth (`[grow]` lines print each stage's speed)

        OAZY_DATA=/root/data bash runpod/train.sh seed --preset big --tokens 0.5e9 --micro 32 --name seed-smoke

2. **Side test: the tiny core alone, longer.** It stays 4 x 384 (with the whole library) all run; watch `[logres]`
   for a late jump

        OAZY_DATA=/root/data bash runpod/train.sh seed --preset big --tokens 5e9 --micro 32 --set hold=1 --name seed-hold

3. **The race: equal hours with Bee5, Grokfast on and off** (90 minutes each, about 2B tokens for Bee5)

        OAZY_DATA=/root/data bash runpod/train.sh bee5 --preset big --minutes 90 --micro 32 --name bee5-90m
        OAZY_DATA=/root/data bash runpod/train.sh seed --preset big --minutes 90 --micro 32 --name seed-90m
        OAZY_DATA=/root/data bash runpod/train.sh seed --preset big --minutes 90 --micro 32 --set grokfast=0 --name seed-nogf-90m

4. **The full run** (pick the length from the speed check: `python3 -m seed.speedcheck --tokens 40e9` prints its hours
   and cost; the Lab's guess was 40-50B tokens for ~$100)

        OAZY_DATA=/root/data bash runpod/train.sh seed --preset big --tokens 40e9 --micro 32

`python3 -m seed.chat --port 8002` talks to it; `python3 -m oazy.report --run seed` writes its report.

## Honest limits

- It will know like a big model but reason like a ~100M one. Grokking on real, messy text is a hope, not a proven
  thing. The sign we hope for is a late jump on LogRes, not just a lower loss.
- Product-key memories can crowd onto a few rows. The questions are batch-normed for that (Lample's fix), and
  `[library]` shows how many rows are read.
- GPU speed is **not measured yet**: the library's 1B weights mean ~16 GB of optimizer state and a 4 GB gradient each
  pass, and the reading is memory-bound. The Lab's guess was 3-4M tokens/s early and ~0.8M fully grown; the speed
  check gives the real numbers.
