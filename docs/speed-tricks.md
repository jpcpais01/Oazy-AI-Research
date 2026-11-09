# Our speed-up and optimization tricks

Every trick our models use to train faster or learn more from each word, in one place. Written Nov 9 2026, from the
code on `main` and the PR history (up to #160; Seed added with its own PR).

There are two kinds of trick:

- **Training speed**: more words read per second on the same GPU. The model ends up the same, only sooner.
- **Learning efficiency**: each word teaches the model more. The speed stays about the same, but the model gets
  better for the same number of words.

João's rule (Nov 7): every new model starts from the speed and training settings of the latest best model. So most
tricks below are on for every model built since Bee4 (Coral Mini, Plain-100, Pegasus, Alpha Star, Beta, Bee5), all
of which copy `RECIPE["bee4"]` in `oazy/train.py`.

## The numbers we have measured

All on one B200, `--preset big`, micro 32.

| model | words per second | what it means |
|---|---|---|
| Bee3 (Nov 6) | 126k with the mentor, 181k alone | ~7% of the chip used. The starting point |
| Coral (Nov 6) | 241k whole run, then **252k** after the stored RoPE table (#76) | 1B experts, ~12% of the chip |
| Bee4 (Nov 7) | **475k** student alone at full depth, **~466k** for the whole run | ~28% of the chip used. 10,000M tokens in ~6 hours |
| Plain-100 / Pegasus 2 race (Nov 8) | 244k / 181k | the two shared one GPU, so not a clean number |
| Pegasus 1 | about 30% slower than Bee4 | |
| Bee5 | **not measured yet** | `python3 -m bee5.speedcheck` measures it; the race against Bee4 is the real test |
| Seed | **not measured yet** | `python3 -m seed.speedcheck` measures each bloom stage; runs print each stage's real speed at every growth |

"Share of the chip" is MFU: how much of the GPU's top arithmetic speed we really use. Big labs reach 35-50%.

## Who uses what

A quick map. ✓ = on by default, `·` = off or not there.

| trick | Coral | Bee4 | Coral Mini | Plain-100, Pegasus, Alpha Star, Beta | Bee5 | Seed |
|---|---|---|---|---|---|---|
| compile, bf16, fused loss in slices | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Muon (batched) | ✓ | ✓ | ✓ | ✓ | NorMuon | NorMuon + Grokfast (core) |
| short windows first | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| growing depth | ✓ | ✓ | ✓ | · | · | depth **and width**, function kept |
| FP8 | ✓ (attention + shared MLP) | · | · | · | · | switch (`--fp8 1`) |
| plain causal attention (no document mask) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| batch ramp | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| WSD cool-down | last 20% | last 30% | last 30% | last 30% | last 30% | last 30% |
| the mentor | 1 in 4, gone by 30% | 1 in 4 -> 1 in 10, kept | as Bee4 | as Bee4 | as Bee4 | as Bee4 |
| speedrun body tricks (value tables, U-net, ReLU², ...) | · | · | · | · | ✓ | ✓ |
| next-3-words early lesson | · | · | · | · | ✓ | ✓ |
| product-key library (sparse rows) | · | · | · | Alpha Star's small drawer | · | ✓ 1.3M rows |
| data selection `--select` | · | · | · | off until its A/B | refused until its A/B | refused until its A/B |

---

## Part 1: training speed

### torch.compile

**What it does.** PyTorch reads the whole model once and rewrites it into fewer, bigger GPU programs ("kernels"),
so the GPU spends less time starting small jobs and moving numbers around. The mentor is compiled too, and so is
each slice of the loss.

**Who.** Every model, on the GPU.

**Gain.** Not measured on its own. It is the base every other number above sits on. If compiling fails, training
says so and carries on uncompiled (slower, same results).

**Switch.** `--compile 0` turns it off. `--set compile_mode=max-autotune-no-cudagraphs` tries harder kernels;
`--set compile_mode=reduce-overhead` uses CUDA graphs (see "dropped").

### bf16 (half-size numbers)

**What it does.** The model's arithmetic runs in bfloat16, a 16-bit number format, instead of 32-bit. The GPU does
these about twice as fast and they take half the memory. The weights themselves are kept in 32-bit, so small
updates are not lost. TF32 is also turned on for any 32-bit products left.

**Who.** Every model on the GPU (`amp()` in `oazy/train.py`).

**Switch.** Always on with a GPU.

### The loss in slices (fused output layer + loss)

**What it does.** The last step scores all 49,152 word pieces for every position. Done the normal way, that score
table is huge (about 6 GB per pass for 32 x 2,048 words). We do it 1,024 words at a time: score the slice, take the
loss and the mentor's lesson, push the gradient straight back, throw the slice away. The full table never exists.
Each slice is one compiled function. Same numbers as the slow way (tested).

**Who.** All Bees since Bee3: Coral, Bee4, Coral Mini, Plain-100 and its family, Bee5 (extended to its 3-word lesson
and new score cap).

**Gain.** Mostly memory, which lets us use micro 32. In Bee3's profile the word table plus loss was 16% of a pass.

**Switch.** `--loss plain` for the slow way. `OAZY_LOSS_COMPILE=0` stops compiling the slices.

**Code.** `oazy/lmloss.py`, PRs #43 (Bee3), 759ee4c (compiled slices).

### Muon, and Muon batched

**What it does.** Muon is the optimizer for the big weight grids. Normal optimizers let the strong directions of a
grid dominate each update, so weak skills barely learn. Muon "straightens" each update so every direction moves the
same amount. In the public NanoGPT speedrun it reached GPT-2's loss with about 1.35x fewer words than AdamW. This
is a learning trick, but its cost is a speed matter, so it sits here.

**Batched** (PR #104): all grids of the same shape are straightened together in one stacked product instead of one
by one. Exactly the same update, far fewer GPU jobs.

**Who.** Every model since Ember. Bee5 uses NorMuon (Part 2).

**Gain.** Bee4's optimizer step went from **46 ms to 9 ms** a step.

**Switch.** `--opt adamw` for plain AdamW. `--muon-lr`, `--muon-wd`.

**Code.** `oazy/muon.py`, `bee5/optim.py`.

### Short windows first (context warm-up)

**What it does.** The model reads windows of 512 words for the first 15% of the run, then 1,024 until 30%, then
2,048. The words per step stay the same, so each pass simply takes more, shorter windows. Short windows are cheaper
(attention costs grow with length) and early on the model has not learned to use long context anyway. The held-out
score is always on 1,024-word windows, so every model is still judged the same way.

**Who.** Bee4 and everything built on it: Coral Mini, Plain-100, Plain-100F, Pegasus 1 and 2, Alpha Star, Beta, Bee5.
Coral reads 2,048 from the start.

**Gain.** Part of the 1.05-1.64x faster early stages Bee4's speed check measured (together with growing depth).

**Switch.** `--set ctx=2048:0` for one length all run; any plan like `--set ctx=512:0,1024:0.15,2048:0.3`.

**Code.** `BEE4_CTX` in `bee4/model.py`, PR #104.

### Growing depth

**What it does.** The model starts shallow and gets deeper: Bee4 has 6 layers for the first 10%, 9 until 25%, then
12. Each time, the top layers are copied and stacked on top, so the new layers start as working layers, not blank
ones. Early training is much cheaper per word, and the deep model starts from a good place.

**Who.** Frog, Coral (8 -> 12 -> 20), Bee4, Coral Mini. Not Plain-100, Pegasus, Alpha Star (different shapes) and not
Bee5 (its U-net skips and value tables tie layer 1 to layer 12, which clashes with copying layers).

**Gain.** About 8% of the whole run's work saved for Bee4. Its early stages ran 1.05-1.64x faster.

**Switch.** `--set grow=12:0` for full depth from the start.

**Code.** `frog/model.py` (`grow_to`), PR #70.

### Plain causal attention (no per-document mask)

**What it does.** We pack several documents into one 2,048-word window. Keeping them apart (each document only sees
itself) needs a special attention kernel. We dropped that rule: a document may see the ones before it in the window.
That lets attention use PyTorch's fastest built-in kernel. The mentor still reads each document on its own.

**Who.** Coral (PR #75) and every model since.

**Gain.** The document mask was **~40% of a pass** in Coral's B200 profile. Coral went from 137k/180k to 177k/256k
words a second (with/without the mentor).

**Switch.** `--set attn_docs=1` brings the document rule back.

### Stored RoPE table

**What it does.** RoPE tells attention where each word sits, using sine and cosine waves. The compiled pass was
working out those sines and cosines again for every number, every pass. Now they are computed once and stored (for
8,192 positions). Same numbers.

**Who.** Every RoPE model since Coral.

**Gain.** ~30 ms of a 370 ms pass (~8%). Coral 241k -> 252k words a second.

**Code.** PR #76. Always on.

### cuDNN attention

**What it does.** Asks for NVIDIA's own attention kernel for Blackwell GPUs. Same maths.

**Who.** Available to Bee4 and everything after. Off by default.

**Gain.** Within 1% on Bee4's B200 speed check, so we left it off.

**Switch.** `--set attn_backend=cudnn`. The speed checks try it and keep it only if it is 2%+ faster.

### Fixed room per expert (expert models only)

**What it does.** In Coral and Coral Mini, a router sends each word to 2 experts. In training each expert gets a
fixed number of slots (1.25x its fair share), so every shape is fixed and all experts run as one batched product the
compiler can see as one graph. A word that finds its expert full skips it (it still has the shared MLP and its other
expert); about 1% of picks. Evaluation and chat drop nothing.

**Who.** Coral, Coral Mini.

**Code.** `coral/model.py`.

### Few CPU threads

**What it does.** PyTorch starts one CPU thread per core (96 on a B200 pod). On a GPU the CPU only does small jobs,
and those idle threads spin and starve the one thread that feeds the GPU. We cap them at 4.

**Who.** Every run on a GPU (`few_cpu_threads` in `oazy/data.py`).

**Gain.** Fixed a case where the GPU was only 74% busy with ~24 cores burnt.

### Read the data from the pod's own disk

**What it does.** The trainer reads each batch straight from disk with no read-ahead. Reading from the network volume
was too slow and left the GPU waiting (~73% busy). So we copy Sado to the pod's local disk first and point the
trainer there. The data files are memory-mapped and each batch goes to the GPU through pinned memory.

**Who.** Every run on Sado.

**How.**

    mkdir -p /root/data && cp -r /workspace/oazy_data/sado /root/data/
    ln -s /workspace/oazy_data/hellaswag_val.jsonl /workspace/oazy_data/evals /root/data/
    OAZY_DATA=/root/data bash runpod/train.sh <model> ...

A background read-ahead in the trainer would remove the need for the copy. Not built yet.

### The mentor only on its own passes

**What it does.** The mentor (SmolLM2-360M) is a second model that must read the text to teach. Since Frog, it does
not teach a bit of every pass. A running tally hands it whole passes instead (1 in 4 at the start), and it reads only
those. On the others it does not run at all. The mentor is compiled, in bf16, and hands over its last hidden layer
instead of 49,152 scores, so its lesson goes through the same loss slices.

**Who.** Every model with a mentor since Frog.

**Cost.** Bee4: 101 ms per teaching pass, about 12% of passes over the run. Coral: about 3-4% of speed (it left at
30%). Bee5's README estimates ~10%.

**Switch.** `--mentor none` turns it off. `--mentor-weight`, `--mentor-until`, `--mentor-floor`, `--mentor-off` set
the schedule.

### Evaluation cost

**What it does.** During training we stop to check the model: held-out loss every 100M words (128 windows),
HellaSwag on 1,000 questions every 200M, samples every 200M, ARC-Easy, PIQA and LAMBADA (500 questions each) and LogRes
every 500M, and the full tests at the end. The time spent on checks is **not** counted in the run's clock or its
words per second, so speed numbers compare training only.

**Cost.** Not measured on its own. To lower it, use a preset with longer intervals, or fewer questions
(`hs_quick`, `evals_quick` in `PRESETS`).

### Checkpoints

Saved every 10 minutes (`--ckpt-min`), on Ctrl-C and at the end, written to a temporary file first so a crash
never leaves a broken one. A restart resumes exactly where it stopped.

### The speed checks

Not a trick but the tool that tells us which tricks pay:

- `python3 -m oazy.speedcheck --model bee4 --preset big`: measures each switch at full depth, keeps it only if it is
  2%+ faster with the same loss and gradients, picks the micro batch, times the shallow stages, the mentor and the
  optimizer, and prints the train line to use (~15-20 min).
- `python3 -m bee5.speedcheck`: the same for Bee5 against Bee4 (~10-15 min).
- `python3 -m oazy.profile --model <m> --preset big --micro 32`: where each pass's time goes, kernel by kernel.

Bigger micro batches (48, 64) were within 1% of 32 on Bee4, so 32 stays.

---

## Part 2: learning efficiency

### WSD learning rate with a long cool-down

**What it does.** The learning rate warms up, stays flat for most of the run, then falls to zero over the cool-down
("warm-up, stable, decay"). The fall follows a square-root shape (1 minus the square root). The best text mix,
heavy in chat, is fed during the cool-down. Studies of WSD find a cool-down of 20-30% of the run is about best.

**Who.** Every Bee. Coral cools down over the last 20%; Bee4 and everything since over the **last 30%** (PR #99).

**Switch.** `--decay 0.3`, `--decay-shape sqrt|linear`, `--schedule cos|wsd`.

### Batch ramp

**What it does.** The first steps use smaller batches (262k words a step), growing to the full 524k over the first
5% of the run. Early on, small batches learn more per word; later, big batches give steadier steps.

**Who.** Coral and everything since (Bee3 and Frog started at 65k).

**Switch.** `--batch-ramp 262144:0.05`, or `--batch-ramp none`.

### Muon with weight decay

Muon (Part 1) also pulls each grid a little towards zero every step (0.1, on AdamW's scale). On long runs, without it
the grids drift large and lose precision in bf16 (what Moonshot found with Kimi K2).

**Who.** Bee3 and every model since. **Switch.** `--muon-wd`.

### The mentor (distillation)

**What it does.** On a share of the passes, the model learns to match SmolLM2-360M's guesses instead of only the one
real next word. A guess spread over many likely words carries much more than one right answer, so early learning is
faster. The share falls over the run so the model ends on real text.

**Who.** Coral: 1 in 4, gone by 30%. Bee4 and everything since: 1 in 4, falling to 1 in 10 by 30%, kept, then off
over the last 10%.

**Gain.** Not measured cleanly yet. The Bee5 race can test it with `--mentor none --name bee5-nomentor`.

### Data selection (`--select 1`)

**What it does.** Each word counts by how much is still worth learning from it (the model's loss minus the
mentor's). The 70% most useful words count fully, the rest a quarter. And 15% of each pass are rereads of passages
the model was close to getting. Off for the first 2% of the run.

**Who.** Nobody by default yet. It goes into the best recipe only if it wins the A/B on Plain-100
(`ab-plain` vs `ab-select`, 0.5B words each on Sado). Bee5 refuses it until then.

**Cost.** +5.7% per step on CPU. Rereads take the place of new text.

**Code.** `oazy/select.py`, PR #150.

### Bee5's speedrun tricks

Bee5 (PR #159) takes every trick from the public NanoGPT speedrun that has been proven there and fits a 200M model
trained on 10,000M words. They are all on by default. Full details in `bee5/README.md`.

| trick | what it does, plainly |
|---|---|
| **zero-init output layers** | each layer's last product starts at zero, so at the start the layer does nothing and switches on as it learns. Training starts calm |
| **ReLU²** instead of SwiGLU | the MLP squares its positive part. Same size, one product fewer. `--set mlp=swiglu` to compare |
| **value tables** (value embeddings) | a small second word table added straight into attention in layers 1-2 and 11-12, so late layers can recall "which word was this" for free |
| **U-net skips** | layer i's output is fed, with a learned gate, into layer 13-i: the second half can reuse the first half's work |
| **x0 mixing** | every layer reads a learned mix of the running stream and the original word vectors |
| **smear** | each word gets a gated share of the previous word's vector before layer 1: word pairs for free |
| **QK-norm** | questions and keys are normalised before they meet, so attention scores cannot blow up |
| **half-truncated RoPE** | half of each head knows positions, the other half carries content only |
| **sparse attention gate** | each head can say "nothing to add here", from 16 numbers. Cheaper than Bee4's full output gate |
| **key offset** on layers 4, 8, 12 | part of each key holds the previous word, so one layer can do "find where this came before, copy what followed" |
| **back-out** | the score head subtracts a learned share of the stream at 2/3 depth: notes early layers wrote only for later context |
| **asymmetric score cap** | scores squashed into (0, 23) instead of ±30. `--set cap=tanh softcap=15` for the old shape |
| **NorMuon** | Muon, then each neuron's step divided by its own running size (Adam-like, per neuron) |
| **Polar Express** | a better 5-step recipe for Muon's straightening. Same cost, closer to exact |
| **cautious weight decay** | decay only where the update already pulls a weight towards zero |
| **momentum warm-up** | Muon's momentum goes 0.85 -> 0.95 over the first 300 steps |
| **Adam groups** | own rates for the word table, value tables, gates and small learned numbers |
| **next-3-words early lesson** | for the first 1/12 of the run each position also learns the 2nd and 3rd next words, fading to the next word alone by 1/6. ~10% extra cost early, nothing after. Held-out is always the next word alone |

**Gain.** Not measured yet. The 0.5B race against Bee4 is the test:

    OAZY_DATA=/root/data bash runpod/train.sh bee5 --preset big --tokens 0.5e9 --micro 32 --data-sub sado --name bee5-race
    OAZY_DATA=/root/data bash runpod/train.sh bee4 --preset big --tokens 0.5e9 --micro 32 --data-sub sado --name bee4-race

### Seed's tricks

Seed (`seed/README.md`) is Bee5's body and recipe, plus four things of its own. Full details in its README.

| trick | what it does, plainly |
|---|---|
| **Seed & Bloom** (growing width and depth) | the core starts 4 x 384 and is copied bigger at 20%, 40% and 60% of the run (6 x 512, 9 x 640, 12 x 768). Each growth keeps the model's function exactly: new numbers start at 0, readers and writers are rescaled to match, new layers copy their neighbour's reading weights and write zero at first. The early stages are much smaller, so they run much faster |
| **product-key library** | 1.3M rows of 768 (1.0B weights), read by two layers: each word picks 64 rows by comparing two half-questions with 1,152 half-keys each. Cheap to read; only the rows read are trained. Its questions are batch-normed so many rows stay in use |
| **row-wise Adam for the library** | a row moves only in a step where it was read (no momentum drift, no decay), so a rare fact is kept between uses |
| **Grokfast + extra decay on the core** | the slow average of the core's gradients (alpha 0.98) is added back twice, and the core's decay is 0.2 (Bee5: 0.1), to favour simple general rules. `--set grokfast=0` turns both off |

**Gain.** Not measured yet. The checks, in order: `--tokens 0.5e9 --name seed-smoke` (nothing breaks, the loss is
smooth at each growth), `--tokens 5e9 --set hold=1 --name seed-hold` (the tiny core alone: does LogRes jump late?),
then a 90-minute race against Bee5 with Grokfast on and off (`seed/README.md` has the lines).

---

## Part 3: tried and dropped

| trick | why we dropped it |
|---|---|
| **FP8 matrix products** | On the B200 with Bee4 (Nov 7) it was *slower*, 445k vs 475k words a second, and it moved the gradients (cosine 0.953). At width 1,024 the products are too small for FP8's scaling cost to pay. Off since PR #113. `--fp8 1` still works; Coral used it on attention and the shared MLP. A fused FP8 score head is the one worth building next |
| **CUDA graphs** (`compile_mode=reduce-overhead`) | crash with our training step |
| **autotuned compile, cuDNN attention, micro 48/64** | all within 1% on Bee4. Still available as switches |
| **flex attention** | used for the per-document rule (~40% of a pass, see Part 1). Pegasus tried it again for its local window (PR #142): every setting failed to compile on the B200's Triton. Pegasus runs ~15% faster without it. Retry after a torch upgrade |
| **sliding attention windows** (long/short) | a speedrun trick that needs FlashAttention-3 on H100. On the B200 we only have the fused kernel (no windows) or flex attention (slower). Attention is about a quarter of the work at 2,048, so even a perfect window would save ~15% |
| **growing depth for Bee5** | clashes with U-net skips and value tables |
| **document-only attention, document-aligned windows** | ~40% slower on the B200 |
| **hashed word-pair tables** | the speedrun's biggest recent win, but it is a giant lookup table (65 billion numbers), not a 200M model |
| **untying the word table** | ends with two 50M tables: 250M, not 200M |
| **YaRN, EMA weight averaging, learning-rate floor, speedrun hyperparameters** | they suit 3-minute runs; our runs are ~20x longer |
| **multi-GPU** | the trainer has no DDP yet. João chose one B200 for Coral (Nov 6) |
| **Memtrix memory layers** | their kernel was the slowest part of Bee3 and Frog, and their long-text advantage does not show at 2,048 words. `--set pattern=MMMA` brings them back for Coral |
| **a 1.7B mentor** | considered for Bee4; João picked the 360M as about 20% faster overall |
| **Plain-100F's next-sentence head** | a learning trick: no gain in its race. Dropped |

## Part 4: older tricks (Memtrix-era models)

These sped up models we have retired (Bee0.2B to Frog), and they still work if those models come back:

- **The memory kernel as one compiled operator** (PR #70): so the whole model compiles as one graph.
- **Document offsets once per window** (759ee4c): they were worked out in all 18 memory layers, each time stopping the GPU.
- **The hint conv in word-major layout** (PR #60): two transposes and two copies fewer per layer per pass.

## How to turn things on and off: cheat sheet

| switch | what it does |
|---|---|
| `--compile 0` | no torch.compile |
| `--fp8 1` | FP8 products |
| `--loss plain` | the full score table (slower, same numbers) |
| `--opt adamw` | AdamW instead of Muon |
| `--mentor none` | no mentor |
| `--select 1` | data selection |
| `--decay 0.3` | cool-down share |
| `--batch-ramp none` | no batch ramp |
| `--set ctx=2048:0` | no short-window warm-up |
| `--set grow=12:0` | no growing depth |
| `--set attn_docs=1` | documents kept apart |
| `--set attn_backend=cudnn` | cuDNN attention |
| `--set compile_mode=...` | `max-autotune-no-cudagraphs` or `reduce-overhead` |
| `--set mlp=swiglu` | Bee5 with SwiGLU instead of ReLU² |
| `--set grokfast=0` | Seed without Grokfast and its extra core decay |
| `--set hold=1` | Seed stays the 4 x 384 seed all run (no growing) |
| `--set bloom=12x768:0` | Seed full size from the start |
| `OAZY_DATA=/root/data` | read data from the pod's local copy |
| `OAZY_LOSS_COMPILE=0` | loss slices uncompiled |
