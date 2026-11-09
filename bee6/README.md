# Bee6-200M: Bee5 made fast

**Next-word model**, exactly like GPT-2 and every Bee: it reads text and guesses the very next token, and it is scored
the same way (held-out loss per GPT-2 token, HellaSwag, ARC-Easy, PIQA, LAMBADA, LogRes).

João's ask (Nov 9): *"a 200M parameter model training at 3M tokens a second"*, then *"build idea 2 3 4 5. 200M params.
Call it Bee6."*

The honest wall first. One word through a dense 200M model, forwards and backwards, is about 1.2 billion operations.
3M words a second would be 3.6 million billion a second, and one B200 tops out around 2.25 in bf16. So a plain dense
200M model cannot reach 3M on one B200 even at perfect use of the chip. Bee6 attacks it from all three sides: use the
chip better (idea 2), cheaper maths (idea 3), and skip work (ideas 4 and 5). The lab's guess when stacked is about
2.3-2.6M tokens a second. **Nothing here has been timed on the B200 yet**; `python3 -m bee6.speedcheck` measures
each idea on its own row, and the race says what each one costs in quality.

| | Bee5 | Bee6 (defaults) |
|---|---|---|
| body | 12 x 1,024, every proven speedrun trick (bee5/README.md) | the same |
| q, k, v | three products | **one** product (the same weights, the same maths) |
| loss slices | 1,024 words | **4,096** words (the same numbers, fewer and bigger products) |
| layer products | bf16 | **FP8** forward and input gradient; the weight gradient stays bf16 |
| layers 3, 5, 7, 9, 11 | every word | **the top 25% of words** each (Mixture of Depths), the rest skip the layer |
| growth | none | **bloom** 4x512 -> 6x768 -> 9x1,024 -> 12x1,024, a switch, off by default |
| parameters | 201.3M | 202.0M (+0.66M for the five routers and their helpers) |
| recipe | Bee4's on Sado | **Bee5's exactly**: Sado, SmolLM2-360M mentor, NorMuon + Adam groups, windows 512 -> 2,048, batch ramp, WSD 30% |

## The four ideas, and their switches

Every idea is a `--set` switch, so its speed and its quality effect can be measured alone.

### Idea 2: use the GPU better

- **One q/k/v product** (`fuse_qkv=1`). Attention's three input products become one big one (1,024 -> 1,536). The
  weights stay three separate grids, so NorMuon straightens each on its own as before. Tested equal to three.
- **Bigger loss slices** (`loss_chunk=4096`). The last step scores 49,152 word pieces per word; it is done in slices so
  the full table never exists. Bee5 used slices of 1,024 words: 64 slices a pass, each ending in a 49,152 x 1,024
  product added into a 200 MB gradient. 4,096-word slices do that 16 times a pass instead of 64. A few GB more
  memory at the peak (the B200 has 180). Tested equal.
- **Fixed shapes everywhere, no padding.** Mixture of Depths uses a fixed number of words per routed layer, the
  windows pack whole documents (no padding), and Bee6 compiles with fixed shapes (one compiled version per window
  length and bloom stage, never a slow general one). There was no padding to remove: the 49,152 word pieces already
  divide by 128.
- **CUDA graphs** (`graphs=1`). The GPU replays the whole recorded pass instead of being handed ~1,000 small jobs one by
  one. They crashed with our step before (`compile_mode=reduce-overhead`); Bee6 marks each pass's start, which is what
  that crash needed. Off by default: the speed check turns it on only if it runs and is 2%+ faster.
- **A profile with a kernel list**: `python3 -m bee6.profile` splits a pass into layers / word table + loss / optimizer,
  gives the chip use, and lists GPU time by kind of kernel and the slowest kernels by name, so the next bottleneck has
  a name.

The lab's aim is 25% -> 40% of the chip. How much of that these reach is the speed check's number.

### Idea 3: real FP8 (`fp8=1`)

The B200 multiplies 8-bit numbers twice as fast as bf16. Bee4 tried it with torchao (Nov 7) and it was slower and
moved the gradients. Bee6 has its own small FP8 product (`bee6/fp8.py`), the NanoGPT speedrun's way:

- only the layers' big products: q/k/v (fused), the attention output, the MLP's up and down (~70% of the arithmetic);
- everything delicate stays bf16 or 32-bit: the norms, the word table and score head, the value tables, every gate,
  the router and its helper, the loss;
- one scale per tensor per pass from its largest value: inputs and weights in e4m3, gradients in e5m2. Scales are
  worked out fresh every pass, so there is nothing to store and nothing to grow when Bee6 blooms;
- mode 1 (default): the forward product and the input's gradient in FP8, the weight's gradient in bf16 (it is what the
  optimizer reads). `fp8=2` puts the weight gradient in FP8 too; `fp8=0` is bf16;
- **a startup check**: before training, the same batch goes through the model in bf16 and in FP8. If the loss is more
  than 2% apart or the gradients less than 0.9 alike, or FP8 does not run on this GPU, Bee6 says so and trains in bf16.

### Idea 4: Mixture of Depths (`mod_layers=2,4,6,8,10 mod_cap=0.25`)

"The", "of" and commas do not need 12 layers of thought. In every other layer (3, 5, 7, 9 and 11, counting from 1), a
router scores every word and only the top 25% of each window go through the layer: attention among themselves at their
own positions, then the MLP. Their change is multiplied by 2 x sigmoid(score), which is how the router learns which
words to pick. The other 75% hop past unchanged. Those five layers do a quarter of the work. Raposo et al. (2024) found
about equal quality at much less compute.

The catch, and how Bee6 handles it: "top 25% of the window" depends on later words, so it cannot be used when reading
(the model would peek at the future). As in the paper, each routed layer has a small helper (1,024 -> 128 -> 1) that
learns, on the side, to guess from each word alone whether the router would pick it (`mod_aux=0.01`; the helper reads a
detached copy, so it never pulls on the model). **Everything that reads uses the helper's guess**: the held-out loss,
HellaSwag, LogRes, the samples and chat. So every score is honest and causal (tested: changing later words never
changes earlier scores). The trainer's log shows `mod_agree`, how often the helper agrees with the router. Routed
layers never shift their keys (the key-offset layers are 4, 8, 12).

`mod_eval=topk` reads with the router's own top 25% instead (it looks ahead: for diagnosis only). `mod_layers=` turns
the idea off; `mod_cap=0.125` is the paper's best setting, worth a race.

### Idea 5: bloom (`bloom=on`, off by default)

João's Seed schedule at Bee6's sizes, 20/20/20/40 of the tokens:

| stage | body | share of the run |
|---|---|---|
| 1 | 4 layers of 512 (8 heads) | 0-20% |
| 2 | 6 layers of 768 (12 heads) | 20-40% |
| 3 | 9 layers of 1,024 | 40-60% |
| 4 | 12 layers of 1,024 = plain Bee6 | 60-100% |

Each growth keeps the model's answers exactly (tested to 1e-13 in 64-bit, on the scores, through all three growths):

- **wider**: the stream gets new numbers that start at 0. Bee6's norms have no gains, so the only effect of the zeros
  is to change the norm's size by a fixed factor; every product reading a norm is divided by it and every product
  writing into the stream is multiplied by it. New heads and MLP neurons start with random weights in and zero out.
- **deeper**: new layers go in the middle (front half of the old layers stays at the front, back half at the end), so
  every U-net skip still joins the same two old layers and the value tables stay on the first two and last two layers.
  A new layer starts with zero output weights, an exact no-op; its new skip starts at 0. (With bloom on, Bee6 uses
  plain skip weights instead of Bee5's sigmoid ones, so a skip can start at exactly 0 and still learn.)
- **the router** grows with the width like every product that reads a norm; new layers in the every-other pattern get a
  router, old ones keep theirs. **FP8 scales** have nothing to grow (fresh each pass). The back-out stays on its layer.

After each growth the trainer makes a new optimizer (the weights are new, bigger tensors) and the model recompiles for
its new shape. A checkpoint taken after a growth loads into a fresh model (it grows to the checkpoint's stage first).
Plain Bee6 is measured first; bloom is its own race.

Seed (PR #165) landed while Bee6 was built and grows the same way (middle insertion, plain skip weights, function
kept). Its growth is tied to its library body, so Bee6 has its own, which also grows the routers. One difference: Seed
carries the optimizer's memory over a growth, Bee6 starts a fresh optimizer (a possible follow-up).

## Run it (on the B200 pod, network volume, local copy of Sado)

    git pull
    python3 -m bee6.tests --device cuda       # ~3 min: also real FP8 and the compiled step on the GPU
    python3 -m bee6.speedcheck                 # ~15-20 min: one row per idea, the whole-run speed, the train line
    python3 -m bee6.profile                    # optional: where a pass's time goes, kernel by kernel

The race against Bee5, one line (Bee5 500M tokens, Bee6 the same 500M, Bee6 again for exactly Bee5's training time,
then the table; finished runs are kept, so it can be re-run safely):

    tmux new-window -t oazy -n race6 "OAZY_DATA=/root/data bash runpod/bee6_race.sh; bash"

Then the same with bloom on (reuses the Bee5 run):

    tmux new-window -t oazy -n race6b "OAZY_DATA=/root/data bash runpod/bee6_race.sh bloom; bash"

The table again any time: `python3 -m bee6.race` (or `--bee6 bee6bloom`). It shows tokens, training time, speed,
held-out loss, HellaSwag and LogRes side by side, and a verdict at equal tokens and at equal hours. At 0.5B tokens a
held-out gap under 0.02 is a tie, to confirm at ~2B.

To measure one idea's quality alone, add `--set` to a race, for example `--set mod_layers= --name bee6-nomod` (Bee6
without Mixture of Depths) or `--set fp8=0 --name bee6-bf16`.

The full run (the speed check prints the exact line and hours):

    OAZY_DATA=/root/data bash runpod/train.sh bee6 --preset big --tokens 10e9 --micro 32

## Files

- `bee6/model.py`: `Bee6Config`, `Bee6` (Bee5's body with the fused, FP8 and routed layers).
- `bee6/fp8.py`: the FP8 product and the startup check.
- `bee6/bloom.py`: the bloom plan and the function-preserving growth.
- `bee6/speedcheck.py`: one row per idea, the whole-run speed. `bee6/profile.py`: the kernel list.
- `bee6/race.py`: the race table. `runpod/bee6_race.sh`: the race in one line.
- `bee6/tests.py`: self-tests. `bee6/chat.py`: the chat app.
- `oazy/train.py`: `RECIPE["bee6"]` (Bee5's), the presets, the startup-check and bloom hooks.
