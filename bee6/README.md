# Bee6-200M: Bee5 made fast

**Next-word model**, exactly like GPT-2 and every Bee: it reads text and guesses the very next token, and it is scored
the same way (held-out loss per GPT-2 token, HellaSwag, ARC-Easy, PIQA, LAMBADA, LogT).

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
HellaSwag, LogT, the samples and chat. So every score is honest and causal (tested: changing later words never
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

## Sparse MLP (`sparse_mlp=0.10`, off by default)

João's idea (Nov 9): like a mixture of experts, but each word lights up only the ~10% of the network that matters for
it, picked by a dimension-reducing step. Bee6 stays a plain NEXT-WORD model; only how much of each MLP runs changes.

- **Blocks**: each layer's MLP (1,024 -> 3,840 -> 1,024, ReLU²) is cut into 30 blocks of 128 neurons (`sparse_block`).
  A block is on or off as a whole, so the GPU still multiplies whole tiles.
- **The picker**: the word's vector (1,024 numbers) is squeezed to 128 (`sparse_rank`), and those 128 score the 30
  blocks. The top 10% (3 blocks) run; 27 are skipped. It reads only that word's vector: causal, always.
- **Learning to pick**: the first 10% of the tokens (`sparse_from`) the whole MLP runs (exactly Bee6) and the picker
  learns to guess each word's busiest blocks (activity x the block's output strength), a small lesson on the side
  (`sparse_aux=0.01`, on a copy of the vector, so it never pulls on the model). Then the switch flips for good (it is
  saved in the checkpoint). After it, 1 word in 16 (`sparse_check`) still gets the full up-product without gradient, so
  the picker keeps its lesson and the log shows `sparse_recall`: the share of each word's truly top blocks it ran
  (chance = 10%). A gate that is exactly 1 but carries the gradient tells the picker whether a block it ran helped.
- **Picking where a block stands out**: blocks are ranked by how far each one's score sits above its usual for
  that word (its mean and spread over words: the pass's own while learning, running averages when reading, like
  batch norm, so reading stays word by word). Without it a block busy for most words became every word's pick
  in the tests, and 2 of 3 picks overflowed.
- **Balancing**: on top, DeepSeek-V3's bias balancing, as in Coral (`sparse_balance=0.01` per step): busy blocks are picked a
  little less, idle ones a little more, so every block is used and learns. `sparse_uneven` = busiest block / average.
- **On the GPU**: every block gets a fixed-size queue (`sparse_capacity=1.25` x its fair share, static shapes for
  torch.compile), the words are gathered in, all 30 blocks run as one batched product (bf16) and the results go back.
  A word that finds a queue full loses that block (`sparse_dropped`). Reading a batch uses queues long enough for all.
- **Writing a word** (chat, CPU, phone) takes the 3 picked blocks' weights directly: ~10% of each MLP read per word.

What to expect: the MLP's arithmetic drops 10x, but moving words into queues and back is not free, so the GPU gain is
smaller (the lab's guess was 1.5-2x, overlapping with Mixture of Depths). `python3 -m bee6.speedcheck` measures it
(rows: Bee6 default, + sparse MLP, the same without Mixture of Depths, blocks of 64, and the MLP alone, dense against
sparse). On the CPU: `python3 -m bee6.sparsebench` (dense against sparse writing speed, 4 and 1 threads).

The race (plain Bee6 at 500M tokens, `bee6-race`, kept if it finished on the same GPU; Bee6 + sparse at the same tokens;
Bee6 + sparse for plain Bee6's hours; `nomod` adds sparse without Mixture of Depths; the speed check runs first):

    tmux new-window -t oazy -n sparse "OAZY_DATA=/root/data bash runpod/bee6_sparse.sh; bash"
    tmux new-window -t oazy -n sparse "OAZY_DATA=/root/data bash runpod/bee6_sparse.sh nomod; bash"

The table again: `python3 -m bee6.sparse_race` (also `<runs>/bee6-sparse-race.md`).

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
held-out loss, HellaSwag and LogT side by side, and a verdict at equal tokens and at equal hours. At 0.5B tokens a
held-out gap under 0.02 is a tie, to confirm at ~2B.

To measure one idea's quality alone, add `--set` to a race, for example `--set mod_layers= --name bee6-nomod` (Bee6
without Mixture of Depths) or `--set fp8=0 --name bee6-bf16`.

The full run (the speed check prints the exact line and hours):

    OAZY_DATA=/root/data bash runpod/train.sh bee6 --preset big --tokens 10e9 --micro 32

## World practice (`--world 0.10`, off by default)

João's question (Nov 9): give a normal transformer ~10% of its reading as the worldfirst world, where it watches a scene
change and answers questions about it. Is it better on the benchmarks than the same model without? Bee6 stays a plain
NEXT-WORD model: the episodes are just more text, learned word by word, answers included (like the worldfirst rival).

    There is a line. Vasco is right behind Cora. ... Simao and Zoe swap places. Is Vasco ahead of Simao? Answer: no.

- **What**: `worldfirst/world.py`'s World 2, fifteen families (sums, lines, boxes, family, falling, machines, cups,
  minds, walks, carry, time, cause, wants, count, and the mixed house; a run started on World 1 stays on it), practice
  levels only, made fresh on the fly by 3 CPU helper processes (`--world-workers`), never stored, never a
  secret exam's. Cut into SmolLM2's pieces, `<|endoftext|>` between episodes, packed into the same windows.
- **How much**: `--world w` = 3w of the passes for the first 20% of the run, then w/2 to the end (cool-down included),
  w on average. `--world 0.10` = 30% then 5%; `--world 0.05` (half) = 15% then 2.5%; 0 = off. Whole passes go to the
  world, spread evenly. The token count includes the world's, so a world run reads 10% less Sado at the same compute.
- **Clean**: the mentor never reads a world pass; data selection and bloom are refused with it; the training-loss line
  stays the text's (`world_loss_own` is logged apart); held-out loss is Sado's held-out text only.
- **The exam**: the world's secret exams (longer, deeper, new mixes, more moves) and fresh practised scenes, asked as the
  same plain text: the model picks each answer word among the rival's candidates by its own probabilities.
  `python3 -m oazy.worldmix exam --run bee6-race bee6-world10` (any SmolLM2-piece checkpoint, the plain one too).

The race (plain Bee6 = `bee6-race`, the same run `bee6_race.sh` makes, kept if it already finished; then +world 10%;
`half` adds +world 5%; then every run's world exam and one table):

    tmux new-window -t oazy -n world6 "OAZY_DATA=/root/data bash runpod/bee6_world.sh; bash"
    tmux new-window -t oazy -n world6 "OAZY_DATA=/root/data bash runpod/bee6_world.sh half; bash"

The table again: `python3 -m bee6.world_race` (also saved as `<runs>/bee6-world-race.md`). If the world run wins,
confirm at ~2B tokens before `world=0.10` joins the recipe. Tests: `python3 -m oazy.worldmix_tests`. To see what the
model reads: `python3 -m oazy.worldmix show`.

## Bee6-MoE: fine-grained experts (`--model bee6moe`, 872M, 202M per word)

João, Nov 10: keep what Bee6 pays for on every word and make the thinking part ~8x bigger, for runs of 10B tokens and
more. Still a NEXT-WORD model. Each layer's MLP (one ReLU² MLP of 3,840) becomes 1 shared expert + 80 routed experts,
all ReLU² of 384; the router (sigmoid scores, DeepSeek-V3 bias balancing, a 1e-4 balance loss) picks 9 per word, choosing where each expert stands out for the word (its score centred on its usual score, the sparse MLP's lesson: without it the tiny run put every word on one expert). So a word
runs 10 x 384 = 3,840 neurons, Bee6's amount, out of 31,104. Attention, the word table, the value tables, MoD and FP8
attention are Bee6's; the experts run in bf16 in fixed queues (1.25x a fair share, as Coral); reading drops nothing.
NorMuon straightens every expert's grid on its own. Recipe = Bee6's.

|                 | Bee6-200M | Bee6-MoE |
|-----------------|-----------|----------|
| parameters      | 202.0M    | 872M (765M in experts) |
| at work per word| 202.0M    | 202M |
| speed (estimate)| B200 ~500k tok/s | ~330-400k tok/s (routing, spare queue room, bf16 experts) |

    python3 -m bee6.moe_tests --device cuda     # tests + Bee6 vs Bee6-MoE speed at full size, ~3 min
    python3 -m oazy.train --model bee6moe --preset big --micro 32 --tokens 10e9 --data-sub emerald --name bee6moe-emerald-10b

The log's `experts_uneven` (busiest expert / average, ~1.0-1.5 is healthy), `experts_dropped` (share of picks that found
a full queue, want < 5%) and `experts_idle` (experts no word picked in a step, want 0) say whether the experts are used.
If memory runs out at micro 32, use `--micro 16` (the batch stays the same, more accumulation).

## Files

- `bee6/model.py`: `Bee6Config`, `Bee6` (Bee5's body with the fused, FP8 and routed layers).
- `bee6/fp8.py`: the FP8 product and the startup check.
- `bee6/bloom.py`: the bloom plan and the function-preserving growth.
- `bee6/speedcheck.py`: one row per idea, the whole-run speed. `bee6/profile.py`: the kernel list.
- `bee6/race.py`: the race table. `runpod/bee6_race.sh`: the race in one line.
- `bee6/sparse.py`: the sparse MLP. `bee6/sparse_race.py` + `runpod/bee6_sparse.sh`: its race.
  `bee6/sparsebench.py`: CPU writing speed, dense against sparse.
- `bee6/experts.py`: the fine-grained experts. `bee6/moe.py`: Bee6-MoE. `bee6/moe_tests.py`: its tests.
- `bee6/tests.py`: self-tests. `bee6/chat.py`: the chat app.
- `oazy/worldmix.py`: world practice (the stream, the share, the exam); `oazy/worldmix_tests.py`.
  `bee6/world_race.py` + `runpod/bee6_world.sh`: the world race.
- `oazy/train.py`: `RECIPE["bee6"]` (Bee5's), the presets, the startup-check and bloom hooks.
