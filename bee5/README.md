# Bee5-200M: the best plain 200M transformer we know how to build

**Next-word model**, exactly like GPT-2 and every Bee: it reads text and guesses the very next token, and it is scored
the same way (held-out loss per GPT-2 token, HellaSwag, ARC-Easy, PIQA, LAMBADA).

João's ask (Nov 9): *"200M parameters, modern transformer, no different ideas. But super optimized with all the
tricks, as if we are trying to beat the speed run. The best 200M parameter base model."*

So Bee5 is Bee4's body, rebuilt with every trick from the public NanoGPT speedrun
([modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt), checked up to record #92, PR #360, Aug 30 2026, and
its 350M "medium" track) that has been proven there **and** carries over to our setting: a 200M model, SmolLM2's
pieces, one B200, 10,000M tokens (about 20x longer than a speedrun). Tricks that only pay inside a 3-minute run, or
that are really new architectures, are left out, and each is listed below with the reason.

| | Bee4 | Bee5 |
|---|---|---|
| width x layers | 1,024 x 12 | 1,024 x 12 |
| attention | 16 query / 4 key-value heads of 64, RoPE, a full output gate | the same heads; QK-norm, RoPE on half of each head, a small per-head gate, keys shifted one word on layers 4, 8, 12 |
| thinking part | SwiGLU 2,816 | ReLU² 3,840 (the same weight count as SwiGLU 2,560) |
| extra paths | none | 2 value tables, U-net skips, the input mixed into every layer, one-word look-back, back-out |
| scores | capped at ±30 | squashed into (0, 23), asymmetric |
| optimizer | Muon + AdamW | NorMuon (Polar Express, per-neuron step, cautious decay, momentum warm-up) + Adam groups |
| early lesson | next word | next 3 words, fading to the next word by 1/6 of the run |
| growing depth | 6 -> 9 -> 12 layers | none (full depth from the start) |
| parameters | 198.2M | **201.3M**, everything counted: word table 50.3M, value tables 25.2M, layers 125.8M |
| text | Tagus20 | Sado |

## Every trick: kept or dropped, and why

"Record N" is the speedrun's record table (README of modded-nanogpt).

### Kept: the body

| trick | where it comes from | what it does, plainly |
|---|---|---|
| **QK-norm** | record 5 (Bee4 had it) | questions and keys are normalised before they meet, so attention scores cannot blow up. Learned gains per head let the model choose how sharp each head is (instead of a hand-set attention scale) |
| **ReLU²** instead of SwiGLU | record 5; Primer (So et al. 2021) | the MLP squares its positive part. Same weight count as Bee4-style SwiGLU 2,560, one product fewer. *Measured both ways*: `--set mlp=swiglu` is the race's control |
| **zero-init output projections** | record 5 | every layer's last product starts at zero, so at the start each layer is a no-op and the model is "just the word table"; layers switch on as they learn |
| **half-truncated RoPE** | record 17 | half of each head's pairs rotate with position, the other half carry content only (no position); the slowest turns once per ~6,400 words |
| **value tables** (value embeddings), gated | records 14-17, 55, 63 | a second, small word table added straight into the attention values of layers 1-2 and 11-12 (U-net pattern). It lets late layers recall "which word was this" without carrying it through the stream. At key-value width (256), so two tables cost 25M parameters, not 100M |
| **U-net skips**, gated | records 11, 45, 55 | layer i's output is added into layer 13-i's input with a learned gate: the second half of the network can reuse the first half's work directly |
| **x0 mixing** (residual and input lambdas) | records 9, 40 | each layer reads a learned mix of the running stream and the original (normed) word vectors |
| **smear** (one-word look-back) | record 34 | each word's vector gets a gated share of the previous word's before layer 1: bigrams for free |
| **sparse attention gate** | record 28 | each head can say "nothing to add here" from 16 numbers of its input. Replaces Bee4's full 1,024 x 1,024 output gate (1M weights a layer, and a product) |
| **partial key offset** | record 49 | on layers 4, 8, 12 the position-free half of each key holds the *previous* word's: one layer can then do "find where this word came before, copy what followed" (induction) |
| **back-out** | record 40, medium track 15 | the head reads the stream minus a learned share of the stream at 2/3 depth: what early layers wrote only for later context is taken back out |
| **asymmetric score cap** 23·sigmoid((s+5)/7.5) | records 18, 19, 54 | scores are squashed into (0, 23) instead of ±30; the speedrun found the asymmetric shape best. `--set cap=tanh softcap=15` is the symmetric one |
| **tied word table** | records 51-53 (re-tied) | the input table is the score table; std 0.005 start |

### Kept: training

| trick | where | plainly |
|---|---|---|
| **NorMuon** | records 41-42 (Li et al. 2025) | after Muon straightens the update, each neuron's step is divided by its own running size (Adam-like, per neuron); the total size stays Muon's, so Bee4's Muon rate carries over |
| **Polar Express** | record 38 (Amsel et al. 2025) | a better 5-step recipe for straightening (each step has its own coefficients); same cost as Newton-Schulz, closer to exact |
| **cautious weight decay** | records 43, 50 | decay only where the update already pulls a weight towards zero |
| **momentum warm-up** 0.85 -> 0.95 | record 9 | the first 300 steps use shorter momentum |
| **Adam groups** | speedrun's layout | word table betas (0.8, 0.95); value tables and the learned scalars 10x the rate; gates at the rate; eps 1e-10 (record 20) |
| **3-next-word lesson early** (multi-token prediction) | record 53, medium track 16 | for the first 1/12 of the run each position also learns the 2nd and 3rd next words (weights 1, 0.5, 0.25 -> 0), then the 2nd only, then from 1/6 the next word alone. Shares the same scores, so it costs ~10% of a pass early and nothing after. Held-out is always the next word alone |
| **batch and window schedules** | records 46, 72 | Bee4 already has both: 262k -> 524k tokens a step over the first 5%, windows 512 -> 1,024 -> 2,048 |
| **WSD, cool-down over the last 30%** | records 26 (0.45), medium (0.7) | Bee4's; the speedrun's long cool-downs suit its short runs; 20-30% is the long-run optimum (Hägele et al. 2024) |
| **the mentor** (SmolLM2-360M, 1 pass in 4 -> 1 in 10) | ours (Bee4) | kept as a switch: `--mentor none` turns it off, and the race measures whether it is worth its ~10% cost |
| **loss in slices** | ours (Bee4) | extended to the 3-word lesson and the new cap; tested equal to the slow way |
| **compile, cuDNN attention, bf16** | ours | as Bee4; the speed check tries cuDNN and autotuned compile and keeps what is 2%+ faster |

### Dropped, and why

| trick | why not |
|---|---|
| **FP8 matrix products** | Bee4's check on the B200 (Nov 7): torchao's FP8 was *slower* (445k vs 475k tok/s) and moved the gradients (cosine 0.953). At width 1,024 the products are too small for FP8's per-step scaling to pay; the speedrun gets its FP8 wins from hand-written kernels with fixed scales tuned for 8 H100s. `--fp8 1` still works and is re-checked by Frog's FP8 check; a fused FP8 score head is the one worth building next |
| **long/short sliding windows** | a speed trick for FlashAttention-3 on H100 (Hopper only). On the B200 our options are PyTorch's fused kernel (no windows) or flex attention, which was ~40% slower in Bee4's check. At 2,048 words attention is about a quarter of the work, so even a perfect window would save ~15% |
| **YaRN** | it stretches positions when the speedrun widens its attention windows mid-run and checks at longer windows than it trained on; Bee5 trains its last 70% at full length and is scored at 1,024 |
| **document-only attention, BOS-aligned windows** | Bee4 measured document masking at ~40% slower on the B200; aligning windows to document starts would change the shared text reader every model uses (~1% in the speedrun). A good follow-up |
| **untie the table at 2/3** | it ends with two 50M tables: 250M parameters, not 200M, and a bigger phone file |
| **hashed bigram/trigram tables** (records 62, 83, 92) | the speedrun's biggest recent win, but it is a huge lookup table (65 *billion* parameters in record 92): not a 200M model |
| **MUDD connections, XSA, DC-MHA, paired heads, hyper-connections** (records 58, 73, 81-85) | new architecture ideas, proven only at 124M and 0.5B tokens; João asked for no new ideas. The best candidates for a "Bee5+" |
| **sampled softmax, ANVIL optimizer, depth reduction, mixed-width QK** (record 92) | one-week-old record, tuned for a 40-second run on 8 GPUs; not yet tested by anyone at longer training |
| **prefix-token lesson** (record 88) | small (1%), unclear outside the speedrun |
| **tail weight averaging (EMA)** | WSD's cool-down to zero already gives most of what averaging gives (Hägele et al. 2024) |
| **learning rate floor 0.1** | the speedrun's runs are too short to reach zero usefully; long runs do best decaying fully |
| **speedrun hyperparameters** (75x value-table rate, weight decay ∝ rate²) | tuned for ~5,000 steps; Bee5 runs ~19,000, so Bee4's long-run rates stay and only the forms change |
| **growing depth** (ours, Bee4) | it saved ~8% of the work but clashes with the U-net skips and value tables, which tie layer 1 to layer 12 |
| **Adam every other step, CUDA graphs, custom kernels, 8-GPU sharding** | multi-GPU or kernel engineering; CUDA graphs failed with our training step in Bee4's check |
| **data selection** (PR #150) | left off until its A/B on Plain-100M lands (`--select` is refused for now) |

## Run it (on the B200 pod, network volume, local copy of Sado)

    git pull
    python3 -m bee5.tests --device cuda              # ~3 min
    python3 -m bee5.speedcheck                        # ~10-15 min: tok/s, chip use, memory vs Bee4; prints the train line

The 0.5B race against Bee4 and Plain-100M at equal tokens (same Sado, same recipe; ~20 min each):

    OAZY_DATA=/root/data bash runpod/train.sh bee5 --preset big --tokens 0.5e9 --micro 32 --data-sub sado --name bee5-race
    OAZY_DATA=/root/data bash runpod/train.sh bee4 --preset big --tokens 0.5e9 --micro 32 --data-sub sado --name bee4-race
    # Plain-100M's race is already on the dashboard (plain100-race, 3.385)

Optional controls (each another ~20 min): `--mentor none --name bee5-nomentor` (is the mentor worth it?) and
`--set mlp=swiglu --name bee5-swiglu` (ReLU² vs SwiGLU at equal size).

The full run (~10,000M tokens; the speed check prints the exact line and hours):

    OAZY_DATA=/root/data bash runpod/train.sh bee5 --preset big --tokens 10e9 --micro 32

Speedrun mode (the records' own text and target; optional):

    python3 -m bee5.speedrun --prepare 10            # FineWeb, GPT-2 pieces, 1,000M tokens (~2 GB)
    python3 -m bee5.speedrun                         # minutes and tokens to reach 3.28 on one B200

## Files

- `bee5/model.py`: `Bee5Config`, `Bee5` (the body, the paths, reading word by word for chat).
- `bee5/loss.py`: the loss in slices with the multi-word lesson, the caps and the mentor.
- `bee5/optim.py`: NorMuon (Polar Express, per-neuron step, cautious decay, momentum warm-up) and the Adam groups.
- `bee5/speedcheck.py`: speed vs Bee4 on the GPU, the switches, the whole-run estimate.
- `bee5/speedrun.py`: speedrun mode.
- `bee5/tests.py`: self-tests. `bee5/chat.py`: the chat app.
- `oazy/train.py`: `RECIPE["bee5"]` (Bee4's, reading Sado), the presets, the optimizer hook.
