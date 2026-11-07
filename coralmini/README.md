# Coral Mini: Coral's experts at 200M, raised like Bee4

**Next-word model**, exactly like GPT-2, every Bee and Coral: it reads text and guesses the very next token, and it
is scored the same way (held-out loss per GPT-2 token on 1,024-token FineWeb-Edu windows, HellaSwag, ARC-Easy, PIQA,
LAMBADA).

Coral Mini is Coral made small. It has Coral's layer (gated attention, then a shared MLP plus a colony of small
"expert" MLPs, of which a router picks 2 per word) at Bee4's size (width 1,024, 12 layers, about 198M parameters).
It is trained exactly like Bee4: João's rule is that the speed and training settings always come from the latest
best model.

| | Coral | Bee4-200M | Coral Mini |
|---|---|---|---|
| width, layers | 1,024, 20 attention | 1,024, 12 attention | 1,024, 12 attention |
| attention | 16 query / 4 key-value heads of 64, RoPE | the same | the same |
| thinking part per layer | 32 experts of 448 (top 2) + shared MLP 512 | one MLP of 2,816 | **16 experts of 160 (top 2) + shared MLP 256** |
| parameters in total | 1,037M | 198.2M | **198.4M** |
| at work per word | 211M (161M without the word table) | 198.2M (all) | **115.8M** (65.5M without the word table) |
| recipe | Coral's | Bee4's | **Bee4's, exactly** |

(16 x 160 + 256 = 2,816: the experts hold exactly the parameters of Bee4's one MLP, cut into pieces. Each word uses
the shared 256 and two experts of 160, so 576 of the 2,816.)

## Plainly: what an expert is

Every layer has a look-back part (attention) and a thinking part. In Bee4 the thinking part is one MLP that every
word goes through. In Coral Mini it is cut into 16 small MLPs ("experts") plus one small shared MLP. A tiny
**router** scores the 16 experts for each word and sends the word to its best 2. So all the knowledge is still
there (198M parameters, like Bee4), but each word only pays for about 115M of work. The hope: about Bee4's
quality per word read, a good deal faster per word.

Honest caveats:
- **The experts are small (160 wide).** Small matrix products use the GPU less well than Bee4's one big MLP, and
  the router and the sorting of words to experts cost time too. So it will not be 198/115 = 1.7x faster; my guess
  is **1.1-1.4x Bee4's speed** (about 500-650k tokens/s at full depth). The speed check measures it.
- **Per word read, a sparse model learns a bit less than a dense model of the same total size.** With 2 of 16 picked
  and Bee4's total, I would guess Coral Mini lands a little under Bee4 per token (maybe 0.01-0.03 held-out loss),
  and gets ahead only if it is given the time it saves. The comparison is exactly what this run is for.

## What it keeps from Bee4 (everything)

`RECIPE["coralmini"] = RECIPE["bee4"]`, and the same big-preset numbers:

- growing depth: 6 layers for the first 10%, 9 until 25%, then 12 (each growth copies the top layers, experts
  and their balancing biases included)
- short windows first: 512 words until 15%, 1,024 until 30%, then 2,048 (held-out always on 1,024)
- the mentor, SmolLM2-360M: 1 pass in 4 at the start, 1 in 10 from 30%, kept, off over the last 10%
- the cool-down (learning rate falls, chat-heavy text) over the last 30%, the square-root shape
- bf16 (FP8 was slower on the B200 for Bee4), batched Muon with weight decay 0.1, the batch ramp from 262k,
  524k-token steps, learning rates 1.8e-3 / Muon 0.03, Tagus20, plain causal attention on the fused kernel
- Bee4's speed switches (`--set attn_backend=cudnn`, `--set compile_mode=...`) and its speed check

## What it adds (what any expert model needs, all from Coral)

- **Balancing.** The risk with experts is that the router falls in love with a few and the rest never learn.
  Each expert has a small bias added to its score only when choosing; after every step a busy expert's bias goes
  down by 0.001 and an idle one's goes up (DeepSeek-V3's way). A tiny balance loss (1e-4 x) is a safety net.
- **Fixed room per expert in training.** Each expert takes at most 1.25x its fair share of a pass, so every shape
  is fixed and the 16 experts run as one batched matrix product. A word that finds its expert full skips it (it
  keeps the shared MLP and its other expert). Evaluation and chat drop nothing.
- **Muon on every expert's grids** too (each expert straightened on its own); the router stays on AdamW.
- **The log line** shows `experts 1.05 uneven, 0.4% dropped`: the busiest expert's work over the average (1.00 is
  perfect) and the share of picks that found their expert full. The dashboard shows the same, and "M at work".

## Run it

See `docs/RUNPOD.md`, section "Coral Mini". Only when the GPU is free (not while Bee4 trains):

    python3 -m coralmini.tests --device cuda
    bash runpod/train.sh coralmini --preset big --tokens 10e9 --micro 32

## Files

- `coralmini/model.py`: `CoralMiniConfig` (Bee4's body, 16 experts of 160 + shared 256) and `CoralMini` (Coral's
  experts, Bee4's compile modes and window warmup).
- `coral/model.py`: the experts, the router, the balancing (shared with Coral).
- `oazy/train.py`: `RECIPE["coralmini"]`, `MENTOR["coralmini"]`, the presets' `batch_coralmini` / `lr_coralmini` /
  `muon_lr_coralmini`.
- `coralmini/tests.py`: self-tests (sizes, Bee4's body and recipe, the experts, balancing, Muon, growing, the speed
  check, a whole tiny run and its resume, and on the GPU the compiled step).
