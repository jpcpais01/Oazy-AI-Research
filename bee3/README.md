# Bee3: Bee_v2_0.2B's body, raised better

**Next-word model**, exactly like GPT-2, Bee0.2B and Bee_v2_0.2B: it reads text and guesses the very
next token, and it is scored the same way (held-out loss per GPT-2 token on 1,024-token FineWeb-Edu
windows, HellaSwag, and from Bee3 on also ARC-Easy, PIQA and LAMBADA).

**The body does not change.** 18 Memtrix memory layers + 6 gated attention layers (3:1), d = 768,
203M parameters, SmolLM2's 49,152 pieces, SmolLM2-360M as mentor. So Bee2 against Bee3 measures two
things only: **the text** (Tagus, seven kinds of text, `oazy/tagus.py`) and **the recipe** below.
Same length as Bee2's run: 2.5B tokens.

## What changed, and why (from Bee2's logs)

| | Bee_v2_0.2B | Bee3 | why |
|---|---|---|---|
| mentor | 60% → 0 by halfway, then gone | 60% → 25% by halfway, **kept** at 25%, fades out over the last 10% | Bee2 stalled from 1.0B to 1.5B tokens, right where the mentor left; 0.6 nats of its lesson were still unlearned |
| documents in a window | bleed into each other | **each read on its own**: attention, memory and the 4-word hint all stop at `<\|endoftext\|>`; the mentor reads with the same rule | a Wikipedia article was allowed to "continue" into a chat; every such first word was taught wrong |
| windows | 1,024 until the mentor left, then 2,048 | **2,048 from the start** | SmolLM2 reads 8,192, so there was no reason to switch mid-run |
| learning rate | 1.2e-3 AdamW / 0.02 Muon | **2.4e-3 / 0.04** (twice) | Bee2's curve was smooth and its gradients were far below the clip; the rate was never tested |
| tokens per step | 524k | 64k rising to **262k** over the first 10% | small steps early learn more per token; the ramp is where the loss falls fastest |
| Muon weight decay | none | **0.1** (decoupled, scaled like AdamW's) | the standard setting of the Muon speedruns; keeps the grids from drifting over a long run |
| cool-down | linear to 0 over the last 20% | **1 − √(1 − t)** over the last 20% | holds the rate higher for longer, a better-tested shape |
| the loss | the full 49,152-wide score table, in memory, twice | **in slices** (`oazy/lmloss.py`) | the score tables were the largest thing in memory; now neither the student's nor the mentor's exist whole |
| after training | pre-training only | **a chat fine-tune** (`oazy/sft.py`) | the model learns to answer, to stop and to wait, scored on the assistant's words only |

**The learning rate is the one unproven change.** Twice Bee2's, by decision, without a pilot. The clip
stays at 1.0, so if the loss turns noisy the fix is `--lr 1.8e-3 --muon-lr 0.03` on resume; WSD keeps
the long flat part, so a slightly-too-hot rate costs little.

## How each piece works

**No bleed between documents** (`doc_mask=1`). A training window holds pieces of several documents, one
after another. Bee3 finds every `<|endoftext|>` and (1) wipes the memory at the next word (the memory's
forget-gate is forced to 0 there; with the flash-linear-attention kernels the window is split into
documents instead, `memtrix/memory.py`), (2) lets attention look back only inside the same document
(a boolean mask, `bee/model.py`), (3) stops the 4-word hint convolution at the boundary, and (4) hands
the mentor the same mask. The self-tests check that two documents packed together read *exactly* as
they do apart.

**The loss in slices** (`oazy/lmloss.py`). For 1,024 tokens at a time: the scores against the whole
word table, the soft cap, the cross-entropy and the mentor's lesson (KL against the mentor's scores,
formed from *its* token vectors and word table the same way), and straight away their gradients.
Only the gradients for the token vectors and the word table are kept. It is exact to float precision
(`python3 -m bee3.tests` checks it against the plain way); `--loss plain` is the old path.

**The mentor the whole run.** `fade()` in `ember/mentor.py`: its share goes from 0.6 at the start to a
floor of 0.25 at halfway, stays there, and goes to 0 over the last 10% so the final cool-down is the
student's own. It costs mentor reading over the whole run instead of half; a 360M mentor reads at
roughly a third of the student's training cost.

**The batch ramp** (`--batch-ramp 65536:0.1`): tokens per step grow linearly from 64k to the preset's
262k over the first 10% of the run, by changing the number of micro-batches per step; nothing else
changes. The step count in the logs reflects it.

## Run it (RunPod)

See `docs/RUNPOD.md`, section "Bee3". In short:

```bash
cd /workspace/oazy && git pull
nice -n 10 python3 -m oazy.tagus --run-tokens 2.5e9      # once: Tagus, our dataset (CPU + download)
python3 -m oazy.evals --prepare                          # once: ARC-Easy, PIQA, LAMBADA (tiny downloads)
python3 -m bee3.tests --device cuda                      # self-tests with the real kernels and tokenizer
python3 -m oazy.profile --model bee3 --preset big        # where a step's time goes; tokens/s; MFU
bash runpod/train.sh bee3 --preset big --tokens 2.5e9    # the run (~6 h on a B200 at Bee2's speed; faster if the profile says so)
python3 -m oazy.sft --run bee3                           # afterwards: the chat fine-tune (10-20 min)
python3 -m bee3.chat --run bee3-chat --port 8002         # talk to it
```

The dashboard shows Bee3 next to Bee2 and the GPT-2s, with two new cards: the extra tests and the
held-out loss per kind of text (one slice of Tagus each).

## Speed

Bee2 ran at ~103k tokens/s with the mentor and ~139k without on a B200, which is roughly 8% of the
chip's arithmetic. `python3 -m oazy.profile` takes one micro-batch apart on the GPU (memory layers,
attention, MLPs, word table + loss, mentor, optimizer) and names the ten slowest kernels, so the
next round of speed work aims at a measured bottleneck, not a guess. The loss in slices and the mentor
handing over token vectors instead of 49k scores are the first two moves; the profile on the pod
decides the next ones. For honesty: well-tuned training of models this size reaches 35-50% of a
chip's peak, not 100%; a run in the 20-30% range would already be 3x Bee2's speed.

## Files

- `bee3/model.py`: `Bee3Config` (SmolLM2 pieces, 2,048 windows, `doc_mask=1`) and `Bee3`, Bee2's body.
- `oazy/train.py`: `RECIPE["bee3"]`, the presets' `lr_bee3` / `muon_lr_bee3` / `batch_bee3`, the new flags
  (`--muon-wd`, `--batch-ramp`, `--decay-shape`, `--loss`, `--mentor-floor`, `--mentor-off`).
- `oazy/lmloss.py`: the loss in slices. `memtrix/memory.py`, `bee/model.py`, `ember/mentor.py`: the document rule.
- `oazy/sft.py`: the chat fine-tune. `oazy/evals.py`: ARC-Easy, PIQA, LAMBADA. `oazy/profile.py`: the profiler.
- `bee3/tests.py`: self-tests (exactness of the slices, of the document rule, the recipe knobs, a whole tiny run).
- `bee3/chat.py`: the chat app (`--run bee3-chat` after the fine-tune).
