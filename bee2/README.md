# Bee0.2B v2: Bee0.2B with SmolLM2-360M as its mentor

**Next-word model**, exactly like Bee0.2B and GPT-2: it reads text and guesses the very next token.

**What changes, and only this:** the mentor (the bigger model whose full guesses Bee learns from in
the first half of training) is **SmolLM2-360M** (Hugging Face, 2024) instead of OpenAI's GPT-2 large
(2019). Everything else is Bee0.2B's: the body (18 Memtrix memory layers + 6 gated attention layers,
3:1), Muon, the WSD schedule, the 1,024 → 2,048 windows, the FineWeb-Edu + DCLM + conversations mix,
524k tokens per step. So Bee0.2B against Bee0.2B v2 measures one thing: the teacher.

| mentor | size | HellaSwag (published) | reading cost per token | trained on |
|---|---|---|---|---|
| GPT-2 large (Bee0.2B) | 774M | ~45% | 1x | 40 GB of 2019 Reddit links |
| **SmolLM2-360M (v2)** | 362M | **~54%** | **~0.5x** | 4T tokens of modern web, code, maths, conversations |

A smarter teacher at half the price. Its one condition: a student and its mentor must cut text into
the same pieces, or their guesses cannot be compared piece by piece. So v2 reads **SmolLM2's
tokenizer** (49,152 pieces) instead of GPT-2's (50,257). Its word table is 0.9M numbers smaller:
**203.3M parameters** (Bee0.2B: 204.2M).

## The same text, re-cut

`python3 -m oazy.retok` turns the prepared GPT-2 token files back into text, document by document
(exact: GPT-2's pieces are bytes), and cuts each document again with SmolLM2's tokenizer into
`<data>/smol/`. Nothing is downloaded again, and v2 reads exactly Bee0.2B's documents, including
the same held-out ones. By default it trains on the **same amount of text** as Bee0.2B's 5B GPT-2
tokens (5B x the token ratio, printed at the start; SmolLM2 usually needs a few % fewer tokens).

## Keeping the comparison fair

A loss "per token" depends on how big the tokens are, so v2's raw loss is not directly comparable
with the GPT-2-tokenizer models. Three things fix that:

- **The dashboard's loss is "per GPT-2 token".** v2's own loss is multiplied by its token ratio on
  the same text (SmolLM2 tokens per GPT-2 token). This is exact arithmetic: the total surprise over
  a text is the same number however the text is cut, so it is just re-divided by GPT-2's count. Its
  raw numbers are kept as `val_loss_own` / `loss_own`.
- **Bits per byte** (`val_bpb`) is now logged for every model: the surprise per byte of text, which
  does not care about tokenizers at all.
- **HellaSwag per letter** (`acc_char`, what lm-evaluation-harness calls acc_norm) is now logged
  next to the usual per-token score. The per-token one stays on the charts (same method for
  everyone); the per-letter one is the tokenizer-free head-to-head. To re-score any finished model
  both ways: `python3 -m oazy.hellaswag --ckpt /workspace/oazy_runs/bee/ckpt.pt`.

## The mentor only early (the time saver)

As with Bee0.2B, the mentor's share of the lesson fades from 60% to 0 by **halfway**, and from then on
it is not run at all. That is the "mentor for the first part only" trick, and it is the default.
With SmolLM2-360M the first half is roughly 1.4x faster than Bee0.2B's (a rough estimate:
`--bench` measures it), so the whole run should be about 2-3 hours shorter. To save more, shorten
it: `--mentor-until 0.3`. To try a stronger teacher: `--mentor smollm2-1.7b` (~68% HellaSwag,
but ~2x GPT-2 large's cost).

## Run it (RunPod)

See `docs/RUNPOD.md`, section "Bee0.2B v2". In short:

```bash
nice -n 10 python3 -m oazy.retok                         # once: re-cut the text (CPU; fine while Bee0.2B trains)
python3 -m bee2.tests --device cuda
bash runpod/train.sh bee2 --preset big
python3 -m bee2.chat --port 8002
```

## Files

`model.py` (Bee0.2B's model with SmolLM2's vocabulary), `chat.py` (Bee0.2B's chat app, opening
`runs/bee2`), `tests.py`. Shared: `oazy/tok.py` (the two tokenizers), `oazy/retok.py` (re-cutting
the text), `ember/mentor.py` (SmolLM2 mentors).
