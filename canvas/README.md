# Canvas-135: think in meanings, the near future all at once, words last (Nov 21 2026)

João: "predicting words is wrong. It should be predicting thoughts and meanings. And predicting just one is wrong. It
should predict everything at the same time ... Diffusion must happen all at once, slowly. Some parts are more sure than
others ... we just need a mouth that talks in the end."

```
words ─► ear (4 word layers, 64 pieces of sight)
            │  every 16 pieces: the chunk's MEANING (128 numbers)
            ▼
         mind (4 layers of 768 over the meanings, causal: one position per chunk)
            │
            ▼
         canvas: the next 4 chunks' meanings, all at once, from fog (noise 1)
            │  refined together in 4 steps (DDIM, deterministic); every slot sees the mind and every other slot;
            │  each slot says how sure it is, and a sure slot drops its noise faster
            ▼
         mouth (8 word layers, 64 pieces of sight, reading the canvas before layers 5, 7, 9, 11) ─► the next word
```

| | Bee-135 | Canvas-135 |
|---|---|---|
| weights | 132.2M | 131.8M (mind + canvas + reads 42.1M) |
| word layers | 24 of 576, full sight | 12 of 576, 64 pieces of sight |
| what predicts the future | the next word | the meanings of the next 4 chunks (64 pieces), jointly, by denoising |
| next-token? | yes | the mind is not; the mouth writes word by word, from the thought |

Lessons: the next word (through the canvas exactly as it is built at reading time), diffusion (the true future
meanings, each slot noised to its own level, 10% given clean), honest foresight (the settled canvas against the true
future), how sure (Gaussian likelihood of each slot's own error), SIGReg (LeJEPA) on the meanings. The meanings are
the ear's own, held still as targets (JEPA). Nothing is labelled: what a "meaning" is, the model invents.

Hand-set, on purpose for a first test: the 16-piece rhythm, 4 slots, 4 steps. Learned boundaries and a learned "stop
when settled" are the next step if this works (Pegasus collapsed when it learned boundaries from the start).

```
python3 -m canvas.tests                  # CPU, ~2 min, 14 checks
python3 -m canvas.tests --device cuda    # + a race-size compiled step and its speed against Bee-135
bash runpod/canvasab.sh                  # the A/B: Bee-135 (shared with the judge A/B) and Canvas-135, 2B tokens each
python3 -m canvas.ab                     # the table and the verdict
```

Rule fixed before the run: kept only if reasoning (mean of ARC-E, ARC-C, CSQA, Winogrande) is +1.5 points or more over
Bee-135, HellaSwag no more than 3 points under, and the held-out loss rises 0.05 or more with the canvas switched off.
The log's `[core]` line is the canvas-off check; `[loop]` gives the held-out loss with 1, 2, 4 and 8 refining steps
(lower with more steps = thinking longer helps).
