# Engram: a language model that learns while it reads

*Working name. Alternatives: Imprint, Hebb, Plastic.*

```
   word ──► meaning (+ a hint of the last 3 words)
                 │
     ┌───────────┼──────────── one block (×12) ─────────────────────┐
     │   key  "this is the situation"     value "what goes with it" │
     │           │                              │                   │
     │      memory M (64×64) ── guess = M·key   │                   │
     │           │                              ▼                   │
     │           │                  surprise = value − guess        │
     │   fade a little, then  M ← M + trust × surprise ⊗ key        │
     │           │                                                  │
     │   question "what comes next?" ──► answer = M·question        │
     │           └──► small per-word layer ──► next block           │
     └──────────────────────────────────────────────────────────────┘
                 │
            next-word probabilities
```

**The one rule:** read a word, ask the memory what it expected, fix the memory by its surprise.
Every block does the same thing on the block below. The memories start blank for each text, never
grow, and are the *only* way information travels more than 3 words (no attention, no position table),
so the model cannot route around its memory.

* `memory.py` the rule, three ways: word by word (reference), chunked PyTorch, fla Triton kernel; tested equal.
* `model.py` Engram (117M at GPT-2 small width/depth), `gpt.py` GPT-2 small baseline (124M) + OpenAI weights loader.
* `data.py` FineWeb-Edu → tokens; `train.py` training (same data order and schedule for both models).
* `probes.py` "is the memory used?" checks: context curve, loss by position, exact recall of a repeated passage.
* `hellaswag.py`, `compare.py` (measures OpenAI GPT-2), `dashboard.py`, `chat.py`, `tests.py`, `synth.py` (CPU toy data).

Family: fast-weight programmers, DeltaNet, Gated DeltaNet, Titans. RunPod guide: `docs/ENGRAM_RUNPOD.md`.
