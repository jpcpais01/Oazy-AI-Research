# Oazy AI Research

Our own language-model architectures, trained from scratch on FineWeb-Edu and measured against
GPT-2 small (124M) on the same text, with the same budget, by the same tests.

| model | idea | code |
|---|---|---|
| **Engram** | learns while it reads: every block holds small memories (64 × 64 numbers) that are fixed by their own surprise at each word; no attention, fixed memory however long the text | [`engram/`](engram/) |
| **Tide** | Engram plus a belt of drafts of the next 8 words, each cleaned up once per word as it moves towards "now" | [`tide/`](tide/) |
| GPT-2 | the baseline: GPT-2 small, trained here exactly like our models | [`oazy/gpt.py`](oazy/gpt.py) |

**The lab** ([`oazy/`](oazy/)) is shared by every model: the data pipeline, training (same text in
the same order for every model), HellaSwag, the "is the memory used?" checks, the comparison with
OpenAI's GPT-2, and the app: a live dashboard of the models in training, plus each model's chat.

```
oazy/          the lab: data.py, train.py, hellaswag.py, probes.py, compare.py, dashboard.py, web/ (the app)
engram/        Engram: memory.py (the one rule), model.py, chat.py, tests.py
tide/          Tide: model.py, chat.py, tests.py
runpod/        setup.sh (once per pod), train.sh <model>
docs/RUNPOD.md how to run everything on a RunPod GPU
```

Adding a model: a package with `model.py` (forward(idx, targets) → logits, loss, stats; `_body`;
`generate`), registered in `oazy/train.py` (`build`, `load_model`) and in the app's `MODELS` list.

Results so far (1.58B tokens, one B300): HellaSwag Engram 28.4%, GPT-2 trained the same way 27.8%,
OpenAI GPT-2 29.6%.
