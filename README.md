# Oazy AI Research

Our own language-model architectures, trained from scratch on FineWeb-Edu and measured against
GPT-2 small (124M) on the same text, with the same budget, by the same tests.

| model | idea | code |
|---|---|---|
| **Memtrix** | learns while it reads: every block holds small memories (64 × 64 numbers) that are fixed by their own surprise at each word; no attention, fixed memory however long the text | [`memtrix/`](memtrix/) |
| **Tide** | Memtrix plus a belt of drafts of the next 8 words, each cleaned up once per word as it moves towards "now" | [`tide/`](tide/) |
| **Fern** | writes 16 words at a time, coarse to fine, in 5 rounds: where the passage is going first, then the middle, then the details; **not a next-word model**. Reads the past with Memtrix's memories | [`fern/`](fern/) |
| **Ember** | Memtrix's body raised to learn fast: grows 4 → 8 → 12 blocks, learns from GPT-2 medium early on, Muon optimizer; next-word | [`ember/`](ember/) |
| **Fern2** | Fern at 119M (under GPT-2's 124M), raised the Ember way: both halves grow, the reader also learns next words from GPT-2 medium, Muon; **not a next-word model** | [`fern2/`](fern2/) |
| **Fern3** | Fern that remembers like you do: each finished passage is folded into one gist, the model thinks with gists, and for each new passage recalls the 3 old passages whose gists match best and reads them in full. No fixed-size memory; 117M; **not a next-word model** | [`fern3/`](fern3/) |
| GPT-2 | the baseline: GPT-2 small, trained here exactly like our models | [`oazy/gpt.py`](oazy/gpt.py) |

**The lab** ([`oazy/`](oazy/)) is shared by every model: the data pipeline, training (same text in
the same order for every model), HellaSwag, the "is the memory used?" checks, the comparison with
OpenAI's GPT-2, and the app: a live dashboard of the models in training, plus each model's chat.

```
oazy/          the lab: data.py, train.py, hellaswag.py, probes.py, compare.py, dashboard.py, web/ (the app)
memtrix/        Memtrix: memory.py (the one rule), model.py, chat.py, tests.py
tide/          Tide: model.py, chat.py, tests.py
fern/          Fern: model.py, chat.py, tests.py
ember/         Ember: model.py (growing), mentor.py (distillation), tests.py
fern2/         Fern2: model.py, tests.py (chat: python -m fern.chat --run fern2)
fern3/         Fern3: model.py, tests.py (chat: python -m fern.chat --run fern3)
runpod/        setup.sh (once per pod), train.sh <model>
docs/RUNPOD.md how to run everything on a RunPod GPU
```

Adding a model: a package with `model.py` (forward(idx, targets) → logits, loss, stats; `_body`;
`generate`), registered in `oazy/train.py` (`build`, `load_model`) and in the app's `MODELS` list.

Results so far (1.58B tokens, one B300): HellaSwag Memtrix 28.4%, GPT-2 trained the same way 27.8%,
OpenAI GPT-2 29.6%.
