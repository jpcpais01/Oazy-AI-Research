# Oazy AI Research

Our own language-model architectures, trained from scratch and measured against GPT-2 small (124M)
on the same text, with the same budget, by the same tests (held-out loss and HellaSwag).

Everything runs on one RunPod GPU: **[docs/RUNPOD.md](docs/RUNPOD.md)** has every command.

## The models

**Main line** (next-word models, ~200M parameters, 5B tokens):

| model | idea | code |
|---|---|---|
| **Bee0.2B** | our main model (204M): a hybrid of 18 Memtrix memory layers and 6 attention layers (3:1, like Qwen3-Next), learns from GPT-2 large early on, Muon optimizer, a FineWeb-Edu + DCLM mix with conversations at the end | [`bee/`](bee/) |
| **Bee_v2_0.2B** | Bee0.2B's body (203M) with a smarter, cheaper mentor: SmolLM2-360M instead of GPT-2 large, so it reads SmolLM2's tokenizer (the same text, re-cut) | [`bee2/`](bee2/) |

**Experiments** (smaller, ~120M, each testing one idea against GPT-2 small):

| model | idea | next-word? | code |
|---|---|---|---|
| **Memtrix** | learns while it reads: every block holds small memories (64 × 64 numbers) that are fixed by their own surprise at each word; no attention, fixed memory however long the text | yes | [`memtrix/`](memtrix/) |
| **Tide** | Memtrix plus a belt of drafts of the next 8 words, each cleaned up once per word as it moves towards "now" | yes | [`tide/`](tide/) |
| **Ember** | Memtrix raised to learn fast: grows 4 → 8 → 12 blocks, learns from GPT-2 medium early on, Muon | yes | [`ember/`](ember/) |
| **Fern** | writes 16 words at a time, coarse to fine, in 5 rounds: where the passage is going first, then the middle, then the details (198M) | **no** | [`fern/`](fern/) |
| **Fern2** | Fern at 119M, raised the Ember way (grows, GPT-2 medium mentor, Muon) | **no** | [`fern2/`](fern2/) |
| **Fern3** | Fern that remembers like you do: each finished passage is folded into one gist, and for each new passage it recalls the 3 old passages whose gists match best and reads them in full (117M) | **no** | [`fern3/`](fern3/) |
| GPT-2 | the baseline: GPT-2 small, trained here exactly like our models | yes | [`oazy/gpt.py`](oazy/gpt.py) |

Results so far (1.58B tokens, one B300): HellaSwag Memtrix 28.4%, GPT-2 trained the same way 27.8%,
OpenAI's GPT-2 29.6%.

## What's where

```
bee/  bee2/                       the main models: model.py, chat.py, tests.py, README.md
memtrix/ tide/ ember/             experiments (memtrix/memory.py is the memory rule Bee uses too)
fern/ fern2/ fern3/               experiments that are not next-word (all chat through fern/chat.py)
oazy/                             the lab, shared by every model:
  data.py mix.py tok.py retok.py    text: FineWeb-Edu, the Bee mix, tokenizers, re-cutting for SmolLM2
  train.py muon.py                  training (same text in the same order for every model), Muon
  hellaswag.py probes.py compare.py tests: HellaSwag, "is the memory used?", OpenAI's GPT-2
  dashboard.py web/                 the app: live dashboard of training runs, plus each model's chat
  gpt.py sample.py synth.py         GPT-2 baseline, text generation, a tiny corpus for CPU tests
  site/                             the public landing page
runpod/                           setup.sh (once per pod), train.sh <model>
docs/RUNPOD.md                    how to run everything on a RunPod GPU
```

**The landing page** ([`oazy/site/index.html`](oazy/site/index.html)) is one self-contained file: open it in any
browser (double-click it, no server needed). The light tunnel in its hero is a WebGL shader written for Oazy; it
lowers its own resolution on slow GPUs, pauses when scrolled away, and holds still for people who ask for reduced motion.
Model results on it are placeholders until the runs finish.
On Vercel: import this repo, set **Root Directory** to `oazy/site` (Framework Preset: Other, no build command),
then in Settings → Git set the **Production Branch** to `oazy-ai-research`. Only the page is published, not the code.

**Adding a model:** a package with `model.py` (`forward(idx, targets)` → logits, loss, stats; `_body`;
`generate`), registered in `oazy/train.py` (`build`, `load_model`), in `runpod/train.sh`, and in the app's
`MODELS` list.

Memtrix was called Engram until Nov 5 2026: old run folders (`runs/engram`, `/workspace/engram_*`) and
`--model engram` still work.

This repo started as a candle-flame simulation; it is still in this branch's history (`git checkout 0286d44`).
