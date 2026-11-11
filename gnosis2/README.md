# Gnosis2 and Opal2: the best shot (Nov 11 2026)

João asked for the best model we can make right now, around 300M weights and 10B tokens, using everything we know.
Gnosis2 is still a **next-word model** (it reads text and guesses the very next token, like every Bee and Gnosis),
scored the same way. What is new is mostly not the body. It is what the model reads and how it is finished.

## Why this shape

Over the last week almost every new *body* we tried came out level with plain Bee6 or behind it. Alpha Star's loops
tied, Logos1 and Logos1.1 lost, Tide's drafts cost loss, and MoE won only by spending more weights. The things that
moved us were all about the text. The past year's papers say the same: at our size the 2-6x gains come from what a
model reads and what it is rewarded for. So Gnosis2 keeps Gnosis's body (with one small change that has evidence on
reasoning) and puts the effort into three new kinds of text and a finishing lesson.

## The three stages

| stage | what | where | cost |
|---|---|---|---|
| 1. model | Gnosis (22 x 1,024, ~307.5M) + Canon layers | `gnosis2/model.py` | +98k weights, <1% compute |
| 2. text | Opal2 = Opal + logic warm-up + Field1 worked out + text that shows its thinking | `oazy/opal2.py`, `oazy/proc.py`, `oazy/field1/work.py`, `oazy/think.py` | CPU minutes + 2-3 GPU hours |
| 3. finish | the reasoning lesson: it writes its own work, the simulator grades it (GRPO) | `gnosis2/rl.py` | ~2-3 h on an H200 |

### 1. Canon layers (the one change to the body)

There is a tiny causal mix, 4 words wide (one weight per channel per offset), on the input of attention and of the
MLP, in the 12 layers every word goes through. Attention is good at reaching far back and poor at the simple job of
mixing a word with its neighbours, which it spends whole layers doing. Canon layers do that job almost for free.
Allen-Zhu's controlled tests (Physics of Language Models 4.1, NeurIPS 2025) found 2-4x deeper multi-step reasoning.
The weights start at zero, so Gnosis2 starts as Gnosis exactly (tested).

### 2. Opal2

- **proc, the logic warm-up**: brackets, cellular automata, sorting, stacks, sums, counting, with no words at all.
  The model reads only this for the first 0.2% of the run (20M pieces), then gets a 0.5% pinch until 10%.
  Procedural Pretraining (Jan 2026) reports the same loss with 55-86% of the text after this kind of warm-up.
- **work, Field1 worked out**: every scene's answer is reached step by step by the simulator itself. The clues are
  gathered in chain order, the events are put back in the order they happened, and the families that change step by
  step (lines, stacks, count) say what is true after each event. Because the simulator knows the truth, the reasoning
  is never wrong. This part is 8% of the run.
- **think, text that shows its thinking**: Opal passages followed by the thinking a careful reader does, written
  by Qwen3-1.7B. This part is 12% of the run. Thinking Augmented Pre-training reports ~3x learning per token, and Tufa
  Labs got 3-6x fewer tokens to the same maths scores at 0.8B.
- **field1** drops from 20% to 7%, since the worked part now carries most of the world.
- **text**: Opal's own text is the rest (~73%), in its own proportions, with its cool-down from 70%. The religious
  text keeps its exact 2%.

A part that is not built is skipped (the run says so) and the others fill its share.

### 3. The reasoning lesson

Gnosis2 gets a fresh Field1 scene and question. It writes its own work, and the simulator says right or wrong. Each
step takes 16 questions with 8 answers each. An answer is pushed up or down by how it did against its own group, and
a small KL penalty keeps the English from drifting. Families the model gets right about half the time come up more
often, because those are the ones it can learn from. It needs no more text, only a grader, and Field1 is a perfect
grader that never runs out. The world exams (secret, practised, Real World, Pro) are never trained on, so they measure
whether the skill transfers.

## Not in it, and why

- **The sandwich loop** (looping the middle layers): it saves weights, not compute, and our cap is tokens at fixed
  compute. Our own Alpha Star loop tied, so it is a separate race if 1-3 leave logic weak.
- **Diffusion LMs**: they only beat next-word models when data is repeated many times, and we have fresh text.
- **Energy-based transformers**: several times the training compute.
- **The SmolLM2 mentor**: off, as in the Gnosis run (`--mentor none`). It was never A/B tested since Bee3, and João
  suspects it pulls the model toward ordinary-LM habits.

## Run it

```
# CPU pod, on the volume (Field1 must exist; it is built first if not)
cd /workspace/oazy && git pull && nice -n 10 python3 -m oazy.opal2 build

# any GPU (H200 ~2-3 h, or overnight on the 5090): text that shows its thinking, needs vllm (pip install vllm)
python3 -m oazy.think --tokens 400e6

# the GPU pod: tests first, then the 10B run (~11-13 h on an H200)
python3 -m gnosis2.tests --device cuda
OAZY_DATA=/workspace/data OAZY_RUNS=/workspace/runs bash runpod/train.sh gnosis2 --preset big --tokens 10e9 --micro 32 --data-sub opal2 --mentor none --name gnosis2-10b

# the finish (~2-3 h), then the exams on both
python3 -m gnosis2.rl --run gnosis2-10b
OAZY_RUNS=/workspace/runs python3 -m oazy.worldmix exam --run gnosis2-10b-reason --run gnosis2-10b
```

Read the new text first: `python3 -m oazy.opal2 show --part work --family lines`, `--part proc`, `--part ask`,
`--part think`.

The comparison is gnosis-opal-10b (same body without Canon, plain Opal, no finish). The thing to watch is Real World
and Pro. A rise there that holds through training would mean the logic transfers and is not just wording.
