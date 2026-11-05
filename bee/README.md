# Bee0.2B: our main model

**Goal:** the best ~200M-parameter model we can train on 5B tokens, on one rented GPU.
**Next-word model:** it reads text and guesses the very next word, like GPT-2. Scored the same way
(loss on the same held-out FineWeb-Edu text, HellaSwag).

Instead of inventing a new principle, Bee assembles what the best 2025 models proved works, plus our
own memory layer. That makes it the low-risk "strong baseline" the wilder ideas (Fern, Fern3) can be
measured against.

## The body (204M parameters)

```
 words in ─► word table (50k words x 768, shared with the output)
             ┌────────────────────────────────────────────────┐
             │ memory    memory    memory    attention        │  x 6   = 24 layers
             │ (Memtrix) (Memtrix) (Memtrix) (looks back)      │        18 memory + 6 attention
             └────────────────────────────────────────────────┘
             each layer = norm ─► memory or attention ─► + ─► norm ─► SwiGLU "thinking" ─► +
 next word ◄─ norm ─► scores against the word table, softly capped at ±30
```

- **Memory layers (3 of 4):** Memtrix's learn-while-reading memory (`memtrix/model.py`, same code):
  6 memories of 128 × 128 numbers per layer, fixed by their own surprise word after word. Cheap, fixed
  size, and they carry word order. In the research world this is Gated DeltaNet.
- **Attention layers (1 of 4):** plain look-back attention, so any word can copy a name or a number
  from far back exactly (what memories are bad at). 12 question heads share 4 key/value heads,
  QK-norm, and a sigmoid gate on the output (Qwen3-Next's "gated attention", for stable training).
  **No position signal** in attention (NoPE, as in Kimi Linear): the memory layers already know the
  order, and this lets Bee read texts longer than its training windows.
- **Modern details GPT-2 lacks:** RMSNorm, SwiGLU, no bias terms, tied word tables, logit soft-cap.

This 3:1 memory/attention pattern is exactly Qwen3-Next's and Kimi Linear's (2025).

## How it is raised (`oazy/train.py --model bee`)

| | what | why |
|---|---|---|
| mentor | GPT-2 **large** (774M) shows its full guess for every word; its share fades 60% → 0 by halfway | richer lesson per word; GPT-2 large is clearly better than Bee will be (HellaSwag ~45%), medium might not be by the end |
| optimizer | Muon on every weight grid, AdamW on the rest (word table: no weight decay) | ~1.3x faster learning (nanoGPT speedrun, Kimi) |
| schedule | warm-up, flat, then a straight fall to zero over the last 20% ("WSD") | the fall lines up with the best data, and beats cosine at a fixed budget |
| windows | 1,024 words while the mentor teaches (GPT-2 reads at most 1,024), 2,048 after | cheap first, longer memory later; no text is repeated or skipped at the switch |
| text | 0-80%: 65% FineWeb-Edu + 35% DCLM. 80-100%: 45% FWE + 25% DCLM + 30% conversations | overall ~61/33/6; the conversations teach it to answer, last so they stick (`oazy/mix.py`) |
| batch | 524k words per step (32 x 1,024, or 16 x 2,048, times 16) | |

## Run it (RunPod, one B200)

```bash
cd /workspace/oazy && git pull
bash runpod/setup.sh bee                      # once: tests + DCLM (2B tokens) + conversations (~1 h)
python3 -m oazy.train --bench --model bee --preset big   # speed, including the mentor
bash runpod/train.sh bee --preset big         # 5B tokens; the app on port 8000 shows it live
python3 -m bee.chat --port 8002               # talk to it (from its latest checkpoint)
```

On a 32 GB GPU (RTX 5090) add `--micro 8` (same words per step, smaller pieces).

**Cost (estimate, check with `--bench`):** about 10 hours on one B200: ~6 h for the first half (Bee
plus GPT-2 large reading along) and ~4 h for the second. At ~$6/h that is about $60 (≈55 €), plus
~1 h for data. A 5090 is ~6-8x slower: similar money, three days.

**What to expect (a guess):** clearly ahead of GPT-2 small (124M): HellaSwag in the mid-30s versus
29.6%, val loss well below our GPT-2 baseline. And, after the conversations, a model you can actually
ask things, within the limits of 200M parameters.

## Measure each choice (same tokens, own `--name`)

```bash
python3 -m oazy.train --model bee --preset big --name bee-no-mentor --mentor none
python3 -m oazy.train --model bee --preset big --name bee-medium    --mentor gpt2-medium
python3 -m oazy.train --model bee --preset big --name bee-adamw     --opt adamw
python3 -m oazy.train --model bee --preset big --name bee-cos       --schedule cos
python3 -m oazy.train --model bee --preset big --name bee-rope      --set rope=1
python3 -m oazy.train --model bee --preset big --name bee-all-mem   --set pattern=M
python3 -m oazy.train --model bee --preset big --name bee-fwe-only  --mix "fwe:1@0"
```

Other settings: `--set pattern=MMMA n_layer=24 d_model=768 mlp_hidden=1792 ctx=1024:0,2048:0.5`,
`--mix "fwe:0.65,dclm:0.35@0;fwe:0.45,dclm:0.25,chat:0.3@0.8"`, `--mentor-weight 0.6 --mentor-until 0.5`.

## Files

`model.py` (the model), `chat.py` + `web/chat.html` (chat with a look inside), `tests.py`
(`python3 -m bee.tests`: size, no peeking at later words, stream reading = reading at once, the
optimizer split, the mix, the schedule, the mentor). The text mix lives in `oazy/mix.py`.
