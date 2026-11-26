# Survive: a creature that learns to stay alive

A creature in a 3D arena with two stereo eyes (120°), two ears, a 4-sensor intensity sense and
body feelings (energy, health, touch, on-ground, holding). Four raw muscles: push, turn, jump, grip.
Nothing is scripted and nothing points it at food or tools: its only drive is wellbeing =
(energy + health) / 2, read from its own predicted senses, and the end of its life.

The world (`world.py`): 11.3 m arena, static boxes and 1.4 m ledges, movable crates (push or carry),
lava, food of 3 kinds (some on ledges), a predator that chases and bites, a harmless wanderer and a
strange being that drains health if the creature lingers near it. Metabolism: idle drain d0 = 1/300
per step, +0.2 d0 for any muscle use, effort and speed² costs, +12 d0 per jump, carrying doubles it.

The mind (`model.py`), the "clean" recipe that won the Nov 25-26 2026 tests (465 vs 235 steps for a
picture-redrawing mind): a whole-state JEPA anchored by the body.
- eye -> 4096 numbers kept rich by VICReg (variance + covariance); no picture loss;
- GRU memory (512) + 32x16 categorical moment, guessed before looking (prior) and corrected after
  looking (posterior); the KL pulls the guess toward the correction;
- heads predict the senses and staying alive; actor and critic learn only in imagination (15 steps);
- a detached mind's-eye decoder draws what the inner state holds, for people only.

Run: `bash runpod/survive.sh` (MIN, RES, OUT, BATCH...). Score: `python3 -m survive.bench <mind.pt>`.

## The reference recipe (Nov 26 2026, run `ref`): every later idea changes exactly one thing

| part | setting |
|---|---|
| eyes | 64 px, 2 x 120°, VICReg var 1 + cov 1 on 2048 random moments per update |
| memory / moment | GRU 512, 32 x 16 categorical, KL 0.5 dyn + 0.1 rep, free bits 1 |
| heads | 29 senses (MSE, summed), alive (BCE) |
| imagination | 15 steps, γ 0.997 (~33 s), λ 0.95, Dreamer-v3 return scaling (5-95% of all imagined steps), slow critic 0.02 |
| actor | tanh-Gaussian, entropy 3e-4, dynamics backprop, lr 8e-5 |
| world model | lr 3e-4, clip 1000 |
| life | 512 worlds, train mix; batch 256 clips x 32 moments, 1 update per step (16 studies per moment) |
| speed | torch.compile, bf16, ~50k frames learned/s on an RTX 4090 |
| yardstick | `survive.bench`, 1024 fresh lives each in test / hard / train worlds |

Sleep test: `python3 -m survive.sleep <mind.pt> --mode hard` (days of frozen living, nights of replay).
