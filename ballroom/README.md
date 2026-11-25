# Ballroom

A tiny model that learns how a ball moves in a 2D room, scored against the exact simulator.
Not next-token: next-state.

- `sim.py`: the exact simulator. A circle against boxes (the walls are boxes too), gravity, bounce, rolling friction. Every room comes to rest.
- `model.py`: each box is a token seen relative to the ball; a small transformer (about 0.45M weights) reads them.
  - step model ("law"): one moment of motion (1/30 s), rolled forward until it says the ball rests.
  - one-shot model ("intuition"): guesses the resting place directly.
- `train.py`: trains either model. The step model learns from true paths, nudged states and the states it visits itself, all labelled by the simulator.
- `test.py`: fresh rooms with 2-3 platforms (like training) and 4-5 (never seen).
- `eye.py --view 2` (VIEW=2 in runpod/eye.sh): the room stays in view; every imagined step looks at the room picture again, only the ball is imagined.
- `export.py`, `editor/`, `editor.py`: the room editor. Drag things, press Run, watch the true path and the model's path. Pixel models (every `eye.pt` under the runs folder) appear in its model picker and run on the machine serving the page.

- `eye.py`: the same task from pixels. A picture of the room goes in, a few tokens of inner state come out, a small transformer steps that state forward, and the position is read out. Also not next-token. Its test nudges the inner state to give a cloud of answers. One line: `bash runpod/eye.sh` (20 minutes).

  `--emotion 1` (EMOTION=1): learning strength = surprise x learnable x importance. Half of each batch is drawn where the model's own recent error was high, rooms where the simulator's own nudged cloud splits (unpredictable) pull less, and steps at rest count double.
  `--odds 4` (ODDS=4): a world model with chance. Every step draws a little noise, and from the start picture its clouds are scored against the exact simulator's own cloud of nudged copies (energy score), so it learns the odds of each future instead of one answer. The test says how far its odds are from the real ones.
- `robot.py`: a cart with a cannon looks at a picture of the room and a target, proposes 16 throws, imagines each with noise and fires the one that lands most often. Learns by hindsight from random throws; its body is off by a little noise every throw. One line: `bash runpod/robot.sh`.

One line on a GPU pod: `bash runpod/ballroom.sh` (about 15 minutes, then the editor is on port 8000).
