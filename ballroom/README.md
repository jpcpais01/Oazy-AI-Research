# Ballroom

A tiny model that learns how a ball moves in a 2D room, scored against the exact simulator.
Not next-token: next-state.

- `sim.py`: the exact simulator. A circle against boxes (the walls are boxes too), gravity, bounce, rolling friction. Every room comes to rest.
- `model.py`: each box is a token seen relative to the ball; a small transformer (about 0.45M weights) reads them.
  - step model ("law"): one moment of motion (1/30 s), rolled forward until it says the ball rests.
  - one-shot model ("intuition"): guesses the resting place directly.
- `train.py`: trains either model. The step model learns from true paths, nudged states and the states it visits itself, all labelled by the simulator.
- `test.py`: fresh rooms with 2-3 platforms (like training) and 4-5 (never seen).
- `export.py`, `editor/`, `editor.py`: the room editor. Drag things, press Run, watch the true path and the model's path.

- `eye.py`: the same task from pixels. A picture of the room goes in, a few tokens of inner state come out, a small transformer steps that state forward, and the position is read out. Also not next-token. Its test nudges the inner state to give a cloud of answers. One line: `bash runpod/eye.sh` (20 minutes).

One line on a GPU pod: `bash runpod/ballroom.sh` (about 15 minutes, then the editor is on port 8000).
