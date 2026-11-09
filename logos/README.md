# Logos1-200M

Bee6's word stream with Logos1's way of thinking built in beside it. It is a **next-word model**, scored like every
Bee (HellaSwag, ARC, PIQA, LogT, and the world's secret exams as plain text). The whole design is at the top of
`model.py`.

- **Word stream:** Bee6, 10 layers of 1,024 instead of 12, with all of Bee6's speed tricks.
- **Linker:** after layer 4, each piece says whether it names a thing and points back to the earlier piece that names
  the same thing, as Logos1's linker does in world-link.
- **Laws core:** every 32 pieces, the 16 things mentioned most recently before that point go through a shared core of
  2 layers of 768, for 3 passes. The core sees only things, never words.
- **Read-back:** before layers 6 and 9, each piece reads the things of its stretch through cross-attention. It only
  sees things of its own document, built from text before the stretch began.
- **Answer key:** on the world's episodes, Emerald's marks grade the linker, exactly as in world-link. Ordinary text
  has no marks.

Size: 201.1M, against Bee6's 202.0M. It uses Bee6's recipe exactly. In the race, Bee6 gets the same answer key
(`--set teach=1`): the same link head on its stream after layer 4, graded the same way, as the rival in world-link.

```bash
python3 -m logos.tests                                              # CPU checks (~10 min)
tmux new-window -t oazy -n logos "bash runpod/logos_race.sh; bash"  # vs Bee6 on Emerald, 2B tokens each, then the table
```

During training the log shows `[core]`: the held-out loss with the core on and with it off, on Sado text and on the
world. If the gap stays near zero, the stream is ignoring the core.
