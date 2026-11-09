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

## Logos1.1: the core gets the job of a law (Nov 9, night)

At 2B tokens on Emerald, Logos1-200M lost to a plain Bee6 almost everywhere, world secret exams included (51.5% against
58.6%), although it did use its core (switching the core off cost about 0.03). The reading: the core had no job of its
own. Its only signal was a faint trickle from the next-word loss through the read-back, and on 95% of the text
(ordinary web text) that is close to nothing.

Logos1.1 gives it the job of a law, in João's sense: from how things are now, say how they will be next.

- **The law's guess (`--set predict=1`):** before each stretch, for every thing the core holds, guess the mean of the
  stream's vectors at that thing's mentions in the coming stretch (after layer 4, held still as the target). It is
  graded by cosine once the stretch is read. It works on all text, no marks: any text's things change from one
  paragraph to the next. The stretch reads the core's output, which is now its guess about that very stretch, made
  from the past only.
- **Focus (`--set focus=1`):** the core also gets the stream at the last piece before the stretch as one more
  member, so what it thinks can depend on what is being asked. It sees only text before the stretch.

The `[core]` line adds "law guesses the next stretch's things: cosine X vs copying Y". Copying is the thing as it was.
The law has to beat it, or it is not learning anything. If both climb towards 1.0, the stream is collapsing its
vectors to make the guess easy: stop and report that.

```bash
tmux new-window -t oazy -n logos11 "OAZY_DATA=/root/data OAZY_RUNS=/workspace/runs bash runpod/logos11_race.sh; bash"
```

That trains Logos1.1, then Logos1.1 + focus, then Bee6 with the answer key, at 0.5B tokens each, then writes the table
(`<runs>/logos11-race-0.5b.md`).
