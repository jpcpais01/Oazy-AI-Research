# Ember: a mind raised to learn fast

*Working name.* Still a **next-word** model, scored exactly like GPT-2 (loss, HellaSwag).

Ember's body is Engram's learn-while-reading memory, unchanged (117M once fully grown, vs GPT-2's
124M). What changes is how it is *raised*, because the cost we want to cut is training:

```
 tokens seen   0%        10%                25%                     50%                    100%
 depth         4 blocks ─► 8 blocks ─────────► 12 blocks ───────────────────────────────────────►
 mentor        GPT-2 medium's guesses, 60% of the lesson ──fading──► 0  (Ember alone after this)
 optimizer     Muon on every weight grid, AdamW on the rest, the whole way
```

1. **It grows, like a child's brain.** Starts 4 blocks deep (about 3x cheaper per word in the
   blocks), copies its blocks to become 8 deep at 10%, then 12 at 25%. The copies start out knowing
   what the originals knew. `ember/model.py`
2. **It learns from a mentor.** For the first half of training it also matches GPT-2 medium's full
   guess for every word ("70% cat, 20% dog..."), a much richer lesson than one right answer. The
   mentor's share fades to zero by 50%. `ember/mentor.py`
3. **It learns evenly.** Muon straightens each weight update so weak directions learn as fast as
   strong ones. `oazy/muon.py`

Honest expectations: each idea is known to give roughly 1.3-2x on its own (Muon ~1.35x in the
nanoGPT speedrun, stacking ~1.5x in G_stack, distillation 1.5-3x when the mentor is clearly better).
Together, maybe 2-4x fewer GPU-hours to reach Engram's loss. Not 1000x: no honest method gives that.
The mentor is GPT-2 medium, so beating GPT-2 *small* with it is a fair but assisted win (Gemma 3
270M was trained the same way, from a bigger Gemma).

## Run it (RunPod)

```bash
python3 -m ember.tests --device cuda
bash runpod/train.sh ember --tokens 2e9
```

Measure each idea on its own (same tokens, compare wall-clock time to the same loss):

```bash
python3 -m oazy.train --model ember --name ember-no-mentor --mentor none
python3 -m oazy.train --model ember --name ember-no-grow   --set grow=12:0
python3 -m oazy.train --model ember --name ember-adamw     --opt adamw
```

Options: `--mentor gpt2-large` (stronger, ~2x the mentor cost), `--mentor-weight 0.6`,
`--mentor-until 0.5`, `--muon-lr 0.02`, `--set grow=4:0,8:0.1,12:0.25 grow_mode=stack|quiet`.
Chat: `python3 -m engram.chat --run ember --port 8002` (same lid-off view as Engram).
