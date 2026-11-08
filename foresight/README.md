# Plain-100F: Plain-100M that learns to foresee the next sentence

**Next-word model.** Plain-100M exactly (10 layers of 768, Bee4's layers, 106.6M parameters, the same recipe, mentor and
text), plus one small part used only while training: the **foresight head**.

## What the head does, in plain words

At the end of every sentence (or every 16 pieces when a sentence runs long), the head looks at the model's last hidden
state, the model's whole "understanding so far", and guesses which content words the next stretch of text will use: the
words up to the next sentence end, at most 32 pieces, as a bag, so order does not matter. Filler is not in the bag: the
300 most common pieces of the training text (the, of, and, ...), punctuation, spaces and the document marks.

The answer key is the real next sentence, so the model cannot cheat by inventing an easy code, which is how both Pegasus
versions collapsed. The only way to do well is for the hidden state at a sentence end to hold what comes next. The bet:
learning that makes the trunk plan a little further ahead, and that helps the next-word guesses too.

- **The head**: a norm and a 2-layer MLP (768 -> 1,536 -> 768), then the tied word table for the scores. About 2.4M
  numbers, learned with Muon like the trunk's grids.
- **The loss**: a softmax over the content words, averaged over every (read point, bag word) pair, in nats per bag
  word. Weight 0.1 next to the next-word loss, growing from 0 over the first 5% of the run. Its gradient reaches the
  whole trunk.
- **Removed after training**: the held-out loss, HellaSwag, LogRes, chat and the phone read only the trunk and the word
  table, so the size and speed there are Plain-100M's. The head's weights stay in the checkpoint so a run can resume.
- **Read points per pass**: about one in ten pieces; at most 8,192 per pass (a pass of 65k pieces has about 6k), so the
  extra work is roughly 5% of the word-table work.

## What the log shows

- Each training line: `foresight X (x0.1)`, the head's loss on that step and its weight.
- At each probe: `[foresight] held-out X nats per bag word vs Y blind`. The blind guess picks each content word by how
  common it is. The gap between the two is how much the model knows about the next sentence.

## Run it (on the B200 pod)

```
cd /workspace/oazy && git pull
python3 -m foresight.tests --device cuda          # ~3 min: everything, plus the compiled step and the speed cost
OAZY_DATA=/root/data bash runpod/train.sh plain100f --preset big --tokens 0.5e9 --micro 32 --data-sub sado --name plain100f-race
```

Compare against the existing `plain100-race` (the same command with `plain100`): held-out loss, loss by position (the
`[memory]` probe), HellaSwag and LogRes (`python3 -m oazy.logres --run plain100-race --run plain100f-race`).

It is kept apart from the in-training data selection build, so each idea is tested alone.
