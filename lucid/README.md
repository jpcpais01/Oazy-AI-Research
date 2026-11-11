# Lucid-300M

A new kind of language model: it keeps a **picture** of what it is reading and **settles it by surprise**, instead of
looking back over every word. It is still scored as a next-word model (every test reads the odds it gives each next
piece), so it races Gnosis on the same text and the same tests. The whole design is at the top of `model.py`.

## The idea

A transformer has no picture of anything. Its only memory is the transcript: at every word, in every layer, it looks
back over every word read so far. It never has to make sense of the text, because it can always re-read it.

Lucid cannot re-read. Its word layers see only the last 64 pieces. Everything older reaches it only through the
**mind**: 16 slots (vectors of 1,024) that are its current understanding. To get the next word right it has to keep a
good picture, because the picture is all it has.

Every 32 pieces (a **beat**) the mind settles, in 3 passes:

1. **imagine**: from the picture alone, imagine what this beat should look like
2. **surprise**: what the ear actually heard, minus what was imagined
3. **claim**: the slots compete for each surprise (softmax over the slots, as in Slot Attention); a slot changes only by
   the share it claims, through a gate
4. **law**: the slots look at each other and think (one set of weights for every beat and every pass)

The mind only hears what it did not expect (predictive coding). Passes are thinking: `--set settle_eval=6` reads with
twice the thinking, and the log checks whether that helps.

Two lessons of its own, besides the next word: **foresight** (the picture before a beat must imagine that beat; a
picture that holds how things work does this, one that only stores what happened does not) and **explain** (each pass
must explain the beat a little better than the one before, so the passes really settle).

| part | what | size |
|---|---|---|
| ear | 4 word layers, Gnosis's layers but seeing 64 pieces back | |
| mind | 16 slots, 3 passes per 32-piece beat: imagine, claim, law (2 layers, ReLU² 3,840) | 51.4M with the reads |
| voice | 12 more word layers, 64 back, reading the picture before layers 7, 10, 13, 16 | |
| total | | 294.7M (Gnosis 307.5M) |

Kept from Gnosis: SmolLM2's pieces, the tied word table, fused q/k/v, QK-norm, ReLU², value tables, U-net skips, FP8
products, NorMuon, windows 512 -> 2,048, the batch ramp, WSD, the early 3-word lesson. Gnosis's recipe exactly.

## Run it

```bash
python3 -m lucid.tests                                     # CPU checks (~3 min)
OAZY_DATA=/workspace/data bash runpod/train.sh lucid --preset big --tokens 10e9 --micro 32 --data-sub opal --mentor none --name lucid-opal-10b
```

What to watch in the log:

- `[core]` the held-out loss with the mind on and off. Off = only the last 64 pieces are left. If the gap stays near
  zero, the voice is ignoring the picture (what happened to Logos).
- `[loop]` the held-out loss reading with 1, 2, 3 and 6 passes per beat. If 6 beats 3, thinking longer helps.
- train lines `mind_foresee` (how far the picture's guess of the next beat is from it; should fall) and `mind_settled`
  (after the passes; should sit below foresee).

## What usually kills models like this

- **Speed.** The picture is updated beat after beat, one after another. The beats run on only 16 slots (tiny products)
  and one beat is compiled once and reused, but expect it slower per token than Gnosis; first GPU run will tell.
- **The picture gets ignored.** Handled by the short sight: past 64 pieces there is nothing else to read.
- **Exact copying from far back** (a name 500 pieces ago). Transformers are very good at this, models with a fixed
  memory are worse. It shows in the `repeat recall` line.
- **Data, not engines, decides at our size.** Every engine change so far tied with Bee6. Lucid is a bet that forcing a
  picture changes *what* is learned, not just how fast.

Not built yet: keeping the picture between reads in chat (it re-reads the text each time; same numbers, slower), a
picture reset at document starts, Canon layers in the word layers (once Gnosis2 has them).
