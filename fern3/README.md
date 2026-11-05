# Fern3: Fern that remembers like you do

*Working name.* **Not a next-word model.** Fern3 writes like Fern: 16 words at a time, in 5 rounds,
coarse to fine. What's new is its memory. Memtrix's fixed-size memory (used by Fern and Fern2)
blurs old details. Fern3 instead keeps a gist index plus a shelf of the actual words:

```
the past:  [ p1 ][ p2 ][ p3 ] ... [ p250 ][ p251 ][ p252 ][ p253 ][ p254 ][ p255 ]  [ writing p256 ]
gists:       g1    g2    g3   ...   g250    g251    g252    g253  |  g254    g255
thinking:    ───────── a small model over the gists, in order ────────────►  "where are we?"
recall:            ▲                       ▲            ▲             (top 3 by gist match)
in full:   the 3 recalled passages + the last 2 passages, plus a summary of all gists
```

1. **Fold.** Each finished 16-word passage is folded into one gist, two neighbours at a time
   (16 → 8 → 4 → 2 → 1). This is Fern's unfolding run backwards.
2. **Shelf.** Every old passage is kept word for word. Text is cheap: a book is a few hundred KB.
3. **Think with gists.** A small model reads the gists in order. That's one per 16 words, so a
   4,096-word text is only 256 gists.
4. **Recall.** Before writing a passage, Fern3 asks "what do I need?" and every old gist answers
   with how well it matches. The 3 best passages are read in full, each in proportion to its match.
5. **Unfold.** Fern's own unfolder writes the passage. It looks at the last 2 passages, the 3
   recalled ones and the gist summary, 82 items in all, however long the text is.

**How recall learns which passage to fetch.** This is the hard part, because picking is a yes/no
choice. Three things help:
- The unfolder listens to each recalled passage in proportion to its match. A recall that helped
  raises its gist's score, and one that didn't lowers it.
- **Hindsight.** After a passage is written, the old passage that shares the most words with it is
  the one that would have been worth recalling. Recall's scores are pulled towards it.
- **Warm-up.** For the first 10% of training it recalls 8 passages and listens to each equally, so
  it first learns to use recalled words at all.

The self-tests include a small exercise. A marked passage is hidden somewhere among the first 6,
then repeated 5 to 10 passages later. Fern3 learns to recall the right one (8/64 → 64/64 correct)
and writes its words again from memory (loss 5.7 → 0.1).

**Training** keeps Fern2's fast recipe. The folder and the unfolder grow (folder 2 → 4 → 5 blocks,
unfolder 1 → 2 → 3). The folder also guesses each next word, and GPT-2 medium teaches that guess
through its own small layer. Muon is used throughout. Windows are 4,096 words, so recall has
something to recall, with the same 262k words per step. Size: 117.2M (GPT-2 small is 124M).

**Honest expectations.** The cost per word should be about Fern2's or less. Real text is much
messier than the exercise, so whether recall pays off on FineWeb-Edu is exactly what the first run
will show. Watch "repeat recall" and the per-position loss on the dashboard. The validation loss
uses 4,096-word windows, which gives a slight edge over 1,024-word models.

```bash
python3 -m fern3.tests --device cuda
bash runpod/train.sh fern3 --tokens 2e9
python3 -m fern.chat --run fern3 --port 8002
```
