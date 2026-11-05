# Fern

**Fern is not a next-word model.** It writes 16 words at a time, coarse to fine, the way a fern
frond unfurls:

```
the passage:   1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
round 1                                                     ●    where is this going?
round 2                             ●                            the middle of the way
round 3                 ●                       ●                the middle of each half
round 4           ●           ●           ●           ●
round 5        ●     ●     ●     ●     ●     ●     ●     ●       the last details
```

One rule, five times: **decide the middle of every stretch whose two ends are known.** All the
middles of a round are decided together, each knowing what is fixed on both sides of it.

- **Reader** (the past): 18 blocks of Engram's learn-while-reading memory over the finished text.
  Memory is fixed-size: old text is kept as gist, recent text in detail.
- **Unfolder** (the passage): 6 blocks where the slots of a round glance at each other, at the
  earlier rounds (the coarser floors of the frond) and at the reader's last 16 words. The glance is
  attention inside a window of at most 47 slots, never over the whole text.
- **Scoring**: every word gets an exact probability in the round it is decided, so the loss is a
  true per-word loss, comparable with GPT-2's, and HellaSwag is scored the usual way (a passage
  starts where the ending starts).

198M parameters. `python -m fern.tests`, `bash runpod/train.sh fern --tokens 2e9`,
`python -m fern.chat --port 8002`. See [`model.py`](model.py) for the full story.
