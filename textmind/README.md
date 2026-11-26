# Textmind: the creature's mind, for text

The same idea as the `survive` creature, with sentences as moments:

| creature | textmind |
|---|---|
| eyes see a picture every 0.1 s | the reader reads one sentence |
| memory h (GRU 512) | memory h (GRU 512), one step per sentence |
| moment z (32 x 16): guessed before looking, corrected after | same: guessed before reading the sentence, corrected after |
| heads predict its senses (the anchor) | the mouth says the sentence back from (h, z) only (the anchor) |
| dreams 15 steps with the prior | dreams whole sentences with the prior, then the mouth says them |

Phase 1 only (no "wellbeing" for text yet): learn the world of stories. The test is honest:
both minds read exactly the same bytes, and both are scored in **bits per byte** on held-out
stories. For the sentence mind that is mouth bits + surprise (KL) bits, an upper bound on its
true -log2 p(story); for the next-byte transformer it is exact. Same data, batch, steps, optimizer.

Run on the pod (downloads TinyStories once):

    python3 -m textmind.train --arm gpt  --minutes 40 --out /workspace/oazy_runs/textmind/gpt
    python3 -m textmind.train --arm sent --minutes 40 --out /workspace/oazy_runs/textmind/sent
    python3 -m textmind.compare /workspace/oazy_runs/textmind
