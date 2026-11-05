# Tide: a language model that drafts its next thoughts before they arrive

*Working name. Alternatives: Foresight, Herald, Prelude.*

**Tide is a next-word model.** It is scored exactly like GPT-2: the loss of its guess of the very
next word, and HellaSwag by the same rule. What is new is how that guess is made.

```
                       the past                    │ now │          the near future (the belt)
                                                   │     │
  "... she opened the door and"  ──► Engram's      │     │  slot 1     slot 2     slot 3   ...  slot 8
                                     memories ─────┼─────┼─► (next)    (+2)       (+3)          (+8, blank)
                                     (long past)   │     │     ▲ ◄──────── ▲ ◄──────── ▲ ◄─ ... ◄─ ▲
                                                   │     │     the drafts behind each slot are heard
  every new word:  ① the belt moves one place towards now, a blank thought joins at the far end
                   ② each slot gets ONE clean-up step, far end first: own draft + memories + drafts behind
                   ③ slot 1 is the next-word guess
```

A thought is sketched 8 words ahead, cleaned 8 times on its way in, and when it arrives it has
weighed both the past (the memories) and the plans queued behind it (the drafts of the words after
it). That is diffusion spread over time: one clean-up per slot per word, no "roll until it settles".

**Why it should not fail like Basin.** Basin's blank slots were moved by the slope of an energy and
never converged, and nothing forced them to mean anything. In Tide every clean-up is a learned
network applied a fixed number of times, and every slot is trained against the real word it drafts
(slot d against the word d ahead), so the drafts must become real guesses of the future.

**Context**, all fixed-size whatever the text length, no attention anywhere:

| what | how |
|---|---|
| long past | Engram's 12 × 12 learn-while-reading memories (64 × 64 numbers each), the backbone |
| near past | the same memories' fast-fading heads, plus a 4-word convolution |
| near future | the belt of 8 drafts |

**Size.** 121.6M parameters (Engram's 116.8M + a 4.8M clean-up network shared by all slots) vs
GPT-2 small's 124.4M. Each word costs about 1.4 × an Engram word to train.

* `model.py` the model and `draft_report` (how good is the draft of the word d ahead).
* `tests.py` no peeking at future words, word by word == training, the live belt == what training scores.
* `chat.py`, `web/chat.html` talk to it and watch the belt; see how each word's thought arrived.
* Training, data, HellaSwag and the dashboard are Engram's: `python -m engram.train --model tide`.

RunPod guide: `docs/TIDE_RUNPOD.md`. Family: multi-token prediction, rolling diffusion, diffusion
forcing, block diffusion; fast-weight memories (DeltaNet) as in Engram.
