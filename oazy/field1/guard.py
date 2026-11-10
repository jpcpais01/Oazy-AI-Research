"""Field1's walls: what it must never say, checked on every episode as it is written.

Three exams must stay unseen, so Field1 can never teach their exact words:

  Real World  the held-back phrasings of oazy/realworld.py ("is directly ahead of", "who is last in the queue", "next")
  Pro         Real World Pro's new names and things (Abigail, Ambrose, the car, the mug, crimson, the lounge, ...)
  LogT        the 1,600 questions of oazy/logres_v1.jsonl: no run of 8 words from any of them, none of LogT's made-up
              words (blim, dake, ...), and never its "Question:" label

`leaks(text)` lists what an episode says that it must not; the builder writes the episode again with other dice when
the list is not empty (and counts how often, in Field1's meta.json). `report()` is the wall check of the self-tests.
"""
from __future__ import annotations

import functools
import json
import re
from pathlib import Path

NGRAM = 8
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[.?,:!;]")


def words(text: str):
    """The text as the Real World check reads it: lower case, punctuation as words of their own."""
    return _WORD.findall(text.lower())


@functools.lru_cache(maxsize=1)
def held_back():
    """(phrases, single words) that Field1 may never use."""
    from .. import realworld as RW
    phrases = set()
    for fam in list(RW.RULES) + ["__common__"]:
        for _, new in RW._table(fam if fam != "__common__" else "none"):
            p = " ".join(new)
            if p not in RW.SHORTER:
                phrases.add(p)
    single = set()
    for w in RW.PRO_FEMALE + RW.PRO_MALE + RW.VEHICLES + [x for v in RW.PRO_WORDS.values() for x in v]:
        w = w.lower()
        single |= {w, w + "s"} | ({w + "es"} if w.endswith(("s", "sh", "ch", "x")) else set())
    single |= {x for x in logt_words()}
    multi = {p for p in phrases if " " in p}
    single |= {p for p in phrases if " " not in p and p not in RW.SHORTER}
    return frozenset(multi), frozenset(single)


@functools.lru_cache(maxsize=1)
def logt_words():
    from .. import logres
    return frozenset(w.lower() for w in logres.WORDS) | frozenset(w.lower() + "s" for w in logres.WORDS)


@functools.lru_cache(maxsize=1)
def logt_grams():
    """Every run of NGRAM words of LogT's questions (their facts and their question lines)."""
    p = Path(__file__).resolve().parent.parent / "logres_v1.jsonl"
    grams = set()
    if not p.exists():
        return frozenset()
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        ws = words(json.loads(line)["ctx"])
        for i in range(len(ws) - NGRAM + 1):
            grams.add(" ".join(ws[i:i + NGRAM]))
    return frozenset(grams)


@functools.lru_cache(maxsize=1)
def _phrase_re():
    multi, _ = held_back()
    alts = sorted((re.escape(p) for p in multi), key=len, reverse=True)
    return re.compile(r"(?<![a-z0-9'])(?:" + "|".join(alts) + r")(?![a-z0-9'])")


def leaks(text: str):
    """What this text says that Field1 must never say (an empty list = clean)."""
    out = []
    ws = words(text)
    joined = " ".join(ws)
    m = _phrase_re().search(joined)
    if m:
        out.append("Real World: " + m.group(0))
    _, single = held_back()
    bad = single.intersection(ws)
    if bad:
        out += ["Pro/held-back word: " + w for w in sorted(bad)]
    if "question:" in text.lower():
        out.append("LogT label: Question:")
    grams = logt_grams()
    if grams:
        for i in range(len(ws) - NGRAM + 1):
            g = " ".join(ws[i:i + NGRAM])
            if g in grams:
                out.append("LogT 8 words: " + g)
                break
    return out


def report(texts):
    """{what: count} over many texts (the self-tests want it empty)."""
    seen = {}
    for t in texts:
        for x in leaks(t):
            seen[x] = seen.get(x, 0) + 1
    return seen
