"""The logic warm-up: abstract structure before any words (Opal2's "proc" part, João, Nov 11).

Before a model reads English, it reads a few million pieces of pure procedure: brackets that must close in the right
order, rows of a cellular automaton, lists to sort, stacks to push and pop, sums with their carries. No words, no
facts. The point (Procedural Pretraining, arXiv 2601.21725, Jan 2026) is that attention and the MLPs come out of it
with ready-made circuits for keeping track (what is open, what is on top, which item came before) and for applying a
rule the same way every time, and the language that follows builds on them: GPT-2-sized models reached the same loss
with 55-86% of the text after 2-20M procedural pieces, needle-in-a-haystack went from 10% to 98% after brackets alone,
and shuffling the procedural pieces (same symbols, no structure) took the gain away.

Tasks (each a line; several lines make a document; symbols are spaced so each one is its own piece):
    dyck      a balanced string of four kinds of brackets          ( [ < > ] { } )
    close     the closing brackets an open string still needs      ( [ { => } ] )
    eca       a cellular automaton (rules 30, 90, 110, 184), rows    0 1 1 0 ... / 1 1 0 1 ...
    sort      a list sorted                                          9 2 7 => 2 7 9
    dedup     a list with repeats dropped, first ones kept          b a b c a => b a c
    union     two sets joined, in order of first sight               a c | b c => a c b
    stack     pushes and pops, then the stack from the bottom        push 3 push 5 pop push 2 => 3 2
    reverse   a list backwards                                       x y z => z y x
    add       two numbers added, digit by digit, with the carries    4 7 8 + 9 5 => 5 7 3
    count     how many of one symbol                                 a b a c a # a => 3

Opal2 reads it as pure warm-up for the first 0.2% of the run (20M pieces of a 10B run), then a pinch (0.5%) until 10%
so the circuits are not forgotten while words take over. Built on the CPU in about a minute.

    python3 -m oazy.opal2 build --part proc
    python3 -m oazy.opal2 show --part proc
"""
from __future__ import annotations

import random

VERSION = 1
BRACKETS = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]
LETTERS = list("abcdefghijklmnop")
RULES = (30, 90, 110, 184)
TASKS = {"dyck": 2, "close": 2, "eca": 2, "sort": 2, "dedup": 1, "union": 1, "stack": 2, "reverse": 1, "add": 2,
         "count": 1}


def _dyck(r, n):
    out, st = [], []
    while len(out) < n or st:
        if st and (len(out) + len(st) >= n or r.random() < 0.45):
            out.append(st.pop())
        else:
            o, c = r.choice(BRACKETS)
            out.append(o)
            st.append(c)
    return out


def dyck(r):
    return "dyck: " + " ".join(_dyck(r, r.randint(4, 40)))


def close(r):
    full = _dyck(r, r.randint(4, 30))
    cut = r.randint(1, len(full) - 1)
    st = []
    pairs = dict(BRACKETS)
    for s in full[:cut]:
        if s in pairs:
            st.append(pairs[s])
        else:
            st.pop()
    return "close: " + " ".join(full[:cut]) + " => " + (" ".join(reversed(st)) if st else "-")


def eca(r):
    rule = r.choice(RULES)
    w = r.randint(12, 24)
    row = [r.randint(0, 1) for _ in range(w)]
    rows = [row]
    for _ in range(r.randint(3, 8)):
        row = [(rule >> (row[(i - 1) % w] * 4 + row[i] * 2 + row[(i + 1) % w])) & 1 for i in range(w)]
        rows.append(row)
    return f"eca {rule}: " + " / ".join(" ".join(map(str, x)) for x in rows)


def _items(r):
    if r.random() < 0.5:
        return [str(r.randint(0, 9)) for _ in range(r.randint(2, 14))], int
    return [r.choice(LETTERS) for _ in range(r.randint(2, 14))], str


def sort(r):
    xs, key = _items(r)
    return "sort: " + " ".join(xs) + " => " + " ".join(sorted(xs, key=key))


def dedup(r):
    xs, _ = _items(r)
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return "dedup: " + " ".join(xs) + " => " + " ".join(out)


def union(r):
    a = r.sample(LETTERS, r.randint(1, 6))
    b = r.sample(LETTERS, r.randint(1, 6))
    out = list(a) + [x for x in b if x not in a]
    return "union: " + " ".join(a) + " | " + " ".join(b) + " => " + " ".join(out)


def stack(r):
    ops, st = [], []
    for _ in range(r.randint(2, 14)):
        if st and r.random() < 0.4:
            ops.append("pop")
            st.pop()
        else:
            v = str(r.randint(0, 9))
            ops.append(f"push {v}")
            st.append(v)
    return "stack: " + " ".join(ops) + " => " + (" ".join(st) if st else "-")


def reverse(r):
    xs, _ = _items(r)
    return "reverse: " + " ".join(xs) + " => " + " ".join(reversed(xs))


def add(r):
    a, b = r.randint(0, 10 ** r.randint(1, 6)), r.randint(0, 10 ** r.randint(1, 6))
    return "add: " + " ".join(str(a)) + " + " + " ".join(str(b)) + " => " + " ".join(str(a + b))


def count(r):
    xs = [r.choice(LETTERS[:5]) for _ in range(r.randint(3, 16))]
    x = r.choice(xs)
    return "count: " + " ".join(xs) + " # " + x + " => " + str(xs.count(x))


GEN = {"dyck": dyck, "close": close, "eca": eca, "sort": sort, "dedup": dedup, "union": union, "stack": stack,
       "reverse": reverse, "add": add, "count": count}


def document(seed) -> str:
    """One document: 6-24 lines; a document keeps to one or two tasks, so a model sees a rule several times in a row."""
    r = random.Random(f"proc|{VERSION}|{seed}")
    names = list(TASKS)
    kinds = r.choices(names, weights=[TASKS[k] for k in names], k=r.choice((1, 1, 2)))
    return "\n".join(GEN[r.choice(kinds)](r) for _ in range(r.randint(6, 24)))


def texts(seed, k, n):
    """n documents of job k (the builder's helper call)."""
    return [document((seed, k, i)) for i in range(n)]
