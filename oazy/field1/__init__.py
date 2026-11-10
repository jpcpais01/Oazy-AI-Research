"""Field1: the world part of Opal, told the way people talk (João, Nov 10).

Opal is our training set: the books, web, maths, code and chat text we read already, plus Field1. Field1 replaces the
old world practice text, whose one or two fixed sentences per kind of event taught the models the wording, not the
logic: during training the Real World exams (the same scenes in other words) fell, and Real World Pro sat at chance.

Underneath, Field1 is still a small world that knows the truth: every scene is simulated, so every answer is right
by construction. On top, it is told in many voices: short stories in the present or the past, facts in any order with
irrelevant details between them, people said as "she" and "him" once named, events told out of order ("Before that,
she had ..."), and the questions asked and answered in ten ways (a plain "Answer:", a short spoken reason, people
talking, a teacher and a class, a puzzle and its solution, a numbered list, thinking it through, ...).

Families (each with its own simulator; the sizes stay inside what the world's practice uses, so the secret exams stay
one step beyond):
    lines      who is in front of whom, after swaps, people going to the front or the back, leaving, joining
    rows       left and right on a shelf or a bench
    compare    taller, older, faster in chains; ages with numbers
    schedule   plans hung on plans, moved together; days of the week
    places     things in containers in containers, moved, carried between rooms, emptied out
    shells     things hidden under cups that are swapped and lifted
    stacks     piles of things; take one out and what was on it comes down
    walks      turning and walking: which way someone faces, where they are
    paths      rooms joined by doors: the way to what someone wants
    family     parents, children, brothers, sisters, grandparents, aunts, uncles, cousins
    minds      people think what they last saw, not what is true
    count      how many, after gifts, finds, losses, buying, eating, sharing
    house      the mixed one: things carried about a house while someone is away
    switches   machines that follow switches, chains of if-then
    rules      if-then chains with animals and weather, what follows and what cannot be told
    sets       all, some and none: what must be true of one member
    clues      "not in the attic, not in the kitchen": what is left; little logic grids
    numbers    sums written out with their carries, take-aways, a few times-tables, money
    machines   a hidden rule found from a few tries; number patterns

Nothing of the three unseen exams is ever written (oazy/field1/guard.py): not the Real World wording, not Real World
Pro's names and things, not LogT's questions. Every episode is checked as it is made, and made again if it slips.

    python3 -m oazy.opal show --family lines --n 3      # read some
    python3 -m oazy.opal build                          # write Field1 into Opal (CPU pod)
"""
from __future__ import annotations

import random

from . import core, guard
from .fam_order import lines, rows, compare, schedule
from .fam_space import places, shells, stacks, walks, paths
from .fam_people import family, minds, count, house
from .fam_logic import switches, rules, sets, clues, numbers, machines

VERSION = 1
GEN = {"lines": lines, "rows": rows, "compare": compare, "schedule": schedule,
       "places": places, "shells": shells, "stacks": stacks, "walks": walks, "paths": paths,
       "family": family, "minds": minds, "count": count, "house": house,
       "switches": switches, "rules": rules, "sets": sets, "clues": clues, "numbers": numbers, "machines": machines}
FAMILIES = tuple(GEN)
# how often each family comes up (the richer scenes a bit more often)
WEIGHTS = {"lines": 7, "rows": 4, "compare": 5, "schedule": 5, "places": 8, "shells": 5, "stacks": 4, "walks": 5,
           "paths": 5, "family": 7, "minds": 7, "count": 7, "house": 6, "switches": 5, "rules": 5, "sets": 5,
           "clues": 6, "numbers": 6, "machines": 5}
TRIES = 20


def episode(family: str, seed) -> tuple:
    """One episode of a family -> (text, tries). Seeded: the same (family, seed) always gives the same text. An episode
    that says anything it must not (guard.leaks) is made again with other dice; `tries` counts the makes."""
    for t in range(TRIES):
        r = random.Random(f"field1|{VERSION}|{family}|{seed}|{t}")
        st, qas = GEN[family](r)
        text = core.frame(r, st.tell(), qas, outsiders=())
        if not guard.leaks(text):
            return text, t + 1
    raise RuntimeError(f"Field1 {family} seed {seed}: every try said something held back: {guard.leaks(text)}")


def pick(seed, k, n):
    """The families of job k: n draws by WEIGHTS."""
    r = random.Random(f"field1-pick|{seed}|{k}")
    fams = list(WEIGHTS)
    return r.choices(fams, weights=[WEIGHTS[f] for f in fams], k=n)


def show(family=None, n=3, seed=0):
    out = []
    for f in ([family] if family else FAMILIES):
        for i in range(n):
            out.append((f, episode(f, (seed, i))[0]))
    return out
