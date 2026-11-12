"""The ledger: scenes of plain facts, each told once, asked later. No reasoning is needed to answer, only to have kept.

Field1's scenes ask for reasoning (who is behind whom after three swaps), which a small model on few tokens gets wrong
with the story in full view; an exam made of them alone could not tell forgetting from not understanding. A ledger
scene is the other half of the measurement: a model that can see the story answers nearly always, so every miss after
a gap is a fact that was not kept.

    scene(i, seed) -> {"story": "Rui keeps the coin in the cellar. Mina comes from Porto. ...",
                       "lines": [(lead, answer, choices, kind)]}
        lead + " " + answer + "." is a line of the question block: a question ("Where does Rui keep the coin? Answer:"
        + "the cellar") or the fact told again with its end left open ("Rui keeps the coin in" + "the cellar").
        kind: "facts", or "changed" for a fact the story changed after telling it (the newer value counts).
"""
from __future__ import annotations

import random

NAMES = ("Rui Mina Seth Victor Jonas Amira Yuki Ingrid Rafael Ida Dora Joe Adam Tiago Kofi Ned Rina Helena Ayla Chen Dino "
         "Bianca Ben Julia Emil Vito Gil Beatriz Leo Xenia Pedro Carlos Gina Nell Javier Zoe Elif Diana Paula Hugo Marta "
         "Omar Lena Felix Nora Ivan Clara Tomas Sara Luis Anna Mateo Olga Pablo Rosa Samir Tess Uma Wren Yara").split()
THINGS = "coin key map ring letter watch photo ticket knife candle notebook medal lamp hammer violin kite rope mirror".split()
PLACES = "cellar attic garage shed drawer cupboard wardrobe pantry basket trunk safe kitchen garden library hallway barn".split()
COLOURS = "red blue green yellow black white purple orange brown pink grey silver".split()
CITIES = "Porto Lisbon Madrid Paris Rome Berlin Vienna Prague Dublin Oslo Athens Cairo Tokyo Lima Boston Sydney".split()
FOODS = "rice bread soup cheese pasta fish salad chicken beans pizza honey curry".split()
JOBS = "baker teacher nurse farmer pilot painter lawyer driver singer doctor builder sailor".split()
DAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
PETS = "dog cat rabbit parrot horse turtle goat hamster snake pony".split()

# name -> (pool of values, needs a thing, tellings, question, how the answer is said, the telling with its end left open,
#          how a change is told or None)
RELS = {
    "keeps": (PLACES, True, ["{p} keeps the {x} in the {v}.", "The {x} that belongs to {p} is kept in the {v}."],
              "Where does {p} keep the {x}?", "the {v}", "{p} keeps the {x} in", "Later, {p} moved the {x} to the {v}."),
    "colour": (COLOURS, True, ["{p}'s {x} is {v}.", "{p} has a {v} {x}."],
               "What colour is {p}'s {x}?", "{v}", "The colour of {p}'s {x} is", "Later, {p} painted the {x} {v}."),
    "city": (CITIES, False, ["{p} comes from {v}.", "{p} was born in {v}."],
             "Where does {p} come from?", "{v}", "{p} comes from", None),
    "food": (FOODS, False, ["{p}'s favourite food is {v}.", "More than anything, {p} likes to eat {v}."],
             "What is {p}'s favourite food?", "{v}", "{p}'s favourite food is", "Now {p}'s favourite food is {v}."),
    "job": (JOBS, False, ["{p} works as a {v}.", "{p} is a {v}."],
            "What does {p} work as?", "a {v}", "{p} works as", "Now {p} works as a {v}."),
    "day": (DAYS, False, ["{p}'s appointment is on {v}.", "{p} has an appointment on {v}."],
            "On which day is {p}'s appointment?", "{v}", "{p}'s appointment is on", "{p}'s appointment was moved to {v}."),
    "number": (None, False, ["{p}'s locker number is {v}.", "{p} has locker number {v}."],
               "What is {p}'s locker number?", "{v}", "{p}'s locker number is", "{p}'s locker number was changed to {v}."),
    "pet": (PETS, False, ["{p}'s pet is a {v}.", "{p} has a {v} at home."],
            "What kind of pet does {p} have?", "a {v}", "{p}'s pet is", None),
}
N_CHOICES = 4


def _value(rel, r: random.Random, but=()):
    pool = RELS[rel][0]
    while True:
        v = str(r.randint(10, 99)) if pool is None else r.choice(pool)
        if v not in but:
            return v


def scene(i, seed="ledger") -> dict:
    r = random.Random(f"keep-ledger|{seed}|{i}")
    people = r.sample(NAMES, r.randint(2, 5))
    facts, keys = [], set()                       # [person, relation, thing, value, old value or None]
    for _ in range(r.randint(3, 10)):
        p, rel = r.choice(people), r.choice(list(RELS))
        x = r.choice(THINGS) if RELS[rel][1] else ""
        if (p, rel, x) in keys:
            continue
        keys.add((p, rel, x))
        facts.append([p, rel, x, _value(rel, r), None])
    told = [r.choice(RELS[rel][2]).format(p=p, x=x, v=v) for p, rel, x, v, _ in facts]
    can = [j for j, f in enumerate(facts) if RELS[f[1]][6]]
    changed = None
    if can and r.random() < 0.4:                  # one fact changes after it was told (and is always asked about)
        j = changed = r.choice(can)
        p, rel, x, v, _ = facts[j]
        new = _value(rel, r, but=(v,))
        facts[j][3], facts[j][4] = new, v
        told.insert(r.randint(j + 1, len(told)), RELS[rel][6].format(p=p, x=x, v=new))
    lines = []
    asked = r.sample(range(len(facts)), min(len(facts), r.randint(2, 5)))
    if changed is not None and changed not in asked:
        asked[r.randrange(len(asked))] = changed
    for j in asked:
        p, rel, x, v, old = facts[j]
        pool, _, _, ask, say, cloze, _ = RELS[rel]
        others = [f[3] for f in facts if f[1] == rel and f[3] != v] + ([old] if old else [])
        ch = [v]
        for o in others:
            if o not in ch and len(ch) < N_CHOICES:
                ch.append(o)
        while len(ch) < N_CHOICES:
            ch.append(_value(rel, r, but=ch))
        r.shuffle(ch)
        lead = (ask.format(p=p, x=x) + " Answer:") if r.random() < 0.6 else cloze.format(p=p, x=x)
        lines.append((lead, say.format(v=v), [say.format(v=c) for c in ch], "changed" if old else "facts"))
    return {"story": " ".join(told), "lines": lines}
