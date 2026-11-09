"""The world: a small universe of things and links, told in simple English, that answers truthfully after every action.

Every episode is one scene (sentences about things and how they are linked, with fresh random names and colours),
one action that changes it, and a few questions about the scene *after* the action. The world knows the true new
scene, so every answer is right by construction.

World 2 (the default since Nov 9 evening) has fifteen families of laws underneath:

    sums      adding two numbers, with carries                  ("The first number is 0 4 7 2 . ...")
    lines     who stands in front of whom (order chains)         ("Ana stands just in front of Bo .")
    boxes     what is inside what (nesting)                      ("The jar is in the box .")
    family    parents, children, brothers, sisters, both ways    ("Paulo is the father of João .")
    falling   blocks fall until they rest on something           ("The red block is nailed at level 6 .")
    machines  a hidden rule, learned from a few tries            ("Feed it 3 and it gives 5 .")
    cups      things go on existing when hidden and moved        ("The ring is under the red cup .")
    minds     people believe what they last saw, not what is     ("Ana leaves the room . Bo moves the ball ...")
    walks     where you are and which way you face after a walk  ("Rui turns left and walks 2 steps .")
    carry     containers in containers, carried between rooms    ("Ana carries the bag to the attic .")
    time      plans hung on plans; move one, the rest follow      ("The lunch is 2 hours after the market .")
    cause     switches and the machines that follow them          ("The bell is on exactly when the fan is off .")
    wants     the way through the doors to what someone wants     ("Rui wants the key . A door joins ...")
    count     how many, who has more, after gives and losses      ("Ana gives 2 shells to Bo .")
    house     the mixed one: containers, who saw what, the way, coins, in one house, and questions that need
              two of these at once in practice, three at once in the secret exam

World 1 is the first nine with fewer levels; `use(1)` brings it back exactly as it was (its runs carry on in it).

Each family has levels (difficulty) used for training, and SECRET EXAMS that no training episode can come from, by
construction, one step beyond everything practised: longer sums and lines, deeper nesting, bigger falls, family links
chained in ways never asked, machines mixing steps never mixed, longer chains of plans and causes, wants farther
away, more moves in a row, and questions that need three families at once. `tests.py` checks that the walls hold.

Pure Python, seeded: `make(family, level, split, seed)` always gives the same episode.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field

FAMILIES1 = ("sums", "lines", "boxes", "family", "falling", "machines", "cups", "minds", "walks")

# World 1 (Nov 9 morning): the nine families above. Kept exactly as it was, so the runs made on it can still be
# examined and carried on (`use(1)`; worldfirst.run picks it by itself for those runs).
TRAIN_LEVELS1 = {
    "sums": [1, 2, 3, 4, 5, 6, 7, 8],      # digits per number
    "lines": [1, 2, 3, 4, 5, 6],           # people in the line = level + 2 (3..8)
    "boxes": [1, 2, 3, 4, 5, 6],           # boxes around the deepest thing
    "family": [1, 2, 3],                   # links chained: one, two, three
    "falling": [1, 2, 3, 4],               # blocks that move when something lets go
    "machines": [1, 2],                    # one hidden step / two hidden steps
    "cups": [1, 2, 3, 4],                  # moves of the cups (or of what hides under them)
    "minds": [1, 2, 3],                    # things moved while people come and go
    "walks": [1, 2, 3, 4],                 # legs of a walk (turn, then walk)
}
SECRET1 = {
    "sums": {"longer": [9, 10, 11, 12]},
    "lines": {"longer": [7, 8, 9, 10],         # 9..12 people
              "more-moves": [3, 4]},           # 3 or 4 moves in a row (training only ever has 1 or 2); 6 people
    "boxes": {"deeper": [7, 8, 9],
              "more-moves": [3, 4]},           # 3 or 4 moves in a row; 4 boxes deep
    "family": {"new-pairs": [2, 3], "deeper": [4]},   # new-pairs: two links (or three) chained in a way never asked
    "falling": {"bigger": [5, 6, 7]},
    "machines": {"new-mixes": [2], "deeper": [3]},
    "cups": {"more-moves": [6, 7, 8]},
    "minds": {"more-moves": [4, 5, 6]},
    "walks": {"longer": [5, 6, 7]},
}

# World 2 (Nov 9 evening): the races filled World 1's practised kinds in about an hour, so the world was the ceiling.
# Six new families, deeper practice on the old ones, and the secret exams moved one step beyond all of it.
FAMILIES2 = FAMILIES1 + ("carry", "time", "cause", "wants", "count", "house")
TRAIN_LEVELS2 = {
    "sums": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "lines": [1, 2, 3, 4, 5, 6, 7, 8],     # 3..10 people
    "boxes": [1, 2, 3, 4, 5, 6, 7],
    "family": [1, 2, 3],
    "falling": [1, 2, 3, 4, 5],
    "machines": [1, 2],
    "cups": [1, 2, 3, 4, 5, 6],
    "minds": [1, 2, 3, 4],
    "walks": [1, 2, 3, 4, 5, 6],
    "carry": [1, 2, 3, 4],                 # containers around the main thing, while whole containers are carried about
    "time": [1, 2, 3, 4],                  # plans hung on plans: the longest chain of "N hours after/before"
    "cause": [1, 2, 3, 4],                 # switches, and machines that follow them: the longest chain of causes
    "wants": [1, 2, 3, 4],                 # doors between someone and what they want
    "count": [1, 2, 3, 4, 5],              # gives, finds and losses in a row
    "house": [1, 2, 3],                    # the mixed family: moves in a house (questions mix two families)
}
SECRET2 = {
    "sums": {"longer": [11, 12, 13, 14]},
    "lines": {"longer": [9, 10, 11, 12],       # 11..14 people
              "more-moves": [4, 5]},           # training has 1 to 3 moves; 6 people
    "boxes": {"deeper": [8, 9, 10],
              "more-moves": [4, 5]},
    "family": {"new-pairs": [2, 3], "deeper": [4]},
    "falling": {"bigger": [6, 7]},
    "machines": {"new-mixes": [2], "deeper": [3]},
    "cups": {"more-moves": [8, 9, 10]},
    "minds": {"more-moves": [5, 6, 7]},
    "walks": {"longer": [7, 8, 9]},
    "carry": {"deeper": [5, 6], "more-moves": [3, 4]},
    "time": {"longer": [5, 6, 7]},
    "cause": {"deeper": [5, 6, 7]},
    "wants": {"farther": [5, 6, 7]},
    "count": {"more-moves": [7, 8, 9]},
    "house": {"three-way": [2, 3]},            # questions that need three families at once (training only ever mixes two)
}
WORLDS = {1: (FAMILIES1, TRAIN_LEVELS1, SECRET1, (1, 2)),
          2: (FAMILIES2, TRAIN_LEVELS2, SECRET2, (1, 2, 3))}
LATEST = 2


def use(version: int):
    """Switch the whole module to World `version` (1 or 2). Everything reads FAMILIES, TRAIN_LEVELS, SECRET and
    TRAIN_MOVES through this module, so one call changes what every part of a run sees."""
    global VERSION, FAMILIES, TRAIN_LEVELS, SECRET, TRAIN_MOVES
    FAMILIES, TRAIN_LEVELS, SECRET, TRAIN_MOVES = WORLDS[int(version)]
    VERSION = int(version)


use(LATEST)
MOVES_PEOPLE, MOVES_DEPTH = 6, 4   # the size of the scene in the "more-moves" exams of lines and boxes

# How the world speaks. natural=True: people are often "she"/"he"/"her"/"him" once just named, and a sum is written
# smallest digit first (for both models). Set by run.py (--natural 1); the scenes and answers are otherwise the same.
STYLE = {"natural": False}
PRONOUNS = ("she", "he", "her", "him")
_LEAD = {"then", "after", "that"}

# Answers that are not things: yes/no and the ten digits ("class" answers).
CLASSES = ("yes", "no") + tuple(str(i) for i in range(10))
YESNO = (0, 1)
DIGITS = tuple(range(2, 12))

FEMALE = ("Ana Bea Clara Dora Eva Fia Gina Hana Ines Joana Kira Lia Mara Nina Olga Pia Rita Sara Tania Vera "
          "Zoe Alba Bela Cora Dina Elsa Flor Greta Ida Jade Lara Luna Maya Noor Rosa Sofia Tara Uma Wanda Yara").split()
MALE = ("Bo Cy Dan Edu Filipe Gil Hugo Ivo Joao Kai Leo Mau Nuno Omar Paulo Rui Sam Tiago Ugo Vasco "
        "Abel Bruno Caio Dino Emil Fabio Goncalo Henri Igor Jonas Luis Marco Nelson Otto Pedro Raul Simao Tomas Victor Zeca").split()
NAMES = FEMALE + MALE
_FEM, _MAL = set(FEMALE), set(MALE)
COLOURS = "red blue green yellow black white orange purple pink brown grey gold silver".split()
CONTAINERS = "box bag jar cup chest basket crate bowl tin pot case drawer sack vase pouch bucket".split()
ITEMS = "key coin ring ball pen stone shell bead marble button".split()


@dataclass
class Question:
    tokens: list            # the question's words
    mentions: list          # thing id per word, or -1
    answer: list            # the answer's words
    specs: list             # per answer word: (kind, target): ("class", class id) | ("pointer", thing) | ("coldigit", (thing, digit))
    tags: dict = field(default_factory=dict)   # e.g. {"reversed": True}


@dataclass
class Episode:
    family: str
    level: int
    split: str              # "train" or the name of a secret exam
    tokens: list            # scene + action, as words
    mentions: list          # thing id per word, or -1
    things: list            # each thing's key word (what a "who/what" answer says)
    questions: list
    new_scene: str = ""     # the true scene after the action, in words
    facts: dict = field(default_factory=dict)   # the true scene after the action, as structured facts

    def text(self, answers=True) -> str:
        out = list(self.tokens)
        for q in self.questions:
            out += q.tokens + (["answer", ":"] + q.answer + ["."] if answers else [])
        return " ".join(out)


class _Build:
    """Collects words, and which words name which thing."""

    def __init__(self, rng):
        self.rng = rng
        self.toks, self.ment, self.things, self.qs = [], [], [], []
        self.prev = set()                       # things named in the last sentence (for "she", "him")
        self.natural = STYLE["natural"]
        self.prng = random.Random(rng.random()) if self.natural else None   # own dice: the scene stays the same

    def thing(self, key):
        self.things.append(key)
        return len(self.things) - 1

    def _gender(self, t):
        k = self.things[t]
        return "f" if k in _FEM else "m" if k in _MAL else None

    def _emit(self, parts, toks, ment, talk=False):
        start = len(toks)
        for p in parts:
            if isinstance(p, tuple):
                t, words = p
                g = self._gender(t) if talk else None
                if (g and t in self.prev and len(words.split()) == 1 and self.prng.random() < 0.5
                        and not any(self._gender(o) == g for o in self.prev if o != t)):
                    subj = all(w in _LEAD for w in toks[start:])
                    words = {("f", True): "she", ("m", True): "he", ("f", False): "her", ("m", False): "him"}[(g, subj)]
                for w in words.split():
                    toks.append(w)
                    ment.append(t)
            else:
                for w in p.split():
                    toks.append(w)
                    ment.append(-1)

    def say(self, *parts):
        n0 = len(self.toks)
        self._emit(parts, self.toks, self.ment, talk=self.natural)
        self.prev = {m for m in self.ment[n0:] if m >= 0}
        self.toks.append(".")
        self.ment.append(-1)

    def ask(self, parts, answer, specs, **tags):
        toks, ment = [], []
        self._emit(parts, toks, ment)
        toks.append("?")
        ment.append(-1)
        if any(q.tokens == toks for q in self.qs):     # the same question twice tells nothing new
            return
        self.qs.append(Question(toks, ment, answer, specs, tags))

    def ask_class(self, parts, value, **tags):
        value = "yes" if value is True else "no" if value is False else str(value)
        self.ask(parts, [value], [("class", CLASSES.index(value))], **tags)

    def ask_pointer(self, parts, thing, **tags):
        self.ask(parts, [self.things[thing]], [("pointer", thing)], **tags)

    def episode(self, family, level, split, new_scene="", facts=None):
        return Episode(family, level, split, self.toks, self.ment, self.things, self.qs, new_scene, facts or {})


def _shuffled(rng, xs):
    xs = list(xs)
    rng.shuffle(xs)
    return xs


# ---------------------------------------------------------------- sums: carry addition
def gen_sums(rng, level, split):
    L = level
    a = [rng.randrange(1, 10)] + [rng.randrange(10) for _ in range(L - 1)]
    b = [rng.randrange(1, 10)] + [rng.randrange(10) for _ in range(L - 1)]
    A, Bd = [0] + a, [0] + b                      # a leading 0, so the sum has as many digits as the columns
    total = int("".join(map(str, A))) + int("".join(map(str, Bd)))
    S = [int(c) for c in str(total).zfill(L + 1)]
    B = _Build(rng)
    cols = [B.thing(f"column{j}") for j in range(L + 1)]   # column 0 is the leftmost (biggest) one
    first = rng.choice(["the first number is", "on the top line is the number", "write down", "here is the number"])
    second = rng.choice(["the second number is", "on the line below is", "under it write", "the other number is"])
    B.say(first, *[(cols[j], str(d)) for j, d in enumerate(A)])
    B.say(second, *[(cols[j], str(d)) for j, d in enumerate(Bd)])
    B.say(rng.choice(["add the two numbers", "now add them up", "put them together by adding", "add them"]))
    order = list(range(L + 1))
    ask = [rng.choice(["what is the sum", "what number do you get", "what do they make together"])]
    if B.natural:                               # written smallest digit first, as people add
        order.reverse()
        ask += [", smallest digit first"]
    B.ask(ask, [str(S[j]) for j in order], [("coldigit", (cols[j], S[j])) for j in order])
    carries, c = 0, 0
    for j in reversed(range(L + 1)):
        c = (A[j] + Bd[j] + c) // 10
        carries += c
    return B.episode("sums", level, split, "the sum is " + " ".join(map(str, S)),
                     {"a": A, "b": Bd, "sum": S, "carries": carries})


# ---------------------------------------------------------------- lines: order chains
def _moves(rng, level, split):
    """(moves, scene level): training has 1 or 2 moves; the "more-moves" exam has `level` moves."""
    if split == "more-moves":
        return level, None
    return rng.choice(TRAIN_MOVES), level


def gen_lines(rng, level, split):
    moves, lv = _moves(rng, level, split)
    n = (lv + 2) if lv is not None else MOVES_PEOPLE
    B = _Build(rng)
    names = rng.sample(NAMES, n)
    people = [B.thing(x) for x in names]
    order = _shuffled(rng, people)
    sents = []
    for i in range(n - 1):
        x, y = order[i], order[i + 1]
        sents.append(rng.choice([
            [(x, B.things[x]), "stands just in front of", (y, B.things[y])],
            [(y, B.things[y]), "stands just behind", (x, B.things[x])],
            [(y, B.things[y]), "is right behind", (x, B.things[x])],
            [(x, B.things[x]), "is right in front of", (y, B.things[y])],
        ]))
    if rng.random() < 0.5:
        B.say(rng.choice(["some friends wait in a line", "people stand in a line", "there is a line"]))
    for s in _shuffled(rng, sents):
        B.say(*s)
    for m in range(moves):
        pre = [rng.choice(["then", "after that"])] if m else []
        kind = rng.choice(["swap", "front", "back"])
        if kind == "swap":
            x, y = rng.sample(people, 2)
            B.say(*pre, (x, B.things[x]), "and", (y, B.things[y]), "swap places")
            i, j = order.index(x), order.index(y)
            order[i], order[j] = order[j], order[i]
        elif kind == "front":
            x = rng.choice(order[1:])
            B.say(*pre, (x, B.things[x]), rng.choice(["walks to the front of the line", "moves to the very front"]))
            order.remove(x)
            order.insert(0, x)
        else:
            x = rng.choice(order[:-1])
            B.say(*pre, (x, B.things[x]), rng.choice(["goes to the back of the line", "moves to the very back"]))
            order.remove(x)
            order.append(x)
    pos = {p: i for i, p in enumerate(order)}
    nm = lambda p: (p, B.things[p])  # noqa: E731
    for _ in range(12):
        if len(B.qs) == 3:
            break
        q = rng.choice(["ahead", "ahead", "front", "back", "behind", "infront"])
        if q == "ahead":
            if rng.random() < 0.6:          # far apart: needs the whole chain
                i = rng.randrange(0, max(1, n // 2))
                j = rng.randrange(min(n - 1, i + max(2, n // 2)), n) if i + 2 < n else n - 1
            else:
                i, j = rng.sample(range(n), 2)
            x, y = (order[i], order[j]) if rng.random() < 0.5 else (order[j], order[i])
            B.ask_class(["is", nm(x), "ahead of", nm(y)], pos[x] < pos[y])
        elif q == "front":
            B.ask_pointer(["who is at the front of the line"], order[0])
        elif q == "back":
            B.ask_pointer(["who is at the back of the line"], order[-1])
        elif q == "behind":
            x = rng.choice(order[:-1])
            B.ask_pointer(["who stands just behind", nm(x)], order[pos[x] + 1])
        else:
            x = rng.choice(order[1:])
            B.ask_pointer(["who stands just in front of", nm(x)], order[pos[x] - 1])
    return B.episode("lines", level, split, " , ".join(B.things[p] for p in order),
                     {"order": [B.things[p] for p in order], "moves": moves})


# ---------------------------------------------------------------- boxes: inside-of nesting
def gen_boxes(rng, level, split):
    moves, lv = _moves(rng, level, split)
    d = lv if lv is not None else MOVES_DEPTH
    B = _Build(rng)
    table = B.thing("table")
    n_extra = rng.randint(1, 2)
    extra_depths = [rng.randint(1, min(2, d)) for _ in range(n_extra)]
    nouns = rng.sample(CONTAINERS, d + sum(extra_depths))
    items = rng.sample(ITEMS, 1 + n_extra)
    colour = {}
    parent = {}

    def chain(depth, item_word, nouns_):
        boxes = [B.thing(w) for w in nouns_[:depth]]
        it = B.thing(item_word)
        parent[it] = boxes[0]
        for i in range(depth - 1):
            parent[boxes[i]] = boxes[i + 1]
        parent[boxes[-1]] = table
        return it, boxes

    k = 0
    main_item, main = chain(d, items[0], nouns[k:k + d]); k += d
    others = []
    for e, ed in enumerate(extra_depths):
        others.append(chain(ed, items[1 + e], nouns[k:k + ed])); k += ed
    boxes_all = [t for t in range(len(B.things)) if B.things[t] in CONTAINERS]
    for t in boxes_all:
        colour[t] = rng.choice(COLOURS)

    def nm(t):
        if t in colour and rng.random() < 0.5:
            return (t, colour[t] + " " + B.things[t])
        return (t, B.things[t])

    sents = []
    for x, p in parent.items():
        if p == table:
            sents.append(rng.choice([["the", nm(x), "is on the", (table, "table")], ["the", nm(x), "stands on the", (table, "table")]]))
        else:
            sents.append(rng.choice([["the", nm(x), "is in the", nm(p)], ["the", nm(p), "holds the", nm(x)],
                                     ["inside the", nm(p), "is the", nm(x)], ["the", nm(x), "sits inside the", nm(p)]]))
    for s in _shuffled(rng, sents):
        B.say(*s)
    start_parent = dict(parent)

    def inside(x, y):                 # is x (somewhere) inside y?
        p = parent.get(x)
        while p is not None and p != table:
            if p == y:
                return True
            p = parent.get(p)
        return False

    movable = [t for t in parent]
    for m in range(moves):
        pre = [rng.choice(["then", "after that"])] if m else []
        kind = rng.choice(["out", "into", "into"])
        inner = [t for t in movable if parent[t] != table]
        if kind == "out" and inner:
            x = rng.choice(inner)
            B.say(*pre, "take the", nm(x), "out and put it on the", (table, "table"))
            parent[x] = table
            continue
        for _ in range(1000):
            x = rng.choice(movable)
            y = rng.choice(boxes_all)
            if y != x and y != parent[x] and not inside(y, x):
                break
        else:
            raise RuntimeError("no box move found")
        B.say(*pre, rng.choice(["put the", "move the"]), nm(x), "into the", nm(y))
        parent[x] = y

    def top(x):
        p = x
        while parent[p] != table:
            p = parent[p]
        return p

    deep = [t for t in movable if parent[t] != table]
    for _ in range(12):
        if len(B.qs) == 3:
            break
        q = rng.choice(["inside", "inside", "holds", "outer"])
        if q == "inside" or not deep:
            x = main_item if rng.random() < 0.6 else rng.choice(movable)
            want = rng.random() < 0.5
            cands = [y for y in boxes_all if y != x and inside(x, y) == want]
            y = rng.choice(cands) if cands else rng.choice([y for y in boxes_all if y != x])
            B.ask_class(["is the", nm(x), "inside the", nm(y)], inside(x, y))
        elif q == "holds":
            x = rng.choice(movable)
            B.ask_pointer(["what holds the", nm(x), "right now"] if parent[x] != table else ["what is the", nm(x), "on"], parent[x])
        else:
            x = main_item if (parent[main_item] != table and rng.random() < 0.6) else rng.choice(deep)
            B.ask_pointer(["which thing on the table has the", nm(x), "inside it"], top(x))
    new = " ; ".join(f"{B.things[x]} in {B.things[p]}" for x, p in parent.items())
    return B.episode("boxes", level, split, new, {"parent": {B.things[x]: B.things[p] for x, p in parent.items()},
                                                  "start_parent": {B.things[x]: B.things[p] for x, p in start_parent.items()},
                                                  "moves": moves})


# ---------------------------------------------------------------- family: links both ways, chained
FAM_RELS = ("father", "mother", "son", "daughter", "brother", "sister")
FAM_PAIRS = [(a, b) for a in FAM_RELS for b in FAM_RELS]
FAM_HELD_OUT = [("brother", "mother"), ("sister", "father"), ("daughter", "brother"), ("mother", "mother"), ("son", "sister")]
FAM_TRAIN_PAIRS = [p for p in FAM_PAIRS if p not in FAM_HELD_OUT]
_new_link = lambda c: any(tuple(c[i:i + 2]) in FAM_HELD_OUT for i in range(len(c) - 1))  # noqa: E731
FAM_TRIPLES = list(itertools.product(FAM_RELS, repeat=3))
FAM_TRAIN_TRIPLES = [t for t in FAM_TRIPLES if not _new_link(t)]
FAM_NEW_TRIPLES = [t for t in FAM_TRIPLES if _new_link(t)]


def _family_tree(rng, generations=3):
    """People with genders and parents. Returns (gender, parents) dicts over person ids 0..n-1."""
    gender, parents = {}, {}
    nxt = itertools.count()

    def person(g=None):
        i = next(nxt)
        gender[i] = g or rng.choice("mf")
        parents[i] = set()
        return i

    gp = [person("m"), person("f")]
    gen = []
    for _ in range(rng.randint(2, 3)):
        c = person()
        parents[c] = set(gp)
        gen.append(c)
    for g in range(generations - 2):
        new = []
        for p in gen:
            if g >= 1 and len(gender) >= 20:        # the fourth generation stays small, so scenes stay short
                break
            if rng.random() < (0.75 if g == 0 else 0.5):
                spouse = person("f" if gender[p] == "m" else "m") if rng.random() < 0.5 else None
                for _ in range(rng.randint(1, 3 if g == 0 else 2)):
                    c = person()
                    parents[c] = {p} | ({spouse} if spouse is not None else set())
                    new.append(c)
        gen = new or gen
    return gender, parents


def _rel(r, x, gender, parents):
    kids = lambda p: {c for c, ps in parents.items() if p in ps}  # noqa: E731
    if r in ("father", "mother"):
        out = parents[x]
    elif r in ("son", "daughter"):
        out = kids(x)
    else:
        out = {s for p in parents[x] for s in kids(p)} - {x}
    want = "m" if r in ("father", "son", "brother") else "f"
    return {y for y in out if gender[y] == want}


def _chain(rels, x, gender, parents):
    """'the r1 of the r2 of x': apply the last relation first."""
    cur = {x}
    for r in reversed(rels):
        cur = {y for c in cur for y in _rel(r, c, gender, parents)}
    return cur


def gen_family(rng, level, split):
    for _ in range(200):
        ep = _try_family(rng, level, split)
        if ep is not None:
            return ep
    raise RuntimeError("could not build a family episode")


def _try_family(rng, level, split):
    gender, parents = _family_tree(rng, 4 if level >= 3 else 3)
    B = _Build(rng)
    fem, mal = _shuffled(rng, FEMALE), _shuffled(rng, MALE)
    ids = {}
    for p in sorted(gender):
        ids[p] = B.thing((fem if gender[p] == "f" else mal).pop())
    stated = {}                                    # (parent, child) -> "parent-form" | "child-form"
    sents = []
    for c, ps in parents.items():
        for p in ps:
            form = rng.choice(["parent-form", "child-form"])
            stated[(p, c)] = form
            if form == "parent-form":
                sents.append([(ids[p], B.things[ids[p]]), "is the", "father" if gender[p] == "m" else "mother", "of", (ids[c], B.things[ids[c]])])
            else:
                sents.append([(ids[c], B.things[ids[c]]), "is the", "son" if gender[c] == "m" else "daughter", "of", (ids[p], B.things[ids[p]])])
    for s in _shuffled(rng, sents):
        B.say(*s)
    # the action: a baby is born
    cands = [p for p in gender if any(p in ps for ps in parents.values()) or rng.random() < 0.3]
    p = rng.choice(cands)
    co = {q for c, ps in parents.items() if p in ps for q in ps} - {p}
    baby = max(gender) + 1
    gender[baby] = rng.choice("mf")
    parents[baby] = {p} | co
    ids[baby] = B.thing((fem if gender[baby] == "f" else mal).pop())
    word = "boy" if gender[baby] == "m" else "girl"
    nm = lambda q: (ids[q], B.things[ids[q]])  # noqa: E731
    if co:
        q = next(iter(co))
        B.say(nm(p), "and", nm(q), "have a new baby", word, "named", nm(baby))
    else:
        B.say(nm(p), "has a new baby", word, "named", nm(baby))
    people = sorted(gender)

    def phrase(rels, x):
        out = ["who is"]
        for r in rels:
            out += ["the", r, "of"]
        return out + [nm(x)]

    asked = 0
    chains = []
    for _ in range(60):
        asked = len(B.qs)
        if asked == 3:
            break
        depth = level
        if level >= 2 and split == "train" and rng.random() < 0.3:
            depth = level - 1                      # a few shorter chains keep the shorter ones fresh
        if depth == 1:
            r = (rng.choice(FAM_RELS),)
        elif split == "train":
            r = rng.choice(FAM_TRAIN_PAIRS if depth == 2 else FAM_TRAIN_TRIPLES)
        elif split == "new-pairs":
            r = rng.choice(FAM_HELD_OUT if depth == 2 else FAM_NEW_TRIPLES)
        else:
            r = tuple(rng.choice(FAM_RELS) for _ in range(depth))
        x = baby if (asked == 0 and rng.random() < 0.5) else rng.choice(people)
        n_before = len(B.qs)
        ans = _chain(r, x, gender, parents)
        tags = {}
        if len(ans) == 1 and rng.random() < 0.6:
            a = next(iter(ans))
            if depth == 1:
                tags = _rev_tag(r[0], a, x, stated)
            B.ask_pointer(phrase(r, x), ids[a], **tags)
            asked += 1
        elif depth == 1 or len(ans) >= 1:
            truth = rng.random() < 0.5 and len(ans) > 0
            pool = list(ans) if truth else [y for y in people if y not in ans and y != x]
            if not pool:
                continue
            a = rng.choice(pool)
            if depth == 1:
                tags = _rev_tag(r[0], a, x, stated)
            parts = ["is", nm(a)]
            for i, rr in enumerate(r):
                parts += ["the", rr, "of"]
            B.ask_class(parts + [nm(x)], a in ans, **tags)
            asked += 1
        if len(B.qs) > n_before:
            chains.append(list(r))
    if len(B.qs) < 3:
        return None
    links = " ; ".join(f"{B.things[ids[p]]} parent of {B.things[ids[c]]}" for c, ps in parents.items() for p in ps)
    return B.episode("family", level, split, links,
                     {"parents": {B.things[ids[c]]: sorted(B.things[ids[p]] for p in ps) for c, ps in parents.items()},
                      "asked": chains})


def _rev_tag(r, a, x, stated):
    """Was the link asked about stated the other way round? ("C is the son of P", then "who is the father of C?")"""
    if r in ("father", "mother"):
        form = stated.get((a, x))
        return {} if form is None else {"reversed": form == "child-form"}
    if r in ("son", "daughter"):
        form = stated.get((x, a))
        return {} if form is None else {"reversed": form == "parent-form"}
    return {}


# ---------------------------------------------------------------- falling: blocks fall until they rest
def gen_falling(rng, level, split):
    movers = level                         # blocks that fall when the nailed one comes loose (it and what is on it)
    s = movers - 1                         # loose blocks resting on the nailed one
    t = rng.randint(1 if s == 0 else 0, 2)  # blocks stacked on the floor below it (at least two blocks in all)
    hi = 9 - s
    f = rng.randint(t + 1, max(t + 1, min(hi, t + 4)))
    B = _Build(rng)
    floor = B.thing("floor")
    cols = rng.sample(COLOURS, t + 1 + s + 2)
    level_of, on = {}, {}
    nailed = set()
    stack_floor = [B.thing(cols.pop()) for _ in range(t)]
    for i, b in enumerate(stack_floor):
        level_of[b] = i
        on[b] = floor if i == 0 else stack_floor[i - 1]
    F = B.thing(cols.pop())
    level_of[F] = f
    nailed.add(F)
    on[F] = None
    above = [B.thing(cols.pop()) for _ in range(s)]
    prev = F
    for i, b in enumerate(above):
        level_of[b] = f + 1 + i
        on[b] = prev
        prev = b
    top = level_of[prev]
    G = None
    if top + 2 <= 9 and rng.random() < 0.5:   # a second nailed block higher up, which stays put
        G = B.thing(cols.pop())
        level_of[G] = rng.randint(top + 2, 9)
        nailed.add(G)
        on[G] = None
    nm = lambda b: (b, B.things[b] + " block") if b != floor else (floor, "floor")  # noqa: E731
    sents = []
    for b in level_of:
        if b in nailed:
            sents.append(rng.choice([["the", nm(b), "is nailed at level", str(level_of[b])],
                                     ["the", nm(b), "is fixed in place at level", str(level_of[b])]]))
        elif on[b] == floor:
            sents.append(rng.choice([["the", nm(b), "lies on the", nm(floor)], ["the", nm(b), "sits on the", nm(floor)]]))
        else:
            sents.append(rng.choice([["the", nm(b), "rests on the", nm(on[b])], ["the", nm(b), "lies on top of the", nm(on[b])]]))
    if not any(on.get(b) == floor for b in level_of):
        sents.append(["nothing is on the", nm(floor), "yet"])
    for snt in _shuffled(rng, sents):
        B.say(*snt)
    # the action: the nailed block comes loose and falls, with everything on it
    B.say("the", nm(F), rng.choice(["comes loose", "is set free", "is no longer nailed"]))
    land = stack_floor[-1] if stack_floor else floor
    drop = f - t
    nailed.discard(F)
    on[F] = land
    for b in [F] + above:
        level_of[b] -= drop
    blocks = [b for b in level_of]
    for _ in range(12):
        if len(B.qs) == 3:
            break
        q = rng.choice(["level", "level", "on", "above"])
        x = rng.choice(([F] + above) * 2 + blocks)
        if q == "level":
            B.ask_class(["what level is the", nm(x), "at now"], level_of[x])
        elif q == "on" and x not in nailed:
            B.ask_pointer(["what is the", nm(x), "resting on now"], on[x])
        else:
            y = rng.choice([b for b in blocks if b != x])
            B.ask_class(["is the", nm(x), "above the", nm(y), "now"], level_of[x] > level_of[y])
    new = " ; ".join(f"{B.things[b]} at {level_of[b]}" for b in sorted(blocks, key=lambda b: level_of[b]))
    return B.episode("falling", level, split, new, {"level": {B.things[b]: level_of[b] for b in blocks}, "movers": 1 + s,
                                                    "dropped": drop})


# ---------------------------------------------------------------- machines: hidden rules
PRIMS = {
    "+1": lambda x: (x + 1) % 10, "+2": lambda x: (x + 2) % 10, "+3": lambda x: (x + 3) % 10,
    "-1": lambda x: (x - 1) % 10, "x2": lambda x: (2 * x) % 10, "x3": lambda x: (3 * x) % 10,
    "mirror": lambda x: 9 - x,
}
# Held out: six mixes whose rule no other mix (or single step) makes, so each is truly new.
MACH_HELD_OUT = [("+2", "x3"), ("+3", "mirror"), ("-1", "x2"), ("x2", "+3"), ("x3", "mirror"), ("mirror", "+2")]


def _table(steps):
    out = []
    for x in range(10):
        for s in steps:
            x = PRIMS[s](x)
        out.append(x)
    return tuple(out)


def _machine_sets():
    singles = {_table((p,)) for p in PRIMS}
    pairs_train = {_table(p) for p in itertools.permutations(PRIMS, 2) if p not in MACH_HELD_OUT} - singles
    seen = singles | pairs_train
    new_mix = {_table(p) for p in MACH_HELD_OUT} - seen
    triples = {_table(p) for p in itertools.product(PRIMS, repeat=3)} - seen - new_mix
    return {1: sorted(singles), 2: sorted(pairs_train), "new-mixes": sorted(new_mix), "deeper": sorted(triples)}


MACHINES = _machine_sets()
MACH_UNIVERSE = sorted({f for v in MACHINES.values() for f in v})


def gen_machines(rng, level, split):
    pool = MACHINES[level] if split == "train" else MACHINES[split]
    for _ in range(500):
        f = rng.choice(pool)
        n_ex = {1: 4, 2: 5, 3: 6}[level]
        xs = rng.sample(range(10), 10)
        ex, rest = xs[:n_ex], xs[n_ex:]
        queries = rest[:2]
        ok = True
        for _ in range(4):            # add examples until every rule that fits them agrees on the questions
            fits = [g for g in MACH_UNIVERSE if all(g[x] == f[x] for x in ex)]
            if all(g[q] == f[q] for g in fits for q in queries):
                break
            if len(rest) <= 3:
                ok = False
                break
            ex.append(rest.pop())
            queries = rest[:2]
        else:
            ok = False
        if ok:
            break
    B = _Build(rng)
    if rng.random() < 0.5:
        B.say(rng.choice(["a machine changes every number it is fed", "here is a machine with a hidden rule"]))
    for x in ex:                       # each try: what goes in and what comes out are two things
        a, o = B.thing(f"in{x}"), B.thing(f"out{x}")
        B.say(*rng.choice([["it turns", (a, str(x)), "into", (o, str(f[x]))],
                           ["feed it", (a, str(x)), "and it gives", (o, str(f[x]))],
                           ["put in", (a, str(x)), "and out comes", (o, str(f[x]))]]))
    B.say(rng.choice(["now you try new numbers", "you feed it two more numbers", "then you try again"]))
    for q in queries:
        B.ask_class([rng.choice(["what comes out for", "what does it give for", "what do you get for"]), str(q)], f[q])
    return B.episode("machines", level, split, " ".join(f"{x}->{f[x]}" for x in range(10)), {"rule": list(f)})


# ---------------------------------------------------------------- cups: hidden things keep existing
def gen_cups(rng, level, split):
    moves = level
    n = rng.randint(3, 5)
    B = _Build(rng)
    cups = [B.thing(c) for c in rng.sample(COLOURS, n)]
    items = [B.thing(w) for w in rng.sample(ITEMS, rng.randint(1, 2))]
    at = {c: i + 1 for i, c in enumerate(_shuffled(rng, cups))}          # cup -> place 1..n
    under = {it: rng.choice(cups) for it in items}
    cup = lambda c: ["the", (c, B.things[c]), "cup"]  # noqa: E731
    it_ = lambda i: ["the", (i, B.things[i])]  # noqa: E731
    sents = [[*cup(c), "is at place", str(at[c])] for c in cups]
    sents += [[*it_(i), rng.choice(["is under", "hides under"]), *cup(under[i])] for i in items]
    B.say("there are", str(n), "cups in a row")
    for x in _shuffled(rng, sents):
        B.say(*x)
    for m in range(moves):
        pre = [rng.choice(["then", "after that"])] if m else []
        kind = rng.choice(["places", "places", "cups", "item"])
        if kind == "places":
            p, q = rng.sample(range(1, n + 1), 2)
            cp = next(c for c in cups if at[c] == p)
            cq = next(c for c in cups if at[c] == q)
            B.say(*pre, "the cups at place", str(p), "and place", str(q), "swap")
            at[cp], at[cq] = q, p
        elif kind == "cups":
            a, b = rng.sample(cups, 2)
            B.say(*pre, *cup(a), "and", *cup(b), "swap places")
            at[a], at[b] = at[b], at[a]
        else:
            i = rng.choice(items)
            to = rng.choice([c for c in cups if c != under[i]])
            B.say(*pre, "someone lifts", *cup(under[i]), "and moves", *it_(i), "under", *cup(to))
            under[i] = to
    for _ in range(20):
        if len(B.qs) == 3:
            break
        q = rng.choice(["which", "place", "place", "is", "atplace"])
        i = rng.choice(items)
        if q == "which":
            B.ask_pointer(["which cup hides", *it_(i), "now"], under[i])
        elif q == "place":
            B.ask_class(["at which place is", *it_(i), "now"], at[under[i]])
        elif q == "is":
            c = under[i] if rng.random() < 0.5 else rng.choice(cups)
            B.ask_class(["is", *it_(i), "under", *cup(c), "now"], under[i] == c)
        else:
            k = rng.randint(1, n)
            B.ask_pointer(["which cup is at place", str(k), "now"], next(c for c in cups if at[c] == k))
    new = " ; ".join(f"{B.things[c]} at {at[c]}" for c in cups) + " ; " + " ; ".join(f"{B.things[i]} under {B.things[under[i]]}" for i in items)
    return B.episode("cups", level, split, new, {"moves": moves, "at": {B.things[c]: at[c] for c in cups},
                                                 "under": {B.things[i]: B.things[under[i]] for i in items}})


# ---------------------------------------------------------------- minds: people believe what they last saw
PLACES = "box basket drawer cupboard bag chest shelf jar".split()


def gen_minds(rng, level, split):
    moves = level
    B = _Build(rng)
    people = [B.thing(x) for x in rng.sample(NAMES, rng.randint(2, 4))]
    places = [B.thing(w) for w in rng.sample(PLACES, rng.randint(3, 4))]
    objs = [B.thing(w) for w in rng.sample(ITEMS, rng.randint(1, 2))]
    nm = lambda t: (t, B.things[t])  # noqa: E731
    where = {}
    thinks = {p: {} for p in people}
    present = set(people)
    names = [nm(p) for p in people]
    joined = []
    for i, x in enumerate(names):
        joined += ([","] if 0 < i < len(names) - 1 else ["and"] if i else []) + [x]
    B.say(*joined, "are in the room")
    there = []
    for i, pl in enumerate(places):
        there += ([","] if 0 < i < len(places) - 1 else ["and"] if i else []) + ["a", nm(pl)]
    B.say("in the room there is", *there)
    for o in objs:
        p, pl = rng.choice(people), rng.choice(places)
        B.say(nm(p), "puts the", nm(o), "in the", nm(pl))
        where[o] = pl
        for q in present:
            thinks[q][o] = pl
    for m in range(moves):
        if rng.random() < 0.75:                     # someone comes or goes first
            gone = [p for p in people if p not in present]
            if gone and (rng.random() < 0.4 or len(present) == 1):
                p = rng.choice(gone)
                B.say(nm(p), rng.choice(["comes back", "comes back into the room", "returns"]))
                present.add(p)
            elif len(present) > 1:
                p = rng.choice(sorted(present))
                B.say(nm(p), rng.choice(["leaves the room", "goes out", "walks out of the room"]))
                present.discard(p)
        mover = rng.choice(sorted(present))
        o = rng.choice(objs)
        to = rng.choice([pl for pl in places if pl != where[o]])
        if rng.random() < 0.5:
            B.say(nm(mover), "moves the", nm(o), "from the", nm(where[o]), "to the", nm(to))
        else:
            B.say(nm(mover), *rng.choice([["moves the", nm(o), "to the"], ["puts the", nm(o), "into the"]]), nm(to))
        where[o] = to
        for q in present:
            thinks[q][o] = to
    wrong = [(p, o) for p in people for o in objs if thinks[p][o] != where[o]]
    for _ in range(30):
        if len(B.qs) == 3:
            break
        q = rng.choice(["thinks", "thinks", "where", "knows"])
        p, o = rng.choice(wrong) if (wrong and rng.random() < 0.6) else (rng.choice(people), rng.choice(objs))
        if q == "thinks":
            B.ask_pointer(["where does", nm(p), "think the", nm(o), "is"], thinks[p][o], belief=thinks[p][o] != where[o])
        elif q == "where":
            B.ask_pointer(["where is the", nm(o), "really"], where[o])
        else:
            B.ask_class(["does", nm(p), "know where the", nm(o), "is"], thinks[p][o] == where[o])
    new = " ; ".join(f"{B.things[o]} in {B.things[where[o]]}" for o in objs)
    return B.episode("minds", level, split, new, {"moves": moves, "where": {B.things[o]: B.things[where[o]] for o in objs},
                                                  "thinks": {B.things[p]: {B.things[o]: B.things[v] for o, v in d.items()}
                                                             for p, d in thinks.items()}})


# ---------------------------------------------------------------- walks: where you are and which way you face
DIRS = ("north", "east", "south", "west")
STEP = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}
HORIZON = ("mountains", "sea", "forest", "desert")
LANDMARKS = "bakery church school bridge tower farm mill shop inn gate fountain barn".split()


def gen_walks(rng, level, split):
    legs = level
    for _ in range(1000):
        head = rng.randrange(4)
        h, x, y, plan, path = head, 0, 0, [], [(0, 0)]
        for i in range(legs):
            turn = rng.choice(["left", "right", "around"] + ([None] if i == 0 else []))
            h = (h + {"left": 3, "right": 1, "around": 2, None: 0}[turn]) % 4
            k = rng.randint(1, 3)
            x, y = x + STEP[h][0] * k, y + STEP[h][1] * k
            plan.append((turn, k))
            path.append((x, y))
        if all(abs(a) <= 9 and abs(b) <= 9 for a, b in path):
            break
    B = _Build(rng)
    walker = B.thing(rng.choice(NAMES))
    well = B.thing("well")
    hz = [B.thing(w) for w in HORIZON]
    side = dict(zip(_shuffled(rng, range(4)), hz))     # direction -> what lies that way
    end = (x, y)
    spots = {(0, 0): well}
    marks = rng.sample(LANDMARKS, 4)
    cands = [p for p in set(path[1:]) if p != (0, 0)]
    if end != (0, 0):
        spots[end] = B.thing(marks.pop())
        cands = [p for p in cands if p != end]
    for p in rng.sample(cands, min(len(cands), rng.randint(1, 2))):
        spots[p] = B.thing(marks.pop())
    while len(spots) < 3:
        p = (rng.randint(-4, 4), rng.randint(-4, 4))
        if p not in spots:
            spots[p] = B.thing(marks.pop())
    nm = lambda t: (t, B.things[t])  # noqa: E731
    sents = [["the", nm(side[d]), "lie" if B.things[side[d]] == "mountains" else "lies", "to the", DIRS[d]] for d in range(4)]
    for (px, py), t in spots.items():
        if t == well:
            continue
        parts = []
        if py:
            parts += [str(abs(py)), "step" if abs(py) == 1 else "steps", "north" if py > 0 else "south"]
        if px:
            parts += (["and"] if parts else []) + [str(abs(px)), "step" if abs(px) == 1 else "steps", "east" if px > 0 else "west"]
        sents.append(["the", nm(t), "is", *parts, "of the", nm(well)])
    for snt in _shuffled(rng, sents):
        B.say(*snt)
    B.say(nm(walker), "stands at the", nm(well), "facing", *(rng.choice([[DIRS[head]], ["the", nm(side[head])]])))
    for i, (turn, k) in enumerate(plan):
        pre = [rng.choice(["then", "after that"])] if i else []
        go = ["walks", str(k), "step" if k == 1 else "steps", rng.choice(["forward", "ahead"])]
        if turn is None:
            B.say(*pre, nm(walker), *go)
        else:
            B.say(*pre, nm(walker), "turns", turn, "and", *go)
    where_ = spots.get(end)
    for _ in range(30):
        if len(B.qs) == 3:
            break
        q = rng.choice(["where", "where", "face", "dir"])
        if q == "where" and where_ is not None:
            B.ask_pointer(["where is", nm(walker), "now"], where_)
        elif q == "face":
            B.ask_pointer(["what does", nm(walker), "face now"], side[h])
        else:
            (px, py), t = rng.choice(list(spots.items()))
            d = rng.randrange(4)
            ahead = {0: y > py, 1: x > px, 2: y < py, 3: x < px}[d]
            B.ask_class(["is", nm(walker), DIRS[d], "of the", nm(t), "now"], ahead)
    return B.episode("walks", level, split, f"at {end}, facing {DIRS[h]}", {"legs": legs, "end": list(end), "facing": DIRS[h]})


# ================================================================ World 2's new families
ROOMS = "kitchen hall cellar attic garden porch study library bedroom bathroom garage pantry yard barn loft shed".split()
PLURAL = {w: w + ("es" if w.endswith("s") or w.endswith("sh") else "s") for w in ITEMS}


def _rooms_tree(rng, names):
    """Rooms joined by doors into a tree (one way between any two rooms). Returns {room: set of next rooms}."""
    nb = {r: set() for r in names}
    for i in range(1, len(names)):
        j = rng.randrange(i)
        nb[names[i]].add(names[j])
        nb[names[j]].add(names[i])
    return nb


def _path(nb, a, b):
    """The rooms from a to b, both ends included (the only way, as the doors make a tree)."""
    prev, todo = {a: None}, [a]
    while todo:
        x = todo.pop(0)
        for y in nb[x]:
            if y not in prev:
                prev[y] = x
                todo.append(y)
    out = [b]
    while out[-1] != a:
        out.append(prev[out[-1]])
    return out[::-1]


def _say_doors(B, rng, nb, nm):
    pairs = sorted({tuple(sorted((a, b))) for a in nb for b in nb[a]})
    sents = []
    for a, b in pairs:
        a, b = (a, b) if rng.random() < 0.5 else (b, a)
        sents.append(rng.choice([["a door joins the", nm(a), "and the", nm(b)],
                                 ["the", nm(a), "has a door to the", nm(b)],
                                 ["from the", nm(a), "a door leads to the", nm(b)]]))
    return sents


def _and_list(parts):
    out = []
    for i, x in enumerate(parts):
        out += ([","] if 0 < i < len(parts) - 1 else ["and"] if i else []) + list(x)
    return out


# ---------------------------------------------------------------- carry: containers inside containers, carried about
CARRY_MOVES = (1, 2)
CARRY_CAP = 4          # in training and in "more-moves", nothing is ever more than 4 containers deep


def gen_carry(rng, level, split):
    """Things in containers in containers, in rooms. Someone carries a whole container to another room, or puts one
    thing into another: what is inside goes along. Asked: which room something is in now, what holds it."""
    if split == "more-moves":
        depth, moves = 3, level
    else:
        depth, moves = level, rng.choice(CARRY_MOVES)
    cap = depth if split == "deeper" else CARRY_CAP
    B = _Build(rng)
    nm = lambda t: (t, B.things[t])  # noqa: E731
    rooms = [B.thing(r) for r in rng.sample(ROOMS, rng.randint(2, 4))]
    people = [B.thing(x) for x in rng.sample(NAMES, rng.randint(1, 2))]
    nouns = rng.sample(CONTAINERS, depth + rng.randint(1, 2))
    words = rng.sample(ITEMS, 2)
    boxes = [B.thing(w) for w in nouns]
    main, other = B.thing(words[0]), B.thing(words[1])
    parent = {main: boxes[0]}
    for i in range(depth - 1):
        parent[boxes[i]] = boxes[i + 1]
    parent[boxes[depth - 1]] = rng.choice(rooms)
    for b in boxes[depth:]:
        parent[b] = rng.choice(rooms)
    parent[other] = rng.choice(boxes[depth:] + rooms)
    pos = {p: rng.choice(rooms) for p in people}
    is_box = set(boxes)

    def up(x):                         # x, then what holds it, and so on, ending with its room
        out = [x]
        while out[-1] not in rooms:
            out.append(parent[out[-1]])
        return out

    room_of = lambda x: up(x)[-1]  # noqa: E731
    nest = lambda x: len(up(x)) - 2   # containers around x  # noqa: E731
    sents = []
    for x, p in parent.items():
        if p in rooms:
            sents.append(rng.choice([["the", nm(x), "is in the", nm(p)], ["the", nm(x), "stands in the", nm(p)]]
                                    if x in is_box else [["the", nm(x), "lies on the floor of the", nm(p)]]))
        else:
            sents.append(rng.choice([["the", nm(x), "is in the", nm(p)], ["the", nm(p), "holds the", nm(x)],
                                     ["inside the", nm(p), "is the", nm(x)]]))
    sents += [[nm(p), "is in the", nm(pos[p])] for p in people]
    for snt in _shuffled(rng, sents):
        B.say(*snt)
    deepest = max(nest(x) for x in parent)
    moved_boxes = 0
    for m in range(moves):
        pre = [rng.choice(["then", "after that"])] if m else []
        who = rng.choice(people)
        for _ in range(200):
            if rng.random() < 0.6:     # carry: a whole container (often one the main thing is in) goes to another room
                x = rng.choice([b for b in up(main)[1:-1]] or boxes) if rng.random() < 0.6 else rng.choice(list(parent))
                to = rng.choice([r for r in rooms if r != room_of(x)] or [None])
                if to is None:
                    continue
                if parent[x] in rooms:
                    B.say(*pre, nm(who), "carries the", nm(x), "to the", nm(to))
                else:
                    B.say(*pre, nm(who), "takes the", nm(x), "out of the", nm(parent[x]), "and carries it to the", nm(to))
                parent[x] = to
                pos[who] = to
                moved_boxes += x in is_box
                break
            x, y = rng.choice(list(parent)), rng.choice(boxes)
            if x == y or parent[x] == y or x in up(y):
                continue
            old = parent[x]
            parent[x] = y
            if max(nest(t) for t in parent) > cap:       # never deeper than the level allows
                parent[x] = old
                continue
            B.say(*pre, nm(who), "puts the", nm(x), "into the", nm(y))
            pos[who] = room_of(y)
            break
        deepest = max(deepest, max(nest(t) for t in parent))
    for _ in range(30):
        if len(B.qs) == 3:
            break
        q = rng.choice(["room", "room", "room", "isin", "holds", "who"])
        x = main if rng.random() < 0.5 else rng.choice([t for t in parent if nest(t) > 0] or list(parent))
        if q == "room":
            B.ask_pointer(["which room is the", nm(x), "in now"], room_of(x))
        elif q == "isin":
            r = room_of(x) if rng.random() < 0.5 else rng.choice(rooms)
            B.ask_class(["is the", nm(x), "in the", nm(r), "now"], room_of(x) == r)
        elif q == "holds" and parent[x] not in rooms:
            B.ask_pointer(["what holds the", nm(x), "right now"], parent[x])
        elif q == "who":
            p = rng.choice(people)
            B.ask_pointer(["which room is", nm(p), "in now"], pos[p])
    new = " ; ".join(f"{B.things[x]} in {B.things[p]}" for x, p in parent.items())
    return B.episode("carry", level, split, new, {"depth": depth, "deepest": deepest, "moves": moves,
                                                  "room": {B.things[x]: B.things[room_of(x)] for x in parent}})


# ---------------------------------------------------------------- time: plans hung on plans, and one of them moves
EVENTS = "breakfast lunch dinner market concert lesson game party walk nap swim meeting show picnic race dance".split()


def gen_time(rng, level, split):
    """Events at hours 0..9: some at a set hour, the others planned N hours after or before another event. One event
    moves, and what is planned from it moves with it. Asked: hours, which comes first, how far apart."""
    for _ in range(2000):
        n_extra = rng.randint(1, 3)
        names = rng.sample(EVENTS, level + 1 + n_extra + rng.randint(0, 1))
        ref, off, depth = {}, {}, {}
        chain = names[:level + 1]
        ref[chain[0]], depth[chain[0]] = None, 0
        for i in range(1, level + 1):
            ref[chain[i]], depth[chain[i]] = chain[i - 1], i
        for e in names[level + 1:]:
            if rng.random() < 0.35:
                ref[e], depth[e] = None, 0
            else:
                p = rng.choice([x for x in ref if depth[x] < level])
                ref[e], depth[e] = p, depth[p] + 1
        hour = {}
        for e in ref:                  # parents come first: names are added in order
            if ref[e] is None:
                hour[e] = rng.randint(0, 9)
            else:
                off[e] = rng.choice([-3, -2, -1, 1, 2, 3])
                hour[e] = hour[ref[e]] + off[e]
        if not all(0 <= h <= 9 for h in hour.values()):
            continue
        moved = rng.choice([chain[rng.randrange(level)]] + [e for e in ref if rng.random() < 0.3])
        shift = rng.choice([-3, -2, -1, 1, 2, 3])
        tied = {moved}
        for e in ref:
            if ref[e] in tied:
                tied.add(e)
        after = {e: hour[e] + (shift if e in tied else 0) for e in ref}
        if all(0 <= h <= 9 for h in after.values()) and chain[-1] in tied:
            break
    else:
        raise RuntimeError("no time episode found")
    B = _Build(rng)
    ids = {e: B.thing(e) for e in ref}
    nm = lambda e: (ids[e], e)  # noqa: E731
    sents = []
    for e in ref:
        if ref[e] is None:
            sents.append(rng.choice([["the", nm(e), "is at hour", str(hour[e])], ["the", nm(e), "starts at hour", str(hour[e])]]))
        else:
            k = abs(off[e])
            sents.append(["the", nm(e), "is", str(k), "hour" if k == 1 else "hours", "after" if off[e] > 0 else "before",
                          "the", nm(ref[e])])
    for snt in _shuffled(rng, sents):
        B.say(*snt)
    k = abs(shift)
    B.say("the", nm(moved), "moves", str(k), "hour" if k == 1 else "hours", "later" if shift > 0 else "earlier",
          ", and what is planned from it moves too")
    deep = [e for e in ref if depth[e] == level]
    evs = list(ref)
    for _ in range(40):
        if len(B.qs) == 3:
            break
        q = rng.choice(["hour", "hour", "before", "first", "gap"])
        x = rng.choice(deep) if rng.random() < 0.6 else rng.choice(evs)
        y = rng.choice([e for e in evs if e != x])
        if q == "hour":
            B.ask_class(["at what hour is the", nm(x), "now"], after[x])
        elif after[x] == after[y]:
            continue
        elif q == "before":
            B.ask_class(["is the", nm(x), "before the", nm(y), "now"], after[x] < after[y])
        elif q == "first":
            a, b = (x, y) if rng.random() < 0.5 else (y, x)
            B.ask_pointer(["which comes first now , the", nm(a), "or the", nm(b)], ids[min((a, b), key=after.get)])
        else:
            a, b = sorted((x, y), key=after.get)
            B.ask_class(["how many hours after the", nm(a), "is the", nm(b), "now"], after[b] - after[a])
    return B.episode("time", level, split, " ; ".join(f"{e} at {after[e]}" for e in sorted(ref, key=after.get)),
                     {"depth": max(depth.values()), "hour": dict(after), "moved": moved})


# ---------------------------------------------------------------- cause: switches, and machines that follow them
DEVICES = "lamp fan bell pump motor heater radio alarm siren horn clock buzzer".split()


def gen_cause(rng, level, split):
    """Switches, and machines that are on exactly when what they follow is: the same, the opposite, both, either.
    Someone flips a switch; the effect runs down the chain. Asked: what is on now, and what changed."""
    for _ in range(500):
        B = _Build(rng)
        nm = lambda t: (t, B.things[t])  # noqa: E731
        sw = [B.thing(c) for c in rng.sample(COLOURS, rng.randint(2, 3))]
        devs = rng.sample(DEVICES, level + rng.randint(1, 3))
        rule, depth = {}, {s: 0 for s in sw}
        parts = list(sw)
        for i, w in enumerate(devs):
            d = B.thing(w)
            if i < level:              # the main chain: each one follows the one before
                first = parts[-1] if i else rng.choice(sw)
            else:
                first = rng.choice([p for p in parts if depth[p] < level])
            ins = [first]
            if rng.random() < 0.45:
                second = rng.choice([p for p in parts if p != first and depth[p] < level] or [None])
                if second is not None:
                    ins.append(second)
            op = rng.choice(["same", "not"]) if len(ins) == 1 else rng.choice(["and", "or"])
            rule[d] = (op, ins)
            depth[d] = 1 + max(depth[p] for p in ins)
            parts.append(d)
        state0 = {s: rng.random() < 0.5 for s in sw}
        flips = [rng.choice(sw)] + ([rng.choice(sw)] if rng.random() < 0.3 else [])

        def run(state):
            v = dict(state)
            for d, (op, ins) in rule.items():
                a = [v[p] for p in ins]
                v[d] = {"same": a[0], "not": not a[0], "and": all(a), "or": any(a)}[op]
            return v
        before = run(state0)
        state1 = dict(state0)
        for s in flips:
            state1[s] = not state1[s]
        now = run(state1)
        top = [d for d in rule if depth[d] == level]
        if top and any(now[d] != before[d] for d in rule):
            break
    sname = lambda s: ["the", nm(s), "switch"]  # noqa: E731
    sents = [[*sname(s), "is", "on" if state0[s] else "off"] for s in sw]
    for d, (op, ins) in rule.items():
        src = lambda p: sname(p) if p in state0 else ["the", nm(p)]  # noqa: E731
        if op in ("same", "not"):
            sents.append(["the", nm(d), "is on exactly when", *src(ins[0]), "is", "on" if op == "same" else "off"])
        else:
            sents.append(["the", nm(d), "is on exactly when", *src(ins[0]), "and" if op == "and" else "or", *src(ins[1]),
                          "are both on" if op == "and" else "is on"])
    for snt in _shuffled(rng, sents):
        B.say(*snt)
    who = B.thing(rng.choice(NAMES))
    for i, s in enumerate(flips):
        B.say(*([rng.choice(["then", "after that"])] if i else []), nm(who), "flips", *sname(s))
    devices = list(rule)
    for _ in range(40):
        if len(B.qs) == 3:
            break
        d = rng.choice(top) if rng.random() < 0.5 else rng.choice(devices)
        q = rng.choice(["on", "on", "changed"])
        if q == "on":
            B.ask_class(["is the", nm(d), "on now"], now[d])
        else:
            B.ask_class(["did the", nm(d), "change"], now[d] != before[d])
    return B.episode("cause", level, split, " ; ".join(f"{B.things[d]} {'on' if now[d] else 'off'}" for d in devices),
                     {"depth": max(depth.values()), "on": {B.things[d]: now[d] for d in devices}})


# ---------------------------------------------------------------- wants: the way to what someone wants
def gen_wants(rng, level, split):
    """Rooms joined by doors. Someone wants a thing; then the thing is carried somewhere else (or they walk off).
    Asked: where they must go, which room they walk into first, how many doors it takes, whether a room is on the way."""
    dist = level
    B = _Build(rng)
    nm = lambda t: (t, B.things[t])  # noqa: E731
    names = rng.sample(ROOMS, dist + 1 + rng.randint(1, 3))
    rooms = [B.thing(r) for r in names]
    path = rooms[:dist + 1]            # the way, after the action, from where they stand to the thing they want
    nb = {r: set() for r in rooms}
    for i in range(1, dist + 1):
        nb[path[i]].add(path[i - 1])
        nb[path[i - 1]].add(path[i])
    for r in rooms[dist + 1:]:
        o = rng.choice([x for x in rooms if nb[x] or x == path[0]])
        nb[r].add(o)
        nb[o].add(r)
    wanter, helper = (B.thing(x) for x in rng.sample(NAMES, 2))
    item = B.thing(rng.choice(ITEMS))
    if rng.random() < 0.5:             # the thing is carried to the far end
        start_item = rng.choice([r for r in rooms if r != path[-1]])
        start_me = path[0]
        action = [nm(helper), "carries the", nm(item), "to the", nm(path[-1])]
    else:                              # or the one who wants it walks off to the far end
        start_item = path[-1]
        start_me = rng.choice([r for r in rooms if r != path[0] and r != path[-1]])
        action = [nm(wanter), "walks to the", nm(path[0])]
    sents = _say_doors(B, rng, nb, nm)
    sents += [[nm(wanter), "is in the", nm(start_me)], ["the", nm(item), "is in the", nm(start_item)]]
    if start_item == path[-1] or rng.random() < 0.5:
        sents.append([nm(helper), "is in the", nm(rng.choice(rooms))])
    for snt in _shuffled(rng, sents):
        B.say(*snt)
    B.say(nm(wanter), rng.choice(["wants the", "would like the", "is looking for the"]), nm(item))
    B.say(*action)
    for _ in range(30):
        if len(B.qs) == 3:
            break
        q = rng.choice(["go", "first", "first", "doors", "via"])
        if q == "go":
            B.ask_pointer(["which room must", nm(wanter), "go to for the", nm(item)], path[-1])
        elif q == "first":
            B.ask_pointer(["which room must", nm(wanter), "walk into first to get the", nm(item)], path[1])
        elif q == "doors":
            B.ask_class(["how many doors must", nm(wanter), "go through to get the", nm(item)], dist)
        else:
            r = rng.choice(path[1:-1] or rooms) if rng.random() < 0.5 else rng.choice(rooms)
            if r in (path[0], path[-1]):
                continue
            B.ask_class(["must", nm(wanter), "go through the", nm(r), "to get the", nm(item)], r in path)
    return B.episode("wants", level, split, " , ".join(B.things[r] for r in path),
                     {"dist": dist, "path": [B.things[r] for r in path],
                      "doors": sorted({tuple(sorted((B.things[a], B.things[b]))) for a in nb for b in nb[a]})})


# ---------------------------------------------------------------- count: how many, and who has more
def gen_count(rng, level, split):
    """People with some things each; then they give, find and lose some. Asked: how many someone has now, who has
    more, how many more, and how many things in all."""
    moves = level
    for _ in range(500):
        people = rng.sample(NAMES, rng.randint(2, 3))
        kinds = rng.sample(ITEMS, rng.randint(1, 2))
        have = {p: {k: rng.randint(0, 5) for k in kinds} for p in people}
        start = {p: dict(v) for p, v in have.items()}
        acts, ok = [], True
        for _m in range(moves):
            k = rng.choice(kinds)
            kind = rng.choice(["give", "give", "find", "lose"])
            p = rng.choice(people)
            if kind == "give":
                q = rng.choice([x for x in people if x != p])
                if have[p][k] == 0 or have[q][k] >= 9:
                    ok = False
                    break
                n = rng.randint(1, min(have[p][k], 9 - have[q][k], 4))
                have[p][k] -= n
                have[q][k] += n
                acts.append(("give", p, n, k, q))
            elif kind == "find":
                if have[p][k] >= 9:
                    ok = False
                    break
                n = rng.randint(1, min(9 - have[p][k], 3))
                have[p][k] += n
                acts.append(("find", p, n, k, None))
            else:
                if have[p][k] == 0:
                    ok = False
                    break
                n = rng.randint(1, min(have[p][k], 3))
                have[p][k] -= n
                acts.append(("lose", p, n, k, None))
        if ok and all(sum(v.values()) <= 9 for v in have.values()):
            break
    else:
        raise RuntimeError("no count episode found")
    B = _Build(rng)
    ids = {p: B.thing(p) for p in people}
    nm = lambda p: (ids[p], p)  # noqa: E731
    num = lambda n, k: ["no", PLURAL[k]] if n == 0 else [str(n), k if n == 1 else PLURAL[k]]  # noqa: E731
    for p in _shuffled(rng, people):
        B.say(nm(p), "has", *_and_list([num(start[p][k], k) for k in kinds]))
    for i, (kind, p, n, k, q) in enumerate(acts):
        pre = [rng.choice(["then", "after that"])] if i else []
        if kind == "give":
            B.say(*pre, nm(p), "gives", *num(n, k), "to", nm(q))
        elif kind == "find":
            B.say(*pre, nm(p), "finds", *num(n, k))
        else:
            B.say(*pre, nm(p), "loses", *num(n, k))
    for _ in range(40):
        if len(B.qs) == 3:
            break
        q = rng.choice(["many", "many", "more", "morethan", "howmuch", "all"])
        k = rng.choice(kinds)
        p = rng.choice(people)
        o = rng.choice([x for x in people if x != p])
        if q == "many":
            B.ask_class(["how many", PLURAL[k], "does", nm(p), "have now"], have[p][k])
        elif q == "all":
            if len(kinds) > 1:
                B.ask_class(["how many things does", nm(p), "have now"], sum(have[p].values()))
        elif have[p][k] == have[o][k]:
            continue
        elif q == "more":
            a, b = (p, o) if rng.random() < 0.5 else (o, p)
            B.ask_pointer(["who has more", PLURAL[k], "now ,", nm(a), "or", nm(b)], ids[max((a, b), key=lambda x: have[x][k])])
        elif q == "morethan":
            B.ask_class(["does", nm(p), "have more", PLURAL[k], "than", nm(o), "now"], have[p][k] > have[o][k])
        else:
            a, b = sorted((p, o), key=lambda x: -have[x][k])
            B.ask_class(["how many more", PLURAL[k], "does", nm(a), "have than", nm(b), "now"], have[a][k] - have[b][k])
    return B.episode("count", level, split, " ; ".join(f"{p} {have[p]}" for p in people),
                     {"moves": moves, "have": have})


# ---------------------------------------------------------------- house: the mixed family
# Each question needs some of four families at once: C containers (what is inside what, carried along), M minds (who
# saw which move), W wants (the way through the doors to what someone wants), N counting (coins in containers).
# Training asks questions that need one or two of them; the secret "three-way" exam asks only ones that need three.
HOUSE_TRAIN_KINDS = ("C", "CN", "CM", "M", "CW", "W", "MW")
HOUSE_SECRET_KINDS = ("CMW", "CMN")
HOUSE_THINGS = [w for w in ITEMS if w != "coin"]


def gen_house(rng, level, split):
    kinds = HOUSE_SECRET_KINDS if split == "three-way" else HOUSE_TRAIN_KINDS
    for _ in range(300):
        ep = _try_house(rng, level, split, kinds)
        if ep is not None:
            return ep
    raise RuntimeError("no house episode found")


def _try_house(rng, level, split, allowed):
    B = _Build(rng)
    nm = lambda t: (t, B.things[t])  # noqa: E731
    rooms = [B.thing(r) for r in rng.sample(ROOMS, rng.randint(3, 5))]
    nb = _rooms_tree(rng, rooms)
    people = [B.thing(x) for x in rng.sample(NAMES, rng.randint(2, 3))]
    boxes = [B.thing(w) for w in rng.sample(CONTAINERS, rng.randint(2, 4))]
    items = [B.thing(w) for w in rng.sample(HOUSE_THINGS, rng.randint(1, 2))]
    parent, coins = {}, {}
    for i, b in enumerate(boxes):
        inner = [c for c in boxes[:i] if parent[c] in rooms]
        parent[b] = rng.choice(inner) if inner and rng.random() < 0.3 else rng.choice(rooms)
        coins[b] = rng.randint(0, 3)
    for it in items:
        parent[it] = rng.choice(boxes) if rng.random() < 0.75 else rng.choice(rooms)
    pos = {p: rng.choice(rooms) for p in people}
    want = {p: rng.choice(items) for p in people}
    is_box = set(boxes)

    def up(x, par):
        out = [x]
        while out[-1] not in rooms:
            out.append(par[out[-1]])
            assert len(out) <= len(par) + 1, "a loop of containers"
        return out

    sents = _say_doors(B, rng, nb, nm)
    for x, p in parent.items():
        if p in rooms:
            sents.append(["the", nm(x), "is in the", nm(p)] if x in is_box else ["the", nm(x), "lies on the floor of the", nm(p)])
        else:
            sents.append(rng.choice([["the", nm(x), "is in the", nm(p)], ["the", nm(p), "holds the", nm(x)]]))
    for b in boxes:
        sents.append(["the", nm(b), "is empty of coins"] if coins[b] == 0 else
                     ["the", nm(b), "has", str(coins[b]), "coin" if coins[b] == 1 else "coins", "in it"])
    for p in people:
        sents.append([nm(p), "is in the", nm(pos[p])])
    for snt in _shuffled(rng, sents):
        B.say(*snt)
    for p in people:
        B.say(nm(p), "wants the", nm(want[p]))
    thinks = {p: dict(parent) for p in people}
    present = set(people)
    def comings(pre, back):
        """Someone goes out of the house, or comes back: who is away misses the moves."""
        gone = [p for p in people if p not in present]
        if gone and (back or rng.random() < 0.5):
            p = rng.choice(gone)
            B.say(*pre, nm(p), rng.choice(["comes back into the house", "comes home"]))
            present.add(p)
            return []
        if len(present) > 1 and not back:
            p = rng.choice(sorted(present))
            B.say(*pre, nm(p), rng.choice(["goes out of the house", "leaves the house"]))
            present.discard(p)
            return []
        return pre

    for m in range(level):
        pre = [rng.choice(["then", "after that"])] if m else []
        if rng.random() < 0.8:
            pre = comings(pre, back=False)
        who = rng.choice(sorted(present))
        for _ in range(200):
            x = rng.choice(boxes) if rng.random() < 0.5 else rng.choice(list(parent))
            if rng.random() < 0.6:
                to = rng.choice([r for r in rooms if r != up(x, parent)[-1]])
                if parent[x] in rooms:
                    B.say(*pre, nm(who), "carries the", nm(x), "to the", nm(to))
                else:
                    B.say(*pre, nm(who), "takes the", nm(x), "out of the", nm(parent[x]), "and carries it to the", nm(to))
                parent[x] = to
                pos[who] = to
                break
            y = rng.choice(boxes)
            if x == y or parent[x] == y or x in up(y, parent) or (x in is_box and len(up(y, parent)) > 2):
                continue
            B.say(*pre, nm(who), "puts the", nm(x), "into the", nm(y))
            parent[x] = y
            pos[who] = up(y, parent)[-1]
            break
        for p in present:              # who is in the house sees the move, and where it went (the whole way up)
            for t in up(x, parent)[:-1]:
                thinks[p][t] = parent[t]
    if rng.random() < 0.7:             # those away come home at the end, knowing only what they saw
        pre = [rng.choice(["then", "after that"])]
        while len(present) < len(people):
            pre = comings(pre, back=True)
    room = lambda x, par: up(x, par)[-1]  # noqa: E731
    inbox = lambda x, par: par[x] in is_box  # noqa: E731
    count = lambda r, par: sum(coins[b] for b in boxes if room(b, par) == r)  # noqa: E731
    cands = []                         # (kind, how to ask it, whether it is about a false belief)
    for x in items + boxes:
        if inbox(x, parent):
            cands.append(("C", lambda x=x: B.ask_pointer(["which room is the", nm(x), "in now"], room(x, parent)), None))
    for r in [r for r in rooms if count(r, parent) <= 9]:
        cands.append(("CN", lambda r=r: B.ask_class(["how many coins are in the", nm(r), "now"], count(r, parent)), None))
    for p in people:
        th = thinks[p]
        for x in items:
            bel = room(x, th) != room(x, parent)
            cands.append(("CM" if inbox(x, th) else "M", lambda p=p, x=x, th=th, bel=bel: B.ask_pointer(
                ["which room does", nm(p), "think the", nm(x), "is in"], room(x, th), belief=bel), bel))
        for r in [r for r in rooms if count(r, th) <= 9]:
            bel = count(r, th) != count(r, parent)
            cands.append(("CMN", lambda p=p, r=r, th=th, bel=bel: B.ask_class(
                ["how many coins does", nm(p), "think are in the", nm(r)], count(r, th), belief=bel), bel))
        w = want[p]
        if p not in present:           # out of the house: not on the way anywhere inside it
            continue
        if room(w, parent) != pos[p]:
            cands.append(("CW" if inbox(w, parent) else "W", lambda p=p, w=w: B.ask_pointer(
                ["which room must", nm(p), "walk into first to get the", nm(w)], _path(nb, pos[p], room(w, parent))[1]), None))
        if room(w, th) != pos[p]:
            bel = room(w, th) != room(w, parent)
            cands.append(("CMW" if inbox(w, th) else "MW", lambda p=p, w=w, th=th, bel=bel: B.ask_pointer(
                ["which room will", nm(p), "walk into first to look for the", nm(w)], _path(nb, pos[p], room(w, th))[1],
                belief=bel), bel))
    by = {}
    for k, ask, bel in cands:
        if k in allowed:
            by.setdefault(k, []).append((ask, bel))
    weight = {k: (1 if len(k) == 1 else 3) for k in by}     # mixed questions far more often than single ones
    used = []
    for _ in range(30):
        if len(B.qs) == 3 or not by:
            break
        k = rng.choices(list(by), [weight[k] for k in by])[0]
        pool = by[k]
        false = [c for c in pool if c[1]]
        ask, _ = rng.choice(false if (false and rng.random() < 0.6) else pool)
        n = len(B.qs)
        ask()
        if len(B.qs) > n:
            B.qs[-1].tags["mix"] = k
            used.append(k)
    if len(B.qs) < 2 or (split == "three-way" and not used):
        return None
    return B.episode("house", level, split, " ; ".join(f"{B.things[x]} in {B.things[room(x, parent)]}" for x in parent),
                     {"moves": level, "kinds": used, "room": {B.things[x]: B.things[room(x, parent)] for x in parent}})


def extra_words():
    """Words World 2 can say that the vocabulary must know even if the sample of episodes it is built from misses one."""
    if VERSION < 2:
        return set()
    return set(ROOMS) | set(EVENTS) | set(DEVICES) | set(PLURAL.values()) | {"switch", "hour", "hours", "coins", "coin"}


GEN = {"sums": gen_sums, "lines": gen_lines, "boxes": gen_boxes, "family": gen_family,
       "falling": gen_falling, "machines": gen_machines, "cups": gen_cups, "minds": gen_minds, "walks": gen_walks,
       "carry": gen_carry, "time": gen_time, "cause": gen_cause, "wants": gen_wants, "count": gen_count, "house": gen_house}


def make(family: str, level: int, split: str = "train", seed: int = 0) -> Episode:
    """One episode. split = "train" (levels in TRAIN_LEVELS) or a secret exam's name (levels in SECRET)."""
    if split == "train":
        assert level in TRAIN_LEVELS[family], (family, level)
    else:
        assert level in SECRET[family][split], (family, level, split)
    rng = random.Random(f"{family}|{level}|{split}|{seed}")
    return GEN[family](rng, level, split)


def exam_sets():
    """Every (family, level, split) that is examined: the training levels ("seen") and every secret exam."""
    out = []
    for fam in FAMILIES:
        for lv in TRAIN_LEVELS[fam]:
            out.append((fam, lv, "train"))
        for sp, lvs in SECRET[fam].items():
            for lv in lvs:
                out.append((fam, lv, sp))
    return out


class Curriculum:
    """Always serve the edge: per family, a frontier level that moves up once the learner gets it right.

    Half of a family's episodes come from the frontier, the other half from the levels below it (so nothing is
    forgotten). The frontier moves up when the learner's running accuracy there passes `promote`."""

    def __init__(self, promote=0.85, on=True, seed=0):
        self.promote, self.on = promote, on
        self.front = {f: (0 if on else len(TRAIN_LEVELS[f]) - 1) for f in FAMILIES}
        self.acc = {(f, lv): None for f in FAMILIES for lv in TRAIN_LEVELS[f]}
        self.seen = {k: 0 for k in self.acc}
        self.rng = random.Random(seed)

    def pick(self):
        f = self.rng.choice(FAMILIES)
        k = self.front[f]
        i = k if (self.rng.random() < 0.5 or k == 0) else self.rng.randrange(0, k + 1)
        return f, TRAIN_LEVELS[f][i]

    def update(self, family, level, correct: float):
        key = (family, level)
        a = self.acc[key]
        self.acc[key] = correct if a is None else 0.98 * a + 0.02 * correct
        self.seen[key] += 1
        lv = TRAIN_LEVELS[family]
        k = self.front[family]
        if self.on and level == lv[k] and k + 1 < len(lv) and self.seen[key] >= 200 and self.acc[key] >= self.promote:
            self.front[family] = k + 1

    def state(self):
        return {f: TRAIN_LEVELS[f][self.front[f]] for f in FAMILIES}
