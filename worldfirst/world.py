"""The world: a small universe of things and links, told in simple English, that answers truthfully after every action.

Every episode is one scene (sentences about things and how they are linked, with fresh random names and colours),
one action that changes it, and a few questions about the scene *after* the action. The world knows the true new
scene, so every answer is right by construction.

Six families of laws sit underneath:

    sums      adding two numbers, with carries                  ("The first number is 0 4 7 2 . ...")
    lines     who stands in front of whom (order chains)         ("Ana stands just in front of Bo .")
    boxes     what is inside what (nesting)                      ("The jar is in the box .")
    family    parents, children, brothers, sisters, both ways    ("Paulo is the father of João .")
    falling   blocks fall until they rest on something           ("The red block is nailed at level 6 .")
    machines  a hidden rule, learned from a few tries            ("Feed it 3 and it gives 5 .")

Each family has levels (difficulty) used for training, and SECRET EXAMS that no training episode can come from, by
construction: longer sums, longer lines, deeper nesting, bigger falls, family links chained in ways never asked,
and machines whose rules mix steps that were never mixed in training. `tests.py` checks that the walls hold.

Pure Python, seeded: `make(family, level, split, seed)` always gives the same episode.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field

FAMILIES = ("sums", "lines", "boxes", "family", "falling", "machines")

# Levels the model may train on.
TRAIN_LEVELS = {
    "sums": [1, 2, 3, 4, 5, 6, 7, 8],      # digits per number
    "lines": [1, 2, 3, 4, 5, 6],           # people in the line = level + 2 (3..8)
    "boxes": [1, 2, 3, 4, 5, 6],           # boxes around the deepest thing
    "family": [1, 2, 3],                   # links chained: one, two, three
    "falling": [1, 2, 3, 4],               # blocks that move when something lets go
    "machines": [1, 2],                    # one hidden step / two hidden steps
}
# The secret exams: split name -> levels. Never trained on.
SECRET = {
    "sums": {"longer": [9, 10, 11, 12]},
    "lines": {"longer": [7, 8, 9, 10],         # 9..12 people
              "more-moves": [3, 4]},           # 3 or 4 moves in a row (training only ever has 1 or 2); 6 people
    "boxes": {"deeper": [7, 8, 9],
              "more-moves": [3, 4]},           # 3 or 4 moves in a row; 4 boxes deep
    "family": {"new-pairs": [2, 3], "deeper": [4]},   # new-pairs: two links (or three) chained in a way never asked
    "falling": {"bigger": [5, 6, 7]},
    "machines": {"new-mixes": [2], "deeper": [3]},
}
TRAIN_MOVES = (1, 2)            # lines and boxes: how many actions a training episode has
MOVES_PEOPLE, MOVES_DEPTH = 6, 4   # the size of the scene in the "more-moves" exams

# Answers that are not things: yes/no and the ten digits ("class" answers).
CLASSES = ("yes", "no") + tuple(str(i) for i in range(10))
YESNO = (0, 1)
DIGITS = tuple(range(2, 12))

FEMALE = ("Ana Bea Clara Dora Eva Fia Gina Hana Ines Joana Kira Lia Mara Nina Olga Pia Rita Sara Tania Vera "
          "Zoe Alba Bela Cora Dina Elsa Flor Greta Ida Jade Lara Luna Maya Noor Rosa Sofia Tara Uma Wanda Yara").split()
MALE = ("Bo Cy Dan Edu Filipe Gil Hugo Ivo Joao Kai Leo Mau Nuno Omar Paulo Rui Sam Tiago Ugo Vasco "
        "Abel Bruno Caio Dino Emil Fabio Goncalo Henri Igor Jonas Luis Marco Nelson Otto Pedro Raul Simao Tomas Victor Zeca").split()
NAMES = FEMALE + MALE
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

    def thing(self, key):
        self.things.append(key)
        return len(self.things) - 1

    def _emit(self, parts, toks, ment):
        for p in parts:
            if isinstance(p, tuple):
                t, words = p
                for w in words.split():
                    toks.append(w)
                    ment.append(t)
            else:
                for w in p.split():
                    toks.append(w)
                    ment.append(-1)

    def say(self, *parts):
        self._emit(parts, self.toks, self.ment)
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
    B.ask([rng.choice(["what is the sum", "what number do you get", "what do they make together"])],
          [str(d) for d in S], [("coldigit", (cols[j], d)) for j, d in enumerate(S)])
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
    for x in ex:
        t = B.thing(f"try{x}")
        B.say(*rng.choice([["it turns", (t, str(x)), "into", (t, str(f[x]))],
                           ["feed it", (t, str(x)), "and it gives", (t, str(f[x]))],
                           ["put in", (t, str(x)), "and out comes", (t, str(f[x]))]]))
    B.say(rng.choice(["now you try new numbers", "you feed it two more numbers", "then you try again"]))
    for q in queries:
        B.ask_class([rng.choice(["what comes out for", "what does it give for", "what do you get for"]), str(q)], f[q])
    return B.episode("machines", level, split, " ".join(f"{x}->{f[x]}" for x in range(10)), {"rule": list(f)})


GEN = {"sums": gen_sums, "lines": gen_lines, "boxes": gen_boxes, "family": gen_family,
       "falling": gen_falling, "machines": gen_machines}


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
