"""Wordings: the same rule said many ways, in plain English, for Loop-135's A/B (gnosis2/loop.py, loopab/).

Why. A model that thinks over meanings must learn that "Ana is taller than Ben", "Ben is shorter than Ana" and "Ben is
not as tall as Ana" are ONE thought. The old loop's best round (h2, Nov 16) was exactly this, four wordings of each
rule: 97% at 8 steps. But it was a closed little language (walled words, fixed sentences), and it did not carry over to
real text. So here: ordinary words, many wordings of every kind of statement, chains told in any order, and a quarter
of the documents say everything twice ("In other words: ..."), each time in other words.

Five kinds of reasoning, each a chain of d steps:
    order      who is taller / older / faster ... (comparisons chained)
    groups     everyone in the choir is in the music society; everyone there gets a ticket; does Mia? (yes / no)
    places     the key is in the box, the box in the bag, the bag was taken to the attic: where is the key?
    causes     if the alarm rings the dog barks; if the dog barks the baby wakes; the alarm rang: which happened?
    counting   Leo had 7 coins, gave 2 away, found 3: how many now?

Training text: chains of 1 to 3 steps (TRAIN_DEPTH), the training wordings only. The held-out test (evals/rules.jsonl,
scored like ARC: the answer that surprises the model least) has three parts of 400 questions each:
    seen          1-3 steps, training wordings, new names and facts
    new_wording   1-3 steps, every fact said in a wording that NEVER appears in training (HELD templates)
    deeper        4-6 steps, training wordings: longer chains than any training document
Every test question is checked against the rules themselves (solve()), so the test has no wrong answers.

    python3 -m oazy.wordings build                 # <data>/opal/wordings (150M SmolLM2 pieces) + <data>/evals/rules.jsonl
    python3 -m oazy.wordings show                  # a few documents and test questions
    python3 -m oazy.wordings test                  # only the test file
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

TRAIN_DEPTH = 3
TEST_DEPTHS = {"seen": (1, 3), "new_wording": (1, 3), "deeper": (4, 6)}
TEST_PER_KIND = 400

NAMES = ("Ana Ben Carla David Elena Felix Grace Hugo Iris Jack Kate Leo Maria Nina Oscar Paula Quinn Rosa Sam Tara "
         "Uma Victor Wendy Yara Zoe Adam Bella Chris Dora Ethan Fiona George Hannah Ivan Julia Kevin Lily Mark Nora "
         "Owen Petra Rita Simon Tom Vera Will Alice Bruno Clara Daniel Emma Frank Gina Henry Ines Jonas Lucy Martin "
         "Olga Pedro Ruth Sofia Theo Ursula Max Mia Noah Eva Luis Joao Marta Rui Ines Tiago Beatriz").split()
ITEMS = ("apples coins stamps marbles cards pencils eggs stickers books shells buttons cookies tickets candles "
         "flowers plates").split()
OBJECTS = ("key coin letter ring watch map photo pen ticket button stone feather card note badge spoon").split()
CONTAINERS = ("box bag basket jar case envelope tin pouch drawer crate bucket backpack suitcase purse chest "
              "folder").split()
ROOMS = ("kitchen garden hall attic garage bedroom office cellar library porch bathroom shed classroom "
         "playroom barn studio").split()
GROUPS = ("the chess club|the choir|the music society|the swim team|the book club|the drama group|the science club|"
          "the art class|the hiking club|the school band|the garden club|the robot team|the film club|the dance group|"
          "the cooking class|the running club|the photo club|the debate team|the night class|the tennis club").split("|")
PROPS = [("gets a free ticket", "get a free ticket"), ("wears a blue badge", "wear a blue badge"),
         ("knows how to swim", "know how to swim"), ("has a key to the gym", "have a key to the gym"),
         ("plays the piano", "play the piano"), ("speaks French", "speak French"), ("owns a bicycle", "own a bicycle"),
         ("eats lunch at noon", "eat lunch at noon"), ("rides the early bus", "ride the early bus"),
         ("writes a weekly letter", "write a weekly letter"), ("grows tomatoes", "grow tomatoes"),
         ("keeps a diary", "keep a diary")]
ORDERS = [("taller", "shorter", "tall"), ("older", "younger", "old"), ("heavier", "lighter", "heavy"),
          ("faster", "slower", "fast"), ("stronger", "weaker", "strong"), ("richer", "poorer", "rich"),
          ("louder", "quieter", "loud"), ("busier", "calmer", "busy")]
EVENTS = [("the alarm rings", "the alarm rang"), ("the dog barks", "the dog barked"),
          ("the baby wakes up", "the baby woke up"), ("the lights go out", "the lights went out"),
          ("the river floods", "the river flooded"), ("the shop closes early", "the shop closed early"),
          ("the bell rings", "the bell rang"), ("the kettle whistles", "the kettle whistled"),
          ("the cat runs away", "the cat ran away"), ("the bus is late", "the bus was late"),
          ("the game is cancelled", "the game was cancelled"), ("the road is closed", "the road was closed"),
          ("the bread burns", "the bread burned"), ("the phone dies", "the phone died"),
          ("the window breaks", "the window broke"), ("the milk goes sour", "the milk went sour"),
          ("the garden gets wet", "the garden got wet"), ("the fire starts", "the fire started"),
          ("the music stops", "the music stopped"), ("the class ends early", "the class ended early"),
          ("the birds fly off", "the birds flew off"), ("the door slams", "the door slammed"),
          ("the floor gets muddy", "the floor got muddy"), ("the train stops", "the train stopped"),
          ("the plants grow", "the plants grew"), ("the ice melts", "the ice melted"),
          ("the clock stops", "the clock stopped"), ("the room gets cold", "the room got cold"),
          ("the crowd cheers", "the crowd cheered"), ("the car will not start", "the car would not start"),
          ("the soup boils over", "the soup boiled over"), ("the roof leaks", "the roof leaked")]

# ------------------------------------------------------------------------------------------------ wordings
# Each kind of statement: TRAIN wordings (the only ones in the training text) and HELD wordings (only in the test).
T = {
    "gt": {   # {A} is more {er} than {B}
        "train": ["{A} is {er} than {B}.", "{B} is {opp} than {A}.", "{B} is not as {base} as {A}.",
                  "Compared with {B}, {A} is {er}.", "{A} is a little {er} than {B}.", "Next to {A}, {B} is {opp}.",
                  "{A} is clearly {er} than {B}.", "{B} is {opp} than {A} is."],
        "held": ["Of the two, {A} and {B}, it is {A} who is {er}.", "{B} cannot match {A} for being {base}.",
                 "Put {A} beside {B} and {B} is the {opp} one."],
    },
    "member": {
        "train": ["{P} is in {G}.", "{P} is a member of {G}.", "{P} belongs to {G}.", "{P} joined {G}.",
                  "{P} is part of {G}.", "{P} is one of the people in {G}."],
        "held": ["{Gc} counts {P} among its members.", "{P} signed up for {G} and stayed."],
    },
    "subset": {
        "train": ["Everyone in {G1} is also in {G2}.", "All members of {G1} are members of {G2}.",
                  "If you are in {G1}, you are in {G2}.", "Being in {G1} means being in {G2}.",
                  "Every member of {G1} belongs to {G2} too.", "Anyone who is in {G1} is in {G2} as well."],
        "held": ["{G2c} takes in every single member of {G1}.", "There is no one in {G1} who is left out of {G2}."],
    },
    "has": {
        "train": ["Everyone in {G} {prop}.", "All members of {G} {propb}.", "If you are in {G}, you {propb}.",
                  "Every member of {G} {prop}.", "People in {G} all {propb}.", "Anyone in {G} {prop}."],
        "held": ["Whoever is in {G} {prop}, without exception.", "Membership of {G} comes with the duty to {propb}."],
    },
    "hasnot": {
        "train": ["No one in {G} {prop}.", "Nobody in {G} {prop}.", "Members of {G} never {propb}.",
                  "If you are in {G}, you do not {propb}.", "None of the people in {G} {propb}."],
        "held": ["Not a single member of {G} {prop}.", "In {G} the rule is plain: you never {propb}."],
    },
    "in": {   # thing {X} inside container {C}
        "train": ["The {X} is in the {C}.", "The {X} is inside the {C}.", "{P} put the {X} in the {C}.",
                  "{P} put the {X} into the {C}.", "The {C} holds the {X}.", "The {X} was placed in the {C}.",
                  "{P} dropped the {X} into the {C}."],
        "held": ["The {X} lies tucked within the {C}.", "You would find the {X} if you opened the {C}."],
    },
    "at": {   # container {C} in room {R}
        "train": ["The {C} is in the {R}.", "{P} carried the {C} to the {R}.", "{P} took the {C} into the {R}.",
                  "The {C} sits in the {R}.", "The {C} was left in the {R}.", "{P} moved the {C} to the {R}."],
        "held": ["The {C} ended up over in the {R}.", "Look in the {R} and the {C} is right there."],
    },
    "cause": {
        "train": ["If {a}, {b}.", "Whenever {a}, {b}.", "Each time {a}, {b}.", "{Bc} whenever {a}.",
                  "When {a}, {b}.", "If {a}, then {b}."],
        "held": ["One thing leads to another: once {a}, {b}.", "It is a rule here that as soon as {a}, {b}."],
    },
    "start": {
        "train": ["This morning, {ap}.", "Today {ap}.", "Earlier today, {ap}.", "Just now, {ap}.", "Yesterday {ap}."],
        "held": ["At some point today, {ap}.", "Without warning, {ap}."],
    },
    "had": {
        "train": ["{P} had {n} {it}.", "{P} started with {n} {it}.", "At first {P} had {n} {it}.",
                  "{P} owned {n} {it}."],
        "held": ["{P} began the day holding {n} {it}."],
    },
    "plus": {
        "train": ["{Q} gave {P} {k} more {it}.", "{P} found {k} more {it}.", "{P} bought {k} more {it}.",
                  "{P} got {k} more {it} from {Q}."],
        "held": ["{Q} handed {P} another {k} {it}.", "Then {k} extra {it} came {P}'s way."],
    },
    "minus": {
        "train": ["{P} gave {k} of the {it} to {Q}.", "{P} lost {k} of the {it}.", "{P} sold {k} of the {it}.",
                  "{Q} took {k} of {P}'s {it}."],
        "held": ["{k} of {P}'s {it} went missing.", "{P} parted with {k} of the {it}."],
    },
}
OTHER_WORDS = ["In other words:", "Said another way:", "Put differently:", "To say it again:", "Once more, in other words:"]
INTROS = ["", "", "", "Here is a short puzzle.", "Read this and answer the question.", "A small story.",
          "Think it through.", "A question about the facts below."]


def held_out_phrases():
    """A phrase that marks each held-out wording (none may appear in training text)."""
    return ["Of the two,", "cannot match", "beside", "among its members", "signed up for", "takes in every single",
            "who is left out", "without exception", "comes with the duty", "Not a single", "the rule is plain",
            "tucked within", "if you opened", "ended up over", "is right there", "leads to another", "It is a rule here",
            "At some point today", "Without warning", "began the day holding", "handed", "s way", "went missing",
            "parted with"]


def cap(s):
    return s[0].upper() + s[1:]


def say(rng, kind, mode, **kw):
    pool = T[kind]["held" if mode == "held" else "train"]
    kw.update({k + "c": cap(v) for k, v in list(kw.items()) if isinstance(v, str) and k in ("G", "G1", "G2", "b")})
    if "b" in kw:
        kw["Bc"] = cap(kw["b"])
    return rng.choice(pool).format(**kw)


# ------------------------------------------------------------------------------------------------ the five kinds
# Each returns a puzzle: dict(kind, depth, facts: [(statement kind, fields)], question, endings, label, spec)
def order(rng, d):
    er, opp, base = rng.choice(ORDERS)
    n = d + 1 + rng.randint(0, 2)
    ppl = rng.sample(NAMES, n)                       # ppl[0] is the most {base}
    i = rng.randint(0, n - 1 - d)
    j = i + d
    facts = [("gt", dict(A=ppl[k], B=ppl[k + 1], er=er, opp=opp, base=base)) for k in range(n - 1)]
    rng.shuffle(facts)
    ask_er = rng.random() < 0.5
    pair = [ppl[i], ppl[j]]
    rng.shuffle(pair)
    word = er if ask_er else opp
    ans = ppl[i] if ask_er else ppl[j]
    return dict(kind="order", depth=d, facts=facts, question=f"Who is {word}, {pair[0]} or {pair[1]}?",
                endings=pair, label=pair.index(ans), spec=dict(chain=ppl, asked=pair, er=ask_er))


def groups(rng, d):
    gs = rng.sample(GROUPS, 2 * d)
    yes_chain, no_chain = gs[:d], gs[d:]
    prop, propb = rng.choice(PROPS)
    p_yes, p_no = rng.sample(NAMES, 2)
    facts = [("member", dict(P=p_yes, G=yes_chain[0])), ("member", dict(P=p_no, G=no_chain[0]))]
    for ch in (yes_chain, no_chain):
        facts += [("subset", dict(G1=ch[k], G2=ch[k + 1])) for k in range(d - 1)]
    facts += [("has", dict(G=yes_chain[-1], prop=prop, propb=propb)),
              ("hasnot", dict(G=no_chain[-1], prop=prop, propb=propb))]
    rng.shuffle(facts)
    who = rng.choice([p_yes, p_no])
    return dict(kind="groups", depth=d, facts=facts, question=f"Does {who} {propb}?", endings=["yes", "no"],
                label=0 if who == p_yes else 1, spec=dict(who=who))


def places(rng, d):
    rooms = rng.sample(ROOMS, 4)
    cs = rng.sample(CONTAINERS, 2 * max(1, d - 1)) if d > 1 else []
    objs = rng.sample(OBJECTS, 2)
    facts, ends = [], {}
    for o, room, ch in ((objs[0], rooms[0], cs[:d - 1]), (objs[1], rooms[1], cs[d - 1:2 * (d - 1)])):
        things = [o] + list(ch)                      # the thing, then each container around it
        for k in range(len(things) - 1):
            facts.append(("in", dict(X=things[k], C=things[k + 1], P=rng.choice(NAMES))))
        facts.append(("at" if len(things) > 1 else "at_obj", dict(C=things[-1], R=room, P=rng.choice(NAMES))))
        ends[o] = room
    facts = [(("at", f) if k == "at_obj" else (k, f)) for k, f in facts]
    rng.shuffle(facts)
    o = rng.choice(objs)
    endings = [f"the {r}" for r in rooms]
    rng.shuffle(endings)
    return dict(kind="places", depth=d, facts=facts, question=f"Where is the {o}?", endings=endings,
                label=endings.index(f"the {ends[o]}"), spec=dict(obj=o))


def causes(rng, d):
    n_ch = 3
    ev = rng.sample(EVENTS, n_ch * (d + 1))
    chains = [ev[c * (d + 1):(c + 1) * (d + 1)] for c in range(n_ch)]
    facts = []
    for ch in chains:
        facts += [("cause", dict(a=ch[k][0], b=ch[k + 1][0])) for k in range(d)]
    hit = rng.randrange(n_ch)
    facts.append(("start", dict(ap=chains[hit][0][1])))
    rng.shuffle(facts)
    endings = [cap(ch[-1][1]) for ch in chains]
    order_ = list(range(n_ch))
    rng.shuffle(order_)
    endings = [endings[i] for i in order_]
    return dict(kind="causes", depth=d, facts=facts, question="Which of these happened?", endings=endings,
                label=order_.index(hit), spec=dict())


def counting(rng, d):
    p, q = rng.sample(NAMES, 2)
    it = rng.choice(ITEMS)
    n = rng.randint(2, 9)
    facts = [("had", dict(P=p, n=n, it=it))]
    for _ in range(d):
        if n > 1 and rng.random() < 0.5:
            k = rng.randint(1, min(4, n - 1))
            n -= k
            facts.append(("minus", dict(P=p, Q=rng.choice([x for x in NAMES if x != p]), k=k, it=it)))
        else:
            k = rng.randint(1, 4)
            n += k
            facts.append(("plus", dict(P=p, Q=rng.choice([x for x in NAMES if x != p]), k=k, it=it)))
    opts = {n}
    while len(opts) < 4:
        c = n + rng.choice([-3, -2, -1, 1, 2, 3])
        if c >= 0:
            opts.add(c)
    endings = [str(x) for x in sorted(opts)]
    return dict(kind="counting", depth=d, facts=facts, question=f"How many {it} does {p} have now?",
                endings=endings, label=endings.index(str(n)), spec=dict(who=p), ordered=True)


KINDS = {"order": order, "groups": groups, "places": places, "causes": causes, "counting": counting}


def text_of(rng, pz, mode):
    return " ".join(say(rng, k, mode, **f) for k, f in pz["facts"])


def solve(row):
    """The answer from the facts alone (a check that every test question is right): -> the index of the right ending."""
    facts = row["facts"]
    kind = row["family"]
    if kind == "order":
        above = {}
        for k, f in facts:
            above.setdefault(f["A"], set()).add(f["B"])

        def beats(a, b):
            seen, todo = set(), [a]
            while todo:
                x = todo.pop()
                for y in above.get(x, ()):
                    if y == b:
                        return True
                    if y not in seen:
                        seen.add(y)
                        todo.append(y)
            return False
        a, b = row["endings"]
        top = a if beats(a, b) else b
        return row["endings"].index(top if row["spec"]["er"] else (b if top == a else a))
    if kind == "groups":
        member = {f["P"]: f["G"] for k, f in facts if k == "member"}
        up = {f["G1"]: f["G2"] for k, f in facts if k == "subset"}
        yes = {f["G"] for k, f in facts if k == "has"}
        g = member[row["spec"]["who"]]
        while g not in yes and g in up:
            g = up[g]
        return 0 if g in yes else 1
    if kind == "places":
        where = {f["X"]: f["C"] for k, f in facts if k == "in"}
        room = {f["C"]: f["R"] for k, f in facts if k == "at"}
        x = row["spec"]["obj"]
        while x not in room:
            x = where[x]
        return row["endings"].index(f"the {room[x]}")
    if kind == "causes":
        nxt = {f["a"]: f["b"] for k, f in facts if k == "cause"}
        past = {p: pa for p, pa in EVENTS}
        start = [f["ap"] for k, f in facts if k == "start"][0]
        e = [p for p, pa in EVENTS if pa == start][0]
        while e in nxt:
            e = nxt[e]
        return row["endings"].index(cap(past[e]))
    if kind == "counting":
        n = 0
        for k, f in facts:
            n = f["n"] if k == "had" else n + f["k"] if k == "plus" else n - f["k"]
        return row["endings"].index(str(n))
    raise ValueError(kind)


# ------------------------------------------------------------------------------------------------ documents
def qa(pz):
    return f"Question: {pz['question']}\nAnswer: {pz['endings'][pz['label']]}"


def document(rng, mode="train"):
    """One training document: 1 to 3 puzzles of 1 to TRAIN_DEPTH steps, training wordings only; a quarter of them
    told twice, the second time in other words."""
    parts = []
    intro = rng.choice(INTROS)
    if intro:
        parts.append(intro)
    for _ in range(rng.choice((1, 1, 2, 3))):
        pz = KINDS[rng.choice(list(KINDS))](rng, rng.randint(1, TRAIN_DEPTH))
        body = text_of(rng, pz, mode)
        if rng.random() < 0.25:
            again = text_of(rng, pz, mode)
            tries = 0
            while again == body and tries < 5:
                again, tries = text_of(rng, pz, mode), tries + 1
            body = f"{body}\n{rng.choice(OTHER_WORDS)} {again}"
        parts.append(f"{body}\n{qa(pz)}")
    return "\n\n".join(parts)


def test_rows(per_kind=TEST_PER_KIND, seed=12345):
    """The held-out test: {ctx, endings, label, kind (seen / new_wording / deeper), family, depth, facts, spec}."""
    rng = random.Random(seed)
    rows = []
    fams = list(KINDS)
    for kind, (lo, hi) in TEST_DEPTHS.items():
        mode = "held" if kind == "new_wording" else "train"
        for i in range(per_kind):
            fam = fams[i % len(fams)]
            pz = KINDS[fam](rng, lo + i // len(fams) % (hi - lo + 1))
            ctx = f"{text_of(rng, pz, mode)}\nQuestion: {pz['question']}\nAnswer:"
            rows.append(dict(ctx=ctx, endings=pz["endings"], label=pz["label"], kind=kind, family=fam,
                             depth=pz["depth"], facts=pz["facts"], spec=pz["spec"]))
    return rows


def write_test(data: Path, per_kind=TEST_PER_KIND):
    rows = test_rows(per_kind)
    bad = sum(solve(r) != r["label"] for r in rows)
    if bad:
        raise SystemExit(f"{bad} test questions disagree with the rules: not written")
    out = Path(data) / "evals"
    out.mkdir(parents=True, exist_ok=True)
    for kind in TEST_DEPTHS:                       # one file per part (the trainer scores each like ARC)
        (out / f"rules_{kind}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows if r["kind"] == kind))
    print(f"rules test: {len(rows)} questions in {out}/rules_*.jsonl ({', '.join(TEST_DEPTHS)})")


def batches(seed=0, per=512):
    rng = random.Random(seed)
    while True:
        yield [document(rng) for _ in range(per)]


def build(data: Path, tokens: float, val_tokens: float, workers: int):
    from .data import write_tokens
    from .opal3 import opal_dir
    out = opal_dir(Path(data)) / "wordings"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train.bin").exists() and json.loads((out / "meta.json").read_text()).get("train_tokens", 0) >= tokens * 0.98:
        print(f"{out} is there already")
    else:
        t0 = time.time()
        nv, nt = write_tokens(batches(), out, int(val_tokens), int(tokens), workers, tokenizer="smollm2")
        (out / "meta.json").write_text(json.dumps({"dataset": "wordings (oazy/wordings.py)", "tokenizer": "smollm2",
                                                   "train_tokens": nt, "val_tokens": nv, "train_depth": TRAIN_DEPTH,
                                                   "minutes": round((time.time() - t0) / 60, 1)}, indent=1))
        print(f"wordings: {nt / 1e6:,.0f}M training pieces in {out}")
    write_test(data)


def main(argv=None):
    from .data import default_dir
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["build", "show", "test"])
    ap.add_argument("--data", default=None)
    ap.add_argument("--tokens", type=float, default=150e6)
    ap.add_argument("--val-tokens", type=float, default=1e6)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args(argv)
    data = Path(a.data or default_dir("data"))
    if a.what == "show":
        rng = random.Random(1)
        for _ in range(3):
            print(document(rng), "\n" + "-" * 60)
        for r in test_rows(5)[::5]:
            print(f"[{r['kind']}, {r['family']}, {r['depth']} steps]\n{r['ctx']} {r['endings'][r['label']]}   "
                  f"(choices: {', '.join(r['endings'])})\n")
    elif a.what == "test":
        write_test(data)
    else:
        build(data, a.tokens, a.val_tokens, a.workers)


if __name__ == "__main__":
    main()
