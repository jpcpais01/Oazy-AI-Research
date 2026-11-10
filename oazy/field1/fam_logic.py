"""Field1, logic: switches and the machines that follow them, if-then chains, all/some/none, clues that leave one
answer, sums with their carries, money, hidden number rules and patterns."""
from __future__ import annotations

import itertools

from .core import (QA, Story, T, COLOURS, ROOMS, SMALL, and_list, num_word, people, things)

# ---------------------------------------------------------------- switches: devices that follow switches
DEVICES = "lamp fan bell pump motor heater radio alarm siren horn clock buzzer heater light".split()
FOLLOW = {
    "same": ["{d} <is|was> on whenever {a} <is|was> on", "{d} <is|was> on when {a} <is|was> on, and off when it <is|was> off",
             "{d} <works|worked> only while {a} <is|was> on", "{d} <follows|followed> {a}: on when it <is|was> on, off when it <is|was> off",
             "if {a} <is|was> on, {d} <is|was> on too, and otherwise {d} <is|was> off"],
    "not": ["{d} <is|was> on only when {a} <is|was> off", "{d} <is|was> off whenever {a} <is|was> on, and on otherwise",
            "{d} always <does|did> the opposite of {a}", "when {a} <is|was> off, {d} <is|was> on, and when {a} <is|was> on, {d} <is|was> off"],
    "and": ["{d} <is|was> on only if {a} and {b} <are|were> both on", "{d} <needs|needed> both {a} and {b} to be on",
            "{d} <comes|came> on when {a} <is|was> on and {b} <is|was> on as well, and <is|was> off otherwise"],
    "or": ["{d} <is|was> on if {a} or {b} <is|was> on", "{d} <is|was> on when at least one of {a} and {b} <is|was> on",
           "either {a} or {b} being on <is|was> enough to keep {d} on; with both off, {d} <is|was> off"],
}


def switches(r):
    for _ in range(200):
        sw = [T("switch", c) for c in r.sample(COLOURS, r.randint(2, 3))]
        devs = [T(x) for x in r.sample(sorted(set(DEVICES)), r.randint(2, 5))]
        rule, depth = {}, {id(s): 0 for s in sw}
        parts = list(sw)
        for i, d in enumerate(devs):
            first = parts[-1] if (i and r.random() < 0.6) else r.choice(parts)
            ins = [first]
            if r.random() < 0.4:
                other = [p for p in parts if p is not first]
                if other:
                    ins.append(r.choice(other))
            op = r.choice(["same", "not"]) if len(ins) == 1 else r.choice(["and", "or"])
            rule[id(d)] = (d, op, ins)
            depth[id(d)] = 1 + max(depth[id(p)] for p in ins)
            parts.append(d)
        if max(depth.values()) <= 4:
            break
    s0 = {id(s): r.random() < 0.5 for s in sw}

    def run(state):
        v = dict(state)
        for d, op, ins in rule.values():
            a = [v[id(p)] for p in ins]
            v[id(d)] = {"same": a[0], "not": not a[0], "and": all(a), "or": any(a)}[op]
        return v
    before = run(s0)
    flips = [r.choice(sw) for _ in range(r.randint(0, 2))]
    s1 = dict(s0)
    for s in flips:
        s1[id(s)] = not s1[id(s)]
    now = run(s1)
    p = people(r, 1)[0]
    st = Story(r, cast=[p])
    if r.random() < 0.5:
        st.add_intro(r.choice(["in the shed there <are|were> some switches and some machines", "{p} <is|was> playing with the switches in the cellar",
                               "the old house <has|had> strange wiring"]), p=p)
    for s in sw:
        st.fact(r.choice(["{s} <is|was> {v}", "{s} <is|was> [set to|turned] {v}", "at first, {s} <is|was> {v}"]),
                s=s, v="on" if s0[id(s)] else "off")
    for d, op, ins in rule.values():
        st.fact(r.choice(FOLLOW[op]), d=d, a=ins[0], b=ins[-1])
    cur = dict(s0)
    for s in flips:
        cur[id(s)] = not cur[id(s)]
        st.event(r.choice(["{p} <flips|flipped|had flipped> {s}", "{p} <turns|turned|had turned> {s} " + ("on" if cur[id(s)] else "off"),
                           "{p} <switches|switched|had switched> {s} " + ("on" if cur[id(s)] else "off")]), p=p, s=s)

    def why(d, said=None):
        said = set() if said is None else said
        if id(d) in said:
            return ""
        said.add(id(d))
        if id(d) not in rule:
            return st.t.fmt("{s} <is|was> " + ("on" if now[id(d)] else "off"), s=d)
        _, op, ins = rule[id(d)]
        bits = [b for b in (why(x, said) for x in ins) if b]
        state = "on" if now[id(d)] else "off"
        return ("; ".join(bits) + ", " if bits else "") + st.t.fmt("so {d} <is|was> " + state, d=d)
    qas = []
    for _ in range(12):
        if len(qas) >= r.choice((1, 2, 3)):
            break
        d = r.choice(devs)
        q = r.choice(["on", "on", "off", "changed"] if flips else ["on", "off"])
        if q == "on":
            qa = QA(st.t.fmt("<is|was> {d} on" + (" now" if flips else "") + "?", d=d), "yes" if now[id(d)] else "no",
                    why(d) if r.random() < 0.7 else "", st.t.fmt("{d} <is|was> " + ("on" if now[id(d)] else "off"), d=d))
        elif q == "off":
            qa = QA(st.t.fmt("<is|was> {d} off" + (" now" if flips else "") + "?", d=d), "no" if now[id(d)] else "yes",
                    why(d) if r.random() < 0.5 else "", "")
        else:
            ch = now[id(d)] != before[id(d)]
            qa = QA(st.t.fmt("<did|did> {d} [change|go from on to off or from off to on]?", d=d), "yes" if ch else "no",
                    st.t.fmt("{d} <was|was> " + ("on" if before[id(d)] else "off") + " at first and <is|was> " +
                             ("on" if now[id(d)] else "off") + " [now|at the end]", d=d), "")
        if all(qa.q != o.q for o in qas):
            qas.append(qa)
    return st, qas


# ---------------------------------------------------------------- rules: if-then chains
HAPPEN = [  # (present, past, gerund-ish noun for "did X happen")
    ("the rooster crows", "the rooster crowed", "did the rooster crow"), ("the porridge boils", "the porridge boiled", "did the porridge boil"),
    ("the lights flicker", "the lights flickered", "did the lights flicker"), ("the gate creaks", "the gate creaked", "did the gate creak"),
    ("the river rises", "the river rose", "did the river rise"), ("the shop closes early", "the shop closed early", "did the shop close early"),
    ("Grandpa sneezes", "Grandpa sneezed", "did Grandpa sneeze"), ("the goats run to the fence", "the goats ran to the fence", "did the goats run to the fence"),
    ("the path gets muddy", "the path got muddy", "did the path get muddy"), ("the windows steam up", "the windows steamed up", "did the windows steam up"),
    ("the milk goes sour", "the milk went sour", "did the milk go sour"), ("the cat runs under the bed", "the cat ran under the bed", "did the cat run under the bed"),
    ("the bread rises", "the bread rose", "did the bread rise"), ("the children go inside", "the children went inside", "did the children go inside"),
    ("the ice melts", "the ice melted", "did the ice melt"), ("the frogs start to sing", "the frogs started to sing", "did the frogs start to sing"),
    ("the owls wake up", "the owls woke up", "did the owls wake up"), ("the parrot squawks", "the parrot squawked", "did the parrot squawk"),
    ("the bees come out", "the bees came out", "did the bees come out"), ("the boat sets off", "the boat set off", "did the boat set off"),
    ("the soup gets cold", "the soup got cold", "did the soup get cold"), ("the flowers open", "the flowers opened", "did the flowers open"),
("the sheep huddle together", "the sheep huddled together", "did the sheep huddle together"),
    ("the puppy hides", "the puppy hid", "did the puppy hide"), ("the fire goes out", "the fire went out", "did the fire go out"),
    ("the farmer opens the barn", "the farmer opened the barn", "did the farmer open the barn"),
    ("the roof leaks", "the roof leaked", "did the roof leak"), ("the drums start", "the drums started", "did the drums start"),
]
IF_T = ["if {a}, {b}", "whenever {a}, {b}", "every time {a}, {b}", "when {a}, {b}", "{b} whenever {a}",
        "{b} if {a}", "as soon as {a}, {b}"]


def rules(r):
    st = Story(r, tense="pres")
    k = r.randint(2, 5)
    evs = r.sample(HAPPEN, k + 3)
    chain, other = evs[:k], evs[k:k + 3]
    for a, b in zip(chain, chain[1:]):
        st.fact(r.choice(IF_T), a=a[0], b=b[0])
    st.fact(r.choice(IF_T), a=other[0][0], b=other[1][0])          # a second chain that never starts
    extra = r.random() < 0.3                                          # a second cause of the last link
    if extra:
        st.fact(r.choice(IF_T), a=other[2][0], b=chain[-1][0])
    told = r.choice(["start", "start", "end_not", "end"])
    when = r.choice(["one day", "this morning", "today", "at noon", "on Sunday"])
    if told == "start":
        st.event(when + ", " + chain[0][1])
        st.add_outro(r.choice(["nothing else <happens|happened> [that day|]", "that <is|was> all that started it"]))
        i = r.randrange(1, k)
        qas = [QA(cap_q(chain[i][2]) + "?", "yes",
                  chain[0][1] + "".join(f", so {y[1]}" for y in chain[1:i + 1]), ""),
               QA(cap_q(other[1][2]) + "?", "no", f"that would only happen if {other[0][0]}, and that did not happen", "")]
        r.shuffle(qas)
    elif told == "end_not":
        st.event(when + ", " + negate(chain[-1][1]))
        i = r.randrange(0, k - 1)
        qas = [QA(cap_q(chain[i][2]) + "?", "no", f"if {chain[i][1]}, then {chain[-1][1]} as well, and that did not happen", "")]
    else:
        st.event(when + ", " + chain[-1][1])
        if extra:
            qas = [QA(cap_q(chain[0][2]) + "?", "we can't tell", f"if {other[2][0]}, that happens too, so we can't be sure", "")]
        else:
            qas = [QA(cap_q(chain[0][2]) + "?", "we can't tell",
                      "nothing says that only " + chain[-2][0] + " can make that happen", "")] if r.random() < 0.3 else []
            if not qas:
                return rules(r)
    st.t.tense = "past"
    return st, qas[:r.choice((1, 2))]


def cap_q(s):
    return s[0].upper() + s[1:]


def negate(past):
    """'the porridge boiled' -> 'the porridge did not boil' (from the 'did ...' form in HAPPEN)."""
    for pres, p, q in HAPPEN:
        if p == past:
            subj_verb = q[4:]                                    # "the porridge boil"
            return subj_verb.replace(" ", " did not ", 1) if not subj_verb.startswith("the ") else \
                "the " + subj_verb[4:].split(" ", 1)[0] + " did not " + subj_verb[4:].split(" ", 1)[1]
    return "it did not happen"


# ---------------------------------------------------------------- sets: all, some, none
GROUPS = [  # (the group, one of them, what some or all may be, where they all may be and what is true of everything there)
    ("hens on {p}'s farm", "hen", ["brown", "white", "speckled"], [("the hen house", "warm"), ("the barn", "dry")]),
    ("books on the top shelf", "book", ["old", "red", "dusty"], [("the big box", "wrapped in paper"), ("the attic", "dusty")]),
    ("children in the choir", "child", ["wearing blue", "from the village", "good at singing"], [("the hall", "quiet"), ("the front row", "standing")]),
    ("boats on the lake", "boat", ["painted white", "made of wood", "tied up"], [("the boathouse", "dry"), ("the north bay", "calm")]),
    ("cats in the yard", "cat", ["asleep", "black", "hungry"], [("the shed", "out of the wind"), ("the kitchen", "warm")]),
    ("apples in the basket", "apple", ["ripe", "green", "sweet"], [("the cellar", "cool"), ("the bowl on the table", "washed")]),
    ("cows in the meadow", "cow", ["lying down", "spotted", "eating grass"], [("the barn", "dry"), ("the top field", "fenced in")]),
    ("dogs at the show", "dog", ["brushed", "small", "well behaved"], [("the big tent", "out of the rain"), ("the ring", "being judged")]),
    ("sheep on the hill", "sheep", ["white", "shorn", "asleep"], [("the pen", "fenced in"), ("the barn", "dry")]),
]
PET = "Bramble Pepper Biscuit Clover Hazel Moss Tilly Patch Smudge Fudge Juniper Sooty Ginger Bonnie Dusty".split()


def sets(r):
    group, one, props, homes = r.choice(GROUPS)
    p = people(r, 1)[0]
    st = Story(r, cast=[p])
    g = group.replace("{p}", p.name)
    prop = r.choice(props)
    named = r.choice(PET) if one in ("hen", "cat", "cow", "dog", "boat", "sheep") else None
    member = named or f"the {one} by the {r.choice(['door', 'window', 'wall', 'gate'])}"
    kind = r.choice(["all", "none", "some", "chain", "chain"])
    if kind == "chain":
        place, prop2 = r.choice(homes)
        st.fact("all the " + g + " <are|were> in " + place)
        st.fact(r.choice(["everything in " + place + " <is|was> " + prop2, "whatever is in " + place + " <stays|stayed> " + prop2]))
        st.fact("{m} <is|was> one of the " + g, m=member)
        qas = [QA(st.t.fmt("<is|was> {m} " + prop2 + "?", m=member), "yes",
                  st.t.fmt("{m} <is|was> one of the " + g + ", they <are|were> all in " + place + ", and everything there <is|was> " + prop2, m=member), "")]
        return st, qas
    st.fact({"all": "all the " + g + " <are|were> " + prop, "none": "none of the " + g + " <is|was> " + prop,
             "some": "some of the " + g + " <are|were> " + prop + ", and some <are|were> not"}[kind])
    st.fact(r.choice(["{m} <is|was> one of the " + g, "{m} <is|was> among the " + g]), m=member)
    ans = {"all": "yes", "none": "no", "some": "we can't tell"}[kind]
    why = {"all": "all of them <are|were>, and {m} <is|was> one of them", "none": "none of them <is|was>, and {m} <is|was> one of them",
           "some": "only some of them <are|were>, and nothing says which"}[kind]
    qas = [QA(st.t.fmt("<is|was> {m} " + prop + "?", m=member), ans, st.t.fmt(why, m=member), "")]
    return st, qas


# ---------------------------------------------------------------- clues: what is left, and little logic grids
def clues(r):
    if r.random() < 0.5:
        return _left(r)
    return _grid(r)


def _left(r):
    kind = r.choice(["room", "room", "day", "colour"])
    p = people(r, 1)[0]
    st = Story(r, cast=[p])
    it = things(r, SMALL, 1, colours=False)[0]
    if kind == "room":
        opts = [T(x) for x in r.sample(ROOMS, r.randint(3, 5))]
        say = lambda o: o.np  # noqa: E731
        st.fact("{x} <is|was> [somewhere] in " + and_list([o.np for o in opts], "or"), x=it)
        no = ["{x} <is|was> not in {o}", "it <is|was> not in {o}", "{p} <has|had> looked in {o}, and it <is|was> not there",
              "{o} <has|had> been searched: no {n}"]
    elif kind == "day":
        opts = r.sample("Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split(), r.randint(3, 4))
        say = lambda o: o  # noqa: E731
        st.fact("{p}'s party <is|was> on " + and_list(opts, "or"), p=p)
        no = ["the party <is|was> not on {o}", "it <is|was> not on {o}", "{p} <is|was> busy on {o}, so not then"]
    else:
        opts = r.sample(COLOURS, r.randint(3, 4))
        say = lambda o: o  # noqa: E731
        st.fact("{p}'s new coat <is|was> " + and_list(opts, "or"), p=p)
        no = ["it <is|was> not {o}", "the coat <is|was> not {o}", "{p} <does|did> not like {o}, so it <is|was> not {o}"]
    keep = r.choice(opts)
    for o in opts:
        if o is not keep:
            st.event(r.choice(no), o=o, x=it, p=p, n=it.noun)
    st.keep_order = True
    left = say(keep)
    q = {"room": st.t.fmt("where <is|was> {x}?", x=it), "day": st.t.fmt("on which day <is|was> the party?"),
         "colour": st.t.fmt("what colour <is|was> the coat?")}[kind]
    st.connect = False
    return st, [QA(q, left, st.t.fmt("every other choice <is|was> ruled out, so it <is|was> {k}", k=left), "")]


GRID = [("pet", ["cat", "dog", "fish", "rabbit", "parrot", "hamster"], ("has", "have", "has a pet"), "the {v}"),
        ("drink", ["milk", "juice", "water", "cocoa", "lemonade"], ("drinks", "drink", "drinks something"), "{v}"),
        ("colour", ["red", "blue", "green", "yellow", "white"], ("wears", "wear", "wears a colour"), "{v}"),
        ("game", ["chess", "cards", "football", "tennis", "skipping"], ("plays", "play", "plays a game"), "{v}")]


def _grid(r):
    n = 3
    ps = people(r, n)
    attr, vals, verb, np = r.choice(GRID)
    vals = r.sample(vals, n)
    sol = dict(zip(map(id, ps), r.sample(vals, n)))
    st = Story(r, tense="pres", cast=ps)
    v3, v1, intro = verb
    opts = and_list([np.format(v=v) for v in vals], "or")
    st.add_intro(r.choice(["{L} each " + v1 + " a different one of these: " + opts,
                           "each of {L} " + intro + ": " + opts + ", and no two of them the same"]), L=and_list([p.name for p in ps]))
    perms = list(itertools.permutations(vals))
    clue_list, alive = [], perms
    for _ in range(20):
        if len(alive) == 1:
            break
        p = r.choice(ps)
        i = ps.index(p)
        if r.random() < 0.6:
            v = r.choice([v for v in vals if v != sol[id(p)]])
            c = (i, v, False)
        else:
            c = (i, sol[id(p)], True)
        new = [a for a in alive if (a[c[0]] == c[1]) == c[2]]
        if len(new) < len(alive):
            clue_list.append(c)
            alive = new
    for i, v, yes in clue_list:
        val = np.format(v=v)
        st.fact(r.choice(["{p} " + v3 + " " + val, "it <is|was> {p} who " + v3 + " " + val] if yes else
                         ["{p} does not " + v1 + " " + val, "{p} never " + v3 + " " + val, "it is not {p} who " + v3 + " " + val]), p=ps[i])
    st.keep_order = r.random() < 0.5
    qas = []
    for p in r.sample(ps, r.choice((1, 2))):
        qas.append(QA(st.t.fmt("what <does|did> {p} " + v1 + "?", p=p),
                      np.format(v=sol[id(p)]), "", ""))
    who_v = r.choice(vals)
    owner = next(p for p in ps if sol[id(p)] == who_v)
    qas.append(QA("who " + v3 + " " + np.format(v=who_v) + "?", owner.name, "", ""))
    return st, qas[:r.choice((1, 2, 3))]


# ---------------------------------------------------------------- numbers: sums with carries, take-aways, money
def numbers(r):
    k = r.random()
    if k < 0.45:
        return _column_sum(r)
    if k < 0.65:
        return _take(r)
    if k < 0.85:
        return _money(r)
    return _times(r)


def _column_sum(r):
    L = r.randint(1, 8)
    a = r.randrange(10 ** (L - 1), 10 ** L) if L > 1 else r.randrange(1, 10)
    b = r.randrange(10 ** (max(1, L - r.randint(0, 2)) - 1), 10 ** L)
    s = a + b
    st = Story(r, tense="pres")
    spaced = r.random() < 0.3
    fmt_n = (lambda n: " ".join(str(n))) if spaced else (lambda n: f"{n:,}" if r.random() < 0.3 and n >= 1000 else str(n))
    st.add_intro(r.choice(["{p} <adds|added> {a} and {b}", "we add {a} and {b}", "add {a} to {b}", "{p} <has|had> to add {a} and {b} for homework",
                           "the numbers {a} and {b} are added together"]), p=people(r, 1)[0], a=fmt_n(a), b=fmt_n(b))
    da, db = str(a)[::-1], str(b)[::-1]
    steps, c = [], 0
    for i in range(max(len(da), len(db))):
        x = int(da[i]) if i < len(da) else 0
        y = int(db[i]) if i < len(db) else 0
        t = x + y + c
        part = f"{x} + {y}" + (f" + {c}" if c else "") + f" = {t}"
        if t >= 10:
            part += r.choice([f", write {t % 10} and carry 1", f": {t % 10} down, 1 to carry", f", so {t % 10}, carrying 1"])
        steps.append(part)
        c = t // 10
    if c:
        steps.append("the last carry 1 goes in front")
    worked = r.random() < 0.6
    why = ("from the right: " + "; ".join(steps)) if worked else ""
    qa = QA(r.choice(["What is {a} plus {b}?", "What do {a} and {b} add up to?", "How much is {a} + {b}?", "{a} + {b} = ?"])
            .format(a=fmt_n(a), b=fmt_n(b)), fmt_n(s), why, f"{fmt_n(a)} + {fmt_n(b)} = {fmt_n(s)}")
    return st, [qa]


def _take(r):
    p = people(r, 1)[0]
    st = Story(r, cast=[p])
    a = r.randint(5, 60)
    b = r.randint(1, a)
    noun = r.choice(["stickers", "pages", "steps", "beads", "pennies", "sweets", "seeds", "stamps"])
    st.fact("{p} <has|had> {a} " + noun, p=p, a=a)
    st.event(r.choice(["{p} <uses|used|had used> {b}", "{p} <gives|gave|had given> away {b}", "{b} of them <are|were|had been> lost"]), p=p, b=b)
    return st, [QA(st.t.fmt("how many " + noun + " <are|were> left?"), st.t.num(a - b), f"{a} - {b} = {a - b}", "")]


def _money(r):
    p = people(r, 1)[0]
    st = Story(r, cast=[p])
    item = r.choice(["pencil", "apple", "notebook", "stamp", "ticket", "bun", "ball", "comb", "ribbon"])
    price = r.randint(1, 9)
    n = r.randint(2, 6)
    pay = price * n + r.randint(0, 10)
    st.fact(r.choice(["a {i} costs {c} coins", "{i}s <are|were> {c} coins each", "at the shop, one {i} <is|was> {c} coins"]), i=item, c=price)
    st.event(r.choice(["{p} <buys|bought|had bought> {n} {i}s", "{p} <gets|got|had got> {n} {i}s"]), p=p, n=n, i=item)
    qas = [QA(st.t.fmt("how many coins <does|did> {p} pay?", p=p), str(price * n), f"{n} x {price} = {price * n}", "")]
    if r.random() < 0.5:
        st.event("{p} <pays|paid|had paid> with {m} coins", p=p, m=pay)
        qas = [QA(st.t.fmt("how many coins <does|did> {p} get back?", p=p), str(pay - price * n),
                  f"{n} x {price} = {price * n}, and {pay} - {price * n} = {pay - price * n}", "")]
    return st, qas


def _times(r):
    a, b = r.randint(2, 9), r.randint(2, 9)
    st = Story(r, tense="pres")
    what = r.choice([("rows of chairs", "chairs"), ("boxes of eggs", "eggs"), ("bags of marbles", "marbles"), ("shelves of books", "books")])
    st.fact(f"there <are|were> {{a}} {what[0]}, with {{b}} {what[1]} in each", a=a, b=b)
    return st, [QA(st.t.fmt(f"how many {what[1]} <are|were> there in all?"), str(a * b),
                   f"{a} x {b} = {a * b}" if r.random() < 0.7 else f"{' + '.join([str(b)] * a)} = {a * b}", "")]


# ---------------------------------------------------------------- machines: a hidden rule, found from tries; patterns
PRIMS = {"+1": (lambda x: (x + 1) % 10, "adds one"), "+2": (lambda x: (x + 2) % 10, "adds two"),
         "+3": (lambda x: (x + 3) % 10, "adds three"), "-1": (lambda x: (x - 1) % 10, "takes one away"),
         "x2": (lambda x: (2 * x) % 10, "doubles it"), "x3": (lambda x: (3 * x) % 10, "makes it three times as big"),
         "mirror": (lambda x: 9 - x, "takes it away from nine")}
MACH_HELD_OUT = [("+2", "x3"), ("+3", "mirror"), ("-1", "x2"), ("x2", "+3"), ("x3", "mirror"), ("mirror", "+2")]
TRY_T = ["you put in {x} and out comes {y}", "{x} goes in, {y} comes out", "feed it {x} and you get {y}",
         "in: {x}, out: {y}", "when {p} says {x}, the machine says {y}", "{x} becomes {y}", "it turns {x} into {y}",
         "{p} tries {x} and <gets|got> {y}"]


def machines(r):
    if r.random() < 0.35:
        return _pattern(r)
    small = r.random() < 0.5                       # world-style: one digit, wrapping round past 9
    if small:
        steps = r.choice([(k,) for k in PRIMS] + [p for p in itertools.permutations(PRIMS, 2) if p not in MACH_HELD_OUT])
        f = lambda x: _run(steps, x)  # noqa: E731
        dom = list(range(10))
        rule_words = " and then ".join(PRIMS[s][1] for s in steps) + " (only the last digit is kept)"
    else:
        kind = r.choice(["add", "add", "times", "times_add"])
        k, m = r.randint(2, 9), r.randint(1, 9)
        f = {"add": lambda x: x + k, "times": lambda x: x * k, "times_add": lambda x: x * min(k, 4) + m}[kind]
        rule_words = {"add": f"adds {k}", "times": f"multiplies by {k}", "times_add": f"multiplies by {min(k, 4)} and then adds {m}"}[kind]
        dom = list(range(1, 13))
    if small:
        univ = [lambda x, st_=st_: _run(st_, x) for st_ in [(k,) for k in PRIMS] +
                [p for p in itertools.permutations(PRIMS, 2) if p not in MACH_HELD_OUT]]
    else:
        univ = ([lambda x, k=k: x + k for k in range(1, 10)] + [lambda x, k=k: x * k for k in range(2, 10)] +
                [lambda x, a=a, m=m: x * a + m for a in range(2, 5) for m in range(1, 10)])
    xs = r.sample(dom, len(dom))
    ask = xs.pop()
    ex = [xs.pop() for _ in range(3)]
    while xs and len({g(ask) for g in univ if all(g(x) == f(x) for x in ex)}) > 1:
        ex.append(xs.pop())
    if len({g(ask) for g in univ if all(g(x) == f(x) for x in ex)}) > 1:
        return machines(r)
    asks = [ask]
    p = people(r, 1)[0]
    st = Story(r, cast=[p])
    st.add_intro(r.choice(["a machine changes numbers by a hidden rule", "{p} <has|had> a number machine", "there <is|was> a box that turns numbers into other numbers",
                           "{p} and a friend <play|played> a number game with a secret rule"]), p=p)
    for x in ex:
        st.event(r.choice(TRY_T), x=x, y=f(x), p=p)
    st.keep_order = True
    st.connect = False
    return st, [QA(r.choice(["What comes out for {x}?", "What does {x} become?", "What would the machine say for {x}?",
                             "And if you put in {x}?"]).format(x=asks[0]), str(f(asks[0])),
                   "the rule " + rule_words if r.random() < 0.6 else "", "")]


def _run(steps, x):
    for s in steps:
        x = PRIMS[s][0](x)
    return x


def _pattern(r):
    kind = r.choice(["step", "step", "double", "two", "down"])
    a = r.randint(1, 9)
    if kind == "step":
        d = r.randint(2, 9)
        seq = [a + d * i for i in range(6)]
        why = f"each one is {d} more than the one before"
    elif kind == "down":
        d = r.randint(1, 5)
        a = d * 6 + r.randint(0, 20)
        seq = [a - d * i for i in range(6)]
        why = f"each one is {d} less than the one before"
    elif kind == "double":
        a = r.randint(1, 4)
        seq = [a * 2 ** i for i in range(6)]
        why = "each one is twice the one before"
    else:
        d1, d2 = r.randint(1, 5), r.randint(1, 5)
        seq, x = [], a
        for i in range(6):
            seq.append(x)
            x += d1 if i % 2 == 0 else d2
        why = f"it goes up by {d1}, then by {d2}, again and again"
    shown, ans = seq[:5], seq[5]
    st = Story(r, tense="pres")
    st.add_intro(r.choice(["here is a pattern: {s}", "look at these numbers: {s}", "a row of numbers: {s}", "the numbers go {s}"]),
                 s=", ".join(map(str, shown)))
    return st, [QA(r.choice(["What comes after {l}?", "Which number comes after that?", "What is the following number?",
                             "What number should come after {l}?"]).format(l=shown[-1]), str(ans), why, "")]
