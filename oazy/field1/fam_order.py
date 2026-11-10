"""Field1, order: lines that change, rows left to right, who is taller or older (with and without numbers), plans in
the day and days of the week."""
from __future__ import annotations

from .core import (QA, Story, T, ANIMALS, COLOURS, SMALL, and_list, ordinal, people, plural, sents, things)

# ---------------------------------------------------------------- lines: who stands in front of whom, after moves
LINE_PLACES = ["at the bakery", "at the ice cream stand", "at the ticket window", "for the ferry", "outside the cinema",
               "for the lift", "at the counter", "for the slide in the park", "at the water fountain", "to buy tickets",
               "for the boat ride", "in the school corridor", "to see the doctor", "at the till", "at the library desk",
               "for the zoo gate to open", "at the fish shop", "for a turn on the swing", "at the bank window",
               "for the toilet", "to get on the train", "at the front door of the school"]
LINE_INTRO = [
    "{n} people <are|were> [waiting|standing] in [a|one] line {where}",
    "a line <has|had> formed {where}",
    "there <is|was> a line {where}",
    "{n} [friends|children|neighbours|cousins] <wait|waited> in line {where}, one behind the other",
    "[at school|in the morning|after lunch], {n} children <line|lined> up {where}",
    "{where}, people <stand|stood> in line, one behind the other",
]
ANIMAL_INTRO = [
    "{n} animals <walk|walked> along the path in single file",
    "a row of animals <waits|waited> at the gate, one behind the other",
    "{n} animals <cross|crossed> the yard in a line, nose to tail",
    "on the farm, {n} animals <stand|stood> in a line by the trough, one behind the other",
]
ADJ = [  # x is the one right in front of y
    "{x} <is|was> [right|just] in front of {y}",
    "{x} <stands|stood> [right|just] in front of {y}",
    "{y} <is|was> [right|just] behind {x}",
    "{y} <stands|stood> [right|just] behind {x}",
    "{y} <comes|came> [right|straight] after {x}",
    "{x} <is|was> one place ahead of {y}",
    "[right|just] behind {x} <stands|stood|is|was> {y}",
    "[right|just] in front of {y} <is|was|stands|stood> {x}",
    "{y} <waits|waited> right behind {x}",
    "{y} <follows|followed> right behind {x}",
    "{x} <is|was> [right|just] ahead of {y}",
    "the one right behind {x} <is|was> {y}",
    "{y} <has|had> {x} [right|just] in front of {y.o}@",
    "{x} <has|had> {y} [right|just] behind {x.o}@",
    "the person [right|just] ahead of {y} <is|was> {x}@",
    "{y} <is|was> standing right behind {x}",
]
SWAP = ["{x} and {y} <swap|swapped|had swapped> places", "{x} <swaps|swapped|had swapped> places with {y}",
        "{x} and {y} <change|changed|had changed> places", "{x} <changes|changed|had changed> places with {y}"]
TO_FRONT = ["{x} <goes|went|had gone> to the front", "{x} <is|was|had been> called to the front",
            "{x} <moves|moved|had moved> up to the very front", "{x} <walks|walked|had walked> to the front of the line",
            "{x} <pushes|pushed|had pushed> past everyone to the front"]
TO_BACK = ["{x} <goes|went|had gone> to the back", "{x} <moves|moved|had moved> to the end of the line",
           "{x} <lets|let|had let> everyone go ahead and <goes|went|gone> to the back",
           "{x} <walks|walked|had walked> to the back of the line"]
LEAVE = ["{x} <leaves|left|had left> the line", "{x} <gets|got|had got> tired of waiting and <goes|went|gone> away",
         "{x} <steps|stepped|had stepped> out of the line and <goes|went|gone> home"]
SERVED = ["{x}, at the front, <is|was|had been> served and <leaves|left|left>",
          "the person at the front, {x}, <gets|got|had got> what {x.s} <wants|wanted|wanted> and <goes|went|gone>@"]
JOIN_BACK = ["{w} <arrives|arrived|had arrived> and <joins|joined|joined> the end of the line",
             "{w} <comes|came|had come> along and <stands|stood|stood> at the back"]
JOIN_AFTER = ["{w} <arrives|arrived|had arrived> and <slips|slipped|slipped> in right behind {x}",
              "{w} <turns|turned|had turned> up and {x} <lets|let|had let> {w.o} in right behind {x.o}@"]


def _who(animal):
    return "which animal" if animal else "who"


def lines(r):
    animal = r.random() < 0.2
    n = r.randint(3, 10)
    if animal:
        ents = things(r, ANIMALS, n + 2, colours=r.random() < 0.6)
    else:
        ents = people(r, n + 2)
    spare, ents = ents[n:], ents[:n]
    order = list(ents)
    r.shuffle(order)
    st = Story(r, cast=[] if animal else list(ents))
    adj = [t for t in ADJ if not (animal and t.endswith("@"))]
    st.add_intro(r.choice(ANIMAL_INTRO if animal else LINE_INTRO), n=n, where=r.choice(LINE_PLACES))
    for a, b in zip(order, order[1:]):
        st.fact(r.choice(adj).rstrip("@"), x=a, y=b)
    if r.random() < 0.3:
        st.fact(r.choice(["{x} <is|was> at the front", "at the front <is|was> {x}", "{x} <is|was> first in line"]), x=order[0])
    moves = r.choice((0, 1, 1, 2, 2, 3))
    def ok(t):
        return not (animal and t.endswith("@"))
    for _ in range(moves):
        kind = r.choice(["swap", "swap", "front", "back", "leave", "join", "served"] if len(order) > 3 else ["swap", "front", "back", "join"])
        if kind == "swap":
            x, y = r.sample(order, 2)
            st.event(r.choice(SWAP), x=x, y=y)
            i, j = order.index(x), order.index(y)
            order[i], order[j] = order[j], order[i]
        elif kind == "front":
            x = r.choice(order[1:])
            st.event(r.choice(TO_FRONT), x=x)
            order.remove(x); order.insert(0, x)
        elif kind == "back":
            x = r.choice(order[:-1])
            st.event(r.choice(TO_BACK), x=x)
            order.remove(x); order.append(x)
        elif kind == "leave":
            x = r.choice(order[1:])
            st.event(r.choice(LEAVE), x=x)
            order.remove(x)
        elif kind == "served":
            x = order[0]
            st.event(r.choice([t for t in SERVED if ok(t)]).rstrip("@"), x=x)
            order.remove(x)
        elif kind == "join" and spare:
            w = spare.pop()
            if r.random() < 0.5:
                st.event(r.choice(JOIN_BACK), w=w)
                order.append(w)
            else:
                x = r.choice(order)
                st.event(r.choice([t for t in JOIN_AFTER if ok(t)]).rstrip("@"), w=w, x=x)
                order.insert(order.index(x) + 1, w)
    pos = {id(e): i for i, e in enumerate(order)}
    names = lambda xs: and_list([e.np if isinstance(e, T) else e.name for e in xs])  # noqa: E731
    summary = r.choice(["from the front, the line <is|was> now {L}", "the order <is|was> now {L}, from front to back",
                        "front to back: {L}", "the line <is|was> now {L}, starting at the front"])
    why = st.t.fmt(summary, L=names(order))
    who = _who(animal)
    now = st.t.now() if moves and r.random() < 0.8 else ""
    qas = []
    for _ in range(10):
        if len(qas) >= r.choice((1, 2, 3, 3)):
            break
        q = r.choice(["front", "back", "ahead", "ahead", "ahead", "behind", "infront", "count", "place", "between", "size"])
        x, y = r.sample(order, 2) if len(order) >= 2 else (order[0], order[0])
        if q == "front":
            qa = QA(st.t.fmt(f"{who} <is|was> [at the front|first in line|at the head of the line] {now}?"),
                    str(order[0]).replace("the ", "", 0), why, st.t.fmt("{x} <is|was> at the front", x=order[0]))
        elif q == "back":
            qa = QA(st.t.fmt(f"{who} <is|was> [at the back|at the end of the line|the last one in line] {now}?"),
                    str(order[-1]), why, st.t.fmt("{x} <is|was> at the back", x=order[-1]))
        elif q == "ahead":
            yes = pos[id(x)] < pos[id(y)]
            if r.random() < 0.5:
                qt, ans = "<is|was> {x} [ahead of|in front of|nearer the front than] {y} " + now + "?", yes
            else:
                qt, ans = "<is|was> {x} [behind|further back than|nearer the back than] {y} " + now + "?", not yes
            gap = abs(pos[id(x)] - pos[id(y)])
            rel = "ahead of" if yes else "behind"
            qa = QA(st.t.fmt(qt, x=x, y=y), "yes" if ans else "no",
                    sents(why, st.t.fmt("so {x} <is|was> " + rel + " {y}, by {k} place" + ("s" if gap > 1 else ""),
                                        x=x, y=y, k=gap)),
                    st.t.fmt("{x} <is|was> " + rel + " {y}", x=x, y=y))
        elif q == "behind" and pos[id(x)] < len(order) - 1:
            z = order[pos[id(x)] + 1]
            qa = QA(st.t.fmt(f"{who} <is|was> [right|just] behind {{x}} {now}?", x=x), str(z), why,
                    st.t.fmt("{z} <is|was> right behind {x}", x=x, z=z))
        elif q == "infront" and pos[id(x)] > 0:
            z = order[pos[id(x)] - 1]
            qa = QA(st.t.fmt(f"{who} <is|was> [right|just] in front of {{x}} {now}?", x=x), str(z), why,
                    st.t.fmt("{z} <is|was> right in front of {x}", x=x, z=z))
        elif q == "count":
            k = pos[id(x)]
            qa = QA(st.t.fmt("how many <are|were> [ahead of|in front of] {x} " + now + "?", x=x), st.t.num(k), why,
                    st.t.fmt("{k} <are|were> ahead of {x}", x=x, k=k) if k > 1 else st.t.fmt("only one <is|was> ahead of {x}", x=x)
                    if k == 1 else st.t.fmt("nobody <is|was> ahead of {x}: {x.s} <is|was> at the front", x=x))
        elif q == "place":
            k = pos[id(x)] + 1
            qa = QA(st.t.fmt("in which place <is|was> {x} " + now + ", counting from the front?", x=x), ordinal(k), why,
                    st.t.fmt("{x} <is|was> " + ordinal(k) + " in line", x=x))
        elif q == "between":
            i, j = sorted((pos[id(x)], pos[id(y)]))
            if j - i != 2:
                continue
            z = order[i + 1]
            qa = QA(st.t.fmt(f"{who} <is|was> between {{x}} and {{y}} {now}?", x=x, y=y), str(z), why,
                    st.t.fmt("{z} <is|was> between {x} and {y}", x=x, y=y, z=z))
        elif q == "size" and moves:
            qa = QA(st.t.fmt("how many <are|were> in the line " + now + "?"), st.t.num(len(order)), why,
                    st.t.fmt("there <are|were> {k} in the line", k=len(order)))
        else:
            continue
        if all(qa.q != o.q for o in qas):
            qas.append(qa)
    return st, qas


# ---------------------------------------------------------------- rows: left and right
ROW_INTRO = [
    "on the shelf there <is|was> a row of things", "{n} things <stand|stood> in a row on the table",
    "a row of things <is|was> set out along the windowsill", "{p} <puts|put> {n} things in a row on the desk",
    "along the bench, there <are|were> {n} things side by side", "{p} <lines|lined> up {n} things on the shelf, from left to right",
]
SEAT_INTRO = [
    "{n} [friends|children|people] <sit|sat> on a long bench, side by side", "{n} people <sit|sat> in a row at the cinema",
    "in the photo, {n} cousins <stand|stood> side by side", "{n} children <sit|sat> in a row on the wall",
]
ROW_ADJ = [  # x is just left of y
    "{x} <is|was> just to the left of {y}", "{y} <is|was> just to the right of {x}",
    "just to the right of {x} <is|was> {y}", "just to the left of {y} <is|was> {x}",
    "{x} <is|was> right beside {y}, on {y.p} left@", "{y} <sits|sat> just to the right of {x}",
    "{x} <is|was> on the left of {y}, [right beside|just beside|beside] {y.o}", "{y} <is|was> on {x.p} right, [right beside|beside] {x.o}",
    "the one just to the right of {x} <is|was> {y}", "{x} <comes|came> just before {y}, going from left to right",
]


# not the things of LogT's rows (cup, lamp, book, vase, ...): its questions stay new
ROW_THINGS = ("candle spoon comb brush shell ribbon stone glove sock map photo plate apple pear lemon orange teapot "
              "jug basket radio globe trophy candlestick pinecone seashell lantern").split()


def rows(r):
    seat = r.random() < 0.4
    n = r.randint(3, 7)
    if seat:
        ents = people(r, n)
    else:
        ents = things(r, ROW_THINGS, n, colours=r.random() < 0.5)
    order = list(ents)
    r.shuffle(order)
    other = people(r, 1)[0]
    st = Story(r, cast=list(ents) if seat else [other])
    st.add_intro(r.choice(SEAT_INTRO if seat else ROW_INTRO), n=n, p=other)
    adj = [t.rstrip("@") for t in ROW_ADJ if seat or not (".o}" in t or ".p}" in t or t.endswith("@"))]
    for a, b in zip(order, order[1:]):
        st.fact(r.choice(adj), x=a, y=b)
    moves = 0 if seat else r.choice((0, 0, 1, 1, 2))
    for _ in range(moves):
        if r.random() < 0.6:
            x, y = r.sample(order, 2)
            st.event("{p} <swaps|swapped|had swapped> {x} and {y}", p=other, x=x, y=y)
            i, j = order.index(x), order.index(y)
            order[i], order[j] = order[j], order[i]
        else:
            x = r.choice(order)
            side = r.choice(["left", "right"])
            st.event("{p} <moves|moved|had moved> {x} to the far " + side + " end", p=other, x=x)
            order.remove(x)
            order.insert(0 if side == "left" else len(order), x)
    pos = {id(e): i for i, e in enumerate(order)}
    L = and_list([str(e) for e in order])
    why = st.t.fmt(r.choice(["from left to right, the row <is|was> {L}", "left to right: {L}", "the order from the left <is|was> {L}"]), L=L)
    what = "who" if seat else "what"
    qas = []
    for _ in range(10):
        if len(qas) >= r.choice((1, 2, 3)):
            break
        q = r.choice(["left", "left", "right", "end", "middle", "rightof", "between", "count"])
        x, y = r.sample(order, 2)
        if q == "left":
            yes = pos[id(x)] < pos[id(y)]
            qa = QA(st.t.fmt("<is|was> {x} [to the left of|further left than|on the left of] {y}?", x=x, y=y),
                    "yes" if yes else "no", why, st.t.fmt("{x} <is|was> " + ("left" if yes else "right") + " of {y}", x=x, y=y))
        elif q == "right":
            yes = pos[id(x)] > pos[id(y)]
            qa = QA(st.t.fmt("<is|was> {x} [to the right of|further right than] {y}?", x=x, y=y),
                    "yes" if yes else "no", why, "")
        elif q == "end":
            side = r.choice(["left", "right"])
            z = order[0] if side == "left" else order[-1]
            qa = QA(st.t.fmt(f"{what} <is|was> at the far {side} end?"), str(z), why,
                    st.t.fmt("{z} <is|was> at the far " + side, z=z))
        elif q == "middle" and len(order) % 2 == 1:
            z = order[len(order) // 2]
            qa = QA(st.t.fmt(f"{what} <is|was> in the middle?"), str(z), why, st.t.fmt("{z} <is|was> in the middle", z=z))
        elif q == "rightof" and pos[id(x)] < len(order) - 1:
            z = order[pos[id(x)] + 1]
            qa = QA(st.t.fmt(f"{what} <is|was> just to the right of {{x}}?", x=x), str(z), why, "")
        elif q == "between":
            i, j = sorted((pos[id(x)], pos[id(y)]))
            k = j - i - 1
            qa = QA(st.t.fmt("how many <are|were> between {x} and {y}?", x=x, y=y), st.t.num(k), why, "")
        elif q == "count":
            k = pos[id(x)]
            qa = QA(st.t.fmt(f"how many <are|were> to the left of {{x}}?", x=x), st.t.num(k), why, "")
        else:
            continue
        if all(qa.q != o.q for o in qas):
            qas.append(qa)
    return st, qas


# ---------------------------------------------------------------- compare: taller, older, faster (chains, numbers)
SCALES = [("taller", "shorter", "tallest", "shortest"), ("older", "younger", "oldest", "youngest"),
          ("faster", "slower", "fastest", "slowest"), ("heavier", "lighter", "heaviest", "lightest"),
          ("stronger", "weaker", "strongest", "weakest")]
MORE_T = ["{a} <is|was> {up} than {b}", "{b} <is|was> {down} than {a}", "{a} <is|was> a bit {up} than {b}",
          "{b} <is|was> not as {base} as {a}", "compared with {a}, {b} <is|was> {down}", "{a} <is|was> {up} than {b} [<is|was>|]"]
BASE = {"taller": "tall", "older": "old", "faster": "fast", "heavier": "heavy", "stronger": "strong"}


def compare(r):
    up, down, most, least = r.choice(SCALES)
    if r.random() < 0.35 and up == "older":                        # ages with numbers
        return _ages(r)
    n = r.randint(3, 6)
    ps = people(r, n)
    order = list(ps)                                               # order[0] is the most `up`
    r.shuffle(order)
    st = Story(r, cast=ps)
    for a, b in zip(order, order[1:]):
        st.fact(r.choice(MORE_T), a=a, b=b, up=up, down=down, base=BASE[up])
    pos = {id(p): i for i, p in enumerate(order)}
    chain = lambda i, j: ", and ".join(  # noqa: E731
        st.t.fmt("{a} <is|was> {up} than {b}", a=order[k], b=order[k + 1], up=up) for k in range(i, j))
    qas = []
    for _ in range(8):
        if len(qas) >= r.choice((1, 2, 2, 3)):
            break
        q = r.choice(["most", "least", "pair", "yes", "yes"])
        x, y = r.sample(order, 2)
        if len(order) > 3 and r.random() < 0.6:          # far apart: the answer needs the chain
            i0 = r.randrange(0, len(order) - 2)
            x, y = (order[i0], order[r.randrange(i0 + 2, len(order))])
            if r.random() < 0.5:
                x, y = y, x
        i, j = sorted((pos[id(x)], pos[id(y)]))
        if q == "most":
            qa = QA(st.t.fmt(f"who <is|was> the {most}?"), order[0].name,
                    st.t.fmt("nobody <is|was> {up} than {a}", a=order[0], up=up), st.t.fmt(f"{{a}} <is|was> the {most}", a=order[0]))
        elif q == "least":
            qa = QA(st.t.fmt(f"who <is|was> the {least}?"), order[-1].name,
                    st.t.fmt("everyone else <is|was> {up} than {a}", a=order[-1], up=up), st.t.fmt(f"{{a}} <is|was> the {least}", a=order[-1]))
        elif q == "pair":
            word = r.choice([up, down])
            win = order[i] if word == up else order[j]
            qa = QA(st.t.fmt(f"of {{x}} and {{y}}, who <is|was> {word}?", x=x, y=y), win.name, chain(i, j),
                    st.t.fmt(f"{{w}} <is|was> {word}", w=win))
        else:
            word = r.choice([up, down])
            yes = (pos[id(x)] < pos[id(y)]) == (word == up)
            qa = QA(st.t.fmt(f"<is|was> {{x}} {word} than {{y}}?", x=x, y=y), "yes" if yes else "no", chain(i, j), "")
        if all(qa.q != o.q for o in qas):
            qas.append(qa)
    return st, qas


def _ages(r):
    n = r.randint(2, 4)
    ps = people(r, n)
    st = Story(r, cast=ps)
    age = {id(ps[0]): r.randint(3, 12)}
    st.fact("{a} <is|was> {k} [years old|]", a=ps[0], k=age[id(ps[0])])
    for i in range(1, n):
        ref = ps[r.randrange(i)]
        d = r.randint(1, 5)
        older = r.random() < 0.5 or age[id(ref)] - d < 1
        age[id(ps[i])] = age[id(ref)] + (d if older else -d)
        st.fact(r.choice(["{a} <is|was> {d} year{s} {w} than {b}", "{b} <is|was> {d} year{s} {w2} than {a}"]),
                a=ps[i], b=ref, d=d, s="s" if d > 1 else "", w="older" if older else "younger", w2="younger" if older else "older")
    st.keep_order = True
    qas = []
    x = r.choice(ps[1:])
    qas.append(QA(st.t.fmt("how old <is|was> {x}?", x=x), st.t.num(age[id(x)]), "",
                  st.t.fmt("{x} <is|was> {k}", x=x, k=age[id(x)])))
    a, b = r.sample(ps, 2)
    if age[id(a)] != age[id(b)]:
        w = a if age[id(a)] > age[id(b)] else b
        qas.append(QA(st.t.fmt("who <is|was> older, {a} or {b}?", a=a, b=b), w.name,
                      st.t.fmt("{a} <is|was> {x} and {b} <is|was> {y}", a=a, b=b, x=age[id(a)], y=age[id(b)]), ""))
        qas.append(QA(st.t.fmt("how many years older <is|was> {w} than {v}?", w=w, v=b if w is a else a),
                      st.t.num(abs(age[id(a)] - age[id(b)])), "", ""))
    return st, qas[:r.choice((1, 2, 3))]


# ---------------------------------------------------------------- time: plans that hang on plans, and days of the week
EVENTS = ("breakfast lunch dinner market concert lesson game party walk nap swim meeting show picnic race dance film "
          "visit class bath story practice").split()
EV_NICE = {"visit": "visit to the dentist", "story": "story hour", "practice": "football practice", "class": "art class",
           "film": "film", "bath": "bath"}
DAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()


def _ev(name):
    return T(EV_NICE.get(name, name))


def _hour(st, h):
    return st.t.fmt(st.r.choice(["{h} o'clock", "{h}"]), h=h) if st.r.random() < 0.8 else f"{h}:00"


def schedule(r):
    if r.random() < 0.3:
        return _days(r)
    n = r.randint(3, 5)
    names = r.sample(EVENTS, n)
    evs = [_ev(x) for x in names]
    st = Story(r)
    p = people(r, 1)[0]
    st.cast = [p]
    st.add_intro(r.choice(["here <is|was> {p.p} plan for the day", "{p} <has|had> a busy day", "{p} <plans|planned> {p.p} day",
                           "this <is|was> how {p.p} [Saturday|Sunday|holiday] <goes|went>"]), p=p)
    base = evs[0]
    hour = {id(base): r.randint(3, 6)}
    dep = {}                                                  # event -> (parent, offset)
    for i in range(1, n):
        par = evs[r.randrange(i)]
        off = r.choice([-2, -1, 1, 1, 2, 3])
        if not 1 <= hour[id(par)] + off <= 11:
            off = -off
        if not 1 <= hour[id(par)] + off <= 11:
            off = 1 if hour[id(par)] < 11 else -1
        hour[id(evs[i])] = hour[id(par)] + off
        dep[id(evs[i])] = (par, off)
    st.fact(r.choice(["{e} <is|was> at {h}", "{e} <starts|started> at {h}", "at {h} there <is|was> {e}"]), e=base, h=_hour(st, hour[id(base)]))
    for e in evs[1:]:
        par, off = dep[id(e)]
        k = abs(off)
        st.fact(r.choice(["{e} <is|was> {k} hour{s} {w} {p}", "{e} <comes|came> {k} hour{s} {w} {p}",
                          "{k} hour{s} {w} {p}, there <is|was> {e}"]),
                e=e, p=par, k="an" if k == 1 and r.random() < 0.5 else k, s="s" if k > 1 else "", w="after" if off > 0 else "before")
    moved = False
    if r.random() < 0.5:                                      # the first plan moves, and what hangs on it moves too
        d = r.choice([-2, -1, 1, 2])
        if all(1 <= hour[id(e)] + d <= 11 for e in evs):
            for e in evs:
                hour[id(e)] += d
            st.event(r.choice(["{e} <is|was|had been> moved {k} hour{s} {w}, and everything else <moves|moved|moved> with it",
                               "{e} <is|was|had been> put {k} hour{s} {w}, and the plans that hang on it <move|moved|moved> along with it"]),
                     e=base, k=abs(d), s="s" if abs(d) > 1 else "", w="later" if d > 0 else "earlier")
            moved = True
    qas = []
    for _ in range(8):
        if len(qas) >= r.choice((1, 2, 3)):
            break
        q = r.choice(["when", "when", "first", "gap"])
        x, y = r.sample(evs, 2)
        if q == "when":
            qa = QA(st.t.fmt("at what time <is|was> {e}" + (" now" if moved else "") + "?", e=x), str(hour[id(x)]),
                    _why_time(st, x, dep, hour), st.t.fmt("{e} <is|was> at {h}", e=x, h=_hour(st, hour[id(x)])))
        elif q == "first" and hour[id(x)] != hour[id(y)]:
            w = x if hour[id(x)] < hour[id(y)] else y
            qa = QA(st.t.fmt("which <comes|came> first, {x} or {y}?", x=x, y=y), w.np,
                    st.t.fmt("{x} <is|was> at {a} and {y} at {b}", x=x, y=y, a=hour[id(x)], b=hour[id(y)]), "")
        elif q == "gap":
            k = abs(hour[id(x)] - hour[id(y)])
            qa = QA(st.t.fmt("how many hours <are|were> there between {x} and {y}?", x=x, y=y), st.t.num(k),
                    st.t.fmt("{x} <is|was> at {a} and {y} at {b}", x=x, y=y, a=hour[id(x)], b=hour[id(y)]), "")
        else:
            continue
        if all(qa.q != o.q for o in qas):
            qas.append(qa)
    return st, qas


def _why_time(st, e, dep, hour):
    steps = []
    while id(e) in dep:
        par, off = dep[id(e)]
        steps.append(st.t.fmt("{e} <is|was> {k} hour{s} {w} {p}", e=e, p=par, k=abs(off), s="s" if abs(off) > 1 else "",
                              w="after" if off > 0 else "before"))
        e = par
    steps.append(st.t.fmt("{e} <is|was> at {h}", e=e, h=hour[id(e)]))
    return ", and ".join(reversed(steps))


def _days(r):
    st = Story(r)
    ps = people(r, 2)
    st.cast = ps
    today = r.randrange(7)
    ev = _ev(r.choice(["party", "picnic", "race", "concert", "game", "visit", "lesson"]))
    k = r.randint(1, 6)
    st.fact(r.choice(["today <is|was> {d}", "it <is|was> {d} [today|that day]", "the day <is|was> a {d}"]), d=DAYS[today])
    st.fact(r.choice(["{e} <is|was> in {k} days", "{e} <is|was> {k} days from today", "{p} <says|said> {e} <is|was> in {k} days"]),
            e=ev, k=k, p=ps[0])
    j = r.randint(1, 3)
    st.fact(r.choice(["{p}'s birthday <is|was> {j} day{s} after {e}", "{j} day{s} after {e} <comes|came> {p}'s birthday"]),
            p=ps[1], j=j, e=ev, s="s" if j > 1 else "")
    st.keep_order = True
    d1, d2 = DAYS[(today + k) % 7], DAYS[(today + k + j) % 7]
    qas = [QA(st.t.fmt("on what day <is|was> {e}?", e=ev), d1, st.t.fmt("{k} days after {d} <is|was> {x}", k=k, d=DAYS[today], x=d1),
              st.t.fmt("{e} <is|was> on {x}", e=ev, x=d1)),
           QA(st.t.fmt("on what day <is|was> {p}'s birthday?", p=ps[1]), d2,
              st.t.fmt("{e} <is|was> on {x}, and {j} day{s} later <is|was> {y}", e=ev, x=d1, j=j, s="s" if j > 1 else "", y=d2), "")]
    return st, qas[:r.choice((1, 2))]
