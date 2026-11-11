"""Field1, space: things inside things, hidden under cups, in piles, people walking about, rooms joined by doors."""
from __future__ import annotations

from .core import (QA, Story, T, COLOURS, ask, note, CONTAINERS, ROOMS, SMALL, PLACES_OUT, and_list, ordinal, people, plural, sents, things)

# ---------------------------------------------------------------- places: inside, carried, emptied
SIZE = {"envelope": 1, "wallet": 1, "pouch": 2, "tin": 2, "jar": 3, "pot": 3, "bowl": 3, "vase": 3, "shoebox": 4, "box": 4,
        "bag": 4, "case": 4, "backpack": 5, "basket": 5, "sack": 5, "bucket": 5, "crate": 6, "chest": 7, "drawer": 6}
IN_T = [  # x is (directly) in c
    "{x} <is|was> in {c}", "{c} <has|had> {x} in it", "inside {c} there <is|was> {x}", "{x} <is|was> tucked inside {c}",
    "{x} <sits|sat> inside {c}", "{c} <holds|held> {x}", "{x} <is|was> kept in {c}", "in {c} <is|was> {x}",
    "{x} <lies|lay> at the bottom of {c}", "someone <has|had> put {x} in {c}", "{x} <is|was> inside {c}",
]
ON_T = ["{c} <is|was> on {f}", "{c} <stands|stood> on {f}", "on {f} there <is|was> {c}", "{c} <sits|sat> on {f}",
        "{f} <has|had> {c} on it"]
ROOM_T = ["{c} <is|was> in {room}", "in {room} there <is|was> {c}", "{c} <is|was> kept in {room}",
          "{c} <stands|stood> in {room}", "{room} <has|had> {c} in it", "{c} <is|was> [over|] in {room}"]
MOVE_IN = ["{p} <takes|took|had taken> {x} out of {c} and <puts|put|put> it in {d}",
           "{p} <moves|moved|had moved> {x} from {c} to {d}", "{p} <slips|slipped|had slipped> {x} into {d}",
           "{p} <tucks|tucked|had tucked> {x} into {d}", "{p} <hides|hid|had hidden> {x} in {d}",
           "{p} <puts|put|had put> {x} into {d}", "{x} <goes|went|had gone> into {d}; {p} <puts|put|had put> it there"]
CARRY = ["{p} <carries|carried|had carried> {c} to {room}", "{p} <takes|took|had taken> {c} into {room}",
         "{p} <moves|moved|had moved> {c} to {room}", "{c} <is|was|had been> carried to {room} by {p}",
         "{p} <lugs|lugged|had lugged> {c} [all the way|] to {room}"]
OUT_ON = ["{p} <takes|took|had taken> {x} out of {c} and <leaves|left|left> it on {f}",
          "{p} <lifts|lifted|had lifted> {x} out of {c} and <sets|set|set> it down on {f}"]
EMPTY = ["{p} <empties|emptied|had emptied> {c} out onto {f}", "{p} <tips|tipped|had tipped> everything in {c} out onto {f}"]
SETTINGS = ["it <is|was> a [busy|quiet|messy|tidy] house", "{p} <is|was> tidying up", "{p} <is|was> getting ready for a trip",
            "{p} <is|was> playing a hiding game", "the family <is|was> moving things around before a party"]


def places(r):
    rooms_mode = r.random() < 0.55
    nrooms = r.randint(2, 4) if rooms_mode else 0
    rooms = [T(x) for x in r.sample(ROOMS, nrooms)]
    furn = [T(x) for x in r.sample(["table", "shelf", "desk", "bench", "floor", "bed", "chair"], 2)]
    depth = r.randint(1, 4)
    n_cont = depth + r.randint(0, 2)
    conts = things(r, [c for c in CONTAINERS if c != "drawer"], n_cont, colours=r.random() < 0.7)
    conts.sort(key=lambda c: SIZE.get(c.noun, 4) + r.random() * 0.1)      # smaller things go inside bigger ones
    if any(SIZE.get(a.noun, 4) >= SIZE.get(b.noun, 4) for a, b in zip(conts[:depth], conts[1:depth])):
        return places(r)
    items = things(r, SMALL, r.randint(1, 2), colours=False)
    ps = people(r, r.randint(1, 3))
    holder = {}                                   # id -> thing that holds it (container / furniture / room)
    obj = {}
    for x in conts + items + rooms + furn:
        obj[id(x)] = x
    # the main nest: items[0] in conts[0] in conts[1] ... ; the other containers stand alone
    main = conts[:depth]
    holder[id(items[0])] = main[0]
    for a, b in zip(main, main[1:]):
        holder[id(a)] = b
    tops = [main[-1]] + conts[depth:]
    for c in tops:
        holder[id(c)] = r.choice(rooms) if rooms_mode else r.choice(furn)
    for it in items[1:]:
        holder[id(it)] = r.choice(conts)

    def inside(x, y):                             # is x somewhere inside y?
        h = holder.get(id(x))
        while h is not None:
            if h is y:
                return True
            h = holder.get(id(h))
        return False

    def room_of(x):
        h = holder.get(id(x))
        while h is not None and h not in rooms:
            h = holder.get(id(h))
        return h

    def chain(x):
        out, h = [x], holder.get(id(x))
        while h is not None:
            out.append(h)
            h = holder.get(id(h))
        return out

    st = Story(r, cast=ps)
    if r.random() < 0.4:
        st.add_intro(r.choice(SETTINGS), p=ps[0])
    for xid, h in holder.items():
        x = obj[xid]
        if h in rooms:
            st.fact(r.choice(ROOM_T), c=x, room=h)
        elif h in furn:
            st.fact(r.choice(ON_T).replace("{f}", "the floor" if h.noun == "floor" else "{f}"), c=x, f=h)
        else:
            st.fact(r.choice(IN_T), x=x, c=h)
    moves = r.choice((0, 1, 1, 2, 2, 3))
    for _ in range(moves):
        p = r.choice(ps)
        kind = r.choice(["in", "in", "carry", "out", "empty"] if rooms_mode else ["in", "in", "out", "empty"])
        if kind == "carry" and rooms_mode:
            c = r.choice(conts)
            dest = r.choice([x for x in rooms if x is not room_of(c)])
            st.event(r.choice(CARRY), p=p, c=c, room=dest)
            holder[id(c)] = dest
        elif kind == "in":
            x = r.choice(items + conts)
            cands = [d for d in conts if d is not x and d is not holder.get(id(x)) and not inside(d, x)
                     and SIZE.get(d.noun, 4) > (SIZE.get(x.noun, 4) if x in conts else 0)]
            if not cands or (x in conts and depth >= 4 and len(chain(x)) > 4):
                continue
            d = r.choice(cands)
            src = holder[id(x)]
            tpl = r.choice(MOVE_IN if src in conts else [t for t in MOVE_IN if "{c}" not in t])
            st.event(tpl, p=p, x=x, c=src, d=d)
            holder[id(x)] = d
            if len(chain(items[0])) > 6:
                return places(r)
        elif kind == "out":
            x = r.choice(items)
            if holder[id(x)] not in conts:
                continue
            f = r.choice(furn) if not rooms_mode else None
            if rooms_mode:                        # it stays in the room it was in: left on the floor there
                rm = room_of(x)
                st.event("{p} <takes|took|had taken> {x} out of {c} and <leaves|left|left> it on the floor of {room}",
                         p=p, x=x, c=holder[id(x)], room=rm)
                holder[id(x)] = rm
            else:
                st.event(r.choice(OUT_ON).replace("{f}", "the floor" if f.noun == "floor" else "{f}"), p=p, x=x, c=holder[id(x)], f=f)
                holder[id(x)] = f
        elif kind == "empty":
            c = r.choice(conts)
            kids = [obj[k] for k, h in holder.items() if h is c]
            if not kids:
                continue
            if rooms_mode:
                rm = room_of(c)
                st.event("{p} <empties|emptied|had emptied> {c} out onto the floor of {room}", p=p, c=c, room=rm)
                for k in kids:
                    holder[id(k)] = rm
            else:
                f = r.choice(furn)
                st.event(r.choice(EMPTY).replace("{f}", "the floor" if f.noun == "floor" else "{f}"), p=p, c=c, f=f)
                for k in kids:
                    holder[id(k)] = f
    now = " " + st.t.now() if moves and r.random() < 0.6 else ""

    def why_of(x):
        ch = chain(x)
        bits = []
        for a, b in zip(ch, ch[1:]):
            if b in furn:
                bits.append(st.t.fmt("{a} <is|was> on " + ("the floor" if b.noun == "floor" else "{b}"), a=a, b=b))
            else:
                bits.append(st.t.fmt("{a} <is|was> in {b}", a=a, b=b))
        return ", ".join(bits[:-1]) + (", and " if len(bits) > 1 else "") + bits[-1]

    qas = []
    for _ in range(12):
        if len(qas) >= r.choice((1, 2, 3)):
            break
        q = r.choice(["room", "room", "holder", "inside", "inside", "find", "top"])
        x = items[0] if r.random() < 0.6 else r.choice(items + conts)
        if q == "room" and rooms_mode:
            rm = room_of(x)
            qa = QA(st.t.fmt("[which room|where] <is|was> {x} in" + now + "?", x=x) if r.random() < 0.5 else
                    st.t.fmt("in which room <is|was> {x}" + now + "?", x=x), rm.np, why_of(x),
                    st.t.fmt("{x} <is|was> in {room}", x=x, room=rm))
        elif q == "holder" and holder[id(x)] in conts:
            h = holder[id(x)]
            qa = QA(st.t.fmt("what <is|was> {x} in" + now + "?", x=x), h.np, "", st.t.fmt("{x} <is|was> in {h}", x=x, h=h))
        elif q == "inside":
            cands = [c for c in conts if c is not x]
            if not cands:
                continue
            want = r.random() < 0.5
            pool = [c for c in cands if inside(x, c) == want] or cands
            c = r.choice(pool)
            yes = inside(x, c)
            qa = QA(st.t.fmt("<is|was> {x} [inside|somewhere inside|in] {c}" + now + "?", x=x, c=c), "yes" if yes else "no",
                    why_of(x), "")
        elif q == "find":
            cands = [c for c in conts if c is not x]
            if not cands:
                continue
            c = r.choice(cands)
            yes = inside(x, c)
            who = r.choice(ps)
            qa = QA(st.t.fmt("if {p} <opens|opened> {c}" + (" and <looks|looked> through everything in it" if r.random() < 0.5 else "") +
                             ", <will|would> {p.s} find {x}?", p=who, c=c, x=x), "yes" if yes else "no", why_of(x), "")
        elif q == "top" and not rooms_mode and x in items and holder[id(x)] in conts:
            ch = chain(x)
            top = ch[-2]
            qa = QA(st.t.fmt("which thing on {f} <has|had> {x} [inside it|somewhere inside it]?", f=ch[-1], x=x)
                    .replace("the floor's", "the floor"), top.np, why_of(x), "")
        else:
            continue
        ask(qa, q, nest=len(chain(x)) - 1)
        if all(qa.q != o.q for o in qas):
            qas.append(qa)
    note(sub="rooms" if rooms_mode else "tables", depth=depth, moves=len(st.events))
    return st, qas


# ---------------------------------------------------------------- shells: under the cups
COVERS = ["cup", "bowl", "hat", "bucket", "beaker", "flowerpot"]
SWAP_C = ["{p} <swaps|swapped|had swapped> {a} and {b}", "{p} <switches|switched|had switched> {a} and {b} around",
          "{a} and {b} <are|were|had been> swapped"]
LIFT = ["{p} <lifts|lifted|had lifted> {a}, <takes|took|taken> {x} out and <puts|put|put> it under {b}",
        "{p} <moves|moved|had moved> {x} from under {a} to under {b}",
        "{p} <slides|slid|had slid> {x} out from under {a} and <hides|hid|hidden> it under {b}"]
SLIDE = ["{p} <slides|slid|had slid> {a} to the far {side} end", "{p} <moves|moved|had moved> {a} all the way to the {side}"]


def shells(r):
    n = r.randint(3, 5)
    noun = r.choice(COVERS)
    cups = [T(noun, c) for c in r.sample(COLOURS, n)]
    items = things(r, SMALL, r.randint(1, 2), colours=False)
    p = people(r, 1)[0]
    at = list(cups)
    r.shuffle(at)                          # left to right
    under = {id(i): r.choice(cups) for i in items}
    st = Story(r, cast=[p])
    st.add_intro(r.choice(["{p} <puts|put> {n} {noun}s upside down in a row on the table",
                           "there <are|were> {n} {noun}s in a row, upside down", "{p} <plays|played> the {noun} game: {n} {noun}s in a row"]),
                 p=p, n=n, noun=noun)
    for a, b in zip(at, at[1:]):
        st.fact(r.choice(["{a} <is|was> just to the left of {b}", "{b} <is|was> just to the right of {a}",
                          "to the right of {a} <is|was> {b}"]), a=a, b=b)
    for it in items:
        st.fact(r.choice(["{x} <is|was> under {c}", "{x} <is|was> hidden under {c}", "{c} <hides|hid> {x}",
                          "under {c} there <is|was> {x}", "{p} <hides|hid> {x} under {c}"]), x=it, c=under[id(it)], p=p)
    for _ in range(r.randint(0, 6)):
        k = r.random()
        if k < 0.45:
            a, b = r.sample(cups, 2)
            st.event(r.choice(SWAP_C), p=p, a=a, b=b)
            i, j = at.index(a), at.index(b)
            at[i], at[j] = at[j], at[i]
        elif k < 0.75:
            it = r.choice(items)
            b = r.choice([c for c in cups if c is not under[id(it)]])
            st.event(r.choice(LIFT), p=p, a=under[id(it)], b=b, x=it)
            under[id(it)] = b
        else:
            a = r.choice(cups)
            side = r.choice(["left", "right"])
            st.event(r.choice(SLIDE), p=p, a=a, side=side)
            at.remove(a)
            at.insert(0 if side == "left" else len(at), a)
    L = and_list([c.np for c in at])
    why = st.t.fmt("from left to right, the {noun}s <are|were> now {L}", noun=noun, L=L)
    qas = []
    for _ in range(10):
        if len(qas) >= r.choice((1, 2, 3)):
            break
        q = r.choice(["which", "which", "is", "where", "what"])
        it = r.choice(items)
        c = under[id(it)]
        if q == "which":
            qa = QA(st.t.fmt("which {noun} <hides|hid> {x} [now|at the end|]?", noun=noun, x=it), c.np, "",
                    st.t.fmt("{x} <is|was> under {c}", x=it, c=c))
        elif q == "is":
            d = c if r.random() < 0.5 else r.choice(cups)
            qa = QA(st.t.fmt("<is|was> {x} under {d}?", x=it, d=d), "yes" if d is c else "no", "", "")
        elif q == "where":
            k = at.index(c) + 1
            word = "on the left" if k == 1 else "on the right" if k == n else ("in the middle" if n == 3 else ordinal(k) + " from the left")
            qa = ask(QA(st.t.fmt("where <is|was> {x}, counting the {noun}s from the left?", x=it, noun=noun), ordinal(k),
                        sents(why, st.t.fmt("{x} <is|was> under {c}, the " + ordinal(k) + " one", x=it, c=c)),
                        st.t.fmt("{x} <is|was> under the {noun} " + word, x=it, noun=noun)), q, _max=n)
        else:
            d = r.choice(cups)
            hid = [i for i in items if under[id(i)] is d]
            qa = QA(st.t.fmt("what <is|was> under {d}?", d=d), hid[0].np if len(hid) == 1 else "nothing" if not hid
                    else and_list([i.np for i in hid]), "", "")
        ask(qa, q)
        if all(qa.q != o.q for o in qas):
            qas.append(qa)
    note(cups=n, things=len(items), moves=len(st.events))
    return st, qas


# ---------------------------------------------------------------- stacks: piles, and what comes down
PILE = ["book", "plate", "box", "brick", "blanket", "pancake", "cushion", "tile", "board", "towel", "basket", "tin"]
ON_TOP = ["{x} <is|was> on top of {y}", "{x} <lies|lay> on {y}", "{y} <is|was> under {x}", "{x} <rests|rested> on {y}",
          "{y} <has|had> {x} on it", "right under {x} <is|was> {y}", "on {y} <is|was> {x}"]


def stacks(r):
    noun = r.choice(PILE)
    n = r.randint(3, 6)
    pile = [T(noun, c) for c in r.sample(COLOURS, n + 2)]
    spare, pile = pile[n:], pile[:n]           # pile[0] is at the bottom
    p = people(r, 1)[0]
    st = Story(r, cast=[p])
    st.add_intro(r.choice(["there <is|was> a pile of {noun}s [on the table|on the floor|by the door]",
                           "{p} <has|had> made a pile of {noun}s", "{n} {noun}s <are|were> piled up, one on another"]),
                 p=p, noun=noun, n=n)
    for a, b in zip(pile[1:], pile):
        st.fact(r.choice(ON_TOP), x=a, y=b)
    if r.random() < 0.3:
        st.fact("{x} <is|was> at the bottom", x=pile[0])
    for _ in range(r.randint(0, 3)):
        k = r.random()
        if k < 0.35 and len(pile) > 2:
            x = r.choice(pile[:-1])
            i = pile.index(x)
            st.event(r.choice(["{p} <pulls|pulled|had pulled> {x} out of the pile, and what was on it <slides|slid|slid> down",
                               "{p} <slides|slid|had slid> {x} out from the middle; the {noun}s above it <come|came|came> down onto {y}"
                               if i > 0 else "{p} <pulls|pulled|had pulled> {x} out from the bottom, and the pile <sinks|sank|sunk> down"]),
                     p=p, x=x, noun=noun, y=pile[i - 1] if i > 0 else x)
            pile.remove(x)
        elif k < 0.6 and spare:
            w = spare.pop()
            st.event(r.choice(["{p} <puts|put|had put> {w} on top", "{p} <adds|added|had added> {w} to the top of the pile"]), p=p, w=w)
            pile.append(w)
        elif k < 0.8 and len(pile) > 2:
            x = pile[-1]
            st.event(r.choice(["{p} <takes|took|had taken> {x} off the top", "{p} <lifts|lifted|had lifted> {x} off the pile"]), p=p, x=x)
            pile.pop()
        elif spare:
            w = spare.pop()
            st.event("{p} <slides|slid|had slid> {w} under the whole pile", p=p, w=w)
            pile.insert(0, w)
    L = and_list([x.np for x in reversed(pile)])
    why = st.t.fmt("from the top down, the pile <is|was> now {L}", L=L)
    qas = []
    for _ in range(10):
        if len(qas) >= r.choice((1, 2, 3)):
            break
        q = r.choice(["top", "bottom", "on", "above", "count"])
        x = r.choice(pile)
        if q == "top":
            qa = QA(st.t.fmt("what <is|was> on top of the pile [now|]?"), pile[-1].np, why, "")
        elif q == "bottom":
            qa = QA(st.t.fmt("what <is|was> at the bottom [now|]?"), pile[0].np, why, "")
        elif q == "on" and pile.index(x) > 0:
            y = pile[pile.index(x) - 1]
            qa = QA(st.t.fmt("what <is|was> {x} lying on [now|]?", x=x), y.np, why,
                    st.t.fmt("{x} <is|was> on {y}", x=x, y=y))
        elif q == "above":
            y = r.choice([z for z in pile if z is not x])
            yes = pile.index(x) > pile.index(y)
            qa = QA(st.t.fmt("<is|was> {x} above {y}?", x=x, y=y), "yes" if yes else "no", why, "")
        elif q == "count":
            qa = QA(st.t.fmt("how many {noun}s <are|were> in the pile [now|]?", noun=noun), st.t.num(len(pile)), why, "")
        else:
            continue
        ask(qa, q)
        if all(qa.q != o.q for o in qas):
            qas.append(qa)
    note(size=len(pile), moves=len(st.events))
    return st, qas


# ---------------------------------------------------------------- walks: turning and walking
DIRS = ("north", "east", "south", "west")
STEP = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}
TURN_T = {"left": ["{p} <turns|turned|had turned> left", "{p} <goes|went|had gone> left", "{p} <takes|took|had taken> a left"],
          "right": ["{p} <turns|turned|had turned> right", "{p} <goes|went|had gone> right", "{p} <takes|took|had taken> a right"],
          "around": ["{p} <turns|turned|had turned> around", "{p} <turns|turned|had turned> back the way {p.s} came@"]}
WALK_T = ["{p} <walks|walked|had walked> {k}", "{p} <goes|went|had gone> {k} straight ahead",
          "{p} <walks|walked|had walked> {k} without turning", "{p} <keeps|kept|had kept> going for {k}",
          "{p} <covers|covered|had covered> {k} in a straight line"]


def walks(r):
    p = people(r, 1)[0]
    st = Story(r, cast=[p])
    lm = T(r.choice(PLACES_OUT))
    h = r.randrange(4)
    x = y = 0
    unit1 = r.choice(["block", "step", "street"])
    unit = unit1 + "s"
    st.add_intro(r.choice(["{p} <stands|stood> by {lm}, facing {d}", "{p} <starts|started> at {lm}, looking {d}",
                           "{p} <leaves|left> {lm} [on foot|in the morning|], facing {d}"]), p=p, lm=lm, d=DIRS[h])
    legs = r.randint(1, 6)
    for i in range(legs):
        turn = r.choice(["none", "left", "right", "left", "right", "around"]) if i or r.random() < 0.5 else "none"
        k = r.randint(1, 5)
        if turn != "none":
            h = (h + {"left": 3, "right": 1, "around": 2}[turn]) % 4
            t = r.choice(TURN_T[turn]).rstrip("@")
            if r.random() < 0.5:
                st.event(t + " and <walks|walked|walked> {k}", p=p, k=plural(k, unit1, not st.t.digits))
            else:
                st.event(t, p=p)
                st.event(r.choice(WALK_T), p=p, k=plural(k, unit1, not st.t.digits), unit="")
        else:
            st.event(r.choice(WALK_T), p=p, k=plural(k, unit1, not st.t.digits), unit="")
        dx, dy = STEP[h]
        x, y = x + dx * k, y + dy * k
    qas = []
    qas.append(ask(QA(st.t.fmt("which way <is|was> {p} facing " + st.t.now() + "?", p=p), DIRS[h], "",
                  st.t.fmt("{p} <is|was> facing " + DIRS[h], p=p)), "facing"))
    axes = [(y, "north", "south"), (x, "east", "west")]
    v, pos, neg = r.choice(axes)
    if v != 0:
        d = pos if v > 0 else neg
        qas.append(ask(QA(st.t.fmt(f"how many {unit} {d} of {{lm}} <is|was> {{p}} " + st.t.now() + "?", lm=lm, p=p), st.t.num(abs(v)),
                      "", st.t.fmt(f"{{p}} <is|was> {{k}} {d} of {{lm}}", p=p, k=plural(abs(v), unit1, not st.t.digits), lm=lm)), "distance"))
    d = r.choice(DIRS)
    yes = {"north": y > 0, "south": y < 0, "east": x > 0, "west": x < 0}[d]
    qas.append(ask(QA(st.t.fmt(f"<is|was> {{p}} [now|] {d} of {{lm}}?", p=p, lm=lm), "yes" if yes else "no", "", ""), "side"))
    if x == 0 and y == 0:
        qas.append(ask(QA(st.t.fmt("<is|was> {p} back where {p.s} started?", p=p), "yes", "", ""), "back"))
    note(legs=legs)
    r.shuffle(qas)
    st.keep_order = True
    return st, qas[:r.choice((1, 2, 3))]


def _route(st, w):
    """'the way goes from the hall through the kitchen to the attic'."""
    mid = (" through " + and_list([x.np for x in w[1:-1]])) if len(w) > 2 else ""
    return st.t.fmt("the way <goes|went> from {a}" + mid + " to {b}", a=w[0], b=w[-1])


# ---------------------------------------------------------------- paths: rooms joined by doors
DOOR_T = ["a door joins {a} and {b}", "there <is|was> a door between {a} and {b}", "{a} <opens|opened> onto {b}",
          "from {a} you can walk into {b}", "{a} and {b} share a door", "{a} <leads|led> into {b}",
          "a door in {a} <goes|went> to {b}", "{b} <is|was> through a door from {a}"]
WANT_T = ["{p} <wants|wanted> {x}", "{p} <needs|needed> {x}", "{p} <is|was> looking for {x}", "{p} <would like|wanted> {x}"]


def paths(r):
    n = r.randint(3, 6)
    rooms = [T(x) for x in r.sample(ROOMS, n)]
    nb = {id(x): [] for x in rooms}
    for i in range(1, n):
        a, b = rooms[i], rooms[r.randrange(i)]
        nb[id(a)].append(b)
        nb[id(b)].append(a)
    st = Story(r)
    for i in range(1, n):
        for b in nb[id(rooms[i])]:
            if rooms.index(b) < i:
                st.fact(r.choice(DOOR_T), a=rooms[i], b=b)
    # a way of up to 4 doors
    def way(a, b):
        prev, todo = {id(a): None}, [a]
        while todo:
            c = todo.pop(0)
            for d in nb[id(c)]:
                if id(d) not in prev:
                    prev[id(d)] = c
                    todo.append(d)
        out, c = [b], b
        while prev[id(c)] is not None:
            c = prev[id(c)]
            out.append(c)
        return out[::-1]
    for _ in range(50):
        a, b = r.sample(rooms, 2)
        w = way(a, b)
        if 2 <= len(w) <= 5:
            break
    p = people(r, 1)[0]
    st.cast = [p]
    it = things(r, SMALL, 1, colours=r.random() < 0.3)[0]
    st.fact("{p} <is|was> in {a}", p=p, a=a)
    st.fact(r.choice(["{x} <is|was> in {b}", "{x} <is|was> somewhere in {b}", "{x} <has|had> been left in {b}"]), x=it, b=b)
    st.fact(r.choice(WANT_T), p=p, x=it)
    why = _route(st, w)
    qas = []
    if len(w) > 2:
        qas.append(ask(QA(st.t.fmt("which room <does|did> {p} [walk into|reach|get to] first on the way to {x}?", p=p, x=it), w[1].np, why, ""), "first"))
    qas.append(ask(QA(st.t.fmt("how many doors <does|did> {p} [go through|open] to get to {x}?", p=p, x=it), st.t.num(len(w) - 1), why, ""), "doors"))
    c = r.choice(rooms)
    if c is not a and c is not b:
        qas.append(ask(QA(st.t.fmt("<does|did> {p} go through {c} on the way to {x}?", p=p, c=c, x=it), "yes" if c in w else "no", why, ""), "through"))
    qas.append(ask(QA(st.t.fmt("which room <is|was> {x} in?", x=it), b.np, "", ""), "where"))
    note(rooms=n, doors=len(w) - 1)
    r.shuffle(qas)
    return st, qas[:r.choice((1, 2, 3))]
