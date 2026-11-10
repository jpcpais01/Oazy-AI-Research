"""Field1, people: families, what people think, how many things people have, and the mixed house."""
from __future__ import annotations

from .core import (QA, Story, T, CONTAINERS, FOODS, OBJECTS_PLURAL, ROOMS, SMALL, P, and_list, people, things, plural)


# ---------------------------------------------------------------- family
class _Fam:
    def __init__(self):
        self.par, self.kids, self.spouse = {}, {}, {}

    def add(self, p, parents=()):
        self.par[id(p)] = list(parents)
        self.kids[id(p)] = []
        for q in parents:
            self.kids[id(q)].append(p)


def family(r):
    f = _Fam()
    gm, gf = people(r, 2, ["f", "m"])
    everyone = [gm, gf]
    f.add(gm), f.add(gf)
    f.spouse[id(gm)], f.spouse[id(gf)] = gf, gm
    n1 = r.randint(2, 4)
    gen1 = people(r, n1)
    gen1 = [p for p in gen1 if p.name not in (gm.name, gf.name)] or gen1
    for p in gen1:
        f.add(p, (gm, gf))
    everyone += gen1
    gen2 = []
    for p in r.sample(gen1, r.randint(1, min(2, len(gen1)))):
        sp = people(r, 1, ["m" if p.sex == "f" else "f"])[0]
        if any(sp.name == q.name for q in everyone):
            continue
        f.add(sp)
        f.spouse[id(p)], f.spouse[id(sp)] = sp, p
        everyone.append(sp)
        for c in people(r, r.randint(1, 3)):
            if any(c.name == q.name for q in everyone):
                continue
            f.add(c, (p, sp))
            gen2.append(c)
            everyone.append(c)
    st = Story(r, cast=everyone)
    word = {"f": {"par": "mother", "kid": "daughter", "sib": "sister", "gp": "grandmother", "gk": "granddaughter",
                  "pib": "aunt", "spouse": "wife"},
            "m": {"par": "father", "kid": "son", "sib": "brother", "gp": "grandfather", "gk": "grandson",
                  "pib": "uncle", "spouse": "husband"}}
    # the facts: couples, and each child tied to a parent or to a brother or sister already tied
    told = set()
    for a in everyone:
        b = f.spouse.get(id(a))
        if b is not None and (id(b), id(a)) not in told:
            told.add((id(a), id(b)))
            st.fact(r.choice(["{a} and {b} <are|were> married", "{a} <is|was> married to {b}", "{a} <is|was> {b}'s " + word[a.sex]["spouse"],
                              "{b}'s " + word[a.sex]["spouse"] + " <is|was> {a}"]), a=a, b=b)
    for unit in [x for x in everyone if f.kids[id(x)] and x.sex == "f"]:
        kids = f.kids[id(unit)]
        dad = f.spouse[id(unit)]
        if r.random() < 0.35 and len(kids) > 1:
            st.fact(r.choice(["{m} and {d} <have|had> {k} children: {L}", "the children of {m} and {d} <are|were> {L}"]),
                    m=unit, d=dad, k=len(kids), L=and_list([k.name for k in kids]))
            continue
        first = kids[0]
        par = r.choice([unit, dad])
        st.fact(r.choice(["{c} <is|was> {p}'s " + word[first.sex]["kid"], "{p} <is|was> {c}'s " + word[par.sex]["par"],
                          "{p} <is|was> the " + word[par.sex]["par"] + " of {c}", "{c}'s " + word[par.sex]["par"] + " <is|was> {p}",
                          "{p} <has|had> a " + word[first.sex]["kid"] + " called {c}"]), c=first, p=par)
        for c in kids[1:]:
            if r.random() < 0.5:
                sib = r.choice(kids[:kids.index(c)])
                st.fact(r.choice(["{c} <is|was> {s}'s " + word[c.sex]["sib"], "{s} <has|had> a " + word[c.sex]["sib"] + ", {c}",
                                  "{s}'s " + word[c.sex]["sib"] + " <is|was> {c}", "{c} and {s} <are|were> " +
                                  ("sisters" if c.sex == s_sex(sib) == "f" else "brothers" if c.sex == s_sex(sib) == "m" else
                                   "brother and sister")]), c=c, s=sib)
            else:
                par = r.choice([unit, dad])
                st.fact(r.choice(["{c} <is|was> {p}'s " + word[c.sex]["kid"], "{p} <is|was> {c}'s " + word[par.sex]["par"],
                                  "{p} <has|had> a " + word[c.sex]["kid"] + " called {c}"]), c=c, p=par)
    # relations, worked out from the tree
    parents = lambda x: f.par[id(x)]  # noqa: E731
    sibs = lambda x: [s for s in everyone if s is not x and parents(x) and set(map(id, parents(s))) == set(map(id, parents(x)))]  # noqa: E731
    grand = lambda x: [g for p in parents(x) for g in parents(p)]  # noqa: E731
    gkids = lambda x: [g for k in f.kids[id(x)] for g in f.kids[id(k)]]  # noqa: E731

    def piblings(x):
        out = []
        for p in parents(x):
            for s in sibs(p):
                out.append(s)
                if f.spouse.get(id(s)) is not None:
                    out.append(f.spouse[id(s)])
        return out
    cousins = lambda x: [k for s in [s for p in parents(x) for s in sibs(p)] for k in f.kids[id(s)]]  # noqa: E731
    rels = {
        "par": lambda x: parents(x), "kid": lambda x: f.kids[id(x)], "sib": sibs, "gp": grand, "gk": gkids,
        "pib": piblings, "cousin": cousins,
    }

    def word_of(rel, p):
        return "cousin" if rel == "cousin" else word[p.sex][rel]
    qas = []
    for _ in range(30):
        if len(qas) >= r.choice((1, 2, 3)):
            break
        x = r.choice(everyone)
        rel = r.choice(["par", "kid", "sib", "gp", "gp", "gk", "pib", "pib", "cousin"])
        found = rels[rel](x)
        q = r.choice(["who", "who", "is", "many"])
        if q == "who":
            sex = r.choice("fm")
            match = [p for p in found if p.sex == sex]
            if len(match) != 1 or rel == "cousin":
                continue
            w = word_of(rel, match[0])
            qa = QA(st.t.fmt("who <is|was> {x}'s " + w + "?", x=x), match[0].name, _why(st, rel, x, match[0], f, word),
                    st.t.fmt("{y} <is|was> {x}'s " + w, x=x, y=match[0]))
        elif q == "is":
            y = r.choice([p for p in everyone if p is not x])
            w = word_of(rel, y)
            yes = any(p is y for p in found)
            qa = QA(st.t.fmt("<is|was> {y} {x}'s " + w + "?", x=x, y=y), "yes" if yes else "no",
                    _why(st, rel, x, y, f, word) if yes else "", "")
        else:
            plural_w = {"kid": "children", "gk": "grandchildren", "cousin": "cousins", "sib": "brothers and sisters"}.get(rel)
            if not plural_w:
                continue
            k = len({id(p) for p in found})
            qa = QA(st.t.fmt("how many " + plural_w + " <does|did> {x} have?", x=x), st.t.num(k), "", "")
        if all(qa.q != o.q for o in qas):
            qas.append(qa)
    return st, qas


def s_sex(p):
    return p.sex


def _why(st, rel, x, y, f, word):
    par = f.par[id(x)]
    try:
        if rel == "gp":
            mid = next(p for p in par if y in f.par[id(p)])
            return st.t.fmt("{x}'s " + word[mid.sex]["par"] + " <is|was> {m}, and {m}'s " + word[y.sex]["par"] + " <is|was> {y}",
                            x=x, m=mid, y=y)
        if rel == "gk":
            mid = next(k for k in f.kids[id(x)] if y in f.kids[id(k)])
            return st.t.fmt("{m} <is|was> {x}'s " + word[mid.sex]["kid"] + ", and {y} <is|was> {m}'s " + word[y.sex]["kid"],
                            x=x, m=mid, y=y)
        if rel == "pib":
            for p in par:
                for s in [s for s in f.kids.get(id(f.par[id(p)][0]), [])] if f.par[id(p)] else []:
                    if s is y:
                        return st.t.fmt("{y} <is|was> the " + word[y.sex]["sib"] + " of {x}'s " + word[p.sex]["par"] + ", {p}",
                                        x=x, y=y, p=p)
                    if f.spouse.get(id(s)) is y:
                        return st.t.fmt("{y} <is|was> married to {s}, the " + word[s.sex]["sib"] + " of {x}'s " +
                                        word[p.sex]["par"], x=x, y=y, s=s)
        if rel == "sib":
            return st.t.fmt("{x} and {y} have the same parents", x=x, y=y)
    except StopIteration:
        return ""
    return ""


# ---------------------------------------------------------------- minds: people think what they last saw
SPOTS = ["box", "basket", "drawer", "cupboard", "bag", "chest", "jar", "bucket", "tin", "pot", "case", "sack"]
LEAVES = ["{p} <goes|went|had gone> out", "{p} <leaves|left|had left> the room", "{p} <goes|went|had gone> out to the garden",
          "{p} <pops|popped|had popped> out to the shop", "{p} <goes|went|had gone> upstairs for a while",
          "{p} <runs|ran|had run> off to play outside"]
COMES = ["{p} <comes|came|had come> back", "{p} <returns|returned|had returned>", "{p} <comes|came|had come> in again",
         "{p} <is|was|had been> back in the room"]
PUT = ["{p} <puts|put|had put> {x} in {c}", "{p} <hides|hid|had hidden> {x} in {c}", "{p} <leaves|left|had left> {x} in {c}"]
MOVE = ["{p} <moves|moved|had moved> {x} from {a} to {c}", "{p} <takes|took|had taken> {x} out of {a} and <puts|put|put> it in {c}",
        "{p} <moves|moved|had moved> {x} to {c}", "while nobody else <is|was> looking, {p} <moves|moved|had moved> {x} into {c}"]


def minds(r):
    ps = people(r, r.randint(2, 4))
    spots = [T(x, c) for x, c in zip(r.sample(SPOTS, r.randint(2, 4)), r.sample(["red", "blue", "green", "", "", ""], 4))]
    objs = things(r, SMALL, r.randint(1, 2), colours=False)
    st = Story(r, cast=ps)
    st.add_intro(r.choice(["{L} <are|were> in the [kitchen|living room|bedroom|classroom|hall]", "{L} <are|were> playing together",
                           "{L} <are|were> at home"]), L=and_list([p.name for p in ps]))
    where, thinks, why = {}, {}, {}
    here = set(map(id, ps))
    for o in objs:
        p, c = r.choice(ps), r.choice(spots)
        st.event(r.choice(PUT), p=p, x=o, c=c)
        where[id(o)] = c
        for q in ps:
            thinks[(id(q), id(o))] = c
            why[(id(q), id(o))] = ""
    for _ in range(r.randint(1, 4)):
        gone = [p for p in ps if id(p) not in here]
        if gone and r.random() < 0.4:
            p = r.choice(gone)
            st.event(r.choice(COMES), p=p)
            here.add(id(p))
        elif len(here) > 1 and r.random() < 0.8:
            p = r.choice([p for p in ps if id(p) in here])
            st.event(r.choice(LEAVES), p=p)
            here.discard(id(p))
        mover = r.choice([p for p in ps if id(p) in here])
        o = r.choice(objs)
        to = r.choice([c for c in spots if c is not where[id(o)]])
        st.event(r.choice(MOVE), p=mover, x=o, a=where[id(o)], c=to)
        for q in ps:
            if id(q) in here:
                thinks[(id(q), id(o))] = to
                why[(id(q), id(o))] = ""
            elif thinks[(id(q), id(o))] is not to:
                why[(id(q), id(o))] = st.t.fmt("{q} was not there when {m} moved {x} to {c}, so {q.s} still <thinks|thought> it <is|was> in {d}",
                                               q=q, m=mover, x=o, c=to, d=thinks[(id(q), id(o))])
        where[id(o)] = to
        if r.random() < 0.2 and len(here) < len(ps):          # someone who saw it tells someone who did not
            teller = r.choice([p for p in ps if id(p) in here])
            told = r.choice([p for p in ps if id(p) not in here])
            st.event("{t} <phones|phoned|had phoned> {u} and <says|said|said> that {x} <is|was|was> now in {c}", t=teller, u=told, x=o, c=to)
            thinks[(id(told), id(o))] = to
            why[(id(told), id(o))] = st.t.fmt("{t} <has|had> told {u} where it <is|was>", t=teller, u=told)
    qas = []
    wrong = [(p, o) for p in ps for o in objs if thinks[(id(p), id(o))] is not where[id(o)]]
    for _ in range(20):
        if len(qas) >= r.choice((1, 2, 3)):
            break
        q = r.choice(["look", "look", "think", "real", "know"])
        p, o = r.choice(wrong) if (wrong and r.random() < 0.6) else (r.choice(ps), r.choice(objs))
        t = thinks[(id(p), id(o))]
        if q == "look":
            qa = QA(st.t.fmt("where <will|would> {p} look for {x} first?", p=p, x=o), t.np, why[(id(p), id(o))],
                    st.t.fmt("{p} <will|would> look in {c}", p=p, c=t))
        elif q == "think":
            qa = QA(st.t.fmt("where <does|did> {p} think {x} <is|was>?", p=p, x=o), t.np, why[(id(p), id(o))], "")
        elif q == "real":
            qa = QA(st.t.fmt("where <is|was> {x} really?", x=o), where[id(o)].np, "", "")
        else:
            yes = t is where[id(o)]
            qa = QA(st.t.fmt("<does|did> {p} know where {x} <is|was>?", p=p, x=o), "yes" if yes else "no", why[(id(p), id(o))], "")
        if all(qa.q != o2.q for o2 in qas):
            qas.append(qa)
    return st, qas


# ---------------------------------------------------------------- count: how many, after gifts, finds and losses
def count(r):
    ps = people(r, r.randint(2, 3))
    food = r.random() < 0.4
    noun = r.choice(FOODS if food else OBJECTS_PLURAL)
    sing = {"cherries": "cherry", "peaches": "peach", "potatoes": "potato", "tomatoes": "tomato", "sweets": "sweet",
            "biscuits": "biscuit"}.get(noun, noun[:-1])
    have = {id(p): r.randint(1, 9) for p in ps}
    start = dict(have)
    st = Story(r, cast=ps)
    w = lambda k: plural(k, sing, not st.t.digits)  # noqa: E731
    for p in ps:
        st.fact(r.choice(["{p} <has|had> {k}", "{p} <starts|started> with {k}", "{p} <has|had> got {k}", "{p}'s bag <has|had> {k} in it"]),
                p=p, k=w(have[id(p)]))
    log = {id(p): [] for p in ps}
    for _ in range(r.randint(1, 5)):
        p = r.choice(ps)
        kind = r.choice(["give", "give", "find", "lose", "buy", "use"])
        if kind == "give" and have[id(p)] > 0:
            q = r.choice([x for x in ps if x is not p])
            k = r.randint(1, have[id(p)])
            st.event(r.choice(["{p} <gives|gave|had given> {k} to {q}", "{p} <gives|gave|had given> {q} {k}",
                               "{q} <gets|got|had got> {k} from {p}", "{p} <passes|passed|had passed> {k} to {q}",
                               "{q} <asks|asked|had asked> for some, and {p} <gives|gave|given> {q.o} {k}"]), p=p, q=q, k=w(k))
            have[id(p)] -= k
            have[id(q)] += k
            log[id(p)].append(f"<gives|gave> away {k}")
            log[id(q)].append(f"<gets|got> {k}")
        elif kind == "find":
            k = r.randint(1, 4)
            st.event(r.choice(["{p} <finds|found|had found> {k} more", "{p} <finds|found|had found> {k} [under the sofa|in a coat pocket|in the garden]"]),
                     p=p, k=w(k))
            have[id(p)] += k
            log[id(p)].append(f"<finds|found> {k}")
        elif kind == "lose" and have[id(p)] > 0:
            k = r.randint(1, have[id(p)])
            st.event(r.choice(["{p} <loses|lost|had lost> {k}", "{p} <loses|lost|had lost> {k} on the way home",
                               "{p} <loses|lost|had lost> {k} somewhere in the park"]), p=p, k=w(k))
            have[id(p)] -= k
            log[id(p)].append(f"<loses|lost> {k}")
        elif kind == "buy":
            k = r.randint(1, 5)
            st.event(r.choice(["{p} <buys|bought|had bought> {k} more", "{p} <gets|got|had got> {k} more at the shop"]), p=p, k=w(k))
            have[id(p)] += k
            log[id(p)].append(f"<buys|bought> {k}")
        elif kind == "use" and have[id(p)] > 0 and food:
            k = r.randint(1, have[id(p)])
            st.event("{p} <eats|ate|had eaten> {k}", p=p, k=w(k))
            have[id(p)] -= k
            log[id(p)].append(f"<eats|ate> {k}")
    def why(p):
        if not log[id(p)]:
            return st.t.fmt("{p} <has|had> {k} and nothing <changes|changed>", p=p, k=start[id(p)])
        return st.t.fmt("{p} <starts|started> with {k}, then " + ", then ".join(log[id(p)]) + ", which <leaves|left> {p.o} with {n}",
                        p=p, k=start[id(p)], n=have[id(p)])
    qas = []
    for _ in range(10):
        if len(qas) >= r.choice((1, 2, 3)):
            break
        q = r.choice(["many", "many", "more", "all", "diff"])
        p = r.choice(ps)
        if q == "many":
            qa = QA(st.t.fmt("how many " + noun + " <does|did> {p} have [now|at the end|]?", p=p), st.t.num(have[id(p)]), why(p),
                    st.t.fmt("{p} <has|had> {k}", p=p, k=w(have[id(p)])))
        elif q == "more":
            a, b = r.sample(ps, 2)
            if have[id(a)] == have[id(b)]:
                continue
            win = a if have[id(a)] > have[id(b)] else b
            qa = QA(st.t.fmt("who <has|had> more " + noun + ", {a} or {b}?", a=a, b=b), win.name,
                    st.t.fmt("{a} <has|had> {x} and {b} <has|had> {y}", a=a, b=b, x=have[id(a)], y=have[id(b)]), "")
        elif q == "all":
            tot = sum(have.values())
            qa = QA(st.t.fmt("how many " + noun + " <do|did> they have [in all|together|altogether]?"), st.t.num(tot),
                    " + ".join(str(have[id(p)]) for p in ps) + f" = {tot}", "")
        else:
            a, b = r.sample(ps, 2)
            if have[id(a)] <= have[id(b)]:
                a, b = b, a
            if have[id(a)] == have[id(b)]:
                continue
            qa = QA(st.t.fmt("how many more " + noun + " <does|did> {a} have than {b}?", a=a, b=b), st.t.num(have[id(a)] - have[id(b)]),
                    f"{have[id(a)]} - {have[id(b)]} = {have[id(a)] - have[id(b)]}", "")
        if all(qa.q != o.q for o in qas):
            qas.append(qa)
    return st, qas


# ---------------------------------------------------------------- house: carried about while someone is away
def house(r):
    n = r.randint(3, 5)
    rooms = [T(x) for x in r.sample(ROOMS, n)]
    nb = {id(x): [] for x in rooms}
    st = Story(r)
    for i in range(1, n):
        a, b = rooms[i], rooms[r.randrange(i)]
        nb[id(a)].append(b)
        nb[id(b)].append(a)
        st.fact(r.choice(["a door joins {a} and {b}", "there <is|was> a door between {a} and {b}", "{a} <opens|opened> onto {b}",
                          "from {a} you can walk into {b}"]), a=a, b=b)
    a, b = r.sample(ps_ := people(r, 2), 2)
    st.cast = ps_
    box = things(r, [c for c in CONTAINERS if c not in ("drawer",)], 1)[0]
    it = things(r, SMALL, 1, colours=False)[0]
    r0, r1 = r.sample(rooms, 2)
    st.fact("{x} <is|was> in {c}", x=it, c=box)
    st.fact("{c} <is|was> in {room}", c=box, room=r0)
    st.event(r.choice(["{a} <leaves|left|had left> the house", "{a} <goes|went|had gone> out to the shops", "{a} <goes|went|had gone> off to school"]), a=a)
    st.event(r.choice(["{b} <carries|carried|had carried> {c} to {room}", "{b} <takes|took|had taken> {c} into {room}"]), b=b, c=box, room=r1)
    st.event(r.choice(["{a} <comes|came|had come> home", "{a} <gets|got|had got> back"]), a=a)
    st.event(r.choice(["{a} <wants|wanted|had wanted> {x}", "{a} <needs|needed|had needed> {x}"]), a=a, x=it)
    st.keep_order = True

    def way(s, t):
        prev, todo = {id(s): None}, [s]
        while todo:
            c = todo.pop(0)
            for d in nb[id(c)]:
                if id(d) not in prev:
                    prev[id(d)] = c
                    todo.append(d)
        out, c = [t], t
        while prev[id(c)] is not None:
            c = prev[id(c)]
            out.append(c)
        return out[::-1]
    qas = [QA(st.t.fmt("which room <will|would> {a} go to first to look for {x}?", a=a, x=it), r0.np,
              st.t.fmt("{a} <was|was> out when {b} moved {c}, so {a} <thinks|thought> it <is|was> still in {r0}", a=a, b=b, c=box, r0=r0), ""),
           QA(st.t.fmt("which room <is|was> {x} really in?", x=it), r1.np,
              st.t.fmt("{x} <is|was> in {c}, and {b} carried {c} to {r1}", x=it, c=box, b=b, r1=r1), "")]
    w = way(r0, r1)
    from .fam_space import _route
    qas.append(QA(st.t.fmt("once {a} <finds|found> out, how many doors <must|did> {a.s} go through from {r0} to reach {x}?", a=a, r0=r0, x=it),
                  st.t.num(len(w) - 1), _route(st, w), ""))
    r.shuffle(qas)
    return st, qas[:r.choice((1, 2, 3))]
