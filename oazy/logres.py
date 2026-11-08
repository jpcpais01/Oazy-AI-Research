"""LogRes (logic and reasoning): can a model work out an answer the text never says?

Every question puts its own facts in front of the model, with made-up names and words, and asks for something
that follows from them but is never written down. Nothing can be looked up or remembered from training text:

    Bo is taller than Cy. Ana is taller than Bo.
    Question: Who is shorter, Ana or Cy?
    Answer: Cy

Eight kinds of puzzle, each at five levels (level = how many steps of reasoning the answer needs, 1 to 5):

  order      who is taller / older / faster (a chain of comparisons, said both ways: "taller" and "shorter")
  time       who arrived earlier (a chain of "before" and "after")
  place      which thing is further left (a row of objects, "left of" and "right of")
  sums       small sums of money and things: add, take away, price x count, change from a purse
  rules      if-then chains ("If the bell rings, the dog barks.") with a second chain that never starts
  sets       "All blims are dakes" chains, with a look-alike chain broken by one "Some"
  negation   "Ana is at the park, the shop or the lake. Ana is not at the park." (one place left each time)
  family     parent / child chains asked as grandparent, great-grandparent, ... and the other way round

1,600 questions (40 per kind and level), made once from a fixed seed and kept in the repo (oazy/logres_v1.jsonl),
so every model, now and later, answers exactly the same ones. The facts are given in a shuffled order, every
chain is said in both directions, which answer is right is balanced, and the answers are named in the question,
so shallow tricks (pick the name said last, said first, listed first, the shortest answer) score about the
guessing level; the self-tests check this.

Scored like HellaSwag (oazy/hellaswag.py): the model "picks" the answer that surprises it least, per letter (the
headline score) and per token. Guessing scores 50% on most questions and less where there are more answers (sums:
25%; negation: 1 in 2 to 1 in 6); the guessing level is always printed next to the score. Nothing is trained on it:
the training text is unchanged.

    python3 -m oazy.logres                                  # score Coral, Bee4, Pegasus 1 Mini, Plain-100M (+ SmolLM2)
    python3 -m oazy.logres --run pegasusmini --run plain100 # just these
    python3 -m oazy.logres --ckpt /path/ckpt.pt             # any checkpoint, print only
    python3 -m oazy.logres --show 3                         # print 3 example questions of every kind and level
    python3 -m oazy.logres --build                          # remake the question file (only if the generator changes)

Each scored run gets one {"kind": "logres"} line in its metrics.jsonl (the dashboard's LogRes column and its
"LogRes by level" table read it) and, once finished, "logres" in its status.json. The trainer runs it with the
other tests (oazy/train.py), so new runs get it on their own.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

VERSION = 1
SEED = 2026_11_08
PER_CELL = 40                            # questions per kind and level
LEVELS = (1, 2, 3, 4, 5)
FILE = Path(__file__).with_name(f"logres_v{VERSION}.jsonl")
FAMILIES = ("order", "time", "place", "sums", "rules", "sets", "negation", "family")
TITLES = {"order": "Order", "time": "Time", "place": "Place", "sums": "Sums", "rules": "If-then",
          "sets": "All / some", "negation": "Negation", "family": "Family"}

# made-up but name-like, short, no famous people
NAMES = ["Ana", "Bo", "Cy", "Dee", "Eli", "Fay", "Gus", "Hal", "Ivy", "Jo", "Kit", "Lou", "Mo", "Ned", "Oli", "Pip",
         "Rue", "Sal", "Tam", "Uma", "Vic", "Wes", "Zed", "Abe", "Bea", "Cal", "Dot", "Eve", "Finn", "Gil", "Hugo",
         "Ida", "Jem", "Kai", "Lev", "Mia", "Nia", "Otto", "Pia", "Rex", "Sam", "Tia", "Ula", "Val", "Wyn", "Zoe"]
COMPARE = [("taller", "shorter"), ("older", "younger"), ("faster", "slower"), ("heavier", "lighter"),
           ("stronger", "weaker"), ("richer", "poorer")]
VERBS = ["arrived", "woke up", "left", "finished", "went to bed", "came home", "started", "ate lunch"]
THINGS = ["cup", "lamp", "book", "vase", "clock", "plant", "box", "bowl", "shoe", "hat", "key", "pen", "mug",
          "jar", "bell", "fan", "doll", "kite"]
ITEMS = ["apple", "pear", "egg", "candle", "stamp", "shell", "card", "ribbon", "pencil", "button", "marble", "cookie"]
EVENTS = ["the bell rings", "the dog barks", "the baby wakes up", "the lamp turns on", "the door opens",
          "the cat hides", "the kettle whistles", "the bird sings", "the gate closes", "the phone buzzes",
          "the light goes out", "the fan starts", "the horn sounds", "the clock stops", "the fish jumps",
          "the alarm beeps", "the window shuts", "the music plays", "the boat leaves", "the train stops",
          "the dog sleeps", "the snow falls", "the tap drips", "the bus arrives", "the owl hoots",
          "the drum beats", "the cart moves", "the flag rises"]
WORDS = ["blim", "dake", "fep", "gorp", "hask", "jiv", "kep", "lorn", "mib", "nusk", "pold", "quam", "rull",
         "sarp", "tib", "vosk", "wug", "yom", "zent", "bip", "dop", "fesk", "glim", "hob", "jask", "krell",
         "lup", "mork", "narb", "pib"]
PLACES = ["the park", "the shop", "the school", "the beach", "the farm", "the lake", "the zoo", "the gym",
          "the bank", "the mill", "the pier", "the barn", "the cafe", "the hill"]
SPOTS = ["the box", "the bag", "the drawer", "the jar", "the basket", "the cupboard", "the shoe", "the pot",
         "the tin", "the chest", "the sack", "the bucket"]


def _or(xs):
    """'A or B', 'A, B or C'."""
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " or " + xs[-1]


def _cap(s):
    return s[0].upper() + s[1:]


def _item(family, level, facts, question, endings, label, prefix=""):
    ctx = " ".join(facts) + "\nQuestion: " + question + "\nAnswer:" + prefix
    return {"family": family, "level": level, "ctx": ctx, "endings": endings, "label": label}


def _two(rng, a, b):
    """Two answers in random order, and which one is a."""
    return ([a, b], 0) if rng.random() < 0.5 else ([b, a], 1)


# ---------------------------------------------------------------------------------------- the eight kinds of puzzle
def _chain_pair(rng, L):
    """A chain of L+1 or L+2 (from level 2) members; a pair L steps apart on it. Returns (n, i, j), i < j."""
    n = L + 1 + (rng.random() < 0.5 if L >= 2 else 0)
    i = rng.randrange(n - L)
    return n, i, i + L


def gen_order(rng, L):
    up, down = rng.choice(COMPARE)
    n, i, j = _chain_pair(rng, L)
    p = rng.sample(NAMES, n)                       # p[0] is the most `up` (tallest)
    facts = [f"{p[k]} is {up} than {p[k + 1]}." if rng.random() < 0.5 else f"{p[k + 1]} is {down} than {p[k]}."
             for k in range(n - 1)]
    rng.shuffle(facts)
    ask_up = rng.random() < 0.5
    ends, lab = _two(rng, p[i] if ask_up else p[j], p[j] if ask_up else p[i])
    return _item("order", L, facts, f"Who is {up if ask_up else down}, {ends[0]} or {ends[1]}?", ends, lab)


def gen_time(rng, L):
    verb = rng.choice(VERBS)
    n, i, j = _chain_pair(rng, L)
    p = rng.sample(NAMES, n)                       # p[0] did it first
    facts = [f"{p[k]} {verb} before {p[k + 1]}." if rng.random() < 0.5 else f"{p[k + 1]} {verb} after {p[k]}."
             for k in range(n - 1)]
    rng.shuffle(facts)
    early = rng.random() < 0.5
    ends, lab = _two(rng, p[i] if early else p[j], p[j] if early else p[i])
    return _item("time", L, facts, f"Who {verb} {'earlier' if early else 'later'}, {ends[0]} or {ends[1]}?", ends, lab)


def gen_place(rng, L):
    n, i, j = _chain_pair(rng, L)
    t = rng.sample(THINGS, n)                      # t[0] is the leftmost
    facts = [f"The {t[k]} is to the left of the {t[k + 1]}." if rng.random() < 0.5
             else f"The {t[k + 1]} is to the right of the {t[k]}." for k in range(n - 1)]
    rng.shuffle(facts)
    left = rng.random() < 0.5
    ends, lab = _two(rng, "the " + (t[i] if left else t[j]), "the " + (t[j] if left else t[i]))
    return _item("place", L, facts, f"Which is further {'left' if left else 'right'}, {ends[0]} or {ends[1]}?", ends, lab)


def _numbers(rng, right, mistakes, shown):
    """Four answers: the right one, then wrong ones (the usual mistakes first, then near numbers) chosen so the right
    answer is equally often the smallest, 2nd, 3rd or largest. No answer is a number written in the question."""
    for _ in range(100):
        r = rng.randrange(4)                       # how many wrong answers are smaller than the right one
        pool = [m for m in mistakes if m != right and m not in shown and m >= 0]
        near = [right + d for d in range(-12, 13) if d and right + d >= 0 and right + d not in shown]
        lo = [m for m in pool if m < right]
        hi = [m for m in pool if m > right]
        rng.shuffle(lo)
        rng.shuffle(hi)
        nlo = [m for m in near if m < right and m not in lo]
        nhi = [m for m in near if m > right and m not in hi]
        rng.shuffle(nlo)
        rng.shuffle(nhi)
        lo, hi = (lo + nlo)[:r], (hi + nhi)[:3 - r]
        if len(lo) == r and len(hi) == 3 - r:
            ends = sorted(lo + hi + [right])
            return [str(v) for v in ends], ends.index(right)
    raise RuntimeError("no answers")


def gen_sums(rng, L):
    who = rng.choice(NAMES)
    it = rng.choice(ITEMS)
    while True:
        if L == 1:                                 # add
            a, b = rng.randint(2, 9), rng.randint(2, 9)
            facts = [f"{who} has {a} {it}s.", f"{who} gets {b} more {it}s."]
            q, right, mis, shown = f"How many {it}s does {who} have now?", a + b, [a + b - 1, a + b + 1, abs(a - b)], {a, b}
        elif L == 2:                               # add, then take away
            a, b = rng.randint(3, 9), rng.randint(2, 9)
            c = rng.randint(1, a + b - 1)
            facts = [f"{who} has {a} {it}s.", f"{who} gets {b} more {it}s.", f"Then {who} gives away {c} {it}s."]
            q, right, mis, shown = f"How many {it}s does {who} have now?", a + b - c, [a + b, a - c, a + b + c], {a, b, c}
        elif L == 3:                               # price x count
            pr, c = rng.randint(2, 6), rng.randint(2, 6)
            facts = [f"One {it} costs {pr} coins.", f"{who} buys {c} {it}s."]
            q, right, mis, shown = f"How many coins does {who} pay?", pr * c, [pr + c, pr * (c - 1), pr * (c + 1)], {pr, c}
        elif L == 4:                               # price x count, then change from a purse
            pr, c = rng.randint(2, 6), rng.randint(2, 5)
            w = pr * c + rng.randint(1, 15)
            facts = [f"{who} has {w} coins.", f"One {it} costs {pr} coins.", f"{who} buys {c} {it}s."]
            q, right, mis, shown = f"How many coins does {who} have left?", w - pr * c, [pr * c, w - pr, w - pr - c], {w, pr, c}
        else:                                      # two prices x two counts, added
            it2 = rng.choice([x for x in ITEMS if x != it])
            p1, p2, c1, c2 = rng.randint(2, 5), rng.randint(2, 5), rng.randint(2, 4), rng.randint(2, 4)
            facts = [f"One {it} costs {p1} coins and one {it2} costs {p2} coins.", f"{who} buys {c1} {it}s and {c2} {it2}s."]
            right = p1 * c1 + p2 * c2
            q, mis, shown = f"How many coins does {who} pay?", [p1 * c1, p2 * c2, (p1 + p2) * (c1 + c2), p1 + p2 + c1 + c2], {p1, p2, c1, c2}
        if right not in shown and right >= 0:
            break
    ends, lab = _numbers(rng, right, mis, shown)
    return _item("sums", L, facts, q[:-1] + f": {_or(ends)}?", ends, lab)


def gen_rules(rng, L):
    ev = rng.sample(EVENTS, 2 * (L + 1))
    a, b = ev[:L + 1], ev[L + 1:]                 # two chains of L rules; only a's first event happens
    rules = [f"If {c[k]}, {c[k + 1]}." for c in (a, b) for k in range(L)]
    rng.shuffle(rules)
    start = _cap(a[0]) + "."
    facts = rules + [start] if rng.random() < 0.5 else [start] + rules
    ends, lab = _two(rng, a[L], b[L])
    return _item("rules", L, facts, f"Which surely happens, {ends[0]} or {ends[1]}?", ends, lab)


def gen_sets(rng, L):
    w = rng.sample(WORDS, 2 * L + 1)
    a = w[:L + 1]                                  # a[0] -> a[1] -> ... -> a[L], all "All"
    b = [a[0]] + w[L + 1:]                         # the look-alike branch from a[0]: one of its links is only "Some"
    weak = rng.randrange(L)
    facts = [f"All {a[k]}s are {a[k + 1]}s." for k in range(L)]
    facts += [f"{'Some' if k == weak else 'All'} {b[k]}s are {b[k + 1]}s." for k in range(L)]
    rng.shuffle(facts)
    who = rng.choice(NAMES)
    member = f"{who} is a {a[0]}."
    facts = facts + [member] if rng.random() < 0.5 else [member] + facts
    ends, lab = _two(rng, a[L], b[L])
    return _item("sets", L, facts, f"Which must {who} be, a {ends[0]} or a {ends[1]}?", ends, lab, prefix=f" {who} is a")


def gen_negation(rng, L):
    if rng.random() < 0.5:
        who, opts = rng.choice(NAMES), rng.sample(PLACES, L + 1)
        first, q = f"{who} is at {_or(opts)}.", f"Where is {who}"
        neg = lambda o: f"{who} is not at {o}."                       # noqa: E731
    else:
        thing, opts = rng.choice(["key", "ring", "coin", "ball", "note", "sock"]), rng.sample(SPOTS, L + 1)
        who, first, q = thing, f"The {thing} is in {_or(opts)}.", f"Where is the {thing}"
        neg = lambda o: f"It is not in {o}." if rng.random() < 0.5 else f"The {thing} is not in {o}."   # noqa: E731
    right = rng.choice(opts)
    nots = [neg(o) for o in opts if o != right]
    rng.shuffle(nots)
    ends = opts[:]
    rng.shuffle(ends)
    return _item("negation", L, [first] + nots, q + f": {_or(ends)}?", ends, ends.index(right))


def _kin(L, up):
    if L == 1:
        return "parent" if up else "child"
    return "great-" * (L - 2) + ("grandparent" if up else "grandchild")


def gen_family(rng, L):
    n = L + 2                                      # one more person than the question needs
    p = rng.sample(NAMES, n)                       # p[0] the oldest; p[k] is p[k+1]'s parent
    facts = [f"{p[k]} is {p[k + 1]}'s parent." if rng.random() < 0.5 else f"{p[k + 1]} is {p[k]}'s child."
             for k in range(n - 1)]
    rng.shuffle(facts)
    i = rng.randrange(n - L)
    up = rng.random() < 0.5                        # ask for the older one (of the younger) or the other way round
    x, right = (p[i + L], p[i]) if up else (p[i], p[i + L])
    wrong = rng.choice([q for q in p if q not in (x, right)])
    ends, lab = _two(rng, right, wrong)
    return _item("family", L, facts, f"Who is {x}'s {_kin(L, up)}, {ends[0]} or {ends[1]}?", ends, lab)


GEN = {"order": gen_order, "time": gen_time, "place": gen_place, "sums": gen_sums, "rules": gen_rules,
       "sets": gen_sets, "negation": gen_negation, "family": gen_family}


def build(seed: int = SEED, per_cell: int = PER_CELL):
    """Every question, in a fixed order (kind, then level). Each kind and level has its own random stream, so
    changing one kind never changes the questions of another. Within each kind and level, each answer slot (first,
    second, ...; for sums: smallest, 2nd, ...) holds the right answer equally often."""
    out = []
    for fi, fam in enumerate(FAMILIES):
        for L in LEVELS:
            rng = random.Random(seed * 1000 + fi * 10 + L)
            seen, labels = set(), defaultdict(int)
            while len(seen) < per_cell:
                it = GEN[fam](rng, L)
                cap = -(-per_cell // len(it["endings"]))       # each answer slot is right equally often
                if it["ctx"] in seen or labels[it["label"]] >= cap:
                    continue
                seen.add(it["ctx"])
                labels[it["label"]] += 1
                it["id"] = f"{fam}-{L}-{len(seen):02d}"
                out.append(it)
    return out


def load(path: Path = FILE):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def subset(items, per_cell: int):
    """The first per_cell questions of every kind and level (the trainer's small test presets use a few)."""
    seen = defaultdict(int)
    out = []
    for it in items:
        if seen[(it["family"], it["level"])] < per_cell:
            seen[(it["family"], it["level"])] += 1
            out.append(it)
    return out


def write(path: Path = FILE):
    rows = build()
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


# ---------------------------------------------------------------------------------------------------- scoring
def evaluate(model, device, items=None, batch: int = 64):
    """Score a model on LogRes. Works for any model with _body(idx) -> logits and a cfg.tokenizer (every Oazy model,
    Pegasus included, and the SmolLM2 / GPT-2 yardsticks).

    The questions are run in groups of exactly the same length, so there is never any padding: a model's answer
    can not depend on which other questions shared its batch (that matters for Pegasus, whose thoughts are cut
    from the text it is given)."""
    import torch
    import torch.nn.functional as F
    from . import tok
    items = items if items is not None else load()
    enc = tok.for_model(model)
    seqs = []                                       # (item, answer, ids, first answer position)
    for qi, it in enumerate(items):
        ctx = enc.encode_ordinary(it["ctx"])
        for ai, e in enumerate(it["endings"]):
            seqs.append((qi, ai, ctx + enc.encode_ordinary(" " + e), len(ctx)))
    by_len = defaultdict(list)
    for s in seqs:
        by_len[len(s[2])].append(s)
    tot = [[0.0] * len(it["endings"]) for it in items]
    cnt = [[0] * len(it["endings"]) for it in items]
    was = model.training
    model.eval()
    with torch.no_grad():
        for T, group in sorted(by_len.items()):
            step = max(1, min(batch, 4096 // T))   # at most ~4k tokens at once: the scores over every word piece are big
            for b0 in range(0, len(group), step):
                g = group[b0:b0 + step]
                x = torch.tensor([s[2] for s in g], dtype=torch.long, device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
                    logits = model._body(x)
                loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1),
                                       reduction="none").view(len(g), T - 1)
                for row, (qi, ai, ids, a) in zip(loss, g):
                    tot[qi][ai] = float(row[a - 1:].sum())
                    cnt[qi][ai] = T - a
    if was:
        model.train()
    per = []
    for it, t, c in zip(items, tot, cnt):
        chars = [len(" " + e) for e in it["endings"]]
        pick = lambda v: min(range(len(v)), key=v.__getitem__)    # noqa: E731
        per.append({"family": it["family"], "level": it["level"], "chance": 1 / len(it["endings"]),
                    "char": int(pick([a / b for a, b in zip(t, chars)]) == it["label"]),
                    "tok": int(pick([a / b for a, b in zip(t, c)]) == it["label"])})
    return summarize(per)


def summarize(per):
    """{"acc_char", "acc_norm", "chance", "n", "levels": {"1": {...}}, "families": {...}, "grid": {kind: [L1..L5]}}."""
    def agg(rows):
        n = max(len(rows), 1)
        return {"acc": sum(r["char"] for r in rows) / n, "acc_norm": sum(r["tok"] for r in rows) / n,
                "chance": sum(r["chance"] for r in rows) / n, "n": len(rows)}
    al = agg(per)
    levels = {str(L): agg([r for r in per if r["level"] == L]) for L in LEVELS}
    fams = {f: agg([r for r in per if r["family"] == f]) for f in FAMILIES}
    grid = {f: [round(agg([r for r in per if r["family"] == f and r["level"] == L])["acc"], 4) for L in LEVELS]
            for f in FAMILIES}
    return {"version": VERSION, "acc_char": al["acc"], "acc_norm": al["acc_norm"], "chance": al["chance"], "n": al["n"],
            "levels": levels, "families": fams, "grid": grid}


# ------------------------------------------------------------------------------------------------ the command
REFS = ("smollm2-360m", "smollm2-135m", "openai-gpt2")
DEFAULT_RUNS = ("coral", "bee4", "pegasusmini", "pegasus2", "plain100") + REFS
NICE = {"coral": "Coral", "bee4": "Bee4", "pegasusmini": "Pegasus 1 Mini", "pegasus2": "Pegasus 2 Mini", "plain100": "Plain-100M",
        "pegasus": "Pegasus 1", "coralmini": "Coral Mini", "smollm2-360m": "SmolLM2-360M",
        "smollm2-135m": "SmolLM2-135M", "openai-gpt2": "GPT-2 124M"}


def _load(run: Path, device):
    if run.name == "openai-gpt2":
        from .gpt import load_openai_gpt2
        return load_openai_gpt2(device)
    if run.name in ("smollm2-360m", "smollm2-135m"):
        from .compare import smol
        return smol(run.name, device)
    from .train import load_model
    try:
        return load_model(run / "ckpt.pt", device, mmap=True)[0]
    except RuntimeError:                           # an old-style checkpoint file cannot be read lazily
        return load_model(run / "ckpt.pt", device)[0]


def _records(run: Path):
    p = run / "metrics.jsonl"
    out = []
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def save(run: Path, r: dict):
    """One {"kind": "logres"} line in the run's metrics.jsonl, and "logres" in status.json unless the run is live
    (its trainer rewrites status.json every few seconds; the dashboard reads the metrics line either way)."""
    from .train import Run
    recs = _records(run)
    base = {k: recs[-1].get(k, 0) for k in ("step", "tokens", "elapsed")} if recs else {"step": 0, "tokens": 0, "elapsed": 0}
    Run(run).log("metrics.jsonl", {"kind": "logres", **base, **r, "full": True, "scored": time.time()})
    try:
        st = json.loads((run / "status.json").read_text())
    except (OSError, json.JSONDecodeError):
        return
    if st.get("state") in ("running", "starting", "finishing"):
        return
    st.setdefault("latest", {})["logres"] = r["acc_char"]
    tmp = run / "status.json.tmp"
    tmp.write_text(json.dumps(st))
    os.replace(tmp, run / "status.json")


def pct(v):
    return "–" if v is None else f"{v * 100:.0f}%"


def table(results):
    """Print the scores by level and by kind, every model side by side, guessing level last."""
    from . import term
    names = list(results)
    chance = next(iter(results.values()))
    term.header("LogRes · logic and reasoning", f"{chance['n']} questions · per letter · higher is better")
    body = [[term.c(NICE.get(n, n), "bold"), pct(results[n]["acc_char"])] + [pct(results[n]["levels"][str(L)]["acc"]) for L in LEVELS]
            for n in names]
    body.append([term.c("guessing", "dim"), pct(chance["chance"])] + [pct(chance["levels"][str(L)]["chance"]) for L in LEVELS])
    term.table(["model", "all"] + [f"level {L}" for L in LEVELS], body, right=tuple(range(1, 7)))
    body = [[TITLES[f], pct(next(iter(results.values()))["families"][f]["chance"])]
            + [pct(results[n]["families"][f]["acc"]) for n in names] for f in FAMILIES]
    term.table(["kind", "guessing"] + [NICE.get(n, n) for n in names], body, right=tuple(range(1, len(names) + 2)))
    term.info("level = steps of reasoning the answer needs. Per token (another way to count) is in metrics.jsonl.")


def show(k: int):
    rows = load()
    for fam in FAMILIES:
        for L in LEVELS:
            for it in [r for r in rows if r["family"] == fam and r["level"] == L][:k]:
                print(f"--- {it['id']}  (right: {it['endings'][it['label']]})")
                print(it["ctx"] + " " + it["endings"][it["label"]])


def main():
    from .data import default_dir, run_dir
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", help="run folder name (repeat for several); default: " + ", ".join(DEFAULT_RUNS))
    ap.add_argument("--ckpt", default=None, help="score one checkpoint file and only print")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--build", action="store_true", help="remake the question file from the generator")
    ap.add_argument("--show", type=int, default=0, help="print N example questions of every kind and level")
    a = ap.parse_args()
    if a.build:
        rows = write()
        print(f"{FILE}: {len(rows)} questions")
        return
    if a.show:
        show(a.show)
        return
    import torch
    from .data import few_cpu_threads
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    few_cpu_threads(device)
    items = load()
    results = {}
    if a.ckpt:
        from .train import load_model
        model, _ = load_model(Path(a.ckpt), device)
        results[Path(a.ckpt).parent.name] = evaluate(model, device, items, a.batch)
        table(results)
        return
    runs = Path(a.runs or default_dir("runs"))
    for name in a.run or DEFAULT_RUNS:
        d = run_dir(runs, name)
        if name not in REFS and not (d / "ckpt.pt").exists():
            print(f"{name}: no checkpoint in {d}, skipped", flush=True)
            continue
        if name in REFS and not d.exists() and not a.run:
            continue                               # a yardstick only when it was measured here (oazy.compare)
        t0 = time.time()
        try:
            model = _load(d, device)
            r = evaluate(model, device, items, a.batch)
        except Exception as e:                     # one broken checkpoint (or no internet for SmolLM2) skips one model
            print(f"{name}: skipped ({type(e).__name__}: {str(e)[:200]})", flush=True)
            continue
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        d.mkdir(parents=True, exist_ok=True)
        save(d, r)
        results[name] = r
        print(f"{NICE.get(name, name)}: {r['acc_char'] * 100:.1f}% (guessing {r['chance'] * 100:.0f}%), "
              + " ".join(f"L{L} {r['levels'][str(L)]['acc'] * 100:.0f}%" for L in LEVELS) + f"  ({time.time() - t0:.0f} s)", flush=True)
    if results:
        table(results)


if __name__ == "__main__":
    main()
