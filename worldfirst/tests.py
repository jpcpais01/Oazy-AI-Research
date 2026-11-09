"""Checks for stage 1: the world tells the truth, the secret exams are walled off, and both models run.

    python3 -m worldfirst.tests            # about a minute on a CPU
"""
from __future__ import annotations

import json
import random
import math
import sys
import traceback

from . import world as W

PASSED, FAILED = [], []


def check(fn):
    try:
        fn()
        PASSED.append(fn.__name__)
        print(f"  ok    {fn.__name__}")
    except Exception:
        FAILED.append(fn.__name__)
        print(f"  FAIL  {fn.__name__}")
        traceback.print_exc()
    return fn


def train_eps(n_per=150):
    for fam in W.FAMILIES:
        for lv in W.TRAIN_LEVELS[fam]:
            for s in range(n_per):
                yield W.make(fam, lv, "train", 2_000_000 + s)


def secret_eps(n_per=150):
    for fam in W.FAMILIES:
        for sp, lvs in W.SECRET[fam].items():
            for lv in lvs:
                for s in range(n_per):
                    yield W.make(fam, lv, sp, s)


@check
def same_seed_same_episode():
    for fam, lv, sp in W.exam_sets():
        assert W.make(fam, lv, sp, 5).text() == W.make(fam, lv, sp, 5).text()
    assert W.make("lines", 2, "train", 1).text() != W.make("lines", 2, "train", 2).text()


@check
def every_episode_is_well_formed():
    for e in list(train_eps(40)) + list(secret_eps(40)):
        assert e.questions, e.text()
        assert len(e.tokens) == len(e.mentions)
        for q in e.questions:
            assert len(q.answer) == len(q.specs) and len(q.tokens) == len(q.mentions)
            for (kind, tgt), w in zip(q.specs, q.answer):
                if kind == "class":
                    assert W.CLASSES[tgt] == w
                elif kind == "pointer":
                    assert e.things[tgt] == w
                else:
                    assert str(tgt[1]) == w
        keys = [k for k in e.things]
        assert len(set(keys)) == len(keys), ("each thing needs its own name", e.text())


@check
def sums_are_right():
    for e in [W.make("sums", lv, "train", s) for lv in W.TRAIN_LEVELS["sums"] for s in range(100)]:
        a, b = int("".join(map(str, e.facts["a"]))), int("".join(map(str, e.facts["b"])))
        assert int("".join(e.questions[0].answer)) == a + b


@check
def lines_answers_follow_the_order():
    for s in range(300):
        e = W.make("lines", 3, "train", s)
        order = e.facts["order"]
        for q in e.questions:
            t = " ".join(q.tokens)
            if t.startswith("is "):
                x, y = q.tokens[1], q.tokens[4]
                assert (order.index(x) < order.index(y)) == (q.answer == ["yes"]), t
            elif "front of the line" in t:
                assert q.answer == [order[0]]
            elif "back of the line" in t:
                assert q.answer == [order[-1]]


@check
def falling_blocks_rest_on_something():
    for e in secret_eps(60):
        if e.family != "falling":
            continue
        lv = e.facts["level"]
        assert all(0 <= v <= 9 for v in lv.values())
        assert len(set(lv.values())) == len(lv), "two blocks at one level"


@check
def secret_exams_are_walled_off():
    """No training episode can be a secret one: checked on the structure of thousands of episodes."""
    def depth(parent, x):
        d = 0
        while parent.get(x, "table") != "table":
            x = parent[x]
            d += 1
        return d

    def secret_like(e):
        f = e.facts
        if e.family == "sums":
            return len(f["a"]) - 1 >= 11
        if e.family == "lines":
            return len(f["order"]) >= 11 or f["moves"] >= 4
        if e.family == "boxes":
            sp = f["start_parent"]
            return max(depth(sp, x) for x in sp) >= 8 or f["moves"] >= 4
        if e.family == "family":
            return any(len(c) >= 4 or W._new_link(c) for c in f["asked"])
        if e.family == "falling":
            return f["movers"] >= 6
        if e.family == "machines":
            return tuple(f["rule"]) in set(W.MACHINES["new-mixes"]) | set(W.MACHINES["deeper"])
        if e.family == "cups":
            return f["moves"] >= 8
        if e.family == "minds":
            return f["moves"] >= 5
        if e.family == "walks":
            return f["legs"] >= 7
        if e.family == "carry":
            return f["deepest"] >= 5 or f["moves"] >= 3
        if e.family in ("time", "cause"):
            return f["depth"] >= 5
        if e.family == "wants":
            return f["dist"] >= 5
        if e.family == "count":
            return f["moves"] >= 7
        if e.family == "house":
            return any(k in W.HOUSE_SECRET_KINDS for k in f["kinds"])
        raise AssertionError("a family with no wall check: " + e.family)
    assert W.VERSION == 2
    n = 0
    for e in train_eps(150):
        assert not secret_like(e), e.text()
        n += 1
    for e in secret_eps(40):
        assert secret_like(e), (e.family, e.split, e.text())
    assert n >= 5000


@check
def machine_rules_are_new_and_answers_unique():
    seen = set(W.MACHINES[1]) | set(W.MACHINES[2])
    assert not (seen & set(W.MACHINES["new-mixes"])) and not (seen & set(W.MACHINES["deeper"]))
    assert len(W.MACHINES["new-mixes"]) == len(W.MACH_HELD_OUT)
    for sp in ("new-mixes", "deeper"):
        for s in range(80):
            e = W.make("machines", 2 if sp == "new-mixes" else 3, sp, s)
            f = e.facts["rule"]
            val = {e.things[m]: int(t) for t, m in zip(e.tokens, e.mentions) if m >= 0}
            ex = [(int(k[3:]), v) for k, v in val.items() if k.startswith("out")]
            assert len(ex) == sum(k.startswith("in") for k in val) >= 4
            fits = [g for g in W.MACH_UNIVERSE if all(g[x] == y for x, y in ex)]
            for q in e.questions:
                x = int(q.tokens[-2])
                assert all(g[x] == f[x] for g in fits), "the examples must pin the answer down"


@check
def family_reversals_are_tagged():
    n = {True: 0, False: 0}
    for s in range(300):
        for q in W.make("family", 1, "train", s).questions:
            if "reversed" in q.tags:
                n[q.tags["reversed"]] += 1
    assert n[True] > 50 and n[False] > 50, n


@check
def more_moves_are_followed():
    """Several moves in a row: the line after them is what replaying the moves gives."""
    for sp, lv in (("train", 3), ("more-moves", 4)):
        for s in range(200):
            e = W.make("lines", lv, sp, s)
            assert (e.facts["moves"] in W.TRAIN_MOVES) == (sp == "train")
            t = " ".join(e.tokens).split(" . ")
            n_moves = sum(any(k in x for k in ("swap places", "front of the line", "very front", "back of the line", "very back"))
                          for x in t)
            assert n_moves == e.facts["moves"], e.text()


@check
def walks_replay_to_the_answer():
    """Replay each walk from its words alone: the place and the facing match the world's answers."""
    turn = {"left": 3, "right": 1, "around": 2}
    for lv, sp in [(lv, "train") for lv in W.TRAIN_LEVELS["walks"]] + [(7, "longer")]:
        for s in range(100):
            e = W.make("walks", lv, sp, s)
            words = e.tokens
            side, spots = {}, {}
            for snt in " ".join(words).split(" . "):
                t = snt.split()
                if t[0] == "the" and t[2] in ("lie", "lies"):
                    side[W.DIRS.index(t[-1])] = t[1]
            i = words.index("facing")
            h = W.DIRS.index(words[i + 1]) if words[i + 1] in W.DIRS else [d for d, w in side.items() if w == words[i + 2]][0]
            x = y = 0
            for j, w in enumerate(words):
                if w == "turns":
                    h = (h + turn[words[j + 1]]) % 4
                if w == "walks":
                    k = int(words[j + 1])
                    x, y = x + W.STEP[h][0] * k, y + W.STEP[h][1] * k
            assert [x, y] == e.facts["end"] and W.DIRS[h] == e.facts["facing"], e.text()
            for q in e.questions:
                if "face" in q.tokens:
                    assert q.answer == [side[h]], e.text()


@check
def minds_follow_who_saw_what():
    """Someone who never left the room always knows where things are; false beliefs do happen."""
    false = 0
    for s in range(300):
        e = W.make("minds", 3, "train", s)
        left = {t.split()[0] for t in " ".join(e.tokens).split(" . ") if "leaves" in t or "goes out" in t or "walks out" in t}
        for p, d in e.facts["thinks"].items():
            if p not in left:
                assert d == e.facts["where"], e.text()
            false += d != e.facts["where"]
        for q in e.questions:
            false += 0
    assert false > 50, false


@check
def resume_carries_on():
    """A long run stopped half way carries on from its saved state, not from the start."""
    import tempfile
    import torch
    from pathlib import Path
    from .data import Vocab
    from .model import make_pair
    from .train import train
    v = Vocab()
    with tempfile.TemporaryDirectory() as d:
        torch.manual_seed(0)
        laws, _ = make_pair(len(v), "tiny")

        def crash(msg):                       # the pod goes away during step 4
            if "step 4/6" in msg:
                raise KeyboardInterrupt

        try:
            train(laws, v, Path(d), steps=6, batch=4, log_every=1, save_minutes=1e-9, eval_every=0, say=crash)
        except KeyboardInterrupt:
            pass
        assert (Path(d) / "train_state.pt").exists()
        laws, _ = make_pair(len(v), "tiny")
        said = []
        info = train(laws, v, Path(d), steps=6, batch=4, log_every=1, eval_every=0, say=said.append)
        assert any("carrying on from step 3" in m for m in said), said
        assert info["steps"] == 6 and not (Path(d) / "train_state.pt").exists()
        steps = [json.loads(x)["step"] for x in open(Path(d) / "metrics.jsonl")]
        assert steps == sorted(steps) and steps[-1] == 6, steps


@check
def curriculum_moves_up():
    c = W.Curriculum()
    assert c.state()["sums"] == 1
    for _ in range(400):
        c.update("sums", 1, 1.0)
    assert c.state()["sums"] == 2
    picks = [c.pick() for _ in range(2000)]
    assert all(lv in W.TRAIN_LEVELS[f] and lv <= c.state()[f] for f, lv in picks)


@check
def world_one_is_unchanged():
    """World 1 (the first nine families) gives exactly the episodes it gave before World 2, so its runs carry on."""
    import hashlib
    W.use(1)
    try:
        assert len(W.FAMILIES) == 9 and W.TRAIN_MOVES == (1, 2)
        for nat, want in ((False, "5e67a138e3d78dac37175e187939572d"), (True, "5d4c87dfb7bf3014115d5386b488343e")):
            W.STYLE["natural"] = nat
            h = hashlib.md5("|".join(W.make(f, lv, sp, k).text() for f, lv, sp in W.exam_sets() for k in range(20)).encode())
            assert h.hexdigest() == want, ("World 1 changed", nat)
    finally:
        W.STYLE["natural"] = False
        W.use(W.LATEST)
    assert len(W.FAMILIES) == 15


def _sentences(e):
    out, cur = [], []
    for w in e.tokens:
        if w == ".":
            out.append(cur)
            cur = []
        else:
            cur.append(w)
    return out


@check
def time_replays_to_the_answer():
    """Work the hours out from the words alone: what is planned from a moved event moves with it."""
    for lv, sp in [(lv, "train") for lv in W.TRAIN_LEVELS["time"]] + [(lv, "longer") for lv in W.SECRET["time"]["longer"]]:
        for s in range(60):
            e = W.make("time", lv, sp, s)
            ref, hour = {}, {}
            moved = None
            for t in _sentences(e):
                if t[2] in ("is", "starts") and t[3] == "at":
                    hour[t[1]] = int(t[-1])
                elif "moves" in t:
                    moved, k = t[1], int(t[3]) * (1 if t[5] == "later" else -1)
                else:
                    ref[t[1]] = (t[-1], int(t[3]) * (1 if t[5] == "after" else -1))
            while len(hour) < len(ref) + sum(1 for x in hour if x not in ref):
                for x, (p, o) in ref.items():
                    if p in hour and x not in hour:
                        hour[x] = hour[p] + o
                if all(x in hour for x in ref):
                    break
            tied = {moved}
            for _ in ref:
                tied |= {x for x, (p, _o) in ref.items() if p in tied}
            now = {x: h + (k if x in tied else 0) for x, h in hour.items()}
            assert now == e.facts["hour"], e.text()
            assert e.facts["depth"] == lv
            for q in e.questions:
                if q.tokens[:4] == ["at", "what", "hour", "is"]:
                    assert q.answer == [str(now[q.tokens[5]])], e.text()


@check
def cause_replays_to_the_answer():
    """Work out every machine from the rules as written, before and after the flips."""
    for lv, sp in [(lv, "train") for lv in W.TRAIN_LEVELS["cause"]] + [(7, "deeper")]:
        for s in range(60):
            e = W.make("cause", lv, sp, s)
            state, rules, flips = {}, [], []
            for t in _sentences(e):
                if t[-1] in ("on", "off") and t[2] == "switch" and t[3] == "is":
                    state[t[1] + " switch"] = t[-1] == "on"
                elif "flips" in t:
                    flips.append(t[-2] + " switch")
                elif "exactly" in t:
                    body = " ".join(t[t.index("when") + 1:])
                    rules.append((t[1], body))

            def name(x):
                x = x.strip()
                return x[4:] if x.startswith("the ") else x

            def run(st):
                v = dict(st)
                for _ in rules:
                    for d, body in rules:
                        try:
                            if body.endswith("are both on"):
                                a, b = body[:-len(" are both on")].split(" and ")
                                v[d] = v[name(a)] and v[name(b)]
                            elif " or " in body:
                                a, b = body[:-len(" is on")].split(" or ")
                                v[d] = v[name(a)] or v[name(b)]
                            else:
                                a, val = body.rsplit(" is ", 1)
                                v[d] = v[name(a)] == (val == "on")
                        except KeyError:
                            pass
                return v
            before = run(state)
            for f in flips:
                state[f] = not state[f]
            now = run(state)
            for d, on in e.facts["on"].items():
                assert now[d] == on, e.text()
            for q in e.questions:
                if q.tokens[0] == "did":
                    assert q.answer == ["yes" if now[q.tokens[2]] != before[q.tokens[2]] else "no"], e.text()
            assert e.facts["depth"] == lv


@check
def count_replays_to_the_answer():
    for lv, sp in [(lv, "train") for lv in W.TRAIN_LEVELS["count"]] + [(9, "more-moves")]:
        for s in range(80):
            e = W.make("count", lv, sp, s)
            unp = {v: k for k, v in W.PLURAL.items()}
            n_ = lambda w: 0 if w == "no" else int(w)  # noqa: E731
            kind = lambda w: unp.get(w, w)  # noqa: E731
            have = {}
            for t in _sentences(e):
                t = [w for w in t if w not in ("then", "after", "that")]
                if t[1] == "has":
                    have[t[0]] = {}
                    rest = [w for w in t[2:] if w not in (",", "and")]
                    for i in range(0, len(rest), 2):
                        have[t[0]][kind(rest[i + 1])] = n_(rest[i])
                elif t[1] == "gives":
                    k, n = kind(t[3]), n_(t[2])
                    have[t[0]][k] -= n
                    have[t[-1]][k] += n
                elif t[1] in ("finds", "loses"):
                    have[t[0]][kind(t[3])] += n_(t[2]) * (1 if t[1] == "finds" else -1)
            assert have == e.facts["have"], e.text()
            assert all(0 <= v <= 9 for d in have.values() for v in d.values())


@check
def wants_follow_the_doors():
    """The way through the doors, worked out from the words: first room, number of doors, rooms on the way."""
    for lv, sp in [(lv, "train") for lv in W.TRAIN_LEVELS["wants"]] + [(lv, "farther") for lv in W.SECRET["wants"]["farther"]]:
        for s in range(60):
            e = W.make("wants", lv, sp, s)
            nb, where = {}, {}
            for t in _sentences(e):
                rooms = [w for w in t if w in W.ROOMS]
                if "door" in t:
                    a, b = rooms
                    nb.setdefault(a, set()).add(b)
                    nb.setdefault(b, set()).add(a)
                elif t[1:4] == ["is", "in", "the"] or t[2:5] == ["is", "in", "the"]:
                    where[t[0] if t[1] == "is" else t[1]] = rooms[0]
                elif "carries" in t or "walks" in t:
                    where[t[-1] if "walks" in t else t[3]] = rooms[0]
                    if "walks" in t:
                        where[t[0]] = rooms[0]
            path = e.facts["path"]
            assert W._path(nb, path[0], path[-1]) == path and len(path) - 1 == lv, e.text()
            for q in e.questions:
                if q.tokens[:4] == ["how", "many", "doors", "must"]:
                    assert q.answer == [str(lv)]
                if "first" in q.tokens:
                    assert q.answer == [path[1]], e.text()


@check
def carried_things_go_along():
    """Replay the carrying and the putting from the words alone: each thing's room is where its outermost container is."""
    for lv, sp in [(lv, "train") for lv in W.TRAIN_LEVELS["carry"]] + [(6, "deeper"), (4, "more-moves")]:
        for s in range(80):
            e = W.make("carry", lv, sp, s)
            par = {}
            for t in _sentences(e):
                t = [w for w in t if w not in ("then", "after", "that")]
                if t[0] == "inside":
                    par[t[-1]] = t[2]
                elif t[2] == "holds":
                    par[t[-1]] = t[1]
                elif t[0] == "the" and t[2] in ("is", "stands", "lies"):
                    par[t[1]] = t[-1]
                elif "carries" in t:
                    par[t[t.index("the") + 1]] = t[-1]
                elif "puts" in t:
                    par[t[3]] = t[-1]
            room = lambda x: x if x in W.ROOMS else room(par[x])  # noqa: E731
            assert {x: room(x) for x in e.facts["room"]} == e.facts["room"], e.text()


@check
def house_mixes_are_walled():
    """The mixed family: training never asks a three-family question, the secret exam asks only those; someone who never
    left the house always believes the truth; some beliefs are false."""
    false = 0
    for sp, lvs in (("train", W.TRAIN_LEVELS["house"]), ("three-way", W.SECRET["house"]["three-way"])):
        for lv in lvs:
            for s in range(150):
                e = W.make("house", lv, sp, s)
                ks = [q.tags["mix"] for q in e.questions]
                assert all((k in W.HOUSE_SECRET_KINDS) == (sp == "three-way") for k in ks), (ks, e.text())
                left = {next(w for w in t if w in W.NAMES) for t in _sentences(e) if "goes" in t or "leaves" in t}
                for q in e.questions:
                    if q.tags.get("belief") is not None:
                        false += q.tags["belief"]
                        if next(w for w in q.tokens if w in W.NAMES) not in left:
                            assert not q.tags["belief"], e.text()
    assert false > 100, false


def torch_checks():
    import torch
    from .data import Vocab, laws_batch, rival_batch, UNK
    from .model import make_pair, n_params

    v = Vocab()

    @check
    def no_unknown_words():
        for e in list(train_eps(30)) + list(secret_eps(30)):
            assert UNK not in v.enc(e.text().split()), e.text()

    rng = random.Random(0)
    eps = [W.make(f, lv, sp, rng.randrange(10 ** 6)) for f, lv, sp in rng.choices(W.exam_sets(), k=24)]

    @check
    def same_size_rival():
        for preset in ("tiny", "small", "base", "big"):
            laws, rival = make_pair(len(v), preset)
            a, b = n_params(laws), n_params(rival)
            assert abs(a - b) / a < 0.01, (preset, a, b)

    @check
    def looped_rival_same_size_more_depth():
        _, rival = make_pair(len(v), "tiny")
        rb = rival_batch(eps[:4], v)
        torch.manual_seed(0)
        before = n_params(rival)
        rival.eval()
        with torch.no_grad():
            l1 = rival.loss_and_hits(rb)[0].item()
            rival.cfg.loops = 3                      # its last 2 layers run 3 times, same weights
            l3 = rival.loss_and_hits(rb)[0].item()
        assert n_params(rival) == before, "looping adds no numbers"
        assert abs(l1 - l3) > 1e-6, "the loops change what it computes"

    @check
    def batches_line_up():
        lb, rb = laws_batch(eps, v), rival_batch(eps, v)
        assert len(lb["meta"]) == len(rb["meta"])
        assert [m["q"] for m in lb["meta"]] == [m["q"] for m in rb["meta"]]
        s = lb["pool"].sum(-1)
        assert torch.allclose(s[lb["kind"] > 0], torch.ones_like(s[lb["kind"] > 0])), "every slot pools to 1"
        # the Laws reader never sees an answer
        for b, e in enumerate(eps):
            n = (lb["tok"][b] != 0).sum().item()
            assert n == len(e.text(answers=False).split())
        # the rival is asked right after "answer :" (or the digit before)
        ids = rb["tok"][rb["ib"], rb["ipos"] + 1]
        assert torch.equal(ids, rb["iy"])

    @check
    def both_models_learn_a_little():
        torch.manual_seed(0)
        laws, rival = make_pair(len(v), "tiny")
        for m in (laws, rival):
            m.train()
            opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
            eps1 = [W.make("machines", 1, "train", s) for s in range(32)]
            from .train import batch_for
            b = batch_for(m, eps1, v, "cpu")
            first = None
            for i in range(40):
                kw = {"passes": 3} if m.kind == "laws" else {}
                loss, hits, _ = m.loss_and_hits(b, **kw)
                opt.zero_grad()
                loss.backward()
                opt.step()
                first = first if first is not None else loss.item()
            assert loss.item() < 0.7 * first, (m.kind, first, loss.item())

    @check
    def laws_thinks_until_settled():
        laws, _ = make_pair(len(v), "tiny")
        laws.eval()
        b = laws_batch(eps, v)
        with torch.no_grad():
            loss, hits, used = laws.loss_and_hits(b, adaptive=True)
        assert used.min() >= 1 and used.max() <= laws.cfg.test_max
        assert hits.shape[0] == len(b["meta"])

    @check
    def no_marks_world_and_finder():
        """Natural speech: pronouns point to the right person and are unambiguous; the finder model learns a little and
        never sees the marks."""
        from .data import finder_batch
        W.STYLE["natural"] = True
        try:
            vv = Vocab()
            n_pron = 0
            for s in range(300):
                e = W.make("lines", 4, "train", s)
                sents, cur = [], []
                for t, m in zip(e.tokens, e.mentions):
                    cur.append((t, m))
                    if t == ".":
                        sents.append(cur)
                        cur = []
                for i, snt in enumerate(sents):
                    for t, m in snt:
                        if t in W.PRONOUNS:
                            n_pron += 1
                            prev = {mm for _, mm in sents[i - 1] if mm >= 0}
                            assert m in prev, e.text()
                            g = "f" if t in ("she", "her") else "m"
                            same = [x for x in prev if (e.things[x] in W._FEM) == (g == "f")]
                            assert same == [m], ("ambiguous pronoun", e.text())
            assert n_pron > 100, n_pron
            e = W.make("sums", 3, "train", 1)
            assert int("".join(reversed(e.questions[0].answer))) == int("".join(map(str, e.facts["a"]))) + int("".join(map(str, e.facts["b"])))
            torch.manual_seed(0)
            laws, rival = make_pair(len(vv), "tiny", finder=12, fixed=3, train_max=3, grad_passes=3, test_max=3)
            assert abs(n_params(laws) - n_params(rival)) / n_params(laws) < 0.01
            eps1 = [W.make(f, 1, "train", s) for s in range(12) for f in ("sums", "cups", "minds")]
            b = finder_batch(eps1, vv)
            assert "pool" not in b and "pos" not in b, "the finder must not get the marks"
            opt = torch.optim.AdamW(laws.parameters(), lr=3e-3)
            first = None
            for i in range(40):
                loss, hits, _ = laws.loss_and_hits(b)
                opt.zero_grad()
                loss.backward()
                opt.step()
                first = first if first is not None else loss.item()
            assert loss.item() < 0.8 * first, (first, loss.item())
            got, n = laws.found(b)
            assert 0 <= got <= n and n > 0
        finally:
            W.STYLE["natural"] = False

    @check
    def marks_as_teacher_groups_words():
        """With marks as a teacher, the finder learns to put a thing's words together (its found score rises), while the
        batch it reads still has no marks in it."""
        from .data import finder_batch
        W.STYLE["natural"] = True
        try:
            vv = Vocab()
            eps1 = [W.make(f, 1, "train", s) for s in range(10) for f in ("lines", "cups", "minds")]
            b = finder_batch(eps1, vv)
            assert "pool" not in b
            scores = {}
            for teach in (0.0, 1.0):
                torch.manual_seed(0)
                laws, _ = make_pair(len(vv), "tiny", finder=12, fixed=3, train_max=3, grad_passes=3, test_max=3, teach=teach)
                laws.train()
                laws.loss_and_hits(b)
                t0 = laws.teach_loss(b["ment"]).item()
                opt = torch.optim.AdamW(laws.parameters(), lr=3e-3)
                for _ in range(60):
                    loss, _, _ = laws.loss_and_hits(b)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                laws.loss_and_hits(b)
                t1 = laws.teach_loss(b["ment"]).item()
                laws.eval()
                got, n = laws.found(b)
                scores[teach] = (t0, t1, got / n)
            assert scores[1.0][1] < 0.8 * scores[1.0][0], scores
            assert scores[1.0][2] > scores[0.0][2], scores
            laws._att = torch.zeros_like(laws._att)
            laws._att[:, 0, :] = 1                                  # every word in one slot: stays finite under bf16
            with torch.autocast("cpu", dtype=torch.bfloat16):
                assert math.isfinite(laws.teach_loss(b["ment"]).item())
        finally:
            W.STYLE["natural"] = False

    @check
    def linker_finds_the_things():
        """The linker: it learns from the marks which words name a thing and which earlier word names the same one, and
        then groups the things far better than chance, from plain text; its batch has no marks in it."""
        from .data import finder_batch
        W.STYLE["natural"] = True
        try:
            vv = Vocab()
            eps1 = [W.make(f, 1, "train", s) for s in range(6) for f in ("lines", "family", "minds", "walks")]
            b = finder_batch(eps1, vv)
            assert "pool" not in b
            torch.manual_seed(0)
            laws, rival = make_pair(len(vv), "tiny", finder=24, finder_kind="link", fixed=3, train_max=3, grad_passes=3,
                                    test_max=3, teach=1.0)
            assert abs(n_params(laws) - n_params(rival)) / n_params(laws) < 0.01
            laws.eval()
            g0, n0 = laws.found(b)
            laws.train()
            opt = torch.optim.AdamW(laws.parameters(), lr=3e-3)
            for _ in range(80):
                loss, _, _ = laws.loss_and_hits(b)
                opt.zero_grad()
                loss.backward()
                opt.step()
            laws.eval()
            g1, n1 = laws.found(b)
            assert g1 / n1 > max(0.5, g0 / n0 + 0.3), (g0 / n0, g1 / n1)
            a = laws.assignment()
            assert a.max() < 24 and a.min() >= -1
            with torch.no_grad():
                _, hits, _ = laws.loss_and_hits(b)
            assert hits.shape[0] == len(b["meta"])
            # fairness: the rival gets the same answer key, through a link head of the same shape, and still matches in size
            from .data import rival_batch
            assert rival.cfg.teach == laws.cfg.teach and hasattr(rival, "link")
            from .model import _rival_count
            assert _rival_count(rival.cfg) == n_params(rival)
            rb = rival_batch(eps1, vv, tags=False)
            assert (rb["ment"] >= 0).sum() > 0 and (rb["tag"] == 0).all()
            rival.train()
            ropt = torch.optim.AdamW(rival.parameters(), lr=3e-3)
            def rl():
                rival(rb)
                from .model import link_loss
                return link_loss(*rival.link(rival._h.float()), rb["tok"] != 0, rb["ment"])
            r0 = rl().item()
            for _ in range(40):
                loss, _, _ = rival.loss_and_hits(rb)
                ropt.zero_grad()
                loss.backward()
                ropt.step()
            assert rl().item() < 0.9 * r0, (r0, rl().item())
        finally:
            W.STYLE["natural"] = False

    @check
    def tiny_run_end_to_end():
        import tempfile
        from .run import main
        with tempfile.TemporaryDirectory() as d:
            main(["--preset", "tiny", "--steps", "6", "--batch", "8", "--eval-n", "2", "--runs", d, "--device", "cpu"])
            import json
            from pathlib import Path
            r = json.loads((Path(d) / "worldfirst" / "results.json").read_text())
            assert r["kind"] == "worldfirst" and r["rows"]["laws"] and r["rows"]["rival"]
            assert "secret" in r["summary"]["laws"]
            assert (Path(d) / "worldfirst" / "report.md").read_text().startswith("# World-first")
            from oazy.dashboard import world_results
            assert world_results(Path(d))[0]["name"] == "worldfirst"


def main():
    try:
        import torch  # noqa: F401
        torch_checks()
    except ImportError:
        print("  (no torch here: skipped the model checks)")
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
