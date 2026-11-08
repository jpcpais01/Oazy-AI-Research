"""Self-tests for LogRes (oazy/logres.py): the question file is exactly what the seeded generator makes, every right
answer is checked by a separate little solver that works it out from the facts alone, the answer is never written
in the facts, shallow tricks score about the guessing level, and the scorer gives the same numbers however the
questions are batched (on tiny Bee4, Pegasus 1 Mini and Plain-100M models).

    python3 -m oazy.logres_tests                 (CPU, under a minute; without internet it uses stand-in tokenizers)
    python3 -m oazy.logres_tests --device cuda
"""
from __future__ import annotations

import argparse
import re
import tempfile
from collections import defaultdict
from pathlib import Path

from oazy import logres, term


def check(name, ok, detail=""):
    term.check(name, ok, detail)


# --------------------------------------------------------------------- a separate solver: facts in, answer out
def _closure(pairs):
    """pairs (a, b) meaning a ranks above b -> every (a, b) that follows."""
    above = defaultdict(set)
    for a, b in pairs:
        above[a].add(b)
    changed = True
    while changed:
        changed = False
        for a in list(above):
            new = set().union(*(above[b] for b in above[a] if b in above)) - above[a]
            if new:
                above[a] |= new
                changed = True
    return above


def _facts(it):
    return it["ctx"].split("\nQuestion:")[0], it["ctx"].split("\nQuestion: ")[1].split("\nAnswer:")[0]


def solve(it):
    """The right answer's text, worked out from the facts only (never from the stored label)."""
    facts, q = _facts(it)
    sents = [s.strip() for s in re.split(r"(?<=\.) ", facts) if s.strip()]
    fam = it["family"]
    if fam in ("order", "time", "place"):
        pairs = []
        for s in sents:
            if fam == "order":
                a, word, b = re.fullmatch(r"(\w+) is (\w+) than (\w+)\.", s).groups()
                ups = [u for u, d in logres.COMPARE if word in (u, d)][0]
                pairs.append((a, b) if word == ups else (b, a))
            elif fam == "time":
                a, word, b = re.fullmatch(r"(\w+) .+ (before|after) (\w+)\.", s).groups()
                pairs.append((a, b) if word == "before" else (b, a))
            else:
                a, word, b = re.fullmatch(r"The (\w+) is to the (left|right) of the (\w+)\.", s).groups()
                pairs.append(("the " + a, "the " + b) if word == "left" else ("the " + b, "the " + a))
        above = _closure(pairs)
        x, y = it["endings"]
        top = x if y in above[x] else y if x in above[y] else None
        assert top is not None, "the two are not linked"
        want_top = (fam == "order" and any(f"is {u}," in q for u, _ in logres.COMPARE)) or \
                   (fam == "time" and " earlier," in q) or (fam == "place" and "further left" in q)
        return top if want_top else (y if top == x else x)
    if fam == "sums":
        n = [int(v) for v in re.findall(r"\d+", facts)]
        L = it["level"]
        do = {1: lambda: n[0] + n[1], 2: lambda: n[0] + n[1] - n[2], 3: lambda: n[0] * n[1],
              4: lambda: n[0] - n[1] * n[2], 5: lambda: n[0] * n[2] + n[1] * n[3]}
        return str(do[L]())
    if fam == "rules":
        rules = [re.fullmatch(r"If (.+), (.+)\.", s).groups() for s in sents if s.startswith("If ")]
        known = {s[:-1].lower() for s in sents if not s.startswith("If ")}
        for _ in rules:
            known |= {b for a, b in rules if a in known}
        got = [e for e in it["endings"] if e in known]
        assert len(got) == 1
        return got[0]
    if fam == "sets":
        alls = [re.fullmatch(r"All (\w+)s are (\w+)s\.", s).groups() for s in sents if s.startswith("All ")]
        member = [re.fullmatch(r"\w+ is a (\w+)\.", s).group(1) for s in sents if " is a " in s][0]
        known = {member}
        for _ in alls:
            known |= {b for a, b in alls if a in known}
        got = [e for e in it["endings"] if e in known]
        assert len(got) == 1
        return got[0]
    if fam == "negation":
        out = [e for e in it["endings"] if f"not at {e}." not in facts and f"not in {e}." not in facts]
        assert len(out) == 1
        return out[0]
    if fam == "family":
        parent = {}                                   # child -> parent
        for s in sents:
            m = re.fullmatch(r"(\w+) is (\w+)'s (parent|child)\.", s)
            a, b, rel = m.groups()
            if rel == "parent":
                parent[b] = a
            else:
                parent[a] = b
        x, rel = re.fullmatch(r"Who is (\w+)'s ([\w-]+), .*", q).groups()
        L = 1 if rel in ("parent", "child") else 2 + rel.count("great-")
        up = rel.endswith("parent")
        def climb(p, k):
            for _ in range(k):
                p = parent.get(p)
            return p
        if up:
            return climb(x, L)
        got = [e for e in it["endings"] if climb(e, L) == x]
        assert len(got) == 1
        return got[0]
    raise ValueError(fam)


def heuristics(rows):
    """Shallow tricks' score minus the guessing level, per kind (in points)."""
    H = defaultdict(lambda: defaultdict(list))
    for r in rows:
        facts, _ = _facts(r)
        E = r["endings"]
        first = lambda e: facts.find(e) if e in facts else 10 ** 9      # noqa: E731
        picks = {"said last": max(range(len(E)), key=lambda i: facts.rfind(E[i])),
                 "said first": min(range(len(E)), key=lambda i: first(E[i])),
                 "said most": max(range(len(E)), key=lambda i: facts.count(E[i])),
                 "listed first": 0, "listed last": len(E) - 1,
                 "shortest": min(range(len(E)), key=lambda i: len(E[i])),
                 "longest": max(range(len(E)), key=lambda i: len(E[i]))}
        for k, p in picks.items():
            H[r["family"]][k].append(int(p == r["label"]) - 1 / len(E))
    return {f: {k: 100 * sum(v) / len(v) for k, v in d.items()} for f, d in H.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    import torch
    dev = torch.device(ap.parse_args().device)
    term.tests_start("LogRes", dev)

    # ---- the questions
    rows = logres.load()
    built = logres.build()
    check("the question file is exactly what the seeded generator makes (nobody edited it by hand)", rows == built,
          f"{len(rows)} questions in {logres.FILE.name}")
    cells = defaultdict(int)
    for r in rows:
        cells[(r["family"], r["level"])] += 1
    check("8 kinds x 5 levels x 40 questions, all different",
          len(cells) == 40 and set(cells.values()) == {40} and len({r["ctx"] for r in rows}) == len(rows))
    bad = []
    for r in rows:
        try:
            if solve(r) != r["endings"][r["label"]]:
                bad.append(r["id"])
        except Exception as e:                       # noqa: BLE001
            bad.append(f"{r['id']} ({type(e).__name__})")
    check("every right answer is confirmed by a separate solver working from the facts alone", not bad,
          f"wrong: {bad[:5]}" if bad else "1,600 of 1,600")
    named = [r["id"] for r in rows if not all(e in _facts(r)[1] for e in r["endings"])]
    check("every answer is named in the question (the model only has to choose)", not named, str(named[:5]))
    stated = []
    for r in rows:
        facts, _ = _facts(r)
        right, others = r["endings"][r["label"]], [e for i, e in enumerate(r["endings"]) if i != r["label"]]
        if r["family"] == "sums" and right in re.findall(r"\d+", facts):
            stated.append(r["id"])
        if r["family"] in ("order", "time", "place", "family") and r["level"] >= 2:
            sents = re.split(r"(?<=\.) ", facts)
            pair = [re.fullmatch(r"Who is (\w+)'s .*", _facts(r)[1]).group(1)] if r["family"] == "family" else others
            has = lambda w, s: re.search(rf"\b{re.escape(w)}\b", s) is not None      # noqa: E731
            if any(has(right, s) and has(o, s) for s in sents for o in pair):
                stated.append(r["id"])
    check("the answer is never written in the facts (no fact links the two, no sum's answer is in the text)", not stated,
          str(stated[:5]))
    slots = defaultdict(lambda: defaultdict(int))
    for r in rows:
        slots[(r["family"], r["level"])][r["label"]] += 1
    lop = [k for k, v in slots.items() if max(v.values()) - min(v.values()) > 1 and len(v) == len(rows[0]["endings"])]
    two = [k for k, v in slots.items() if k[0] not in ("sums", "negation") and v[0] != v[1]]
    check("the right answer sits in each answer slot equally often", not lop and not two, str((lop + two)[:4]))
    H = heuristics(rows)
    worst = max(((v, f, k) for f, d in H.items() for k, v in d.items() if not (f == "negation" and k == "said most")))
    check("shallow tricks (said last / first / most, listed first / last, shortest, longest) stay near guessing",
          worst[0] < 10, f"best trick: {worst[2]} on {worst[1]}, {worst[0]:+.1f} points over guessing")
    neg = H["negation"]["said last"]
    check("negation punishes copying: the place said last is always a ruled-out one", neg < -20, f"{neg:+.0f} points")
    ch = logres.summarize([{"family": r["family"], "level": r["level"], "chance": 1 / len(r["endings"]), "char": 0, "tok": 0}
                           for r in rows])
    check("guessing level reported per level", all(0.15 < v["chance"] < 0.55 for v in ch["levels"].values()),
          ", ".join(f"L{k} {v['chance'] * 100:.0f}%" for k, v in ch["levels"].items()))

    # ---- scoring
    tmp_ctx = tempfile.TemporaryDirectory()
    from bee2.tests import stand_ins
    used = stand_ins(Path(tmp_ctx.name))
    if used:
        term.info(f"no internet: stand-in {' and '.join(used)} tokenizer(s)")
    from bee4.model import Bee4, Bee4Config
    from bee3.tests import SMALL as B_SMALL
    from pegasus.model import Pegasus, PegasusConfig
    from pegasus.tests import SMALL as P_SMALL
    from pegasus.baseline import Plain100, Plain100Config
    from oazy import hellaswag
    torch.manual_seed(0)
    sub = [r for r in rows if r["id"].endswith(("-01", "-02"))]          # 2 of every kind and level: 80
    models = {"Bee4": Bee4(Bee4Config(**B_SMALL)), "Pegasus 1 Mini": Pegasus(PegasusConfig(size="mini", **{**P_SMALL, "max_len": 512, "ctx": "512:0"})),
              "Plain-100M": Plain100(Plain100Config(**B_SMALL))}
    for name, m in models.items():
        m = m.to(dev).eval()
        a = logres.evaluate(m, dev, sub, batch=64)
        b = logres.evaluate(m, dev, sub, batch=1)
        same = abs(a["acc_char"] - b["acc_char"]) <= 1 / a["n"] + 1e-9 and abs(a["acc_norm"] - b["acc_norm"]) <= 1 / a["n"] + 1e-9
        if dev.type == "cpu":                # exactly the same on the CPU; on a GPU bf16 rounding may flip one close call
            same = same and a == b
        check(f"{name}: the same score whichever questions share a batch", same,
              f"{a['acc_char'] * 100:.1f}% per letter on {a['n']} (guessing {a['chance'] * 100:.0f}%; untrained)")
        if name == "Bee4":
            p = Path(tmp_ctx.name) / "sub.jsonl"
            import json
            p.write_text("".join(json.dumps(r) + "\n" for r in sub))
            h = hellaswag.evaluate(m, p, dev)
            check("Bee4: the same as the HellaSwag scorer on the same questions", abs(h["acc_char"] - a["acc_char"]) < 1e-9
                  and abs(h["acc_norm"] - a["acc_norm"]) < 1e-9, f"{h['acc_char'] * 100:.1f}% both ways")
        check(f"{name}: a score for every level and kind", set(a["levels"]) == {"1", "2", "3", "4", "5"}
              and all(a["families"][f]["n"] == 10 for f in logres.FAMILIES) and all(len(v) == 5 for v in a["grid"].values()))
    term.tests_passed("LogRes")


if __name__ == "__main__":
    main()
