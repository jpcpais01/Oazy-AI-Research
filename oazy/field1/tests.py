"""Field1's self-tests (CPU, about a minute): python3 -m oazy.field1.tests

  1. every family makes its episodes, the same ones from the same seed
  2. the walls: none of Real World's wording, Real World Pro's names and things, or LogT's questions, in thousands of
     episodes; and the walls themselves catch what they should
  3. meaning: every way of saying a link (who is in front of whom, left of whom, what is in what, who gives to whom,
     which way a turn goes) is read back by a small reader written separately from the templates, and must say the
     same thing as the world
  4. answers are not lopsided (yes and no both common), questions are not repeated in an episode
  5. the level table (levels.py): every scene notes its level, every question its kind, and its choices hold the answer
  6. Opal: the mix gives Field1 its share; a tiny build writes pieces a reader can read; sado is renamed opal with a link
"""
from __future__ import annotations

import json
import random
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

OK, BAD = [], []


def check(name, cond, info=""):
    (OK if cond else BAD).append(name)
    print(("  ok    " if cond else "  FAIL  ") + name + (f"  ({info})" if info else ""), flush=True)


def t_families():
    from .. import field1 as F
    errs = {}
    for f in F.FAMILIES:
        for i in range(300):
            try:
                F.episode(f, ("t", i))
            except Exception as e:                     # noqa: BLE001
                errs.setdefault(f, repr(e)[:120])
    check("every family makes 300 episodes without an error", not errs, str(errs)[:300])
    same = all(F.episode(f, ("s", 3))[0] == F.episode(f, ("s", 3))[0] for f in F.FAMILIES)
    check("the same seed gives the same episode", same)
    differ = all(F.episode(f, ("s", 3))[0] != F.episode(f, ("s", 4))[0] for f in F.FAMILIES)
    check("other seeds give other episodes", differ)
    texts = [F.episode(f, ("u", i))[0] for f in F.FAMILIES for i in range(200)]
    check("no repeats in 3,800 episodes", len(set(texts)) == len(texts), f"{len(texts) - len(set(texts))} repeats")
    leftovers = [t for t in texts if re.search(r"[<>{}\[\]]|\|", t)]
    check("no template marks left in the text", not leftovers, leftovers[0][:200] if leftovers else "")
    empty_q = [t for t in texts if "??" in t or " ." in t.replace(" ...", "")]
    check("no doubled marks", not empty_q, empty_q[0][:200] if empty_q else "")


def t_walls():
    from .. import field1 as F
    from . import guard, core
    found = {}
    for f in F.FAMILIES:
        for i in range(250):
            for x in guard.leaks(F.episode(f, ("w", i))[0]):
                found[x] = found.get(x, 0) + 1
    check("walls: 4,750 episodes say nothing held back", not found, str(sorted(found.items())[:5]))
    remade = {}
    for f in F.FAMILIES:
        n = 0
        for i in range(200):
            r = random.Random(f"field1|{F.VERSION}|{f}|('w', {i})|0")
            st, qas = F.GEN[f](r)
            n += bool(guard.leaks(core.frame(r, st.tell(), qas)))
        remade[f] = n / 200
    worst = max(remade.items(), key=lambda kv: kv[1])
    check("walls are rarely needed (every family under 5% remade)", worst[1] < 0.05, f"worst {worst[0]} {worst[1]:.1%}")
    check("the wall catches Real World wording", bool(guard.leaks("Ana is directly ahead of Bo.")))
    check("the wall catches 'next' (Real World's 'then')", bool(guard.leaks("Next, Ana goes out.")))
    check("the wall catches Pro names and things", bool(guard.leaks("Abigail puts the acorn in the mug.")))
    q = json.loads((Path(__file__).resolve().parent.parent / "logres_v1.jsonl").read_text().splitlines()[0])["ctx"]
    check("the wall catches a LogT question", bool(guard.leaks(q)))
    check("the wall lets ordinary Field1 words through", not guard.leaks("Ana is right behind Bo. Then Bo goes to the front."))
    from .. import realworld as RW
    import worldfirst.world as W
    # Real World's own check reads the world's practice text; the new names and words of Pro are not in Field1's lists
    from .core import FEMALE, MALE, COLOURS, SMALL, CONTAINERS, ROOMS
    pro = {w.lower() for w in RW.PRO_FEMALE + RW.PRO_MALE + RW.VEHICLES + [x for v in RW.PRO_WORDS.values() for x in v]}
    mine = {w.lower() for w in FEMALE + MALE + COLOURS + SMALL + CONTAINERS + ROOMS}
    check("Field1's names and things share nothing with Pro's", not (pro & mine), str(sorted(pro & mine)))
    check("the world's practice text still passes Real World's own check", not RW.check_held_back(per_level=4))
    del W


# ---------------------------------------------------------------- meaning: small readers, written apart from the templates
AHEAD_W = ("in front of", "ahead of", "ahead", "before", "first")
BEHIND_W = ("behind", "after", "follows", "followed")


def read_line(s, a, b):
    """Who is right in front of whom, read from one sentence about a and b -> (front, back)."""
    low = s.lower()
    ia, ib = low.find(a.lower()), low.find(b.lower())
    first, second = (a, b) if ia < ib else (b, a)
    if re.search(r"\b(has|had)\b", low):               # "Y has X right in front of him": X is the one placed
        target, ref = second, first
    elif re.match(r"\s*(right|just|the one|the person)\b", low):   # "right behind X stands Y"
        target, ref = second, first
    else:
        target, ref = first, second
    ahead = any(w in low for w in AHEAD_W) and not any(w in low for w in BEHIND_W)
    return (target, ref) if ahead else (ref, target)


def read_row(s, a, b):
    """(left one, right one), read from one sentence."""
    low = s.lower()
    ia, ib = low.find(a.lower()), low.find(b.lower())
    first, second = (a, b) if ia < ib else (b, a)
    if re.match(r"\s*(just|right|the one)\b", low):
        target, ref = second, first
    else:
        target, ref = first, second
    if "before" in low:                                # "X comes just before Y, going from left to right"
        return target, ref
    if "on" in low.split() and "'s right" in low:     # "Y is on X's right"
        return ref, target
    left = "left" in low and "right" not in low.replace("right beside", "").replace("right to the left", "left")
    if "on " in low and "left" in low and "right beside" in low and "'s left" in low:   # "X is right beside Y, on Y's left"
        return target, ref
    return (target, ref) if left else (ref, target)


def read_in(s, item, box):
    """(inner, outer), read from one sentence."""
    low = s.lower()
    ii, ib = low.find(item), low.find(box)
    first, second = (item, box) if ii < ib else (box, item)
    if re.search(r"\b(has|had)\b(?! (put|been))|\b(holds|held)\b", low) or re.match(r"\s*(inside|in)\b", low):
        return second, first
    return first, second


def read_give(s, a, b):
    """(giver, getter)."""
    low = s.lower()
    ia, ib = low.find(a.lower()), low.find(b.lower())
    first, second = (a, b) if ia < ib else (b, a)
    if " from " in low or "asks for some" in low or "asked for some" in low:
        return second, first
    return first, second


def t_meaning():
    from .core import Teller, P, T
    from . import fam_order as O, fam_space as S
    r = random.Random(5)
    bad = []
    a, b = P("Ana", "f"), P("Bo", "m")
    for tpl in O.ADJ:
        for tense in ("pres", "past"):
            for _ in range(6):
                s = Teller(r, tense).fmt(tpl.rstrip("@"), x=a, y=b)
                if read_line(s, "Ana", "Bo") != ("Ana", "Bo"):
                    bad.append(s)
    check(f"lines: all {len(O.ADJ)} ways of saying 'X is right in front of Y' read back right", not bad, bad[:2])
    bad = []
    for tpl in O.ROW_ADJ:
        for tense in ("pres", "past"):
            for _ in range(6):
                s = Teller(r, tense).fmt(tpl.rstrip("@"), x=a, y=b)
                if read_row(s, "Ana", "Bo") != ("Ana", "Bo"):
                    bad.append(s)
    check(f"rows: all {len(O.ROW_ADJ)} ways of saying 'X is just left of Y' read back right", not bad, bad[:2])
    bad = []
    key, box = T("key"), T("box", "red")
    for tpl in S.IN_T:
        for _ in range(6):
            s = Teller(r).fmt(tpl, x=key, c=box)
            if read_in(s, "key", "box") != ("key", "box"):
                bad.append(s)
    check(f"places: all {len(S.IN_T)} ways of saying 'X is in Y' read back right", not bad, bad[:2])
    from . import fam_people as PP
    import inspect
    src = inspect.getsource(PP.count)
    gives = [g for g in re.findall(r'"(\{[pq]\}[^"]*(?:gives|gave|gets|got|passes|passed|asks)[^"]*)"', src)
             if "{p}" in g and "{q}" in g]
    bad = []
    for tpl in gives:
        for _ in range(4):
            s = Teller(r).fmt(tpl, p=a, q=b, k="3 shells")
            if read_give(s, "Ana", "Bo") != ("Ana", "Bo"):
                bad.append(s)
    check(f"count: all {len(gives)} ways of saying 'X gives to Y' read back right", len(gives) >= 4 and not bad, bad[:2])
    # walks: the turns turn the way they say
    for side in ("left", "right"):
        bad = [t for t in S.TURN_T[side] if side not in t]
        check(f"walks: every '{side}' turn says {side}", not bad)
    # shells: a lift always names the cup it leaves before the cup it goes to
    bad = [t for t in S.LIFT if t.index("{a}") > t.index("{b}")]
    check("shells: every lift names the cup left first", not bad)


def t_answers():
    from .. import field1 as F
    import random as R
    from .core import frame
    yes = no = 0
    per = {}
    for f in F.FAMILIES:
        for i in range(150):
            r = R.Random(f"a|{f}|{i}")
            st, qas = F.GEN[f](r)
            qs = [q.q for q in qas]
            if len(set(qs)) != len(qs):
                per.setdefault("repeat", f)
            for q in qas:
                a = q.a.lower()
                yes += a == "yes"
                no += a == "no"
                if not q.a.strip():
                    per.setdefault("empty", f)
            frame(r, st.tell(), qas)
    share = yes / max(1, yes + no)
    check("yes and no both common among yes/no answers", 0.3 < share < 0.7, f"yes {share:.0%} of {yes + no}")
    check("no repeated question in an episode, no empty answer", not per, str(per))


def t_opal():
    from .. import mix, opal
    m = mix.parse_mix(mix.resolve_mix("opal"))
    shares = [w.get("field1", 0) for _, w in m]
    check("the opal mix gives Field1 20% of every phase", all(abs(s - 0.20) < 1e-6 for s in shares), str(shares))
    m2 = mix.parse_mix(mix.resolve_mix("opal:0.30:0.1,0.15"))
    check("an Opal plan written out is followed", abs(m2[0][1]["field1"] - 0.30) < 1e-6 and abs(m2[-1][1]["field1"] - 0.15) < 1e-6)
    em = mix.parse_mix(mix.resolve_mix("emerald:flat:0.20"))
    check("Emerald's mix is unchanged (old runs resume as they were)", all("world" in w and "field1" not in w for _, w in em))
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "sado").mkdir()
        (td / "sado" / "train.bin").write_bytes(b"\0\0" * 10)
        d = opal.opal_dir(td)
        check("sado is renamed opal, with a link named sado left", d == td / "opal" and (td / "sado").is_symlink()
              and (td / "sado" / "train.bin").exists())
        check("opal_dir again leaves things as they are", opal.opal_dir(td) == td / "opal")
        # a tiny build, with the real SmolLM2 tokenizer if it can be had, else a stand-in
        from ..sado_tests import smol_stand_in
        smol_stand_in(td)
        opal.build(td, tokens=40_000, val_tokens=5_000, workers=2)
        tr = np.memmap(td / "opal" / "field1" / "train.bin", dtype=np.uint16, mode="r")
        meta = json.loads((td / "opal" / "field1" / "meta.json").read_text())
        check("the build writes Field1's pieces and its meta", tr.size >= 40_000 and meta["train_tokens"] == tr.size, f"{tr.size}")
        from .. import tok
        enc = tok.get("smollm2")
        txt = enc.decode(list(map(int, tr[:3000])))
        check("the pieces read back as Field1 text", "<|endoftext|>" in txt and "?" in txt, txt[:120].replace("\n", " "))
        # the trainer's reader takes Opal
        (td / "opal" / "val.bin").write_bytes(np.zeros(5000, dtype=np.uint16).tobytes())
        (td / "opal" / "train.bin").write_bytes(np.random.default_rng(0).integers(1, 500, 200_000, dtype=np.uint16).tobytes())
        mx = mix.Mix(td / "opal", "opal", 64, 4, lengths=(64,), log=lambda *a, **k: None)
        w = mx.weights(0.5)
        row = mx.src["field1"].row(64) if "field1" in mx.src else []
        check("the trainer's mix reads Field1 rows from Opal", "field1" in mx.src and w.get("field1", 0) > 0 and len(row) == 65,
              str({k: round(v, 3) for k, v in w.items()}))


def t_levels():
    from .. import field1 as F
    from . import levels as L
    no_kind, no_notes, not_in, marked, total = [], [], [], 0, 0
    for f in F.FAMILIES:
        for i in range(60):
            sc = L.scene(f, i)
            if not sc["knobs"]:
                no_notes.append(f)
            for q, a, kind, qk in sc["qas"]:
                total += 1
                if not kind:
                    no_kind.append(f)
                ch = L.choices(f, q, a, sc["story"], knobs=qk)
                if ch is not None:
                    marked += 1
                    if a not in ch or len(set(ch)) < 2:
                        not_in.append((f, q, a, ch))
    check("every family notes its level and every question its kind", not no_kind and not no_notes,
          str(sorted(set(no_kind + no_notes))))
    check("every marked question has its answer among 2+ choices", not not_in, str(not_in[:2])[:300])
    check("almost every question can be marked", marked / total > 0.95, f"{marked} of {total}")
    sc = L.scene("lines", 0)
    st, _ = F.GEN["lines"](__import__("random").Random(f"field1|{F.VERSION}|lines|{('levels', 0)}|0"))
    check("the level table's scenes are the builder's scenes (same dice)", sc["story"] == st.tell())


def main():
    print("Field1 self-tests")
    for t in (t_families, t_walls, t_meaning, t_answers, t_levels, t_opal):
        print(f"\n{t.__name__[2:]}")
        try:
            t()
        except Exception as e:                         # noqa: BLE001
            import traceback
            traceback.print_exc()
            check(f"{t.__name__} ran", False, repr(e)[:200])
    print(f"\n{len(OK)} passed, {len(BAD)} failed" + (": " + ", ".join(BAD) if BAD else ""))
    sys.exit(1 if BAD else 0)


if __name__ == "__main__":
    main()
