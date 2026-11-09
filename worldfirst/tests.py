"""Checks for stage 1: the world tells the truth, the secret exams are walled off, and both models run.

    python3 -m worldfirst.tests            # about a minute on a CPU
"""
from __future__ import annotations

import random
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
            return len(f["a"]) - 1 >= 6
        if e.family == "lines":
            return len(f["order"]) >= 7
        if e.family == "boxes":
            sp = f["start_parent"]
            return max(depth(sp, x) for x in sp) >= 5
        if e.family == "family":
            return any(len(c) >= 3 or tuple(c) in W.FAM_HELD_OUT for c in f["asked"])
        if e.family == "falling":
            return f["movers"] >= 4
        if e.family == "machines":
            return tuple(f["rule"]) in set(W.MACHINES["new-mixes"]) | set(W.MACHINES["deeper"])
    n = 0
    for e in train_eps(150):
        assert not secret_like(e), e.text()
        n += 1
    for e in secret_eps(40):
        assert secret_like(e), (e.family, e.split, e.text())
    assert n >= 3000


@check
def machine_rules_are_new_and_answers_unique():
    seen = set(W.MACHINES[1]) | set(W.MACHINES[2])
    assert not (seen & set(W.MACHINES["new-mixes"])) and not (seen & set(W.MACHINES["deeper"]))
    assert len(W.MACHINES["new-mixes"]) == len(W.MACH_HELD_OUT)
    for sp in ("new-mixes", "deeper"):
        for s in range(80):
            e = W.make("machines", 2 if sp == "new-mixes" else 3, sp, s)
            f = e.facts["rule"]
            ex = [(int(e.tokens[i]), int(e.tokens[j])) for i in range(len(e.tokens)) for j in range(len(e.tokens))
                  if e.mentions[i] >= 0 and e.mentions[i] == e.mentions[j] and i < j]
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
def curriculum_moves_up():
    c = W.Curriculum()
    assert c.state()["sums"] == 1
    for _ in range(400):
        c.update("sums", 1, 1.0)
    assert c.state()["sums"] == 2
    picks = [c.pick() for _ in range(2000)]
    assert all(lv in W.TRAIN_LEVELS[f] and lv <= c.state()[f] for f, lv in picks)


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
        for preset in ("tiny", "small", "base"):
            laws, rival = make_pair(len(v), preset)
            a, b = n_params(laws), n_params(rival)
            assert abs(a - b) / a < 0.01, (preset, a, b)

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
