"""Checks for Keep: the model only sees the past, old text reaches it through the stores alone, the stores obey the
delta rule, the streams and the exam are what they say, and the whole experiment runs.

    python3 -m keep.tests            # about a minute on a CPU; needs no data and no network
"""
from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import ledger
from .model import Keep, KeepConfig, Store, near, norm

PASSED, FAILED = [], []
TMP = Path(tempfile.mkdtemp(prefix="keep-tests-"))


def check(fn):
    try:
        fn()
        PASSED.append(fn.__name__)
        print(f"  ok    {fn.__name__}", flush=True)
    except Exception:
        FAILED.append(fn.__name__)
        print(f"  FAIL  {fn.__name__}", flush=True)
        traceback.print_exc()
    return fn


def stand_in():
    """Offline: a small byte-level BPE in SmolLM2's format, trained on scenes and made-up text; and a data folder."""
    from bee2.tests import fake_docs
    docs = fake_docs(3000, random.Random(1))
    if not os.environ.get("OAZY_SMOL_TOKENIZER"):
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
        from oazy.field1 import levels, FAMILIES
        seen = list(docs)
        for i in range(1500):
            sc = ledger.scene(i, "tok")
            seen.append(sc["story"] + "\n" + "\n".join(f"{lead} {a}." for lead, a, *_ in sc["lines"]))
        for f in FAMILIES:
            for i in range(40):
                sc = levels.scene(f, i, "tok")
                seen.append(sc["story"] + "\n" + "\n".join(f"{q} Answer: {a}." for q, a, *_ in sc["qas"]))
        t = Tokenizer(models.BPE())
        t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        t.decoder = decoders.ByteLevel()
        tr = trainers.BpeTrainer(vocab_size=2000, special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
        t.train_from_iterator(seen, tr)
        t.save(str(TMP / "tok.json"))
        os.environ["OAZY_SMOL_TOKENIZER"] = str(TMP / "tok.json")
    from oazy import tok
    enc = tok.get("smollm2")
    ids = []
    for d in docs:
        ids += [enc.eot_token] + enc.encode_ordinary(d)
    (TMP / "data").mkdir(exist_ok=True)
    np.array(ids, dtype=np.uint16).tofile(TMP / "data" / "train.bin")
    return enc


ENC = stand_in()
TEXT = D.Text(TMP / "data")
CFG = dict(vocab_size=ENC.n_vocab, d=32, n_layer=3, n_head=2, max_len=96)
KEEP = dict(window=8, stores=(1, 3), side=8, k=4, dv=8, d_addr=8, chunk=8)


def models():
    torch.manual_seed(0)
    twin, keep = Keep(KeepConfig(**CFG)), Keep(KeepConfig(**CFG, **KEEP))
    for m in (twin, keep):                        # the zero-initialised outputs would hide every path
        for p in m.parameters():
            if float(p.abs().sum()) == 0:
                torch.nn.init.normal_(p, std=0.05)
    return twin.eval(), keep.eval()


@check
def only_the_past():
    for m in models():
        x = torch.randint(1, CFG["vocab_size"], (2, 64))
        y = x.clone()
        y[:, 40] = (y[:, 40] + 1) % CFG["vocab_size"]
        with torch.no_grad():
            a, b = m.hidden(x)[0], m.hidden(y)[0]
        assert float((a[:, :40] - b[:, :40]).abs().max()) == 0, "a later token changed an earlier place"
        assert float((a[:, 40:] - b[:, 40:]).abs().max()) > 0


@check
def attention_reach():
    """Without stores, a token cannot be felt further than layers x (window - 1) places on."""
    torch.manual_seed(0)
    m = Keep(KeepConfig(**CFG, window=8))
    for p in m.parameters():
        if float(p.abs().sum()) == 0:
            torch.nn.init.normal_(p, std=0.05)
    x = torch.randint(1, CFG["vocab_size"], (2, 64))
    y = x.clone()
    y[:, 0] = (y[:, 0] + 1) % CFG["vocab_size"]
    with torch.no_grad():
        a, b = m.hidden(x)[0], m.hidden(y)[0]
    reach = CFG["n_layer"] * (8 - 1)
    assert float((a[:, reach + 1:] - b[:, reach + 1:]).abs().max()) == 0
    assert float((a[:, :8] - b[:, :8]).abs().max()) > 0


@check
def old_text_comes_through_the_stores():
    """Past the attention's reach in one window, and from one window to the next, only the stores carry anything."""
    _, m = models()
    x = torch.randint(1, CFG["vocab_size"], (2, 96))
    y = x.clone()
    y[:, 0] = (y[:, 0] + 1) % CFG["vocab_size"]
    with torch.no_grad():
        a, b = m.hidden(x)[0], m.hidden(y)[0]
        assert float((a[:, 60:] - b[:, 60:]).abs().max()) > 0, "nothing came through the stores"
        nxt = torch.randint(1, CFG["vocab_size"], (2, 32))
        _, sa, _ = m.hidden(x, m.init_state(2))
        _, sb, _ = m.hidden(y, m.init_state(2))
        ha, hb = m.hidden(nxt, sa)[0], m.hidden(nxt, sb)[0]
        fresh1, fresh2 = m.hidden(nxt, m.init_state(2))[0], m.hidden(nxt, m.init_state(2))[0]
    assert float((ha - hb).abs().max()) > 0, "the next window did not feel what the last one read"
    assert float((fresh1 - fresh2).abs().max()) == 0


@check
def delta_rule():
    """Write strength 1: what was written under an address is what is read under it; a second write of the same thing
    changes nothing; a write touches at most k rows; and what is written under a token's address is the NEXT token."""
    torch.manual_seed(0)
    cfg = KeepConfig(**CFG, **KEEP)
    s = Store(cfg)
    torch.nn.init.zeros_(s.gate.weight)
    torch.nn.init.zeros_(s.gate.bias)             # beta = 2 * sigmoid(0) = 1
    h = torch.randn(1, 1, cfg.d).expand(1, 3 * cfg.chunk, cfg.d).contiguous()
    S0 = torch.zeros(1, s.rows, cfg.dv)
    with torch.no_grad():
        out, S1, beta = s(h, S0.clone())
        v = s.val(norm(h[:, :1]))
        want = s.out(v)
        assert float(out[:, :cfg.chunk].abs().max()) == 0, "the first chunk read an empty store"
        assert torch.allclose(out[:, cfg.chunk:], want.expand(-1, 2 * cfg.chunk, -1), atol=1e-5), "what was written is not what is read"
        assert int((S1.abs().sum(-1) > 0).sum()) <= cfg.k
        assert torch.allclose(beta[:, 1:], torch.ones_like(beta[:, 1:])) and float(beta[:, 0].abs().max()) == 0
        g = torch.randn(1, cfg.chunk, cfg.d)
        _, S2, _ = s(g, torch.zeros(1, s.rows, cfg.dv))
        assert int((S2.abs().sum(-1) > 0).sum()) <= cfg.k * cfg.chunk
        a, b, c = torch.randn(3, cfg.d)                         # ... a b ... then a again: under a's address lies b
        seq = torch.stack([a, b] + [c] * (cfg.chunk - 2) + [a] + [c] * (cfg.chunk - 1))[None]
        out, _, _ = s(seq, torch.zeros(1, s.rows, cfg.dv))
        assert torch.allclose(out[0, cfg.chunk], s.out(s.val(norm(b[None])))[0], atol=1e-5), "a place that comes back does not find what followed it"


@check
def soft_read():
    """The optional soft read is the softmax over every row of the product key's scores, so every row gets a
    gradient; and the k rows of an address are exactly the k best rows."""
    torch.manual_seed(0)
    cfg = KeepConfig(**CFG, **KEEP, read="soft")
    s = Store(cfg)
    hn = norm(torch.randn(2, 5, cfg.d))
    S = torch.randn(2, s.rows, cfg.dv, requires_grad=True)
    s1, s2 = s.scores(hn)
    fast = s.read_all(s1.softmax(-1), s2.softmax(-1), S)
    every = (s1[..., :, None] + s2[..., None, :]).flatten(-2).softmax(-1)           # (2, 5, rows), row = i * side + j
    slow = torch.einsum("btr,brd->btd", every, S)
    assert torch.allclose(fast, slow, atol=1e-5)
    idx, w = s.address(s1, s2)
    best = every.topk(cfg.k, -1).indices
    assert (idx.sort(-1).values == best.sort(-1).values).all(), "the rows written are not the k best rows"
    fast.sum().backward()
    assert int((S.grad.abs().sum(-1) > 0).sum()) == S.shape[0] * s.rows
    out, _, _ = s(torch.randn(2, 16, cfg.d), torch.zeros(2, s.rows, cfg.dv))
    assert out.shape == (2, 16, cfg.d)


@check
def the_window_alone_is_the_mask():
    """Attention computed over the window only gives what a T x T mask gives: whole blocks, a ragged end, one block."""
    torch.manual_seed(0)
    for T, W in ((64, 8), (61, 8), (9, 8), (40, 16), (50, 7)):
        q, k, v = torch.randn(3, 2, 2, T, 6)
        i = torch.arange(T)
        mask = (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < W)
        want = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        assert torch.allclose(near(q, k, v, W), want, atol=1e-5), f"T {T}, window {W}"


@check
def the_sparse_way_is_the_dense_way():
    """The store that touches only its rows (in-place writes, a gradient table filled on the way back) against the
    whole table inside autograd: the same reads, the same store after, the same gradient for every weight and for
    the vectors that came in. From an empty store and from a full one."""
    torch.manual_seed(0)
    cfg = KeepConfig(**CFG, **KEEP)
    s = Store(cfg)
    for start in (torch.zeros(3, s.rows, cfg.dv), torch.randn(3, s.rows, cfg.dv)):
        got = []
        for sparse in (True, False):
            s.sparse = sparse
            s.zero_grad()
            h = torch.randn(3, 5 * cfg.chunk + 3, cfg.d, generator=torch.Generator().manual_seed(1), requires_grad=True)
            before = start.clone()
            out, S, beta = s(h, before)
            assert torch.equal(before, start), "the caller's store was written into"
            (out * torch.randn(out.shape, generator=torch.Generator().manual_seed(2))).sum().backward()
            got.append([out.detach(), S.detach(), h.grad] + [p.grad.clone() for p in s.parameters()])
        s.sparse = True
        assert float(got[0][2].abs().sum()) > 0 and all(float(g.abs().sum()) > 0 for g in got[0][3:])
        for a, b in zip(*got):
            assert torch.allclose(a, b, atol=1e-5, rtol=1e-4), float((a - b).abs().max())
    _, m = models()                                 # and through the whole model: two stores, the layers between
    m.train()
    x = torch.randint(1, CFG["vocab_size"], (2, 65))
    grads = []
    for sparse in (True, False):
        for st in m.store.values():
            st.sparse = sparse
        m.zero_grad()
        m.loss(x[:, :-1], x[:, 1:], m.init_state(2))[0].backward()
        grads.append([p.grad.clone() for p in m.parameters()])
    for st in m.store.values():
        st.sparse = True
    for a, b in zip(*grads):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-3), float((a - b).abs().max())


@check
def later_reads_teach_earlier_writes():
    _, m = models()
    m.train()
    x = torch.randint(1, CFG["vocab_size"], (2, 65))
    state = m.init_state(2)
    loss, new, info = m.loss(x[:, :-1], x[:, 1:], state)
    loss.backward()
    for i, s in m.store.items():
        assert float(s.val.weight.grad.abs().sum()) > 0 and float(s.gate.weight.grad.abs().sum()) > 0, f"store {i} is not taught"
    assert all(not v.requires_grad for v in m.detach(new).values())
    assert set(new) == set(state) and len(info["write"]) == len(m.store)
    asked = torch.zeros(2, 64, dtype=torch.bool)
    asked[:, 50:] = True
    l1, _, i1 = m.loss(x[:, :-1], x[:, 1:], m.init_state(2), asked=asked)
    l4, _, i4 = m.loss(x[:, :-1], x[:, 1:], m.init_state(2), asked=asked, asked_weight=4.0)
    want = float(i1["lm"]) + 3 * float(asked.float().mean()) * float(i1["asked"])
    assert abs(float(l1) - float(i1["lm"])) < 1e-5 and abs(float(l4) - want) < 1e-4, "the asked tokens do not count as they should"
    costly = Keep(KeepConfig(**CFG, **KEEP, write_cost=0.5))
    costly.load_state_dict(m.state_dict())
    l2, _, i2 = costly.loss(x[:, :-1], x[:, 1:], costly.init_state(2))
    assert float(l2) > float(i2["lm"]) and abs(float(i2["lm"]) - float(info["lm"])) < 1e-4, "the write cost is not added on top"


@check
def ledger_scenes():
    kinds = set()
    for i in range(300):
        a, b = ledger.scene(i, "t"), ledger.scene(i, "t")
        assert a == b, "a scene is not the same twice"
        for lead, ans, ch, kind in a["lines"]:
            kinds.add(kind)
            assert ans in ch and len(set(ch)) == len(ch) == ledger.N_CHOICES
            v = ans.split()[-1]
            assert v in a["story"], (ans, a["story"])
            if kind == "changed":                 # the old value is among the choices and was told before the new one
                old = [c.split()[-1] for c in ch if c != ans and c.split()[-1] in a["story"]]
                assert old, (ch, a["story"])
    assert kinds == {"facts", "changed"}
    assert ledger.scene(0, "t") != ledger.scene(0, "u")


@check
def streams():
    g = D.stream(random.Random(3), TEXT, ENC, 256)
    eot, open_block, n_scenes, n_blocks = ENC.eot_token, None, 0, 0
    for _ in range(300):
        t, kind = next(g)
        assert t[0] == eot
        s = ENC.decode(t[1:])
        if kind == D.ASKED:
            assert open_block is not None and s == open_block, "questions without their story before them"
            open_block, n_blocks = None, n_blocks + 1
            continue
        if kind == D.TEXT:
            continue
        for fam in [D.LEDGER] + D._families()[0]:           # is it the next scene of this stream?
            sc = D.scene(fam, n_scenes, (D.TRAIN_SEED, 0))
            if ENC.decode(ENC.encode_ordinary(sc["story"])) == s:
                assert open_block is None, "two scenes open at once"
                open_block = ENC.decode(ENC.encode_ordinary("\n".join(f"{lead} {a}." for lead, a, *_ in sc["lines"])))
                n_scenes += 1
                break
        else:
            raise AssertionError("a story that is not this stream's next scene")
    assert n_scenes > 20 and n_blocks >= n_scenes - 1
    from . import ledger
    g, told, sizes = D.stream(random.Random(4), TEXT, ENC, 64, p_book=1.0, book_max=40), None, set()   # every scene a book
    for _ in range(120):
        t, kind = next(g)
        assert t[0] == eot
        if kind == D.STORY:
            assert told is None, "two books open at once"
            m = next(m for m in range(8, 41) if ENC.decode(t).count(".") == m and
                     [eot] + ENC.encode_ordinary(ledger.book(len(sizes), m, (D.TRAIN_SEED, 0))["stories"][0]) == t[:len(ENC.encode_ordinary(ledger.book(len(sizes), m, (D.TRAIN_SEED, 0))["stories"][0])) + 1])
            told = ledger.book(len(sizes), m, (D.TRAIN_SEED, 0))
            sizes.add((len(sizes), m))
        elif kind == D.ASKED:
            lines = ENC.decode(t[1:]).split("\n")
            assert 8 <= len(lines) <= 24 and len(set(lines)) == len(lines)
            assert set(lines) <= {f"{lead} {a}." for lead, a, *_ in told["lines"]}, "a book is asked what it did not tell"
            told = None
    assert len(sizes) > 5 and len({m for _, m in sizes}) > 3
    r = D.Rows(TEXT, ENC, 3, 64, 256, seed=0)
    (a, ka), (b, kb) = r.next(), r.next()
    assert a.shape == ka.shape == (3, 65) and (a[:, -1] == b[:, 0]).all() and (ka[:, -1] == kb[:, 0]).all(), \
        "a window does not go on where the last one stopped"
    assert not (a[0] == a[1]).all() and set(np.unique(ka)) <= {D.TEXT, D.STORY, D.ASKED}
    again = D.Rows(TEXT, ENC, 3, 64, 256, seed=0).next()[0]
    assert (a == again).all(), "the same seed is not the same stream"
    two = D.Rows(TEXT, ENC, 2, 64, 256, seed=0, first_row=1).next()[0]
    assert (two[0] == a[1]).all(), "a row depends on which worker makes it"
    f = D.Feed(TMP / "data", ENC, 4, 64, 256, seed=0, workers=2)
    try:
        fa, fk = f.next()
    finally:
        f.close()
    assert (fa[:3] == a).all() and (fk[:3] == ka).all(), "the workers' streams are not the streams made here"


@check
def exam_sets():
    eot = ENC.eot_token
    sets = {gap: D.exam_set(ENC, TEXT, gap, 40, q_max=200) for gap in (0, 64, 700)}
    for gap, (prefix, block, marks, spans) in sets.items():
        assert (prefix.shape[1] - spans[:, 1] == gap).all(), "the gap is not exactly the gap"
        assert (prefix[np.arange(40), spans[:, 0]] == eot).all() and (block[:, 0] == eot).all()
        assert len(marks) > 60 and {D.group(m[4]) for m in marks} == {"facts", "changed", "field1"}
        for r, p, cands, right, kind in marks:
            assert block[r, p + 1] == cands[right], "a mark does not point at the answer"
            assert len(set(cands)) == len(cands) >= 2
    assert (sets[0][1] == sets[700][1]).all() and sets[0][2] == sets[64][2], "the scenes change with the gap"
    story = ENC.decode(sets[0][0][0, sets[0][3][0, 0] + 1:sets[0][3][0, 1]])
    assert story == ENC.decode(ENC.encode_ordinary(D.scene(D.LEDGER, 0, (D.EXAM_SEED, 0))["story"]))
    assert D.scene(D.LEDGER, 0, (D.EXAM_SEED, 0)) != D.scene(D.LEDGER, 0, (D.TRAIN_SEED, 0))


@check
def load_sets():
    """The capacity exam: a book of m facts about m different subjects, then some of them asked, evenly over the book."""
    from . import ledger
    bk = ledger.book(0, 300)
    assert len(bk["lines"]) == 300 and len(bk["stories"]) == 38 and len({lead for lead, *_ in bk["lines"]}) > 290
    assert all(a in ch and len(set(ch)) == 4 for _, a, ch, _ in bk["lines"])
    eot = ENC.eot_token
    for m, gap in ((8, 0), (120, 50)):
        prefix, block, marks, spans = D.load_set(ENC, TEXT, m, 6, q_max=200, gap=gap)
        assert (prefix.shape[1] - spans[:, 1] == gap).all() and (prefix[np.arange(6), spans[:, 0]] == eot).all()
        told = ENC.decode(prefix[0, spans[0, 0]:spans[0, 1]])
        assert all(s in told for s in ledger.book(0, m, (D.EXAM_SEED, 0))["stories"]), "the book is not what was told"
        assert len(marks) >= 6 * 4 and {D.group(k) for *_, k in marks} == {"facts"} and {k for *_, k in marks} <= set(D.WHEN)
        if m > 8:
            assert {k for *_, k in marks} == set(D.WHEN)
        for r, p, cands, right, kind in marks:
            assert block[r, p + 1] == cands[right], "a mark does not point at the answer"
    assert ledger.book(0, 2520) and len(D.load_set(ENC, TEXT, 8, 6)[2]) == len(D.load_set(ENC, TEXT, 8, 6, gap=9)[2])


@check
def the_exam_can_be_passed():
    """A model that is handed the answers scores 100%, one that is not scores as guessing does: the marking is right
    for both ways of reading (Keep's windows, the twin's last max_len tokens)."""
    from . import run as R
    prefix, block, marks, _ = D.exam_set(ENC, TEXT, 64, 24, q_max=48)
    answer = {(r, p): cands[right] for r, p, cands, right, _ in marks}

    class Told(Keep):                             # scores the next piece of the block as the block has it
        def hidden(self, x, state=None):
            h, new, info = super().hidden(x, state)
            self.seen = x
            return h, new, info

        def scores_at(self, h, rows, cols):
            nxt = self.seen[rows, cols + 1]
            return torch.nn.functional.one_hot(nxt, self.cfg.vocab_size).float()
    for extra, alone in (({}, True), (KEEP, True), (dict(KEEP, window=0), False)):     # the twin, Keep, Keep-wide read as a stream
        torch.manual_seed(0)
        m = Told(KeepConfig(**CFG, **extra))
        res = R.exam(m, ENC, TEXT, (64,), 24, torch.device("cpu"), batch=10, alone=alone)["64"]
        assert all(v["right"] == 100.0 for v in res["groups"].values()), res["groups"]
        assert sum(v["n"] for v in res["groups"].values()) == len(answer)
        assert ("write" in res) == (bool(extra) and alone)
        res = R.exam(m, ENC, TEXT, (40,), 6, torch.device("cpu"), batch=4, alone=alone, make=D.load_set)["40"]      # and a book of 40 facts
        assert res["groups"]["facts"]["right"] == 100.0 and res["groups"]["facts"]["n"] > 15, res["groups"]


@check
def the_whole_run():
    from . import run as R
    out = TMP / "runs"
    args = ["--preset", "tiny", "--device", "cpu", "--workers", "0", "--data", str(TMP / "data"), "--sub", "", "--runs", str(out),
            "--name", "t", "--tokens", "6e4", "--exam-n", "12", "--every", "1000"]
    quiet = open(os.devnull, "w")
    old, sys.stdout = sys.stdout, quiet
    try:
        R.main(args)
        when = (out / "t" / "keep" / "ckpt.pt").stat().st_mtime_ns
        R.main(args + ["--only", "report"])
    finally:
        sys.stdout = old
    res = json.loads((out / "t" / "results.json").read_text())
    assert set(res["train"]) == {"keep", "full", "wide", "mixed"}
    assert set(res["exam"]) == {"keep", "full", "wide", "wide_alone", "mixed", "mixed_alone"}
    assert (out / "t" / "keep" / "ckpt.pt").stat().st_mtime_ns == when, "a finished model was trained again"
    for arm in R.ROWS:
        assert set(res["exam"][arm]) == {"0", "128", "512"}
        assert "facts" in res["exam"][arm]["0"]["groups"]
    assert "write" in res["exam"]["keep"]["0"] and "write" not in res["exam"]["full"]["0"]
    assert "write" in res["exam"]["wide_alone"]["0"] and "write" not in res["exam"]["wide"]["0"]
    rep = (out / "t" / "report.md").read_text()
    assert "| Keep | plain facts |" in rep and "| Twin | Field1 stories |" in rep and "write strength" in rep
    assert "| Keep-wide mixed | plain facts |" in rep and "| Keep-wide mixed, stores alone | plain facts |" in rep
    assert "| Keep-wide | plain facts |" in rep and "| Keep-wide, stores alone | Field1 stories |" in rep and "Keep-wide's write strength" in rep
    sd = torch.load(out / "t" / "wide" / "ckpt.pt", map_location="cpu", weights_only=False)
    wide = KeepConfig.from_dict(sd["config"])
    assert wide.window == 0 and wide.stores == (2, 4), "Keep-wide is the whole window and the stores"
    sd = torch.load(out / "t" / "keep" / "ckpt.pt", map_location="cpu", weights_only=False)
    assert sd["done"] and KeepConfig.from_dict(sd["config"]).stores == (2, 4)
    from . import mute as Mu                       # the saved model again, with stores muted; and a run with one store
    sys.stdout = quiet
    try:
        Mu.main(["--name", "t", "--runs", str(out), "--data", str(TMP / "data"), "--sub", "", "--gaps", "0,128", "--n", "12", "--device", "cpu"])
        Mu.main(["--name", "t", "--arm", "wide", "--runs", str(out), "--data", str(TMP / "data"), "--sub", "", "--gaps", "0", "--n", "12", "--device", "cpu"])
        R.main(args[:args.index("--name")] + ["--name", "one", "--only", "keep", "--stores", "4", "--books", "0.5", "--book-max", "16", "--tokens", "3e4", "--exam-n", "12", "--every", "1000"])
    finally:
        sys.stdout = old
    rep = (out / "t" / "mute.md").read_text()
    assert "| nothing muted |" in rep and "| store after block 2 muted |" in rep and "| all stores muted |" in rep
    assert "# Keep-wide: which store" in (out / "t" / "mute-wide.md").read_text()
    from . import load as Lo                       # how much the saved models hold at once
    sys.stdout = quiet
    try:
        Lo.main(["--name", "t", "--runs", str(out), "--data", str(TMP / "data"), "--sub", "", "--facts", "4,24", "--n", "4", "--gap", "20", "--device", "cpu"])
    finally:
        sys.stdout = old
    rep = (out / "t" / "load.md").read_text()
    assert "| 4 facts | 24 facts |" in rep and "| Keep |" in rep and "| Twin |" in rep and "| Keep-wide mixed |" in rep and "| In the last third" in rep
    got = json.loads((out / "t" / "load.json").read_text())
    assert set(got["exam"]) == set(R.ARMS) and got["exam"]["keep"]["24"]["groups"]["facts"]["n"] >= 8
    sd = torch.load(out / "one" / "keep" / "ckpt.pt", map_location="cpu", weights_only=False)
    assert KeepConfig.from_dict(sd["config"]).stores == (4,) and sd["summary"]["books"] == 0.5


@check
def the_reach_can_be_cut():
    """Keep-wide with its reach cut for a step is Keep, weight for weight; with it left alone it sees the whole window."""
    _, keep = models()
    wide = Keep(KeepConfig(**CFG, **dict(KEEP, window=0)))
    wide.load_state_dict(keep.state_dict())
    x = torch.randint(1, CFG["vocab_size"], (2, 64))
    with torch.no_grad():
        far = wide.hidden(x)[0]
        wide.reach = KEEP["window"]
        near_, st, _ = wide.hidden(x)
        want, st_k, _ = keep.hidden(x)
        wide.reach = None
        again = wide.hidden(x)[0]
    assert torch.allclose(near_, want, atol=1e-6) and all(torch.allclose(st[i], st_k[i], atol=1e-6) for i in st)
    assert float((far - want).abs().max()) > 1e-3 and torch.equal(far, again)


@check
def a_muted_store_is_not_read():
    """With every store muted nothing passes from one window to the next; with one muted, the other still carries."""
    _, m = models()
    x, y, nxt = (torch.randint(1, CFG["vocab_size"], (2, 64)) for _ in range(3))
    def carried():
        with torch.no_grad():
            sa, sb = m.hidden(x, m.init_state(2))[1], m.hidden(y, m.init_state(2))[1]
            return float((m.hidden(nxt, sa)[0] - m.hidden(nxt, sb)[0]).abs().max())
    try:
        assert carried() > 0
        m.muted = {1}
        assert carried() > 0
        m.muted = {1, 3}
        assert carried() == 0, "a muted store was read"
    finally:
        m.muted = set()


FEED = dict(window=8, stores=(1,), side=8, k=4, dv=8, d_addr=8, chunk=8, feed=True)


def fed(**over):
    torch.manual_seed(0)
    m = Keep(KeepConfig(**CFG, **{**FEED, **over}))
    for p in m.parameters():
        if float(p.abs().sum()) == 0:
            torch.nn.init.normal_(p, std=0.05)
    return m.eval()


@check
def a_chunk_at_a_time_is_the_block():
    """The blocks run chunk by chunk, each handed the keys and values still in reach, give what they give on the whole
    sequence: with a window and without, at a chunk size that divides nothing."""
    from .model import _step
    for win in (8, 0):
        torch.manual_seed(0)
        m = Keep(KeepConfig(**CFG, window=win))
        for p in m.parameters():
            if float(p.abs().sum()) == 0:
                torch.nn.init.normal_(p, std=0.05)
        x = torch.randint(1, CFG["vocab_size"], (2, 61))
        with torch.no_grad():
            want = m.hidden(x)[0]
            past, outs = [None] * len(m.blocks), []
            for a in range(0, 61, 7):
                h = norm(m.emb(x[:, a:a + 7]))
                n = h.size(1)
                for l, blk in enumerate(m.blocks):
                    h, past[l] = _step(blk, h, (m.cos[a:a + n], m.sin[a:a + n]), past[l], win)
                outs.append(norm(h))
                assert win == 0 or past[0][0].size(2) <= win - 1, "more keys were kept than are in reach"
        assert torch.allclose(torch.cat(outs, 1), want, atol=1e-5)


@check
def the_top_writes_the_store():
    """feed: only the past is seen; the store holds what the LAST block says (plain Keep's does not depend on it);
    a chunk reads what the chunks before wrote and not its own writes; a later chunk's loss teaches the writes."""
    m = fed()
    x = torch.randint(1, CFG["vocab_size"], (2, 64))
    y = x.clone()
    y[:, 40] = (y[:, 40] + 1) % CFG["vocab_size"]
    with torch.no_grad():
        a, sa, wa = m.hidden(x, m.init_state(2))
        b = m.hidden(y, m.init_state(2))[0]
    assert float((a[:, :40] - b[:, :40]).abs().max()) == 0 and float((a[:, 40:] - b[:, 40:]).abs().max()) > 0
    assert wa["1"].shape == (2, 64) and float(wa["1"][:, 0].abs().max()) == 0 and float(wa["1"][:, 1:].min()) > 0, "the first token writes nothing"
    plain = Keep(KeepConfig(**CFG, **{**FEED, "feed": False})).eval()
    plain.load_state_dict(m.state_dict())
    def store_after(model, bump):
        last = model.blocks[-1].down.weight
        keep_ = last.detach().clone()
        with torch.no_grad():
            last.add_(bump)
            st = model.hidden(x, model.init_state(2))[1]["1"]
            last.copy_(keep_)
        return st
    assert float((store_after(m, 0.0) - store_after(m, 0.3)).abs().max()) > 1e-4, "the store does not hold what the top says"
    assert float((store_after(plain, 0.0) - store_after(plain, 0.3)).abs().max()) == 0
    with torch.no_grad():                           # the first chunk reads an empty store: as if the store were muted
        m.muted = {1}
        dumb = m.hidden(x, m.init_state(2))[0]
        m.muted = set()
    assert float((a[:, :8] - dumb[:, :8]).abs().max()) == 0 and float((a[:, 8:] - dumb[:, 8:]).abs().max()) > 0
    m.train()
    val = m.store["1"].val.weight
    for sl, some in ((slice(0, 8), False), (slice(56, 64), True)):
        m.zero_grad()
        m.hidden(x, m.init_state(2))[0][:, sl].square().sum().backward()
        assert (val.grad is not None and float(val.grad.abs().sum()) > 0) == some, "which reads teach the writes"


@check
def the_top_points():
    """point: the top files its write under one of the tokens before it. Sure of the token before, it is the plain
    fed store; pointing elsewhere it is not; the pointer is taught by what later reads need."""
    m, p = fed(), fed(point=5)
    p.load_state_dict(m.state_dict(), strict=False)
    x = torch.randint(1, CFG["vocab_size"], (2, 64))
    st = p.store["1"]
    assert st.pb.shape == (5,) and float(st.pb.softmax(0)[0]) > 0.7, "it starts at the token before"
    with torch.no_grad():
        want = m.hidden(x, m.init_state(2))
        st.pb.copy_(torch.tensor([60.0, 0, 0, 0, 0]))
        st.pq.weight.zero_()
        got = p.hidden(x, p.init_state(2))
        assert torch.allclose(got[0], want[0], atol=1e-4) and torch.allclose(got[1]["1"], want[1]["1"], atol=1e-4)
        st.pb.copy_(torch.tensor([0.0, 0, 60.0, 0, 0]))
        far = p.hidden(x, p.init_state(2))
        assert float((far[1]["1"] - want[1]["1"]).abs().max()) > 1e-3
        assert float(far[2]["1"][:, 0].abs().max()) == 0 and float(far[2]["1"][:, 1:].min()) > 0, "only the first token has no token before it"
        st.pb.copy_(torch.tensor([3.0, 0, 0, 0, 0]))
        torch.nn.init.normal_(st.pq.weight, std=0.1)
        # a word that stands twice among the tokens pointed at: its rows get one write, not two (a write at full
        # strength makes the store say exactly what was written, however the pointer is spread)
        st.pb.zero_()
        st.pq.weight.zero_()
        st.gate.weight.zero_()
        st.gate.bias.zero_()
        idx = torch.tensor([[[0, 1, 2, 3]] * 5 + [[9, 9, 9, 9]]])                  # five tokens at one address, then the writing token
        w = torch.full((1, 6, 4), 0.25)
        keys = torch.zeros(1, 6, st.pk.out_features)
        top = torch.randn(1, 1, CFG["d"])
        S1, _ = st.file(top, torch.randn(1, st.rows, st.dv), (idx[:, 5:], w[:, 5:], keys[:, 5:]), (idx[:, :5], w[:, :5], keys[:, :5]))
        assert torch.allclose(S1[0, :4].mean(0), st.val(top)[0, 0], atol=1e-5), "a repeated address was written more than once"
        st.pb.copy_(torch.tensor([3.0, 0, 0, 0, 0]))
        torch.nn.init.normal_(st.pq.weight, std=0.1)
    p.train()
    p.hidden(x, p.init_state(2))[0][:, 32:].square().sum().backward()
    assert all(float(g.grad.abs().sum()) > 0 for g in (st.pq.weight, st.pk.weight, st.pb))


@check
def thinking_runs():
    """keep.think: scenes told again and again, a scene never across two windows, the three models, the exam by
    level and by times told."""
    from . import think as Th
    rows = Th.Rows(ENC, 3, 512, 0, ["switches", "rules"], 3)
    eot = ENC.eot_token
    for _ in range(4):
        w, what = rows.next()
        for r in range(3):
            first = [i for i in range(512) if what[r, i] != D.TEXT]
            assert not first or (what[r, first[0]] == D.STORY and w[r, first[0]] == eot), "a window begins inside a scene"
            assert all(w[r, i] == eot for i in range(512) if what[r, i] == D.TEXT)
    told = Th.scene_tokens(ENC, "switches", 0, "t", 3, 1 << 20)
    one = Th.scene_tokens(ENC, "switches", 0, "t", 1, 1 << 20)
    story, sep, lines = Th.told(ENC, "switches", 0, "t")
    assert told[3] == 3 and len(told[0]) - len(one[0]) == 2 * (len(story) + len(sep)) and told[0][-5:] == one[0][-5:]
    assert Th.scene_tokens(ENC, "switches", 0, "t", 3, len(one[0]) + 5)[3] == 1, "told fewer times when there is no room"
    for p, cands, right, lv, a in told[2]:
        assert told[0][p + 1] == cands[right] and lv.startswith("depth ")
    out = TMP / "runs"
    args = ["--preset", "tiny", "--device", "cpu", "--workers", "0", "--runs", str(out), "--name", "th", "--tokens", "4e4",
            "--families", "switches,rules", "--every", "1000"]
    quiet = open(os.devnull, "w")
    old, sys.stdout = sys.stdout, quiet
    try:
        Th.main(args)
        when = (out / "th" / "feed" / "ckpt.pt").stat().st_mtime_ns
        Th.main(args + ["--only", "report"])
    finally:
        sys.stdout = old
    assert (out / "th" / "feed" / "ckpt.pt").stat().st_mtime_ns == when
    res = json.loads((out / "th" / "think.json").read_text())
    assert set(res["exam"]) == {"feed", "keep", "full"} and set(res["exam"]["feed"]) == {"switches", "rules"}
    cell = res["exam"]["feed"]["switches"]["all"]
    assert set(cell) == {"1", "2", "3"} and all(0 <= c[0] <= c[1] and c[2] <= c[1] and c[1] > 0 for c in cell.values())
    assert any(k.startswith("depth ") for k in res["exam"]["full"]["switches"]) and any(k.startswith("hops ") for k in res["exam"]["keep"]["rules"])
    sd = torch.load(out / "th" / "feed" / "ckpt.pt", map_location="cpu", weights_only=False)
    cfg = KeepConfig.from_dict(sd["config"])
    assert cfg.feed and cfg.point == 4 and cfg.stores == (1,) and not KeepConfig.from_dict(
        torch.load(out / "th" / "keep" / "ckpt.pt", map_location="cpu", weights_only=False)["config"]).feed
    rep = (out / "th" / "think.md").read_text()
    assert "## switches" in rep and "| depth 1 | Keep, stores written from the top |" in rep and "| all | Twin |" in rep and "told 3x" in rep


if __name__ == "__main__":
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed" + (": " + ", ".join(FAILED) if FAILED else ""))
    sys.exit(1 if FAILED else 0)
