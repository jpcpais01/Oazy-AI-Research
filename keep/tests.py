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
from .model import Keep, KeepConfig, Store, norm

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
    for extra in ({}, KEEP):
        torch.manual_seed(0)
        m = Told(KeepConfig(**CFG, **extra))
        res = R.exam(m, ENC, TEXT, (64,), 24, torch.device("cpu"), batch=10)["64"]
        assert all(v["right"] == 100.0 for v in res["groups"].values()), res["groups"]
        assert sum(v["n"] for v in res["groups"].values()) == len(answer)
        assert ("write" in res) == bool(extra)


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
    assert set(res["exam"]) == set(res["train"]) == {"keep", "full"}
    assert (out / "t" / "keep" / "ckpt.pt").stat().st_mtime_ns == when, "a finished model was trained again"
    for arm in R.ARMS:
        assert set(res["exam"][arm]) == {"0", "128", "512"}
        assert "facts" in res["exam"][arm]["0"]["groups"]
    assert "write" in res["exam"]["keep"]["0"] and "write" not in res["exam"]["full"]["0"]
    rep = (out / "t" / "report.md").read_text()
    assert "| Keep | plain facts |" in rep and "| Twin | Field1 stories |" in rep and "write strength" in rep
    sd = torch.load(out / "t" / "keep" / "ckpt.pt", map_location="cpu", weights_only=False)
    assert sd["done"] and KeepConfig.from_dict(sd["config"]).stores == (2, 4)


if __name__ == "__main__":
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed" + (": " + ", ".join(FAILED) if FAILED else ""))
    sys.exit(1 if FAILED else 0)
