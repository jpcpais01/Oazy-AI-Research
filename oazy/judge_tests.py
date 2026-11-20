"""Checks of the judge's lesson (oazy/judge.py), of the judge head (bee6/model.py, `judge=`) and of judge scoring in the
choice tests (oazy/hellaswag.py). CPU, about two minutes (a toy judge is trained for 1,000 steps).

    python3 -m oazy.judge_tests

The made-up world: 8 topics of 30 made-up words. A document is 3 to 8 sentences of ONE topic, and its sentences open
with "First", "Then", "Next", ... in order, so a sentence of another document, or one out of its turn, can be told.
"""
from __future__ import annotations

import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from oazy import term

FAILS = []
ORDER = ["First", "Then", "Next", "After", "Later", "Soon", "Lastly", "Finally"]
SMALL = dict(d_model=64, n_layer=4, attn_heads=4, attn_kv_heads=2, attn_head_dim=16, mlp_hidden=128, max_len=256,
             ctx="128:0", key_offset="1,3", mod_layers="", fp8=0)


def check(name, ok, detail=""):
    term.check(name, ok, detail)
    if not ok:
        FAILS.append(name)


def topics(rng):
    cons, vow = "bdfgklmnprstvz", "aeiou"
    words = set()
    while len(words) < 240:
        words.add("".join(rng.choice(cons) + rng.choice(vow) for _ in range(rng.randint(2, 3))))
    words = sorted(words)
    rng.shuffle(words)
    return [words[i * 30:(i + 1) * 30] for i in range(8)]


def doc(tp, rng, n=None):
    """One document: [sentence, ...] of one topic, in order."""
    w = rng.choice(tp)
    return [ORDER[k] + " " + " ".join(rng.choice(w) for _ in range(rng.randint(4, 9))) + "." for k in range(n or rng.randint(3, 8))]


def stand_in(tmp, texts):
    """A small byte-level BPE in SmolLM2's format (<|endoftext|> = 0), learned from the made-up world."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=700, special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    t.train_from_iterator(texts, tr)
    p = tmp / "judge-standin.json"
    t.save(str(p))
    return p


def runs_of(lab, v):
    """[(start, stop)] of the stretches of marks equal to v."""
    m = np.concatenate([[False], lab == v, [False]])
    d = np.flatnonzero(m[1:] != m[:-1])
    return list(zip(d[::2], d[1::2]))


def main():
    from oazy import tok
    rng = random.Random(0)
    tmp = Path(tempfile.mkdtemp())
    tp = topics(rng)
    docs = [doc(tp, rng) for _ in range(4000)]
    old = os.environ.get("OAZY_SMOL_TOKENIZER")
    os.environ["OAZY_SMOL_TOKENIZER"] = str(stand_in(tmp, [" ".join(d) for d in docs[:1500]] + ['He said "No." (Why?) It is 3.14 m!\n\nNew line.']))
    tok.get.cache_clear()
    try:
        body(tmp, tp, docs, rng)
    finally:
        if old is None:
            os.environ.pop("OAZY_SMOL_TOKENIZER", None)
        else:
            os.environ["OAZY_SMOL_TOKENIZER"] = old
        tok.get.cache_clear()
    if FAILS:
        print(f"\n{len(FAILS)} check(s) FAILED: " + "; ".join(FAILS))
        sys.exit(1)
    term.tests_passed("judge")


def body(tmp, tp, docs, rng):
    from oazy import tok, hellaswag
    from oazy.judge import Decoys
    from oazy.mix import Mix
    enc = tok.get("smollm2")
    eot = enc.eot_token

    # ---------------------------------------------------------------- where a sentence closes
    D = Decoys("smollm2", rate=0.15)
    ids = np.array(enc.encode_ordinary('Soon bade kilo. He said "No." (Why?) It is 3.14 m!\n\nNew line. Mr. Tom left') + [eot]
                   + enc.encode_ordinary("First bade kilo. Then"))
    starts, stops, ok, dc, kind = D.spans(ids)
    got = [enc.decode(ids[a:b + 1]) for a, b in zip(starts, stops)]
    want = ["Soon bade kilo.", ' He said "No."', " (Why?)", " It is 3.14 m!", "\n\nNew line.", " Mr.", " Tom left<|endoftext|>", "First bade kilo."]
    check("a sentence closes at . ! ? (quotes and brackets allowed) before a new word or line, and at a document's end; "
          "3.14 does not close one", got == want, f"{got}")
    check("documents are told apart, and the row's first stretch and a stretch with <|endoftext|> are never swapped",
          list(dc) == [0, 0, 0, 0, 0, 0, 0, 1] and not ok[0] and not ok[6], f"{list(dc)} {list(ok)}")

    # ---------------------------------------------------------------- one row's swaps
    stream = np.array([t for d in docs for t in enc.encode_ordinary(" ".join(d)) + [eot]], dtype=np.uint16)
    T = 257
    rows = [stream[i * T:(i + 1) * T] for i in range(400)]
    g = np.random.default_rng(0)
    out = [D.row(r, g) for r in rows]
    same_len = all(len(o) == T and len(l) == T for o, l in out)
    same_bag = all(np.array_equal(np.sort(o), np.sort(r)) for (o, _), r in zip(out, rows))
    check("a swap keeps the row's length and its pieces (the same pieces in another order)", same_len and same_bag)
    n_dec = sum(len(runs_of(l, 0)) for _, l in out)
    n_ok = sum(int(D.spans(r.astype(np.int64))[2].sum()) for r in rows)
    check("about 15% of the swappable sentences are decoys", 0.11 < n_dec / n_ok < 0.19, f"{n_dec} of {n_ok} = {n_dec / n_ok:.1%}")
    ends_ok = tails = wholes = starts_ok = quiet_ok = moved = 0
    for (o, l), r in zip(out, rows):
        oi = o.astype(np.int64)
        st, sp, _, _, _ = D.spans(oi)
        for a, b in runs_of(l, 0):
            ends_ok += int(b - 1 in set(sp))                              # a decoy runs to its sentence's close
            whole = a in set(st)
            wholes += int(whole)
            tails += int(not whole)
            starts_ok += int(whole or bool(D.word[oi[a]]))                # a moved ending starts where a word starts
            nxt = np.flatnonzero(sp >= b)                                 # the sentence right after: not graded
            if len(nxt):
                seg = l[b:sp[nxt[0]] + 1]
                quiet_ok += int(bool(((seg == -1) | (seg == 0)).all()))
            else:
                quiet_ok += int(bool((l[b:] == -1).all()))
        moved += int((o != r).any())
    check("every decoy runs to a sentence's close, and a moved ending starts where a word starts",
          ends_ok == n_dec and starts_ok == n_dec, f"{ends_ok} and {starts_ok} of {n_dec}")
    check("both kinds are made: whole sentences and endings", wholes > 0.25 * n_dec and tails > 0.25 * n_dec, f"{wholes} whole, {tails} endings")
    check("the sentence right after a decoy is not graded", quiet_ok == n_dec, f"{quiet_ok} of {n_dec}")
    g1, g2 = np.random.default_rng(5), np.random.default_rng(5)
    a, b = D.row(rows[3], g1), D.row(rows[3], g2)
    check("the same generator gives the same swaps (a resumed run reads the same lesson)",
          np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))
    o, l = Decoys("smollm2", rate=0.0).row(rows[3], np.random.default_rng(0))
    check("rate 0 leaves the text as it is, all marked real", np.array_equal(o, rows[3]) and bool((l == 1).all()))
    o, l = next((o, l) for o, l in out if (l == 0).any())
    a, b = runs_of(l, 0)[0]
    print("    an example (the decoy between ⟦ ⟧):  …" + enc.decode(o[max(0, a - 24):a].tolist()) + " ⟦" + enc.decode(o[a:b].tolist())
          + "⟧" + enc.decode(o[b:b + 16].tolist()) + "…")

    # ---------------------------------------------------------------- in the batch maker
    data = tmp / "data"
    (data / "smol").mkdir(parents=True)
    stream.tofile(data / "smol" / "train.bin")
    try:
        m0 = Mix(data / "smol", "fwe:1@0", 128, 8, seed=0, lengths=[128], log=lambda *a, **k: None)
        m1 = Mix(data / "smol", "fwe:1@0", 128, 8, seed=0, lengths=[128], log=lambda *a, **k: None)
    except SystemExit:
        (data / "smol" / "train.bin").rename(data / "train.bin")
        m0 = Mix(data, "fwe:1@0", 128, 8, seed=0, lengths=[128], log=lambda *a, **k: None)
        m1 = Mix(data, "fwe:1@0", 128, 8, seed=0, lengths=[128], log=lambda *a, **k: None)
    m1.judge = Decoys("smollm2", rate=0.3)
    dev = torch.device("cpu")
    x0, y0 = m0.get(dev)
    x1, y1 = m1.get(dev)
    jm = m1.last_judge
    full0, full1 = torch.cat([x0, y0[:, -1:]], 1), torch.cat([x1, y1[:, -1:]], 1)
    check("the batch maker hands the swapped rows and their marks (B, T + 1), the same text as without the judge",
          jm is not None and tuple(jm.shape) == (8, 129) and bool((torch.sort(full0, 1).values == torch.sort(full1, 1).values).all())
          and bool((full0 != full1).any()) and bool(((jm == 0).sum() > 0)) and m0.last_judge is None,
          f"{None if jm is None else tuple(jm.shape)}")
    check("where the marks say real and nothing moved before, the text is the original",
          all(bool((full1[i, :int((jm[i] == 0).nonzero()[0])] == full0[i, :int((jm[i] == 0).nonzero()[0])]).all())
              for i in range(8) if (jm[i] == 0).any()))

    # ---------------------------------------------------------------- the head and its loss
    from gnosis2.model import Gnosis2, Gnosis2Config
    V = enc.n_vocab
    torch.manual_seed(0)
    plain = Gnosis2(Gnosis2Config(**SMALL, vocab_size=V))
    torch.manual_seed(0)
    model = Gnosis2(Gnosis2Config(**SMALL, vocab_size=V, judge=10.0))
    n0, n1 = sum(p.numel() for p in plain.parameters()), sum(p.numel() for p in model.parameters())
    check("judge=0 builds the body as before; the judge head adds one row of numbers (d_model + 1)",
          plain.judge_head is None and n1 - n0 == SMALL["d_model"] + 1, f"{n1 - n0} weights")
    sd = {k: v for k, v in model.state_dict().items() if not k.startswith("judge_head")}
    plain.load_state_dict(sd)
    plain.train(), model.train()
    x, y = x1, y1
    yj = y.masked_fill(jm[:, 1:] == 0, -1)
    lp, _ = plain.losses(x, yj)
    l_no, _ = model.losses(x, yj)
    model.feed_judge(jm[:, :-1])
    l_j, st = model.losses(x, yj)
    model.judge_report()
    jl, sums = model.judge_loss(model._trunk(x), jm[:, :-1])
    check("without marks the model's loss is the plain body's; with marks it is next word + 10 x the judge's loss",
          abs(float(lp.detach()) - float(l_no.detach())) < 1e-5 and abs(float(l_j.detach()) - (float(l_no.detach()) + 10.0 * float(jl.detach()))) < 1e-4,
          f"{float(lp):.5f} {float(l_no):.5f} {float(l_j):.5f} judge {float(jl):.5f}")
    ramp = Gnosis2(Gnosis2Config(**SMALL, vocab_size=V, judge=10.0, judge_ramp=0.1))
    ramp.load_state_dict(model.state_dict())
    ramp.train()
    got, j1 = [], float(jl.detach())
    for pr, share in ((0.0, 0.0), (0.05, 0.5), (0.5, 1.0)):           # (the next-word loss has its own early extras: same progress)
        ramp.set_progress(pr)
        plain.set_progress(pr)
        ramp.feed_judge(jm[:, :-1])
        got.append(float(ramp.losses(x, yj)[0].detach()) - float(plain.losses(x, yj)[0].detach()) - 10.0 * share * j1)
    plain.set_progress(1.0)
    check("judge_ramp brings the judge's weight in: none at the start, half of it halfway up the ramp, all of it after",
          max(abs(g) for g in got) < 1e-4, f"off by {max(abs(g) for g in got):.1e}")
    l_full, _ = plain.losses(x, y)
    check("the next-word loss is not asked to write a decoy (its pieces are left out)", abs(float(l_full) - float(lp)) > 1e-6
          and model._judge is None, f"{float(l_full):.5f} with them, {float(lp):.5f} without")
    l_j.backward()                                                             # the whole lesson has a way back
    model.zero_grad()
    (10.0 * model.judge_loss(model._trunk(x), jm[:, :-1])[0]).backward()       # the judge's loss alone
    gh = float(model.judge_head.weight.grad.abs().sum())
    gb = sum(float(p.grad.abs().sum()) for n, p in model.named_parameters() if n.startswith("blocks.0.") and p.grad is not None)
    check("the judge's loss reaches the head and the body", gh > 0 and gb > 0, f"{gh:.2e} {gb:.2e}")
    z = model.judge_scores(model._trunk(x))
    graded = jm[:, :-1] >= 0
    ref = torch.nn.functional.binary_cross_entropy_with_logits(z[graded], (jm[:, :-1][graded] > 0).float())
    check("the judge's loss is the yes/no loss on the graded pieces only", abs(float(ref) - float(jl)) < 1e-5,
          f"{float(ref):.5f} {float(jl):.5f}; decoys {float(sums[5] / sums[1]):.1%} of the graded pieces")

    # ---------------------------------------------------------------- it learns, and the tests can be scored by it
    torch.manual_seed(0)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.0)
    m1.judge = Decoys("smollm2", rate=0.3)
    m1.set_batch(16)
    first = None
    for step in range(1000):                               # (the judge starts to learn once the body knows the words)
        x, y = m1.get(dev)
        jm = m1.last_judge
        model.feed_judge(jm[:, :-1])
        loss, st = model.losses(x, y.masked_fill(jm[:, 1:] == 0, -1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in (99, 899):                              # the log's numbers: the first 100 steps, then the last 100
            now = model.judge_report()
            first = first or now
    last = model.judge_report()
    check("the judge learns: its yes on decoys falls well under its yes on real text",
          last["judge_yes_real"] - last["judge_yes_decoy"] > 0.2 and last["judge"] < first["judge"] - 0.2,
          f"after 1,000 steps: yes on real {last['judge_yes_real']:.2f}, on decoys {last['judge_yes_decoy']:.2f}; "
          f"loss {first['judge']:.3f} -> {last['judge']:.3f}")
    x, y = m1.get(dev)
    model.feed_judge(m1.last_judge[:, :-1])
    model.losses(x, y)
    rep = model.thought_report()
    check("the run's log gets the judge's numbers, summed since it last read them",
          all(k in rep for k in ("judge", "judge_yes_real", "judge_yes_decoy", "judge_decoy_share")) and model._jr is None
          and "judge_decoys_per_1k" in m1.judge.report())
    qs = []
    for _ in range(200):                                    # the right ending: the document's own next sentence; the wrong
        d = doc(tp, rng, 8)                                 # ones: later sentences of the same document (on topic, out of turn)
        others = [d[5], d[6], d[7]]
        lab = rng.randrange(4)
        ends = others[:lab] + [d[3]] + others[lab:]
        qs.append({"ctx": " ".join(d[:3]), "endings": ends, "label": lab})
    path = tmp / "hs.jsonl"
    path.write_text("".join(json.dumps(q) + "\n" for q in qs))
    r = hellaswag.evaluate(model, path, dev)
    check("the choice tests are also scored by the judge, and a trained judge picks the ending that belongs",
          "acc_judge" in r and "acc_judge_mean" in r and r["acc_judge"] > 0.5,
          f"by the judge {r.get('acc_judge', 0):.0%} (mean over the ending {r.get('acc_judge_mean', 0):.0%}), "
          f"by next-word loss {r['acc_norm']:.0%}; guessing 25%")
    r0 = hellaswag.evaluate(plain, path, dev)
    check("a model without a judge head is scored as before", "acc_judge" not in r0 and "acc_norm" in r0)
    (tmp / "evals").mkdir()
    (tmp / "evals" / "piqa.jsonl").write_text(path.read_text())
    from oazy import evals
    e = evals.evaluate_all(model, tmp, dev)
    check("the other choice tests carry the judge's score too", "piqa_judge" in e and "piqa" in e, f"{sorted(e)}")


if __name__ == "__main__":
    main()
