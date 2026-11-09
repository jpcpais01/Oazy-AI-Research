"""Self-tests for world practice in a next-word model (oazy/worldmix.py): the share over the run, the episodes as plain
text, the stream (practice levels only, helpers = in-process, resumable), the exam (a model that knows the answers gets
them all, one that does not gets about chance), and a whole tiny Bee6 run with --world, end to end, with a resume, plus
the race table.

    python3 -m oazy.worldmix_tests        (CPU, ~2 min; without internet it uses stand-in tokenizers)
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from oazy import term


def check(name, ok, detail=""):
    term.check(name, ok, detail)


class Oracle(torch.nn.Module):
    """A stand-in 'model' that knows every exam episode by heart: after a run of pieces it has seen, it is sure of the
    next one; anywhere else it knows nothing. It should get every exam question right."""

    def __init__(self, seqs, vocab):
        super().__init__()
        self.next = {}
        for s in seqs:
            for t in range(len(s) - 1):
                self.next[tuple(s[:t + 1])] = s[t + 1]
        self.vocab = vocab
        self.cfg = type("C", (), {"tokenizer": "smollm2"})()

    def _body(self, x):
        out = torch.zeros(x.shape[0], x.shape[1], self.vocab)
        for b in range(x.shape[0]):
            row = x[b].tolist()
            for t in range(len(row)):
                n = self.next.get(tuple(row[:t + 1]))
                if n is not None:
                    out[b, t, n] = 30.0
        return out


class Blank(Oracle):
    def __init__(self, vocab):
        super().__init__([], vocab)


def main():
    term.tests_start("world practice", "cpu")
    from bee2.tests import stand_ins, fake_docs
    from worldfirst import world as W
    from oazy import worldmix as M, tok
    tmp_ctx = tempfile.TemporaryDirectory()
    tmp = Path(tmp_ctx.name)
    used = stand_ins(tmp)
    if used:
        term.info(f"no internet: stand-in {' and '.join(used)} tokenizer(s)")
    enc = tok.get("smollm2")

    # ---- the share over the run
    ps = np.linspace(0, 1, 200_001)
    avg10 = float(np.mean([M.share_at(p, 0.10) for p in ps]))
    avg5 = float(np.mean([M.share_at(p, 0.05) for p in ps]))
    check("--world 0.10: 30% of the passes for the first 20%, then 5%; 10% overall (0.05: 15% then 2.5%; 0 = off)",
          abs(M.share_at(0.1, 0.10) - 0.3) < 1e-12 and abs(M.share_at(0.9, 0.10) - 0.05) < 1e-12 and abs(avg10 - 0.10) < 1e-4
          and abs(M.share_at(0.0, 0.05) - 0.15) < 1e-12 and abs(avg5 - 0.05) < 1e-4 and M.share_at(0.5, 0.0) == 0.0,
          f"{avg10:.4f} / {avg5:.4f}")

    # ---- an episode as plain text
    e = W.make("lines", 2, "train", 5)
    text, spans = M.episode_parts(e)
    words = [text[a:b].strip() for a, b, *_ in spans]
    flat = [w for q in e.questions for w in q.answer]
    check("an episode reads as plain sentences, its answers in place", text[0].isupper() and " ." not in text
          and "Answer: " in text and words == flat and all(text[a] == " " for a, *_ in spans), text[:110] + " ...")
    ids = M.encode_episodes(enc, [e])[0]
    check("cut at the answers, its pieces are the same text", enc.decode(ids) == text)
    ok_c = True
    for fam in W.FAMILIES:
        for s_ in range(30):
            ee = W.make(fam, W.TRAIN_LEVELS[fam][-1], "train", s_)
            for *_, kind, w, cands, qi in M.episode_parts(ee)[1]:
                ok_c &= w in cands and len(cands) >= 2 and len(set(cands)) == len(cands)
    check("every answer word is among its candidates (yes/no, a digit, a name or thing of the scene)", ok_c)

    # ---- the stream: practice levels only, the same with helpers, resumable
    seen = {(f, lv) for k in range(40) for f, lv in M.picks(0, k)}
    secret = {(f, lv) for f in W.FAMILIES for sp, lvs in W.SECRET[f].items() for lv in lvs}
    allowed = {(f, lv) for f in W.FAMILIES for lv in W.TRAIN_LEVELS[f]}
    check("the episodes come from practised levels only, never a secret exam's, every family and level in turn",
          seen <= allowed and seen == allowed and all(f in {x for x, _ in seen} for f in W.FAMILIES),
          f"{len(seen)} (family, level) pairs; secret pairs never used: {len(secret - seen) == len(secret - allowed)}")
    s0 = M.WorldStream("smollm2", seed=0, workers=0)
    a0 = s0.rows(4, 64)
    a1 = s0.rows(4, 64)
    s2 = M.WorldStream("smollm2", seed=0, workers=2)
    b0 = s2.rows(4, 64)
    b1 = s2.rows(4, 64)
    s2.close()
    check("two helper processes make the very same windows as the trainer alone", np.array_equal(a0, b0) and np.array_equal(a1, b1))
    eot = enc.eot_token
    n_eot = int((np.concatenate([a0, a1]) == eot).sum())
    check("episodes back to back, <|endoftext|> between them", n_eot >= 2, f"{n_eot} ends in 8 windows of 64")
    st = s0.state()
    s3 = M.WorldStream("smollm2", seed=0, workers=0, state=st)
    c0 = s3.rows(4, 64)
    j0 = M.job((0, 0, "smollm2"))
    check("a resumed stream carries on with new episodes (the next job), and a job is always the same episodes",
          st["job"] >= 1 and not np.array_equal(c0, a0) and np.array_equal(j0, M.job((0, 0, "smollm2")))
          and not np.array_equal(M.job((0, 1, "smollm2"))[:50], j0[:50]))
    x, y = s0.get(3, 32, torch.device("cpu"))
    check("get() gives (x, y) like the text reader: y is x moved by one", x.shape == (3, 32) and torch.equal(x[:, 1:], y[:, :-1]))

    # ---- the exam: a model that knows the answers gets them all; a blank one about chance
    sets = [("lines", 7, "longer"), ("sums", 9, "longer"), ("family", 4, "deeper"), ("minds", 4, "more-moves"),
            ("walks", 5, "longer"), ("lines", 1, "train")]
    seqs = []
    for s in sets:
        for i in range(3):
            seqs.append(M._items(enc, W.make(s[0], s[1], s[2], i))[0])
    V = enc.n_vocab
    r = M.exam(Oracle(seqs, V), torch.device("cpu"), n=3, sets=sets)
    check("a model that knows every answer scores 100% on secret and practised exams", r["secret"] == 1.0 and r["seen"] == 1.0,
          f"secret {r['secret']:.2f}, practised {r['seen']:.2f}")
    rb = M.exam(Blank(V), torch.device("cpu"), n=3, sets=sets)
    check("one that knows nothing scores low (ties go to the first candidate)", rb["secret"] < 0.6, f"secret {rb['secret']:.2f}")
    ch = M.chance(sets, n=3)
    check("chance on the same exams is worked out for the table", 0 < ch["secret"] < 0.5, f"{ch['secret'] * 100:.1f}%")

    # ---- a whole tiny Bee6 run with --world, then a resume, then the race table
    from oazy.data import get_tokenizer, EOT
    from oazy import retok, train
    gpt = get_tokenizer()
    rng = random.Random(0)
    data = tmp / "data"
    for sub, nd in (("", 1500), ("chat", 300)):
        d = data / sub if sub else data
        d.mkdir(parents=True, exist_ok=True)
        for split, kk in (("val", nd // 5), ("train", nd)):
            ds = fake_docs(kk, rng)
            if sub == "chat":
                ds = [f"User: {u}\n\nAssistant: {v_}" for u, v_ in zip(ds, fake_docs(kk, rng))]
            np.fromiter((t for x_ in ds for t in gpt.encode_ordinary(x_) + [EOT]), dtype=np.uint16).tofile(d / f"{split}.bin")
    (data / "hellaswag_val.jsonl").write_text("".join(
        json.dumps({"ctx": " ".join(fake_docs(1, rng)), "endings": fake_docs(4, rng), "label": rng.randint(0, 3)}) + "\n" for _ in range(24)))
    retok.run(data, "smollm2", ["fwe", "chat"], workers=2)
    runs = tmp / "runs"
    base = ["train", "--model", "bee6", "--preset", "smoke", "--data", str(data), "--runs", str(runs), "--mentor", "smol-random",
            "--mentor-share", "1", "--mentor-weight", "0.5", "--mentor-until", "1.0",
            "--device", "cpu", "--compile", "0", "--tokens", "60000", "--batch-ramp", "512:0.5", "--mix", "fwe:0.9,chat:0.1@0",
            "--micro", "2", "--data-sub", "smol", "--set", "ctx=64:0,128:0.3", "attn_head_dim=16", "attn_heads=4", "attn_kv_heads=2",
            "fp8=0"]
    sys.argv = base + ["--name", "plain"]
    train.main()
    argv = base + ["--name", "world", "--world", "0.10", "--world-workers", "1"]
    sys.argv = argv
    train.main()
    rd = runs / "world"
    recs = [json.loads(l_) for l_ in (rd / "metrics.jsonl").read_text().splitlines()]
    tr = [x_ for x_ in recs if x_["kind"] == "train"]
    stw = json.loads((rd / "status.json").read_text())
    cf = json.loads((rd / "config.json").read_text())
    wt = tr[-1]["world_tokens"]
    share = wt / stw["tokens"]
    check("a tiny Bee6 run with --world 0.10 finishes, and about 10% of its tokens were world episodes",
          stw["state"] == "done" and cf["world"] == 0.10 and 0.06 < share < 0.16, f"{wt:,} of {stw['tokens']:,} tokens ({share:.1%})")
    early = [x_["world_share"] for x_ in tr if x_["tokens"] < 0.2 * stw["tokens"]]
    late = [x_["world_share"] for x_ in tr if x_["tokens"] > 0.3 * stw["tokens"]]
    check("30% early, 5% after, and the world's loss logged on its own (the training loss stays the text's)",
          early and late and all(abs(v - 0.3) < 1e-9 for v in early) and all(abs(v - 0.05) < 1e-9 for v in late)
          and any("world_loss_own" in x_ for x_ in tr))
    wx = [x_ for x_ in recs if x_["kind"] == "world"]
    ex = json.loads((rd / "world_exam.json").read_text())
    check("the world's exams run during training (secret only) and in full at the end, into world_exam.json",
          len(wx) >= 2 and not wx[0]["full"] and wx[-1]["full"] and "secret" in ex and "seen" in ex and ex["rows"],
          f"secret {ex['secret'] * 100:.0f}%, practised {ex['seen'] * 100:.0f}% (a tiny model: noise)")
    ck = torch.load(rd / "ckpt.pt", weights_only=False)
    check("the checkpoint keeps the world's tally, its token count and the next job", ck["world"]["tokens"] == wt
          and ck["world"]["stream"]["job"] >= 1)
    n_before = len(recs)
    sys.argv = argv
    train.main()
    recs2 = [json.loads(l_) for l_ in (rd / "metrics.jsonl").read_text().splitlines()]
    check("run again, it picks up its checkpoint instead of starting over", len(recs2) == n_before
          and json.loads((rd / "status.json").read_text())["state"] == "done")
    sys.argv = base + ["--name", "bad", "--world", "0.1", "--select", "1"]
    try:
        train.main()
        refused = False
    except SystemExit as e_:
        refused = "--select" in str(e_)
    check("--world refuses --select (and bloom): the race measures the world alone", refused)

    from oazy.worldmix import main as wm
    wm(["exam", "--runs", str(runs), "--run", "plain", "world", "--n", "2", "--device", "cpu"])
    check("the exam on a run folder: the plain run gets one too; the world run's own is kept",
          (runs / "plain" / "world_exam.json").exists())
    from bee6 import world_race
    got = world_race.main(["--runs", str(runs), "--arms", "plain", "world"])
    md = (runs / "bee6-world-race.md").read_text()
    check("the race table: both runs side by side, a verdict, the secret exams family by family",
          set(got) == {"plain", "world"} and "Verdict" in md and "family by family" in md and "\x1b" not in md)
    tmp_ctx.cleanup()
    term.tests_passed("world practice")


if __name__ == "__main__":
    main()
