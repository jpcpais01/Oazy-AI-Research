"""Self-tests for choosing what to learn from (oazy/select.py): the weighed loss in slices against the slow way, the
weights on filler / noise / worth-learning words, the mentor's table and the fallback, rereading, a tiny Plain-100M
run with it on (and resumed), and the cost.

    python3 -m oazy.select_tests                 (CPU, ~1 min; without internet it uses stand-in tokenizers)
    python3 -m oazy.select_tests --device cuda   (on the pod: also the compiled slices and the speed on the GPU)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bee3.tests import check
from oazy import term


def slow_weighted(h, W, targets, ref, thr, floor, wscale, th=None, tW=None, kd_w=0.0):
    """The weighed loss the slow way (full 32-bit scores): what oazy/lmloss.py must reproduce."""
    z = F.linear(h, W).float()
    lp = torch.log_softmax(z, -1)
    tok = -lp.gather(1, targets[:, None])[:, 0]
    if th is not None:
        tlp = torch.log_softmax(F.linear(th, tW).float(), -1)
        r = -tlp.gather(1, targets[:, None])[:, 0]
    else:
        r = ref
    w = torch.where(tok.detach() - r > thr, torch.ones_like(tok), torch.full_like(tok, floor)) * wscale
    loss = (1 - kd_w) * (w * tok).mean()
    if th is not None and kd_w > 0:
        loss = loss + kd_w * (tlp.exp() * (tlp - lp)).sum(-1).mean()
    return loss, tok.detach()


class Fixed:
    """A stand-in Select with fixed numbers, to compare the slices with the slow way."""

    def __init__(self, ref, thr, floor, wscale):
        self.r, self.t, self.f, self.s = ref, torch.tensor(thr), torch.tensor(floor), torch.tensor(wscale)
        self.seen = None

    def before(self, targets, exact=False):
        return self.r, self.t, self.f, self.s

    def after(self, targets, tok, ttok, ref):
        self.seen = (tok, ttok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    dev = torch.device(ap.parse_args().device)
    term.tests_start("Choosing what to learn from", dev)
    torch.manual_seed(0)
    from oazy.lmloss import head_loss, plain_loss
    from oazy.select import Select, Replay, DEFAULTS

    # ---- 1. the weighed loss in slices = the slow way (loss and gradients)
    N, D, V = 300, 16, 97
    for mentor in (False, True):
        h = torch.randn(N, D, device=dev, requires_grad=True)
        W = torch.randn(V, D, device=dev, requires_grad=True)
        t = torch.randint(0, V, (N,), device=dev)
        ref = torch.rand(N, device=dev) * 4
        th = torch.randn(N, D, device=dev) if mentor else None
        tW = torch.randn(V, D, device=dev) if mentor else None
        kd_w = 0.3 if mentor else 0.0
        fx = Fixed(ref, 0.5, 0.25, 1.3)
        loss, ce, _ = head_loss(h, W, t, 0.0, th, tW, kd_w, chunk=64, select=fx)
        loss.backward()
        gh, gW = h.grad.clone(), W.grad.clone()
        h.grad = W.grad = None
        sl, tok = slow_weighted(h, W, t, ref, 0.5, 0.25, 1.3, th, tW, kd_w)
        sl.backward()
        ok = (abs(float(loss) - float(sl)) < 1e-4 and torch.allclose(gh, h.grad, atol=1e-5) and torch.allclose(gW, W.grad, atol=1e-5)
              and abs(float(ce) - float(tok.mean())) < 1e-4 and torch.allclose(fx.seen[0], tok, atol=1e-4))
        check("the weighed loss in slices = the slow way, loss and gradients" + (" (with the mentor's own scores)" if mentor else ""),
              ok, f"{float(loss):.5f} vs {float(sl):.5f}; the plain next-word loss still reported ({float(ce):.3f})")
    h = torch.randn(N, D, device=dev)
    W = torch.randn(V, D, device=dev)
    a = head_loss(h, W, t, 0.0, chunk=64)
    b = plain_loss(h, W, t)
    check("without it, the loss is exactly as before", abs(float(a[0]) - float(b[0])) < 1e-4)

    # ---- 2. which words count: worth learning 1, filler and noise the floor
    V = 50
    s = Select(V, dev, keep=0.5, floor=0.25, start=0.0, min_count=1)
    s.progress = 0.5
    tgt = torch.tensor([1, 2, 3, 4], device=dev)
    # word 1: the model 3.0, the mentor 0.2 (worth learning)   word 2: both ~0.1 (filler)
    # word 3: both ~6 (noise)                                   word 4: the model 4.0, the mentor 1.0 (worth learning)
    mine = torch.tensor([3.0, 0.1, 6.0, 4.0], device=dev)
    ment = torch.tensor([0.2, 0.15, 6.2, 1.0], device=dev)
    for _ in range(30):                                   # the mentor reads: its table fills, the cut settles
        r, *_ = s.before(tgt, exact=True)
        s.after(tgt, mine, ment, r)
    for _ in range(30):                                   # then passes without it: its table stands in
        r, *_ = s.before(tgt)
        s.after(tgt, mine, None, r)
    r, thr, fl, sc = s.before(tgt)
    w = torch.where(mine - r > thr, torch.ones_like(mine), fl.expand(4))
    check("worth learning (the mentor knows it, the model not yet) counts fully; filler and noise count 0.25",
          w.tolist() == [1.0, 0.25, 0.25, 1.0], f"excess {[round(v, 2) for v in (mine - r).tolist()]}, cut {float(thr):.2f}")
    check("on passes without the mentor, its average on that word piece stands in", torch.allclose(r, ment, atol=1e-5))
    s2 = Select(V, dev, keep=0.5, floor=0.25, start=0.0)
    for _ in range(5):
        r2, *_ = s2.before(tgt)
        s2.after(tgt, mine, None, r2)
    r2, *_ = s2.before(tgt)
    check("without any mentor, the model's own history on each word piece is the reference", torch.allclose(r2, mine, atol=1e-4))
    s3 = Select(V, dev, start=0.5)
    s3.progress = 0.1
    r3, thr3, _, sc3 = s3.before(tgt)
    check("before `start` (2% of the run) every word counts the same", float(thr3) == -float("inf") and float(sc3) == 1.0)
    big = Select(1000, dev, keep=0.7, floor=0.25, start=0.0)
    big.progress = 1.0
    for _ in range(40):
        tt = torch.randint(0, 1000, (4096,), device=dev)
        lo = torch.rand(4096, device=dev) * 5
        rr, *_ = big.before(tt)
        big.after(tt, lo, torch.rand(4096, device=dev) * 3, rr)
    rep = big.report()
    check("about 70% of the words count fully; the weights average 1 after scaling",
          0.66 < rep["select_kept"] < 0.74 and abs(float(big.mean_w) - (0.25 + 0.75 * rep["select_kept"])) < 0.05,
          f"{rep['select_kept']:.1%} kept, mean weight {rep['select_w']:.3f}")

    drift = Select(1000, dev, keep=0.7, floor=0.25, start=0.0)
    drift.progress = 1.0
    gen = torch.Generator(device="cpu").manual_seed(1)
    for i in range(400):                                  # a real run: the loss falls slowly, each pass's text differs
        tt = torch.randint(0, 1000, (4096,), generator=gen).to(dev)
        lo = (torch.rand(4096, generator=gen) * 5 + 3.0 - 0.002 * i + 0.2 * torch.randn((), generator=gen)).to(dev)
        rr, *_ = drift.before(tt)
        drift.after(tt, lo, None, rr)
        if i == 199:
            drift.report()
    rep = drift.report()
    check("with the loss falling and the text changing from pass to pass, still about 70%", 0.62 < rep["select_kept"] < 0.78,
          f"{rep['select_kept']:.1%} kept")

    # ---- 3. rereading
    T, rows = 8, 20
    rp = Replay(T, dev, share=0.15, reads=2, size=16)
    total_rows = total_re = 0
    for i in range(40):
        x = torch.full((rows, T), i, device=dev, dtype=torch.long)
        x[:, 0] = torch.arange(rows, device=dev) + 100 * i
        y = x + 1
        x, y, rr = rp.mix_in(x, y)
        sc = torch.where(x[:, 0] % 2 == 0, torch.full((rows,), 5.0, device=dev), torch.zeros(rows, device=dev))   # even = hard
        rp.offer(x, y, sc, rr)
        total_rows += rows
        total_re += int((rr > 0).sum())
        assert int(rr.max()) <= 2
    check("15% of the rows are rereads; never more than 2 rereads per passage", abs(total_re / total_rows - 0.15) < 0.02,
          f"{total_re / total_rows:.1%} of {total_rows} rows")
    check("the buffer keeps only the passages closest to clicking (the hard ones)",
          bool(((rp.x[:, 0] % 2 == 0) | ~torch.isfinite(rp.score)).all()))
    rp2 = Replay(T, dev, share=0.5, reads=3, size=8)
    x = torch.arange(4 * T, device=dev).view(4, T)
    rp2.offer(x, x + 1, torch.tensor([9.0, 8.0, 7.0, 6.0], device=dev), torch.zeros(4, dtype=torch.long, device=dev))
    x2, _, rr2 = rp2.mix_in(torch.zeros(4, T, dtype=torch.long, device=dev), torch.zeros(4, T, dtype=torch.long, device=dev))
    rp2.offer(x2, x2, torch.tensor([0.0, 0.0, -1.0, -1.0], device=dev), rr2)     # they clicked: low scores now
    check("a reread passage that has clicked drops behind the ones still waiting", int(rr2[:2].min()) == 1
          and float(rp2.score.max()) == 7.0)
    rp.reset(16)
    check("when the windows grow, the buffer starts again", rp.T == 16 and not torch.isfinite(rp.score).any())

    # ---- 4. a whole tiny Plain-100M run with it on, then resumed
    from bee2.tests import stand_ins, fake_docs
    from oazy.data import get_tokenizer, EOT
    from oazy import retok, train
    tmp_ctx = tempfile.TemporaryDirectory()
    tmp = Path(tmp_ctx.name)
    used_ = stand_ins(tmp)
    if used_:
        term.info(f"no internet: stand-in {' and '.join(used_)} tokenizer(s)")
    gpt = get_tokenizer()
    rng = random.Random(0)
    data = tmp / "data"
    for sub, nd in (("", 1500), ("chat", 300)):
        d = data / sub if sub else data
        d.mkdir(parents=True, exist_ok=True)
        for split, kk in (("val", nd // 5), ("train", nd)):
            ds = fake_docs(kk, rng)
            if sub == "chat":
                ds = [f"User: {u_}\n\nAssistant: {v_}" for u_, v_ in zip(ds, fake_docs(kk, rng))]
            np.fromiter((t_ for x_ in ds for t_ in gpt.encode_ordinary(x_) + [EOT]), dtype=np.uint16).tofile(d / f"{split}.bin")
    (data / "hellaswag_val.jsonl").write_text("".join(
        json.dumps({"ctx": " ".join(fake_docs(1, rng)), "endings": fake_docs(4, rng), "label": rng.randint(0, 3)}) + "\n" for _ in range(24)))
    retok.run(data, "smollm2", ["fwe", "chat"], workers=2)
    runs = tmp / "runs"
    base = ["train", "--model", "plain100", "--preset", "smoke", "--data", str(data), "--runs", str(runs), "--mentor", "smol-random",
            "--device", str(dev), "--compile", "0", "--batch-ramp", "512:0.5", "--mix", "fwe:0.9,chat:0.1@0",
            "--micro", "4", "--data-sub", "smol"]
    sys.argv = base + ["--tokens", "40000", "--select", "1", "--name", "sel"]
    train.main()
    rd = runs / "sel"
    recs = [json.loads(l_) for l_ in (rd / "metrics.jsonl").read_text().splitlines()]
    tr = [r for r in recs if r["kind"] == "train"]
    cfg_run = json.loads((rd / "config.json").read_text())
    st = json.loads((rd / "status.json").read_text())
    on = [r for r in tr if r.get("select_on")]
    check("a whole tiny Plain-100M run with it on: it says so, and logs the picks and the rereads",
          st["state"] == "done" and cfg_run.get("select") is True and cfg_run["select_keep"] == DEFAULTS["select_keep"]
          and on and all("replay_share" in r for r in on),
          f"val loss {st['latest']['val_loss']:.3f}, {on[-1]['select_kept']:.0%} of words counted fully, "
          f"{np.mean([r['replay_share'] for r in on]):.0%} rereads")
    check("the mentor's own scores are used on its passes (1 in 4 early, 1 in 10 later)",
          any(r["select_exact"] > 0 for r in tr) and all(r["select_exact"] < 0.5 for r in tr),
          f"{np.mean([r['select_exact'] for r in tr]):.0%} of passes exact")
    ck = torch.load(rd / "ckpt.pt", map_location="cpu", weights_only=False)
    check("the checkpoint keeps the tables and the buffer", "select" in ck and float(ck["select"]["m_cnt"].sum()) > 0
          and ck["replay"] is not None)
    s4 = Select(49152, dev)
    s4.load_state_dict(ck["select"])
    rp4 = Replay(ck["replay"]["T"], dev, size=ck["replay"]["x"].shape[0])
    rp4.load_state_dict(ck["replay"])
    check("they load back exactly", torch.equal(s4.m_sum.cpu(), ck["select"]["m_sum"]) and torch.equal(rp4.score.cpu(), ck["replay"]["score"]))
    # stopped half way (Ctrl-c at the second check), then the same command again: it resumes with them
    import os
    import signal
    real_vl, calls = train.val_loss, {"n": 0}

    def vl_then_stop(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3:                               # (two held-out sets per check)
            os.kill(os.getpid(), signal.SIGINT)
        return real_vl(*a, **k)
    train.val_loss = vl_then_stop
    sys.argv = base + ["--tokens", "40000", "--select", "1", "--name", "sel2"]
    train.main()
    train.val_loss = real_vl
    half = json.loads((runs / "sel2" / "status.json").read_text())
    ck2 = torch.load(runs / "sel2" / "ckpt.pt", map_location="cpu", weights_only=False)
    m_before = float(ck2["select"]["m_cnt"].sum())
    train.main()
    tr2 = [json.loads(l_) for l_ in (runs / "sel2" / "metrics.jsonl").read_text().splitlines()]
    tr2 = [r for r in tr2 if r["kind"] == "train" and r["tokens"] > half["tokens"]]
    ck3 = torch.load(runs / "sel2" / "ckpt.pt", map_location="cpu", weights_only=False)
    check("stopped half way and started again, it carries on with its tables", half["state"] == "stopped" and tr2
          and all("select_kept" in r for r in tr2) and float(ck3["select"]["m_cnt"].sum()) > m_before > 0,
          f"stopped at {half['tokens']:,} tokens; the mentor's table {m_before:,.0f} -> {float(ck3['select']['m_cnt'].sum()):,.0f} words")
    sys.argv = base + ["--tokens", "20000", "--name", "plain"]
    train.main()
    trp = [json.loads(l_) for l_ in (runs / "plain" / "metrics.jsonl").read_text().splitlines()]
    check("without --select nothing changes (off by default until the A/B)",
          not any("select_kept" in r for r in trp) and "select" not in json.loads((runs / "plain" / "config.json").read_text()))

    # ---- 5. the cost: the loss with and without it, on a Plain-100M-like head (words x 49,152)
    Nn, Dd, Vv = (8192, 768, 49152) if dev.type == "cuda" else (1024, 256, 49152)
    h = torch.randn(Nn, Dd, device=dev, requires_grad=True)
    W = torch.randn(Vv, Dd, device=dev, requires_grad=True)
    t = torch.randint(0, Vv, (Nn,), device=dev)
    sel = Select(Vv, dev, start=0.0)
    sel.progress = 1.0

    def timed(select):
        for _ in range(2):
            head_loss(h, W, t, 0.0, select=select)[0].backward()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5):
            head_loss(h, W, t, 0.0, select=select)[0].backward()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        return (time.time() - t0) / 5
    a, b = timed(None), timed(sel)
    check("it costs little: the word-table step with it vs without", b < a * 1.15,
          f"{a * 1e3:.1f} ms -> {b * 1e3:.1f} ms ({(b / a - 1) * 100:+.1f}%; the body of the model, not touched, is most of a step)")
    term.tests_passed("Choosing what to learn from")


if __name__ == "__main__":
    main()
