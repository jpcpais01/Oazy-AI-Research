"""Self-tests for Think-135 (gnosis2/think.py, gnosis2/thinkab.py).

    python3 -m gnosis2.think_tests                 (CPU, about a minute)
    python3 -m gnosis2.think_tests --device cuda   (on the pod: plus a compiled bf16 step at race size and its speed)
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

from oazy import term
from gnosis2.loop_tests import SMALL, jiggle

TH = dict(thought=1, thought_at=2, thought_dim=8, thought_width=32, thought_cap=16, thought_every=8)
FAILS = []


def check(name, ok, detail=""):
    term.check(name, ok, detail)
    if not ok:
        FAILS.append(name)


def model_tests(dev):
    from gnosis2.model import Gnosis2, Gnosis2Config
    from gnosis2 import think as THINK
    torch.manual_seed(0)
    plain = Gnosis2(Gnosis2Config(**SMALL)).to(dev)
    m = Gnosis2(Gnosis2Config(**SMALL, **TH)).to(dev)
    m.load_state_dict(plain.state_dict(), strict=False)
    x = torch.randint(0, 512, (2, 64), device=dev)
    plain.eval(), m.eval()
    with THINK.reading(m, "oracle"):
        d = float((m._body(x) - plain._body(x)).abs().max())
    check("it starts as Bee-135 exactly (the thought's way into the stream starts at zero)", d < 1e-5, f"{d:.1e}")
    new = sum(p.numel() for n, p in m.named_parameters() if n.startswith("mind."))
    check("only the mind is new", sum(p.numel() for p in m.parameters()) - sum(p.numel() for p in plain.parameters()) == new,
          f"{new:,} weights")
    jiggle(m)
    a = m._body(x)
    y = x.clone()
    y[:, 40] = (y[:, 40] + 1) % 512
    b = m._body(y)
    check("no thoughts (the default reading) is a plain next-word model: nothing before a changed word moves",
          float((a[:, :40] - b[:, :40]).abs().max()) == 0 and float((a[:, 40:] - b[:, 40:]).abs().max()) > 0)
    with THINK.reading(m, "oracle"):
        torch.manual_seed(5)
        a = m._body(x)
        _, ex = m._cap
        torch.manual_seed(5)
        b = m._body(y)
    ok, inside = True, 0.0
    for r in range(2):
        end = ex["end"][r]
        s = max(0, int((end == end[40]).nonzero()[0]) - 1)     # the first place that reads word 40's sentence's thought
        ok &= float((a[r, :s] - b[r, :s]).abs().max()) == 0 if s else True
        if s < 40:
            inside = max(inside, float((a[r, s:40] - b[r, s:40]).abs().max()))
    check("with the true thoughts a word reads its OWN sentence's future, never a later sentence's", ok and inside > 0,
          f"earlier sentences move 0, earlier words of the same sentence {inside:.3f}")
    lens = (ex["end"][:, 1:] != ex["end"][:, :-1]).sum(1).float().mean()
    check("sentences are cut (16 pieces at most here)", int((ex["b"].float().sum(1) >= 4).all()), f"{float(lens) + 1:.0f} a row")
    h, _ = m._cap
    z, s = m.mind.think(THINK.context(m, h, ex)[:, 10], 6)
    check("the mind makes thoughts of size 1, another one from other noise",
          z.shape == (2, 6, 8) and abs(float(z.pow(2).mean()) - 1) < 1e-3 and float((z[:, 0] - z[:, 1]).abs().mean()) > 1e-3)
    zz = torch.randn(2, 64, 8, device=dev)
    own = torch.zeros(2, 64, dtype=torch.bool, device=dev)
    own[:, 30:40] = True
    torch.manual_seed(5)
    with THINK.reading(m, "oracle"):
        a = m._body(x)
    with THINK.reading(m, "oracle", (zz, own)):
        torch.manual_seed(5)
        c = m._body(x)
    check("a given thought changes the words that read it and none before",
          float((c[:, :30] - a[:, :30]).abs().max()) == 0 and float((c[:, 30:40] - a[:, 30:40]).abs().max()) > 0)
    m.train()
    loss, st = m.losses(x[:, :-1], x[:, 1:])
    loss.backward()
    none = [n for n, p in m.named_parameters() if n.startswith("mind.") and (p.grad is None or float(p.grad.abs().sum()) == 0)]
    check("the lessons reach every part of the mind and the body", not none and all(k in st for k in ("thought_err", "thought_sure")),
          f"no gradient: {none}" if none else f"thought error {float(st['thought_err']):.2f} (guessing nothing: {float(st['thought_blur']):.2f})")
    r = THINK.report(m, torch.randint(0, 512, (4, 65)).numpy(), dev, 2)
    check("held-out loss three ways (no thoughts, one made thought, the true thoughts)", len(r["pass_loss"]) == 3)
    # a toy: every sentence is one random word said 7 times. Only its thought can tell the first one
    torch.manual_seed(0)
    cfg = Gnosis2Config(**{**SMALL, "n_layer": 4, "key_offset": "1"}, **{**TH, "thought_at": 1, "thought_every": 4})
    t = Gnosis2(cfg).to(dev).train()
    opt = torch.optim.Adam(t.parameters(), lr=3e-3)

    def batch(n=32):
        w = (torch.randint(0, 60, (n, 8, 1), device=dev) * 7 + 1).expand(n, 8, 8).clone()   # one word a sentence, said 7 times
        w[:, :, -1] = 7                                                               # piece 7 closes a sentence here
        return w.reshape(n, 64)
    for _ in range(150 if dev.type == "cpu" else 300):
        xb = batch()
        loss, st = t.losses(xb[:, :-1], xb[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    t.eval()
    r = THINK.report(t, batch(16).cpu().numpy(), dev, 8)["pass_loss"]
    check("a toy learns to use the thought: the true thoughts give a lower loss than none", r[9] < r[0] - 0.05,
          f"none {r[0]:.3f}, a made thought {r[1]:.3f}, the true thoughts {r[9]:.3f}")


def gpu_checks(dev):
    from gnosis2.model import Gnosis2, Gnosis2Config
    from gnosis2.thinkab import ARMS
    from bee5 import optim as O
    torch.set_float32_matmul_precision("high")
    sp = {}
    for arm in ("plain", "think"):
        torch.manual_seed(0)
        m = Gnosis2(Gnosis2Config(**ARMS[arm])).to(dev).train()
        m.startup_check(dev, say=lambda *a, **k: None)
        m.enable_compile()
        opt = O.make(m, 1e-3, dev)
        x = torch.randint(0, 49152, (16, 2049), device=dev)
        ts = []
        for i in range(14):
            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, st = m.losses(x[:, :-1], x[:, 1:])
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            ts.append(time.time() - t0)
        sp[arm] = 16 * 2048 / (sum(ts[6:]) / len(ts[6:]))
        check(f"{arm}: a compiled bf16 training step at race size", bool(torch.isfinite(loss)), f"{sp[arm] / 1e3:.0f}k words/s")
        if arm == "think":
            m.eval()
            for L in (40, 77, 130):
                m._body(x[:2, :L])
            m.train()
            t0 = time.time()
            for i in range(4):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, st = m.losses(x[:, :-1], x[:, 1:])
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            after = 16 * 2048 * 4 / (time.time() - t0)
            check("training is as fast after tests of many lengths", after > 0.8 * sp[arm], f"{after / 1e3:.0f}k words/s")
        del m, opt
        torch.cuda.empty_cache()
    check("Think-135 trains at 0.9x Bee-135's speed or better", sp["think"] >= 0.9 * sp["plain"], f"{sp['think'] / sp['plain']:.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    term.header("Think-135: self-tests")
    model_tests(dev)
    if dev.type == "cuda":
        gpu_checks(dev)

    # ---- a tiny run through the trainer, then scored by meaning
    from bee2.tests import stand_ins, fake_docs
    from oazy import tok
    tmp_ctx = tempfile.TemporaryDirectory()
    tmp = Path(tmp_ctx.name)
    stand_ins(tmp)
    enc = tok.get("smollm2")
    data = tmp / "data"
    rng = random.Random(0)
    sado = data / "sado"
    sado.mkdir(parents=True)
    for split, k_ in (("val", 300), ("train", 1500)):
        np.fromiter((t for d_ in fake_docs(k_, rng) for t in enc.encode_ordinary(d_) + [enc.eot_token]),
                    dtype=np.uint16).tofile(sado / f"{split}.bin")
    (data / "hellaswag_val.jsonl").write_text("".join(
        json.dumps({"ctx": " ".join(fake_docs(1, rng)), "endings": fake_docs(4, rng), "label": rng.randint(0, 3)}) + "\n"
        for _ in range(24)))
    (data / "evals").mkdir()
    (data / "evals" / "winogrande.jsonl").write_text("".join(
        json.dumps({"ctxs": fake_docs(2, rng), "endings": [fake_docs(1, rng)[0]] * 2, "label": rng.randint(0, 1)}) + "\n"
        for _ in range(12)))
    from oazy import train
    from gnosis2 import thinkab
    runs = tmp / "runs"
    name = thinkab.RUNS["think"]
    sys.argv = ["train", "--model", "gnosis2", "--preset", "smoke", "--data", str(data), "--runs", str(runs), "--mentor",
                "none", "--device", "cpu", "--compile", "0", "--tokens", "40000", "--batch-ramp", "512:0.5", "--micro", "2",
                "--set", "ctx=64:0,128:0.3", "attn_head_dim=16", "attn_heads=4", "attn_kv_heads=2", "fp8=0", "mod_layers=",
                "key_offset=1", "thought=1", "thought_at=1", "thought_dim=8", "thought_width=32", "thought_cap=16",
                "--name", name, "--data-sub", "sado"]
    train.main()
    rd = runs / name
    stw = json.loads((rd / "status.json").read_text())
    recs = [json.loads(l_) for l_ in (rd / "metrics.jsonl").read_text().splitlines()]
    loop = [r for r in recs if r["kind"] == "loop"]
    tr = [r for r in recs if r["kind"] == "train" and r.get("thought_err") is not None]
    check("a tiny run finishes; the thought's numbers and the three held-out losses are logged",
          stw["state"] == "done" and loop and tr, f"thought error {tr[-1]['thought_err']:.2f}" if tr else "")
    out = thinkab.score(rd, data, torch.device("cpu"), say=lambda *a, **k: None)
    check("its checkpoint loads and is scored by meaning (1, 4, 16, 64 thoughts; the surest; words under thoughts)",
          set(out) == {"hs", "winogrande"} and all(len(r["meaning"]) == 4 and len(r["sure"]) == 4 for r in out.values()),
          f"HellaSwag stand-in: meaning 64 {out['hs']['meaning']['64'] * 100:.0f}%, words {out['hs']['words'] * 100:.0f}%")
    thinkab.main(["--runs", str(runs), "--data", str(data)])
    tmp_ctx.cleanup()
    if FAILS:
        raise SystemExit(f"{len(FAILS)} check(s) failed: {FAILS}")
    term.tests_passed("Think-135")


if __name__ == "__main__":
    main()
