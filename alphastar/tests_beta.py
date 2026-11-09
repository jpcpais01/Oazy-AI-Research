"""Self-tests for Pegasus Beta (alphastar/beta.py): the stop chances, its start at the prior, causality with its own
stop, a word that is done keeps its thought, the lesson (next word + the stop head's + the next thought), where the
gradients go, the depth it practises, that it learns, one whole tiny run; on a GPU the compiled step and the speed.

    python3 -m alphastar.tests_beta                 (CPU, a few minutes)
    python3 -m alphastar.tests_beta --device cuda   (on the pod: also the compiled step and the speed race)
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

from bee3.tests import check
from oazy import term
from alphastar.tests import SMALL

BSMALL = dict(SMALL, ponder_max=4, ponder_prior=2.5, ponder_rows=2, halt_hidden=32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-race", action="store_true")
    a = ap.parse_args()
    dev = torch.device(a.device)
    term.tests_start("Pegasus Beta", dev)
    from bee2.tests import stand_ins
    tmp_ctx = tempfile.TemporaryDirectory()
    tmp = Path(tmp_ctx.name)
    if stand_ins(tmp):
        term.info("no internet: stand-in tokenizers")
    from alphastar.beta import PegasusBeta, BetaConfig, geometric, stop_chances
    from alphastar.model import AlphaStar, AlphaStarConfig
    from oazy import tok
    enc = tok.get("smollm2")

    full = PegasusBeta(BetaConfig())
    na, nb_ = AlphaStar(AlphaStarConfig()).num_params(), full.num_params()
    check("Alpha Star's size + a small stop head", 0 < nb_ - na < 0.5e6, f"{nb_ / 1e6:.2f}M (Alpha Star {na / 1e6:.2f}M)")
    del full

    g = geometric(0.25, 8)
    lam = torch.rand(5, 8)
    p = stop_chances(lam)
    check("stop chances: each word's chances over the passes add up to 1 (the last pass takes what is left)",
          torch.allclose(p.sum(-1), torch.ones(5)) and abs(float(g.sum()) - 1) < 1e-6 and torch.allclose(p[:, 0], lam[:, 0]))

    torch.manual_seed(0)
    m = PegasusBeta(BetaConfig(**BSMALL)).to(dev)
    V = m.cfg.vocab_size
    text = ("The cat sat on the mat. It was warm! Was the garden happy? Yes. The dog ran to the old red barn by the river. "
            "A very long sentence without any stop that keeps going and going on and on past sixteen pieces for sure here "
            "and there and everywhere around and then some more words to make it long enough for every tokenizer we use")
    ids = enc.encode_ordinary(text)
    counts = np.zeros(V, dtype=np.int64)
    np.add.at(counts, np.asarray(ids * 50, dtype=np.int64), 1)
    m.prepare_foresight(enc, counts)
    with torch.no_grad():
        l0 = float(m.lam(torch.randn(3, m.cfg.d_model, device=dev)).mean())
    check("it starts at the prior: a 1-in-2.5 chance to stop at each pass, its habits = the prior",
          abs(l0 - 1 / 2.5) < 0.01 and torch.allclose(m.habit.cpu(), geometric(1 / 2.5, 4)))

    # ---- reading with its own stop
    m.eval()
    gq = torch.Generator().manual_seed(0)
    x = torch.randint(1, V, (2, 48), generator=gq).to(dev)
    x2 = x.clone()
    x2[:, 30:] = torch.randint(1, V, (2, 18), generator=gq).to(dev)
    with torch.no_grad():
        a1, a2 = m._body(x), m._body(x2)
        err = (a1[:, :30] - a2[:, :30]).abs().max().item()
        check("causal with its own stop: changing later words moves nothing earlier", err < 1e-5, f"max change {err:.1e}")
        e = m._read(x)
        b = m.halt[2].bias.clone()
        m.halt[2].bias.fill_(20.0)
        _, u1 = m.think_test(e)
        m.halt[2].bias.fill_(-20.0)
        _, u2 = m.think_test(e)
        m.halt[2].bias.copy_(b)
        seen = []
        _, um = m.think_test(e, every=lambda n, s: seen.append(s.clone()))
        kept = all(torch.equal(seen[i][um < i + 1], seen[i - 1][um < i + 1]) for i in range(1, len(seen)))
    check("it stops after pass 1 when sure, thinks the full 8 when never sure", float(u1.max()) == 1 and float(u2.min()) == 8)
    check("each word stops at its median (chance of having stopped >= 1/2); a word that is done keeps its thought",
          kept, f"at the start: {float(um.mean()):.2f} passes on average (the prior's median is 2)")

    # ---- the lesson
    m.train()
    xb = torch.tensor([ids[:64]] * 4, device=dev)
    T = xb.shape[1] - 1
    m.cfg.train_noise = "table"
    m.force_passes = 3
    torch.manual_seed(3)
    loss, st = m.losses(xb[:, :-1], xb[:, 1:])
    m.set_progress(1.0)
    check("the lesson: next word (at the practised depth) + the stop head's + 0.1 x the next thought",
          "ponder" in st and "fore" in st and abs(loss.item() - (st["ce"].item() + st["ponder"].item())) < 0.1 * st["fore"].item() + 1e-3
          and 1 <= float(st["passes"]) <= 4,
          f"next word {st['ce'].item():.3f}, stop head {st['ponder'].item():.3f}, next thought {st['fore'].item():.3f}, "
          f"expected passes {float(st['passes']):.2f}")
    m.eval()
    m.cfg.test_passes = 3
    with torch.no_grad():
        _, lv, _ = m(xb[:, :-1], xb[:, 1:])
    m.cfg.test_passes = 0
    m.train()
    check("the next word is learned at the practised depth (3 passes here = reading with 3)", abs(lv.item() - st["ce"].item()) < 1e-4,
          f"{st['ce'].item():.4f} vs {lv.item():.4f}")
    h0 = m.habit.clone()
    loss.backward()
    gr = {n: (p_.grad.abs().sum().item() if p_.grad is not None else 0.0) for n, p_ in m.named_parameters()}
    check("gradients reach the stop head, the core, the reader, the drawer and the speaker",
          all(gr[k] > 0 for k in ("halt.2.weight", "halt.0.weight", "core.0.attn.q.weight", "reader.0.attn.q.weight",
                                  "reader.1.drawer.values.weight", "speaker.1.mlp.up.weight")))
    m.zero_grad(set_to_none=True)
    m.force_passes = None
    with torch.no_grad():
        m.habit.copy_(torch.tensor([0.0, 0.0, 0.0, 1.0]))
    d = [m.sample_passes() for _ in range(4000)]
    pri = geometric(1 / 2.5, 4).numpy()
    want = 0.5 * np.array([0, 0, 0, 1.0]) + 0.5 * pri
    got = np.bincount(d, minlength=5)[1:] / len(d)
    check("the depth it practises: half from its own habits, half from the prior", np.abs(got - want).max() < 0.03,
          " ".join(f"{v:.2f}" for v in got) + " vs " + " ".join(f"{v:.2f}" for v in want))
    with torch.no_grad():
        m.habit.copy_(h0)

    # ---- it learns
    torch.manual_seed(1)
    m.cfg.train_noise = "random"
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for _ in range(80):
        l_, st_ = m.losses(xb[:, :-1], xb[:, 1:])
        first = first if first is not None else st_["ce"].item()
        l_.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    r = m.loop_report(np.asarray(xb.cpu()), dev, batch=2, curve=(1, 2, 4, 8))
    check("it learns: the loss falls; the probe shows where it chooses to stop and the filler / content split",
          r["adaptive_loss"] < first - 1.0 and abs(sum(r["halt_hist"]) - 1) < 1e-4 and "passes_content" in r,
          f"{first:.2f} -> {r['adaptive_loss']:.2f} after {r['adaptive_passes']:.1f} passes (filler {r['passes_filler']:.1f}, "
          f"content {r['passes_content']:.1f}) | stops " + " ".join(f"{v * 100:.0f}%" for v in r["halt_hist"][:6]) +
          f" | habits now " + " ".join(f"{v:.2f}" for v in m.habit.tolist()))
    del opt

    from oazy.train import RECIPE, MENTOR, BEES, FAST, PRESETS
    pb = PRESETS["big"]
    check("Plain-100M's recipe, mentor, steps and learning rates", RECIPE["pegasusbeta"] == RECIPE["plain100"]
          and MENTOR["pegasusbeta"] == MENTOR["plain100"] and "pegasusbeta" in BEES and "pegasusbeta" in FAST
          and pb["batch_pegasusbeta"] == pb["batch_plain100"] and pb["lr_pegasusbeta"] == pb["lr_plain100"])

    # ---- one whole tiny run
    from bee2.tests import fake_docs
    from oazy.data import get_tokenizer, EOT
    from oazy import retok, train
    gpt = get_tokenizer()
    rng = random.Random(0)
    data = tmp / "data"
    for sub, nd_ in (("", 1500), ("chat", 300)):
        dd = data / sub if sub else data
        dd.mkdir(parents=True, exist_ok=True)
        for split, kk in (("val", nd_ // 5), ("train", nd_)):
            ds = fake_docs(kk, rng)
            if sub == "chat":
                ds = [f"User: {u_}\n\nAssistant: {v_}" for u_, v_ in zip(ds, fake_docs(kk, rng))]
            np.fromiter((t_ for x_ in ds for t_ in gpt.encode_ordinary(x_) + [EOT]), dtype=np.uint16).tofile(dd / f"{split}.bin")
    (data / "hellaswag_val.jsonl").write_text("".join(
        json.dumps({"ctx": " ".join(fake_docs(1, rng)), "endings": fake_docs(4, rng), "label": rng.randint(0, 3)}) + "\n" for _ in range(24)))
    retok.run(data, "smollm2", ["fwe", "chat"], workers=2)
    runs = tmp / "runs"
    argv = ["train", "--model", "pegasusbeta", "--preset", "smoke", "--data", str(data), "--runs", str(runs), "--mentor", "none",
            "--device", str(dev), "--compile", "0", "--tokens", "40000", "--batch-ramp", "512:0.5", "--mix", "fwe:0.9,chat:0.1@0",
            "--micro", "2", "--data-sub", "smol"]
    sys.argv = argv
    train.main()
    rd = runs / "pegasusbeta"
    recs = [json.loads(l_) for l_ in (rd / "metrics.jsonl").read_text().splitlines()]
    tr = [r_ for r_ in recs if r_["kind"] == "train"]
    stt = json.loads((rd / "status.json").read_text())
    loops = [r_ for r_ in recs if r_["kind"] == "loop"]
    check("a whole tiny Pegasus Beta run (passes on every line, where it stops in the loop probe, HellaSwag)",
          stt["state"] == "done" and loops and "halt_hist" in loops[-1] and all("passes" in r_ for r_ in tr)
          and any(r_["kind"] == "hellaswag" for r_ in recs), f"val loss {stt['latest']['val_loss']:.3f}")
    mdl, ck = train.load_model(rd / "ckpt.pt", dev)
    check("its checkpoint loads as Pegasus Beta, its habits included", ck["kind"] == "pegasusbeta" and type(mdl).__name__ == "PegasusBeta"
          and "habit" in ck["model"])
    sys.argv = argv
    train.main()
    del mdl

    if dev.type == "cuda":
        mg = PegasusBeta(BetaConfig(**{**BSMALL, "max_len": 2048, "ctx": "2048:0", "fore_max": 8192, "train_noise": "table"})).to(dev).train()
        mg.prepare_foresight(enc, np.ones(V, dtype=np.int64))
        mg.force_passes = 3
        xg = torch.randint(1, V, (4, 1024), device=dev)
        torch.manual_seed(0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            le, _ = mg.losses(xg, xg)
        le.backward()
        mg.zero_grad(set_to_none=True)
        mg.enable_compile()
        torch.manual_seed(0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lc, _ = mg.losses(xg, xg)
        lc.backward()
        check("the compiled step matches", torch.isfinite(lc).item() and abs(lc.item() - le.item()) < 0.05, f"{lc.item():.4f} vs {le.item():.4f}")
        del mg
        if not a.no_race:
            from pegasus.baseline import Plain100, Plain100Config
            xb = torch.randint(1, V, (32, 2048), device=dev)
            speed, mem = {}, {}
            for name, mk in (("Plain-100M", lambda: Plain100(Plain100Config())), ("Pegasus Beta", lambda: PegasusBeta(BetaConfig()))):
                torch._dynamo.reset()
                torch.manual_seed(0)
                mf = mk().to(dev).train()
                if hasattr(mf, "prepare_foresight"):
                    mf.prepare_foresight(enc, np.arange(V)[::-1].copy())
                mf.enable_compile()
                torch.cuda.reset_peak_memory_stats()

                def lesson():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lf, _ = mf.losses(xb, xb)
                    lf.backward()
                    mf.zero_grad(set_to_none=True)
                for _ in range(3):
                    lesson()
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(10):
                    lesson()
                torch.cuda.synchronize()
                speed[name] = 10 * xb.numel() / (time.time() - t0)
                mem[name] = torch.cuda.max_memory_allocated() / 2 ** 30
                del mf
                torch.cuda.empty_cache()
            term.info("speed (compiled bf16, 32 x 2,048, no mentor): " + ", ".join(
                f"{k} {v / 1e3:.0f}k tok/s ({mem[k]:.0f} GB)" for k, v in speed.items())
                + f" | Beta at {speed['Pegasus Beta'] / speed['Plain-100M']:.2f}x Plain's speed")
            check("speed race at full size ran", speed["Pegasus Beta"] > 0)
    tmp_ctx.cleanup()
    term.tests_passed("Pegasus Beta")


if __name__ == "__main__":
    main()
