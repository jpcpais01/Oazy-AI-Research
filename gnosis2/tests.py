"""Self-tests for Gnosis2 and Opal2: the Canon layers (start as nothing, so Gnosis2 starts as Gnosis exactly; stay
causal; word by word = all at once; learn in Adam's group), the size; the logic warm-up (every line's answer right);
Field1 worked out (unchanged Field1, the walls, every question grades its own worked answer as right, the grader); the
Opal2 mix; one whole tiny run on Opal2 end to end; and the reasoning lesson on its checkpoint.

    python3 -m gnosis2.tests                 (CPU, ~2 min; without internet it uses stand-in tokenizers)
    python3 -m gnosis2.tests --device cuda   (on the pod: the same, plus a compiled bf16 step on the GPU)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from oazy import term

SMALL = dict(d_model=64, n_layer=6, attn_heads=4, attn_kv_heads=2, attn_head_dim=16, mlp_hidden=128, max_len=128,
             ctx="128:0", vocab_size=512, key_offset="1,5", mod_layers="2,4", fp8=0)
FIELD1_PRINT = "6c0f29bfd9ba6ec2"          # 3,000 Field1 episodes before Opal2 (the notes must not change one byte)


def check(name, ok, detail=""):
    term.check(name, ok, detail)


def jiggle(m, s=0.05, only=None):
    with torch.no_grad():
        for n, p in m.named_parameters():
            if only is None or only(n):
                p.add_(torch.randn_like(p) * s)
    return m


def model_tests(dev):
    from bee6.model import Bee6, Bee6Config
    from gnosis2.model import Gnosis2, Gnosis2Config
    torch.manual_seed(0)
    g6 = jiggle(Bee6(Bee6Config(**SMALL))).to(dev).eval()
    g7 = Gnosis2(Gnosis2Config(**SMALL)).to(dev).eval()
    missing = g7.load_state_dict(g6.state_dict(), strict=False)
    x = torch.randint(1, 512, (2, 96), device=dev)
    with torch.no_grad():
        a, b = g6(x)[0], g7(x)[0]
    check("Canon layers start at zero: Gnosis2 with Gnosis's weights gives Gnosis's scores exactly",
          torch.equal(a, b) and all(".canon_" in k for k in missing.missing_keys),
          f"{len(missing.missing_keys)} Canon weights new, largest gap {(a - b).abs().max().item():.1e}")
    n_canon = sum(1 for k in missing.missing_keys)
    routed = len(Gnosis2Config(**SMALL).mod_set())
    check("Canon-A and Canon-C in every layer that is not routed (2 per layer), none in the routed ones",
          n_canon == 2 * (SMALL["n_layer"] - routed), f"{n_canon} Canon layers")
    jiggle(g7, 0.3, lambda n: ".canon_" in n)
    with torch.no_grad():
        z = g7(x)[0]
        x2 = x.clone()
        x2[:, 60:] = torch.randint(1, 512, (2, 36), device=dev)
        z2 = g7(x2)[0]
    check("still causal: changing words from 60 on leaves every score before 60 unchanged",
          torch.allclose(z[:, :60], z2[:, :60], atol=1e-5) and not torch.allclose(z[:, 60:], z2[:, 60:]),
          f"gap before {(z[:, :60] - z2[:, :60]).abs().max().item():.1e}")
    check("the Canon layers matter once learned", not torch.allclose(z, b, atol=1e-3))
    with torch.no_grad():
        full, _ = g7.read(x[:1, :40])
        st = None
        outs = []
        for i in range(0, 40, 7):
            lg, st = g7.read(x[:1, i:min(i + 7, 40)], st)
            outs.append(lg)
        step = torch.cat(outs, 1)
    check("reading on in pieces (chat, samples, the reasoning lesson) = reading all at once",
          torch.allclose(full, step, atol=1e-4), f"largest gap {(full - step).abs().max().item():.1e}")
    with torch.no_grad():
        lg_b, _ = g7.read(torch.stack([x[0, :30], x[1, :30]]))
        lg_0, _ = g7.read(x[:1, :30])
    check("it reads a batch on as it reads one", torch.allclose(lg_b[:1], lg_0, atol=1e-4))
    from bee5.optim import groups
    mu, ad = groups(g7)
    names = {id(p): n for n, p in g7.named_parameters()}
    check("Canon weights learn in Adam's gates group, never in Muon's",
          all(".canon_" not in n for n, _ in mu) and sum(".canon_" in names[id(p)] for p in ad["gates"]) == n_canon)
    g7.train()
    g7.set_progress(0.0)
    loss, stats = g7.losses(x, x)
    loss.backward()
    gc = [p.grad for n, p in g7.named_parameters() if ".canon_" in n]
    check("a training step: finite loss, every Canon weight gets a gradient",
          torch.isfinite(loss).item() and all(g is not None and g.abs().sum() > 0 for g in gc), f"loss {loss.item():.3f}")
    from oazy.train import build
    big, cfg = build("gnosis2", "big")
    gn, _ = build("gnosis", "big")
    nb, ng = big.num_params(), gn.num_params()
    check("full size: Gnosis2 is Gnosis + 98k Canon weights (307.5M + 0.1M)",
          0 < nb - ng < 2e5 and big.cfg.n_layer == 22 and cfg.d_model == 1024,
          f"Gnosis2 {nb / 1e6:.2f}M, Gnosis {ng / 1e6:.2f}M; {big.layout()}")
    del big, gn


def data_tests():
    from oazy import proc, field1 as F
    from oazy.field1 import work, guard
    from oazy.mix import OPAL2_MIX, parse_mix, with_cooldown, WORLD_SOURCES
    # the logic warm-up: every answer right
    r = random.Random(0)
    bad = 0
    for _ in range(400):
        s = proc.sort(r).split(": ", 1)[1].split(" => ")
        xs = s[0].split()
        bad += sorted(xs, key=int if xs[0].isdigit() else str) != s[1].split()
        a = proc.add(r).split(": ", 1)[1]
        l, rr = a.split(" => ")
        u, v = l.split(" + ")
        bad += int(u.replace(" ", "")) + int(v.replace(" ", "")) != int(rr.replace(" ", ""))
        c = proc.close(r).split(": ", 1)[1]
        pre, need = c.split(" => ")
        whole = pre.split() + ([] if need == "-" else need.split())
        st, ok = [], True
        pairs = dict(proc.BRACKETS)
        for t in whole:
            if t in pairs:
                st.append(pairs[t])
            elif not st or st.pop() != t:
                ok = False
        bad += not (ok and not st)
    d1, d2 = proc.document((0, 1)), proc.document((0, 1))
    check("the logic warm-up: 1,200 sorts, sums and bracket closings all right; documents repeat from their seed",
          bad == 0 and d1 == d2 and len(d1.splitlines()) >= 6, f"{bad} wrong; e.g. {d1.splitlines()[0][:70]!r}")
    # Field1 itself unchanged
    h = hashlib.sha256()
    for i in range(3000):
        f = F.FAMILIES[i % len(F.FAMILIES)]
        h.update(F.episode(f, (0, i))[0].encode())
    check("Field1 is byte for byte what it was (the state notes are said only in the worked part)",
          h.hexdigest()[:16] == FIELD1_PRINT, h.hexdigest()[:16])
    # worked episodes and questions
    leaks, wrong, notes = {}, 0, 0
    for i in range(760):
        f = F.FAMILIES[i % len(F.FAMILIES)]
        t, _ = work.episode(f, ("t", i))
        q = work.question(f, ("t", i))
        for x in guard.leaks(t) + guard.leaks(q["prompt"] + q["work"]):
            leaks[x] = leaks.get(x, 0) + 1
        wrong += work.grade(q["work"], q["answer"]) < 1
        notes += f in ("lines", "count", "stacks") and (" now" in q["work"] or "pile" in q["work"])
    check("Field1 worked out: 760 episodes and questions, none says anything of Real World, Pro or LogT; every "
          "question's own work grades as right", not leaks and wrong == 0, f"{wrong} wrong, leaks {leaks}")
    check("the families that change step by step say what is true after each event", notes > 100, f"{notes} of 120")
    q = work.question("lines", ("t", 3))
    check("a question ends where the work starts, so a model that continues text starts to think",
          q["prompt"].rstrip().endswith(tuple(work.HEAD)) and "answer is" in q["work"].lower())
    g = work.grade
    check("the grader: right answers in other words count, wrong or missing ones do not",
          g("so the answer is The Red Box.", "the red box") == 1 and g("The answer is three.", "3") == 1
          and g("So the answer is 3rd.", "third") == 1 and g("So the answer is yes. Then no", "yes") == 1
          and g("So the answer is Ana.", "Bo") == 0 and g("I am not sure.", "Bo") == 0
          and g("the answer is Bo, then the answer is Ana.", "Ana") == 1)
    plan = parse_mix(with_cooldown(OPAL2_MIX, 0.7))
    sums = [abs(sum(w.values()) - 1) < 1e-4 for _, w in plan]
    check("the Opal2 mix: the warm-up alone first, a pinch to 10%, think 12% / work 8% / Field1 7%, religion 2% exact, "
          "cool-down at 70%", all(sums) and plan[0][1] == {"proc": 1.0} and plan[1][1]["proc"] == 0.005
          and "proc" not in plan[2][1] and plan[-1][0] == 0.7 and abs(plan[-1][1]["think"] - 0.12) < 1e-9
          and abs(sum(v for k, v in plan[2][1].items() if k in ("religion", "faith", "creed")) - 0.0201) < 1e-9
          and "work" in WORLD_SOURCES, " | ".join(f"@{a:g}" for a, _ in plan))


def run_tests(dev):
    from bee2.tests import stand_ins, fake_docs
    from oazy import train, tok, opal, opal2
    tmp_ctx = tempfile.TemporaryDirectory()
    tmp = Path(tmp_ctx.name)
    used = stand_ins(tmp)
    if used:
        term.info(f"no internet: stand-in {' and '.join(used)} tokenizer(s)")
    enc = tok.get("smollm2")
    rng = random.Random(0)
    data = tmp / "data"
    root = data                    # Opal lives in <data>/opal, as on the pod
    op = root / "opal"
    op.mkdir(parents=True, exist_ok=True)
    for split, k in (("val", 300), ("train", 1500)):
        np.fromiter((t for d in fake_docs(k, rng) for t in enc.encode_ordinary(d) + [enc.eot_token]),
                    dtype=np.uint16).tofile(op / f"{split}.bin")
    (data / "hellaswag_val.jsonl").write_text("".join(json.dumps({"ctx": " ".join(fake_docs(1, rng)),
                                              "endings": fake_docs(4, rng), "label": rng.randint(0, 3)}) + "\n" for _ in range(16)))
    opal.build(root, tokens=60_000, val_tokens=8_000, workers=1)
    opal2.PARTS["proc"].update(tokens=20_000, val=4_000)
    opal2.PARTS["work"].update(tokens=60_000, val=8_000)
    opal2.build(root, workers=1, say=lambda s: None)
    check("Opal2's parts build into Opal's folder (proc, field1work), with their held-out pieces",
          all((op / p / f).exists() for p in ("proc", "field1work") for f in ("train.bin", "val.bin", "meta.json")))
    runs = tmp / "runs"
    argv = ["train", "--model", "gnosis2", "--preset", "smoke", "--data", str(data), "--runs", str(runs), "--mentor", "none",
            "--device", str(dev), "--compile", "0", "--tokens", "60000", "--batch-ramp", "512:0.5", "--micro", "2",
            "--data-sub", "opal2", "--set", "ctx=64:0,128:0.3", "attn_head_dim=16", "attn_heads=4", "attn_kv_heads=2", "fp8=0"]
    sys.argv = argv
    train.main()
    rd = runs / "gnosis2"
    st = json.loads((rd / "status.json").read_text())
    recs = [json.loads(x) for x in (rd / "metrics.jsonl").read_text().splitlines()]
    cfgj = json.loads((rd / "config.json").read_text()) if (rd / "config.json").exists() else {}
    mix = str(cfgj.get("mix", cfgj.get("args", {}).get("mix", "")))
    check("a whole tiny run of Gnosis2 on Opal2 finishes with held-out loss, reading the warm-up, Field1 and its "
          "worked part", st["state"] == "done" and "val_loss" in st["latest"] and "proc:1@0" in json.dumps(cfgj)
          and "work:" in json.dumps(cfgj), f"val loss {st['latest'].get('val_loss', float('nan')):.3f}")
    check("the world exams run on Opal2 (its world rows count as world)", any(r.get("kind") == "world" for r in recs)
          or "world" in json.dumps(st), "")
    mdl, ck = train.load_model(rd / "ckpt.pt", dev)
    check("its checkpoint loads as Gnosis2", ck["kind"] == "gnosis2" and type(mdl).__name__ == "Gnosis2")
    out, _ = mdl.generate(enc.encode_ordinary("The cat"), max_new_tokens=5)
    check("the trained model writes", len(out) <= 5, repr(enc.decode(out)))
    from gnosis2 import rl
    acc0, acc = rl.main(["--run", "gnosis2", "--runs", str(runs), "--steps", "3", "--prompts", "2", "--group", "3",
                         "--max-new", "12", "--val-per", "1", "--every", "3", "--device", str(dev), "--micro", "4"])
    rr = runs / "gnosis2-reason"
    rec = json.loads((rr / "reason.json").read_text())
    m2, c2 = train.load_model(rr / "ckpt.pt", dev)
    lines = [json.loads(x) for x in (rr / "rl.jsonl").read_text().splitlines()]
    check("the reasoning lesson runs on it: questions, groups of answers, grades, a step, held-out scores, a checkpoint "
          "that loads as Gnosis2", type(m2).__name__ == "Gnosis2" and rec["steps"] == 3 and 0 <= acc <= 1
          and sum(1 for x in lines if x["kind"] == "step") == 3 and len(rec["families"]) == 19,
          f"held-out {acc0:.0%} -> {acc:.0%} (a random tiny model: zero is expected)")
    tmp_ctx.cleanup()


def rl_update_test(dev):
    """The lesson's learning step itself (a tiny random model never gets an answer right, so the grades are faked
    here: half right, half wrong): advantages, the policy and KL terms, a step that moves the weights."""
    from bee2.tests import stand_ins
    from gnosis2.model import Gnosis2, Gnosis2Config
    from gnosis2 import rl
    from oazy.field1 import work
    import os
    tmp_ctx = tempfile.TemporaryDirectory()
    tmp = Path(tmp_ctx.name)
    env = {k: os.environ.get(k) for k in ("OAZY_SMOL_TOKENIZER", "OAZY_GPT2_TIKTOKEN")}
    stand_ins(tmp)
    cfg = Gnosis2Config(**{**SMALL, "vocab_size": 49152})
    m = jiggle(Gnosis2(cfg))
    (tmp / "runs" / "tiny").mkdir(parents=True)
    torch.save({"model": m.state_dict(), "opt": {}, "kind": "gnosis2", "config": cfg.to_dict()}, tmp / "runs" / "tiny" / "ckpt.pt")
    real, calls = work.grade, [0]

    def fake(text, answer):
        calls[0] += 1
        return float(calls[0] % 2)
    work.grade = fake
    try:
        rl.main(["--run", "tiny", "--runs", str(tmp / "runs"), "--steps", "2", "--prompts", "2", "--group", "4",
                 "--max-new", "8", "--val-per", "1", "--every", "2", "--device", str(dev), "--micro", "4", "--lr", "1e-3"])
    finally:
        work.grade = real
    lines = [json.loads(x) for x in (tmp / "runs" / "tiny-reason" / "rl.jsonl").read_text().splitlines()]
    steps = [x for x in lines if x["kind"] == "step"]
    after = torch.load(tmp / "runs" / "tiny-reason" / "ckpt.pt", weights_only=False)["model"]
    moved = max((after[k].float() - v.float()).abs().max().item() for k, v in m.state_dict().items() if v.is_floating_point())
    check("the lesson's learning step: mixed grades give advantages, a KL term, a gradient and weights that move",
          all(x["useful"] > 0 and "kl" in x and x["gnorm"] > 0 for x in steps) and moved > 0,
          f"useful {steps[0]['useful']:.0%}, KL {steps[-1]['kl']:.2e}, largest weight change {moved:.1e}")
    tmp_ctx.cleanup()
    for k, v in env.items():                # the stand-ins lived in this test's folder: forget them
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    from oazy import tok
    for f in vars(tok).values():
        if hasattr(f, "cache_clear"):
            f.cache_clear()


def gpu_tests(dev):
    from gnosis2.model import Gnosis2, Gnosis2Config
    m = jiggle(Gnosis2(Gnosis2Config(**{**SMALL, "vocab_size": 49152, "fp8": 1}))).to(dev).train()
    m.startup_check(dev, lambda s: term.info(s))
    m.set_progress(0.0)
    x = torch.randint(1, 49152, (4, 128), device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = m.losses(x, x)
    loss.backward()
    m.zero_grad(set_to_none=True)
    m.enable_compile()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lc, _ = m.losses(x, x)
    lc.backward()
    check("the compiled bf16 / FP8 training step on the GPU runs and agrees", torch.isfinite(lc).item()
          and abs(lc.item() - loss.item()) < 0.05, f"{lc.item():.4f} vs {loss.item():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    term.out(term.c("━━ Gnosis2 + Opal2 self-tests", "bold"))
    model_tests(dev)
    rl_update_test(dev)
    data_tests()
    run_tests(dev)
    if dev.type == "cuda":
        gpu_tests(dev)
    term.tests_passed("Gnosis2 + Opal2")


if __name__ == "__main__":
    main()
