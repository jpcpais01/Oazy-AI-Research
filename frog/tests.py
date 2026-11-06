"""Self-tests for Frog: Bee3's body at 406M (only the sizes differ); Bee3's recipe plus the mentor on a share
of the sequences; that share's tally; a mentor pass learns only the mentor's guesses; one whole (tiny) run
end to end. Bee3's own tests (python3 -m bee3.tests) cover the document rule and the loss in slices,
which Frog shares unchanged.

    python -m frog.tests                 (CPU, ~2 min; without internet it uses stand-in tokenizers)
    python -m frog.tests --device cuda   (on the pod: the real tokenizer, the fast memory kernel, bf16)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from bee3.tests import check, SMALL
from oazy import term


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    dev = torch.device(ap.parse_args().device)
    term.tests_start("Frog", dev)
    torch.manual_seed(0)
    from frog.model import Frog, FrogConfig
    from bee3.model import Bee3Config
    from oazy import tok

    # ---- the body: Bee3's, wider and a little deeper
    big = Frog(FrogConfig())
    n = big.full_params()
    check("full size is about 0.4B", 395e6 < n < 415e6, f"{n / 1e6:.1f}M parameters once grown, {big.layout()}")
    check("21 memory layers + 7 attention layers (Bee3's 3:1)", big.layout() == "21 memory + 7 attention layers")
    diff = {k for k, v in Bee3Config().to_dict().items() if getattr(big.cfg, k) != v}
    check("same body as Bee3; only the sizes, the growing and the speed switches differ",
          diff == {"d_model", "n_layer", "n_head", "attn_heads", "mlp_hidden", "flex_attn", "fused_gates", "memory_op"}, str(diff))
    check("it starts 8 layers deep and grows to 28 (16 at 15%, 28 at 35%), pattern intact",
          big.depth == 8 and big.depth_at(0.149) == 8 and big.depth_at(0.15) == 16 and big.depth_at(0.35) == 28
          and abs(big.full_params() - n) < 1e5 and n - big.num_params() > 200e6, f"{big.num_params() / 1e6:.0f}M at the start")
    big.grow_to(1.0)
    check("fully grown: 28 layers, 3 memory : 1 attention all the way", "".join(b.kind for b in big.blocks) == "MMMA" * 7
          and abs(big.num_params() - n) < 1e5)
    from oazy import fp8
    picked = [f for f, mod in big.named_modules() if fp8.keep(mod, f)]
    check("FP8 goes to the big products only (projections, MLPs), not the word table or the small gates",
          len(picked) == 21 * 5 + 7 * 7 and not any(f.endswith((".trust", ".fade")) for f in picked), f"{len(picked)} layers")
    check("memories and heads fill the width", big.cfg.n_head * big.cfg.head_v == big.cfg.d_model
          and big.cfg.attn_heads * big.cfg.attn_head_dim == big.cfg.d_model)
    check("2,048-word windows, no bleed between documents, SmolLM2's pieces",
          big.ctx_at(0) == 2048 and big.cfg.doc_mask == 1 and big.cfg.tokenizer == "smollm2" and tok.eot_id("smollm2") == 0)
    del big

    # ---- the recipe: Bee3's, plus the mentor share and bigger steps
    from oazy.train import RECIPE, PRESETS, MENTOR, BEES, FAST, mentor_pass
    from ember.mentor import fade
    check("Bee3's recipe, plus the mentor on a share of the sequences and FP8",
          {k: v for k, v in RECIPE["frog"].items() if k not in ("mentor_share", "fp8")} == RECIPE["bee3"]
          and RECIPE["frog"]["mentor_share"] == 1 and RECIPE["frog"]["fp8"] == 1
          and "mentor_share" not in RECIPE["bee3"])
    pb = PRESETS["big"]
    check("big preset: 524k-token steps, learning rates 1.8e-3 / Muon 0.03 (Bee3: 262k, 2.4e-3 / 0.04)",
          pb["batch_frog"] == 524288 and pb["lr_frog"] == 1.8e-3 and pb["muon_lr_frog"] == 0.03 and pb["batch_bee3"] == 262144)
    check("SmolLM2-360M mentor, raised the Bee way", MENTOR["frog"] == "smollm2-360m" and "frog" in BEES and "frog" in FAST)

    # the tally: over any stretch the mentor gets a share w of the passes, evenly spread
    credit, picks = 0.0, []
    for _ in range(400):
        pw, credit = mentor_pass(credit, 0.25)
        picks.append(pw)
    check("at w = 0.25 exactly one pass in four goes to the mentor, every fourth",
          sum(picks) == 100 and all(sum(picks[i:i + 4]) == 1 for i in range(0, 400, 4)))
    credit, used, want = 0.0, 0.0, 0.0
    worst = 0.0
    for i in range(20000):                       # Bee3's whole schedule, one pass at a time
        w = fade(i / 20000, 0.6, 0.5, None, 0.25, 0.1)
        pw, credit = mentor_pass(credit, w)
        used, want = used + pw, want + w
        worst = max(worst, abs(used - want))
    check("over the whole schedule the mentor's passes track w (never more than one pass off)", worst <= 1.0 + 1e-6,
          f"mentor read {used / 20000:.1%} of the passes (average w {want / 20000:.1%})")

    # a mentor pass learns only the mentor's guesses; the others only the real next word
    from ember.mentor import Mentor
    me = Mentor("smol-random", dev, compile=False)
    m = Frog(FrogConfig(**SMALL)).to(dev).train()
    x = torch.randint(1, 49152, (2, 64), device=dev)
    x[:, 30] = 0
    y = torch.roll(x, -1, 1)
    l_m, st_m = m.losses(x, y, me, 1.0)
    l_s, st_s = m.losses(x, y, None, 0.0)
    check("a mentor pass: the lesson is the mentor's alone", abs(l_m.item() - float(st_m["kd"])) < 1e-4 and "ce" in st_m,
          f"loss {l_m.item():.4f} = mentor term {float(st_m['kd']):.4f}; next-word {float(st_m['ce']):.3f} still logged")
    check("any other pass: the real next word alone, no mentor", abs(l_s.item() - float(st_s["ce"])) < 1e-6 and "kd" not in st_s)
    l_mix, st_mix = m.losses(x, y, me, 0.25)
    mixed = 0.75 * float(st_mix["ce"]) + 0.25 * float(st_mix["kd"])
    check("on average the same lesson as Bee3's mix (0.75 x next word + 0.25 x mentor)",
          abs(0.25 * l_m.item() + 0.75 * l_s.item() - mixed) < 1e-3)

    # ---- the speed switches compute the same thing as the plain way
    from memtrix import memory
    from bee.model import doc_block_mask
    mem = m.blocks[0].mem
    xin = torch.randn(2, 64, SMALL["d_model"], device=dev)
    resets, mask, segs = m.doc_info(x)
    with torch.no_grad():
        q, k, v, tr, lf, _ = mem.project(xin, None, resets)
        o1, _ = memory.delta_rule(q, k, v, lf, tr.to(q.dtype), resets=resets)
        qr, kr, vr, tl, fr, _ = mem.project(xin, None, resets, raw=True)
        o2, _ = memory.delta_rule_raw(qr, kr, vr, fr, tl, mem.A_log, mem.dt_bias, resets=resets, segs=segs)
    e = (o1 - o2).abs().max().item()
    check("memory from the un-squashed inputs = the plain way (the fused-gates path's maths)", e < (5e-2 if dev.type == "cuda" else 1e-5),
          f"max diff {e:.1e}")
    att = m.blocks[3].attn
    with torch.no_grad():
        xa = torch.randn(2, 64, SMALL["d_model"], device=dev)
        y1, _ = att(xa, None, mask)
        y2, _ = att(xa, None, doc_block_mask(resets))
    e = (y1 - y2).abs().max().item()
    check("flex attention with the document block mask = the yes/no grid", e < (3e-2 if dev.type == "cuda" else 1e-4), f"max diff {e:.1e}")
    if dev.type == "cuda":
        from bee.model import flex_check
        term.info(f"memory kernel as a compiled operator: {'yes' if memory.op_ok(dev) else 'no'}; fused gates: "
              f"{'yes' if memory.fused_gates_ok(dev) else 'no'}; compiled flex attention: {'yes' if flex_check(dev, FrogConfig()) else 'no'}; "
              f"FP8: {'yes' if fp8.available(dev) else 'no'}")
    else:
        memory.use_reference_op()          # the operator's plumbing with plain-PyTorch pieces (the GPU checks the real ones)
    if memory._OP_OK:
        m2 = Frog(FrogConfig(**SMALL, vocab_size=320))
        xs = torch.randint(1, 320, (2, 96), device=dev)
        xs[:, 30] = 0
        xs[1, 70] = 0
        m2.to(dev)
        ys = torch.roll(xs, -1, 1)
        keep_op = memory._OP_OK
        memory._OP_OK = False
        l1, _ = m2.losses(xs, ys)
        g1 = torch.autograd.grad(l1, list(m2.parameters()))
        memory._OP_OK = keep_op
        l2, _ = m2.losses(xs, ys)
        g2 = torch.autograd.grad(l2, list(m2.parameters()))
        e = max(abs(l1.item() - l2.item()), *[(a - b).abs().max().item() for a, b in zip(g1, g2)])
        check("memory kernel as an operator = the usual call (loss and every gradient, documents packed)",
              e < (5e-2 if dev.type == "cuda" else 1e-5), f"max diff {e:.1e}")
        r_, mk, sg = m2.doc_info(xs)
        f = torch.compile(m2._trunk, fullgraph=True, backend="aot_eager" if dev.type == "cpu" else "inductor")
        hc = f(xs, r_, m2.attn_mask(xs, r_, mk), sg)
        hc.float().sum().backward()
        check("the whole stack compiles as ONE graph with the memory operator (no stop at any memory layer)", torch.isfinite(hc).all().item())

    # ---- growing: copies of the top groups on top, starting as working layers; checkpoints from any depth
    mg = Frog(FrogConfig(**{**SMALL, "grow": "4:0,8:0.5"}, vocab_size=320)).to(dev)
    grew = mg.grow_to(0.5)
    same = all((a - b).abs().max().item() == 0 for a, b in zip(mg.blocks[4].parameters(), mg.blocks[0].parameters()))
    check("at 50% it grows 4 -> 8 layers by copying the top group on top (the copies start as that group)",
          grew and mg.depth == 8 and same and "".join(b.kind for b in mg.blocks) == "MMMAMMMA" and not mg.grow_to(0.9))
    fresh = Frog(FrogConfig(**{**SMALL, "grow": "4:0,8:0.5"}, vocab_size=320)).to(dev)
    fresh.load_state_dict(mg.state_dict())
    check("a checkpoint from any depth loads (it grows to match first)", fresh.depth == 8)
    try:
        Frog(FrogConfig(**{**SMALL, "grow": "3:0,8:0.5"}))
        bad = False
    except ValueError:
        bad = True
    check("a plan must grow in whole groups of 3 memory + 1 attention", bad)

    # ---- one whole tiny run on stand-in text: Frog's recipe end to end
    from bee2.tests import stand_ins, fake_docs
    from oazy.data import get_tokenizer, EOT
    from oazy import retok, train
    tmp_ctx = tempfile.TemporaryDirectory()
    tmp = Path(tmp_ctx.name)
    used_ = stand_ins(tmp)
    if used_:
        term.info(f"no internet: stand-in {' and '.join(used_)} tokenizer(s)")
    enc = tok.get("smollm2")
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
    sys.argv = ["train", "--model", "frog", "--preset", "smoke", "--data", str(data), "--runs", str(runs), "--mentor", "smol-random",
                "--device", str(dev), "--compile", "0", "--tokens", "60000", "--batch-ramp", "512:0.5", "--mix", "fwe:0.9,chat:0.1@0",
                "--micro", "2", "--set", "n_layer=8", "grow=4:0,8:0.4"]
    train.main()
    rd = runs / "frog"
    cfg = json.loads((rd / "config.json").read_text())
    recs = [json.loads(l) for l in (rd / "metrics.jsonl").read_text().splitlines()]
    tr = [r for r in recs if r["kind"] == "train"]
    check("the run used Frog's recipe", cfg["mentor_share"] is True and cfg["doc_mask"] and cfg["mentor_floor"] == 0.25
          and cfg["muon_wd"] == 0.1 and cfg["decay_shape"] == "sqrt" and cfg["batch_ramp"] == "512:0.5")
    passes = sum(r.get("mentor_passes", 0) for r in tr)
    check("the mentor taught some passes and its lesson was logged", passes > 0 and all(r["mentor_kl"] > 0 for r in tr if "mentor_kl" in r),
          f"{passes} mentor passes logged")
    mws = [r.get("mentor_w", 0) for r in tr]
    check("its share followed Bee3's schedule", mws[0] > 0.4 and min(mws[len(mws) // 2:-1]) >= 0.249, f"{mws[0]:.2f} ... {mws[len(mws) // 2]:.2f}")
    st = json.loads((rd / "status.json").read_text())
    check("the run finished", st["state"] == "done", f"val loss {st['latest']['val_loss']:.3f}")
    mdl, ck = train.load_model(rd / "ckpt.pt", dev)
    check("its checkpoint loads as Frog", ck["kind"] == "frog" and type(mdl).__name__ == "Frog")
    deps = [r.get("depth") for r in tr]
    check("it grew inside the run (4 -> 8 layers at 40%) and finished fully grown", deps[0] == 4 and deps[-1] == 8 and mdl.depth == 8
          and any(r["kind"] == "grow" for r in recs), f"{deps[0]} -> {deps[-1]} layers")
    check("FP8 asked for by the recipe (used only where the GPU has it and the check passes)",
          "fp8" in cfg and (cfg["fp8"] is False or fp8.available(dev)), f"fp8 {cfg['fp8']}")
    out, _ = mdl.generate(enc.encode_ordinary("The cat"), max_new_tokens=5)
    check("it writes", len(out) <= 5, repr(enc.decode(out)))

    if dev.type == "cuda":
        from memtrix import memory
        term.info(f"fast memory kernel: {'yes' if memory._FLA_OK else 'no'}; packed documents in the kernel: {'yes' if memory._FLA_SEG else 'no'}")
        mg = Frog(FrogConfig(**{**SMALL, "head_k": 64, "head_v": 64})).to(dev)
        xg = torch.randint(1, 49152, (4, 128), device=dev)
        xg[:, 50] = 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss, _ = mg(xg, xg)
        loss.backward()
        check("bf16 training step on the GPU with the document rule", torch.isfinite(loss).item())
    tmp_ctx.cleanup()
    term.tests_passed("Frog")


if __name__ == "__main__":
    main()
