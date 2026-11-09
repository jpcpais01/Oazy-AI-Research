"""Self-tests for Bee6: every idea off = Bee5 exactly; one q/k/v product = three; FP8's rounding against bf16; Mixture of
Depths (the gathered words = the masked way, the helper keeps reading causal, word by word = all at once, the router
and helper learn); bloom keeps the model's answers exactly at every stage and a checkpoint taken after a bloom loads;
the recipe; the speed check; and one whole tiny run with bloom, end to end, with a resume.

    python3 -m bee6.tests                 (CPU, ~2-3 min; without internet it uses stand-in tokenizers)
    python3 -m bee6.tests --device cuda   (on the pod: also real FP8 and the compiled step on the GPU)
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
import torch.nn.functional as F

from oazy import term

SMALL = dict(d_model=64, n_layer=6, attn_heads=4, attn_kv_heads=2, attn_head_dim=16, mlp_hidden=128, max_len=128,
             ctx="128:0", vocab_size=512, key_offset="1,5", mod_layers="2,4")


def check(name, ok, detail=""):
    term.check(name, ok, detail)


def jiggle(m, s=0.05):
    """Leave the zero start, so every path (outputs, value tables, gates, skips, routers) matters."""
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * s)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    dev = torch.device(ap.parse_args().device)
    term.tests_start("Bee6-200M", dev)
    torch.manual_seed(0)
    from bee5.model import Bee5, Bee5Config
    from bee6.model import Bee6, Bee6Config, BLOOM
    from bee6 import fp8 as FP8
    from bee6 import bloom as BL

    # ---- the size
    big = Bee6(Bee6Config())
    b5 = Bee5(Bee5Config())
    extra = big.num_params() - b5.num_params()
    check("Bee5's 201M, plus the routers and their helpers (0.66M, 0.3%)",
          abs(b5.num_params() - 201.3e6) < 0.2e6 and 0 < extra < 1e6,
          f"{big.num_params() / 1e6:.2f}M = Bee5 {b5.num_params() / 1e6:.2f}M + {extra:,}; {big.layout()}")
    c = big.cfg
    check("defaults: fused q/k/v, 4,096-word slices, FP8 mode 1, layers 3/5/7/9/11 take the top 25%, bloom off",
          c.fuse_qkv == 1 and c.loss_chunk == 4096 and c.fp8 == 1 and c.mod_set() == {2, 4, 6, 8, 10} and c.mod_cap == 0.25
          and not c.bloom and not c.mod_set() & c.offset_layers())
    del big, b5

    # ---- every idea off = Bee5; one q/k/v product = three
    torch.manual_seed(1)
    x = torch.randint(1, 512, (2, 64), device=dev)
    x[:, 30] = 0
    m5 = jiggle(Bee5(Bee5Config(**{k: v for k, v in SMALL.items() if k != "mod_layers"})).to(dev))
    outs = []
    for fuse in (0, 1):
        m6 = Bee6(Bee6Config(**{**SMALL, "mod_layers": "", "fp8": 0, "fuse_qkv": fuse, "loss_chunk": 1024})).to(dev)
        m6.load_state_dict({**m5.state_dict(), "bloom_at": m6.bloom_at}, strict=True)
        for m in (m5, m6):
            m.train()
            m.set_progress(0.0)
        outs.append(m6.losses(x, x)[0].item())
    l5 = m5.losses(x, x)[0].item()
    check("Bee6 with every idea off = Bee5, to the last digit; one q/k/v product = three",
          abs(outs[0] - l5) < 1e-6 and abs(outs[1] - l5) < 1e-5, f"{l5:.6f} / {outs[0]:.6f} / {outs[1]:.6f}")
    m6.loss_chunk = 16
    l16 = m6.losses(x, x)[0].item()
    check("the loss slice size changes nothing but speed", abs(l16 - outs[1]) < 1e-5, f"{l16:.6f}")

    # ---- FP8 (its rounding, on the CPU)
    torch.manual_seed(2)
    xa = torch.randn(64, 96, requires_grad=True)
    w = (torch.randn(48, 96) * 0.1).requires_grad_()
    ref = F.linear(xa, w)
    gy = torch.randn_like(ref)
    ref.backward(gy)
    gx0, gw0 = xa.grad.clone(), w.grad.clone()
    FP8.EMULATE = True
    res = {}
    for mode in (1, 2):
        xa.grad = w.grad = None
        y = FP8.linear(xa, w, mode)
        y.backward(gy)
        res[mode] = (FP8.cosine(y, ref), FP8.cosine(xa.grad, gx0), FP8.cosine(w.grad, gw0), (w.grad - gw0).abs().max().item())
    check("FP8 product: output and input gradient within FP8's rounding of bf16; mode 1 keeps the weight gradient exact",
          all(r[0] > 0.998 and r[1] > 0.99 for r in res.values()) and res[1][3] < 1e-5 and res[2][2] > 0.99,
          " · ".join(f"mode {k}: out {v[0]:.4f}, input grad {v[1]:.4f}, weight grad {v[2]:.4f}" for k, v in res.items()))
    xo, wo = torch.randn(10, 90), torch.randn(40, 90)
    check("a size FP8 cannot take (not divisible by 16) quietly stays in bf16", torch.equal(FP8.linear(xo, wo, 1), F.linear(xo, wo)))
    mf = jiggle(Bee6(Bee6Config(**{**SMALL, "fp8": 1})).to(dev))
    ok, gap, cosv = FP8.check(mf, dev, T=64, B=2)
    check("the startup check: the whole model in FP8 vs bf16 on one batch (loss, gradients)", ok, f"loss {gap:.2%} apart, gradients {cosv:.3f} alike")
    mf.startup_check(torch.device("cpu"), lambda s_: None)
    check("on a device without FP8 Bee6 says so and trains in bf16", mf.cfg.fp8 == 0)
    FP8.EMULATE = False

    # ---- Mixture of Depths
    torch.manual_seed(3)
    m = jiggle(Bee6(Bee6Config(**{**SMALL, "fp8": 0})).to(dev))
    blk = m.blocks[2]
    xs = torch.randn(2, 64, 64, device=dev)
    cos, sin = m.rope_cos[:64], m.rope_sin[:64]
    ve = None
    blk.train()
    r, xn = blk.scores(xs)
    a, _ = blk.through_topk(xs, ve, cos, sin, r, xn)
    b = blk.through_mask(xs, ve, cos, sin, r, blk.topk_mask(r))
    e = (a - b).abs().max().item()
    changed = (a - xs).abs().amax(-1) > 0
    check("a routed layer: the top 25% gathered and put back = the masked way; the other words pass unchanged",
          e < 1e-5 and int(changed.sum()) == 2 * 16, f"max diff {e:.1e}; {int(changed.sum())} of 128 words changed")
    m.eval()
    with torch.no_grad():
        full = m(x)[0]
        y2 = x.clone()
        y2[:, 40:] = torch.randint(1, 512, (2, 24), device=dev)
        part = m(y2)[0]
    e = (full[:, :40] - part[:, :40]).abs().max().item()
    check("reading (the helper picks the words): changing later words never changes earlier scores", e < 1e-5, f"max diff {e:.1e}")
    m.cfg.mod_eval = "topk"
    with torch.no_grad():
        e_top = (m(x)[0][:, :40] - m(y2)[0][:, :40]).abs().max().item()
    m.cfg.mod_eval = "predict"
    check("... while the router's own top 25% does look ahead (which is why reading uses the helper)", e_top > 1e-4, f"{e_top:.1e}")
    with torch.no_grad():
        whole, _ = m.read(x[:1, :40])
        lg, st = m.read(x[:1, :25])
        for t in range(25, 40):
            lg, st = m.read(x[:1, t:t + 1], st)
        e = (lg[0, -1] - whole[0, -1]).abs().max().item()
        e2 = (whole[0] - m(x[:1, :40])[0][0]).abs().max().item()
    check("reading word by word = all at once (routed layers remember which words were picked)", e < 1e-4 and e2 < 1e-4,
          f"max diff {e:.1e}, read vs forward {e2:.1e}")
    out, _ = m.generate([5, 6, 7], max_new_tokens=6, eos=None)
    check("it writes", len(out) == 6)
    m.train()
    m.set_progress(1.0)
    m.zero_grad(set_to_none=True)
    loss, stt = m.losses(x, x)
    loss.backward()
    gr = m.blocks[2].router.weight.grad.abs().sum().item()
    gp = m.blocks[2].pred[0].weight.grad.abs().sum().item()
    check("the router learns (through 2 x sigmoid of its score) and so does the helper (its own small lesson)",
          gr > 0 and gp > 0 and "mod_agree" in stt, f"router grad {gr:.2e}, helper grad {gp:.2e}, agreement {float(stt['mod_agree']):.2f}")
    from bee5 import optim as O
    muon, adam = O.groups(m)
    ids = [id(p_) for _, p_ in muon] + [id(p_) for ps_ in adam.values() for p_ in ps_]
    check("every parameter is in exactly one optimizer group (routers and helpers with the gates, in Adam)",
          len(ids) == len(set(ids)) == len([p_ for p_ in m.parameters() if p_.requires_grad]) and len(muon) == 6 * 6)

    torch.manual_seed(4)
    from oazy.train import make_optimizer
    mm = Bee6(Bee6Config(**{**SMALL, "fp8": 0})).to(dev).train()
    mm.set_progress(0.0)
    opt = make_optimizer(mm, 3e-3, dev, "muon", 0.03, 0.0)
    for g in opt.param_groups:
        g["lr"] = 3e-3 * g["lr_scale"]
    xb = torch.randint(1, 512, (4, 64), device=dev)
    first, agrees = None, []
    for i in range(80):
        loss, stt = mm.losses(xb, xb)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        first = first if first is not None else stt["ce"].item()
        agrees.append(float(stt["mod_agree"]))
    mm.eval()
    last = mm.losses(xb, xb)[0].item()
    check("80 steps on one batch: the loss falls fast (reading with the helper's picks), and the helper learns the router",
          last < first - 2.0 and np.mean(agrees[-10:]) > max(0.7, np.mean(agrees[:5])),
          f"{first:.2f} -> {last:.2f}; helper agrees {np.mean(agrees[:5]):.2f} -> {np.mean(agrees[-10:]):.2f}")

    # ---- bloom
    torch.manual_seed(5)
    bc = Bee6Config(**{**SMALL, "n_layer": 12, "d_model": 192, "attn_heads": 12, "attn_kv_heads": 3, "mlp_hidden": 720,
                       "key_offset": "3,7,11", "mod_layers": "2,4,6,8,10", "fp8": 0,
                       "bloom": "4x64:0,6x128:0.2,9x192:0.4,12x192:0.6"})
    mb = Bee6(bc).to(dev).double()       # in 64-bit, so "the same" can be checked to the last digits
    sizes = [(bc.n_layer, bc.d_model, bc.attn_heads, bc.attn_kv_heads, bc.mlp_hidden)]
    check("bloom starts at the plan's first stage (4 layers of 64: 4 heads, 1 key-value head, MLP 240)",
          sizes[0] == (4, 64, 4, 1, 240) and mb.cfg.bloom_stage == 0, str(sizes[0]))
    jiggle(mb)
    gaps, kinds = [], []
    xb2 = torch.randint(1, 512, (2, 64), device=dev)
    for q in (0.25, 0.45, 0.7):
        mb.train()
        mb.set_progress(1.0)
        with torch.no_grad():
            before = mb.losses(xb2, xb2)[1]["ce"].item()
            mb.eval()
            lb = mb(xb2)[0]
        mb.grow_to(q)
        mb.train()
        with torch.no_grad():
            after = mb.losses(xb2, xb2)[1]["ce"].item()
            mb.eval()
            la = mb(xb2)[0]
        gaps.append(((la - lb).abs().max().item(), abs(after - before)))
        kinds.append(f"{mb.cfg.n_layer}x{mb.cfg.d_model} mod {mb.cfg.mod_layers} off {mb.cfg.key_offset}")
        sizes.append((mb.cfg.n_layer, mb.cfg.d_model, mb.cfg.attn_heads, mb.cfg.attn_kv_heads, mb.cfg.mlp_hidden))
        jiggle(mb, 0.02)              # learn a little in between, so the next growth starts from a used model
    check("each bloom keeps the model's answers: same training loss, same scores (wider, then deeper, then both)",
          max(g[0] for g in gaps) < 1e-9 and max(g[1] for g in gaps) < 1e-5,
          "scores " + " · ".join(f"{g[0]:.0e}" for g in gaps) + "; loss " + " · ".join(f"{g[1]:.0e}" for g in gaps) + " (the training path, its slices 32-bit)")
    check("the stages: 4x64 -> 6x128 -> 9x192 -> 12x192, heads and MLP growing with the width",
          [s[:2] for s in sizes] == [(4, 64), (6, 128), (9, 192), (12, 192)] and sizes[-1][2:] == (12, 3, 720), " -> ".join(kinds))
    plain = Bee6(Bee6Config(**{**{k: v for k, v in bc.__dict__.items() if k in Bee6Config.__dataclass_fields__},
                               "bloom": "", "bloom_stage": -1, "backout_layer": -1}))
    check("fully grown, it has the plain model's size (with the same routed layers)", mb.num_params() == plain.num_params(),
          f"{mb.num_params():,} vs {plain.num_params():,}")
    check("routers grew with the width and new layers in the pattern got one; layers never both route and shift keys",
          all(b_.router.weight.shape[1] == 192 for b_ in mb.blocks if b_.routed) and not (mb.cfg.mod_set() & mb.cfg.offset_layers())
          and len(mb.cfg.mod_set()) >= 4, f"routed {sorted(mb.cfg.mod_set())}, shifted keys {sorted(mb.cfg.offset_layers())}")
    mb = mb.float()
    fresh = Bee6(Bee6Config(**{**SMALL, "n_layer": 12, "d_model": 192, "attn_heads": 12, "attn_kv_heads": 3, "mlp_hidden": 720,
                                "key_offset": "3,7,11", "mod_layers": "2,4,6,8,10", "fp8": 0,
                                "bloom": "4x64:0,6x128:0.2,9x192:0.4,12x192:0.6"})).to(dev)
    xb2 = xb2.to(dev)
    fresh.load_state_dict(mb.state_dict())
    fresh.eval()
    mb.eval()
    with torch.no_grad():
        e = (fresh(xb2)[0] - mb(xb2)[0]).abs().max().item()
    check("a checkpoint taken after the blooms loads into a new model (it grows to the checkpoint's stage first)", e < 1e-5,
          f"max diff {e:.1e}")
    check("Bee6's plan: 4x512 (20%) -> 6x768 (20%) -> 9x1,024 (20%) -> 12x1,024 (40%), off unless --set bloom=on",
          Bee6Config(bloom="on").plan() == [(4, 512, 0.0), (6, 768, 0.2), (9, 1024, 0.4), (12, 1024, 0.6)] and BLOOM)
    bad = False
    try:
        Bee6(Bee6Config(**{**SMALL, "bloom": "4x64:0,6x48:0.5"}))
    except ValueError:
        bad = True
    check("a plan that does not end at the model's size is refused", bad)

    # ---- the recipe
    from oazy.train import RECIPE, PRESETS, MENTOR, BEES, FAST, NICE, KINDS
    check("Bee5's recipe exactly (Sado, the mentor 1 in 4 -> 1 in 10, WSD 30%, the ramp), the trainer's own FP8 off",
          RECIPE["bee6"] == RECIPE["bee5"] and RECIPE["bee6"]["fp8"] == 0)
    pb = PRESETS["big"]
    check("big preset: 524k-token steps, 1.8e-3 / Muon 0.03; SmolLM2-360M mentor", pb["batch_bee6"] == 524288
          and pb["lr_bee6"] == 1.8e-3 and pb["muon_lr_bee6"] == 0.03 and MENTOR["bee6"] == "smollm2-360m"
          and "bee6" in BEES and "bee6" in FAST and "bee6" in KINDS and NICE["bee6"] == "Bee6-200M")

    # ---- the speed check (quick version)
    from bee6 import speedcheck
    sc = speedcheck.main(["--preset", "smoke", "--device", str(dev), "--passes", "1", "--runs", tempfile.mkdtemp()])
    check("the speed check runs every rung (Bee5, ideas 2-5) and prints a train line",
          sc["line"].startswith("OAZY_DATA=/root/data bash runpod/train.sh bee6") and sc["tok_s"] > 0 and sc["off"] is not None
          and len(sc["bloom_stage_tok_s"]) == 4, sc["line"])

    # ---- one whole tiny run with bloom, then a resume
    from bee2.tests import stand_ins, fake_docs
    from oazy.data import get_tokenizer, EOT
    from oazy import retok, train, tok
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
    pre = train.PRESETS["smoke"]["bee"]
    plan = f"2x{pre['d_model'] // 2}:0,{pre['n_layer']}x{pre['d_model']}:0.5"
    argv = ["train", "--model", "bee6", "--preset", "smoke", "--data", str(data), "--runs", str(runs), "--mentor", "smol-random",
            "--device", str(dev), "--compile", "0", "--tokens", "60000", "--batch-ramp", "512:0.5", "--mix", "fwe:0.9,chat:0.1@0",
            "--micro", "2", "--data-sub", "smol", "--set", "ctx=64:0,128:0.3", "attn_head_dim=16", "attn_heads=4", "attn_kv_heads=2",
            f"bloom={plan}"]
    sys.argv = argv
    train.main()
    rd = runs / "bee6"
    recs = [json.loads(l_) for l_ in (rd / "metrics.jsonl").read_text().splitlines()]
    grows = [r for r in recs if r["kind"] == "grow"]
    tr = [r for r in recs if r["kind"] == "train"]
    st = json.loads((rd / "status.json").read_text())
    check("a whole tiny run with bloom: it grew once at half way, trained on, and finished with held-out loss and HellaSwag",
          len(grows) == 1 and st["state"] == "done" and "val_loss" in st["latest"] and tr[-1].get("depth") == pre["n_layer"],
          f"grew at {grows[0]['tokens'] if grows else '-':,} tokens; val loss {st['latest'].get('val_loss', float('nan')):.3f}")
    check("the log shows how often the helper agrees with the router", any("mod_agree" in r for r in tr))
    mdl, ck = train.load_model(rd / "ckpt.pt", dev)
    check("its checkpoint loads as Bee6, fully grown", ck["kind"] == "bee6" and type(mdl).__name__ == "Bee6"
          and mdl.cfg.n_layer == pre["n_layer"] and mdl.cfg.bloom_stage == 1)
    out, _ = mdl.generate(enc.encode_ordinary("The cat"), max_new_tokens=5)
    check("the trained model writes", len(out) <= 5, repr(enc.decode(out)))
    n_before = len(recs)
    sys.argv = argv
    train.main()
    recs2 = [json.loads(l_) for l_ in (rd / "metrics.jsonl").read_text().splitlines()]
    st2 = json.loads((rd / "status.json").read_text())
    check("run again, it picks up its grown checkpoint and optimizer instead of starting over",
          st2["state"] == "done" and len(recs2) == n_before and st2["tokens"] == st["tokens"], f"{st2['tokens']:,} tokens")

    if dev.type == "cuda":
        mg = jiggle(Bee6(Bee6Config(**{**SMALL, "vocab_size": 49152, "fp8": 1}))).to(dev).train()
        mg.startup_check(dev, lambda s_: term.info(s_))
        check("FP8 on this GPU passes the startup check (or Bee6 says why not)", mg.cfg.fp8 in (0, 1))
        mg.set_progress(0.0)
        xg = torch.randint(1, 49152, (4, 128), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = mg.losses(xg, xg)
        loss.backward()
        check("bf16 / FP8 training step on the GPU", torch.isfinite(loss).item())
        mg.eval()
        wo, _ = mg.generate([5, 6, 7], max_new_tokens=4, eos=None)       # outside autocast, as the trainer's samples
        with torch.no_grad():
            lw, _ = mg.read(xg[:1, :32])
        mg.train()
        check("with FP8 on, it writes and reads outside autocast (the trainer's samples)", len(wo) == 4 and lw.dtype == torch.float32)
        mg.zero_grad(set_to_none=True)
        mg.enable_compile()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lc, _ = mg.losses(xg, xg)
        lc.backward()
        check("the compiled training step runs and agrees", torch.isfinite(lc).item() and abs(lc.item() - loss.item()) < 0.05,
              f"{lc.item():.4f} vs {loss.item():.4f}")
    tmp_ctx.cleanup()
    term.tests_passed("Bee6-200M")


if __name__ == "__main__":
    main()
