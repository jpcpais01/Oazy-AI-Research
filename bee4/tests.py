"""Self-tests for Bee4: the size (198M, all attention, dense), RoPE and plain causal attention, growing, FP8's
choice of layers, Coral's recipe, a resume after a growth, and one whole (tiny) run end to end.

    python -m bee4.tests                 (CPU, ~1 min; without internet it uses stand-in tokenizers)
    python -m bee4.tests --device cuda   (on the pod: also bf16, the compiled step, FP8)
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

from bee3.tests import check
from oazy import term

SMALL = dict(d_model=64, n_layer=4, attn_heads=4, attn_kv_heads=2, attn_head_dim=16, mlp_hidden=96, max_len=128, ctx="128:0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    dev = torch.device(ap.parse_args().device)
    term.tests_start("Bee4-200M", dev)
    torch.manual_seed(0)
    from bee4.model import Bee4, Bee4Config
    from oazy import tok

    # ---- the size and the body
    big = Bee4(Bee4Config())
    n = big.full_params()
    check("about 200M parameters once grown, all at work", 190e6 < n < 205e6,
          f"{n / 1e6:.1f}M ({(n - big.embed.weight.numel()) / 1e6:.0f}M without the word table), {big.layout()}")
    check("12 layers, all attention, with RoPE (base 50,000); 2,048 windows, SmolLM2's pieces",
          big.cfg.kinds() == ["A"] * 12 and big.cfg.rope == 1 and big.cfg.rope_base == 50000 and big.ctx_at(0) == 2048
          and big.cfg.doc_mask == 1 and big.cfg.tokenizer == "smollm2" and tok.eot_id("smollm2") == 0)
    check("no Memtrix and no experts: one MLP of 2,816 per layer", not any(hasattr(b, "mem") or hasattr(b, "moe")
          for b in big.blocks) and big.blocks[0].mlp is not None)
    _x = torch.tensor([[5, 6, 0, 7, 8, 9]])
    _r, _m, _s = big.doc_info(_x)
    check("attention runs plain causal on the fused kernel (attn_docs=0); the mentor still reads each document alone",
          big.cfg.attn_docs == 0 and big.attn_mask(_x, _r, _m) is None and _m is not None and not bool(_m[0, 0, 3, 1]) and _s is None)
    check("it starts 6 layers deep and grows to 9 at 10%, 12 at 25% (Coral's timing)", big.depth == 6
          and big.depth_at(0.099) == 6 and big.depth_at(0.1) == 9 and big.depth_at(0.25) == 12)
    from oazy import fp8
    picked = [f for f, mod in big.named_modules() if fp8.keep(mod, f)]
    check("FP8 goes to attention and the MLP of every layer, not the word table",
          len(picked) == 6 * len([f for f in picked if f.startswith("blocks.0.")]) and not any(f.startswith("embed") for f in picked),
          f"{len(picked)} layers at depth 6")
    del big

    # ---- reading = training; growing
    m = Bee4(Bee4Config(**SMALL)).to(dev)
    x = torch.randint(1, 49152, (2, 64), device=dev)
    x[:, 30] = 0
    m.train()
    lt, _ = m.losses(x, x)
    m.eval()
    le, _ = m.losses(x, x)
    with torch.no_grad():
        logits = m(x)[0]
        full = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x.reshape(-1))
    check("training loss = reading loss = the full scores' loss (same attention rule everywhere)",
          abs(lt.item() - le.item()) < 1e-4 and abs(le.item() - full.item()) < 1e-3, f"{lt.item():.4f} / {le.item():.4f} / {full.item():.4f}")
    with torch.no_grad():
        lg, states = m.read(x[:1, :20])
        lg2, _ = m.read(x[:1, 20:21], states)
        whole, _ = m.read(x[:1, :21])
    e = (lg2[0, -1] - whole[0, -1]).abs().max().item()
    check("chat reads word by word the same as all at once (RoPE positions carried along)", e < 1e-4, f"max diff {e:.1e}")
    g = Bee4(Bee4Config(**{**SMALL, "n_layer": 8, "grow": "4:0,8:0.5"}))
    with torch.no_grad():
        g.blocks[3].mlp.down.weight.fill_(0.01)
    g.grow_to(0.5)
    check("it grows 4 -> 8 by copying the top layers", g.depth == 8
          and torch.equal(g.blocks[7].mlp.down.weight, g.blocks[3].mlp.down.weight))
    sd = g.state_dict()
    g2 = Bee4(Bee4Config(**{**SMALL, "n_layer": 8, "grow": "4:0,8:0.5"}))
    g2.load_state_dict(sd)
    check("a grown checkpoint loads into a fresh (shallow) Bee4", g2.depth == 8)

    # ---- the recipe
    from oazy.train import RECIPE, PRESETS, MENTOR, BEES, FAST
    from ember.mentor import fade
    rc = RECIPE["bee4"]
    ws = [fade(q, rc["mentor_weight"], rc["mentor_until"], None, rc["mentor_floor"], rc["mentor_off"]) for q in (0, 0.3, 0.8, 0.95, 1.0)]
    same = {k: v for k, v in rc.items() if not k.startswith("mentor_")} == {k: v for k, v in RECIPE["coral"].items() if not k.startswith("mentor_")}
    check("Coral's recipe (FP8, the ramp, Muon decay, sqrt cool-down, Tagus20); the mentor 1 in 4 -> 1 in 10 by 30%, kept, off over the last 10%",
          same and rc["mentor_share"] == 1 and rc["fp8"] == 1 and abs(ws[0] - 0.25) < 1e-9 and abs(ws[1] - 0.1) < 1e-9
          and abs(ws[2] - 0.1) < 1e-9 and 0 < ws[3] < 0.1 and ws[4] < 1e-9
          and rc["data_sub"] == "tagus20" and rc["decay_shape"] == "sqrt", " ".join(f"{w:.2f}" for w in ws))
    pb = PRESETS["big"]
    check("big preset: 524k-token steps, learning rates 1.8e-3 / Muon 0.03", pb["batch_bee4"] == 524288
          and pb["lr_bee4"] == 1.8e-3 and pb["muon_lr_bee4"] == 0.03)
    check("SmolLM2-360M mentor, kept all run, raised the Bee way", MENTOR["bee4"] == "smollm2-360m" and "bee4" in BEES and "bee4" in FAST)

    # ---- one whole tiny run on stand-in text: Bee4's recipe end to end, then a resume
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
    argv = ["train", "--model", "bee4", "--preset", "smoke", "--data", str(data), "--runs", str(runs), "--mentor", "smol-random",
            "--device", str(dev), "--compile", "0", "--tokens", "60000", "--batch-ramp", "512:0.5", "--mix", "fwe:0.9,chat:0.1@0",
            "--micro", "2", "--set", "n_layer=8", "grow=4:0,8:0.4"]
    sys.argv = argv
    try:
        train.main()
        missing = False
    except SystemExit as e:
        missing = "tagus20" in str(e)
    check("by default it reads Tagus20 and says how to build it when it is missing", missing)
    sys.argv = argv + ["--data-sub", "smol"]
    train.main()
    rd = runs / "bee4"
    recs = [json.loads(l_) for l_ in (rd / "metrics.jsonl").read_text().splitlines()]
    tr = [r for r in recs if r["kind"] == "train"]
    cfg_run = json.loads((rd / "config.json").read_text())
    check("the run used Bee4's recipe", cfg_run["mentor_share"] is True and cfg_run["mentor_weight"] == 0.25
          and cfg_run.get("mentor_floor") == 0.1 and cfg_run.get("mentor_off") == 0.1, cfg_run["layout"])
    mws = [r.get("mentor_w", 0) for r in tr]
    mid = [w for r, w in zip(tr, mws) if 0.35 < r.get("progress", r["tokens"] / 60000) < 0.85]
    check("the mentor taught all run: 1 in 4 early, 1 in 10 in the middle", 0.15 < mws[0] <= 0.25
          and mid and all(abs(w - 0.1) < 1e-6 for w in mid), f"{mws[0]:.2f} ... {mid[0] if mid else '?'} ... {mws[-1]:.2f}")
    deps = [r.get("depth") for r in tr]
    check("it grew inside the run (4 -> 8 layers at 40%)", deps[0] == 4 and deps[-1] == 8)
    st = json.loads((rd / "status.json").read_text())
    check("the run finished", st["state"] == "done", f"val loss {st['latest']['val_loss']:.3f}")
    mdl, ck = train.load_model(rd / "ckpt.pt", dev)
    check("its checkpoint loads as Bee4, fully grown", ck["kind"] == "bee4" and type(mdl).__name__ == "Bee4" and mdl.depth == 8)
    out, _ = mdl.generate(enc.encode_ordinary("The cat"), max_new_tokens=5)
    check("it writes", len(out) <= 5, repr(enc.decode(out)))
    # the same line again (a stopped pod, train.sh run again): it picks up its checkpoint, grown, instead of starting over
    n_before = len(recs)
    sys.argv = argv + ["--data-sub", "smol"]
    train.main()
    recs2 = [json.loads(l_) for l_ in (rd / "metrics.jsonl").read_text().splitlines()]
    tr2 = [r for r in recs2[n_before:] if r["kind"] == "train"]
    st2 = json.loads((rd / "status.json").read_text())
    check("run again, it resumes from its checkpoint (grown) instead of starting over",
          st2["state"] == "done" and all(r.get("depth") == 8 for r in tr2) and st2["latest"].get("tokens", 1) > 0,
          f"{len(tr2)} new training lines")

    if dev.type == "cuda":
        mg = Bee4(Bee4Config(**SMALL)).to(dev)
        xg = torch.randint(1, 49152, (4, 256), device=dev)
        xg[:, 50] = 0
        mg.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = mg.losses(xg, xg)
        loss.backward()
        check("bf16 training step on the GPU", torch.isfinite(loss).item())
        mg.zero_grad(set_to_none=True)
        mg.enable_compile()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lc, _ = mg.losses(xg, xg)
        lc.backward()
        check("the compiled training step runs", torch.isfinite(lc).item() and abs(lc.item() - loss.item()) < 0.05,
              f"{lc.item():.4f} vs {loss.item():.4f}")
        mf, used = fp8.convert(Bee4(Bee4Config(**{**SMALL, "d_model": 256, "attn_heads": 4, "attn_head_dim": 64,
                                                  "mlp_hidden": 512})).to(dev).train(), dev, print)
        term.info(f"FP8 on this GPU: {'yes' if used else 'no'}")
    tmp_ctx.cleanup()
    term.tests_passed("Bee4-200M")


if __name__ == "__main__":
    main()
