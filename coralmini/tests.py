"""Self-tests for Coral Mini: the size (about 198M in total, about 115M at work per word), Coral's experts at 16 x 160
(top 2) + a shared MLP of 256, Bee4's body and recipe (growing 6 -> 9 -> 12, short windows first, the mentor kept,
the cool-down over the last 30%, bf16), the experts' balancing, Muon on the expert stacks, the speed check, and one
whole (tiny) run end to end with its resume.

    python -m coralmini.tests                 (CPU, ~2 min; without internet it uses stand-in tokenizers)
    python -m coralmini.tests --device cuda   (on the pod: also bf16, the compiled step, FP8)
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

SMALL = dict(d_model=64, n_layer=4, attn_heads=4, attn_kv_heads=2, attn_head_dim=16, mlp_hidden=32, max_len=128, ctx="128:0",
             n_expert=8, top_k=2, expert_hidden=16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    dev = torch.device(ap.parse_args().device)
    term.tests_start("Coral Mini", dev)
    torch.manual_seed(0)
    from coralmini.model import CoralMini, CoralMiniConfig
    from bee4.model import Bee4, Bee4Config
    from oazy import tok

    # ---- the size and the body
    big = CoralMini(CoralMiniConfig())
    n, act, emb = big.full_params(), big.active_params(), big.embed.weight.numel()
    bee4 = Bee4(Bee4Config())
    nb = bee4.full_params()
    term.info(f"Coral Mini: {n / 1e6:.1f}M in total ({(n - emb) / 1e6:.1f}M without the word table), "
              f"{act / 1e6:.1f}M at work per word ({(act - emb) / 1e6:.1f}M without the word table); Bee4 {nb / 1e6:.1f}M")
    check("about 200M parameters once grown, within 1% of Bee4's", 190e6 < n < 205e6 and abs(n - nb) / nb < 0.01,
          f"{n / 1e6:.1f}M vs Bee4 {nb / 1e6:.1f}M, {big.layout()}")
    check("about 115M at work per word (2 of 16 experts + the shared MLP + attention + the word table)",
          105e6 < act < 125e6, f"{act / 1e6:.1f}M at work, {(act - emb) / 1e6:.1f}M without the word table")
    c = big.cfg
    check("16 experts of 160, top 2, + a shared MLP of 256: the same thinking width as Bee4's MLP (2,816)",
          c.n_expert == 16 and c.top_k == 2 and c.expert_hidden == 160 and c.mlp_hidden == 256
          and c.n_expert * c.expert_hidden + c.mlp_hidden == bee4.cfg.mlp_hidden
          and all(hasattr(b, "moe") and b.moe.E == 16 for b in big.blocks))
    same_body = all(getattr(c, k) == getattr(bee4.cfg, k) for k in
                    ("d_model", "n_layer", "pattern", "attn_heads", "attn_kv_heads", "attn_head_dim", "rope", "rope_base",
                     "attn_docs", "grow", "ctx", "max_len", "tokenizer", "doc_mask"))
    check("Bee4's body: width 1,024, 12 layers of attention with RoPE, 6 -> 9 at 10% -> 12 at 25%, windows 512 -> 1,024 -> 2,048",
          same_body and c.kinds() == ["A"] * 12 and big.depth == 6 and big.depth_at(0.1) == 9 and big.depth_at(0.25) == 12
          and big.ctx_at(0) == 512 and big.ctx_at(0.15) == 1024 and big.ctx_at(0.3) == 2048 and tok.eot_id("smollm2") == 0)
    _x = torch.tensor([[5, 6, 0, 7, 8, 9]])
    _r, _m, _ = big.doc_info(_x)
    check("attention runs plain causal on the fused kernel (attn_docs=0); the mentor still reads each document alone",
          big._attn_mask(_x, _r, _m) is None and _m is not None and not bool(_m[0, 0, 3, 1]))
    del big, bee4

    # ---- the experts at this size; reading = training when nothing overflows
    from coral.model import Experts
    ex = Experts(CoralMiniConfig(**SMALL, capacity=8.0)).to(dev)
    h = torch.randn(3, 40, 64, device=dev)
    ex.train()
    y_cap = ex(h)
    ex.eval()
    e = (y_cap - ex(h)).abs().max().item()
    check("fixed-capacity experts (training) = one expert at a time (reading), when nothing overflows", e < 1e-5, f"max diff {e:.1e}")
    ex2 = Experts(CoralMiniConfig(**SMALL, balance_rate=0.05)).to(dev)
    with torch.no_grad():
        ex2.router.weight.normal_(0, 0.1)
        ex2.router.weight[0] += 0.5
    xb = torch.randn(1, 512, 64, device=dev)
    ex2.train()
    first = None
    for _ in range(80):
        ex2(xb)
        load, _ = ex2.balance()
        u = (load.max() / load.mean()).item()
        first = u if first is None else first
    check("balancing biases even the work out (DeepSeek-V3)", u < first and u < 2.0, f"busiest / average {first:.2f} -> {u:.2f}")

    m = CoralMini(CoralMiniConfig(**SMALL, capacity=8.0)).to(dev)
    x = torch.randint(1, 49152, (2, 64), device=dev)
    x[:, 30] = 0
    m.train()
    lt, st = m.losses(x, x)
    m.eval()
    le, _ = m.losses(x, x)
    with torch.no_grad():
        logits = m(x)[0]
        full = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x.reshape(-1))
    check("training loss = reading loss + the 1e-4 balance term = the full scores' loss (+ that term)",
          abs(lt.item() - le.item() - m.cfg.aux_weight * float(st["moe_aux"]) * len(m.blocks)) < 1e-4
          and abs(le.item() - full.item()) < 1e-3, f"{lt.item():.4f} / {le.item():.4f} / {full.item():.4f}")
    with torch.no_grad():
        lg, states = m.read(x[:1, :20])
        lg2, _ = m.read(x[:1, 20:21], states)
        whole, _ = m.read(x[:1, :21])
    e = (lg2[0, -1] - whole[0, -1]).abs().max().item()
    check("chat reads word by word the same as all at once", e < 1e-4, f"max diff {e:.1e}")
    m.train()
    lt.backward()
    from oazy.train import make_optimizer, RECIPE, PRESETS, MENTOR, BEES, FAST
    opt = make_optimizer(m, 1e-3, dev, "muon", 0.03, 0.1)
    muon = opt.opts[0]
    stacks = [p for g in muon.param_groups for p in g["params"] if p.dim() == 3]
    opt.step()
    m.after_step()
    check("Muon straightens every expert's grids, the router stays on AdamW", len(stacks) == 2 * len(m.blocks)
          and not any(p is m.blocks[0].moe.router.weight for g in muon.param_groups for p in g["params"]))
    check("the log line gets the experts' evenness and overflow", set(m.moe_report()) == {"moe_uneven", "moe_dropped"})
    g = CoralMini(CoralMiniConfig(**{**SMALL, "n_layer": 8, "grow": "4:0,8:0.5"}))
    g.blocks[3].moe.bias.fill_(0.3)
    g.grow_to(0.5)
    check("it grows 4 -> 8 by copying the top layers, experts and their biases included", g.depth == 8
          and abs(g.blocks[7].moe.bias[0].item() - 0.3) < 1e-6)
    mc = CoralMini(CoralMiniConfig(**SMALL, capacity=8.0, attn_backend="cudnn", compile_mode="max-autotune-no-cudagraphs")).to(dev)
    mc.load_state_dict(m.state_dict())
    mc.eval()
    m.eval()
    with torch.no_grad():
        e = (mc(x)[0] - m(x)[0]).abs().max().item()
    check("Bee4's speed switches (cuDNN attention, compile mode) leave the maths alone", e < (2e-2 if dev.type == "cuda" else 1e-5),
          f"max diff {e:.1e}")
    from oazy import speedcheck
    sys.argv = ["speedcheck", "--model", "coralmini", "--preset", "smoke", "--device", str(dev), "--passes", "1",
                "--runs", str(Path(tempfile.mkdtemp()))]
    sc = speedcheck.main()
    check("the speed check runs and prints a train line", sc["line"].startswith("bash runpod/train.sh coralmini") and sc["tok_s"] > 0,
          sc["line"])

    # ---- the recipe: exactly Bee4's
    rc = RECIPE["coralmini"]
    pb = PRESETS["big"]
    check("Bee4's recipe exactly (mentor 1 in 4 -> 1 in 10 kept, off over the last 10%; cool-down last 30%; bf16; Tagus20)",
          rc == RECIPE["bee4"] and rc["decay"] == 0.3 and rc["fp8"] == 0 and rc["data_sub"] == "tagus20" and rc["mentor_floor"] == 0.1)
    check("big preset: Bee4's 524k-token steps and learning rates (1.8e-3 / Muon 0.03)", pb["batch_coralmini"] == pb["batch_bee4"] == 524288
          and pb["lr_coralmini"] == pb["lr_bee4"] and pb["muon_lr_coralmini"] == pb["muon_lr_bee4"])
    check("SmolLM2-360M mentor, raised the Bee way", MENTOR["coralmini"] == MENTOR["bee4"] == "smollm2-360m"
          and "coralmini" in BEES and "coralmini" in FAST)

    # ---- one whole tiny run on stand-in text, then a resume
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
    ev = data / "evals"
    ev.mkdir()
    for name, k in (("arc_easy", 4), ("piqa", 2)):
        (ev / f"{name}.jsonl").write_text("".join(json.dumps({"ctx": "Question: " + " ".join(fake_docs(1, rng)) + "\nAnswer:",
                                                               "endings": fake_docs(k, rng), "label": rng.randint(0, k - 1)}) + "\n" for _ in range(30)))
    (ev / "lambada.jsonl").write_text("".join(json.dumps({"ctx": " ".join(fake_docs(1, rng)), "target": " cat"}) + "\n" for _ in range(30)))
    retok.run(data, "smollm2", ["fwe", "chat"], workers=2)
    runs = tmp / "runs"
    argv = ["train", "--model", "coralmini", "--preset", "smoke", "--data", str(data), "--runs", str(runs), "--mentor", "smol-random",
            "--device", str(dev), "--compile", "0", "--tokens", "60000", "--batch-ramp", "512:0.5", "--mix", "fwe:0.9,chat:0.1@0",
            "--micro", "2", "--set", "n_layer=8", "grow=4:0,8:0.4", "ctx=64:0,128:0.3"]
    sys.argv = argv
    try:
        train.main()
        missing = False
    except SystemExit as e:
        missing = "tagus20" in str(e)
    check("by default it reads Tagus20 and says how to build it when it is missing", missing)
    sys.argv = argv + ["--data-sub", "smol"]
    train.main()
    rd = runs / "coralmini"
    recs = [json.loads(l_) for l_ in (rd / "metrics.jsonl").read_text().splitlines()]
    tr = [r for r in recs if r["kind"] == "train"]
    cfg_run = json.loads((rd / "config.json").read_text())
    check("the run used Bee4's recipe, and logged the parameters at work", cfg_run["mentor_share"] is True
          and cfg_run["mentor_weight"] == 0.25 and cfg_run.get("mentor_floor") == 0.1 and cfg_run.get("decay") == 0.3
          and cfg_run["params_active"] < cfg_run["params"], cfg_run["layout"])
    check("the experts' evenness and overflow were logged", all("moe_uneven" in r and "moe_dropped" in r for r in tr))
    mws = [r.get("mentor_w", 0) for r in tr]
    mid = [w for r, w in zip(tr, mws) if 0.35 < r.get("progress", r["tokens"] / 60000) < 0.85]
    check("the mentor taught all run: 1 in 4 early, 1 in 10 in the middle", 0.15 < mws[0] <= 0.25
          and mid and all(abs(w - 0.1) < 1e-6 for w in mid), f"{mws[0]:.2f} ... {mid[0] if mid else '?'} ... {mws[-1]:.2f}")
    ctxs = [r.get("context") for r in tr]
    deps = [r.get("depth") for r in tr]
    check("the windows grew (64 -> 128) and the model grew (4 -> 8 layers) inside the run",
          ctxs[0] == 64 and ctxs[-1] == 128 and deps[0] == 4 and deps[-1] == 8)
    st = json.loads((rd / "status.json").read_text())
    check("the run finished", st["state"] == "done", f"val loss {st['latest']['val_loss']:.3f}")
    mdl, ck = train.load_model(rd / "ckpt.pt", dev)
    check("its checkpoint loads as Coral Mini, grown, balancing biases included", ck["kind"] == "coralmini"
          and type(mdl).__name__ == "CoralMini" and mdl.depth == 8 and any(b.moe.bias.abs().sum() > 0 for b in mdl.blocks))
    out, _ = mdl.generate(enc.encode_ordinary("The cat"), max_new_tokens=5)
    check("it writes", len(out) <= 5, repr(enc.decode(out)))
    n_before = len(recs)
    sys.argv = argv + ["--data-sub", "smol"]
    train.main()
    recs2 = [json.loads(l_) for l_ in (rd / "metrics.jsonl").read_text().splitlines()]
    tr2 = [r for r in recs2[n_before:] if r["kind"] == "train"]
    st2 = json.loads((rd / "status.json").read_text())
    check("run again, it resumes from its checkpoint (grown) instead of starting over",
          st2["state"] == "done" and all(r.get("depth") == 8 for r in tr2), f"{len(tr2)} new training lines")

    if dev.type == "cuda":
        mg = CoralMini(CoralMiniConfig(**SMALL)).to(dev)
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
        check("the compiled training step runs (fixed shapes)", torch.isfinite(lc).item() and abs(lc.item() - loss.item()) < 0.05,
              f"{lc.item():.4f} vs {loss.item():.4f}")
    tmp_ctx.cleanup()
    term.tests_passed("Coral Mini")


if __name__ == "__main__":
    main()
