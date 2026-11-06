"""Self-tests for Coral: the size (1.0B total, ~160M at work per word), the experts (fixed-capacity training =
dropless reading when nothing overflows; gradients reach every part), the balancing biases, Muon on the expert
stacks, the recipe, growing, the hybrid switch, and one whole (tiny) run end to end.

    python -m coral.tests                 (CPU, ~2 min; without internet it uses stand-in tokenizers)
    python -m coral.tests --device cuda   (on the pod: also bf16, the compiled step, flex attention, FP8)
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

SMALL = dict(d_model=64, n_layer=4, n_head=2, head_k=16, head_v=16, attn_heads=4, attn_kv_heads=2, attn_head_dim=16,
             mlp_hidden=32, max_len=128, ctx="128:0", n_expert=8, top_k=2, expert_hidden=32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    dev = torch.device(ap.parse_args().device)
    term.tests_start("Coral", dev)
    torch.manual_seed(0)
    from coral.model import Coral, CoralConfig, Experts
    from oazy import tok

    # ---- the size
    big = Coral(CoralConfig())
    n, act = big.full_params(), big.active_params()
    check("about 1.0B parameters once grown", 1.00e9 < n < 1.07e9, f"{n / 1e6:.0f}M, {big.layout()}")
    check("about 160M at work per word (2 of 32 experts + the shared MLP + attention + the word table)",
          190e6 < act < 225e6 and act - big.embed.weight.numel() < 175e6,
          f"{act / 1e6:.0f}M at work, {(act - big.embed.weight.numel()) / 1e6:.0f}M without the word table")
    check("all attention, with RoPE (nothing else tells it the order); 2,048 windows, SmolLM2",
          big.cfg.kinds() == ["A"] * 20 and big.cfg.rope == 1 and big.ctx_at(0) == 2048 and big.cfg.doc_mask == 1
          and big.cfg.tokenizer == "smollm2" and tok.eot_id("smollm2") == 0)
    _x = torch.tensor([[5, 6, 0, 7, 8, 9]])
    _r, _m, _ = big.doc_info(_x)
    check("attention runs plain causal on the fused kernel (attn_docs=0); the mentor still reads each document alone",
          big.cfg.attn_docs == 0 and big._attn_mask(_x, _r, _m) is None and _m is not None and not bool(_m[0, 0, 3, 1]))
    check("it starts 8 layers deep and grows to 12 at 10%, 20 at 25%", big.depth == 8 and big.depth_at(0.099) == 8
          and big.depth_at(0.1) == 12 and big.depth_at(0.25) == 20)
    from oazy import fp8
    picked = [f for f, mod in big.named_modules() if fp8.keep(mod, f)]
    check("FP8 goes to attention and the shared MLP (7 products a layer), not the router or the word table",
          len(picked) == 8 * 7 and not any("router" in f for f in picked), f"{len(picked)} layers at depth 8")
    hyb = Coral(CoralConfig(pattern="MMMA"))
    check("pattern=MMMA: the Memtrix hybrid (15 memory + 5 attention, no RoPE)", hyb.cfg.kinds().count("M") == 15
          and hyb.cfg.rope == 0, hyb.layout())
    del big, hyb

    # ---- the experts
    cfg = CoralConfig(**SMALL, capacity=8.0)       # room for everyone: nothing overflows
    ex = Experts(cfg).to(dev)
    h = torch.randn(3, 40, cfg.d_model, device=dev)
    ex.train()
    y_cap = ex(h)
    ex.eval()
    y_all = ex(h)
    e = (y_cap - y_all).abs().max().item()
    check("fixed-capacity experts (training) = one expert at a time (reading), when nothing overflows", e < 1e-5, f"max diff {e:.1e}")
    ex.train()
    ex.balance()                                   # start the count afresh
    ex.cfg.capacity = 0.5                          # half the room: some words overflow
    y_tight = ex(h)
    load, dropped = ex.balance()
    check("with too little room some word-expert pairs overflow, and are counted", 0 < dropped[0] < dropped[1]
          and load.sum().item() == 3 * 40 * 2 and torch.isfinite(y_tight).all().item(),
          f"{int(dropped[0])} of {int(dropped[1])} overflowed")
    ex.cfg.capacity = 1.25
    ex.zero_grad()
    out, aux = ex(h, with_aux=True)
    (out.square().mean() + aux).backward()
    check("gradients reach the router and every expert grid", ex.router.weight.grad.abs().sum() > 0
          and (ex.w_up.grad.abs().sum((1, 2)) > 0).all() and (ex.w_down.grad.abs().sum((1, 2)) > 0).all())
    # balancing: an expert the router loves gets its bias pushed down until the work evens out
    ex2 = Experts(CoralConfig(**SMALL, balance_rate=0.05)).to(dev)
    with torch.no_grad():
        ex2.router.weight.normal_(0, 0.1)
        ex2.router.weight[0] += 0.5                 # almost every word prefers expert 0
    x = torch.randn(1, 512, cfg.d_model, device=dev)
    ex2.train()
    first = None
    for i in range(80):
        ex2(x)
        load, _ = ex2.balance()
        u = (load.max() / load.mean()).item()
        first = u if first is None else first
    check("balancing biases even the work out without a loss (DeepSeek-V3)", u < first and u < 2.0 and ex2.bias[0] < 0,
          f"busiest / average {first:.2f} -> {u:.2f}")

    # ---- the model: training trunk = reading trunk; the balance loss; Muon on the stacks
    m = Coral(CoralConfig(**SMALL, capacity=8.0)).to(dev)
    x = torch.randint(1, 49152, (2, 64), device=dev)
    x[:, 30] = 0
    m.train()
    lt, st = m.losses(x, x)
    m.eval()
    le, _ = m.losses(x, x)
    m.train()
    check("training loss (fixed capacity, + 1e-4 x balance) = reading loss (dropless) + the balance term",
          abs(lt.item() - le.item() - m.cfg.aux_weight * float(st["moe_aux"]) * len(m.blocks)) < 1e-4 and "moe_aux" in st,
          f"{lt.item():.4f} vs {le.item():.4f}")
    lt.backward()
    from oazy.train import make_optimizer, RECIPE, PRESETS, MENTOR, BEES, FAST
    opt = make_optimizer(m, 1e-3, dev, "muon", 0.03, 0.1)
    muon = opt.opts[0]
    stacks = [p for g in muon.param_groups for p in g["params"] if p.dim() == 3]
    before = [p.detach().clone() for p in stacks]
    opt.step()
    m.after_step()
    check("Muon straightens every expert's grids (the stacks), the router stays on AdamW",
          len(stacks) == 2 * len(m.blocks) and all((p - b).abs().sum() > 0 for p, b in zip(stacks, before))
          and not any(p is m.blocks[0].moe.router.weight for g in muon.param_groups for p in g["params"]))
    rep = m.moe_report()
    check("the log line gets the experts' evenness and overflow", set(rep) == {"moe_uneven", "moe_dropped"} and rep["moe_uneven"] >= 1)
    _, _, infos = m.inspect(x[:1, :5])
    check("reading with the lid off shows which 2 experts each layer picked", all(len(i["experts"]) == 2 for i in infos))
    # growing keeps the experts' biases with the copies
    g = Coral(CoralConfig(**{**SMALL, "n_layer": 8, "grow": "4:0,8:0.5"}))
    g.blocks[3].moe.bias.fill_(0.3)
    g.grow_to(0.5)
    check("it grows 4 -> 8 by copying the top layers, experts and their biases included", g.depth == 8
          and abs(g.blocks[7].moe.bias[0].item() - 0.3) < 1e-6)

    # ---- the recipe
    from ember.mentor import fade
    rc = RECIPE["coral"]
    check("Frog's recipe; the mentor on 1 pass in 4 at the start, fading to none by 30%; its own 20,000M-token Tagus",
          rc["mentor_share"] == 1 and rc["fp8"] == 1 and rc["mentor_weight"] == 0.25 and rc["mentor_until"] == 0.3
          and fade(0.0, 0.25, 0.3, None, 0, 0) == 0.25 and fade(0.3, 0.25, 0.3, None, 0, 0) == 0
          and rc["data_sub"] == "tagus20" and RECIPE["frog"].get("data_sub") is None)
    pb = PRESETS["big"]
    check("big preset: 1M-token steps, learning rates 1.8e-3 / Muon 0.03", pb["batch_coral"] == 1048576
          and pb["lr_coral"] == 1.8e-3 and pb["muon_lr_coral"] == 0.03)
    check("SmolLM2-360M mentor, raised the Bee way", MENTOR["coral"] == "smollm2-360m" and "coral" in BEES and "coral" in FAST)

    # ---- Tagus at 20,000M: the held-out documents are kept out of training
    from oazy import tagus
    keys = {tagus._key("held out text one"), tagus._key("two")}
    kept = [t for b in tagus.without(iter([["held out text one ", "fresh"], ["two"], ["more", "fresh 2"]]), keys) for t in b]
    check("Tagus --into: held-out documents met in the stream are left out", kept == ["fresh", "more", "fresh 2"])
    need = tagus.needs(20e9)
    check("a 20,000M-token run reads about 21,000M tokens of Tagus (5% margin)", 20.9e9 < sum(need.values()) < 21.1e9,
          ", ".join(f"{k} {v / 1e6:,.0f}M" for k, v in sorted(need.items(), key=lambda kv: -kv[1])))

    # ---- one whole tiny run on stand-in text: Coral's recipe end to end
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
    argv = ["train", "--model", "coral", "--preset", "smoke", "--data", str(data), "--runs", str(runs), "--mentor", "smol-random",
            "--device", str(dev), "--compile", "0", "--tokens", "60000", "--batch-ramp", "512:0.5", "--mix", "fwe:0.9,chat:0.1@0",
            "--micro", "2", "--set", "n_layer=8", "grow=4:0,8:0.4"]
    sys.argv = argv
    try:
        train.main()
        missing = False
    except SystemExit as e:
        missing = "tagus20" in str(e)
    check("by default it reads its own Tagus (tagus20) and says how to build it when it is missing", missing)
    sys.argv = argv + ["--data-sub", "smol"]
    train.main()
    rd = runs / "coral"
    recs = [json.loads(l_) for l_ in (rd / "metrics.jsonl").read_text().splitlines()]
    tr = [r for r in recs if r["kind"] == "train"]
    cfg_run = json.loads((rd / "config.json").read_text())
    check("the run used Coral's recipe", cfg_run["mentor_share"] is True and cfg_run["mentor_weight"] == 0.25
          and cfg_run["mentor_until"] == 0.3 and cfg_run["params_active"] < cfg_run["params"], cfg_run["layout"])
    check("the experts' evenness and overflow were logged", all("moe_uneven" in r and "moe_dropped" in r for r in tr))
    mws = [r.get("mentor_w", 0) for r in tr]
    check("the mentor taught early and left by 30%", 0.1 < mws[0] <= 0.25 and mws[-1] == 0, f"{mws[0]:.2f} ... {mws[-1]:.2f}")
    st = json.loads((rd / "status.json").read_text())
    check("the run finished", st["state"] == "done", f"val loss {st['latest']['val_loss']:.3f}")
    mdl, ck = train.load_model(rd / "ckpt.pt", dev)
    check("its checkpoint loads as Coral, balancing biases included", ck["kind"] == "coral" and type(mdl).__name__ == "Coral"
          and any(b.moe.bias.abs().sum() > 0 for b in mdl.blocks))
    deps = [r.get("depth") for r in tr]
    check("it grew inside the run (4 -> 8 layers at 40%)", deps[0] == 4 and deps[-1] == 8 and mdl.depth == 8)
    out, _ = mdl.generate(enc.encode_ordinary("The cat"), max_new_tokens=5)
    check("it writes", len(out) <= 5, repr(enc.decode(out)))

    if dev.type == "cuda":
        from bee.model import flex_check
        mg = Coral(CoralConfig(**SMALL)).to(dev)
        xg = torch.randint(1, 49152, (4, 256), device=dev)
        xg[:, 50] = 0
        mg.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = mg.losses(xg, xg)
        loss.backward()
        check("bf16 training step on the GPU with the document rule", torch.isfinite(loss).item())
        mg.zero_grad(set_to_none=True)
        mg.enable_compile()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lc, _ = mg.losses(xg, xg)
        lc.backward()
        check("the compiled training step runs (fixed shapes, one graph per depth)", torch.isfinite(lc).item(),
              f"flex attention: {'yes' if flex_check(dev, mg.cfg) else 'no'}")
        mf, used = fp8.convert(Coral(CoralConfig(**{**SMALL, "d_model": 256, "attn_heads": 4, "attn_head_dim": 64,
                                                    "mlp_hidden": 256})).to(dev).train(), dev, print)
        term.info(f"FP8 on this GPU: {'yes' if used else 'no'}")
    tmp_ctx.cleanup()
    term.tests_passed("Coral")


if __name__ == "__main__":
    main()
