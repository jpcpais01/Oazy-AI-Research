"""Self-tests for SmolLM2 from scratch: the real sizes (134.5M / 361.8M), random weights, the loss both ways (stock: the
full score grid; ours: in slices, with the mentor) against Hugging Face's own loss, the two optimizers, a few steps that
overfit one batch, the recipes, and one whole tiny run of each name end to end with a resume.

    python3 -m smol.tests                 (CPU, ~1-2 min; without internet it uses stand-in tokenizers)
    python3 -m smol.tests --device cuda   (on the pod: also bf16 and the compiled step)
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

SMALL = dict(d_model=64, n_layer=3, attn_heads=4, attn_kv_heads=2, mlp_hidden=128, max_len=128, ctx="128:0",
             vocab_size=512, hub=0, rope_theta=10000.0, init_std=0.02)


def check(name, ok, detail=""):
    term.check(name, ok, detail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    dev = torch.device(ap.parse_args().device)
    term.tests_start("SmolLM2 from scratch", dev)
    torch.manual_seed(0)
    from smol.model import Smol, SmolConfig

    # ---- the sizes: exactly SmolLM2's
    for size, want in (("135m", 134_515_008), ("360m", 361_821_120)):
        m = Smol(SmolConfig(size=size, hub=0))
        n = m.num_params()
        check(f"SmolLM2-{size}'s body, {want / 1e6:.1f}M parameters (the tied word table once)", n == want,
              f"{n:,}; {m.layout()}")
        if size == "135m":
            std = m.hf.model.layers[0].mlp.up_proj.weight.std().item()
            check("random weights, Hugging Face's own start (normal, std = 1/24)", abs(std - 1 / 24) < 0.002, f"std {std:.4f}")
            check("the word table is tied to the score head", m.hf.lm_head.weight.data_ptr() == m.embed.weight.data_ptr())
        del m
    c = SmolConfig()
    check("stock: 2,048-word windows throughout; ours: 512 -> 1,024 -> 2,048",
          c.ctx == "2048:0" and SmolConfig(stock=0).ctx == "512:0,1024:0.15,2048:0.3")

    # ---- the loss, both ways, against LlamaForCausalLM's own
    m = Smol(SmolConfig(**SMALL)).to(dev).train()
    x = torch.randint(1, 512, (2, 32), device=dev)
    y = torch.randint(1, 512, (2, 32), device=dev)
    ref = m.hf(input_ids=x, use_cache=False).logits.float()
    ce_ref = F.cross_entropy(ref.reshape(-1, 512), y.reshape(-1))
    ls, _ = m.losses(x, y)
    m.cfg.stock = 0
    lf, st = m.losses(x, y)
    check("the stock loss (full scores) and ours (in slices) both equal Hugging Face's", abs(ls.item() - ce_ref.item()) < 1e-4
          and abs(lf.item() - ce_ref.item()) < 1e-4, f"{ls.item():.5f} / {lf.item():.5f} vs {ce_ref.item():.5f}")
    lf.backward()
    g1 = m.embed.weight.grad.clone()
    m.zero_grad(set_to_none=True)
    type(m).fused_loss = False
    lp, _ = m.losses(x, y)
    lp.backward()
    type(m).fused_loss = True
    check("slices give the same gradients as the full grid", torch.allclose(g1, m.embed.weight.grad, atol=1e-5))
    m.zero_grad(set_to_none=True)

    class FakeMentor:                       # the mentor's interface: hidden() -> (vectors, word table)
        def __init__(s):
            s.W = torch.randn(512, 16, device=dev)
            s.P = torch.randn(64, 16, device=dev) * 0.1

        def hidden(s, idx, mask=None):
            return m.embed(idx).detach() @ s.P, s.W
    me = FakeMentor()
    lk, stk = m.losses(x, y, me, 0.5)
    th, tW = me.hidden(x)
    from ember.mentor import mentor_loss
    kd_ref = mentor_loss(m._body(x), F.linear(th, tW))
    check("ours: the mentor's lesson mixes in (KL in slices = the slow way)", "kd" in stk and abs(stk["kd"].item() - kd_ref.item()) < 1e-3
          and abs(lk.item() - (0.5 * ce_ref.item() + 0.5 * kd_ref.item())) < 1e-3, f"KL {stk['kd'].item():.4f} vs {kd_ref.item():.4f}")

    # ---- optimizers
    from oazy.muon import Combined
    oa = m.make_optimizer(1e-3, dev, "adamw")
    grp = {round(g["weight_decay"], 4): sum(p.numel() for p in g["params"]) for g in oa.param_groups}
    check("stock: AdamW, weight decay 0.01 on every grid, none on the norms", isinstance(oa, torch.optim.AdamW)
          and set(grp) == {0.01, 0.0} and sum(grp.values()) == m.num_params(), str(grp))
    om = m.make_optimizer(1e-3, dev, "muon", 0.03, 0.1)
    n_mu = sum(p.numel() for p in om.opts[0].param_groups[0]["params"])
    n_all = sum(p.numel() for g in om.param_groups for p in g["params"])
    check("ours: Muon on the attention and MLP grids, AdamW for the word table and norms, each parameter once",
          isinstance(om, Combined) and n_all == m.num_params() and n_mu == m.num_params(non_embedding=True) - 64 * (2 * 3 + 1),
          f"Muon {n_mu:,}, all {n_all:,}")
    for kind, lr in (("adamw", 3e-3), ("muon", 3e-3)):
        torch.manual_seed(1)
        mm = Smol(SmolConfig(**{**SMALL, "stock": int(kind == "adamw")})).to(dev).train()
        o = mm.make_optimizer(lr, dev, kind, 0.03, 0.0)
        first = None
        for _ in range(30):
            loss, _ = mm.losses(x, y)
            loss.backward()
            o.step()
            o.zero_grad(set_to_none=True)
            first = first or loss.item()
        check(f"{kind}: it learns one batch by heart", loss.item() < first * 0.5, f"{first:.2f} -> {loss.item():.2f}")
    mm.eval()
    out, _ = mm.generate([1, 2, 3], max_new_tokens=6, top_k=5)
    check("it writes word by word (the key-value cache)", len(out) == 6)

    # ---- the recipes
    from oazy.train import RECIPE, PRESETS, MENTOR, BEES, FAST, NICE
    rb = PRESETS["big"]
    check("smol = their recipe: AdamW 3e-3, no mentor, no ramp, plain WSD, Sado", "smol" in BEES and "smol" not in FAST
          and rb["lr_smol"] == 3e-3 and RECIPE["smol"] == {"data_sub": "sado"})
    check("smolbee = our latest recipe (Bee5's), only the body differs", RECIPE["smolbee"] == RECIPE["bee5"]
          and "smolbee" in FAST and MENTOR["smolbee"] == "smollm2-360m" and rb["lr_smolbee"] == rb["lr_bee5"]
          and rb["muon_lr_smolbee"] == rb["muon_lr_bee5"] and rb["batch_smolbee"] == rb["batch_bee5"] == rb["batch_smol"]
          and "smol" in NICE)

    # ---- one whole tiny run of each, on stand-in text, then a resume
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
    for kind in ("smol", "smolbee"):
        argv = ["train", "--model", kind, "--preset", "smoke", "--data", str(data), "--runs", str(runs), "--device", str(dev),
                "--compile", "0", "--tokens", "60000", "--mix", "fwe:0.9,chat:0.1@0", "--micro", "2", "--data-sub", "smol"]
        if kind == "smolbee":
            argv += ["--mentor", "smol-random", "--batch-ramp", "512:0.5", "--set", "ctx=64:0,128:0.3"]
        sys.argv = argv
        train.main()
        rd = runs / kind
        recs = [json.loads(l_) for l_ in (rd / "metrics.jsonl").read_text().splitlines()]
        tr = [r for r in recs if r["kind"] == "train"]
        cfg_run = json.loads((rd / "config.json").read_text())
        st = json.loads((rd / "status.json").read_text())
        check(f"{kind}: the run finished, with held-out loss and HellaSwag", st["state"] == "done" and "val_loss" in st["latest"],
              f"val loss {st['latest'].get('val_loss', float('nan')):.3f}; {cfg_run['layout']}")
        if kind == "smol":
            check("smol: AdamW, no mentor, 128-word windows throughout", cfg_run["optimizer"] == "adamw"
                  and cfg_run["mentor"] == "none" and {r.get("context") for r in tr} == {128})
        else:
            mws = [r.get("mentor_w", 0) for r in tr]
            ctxs = [r.get("context") for r in tr]
            check("smolbee: Muon, the mentor teaching, the windows growing (64 -> 128)", cfg_run["optimizer"] == "muon"
                  and cfg_run["mentor_share"] is True and mws[0] > 0.15 and ctxs[0] == 64 and ctxs[-1] == 128,
                  f"mentor {mws[0]:.2f} ... {mws[-1]:.2f}")
        mdl, ck = train.load_model(rd / "ckpt.pt", dev)
        check(f"{kind}: its checkpoint loads back", ck["kind"] == kind and type(mdl).__name__ == "Smol"
              and mdl.cfg.stock == int(kind == "smol"))
        out, _ = mdl.generate(enc.encode_ordinary("The cat"), max_new_tokens=5)
        check(f"{kind}: the trained model writes", len(out) <= 5, repr(enc.decode(out)))
        n_before = len(recs)
        sys.argv = argv
        train.main()
        recs2 = [json.loads(l_) for l_ in (rd / "metrics.jsonl").read_text().splitlines()]
        st2 = json.loads((rd / "status.json").read_text())
        check(f"{kind}: run again, it picks up its checkpoint instead of starting over",
              st2["state"] == "done" and len(recs2) == n_before and st2["tokens"] == st["tokens"])

    if dev.type == "cuda":
        for stock in (1, 0):
            mg = Smol(SmolConfig(**{**SMALL, "vocab_size": 49152, "stock": stock})).to(dev).train()
            xg = torch.randint(1, 49152, (4, 128), device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = mg.losses(xg, xg)
            loss.backward()
            mg.zero_grad(set_to_none=True)
            mg.enable_compile()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lc, _ = mg.losses(xg, xg)
            lc.backward()
            check(f"{'stock' if stock else 'ours'}: bf16 and the compiled step on the GPU agree",
                  torch.isfinite(lc).item() and abs(lc.item() - loss.item()) < 0.05, f"{lc.item():.4f} vs {loss.item():.4f}")
    tmp_ctx.cleanup()
    term.tests_passed("SmolLM2 from scratch")


if __name__ == "__main__":
    main()
