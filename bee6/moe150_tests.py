"""Self-tests for MoE-150M and Dense-30M (bee6/moe150.py): their sizes, the same work per word, the Opal mix they read,
one whole tiny run of each through the trainer (with a resume for the MoE), and on a GPU their real speed at full size.

    python3 -m bee6.moe150_tests                  (CPU, ~1-2 min)
    python3 -m bee6.moe150_tests --device cuda    (on the pod: also tokens per second at full size, compiled)
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


def check(name, ok, detail=""):
    term.check(name, ok, detail)


def tables(m):
    return sum(p.numel() for n, p in m.named_parameters() if n.startswith(("embed", "ve.")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    dev = torch.device(ap.parse_args().device)
    term.tests_start("MoE-150M", dev)
    from bee6.model import Bee6, Bee6Config
    from bee6.moe import Bee6MoE, Bee6MoEConfig
    from oazy.train import MOE150, DENSE30, RECIPE
    from oazy.mix import resolve_mix, OPAL100_TEXT_MIX

    moe = Bee6MoE(Bee6MoEConfig(**MOE150))
    den = Bee6(Bee6Config(**DENSE30))
    n, a, t = moe.num_params(), moe.active_params(), tables(moe)
    check("MoE-150M: ~150M weights in all, ~30M of its layers at work per word",
          145e6 < n < 155e6 and 28e6 < a - t < 31e6, f"{n / 1e6:.1f}M in all, {(a - t) / 1e6:.1f}M layers per word, {a / 1e6:.1f}M counting tables")
    nd, td = den.num_params(), tables(den)
    check("Dense-30M: the same body and the same work per word as the MoE (within 1%)",
          abs((nd - td) - (a - t)) / (a - t) < 0.01, f"{nd / 1e6:.1f}M in all, {(nd - td) / 1e6:.1f}M layers")
    c = moe.cfg
    check("only the clean tricks: no FP8, no Mixture of Depths, no bloom; MTP gone by 5%",
          c.fp8 == 0 and c.mod_layers == "" and not c.bloom and c.mtp_until == 0.05 and c.fuse_qkv == 1 and c.loss_chunk == 4096)
    check("the recipe is Bee6's", RECIPE["moe150"] == RECIPE["bee6"] == RECIPE["dense30"])
    mix = resolve_mix("opal100:flat:0.02")
    phases = [dict((k, float(v)) for k, v in (s.split(":") for s in p.split("@")[0].split(","))) for p in mix.split(";")]
    check("Opal for it: Field1 2% of every phase, chat 12% of the cool-down text, every phase adds to 1",
          all(abs(p["field1"] - 0.02) < 1e-9 and abs(sum(p.values()) - 1) < 1e-4 for p in phases)
          and abs(phases[-1]["chat"] - 0.12 * 0.98) < 1e-4 and "@0.8" in OPAL100_TEXT_MIX, mix[:90] + "…")

    torch.manual_seed(0)
    for p in moe.parameters():           # a jiggled MoE-150M: reading is causal at full shape (a short window)
        with torch.no_grad():
            p.add_(torch.randn_like(p) * 0.02)
    moe.eval()
    x = torch.randint(1, 49152, (1, 64))
    y = x.clone()
    y[0, 40:] = torch.randint(1, 49152, (24,))
    with torch.no_grad():
        l1, l2 = moe(x)[0], moe(y)[0]
    check("full-size MoE-150M reads causally (changing words 40+ leaves 0..39 the same)",
          torch.allclose(l1[:, :40].float(), l2[:, :40].float(), atol=1e-4))
    del moe, den

    # ---- a whole tiny run of each through the trainer
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
    for sub, nd_ in (("", 1500), ("chat", 300)):
        d = data / sub if sub else data
        d.mkdir(parents=True, exist_ok=True)
        for split, kk in (("val", nd_ // 5), ("train", nd_)):
            ds = fake_docs(kk, rng)
            np.fromiter((t_ for x_ in ds for t_ in gpt.encode_ordinary(x_) + [EOT]), dtype=np.uint16).tofile(d / f"{split}.bin")
    (data / "hellaswag_val.jsonl").write_text("".join(
        json.dumps({"ctx": " ".join(fake_docs(1, rng)), "endings": fake_docs(4, rng), "label": rng.randint(0, 3)}) + "\n" for _ in range(24)))
    retok.run(data, "smollm2", ["fwe", "chat"], workers=2)
    runs = tmp / "runs"
    for kind in ("moe150", "dense30"):
        argv = ["train", "--model", kind, "--preset", "smoke", "--data", str(data), "--runs", str(runs), "--mentor", "none",
                "--device", str(dev), "--compile", "0", "--tokens", "40000", "--batch-ramp", "512:0.5", "--mix", "fwe:0.9,chat:0.1@0",
                "--micro", "2", "--data-sub", "smol", "--set", "ctx=64:0,128:0.3", "attn_head_dim=16", "attn_heads=4", "attn_kv_heads=2"]
        sys.argv = argv
        train.main()
        rd = runs / kind
        recs = [json.loads(l_) for l_ in (rd / "metrics.jsonl").read_text().splitlines()]
        st = json.loads((rd / "status.json").read_text())
        mdl, ck = train.load_model(rd / "ckpt.pt", dev)
        check(f"{kind}: a whole tiny run trains, finishes with held-out loss, and its checkpoint loads as {kind}",
              st["state"] == "done" and "val_loss" in st["latest"] and ck["kind"] == kind
              and type(mdl).__name__ == ("Bee6MoE" if kind == "moe150" else "Bee6"),
              f"val loss {st['latest'].get('val_loss', float('nan')):.3f}")
        if kind == "moe150":
            tr = [r for r in recs if r["kind"] == "train"]
            check("the MoE's log shows its experts", "experts_uneven" in tr[-1], f"uneven {tr[-1].get('experts_uneven', 0):.2f}")
            n_before = len(recs)
            sys.argv = argv
            train.main()
            st2 = json.loads((rd / "status.json").read_text())
            check("run again, it picks up its checkpoint",
                  len((rd / "metrics.jsonl").read_text().splitlines()) == n_before and st2["tokens"] == st["tokens"])
    tmp_ctx.cleanup()
    if dev.type == "cuda":
        speed(dev)
    term.tests_passed("MoE-150M")


def speed(dev, micro=16, T=2048, steps=6):
    """Full size, compiled, forward + back at 2,048 words: Dense-30M vs MoE-150M (no optimizer, no tests)."""
    from bee6.model import Bee6, Bee6Config
    from bee6.moe import Bee6MoE, Bee6MoEConfig
    from oazy.train import MOE150, DENSE30
    res = {}
    for name, cls, Cfg, kw in (("Dense-30M", Bee6, Bee6Config, DENSE30), ("MoE-150M", Bee6MoE, Bee6MoEConfig, MOE150)):
        torch.manual_seed(0)
        m = cls(Cfg(**{**kw, "ctx": f"{T}:0"})).to(dev).train()
        m.startup_check(dev, lambda s_: None)
        m.set_progress(0.5)
        m.enable_compile()
        x = torch.randint(1, 49152, (micro, T), device=dev)
        torch.cuda.reset_peak_memory_stats()
        for i in range(steps + 3):
            if i == 3:
                torch.cuda.synchronize()
                t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = m.losses(x, x)
            loss.backward()
            m.zero_grad(set_to_none=True)
            m.after_step()
        torch.cuda.synchronize()
        res[name] = micro * T * steps / (time.time() - t0)
        mem = torch.cuda.max_memory_allocated() / 1e9
        term.info(f"{name}: {res[name] / 1e3:.0f}k tokens/s forward + back, peak memory {mem:.0f} GB at micro {micro}; "
                  f"100B tokens ~{100e9 / res[name] / 3600 * 1.1:.0f} h (with the optimizer and tests, +~10%)")
        del m, x
        torch.cuda.empty_cache()
    check("speed at full size: MoE-150M vs Dense-30M", res["MoE-150M"] > 0, f"{res['MoE-150M'] / res['Dense-30M']:.2f}x")


if __name__ == "__main__":
    main()
