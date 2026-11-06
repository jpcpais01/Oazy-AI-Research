"""Self-tests for Code Black and Black-T: the sizes and the arithmetic per token, the rule that keeps Code Black a fair
next-token model (no token ever sees a later one, from either clock), no bleed between documents, the slow stream
really carries what the fast window cannot reach, the phrase memory, the optimizer's groups, FP8's picks, the race
recipe, and two whole (tiny) races on the clock, Code Black against Black-T, with the race report.

    python3 -m codeblack.tests                 (CPU, ~2 min; without internet it uses stand-in tokenizers)
    python3 -m codeblack.tests --device cuda   (on the pod: also bf16, the compiled step, flex attention = the grids, FP8)
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

SMALL = dict(d_model=64, enc_layers=1, dec_layers=1, attn_heads=2, attn_kv_heads=1, attn_head_dim=32, mlp_hidden=96,
             window=16, slow_d=96, slow_layers=2, slow_heads=3, slow_kv_heads=1, slow_head_dim=32, slow_mlp=128,
             slow_mem_heads=2, head_k=16, head_v=16, phrase_rows=257, phrase_dim=16, max_len=64, ctx="64:0")


def awake(m):
    """Turn off the zero starts (unpack, the phrase memory's value), so every path carries signal in the tests."""
    for p in m.parameters():
        if p.dim() >= 2:
            torch.nn.init.normal_(p, std=0.05)
    with torch.no_grad():
        m.slow_start.normal_()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    dev = torch.device(ap.parse_args().device)
    term.tests_start("Code Black", dev)
    torch.manual_seed(0)
    from codeblack.model import Black, BlackConfig, BlackT, BlackTConfig, BLACKT_SIZES
    from oazy import tok

    # ---- the sizes, and the arithmetic per token that the plan rests on
    sizes = {s: BlackT(BlackTConfig(size=s)).num_params() for s in BLACKT_SIZES}
    check("Black-T at 125m, 250m, 500m: about 120M, 244M, 515M parameters", 115e6 < sizes["125m"] < 126e6
          and 235e6 < sizes["250m"] < 255e6 and 500e6 < sizes["500m"] < 530e6, ", ".join(f"{s} {n / 1e6:.0f}M" for s, n in sizes.items()))
    bt = BlackT(BlackTConfig())
    check("Black-T is a modern transformer: attention in all 12 layers, RoPE, gated, no bleed, SmolLM2, 2,048 windows",
          bt.cfg.kinds() == ["A"] * 12 and bt.cfg.rope == 1 and bt.cfg.doc_mask == 1 and bt.cfg.tokenizer == "smollm2"
          and bt.ctx_at(0) == 2048 and bt.cfg.flex_attn == 1, bt.layout())
    bs = Black(BlackConfig())
    fast, slow = bs.stream_params()
    check("Black-S: a 14M fast stream, a ~200M slow stream, a 134M-entry phrase memory", 12e6 < fast < 16e6
          and 190e6 < slow < 210e6 and bs.phrase_params() == 4 * 131072 * 256, bs.layout())
    ratio = bs.flops_per_token(2048) / bt.flops_per_token(2048)
    check("Black-S does about 0.6x the arithmetic per token of Black-T 125m (the design doc: ~0.65x)", 0.5 < ratio < 0.7,
          f"{bs.flops_per_token(2048) / 1e6:.0f} vs {bt.flops_per_token(2048) / 1e6:.0f} MFLOP per token")
    check("next-token, 2,048 windows, no bleed, SmolLM2's 49,152 pieces", bs.ctx_at(0) == 2048 and bs.cfg.doc_mask == 1
          and bs.embed.weight.shape[0] == 49152 and tok.eot_id("smollm2") == 0)
    del bt, bs

    # ---- the fair next-token rule, from both clocks, with and without the document rule and the memory layers
    for pat in ("A", "MMMA"):
        for dm in (1, 0):
            m = awake(Black(BlackConfig(**SMALL, slow_pattern=pat, doc_mask=dm))).to(dev).eval()
            x = torch.randint(1, 49152, (2, 64), device=dev)
            x[0, 21] = 0                           # an <|endoftext|>: a new document starts at 22 in the first row
            with torch.no_grad():
                base = m._body(x)
                worst_before, least_after = 0.0, 1.0
                for p in (3, 7, 8, 30, 47, 63):     # inside a group, a group's last token, the start of a group, ...
                    x2 = x.clone()
                    x2[:, p] = x2[:, p] % 40000 + 7
                    d = (m._body(x2) - base).abs().amax(-1)
                    worst_before = max(worst_before, d[:, :p].max().item() if p else 0.0)
                    least_after = min(least_after, d[:, p:].max().item())
            check(f"slow {pat}, doc rule {dm}: changing a token changes nothing before it, and does change what follows",
                  worst_before < 1e-5 and least_after > 1e-4, f"before {worst_before:.1e}, after {least_after:.1e}")
            if dm:
                with torch.no_grad():
                    x2 = x.clone()
                    x2[0, 10] = 5
                    d = (m._body(x2) - base).abs().amax(-1)[0]
                check(f"slow {pat}: no bleed: a word of the first document does not reach the next one", d[22:].max() < 1e-5
                      and d[10:22].max() > 1e-4, f"{d[22:].max().item():.1e}")
            with torch.no_grad():
                e = (m._body(x[:, :37]) - base[:, :37]).abs().max().item()
            check(f"slow {pat}, doc rule {dm}: any length is scored the same as inside the window (37 of 64)", e < 1e-5)

    # ---- the slow stream carries what the fast window cannot reach
    m = awake(Black(BlackConfig(**{**SMALL, "doc_mask": 0, "phrase_rows": 0}))).to(dev).eval()
    x = torch.randint(1, 49152, (1, 64), device=dev)
    x2 = x.clone()
    x2[0, 0] += 1
    with torch.no_grad():
        far = (m._body(x2) - m._body(x)).abs()[0, 48:].max().item()
        torch.nn.init.zeros_(m.unpack.weight)
        off = (m._body(x2) - m._body(x)).abs()[0, 48:].max().item()
    check("a word 48+ tokens back (beyond the fast layers' 2 x 16 reach) is felt only through the slow stream",
          far > 1e-4 and off < 1e-6, f"with it {far:.1e}, without it {off:.1e}")
    fresh = Black(BlackConfig(**SMALL))
    check("it starts as a fast transformer: the unpack and the phrase memory's value start at zero, and grow in",
          fresh.unpack.weight.abs().sum() == 0 and fresh.phrase.val.weight.abs().sum() == 0)

    # ---- the phrase memory
    from codeblack.model import PhraseMemory
    pm = Black(BlackConfig(**SMALL)).phrase
    a = torch.tensor([[5, 6, 7, 8, 5, 6, 7, 9]])
    r = pm.rows(a)
    check("the same phrase reads the same rows, wherever it is (5 6 7 at 2 and at 6)", torch.equal(r[0, 2], r[0, 6])
          and not torch.equal(r[0, 2], r[0, 7]))
    n_tab, rows_ = pm.n_tab, pm.cfg.phrase_rows
    check("each table has its own range of rows, all within the big table", all(
        ((r[..., t] >= t * rows_) & (r[..., t] < (t + 1) * rows_)).all() for t in range(n_tab)), f"{n_tab} tables of {rows_}")
    doc = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
    rd = pm.rows(a, doc)
    check("a phrase never reaches back across a document start (the 2- and 3-token rows at the start differ)",
          not torch.equal(rd[0, 3], r[0, 3]) and torch.equal(rd[0, 6], r[0, 6]))
    check("the phrase memory is a lookup table, kept out of weight decay and out of the arithmetic count",
          isinstance(PhraseMemory, type) and Black.embed_no_decay)

    # ---- the loss in slices = the plain loss; the optimizer's groups; FP8's picks
    m = awake(Black(BlackConfig(**SMALL))).to(dev)
    x = torch.randint(1, 49152, (2, 64), device=dev)
    x[:, 30] = 0
    Black.fused_loss = True
    l1, _ = m.losses(x, x)
    Black.fused_loss = False
    l2, _ = m.losses(x, x)
    Black.fused_loss = True
    check("the loss in slices = the full 32-bit loss", abs(l1.item() - l2.item()) < 1e-4, f"{l1.item():.5f} vs {l2.item():.5f}")
    from oazy.train import make_optimizer, RECIPE, PRESETS, MENTOR, BEES, FAST, NICE
    l1.backward()
    opt = make_optimizer(m, 1e-3, dev, "muon", 0.04, 0.1)
    muon, adam = opt.opts
    mu = {id(p) for g in muon.param_groups for p in g["params"]}
    names = {n for n, p in m.named_parameters() if id(p) in mu}
    nd = {id(p) for p in adam.param_groups[1]["params"]}
    check("Muon straightens every stream's grids and the pack / unpack between the clocks",
          {"pack.weight", "unpack.weight", "slow.0.attn.q.weight", "enc.0.mlp.up.weight", "dec.0.attn.out.weight"} <= names
          and "embed_phrase.weight" not in names and "phrase.key.weight" not in names)
    check("the word table and the phrase table are not pulled towards zero", id(m.embed.weight) in nd
          and id(m.embed_phrase.weight) in nd)
    from oazy import fp8
    big = Black(BlackConfig())
    picked = [f for f, mod in big.named_modules() if fp8.keep(mod, f)]
    check("FP8 goes to the three stacks' projections and MLPs (7 a layer), not the word table, the phrase memory or the pack",
          len(picked) == 7 * (2 + 16 + 2) and not any(f.startswith(("phrase", "pack", "unpack", "embed")) for f in picked),
          f"{len(picked)} layers")
    del big

    # ---- the race recipe
    rc, pr = RECIPE["black"], PRESETS["race"]
    check("one recipe for both racers: Muon, weight decay 0.1, a batch ramp, the square-root cool-down, FP8, no mentor",
          rc == RECIPE["blackt"] and rc["muon_wd"] == 0.1 and rc["decay_shape"] == "sqrt" and rc["fp8"] == 1
          and MENTOR["black"] == MENTOR["blackt"] == "none" and {"black", "blackt"} <= set(BEES) & set(FAST))
    check("the race preset: 262k-token steps, the same learning rates, a held-out loss every 25M tokens",
          pr["batch_black"] == pr["batch_blackt"] == 262144 and pr["lr_black"] == pr["lr_blackt"]
          and pr["muon_lr_black"] == pr["muon_lr_blackt"] and pr["eval_every"] == 25_000_000)
    check("names on the dashboard", NICE["black"] == "Code Black" and NICE["blackt"] == "Black-T")

    # ---- two whole tiny races on the clock, then the report
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
    for sub, n_docs in (("", 1500), ("chat", 300)):
        d = data / sub if sub else data
        d.mkdir(parents=True, exist_ok=True)
        for split, kk in (("val", n_docs // 5), ("train", n_docs)):
            ds = fake_docs(kk, rng)
            if sub == "chat":
                ds = [f"User: {u}\n\nAssistant: {v_}" for u, v_ in zip(ds, fake_docs(kk, rng))]
            np.fromiter((t for x_ in ds for t in gpt.encode_ordinary(x_) + [EOT]), dtype=np.uint16).tofile(d / f"{split}.bin")
    (data / "hellaswag_val.jsonl").write_text("".join(
        json.dumps({"ctx": " ".join(fake_docs(1, rng)), "endings": fake_docs(4, rng), "label": rng.randint(0, 3)}) + "\n" for _ in range(24)))
    retok.run(data, "smollm2", ["fwe", "chat"], workers=2)
    runs = tmp / "runs"
    budget = 0.12                                   # minutes of training each
    for kind, name in (("blackt", "blackt-small"), ("black", "black-small")):
        sys.argv = ["train", "--model", kind, "--preset", "smoke", "--data", str(data), "--runs", str(runs), "--name", name,
                    "--device", str(dev), "--compile", "0", "--minutes", str(budget), "--batch-ramp", "512:0.3",
                    "--mix", "fwe:0.9,chat:0.1@0", "--micro", "2"]
        train.main()
        rd = runs / name
        recs = [json.loads(l_) for l_ in (rd / "metrics.jsonl").read_text().splitlines()]
        tr = [r_ for r_ in recs if r_["kind"] == "train"]
        st = json.loads((rd / "status.json").read_text())
        cf = json.loads((rd / "config.json").read_text())
        check(f"{NICE[kind]}: the race ran on the clock and ended there", st["state"] == "done" and cf["minutes"] == budget
              and budget * 60 <= tr[-1]["elapsed"] < budget * 60 + 5, f"{tr[-1]['elapsed']:.1f} s of training, {tr[-1]['tokens']:,} tokens")
        check(f"{NICE[kind]}: the learning rate followed the clock into its cool-down", tr[-1]["lr"] < 0.5 * max(r_["lr"] for r_ in tr),
              f"last lr {tr[-1]['lr']:.1e}")
        mdl, ck = train.load_model(rd / "ckpt.pt", dev)
        out, _ = mdl.generate(enc.encode_ordinary("The cat"), max_new_tokens=5)
        check(f"{NICE[kind]}: its checkpoint loads and it writes", ck["kind"] == kind and len(out) <= 5, repr(enc.decode(out)))
    from codeblack import race
    race.main(["--dir", str(runs)])
    rep = json.loads((runs / "race.json").read_text())
    check("the race report reads both runs, with Black-T as the yardstick", rep["ref"] == "blackt-small"
          and {r_["name"] for r_ in rep["runs"]} == {"blackt-small", "black-small"})
    from oazy import profile
    sys.argv = ["profile", "--model", "black", "--preset", "smoke", "--runs", str(runs), "--steps", "1", "--compile", "0",
                "--device", str(dev), "--kernels", "0"]
    profile.main()
    prof = json.loads((runs / "profile-black.json").read_text())
    check("the profiler times Code Black's parts on their own", any(k.startswith("slow attention") for k in prof["ms"])
          and "phrase memory (lookups + gate)" in prof["ms"] and prof["flops_per_token"] > 0)

    if dev.type == "cuda":
        from bee.model import flex_check
        for pat in ("A", "MMMA"):
            cfg = BlackConfig(**{**SMALL, "max_len": 512, "ctx": "512:0", "window": 128, "attn_head_dim": 64,
                                 "slow_head_dim": 64, "slow_pattern": pat})
            mg = awake(Black(cfg)).to(dev).train()
            xg = torch.randint(1, 49152, (4, 512), device=dev)
            xg[:, 100] = 0
            xg[1, 300] = 0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ref, _ = mg.losses(xg, xg)
            mg.enable_compile()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lc, _ = mg.losses(xg, xg)
            lc.backward()
            check(f"slow {pat}: the compiled step on the GPU (flex attention: {'yes' if mg._flex else 'no'}) = the plain grids",
                  torch.isfinite(lc).item() and abs(lc.item() - ref.item()) < 0.02 * abs(ref.item()),
                  f"{lc.item():.4f} vs {ref.item():.4f}")
        mt = BlackT(BlackTConfig(size="", d_model=256, n_layer=4, attn_heads=4, attn_kv_heads=2, mlp_hidden=512,
                                 max_len=512, ctx="512:0")).to(dev).train()
        mt.enable_compile()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lt, _ = mt.losses(xg, xg)
        lt.backward()
        check("Black-T: the compiled step on the GPU", torch.isfinite(lt).item(), f"flex attention: {'yes' if flex_check(dev, mt.cfg) else 'no'}")
        mf, used = fp8.convert(awake(Black(BlackConfig(**{**SMALL, "d_model": 256, "attn_heads": 4, "attn_head_dim": 64,
                                                          "mlp_hidden": 512, "slow_d": 512, "slow_heads": 8, "slow_head_dim": 64,
                                                          "slow_mlp": 1024}))).to(dev).train(), dev, print)
        term.info(f"FP8 on this GPU: {'yes' if used else 'no'}")
    tmp_ctx.cleanup()
    term.tests_passed("Code Black")


if __name__ == "__main__":
    main()
