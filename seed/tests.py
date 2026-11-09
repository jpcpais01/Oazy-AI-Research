"""Self-tests for Seed: the size (core, library, word table), the bloom's plan (roles fixed by the grown core's
positions, U-net pairs kept), every growth keeping the function exactly (scores, the training loss and word-by-word
reading before vs after), a grown checkpoint loading into a fresh seed, the optimizer (every parameter in exactly one
group, Grokfast and the extra decay on the core only, library rows that were not read stay still, the optimizer's
memory carried through a growth), the library probe, and one whole tiny run that grows three times, then a resume.

    python3 -m seed.tests                 (CPU, ~1-2 min; without internet it uses stand-in tokenizers)
    python3 -m seed.tests --device cuda   (on the pod: also bf16, the compiled step, a growth on the GPU)
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

from oazy import term

SMALL = dict(vocab_size=512, d_model=160, n_layer=12, bloom="4x64:0,6x96:0.2,9x128:0.4,12x160:0.6", attn_head_dim=16,
             max_len=128, ctx="128:0", lib_side=16, lib_dim=48, lib_heads=2, lib_k=4, lib_qdim=16)


def check(name, ok, detail=""):
    term.check(name, ok, detail)


def logits(m, x):
    was = m.training
    m.eval()
    with torch.no_grad():
        z = m(x)[0]
    m.train(was)
    return z


def train_loss(m, x, y):
    """The training loss (the early 3-word lesson on), leaving the library questions' running averages alone."""
    bns = [b.look.qbn for b in m.blocks if b.look is not None]
    saved = [(bn.running_mean.clone(), bn.running_var.clone(), bn.num_batches_tracked.clone()) for bn in bns]
    loss = m.losses(x, y)[0].item()
    for bn, (rm, rv, nb) in zip(bns, saved):
        bn.running_mean.copy_(rm)
        bn.running_var.copy_(rv)
        bn.num_batches_tracked.copy_(nb)
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    dev = torch.device(ap.parse_args().device)
    term.tests_start("Seed", dev)
    torch.manual_seed(0)
    from seed.model import Seed, SeedConfig, bloom_plan, insert_plan, roles
    from seed import optim as O
    from oazy.train import carry_state

    # ---- the size
    with torch.device("meta"):
        big = Seed(SeedConfig(), stage=3)
    c = big.cfg
    core = big.core_params()
    look = sum(p.numel() for n, p in big.named_parameters() if ".look." in n)
    check("the grown core: 12 layers of 768, 12 query / 6 key-value heads of 64, ReLU² 2,688, ~71M (+ ~4M for the two readers)",
          big.depth == 12 and big.width == 768 and c.dims(768) == (12, 6, 2688) and 69e6 < core - look < 73e6,
          f"layers {(core - look) / 1e6:.1f}M, library readers {look / 1e6:.2f}M")
    check("the library: 1,152 x 1,152 = 1,327,104 rows of 768, about 1.0B weights, full size from day one",
          big.lib.rows() == 1_327_104 and 1.0e9 < big.library_params() < 1.03e9, f"{big.library_params() / 1e6:.1f}M")
    check("the word table is SmolLM2's 49,152 pieces x 768 (37.7M), tied to the score head",
          big.embed.weight.shape == (49152, 768), f"{big.embed.weight.numel() / 1e6:.1f}M")
    n, act = big.full_params(), big.active_params()
    check("about 1.2B in all, about 110M at work for each word", 1.1e9 < n < 1.25e9 and 100e6 < act < 120e6,
          f"{n / 1e6:.0f}M in all, {act / 1e6:.1f}M at work")
    with torch.device("meta"):
        s0 = Seed(SeedConfig())
    check("it starts as a 4 x 384 seed with the whole library", s0.depth == 4 and s0.width == 384
          and s0.lib.rows() == big.lib.rows(), s0.layout())
    check("windows 512 -> 1,024 -> 2,048, SmolLM2's pieces (Bee5's)", s0.ctx_at(0) == 512 and s0.ctx_at(0.15) == 1024
          and s0.ctx_at(0.3) == 2048 and c.tokenizer == "smollm2")
    check("the bloom: 4x384 for 20%, 6x512 for 20%, 9x640 for 20%, 12x768 for the last 40%",
          c.stages() == [(4, 384, 0.0), (6, 512, 0.2), (9, 640, 0.4), (12, 768, 0.6)]
          and [s0.stage_at(p) for p in (0, 0.19, 0.2, 0.39, 0.4, 0.59, 0.6, 1.0)] == [0, 0, 1, 1, 2, 2, 3, 3])
    with torch.device("meta"):
        h = Seed(SeedConfig(hold=1))
    check("--set hold=1 (the side test): it stays the 4 x 384 seed with the whole library all run",
          h.stage_at(0.99) == 0 and h.full_params() == h.num_params() and "held" in h.layout(),
          f"{h.full_params() / 1e6:.0f}M, {h.active_params() / 1e6:.0f}M at work")
    del big, s0, h

    # ---- the plan
    orders, sources, final = bloom_plan(SeedConfig())
    r = roles(SeedConfig())
    at = {final[lab]: v for lab, v in r.items()}
    check("roles by grown position (Bee5's pattern): value tables 1-2 and 11-12, shifted keys 4, 8, 12, back-out 8, "
          "library read at 5 and 8",
          [p for p in range(12) if at[p]["ve"] is not None] == [0, 1, 10, 11]
          and [p for p in range(12) if at[p]["offset"]] == [3, 7, 11]
          and [p for p in range(12) if at[p]["backout"]] == [7] and [p for p in range(12) if at[p]["lib"]] == [4, 7])
    check("the library readers and the back-out are seed layers (there from day one)",
          all(lab < 4 for lab, v in r.items() if v["lib"] or v["backout"]), f"grown order {orders[-1]}")
    ok = True
    for nn_ in range(2, 13):
        for k in range(0, 6):
            seq = insert_plan(nn_, k)
            new = [("o", i) if kd == "old" else ("n", j) for j, (kd, i) in enumerate(seq)]
            n2 = nn_ + k
            for i in range(n2 // 2):
                a, b = new[i], new[n2 - 1 - i]
                if a[0] == "o" and b[0] == "o" and not (a[1] < nn_ // 2 and b[1] == nn_ - 1 - a[1]):
                    ok = False
    check("new layers go in symmetrically: every U-net pair of two old layers was a pair before (any sizes)", ok)

    # ---- every growth keeps the function
    torch.manual_seed(0)
    cfg = SeedConfig(**SMALL)
    m = Seed(cfg).to(dev)
    m.set_progress(0.0)
    opt = m.make_optimizer(3e-3, dev)
    x = torch.randint(0, 512, (4, 64), device=dev)
    y = torch.roll(x, -1, 1)
    first = None
    diffs, ldiffs, rdiffs = [], [], []
    for stage in range(4):
        for _ in range(12):
            loss, _ = m.losses(x, y)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        first = first if first is not None else loss.item()
        if stage == 3:
            break
        a = logits(m, x)
        la = train_loss(m, x, y)
        with torch.no_grad():
            ra = torch.cat([m.read(x[:1, :40])[0], m.read(x[:1, 40:41], m.read(x[:1, :40])[1])[0]], 1)
        m.grow_to(cfg.stages()[stage + 1][2])
        old, opt = opt, m.make_optimizer(3e-3, dev)
        carry_state(old, opt)
        m.carry_opt(old, opt)
        b = logits(m, x)
        lb = train_loss(m, x, y)
        with torch.no_grad():
            rb = torch.cat([m.read(x[:1, :40])[0], m.read(x[:1, 40:41], m.read(x[:1, :40])[1])[0]], 1)
        diffs.append((a - b).abs().max().item() / a.abs().max().item())
        ldiffs.append(abs(la - lb))
        rdiffs.append((ra - rb).abs().max().item())
    check("each growth (4x64 -> 6x96 -> 9x128 -> 12x160) keeps the scores (to rounding), after real training",
          max(diffs) < 1e-5, " ".join(f"{d:.1e}" for d in diffs))
    check("... and the training loss (with the early 3-word lesson) does not jump", max(ldiffs) < 1e-4,
          " ".join(f"{d:.1e}" for d in ldiffs))
    check("... and word-by-word reading (smear, shifted keys, value tables carried along)", max(rdiffs) < 1e-4)
    check("it kept learning across the growths", loss.item() < first - 1.0, f"{first:.2f} -> {loss.item():.2f}")
    with torch.no_grad():
        full = m(x[:1, :41])[0]
        lg, st_ = m.read(x[:1, :40])
        lg2, _ = m.read(x[:1, 40:41], st_)
    check("reading word by word = reading all at once (grown)", torch.allclose(torch.cat([lg, lg2], 1), full, atol=1e-4))
    names_of = lambda mm: {k: tuple(v.shape) for k, v in mm.state_dict().items()}   # noqa: E731
    built = Seed(cfg, stage=3)
    check("the grown body is exactly the body built at full size (every name and shape)", names_of(built) == names_of(m))
    fresh = Seed(cfg).to(dev)
    fresh.load_state_dict(m.state_dict())
    check("a grown checkpoint loads into a fresh seed (it grows first) and gives the same scores",
          fresh.stage == 3 and torch.allclose(logits(fresh, x), logits(m, x), atol=1e-5))
    del built, fresh

    # ---- the optimizer
    core, adam, rows = O.groups(m)
    allp = {id(p) for p in m.parameters() if p.requires_grad}
    got = [id(p) for _, p in core] + [id(p) for ps in adam.values() for p in ps] + [id(p) for p in rows]
    check("every parameter is in exactly one group", sorted(got) == sorted(allp) and len(got) == len(set(got)),
          f"core grids {len(core)}, " + ", ".join(f"{k} {len(v)}" for k, v in adam.items()) + f", library rows {len(rows)}")
    check("the core = the layers' grids (attention, MLP, the library readers); the library and the word table are not",
          all(n.startswith("blocks.") for n, _ in core) and any(".look." in n for n, _ in core)
          and not any(p is m.lib.values for _, p in core))
    mu = opt.opts[0]
    check("Grokfast on the core (alpha 0.98, x2), the core's decay 0.2 (Bee5: 0.1)",
          opt.lamb == 2.0 and opt.alpha == 0.98 and mu.param_groups[0]["weight_decay"] == 0.2
          and len(opt.gf) == len(core) and all(any(p is q for _, q in core) for p in opt.gf))
    off = Seed(SeedConfig(**SMALL, grokfast=0)).make_optimizer(3e-3, dev, muon_wd=0.1)
    check("--set grokfast=0: plain NorMuon with Bee5's decay", off.lamb == 0.0 and off.opts[0].param_groups[0]["weight_decay"] == 0.1)
    check("the optimizer's memory came through the growths (momentum, Adam moments, Grokfast's average)",
          all(p in mu.state for _, p in core) and all(p in opt.gf for _, p in core)
          and all(p in opt.opts[1].state for p in adam["table"]))
    before = m.lib.values.detach().clone()
    m.lib.track = torch.zeros(m.lib.rows(), dtype=torch.long, device=dev)
    loss, _ = m.losses(x[:1, :8], y[:1, :8])
    loss.backward()
    read = m.lib.track > 0
    m.lib.track = None
    opt.step()
    opt.zero_grad(set_to_none=True)
    moved = (m.lib.values - before).abs().amax(1) > 0
    check("library rows that were not read this step do not move; the rows read do",
          bool((moved == read).all()) and 0 < int(read.sum()) < m.lib.rows(),
          f"{int(read.sum())} of {m.lib.rows()} rows read and moved")
    sd = opt.state_dict()
    opt2 = m.make_optimizer(3e-3, dev)
    opt2.load_state_dict(sd)
    check("the optimizer saves and loads (with Grokfast's averages)", len(opt2.gf) == len(core))

    # ---- the library probe
    wins = torch.randint(0, 512, (8, 65))
    rep = m.library_report(wins, dev, batch=4)
    check("the library probe: rows read, and the loss with the library closed", 0 < rep["lib_used"] <= 1
          and rep["lib_loss_off"] != rep["lib_loss_on"], f"{rep['lib_used'] * 100:.0f}% read; held-out {rep['lib_loss_on']:.3f}, "
          f"closed {rep['lib_loss_off']:.3f}")
    for bad in ("4x64:0,6x96:0.2,9x128:0.4,12x160:0.6", ):
        try:
            SeedConfig(**{**SMALL, "bloom": bad, "lib_at": "1,4"}).stages()
            roles(SeedConfig(**{**SMALL, "bloom": bad, "lib_at": "1,4"}))
            raised = False
        except ValueError:
            raised = True
    check("a library reader that would be born late is refused (the library must be read from day one)", raised)
    del m, opt

    # ---- one whole tiny run on stand-in text, then a resume
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
    argv = ["train", "--model", "seed", "--preset", "smoke", "--data", str(data), "--runs", str(runs), "--mentor", "smol-random",
            "--device", str(dev), "--compile", "0", "--tokens", "60000", "--batch-ramp", "512:0.5", "--mix", "fwe:0.9,chat:0.1@0",
            "--micro", "2", "--set", "ctx=64:0,128:0.3"]
    sys.argv = argv
    try:
        train.main()
        missing = False
    except SystemExit as e:
        missing = "sado" in str(e)
    check("by default it reads Sado and says how to get it when it is missing", missing)
    sys.argv = argv + ["--data-sub", "smol"]
    train.main()
    rd = runs / "seed"
    recs = [json.loads(l_) for l_ in (rd / "metrics.jsonl").read_text().splitlines()]
    cfg_run = json.loads((rd / "config.json").read_text())
    check("the run used Seed's body and Bee5's recipe (the mentor 1 in 4, the cool-down over the last 30%)",
          cfg_run["mentor_share"] is True and cfg_run["mentor_weight"] == 0.25 and cfg_run.get("decay") == 0.3
          and "library" in cfg_run["layout"] and cfg_run.get("params_active"), cfg_run["layout"])
    grows = [r_ for r_ in recs if r_["kind"] == "grow"]
    check("it grew three times inside the run, at 20%, 40% and 60%, and logged each stage's speed",
          [(g["depth"], g["width"]) for g in grows] == [(6, 96), (9, 128), (12, 160)] and all("stage_tok_s" in g for g in grows),
          " ".join(f"{g['tokens']:,}" for g in grows))
    kinds = {r_["kind"] for r_ in recs}
    check("held-out loss, HellaSwag, LogT and the library probe were logged", {"eval", "hellaswag", "logres", "library"} <= kinds,
          ", ".join(sorted(kinds)))
    st = json.loads((rd / "status.json").read_text())
    check("the run finished", st["state"] == "done" and "val_loss" in st["latest"],
          f"val loss {st['latest'].get('val_loss', float('nan')):.3f}")
    mdl, ck = train.load_model(rd / "ckpt.pt", dev)
    check("its checkpoint loads as Seed, fully grown", ck["kind"] == "seed" and type(mdl).__name__ == "Seed" and mdl.depth == 12)
    out, _ = mdl.generate(enc.encode_ordinary("The cat"), max_new_tokens=5)
    check("the trained model writes", len(out) <= 5, repr(enc.decode(out)))
    n_before = len(recs)
    sys.argv = argv + ["--data-sub", "smol"]
    train.main()
    recs2 = [json.loads(l_) for l_ in (rd / "metrics.jsonl").read_text().splitlines()]
    st2 = json.loads((rd / "status.json").read_text())
    check("run again, it picks up its checkpoint (grown, its optimizer too) instead of starting over",
          st2["state"] == "done" and len(recs2) == n_before and st2["tokens"] == st["tokens"], f"{st2['tokens']:,} tokens")

    if dev.type == "cuda":
        mg = Seed(SeedConfig(**{**SMALL, "vocab_size": 49152})).to(dev).train()
        mg.set_progress(0.0)
        og = mg.make_optimizer(3e-3, dev)
        xg = torch.randint(1, 49152, (4, 128), device=dev)
        for _ in range(3):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = mg.losses(xg, xg)
            loss.backward()
            og.step()
            og.zero_grad(set_to_none=True)
        check("bf16 training steps on the GPU", torch.isfinite(loss).item())
        mg.enable_compile()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lc, _ = mg.losses(xg, xg)
        check("the compiled step runs and agrees", torch.isfinite(lc).item() and abs(lc.item() - loss.item()) < 0.5,
              f"{lc.item():.4f} vs {loss.item():.4f}")
        a = logits(mg, xg)
        mg.grow_to(0.2)
        old, og = og, mg.make_optimizer(3e-3, dev)
        carry_state(old, og)
        mg.carry_opt(old, og)
        b = logits(mg, xg)
        check("a growth on the GPU keeps the scores", (a - b).abs().max().item() < 1e-3 * a.abs().max().item())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lc, _ = mg.losses(xg, xg)
        lc.backward()
        og.step()
        check("and the grown, recompiled model trains on", torch.isfinite(lc).item())
    tmp_ctx.cleanup()
    term.tests_passed("Seed")


if __name__ == "__main__":
    main()
