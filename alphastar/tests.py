"""Self-tests for Pegasus Alpha Star (alphastar/model.py): the size and its split, causality (also when it stops by
itself), word-by-word reading equal to whole-window reading, the training draw of passes and its curriculum, gradients
through the last 2 passes only, the adaptive stop, the fact drawer, the next-thought lesson on every learning pass, that
it learns, the recipe, one whole tiny run; on a GPU also the compiled step and the speed against Plain-100M at full size.

    python3 -m alphastar.tests                 (CPU, a few minutes)
    python3 -m alphastar.tests --device cuda   (on the pod: also the compiled step and the speed race, ~5 min)
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
import torch.nn.functional as F

from bee3.tests import check
from oazy import term

SMALL = dict(d_model=64, attn_heads=4, attn_kv_heads=2, attn_head_dim=16, mlp_hidden=128, max_len=128, ctx="128:0",
             drawer_keys=16, drawer_heads=2, drawer_k=4, drawer_qdim=32, fore_hidden=96, fore_top=20, fore_max=512,
             fore_warmup=0.0, test_max=8, gen_passes=3, noise_rows=256)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-race", action="store_true", help="skip the GPU speed race")
    a = ap.parse_args()
    dev = torch.device(a.device)
    term.tests_start("Pegasus Alpha Star", dev)
    from bee2.tests import stand_ins
    tmp_ctx = tempfile.TemporaryDirectory()
    tmp = Path(tmp_ctx.name)
    used_ = stand_ins(tmp)
    if used_:
        term.info(f"no internet: stand-in {' and '.join(used_)} tokenizer(s)")
    from alphastar.model import AlphaStar, AlphaStarConfig, Drawer
    from pegasus.baseline import Plain100, Plain100Config
    from oazy import tok
    enc = tok.get("smollm2")

    # ---- size
    full = AlphaStar(AlphaStarConfig())
    n_full, n_plain = full.num_params(), Plain100(Plain100Config()).num_params()
    sp = full.split()
    check("about 100M numbers, near Plain-100M (the next-thought head is training-only, not counted)",
          95e6 < n_full < 108e6, f"{n_full / 1e6:.1f}M vs Plain-100M {n_plain / 1e6:.1f}M | "
          + ", ".join(f"{k} {v:.1f}M" for k, v in sp.items()))
    check("its depth while working: 2 + 4 x passes + 2 layers (28 at the mean of 6, 68 at 16)",
          full.cfg.depth_at(6) == 28 and full.cfg.depth_at(16) == 68)
    del full

    torch.manual_seed(0)
    m = AlphaStar(AlphaStarConfig(**SMALL)).to(dev)
    V = m.cfg.vocab_size
    text = ("The cat sat on the mat. It was warm! Was the garden happy? Yes. The dog ran to the old red barn by the river. "
            "A very long sentence without any stop that keeps going and going on and on past sixteen pieces for sure here "
            "and there and everywhere around")
    ids = enc.encode_ordinary(text)
    counts = np.zeros(V, dtype=np.int64)
    np.add.at(counts, np.asarray(ids * 50, dtype=np.int64), 1)
    m.prepare_foresight(enc, counts)

    # ---- causality and reading
    m.eval()
    g = torch.Generator().manual_seed(0)
    x = torch.randint(1, V, (2, 48), generator=g).to(dev)
    x2 = x.clone()
    x2[:, 30:] = torch.randint(1, V, (2, 18), generator=g).to(dev)
    with torch.no_grad():
        for name, tp in (("fixed 5 passes", 5), ("stopping by itself", 0)):
            m.cfg.test_passes = tp
            a1, a2 = m._body(x), m._body(x2)
            err = (a1[:, :30] - a2[:, :30]).abs().max().item()
            check(f"causal ({name}): changing later words moves nothing earlier", err < 1e-5, f"max change {err:.1e}")
        m.cfg.test_passes = 3
        whole = m._body(x[:1])
        logits, st = m.read(x[:1, :20], passes=3)
        parts = [logits]
        for t in range(20, 48):
            lg, st = m.read(x[:1, t:t + 1], st)
            parts.append(lg)
        stream = torch.cat(parts, 1)
        err = (whole - stream).abs().max().item()
        check("word-by-word reading (every pass keeps its keys) gives the whole-window scores", err < 1e-4, f"max diff {err:.1e}")
        m.cfg.test_passes = 0
        out, _ = m.generate(enc.encode_ordinary("The cat"), max_new_tokens=4)
        check("it writes word by word", len(out) <= 4)

    # ---- the adaptive stop
    with torch.no_grad():
        e = m._read(x)
        _, u_all = m.think_test(e, eps=0.0)
        _, u_none = m.think_test(e, eps=1e9)
        seen = []
        s_fin, u_mid = m.think_test(e, eps=0.3, every=lambda n, s: seen.append(s.clone()))
        froze = True
        for i in range(1, len(seen)):
            done = u_mid < i            # words that had stopped before pass i+1 keep their thought
            if done.any():
                froze &= torch.equal(seen[i][done], seen[i - 1][done])
    check("adaptive: eps 0 thinks the full 8 passes, a huge eps stops after 2 (the first always moves)",
          float(u_all.min()) == 8 and float(u_none.max()) == 2, f"{float(u_all.mean()):.0f} and {float(u_none.mean()):.0f}")
    check("adaptive: a word that is done keeps its thought while others go on", froze,
          f"eps 0.3: {float(u_mid.mean()):.1f} passes on average ({float(u_mid.min()):.0f} to {float(u_mid.max()):.0f})")

    # ---- the training draw and its curriculum
    m.train()
    m.set_progress(0.0)
    d0 = [m.sample_passes() for _ in range(4000)]
    m.set_progress(0.15)
    mid = m._mean
    m.set_progress(0.5)
    d1 = [m.sample_passes() for _ in range(4000)]
    check("passes are drawn at random: mean 2 at the start, 6 once the curriculum is over, 1 to 12",
          abs(np.mean(d0) - 2) < 0.3 and abs(np.mean(d1) - 6) < 0.4 and min(d0 + d1) >= 1 and max(d0 + d1) <= 12
          and abs(mid - 4) < 1e-9, f"start {np.mean(d0):.2f}, at 15% mean {mid:g}, after 30% {np.mean(d1):.2f} "
          f"(spread {np.percentile(d1, 10):.0f}-{np.percentile(d1, 90):.0f})")

    # ---- training: the lesson and where its gradient goes
    m.cfg.train_noise = "table"
    m.force_passes = 5
    xb = torch.tensor([ids[:64]] * 4, device=dev)
    loss, st = m.losses(xb[:, :-1], xb[:, 1:])
    m.eval()
    m.cfg.test_passes = 5
    with torch.no_grad():
        _, lv, stv = m(xb[:, :-1], xb[:, 1:])
    m.cfg.test_passes = 0
    m.train()
    check("training with 5 passes = reading with 5 passes (the same thoughts; only the gradients are cut)",
          abs(st["ce"].item() - lv.item()) < 1e-4 and float(st["passes"]) == 5 and "fore" not in stv,
          f"{st['ce'].item():.4f} vs {lv.item():.4f}; reading has no next-thought loss")
    # the next-thought lesson: passes 4 and 5 learn, weights 4/9 and 5/9 of 0.1
    pts_seed = 7
    torch.manual_seed(pts_seed)
    loss, st = m.losses(xb[:, :-1], xb[:, 1:])
    with torch.no_grad():
        e = m._read(xb[:, :-1])
        s = m.start(4, xb.shape[1] - 1, 0, dev, e.dtype, True)
        th = []
        for _ in range(5):
            s = m._step(s, e)
            th.append(s)
        torch.manual_seed(pts_seed)
        pts = m.fore_points(xb[:, :-1], xb[:, 1:])
        f4, _ = m.fore_nll(th[3], pts)
        f5, _ = m.fore_nll(th[4], pts)
    want = st["ce"].item() + 0.1 * (4 / 9 * f4.item() + 5 / 9 * f5.item())
    check("the loss = next-word + 0.1 x the next-thought loss on the learning passes, later passes weigh more (4/9, 5/9)",
          abs(loss.item() - want) < 1e-3 and abs(st["fore"].item() - f5.item()) < 1e-4,
          f"next-word {st['ce'].item():.3f}, next thought pass 4 {f4.item():.3f}, pass 5 {f5.item():.3f}")
    loss.backward()
    gr = {n: (p.grad.abs().sum().item() if p.grad is not None else 0.0) for n, p in m.named_parameters()}
    check("the gradient reaches every part: reader, drawer (facts, keys, questions), core, speaker, head",
          all(gr[k] > 0 for k in ("embed.weight", "reader.0.attn.q.weight", "reader.1.drawer.values.weight",
                                  "reader.1.drawer.keys", "reader.1.drawer.query.weight", "adapt.weight",
                                  "core.0.attn.q.weight", "speaker.1.mlp.up.weight", "fore.up.weight")))
    m.zero_grad(set_to_none=True)
    # gradients only through the last 2 passes: what passes 1-3 did is a fixed starting point
    e = m._read(xb[:, :-1])
    outs = m.think_train(e, 5)
    s_in = outs[0]
    check("only the last 2 passes keep their work for the backward pass", len(outs) == 2 and outs[0].requires_grad)
    m.zero_grad(set_to_none=True)
    # the starting thought of pass 4 does not depend (for the gradient) on the reader through passes 1-3:
    e2 = m._read(xb[:, :-1]).detach().requires_grad_(True)
    o_a = m.think_train(e2, 5)[-1].sum()
    g_trunc = torch.autograd.grad(o_a, e2)[0]
    s3 = m.start(4, xb.shape[1] - 1, 0, dev, e2.dtype, True)
    with torch.no_grad():
        for _ in range(3):
            s3 = m._step(s3, e2)
    s3 = m._step(m._step(s3, e2), e2).sum()
    g_ref = torch.autograd.grad(s3, e2)[0]
    check("its gradient equals the gradient of passes 4-5 alone from a fixed pass-3 thought", torch.allclose(g_trunc, g_ref, atol=1e-5),
          f"max diff {(g_trunc - g_ref).abs().max().item():.1e}")
    del s_in, outs

    # ---- the fact drawer
    d = m.reader[1].drawer
    with torch.no_grad():
        h = torch.randn(10, 64, device=dev)
        idx, w = d.pick(h)
        q = d.qnorm(d.query(h).view(10, d.H, 2, d.half))
        full_s = (torch.einsum("nhd,hkd->nhk", q[:, :, 0], d.keys[:, 0])[..., :, None]
                  + torch.einsum("nhd,hkd->nhk", q[:, :, 1], d.keys[:, 1])[..., None, :]).reshape(10, d.H, -1)
        best = full_s.topk(d.k, -1).indices.sort(-1).values
        got = idx.view(10, d.H, d.k).sort(-1).values
        y = d(h)
        ref = torch.zeros(10, 64, device=dev)
        for n_ in range(10):
            ref[n_] = (w[n_, :, None] * d.values.weight[idx[n_]]).sum(0)
        ref = d.out(ref * F.silu(d.gate(h)))
    check("the drawer picks exactly the best facts of all 16 x 16 (product keys find them from the halves)",
          torch.equal(best, got), f"{d.H} heads x {d.k} picks of {d.n * d.n}")
    check("its answer = the picked facts mixed by a softmax of their scores, gated (Meta's memory layer)",
          torch.allclose(y, ref, atol=1e-5) and torch.allclose(w.view(10, d.H, d.k).sum(-1), torch.ones(10, d.H, device=dev)))
    from oazy.train import make_optimizer
    opt = make_optimizer(m, 1e-3, dev, "muon", 0.02, 0.1)
    adam = opt.opts[1] if hasattr(opt, "opts") else opt
    nd = [gp for gp in adam.param_groups if gp["weight_decay"] == 0][0]["params"]
    muon = opt.opts[0].param_groups[0]["params"] if hasattr(opt, "opts") else []
    check("the facts get no weight decay (a fact used rarely must not fade); the core's adapter learns with Muon",
          any(p is d.values.weight for p in nd) and any(p is m.adapt.weight for p in muon))
    del opt

    # ---- the probes
    m.eval()
    r = m.loop_report(np.asarray(xb.cpu()), dev, batch=2, curve=(1, 2, 4, 8))
    check("the loop probe: loss by passes, the adaptive stop, the next thought by pass, the drawer's use",
          set(r["pass_loss"]) == {"1", "2", "4", "8"} and 1 <= r["adaptive_passes"] <= 8 and len(r["thought_by_pass"]) == 8
          and 0 < r["drawer_used"] <= 1,
          f"passes {' '.join(f'{k}:{v:.2f}' for k, v in r['pass_loss'].items())} | adaptive {r['adaptive_passes']:.1f} | "
          f"facts used {r['drawer_used'] * 100:.0f}%")
    m.train()

    # ---- it learns, and on repeated text it learns to use its passes
    torch.manual_seed(1)
    m.cfg.train_noise = "random"
    m.force_passes = None
    m.set_progress(1.0)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for i in range(80):
        l_, st_ = m.losses(xb[:, :-1], xb[:, 1:])
        first = first if first is not None else st_["ce"].item()
        l_.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    r = m.loop_report(np.asarray(xb.cpu()), dev, batch=2, curve=(1, 2, 4, 8))
    rep = m.foresight_report(np.asarray(xb.cpu()), dev, batch=2)
    check("it learns: the next-word loss falls, and the next-thought head beats the blind guess",
          r["adaptive_loss"] < first - 1.0 and rep["fore_val"] < rep["fore_blind"],
          f"{first:.2f} -> {r['adaptive_loss']:.2f}; next thought {rep['fore_val']:.2f} vs blind {rep['fore_blind']:.2f} | "
          f"by passes {' '.join(f'{k}:{v:.2f}' for k, v in r['pass_loss'].items())}")
    del opt

    # ---- the recipe
    from oazy.train import RECIPE, MENTOR, BEES, FAST, PRESETS
    pb = PRESETS["big"]
    check("Plain-100M's recipe, mentor, steps and learning rates exactly", RECIPE["alphastar"] == RECIPE["plain100"]
          and MENTOR["alphastar"] == MENTOR["plain100"] and "alphastar" in BEES and "alphastar" in FAST
          and pb["batch_alphastar"] == pb["batch_plain100"] and pb["lr_alphastar"] == pb["lr_plain100"]
          and pb["muon_lr_alphastar"] == pb["muon_lr_plain100"] and AlphaStarConfig().ctx == Plain100Config().ctx)
    mo = AlphaStar(AlphaStarConfig(**{**SMALL, "drawer": 0, "thought": 0}))
    check("the drawer and the next-thought lesson switch off (--set drawer=0, --set thought=0)",
          not any(isinstance(x_, Drawer) for x_ in mo.modules()) and mo.head_params() == 0 and mo.foresight_ready())
    del mo

    # ---- one whole tiny run on stand-in text
    from bee2.tests import fake_docs
    from oazy.data import get_tokenizer, EOT
    from oazy import retok, train
    gpt = get_tokenizer()
    rng = random.Random(0)
    data = tmp / "data"
    for sub, nd_ in (("", 1500), ("chat", 300)):
        dd = data / sub if sub else data
        dd.mkdir(parents=True, exist_ok=True)
        for split, kk in (("val", nd_ // 5), ("train", nd_)):
            ds = fake_docs(kk, rng)
            if sub == "chat":
                ds = [f"User: {u_}\n\nAssistant: {v_}" for u_, v_ in zip(ds, fake_docs(kk, rng))]
            np.fromiter((t_ for x_ in ds for t_ in gpt.encode_ordinary(x_) + [EOT]), dtype=np.uint16).tofile(dd / f"{split}.bin")
    (data / "hellaswag_val.jsonl").write_text("".join(
        json.dumps({"ctx": " ".join(fake_docs(1, rng)), "endings": fake_docs(4, rng), "label": rng.randint(0, 3)}) + "\n" for _ in range(24)))
    retok.run(data, "smollm2", ["fwe", "chat"], workers=2)
    runs = tmp / "runs"
    mentor = "smol-random"
    try:
        import transformers  # noqa: F401
    except ImportError:
        mentor = "none"
    argv = ["train", "--model", "alphastar", "--preset", "smoke", "--data", str(data), "--runs", str(runs), "--mentor", mentor,
            "--device", str(dev), "--compile", "0", "--tokens", "40000", "--batch-ramp", "512:0.5", "--mix", "fwe:0.9,chat:0.1@0",
            "--micro", "2", "--data-sub", "smol"]
    sys.argv = argv
    train.main()
    rd = runs / "alphastar"
    recs = [json.loads(l_) for l_ in (rd / "metrics.jsonl").read_text().splitlines()]
    tr = [r_ for r_ in recs if r_["kind"] == "train"]
    stt = json.loads((rd / "status.json").read_text())
    check("a whole tiny Alpha Star run (passes and next thought on every line, the loop probe, HellaSwag, samples)",
          stt["state"] == "done" and any(r_["kind"] == "loop" for r_ in recs) and any(r_["kind"] == "hellaswag" for r_ in recs)
          and all("passes" in r_ and "fore_loss" in r_ for r_ in tr) and (rd / "samples.jsonl").exists(),
          f"val loss {stt['latest']['val_loss']:.3f}, passes {np.mean([r_['passes'] for r_ in tr]):.1f}, mentor {mentor}")
    mdl, ck = train.load_model(rd / "ckpt.pt", dev)
    check("its checkpoint loads as Pegasus Alpha Star, tables included", ck["kind"] == "alphastar"
          and type(mdl).__name__ == "AlphaStar" and mdl.foresight_ready())
    sys.argv = argv
    train.main()          # resumes: the run is finished, nothing breaks
    del mdl

    if dev.type == "cuda":
        mg = AlphaStar(AlphaStarConfig(**{**SMALL, "max_len": 2048, "ctx": "2048:0", "fore_max": 8192,
                                          "train_noise": "table"})).to(dev).train()
        mg.prepare_foresight(enc, np.ones(V, dtype=np.int64))
        mg.force_passes = 4
        xg = torch.randint(1, V, (4, 1024), device=dev)
        xg[:, 50] = 0
        torch.manual_seed(0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            le, _ = mg.losses(xg, xg)
        le.backward()
        mg.zero_grad(set_to_none=True)
        mg.enable_compile()
        torch.manual_seed(0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lc, _ = mg.losses(xg, xg)
        lc.backward()
        check("the compiled step matches", torch.isfinite(lc).item() and abs(lc.item() - le.item()) < 0.05,
              f"{lc.item():.4f} vs {le.item():.4f}")
        del mg
        if not a.no_race:
            # 32 x 2,048 = one real pass (--micro 32 at full windows), at the mean of 6 passes
            xb = torch.randint(1, V, (32, 2048), device=dev)
            speed, mem = {}, {}
            for name, mk in (("Plain-100M", lambda: Plain100(Plain100Config())),
                             ("Alpha Star (6 passes)", lambda: AlphaStar(AlphaStarConfig()))):
                torch._dynamo.reset()
                torch.manual_seed(0)
                mf = mk().to(dev).train()
                if hasattr(mf, "prepare_foresight"):
                    mf.prepare_foresight(enc, np.arange(V)[::-1].copy())
                    mf.force_passes = 6
                mf.enable_compile()
                torch.cuda.reset_peak_memory_stats()

                def lesson():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lf, _ = mf.losses(xb, xb)
                    lf.backward()
                    mf.zero_grad(set_to_none=True)
                for _ in range(3):
                    lesson()
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(10):
                    lesson()
                torch.cuda.synchronize()
                speed[name] = 10 * xb.numel() / (time.time() - t0)
                mem[name] = torch.cuda.max_memory_allocated() / 2 ** 30
                del mf
                torch.cuda.empty_cache()
            pr, pa = speed["Plain-100M"], speed["Alpha Star (6 passes)"]
            term.info(f"speed (compiled bf16, 32 x 2,048, no mentor): Plain-100M {pr / 1e3:.0f}k tok/s ({mem['Plain-100M']:.0f} GB), "
                      f"Alpha Star at 6 passes {pa / 1e3:.0f}k tok/s ({mem['Alpha Star (6 passes)']:.0f} GB), "
                      f"{pa / pr:.2f}x Plain's speed")
            check("speed race at full size ran (expected about 0.6-0.7x Plain-100M at 6 passes: it does ~28 layers of work, not 10)",
                  pa > 0)
    tmp_ctx.cleanup()
    term.tests_passed("Pegasus Alpha Star")


if __name__ == "__main__":
    main()
