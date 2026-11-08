"""Self-tests for Pegasus 1, Pegasus 1 Mini and Plain-100M: the sizes, that no token can see a later one (the
thoughts, the plan and the notebook included), the meaning pieces, that every part learns, the recipe, and one
whole (tiny) run of each model end to end.

    python -m pegasus.tests                 (CPU, a few minutes; without internet it uses stand-in tokenizers)
    python -m pegasus.tests --device cuda   (on the pod: also bf16 and the compiled step)
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

SMALL = dict(d_tok=64, tok_heads=2, tok_kv_heads=1, tok_mlp=128, enc_layers=1, spk_layers=2, window=16,
             d_think=64, think_layers=2, think_heads=2, think_kv_heads=1, think_mlp=128, head_dim=32,
             max_len=128, ctx="128:0", thought_len=6, max_thoughts=2)


def c_len(m):
    return m.cfg.thought_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    dev = torch.device(ap.parse_args().device)
    term.tests_start("Pegasus 1", dev)
    torch.manual_seed(0)
    from bee2.tests import stand_ins
    tmp_ctx = tempfile.TemporaryDirectory()
    tmp = Path(tmp_ctx.name)
    used_ = stand_ins(tmp)
    if used_:
        term.info(f"no internet: stand-in {' and '.join(used_)} tokenizer(s)")
    from pegasus.model import Pegasus, PegasusConfig
    from pegasus.baseline import Plain100, Plain100Config

    # ---- sizes
    full = Pegasus(PegasusConfig())
    n = full.num_params()
    pp = full.part_params()
    check("Pegasus 1: about 400M parameters, most of them in the Thinker", 380e6 < n < 420e6 and pp["thinker"] > 0.6 * n,
          f"{n / 1e6:.1f}M: " + ", ".join(f"{k} {v / 1e6:.1f}M" for k, v in pp.items()))
    check("its cost per token is that of a much smaller normal model (the Thinker steps once per thought)",
          full.work_per_token() < 0.5 * n, f"~{full.work_per_token() / 1e6:.0f}M at work per token")
    del full
    mini = Pegasus(PegasusConfig(size="mini"))
    plain = Plain100(Plain100Config())
    nm, npl = mini.num_params(), plain.num_params()
    check("Pegasus 1 Mini and Plain-100M: about 100M each, within 10% of each other", 90e6 < nm < 120e6 and abs(nm - npl) < 0.1 * npl,
          f"Mini {nm / 1e6:.1f}M (~{mini.work_per_token() / 1e6:.0f}M at work per token), Plain {npl / 1e6:.1f}M")
    check("Plain-100M does not grow; neither does Pegasus", plain.depth == plain.cfg.n_layer == 10 and not hasattr(mini, "grow_to"))
    c = mini.cfg
    check("Bee4's windows (512 -> 1,024 at 15% -> 2,048 at 30%), SmolLM2's pieces, 8 pieces of 1,000 meanings",
          mini.ctx_at(0) == 512 and mini.ctx_at(0.15) == 1024 and mini.ctx_at(0.3) == 2048 and c.tokenizer == "smollm2"
          and c.pieces == 8 and c.n_codes() == 1000)
    del mini, plain

    # ---- the meaning pieces (FSQ)
    m = Pegasus(PegasusConfig(**SMALL)).to(dev)
    g = m.fsq.grid
    check("the dictionary: 1,000 different meanings, each 4 numbers in [-1, 1]", g.shape == (1000, 4)
          and len({tuple(r.tolist()) for r in g}) == 1000 and g.abs().max() <= 1.0)
    u = (torch.randn(5, 4, device=dev) * 3).requires_grad_()
    val, code = m.fsq(u)
    check("a piece's numbers are its meaning's grid point", torch.allclose(val.float(), g[code].to(val.device), atol=1e-6))
    val.sum().backward()
    check("rounding passes gradients straight through (to the Reader)", u.grad is not None and u.grad.abs().sum() > 0)

    # ---- the first batch sets where thoughts start, so they begin near thought_len tokens
    T = 128
    x = torch.randint(1, 49152, (2, T), device=dev)
    x[:, 40] = 0                                              # a document boundary inside
    m.eval()
    with torch.no_grad():
        before = float(m._core(x)[3]["thought_len"])
    m.calibrate(x)
    with torch.no_grad():
        after = float(m._core(x)[3]["thought_len"])
    edge = m.route_edge.item()
    m.calibrate(torch.randint(1, 49152, (2, T), device=dev))
    check("the first batch sets the boundary's edge once: thoughts start near thought_len tokens",
          bool(m.calibrated) and m.route_edge.item() == edge and 0.5 * c_len(m) <= after <= 2 * c_len(m),
          f"{before:.1f} tokens per thought before, {after:.1f} after (aim {c_len(m)})")
    sd = m.state_dict()
    m2 = Pegasus(PegasusConfig(**SMALL)).to(dev)
    m2.load_state_dict(sd)
    check("a checkpoint keeps it (no second calibration after a resume)", bool(m2.calibrated) and m2.route_edge.item() == edge)
    del m2

    # ---- no peeking: changing later tokens never changes an earlier token's scores
    with torch.no_grad():
        base = m._body(x).float()
        worst = 0.0
        for s_ in (1, 17, 41, 64, 100, 127):
            y = x.clone()
            y[:, s_:] = torch.randint(1, 49152, (2, T - s_), device=dev)
            worst = max(worst, (m._body(y).float()[:, :s_] - base[:, :s_]).abs().max().item())
    check("no token sees a later one: thoughts, plan, notebook and words all causal", worst < 1e-4, f"max change {worst:.1e}")
    with torch.no_grad():
        p8 = m._body(x[:, :50]).float()
    e = (p8 - base[:, :50]).abs().max().item()
    check("reading a shorter text gives the same scores (padding at the end changes nothing)", e < 1e-4, f"max diff {e:.1e}")

    # ---- flex attention (the compiled steps) does the same sums as the yes/no grids
    from pegasus.model import window_mask
    import warnings
    warnings.filterwarnings("ignore", message=".*flex_attention called without torch.compile.*")   # (a check, not speed)
    with torch.no_grad():
        a_ = m._core(x)[0].float()
        b_ = m._core(x, window_mask(T, m.cfg.window, dev))[0].float()
    e = (a_ - b_).abs().max().item()
    check("flex attention (local window and notebook) gives the same outputs as the yes/no grids", e < 1e-4,
          f"max diff {e:.1e}")

    # ---- thoughts
    with torch.no_grad():
        _, _, _, st = m._core(x)
    check("thoughts form (about thought_len tokens each, between 1 and the window)", 1 <= float(st["thought_len"]) <= T,
          f"{float(st['thought_len']):.1f} tokens per thought at the start")

    # ---- every part learns
    m.train()
    loss, stats = m.losses(x, x)
    loss.backward()
    parts = {"Reader": m.reader[0].attn.q.weight, "routing": m.route_edge, "pieces": m.to_pieces.weight,
             "Thinker": m.thinker[0].attn.q.weight, "guesser": m.guess_fc2.weight, "Speaker": m.speaker[0].attn.q.weight,
             "plan": m.plan.weight, "word table": m.embed.weight}
    if m.speaker[-1].note is not None:
        parts["notebook"] = m.speaker[-1].note.k.weight
    dead = [k for k, p_ in parts.items() if p_.grad is None or p_.grad.abs().sum() == 0]
    check("one lesson reaches every part (Reader, routing, pieces, Thinker, guesser, Speaker, notebook)", not dead,
          "no gradient: " + ", ".join(dead) if dead else f"loss {loss.item():.3f} (next-word {float(stats['ce']):.3f})")
    m.zero_grad(set_to_none=True)
    m.eval()
    with torch.no_grad():
        _, ce, _ = m(x, x)
        lg = m._body(x).float()
        ref = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1))
        _, ce2, _ = m(x[:, :-1], x[:, 1:])
    check("the held-out loss is the next-word loss only (no meaning or spring terms)", abs(ce2.item() - ref.item()) < 1e-3,
          f"{ce2.item():.4f} vs {ref.item():.4f}")
    from oazy import tok
    enc = tok.get("smollm2")
    out, _ = m.generate(enc.encode_ordinary("The cat"), max_new_tokens=4)
    check("it writes word by word", len(out) <= 4)

    # ---- the recipe
    from oazy.train import RECIPE, MENTOR, BEES, FAST, PRESETS
    pb = PRESETS["big"]
    check("Bee4's recipe for all three (the same mentor, cool-down, windows, steps, learning rates)",
          all(RECIPE[k] == RECIPE["bee4"] and MENTOR[k] == MENTOR["bee4"] and k in BEES and k in FAST
              and pb["batch_" + k] == pb["batch_bee4"] and pb["lr_" + k] == pb["lr_bee4"] and pb["muon_lr_" + k] == pb["muon_lr_bee4"]
              for k in ("pegasus", "pegasusmini", "plain100")))

    # ---- one whole tiny run of each on stand-in text
    from bee2.tests import fake_docs
    from oazy.data import get_tokenizer, EOT
    from oazy import retok, train
    gpt = get_tokenizer()
    rng = random.Random(0)
    data = tmp / "data"
    for sub, nd in (("", 1500), ("chat", 300)):
        d = data / sub if sub else data
        d.mkdir(parents=True, exist_ok=True)
        for split, kk in (("val", nd // 5), ("train", nd)):
            ds = fake_docs(kk, rng)
            if sub == "chat":
                ds = [f"User: {u_}\n\nAssistant: {v_}" for u_, v_ in zip(ds, fake_docs(kk, rng))]
            np.fromiter((t for x_ in ds for t in gpt.encode_ordinary(x_) + [EOT]), dtype=np.uint16).tofile(d / f"{split}.bin")
    (data / "hellaswag_val.jsonl").write_text("".join(
        json.dumps({"ctx": " ".join(fake_docs(1, rng)), "endings": fake_docs(4, rng), "label": rng.randint(0, 3)}) + "\n" for _ in range(24)))
    retok.run(data, "smollm2", ["fwe", "chat"], workers=2)
    runs = tmp / "runs"
    for kind in ("pegasusmini", "plain100"):
        argv = ["train", "--model", kind, "--preset", "smoke", "--data", str(data), "--runs", str(runs), "--mentor", "smol-random",
                "--device", str(dev), "--compile", "0", "--tokens", "40000", "--batch-ramp", "512:0.5", "--mix", "fwe:0.9,chat:0.1@0",
                "--micro", "2", "--data-sub", "smol"]
        sys.argv = argv
        train.main()
        rd = runs / kind
        recs = [json.loads(l_) for l_ in (rd / "metrics.jsonl").read_text().splitlines()]
        tr = [r for r in recs if r["kind"] == "train"]
        st = json.loads((rd / "status.json").read_text())
        extra = ""
        if kind == "pegasusmini":
            extra = f", {tr[-1].get('thought_len', 0):.1f} tokens per thought, meaning loss {tr[-1].get('meaning_loss', 0):.2f}"
        check(f"a whole tiny {train.NICE[kind]} run (mentor on a share of the passes, windows growing, HellaSwag, samples)",
              st["state"] == "done" and any(r.get("mentor_w") for r in tr) and any(r["kind"] == "hellaswag" for r in recs)
              and (kind != "pegasusmini" or "thought_len" in tr[-1]),
              f"val loss {st['latest']['val_loss']:.3f}{extra}")
        mdl, ck = train.load_model(rd / "ckpt.pt", dev)
        check(f"its checkpoint loads as {type(mdl).__name__}", ck["kind"] == kind)
        sys.argv = argv
        train.main()
        check("run again, it sees it is finished instead of starting over", json.loads((rd / "status.json").read_text())["state"] == "done")

    if dev.type == "cuda":
        mg = Pegasus(PegasusConfig(**{**SMALL, "max_len": 2048, "window": 256, "max_thoughts": 8, "thought_len": 16})).to(dev)
        xg = torch.randint(1, 49152, (4, 1024), device=dev)
        xg[:, 50] = 0
        mg.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = mg.losses(xg, xg)                     # (the first lesson also calibrates the edge)
        loss.backward()
        check("bf16 training step on the GPU", torch.isfinite(loss).item())
        mg.zero_grad(set_to_none=True)
        mg.enable_compile()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lc, _ = mg.losses(xg, xg)
        lc.backward()
        check("the compiled training step runs", torch.isfinite(lc).item() and abs(lc.item() - loss.item()) < 0.05,
              f"{lc.item():.4f} vs {loss.item():.4f}")
        # the same lesson's gradients through flex attention and through the yes/no grids (fp32)
        mg.zero_grad(set_to_none=True)
        T2 = xg.shape[1]

        def grads(w):
            mg.zero_grad(set_to_none=True)
            s_, me_, r_, _ = mg._core(xg, w)
            (s_.float().pow(2).mean() + me_ + r_).backward()
            return {n_: q_.grad.float().clone() for n_, q_ in mg.named_parameters() if q_.grad is not None}
        ga, gb = grads(None), grads(window_mask(T2, mg.cfg.window, dev))
        worst = max(((ga[k] - gb[k]).norm() / ga[k].norm().clamp_min(1e-12)).item() for k in ga)
        check("flex attention gives the same gradients (every weight)", set(ga) == set(gb) and worst < 1e-3,
              f"largest relative difference {worst:.1e}")
        mg.zero_grad(set_to_none=True)
        del mg

        # how much faster: the real Pegasus 1, one compiled bf16 lesson of 8 x 2,048 tokens, grids vs flex
        import time
        speed = {}
        xb = torch.randint(1, 49152, (8, 2048), device=dev)
        for flex in (0, 1):
            torch.manual_seed(0)
            mf = Pegasus(PegasusConfig(flex=flex)).to(dev)
            mf.calibrate(xb)
            mf.enable_compile()
            mf.train()

            def lesson():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lf, _ = mf.losses(xb, xb)
                lf.backward()
                mf.zero_grad(set_to_none=True)
            for _ in range(3):
                lesson()                                    # (compiles)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(10):
                lesson()
            torch.cuda.synchronize()
            speed[flex] = 10 * xb.numel() / (time.time() - t0)
            del mf
            torch.cuda.empty_cache()
        check("Pegasus 1 with flex attention is faster (the model alone, no mentor)", speed[1] > speed[0],
              f"{speed[0] / 1e3:.0f}k -> {speed[1] / 1e3:.0f}k tokens/s ({speed[1] / speed[0] - 1:+.0%})")
    tmp_ctx.cleanup()
    term.tests_passed("Pegasus 1")


if __name__ == "__main__":
    main()
