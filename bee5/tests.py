"""Self-tests for Bee5: the size and the body, the loss in slices against the slow way (with the early 3-word lesson
and the mentor), reading word by word = all at once (smear, key offset and value tables carried along), the
optimizer (Polar Express, NorMuon, cautious decay, momentum warm-up, every parameter in exactly one group), a few
steps that overfit one batch, the recipe, the speed check, and one whole tiny run end to end with a resume.

    python3 -m bee5.tests                 (CPU, ~1-2 min; without internet it uses stand-in tokenizers)
    python3 -m bee5.tests --device cuda   (on the pod: also bf16, the compiled step, the speed check on the GPU's kernels)
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
             ctx="128:0", vocab_size=512, key_offset="1,4")


def check(name, ok, detail=""):
    term.check(name, ok, detail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    dev = torch.device(ap.parse_args().device)
    term.tests_start("Bee5-200M", dev)
    torch.manual_seed(0)
    from bee5.model import Bee5, Bee5Config, mtp_weights
    from bee5 import loss as L
    from bee5 import optim as O

    # ---- the size and the body
    big = Bee5(Bee5Config())
    n = big.num_params()
    check("about 200M parameters, all of them counted (the word table once, the two value tables too)", 195e6 < n < 205e6,
          f"{n / 1e6:.1f}M: word table {big.embed.weight.numel() / 1e6:.1f}M, value tables {(big.table_params() - big.embed.weight.numel()) / 1e6:.1f}M, "
          f"layers {(n - big.table_params()) / 1e6:.1f}M; {big.layout()}")
    sw = Bee5(Bee5Config(mlp="swiglu"))
    check("SwiGLU (2,560) has exactly ReLU²'s (3,840) weight count, so the race compares like with like",
          sw.num_params() == n, f"{sw.num_params() / 1e6:.2f}M vs {n / 1e6:.2f}M")
    del sw
    c = big.cfg
    check("12 layers of 1,024, 16 query / 4 key-value heads of 64, SmolLM2's pieces, windows 512 -> 1,024 -> 2,048",
          c.n_layer == 12 and c.d_model == 1024 and c.attn_heads == 16 and c.attn_kv_heads == 4 and c.tokenizer == "smollm2"
          and big.ctx_at(0) == 512 and big.ctx_at(0.15) == 1024 and big.ctx_at(0.3) == 2048)
    check("value tables on layers 1-2 and 11-12, keys shifted on layers 4, 8, 12, U-net skips 1->12 ... 6->7, back-out at 8",
          c.ve_index() == [0, 1] + [None] * 8 + [0, 1] and c.offset_layers() == {3, 7, 11} and len(big.skip_gate) == 6
          and big.backout_at == 7)
    zero = all(b.attn.out.weight.abs().max() == 0 and b.mlp.down.weight.abs().max() == 0 for b in big.blocks)
    check("every output projection starts at zero, the value tables too", zero and all(t.weight.abs().max() == 0 for t in big.ve))
    cos = big.rope_cos
    check("RoPE turns half of each head (16 of 32 pairs); the other half carries no position",
          bool((cos[:, 16:] == 1).all()) and not bool((cos[100, :16] == 1).all()))
    del big

    # ---- the loss in slices = the slow way (early 3-word lesson, the mentor, both caps)
    torch.manual_seed(1)
    V, D, N = 300, 32, 70
    for kind in ("sig", "tanh"):
        cfg = Bee5Config(**{**SMALL, "vocab_size": V, "cap": kind})
        h = torch.randn(N, D, dtype=torch.float64, requires_grad=True)
        W = (torch.randn(V, D, dtype=torch.float64) * 0.3).requires_grad_()
        tg = torch.randint(0, V, (N, 3))
        tg[-1, 1:] = -1
        tg[-2, 2] = -1
        tg[5, :] = -1
        th, tW = torch.randn(N, 16, dtype=torch.float64), torch.randn(V, 16, dtype=torch.float64)
        for w, kw in (((1.0,), 0.0), ((1.0, 0.5, 0.25), 0.0), ((1.0, 0.5), 0.3)):
            K = len(w)
            outs = []
            for fn in (L.head_loss, L.plain_loss):
                h.grad = W.grad = None
                loss, ce, kd = fn(h, W, tg[:, :K], w, cfg, th if kw else None, tW if kw else None, kw, 16)
                loss.backward()
                outs.append((loss.item(), ce.item(), kd.item(), h.grad.clone(), W.grad.clone()))
            (a, b) = outs
            e = max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]), (a[3] - b[3]).abs().max().item(), (a[4] - b[4]).abs().max().item())
            check(f"the loss in slices = the slow way ({kind} cap, {K} next word{'s' if K > 1 else ''}"
                  f"{', mentor 0.3' if kw else ''}): loss, gradients", e < 1e-5, f"max diff {e:.1e} (fp32 inside)")
    check("the early lesson: 3 next words (1, 0.5, 0.25 -> 0), then 2 (1, 0.5 -> 0) by 1/6, then the next word alone",
          mtp_weights(0, 1 / 6) == (1.0, 0.5, 0.25) and len(mtp_weights(0.1, 1 / 6)) == 2 and mtp_weights(0.17, 1 / 6) == (1.0,)
          and abs(mtp_weights(1 / 24, 1 / 6)[2] - 0.125) < 1e-9)

    # ---- the model: training loss = reading loss = full scores; word by word = all at once
    torch.manual_seed(2)
    m = Bee5(Bee5Config(**SMALL)).to(dev)
    with torch.no_grad():                  # leave the zero start, so every path (value tables, gates, skips) matters
        for p_ in m.parameters():
            p_.add_(torch.randn_like(p_) * 0.05)
    x = torch.randint(1, 512, (2, 64), device=dev)
    x[:, 30] = 0
    m.train()
    m.set_progress(1.0)
    lt, _ = m.losses(x, x)
    m.eval()
    le, _ = m.losses(x, x)
    with torch.no_grad():
        logits = m(x)[0]
        full = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x.reshape(-1))
    check("training loss = held-out loss = the full scores' loss (late in the run: next word only)",
          abs(lt.item() - le.item()) < 1e-5 and abs(le.item() - full.item()) < 1e-4, f"{lt.item():.4f} / {le.item():.4f} / {full.item():.4f}")
    m.train()
    m.set_progress(0.0)
    l3, _ = m.losses(x, x)
    m.eval()
    check("early in the run the lesson counts 3 next words (a larger loss); held-out stays the next word alone",
          l3.item() > lt.item() * 1.3 and abs(m.losses(x, x)[0].item() - le.item()) < 1e-6, f"{l3.item():.3f} vs {lt.item():.3f}")
    with torch.no_grad():
        whole, _ = m.read(x[:1, :40])
        lg, st = m.read(x[:1, :25])
        for t in range(25, 40):
            lg, st = m.read(x[:1, t:t + 1], st)
        e = (lg[0, -1] - whole[0, -1]).abs().max().item()
        e2 = (whole[0] - m(x[:1, :40])[0][0]).abs().max().item()
        lgi, _, infos = m.inspect(x[:1, :1])
    check("reading word by word = all at once (look-back, shifted keys, value tables, positions carried along)", e < 1e-4 and e2 < 1e-4,
          f"max diff {e:.1e}, read vs forward {e2:.1e}")
    check("the chat app's view: one entry per layer, where the word looked", len(infos) == 6 and infos[0]["kind"] == "A"
          and infos[0]["looks"].shape[-1] == 1)
    out, _ = m.generate([5, 6, 7], max_new_tokens=6, eos=None)
    check("it writes", len(out) == 6)

    # ---- the optimizer
    torch.manual_seed(3)
    G = torch.randn(3, 96, 64)
    s = torch.linalg.svdvals(O.polar_express(G).float())
    check("Polar Express straightens the update: every direction about the same strength", float(s.min()) > 0.6 and float(s.max()) < 1.25,
          f"singular values {float(s.min()):.2f} .. {float(s.max()):.2f}")
    ps = [torch.nn.Parameter(torch.randn(64, 32)) for _ in range(3)] + [torch.nn.Parameter(torch.randn(32, 64))]
    qs = [torch.nn.Parameter(q.detach().clone()) for q in ps]
    o1, o2 = O.NorMuon(ps, lr=0.02, weight_decay=0.1), O.NorMuon(qs, lr=0.02, weight_decay=0.1)
    o2.batched = False
    for _ in range(3):
        for p1, p2 in zip(ps, qs):
            p1.grad = torch.randn_like(p1)
            p2.grad = p1.grad.clone()
        o1.step()
        o2.step()
    e = max((p1 - p2).abs().max().item() for p1, p2 in zip(ps, qs))
    check("NorMuon: same-shape grids together = one by one", e < 1e-5, f"max diff {e:.1e}")
    st0 = o1.state[ps[0]]["second"]
    check("NorMuon keeps one running size per neuron (rows of a tall grid, columns of a wide one)",
          tuple(st0.shape) == (64, 1) and tuple(o1.state[ps[3]]["second"].shape) == (1, 64))
    p = torch.nn.Parameter(torch.tensor([[1.0, -1.0, 1.0, -1.0]] * 4))
    p.grad = torch.tensor([[1.0, 1.0, -1.0, -1.0]] * 4)
    oc = O.NorMuon([p], lr=0.0, weight_decay=1.0)
    oc.param_groups[0]["lr"] = 1e-9
    oc.param_groups[0]["lr_scale"] = 1e-9 / 0.5        # decay 0.5 on AdamW's scale
    before = p.detach().clone()
    oc.step()
    shrunk = (p.detach().abs() < before.abs() - 1e-3)
    check("cautious decay: only weights the update already pulls towards zero shrink", bool(shrunk[:, [0, 3]].all())
          and not bool(shrunk[:, [1, 2]].any()))
    o3 = O.NorMuon([torch.nn.Parameter(torch.randn(8, 8))])
    g0 = o3.param_groups[0]
    check("Muon's momentum warms up 0.85 -> 0.95 over 300 steps", g0["momentum_start"] == 0.85 and g0["momentum"] == 0.95
          and g0["momentum_warm"] == 300)
    muon, adam = O.groups(m)
    ids = [id(p_) for _, p_ in muon] + [id(p_) for ps_ in adam.values() for p_ in ps_]
    check("every parameter is in exactly one group: the grids to NorMuon; table, value tables, scalars, gates to Adam",
          len(ids) == len(set(ids)) == len([p_ for p_ in m.parameters() if p_.requires_grad])
          and len(muon) == 6 * 6 and len(adam["ve"]) == 2 and len(adam["table"]) == 1, f"{len(muon)} grids, " +
          ", ".join(f"{k} {len(v)}" for k, v in adam.items()))
    from oazy.train import make_optimizer
    opt = make_optimizer(m, 1.8e-3, dev, "muon", 0.03, 0.1)
    scales = {g.get("name", "muon"): g["lr_scale"] for g in opt.param_groups}
    check("the trainer gets Bee5's own groups (each with its share of the schedule)", abs(scales["muon"] - 0.03 / 1.8e-3) < 1e-9
          and scales["ve"] == 10.0 and scales["table"] == 1.0, str({k: round(v, 2) for k, v in scales.items()}))

    # ---- a few steps on one batch: the loss must fall fast (everything is wired to learn)
    torch.manual_seed(4)
    mm = Bee5(Bee5Config(**SMALL)).to(dev).train()
    mm.set_progress(0.0)
    opt = make_optimizer(mm, 3e-3, dev, "muon", 0.03, 0.0)
    for g in opt.param_groups:
        g["lr"] = 3e-3 * g["lr_scale"]
    xb = torch.randint(1, 512, (4, 64), device=dev)
    first = None
    for i in range(60):
        loss, stt = mm.losses(xb, xb)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        first = first if first is not None else stt["ce"].item()
    mm.eval()
    last = mm.losses(xb, xb)[0].item()
    moved = {k: float(getattr(mm, k).detach().abs().sub(v).abs().max()) for k, v in (("mu", 0.0), ("smear_lam", 0.0))}
    check("60 steps on one batch: the loss falls well below the start (ln 512 = 6.24) and the learned paths move",
          last < first - 2.0 and all(v > 1e-4 for v in moved.values()) and float(mm.ve[0].weight.abs().max()) > 0,
          f"{first:.2f} -> {last:.2f}; x0 weights {moved['mu']:.3f}, look-back {moved['smear_lam']:.3f}")

    # ---- the recipe
    from oazy.train import RECIPE, PRESETS, MENTOR, BEES, FAST, NICE
    rc = RECIPE["bee5"]
    same = {k: v for k, v in rc.items() if k != "data_sub"} == {k: v for k, v in RECIPE["bee4"].items() if k != "data_sub"}
    check("Bee4's recipe (the mentor 1 in 4 -> 1 in 10, cool-down over the last 30%, bf16, the ramp), reading Sado",
          same and rc["data_sub"] == "sado" and rc["fp8"] == 0 and rc["decay"] == 0.3 and rc["mentor_share"] == 1)
    pb = PRESETS["big"]
    check("big preset: 524k-token steps, 1.8e-3 / Muon 0.03 (Bee4's); SmolLM2-360M mentor", pb["batch_bee5"] == 524288
          and pb["lr_bee5"] == 1.8e-3 and pb["muon_lr_bee5"] == 0.03 and MENTOR["bee5"] == "smollm2-360m"
          and "bee5" in BEES and "bee5" in FAST and NICE["bee5"] == "Bee5-200M")

    # ---- the speed check (quick version)
    from bee5 import speedcheck
    sc = speedcheck.main(["--preset", "smoke", "--device", str(dev), "--passes", "1", "--runs", tempfile.mkdtemp()])
    check("the speed check runs (Bee5 and Bee4), and prints a train line", sc["line"].startswith("OAZY_DATA=/root/data bash runpod/train.sh bee5")
          and sc["tok_s"] > 0 and sc["vs"] is not None, sc["line"])

    # ---- speedrun mode on stand-in shards (the records' file format: a 256-number header, then GPT-2 pieces)
    from bee5 import speedrun
    with tempfile.TemporaryDirectory() as sd:
        f = speedrun.folder(sd)
        f.mkdir(parents=True)
        rs = np.random.default_rng(0)
        for name, ntok in (("fineweb_val_000000.bin", 20_000), ("fineweb_train_000001.bin", 120_000)):
            head = np.zeros(speedrun.HEADER, dtype=np.int32)
            head[:3] = (20240520, 1, ntok)
            body = (np.arange(ntok) * 7 % 97 + rs.integers(0, 3, ntok)).astype(np.uint16)   # learnable stand-in text
            with open(f / name, "wb") as fh:
                fh.write(head.tobytes())
                fh.write(body.tobytes())
        rr = speedrun.main(["--data", sd, "--tokens", "60000", "--val-every", "30000", "--micro", "4", "--batch-tokens", "2048",
                            "--target", "5.0", "--val-tokens", "8192", "--warmup", "3", "--lr", "6e-3", "--device", str(dev), "--compile", "0", "--set",
                            "d_model=64", "n_layer=4", "attn_heads=4", "attn_kv_heads=2", "attn_head_dim=16", "mlp_hidden=128",
                            "max_len=128", "ctx=64:0,128:0.3"])
    check("speedrun mode reads the records' shards in order, scores the fixed held-out tokens, reports the target",
          rr["final"]["val"] < 5.0 and rr["log"][1]["val"] < rr["log"][0]["val"] and rr["hit"] is not None and len(rr["log"]) == 2, f"held-out {rr['final']['val']:.3f}, "
          f"target 5.0 at {rr['hit']['tokens'] if rr['hit'] else '-'} tokens")

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
    argv = ["train", "--model", "bee5", "--preset", "smoke", "--data", str(data), "--runs", str(runs), "--mentor", "smol-random",
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
    rd = runs / "bee5"
    recs = [json.loads(l_) for l_ in (rd / "metrics.jsonl").read_text().splitlines()]
    tr = [r for r in recs if r["kind"] == "train"]
    cfg_run = json.loads((rd / "config.json").read_text())
    check("the run used Bee5's body and Bee4's recipe", cfg_run["mentor_share"] is True and cfg_run["mentor_weight"] == 0.25
          and cfg_run.get("decay") == 0.3 and "value tables" in cfg_run["layout"], cfg_run["layout"])
    mws = [r.get("mentor_w", 0) for r in tr]
    check("the mentor taught (1 in 4 at the start)", 0.15 < mws[0] <= 0.25, f"{mws[0]:.2f} ... {mws[-1]:.2f}")
    ctxs = [r.get("context") for r in tr]
    check("the windows grew inside the run (64 -> 128 words at 30%)", ctxs[0] == 64 and ctxs[-1] == 128)
    st = json.loads((rd / "status.json").read_text())
    check("the run finished, with held-out loss and HellaSwag", st["state"] == "done" and "val_loss" in st["latest"],
          f"val loss {st['latest'].get('val_loss', float('nan')):.3f}")
    mdl, ck = train.load_model(rd / "ckpt.pt", dev)
    check("its checkpoint loads as Bee5", ck["kind"] == "bee5" and type(mdl).__name__ == "Bee5")
    out, _ = mdl.generate(enc.encode_ordinary("The cat"), max_new_tokens=5)
    check("the trained model writes", len(out) <= 5, repr(enc.decode(out)))
    n_before = len(recs)
    sys.argv = argv + ["--data-sub", "smol"]
    train.main()                         # the same line again (a pod restart): it loads the checkpoint and its optimizer
    recs2 = [json.loads(l_) for l_ in (rd / "metrics.jsonl").read_text().splitlines()]
    st2 = json.loads((rd / "status.json").read_text())
    check("run again, it picks up its checkpoint (Bee5's optimizer groups load back) instead of starting over",
          st2["state"] == "done" and len(recs2) == n_before and st2["tokens"] == st["tokens"], f"{st2['tokens']:,} tokens")

    if dev.type == "cuda":
        mg = Bee5(Bee5Config(**{**SMALL, "vocab_size": 49152})).to(dev).train()
        mg.set_progress(0.0)
        xg = torch.randint(1, 49152, (4, 128), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = mg.losses(xg, xg)
        loss.backward()
        check("bf16 training step on the GPU (3 next words)", torch.isfinite(loss).item())
        mg.zero_grad(set_to_none=True)
        mg.enable_compile()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lc, _ = mg.losses(xg, xg)
        lc.backward()
        check("the compiled training step runs and agrees", torch.isfinite(lc).item() and abs(lc.item() - loss.item()) < 0.05,
              f"{lc.item():.4f} vs {loss.item():.4f}")
        mc = Bee5(Bee5Config(**{**SMALL, "vocab_size": 49152, "attn_backend": "cudnn"})).to(dev)
        mc.load_state_dict(mg.state_dict())
        mc.eval()
        mg.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            e = (mc(xg)[0].float() - mg(xg)[0].float()).abs().max().item()
        check("cuDNN attention leaves the maths alone", e < 0.1, f"max diff {e:.1e}")
    tmp_ctx.cleanup()
    term.tests_passed("Bee5-200M")


if __name__ == "__main__":
    main()
