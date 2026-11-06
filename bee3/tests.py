"""Self-tests for Bee3: the body is Bee2's; no bleed between documents; the loss in slices gives the same
numbers as the full scores; the recipe's knobs (mentor floor, batch ramp, square-root cool-down, Muon weight
decay); the mentor reads each document on its own; and one whole (tiny) run end to end.

    python -m bee3.tests                 (CPU, ~2 min; without internet it uses stand-in tokenizers)
    python -m bee3.tests --device cuda   (on the pod: the real tokenizer, the fast memory kernel, bf16)
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


def check(name, ok, detail=""):
    print(f"  {'ok ' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        raise SystemExit(f"self-test failed: {name}")


SMALL = dict(d_model=64, n_layer=8, n_head=2, head_k=16, head_v=16, attn_heads=4, attn_kv_heads=2,
             attn_head_dim=16, mlp_hidden=96, max_len=128, ctx="128:0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    dev = torch.device(ap.parse_args().device)
    print(f"Bee3 self-tests on {dev}")
    torch.manual_seed(0)
    from bee3.model import Bee3, Bee3Config
    from bee2.model import Bee2Config
    from bee.model import Bee, BeeConfig
    from oazy import tok
    from oazy.lmloss import head_loss, plain_loss
    from memtrix import memory

    # ---- the body is Bee2's; what differs is how it reads
    big = Bee3(Bee3Config())
    n = big.num_params()
    check("full size is about 0.2B", 195e6 < n < 215e6, f"{n / 1e6:.1f}M parameters, {big.layout()}")
    diff = {k: (getattr(big.cfg, k), v) for k, v in Bee2Config().to_dict().items() if getattr(big.cfg, k) != v}
    check("same body as Bee_v2_0.2B; only the windows and the document rule differ", set(diff) == {"ctx", "doc_mask"}, str(diff))
    check("one window length, 2,048, the whole run", big.ctx_at(0) == 2048 and big.ctx_at(0.99) == 2048)
    check("SmolLM2's pieces; <|endoftext|> is 0", big.cfg.tokenizer == "smollm2" and tok.eot_id("smollm2") == 0)
    del big

    # ---- no bleed between documents: a packed window reads each document exactly as if alone
    m = Bee3(Bee3Config(**SMALL, vocab_size=320)).to(dev).eval()
    a, b, c = (torch.randint(1, 320, (1, k), device=dev) for k in (30, 50, 20))
    eot = torch.zeros(1, 1, dtype=torch.long, device=dev)
    packed = torch.cat([a, eot, b, eot, c], 1)
    with torch.no_grad():
        full = m._body(packed)
        eb = (full[:, 31:81] - m._body(b)).abs().max().item()
        ec = (full[:, 82:] - m._body(c)).abs().max().item()
        resets, mask = m.doc_info(packed)
    check("documents in one window read exactly as if alone (memories, attention, convolution)", eb < 1e-3 and ec < 1e-3,
          f"max diff {max(eb, ec):.1e}")
    check("a document starts right after <|endoftext|>", resets[0].nonzero()[:, 0].tolist() == [31, 82] and mask.shape == (1, 1, 102, 102)
          and not mask[0, 0, 40, 10] and mask[0, 0, 40, 35] and not mask[0, 0, 35, 40])
    m0 = Bee(BeeConfig(**{**SMALL, "vocab_size": 320}, tokenizer="smollm2")).to(dev).eval()
    m0.load_state_dict(m.state_dict())
    with torch.no_grad():
        bleed = (m0._body(packed)[:, 31:81] - m._body(b)).abs().max().item()
    check("without the rule (Bee2) the documents bleed into each other", bleed > 1e-2, f"diff {bleed:.2f}")
    # the memory rule on its own: documents starting blank = running them one by one
    B, T, H, K = 2, 150, 3, 16
    q, k = (torch.nn.functional.normalize(torch.randn(B, T, H, K, device=dev), dim=-1) for _ in range(2))
    v, lf, tr = torch.randn(B, T, H, K, device=dev), -torch.rand(B, T, H, device=dev) * 0.3, torch.rand(B, T, H, device=dev)
    rs = torch.zeros(B, T, dtype=torch.bool, device=dev)
    rs[0, 40] = rs[0, 100] = rs[1, 7] = True
    o, S = memory.delta_rule(q, k, v, lf, tr, resets=rs)
    ref = torch.zeros_like(o)
    for bi in range(B):
        cuts = [0] + rs[bi].nonzero()[:, 0].tolist() + [T]
        for s0, s1 in zip(cuts[:-1], cuts[1:]):
            ref[bi:bi + 1, s0:s1], Sb = memory.recurrent(q[bi:bi + 1, s0:s1], k[bi:bi + 1, s0:s1], v[bi:bi + 1, s0:s1],
                                                        lf[bi:bi + 1, s0:s1], tr[bi:bi + 1, s0:s1])
        check(f"memory starts blank at each document (row {bi}, final state too)", (S[bi] - Sb[0]).abs().max().item() < 2e-3)
    tol = 5e-2 if dev.type == "cuda" else 1e-4
    check("memory starts blank at each document (answers)", (o - ref).abs().max().item() < tol, f"max diff {(o - ref).abs().max().item():.1e}")
    cu, last = memory.segments(rs)
    check("the kernel's packed form: offsets and each row's last document", cu.tolist() == [0, 40, 100, 150, 157, 300] and last.tolist() == [2, 4])

    # ---- the loss in slices = the loss on the full scores (and its gradients)
    N, D, V = 37, 16, 50
    h = torch.randn(N, D, device=dev, requires_grad=True)
    W = torch.randn(V, D, device=dev, requires_grad=True)
    tg = torch.randint(0, V, (N,), device=dev)
    tg[3] = tg[10] = -1                                   # not scored (the chat fine-tune does this)
    th, tW = torch.randn(N, 12, device=dev), torch.randn(47, 12, device=dev)
    worst = 0.0
    for kd in (0.0, 0.6):
        for cap in (0.0, 5.0):
            l1, c1, k1 = head_loss(h, W, tg, cap, th, tW, kd, chunk=8)
            g1 = torch.autograd.grad(l1, (h, W))
            l2, c2, k2 = plain_loss(h, W, tg, cap, th, tW, kd)
            g2 = torch.autograd.grad(l2, (h, W))
            worst = max(worst, abs(l1.item() - l2.item()), abs(c1.item() - c2.item()), abs(k1.item() - k2.item()),
                        *[(x - y).abs().max().item() for x, y in zip(g1, g2)])
    tol = 1e-2 if dev.type == "cuda" else 1e-3                # the GPU's fp32 matmuls round more (TF32)
    check("loss in slices = loss on the full scores, gradients too (with/without soft cap and mentor)", worst < tol, f"max diff {worst:.1e}")
    if dev.type == "cuda":                                    # as in training: bf16 under autocast, the mentor in bf16
        th16, tW16 = th.to(torch.bfloat16), tW.to(torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l1, _, _ = head_loss(h, W, tg, 5.0, th16, tW16, 0.6, chunk=8)
            l2, _, _ = plain_loss(h, W, tg, 5.0, th16, tW16, 0.6)
        g1 = torch.autograd.grad(l1, (h, W))
        g2 = torch.autograd.grad(l2, (h, W))
        w16 = max(abs(l1.item() - l2.item()), *[(x - y).abs().max().item() for x, y in zip(g1, g2)])
        check("the same under bf16 autocast with a bf16 mentor (the training path)", w16 < 5e-2, f"max diff {w16:.1e}")
    m.train()
    x, y = packed, torch.roll(packed, -1, 1)
    _, l1, _ = m(x, y)
    g1 = torch.autograd.grad(l1, m.embed.weight)[0]
    Bee.fused_loss = False
    _, l2, _ = m(x, y)
    g2 = torch.autograd.grad(l2, m.embed.weight)[0]
    Bee.fused_loss = True
    check("the model's loss, both ways", abs(l1.item() - l2.item()) < 1e-3 and (g1 - g2).abs().max().item() < 1e-3,
          f"{l1.item():.4f} vs {l2.item():.4f}")

    # ---- the mentor reads each document on its own, and hands over token vectors, not scores
    from ember.mentor import Mentor, fade
    me = Mentor("smol-random", dev, compile=False)
    mm = Bee3(Bee3Config(**SMALL)).to(dev).train()
    _, mask = mm.doc_info(packed)
    hh, hw = me.hidden(packed, mask)
    hb, _ = me.hidden(b, None)
    check("mentor: token vectors + word table", hh.shape[:2] == packed.shape and hw.shape[0] == 49152)
    rel = (hh[:, 31:81] - hb).abs().max().item() / (hb.abs().max().item() + 1e-6)
    check("mentor reads each document on its own", me._mask_form != "none" and rel < (2e-2 if dev.type == "cuda" else 1e-3),
          f"mask form {me._mask_form}, rel. error {rel:.4f}")    # bf16 on the GPU: ~0.5% rounding is normal
    loss, st = mm.losses(x, y, me, 0.5)
    Bee.fused_loss = False
    loss2, st2 = mm.losses(x, y, me, 0.5)
    Bee.fused_loss = True
    check("student + mentor lesson in slices = the full-score lesson", abs(loss.item() - loss2.item()) < 1e-3 and "kd" in st,
          f"loss {loss.item():.3f}, next-word {float(st['ce']):.3f}, mentor {float(st['kd']):.3f}")

    # ---- the recipe's knobs
    f = lambda p_: fade(p_, 0.6, 0.5, None, 0.25, 0.1)   # noqa: E731
    check("mentor share: 0.6 -> 0.25 by halfway, held, -> 0 over the last 10%",
          abs(f(0) - 0.6) < 1e-9 and abs(f(0.25) - 0.425) < 1e-9 and abs(f(0.5) - 0.25) < 1e-9 and abs(f(0.8) - 0.25) < 1e-9
          and abs(f(0.95) - 0.125) < 1e-9 and f(1.0) == 0.0)
    check("Bee2's fade is unchanged (floor 0, off 0)", fade(0.25, 0.6, 0.5) == 0.3 and fade(0.5, 0.6, 0.5) == 0.0)
    from oazy.train import lr_wsd, batch_at, make_optimizer, RECIPE
    check("square-root cool-down: flat, then falls faster at first and lands at 0",
          lr_wsd(0.5, 100, 1.0, 10, shape="sqrt") == 1.0 and abs(lr_wsd(0.9, 900, 1.0, 10, shape="sqrt") - (1 - 0.5 ** 0.5)) < 1e-9
          and lr_wsd(1.0, 1000, 1.0, 10, shape="sqrt") == 0.0 and abs(lr_wsd(0.9, 900, 1.0, 10) - 0.5) < 1e-9)
    check("batch ramp: 64k at the start, the full batch by 10%",
          batch_at(0.0, 262144, "65536:0.1") == 65536 and batch_at(0.05, 262144, "65536:0.1") == 163840
          and batch_at(0.1, 262144, "65536:0.1") == 262144 and batch_at(0.0, 2048, "65536:0.1") == 2048 and batch_at(0.5, 100, None) == 100)
    check("Bee3's recipe is on by default", RECIPE["bee3"]["mentor_floor"] == 0.25 and RECIPE["bee3"]["muon_wd"] == 0.1)
    opt = make_optimizer(m, 1e-3, dev, "muon", 0.02, muon_wd=0.1)
    grid = m.blocks[0].mlp.up.weight
    before = grid.detach().clone()
    for g_ in opt.param_groups:
        g_["lr"] = 1e-3 * g_.get("lr_scale", 1.0)
    grid.grad = torch.zeros_like(grid)
    opt.step()
    shrink = (grid.detach() / before).mean().item()
    check("Muon weight decay on AdamW's scale: a grid with no gradient shrinks by lr x 0.1 per step", abs(shrink - (1 - 1e-4)) < 1e-6,
          f"x{shrink:.6f}")
    opt.zero_grad(set_to_none=True)

    # ---- one whole tiny run on stand-in text: the Bee3 recipe end to end
    from bee2.tests import stand_ins, fake_docs
    from oazy.data import get_tokenizer, EOT
    from oazy import retok, train
    tmp_ctx = tempfile.TemporaryDirectory()
    tmp = Path(tmp_ctx.name)
    used = stand_ins(tmp)
    if used:
        print(f"  info  no internet: stand-in {' and '.join(used)} tokenizer(s)")
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
    sys.argv = ["train", "--model", "bee3", "--preset", "smoke", "--data", str(data), "--runs", str(runs), "--mentor", "smol-random",
                "--device", str(dev), "--compile", "0", "--tokens", "60000", "--batch-ramp", "512:0.5", "--mix", "fwe:0.9,chat:0.1@0"]
    train.main()
    rd = runs / "bee3"
    cfg = json.loads((rd / "config.json").read_text())
    recs = [json.loads(l) for l in (rd / "metrics.jsonl").read_text().splitlines()]
    tr = [r for r in recs if r["kind"] == "train"]
    ev = [r for r in recs if r["kind"] == "eval"]
    check("the run used Bee3's recipe", cfg["doc_mask"] and cfg["mentor_floor"] == 0.25 and cfg["mentor_off"] == 0.1
          and cfg["muon_wd"] == 0.1 and cfg["decay_shape"] == "sqrt" and cfg["loss"] == "fused" and cfg["batch_ramp"] == "512:0.5")
    check("the batch ramped up", tr[0]["batch"] < tr[-1]["batch"] == 2048, f"{tr[0]['batch']} -> {tr[-1]['batch']}")
    mws = [r.get("mentor_w", 0) for r in tr]
    check("the mentor stayed the whole run and faded out at the very end", mws[0] > 0.4 and all(w > 0 for w in mws[:-1])
          and min(mws[len(mws) // 2:-1]) >= 0.249 and mws[-1] < 0.25, f"{mws[0]:.2f} ... {mws[len(mws) // 2]:.2f} ... {mws[-1]:.3f}")
    check("held-out loss per GPT-2 token, bits per byte, per-source loss", ev and all("val_bpb" in e and "val_loss_chat" in e for e in ev))
    st = json.loads((rd / "status.json").read_text())
    check("the run finished", st["state"] == "done", f"val loss {st['latest']['val_loss']:.3f}")
    mdl, ck = train.load_model(rd / "ckpt.pt", dev)
    check("its checkpoint loads as Bee3", ck["kind"] == "bee3" and mdl.cfg.doc_mask == 1)
    out, _ = mdl.generate(enc.encode_ordinary("The cat"), max_new_tokens=5)
    check("it writes", len(out) <= 5, repr(enc.decode(out)))

    if dev.type == "cuda":
        print(f"  info  fast memory kernel: {'yes' if memory._FLA_OK else 'no'}; packed documents in the kernel: {'yes' if memory._FLA_SEG else 'no'}")
        mg = Bee3(Bee3Config(**{**SMALL, "head_k": 64, "head_v": 64})).to(dev)
        xg = torch.randint(1, 49152, (4, 128), device=dev)
        xg[:, 50] = 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss, _ = mg(xg, xg)
        loss.backward()
        check("bf16 training step on the GPU with the document rule", torch.isfinite(loss).item())
    tmp_ctx.cleanup()
    print("all Bee3 self-tests passed")


if __name__ == "__main__":
    main()
