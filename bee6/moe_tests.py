"""Self-tests for Bee6-MoE (bee6/moe.py, bee6/experts.py): the size (872M, Bee6's 202M at work per word); the queues =
the weights taken directly = a word-by-word reference; reading never depends on the other words of a batch, and stays
causal; full queues drop and say so; every part learns (experts, shared expert, router); NorMuon straightens each
expert's grid; the balancing biases even the experts out; one whole tiny run through the trainer, with a resume.

    python3 -m bee6.moe_tests                  (CPU, ~1-2 min; without internet it uses stand-in tokenizers)
    python3 -m bee6.moe_tests --device cuda    (on the pod: also the compiled step, and Bee6 vs Bee6-MoE speed at full size)
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

from oazy import term

SMALL = dict(d_model=64, n_layer=6, attn_heads=4, attn_kv_heads=2, attn_head_dim=16, max_len=128, ctx="128:0",
             vocab_size=512, key_offset="1,5", mod_layers="2,4", fp8=0, experts=8, expert_top=2, expert_hidden=16)


def check(name, ok, detail=""):
    term.check(name, ok, detail)


def jiggle(m, s=0.05):
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * s)
    return m


def reference(ex, x):
    """Word by word, expert by expert, in float32: what the experts layer means."""
    idx, w = ex.route(x)[:2]
    out = []
    for n in range(x.shape[0]):
        y = ex.shared(x[n:n + 1]).float()
        for j, e in enumerate(idx[n].tolist()):
            g = ex.S + e
            a = F.relu(F.linear(x[n:n + 1], ex.w_up[g])).square()
            y = y + w[n, j] * F.linear(a, ex.w_down[g]).float()
        out.append(y)
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    dev = torch.device(ap.parse_args().device)
    term.tests_start("Bee6-MoE", dev)
    from bee6.model import Bee6, Bee6Config
    from bee6.moe import Bee6MoE, Bee6MoEConfig
    from bee6.experts import Experts

    # ---- the size
    big, b6 = Bee6MoE(Bee6MoEConfig()), Bee6(Bee6Config())
    n, act = big.num_params(), big.active_params()
    check("Bee6-MoE is ~872M, with Bee6's ~202M at work per word (1 shared + 9 of 80 experts of 384 = Bee6's 3,840)",
          860e6 < n < 885e6 and abs(act - b6.num_params()) < 2e6,
          f"{n / 1e6:.1f}M total, {act / 1e6:.1f}M per word (Bee6 {b6.num_params() / 1e6:.1f}M); {big.layout()}")
    del big, b6

    # ---- the layer alone: queues = direct = the reference
    torch.manual_seed(3)
    cfg = Bee6MoEConfig(**SMALL)
    ex = jiggle(Experts(cfg).to(dev), 0.1)
    x = torch.randn(200, 64, device=dev)
    ex.balance(torch.ones(8), torch.randn(8, device=dev) * 0.1, torch.ones(8, device=dev))   # some running averages
    ex.eval()
    ref = reference(ex, x)
    e_read = (ex(x) - ref).abs().max().item()
    e_dir = (ex(x[:5]) - ref[:5]).abs().max().item()
    ex.train()
    cfg.expert_capacity = 100.0
    e_q = (ex(x) - reference(ex, x)).abs().max().item()
    cfg.expert_capacity = 1.25
    check("the experts' queues (reading, and learning with room for all) = each word's experts taken directly = "
          "word by word", max(e_read, e_dir, e_q) < 1e-4, f"max diff {max(e_read, e_dir, e_q):.1e}")
    ex.eval()
    alone = torch.cat([ex(x[i:i + 1]) for i in range(0, 40)])
    e_alone = (ex(x)[:40] - alone).abs().max().item()
    check("reading: a word's answer does not depend on the other words of the batch", e_alone < 1e-4, f"max diff {e_alone:.1e}")
    ex.train()
    with torch.no_grad():
        ex.router.weight.zero_()
        ex.router.weight[:, 0] = 5.0                     # every word wants the same experts: their queues overflow
        ex.bias_route[:3] = 50.0                         # (by the balancing biases, which the centring keeps)
    _, (_, counts, drop, _, _) = ex(x, with_aux=True)
    check("a full queue drops (and says how much), the answers stay finite", 0.2 < float(drop) < 1.0
          and int(counts.max()) == 200, f"dropped {float(drop):.0%}")

    # ---- the whole model
    torch.manual_seed(5)
    m = jiggle(Bee6MoE(Bee6MoEConfig(**SMALL)).to(dev))
    m.set_progress(0.0)
    xi = torch.randint(1, 512, (2, 64), device=dev)
    m.train()
    loss, st = m.losses(xi, xi)
    loss.backward()
    b = m.blocks[0].mlp
    got = {k: (p.grad is not None and p.grad.abs().sum().item() > 0) for k, p in
           (("routed", b.w_up), ("down", b.w_down), ("router", b.router.weight))}
    sh = b.w_up.grad[0].abs().sum().item() > 0
    check("every part learns: the routed experts, the shared expert, the router (through its weights)",
          all(got.values()) and sh and torch.isfinite(loss).item(), f"{got}, shared {sh}")
    m.zero_grad(set_to_none=True)
    m.eval()
    with torch.no_grad():
        l1, _ = m.read(xi)
        x2 = xi.clone()
        x2[:, 40:] = torch.randint(1, 512, (2, 24), device=dev)
        l2, _ = m.read(x2)
    e_c = (l1[:, :40] - l2[:, :40]).abs().max().item()
    check("reading Bee6-MoE is causal (later words never change earlier scores)", e_c < 1e-4, f"max diff {e_c:.1e}")

    # ---- NorMuon on the expert stacks, balancing, learning
    from oazy.train import make_optimizer
    m.train()
    opt = make_optimizer(m, 3e-3, dev, "muon", 0.03, 0.1)
    mu = opt.opts[0]
    stacks = [p for g in mu.param_groups for p in g["params"] if p.dim() == 3]
    check("NorMuon straightens the experts (every layer's up and down stacks)", len(stacks) == 2 * SMALL["n_layer"],
          f"{len(stacks)} stacks")
    first = last = None
    rep = {}
    for i in range(60):
        loss, _ = m.losses(xi, xi)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        m.after_step()
        rep = m.thought_report()
        first = first if first is not None else loss.item()
        last = loss.item()
    second = mu.state[stacks[0]]["second"]
    bias = sum(float(bl.mlp.bias_route.abs().sum()) for bl in m.blocks)
    check("60 steps on one batch: the loss falls, the balancing biases move, the log shows the experts",
          last < first - 1.5 and bias > 0 and {"experts_dropped", "experts_uneven", "experts_idle"} <= rep.keys()
          and second.shape[:1] == stacks[0].shape[:1],
          f"{first:.2f} -> {last:.2f}; busiest / average {rep.get('experts_uneven', float('nan')):.2f}, "
          f"dropped {rep.get('experts_dropped', float('nan')):.1%}")
    bad = False
    try:
        Bee6MoE(Bee6MoEConfig(**{**SMALL, "bloom": "on"}))
    except ValueError:
        bad = True
    check("experts with bloom is refused (not built)", bad)

    # ---- one whole tiny run through the trainer, then a resume
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
    argv = ["train", "--model", "bee6moe", "--preset", "smoke", "--data", str(data), "--runs", str(runs), "--mentor", "smol-random",
            "--device", str(dev), "--compile", "0", "--tokens", "40000", "--batch-ramp", "512:0.5", "--mix", "fwe:0.9,chat:0.1@0",
            "--micro", "2", "--data-sub", "smol", "--set", "ctx=64:0,128:0.3", "attn_head_dim=16", "attn_heads=4", "attn_kv_heads=2"]
    sys.argv = argv
    train.main()
    rd = runs / "bee6moe"
    recs = [json.loads(l_) for l_ in (rd / "metrics.jsonl").read_text().splitlines()]
    tr = [r for r in recs if r["kind"] == "train"]
    st = json.loads((rd / "status.json").read_text())
    check("a whole tiny run: it trains, the log shows the experts, it finishes with held-out loss and HellaSwag",
          st["state"] == "done" and "val_loss" in st["latest"] and "experts_uneven" in tr[-1] and tr[-1]["experts_dropped"] < 0.2,
          f"busiest / average {tr[-1].get('experts_uneven', float('nan')):.2f}, dropped {tr[-1].get('experts_dropped', float('nan')):.1%}, "
          f"val loss {st['latest'].get('val_loss', float('nan')):.3f}")
    mdl, ck = train.load_model(rd / "ckpt.pt", dev)
    cfgj = json.loads((rd / "config.json").read_text()) if (rd / "config.json").exists() else {}
    check("its checkpoint loads as Bee6-MoE with its experts and balancing biases; the run records the per-word size",
          ck["kind"] == "bee6moe" and type(mdl).__name__ == "Bee6MoE" and mdl.cfg.experts == 8
          and float(mdl.blocks[0].mlp.bias_route.abs().sum()) > 0 and cfgj.get("params_active", 0) > 0,
          f"{cfgj.get('params', 0):,} params, {cfgj.get('params_active', 0):,} per word")
    out, _ = mdl.generate(enc.encode_ordinary("The cat"), max_new_tokens=5)
    check("the trained model writes, one word at a time", len(out) <= 5, repr(enc.decode(out)))
    n_before = len(recs)
    sys.argv = argv
    train.main()
    recs2 = [json.loads(l_) for l_ in (rd / "metrics.jsonl").read_text().splitlines()]
    st2 = json.loads((rd / "status.json").read_text())
    check("run again, it picks up its checkpoint instead of starting over",
          st2["state"] == "done" and len(recs2) == n_before and st2["tokens"] == st["tokens"], f"{st2['tokens']:,} tokens")

    if dev.type == "cuda":
        mg = jiggle(Bee6MoE(Bee6MoEConfig(**{**SMALL, "vocab_size": 49152, "fp8": 1}))).to(dev).train()
        mg.startup_check(dev, lambda s_: term.info(s_))
        mg.set_progress(0.0)
        xg = torch.randint(1, 49152, (4, 128), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = mg.losses(xg, xg)
        loss.backward()
        mg.zero_grad(set_to_none=True)
        mg.enable_compile()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lc, _ = mg.losses(xg, xg)
        lc.backward()
        check("the compiled training step runs and agrees", torch.isfinite(lc).item() and abs(lc.item() - loss.item()) < 0.05,
              f"{lc.item():.4f} vs {loss.item():.4f}")
        del mg
        speed(dev)
    tmp_ctx.cleanup()
    term.tests_passed("Bee6-MoE")


def speed(dev, micro=8, T=2048, steps=6):
    """Full size, compiled, one pass forward + back at 2,048 words: Bee6 vs Bee6-MoE (the student alone, no mentor)."""
    from bee6.model import Bee6, Bee6Config
    from bee6.moe import Bee6MoE, Bee6MoEConfig
    res = {}
    for name, cls, Cfg in (("Bee6", Bee6, Bee6Config), ("Bee6-MoE", Bee6MoE, Bee6MoEConfig)):
        torch.manual_seed(0)
        m = jiggle(cls(Cfg(ctx=f"{T}:0"))).to(dev).train()
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
        term.info(f"{name}: {res[name] / 1e3:.0f}k tokens/s forward + back (student alone), peak memory {mem:.0f} GB at micro {micro}")
        del m, x
        torch.cuda.empty_cache()
    r = res["Bee6-MoE"] / res["Bee6"]
    check("speed at full size: Bee6-MoE vs Bee6 (expected ~0.65-0.8x)", r > 0.4, f"{r:.2f}x")


if __name__ == "__main__":
    main()
