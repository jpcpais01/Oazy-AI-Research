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
    check("MoE-150M: ~150M weights in all, ~30M of its layers at work per word, 8,192 pieces",
          145e6 < n < 155e6 and 28e6 < a - t < 31e6 and moe.cfg.vocab_size == 8192 and moe.cfg.tokenizer == "oazy8k",
          f"{n / 1e6:.1f}M in all, {(a - t) / 1e6:.1f}M layers per word, {a / 1e6:.1f}M counting the {t / 1e6:.1f}M of tables")
    nd, td = den.num_params(), tables(den)
    check("Dense-30M: the same body and the same work per word as the MoE (within 1%)",
          abs((nd - td) - (a - t)) / (a - t) < 0.01, f"{nd / 1e6:.1f}M in all, {(nd - td) / 1e6:.1f}M layers")
    c = moe.cfg
    check("only the clean tricks: no FP8, no Mixture of Depths, no bloom; MTP gone by 5%; the grouped expert product",
          c.fp8 == 0 and c.mod_layers == "" and not c.bloom and c.mtp_until == 0.05 and c.fuse_qkv == 1
          and c.loss_chunk == 4096 and c.expert_kernel == "grouped" and moe.blocks[0].mlp.kernel == "grouped")
    check("the recipe is Bee6's", RECIPE["moe150"] == RECIPE["bee6"] == RECIPE["dense30"])
    mix = resolve_mix("opal100:flat:0.02")
    phases = [dict((k, float(v)) for k, v in (s.split(":") for s in p.split("@")[0].split(","))) for p in mix.split(";")]
    check("Opal for it: Field1 2% of every phase, chat 12% of the cool-down text, every phase adds to 1",
          all(abs(p["field1"] - 0.02) < 1e-9 and abs(sum(p.values()) - 1) < 1e-4 for p in phases)
          and abs(phases[-1]["chat"] - 0.12 * 0.98) < 1e-4 and "@0.8" in OPAL100_TEXT_MIX, mix[:90] + "…")

    # ---- the grouped product = the queues = word by word, forward and back
    from bee6.experts import Experts
    from bee6.moe import grouped_works
    from bee6.moe_tests import reference
    ok, why = grouped_works(torch.device("cpu"), 64, 48)
    check("torch's grouped product works here (forward and back match expert by expert)", ok, why)
    sm = Bee6MoEConfig(d_model=64, experts=8, expert_top=2, expert_hidden=32, expert_kernel="grouped")
    torch.manual_seed(1)
    ex = Experts(sm)
    with torch.no_grad():
        ex.w_down.normal_(0, 0.05)
        ex.router.weight.normal_(0, 0.5)
    xs = torch.randn(300, 64, requires_grad=True)
    ex.train()
    yg, (_, cnt, drop, _, _) = ex(xs, with_aux=True)
    g = torch.randn_like(yg)
    (yg * g).sum().backward()
    gx, gu = xs.grad.clone(), ex.w_up.grad.clone()
    xs.grad = None
    ex.zero_grad()
    ex.kernel = "queues"
    sm.expert_capacity = 8.0                        # room for every word: the queues drop nothing either
    yq, _ = ex(xs, with_aux=True)
    (yq * g).sum().backward()
    with torch.no_grad():
        yr = reference(ex, xs)
    check("the grouped product = the queues = word by word (answers and gradients), and it drops nothing",
          torch.allclose(yg, yq, atol=1e-5) and torch.allclose(yg, yr.to(yg.dtype), atol=1e-5)
          and torch.allclose(gx, xs.grad, atol=1e-5) and torch.allclose(gu, ex.w_up.grad, atol=1e-5) and float(drop) == 0,
          f"max gap {float((yg - yq).abs().max()):.1e}")
    ex.kernel = "grouped"
    ex.eval()
    with torch.no_grad():
        one = torch.cat([ex(xs[i:i + 30]) for i in range(0, 300, 30)])
        allw = ex(xs)
    check("reading with the grouped product never depends on the other words of a batch", torch.allclose(one, allw, atol=1e-5))

    torch.manual_seed(0)
    for p in moe.parameters():           # a jiggled MoE-150M: reading is causal at full shape (a short window)
        with torch.no_grad():
            p.add_(torch.randn_like(p) * 0.02)
    moe.eval()
    x = torch.randint(1, 8192, (1, 64))
    y = x.clone()
    y[0, 40:] = torch.randint(1, 8192, (24,))
    with torch.no_grad():
        l1, l2 = moe(x)[0], moe(y)[0]
    check("full-size MoE-150M reads causally (changing words 40+ leaves 0..39 the same)",
          torch.allclose(l1[:, :40].float(), l2[:, :40].float(), atol=1e-4))
    del moe, den

    # ---- our 8k pieces: learned from a (tiny, fake) Opal, Opal re-cut, then a whole tiny run of each through the trainer
    from bee2.tests import stand_ins, fake_docs
    from oazy.data import get_tokenizer, EOT
    from oazy import retok, train, tok, tok8k
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
            if sub == "chat":
                ds = [f"User: {u} 12 + 34 = 46\n\nAssistant: {v_}" for u, v_ in zip(ds, fake_docs(kk, rng))]
            np.fromiter((t_ for x_ in ds for t_ in gpt.encode_ordinary(x_) + [EOT]), dtype=np.uint16).tofile(d / f"{split}.bin")
    (data / "hellaswag_val.jsonl").write_text("".join(
        json.dumps({"ctx": " ".join(fake_docs(1, rng)), "endings": fake_docs(4, rng), "label": rng.randint(0, 3)}) + "\n" for _ in range(24)))
    retok.run(data, "smollm2", ["fwe", "chat"], workers=2)
    (data / "smol").rename(data / "opal")                 # a fake Opal: web text, chat, and the chat again as Field1
    import shutil
    shutil.copytree(data / "opal" / "chat", data / "opal" / "field1")
    tok8k.train(data, 0.002, 600)
    tk8 = tok.get("oazy8k")
    tsm = tok.get("smollm2")
    tok8k.cut(data, 2)
    o, n8 = data / "opal" / "train.bin", data / "opal8k" / "train.bin"
    a_ = np.fromfile(o, dtype=np.uint16)
    b_ = np.fromfile(n8, dtype=np.uint16)
    ea, eb = np.flatnonzero(a_ == 0), np.flatnonzero(b_ == 0)
    same = len(ea) == len(eb) and all(tsm.decode(a_[(ea[i - 1] + 1 if i else 0):ea[i]].tolist())
                                      == tk8.decode(b_[(eb[i - 1] + 1 if i else 0):eb[i]].tolist()) for i in range(0, len(ea), 97))
    meta = json.loads((data / "opal8k" / "meta.json").read_text())
    check("8k pieces: <|endoftext|> = 0, digits one by one; re-cut Opal = the same documents, the same text, every folder",
          tk8.eot_token == 0 and len(tk8.encode_ordinary("2024")) == 4 and same
          and all((data / "opal8k" / d / "train.bin").exists() for d in ("chat", "field1"))
          and meta["tokenizer"] == "oazy8k" and meta["val"]["ratio"] > 0,
          f"{len(ea)} documents, {meta['train']['bytes'] / max(meta['train']['tokens'], 1):.2f} bytes per piece")
    runs = tmp / "runs"
    for kind in ("moe150", "dense30"):
        argv = ["train", "--model", kind, "--preset", "smoke", "--data", str(data), "--runs", str(runs), "--mentor", "none",
                "--device", str(dev), "--compile", "0", "--tokens", "40000", "--batch-ramp", "512:0.5",
                "--micro", "2", "--data-sub", "opal", "--set", "ctx=64:0,128:0.3", "attn_head_dim=16", "attn_heads=4", "attn_kv_heads=2"]
        sys.argv = argv
        train.main()
        rd = runs / kind
        recs = [json.loads(l_) for l_ in (rd / "metrics.jsonl").read_text().splitlines()]
        st = json.loads((rd / "status.json").read_text())
        mdl, ck = train.load_model(rd / "ckpt.pt", dev)
        cfgj = json.loads((rd / "config.json").read_text())
        check(f"{kind}: a whole tiny run on the re-cut Opal (Field1 2%) trains, finishes with held-out loss and bits per "
              f"byte, keeps its tokenizer, and its checkpoint loads",
              st["state"] == "done" and "val_bpb" in st["latest"] and ck["kind"] == kind and (rd / "tokenizer.json").exists()
              and type(mdl).__name__ == ("Bee6MoE" if kind == "moe150" else "Bee6") and "field1:0.02" in cfgj.get("mix", ""),
              f"val loss {st['latest'].get('val_loss', float('nan')):.3f}, {st['latest'].get('val_bpb', float('nan')):.3f} bits per byte")
        if kind == "moe150":
            tr = [r for r in recs if r["kind"] == "train"]
            check("the MoE's log shows its experts, none dropped", "experts_uneven" in tr[-1] and tr[-1]["experts_dropped"] == 0,
                  f"uneven {tr[-1].get('experts_uneven', 0):.2f}")
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


def speed(dev, T=2048, steps=8):
    """Full size, compiled, forward + back at 2,048 words, the student alone (no optimizer, no tests): Dense-30M, and
    MoE-150M with the grouped product and with the queues, at micro 64 and 128, with and without CUDA graphs."""
    from bee6.model import Bee6, Bee6Config
    from bee6.moe import Bee6MoE, Bee6MoEConfig
    from oazy.train import MOE150, DENSE30
    rows = [("Dense-30M", Bee6, Bee6Config, DENSE30, 64, 0), ("MoE-150M queues", Bee6MoE, Bee6MoEConfig,
            {**MOE150, "expert_kernel": "queues"}, 64, 0), ("MoE-150M grouped", Bee6MoE, Bee6MoEConfig, MOE150, 64, 0),
            ("MoE-150M grouped", Bee6MoE, Bee6MoEConfig, MOE150, 128, 0), ("MoE-150M grouped + graphs", Bee6MoE, Bee6MoEConfig, MOE150, 64, 1)]
    best = {}
    for name, cls, Cfg, kw, micro, graphs in rows:
        try:
            torch.manual_seed(0)
            m = cls(Cfg(**{**kw, "ctx": f"{T}:0", "graphs": graphs})).to(dev).train()
            m.startup_check(dev, lambda s_: term.info(s_))
            m.set_progress(0.5)
            m.enable_compile()
            x = torch.randint(1, 8192, (micro, T), device=dev)
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
            r = micro * T * steps / (time.time() - t0)
            mem = torch.cuda.max_memory_allocated() / 1e9
            term.info(f"{name:28s} micro {micro:3d}: {r / 1e3:5.0f}k tokens/s, peak {mem:.0f} GB; "
                      f"100B tokens ~{100e9 / r / 3600 * 1.12:.0f} h (+~12% optimizer and tests)")
            key = name.split(" ")[0]
            if r > best.get(key, (0, ""))[0]:
                best[key] = (r, f"{name}, micro {micro}")
            del m, x
        except Exception as e:
            term.info(f"{name:28s} micro {micro:3d}: failed ({type(e).__name__}: {str(e).splitlines()[0][:100] if str(e) else ''})")
        torch.cuda.empty_cache()
    for k, (r, how) in best.items():
        term.info(f"fastest {k}: {how}, {r / 1e3:.0f}k tokens/s")
    check("speed at full size measured", "MoE-150M" in best)


if __name__ == "__main__":
    main()
