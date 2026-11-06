"""Self-tests for Bee0.2B: size and layout, no peeking at future words, chat-style reading gives the same
answers as reading all at once, the optimizer split, the text mix, the schedule and the mentor.

    python -m bee.tests                 (CPU, ~1 min)
    python -m bee.tests --device cuda   (also checks the fast memory kernel and bf16)
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch

from .model import Bee, BeeConfig, ctx_at
from memtrix import memory
from oazy.mix import Mix, parse_mix, DEFAULT_MIX
from oazy.train import make_optimizer, lr_wsd, MUON_GRIDS
from oazy import term


def check(name, ok, detail=""):
    term.check(name, ok, detail)          # ✓ / ✗ in the lab's style (oazy/term.py)


SMALL = dict(vocab_size=320, d_model=64, n_layer=8, n_head=2, head_k=16, head_v=16, attn_heads=4, attn_kv_heads=2,
             attn_head_dim=16, mlp_hidden=96, max_len=128, ctx="64:0,128:0.5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    dev = torch.device(ap.parse_args().device)
    term.tests_start("Bee0.2B", dev)
    torch.manual_seed(0)

    # ---- the real size
    big = Bee(BeeConfig())
    n = big.num_params()
    check("full size is about 0.2B", 195e6 < n < 215e6, f"{n / 1e6:.1f}M parameters, {big.layout()}")
    check("3 memory layers to 1 attention layer", big.cfg.kinds()[:8] == list("MMMAMMMA") and big.cfg.kinds().count("A") == 6)
    check("no bias terms in the big grids",
          all(m.bias is None for nm, m in big.named_modules() if isinstance(m, torch.nn.Linear) and not nm.endswith(("trust", "fade"))))
    del big

    # ---- no peeking, and reading as a stream = reading at once
    for rope in (0, 1):
        m = Bee(BeeConfig(**SMALL, rope=rope)).to(dev).eval()
        x = torch.randint(0, 320, (2, 100), device=dev)
        with torch.no_grad():
            a = m._body(x)
            x2 = x.clone()
            x2[:, 60:] = torch.randint(0, 320, (2, 40), device=dev)
            b = m._body(x2)
            peek = (a[:, :60] - b[:, :60]).abs().max().item()
            check(f"no peeking at later words (rope={rope})", peek < 1e-4, f"max change {peek:.2e}")
            st, outs = None, []
            for s, e in [(0, 37), (37, 38), (38, 39), (39, 70), (70, 100)]:
                lo, st = m.read(x[:, s:e], st)
                outs.append(lo)
            diff = (torch.cat(outs, 1) - a).abs().max().item()
            check(f"word-by-word reading = reading at once (rope={rope})", diff < 2e-3, f"max diff {diff:.2e}")
            # the lid-off view gives the same answer too
            st2, lo = None, None
            for t in range(20):
                lo, st2, infos = m.inspect(x[:1, t:t + 1], st2)
            diff = (lo[0, -1] - a[0, 19]).abs().max().item()
            check(f"lid-off reading = reading at once (rope={rope})", diff < 2e-3, f"max diff {diff:.2e}")
            att = [i for i in infos if i["kind"] == "A"][0]["looks"]
            check("attention looks back over every word so far", att.shape == (20,) and abs(float(att.sum()) - 1) < 1e-4)

    # ---- the optimizer split
    m = Bee(BeeConfig(**SMALL)).to(dev)
    opt = make_optimizer(m, 1e-3, dev, "muon")
    muon_ids = {id(p) for p in opt.opts[0].param_groups[0]["params"]}
    grids = [n for n, p in m.named_parameters() if p.dim() == 2 and n.startswith("blocks") and "conv" not in n
             and not n.endswith(("trust.weight", "fade.weight"))]
    missing = [n for n, p in m.named_parameters() if n in grids and id(p) not in muon_ids]
    check("Muon moves every big grid inside the blocks", not missing, str(missing[:3]))
    check("the word table stays on AdamW, without weight decay",
          id(m.embed.weight) not in muon_ids and any(id(m.embed.weight) in {id(q) for q in g["params"]} and g["weight_decay"] == 0
                                                  for g in opt.opts[1].param_groups))
    check("Muon names cover attention", any(s.startswith("attn.") for s in MUON_GRIDS))

    # ---- it learns (a repeating pattern, a few dozen steps)
    torch.manual_seed(1)
    pat = torch.randint(0, 320, (40,), device=dev)
    data = pat.repeat(8)[None].repeat(8, 1)
    first = last = None
    for i in range(40):
        _, loss, _ = m(data[:, :-1], data[:, 1:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        first = first if first is not None else loss.item()
        last = loss.item()
    check("it learns (Muon + AdamW)", last < first * 0.5, f"loss {first:.2f} -> {last:.2f}")

    # ---- context plan and schedule
    cfg = BeeConfig()
    check("1,024-word windows first, 2,048 from halfway", ctx_at(cfg, 0) == 1024 and ctx_at(cfg, 0.49) == 1024 and ctx_at(cfg, 0.5) == 2048)
    check("WSD: warm-up, flat, straight fall to zero",
          lr_wsd(0.0, 0, 1.0, 10) == 0.1 and lr_wsd(0.5, 100, 1.0, 10) == 1.0 and abs(lr_wsd(0.9, 900, 1.0, 10) - 0.5) < 1e-9
          and lr_wsd(1.0, 1000, 1.0, 10) == 0.0)

    # ---- the text mix
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "dclm").mkdir()
        (d / "chat").mkdir()
        rng = np.random.default_rng(0)
        # each source's words come from its own range, so we can tell them apart
        (rng.integers(0, 1000, 200_000)).astype(np.uint16).tofile(d / "train.bin")
        (rng.integers(1000, 2000, 200_000)).astype(np.uint16).tofile(d / "dclm" / "train.bin")
        (rng.integers(2000, 3000, 50_000)).astype(np.uint16).tofile(d / "chat" / "train.bin")
        plan = parse_mix(DEFAULT_MIX)
        check("the default mix is 65/35, then 45/25/30 from 80%", plan[0][1] == {"fwe": .65, "dclm": .35} and plan[1][0] == 0.8)
        mx = Mix(d, DEFAULT_MIX, 64, 64, lengths=(64, 128))
        check("slots fit two short or one long window", mx.slot == 130)

        def share(x):
            s = x[:, 0]
            return [float(((s >= a) & (s < a + 1000)).float().mean()) for a in (0, 1000, 2000)]
        mx.progress = 0.1
        sh = np.mean([share(mx.get(torch.device("cpu"))[0]) for _ in range(30)], 0)
        check("early: ~65% FineWeb-Edu, ~35% DCLM, no chat", abs(sh[0] - .65) < .05 and sh[2] == 0, str(np.round(sh, 3)))
        mx.progress = 0.9
        sh = np.mean([share(mx.get(torch.device("cpu"))[0]) for _ in range(30)], 0)
        check("late: ~30% chat", abs(sh[2] - .30) < .05, str(np.round(sh, 3)))
        # resume gives exactly the same batches
        st = mx.state()
        a = mx.get(torch.device("cpu"))[0]
        mx2 = Mix(d, DEFAULT_MIX, 64, 64, lengths=(64, 128))
        mx2.set_state(st)
        mx2.progress = 0.9
        b = mx2.get(torch.device("cpu"))[0]
        check("resuming repeats nothing and skips nothing", torch.equal(a, b))
        # switching to longer windows never re-reads a slot
        m3 = Mix(d, "fwe:1@0", 64, 1, lengths=(64, 128))
        starts = []
        for i in range(50):
            if i == 25:
                m3.set_block(128)
            x = m3.get(torch.device("cpu"))[0]
            starts.append(tuple(x[0, :8].tolist()))
        check("longer windows start on fresh text", len(set(starts)) == len(starts) and m3.T == 128)
        # a missing source is skipped and the rest rescaled
        (d / "chat" / "train.bin").unlink()
        m5 = Mix(d, "fwe:0.5,chat:0.5@0", 64, 8, lengths=(64,), log=lambda *a: None)
        check("a missing source is skipped", list(m5.weights(0)) == ["fwe"])

    # ---- the mentor reads long windows in pieces
    from ember.mentor import Mentor
    me = Mentor("random", dev, compile=False)
    g = me.guess(torch.randint(0, 50257, (1, 1500), device=dev))
    check("the mentor reads 1,500 words in two pieces", g.shape == (1, 1500, 50257))

    # ---- GPU: fast kernel and bf16
    if dev.type == "cuda":
        ok = memory.fla_check(dev)
        term.info(f"fast memory kernel: {'yes' if ok else 'no (pure PyTorch, slower)'}")
        m = Bee(BeeConfig(**{**SMALL, "head_k": 64, "head_v": 64})).to(dev)
        x = torch.randint(0, 320, (4, 128), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss, _ = m(x, x)
        loss.backward()
        check("bf16 training step on the GPU", torch.isfinite(loss).item())
    term.tests_passed("Bee0.2B")


if __name__ == "__main__":
    main()
