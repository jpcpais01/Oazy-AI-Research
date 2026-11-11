"""Self-tests for Lucid-300M (lucid/model.py): its size against Gnosis, the short sight is exactly 64 pieces, honest
next-word reading (nothing looks ahead, any length reads the same), the picture carries what the word layers can no
longer see, a few steps teach the picture to foresee, more passes at reading time run, the grids go to NorMuon, and a
whole tiny run through the trainer (the mind-on/off and passes checks logged, a checkpoint that loads again).

    python3 -m lucid.tests        (CPU, a few minutes; without internet it uses stand-in tokenizers)
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from oazy import term


def check(name, ok, detail=""):
    term.check(name, ok, detail)


def small(**kw):
    from lucid.model import Lucid, LucidConfig
    cfg = dict(fp8=0, d_model=64, n_layer=6, ear_layers=2, read_at="3,5", attn_heads=2, attn_kv_heads=1, attn_head_dim=32,
               mlp_hidden=128, law_hidden=128, max_len=256, ctx="128:0", vocab_size=300, key_offset="3", window=16, beat=8,
               slots=4)
    cfg.update(kw)
    return Lucid(LucidConfig(**cfg))


def wake(m, std=0.05):
    """Every output starts at zero (each part a no-op); give them small random weights so every path is tested."""
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() >= 2 and float(p.abs().sum()) == 0:
                p.normal_(0, std)


def main():
    term.tests_start("Lucid-300M", "cpu")
    from lucid.model import Lucid, LucidConfig, local
    from bee6.model import Bee6, Bee6Config
    from oazy.train import GNOSIS

    # ---- the size
    ng = sum(p.numel() for p in Bee6(Bee6Config(fp8=0, **GNOSIS)).parameters())
    big = Lucid(LucidConfig(fp8=0))
    nn_ = big.num_params()
    check("Lucid-300M is about Gnosis's size (within 5%)", abs(nn_ - ng) / ng < 0.05,
          f"{nn_ / 1e6:.1f}M (the mind and its reads {big.mind_params() / 1e6:.1f}M) vs Gnosis {ng / 1e6:.1f}M")
    del big

    # ---- short sight: exactly the last W pieces
    torch.manual_seed(0)
    B, T, H, KV, hd, W = 2, 96, 4, 2, 8, 32
    q, k, v = torch.randn(B, T, H, hd), torch.randn(B, T, KV, hd), torch.randn(B, T, KV, hd)
    i = torch.arange(T)
    m_ = (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < W)
    ref = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2).repeat_interleave(2, 1),
                                         v.transpose(1, 2).repeat_interleave(2, 1), attn_mask=m_).transpose(1, 2)
    err = float((local(q, k, v, W) - ref).abs().max())
    check("the word layers see exactly the last W pieces (block-local attention = the full masked one)", err < 1e-5, f"{err:.1e}")

    # ---- honest reading
    torch.manual_seed(0)
    m = small()
    wake(m)
    m.eval()
    x = torch.randint(1, 300, (2, 100))
    with torch.no_grad():
        a = m.scores(m._trunk(x))
        x2 = x.clone()
        x2[:, 60] = 7
        b = m.scores(m._trunk(x2))
        before, after = float((a[:, :60] - b[:, :60]).abs().max()), float((a[:, 60:] - b[:, 60:]).abs().max())
        check("changing piece 60 changes nothing before it, and something after", before == 0.0 and after > 0,
              f"before {before:.1e}, after {after:.3f}")
        p = m.scores(m._trunk(x[:, :70]))
        gap = float((p - a[:, :70]).abs().max())
        check("any length reads the same (70 pieces = the first 70 of 100)", gap < 1e-5, f"{gap:.1e}")

        # ---- the picture carries what the word layers no longer see (one layer of sight: 16 back)
        m1 = small(n_layer=2, ear_layers=1, read_at="1", key_offset="")
        wake(m1)
        m1.eval()
        x3 = x.clone()
        x3[:, 5] = 9
        far = float((m1.scores(m1._trunk(x)) - m1.scores(m1._trunk(x3)))[:, 40:].abs().max())
        m1.mind_on = False
        blind = float((m1.scores(m1._trunk(x)) - m1.scores(m1._trunk(x3)))[:, 40:].abs().max())
        m1.mind_on = True
        check("a piece 35+ back reaches the next word only through the picture", far > 1e-4 and blind == 0.0,
              f"with the mind {far:.2e}, without {blind:.1e}")

    # ---- learning: the next word, and the picture foresees better
    torch.manual_seed(1)
    m = small()
    m.train()
    opt = m.make_optimizer(3e-3, torch.device("cpu"))
    pat = torch.randint(1, 300, (1, 24))                   # a text that repeats: the picture can learn to foresee it
    xs = torch.cat([pat] * 6, 1)[:, :129].repeat(4, 1)
    first = None
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        loss, st = m.losses(xs[:, :-1], xs[:, 1:])
        loss.backward()
        opt.step()
        if first is None:
            first = (float(st["ce"]), float(st["mind_foresee"]))
    check("a few steps teach the next word and the picture's foresight", float(st["ce"]) < first[0] - 0.5
          and float(st["mind_foresee"]) < first[1], f"loss {first[0]:.2f} -> {float(st['ce']):.2f}, foresight distance "
          f"{first[1]:.3f} -> {float(st['mind_foresee']):.3f}, settled {float(st['mind_settled']):.3f}")
    muon = {id(p) for g in opt.opts[0].param_groups for p in g["params"]} if len(opt.opts) > 1 else set()
    names = {id(p): n for n, p in m.named_parameters()}
    mine = [names[i] for i in muon if names[i].startswith(("mind.", "reads."))]
    check("the mind's and the reads' grids are on NorMuon, the starting picture is not",
          any("law" in n for n in mine) and any("reads" in n for n in mine) and "mind.s0" not in mine, f"{len(mine)} grids")
    r = m.loop_report(xs[:2].numpy(), torch.device("cpu"), batch=2)
    check("reading with 1, 2, 3 and 6 passes per beat runs", sorted(r["pass_loss"]) == [1, 2, 3, 6],
          " ".join(f"{k}: {v:.3f}" for k, v in r["pass_loss"].items()) + f" | still moves {r['move_last']:.1%}")
    r = m.core_report(xs[:2].numpy(), torch.device("cpu"), batch=2)
    check("the mind-off check runs (and the picture is worth something on a repeating text)", r["core_gap"] > 0,
          f"on {r['on']:.3f}, off {r['core_off']:.3f}")
    ids, _ = m.generate([1, 2, 3], max_new_tokens=5)
    check("it writes", len(ids) == 5)

    # ---- a tiny run through the trainer
    from bee2.tests import stand_ins, fake_docs
    from oazy import tok
    tmp_ctx = tempfile.TemporaryDirectory()
    tmp = Path(tmp_ctx.name)
    used = stand_ins(tmp)
    if used:
        term.info(f"no internet: stand-in {' and '.join(used)} tokenizer(s)")
    enc = tok.get("smollm2")
    data = tmp / "data"
    rng = random.Random(0)
    sado = data / "sado"
    sado.mkdir(parents=True)
    for split, k_ in (("val", 300), ("train", 1500)):
        np.fromiter((t for d_ in fake_docs(k_, rng) for t in enc.encode_ordinary(d_) + [enc.eot_token]),
                    dtype=np.uint16).tofile(sado / f"{split}.bin")
    (data / "hellaswag_val.jsonl").write_text("".join(
        json.dumps({"ctx": " ".join(fake_docs(1, rng)), "endings": fake_docs(4, rng), "label": rng.randint(0, 3)}) + "\n"
        for _ in range(24)))
    from oazy import train
    runs = tmp / "runs"
    sys.argv = ["train", "--model", "lucid", "--preset", "smoke", "--data", str(data), "--runs", str(runs), "--mentor", "none",
                "--device", "cpu", "--compile", "0", "--tokens", "40000", "--batch-ramp", "512:0.5", "--micro", "2",
                "--set", "ctx=64:0,128:0.3", "attn_head_dim=16", "attn_heads=4", "attn_kv_heads=2", "fp8=0",
                "--name", "ns", "--data-sub", "sado"]
    train.main()
    rd = runs / "ns"
    stw = json.loads((rd / "status.json").read_text())
    recs = [json.loads(l_) for l_ in (rd / "metrics.jsonl").read_text().splitlines()]
    core = [r for r in recs if r["kind"] == "core"]
    loop = [r for r in recs if r["kind"] == "loop"]
    mind = [r for r in recs if r["kind"] == "train" and r.get("mind_foresee") is not None]
    check("a tiny Lucid run finishes; the mind-on/off check, the passes check and the mind's numbers are logged",
          stw["state"] == "done" and core and loop and mind,
          f"mind gap {core[-1]['core_gap']:+.4f}" if core else "")
    model, _ = train.load_model(rd / "ckpt.pt", torch.device("cpu"))
    check("its checkpoint loads again as Lucid", type(model).__name__ == "Lucid")
    tmp_ctx.cleanup()
    term.tests_passed("Lucid-300M")


if __name__ == "__main__":
    main()
