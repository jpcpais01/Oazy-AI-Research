"""Self-tests for Logos1-200M (logos/model.py): the size against Bee6, honest next-word reading (nothing looks ahead,
the things of a stretch come only from before it, any length reads the same), the core changes what it says, the
marks' answer key reaches the linker, the grids go to NorMuon, Emerald's marks line up with its pieces, and a whole
tiny run on Emerald through the trainer (marks fed, the core-off check logged, the world's exams, a checkpoint that
loads again).

    python3 -m logos.tests        (CPU, a few minutes; without internet it uses stand-in tokenizers)
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from oazy import term


def check(name, ok, detail=""):
    term.check(name, ok, detail)


def small(**kw):
    from logos.model import Logos, LogosConfig
    cfg = LogosConfig(fp8=0, d_model=64, n_layer=6, attn_heads=2, attn_kv_heads=1, attn_head_dim=32, mlp_hidden=128,
                      max_len=256, ctx="128:0", d_core=64, core_heads=4, chunk=16, things=8, link_dim=16, vocab_size=300,
                      key_offset="3", **kw)
    return Logos(cfg)


def main():
    term.tests_start("Logos1-200M", "cpu")
    from logos.model import Logos, LogosConfig
    from bee6.model import Bee6, Bee6Config

    # ---- the size
    nb = sum(p.numel() for p in Bee6(Bee6Config(fp8=0)).parameters())
    nl = sum(p.numel() for p in Logos(LogosConfig(fp8=0)).parameters())
    check("Logos1-200M is Bee6's size (within 1%)", abs(nl - nb) / nb < 0.01, f"{nl / 1e6:.1f}M vs Bee6 {nb / 1e6:.1f}M")

    # ---- honest reading
    torch.manual_seed(0)
    m = small()
    with torch.no_grad():                         # wake everything up (outputs start at zero) and make words name things
        for r in m.reads.values():
            torch.nn.init.normal_(r.out.weight, std=0.05)
        for c in m.core:
            torch.nn.init.normal_(c.out.weight, std=0.05)
            torch.nn.init.normal_(c.down.weight, std=0.05)
        torch.nn.init.constant_(m.lk_ment.bias, 3.0)
    m.eval()
    x = torch.randint(1, 300, (2, 128))
    x[0, 50] = 0
    with torch.no_grad():
        a = m.scores(m._trunk(x))
        x2 = x.clone()
        x2[:, 70] = x2[:, 70] % 299 + 1
        b = m.scores(m._trunk(x2))
        check("a change at piece 70 changes nothing before it (next-word reading never looks ahead)",
              (a[:, :70] - b[:, :70]).abs().max().item() < 1e-5 and (a[:, 70:] - b[:, 70:]).abs().max().item() > 1e-4)
        x3 = x.clone()
        x3[:, 64:80] = x3[:, 64:80] % 299 + 1     # stretch 4 (pieces 64-79) changed: stretch 4's things are built before it
        S1, _, _ = m.think(m._run_to_link(x), m.docs(x))
        S2, _, _ = m.think(m._run_to_link(x3), m.docs(x3))
        check("the things a stretch reads come only from pieces before it began",
              (S1[:, :5] - S2[:, :5]).abs().max().item() < 1e-5 and (S1[:, 5:] - S2[:, 5:]).abs().max().item() > 1e-5)
        m.core_on = False
        c = m.scores(m._trunk(x))
        m.core_on = True
        check("the core changes what it says (switched off, the scores move)", (a - c).abs().max().item() > 1e-3)
        d = m.scores(m._trunk(x[:, :77]))
        check("any length reads the same (77 pieces, padded inside)", (d - a[:, :77]).abs().max().item() < 1e-4)
        lg, st = m.read(x[:1, :40])
        lg2, _ = m.read(x[:1, 40:42], st)
        check("reading on from a state gives the same scores", (lg2 - a[:1, 40:42]).abs().max().item() < 1e-4)

    # ---- learning
    m.train()
    y = torch.randint(1, 300, (2, 128))
    marks = torch.full((2, 128), 255, dtype=torch.int16)
    marks[1, [5, 20]] = 0
    marks[1, [9, 40]] = 1
    m.feed_marks(marks, torch.tensor([False, True]))
    loss, st = m.losses(x, y)
    loss.backward()
    named = dict(m.named_parameters())
    got = {n: named[n].grad is not None and named[n].grad.abs().sum().item() > 0
           for n in ("lk_q.weight", "lk_ment.weight", "core.0.qkv.weight", "core_in.weight", f"reads.{next(iter(m.reads))}.q.weight")}
    check("the marks' answer key reaches the linker, and the next word reaches the core and the read-back",
          "link" in st and all(got.values()), str(got))
    m.zero_grad()
    loss2, st2 = m.losses(x, y)
    check("without marks (ordinary text) there is no answer key", "link" not in st2)
    opt = m.make_optimizer(1e-3, torch.device("cpu"))
    grids = {id(p) for g in opt.opts[0].param_groups for p in g["params"]}
    check("the core's, the read-back's and the linker's grids learn with NorMuon, like the stream's",
          type(opt.opts[0]).__name__ == "NorMuon"
          and all(id(named[n]) in grids for n in ("core.0.qkv.weight", "core_out.weight", "lk_q.weight")))

    # ---- Logos1.1: the core's own job (guess the next stretch's things) and the focus member
    torch.manual_seed(1)
    m11 = small(predict=1.0, focus=1)
    with torch.no_grad():
        for r in m11.reads.values():
            torch.nn.init.normal_(r.out.weight, std=0.05)
        for c in m11.core:
            torch.nn.init.normal_(c.out.weight, std=0.05)
            torch.nn.init.normal_(c.down.weight, std=0.05)
        torch.nn.init.constant_(m11.lk_ment.bias, 3.0)
    m11.eval()
    with torch.no_grad():
        a = m11.scores(m11._trunk(x))
        x2 = x.clone()
        x2[:, 70] = x2[:, 70] % 299 + 1
        b = m11.scores(m11._trunk(x2))
        check("Logos1.1 (law + focus) still never looks ahead",
              (a[:, :70] - b[:, :70]).abs().max().item() < 1e-5 and (a[:, 70:] - b[:, 70:]).abs().max().item() > 1e-4)
    m11.train()
    loss, st = m11.losses(x, y)
    loss.backward()
    named = dict(m11.named_parameters())
    g = {n: named[n].grad is not None and named[n].grad.abs().sum().item() > 0
         for n in ("pred_out.weight", "core.0.qkv.weight", "focus_emb", "core_in.weight")}
    check("the law's lesson is there on plain text (no marks) and reaches the core, its guess and the focus",
          all(k in st for k in ("law_cos", "copy_cos")) and all(g.values()) and float(st["law_n"]) > 0,
          f"law {float(st['law_cos']):.3f} vs copy {float(st['copy_cos']):.3f}; {g}")
    torch.manual_seed(2)
    m12 = small(predict=1.0)
    with torch.no_grad():
        torch.nn.init.constant_(m12.lk_ment.bias, 3.0)
    opt = m12.make_optimizer(3e-3, torch.device("cpu"))
    first = None
    for _ in range(30):
        m12.zero_grad(set_to_none=True)
        l_, s_ = m12.losses(x, y)
        l_.backward()
        opt.step()
        first = first if first is not None else float(s_["law_cos"])
    check("a few steps teach the law to guess better", float(s_["law_cos"]) > first + 0.05,
          f"law cosine {first:.3f} -> {float(s_['law_cos']):.3f} (copying: {float(s_['copy_cos']):.3f})")
    r = m12.core_report(np.concatenate([x[:2].numpy(), y[:2, -1:].numpy()], 1), torch.device("cpu"), batch=2)
    check("the core-off check also reports the law against copying", "law_cos" in r and "copy_cos" in r)
    nb11 = sum(p.numel() for p in Logos(LogosConfig(fp8=0, predict=1.0, focus=1)).parameters())
    check("Logos1.1 is still Bee6's size (within 1%)", abs(nb11 - nb) / nb < 0.01, f"{nb11 / 1e6:.1f}M")

    # ---- Emerald's marks + a tiny run through the trainer
    from bee2.tests import stand_ins, fake_docs
    from oazy import emerald, tok, worldmix
    tmp_ctx = tempfile.TemporaryDirectory()
    tmp = Path(tmp_ctx.name)
    used = stand_ins(tmp)
    if used:
        term.info(f"no internet: stand-in {' and '.join(used)} tokenizer(s)")
    enc = tok.get("smollm2")
    a_ids = worldmix.job((0, 2, "smollm2"))
    b_ids, b_mk = worldmix.job_marks((0, 2, "smollm2"))
    from worldfirst import world as W
    e = W.make(W.FAMILIES[0], W.TRAIN_LEVELS[W.FAMILIES[0]][0], "train", 1_000_123)
    ids, mk = worldmix.encode_with_marks(enc, [e])
    named_words = [enc.decode([t]) for t, k in zip(ids[0], mk[0]) if k != 255]
    check("Emerald's marks: the very same pieces as before, one mark per piece, at the words that name things",
          np.array_equal(a_ids, b_ids) and b_mk.size == b_ids.size and 0 < (b_mk != 255).mean() < 0.5,
          f"{(b_mk != 255).mean():.1%} of pieces; e.g. {' '.join(named_words[:6])}")
    data = tmp / "data"
    rng = random.Random(0)
    sado = data / "sado"
    sado.mkdir(parents=True)
    for split, k in (("val", 300), ("train", 1500)):
        np.fromiter((t for d_ in fake_docs(k, rng) for t in enc.encode_ordinary(d_) + [enc.eot_token]),
                    dtype=np.uint16).tofile(sado / f"{split}.bin")
    (data / "hellaswag_val.jsonl").write_text("".join(
        json.dumps({"ctx": " ".join(fake_docs(1, rng)), "endings": fake_docs(4, rng), "label": rng.randint(0, 3)}) + "\n"
        for _ in range(24)))
    out = emerald.build(data, tokens=60_000, val_tokens=20_000, workers=1)
    (out / "train.marks").unlink()
    (out / "val.marks").unlink()
    meta = json.loads((out / "meta.json").read_text())
    meta.pop("marks")
    (out / "meta.json").write_text(json.dumps(meta))
    emerald.add_marks(data, workers=1)
    again = np.fromfile(out / "train.marks", dtype=np.uint8)
    check("an Emerald built before the marks gets them added, checked against its pieces",
          again.size == np.fromfile(out / "train.bin", dtype=np.uint16).size and json.loads((out / "meta.json").read_text())["marks"])
    from oazy import train
    runs = tmp / "runs"
    sys.argv = ["train", "--model", "logos", "--preset", "smoke", "--data", str(data), "--runs", str(runs), "--mentor", "none",
                "--device", "cpu", "--compile", "0", "--tokens", "40000", "--batch-ramp", "512:0.5", "--micro", "2",
                "--set", "ctx=64:0,128:0.3", "attn_head_dim=16", "attn_heads=4", "attn_kv_heads=2", "fp8=0",
                "--name", "lg", "--data-sub", "emerald"]
    train.main()
    rd = runs / "lg"
    stw = json.loads((rd / "status.json").read_text())
    recs = [json.loads(l_) for l_ in (rd / "metrics.jsonl").read_text().splitlines()]
    core = [r for r in recs if r["kind"] == "core"]
    links = [r for r in recs if r["kind"] == "train" and r.get("link") is not None]
    check("a tiny Logos1 run on Emerald finishes, the marks reach its linker, the core-off check and the world's exams are logged",
          stw["state"] == "done" and core and "world_core_gap" in core[-1] and links and (rd / "world_exam.json").exists(),
          f"core gap {core[-1]['core_gap']:+.4f}, world {core[-1].get('world_core_gap', 0):+.4f}" if core else "")
    model, _ = train.load_model(rd / "ckpt.pt", torch.device("cpu"))
    check("its checkpoint loads again as Logos1-200M", type(model).__name__ == "Logos")
    tmp_ctx.cleanup()
    term.tests_passed("Logos1-200M")


if __name__ == "__main__":
    main()
