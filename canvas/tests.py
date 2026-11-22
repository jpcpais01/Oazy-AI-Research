"""Self-tests for Canvas-135 (canvas/model.py): its size against Bee-135, honest next-word reading (nothing looks ahead,
any length reads the same), the canvas carries what the word layers can no longer see, the canvas's attention grid,
SIGReg tells a collapse from a Gaussian, a few steps teach the words and the canvas, more refining steps at reading time
run, the grids go to NorMuon, and a whole tiny run through the trainer (the canvas-off and steps checks logged, a
checkpoint that loads again).

    python3 -m canvas.tests                 (CPU, a few minutes; without internet it uses stand-in tokenizers)
    python3 -m canvas.tests --device cuda   (also: one race-size step, compiled, and its speed against Bee-135)
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


def small(**kw):
    from canvas.model import CanvasModel, CanvasConfig
    cfg = dict(fp8=0, d_model=64, n_layer=6, ear_layers=2, read_at="3,5", attn_heads=2, attn_kv_heads=1, attn_head_dim=32,
               mlp_hidden=128, max_len=256, ctx="128:0", vocab_size=300, key_offset="3", window=16, beat=8, slots=3,
               steps=3, z_dim=16, mind_dim=64, mind_heads=2, mind_layers=1, mind_hidden=128, canvas_layers=1, sig_dirs=16)
    cfg.update(kw)
    return CanvasModel(CanvasConfig(**cfg))


def wake(m, std=0.05):
    """Every output starts at zero (each part a no-op); give them small random weights so every path is tested."""
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() >= 2 and float(p.abs().sum()) == 0:
                p.normal_(0, std)


def gpu_checks(device):
    from canvas.model import CanvasModel, CanvasConfig
    from gnosis2.model import Gnosis2, Gnosis2Config
    from gnosis2.race135 import B135
    from oazy.train import GNOSIS
    T, B = 2048, 8
    x = torch.randint(0, 49152, (B, T + 1), device=device)
    res = {}
    for name, m in (("Canvas-135", CanvasModel(CanvasConfig())),
                    ("Canvas-2", CanvasModel(CanvasConfig(window=0, lean=1, attn_backend="cudnn"))),
                    ("Bee-135", Gnosis2(Gnosis2Config(**{**GNOSIS, **B135, "lean": 1, "attn_backend": "cudnn"})))):
        m.to(device).train()
        m.enable_compile()
        opt = m.make_optimizer(1e-3, device)
        times = []
        for i in range(8):
            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, st = m.losses(x[:, :-1], x[:, 1:])
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if i >= 3:
                times.append(time.time() - t0)
        res[name] = B * T / float(np.median(times))
        check(f"{name}: a race-size step runs compiled on the GPU", bool(torch.isfinite(loss)), f"{res[name] / 1e3:,.0f}k tok/s, loss {float(loss):.2f}")
        del m, opt
        torch.cuda.empty_cache()
    term.info(f"Canvas-135 runs at {res['Canvas-135'] / res['Bee-135']:.2f}x, Canvas-2 (full sight, lean, cuDNN) at "
              f"{res['Canvas-2'] / res['Bee-135']:.2f}x Bee-135 lean + cuDNN's speed (8 x 2,048, before tuning)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    device = torch.device(a.device)
    term.tests_start("Canvas-135", a.device)
    from canvas.model import CanvasModel, CanvasConfig, sigreg, sigmas
    from gnosis2.model import Gnosis2, Gnosis2Config
    from gnosis2.race135 import B135
    from oazy.train import GNOSIS

    # ---- the size
    nb = Gnosis2(Gnosis2Config(**{**GNOSIS, **B135})).num_params()
    big = CanvasModel(CanvasConfig())
    nc = big.num_params()
    check("Canvas-135 is about Bee-135's size (within 3%)", abs(nc - nb) / nb < 0.03,
          f"{nc / 1e6:.1f}M (the mind, the canvas and their reads {big.mind_params() / 1e6:.1f}M) vs Bee-135 {nb / 1e6:.1f}M")
    del big

    # ---- the noise schedule, the canvas's grid
    s = sigmas(4, 0.15)
    check("the refining steps go from fog (noise 1) down to 0.15", abs(s[0] - 1) < 1e-9 and abs(s[-1] - 0.15) < 1e-9
          and all(a_ > b_ for a_, b_ in zip(s, s[1:])), " ".join(f"{v:.2f}" for v in s))
    m = small()
    g = m.canvas.mask(5, torch.device("cpu"))       # 5 boundaries x 3 slots look at 5 mind positions + 15 slots
    ok = g.shape == (15, 20) and bool(g[0, 0]) and not bool(g[0, 1]) and bool(g[14, :5].all()) \
        and bool(g[3, 5 + 3:5 + 6].all()) and not bool(g[3, 5:5 + 3].any()) and not bool(g[3, 5 + 6:].any())
    check("a slot sees the mind up to its own boundary and every slot of its own canvas, nothing else", ok)

    # ---- SIGReg: a collapse costs a lot, a standard normal almost nothing
    torch.manual_seed(0)
    gauss, flat = float(sigreg(torch.randn(4096, 32), 64)), float(sigreg(torch.zeros(4096, 32), 64))
    check("SIGReg is near 0 on a standard normal and large on a collapse", gauss < 0.01 and flat > 0.2,
          f"normal {gauss:.4f}, collapse {flat:.3f}")

    # ---- honest reading: changing a piece changes nothing before it, at any length
    torch.manual_seed(0)
    m = small()
    wake(m)
    m.eval()
    x = torch.randint(1, 300, (2, 100))
    base = m.scores(m._trunk(x))
    worst = 0.0
    for p in (0, 7, 8, 9, 37, 64, 99):
        x2 = x.clone()
        x2[:, p] = (x2[:, p] + 1) % 300
        d = (m.scores(m._trunk(x2)) - base)[:, :p].abs().max() if p else torch.tensor(0.0)
        worst = max(worst, float(d.detach()))
    check("honest: changing any piece changes nothing before it (the canvas sees only finished chunks)", worst == 0.0,
          f"largest change before it {worst:.1e}")
    a1 = m.scores(m._trunk(x[:, :53]))
    check("any length reads the same as the start of a longer text", float((a1 - base[:, :53]).abs().max().detach()) < 1e-4)

    # ---- Canvas-2: the word layers see the whole text (window=0); still honest, and lean gives the same numbers
    torch.manual_seed(0)
    m = small(window=0)
    wake(m)
    m.eval()
    base = m.scores(m._trunk(x))
    worst = 0.0
    for p in (0, 7, 8, 9, 37, 64, 99):
        x2 = x.clone()
        x2[:, p] = (x2[:, p] + 1) % 300
        d = (m.scores(m._trunk(x2)) - base)[:, :p].abs().max() if p else torch.tensor(0.0)
        worst = max(worst, float(d.detach()))
    a1 = m.scores(m._trunk(x[:, :53]))
    check("Canvas-2 (full sight): honest, and any length reads the same", worst == 0.0
          and float((a1 - base[:, :53]).abs().max().detach()) < 1e-4, f"largest change before it {worst:.1e}")
    far = x.clone()
    far[:, 3] = (far[:, 3] + 1) % 300
    m.mind_on = False                                 # with the canvas off, only the word layers can carry piece 3
    off0, off1 = m.scores(m._trunk(x)), m.scores(m._trunk(far))
    m.mind_on = True
    check("Canvas-2's word layers themselves see far back (piece 3 changes the guess at piece 99, canvas off)",
          float((off1 - off0)[:, 99].abs().max().detach()) > 0)
    m.cfg.lean = 1
    try:
        lean = m.scores(m._trunk(x))
        gap = float((lean - base).abs().max().detach())
        check("Canvas-2 lean: the same numbers as the plain layers", gap < 2e-3 * float(base.abs().max().detach()) + 1e-3,
              f"largest gap {gap:.1e}")
    except Exception as e:                            # lean's kernels may need a GPU: the run checks them at its start
        term.info(f"lean not checkable here ({type(e).__name__}); a GPU run checks it against the plain layers at its start")
    m.cfg.lean = 0

    # ---- the canvas is the only way past the window (one ear layer and one mouth layer: 16 pieces of sight in all)
    m = small(n_layer=2, ear_layers=1, read_at="1", key_offset="")
    wake(m)
    m.eval()
    x3 = x.clone()
    x3[:, 5] = (x3[:, 5] + 1) % 300
    far = float((m.scores(m._trunk(x)) - m.scores(m._trunk(x3)))[:, 40:].abs().max())
    m.mind_on = False
    blind = float((m.scores(m._trunk(x)) - m.scores(m._trunk(x3)))[:, 40:].abs().max())
    m.mind_on = True
    check("a piece 35+ back reaches the next word only through the canvas", far > 1e-4 and blind == 0.0,
          f"with the canvas {far:.2e}, without {blind:.1e}")

    # ---- learning: the words, the diffusion lesson, the foresight
    torch.manual_seed(1)
    m = small()
    m.train()
    opt = m.make_optimizer(3e-3, torch.device("cpu"))
    pat = torch.randint(1, 300, (1, 40))                   # a text that repeats past the window: only the canvas can carry it
    xs = torch.cat([pat] * 6, 1)[:, :129].repeat(4, 1)
    first = None
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        loss, st = m.losses(xs[:, :-1], xs[:, 1:])
        loss.backward()
        opt.step()
        if first is None:
            first = {k: float(st[k]) for k in ("ce", "canvas_diffuse")}
    check("a few steps teach the next word and the denoising", float(st["ce"]) < first["ce"] - 0.5
          and float(st["canvas_diffuse"]) < first["canvas_diffuse"],
          f"loss {first['ce']:.2f} -> {float(st['ce']):.2f}, denoising {first['canvas_diffuse']:.3f} -> "
          f"{float(st['canvas_diffuse']):.3f}, foresight {float(st['canvas_honest']):.3f}, sure {float(st['canvas_sure']):.2f}, "
          f"SIGReg {float(st['canvas_sig']):.3f}")
    muon = {id(p) for g_ in opt.opts[0].param_groups for p in g_["params"]} if len(opt.opts) > 1 else set()
    names = {id(p): n for n, p in m.named_parameters()}
    mine = [names[i] for i in muon if not names[i].startswith("blocks.")]
    check("the mind's, the canvas's and the reads' grids are on NorMuon; the tables and the error head are not",
          any(n.startswith("mind.") for n in mine) and any(n.startswith("canvas.layers") for n in mine)
          and any(n.startswith("reads.") for n in mine) and "canvas.slot_pos" not in mine and "canvas.err.weight" not in mine,
          f"{len(mine)} grids")
    r = m.loop_report(xs[:2].numpy(), torch.device("cpu"), batch=2)
    check("reading with 1, 2, 3 and 6 refining steps runs", sorted(r["pass_loss"]) == [1, 2, 3, 6],
          " ".join(f"{k}: {v:.3f}" for k, v in r["pass_loss"].items()) + f" | still moves {r['move_last']:.1%}")
    r = m.core_report(xs[:2].numpy(), torch.device("cpu"), batch=2)
    check("the canvas-off check runs (and the canvas is worth something on a text that repeats past the window)",
          r["core_gap"] > 0, f"on {r['on']:.3f}, off {r['core_off']:.3f}")
    ids, _ = m.generate([1, 2, 3], max_new_tokens=5)
    check("it writes", len(ids) == 5)

    if device.type == "cuda":
        gpu_checks(device)

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
    sys.argv = ["train", "--model", "canvas", "--preset", "smoke", "--data", str(data), "--runs", str(runs), "--mentor", "none",
                "--device", "cpu", "--compile", "0", "--tokens", "40000", "--batch-ramp", "512:0.5", "--micro", "2",
                "--set", "ctx=64:0,128:0.3", "attn_head_dim=16", "attn_heads=4", "attn_kv_heads=2", "fp8=0",
                "--name", "cv", "--data-sub", "sado"]
    train.main()
    rd = runs / "cv"
    stw = json.loads((rd / "status.json").read_text())
    recs = [json.loads(l_) for l_ in (rd / "metrics.jsonl").read_text().splitlines()]
    core = [r for r in recs if r["kind"] == "core"]
    loop = [r for r in recs if r["kind"] == "loop"]
    cv = [r for r in recs if r["kind"] == "train" and r.get("canvas_diffuse") is not None]
    check("a tiny Canvas run finishes; the canvas-off check, the steps check and the canvas's numbers are logged",
          stw["state"] == "done" and core and loop and cv, f"canvas gap {core[-1]['core_gap']:+.4f}" if core else "")
    model, _ = train.load_model(rd / "ckpt.pt", torch.device("cpu"))
    check("its checkpoint loads again as Canvas", type(model).__name__ == "CanvasModel")
    tmp_ctx.cleanup()
    term.tests_passed("Canvas-135")


if __name__ == "__main__":
    main()
