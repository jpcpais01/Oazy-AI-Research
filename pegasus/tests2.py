"""Self-tests for Pegasus 2 Mini: its size against Plain-100M, fixed-then-learned thoughts, that no token can see a later
one, flex attention = the grids, that every part learns, the probes, the recipe, one whole tiny run; on a GPU also the
compiled step and a speed race against Plain-100M at full size.

    python -m pegasus.tests2                 (CPU, a few minutes)
    python -m pegasus.tests2 --device cuda   (on the pod: also the compiled step and the speed race, ~5 min)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from bee3.tests import check
from oazy import term

SMALL = dict(d_tok=64, tok_heads=2, tok_kv_heads=1, tok_mlp=128, enc_layers=1, spk_layers=2, window=16,
             d_think=64, think_layers=2, think_heads=2, think_kv_heads=1, think_mlp=128, head_dim=32,
             max_len=128, ctx="128:0", thought_len=6, max_thoughts=2, fixed_len=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-race", action="store_true", help="skip the GPU speed race")
    a = ap.parse_args()
    dev = torch.device(a.device)
    warnings.filterwarnings("ignore", message=".*flex_attention called without torch.compile.*")
    term.tests_start("Pegasus 2 Mini", dev)
    torch.manual_seed(0)
    from bee2.tests import stand_ins
    tmp_ctx = tempfile.TemporaryDirectory()
    tmp = Path(tmp_ctx.name)
    used_ = stand_ins(tmp)
    if used_:
        term.info(f"no internet: stand-in {' and '.join(used_)} tokenizer(s)")
    from pegasus.model2 import Pegasus2, Pegasus2Config, sentence_end_ids
    from pegasus.model import window_mask
    from pegasus.baseline import Plain100, Plain100Config

    # ---- sizes
    full = Pegasus2(Pegasus2Config())
    plain = Plain100(Plain100Config())
    n, npl = full.num_params(), plain.num_params()
    pp = full.part_params()
    check("about Plain-100M's size (within 5%), and cheaper per token", abs(n - npl) < 0.05 * npl and full.work_per_token() < npl,
          f"{n / 1e6:.1f}M vs Plain {npl / 1e6:.1f}M; ~{full.work_per_token() / 1e6:.0f}M at work per token; "
          + ", ".join(f"{k} {v / 1e6:.1f}M" for k, v in pp.items()))
    check("the word side is a real language model: Reader + Speaker are one stream of 9 layers of 640",
          full.cfg.enc_layers + full.cfg.spk_layers == 9 and full.cfg.d_tok == 640 and pp["speaker"] > pp["thinker"])
    del full, plain

    m = Pegasus2(Pegasus2Config(**SMALL)).to(dev)
    c = m.cfg
    ends = sentence_end_ids(c.tokenizer)
    check("sentence ends are known (fixed thoughts start after them)", len(ends) >= 2, f"{len(ends)} end tokens")

    # ---- the schedule
    m.set_progress(0.05)
    f1, w1 = m.fixed_now(), m.meaning_now()
    m.set_progress(0.15)
    f2, w2 = m.fixed_now(), m.meaning_now()
    m.set_progress(0.5)
    f3, w3 = m.fixed_now(), m.meaning_now()
    check("fixed thoughts until 10%, then learned; the meaning loss off until 10%, half at 15%, full from 20%",
          f1 and not f2 and not f3 and w1 == 0 and abs(w2 - c.meaning_weight / 2) < 1e-6 and w3 == c.meaning_weight)

    # ---- fixed thoughts
    from oazy import tok
    enc = tok.get("smollm2")
    text = "The cat sat on the mat. It was warm! Was it happy? Yes. " * 3
    ids = enc.encode_ordinary(text)[:60]
    x1 = torch.tensor([ids], device=dev)
    st = m._fixed_starts(x1)[0]
    after_end = [i for i in range(1, len(ids)) if ids[i - 1] in ends]
    check("fixed: a new thought after every sentence end (and every fixed_len tokens)",
          all(bool(st[i]) for i in after_end) and bool(st[0]) and len(after_end) > 3, f"{int(st.sum())} thoughts in {len(ids)} tokens")

    # ---- no peeking, in both phases
    T = 128
    x = torch.randint(1, 49152, (2, T), device=dev)
    x[:, 40] = 0
    m.eval()
    worst = 0.0
    for prog in (0.05, 0.5):
        m.set_progress(prog)
        with torch.no_grad():
            base = m._body(x).float()
            for s_ in (1, 17, 41, 64, 100, 127):
                y = x.clone()
                y[:, s_:] = torch.randint(1, 49152, (2, T - s_), device=dev)
                worst = max(worst, (m._body(y).float()[:, :s_] - base[:, :s_]).abs().max().item())
    check("no token sees a later one (fixed and learned thoughts, plan, pages, notebook)", worst < 1e-4, f"max change {worst:.1e}")

    # ---- flex = grids
    with torch.no_grad():
        a_ = m._core(x, None, False, False)[0].float()
        b_ = m._core(x, window_mask(T, c.window, dev), True, False)[0].float()
    e = (a_ - b_).abs().max().item()
    check("flex attention (window and 2-pages-per-thought notebook) gives the same outputs as the grids", e < 1e-4, f"max diff {e:.1e}")

    # ---- every part learns
    m.train()
    dead_all = []
    for prog, name in ((0.05, "fixed"), (0.5, "learned")):
        m.set_progress(prog)
        m.zero_grad(set_to_none=True)
        loss, stats = m.losses(x, x)
        loss.backward()
        parts = {"Reader": m.reader[0].attn.q.weight, "router": m.route_edge, "pages": m.pool_q, "Thinker": m.thinker[0].attn.q.weight,
                 "pieces": m.to_pieces.weight, "guesser": m.guess_fc2.weight, "Speaker": m.speaker[0].attn.q.weight,
                 "plan": m.plan.weight, "notebook": m.speaker[-1].note.k.weight, "word table": m.embed.weight}
        if name == "fixed":
            parts.pop("guesser")                     # (the meaning loss is still off while thoughts are fixed)
        dead_all += [f"{k} ({name})" for k, p_ in parts.items() if p_.grad is None or p_.grad.abs().sum() == 0]
        if name == "learned":
            rep = m.thought_report()
    check("one lesson reaches every part, with fixed and with learned thoughts", not dead_all,
          "no gradient: " + ", ".join(dead_all) if dead_all else
          f"{rep['thought_len']:.1f} tokens per thought, {rep.get('codes_used', 0):.0f} meanings in use, recon {rep['recon']:.3f}")
    m.zero_grad(set_to_none=True)

    # ---- probes
    win = torch.cat([x, x[:, :1]], 1).cpu().numpy()
    r = m.plan_report(win, dev, batch=2)
    check("the plan probe: loss with the plan, without it, without the notebook; meanings in use",
          all(k in r for k in ("on", "plan_off", "notebook_off", "plan_gap", "codes_used")) and r["plan_off"] != r["on"],
          f"on {r['on']:.3f}, plan off {r['plan_off']:.3f}, notebook off {r['notebook_off']:.3f}, {r['codes_used']:.0f} meanings")
    m.eval()
    m.set_progress(0.05)                             # (an untrained router makes 2-token thoughts that overflow the room)
    with torch.no_grad():
        lg = m._body(x).float()
        ref = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1))
        _, ce2, _ = m(x[:, :-1], x[:, 1:])
    check("the held-out loss is the next-word loss only", abs(ce2.item() - ref.item()) < 1e-3, f"{ce2.item():.4f} vs {ref.item():.4f}")
    out, _ = m.generate(enc.encode_ordinary("The cat"), max_new_tokens=4)
    check("it writes word by word", len(out) <= 4)

    # ---- the recipe
    from oazy.train import RECIPE, MENTOR, BEES, FAST, PRESETS
    pb = PRESETS["big"]
    check("Plain-100M's recipe and mentor exactly", RECIPE["pegasus2"] == RECIPE["plain100"] and MENTOR["pegasus2"] == MENTOR["plain100"]
          and "pegasus2" in BEES and "pegasus2" in FAST and pb["batch_pegasus2"] == pb["batch_plain100"]
          and pb["lr_pegasus2"] == pb["lr_plain100"] and pb["muon_lr_pegasus2"] == pb["muon_lr_plain100"])

    # ---- one whole tiny run on stand-in text
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
    mentor = "smol-random"
    try:
        import transformers  # noqa: F401
    except ImportError:
        mentor = "none"
    argv = ["train", "--model", "pegasus2", "--preset", "smoke", "--data", str(data), "--runs", str(runs), "--mentor", mentor,
            "--device", str(dev), "--compile", "0", "--tokens", "40000", "--batch-ramp", "512:0.5", "--mix", "fwe:0.9,chat:0.1@0",
            "--micro", "2", "--data-sub", "smol"]
    sys.argv = argv
    train.main()
    rd = runs / "pegasus2"
    recs = [json.loads(l_) for l_ in (rd / "metrics.jsonl").read_text().splitlines()]
    tr = [r_ for r_ in recs if r_["kind"] == "train"]
    stt = json.loads((rd / "status.json").read_text())
    check("a whole tiny Pegasus 2 Mini run (thoughts learned by the end, plan probe, HellaSwag, samples)",
          stt["state"] == "done" and any(r_["kind"] == "plan" for r_ in recs) and any(r_["kind"] == "hellaswag" for r_ in recs)
          and all("fixed" in r_ and "codes_used" in r_ for r_ in tr) and tr[-1]["fixed"] == 0,
          f"val loss {stt['latest']['val_loss']:.3f}, mentor {mentor}")
    mdl, ck = train.load_model(rd / "ckpt.pt", dev)
    check("its checkpoint loads as Pegasus2", ck["kind"] == "pegasus2" and type(mdl).__name__ == "Pegasus2")

    if dev.type == "cuda":
        mg = Pegasus2(Pegasus2Config(**{**SMALL, "max_len": 2048, "window": 256, "max_thoughts": 8, "thought_len": 16,
                                        "fixed_len": 32})).to(dev)
        xg = torch.randint(1, 49152, (4, 1024), device=dev)
        xg[:, 50] = 0
        mg.train()
        for prog in (0.05, 0.5):
            mg.set_progress(prog)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                le, _ = mg.losses(xg, xg)
            le.backward()
            mg.zero_grad(set_to_none=True)
            mg.enable_compile()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lc, _ = mg.losses(xg, xg)
            lc.backward()
            mg.zero_grad(set_to_none=True)
            mg._compiled = None
            check(f"the compiled step (flex attention) matches ({'fixed' if prog < 0.1 else 'learned'} thoughts)",
                  torch.isfinite(lc).item() and abs(lc.item() - le.item()) < 0.05, f"{lc.item():.4f} vs {le.item():.4f}")
        del mg
        if not a.no_race:
            xb = torch.randint(1, 49152, (8, 2048), device=dev)
            speed = {}
            for name, mk in (("Plain-100M", lambda: Plain100(Plain100Config())), ("Pegasus 2 Mini", lambda: Pegasus2(Pegasus2Config()))):
                torch._dynamo.reset()
                torch.manual_seed(0)
                mf = mk().to(dev).train()
                if hasattr(mf, "set_progress"):
                    mf.set_progress(0.5)
                mf.enable_compile()

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
                del mf
                torch.cuda.empty_cache()
            pr, pg = speed["Plain-100M"], speed["Pegasus 2 Mini"]
            check("speed race at full size (compiled bf16, 8 x 2,048, no mentor): Pegasus 2 Mini within 25% of Plain-100M",
                  pg > 0.75 * pr, f"Plain-100M {pr / 1e3:.0f}k, Pegasus 2 Mini {pg / 1e3:.0f}k tokens/s ({pg / pr - 1:+.0%})")
    tmp_ctx.cleanup()
    term.tests_passed("Pegasus 2 Mini")


if __name__ == "__main__":
    main()
