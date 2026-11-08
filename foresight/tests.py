"""Self-tests for Plain-100F (Plain-100M + the training-only foresight head): the same size and starting weights as
Plain-100M, where the head reads, what goes in the bags, that the held-out loss is the next-word loss only, that the
gradient reaches the whole trunk, that the head learns, the recipe, one whole tiny run; on a GPU also the compiled step
and the speed cost against Plain-100M at full size.

    python3 -m foresight.tests                 (CPU, a few minutes)
    python3 -m foresight.tests --device cuda   (on the pod: also the compiled step and the speed race, ~3 min)
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

from bee3.tests import check
from oazy import term

SMALL = dict(d_model=64, n_layer=2, attn_heads=4, attn_kv_heads=2, attn_head_dim=16, mlp_hidden=128, max_len=128,
             ctx="128:0", fore_hidden=96, fore_top=20, fore_max=512, fore_warmup=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-race", action="store_true", help="skip the GPU speed race")
    a = ap.parse_args()
    dev = torch.device(a.device)
    term.tests_start("Plain-100F", dev)
    from bee2.tests import stand_ins
    tmp_ctx = tempfile.TemporaryDirectory()
    tmp = Path(tmp_ctx.name)
    used_ = stand_ins(tmp)
    if used_:
        term.info(f"no internet: stand-in {' and '.join(used_)} tokenizer(s)")
    from foresight.model import Plain100F, Plain100FConfig, sentence_end_table, filler_table
    from pegasus.baseline import Plain100, Plain100Config
    from oazy import tok
    enc = tok.get("smollm2")

    # ---- sizes and starting weights
    torch.manual_seed(0)
    full = Plain100F(Plain100FConfig())
    torch.manual_seed(0)
    plain = Plain100(Plain100Config())
    sp, sf = plain.state_dict(), full.state_dict()
    same = all(torch.equal(sp[k], sf[k]) for k in sp)
    check("the same size as Plain-100M once the head is set aside (the head is training-only)",
          full.num_params() == plain.num_params(), f"{full.num_params() / 1e6:.1f}M, head {full.head_params() / 1e6:.2f}M")
    check("the same starting weights as Plain-100M (same seed), so a race differs only by the head",
          same and set(sp) <= set(sf), f"{len(sp)} tensors compared")
    del full, plain, sp, sf

    # ---- the tables
    V = 49152
    ends = sentence_end_table(enc, V)
    period = enc.encode_ordinary(".")
    check("sentence ends are known (. ! ? and line breaks)", ends.sum() >= 3 and (len(period) != 1 or ends[period[0]]),
          f"{int(ends.sum())} end pieces")
    torch.manual_seed(0)
    m = Plain100F(Plain100FConfig(**SMALL)).to(dev)
    text = ("The cat sat on the mat. It was warm! Was the garden happy? Yes. "
            "A very long sentence without any stop that keeps going and going on and on past sixteen pieces for sure here "
            "and there and everywhere around")
    ids = enc.encode_ordinary(text)
    counts = np.zeros(V, dtype=np.int64)
    np.add.at(counts, np.asarray(ids * 50, dtype=np.int64), 1)
    r = m.prepare_foresight(enc, counts)
    skip = m.fore_skip.cpu().numpy()
    common = np.argsort(-counts, kind="stable")[:5]
    check("filler is left out of the bags: punctuation, the document mark, the most common pieces",
          skip[enc.eot_token] and (len(period) != 1 or skip[period[0]]) and skip[common].all() and r["content"] > 0,
          f"{r['filler']:,} filler, {r['content']:,} content pieces")
    check("the tables are kept with the model (a resume reads the same)", m.foresight_ready()
          and all(k in m.state_dict() for k in ("fore_end", "fore_skip", "fore_prior", "fore_ready")))

    # ---- where it reads, what it guesses
    x = torch.tensor([ids[:120] + [enc.eot_token] * max(0, 121 - len(ids))][:1], device=dev)[:, :121]
    xi, yi = x[:, :-1], x[:, 1:]
    rp = m.read_points(xi)[0].cpu().numpy()
    tks = xi[0].cpu().numpy()
    at_end = [i for i in range(len(tks)) if ends[tks[i]]]
    gaps, last = [], -1
    for i in range(len(tks)):
        if ends[tks[i]] or tks[i] == enc.eot_token:
            last = i
        elif rp[i]:
            gaps.append(i - last)
    check("it reads after every sentence end, and every 16 pieces when no sentence ends",
          all(rp[i] for i in at_end) and gaps and all(g % 16 == 0 for g in gaps) and not any(rp[tks == enc.eot_token]),
          f"{int(rp.sum())} read points in {len(tks)} pieces ({len(at_end)} at sentence ends)")
    b = torch.zeros(len(at_end), dtype=torch.long, device=dev)
    t = torch.tensor(at_end, device=dev)
    bags = m.bags(xi, yi, b, t).cpu().numpy()
    ok = True
    for row, ti in zip(bags, at_end):
        nxt = []
        for j in range(ti + 1, min(ti + 1 + m.cfg.fore_span, len(ids))):
            nxt.append(ids[j])
            if ends[ids[j]]:
                break
        want = {p for p in nxt if not skip[p]}
        got = [p for p in row if p >= 0]
        ok &= set(got) == want and len(got) == len(set(got))
    first = [enc.decode([p]) for p in bags[0] if p >= 0]
    check("each bag = the content pieces of the text up to the next sentence end, each once, no order",
          ok, f"after '{enc.decode(ids[:at_end[0] + 1])[-12:]}': {first[:8]}")

    # ---- the loss
    m.train()
    xb = torch.tensor([ids[:64]] * 4, device=dev)
    loss, st = m.losses(xb[:, :-1], xb[:, 1:])
    check("training: the next-word loss + 0.1 x the foresight loss", "fore" in st and abs(
        loss.item() - (st["ce"].item() + 0.1 * st["fore"].item())) < 1e-3,
        f"next-word {st['ce'].item():.3f}, foresight {st['fore'].item():.3f}")
    loss.backward()
    g_head = m.fore.up.weight.grad.abs().sum().item()
    m.zero_grad(set_to_none=True)
    m._fore_w = 0.0
    l0, _ = m.losses(xb[:, :-1], xb[:, 1:])
    m._fore_w = 1.0
    lf, stf = m.losses(xb[:, :-1], xb[:, 1:])
    (lf - l0).backward()          # only the foresight loss's gradient
    g_first = m.blocks[0].attn.q.weight.grad.abs().sum().item()
    check("its gradient reaches the head and the whole trunk (down to the first layer)", g_head > 0 and g_first > 0,
          f"head {g_head:.2e}, layer 1 {g_first:.2e}")
    m.zero_grad(set_to_none=True)
    m.set_progress(0.0)
    m2 = Plain100F(Plain100FConfig(**{**SMALL, "fore_warmup": 0.05}))
    m2.set_progress(0.0)
    w0 = m2._fore_w
    m2.set_progress(0.025)
    w1 = m2._fore_w
    m2.set_progress(0.5)
    check("the weight warms up from 0 to 0.1 over the first 5%", w0 == 0 and abs(w1 - 0.05) < 1e-9 and m2._fore_w == 0.1)
    del m2
    m.eval()
    with torch.no_grad():
        lg = m._body(xb[:, :-1]).float()
        ref = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), xb[:, 1:].reshape(-1))
        _, lv, stv = m(xb[:, :-1], xb[:, 1:])
    check("the held-out loss is the next-word loss only (no head)", "fore" not in stv and abs(lv.item() - ref.item()) < 1e-3,
          f"{lv.item():.4f} vs {ref.item():.4f}")
    out, _ = m.generate(enc.encode_ordinary("The cat"), max_new_tokens=4)
    check("it writes word by word, like Plain-100M", len(out) <= 4)

    # ---- it learns: on repeated text, the head beats the blind guess
    torch.manual_seed(1)
    m.train()
    m._fore_w = 1.0
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(60):
        l_, _ = m.losses(xb[:, :-1], xb[:, 1:])
        l_.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    rep = m.foresight_report(np.asarray(xb.cpu()), dev, batch=4)
    check("it learns: the head guesses the next sentence's words better than the blind guess",
          rep["fore_val"] < rep["fore_blind"] - 0.5, f"{rep['fore_val']:.2f} vs blind {rep['fore_blind']:.2f} nats per bag word")

    # ---- the recipe
    from oazy.train import RECIPE, MENTOR, BEES, FAST, PRESETS
    pb = PRESETS["big"]
    check("Plain-100M's recipe, mentor, steps and learning rates exactly", RECIPE["plain100f"] == RECIPE["plain100"]
          and MENTOR["plain100f"] == MENTOR["plain100"] and "plain100f" in BEES and "plain100f" in FAST
          and pb["batch_plain100f"] == pb["batch_plain100"] and pb["lr_plain100f"] == pb["lr_plain100"]
          and pb["muon_lr_plain100f"] == pb["muon_lr_plain100"])
    from oazy.train import MUON_GRIDS
    check("the head's two grids learn with Muon, like the trunk's", all(n.endswith(MUON_GRIDS) for n in ("fore.up.weight", "fore.down.weight")))

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
            np.fromiter((t_ for x_ in ds for t_ in gpt.encode_ordinary(x_) + [EOT]), dtype=np.uint16).tofile(d / f"{split}.bin")
    (data / "hellaswag_val.jsonl").write_text("".join(
        json.dumps({"ctx": " ".join(fake_docs(1, rng)), "endings": fake_docs(4, rng), "label": rng.randint(0, 3)}) + "\n" for _ in range(24)))
    retok.run(data, "smollm2", ["fwe", "chat"], workers=2)
    runs = tmp / "runs"
    mentor = "smol-random"
    try:
        import transformers  # noqa: F401
    except ImportError:
        mentor = "none"
    argv = ["train", "--model", "plain100f", "--preset", "smoke", "--data", str(data), "--runs", str(runs), "--mentor", mentor,
            "--device", str(dev), "--compile", "0", "--tokens", "40000", "--batch-ramp", "512:0.5", "--mix", "fwe:0.9,chat:0.1@0",
            "--micro", "2", "--data-sub", "smol"]
    sys.argv = argv
    train.main()
    rd = runs / "plain100f"
    recs = [json.loads(l_) for l_ in (rd / "metrics.jsonl").read_text().splitlines()]
    tr = [r_ for r_ in recs if r_["kind"] == "train"]
    stt = json.loads((rd / "status.json").read_text())
    check("a whole tiny Plain-100F run (foresight logged on every line and on held-out text, HellaSwag, samples)",
          stt["state"] == "done" and any(r_["kind"] == "foresight" for r_ in recs) and any(r_["kind"] == "hellaswag" for r_ in recs)
          and all("fore_loss" in r_ for r_ in tr) and tr[-1]["fore_loss"] > 0,
          f"val loss {stt['latest']['val_loss']:.3f}, foresight {tr[-1]['fore_loss']:.2f}, mentor {mentor}")
    mdl, ck = train.load_model(rd / "ckpt.pt", dev)
    check("its checkpoint loads as Plain-100F, tables included", ck["kind"] == "plain100f" and type(mdl).__name__ == "Plain100F"
          and mdl.foresight_ready())
    sys.argv = argv
    train.main()          # resumes: the run is finished, nothing breaks
    del mdl

    if dev.type == "cuda":
        mg = Plain100F(Plain100FConfig(**{**SMALL, "max_len": 2048, "ctx": "2048:0", "fore_max": 8192})).to(dev).train()
        mg.prepare_foresight(enc, np.ones(V, dtype=np.int64))
        xg = torch.randint(1, V, (4, 1024), device=dev)
        xg[:, 50] = 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            le, _ = mg.losses(xg, xg)
        le.backward()
        mg.zero_grad(set_to_none=True)
        mg.enable_compile()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lc, _ = mg.losses(xg, xg)
        lc.backward()
        check("the compiled step matches", torch.isfinite(lc).item() and abs(lc.item() - le.item()) < 0.05,
              f"{lc.item():.4f} vs {le.item():.4f}")
        del mg
        if not a.no_race:
            # 32 x 2,048 = one real pass (--micro 32 at full windows)
            xb = torch.randint(1, V, (32, 2048), device=dev)
            speed = {}
            for name, mk in (("Plain-100M", lambda: Plain100(Plain100Config())), ("Plain-100F", lambda: Plain100F(Plain100FConfig()))):
                torch._dynamo.reset()
                torch.manual_seed(0)
                mf = mk().to(dev).train()
                if hasattr(mf, "prepare_foresight"):
                    mf.prepare_foresight(enc, np.arange(V)[::-1].copy())
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
            pr, pf = speed["Plain-100M"], speed["Plain-100F"]
            check("speed race at full size (compiled bf16, 32 x 2,048, no mentor): the head costs under 15%",
                  pf > 0.85 * pr, f"Plain-100M {pr / 1e3:.0f}k, Plain-100F {pf / 1e3:.0f}k tokens/s ({pf / pr - 1:+.0%})")
    tmp_ctx.cleanup()
    term.tests_passed("Plain-100F")


if __name__ == "__main__":
    main()
