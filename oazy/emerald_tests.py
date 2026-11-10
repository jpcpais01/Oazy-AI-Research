"""Self-tests for Emerald (oazy/emerald.py): the world's text is built into Sado's folder (practice only, whole episodes,
held-out from its own seed), the emerald mix (30% early, then 5%, the rest in Sado's proportions; any share), and a
whole tiny Bee6 run with --data-sub emerald that reads it and takes the world's exams.

    python3 -m oazy.emerald_tests        (CPU, ~1 min; without internet it uses stand-in tokenizers)
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

from oazy import term


def check(name, ok, detail=""):
    term.check(name, ok, detail)


def main():
    term.tests_start("Emerald", "cpu")
    from bee2.tests import stand_ins, fake_docs
    from oazy import emerald, tok, worldmix
    from oazy.mix import resolve_mix, parse_mix, SADO_MIX, emerald_mix
    tmp_ctx = tempfile.TemporaryDirectory()
    tmp = Path(tmp_ctx.name)
    used = stand_ins(tmp)
    if used:
        term.info(f"no internet: stand-in {' and '.join(used)} tokenizer(s)")
    enc = tok.get("smollm2")

    # ---- the mix
    plan = parse_mix(resolve_mix("emerald"))
    ws = [(at, w.get("world", 0)) for at, w in plan]
    check("emerald = 30% world for the first 20%, then 5% (cool-down too); every phase adds up to 1",
          [round(w, 6) for _, w in ws] == [0.3, 0.05, 0.05] and [at for at, _ in ws] == [0.0, 0.2, 0.8]
          and all(abs(sum(w.values()) - 1) < 1e-6 for _, w in plan), str(ws))
    s0, e0 = parse_mix(SADO_MIX)[0][1], plan[1][1]
    check("the other sources keep Sado's proportions", abs(e0["fwe"] / e0["dclm"] - s0["fwe"] / s0["dclm"]) < 1e-4)
    check("emerald:0.05 halves it, emerald:0 is plain Sado",
          abs(parse_mix(resolve_mix("emerald:0.05"))[0][1]["world"] - 0.15) < 1e-9 and emerald_mix(0) == SADO_MIX)

    # ---- the plan written out (--emerald-plan)
    def wplan(spec):
        return [(at, round(w.get("world", 0), 6)) for at, w in parse_mix(resolve_mix("emerald:" + spec))]
    check("plan 0.30:0.2,0.05 = today's emerald, word for word", resolve_mix("emerald:0.30:0.2,0.05") == resolve_mix("emerald"))
    check("plan flat:0.20 = 20% world in every phase, cool-down too", wplan("flat:0.20") == [(0.0, 0.2), (0.8, 0.2)],
          str(wplan("flat:0.20")))
    check("plan 0.30:0.1,0.15 = 30% until 10%, then 15% (cool-down too)",
          wplan("0.30:0.1,0.15") == [(0.0, 0.3), (0.1, 0.15), (0.8, 0.15)], str(wplan("0.30:0.1,0.15")))
    from oazy.mix import emerald_plan, plan_average
    check("plan averages: flat 0.10 = 10%, 0.30:0.1,0.15 = 16.5%",
          abs(plan_average(emerald_plan("flat:0.10")) - 0.10) < 1e-9 and abs(plan_average(emerald_plan("0.30:0.1,0.15")) - 0.165) < 1e-9)
    bad = 0
    for spec in ("0.3:0.5,0.1:0.2,0", "1.5", "0.3,0.1"):
        try:
            emerald_plan(spec)
        except ValueError:
            bad += 1
    check("a plan out of order, over 100% or with a share missing its 'until' is refused", bad == 3)

    # ---- the build
    data = tmp / "data"
    rng = random.Random(0)
    sado = data / "sado"
    sado.mkdir(parents=True)
    for split, k in (("val", 300), ("train", 1500)):
        np.fromiter((t for d in fake_docs(k, rng) for t in enc.encode_ordinary(d) + [enc.eot_token]),
                    dtype=np.uint16).tofile(sado / f"{split}.bin")
    (data / "hellaswag_val.jsonl").write_text("".join(
        json.dumps({"ctx": " ".join(fake_docs(1, rng)), "endings": fake_docs(4, rng), "label": rng.randint(0, 3)}) + "\n"
        for _ in range(24)))
    out = emerald.build(data, tokens=60_000, val_tokens=20_000, workers=1)
    meta = json.loads((out / "meta.json").read_text())
    tr = np.fromfile(out / "train.bin", dtype=np.uint16)
    va = np.fromfile(out / "val.bin", dtype=np.uint16)
    check("the world is written into sado/world/ (train + held-out), whole episodes each ending with <|endoftext|>",
          meta["train_tokens"] == tr.size >= 60_000 and va.size >= 20_000 and tr[-1] == enc.eot_token and va[-1] == enc.eot_token,
          f"{tr.size:,} + {va.size:,} tokens")
    text = enc.decode(tr[:4000].tolist())
    check("it reads as the world's plain text, answers included", "Answer:" in text and "?" in text, text[:90].replace("\n", " "))
    first_tr = enc.decode(tr[: int(np.flatnonzero(tr == enc.eot_token)[0])].tolist())
    first_va = enc.decode(va[: int(np.flatnonzero(va == enc.eot_token)[0])].tolist())
    check("the held-out episodes come from another seed than the training ones", first_tr != first_va)
    from worldfirst import world as W
    lv_ok = all(lv in W.TRAIN_LEVELS[f] for k in range(3) for f, lv in worldmix.picks(emerald.TRAIN_SEED, k))
    check("only practice levels: the secret exams are never written into it", lv_ok)
    again = emerald.build(data, tokens=60_000, val_tokens=20_000, workers=1)
    check("a finished build is left alone", again == out and (out / "train.bin").stat().st_size == tr.nbytes)

    # ---- a tiny Bee6 run on Emerald
    from oazy import train
    runs = tmp / "runs"
    base = ["train", "--model", "bee6", "--preset", "smoke", "--data", str(data), "--runs", str(runs), "--mentor", "none",
            "--device", "cpu", "--compile", "0", "--tokens", "40000", "--batch-ramp", "512:0.5",
            "--micro", "2", "--set", "ctx=64:0,128:0.3", "attn_head_dim=16", "attn_heads=4", "attn_kv_heads=2", "fp8=0"]
    sys.argv = base + ["--name", "em", "--data-sub", "emerald"]
    train.main()
    rd = runs / "em"
    cf = json.loads((rd / "config.json").read_text())
    stw = json.loads((rd / "status.json").read_text())
    recs = [json.loads(l_) for l_ in (rd / "metrics.jsonl").read_text().splitlines()]
    wx = [x_ for x_ in recs if x_["kind"] == "world"]
    check("a tiny Bee6 run with --data-sub emerald reads the world as part of its mix and takes the world's exams",
          stw["state"] == "done" and "world:0.3" in cf["mix"] and wx and wx[-1]["full"] and (rd / "world_exam.json").exists(),
          cf["mix"][:80])
    sys.argv = base + ["--name", "bad", "--data-sub", "emerald", "--world", "0.1"]
    try:
        train.main()
        refused = False
    except SystemExit as e_:
        refused = "--world" in str(e_)
    check("Emerald refuses --world on top (the world would be there twice)", refused)
    tmp_ctx.cleanup()
    term.tests_passed("Emerald")


if __name__ == "__main__":
    main()
