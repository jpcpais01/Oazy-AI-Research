"""Where the 135M bodies' training time goes on this GPU, and what each speed switch buys (João, Nov 18: Bee-135 trains
at ~530k tokens a second on a B300, far under what its size suggests).

    python3 -m gnosis2.speed135                 # about 10 minutes on a free GPU; trains nothing, reads no data
    python3 -m gnosis2.speed135 --only bee,phi  # some rows only

Each row builds the body at full size, compiles it, and times a training pass of 32 x 2,048 words on made-up tokens
(bee6/profile.py): the whole pass, the layers alone, the word table + loss alone, one optimizer step. The first row
also lists the slowest GPU kernels. The table at the end gives every row's speed against Bee-135's, the share of the
chip's arithmetic in use, and a run's speed with the optimizer step counted in (a step every `--passes` passes).
Rows that change the model (MoD, the wide body) measure SPEED only: what they do to the loss needs a training run.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from .race135 import set_words

WIDE = ("n_layer=12 d_model=832 attn_heads=13 attn_kv_heads=13 attn_head_dim=64 mlp_hidden=3328 mod_layers= "
        "key_offset=3,7,11 fp8=0").split()                 # half the layers, wider (a bigger model: read its "chip in use")
ROWS = [
    # key, name, switches on top of Bee-135's, extra profile options, what it tells
    ("bee", "Bee-135 (the race's arm B)", [], [], "the reference"),
    ("phi", "Phi-135 (phrase hint, no value tables)", None, [], "the hint against the value tables"),
    ("fp8", "Bee-135 + FP8", ["fp8=1"], [], "8-bit matrix products"),
    ("mod", "Bee-135 + MoD on every other layer", ["mod_layers=2,4,6,8,10,12,14,16,18,20,22"], [], "layer skipping (changes the model)"),
    ("micro64", "Bee-135, 64 sequences a pass", [], ["--micro", "64"], "is the pass too small for this chip"),
    ("ctx1024", "Bee-135, 1,024-word windows", [], ["--ctx", "1024"], "attention's share"),
    ("chunk", "Bee-135, loss in bigger chunks", ["loss_chunk=16384"], [], "the word table + loss step"),
    ("wide", "12 layers of 832 (wider, bigger)", WIDE, [], "do wide layers use the chip better (a different, bigger model)"),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="comma-separated row keys: " + ",".join(k for k, *_ in ROWS))
    ap.add_argument("--preset", default="big")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--passes", type=int, default=8, help="passes per optimizer step in a run (524k tokens a step = 8)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--set", nargs="*", default=None, metavar="KEY=VALUE", help="replace Bee-135's switches (for a small try-out)")
    a = ap.parse_args(argv)
    import torch
    from bee6 import profile
    from oazy import term
    from oazy.data import default_dir
    pick = set(a.only.split(",")) if a.only else None
    base = a.set if a.set is not None else set_words("b")
    phi = (a.set + ["n_ve=0", "hint_rows=73000", "hint_dim=32", "hint_heads=4", "hint_orders=2,3"]) if a.set is not None else set_words("c")
    got = []
    for i, (key, name, extra, opts, why) in enumerate(ROWS):
        if pick and key not in pick:
            continue
        sets = phi if extra is None else (extra if key == "wide" and a.set is None else base + extra)
        term.header(f"{name}", why, color="blue")
        argv2 = ["--model", "gnosis2", "--preset", a.preset, "--steps", str(a.steps), "--kernels", "15" if not got else "0",
                 *(["--device", a.device] if a.device else []), *(["--runs", a.runs] if a.runs else []), *opts, "--set", *sets]
        try:
            r = profile.main(argv2)
            got.append((key, name, why, r, None))
        except (SystemExit, Exception) as e:               # a row that cannot run here says so; the others still do
            term.out(term.c(f"  this row did not run: {type(e).__name__}: {str(e)[:300]}", "yellow"), flush=True)
            got.append((key, name, why, None, f"{type(e).__name__}: {str(e)[:80]}"))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    ok = [g for g in got if g[3]]
    if not ok:
        return got
    ref = next((g[3] for g in ok if g[0] == "bee"), ok[0][3])

    def run_speed(r):
        ms = r["ms"]
        return r["tokens"] * a.passes / (a.passes * ms["whole pass"] + ms["optimizer step (once per 8 passes)"]) * 1000
    body = []
    for key, name, why, r, err in got:
        if not r:
            body.append([name, "did not run", "", "", "", "", "", "", err or ""])
            continue
        ms = r["ms"]
        body.append([name, f"{r['params'] / 1e6:.1f}M", term.rate(r["tok_s"]), f"{r['tok_s'] / ref['tok_s']:.2f}x",
                     f"{ms['layers (forward + backward)'] / ms['whole pass']:.0%}", f"{ms['word table + loss'] / ms['whole pass']:.0%}",
                     f"{ms['optimizer step (once per 8 passes)']:.0f} ms", term.rate(run_speed(r)),
                     (f"{r['mfu'] * 100:.0f}%" if r.get("mfu") else "-")])
    term.header("The 135M bodies' speed", f"{ok[0][3]['gpu']} · one training pass, made-up tokens · a step every {a.passes} passes")
    term.table(["", "size", "pass speed", "vs Bee-135", "layers", "table + loss", "optimizer step", "run speed", "chip in use"],
               body, right=(1, 2, 3, 4, 5, 6, 7, 8))
    term.out(term.c("  layers / table + loss = their share of a pass, each timed alone (they need not add to 100%). run speed = with "
                    "the optimizer step, without tests and data loading: a real run sits a little under it.", "dim"))
    out = Path(a.runs or default_dir("runs")) / "speed135.json"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([{"key": k, "name": n, "result": r, "error": e} for k, n, _, r, e in got], indent=1, default=str))
        term.out(term.c(f"  saved {out}", "dim"))
    except OSError:
        pass
    return got


if __name__ == "__main__":
    main()
