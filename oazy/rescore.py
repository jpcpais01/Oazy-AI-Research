"""Score finished runs' HellaSwag "per letter", for runs that were tested before that score existed.

The dashboard and landing page show HellaSwag scored per letter (`acc_char`: an ending's total
surprise divided by its number of letters; lm-evaluation-harness calls it acc_norm, and the SmolLM2
and other published small-model numbers are this kind). The trainer has logged it since Nov 5, 23:12
(PR #39). Runs tested before that (Bee_v2_0.2B and older) only have the per-token score, so the
dashboard shows "–" for them until they are re-tested from their checkpoint:

    python3 -m oazy.rescore                  # every finished run that lacks it (and GPT-2, if missing)
    python3 -m oazy.rescore --run bee2       # just one
    python3 -m oazy.rescore --list           # only print what each run has, change nothing

It runs all 10,042 questions once per run (a few minutes on a GPU for a 200M model), appends one
"full" HellaSwag record to the run's metrics.jsonl and adds `hellaswag_char_full` to its status.json.
Nothing else in the run is touched. While another run trains on the same GPU it slows that run a
little for those minutes; `--device cpu` avoids that but takes much longer.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .data import default_dir


def _records(run: Path):
    p = run / "metrics.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def _status(run: Path):
    try:
        return json.loads((run / "status.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def has_char(run: Path) -> bool:
    """True once any HellaSwag check of this run was scored per letter (all runs since Nov 5, 23:12)."""
    return any(r.get("kind") == "hellaswag" and r.get("acc_char") is not None for r in _records(run))


def describe(run: Path) -> str:
    hs = [r for r in _records(run) if r.get("kind") == "hellaswag"]
    full = [r for r in hs if r.get("full")]
    last = (full or hs or [None])[-1]
    if last is None:
        return "no HellaSwag yet"
    tok = f"{last['acc_norm'] * 100:.1f}% per token" if last.get("acc_norm") is not None else "?"
    ch = f"{last['acc_char'] * 100:.1f}% per letter" if last.get("acc_char") is not None else "per letter missing"
    return f"{ch}, {tok} ({'all' if last.get('full') else last.get('n', '?')} questions)"


def rescore(run: Path, data: Path, device):
    import time
    import torch
    from . import hellaswag
    from .train import Run, load_model
    if run.name == "openai-gpt2":
        from .gpt import load_openai_gpt2
        model = load_openai_gpt2(device)
    else:
        try:
            model, _ = load_model(run / "ckpt.pt", device, mmap=True)
        except RuntimeError:               # an old-style checkpoint file cannot be read lazily
            model, _ = load_model(run / "ckpt.pt", device)
    t0 = time.time()
    r = hellaswag.evaluate(model, data / "hellaswag_val.jsonl", device)
    recs = _records(run)
    base = {k: recs[-1].get(k, 0) for k in ("step", "tokens", "elapsed")} if recs else {"step": 0, "tokens": 0, "elapsed": 0}
    Run(run).log("metrics.jsonl", {"kind": "hellaswag", **base, **r, "full": True, "sample": "spread", "rescored": time.time()})
    st = _status(run)
    if st:
        st.setdefault("latest", {})["hellaswag_char_full"] = r["acc_char"]
        st["latest"].setdefault("hellaswag_full", r["acc_norm"])
        tmp = run / "status.json.tmp"
        tmp.write_text(json.dumps(st))
        os.replace(tmp, run / "status.json")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"{run.name}: {r['acc_char'] * 100:.2f}% per letter, {r['acc_norm'] * 100:.2f}% per token "
          f"({r['n']} questions, {time.time() - t0:.0f} s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", help="run folder name (repeat for several); default: all that need it")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--list", action="store_true", help="print what each run has and stop")
    a = ap.parse_args()
    runs, data = Path(a.runs or default_dir("runs")), Path(a.data or default_dir("data"))
    folders = sorted(p for p in runs.iterdir() if p.is_dir() and (p / "metrics.jsonl").exists())
    if a.list:
        for p in folders:
            print(f"{p.name:24s} {describe(p)}")
        return
    if a.run:
        todo = [runs / n for n in a.run]
    else:
        todo = [p for p in folders if not has_char(p) and ((p / "ckpt.pt").exists() or p.name == "openai-gpt2")
                and _status(p).get("state") not in ("running", "starting", "finishing")]   # never a live run
    if not todo:
        print("Every finished run already has HellaSwag per letter.")
        return
    import torch
    from .data import few_cpu_threads
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    few_cpu_threads(device)
    print("Re-testing HellaSwag per letter for: " + ", ".join(p.name for p in todo), flush=True)
    for p in todo:
        try:
            rescore(p, data, device)
        except Exception as e:      # one broken checkpoint should not stop the others
            print(f"{p.name}: skipped ({type(e).__name__}: {e})", flush=True)


if __name__ == "__main__":
    main()
