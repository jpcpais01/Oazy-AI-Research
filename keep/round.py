"""Round: a set of think runs (keep/think.py) trained one after the other without anyone watching, and ONE table.

    python3 -m keep.round r2              # the runs listed in runpod/rounds/r2.txt
    python3 -m keep.round r2 --table      # start nothing: the table of these runs as they stand (also runs started by hand)
    bash runpod/round.sh r2               # the same in the tmux window "round-r2" (what to paste on the pod)
    cat <runs>/round-r2.md                # the table, rewritten whenever a run starts or ends

A round file has a run to a line: its name, a colon, the options given to `python3 -m keep.think` (lines that start
with # say what the round is for and head the table). Runs start in the file's order, `--at-once` at a time (2), and a
new one only while the GPU has `--free` MiB to spare (55,000: a fed run holds 30 to 48 GB and more at its exams), so
no run dies of another's memory. A run that is already examined is not trained again, so a round that was stopped can
be started again with the same line. A run that dies of memory is tried once more, after the others.

The table says of each run what the fed model learned on circuits (trained to depth 3):
    LOOP          depth 8 right (90% or more) at the most tellings: reading again is reasoning
    partly        depth 8 between 75 and 90%
    n steps       depth 1 to n right at 90% or more (at their best telling), depth 8 below 75%: the loop is not there,
                  or half formed (each step still loses accuracy)
    lookup        not even depth 1
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

from .run import runs_dir

ROUNDS = Path(__file__).resolve().parent.parent / "runpod" / "rounds"


def read_round(path: Path):
    """-> (what the round is for, [(name, options)])."""
    about, runs = [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("#"):
            about.append(line.lstrip("# "))
        elif line:
            name, _, opts = line.partition(":")
            runs.append((name.strip(), opts.strip()))
    assert len({n for n, _ in runs}) == len(runs), "two runs of one name"
    return " ".join(about), runs


def free_mib() -> float:
    """What the GPU has free, in MiB (no GPU here: all one could ask for)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
        return min(float(v) for v in out.stdout.split())
    except Exception:
        return float("inf")


def examined(runs: Path, name: str, arm: str):
    """The arm's exam of a finished run: {level: {tellings: [right, asked, commonest]}} over all its families, or None."""
    p = runs / name / "think.json"
    if not p.exists():
        return None
    ex = json.loads(p.read_text()).get("exam", {}).get(arm)
    return {lv: row for fam in ex.values() for lv, row in fam.items()} if ex else None


def share(row, r):
    return 100 * row[r][0] / max(1, row[r][1])


def steps(ex: dict) -> int:
    """How many steps of reasoning hold: the deepest d with every depth 1..d right at 90% or more at its best telling."""
    d = 0
    while f"depth {d + 1}" in ex and max(share(ex[f"depth {d + 1}"], r) for r in ex[f"depth {d + 1}"]) >= 90:
        d += 1
    return d


def verdict(ex: dict) -> str:
    d8 = ex.get("depth 8")
    if not d8:
        return ""
    last = max(d8, key=int)
    if share(d8, last) >= 90:
        return "LOOP"
    n = steps(ex)
    return "partly" if share(d8, last) >= 75 else f"{n} step{'s' * (n != 1)}" if n else "lookup"


def table(rnd: str, about: str, todo, state: dict, runs: Path, arm: str) -> str:
    """One line a run: depth 0 after one telling (is a fact filed where it is looked for at once?), depth 4 and depth 8
    by tellings, and the verdict."""
    shown = None
    L = [f"# Round {rnd}", "", about, "", f"Written {time.strftime('%H:%M UTC', time.gmtime())}. "
         + ", ".join(f"{c} {k}" for k in ("done", "running", "waiting", "failed", "no exam yet") for c in [sum(1 for n, _ in todo if state[n].startswith(k))] if c or k == "done") + ".", ""]
    rows = []
    for name, opts in todo:
        ex = examined(runs, name, arm) if state[name].startswith("done") else None
        cells = ["", "", "", ""]
        if ex:
            tell = sorted(ex.get("all", {}), key=int)
            shown = shown or (["1", "4", "8", "12"] if {"1", "4", "8", "12"} <= set(tell) else tell[:4])
            line = lambda lv: "/".join(f"{share(ex[lv], r):.0f}" for r in shown if r in ex[lv]) if lv in ex else ""
            cells = [f"{share(ex['depth 0'], tell[0]):.0f}" if "depth 0" in ex else "", line("depth 4"), line("depth 8"), verdict(ex)]
        rows.append(f"| {name} | {state[name]} | " + " | ".join(cells) + " |")
    at = "/".join(f"{r}x" for r in shown) if shown else "tellings"
    L += [f"| Run | State | Depth 0 at 1x | Depth 4 at {at} | Depth 8 at {at} | Verdict |", "|---|---|---|---|---|---|"] + rows
    L += ["", "Share right (%). Trained to depth 3. Depth 8: always giving the commonest answer scores about 57%, a reasoner of three "
          "steps 63%. LOOP = depth 8 at 90% or more at the most tellings; n steps = depth 1 to n at 90% or more, no further.", ""] + [f"- {name}: `{opts}`" for name, opts in todo]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("round", help="a name in runpod/rounds (r2), or the path of a round file")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--arm", default="feed", choices=["feed", "keep", "full"], help="the model every run trains (--only)")
    ap.add_argument("--at-once", type=int, default=2)
    ap.add_argument("--free", type=float, default=55000, help="start a run only while the GPU has this many MiB free")
    ap.add_argument("--settle", type=float, default=120, help="seconds between two starts (the first has to take its memory)")
    ap.add_argument("--poll", type=float, default=20)
    ap.add_argument("--table", action="store_true", help="start nothing: print the table of the runs as they stand")
    args = ap.parse_args(argv)
    path = Path(args.round) if Path(args.round).exists() else ROUNDS / f"{args.round}.txt"
    rnd = path.stem
    about, todo = read_round(path)
    runs = Path(args.runs or runs_dir())
    runs.mkdir(parents=True, exist_ok=True)
    out = runs / f"round-{rnd}.md"
    state = {n: "done" if examined(runs, n, args.arm) else "waiting" for n, _ in todo}
    waiting = [(n, o) for n, o in todo if state[n] == "waiting"]
    going, tried, last = {}, {}, 0.0

    def write():
        out.write_text(table(rnd, about, todo, state, runs, args.arm))

    if args.table:
        state = {n: "done" if v == "done" else "no exam yet" for n, v in state.items()}
        print(table(rnd, about, todo, state, runs, args.arm), flush=True)
        return
    write()
    print(f"round {rnd}: {len(todo)} runs, {len(todo) - len(waiting)} done already. Table: {out}", flush=True)
    while waiting or going:
        for name, (proc, log) in list(going.items()):
            if proc.poll() is None:
                continue
            log.close()
            del going[name]
            text = (runs / f"think-{args.arm}-{name}.log").read_text(errors="replace")[-4000:]
            if examined(runs, name, args.arm):
                state[name] = "done"
            elif "out of memory" in text and tried[name] < 2:
                state[name] = "waiting"
                waiting.append((name, dict(todo)[name]))
            else:
                state[name] = "failed (out of memory)" if "out of memory" in text else f"failed (see think-{args.arm}-{name}.log)"
            print(f"{time.strftime('%H:%M')} {name}: {state[name]}", flush=True)
            write()
        if waiting and len(going) < args.at_once and time.time() - last > args.settle and free_mib() > args.free:
            name, opts = waiting.pop(0)
            log = open(runs / f"think-{args.arm}-{name}.log", "a")
            cmd = [sys.executable, "-m", "keep.think", "--runs", str(runs), "--only", args.arm, "--name", name] + shlex.split(opts)
            going[name] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=Path(__file__).resolve().parent.parent), log)
            tried[name] = tried.get(name, 0) + 1
            state[name], last = "running", time.time()
            print(f"{time.strftime('%H:%M')} {name}: started ({opts})", flush=True)
            write()
        time.sleep(args.poll)
    print(out.read_text(), flush=True)
    print(f"written: {out}", flush=True)


if __name__ == "__main__":
    main()
