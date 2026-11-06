"""The lab at a glance: what runs, how far each run is, which text is ready, the GPU and the disk.

    python3 -m oazy.status            # once
    watch -c -n 30 python3 -m oazy.status   # refreshed every 30 s (Ctrl-C to leave)

It only reads: status files in the runs folder, the data folders' sizes, tmux, nvidia-smi and the disk.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from .data import default_dir
from . import term

STATE = {"running": "green", "starting": "cyan", "paused": "yellow", "finishing": "cyan", "done": "blue", "stopped": "grey"}
SOURCES = {"fwe": "", "dclm": "dclm", "cosmo": "cosmo", "wiki": "wiki", "math": "math", "code": "code", "chat": "chat"}


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def gpus():
    q = sh(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits"])
    rows = []
    for line in filter(None, q.splitlines()):
        name, util, used, tot, temp, power = [x.strip() for x in line.split(",")]
        u = float(util or 0)
        rows.append([name.replace("NVIDIA ", ""), term.bar(u / 100, 12, "green" if u > 60 else "yellow") + f" {u:3.0f}%",
                     f"{float(used) / 1024:.0f} / {float(tot) / 1024:.0f} GB", f"{temp}°C", f"{float(power or 0):.0f} W"])
    return rows


def windows():
    out = sh(["tmux", "list-windows", "-a", "-F", "#{session_name}:#{window_name}\t#{pane_current_command}"])
    rows = []
    for line in filter(None, out.splitlines()):
        name, cmd = (line.split("\t") + [""])[:2]
        busy = cmd not in ("bash", "zsh", "sh", "fish")
        rows.append([name, term.c("busy", "green") if busy else term.c("idle", "grey"), cmd if busy else ""])
    return rows


def runs(folder: Path, every: bool):
    rows = []
    now = time.time()
    for st_path in sorted(folder.glob("*/status.json"), key=lambda p: -p.stat().st_mtime):
        try:
            st = json.loads(st_path.read_text())
        except Exception:
            continue
        state = st.get("state", "?")
        age = now - float(st.get("updated") or st_path.stat().st_mtime)
        if state in ("running", "starting", "finishing") and age > 600:
            state = "stalled?"                          # says running but has not written for 10 minutes
        if not every and state in ("done", "stopped") and age > 3 * 86400:
            continue
        tok, total = st.get("tokens") or 0, st.get("total_tokens") or 0
        frac = tok / total if total else 0
        lat = st.get("latest") or {}
        loss = lat.get("val_loss")
        rows.append([st_path.parent.name, term.c(state, STATE.get(state, "red")),
                     term.bar(frac, 12) + f" {frac * 100:5.1f}%", f"{term.tokens(tok)} / {term.tokens(total)}",
                     term.rate(st["tok_s"]) if st.get("tok_s") and state == "running" else "",
                     term.duration(st["eta_s"]) if st.get("eta_s") and state == "running" else "",
                     f"{loss:.3f}" if loss else "", term.duration(age) + " ago"])
    return rows


def data(root: Path):
    rows = []
    for name in ("smol", "tagus20"):
        d = root / name
        if not d.exists():
            continue
        sizes = {k: ((d / sub / "train.bin") if sub else (d / "train.bin")) for k, sub in SOURCES.items()}
        have = {k: p.stat().st_size // 2 if p.exists() else 0 for k, p in sizes.items()}
        total = sum(have.values())
        parts = " ".join(f"{k} {term.tokens(v)}" for k, v in have.items() if v)
        rows.append([name, term.tokens(total), term.c(parts or "empty", "dim")])
    return rows


def tagus_build():
    """The last progress line of a Tagus build running in tmux (runpod: /workspace/tagus20.log)."""
    log = Path("/workspace/tagus20.log")
    if not log.exists() or "tagus" not in sh(["tmux", "ls"]):
        return None
    lines = [re.sub(r"\033\[[0-9;]*m", "", ln).strip() for ln in log.read_text(errors="ignore").splitlines()[-40:]]
    last = [ln for ln in lines if "%" in ln and "tokens" in ln]
    head = [ln for ln in lines if ln.startswith("━━")]
    return (head[-1].strip("━ ") if head else "building"), (last[-1] if last else lines[-1] if lines else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--all", action="store_true", help="also runs that finished more than 3 days ago")
    a = ap.parse_args()
    runs_dir, data_dir = Path(a.runs or default_dir("runs")), Path(a.data or default_dir("data"))
    term.header("Oazy lab", time.strftime("%H:%M:%S"))
    g = gpus()
    if g:
        term.out("")
        term.table(["GPU", "busy", "memory", "temp", "power"], g)
    r = runs(runs_dir, a.all)
    term.out("")
    if r:
        term.table(["run", "state", "progress", "tokens", "speed", "left", "val loss", "updated"], r, right=(4, 5, 6))
    else:
        term.out(term.c(f"  no runs in {runs_dir}", "dim"))
    w = windows()
    if w:
        term.out("")
        term.table(["tmux window", "", "running"], w)
    dd = data(data_dir)
    if dd:
        term.out("")
        term.table(["text", "tokens", "sources"], dd, right=(1,))
    tb = tagus_build()
    if tb:
        term.rows([("Tagus build", f"{tb[0]}  {tb[1]}")])
    du = shutil.disk_usage(data_dir if data_dir.exists() else "/")
    term.out("")
    term.rows([("disk", f"{term.bar(du.used / du.total, 16, 'yellow' if du.free < 50e9 else 'green')} "
                        f"{du.free / 1e9:,.0f} GB free of {du.total / 1e9:,.0f} GB")])
    term.out(term.c("\n  watch a run: tmux attach -t oazy  (Ctrl-b then a window number; Ctrl-b then d to leave)", "dim"))


if __name__ == "__main__":
    main()
