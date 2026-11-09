"""Oazy AI Research: the live training dashboard (and its pause / resume / end-early buttons). Reads the run folders and serves a web page that refreshes itself.
"/" is the page: what is live, the race in charts, every run as a tile (the old detailed "/classic" page was removed Nov 9).

    python -m oazy.dashboard --runs /workspace/oazy_runs --port 8000

Then open http://localhost:8000 (on RunPod: the pod's HTTP port 8000 link).
Uses only the Python standard library.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, unquote

from .data import run_dir

WEB = Path(__file__).parent / "web"
REPO = Path(__file__).resolve().parent.parent
APK = REPO / "phone" / "dist" / "oazy.apk"
PHONE_JOBS: dict = {}                     # run -> the phone.export process the library's button started


def read_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass                      # a line being written right now
    return out


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


_VER = {"t": 0.0, "v": None}


def version():
    """The lab's version: v<number of commits>, from git (so every `git pull` moves it on), plus the commit."""
    if time.time() - _VER["t"] > 30:
        repo = Path(__file__).resolve().parent.parent
        try:
            git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True,  # noqa: E731
                                            timeout=5).stdout.strip()
            n, sha, when = git("rev-list", "--count", "HEAD"), git("rev-parse", "--short", "HEAD"), git("log", "-1", "--format=%cs")
            _VER["v"] = {"version": f"v{n}" if n else None, "commit": sha or None, "date": when or None}
        except Exception:
            _VER["v"] = {"version": None, "commit": None, "date": None}
        _VER["t"] = time.time()
    return _VER["v"]


REPORT_FILES = ("config.json", "status.json", "metrics.jsonl", "samples.jsonl", "sft.json", "sft.jsonl")


def downloads(runs: Path, name: str):
    """What the model library offers to download for a run (and its chat version, <name>-chat):
    a small report bundle (status, config, every logged number, samples), the checkpoint, the phone files."""
    out = []
    for rn, label in ((name, ""), (name + "-chat", "chat ")):
        d = run_dir(runs, rn)
        if not d.is_dir():
            continue
        if any((d / f).exists() for f in REPORT_FILES):
            out.append({"run": rn, "file": "report.zip", "label": label + "report", "bytes": None})
        if (d / "ckpt.pt").is_file():
            out.append({"run": rn, "file": "ckpt.pt", "label": label + "checkpoint", "bytes": (d / "ckpt.pt").stat().st_size})
        job = PHONE_JOBS.get(rn)
        busy = job is not None and job.poll() is None
        phone = sorted((d / "phone").glob("*.oazy")) if (d / "phone").is_dir() else []
        for f in ([] if busy else phone):
            out.append({"run": rn, "file": f.name, "label": label + "phone file", "bytes": f.stat().st_size, "phone": True})
        if (d / "ckpt.pt").is_file() and (busy or not phone):
            failed = job is not None and not busy and job.returncode != 0
            out.append({"run": rn, "make_phone": True, "busy": busy, "failed": failed, "label": label + "phone file"})
    return out


def world_results(runs: Path):
    """World-first results (worldfirst/run.py writes <runs>/<name>/results.json with kind "worldfirst"), newest first."""
    out = []
    if runs.exists():
        for f in runs.glob("*/results.json"):
            r = read_json(f)
            if isinstance(r, dict) and r.get("kind") == "worldfirst":
                out.append(r)
    return sorted(out, key=lambda r: r.get("when") or "", reverse=True)


def _is_world(d: Path) -> bool:
    return d.is_dir() and ((d / "laws" / "metrics.jsonl").exists() or (d / "rival" / "metrics.jsonl").exists())


def world_runs(runs: Path):
    """The world-model runs (worldfirst/), newest first: name, phase, and when it last moved."""
    out = []
    if runs.exists():
        for d in runs.iterdir():
            if _is_world(d):
                r = world_run(runs, d.name, metrics=False)
                out.append({"name": d.name, "phase": r["phase"], "updated": r["updated"]})
    return sorted(out, key=lambda r: r["updated"], reverse=True)


def world_run(runs: Path, name: str, metrics: bool = True):
    """Everything the world dashboard shows for one run: both models' progress and logs, and the exam results."""
    d = runs / name
    out = {"name": name, "models": {}, "results": None, "status": read_json(d / "status.json")}
    updated = 0.0
    for kind in ("laws", "rival"):
        md = d / kind
        if not md.is_dir():
            continue
        st = read_json(md / "status.json")
        m = read_jsonl(md / "metrics.jsonl") if (metrics or st is None) else []
        if st is None and m:                      # a run from before status.json: read the last line of its log
            tr = [x for x in m if x.get("kind") == "train"]
            last = tr[-1] if tr else {}
            st = {"state": "done" if (md / "ckpt.pt").exists() else "training", "kind": kind, "step": last.get("step"),
                  "steps": None, "minutes": last.get("minutes"), "acc": last.get("acc"), "frontier": last.get("frontier")}
        cfg = read_json(md / "config.json")
        out["models"][kind] = {"status": st, "config": cfg, "metrics": m if metrics else None}
        for f in ("metrics.jsonl", "status.json"):
            if (md / f).exists():
                updated = max(updated, (md / f).stat().st_mtime)
    res = read_json(d / "results.json")
    if isinstance(res, dict) and res.get("kind") == "worldfirst":
        out["results"] = res
        updated = max(updated, (d / "results.json").stat().st_mtime)
    ph = (out["status"] or {}).get("phase")
    if ph is None:                                # infer it
        ms = out["models"]
        ph = ("done" if out["results"] else
              "exams" if all((ms.get(k) or {}).get("status", {}) and ms[k]["status"].get("state") == "done" for k in ("laws", "rival")) else
              "rival" if "rival" in ms else "laws")
    out["phase"] = ph
    out["world"] = world_info(d)
    out["live"] = ph != "done" and time.time() - updated < 300
    out["updated"] = updated
    return out


def world_info(d: Path):
    """Which world a run lives in (<run>/world.json; none: World 1), its families, and per family how many levels are
    practised and how many are secret."""
    try:
        from worldfirst import world as W
    except Exception:
        return None
    v = (read_json(d / "world.json") or {}).get("version", 1)
    if v not in W.WORLDS:
        return None
    fams, train, secret, _ = W.WORLDS[v]
    return {"version": v, "families": list(fams),
            "levels": {f: [len(train[f]), max(len(x) for x in secret[f].values())] for f in fams}}


def world_example(family: str, split: str):
    """One fresh episode of the world (worldfirst/world.py), with its answers, for the dashboard's 'what it is asked'."""
    import random
    from worldfirst import world as W
    if family not in W.FAMILIES:
        return None
    if split == "seen":
        lv, sp = random.choice(W.TRAIN_LEVELS[family]), "train"
    else:
        sp = split if split in W.SECRET[family] else random.choice(list(W.SECRET[family]))
        lv = random.choice(W.SECRET[family][sp])
    e = W.make(family, lv, sp, random.randrange(10 ** 9))
    def sentences(toks):
        out, cur = [], []
        for t in toks:
            cur.append(t)
            if t in (".", "?"):
                out.append(" ".join(cur[:-1]) + t)
                cur = []
        return out
    return {"family": family, "level": lv, "split": "seen" if sp == "train" else sp, "scene": sentences(e.tokens),
            "questions": [{"q": sentences(q.tokens)[0], "a": " ".join(q.answer)} for q in e.questions],
            "after": e.new_scene}



# ---------------------------------------------------------------- the visual dashboard (dashboard.html)
# What is really live: a trainer process on this machine (its command line names the run); where processes can't be
# seen (no /proc), a status written in the last 10 minutes. A run whose status still says "running"
# but whose trainer is gone was cut off (pod stopped, Ctrl-C twice, crash): it is NOT shown as live.
LIVE_STATES = ("running", "paused", "finishing", "starting")
_PROCS = {"t": 0.0, "names": None}


def trainer_names():
    """Run names of the oazy.train processes on this machine (None: no /proc to look at, e.g. a Mac)."""
    if time.time() - _PROCS["t"] < 5:
        return _PROCS["names"]
    names = None
    proc = Path("/proc")
    if proc.is_dir():
        names = set()
        for d in proc.iterdir():
            if not d.name.isdigit():
                continue
            try:
                a = (d / "cmdline").read_bytes().split(b"\0")
            except OSError:
                continue
            a = [x.decode("utf-8", "replace") for x in a]
            if "oazy.train" not in a:
                continue
            opt = lambda k: next((x.split("=", 1)[1] for x in a if x.startswith(k + "=")), None) or \
                (a[a.index(k) + 1] if k in a and a.index(k) + 1 < len(a) else None)  # noqa: E731
            n = opt("--name") or opt("--model")
            if n:
                names.add(n)
    _PROCS.update(t=time.time(), names=names)
    return names


def liveness(name: str, st: dict | None, age: float) -> str:
    """live | paused | finishing | starting | done | stopped | cut (cut off: says running, nothing is)."""
    st = st or {}
    state = st.get("state")
    if state == "done":
        return "done"
    procs = trainer_names()
    # this machine's processes decide when we can see them (a short grace for a restart); otherwise the clock does
    alive = (name in procs or age < 90) if procs is not None else age < 600
    if state in LIVE_STATES:
        if alive:
            return "live" if state == "running" else state
        return "cut"
    return "stopped"


def _fin(v):
    return isinstance(v, (int, float)) and v == v and abs(v) != float("inf")


def _thin(pts, n=240):
    """At most n points: the median of each stretch (the training loss jumps around; the median keeps its shape)."""
    if len(pts) <= n:
        return pts
    out, k = [], len(pts) / n
    for i in range(n):
        b = pts[int(i * k): max(int(i * k) + 1, int((i + 1) * k))]
        ys = sorted(p[1] for p in b)
        out.append([b[len(b) // 2][0], ys[len(ys) // 2]])
    return out


_SERIES: dict = {}


def series(d: Path):
    """The numbers the visual dashboard draws for one run, small: x is always tokens read."""
    f = d / "metrics.jsonl"
    try:
        key = (f.stat().st_mtime, f.stat().st_size)
    except OSError:
        return {}
    hit = _SERIES.get(str(f))
    if hit and hit[0] == key:
        return hit[1]
    train, val, hs, logt, world, arc, piqa, share = [], [], [], [], [], [], [], []
    for x in read_jsonl(f):
        k, t = x.get("kind"), x.get("tokens")
        if not _fin(t):
            continue
        if k == "train" and _fin(x.get("loss")):
            train.append([t, x["loss"]])
            if _fin(x.get("world_share")):
                share.append([t, x["world_share"]])
        elif k == "eval" and _fin(x.get("val_loss")):
            val.append([t, x["val_loss"]])
        elif k == "hellaswag" and _fin(x.get("acc_char")) and (x.get("full") or x.get("sample") == "spread" or (x.get("n") or 0) < 1000):
            hs.append([t, x["acc_char"], 1 if x.get("full") else 0])
        elif k == "logres" and _fin(x.get("acc_char")):
            logt.append([t, x["acc_char"], 0, x.get("chance")])      # every check is all 1,600 puzzles
        elif k == "world" and _fin(x.get("secret")):
            world.append([t, x["secret"], 1 if x.get("full") else 0, x.get("seen") if _fin(x.get("seen")) else None])
        elif k == "evals":
            if _fin(x.get("arc_easy")):
                arc.append([t, x["arc_easy"], 1 if x.get("full") else 0])
            if _fin(x.get("piqa")):
                piqa.append([t, x["piqa"], 1 if x.get("full") else 0])
    out = {"train": _thin(train), "val": val, "hs": hs, "logt": logt, "world": world, "arc": arc, "piqa": piqa,
           "share": _thin(share, 120)}
    _SERIES[str(f)] = (key, out)
    return out


def plan_turns(cf: dict):
    """The planned turns of a run, in tokens: where the world's share drops (Emerald) and where the cool-down starts."""
    total = cf.get("total_tokens") or 0
    out = []
    if not total:
        return out
    w = cf.get("world") or 0
    if w > 0:
        try:
            from .worldmix import EARLY, EARLY_X, LATE_X
        except Exception:
            EARLY, EARLY_X, LATE_X = 0.2, 3.0, 0.5
        out.append({"at": EARLY * total, "short": "world drops",
                    "label": f"world text {min(1, w * EARLY_X):.0%} → {w * LATE_X:.1%}".replace(".0%", "%")})
    if cf.get("schedule") == "wsd":
        dec = cf.get("decay", 0.2)
        out.append({"at": (1 - dec) * total, "short": "cool-down", "label": f"cool-down: learning rate falls to 0 over the last {dec:.0%}"})
    if cf.get("mentor") not in (None, "none") and cf.get("mentor_until") and not cf.get("mentor_floor"):
        out.append({"at": cf["mentor_until"] * total, "short": "mentor off", "label": "the mentor stops"})
    return sorted(out, key=lambda t: t["at"])


def overview(runs: Path):
    """Every run, with what the visual dashboard needs to draw its tile: state, progress, latest scores, a sparkline."""
    out = []
    if not runs.exists():
        return out
    for d in runs.iterdir():
        if not (d.is_dir() and (d / "config.json").exists()):
            continue
        try:
            out.append(_tile(d))
        except Exception as e:                     # one odd folder must never blank the page
            print(f"[overview] {d.name}: {type(e).__name__}: {e}", flush=True)
    return sorted(out, key=lambda r: r["updated"], reverse=True)


def _tile(d: Path):
    """One run's tile on the visual dashboard."""
    cf, st = read_json(d / "config.json") or {}, read_json(d / "status.json") or {}
    mt = max([(d / f).stat().st_mtime for f in ("status.json", "metrics.jsonl") if (d / f).exists()] or [0])
    age = time.time() - max(mt, st.get("updated") or 0)
    live = liveness(d.name, st, age)
    s = series(d)
    L = st.get("latest") or {}
    last = lambda k, i=1: (lambda a: a[i] if len(a) > i else None)((s.get(k) or [[]])[-1])  # noqa: E731
    return {
        "name": d.name, "model": cf.get("model"), "params": cf.get("params"), "active": cf.get("params_active"),
        "gpu": cf.get("gpu"), "total": cf.get("total_tokens"), "tokens": st.get("tokens") or 0, "state": st.get("state"),
        "live": live, "age": age, "updated": max(mt, st.get("updated") or 0), "tok_s": st.get("tok_s"), "eta_s": st.get("eta_s"),
        "elapsed": st.get("elapsed"), "ended_early": st.get("ended_early"), "reference": bool(cf.get("trained_tokens")),
        "world": cf.get("world") or 0,
        "turns": plan_turns(cf),
        "val": last("val"), "hs": last("hs"), "hs_full": bool(s.get("hs") and s["hs"][-1][2]), "logt": last("logt"),
        "world_secret": last("world"), "world_seen": last("world", 3), "arc": last("arc"), "piqa": last("piqa"),
        "hs_ref": L.get("hellaswag_char_full") if L.get("hellaswag_char_full") is not None else L.get("hellaswag_char"),
        "share": last("share"),
        "spark": [[p[0], round(p[1], 4)] for p in _thin(s.get("val") or s.get("train") or [], 40)],
    }


def make_phone(runs: Path, rn: str) -> bool:
    """Convert a run for the phone app in the background (phone.export, CPU, a few minutes)."""
    job = PHONE_JOBS.get(rn)
    if job is not None and job.poll() is None:
        return True
    d = run_dir(runs, rn)
    if not (d / "ckpt.pt").is_file():
        return False
    log = open(d / "phone-export.log", "a")
    PHONE_JOBS[rn] = subprocess.Popen([sys.executable, "-m", "phone.export", "--run", rn, "--ckpt", str(d / "ckpt.pt"), "--threads", "4"],
                                      cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT, preexec_fn=lambda: os.nice(10))
    return True


def make_handler(runs: Path, web: Path = WEB):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def js(self, obj):
            # NaN / Infinity (a broken training step) are not JSON: the page could not read anything at all
            body = re.sub(r'(?<![\w"])(-?Infinity|NaN)(?![\w"])', "null", json.dumps(obj))
            self.send(200, body.encode())

        def download(self, rn: str, fname: str):
            """GET /download/<run>/<file>: report.zip (made on the fly), ckpt.pt, or a phone/*.oazy; resumable."""
            if ".." in rn or "/" in rn or not re.fullmatch(r"[\w.-]+", fname):
                return self.send(404, b"{}")
            if rn == "app" and fname == "oazy.apk":
                return self.stream(APK, "oazy.apk", "application/vnd.android.package-archive")
            d = run_dir(runs, rn)
            if fname in ("report.md", "results.json") and (d / fname).is_file():      # a world run's report
                return self.stream(d / fname, f"{rn}-{fname}", "text/markdown" if fname.endswith(".md") else "application/json")
            if fname == "report.zip":
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                    for f in REPORT_FILES:
                        if (d / f).is_file():
                            z.write(d / f, f"{rn}/{f}")
                body = buf.getvalue()
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", f'attachment; filename="{rn}-report.zip"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return self.wfile.write(body)
            f = d / fname if fname == "ckpt.pt" else d / "phone" / fname if fname.endswith(".oazy") else None
            job = PHONE_JOBS.get(rn)
            if f is None or (fname.endswith(".oazy") and job is not None and job.poll() is None):
                return self.send(404, b"{}")
            return self.stream(f, f"{rn}-{fname}" if fname == "ckpt.pt" else fname)

        def stream(self, f: Path, filename: str, ctype="application/octet-stream"):
            if not f.is_file():
                return self.send(404, b"{}")
            size, start, end = f.stat().st_size, 0, None
            m = re.fullmatch(r"bytes=(\d+)-(\d*)", self.headers.get("Range") or "")
            if m:
                start, end = int(m.group(1)), int(m.group(2)) if m.group(2) else None
            end = size - 1 if end is None or end >= size else end
            if start > end:
                self.send_response(416); self.send_header("Content-Range", f"bytes */{size}"); self.end_headers(); return
            self.send_response(206 if m else 200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if m:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with open(f, "rb") as fh:
                fh.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = fh.read(min(1 << 20, left))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    left -= len(chunk)

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            parts = [p for p in path.split("/") if p]
            if not parts:
                return self.send(200, (web / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
            if parts == ["api", "overview"]:
                return self.js({"now": time.time(), "runs": overview(runs), "world": world_runs(runs)})
            if len(parts) == 1 and parts[0].endswith((".js", ".css", ".svg")) and (web / parts[0]).is_file():
                ctype = "text/javascript" if parts[0].endswith(".js") else "image/svg+xml" if parts[0].endswith(".svg") else "text/css"
                return self.send(200, (web / parts[0]).read_bytes(), ctype + "; charset=utf-8")
            if len(parts) == 3 and parts[0] == "download":
                return self.download(parts[1], parts[2])
            if parts == ["api", "version"]:
                return self.js(version())
            if parts == ["api", "world"]:
                return self.js(world_results(runs))
            if parts == ["world"]:
                return self.send(200, (web / "world.html").read_bytes(), "text/html; charset=utf-8")
            if parts == ["api", "world", "runs"]:
                return self.js(world_runs(runs))
            if len(parts) == 4 and parts[:3] == ["api", "world", "run"] and ".." not in parts[3] and _is_world(runs / parts[3]):
                return self.js(world_run(runs, parts[3]))
            if len(parts) == 5 and parts[:3] == ["api", "world", "example"]:
                ex = world_example(parts[3], parts[4])
                return self.js(ex) if ex else self.send(404, b"{}")
            if parts == ["api", "runs"]:
                out = []
                if runs.exists():
                    for d in sorted(runs.iterdir()):
                        if d.is_dir() and (d / "config.json").exists():
                            name = d.name
                            if name == "engram":                  # Memtrix's folder from before the rename
                                if (runs / "memtrix").exists():
                                    continue
                                name = "memtrix"
                            out.append({"name": name, "config": read_json(d / "config.json"),
                                        "status": read_json(d / "status.json")})
                return self.js(out)
            if len(parts) >= 4 and parts[:2] == ["api", "run"]:
                d = run_dir(runs, parts[2])
                if ".." in parts[2] or not d.is_dir():
                    return self.send(404, b"{}")
                what = parts[3]
                if what == "downloads":
                    return self.js(downloads(runs, parts[2]))
                if what == "series":
                    return self.js(series(d))
                if what == "metrics":
                    return self.js(read_jsonl(d / "metrics.jsonl"))
                if what == "samples":
                    return self.js(read_jsonl(d / "samples.jsonl"))
                if what == "snapshots":
                    return self.js(read_jsonl(d / "snapshots.jsonl"))
                if what == "snapshot" and len(parts) == 5 and parts[4].endswith(".json") and "/" not in parts[4]:
                    f = d / "snapshots" / parts[4]
                    if f.exists():
                        return self.send(200, f.read_bytes())
            self.send(404, b"{}")

        def do_POST(self):
            """POST /api/run/<name>/control {"action": "pause" | "resume" | "end"}: the trainer picks it up within a step."""
            parts = [p for p in unquote(urlparse(self.path).path).split("/") if p]
            if len(parts) == 4 and parts[:2] == ["api", "run"] and parts[3] == "phone" and ".." not in parts[2]:
                return self.js({"ok": make_phone(runs, parts[2])})
            if len(parts) != 4 or parts[:2] != ["api", "run"] or parts[3] != "control" or ".." in parts[2]:
                return self.send(404, b"{}")
            d = run_dir(runs, parts[2])
            st = read_json(d / "status.json") or {}
            try:
                n = int(self.headers.get("Content-Length") or 0)
                action = json.loads(self.rfile.read(min(n, 4096)) or b"{}").get("action")
            except Exception:
                action = None
            if action not in ("pause", "resume", "end") or not d.is_dir():
                return self.send(400, b'{"error": "bad request"}')
            if st.get("state") not in ("running", "starting", "paused"):
                return self.send(409, json.dumps({"error": f"{parts[2]} is not training right now"}).encode())
            tmp = d / "control.json.tmp"
            tmp.write_text(json.dumps({"action": action}))
            tmp.replace(d / "control.json")
            return self.js({"ok": True, "action": action})

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    from .data import default_dir
    a.runs = a.runs or default_dir("runs")
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(Path(a.runs)))
    print(f"dashboard on http://{a.host}:{a.port}  (runs: {a.runs})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
