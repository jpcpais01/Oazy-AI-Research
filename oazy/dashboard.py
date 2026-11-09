"""Oazy AI Research: the live training dashboard (and its pause / resume / end-early buttons). Reads the run folders and serves a web page that refreshes itself.

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
    out["live"] = ph != "done" and time.time() - updated < 300
    out["updated"] = updated
    return out


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
