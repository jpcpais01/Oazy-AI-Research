"""Engram live training dashboard. Reads the run folders and serves a web page that refreshes itself.

    python -m oazy.dashboard --runs /workspace/oazy_runs --port 8000

Then open http://localhost:8000 (on RunPod: the pod's HTTP port 8000 link).
Uses only the Python standard library.
"""
from __future__ import annotations

import argparse
import json
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, unquote

WEB = Path(__file__).parent / "web"


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
            self.send(200, json.dumps(obj).encode())

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            parts = [p for p in path.split("/") if p]
            if not parts:
                return self.send(200, (web / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
            if len(parts) == 1 and parts[0].endswith((".js", ".css", ".svg")) and (web / parts[0]).is_file():
                ctype = "text/javascript" if parts[0].endswith(".js") else "image/svg+xml" if parts[0].endswith(".svg") else "text/css"
                return self.send(200, (web / parts[0]).read_bytes(), ctype + "; charset=utf-8")
            if parts == ["api", "runs"]:
                out = []
                if runs.exists():
                    for d in sorted(runs.iterdir()):
                        if d.is_dir() and (d / "config.json").exists():
                            out.append({"name": d.name, "config": read_json(d / "config.json"),
                                        "status": read_json(d / "status.json")})
                return self.js(out)
            if len(parts) >= 4 and parts[:2] == ["api", "run"]:
                d = runs / parts[2]
                if ".." in parts[2] or not d.is_dir():
                    return self.send(404, b"{}")
                what = parts[3]
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
