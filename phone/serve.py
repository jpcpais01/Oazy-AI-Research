"""Share the phone app and the phone models from the pod, so the phone can fetch them over the pod's port link.

    python3 -m phone.serve                 # on port 8002, which the pod shows on its 8001 link (docs/RUNPOD.md)
    python3 -m phone.serve --run coral-chat   # the one command: converts that run for the phone first if needed

The page lists the app (phone/dist/oazy.apk, if built) and every model written by phone.export (<runs>/*/phone/*.oazy),
each with an "Open in Oazy" button (the app then downloads it itself, and can resume a cut download) and a plain
download link. Downloads can resume (HTTP Range). Only these files are served, read-only. Stop it with Ctrl-C when
done; the chat apps use port 8002 too (docs/RUNPOD.md), so do not run both at once.
"""
from __future__ import annotations

import argparse
import html
import os
import subprocess
import sys
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote

HERE = Path(__file__).resolve().parent


def files(runs: Path):
    out = {}
    apk = HERE / "dist" / "oazy.apk"
    if apk.is_file():
        out["oazy.apk"] = apk
    for p in sorted(runs.glob("*/phone/*.oazy")):
        out.setdefault(p.name, p)
    return out


def header(p: Path):
    """The model file's settings (name, size, tokens trained ...)."""
    with open(p, "rb") as f:
        if f.read(8) != b"OAZYPHN1":
            return {}
        n = int.from_bytes(f.read(8), "little")
        text = f.read(min(n, 1 << 20)).decode(errors="replace")
    return dict(l.split(" ", 1) for l in text.splitlines() if " " in l and not l.startswith("tensor "))


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Oazy · phone</title><style>
body{{margin:0;background:#050507;color:#ecebf3;font:15px/1.6 Inter,system-ui,sans-serif;padding:28px 16px 60px}}
.w{{max-width:560px;margin:0 auto}} .k{{font:500 10.5px ui-monospace,monospace;letter-spacing:.24em;color:#b4b3c4}}
h1{{font:600 30px/1.1 system-ui,sans-serif;letter-spacing:-.02em;margin:10px 0 8px}} p{{color:#b4b3c4;margin:0 0 22px}}
.c{{border:1px solid rgba(255,255,255,.12);background:linear-gradient(180deg,rgba(255,255,255,.06),rgba(255,255,255,.02));border-radius:20px;padding:16px 18px;margin:0 0 12px;
box-shadow:inset 0 1px 0 rgba(255,255,255,.18),-.75px 0 0 rgba(255,95,174,.2),.75px 0 0 rgba(79,216,255,.22)}}
.c b{{font:600 18px system-ui,sans-serif}} .c small{{display:block;font:11px ui-monospace,monospace;letter-spacing:.06em;color:#7d7c8f;margin:2px 0 12px}}
.c i{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:8px;background:var(--a);box-shadow:0 0 10px var(--a)}}
a.b{{display:inline-block;padding:10px 16px;border-radius:999px;font:500 11px ui-monospace,monospace;letter-spacing:.14em;text-transform:uppercase;text-decoration:none;margin:0 8px 6px 0;
color:#ecebf3;border:1px solid rgba(255,255,255,.16);background:rgba(255,255,255,.05)}} a.p{{color:#0b0b10;background:linear-gradient(180deg,#fff,#dcd8f5);border-color:#fff}}
</style></head><body><div class="w"><div class="k">OAZY AI RESEARCH · PHONE</div><h1>Take a model with you</h1>
<p>Install the app once, then tap <b>Open in Oazy</b> next to a model: the app downloads it and runs it on the phone, offline.</p>{rows}</div></body></html>"""


def page(host: str, fs: dict) -> bytes:
    rows = []
    for name, p in fs.items():
        size = p.stat().st_size
        link = f"https://{host}/{quote(name)}"
        if name.endswith(".apk"):
            rows.append(f'<div class="c" style="--a:#8f7bff"><i></i><b>The Oazy app</b><small>ANDROID · {size / 1e6:.1f} MB · allow installs from your browser when asked</small>'
                        f'<a class="b p" href="/{quote(name)}">Install the app</a></div>')
            continue
        h = header(p)
        tok = int(h.get("tokens_trained", 0) or 0)
        meta = f"{h.get('quant', '').upper()} · {size / 1e9:.2f} GB · {int(h.get('params', 0)) / 1e6:,.0f}M parameters" + (f" · {tok / 1e6:,.0f}M tokens" if tok else "")
        meta += f" · exported {h.get('exported', '')}"
        intent = f"intent://download?url={quote(link, safe='')}#Intent;scheme=oazy;package=ai.oazy.phone;end"
        rows.append(f'<div class="c" style="--a:{html.escape(h.get("accent", "#8f7bff"))}"><i></i><b>{html.escape(h.get("label", name))}</b><small>{html.escape(meta)}</small>'
                    f'<a class="b p" href="{html.escape(intent)}">Open in Oazy</a><a class="b" href="/{quote(name)}" download>Download the file</a></div>')
    if not rows:
        rows.append("<p>No phone models yet: python3 -m phone.export --run coral</p>")
    return PAGE.format(rows="".join(rows)).encode()


def handler(runs: Path):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            pass

        def do_HEAD(self):
            self.do_GET(head=True)

        def do_GET(self, head=False):
            fs = files(runs)
            name = unquote(self.path.split("?")[0].lstrip("/"))
            if name == "":
                host = self.headers.get("X-Forwarded-Host") or self.headers.get("Host") or "localhost"
                body = page(host, fs)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if not head:
                    self.wfile.write(body)
                return
            p = fs.get(name)
            if p is None:
                self.send_error(404)
                return
            size = p.stat().st_size
            start, end = 0, size - 1
            m = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range", ""))
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
                if start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            ctype = "application/vnd.android.package-archive" if name.endswith(".apk") else "application/octet-stream"
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Content-Disposition", f'attachment; filename="{name}"')
            self.end_headers()
            if head:
                return
            with open(p, "rb") as f:
                f.seek(start)
                left = end - start + 1
                try:
                    while left > 0:
                        b = f.read(min(1 << 20, left))
                        if not b:
                            break
                        self.wfile.write(b)
                        left -= len(b)
                except (BrokenPipeError, ConnectionResetError):
                    pass
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--runs", default=None, help="the runs folder (default: the lab's)")
    ap.add_argument("--run", default=None, help="comma-separated runs to convert for the phone first (skipped if done)")
    a = ap.parse_args()
    from oazy.data import default_dir
    runs = Path(a.runs or default_dir("runs"))
    for r in [x.strip() for x in (a.run or "").split(",") if x.strip()]:
        if list((runs / r / "phone").glob("*.oazy")):
            print(f"[phone] {r}: already converted", flush=True)
            continue
        print(f"[phone] {r}: converting for the phone (a few minutes, CPU only) ...", flush=True)
        cmd = [sys.executable, "-m", "phone.export", "--run", r]
        if a.runs:
            cmd += ["--ckpt", str(runs / r / "ckpt.pt")]
        if subprocess.call(cmd) != 0:
            raise SystemExit(f"[phone] converting {r} failed (see above)")
    fs = files(runs)
    print(f"[phone] sharing {len(fs)} file(s) on port {a.port}: " + ", ".join(fs) if fs else "[phone] nothing to share yet", flush=True)
    pod = os.environ.get("RUNPOD_POD_ID")
    link = f"https://{pod}-8001.proxy.runpod.net" if pod else "the pod's 8001 link (RunPod: Connect -> HTTP 8001)"
    print(f"[phone] on the phone, open in Chrome:  {link}", flush=True)
    print(f"[phone]   1. Install the app   2. Open in Oazy next to the model   (Ctrl-C here when done)", flush=True)
    os.nice(5)
    ThreadingHTTPServer(("0.0.0.0", a.port), handler(runs)).serve_forever()


if __name__ == "__main__":
    main()
