"""Open the room editor on this machine:  python3 -m ballroom.editor [--port 8010]"""
import argparse, gzip, http.server, os

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "editor")
WEIGHTS = os.path.join(HERE, "weights.json")
packed = {}                                   # the model file, squeezed once per version


def version():
    return int(os.path.getmtime(WEIGHTS)) if os.path.exists(WEIGHTS) else 0


class H(http.server.BaseHTTPRequestHandler):
    def send(self, code, body, kind, extra=()):
        self.send_response(code); self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):          # always the newest page; the model file is named by its version
            body = ("<!doctype html><html><head><meta charset='utf-8'>"
                    "<meta name='viewport' content='width=device-width, initial-scale=1'>"
                    f"<script>window.WEIGHTS_V={version()}</script></head><body style='margin:0'>"
                    + open(os.path.join(HERE, "index.html"), encoding="utf-8").read() + "</body></html>").encode()
            self.send(200, body, "text/html; charset=utf-8", [("Cache-Control", "no-store")])
        elif path == "/weights.json" and os.path.exists(WEIGHTS):
            v = version()
            if packed.get("v") != v:
                raw = open(WEIGHTS, "rb").read()
                packed.update(v=v, size=len(raw), body=gzip.compress(raw, 1))
            self.send(200, packed["body"], "application/json", [("Content-Encoding", "gzip"), ("X-Size", str(packed["size"])),
                                                                 ("Cache-Control", "public, max-age=31536000, immutable")])
        else:
            self.send(404, b"not found", "text/plain")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8010)
    a = ap.parse_args()
    print(f"room editor on http://localhost:{a.port}" + ("" if version() else "  (no model file yet: run python3 -m ballroom.export)"), flush=True)
    http.server.ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
