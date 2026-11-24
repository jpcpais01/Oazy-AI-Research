"""Open the room editor on this machine:  python3 -m ballroom.editor [--port 8010]"""
import argparse, http.server, os

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "editor")


class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=HERE, **k)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = ("<!doctype html><html><head><meta charset='utf-8'>"
                    "<meta name='viewport' content='width=device-width, initial-scale=1'></head><body style='margin:0'>"
                    + open(os.path.join(HERE, "index.html"), encoding="utf-8").read() + "</body></html>").encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        else:
            super().do_GET()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8010)
    a = ap.parse_args()
    print(f"room editor on http://localhost:{a.port}")
    http.server.ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
