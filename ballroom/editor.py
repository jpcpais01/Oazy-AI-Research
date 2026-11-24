"""Open the room editor on this machine:  python3 -m ballroom.editor [--port 8010] [--runs DIR]

The objects model runs inside the page. Pixel models (ballroom.eye) run here, on this machine's GPU if it has one:
every eye.pt under --runs shows up in the page's model picker as soon as it is saved.
"""
import argparse, glob, gzip, http.server, json, os, threading
import numpy as np

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "editor")
WEIGHTS = os.path.join(HERE, "weights.json")
packed = {}                                   # the model file, squeezed once per version
RUNS = None
nets, lock = {}, threading.Lock()             # pixel models, loaded once each (reloaded when the file changes)


def pixel_models():
    """Every pixel model under RUNS, by a short name: eye, eye_emotion, eye_v1 ..."""
    out = {}
    for f in sorted(glob.glob(os.path.join(RUNS, "*", "eye.pt")) + glob.glob(os.path.join(RUNS, "eye*.pt"))):
        rel = os.path.relpath(f, RUNS)
        name = os.path.dirname(rel) if rel.endswith(os.sep + "eye.pt") else rel[:-3]
        out[name] = f
    return out


def eye_net(path):
    import torch
    from .eye import Eye
    t = os.path.getmtime(path)
    if path not in nets or nets[path][0] != t:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ck = torch.load(path, map_location=dev, weights_only=False)
        net = Eye(**ck["cfg"]).to(dev); net.load_state_dict(ck["model"], strict=False); net.eval()
        nets[path] = (t, net, dev, ck.get("steps"), ck.get("plat"))
    return nets[path]


def eye_run(q):
    """One room through a pixel model: the main ball's path step by step, its one-shot guess, and where each nudged copy ends."""
    import torch
    from . import sim
    from .eye import render
    path = pixel_models().get(q["model"])
    if not path:
        raise ValueError("no such model: " + str(q["model"]))
    with lock:                                                    # one room at a time on the GPU
        _, net, dev, _, _ = eye_net(path)
        balls = [q["ball"]] + list(q.get("starts", []))
        n = len(balls)
        boxes = np.zeros((n, sim.MAX_BOX, 4)); boxes[:, :, :2] = sim.FAR; boxes[:, :4] = sim.WALLS
        for i, pl in enumerate(q["plats"][:sim.MAX_BOX - 4]):
            boxes[:, 4 + i] = pl
        p0 = np.array([[b["x"], b["y"]] for b in balls], dtype=np.float64)
        v0 = np.array([[b["vx"], b["vy"]] for b in balls], dtype=np.float64)
        p1, _ = sim.step(p0, v0, boxes, sim.SUB)                  # the second picture: the ball 1/30 s later
        t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev)
        with torch.no_grad():
            z = net.see(render(t(p0), t(p1), t(boxes), net.res))
            shot = net.shot(z[:1])[0].tolist()
            law, ends = [], [None] * n
            live = torch.ones(n, dtype=torch.bool, device=dev)
            for k in range(361):
                y = net.read(z)
                if live[0]:
                    law += y[0, :2].tolist()
                done = live & ((y[:, 2] > 0) | (k == 360))
                for i in done.nonzero()[:, 0].tolist():
                    ends[i] = y[i, :2].tolist()
                live &= ~done
                if not live.any():
                    break
                z = net.step(z)
    return {"law": law, "done": True, "shot": shot, "ends": ends[1:]}


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
        elif path == "/models":
            ms = []
            for name, f in pixel_models().items():
                ms.append({"name": name, "age": int(os.path.getmtime(f))})
            self.send(200, json.dumps(ms).encode(), "application/json", [("Cache-Control", "no-store")])
        else:
            self.send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path.split("?")[0] != "/eye":
            return self.send(404, b"not found", "text/plain")
        try:
            q = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            self.send(200, json.dumps(eye_run(q)).encode(), "application/json", [("Cache-Control", "no-store")])
        except Exception as e:                                    # the page shows the reason
            self.send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--runs", default=None, help="where trained pixel models live (default: the runs folder's ballroom/)")
    a = ap.parse_args()
    if a.runs is None:
        from oazy.data import default_dir
        a.runs = os.path.join(default_dir("runs"), "ballroom")
    RUNS = a.runs
    print(f"pixel models from {RUNS}: " + (", ".join(pixel_models()) or "none yet"), flush=True)
    print(f"room editor on http://localhost:{a.port}" + ("" if version() else "  (no model file yet: run python3 -m ballroom.export)"), flush=True)
    http.server.ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
