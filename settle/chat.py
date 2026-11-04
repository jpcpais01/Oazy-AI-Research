"""Talk to a trained Settle model and watch its sheet settle on every word.

    python -m settle.chat --ckpt /workspace/settle_runs/settle/ckpt.pt --port 8001

Then open http://localhost:8001 (on RunPod: the pod's HTTP port 8001 link).
The sheet is kept between messages, so the model's thinking continues instead of restarting.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import torch

from .data import get_tokenizer, EOT
from .model import Settle, SettleConfig, sample_logits
from . import viz

WEB = Path(__file__).parent / "web"
CALIBRATION = [
    viz.PROBE_TEXT,
    "Tom had a big blue truck. He liked to push it in the sand. One day the truck broke, and Tom cried. "
    "His mom hugged him and said they could fix it together.",
    "The sun was hot, so the cat went to sleep under the tree. A little bird sang a happy song.",
]


def hexes(rgb):
    return ["#%02x%02x%02x" % tuple(int(v) for v in c) for c in rgb.reshape(-1, 3)]


class Brain:
    def __init__(self, ckpt: Path, device):
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        if st.get("kind", "settle") != "settle":
            raise SystemExit("this checkpoint is the GPT baseline; the chat app needs a Settle checkpoint")
        self.cfg = SettleConfig.from_dict(st["config"])
        self.model = Settle(self.cfg)
        self.model.load_state_dict(st["model"])
        self.model.to(device).eval()
        self.device = device
        self.enc = get_tokenizer()
        self.meta = {"step": st.get("step"), "tokens": st.get("tokens"), "params": self.model.num_params()}
        self.lock = threading.Lock()
        # colours: fit the palette on a few settled passages so colours mean the same thing all session
        cells, gists = [], []
        for text in CALIBRATION:
            s = self.model.new_state(device)
            for t in self.enc.encode_ordinary(text):
                self.model.feed(s, t)
            cells.append(s["sheet"])
            gists.append(s["ghist"][-1:])
        self.pal = viz.Palette(torch.cat(cells))
        g = []
        for text in CALIBRATION:
            s = self.model.new_state(device)
            for t in self.enc.encode_ordinary(text):
                _, info = self.model.feed(s, t)
                g.append(info["gist"])
        self.gpal = viz.Palette(torch.stack(g)[:, None])
        self.reset()

    def reset(self):
        self.state = self.model.new_state(self.device)
        self.pieces = []
        self.logits = None

    def info(self):
        c = self.cfg
        return {"rows": c.rows, "channels": c.channels, "lens_width": c.lens_width, "cap": c.settle_cap,
                "lens_cap": c.lens_cap, "dil": [2 ** r for r in range(c.rows)], "link_heads": c.link_heads, **self.meta}

    def _word_events(self, token, source):
        m, cfg = self.model, self.cfg
        i = len(self.pieces)
        piece = viz.token_strings(self.enc, [token])[0]
        self.pieces.append(piece)
        yield {"type": "word", "i": i, "text": piece, "source": source}
        t0 = time.time()
        logits, info = m.feed(self.state, token, record=True)
        ms = (time.time() - t0) * 1000
        for k, (f, ch) in enumerate(zip(info["frames"], info["changes"])):
            yield {"type": "step", "i": i, "k": k + 1, "col": hexes(self.pal.rgb(f)),
                   "change": [round(float(v), 5) for v in ch]}
        W = cfg.lens_width
        lens_frames = [hexes(self.pal.rgb(L)) for L in info["lens_frames"]]
        revision = (info["lens"] - info["anchor"]).pow(2).mean((1, 2)).sqrt()
        yield {"type": "lens", "i": i, "start": i - W + 1, "frames": lens_frames,
               "final": hexes(self.pal.rgb(info["lens"])), "revision": [round(float(v), 4) for v in revision]}
        probs = torch.softmax(logits.float(), -1)
        pv, pi = torch.topk(probs, 6)
        links = []
        if info["links"] is not None:
            att = info["links"]                          # (i+1, heads)
            for h in range(att.shape[1]):
                w, j = torch.topk(att[:, h], min(3, att.shape[0]))
                links.append([[int(jj), round(float(ww), 3)] for jj, ww in zip(j, w)])
        yield {"type": "settled", "i": i, "steps": info["steps"], "lens_steps": info["lens_steps"],
               "final": hexes(self.pal.rgb(info["column"])),
               "gist": hexes(self.gpal.rgb(info["gist"][None, None]))[0],
               "links": links, "ms": round(ms, 1),
               "top": [[viz.token_strings(self.enc, [int(t)])[0], round(float(p), 4)] for p, t in zip(pv, pi)]}
        self.logits = logits

    def run(self, text, n_new, temperature, top_k):
        with self.lock:
            ids = self.enc.encode_ordinary(text) if text else []
            if not ids and self.logits is None:
                ids = [EOT]
            for t in ids:
                yield from self._word_events(t, "you")
            for _ in range(n_new):
                nxt = sample_logits(self.logits, temperature, top_k)
                if nxt == EOT:
                    yield {"type": "eos"}
                    break
                yield from self._word_events(nxt, "model")
            yield {"type": "end"}


def make_handler(brain: Brain):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            name = u.path.strip("/")
            if name == "":
                return self.send(200, (WEB / "chat.html").read_bytes(), "text/html; charset=utf-8")
            if name.endswith((".js", ".css")) and "/" not in name and (WEB / name).is_file():
                ctype = "text/javascript" if name.endswith(".js") else "text/css"
                return self.send(200, (WEB / name).read_bytes(), ctype + "; charset=utf-8")
            if name == "api/info":
                return self.send(200, json.dumps(brain.info()).encode())
            if name == "api/reset":
                with brain.lock:
                    brain.reset()
                return self.send(200, b"{}")
            if name == "api/stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    for ev in brain.run(q.get("text", ""), int(q.get("n", 60)), float(q.get("temp", 0.8)),
                                        int(q.get("top_k", 40))):
                        self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self.close_connection = True
                return
            self.send(404, b"{}")

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/settle_runs/settle/ckpt.pt")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"loading {a.ckpt} on {device} ...", flush=True)
    brain = Brain(Path(a.ckpt), device)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(brain))
    print(f"chat on http://{a.host}:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
