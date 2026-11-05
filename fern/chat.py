"""Talk to a trained Fern and watch every passage unfold, round by round.

    python -m fern.chat --port 8002                (on RunPod, see docs/RUNPOD.md for the port)
    python -m fern.chat --run fern2 --port 8002    (Fern2 unfolds the same way)

Fern writes 16 words at a time, coarse to fine: first where the passage is going (word 16), then
the middle of the way (word 8), then the middle of each half, and so on for 5 rounds. The page
shows each passage as a frond: one row per round, every word in the round it was decided, with
the chance Fern gave it and what else it considered.
"""
from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import torch

from engram import chat as engram_chat
from oazy.data import get_tokenizer, default_dir, EOT
from oazy.probes import token_strings
from oazy.train import load_model

WEB = Path(__file__).parent / "web"


class FernBrain:
    def __init__(self, ckpt: Path, device):
        self.model, st = load_model(ckpt, device)
        if st.get("kind") not in ("fern", "fern2"):
            raise SystemExit("this checkpoint is not a Fern or a Fern2")
        self.cfg = self.model.cfg
        self.device = device
        self.enc = get_tokenizer()
        self.meta = {"step": st.get("step"), "tokens": st.get("tokens"), "params": self.model.num_params(),
                     "passage": self.cfg.passage, "rounds": self.model.rounds,
                     "reader_layers": self.cfg.n_layer, "unfold_layers": self.cfg.unfold_layers}
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.ids = []

    def words(self, ids):
        return token_strings(self.enc, [int(t) for t in ids])

    def run(self, text, n_passages, temperature, top_k):
        with self.lock:
            new = self.enc.encode_ordinary(text) if text else []
            if not new and not self.ids:
                new = [EOT]
            self.ids += new
            yield {"type": "read", "words": self.words(new), "eot": [t == EOT for t in new]}
            for _ in range(n_passages):
                rec = []
                toks = self.model.unfold_passage(self.ids[-4096:], temperature, top_k, rec, alts=5)
                rounds = []
                for r in rec:
                    rounds.append({"round": r["round"], "words": [
                        {"at": i, "w": self.words([t])[0], "p": round(p, 4), "eot": t == EOT,
                         "alts": [[self.words([a])[0], round(v, 4)] for a, v in alts]}
                        for i, t, p, alts in r["words"]]})
                self.ids += toks
                yield {"type": "passage", "rounds": rounds}
                if EOT in toks:
                    break
            yield {"type": "end"}


def make_handler(brain: FernBrain):
    Base = engram_chat.make_handler(brain)

    class H(Base):
        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            name = u.path.strip("/")
            if name == "":
                return self.send(200, (WEB / "chat.html").read_bytes(), "text/html; charset=utf-8")
            if name == "api/info":
                return self.send(200, json.dumps(brain.meta).encode())
            if name == "api/reset":
                brain.reset()
                return self.send(200, b"{}")
            if name == "api/stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    for ev in brain.run(q.get("text", ""), int(q.get("n", 4)), float(q.get("temp", 0.8)),
                                        int(q.get("top_k", 40))):
                        self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self.close_connection = True
                return
            if name.startswith("api/"):
                return self.send(404, b"{}")
            return super().do_GET()                     # the shared look: style.css, common.js, icon.svg

    return H


def main():
    from http.server import ThreadingHTTPServer
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--run", default="fern", help="which run folder to open (fern or fern2)")
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    a.ckpt = a.ckpt or f"{default_dir('runs')}/{a.run}/ckpt.pt"
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"loading {a.ckpt} on {device} ...", flush=True)
    brain = FernBrain(Path(a.ckpt), device)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(brain))
    print(f"chat on http://{a.host}:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
