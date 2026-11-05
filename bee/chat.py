"""Talk to Bee0.2B, and see inside it while it answers.

    python -m bee.chat --port 8002            (on RunPod: the pod's port 8002 link, see docs/RUNPOD.md)

Two ways to use it:
  Chat    a conversation. Bee saw conversations written as "User: ... / Assistant: ..." near the
          end of its training (oazy/mix.py), so it knows the shape of one. Before that phase it is a
          plain text model and will mostly ramble.
  Write   Bee carries on any text you start.

The conversation is kept between messages (memories and attention keys carry on, nothing is
re-read). Click any word to see what Bee expected there, how surprised each memory layer was, and
which earlier words each attention layer looked back at.
"""
from __future__ import annotations

import argparse
import json
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import torch
import torch.nn.functional as F

from oazy.data import get_tokenizer, default_dir, EOT
from oazy.sample import sample_logits
from oazy.probes import token_strings
from oazy.train import load_model

WEB = Path(__file__).parent / "web"
APP = Path(__file__).resolve().parent.parent / "oazy" / "web"     # shared look: style.css, common.js, icon.svg


class Brain:
    def __init__(self, ckpt: Path, device):
        self.model, st = load_model(ckpt, device)
        if st.get("kind") != "bee":
            raise SystemExit("this checkpoint is not a Bee0.2B checkpoint")
        self.device = device
        self.enc = get_tokenizer()
        kinds = self.model.cfg.kinds()
        self.mem_layers = [i for i, k in enumerate(kinds) if k == "M"]
        self.att_layers = [i for i, k in enumerate(kinds) if k == "A"]
        self.meta = {"step": st.get("step"), "tokens": st.get("tokens"), "params": self.model.num_params(),
                     "layout": self.model.layout(), "mem_layers": self.mem_layers, "att_layers": self.att_layers,
                     "layers": len(kinds)}
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.states, self.logits, self.n = None, None, 0
        self.need_sep = False            # does the next chat turn need a blank line first?

    @torch.no_grad()
    def _read(self, token, source):
        p_word = None if self.logits is None else float(F.softmax(self.logits.float()[:50257], -1)[token])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            logits, self.states, infos = self.model.inspect(torch.tensor([[token]], device=self.device), self.states)
        self.logits = logits[0, -1].float()
        probs = F.softmax(self.logits[:50257], -1)
        pv, pi = torch.topk(probs, 6)
        looks = []
        for inf in infos:
            if inf["kind"] == "A":
                w = inf["looks"].float()
                v, ix = torch.topk(w, min(3, w.numel()))
                looks.append([[int(i), round(float(x), 3)] for x, i in zip(v, ix)])
        i = self.n
        self.n += 1
        return {"type": "word", "i": i, "text": token_strings(self.enc, [token])[0], "source": source,
                "p": None if p_word is None else round(p_word, 5),
                "surprise": [round(inf["surprise"], 3) for inf in infos if inf["kind"] == "M"],
                "looks": looks,
                "top": [[token_strings(self.enc, [int(t)])[0], round(float(p), 4)] for p, t in zip(pv, pi)]}

    def _feed(self, text, source):
        for t in self.enc.encode_ordinary(text):
            yield self._read(t, source)

    def run(self, text, mode, n_new, temperature, top_k):
        with self.lock:
            if self.n == 0:
                yield self._read(EOT, "frame")                  # "a new text starts here"
            if mode == "chat":
                yield from self._feed(("\n\n" if self.need_sep else "") + "User:", "frame")
                yield from self._feed(" " + text.strip(), "you")
                yield from self._feed("\n\nAssistant:", "frame")
            elif text:
                yield from self._feed(text, "you")
            out = ""
            self.need_sep = True
            for _ in range(n_new):
                t = sample_logits(self.logits, temperature, top_k)
                if t == EOT:
                    yield {"type": "eos"}
                    break
                piece = self.enc.decode([t])
                if mode == "chat" and piece.strip() in ("User", "User:") and (out.endswith("\n") or not out.strip()):
                    self.need_sep = False                       # Bee handed the turn back: "\n\n" is already read
                    break
                out += piece
                yield self._read(t, "model")
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
            if name.endswith((".js", ".css", ".svg")) and "/" not in name and (APP / name).is_file():
                ctype = "text/javascript" if name.endswith(".js") else "image/svg+xml" if name.endswith(".svg") else "text/css"
                return self.send(200, (APP / name).read_bytes(), ctype + "; charset=utf-8")
            if name == "api/info":
                return self.send(200, json.dumps(brain.meta).encode())
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
                    for ev in brain.run(q.get("text", ""), q.get("mode", "chat"), int(q.get("n", 120)),
                                        float(q.get("temp", 0.7)), int(q.get("top_k", 40))):
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
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--run", default="bee", help="which run folder to open")
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    a.ckpt = a.ckpt or f"{default_dir('runs')}/{a.run}/ckpt.pt"
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"loading {a.ckpt} on {device} ...", flush=True)
    brain = Brain(Path(a.ckpt), device)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(brain))
    print(f"Bee0.2B chat on http://{a.host}:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
