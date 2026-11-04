"""Talk to a trained Engram and watch its memories change, word by word.

    python -m engram.chat --ckpt /workspace/engram_runs/engram/ckpt.pt --port 8001

Then open http://localhost:8001 (on RunPod: the pod's HTTP port 8001 link).
The text is kept between messages, so the memories keep everything read so far (that is the
point of the model: no re-reading, the memory simply carries on). "New text" wipes them.
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

from .data import get_tokenizer, EOT
from .model import sample_logits
from .probes import _img, token_strings
from .train import load_model

WEB = Path(__file__).parent / "web"


def r3(x):
    return [round(float(v), 3) for v in x]


class Brain:
    def __init__(self, ckpt: Path, device):
        self.model, st = load_model(ckpt, device)
        if st.get("kind", "engram") != "engram":
            raise SystemExit("this checkpoint is the GPT baseline; the chat app needs an Engram checkpoint")
        self.cfg = self.model.cfg
        self.device = device
        self.enc = get_tokenizer()
        self.meta = {"step": st.get("step"), "tokens": st.get("tokens"), "params": self.model.num_params(),
                     "layers": self.cfg.n_layer, "heads": self.cfg.n_head, "k": self.cfg.head_k, "v": self.cfg.head_v,
                     "memory_numbers": self.model.memory_numbers()}
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.states = [None] * self.cfg.n_layer
        self.logits = None
        self.n = 0

    @torch.no_grad()
    def _read(self, token, source, layer, head):
        """Read one word: update every memory, and report what happened inside."""
        m = self.model
        p_word = None
        if self.logits is not None:
            p_word = float(F.softmax(self.logits.float()[:50257], -1)[token])
        old = self.states[layer][1][0, head].clone() if self.states[layer] is not None else None
        x = m.embed(torch.tensor([[token]], device=self.device))
        new, infos = [], []
        for b, s in zip(m.blocks, self.states):
            y, s, info = b.mem.inspect(b.n1(x), s)
            x = x + y
            if b.mlp is not None:
                x = x + b.mlp(b.n2(x))
            new.append(s)
            infos.append(info)
        self.states = new
        self.logits = F.linear(m.norm(x), m.embed.weight)[0, -1]
        S = new[layer][1][0, head]
        kept = infos[layer]["fade"][0, 0, head]
        stamp = S - kept * old if old is not None else S
        probs = F.softmax(self.logits.float()[:50257], -1)
        pv, pi = torch.topk(probs, 6)
        mem, ms = _img(S)
        st, ss = _img(stamp)
        i = self.n
        self.n += 1
        return {
            "type": "word", "i": i, "text": token_strings(self.enc, [token])[0], "source": source,
            "p": None if p_word is None else round(p_word, 5),
            "surprise": r3(inf["surprise"][0, 0].mean() for inf in infos),       # per block
            "trust": r3(inf["trust"][0, 0].mean() for inf in infos),
            "push": [r3(inf["push"][0, 0]) for inf in infos],                    # block x memory
            "top": [[token_strings(self.enc, [int(t)])[0], round(float(p), 4)] for p, t in zip(pv, pi)],
            "mem": mem, "mem_scale": ms, "stamp": st, "stamp_scale": ss,
        }

    def view(self, layer, head):
        if self.states[layer] is None:
            return {"empty": True}
        mem, ms = _img(self.states[layer][1][0, head])
        return {"mem": mem, "mem_scale": ms}

    def run(self, text, n_new, temperature, top_k, layer, head):
        with self.lock:
            ids = self.enc.encode_ordinary(text) if text else []
            if not ids and self.n == 0:
                ids = [EOT]
            for t in ids:
                yield self._read(t, "you", layer, head)
            for _ in range(n_new):
                t = sample_logits(self.logits, temperature, top_k)
                if t == EOT:
                    yield {"type": "eos"}
                    break
                yield self._read(t, "model", layer, head)
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
            L, Hh = brain.cfg.n_layer, brain.cfg.n_head
            layer = min(L - 1, max(0, int(q.get("layer", L - 1))))
            head = min(Hh - 1, max(0, int(q.get("head", 0))))
            if name == "api/info":
                return self.send(200, json.dumps(brain.meta).encode())
            if name == "api/reset":
                with brain.lock:
                    brain.reset()
                return self.send(200, b"{}")
            if name == "api/view":
                with brain.lock:
                    return self.send(200, json.dumps(brain.view(layer, head)).encode())
            if name == "api/stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    for ev in brain.run(q.get("text", ""), int(q.get("n", 60)), float(q.get("temp", 0.8)),
                                        int(q.get("top_k", 40)), layer, head):
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
    ap.add_argument("--ckpt", default="/workspace/engram_runs/engram/ckpt.pt")
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
