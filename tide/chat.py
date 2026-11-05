"""Talk to a trained Tide and watch its drafts roll in, word by word.

    python -m tide.chat --ckpt /workspace/engram_runs/tide/ckpt.pt --port 8001

Then open http://localhost:8001 (on RunPod: the pod's HTTP port 8001 link).
Shows the belt of drafts (what Tide expects 1, 2, ... 8 words ahead) live, and for every word how
it arrived: what the draft of that word looked like 8 words before, 7 words before, ... 1 word before.
Engram's memory views (surprise per block) are kept too, since Tide reads with Engram's memories.
"""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F

from engram import chat as engram_chat
from engram.data import get_tokenizer
from engram.probes import _img, token_strings
from engram.train import load_model

WEB = Path(__file__).parent / "web"
VOCAB = 50257


def r3(x):
    return [round(float(v), 3) for v in x]


class TideBrain(engram_chat.Brain):
    def __init__(self, ckpt: Path, device):
        import threading
        self.model, st = load_model(ckpt, device)
        if st.get("kind") != "tide":
            raise SystemExit("this checkpoint is not a Tide; use engram.chat for Engram")
        self.cfg = self.model.cfg
        self.device = device
        self.enc = get_tokenizer()
        self.meta = {"step": st.get("step"), "tokens": st.get("tokens"), "params": self.model.num_params(),
                     "layers": self.cfg.n_layer, "heads": self.cfg.n_head, "k": self.cfg.head_k, "v": self.cfg.head_v,
                     "memory_numbers": self.model.memory_numbers(), "lookahead": self.cfg.lookahead}
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.states = [None] * self.cfg.n_layer
        self.belt_state = None
        self.logits = None
        self.n = 0
        self.history = deque(maxlen=self.cfg.lookahead)    # belt probabilities after each recent word

    def _words(self, ids):
        return token_strings(self.enc, [int(t) for t in ids])

    @torch.no_grad()
    def _read(self, token, source, layer, head):
        m, K = self.model, self.cfg.lookahead
        # how this word arrived: the drafts of it made 1..K words ago
        arrival = []
        for back, probs in enumerate(reversed(self.history)):          # back = 0: made just before this word
            d = back                                                   # slot d drafted the word d+1 ahead
            p = probs[d]
            top = int(p.argmax())
            arrival.append({"ahead": d + 1, "top": self._words([top])[0], "top_p": round(float(p[top]), 4),
                            "p": round(float(p[token]), 5)})
        p_word = arrival[0]["p"] if arrival else None
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
        raw = m.belt(m.norm(x), self.belt_state)                       # (K, 1, 1, D)
        self.belt_state = raw[:, :, -1].transpose(0, 1).contiguous()   # (1, K, D)
        probs = F.softmax(m.draft_logits(self.belt_state)[0, :, :VOCAB].float(), -1)   # (K, vocab)
        self.history.append(probs)
        self.logits = torch.log(probs[0] + 1e-12)
        belt = []
        for d in range(K):
            pv, pi = torch.topk(probs[d], 4)
            belt.append([[w, round(float(p), 4)] for w, p in zip(self._words(pi), pv)])
        S = new[layer][1][0, head]
        kept = infos[layer]["fade"][0, 0, head]
        stamp = S - kept * old if old is not None else S
        mem, ms = _img(S)
        st, ss = _img(stamp)
        i = self.n
        self.n += 1
        return {
            "type": "word", "i": i, "text": self._words([token])[0], "source": source,
            "p": p_word, "arrival": arrival, "belt": belt,
            "surprise": r3(inf["surprise"][0, 0].mean() for inf in infos),
            "trust": r3(inf["trust"][0, 0].mean() for inf in infos),
            "push": [r3(inf["push"][0, 0]) for inf in infos],
            "top": belt[0][:6],
            "mem": mem, "mem_scale": ms, "stamp": st, "stamp_scale": ss,
        }


def make_handler(brain):
    Base = engram_chat.make_handler(brain)

    class H(Base):
        def do_GET(self):
            if self.path.split("?")[0].strip("/") == "":
                return self.send(200, (WEB / "chat.html").read_bytes(), "text/html; charset=utf-8")
            return super().do_GET()

    return H


def main():
    from http.server import ThreadingHTTPServer
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/engram_runs/tide/ckpt.pt")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"loading {a.ckpt} on {device} ...", flush=True)
    brain = TideBrain(Path(a.ckpt), device)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(brain))
    print(f"chat on http://{a.host}:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
