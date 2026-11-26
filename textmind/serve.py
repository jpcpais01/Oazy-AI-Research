"""Textmind: watch the sentence mind read and dream.

    python3 -m textmind.serve --port 8001 [--model /workspace/oazy_runs/textmind/sent/model.pt]

You type the start of a story. The mind reads it one sentence at a time, exactly as in
training, and the page shows for every sentence:
  - what it expected before reading it (a guess drawn from memory alone, the prior, said by the mouth)
  - how surprised it was when it read it (the KL between guess and correction, in bits)
  - what it kept of it (the corrected inner state, said back by the mouth)
Then it dreams on: memory steps forward with guessed moments only (no reading), and the mouth
says each dreamed moment. Several dreams side by side, from the same memory.

The model file is reloaded when it changes on disk (a training run that finishes shows up on reload).
"""
from __future__ import annotations

import argparse
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch
import torch.nn.functional as F

from .data import EOS, PAD, SPLIT, VOCAB
from .model import H_DIM, Z_DIM, SentenceMind, dist, kl_cat, sample

PAGE = Path(__file__).parent / "page.html"
LN2 = math.log(2)
MAX_SENTS = 16


def to_bytes(sents, L):
    """Sentences -> (S, L+1) long tensor like data.stories_to_tensor (too long ones are cut)."""
    x = torch.full((len(sents), L + 1), PAD, dtype=torch.long)
    cut = []
    for i, s in enumerate(sents):
        b = list(s.encode())
        cut.append(len(b) > L)
        b = b[:L]
        x[i, :len(b)] = torch.tensor(b, dtype=torch.long)
        x[i, len(b)] = EOS
    return x, cut


def to_text(row):
    b = []
    for v in row.tolist():
        if v in (EOS, PAD):
            break
        b.append(v)
    return bytes(b).decode("utf-8", errors="replace")


def split_sentences(text):
    return [s.strip() for s in SPLIT.split(" ".join(text.split())) if s.strip()][:MAX_SENTS]


class Mind:
    def __init__(self, path, dev):
        self.path, self.dev = Path(path), dev
        self.m, self.info, self.mtime = None, None, None
        self.lock = threading.Lock()

    def get(self):
        """The model, reloaded if the file changed. None when there is no trained model yet."""
        if not self.path.exists():
            return None
        mt = self.path.stat().st_mtime
        if mt != self.mtime:
            ck = torch.load(self.path, map_location="cpu")
            a = ck.get("args", {})
            m = SentenceMind(a.get("L", 96))
            m.load_state_dict(ck["model"])
            m.to(self.dev).eval()
            res = ck.get("result", {})
            self.m, self.mtime = m, mt
            self.info = dict(path=str(self.path), L=m.L, params=round(sum(p.numel() for p in m.parameters()) / 1e6, 1),
                             minutes=res.get("minutes"), steps=res.get("steps"), stories=res.get("stories"),
                             valid_bpb=res.get("valid_bpb"))
        return self.m

    @torch.no_grad()
    def think(self, text, n_more=4, samples=3, temp=0.8, seed=None):
        with self.lock:
            m = self.get()
            if m is None:
                return dict(error="no trained model yet", model=None)
            if seed is not None:
                torch.manual_seed(seed)
            sents = split_sentences(text)
            if not sents:
                return dict(error="type at least one sentence", model=self.info)
            x, cut = to_bytes(sents, m.L)
            x = x.to(self.dev)
            S = x.shape[0]
            e = m.read(x)                                                   # S, 512
            h = torch.zeros(1, H_DIM, device=self.dev)
            z = torch.zeros(1, Z_DIM, device=self.dev)
            hs, z_post, z_prior, kls = [], [], [], []
            for t in range(S):
                if t > 0:
                    h = m.gru(m.z_in(z).float(), h.float())
                p = dist(m.prior(h))
                q = dist(m.post(torch.cat([h, e[t:t + 1].to(h.dtype)], -1)))
                kls.append(kl_cat(q, p).item() / LN2)
                z_prior.append(sample(p))
                z = sample(q)
                z_post.append(z)
                hs.append(h)
            H = torch.cat(hs)
            f_post = torch.cat([H, torch.cat(z_post)], -1)                  # S, FEAT
            f_prior = torch.cat([H, torch.cat(z_prior)], -1)
            # the mouth's cost of each typed sentence from what it kept (bits per byte)
            lg = m.say_logits(f_post, x).float()
            nll = F.cross_entropy(lg.reshape(-1, VOCAB), x.reshape(-1), ignore_index=PAD, reduction="none").view(S, -1).sum(1)
            nb = (x != PAD).sum(1).clamp(min=1)
            mouth_bpb = (nll / nb / LN2).tolist()
            expected = [to_text(r) for r in m.speak(f_prior, temp)]
            kept = [to_text(r) for r in m.speak(f_post, temp)]
            # dreams: the same memory, several futures (prior only, no reading)
            hh, zz = h.expand(samples, -1).contiguous(), z.expand(samples, -1).contiguous()
            dreams = [[] for _ in range(samples)]
            for _ in range(n_more):
                hh = m.gru(m.z_in(zz).float(), hh.float())
                p = dist(m.prior(hh))
                zz = sample(p)
                # how sure it is about this moment: entropy of the guess, in bits (0 = certain)
                ent = (-(p * torch.log(p)).sum((-1, -2)) / LN2).tolist()
                said = m.speak(torch.cat([hh, zz], -1), temp)
                for i in range(samples):
                    dreams[i].append(dict(text=to_text(said[i]), open_bits=round(ent[i], 1)))
            rows = [dict(text=s, cut=c, expected=ex, kept=kp, surprise_bits=round(k, 2), mouth_bpb=round(mb, 3),
                         bytes=int(n))
                    for s, c, ex, kp, k, mb, n in zip(sents, cut, expected, kept, kls, mouth_bpb, nb.tolist())]
            total_bits = sum(r["surprise_bits"] + r["mouth_bpb"] * r["bytes"] for r in rows)
            return dict(model=self.info, sentences=rows, dreams=dreams,
                        story_bpb=round(total_bits / max(1, sum(r["bytes"] for r in rows)), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/oazy_runs/textmind/sent/model.pt")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    mind = Mind(a.model, dev)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def send(self, obj, ctype="application/json"):
            body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/":
                    self.send(PAGE.read_bytes(), "text/html; charset=utf-8")
                elif u.path == "/api/info":
                    mind.get()
                    self.send(dict(model=mind.info))
                elif u.path == "/api/think":
                    n = max(1, min(8, int(q.get("n", 4))))
                    k = max(1, min(4, int(q.get("samples", 3))))
                    temp = max(0.1, min(1.5, float(q.get("temp", 0.8))))
                    self.send(mind.think(q.get("text", "")[:4000], n, k, temp))
                else:
                    self.send_error(404)
            except Exception as e:  # noqa
                self.send(dict(error=str(e)))

    print(f"Textmind page on http://0.0.0.0:{a.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
