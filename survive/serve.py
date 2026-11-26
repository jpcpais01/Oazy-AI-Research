"""Creature: watch a saved mind live in a fresh world.

    python3 -m survive.serve --runs /workspace/oazy_runs/survive --port 8000

Shows only what the creature really sees (its two eyes, real pixels), a map from above (for us:
it never sees this), its body feelings and senses, what its inner state imagines 0.3-1.2 s ahead
(drawn by the detached mind's-eye decoder), its fear (imagined chance of dying within 1.2 s),
how it learned (training curves) and an About card. A background thread runs the world ahead of
the page and keeps a buffer of frames, so playback is smooth.
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch

from .model import load_mind, Mind, wellbeing
from .render import render, set_eyes
from .train import Actor
from .world import DEATH_CAUSES, L, World

PAGE = (Path(__file__).parent / "page.html")


def mind_name(path):
    p = Path(path)
    ck = p.stem.replace("mind_", "")
    return f"{p.parent.name} · {ck}"


class Sim:
    def __init__(self, runs, dev):
        self.runs, self.dev = runs, dev
        self.lock = threading.Lock()
        self.frames, self.seq = [], 0
        self.cfg = None
        self.want = None
        threading.Thread(target=self.loop, daemon=True).start()

    def minds(self):
        out = []
        for f in sorted(glob.glob(os.path.join(self.runs, "*", "mind_*min.pt")), key=os.path.getmtime):
            out.append(dict(path=f, name=mind_name(f)))
        return out

    def select(self, path, mode, seed):
        with self.lock:
            self.want = dict(path=path, mode=mode, seed=seed)

    def _load(self, w):
        ck = torch.load(w["path"], map_location=self.dev)
        res = ck["args"]["res"]
        set_eyes(ck["args"])
        m = Mind(res).to(self.dev)
        load_mind(m, ck["mind"])
        m.eval()
        world = World(1, self.dev, mode=w["mode"], seed=w["seed"])
        act = Actor(m, world, res, self.dev, deterministic=True)
        return m, world, act, res

    @torch.no_grad()
    def imagine(self, m, act, steps=12):
        h, z = act.h.clone(), act.z.clone()
        pics, cont_p, wb = [], 1.0, []
        for k in range(1, steps + 1):
            a, _ = m.act(torch.cat([h, z], -1), deterministic=True)
            h = m.step_h(h, z, a)
            z = m.sample(m.imagine_z(h))
            ft = torch.cat([h, z], -1)
            c = torch.sigmoid(m.cont_head(ft)).item()
            cont_p *= c
            wb.append(wellbeing(m.sense_head(ft)).item())
            if k % 3 == 0:
                pic = m.minds_eye(ft).clamp(0, 1)[0]
                pics.append(base64.b64encode((pic * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy().tobytes()).decode())
        return pics, 1 - cont_p, wb

    def loop(self):
        m = world = act = None
        life, t_life, food = 0, 0, 0
        while True:
            with self.lock:
                w, self.want = self.want, None
                nbuf = len(self.frames)
            if w is not None:
                try:
                    m, world, act, res = self._load(w)
                    self.res = res
                    with self.lock:
                        self.frames, self.cfg = [], w
                        self.seq += 1000
                    life, t_life, food = 1, 0, 0
                except Exception as e:  # noqa
                    print("load failed", e, flush=True)
                continue
            if m is None or nbuf >= 120:
                time.sleep(0.05)
                continue
            a, rec, _ = act.observe_act(bf16=self.dev.type == "cuda")
            img = rec[0][0]
            fr = dict(seq=self.seq, life=life, t=t_life, food=food)
            fr["eyes"] = base64.b64encode(img.permute(1, 2, 0).contiguous().cpu().numpy().tobytes()).decode()
            fr["res"] = img.shape[-1]
            if t_life % 3 == 0:
                fr["dream"], fr["fear"], fr["wb"] = self.imagine(m, act)
            fr["state"] = self.state(world)
            fr["act"] = [round(x, 2) for x in a[0].tolist()]
            act.advance(a)
            done, trunc, cause, ate = world.step(a)
            t_life += 1
            if ate[0] >= 0:
                food += 1
            if done[0] or trunc[0]:
                fr["end"] = "lived 100 s" if trunc[0] else f"died: {DEATH_CAUSES[int(cause[0])]}"
                world.reset(torch.arange(1, device=self.dev))
                act.first[:] = True
                life, t_life, food = life + 1, 0, 0
            with self.lock:
                self.seq += 1
                self.frames.append(fr)

    def state(self, w):
        r = lambda x: [round(v, 2) for v in x.tolist()]
        s = w.senses[0]
        return dict(
            pos=r(w.pos[0]), head=round(w.head[0].item(), 3), L=L,
            boxes=[r(torch.cat([w.bmin[0, i], w.bmax[0, i], w.bcol[0, i]])) + [int(w.bmov[0, i]), int(w.held[0] == i)]
                   for i in range(w.bmin.shape[1]) if w.bact[0, i]],
            lava=[r(torch.cat([w.lmin[0, i], w.lmax[0, i]])) for i in range(w.lmin.shape[1]) if w.lact[0, i]],
            food=[r(w.fpos[0, i]) + [int(w.fkind[0, i])] for i in range(w.fpos.shape[1]) if w.fup[0, i]],
            pred=r(w.pred[0]), wand=r(w.wand[0]), strange=r(w.strange[0]),
            energy=round(w.energy[0].clamp(0, 1).item(), 3), health=round(w.health[0].clamp(0, 1).item(), 3),
            body=r(s[:5]), intensity=r(s[5:21]), ears=r(s[21:29]))

    def take(self, since):
        with self.lock:
            fr = [f for f in self.frames if f["seq"] > since]
            # drop what the page has already taken
            self.frames = [f for f in self.frames if f["seq"] > since]
            return fr[:30]


def about(path):
    ck = torch.load(path, map_location="cpu")
    a = ck["args"]
    d = Path(path)
    out = dict(name=mind_name(path), path=path, minutes=round(ck.get("minutes", 0), 1), frames=ck.get("frames"),
               updates=ck.get("updates"), res=a["res"], gamma=a["gamma"], horizon=a["horizon"],
               params=round(sum(v.numel() for v in ck["mind"].values()) / 1e6, 1))
    b = d.with_suffix(".bench.json")
    if b.exists():
        out["bench"] = json.loads(b.read_text())
    return out


def curves(path):
    log = Path(path).parent / "log.jsonl"
    rows = []
    if log.exists():
        for line in log.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:  # noqa
                pass
    mins = torch.load(path, map_location="cpu").get("minutes", 1e9)
    return [r for r in rows if r.get("min", 0) <= mins + 0.01]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/workspace/oazy_runs/survive")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    sim = Sim(a.runs, dev)
    ms = sim.minds()
    if ms:
        sim.select(ms[-1]["path"], "test", 1)

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
                elif u.path == "/api/minds":
                    self.send(dict(minds=sim.minds(), cfg=sim.cfg))
                elif u.path == "/api/frames":
                    self.send(dict(frames=sim.take(int(q.get("since", -1))), cfg=sim.cfg))
                elif u.path == "/api/select":
                    sim.select(q["mind"], q.get("mode", "test"), int(q.get("seed", int(time.time()) % 100000)))
                    self.send(dict(ok=True))
                elif u.path == "/api/about":
                    self.send(about(q["mind"]))
                elif u.path == "/api/curves":
                    self.send(dict(rows=curves(q["mind"])))
                else:
                    self.send_error(404)
            except Exception as e:  # noqa
                self.send(dict(error=str(e)))

    print(f"Creature page on http://0.0.0.0:{a.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
