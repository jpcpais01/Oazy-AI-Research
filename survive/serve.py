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
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch

from .model import gdist, gsample, load_mind, make_mind, wellbeing
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
            d = dict(path=f, name=mind_name(f))
            b = Path(f).with_suffix(".bench.json")       # survival in new worlds (test bench), seconds alive
            if b.exists():
                try:
                    t = json.loads(b.read_text()).get("test")
                    if t and t.get("lives", 0) >= 32:              # a 1-life smoke bench is not a score
                        d["score_s"] = round(t["steps"] / 10)
                        d["score_lives"] = t.get("lives")
                except Exception:  # noqa
                    pass
            if "score_s" not in d:                            # else the 64-life quick test logged at that minute
                q = self._quick(Path(f))
                if q is not None:
                    d["score_s"], d["score_lives"] = round(q / 10), 64
            out.append(d)
        return out

    _qcache = {}

    def _quick(self, f):
        log = f.parent / "log.jsonl"
        try:
            mins = float(f.stem.replace("mind_", "").replace("min", ""))
        except ValueError:
            return None
        if not log.exists():
            return None
        key = (str(log), log.stat().st_mtime)
        if key not in self._qcache:
            rows = {}
            for line in log.read_text().splitlines():
                if '"test_steps"' in line:
                    try:
                        r = json.loads(line); rows[round(r["min"])] = r["test_steps"]
                    except Exception:  # noqa
                        pass
            self._qcache[key] = rows
        return self._qcache[key].get(round(mins))

    def select(self, path, mode, seed, det=True):
        with self.lock:
            self.want = dict(path=path, mode=mode, seed=seed, det=bool(det))

    def _load(self, w):
        ck = torch.load(w["path"], map_location=self.dev)
        res = ck["args"]["res"]
        set_eyes(ck["args"])
        m = make_mind(ck["args"]).to(self.dev)
        load_mind(m, ck["mind"])
        m.eval()
        world = World(1, self.dev, mode=w["mode"], seed=w["seed"])
        act = Actor(m, world, res, self.dev, deterministic=w.get("det", True))   # habit, or with its training wobble
        self.motion = self._motion_probe(m, res, w["path"])
        return m, world, act, res

    _probes = {}

    @torch.no_grad()
    def _motion_probe(self, m, res, path, n=64, steps=300):
        """How does the body move, read from the inner state? A ridge map from (memory h, moment z, chosen
        muscles a) to the body-frame motion of the next step (forward m, sideways m, turn rad), fitted on
        real lives with this mind and scored on held-out lives. Used to turn each dream into a path."""
        if path in self._probes:
            return self._probes[path]
        w = World(n, self.dev, mode="train", seed=777)
        act = Actor(m, w, res, self.dev, deterministic=True)
        X, Y = [], []
        for _ in range(steps):
            a, _, _ = act.observe_act(bf16=self.dev.type == "cuda")
            x = torch.cat([act.h.float(), act.z.float().reshape(n, -1), a.float()], -1)
            p0, h0 = w.pos[:, :2].clone(), w.head.clone()
            act.advance(a)
            done, trunc, *_ = w.step(a)
            dp = w.pos[:, :2] - p0
            c, s_ = torch.cos(h0), torch.sin(h0)
            fwd, side = c * dp[:, 0] + s_ * dp[:, 1], -s_ * dp[:, 0] + c * dp[:, 1]
            dh = torch.remainder(w.head - h0 + math.pi, 2 * math.pi) - math.pi
            ok = ~(done | trunc) & (w.dying == 0)
            X.append(x[ok]); Y.append(torch.stack([fwd, side, dh], -1)[ok])
            if (done | trunc).any():
                w.reset(torch.nonzero(done | trunc).squeeze(-1)); act.first[done | trunc] = True
        X, Y = torch.cat(X), torch.cat(Y)
        perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(0)).to(X.device)
        X, Y = X[perm], Y[perm]
        k = len(X) * 3 // 4
        mu, sd = X[:k].mean(0), X[:k].std(0) + 1e-3
        A = torch.cat([(X - mu) / sd, torch.ones(len(X), 1, device=X.device)], 1)
        Wt = torch.linalg.solve(A[:k].T @ A[:k] + 30.0 * torch.eye(A.shape[1], device=X.device), A[:k].T @ Y[:k])
        P = A[k:] @ Wt
        r2 = (1 - ((P - Y[k:]) ** 2).mean(0) / Y[k:].var(0).clamp(min=1e-9)).tolist()
        pr = dict(mu=mu, sd=sd, W=Wt, r2=[round(v, 3) for v in r2])
        print("motion probe", path, "R2 fwd/side/turn", pr["r2"], flush=True)
        self._probes[path] = pr
        return pr

    def _dream_step(self, m):
        """The dream step (act, remember, imagine the next moment, its feelings), compiled once per mind."""
        if getattr(self, "_step_for", None) is not m:
            def f(h, z):
                a, _ = m.act(torch.cat([h, z], -1), deterministic=True)
                h2 = m.step_h(h, z, a)
                z2 = m.sample(m.imagine_z(h2))
                ft = torch.cat([h2, z2], -1)
                return a, h2, z2, torch.sigmoid(m.cont_head(ft)).reshape(-1), wellbeing(m.sense_head(ft)).reshape(-1)
            try:
                self._step = torch.compile(f, mode="reduce-overhead", dynamic=False)
            except Exception:  # noqa
                self._step = f
            self._step_for = m
        return self._step

    DREAM_AT = (2, 5, 10, 20, 50, 100)          # imagined pictures at +0.2, 0.5, 1, 2, 5, 10 s

    @torch.no_grad()
    def imagine(self, m, act, steps=100, n=16):
        """n dreams from the same moment (the moment slots are sampled, so each dream differs). Pictures come
        from the first dream; expected wellbeing and the death chance are averaged over all n, so they show
        what it expects rather than one random dream."""
        h, z = act.h.expand(n, -1).clone(), act.z.expand(n, -1).clone()
        code, ctr = act.code.expand(n, -1).clone(), int(act.ctr[0])
        pics, conts, wbs, mot = [], [], [], []
        pr = getattr(self, "motion", None)
        step = None if m.intent else self._dream_step(m)
        for k in range(1, steps + 1):
            if step is not None:                    # one compiled, graph-captured dream step (far fewer kernel launches)
                a, h2, z2, ct, wbt = step(h, z)
                if pr is not None:
                    mot.append((h, z, a.clone()))
                h, z = h2.clone(), z2.clone()
                conts.append(ct.clone()); wbs.append(wbt.clone())
                if k in self.DREAM_AT:
                    pics.append(m.minds_eye(torch.cat([h[:1], z[:1]], -1)).clamp(0, 1)[0])
                continue
            x = torch.cat([h, z], -1)
            if m.intent:                            # the will keeps choosing inside the dream too
                if ctr >= m.intent:
                    code, ctr = gsample(gdist(m.will(x), m.g_cls), True).float(), 0
                ctr += 1
                x = torch.cat([x, code], -1)
            a, _ = m.act(x, deterministic=True)
            if pr is not None:                      # how this dreamed step would move the body (decoded after the loop)
                mot.append((h, z, a))
            h = m.step_h(h, z, a)
            z = m.sample(m.imagine_z(h))
            ft = torch.cat([h, z], -1)
            conts.append(torch.sigmoid(m.cont_head(ft)).reshape(n))
            wbs.append(wellbeing(m.sense_head(ft)).reshape(n))
            if k in self.DREAM_AT:
                pics.append(m.minds_eye(ft[:1]).clamp(0, 1)[0])         # copied to the CPU after the loop, all at once
        # everything the page needs is gathered on the GPU and copied to the CPU in ONE transfer: each copy
        # waits for the GPU, which is shared with training, so a dozen small copies made the viewer stall
        C = torch.cumprod(torch.stack(conts), 0)                         # steps, n: chance each dream is still alive
        alive = C.mean(1)
        W = torch.stack(wbs)                                             # steps, n
        q = torch.quantile(W.float(), torch.tensor([0.1, 0.9], device=W.device), dim=1)
        pack = [alive.float(), W.mean(1).float(), q.reshape(-1)]
        use_paths = bool(mot) and pr is not None and pr["r2"][0] > 0.5 and pr["r2"][2] > 0.5   # forward and turn carry the path; sideways slip is tiny
        if use_paths:
            xa = torch.cat([torch.stack([q_[0] for q_ in mot]), torch.stack([q_[1] for q_ in mot]).reshape(len(mot), n, -1),
                            torch.stack([q_[2] for q_ in mot])], -1).float()   # steps, n, features
            M = torch.cat([(xa - pr["mu"]) / pr["sd"], torch.ones(*xa.shape[:2], 1, device=xa.device)], -1) @ pr["W"]   # steps, n, (fwd, side, turn)
            pos = act.w.pos[0, :2].float().expand(n, -1); hd = act.w.head[0].float().expand(n)
            # integrate all steps at once: heading before each step, then rotate the body-frame steps into the world
            hds = hd[None] + torch.cat([torch.zeros_like(M[:1, :, 2]), M[:-1, :, 2]], 0).cumsum(0)      # steps, n
            c, s_ = torch.cos(hds), torch.sin(hds)
            d = torch.stack([c * M[..., 0] - s_ * M[..., 1], s_ * M[..., 0] + c * M[..., 1]], -1)       # steps, n, 2
            P = torch.cat([pos[None], pos[None] + d.cumsum(0)], 0).clamp(0.25, L - 0.25).transpose(0, 1)[:, ::2]   # n, 51 (every 0.2 s)
            die = (C < 0.5).float().argmax(0) + (C.min(0).values >= 0.5).long() * steps
            pack += [P.reshape(-1), W.mean(0).float(), die.float()]
        flat = torch.cat(pack).cpu()
        pic_bytes = (torch.stack(pics) * 255).byte().permute(0, 2, 3, 1).contiguous().cpu().numpy() if pics else []
        pics = [base64.b64encode(b.tobytes()).decode() for b in pic_bytes]
        k = 0
        def take(cnt):
            nonlocal k
            out = flat[k:k + cnt]; k += cnt
            return out
        alive = take(steps); wbm = take(steps); qq = take(2 * steps).reshape(2, steps)
        self.wb_band = [[round(v, 3) for v in qq[0].tolist()], [round(v, 3) for v in qq[1].tolist()]]
        wb = [round(v, 3) for v in wbm.tolist()]
        self.paths = None
        if use_paths:
            Pc = take(n * P.shape[1] * 2).reshape(n, -1, 2); wbd = take(n); dd = take(n)
            self.paths = dict(pts=[[[round(v, 2) for v in q_] for q_ in Pc[i].tolist()] for i in range(n)],
                              wb=[round(v, 3) for v in wbd.tolist()], die=[int(v) if v < steps else None for v in dd.tolist()], r2=pr["r2"])
            # honesty check: where the dreams put it at +1.6 s and +4 s vs where it really is then (live, this life)
            self._pending.append((self._t, Pc[:, 8].clone(), Pc[:, 20].clone()))
            self.paths["err"] = {k_: (round(sorted(v)[len(v) // 2], 2) if v else None) for k_, v in self._errs.items()}
        return pics, 1 - alive[11].item(), wb, 1 - alive[-1].item()

    def _check_paths(self, world, t_life):
        """Score past dream paths against the real position now (median over the last 300 checks)."""
        p = world.pos[0, :2].cpu()
        keep = []
        for (t0, p16, p40) in self._pending:
            for dt, pp, key in ((16, p16, "mean_1.6s"), (40, p40, "mean_4s")):
                if t_life - t0 == dt:
                    self._errs[key].append((pp.mean(0) - p).norm().item())
                    self._errs[key.replace("mean", "best")].append((pp - p).norm(dim=-1).min().item())
                    self._errs[key] = self._errs[key][-300:]; self._errs[key.replace("mean", "best")] = self._errs[key.replace("mean", "best")][-300:]
            if t_life - t0 < 40:
                keep.append((t0, p16, p40))
        self._pending = keep

    def loop(self):
        m = world = act = None
        self._pending, self._t = [], 0
        self._errs = {"mean_1.6s": [], "best_1.6s": [], "mean_4s": [], "best_4s": []}
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
                    self._pending = []
                    self._errs = {k: [] for k in self._errs}
                except Exception as e:  # noqa
                    print("load failed", e, flush=True)
                continue
            if m is None or nbuf >= 120:
                time.sleep(0.05)
                continue
            self._t = t_life
            self._check_paths(world, t_life)
            a, rec, _ = act.observe_act(bf16=self.dev.type == "cuda")
            img = rec[0][0]
            fr = dict(seq=self.seq, life=life, t=t_life, food=food)
            fr["eyes"] = base64.b64encode(img.permute(1, 2, 0).contiguous().cpu().numpy().tobytes()).decode()
            fr["res"] = img.shape[-1]
            if t_life % 3 == 0:
                fr["dream"], fr["fear"], fr["wb"], fr["fear10"] = self.imagine(m, act)
                fr["wb_band"] = self.wb_band
                if self.paths is not None:
                    fr["paths"] = self.paths
            fr["state"] = self.state(world)
            fr["act"] = [round(x, 2) for x in a[0].tolist()]
            if m.intent:                                   # the will's current intention (the slow level)
                code = act.code[:1]
                fr["intent"] = dict(slots=code.view(-1, m.g_cls).argmax(-1).tolist(), age=int(act.ctr[0]), every=int(m.intent))
                if t_life % 3 == 0:                        # the wanted memory, drawn by the mind's eye
                    with torch.no_grad():
                        g = m.goal_dec(code)
                        ft = torch.cat([g, m.sample(m.imagine_z(g))], -1)
                        pic = m.minds_eye(ft).clamp(0, 1)[0]
                    fr["intent"]["pic"] = base64.b64encode((pic * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy().tobytes()).decode()
            act.advance(a)
            done, trunc, cause, ate = world.step(a)
            t_life += 1
            if ate[0] >= 0:
                food += 1
            if done[0] or trunc[0]:
                fr["end"] = "lived 100 s" if trunc[0] else f"died: {DEATH_CAUSES[int(cause[0])]}"
                world.reset(torch.arange(1, device=self.dev))
                act.first[:] = True
                self._pending = []
                life, t_life, food = life + 1, 0, 0
            with self.lock:
                self.seq += 1
                self.frames.append(fr)

    def state(self, w):
        """The world for the page, copied to the CPU in one go (one GPU sync instead of a hundred)."""
        g = lambda x: x[0].detach().float().reshape(-1)
        parts = dict(pos=g(w.pos), head=g(w.head), gaze=g(torch.as_tensor(w.gaze, device=w.pos.device).reshape(1, -1)),
                     bmin=g(w.bmin), bmax=g(w.bmax), bcol=g(w.bcol), bmov=g(w.bmov), bact=g(w.bact), held=g(w.held),
                     lmin=g(w.lmin), lmax=g(w.lmax), lact=g(w.lact), fpos=g(w.fpos), fkind=g(w.fkind), fup=g(w.fup),
                     pred=g(w.pred), wand=g(w.wand), strange=g(w.strange), dying=g(w.dying), energy=g(w.energy), health=g(w.health), sen=g(w.senses))
        flat = torch.cat(list(parts.values())).cpu().tolist()
        c, k = {}, 0
        for name, t in parts.items():
            c[name] = flat[k:k + t.numel()]; k += t.numel()
        r = lambda xs: [round(v, 2) for v in xs]
        NB, NL, NF = w.bmin.shape[1], w.lmin.shape[1], w.fpos.shape[1]
        bmin, bmax, bcol = (c[n] for n in ("bmin", "bmax", "bcol"))
        sen = c["sen"]
        return dict(
            pos=r(c["pos"]), head=round(c["head"][0], 3), gaze=round(c["gaze"][0], 3), L=L,
            boxes=[r(bmin[3 * i:3 * i + 3] + bmax[3 * i:3 * i + 3] + bcol[3 * i:3 * i + 3]) + [int(c["bmov"][i]), int(c["held"][0] == i)]
                   for i in range(NB) if c["bact"][i]],
            lava=[r(c["lmin"][2 * i:2 * i + 2] + c["lmax"][2 * i:2 * i + 2]) for i in range(NL) if c["lact"][i]],
            food=[r(c["fpos"][3 * i:3 * i + 3]) + [int(c["fkind"][i])] for i in range(NF) if c["fup"][i]],
            pred=r(c["pred"]), wand=r(c["wand"]), strange=r(c["strange"]),
            dying=int(c["dying"][0]), energy=round(min(max(c["energy"][0], 0), 1), 3), health=round(min(max(c["health"][0], 0), 1), 3),
            body=r(sen[:5]), intensity=r(sen[5:21]), ears=r(sen[21:29]), stamina=round(sen[29], 3), hunger=round(sen[30], 3))

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
               params=round(sum(v.numel() for v in ck["mind"].values()) / 1e6, 1), drive=a.get("drive", "linear"))
    # how this mind was built (what the About card describes); missing keys = the old default
    keys = dict(recon=0, smell=0, stamina=0, metab=1, world_v=1, jumps="", jump_w=0, init="", gaze=0, fovea=0, eye_sep=0.08,
                life_cap=100, light=0, matter=0, intent=0, heat=1, back_speed=1, ent_per=0, long_fresh=0, seed=0)
    out["cfg"] = {k: a.get(k, v) for k, v in keys.items()}
    b = d.with_suffix(".bench.json")
    if b.exists():
        bb = {k: v for k, v in json.loads(b.read_text()).items() if isinstance(v, dict) and v.get("lives", 0) >= 32}
        if bb:
            out["bench"] = bb
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
                    sim.select(q["mind"], q.get("mode", "test"), int(q.get("seed", int(time.time()) % 100000)),
                               det=q.get("act", "habit") != "wobble")
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
