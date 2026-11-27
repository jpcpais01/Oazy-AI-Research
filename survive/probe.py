"""Does the mind KNOW where food / the predator are? Linear probes on its inner state.
Runs lives with the mind, collects the inner state (h, z) and the eye's embedding each step, and the
true egocentric direction/distance of the nearest food and of the predator. Ridge regression on half
the lives, scored on the other half. High score = it knows; low = it doesn't see/remember it.
Also: AUC of "some food within 3 m and 30 deg of where the eyes look" (0.5 = blind to it).
   python3 -m survive.probe <mind.pt> [--n 192]     (writes <mind>.probe.json)
"""
import math, argparse, json
import torch
from .model import make_mind, load_mind
from .render import set_eyes
from .train import Actor
from . import world as W

ap = argparse.ArgumentParser(); ap.add_argument("mind"); ap.add_argument("--n", type=int, default=192)
ap.add_argument("--steps", type=int, default=400); ap.add_argument("--mode", default="test")
a = ap.parse_args()
dev = torch.device("cuda")
ck = torch.load(a.mind, map_location=dev); args = ck["args"]; set_eyes(args)
m = make_mind(args).to(dev); load_mind(m, ck["mind"]); m.eval()
w = W.World(a.n, dev, mode=a.mode, seed=999)
act = Actor(m, w, args["res"], dev, deterministic=True)
H, Z, Y, LIVE = [], [], [], []
PY, PM = [], []          # world 5: the nearest live prey and the nearest carcass (direction, distance, in view)
alive = torch.ones(a.n, dtype=torch.bool, device=dev)


def ego(v, head):
    c, s = torch.cos(-head), torch.sin(-head)
    return torch.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


with torch.no_grad():
    for t in range(a.steps):
        x, _, _ = act.observe_act(True); act.advance(x)
        p = w.pos[:, :2]
        fv = w.fpos[..., :2] - p[:, None]
        fd = fv.norm(dim=-1); fd = torch.where(w.fup, fd, torch.full_like(fd, 99.0))
        k = fd.argmin(-1)
        nf = fv[torch.arange(a.n, device=dev), k]; nd = nf.norm(dim=-1)
        fe = ego(nf, w.head) / nd.clamp(min=1e-3)[:, None]
        pv = w.pred - p; pdist = pv.norm(dim=-1); pe = ego(pv, w.head) / pdist.clamp(min=1e-3)[:, None]
        # targets: food dir (cos, sin), food dist (log), food ahead within 45deg, pred dir, pred dist(log), energy
        ang = torch.atan2(*(ego(fv, (w.head + w.gaze)[:, None].expand_as(fd)).unbind(-1))[::-1])
        vis = (w.fup & (fd < 3.0) & (ang.abs() < 0.5)).any(-1).float()       # some food within 3 m and 30 deg of straight ahead
        y = torch.stack([fe[:, 0], fe[:, 1], torch.log(nd.clamp(0.1, 15)), (fe[:, 0] > 0.707).float(),
                         pe[:, 0], pe[:, 1], torch.log(pdist.clamp(0.1, 15)), w.energy, vis], -1)
        H.append(act.h.float().clone()); Z.append(act.z.float().reshape(a.n, -1).clone()); Y.append(y); LIVE.append(alive & (w.dying == 0))
        look = (w.head + w.gaze)
        cols, msk = [], []
        for has in (w.prey_alive, w.prey_dead):
            v = w.prey - p[:, None]; d = torch.where(has, v.norm(dim=-1), torch.full_like(v[..., 0], 99.0))
            j = d.argmin(-1); nv = v[torch.arange(a.n, device=dev), j]; ndist = nv.norm(dim=-1)
            e = ego(nv, w.head) / ndist.clamp(min=1e-3)[:, None]
            an = torch.atan2(*(ego(v, look[:, None].expand_as(d)).unbind(-1))[::-1])
            inv = (has & (d < 3.0) & (an.abs() < 0.7)).any(-1).float()           # within 3 m and 40 deg of where the eyes look
            cols += [e[:, 0], e[:, 1], torch.log(ndist.clamp(0.1, 15)), inv]; msk.append(has.any(-1))
        PY.append(torch.stack(cols, -1)); PM.append(torch.stack(msk, -1))
        done, trunc, *_ = w.step(x); alive &= ~(done | trunc)
        if not alive.any(): break
H, Z, Y, L = (torch.stack(v, 1) for v in (H, Z, Y, LIVE))   # n,T,...
names = ["food_cos", "food_sin", "food_logdist", "food_ahead", "pred_cos", "pred_sin", "pred_logdist", "energy", "food_in_view_3m"]


def fit(X, lam=10.0):
    tr = torch.zeros(a.n, dtype=torch.bool, device=dev); tr[: a.n // 2] = True
    out = {}
    Xtr, Ytr = X[tr][L[tr]], Y[tr][L[tr]]; Xte, Yte = X[~tr][L[~tr]], Y[~tr][L[~tr]]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-3
    Xtr = torch.cat([(Xtr - mu) / sd, torch.ones(len(Xtr), 1, device=dev)], 1); Xte = torch.cat([(Xte - mu) / sd, torch.ones(len(Xte), 1, device=dev)], 1)
    Wt = torch.linalg.solve(Xtr.T @ Xtr + lam * torch.eye(Xtr.shape[1], device=dev), Xtr.T @ Ytr)
    P = Xte @ Wt
    r2 = 1 - ((P - Yte) ** 2).mean(0) / Yte.var(0).clamp(min=1e-6)
    for i, nm in enumerate(names): out[nm] = round(r2[i].item(), 3)
    # angle error of the food direction (degrees), and of the predator
    fa = torch.atan2(P[:, 1], P[:, 0]) - torch.atan2(Yte[:, 1], Yte[:, 0])
    out["food_angle_err_deg"] = round(torch.rad2deg(torch.remainder(fa + math.pi, 2 * math.pi) - math.pi).abs().median().item(), 1)
    pa = torch.atan2(P[:, 5], P[:, 4]) - torch.atan2(Yte[:, 5], Yte[:, 4])
    out["pred_angle_err_deg"] = round(torch.rad2deg(torch.remainder(pa + math.pi, 2 * math.pi) - math.pi).abs().median().item(), 1)
    out["samples"] = int(len(Xte))
    return out


def mlp(X):
    tr = torch.zeros(a.n, dtype=torch.bool, device=dev); tr[: a.n // 2] = True
    Xtr, Ytr = X[tr][L[tr]], Y[tr][L[tr]]; Xte, Yte = X[~tr][L[~tr]], Y[~tr][L[~tr]]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-3; ym, ys = Ytr.mean(0), Ytr.std(0) + 1e-3
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[-1], 512), torch.nn.SiLU(), torch.nn.Linear(512, 512), torch.nn.SiLU(), torch.nn.Linear(512, Y.shape[-1])).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    Xn, Yn = (Xtr - mu) / sd, (Ytr - ym) / ys
    for it in range(3000):
        i = torch.randint(0, len(Xn), (1024,), device=dev)
        loss = ((net(Xn[i]) - Yn[i]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): P = net((Xte - mu) / sd) * ys + ym
    r2 = 1 - ((P - Yte) ** 2).mean(0) / Yte.var(0).clamp(min=1e-6)
    out = {nm: round(r2[i].item(), 3) for i, nm in enumerate(names)}
    fa = torch.atan2(P[:, 1], P[:, 0]) - torch.atan2(Yte[:, 1], Yte[:, 0])
    out["food_angle_err_deg"] = round(torch.rad2deg(torch.remainder(fa + math.pi, 2 * math.pi) - math.pi).abs().median().item(), 1)
    return out


res = {"h (memory)": fit(H), "z (now)": fit(Z), "h+z": fit(torch.cat([H, Z], -1)), "MLP h+z": mlp(torch.cat([H, Z], -1))}

def auc(X):
    tr = torch.zeros(a.n, dtype=torch.bool, device=dev); tr[: a.n // 2] = True
    Xtr, ytr = X[tr][L[tr]], Y[tr][L[tr]][:, -1]; Xte, yte = X[~tr][L[~tr]], Y[~tr][L[~tr]][:, -1]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-3
    lin = torch.nn.Linear(X.shape[-1], 1).to(dev); opt = torch.optim.Adam(lin.parameters(), 1e-3)
    pw = ((1 - ytr).sum() / ytr.sum().clamp(min=1))
    for it in range(2000):
        i = torch.randint(0, len(Xtr), (2048,), device=dev)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lin((Xtr[i] - mu) / sd).squeeze(-1), ytr[i], pos_weight=pw)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): s_ = lin((Xte - mu) / sd).squeeze(-1)
    pos, neg = s_[yte > 0.5], s_[yte < 0.5]
    j = torch.randint(0, len(pos), (200000,), device=dev); k = torch.randint(0, len(neg), (200000,), device=dev)
    return dict(auc=round((pos[j] > neg[k]).float().mean().item(), 3), pos_share=round(yte.mean().item(), 3))
res["food_in_view_auc_h+z"] = auc(torch.cat([H, Z], -1)); res["food_in_view_auc_z"] = auc(Z)
PYt, PMt = torch.stack(PY, 1), torch.stack(PM, 1)
if PMt[..., 0].any():
    X = torch.cat([H, Z], -1)
    tr = torch.zeros(a.n, dtype=torch.bool, device=dev); tr[: a.n // 2] = True
    for gi, nm in enumerate(["prey", "carcass"]):
        ok = L & PMt[..., gi]
        if ok[tr].sum() < 200 or ok[~tr].sum() < 200:
            res[nm] = "too few moments"; continue
        Yg = PYt[..., 4 * gi:4 * gi + 4]
        Xtr, Ytr, Xte, Yte = X[tr][ok[tr]], Yg[tr][ok[tr]], X[~tr][ok[~tr]], Yg[~tr][ok[~tr]]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-3
        A = torch.cat([(Xtr - mu) / sd, torch.ones(len(Xtr), 1, device=dev)], 1); B = torch.cat([(Xte - mu) / sd, torch.ones(len(Xte), 1, device=dev)], 1)
        Wt = torch.linalg.solve(A.T @ A + 10 * torch.eye(A.shape[1], device=dev), A.T @ Ytr); P = B @ Wt
        r2 = 1 - ((P - Yte) ** 2).mean(0) / Yte.var(0).clamp(min=1e-6)
        err = torch.rad2deg(torch.remainder(torch.atan2(P[:, 1], P[:, 0]) - torch.atan2(Yte[:, 1], Yte[:, 0]) + math.pi, 2 * math.pi) - math.pi).abs().median()
        yv, sv = Yte[:, 3], P[:, 3]; pos, neg = sv[yv > .5], sv[yv < .5]
        au = None
        if len(pos) > 20 and len(neg) > 20:
            jj = torch.randint(0, len(pos), (200000,), device=dev); kk = torch.randint(0, len(neg), (200000,), device=dev)
            au = round((pos[jj] > neg[kk]).float().mean().item(), 3)
        res[nm] = dict(logdist_r2=round(r2[2].item(), 3), angle_err_deg=round(err.item(), 1), in_view_auc=au,
                       in_view_share=round(yv.mean().item(), 3), samples=int(len(Yte)))
print(json.dumps(res, indent=1))
r = res["h+z"]
print(f"probe: food dir angle err {r['food_angle_err_deg']} deg (chance 90), food logdist R2 {r['food_logdist']}, "
      f"food-in-view AUC {res['food_in_view_auc_h+z']['auc']} (0.5 blind) | predator logdist R2 {r['pred_logdist']}, "
      f"angle err {r['pred_angle_err_deg']} | energy R2 {r['energy']}"
      + "".join(f" | {k}: logdist R2 {res[k]['logdist_r2']}, angle err {res[k]['angle_err_deg']}, in-view AUC {res[k]['in_view_auc']} (in view {res[k]['in_view_share']})"
                for k in ("prey", "carcass") if isinstance(res.get(k), dict)))
json.dump(res, open(a.mind.replace(".pt", ".probe.json"), "w"), indent=1)
