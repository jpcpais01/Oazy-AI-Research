"""Predator encounters, one by one: how do chases start and end?
Runs N test lives and cuts every chase out of them. A chase starts when the predator is hunting and
comes within 3 m, and ends when it bites (outcome 'bite'), when it gives up / falls back beyond 4 m
('escape'), or when the life ends ('death' if the predator killed it). Per chase: start distance,
the creature's stamina and health at the start, its speed, how hard it pushes, how often it jumps,
whether it faces away, and the predator's own stamina.
   python3 -m survive.chases <mind.pt> [--lives 128] [--mode test]     (writes <mind>.chases.json)
"""
import math, argparse, json
import torch
from .model import make_mind, load_mind
from .render import set_eyes
from .train import Actor
from . import world as W

ap = argparse.ArgumentParser(); ap.add_argument("mind"); ap.add_argument("--lives", type=int, default=128)
ap.add_argument("--mode", default="test"); ap.add_argument("--max-steps", type=int, default=3000)
a = ap.parse_args()
dev = torch.device("cuda")
ck = torch.load(a.mind, map_location=dev); args = ck["args"]; set_eyes(args)
m = make_mind(args).to(dev); load_mind(m, ck["mind"]); m.eval()
n = a.lives
w = W.World(n, dev, mode=a.mode, seed=4321)
act = Actor(m, w, args["res"], dev, deterministic=True)
alive = torch.ones(n, dtype=torch.bool, device=dev)
in_chase = torch.zeros(n, dtype=torch.bool, device=dev)
acc = {k: torch.zeros(n, device=dev) for k in ["t", "d0", "st0", "hp0", "pst0", "speed", "push", "jump", "away", "behind", "stam_min"]}
chases = []
dash = dict(sprints=0, bites=0, escapes=0)              # world 5: the dasher's sprints that start within 4 m
d_on = torch.zeros(n, dtype=torch.bool, device=dev); d_bit = torch.zeros(n, dtype=torch.bool, device=dev)


def close(mask, outcome):
    idx = torch.nonzero(mask).squeeze(-1).tolist()
    for i in idx:
        t = acc["t"][i].item()
        if t < 1: continue
        chases.append(dict(outcome=outcome, steps=t, d0=acc["d0"][i].item(), stam0=acc["st0"][i].item(), hp0=acc["hp0"][i].item(),
                           pred_stam0=acc["pst0"][i].item(), speed=acc["speed"][i].item() / t, push=acc["push"][i].item() / t,
                           jump=acc["jump"][i].item() / t, away=acc["away"][i].item() / t, behind=acc["behind"][i].item() / t,
                           stam_min=acc["stam_min"][i].item()))


with torch.no_grad():
    for step in range(min(a.max_steps, W.MAX_STEPS * 10) + W.LIGHT):
        x, _, _ = act.observe_act(True); act.advance(x)
        live = alive & (w.dying == 0)
        p = w.pos[:, :2]; d = (w.pred - p).norm(dim=-1)
        hunting = w.pred_mem_t < W.PRED_FORGET
        start = live & ~in_chase & hunting & (d < 3.0)
        for k, v in [("t", 0.0), ("speed", 0.0), ("push", 0.0), ("jump", 0.0), ("away", 0.0), ("behind", 0.0)]:
            acc[k] = torch.where(start, torch.zeros_like(acc[k]), acc[k])
        acc["d0"] = torch.where(start, d, acc["d0"]); acc["st0"] = torch.where(start, w.stamina, acc["st0"])
        acc["hp0"] = torch.where(start, w.health, acc["hp0"]); acc["pst0"] = torch.where(start, w.pred_stam, acc["pst0"])
        acc["stam_min"] = torch.where(start, w.stamina, acc["stam_min"])
        in_chase |= start
        h0 = w.health.clone()
        don0 = w.dash_on.clone()
        done, trunc, cause, _ = w.step(x)
        dd = (w.dash - w.pos[:, :2]).norm(dim=-1)
        newd = live & ~d_on & w.dash_on & (dd < 4.0)
        dash["sprints"] += int(newd.sum()); d_on |= newd
        bitd = d_on & don0 & (dd < 0.8) & (w.health < h0 - 0.2)
        dash["bites"] += int(bitd.sum()); d_bit |= bitd
        endd = d_on & ~w.dash_on
        dash["escapes"] += int((endd & ~d_bit).sum()); d_on &= ~endd; d_bit &= ~endd
        v = (w.pos[:, :2] - p) / W.DT
        rel = w.pred - w.pos[:, :2]; dn = rel.norm(dim=-1).clamp(min=1e-3)
        ang = torch.remainder(torch.atan2(rel[:, 1], rel[:, 0]) - w.head + math.pi, 2 * math.pi) - math.pi
        c = in_chase & live
        cf = c.float()
        acc["t"] += cf; acc["speed"] += cf * v.norm(dim=-1); acc["push"] += cf * x[:, 0].float()
        acc["jump"] += cf * (x[:, 2] > 0.5).float(); acc["away"] += cf * (-(rel * v).sum(-1) / dn)
        acc["behind"] += cf * (ang.abs() > 2.0).float(); acc["stam_min"] = torch.where(c, torch.minimum(acc["stam_min"], w.stamina), acc["stam_min"])
        bit = c & (w.health < h0 - 0.2) & ~bitd
        over = (done | trunc) | (w.dying > 0)                        # the life ended this step (the light starts)
        killed = c & over & (w.dcause == W.DEATH_CAUSES.index("predator"))
        ended_other = c & over & ~killed
        escaped = c & ~bit & ~over & ((dn > 4.0) | (w.pred_mem_t >= W.PRED_FORGET))
        close(killed, "death"); close(bit & ~killed, "bite"); close(escaped, "escape"); close(ended_other, "other")
        in_chase &= ~(killed | bit | escaped | ended_other)
        alive &= ~(done | trunc)
        if not alive.any(): break


def summ(cs):
    if not cs: return {}
    k = ["steps", "d0", "stam0", "hp0", "pred_stam0", "speed", "push", "jump", "away", "behind", "stam_min"]
    return {x: round(sum(c[x] for c in cs) / len(cs), 3) for x in k} | dict(n=len(cs))


out = dict(mind=a.mind, mode=a.mode, lives=n, chases=len(chases), chases_per_life=round(len(chases) / n, 2),
           outcome_share={o: round(sum(c["outcome"] == o for c in chases) / max(1, len(chases)), 3) for o in ["escape", "bite", "death", "other"]},
           by_outcome={o: summ([c for c in chases if c["outcome"] == o]) for o in ["escape", "bite", "death"]})
# escape rate by stamina at the start
bins = [(0, .2), (.2, .5), (.5, .8), (.8, 1.01)]
out["escape_by_stam0"] = {f"{lo:.1f}-{hi:.1f}": (lambda cs: dict(n=len(cs), escape=round(sum(c["outcome"] == "escape" for c in cs) / max(1, len(cs)), 2)))(
    [c for c in chases if lo <= c["stam0"] < hi]) for lo, hi in bins}
out["escape_by_jumping"] = {nm: (lambda cs: dict(n=len(cs), escape=round(sum(c["outcome"] == "escape" for c in cs) / max(1, len(cs)), 2)))(
    [c for c in chases if f(c)]) for nm, f in [("jumps<10%", lambda c: c["jump"] < .1), ("jumps>=10%", lambda c: c["jump"] >= .1)]}
out["dasher"] = dash
print(json.dumps(out, indent=1))
e, b, dd = (out["by_outcome"].get(o, {}) for o in ["escape", "bite", "death"])
print(f"chases: {out['chases_per_life']} a life; escape {out['outcome_share']['escape']:.0%}, bite {out['outcome_share']['bite']:.0%}, death {out['outcome_share']['death']:.0%} | "
      f"stamina at start: escape {e.get('stam0', 0):.2f} vs bite {b.get('stam0', 0):.2f}; speed {e.get('speed', 0):.2f} vs {b.get('speed', 0):.2f} m/s; "
      f"push {e.get('push', 0):+.2f} vs {b.get('push', 0):+.2f}; jumping {e.get('jump', 0):.2f} vs {b.get('jump', 0):.2f}; faces away {e.get('behind', 0):.2f} vs {b.get('behind', 0):.2f}")
if dash["sprints"]:
    print(f"dasher: {dash['sprints'] / n:.2f} sprints within 4 m a life; {dash['bites']} bites, {dash['escapes']} sprints ended without a bite")
json.dump(out, open(a.mind.replace(".pt", f".chases_{a.mode}.json"), "w"), indent=1)
