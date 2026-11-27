"""What does a mind actually DO? Live N lives on habit, record everything, summarise behaviour.

    python3 -m survive.report /workspace/oazy_runs/survive/v4_s1/mind_60min.pt [--lives 256] [--mode test]

Writes next to the mind:
  <mind>.report.json   behaviour numbers (all lives) + a few lines a person can read
  <mind>.report.svg    top-down paths of the first lives (creature path coloured by time, the
                       predator dashed, food eaten as rings, death as a cross)
  <mind>.trace.pt      the full per-step history of the first --keep lives (positions, heading,
                       muscles, senses, predator, food) for any later question

Cheap: one batch of lives in one world mode, no learning (~1-2 min on the 4090 for 256 lives).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .bench import KINDS
from .model import load_mind, make_mind, wellbeing
from .render import set_eyes
from .train import Actor
from . import world as _world
from .world import DEATH_CAUSES, L, World


@torch.no_grad()
def record(mind, res, dev, mode, n, seed=4242, bf16=True):
    w = World(n, dev, mode=mode, seed=seed)
    act = Actor(mind, w, res, dev, deterministic=True)
    room = dict(bmin=w.bmin.clone(), bmax=w.bmax.clone(), bact=w.bact.clone(), bmov=w.bmov.clone(),
                lmin=w.lmin.clone(), lmax=w.lmax.clone(), lact=w.lact.clone(),
                fpos=w.fpos.clone(), fup=w.fup.clone())
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    cause = torch.zeros(n, dtype=torch.long, device=dev)
    T = _world.MAX_STEPS + _world.LIGHT
    keys = ["pos", "head", "gaze", "act", "pred", "pred_hunt", "energy", "health", "stamina", "ate", "live", "wb",
            "food_d", "bite", "lava_heat", "food_ang", "value", "dream_de", "dream_cont", "matter",
            "dash", "dash_on", "dash_bite", "reach", "tip", "hits", "kills", "old_age", "prey_d"]
    tr = {k: [] for k in keys}
    amp = bool(dev.type == "cuda" and bf16)
    for ti in range(T):
        a, _, emb = act.observe_act(bf16)
        # what the mind thinks of this moment: its critic's value, and a 1.5 s dream on its own habit
        # (every 5th step): how much energy it expects to gain, and whether it expects to live
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            ft = torch.cat([act.h, act.z], -1)
            x = torch.cat([ft, act.code.to(ft.dtype)], -1) if mind.intent else ft
            tr["value"].append(mind.critic(x).float().squeeze(-1))
            tr["matter"].append(mind.matter(emb).float()[:, 0] if getattr(mind, "matter", None) is not None
                                else torch.zeros(n, device=dev))
            if ti % 5 == 0:
                e_now = mind.sense_head(ft).float()[:, 0]
                h, z = act.h, act.z
                for _k in range(15):
                    xx = torch.cat([h, z], -1)
                    xx = torch.cat([xx, act.code.to(xx.dtype)], -1) if mind.intent else xx
                    aa, _ = mind.act(xx, deterministic=True)
                    h = mind.step_h(h, z, aa)
                    z = mind.sample(mind.imagine_z(h))
                se = mind.sense_head(torch.cat([h, z], -1)).float()
                tr["dream_de"].append(se[:, 0] - e_now)
                tr["dream_cont"].append(torch.sigmoid(mind.cont_head(torch.cat([h, z], -1)).float().squeeze(-1)))
            else:
                tr["dream_de"].append(torch.full((n,), float("nan"), device=dev))
                tr["dream_cont"].append(torch.full((n,), float("nan"), device=dev))
        act.advance(a)
        hp0 = w.health.clone()
        php0, palive0 = w.prey_hp.clone(), w.prey_alive.clone()
        don0 = w.dash_on.clone()
        done, trunc, c, ate = w.step(a)
        # world 5: the arm, the small prey, the dasher (all zeros / never in older worlds)
        tr["dash"].append(w.dash.clone()); tr["dash_on"].append(don0 | w.dash_on)
        tr["dash_bite"].append(don0 & ((w.dash - w.pos[:, :2]).norm(dim=-1) < 0.8) & (w.health < hp0 - 0.2))
        tr["reach"].append(w.arm_r.clone()); tr["tip"].append(w.tip.clone())
        tr["hits"].append((w.prey_hp < php0 - 1e-4).sum(-1))
        died = palive0 & w.prey_dead
        hit_now = w.prey_hp < php0 - 1e-4
        tr["kills"].append((died & hit_now).sum(-1))                   # killed by the arm
        tr["old_age"].append((died & ~hit_now).sum(-1))               # died on its own
        dpr = (w.prey - w.pos[:, None, :2]).norm(dim=-1)
        tr["prey_d"].append(torch.where(w.prey_alive, dpr, torch.full_like(dpr, 99.0)).min(-1).values)
        live = alive & (w.dying == 0)
        p = w.pos
        fd = (w.fpos[..., :2] - p[:, None, :2]).norm(dim=-1)
        fd = torch.where(w.fup, fd, torch.full_like(fd, 99.0))
        fi = fd.argmin(-1)
        fd = fd.min(-1).values
        to_f = w.fpos[torch.arange(n, device=dev), fi, :2] - p[:, :2]
        fang = torch.atan2(to_f[:, 1], to_f[:, 0]) - w.head
        tr["food_ang"].append(torch.atan2(torch.sin(fang), torch.cos(fang)).abs())
        tr["pos"].append(p.clone()); tr["head"].append(w.head.clone()); tr["gaze"].append(w.gaze.clone())
        tr["act"].append(a.clone()); tr["pred"].append(w.pred.clone())
        tr["pred_hunt"].append(w.pred_mem_t < _world.PRED_FORGET)
        tr["energy"].append(w.energy.clone()); tr["health"].append(w.health.clone())
        tr["stamina"].append(w.stamina.clone()); tr["ate"].append(ate.clone()); tr["live"].append(live.clone())
        tr["wb"].append(wellbeing(w.senses))
        tr["food_d"].append(fd)
        tr["bite"].append((w.health < hp0 - 0.2) & live & ~tr["dash_bite"][-1])
        tr["lava_heat"].append(w.senses[:, 5 + 12])                 # heat felt under the feet (log1p)
        cause = torch.where(alive & done, c, cause)
        alive &= ~(done | trunc)
        if not alive.any():
            break
    tr = {k: torch.stack(v, 1) for k, v in tr.items()}                 # n, T, ...
    tr["cause"] = cause
    return tr, room


def summarise(tr, room):
    live = tr["live"].float()                                         # n,T
    n, T = live.shape
    steps = live.sum(1)
    S = steps.sum().clamp(min=1)
    mean = lambda x: ((x.float() * live).sum() / S).item()
    p = tr["pos"][..., :2]
    a = tr["act"]
    # where it spends its time
    dwall = torch.minimum(p, L - p).min(-1).values                    # distance to the nearest arena wall
    corner = (torch.minimum(p, L - p) < 1.0).all(-1)
    bmin, bmax, bact = room["bmin"][:, None], room["bmax"][:, None], room["bact"][:, None]
    dx = torch.maximum(bmin[..., :2] - p[:, :, None], p[:, :, None] - bmax[..., :2]).clamp(min=0).norm(dim=-1)
    dbox = torch.where(bact & (bmax[..., 2] > 0.3), dx, torch.full_like(dx, 99.0)).min(-1).values
    # how it moves
    vel = (p[:, 1:] - p[:, :-1]).norm(dim=-1) / _world.DT
    vel = torch.cat([vel, vel[:, -1:]], 1)
    k = 50                                                             # 5 s windows: net move / path length
    if T > k:
        net = (p[:, k:] - p[:, :-k]).norm(dim=-1)
        path = torch.cumsum(torch.cat([torch.zeros_like(vel[:, :1]), vel[:, :-1] * _world.DT], 1), 1)
        plen = (path[:, k:] - path[:, :-k]).clamp(min=1e-3)
        straight = net / plen
        lk = live[:, k:] * live[:, :-k]
        straight_m = ((straight * lk).sum() / lk.sum().clamp(min=1)).item()
        still = ((plen < 0.5).float() * lk).sum() / lk.sum().clamp(min=1)
    else:
        straight_m, still = float("nan"), torch.tensor(float("nan"))
    cells = torch.floor(p / 0.5).long()
    cid = cells[..., 0] * 64 + cells[..., 1]
    area = []
    for i in range(n):
        m = live[i] > 0
        area.append(len(torch.unique(cid[i][m])) * 0.25 / max(m.sum().item() * _world.DT / 60, 1e-3))
    # predator
    dp = (tr["pred"] - p).norm(dim=-1)
    near = (dp < 3.0) & tr["pred_hunt"] & (live > 0)
    ddp = torch.cat([dp[:, 1:] - dp[:, :-1], torch.zeros_like(dp[:, :1])], 1) / _world.DT
    my_away = torch.cat([((p[:, 1:] - p[:, :-1]) * (p[:, :-1] - tr["pred"][:, :-1])).sum(-1)
                         / dp[:, :-1].clamp(min=1e-3), torch.zeros_like(dp[:, :1])], 1) / _world.DT
    fwd = torch.stack([torch.cos(tr["head"]), torch.sin(tr["head"])], -1)
    to_pred = (tr["pred"] - p) / dp[..., None].clamp(min=1e-3)
    facing = (fwd * to_pred).sum(-1)                                   # 1 = looking at it (body, not gaze)
    nm = near.float()
    nn_ = nm.sum().clamp(min=1)
    # food
    first_meal = []
    for i in range(n):
        e = (tr["ate"][i] >= 0).nonzero()
        first_meal.append(e[0, 0].item() * _world.DT if len(e) else float("nan"))
    fm = torch.tensor(first_meal)
    meals = (tr["ate"] >= 0).float().sum(1)
    hungry = (tr["energy"] < 0.3) & (live > 0)
    hm = hungry.float(); hn = hm.sum().clamp(min=1)
    dfood = tr["food_d"].clamp(max=20)
    dfd = torch.cat([dfood[:, 1:] - dfood[:, :-1], torch.zeros_like(dfood[:, :1])], 1) / _world.DT
    dfd = torch.where(dfd.abs() < 3, dfd, torch.zeros_like(dfd))          # ignore jumps when food is eaten/appears
    # does it SEE food and EXPECT to eat? (food ahead = nearest food within 1.5 m, within 35 deg of the body's heading)
    ahead = (dfood < 1.5) & (tr["food_ang"] < math.radians(35)) & (live > 0)
    none_ = (dfood > 3.0) & (live > 0)
    eat2 = torch.zeros_like(live)
    ate_ = (tr["ate"] >= 0).float()
    for d in range(1, 21):
        eat2[:, :-d] = torch.maximum(eat2[:, :-d], ate_[:, d:])
    def cm(x, m):
        ok = m & ~torch.isnan(x)
        return round((x[ok].float().mean().item() if ok.any() else float("nan")), 4)
    probe = dict(
        share_food_ahead=round(mean(ahead), 3),
        eats_within_2s_if_food_ahead=cm(eat2, ahead), eats_within_2s_if_no_food_near=cm(eat2, none_),
        value_food_ahead=cm(tr["value"], ahead), value_no_food=cm(tr["value"], none_),
        dream_energy_gain_food_ahead=cm(tr["dream_de"], ahead), dream_energy_gain_no_food=cm(tr["dream_de"], none_),
        dream_alive_food_ahead=cm(tr["dream_cont"], ahead), dream_alive_no_food=cm(tr["dream_cont"], none_),
        eye_matter_eat_food_ahead=cm(tr["matter"], ahead), eye_matter_eat_no_food=cm(tr["matter"], none_),
    )
    cause = tr["cause"]
    dead = cause > 0
    out = dict(probe=probe,
        lives=n, steps=round(steps.mean().item(), 1), meals=round(meals.mean().item(), 2),
        deaths_pct={DEATH_CAUSES[i]: round(100 * (cause == i).float().mean().item(), 1) for i in range(1, len(DEATH_CAUSES))},
        first_meal_s=dict(median=round(fm.nanmedian().item(), 1) if (~fm.isnan()).any() else None,
                          share_never=round(fm.isnan().float().mean().item(), 2)),
        place=dict(near_wall_lt_0_6m=round(mean(dwall < 0.6), 3), in_corner_1m=round(mean(corner), 3),
                   touching_box_lt_0_15m=round(mean(dbox < 0.15), 3), open_floor_gt_1m=round(mean((dwall > 1) & (dbox > 1)), 3),
                   area_m2_per_min=round(sum(area) / n, 2)),
        moving=dict(speed_ms=round(mean(vel), 2), forward=round(mean(a[..., 0] > 0.05), 3), backward=round(mean(a[..., 0] < -0.05), 3),
                    turn_mean=round(mean(a[..., 1]), 3), turn_abs=round(mean(a[..., 1].abs()), 3),
                    straightness_5s=round(straight_m, 3), stuck_5s=round(still.item(), 3),
                    jump_per_min=round(mean(a[..., 2] > 0.5) * 600, 1), grip=round(mean(a[..., 3] > 0), 3),
                    gaze_mean=round(mean(a[..., 4]), 3), gaze_abs=round(mean(a[..., 4].abs()), 3)),
        predator=dict(near_hunting_share=round(mean(near), 3),
                      when_near_dist_change_ms=round(((ddp * nm).sum() / nn_).item(), 3),
                      when_near_my_speed_away_ms=round(((my_away * nm).sum() / nn_).item(), 3),
                      when_near_facing_it=round(((facing * nm).sum() / nn_).item(), 3),
                      when_near_backing=round((((a[..., 0] < -0.05).float() * nm).sum() / nn_).item(), 3),
                      bites_per_life=round(tr["bite"].float().sum(1).mean().item(), 2)),
        hunger=dict(share_hungry=round(mean(hungry), 3),
                    when_hungry_food_dist_change_ms=round(((dfd * hm).sum() / hn).item(), 3),
                    when_hungry_speed_ms=round(((vel * hm).sum() / hn).item(), 2),
                    when_hungry_near_wall=round((((dwall < 0.6).float() * hm).sum() / hn).item(), 3),
                    food_dist_m=round(mean(dfood), 2)),
        wellbeing=round(mean(tr["wb"]), 3),
    )
    # world 5: hunting with the arm, the dasher
    lv = live > 0
    ate_k = tr["ate"]
    dd = (tr["dash"] - p).norm(dim=-1)
    dash_close = tr["dash_on"] & (dd < 4.0) & lv
    out["hunt"] = dict(
        arm_out_share=round(mean(tr["reach"] > 0.3), 3), arm_reach_mean=round(mean(tr["reach"]), 3),
        prey_within_2m_share=round(mean(tr["prey_d"] < 2.0), 3),
        hits_per_life=round((tr["hits"].float() * live).sum(1).mean().item(), 2),
        kills_per_life=round((tr["kills"].float() * live).sum(1).mean().item(), 2),
        old_age_deaths_per_life=round((tr["old_age"].float() * live).sum(1).mean().item(), 2),
        carcasses_eaten_per_life=round(((ate_k == 3).float() * live).sum(1).mean().item(), 2),
        fruits_per_life=round((((ate_k >= 0) & (ate_k < 3)).float() * live).sum(1).mean().item(), 2),
        arm_out_when_prey_near=round((((tr["reach"] > 0.3) & (tr["prey_d"] < 1.5) & lv).float().sum() /
                                      ((tr["prey_d"] < 1.5) & lv).float().sum().clamp(min=1)).item(), 3),
        dasher_sprints_near_share=round(mean(dash_close), 3),
        dasher_bites_per_life=round(tr["dash_bite"].float().sum(1).mean().item(), 2))
    # a few plain sentences
    s = []
    pl, mv, pr, hu = out["place"], out["moving"], out["predator"], out["hunger"]
    s.append(f"lives {out['steps'] / 10:.0f} s, {out['meals']} meals; deaths " +
             ", ".join(f"{k} {v:.0f}%" for k, v in out["deaths_pct"].items() if v >= 1))
    s.append(f"time: {100 * pl['near_wall_lt_0_6m']:.0f}% hugging the arena wall, {100 * pl['in_corner_1m']:.0f}% in corners, "
             f"{100 * pl['touching_box_lt_0_15m']:.0f}% touching blocks, {100 * pl['open_floor_gt_1m']:.0f}% in the open; "
             f"covers {pl['area_m2_per_min']} m2 of new floor a minute")
    s.append(f"moves at {mv['speed_ms']} m/s, {100 * mv['forward']:.0f}% forward / {100 * mv['backward']:.0f}% backward; "
             f"5 s paths are {100 * mv['straightness_5s']:.0f}% straight (low = circling), stuck {100 * mv['stuck_5s']:.0f}% of 5 s windows; "
             f"turn bias {mv['turn_mean']:+.2f}")
    s.append(f"predator hunting within 3 m {100 * pr['near_hunting_share']:.0f}% of the time; then distance changes "
             f"{pr['when_near_dist_change_ms']:+.2f} m/s, it moves away at {pr['when_near_my_speed_away_ms']:+.2f} m/s, "
             f"faces it {pr['when_near_facing_it']:+.2f}, backs up {100 * pr['when_near_backing']:.0f}%; {pr['bites_per_life']} bites a life")
    s.append(f"hungry (energy < 0.3) {100 * hu['share_hungry']:.0f}% of the time; then nearest food distance changes "
             f"{hu['when_hungry_food_dist_change_ms']:+.2f} m/s (negative = heading to food), speed {hu['when_hungry_speed_ms']} m/s, "
             f"near a wall {100 * hu['when_hungry_near_wall']:.0f}%")
    s.append(f"food within 1.5 m ahead {100 * probe['share_food_ahead']:.0f}% of the time; then it eats within 2 s "
             f"{100 * probe['eats_within_2s_if_food_ahead']:.0f}% (vs {100 * probe['eats_within_2s_if_no_food_near']:.0f}% with no food near); "
             f"its 1.5 s dream expects energy {probe['dream_energy_gain_food_ahead']:+.3f} (vs {probe['dream_energy_gain_no_food']:+.3f}); "
             f"value {probe['value_food_ahead']:.3f} vs {probe['value_no_food']:.3f}; eye's 'I will eat' {probe['eye_matter_eat_food_ahead']:.3f} vs {probe['eye_matter_eat_no_food']:.3f}")
    hu5 = out["hunt"]
    if hu5["prey_within_2m_share"] > 0 or hu5["dasher_sprints_near_share"] > 0:
        s.append(f"hunting: arm out (> 0.3 m) {100 * hu5['arm_out_share']:.0f}% of the time ({100 * hu5['arm_out_when_prey_near']:.0f}% when prey is within 1.5 m); "
                 f"prey within 2 m {100 * hu5['prey_within_2m_share']:.0f}%; {hu5['hits_per_life']} hits, {hu5['kills_per_life']} kills (+{hu5['old_age_deaths_per_life']} prey dying of age), "
                 f"{hu5['carcasses_eaten_per_life']} carcasses eaten vs {hu5['fruits_per_life']} fruits a life; "
                 f"dasher sprinting within 4 m {100 * hu5['dasher_sprints_near_share']:.0f}%, {hu5['dasher_bites_per_life']} dasher bites a life")
    out["story"] = s
    return out


def svg(tr, room, lives=6, cols=3):
    """Top-down paths of the first lives."""
    sz, pad = 300, 12
    rows = math.ceil(lives / cols)
    W, H = cols * (sz + pad) + pad, rows * (sz + pad + 18) + pad
    k = sz / L
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="11">',
         f'<rect width="{W}" height="{H}" fill="#f6f5f2"/>']
    live = tr["live"]
    for i in range(min(lives, live.shape[0])):
        x0, y0 = pad + (i % cols) * (sz + pad), pad + (i // cols) * (sz + pad + 18) + 16
        X = lambda v: x0 + v * k
        Y = lambda v: y0 + sz - v * k
        o.append(f'<rect x="{x0}" y="{y0}" width="{sz}" height="{sz}" fill="#fff" stroke="#999"/>')
        for j in range(room["lmin"].shape[1]):
            if room["lact"][i, j]:
                a, b = room["lmin"][i, j].tolist(), room["lmax"][i, j].tolist()
                o.append(f'<rect x="{X(a[0]):.1f}" y="{Y(b[1]):.1f}" width="{(b[0] - a[0]) * k:.1f}" height="{(b[1] - a[1]) * k:.1f}" fill="#f2a06b" opacity=".6"/>')
        for j in range(room["bmin"].shape[1]):
            if room["bact"][i, j]:
                a, b = room["bmin"][i, j].tolist(), room["bmax"][i, j].tolist()
                col = "#c9a878" if room["bmov"][i, j] else ("#8c8aa0" if b[2] > 1.2 else "#a29d92")
                o.append(f'<rect x="{X(a[0]):.1f}" y="{Y(b[1]):.1f}" width="{(b[0] - a[0]) * k:.1f}" height="{(b[1] - a[1]) * k:.1f}" fill="{col}"/>')
        for j in range(room["fpos"].shape[1]):                          # food at birth
            if room["fup"][i, j]:
                f = room["fpos"][i, j].tolist()
                o.append(f'<circle cx="{X(f[0]):.1f}" cy="{Y(f[1]):.1f}" r="2.5" fill="#e0b81e"/>')
        m = live[i].nonzero().squeeze(-1)
        if len(m) == 0:
            continue
        T = m[-1].item() + 1
        p, q = tr["pos"][i, :T, :2].tolist(), tr["pred"][i, :T].tolist()
        step = max(1, T // 600)
        pts = " ".join(f"{X(v[0]):.1f},{Y(v[1]):.1f}" for v in q[::step])
        o.append(f'<polyline points="{pts}" fill="none" stroke="#7b55c7" stroke-width="1" stroke-dasharray="3 3" opacity=".55"/>')
        for t in range(0, T - step, step):                            # creature path: blue (birth) -> red (end)
            f = t / max(T - 1, 1)
            col = f"rgb({int(40 + 200 * f)},{int(90 - 40 * f)},{int(220 - 180 * f)})"
            o.append(f'<line x1="{X(p[t][0]):.1f}" y1="{Y(p[t][1]):.1f}" x2="{X(p[t + step][0]):.1f}" y2="{Y(p[t + step][1]):.1f}" stroke="{col}" stroke-width="1.6"/>')
        if tr["dash_on"][i, :T].any() or tr["dash"][i, :T].abs().sum() > 0:
            dq = tr["dash"][i, :T].tolist()
            on = tr["dash_on"][i, :T].tolist()
            for t in range(0, T - step, step):                        # the dasher: faint, bold red while it sprints
                if dq[t][0] > -5:
                    o.append(f'<line x1="{X(dq[t][0]):.1f}" y1="{Y(dq[t][1]):.1f}" x2="{X(dq[t + step][0]):.1f}" y2="{Y(dq[t + step][1]):.1f}" '
                             f'stroke="#d2452f" stroke-width="{1.8 if on[t] else 0.6}" opacity="{0.8 if on[t] else 0.25}"/>')
        for t in (tr["ate"][i, :T] >= 0).nonzero().squeeze(-1).tolist():
            carc = tr["ate"][i, t].item() == 3
            o.append(f'<circle cx="{X(p[t][0]):.1f}" cy="{Y(p[t][1]):.1f}" r="{7 if carc else 5}" fill="none" stroke="{"#e07b1a" if carc else "#2a9d3a"}" stroke-width="{3 if carc else 2}"/>')
        for t in (tr["kills"][i, :T] > 0).nonzero().squeeze(-1).tolist():
            q_ = tr["tip"][i, t].tolist()
            o.append(f'<text x="{X(q_[0]) - 5:.1f}" y="{Y(q_[1]) + 5:.1f}" fill="#e07b1a" font-size="15" font-weight="bold">✶</text>')
        for t in tr["dash_bite"][i, :T].nonzero().squeeze(-1).tolist():
            o.append(f'<circle cx="{X(p[t][0]):.1f}" cy="{Y(p[t][1]):.1f}" r="3" fill="#d2452f"/>')
        for t in tr["bite"][i, :T].nonzero().squeeze(-1).tolist():
            o.append(f'<circle cx="{X(p[t][0]):.1f}" cy="{Y(p[t][1]):.1f}" r="3" fill="#7b55c7"/>')
        o.append(f'<circle cx="{X(p[0][0]):.1f}" cy="{Y(p[0][1]):.1f}" r="4" fill="#2850dc"/>')
        e = p[T - 1]
        o.append(f'<text x="{X(e[0]) - 4:.1f}" y="{Y(e[1]) + 4:.1f}" fill="#c0282c" font-size="13" font-weight="bold">x</text>')
        c = DEATH_CAUSES[tr["cause"][i].item()]
        meals = int((tr["ate"][i, :T] >= 0).sum())
        o.append(f'<text x="{x0}" y="{y0 - 4}">life {i}: {T / 10:.0f} s, {meals} meals, end: {c}</text>')
    o.append("</svg>")
    return "\n".join(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mind")
    ap.add_argument("--lives", type=int, default=256)
    ap.add_argument("--mode", default="test")
    ap.add_argument("--keep", type=int, default=16, help="lives whose full history is saved")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.mind, map_location=dev)
    set_eyes(ck["args"])
    mind = make_mind(ck["args"]).to(dev)
    load_mind(mind, ck["mind"])
    mind.eval()
    tr, room = record(mind, ck["args"]["res"], dev, a.mode, a.lives)
    tr = {k: v.cpu() for k, v in tr.items()}
    room = {k: v.cpu() for k, v in room.items()}
    out = dict(mind=a.mind, minutes=round(ck.get("minutes", 0), 1), mode=a.mode, **summarise(tr, room))
    base = Path(a.mind).with_suffix("")
    Path(f"{base}.report.json").write_text(json.dumps(out, indent=1))
    Path(f"{base}.report.svg").write_text(svg(tr, room))
    torch.save(dict(trace={k: v[:a.keep] for k, v in tr.items()}, room={k: v[:a.keep] for k, v in room.items()}),
               f"{base}.trace.pt")
    print("\n".join(out["story"]))
    print("saved", f"{base}.report.json", f"{base}.report.svg", f"{base}.trace.pt")


if __name__ == "__main__":
    main()
