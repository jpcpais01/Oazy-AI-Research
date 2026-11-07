// Forecasts: where a run's losses and HellaSwag score will land at the planned end of training.
//
// The idea in plain words. A model's loss falls with how much it has learned, and how much it learns depends on
// the learning rate it has had so far, not just on the words read: a word read at half the learning rate teaches
// about half as much. So the curve is fitted against "learning so far" (the sum of the learning rate over the
// words read), which we know in advance for the whole run, because the schedule is planned. On top of that, when
// the learning rate is lowered (the cool-down over the last 20%, or a cosine's slow fall), the loss drops by an
// extra amount that builds up over a few hundred steps: the steps get small enough to settle into the valley
// instead of bouncing across it. That drop is the second term.
//
//     loss(t) = floor + A × (learning so far)^(-alpha)  −  C × (how much the learning rate has come down, settled)
//
// This is the "learning-rate annealing" law of Tissue et al. (2024, "Scaling Law with Learning Rate Annealing"),
// which fits whole runs under any schedule (constant, cosine, warmup-stable-decay) with one set of numbers.
//
// Before a run reaches its cool-down, its own points say nothing about how big the cool-down drop C will be.
// Then C comes from the runs on the pod that already finished a cool-down (as a share of the loss), and the
// range shown is widened by how much that share varied. Once the cool-down starts, the run's own points take over.
//
// The training loss also gets a jump where the text mix changes (Tagus at 80%: more conversations, which are
// easier to predict), the same day the learning rate starts falling; the held-out text does not change, so the
// held-out loss has no jump. The jump's size is learned the same way: from earlier runs, then from the run itself.
//
// HellaSwag follows the held-out loss closely: the score is fitted as a straight line against the run's own
// held-out loss, and the held-out forecast is pushed through it.
//
// The range: the forecast is replayed from earlier moments of the same run, looking exactly as far ahead (as a share
// of the words read then) as it now looks, and checked against what really happened. Their typical miss is the
// range, plus the uncertainty of the cool-down drop. Forecast.backtest() runs the whole method on finished runs from
// many cut points, which is how the dashboard reports how often the range held.
//
// Works in the browser (window.Forecast) and in node (module.exports), for the tests in oazy/web/forecast.test.js.

const Forecast = (() => {
  const Z = 1.28;                  // the range is ±1.28 typical misses: about 8 in 10 outcomes inside it, if the misses are normal
  const LAG_STEPS = 300;           // how many steps a lower learning rate takes to pay off fully (the law's momentum)
  const DECAY = 0.2;               // the WSD cool-down is the last 20% (oazy/train.py lr_wsd), unless the run says (Bee4: 0.3)
  const decayOf = (cfg) => (cfg && cfg.decay > 0 && cfg.decay < 1 ? cfg.decay : DECAY);
  const DEFAULT_PRIOR = {          // used only when no run on the pod has finished a cool-down yet (rough, from the literature)
    val: { g: 0.035, sd: 0.02, n: 0 }, loss: { g: 0.035, sd: 0.02, n: 0 }, jump: { g: -0.03, sd: 0.03, n: 0 },
  };

  // ---------- the schedule ----------

  // the learning rate the trainer uses at a share p of the run, relative to its peak (after the warmup)
  function etaPlanned(cfg, p) {
    if (cfg.schedule === "wsd") {
      const left = Math.min(1, Math.max(0, (1 - p) / decayOf(cfg)));
      return cfg.decay_shape === "sqrt" && left < 1 ? 1 - Math.sqrt(1 - left) : left;
    }
    return 0.05 + 0.95 * 0.5 * (1 + Math.cos(Math.PI * Math.min(1, p)));     // cosine (lr_at)
  }

  // where the text mix changes, as shares of the run: "a@0;b@0.8" -> [0.8]
  function mixSwitches(cfg) {
    if (!cfg.mix || typeof cfg.mix !== "string") return [];
    return cfg.mix.split(";").map((s) => +s.split("@")[1]).filter((x) => x > 0 && x < 1);
  }

  // ---------- the recipe's other schedules (they follow the share of the run, so a longer run stretches them) ----------

  // the model's depth at a share p of the run: "8:0,12:0.1,20:0.25" (ember/model.py parse_plan, depth_at)
  function depthPlanned(cfg, p) {
    const full = cfg.config?.n_layer;
    if (!cfg.grow || !full) return full || 1;
    const plan = String(cfg.grow).split(",").map((x) => x.split(":").map(Number)).filter((x) => x.length === 2 && isFinite(x[0]));
    let d = plan.length ? plan[0][0] : full;
    for (const [dd, w] of plan) if (p >= w) d = dd;
    return d;
  }
  // the mentor's share at a share p of the run (ember/mentor.py fade)
  function mentorPlanned(cfg, p) {
    const w0 = cfg.mentor_weight || 0, until = cfg.mentor_until || 0, floor = cfg.mentor_floor || 0, off = cfg.mentor_off || 0;
    if (!w0 || until <= 0) return 0;
    let share = floor + Math.max(0, (w0 - floor) * (1 - Math.min(p, until) / until));
    if (off > 0 && p > 1 - off) share *= Math.max(0, (1 - p) / off);
    return Math.max(0, share);
  }
  // does the recipe change the model along the way (growing, a fading mentor)? Then a longer run is not just more of the same
  const recipeMoves = (cfg) => !!(cfg.grow && cfg.config?.n_layer && depthPlanned(cfg, 0) < cfg.config.n_layer) || !!cfg.mentor_weight;
  // a logged value by tokens (step-wise: the last one logged at or before t)
  function stepAt(rows) {
    return (t) => { let v = null; for (const [x, y] of rows) { if (x > t) break; v = y; } return v; };
  }

  // Everything the fit needs about a run, on a grid of tokens from 0 to the planned end: the learning rate (logged
  // where it was logged, planned after), "learning so far" S1 and "settled fall" S2. Returns null if there is no plan.
  // hypo: a longer run of `hypo` tokens on the same recipe (same warmup and peak, its cool-down over ITS last 20%,
  // its text change at ITS 80%): the learning rate logged so far is used only up to the end of the warmup.
  // eff: {beta, mu}: a word read by a smaller model, or with the mentor's help, counts as (depth/full)^beta × (1 + mu ×
  // mentor share) of a word: the growing recipe's learning speed, fitted on the run's own early stretch (see stretch()).
  function plan(cfg, train, lagSteps = LAG_STEPS, hypo = 0, eff = null) {
    const end = hypo || cfg.total_tokens;
    if (!(end > 0) || cfg.minutes) return null;          // races on the clock follow time, not tokens: no forecast
    const peak = cfg.lr || Math.max(...train.map((x) => x.lr || 0)) || 1;
    const logged = train.filter((x) => x.lr != null && x.tokens > 0).map((x) => [x.tokens, x.lr / peak]);
    const warmEnd = (logged.find((x) => x[1] >= 0.98) || [0])[0];
    const lastLog = hypo ? warmEnd : logged.length ? logged[logged.length - 1][0] : 0;
    const N = 2000, dt = end / N;
    const eta = new Float64Array(N + 1);
    let j = 0;
    for (let i = 0; i <= N; i++) {
      const t = i * dt;
      if (t <= lastLog && logged.length > 1) {
        while (j < logged.length - 2 && logged[j + 1][0] < t) j++;
        const [t0, e0] = logged[j], [t1, e1] = logged[j + 1];
        eta[i] = t <= t0 ? e0 : e0 + (e1 - e0) * Math.min(1, (t - t0) / Math.max(1, t1 - t0));
      } else eta[i] = t < warmEnd ? 1 : etaPlanned(cfg, t / end);
    }
    // how much each word counts (1 without eff): from the logs while the real run has them, else the recipe
    let wt = null;
    if (eff && (eff.beta || eff.mu)) {
      const full = cfg.config?.n_layer || 1, realEnd = cfg.total_tokens;
      const dLog = hypo ? null : stepAt(train.filter((x) => x.depth != null).map((x) => [x.tokens, x.depth]));
      const mLog = hypo ? null : stepAt(train.filter((x) => x.mentor_w != null || x.kind === "train").map((x) => [x.tokens, x.mentor_w || 0]));
      const lastT = train.length ? train[train.length - 1].tokens : 0;
      wt = new Float64Array(N + 1);
      for (let i = 0; i <= N; i++) {
        const t = i * dt, live = !hypo && t <= lastT;
        const d = (live && dLog(t)) || depthPlanned(cfg, t / (hypo ? end : realEnd));
        const mw = live ? (mLog(t) ?? 0) : mentorPlanned(cfg, t / (hypo ? end : realEnd));
        wt[i] = Math.pow(Math.min(1, d / full), eff.beta || 0) * Math.max(0.05, 1 + (eff.mu || 0) * mw);
      }
    }
    // S1: learning so far, in billions of tokens at the peak rate (the same unit for every length of run)
    // S2: how much of the learning rate's fall has paid off: each fall pays off over ~LAG_STEPS steps
    const L = lagSteps * (cfg.batch_tokens || 524288);
    const k = Math.exp(-dt / L);
    const S1 = new Float64Array(N + 1), S2 = new Float64Array(N + 1);
    let m = 0, top = 0;
    for (let i = 1; i <= N; i++) {
      S1[i] = S1[i - 1] + 0.5 * (eta[i] * (wt ? wt[i] : 1) + eta[i - 1] * (wt ? wt[i - 1] : 1)) * dt / 1e9;
      if (i * dt > warmEnd) {
        top = Math.max(top, eta[i - 1]);
        const fall = Math.max(0, eta[i - 1] - eta[i]);      // only falls count; the warmup's rise does not
        m = m * k + fall;
        S2[i] = S2[i - 1] + m * (1 - k);
      } else S2[i] = 0;
    }
    const at = (A) => (t) => {
      const x = Math.max(0, Math.min(N, t / dt)), i = Math.min(N - 1, Math.floor(x)), u = x - i;
      return A[i] + (A[i + 1] - A[i]) * u;
    };
    const decayAt = cfg.schedule === "wsd" ? end * (1 - decayOf(cfg)) : 0;
    return { lag: lagSteps, eff, end, S1: at(S1), S2: at(S2), eta: at(eta), S2end: S2[N], decayAt, switches: mixSwitches(cfg).map((p) => p * end), warmEnd };
  }

  // ---------- smoothing ----------

  // the noisy training loss into ~n even bins of tokens: [tokens, median, noise of the median]
  function bins(pts, n = 60) {
    if (pts.length <= n) return pts.map((p) => [p[0], p[1], null]);
    const t0 = pts[0][0], t1 = pts[pts.length - 1][0], w = (t1 - t0) / n || 1, B = Array.from({ length: n }, () => []);
    for (const p of pts) B[Math.min(n - 1, Math.floor((p[0] - t0) / w))].push(p);
    return B.filter((b) => b.length >= 2).map((b) => {
      const ys = b.map((p) => p[1]).sort((u, v) => u - v), med = ys[ys.length >> 1];
      const mad = ys.map((y) => Math.abs(y - med)).sort((u, v) => u - v)[ys.length >> 1] * 1.4826;
      return [b[b.length >> 1][0], med, mad * 1.25 / Math.sqrt(b.length)];
    });
  }

  // ---------- the fit ----------

  // weighted least squares for a few unknowns (normal equations, Gaussian elimination); ridge: [[index, value, weight]]
  function lsq(rows, ys, ws, ridge = []) {
    const k = rows[0].length, M = Array.from({ length: k }, () => new Float64Array(k + 1));
    for (let r = 0; r < rows.length; r++) {
      const x = rows[r], w = ws[r];
      for (let a = 0; a < k; a++) { M[a][k] += w * x[a] * ys[r]; for (let b = 0; b < k; b++) M[a][b] += w * x[a] * x[b]; }
    }
    for (const [i, v, w] of ridge) { M[i][i] += w; M[i][k] += w * v; }
    for (let c = 0; c < k; c++) {
      let p = c;
      for (let r = c + 1; r < k; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      if (Math.abs(M[p][c]) < 1e-14) return null;
      [M[c], M[p]] = [M[p], M[c]];
      for (let r = 0; r < k; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let q = c; q <= k; q++) M[r][q] -= f * M[c][q]; }
    }
    return M.map((row, i) => row[k] / row[i]);
  }

  // Fit loss = E + A·S1^-alpha − C·S2 (+ J after a mix change, for the training loss).
  // pts: [[tokens, value, noise|null]]. prior: {C0, sdC} keeps C near the prior when the points cannot tell (before
  // the cool-down); jump: null (no jump term), or {at, J0, sdJ}. Returns the curve and its numbers, or null.
  function fitLaw(pts, P, { prior = null, jump = null, alpha: fixed = null } = {}) {
    if (pts.length < 6) return null;
    const s1 = pts.map((p) => P.S1(p[0])), s2 = pts.map((p) => P.S2(p[0]));
    const after = jump ? pts.map((p) => (p[0] >= jump.at ? 1 : 0)) : null;
    const ys = pts.map((p) => p[1]);
    // the scatter of the points around a smooth curve, to weigh the points and the priors against each other
    const noise = pts.map((p) => p[2]).filter((x) => x > 0);
    const floor = 1e-4;
    const ws0 = pts.map((p) => 1 / Math.max(floor, p[2] || 0) ** 2);
    const uniform = noise.length < pts.length / 2;
    let best = null, grid = [];
    const tryAlpha = (alpha, ws, sig) => {
      const rows = pts.map((_, i) => [1, Math.pow(s1[i], -alpha), -s2[i], ...(jump ? [after[i]] : [])]);
      const ridge = [];
      if (prior) ridge.push([2, prior.C0, 1 / Math.max(1e-9, prior.sdC) ** 2 * sig]);
      else ridge.push([2, 0, 1e-6]);
      if (jump) ridge.push([3, jump.J0, 1 / Math.max(1e-9, jump.sdJ) ** 2 * sig]);
      const c = lsq(rows, ys, ws, ridge);
      if (!c || !(c[1] > 0) || c[2] < 0) return null;
      let e = 0;
      for (let i = 0; i < rows.length; i++) { let f = 0; for (let q = 0; q < c.length; q++) f += rows[i][q] * c[q]; e += ws[i] * (ys[i] - f) ** 2; }
      return { alpha, c, e };
    };
    // two passes: first a rough scale of the scatter, then the real fit with the priors weighed against it
    let sig = 1, ws = uniform ? pts.map(() => 1) : ws0;
    for (let pass = 0; pass < 2; pass++) {
      best = null;
      if (fixed) {                    // the shape borrowed from the held-out loss: only the levels are fitted
        best = tryAlpha(fixed, ws, sig);
        if (!best) return null;
        const dof = Math.max(1, pts.length - 2 - (jump ? 1 : 0));
        sig = uniform ? best.e / dof : Math.max(1, best.e / dof);
        continue;
      }
      grid = [];
      for (let i = 0; i <= 60; i++) {
        const r = tryAlpha(0.03 * Math.pow(100, i / 60), ws, sig);              // alpha from 0.03 to 3
        if (r) grid.push(r);
        if (r && (!best || r.e < best.e)) best = r;
      }
      if (!best) return null;
      let lo = best.alpha / 1.08, hi = best.alpha * 1.08;
      for (let q = 0; q < 24; q++) {
        const m1 = hi - (hi - lo) * 0.618, m2 = lo + (hi - lo) * 0.618, r1 = tryAlpha(m1, ws, sig), r2 = tryAlpha(m2, ws, sig);
        if (!r1 || !r2) break;
        if (r1.e < r2.e) hi = m2; else lo = m1;
      }
      const r = tryAlpha((lo + hi) / 2, ws, sig);
      if (r && r.e < best.e) best = r;
      // weights are 1/noise²; with uniform weights the scatter itself sets how strongly the priors pull
      const dof = Math.max(1, pts.length - 3 - (jump ? 1 : 0));
      sig = uniform ? best.e / dof : Math.max(1, best.e / dof);
    }
    const [E, A, C, J = 0] = best.c, alpha = best.alpha;
    const resid = pts.map((p, i) => p[1] - (E + A * Math.pow(s1[i], -alpha) - C * s2[i] + (jump ? J * after[i] : 0)));
    const rmse = Math.sqrt(resid.reduce((a, r) => a + r * r, 0) / resid.length);
    const f = (t, withJump = true) => E + A * Math.pow(P.S1(t), -alpha) - C * P.S2(t) + (jump && withJump && t >= jump.at ? J : 0);
    const noCool = (t) => E + A * Math.pow(P.S1(t), -alpha) + (jump && t >= jump.at ? J : 0);
    // the other curve shapes the points can't rule out (about two standard errors): they agree where there are
    // points and part ways far beyond them, which is how unsure a long-range forecast is
    const dof = Math.max(1, pts.length - 3 - (jump ? 1 : 0));
    const alts = (fixed ? [best] : grid.filter((r) => r.e <= best.e * (1 + 4 / dof))).map((r) => {
      const [e, a, c, j = 0] = r.c;
      return (PP, t, ja) => e + a * Math.pow(PP.S1(t), -r.alpha) - c * PP.S2(t) + (ja != null && t >= ja ? j : 0);
    });
    return { E, A, C, J, alpha, f, noCool, rmse, n: pts.length, alts };
  }

  // ---------- one forecast ----------

  // where the fit may start: after the model's own growth spurts, the batch ramp and a fading mentor, which bend the
  // curve in ways the law does not know; but never so late that too little is left to fit
  function fitStart(cfg, metrics, now) {
    let s = 0;
    for (const x of metrics) if ((x.kind === "grow" || x.kind === "context") && x.tokens < now) s = Math.max(s, x.tokens);
    if (cfg.batch_ramp) s = Math.max(s, (+String(cfg.batch_ramp).split(":")[1] || 0) * cfg.total_tokens);
    if (cfg.mentor_weight && cfg.mentor_until) s = Math.max(s, cfg.mentor_until * cfg.total_tokens);
    s = Math.max(s * 1.05, now * 0.2);
    return s > now * 0.6 ? now * 0.4 : s;
  }

  // The series a forecast is made for, from a run's metrics (only what was known at `upto`).
  function series(metrics, key, upto = Infinity) {
    if (key === "loss") return bins(metrics.filter((x) => x.kind === "train" && x.loss != null && x.tokens <= upto).map((x) => [x.tokens, x.loss]));
    if (key === "val") return metrics.filter((x) => x.kind === "eval" && x.val_loss != null && x.tokens <= upto).map((x) => [x.tokens, x.val_loss, null]);
    return [];
  }

  // where the training loss jumps (a mix change), if it falls inside the run; the first one after the fit start
  function jumpAt(P, from) { return P.switches.find((t) => t > from) ?? null; }

  // How quickly a lower learning rate pays off differs between models. Once the run is far enough into its
  // cool-down to tell, it is chosen by which fits the run's own points best; before that, it is what the finished
  // cool-downs on the pod showed (or 300 steps).
  const LAGS = [75, 150, 300, 600, 1200];
  function bestCore(cfg, metrics, key, now, priors, plans) {
    const P0 = plans(priors.lag || LAG_STEPS);
    if (!P0) return null;
    const seen = P0.S2(now) / Math.max(1e-9, P0.S2end);
    if (!(seen > 0.08) || P0.S2end < 1e-6) { const c = core(cfg, metrics, P0, key, now, priors); return c && { ...c, P: P0 }; }
    let best = null;
    for (const lag of LAGS) {
      const P = plans(lag), c = core(cfg, metrics, P, key, now, priors);
      if (c && (!best || c.fit.rmse < best.fit.rmse)) best = { ...c, P };
    }
    return best;
  }
  function planner(cfg, metrics) {
    const train = metrics.filter((x) => x.kind === "train"), memo = {};
    return (lag) => (lag in memo ? memo[lag] : (memo[lag] = plan(cfg, train, lag)));
  }

  // The bare forecast from the points known up to `now`: the fitted curve and the numbers behind it.
  function core(cfg, metrics, P, key, now, priors) {
    const from = fitStart(cfg, metrics, now);
    const pts = series(metrics, key, now).filter((p) => p[0] >= from);
    if (pts.length < 6) return null;
    const level = pts[pts.length - 1][1];
    const pr = priors[key] || DEFAULT_PRIOR[key];
    // the cool-down drop C, as a share of the loss: C × S2(end) ≈ g × loss
    const prior = P.S2end > 1e-6 ? { C0: pr.g * level / P.S2end, sdC: Math.max(pr.sd, 0.25 * pr.g) * level / P.S2end } : null;
    let jump = null;
    const ja = key === "loss" ? jumpAt(P, from) : null;
    if (ja != null) { const pj = priors.jump || DEFAULT_PRIOR.jump; jump = { at: ja, J0: pj.g * level, sdJ: Math.max(pj.sd, 0.01) * level }; }
    // the training loss is noisy; its shape over the run is the held-out loss's (same model, same learning), so
    // its alpha comes from the held-out fit when there is one, and only its own levels are fitted
    let alpha = null;
    if (key === "loss") { const v = core(cfg, metrics, P, "val", now, priors); if (v) alpha = v.fit.alpha; }
    const fit = fitLaw(pts, P, { prior, jump, alpha }) || (alpha ? fitLaw(pts, P, { prior, jump }) : null);
    if (!fit) return null;
    return { fit, pts, from, level, prior, jump, pr };
  }

  // the value really reached at time t (for checking replays against): the nearest point, or bins around it
  function actualAt(pts, t) {
    let best = null;
    for (const p of pts) if (!best || Math.abs(p[0] - t) < Math.abs(best[0] - t)) best = p;
    return best && Math.abs(best[0] - t) < 0.06 * t ? best[1] : null;
  }

  // A full forecast for a metric ("loss" or "val") of a run, from the points known up to `now` (default: all).
  // Returns {curve: [[t, y]], band: [[t, lo, hi]], yEnd, lo, hi, sd, cool, jump, checks, ...} or null.
  function metric(cfg, metrics, key, { now = null, priors = {}, replays = true } = {}) {
    const plans = planner(cfg, metrics), P0 = plans(priors.lag || LAG_STEPS);
    if (!P0) return null;
    const known = series(metrics, key);
    if (!known.length) return null;
    now = Math.min(now ?? Infinity, known[known.length - 1][0]);
    if (now < P0.end * 0.05 || now >= P0.end * 0.995) return null;
    const c = bestCore(cfg, metrics, key, now, priors, plans);
    if (!c) return null;
    const { fit, P } = c, end = P.end;
    // anchor: the curve passes through where the line is now (the latest few points), not where the fit says
    const tail = c.pts.slice(-4), shift = tail.reduce((a, p) => a + p[1] - fit.f(p[0]), 0) / tail.length;
    const g = (t) => fit.f(t) + shift;
    const yEnd = g(end);
    // replays: the same method from earlier moments, as far ahead (as a share of the words read then) as now
    const r = (end - now) / now, checks = [];
    if (replays) {
      const all = series(metrics, key);
      for (let i = 0; i < 8; i++) {
        const tc = now / (1 + r) * (0.55 + 0.45 * i / 7);
        if (tc < P.end * 0.04) continue;
        const cc = bestCore(cfg, metrics, key, tc, priors, plans);
        if (!cc) continue;
        const tt = cc.pts.slice(-4), sh = tt.reduce((a, p) => a + p[1] - cc.fit.f(p[0]), 0) / tt.length;
        const target = tc * (1 + r), real = actualAt(all, target);
        if (real == null || target > now * 1.001) continue;
        checks.push({ at: tc, target, said: cc.fit.f(target) + sh, real, err: (cc.fit.f(target) + sh - real) / real });
      }
    }
    // the typical miss: from the replays when there are enough, otherwise a cautious guess that grows with the
    // distance still to go (2% of the loss for every doubling of the words read)
    const rel = checks.length >= 3 ? Math.sqrt(checks.reduce((a, x) => a + x.err * x.err, 0) / checks.length) : null;
    const relFloor = 0.002 + 0.004 * r;
    const relSd = rel != null ? Math.max(rel, relFloor) : Math.max(relFloor, 0.02 * Math.log2(1 + r));
    // the cool-down drop still to come, and how unsure it is
    const coolLeft = fit.C * (P.S2end - P.S2(now));
    const seenCool = P.S2(now) / Math.max(1e-9, P.S2end);
    const coolFromPrior = !!c.prior && seenCool < 0.25;
    const sdCool = c.prior ? c.prior.sdC * (P.S2end - P.S2(now)) * (coolFromPrior ? 1 : Math.max(0.2, 1 - seenCool)) : 0;
    const jumpLeft = c.jump && now < c.jump.at ? fit.J : 0;
    const sdJump = c.jump && now < c.jump.at + 0.02 * end ? c.jump.sdJ : 0;
    const sdAt = (t) => {
      const u = Math.max(0, (t - now) / (end - now));
      const cs = P.S2end > 1e-9 ? (P.S2(t) - P.S2(now)) / Math.max(1e-9, P.S2end - P.S2(now)) : 0;
      return Math.sqrt((relSd * yEnd * u) ** 2 + (sdCool * cs) ** 2 + (c.jump && t >= c.jump.at ? sdJump : 0) ** 2);
    };
    const curve = [], band = [], n = 64;
    for (let i = 0; i <= n; i++) {
      const t = now + (end - now) * i / n;
      if (c.jump && t > c.jump.at && curve.length && curve[curve.length - 1][0] < c.jump.at) {   // the jump: a vertical step
        curve.push([c.jump.at, g(c.jump.at - 1)]); band.push([c.jump.at, g(c.jump.at - 1) - Z * sdAt(c.jump.at - 1), g(c.jump.at - 1) + Z * sdAt(c.jump.at - 1)]);
        curve.push([c.jump.at, g(c.jump.at)]); band.push([c.jump.at, g(c.jump.at) - Z * sdAt(c.jump.at), g(c.jump.at) + Z * sdAt(c.jump.at)]);
      }
      const y = g(t), s = sdAt(t);
      curve.push([t, y]); band.push([t, y - Z * s, y + Z * s]);
    }
    const sd = sdAt(end);
    return {
      key, now, end, curve, band, yEnd, lo: yEnd - Z * sd, hi: yEnd + Z * sd, sd, g,
      coolLeft, coolFromPrior, priorRuns: c.pr.n, priorFrom: c.pr.from || [], seenCool, jumpLeft, jumpAt: c.jump?.at ?? null,
      checks, relSd, lag: P.lag, fit: { E: fit.E, A: fit.A, C: fit.C, J: fit.J, alpha: fit.alpha, rmse: fit.rmse, from: c.from, n: fit.n },
      _: { cfg, metrics, law: fit, P, shift, sdCool, jump: c.jump, prior: c.prior, pts: c.pts },
    };
  }

  // A forecast stretched to a longer run of H tokens on the same recipe: what the loss would reach had the run been
  // planned that long, assuming fresh text all the way. Everything the recipe ties to the share of the run moves with
  // it: the flat learning rate lasts longer and the cool-down and text change sit in ITS last 20%; the model grows at
  // ITS 10% and 25% (so it stays small for longer); the mentor fades by ITS 30%. How much a word read by the smaller
  // model, or with the mentor, counts is learned from the real run's own early stretch (eff: beta, mu), together with
  // the curve. The line is that whole other run, from its start (it parts from the real one early when the model grows).
  // The range adds how far apart the fits the points can't rule out drift that far out, and 1% of the loss for every
  // doubling beyond the planned end. borrow: the held-out stretch, whose shape and eff the training loss reuses.
  const BETAS = [0, 0.25, 0.5, 0.75, 1, 1.5, 2], MUS = [-0.5, -0.25, 0, 0.25, 0.5, 1, 2];
  function stretch(F, H, { borrow = null } = {}) {
    if (!F || !(H > F.end * 1.01)) return null;
    const { cfg, metrics, law, P, shift, sdCool, jump, prior } = F._;
    const train = metrics.filter((x) => x.kind === "train");
    const moves = recipeMoves(cfg);
    const ja = jump ? H * (jump.at / P.end) : null;
    const planH = {}, PHof = (eff) => { const k = eff ? eff.beta + ":" + eff.mu : "-"; return planH[k] || (planH[k] = plan(cfg, train, P.lag, H, eff)); };
    let fits;                                       // [{eff, law, rmse}], best first
    if (!moves) fits = [{ eff: null, law, rmse: law.rmse }];
    else if (borrow?.fits) {                        // the training loss: the held-out stretch's eff and shape, its own levels
      const pts = F.key === "loss" ? series(metrics, "loss", F.now) : F._.pts;
      const from = Math.max(2 * (P.warmEnd || 0), 0.02 * P.end);
      fits = borrow.fits.map((b) => {
        const Pe = plan(cfg, train, P.lag, 0, b.eff);
        const lw = fitLaw(pts.filter((p) => p[0] >= from), Pe, { prior, jump, alpha: b.law.alpha });
        return lw && { eff: b.eff, law: lw, rmse: lw.rmse, Pe };
      }).filter(Boolean);
    } else {                                         // the held-out loss: learn eff and the curve together from the run's early stretch on
      const from = Math.max(2 * (P.warmEnd || 0), 0.02 * P.end);
      const pts = series(metrics, F.key, F.now).filter((p) => p[0] >= from);
      fits = [];
      for (const beta of BETAS) for (const mu of (cfg.mentor_weight ? MUS : [0])) {
        if (!cfg.grow && beta) continue;
        const eff = { beta, mu }, Pe = plan(cfg, train, P.lag, 0, eff), lw = fitLaw(pts, Pe, { prior });
        if (lw) fits.push({ eff, law: lw, rmse: lw.rmse, Pe });
      }
      fits.sort((a, b) => a.rmse - b.rmse);
      if (fits.length) {                             // keep the ones the points can't rule out (about two standard errors)
        const best = fits[0].rmse ** 2, dof = Math.max(1, pts.length - 5);
        fits = fits.filter((f) => f.rmse ** 2 <= best * (1 + 4 / dof)).slice(0, 12);
      }
    }
    if (!fits || !fits.length) return null;
    const top = fits[0], PH = PHof(top.eff);
    if (!PH) return null;
    const val = (f, PP, t) => f.law.E + f.law.A * Math.pow(PP.S1(t), -f.law.alpha) - f.law.C * PP.S2(t) + (ja != null && t >= ja ? f.law.J : 0);
    // anchor: the stretched run passes as far from the real line as the fit does where the real line is now
    const sh = moves ? (() => { const Pe = top.Pe || plan(cfg, train, P.lag, 0, top.eff), tail = (F.key === "loss" ? series(metrics, "loss", F.now) : F._.pts).slice(-4);
      return tail.reduce((a, p) => a + p[1] - (top.law.E + top.law.A * Math.pow(Pe.S1(p[0]), -top.law.alpha) - top.law.C * Pe.S2(p[0]) + (jump && p[0] >= jump.at ? top.law.J : 0)), 0) / tail.length; })() : shift;
    const g = (t) => val(top, PH, t) + sh;
    const from = moves ? Math.max(2 * (P.warmEnd || 0), 0.02 * P.end) : cfg.schedule === "wsd" ? Math.min(F.now, P.decayAt || F.now) : F.now;
    const spread = (t) => {                     // half the gap between the fits the points allow, as a share
      if (borrow && !moves) return borrow.spread(t);
      const v = [];
      for (const f of fits) { const PP = PHof(f.eff); if (PP) for (const a of f.law.alts) v.push(a(PP, t, ja)); }
      return v.length > 1 ? (Math.max(...v) - Math.min(...v)) / 2 / Math.max(1e-9, val(top, PH, t)) : 0;
    };
    const sdAt = (t) => {
      const y = g(t), u = Math.min(1, Math.max(0, (t - F.now) / (F.end - F.now)));
      const beyond = t > F.end ? 0.01 * Math.log2(t / F.end) : 0;
      return Math.sqrt((F.relSd * y * u) ** 2 + (spread(t) / 2 * y) ** 2 + (beyond * y) ** 2 + (sdCool * (t > F.now ? 1 : 0)) ** 2);
    };
    const curve = [], band = [], n = 110;
    for (let i = 0; i <= n; i++) {              // even steps on a log scale: the chart shows these on one
      const t = from * Math.pow(H / from, i / n);
      if (ja != null && t > ja && curve.length && curve[curve.length - 1][0] < ja) {
        for (const tt of [ja - 1, ja]) { const y = g(tt), sd = sdAt(tt); curve.push([ja, y]); band.push([ja, y - Z * sd, y + Z * sd]); }
      }
      const y = g(t), sd = sdAt(t);
      curve.push([t, y]); band.push([t, y - Z * sd, y + Z * sd]);
    }
    const sd = sdAt(H), yEnd = g(H);
    return { key: F.key, hypothetical: true, planned: F.end, now: F.now, end: H, from, curve, band, g, yEnd, lo: yEnd - Z * sd, hi: yEnd + Z * sd, sd,
      fit: F.fit, spread, fits, eff: top.eff, moves, jumpAt: ja, jumpLeft: ja != null && F.now < ja ? top.law.J : 0,
      minutes: clock(cfg, train, H) };
  }

  // Training time of a run planned H tokens long (tokens -> minutes), from the real run's own speeds: each word costs
  // a time that grows with the model's depth, plus the mentor's share of a pass (1/speed = a(depth) + c × mentor share,
  // fitted on the logged speeds). The longer run spends longer small (fast) and longer with the mentor (slow).
  function clock(cfg, train, H) {
    const rows = train.filter((x) => x.tok_s > 0 && x.tokens > 0.01 * (cfg.total_tokens || 1));
    if (rows.length < 10) return null;
    const full = cfg.config?.n_layer || 1;
    const depths = [...new Set(rows.map((x) => x.depth || full))].sort((a, b) => a - b);
    // columns: one per depth seen, then the mentor share
    const X = rows.map((x) => [...depths.map((d) => ((x.depth || full) === d ? 1 : 0)), x.mentor_w || 0]);
    const y = rows.map((x) => 1 / x.tok_s);
    const anyMentor = rows.some((x) => x.mentor_w > 0);
    const c = lsq(anyMentor ? X : X.map((r) => r.slice(0, -1)), y, rows.map(() => 1), anyMentor ? [] : []);
    if (!c) return null;
    const a = {}; depths.forEach((d, i) => { a[d] = c[i]; });
    const cm = anyMentor ? Math.max(0, c[depths.length]) : 0;
    const aOf = (d) => {                            // a depth never seen yet: in proportion to depth from the nearest one seen
      if (a[d] > 0) return a[d];
      const near = depths.reduce((b, x) => (Math.abs(x - d) < Math.abs(b - d) ? x : b), depths[0]);
      return a[near] * d / near;
    };
    const N = 1000, dt = H / N, mins = new Float64Array(N + 1);
    for (let i = 1; i <= N; i++) {
      const p = (i - 0.5) / N;
      mins[i] = mins[i - 1] + dt * (aOf(depthPlanned(cfg, p)) + cm * mentorPlanned(cfg, p)) / 60;
    }
    return (t) => { const x = Math.max(0, Math.min(N, t / dt)), i = Math.min(N - 1, Math.floor(x)); return mins[i] + (mins[i + 1] - mins[i]) * (x - i); };
  }

  // HellaSwag: a straight line against the run's own held-out loss, and the held-out forecast pushed through it.
  function hellaswag(cfg, metrics, valF, { now = null, ok = (x) => x.kind === "hellaswag" } = {}) {
    if (!valF) return null;
    const ev = metrics.filter((x) => x.kind === "eval" && x.val_loss != null).map((x) => [x.tokens, x.val_loss]);
    const vAt = (t) => {                                  // the held-out loss at t, between its measurements
      for (let i = 1; i < ev.length; i++) if (ev[i][0] >= t) { const [a, b] = [ev[i - 1], ev[i]]; return a[1] + (b[1] - a[1]) * (t - a[0]) / Math.max(1, b[0] - a[0]); }
      return null;
    };
    now = now ?? valF.now;
    const hs = metrics.filter((x) => ok(x) && !x.full && x.acc_norm != null && x.tokens <= now).map((x) => [x.tokens, x.acc_norm, vAt(x.tokens)]).filter((p) => p[2] != null);
    // near guessing (25%) the score sits on the floor and does not follow the loss yet: those points are left out,
    // and until at least 4 points are clear of it there is no forecast (its own points can't tell how fast it will rise)
    const off = hs.filter((p) => p[1] > 0.265);
    if (off.length < 4) return null;
    // just above the floor the score bends upward as the loss falls, so the newest points (the current slope) count
    const use = off.filter((p) => p[0] >= valF.fit.from * 0.8);
    const pts = use.length >= 10 ? use.slice(-Math.max(8, Math.ceil(use.length / 2))) : use.length >= 5 ? use : off.slice(-8);
    const c = lsq(pts.map((p) => [1, p[2]]), pts.map((p) => p[1]), pts.map(() => 1));
    if (!c || !(c[1] < 0)) return null;
    const [p0, q] = c, n = pts.length, res = Math.sqrt(pts.reduce((a, p) => a + (p[1] - p0 - q * p[2]) ** 2, 0) / Math.max(1, n - 2));
    // how sure the slope is: few points over a short stretch of loss leave it loose, and the forecast must reach
    // beyond the losses seen, so that doubt grows with the distance
    const mv = pts.reduce((a, p) => a + p[2], 0) / n, sxx = pts.reduce((a, p) => a + (p[2] - mv) ** 2, 0);
    const seQ = Math.max(res, 0.01) / Math.sqrt(Math.max(1e-9, sxx));
    const last = pts[pts.length - 1], shift = pts.slice(-3).reduce((a, p) => a + p[1] - (p0 + q * p[2]), 0) / Math.min(3, n) * 0.5;
    const hsOf = (v) => Math.max(0.25, Math.min(1, p0 + q * v + shift));
    // a stretched run starts early, at losses still on the floor: the line starts where the loss is no higher than
    // at the first point clear of the floor (before that it would only say "guessing", with a huge range)
    const vTop = Math.max(...off.map((p) => p[2]));
    const keep = ([t]) => t >= valF.now || valF.g(t) <= vTop;
    const curve = valF.curve.filter(keep).map(([t, v]) => [t, hsOf(v)]);
    const band = valF.band.filter(keep).map(([t, lo, hi]) => {
      const y = hsOf(valF.g(t)), u = Math.max(0, (t - valF.now) / (valF.end - valF.now));
      // + the line may bend (near the floor the score speeds up as the loss falls): half the rise still to come
      const bend = 0.5 * Math.abs(q * (valF.g(t) - last[2]));
      const s = Math.sqrt((q * (hi - lo) / (2 * Z)) ** 2 + (seQ * (valF.g(t) - last[2])) ** 2 + (res * u) ** 2 / 2 + (0.006 * u) ** 2 + bend ** 2);
      return [t, Math.max(0.25, y - Z * s), Math.min(1, y + Z * s)];
    });
    if (!curve.length || !band.length) return null;
    const b = band[band.length - 1], yEnd = curve[curve.length - 1][1];
    return { key: "hs", curve, band, yEnd, lo: b[1], hi: b[2], sd: (b[2] - b[1]) / (2 * Z), perLoss: q, res, n, at: last[0], now: valF.now, end: valF.end };
  }

  // ---------- learning from finished cool-downs ----------

  // For every run that went through its whole cool-down: how big the drop was, as a share of the loss, for the
  // held-out and the training loss, and how big the training loss's jump at the mix change was. These are the
  // priors for runs that have not reached theirs yet.
  function priorsFrom(runs) {
    const acc = { val: [], loss: [], jump: [], lag: [] }, from = [];
    for (const { name, cfg, metrics } of runs) {
      const train = metrics.filter((x) => x.kind === "train"), plans = planner(cfg, metrics);
      const P = plans(LAG_STEPS);
      if (!P || cfg.schedule !== "wsd" || !train.length) continue;
      const last = train[train.length - 1].tokens;
      if (last < P.end * 0.97) continue;                 // did not finish its cool-down
      const now = last, wide = { g: 0.04, sd: 1, n: 0 };
      let used = false;
      for (const key of ["val", "loss"]) {
        const c = bestCore(cfg, metrics, key, now, { val: wide, loss: wide, jump: { g: 0, sd: 1, n: 0 } }, plans);
        if (!c) continue;
        const base = c.fit.noCool(c.P.end);
        acc[key].push(c.fit.C * c.P.S2end / base);
        if (key === "val") acc.lag.push(Math.log(c.P.lag));
        if (key === "loss" && c.jump) acc.jump.push(c.fit.J / (c.fit.f(c.jump.at - 1, false)));
        used = true;
      }
      if (used) from.push(name);
    }
    const out = {};
    for (const k of ["val", "loss", "jump"]) {
      const v = acc[k];
      if (!v.length) { out[k] = { ...DEFAULT_PRIOR[k], from: [] }; continue; }
      const mean = v.reduce((a, x) => a + x, 0) / v.length;
      const sd = v.length > 1 ? Math.sqrt(v.reduce((a, x) => a + (x - mean) ** 2, 0) / (v.length - 1)) : 0;
      out[k] = { g: mean, sd: Math.max(sd, 0.35 * Math.abs(mean), 0.006), n: v.length, from };
    }
    out.lag = acc.lag.length ? Math.round(Math.exp(acc.lag.reduce((a, x) => a + x, 0) / acc.lag.length)) : LAG_STEPS;
    return out;
  }

  // ---------- checking the method on runs that already ended ----------

  // From several cut points (shares of the words the run read in the end), forecast its last value and compare.
  // Each run is checked with priors learned from the OTHER runs only, as if it were the first to get there.
  // Returns [{name, key, at, said, lo, hi, real, err, inside}].
  function backtest(runs, { cuts = [0.3, 0.5, 0.7, 0.85], old = null } = {}) {
    const out = [];
    for (const r of runs) {
      const train = r.metrics.filter((x) => x.kind === "train");
      if (!train.length) continue;
      const last = train[train.length - 1].tokens, P = plan(r.cfg, train);
      if (!P) continue;
      const priors = priorsFrom(runs.filter((x) => x !== r));
      // a run that stopped early is checked at the point it reached (its schedule still known)
      const cfgAt = { ...r.cfg, total_tokens: r.cfg.total_tokens };
      for (const key of ["val", "loss", "hs"]) {
        const fin = key === "hs" ? r.metrics.filter((x) => x.kind === "hellaswag" && x.acc_norm != null).at(-1)
          : key === "val" ? r.metrics.filter((x) => x.kind === "eval" && x.val_loss != null).at(-1) : null;
        const real = key === "loss" ? median(train.filter((x) => x.tokens >= last * 0.985).map((x) => x.loss)) : fin ? (key === "hs" ? fin.acc_norm : fin.val_loss) : null;
        if (real == null) continue;
        const endT = key === "loss" ? last : fin.tokens;
        for (const cut of cuts) {
          const now = endT * cut;
          const cf = { ...cfgAt };
          const m = r.metrics.filter((x) => x.tokens <= now || x.kind === "train");    // train rows keep the schedule
          const mm = m.map((x) => (x.kind === "train" && x.tokens > now ? { kind: "train", tokens: x.tokens, lr: x.lr } : x));
          const f = key === "hs" ? hellaswag(cf, mm, metric(cf, mm, "val", { now, priors }), { now }) : metric(cf, mm, key, { now, priors });
          if (!f) continue;
          const { y: said, lo, hi } = at(f, endT);
          const row = { name: r.name, key, at: now, cut, said, lo, hi, real, err: said - real, inside: real >= lo && real <= hi };
          if (old) { const o = old(mm, key, now, endT); if (o != null) row.oldErr = o - real; }
          out.push(row);
        }
      }
    }
    return out;
  }
  // read a forecast at any time between now and the end: {y, lo, hi}
  function at(f, t) {
    const C = f.curve, B = f.band;
    let k = 1;
    while (k < C.length - 1 && C[k][0] < t) k++;
    const u = Math.max(0, Math.min(1, (t - C[k - 1][0]) / Math.max(1, C[k][0] - C[k - 1][0])));
    const mix = (a, b) => a + (b - a) * u;
    return { y: mix(C[k - 1][1], C[k][1]), lo: mix(B[k - 1][1], B[k][1]), hi: mix(B[k - 1][2], B[k][2]) };
  }
  const median = (v) => { if (!v.length) return null; const s = [...v].sort((a, b) => a - b); return s[s.length >> 1]; };

  return { plan, etaPlanned, depthPlanned, mentorPlanned, clock, at, stretch, fitLaw, metric, hellaswag, priorsFrom, backtest, bins, Z, DEFAULT_PRIOR };
})();

if (typeof module !== "undefined") module.exports = Forecast;
