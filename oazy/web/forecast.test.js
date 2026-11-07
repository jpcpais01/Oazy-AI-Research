// Checks the forecasts (forecast.js) on simulated runs whose true curve is NOT the law the forecast fits:
//     node oazy/web/forecast.test.js
// Each simulated run has a warmup, a flat learning rate, a 1-minus-square-root cool-down over the last 20%, a text
// mix change at 80% (a step in the training loss only), noisy training loss and a HellaSwag line that follows the
// held-out loss. The forecasts are replayed from 30-95% of the way and must beat the old method and keep their range.
const F = require("./forecast.js");

function rng(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
function gauss(r) { return Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r()); }

// truth: E + B·(learning so far)^-a + kap·(the learning rate, smoothed over lagSteps steps): a noise floor that
// sinks as the rate falls, a different mechanism from the fitted law's
// HellaSwag: sits at guessing (25%) while the loss is high, then rises about in a straight line as the loss falls
const hsOf = (v) => 0.25 + 0.17 * 0.2 * Math.log1p(Math.exp((3.6 - v) / 0.2));
// recipe (optional): a model that grows ("8:0,12:0.1,20:0.25") and a mentor that fades, like Coral. Truth: a word read by
// the smaller model counts 0.3 + 0.7·(depth/full)^1.3, with the mentor ×(1 + 0.5·share) (not the forecast's own form);
// speed: 1/tok_s = depth·2e-7 + share·3e-6 (+ noise).
function simulate({ end, batch, E, B, a, kap, lagSteps, jump = -0.07, gap = 0.06, noise = 0.03, seed, recipe = false }) {
  const r = rng(seed), cfg = { total_tokens: end, lr: 1.8e-3, schedule: "wsd", decay_shape: "sqrt", batch_tokens: batch, mix: "x:1@0;y:1@0.8",
    ...(recipe ? { grow: "8:0,12:0.1,20:0.25", config: { n_layer: 20 }, mentor_weight: 0.25, mentor_until: 0.3, mentor_floor: 0, mentor_off: 0 } : {}) };
  let elapsed = 0;
  const m = [], warm = 180 * batch;
  let S1 = 0, em = 1, t = 0, step = 0, nextEval = 0, nextLog = 0, nextHs = 0;
  while (t < end) {
    const p = t / end, eta = t < warm ? (t + batch) / warm : F.etaPlanned(cfg, p);
    const d = recipe ? F.depthPlanned(cfg, p) : 20, mw = recipe ? F.mentorPlanned(cfg, p) : 0;
    S1 += eta * batch / 1e9 * (recipe ? (0.3 + 0.7 * Math.pow(d / 20, 1.3)) * (1 + 0.5 * mw) : 1);
    const tps = 1 / (d * 2e-7 + mw * 3e-6);
    elapsed += batch / tps;
    const k = Math.exp(-1 / lagSteps); em = em * k + eta * (1 - k);
    const val = E + B * Math.pow(Math.max(S1 / 10, 1e-4), -a) + kap * em;
    t += batch; step++;
    if (t >= nextLog) { nextLog += 20e6; m.push({ kind: "train", tokens: t, step, elapsed, tok_s: tps * (1 + 0.02 * gauss(r)), ...(recipe ? { depth: d, mentor_w: mw } : {}), lr: eta * 1.8e-3, loss: val + gap + (p >= 0.8 ? jump : 0) + noise * gauss(r) }); }
    if (t >= nextEval) { nextEval += 250e6; m.push({ kind: "eval", tokens: t, step, val_loss: val + 0.002 * gauss(r) }); }
    if (t >= nextHs) { nextHs += 500e6; m.push({ kind: "hellaswag", tokens: t, acc_norm: hsOf(val) + 0.012 * gauss(r), sample: "spread" }); }
  }
  const last = m.filter((x) => x.kind === "eval").at(-1).val_loss;
  m.push({ kind: "hellaswag", tokens: t, acc_norm: hsOf(last), full: true });
  return { cfg, metrics: m, elapsed };
}

// the old dashboard forecast: y = a + b·tokens^-c fitted to the newest 30% of the points, anchored at the last one
function oldForecast(metrics, key, now, endT) {
  const pts = key === "loss" ? metrics.filter((x) => x.kind === "train" && x.tokens <= now).map((x) => [x.tokens, x.loss])
    : key === "val" ? metrics.filter((x) => x.kind === "eval" && x.tokens <= now).map((x) => [x.tokens, x.val_loss]) : null;
  if (!pts) return null;
  let use = pts.slice(-Math.max(8, Math.ceil(pts.length * 0.3)));
  if (use.length > 40) { const n = Math.ceil(use.length / 40); use = use.filter((_, i) => i % n === 0); }
  let best = null;
  const xs = use.at(-1)[0];
  for (let i = 0; i <= 80; i++) {
    const c = 0.02 * Math.pow(100, i / 80), u = use.map((p) => Math.pow(p[0] / xs, -c));
    const n = u.length, su = u.reduce((s, v) => s + v, 0), sy = use.reduce((s, p) => s + p[1], 0);
    const suu = u.reduce((s, v) => s + v * v, 0), suy = u.reduce((s, v, j) => s + v * use[j][1], 0), d = n * suu - su * su;
    if (Math.abs(d) < 1e-12) continue;
    const b = (n * suy - su * sy) / d, a0 = (sy - b * su) / n;
    const e = use.reduce((s, p, j) => s + (p[1] - a0 - b * u[j]) ** 2, 0);
    if (!best || e < best.e) best = { a0, b, c, e };
  }
  if (!best) return null;
  const f = (x) => best.a0 + best.b * Math.pow(x / xs, -best.c);
  return f(endT) + (use.at(-1)[1] - f(use.at(-1)[0]));
}

const runs = [];
for (let i = 0; i < 6; i++) runs.push({ name: "sim" + i, ...simulate({ seed: i + 1, end: (6 + 2 * i) * 1e9, batch: (i % 2 ? 1 : 0.5) * 2 ** 20,
  E: 1.6 + 0.1 * i, B: 0.8 + 0.1 * i, a: 0.35 + 0.05 * i, kap: 0.08 + 0.02 * i, lagSteps: [100, 150, 300, 500, 200, 400][i] }) });

const t0 = Date.now();
const rows = F.backtest(runs, { cuts: [0.3, 0.5, 0.7, 0.85, 0.95], old: oldForecast });
let failed = 0;
const check = (ok, msg) => { console.log((ok ? "ok   " : "FAIL ") + msg); if (!ok) failed++; };
for (const [key, maxMiss] of [["val", 0.03], ["loss", 0.05], ["hs", 0.02]]) {
  const R = rows.filter((x) => x.key === key), mae = (v) => v.reduce((s, x) => s + Math.abs(x), 0) / v.length;
  const miss = mae(R.map((x) => x.err)), cover = R.filter((x) => x.inside).length / R.length;
  const O = R.filter((x) => x.oldErr != null), old = O.length ? mae(O.map((x) => x.oldErr)) : null;
  console.log(`${key}: ${R.length} forecasts, typical miss ${miss.toFixed(4)}${old != null ? `, old method ${old.toFixed(4)}` : ""}, range held ${(cover * 100).toFixed(0)}%`);
  check(R.length >= 20, `${key}: forecasts were made`);
  check(miss < maxMiss, `${key}: typical miss under ${maxMiss}`);
  if (old != null) check(miss < old, `${key}: better than the old method`);
  const need = key === "hs" ? 0.65 : 0.7;     // HellaSwag bends upward just above the 25% floor, which a line only half sees
  check(cover >= need, `${key}: the range holds at least ${need * 100}% of the time`);
}
// ---- the 100B / 1T stretch: from 60% of a 13B run of a growing, mentored model, forecast whole runs planned 4x and 8x
// longer (simulated with the same truth), at their end and halfway, and their training time
let inStretch = 0, nStretch = 0, timeErr = [];
const t1 = Date.now();
for (const [i, [E, B, a, kap, lag]] of [[1.6, 0.8, 0.35, 0.08, 100], [1.9, 1.1, 0.5, 0.14, 300], [1.7, 1.0, 0.42, 0.1, 500]].entries()) {
  const base = { batch: 2 ** 20, E, B, a, kap, lagSteps: lag, seed: 11 + i, recipe: true };
  const real = simulate({ ...base, end: 13e9 }), now = 13e9 * 0.6;
  const cut = real.metrics.filter((x) => x.tokens <= now || x.kind === "train").map((x) => (x.kind === "train" && x.tokens > now ? { kind: "train", tokens: x.tokens, lr: x.lr } : x));
  const fv = F.metric(real.cfg, cut, "val", { now });
  for (const H of [52e9, 104e9]) {
    const st = F.stretch(fv, H), long = simulate({ ...base, end: H });
    const ev = long.metrics.filter((x) => x.kind === "eval");
    for (const at of [0.5, 1]) {
      const truth = ev.reduce((b, x) => (Math.abs(x.tokens - at * H) < Math.abs(b.tokens - at * H) ? x : b)).val_loss;
      const f = F.at(st, at * H), inside = truth >= f.lo && truth <= f.hi;
      nStretch++; inStretch += inside;
      console.log(`stretch sim${i} to ${H / 1e9}B, at ${at * 100}%: said ${f.y.toFixed(3)} (${f.lo.toFixed(3)}–${f.hi.toFixed(3)}), truth ${truth.toFixed(3)}${inside ? "" : "  OUTSIDE"}`);
    }
    if (st.minutes) timeErr.push(st.minutes(H) * 60 / long.elapsed - 1);
  }
}
check(inStretch >= 0.75 * nStretch, `stretch: the truth inside the range at least 3 times in 4 (${inStretch} of ${nStretch})`);
check(timeErr.length && timeErr.every((e) => Math.abs(e) < 0.05), `stretch: training time within 5% (${timeErr.map((e) => (e * 100).toFixed(1) + "%").join(", ")})`);
check(Date.now() - t1 < 8000, `stretch fast enough (${Date.now() - t1} ms for ${nStretch / 2} stretches)`);
check(Date.now() - t0 < 5000, `fast enough for the page (${Date.now() - t0} ms for ${rows.length} forecasts)`);
if (process.env.SHOW) for (const x of rows.filter((x) => x.key === process.env.SHOW)) console.log(x.name, x.cut, x.said.toFixed(3), x.real.toFixed(3), x.lo.toFixed(3), x.hi.toFixed(3));
process.exit(failed ? 1 : 0);
