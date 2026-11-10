// The visual training dashboard (dashboard.html; data from oazy/dashboard.py: /api/overview and /api/run/<name>/series).
// What is live comes from the server, which looks for the trainer process itself: a run that still says "running"
// but whose trainer is gone shows as CUT OFF, never as live.
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
// token counts always in whole millions, "1,240M" (João's rule)
const fmtM = (v) => Math.round((v || 0) / 1e6).toLocaleString("en-US") + "M";
const fmtL = (v) => v == null ? "–" : v.toFixed(3);
const pct = (v, d = 1) => v == null ? "–" : (v * 100).toFixed(d) + "%";
const fmtDur = (s) => s == null ? "–" : s < 60 ? "under a minute" : s < 3600 ? Math.round(s / 60) + " min"
  : s < 86400 ? `${Math.floor(s / 3600)} h ${String(Math.round((s % 3600) / 60)).padStart(2, "0")} min` : (s / 86400).toFixed(1) + " days";
const ago = (s) => s < 90 ? "just now" : s < 3600 ? Math.round(s / 60) + " min ago" : s < 86400 ? Math.round(s / 3600) + " h ago" : Math.round(s / 86400) + " d ago";

const NICE = { bee6: "Bee6", logos: "Logos1", bee5: "Bee5", bee4: "Bee4", bee3: "Bee3", bee2: "Bee2", bee: "Bee", seed: "Seed", coral: "Coral",
  coralmini: "Coral Mini", plain100: "Plain-100M", plain100f: "Plain-100F", alphastar: "Pegasus Alpha Star", pegasusbeta: "Pegasus Beta",
  pegasus: "Pegasus 1", pegasusmini: "Pegasus 1 Mini", pegasus2: "Pegasus 2 Mini", smol: "SmolLM2 (their recipe)", smolbee: "SmolLM2 (our recipe)",
  gpt: "GPT-2", memtrix: "Memtrix", tide: "Tide", fern: "Fern", fern2: "Fern2", fern3: "Fern3", ember: "Ember", frog: "Frog" };
const NOT_NEXT = new Set(["fern", "fern2", "fern3"]);
const LIVE = new Set(["live", "paused", "finishing", "starting"]);
const BADGE = { live: "live", paused: "paused", finishing: "final tests", starting: "starting", done: "done", cut: "cut off", stopped: "stopped" };
const badge = (lv) => `<span class="st ${lv}"><i></i>${BADGE[lv] || lv}</span>`;
const HS_GPT2 = 0.3114;          // OpenAI GPT-2 small, HellaSwag per letter (the line to beat)

// one colour per run, kept while it is on the page: the lab's accents (violet, amber, cyan, pink, lime), checked for
// colour-blind separation on the black stage (lime/amber is the closest pair, so every line also carries its value at
// its end); teal is kept for "live", so it is not in here. At most 5 runs on the charts at once.
const PAL = ["#8f7bff", "#ffb347", "#4fd8ff", "#ff5fae", "#b6f36b"];
let SLOT = {};
try { SLOT = JSON.parse(localStorage.getItem("oazy.v.slot") || "{}"); } catch (e) { SLOT = {}; }
function paint(names) {
  const used = new Set();
  // a slot from an older, longer palette (saved in this browser) would have no colour at all: black on black
  for (const n of Object.keys(SLOT)) if (!(SLOT[n] >= 0 && SLOT[n] < PAL.length)) delete SLOT[n];
  for (const n of names) if (SLOT[n] != null && !used.has(SLOT[n])) used.add(SLOT[n]); else delete SLOT[n];
  for (const n of names) if (SLOT[n] == null) {
    let h = 0; for (const c of n) h = (h * 31 + c.charCodeAt(0)) >>> 0;
    let s = h % PAL.length; for (let k = 0; k < PAL.length && used.has(s); k++) s = (s + 1) % PAL.length;
    SLOT[n] = s; used.add(s);
  }
  try { localStorage.setItem("oazy.v.slot", JSON.stringify(SLOT)); } catch (e) { /* private window */ }
}
const colorOf = (n) => PAL[SLOT[n]] || PAL[0];

let RUNS = [], WORLD = [], SER = {}, PICK = null, FILTER = "all", MORE = false, NOW = Date.now() / 1000;
try { PICK = JSON.parse(localStorage.getItem("oazy.v.pick") || "null"); } catch (e) { PICK = null; }
const savePick = () => { try { localStorage.setItem("oazy.v.pick", JSON.stringify(PICK)); } catch (e) { /* */ } };
const runOf = (n) => RUNS.find((r) => r.name === n);
const ours = () => RUNS.filter((r) => !r.reference);
// published models measured on the pod (SmolLM2, GPT-2): no tile, a labelled line on the tests they took
const REF_NAMES = { "smollm2-135m": "SmolLM2-135M", "smollm2-360m": "SmolLM2-360M", "smollm2-1.7b": "SmolLM2-1.7B", "openai-gpt2": "OpenAI GPT-2" };
const yardsticks = (k) => RUNS.filter((r) => r.reference && r[k] != null).map((r) => ({ y: r[k], label: REF_NAMES[r.name] || r.name, always: true }));
const isLive = (r) => LIVE.has(r.live);
const nice = (r) => NICE[r.model] || r.model || "";
const fracOf = (r) => r.total ? Math.min(1, (r.tokens || 0) / r.total) : (r.live === "done" ? 1 : 0);

// the race: the live runs and their siblings (same ending of the name: bee6-teach-emerald-2b, logos-emerald-2b
// and bee6-emerald-2b all end in "emerald-2b"); with nothing live, the latest run's
const groupOf = (n) => { const p = n.split("-"); return p.length >= 3 ? p.slice(-2).join("-") : p.length === 2 ? p[1] : n; };
function autoSet() {
  const R = ours();
  let seeds = R.filter(isLive);
  if (!seeds.length && R.length) seeds = [R[0]];
  const groups = new Set(seeds.map((r) => groupOf(r.name)));
  const out = R.filter((r) => seeds.includes(r) || groups.has(groupOf(r.name)));
  return { names: out.slice(0, PAL.length).map((r) => r.name), groups: [...groups] };
}
function compareSet() {
  if (PICK) return PICK.filter((n) => runOf(n));
  return autoSet().names;
}

async function getJSON(u) { const r = await fetch(u, { cache: "no-store" }); if (!r.ok) throw new Error(u + " " + r.status); return r.json(); }

async function refresh() {
  try {
    const o = await getJSON("api/overview");
    RUNS = o.runs || []; WORLD = o.world || []; NOW = o.now || Date.now() / 1000;
    const want = new Set([...compareSet(), ...RUNS.filter(isLive).map((r) => r.name)]);
    await Promise.all([...want].map(async (n) => {
      const r = runOf(n);
      if (SER[n] && SER[n].u === r.updated) return;
      try { SER[n] = { u: r.updated, d: await getJSON(`api/run/${encodeURIComponent(n)}/series`) }; } catch (e) { /* keep the old */ }
    }));
    await loadWriting(compareSet());
    const live = RUNS.filter((r) => isLive(r) && !r.reference).length;
    $("pulse").className = "pill" + (live ? " live" : "");
    $("pulse").lastElementChild.textContent = live ? `${live} live` : "nothing live";
  } catch (e) {
    $("pulse").className = "pill err";
    $("pulse").lastElementChild.textContent = "can't reach the pod";
  }
  draw();
}

function draw() {
  const set = compareSet();
  paint(set);
  drawNow(); drawWorld(); drawRace(set); drawTiles(set); drawWriting(set);
}

// ---------------------------------------------------------------- what they write
// one prompt at a time (the trainer's four, oazy/train.py SAMPLE_PROMPTS); under it, each charted run's newest writing,
// three lines long until tapped
let WRITE = {}, PROMPT = null, OPEN = new Set();
try { PROMPT = localStorage.getItem("oazy.v.prompt"); } catch (e) { PROMPT = null; }
const SHORT = (p) => { const w = p.replace(/[,.]+$/, "").split(" "); return w.length > 4 ? w.slice(0, 4).join(" ") + "…" : w.join(" "); };
async function loadWriting(set) {
  await Promise.all(set.map(async (n) => {
    const r = runOf(n);
    if (!r || (WRITE[n] && WRITE[n].u === r.updated)) return;
    try { WRITE[n] = { u: r.updated, d: await getJSON(`api/run/${encodeURIComponent(n)}/writing`) }; } catch (e) { /* keep the old */ }
  }));
}
function drawWriting(set) {
  const prompts = [];
  for (const n of set) for (const x of (WRITE[n] || {}).d || []) if (!prompts.includes(x.prompt)) prompts.push(x.prompt);
  if (!prompts.length) {
    $("prompts").hidden = true;
    $("writeBox").innerHTML = `<p class="vhint">${set.length ? "No writing yet: runs write a little at each check." : "Tap a run above to see what it writes."}</p>`;
    return;
  }
  if (!prompts.includes(PROMPT)) PROMPT = prompts[0];
  $("prompts").hidden = false;
  $("prompts").innerHTML = prompts.map((p) => `<button data-p="${esc(p)}" class="${p === PROMPT ? "on" : ""}" title="${esc(p)}">${esc(SHORT(p))}</button>`).join("");
  const rows = set.map((n) => [n, ((WRITE[n] || {}).d || []).find((x) => x.prompt === PROMPT)]).filter(([, x]) => x);
  const body = (t) => { t = (t || "").trim(); return t.startsWith(PROMPT) ? `<b>${esc(PROMPT)}</b>${esc(t.slice(PROMPT.length))}` : esc(t); };
  $("writeBox").innerHTML = rows.map(([n, x]) =>
    `<button class="wrow${OPEN.has(n) ? " open" : ""}" data-w="${esc(n)}" style="--c:${colorOf(n)}">
      <span class="who"><i></i><b>${esc(n)}</b><span>after ${fmtM(x.tokens)} tokens</span></span>
      <span class="txt">${body(x.text)}</span></button>`).join("");
}

// ---------------------------------------------------------------- the live runs
function lastTwo(pts) { const a = pts || []; return [a[a.length - 1], a[a.length - 2]]; }
function numTile(k, pts, fmt, lowerBetter, extra) {
  const [a, b] = lastTwo(pts);
  let d = extra || "";
  if (a && b && !extra) {
    const diff = a[1] - b[1], good = lowerBetter ? diff < 0 : diff > 0;
    const s = lowerBetter ? Math.abs(diff).toFixed(3) : (Math.abs(diff) * 100).toFixed(1);
    d = Math.abs(diff) < 1e-9 ? "same as last" : `<span class="${good ? "up" : "down"}">${diff < 0 ? "▼" : "▲"} ${s}</span> since last`;
  } else if (a && !extra) d = "first check";
  return `<div class="num-t"><div class="k">${k}</div><div class="v${a ? "" : " na"}">${a ? fmt(a[1]) : "–"}</div><div class="d">${d || "not checked yet"}</div></div>`;
}
function phaseOf(r) {
  const t = r.tokens || 0, T = r.turns || [];
  const cool = T.find((x) => x.short === "cool-down"), wd = T.find((x) => x.short === "world drops");
  const share = r.share != null ? ` · world text ${pct(r.share, r.share < 0.1 ? 1 : 0).replace(".0%", "%")}` : "";
  let now;
  if (r.live === "finishing") now = "<b>Final tests</b> · full HellaSwag and the rest, a few minutes";
  else if (cool && t >= cool.at) now = `<b>Cool-down</b> · learning rate falling to 0${share}`;
  else if (wd && t < wd.at) now = `<b>World-heavy start</b>${share}`;
  else now = `<b>Main stretch</b>${share}`;
  const next = T.find((x) => x.at > t);
  const nx = next && r.tok_s && r.live === "live" ? ` · next: ${next.short} in ${fmtDur((next.at - t) / r.tok_s)}` : "";
  return now + nx;
}
function track(r, color) {
  const T = r.total || 0, f = fracOf(r), turns = (r.turns || []).filter((x) => T && x.at > 0 && x.at < T);
  const wd = turns.find((x) => x.short === "world drops"), cool = turns.find((x) => x.short === "cool-down");
  const p = (x) => (x / T * 100).toFixed(2) + "%";
  const zones = (wd ? `<div class="zone world" style="left:0;width:${p(wd.at)}"></div>` : "")
    + (cool ? `<div class="zone cool" style="left:${p(cool.at)};right:0"></div>` : "");
  const marks = turns.map((x) => {
    const past = (r.tokens || 0) >= x.at, fr = x.at / T;
    const edge = fr < 0.12 ? " edge-l" : fr > 0.88 ? " edge-r" : "";
    return `<div class="turn${past ? " past" : ""}" style="left:${p(x.at)}" title="${esc(x.label)}"></div>
      <div class="tlabel${past ? " past" : ""}${edge}" style="left:${p(x.at)}">${esc(x.short)} · ${fmtM(x.at)}</div>`;
  }).join("");
  return `<div class="track${r.live === "live" ? " on" : ""}" style="--acc:${color}"><div class="rail-t">${zones}<div class="fill" style="width:${(f * 100).toFixed(2)}%"></div></div>
    ${marks}<div class="bead" style="left:${(f * 100).toFixed(2)}%"></div></div>`;
}
const CTL = {};          // run -> {armed: time} (end needs a second tap) or {sent: action}
function controls(r) {
  const c = CTL[r.name] || {};
  if (r.live === "finishing") return "";
  const end = c.armed && Date.now() - c.armed < 4000
    ? `<button class="btn sure" data-act="end" data-run="${esc(r.name)}">Tap again to end</button>`
    : `<button class="btn danger" data-act="arm" data-run="${esc(r.name)}">End now</button>`;
  const pr = r.live === "paused" ? `<button class="btn primary" data-act="resume" data-run="${esc(r.name)}">Resume</button>`
    : `<button class="btn" data-act="pause" data-run="${esc(r.name)}">Pause</button>`;
  const note = c.sent ? `<span class="note">${c.sent === "pause" ? "pausing at the next step…" : c.sent === "resume" ? "resuming…" : "ending: saving, then the final tests…"}</span>` : "";
  return pr + end + note;
}
function liveCard(r) {
  const s = (SER[r.name] || {}).d || {}, color = colorOf(r.name), f = fracOf(r);
  const left = r.live === "live" && r.eta_s != null ? `<b>${fmtDur(r.eta_s)}</b> left` : r.live === "paused" ? "paused, GPU idle" : BADGE[r.live];
  const hsFull = (s.hs || []).filter((x) => x[2]);
  // the name in the landing page's way: the family plain, the rest in the run's accent ("logos" + "-emerald-2b")
  const cut = r.name.indexOf("-"), title = cut > 0 ? `${esc(r.name.slice(0, cut))}<span class="acc">${esc(r.name.slice(cut))}</span>` : esc(r.name);
  const chips = [[nice(r), true], r.params ? [Math.round(r.params / 1e6) + "M params"] : null, [NOT_NEXT.has(r.model) ? "not next-token" : "next-token"],
    r.gpu ? [r.gpu.replace(/^NVIDIA\s+/, "")] : null].filter(Boolean);
  return `<article class="card live-card" style="--acc:${color}">
    <div class="lc-top"><div class="lc-name">${title}</div>${badge(r.live)}</div>
    <div class="chips">${chips.map(([t, a]) => `<span class="${a ? "acc" : ""}">${esc(t)}</span>`).join("")}</div>
    <div class="lc-big"><span class="pct">${Math.floor(f * 100)}<small>%</small></span>
      <span class="left">${fmtM(r.tokens)} of ${fmtM(r.total)} tokens · ${left}</span></div>
    ${track(r, color)}
    <span class="phase">${phaseOf(r)}</span>
    <div class="nums">${numTile("Held-out loss", s.val, fmtL, true)}
      ${numTile("HellaSwag", s.hs, (v) => pct(v), false, hsFull.length ? "final, all 10,042" : "")}
      ${numTile("LogT", s.logt, (v) => pct(v), false)}
      ${numTile("World secret", s.world, (v) => pct(v), false)}</div>
    <div class="lc-ctl">${controls(r)}<span class="meta">${r.tok_s ? Math.round(r.tok_s / 1e3) + "k tok/s" : ""}${r.elapsed ? " · " + fmtDur(r.elapsed) + " in" : ""}</span></div>
  </article>`;
}
function drawNow() {
  const R = ours(), live = R.filter(isLive);
  const cut = R.filter((r) => r.live === "cut" && r.age < 86400);
  let h = cut.map((r) => `<div class="warnbar">${badge("cut")}<span><b>${esc(r.name)}</b> stopped ${ago(r.age)} at ${fmtM(r.tokens)} of ${fmtM(r.total)}, unfinished. The same command resumes it.</span></div>`).join("");
  if (live.length) h += live.map(liveCard).join("");
  else {
    const last = R[0];
    h += `<div class="card idle"><img src="icon.svg" alt=""><div><div class="t">Nothing is training right now</div>
      <div class="s">${last ? `Last: <b>${esc(last.name)}</b>, ${BADGE[last.live]} ${ago(last.age)}.` : "Start a run and it shows up here by itself."}</div></div></div>`;
  }
  $("nowBox").innerHTML = h;
  // the pod at a glance, under the head's title; the hero's light slows right down when nothing trains
  const L0 = live.find((r) => r.live === "live") || live[0], done = R.filter((r) => r.live === "done").length;
  const cell = (k, v, h2) => `<div><div class="k">${k}</div><div class="v">${v}</div><div class="h">${h2 || "&nbsp;"}</div></div>`;
  $("pod").innerHTML = cell("Training", live.length ? `${live.length}<small>${live.length > 1 ? "runs" : "run"}</small>` : "–", L0 ? esc(L0.name) : "nothing right now")
    + cell("Progress", L0 ? `${Math.floor(fracOf(L0) * 100)}<small>%</small>` : "–", L0 ? `${fmtM(L0.tokens)} of ${fmtM(L0.total)}` : "")
    + cell("Finishes in", L0 && L0.eta_s != null && L0.live === "live" ? fmtDur(L0.eta_s) : "–", L0 && L0.tok_s ? `${Math.round(L0.tok_s / 1e3)}k tokens/s` : "")
    + cell("Runs", `${R.length}`, `${done} finished`);
  try { LAB.setPace(live.length ? 1 : 0.15); } catch (e) { /* no light */ }
}
function drawWorld() {
  const live = WORLD.filter((w) => w.phase !== "done" && NOW - (w.updated || 0) < 600);
  $("worldNow").hidden = !live.length;
  $("worldNow").innerHTML = live.length ? `<div class="wstrip">${live.map((w) => `<a href="world#${encodeURIComponent(w.name)}">${badge("live")}<span>${esc(w.name)}</span><span class="k">world race · ${esc(w.phase)} →</span></a>`).join("")}</div>` : "";
}

document.addEventListener("click", async (e) => {
  const b = e.target.closest("[data-act]");
  if (b) {
    const n = b.dataset.run, act = b.dataset.act;
    if (act === "arm") { CTL[n] = { armed: Date.now() }; drawNow(); setTimeout(drawNow, 4100); return; }
    try {
      const r = await fetch(`api/run/${encodeURIComponent(n)}/control`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action: act }) });
      if (!r.ok) throw new Error();
      CTL[n] = { sent: act };
    } catch (err) { CTL[n] = {}; }
    drawNow(); setTimeout(refresh, 3000);
    return;
  }
  const t = e.target.closest(".tile");
  if (t) {
    const n = t.dataset.run, cur = compareSet();
    PICK = cur.includes(n) ? cur.filter((x) => x !== n) : [...cur, n].slice(-PAL.length);
    savePick(); refresh(); return;
  }
  const pb = e.target.closest("#prompts button");
  if (pb) { PROMPT = pb.dataset.p; try { localStorage.setItem("oazy.v.prompt", PROMPT); } catch (err) { /* */ } drawWriting(compareSet()); return; }
  const w = e.target.closest(".wrow");
  if (w) { const n = w.dataset.w; OPEN.has(n) ? OPEN.delete(n) : OPEN.add(n); w.classList.toggle("open"); return; }
  const f = e.target.closest("#filter button");
  if (f) { FILTER = f.dataset.f; MORE = false; document.querySelectorAll("#filter button").forEach((x) => x.classList.toggle("on", x === f)); drawTiles(compareSet()); return; }
  if (e.target.closest("#raceReset")) { PICK = null; savePick(); refresh(); return; }
  if (e.target.closest(".vmore")) { MORE = true; drawTiles(compareSet()); }
});

// ---------------------------------------------------------------- charts (SVG, no libraries)
function ticks(lo, hi, n) {
  const span = hi - lo || 1, s0 = Math.pow(10, Math.floor(Math.log10(span / n)));
  const step = s0 * ([1, 2, 2.5, 5, 10].find((m) => span / (s0 * m) <= n * 1.3) || 10);
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  out.step = step; return out;
}
const dec = (step) => { for (let d = 0; d < 5; d++) if (Math.abs(Math.round(step * 10 ** d) - step * 10 ** d) < 1e-6) return d; return 5; };
const HOV = {};    // chart id -> its hover state, so a refresh keeps the readout working

// opt: series [{name, color, pts [[x, y, final?]], faint [[x, y]], live}], turns [{at, short}], xmax, fmt, refs [{y, label}],
// crop (leave the first stretch out of the y range: early losses are huge), pctAxis
function lineChart(id, opt) {
  const el = $(id), W = el.clientWidth, H = el.clientHeight;
  // the check at 0 tokens is an untrained model (loss ~10.8, chance-level tests): it only squashes the picture
  const S = opt.series.map((s) => ({ ...s, pts: s.pts.filter((p) => p[0] > 0), faint: (s.faint || []).filter((p) => p[0] > 0) }))
    .filter((s) => s.pts.length || s.faint.length);
  if (!S.length) { el.innerHTML = `<div class="empty">${opt.empty || "No checks yet"}</div>`; HOV[id] = null; return; }
  const m = { l: 44, r: 52, t: 20, b: 24 };
  const xmax = opt.xmax || Math.max(...S.flatMap((s) => s.pts.concat(s.faint || []).map((p) => p[0])));
  const cut = opt.crop ? opt.crop * xmax : -1;
  let ys = S.flatMap((s) => s.pts.filter((p) => p[0] >= cut).map((p) => p[1]));
  if (!ys.length) ys = S.flatMap((s) => s.pts.map((p) => p[1]));
  if (opt.faintInRange) ys = ys.concat(S.flatMap((s) => (s.faint || []).filter((p) => p[0] >= Math.max(cut, 0.25 * xmax)).map((p) => p[1])));
  for (const r of opt.refs || []) if (r.always) ys.push(r.y);
  if (!ys.length) ys = S.flatMap((s) => (s.faint || []).map((p) => p[1]));
  let lo = Math.min(...ys), hi = Math.max(...ys);
  const pad = Math.max((hi - lo) * 0.12, opt.minSpan || 0.01);
  lo -= pad; hi += pad;
  if (opt.floor0 != null) lo = Math.max(lo, opt.floor0);
  const X = (x) => m.l + (x / xmax) * (W - m.l - m.r), Y = (y) => m.t + (1 - (y - lo) / (hi - lo)) * (H - m.t - m.b);
  const yt = ticks(lo, hi, 4), yd = dec(yt.step);
  const xt = ticks(0, xmax, W < 420 ? 3 : 5).filter((v) => v > 0);
  const fy = opt.pctAxis ? (v) => (v * 100).toFixed(Math.max(0, yd - 2)) + "%" : (v) => v.toFixed(yd);
  let g = `<g class="grid">${yt.map((v) => `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${m.l - 7}" y="${Y(v) + 3.5}" text-anchor="end">${fy(v)}</text>`).join("")}
    ${xt.map((v) => `<text x="${X(v)}" y="${H - 6}" text-anchor="middle">${fmtM(v)}</text>`).join("")}</g>`;
  for (const t of opt.turns || []) {
    if (!(t.at > 0 && t.at < xmax)) continue;
    const x = X(t.at), anchor = x > W - m.r - 40 ? "end" : x < m.l + 40 ? "start" : "middle";
    g += `<line class="turnl" x1="${x}" x2="${x}" y1="${m.t - 4}" y2="${H - m.b}"/><text class="turnt" x="${x}" y="${m.t - 8}" text-anchor="${anchor}">${esc(t.short)}</text>`;
  }
  const used = [];                                 // label heights already taken: a close second line labels above itself
  for (const r of opt.refs || []) {
    if (r.y < lo || r.y > hi) continue;
    const below = Y(r.y) + 13, ty = used.some((u) => Math.abs(u - below) < 13) ? Y(r.y) - 5 : below;
    used.push(ty);
    g += `<line class="refl" x1="${m.l}" x2="${W - m.r}" y1="${Y(r.y)}" y2="${Y(r.y)}"/><text class="reft" x="${W - m.r - 4}" y="${ty}" text-anchor="end">${esc(r.label)}</text>`;
  }
  const path = (pts) => pts.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(Math.min(hi, Math.max(lo, p[1]))).toFixed(1)}`).join("");
  let lines = "", dots = "", ends = [];
  for (const s of S) {
    // the training line starts where it comes into the picture (no wall dropping from far above the chart)
    if (s.faint.length) { const k = s.faint.findIndex((p) => p[1] >= lo && p[1] <= hi); if (k >= 0) s.faint = s.faint.slice(k); else s.faint = []; }
    if (s.faint.length) lines += `<path d="${path(s.faint)}" fill="none" stroke="${s.color}" stroke-width="1.1" stroke-opacity=".3" stroke-linejoin="round"/>`;
    if (!s.pts.length) continue;
    lines += `<path d="${path(s.pts)}" fill="none" stroke="${s.color}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>`;
    if (s.pts.length < 30) dots += s.pts.filter((p) => !p[2]).map((p) => `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="2.6" fill="${s.color}"/>`).join("");
    dots += s.pts.filter((p) => p[2]).map((p) => `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="5" fill="#050507" stroke="${s.color}" stroke-width="2.5"><title>final score</title></circle>`).join("");
    const L = s.pts[s.pts.length - 1];
    if (s.live) dots += `<circle class="pulse" cx="${X(L[0])}" cy="${Y(L[1])}" r="4" fill="${s.color}"/><circle cx="${X(L[0])}" cy="${Y(L[1])}" r="4" fill="${s.color}" stroke="#050507" stroke-width="1.5"/>`;
    ends.push({ y: Y(Math.min(hi, Math.max(lo, L[1]))), x: X(L[0]), v: opt.fmt(L[1]), c: s.color });
  }
  // the newest value written just after the end of each line (no line out to the edge: that read as a forecast),
  // nudged apart so two never sit on top of each other
  ends.sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 14) ends[i].y = ends[i - 1].y + 14;
  const endT = ends.map((e) => { const tx = Math.min(e.x + 10, W - m.r + 6);
    return `<text class="endv" x="${tx}" y="${e.y + 4}" fill="${e.c}">${esc(e.v)}</text>`; }).join("");
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(opt.label || "")}"><defs><clipPath id="${id}clip"><rect x="${m.l - 6}" y="${m.t - 6}" width="${W - m.l - m.r + 14}" height="${H - m.t - m.b + 12}"/></clipPath></defs>
    ${g}<g clip-path="url(#${id}clip)">${lines}${dots}</g>${endT}<g class="hov"></g><rect x="${m.l}" y="0" width="${W - m.l - m.r}" height="${H}" fill="transparent" class="hit"/></svg>`;
  HOV[id] = { S, X, Y, xmax, m, W, H, fmt: opt.fmt, inv: (px) => (px - m.l) / (W - m.l - m.r) * xmax, lo, hi };
  hookHover(id);
}
function hookHover(id) {
  const el = $(id), svg = el.querySelector("svg"), hit = svg.querySelector(".hit"), hov = svg.querySelector(".hov"), tip = $("tip");
  let hideT = null;
  const show = (ev) => {
    const h = HOV[id]; if (!h) return;
    const box = svg.getBoundingClientRect(), px = (ev.clientX - box.left) * (h.W / box.width);
    const x = Math.max(0, Math.min(h.xmax, h.inv(px)));
    const rows = [];
    let mark = `<line class="cross" x1="${h.X(x)}" x2="${h.X(x)}" y1="${h.m.t}" y2="${h.H - h.m.b}"/>`;
    for (const s of h.S) {
      const P = s.pts.length ? s.pts : s.faint;
      if (!P || !P.length || P[0][0] > x + h.xmax * 0.02) continue;
      let best = P[0]; for (const p of P) if (Math.abs(p[0] - x) < Math.abs(best[0] - x)) best = p;
      if (Math.abs(best[0] - x) > h.xmax * 0.12) continue;
      rows.push(`<div class="r"><i style="background:${s.color}"></i><span>${esc(s.name)}</span><b>${esc(h.fmt(best[1]))}${best[2] ? " ✓" : ""}</b></div>`);
      mark += `<circle class="tipdot" cx="${h.X(best[0])}" cy="${h.Y(Math.min(h.hi, Math.max(h.lo, best[1])))}" r="4.5" fill="${s.color}"/>`;
    }
    hov.innerHTML = mark;
    if (!rows.length) { tip.hidden = true; return; }
    tip.innerHTML = `<div class="h">at ${fmtM(x)} tokens</div>${rows.join("")}`;
    tip.hidden = false;
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    let left = ev.clientX + 14, top = box.top + 8;
    if (left + tw > window.innerWidth - 8) left = ev.clientX - tw - 14;
    if (left < 8) left = 8;
    if (top < 60) top = Math.min(window.innerHeight - th - 8, box.bottom - th - 4);
    tip.style.left = left + "px"; tip.style.top = top + "px";
  };
  const hide = () => { tip.hidden = true; hov.innerHTML = ""; };
  hit.addEventListener("pointermove", (e) => { clearTimeout(hideT); show(e); });
  hit.addEventListener("pointerdown", (e) => { clearTimeout(hideT); show(e); });
  hit.addEventListener("pointerleave", (e) => { if (e.pointerType === "mouse") hide(); else hideT = setTimeout(hide, 2500); });
  hit.addEventListener("pointerup", (e) => { if (e.pointerType !== "mouse") { clearTimeout(hideT); hideT = setTimeout(hide, 2500); } });
}

function drawRace(set) {
  const R = set.map(runOf).filter(Boolean);
  const auto = !PICK;
  $("raceReset").hidden = auto;
  const groups = auto ? autoSet().groups : [];
  $("raceTitle").textContent = auto && groups.length === 1 && R.length > 1 ? `The race · ${groups[0]}` : R.length > 1 ? "Side by side" : "The run";
  const lineKey = (dash) => `<svg viewBox="0 0 22 8"><line x1="1" x2="21" y1="4" y2="4" stroke="#b4b3c4" stroke-width="${dash ? 1.2 : 2.4}" stroke-opacity="${dash ? 0.5 : 1}"/></svg>`;
  $("legend").innerHTML = R.map((r) => `<span class="lg" style="--c:${colorOf(r.name)}"><i></i>${esc(r.name)}${isLive(r) ? badge(r.live) : r.live === "cut" ? badge("cut") : ""}</span>`).join("")
    + (R.length ? `<span class="key">${lineKey(false)}held-out</span><span class="key">${lineKey(true)}training</span><span class="key"><svg viewBox="0 0 22 8"><circle cx="11" cy="4" r="3" fill="none" stroke="#b4b3c4" stroke-width="1.6"/></svg>final score</span>` : "");
  if (R.length) $("legend").innerHTML = $("legend").innerHTML.replace('<span class="key">', '<span class="sep"></span><span class="key">');
  const ref = R.find(isLive) || R[0];
  const xmax = Math.max(1, ...R.map((r) => Math.max(r.total || 0, r.tokens || 0)));
  const turns = ref ? ref.turns || [] : [];
  const sOf = (r, k) => ((SER[r.name] || {}).d || {})[k] || [];
  const base = (k) => R.map((r) => ({ name: r.name, color: colorOf(r.name), pts: sOf(r, k), live: r.live === "live" }));
  const empty = R.length ? null : "Tap a run below to draw it here";
  lineChart("cLoss", { series: R.map((r) => ({ name: r.name, color: colorOf(r.name), pts: sOf(r, "val"), faint: sOf(r, "train"), live: r.live === "live" })),
    turns, xmax, fmt: fmtL, crop: 0.08, faintInRange: true, minSpan: 0.02, empty, label: "Held-out loss" });
  const hs = base("hs"), hsVals = hs.flatMap((s) => s.pts.map((p) => p[1]));
  lineChart("cHs", { series: hs, turns, xmax, fmt: (v) => pct(v), pctAxis: true, minSpan: 0.01, empty,
    refs: [{ y: HS_GPT2, label: "OpenAI GPT-2", always: hsVals.length && Math.max(...hsVals) > 0.27 }, { y: 0.25, label: "guessing" }], label: "HellaSwag" });
  const lt = base("logt"), chance = (R.map((r) => sOf(r, "logt").slice(-1)[0]).find((p) => p && p[3] != null) || [])[3];
  lineChart("cLogt", { series: lt, turns, xmax, fmt: (v) => pct(v), pctAxis: true, minSpan: 0.01, empty,
    refs: [...(chance ? [{ y: chance, label: "guessing" }] : []), ...yardsticks("logt")], label: "LogT" });
  lineChart("cWorld", { series: base("world"), turns, xmax, fmt: (v) => pct(v), pctAxis: true, minSpan: 0.02, refs: yardsticks("world_secret"),
    empty: R.length ? "No world exams in these runs (only runs with world text take them)" : empty, label: "World exams" });
  // PIQA (2 choices) and ARC-Easy (4): a quick check on the tests' schedule, every question at the end (the ring)
  lineChart("cPiqa", { series: base("piqa"), turns, xmax, fmt: (v) => pct(v), pctAxis: true, minSpan: 0.01,
    refs: [{ y: 0.5, label: "guessing" }], empty: R.length ? "No PIQA checks yet" : empty, label: "PIQA" });
  lineChart("cArc", { series: base("arc"), turns, xmax, fmt: (v) => pct(v), pctAxis: true, minSpan: 0.01,
    refs: [{ y: 0.25, label: "guessing" }], empty: R.length ? "No ARC checks yet" : empty, label: "ARC-Easy" });
}

// ---------------------------------------------------------------- every run, as a tile
function spark(pts, color) {
  if (!pts || pts.length < 2) return "";
  const xs = pts.map((p) => p[0]), ys = pts.slice(Math.floor(pts.length * 0.1)).map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), lo = Math.min(...ys), hi = Math.max(...ys) || 1;
  const X = (x) => 2 + (x - x0) / ((x1 - x0) || 1) * 72, Y = (y) => 3 + (1 - (Math.min(hi, y) - lo) / ((hi - lo) || 1)) * 24;
  return `<svg class="spark" viewBox="0 0 76 30" aria-hidden="true"><path d="${pts.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("")}" fill="none" stroke="${color}" stroke-width="1.6" stroke-linejoin="round"/></svg>`;
}
function tile(r, set) {
  const on = set.includes(r.name), c = on ? colorOf(r.name) : "#7d7c8f";
  const f = fracOf(r);
  const when = r.live === "done" ? `finished ${ago(r.age)}${r.ended_early ? " (ended early)" : ""}`
    : r.live === "cut" ? `cut off ${ago(r.age)} at ${Math.round(f * 100)}%`
      : r.live === "stopped" ? `stopped ${ago(r.age)} at ${Math.round(f * 100)}%, resumable`
        : r.live === "live" ? `${Math.round(f * 100)}% · ${fmtDur(r.eta_s)} left` : BADGE[r.live];
  const kv = (k, v) => `<div class="kv"><div class="k">${k}</div><div class="v${v === "–" ? " na" : ""}">${v}</div></div>`;
  return `<button class="tile${on ? " pick" : ""}${!isLive(r) && r.age > 3 * 86400 ? " old" : ""}" data-run="${esc(r.name)}" style="--c:${c}" aria-pressed="${on}">
    <div class="t1">${on ? `<span class="sw" style="background:${c}"></span>` : ""}<span class="nm">${esc(r.name)}</span>${badge(r.live)}</div>
    <div class="t2">${esc(nice(r))}${r.params ? " · " + Math.round(r.params / 1e6) + "M" : ""} · ${fmtM(r.tokens)} of ${fmtM(r.total)}</div>
    <div class="mini ${isLive(r) ? "live" : r.live}"><div style="width:${(f * 100).toFixed(1)}%"></div></div>
    <div class="t3">${kv("loss", fmtL(r.val))}${kv("HellaSwag", r.hs != null ? pct(r.hs) : r.hs_ref != null ? pct(r.hs_ref) : "–")}${r.world_secret != null ? kv("world", pct(r.world_secret, 0)) : r.logt != null ? kv("LogT", pct(r.logt, 0)) : ""}${spark(r.spark, c)}</div>
    <div class="t4">${when}</div></button>`;
}
function drawTiles(set) {
  const keep = (r) => FILTER === "all" || (FILTER === "done" ? r.live === "done" : r.live === "stopped" || r.live === "cut");
  const R = ours().filter(keep).sort((a, b) => (isLive(b) - isLive(a)) || (set.includes(b.name) - set.includes(a.name)) || (b.updated - a.updated));
  const N = MORE ? R.length : 12;
  $("tiles").innerHTML = R.length ? R.slice(0, N).map((r) => tile(r, set)).join("") : `<p class="vhint">No runs here.</p>`;
  const old = document.querySelector(".vmore"); if (old) old.remove();
  if (R.length > N) $("tiles").insertAdjacentHTML("afterend", `<button class="btn vmore">Show all ${R.length} runs</button>`);
}

fetch("api/version").then((r) => r.json()).then((v) => { if (v && v.version) { $("ver").textContent = v.version; $("ver").title = `commit ${v.commit}`; } }).catch(() => {});
let rz = null;
window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => drawRace(compareSet()), 150); });
refresh();
setInterval(refresh, 15000);
