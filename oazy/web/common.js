// Shared drawing helpers for the Oazy dashboard and chat app (no libraries).
"use strict";

const UI = (() => {
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const FONT = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif";
  const font = (size, weight = 400) => `${weight} ${size}px ${FONT}`;

  function fitCanvas(cv, h) {
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth;
    if (h != null) cv.style.height = h + "px";
    const H = cv.clientHeight;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(H * dpr)) {
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(H * dpr);
    }
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h: H };
  }

  function b64bytes(s) {
    const bin = atob(s);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  function niceTicks(lo, hi, n) {
    const span = hi - lo || 1;
    const step0 = Math.pow(10, Math.floor(Math.log10(span / n)));
    // the finest round step that gives at most ~n ticks (a little over is fine: 3.0 3.5 4.0 … beats 2 4 6)
    const mult = [1, 2, 2.5, 5, 10].find((m) => span / (step0 * m) <= n * 1.3) || 10;
    const step = step0 * mult;
    const ticks = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) ticks.push(+v.toFixed(10));
    ticks.step = step;
    return ticks;
  }
  // decimals a tick label needs at this step: 0.5 -> 1, 0.25 -> 2, 2 -> 0
  const tickDec = (step) => { for (let d = 0; d < 6; d++) if (Math.abs(Math.round(step * 10 ** d) - step * 10 ** d) < 1e-6) return d; return 6; };

  function fmtTokens(v) {
    if (v >= 1e9) return (v / 1e9).toFixed(v % 1e9 ? 1 : 0) + "B";
    if (v >= 1e6) return (v / 1e6).toFixed(v % 1e6 ? 1 : 0) + "M";
    if (v >= 1e3) return (v / 1e3).toFixed(0) + "k";
    return String(Math.round(v));
  }

  // "#rrggbb" or "rgb(r,g,b)" -> "rgba(r,g,b,a)"
  function alpha(color, a) {
    if (color.startsWith("#")) {
      const n = parseInt(color.length === 4 ? color.replace(/#(.)(.)(.)/, "$1$1$2$2$3$3") : color.slice(1), 16);
      return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
    }
    const m = color.match(/[\d.]+/g);
    return m ? `rgba(${m[0]},${m[1]},${m[2]},${a})` : color;
  }

  function placeTip(tip, x, y, boxW, boxH) {
    tip.style.display = "block";
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    let left = x + 16;
    if (left + tw > boxW - 2) left = x - 16 - tw;
    if (left < 2) left = 2;
    let top = y;
    if (boxH != null && top + th > boxH) top = Math.max(2, boxH - th);
    tip.style.left = left + "px";
    tip.style.top = top + "px";
  }

  // Line chart with a crosshair tooltip.
  // series: [{name, color, dash, width, alpha, area, dots, noTip, tipRange, endDot, band: [[x,lo,hi]], marks: [[x,y,alpha]], endLabel, labelAlpha, points: [[x,y],...]}]
  // ghost: only in the tooltip, never drawn. sparse: measured only now and then, so the tooltip shows it only near
  // one of its real points. pts: a dot of that radius on every point (the real measurements of a sparse line).
  function lineChart(cv, tip, opt) {
    const state = { opt, hoverX: null, hoverY: 0 };
    function draw() {
      const { ctx, w, h } = fitCanvas(cv);
      const o = state.opt;
      ctx.clearRect(0, 0, w, h);
      const hasEnd = o.series.some((s) => s.endLabel && s.points.length);
      const pad = { l: 44, r: hasEnd ? 58 : 12, t: 12, b: 26 };
      // xMin: leave out everything left of it (with a log axis, the first few steps would otherwise take half the width)
      const xIn = o.xMin != null ? (p) => p[0] >= o.xMin : () => true;
      const all = o.series.flatMap((s) => s.points.filter(xIn));
      const extra = o.series.flatMap((s) => [...(s.band || []).flatMap((b) => [[b[0], b[1]], [b[0], b[2]]]), ...(s.marks || [])]);
      if (!all.length) {
        ctx.fillStyle = css("--muted");
        ctx.font = font(13);
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        if (o.empty !== "") ctx.fillText(o.empty || "Waiting for data…", w / 2, h / 2);
        tip.style.display = "none";
        return;
      }
      // xLog: a logarithmic x axis (1B, 10B, 100B…), so a short run and a forecast far beyond it both fit
      const pos = o.xLog ? all.map((p) => p[0]).filter((v) => v > 0) : null;
      const lmin = o.xLog && pos.length ? Math.log10(Math.min(...pos)) : 0;
      const L = o.xLog ? (v) => (v > 0 ? Math.max(lmin, Math.log10(v)) : lmin) : (v) => v;
      const Linv = o.xLog ? (v) => Math.pow(10, v) : (v) => v;
      let x0 = Math.min(...all.map((p) => L(p[0]))), x1 = Math.max(...all.map((p) => L(p[0])));
      if (x1 === x0) x1 = x0 + 1;
      const fin = (v) => isFinite(v);
      let ys = all.concat(extra).map((p) => p[1]).filter(fin);
      if (o.yFocus) {
        // Fit the view to the lines that are drawn (not the hover-only ones), leaving out the first yFocus of the x
        // range, so the steep first stretch (a loss of 11 at the start) cannot flatten the part worth reading: it
        // runs off the top instead. The forecast cones and earlier-forecast dots may widen the view, but only a
        // little: a wild early guess is pinned to the edge.
        const xc = x0 + o.yFocus * (x1 - x0);
        let line = o.series.filter((s) => !s.ghost).flatMap((s) => s.points.filter((p) => xIn(p) && L(p[0]) >= xc).map((p) => p[1])).filter(fin);
        if (!line.length) line = o.series.filter((s) => !s.ghost).flatMap((s) => s.points.map((p) => p[1])).filter(fin);
        if (line.length) {
          let lo = Math.min(...line), hi = Math.max(...line);
          const keep = o.series.filter((s) => s.keep && !s.ghost).flatMap((s) => s.points.map((p) => p[1])).filter(fin);
          if (keep.length) { lo = Math.min(lo, ...keep); hi = Math.max(hi, ...keep); }
          const room = Math.max(hi - lo, 0.1) * 0.25, ex = extra.map((p) => p[1]).filter(fin);
          if (ex.length) { lo = Math.min(lo, Math.max(lo - room, Math.min(...ex))); hi = Math.max(hi, Math.min(hi + room, Math.max(...ex))); }
          ys = [lo, hi];
        }
      }
      let y0 = o.yMin != null ? o.yMin : Math.min(...ys), y1 = o.yMax != null ? o.yMax : Math.max(...ys);
      if (y1 === y0) { y1 += 1; y0 -= o.yMin != null ? 0 : 1; }
      const padY = (y1 - y0) * 0.08;
      if (o.yMin == null) y0 -= padY;
      if (o.yMax == null) y1 += padY;
      const X = (v) => pad.l + ((L(v) - x0) / (x1 - x0)) * (w - pad.l - pad.r);
      const Y = (v) => pad.t + (1 - (v - y0) / (y1 - y0)) * (h - pad.t - pad.b);
      const bottom = h - pad.b;
      // grid + axes
      ctx.font = font(11);
      ctx.lineWidth = 1;
      const yt = niceTicks(y0, y1, Math.max(2, Math.min(5, Math.floor((h - 40) / 40))));
      for (const v of yt) {
        const y = Math.round(Y(v)) + 0.5;
        if (y < pad.t - 1 || y > bottom + 1) continue;
        ctx.strokeStyle = css("--grid");
        ctx.setLineDash([2, 4]);
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = css("--muted");
        ctx.textAlign = "right"; ctx.textBaseline = "middle";
        ctx.fillText((o.yFmt || ((x, d) => x.toFixed(d)))(v, tickDec(yt.step)), pad.l - 8, y);
      }
      ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillStyle = css("--muted");
      let lastLab = null;
      const xticks = o.xLog ? (() => {
        const t = [], room = Math.max(2, Math.floor((w - 100) / 70));
        for (let k = Math.ceil(x0 - 1e-9); k <= Math.floor(x1 + 1e-9); k++) for (const m of [1, 2, 5]) {
          const lv = k + Math.log10(m);
          if (lv >= x0 - 1e-9 && lv <= x1 + 1e-9 && (m === 1 || x1 - x0 < room / 3)) t.push(Math.pow(10, lv));
        }
        return t;
      })() : niceTicks(x0, x1, Math.max(2, Math.floor((w - 100) / 90)));
      for (const v of xticks) {
        const lab = (o.xFmt || ((x) => x))(v);
        if (lab === lastLab) continue;
        lastLab = lab;
        ctx.fillText(lab, X(v), bottom + 8);
      }
      ctx.strokeStyle = css("--axis");
      ctx.beginPath(); ctx.moveTo(pad.l, bottom + 0.5); ctx.lineTo(w - pad.r, bottom + 0.5); ctx.stroke();
      // vlines: [[x, label]], thin markers such as "planned end"
      for (const [vx, vl] of o.vlines || []) {
        const xx = Math.round(X(vx)) + 0.5;
        if (!(xx >= pad.l && xx <= w - pad.r)) continue;
        ctx.strokeStyle = css("--muted"); ctx.globalAlpha = 0.5; ctx.setLineDash([1, 3]);
        ctx.beginPath(); ctx.moveTo(xx, pad.t); ctx.lineTo(xx, bottom); ctx.stroke();
        ctx.setLineDash([]); ctx.globalAlpha = 0.8; ctx.fillStyle = css("--muted"); ctx.font = font(10.5);
        ctx.textAlign = xx > (pad.l + w - pad.r) / 2 ? "right" : "left"; ctx.textBaseline = "top";
        if (vl) ctx.fillText(vl, xx + (ctx.textAlign === "left" ? 4 : -4), pad.t + 2);
        ctx.globalAlpha = 1;
      }
      // lines
      ctx.save();
      ctx.beginPath(); ctx.rect(pad.l, pad.t - 4, w - pad.l - pad.r, bottom - pad.t + 6); ctx.clip();
      for (const s of o.series) {
        if (!s.points.length || s.ghost) continue;      // ghost: only in the tooltip
        const pts = s.points.filter((p) => isFinite(p[1]) && xIn(p));
        const cut = o.xMin != null ? s.points.findLastIndex((p) => p[0] < o.xMin) : -1;
        if (cut >= 0 && cut < s.points.length - 1) {     // a line that starts left of xMin starts at its edge
          const [p0, p1] = [s.points[cut], s.points[cut + 1]], u = (o.xMin - p0[0]) / Math.max(1e-12, p1[0] - p0[0]);
          pts.unshift([o.xMin, p0[1] + (p1[1] - p0[1]) * u]);
        }
        if (s.area && pts.length > 1) {
          const g = ctx.createLinearGradient(0, pad.t, 0, bottom);
          g.addColorStop(0, alpha(s.color, 0.22));
          g.addColorStop(1, alpha(s.color, 0));
          ctx.fillStyle = g;
          ctx.beginPath();
          ctx.moveTo(X(pts[0][0]), bottom);
          pts.forEach((p) => ctx.lineTo(X(p[0]), Y(p[1])));
          ctx.lineTo(X(pts[pts.length - 1][0]), bottom);
          ctx.closePath(); ctx.fill();
        }
        ctx.strokeStyle = s.color;
        ctx.globalAlpha = s.alpha || 1;
        ctx.lineWidth = Math.max(0.8, Math.min(1.5, (s.width || 2) * 0.65));   // thin, crisp lines
        ctx.setLineDash(s.dash || []);
        ctx.lineJoin = "round"; ctx.lineCap = "round";
        ctx.beginPath();
        pts.forEach((p, i) => (i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1]))));
        ctx.stroke();
        ctx.setLineDash([]);
        if (s.band && s.band.length > 1) {                // a faint cone: how sure the forecast is
          ctx.fillStyle = alpha(s.color, s.bandAlpha ?? 0.1);
          ctx.beginPath();
          s.band.forEach((b, i) => (i ? ctx.lineTo(X(b[0]), Y(b[2])) : ctx.moveTo(X(b[0]), Y(b[2]))));
          for (let i = s.band.length - 1; i >= 0; i--) ctx.lineTo(X(s.band[i][0]), Y(s.band[i][1]));
          ctx.closePath(); ctx.fill();
        }
        if (s.marks) for (const [mx, my, ma] of s.marks) {   // earlier forecasts, older ones fainter
          ctx.globalAlpha = (s.alpha || 1) * ma; ctx.fillStyle = s.color;
          const yy = Y(my), py = Math.max(pad.t, Math.min(bottom, yy));   // off the chart: pinned to its edge
          ctx.beginPath(); ctx.arc(X(mx), py, py === yy ? 2 : 1.4, 0, 7); ctx.fill();
        }
        ctx.globalAlpha = s.alpha || 1;
        if (s.pts) {                                      // a dot on every real measurement
          ctx.fillStyle = s.color;
          for (const p of pts) { ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), s.pts, 0, 7); ctx.fill(); }
        }
        if (s.endDot && pts.length) {                     // a hollow ring where a forecast lands
          const p = pts[pts.length - 1];
          ctx.fillStyle = css("--solid"); ctx.lineWidth = 1.4;
          ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 3.6, 0, 7); ctx.fill(); ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }
      ctx.restore();
      // direct labels at the end of lines
      const labels = [];
      for (const s of o.series) {
        if (!s.endLabel || !s.points.length) continue;
        const p = s.points[s.points.length - 1];
        labels.push({ y: Y(p[1]), x: X(p[0]), text: s.endLabel === true ? (o.endFmt || o.tipFmt || o.yFmt || ((x) => x))(p[1]) : s.endLabel, color: s.color, alpha: s.labelAlpha || 1 });
      }
      labels.sort((a, b) => a.y - b.y);
      for (let i = 1; i < labels.length; i++) if (labels[i].y - labels[i - 1].y < 15) labels[i].y = labels[i - 1].y + 15;
      for (const L of labels) {
        ctx.fillStyle = L.color; ctx.globalAlpha = L.alpha;
        ctx.font = font(11.5, 600); ctx.textAlign = "left"; ctx.textBaseline = "middle";
        ctx.fillText(L.text, w - pad.r + 8, Math.max(pad.t, Math.min(bottom, L.y)));
        ctx.globalAlpha = 1;
      }
      // crosshair
      if (state.hoverX != null && state.hoverX >= pad.l && state.hoverX <= w - pad.r) {
        const xv = Linv(x0 + ((state.hoverX - pad.l) / (w - pad.l - pad.r)) * (x1 - x0));
        const g = ctx.createLinearGradient(0, pad.t, 0, bottom);
        g.addColorStop(0, alpha(css("--text"), 0)); g.addColorStop(0.5, alpha(css("--text"), 0.35)); g.addColorStop(1, alpha(css("--text"), 0.1));
        ctx.strokeStyle = g;
        ctx.beginPath(); ctx.moveTo(state.hoverX + 0.5, pad.t); ctx.lineTo(state.hoverX + 0.5, bottom); ctx.stroke();
        const rows = [];
        for (const s of o.series) {
          if (!s.points.length || s.noTip) continue;
          if (s.tipRange && (xv < s.points[0][0] || xv > s.points[s.points.length - 1][0])) continue;
          let best = s.points[0];
          for (const p of s.points) if (Math.abs(L(p[0]) - L(xv)) < Math.abs(L(best[0]) - L(xv))) best = p;
          // a ghost or sparse line (e.g. held-out loss, measured only now and then) is shown only near one of its real
          // points, so the hover never repeats an old measurement as if it were new
          if ((s.ghost || s.sparse) && Math.abs(L(best[0]) - L(xv)) > 0.03 * (x1 - x0)) continue;
          if (!s.ghost) {
            ctx.fillStyle = css("--solid");
            ctx.beginPath(); ctx.arc(X(best[0]), Y(best[1]), 4.5, 0, 7); ctx.fill();
            ctx.strokeStyle = s.color; ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.arc(X(best[0]), Y(best[1]), 3, 0, 7); ctx.stroke();
          }
          const f = s.tipFmt || o.tipFmt || o.yFmt || ((x) => x);
          let val = f(best[1]);
          if (s.band && s.band.length) {                  // a forecast: its likely range too
            let bb = s.band[0];
            for (const b of s.band) if (Math.abs(b[0] - best[0]) < Math.abs(bb[0] - best[0])) bb = b;
            if (bb[2] - bb[1] > 0) val += ` <small>(${f(bb[1])}–${f(bb[2])})</small>`;
          }
          rows.push(`<div class="tiprow"><span class="sw" style="background:${s.color};opacity:${s.alpha && s.alpha < 1 ? 0.5 : 1}"></span>${s.name}<b>${val}</b></div>`);
        }
        const note = o.tipNote ? o.tipNote(xv) : "";
        tip.innerHTML = `<div class="tiphead">${(o.xTip || o.xFmt || ((x) => x))(xv)}</div>` + rows.join("") + (note ? `<div class="tipnote">${note}</div>` : "");
        placeTip(tip, state.hoverX, 6, w);
      } else tip.style.display = "none";
    }
    cv.addEventListener("mousemove", (e) => { state.hoverX = e.offsetX; draw(); });
    cv.addEventListener("mouseleave", () => { state.hoverX = null; draw(); });
    return { draw, set(opt) { state.opt = opt; draw(); } };
  }


  // Fit y = a + b * x^-c, the usual shape of a learning curve (loss falls fast, then ever slower toward a floor a).
  // For each c on a grid, a and b come from ordinary least squares; the best c is then refined.
  // Returns {a, b, c, f, rmse} or null. x must be > 0.
  function fitPowerLaw(pts) {
    pts = pts.filter((p) => p[0] > 0 && isFinite(p[1]));
    if (pts.length < 4) return null;
    const xs = Math.max(...pts.map((p) => p[0]));           // scale x to (0, 1] for stable numbers
    const solve = (c) => {
      let n = 0, su = 0, sy = 0, suu = 0, suy = 0;
      for (const [x, y] of pts) { const u = Math.pow(x / xs, -c); n++; su += u; sy += y; suu += u * u; suy += u * y; }
      const d = n * suu - su * su;
      if (Math.abs(d) < 1e-12) return null;
      const b = (n * suy - su * sy) / d, a = (sy - b * su) / n;
      let e = 0;
      for (const [x, y] of pts) { const r = y - (a + b * Math.pow(x / xs, -c)); e += r * r; }
      return { a, b, c, e };
    };
    let best = null;
    for (let i = 0; i <= 80; i++) {
      const r = solve(0.02 * Math.pow(100, i / 80));          // c from 0.02 to 2
      if (r && (!best || r.e < best.e)) best = r;
    }
    if (!best) return null;
    let lo = best.c / 1.06, hi = best.c * 1.06;                // golden-section polish
    for (let k = 0; k < 30; k++) {
      const m1 = hi - (hi - lo) * 0.618, m2 = lo + (hi - lo) * 0.618;
      const r1 = solve(m1), r2 = solve(m2);
      if (!r1 || !r2) break;
      if (r1.e < r2.e) hi = m2; else lo = m1;
    }
    const r = solve((lo + hi) / 2);
    if (r && r.e < best.e) best = r;
    const { a, b, c } = best;
    return { a, b, c, xs, rmse: Math.sqrt(best.e / pts.length), f: (x) => a + b * Math.pow(x / xs, -c) };
  }

  // Where a learning curve is headed: smooth the points (log-spaced bins, medians), leave out the early rush,
  // fit y = a + b * x^-c, and anchor it to where the line is now. Returns {g, last, yEnd, rmse} or null.
  // recent: fit only the newest share of the points (by count; at least minPts of them), instead of skipping an early share of the tokens
  function forecastEnd(pts, end, { minPts = 8, skip = 0.2, recent = 0, bins = 40, lo = -Infinity, hi = Infinity, dir = -1 } = {}) {
    if (!pts.length) return null;
    const xl = pts[pts.length - 1][0];
    if (!(xl > 0)) return null;
    let use = recent > 0 ? pts.filter((p) => p[0] > 0) : pts.filter((p) => p[0] >= xl * skip && p[0] > 0);
    if (recent > 0) use = use.slice(-Math.max(minPts, Math.ceil(use.length * recent)));
    if (use.length > bins) {
      const l0 = Math.log(use[0][0]), l1 = Math.log(xl) + 1e-9, B = Array.from({ length: bins }, () => []);
      for (const p of use) B[Math.min(bins - 1, Math.floor((Math.log(p[0]) - l0) / (l1 - l0) * bins))].push(p);
      use = B.filter((b) => b.length).map((b) => { const ys = b.map((p) => p[1]).sort((u, v) => u - v); return [b[b.length >> 1][0], ys[ys.length >> 1]]; });
    }
    if (use.length < minPts) return null;
    const fit = fitPowerLaw(use);
    if (!fit || Math.sign(fit.b) !== -dir) return null;      // loss must fall, accuracy rise
    const last = use[use.length - 1], shift = last[1] - fit.f(last[0]);
    const g = (x) => Math.max(lo, Math.min(hi, fit.f(x) + shift));
    const yEnd = g(end);
    return isFinite(yEnd) ? { g, last, yEnd, rmse: fit.rmse } : null;
  }

  // The same forecast replayed from earlier moments of the run (using only the points known then):
  // how the predicted end value has been moving. Returns [[tokens then, predicted end], ...], oldest first.
  function forecastHistory(pts, end, opt, n = 12) {
    const xl = pts.length ? pts[pts.length - 1][0] : 0, out = [];
    for (let i = 0; i < n; i++) {
      const t = xl * (0.25 + 0.75 * i / (n - 1));
      const f = forecastEnd(pts.filter((p) => p[0] <= t), end, opt);
      if (f && (!out.length || f.last[0] > out[out.length - 1][0])) out.push([f.last[0], f.yEnd]);
    }
    return out;
  }

  // Colour ramp: deep sea -> blue -> aqua -> white foam
  const HEAT = [[10, 22, 44], [22, 64, 150], [38, 140, 240], [90, 220, 255], [228, 252, 255]];
  function heatRGB(t) {
    t = Math.max(0, Math.min(1, t));
    const f = t * (HEAT.length - 1), i = Math.min(HEAT.length - 2, Math.floor(f)), u = f - i;
    const a = HEAT[i], b = HEAT[i + 1];
    return a.map((v, k) => Math.round(v + (b[k] - v) * u));
  }
  const heat = (t) => `rgb(${heatRGB(t).join(",")})`;
  const heatCSS = () => `linear-gradient(90deg, ${HEAT.map((c) => `rgb(${c.join(",")})`).join(", ")})`;

  function escapeHtml(s) {
    return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }

  // Keeps the filled part of a range slider in sync (the CSS reads --p).
  function syncRange(el) {
    const lo = +el.min || 0, hi = +el.max || 100;
    el.style.setProperty("--p", (hi > lo ? ((+el.value - lo) / (hi - lo)) * 100 : 0) + "%");
  }
  function watchRanges(root = document) {
    root.querySelectorAll("input[type=range]").forEach((el) => {
      syncRange(el);
      el.addEventListener("input", () => syncRange(el));
    });
  }

  // A word piece as shown on the sheet: newlines and spaces made visible.
  const pieceLabel = (s) => s.replace(/\n/g, "⏎").trim() || "␣";

  // Loss in plain words: on average, the real next word was as hard to guess as picking from N equally likely words.
  const choices = (loss) => Math.exp(loss);
  const fmtChoices = (loss) => { const n = choices(loss); return n < 10 ? n.toFixed(1) : n < 1000 ? Math.round(n).toString() : fmtTokens(n); };

  return { css, font, fitCanvas, b64bytes, lineChart, fmtTokens, heat, heatRGB, heatCSS, escapeHtml, niceTicks, tickDec, alpha,
           placeTip, fitPowerLaw, forecastEnd, forecastHistory, syncRange, watchRanges, pieceLabel, choices, fmtChoices };
})();
