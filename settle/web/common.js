// Shared drawing helpers for the Settle dashboard and chat app (no libraries).
"use strict";

const Settle = (() => {
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

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
    const mult = [1, 2, 2.5, 5, 10].find((m) => span / (step0 * m) <= n) || 10;
    const step = step0 * mult;
    const ticks = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) ticks.push(+v.toFixed(10));
    return ticks;
  }

  function fmtTokens(v) {
    if (v >= 1e9) return (v / 1e9).toFixed(v % 1e9 ? 1 : 0) + "B";
    if (v >= 1e6) return (v / 1e6).toFixed(v % 1e6 ? 1 : 0) + "M";
    if (v >= 1e3) return (v / 1e3).toFixed(0) + "k";
    return String(Math.round(v));
  }

  // Line chart with a crosshair tooltip. series: [{name, color, dash, width, points: [[x,y],...]}]
  function lineChart(cv, tip, opt) {
    const state = { opt, hoverX: null };
    function draw() {
      const { ctx, w, h } = fitCanvas(cv);
      const o = state.opt;
      ctx.clearRect(0, 0, w, h);
      const pad = { l: 46, r: 14, t: 10, b: 26 };
      const all = o.series.flatMap((s) => s.points);
      if (!all.length) {
        ctx.fillStyle = css("--muted");
        ctx.font = "13px Inter, system-ui, sans-serif";
        ctx.fillText(o.empty || "waiting for data…", pad.l, h / 2);
        return;
      }
      let x0 = Math.min(...all.map((p) => p[0])), x1 = Math.max(...all.map((p) => p[0]));
      if (x1 === x0) x1 = x0 + 1;
      let ys = all.map((p) => p[1]).filter((v) => isFinite(v));
      let y0 = o.yMin != null ? o.yMin : Math.min(...ys), y1 = o.yMax != null ? o.yMax : Math.max(...ys);
      if (o.yClip) y1 = Math.min(y1, o.yClip(ys));
      if (y1 === y0) { y1 += 1; y0 -= o.yMin != null ? 0 : 1; }
      const padY = (y1 - y0) * 0.06;
      if (o.yMin == null) y0 -= padY;
      y1 += padY;
      const X = (v) => pad.l + ((v - x0) / (x1 - x0)) * (w - pad.l - pad.r);
      const Y = (v) => pad.t + (1 - (v - y0) / (y1 - y0)) * (h - pad.t - pad.b);
      // grid + axes
      ctx.font = "11px Inter, system-ui, sans-serif";
      ctx.strokeStyle = css("--grid");
      ctx.fillStyle = css("--muted");
      ctx.lineWidth = 1;
      for (const v of niceTicks(y0, y1, 5)) {
        const y = Math.round(Y(v)) + 0.5;
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
        ctx.textAlign = "right"; ctx.textBaseline = "middle";
        ctx.fillText((o.yFmt || ((x) => x))(v), pad.l - 6, y);
      }
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      for (const v of niceTicks(x0, x1, Math.max(2, Math.floor((w - 80) / 90)))) {
        ctx.fillText((o.xFmt || ((x) => x))(v), X(v), h - pad.b + 7);
      }
      ctx.strokeStyle = css("--axis");
      ctx.beginPath(); ctx.moveTo(pad.l, h - pad.b + 0.5); ctx.lineTo(w - pad.r, h - pad.b + 0.5); ctx.stroke();
      // lines
      ctx.save();
      ctx.beginPath(); ctx.rect(pad.l, pad.t - 2, w - pad.l - pad.r, h - pad.t - pad.b + 4); ctx.clip();
      for (const s of o.series) {
        if (!s.points.length) continue;
        ctx.strokeStyle = s.color;
        ctx.globalAlpha = s.alpha || 1;
        ctx.lineWidth = s.width || 2;
        ctx.setLineDash(s.dash || []);
        ctx.lineJoin = "round";
        ctx.beginPath();
        s.points.forEach((p, i) => (i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1]))));
        ctx.stroke();
        if (s.dots) {
          ctx.fillStyle = s.color;
          for (const p of s.points) { ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 2.5, 0, 7); ctx.fill(); }
        }
      }
      ctx.restore();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      // crosshair
      if (state.hoverX != null && state.hoverX >= pad.l && state.hoverX <= w - pad.r) {
        const xv = x0 + ((state.hoverX - pad.l) / (w - pad.l - pad.r)) * (x1 - x0);
        ctx.strokeStyle = css("--axis");
        ctx.beginPath(); ctx.moveTo(state.hoverX + 0.5, pad.t); ctx.lineTo(state.hoverX + 0.5, h - pad.b); ctx.stroke();
        const rows = [];
        for (const s of o.series) {
          if (!s.points.length || s.noTip) continue;
          let best = s.points[0];
          for (const p of s.points) if (Math.abs(p[0] - xv) < Math.abs(best[0] - xv)) best = p;
          ctx.fillStyle = css("--surface");
          ctx.beginPath(); ctx.arc(X(best[0]), Y(best[1]), 5, 0, 7); ctx.fill();
          ctx.fillStyle = s.color;
          ctx.beginPath(); ctx.arc(X(best[0]), Y(best[1]), 3.5, 0, 7); ctx.fill();
          rows.push(`<div class="tiprow"><span class="sw" style="background:${s.color}"></span>${s.name}<b>${(o.tipFmt || o.yFmt || ((x) => x))(best[1])}</b></div>`);
        }
        tip.innerHTML = `<div class="tiphead">${(o.xTip || o.xFmt || ((x) => x))(xv)}</div>` + rows.join("");
        tip.style.display = "block";
        const r = cv.getBoundingClientRect();
        const left = state.hoverX + 14 + tip.offsetWidth > w ? state.hoverX - 14 - tip.offsetWidth : state.hoverX + 14;
        tip.style.left = left + "px";
        tip.style.top = "8px";
      } else tip.style.display = "none";
    }
    cv.addEventListener("mousemove", (e) => { state.hoverX = e.offsetX; draw(); });
    cv.addEventListener("mouseleave", () => { state.hoverX = null; draw(); });
    return { draw, set(opt) { state.opt = opt; draw(); } };
  }

  // Colour ramp for "how many steps": dark ember -> bright amber -> white-hot
  function heat(t) {
    t = Math.max(0, Math.min(1, t));
    const stops = [[38, 30, 26], [120, 45, 20], [217, 89, 38], [245, 166, 80], [255, 236, 200]];
    const f = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(f)), u = f - i;
    const a = stops[i], b = stops[i + 1];
    return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * u)).join(",")})`;
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }

  return { css, fitCanvas, b64bytes, lineChart, fmtTokens, heat, escapeHtml, niceTicks };
})();
