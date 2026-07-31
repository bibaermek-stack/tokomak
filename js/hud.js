/* ==========================================================================
   hud.js — scientific monitoring interface
   --------------------------------------------------------------------------
   A small canvas charting toolkit plus the panel definitions that make up
   the control-room dashboard.  Every widget reads from the live Tokamak
   state or from the rolling History buffers, so nothing here is faked.
   ========================================================================== */
(function (global) {
  'use strict';

  /* ============================================================== palette */
  const C = {
    cyan:   '#3fe0ff',
    blue:   '#5b8cff',
    mag:    '#ff5ce0',
    violet: '#a98bff',
    green:  '#4dffb0',
    amber:  '#ffb648',
    red:    '#ff4d6a',
    white:  '#e8f6ff',
    dim:    'rgba(150,200,230,0.45)',
    grid:   'rgba(90,180,220,0.085)',
    grid2:  'rgba(90,180,220,0.18)',
    axis:   'rgba(120,200,240,0.35)'
  };
  const FONT = 'ui-monospace, "Cascadia Mono", "SF Mono", Consolas, monospace';

  /* =============================================================== format */
  function fmt(v, d) {
    if (!isFinite(v)) return '——';
    const a = Math.abs(v);
    if (a >= 1e9) return (v / 1e9).toFixed(d != null ? d : 2) + 'G';
    if (a >= 1e6) return (v / 1e6).toFixed(d != null ? d : 2) + 'M';
    if (a >= 1e3) return (v / 1e3).toFixed(d != null ? d : 2) + 'k';
    return v.toFixed(d != null ? d : 2);
  }
  function expo(v, d) {
    if (!isFinite(v) || v === 0) return '0.00';
    const e = Math.floor(Math.log10(Math.abs(v)));
    return (v / Math.pow(10, e)).toFixed(d == null ? 2 : d) + 'e' + (e >= 0 ? '+' : '') + e;
  }

  /* ============================================================ colormaps */
  /* Perceptual "plasma"-like ramp: deep indigo -> magenta -> orange -> white */
  const PLASMA_STOPS = [
    [0.00, 12, 7, 60], [0.15, 76, 10, 122], [0.32, 140, 22, 130],
    [0.50, 196, 48, 106], [0.68, 238, 96, 64], [0.85, 251, 168, 46],
    [1.00, 252, 250, 190]
  ];
  const THERMAL_STOPS = [
    [0.00, 4, 10, 26], [0.20, 12, 60, 120], [0.42, 30, 170, 210],
    [0.62, 150, 230, 200], [0.80, 255, 200, 90], [1.00, 255, 255, 240]
  ];
  function ramp(stops, t) {
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    for (let i = 0; i < stops.length - 1; i++) {
      const a = stops[i], b = stops[i + 1];
      if (t >= a[0] && t <= b[0]) {
        const k = (t - a[0]) / (b[0] - a[0]);
        return [a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k,
                a[3] + (b[3] - a[3]) * k];
      }
    }
    const l = stops[stops.length - 1];
    return [l[1], l[2], l[3]];
  }
  const rgbStr = c => 'rgb(' + (c[0] | 0) + ',' + (c[1] | 0) + ',' + (c[2] | 0) + ')';

  /* =============================================================== history */
  const HIST_N = 1400;              // samples retained
  const HIST_DT = 0.04;             // shot-time between samples [s]

  class History {
    constructor(channels) {
      this.n = HIST_N; this.head = 0; this.count = 0;
      this.ch = {};
      channels.forEach(k => { this.ch[k] = new Float32Array(HIST_N); });
      this.time = new Float32Array(HIST_N);
      this.acc = 0;
      /* rolling 2-D profile history for the spectrogram widget */
      this.pw = 190; this.ph = Tokamak.NR;
      this.prof = new Float32Array(this.pw * this.ph);
      this.profHead = 0;
    }
    push(P, dt) {
      this.acc += dt;
      if (this.acc < HIST_DT) return false;
      this.acc = 0;
      const i = this.head;
      const c = this.ch;
      c.Te[i] = P.Te;         c.Ti[i] = P.Ti;        c.ne[i] = P.ne;
      c.nG[i] = P.nG;         c.Pfus[i] = P.Pfus;    c.Q[i] = Math.min(P.Q, 60);
      c.Ip[i] = P.Ip;         c.Bt[i] = P.Bt;        c.tauE[i] = P.tauE;
      c.tauE98[i] = P.tauE98; c.Prad[i] = P.Prad;    c.Pbrem[i] = P.Pbrem;
      c.Pline[i] = P.Pline;   c.Psync[i] = P.Psync;  c.Paux[i] = P.Paux;
      c.Palpha[i] = P.Palpha; c.Pohm[i] = P.Pohm;    c.W[i] = P.W;
      c.beta[i] = P.beta;     c.betaN[i] = P.betaN;  c.q95[i] = P.q95;
      c.neutron[i] = P.neutronRate; c.pVac[i] = P.pVac; c.Zeff[i] = P.Zeff;
      c.qDiv[i] = P.qDiv;     c.Vloop[i] = P.Vloop;  c.fluxTor[i] = P.fluxTor;
      c.fluxPol[i] = P.fluxPol; c.mirnov[i] = P.mirnov; c.dAlpha[i] = P.dAlpha;
      c.nHe[i] = P.nHe;       c.fG[i] = P.fG;        c.H98[i] = P.H98;
      this.time[i] = P.t;
      this.head = (i + 1) % this.n;
      if (this.count < this.n) this.count++;

      /* profile column */
      const col = this.profHead;
      for (let r = 0; r < this.ph; r++) this.prof[r * this.pw + col] = P.TeProf[r];
      this.profHead = (col + 1) % this.pw;
      return true;
    }
    /* iterate the last `span` samples oldest -> newest */
    each(span, fn) {
      const n = Math.min(span, this.count);
      for (let k = 0; k < n; k++) {
        const idx = (this.head - n + k + this.n * 2) % this.n;
        fn(idx, k / Math.max(1, n - 1), k);
      }
      return n;
    }
    max(key, span) {
      let m = -Infinity;
      this.each(span, i => { const v = this.ch[key][i]; if (v > m) m = v; });
      return m === -Infinity ? 1 : m;
    }
  }

  /* ================================================================ chart */
  /* Reusable drawing primitives shared by every panel. */
  const G = {
    frame(x, ctx, w, h, title, unit) {
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = 'rgba(3,9,17,0.62)';
      ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = 'rgba(70,190,240,0.20)';
      ctx.lineWidth = 1;
      ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
      /* corner ticks */
      ctx.strokeStyle = 'rgba(70,220,255,0.55)';
      const L = 7;
      ctx.beginPath();
      ctx.moveTo(0.5, L); ctx.lineTo(0.5, 0.5); ctx.lineTo(L, 0.5);
      ctx.moveTo(w - L, 0.5); ctx.lineTo(w - 0.5, 0.5); ctx.lineTo(w - 0.5, L);
      ctx.moveTo(0.5, h - L); ctx.lineTo(0.5, h - 0.5); ctx.lineTo(L, h - 0.5);
      ctx.moveTo(w - L, h - 0.5); ctx.lineTo(w - 0.5, h - 0.5); ctx.lineTo(w - 0.5, h - L);
      ctx.stroke();
      if (title) {
        ctx.font = '600 8.5px ' + FONT;
        ctx.fillStyle = 'rgba(120,225,255,0.92)';
        ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        ctx.fillText(title.toUpperCase(), 7, 5);
        if (unit) {
          ctx.textAlign = 'right';
          ctx.fillStyle = 'rgba(140,190,215,0.55)';
          ctx.fillText(unit, w - 7, 5);
        }
      }
    },

    /* plot box helper: returns the inner rectangle */
    box(w, h, top) {
      return { x: 30, y: top == null ? 19 : top, w: w - 38, h: h - (top == null ? 19 : top) - 13 };
    },

    grid(ctx, b, nx, ny, yMin, yMax, yFmt, log) {
      ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i <= nx; i++) {
        const x = Math.round(b.x + b.w * i / nx) + 0.5;
        ctx.moveTo(x, b.y); ctx.lineTo(x, b.y + b.h);
      }
      for (let i = 0; i <= ny; i++) {
        const y = Math.round(b.y + b.h * i / ny) + 0.5;
        ctx.moveTo(b.x, y); ctx.lineTo(b.x + b.w, y);
      }
      ctx.stroke();
      ctx.strokeStyle = C.axis;
      ctx.beginPath();
      ctx.moveTo(b.x + 0.5, b.y); ctx.lineTo(b.x + 0.5, b.y + b.h);
      ctx.lineTo(b.x + b.w, b.y + b.h);
      ctx.stroke();
      /* y labels */
      ctx.font = '7.5px ' + FONT;
      ctx.fillStyle = C.dim;
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (let i = 0; i <= ny; i++) {
        const f = 1 - i / ny;
        const v = log
          ? Math.pow(10, yMin + (yMax - yMin) * f)
          : yMin + (yMax - yMin) * f;
        const y = b.y + b.h * i / ny;
        ctx.fillText(yFmt ? yFmt(v, log ? yMin + (yMax - yMin) * f : v) : fmt(v, 1),
                     b.x - 4, y);
      }
    },

    series(ctx, b, hist, key, span, yMin, yMax, color, opt) {
      opt = opt || {};
      const arr = hist.ch[key];
      const log = opt.log;
      const map = v => {
        let f;
        if (log) {
          const l = Math.log10(Math.max(v, 1e-30));
          f = (l - yMin) / (yMax - yMin);
        } else f = (v - yMin) / (yMax - yMin);
        return b.y + b.h * (1 - Math.max(0, Math.min(1, f)));
      };
      ctx.save();
      ctx.beginPath();
      ctx.rect(b.x, b.y - 1, b.w, b.h + 2);
      ctx.clip();

      if (opt.fill) {
        ctx.beginPath();
        let started = false, lastX = b.x;
        hist.each(span, (i, f) => {
          const x = b.x + b.w * f, y = map(arr[i]);
          if (!started) { ctx.moveTo(x, b.y + b.h); ctx.lineTo(x, y); started = true; }
          else ctx.lineTo(x, y);
          lastX = x;
        });
        if (started) {
          ctx.lineTo(lastX, b.y + b.h);
          ctx.closePath();
          const g = ctx.createLinearGradient(0, b.y, 0, b.y + b.h);
          g.addColorStop(0, opt.fill);
          g.addColorStop(1, 'rgba(0,0,0,0)');
          ctx.fillStyle = g; ctx.fill();
        }
      }

      ctx.beginPath();
      let started = false, lx = 0, ly = 0;
      hist.each(span, (i, f) => {
        const x = b.x + b.w * f, y = map(arr[i]);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
        lx = x; ly = y;
      });
      ctx.strokeStyle = color;
      ctx.lineWidth = opt.width || 1.35;
      ctx.lineJoin = 'round';
      if (opt.glow !== false) { ctx.shadowColor = color; ctx.shadowBlur = 6; }
      ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.restore();

      if (started && opt.head !== false) {
        ctx.fillStyle = color;
        ctx.beginPath(); ctx.arc(lx, ly, 1.9, 0, 6.284); ctx.fill();
      }
      return { x: lx, y: ly };
    },

    legend(ctx, b, items) {
      ctx.font = '7.5px ' + FONT;
      ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      let x = b.x + 3;
      items.forEach(it => {
        ctx.fillStyle = it.c;
        ctx.fillRect(x, b.y + 3, 7, 2);
        ctx.fillStyle = 'rgba(190,225,245,0.75)';
        ctx.fillText(it.t, x + 10, b.y);
        x += 12 + ctx.measureText(it.t).width + 8;
      });
    },

    value(ctx, x, y, text, color, size, align) {
      ctx.font = '700 ' + (size || 13) + 'px ' + FONT;
      ctx.fillStyle = color || C.white;
      ctx.textAlign = align || 'right';
      ctx.textBaseline = 'alphabetic';
      ctx.shadowColor = color || C.white; ctx.shadowBlur = 8;
      ctx.fillText(text, x, y);
      ctx.shadowBlur = 0;
    },

    /* circular gauge with a coloured arc and a needle */
    gauge(ctx, cx, cy, R, v, vmin, vmax, label, unit, color, danger) {
      const a0 = Math.PI * 0.75, a1 = Math.PI * 2.25;
      const f = Math.max(0, Math.min(1, (v - vmin) / (vmax - vmin)));
      ctx.lineCap = 'butt';
      /* track */
      ctx.beginPath(); ctx.arc(cx, cy, R, a0, a1);
      ctx.strokeStyle = 'rgba(90,150,190,0.16)'; ctx.lineWidth = R * 0.20; ctx.stroke();
      /* danger band */
      if (danger != null) {
        const df = Math.max(0, Math.min(1, (danger - vmin) / (vmax - vmin)));
        ctx.beginPath(); ctx.arc(cx, cy, R, a0 + (a1 - a0) * df, a1);
        ctx.strokeStyle = 'rgba(255,77,106,0.30)'; ctx.lineWidth = R * 0.20; ctx.stroke();
      }
      /* value arc */
      ctx.beginPath(); ctx.arc(cx, cy, R, a0, a0 + (a1 - a0) * f);
      ctx.strokeStyle = color; ctx.lineWidth = R * 0.20;
      ctx.shadowColor = color; ctx.shadowBlur = 10; ctx.stroke(); ctx.shadowBlur = 0;
      /* ticks */
      ctx.strokeStyle = 'rgba(150,210,240,0.35)'; ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i <= 10; i++) {
        const a = a0 + (a1 - a0) * i / 10;
        const r0 = R * 1.14, r1 = R * (i % 5 === 0 ? 1.26 : 1.20);
        ctx.moveTo(cx + Math.cos(a) * r0, cy + Math.sin(a) * r0);
        ctx.lineTo(cx + Math.cos(a) * r1, cy + Math.sin(a) * r1);
      }
      ctx.stroke();
      /* needle */
      const an = a0 + (a1 - a0) * f;
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(an) * R * 0.35, cy + Math.sin(an) * R * 0.35);
      ctx.lineTo(cx + Math.cos(an) * R * 1.10, cy + Math.sin(an) * R * 1.10);
      ctx.strokeStyle = C.white; ctx.lineWidth = 1.3; ctx.stroke();

      ctx.textAlign = 'center';
      ctx.font = '700 ' + (R * 0.46).toFixed(0) + 'px ' + FONT;
      ctx.fillStyle = C.white; ctx.textBaseline = 'middle';
      ctx.fillText(v >= 100 ? v.toFixed(0) : v.toFixed(v < 10 ? 2 : 1), cx, cy + R * 0.06);
      ctx.font = '7.5px ' + FONT;
      ctx.fillStyle = 'rgba(140,200,230,0.7)';
      ctx.fillText(unit, cx, cy + R * 0.42);
      ctx.fillStyle = 'rgba(120,225,255,0.9)';
      ctx.fillText(label, cx, cy + R * 1.52);
    },

    /* small labelled numeric readout */
    readout(ctx, x, y, w, h, label, val, unit, color, bar) {
      ctx.fillStyle = 'rgba(255,255,255,0.028)';
      ctx.fillRect(x, y, w, h);
      ctx.strokeStyle = 'rgba(90,180,220,0.16)';
      ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
      ctx.font = '7px ' + FONT;
      ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillStyle = 'rgba(130,185,215,0.75)';
      ctx.fillText(label, x + 4, y + 3.5);
      ctx.font = '700 11px ' + FONT;
      ctx.fillStyle = color || C.white;
      ctx.textBaseline = 'alphabetic';
      ctx.fillText(val, x + 4, y + h - (bar != null ? 8 : 5));
      if (unit) {
        ctx.font = '7px ' + FONT;
        ctx.fillStyle = 'rgba(130,185,215,0.6)';
        ctx.textAlign = 'right';
        ctx.fillText(unit, x + w - 4, y + h - (bar != null ? 9 : 6));
      }
      if (bar != null) {
        const bw = (w - 8) * Math.max(0, Math.min(1, bar));
        ctx.fillStyle = 'rgba(90,150,190,0.18)';
        ctx.fillRect(x + 4, y + h - 5, w - 8, 2);
        ctx.fillStyle = color || C.cyan;
        ctx.fillRect(x + 4, y + h - 5, bw, 2);
      }
    }
  };

  /* ================================================================ panels */
  /* Each entry: id, title, unit, slot, h (css height), draw(ctx,w,h,P,H,t)  */
  const PANELS = [

    /* ---------------------------------------------------------- LEFT ---- */
    {
      id: 'temp', title: 'Plasma Temperature vs Time', unit: 'keV', slot: 'left', h: 96,
      draw(ctx, w, h, P, H) {
        const b = G.box(w, h);
        const mx = Math.max(H.max('Ti', 900), 5) * 1.18;
        G.grid(ctx, b, 6, 4, 0, mx, v => v.toFixed(0));
        G.series(ctx, b, H, 'Te', 900, 0, mx, C.cyan, { fill: 'rgba(63,224,255,0.13)' });
        G.series(ctx, b, H, 'Ti', 900, 0, mx, C.mag);
        G.legend(ctx, { x: b.x, y: b.y - 12 }, [{ c: C.mag, t: 'Ti' }, { c: C.cyan, t: 'Te' }]);
        G.value(ctx, w - 8, h - 3, 'Ti ' + P.Ti.toFixed(2) + '  Te ' + P.Te.toFixed(2), C.white, 9);
      }
    },
    {
      id: 'dens', title: 'Plasma Density vs Time', unit: '10²⁰ m⁻³', slot: 'left', h: 92,
      draw(ctx, w, h, P, H) {
        const b = G.box(w, h);
        const mx = Math.max(H.max('nG', 900), H.max('ne', 900)) * 1.2 || 1;
        G.grid(ctx, b, 6, 4, 0, mx, v => v.toFixed(1));
        /* Greenwald limit */
        ctx.save(); ctx.setLineDash([3, 3]);
        G.series(ctx, b, H, 'nG', 900, 0, mx, 'rgba(255,77,106,0.85)',
          { glow: false, head: false, width: 1 });
        ctx.restore();
        G.series(ctx, b, H, 'ne', 900, 0, mx, C.violet, { fill: 'rgba(169,139,255,0.16)' });
        G.series(ctx, b, H, 'nHe', 900, 0, mx, C.amber, { width: 1, glow: false });
        G.legend(ctx, { x: b.x, y: b.y - 12 }, [
          { c: C.violet, t: 'ne' }, { c: C.amber, t: 'nHe' }, { c: '#ff4d6a', t: 'nGW' }]);
        G.value(ctx, w - 8, h - 3, 'f_GW ' + P.fG.toFixed(2), P.fG > 1 ? C.red : C.white, 9);
      }
    },
    {
      id: 'pfus', title: 'Fusion Power Output', unit: 'MW', slot: 'left', h: 104,
      draw(ctx, w, h, P, H) {
        const b = G.box(w, h);
        const mx = Math.max(H.max('Pfus', 900) * 1.2, 20);
        G.grid(ctx, b, 6, 4, 0, mx, v => v.toFixed(0));
        G.series(ctx, b, H, 'Pfus', 900, 0, mx, C.mag,
          { fill: 'rgba(255,92,224,0.22)', width: 1.7 });
        G.series(ctx, b, H, 'Palpha', 900, 0, mx, C.amber, { width: 1, glow: false });
        G.series(ctx, b, H, 'Paux', 900, 0, mx, C.cyan, { width: 1, glow: false });
        G.legend(ctx, { x: b.x, y: b.y - 12 }, [
          { c: C.mag, t: 'P_fus' }, { c: C.amber, t: 'P_α' }, { c: C.cyan, t: 'P_aux' }]);
        G.value(ctx, w - 8, h - 16, P.Pfus.toFixed(1) + ' MW', C.mag, 16);
        G.value(ctx, w - 8, h - 4, 'Q = ' + (P.Q > 900 ? '∞' : P.Q.toFixed(2)),
          P.Q >= 10 ? C.green : C.white, 10);
      }
    },
    {
      id: 'neutron', title: 'Neutron Flux — 14.1 MeV', unit: 'n·s⁻¹', slot: 'left', h: 88,
      draw(ctx, w, h, P, H) {
        const b = G.box(w, h);
        const lo = 12, hi = 21;
        G.grid(ctx, b, 6, 3, lo, hi, v => '1e' + Math.round(Math.log10(v)), true);
        G.series(ctx, b, H, 'neutron', 900, lo, hi, C.green,
          { log: true, fill: 'rgba(77,255,176,0.13)' });
        G.value(ctx, w - 8, h - 4, expo(P.neutronRate, 2) + ' n/s', C.green, 10);
      }
    },
    {
      id: 'tau', title: 'Energy Confinement Time', unit: 's', slot: 'left', h: 88,
      draw(ctx, w, h, P, H) {
        const b = G.box(w, h);
        const mx = Math.max(H.max('tauE', 900), H.max('tauE98', 900)) * 1.25 || 1;
        G.grid(ctx, b, 6, 3, 0, mx, v => v.toFixed(1));
        G.series(ctx, b, H, 'tauE98', 900, 0, mx, 'rgba(91,140,255,0.75)',
          { width: 1, glow: false });
        G.series(ctx, b, H, 'tauE', 900, 0, mx, C.cyan, { fill: 'rgba(63,224,255,0.12)' });
        G.legend(ctx, { x: b.x, y: b.y - 12 }, [
          { c: C.cyan, t: 'τE' }, { c: C.blue, t: 'IPB98(y,2)' }]);
        G.value(ctx, w - 8, h - 4, 'τE ' + P.tauE.toFixed(2) + 's   H98 ' + P.H98.toFixed(2),
          C.white, 9);
      }
    },
    {
      id: 'rad', title: 'Radiation Losses', unit: 'MW', slot: 'left', h: 92,
      draw(ctx, w, h, P, H) {
        const b = G.box(w, h);
        const mx = Math.max(H.max('Prad', 900) * 1.25, 10);
        G.grid(ctx, b, 6, 3, 0, mx, v => v.toFixed(0));
        G.series(ctx, b, H, 'Prad', 900, 0, mx, C.red, { fill: 'rgba(255,77,106,0.16)' });
        G.series(ctx, b, H, 'Pbrem', 900, 0, mx, C.amber, { width: 1, glow: false });
        G.series(ctx, b, H, 'Psync', 900, 0, mx, C.blue, { width: 1, glow: false });
        G.legend(ctx, { x: b.x, y: b.y - 12 }, [
          { c: C.red, t: 'total' }, { c: C.amber, t: 'brems' }, { c: C.blue, t: 'sync' }]);
        G.value(ctx, w - 8, h - 4, P.Prad.toFixed(1) + ' MW  f_rad ' +
          (P.Prad / Math.max(P.Paux + P.Pohm + P.Palpha, 0.1)).toFixed(2), C.white, 9);
      }
    },
    {
      id: 'xsec', title: 'Plasma Cross-Section — Poloidal Flux', unit: 'ψ', slot: 'left', h: 176,
      draw(ctx, w, h, P) {
        ctx.save();
        const cx = w * 0.5, cy = h * 0.52 + 6;
        const sc = Math.min(w * 0.40, (h - 34) * 0.5) / (P.M.a * P.M.kappa * 0.62);
        const R0 = P.M.a, a = P.M.a, k = P.M.kappa, d = P.M.delta;
        const S = sc * 0.5;

        /* vessel outline */
        ctx.strokeStyle = 'rgba(120,180,210,0.35)'; ctx.lineWidth = 1.4;
        ctx.beginPath();
        for (let i = 0; i <= 64; i++) {
          const th = i / 64 * Math.PI * 2;
          const rr = 1.34, kk = k * 1.05;
          const X = cx + S * rr * a * Math.cos(th + 0.12 * Math.sin(th));
          const Y = cy - S * rr * a * kk * Math.sin(th) * 0.78;
          i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
        }
        ctx.closePath(); ctx.stroke();

        /* flux surfaces, coloured by local temperature */
        const NRp = Tokamak.NR;
        for (let s = NRp - 2; s >= 1; s -= 2) {
          const rho = Tokamak.RHO[s];
          const Tn = Math.min(1, P.TeProf[s] / Math.max(P.TeProf[0], 0.4));
          const col = ramp(PLASMA_STOPS, Tn);
          ctx.beginPath();
          for (let i = 0; i <= 56; i++) {
            const th = i / 56 * Math.PI * 2;
            const X = cx + S * a * rho * Math.cos(th + d * rho * Math.sin(th));
            const Y = cy - S * a * rho * k * Math.sin(th) * 0.78 -
                      S * 0.10 * (1 - rho * rho);          /* Shafranov shift */
            i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
          }
          ctx.closePath();
          ctx.strokeStyle = 'rgba(' + (col[0] | 0) + ',' + (col[1] | 0) + ',' +
                            (col[2] | 0) + ',0.85)';
          ctx.lineWidth = 1.1; ctx.stroke();
        }

        /* separatrix + divertor legs through the X-point */
        const xpX = cx - S * a * 0.30, xpY = cy + S * a * k * 0.78 * 1.02;
        ctx.strokeStyle = P.hMode ? 'rgba(255,92,224,0.95)' : 'rgba(120,200,240,0.7)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (let i = 0; i <= 72; i++) {
          const th = -Math.PI * 0.72 + (i / 72) * Math.PI * 2 * 0.86;
          const X = cx + S * a * 1.0 * Math.cos(th + d * Math.sin(th));
          const Y = cy - S * a * k * Math.sin(th) * 0.78 - S * 0.10;
          i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
        }
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(xpX, xpY);
        ctx.lineTo(cx - S * a * 0.88, cy + S * a * k * 0.78 * 1.30);
        ctx.moveTo(xpX, xpY);
        ctx.lineTo(cx + S * a * 0.55, cy + S * a * k * 0.78 * 1.32);
        ctx.strokeStyle = 'rgba(255,140,220,0.8)'; ctx.lineWidth = 1.2; ctx.stroke();
        /* X-point marker */
        ctx.strokeStyle = C.white; ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(xpX - 3, xpY - 3); ctx.lineTo(xpX + 3, xpY + 3);
        ctx.moveTo(xpX + 3, xpY - 3); ctx.lineTo(xpX - 3, xpY + 3);
        ctx.stroke();
        /* magnetic axis */
        ctx.fillStyle = C.white;
        ctx.beginPath(); ctx.arc(cx, cy - S * 0.10, 1.8, 0, 6.284); ctx.fill();

        ctx.font = '7px ' + FONT; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        ctx.fillStyle = 'rgba(150,210,235,0.8)';
        ctx.fillText('κ=' + P.M.kappa.toFixed(2) + '  δ=' + P.M.delta.toFixed(2), 8, h - 22);
        ctx.fillText('q95=' + P.q95.toFixed(2) + '  q0=' + P.q0.toFixed(2), 8, h - 13);
        ctx.textAlign = 'right';
        ctx.fillStyle = P.hMode ? C.mag : 'rgba(150,210,235,0.8)';
        ctx.fillText(P.hMode ? 'H-MODE PEDESTAL' : 'L-MODE', w - 8, h - 13);
        ctx.restore();
      }
    },

    /* --------------------------------------------------------- RIGHT ---- */
    {
      id: 'gauges', title: 'Primary Machine Parameters', unit: '', slot: 'right', h: 128,
      draw(ctx, w, h, P) {
        const R = Math.min(w / 4.9, 30);
        const y = h * 0.44;
        const xs = [w * 0.145, w * 0.383, w * 0.62, w * 0.857];
        G.gauge(ctx, xs[0], y, R, P.Bt, 0, 6, 'TOROIDAL FIELD', 'T', C.cyan);
        G.gauge(ctx, xs[1], y, R, P.Ip, 0, 17, 'PLASMA CURRENT', 'MA', C.violet, 16);
        G.gauge(ctx, xs[2], y, R, Math.min(P.Q, 25), 0, 25, 'FUSION GAIN', 'Q', C.mag);
        G.gauge(ctx, xs[3], y, R, P.betaN, 0, 5, 'NORMALISED BETA', 'βN', C.amber, 4.0);
      }
    },
    {
      id: 'scope', title: 'Oscilloscope — MHD & Dα', unit: 'a.u.', slot: 'right', h: 108,
      draw(ctx, w, h, P, H, t) {
        const b = G.box(w, h);
        /* phosphor grid */
        ctx.strokeStyle = 'rgba(80,255,180,0.09)'; ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i <= 10; i++) {
          const x = Math.round(b.x + b.w * i / 10) + 0.5;
          ctx.moveTo(x, b.y); ctx.lineTo(x, b.y + b.h);
        }
        for (let i = 0; i <= 6; i++) {
          const y = Math.round(b.y + b.h * i / 6) + 0.5;
          ctx.moveTo(b.x, y); ctx.lineTo(b.x + b.w, y);
        }
        ctx.stroke();
        ctx.strokeStyle = 'rgba(80,255,180,0.22)';
        ctx.beginPath();
        ctx.moveTo(b.x, b.y + b.h / 2 + 0.5); ctx.lineTo(b.x + b.w, b.y + b.h / 2 + 0.5);
        ctx.stroke();

        /* Mirnov coil dB/dt — high time resolution, drawn directly */
        ctx.beginPath();
        const N = 260;
        for (let i = 0; i < N; i++) {
          const f = i / (N - 1);
          const tt = t * 1000 - (1 - f) * 220;
          const base = Math.sin(tt * 0.11 + P.mhdPhase) * 0.35 +
                       Math.sin(tt * 0.29) * 0.18 + Math.sin(tt * 0.53) * 0.09;
          const amp = (P.hMode ? 0.35 : 0.9) * Math.min(2, P.beta * 0.6 + 0.2);
          let v = base * amp;
          v += (Math.random() - 0.5) * 0.10;
          v += P.elmFlash * Math.sin(tt * 1.9) * 2.2 * Math.exp(-(1 - f) * 3);
          v += P.sawFlash * Math.sin(tt * 3.1) * 1.4;
          if (P.disrupted) v += (Math.random() - 0.5) * 4;
          const y = b.y + b.h / 2 - Math.max(-1, Math.min(1, v / 2.2)) * b.h * 0.45;
          i ? ctx.lineTo(b.x + b.w * f, y) : ctx.moveTo(b.x + b.w * f, y);
        }
        ctx.strokeStyle = '#5dffa0'; ctx.lineWidth = 1.1;
        ctx.shadowColor = '#5dffa0'; ctx.shadowBlur = 6; ctx.stroke(); ctx.shadowBlur = 0;

        /* D-alpha trace */
        G.series(ctx, b, H, 'dAlpha', 170, 0, 5, C.mag, { width: 1.2 });
        G.legend(ctx, { x: b.x, y: b.y - 12 }, [
          { c: '#5dffa0', t: 'Mirnov dB/dt' }, { c: C.mag, t: 'Dα' }]);
        ctx.font = '7px ' + FONT; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
        ctx.fillStyle = 'rgba(150,210,235,0.7)';
        ctx.fillText('2 ms/div   ELM #' + P.elmCount + '  ST #' + P.sawCount, w - 8, h - 3);
      }
    },
    {
      id: 'coils', title: 'Superconducting Coil Currents', unit: 'kA', slot: 'right', h: 112,
      draw(ctx, w, h, P) {
        const pad = 8;
        const bw = (w - pad * 2) / P.M.nTF;
        const top = 24, bh = 40;
        ctx.font = '7px ' + FONT; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        ctx.fillStyle = 'rgba(130,185,215,0.7)';
        ctx.fillText('TF 1–18  (Nb₃Sn @ 4.5 K)', pad, top - 10);
        for (let i = 0; i < P.M.nTF; i++) {
          const v = P.coilI[i] / 72;
          const x = pad + i * bw;
          ctx.fillStyle = 'rgba(90,150,190,0.14)';
          ctx.fillRect(x + 1, top, bw - 2, bh);
          const hh = bh * Math.max(0, Math.min(1, v));
          const col = ramp(THERMAL_STOPS, v);
          ctx.fillStyle = rgbStr(col);
          ctx.fillRect(x + 1, top + bh - hh, bw - 2, hh);
        }
        /* PF coils, bipolar */
        const top2 = top + bh + 18, bh2 = 30;
        ctx.fillStyle = 'rgba(130,185,215,0.7)';
        ctx.fillText('PF 1–6  (NbTi, bipolar)', pad, top2 - 10);
        const bw2 = (w - pad * 2) / P.M.nPF;
        const mid = top2 + bh2 / 2;
        ctx.strokeStyle = 'rgba(120,200,240,0.25)';
        ctx.beginPath(); ctx.moveTo(pad, mid + 0.5); ctx.lineTo(w - pad, mid + 0.5); ctx.stroke();
        for (let i = 0; i < P.M.nPF; i++) {
          const v = P.pfI[i] / 45;
          const x = pad + i * bw2;
          const hh = Math.max(-1, Math.min(1, v)) * bh2 / 2;
          ctx.fillStyle = v >= 0 ? 'rgba(63,224,255,0.85)' : 'rgba(255,92,224,0.85)';
          ctx.fillRect(x + 2, hh >= 0 ? mid - hh : mid, bw2 - 4, Math.abs(hh));
          ctx.font = '6.5px ' + FONT; ctx.textAlign = 'center';
          ctx.fillStyle = 'rgba(150,200,225,0.65)';
          ctx.fillText('PF' + (i + 1), x + bw2 / 2, top2 + bh2 + 3);
          ctx.textAlign = 'left';
        }
        ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
        ctx.font = '7px ' + FONT; ctx.fillStyle = 'rgba(150,210,235,0.75)';
        ctx.fillText('TF stored energy 41 GJ   quench detect: ARMED', w - pad, h - 3);
      }
    },
    {
      id: 'power', title: 'Power Balance Diagram', unit: 'MW', slot: 'right', h: 122,
      draw(ctx, w, h, P) {
        const pad = 10, top = 24;
        const inputs = [
          { l: 'NBI', v: P.Pnbi, c: C.cyan },
          { l: 'ICRH', v: P.Picr, c: C.blue },
          { l: 'ECRH', v: P.Pecr, c: C.violet },
          { l: 'OHMIC', v: P.Pohm, c: C.amber },
          { l: 'ALPHA', v: P.Palpha, c: C.mag }
        ];
        const outputs = [
          { l: 'TRANSPORT', v: Math.max(P.W / P.tauE, 0), c: C.green },
          { l: 'BREMS', v: P.Pbrem, c: C.amber },
          { l: 'LINE RAD', v: P.Pline, c: C.red },
          { l: 'SYNCHROTRON', v: P.Psync, c: C.blue }
        ];
        const tin = inputs.reduce((a, x) => a + x.v, 0);
        const tout = outputs.reduce((a, x) => a + x.v, 0);
        const scale = (w - pad * 2 - 62) / Math.max(tin, tout, 1);

        ctx.font = '7px ' + FONT; ctx.textBaseline = 'middle';
        const row = (arr, y0) => {
          arr.forEach((it, i) => {
            const y = y0 + i * 12;
            ctx.textAlign = 'right';
            ctx.fillStyle = 'rgba(140,195,220,0.8)';
            ctx.fillText(it.l, pad + 58, y + 4);
            const bw = Math.max(1, it.v * scale);
            ctx.fillStyle = it.c;
            ctx.shadowColor = it.c; ctx.shadowBlur = 5;
            ctx.fillRect(pad + 62, y, bw, 6);
            ctx.shadowBlur = 0;
            ctx.textAlign = 'left';
            ctx.fillStyle = 'rgba(230,245,255,0.9)';
            ctx.fillText(it.v.toFixed(1), pad + 66 + bw, y + 4);
          });
        };
        ctx.fillStyle = 'rgba(120,225,255,0.85)';
        ctx.textAlign = 'left';
        ctx.fillText('HEATING  Σ ' + tin.toFixed(1) + ' MW', pad, top - 6);
        row(inputs, top);
        const y2 = top + inputs.length * 12 + 12;
        ctx.fillStyle = 'rgba(120,225,255,0.85)';
        ctx.fillText('LOSSES  Σ ' + tout.toFixed(1) + ' MW', pad, y2 - 6);
        row(outputs, y2);

        const net = P.Pfus * 0.33 - P.Paux / 0.4;
        ctx.textAlign = 'right'; ctx.font = '700 9px ' + FONT;
        ctx.fillStyle = net > 0 ? C.green : C.red;
        ctx.fillText('NET ELECTRIC ' + (net >= 0 ? '+' : '') + net.toFixed(1) + ' MWe',
          w - pad, h - 8);
      }
    },
    {
      id: 'heatflux', title: 'Divertor Heat Flux Distribution', unit: 'MW·m⁻²', slot: 'right', h: 104,
      draw(ctx, w, h, P, H, t) {
        const b = { x: 30, y: 24, w: w - 46, h: h - 42 };
        const NX = 44, NY = 20;
        const cw = b.w / NX, ch = b.h / NY;
        const q0 = P.qDiv;
        for (let i = 0; i < NX; i++) {
          /* toroidal angle */
          const phi = i / NX * Math.PI * 2;
          const ripple = 1 + 0.10 * Math.cos(phi * 18) + 0.05 * Math.sin(phi * 3 + t * 0.4);
          for (let j = 0; j < NY; j++) {
            /* poloidal distance from the strike point */
            const s = (j / (NY - 1) - 0.42) * 3.2;
            const lambda = 0.34 + 0.25 * P.elmFlash;
            const prof = Math.exp(-Math.pow(s / lambda, 2)) +
                         0.45 * Math.exp(-Math.pow((s + 1.35) / 0.42, 2));
            const v = q0 * prof * ripple / 22;
            ctx.fillStyle = rgbStr(ramp(PLASMA_STOPS, Math.min(1, v)));
            ctx.fillRect(b.x + i * cw, b.y + j * ch, cw + 0.6, ch + 0.6);
          }
        }
        ctx.strokeStyle = 'rgba(120,200,240,0.3)';
        ctx.strokeRect(b.x + 0.5, b.y + 0.5, b.w, b.h);
        /* colour bar */
        for (let i = 0; i < 20; i++) {
          ctx.fillStyle = rgbStr(ramp(PLASMA_STOPS, i / 19));
          ctx.fillRect(b.x, h - 14, b.w / 20, 5);
          ctx.fillRect(b.x + i * b.w / 20, h - 14, b.w / 20 + 0.6, 5);
        }
        ctx.font = '7px ' + FONT; ctx.fillStyle = 'rgba(150,210,235,0.75)';
        ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        ctx.fillText('0', b.x, h - 8);
        ctx.textAlign = 'right';
        ctx.fillText('22 MW/m²', b.x + b.w, h - 8);
        ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        ctx.fillStyle = 'rgba(140,195,220,0.8)';
        ctx.fillText('OUT', b.x - 4, b.y + b.h * 0.42);
        ctx.fillText('IN', b.x - 4, b.y + b.h * 0.88);
        G.value(ctx, w - 8, 18, 'peak ' + (q0 * 1.45).toFixed(1),
          q0 * 1.45 > 10 ? C.red : C.white, 9);
      }
    },
    {
      id: 'flux', title: 'Magnetic Flux & Loop Voltage', unit: 'Wb / V', slot: 'right', h: 90,
      draw(ctx, w, h, P, H) {
        const b = G.box(w, h);
        const mx = Math.max(H.max('fluxTor', 900), 1) * 1.2;
        G.grid(ctx, b, 6, 3, 0, mx, v => v.toFixed(0));
        G.series(ctx, b, H, 'fluxTor', 900, 0, mx, C.cyan, { fill: 'rgba(63,224,255,0.12)' });
        G.series(ctx, b, H, 'fluxPol', 900, 0, mx, C.violet, { width: 1.2 });
        G.legend(ctx, { x: b.x, y: b.y - 12 }, [
          { c: C.cyan, t: 'Φ_tor' }, { c: C.violet, t: 'Ψ_pol' }]);
        G.value(ctx, w - 8, h - 4,
          'Φ ' + P.fluxTor.toFixed(1) + ' Wb   Vloop ' + P.Vloop.toFixed(2) + ' V',
          C.white, 9);
      }
    },
    {
      id: 'pvac', title: 'Vacuum Chamber Pressure', unit: 'Pa', slot: 'right', h: 84,
      draw(ctx, w, h, P, H) {
        const b = G.box(w, h);
        const lo = -6, hi = -2;
        G.grid(ctx, b, 6, 4, lo, hi, v => '1e' + Math.round(Math.log10(v)), true);
        G.series(ctx, b, H, 'pVac', 900, lo, hi, C.green,
          { log: true, fill: 'rgba(77,255,176,0.12)' });
        G.value(ctx, w - 8, h - 4, expo(P.pVac, 2) + ' Pa', C.green, 10);
      }
    },

    /* -------------------------------------------------------- BOTTOM ---- */
    {
      id: 'spectro', title: 'Te(ρ,t) Colour Heat Map', unit: 'keV', slot: 'bottom', h: 132, flex: 1.35,
      draw(ctx, w, h, P, H) {
        const b = { x: 26, y: 22, w: w - 42, h: h - 46 };
        if (!this._img) {
          this._off = document.createElement('canvas');
          this._off.width = H.pw; this._off.height = H.ph;
          this._octx = this._off.getContext('2d');
          this._img = this._octx.createImageData(H.pw, H.ph);
        }
        const img = this._img, d = img.data;
        let peak = 0.5;
        for (let i = 0; i < H.prof.length; i++) if (H.prof[i] > peak) peak = H.prof[i];
        for (let r = 0; r < H.ph; r++) {
          for (let c = 0; c < H.pw; c++) {
            /* shift so the newest column is on the right */
            const src = (c + H.profHead) % H.pw;
            const v = H.prof[r * H.pw + src] / peak;
            const col = ramp(PLASMA_STOPS, v);
            const o = (r * H.pw + c) * 4;
            d[o] = col[0]; d[o + 1] = col[1]; d[o + 2] = col[2]; d[o + 3] = 255;
          }
        }
        this._octx.putImageData(img, 0, 0);
        ctx.save();
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(this._off, b.x, b.y, b.w, b.h);
        ctx.restore();
        ctx.strokeStyle = 'rgba(120,200,240,0.3)';
        ctx.strokeRect(b.x + 0.5, b.y + 0.5, b.w, b.h);
        ctx.font = '7px ' + FONT; ctx.fillStyle = 'rgba(150,210,235,0.75)';
        ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        ctx.fillText('ρ=0', b.x - 4, b.y + 4);
        ctx.fillText('ρ=1', b.x - 4, b.y + b.h - 4);
        ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        ctx.fillText('t − 56 s', b.x, b.y + b.h + 5);
        ctx.textAlign = 'right';
        ctx.fillText('now', b.x + b.w, b.y + b.h + 5);
        ctx.textAlign = 'center';
        ctx.fillStyle = 'rgba(200,235,250,0.9)';
        ctx.fillText('peak ' + peak.toFixed(1) + ' keV', b.x + b.w / 2, b.y + b.h + 5);
      }
    },
    {
      id: 'fieldlines', title: 'Magnetic Field Line Topology', unit: 'q(ρ)', slot: 'bottom', h: 132, flex: 1,
      draw(ctx, w, h, P, H, t) {
        const cx = w * 0.5, cy = h * 0.52 + 4;
        const R = Math.min(w * 0.40, (h - 40) * 0.86);
        const r0 = R * 0.62, rm = R * 0.30;
        /* project a helical field line on the torus */
        const draw = (rho, col, phase, width) => {
          const q = P.q0 + (P.q95 - P.q0) * rho * rho;
          ctx.beginPath();
          const N = 420;
          for (let i = 0; i <= N; i++) {
            const u = i / N * Math.PI * 2 * 3;             // 3 toroidal turns
            const v = u / Math.max(q, 0.4) + phase;        // poloidal winding
            const rr = r0 + rm * rho * Math.cos(v);
            const X = cx + rr * Math.cos(u);
            const Y = cy + (rr * Math.sin(u)) * 0.34 + rm * rho * Math.sin(v) * 0.86;
            i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
          }
          ctx.strokeStyle = col; ctx.lineWidth = width || 1;
          ctx.shadowColor = col; ctx.shadowBlur = 5; ctx.stroke(); ctx.shadowBlur = 0;
        };
        /* torus silhouette */
        ctx.strokeStyle = 'rgba(120,180,210,0.22)'; ctx.lineWidth = 1;
        [r0 - rm, r0 + rm].forEach(rr => {
          ctx.beginPath();
          ctx.ellipse(cx, cy, rr, rr * 0.34, 0, 0, Math.PI * 2);
          ctx.stroke();
        });
        draw(0.30, 'rgba(63,224,255,0.75)', t * 0.25, 1);
        draw(0.62, 'rgba(169,139,255,0.8)', -t * 0.18 + 1.7, 1.1);
        draw(0.92, 'rgba(255,92,224,0.85)', t * 0.12 + 3.1, 1.2);
        ctx.font = '7px ' + FONT; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        ctx.fillStyle = 'rgba(150,210,235,0.8)';
        ctx.fillText('ρ=0.3  q=' + (P.q0 + (P.q95 - P.q0) * 0.09).toFixed(2), 8, h - 30);
        ctx.fillText('ρ=0.6  q=' + (P.q0 + (P.q95 - P.q0) * 0.38).toFixed(2), 8, h - 21);
        ctx.fillText('ρ=0.9  q=' + (P.q0 + (P.q95 - P.q0) * 0.85).toFixed(2), 8, h - 12);
        ctx.textAlign = 'right';
        ctx.fillStyle = P.q0 < 1 ? C.amber : 'rgba(150,210,235,0.8)';
        ctx.fillText(P.q0 < 1 ? 'q=1 SURFACE PRESENT — SAWTEETH' : 'NO q=1 SURFACE',
          w - 8, h - 12);
      }
    },
    {
      id: 'profiles', title: 'Radial Profiles', unit: 'n, T vs ρ', slot: 'bottom', h: 132, flex: 1,
      draw(ctx, w, h, P) {
        const b = { x: 30, y: 24, w: w - 44, h: h - 44 };
        G.grid(ctx, b, 5, 4, 0, 1, () => '');
        const NRp = Tokamak.NR;
        const plot = (arr, scale, col, fill) => {
          ctx.beginPath();
          for (let i = 0; i < NRp; i++) {
            const x = b.x + b.w * Tokamak.RHO[i];
            const y = b.y + b.h * (1 - Math.min(1, arr[i] / scale));
            i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
          }
          if (fill) {
            ctx.lineTo(b.x + b.w, b.y + b.h); ctx.lineTo(b.x, b.y + b.h); ctx.closePath();
            const g = ctx.createLinearGradient(0, b.y, 0, b.y + b.h);
            g.addColorStop(0, fill); g.addColorStop(1, 'rgba(0,0,0,0)');
            ctx.fillStyle = g; ctx.fill();
            ctx.beginPath();
            for (let i = 0; i < NRp; i++) {
              const x = b.x + b.w * Tokamak.RHO[i];
              const y = b.y + b.h * (1 - Math.min(1, arr[i] / scale));
              i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
            }
          }
          ctx.strokeStyle = col; ctx.lineWidth = 1.4;
          ctx.shadowColor = col; ctx.shadowBlur = 5; ctx.stroke(); ctx.shadowBlur = 0;
        };
        const Tmax = Math.max(P.TiProf[0], P.TeProf[0], 1) * 1.1;
        const nmax = Math.max(P.nProf[0], 0.1) * 1.15;
        const smax = Math.max(P.sfProf[0], 1e-4) * 1.15;
        plot(P.sfProf, smax, 'rgba(255,182,72,0.9)', 'rgba(255,182,72,0.20)');
        plot(P.nProf, nmax, C.violet);
        plot(P.TeProf, Tmax, C.cyan);
        plot(P.TiProf, Tmax, C.mag);
        G.legend(ctx, { x: b.x, y: 12 }, [
          { c: C.mag, t: 'Ti' }, { c: C.cyan, t: 'Te' },
          { c: C.violet, t: 'ne' }, { c: C.amber, t: 'S_fus' }]);
        ctx.font = '7px ' + FONT; ctx.fillStyle = 'rgba(150,210,235,0.75)';
        ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        ctx.fillText('ρ = 0', b.x, b.y + b.h + 4);
        ctx.textAlign = 'right';
        ctx.fillText('ρ = 1', b.x + b.w, b.y + b.h + 4);
        ctx.textAlign = 'center';
        ctx.fillText('T0=' + Tmax.toFixed(1) + 'keV  n0=' + nmax.toFixed(2), b.x + b.w / 2, b.y + b.h + 4);
      }
    },
    {
      id: 'sensors', title: 'Sensor Readouts', unit: '', slot: 'bottom', h: 132, flex: 1.25,
      draw(ctx, w, h, P) {
        const cols = 4, rows = 4;
        const pad = 7, gap = 4;
        const cw = (w - pad * 2 - gap * (cols - 1)) / cols;
        const chh = (h - 24 - pad - gap * (rows - 1)) / rows;
        const items = [
          ['W_TH', P.W.toFixed(1), 'MJ', C.cyan, P.W / 400],
          ['β_TOR', P.beta.toFixed(2), '%', C.amber, P.beta / 5],
          ['β_N', P.betaN.toFixed(2), '—', P.betaN > 3.5 ? C.red : C.green, P.betaN / 5],
          ['Z_EFF', P.Zeff.toFixed(2), '—', C.violet, (P.Zeff - 1) / 3.5],
          ['q95', P.q95.toFixed(2), '—', P.q95 < 3 ? C.amber : C.cyan, P.q95 / 8],
          ['l_i', P.li.toFixed(2), '—', C.blue, P.li / 1.5],
          ['n_He/n_e', (P.nHe / Math.max(P.ne, 1e-6) * 100).toFixed(1), '%', C.amber,
            P.nHe / Math.max(P.ne, 1e-6) / 0.15],
          ['f_DT', ((P.fDT || 0) * 100).toFixed(0), '%', C.green, P.fDT || 0],
          ['V_LOOP', P.Vloop.toFixed(2), 'V', C.cyan, P.Vloop / 3],
          ['W_MAG', P.Wmag.toFixed(0), 'MJ', C.violet, P.Wmag / 400],
          ['P_SEP', P.Psep.toFixed(1), 'MW', C.mag, P.Psep / 150],
          ['P_LH', P.Pthresh.toFixed(1), 'MW', C.blue, P.Pthresh / 150],
          ['q_DIV', P.qDiv.toFixed(1), 'MW/m²', P.qDiv > 10 ? C.red : C.green, P.qDiv / 15],
          ['τ_He', (5 * P.tauE).toFixed(2), 's', C.amber, P.tauE * 5 / 12],
          ['n/nGW', P.fG.toFixed(2), '—', P.fG > 1 ? C.red : C.cyan, P.fG],
          ['P_NET', (P.Pfus * 0.33 - P.Paux / 0.4).toFixed(0), 'MWe',
            P.Pfus * 0.33 - P.Paux / 0.4 > 0 ? C.green : C.red,
            (P.Pfus * 0.33 - P.Paux / 0.4) / 150]
        ];
        items.forEach((it, i) => {
          const c = i % cols, r = (i / cols) | 0;
          G.readout(ctx, pad + c * (cw + gap), 24 + r * (chh + gap), cw, chh,
            it[0], it[1], it[2], it[3], it[4]);
        });
      }
    },
    {
      id: 'status', title: 'Reactor Status Dashboard', unit: '', slot: 'bottom', h: 132, flex: 1.1,
      draw(ctx, w, h, P, H, t) {
        const pad = 8;
        let y = 24;
        /* subsystem LEDs */
        const sys = [
          ['TF MAGNET', P.Bt > 4.5 ? 2 : P.Bt > 1 ? 1 : 0],
          ['PF / CS', P.Ip > 1 ? 2 : 1],
          ['CRYOPLANT', 2],
          ['VACUUM', P.pVac < 1e-3 ? 2 : 1],
          ['FUELLING', P.gas > 0.05 ? 2 : 1],
          ['NBI', P.Pnbi > 1 ? 2 : 0],
          ['ICRH', P.Picr > 1 ? 2 : 0],
          ['ECRH', P.Pecr > 1 ? 2 : 0],
          ['TRITIUM PLANT', 2],
          ['DIVERTOR', P.qDiv > 12 ? 1 : 2],
          ['DISRUPT MIT.', P.disrupted ? 0 : 2],
          ['NEUTRON SHIELD', 2]
        ];
        const cols = 2;
        const cw = (w - pad * 2) / cols;
        ctx.font = '7px ' + FONT; ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
        sys.forEach((s, i) => {
          const c = i % cols, r = (i / cols) | 0;
          const x = pad + c * cw, yy = y + r * 11 + 5;
          const col = s[1] === 2 ? '#4dffb0' : s[1] === 1 ? '#ffb648' : '#ff4d6a';
          const blink = s[1] === 0 ? (Math.sin(t * 8) > 0 ? 1 : 0.25) : 1;
          ctx.globalAlpha = blink;
          ctx.fillStyle = col;
          ctx.shadowColor = col; ctx.shadowBlur = 5;
          ctx.beginPath(); ctx.arc(x + 3, yy, 2.4, 0, 6.284); ctx.fill();
          ctx.shadowBlur = 0; ctx.globalAlpha = 1;
          ctx.fillStyle = 'rgba(180,215,235,0.85)';
          ctx.fillText(s[0], x + 10, yy);
        });
        y += Math.ceil(sys.length / cols) * 11 + 10;

        /* event log */
        ctx.strokeStyle = 'rgba(90,180,220,0.16)';
        ctx.beginPath(); ctx.moveTo(pad, y - 4); ctx.lineTo(w - pad, y - 4); ctx.stroke();
        ctx.font = '7px ' + FONT; ctx.textBaseline = 'top';
        ctx.fillStyle = 'rgba(120,225,255,0.8)';
        ctx.fillText('EVENT LOG', pad, y);
        y += 10;
        P.alarms.slice(0, 4).forEach(a => {
          const col = a.level === 'crit' ? C.red : a.level === 'warn' ? C.amber : C.green;
          ctx.fillStyle = col;
          ctx.fillText('t=' + a.t.toFixed(1).padStart(6) + 's', pad, y);
          ctx.fillStyle = 'rgba(200,230,245,0.85)';
          const txt = a.text.length > 34 ? a.text.slice(0, 33) + '…' : a.text;
          ctx.fillText(txt, pad + 52, y);
          y += 9;
        });
      }
    }
  ];

  /* ================================================================== HUD */
  class HUD {
    constructor(root) {
      this.root = root;
      this.panels = [];
      this.hist = new History([
        'Te', 'Ti', 'ne', 'nG', 'Pfus', 'Q', 'Ip', 'Bt', 'tauE', 'tauE98',
        'Prad', 'Pbrem', 'Pline', 'Psync', 'Paux', 'Palpha', 'Pohm', 'W',
        'beta', 'betaN', 'q95', 'neutron', 'pVac', 'Zeff', 'qDiv', 'Vloop',
        'fluxTor', 'fluxPol', 'mirnov', 'dAlpha', 'nHe', 'fG', 'H98'
      ]);
      this.build();
      this.visible = true;
      this._acc = 0;
    }

    build() {
      const slots = {
        left: document.getElementById('hud-left'),
        right: document.getElementById('hud-right'),
        bottom: document.getElementById('hud-bottom')
      };
      PANELS.forEach(spec => {
        const el = document.createElement('div');
        el.className = 'panel';
        el.style.height = spec.h + 'px';
        if (spec.flex) el.style.flex = spec.flex + ' 1 0';
        const cv = document.createElement('canvas');
        el.appendChild(cv);
        slots[spec.slot].appendChild(el);
        const p = {
          spec, el, cv, ctx: cv.getContext('2d'), w: 0, h: 0,
          state: Object.create(null)
        };
        this.panels.push(p);
      });
      this.topbar = document.getElementById('hud-top');
    }

    resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      this.panels.forEach(p => {
        const r = p.el.getBoundingClientRect();
        const W = Math.max(2, Math.round(r.width)), Hh = Math.max(2, Math.round(r.height));
        if (W === p.w && Hh === p.h) return;
        p.w = W; p.h = Hh;
        p.cv.width = Math.round(W * dpr); p.cv.height = Math.round(Hh * dpr);
        p.cv.style.width = W + 'px'; p.cv.style.height = Hh + 'px';
        p.dpr = dpr;
      });
    }

    update(P, dt, t) {
      this.hist.push(P, dt);
      if (!this.visible) return;
      this._acc += dt;
      if (this._acc < 1 / 34) return;      /* redraw the dashboard at ~34 Hz */
      this._acc = 0;
      this.resize();
      this.panels.forEach(p => {
        const ctx = p.ctx;
        ctx.setTransform(p.dpr, 0, 0, p.dpr, 0, 0);
        G.frame(null, ctx, p.w, p.h, p.spec.title, p.spec.unit);
        try {
          p.spec.draw.call(p.state, ctx, p.w, p.h, P, this.hist, t);
        } catch (e) {
          ctx.fillStyle = '#ff4d6a'; ctx.font = '9px ' + FONT;
          ctx.fillText('ERR ' + e.message, 8, p.h / 2);
        }
      });
      this.drawTop(P, t);
    }

    drawTop(P, t) {
      const el = this.topbar;
      if (!el._built) {
        el.innerHTML =
          '<div class="tb-brand"><b>ITER-CLASS TOKAMAK</b>' +
          '<span>CONTROLLED THERMONUCLEAR FUSION · MAGNETIC CONFINEMENT</span></div>' +
          '<div class="tb-stats"></div>' +
          '<div class="tb-mode"><span class="mode-txt">IDLE</span>' +
          '<span class="mode-sub">DISCHARGE —</span></div>';
        el._stats = el.querySelector('.tb-stats');
        el._mode = el.querySelector('.mode-txt');
        el._sub = el.querySelector('.mode-sub');
        el._built = true;
      }
      const stat = (l, v, u, cls) =>
        '<div class="tb-stat ' + (cls || '') + '"><i>' + l + '</i><b>' + v +
        '</b><em>' + u + '</em></div>';
      el._stats.innerHTML =
        stat('P_FUS', P.Pfus.toFixed(1), 'MW', 'hl') +
        stat('Q', P.Q > 900 ? '∞' : P.Q.toFixed(2), '', P.Q >= 10 ? 'good' : '') +
        stat('T_i', P.Ti.toFixed(1), 'keV') +
        stat('n_e', P.ne.toFixed(2), '10²⁰m⁻³') +
        stat('I_p', P.Ip.toFixed(2), 'MA') +
        stat('B_t', P.Bt.toFixed(2), 'T') +
        stat('τ_E', P.tauE.toFixed(2), 's') +
        stat('W', P.W.toFixed(0), 'MJ') +
        stat('N', expo(P.neutronRate, 1), 'n/s');
      const m = P.mode;
      el._mode.textContent = m;
      el._mode.className = 'mode-txt ' +
        (m === 'DISRUPTION' ? 'crit' : m === 'BURN' ? 'burn' :
         m === 'H-MODE' ? 'good' : '');
      el._sub.textContent = 'SHOT #' + P.shotNo + '   t = ' + P.t.toFixed(2) + ' s';
    }

    toggle() {
      this.visible = !this.visible;
      this.root.style.opacity = this.visible ? '1' : '0';
      this.root.style.pointerEvents = this.visible ? '' : 'none';
    }
  }

  global.HUD = HUD;
  global.HUD_COLORS = C;
})(window);
