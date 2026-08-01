/* ==========================================================================
   render.js — WebGL2 renderer for the tokamak
   --------------------------------------------------------------------------
   Full-screen SDF raymarch into HDR (RGBA16F when available), a three-level
   bloom pyramid, then an ACES composite with heat shimmer and grain.

   The camera rig carries three states:
     landing   — exterior beauty shot, slow drift around the cutaway
     flight    — the 7 s cinematic move from outside to inside the plasma
     interior  — free orbit within the vacuum vessel

   The flight is a keyframed path through radius / height / pitch / field of
   view, with the cutaway wedge kept open along the camera bearing until the
   lens is inside the first wall, then closed so the torus reads complete.
   ========================================================================== */
(function (global) {
  'use strict';

  /* A raymarched scene costs per pixel, so device pixel ratio matters far
     more than any step count: rendering a 1080p screen at dpr 2 is four
     times the work of dpr 1 for no visible gain on a soft, bloomed image.
     Each preset therefore caps dpr as well as the internal scale.        */
  /* Resolution multiplier for exterior shots, where the volumetric march is
     mostly idle.  Clamped to 1.0 effective scale inside resize(). */
  const EXT_BOOST = 2.3;

  const QUALITY = {
    low:    { scale: 0.45, dpr: 1.00, steps: 24, scene: 56,  lights: 4,  ao: 2, bloom: 3 },
    medium: { scale: 0.60, dpr: 1.00, steps: 36, scene: 72,  lights: 6,  ao: 3, bloom: 3 },
    high:   { scale: 0.80, dpr: 1.00, steps: 56, scene: 96,  lights: 8,  ao: 3, bloom: 3 },
    ultra:  { scale: 1.00, dpr: 1.25, steps: 96, scene: 140, lights: 12, ao: 4, bloom: 3 }
  };

  /* --------------------------------------------------------------------- */
  /*  Cinematic flight path.  p runs 0 (outside the cryostat) -> 1 (inside
      the plasma ring).  Radii are chosen against the machine layout:
        first wall 3.67 | vessel outer 3.80 | TF outer leg 3.96-4.48
        PF ring 4.66-5.18 | cryostat 5.30                                  */
  /* --------------------------------------------------------------------- */
  /* The machine's bounding sphere is ~7 units, so the establishing shot
     needs a ~19 unit standoff at a 24 deg half-vertical field of view. */
  const PATH = [
    { p: 0.00, radius: 20.60, height: 0.80, pitch: 0.30, fov: 0.84, cut: 1.00, ext: 1.00 },
    { p: 0.18, radius: 16.10, height: 0.70, pitch: 0.27, fov: 0.85, cut: 1.00, ext: 1.00 },
    { p: 0.36, radius: 11.00, height: 0.56, pitch: 0.22, fov: 0.88, cut: 1.00, ext: 0.98 },
    { p: 0.53, radius:  7.60, height: 0.44, pitch: 0.15, fov: 0.95, cut: 1.00, ext: 0.88 },
    { p: 0.68, radius:  5.55, height: 0.34, pitch: 0.09, fov: 1.04, cut: 1.00, ext: 0.62 },
    { p: 0.80, radius:  4.40, height: 0.27, pitch: 0.05, fov: 1.14, cut: 1.00, ext: 0.34 },
    { p: 0.90, radius:  3.60, height: 0.22, pitch: 0.03, fov: 1.23, cut: 0.85, ext: 0.14 },
    { p: 0.96, radius:  3.51, height: 0.19, pitch: 0.02, fov: 1.28, cut: 0.36, ext: 0.04 },
    { p: 1.00, radius:  3.45, height: 0.18, pitch: 0.02, fov: 1.30, cut: 0.00, ext: 0.00 }
  ];
  const CHANNELS = ['radius', 'height', 'pitch', 'fov', 'cut', 'ext'];

  function hermite(p0, p1, m0, m1, t) {
    const t2 = t * t, t3 = t2 * t;
    return (2 * t3 - 3 * t2 + 1) * p0 + (t3 - 2 * t2 + t) * m0 +
           (-2 * t3 + 3 * t2) * p1 + (t3 - t2) * m1;
  }

  /* Catmull-Rom sample of the path at parameter p. */
  function samplePath(p, out) {
    p = Math.max(0, Math.min(1, p));
    let i = 0;
    while (i < PATH.length - 2 && p > PATH[i + 1].p) i++;
    const a = PATH[i], b = PATH[i + 1];
    const prev = PATH[Math.max(0, i - 1)], next = PATH[Math.min(PATH.length - 1, i + 2)];
    const span = Math.max(b.p - a.p, 1e-6);
    const t = (p - a.p) / span;
    for (const k of CHANNELS) {
      const m0 = (b[k] - prev[k]) / Math.max(b.p - prev.p, 1e-6) * span;
      const m1 = (next[k] - a[k]) / Math.max(next.p - a.p, 1e-6) * span;
      out[k] = hermite(a[k], b[k], m0, m1, t);
    }
    return out;
  }

  /* Gentle slow-in / slow-out.  The path radii are already close to
     geometric, so a mild ease keeps the apparent zoom rate near constant —
     a hard ease makes the middle of the move rush.                        */
  function easeFlight(x) {
    x = Math.max(0, Math.min(1, x));
    return x * x * (3 - 2 * x);
  }

  /* ===================================================================== */
  class Renderer {
    constructor(canvas) {
      this.canvas = canvas;
      const gl = canvas.getContext('webgl2', {
        alpha: false, antialias: false, depth: false, stencil: false,
        premultipliedAlpha: false, preserveDrawingBuffer: true,
        powerPreference: 'high-performance'
      });
      if (!gl) throw new Error('WebGL2 is required for this simulation.');
      this.gl = gl;

      this.floatOK = !!gl.getExtension('EXT_color_buffer_float');
      gl.getExtension('OES_texture_float_linear');
      this.hdrFmt = this.floatOK ? gl.RGBA16F : gl.RGBA8;
      this.hdrType = this.floatOK ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE;

      this.quality = 'medium';
      this.renderScale = QUALITY.medium.scale;
      this.autoScale = true;
      this.frameMs = 16;
      this._slow = 0; this._fast = 0;

      this.buildQuad();
      this.buildPrograms();
      this.targets = {};
      this.w = 0; this.h = 0;

      /* --- camera ------------------------------------------------------- */
      this.cam = {
        radius: 14.0, yaw: 0.0, pitch: 0.30, height: 0.80,
        fov: 0.62, targetY: 0.02,
        autoOrbit: true, orbitSpeed: 0.020, shake: 0
      };
      this.userYaw = 0; this.userPitch = 0; this.userZoom = 0;

      /* --- shot state --------------------------------------------------- */
      this.shot = {
        mode: 'landing',        // landing | flight | interior
        flight: 0,              // 0..1 progress along PATH
        duration: 7.0,          // seconds
        cut: 1.0,
        ext: 1.0,
        cutCenter: 0.0,         // bearing of the cutaway wedge
        fade: 0.0
      };
      this._pathOut = {};

      /* The two shots need different grades: the exterior is a dark-studio
         product shot, the interior is a blazing light box. A single exposure
         either crushes the machine or washes out the plasma.               */
      this.post = {
        exposureIn: 0.46, exposureOut: 1.30,
        vignetteIn: 0.82, vignetteOut: 0.52,
        bloom: 0.30, shimmer: 1.0, grain: 0.035
      };
    }

    /* ------------------------------------------------------------------ */
    buildQuad() {
      const gl = this.gl;
      this.vao = gl.createVertexArray();
      gl.bindVertexArray(this.vao);
      const buf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      gl.bindVertexArray(null);
    }

    compile(type, src) {
      const gl = this.gl;
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(s);
        console.error(log + '\n' + src.split('\n')
          .map((l, i) => String(i + 1).padStart(4) + ' | ' + l).join('\n'));
        throw new Error('Shader compile failed: ' + log);
      }
      return s;
    }

    link(fragSrc) {
      const gl = this.gl;
      const p = gl.createProgram();
      gl.attachShader(p, this.compile(gl.VERTEX_SHADER, SHADERS.VERT));
      gl.attachShader(p, this.compile(gl.FRAGMENT_SHADER, fragSrc));
      gl.bindAttribLocation(p, 0, 'aPos');
      gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS))
        throw new Error('Link failed: ' + gl.getProgramInfoLog(p));
      const u = {};
      const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
      for (let i = 0; i < n; i++) {
        const info = gl.getActiveUniform(p, i);
        u[info.name] = gl.getUniformLocation(p, info.name);
      }
      return { prog: p, u };
    }

    buildPrograms() {
      this.pScene     = this.link(SHADERS.SCENE);
      this.pBright    = this.link(SHADERS.BRIGHT);
      this.pBlur      = this.link(SHADERS.BLUR);
      this.pComposite = this.link(SHADERS.COMPOSITE);
    }

    /* ------------------------------------------------------------------ */
    makeTarget(w, h, fmt, type) {
      const gl = this.gl;
      w = Math.max(1, w | 0); h = Math.max(1, h | 0);
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, fmt, w, h, 0, gl.RGBA, type, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      const fbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return { tex, fbo, w, h };
    }

    disposeTargets() {
      const gl = this.gl;
      const kill = t => { gl.deleteTexture(t.tex); gl.deleteFramebuffer(t.fbo); };
      if (this.targets.scene) kill(this.targets.scene);
      if (this.targets.mips) this.targets.mips.forEach(m => { kill(m.a); kill(m.b); });
      this.targets = {};
    }

    resize(cssW, cssH) {
      const gl = this.gl;
      this._cssW = cssW; this._cssH = cssH;
      const Q = QUALITY[this.quality];
      const dpr = Math.min(global.devicePixelRatio || 1, Q.dpr);
      /* The exterior shot has almost no volumetric cost — the plasma march
         only runs through the cutaway — so it can carry a far higher
         internal resolution than the interior at the same preset.  Without
         this the machine's panel seams and coil detail get washed out by a
         scale chosen for the expensive interior view.                    */
      const eff = Math.min(1.0, this.renderScale * (this._extBoost || 1));
      const W = Math.max(2, Math.round(cssW * dpr * eff));
      const H = Math.max(2, Math.round(cssH * dpr * eff));
      if (W === this.w && H === this.h) return;
      this.w = W; this.h = H;
      this.canvas.width = Math.round(cssW * dpr);
      this.canvas.height = Math.round(cssH * dpr);
      this.canvas.style.width = cssW + 'px';
      this.canvas.style.height = cssH + 'px';

      this.disposeTargets();
      const F = this.hdrFmt, T = this.hdrType;
      this.targets.scene = this.makeTarget(W, H, F, T);
      this.targets.mips = [];
      let mw = W, mh = H;
      for (let i = 0; i < 3; i++) {
        mw = Math.max(2, mw >> 1); mh = Math.max(2, mh >> 1);
        this.targets.mips.push({
          a: this.makeTarget(mw, mh, F, T),
          b: this.makeTarget(mw, mh, F, T), w: mw, h: mh
        });
      }
      gl.viewport(0, 0, W, H);
    }

    setQuality(name) {
      if (!QUALITY[name]) return;
      this.quality = name;
      this.renderScale = QUALITY[name].scale;
      this.w = 0;
    }

    /* ------------------------------------------------------------------ */
    drawQuad() {
      const gl = this.gl;
      gl.bindVertexArray(this.vao);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindVertexArray(null);
    }

    bindTarget(t) {
      const gl = this.gl;
      if (t) { gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbo); gl.viewport(0, 0, t.w, t.h); }
      else { gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, this.canvas.width, this.canvas.height); }
    }

    /* ------------------------------------------------------------------ */
    /*  Shot state machine                                                  */
    /* ------------------------------------------------------------------ */
    startFlight() {
      if (this.shot.mode !== 'landing' && this.shot.mode !== 'anatomy') return;
      this.shot.mode = 'flight';
      this.shot.flight = 0;
      this._flightYaw0 = this.cam.yaw + this.userYaw;
      this.userYaw = 0; this.userPitch = 0; this.userZoom = 0;
    }

    toLanding() {
      this.shot.mode = 'landing';
      this.shot.flight = 0;
      this.userYaw = 0; this.userPitch = 0; this.userZoom = 0;
      this.cam.autoOrbit = true;
    }

    toAnatomy() {
      this.shot.mode = 'anatomy';
      this.shot.flight = 0;
      this.userYaw = 0; this.userPitch = 0; this.userZoom = 0;
    }

    /* Project a world point to CSS pixels through the same equidistant
       fisheye the shader uses.  Returns null when the point falls outside
       the frame or behind the lens.                                      */
    project(P, cssW, cssH) {
      const c = this._camPos, m = this._camMat;
      if (!c || !m) return null;
      let d = [P[0] - c[0], P[1] - c[1], P[2] - c[2]];
      const L = Math.hypot(d[0], d[1], d[2]);
      if (L < 1e-6) return null;
      d = [d[0] / L, d[1] / L, d[2] / L];
      /* camMat is orthonormal, so its inverse is its transpose */
      const vx = m[0] * d[0] + m[1] * d[1] + m[2] * d[2];
      const vy = m[3] * d[0] + m[4] * d[1] + m[5] * d[2];
      const vz = m[6] * d[0] + m[7] * d[1] + m[8] * d[2];
      const ang = Math.acos(Math.max(-1, Math.min(1, -vz)));
      const rr = Math.hypot(vx, vy);
      if (rr < 1e-6) return { x: cssW / 2, y: cssH / 2, dist: L, ang };
      const k = (ang / this.cam.fov) / rr;
      const ux = vx * k, uy = vy * k;
      return {
        x: ux * cssH + cssW / 2,
        y: cssH / 2 - uy * cssH,
        dist: L,
        ang: ang / this.cam.fov      /* 1.0 = frame edge vertically */
      };
    }

    updateShot(dt, t) {
      const s = this.shot;
      if (s.mode === 'flight') {
        s.flight = Math.min(1, s.flight + dt / s.duration);
        if (s.flight >= 1) { s.mode = 'interior'; this.cam.autoOrbit = true; }
      }
      const c = this.cam;

      if (s.mode === 'landing' || s.mode === 'anatomy') {
        const k = samplePath(0, this._pathOut);
        c.height = k.height; c.pitch = k.pitch; c.fov = k.fov;
        s.cut = k.cut; s.ext = k.ext;
        if (s.mode === 'anatomy') {
          /* pulled in a little and turning slowly, so the callout leaders
             stay readable rather than sweeping across the frame */
          c.radius = k.radius * 0.92 + this.userZoom;
          c.yaw = s.cutCenter + Math.sin(t * 0.045) * 0.20;
        } else {
          c.radius = k.radius + this.userZoom;
          c.yaw = s.cutCenter + Math.sin(t * 0.09) * 0.34;
        }
      } else if (s.mode === 'flight') {
        const e = easeFlight(s.flight);
        const k = samplePath(e, this._pathOut);
        c.radius = k.radius; c.height = k.height; c.pitch = k.pitch; c.fov = k.fov;
        s.cut = k.cut; s.ext = k.ext;
        /* settle the bearing onto the wedge centre over the first third */
        const align = Math.min(1, s.flight / 0.32);
        const a = align * align * (3 - 2 * align);
        c.yaw = this._flightYaw0 * (1 - a) + s.cutCenter * a +
                Math.sin(e * Math.PI) * 0.16;     /* slight arc, keeps it alive */
      } else {
        s.cut = 0; s.ext = 0;
        c.fov = Math.max(0.72, Math.min(1.45, 1.30 + this.userZoom * 0.22));
        c.radius = 3.45 + this.userZoom;
        c.height = 0.18; c.pitch = 0.02;
        if (c.autoOrbit) c.yaw += dt * c.orbitSpeed;
      }
    }

    /* Build the camera basis for this frame. */
    cameraBasis(t, dtSec, phys) {
      const c = this.cam;
      const shakeTarget = phys
        ? phys.elmFlash * 0.35 + phys.sawFlash * 0.15 + (phys.disrupted ? 0.9 : 0)
        : 0;
      /* the machine is only felt from inside */
      const inside = this.shot.mode === 'interior' ? 1 : 0;
      c.shake += (shakeTarget * inside - c.shake) * Math.min(1, dtSec * 8);
      const sx = Math.sin(t * 47.3) * c.shake * 0.010;
      const sy = Math.sin(t * 39.1 + 1.7) * c.shake * 0.010;

      const yaw = c.yaw + this.userYaw + sx;
      const pitch = c.pitch + this.userPitch + sy;
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      const pos = [
        Math.sin(yaw) * c.radius * cp,
        c.height + sp * c.radius,
        Math.cos(yaw) * c.radius * cp
      ];
      if (this.shot.mode === 'interior') {
        const brk = 1 + Math.sin(t * 0.11) * 0.012;
        pos[0] *= brk; pos[2] *= brk;
      }

      const tgt = [0, c.targetY, 0];
      let f = [tgt[0] - pos[0], tgt[1] - pos[1], tgt[2] - pos[2]];
      const fl = Math.hypot(f[0], f[1], f[2]); f = f.map(v => v / fl);
      /* right = normalize(forward x worldUp), with worldUp = (0,1,0) */
      let r = [-f[2], 0, f[0]];
      const rl = Math.hypot(r[0], r[1], r[2]); r = r.map(v => v / rl);
      const u = [
        r[1] * f[2] - r[2] * f[1],
        r[2] * f[0] - r[0] * f[2],
        r[0] * f[1] - r[1] * f[0]
      ];
      this._camMat = new Float32Array([r[0], r[1], r[2], u[0], u[1], u[2], -f[0], -f[1], -f[2]]);
      this._camPos = pos;
    }

    /* ------------------------------------------------------------------ */
    render(phys, t, dtSec, view) {
      const gl = this.gl;
      const Q = QUALITY[this.quality];

      /* Two resolution regimes, switched once as the camera crosses into the
         machine.  Kept binary on purpose: every change reallocates the whole
         framebuffer chain, and one hitch in the middle of a fast move is
         invisible where four spread along it would not be.               */
      /* hysteresis, so an interpolated ext that wobbles across the midpoint
         cannot reallocate the framebuffers twice on the way in */
      const wasExt = this._extBoost === EXT_BOOST;
      const wantBoost = (this.shot.ext > (wasExt ? 0.45 : 0.55)) ? EXT_BOOST : 1;
      if (wantBoost !== this._extBoost) {
        this._extBoost = wantBoost;
        this.w = 0;
        if (this._cssW) this.resize(this._cssW, this._cssH);
      }

      this.cameraBasis(t, dtSec, phys);

      const P = phys;
      const Tn = Math.min(1, P.Te / 22);
      const Dn = Math.min(1, P.ne / 1.3);
      const emis = 0.10 + 1.35 * Math.pow(Dn, 0.75) *
                   (0.35 + 0.65 * Math.min(1, P.Te / 6));
      const instab = Math.min(1.6,
        0.30 + 0.55 * (1 - Math.min(1, P.tauE / 3)) +
        0.5 * P.betaN / 4 + (P.disrupted ? 1.2 : 0));

      const cutView = view.crossSection ? 1 : 0;
      const cut = this.shot.mode === 'interior'
        ? cutView : Math.max(this.shot.cut, cutView);

      const S = this.pScene;
      gl.useProgram(S.prog);
      this.bindTarget(this.targets.scene);
      gl.uniform2f(S.u.uRes, this.targets.scene.w, this.targets.scene.h);
      gl.uniform1f(S.u.uTime, t);
      gl.uniform3fv(S.u.uCamPos, this._camPos);
      gl.uniformMatrix3fv(S.u.uCamMat, false, this._camMat);
      gl.uniform1f(S.u.uFovHalf, this.cam.fov);
      gl.uniform1f(S.u.uRot, P.rotPhase);
      gl.uniform1f(S.u.uEmis, emis);
      gl.uniform1f(S.u.uTemp, Tn);
      gl.uniform1f(S.u.uDens, Dn);
      gl.uniform1f(S.u.uInstab, instab);
      gl.uniform1f(S.u.uElm, P.elmFlash);
      gl.uniform1f(S.u.uSaw, P.sawFlash);
      gl.uniform1f(S.u.uIgnite, P.ignition);
      gl.uniform1f(S.u.uDisrupt, P.disrupted ? 1 : 0);
      gl.uniform1f(S.u.uNbi, Math.min(1, P.Pnbi / 33));
      gl.uniform1f(S.u.uRf, Math.min(1, (P.Picr + P.Pecr) / 40));
      gl.uniform1f(S.u.uElong, P.M.kappa * 0.95);
      gl.uniform1f(S.u.uHmode, P.hMode ? 1 : 0);
      /* The exterior shot spends almost nothing on the volumetric march, so
         it gets the settings that actually shape how the machine reads —
         occlusion contact shadows, a smooth ring light and a full-detail
         SDF — regardless of the preset chosen for the interior. */
      const ext = this.shot.ext > 0.5;
      const steps  = Q.steps;
      const scene  = ext ? Math.max(Q.scene, 120) : Q.scene;
      const lights = ext ? Math.max(Q.lights, 10) : Q.lights;
      const ao     = ext ? 5 : Q.ao;
      const detail = ext ? 1 : (Q.steps >= 56 ? 1 : 0);

      gl.uniform1i(S.u.uSteps, steps);
      gl.uniform1i(S.u.uSceneSteps, scene);
      gl.uniform1i(S.u.uLights, lights);
      gl.uniform1i(S.u.uAOSteps, ao);
      gl.uniform1f(S.u.uDetail,
        this.detailOverride != null ? this.detailOverride : detail);
      gl.uniform1f(S.u.uHighlight, view.highlight || 0);
      gl.uniform1f(S.u.uFieldLines, view.fieldLines ? 1 : 0);
      gl.uniform1f(S.u.uParticles, view.particles ? 1 : 0);
      gl.uniform1f(S.u.uCut, cut);
      /* shot.cutCenter is a camera yaw; the shader measures phi from +x */
      gl.uniform1f(S.u.uCutCenter, Math.PI * 0.5 - this.shot.cutCenter);
      gl.uniform1f(S.u.uExtLight, this.shot.ext);
      gl.uniform1f(S.u.uShowPlasma, view.plasma === false ? 0 : 1);
      gl.uniform1f(S.u.uShowCoils, view.coils === false ? 0 : 1);
      gl.uniform1f(S.u.uHeatMap, view.heatMap ? 1 : 0);
      gl.uniform1f(S.u.uQdiv, Math.min(1, P.qDiv / 16));
      this.drawQuad();

      /* ---------- bloom pyramid ----------------------------------------- */
      const mips = this.targets.mips;
      const B = this.pBright;
      gl.useProgram(B.prog);
      this.bindTarget(mips[0].a);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.targets.scene.tex);
      gl.uniform1i(B.u.uTex, 0);
      gl.uniform2f(B.u.uTexel, 1 / this.targets.scene.w, 1 / this.targets.scene.h);
      gl.uniform1f(B.u.uThreshold, 1.00);
      gl.uniform1f(B.u.uSoft, 0.45);
      this.drawQuad();

      const BL = this.pBlur;
      gl.useProgram(BL.prog);
      const blur = (src, dst, dx, dy) => {
        this.bindTarget(dst);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, src.tex);
        gl.uniform1i(BL.u.uTex, 0);
        gl.uniform2f(BL.u.uDir, dx, dy);
        this.drawQuad();
      };
      const nMips = Q.bloom;
      for (let i = 0; i < nMips; i++) {
        const m = mips[i];
        if (i > 0) {
          blur(mips[i - 1].a, m.b, 1 / mips[i - 1].w, 0);
          blur(m.b, m.a, 0, 1 / m.h);
        }
        blur(m.a, m.b, 1.4 / m.w, 0);
        blur(m.b, m.a, 0, 1.4 / m.h);
      }
      for (let i = nMips; i < 3; i++) {
        this.bindTarget(mips[i].a);
        gl.clearColor(0, 0, 0, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);
      }

      /* ---------- composite --------------------------------------------- */
      const C = this.pComposite;
      gl.useProgram(C.prog);
      this.bindTarget(null);
      const bind = (unit, tex, loc) => {
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.uniform1i(loc, unit);
      };
      bind(0, this.targets.scene.tex, C.u.uScene);
      bind(1, mips[0].a.tex, C.u.uBloom0);
      bind(2, mips[1].a.tex, C.u.uBloom1);
      bind(3, mips[2].a.tex, C.u.uBloom2);
      gl.uniform2f(C.u.uRes, this.canvas.width, this.canvas.height);
      gl.uniform1f(C.u.uTime, t);
      const k = this.shot.ext;                       /* 1 outside, 0 inside */
      const po = this.post;
      gl.uniform1f(C.u.uBloomAmt, po.bloom * (1 + 0.5 * P.elmFlash));
      gl.uniform1f(C.u.uExposure, po.exposureIn + (po.exposureOut - po.exposureIn) * k);
      /* heat shimmer belongs to the interior shot */
      gl.uniform1f(C.u.uShimmer, po.shimmer * (1 - k));
      gl.uniform1f(C.u.uElm, P.elmFlash);
      gl.uniform1f(C.u.uDisrupt, P.disrupted ? 1 : 0);
      gl.uniform1f(C.u.uGrain, po.grain);
      gl.uniform1f(C.u.uVignette, po.vignetteIn + (po.vignetteOut - po.vignetteIn) * k);
      gl.uniform1f(C.u.uFade, this.shot.fade);
      this.drawQuad();
    }

    /* One-off GPU probe: time a few worst-case (interior) frames and pick a
       starting preset.  readPixels forces a pipeline flush, so unlike a bare
       performance.now() around the draw calls this actually waits for the
       GPU.  Costs ~150 ms once, at boot, and saves guessing.             */
    probe(phys, cssW, cssH) {
      const gl = this.gl;
      const px = new Uint8Array(4);
      const savedMode = this.shot.mode, savedCut = this.shot.cut,
            savedExt = this.shot.ext, savedAuto = this.autoScale;
      this.autoScale = false;
      this.setQuality('medium');
      this.resize(cssW, cssH);
      this.shot.mode = 'interior';
      this.updateShot(0, 0);
      const view = { plasma: true, coils: true };
      const sync = () => {
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
      };
      /* Warm up properly: the driver compiles the shader and allocates the
         framebuffer chain lazily on first draw, so a single warm frame
         reads far too fast and the probe over-rates the GPU. */
      for (let i = 0; i < 3; i++) this.render(phys, i * 0.016, 0.016, view);
      sync();
      let ms = 0;
      for (let round = 0; round < 2; round++) {
        const t0 = performance.now();
        for (let i = 0; i < 3; i++) this.render(phys, i * 0.016, 0.016, view);
        sync();
        ms = (performance.now() - t0) / 3;   /* keep the settled round */
      }

      /* thresholds are on a medium-preset frame and aim for ~60 fps headroom */
      const q = ms > 32 ? 'low' : ms > 17 ? 'medium' : ms > 9 ? 'high' : 'ultra';
      this.shot.mode = savedMode; this.shot.cut = savedCut; this.shot.ext = savedExt;
      this.autoScale = savedAuto;
      this.setQuality(q);
      this.resize(cssW, cssH);
      return { ms, quality: q };
    }

    /* Adaptive resolution, then adaptive quality.  frameMs must be the real
       frame-to-frame delta — GPU work is asynchronous, so timing the JS
       render call measures almost nothing.                               */
    adapt(frameMs, cssW, cssH) {
      this.frameMs += (frameMs - this.frameMs) * 0.10;
      if (!this.autoScale) return;
      const base = QUALITY[this.quality].scale;
      const minScale = base * 0.5;
      let s = this.renderScale;

      if (this.frameMs > 30) { this._slow++; this._fast = 0; s -= 0.035; }
      else if (this.frameMs < 19) { this._fast++; this._slow = 0; if (s < base) s += 0.012; }
      else { this._slow = 0; this._fast = 0; }

      /* already at the resolution floor and still dropping frames: step the
         whole preset down rather than degrading resolution further */
      if (this._slow > 45 && s <= minScale + 1e-3) {
        const order = ['ultra', 'high', 'medium', 'low'];
        const i = order.indexOf(this.quality);
        if (i >= 0 && i < order.length - 1) {
          this._slow = 0;
          this.setQuality(order[i + 1]);
          if (this.onQualityChange) this.onQualityChange(this.quality);
          this.resize(cssW, cssH);
          return;
        }
      }

      s = Math.max(minScale, Math.min(base, s));
      if (Math.abs(s - this.renderScale) > 0.004) {
        this.renderScale = s;
        this.w = 0;
        this.resize(cssW, cssH);
      }
    }
  }

  Renderer.QUALITY = QUALITY;
  Renderer.PATH = PATH;
  global.Renderer = Renderer;
})(window);
