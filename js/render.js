/* ==========================================================================
   render.js — WebGL2 deferred-lite renderer for the tokamak interior
   --------------------------------------------------------------------------
   Full-screen raymarch into HDR (RGBA16F when available), a three-level
   bloom pyramid, then an ACES composite with heat shimmer and grain.
   Resolution scale adapts to keep the frame budget.
   ========================================================================== */
(function (global) {
  'use strict';

  const QUALITY = {
    low:    { scale: 0.55, steps: 42,  lights: 6,  bloom: 2 },
    medium: { scale: 0.75, steps: 72,  lights: 10, bloom: 3 },
    high:   { scale: 1.00, steps: 112, lights: 14, bloom: 3 },
    ultra:  { scale: 1.00, steps: 176, lights: 20, bloom: 3 }
  };

  class Renderer {
    constructor(canvas) {
      this.canvas = canvas;
      const opts = {
        alpha: false, antialias: false, depth: false, stencil: false,
        premultipliedAlpha: false, preserveDrawingBuffer: true,
        powerPreference: 'high-performance'
      };
      const gl = canvas.getContext('webgl2', opts);
      if (!gl) throw new Error('WebGL2 is required for this simulation.');
      this.gl = gl;

      this.floatOK = !!gl.getExtension('EXT_color_buffer_float');
      gl.getExtension('OES_texture_float_linear');
      this.hdrFmt = this.floatOK ? gl.RGBA16F : gl.RGBA8;
      this.hdrType = this.floatOK ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE;

      this.quality = 'high';
      this.renderScale = QUALITY.high.scale;
      this.autoScale = true;
      this.frameMs = 16;

      this.buildQuad();
      this.buildPrograms();
      this.targets = {};
      this.w = 0; this.h = 0;

      /* --- camera ------------------------------------------------------- */
      this.cam = {
        radius: 3.45, yaw: 0.0, pitch: 0.02, height: 0.18,
        fov: 1.30,                       // half-angle of the fisheye [rad]
        targetY: 0.02,
        autoOrbit: true, orbitSpeed: 0.020, shake: 0
      };
      this.userYaw = 0; this.userPitch = 0;

      /* --- post parameters ---------------------------------------------- */
      this.post = {
        exposure: 0.92, bloom: 0.28, shimmer: 1.0,
        grain: 0.035, vignette: 0.62
      };
    }

    /* ------------------------------------------------------------------ */
    buildQuad() {
      const gl = this.gl;
      this.vao = gl.createVertexArray();
      gl.bindVertexArray(this.vao);
      const buf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER,
        new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
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
        const numbered = src.split('\n')
          .map((l, i) => String(i + 1).padStart(4) + ' | ' + l).join('\n');
        console.error(log + '\n' + numbered);
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
      /* cache uniform locations */
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
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
        gl.TEXTURE_2D, tex, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return { tex, fbo, w, h };
    }

    disposeTargets() {
      const gl = this.gl;
      for (const k in this.targets) {
        const t = this.targets[k];
        if (!t) continue;
        const arr = Array.isArray(t) ? t : [t];
        arr.forEach(x => { gl.deleteTexture(x.tex); gl.deleteFramebuffer(x.fbo); });
      }
      this.targets = {};
    }

    resize(cssW, cssH) {
      const gl = this.gl;
      const dpr = Math.min(global.devicePixelRatio || 1, 2);
      const W = Math.max(2, Math.round(cssW * dpr * this.renderScale));
      const H = Math.max(2, Math.round(cssH * dpr * this.renderScale));
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
          b: this.makeTarget(mw, mh, F, T),
          w: mw, h: mh
        });
      }
      /* flatten for disposal bookkeeping */
      this.targets._flat = [];
      this.targets.mips.forEach(m => this.targets._flat.push(m.a, m.b));
      gl.viewport(0, 0, W, H);
    }

    setQuality(name) {
      if (!QUALITY[name]) return;
      this.quality = name;
      this.renderScale = QUALITY[name].scale;
      this.w = 0;   // force a resize on the next frame
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
      if (t) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbo);
        gl.viewport(0, 0, t.w, t.h);
      } else {
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      }
    }

    /* ------------------------------------------------------------------ */
    /*  Build the camera basis. The view sits just inside the outer wall,
        looking radially inward at the central column — matching a wide
        angle port-view camera.                                            */
    /* ------------------------------------------------------------------ */
    cameraBasis(t, dtSec, phys) {
      const c = this.cam;
      if (c.autoOrbit) c.yaw += dtSec * c.orbitSpeed;

      /* vibration from ELMs, sawteeth and disruptions */
      const shakeTarget = phys
        ? phys.elmFlash * 0.35 + phys.sawFlash * 0.15 +
          (phys.disrupted ? 1.0 : 0) * 0.9
        : 0;
      c.shake += (shakeTarget - c.shake) * Math.min(1, dtSec * 8);
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
      /* slow breathing of the camera radius keeps the shot alive */
      const brk = 1 + Math.sin(t * 0.11) * 0.012;
      pos[0] *= brk; pos[2] *= brk;

      const tgt = [0, c.targetY, 0];
      let f = [tgt[0] - pos[0], tgt[1] - pos[1], tgt[2] - pos[2]];
      let fl = Math.hypot(f[0], f[1], f[2]); f = f.map(v => v / fl);
      const upW = [0, 1, 0];
      let r = [
        f[1] * upW[2] - f[2] * upW[1],
        f[2] * upW[0] - f[0] * upW[2],
        f[0] * upW[1] - f[1] * upW[0]
      ];
      const rl = Math.hypot(r[0], r[1], r[2]); r = r.map(v => v / rl);
      const u = [
        r[1] * f[2] - r[2] * f[1],
        r[2] * f[0] - r[0] * f[2],
        r[0] * f[1] - r[1] * f[0]
      ];
      /* column-major mat3( right, up, -forward ) */
      this._camMat = new Float32Array([
        r[0], r[1], r[2],
        u[0], u[1], u[2],
        -f[0], -f[1], -f[2]
      ]);
      this._camPos = pos;
    }

    /* ------------------------------------------------------------------ */
    render(phys, t, dtSec, view) {
      const gl = this.gl;
      const Q = QUALITY[this.quality];
      this.cameraBasis(t, dtSec, phys);

      /* ---------- map physics state to shader parameters ---------------- */
      const P = phys;
      const Tn = Math.min(1, P.Te / 22);
      const Dn = Math.min(1, P.ne / 1.3);
      /* visible emissivity: rises with density, saturates with temperature
         (hot core is optically thin, so brightness is edge dominated)      */
      const emis = 0.10 + 1.35 * Math.pow(Dn, 0.75) *
                   (0.35 + 0.65 * Math.min(1, P.Te / 6));
      const instab = Math.min(1.6,
        0.30 + 0.55 * (1 - Math.min(1, P.tauE / 3)) +
        0.5 * P.betaN / 4 + (P.disrupted ? 1.2 : 0));

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
      gl.uniform1i(S.u.uSteps, Q.steps);
      gl.uniform1i(S.u.uLights, Q.lights);
      gl.uniform1f(S.u.uFieldLines, view.fieldLines ? 1 : 0);
      gl.uniform1f(S.u.uParticles, view.particles ? 1 : 0);
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
      gl.uniform1f(B.u.uThreshold, 1.55);
      gl.uniform1f(B.u.uSoft, 0.55);
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
          /* downsample the previous level into this one */
          blur(mips[i - 1].a, m.b, 1 / mips[i - 1].w, 0);
          blur(m.b, m.a, 0, 1 / m.h);
        }
        blur(m.a, m.b, 1.4 / m.w, 0);
        blur(m.b, m.a, 0, 1.4 / m.h);
      }
      for (let i = nMips; i < 3; i++) {   // clear unused levels
        this.bindTarget(mips[i].a);
        gl.clearColor(0, 0, 0, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);
      }

      /* ---------- composite to the default framebuffer ------------------- */
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
      gl.uniform1f(C.u.uBloomAmt, this.post.bloom * (1 + 0.5 * P.elmFlash));
      gl.uniform1f(C.u.uExposure, this.post.exposure);
      gl.uniform1f(C.u.uShimmer, this.post.shimmer);
      gl.uniform1f(C.u.uElm, P.elmFlash);
      gl.uniform1f(C.u.uDisrupt, P.disrupted ? 1 : 0);
      gl.uniform1f(C.u.uGrain, this.post.grain);
      gl.uniform1f(C.u.uVignette, this.post.vignette);
      this.drawQuad();
    }

    /* Adaptive resolution: keep the GPU inside a ~16 ms budget. */
    adapt(frameMs, cssW, cssH) {
      this.frameMs += (frameMs - this.frameMs) * 0.08;
      if (!this.autoScale) return;
      const base = QUALITY[this.quality].scale;
      let s = this.renderScale;
      if (this.frameMs > 26 && s > base * 0.55) s -= 0.03;
      else if (this.frameMs < 15 && s < base) s += 0.01;
      if (Math.abs(s - this.renderScale) > 0.004) {
        this.renderScale = Math.max(0.4, Math.min(base, s));
        this.w = 0;
        this.resize(cssW, cssH);
      }
    }
  }

  Renderer.QUALITY = QUALITY;
  global.Renderer = Renderer;
})(window);
