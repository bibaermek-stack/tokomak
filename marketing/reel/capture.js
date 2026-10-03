/* Capture real footage of the SINTEZ site, frame by frame, on a virtual clock.
 *
 * The renderer adapts to slow frames (it drops resolution, then the whole
 * quality preset), which is why headless captures used to come out dark and
 * soft. Here time only advances when we say so: requestAnimationFrame and
 * performance.now are replaced, autoScale is switched off and the preset is
 * pinned, so every captured frame is the full-quality frame the site would
 * draw on a fast GPU, regardless of how long the software renderer takes.
 *
 * usage: node capture.js <url> <outdir> [--only landing,flight,interior,hud]
 */
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright-core');

const URL = process.argv[2] || 'http://localhost:8099/index.html';
const OUT = process.argv[3] || './cap';
const ONLY = (process.argv.find(a => a.startsWith('--only=')) || '').slice(7).split(',').filter(Boolean);
const EXE = process.env.CHROME || '/opt/pw-browsers/chromium';
const FPS = 30;
const want = k => !ONLY.length || ONLY.includes(k);

const CLOCK = `(() => {
  let vt = 0; const q = [];
  window.requestAnimationFrame = cb => { q.push(cb); return q.length; };
  window.cancelAnimationFrame = () => {};
  performance.now = () => vt;
  const base = Date.now(); Date.now = () => base + vt;
  window.__step = ms => { vt += ms; const cbs = q.splice(0); for (const cb of cbs) cb(vt); };
})();`;

/* Everything that is page chrome rather than footage. */
const CLEAN = `#landing,#topnav,#article,#scrim,#flight-ui,#anatomy-view,#an-markers,#hud{opacity:0!important;pointer-events:none!important}`;

async function open(browser, w, h, scale) {
  const QUALITY = process.env.QUALITY || 'high';
  const HDR8 = process.env.HDR8 !== '0';
  const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: scale });
  await page.addInitScript(CLOCK);
  await page.goto(URL, { waitUntil: 'networkidle' });
  /* boot() runs on real setTimeouts, then warms the physics and probes the
     GPU; only after #boot is removed is the renderer's preset final */
  await page.waitForFunction(() => window.SIM && window.SIM.renderer && !document.getElementById('boot'),
                             null, { timeout: 600000, polling: 250 });
  await page.evaluate(([QUALITY, HDR8]) => {
    const r = window.SIM.renderer;
    r.autoScale = false;
    r.setQuality(QUALITY);
    r.renderScale = 1.0; r.w = 0;
    /* SwiftShader leaves a few NaN texels in the half-float scene target;
       the bloom pyramid then smears each one into a mip-sized black block.
       8-bit targets convert NaN to 0 on write, so nothing spreads. */
    if (HDR8) { const gl = r.gl; r.hdrFmt = gl.RGBA8; r.hdrType = gl.UNSIGNED_BYTE; }
    r.resize(window.innerWidth, window.innerHeight);
  }, [QUALITY, HDR8]);
  /* the first frames after a resize compile shaders and allocate targets;
     they come out black, so draw and throw a few away */
  for (let i = 0; i < 6; i++) { await page.evaluate(() => window.__step(1000 / 30)); await page.screenshot({ type: 'jpeg', quality: 10, timeout: 180000 }); }
  return page;
}

/* Advance n frames of site time but only draw the last one: the camera,
   physics and HUD all move on every step, the GPU does one frame of work.
   Queuing dozens of full software-rendered frames otherwise stalls the
   screenshot behind them. */
const step = (page, n = 1) => page.evaluate(k => {
  const r = window.SIM.renderer;
  const real = r.__realRender || (r.__realRender = r.render);
  r.render = function () {};
  for (let i = 0; i < k - 1; i++) window.__step(1000 / 30);
  r.render = real;
  window.__step(1000 / 30);
}, n);

async function shoot(page, dir, name, frames, opts = {}) {
  fs.mkdirSync(dir, { recursive: true });
  const every = opts.every || 1;           /* sim frames advanced per captured frame */
  /* START lets several processes share one clip: SwiftShader runs one
     core per process, so segments captured side by side finish together */
  const start = +(process.env.START || 0);
  if (start) await step(page, start * every);
  const t0 = Date.now();
  for (let i = 0; i < frames; i++) {
    await step(page, every);
    await page.screenshot({ path: path.join(dir, `${name}_${String(i + start).padStart(4, '0')}.jpg`),
                            type: 'jpeg', quality: 92, timeout: 180000 });
  }
  console.log(`${name}: ${frames} кадр, ${((Date.now() - t0) / frames).toFixed(0)} мс/кадр`);
}

(async () => {
  const browser = await chromium.launch({ executablePath: EXE,
    args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--disable-gpu-watchdog', '--disable-gpu-process-crash-limit'] });

  /* ---- 3D footage, portrait 1080x1920 ---------------------------------- */
  if (want('landing') || want('flight') || want('interior')) {
    const page = await open(browser, 1080, 1920, 1);
    await page.addStyleTag({ content: CLEAN + `#gl{transform:translateY(-10%)}body{background:#00000e}` });
    /* portrait has a narrow horizontal field of view: pull the establishing
       shot back so the whole machine fits, the way the landscape site frames it */
    /* The shader's projection is equidistant and normalised by frame height,
       so a portrait frame keeps the vertical angle and loses ~2.8x of the
       horizontal one. Pulling the camera back is not an option -- the march
       stops at 26 units -- so widen the angle instead, easing it back to the
       site's own value as the flight enters the plasma. */
    await page.evaluate(([kOut, kIn, z]) => {
      const r = window.SIM.renderer;
      r.userZoom = z;
      const orig = r.updateShot.bind(r);
      r.updateShot = function (dt, t) {
        orig(dt, t);
        const s = this.shot; let k = kOut;
        if (s.mode === 'flight') { const e = s.flight; k = kOut + (kIn - kOut) * e * e * (3 - 2 * e); }
        else if (s.mode === 'interior') k = kIn;
        this.cam.fov *= k;
      };
    }, [+(process.env.FOVK_OUT || 2.0), +(process.env.FOVK_IN || 1.25), +(process.env.ZOOM || 0)]);
    await step(page, 30);                                   /* settle */

    if (want('landing')) await shoot(page, OUT + '/landing', 'landing', +(process.env.N_LANDING || 105));

    if (want('flight') || want('interior')) {
      await page.evaluate(() => { window.SIM.renderer.userZoom = 0; window.SIM.startFlight(); });
      await page.addStyleTag({ content: CLEAN });
      /* the flight runs ~7 s; take every 2nd frame so it plays in ~3.5 s */
      if (want('flight')) await shoot(page, OUT + '/flight', 'flight', +(process.env.N_FLIGHT || 112), { every: 2 });
      else await step(page, 240);
      await page.addStyleTag({ content: CLEAN });
      if (want('interior')) await shoot(page, OUT + '/interior', 'interior', +(process.env.N_INTERIOR || 105));
    }
    await page.close();
  }

  /* ---- control room, landscape, run to a steady burn -------------------- */
  if (want('hud')) {
    const page = await open(browser, 1920, 1200, +(process.env.HUD_SCALE || 2));
    await page.evaluate(() => window.SIM.startFlight());
    await step(page, 260);                                  /* through the flight */
    /* fast-forward the discharge without drawing 3D every step, but keep
       feeding the HUD so its history charts fill exactly as they would live */
    await page.evaluate(() => {
      const { phys, hud } = window.SIM;
      for (let i = 0; i < 1500; i++) { phys.step(0.05); hud.hist.push(phys, 0.05); }
    });
    await step(page, 45);
    fs.mkdirSync(OUT + '/hud', { recursive: true });
    await page.screenshot({ path: OUT + '/hud/hud_full.png', timeout: 180000 });
    for (const id of ['hud-left', 'hud-right', 'hud-bottom', 'hud-top']) {
      const el = await page.$('#' + id);
      if (el) await el.screenshot({ path: `${OUT}/hud/${id}.png`, timeout: 180000 });
    }
    /* each panel on its own, for the portrait "cards" beats */
    const ids = await page.evaluate(() => window.SIM.hud.panels.map(p => p.spec.id));
    for (const id of ids) {
      const h = await page.evaluateHandle(i => window.SIM.hud.panels.find(p => p.spec.id === i).el, id);
      await h.asElement().screenshot({ path: `${OUT}/hud/panel_${id}.png`, timeout: 180000 });
    }
    const s = await page.evaluate(() => { const p = window.SIM.phys;
      return { t: p.t.toFixed(1), Pfus: p.Pfus.toFixed(1), Q: p.Q.toFixed(2), H98: p.H98.toFixed(2),
               Te: p.Te.toFixed(2), Ti: p.Ti.toFixed(2), hist: window.SIM.hud.hist.count }; });
    console.log('hud күйі', JSON.stringify(s));
    await page.close();
  }
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
