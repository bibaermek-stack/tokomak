/* ==========================================================================
   main.js — boot sequence, simulation loop and operator controls
   ========================================================================== */
(function () {
  'use strict';

  const $ = id => document.getElementById(id);

  /* ------------------------------------------------------------ boot ---- */
  const BOOT_LINES = [
    'CRYOPLANT ...................... 4.5 K NOMINAL',
    'TF COIL SYSTEM ................. 18/18 SUPERCONDUCTING',
    'CENTRAL SOLENOID ............... FLUX SWING ARMED',
    'TORUS VACUUM ................... 1.0e-06 Pa',
    'D–T FUEL INJECTION ............. INTERLOCKS CLEAR',
    'DIAGNOSTIC SUITE ............... 41 CHANNELS ONLINE',
    'PLASMA CONTROL SYSTEM .......... REAL-TIME LOOP 1 kHz',
    'BREAKDOWN SEQUENCE ............. READY'
  ];

  function boot(done) {
    const log = $('boot-log'), bar = document.querySelector('.boot-bar i');
    let i = 0;
    const tick = () => {
      if (i < BOOT_LINES.length) {
        log.innerHTML += BOOT_LINES[i].replace(/(\S+)$/, '<b>$1</b>') + '<br>';
        if (log.children.length > 8) log.removeChild(log.firstChild);
        bar.style.width = ((i + 1) / BOOT_LINES.length * 100) + '%';
        i++;
        setTimeout(tick, 130 + Math.random() * 90);
      } else {
        setTimeout(() => {
          $('boot').classList.add('gone');
          setTimeout(() => $('boot').remove(), 800);
          done();
        }, 320);
      }
    };
    tick();
  }

  /* ------------------------------------------------------------ setup --- */
  let renderer, phys, hud;
  const canvas = $('gl');

  try {
    renderer = new Renderer(canvas);
  } catch (e) {
    const f = $('fatal');
    f.hidden = false;
    f.textContent = 'UNABLE TO INITIALISE THE REACTOR VIEW — ' + e.message +
      '  ::  This simulation requires a WebGL2-capable browser.';
    $('boot').remove();
    return;
  }

  phys = new Tokamak();
  hud = new HUD($('hud'));

  const view = { fieldLines: false, particles: false };
  const sim = { rate: 1.0, paused: false, running: false };

  /* ------------------------------------------------------------ layout -- */
  function layout() {
    const c = $('hud-center').getBoundingClientRect();
    renderer.resize(window.innerWidth, window.innerHeight);
    hud.resize();
    void c;
  }
  window.addEventListener('resize', layout);

  /* ------------------------------------------------------ camera input -- */
  let dragging = false, lastX = 0, lastY = 0;
  canvas.addEventListener('pointerdown', e => {
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
    renderer.cam.autoOrbit = false;
    $('b-orbit').classList.remove('on');
  });
  canvas.addEventListener('pointermove', e => {
    if (!dragging) return;
    renderer.userYaw   -= (e.clientX - lastX) * 0.005;
    renderer.userPitch += (e.clientY - lastY) * 0.003;
    renderer.userPitch = Math.max(-0.65, Math.min(0.65, renderer.userPitch));
    lastX = e.clientX; lastY = e.clientY;
  });
  const endDrag = () => { dragging = false; };
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    const c = renderer.cam;
    c.radius = Math.max(0.35, Math.min(3.55, c.radius + e.deltaY * 0.0011));
    /* pull the field of view in as the camera approaches the plasma */
    c.fov = Math.max(0.72, Math.min(1.45, 0.72 + (c.radius - 0.35) * 0.23));
  }, { passive: false });

  /* --------------------------------------------------------- controls --- */
  const sl = {
    nbi: $('c-nbi'), icr: $('c-icr'), ecr: $('c-ecr'),
    gas: $('c-gas'), ip: $('c-ip'), bt: $('c-bt'), rate: $('c-rate')
  };
  const lab = {
    nbi: $('v-nbi'), icr: $('v-icr'), ecr: $('v-ecr'),
    gas: $('v-gas'), ip: $('v-ip'), bt: $('v-bt'), rate: $('v-rate')
  };

  function manual() {
    if (!phys.autoPilot) return;
    phys.autoPilot = false;
    $('b-auto').classList.remove('on');
    phys.pushAlarm('MANUAL CONTROL ENGAGED', 'warn');
  }

  sl.nbi.addEventListener('input', () => { manual(); phys.PnbiSet = +sl.nbi.value; });
  sl.icr.addEventListener('input', () => { manual(); phys.PicrSet = +sl.icr.value; });
  sl.ecr.addEventListener('input', () => { manual(); phys.PecrSet = +sl.ecr.value; });
  sl.gas.addEventListener('input', () => { manual(); phys.gasSet  = +sl.gas.value; });
  sl.ip .addEventListener('input', () => { manual(); phys.IpSet   = +sl.ip.value; });
  sl.bt .addEventListener('input', () => { phys.Bt = +sl.bt.value; });
  sl.rate.addEventListener('input', () => {
    sim.rate = +sl.rate.value;
    sim.paused = sim.rate === 0;
    lab.rate.textContent = sim.rate.toFixed(2) + '×';
  });

  /* push the scripted programme back into the sliders while on autopilot */
  function syncSliders() {
    if (!phys.autoPilot) return;
    sl.nbi.value = phys.PnbiSet; sl.icr.value = phys.PicrSet;
    sl.ecr.value = phys.PecrSet; sl.gas.value = phys.gasSet;
    sl.ip.value  = phys.IpSet;
  }
  function syncLabels() {
    lab.nbi.textContent = phys.Pnbi.toFixed(1) + ' MW';
    lab.icr.textContent = phys.Picr.toFixed(1) + ' MW';
    lab.ecr.textContent = phys.Pecr.toFixed(1) + ' MW';
    lab.gas.textContent = phys.gas.toFixed(2);
    lab.ip.textContent  = phys.Ip.toFixed(2) + ' MA';
    lab.bt.textContent  = phys.Bt.toFixed(2) + ' T';
  }

  const toggle = (btn, fn) => {
    btn.addEventListener('click', () => {
      const on = btn.classList.toggle('on');
      fn(on);
    });
  };

  $('b-auto').addEventListener('click', () => {
    phys.autoPilot = !phys.autoPilot;
    $('b-auto').classList.toggle('on', phys.autoPilot);
    phys.pushAlarm(phys.autoPilot ? 'AUTOPILOT RE-ENGAGED'
                                  : 'MANUAL CONTROL ENGAGED', 'warn');
  });
  $('b-pellet').addEventListener('click', () => phys.injectPellet());
  $('b-elm').addEventListener('click', () => phys.forceELM());
  $('b-disrupt').addEventListener('click',
    () => phys.triggerDisruption('OPERATOR INITIATED'));
  $('b-mitigate').addEventListener('click', () => phys.mitigate());
  $('b-reset').addEventListener('click', () => {
    phys.reset(); phys.autoPilot = true;
    $('b-auto').classList.add('on');
  });

  toggle($('b-orbit'), on => { renderer.cam.autoOrbit = on; });
  toggle($('b-lines'), on => { view.fieldLines = on; });
  toggle($('b-parts'), on => { view.particles = on; });

  const QLIST = ['low', 'medium', 'high', 'ultra'];
  $('b-quality').addEventListener('click', e => {
    const cur = QLIST.indexOf(e.target.dataset.q);
    const next = QLIST[(cur + 1) % QLIST.length];
    e.target.dataset.q = next;
    e.target.textContent = 'QUALITY: ' + next.toUpperCase();
    renderer.setQuality(next);
    layout();
  });

  /* ------------------------------------------------------- keyboard ----- */
  window.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT') return;
    switch (e.key.toLowerCase()) {
      case 'h': hud.toggle(); break;
      case 'c': $('controls').classList.toggle('hidden'); break;
      case 'p':
        sim.paused = !sim.paused;
        sl.rate.value = sim.paused ? 0 : sim.rate || 1;
        lab.rate.textContent = sim.paused ? 'PAUSED' : sim.rate.toFixed(2) + '×';
        break;
      case 'f':
        if (document.fullscreenElement) document.exitFullscreen();
        else document.documentElement.requestFullscreen();
        break;
      case 'o': $('b-orbit').click(); break;
      case 'l': $('b-lines').click(); break;
      case 't': $('b-parts').click(); break;
      case ' ': e.preventDefault(); phys.injectPellet(); break;
      case 'e': phys.forceELM(); break;
      case 'r': $('b-reset').click(); break;
      case 's': {
        const a = document.createElement('a');
        a.download = 'tokamak_shot' + phys.shotNo + '_t' + phys.t.toFixed(1) + '.png';
        a.href = canvas.toDataURL('image/png');
        a.click();
        break;
      }
      case '1': case '2': case '3': case '4': {
        const q = QLIST[+e.key - 1];
        $('b-quality').dataset.q = q;
        $('b-quality').textContent = 'QUALITY: ' + q.toUpperCase();
        renderer.setQuality(q); layout();
        break;
      }
    }
  });

  document.addEventListener('visibilitychange', () => { last = performance.now(); });

  /* ------------------------------------------------------------ loop ---- */
  let last = performance.now();
  let t = 0;
  let labelAcc = 0;

  function frame(now) {
    requestAnimationFrame(frame);
    const raw = (now - last) / 1000;
    last = now;
    const dt = Math.min(raw, 0.05);          // clamp after a tab switch
    t += dt;

    const simDt = sim.paused ? 0 : dt * sim.rate;
    if (simDt > 0) phys.step(simDt);

    renderer.render(phys, t, dt, view);
    hud.update(phys, simDt, t);

    labelAcc += dt;
    if (labelAcc > 0.1) { labelAcc = 0; syncSliders(); syncLabels(); }

    renderer.adapt((performance.now() - now), window.innerWidth, window.innerHeight);
  }

  /* ------------------------------------------------------------ start --- */
  boot(() => {
    layout();
    sim.running = true;
    last = performance.now();
    requestAnimationFrame(frame);
  });

  /* expose for console tinkering */
  window.SIM = { phys, renderer, hud, sim, view };
})();
