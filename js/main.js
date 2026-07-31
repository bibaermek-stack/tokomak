/* ==========================================================================
   main.js — stage machine, simulation loop and operator controls

   Stages:  landing -> flight -> sim   (and back via EXIT)
   ========================================================================== */
(function () {
  'use strict';

  const $ = id => document.getElementById(id);
  const body = document.body;

  /* ------------------------------------------------------------ boot ---- */
  const BOOT_LINES = [
    'КРИОГЕНДІК ЖҮЙЕ ................ 4.5 K ҚАЛЫПТЫ',
    'TF КАТУШКА ЖҮЙЕСІ .............. 18/18 АСҚЫН ӨТКІЗГІШ',
    'ОРТАЛЫҚ СОЛЕНОИД ............... ФЛЮКС ДАЙЫН',
    'ТОР ВАКУУМЫ .................... 1.0e-06 Па',
    'D–T ОТЫН ЕНГІЗУ ................ БҰҒАТТАУ АШЫҚ',
    'ДИАГНОСТИКА .................... 41 АРНА ЖҰМЫСТА',
    'ПЛАЗМАНЫ БАСҚАРУ ............... НАҚТЫ УАҚЫТ 1 кГц',
    'ІСКЕ ҚОСУ ТІЗБЕГІ .............. ДАЙЫН'
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
        setTimeout(tick, 120 + Math.random() * 80);
      } else {
        setTimeout(() => {
          $('boot').classList.add('gone');
          setTimeout(() => { const b = $('boot'); if (b) b.remove(); }, 800);
          done();
        }, 300);
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
    f.textContent = 'РЕАКТОР КӨРІНІСІН ІСКЕ ҚОСУ МҮМКІН БОЛМАДЫ — ' + e.message +
      '  ::  Бұл симуляция үшін WebGL2 қолдайтын браузер қажет.';
    const b = $('boot'); if (b) b.remove();
    return;
  }

  phys = new Tokamak();
  hud = new HUD($('hud'));

  const view = {
    fieldLines: false, particles: false,
    plasma: true, coils: true, heatMap: false, crossSection: false
  };
  const sim = { rate: 1.0, paused: false, stage: 'landing' };

  /* --------------------------------------------------------- narration -- */
  /* Captions keyed to the flight path, so the move reads like a documentary */
  const BEATS = [
    { p: 0.00, t: 'КРИОСТАТҚА ЖАҚЫНДАУ',        s: 'Биіктігі 29 метр · 23 000 тонна' },
    { p: 0.30, t: 'ЖЫЛУ ҚАЛҚАНЫНАН ӨТУ',        s: 'Ішіндегінің бәрі 4.5 кельвинде' },
    { p: 0.50, t: 'ТОРОИДАЛДЫ КАТУШКАЛАР АРҚЫЛЫ', s: '18 Nb₃Sn магниті · 41 ГДж энергия' },
    { p: 0.70, t: 'ВАКУУМДЫҚ КАМЕРАНЫ КЕСІП ӨТУ', s: '10⁻⁶ Па · қос қабырғалы болат' },
    { p: 0.86, t: 'ПЛАЗМА КАМЕРАСЫНА КІРУ',     s: 'Бериллий қабырға · ар жағында 150 млн °C' },
    { p: 0.97, t: 'ПЛАЗМА САҚИНАСЫНЫҢ ІШІНДЕ',  s: 'Диагностика іске қосылуда' }
  ];

  function layout() {
    renderer.resize(window.innerWidth, window.innerHeight);
    hud.resize();
  }
  window.addEventListener('resize', layout);

  /* ---------------------------------------------------- stage handling -- */
  function setStage(s) {
    sim.stage = s;
    body.dataset.stage = s;
  }

  function startFlight() {
    if (sim.stage !== 'landing') return;
    setStage('flight');
    renderer.startFlight();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function finishFlight() {
    setStage('sim');
    $('flight-ui').style.opacity = '';
  }

  function exitToLanding() {
    setStage('landing');
    renderer.toLanding();
  }

  $('btn-start').addEventListener('click', startFlight);
  $('btn-start-2').addEventListener('click', startFlight);
  $('nav-launch').addEventListener('click', startFlight);
  $('btn-skip').addEventListener('click', () => {
    renderer.shot.flight = 1;
    renderer.shot.mode = 'interior';
    finishFlight();
  });
  $('b-exit').addEventListener('click', exitToLanding);

  /* ------------------------------------------------------ camera input -- */
  let dragging = false, lastX = 0, lastY = 0;
  canvas.addEventListener('pointerdown', e => {
    if (sim.stage === 'flight') return;
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
    if (sim.stage === 'sim') {
      renderer.cam.autoOrbit = false;
      $('b-orbit').classList.remove('on');
    }
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
    /* on the landing page the wheel belongs to the document, not the model */
    if (sim.stage !== 'sim') return;
    e.preventDefault();
    renderer.userZoom = Math.max(-1.05, Math.min(0.9,
      renderer.userZoom + e.deltaY * 0.0012));
  }, { passive: false });

  /* --------------------------------------------------------- controls --- */
  const sl = {
    nbi: $('c-nbi'), icr: $('c-icr'), ecr: $('c-ecr'),
    gas: $('c-gas'), ip: $('c-ip'), bt: $('c-bt')
  };
  const lab = {
    nbi: $('v-nbi'), icr: $('v-icr'), ecr: $('v-ecr'),
    gas: $('v-gas'), ip: $('v-ip'), bt: $('v-bt')
  };

  function manual() {
    if (!phys.autoPilot) return;
    phys.autoPilot = false;
    $('b-auto').classList.remove('on');
    phys.pushAlarm('ҚОЛМЕН БАСҚАРУ ҚОСЫЛДЫ', 'warn');
  }

  sl.nbi.addEventListener('input', () => { manual(); phys.PnbiSet = +sl.nbi.value; });
  sl.icr.addEventListener('input', () => { manual(); phys.PicrSet = +sl.icr.value; });
  sl.ecr.addEventListener('input', () => { manual(); phys.PecrSet = +sl.ecr.value; });
  sl.gas.addEventListener('input', () => { manual(); phys.gasSet  = +sl.gas.value; });
  sl.ip .addEventListener('input', () => { manual(); phys.IpSet   = +sl.ip.value; });
  sl.bt .addEventListener('input', () => { phys.Bt = +sl.bt.value; });

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

  const toggle = (id, fn) => {
    const btn = $(id);
    btn.addEventListener('click', () => fn(btn.classList.toggle('on'), btn));
  };

  $('b-auto').addEventListener('click', () => {
    phys.autoPilot = !phys.autoPilot;
    $('b-auto').classList.toggle('on', phys.autoPilot);
    phys.pushAlarm(phys.autoPilot ? 'АВТОПИЛОТ ҚАЙТА ҚОСЫЛДЫ'
                                  : 'ҚОЛМЕН БАСҚАРУ ҚОСЫЛДЫ', 'warn');
  });
  $('b-pellet').addEventListener('click', () => phys.injectPellet());
  $('b-elm').addEventListener('click', () => phys.forceELM());
  $('b-disrupt').addEventListener('click',
    () => phys.triggerDisruption('ОПЕРАТОР ІСКЕ ҚОСТЫ'));
  $('b-mitigate').addEventListener('click', () => phys.mitigate());
  $('b-reset').addEventListener('click', () => {
    phys.reset(); phys.autoPilot = true;
    $('b-auto').classList.add('on');
    setPaused(false);
  });

  function setPaused(p) {
    sim.paused = p;
    const b = $('b-pause');
    b.classList.toggle('on', p);
    b.textContent = p ? '▶ ЖАЛҒАСТЫРУ' : '⏸ ТОҚТАТУ';
  }
  $('b-pause').addEventListener('click', () => setPaused(!sim.paused));

  toggle('b-orbit',  on => { renderer.cam.autoOrbit = on; });
  toggle('b-lines',  on => { view.fieldLines = on; });
  toggle('b-parts',  on => { view.particles = on; });
  toggle('b-plasma', on => { view.plasma = on; });
  toggle('b-heat',   on => { view.heatMap = on; });
  toggle('b-xsec',   on => { view.crossSection = on; });

  const QLIST = ['low', 'medium', 'high', 'ultra'];
  const QNAME = { low: 'ТӨМЕН', medium: 'ОРТА', high: 'ЖОҒАРЫ', ultra: 'ЕҢ ЖОҒАРЫ' };
  const setQuality = q => {
    const btn = $('b-quality');
    btn.dataset.q = q;
    btn.textContent = 'САПА: ' + QNAME[q];
    renderer.setQuality(q);
    layout();
  };
  $('b-quality').addEventListener('click', e => {
    const cur = e.currentTarget.dataset.q;
    setQuality(QLIST[(QLIST.indexOf(cur) + 1) % QLIST.length]);
  });

  /* ------------------------------------------------------- keyboard ----- */
  window.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT') return;
    const k = e.key.toLowerCase();
    if (k === 'enter' && sim.stage === 'landing') { startFlight(); return; }
    if (k === 'escape' && sim.stage === 'sim') { exitToLanding(); return; }
    if (sim.stage !== 'sim') return;
    switch (k) {
      case 'h': hud.toggle(); break;
      case 'c': $('controls').classList.toggle('hidden'); break;
      case 'p': setPaused(!sim.paused); break;
      case 'f':
        if (document.fullscreenElement) document.exitFullscreen();
        else document.documentElement.requestFullscreen();
        break;
      case 'o': $('b-orbit').click(); break;
      case 'l': $('b-lines').click(); break;
      case 't': $('b-parts').click(); break;
      case 'x': $('b-xsec').click(); break;
      case 'm': $('b-heat').click(); break;
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
      case '1': case '2': case '3': case '4':
        setQuality(QLIST[+k - 1]);
        break;
    }
  });

  document.addEventListener('visibilitychange', () => { last = performance.now(); });

  /* ------------------------------------------------------------ loop ---- */
  const flBar = document.querySelector('.fl-bar i');
  const flCapT = document.querySelector('.fl-caption b');
  const flCapS = document.querySelector('.fl-caption span');
  let lastBeat = -1;

  function updateFlightUI(p) {
    flBar.style.width = (p * 100).toFixed(1) + '%';
    let idx = 0;
    for (let i = 0; i < BEATS.length; i++) if (p >= BEATS[i].p) idx = i;
    if (idx !== lastBeat) {
      lastBeat = idx;
      flCapT.textContent = BEATS[idx].t;
      flCapS.textContent = BEATS[idx].s;
    }
  }

  let last = performance.now();
  let t = 0;
  let labelAcc = 0;

  function frame(now) {
    requestAnimationFrame(frame);
    const raw = (now - last) / 1000;
    last = now;
    const dt = Math.min(raw, 0.05);
    t += dt;

    /* the plasma keeps evolving during the landing shot so the glow lives,
       but the dashboard history only matters once we are inside */
    const simDt = sim.paused ? 0 : dt * sim.rate;
    if (simDt > 0) phys.step(simDt);

    renderer.updateShot(dt, t);

    if (sim.stage === 'flight') {
      updateFlightUI(renderer.shot.flight);
      if (renderer.shot.mode === 'interior') finishFlight();
    }

    renderer.render(phys, t, dt, view);
    /* Keep filling the diagnostic history during the landing and flight so
       the charts already have context the moment the operator arrives. */
    if (sim.stage === 'sim') hud.update(phys, simDt, t);
    else hud.hist.push(phys, simDt);

    labelAcc += dt;
    if (labelAcc > 0.1 && sim.stage === 'sim') {
      labelAcc = 0; syncSliders(); syncLabels();
    }

    renderer.adapt(performance.now() - now, window.innerWidth, window.innerHeight);
  }

  /* ------------------------------------------------------------ start --- */
  boot(() => {
    layout();
    /* give the landing shot a plasma that is already burning */
    for (let i = 0; i < 3400; i++) phys.step(0.01);
    last = performance.now();
    requestAnimationFrame(frame);
  });

  window.SIM = { phys, renderer, hud, sim, view, startFlight, exitToLanding };
})();
