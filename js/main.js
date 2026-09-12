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
    plasma: true, coils: true, heatMap: false, crossSection: false,
    highlight: 0
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

  /* --------------------------------------------------------- anatomy --- */
  /* Callout anchors are given in machine coordinates: minor radius r,
     height y, and a bearing offset from the centre of the cutaway wedge —
     so a label stays welded to its component as the model turns.
     `id` matches the material id the shader uses to pick out the part
     (20 is the plasma itself, which is volumetric, not a surface).       */
  /* Negative bearings put the callouts on the right-hand cut face, clear of
     the description panel on the left.  Heights are spread so the leaders
     do not pile up on top of one another.                               */
  const PARTS = [
    { id: 5,  r: 3.62, y: 2.05, dphi: -1.02, name: 'Плазма камерасы',
      desc: 'Тор тәрізді вакуумдық камера, 10⁻⁶ Па' },
    { id: 6,  r: 3.22, y: -1.25, dphi: -1.02, name: 'Бірінші қабырға',
      desc: 'Бериллий панельдер, нейтронды сіңіреді' },
    { id: 3,  r: 4.92, y: -0.85, dphi: -1.16, name: 'Полоидалды магнит',
      desc: 'Плазманың пішіні мен орнын реттейді' },
    { id: 1,  r: 5.32, y: 3.05, dphi: -1.32, name: 'Криостат',
      desc: 'Магниттерді 4.5 K-де ұстайтын қаптама' },
    { id: 2,  r: 4.30, y: 0.35, dphi: -1.02, name: 'Тороидалды магнит',
      desc: '18 D-пішінді Nb₃Sn катушка, 5.3 Тл' },
    { id: 20, r: 2.28, y: 0.55, dphi: -1.02, name: 'Плазма',
      desc: '150 млн °C, дейтерий–тритий' }
  ];

  let anIndex = -1, anPinned = -1;
  const anList = $('an-list');
  const anMarkers = $('an-markers');
  PARTS.forEach((part, i) => {
    const li = document.createElement('li');
    li.innerHTML = '<span><b>' + part.name + '</b><span>' + part.desc + '</span></span>';
    li.addEventListener('mouseenter', () => setHighlight(i));
    li.addEventListener('click', () => pinHighlight(anPinned === i ? -1 : i));
    anList.appendChild(li);
    part.li = li;

    const m = document.createElement('div');
    m.className = 'an-marker';
    m.textContent = String(i + 1);
    anMarkers.appendChild(m);
    part.marker = m;
  });
  anList.addEventListener('mouseleave', () => setHighlight(anPinned));

  function setHighlight(i) {
    anIndex = i;
    view.highlight = i >= 0 ? PARTS[i].id : 0;
    PARTS.forEach((p, k) => {
      p.li.classList.toggle('active', k === i);
      p.marker.classList.toggle('active', k === i);
      p.marker.classList.toggle('dim', i >= 0 && k !== i);
    });
  }
  function pinHighlight(i) { anPinned = i; setHighlight(i); }

  /* Project every anchor to screen space each frame. */
  function updateMarkers() {
    const W = window.innerWidth, H = window.innerHeight;
    const base = Math.PI * 0.5 - renderer.shot.cutCenter;
    for (const p of PARTS) {
      const phi = base + p.dphi;
      const s = renderer.project(
        [Math.cos(phi) * p.r, p.y, Math.sin(phi) * p.r], W, H);
      const m = p.marker;
      if (!s || s.ang > 0.94) { m.classList.add('off'); continue; }
      m.classList.remove('off');
      m.style.left = s.x.toFixed(1) + 'px';
      m.style.top = s.y.toFixed(1) + 'px';
    }
  }

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

  /* A discharge runs 150 s and only the burn phase looks like anything: at
     breakdown the chamber is dark and cold, during ramp-down it is a dull
     haze.  The landing model free-runs so it stays alive, but arriving
     inside should always land on ignition — so the flight rewinds the shot
     to just before burn and the 7 s move ends as the plasma lights up.   */
  let burnState = null;

  function startFlight() {
    if (sim.stage !== 'landing' && sim.stage !== 'anatomy') return;
    pinHighlight(-1);
    if (burnState) phys.restore(burnState);
    setStage('flight');
    renderer.startFlight();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function showAnatomy() {
    if (sim.stage !== 'landing') return;
    setStage('anatomy');
    renderer.toAnatomy();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function leaveAnatomy() {
    pinHighlight(-1);
    setStage('landing');
    renderer.toLanding();
  }

  function finishFlight() {
    setStage('sim');
    $('flight-ui').style.opacity = '';
  }

  function exitToLanding() {
    if (advisor) advisor.setFocus(false);
    const box = $('fault');
    if (box) box.hidden = true;
    faulted = false;
    setStage('landing');
    renderer.toLanding();
  }

  $('btn-start').addEventListener('click', startFlight);
  $('btn-start-2').addEventListener('click', startFlight);
  $('nav-launch').addEventListener('click', startFlight);
  $('btn-anatomy').addEventListener('click', showAnatomy);
  $('an-back').addEventListener('click', leaveAnatomy);
  $('an-start').addEventListener('click', startFlight);
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
    if (sim.stage !== 'sim' && sim.stage !== 'anatomy') return;
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

  let advisor = null;
  if (window.Advisor) {
    advisor = new Advisor({
      phys, sliders: sl, manual, syncLabels,
      onLaunch: function () {}
    });
  }

  let faulted = false;
  let overheatAcc = 0;
  let disruptAcc = 0;
  function tripFault(cause) {
    if (faulted || sim.stage !== 'sim') return;
    faulted = true;
    sim.paused = true;
    phys.autoPilot = false;
    phys.PnbiSet = 0; phys.PicrSet = 0; phys.PecrSet = 0; phys.gasSet = 0;
    if (advisor) { advisor.setFocus(false); advisor.hideRail(); }
    const box = $('fault');
    const causeEl = $('fault-cause');
    const logEl = $('fault-log');
    if (causeEl) causeEl.textContent = cause;
    if (logEl) {
      logEl.innerHTML = '';
      const rows = (phys.alarms || []).slice(0, 8);
      if (!rows.length) {
        const li = document.createElement('li');
        li.textContent = cause;
        logEl.appendChild(li);
      }
      rows.forEach(function (a) {
        const li = document.createElement('li');
        li.textContent = 't=' + Number(a.t).toFixed(1) + ' с  ·  ' + a.text;
        logEl.appendChild(li);
      });
    }
    if (box) box.hidden = false;
  }
  function clearFault() {
    faulted = false;
    overheatAcc = 0;
    disruptAcc = 0;
    phys.reset();
    phys.autoPilot = true;
    $('b-auto').classList.add('on');
    setPaused(false);
    if (advisor) advisor._cool = {};
    const box = $('fault');
    if (box) box.hidden = true;
  }
  $('fault-restart').addEventListener('click', clearFault);

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
  /* the renderer can step the preset down on its own if it cannot hold
     frame rate — keep the button label honest when it does */
  renderer.onQualityChange = q => {
    const btn = $('b-quality');
    btn.dataset.q = q;
    btn.textContent = 'САПА: ' + QNAME[q];
    phys.pushAlarm('САПА АВТОМАТТЫ ТӨМЕНДЕДІ: ' + QNAME[q], 'warn');
  };

  /* ------------------------------------------------------- keyboard ----- */
  window.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    const k = e.key.toLowerCase();
    if (k === 'enter' && (sim.stage === 'landing' || sim.stage === 'anatomy')) {
      startFlight(); return;
    }
    if (k === 'escape') {
      if (document.body.classList.contains('ai-focus')) {
        if (advisor) advisor.setFocus(false);
        return;
      }
      if (sim.stage === 'sim') exitToLanding();
      else if (sim.stage === 'anatomy') leaveAnatomy();
      return;
    }
    if (k === 'a' && sim.stage === 'landing') { showAnatomy(); return; }
    if (sim.stage !== 'sim') return;
    switch (k) {
      case 'h': hud.toggle(); break;
      case 'i':
        if (advisor) advisor.setFocus(!advisor.focused);
        break;
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
    /* keep the landing model in its glowing phase however long it is left
       running; inside the simulation the discharge is free to play out */
    if (sim.stage !== 'sim' && burnState && (phys.t > 112 || phys.disrupted)) {
      phys.restore(burnState);
    }

    renderer.updateShot(dt, t);

    if (sim.stage === 'flight') {
      updateFlightUI(renderer.shot.flight);
      if (renderer.shot.mode === 'interior') finishFlight();
    }

    renderer.render(phys, t, dt, view);
    if (sim.stage === 'anatomy') updateMarkers();
    /* Keep filling the diagnostic history during the landing and flight so
       the charts already have context the moment the operator arrives. */
    if (sim.stage === 'sim') hud.update(phys, simDt, t);
    else hud.hist.push(phys, simDt);

    labelAcc += dt;
    if (labelAcc > 0.1 && sim.stage === 'sim') {
      labelAcc = 0; syncSliders(); syncLabels();
      if (!faulted && advisor) advisor.watch(phys, sim.stage);
      if (!faulted) {
        const pending = advisor && advisor.pending && advisor.pending.length;
        if (phys.qDivSteady > 16) overheatAcc += 0.1;
        else overheatAcc = Math.max(0, overheatAcc - 0.12);
        if (phys.disrupted) disruptAcc += 0.1;
        else disruptAcc = 0;
        /* give the right-side advisor time to ask; only trip if ignored */
        if (overheatAcc > 22 && (!pending || overheatAcc > 45)) {
          phys.pushAlarm('ДИВЕРТОР ҚЫЗУЫ — АВАРИЯЛЫҚ ТОҚТАТУ', 'crit');
          tripFault('Токамак қатты қызды (q_div = ' + phys.qDiv.toFixed(1) +
            ' МВт/м²). Жұмысты жалғастыру қауіпті.');
        } else if (disruptAcc > 14 && phys.Ip < 0.4 && (!pending || disruptAcc > 28)) {
          const last = (phys.alarms[0] && phys.alarms[0].text) || 'ДИЗРУПЦИЯ';
          tripFault('Плазма жұмысын тоқтатты. ' + last);
        }
      }
    }

    /* the real frame delta, not the JS render time — GPU work is async */
    renderer.adapt(raw * 1000, window.innerWidth, window.innerHeight);
  }

  /* ------------------------------------------------------------ start --- */
  boot(() => {
    layout();
    /* Run up to the moment the plasma ignites and keep that state, then
       carry on a little further so the landing shot opens mid-burn. */
    for (let i = 0; i < 2400; i++) phys.step(0.01);
    burnState = phys.snapshot();
    for (let i = 0; i < 1000; i++) phys.step(0.01);
    /* size the renderer to whatever this GPU can actually sustain */
    try {
      const r = renderer.probe(phys, window.innerWidth, window.innerHeight);
      const btn = $('b-quality');
      btn.dataset.q = r.quality;
      btn.textContent = 'САПА: ' + QNAME[r.quality];
      renderer.toLanding();
    } catch (e) { /* probing is best-effort */ }
    hud.resize();
    last = performance.now();
    requestAnimationFrame(frame);
  });

  window.SIM = { phys, renderer, hud, sim, view, PARTS, advisor,
                 startFlight, exitToLanding, showAnatomy, leaveAnatomy,
                 setHighlight, updateMarkers };
})();
