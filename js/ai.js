/* ==========================================================================
   ai.js — Кеңесші panel: advice + confirm-then-apply. No model names here.
   ========================================================================== */
(function (global) {
  'use strict';

  const $ = id => document.getElementById(id);

  function detectAlarms(phys) {
    const hits = [];
    if (!phys || phys.t < 5) return hits;
    if (phys.disrupted) {
      hits.push({
        id: 'disrupt',
        msg: 'Дизрупция. MGI іске қосайын ба?',
        proposal: {
          actuator: 'mitigate', from: 0, to: 1, unit: '', danger: true,
          reason: 'токты бәсеңдету', imas_path: 'disruption.global_quantities.ip'
        }
      });
      return hits;
    }
    if (phys.fG > 0.90) {
      hits.push({
        id: 'fg',
        msg: 'fG=' + phys.fG.toFixed(2) + ' — Гринвальд шегіне жақын. Газды азайтайын ба?',
        proposal: {
          actuator: 'gas', from: phys.gasSet,
          to: Math.max(0, +(phys.gasSet * 0.55).toFixed(2)),
          unit: '', danger: false, reason: 'тығыздық шегі',
          imas_path: 'pulse_schedule.density_control.gas_puff'
        }
      });
    }
    if (phys.q95 < 2.6 && phys.Ip > 1) {
      hits.push({
        id: 'q95',
        msg: 'q95=' + phys.q95.toFixed(2) + ' төмен. Ip-ны сәл түсірейін бе?',
        proposal: {
          actuator: 'ip', from: phys.IpSet,
          to: Math.max(0.2, +(phys.IpSet * 0.92).toFixed(2)),
          unit: 'МА', danger: false, reason: 'kink тұрақтылығы',
          imas_path: 'summary.global_quantities.q_95'
        }
      });
    }
    if (phys.betaN > 3.5) {
      hits.push({
        id: 'betan',
        msg: 'βN=' + phys.betaN.toFixed(2) + ' жоғары. NBI-ді азайтайын ба?',
        proposal: {
          actuator: 'p_nbi', from: phys.PnbiSet,
          to: Math.max(0, +(phys.PnbiSet * 0.75).toFixed(1)),
          unit: 'МВт', danger: false, reason: 'бета шегі',
          imas_path: 'summary.global_quantities.beta_normal'
        }
      });
    }
    if (phys.qDiv > 8) {
      hits.push({
        id: 'qdiv',
        msg: 'q_div=' + phys.qDiv.toFixed(1) + ' МВт/м². Қыздыруды азайтайын ба?',
        proposal: {
          actuator: 'p_nbi', from: phys.PnbiSet,
          to: Math.max(0, +(phys.PnbiSet * 0.7).toFixed(1)),
          unit: 'МВт', danger: false, reason: 'дивертор жылуы',
          imas_path: 'divertors.tungsten.target_heat_flux'
        }
      });
    }
    const lastAl = (phys.alarms && phys.alarms.length)
      ? String(phys.alarms[phys.alarms.length - 1]) : '';
    if (/H-L КЕРІ/.test(lastAl)) {
      hits.push({
        id: 'hl',
        msg: 'H-L кері ауысу. Қыздыруды ұстап тұрайын ба?',
        proposal: {
          actuator: 'p_nbi', from: phys.PnbiSet,
          to: Math.min(50, +(phys.PnbiSet + 4).toFixed(1)),
          unit: 'МВт', danger: false, reason: 'H-mode ұстау',
          imas_path: 'nbi.power_launched'
        }
      });
    }
    return hits;
  }

  function snapshot(phys) {
    return {
      t: phys.t, Ip: phys.Ip, Bt: phys.Bt, Te: phys.Te, Ti: phys.Ti,
      ne: phys.ne, nG: phys.nG, fG: phys.fG, Pfus: phys.Pfus, Q: phys.Q,
      tauE: phys.tauE, H98: phys.H98, betaN: phys.betaN, q95: phys.q95,
      Zeff: phys.Zeff, W: phys.W, Vloop: phys.Vloop, Prad: phys.Prad,
      Pnbi: phys.Pnbi, Picr: phys.Picr, Pecr: phys.Pecr,
      PnbiSet: phys.PnbiSet, PicrSet: phys.PicrSet, PecrSet: phys.PecrSet,
      gasSet: phys.gasSet, IpSet: phys.IpSet,
      qDiv: phys.qDiv, neutronRate: phys.neutronRate,
      hMode: phys.hMode, disrupted: phys.disrupted, mode: phys.mode,
      alarms: (phys.alarms || []).slice(-6)
    };
  }

  function Advisor(opts) {
    this.phys = opts.phys;
    this.sl = opts.sliders;
    this.manual = opts.manual;
    this.syncLabels = opts.syncLabels;
    this.base = '';
    this.mode = 'live';
    this.history = [];
    this.busy = false;
    this.available = false;
    this.pending = null;
    this.dangerArmed = false;
    this.focused = false;
    this._cool = {};
    this._watchAt = 0;
    this._launched = false;
    this.onLaunch = opts.onLaunch || function () {};
    this._bind();
    this.pendingFiles = [];
    this.log('bot', 'Сөйлесу осында. Растау — оң жақта. Сурет/файл қосуға болады.');
    this._detect();
  }

  Advisor.prototype.clearAlarm = function () {
    const tab = $('ai-tab');
    const launch = $('ai-launch');
    if (tab) tab.classList.remove('alarm');
    if (launch) launch.classList.remove('alarm');
  };

  Advisor.prototype.setFocus = function (on) {
    this.focused = !!on;
    document.body.classList.toggle('ai-focus', this.focused);
    const tab = $('ai-tab');
    if (tab) tab.classList.toggle('on', this.focused);
    if (this.focused) {
      const input = $('ai-input');
      if (input && !input.disabled) setTimeout(function () { input.focus(); }, 80);
    }
  };

  Advisor.prototype.watch = function (phys, stage) {
    if (stage !== 'sim' || !phys) return;
    if ($('fault') && !$('fault').hidden) return;
    const now = performance.now();
    if (now - this._watchAt < 600) return;
    this._watchAt = now;
    const hits = detectAlarms(phys);
    if (!hits.length && !(this.pending && this.pending.length)) {
      this.clearAlarm();
      return;
    }
    for (let i = 0; i < hits.length; i++) {
      const h = hits[i];
      if (now - (this._cool[h.id] || 0) < 22000) continue;
      this._cool[h.id] = now;
      this._raise(h);
      break;
    }
  };

  Advisor.prototype._raise = function (hit) {
    const tab = $('ai-tab');
    const launch = $('ai-launch');
    if (tab) tab.classList.add('alarm');
    if (launch) launch.classList.add('alarm');
    this._launched = false;
    if (hit.proposal) {
      this.pending = [hit.proposal];
      this.dangerArmed = false;
    }
    this.renderRail(hit.msg);
    if (!this.available) {
      this.log('alarm', hit.msg);
      return;
    }
    this.send('[ДАБЫЛ] ' + hit.msg, { alarm: true });
  };

  Advisor.prototype.launchNow = function () {
    this.clearAlarm();
    if (this.pending && this.pending.length) {
      this.dangerArmed = true;
      this.confirm();
      this._launched = true;
    }
    this.setFocus(false);
    this.onLaunch();
  };

  Advisor.prototype._bind = function () {
    const self = this;
    const form = $('ai-form');
    const input = $('ai-input');
    const tab = $('ai-tab');
    const launch = $('ai-launch');
    const attach = $('ai-attach');
    const panel = $('ai-panel');

    form.addEventListener('submit', function (e) {
      e.preventDefault();
      const text = input.value.trim();
      if ((!text && !self.pendingFiles.length) || self.busy) return;
      input.value = '';
      self.clearAlarm();
      self.send(text);
    });
    if (attach) {
      attach.addEventListener('change', function () {
        self.addFiles(attach.files);
        attach.value = '';
      });
    }
    if (panel) {
      panel.addEventListener('dragover', function (e) {
        e.preventDefault();
      });
      panel.addEventListener('drop', function (e) {
        e.preventDefault();
        if (e.dataTransfer && e.dataTransfer.files) self.addFiles(e.dataTransfer.files);
      });
    }
    if (input) {
      input.addEventListener('paste', function (e) {
        const items = e.clipboardData && e.clipboardData.files;
        if (items && items.length) self.addFiles(items);
      });
    }
    if (tab) {
      tab.addEventListener('click', function () { self.setFocus(true); });
    }
    if (launch) {
      launch.addEventListener('click', function () { self.launchNow(); });
    }
    $('ai-card').addEventListener('click', function (e) {
      const act = e.target.getAttribute('data-act');
      if (act === 'yes') self.confirm();
      if (act === 'no') self.reject();
    });
    const railYes = $('ai-rail-yes');
    const railNo = $('ai-rail-no');
    if (railYes) railYes.addEventListener('click', function () { self.confirm(); });
    if (railNo) railNo.addEventListener('click', function () { self.reject(); });
  };

  Advisor.prototype.renderRail = function (msg) {
    const rail = $('ai-rail');
    if (!rail) return;
    const items = this.pending || [];
    if (!items.length) { rail.classList.remove('open'); return; }
    rail.classList.add('open');
    const m = $('ai-rail-msg');
    const a = $('ai-rail-act');
    if (m) m.textContent = msg || 'Ұсыныс. Растау немесе Жоқ.';
    if (a) {
      a.textContent = items.map(function (p) {
        return (p.actuator || '').toUpperCase() + '  ' +
          (p.from == null ? '—' : p.from) + ' → ' + p.to + ' ' + (p.unit || '');
      }).join('\n');
    }
  };

  Advisor.prototype.hideRail = function () {
    const rail = $('ai-rail');
    if (rail) rail.classList.remove('open');
  };

  Advisor.prototype.addFiles = function (list) {
    if (!list || !list.length) return;
    for (let i = 0; i < list.length; i++) {
      if (this.pendingFiles.length >= 4) break;
      this.pendingFiles.push(list[i]);
    }
    this._renderAttach();
  };

  Advisor.prototype._renderAttach = function () {
    const box = $('ai-attach-list');
    if (!box) return;
    box.innerHTML = '';
    const self = this;
    this.pendingFiles.forEach(function (f, idx) {
      const chip = document.createElement('div');
      chip.className = 'ai-chip';
      if (f.type && f.type.indexOf('image/') === 0) {
        const img = document.createElement('img');
        img.src = URL.createObjectURL(f);
        chip.appendChild(img);
      }
      const span = document.createElement('span');
      span.textContent = f.name;
      chip.appendChild(span);
      const x = document.createElement('button');
      x.type = 'button';
      x.textContent = '×';
      x.addEventListener('click', function () {
        self.pendingFiles.splice(idx, 1);
        self._renderAttach();
      });
      chip.appendChild(x);
      box.appendChild(chip);
    });
  };

  Advisor.prototype._uploadPending = function () {
    const self = this;
    const files = this.pendingFiles.slice();
    this.pendingFiles = [];
    this._renderAttach();
    const names = [];
    let chain = Promise.resolve();
    files.forEach(function (f) {
      chain = chain.then(function () {
        const fd = new FormData();
        fd.append('file', f, f.name);
        return fetch(self.base + '/ai/upload', { method: 'POST', body: fd })
          .then(function (r) { return r.json(); })
          .then(function (j) {
            if (j && j.ok && j.name) names.push(j.name);
          });
      });
    });
    return chain.then(function () { return names; });
  };

  Advisor.prototype._detect = function () {
    const self = this;
    const candidates = [];
    if (global.AI_BASE) candidates.push(String(global.AI_BASE).replace(/\/$/, ''));
    candidates.push('');
    const port = location.port;
    if (port === '8777' || location.protocol === 'file:') {
      candidates.push('http://127.0.0.1:8001');
    }
    if (location.hostname.indexOf('railway.app') === -1) {
      candidates.push('https://tokomak-production.up.railway.app');
    }
    (function next(i) {
      if (i >= candidates.length) {
        self._status(false);
        setTimeout(function () { self._detect(); }, 8000);
        return;
      }
      const b = candidates[i];
      fetch(b + '/ai/health', { cache: 'no-store' }).then(function (r) {
        if (!r.ok) throw new Error('no');
        return r.json();
      }).then(function (j) {
        self.base = b;
        self._status(!!j.ai_available);
        if (!j.ai_available) setTimeout(function () { self._detect(); }, 8000);
      }).catch(function () { next(i + 1); });
    })(0);
  };

  Advisor.prototype._status = function (ok) {
    this.available = ok;
    const input = $('ai-input');
    const send = $('ai-send');
    input.disabled = !ok;
    send.disabled = !ok;
    $('ai-avail').textContent = ok ? 'ОНЛАЙН' : 'КҮТУ';
    $('ai-avail').classList.toggle('on', ok);
    if (!ok) {
      input.placeholder = 'Кеңесші әлі қосылмаған';
    } else {
      input.placeholder = 'NBI-ді 33 МВт-қа көтер · Q қандай · 120 с есепте';
    }
  };

  Advisor.prototype.log = function (who, text, extra) {
    const confirm = (who === 'alarm' || who === 'confirm');
    const box = $(confirm ? 'ai-confirm-log' : 'ai-log') || $('ai-log');
    const row = document.createElement('div');
    row.className = 'ai-msg ' + who;
    const h = document.createElement('b');
    h.textContent = who === 'user' ? 'СІЗ'
      : (who === 'alarm' ? 'ДАБЫЛ' : (who === 'confirm' ? 'РАСТАУ' : 'КЕҢЕСШІ'));
    const p = document.createElement('p');
    p.textContent = text;
    row.appendChild(h);
    row.appendChild(p);
    if (extra) row.appendChild(extra);
    box.appendChild(row);
    box.scrollTop = box.scrollHeight;
  };

  Advisor.prototype.send = function (text, opt) {
    const self = this;
    opt = opt || {};
    if (!this.available) {
      this.log('bot', 'Кеңесші әлі қосылмаған.');
      return;
    }
    const nPending = this.pendingFiles.length;
    const label = text || (nPending ? ('файл: ' + nPending) : '');
    if (!opt.alarm) {
      this.log('user', label);
      this.history.push({ role: 'user', content: text || label });
    } else {
      this.log('alarm', String(text || '').replace(/^\[ДАБЫЛ\]\s*/, ''));
    }
    this.busy = true;
    $('ai-send').classList.add('on');

    this._uploadPending().then(function (names) {
      const body = {
        message: text || (names.length ? 'осы файлды қара' : ''),
        history: opt.alarm ? [] : self.history.slice(-8),
        snapshot: snapshot(self.phys),
        attachments: names
      };
      return fetch(self.base + '/ai/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
    }).then(function (r) {
      return r.json().then(function (j) {
        if (j.detail && typeof j.detail === 'object') j = j.detail;
        return { ok: r.ok, j: j };
      });
    }).then(function (pack) {
      const j = pack.j || {};
      if (j.ai_available === false) self._status(false);
      const reply = j.reply || 'Жауап жоқ.';
      if (opt.alarm) {
        self.log('confirm', reply, self._extras(j));
      } else {
        self.log('bot', reply, self._extras(j));
        self.history.push({ role: 'assistant', content: reply });
      }
      if (j.proposals && j.proposals.length && !self._launched) self.showCard(j.proposals);
    }).catch(function () {
      self.log(opt.alarm ? 'confirm' : 'bot', 'Кеңесші уақытша қолжетімсіз.');
    }).then(function () {
      self.busy = false;
      $('ai-send').classList.remove('on');
    });
  };

  Advisor.prototype._extras = function (j) {
    const wrap = document.createElement('div');
    wrap.className = 'ai-extra';
    if (j.citations && j.citations.length) {
      j.citations.slice(0, 6).forEach(function (c) {
        const s = document.createElement('span');
        s.className = 'ai-cite';
        s.textContent = c.path + (c.units ? ' [' + c.units + ']' : '');
        s.title = c.documentation || '';
        wrap.appendChild(s);
      });
    }
    if (j.table && j.table.length) {
      wrap.appendChild(self._table(j.table));
    }
    if (j.files && j.files.length) {
      wrap.appendChild(self._files(j.files));
    }
    if (j.simulate && j.simulate.final) {
      const f = j.simulate.final;
      const s = document.createElement('span');
      s.className = 'ai-cite';
      s.textContent = 'есеп ' + (j.simulate.machine || '') +
        '  Q=' + (f.Q != null ? Number(f.Q).toFixed(2) : '—') +
        '  Pfus=' + (f.Pfus != null ? Number(f.Pfus).toFixed(1) : '—') + ' МВт';
      wrap.appendChild(s);
    }
    return wrap.childNodes.length ? wrap : null;
  };

  Advisor.prototype._files = function (items) {
    const box = document.createElement('div');
    box.className = 'ai-files';
    const self = this;
    items.forEach(function (f) {
      const row = document.createElement('div');
      row.className = 'ai-file';
      if (f.op === 'list' && f.files) {
        row.textContent = f.files.length
          ? ('файлдар: ' + f.files.map(function (x) { return x.name; }).join(', '))
          : 'файл жоқ';
        box.appendChild(row);
        return;
      }
      const label = document.createElement('span');
      label.textContent = (f.ok ? '✓ ' : '✗ ') + (f.op || '') + ' ' + (f.name || '');
      row.appendChild(label);
      if (f.ok && f.name) {
        const a = document.createElement('a');
        a.href = (self.base || '') + '/ai/files/' + encodeURIComponent(f.name);
        a.textContent = 'жүктеу';
        a.download = f.name;
        row.appendChild(a);
      }
      if (f.content) {
        const pre = document.createElement('pre');
        pre.className = 'ai-file-body';
        pre.textContent = f.content.slice(0, 4000);
        row.appendChild(pre);
      }
      if (f.error) {
        const err = document.createElement('i');
        err.textContent = f.error;
        row.appendChild(err);
      }
      box.appendChild(row);
    });
    return box;
  };

  Advisor.prototype._table = function (rows) {
    const tbl = document.createElement('table');
    tbl.className = 'ai-table';
    const head = document.createElement('thead');
    head.innerHTML = '<tr><th>IMAS</th><th>МӘН</th><th>БІРЛІК</th></tr>';
    tbl.appendChild(head);
    const body = document.createElement('tbody');
    rows.forEach(function (r) {
      const tr = document.createElement('tr');
      const a = document.createElement('td');
      a.textContent = r.path || '';
      const b = document.createElement('td');
      b.textContent = (r.value == null ? '—' : r.value);
      const c = document.createElement('td');
      c.textContent = r.unit || '';
      tr.appendChild(a); tr.appendChild(b); tr.appendChild(c);
      body.appendChild(tr);
    });
    tbl.appendChild(body);
    return tbl;
  };

  Advisor.prototype.showCard = function (proposals) {
    this.pending = proposals;
    this.dangerArmed = false;
    const card = $('ai-card');
    card.hidden = false;
    const lines = proposals.map(function (p) {
      const a = (p.actuator || '').toUpperCase();
      const from = p.from == null ? '—' : p.from;
      const unit = p.unit || '';
      return a + '  ' + from + ' → ' + p.to + ' ' + unit +
        (p.danger ? '  ⚠' : '');
    }).join('\n');
    const reason = proposals.map(function (p) { return p.reason; }).filter(Boolean).join(' · ');
    card.innerHTML =
      '<div class="ai-card-body"><pre>' + lines + '</pre>' +
      (reason ? '<p>' + reason + '</p>' : '') +
      '</div><div class="ai-card-btns">' +
      '<button data-act="yes">РАСТАУ</button>' +
      '<button data-act="no">ЖОҚ</button></div>';
    if (proposals.some(function (p) { return p.danger; })) {
      card.classList.add('danger');
    } else {
      card.classList.remove('danger');
    }
    this.renderRail(reason || 'Ұсыныс');
  };

  Advisor.prototype.reject = function () {
    this.pending = null;
    this.dangerArmed = false;
    $('ai-card').hidden = true;
    this.hideRail();
    this.clearAlarm();
    this.log('confirm', 'Ұсыныс қабылданбады. Пульт өзгермеді.');
  };

  Advisor.prototype.confirm = function () {
    const items = this.pending;
    if (!items) return;
    const danger = items.some(function (p) { return p.danger; });
    if (danger && !this.dangerArmed) {
      this.dangerArmed = true;
      this.log('confirm', 'Қауіпті әрекет. Тағы бір рет РАСТАУ басыңыз.');
      return;
    }
    const phys = this.phys;
    const sl = this.sl;
    const self = this;
    items.forEach(function (p) {
      const to = +p.to;
      switch (p.actuator) {
        case 'p_nbi':
          self.manual(); phys.PnbiSet = to; sl.nbi.value = to; break;
        case 'p_icrf':
          self.manual(); phys.PicrSet = to; sl.icr.value = to; break;
        case 'p_ecrf':
          self.manual(); phys.PecrSet = to; sl.ecr.value = to; break;
        case 'gas':
          self.manual(); phys.gasSet = to; sl.gas.value = to; break;
        case 'ip':
          self.manual(); phys.IpSet = to; sl.ip.value = to; break;
        case 'bt':
          phys.Bt = to; sl.bt.value = to; break;
        case 'pellet':
          phys.injectPellet(); break;
        case 'elm':
          phys.forceELM(); break;
        case 'disrupt':
          phys.triggerDisruption('КЕҢЕСШІ РАСТАДЫ'); break;
        case 'mitigate':
          phys.mitigate(); break;
        default:
          break;
      }
    });
    this.syncLabels();
    this.pending = null;
    this.dangerArmed = false;
    $('ai-card').hidden = true;
    this.hideRail();
    this.clearAlarm();
    this.log('confirm', 'Расталды. Сетпойнттер жазылды.');
  };

  global.Advisor = Advisor;
  global.detectAlarms = detectAlarms;
})(window);
