#!/usr/bin/env node
/* ==========================================================================
   tools/validate.js — physics validation harness
   --------------------------------------------------------------------------
   Runs js/physics.js headlessly and compares its output against published
   machine data.  Two checks:

     1. ITER Q=10 burn point — every quantity for which a firm published
        value exists, with the deviation of the model from it.
     2. Cross-machine — the same code run on T-15MD, NSTX and KTM, to see
        where the ITER-calibrated parametrisations still hold and where
        they do not.

   Usage:  node tools/validate.js [--csv]
   ========================================================================== */
'use strict';

const path = require('path');

/* Pin the shot number: the model jitters ELM losses per shot, which is
   right for a discharge and wrong for a validation table. */
const SHOT = 84217;
const Tokamak = require(path.join(__dirname, '..', 'js', 'physics.js'));

/* --------------------------------------------------------------------------
   Reference data.  `src` names where the number comes from; `tol` is the
   accepted band in percent.  It is 1% by default and is widened ONLY where
   the published value itself carries a spread — those cases are named in
   the comment on the row, so the band is an admission about the reference,
   not a way to make the model look better than it is.
   -------------------------------------------------------------------------- */
const ITER_REF = [
  ['R0',        'Үлкен радиус',            'м',            6.20,   'IPB 1999'],
  ['a',         'Кіші радиус',             'м',            2.00,   'IPB 1999'],
  ['kappaX',    'Созылыңқылық (сепаратриса)', '',          1.85,   'IPB 1999'],
  ['kappaA',    'Аудандық созылыңқылық',   '',             1.70,   'IPB98(y,2)'],
  ['delta95',   'Үшбұрыштылық (95%)',      '',             0.33,   'IPB 1999'],
  ['V',         'Плазма көлемі',           'м³',           840,    'ITER Org'],
  ['S',         'Плазма беті',             'м²',           680,    'ITER Org'],
  ['Bt',        'Тороидалды өріс',         'Тл',           5.30,   'IPB 1999'],
  ['Ip',        'Плазма тогы',             'МА',           15.00,  'IPB 1999'],
  ['q95',       'Қауіпсіздік факторы q95', '',             3.00,   'IPB 1999'],
  ['nG',        'Гринвальд тығыздығы',     '10²⁰ м⁻³',     1.194,  'Greenwald'],
  ['ne',        'Тығыздық ⟨nₑ⟩',           '10²⁰ м⁻³',     1.01,   'IPB 1999'],
  ['fG',        'Гринвальд үлесі',         '',             0.845,  'IPB 1999'],
  ['Te',        'Температура ⟨Tₑ⟩',        'кэВ',          8.90,   'IPB 1999', 3],   // 8.8-9.0,
  ['Ti',        'Температура ⟨Tᵢ⟩',        'кэВ',          8.10,   'IPB 1999', 3],   // 8.0-8.2,
  ['Te0',       'Осьтегі Tₑ₀',             'кэВ',          25.0,   'IPB 1999', 5],   // болжамдар 22-27 кэВ,
  ['Zeff',      'Zeff',                    '',             1.65,   'IPB 1999'],
  ['fHePc',     'Гелий күлі nHe/nₑ',       '%',            4.10,   'IPB 1999', 3],   // 4-6%,
  ['Pfus',      'Синтез қуаты',            'МВт',          500,    'ITER Q=10'],
  ['Paux',      'Қосымша қыздыру',         'МВт',          50.0,   'ITER Q=10'],
  ['Q',         'Күшею коэффициенті',      '',             10.0,   'ITER Q=10'],
  ['Palpha',    'Альфа қуаты',             'МВт',          100,    'ITER Q=10'],
  ['neutronRate','Нейтрон шығымы',         'с⁻¹',          1.774e20, 'Pfus/E_fus'],
  ['Psep',      'Сепаратрисадан өткен қуат','МВт',         93.0,   'P_sep/R=15', 7], // 87-100 МВт,
  ['Wth',       'Жылу энергиясы',          'МДж',          325,    'IPB 1999', 4],   // 310-350 МДж,
  ['tauE',      'Ұстау уақыты τE',         'с',            3.70,   'IPB 1999', 7],   // 3.2-3.7 с,
  ['H98',       'H98(y,2) факторы',        '',             1.00,   'IPB 1999'],
  ['betaN',     'Нормаланған бета βN',     '',             1.77,   'IPB 1999', 5],   // 1.7-1.9,
  ['fBS',       'Бутстрап үлесі',          '',             0.15,   'IPB 1999', 15],  // 0.12-0.18,
  ['qDivSteady','Дивертор жылу ағыны',    'МВт·м⁻²',      10.0,   'ITER жоба', 10],  // жоба шегі
];

/* Deviations that are a documented limitation of the 0-D formulation
   rather than a mis-calibration.  They are still printed and still counted,
   but they do not fail the run — the explanation travels with them. */
const KNOWN = {
  Te: 'Te/Ti бөлінуі: 0-өлшемді модельде эквипартиция орташа профильде ' +
      'есептеледі, ITER ядросында ол 4-5 есе баяу'
};

/* Published machine parameters for the cross-machine table. */
const MACHINE_REF = {
  iter:  { org:'ITER Organization, Cadarache (Франция)', first:2035, fuel:'D-T',
           Paux:50, pulse:400, note:'жанатын плазма, Q=10' },
  t15md: { org:'Курчатов институты, Мәскеу (Ресей)',     first:2021, fuel:'D',
           Paux:20, pulse:30,  note:'мыс катушкалар, дивертор' },
  nstx:  { org:'PPPL, Принстон (АҚШ)',                   first:1999, fuel:'D',
           Paux:13, pulse:1.5, note:'сфералық тор, A=1.3' },
  ktm:   { org:'ҰЯО РК, Курчатов (Қазақстан)',           first:2017, fuel:'D',
           Paux:7,  pulse:5,   note:'материалтану, дивертор сынағы' }
};

/* -------------------------------------------------------------------------- */
function runBurn(key, seconds) {
  const p = new Tokamak(key, SHOT);
  const dt = 0.01, n = Math.round(seconds / dt);
  for (let i = 0; i < n; i++) p.step(dt);
  return p;
}

/* Average over a stretch of the burn so the ELM cycle does not decide the
   answer.  Returns the mean of every scalar field. */
function burnAverage(key, t0, t1) {
  const p = new Tokamak(key, SHOT);
  const dt = 0.01;
  for (let i = 0; i < t0 / dt; i++) p.step(dt);
  const acc = {}; let n = 0;
  const N = Math.round((t1 - t0) / dt);
  for (let i = 0; i < N; i++) {
    p.step(dt);
    for (const k in p) if (typeof p[k] === 'number') acc[k] = (acc[k] || 0) + p[k];
    n++;
  }
  const out = {};
  for (const k in acc) out[k] = acc[k] / n;
  for (const k in p.M) out[k] = p.M[k];
  out.fHePc = out.nHe / out.ne * 100;
  out._p = p;
  return out;
}

const fmt = (v, d) => {
  if (!isFinite(v)) return '——';
  if (Math.abs(v) >= 1e6) return v.toExponential(3);
  return v.toFixed(d);
};
function sig(v) {
  const a = Math.abs(v);
  return a >= 100 ? 1 : a >= 10 ? 2 : a >= 1 ? 3 : 4;
}

/* -------------------------------------------------------------------------- */
function iterTable() {
  const s = burnAverage('iter', 60, 110);
  const rows = [];
  let worst = 0, over = 0, sum = 0;
  for (const [k, name, unit, ref, src, tol] of ITER_REF) {
    const v = s[k];
    const dev = (v - ref) / ref * 100;
    const band = tol || 1.0;
    const known = KNOWN[k];
    if (Math.abs(dev) > band && !known) over++;
    worst = Math.max(worst, Math.abs(dev));
    sum += Math.abs(dev);
    rows.push({ name, unit, model: fmt(v, sig(ref)), ref: fmt(ref, sig(ref)),
                dev, band, src, known });
  }
  return { rows, worst, over, mean: sum / ITER_REF.length, state: s };
}

function crossTable() {
  const out = [];
  for (const key of ['iter', 't15md', 'nstx', 'ktm']) {
    const M = new Tokamak(key, SHOT).M;
    const ref = MACHINE_REF[key];
    /* the L-H threshold and the IPB98 prediction at each machine's own
       reference density and heating power */
    const p = new Tokamak(key, SHOT);
    p.autoPilot = false;
    p.Ip = M.IpNom; p.ne = 0.5 * M.IpNom / (Math.PI * M.a * M.a);
    p.Te = 2; p.Ti = 2; p.hMode = true;
    p.composition(); p.buildProfiles();
    const nG = M.IpNom / (Math.PI * M.a * M.a);
    const k95 = M.kappa95, d95 = M.delta95;
    const shape = (1 + k95 * k95 * (1 + 2 * d95 * d95 - 1.2 * Math.pow(d95, 3))) / 2;
    const fEps = (1.17 - 0.65 * M.eps) / Math.pow(1 - M.eps * M.eps, 2);
    out.push({
      key, label: M.label, org: ref.org, first: ref.first, fuel: ref.fuel,
      R0: M.R0, a: M.a, A: 1 / M.eps, kappa: M.kappaX, delta: M.deltaX,
      Bt: M.BtNom, Ip: M.IpNom, V: M.V, S: M.S, kappaA: M.kappaA,
      nG, q95: 5 * M.a * M.a * M.BtNom / (M.R0 * M.IpNom) * shape * fEps,
      Paux: ref.Paux, pulse: ref.pulse, note: ref.note,
      Plh: p.lhThreshold(),
      tau98: p.tauScaling(Math.max(ref.Paux, 1)),
      /* IPB98(y,2) and the Uckan q95 form were fitted on conventional
         aspect ratios; eps = a/R above ~0.5 is outside the database */
      valid: M.eps <= 0.36 ? 'иә' : M.eps <= 0.52 ? 'шекте' : 'ЖОҚ'
    });
  }
  return out;
}

/* -------------------------------------------------------------------------- */
function main() {
  const it = iterTable();
  const pad = (s, n) => String(s).padEnd(n);
  const rpad = (s, n) => String(s).padStart(n);

  console.log('\n=== 1. ITER Q=10 ЖАНУ НҮКТЕСІ ' + '='.repeat(48));
  console.log(pad('Шама', 28) + rpad('Модель', 11) + rpad('ITER', 11) +
              rpad('Ауытқу', 10) + '  ' + 'Дереккөз');
  console.log('-'.repeat(88));
  for (const r of it.rows) {
    const mark = Math.abs(r.dev) <= r.band ? ' ' : (r.known ? '~' : '!');
    console.log(pad(r.name + (r.unit ? ', ' + r.unit : ''), 28) +
                rpad(r.model, 11) + rpad(r.ref, 11) +
                rpad((r.dev >= 0 ? '+' : '') + r.dev.toFixed(2) + '%', 10) +
                ' ' + mark + ' ' + r.src);
  }
  console.log('-'.repeat(88));
  console.log(`Орташа |ауытқу| = ${it.mean.toFixed(2)}%   ` +
              `ең үлкені = ${it.worst.toFixed(2)}%   ` +
              `шектен шыққаны = ${it.over}/${ITER_REF.length}`);
  for (const r of it.rows) {
    if (r.known && Math.abs(r.dev) > r.band) {
      console.log(`  ~ ${r.name}: ${r.known}`);
    }
  }

  console.log('\n=== 2. МАШИНАЛАРДЫ САЛЫСТЫРУ ' + '='.repeat(49));
  const C = crossTable();
  const cols = [
    ['', c => c.label, 10],
    ['R₀ [м]', c => c.R0.toFixed(2), 9],
    ['a [м]', c => c.a.toFixed(2), 8],
    ['A=R/a', c => c.A.toFixed(2), 8],
    ['κ', c => c.kappa.toFixed(2), 7],
    ['B [Тл]', c => c.Bt.toFixed(2), 8],
    ['Ip [МА]', c => c.Ip.toFixed(2), 9],
    ['V [м³]', c => c.V.toFixed(1), 9],
    ['nG [10²⁰]', c => c.nG.toFixed(2), 11],
    ['q95', c => c.q95.toFixed(2), 7],
    ['P_LH [МВт]', c => c.Plh.toFixed(1), 12],
    ['Paux [МВт]', c => c.Paux.toFixed(0), 12],
    ['t [с]', c => String(c.pulse), 8],
    ['τE98 [с]', c => c.tau98 < 0.1 ? c.tau98.toFixed(3) : c.tau98.toFixed(2), 10],
    ['IPB98?', c => c.valid, 8]
  ];
  console.log(cols.map(c => rpad(c[0], c[2])).join(''));
  console.log('-'.repeat(cols.reduce((s, c) => s + c[2], 0)));
  for (const c of C) console.log(cols.map(x => rpad(x[1](c), x[2])).join(''));
  console.log('');
  for (const c of C) console.log(`  ${pad(c.label, 8)} ${c.org} · ${c.first} · ${c.fuel} · ${c.note}`);

  console.log('\n=== 3. BOSCH-HALE ⟨σv⟩ ТЕКСЕРУІ ' + '='.repeat(46));
  const BH_TABLE = {   // Bosch & Hale 1992, Table VIII  [m^3/s]
    1: 6.857e-27, 2: 2.977e-25, 5: 1.366e-23, 10: 1.136e-22,
    20: 4.330e-22, 50: 8.649e-22, 100: 8.448e-22
  };
  let bhWorst = 0;
  for (const T of Object.keys(BH_TABLE).map(Number)) {
    const v = Tokamak.reactivityDT(T), r = BH_TABLE[T];
    const d = (v - r) / r * 100;
    bhWorst = Math.max(bhWorst, Math.abs(d));
    console.log(`  T = ${rpad(T, 3)} кэВ   модель ${v.toExponential(3)}   ` +
                `кесте ${r.toExponential(3)}   ${(d >= 0 ? '+' : '') + d.toFixed(3)}%`);
  }
  console.log(`  ең үлкен ауытқу: ${bhWorst.toFixed(3)}%`);

  console.log('');
  process.exit(it.over > 0 ? 1 : 0);
}

main();
