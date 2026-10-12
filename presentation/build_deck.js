/* SINTEZ -- IFARP-2026 conference deck (15 slides, Kazakh).
 *   npm install pptxgenjs && python3 make_formulas.py && node build_deck.js
 * Every number comes from a command that was run on this repository:
 *   node tools/validate.js, python -m tokamak.validation, ml.train_transport_surrogate,
 *   paper/results.json (control experiments).  Screenshots are of the running site. */
const pptxgen = require("pptxgenjs");
const path = require("path");
const { applyTheme } = require("/root/.claude/skills/synced/1a4d98e9-271c-4eeb-a699-0114d3e2ea33_962935ee-302f-405f-adab-150d0543d0ac/pptx/scripts/apply_theme.js");

const A = (f) => path.join(__dirname, "assets", f);
const OUT = path.join(__dirname, "SINTEZ_IFARP-2026.pptx");

const THEME = {
  name: "SINTEZ Plasma",
  headFontFace: "Calibri",
  bodyFontFace: "Calibri",
  // dark deck: lt1 is the (dark) page, dk1 the (light) text
  colors: {
    dk1: "EEF1FF", lt1: "0B0F1E", dk2: "9AA6D0", lt2: "141B36",
    accent1: "3EE0F0", accent2: "E94CC8", accent3: "FFB24A",
    accent4: "5BE3A0", accent5: "7C8CFF", accent6: "FF6B6B",
    hlink: "3EE0F0", folHlink: "9AA6D0",
  },
};
const H = THEME.colors; // hex values, for chart options that accept hex only

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "SINTEZ: интерактивті токамак симуляторы";
pres.author = "Уәлихан А.Ш., Ермекұлы Б.";
pres.company = "Қожа Ахмет Яссауи атындағы ХҚТУ";
const C = pres.SchemeColor;

for (const t of ["Кіріспе", "Жүйе", "Валидация", "AI", "Қорытынды"]) pres.addSection({ title: t });

/* ---------------- layouts ---------------- */
pres.defineSlideMaster({
  title: "TITLE",
  background: { color: C.background1 },
  objects: [],
  slideNumber: undefined,
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  objects: [
    { text: { text: "SINTEZ · IFARP-2026", options: { x: 0.5, y: 5.25, w: 4, h: 0.28, fontSize: 10, color: C.text2, margin: 0, isTextBox: true } } },
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.25, w: 9, h: 0.75, fontSize: 30, bold: true, color: C.text1, align: "left", valign: "middle", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 9.0, y: 5.25, w: 0.5, h: 0.28, fontSize: 10, color: C.text2, align: "right" },
});

/* ---------------- helpers ---------------- */
const card = (s, x, y, w, h, name, tint) =>
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.08, objectName: name || "card",
    fill: { color: tint || C.background2 }, line: { color: C.text2, width: 0.5, transparency: 70 },
  });

const badge = (s, x, y, n, color, d = 0.34) => {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: color || C.accent1 }, line: { color: color || C.accent1, width: 0 }, objectName: "badge " + n });
  s.addText(String(n), { x, y, w: d, h: d, fontSize: 13, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
};

// "X_ab" -> X with subscript run "ab" (plain-text formulas in labels)
const SUB = /_([A-Za-z0-9А-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі]+)/;
const sr = (str) => {
  if (typeof str !== "string" || !SUB.test(str)) return str;
  const parts = str.split(new RegExp(SUB.source, "g"));
  const runs = [];
  parts.forEach((p, i) => { if (p !== "") runs.push(i % 2 ? { text: p, options: { subscript: true } } : { text: p }); });
  return runs;
};
const T = (s, text, o) => s.addText(sr(text), Object.assign({ fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true }, o));

const img = (s, file, x, y, w, h, alt) => s.addImage({ path: A(file), x, y, w, h, altText: alt || file });

// image with known pixel size scaled to a given width
const fimg = (s, file, x, y, w, px) => {
  const h = (w * px[1]) / px[0];
  s.addImage({ path: A(file), x, y, w, h, altText: "формула " + file });
  return h;
};
const FPX = { f_balance: [1703, 216], f_circuit: [1315, 216], f_fusion: [2109, 213], f_gae: [1693, 530], f_gs: [1831, 195], f_ipb: [2201, 445], f_loss: [1811, 336], f_mlp: [2968, 419], f_ppo: [2934, 469], f_qp: [3130, 471], f_rlt: [804, 227], f_target: [2868, 422], f_transport: [1691, 227], f_troyon: [1015, 241] };
const formula = (s, name, x, y, w) => fimg(s, name + ".png", x, y, w, FPX[name]);

const arrow = (s, x1, y1, x2, y2, color) => {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), flipH: x2 < x1, flipV: y2 < y1,
    line: { color: color || C.text2, width: 1.75, endArrowType: "triangle" }, objectName: "arrow" });
};

const title = (s, t) => s.addText(t, { placeholder: "title" });

/* =================== 1. title =================== */
let s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Кіріспе" });
img(s, "tokamak_crop.jpg", 5.55, 0.45, 4.05, 3.66, "SINTEZ: токамактың 3D қимасы (сайттан скриншот)");
T(s, "IFARP-2026", { x: 0.6, y: 0.7, w: 4.5, h: 0.3, fontSize: 14, bold: true, color: C.accent1, charSpacing: 4 });
T(s, "SINTEZ", { x: 0.6, y: 1.05, w: 4.8, h: 0.95, fontSize: 60, bold: true, color: C.text1, fontFace: "Calibri" });
T(s, "Токамак реакторындағы D–T термоядролық синтез процесін нақты физикалық модельдер арқылы есептейтін интерактивті симулятор", { x: 0.6, y: 2.1, w: 4.7, h: 1.5, fontSize: 18, color: C.text1 });
T(s, "Уәлихан А.Ш., Ермекұлы Б.", { x: 0.6, y: 4.2, w: 8.5, h: 0.32, fontSize: 16, bold: true, color: C.accent3 });
T(s, "Қожа Ахмет Яссауи атындағы Халықаралық қазақ-түрік университеті, Түркістан", { x: 0.6, y: 4.58, w: 8.8, h: 0.3, fontSize: 14, color: C.text2 });
s.addNotes("Сәлеметсіздер ме! Біз SINTEZ жобасын таныстырамыз: бұл токамактағы D–T синтез процесін нақты физика теңдеулерімен нақты уақытта есептейтін интерактивті симулятор. Жоба Қазақстан Республикасында авторлық құқықпен тіркелген (№ 78976).");

/* =================== 2. problem & goal =================== */
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Кіріспе" });
title(s, "Мәселе және мақсат");
card(s, 0.5, 1.2, 5.3, 1.75, "мәселе");
T(s, "МӘСЕЛЕ", { x: 0.75, y: 1.33, w: 4.8, h: 0.25, fontSize: 12, bold: true, color: C.accent2, charSpacing: 3 });
T(s, "Оқу және көрсету құралдарының көбі — алдын ала жазылған анимация. Олар жұмыс режимі плазма параметрлеріне қалай әсер ететінін көрсетпейді.", { x: 0.75, y: 1.65, w: 4.8, h: 1.25, fontSize: 17 });
card(s, 0.5, 3.15, 5.3, 1.9, "мақсат");
T(s, "МАҚСАТ", { x: 0.75, y: 3.28, w: 4.8, h: 0.25, fontSize: 12, bold: true, color: C.accent1, charSpacing: 3 });
T(s, "Плазма күйі магниттік ұстау физикасының теңдеулерін нақты уақытта шешу арқылы алынатын интерактивті симулятор жасау.", { x: 0.75, y: 3.6, w: 4.8, h: 1.4, fontSize: 17 });
const stat = (y, big, lab, col) => {
  card(s, 6.1, y, 3.4, 1.18, "көрсеткіш " + big);
  T(s, big, { x: 6.3, y: y + 0.1, w: 3.0, h: 0.62, fontSize: 36, bold: true, color: col, valign: "middle" });
  T(s, lab, { x: 6.3, y: y + 0.74, w: 3.0, h: 0.34, fontSize: 14, color: C.text2 });
};
stat(1.2, "12", "құрылғы бір модельде, KTM қоса", C.accent1);
stat(2.54, "40+", "диагностикалық панель", C.accent2);
stat(3.88, "60 к/с", "WebGL 2.0, 3D реактор", C.accent3);
s.addNotes("Проблема: көп оқу құралдары — дайын анимация. Бізде плазма күйі әр кадрда физика теңдеулерінен есептеледі, сондықтан оператор қыздыруды, тоқты, өрісті өзгертіп, нәтижесін бірден көреді.");

/* =================== 3. architecture =================== */
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Жүйе" });
title(s, "Жүйенің архитектурасы");
// client
card(s, 0.5, 1.2, 2.7, 2.95, "клиент");
T(s, "БРАУЗЕР", { x: 0.7, y: 1.32, w: 2.3, h: 0.25, fontSize: 12, bold: true, color: C.accent1, charSpacing: 3 });
T(s, [
  { text: "JavaScript · WebGL 2.0", options: { bullet: true, breakLine: true } },
  { text: "3D реактор, 60 к/с", options: { bullet: true, breakLine: true } },
  { text: "40+ диагностика панелі", options: { bullet: true, breakLine: true } },
  { text: "0D модель (physics.js)", options: { bullet: true, breakLine: true } },
  { text: "Басқару элементтері", options: { bullet: true } },
], { x: 0.7, y: 1.65, w: 2.35, h: 2.4, fontSize: 14, paraSpaceAfter: 5 });
// api
card(s, 3.95, 1.2, 1.95, 2.95, "api");
T(s, "API", { x: 4.1, y: 1.32, w: 1.65, h: 0.25, fontSize: 12, bold: true, color: C.accent3, charSpacing: 3 });
T(s, "FastAPI", { x: 4.1, y: 1.7, w: 1.65, h: 0.4, fontSize: 20, bold: true, align: "left" });
T(s, "Python 3.11\nHTTP / JSON\n/models, /ml, /control", { x: 4.1, y: 2.2, w: 1.7, h: 1.8, fontSize: 14, color: C.text2 });
arrow(s, 3.2, 2.67, 3.95, 2.67, C.accent1);
arrow(s, 3.95, 3.0, 3.2, 3.0, C.accent1);
// backend stack
const box = (y, h, head, body, col) => {
  card(s, 6.45, y, 3.05, h, "блок " + head);
  T(s, head, { x: 6.62, y: y + 0.1, w: 2.75, h: 0.28, fontSize: 14, bold: true, color: col });
  T(s, body, { x: 6.62, y: y + 0.42, w: 2.75, h: h - 0.48, fontSize: 13, color: C.text1 });
};
box(1.2, 1.1, "Физика пакеті", "0D · 1.5D · МГД · дизрупция · SOL/дивертор · отын циклі", C.accent2);
box(2.42, 0.85, "ML модульдері", "суррогат, үзіліс болжау, тепе-теңдік", C.accent4);
box(3.39, 0.76, "Магниттік басқару", "1 кГц ко-симуляция, PPO + QP", C.accent5);
arrow(s, 5.9, 2.67, 6.45, 1.75, C.accent3);
arrow(s, 5.9, 2.75, 6.45, 2.85, C.accent3);
arrow(s, 5.9, 2.85, 6.45, 3.77, C.accent3);
// deployment strip
card(s, 0.5, 4.38, 9.0, 0.7, "орналастыру");
T(s, [
  { text: "Railway: ", options: { bold: true, color: C.accent3 } },
  { text: "бір процесс (uvicorn) сайтты да, API-ды да береді.  Интерфейс пен құжаттама — толық қазақ тілінде." },
], { x: 0.7, y: 4.38, w: 8.6, h: 0.7, fontSize: 14, valign: "middle" });
s.addNotes("Клиент JavaScript пен WebGL 2.0-де жұмыс істейді. Физика пакеті Python-да, FastAPI арқылы HTTP/JSON беріледі. ML және басқару модульдері сол API-ға қосылған. Бәрі Railway-да бір процесс ретінде жұмыс істейді.");

/* =================== 4. physics models =================== */
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Жүйе" });
title(s, "Физикалық модельдер");
const cards4 = [
  ["0D модель", "Энергия және бөлшек балансы, Bosch–Hale ⟨σv⟩, тежеу және синхротрон шығыны, гелий күлі", C.accent1],
  ["1.5D модель", "Град–Шафранов тепе-теңдігі, жылу және бөлшек тасымалы, педестал, ток диффузиясы", C.accent2],
  ["МГД тұрақтылығы", "Тройон шегі, RWM, баллондық шекара, NTM, Кадомцев, ELM циклі", C.accent3],
  ["Дизрупциялар", "Термиялық және ток өшуі, қашқын электрондар, гало тоқтары, VDE", C.accent6],
  ["Шеткі қабат", "Екі нүктелі дивертор моделі, Eich λ_q, қоспаны себу", C.accent4],
  ["Отын циклі", "Тритий өсіру, нейтрон жүктемесі, таза электр қуаты, Q_eng", C.accent5],
];
cards4.forEach((c, i) => {
  const x = 0.5 + (i % 3) * 3.05, y = 1.2 + Math.floor(i / 3) * 1.98;
  card(s, x, y, 2.9, 1.82, "карта " + c[0]);
  badge(s, x + 0.18, y + 0.16, i + 1, c[2]);
  T(s, c[0], { x: x + 0.62, y: y + 0.16, w: 2.2, h: 0.34, fontSize: 16, bold: true, valign: "middle" });
  T(s, c[1], { x: x + 0.18, y: y + 0.62, w: 2.56, h: 1.15, fontSize: 14.5, color: C.text1 });
});
s.addNotes("Модель екі деңгейде: 0-өлшемді баланс және 1.5-өлшемді радиалды профильдер. Үстіне МГД тұрақтылығы, дизрупциялар, шеткі қабат пен дивертор, отын циклі қосылған.");

/* =================== 5. formulas =================== */
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Жүйе" });
title(s, "Негізгі теңдеулер");
const F5 = [
  ["Энергия балансы (0D)", "f_balance", 3.6],
  ["Синтез қуаты", "f_fusion", 4.1],
  ["Ұстау уақыты IPB98(y,2)", "f_ipb", 3.5],
  ["Жылу тасымалы (1D)", "f_transport", 4.1],
  ["Град–Шафранов теңдеуі", "f_gs", 3.9],
  ["Тройон шегі", "f_troyon", 2.0],
];
F5.forEach((f, i) => {
  const x = 0.5 + (i % 2) * 4.6, y = 1.2 + Math.floor(i / 2) * 1.3;
  card(s, x, y, 4.4, 1.2, "формула " + f[0]);
  T(s, f[0], { x: x + 0.2, y: y + 0.08, w: 4.0, h: 0.25, fontSize: 12, color: C.accent1, bold: true });
  const w = f[2], h = (w * FPX[f[1]][1]) / FPX[f[1]][0];
  formula(s, f[1], x + (4.4 - w) / 2, y + 0.38 + (0.78 - h) / 2, w);
});
s.addNotes("Негізгі теңдеулер: энергия балансы, синтез қуаты, IPB98(y,2) ұстау уақыты скейлингі, 1D тасымал теңдеуі, Град–Шафранов тепе-теңдігі және Тройон шегі. Бесеуі бірге жұмыс істейтін бір модельді құрайды.");

/* =================== 6. 3D interface =================== */
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Жүйе" });
title(s, "Интерфейс: 3D токамак");
img(s, "anatomy_crop.jpg", 0.5, 1.2, 5.9, 3.33, "Сайттан скриншот: токамактың негізгі тораптары");
T(s, "Сайттан алынған скриншот", { x: 0.5, y: 4.62, w: 5.9, h: 0.25, fontSize: 11, color: C.text2 });
const b6 = [
  ["Белгіленген 3D модель", "плазма камерасы, бірінші қабырға, магниттер, криостат"],
  ["Плазма ядросына ұшу", "7 секундтық кіру кезінде модель жұмыс істеп тұрады"],
  ["Қазақ тілінде", "интерфейс пен түсіндірмелер толық қазақша"],
];
b6.forEach((b, i) => {
  const y = 1.2 + i * 1.12;
  card(s, 6.65, y, 2.85, 1.02, "тезис " + (i + 1));
  T(s, b[0], { x: 6.8, y: y + 0.08, w: 2.6, h: 0.3, fontSize: 14, bold: true, color: C.accent1 });
  T(s, b[1], { x: 6.8, y: y + 0.4, w: 2.6, h: 0.58, fontSize: 12.5, color: C.text1 });
});
s.addNotes("Бұл сайттың нақты скриншоты. Тораптар нөмірленген, әрқайсысы туралы түсіндірме бар. Қолданушы реакторға кіріп, плазманың ішін көре алады.");

/* =================== 7. control room =================== */
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Жүйе" });
title(s, "Басқару кабинасы");
img(s, "hud_full.jpg", 0.5, 1.15, 6.3, 3.94, "Сайттан скриншот: басқару кабинасы, ITER класты режим");
const b7 = [
  ["Тарих графиктері", "T_i, T_e, n_e, P_син, τ_E, сәулелену шығыны"],
  ["Қуат және жылу ағыны", "қуат балансы, дивертор жылу картасы, МГД осциллографы"],
  ["Оператор басқаруы", "NBI, ICRH, ECRH, газ, I_p, B_t, пеллет, ELM, дизрупция"],
];
b7.forEach((b, i) => {
  const y = 1.15 + i * 1.08;
  card(s, 7.0, y, 2.5, 1.0, "панель " + (i + 1));
  T(s, b[0], { x: 7.13, y: y + 0.07, w: 2.3, h: 0.28, fontSize: 13.5, bold: true, color: C.accent2 });
  T(s, b[1], { x: 7.13, y: y + 0.37, w: 2.3, h: 0.6, fontSize: 12, color: C.text1 });
});
T(s, "Скриншотта: Q = 9.70, P_син = 485 МВт (ITER класы, тұрақты жану)", { x: 7.0, y: 4.42, w: 2.5, h: 0.68, fontSize: 12, color: C.accent3 });
s.addNotes("Басқару кабинасында 40-тан астам диагностикалық панель. Скриншотта ITER класты режим: Q шамамен 9.7, синтез қуаты 485 МВт. Оператор қыздыруды, газды, тоқты және өрісті өзгерте алады, пеллет пен ELM импульсін, тіпті дизрупцияны іске қоса алады.");

/* =================== 8. devices =================== */
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Жүйе" });
title(s, "12 құрылғы — бір модельде");
const dev = [["EU-DEMO", 17.75], ["ITER", 15.0], ["SPARC", 8.7], ["JET", 4.8], ["DIII-D", 2.0], ["KSTAR", 2.0], ["T-15MD", 2.0], ["NSTX-U", 2.0], ["MAST-U", 2.0], ["EAST", 1.0], ["NSTX", 1.0], ["KTM", 0.75]];
s.addChart(pres.charts.BAR, [{ name: "Плазма тогы, МА", labels: dev.map((d) => d[0]), values: dev.map((d) => d[1]) }], {
  x: 0.5, y: 1.15, w: 5.9, h: 3.95, barDir: "bar", chartColors: dev.map((d) => (d[0] === "KTM" ? H.accent3 : H.accent1)),
  catAxisOrientation: "maxMin", showValue: true, dataLabelColor: H.dk1, dataLabelFontSize: 11, dataLabelFontFace: "+mn-lt", dataLabelFormatCode: "0.0#",
  catAxisLabelColor: H.dk1, catAxisLabelFontSize: 12, catAxisLabelFontFace: "+mn-lt", valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
  showLegend: false, showTitle: true, title: "Номиналды плазма тогы, МА", titleColor: H.dk2, titleFontSize: 13, titleFontFace: "+mn-lt", barGapWidthPct: 40,
});
card(s, 6.65, 1.2, 2.85, 3.85, "KTM");
T(s, "KTM", { x: 6.85, y: 1.32, w: 2.45, h: 0.45, fontSize: 26, bold: true, color: C.accent3 });
T(s, "Қазақстан материалтану токамағы (ҚР ҰЯО, Курчатов)", { x: 6.85, y: 1.8, w: 2.45, h: 0.75, fontSize: 13, color: C.text1 });
[["R₀ / a", "0.86 / 0.43 м"], ["B₀", "1.0 Тл"], ["I_p", "0.75 МА"]].forEach((r, i) => {
  T(s, r[0], { x: 6.85, y: 2.75 + i * 0.62, w: 1.0, h: 0.5, fontSize: 14, color: C.text2, valign: "middle" });
  T(s, r[1], { x: 7.75, y: 2.75 + i * 0.62, w: 1.65, h: 0.5, fontSize: 15, bold: true, valign: "middle" });
});
T(s, "Салыстыру: бір модель, бір интерфейс", { x: 6.85, y: 4.55, w: 2.5, h: 0.45, fontSize: 12, color: C.text2 });
s.addNotes("Симуляторда 12 құрылғы: ITER, EU-DEMO, SPARC, JET, DIII-D, EAST, KSTAR, T-15MD, NSTX, NSTX-U, MAST-U және Қазақстан материалтану токамағы KTM. Барлығы бір модельде салыстырылады. Параметрлер жобаның machines.py файлынан алынған.");

/* =================== 9. validation =================== */
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Валидация" });
title(s, "Валидация: ITER Q = 10");
const vst = [["30", "шама салыстырылды", C.accent1], ["0.93 %", "орташа ауытқу", C.accent4], ["5.52 %", "ең үлкен ауытқу", C.accent3], ["79 / 79", "компонент тексеруі өтті", C.accent2]];
vst.forEach((v, i) => {
  const x = 0.5 + (i % 2) * 1.85, y = 1.2 + Math.floor(i / 2) * 1.6;
  card(s, x, y, 1.75, 1.5, "көрсеткіш " + v[0]);
  T(s, v[0], { x: x + 0.1, y: y + 0.2, w: 1.55, h: 0.62, fontSize: 28, bold: true, color: v[2], align: "center", valign: "middle" });
  T(s, v[1], { x: x + 0.1, y: y + 0.88, w: 1.55, h: 0.55, fontSize: 12.5, color: C.text2, align: "center" });
});
T(s, "Валидацияланбағандар тізімі ашық: validation.py", { x: 0.5, y: 4.5, w: 3.6, h: 0.55, fontSize: 12, color: C.accent3 });
const hd = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.accent1 }, fontSize: 13, align: "center" } });
const row = (n, a, b, d, big) => [
  { text: sr(n), options: { align: "left" } }, { text: a, options: { align: "right" } }, { text: b, options: { align: "right" } },
  { text: d, options: { align: "right", bold: !!big, color: big ? C.accent3 : C.text1 } },
];
s.addTable([
  [hd("Шама"), hd("Модель"), hd("ITER"), hd("Ауытқу")],
  row("Синтез қуаты, МВт", "498.0", "500", "−0.41 %"),
  row("Күшею коэфф. Q", "9.96", "10.0", "−0.41 %"),
  row("τ_E, с", "3.683", "3.700", "−0.46 %"),
  row("Жылу энергиясы, МДж", "317.9", "325.0", "−2.18 %"),
  row("β_N", "1.698", "1.770", "−4.05 %"),
  row("⟨T_e⟩, кэВ", "8.408", "8.900", "−5.52 %", true),
], { x: 4.3, y: 1.2, w: 5.2, colW: [2.1, 1.0, 1.0, 1.1], rowH: 0.44, fontSize: 13, color: C.text1, fill: { color: C.background2 }, border: { type: "solid", color: C.text2, pt: 0.5 }, valign: "middle", margin: [0, 0.08, 0, 0.08] });
T(s, "Эталон: ITER Physics Basis (1999). Ең үлкен ауытқу — 0D модельде электрон мен ион температурасының бөлінбеуінен.", { x: 4.3, y: 4.45, w: 5.2, h: 0.6, fontSize: 12, color: C.text2 });
s.addNotes("Валидация автоматты: 30 шама ITER Q=10 жобалық режимімен салыстырылды, орташа ауытқу 0.93 %, ең үлкені 5.52 % (электрон температурасы), толерантттан шыққаны жоқ. Python жағында 79 компонент тексеруі өтті. Валидацияланбаған тұстардың тізімі ашық жазылған.");

/* =================== 10. TORAX comparison =================== */
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "AI" });
title(s, "Google DeepMind TORAX-пен салыстыру");
const th = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.accent1 }, fontSize: 13 } });
const lab = (t) => ({ text: t, options: { bold: true, color: C.accent1, fontSize: 12 } });
const cl = (t) => ({ text: sr(t), options: { fontSize: 12 } });
s.addTable([
  [th(""), th("TORAX (Google DeepMind)"), th("SINTEZ")],
  [lab("Мақсаты"), cl("Өзек тасымалының жылдам симуляторы: пульс және контроллер жобалау"), cl("Интерактивті 0D + 1.5D симулятор: оқыту, салыстыру, жылдам бағалау")],
  [lab("Платформа"), cl("Python + JAX; автоматты дифференциалдау"), cl("Python (NumPy/SciPy), FastAPI; JS/WebGL клиент")],
  [lab("Теңдеулер"), cl("Ион/электрон жылуы, бөлшек, ток диффузиясы; Newton–Raphson, оңтайландыру"), cl("Te/Ti/ne тасымалы, ток диффузиясы + МГД, дизрупция, SOL, отын циклі")],
  [lab("Суррогат"), cl("QLKNN — QuaLiKiz кодының суррогаты (~10⁴ есе жылдам)"), cl("MLP 11→128→64→1; өз аналитикалық заңдарымыздан оқытылған")],
  [lab("Валидация"), cl("RAPTOR кодымен салыстырылған"), cl("ITER Q=10: 30 шама, орташа 0.93 %")],
  [lab("Басқару"), cl("Градиентті оңтайландыру"), cl("1 кГц тұйық контур: PPO + QP-сүзгі")],
], { x: 0.5, y: 1.15, w: 9.0, colW: [1.35, 3.85, 3.8], rowH: [0.38, 0.52, 0.42, 0.52, 0.52, 0.42, 0.42], color: C.text1, fill: { color: C.background2 }, border: { type: "solid", color: C.text2, pt: 0.5 }, valign: "middle", margin: [0, 0.1, 0, 0.1] });
card(s, 0.5, 4.5, 9.0, 0.6, "ескерту", C.background2);
T(s, [
  { text: "Адал ескерту: ", options: { bold: true, color: C.accent3 } },
  { text: "біздің суррогат QLKNN-нің баламасы емес — ол гирокинетикаға емес, өз тасымал заңдарымызға оқытылған, сондықтан жылдамдық бермейді (122 vs 148 мкс)." },
], { x: 0.7, y: 4.5, w: 8.6, h: 0.6, fontSize: 12.5, valign: "middle" });
s.addNotes("TORAX — Google DeepMind-тың ашық коды: JAX негізінде, дифференциалданатын өзек тасымалы симуляторы, QLKNN нейрожелілік суррогатымен және RAPTOR-мен салыстырылған. Біздің жұмыс басқа міндетті шешеді: интерактивті көпфизикалы симулятор. Біз TORAX-стиліндегі суррогатты жасадық, бірақ ол QuaLiKiz-ке емес, өз заңдарымызға оқытылған — сондықтан жылдамдық бермейді. Мұны ашық айтамыз. Дереккөздер: Citrin et al., arXiv:2406.06718; github.com/google-deepmind/torax.");

/* =================== 11. training =================== */
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "AI" });
title(s, "Модельдерді оқыту және формулалар");
// left: surrogate
card(s, 0.5, 1.15, 4.5, 3.95, "суррогат");
T(s, "1 · Тасымал суррогаты (MLP)", { x: 0.7, y: 1.25, w: 4.1, h: 0.32, fontSize: 15, bold: true, color: C.accent4 });
const chip = (x, y, w, t, col) => {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.42, rectRadius: 0.06, fill: { color: C.background1 }, line: { color: col, width: 1 }, objectName: "chip " + t });
  T(s, t, { x, y, w, h: 0.42, fontSize: 11.5, color: C.text1, align: "center", valign: "middle" });
};
chip(0.7, 1.65, 1.3, "65 000 үлгі", C.accent4);
chip(2.1, 1.65, 1.2, "Нормалау", C.accent4);
chip(3.4, 1.65, 1.4, "MLP + Adam", C.accent4);
formula(s, "f_target", 0.7, 2.25, 4.1);
formula(s, "f_mlp", 0.7, 2.95, 4.1);
formula(s, "f_loss", 0.95, 3.7, 2.3);
T(s, [
  { text: "R² (тест) = 0.943", options: { bold: true, color: C.accent4, breakLine: true } },
  { text: "MAE = 0.76 м²/с", options: { color: C.text1 } },
], { x: 3.3, y: 3.72, w: 1.6, h: 0.6, fontSize: 13 });
T(s, "9 857 параметр; инференс — таза NumPy; χ-ны шектеу: 0.05–15 м²/с", { x: 0.7, y: 4.45, w: 4.1, h: 0.55, fontSize: 12, color: C.text2 });
// right: RL
card(s, 5.2, 1.15, 4.3, 3.95, "rl");
T(s, "2 · RL-контроллер (PPO)", { x: 5.4, y: 1.25, w: 3.9, h: 0.32, fontSize: 15, bold: true, color: C.accent5 });
chip(5.4, 1.65, 1.2, "ПД → BC", C.accent5);
chip(6.7, 1.65, 1.2, "PPO + GAE", C.accent5);
chip(8.0, 1.65, 1.3, "Рандомизация", C.accent5);
formula(s, "f_ppo", 5.4, 2.25, 3.9);
formula(s, "f_gae", 5.4, 3.05, 2.3);
T(s, "γ = 0.99, λ = 0.95, ε = 0.2\nlr = 1e-4, 8 эпоха, minibatch 512", { x: 7.8, y: 3.15, w: 1.65, h: 0.9, fontSize: 11.5, color: C.text2 });
T(s, "Актор 24 072 параметр, 16 мкс; асимметриялы критик плант күйін көреді", { x: 5.4, y: 4.45, w: 3.9, h: 0.55, fontSize: 12, color: C.text2 });
s.addNotes("Бірінші модель — тасымал суррогаты: 11 локальды белгіден (R/L_T, q, Te, ne, ...) жылу өткізгіштік χ-ны болжайтын MLP, 65 мың үлгіде оқытылған, тест R² 0.943, MAE 0.76 м²/с. Екінші — RL-контроллер: алдымен эксперт ПД-реттегіштен behaviour cloning, содан кейін PPO, плантты рандомизациялаумен. Барлық сан скриптті іске қосқанда алынды.");

/* =================== 12. control module =================== */
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "AI" });
title(s, "Магниттік басқару модулі");
const fb = (x, w, t, col) => {
  card(s, x, 1.2, w, 0.78, "блок " + t);
  T(s, t, { x: x + 0.05, y: 1.2, w: w - 0.1, h: 0.78, fontSize: 12.5, bold: true, color: col, align: "center", valign: "middle" });
};
fb(0.5, 1.15, "Симулятор\n1 кГц", C.accent1);
fb(1.95, 1.15, "Датчиктер + кідіріс 1–3 мс", C.accent3);
fb(3.4, 1.15, "Нейро-контроллер", C.accent5);
fb(4.85, 0.95, "QP-сүзгі", C.accent4);
arrow(s, 1.65, 1.59, 1.95, 1.59, C.text2);
arrow(s, 3.1, 1.59, 3.4, 1.59, C.text2);
arrow(s, 4.55, 1.59, 4.85, 1.59, C.text2);
s.addShape(pres.shapes.LINE, { x: 0.9, y: 2.2, w: 4.4, h: 0, line: { color: C.text2, width: 1.5, dashType: "dash" }, objectName: "кері байланыс" });
s.addShape(pres.shapes.LINE, { x: 5.3, y: 1.98, w: 0, h: 0.22, line: { color: C.text2, width: 1.5, dashType: "dash" }, objectName: "кері байланыс 2" });
s.addShape(pres.shapes.LINE, { x: 0.9, y: 1.98, w: 0, h: 0.22, line: { color: C.text2, width: 1.5, endArrowType: "triangle", dashType: "dash" }, flipV: true, objectName: "кері байланыс 3" });
T(s, "катушка кернеулері", { x: 1.6, y: 2.25, w: 3, h: 0.25, fontSize: 11, color: C.text2, align: "center" });
T(s, "Плант: тізбек теңдеуі", { x: 0.55, y: 2.55, w: 3, h: 0.22, fontSize: 11.5, bold: true, color: C.accent1 });
formula(s, "f_circuit", 0.55, 2.8, 2.6);
T(s, "QP қауіпсіздік сүзгісі", { x: 0.55, y: 3.3, w: 3, h: 0.22, fontSize: 11.5, bold: true, color: C.accent4 });
formula(s, "f_qp", 0.55, 3.55, 5.0);
T(s, "Плазма тік тұрақсыз: γ ≈ 79 с⁻¹ (KTM-ға ұқсас параметрлер)", { x: 0.55, y: 4.5, w: 5.3, h: 0.55, fontSize: 12.5, color: C.text2 });
s.addChart(pres.charts.BAR, [{ name: "RMS орын қатесі, мм", labels: ["ПИД", "PPO-0", "PPO-DR", "PPO-DR-AC"], values: [22.0, 5.6, 6.0, 6.3] }], {
  x: 6.0, y: 1.15, w: 3.5, h: 2.7, barDir: "col", chartColors: [H.accent3, H.accent5, H.accent5, H.accent4],
  showValue: true, dataLabelColor: H.dk1, dataLabelFontSize: 12, dataLabelFontFace: "+mn-lt", dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0",
  catAxisLabelColor: H.dk1, catAxisLabelFontSize: 11, catAxisLabelFontFace: "+mn-lt", valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
  showLegend: false, showTitle: true, title: "Орын қатесі, мм (30 сид, рандомдалған плант)", titleColor: H.dk2, titleFontSize: 12, titleFontFace: "+mn-lt", barGapWidthPct: 50,
});
card(s, 6.0, 3.95, 3.5, 1.15, "нәтиже");
T(s, [
  { text: "Аман қалу: ", options: { color: C.text2 } },
  { text: "ПИД 24/30, PPO 28/30", options: { bold: true, color: C.accent4, breakLine: true } },
  { text: "Тек модельдеуде; QP шекті кепілдемейді.", options: { color: C.accent3, fontSize: 12 } },
], { x: 6.15, y: 4.0, w: 3.25, h: 1.05, fontSize: 13.5, valign: "middle" });
s.addNotes("Тұйық контур: симулятор — датчиктер мен кідіріс — нейрожелілік контроллер — QP қауіпсіздік сүзгісі — катушкаларға кернеу. Плазма тік тұрақсыз, тіпті ашық контурда 30 эпизодтың тек 2-інде ғана аман қалады. Рандомдалған плантта ПИД 30-дан 24-те, PPO 28-де аман қалды, орын қатесі 22 мм-ден 6 мм-ге дейін азайды. Бұл тек модельдеу нәтижесі, нақты қондырғыда тексерілмеген.");

/* =================== 13. certificate =================== */
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Қорытынды" });
title(s, "Авторлық құқық куәлігі");
img(s, "cert.jpg", 0.7, 1.1, 2.83, 4.0, "Авторлық құқық куәлігі № 78976");
const kv = [
  ["Куәлік", "№ 78976, 2026 жылғы 28 қыркүйек"],
  ["Объект түрі", "компьютерлік бағдарлама (бағдарламалық қамтылым)"],
  ["Атауы", "SINTEZ: токамак реакторындағы термоядролық синтез процесін нақты физикалық модельдер арқылы есептейтін интерактивті симулятор"],
  ["Жасалған күні", "23.09.2026"],
  ["Авторлар", "Уәлихан Алмагүл Шоқанқызы, Ермекұлы Бейбарыс"],
];
const kh = [0.62, 0.62, 1.2, 0.62, 0.62];
let ky = 1.1;
kv.forEach((r, i) => {
  card(s, 3.8, ky, 5.7, kh[i], "куәлік " + r[0]);
  T(s, r[0].toUpperCase(), { x: 3.98, y: ky + 0.07, w: 5.3, h: 0.22, fontSize: 10.5, bold: true, color: C.accent1, charSpacing: 2 });
  T(s, r[1], { x: 3.98, y: ky + 0.3, w: 5.35, h: kh[i] - 0.33, fontSize: 13.5 });
  ky += kh[i] + 0.1;
});
s.addNotes("SINTEZ Қазақстан Республикасының Ұлттық зияткерлік меншік институтында авторлық құқықпен тіркелген: куәлік № 78976, 2026 жылғы 28 қыркүйек. Құжаттың түпнұсқалығын kazpatent.kz сайтында тексеруге болады.");

/* =================== 14. conclusions =================== */
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Қорытынды" });
title(s, "Қорытынды және жоспар");
card(s, 0.5, 1.2, 4.4, 3.85, "қорытынды");
T(s, "НӘТИЖЕ", { x: 0.7, y: 1.32, w: 4.0, h: 0.25, fontSize: 12, bold: true, color: C.accent4, charSpacing: 3 });
T(s, [
  { text: "Нақты физика теңдеулері браузерде нақты уақытта шешіледі", options: { bullet: true, breakLine: true } },
  { text: "ITER Q = 10 режимі: 30 шама, орташа ауытқу 0.93 %", options: { bullet: true, breakLine: true } },
  { text: "12 құрылғы, соның ішінде қазақстандық KTM", options: { bullet: true, breakLine: true } },
  { text: "Интерфейс пен құжаттама қазақ тілінде", options: { bullet: true } },
], { x: 0.7, y: 1.65, w: 4.0, h: 3.3, fontSize: 17, paraSpaceAfter: 12 });
card(s, 5.1, 1.2, 4.4, 3.85, "шектеулер");
T(s, "ШЕКТЕУЛЕР ЖӘНЕ ЖОСПАР", { x: 5.3, y: 1.32, w: 4.0, h: 0.25, fontSize: 12, bold: true, color: C.accent3, charSpacing: 3 });
T(s, [
  { text: "Турбуленттік тасымал эмпирикалық, гирокинетика жоқ", options: { bullet: true, breakLine: true } },
  { text: "МГД — редукцияланған модельдер", options: { bullet: true, breakLine: true } },
  { text: "Суррогатты нақты тәжірибелік деректерге оқыту", options: { bullet: true, breakLine: true } },
  { text: "Контроллер тек симуляцияда тексерілген", options: { bullet: true } },
], { x: 5.3, y: 1.65, w: 4.0, h: 3.3, fontSize: 17, paraSpaceAfter: 12 });
s.addNotes("Қорытынды: симулятор физиканы нақты уақытта есептейді және ITER режимімен 1 % шамасында сәйкес келеді. Шектеулерді ашық айтамыз: тасымал эмпирикалық, МГД редукцияланған, суррогат әзірге өз заңдарымыздан оқытылған, контроллер симуляцияда ғана тексерілген. Келесі қадам — нақты тәжірибелік деректер.");

/* =================== 15. thanks + QR =================== */
s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Қорытынды" });
T(s, "Назарларыңызға\nрақмет!", { x: 0.6, y: 1.2, w: 5.2, h: 1.5, fontSize: 38, bold: true, color: C.text1 });
T(s, "SINTEZ — онлайн демо", { x: 0.6, y: 3.0, w: 5.0, h: 0.4, fontSize: 20, bold: true, color: C.accent1 });
T(s, "QR-кодты сканерлеп, симуляторды өз құрылғыңызда ашыңыз", { x: 0.6, y: 3.45, w: 4.8, h: 0.7, fontSize: 15, color: C.text1 });
T(s, "Уәлихан А.Ш., Ермекұлы Б.\nҚожа Ахмет Яссауи атындағы ХҚТУ, Түркістан", { x: 0.6, y: 4.4, w: 5.2, h: 0.7, fontSize: 13, color: C.text2 });
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.2, y: 1.0, w: 3.2, h: 3.2, rectRadius: 0.12, fill: { color: "FFFFFF" }, line: { color: "FFFFFF", width: 0 }, objectName: "QR фон" });
s.addImage({ path: A("qr.png"), x: 6.35, y: 1.15, w: 2.9, h: 2.9, altText: "QR-код: https://tokomak-production.up.railway.app" });
T(s, "tokomak-production.up.railway.app", { x: 5.9, y: 4.35, w: 3.8, h: 0.35, fontSize: 14, color: C.accent3, align: "center", bold: true });
s.addNotes("Назарларыңызға рақмет! QR-код арқылы SINTEZ симуляторын онлайн ашуға болады. Сұрақтарыңызға дайынмын.");

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("written", OUT);
})();
