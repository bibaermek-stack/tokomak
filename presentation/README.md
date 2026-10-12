# SINTEZ — IFARP-2026 конференция презентациясы

`SINTEZ_IFARP-2026.pptx` — 15 слайд (қазақша), әр слайдта баяндамашыға арналған ескертпелер бар.

| Слайд | Мазмұны |
|---|---|
| 1–2 | тақырып, мәселе және мақсат |
| 3–8 | архитектура, физикалық модельдер, теңдеулер, сайт скриншоттары, 12 құрылғы |
| 9 | ITER Q=10 валидациясы |
| 10–12 | Google DeepMind TORAX-пен салыстыру, модельдерді оқыту және формулалар, басқару модулі |
| 13 | авторлық құқық куәлігі № 78976 |
| 14–15 | қорытынды, QR-код (онлайн демо) |

## Сандардың шығуы

Барлық сан іске қосылған командадан алынған:

* валидация (30 шама, 0.93 % / 5.52 %): `node tools/validate.js`; 79/79: `python -m tokamak.validation`;
* суррогат (R² 0.943, MAE 0.76 м²/с, 9 857 параметр, 122 vs 148 мкс): `python -m ml.train_transport_surrogate` + өлшеу;
* басқару нәтижелері: `paper/results.json` (30 сид, рандомдалған плант);
* құрылғы параметрлері: `backend/tokamak/machines.py`;
* скриншоттар — `marketing/reel/capture.js` тәсілімен (виртуалды сағат, SwiftShader) түсірілген нақты сайт.

QR-код `https://tokomak-production.up.railway.app` мекенжайына апарады (`js/ai.js` ішінен алынған).

## Қайта құру

```bash
cd presentation
npm install pptxgenjs react react-dom react-icons sharp
python3 make_formulas.py          # формулалар → assets/f_*.png
NODE_PATH=$PWD/node_modules node build_deck.js
```
