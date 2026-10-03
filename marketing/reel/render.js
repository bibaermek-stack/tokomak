/* Render composer.html frame by frame to a JPEG sequence.
 * usage: node render.js <url-of-composer> <outdir> [t0] [t1] [step]
 * t0/t1/step let a quick preview render only a few instants. */
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright-core');

const URL = process.argv[2] || 'http://localhost:8098/composer.html';
const OUT = process.argv[3] || './frames';
const EXE = process.env.CHROME || '/opt/pw-browsers/chromium';

(async () => {
  const browser = await chromium.launch({ executablePath: EXE });
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 }, deviceScaleFactor: 1 });
  /* tell the composer how many frames each captured clip actually has */
  const base = path.resolve(process.env.CAP || './cap');
  const counts = {};
  for (const c of ['landing', 'flight', 'interior']) {
    try { counts[c] = fs.readdirSync(path.join(base, c)).filter(f => f.endsWith('.jpg')).length; } catch { counts[c] = 0; }
  }
  /* a partial render before capture finishes must use the final counts,
     or a clip would be stretched to fit the frames that exist so far */
  if (process.env.CLIP_N) Object.assign(counts, JSON.parse(process.env.CLIP_N));
  await page.addInitScript(c => { window.CLIP_N = c; }, counts);
  console.log('клиптер:', JSON.stringify(counts));
  await page.goto(URL, { waitUntil: 'networkidle' });
  await page.evaluate(() => document.fonts.ready);
  const DUR = await page.evaluate(() => window.DUR), FPS = await page.evaluate(() => window.FPS);
  const t0 = +(process.argv[4] ?? 0), t1 = +(process.argv[5] ?? DUR), stepS = +(process.argv[6] ?? 1 / FPS);
  fs.mkdirSync(OUT, { recursive: true });
  const n = Math.round((t1 - t0) / stepS);
  const st = Date.now();
  for (let i = 0; i < n; i++) {
    const t = t0 + i * stepS;
    await page.evaluate(tt => window.render(tt), t);
    /* frames are named by their place on the whole timeline, so separate
       ranges can be rendered at different times into one sequence */
    const name = process.argv[6] ? `t_${t.toFixed(2)}.jpg` : `f_${String(Math.round(t * FPS)).padStart(4, '0')}.jpg`;
    await page.screenshot({ path: path.join(OUT, name), type: 'jpeg', quality: 93 });
  }
  console.log(`${n} кадр, ${((Date.now() - st) / Math.max(n, 1)).toFixed(0)} мс/кадр`);
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
