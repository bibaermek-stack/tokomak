// Replace pandoc's math spans (\( .. \) and \[ .. \]) with KaTeX-rendered HTML.   usage: node katex_render.js < in.html > out.html
const katex = require("../node_modules/katex");
const fs = require("fs");

const unescape = (s) => s.replace(/&#(\d+);/g, (_, n) => String.fromCharCode(+n)).replace(/&#x([0-9a-f]+);/gi, (_, h) => String.fromCharCode(parseInt(h, 16)))
  .replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&amp;/g, "&");
let html = fs.readFileSync(0, "utf8");
let n = 0;
const render = (tex, display) => {
  n += 1;
  return katex.renderToString(unescape(tex), {
    displayMode: display, throwOnError: true, output: "html", strict: "error", trust: false,
    macros: {
      "\\d": "\\mathrm{d}", "\\e": "\\mathrm{e}", "\\R": "\\mathbb{R}",
      "\\T": "T", "\\kev": "\\,\\mathrm{keV}", "\\MW": "\\,\\mathrm{MW}", "\\m2s": "\\,\\mathrm{m^2/s}",
    },
  });
};
html = html.replace(/<span class="math display">\\\[([\s\S]*?)\\\]<\/span>/g, (_, t) => render(t, true));
html = html.replace(/<span class="math inline">\\\(([\s\S]*?)\\\)<\/span>/g, (_, t) => render(t, false));
process.stdout.write(html);
process.stderr.write(`katex: ${n} formulas\n`);
