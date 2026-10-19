#!/usr/bin/env python3
"""Build the article: sections/*.md + data/*.json  ->  build/article_kk.html (self-contained)  ->  article_kk.pdf.

    python build.py [--no-pdf]

Source syntax (Markdown + LaTeX, rendered with KaTeX at build time; text black on white, figures and screenshots in colour):

  {{= expr }}                    Python expression; D is the dict of data/*.json, helpers c(x, n), pct(x, n), sg(x, n)
  $...$   $$...$$ {#eq:name}     inline / numbered display math (KaTeX)
  :::fig name path [shot|wide]   figure; the lines up to ::: are the caption (Markdown)
  :::table name                  table; first paragraph is the caption, the rest a pipe table
  @fig:name  @tbl:name  @eq:name cross-references ("3-сурет", "2-кесте", "(5)")
  [@key; @key2]                  citation, numbered by first appearance, list from refs.py

Needs: pandoc, node with `npm install` done in this folder (KaTeX), Chromium for the PDF.
"""
import base64
import json
import math
import mimetypes
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
sys.path.insert(0, str(HERE))
from refs import REFS  # noqa: E402

BUILD.mkdir(exist_ok=True)


# ------------------------------------------------------------------------------------------ number formatting
def c(x, n=1):
    """Decimal comma, true minus."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:.{n}f}".replace(".", ",").replace("-", "−")


def pct(x, n=0):
    if x is None:
        return "—"
    return c(100.0 * x, n) + " %"


def cm(x, n=1):
    """Number for use inside $...$: the decimal comma needs braces, otherwise TeX adds a space after it."""
    return c(x, n).replace(",", "{,}")


def pctm(x, n=0):
    """Percent for use inside $...$ (a bare % starts a comment in TeX)."""
    return c(100.0 * x, n).replace(",", "{,}") + "\\,\\%"


def sg(x, n=2):
    """n significant figures, decimal comma."""
    if x == 0:
        return "0"
    d = n - 1 - int(math.floor(math.log10(abs(x))))
    return c(round(x, d), max(d, 0))


def sci(x, n=1):
    """LaTeX scientific notation (inside $...$)."""
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    return f"{cm(m, n)}\\times10^{{{e}}}"


def thou(n):
    return f"{int(n):,}".replace(",", "\u00a0")


D = {p.stem: json.loads(p.read_text()) for p in sorted((HERE / "data").glob("*.json"))}
from values import make as _make_values  # noqa: E402
V = _make_values(D)
NS = {"pctm": pctm, "D": D, "V": V, "S": D.get("shrx_results"), "T": D.get("torax_results"), "E": D.get("ensemble"), "I": D.get("inloop"), "C": D.get("chaos"), "U": D.get("uncertainty"), "Sx": D.get("st_teacher"),
      "B": D.get("db5"), "c": c, "pct": pct, "sg": sg, "sci": sci, "thou": thou, "np": np, "math": math, "len": len, "round": round,
      "min": min, "max": max, "sum": sum, "abs": abs, "sorted": sorted, "int": int, "float": float, "range": range, "zip": zip,
      "enumerate": enumerate, "list": list, "dict": dict}


def expand_macros(text):
    """Evaluate {{= expr }}.  Inside $...$ the helpers switch to their math-safe versions (decimal comma in braces, TeX-safe percent)."""
    ns_math = {**NS, "c": cm, "pct": pctm}

    def run(expr, ns):
        try:
            return str(eval(expr, ns))
        except Exception as e:                       # a wrong number must stop the build, not slip into the text
            if "--draft" in sys.argv:                # draft builds (layout checks while data are still arriving) show a visible mark
                DRAFT_MISSING.append(expr[:60])
                return "[?]"
            raise SystemExit(f"macro failed: {{{{= {expr} }}}}\n  -> {type(e).__name__}: {e}")

    def in_math(m):
        return re.sub(r"\{\{=(.+?)\}\}", lambda mm: run(mm.group(1).strip(), ns_math), m.group(0), flags=re.S)
    text = re.sub(r"\$\$.+?\$\$|(?<!\$)\$(?!\$)[^$\n]+?\$", in_math, text, flags=re.S)
    return re.sub(r"\{\{=(.+?)\}\}", lambda m: run(m.group(1).strip(), NS), text, flags=re.S)


# ------------------------------------------------------------------------------------------ numbering
FIG, TBL, EQ, SEC = {}, {}, {}, {}


def number_sections(text):
    """Section numbers as pandoc will assign them (# = 1, ## = 1.1, ### = 1.1.1); ``{-}`` headings are unnumbered."""
    counters = [0, 0, 0]
    for line in text.splitlines():
        m = re.match(r"^(#{1,3})[ \t]+(.*?)[ \t]*(\{[^}]*\})?[ \t]*$", line)
        if not m:
            continue
        attrs = m.group(3) or ""
        if "-" in attrs.replace("#", "").split() or re.search(r"\{\s*-\s*\}", attrs) or ".unnumbered" in attrs:
            continue
        lvl = len(m.group(1)) - 1
        counters[lvl] += 1
        for j in range(lvl + 1, 3):
            counters[j] = 0
        sid = re.search(r"#sec-([\w-]+)", attrs)
        if sid:
            SEC[sid.group(1)] = ".".join(str(n) for n in counters[:lvl + 1])


def number_blocks(text):
    number_sections(text)
    # equations: $$ ... $$ {#eq:name}
    def eq(m):
        name = m.group(2)
        n = EQ.setdefault(name, len(EQ) + 1)
        body = m.group(1).strip()
        return f'<div class="eq" id="eq-{name}">\n\n$${body} \\tag{{{n}}}$$\n\n</div>'
    text = re.sub(r"\$\$(.+?)\$\$[ \t]*\{#eq:([\w-]+)\}", eq, text, flags=re.S)

    # figures
    def fig(m):
        name, path, opts, cap = m.group(1), m.group(2), (m.group(3) or "").split(), m.group(4).strip()
        n = FIG.setdefault(name, len(FIG) + 1)
        cls = "shot" if "shot" in opts else ""
        w = next((o[2:] for o in opts if o.startswith("w=")), None)
        style = f' style="width:{w}%"' if w else ""
        return (f'<figure id="fig-{name}" class="{cls}">\n<img src="{path}" alt="Сурет {n}"{style}>\n<figcaption>\n\n'
                f'**{n}-сурет.** {cap}\n\n</figcaption>\n</figure>')
    text = re.sub(r"^:::fig[ \t]+([\w-]+)[ \t]+(\S+)([^\n]*)\n(.*?)^:::[ \t]*$", fig, text, flags=re.S | re.M)

    # tables
    def tbl(m):
        name, body = m.group(1), m.group(2).strip()
        n = TBL.setdefault(name, len(TBL) + 1)
        cap, _, table = body.partition("\n\n")
        return (f'<div class="tablewrap" id="tbl-{name}">\n<div class="tcap">\n\n**{n}-кесте.** {cap.strip()}\n\n</div>\n\n'
                f'{table.strip()}\n\n</div>')
    text = re.sub(r"^:::table[ \t]+([\w-]+)[ \t]*\n(.*?)^:::[ \t]*$", tbl, text, flags=re.S | re.M)

    def ref(m):
        kind, name = m.group(1), m.group(2)
        if kind == "sec":
            if name not in SEC:
                raise SystemExit(f"unknown reference @sec:{name}")
            return f'<a href="#sec-{name}">{SEC[name]}</a>'
        table = {"fig": FIG, "tbl": TBL, "eq": EQ}[kind]
        if name not in table:
            raise SystemExit(f"unknown reference @{kind}:{name}")
        n = table[name]
        label = {"fig": f"{n}-сурет", "tbl": f"{n}-кесте", "eq": f"({n})"}[kind]
        return f'<a href="#{kind}-{name}">{label}</a>'
    return re.sub(r"@(fig|tbl|eq|sec):([\w-]+)", ref, text)


CITED = []
DRAFT_MISSING = []


def cite(text):
    def one(m):
        keys = re.findall(r"@([\w:-]+)", m.group(0))
        nums = []
        for k in keys:
            if k not in REFS:
                raise SystemExit(f"unknown citation @{k}")
            if k not in CITED:
                CITED.append(k)
            nums.append(CITED.index(k) + 1)
        nums = sorted(set(nums))
        # collapse runs: 1,2,3 -> 1–3
        out, i = [], 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ", ".join(map(str, nums[i:j + 1])))
            i = j + 1
        return "[" + ", ".join(out) + "]"
    return re.sub(r"\[@[\w:-]+(?:\s*;\s*@[\w:-]+)*\]", one, text)


# ------------------------------------------------------------------------------------------ pipeline
def read_sections():
    parts = []
    for p in sorted((HERE / "sections").glob("*.md")):
        parts.append(p.read_text())
    return "\n\n".join(parts)


def run(cmd, inp=None):
    r = subprocess.run(cmd, input=inp, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"{cmd[0]} failed:\n{r.stderr[-3000:]}")
    if r.stderr.strip():
        print(r.stderr.strip())
    return r.stdout


def katex_css():
    css = (HERE / "node_modules/katex/dist/katex.min.css").read_text()
    fonts = HERE / "node_modules/katex/dist/fonts"

    def font(m):
        f = fonts / m.group(1)
        return f'url(data:font/woff2;base64,{base64.b64encode(f.read_bytes()).decode()}) format("woff2")'
    css = re.sub(r"url\(fonts/([\w.-]+\.woff2)\) format\(\"woff2\"\),url\(fonts/[\w.-]+\.woff\) format\(\"woff\"\),"
                 r"url\(fonts/[\w.-]+\.ttf\) format\(\"truetype\"\)", font, css)
    assert "fonts/" not in css, "a KaTeX font reference was left unembedded"
    return css


def embed_images(html):
    def sub(m):
        path = HERE / m.group(1)
        if not path.is_file():
            if "--draft" in sys.argv:
                return 'src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw=="'
            raise SystemExit(f"missing image {path}")
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        return f'src="data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"'
    return re.sub(r'src="((?:figures|screenshots)/[^"]+)"', sub, html)


def refs_html():
    items = "\n".join(f"<li>{REFS[k]}</li>" for k in CITED)
    return f'<section class="refs">\n<h1 class="unnumbered">Әдебиеттер</h1>\n<ol>\n{items}\n</ol>\n</section>'


def stray_tex_check(html):
    """TeX that was meant for KaTeX but sits outside $...$ would be printed literally: look for it in the text left after removing the math and code."""
    t = re.sub(r'<span class="math (?:inline|display)">.*?</span>', "", html, flags=re.S)
    t = re.sub(r"<(code|pre)[ >].*?</\1>", "", t, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    bad = re.findall(r".{25}(?:\\[A-Za-z]{2,}|\{,\}|\^\{|_\{|\$).{25}", t)
    if bad:
        msg = "TeX outside math delimiters:\n  " + "\n  ".join(b.strip() for b in bad[:8])
        if "--draft" in sys.argv:
            print("WARNING:", msg)
        else:
            raise SystemExit(msg)


def main():
    text = read_sections()
    text = expand_macros(text)           # macros may add whole figures/tables (conditional blocks), so they come before the numbering
    text = number_blocks(text)           # numbering sees the document in reading order
    text = cite(text)
    md = BUILD / "article.md"
    md.write_text(text)
    body = run(["pandoc", "-f", "markdown+tex_math_dollars+raw_html+pipe_tables+smart+fenced_divs", "-t", "html5", "--mathjax",
                "--wrap=none", "--number-sections", "-V", "lang=kk"], text)
    stray_tex_check(body)
    body = run(["node", str(HERE / "tools/katex_render.js")], body)
    body = body.replace("<!--REFS-->", refs_html())
    if "<!--REFS-->" not in text and CITED:
        body += refs_html()
    style = (HERE / "style.css").read_text()
    html = ('<!DOCTYPE html>\n<html lang="kk"><head><meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            '<title>SHRX: токамак плазмасының турбуленттік тасымалының нейрожелілік моделі және оны TORAX-пен салыстыру</title>\n'
            f'<style>{katex_css()}</style>\n<style>{style}</style>\n</head><body>\n{body}\n</body></html>\n')
    html = embed_images(html)
    out = BUILD / "article_kk.html"
    out.write_text(html)
    if DRAFT_MISSING:
        print(f"DRAFT: {len(DRAFT_MISSING)} macros could not be evaluated, e.g. {DRAFT_MISSING[:2]}")
    assert "[?]" not in html or "--draft" in sys.argv, "unevaluated macro in the final build"
    print(f"html {out.stat().st_size / 1e6:.2f} MB; {len(EQ)} equations, {len(FIG)} figures, {len(TBL)} tables, {len(CITED)} references")
    if "--no-pdf" not in sys.argv:
        pdf = HERE / "article_kk.pdf"
        run([CHROME, "--headless=new", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage", "--no-pdf-header-footer",
             f"--print-to-pdf={pdf}", "--virtual-time-budget=20000", f"file://{out}"])
        print(f"pdf {pdf.stat().st_size / 1e6:.2f} MB")
        (HERE / "article_kk.html").write_text(html)


if __name__ == "__main__":
    main()
