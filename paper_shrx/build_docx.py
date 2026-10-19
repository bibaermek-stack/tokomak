#!/usr/bin/env python3
"""Word version of the article: the same sections and numbers as build.py  ->  article_kk.docx.

    python build_docx.py [--draft]          (needs pandoc >= 3, python-docx and pillow; LibreOffice only to check the result)

Formulas become native Word equations (pandoc converts the TeX to OMML), figures are the 300 dpi PNG copies written by make_figures.py,
tables are real Word tables.  Numbers (equations, figures, tables, sections, references) are the ones build.py assigns, so both files agree.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build as B  # noqa: E402

TEXT_W_IN = 6.5          # text width of the A4 page with 2 cm margins is 6.69 in; leave a little room
OUT = HERE / "article_kk.docx"
LUA = HERE / "tools" / "docx_filter.lua"


def image_for(path):
    """Word cannot embed SVG: use the PNG copy of a vector figure."""
    p = HERE / path
    if p.suffix == ".svg":
        p = HERE / "figures" / "png" / (p.stem + ".png")
    return p


def width_in(p, opts):
    w = next((o[2:] for o in opts if o.startswith("w=")), None)
    if w:
        return TEXT_W_IN * float(w) / 100
    if p.is_file():
        with Image.open(p) as im:
            return min(im.width / 300.0, TEXT_W_IN)
    return TEXT_W_IN


def transform(text):
    B.number_sections(text)
    # headings: numbers written into the text (the docx writer has no reliable --number-sections)
    counters = [0, 0, 0]
    out = []
    for line in text.splitlines():
        m = re.match(r"^(#{1,3})[ \t]+(.*?)[ \t]*(\{[^}]*\})?[ \t]*$", line)
        if not m:
            out.append(line)
            continue
        attrs = m.group(3) or ""
        if re.search(r"\{\s*-\s*\}", attrs) or ".unnumbered" in attrs:
            out.append(f"{m.group(1)} {m.group(2)} {{-}}")
            continue
        lvl = len(m.group(1)) - 1
        counters[lvl] += 1
        for j in range(lvl + 1, 3):
            counters[j] = 0
        out.append(f"{m.group(1)} {'.'.join(str(n) for n in counters[:lvl + 1])} {m.group(2)}")
    text = "\n".join(out)
    for big, kw in (("\\Bigl", "\\left"), ("\\Bigr", "\\right"), ("\\bigl", "\\left"), ("\\bigr", "\\right")):   # texmath knows only \left / \right
        text = text.replace(big, kw)

    # front matter (raw HTML in 00_front.md)
    for cls, style in (("title", "Title"), ("subtitle", "Subtitle"), ("authors", "Authors"), ("affil", "Affiliation")):
        text = re.sub(rf'<div class="{cls}">(.*?)</div>', lambda m, s=style: f'::: {{custom-style="{s}"}}\n{m.group(1).strip()}\n:::', text, flags=re.S)
    text = re.sub(r'<p class="udk">(.*?)</p>', lambda m: f'::: {{custom-style="UDK"}}\n{m.group(1).strip()}\n:::', text, flags=re.S)
    text = text.replace('<div class="abstract">', "").replace('<hr class="sep">', "")
    text = re.sub(r"^</div>[ \t]*$", "", text, flags=re.M)

    # equations
    def eq(m):
        name = m.group(2)
        n = B.EQ.setdefault(name, len(B.EQ) + 1)
        body = " ".join(m.group(1).split())
        return f'::: {{.eqn num="{n}"}}\n$${body}$$\n:::'
    text = re.sub(r"\$\$(.+?)\$\$[ \t]*\{#eq:([\w-]+)\}", eq, text, flags=re.S)

    # figures
    def fig(m):
        name, path, opts, cap = m.group(1), m.group(2), (m.group(3) or "").split(), " ".join(m.group(4).split())
        n = B.FIG.setdefault(name, len(B.FIG) + 1)
        p = image_for(path)
        if not p.is_file() and "--draft" not in sys.argv:
            raise SystemExit(f"missing image {p}")
        return f"![**{n}-сурет.** {cap}]({p.relative_to(HERE) if p.is_file() else path}){{width={width_in(p, opts):.2f}in}}"
    text = re.sub(r"^:::fig[ \t]+([\w-]+)[ \t]+(\S+)([^\n]*)\n(.*?)^:::[ \t]*$", fig, text, flags=re.S | re.M)

    # tables
    def tbl(m):
        name, body = m.group(1), m.group(2).strip()
        n = B.TBL.setdefault(name, len(B.TBL) + 1)
        cap, _, table = body.partition("\n\n")
        return f"{table.strip()}\n\n: **{n}-кесте.** {' '.join(cap.split())}\n"
    text = re.sub(r"^:::table[ \t]+([\w-]+)[ \t]*\n(.*?)^:::[ \t]*$", tbl, text, flags=re.S | re.M)

    def ref(m):
        kind, name = m.group(1), m.group(2)
        if kind == "sec":
            return B.SEC[name]
        n = {"fig": B.FIG, "tbl": B.TBL, "eq": B.EQ}[kind][name]
        return {"fig": f"{n}-сурет", "tbl": f"{n}-кесте", "eq": f"({n})"}[kind]
    return re.sub(r"@(fig|tbl|eq|sec):([\w-]+)", ref, text)


def refs_md():
    items = "\n".join(f"{i}. {B.REFS[k]}" for i, k in enumerate(B.CITED, 1))
    return f"# Әдебиеттер {{-}}\n\n{items}\n"


def main():
    raw = B.read_sections()
    expanded = B.expand_macros(raw)      # same order as build.py: macros first (they may add figures/tables), then the numbering
    text = transform(expanded)
    text = B.cite(text)
    text = text.replace("<!--REFS-->", refs_md())
    if "<!--REFS-->" not in raw and B.CITED:
        text += "\n\n" + refs_md()
    md = B.BUILD / "article_docx.md"
    md.write_text(text)
    ref_doc = B.BUILD / "reference.docx"
    ref_doc.write_bytes(subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"], capture_output=True, check=True).stdout)
    raw_docx = B.BUILD / "article_raw.docx"
    r = subprocess.run(["pandoc", "-f", "markdown+tex_math_dollars+pipe_tables+smart+fenced_divs+implicit_figures", "-t", "docx",
                        "--lua-filter", str(LUA), "--reference-doc", str(ref_doc), "--resource-path", str(HERE), "-o", str(raw_docx),
                        str(md)], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"pandoc failed:\n{r.stderr[-3000:]}")
    if r.stderr.strip():
        print("pandoc:", r.stderr.strip()[:3000])
    from docx_post import finish          # noqa: E402  (python-docx, kept in a separate module so that build.py does not need it)
    finish(raw_docx, OUT)
    print(f"docx {OUT.stat().st_size / 1e6:.2f} MB; {len(B.EQ)} equations, {len(B.FIG)} figures, {len(B.TBL)} tables, {len(B.CITED)} references")


if __name__ == "__main__":
    main()
