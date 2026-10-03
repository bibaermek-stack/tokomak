"""Builds the article (.docx) from the Markdown parts and results.json.

    python build_paper.py results.json out.docx

* ``{{path|fmt}}``      value from results.json (dotted path), Python format,
                         decimal comma;
* ``{{= expr }}``       Python expression; ``r`` is the results dict, ``pm``
                         formats a number with a decimal comma;
* ``{{table:name}}``    a results table, see ``TABLES``;
* ``[@key; @key2]``     citations, numbered by first appearance.

Pandoc turns the LaTeX into native Word equations.  Pandoc drops ``\\tag``,
so display equations are numbered here: each display equation is moved into
a borderless three-column table (empty | equation | number).
"""

import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from scipy.stats import wilcoxon

HERE = Path(__file__).parent
PANDOC = "/usr/local/lib/python3.11/dist-packages/pypandoc/files/pandoc"

sys.path.insert(0, str(HERE))
from refs import REFS  # noqa: E402

COND_KK = {"nominal": "номиналды", "randomised": "рандомизацияланған",
           "gain 1.3": "күшейту 1,3", "gain 0.7": "күшейту 0,7",
           "wall x2": "камера ×2", "wall x0.5": "камера ×0,5",
           "plasma x4": "плазма ×4", "plasma x0.25": "плазма ×0,25",
           "delay 4 ms": "кідіріс 4 мс", "all adverse": "бәрі қолайсыз"}
NAMES = {"open_loop": "Ашық контур", "PID": "ПИД", "PPO-0": "PPO-0",
         "PPO-DR": "PPO-DR", "PPO-DR-AC": "PPO-DR-AC"}


# --- number formatting ----------------------------------------------------------------
def pm(x, fmt=".1f"):
    if x is None:
        return "—"
    return format(x, fmt).replace(".", ",").replace("-", "−")


def pct(x, nd=0):
    return pm(100 * x, f".{nd}f") + " %"


def lookup(r, path):
    cur = r
    for part in path.split("."):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


# --- tables ---------------------------------------------------------------------------------
def pipe(header, rows, align=None, widths=None):
    """Pipe table.  ``widths`` are relative column widths (pandoc reads them
    from the number of dashes in the separator row)."""
    align = align or ["l"] + ["r"] * (len(header) - 1)
    widths = widths or [8] * len(header)
    sep = []
    for a, w in zip(align, widths):
        d = "-" * max(w, 4)
        sep.append(d[:-1] + ":" if a == "r" else (":" + d[1:-1] + ":" if a == "c"
                                                  else ":" + d[1:]))
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(sep) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def cell(d, key="pos_mm_mean", se="pos_mm_se"):
    """'5,1 ± 0,3 (30/30)' -- error with its standard error and survival."""
    s = f"{int(round(d['survival'] * d['n']))}/{d['n']}"
    if d[key] is None:
        return f"— ({s})"
    e = f" ± {pm(d[se], '.1f')}" if d.get(se) is not None else ""
    return f"{pm(d[key], '.1f')}{e} ({s})"


def table_main(r):
    rows = []
    for k in ("open_loop", "PID", "PPO-0", "PPO-DR", "PPO-DR-AC"):
        if k not in r["main"]:
            continue
        for cond, lab in (("nominal", "номиналды"), ("randomised", "рандом.")):
            d = r["main"][k][cond]
            rows.append([NAMES[k], lab, f"{round(d['survival'] * d['n'])}/{d['n']}",
                         pm(d["pos_mm_mean"], ".1f") if d["pos_mm_mean"] is not None else "—",
                         pm(d["shape_mm_mean"], ".1f") if d["shape_mm_mean"] is not None else "—",
                         f"{pm(d['return_mean'], '.0f')} ± {pm(d['return_se'], '.0f')}",
                         pct(d["qp_frac"])])
    return pipe(["Контроллер", "Плант", "Аман қалу", "e_pos, мм", "e_sh, мм",
                 "Қайтарым", "QP араласуы"], rows,
                ["l", "l", "r", "r", "r", "r", "r"],
                [15, 14, 10, 9, 9, 17, 12])


def table_ood(r):
    ctrls = [k for k in ("PID", "PPO-0", "PPO-DR", "PPO-DR-AC") if k in r["ood"]]
    conds = list(r["ood"][ctrls[0]])
    rows = [[COND_KK.get(c, c)] + [cell(r["ood"][k][c]) for k in ctrls]
            for c in conds]
    return pipe(["Плант"] + [NAMES[k] for k in ctrls], rows,
                ["l"] + ["r"] * len(ctrls), [14] + [17] * len(ctrls))


def table_delay(r):
    ctrls = [k for k in ("PID", "PPO-0", "PPO-DR", "PPO-DR-AC") if k in r["delay"]]
    ds = sorted(r["delay"][ctrls[0]], key=int)
    rows = [[f"{d} мс"] + [cell(r["delay"][k][d]) for k in ctrls] for d in ds]
    return pipe(["Кідіріс"] + [NAMES[k] for k in ctrls], rows,
                ["l"] + ["r"] * len(ctrls), [10] + [18] * len(ctrls))


def table_safety(r):
    rows = []
    for k in ("PPO-0", "PPO-DR", "PPO-DR-AC"):
        if k not in r["safety"]:
            continue
        for cond, d in r["safety"][k].items():
            on, off = d["filter_on"], d["filter_off"]
            rows.append([NAMES[k], COND_KK.get(cond, cond),
                         f"{round(on['survival'] * on['n'])}/{on['n']}",
                         pct(on["over_limit_frac"], 2),
                         f"{round(off['survival'] * off['n'])}/{off['n']}",
                         pct(off["over_limit_frac"], 2),
                         pm(off["max_I_over"], ".2f")])
    return pipe(["Саясат", "Плант", "Аман қалу (сүзгімен)", "Шек бұзылды (сүзгімен)",
                 "Аман қалу (сүзгісіз)", "Шек бұзылды (сүзгісіз)",
                 "Ең үлкен асып кету I/I_max (сүзгісіз)"], rows,
                ["l", "l", "r", "r", "r", "r", "r"], [12, 12, 12, 13, 12, 13, 14])


def table_cost(r):
    rows = []
    for k, d in r["cost"].items():
        rows.append([NAMES.get(k, k), str(d["actor_in"]), str(d["critic_in"]),
                     f"{d['actor_params']:,}".replace(",", " "),
                     f"{d['actor_flops']:,}".replace(",", " "),
                     pm(d["inference_us"], ".0f")])
    return pipe(["Саясат", "Актор кірісі", "Критик кірісі", "Актор параметрлері",
                 "FLOP / инференс", "Уақыт, мкс"], rows,
                ["l", "r", "r", "r", "r", "r"], [14, 12, 12, 18, 16, 12])


def table_growth(r):
    rows = {}
    for x in r["growth"]:
        rows.setdefault(x["wall_mm"], {"g": x["gamma"]})[x["dt_ms"]] = x
    dts = sorted({x["dt_ms"] for x in r["growth"]}, reverse=True)
    out = []
    for wm in sorted(rows, reverse=True):
        row = rows[wm]
        out.append([pm(wm, ".2f"), pm(row["g"], ".0f")]
                   + ["иә" if row[dt]["stabilisable"] else "жоқ" for dt in dts])
    return pipe(["Қалыңдық, мм", "γ, с⁻¹"] + [f"{1/dt:g} кГц" for dt in dts], out,
                ["r", "r"] + ["c"] * len(dts), [16, 14] + [14] * len(dts))


def table_seeds(r):
    """Retraining check: each variant, each training seed, four conditions."""
    sd = r["seeds_check"]
    order = ["nominal", "randomised", "gain 1.3", "wall x2", "delay 4 ms"]
    conds = [c for c in order if c in next(iter(sd.values()))]
    rows = []
    for name in sorted(sd, key=lambda n: (["PPO-0", "PPO-DR", "PPO-DR-AC"].index(
            n.split("#")[0]), int(n.split("#")[1]))):
        var, k = name.split("#")
        rows.append([var, k] + [cell(sd[name][c]) for c in conds])
    return pipe(["Нұсқа", "Сид"] + [COND_KK.get(c, c) for c in conds], rows, ["l", "r"] + ["r"] * len(conds),
                [14, 7] + [16] * len(conds))


def seed_stat(r, variant, cond, key="survival"):
    """Mean and range of a metric over the training seeds of a variant."""
    v = [r["seeds_check"][n][cond][key] for n in r["seeds_check"]
         if n.split("#")[0] == variant]
    return float(np.mean(v)), float(np.min(v)), float(np.max(v)), len(v)


def sl(r, variant, cond, key="survival", scale=30, nd=0):
    """The values of a metric over a variant's training seeds, as '27, 28, 26'."""
    names = sorted(n for n in r["seeds_check"] if n.split("#")[0] == variant)
    vals = [r["seeds_check"][n][cond][key] for n in names]
    return ", ".join(pm(v * scale, f".{nd}f") if v is not None else "—"
                     for v in vals)


def smean(r, variant, cond, key="survival", scale=30, nd=1):
    """Mean of a metric over a variant's training seeds."""
    names = [n for n in r["seeds_check"] if n.split("#")[0] == variant]
    vals = [r["seeds_check"][n][cond][key] for n in names
            if r["seeds_check"][n][cond][key] is not None]
    return pm(float(np.mean(vals)) * scale, f".{nd}f") if vals else "—"


def limit_ratio_range(r):
    """Bracket for the factor by which 10x sampling raises the stabilisable
    growth rate.  The limit lies between the largest stabilisable and the
    smallest unstabilisable gamma on the thickness grid, so the ratio does too."""
    g = r["growth"]

    def lim(dt):
        ok = [x["gamma"] for x in g if x["dt_ms"] == dt and x["stabilisable"]]
        no = [x["gamma"] for x in g if x["dt_ms"] == dt and not x["stabilisable"]]
        return max(ok), min(no)
    lo1, hi1 = lim(1.0)
    lo10, hi10 = lim(0.1)
    return f"{pm(lo10 / hi1, '.1f')}–{pm(hi10 / lo1, '.1f')}"


def fail_all(r, ctrls=("PID", "PPO-0", "PPO-DR", "PPO-DR-AC"),
             section="main", cond="randomised"):
    """Indices of the seeds on which every listed controller lost the plasma."""
    sec = r[section]
    n = len(sec[ctrls[0]][cond]["episodes"])
    return [i for i in range(n)
            if all(sec[c][cond]["episodes"][i]["terminated"] for c in ctrls)]


def draw_text(r, idx):
    d = r["draws"][idx]
    return (f"сид {d['seed']}: γ = {pm(d['gamma'], '.0f')} с⁻¹, кідіріс "
            f"{d['delay']} мс, камера ×{pm(d['wall_res'], '.2f')}, "
            f"плазма ×{pm(d['plasma_res'], '.2f')}")


def lost_by(r, ctrl, section="main", cond="randomised"):
    sec = r[section][ctrl]
    ep = sec[cond]["episodes"] if cond else sec["episodes"]
    return [i for i, e in enumerate(ep) if e["terminated"]]


def corr_gamma_delay(r, idxs):
    """'gamma*delay' of the drawn plants of the given seed indices."""
    return [r["draws"][i]["gamma"] * r["draws"][i]["delay"] * 1e-3 for i in idxs]


TABLES = {"seeds": table_seeds, "main": table_main, "ood": table_ood, "delay": table_delay,
          "safety": table_safety, "cost": table_cost, "growth": table_growth}


def paired_p(r, section, a, b, cond=None, key="return"):
    """Wilcoxon signed-rank p-value of controller a vs b on the same seeds."""
    sec = r[section]
    ea = sec[a][cond]["episodes"] if cond else sec[a]["episodes"]
    eb = sec[b][cond]["episodes"] if cond else sec[b]["episodes"]
    x = np.array([e[key] for e in ea]); y = np.array([e[key] for e in eb])
    if np.allclose(x, y):
        return 1.0
    return float(wilcoxon(x, y).pvalue)


def pval(p):
    return "p < 0,001" if p < 1e-3 else f"p = {pm(p, '.3f')}"


_PROSE = [
    (r"\bl_i\b", r"$l_i$"), (r"β_p", r"$\beta_p$"), (r"β_N", r"$\beta_N$"),
    (r"j_φ", r"$j_\varphi$"), (r"\bI_p\b", r"$I_p$"),
    (r"\bI_max\b", r"$I_{\max}$"), (r"\bV_max\b", r"$V_{\max}$"),
    (r"\bR₀\b", r"$R_0$"), (r"\bB₀\b", r"$B_0$"), (r"\bq₉₅\b", r"$q_{95}$"),
    (r"\be_pos\b", r"$e_{\mathrm{pos}}$"), (r"\be_sh\b", r"$e_{\mathrm{sh}}$"),
    (r"\bZ_ref\b", r"$Z_{\mathrm{ref}}$"), (r"\bZ_c\b", r"$Z_c$"),
    (r"\bR_c\b", r"$R_c$"), (r"\bI_max\b", r"$I_{\max}$"),
]


def prose_fix(text):
    """Subscripted symbols typed as plain text become equations -- outside
    existing math, so nothing already in $...$ is touched."""
    parts = re.split(r"(\$\$.*?\$\$|\$[^$\n]*?\$)", text, flags=re.S)
    for i in range(0, len(parts), 2):
        for pat, rep in _PROSE:
            parts[i] = re.sub(pat, lambda m, rep=rep: rep, parts[i])
    for i in range(1, len(parts), 2):
        parts[i] = _bars(parts[i])
    return "".join(parts)


def _bars(m):
    """A lone '|' in a Word equation is mis-imported by LibreOffice (as a
    logical or); delimiters and the conditioning bar are spelled so that
    both Word and LibreOffice draw them."""
    m = m.replace(r"\cdot|", r"\cdot\mid ")
    m = re.sub(r"(\\mathbf\{A\}(?:_t)?)\|(\\mathbf\{S\}(?:_t)?)",
               lambda k: k.group(1) + r"\mid " + k.group(2), m)
    m = re.sub(r"(?<!\\)\|([^|$]+?)(?<!\\)\|",
               lambda k: r"\left|" + k.group(1) + r"\right|", m)
    return m


# --- macro expansion -------------------------------------------------------------------------------
def expand(text, r):
    def table(m):
        return TABLES[m.group(1)](r)
    text = re.sub(r"\{\{table:(\w+)\}\}", table, text)

    def expr(m):
        env = {"r": r, "pm": pm, "pct": pct, "paired_p": paired_p,
               "pval": pval, "np": np, "NAMES": NAMES,
               "seed_stat": seed_stat, "fail_all": fail_all,
               "sl": sl, "smean": smean, "limit_ratio_range": limit_ratio_range,
               "draw_text": draw_text, "lost_by": lost_by,
               "corr_gamma_delay": corr_gamma_delay}
        return str(eval(m.group(1), env))
    text = re.sub(r"\{\{=\s*(.+?)\s*\}\}", expr, text)

    def val(m):
        v = lookup(r, m.group(1))
        fmt = m.group(2)
        return pm(v, fmt) if fmt else str(v)
    return re.sub(r"\{\{([\w\.]+)(?:\|([^}]*))?\}\}", val, text)


def number_citations(text):
    order = []

    def sub(m):
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
        nums = []
        for k in keys:
            if k not in REFS:
                raise KeyError(f"unknown reference key {k!r}")
            if k not in order:
                order.append(k)
            nums.append(order.index(k) + 1)
        return "[" + ", ".join(str(n) for n in sorted(set(nums))) + "]"
    text = re.sub(r"\[(@[\w]+(?:\s*;\s*@[\w]+)*)\]", sub, text)
    return text, order


# --- reference document and post-processing ------------------------------------------------------
def make_reference(path):
    subprocess.run([PANDOC, "-o", str(path), "--print-default-data-file",
                    "reference.docx"], check=True)
    d = Document(str(path))
    TNR = "Times New Roman"

    def font(st, size, bold=None, italic=None):
        st.font.name = TNR
        st.font.size = Pt(size)
        rpr = st.element.get_or_add_rPr()
        rf = rpr.find(qn("w:rFonts"))
        if rf is None:
            rf = OxmlElement("w:rFonts"); rpr.append(rf)
        for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            rf.set(qn(a), TNR)
        for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
            if rf.get(qn(a)) is not None:
                del rf.attrib[qn(a)]
        st.font.color.rgb = None
        if bold is not None:
            st.font.bold = bold
        if italic is not None:
            st.font.italic = italic

    def para(st, align=None, indent=None, space=(0, 0), spacing=1.5,
             keep_next=False):
        pf = st.paragraph_format
        if align is not None:
            pf.alignment = align
        pf.first_line_indent = Cm(indent) if indent is not None else None
        pf.space_before, pf.space_after = Pt(space[0]), Pt(space[1])
        pf.line_spacing = spacing
        pf.keep_with_next = keep_next

    styles = d.styles
    for name in ("Normal", "Body Text", "First Paragraph", "Compact"):
        if name in [s.name for s in styles]:
            font(styles[name], 14)
    para(styles["Normal"], WD_ALIGN_PARAGRAPH.JUSTIFY, 1.25)
    for name in ("Body Text", "First Paragraph"):
        para(styles[name], WD_ALIGN_PARAGRAPH.JUSTIFY, 1.25)
    para(styles["Compact"], WD_ALIGN_PARAGRAPH.LEFT, 0.0, spacing=1.0)
    styles["Compact"].font.size = Pt(11)
    for name, size, al in (("Heading 1", 14, WD_ALIGN_PARAGRAPH.CENTER),
                           ("Heading 2", 14, WD_ALIGN_PARAGRAPH.LEFT),
                           ("Heading 3", 14, WD_ALIGN_PARAGRAPH.LEFT)):
        font(styles[name], size, bold=True, italic=False)
        para(styles[name], al, 0.0 if al == WD_ALIGN_PARAGRAPH.CENTER else 1.25,
             space=(12, 6), keep_next=True)
    font(styles["Title"], 16, bold=True)
    para(styles["Title"], WD_ALIGN_PARAGRAPH.CENTER, 0.0, space=(0, 12),
         spacing=1.15)
    for name in ("Subtitle", "Author", "Date", "Abstract"):
        if name in [s.name for s in styles]:
            font(styles[name], 14)
            para(styles[name], WD_ALIGN_PARAGRAPH.CENTER, 0.0, space=(0, 10),
                 spacing=1.15)
    styles["Subtitle"].font.italic = False
    styles["Subtitle"].font.size = Pt(13)
    font(styles["Image Caption"], 12, italic=False)
    para(styles["Image Caption"], WD_ALIGN_PARAGRAPH.CENTER, 0.0, space=(3, 10),
         spacing=1.15)
    for name in ("Captioned Figure", "Figure"):
        if name in [s.name for s in styles]:
            para(styles[name], WD_ALIGN_PARAGRAPH.CENTER, 0.0, spacing=1.0,
                 keep_next=True)
    font(styles["Bibliography"], 14)
    para(styles["Bibliography"], WD_ALIGN_PARAGRAPH.JUSTIFY, 1.25, spacing=1.15)

    s = d.sections[0]
    s.page_width, s.page_height = Cm(21.0), Cm(29.7)
    s.top_margin, s.bottom_margin = Cm(2), Cm(2)
    s.left_margin, s.right_margin = Cm(3), Cm(1.5)
    # page number, bottom centre
    p = s.footer.paragraphs[0] if s.footer.paragraphs else s.footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for el in ("begin", None, "end"):
        r = p.add_run()
        r.font.name = TNR; r.font.size = Pt(12)
        if el is None:
            it = OxmlElement("w:instrText")
            it.set(qn("xml:space"), "preserve"); it.text = " PAGE "
            r._r.append(it)
        else:
            fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), el)
            r._r.append(fc)
    d.save(str(path))


def fix_math_props(path):
    """OMML allows either <m:nor/> or <m:sty/> in a run's properties, not both;
    pandoc writes both for \\text{...}."""
    d = Document(str(path))
    for rpr in d.element.body.iter(qn("m:rPr")):
        if rpr.find(qn("m:nor")) is not None:
            for sty in rpr.findall(qn("m:sty")):
                rpr.remove(sty)
    d.save(str(path))


def fix_section(path):
    """pgMar needs header/footer distances to validate against the schema."""
    d = Document(str(path))
    for sec in d.sections:
        pg = sec._sectPr.find(qn("w:pgMar"))
        if pg is not None:
            for k, v in (("w:header", 709), ("w:footer", 709), ("w:gutter", 0)):
                if pg.get(qn(k)) is None:
                    pg.set(qn(k), str(v))
    d.save(str(path))


def set_cell_borders(cell, **kw):
    tcPr = cell._tc.get_or_add_tcPr()
    b = tcPr.find(qn("w:tcBorders"))
    if b is None:
        b = OxmlElement("w:tcBorders"); tcPr.append(b)
    for edge in ("top", "left", "bottom", "right"):
        el = b.find(qn(f"w:{edge}"))
        if el is None:
            el = OxmlElement(f"w:{edge}"); b.append(el)
        v = kw.get(edge)
        if v:
            el.set(qn("w:val"), "single"); el.set(qn("w:sz"), str(v))
            el.set(qn("w:color"), "000000")
        else:
            el.set(qn("w:val"), "nil")


def postprocess(path):
    d = Document(str(path))
    body = d.element.body
    TNR = "Times New Roman"

    # 1. numbered display equations -> borderless 3-column tables
    paras = [p for p in d.paragraphs
             if p._p.find(".//" + qn("m:oMathPara")) is not None
             and not p.text.strip()]
    for n, p in enumerate(paras, 1):
        math = p._p.find(".//" + qn("m:oMathPara"))
        tbl = d.add_table(rows=1, cols=3)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False
        widths = (Cm(1.5), Cm(13.5), Cm(1.5))
        # the grid, not the cell widths, is what LibreOffice lays out by
        for gc, w in zip(tbl._tbl.tblGrid.findall(qn("w:gridCol")), widths):
            gc.set(qn("w:w"), str(int(w.twips)))
        tw = OxmlElement("w:tblW")
        tw.set(qn("w:w"), str(int(sum(w.twips for w in widths))))
        tw.set(qn("w:type"), "dxa")
        old = tbl._tbl.tblPr.find(qn("w:tblW"))
        if old is not None:
            tbl._tbl.tblPr.remove(old)
        style = tbl._tbl.tblPr.find(qn("w:tblStyle"))
        if style is not None:               # schema order: tblStyle, tblW, jc
            style.addnext(tw)
        else:
            tbl._tbl.tblPr.insert(0, tw)
        for c, w in zip(tbl.rows[0].cells, widths):
            c.width = w
            set_cell_borders(c)
        mid = tbl.rows[0].cells[1]
        mp = mid.paragraphs[0]
        mp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        mp.paragraph_format.first_line_indent = Cm(0)
        mp._p.append(copy.deepcopy(math))
        num = tbl.rows[0].cells[2].paragraphs[0]
        num.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        num.paragraph_format.first_line_indent = Cm(0)
        r = num.add_run(f"({n})"); r.font.name = TNR; r.font.size = Pt(14)
        for c in tbl.rows[0].cells:
            cp = c.paragraphs[0]
            cp.paragraph_format.line_spacing = 1.0
            cp.paragraph_format.space_after = Pt(0)
            tcv = c._tc.get_or_add_tcPr()
            va = OxmlElement("w:vAlign"); va.set(qn("w:val"), "center")
            tcv.append(va)
        p._p.addprevious(tbl._tbl)
        p._p.getparent().remove(p._p)

    # 2. data tables: three-line (booktabs-like) style, smaller font
    for t in d.tables:
        if len(t.columns) == 3 and not t.rows[0].cells[0].text.strip() \
                and not t.rows[0].cells[1].text.strip() is False:
            pass
        first = t.rows[0].cells[0]
        if len(t.columns) == 3 and first.text.strip() == "" \
                and any(c.paragraphs[0]._p.find(".//" + qn("m:oMath")) is not None
                        for c in t.rows[0].cells):
            continue  # an equation table
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        n = len(t.rows)
        for i, row in enumerate(t.rows):
            for c in row.cells:
                set_cell_borders(c, top=12 if i == 0 else (6 if i == 1 else 0),
                                 bottom=12 if i == n - 1 else (6 if i == 0 else 0))
                for cp in c.paragraphs:
                    cp.paragraph_format.first_line_indent = Cm(0)
                    cp.paragraph_format.line_spacing = 1.0
                    cp.paragraph_format.space_after = Pt(1)
                    for rn in cp.runs:
                        rn.font.name = TNR; rn.font.size = Pt(10.5)
                        if i == 0:
                            rn.font.bold = True
        # keep the header row on every page
        trPr = t.rows[0]._tr.get_or_add_trPr()
        h = OxmlElement("w:tblHeader"); h.set(qn("w:val"), "true"); trPr.append(h)
        for row in t.rows:
            trPr = row._tr.get_or_add_trPr()
            cs = OxmlElement("w:cantSplit"); cs.set(qn("w:val"), "true")
            trPr.append(cs)

    # 3. table captions: keep with the table, no indent
    for p in d.paragraphs:
        if re.match(r"^\d+-кесте\.", p.text.strip()):
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.keep_with_next = True
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.line_spacing = 1.15
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        if p.style.name in ("Image Caption", "Captioned Figure", "Figure"):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Cm(0)
    d.save(str(path))


def main(results, out):
    r = json.load(open(results))
    front = (HERE / "part0_front.md").read_text()
    parts = [front] + [(HERE / f).read_text() for f in
                       ("part1_intro_lit.md", "part2_methods.md",
                        "part3_results.md", "part4_discussion.md")]
    text = "\n\n".join(parts)
    text = expand(text, r)
    text = prose_fix(text)
    text = re.sub(r"\\tag\{\d+\}", "", text)
    text, order = number_citations(text)
    text += "\n\n## Пайдаланылған әдебиеттер тізімі\n\n"
    text += "\n".join(f"{i}. {REFS[k]}\n" for i, k in enumerate(order, 1))
    md = HERE / "article.built.md"
    md.write_text(text)

    ref = HERE / "reference.docx"
    make_reference(ref)
    subprocess.run([PANDOC, str(md), "-f",
                    "markdown+tex_math_dollars+pipe_tables+implicit_figures"
                    "+fenced_divs+raw_attribute-smart-auto_identifiers",
                    "-o", str(out), "--reference-doc", str(ref),
                    "--shift-heading-level-by=-1",
                    "--resource-path", str(HERE)], check=True)
    postprocess(out)
    fix_section(out)
    fix_math_props(out)
    print("references:", len(order), "->", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
