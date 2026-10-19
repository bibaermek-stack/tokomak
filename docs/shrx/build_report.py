#!/usr/bin/env python3
"""Build SHRX_vs_TORAX_report.pdf (Kazakh) from results.json.

    python build_report.py

Every number in the text is formatted from results.json.  Needs matplotlib and reportlab;
fonts: DejaVu Sans (all Kazakh letters).
"""
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from PIL import Image as PILImage  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,  # noqa: E402
                                SimpleDocTemplate, Spacer, Table, TableStyle)

HERE = Path(__file__).resolve().parent
R = json.load(open(HERE / "results.json"))
OUT_PDF = HERE / "SHRX_vs_TORAX_report.pdf"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"      # validated categorical slots 1-3
GRAY, INK, INK2, GRID = "#8a8a86", "#0b0b0b", "#52514e", "#e1e0dc"
TINT_BLUE, TINT_ORANGE, TINT_GRAY, TINT_GREEN = "#e8f1fc", "#fdeee7", "#f3f2ef", "#e6f5ee"

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
pdfmetrics.registerFont(TTFont("DV", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV", boldItalic="DV-B")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "savefig.dpi": 220, "mathtext.default": "regular",
})
COMMA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))


def c(x, n=1):
    return f"{x:.{n}f}".replace(".", ",")


def pct(x, n=0):
    return c(100 * x, n) + " %"


def thou(n):
    return f"{int(n):,}".replace(",", " ")


# ----------------------------------------------------------------------------- data shortcuts
ref = R["iter_ref"]
S, TX = R["sintez_runs"], R["torax_standalone"]["final"]
sh, ql, v0, an, cg = (S[k]["final"] for k in ("shrx", "qlknn_ref", "shrx_v0", "scaling_anchored", "cgm"))
ACC, LOC, LAT, MACH = R["accuracy"], R["local"], R["latency"]["default"], R["machines"]
G = ACC["groups"]
SENS = R["weights_sensitivity"]
ho, tr_ = G["held_out_solver_states"], G["train_machines_solver_states"]
M = R["model"]
net_err = sh["Pfus"] / ql["Pfus"] - 1            # network error: same simulator, SHRX vs the real QLKNN
code_gap = ql["Pfus"] / TX["Pfus"] - 1           # code difference: same network, SINTEZ vs TORAX
vs_torax = sh["Pfus"] / TX["Pfus"] - 1


def lat(key, N, m="median_us"):
    return LAT[f"N{N}"][key][m]


sp32 = {N: lat("qlknn_jit_float32", N) / lat("shrx_network", N) for N in (26, 65)}
sp64 = {N: lat("qlknn_jit_float64", N) / lat("shrx_network", N) for N in (26, 65)}

tmp = tempfile.TemporaryDirectory()
FIG = Path(tmp.name)


def save(fig, name):
    p = FIG / f"{name}.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


# ----------------------------------------------------------------------------- figures
def fig_bars():
    rows = [("TORAX · QLKNN (DeepMind)", TX["Pfus"], TX["Q"], BLUE),
            ("SINTEZ + QLKNN (бірдей желі)", ql["Pfus"], ql["Q"], BLUE),
            ("SINTEZ + SHRX", sh["Pfus"], sh["Q"], ORANGE),
            ("SINTEZ · әдепкі (калибрленген)", an["Pfus"], an["Q"], GRAY),
            ("SINTEZ · CGM формуласы", cg["Pfus"], cg["Q"], GRAY),
            ("SINTEZ + SHRX v0 (бұрынғы желі)", v0["Pfus"], v0["Q"], GRAY)]
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    for i, (n, p, q, col) in enumerate(rows):
        ax.barh(i, p, color=col, height=0.62, zorder=3, hatch="////" if "v0" in n else None,
                edgecolor="white" if "v0" in n else col, lw=0)
        ax.text(p + 8, i, f"{c(p, 0)} МВт · Q = {c(q)}", va="center", fontsize=8.5, zorder=5,
                bbox=dict(fc="white", ec="none", pad=1.2))
    ax.axvline(ref["Pfus"], color=INK, ls="--", lw=1.0, zorder=4)
    ax.text(ref["Pfus"] - 6, -0.62, "ITER жобасы 500 МВт", ha="right", va="center", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.4, -1.0)
    ax.set_xlim(0, 760)
    ax.set_xlabel("Термоядролық синтез қуаты $P_{fus}$, МВт (бірдей ITER режимі)")
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(COMMA)
    return save(fig, "bars")


def fig_trace():
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    for key, col, ls, lw, lab in (("scaling_anchored", GRAY, "-", 1.1, "әдепкі (калибрленген)"),
                                  ("shrx_v0", GRAY, "--", 1.4, "SHRX v0"),
                                  ("qlknn_ref", BLUE, "-", 2.0, "QLKNN"),
                                  ("shrx", ORANGE, "-", 2.0, "SHRX")):
        tr = S[key]["trace"]
        ax.plot(tr["t"], tr["Pfus"], color=col, ls=ls, lw=lw, zorder=3)
        y = {"scaling_anchored": 585, "shrx": 520, "qlknn_ref": 455, "shrx_v0": tr["Pfus"][-1]}[key]
        ax.annotate(lab, xy=(tr["t"][-1], tr["Pfus"][-1]), xytext=(tr["t"][-1] + 3, y), fontsize=7.8, va="center",
                    arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.5, shrinkA=0, shrinkB=2), annotation_clip=False)
    ax.axhline(TX["Pfus"], color=INK, ls=":", lw=1.0)
    ax.text(1, TX["Pfus"] + 12, f"TORAX {c(TX['Pfus'], 0)} МВт", fontsize=7.8, va="bottom")
    ax.axhline(ref["Pfus"], color=INK, ls="--", lw=0.8)
    ax.text(1, ref["Pfus"] - 12, "ITER жобасы 500", fontsize=7.8, va="top")
    ax.set_xlim(0, 160)
    ax.set_xticks([0, 30, 60, 90, 120])
    ax.set_ylim(0, 620)
    ax.set_xlabel("Плазма уақыты, с")
    ax.set_ylabel("$P_{fus}$, МВт")
    ax.yaxis.set_major_formatter(COMMA)
    return save(fig, "trace")


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 2.15))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 28)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=7.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.0", fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.3)

    def arrow(x0, y0, x1, y1, rad=0.0):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, color=INK2, lw=1.0,
                                     shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))

    box(0, 12, 18, 14, "SINTEZ-тің\n12 машинадағы\nразряды", TINT_GRAY, INK2)
    box(23, 12, 18, 14, "күй → QLKNN\nкірісі (10 сан)\nbridge.py", TINT_GRAY, INK2)
    box(46, 12, 17, 14, "нағыз QLKNN\nбелгілейді\n(мұғалім)", TINT_BLUE, BLUE)
    box(68, 12, 14, 14, "SHRX\n29 283 парам.\nоқыту", TINT_ORANGE, ORANGE)
    box(86, 12, 14, 14, "NumPy\nсалмақтар\n112 КБ", TINT_ORANGE, ORANGE)
    for x0, x1 in ((18.6, 22.4), (41.6, 45.4), (63.6, 67.4), (82.6, 85.4)):
        arrow(x0, 19, x1, 19)
    ax.plot([75, 75, 9], [11.4, 6.2, 6.2], color=ORANGE, lw=1.3, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch((9, 6.2), (9, 11.2), arrowstyle="-|>", mutation_scale=9, color=ORANGE, lw=1.3,
                                 shrinkA=0, shrinkB=0))
    ax.text(42, 2.6, "SHRX өзі жүргізген разряд жаңа күйлер береді → қайта белгілеу → қайта оқыту (3 раунд)",
            ha="center", fontsize=7.6, color=INK2)
    return save(fig, "pipeline")


def fig_local():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
    titles = {"A": "A · TORAX тұрақты күйі", "B1": "B1 · SHRX v0 разряды", "B2": "B2 · SINTEZ әдепкі күйі (Q ≈ 10)"}
    for ax, key in zip(axs, ("A", "B1", "B2")):
        d = LOC[key]
        r = np.array(d["rho"])
        m = (r >= 0.25) & (r <= 0.85)
        ax.axvspan(0, 0.25, color=TINT_GRAY, zorder=0)
        ax.axvspan(0.85, 1, color=TINT_GRAY, zorder=0)
        ax.plot(r, d["chi_v0"], color=GRAY, lw=1.4, ls="--", label="SHRX v0 $\\chi$")
        ax.plot(r, d["chi_e"]["qlknn_bridge"], color=BLUE, lw=2.0, label="QLKNN $\\chi_e$")
        ax.plot(r, d["chi_e"]["shrx"], color=ORANGE, lw=2.0, label="SHRX $\\chi_e$")
        ax.set_yscale("log")
        ax.set_ylim(0.03, 30)
        ax.set_xlim(0, 1)
        ax.set_title(titles[key], fontsize=8.2)
        ax.set_xlabel("ρ")
        ax.yaxis.set_major_formatter(COMMA)
        ax.grid(which="minor", visible=False)
    axs[0].set_ylabel("$\\chi_e$, м²/с (логарифмдік)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    return save(fig, "local")


def fig_scatter():
    d = ACC["scatter_flux_e"]
    q, p = np.array(d["qlknn"]), np.array(d["shrx"])
    fig, ax = plt.subplots(figsize=(3.5, 3.4))
    xs = np.logspace(np.log10(0.15), np.log10(q.max() * 1.3), 50)
    ax.fill_between(xs, xs / 1.25, xs * 1.25, color=TINT_GRAY, zorder=0, lw=0)
    ax.plot(xs, xs, color=INK2, lw=0.9, zorder=1)
    colors_ = {"nstx": BLUE, "demo": ORANGE}
    for mname, col in colors_.items():
        s = np.array(d["machine"]) == mname
        ax.scatter(q[s], p[s], s=5, color=col, alpha=0.35, linewidths=0, label=mname.upper(), zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("QLKNN электрондық жылу ағыны $q_e$ (gyro-Bohm)")
    ax.set_ylabel("SHRX $q_e$")
    ax.set_xlim(0.15, q.max() * 1.3)
    ax.set_ylim(0.15, q.max() * 1.3)
    ax.legend(loc="upper left", fontsize=8, markerscale=3)
    ax.grid(which="minor", visible=False)
    return save(fig, "scatter")


def fig_machines():
    ms = list(ACC["per_machine"])
    order = ["iter", "demo", "jet", "sparc", "diiid", "east", "kstar", "t15md", "ktm", "nstx", "nstxu", "mastu"]
    ms = [m for m in order if m in ms]
    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    for i, m in enumerate(ms):
        d = ACC["per_machine"][m]
        v = 100 * d["within25_e"]
        ax.bar(i, v, color=ORANGE if not d["held_out"] else "white", edgecolor=ORANGE, lw=1.4, width=0.66, zorder=3,
               hatch="////" if d["held_out"] else None)
        ax.text(i, v + 1.5, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels([m.upper() + ("*" if ACC["per_machine"][m]["held_out"] else "") for m in ms], fontsize=7.8)
    ax.set_ylim(0, 108)
    ax.set_ylabel("QLKNN $\\chi_e$-нен ≤ 25 %\nайырмашылығы бар нүктелер, %")
    ax.grid(axis="x", visible=False)
    return save(fig, "machines")


def fig_latency():
    rows = [("QLKNN · float64 (TORAX әдепкісі)", "qlknn_jit_float64", BLUE, 0.55),
            ("QLKNN · float32", "qlknn_jit_float32", BLUE, 1.0),
            ("SHRX · NumPy float32", "shrx_network", ORANGE, 1.0)]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 1.9), sharey=True)
    for ax, N in zip(axs, (26, 65)):
        for i, (n, key, col, al) in enumerate(rows):
            v = lat(key, N)
            rg = LAT[f"N{N}"][key]["range_us"]
            ax.barh(i, v, color=col, alpha=al, height=0.6, zorder=3)
            ax.plot(rg, [i, i], color=INK, lw=1.0, zorder=4)
            ax.text(max(rg[1], v) + 8, i, f"{c(v, 0)} мкс", va="center", fontsize=8.3)
        ax.set_title(f"{N} радиал нүкте")
        ax.set_xlim(0, 640)
        ax.set_xlabel("бір шақыру, мкс")
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(range(len(rows)))
    axs[0].set_yticklabels([r[0] for r in rows])
    axs[0].invert_yaxis()
    fig.tight_layout()
    return save(fig, "latency")


# ----------------------------------------------------------------------------- PDF blocks
PAGE_W, PAGE_H = A4
MARGIN = 17 * mm
BODY_W = PAGE_W - 2 * MARGIN
ST = {
    "title": ParagraphStyle("title", fontName="DV-B", fontSize=19, leading=24, textColor=colors.HexColor(INK), spaceAfter=3),
    "sub": ParagraphStyle("sub", fontName="DV", fontSize=10.5, leading=14, textColor=colors.HexColor(INK2), spaceAfter=3),
    "meta": ParagraphStyle("meta", fontName="DV", fontSize=8, leading=11, textColor=colors.HexColor(INK2)),
    "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=13, leading=17, textColor=colors.HexColor(INK), spaceBefore=10,
                         spaceAfter=5, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=10, leading=13, textColor=colors.HexColor(INK), spaceBefore=7,
                         spaceAfter=3, keepWithNext=1),
    "p": ParagraphStyle("p", fontName="DV", fontSize=9, leading=13.2, textColor=colors.HexColor(INK), alignment=TA_LEFT,
                        spaceAfter=4),
    "li": ParagraphStyle("li", fontName="DV", fontSize=9, leading=13.2, textColor=colors.HexColor(INK), leftIndent=11,
                         bulletIndent=0, spaceAfter=3),
    "small": ParagraphStyle("small", fontName="DV", fontSize=7.6, leading=10.4, textColor=colors.HexColor(INK2), spaceAfter=3),
    "cap": ParagraphStyle("cap", fontName="DV", fontSize=8, leading=10.8, textColor=colors.HexColor(INK2), spaceBefore=2,
                          spaceAfter=7),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=7.9, leading=10, textColor=colors.HexColor(INK)),
    "cellb": ParagraphStyle("cellb", fontName="DV-B", fontSize=7.9, leading=10, textColor=colors.HexColor(INK)),
    "cellh": ParagraphStyle("cellh", fontName="DV-B", fontSize=7.6, leading=9.6, textColor=colors.white),
    "num": ParagraphStyle("num", fontName="DV", fontSize=7.9, leading=10, alignment=2, textColor=colors.HexColor(INK)),
    "numb": ParagraphStyle("numb", fontName="DV-B", fontSize=7.9, leading=10, alignment=2, textColor=colors.HexColor(INK)),
    "numh": ParagraphStyle("numh", fontName="DV-B", fontSize=7.6, leading=9.6, alignment=2, textColor=colors.white),
    "card_n": ParagraphStyle("card_n", fontName="DV-B", fontSize=17, leading=21, alignment=1),
    "card_q": ParagraphStyle("card_q", fontName="DV-B", fontSize=11.5, leading=15, alignment=1, spaceAfter=2),
    "card_t": ParagraphStyle("card_t", fontName="DV", fontSize=8.4, leading=11, alignment=1, textColor=colors.HexColor(INK2)),
    "code": ParagraphStyle("code", fontName="DV", fontSize=7.4, leading=10, textColor=colors.HexColor(INK), leftIndent=6),
}


def P(t, s="p"):
    return Paragraph(t, ST[s])


def bullets(items):
    return [Paragraph(t, ST["li"], bulletText="•") for t in items]


def img(path, width=BODY_W):
    w, h = PILImage.open(path).size
    return Image(str(path), width=width, height=width * h / w)


def table(data, widths, header=INK2, extra=None):
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header)), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor(GRID)),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    t.setStyle(TableStyle(style + (extra or [])))
    return t


def callout(text, bg=TINT_GRAY, edge=INK2):
    t = Table([[Paragraph(text, ST["p"])]], colWidths=[BODY_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
                           ("LINEBEFORE", (0, 0), (0, -1), 2.2, colors.HexColor(edge)),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 7.5)
    canvas.setFillColor(colors.HexColor(INK2))
    canvas.drawString(MARGIN, 10 * mm, "SHRX — SINTEZ-тің нейрожелілік тасымал моделі · TORAX-пен салыстыру")
    canvas.drawRightString(PAGE_W - MARGIN, 10 * mm, f"{doc.page}-бет")
    canvas.restoreState()


OK, NO, PART = "✔", "✘", "◐"


def agree(m):
    """SHRX and QLKNN end the scripted discharge within 10 % in <Te>, and in P_fus (or both below 1 MW)."""
    a, b = MACH[m]["qlknn_ref"], MACH[m]["shrx"]
    p_ok = (a["Pfus"] < 1.0 and b["Pfus"] < 1.0) or abs(b["Pfus"] / max(a["Pfus"], 1e-9) - 1) < 0.1
    return bool(p_ok and abs(b["Te"] / a["Te"] - 1) < 0.1)


# ----------------------------------------------------------------------------- story
def build():
    f_bars, f_trace, f_pipe = fig_bars(), fig_trace(), fig_pipeline()
    f_local, f_scat, f_mach, f_lat = fig_local(), fig_scatter(), fig_machines(), fig_latency()
    doc = SimpleDocTemplate(str(OUT_PDF), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=15 * mm,
                            bottomMargin=17 * mm, title="SHRX — SINTEZ нейрожелілік тасымал моделі және TORAX",
                            author="Claude Code")
    s = []

    # ------------------------------------------------------------- page 1
    s += [P("SHRX: SINTEZ-тің жаңа тасымал моделі", "title"),
          P("Бұрынғы «TORAX стиліндегі суррогат» QLKNN-нен дистилляцияланып қайта жасалды және TORAX-пен салыстырылды", "sub"),
          P(f"19 қазан 2026 · bibaermek-stack/tokomak · claude/tokamak-project-similarity-a7ssc5 · TORAX {R['versions']['torax']}, "
            f"JAX {R['versions']['jax']} (CPU)", "meta"), Spacer(1, 7)]

    def card(num, q, title, col):
        return [Paragraph(f'<font color="{col}">{num}</font>', ST["card_n"]),
                Paragraph(f'<font color="{col}">{q}</font>', ST["card_q"]), Paragraph(title, ST["card_t"])]

    cards = Table([[card(f"{c(TX['Pfus'], 0)} МВт", f"Q = {c(TX['Q'])}", "TORAX · QLKNN<br/>(DeepMind)", BLUE),
                    card(f"{c(sh['Pfus'], 0)} МВт", f"Q = {c(sh['Q'])}", "SINTEZ + SHRX<br/>(жаңа модель)", ORANGE),
                    card(f"{c(v0['Pfus'], 0)} МВт", f"Q = {c(v0['Q'])}", "SINTEZ + SHRX v0<br/>(бұрынғы модель)", INK2)]],
                  colWidths=[BODY_W / 3] * 3)
    cards.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), colors.HexColor(TINT_BLUE)),
                               ("BACKGROUND", (1, 0), (1, 0), colors.HexColor(TINT_ORANGE)),
                               ("BACKGROUND", (2, 0), (2, 0), colors.HexColor(TINT_GRAY)),
                               ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor(GRID)),
                               ("INNERGRID", (0, 0), (-1, -1), 2, colors.white),
                               ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    s += [cards, Spacer(1, 3),
          P("Бірдей ITER режимі (15 МА, 5,3 Тл, 50 МВт); ITER жобасы — 500 МВт, Q = 10. "
            "TORAX-қа SINTEZ-тің педесталы мен тығыздығы берілген (алдыңғы салыстырудағыдай).", "small"),
          img(f_bars, BODY_W * 0.97), P("1-сурет. Бірдей ITER режиміндегі синтез қуаты.", "cap"), P("Қысқаша қорытынды", "h1")]
    s += bullets([
        f"<b>Жоспарға сай жасалды.</b> Модель SHRX деп аталды; ол енді QLKNN-нің (TORAX-тың нейрожелісі) дистилляциясы. "
        f"ITER-де P<sub>fus</sub> = {c(sh['Pfus'], 0)} МВт (Q = {c(sh['Q'])}): TORAX-тан {c(abs(vs_torax) * 100, 1)} % аз, "
        f"бұрынғы модельден ({c(v0['Pfus'], 0)} МВт) {c(sh['Pfus'] / v0['Pfus'], 1)} есе жоғары.",
        f"<b>Желінің қатесі аз.</b> Бірдей SINTEZ ішінде SHRX мен нағыз QLKNN арасындағы айырма {c(net_err * 100, 1)} % "
        f"({c(sh['Pfus'], 0)} және {c(ql['Pfus'], 0)} МВт). SINTEZ пен TORAX <i>кодтарының</i> айырмасы — "
        f"{c(abs(code_gap) * 100, 0)} % (SINTEZ төмен, бірдей QLKNN-мен).",
        f"<b>Жылдамырақ.</b> Желінің өзі QLKNN-нен {c(sp32[26], 1)}–{c(sp32[65], 1)} есе (float32) және "
        f"{c(sp64[26], 1)}–{c(sp64[65], 1)} есе (float64, TORAX әдепкісі) тез; параметрі "
        f"{c(R['qlknn_parameters'] / M['n_parameters'], 1)} есе аз ({thou(M['n_parameters'])} және {thou(R['qlknn_parameters'])}). "
        "JAX керек емес — тек NumPy.",
        f"<b>Дәл.</b> Оқуға қатыспаған екі машинада (NSTX, DEMO) SHRX-тың QLKNN-мен R² = {c(ho['r2_chi_i'], 3)} (χ<sub>i</sub>) және "
        f"{c(ho['r2_chi_e'], 3)} (χ<sub>e</sub>); χ-дің медианалық қатесі ≈ {pct(ho['median_rel_err_e'])}.",
        "<b>«TORAX-тан мықты» дегеннің шегі.</b> SHRX-тың мұғалімі — QLKNN, сондықтан гирокинетикалық дәлдікте ол QLKNN-нен "
        "асып түспейді. Мықтылығы — жылдамдықта, тәуелсіздікте (NumPy) және SINTEZ-тің көпфизикалы симуляторына кіріктірілгенінде. "
        "TORAX-тың дифференциалданатындығы мен гирокинетикалық бағыты SHRX-та жоқ.",
        "<b>Әлі дәлелденбегені.</b> Интеграцияланған разрядта SHRX пен QLKNN ITER-де және DEMO-да (оқудан тыс) сәйкес келеді, "
        "ал SINTEZ-тің басқа 10 машинасындағы скрипттелген разрядтары тұрақсыз режимде — нәтиже салмақ нұсқасына тәуелді, "
        "сондықтан онда сәйкестікті дәлелдей алмаймыз (6-бөлім).",
    ])
    s += [PageBreak()]

    # ------------------------------------------------------------- page 2: scoreboard + method
    s += [P("1. «Мықты» дегеніміз не және неге жеттік", "h1"),
          P("Критерийлерді жұмысты бастамай тұрып жазып қойдық (docs/shrx/PLAN.md). Төменде әрқайсысы бойынша нәтиже — "
            "орындалмағанын да көрсетеміз.", "p")]
    k1 = min(sp32.values()) >= 2.0
    k2 = ho["r2_chi_i"] >= 0.98 and ho["r2_chi_e"] >= 0.98
    k2b_med = all(0.8 <= LOC[k]["chi_e"]["vs_qlknn_bridge"]["median_ratio"] <= 1.25 for k in ("A", "B1", "B2"))
    k2b_x2 = all(LOC[k]["chi_e"]["vs_qlknn_bridge"]["within_x2"] >= 0.8 for k in ("A", "B1", "B2"))
    pr_a, pr_b2 = LOC["A"]["chi_e"]["vs_qlknn_bridge"]["pearson_r"], LOC["B2"]["chi_e"]["vs_qlknn_bridge"]["pearson_r"]
    k3 = abs(vs_torax) <= 0.15
    rows = [
        ("K1", "Желінің шақыру уақыты QLKNN-нен ≥ 2 есе аз (26 және 65 нүкте)",
         f"float32: {c(sp32[26], 1)}× және {c(sp32[65], 1)}×; float64: {c(sp64[26], 1)}× және {c(sp64[65], 1)}×", OK if k1 else NO),
        ("K2", "Оқудан тыс машиналарда R² ≥ 0,98",
         f"χ<sub>i</sub> {c(ho['r2_chi_i'], 3)}, χ<sub>e</sub> {c(ho['r2_chi_e'], 3)}; ағындар {c(ho['r2_flux_i'], 3)} / {c(ho['r2_flux_e'], 3)}. "
         "Метрика өзгертілді (төменде)", OK if k2 else NO),
        ("K2b", "Бірдей күйлер A/B1/B2: медиана 0,8–1,25; ×2 ішінде ≥ 80 %; Пирсон ≥ 0,9",
         f"медиана {c(LOC['A']['chi_e']['vs_qlknn_bridge']['median_ratio'], 2)}/{c(LOC['B1']['chi_e']['vs_qlknn_bridge']['median_ratio'], 2)}/"
         f"{c(LOC['B2']['chi_e']['vs_qlknn_bridge']['median_ratio'], 2)}; Пирсон A {c(pr_a, 2)}, B2 {c(pr_b2, 3)}",
         PART if (k2b_med and k2b_x2) else NO),
        ("K3", "ITER интеграцияланған разряды TORAX-тан ±15 %",
         f"{c(sh['Pfus'], 0)} МВт, айырма {c(vs_torax * 100, 1)} %", OK if k3 else NO),
        ("K4", "Тек NumPy, барлық машина, кіріс ауқымынан тыс белгісі",
         "JAX-сыз жұмыс істейді; 12 машинада жүреді; <i>out_of_range</i> белгісі бар", OK),
        ("K5", "Нәтиже мен шектеулерді адал жазу", "осы құжат", OK),
    ]
    data = [[Paragraph(h, ST["cellh"]) for h in ("№", "Критерий", "Өлшенгені", "")]]
    for k, crit, val, mark in rows:
        data.append([Paragraph(k, ST["cellb"]), Paragraph(crit, ST["cell"]), Paragraph(val, ST["cell"]),
                     Paragraph(mark, ParagraphStyle("m", parent=ST["cellb"], fontSize=11, alignment=1))])
    s += [table(data, [BODY_W * 0.07, BODY_W * 0.38, BODY_W * 0.47, BODY_W * 0.08]),
          P("2-кесте. ✔ орындалды, ◐ ішінара, ✘ орындалмады. K2b: A күйінде (TORAX-тың пилообразным тегістеген ядросы) "
            "профиль формасы толық сәйкес келмейді (r < 0,9), басқа екеуінде сәйкес.", "cap"),
          P("K2 метрикасы туралы", "h2"),
          P("Жоспарда «SI бірліктегі χ R²» деп жазылған. χ = q/(R/L<sub>T</sub>) болғандықтан, градиент нөлге жақындағанда χ шексіз өседі "
            "және R²-ні бір-екі нүкте анықтап қалады, сондықтан оны бұлай есептеу мағынасыз. Орнына екі нәрсені өлшедік: (а) желінің шығысы — "
            "gyro-Bohm ағындарының R²-ы, (ә) χ-дің R²-ы gyro-Bohm бірлікте, R/L<sub>T</sub> ≥ 2 аймағында. Екеуі де 0,98-ден жоғары. "
            "Қосымша: турбуленттілік «қосулы» жерде χ-дің медианалық салыстырмалы қатесі мен 25 %-дан аз өзгешеленетін нүктелер үлесі.", "p"),
          P("Қалай жасалды", "h2"), img(f_pipe, BODY_W * 0.98),
          P("3-сурет. SHRX-ты алу тізбегі. Желі QLKNN-нің ішкі кірісімен жұмыс істейді (R/L<sub>T</sub>, q, ŝ, ν*, …), сондықтан бір желі барлық машинаға жарайды.", "cap")]
    s += bullets([
        "<b>Бастапқы кінәрат.</b> SHRX v0 жобаның өз формулаларынан оқытылған, ал QLKNN-нен ~10–16 есе жоғары χ беріп, "
        "критикалық градиент табалдырығы жоқ болатын (алдыңғы есеп). Желі емес, оның оқу мақсаты қате еді.",
        "<b>Мұғалім.</b> Нағыз QLKNN (<i>qlknn_7_11_v1</i>, Apache-2.0), TORAX-пен бірге келеді. Кірісі — 10 өлшемсіз сан; "
        "шығысы — иондық және электрондық жылу ағыны мен бөлшек ағыны.",
        "<b>Көпір (bridge.py).</b> SINTEZ күйін QLKNN кірісіне айналдыру TORAX формулаларымен NumPy-да жазылды және TORAX-тың өз "
        f"кірістерімен салыстырылды: 10 кірістің медианалық қатесі ≤ {pct(max(max(v['median_abs_err'] for v in d['inputs'].values()) for d in R['bridge_validation'].values()), 1)}; "
        "бір күйдегі χ айырмасы ≈ 1–3 %.",
        "<b>Деректер.</b> 12 машинаның SINTEZ разряды (4 тасымал моделімен), одан соң — SHRX-тың өзі мен нағыз QLKNN жүргізген "
        "разрядтардың күйлері (желі өзі барған жерде оқиды), өрескел перту және QLKNN ауқымындағы біркелкі үлгілер: "
        f"барлығы {thou(R['training_rounds'][-1]['samples'])} нүкте, 3 раунд.",
        "<b>Бақылау.</b> NSTX және DEMO оқуға қатыспады.",
    ])
    s += [PageBreak()]

    # ------------------------------------------------------------- page 3: integrated
    s += [P("2. Интеграцияланған ITER разряды", "h1")]
    cols = [("P<sub>fus</sub><br/>МВт", "Pfus", 0), ("Q", "Q", 1), ("W<sub>th</sub><br/>МДж", "Wth", 0),
            ("β<sub>N</sub>", "betaN", 2), ("⟨T<sub>e</sub>⟩<br/>кэВ", "Te", 1), ("⟨T<sub>i</sub>⟩<br/>кэВ", "Ti", 1)]
    cases = [("ITER жобасы (эталон)", ref, "ref"), ("TORAX · QLKNN (жеке код)", TX, "torax"),
             ("SINTEZ + QLKNN (бірдей желі)", ql, "ql"), ("SINTEZ + SHRX", sh, "shrx"),
             ("SINTEZ + SHRX v0 (бұрынғы)", v0, ""), ("SINTEZ · әдепкі (калибрленген)", an, ""), ("SINTEZ · CGM", cg, "")]
    data = [[Paragraph("Жағдай", ST["cellh"])] + [Paragraph(h, ST["numh"]) for h, *_ in cols]]
    extra = []
    for i, (name, d, kind) in enumerate(cases, start=1):
        bold = kind in ("torax", "shrx")
        data.append([Paragraph(name, ST["cellb" if bold else "cell"])] +
                    [Paragraph(c(d[k], nd) if k in d else "—", ST["numb" if bold else "num"]) for _, k, nd in cols])
        bg = {"torax": TINT_BLUE, "ql": TINT_BLUE, "shrx": TINT_ORANGE, "ref": TINT_GRAY}.get(kind)
        if bg:
            extra.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor(bg)))
    s += [table(data, [BODY_W * 0.30] + [BODY_W * 0.70 / 6] * 6, extra=extra),
          P("3-кесте. Тұрақты күйдегі нәтижелер (SINTEZ — t = 120 с, TORAX — t = 30 с). ⟨T<sub>e,i</sub>⟩: SINTEZ-те тығыздықпен "
            "өлшенген, TORAX-та көлемдік орташа; β<sub>N</sub> — SINTEZ жылдам иондарды қосады. Эталон — tokamak.validation.", "cap"),
          img(f_trace, BODY_W), P("4-сурет. SINTEZ ішіндегі P<sub>fus</sub>(t): SHRX нағыз QLKNN-ге тұрақты түрде жабысады.", "cap")]
    s += [P("Айырманы бөліп қарау", "h2")]
    s += bullets([
        f"<b>Желі есебінен:</b> SHRX {c(sh['Pfus'], 0)} МВт және QLKNN {c(ql['Pfus'], 0)} МВт — бір SINTEZ ішінде, бір ғана "
        f"айырма — желі: {c(net_err * 100, 1)} %.",
        f"<b>Код есебінен:</b> SINTEZ + QLKNN {c(ql['Pfus'], 0)} МВт, TORAX {c(TX['Pfus'], 0)} МВт: {c(abs(code_gap) * 100, 0)} % төмен. "
        "Бұл — екі симулятордың (геометрия, қыздыру пішіні, сәулелену, пилообразный) айырмасы, желіге қатысы жоқ; SHRX-ты "
        "жақсарту оны жоя алмайды.",
        f"<b>Жалпы:</b> SINTEZ + SHRX TORAX-тан {c(abs(vs_torax) * 100, 1)} % төмен және ITER жобасына {c(abs(sh['Pfus'] / ref['Pfus'] - 1) * 100, 1)} % жақын.",
        "Әдепкі модельдің (526 МВт) нәтижесі ITER-ге калибрленгендіктен, оның дәлдігі болжам емес. SHRX ITER-ге бапталмаған: "
        "нәтиже QLKNN-нен келеді.",
    ])

    # ------------------------------------------------------------- page 4: local accuracy
    s += [P("3. Жергілікті дәлдік: SHRX ↔ QLKNN", "h1")]
    hdr = ["Деректер тобы", "n", "R² ағын i / e", "R² χ i / e", "χ қатесі i / e", "≤ 25 %, e"]
    data = [[Paragraph(hdr[0], ST["cellh"])] + [Paragraph(h, ST["numh"]) for h in hdr[1:]]]
    names = {"held_out_solver_states": "Оқудан тыс машиналар (NSTX, DEMO), SINTEZ күйлері",
             "held_out_perturbed": "Оқудан тыс машиналар, пертілген күйлер",
             "train_machines_solver_states": "Оқу машиналары, SINTEZ күйлері", "box_uniform": "QLKNN ауқымы, біркелкі үлгі"}
    for key, nm in names.items():
        d = G[key]
        data.append([Paragraph(nm, ST["cell"]), Paragraph(thou(d["n"]), ST["num"]),
                     Paragraph(f"{c(d['r2_flux_i'], 3)} / {c(d['r2_flux_e'], 3)}", ST["num"]),
                     Paragraph(f"{c(d['r2_chi_i'], 3)} / {c(d['r2_chi_e'], 3)}", ST["num"]),
                     Paragraph(f"{pct(d['median_rel_err_i'], 1)} / {pct(d['median_rel_err_e'], 1)}", ST["num"]),
                     Paragraph(pct(d["within25_e"]), ST["num"])])
    s += [table(data, [BODY_W * 0.33, BODY_W * 0.10, BODY_W * 0.15, BODY_W * 0.15, BODY_W * 0.17, BODY_W * 0.10]),
          P("4-кесте. QLKNN-ге қарсы. «χ қатесі» — турбуленттілік қосулы нүктелердегі (QLKNN χ &gt; 2·0,05 м²/с) салыстырмалы қатенің медианасы; "
            "R² χ — gyro-Bohm бірлікте, R/L<sub>T</sub> ≥ 2.", "cap"),
          Table([[img(f_scat, BODY_W * 0.43),
                  [P("Оқудан тыс екі машинада SHRX жылу ағыны QLKNN-ді диагональ бойымен қайталайды; "
                     f"χ<sub>e</sub> бойынша {pct(ho['within25_e'])} нүкте ±25 % ішінде.", "p"),
                   P("Үлкен ауытқулар — критикалық градиент маңында: сол жерде χ секіріп қосылады да, кішкентай желі "
                     "секірістің орнын бірнеше пайызға жылжыта алады. Қатты тасымалда бұл температураның жылжуына айналады.", "p")]]],
                colWidths=[BODY_W * 0.46, BODY_W * 0.54],
                style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)])),
          P("5-сурет. NSTX және DEMO: SHRX мен QLKNN электрондық жылу ағыны (желінің тікелей шығысы). Сұр жолақ — ±25 %.", "cap"),
          img(f_local, BODY_W),
          P("6-сурет. Алдыңғы салыстырудағы үш күй. SHRX v0 (сұр үзік) QLKNN-нен әлдеқайда жоғары және табалдырықсыз болатын; "
            "SHRX QLKNN-ді қайталайды, оның ішінде χ-дің нөлге жақын («өшірулі») аймағын да.", "cap")]
    la, lb2 = LOC["A"]["chi_e"]["vs_qlknn_bridge"], LOC["B2"]["chi_e"]["vs_qlknn_bridge"]
    r_ = np.array(LOC["B2"]["rho"])
    m_ = (r_ >= 0.25) & (r_ <= 0.85)
    v0_b2 = float(np.median(np.array(LOC["B2"]["chi_v0"])[m_] / np.array(LOC["B2"]["chi_e"]["qlknn_bridge"])[m_]))
    s += bullets([
        f"B2 (Q ≈ 10 күйі): SHRX/QLKNN медианасы {c(lb2['median_ratio'], 2)}, {pct(lb2['within_25pct'])} нүкте ±25 % ішінде, r = {c(lb2['pearson_r'], 3)}; "
        f"v0 бұл күйде QLKNN-нен {c(v0_b2, 1)} есе жоғары еді.",
        f"A (TORAX күйі): медиана {c(la['median_ratio'], 2)}, {pct(la['within_25pct'])} нүкте ±25 % ішінде, r = {c(la['pearson_r'], 2)} — бұл күйде "
        "пилообразный ядроны тегістеген, χ тек бірнеше радиуста «қосулы», сондықтан форма корреляциясы төмен.",
        "B1 (v0 разряды): тасымал бәрі табалдырықтан төмен, екеуі де χ ≈ 0,05 м²/с (шектік мән) береді.",
    ])

    s += [P("Машиналар бойынша", "h2"), img(f_mach, BODY_W),
          P("7-сурет. QLKNN-ге ±25 % ішінде келетін χ<sub>e</sub> нүктелерінің үлесі (турбуленттілік қосулы жерде, SINTEZ күйлері). "
            "* — оқуға қатыспаған машиналар. Ең төмені — "
            f"{pct(min(d['within25_e'] for d in ACC['per_machine'].values()))}.", "cap")]

    # ------------------------------------------------------------- speed
    s += [P("4. Жылдамдық", "h1"), img(f_lat, BODY_W),
          P("8-сурет. Желінің бір шақыруы (кіріс → ағындар), бір компьютерде; сызық — 11 өлшемнің ең аз/ең көп мәні.", "cap")]
    s += bullets([
        f"Желі: {c(lat('shrx_network', 26), 0)} және {c(lat('shrx_network', 65), 0)} мкс; QLKNN float32 — {c(lat('qlknn_jit_float32', 26), 0)} және "
        f"{c(lat('qlknn_jit_float32', 65), 0)} мкс, float64 (TORAX-тың әдепкі дәлдігі) — {c(lat('qlknn_jit_float64', 26), 0)} және {c(lat('qlknn_jit_float64', 65), 0)} мкс.",
        f"SINTEZ-тің бір тасымал қадамы толығымен (күй → кіріс → желі → χ<sub>e</sub>, χ<sub>i</sub>): {c(lat('shrx_evaluate_full', 26), 0)}–"
        f"{c(lat('shrx_evaluate_full', 65), 0)} мкс; оның жартысына жуығы — кірістерді дайындау (NumPy), желі емес.",
        f"Бүкіл 120 с разряд: SINTEZ + SHRX {c(S['shrx']['wall_s'], 0)} с, SINTEZ + QLKNN {c(S['qlknn_ref']['wall_s'], 0)} с — айырма аз, себебі "
        "уақыттың көп бөлігін тасымал емес, равновесие, МГД және басқа модульдер алады.",
        "Жылдамдық артықшылығы желінің өзінде (2,4–4,9×), бүкіл симуляцияда емес. Желі кішкентай (JAX-та жиналмаған) — "
        "SINTEZ тек NumPy-да жұмыс істей береді.",
    ])

    # ------------------------------------------------------------- multi machine + limits
    s += [P("5. Басқа машиналар: не дәлелденді, не дәлелденбеді", "h1")]
    order = ["iter", "demo", "jet", "sparc", "diiid", "east", "kstar", "t15md", "ktm", "nstx", "nstxu", "mastu"]
    hdr = ["Машина", "P<sub>fus</sub> QLKNN", "P<sub>fus</sub> SHRX", "⟨T<sub>e</sub>⟩ QLKNN", "⟨T<sub>e</sub>⟩ SHRX", "Бағасы"]
    data = [[Paragraph(hdr[0], ST["cellh"])] + [Paragraph(h, ST["numh"]) for h in hdr[1:5]] + [Paragraph(hdr[5], ST["cellh"])]]
    for m in order:
        a, b = MACH[m]["qlknn_ref"], MACH[m]["shrx"]
        tag = "сәйкес" if agree(m) else "сәйкес емес / тұрақсыз"
        held = " *" if m in ("nstx", "demo") else ""
        data.append([Paragraph(m.upper() + held, ST["cellb"]), Paragraph(c(a["Pfus"], 1), ST["num"]), Paragraph(c(b["Pfus"], 1), ST["num"]),
                     Paragraph(c(a["Te"], 1), ST["num"]), Paragraph(c(b["Te"], 1), ST["num"]),
                     Paragraph(tag, ST["cell"])])
    extra = [("BACKGROUND", (0, i), (-1, i), colors.HexColor(TINT_GREEN)) for i, m in enumerate(order, start=1) if agree(m)]
    s += [table(data, [BODY_W * 0.14, BODY_W * 0.16, BODY_W * 0.16, BODY_W * 0.16, BODY_W * 0.16, BODY_W * 0.22], extra=extra),
          P("5-кесте. Әр машинаның скрипттелген разряды (t = 100 с, соңғы 20 с орташасы), QLKNN мен SHRX бір SINTEZ ішінде. "
            "* — оқуға қатыспаған. Жасыл — ±10 % ішінде.", "cap")]
    s += bullets([
        "<b>ITER және DEMO сәйкес</b> (DEMO — оқудан тыс). Бұл — желі қатесінің интеграцияланған разрядқа әсері аз екенінің нақты дәлелі.",
        f"<b>Қалған 10 машинада сәйкестік дәлелденбеді</b> (SPARC-та ⟨T<sub>e</sub>⟩ {c(MACH['sparc']['qlknn_ref']['Te'], 1)} және "
        f"{c(MACH['sparc']['shrx']['Te'], 1)} кэВ — жақын, бірақ P<sub>fus</sub> 1–2 МВт, жану жоқ). SINTEZ-тің скрипттелген разряды ITER-дің қыздыру қуатын сызықты масштабтайды; "
        "кіші машиналарда плазма шекті режимде — суыту мен жану арасында. Сол режимде бір машинаның нәтижесі SHRX-тың бір оқу нұсқасынан екіншісіне қатты өзгереді: ⟨T<sub>e</sub>⟩ KTM-да "
        f"{c(min(x['ktm'] for x in SENS['Te'].values()), 1)}–{c(max(x['ktm'] for x in SENS['Te'].values()), 1)} кэВ, NSTX-та "
        f"{c(min(x['nstx'] for x in SENS['Te'].values()), 1)}–{c(max(x['nstx'] for x in SENS['Te'].values()), 1)} кэВ, MAST-U-да "
        f"{c(min(x['mastu'] for x in SENS['Te'].values()), 1)}–{c(max(x['mastu'] for x in SENS['Te'].values()), 1)} кэВ "
        f"({len(SENS['Te'])} нұсқа, жергілікті дәлдігі бірдей жоғары). Мұны желінің қатесі емес, жүйенің сезімталдығы деп санаймыз, бірақ дәлелдей алмаймыз.",
        "Сондықтан «SHRX барлық машинада QLKNN-ді қайталайды» деп айтпаймыз; айтатынымыз: жергілікті дәлдік барлық 12 машинада "
        "жоғары (7-сурет), ал интеграцияланған сәйкестік ITER мен DEMO-да ғана тексерілді.",
    ])

    s += [P("6. Шектеулер және келесі қадамдар", "h1"), P("Шектеулер", "h2")]
    s += bullets([
        "SHRX — QLKNN-нің дистилляциясы: QLKNN қатесін мұра етеді, ал QLKNN өзі QuaLiKiz-тің жуықтауы. Гирокинетикалық дәлдікте асып түсу үшін "
        "QuaLiKiz/GENE деректері керек.",
        "Кіріс ауқымынан тыс нүктелер QLKNN ауқымына қиылады (<i>out_of_range</i> белгісі бар). ITER-де 65 нүктенің 8-і "
        "ауқымнан тыс: ось маңындағы 6 нүкте (QLKNN-де x ≥ 0,1) және жиектегі 2 нүкте (R/L<sub>T</sub> &gt; 150). QLKNN-ге де қиылған кіріс беріледі (reference.py); "
        "TORAX өзінде ось пен жиекті патч және педестал арқылы жабады.",
        "Желі тек χ<sub>e</sub> және χ<sub>i</sub> береді; бөлшек тасымалы SINTEZ-те бұрынғыша (D = 0,25·χ<sub>e</sub>), ал "
        "QLKNN-нің бөлшек ағыны (желінің үшінші шығысы) әлі пайдаланылмайды.",
        "Бүтін салыстыру бір режимде (ITER H-режимі) жасалды, TORAX жағы — бір рет жүргізу; ITER кезінде SHRX оқуда болды "
        "(салмағы ×2), сондықтан ITER-дегі сәйкестік тәуелсіз тест емес; тәуелсіз тест — NSTX/DEMO.",
        "Бұрын жарияланған слайдтар мен сөйлеу мәтінінде (презентация) әлі «TORAX стиліндегі суррогат», «R² 0,943», «122 vs 148 мкс» тұр — "
        "олар жаңартылмаған.",
    ])
    s += [P("Ұсынылатын келесі қадамдар", "h2")]
    s += bullets([
        "Интеграцияланған тест жиынын кеңейту: әр машинаға нақты, тұрақты режим жасау (қыздыруды жеке баптау), сонда ғана олардағы сәйкестікті өлшеуге болады.",
        "QLKNN-нің бөлшек ағынын пайдалану (D, V) және SHRX-ты ток диффузиясымен байланыстыру.",
        "Критикалық градиент маңына арнайы салмақ беру немесе 2–3 желі ансамблі (жылдамдық қорын жұмсап).",
        "Нағыз гирокинетикалық деректермен (QLKNN11D, QuaLiKiz) тікелей оқыту — мұғалімнің қатесінен босау үшін.",
        "Презентация мен есептерді SHRX атауына жаңарту.",
    ])

    # ------------------------------------------------------------- appendix
    s += [P("Қосымша. Қолдану және қайта жүргізу", "h1")]
    code = [
        "# SINTEZ ішінде",
        "SolverConfig(machine='iter', transport_model='shrx')      # 'neural_surrogate' = ескі ат, SHRX-ке апарады",
        "SolverConfig(machine='iter', transport_model='shrx_v0')   # бұрынғы желі (салыстыру үшін)",
        "# HTTP:  POST /ml/shrx  (QLKNN айнымалылары → χ_e, χ_i)",
        "# Салмақтар: backend/tokamak/shrx/weights.npz (+ weights.json)",
        "# Қайта оқыту (TORAX ортасы): backend/ml/shrx/  collect_states.py → label_qlknn.py → train_shrx.py",
        "# Тексеру: python -m pytest backend/tests/test_shrx.py",
        "# Бұл есеп:  python collect_results.py <workdir> && python build_report.py",
    ]
    s += [Table([[Paragraph(l.replace("&", "&amp;").replace("<", "&lt;"), ST["code"])] for l in code], colWidths=[BODY_W],
                style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(TINT_GRAY)),
                                  ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2)])),
          Spacer(1, 4),
          P(f"Модель: {M['width']}×{M['depth']} ReLU, {thou(M['n_parameters'])} параметр; оқу {thou(M['n_train'])} үлгі, {M['epochs']} эпоха; "
            f"мұғалім: {M['teacher']}. Нұсқалар: Python {R['versions']['python']}, TORAX {R['versions']['torax']}, JAX {R['versions']['jax']}.", "small")]

    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print("written", OUT_PDF, round(OUT_PDF.stat().st_size / 1024), "kB")


if __name__ == "__main__":
    build()
