#!/usr/bin/env python3
"""Build SINTEZ_vs_TORAX_report.pdf (Kazakh) from results.json.

    python build_report.py

Every number in the text is formatted from results.json, so the PDF cannot drift from the data.
Needs matplotlib and reportlab; fonts: DejaVu Sans (has all Kazakh letters).
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
OUT_PDF = HERE / "SINTEZ_vs_TORAX_report.pdf"

# --------------------------------------------------------------------------- style
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"      # validated categorical slots 1-3
GRAY, INK, INK2, GRID = "#8a8a86", "#0b0b0b", "#52514e", "#e1e0dc"
TINT_BLUE, TINT_ORANGE, TINT_GRAY = "#e8f1fc", "#fdeee7", "#f3f2ef"

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
pdfmetrics.registerFont(TTFont("DV", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV", boldItalic="DV-B")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 220, "mathtext.default": "regular",
})
COMMA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))


def c(x, n=1):
    """Kazakh decimal comma."""
    return f"{x:.{n}f}".replace(".", ",")


def thou(n):
    """Integer with a space as thousands separator."""
    return f"{int(n):,}".replace(",", " ")


# --------------------------------------------------------------------------- data shortcuts
S, T, X = R["sintez"], R["torax"], R["cross"]
ref = R["iter_ref"]
tx, nn, an = T["main"]["final"], S["neural_surrogate"]["final"], S["scaling_anchored"]["final"]
cg, bg, gb = S["cgm"]["final"], S["bohm_gyrobohm"]["final"], S["gyrobohm"]["final"]
ns, rad = T["no_sawteeth"]["final"], T["more_radiation"]["final"]
lat, fid = R["latency"], R["fidelity_to_teacher"]
xa, xb1, xb2 = X["torax_state"], X["sintez_neural_surrogate"], X["sintez_scaling_anchored"]
mask_lo, mask_hi = R["cross_mask"]
rho_x = np.array(xa["rho"])
mk = (rho_x >= mask_lo) & (rho_x <= mask_hi)

# sanity gates: if the data stop supporting the story told below, fail loudly instead of printing it
assert abs(tx["Pfus"] / ref["Pfus"] - 1) < 0.10, "TORAX no longer within 10 % of the ITER design point"
assert nn["Pfus"] < 0.25 * tx["Pfus"], "neural surrogate no longer far below TORAX"
assert xa["metrics_vs_e"]["median_ratio"] > 3, "NN chi no longer well above QLKNN on the TORAX state"
assert xb1["metrics_vs_e"]["median_ratio"] > 3, "NN chi no longer well above QLKNN on the SINTEZ-NN state"

tmp = tempfile.TemporaryDirectory()
FIG = Path(tmp.name)


def save(fig, name):
    p = FIG / f"{name}.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- figures
def fig_schematic():
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=7.6, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=INK, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
                                     color=INK2, lw=1.0, shrinkA=0, shrinkB=0))

    ax.text(0, 51, "A · Толық ITER разряды (интеграцияланған)", fontsize=9, fontweight="bold", va="top")
    box(0, 29, 27, 15, "Бірдей енгізулер\n15 МА · 5,3 Тл · 50 МВт\n$n_e(\\rho)$ · $T_{ped}$ · $Z_{eff}$", TINT_GRAY, INK2)
    box(36, 36, 31, 8, "SINTEZ 1,5D\n+ нейрожелілік суррогат", TINT_ORANGE, ORANGE)
    box(36, 26, 31, 8, "TORAX 1D\n+ QLKNN (DeepMind)", TINT_BLUE, BLUE)
    box(76, 29, 24, 15, "$P_{fus}$ · $Q$ · $W_{th}$\n$T(\\rho)$ профильдері", TINT_GRAY, INK2)
    arrow(27.8, 38, 35.4, 40)
    arrow(27.8, 35, 35.4, 30)
    arrow(67.8, 40, 75.4, 38)
    arrow(67.8, 30, 75.4, 35)

    ax.text(0, 21, "B · Бір плазма күйі — екі желі", fontsize=9, fontweight="bold", va="top")
    box(0, 1.5, 27, 15, "Бір қатқан күй\n$T_e$, $T_i$, $n_e$, $q$", TINT_GRAY, INK2)
    box(36, 9, 31, 7.5, "SINTEZ желісі → $\\chi(\\rho)$", TINT_ORANGE, ORANGE)
    box(36, 1, 31, 7.5, "QLKNN → $\\chi_e(\\rho)$, $\\chi_i(\\rho)$", TINT_BLUE, BLUE)
    box(76, 1.5, 24, 15, "$\\chi(\\rho)$ бойынша\nсалыстыру", TINT_GRAY, INK2)
    arrow(27.8, 11, 35.4, 12.5)
    arrow(27.8, 7, 35.4, 5)
    arrow(67.8, 12.5, 75.4, 11)
    arrow(67.8, 5, 75.4, 7)
    return save(fig, "f1_schematic")


def fig_bars():
    items = [
        ("TORAX · QLKNN", tx, BLUE, None),
        ("SINTEZ · скейлингке бекітілген", an, AQUA, None),
        ("SINTEZ · CGM", cg, GRAY, None),
        ("SINTEZ · нейрожелі суррогат", nn, ORANGE, None),
        ("SINTEZ · Бом/гиро-Бом", bg, GRAY, None),
        ("SINTEZ · гиро-Бом", gb, GRAY, None),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    for i, (name, d, col, _) in enumerate(items):
        ax.barh(i, d["Pfus"], color=col, height=0.62, zorder=3)
        ax.text(d["Pfus"] + 8, i, f"{c(d['Pfus'], 0)} МВт · Q = {c(d['Q'])}", va="center", fontsize=8.5,
                zorder=5, bbox=dict(fc="white", ec="none", pad=1.2))
    ax.axvline(ref["Pfus"], color=INK, ls="--", lw=1.0, zorder=4)
    ax.text(ref["Pfus"] - 6, -0.62, "ITER жобасы 500 МВт", ha="right", va="center", fontsize=8)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([n for n, *_ in items])
    ax.set_ylim(len(items) - 0.4, -1.0)
    ax.set_xlim(0, 760)
    ax.set_xlabel("Термоядролық синтез қуаты $P_{fus}$, МВт (t = 120 с, бірдей ITER режимі)")
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(COMMA)
    return save(fig, "f2_bars")


def fig_traces():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True,
                               gridspec_kw={"width_ratios": [1.6, 1]})
    spec = [("scaling_anchored", AQUA, "-", 1.6), ("cgm", GRAY, "-", 1.2), ("neural_surrogate", ORANGE, "-", 2.2),
            ("bohm_gyrobohm", GRAY, "-", 1.2), ("gyrobohm", GRAY, "--", 1.0)]
    for key, col, ls, lw in spec:
        tr = S[key]["trace"]
        a.plot(tr["t"], tr["Pfus"], color=col, ls=ls, lw=lw, zorder=3)
    end = S["scaling_anchored"]["trace"]["t"][-1]
    lead = dict(arrowstyle="-", color=GRAY, lw=0.6, shrinkA=0, shrinkB=2)
    for text, y_line, y_text in [("скейлингке бекітілген", S["scaling_anchored"]["trace"]["Pfus"][-1], 560),
                                 ("CGM", S["cgm"]["trace"]["Pfus"][-1], 396),
                                 ("нейрожелі", S["neural_surrogate"]["trace"]["Pfus"][-1], 150),
                                 ("Бом/гиро-Бом ≈ гиро-Бом", S["bohm_gyrobohm"]["trace"]["Pfus"][-1], 25)]:
        a.annotate(text, xy=(end, y_line), xytext=(end + 3, y_text), fontsize=7.5, va="center",
                   arrowprops=lead, annotation_clip=False)
    a.axhline(ref["Pfus"], color=INK, ls="--", lw=0.9)
    a.text(2, ref["Pfus"] + 14, "ITER жобасы 500 МВт", fontsize=7.5, va="bottom")
    a.set_xlim(0, 175)
    a.set_xticks([0, 30, 60, 90, 120])
    a.set_ylim(0, 860)
    a.set_title("SINTEZ: бүкіл разряд (0 → 120 с)")
    a.set_xlabel("Плазма уақыты, с")
    a.set_ylabel("$P_{fus}$, МВт")
    b.plot(np.array(T["no_sawteeth"]["trace_t"]), T["no_sawteeth"]["trace_Pfus"], color=BLUE, lw=1.0,
           ls="--", alpha=0.7, zorder=2)
    b.plot(np.array(T["main"]["trace_t"]), T["main"]["trace_Pfus"], color=BLUE, lw=2.0, zorder=3)
    b.text(29.5, 700, "пилообразныйсыз\n(физикалық емес)", ha="right", va="center", fontsize=7, color=INK2,
           linespacing=1.1)
    b.text(29.5, 330, "TORAX · QLKNN", ha="right", va="center", fontsize=7.5)
    b.axhline(ref["Pfus"], color=INK, ls="--", lw=0.9)
    b.set_xlim(0, 30)
    b.set_title("TORAX: тұрақты режимге жинақталу")
    b.set_xlabel("Плазма уақыты, с")
    for ax in (a, b):
        ax.yaxis.set_major_formatter(COMMA)
    return save(fig, "f3_traces")


def fig_profiles():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7))
    tp = T["main"]["profiles"]
    sets = [("$T_e$, кэВ", "Te", "T_e"), ("$T_i$, кэВ", "Ti", "T_i")]
    for ax, (ttl, ks, kt) in zip(axs[:2], sets):
        ax.plot(S["scaling_anchored"]["profiles"]["rho"], S["scaling_anchored"]["profiles"][ks],
                color=AQUA, lw=1.6, label="SINTEZ · скейлингке бекітілген")
        ax.plot(S["neural_surrogate"]["profiles"]["rho"], S["neural_surrogate"]["profiles"][ks],
                color=ORANGE, lw=2.2, label="SINTEZ · нейрожелі")
        ax.plot(tp["rho_cell"], tp[kt], color=BLUE, lw=2.0, label="TORAX · QLKNN")
        ax.set_title(ttl)
        ax.set_xlabel("ρ")
        ax.set_ylim(0, None)
        ax.yaxis.set_major_formatter(COMMA)
    ax = axs[2]
    ax.plot(S["scaling_anchored"]["profiles"]["rho"], S["scaling_anchored"]["profiles"]["q"], color=AQUA, lw=1.6)
    ax.plot(S["neural_surrogate"]["profiles"]["rho"], S["neural_surrogate"]["profiles"]["q"], color=ORANGE, lw=2.2)
    ax.plot(tp["rho_face"], tp["q"], color=BLUE, lw=2.0)
    ax.set_title("q")
    ax.set_xlabel("ρ")
    ax.set_ylim(0, 4.5)
    ax.yaxis.set_major_formatter(COMMA)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    return save(fig, "f4_profiles")


def fig_chi():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
    for ax, (key, ttl) in zip(axs, [("torax_state", "A · TORAX күйі"),
                                    ("sintez_neural_surrogate", "B1 · SINTEZ нейрожелі күйі"),
                                    ("sintez_scaling_anchored", "B2 · SINTEZ әдепкі күйі (Q ≈ 10)")]):
        d = X[key]
        r = np.array(d["rho"])
        ax.axvspan(0, mask_lo, color=TINT_GRAY, zorder=0)
        ax.axvspan(mask_hi, 1, color=TINT_GRAY, zorder=0)
        ax.plot(r, d["chi_qlknn_i"], color=BLUE, lw=1.3, ls="--", label="QLKNN $\\chi_i$")
        ax.plot(r, d["chi_qlknn_e"], color=BLUE, lw=2.0, label="QLKNN $\\chi_e$")
        ax.plot(r, d["chi_nn"], color=ORANGE, lw=2.2, label="SINTEZ желісі $\\chi$")
        ax.set_yscale("log")
        ax.set_ylim(0.03, 30)
        ax.set_xlim(0, 1)
        ax.set_title(ttl, fontsize=8.2)
        ax.set_xlabel("ρ")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
        ax.grid(which="minor", visible=False)
    axs[0].set_ylabel("$\\chi$, м²/с (логарифмдік шкала)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    return save(fig, "f5_chi")


def fig_scatter():
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    xs = np.logspace(np.log10(0.03), np.log10(30), 50)
    ax.fill_between(xs, xs / 2, xs * 2, color=TINT_GRAY, zorder=0, lw=0)
    ax.plot(xs, xs, color=INK2, lw=0.9, zorder=1)
    ax.text(0.045, 0.09, "y = x", fontsize=7.5, color=INK2, rotation=40)
    for key, mkr, face, lab in [("torax_state", "o", INK, "A"), ("sintez_neural_surrogate", "^", GRAY, "B1"),
                                ("sintez_scaling_anchored", "s", "white", "B2")]:
        d = X[key]
        ax.scatter(np.array(d["chi_qlknn_e"])[mk], np.array(d["chi_nn"])[mk], marker=mkr, s=26,
                   facecolor=face, edgecolor=INK, lw=0.9, label=lab, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.03, 30)
    ax.set_ylim(0.03, 30)
    ax.set_xlabel("QLKNN $\\chi_e$, м²/с")
    ax.set_ylabel("SINTEZ желісі $\\chi$, м²/с")
    ax.legend(loc="lower right", fontsize=8, title="күй", title_fontsize=8)
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    ax.grid(which="minor", visible=False)
    return save(fig, "f6_scatter")


def fig_latency():
    rows = [("QLKNN · JAX jit · 26 нүкте", lat["qlknn_N26_us"], lat["qlknn_N26_range"], BLUE),
            ("QLKNN · JAX jit · 65 нүкте", lat["qlknn_N65_us"], lat["qlknn_N65_range"], BLUE),
            ("SINTEZ желісі · NumPy · 26 нүкте", lat["our_nn_N26_us"], lat["our_nn_N26_range"], ORANGE),
            ("SINTEZ желісі · NumPy · 65 нүкте", lat["our_nn_N65_us"], lat["our_nn_N65_range"], ORANGE),
            ("Бом/гиро-Бом формуласы · 65 нүкте", lat["our_bgb_N65_us"], lat["our_bgb_N65_range"], GRAY),
            ("CGM формуласы · 65 нүкте", lat["our_cgm_N65_us"], lat["our_cgm_N65_range"], GRAY)]
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    for i, (n, v, rg, col) in enumerate(rows):
        ax.barh(i, v, color=col, height=0.6, zorder=3)
        ax.plot(rg, [i, i], color=INK, lw=1.2, zorder=4)
        ax.plot([rg[0]] * 2, [i - 0.12, i + 0.12], color=INK, lw=1.2, zorder=4)
        ax.plot([rg[1]] * 2, [i - 0.12, i + 0.12], color=INK, lw=1.2, zorder=4)
        ax.text(max(rg[1], v) + 8, i, f"{c(v, 0)} мкс", va="center", fontsize=8.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 420)
    ax.set_xlabel("Бір шақыру уақыты, мкс (9 өлшемнің медианасы; сызық — ең аз/ең көп)")
    ax.grid(axis="y", visible=False)
    return save(fig, "f7_latency")


# --------------------------------------------------------------------------- PDF building blocks
PAGE_W, PAGE_H = A4
MARGIN = 17 * mm
BODY_W = PAGE_W - 2 * MARGIN

st = {
    "title": ParagraphStyle("title", fontName="DV-B", fontSize=19, leading=24, textColor=colors.HexColor(INK),
                            spaceAfter=3),
    "sub": ParagraphStyle("sub", fontName="DV", fontSize=10.5, leading=14, textColor=colors.HexColor(INK2),
                          spaceAfter=3),
    "meta": ParagraphStyle("meta", fontName="DV", fontSize=8, leading=11, textColor=colors.HexColor(INK2)),
    "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=13, leading=17, textColor=colors.HexColor(INK),
                         spaceBefore=10, spaceAfter=5, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=10, leading=13, textColor=colors.HexColor(INK),
                         spaceBefore=7, spaceAfter=3, keepWithNext=1),
    "p": ParagraphStyle("p", fontName="DV", fontSize=9, leading=13.2, textColor=colors.HexColor(INK),
                        alignment=TA_LEFT, spaceAfter=4),
    "li": ParagraphStyle("li", fontName="DV", fontSize=9, leading=13.2, textColor=colors.HexColor(INK),
                         leftIndent=11, bulletIndent=0, spaceAfter=3),
    "small": ParagraphStyle("small", fontName="DV", fontSize=7.6, leading=10.4, textColor=colors.HexColor(INK2),
                            spaceAfter=3),
    "cap": ParagraphStyle("cap", fontName="DV", fontSize=8, leading=10.8, textColor=colors.HexColor(INK2),
                          spaceBefore=2, spaceAfter=7),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=7.9, leading=10, textColor=colors.HexColor(INK)),
    "cellb": ParagraphStyle("cellb", fontName="DV-B", fontSize=7.9, leading=10, textColor=colors.HexColor(INK)),
    "cellh": ParagraphStyle("cellh", fontName="DV-B", fontSize=7.6, leading=9.6, textColor=colors.white),
    "num": ParagraphStyle("num", fontName="DV", fontSize=7.9, leading=10, alignment=2,
                          textColor=colors.HexColor(INK)),
    "numb": ParagraphStyle("numb", fontName="DV-B", fontSize=7.9, leading=10, alignment=2,
                           textColor=colors.HexColor(INK)),
    "numh": ParagraphStyle("numh", fontName="DV-B", fontSize=7.6, leading=9.6, alignment=2, textColor=colors.white),
    "card_n": ParagraphStyle("card_n", fontName="DV-B", fontSize=17, leading=21, alignment=1),
    "card_q": ParagraphStyle("card_q", fontName="DV-B", fontSize=11.5, leading=15, alignment=1, spaceAfter=2),
    "card_t": ParagraphStyle("card_t", fontName="DV", fontSize=8.4, leading=11, alignment=1,
                             textColor=colors.HexColor(INK2)),
    "code": ParagraphStyle("code", fontName="DV", fontSize=7.4, leading=10, textColor=colors.HexColor(INK),
                           leftIndent=6),
}


def P(t, s="p"):
    return Paragraph(t, st[s])


def bullets(items):
    return [Paragraph(t, st["li"], bulletText="•") for t in items]


def img(path, width=BODY_W):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    return Image(str(path), width=width, height=width * h / w)


def table(data, widths, header_color=INK2, zebra=True, extra=None, repeat=1):
    t = Table(data, colWidths=widths, repeatRows=repeat)
    style = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor(GRID)),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if extra:
        style += extra
    t.setStyle(TableStyle(style))
    return t


def callout(text, bg=TINT_GRAY, edge=INK2):
    t = Table([[Paragraph(text, st["p"])]], colWidths=[BODY_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
                           ("LINEBEFORE", (0, 0), (0, -1), 2.2, colors.HexColor(edge)),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 7.5)
    canvas.setFillColor(colors.HexColor(INK2))
    canvas.drawString(MARGIN, 10 * mm, "SINTEZ нейрожелілік суррогаты және Google DeepMind TORAX — салыстыру")
    canvas.drawRightString(PAGE_W - MARGIN, 10 * mm, f"{doc.page}-бет")
    canvas.restoreState()


# --------------------------------------------------------------------------- story
def build():
    f_schem, f_bars, f_trace = fig_schematic(), fig_bars(), fig_traces()
    f_prof, f_chi, f_scat, f_lat = fig_profiles(), fig_chi(), fig_scatter(), fig_latency()

    doc = SimpleDocTemplate(str(OUT_PDF), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=15 * mm, bottomMargin=17 * mm,
                            title="SINTEZ нейрожелілік суррогаты және Google DeepMind TORAX — салыстыру",
                            author="Claude Code")
    s = []

    # ---------------------------------------------------------------- page 1: summary
    s += [P("SINTEZ нейрожелілік суррогаты және Google DeepMind TORAX", "title"),
          P("Бір ITER сценарийіндегі екі нейрожелілік тасымал моделінің нәтижелері", "sub"),
          P("18 қазан 2026 · репозиторий bibaermek-stack/tokomak · тармақ claude/tokamak-project-similarity-a7ssc5 · "
            f"TORAX {R['versions']['torax']}, JAX {R['versions']['jax']} (CPU)", "meta"),
          Spacer(1, 7)]

    def card(num, q, title, col):
        return [Paragraph(f'<font color="{col}">{num}</font>', st["card_n"]),
                Paragraph(f'<font color="{col}">{q}</font>', st["card_q"]), Paragraph(title, st["card_t"])]

    cards = Table([[card(f"{c(tx['Pfus'], 0)} МВт", f"Q = {c(tx['Q'])}", "TORAX · QLKNN (DeepMind, нағыз код)", BLUE),
                    card("500 МВт", "Q = 10", "ITER жобалық нүктесі (эталон)", INK),
                    card(f"{c(nn['Pfus'], 0)} МВт", f"Q = {c(nn['Q'])}", "SINTEZ · нейрожелі суррогат (біздің модель)", ORANGE)]],
                  colWidths=[BODY_W / 3] * 3)
    cards.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), colors.HexColor(TINT_BLUE)),
                               ("BACKGROUND", (1, 0), (1, 0), colors.HexColor(TINT_GRAY)),
                               ("BACKGROUND", (2, 0), (2, 0), colors.HexColor(TINT_ORANGE)),
                               ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor(GRID)),
                               ("INNERGRID", (0, 0), (-1, -1), 2, colors.white),
                               ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    s += [cards, Spacer(1, 4),
          P("Бірдей ITER режимі (15 МА, 5,3 Тл, 50 МВт қосымша қыздыру, бірдей тығыздық профилі мен педестал) "
            "екі модельге берілді; көрсетілгені — тұрақты күйдегі термоядролық синтез қуаты мен күшею коэффициенті.",
            "small"),
          img(f_bars, BODY_W * 0.97),
          P("1-сурет. Бірдей ITER режиміндегі синтез қуаты (t = 120 с) және Q.", "cap")]

    s += [P("Қысқаша қорытынды", "h1")]
    s += bullets([
        f"<b>Нағыз TORAX ITER-дің жобалық нүктесін қайта шығарды</b>: P<sub>fus</sub> = {c(tx['Pfus'], 0)} МВт, "
        f"Q = {c(tx['Q'])}, W<sub>th</sub> = {c(tx['Wth'], 0)} МДж, β<sub>N</sub> = {c(tx['betaN'], 2)} "
        f"(жоба: 500 МВт, Q = 10, 325 МДж, 1,77).",
        f"<b>Біздің нейрожелілік суррогат SINTEZ ішінде {c(nn['Pfus'], 0)} МВт және Q = {c(nn['Q'])} берді</b> — "
        f"TORAX-тан {c(tx['Pfus'] / nn['Pfus'])} есе аз. Нәтиже гиро-Бом моделіне ({c(gb['Pfus'], 0)} МВт) жақын, "
        f"ал CGM ({c(cg['Pfus'], 0)} МВт) пен әдепкі модельден ({c(an['Pfus'], 0)} МВт) әлдеқайда төмен.",
        f"<b>Себеп желінің өзінде емес, оның оқу деректерінде.</b> Желі өз «мұғалімін» жақсы қайталайды (жаңа "
        f"деректе R² = {c(fid['r2'], 3)}), бірақ мұғалімнің жылу өткізгіштігі χ QLKNN-нен әлдеқайда жоғары: "
        f"TORAX күйінде медиана бойынша {c(xa['metrics_vs_e']['median_ratio'])} есе. Сонымен қатар критикалық "
        f"градиент табалдырығы жоқ — QLKNN табалдырықтан төмен χ-ді нөлге жақын қояды, желі қоймайды.",
        f"<b>Жылдамдық артықшылығы жоқ.</b> Желінің бір шақыруы (65 нүкте) ≈ {c(lat['our_nn_N65_us'], 0)} мкс, "
        f"QLKNN ≈ {c(lat['qlknn_N65_us'], 0)} мкс — бір рет; қарапайым формула (CGM) шамамен "
        f"{c(lat['our_nn_N65_us'] / lat['our_cgm_N65_us'], 0)} есе жылдам.",
        "<b>Қорытынды:</b> қазіргі суррогат архитектурасы жағынан TORAX-тың QLKNN-іне ұқсайды, бірақ оның "
        "дәлдігіне жетпейді. ITER тәрізді режимдерді болжау үшін QLKNN-мен калибрлеу (дистилляция) қажет.",
    ])
    s += [Spacer(1, 3),
          callout("<b>Маңызды ескерту.</b> TORAX біздің репозиторийде жоқ — онда тек «TORAX стиліндегі» суррогат бар. "
                  "Салыстыру үшін DeepMind-тың ресми TORAX 1.4.3 нұсқасы жеке виртуалды ортаға орнатылды "
                  "(жоба тәуелділіктері өзгермеді). SINTEZ-тың әдепкі моделі ITER-дің Q = 10 нүктесіне "
                  "калибрленген (transport.py: R_OVER_LT_CRIT), сондықтан оның "
                  f"{c(an['Pfus'], 0)} МВт-ы болжам емес, калибрлеу нәтижесі. TORAX-тың ядро тасымалы (QLKNN) "
                  f"ITER-ге бапталмаған; оның {c(tx['Pfus'], 0)} МВт-ы — педестал мен тығыздық SINTEZ-тен алынған "
                  "жағдайдағы болжам.", TINT_ORANGE, ORANGE),
          PageBreak()]

    # ---------------------------------------------------------------- page 2: setup
    s += [P("1. Тәжірибе қалай құрылды", "h1"),
          P("«Біздің симуляцияға салу» дегенді екі түрлі орындадық. <b>А тәжірибесі</b> — бүкіл ITER разрядын "
            "екі кодта бірдей енгізулермен жүргізу. <b>В тәжірибесі</b> — бір ғана плазма күйін екі желіге беріп, "
            "олардың жылу өткізгіштігін χ(ρ) салыстыру: бұл код айырмашылықтарынан тәуелсіз, таза «желі–желі» тексеру.",
            "p"),
          img(f_schem, BODY_W * 0.93),
          P("2-сурет. Екі тәжірибенің сызбасы. Қызғылт сары — SINTEZ, көк — TORAX.", "cap")]

    cw = [BODY_W * 0.19, BODY_W * 0.405, BODY_W * 0.405]
    hdr = [Paragraph("Параметр", st["cellh"]), Paragraph("SINTEZ", st["cellh"]), Paragraph("TORAX", st["cellh"])]
    rows = [
        ("Машина", "ITER: R₀ = 6,2 м, a = 2,0 м, B₀ = 5,3 Тл, I<sub>p</sub> = 15 МА",
         "бірдей (ITER пішіні, TORAX-пен бірге келетін CHEASE файлы)"),
        ("Қосымша қыздыру", "50 МВт: NBI 33 + ICRF 17",
         f"50 МВт, бір кең гаусс көзі; электрондарға {c(100 * T['main']['config']['electron_heat_fraction'], 0)} % "
         "(SINTEZ-тің баяулау бөлінісі бойынша)"),
        ("Тығыздық", "өз тасымалынан және отын бақылауынан есептеледі",
         "SINTEZ нейрожелі разрядының n_e(ρ) профилі, <b>бекітілген</b> (дамымайды)"),
        ("Педестал", f"EPED тәрізді модель, T<sub>ped</sub> = {c(nn['Tped'], 2)} кэВ",
         f"сол T<sub>ped</sub> және n<sub>ped</sub> = {c(T['main']['config']['n_ped_m3'] / 1e20, 2)}·10²⁰ м⁻³, ρ = 0,93"),
        ("Қоспа", f"Z<sub>eff</sub> = {c(nn['Zeff'], 2)}", f"Z<sub>eff</sub> = {c(T['main']['config']['Zeff'], 2)}, Ne қоспасы"),
        ("Сәулелену", f"тежеу + синхротрон + сызықтық: {c(nn['Prad'], 1)} МВт",
         f"тежеу + циклотрон + Ne (Mavrin): {c(tx['Prad'], 1)} МВт"),
        ("Тасымал", "таңдалатын 5 модель; нейрожелі желі тек χ<sub>e</sub> береді (иондар — әдепкі қатты форма)",
         "QLKNN (qlknn_7_11_v1): χ<sub>e</sub>, χ<sub>i</sub>, D, V; TORAX-тың өз ішкі/сыртқы патчтарымен"),
        ("Пилообразный, ток", "Кадомцев; Грэд–Шафранов тепе-теңдігі", "TORAX simple sawtooth; ток диффузиясы"),
        ("Радиал тор", "65 нүкте", "25 ұяшық"),
    ]
    data = [hdr] + [[Paragraph(a, st["cellb"]), Paragraph(b, st["cell"]), Paragraph(d, st["cell"])]
                    for a, b, d in rows]
    s += [table(data, cw, extra=[("BACKGROUND", (1, 1), (1, -1), colors.HexColor(TINT_ORANGE)),
                                 ("BACKGROUND", (2, 1), (2, -1), colors.HexColor(TINT_BLUE))]),
          P("1-кесте. Не бірдей, не өзгеше. Тығыздық, педестал, қыздыру, Z<sub>eff</sub> бірдей берілді — "
            "сонда нәтижедегі айырмашылық негізінен ядро тасымалынан шығады.", "cap")]

    s += [P("Екі модель нені білдіреді", "h2")]
    s += bullets([
        f"<b>SINTEZ суррогаты.</b> Көп қабатты перцептрон 11→128→64→1, ReLU, {thou(fid['n_parameters'])} параметр. "
        "Оқу деректері — жобаның өз формулалары (Бом/гиро-Бом мен критикалық градиент қоспасы + 3 % шу), "
        "<b>гирокинетикалық есеп емес</b>. Кіріс: ρ, R/L<sub>T</sub>, R/L<sub>n</sub>, q, T<sub>e</sub>, n<sub>e</sub>, "
        "T<sub>i</sub>/T<sub>e</sub>, B₀, R₀, a, H-режим белгісі.",
        "<b>TORAX QLKNN.</b> google-deepmind/torax 1.4.3, JAX негізінде. qlknn_7_11_v1 желісі QuaLiKiz "
        "гирокинетикалық кодының ресми деректерінде (QLKNN11D + QLKNN7D-edge) оқытылған; ITG, TEM, ETG "
        "режимдерін бөлек береді.",
    ])

    # ---------------------------------------------------------------- page 3: results 1
    s += [PageBreak(), P("2. А тәжірибесі — толық ITER разряды", "h1"),
          P(f"2-кестеде барлық жағдайдың тұрақты күйдегі нәтижесі (SINTEZ — t = 120 с, TORAX — t = 30 с). "
            f"Негізгі салыстыру — <b>TORAX · QLKNN</b> (көк) пен <b>SINTEZ · нейрожелі</b> (қызғылт сары) жолдары.", "p")]

    cols = [("P<sub>fus</sub><br/>МВт", "Pfus", 0), ("Q", "Q", 1), ("W<sub>th</sub><br/>МДж", "Wth", 0),
            ("τ<sub>E</sub><br/>с", "tauE", 2), ("H98", "H_factor", 2), ("β<sub>N</sub>", "betaN", 2),
            ("⟨T<sub>e</sub>⟩<br/>кэВ", "Te", 1), ("⟨T<sub>i</sub>⟩<br/>кэВ", "Ti", 1), ("T<sub>e0</sub><br/>кэВ", "Te0", 1)]
    cases = [("ITER жобасы (эталон)", ref, "ref"),
             ("TORAX · QLKNN", tx, "torax"),
             ("SINTEZ · нейрожелі суррогат", nn, "nn"),
             ("SINTEZ · скейлингке бекітілген (әдепкі)", an, ""),
             ("SINTEZ · CGM", cg, ""),
             ("SINTEZ · Бом/гиро-Бом", bg, ""),
             ("SINTEZ · гиро-Бом", gb, ""),
             ("TORAX · пилообразныйсыз (сезімталдық)", ns, "sens"),
             (f"TORAX · сәулелену +{c(rad['Prad'] - tx['Prad'], 0)} МВт (сезімталдық)", rad, "sens")]
    data = [[Paragraph("Жағдай", st["cellh"])] + [Paragraph(h, st["numh"]) for h, *_ in cols]]
    extra = []
    for i, (name, d, kind) in enumerate(cases, start=1):
        bold = kind in ("torax", "nn")
        row = [Paragraph(name, st["cellb" if bold else "cell"])]
        for _, k, nd in cols:
            row.append(Paragraph(c(d[k], nd), st["numb" if bold else "num"]))
        data.append(row)
        if kind == "torax":
            extra.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor(TINT_BLUE)))
        elif kind == "nn":
            extra.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor(TINT_ORANGE)))
        elif kind == "ref":
            extra.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor(TINT_GRAY)))
    s += [table(data, [BODY_W * 0.30] + [BODY_W * 0.70 / 9] * 9, extra=extra),
          P("2-кесте. Тұрақты күйдегі нәтижелер. τ<sub>E</sub> мен H98 — SINTEZ анықтамасымен "
            "(P<sub>loss</sub> = P<sub>қыздыру</sub> − P<sub>сәулелену</sub>, IPB98(y,2)) екі кодқа да бірдей есептелген; "
            f"TORAX-тың өз анықтамасымен τ<sub>E</sub> = {c(tx['tauE_torax'], 2)} с, H98 = {c(tx['H98_torax'], 2)}. "
            "⟨T⟩: SINTEZ-те тығыздықпен өлшенген, TORAX-та көлемдік орташа. β<sub>N</sub>: SINTEZ жылдам иондарды қосады. "
            "Эталон — tokamak.validation (ITER Physics Basis).", "cap"),
          KeepTogether([img(f_trace, BODY_W),
                        P("3-сурет. P<sub>fus</sub>(t). Уақыт осьтері әртүрлі: SINTEZ — ток көтеру мен қыздыруды қоса "
                          "бүкіл разряд; TORAX — тұрақты режимге жинақталу (есеп 12 кэВ-тық ыстық бастапқы күйден "
                          "басталады, сондықтан t = 0-де P<sub>fus</sub> нөл емес).", "cap")])]

    s += [P("Не көрінеді", "h2")]
    s += bullets([
        f"TORAX {c(tx['Pfus'], 0)} МВт-қа жинақталады және онда тұрады; ITER жобасынан ауытқуы "
        f"{c(abs(tx['Pfus'] / ref['Pfus'] - 1) * 100, 1)} %.",
        f"SINTEZ-тің нейрожелі разряды {c(nn['Pfus'], 0)} МВт-та тұрып қалады: жылу энергиясы {c(nn['Wth'], 0)} МДж "
        f"(TORAX {c(tx['Wth'], 0)}), орташа T<sub>e</sub> = {c(nn['Te'], 1)} кэВ (TORAX {c(tx['Te'], 1)}). "
        f"Нәтиже гиро-Бом ({c(gb['Pfus'], 0)} МВт) пен Бом/гиро-Бом ({c(bg['Pfus'], 0)} МВт) нәтижесіне жақын — "
        "желі өз мұғалімінің осы бөлігін мұралаған.",
        f"TORAX нәтижесі <b>пилообразный моделіне сезімтал</b>: онсыз {c(ns['Pfus'], 0)} МВт (q₀ = {c(ns['q0'], 2)} — "
        "физикалық емес, ITER-де q₀ ≈ 1). Негізгі нәтиже — пилообразный қосулы нұсқа. Сәулеленуді "
        f"{c(rad['Prad'] - tx['Prad'], 0)} МВт арттыру P<sub>fus</sub>-ты тек {c(abs(rad['Pfus'] / tx['Pfus'] - 1) * 100, 1)} % "
        "өзгертеді — нәтиже сәулелену моделіне сезімтал емес.",
    ])

    # ---------------------------------------------------------------- page 4: profiles + cross
    s += [KeepTogether([P("Температура және q профильдері", "h2"), img(f_prof, BODY_W),
          P(f"4-сурет. Тұрақты күйдегі профильдер. SINTEZ нейрожелі разрядында профиль жалпақ "
            f"(T<sub>e0</sub> = {c(nn['Te0'], 1)} кэВ); TORAX-та T<sub>e0</sub> = {c(tx['Te0'], 1)} кэВ "
            f"(пилообразный ядроны тегістейді), әдепкі SINTEZ-те {c(an['Te0'], 1)} кэВ (ITER жобасы ≈ 25 кэВ). "
            "Орталық мәндер пилообразный параметрлеріне байланысты, ал P<sub>fus</sub> негізінен педестал мен "
            "орта радиус градиентімен анықталады.", "cap")])]

    s += [P("3. В тәжірибесі — бір күй, екі желі", "h1"),
          P("Үш плазма күйін алып, әрқайсысында QLKNN-ді (TORAX құралымен, тұрақталған күйде) және SINTEZ желісін "
            "бірдей кірістермен есептедік. <b>A</b> — TORAX-тың өзі жинақталған күйі (QLKNN-мен үйлесімді); "
            f"<b>B1</b> — SINTEZ нейрожелі разрядының соңғы күйі (T<sub>e0</sub> = {c(nn['Te0'], 1)} кэВ); "
            f"<b>B2</b> — SINTEZ әдепкі разрядының соңғы күйі (T<sub>e0</sub> = {c(an['Te0'], 1)} кэВ, Q ≈ 10). "
            f"Салыстыру ρ = {c(mask_lo, 2)}…{c(mask_hi, 2)} аймағында (сыртта TORAX-тың патчтары мен педестал).", "p"),
          img(f_chi, BODY_W),
          P("5-сурет. Жылу өткізгіштік χ(ρ). Сұр аймақ — салыстыруға кірмейтін (TORAX патчтары мен педестал).", "cap")]

    # ---------------------------------------------------------------- page 5: metrics + scatter
    hdr = [Paragraph("Күй", st["cellh"])] + [
        Paragraph(h, st["numh"]) for h in ("Желі/QLKNN χ<sub>e</sub> (медиана)", "Желі/QLKNN χ<sub>i</sub> (медиана)",
                                           "Пирсон r (χ<sub>e</sub>)", "×2 ішінде, %")]
    labels = {"torax_state": "A · TORAX күйі", "sintez_neural_surrogate": "B1 · SINTEZ нейрожелі күйі",
              "sintez_scaling_anchored": "B2 · SINTEZ әдепкі күйі"}
    data = [hdr]
    for key in ("torax_state", "sintez_neural_surrogate", "sintez_scaling_anchored"):
        d = X[key]
        data.append([Paragraph(labels[key], st["cell"]),
                     Paragraph(c(d["metrics_vs_e"]["median_ratio"], 1), st["num"]),
                     Paragraph(c(d["metrics_vs_i"]["median_ratio"], 1), st["num"]),
                     Paragraph(c(d["metrics_vs_e"]["pearson_r"], 2), st["num"]),
                     Paragraph(c(100 * d["metrics_vs_e"]["within_x2"], 0), st["num"])])
    s += [table(data, [BODY_W * 0.34, BODY_W * 0.18, BODY_W * 0.18, BODY_W * 0.15, BODY_W * 0.15]),
          P("3-кесте. Желі χ-і QLKNN-мен салыстырғанда (медиана &gt; 1 — желі жоғары береді; r — профиль формасының "
            "сәйкестігі; ×2 ішінде — желі QLKNN-нен екі еседен аз өзгешеленетін нүктелер үлесі).", "cap")]

    chi_max_b2 = float(np.max(np.array(xb2["chi_qlknn_e"])[mk]))
    s += bullets([
        f"<b>A (TORAX күйі).</b> QLKNN бұл күйде χ<sub>e</sub> ≈ {c(float(np.min(np.array(xa['chi_qlknn_e'])[mk])), 2)}–"
        f"{c(float(np.max(np.array(xa['chi_qlknn_e'])[mk])), 2)} м²/с береді; желі {c(float(np.min(np.array(xa['chi_nn'])[mk])), 0)}–"
        f"{c(float(np.max(np.array(xa['chi_nn'])[mk])), 0)} м²/с — медиана бойынша {c(xa['metrics_vs_e']['median_ratio'], 1)} есе "
        f"жоғары, профиль формасы кері (r = {c(xa['metrics_vs_e']['pearson_r'], 2)}). Бұл — SINTEZ-те плазманың төмен "
        "температурада тепе-теңдікке келуімен сәйкес.",
        f"<b>B1 (SINTEZ нейрожелі күйі).</b> Профиль жалпақ, градиент критикалықтан төмен: QLKNN χ-ді шектік мәнге "
        f"(0,05 м²/с) түсіреді, ал желі {c(float(np.min(np.array(xb1['chi_nn'])[mk])), 1)}–"
        f"{c(float(np.max(np.array(xb1['chi_nn'])[mk])), 1)} м²/с береді. Табалдырықсыз, градиентке әлсіз "
        "тәуелді χ жылуды үнемі шығарып отырады — бұл разрядтың төмен температурада тұрып қалуының ықтимал "
        "түсіндірмесі. Бұл гипотеза: оны желіні қайта оқытып тексеру керек.",
        f"<b>B2 (SINTEZ әдепкі күйі).</b> Градиент жоғары, QLKNN ρ ≈ 0,5-те χ<sub>e</sub> ≈ {c(chi_max_b2, 1)} м²/с-қа дейін "
        f"көтеріледі (қатты тасымал). Желі медиана бойынша {c(xb2['metrics_vs_e']['median_ratio'], 1)} есе жоғары, "
        f"форма сәйкес емес (r = {c(xb2['metrics_vs_e']['pearson_r'], 2)}); {c(100 * xb2['metrics_vs_e']['within_x2'], 0)} % "
        "нүкте ×2 ішінде.",
    ])
    s += [Spacer(1, 3),
          Table([[img(f_scat, BODY_W * 0.50),
                  [P("Желінің өз «мұғалімімен» сәйкестігі жоғары: жаңа деректе "
                     f"R² = {c(fid['r2'], 3)}, MAE = {c(fid['mae'], 2)} м²/с ({thou(fid['n_points'])} нүкте); "
                     f"оқыту есебінде R² = {c(fid['report_json']['test_r2'], 3)}. Бұл «желі өз формуласын жақсы "
                     "үйренген» дегенді ғана білдіреді.", "p"),
                   P(f"Мұғалім өзі де QLKNN-нен алшақ: TORAX күйінде оның медианасы QLKNN χ<sub>e</sub>-нен "
                     f"{c(xa['teacher_vs_e']['median_ratio'], 1)} есе жоғары. Яғни қате желі архитектурасынан емес, "
                     "оқу мақсатынан (мұғалімнен) келеді.", "p")]]],
                colWidths=[BODY_W * 0.52, BODY_W * 0.48],
                style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)])),
          P("6-сурет. Әр нүкте — бір радиал нүкте (ρ = 0,25…0,85). Сұр жолақ — ×2 аралығы.", "cap")]

    # ---------------------------------------------------------------- page 6: speed
    s += [P("4. Жылдамдық", "h1"), img(f_lat, BODY_W),
          P("7-сурет. Тасымал коэффициентін бір рет есептеу уақыты. Бір компьютерде өлшенді; CPU шуы айтарлықтай "
            "(диапазон көрсетілген), сондықтан тек реті маңызды.", "cap")]
    s += bullets([
        f"Желі мен QLKNN бір ретте: 65 нүктеде ≈ {c(lat['our_nn_N65_us'], 0)} және ≈ {c(lat['qlknn_N65_us'], 0)} мкс. "
        "QLKNN-нің басымдығы — жылдамдықта емес, физикалық дәлдікте.",
        f"Қарапайым формулалар желіден жылдам: CGM {c(lat['our_cgm_N65_us'], 0)} мкс, Бом/гиро-Бом {c(lat['our_bgb_N65_us'], 0)} мкс. "
        "Сондықтан бұл желі есептеу үнемдемейді. Презентациядағы «122 мкс (желі) және 148 мкс (формула)» өлшемі "
        "басқа формуламен салыстырылған болуы керек: біздің өлшемде екі формула да желіден жылдам.",
        f"Бүкіл жүгіру уақыты: SINTEZ — 120 с разряд {c(min(S[k]['wall_s'] for k in S), 0)}–{c(max(S[k]['wall_s'] for k in S), 0)} с "
        f"(тепе-теңдік, МГД, дивертор, отын циклін қоса); TORAX — 30 с тұрақты режим {c(min(T[k]['wall_s'] for k in T), 0)}–"
        f"{c(max(T[k]['wall_s'] for k in T), 0)} с (оның ≈ 20 с-ы JIT компиляциясы). Екі код әртүрлі міндет шешеді, "
        "сондықтан бұл сандарды тікелей салыстыруға болмайды.",
    ])

    # ---------------------------------------------------------------- limitations
    s += [P("5. Шектеулер және келесі қадамдар", "h1"), P("Нәтижені оқығанда ескеру керек", "h2")]
    s += bullets([
        "Екі код әртүрлі: SINTEZ — 1,5D (65 нүкте, Грэд–Шафранов), TORAX — 1D (25 ұяшық, ток диффузиясы). Енгізулер "
        "бірдей берілді, бірақ қыздыру көзінің пішіні, геометрия, МГД және сәулелену модельдері бірдей емес.",
        f"Сәулелену есебі әртүрлі: Q ≈ 10 күйде SINTEZ {c(an['Prad'], 0)} МВт, TORAX {c(tx['Prad'], 0)} МВт сәулелендіреді. "
        f"Алайда сезімталдық сынағы (+{c(rad['Prad'] - tx['Prad'], 0)} МВт) P<sub>fus</sub>-ты тек "
        f"{c(abs(rad['Pfus'] / tx['Pfus'] - 1) * 100, 1)} % өзгертті, сондықтан желі айырмашылығын түсіндірмейді.",
        "TORAX-қа SINTEZ-тің T<sub>ped</sub> және n<sub>e</sub>(ρ) берілді — бұл қасақана, тек ядро тасымалының әсерін бөліп көру үшін. "
        "TORAX-тың өз педестал/тығыздық болжамы емес.",
        "SINTEZ-те желі тек электрон каналын (χ<sub>e</sub> және D = 0,25·χ<sub>e</sub>) береді; иондық χ<sub>i</sub> — әдепкі қатты форма. "
        "TORAX-та χ<sub>e</sub>, χ<sub>i</sub>, D, V — бәрі QLKNN-нен. Бұл айырмашылық нәтижені желі пайдасына да, оның қарсысына да ығыстыруы мүмкін.",
        "Бір ғана режим (ITER, H-режим) және бір рет жүргізу; басқа машиналар мен L-режим тексерілмеді. В тәжірибесінде үш күй ғана.",
        "QLKNN-ді SINTEZ ішіне, желіні TORAX ішіне тікелей қосқан жоқпыз: QLKNN кірісін дайындау TORAX геометриясына байланысты. "
        "Сондықтан В тәжірибесінде QLKNN-ді TORAX өз құралымен тұрақталған күйде есептедік.",
    ])
    s += [P("Ұсынылатын келесі қадамдар", "h2")]
    s += bullets([
        "Желіні QLKNN-мен «дистилляциялау»: TORAX-тың QLKNN-ін мұғалім етіп, SINTEZ күйлерінде жаңа оқу жиыны жасау "
        "немесе ашық QLKNN11D деректерінде оқыту.",
        "Критикалық градиент табалдырығын (қаттылықты) оқу деректері мен қателік функциясына енгізу; ион каналы мен "
        "бөлшек тасымалын да желіге беру.",
        "Қабылдау критерийі: <font name='DV'>tokamak.validation.check_integrated</font> (ITER, P<sub>fus</sub> ≈ 500 МВт ± 8 %) "
        f"осы модельмен өтуі тиіс; қазір P<sub>fus</sub> = {c(nn['Pfus'], 0)} МВт — шектен әлдеқайда тыс.",
        "<font name='DV'>neural_transport_surrogate_report.json</font> түзету: activation «tanh» деп жазылған (іс жүзінде ReLU), "
        "inference_latency_ms = 0,08 қолмен жазылған (өлшенгені 0,2–0,3 мс).",
    ])

    # ---------------------------------------------------------------- appendix
    s += [P("Қосымша. Қайта жүргізу", "h1"),
          P("Барлық скрипттер docs/torax_comparison/ ішінде. SINTEZ бөлігі — жобаның өз ортасында; TORAX — бөлек "
            "ортада (жоба тәуелділіктеріне қосылмайды).", "p")]
    code = [
        "# 1) SINTEZ (жоба ортасы), әр модельге ~1 мин",
        "for m in scaling_anchored cgm neural_surrogate bohm_gyrobohm gyrobohm; do",
        "  python run_sintez.py $m work/sintez_$m.json; done",
        "# 2) TORAX (бөлек орта)",
        "python -m venv torax_env && torax_env/bin/pip install torax==1.4.3",
        "torax_env/bin/python run_torax.py work/sintez_neural_surrogate.json work/torax_main.json",
        "torax_env/bin/python run_torax.py work/sintez_neural_surrogate.json work/torax_nosaw.json --no-sawteeth",
        "torax_env/bin/python run_torax.py work/sintez_neural_surrogate.json work/torax_rad.json --line-radiation-fraction 0.11",
        "for m in neural_surrogate scaling_anchored; do NU=1.2 torax_env/bin/python torax_frozen.py \\",
        "  work/sintez_$m.json work/frozen_$m.json; done",
        "torax_env/bin/python export_torax.py work",
        "# 3) талдау және PDF (жоба ортасы)",
        "python analyse.py work && python build_report.py",
    ]
    s += [Table([[Paragraph(l.replace("&", "&amp;").replace("<", "&lt;"), st["code"])] for l in code],
                colWidths=[BODY_W],
                style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(TINT_GRAY)),
                                  ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2)])),
          Spacer(1, 4),
          P(f"Нұсқалар: Python {R['versions']['python']}, TORAX {R['versions']['torax']}, JAX {R['versions']['jax']} "
            f"({R['versions']['jax_backend']}), NumPy {R['versions']['numpy']}. SINTEZ нәтижелері детерминді "
            "(қайта жүргізгенде бірдей). Шикі нәтижелер — results.json.", "small")]

    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print("written", OUT_PDF, round(OUT_PDF.stat().st_size / 1024), "kB")


if __name__ == "__main__":
    build()
