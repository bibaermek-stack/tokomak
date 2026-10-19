"""Black-on-white figure style: no colour, no grey fills.  Series are told apart by line style, marker and hatch."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

for f in Path("/usr/share/fonts/truetype/liberation").glob("LiberationSerif-*.ttf"):
    font_manager.fontManager.addfont(str(f))

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "text.color": "black", "axes.labelcolor": "black", "axes.edgecolor": "black",
    "xtick.color": "black", "ytick.color": "black", "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.minor.width": 0.5, "ytick.minor.width": 0.5, "xtick.direction": "in", "ytick.direction": "in",
    "axes.spines.top": True, "axes.spines.right": True, "axes.grid": False, "axes.facecolor": "white",
    "figure.facecolor": "white", "savefig.facecolor": "white", "legend.frameon": False, "lines.linewidth": 1.1,
    "lines.markersize": 4, "svg.fonttype": "path", "hatch.linewidth": 0.6, "savefig.dpi": 300, "pdf.fonttype": 42,
})
K = "black"
LS = {"-": "-", "--": (0, (5, 2.2)), ":": (0, (1, 1.6)), "-.": (0, (5, 1.6, 1, 1.6)), "..": (0, (1, 3.2)), "---": (0, (9, 3))}
MK = ["o", "s", "^", "v", "D", "x", "+", "*", "P", "X", "h", "<", ">", "p", "d"]
HATCH = ["", "////", "\\\\\\\\", "xxxx", "....", "++++", "----", "oooo"]
COMMA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))


def comma(ax, x=True, y=True):
    if x:
        ax.xaxis.set_major_formatter(COMMA)
    if y:
        ax.yaxis.set_major_formatter(COMMA)


def panel(ax, label, dx=0.0, dy=1.02):
    ax.text(dx, dy, f"({label})", transform=ax.transAxes, ha="left", va="bottom", fontsize=9, fontweight="bold")


def save(fig, name, vector=True):
    path = OUT / f"{name}.{'svg' if vector else 'png'}"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return path
