"""Figures for the paper, drawn from results.json and the training logs.

    python make_figures.py results.json figures/ run_B.json run_C.json run_A.json

Colours follow the entity, not the rank: a controller keeps its colour in
every figure.  Lines also differ in marker and dash, so the figures read in
greyscale print.  Palette: the reference categorical slots 1-3 and 2, plus a
neutral for the open loop.
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

COL = {"open_loop": "#8a8985", "PID": "#eb6834", "PPO-0": "#4a3aa7",
       "PPO-DR": "#1baf7a", "PPO-DR-AC": "#2a78d6"}
MARK = {"open_loop": "x", "PID": "s", "PPO-0": "^", "PPO-DR": "D",
        "PPO-DR-AC": "o"}
DASH = {"open_loop": (0, (3, 2)), "PID": (0, (5, 2)), "PPO-0": "-.",
        "PPO-DR": ":", "PPO-DR-AC": "-"}
LABEL = {"open_loop": "Ашық контур", "PID": "Классикалық (ПИД)",
         "PPO-0": "PPO-0", "PPO-DR": "PPO-DR", "PPO-DR-AC": "PPO-DR-AC"}
COND_KK = {"nominal": "номиналды", "randomised": "рандомиз.",
           "gain 1.3": "күшейту 1,3", "gain 0.7": "күшейту 0,7",
           "wall x2": "камера ×2", "wall x0.5": "камера ×0,5",
           "plasma x4": "плазма ×4", "plasma x0.25": "плазма ×0,25",
           "all adverse": "бәрі қолайсыз"}
INK, MUTED, GRID = "#1a1a19", "#52514e", "#dcdbd5"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 220,
    "savefig.bbox": "tight", "lines.linewidth": 1.6,
})


def save(fig, out, name):
    fig.savefig(f"{out}/{name}.png", facecolor="white")
    plt.close(fig)


# --- 1. architecture ---------------------------------------------------------
def architecture(out):
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, title, lines, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,"
                                    "rounding_size=0.12", fc=fc, ec=MUTED,
                                    lw=1.0))
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="center",
                fontsize=9.5, fontweight="bold")
        for k, t in enumerate(lines):
            ax.text(x + 0.18, y + h - 0.68 - 0.36 * k, t, ha="left",
                    va="center", fontsize=8.2, color=INK)

    def arrow(p, q, text=None, tx=0, ty=0, ha="center"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11,
                                     color=MUTED, lw=1.2))
        if text:
            ax.text((p[0] + q[0]) / 2 + tx, (p[1] + q[1]) / 2 + ty, text,
                    ha=ha, va="center", fontsize=8, color=MUTED)

    box(0.2, 3.7, 4.3, 2.2, "1. Физикалық симулятор",
        ["Грэд—Шафранов тепе-теңдігі (t = 0)", "d/dt[M(x)I] + RI = V",
         "масасыз плазма: күш балансы F(x) = 0", "24 камера сегменті, 56 жіпше"],
        "#e8f0fb")
    box(5.5, 3.7, 4.3, 2.2, "2. Датчиктер мен кідіріс",
        ["FIFO кідіріс τ = 1–3 мс", "гаусстық шу N(0, σ²)",
         "интегратор дрейфі d·t", "қалыпқа келтіру S = (y − μ)/σₛ"], "#fdf0e9")
    box(5.5, 0.3, 4.3, 2.2, "3. Нейрожелілік контроллер",
        ["Actor: π(A | S), тек S оқиды", "Critic: V(S, Z), Z — привилегиялы",
         "PPO + GAE, рандомизация", "A = tanh(W·h + b) ∈ [−1, 1]⁸"], "#e9f7f1")
    box(0.2, 0.3, 4.3, 2.2, "4. Қауіпсіздік сүзгісі",
        ["QP: min ‖W(V − V_target)‖²", "|V| ≤ V_max, |ΔV| ≤ ΔV_max",
         "|I(k+H)| ≤ I_max (H = 5)", "ADMM + active-set polish"], "#efedf8")
    arrow((4.5, 4.8), (5.5, 4.8), "y(t)", ty=0.28)
    arrow((7.65, 3.7), (7.65, 2.5), "S_t", tx=0.35)
    arrow((5.5, 1.4), (4.5, 1.4), "V_target", ty=0.28)
    arrow((2.35, 2.5), (2.35, 3.7), "V*", tx=0.32)
    ax.text(5.0, 6.1, "Δt = 1 мс, f = 1 кГц (lock-step)", ha="center",
            fontsize=8.5, color=MUTED)
    save(fig, out, "fig1_architecture")


# --- 2. training curves --------------------------------------------------------
def training(out, runs):
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for name, path in runs.items():
        h = json.load(open(path))
        st = [r["steps"] / 1e3 for r in h["ppo"]]
        axs[0].plot(st, [r["return"] for r in h["ppo"]], color=COL[name],
                    ls=DASH[name], label=LABEL[name])
        axs[1].plot(st, [r["rms_position_error_mm"] for r in h["ppo"]],
                    color=COL[name], ls=DASH[name], label=LABEL[name])
    first = json.load(open(next(iter(runs.values()))))
    axs[0].axhline(first["expert"]["return"], color=COL["PID"], lw=1.0,
                   ls=DASH["PID"], label=LABEL["PID"])
    axs[1].axhline(first["expert"]["rms_position_error_mm"], color=COL["PID"],
                   lw=1.0, ls=DASH["PID"])
    axs[0].set_xlabel("PPO қадамдары, мың"); axs[0].set_ylabel("Эпизод қайтарымы")
    axs[1].set_xlabel("PPO қадамдары, мың")
    axs[1].set_ylabel("RMS орын қатесі, мм")
    axs[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, out, "fig2_training")


# --- 3. traces -----------------------------------------------------------------
def traces(out, res):
    tr = res["traces"]
    order = [k for k in ("open_loop", "PID", "PPO-DR-AC") if k in tr]
    fig, axs = plt.subplots(3, 1, figsize=(7.0, 6.0), sharex=True)
    for k in order:
        t = np.array(tr[k]["t"]) * 1e3
        z = np.array(tr[k]["Z_c"]) * 1e3
        r = (np.array(tr[k]["R_c"]) - np.array(tr[k]["R_ref"])) * 1e3
        ip = (np.array(tr[k]["Ip"]) / tr[k]["Ip"][0] - 1.0) * 100
        for ax, y in zip(axs, (z, r, ip)):
            ax.plot(t, y, color=COL[k], ls=DASH[k], label=LABEL[k])
    zr = (np.array(tr["PID"]["Z_ref"]) if "PID" in tr else 0) * 1e3
    axs[0].plot(np.array(tr["PID"]["t"]) * 1e3, zr, color=INK, lw=0.8,
                ls=(0, (1, 2)), label="Тірек Z_ref")
    axs[0].set_ylim(-30, 30)
    axs[0].set_ylabel("Z_c, мм"); axs[1].set_ylabel("R_c − R_ref, мм")
    axs[2].set_ylabel("ΔI_p / I_p, %"); axs[2].set_xlabel("t, мс")
    axs[2].set_ylim(-2, 2); axs[1].set_ylim(-15, 15)
    axs[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.22),
                  fontsize=8)
    fig.tight_layout()
    save(fig, out, "fig3_traces")


# --- 4. out-of-distribution plants -----------------------------------------------
def ood(out, res):
    conds = list(next(iter(res["ood"].values())).keys())
    ctrls = [k for k in ("PID", "PPO-0", "PPO-DR", "PPO-DR-AC")
             if k in res["ood"]]
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.7), sharey=True,
                            gridspec_kw={"width_ratios": [1, 1]})
    y = np.arange(len(conds))[::-1]
    for j, k in enumerate(ctrls):
        off = (j - (len(ctrls) - 1) / 2) * 0.17
        axs[0].scatter([res["ood"][k][c]["survival"] * 100 for c in conds],
                       y + off, s=34, marker=MARK[k], color=COL[k],
                       label=LABEL[k], zorder=3)
        xs, ys = [], []
        for yy, c in zip(y, conds):
            d = res["ood"][k][c]
            if d["pos_mm_mean"] is not None and d["survival"] > 0:
                xs.append(d["pos_mm_mean"]); ys.append(yy + off)
        axs[1].scatter(xs, ys, s=34, marker=MARK[k], color=COL[k], zorder=3)
    axs[0].set_yticks(y); axs[0].set_yticklabels([COND_KK.get(c, c) for c in conds])
    axs[0].set_xlabel("Аман қалу, %"); axs[0].set_xlim(-5, 105)
    axs[1].set_xlabel("RMS орын қатесі (аман қалғандарда), мм")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(ctrls), fontsize=8,
               bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    save(fig, out, "fig4_ood")


# --- 5. delay sweep ---------------------------------------------------------------
def delay(out, res):
    ctrls = [k for k in ("PID", "PPO-0", "PPO-DR", "PPO-DR-AC")
             if k in res["delay"]]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for k in ctrls:
        ds = sorted(res["delay"][k], key=int)
        x = [int(d) for d in ds]
        axs[0].plot(x, [res["delay"][k][d]["survival"] * 100 for d in ds],
                    color=COL[k], ls=DASH[k], marker=MARK[k], ms=5,
                    label=LABEL[k])
        axs[1].plot(x, [res["delay"][k][d]["pos_mm_mean"] if
                        res["delay"][k][d]["pos_mm_mean"] is not None
                        else np.nan for d in ds],
                    color=COL[k], ls=DASH[k], marker=MARK[k], ms=5)
    for ax in axs:
        ax.axvspan(1, 3, color="#efeee8", zorder=0)
        ax.set_xlabel("Датчик кідірісі, мс")
    axs[0].set_ylabel("Аман қалу, %"); axs[1].set_ylabel("RMS орын қатесі, мм")
    axs[0].set_ylim(-5, 105)
    axs[0].legend(fontsize=8, loc="lower left")
    axs[1].text(2, 0.93, "оқу аймағы", ha="center", fontsize=7.5, color=MUTED,
                transform=axs[1].get_xaxis_transform())
    fig.tight_layout()
    save(fig, out, "fig5_delay")


# --- 6. safety filter ablation ------------------------------------------------------
def safety(out, res):
    sf = res["safety"]
    names = [k for k in ("PPO-0", "PPO-DR", "PPO-DR-AC") if k in sf]
    conds = list(next(iter(sf.values())).keys())
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.1))
    w = 0.12
    for ax, key, lab in ((axs[0], "over_limit_frac",
                          "Ток шегі бұзылған қадамдар, %"),
                         (axs[1], "survival", "Аман қалу, %")):
        for c_i, c in enumerate(conds):
            for n_i, n in enumerate(names):
                x0 = c_i + (n_i - 1) * 0.30
                on = sf[n][c]["filter_on"][key] * 100
                off = sf[n][c]["filter_off"][key] * 100
                ax.bar(x0 - w / 2 - 0.005, on, w, color=COL[n], lw=0)
                ax.bar(x0 + w / 2 + 0.005, off, w, color="white", ec=COL[n],
                       lw=1.1, hatch="/////")
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([COND_KK.get(c, c) for c in conds], fontsize=8)
        ax.set_ylabel(lab); ax.grid(axis="x", visible=False)
    axs[1].set_ylim(0, 105)
    for n in names:
        axs[1].bar([0], [0], color=COL[n], label=n + ", сүзгі қосулы")
    axs[1].bar([0], [0], color="white", ec=INK, hatch="/////",
               label="штрихталған — ток шегі өшірулі")
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=7.5,
               bbox_to_anchor=(0.5, 1.09))
    fig.tight_layout()
    save(fig, out, "fig6_safety")


# --- 7. controllability limit ----------------------------------------------------------
def growth(out, res):
    rows = res["growth"]
    dts = sorted({r["dt_ms"] for r in rows}, reverse=True)
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    for i, dt in enumerate(dts):
        rr = sorted([r for r in rows if r["dt_ms"] == dt],
                    key=lambda r: r["gamma"])
        for r in rr:
            ax.scatter(r["gamma"], i, s=46, zorder=3,
                       marker="o" if r["stabilisable"] else "x",
                       color="#2a78d6" if r["stabilisable"] else "#e34948")
    ax.set_xscale("log")
    ax.set_yticks(range(len(dts)))
    ax.set_yticklabels([f"Δt = {dt:g} мс\n({1/dt:g} кГц)" for dt in dts])
    ax.set_xlabel("Ашық контурдың өсу жылдамдығы γ, с⁻¹")
    ax.axvline(1 / 2e-3, color=INK, lw=0.8, ls=(0, (1, 2)))
    ax.text(1 / 2e-3 * 1.05, len(dts) - 0.55, "γ·τ = 1", fontsize=8,
            color=MUTED)
    ax.scatter([], [], marker="o", color="#2a78d6", label="ПД ұстай алады")
    ax.scatter([], [], marker="x", color="#e34948", label="ұстай алмайды")
    ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(0.0, 1.0))
    ax.set_ylim(-0.5, len(dts) + 0.35)
    fig.tight_layout()
    save(fig, out, "fig7_growth")


if __name__ == "__main__":
    res = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    runs = {}
    for a in sys.argv[3:]:
        n, p = a.split("=", 1)
        runs[n] = p
    architecture(out)
    if runs:
        training(out, runs)
    for f in (traces, ood, delay, safety, growth):
        try:
            f(out, res)
        except KeyError as e:
            print(f"{f.__name__}: missing {e}")
