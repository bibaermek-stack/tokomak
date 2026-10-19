#!/usr/bin/env python3
"""All figures of the article (white background, black text, coloured data).   python make_figures.py [name ...]   (default: every figure whose data exists)

Data come from data/*.json (frozen copies of the experiment outputs; tools/collect_data.py shows where each one is made).
"""
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from matplotlib.colors import to_rgba

from figstyle import CYC, C_B, C_ENS, C_IP, C_O, C_QL, C_SH, C_TX, C_V0, HATCH, K, LS, MK, OUT, comma, panel, plt, save

HERE = Path(__file__).resolve().parent
D = {p.stem: json.loads(p.read_text()) for p in sorted((HERE / "data").glob("*.json"))}
S, T = D.get("shrx_results"), D.get("torax_results")
FIGS = {}


def figure(f):
    FIGS[f.__name__] = f
    return f


# ----------------------------------------------------------------------------------------------- 1. pipeline
@figure
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(6.9, 3.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 51)
    ax.axis("off")

    def box(x, y, w, h, lines, dashed=False, lw=1.2, fs=7.8, col=K):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.0", fc=to_rgba(col, 0.07), ec=col, lw=lw,
                                    ls=LS["--"] if dashed else "-"))
        n = len(lines)
        for i, t in enumerate(lines):
            ax.text(x + w / 2, y + h / 2 + (n - 1) * 2.2 - i * 4.4, t, ha="center", va="center", fontsize=fs if i else fs + 0.4,
                    fontweight="bold" if (i == 0 and not t.startswith("$")) else "normal", color=col if i == 0 else K)

    def arrow(x0, y0, x1, y1, dashed=False):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=0.9, color=K,
                                     ls=LS["--"] if dashed else "-", shrinkA=0, shrinkB=0))

    W, Y, H = 21.4, 33, 15
    xs = [1, 26.2, 51.4, 77.6]
    box(xs[0], Y, W, H, ["SINTEZ күйі", "$n_e,\\,T_e,\\,T_i,\\,q$", "$\\kappa,\\,B_0,\\,R_0,\\,a$"], col=C_V0)
    box(xs[1], Y, W, H, ["Көпір", "$R/L_{T_i},\\,R/L_{T_e},\\,R/L_n,$", "$q,\\,\\hat{s},\\,x,\\,T_i/T_e,\\,\\nu^{*}$"], col=C_O)
    box(xs[2], Y, W, H, ["SHRX желісі", "$X\\mapsto(q_i,\\,q_e,\\,\\Gamma_e)$", "96$\\times$4, ReLU, 29\u00a0283"], col=C_SH)
    box(xs[3], Y, W, H, ["Ағын $\\to\\chi$", "$\\chi_s=\\dfrac{R}{a}\\dfrac{q_s}{R/L_{T_s}}\\chi_{GB}$", "$[0{,}05;\\,100]$ м$^2$/с"], fs=7.4, col=C_TX)
    for i in range(3):
        arrow(xs[i] + W + 0.5, Y + H / 2, xs[i + 1] - 0.5, Y + H / 2)
    box(54, 11, 45, 14, ["Жылу тасымалы теңдеуі (SINTEZ)", "$\\frac{3}{2}\\,\\partial_t(nT)=\\nabla\\!\\cdot(n\\,\\chi\\,\\nabla T)+S$"])
    arrow(xs[3] + W / 2, Y - 0.4, xs[3] + W / 2, 25.6)
    box(14, 11, 36, 14, ["Жаттығу (желіден тыс)", "QLKNN-мұғалім $\\to$ белгі $y$", "$\\mathrm{arcsinh}$-шығын, он-полиси қайта белгілеу"],
        dashed=True, fs=7.2, col=C_QL)
    arrow(40, 25.6, 56, Y - 0.4, dashed=True)  # weights
    ax.text(51.5, 29.2, "салмақтар", fontsize=7.4, ha="left", va="center")
    ax.plot([76.5, 76.5, 7, 7], [10.6, 4.5, 4.5, Y - 0.6], color=K, lw=0.9)
    ax.add_patch(FancyArrowPatch((7, Y - 3), (7, Y - 0.4), arrowstyle="-|>", mutation_scale=9, lw=0.9, color=K, shrinkA=0, shrinkB=0))
    ax.text(41.7, 5.4, "күй жаңарады (әр уақыт қадамында)", fontsize=7.4, ha="center", va="bottom")
    return save(fig, "pipeline")


# ----------------------------------------------------------------------------------------------- 3. parity
@figure
def fig_parity():
    sc = S["accuracy"]["scatter_flux_e"]
    q, s, m = (np.array(sc[k], dtype=object) for k in ("qlknn", "shrx", "machine"))
    q, s = q.astype(float), s.astype(float)
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 3.2))
    for ax, sel, ttl, lab in ((axs[0], m == "demo", "DEMO (оқудан тыс машина)", "а"), (axs[1], m == "nstx", "NSTX (оқудан тыс машина)", "б")):
        ax.plot(q[sel], s[sel], ls="none", marker="o", ms=1.6, mfc="none", mec=C_SH, mew=0.35, alpha=0.8, rasterized=True)
        lim = [1e-3, 3e2]
        ax.plot(lim, lim, color=K, lw=0.8)
        ax.plot(lim, [v * 1.25 for v in lim], color=K, lw=0.6, ls=LS["--"])
        ax.plot(lim, [v / 1.25 for v in lim], color=K, lw=0.6, ls=LS["--"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("$q_e^{\\mathrm{QLKNN}}$, гиро-Бом бірлігі")
        ax.set_ylabel("$q_e^{\\mathrm{SHRX}}$, гиро-Бом бірлігі")
        ax.set_title(ttl, fontsize=8.5, loc="left", pad=4)
        ax.set_aspect("equal")
        panel(ax, lab, dx=-0.0, dy=1.1)
    fig.subplots_adjust(wspace=0.32)
    return save(fig, "parity", vector=False)


# ----------------------------------------------------------------------------------------------- 4. local chi
@figure
def fig_local_chi():
    loc = S["local"]
    names = [("A", "TORAX ITER күйі"), ("B1", "SINTEZ, $T_{e0}\\approx 6$ кэВ"), ("B2", "SINTEZ, $T_{e0}\\approx 24$ кэВ")]
    fig, axs = plt.subplots(2, 3, figsize=(6.9, 4.4), sharex=True)
    for j, (key, ttl) in enumerate(names):
        d = loc[key]
        rho = np.array(d["rho"])
        for i, ch in enumerate(("e", "i")):
            ax = axs[i, j]
            c = d[f"chi_{ch}"]
            ax.plot(rho, c["qlknn_bridge"], color=C_QL, lw=1.7, label="QLKNN (көпір арқылы)")
            ax.plot(rho, c["shrx"], color=C_SH, lw=1.2, ls=LS["--"], label="SHRX")
            if c.get("torax_native"):
                ax.plot(rho, c["torax_native"], color=C_TX, lw=0, marker="o", ms=3.2, mfc="white", mew=0.8, label="TORAX (өз есебі)")
            ax.set_yscale("log")
            ax.set_ylim(0.03, 150)
            if i == 0:
                ax.set_title(ttl, fontsize=8.5, loc="left")
            if j == 0:
                ax.set_ylabel(f"$\\chi_{ch}$, м$^2$/с")
            if i == 1:
                ax.set_xlabel("$\\rho$")
            panel(ax, "абвгде"[i * 3 + j], dx=0.02, dy=0.86) if False else None
    h, l = axs[0, 0].get_legend_handles_labels()
    h2, l2 = axs[0, 1].get_legend_handles_labels()
    fig.legend(h2, l2, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.subplots_adjust(hspace=0.12, wspace=0.28)
    return save(fig, "local_chi")


# ----------------------------------------------------------------------------------------------- 5. ITER traces
@figure
def fig_iter_trace():
    runs = S["sintez_runs"]
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.9), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axs[0]
    spec = [("qlknn_ref", "SINTEZ + QLKNN", "-", 1.6, C_QL), ("shrx", "SINTEZ + SHRX", "--", 1.3, C_SH),
            ("scaling_anchored", "SINTEZ, IPB98-ге бекітілген", "-.", 1.1, C_IP), ("shrx_v0", "SINTEZ + SHRX v0", ":", 1.6, C_V0)]
    for key, lab, ls, lw, col in spec:
        tr = runs[key]["trace"]
        ax.plot(tr["t"], tr["Pfus"], color=col, ls=LS[ls], lw=lw, label=lab)
    ax.axhline(500, color=K, lw=0.5, ls=LS[":"])
    ax.text(3, 512, "ITER жобасы 500 МВт", fontsize=7.4)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 640)
    ax.set_xlabel("$t$, с")
    ax.set_ylabel("$P_{\\mathrm{fus}}$, МВт")
    ax.legend(loc="lower right", fontsize=7.4)
    panel(ax, "а")
    ax = axs[1]
    tx = T["torax"]["main"]
    ax.plot(tx["trace_t"], tx["trace_Pfus"], color=C_TX, lw=1.4, label="TORAX (QLKNN)")
    for key, ls, lab, col in (("qlknn_ref", "-", "SINTEZ + QLKNN, жазық қиыр", C_QL), ("shrx", "--", "SINTEZ + SHRX, жазық қиыр", C_SH)):
        ax.axhline(runs[key]["final"]["Pfus"], color=col, lw=1.0, ls=LS[ls], label=lab)
    ax.axhline(500, color=K, lw=0.5, ls=LS[":"])
    ax.set_xlim(0, max(tx["trace_t"]))
    ax.set_ylim(0, 640)
    ax.set_xlabel("TORAX модельдеу уақыты, с")
    ax.legend(loc="lower right", fontsize=7.0)
    panel(ax, "б")
    fig.subplots_adjust(wspace=0.22)
    return save(fig, "iter_trace")


# ----------------------------------------------------------------------------------------------- 6. ITER profiles
@figure
def fig_iter_profiles():
    runs = S["sintez_runs"]
    tx = T["torax"]["main"]["profiles"]
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.6))
    for ax, key, lab, yl in ((axs[0], "Te", "а", "$T_e$, кэВ"), (axs[1], "Ti", "б", "$T_i$, кэВ"), (axs[2], "q", "в", "$q$")):
        for run, ls, lw, name, col in (("qlknn_ref", "-", 1.6, "SINTEZ + QLKNN", C_QL), ("shrx", "--", 1.3, "SINTEZ + SHRX", C_SH)):
            p = runs[run]["profiles"]
            ax.plot(p["rho"], p[key], color=col, ls=LS[ls], lw=lw, label=name)
        if key == "q":
            ax.plot(tx["rho_face"], tx["q"], color=C_TX, lw=0, marker="o", ms=2.8, mfc="white", mew=0.7, label="TORAX")
        else:
            ax.plot(tx["rho_cell"], tx["T_e" if key == "Te" else "T_i"], color=C_TX, lw=0, marker="o", ms=2.8, mfc="white", mew=0.7, label="TORAX")
        ax.set_xlabel("$\\rho$")
        ax.set_ylabel(yl)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, None)
        panel(ax, lab)
    axs[0].legend(loc="upper right", fontsize=7.4)
    fig.subplots_adjust(wspace=0.36)
    return save(fig, "iter_profiles")


# ----------------------------------------------------------------------------------------------- 7. in-loop, 12 machines
@figure
def fig_machines():
    M = S["machines"]
    keys = [k for k in M if "qlknn_ref" in M[k] and "shrx" in M[k]]
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.7))
    for ax, (fld, lab, ttl) in zip(axs, (("Wth", "а", "$W_{th}$, МДж"), ("Te", "б", "$\\langle T_e\\rangle$, кэВ"), ("tauE", "в", "$\\tau_E$, с"))):
        xs = np.array([max(M[k]["qlknn_ref"][fld], 1e-3) for k in keys])
        ys = np.array([max(M[k]["shrx"][fld], 1e-3) for k in keys])
        for i, k in enumerate(keys):
            ax.plot(xs[i], ys[i], ls="none", marker=MK[i % len(MK)], ms=4.4, mfc=to_rgba(CYC[i % len(CYC)], 0.3), mec=CYC[i % len(CYC)], mew=0.9, label=k.upper())
        lo, hi = min(xs.min(), ys.min()) / 1.6, max(xs.max(), ys.max()) * 1.6
        ax.plot([lo, hi], [lo, hi], color=K, lw=0.8)
        ax.plot([lo, hi], [lo * 1.3, hi * 1.3], color=K, lw=0.5, ls=LS["--"])
        ax.plot([lo, hi], [lo / 1.3, hi / 1.3], color=K, lw=0.5, ls=LS["--"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(ttl + " (SINTEZ + QLKNN)")
        ax.set_ylabel(ttl + " (SINTEZ + SHRX)")
        ax.set_aspect("equal")
        panel(ax, lab, dy=1.03)
    axs[2].legend(loc="center left", bbox_to_anchor=(1.04, 0.5), fontsize=7, ncol=1, handletextpad=0.3)
    fig.subplots_adjust(wspace=0.5)
    return save(fig, "machines")


# ----------------------------------------------------------------------------------------------- 8. latency
@figure
def fig_latency():
    L = S["latency"]["one_thread"]
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    series = [("SHRX желісі (float32)", "shrx_network", C_SH), ("QLKNN, JAX jit (float32)", "qlknn_jit_float32", C_QL),
              ("QLKNN, JAX jit (float64)", "qlknn_jit_float64", C_B), ("SHRX, толық шақыру", "shrx_evaluate_full", "#F2A97E"),
              ("QLKNN, толық шақыру", "qlknn_evaluate_full", "#A9CDE6")]
    w = 0.16
    for i, (lab, key, col) in enumerate(series):
        vals = [L[f"N{n}"][key]["median_us"] for n in (26, 65)]
        lo = [L[f"N{n}"][key]["median_us"] - L[f"N{n}"][key]["range_us"][0] for n in (26, 65)]
        hi = [L[f"N{n}"][key]["range_us"][1] - L[f"N{n}"][key]["median_us"] for n in (26, 65)]
        x = np.arange(2) + (i - 2) * w
        ax.bar(x, vals, w * 0.92, fc=col, ec=K, lw=0.6, label=lab, yerr=[lo, hi], error_kw=dict(lw=0.6, capsize=1.5))
        for xi, v, h_ in zip(x, vals, hi):
            ax.text(xi, (v + h_) * 1.08, f"{v:.0f}", ha="center", va="bottom", fontsize=6.6)
    ax.set_yscale("log")
    ax.set_ylim(30, 4000)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["26 түйін", "65 түйін"])
    ax.set_ylabel("бір шақыру уақыты, мкс")
    ax.legend(loc="upper left", ncol=2, fontsize=7.0)
    ax.set_ylim(30, 7000)
    return save(fig, "latency")


# ----------------------------------------------------------------------------------------------- 9. ensemble
@figure
def fig_ensemble():
    E = D["ensemble"]
    q = np.linspace(0, 1, 201)
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.7))
    ax = axs[0]
    for name, lab in (("nstx", "NSTX"), ("demo", "DEMO")):
        for key, tag in (("err_quantiles_single", "жалғыз желі"), ("err_quantiles_ensemble", "ансамбль")):
            ls = {("nstx", "err_quantiles_single"): "--", ("nstx", "err_quantiles_ensemble"): "-",
                  ("demo", "err_quantiles_single"): ":", ("demo", "err_quantiles_ensemble"): "-."}[(name, key)]
            ax.plot(np.maximum(E[name][key], 1e-4), q, color=C_V0 if "single" in key else C_ENS, ls=LS[ls], lw=1.3, label=f"{lab}, {tag}")
    ax.axvline(0.25, color=K, lw=0.5, ls=LS[":"])
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 3)
    ax.set_ylim(0, 1)
    ax.set_xlabel("$|\\chi_e^{\\mathrm{NN}}-\\chi_e^{\\mathrm{QLKNN}}|/\\chi_e^{\\mathrm{QLKNN}}$")
    ax.set_ylabel("үлесі (жинақталған)")
    ax.legend(loc="upper left", fontsize=6.4, handlelength=2.8)
    panel(ax, "а")
    ax = axs[1]
    for name, lab, mk, ls, col in (("nstx", "NSTX", "o", "-", C_O), ("demo", "DEMO", "s", "--", C_B)):
        ax.plot(np.arange(1, 11), [100 * v for v in E[name]["median_err_per_decile"]], color=col, ls=LS[ls], marker=mk, ms=3.8, mfc="white", mew=0.8,
                label=lab)
    ax.set_xlabel("алшақтық децилі ($s$ бойынша)")
    ax.set_ylabel("қатенің медианасы, %")
    ax.set_xticks(range(1, 11, 3))
    ax.legend(loc="upper left")
    panel(ax, "б")
    ax = axs[2]
    for name, mk, col in (("nstx", "o", C_O), ("demo", "x", C_B)):
        ax.plot(E[name]["scatter_spread"], E[name]["scatter_err"], ls="none", marker=mk, ms=1.6 if mk == "o" else 2.0, mfc="none", mec=col, mew=0.3,
                alpha=0.8, rasterized=True)
    ax.axvline(0.20, color=K, lw=0.9, ls=LS["--"])
    ax.axhline(0.25, color=K, lw=0.9, ls=LS[":"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-3, 3)
    ax.set_ylim(1e-3, 10)
    ax.set_xlabel("мүшелер арасындағы алшақтық $s$")
    ax.set_ylabel("салыстырмалы қате")
    panel(ax, "в")
    fig.subplots_adjust(wspace=0.5)
    return save(fig, "ensemble", vector=False)


@figure
def fig_threshold():
    E = D["ensemble"]
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.6), sharey=True)
    bins = np.linspace(-0.25, 0.25, 51)
    for ax, key, ttl, lab in ((axs[0], "threshold_train", "оқу машиналары", "а"), (axs[1], "threshold_held_out", "оқудан тыс машиналар (NSTX, DEMO)", "б")):
        for field, ls, name, col in (("err_single", "--", "жалғыз желі", C_V0), ("err_ensemble", "-", "ансамбль", C_ENS)):
            e = np.array(E[key][field])
            h, _ = np.histogram(np.clip(e, -0.25, 0.25), bins=bins)
            ax.stairs(100 * h / h.sum(), bins, color=col, ls=LS[ls], lw=1.3, label=f"{name}: |ε| медианасы {100 * np.median(np.abs(e)):.1f} %".replace(".", ","))
        ax.axvline(0, color=K, lw=0.5)
        ax.set_xlabel("$f_c^{\\mathrm{NN}}/f_c^{\\mathrm{QLKNN}}-1$")
        ax.set_title(ttl, fontsize=8.5, loc="left", pad=3)
        ax.legend(loc="upper left", fontsize=6.6)
        panel(ax, lab, dy=1.1)
    axs[0].set_ylabel("күйлер үлесі, %")
    fig.subplots_adjust(wspace=0.08)
    return save(fig, "threshold")


@figure
def fig_inloop():
    I = D["inloop"]
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.7), gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axs[0]
    names = ["shipped", "member0", "member1", "member2", "member3", "member4", "ensemble"]
    vals = [I["iter"][n]["Pfus"] for n in names]
    x = np.arange(len(names))
    for xi, n, v in zip(x, names, vals):
        col = C_ENS if n == "ensemble" else C_SH
        ax.plot(xi, v, ls="none", marker="D" if n == "ensemble" else ("s" if n == "shipped" else "o"), ms=5, mfc=C_ENS if n == "ensemble" else "white",
                mec=col, mew=1.0)
    ax.axhline(S["sintez_runs"]["qlknn_ref"]["final"]["Pfus"], color=C_QL, lw=1.0, ls=LS["--"])
    ax.text(0.0, S["sintez_runs"]["qlknn_ref"]["final"]["Pfus"] + 1.5, "SINTEZ + QLKNN", fontsize=7, color=C_QL)
    ax.set_xticks(x)
    ax.set_xticklabels(["жеткіз.", "№1", "№2", "№3", "№4", "№5", "ансамбль"], fontsize=6.8, rotation=35, ha="right")
    ax.set_ylabel("$P_{\\mathrm{fus}}$, МВт (ITER)")
    ax.set_ylim(440, 520)
    panel(ax, "а")
    ax = axs[1]
    machs = ["jet", "diiid", "kstar", "ktm"]
    for i, m in enumerate(machs):
        mem = [I["spread"][m]["members"][k]["Te"] for k in I["spread"][m]["members"]]
        ax.plot([i] * len(mem), mem, ls="none", marker="o", ms=4.2, mfc="white", mec=C_SH, mew=0.9, label="алты желі мүшесі" if i == 0 else None)
        ens = I["spread"][m]["ensemble"]["Te"]
        ax.plot(i + 0.18, ens, ls="none", marker="D", ms=5, mfc=C_ENS, mec=C_ENS, label="ансамбль" if i == 0 else None)
        fl = I["floor"].get(m)
        if fl:
            f = [v["Te"] for v in fl.values()]
            ax.plot([i - 0.2] * len(f), f, ls="none", marker="x", ms=5, mec=C_O, mew=1.0,
                    label="бастапқы $T_e$ 10$^{-6}$–10$^{-3}$ өзгертілген (жеткізілетін желі)" if i == 1 else None)
    ax.set_yscale("log")
    ax.set_xticks(range(len(machs)))
    ax.set_xticklabels([m.upper() for m in machs])
    ax.set_ylabel("$\\langle T_e\\rangle$, кэВ (жазық қиыр)")
    ax.legend(loc="lower left", fontsize=6.4)
    panel(ax, "б")
    fig.subplots_adjust(wspace=0.3)
    return save(fig, "inloop")


@figure
def fig_chaos():
    C = D["chaos"]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    mk = {"iter": "o", "jet": "s", "diiid": "^", "kstar": "x"}
    mcol = {"iter": C_QL, "jet": C_SH, "diiid": C_TX, "kstar": C_IP}
    for m, v in C.items():
        t = np.array(v["t"])
        for delta, ls, lw in (("1e-12", "-", 1.1), ("1e-06", "--", 0.9)):
            y = np.maximum(v["deviation"][delta], 1e-16)
            ax.plot(t, y, color=mcol[m], ls=LS[ls], lw=lw, marker=mk[m], markevery=14, ms=3.6, mfc="white", mew=0.7,
                    label=f"{m.upper()}, $\\delta={delta.replace('1e-', '10^{-').replace('06', '6}').replace('12', '12}')}$" if False else None)
    # legend: machines by marker, delta by line style
    for m in C:
        ax.plot([], [], color=mcol[m], marker=mk[m], ms=3.8, mfc="white", mew=0.8, ls="none", label=m.upper())
    ax.plot([], [], color=K, ls=LS["-"], label="$\\delta=10^{-12}$")
    ax.plot([], [], color=K, ls=LS["--"], label="$\\delta=10^{-6}$")
    ax.axhline(2.2e-16, color=K, lw=0.5, ls=LS[":"])
    ax.text(1, 3e-16, "машиналық эпсилон", fontsize=7)
    ax.set_yscale("log")
    ax.set_xlabel("$t$, с")
    ax.set_ylabel("$\\max_\\rho|T_e-T_e^{\\mathrm{ref}}|/T_e^{\\mathrm{ref}}$")
    ax.set_xlim(0, 60)
    ax.set_ylim(1e-16, 30)
    ax.legend(loc="lower right", ncol=2, fontsize=7)
    return save(fig, "chaos")


# ----------------------------------------------------------------------------------------------- DB5 (experimental database)
def _mk(machines):
    return {m: MK[i % len(MK)] for i, m in enumerate(sorted(machines))}


def _mc(machines):
    return {m: CYC[i % len(CYC)] for i, m in enumerate(sorted(machines))}


def _mkr(ax, xs, ys, mk, col, **kw):
    """Open marker with a faint fill in the machine colour."""
    return ax.plot(xs, ys, ls="none", marker=mk, ms=kw.pop("ms", 3.8), mfc=to_rgba(col, 0.25), mec=col, mew=0.7, **kw)


def _parity(ax, x, y, mk, sel_label=None, lim=(0.02, 30), fill=None):
    ax.plot(lim, lim, color=K, lw=0.8)
    ax.plot(lim, [v * 1.3 for v in lim], color=K, lw=0.5, ls=LS["--"])
    ax.plot(lim, [v / 1.3 for v in lim], color=K, lw=0.5, ls=LS["--"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")


@figure
def fig_db5_parity():
    B = D["db5"]
    recs = [d for d in B["records"] if d.get("ok")]
    machines = sorted(set(d["tok"] for d in recs if d.get("valid")))
    mk = _mk(machines + sorted(set(d["tok"] for d in recs) - set(machines)))
    mc = _mc(machines + sorted(set(d["tok"] for d in recs) - set(machines)))
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.9))
    # (a) SINTEZ + SHRX: every run (small crosses) and the valid ones (markers)
    ax = axs[0]
    _parity(ax, None, None, mk)
    xs = [d["W_exp"] for d in recs if not d.get("valid")]
    ys = [d["W_shrx"] for d in recs if not d.get("valid")]
    ax.plot(xs, ys, ls="none", marker="+", ms=2.6, mec="#9A9A9A", mew=0.4, label="жарамсыз жүгіру", rasterized=True)
    for tok in machines:
        sel = [d for d in recs if d["tok"] == tok and d.get("valid")]
        _mkr(ax, [d["W_exp"] for d in sel], [d["W_shrx"] for d in sel], mk[tok], mc[tok], ms=3.8, label=tok)
    ax.set_xlabel("$W_\\mathrm{th}^{\\mathrm{exp}}$, МДж")
    ax.set_ylabel("$W_\\mathrm{th}^{\\mathrm{SINTEZ+SHRX}}$, МДж")
    panel(ax, "а", dy=1.03)
    # (b) TORAX on the valid runs
    ax = axs[1]
    _parity(ax, None, None, mk)
    for tok in machines:
        sel = [d for d in recs if d["tok"] == tok and "W_torax" in d]
        if sel:
            _mkr(ax, [d["W_exp"] for d in sel], [d["W_torax"] for d in sel], mk[tok], mc[tok], ms=3.8)
    ax.set_xlabel("$W_\\mathrm{th}^{\\mathrm{exp}}$, МДж")
    ax.set_ylabel("$W_\\mathrm{th}^{\\mathrm{TORAX}}$, МДж")
    panel(ax, "б", dy=1.03)
    # (c) IPB98(y,2) on the same records
    ax = axs[2]
    _parity(ax, None, None, mk)
    for tok in machines:
        sel = [d for d in recs if d["tok"] == tok and "W_torax" in d and d["H98"]]
        if sel:
            _mkr(ax, [d["W_exp"] for d in sel], [d["W_exp"] / d["H98"] for d in sel], mk[tok], mc[tok], ms=3.8)
    ax.set_xlabel("$W_\\mathrm{th}^{\\mathrm{exp}}$, МДж")
    ax.set_ylabel("$W_\\mathrm{th}^{\\mathrm{IPB98(y,2)}}$, МДж")
    panel(ax, "в", dy=1.03)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=6.8, bbox_to_anchor=(0.5, -0.07), handletextpad=0.2, columnspacing=1.0)
    fig.subplots_adjust(wspace=0.38)
    return save(fig, "db5_parity", vector=False)


@figure
def fig_db5_cdf():
    B = D["db5"]
    recs = [d for d in B["records"] if d.get("ok")]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    series = [("SINTEZ + SHRX, барлық жүгіру", [d["W_shrx"] / d["W_exp"] for d in recs], "..", 1.2, C_SH),
              ("SINTEZ + SHRX, жарамды жүгіру", [d["W_shrx"] / d["W_exp"] for d in recs if d.get("valid")], "-", 1.6, C_SH),
              ("TORAX, жарамды жүгіру", [d["W_torax"] / d["W_exp"] for d in recs if "W_torax" in d], "--", 1.4, C_TX),
              ("IPB98(y,2) регрессиясы", [1.0 / d["H98"] for d in recs if d.get("valid") and d["H98"]], "-.", 1.4, C_IP)]
    for lab, r, ls, lw, col in series:
        a = np.sort(np.abs(np.log(np.array(r))))
        ax.plot(a, np.arange(1, len(a) + 1) / len(a), color=col, ls=LS[ls], lw=lw, label=f"{lab} ($N={len(a)}$)")
    ax.axvline(np.log(1.3), color=K, lw=0.5, ls=LS[":"])
    ax.text(np.log(1.3) + 0.02, 0.03, "$\\pm30$ %", fontsize=7.4)
    ax.set_xlabel("$|\\ln(W^{\\mathrm{pred}}/W^{\\mathrm{exp}})|$")
    ax.set_ylabel("жазбалар үлесі (жинақталған)")
    ax.set_xlim(0, 2.2)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=6.9)
    return save(fig, "db5_cdf")


@figure
def fig_db5_pair():
    B = D["db5"]
    pr = [d for d in B["records"] if "W_torax" in d]
    mk = _mk(set(d["tok"] for d in pr))
    mc = _mc(set(d["tok"] for d in pr))
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    lim = (0.03, 30)
    ax.plot(lim, lim, color=K, lw=0.8)
    ax.plot(lim, [v * 1.3 for v in lim], color=K, lw=0.5, ls=LS["--"])
    ax.plot(lim, [v / 1.3 for v in lim], color=K, lw=0.5, ls=LS["--"])
    for tok in sorted(set(d["tok"] for d in pr)):
        sel = [d for d in pr if d["tok"] == tok]
        _mkr(ax, [d["W_torax"] for d in sel], [d["W_shrx"] for d in sel], mk[tok], mc[tok], ms=4.0, label=tok)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("$W_\\mathrm{th}^{\\mathrm{TORAX}}$, МДж")
    ax.set_ylabel("$W_\\mathrm{th}^{\\mathrm{SINTEZ+SHRX}}$, МДж")
    ax.legend(loc="upper left", fontsize=6.4, ncol=2, handletextpad=0.2)
    return save(fig, "db5_pair")


@figure
def fig_core():
    B = D["db5"]
    cs = [d for d in B["core"] if d["good"]]
    fig, axs = plt.subplots(2, 2, figsize=(6.2, 5.6))
    mk = _mk(set(d["tok"] for d in cs))
    mc = _mc(set(d["tok"] for d in cs))
    for (i, j), (fld, code, lab) in zip(((0, 0), (0, 1), (1, 0), (1, 1)),
                                         (("Te0", "", "SINTEZ + SHRX"), ("Te0_torax", "", "TORAX"), ("Ti0", "", "SINTEZ + SHRX"), ("Ti0_torax", "", "TORAX"))):
        ax = axs[i, j]
        exp = "Te0_exp" if i == 0 else "Ti0_exp"
        lim = (0.3, 30)
        ax.plot(lim, lim, color=K, lw=0.8)
        ax.plot(lim, [v * 1.3 for v in lim], color=K, lw=0.5, ls=LS["--"])
        ax.plot(lim, [v / 1.3 for v in lim], color=K, lw=0.5, ls=LS["--"])
        for tok in sorted(set(d["tok"] for d in cs)):
            sel = [d for d in cs if d["tok"] == tok and d.get(exp) and d.get(fld) is not None]
            if sel:
                _mkr(ax, [d[exp] for d in sel], [d[fld] for d in sel], mk[tok], mc[tok], ms=4.0, label=tok if (i, j) == (0, 0) else None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        sym = "T_{e0}" if i == 0 else "T_{i0}"
        ax.set_xlabel(f"${sym}^{{\\mathrm{{exp}}}}$, кэВ")
        ax.set_ylabel(f"${sym}$ ({lab}), кэВ")
        panel(ax, "абвг"[2 * i + j], dy=1.02)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=6.8, bbox_to_anchor=(0.5, -0.02), handletextpad=0.2, columnspacing=1.0)
    fig.subplots_adjust(wspace=0.35, hspace=0.35)
    return save(fig, "core", vector=False)


@figure
def fig_st_range():
    Sx = D["st_teacher"]
    bins = np.array(Sx["log10_bins"])
    ctr = 0.5 * (bins[1:] + bins[:-1])
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.8), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axs[0]
    gcol = {"scripted ktm": C_QL, "scripted nstx": C_SH, "scripted mastu": C_TX, "records NSTX": C_IP, "records MAST": C_O, "records START": C_B}
    order = [("scripted ktm", "КТМ (сценарий)", "-"), ("scripted nstx", "NSTX (сценарий)", "--"), ("scripted mastu", "MAST-U (сценарий)", ":"),
             ("records NSTX", "NSTX (жазбалар)", "-."), ("records MAST", "MAST (жазбалар)", (0, (8, 2, 1, 2, 1, 2)))]
    for key, lab, ls in order:
        if key in Sx["groups"]:
            ax.plot(ctr, Sx["groups"][key]["xnue_hist"], color=gcol[key], ls=LS.get(ls, ls), lw=1.3, label=lab)
    lo = np.log10(max(Sx["xnue_train_mean"] - Sx["xnue_train_std"], 1e-4))
    hi = np.log10(Sx["xnue_train_mean"] + Sx["xnue_train_std"] * Sx["sigma_box"])
    ax.axvspan(lo, hi, facecolor=to_rgba(C_V0, 0.18), edgecolor=C_V0, hatch="////", lw=0.6)
    ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.92, "оқу ауқымы\n(STEP)", ha="center", va="top", fontsize=7.2,
            bbox=dict(fc="white", ec="none", pad=1.0))
    ax.set_xlabel("$\\log_{10}$ XNUE (соқтығысу жиілігі, $c_s/a$ бірлігінде)")
    ax.set_ylabel("нүктелер үлесі")
    ax.set_xlim(-4, 4)
    ax.legend(loc="upper right", fontsize=6.6)
    panel(ax, "а")
    ax = axs[1]
    groups = [g for g in ("scripted ktm", "scripted nstx", "scripted mastu", "records NSTX", "records MAST", "records START") if g in Sx["groups"]]
    feats = ["XNUE", "BETAE", "DRMAJDX_LOC", "TAUS_2", "Q_LOC", "SHAT"]
    w = 0.8 / len(groups)
    for gi, g in enumerate(groups):
        v = [100 * Sx["groups"][g]["per_feature"][f]["outside"] for f in feats]
        ax.bar(np.arange(len(feats)) + (gi - len(groups) / 2 + 0.5) * w, v, w * 0.92, fc=gcol[g], ec=K, lw=0.4,
               label=g.replace("scripted ", "сценарий ").replace("records ", "жазбалар "))
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(["XNUE", "$\\beta_e$", "d$R_0$/d$r$", "$T_i/T_e$", "$q$", "$\\hat s$"], fontsize=7.4)
    ax.set_ylabel("ауқымнан тыс нүктелер, %")
    ax.set_ylim(0, 150)
    ax.legend(loc="upper right", fontsize=6.0, ncol=2, handlelength=1.4, columnspacing=0.8)
    panel(ax, "б")
    fig.subplots_adjust(wspace=0.32)
    return save(fig, "st_range")


@figure
def fig_st_w():
    B = D["db5"]
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.9), sharey=True)
    mk = {"MAST": "o", "NSTX": "s", "START": "^"}
    mcol = {"MAST": C_O, "NSTX": C_IP, "START": C_B}
    lim = (3e-4, 60)
    for ax, key, lab, ttl in ((axs[0], "st", "а", "SINTEZ + SHRX (QLKNN)"), (axs[1], "st_tglf", "б", "SINTEZ + TGLFNN-STEP")):
        rows = B.get(key)
        ax.plot(lim, lim, color=K, lw=0.8)
        ax.plot(lim, [v * 1.3 for v in lim], color=K, lw=0.5, ls=LS["--"])
        ax.plot(lim, [v / 1.3 for v in lim], color=K, lw=0.5, ls=LS["--"])
        if rows:
            for tok in mk:
                sel = [d for d in rows if d["tok"] == tok]
                _mkr(ax, [d["W_exp"] for d in sel], [d["W_shrx" if key == "st" else "W_tglf"] for d in sel], mk[tok], mcol[tok], ms=4.0,
                     label=tok if key == "st" else None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel("$W_\\mathrm{th}^{\\mathrm{exp}}$, МДж")
        ax.set_title(ttl, fontsize=8.5, loc="left", pad=4)
        panel(ax, lab, dy=1.1)
    axs[0].set_ylabel("$W_\\mathrm{th}^{\\mathrm{pred}}$, МДж")
    axs[0].legend(loc="upper left", fontsize=7)
    fig.subplots_adjust(wspace=0.08)
    return save(fig, "st_w")


@figure
def fig_speedup():
    Sp = D["speed"]["cases"]
    keys = list(Sp)
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    x = np.arange(len(keys))
    w = 0.34
    for i, (tag, lab, col) in enumerate((("median_old", "жылдамдатуға дейін", C_V0), ("median_new", "жылдамдатудан кейін", C_SH))):
        v = [Sp[k][tag] for k in keys]
        ax.bar(x + (i - 0.5) * w, v, w * 0.92, fc=col, ec=K, lw=0.6, label=lab)
        for xi, vi in zip(x, v):
            ax.text(xi + (i - 0.5) * w, vi + 1.5, f"{vi:.0f}", ha="center", va="bottom", fontsize=7)
    for xi, k in zip(x, keys):
        ax.text(xi, max(Sp[k]["median_old"], Sp[k]["median_new"]) + 9, f"{Sp[k]['speedup']:.1f}".replace(".", ",") + "×", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace(":", ", ").replace("scaling_anchored", "IPB98-ге бекітілген").upper().replace("IPB98-ГЕ БЕКІТІЛГЕН", "IPB98-ге бекітілген") for k in keys], fontsize=7.4)
    ax.set_ylabel("жүгіру уақыты, с (120 с плазма)")
    ax.set_ylim(0, max(Sp[k]["median_old"] for k in keys) * 1.3)
    ax.legend(loc="upper right", fontsize=7.4)
    return save(fig, "speedup")


def main():
    names = sys.argv[1:] or list(FIGS)
    for n in names:
        try:
            p = FIGS[n]()
            print("ok  ", n, p.name)
        except KeyError as e:
            print("skip", n, "missing data:", e)


if __name__ == "__main__":
    main()
