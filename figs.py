"""figs.py -- figures for the Neurocomputing manuscript (matplotlib, validated categorical palette)."""
import os
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import norm

OUT = os.environ.get("MILO_OUT", "results/tables")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"CE": "#2a78d6", "FwdT": "#eb6834", "HetG": "#1baf7a", "LD": "#eda100", "LS": "#e87ba4",
       "LT-noN": "#008300", "LT": "#6250d6", "other": "#9a9893"}
NAME = {"CE": "CE", "LS": "LS", "GCE": "GCE", "FwdT": "Forward-T", "CORAL": "CORAL", "SORD": "SORD",
        "MSE": "MSE-reg", "HetG": "Het-Gauss", "LD": "MILO-LD", "LD-flat": "MILO-LD (flat)", "LT": "MILO-LT",
        "LT-noN": "MILO-LT w/o noise", "LT-noC": "MILO-LT w/o cens."}
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "savefig.dpi": 300, "figure.dpi": 110, "font.family": "DejaVu Sans"})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ------------------------------------------------------------------------- Fig. 1 framework
def fig_framework():
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 54); ax.axis("off")

    def box(x, y, w, h, txt, fc="#f4f3f0", ec=INK2, fs=7.2, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=INK,
                fontweight="bold" if bold else "normal", linespacing=1.25)

    def arr(x0, y0, x1, y1, c=INK2, ls="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, color=c, lw=0.9, linestyle=ls))

    ax.text(0.5, 52, "(a) How an ISPU label is generated (known structure)", fontsize=8, color=INK, fontweight="bold")
    xs = [0.5, 20.5, 40.5, 60.5, 80.5]
    txt = ["True concentration\n$c_{p,t}$ (latent)", "Instrument\n$\\tilde c=c\\,e^{\\varepsilon}$, $\\varepsilon\\sim N(0,s_p^2)$",
           "Regulatory map\n$f_p$ (piecewise linear)\n+ rounding", "Published integer\nsub-index $I_{p,t}$",
           "Max over pollutants\n+ class cut-offs\n→ published class $Y_t$"]
    for x, tt in zip(xs, txt):
        box(x, 39, 17.5, 10, tt)
    for i in range(4):
        arr(xs[i] + 18.3, 44, xs[i + 1] - 0.6, 44)

    ax.text(0.5, 33.5, "(b) MILO: learning the truth behind index-coded labels", fontsize=8, color=INK, fontweight="bold")
    fs = 6.4
    box(0.5, 18, 15.5, 12, "Inputs: 14-day\nwindow of published\nsub-indices, masks,\nnetwork means,\ncalendar", fs=fs)
    box(19.5, 18, 13.5, 12, "Backbone\nMLP, GRU,\nTCN or\nTransformer", fs=fs)
    box(36.5, 21.5, 16, 9, "Latent head\n$\\mu_{p}(x),\\ \\sigma_{p}(x)$\nper pollutant", fc="#ecebfa", ec="#6250d6", fs=fs)
    box(56, 21.5, 22, 9, "Index-decoding likelihood\n$\\Phi(\\frac{u_I-\\mu}{\\tau})-\\Phi(\\frac{l_I-\\mu}{\\tau})$,\n$\\tau^2=\\sigma^2+s^2$ (deconvolution)", fc="#ecebfa", ec="#6250d6", fs=fs)
    box(81.5, 21.5, 18, 9, "True-class probs\n$\\Phi((\\ln t_k-\\mu)/\\sigma)$,\ncopula for the max", fc="#ecebfa", ec="#6250d6", fs=fs)
    arr(16.8, 24, 19.1, 24); arr(33.8, 26, 36.1, 26); arr(53.3, 26, 55.6, 26); arr(78.8, 26, 81.1, 26)
    ax.text(67, 19.2, "training: exact likelihood of the published integer", ha="center", fontsize=6, color=INK2)
    box(36.5, 3, 16, 10, "MILO-LD (plug-in)\nsoft labels $q(k\\mid I)$:\nposterior of the\ntrue class", fc="#fdf3df", ec="#eda100", fs=fs)
    box(56, 3, 22, 10, "Any softmax classifier\ntrained on $q(k\\mid I)$", fc="#fdf3df", ec="#eda100", fs=fs)
    box(81.5, 3, 18, 10, "Online risk control\n$\\lambda_{t+1}=\\lambda_t-\\eta(\\ell_t-\\alpha)$\n$\\ell_t=(1-d_t)\\,q_t$", fc="#e6f5ee", ec="#1baf7a", fs=fs)
    arr(26.2, 17.6, 36.2, 9.5, ls="--"); arr(53.3, 8, 55.6, 8); arr(90.5, 21.1, 90.5, 13.6); arr(78.8, 8, 81.1, 8)
    ax.text(91.3, 16.6, "$\\hat p_t=\\Pr(Y^{true}_t\\geq b\\mid x)$", fontsize=6, color=INK2)
    save(fig, "fig1_framework.png")


# ------------------------------------------------------------------------- Fig. 2 instance-dependent noise
def fig_noise():
    from ispu_core import inverse_subindex, SCEN, logsd
    from data import load_panel, KCUT, metrology_tables, eb_prior, log_interval
    stns, dates, V = load_panel()
    pm = V[:, dates <= pd.Timestamp("2017-12-31"), 1].ravel()
    pm = pm[np.isfinite(pm)]
    _, _, mid = log_interval(pm, "PM10")
    lt = np.log(inverse_subindex(KCUT + 0.5, "PM10", "2020"))
    c = np.exp(np.linspace(np.log(12), np.log(260), 600))
    lc = np.log(c)
    fig, axs = plt.subplots(2, 1, figsize=(5.6, 4.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    grid = np.linspace(mid.min() - 1.5, mid.max() + 1.5, 900)
    for sc, colr, lab in [("S1", "#2a78d6", "S1: $u_r$ = 12.5 %"), ("S2", "#eb6834", "S2: $u_r$ = 25 %")]:
        s = logsd(SCEN[sc]["PM10"])
        k = np.searchsorted(lt, lc, side="right")
        lo = np.where(k == 0, -np.inf, lt[np.maximum(k - 1, 0)])
        hi = np.where(k == 2, np.inf, lt[np.minimum(k, 1)])
        flip = norm.cdf((lo - lc) / s) + 1 - norm.cdf((hi - lc) / s)
        axs[0].plot(c, flip, color=colr, lw=1.8, label=f"{lab}, exact (instance-dependent)")
        pri = eb_prior(mid, s, grid)
        _, T = metrology_tables("PM10", s, pri, grid)
        edges = [12, np.exp(lt[0]), np.exp(lt[1]), 260]
        for j in range(3):
            axs[0].hlines(1 - T[j, j], edges[j], edges[j + 1], colors=colr, linestyles="--", lw=1.2,
                          label=f"{sc}, class-conditional $1-T_{{kk}}$" if j == 0 else None)
    for v in np.exp(lt):
        for a in axs:
            a.axvline(v, color=INK2, lw=0.8, ls=":")
    axs[0].set_ylabel("P(published class ≠ true class)")
    axs[0].set_xscale("log"); axs[0].set_ylim(0, 0.75)
    axs[0].legend(fontsize=6.4, loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.16))
    axs[0].text(np.exp(lt[0]) * 1.03, 0.56, "Baik | Sedang", fontsize=6.6, color=INK2)
    axs[0].text(np.exp(lt[1]) * 1.03, 0.56, "Sedang | TS", fontsize=6.6, color=INK2)
    dens, edges_h = np.histogram(np.exp(mid), bins=np.exp(np.linspace(np.log(12), np.log(260), 34)), density=True)
    axs[1].fill_between(np.sqrt(edges_h[1:] * edges_h[:-1]), dens, step="mid", color="#9a9893", alpha=0.6, lw=0)
    axs[1].set_ylabel("Density\n(PM10, 2010–17)")
    axs[1].set_xlabel("True 24-h PM10 concentration (µg m$^{-3}$, log scale)")
    from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
    axs[1].xaxis.set_major_locator(FixedLocator([15, 25, 50, 100, 150, 250]))
    axs[1].xaxis.set_major_formatter(FixedFormatter(["15", "25", "50", "100", "150", "250"]))
    axs[1].xaxis.set_minor_formatter(NullFormatter())
    save(fig, "fig2_instance_noise.png")




# ------------------------------------------------------------------------- result figures
def _load(group):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from analyze import load
    return load(group)


def _style_for(m):
    return COL.get(m, COL["other"])


def fig_taskC():
    df = _load("main")
    d = df[df.task == "C"]
    order = ["CE", "LS", "GCE", "FwdT", "CORAL", "SORD", "MSE", "HetG", "LD-flat", "LD", "LT-noC", "LT-noN", "LT"]
    mets = [("test_true.nll", "NLL vs. true class (↓)"), ("test_true.rps", "RPS vs. true class (↓)"), ("test_true.ece", "ECE vs. true class (↓)")]
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.6), sharey=True)
    y = np.arange(len(order))
    for ax, (col, lab) in zip(axs, mets):
        for bb, mk, off in [("MLP", "o", 0.17), ("GRU", "s", -0.17)]:
            g = d[d.backbone == bb].groupby("method")[col].agg(["mean", "std"]).reindex(order)
            for i, m in enumerate(order):
                c = _style_for(m)
                ax.errorbar(g.loc[m, "mean"], i + off, xerr=g.loc[m, "std"], fmt=mk, ms=4.5, color=c,
                            mfc=c if bb == "GRU" else "white", mec=c, elinewidth=0.9, capsize=1.5, lw=0)
        ax.set_xlabel(lab)
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(y); axs[0].set_yticklabels([NAME[m] for m in order])
    axs[0].invert_yaxis()
    from matplotlib.lines import Line2D
    axs[2].legend([Line2D([], [], marker="s", color=INK2, lw=0, ms=4.5), Line2D([], [], marker="o", color=INK2, mfc="white", lw=0, ms=4.5)],
                  ["GRU", "MLP"], loc="lower right", fontsize=7)
    fig.tight_layout()
    save(fig, "fig3_taskC.png")


def fig_sweep():
    df = _load("sweep")
    df["s"] = df.tag.str.replace("_s", "").astype(float)
    ms = ["LS", "FwdT", "HetG", "LD", "LT"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for ax, (col, lab) in zip(axs, [("test_true.nll", "Δ NLL vs. true class relative to CE (↓)"), ("test_true.ece", "ECE vs. true class (↓)")]):
        base = df[df.method == "CE"].set_index(["s", "seed"])[col]
        for m in ms + (["CE"] if "ece" in col else []):
            x = df[df.method == m].set_index(["s", "seed"])[col]
            v = (x - base) if "nll" in col else x
            g = v.groupby(level=0).agg(["mean", "std"])
            ax.errorbar(g.index, g["mean"], yerr=g["std"], color=_style_for(m), marker="o", ms=4, lw=1.6, capsize=2, label=NAME[m])
        if "nll" in col:
            ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xlabel("Injected measurement noise $s$ (log-sd)")
        ax.set_ylabel(lab, fontsize=7.5)
    axs[1].legend(fontsize=6.8, ncol=2)
    fig.tight_layout()
    save(fig, "fig4_noise_sweep.png")


def fig_misspec():
    main, miss = _load("main"), _load("miss")
    s_true = 0.2462
    rows = []
    for _, r in miss.iterrows():
        rows.append(r.to_dict() | {"s_ass": float(r["method"].split("@")[1])})
    m = pd.DataFrame(rows)
    a = main[(main.task == "C") & main.method.isin(["LT", "LT-noN"]) & (main.seed < 3)].copy()
    a["s_ass"] = np.where(a.method == "LT", s_true, 0.0)
    m = pd.concat([m, a])
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.6))
    for ax, (col, lab) in zip(axs, [("test_true.nll", "NLL vs. true class (↓)"), ("test_latent.crps_log", "CRPS of true log-conc. (↓)"),
                                    ("test_latent.cov90", "Coverage of 90 % interval")]):
        for bb, c in [("MLP", "#2a78d6"), ("GRU", "#6250d6")]:
            g = m[m.backbone == bb].groupby("s_ass")[col].agg(["mean", "std"])
            ax.errorbar(g.index, g["mean"], yerr=g["std"], color=c, marker="o", ms=4, lw=1.6, capsize=2, label=bb)
        ax.axvline(s_true, color=INK2, ls=":", lw=1)
        if "nll" in col:
            ax.set_yscale("log")
        if "cov90" in col:
            ax.axhline(0.9, color=INK2, lw=0.8)
        ax.set_xlabel("Assumed noise $s$ (true $s$ = 0.246)")
        ax.set_ylabel(lab, fontsize=7.5)
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "fig5_misspecification.png")


def fig_rc():
    P = json.load(open(os.path.join(OUT, "rc_post_paths.json")))
    rc = pd.read_csv(os.path.join(OUT, "rc_post.csv"))
    CTL = [("online-metroEB", "#6250d6", "MILO-RC, EB posterior"), ("online-metroPost", "#1baf7a", "MILO-RC, forecast-informed posterior"),
           ("online-pub", "#eb6834", "online, published-label loss"), ("static-pub", "#2a78d6", "static split calibration")]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axs[0]
    for key, c, lab in CTL[:3]:
        tl = pd.Series(P[key]["true_loss"], index=pd.to_datetime(P[key]["day"]))
        r = tl.expanding(min_periods=30).mean()
        ax.plot(r.index, r.values, color=c, lw=1.6, label=lab)
    ax.axhline(0.05, color=INK, lw=0.9, ls="--")
    ax.text(pd.Timestamp("2019-07-01"), 0.0108, "target α = 0.05 (dashed)", fontsize=6.3, color=INK)
    ax.set_ylabel("Cumulative true-miss risk")
    ax.set_ylim(0, 0.12)
    from matplotlib.lines import Line2D as _L
    h, l = ax.get_legend_handles_labels()
    h.append(_L([], [], color=CTL[3][1], marker="s", lw=1.2)); l.append(CTL[3][2] + " (panel b)")
    ax.legend(h, l, fontsize=6.0, loc="upper right")
    ax.tick_params(axis="x", labelrotation=30, labelsize=6.5)
    ax.set_title("(a) Task C, MILO-LT (GRU), α = 0.05", fontsize=7.5, color=INK)
    ax = axs[1]
    c_ = rc[rc.task == "C"]
    mk = {0.02: "o", 0.05: "s", 0.10: "^"}
    for key, c, lab in CTL:
        g = c_[c_.controller == key].groupby("alpha")[["alarm", "risk_true"]].mean()
        ax.plot(g["alarm"] * 100, g["risk_true"], color=c, lw=1.0, alpha=0.6)
        for al, row in g.iterrows():
            ax.plot(row["alarm"] * 100, row["risk_true"], marker=mk[al], color=c, ms=5.5, lw=0)
    for al in mk:
        ax.axhline(al, color=INK2, lw=0.7, ls=":")
        ax.text(66.5, al + 0.002, f"α = {al:g}", fontsize=6, color=INK2)
    ax.set_xlabel("Alarm rate (% of station-days)")
    ax.set_ylabel("Realised true-miss risk")
    ax.set_title("(b) risk–alarm trade-off, 10 runs each", fontsize=7.5, color=INK)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], marker=m, color=INK2, lw=0, ms=5) for m in mk.values()], [f"α = {a:g}" for a in mk], fontsize=6.2, loc="upper right")
    fig.tight_layout()
    save(fig, "fig6_risk_control.png")


def fig_reliability():
    import glob
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data import load_panel, build_task
    stns, dates, V = load_panel()
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
    for ax, bb in zip(axs, ["MLP", "GRU"]):
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="--")
        for m in ["CE", "FwdT", "HetG", "LD", "LT"]:
            P_, Y_ = [], []
            for seed in range(5):
                t = build_task("C", stns, dates, V, seed=seed)
                it = np.where(t["split"] == 2)[0]
                z = np.load(os.path.join(os.environ.get("MILO_RES", "results/runs"), "main") + f"/C_{m}_{bb}_s{seed}.npz")
                P_.append(z["test_p_true"][:, 1:].sum(1)); Y_.append((t["y_true"][it] >= 1).astype(float))
            p, yv = np.concatenate(P_), np.concatenate(Y_)
            bins = np.linspace(0, 1, 11)
            k = np.clip(np.digitize(p, bins) - 1, 0, 9)
            xs = [p[k == i].mean() for i in range(10) if (k == i).sum() > 50]
            ys = [yv[k == i].mean() for i in range(10) if (k == i).sum() > 50]
            ax.plot(xs, ys, marker="o", ms=3.5, lw=1.5, color=_style_for(m), label=NAME[m])
        ax.set_title(bb, fontsize=8)
        ax.set_xlabel("Predicted P(true class ≥ Sedang)")
    axs[0].set_ylabel("Observed frequency (truth)")
    axs[1].legend(fontsize=6.6, loc="lower right")
    fig.tight_layout()
    save(fig, "fig7_reliability.png")


if __name__ == "__main__":
    which = sys.argv[1:] or ["framework", "noise"]
    for w in which:
        globals()[f"fig_{w}"]()
