"""analyze.py -- aggregates the grid results into tables and statistical tests (CSV + JSON)."""
import os
import glob, json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, studentized_range

ROOT = os.environ.get("MILO_RES", "results/runs")
OUT = os.environ.get("MILO_OUT", "results/tables")
os.makedirs(OUT, exist_ok=True)
NAME = {"CE": "CE", "LS": "LS", "GCE": "GCE", "FwdT": "Forward-T", "CORAL": "CORAL", "SORD": "SORD",
        "MSE": "MSE-reg", "HetG": "Het-Gauss", "LD": "MILO-LD", "LD-flat": "MILO-LD (flat)", "LT": "MILO-LT",
        "LT-noN": "MILO-LT w/o noise", "LT-noC": "MILO-LT w/o cens."}
ORDER = list(NAME)
BASELINES = ["CE", "LS", "GCE", "FwdT", "CORAL", "SORD", "MSE", "HetG"]


def _records(group):
    """Per-run JSON files written by run_grid.py, or the bundled <group>.jsonl archive."""
    fs = glob.glob(os.path.join(ROOT, group, "*.json"))
    if fs:
        return [json.load(open(f)) for f in fs]
    with open(os.path.join(ROOT, group + ".jsonl")) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load(group):
    rows = []
    for r in _records(group):
        base = {k: r[k] for k in ("task", "scen", "method", "backbone", "seed", "tag", "epochs", "best_epoch",
                                  "train_time_s", "n_params")}
        for blk in ("test_obs", "test_true", "test_latent", "test_obs_ind", "val_obs", "val_true"):
            if blk in r:
                for k, v in r[blk].items():
                    base[f"{blk}.{k}"] = v
        rows.append(base)
    return pd.DataFrame(rows)


def fmt(m, s, d=3):
    return f"{m:.{d}f} ± {s:.{d}f}"


def main_table(df, task, block, metrics, digits):
    d = df[df.task == task]
    out = []
    for m in ORDER:
        row = {"method": NAME[m]}
        for bb in ["MLP", "GRU"]:
            x = d[(d.method == m) & (d.backbone == bb)]
            for met in metrics:
                col = f"{block}.{met}"
                if col in x and x[col].notna().any():
                    row[f"{bb} {met}"] = fmt(x[col].mean(), x[col].std(ddof=1), digits.get(met, 3))
                else:
                    row[f"{bb} {met}"] = "–"
        out.append(row)
    return pd.DataFrame(out)


def stats(df, task, block, metric, higher_better=False, methods=None):
    """Wilcoxon signed-rank (paired over backbone x seed) of LT and LD versus every other method,
    Holm-adjusted; average ranks; Friedman test; Nemenyi critical difference."""
    methods = methods or ORDER
    d = df[(df.task == task) & df.method.isin(methods)]
    piv = d.pivot_table(index=["backbone", "seed"], columns="method", values=f"{block}.{metric}")
    piv = piv.dropna(axis=1, how="any")
    sign = -1 if higher_better else 1
    ranks = (sign * piv).rank(axis=1).mean(axis=0).sort_values()
    fr = friedmanchisquare(*[piv[c].values for c in piv.columns])
    k, N = piv.shape[1], piv.shape[0]
    cd = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * N))
    res = {"avg_rank": ranks.round(2).to_dict(), "friedman_p": float(fr.pvalue), "nemenyi_cd": float(cd), "N_blocks": int(N)}
    for ours in ("LT", "LD"):
        if ours not in piv:
            continue
        pv, wins = {}, {}
        for m in piv.columns:
            if m == ours:
                continue
            diff = sign * (piv[ours] - piv[m])                       # < 0: ours better
            try:
                p = wilcoxon(diff, alternative="less").pvalue
            except ValueError:
                p = 1.0
            pv[m] = p
            wins[m] = f"{int((diff < 0).sum())}/{int((diff > 0).sum())}"
        ms = sorted(pv, key=pv.get)                                  # Holm step-down
        adj, run = {}, 0.0
        for i, m in enumerate(ms):
            run = max(run, min(1.0, (len(ms) - i) * pv[m]))
            adj[m] = run
        res[f"{ours}_vs"] = {m: {"p_holm": round(adj[m], 4), "wins/losses": wins[m]} for m in ms}
    return res


if __name__ == "__main__":
    df = load("main")
    print("runs:", len(df), df.groupby(["task", "backbone"]).size().to_dict())
    tabs = {}
    tabs["C_true"] = main_table(df, "C", "test_true", ["acc", "f1", "nll", "rps", "ece"], {"acc": 3, "f1": 3, "nll": 3, "rps": 4, "ece": 3})
    tabs["C_obs"] = main_table(df, "C", "test_obs", ["acc", "nll", "rps"], {"rps": 4})
    tabs["C_latent"] = main_table(df, "C", "test_latent", ["crps_log", "nll_log", "cov90"], {"crps_log": 4})
    tabs["A_obs"] = main_table(df, "A", "test_obs", ["acc", "f1", "nll", "rps", "ece"], {"rps": 4})
    tabs["B_obs"] = main_table(df, "B", "test_obs", ["acc", "f1", "nll", "rps", "ece"], {"rps": 4})
    for k, t in tabs.items():
        t.to_csv(os.path.join(OUT, f"table_{k}.csv"), index=False)
        print(f"\n=== {k}\n" + t.to_string(index=False))
    S = {}
    for task, block in [("C", "test_true"), ("A", "test_obs"), ("B", "test_obs"), ("C", "test_obs")]:
        if (df.task == task).any():
            for met in ["nll", "rps", "ece", "brier_b"]:
                S[f"{task}/{block}/{met}"] = stats(df, task, block, met)
            S[f"{task}/{block}/acc"] = stats(df, task, block, "acc", higher_better=True)
    if (df.task == "C").any():
        lat = [m for m in ["MSE", "HetG", "LT", "LT-noN", "LT-noC"]]
        S["C/test_latent/crps_log"] = stats(df, "C", "test_latent", "crps_log", methods=lat)
    json.dump(S, open(os.path.join(OUT, "stats_main.json"), "w"), indent=1)
    for k in S:
        if k.endswith("nll") or k.endswith("crps_log"):
            print("\n", k, "friedman p=%.2e CD=%.2f" % (S[k]["friedman_p"], S[k]["nemenyi_cd"]), S[k]["avg_rank"])
            for o in ("LT", "LD"):
                if f"{o}_vs" in S[k]:
                    print("  ", o, {m: v["p_holm"] for m, v in S[k][f"{o}_vs"].items()})
    # copula vs independence (Task A, latent methods)
    a = df[(df.task == "A") & df["test_obs_ind.nll"].notna()]
    if len(a):
        cop = a.groupby(["method", "backbone"])[["test_obs.nll", "test_obs_ind.nll", "test_obs.rps", "test_obs_ind.rps"]].mean().round(4)
        cop.to_csv(os.path.join(OUT, "table_A_copula.csv"))
        print("\n", cop)
    comp = df.groupby(["task", "backbone", "method"])[["train_time_s", "epochs", "n_params"]].mean().round(2)
    comp.to_csv(os.path.join(OUT, "table_compute.csv"))
