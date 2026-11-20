"""rc_analysis.py -- online risk control on top of the saved test/validation predictions."""
import os, json, itertools
import numpy as np
import pandas as pd
from data import load_panel, build_task
from rc import run_all, online, static_threshold, daily_mean, evaluate

ROOT, OUT = os.path.join(os.environ.get("MILO_RES", "results/runs"), "main"), os.environ.get("MILO_OUT", "results/tables")
MODELS = ["CE", "FwdT", "HetG", "LD", "LT", "LT-noN"]
ALPHAS = {"A": [0.01, 0.02, 0.03], "B": [0.02, 0.05, 0.10], "C": [0.02, 0.05, 0.10]}
ETA = 0.05


def side(t, idx):
    b = t["boundary"]
    d = {"y_ge": (t["y"][idx] >= b).astype(float), "q": t["q_true"][idx, b:].sum(1), "day": t["date"][idx]}
    if "y_true" in t:
        d["yt_ge"] = (t["y_true"][idx] >= b).astype(float)
    return d


if __name__ == "__main__":
    stns, dates, V = load_panel()
    rows, paths = [], {}
    for task in ["A", "B", "C"]:
        for seed in range(5):
            t = build_task(task, stns, dates, V, seed=seed)
            iv, it = np.where(t["split"] == 1)[0], np.where(t["split"] == 2)[0]
            sv, st = side(t, iv), side(t, it)
            b = t["boundary"]
            for m, bb in itertools.product(MODELS, ["MLP", "GRU"]):
                f = os.path.join(ROOT, f"{task}_{m}_{bb}_s{seed}.npz")
                if not os.path.exists(f):
                    continue
                z = np.load(f)
                pv, pt = z["val_p_true"][:, b:].sum(1), z["test_p_true"][:, b:].sum(1)
                for a in ALPHAS[task]:
                    R = run_all(pv, pt, sv, st, a, ETA)
                    for ctl, r in R.items():
                        rows.append({"task": task, "model": m, "backbone": bb, "seed": seed, "alpha": a, "controller": ctl, **r})
                if task == "C" and bb == "GRU" and seed == 0 and m in ("LT", "CE"):
                    a = 0.05
                    lam_met = static_threshold(pv, sv["q"], a)
                    lam_pub = static_threshold(pv, sv["y_ge"], a)
                    for ctl, loss, lam0 in [("online-metro", st["q"], lam_met), ("online-pub", st["y_ge"], lam_pub)]:
                        d, path = online(pt, loss, st["day"], a, ETA, lam0)
                        paths[f"{m}|{ctl}"] = {"lam": path.tolist(), "day": sorted(set(st["day"])),
                                               "true_loss": daily_mean((1 - d) * st["yt_ge"], st["day"]).tolist(),
                                               "metro_loss": daily_mean((1 - d) * st["q"], st["day"]).tolist()}
                    d = (pt >= lam_met).astype(float)
                    paths[f"{m}|static-metro"] = {"day": sorted(set(st["day"])), "true_loss": daily_mean((1 - d) * st["yt_ge"], st["day"]).tolist()}
                if task == "C" and m == "LT":                     # step-size sensitivity
                    for eta in [0.01, 0.02, 0.05, 0.1, 0.2]:
                        lam_met = static_threshold(pv, sv["q"], 0.05)
                        d, _ = online(pt, st["q"], st["day"], 0.05, eta, lam_met)
                        rows.append({"task": task, "model": m, "backbone": bb, "seed": seed, "alpha": 0.05,
                                     "controller": f"eta={eta}", **evaluate(d, st["y_ge"], st["q"], st["day"], st["yt_ge"])})
        print("done", task, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "rc_all.csv"), index=False)
    json.dump(paths, open(os.path.join(OUT, "rc_paths.json"), "w"))
    cols = [c for c in ["alarm", "risk_pub", "risk_metro", "risk_true", "fa_true", "miss_rate_true"] if c in df]
    agg = df.groupby(["task", "alpha", "controller", "model", "backbone"])[cols].agg(["mean", "std"]).round(4)
    agg.to_csv(os.path.join(OUT, "rc_summary.csv"))
    with pd.option_context("display.width", 250, "display.max_rows", 400):
        sel = df[(df.backbone == "GRU") & df.model.isin(["CE", "LT", "LD"])]
        print(sel.groupby(["task", "alpha", "controller", "model"])[cols].mean().round(4))
