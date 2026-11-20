"""rc_post.py -- forecast-informed metrological posterior for MILO-RC.

q_post = P(true class >= b | published integers, latent forecast N(mu, sigma^2) of MILO-LT): the forecast
plays the role of the prior and the published integer that of the likelihood (Eq. 1-2). This is the
posterior that assumption (A4) of Corollary 7 asks for. Compared with controllers using the published
label or the empirical-Bayes posterior that ignores the forecast."""
import os, itertools, json
import numpy as np
import pandas as pd
from scipy.stats import norm
from data import load_panel, build_task
from rc import online, static_threshold, evaluate

ROOT, OUT = os.path.join(os.environ.get("MILO_RES", "results/runs"), "main"), os.environ.get("MILO_OUT", "results/tables")
ALPHAS = {"A": [0.01, 0.02, 0.03], "B": [0.02, 0.05, 0.10], "C": [0.02, 0.05, 0.10]}
ETA = 0.05


def q_post_one(mu, sig, lo, hi, s, thr, G=400):
    a = np.minimum(mu - 6 * sig, np.where(np.isfinite(lo), lo, hi) - 4 * s)
    b = np.maximum(mu + 6 * sig, hi + 4 * s)
    x = a[:, None] + (b - a)[:, None] * np.linspace(0, 1, G)[None]
    w = norm.pdf((x - mu[:, None]) / sig[:, None]) * (norm.cdf((hi[:, None] - x) / s) - norm.cdf((lo[:, None] - x) / s))
    return (w * (x >= thr)).sum(1) / np.maximum(w.sum(1), 1e-300)


def q_post(t, idx, mu, sig):
    """Overall exceedance posterior (product over pollutants published that day)."""
    b = t["boundary"]
    below = np.ones(len(idx))
    for j in range(t["P"]):
        m = t["msk"][idx, j]
        if m.any():
            qj = q_post_one(mu[m, j], sig[m, j], t["lo"][idx][m, j], t["hi"][idx][m, j], t["s"][j], t["lt"][j, b - 1])
            below[m] *= 1 - qj
    return 1 - below


if __name__ == "__main__":
    stns, dates, V = load_panel()
    rows, paths, calib = [], {}, []
    for task in ["C", "A", "B"]:
        for seed in range(5):
            t = build_task(task, stns, dates, V, seed=seed)
            iv, it = np.where(t["split"] == 1)[0], np.where(t["split"] == 2)[0]
            b = t["boundary"]
            for bb in ["MLP", "GRU"]:
                z = np.load(os.path.join(ROOT, f"{task}_LT_{bb}_s{seed}.npz"))
                qv = q_post(t, iv, z["val_mu"], z["val_sig"])
                qt = q_post(t, it, z["test_mu"], z["test_sig"])
                pv, pt = z["val_p_true"][:, b:].sum(1), z["test_p_true"][:, b:].sum(1)
                yv, yt = (t["y"][iv] >= b).astype(float), (t["y"][it] >= b).astype(float)
                qeb_v, qeb_t = t["q_true"][iv, b:].sum(1), t["q_true"][it, b:].sum(1)
                ytr = (t["y_true"][it] >= b).astype(float) if "y_true" in t else None
                dv, dt = t["date"][iv], t["date"][it]
                if ytr is not None:                                  # calibration of the two posteriors
                    for nm, q in (("EB", qeb_t), ("post", qt)):
                        bins = np.linspace(0, 1, 11)
                        k = np.clip(np.digitize(q, bins) - 1, 0, 9)
                        calib.append({"bb": bb, "seed": seed, "q": nm, "mean_q": float(q.mean()), "mean_true": float(ytr.mean()),
                                      "brier": float(np.mean((q - ytr) ** 2)),
                                      "nll": float(-np.mean(ytr * np.log(np.clip(q, 1e-9, 1)) + (1 - ytr) * np.log(np.clip(1 - q, 1e-9, 1))))})
                for a in ALPHAS[task]:
                    for ctl, lv, lt_ in (("online-pub", yv, yt), ("online-metroEB", qeb_v, qeb_t), ("online-metroPost", qv, qt)):
                        lam0 = static_threshold(pv, lv, a)
                        d, path = online(pt, lt_, dt, a, ETA, lam0)
                        rows.append({"task": task, "backbone": bb, "seed": seed, "alpha": a, "controller": ctl,
                                     **evaluate(d, yt, qt, dt, ytr), "risk_metroEB": float(np.mean(pd.Series((1 - d) * qeb_t).groupby(dt).mean()))})
                        if task == "C" and bb == "GRU" and seed == 0 and a == 0.05:
                            dm = pd.Series((1 - d) * ytr).groupby(dt).mean()
                            paths[ctl] = {"day": list(dm.index), "true_loss": dm.values.tolist(), "lam": path.tolist()}
                    for ctl, lv in (("static-pub", yv), ("static-metroPost", qv)):
                        lam = static_threshold(pv, lv, a)
                        rows.append({"task": task, "backbone": bb, "seed": seed, "alpha": a, "controller": ctl,
                                     **evaluate((pt >= lam).astype(float), yt, qt, dt, ytr),
                                     "risk_metroEB": float(np.mean(pd.Series((1 - (pt >= lam)) * qeb_t).groupby(dt).mean()))})
        print("done", task, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "rc_post.csv"), index=False)
    json.dump(paths, open(os.path.join(OUT, "rc_post_paths.json"), "w"))
    pd.DataFrame(calib).to_csv(os.path.join(OUT, "rc_post_calib.csv"), index=False)
    pd.set_option("display.width", 250)
    cols = [c for c in ["alarm", "risk_pub", "risk_metro", "risk_metroEB", "risk_true", "fa_true"] if c in df]
    print(df.groupby(["task", "alpha", "controller"])[cols].mean().round(4).to_string())
    print(pd.DataFrame(calib).groupby(["q"])[["mean_q", "mean_true", "brier", "nll"]].mean().round(4))
