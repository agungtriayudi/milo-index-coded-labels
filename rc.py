"""rc.py -- decision rules and online risk control for exceedance warnings.

Decision on station-day i:  warn (d_i = 1) iff  p_hat_i >= lam,  p_hat_i = P(TRUE class >= b | x_i).
Losses (missed exceedance), revealed one day later:
    published loss   l_i = (1 - d_i) * 1{published class_i >= b}
    metrological loss l_i = (1 - d_i) * q_i,  q_i = P(TRUE class_i >= b | published sub-indices_i)
Online rule (all stations of a day share lam_t; update after the day):
    lam_{t+1} = min(1, lam_t - eta * (mean_i l_{t,i} - alpha))
Deterministic guarantee: (1/T) sum_t mean_i l_{t,i} <= alpha + (lam_1 + eta) / (eta T)."""
import numpy as np


def static_threshold(p, loss_if_no_warn, alpha):
    """Largest lam whose finite-sample-adjusted calibration risk is <= alpha (conformal risk control,
    loss bounded by 1): (n/(n+1)) R_n(lam) + 1/(n+1) <= alpha."""
    n = len(p)
    cand = np.unique(np.concatenate([[0.0, 1.0 + 1e-9], p]))
    best = 0.0
    for lam in cand:                                   # risk is non-decreasing in lam
        r = np.mean((p < lam) * loss_if_no_warn)
        if n / (n + 1) * r + 1 / (n + 1) <= alpha:
            best = lam
        else:
            break
    return best


def online(p, loss_if_no_warn, day, alpha, eta, lam0):
    days, inv = np.unique(day, return_inverse=True)
    lam = lam0
    d = np.zeros(len(p))
    lam_path = np.empty(len(days))
    for k in range(len(days)):
        ix = np.where(inv == k)[0]
        lam_path[k] = lam
        d[ix] = (p[ix] >= lam).astype(float)
        lt = np.mean((1 - d[ix]) * loss_if_no_warn[ix])
        lam = min(1.0, lam - eta * (lt - alpha))
    return d, lam_path


def daily_mean(x, day):
    days, inv = np.unique(day, return_inverse=True)
    return np.bincount(inv, weights=x) / np.bincount(inv)


def evaluate(d, y_pub_ge, q, day, y_true_ge=None):
    out = {"alarm": float(np.mean(d)),
           "risk_pub": float(np.mean(daily_mean((1 - d) * y_pub_ge, day))),
           "risk_metro": float(np.mean(daily_mean((1 - d) * q, day))),
           "fa_pub": float(np.mean(d * (1 - y_pub_ge)))}
    if y_true_ge is not None:
        out["risk_true"] = float(np.mean(daily_mean((1 - d) * y_true_ge, day)))
        out["fa_true"] = float(np.mean(d * (1 - y_true_ge)))
        out["miss_rate_true"] = float(np.sum((1 - d) * y_true_ge) / max(np.sum(y_true_ge), 1))
    return out


def run_all(p_val, p_te, val, te, alpha, eta):
    """val/te: dicts with y_ge (published), q, day, [yt_ge]. Returns dict controller -> metrics."""
    R = {}
    lam_pub = static_threshold(p_val, val["y_ge"], alpha)
    lam_met = static_threshold(p_val, val["q"], alpha)
    yt = te.get("yt_ge")
    R["fixed-0.5"] = evaluate((p_te >= 0.5).astype(float), te["y_ge"], te["q"], te["day"], yt)
    R["static-pub"] = evaluate((p_te >= lam_pub).astype(float), te["y_ge"], te["q"], te["day"], yt) | {"lam": lam_pub}
    R["static-metro"] = evaluate((p_te >= lam_met).astype(float), te["y_ge"], te["q"], te["day"], yt) | {"lam": lam_met}
    d, path = online(p_te, te["y_ge"], te["day"], alpha, eta, lam_pub)
    R["online-pub"] = evaluate(d, te["y_ge"], te["q"], te["day"], yt) | {"lam_end": float(path[-1])}
    d, path = online(p_te, te["q"], te["day"], alpha, eta, lam_met)
    R["online-metro"] = evaluate(d, te["y_ge"], te["q"], te["day"], yt) | {"lam_end": float(path[-1]),
                                                                          "bound": alpha + (lam_met + eta) / (eta * len(np.unique(te["day"])))}
    return R
