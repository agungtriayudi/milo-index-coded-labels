"""metrics.py -- probabilistic and ordinal classification metrics, latent-truth scores."""
import numpy as np
from scipy.stats import norm
from sklearn.metrics import f1_score, cohen_kappa_score

K = 3


def class_metrics(p, y, boundary):
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(1, keepdims=True)
    pred = p.argmax(1)
    present = np.unique(y)
    Y = np.eye(K)[y]
    cdf_p, cdf_y = np.cumsum(p, 1)[:, :-1], np.cumsum(Y, 1)[:, :-1]
    conf = p.max(1)
    bins = np.linspace(0, 1, 16)
    b = np.clip(np.digitize(conf, bins) - 1, 0, 14)
    ece = sum(np.abs(np.mean(pred[b == i] == y[b == i]) - conf[b == i].mean()) * np.mean(b == i)
              for i in range(15) if np.any(b == i))
    pge = p[:, boundary:].sum(1)
    yge = (y >= boundary).astype(float)
    return {"acc": float(np.mean(pred == y)),
            "f1": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
            "mae": float(np.mean(np.abs(pred - y))),
            "qwk": float(cohen_kappa_score(y, pred, weights="quadratic", labels=list(range(K)))),
            "nll": float(-np.mean(np.log(p[np.arange(len(y)), y]))),
            "rps": float(np.mean(np.sum((cdf_p - cdf_y) ** 2, 1) / (K - 1))),
            "ece": float(ece),
            "brier_b": float(np.mean((pge - yge) ** 2))}


def latent_scores(mu, sd, x, scale):
    """CRPS and negative log density of the true (standardised) log-concentration x under N(mu, sd^2);
    `scale` converts back to natural-log units."""
    z = (x - mu) / sd
    crps = sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    nll = 0.5 * z ** 2 + np.log(sd) + 0.5 * np.log(2 * np.pi)
    cover90 = np.mean(np.abs(z) <= norm.ppf(0.95))
    return {"crps_log": float(np.mean(crps) * scale), "nll_log": float(np.mean(nll) + np.log(scale)),
            "cov90": float(cover90), "pit_sd": float(np.std(norm.cdf(z)))}
