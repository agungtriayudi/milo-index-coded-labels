"""
methods.py -- training objectives and predictive class distributions for every compared method.

Each method returns, for any set of rows, two class distributions over the K = 3 ISPU classes:
  p_obs  : distribution of the PUBLISHED class (what the monitoring network will report),
  p_true : distribution of the TRUE class (class of the error-free concentration).
Noise-unaware methods have p_obs = p_true.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm, qmc
from ag import Tensor, log_softmax, softmax, softplus, log, censored_gauss_nll
from nets import ClsHead, CumHead, LatentHead

K = 3
SPEC = {
    "CE":       dict(head="cls", loss="ce"),
    "LS":       dict(head="cls", loss="ls"),
    "GCE":      dict(head="cls", loss="gce"),
    "FwdT":     dict(head="cls", loss="fwd"),
    "CORAL":    dict(head="cum", loss="coral"),
    "SORD":     dict(head="cls", loss="sord"),
    "MSE":      dict(head="lat", loss="mse", noise=False),
    "HetG":     dict(head="lat", loss="gauss", noise=False),
    # ---- proposed (MILO framework) and its ablations
    "LD":       dict(head="cls", loss="msl"),               # MILO-LD: metrological label distributions
    "LD-flat":  dict(head="cls", loss="msl", flat=True),    #   ablation: flat prior instead of EB prior
    "LT":       dict(head="lat", loss="cens", noise=True),  # MILO-LT: latent-truth index-decoding model
    "LT-noN":   dict(head="lat", loss="cens", noise=False), #   ablation: no noise convolution
    "LT-noC":   dict(head="lat", loss="gauss", noise=True), #   ablation: no interval censoring
}
DISPLAY = {"LD": "MILO-LD", "LD-flat": "MILO-LD (flat prior)", "LT": "MILO-LT", "LT-noN": "MILO-LT w/o noise",
           "LT-noC": "MILO-LT w/o censoring"}


class Method:
    def __init__(self, name, task, s_assumed=None):
        base = name.split("@")[0]                       # "MILO@0.125" = MILO with assumed log-sd 0.125
        self.name, self.spec, self.t = name, SPEC[base], task
        self.kind, self.loss_kind = self.spec["head"], self.spec["loss"]
        self.noise = self.spec.get("noise", False)
        s = task["s"] if "@" not in name else np.full(task["P"], float(name.split("@")[1])) / task["sd_lat"]
        self.s2 = s ** 2
        y = task["y"]
        self.Y1 = np.eye(K)[y]
        if self.loss_kind == "ls":
            self.Ysoft = 0.9 * self.Y1 + 0.1 / K
        elif self.loss_kind == "sord":
            d = (np.arange(K)[None] - y[:, None]) ** 2
            e = np.exp(-2.0 * d)
            self.Ysoft = e / e.sum(1, keepdims=True)
        elif self.loss_kind == "msl":
            self.Ysoft = task["q_flat"] if self.spec.get("flat") else task["q_true"]

    def make_head(self, hid, rng):
        if self.kind == "cls":
            return ClsHead(hid, K, rng)
        if self.kind == "cum":
            return CumHead(hid, K, rng)
        return LatentHead(hid, self.t["P"], rng)

    # ------------------------------------------------------------------ training loss
    def loss(self, out, idx):
        t, lk = self.t, self.loss_kind
        if self.kind == "cls":
            if lk == "ce":
                return -(log_softmax(out) * Tensor(self.Y1[idx])).sum(1).mean()
            if lk in ("ls", "sord", "msl"):
                return -(log_softmax(out) * Tensor(self.Ysoft[idx])).sum(1).mean()
            if lk == "gce":
                py = (softmax(out) * Tensor(self.Y1[idx])).sum(1) + 1e-8
                return ((1.0 - py ** 0.7) * (1.0 / 0.7)).mean()
            if lk == "fwd":
                pn = softmax(out) @ Tensor(t["T"])
                return -(log(pn + 1e-8) * Tensor(self.Y1[idx])).sum(1).mean()
        if self.kind == "cum":
            yk = (t["y"][idx][:, None] >= np.arange(1, K)[None]).astype(float)
            return ((softplus(-out) * Tensor(yk) + softplus(out) * Tensor(1.0 - yk)).sum(1) * (1.0 / (K - 1))).mean()
        m, sg = out
        msk = t["msk"][idx].astype(float)
        nobs = max(msk.sum(), 1.0)
        if lk == "mse":
            return ((((m - Tensor(t["mid"][idx])) ** 2) * Tensor(msk)).sum()) * (1.0 / nobs)
        tau = (sg ** 2 + self.s2) ** 0.5 if self.noise else sg
        if lk == "gauss":
            r = (Tensor(t["mid"][idx]) - m) / tau
            return ((log(tau) + (r ** 2) * 0.5) * Tensor(msk)).sum() * (1.0 / nobs)
        if lk == "cens":
            return censored_gauss_nll(m, tau, t["lo"][idx], t["hi"][idx], msk)
        raise ValueError(lk)

    # ------------------------------------------------------------------ predictive distributions
    def raw(self, out):
        if self.kind == "lat":
            return {"mu": out[0].data.copy(), "sig": out[1].data.copy()}
        return {"z": out.data.copy()}

    def _cls_probs(self, raw):
        if self.kind == "cls":
            z = raw["z"] - raw["z"].max(1, keepdims=True)
            p = np.exp(z)
            return p / p.sum(1, keepdims=True)
        ge = 1.0 / (1.0 + np.exp(-raw["z"]))             # P(y >= k), k = 1..K-1, monotone
        cdf = np.concatenate([1.0 - ge, np.ones((len(ge), 1))], 1)
        return np.clip(np.diff(np.concatenate([np.zeros((len(ge), 1)), cdf], 1), axis=1), 1e-9, 1)

    def finalize(self, raw_val, raw_test, iv, it):
        """Return dict(val=..., test=...) of {'p_obs','p_true', and latent params when available}."""
        t = self.t
        if self.kind != "lat":
            res = {}
            for nm, raw in (("val", raw_val), ("test", raw_test)):
                p = self._cls_probs(raw)
                po = p @ t["T"] if self.loss_kind in ("fwd", "msl") else p
                res[nm] = {"p_obs": po, "p_true": p}
            return res
        # latent (index-decoding) methods
        mu_v, mu_t = raw_val["mu"], raw_test["mu"]
        if self.loss_kind == "mse":                       # residual-based spread from validation rows
            mv = t["msk"][iv]
            res_sd = np.array([np.sqrt(np.mean((t["mid"][iv][mv[:, j], j] - mu_v[mv[:, j], j]) ** 2)) for j in range(t["P"])])
            sig_v, sig_t = np.broadcast_to(res_sd, mu_v.shape).copy(), np.broadcast_to(res_sd, mu_t.shape).copy()
        else:
            sig_v, sig_t = raw_val["sig"], raw_test["sig"]
        if self.noise:
            tau_v, tau_t = np.sqrt(sig_v ** 2 + self.s2), np.sqrt(sig_t ** 2 + self.s2)
        else:
            tau_v, tau_t = sig_v, sig_t
        res = {}
        R_obs, R_lat = (None, None)
        if t["P"] > 1:
            R_obs, R_lat = estimate_corr(t["mid"][iv], t["msk"][iv], mu_v, sig_v, tau_v)
        for nm, ix, mu, sg, ta in (("val", iv, mu_v, sig_v, tau_v), ("test", it, mu_t, sig_t, tau_t)):
            d = {"mu": mu, "sig": sg, "tau": ta}
            if t["P"] == 1:
                d["p_obs"] = gauss_class_probs(mu[:, 0], ta[:, 0], t["lt"][0])
                d["p_true"] = gauss_class_probs(mu[:, 0], sg[:, 0], t["lt"][0])
            else:
                act = t["active"][ix]
                d["p_obs"] = copula_class_probs(mu, ta, t["lt"], act, R_obs)
                d["p_true"] = copula_class_probs(mu, sg, t["lt"], act, R_lat)
                d["p_obs_ind"] = indep_class_probs(mu, ta, t["lt"], act)
                d["p_true_ind"] = indep_class_probs(mu, sg, t["lt"], act)
            res[nm] = d
        res["R_obs"], res["R_lat"] = R_obs, R_lat
        return res


# ---------------------------------------------------------------------- aggregation helpers
def gauss_class_probs(mu, sd, lt):
    c0 = norm.cdf((lt[0] - mu) / sd)
    c1 = norm.cdf((lt[1] - mu) / sd)
    return np.clip(np.stack([c0, c1 - c0, 1.0 - c1], 1), 1e-12, 1.0)


def indep_class_probs(mu, sd, lt, act):
    """Overall (max-operator) class under independent pollutant errors (product rule)."""
    cdf0 = np.ones(len(mu)); cdf1 = np.ones(len(mu))
    for j in range(mu.shape[1]):
        a = act[:, j]
        cdf0[a] *= norm.cdf((lt[j, 0] - mu[a, j]) / sd[a, j])
        cdf1[a] *= norm.cdf((lt[j, 1] - mu[a, j]) / sd[a, j])
    return np.clip(np.stack([cdf0, cdf1 - cdf0, 1.0 - cdf1], 1), 1e-12, 1.0)


def _nearest_corr(C):
    C = 0.5 * (C + C.T)
    w, V = np.linalg.eigh(C)
    C = V @ np.diag(np.maximum(w, 1e-4)) @ V.T
    d = np.sqrt(np.diag(C))
    return C / d[:, None] / d[None]


def estimate_corr(mid, msk, mu, sig, tau):
    """Observed and latent (noise-deconvolved) correlation of standardised residuals between
    pollutants, by the method of moments on validation rows:  E[r_p r_q] = rho_pq E[s_p s_q/(t_p t_q)]."""
    P = mu.shape[1]
    r = np.where(msk, (mid - mu) / tau, 0.0)
    Ro, Rl = np.eye(P), np.eye(P)
    for p in range(P):
        for q in range(p + 1, P):
            b = msk[:, p] & msk[:, q]
            if b.sum() < 30:
                continue
            e = np.mean(r[b, p] * r[b, q])
            sp, sq = np.sqrt(np.mean(r[b, p] ** 2)), np.sqrt(np.mean(r[b, q] ** 2))
            Ro[p, q] = Ro[q, p] = np.clip(e / (sp * sq), -0.99, 0.99)
            ratio = np.mean(sig[b, p] * sig[b, q] / (tau[b, p] * tau[b, q]))
            Rl[p, q] = Rl[q, p] = np.clip(e / (sp * sq * ratio), -0.99, 0.99)
    return _nearest_corr(Ro), _nearest_corr(Rl)


_SOBOL = {}


def copula_class_probs(mu, sd, lt, act, R, M=4096, chunk=128):
    """Overall (max-operator) class probabilities under a Gaussian copula of the pollutant errors,
    by randomised quasi-Monte Carlo with common random numbers."""
    P = mu.shape[1]
    if P not in _SOBOL:
        u = qmc.Sobol(d=P, scramble=True, seed=7).random(M)
        _SOBOL[P] = norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
    E = _SOBOL[P] @ np.linalg.cholesky(R).T                       # (M, P) correlated N(0, R)
    out = np.empty((len(mu), K))
    for s in range(0, len(mu), chunk):
        m, d, a = mu[s:s + chunk, None], sd[s:s + chunk, None], act[s:s + chunk, None]
        x = m + d * E[None]                                      # (n, M, P)
        k = (x >= lt[None, None, :, 0]).astype(np.int8) + (x >= lt[None, None, :, 1])
        k = np.where(a, k, 0).max(-1)                            # overall class per draw
        out[s:s + chunk] = np.stack([(k == j).mean(1) for j in range(K)], 1)
    return np.clip(out, 1e-6, 1.0) / np.clip(out, 1e-6, 1.0).sum(1, keepdims=True)
