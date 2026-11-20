"""
data.py -- builds the three forecasting tasks used in the paper from the Jakarta ISPU record.

Task A  next-day overall ISPU class (max over six pollutant sub-indices), P.14/2020 regime only.
Task B  next-day PM10 class over the full 2010-2025 record (PM10 sub-index map identical in both regimes
        below 420 ug/m3, so a single concentration scale applies).
Task C  semi-synthetic version of B with KNOWN truth: published PM10 is treated as the true
        concentration, multiplicative lognormal measurement noise is injected and re-rounded,
        models see only the noisy record and are scored against the truth.

Classes (all tasks): 0 = Baik (<=50), 1 = Sedang (51-100), 2 = Tidak Sehat or worse (>=101).
Every statistic used for scaling / priors is computed on the training period only.
"""
from __future__ import annotations
import glob, os
import numpy as np
import pandas as pd
from scipy.stats import norm, gaussian_kde
from ispu_core import POLS, subindex, inverse_subindex, round_half_up, SCEN, logsd

ALIASES = {"tanggal": "date", "date": "date", "stasiun": "station", "station": "station",
           "pm10": "PM10", "pm25": "PM2.5", "pm2.5": "PM2.5", "so2": "SO2", "co": "CO", "o3": "O3",
           "no2": "NO2", "max": "max", "critical": "critical", "categori": "category", "kategori": "category"}
KCUT = np.array([50, 100])            # 3 ordered classes
K = 3
SPLITS = {"A": ("2021-02-15", "2023-08-31", "2023-12-31", "2025-02-28"),
          "B": ("2010-01-15", "2017-12-31", "2019-06-30", "2025-02-28")}
SPLITS["C"] = SPLITS["B"]
BOUNDARY = {"A": 2, "B": 1, "C": 1}  # risk-control boundary: class >= BOUNDARY  (A: >=TS, B/C: >=Sedang)


def cls3(I):
    I = np.asarray(I, float)
    return np.searchsorted(KCUT, I, side="left")


def load_panel(pattern=os.environ.get("MILO_DATA", "data/ispu_dki[1-5].csv")):
    fr = []
    for f in sorted(glob.glob(pattern)):
        if "_all" in os.path.basename(f):
            continue
        d = pd.read_csv(f, dtype=str)
        d.columns = [ALIASES.get(c.strip().lower(), c.strip()) for c in d.columns]
        fr.append(d)
    df = pd.concat(fr, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["stn"] = df["station"].str.extract(r"(DKI\s*\d)", expand=False).str.replace(" ", "")
    for p in POLS:
        df[p] = pd.to_numeric(df[p], errors="coerce") if p in df else np.nan
        df.loc[df[p] <= 0, p] = np.nan                       # 0 = "TIDAK ADA DATA"
    df = df.drop_duplicates(subset=["date", "stn"])
    stns = sorted(df["stn"].dropna().unique())
    dates = pd.date_range(df["date"].min(), df["date"].max())
    full = df.set_index(["stn", "date"]).reindex(pd.MultiIndex.from_product([stns, dates]))
    V = np.stack([full[p].to_numpy(float).reshape(len(stns), len(dates)) for p in POLS], -1)
    return stns, dates, V          # V: (S, D, 6) integer sub-indices, NaN = missing


# ----------------------------------------------------------------------------- metrology helpers
def log_interval(I, p):
    """Log-concentration interval of a published integer sub-index (rounding half-up)."""
    I = np.asarray(I, float)
    lo = np.where(I >= 1, np.log(np.maximum(inverse_subindex(np.maximum(I - 0.5, 1e-9), p, "2020"), 1e-300)), -np.inf)
    hi = np.log(inverse_subindex(I + 0.5, p, "2020"))
    mid = np.log(inverse_subindex(np.maximum(I, 0.25), p, "2020"))
    return lo, hi, mid


def log_thresholds(p):
    """Concentration thresholds (log) of the 3-class scheme under the rounded convention."""
    return np.log(inverse_subindex(KCUT + 0.5, p, "2020"))


def eb_prior(logc, s, grid):
    """Empirical-Bayes prior density of the TRUE log-concentration on `grid`: Gaussian KDE of the
    training-period values after moment-matched deconvolution shrinkage (removes s^2 from the variance)."""
    x = logc[np.isfinite(logc)]
    m, v = x.mean(), x.var()
    xs = m + (x - m) * np.sqrt(max(1.0 - s * s / v, 0.05))
    if xs.size > 6000:
        xs = np.random.default_rng(0).choice(xs, 6000, replace=False)
    dens = gaussian_kde(xs)(grid)
    return dens / dens.sum()


def metrology_tables(p, s, prior, grid, imax=500):
    """Posterior class probabilities of the TRUE class for every published integer I = 0..imax, and the
    class-level transition matrix T[k, j] = P(published class j | true class k), both under the same
    empirical-Bayes prior and lognormal measurement model (so the two are mutually consistent)."""
    I = np.arange(imax + 1)
    lo, hi, _ = log_interval(I, p)
    lt = log_thresholds(p)
    kgrid = np.searchsorted(lt, grid, side="right")          # true class of each grid point
    if s <= 0:
        return np.eye(K)[cls3(np.maximum(I, 0))], np.eye(K)
    L = norm.cdf((hi[:, None] - grid[None]) / s) - norm.cdf((lo[:, None] - grid[None]) / s)
    post = L * prior[None]
    tot = post.sum(1, keepdims=True)
    post /= np.maximum(tot, 1e-300)
    q = np.stack([post[:, kgrid == k].sum(1) for k in range(K)], 1)
    far = tot[:, 0] < 1e-12                                   # integers outside the prior support
    q[far] = np.eye(K)[cls3(I[far])]
    edges = np.concatenate([[-np.inf], lt, [np.inf]])
    Pj = np.stack([norm.cdf((edges[j + 1] - grid) / s) - norm.cdf((edges[j] - grid) / s) for j in range(K)], 1)
    T = np.stack([(prior[kgrid == k, None] * Pj[kgrid == k]).sum(0) / max(prior[kgrid == k].sum(), 1e-300) for k in range(K)])
    T = (T + 1e-4) / (T + 1e-4).sum(1, keepdims=True)
    return q, T


# ----------------------------------------------------------------------------- task construction
def build_task(task, stns, dates, V, seed=0, scen=None, L=14, min_obs=7, s_override=None, s_inject=None):
    """Return a dict with inputs, targets, metrology tables and splits. `scen` selects the assumed
    measurement-noise scenario (S1/S2); for Task C it is also the TRUE injected noise."""
    scen = scen or ("S2" if task == "C" else "S1")
    rng = np.random.default_rng(1000 + seed)
    pols = POLS if task == "A" else ["PM10"]
    P = len(pols)
    Vp = V[:, :, [POLS.index(p) for p in pols]].copy()
    if task == "A":
        Vp[:, dates < pd.Timestamp("2021-02-01")] = np.nan
    s_true = np.array([logsd(SCEN[scen][p]) for p in pols])
    if s_inject is not None:                          # Task C noise sweep: explicit log-sd
        s_true = np.full(P, float(s_inject)); scen = f"s{s_inject:g}"
    Vtrue = None
    if task == "C":                                   # inject known measurement noise
        Vtrue = Vp.copy()
        ok = np.isfinite(Vp)
        c = inverse_subindex(Vp[ok] + rng.uniform(-0.5, 0.5, ok.sum()), "PM10", "2020")
        Ctrue = np.full(Vp.shape, np.nan); Ctrue[ok] = c
        cobs = c * np.exp(s_true[0] * rng.standard_normal(ok.sum()))
        Vp[ok] = np.maximum(round_half_up(subindex(cobs, "PM10", "2020")), 1.0)
    s_assumed = s_true if s_override is None else np.full(P, float(s_override))

    t0, tr_end, va_end, te_end = [pd.Timestamp(x) for x in SPLITS[task]]
    S, D = Vp.shape[:2]
    obs = np.isfinite(Vp)
    lv = np.log1p(np.where(obs, Vp, 0.0))
    tr_days = dates <= tr_end
    m_in = np.array([lv[:, tr_days, j][obs[:, tr_days, j]].mean() for j in range(P)])
    s_in = np.array([lv[:, tr_days, j][obs[:, tr_days, j]].std() for j in range(P)])
    z = np.where(obs, (lv - m_in) / s_in, 0.0)
    nb = obs.sum(0)                                    # network mean (D, P)
    zbar = np.where(nb > 0, z.sum(0) / np.maximum(nb, 1), 0.0)
    feat = np.concatenate([z, obs.astype(float), np.broadcast_to(zbar, (S, D, P)),
                           np.broadcast_to((nb > 0).astype(float), (S, D, P))], -1)   # (S, D, 4P)

    rows = []
    any_obs = obs.any(-1)
    for si in range(S):
        for di in range(L, D):
            d = dates[di]
            if d < t0 or d > te_end or not any_obs[si, di]:
                continue
            if any_obs[si, di - L:di].sum() < min_obs:
                continue
            if task != "A" and not obs[si, di, 0]:
                continue
            rows.append((si, di))
    rows = np.array(rows)
    si, di = rows[:, 0], rows[:, 1]
    N = len(rows)
    widx = di[:, None] + np.arange(-L, 0)[None]
    Xseq = feat[si[:, None], widx]                                       # (N, L, 4P)
    doy = dates[di].dayofyear.to_numpy()
    dow = dates[di].dayofweek.to_numpy()
    Xstat = np.concatenate([np.eye(S)[si], np.stack([np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)], 1),
                            np.eye(7)[dow]], 1)
    I = Vp[si, di]                                                       # (N, P) published (noisy in C)
    Imax = np.nanmax(I, 1)
    y = cls3(Imax)
    tdates = dates[di]
    split = np.where(tdates <= tr_end, 0, np.where(tdates <= va_end, 1, 2))
    active = obs[si[:, None], widx].any(1)                               # pollutants seen in the window

    # latent targets (log-concentration of the published integers), standardised on training rows
    lo = np.full((N, P), -np.inf); hi = np.full((N, P), np.inf); mid = np.zeros((N, P)); msk = np.isfinite(I)
    for j, p in enumerate(pols):
        ok = msk[:, j]
        lo[ok, j], hi[ok, j], mid[ok, j] = log_interval(I[ok, j], p)
    trm = split == 0
    m_lat = np.array([mid[trm & msk[:, j], j].mean() for j in range(P)])
    sd_lat = np.array([mid[trm & msk[:, j], j].std() for j in range(P)])
    lt = np.stack([log_thresholds(p) for p in pols])                     # (P, 2)

    # metrology: EB priors, posterior tables, transition matrices
    q_tab, T_p, q_flat_tab = [], [], []
    for j, p in enumerate(pols):
        Itr = Vp[:, tr_days, j].ravel()
        Itr = Itr[np.isfinite(Itr)]
        _, _, midtr = log_interval(Itr, p)
        grid = np.linspace(midtr.min() - 1.5, midtr.max() + 1.5, 900)
        pri = eb_prior(midtr, s_assumed[j], grid)
        qt, Tt = metrology_tables(p, s_assumed[j], pri, grid)
        q_tab.append(qt); T_p.append(Tt)
        q_flat_tab.append(metrology_tables(p, s_assumed[j], np.ones_like(grid) / grid.size, grid)[0])
    # soft labels / posteriors of the overall TRUE class (product rule, independent pollutants)
    def overall_posterior(tabs):
        cdf = np.ones((N, K))
        for j in range(P):
            ok = msk[:, j]
            qj = np.zeros((N, K)); qj[:, 0] = 1.0
            qj[ok] = tabs[j][np.clip(I[ok, j].astype(int), 0, 500)]
            cdf *= np.cumsum(qj, 1)
        q = np.clip(np.diff(np.concatenate([np.zeros((N, 1)), cdf], 1), axis=1), 0, 1)
        return q / q.sum(1, keepdims=True)
    q_true, q_flat = overall_posterior(q_tab), overall_posterior(q_flat_tab)
    if P == 1:
        T = T_p[0]
    else:                                              # overall-class T by joint simulation
        T = np.zeros((K, K))
        trI = Vp[:, tr_days].reshape(-1, P)
        trI = trI[np.isfinite(trI).any(1)]
        for _ in range(10):
            Iobs = np.full(trI.shape, np.nan)
            for j, p in enumerate(pols):
                ok = np.isfinite(trI[:, j])
                c = inverse_subindex(trI[ok, j] + rng.uniform(-0.5, 0.5, ok.sum()), p, "2020")
                Iobs[ok, j] = round_half_up(subindex(c * np.exp(s_assumed[j] * rng.standard_normal(ok.sum())), p, "2020"))
            np.add.at(T, (cls3(np.nanmax(trI, 1)), cls3(np.nanmax(Iobs, 1))), 1.0)
        T = (T + 1e-3) / (T + 1e-3).sum(1, keepdims=True)

    out = dict(task=task, pols=pols, P=P, N=N, Xseq=Xseq.astype(np.float64), Xstat=Xstat.astype(np.float64),
               I=I, y=y, split=split, stn=si, date=np.array(tdates.strftime("%Y-%m-%d")), active=active,
               lo=(lo - m_lat) / sd_lat, hi=(hi - m_lat) / sd_lat, mid=(mid - m_lat) / sd_lat, msk=msk,
               m_lat=m_lat, sd_lat=sd_lat, lt=(lt - m_lat[:, None]) / sd_lat[:, None],
               s=s_assumed / sd_lat, s_true=s_true / sd_lat, scen=scen, q_true=q_true, q_flat=q_flat, T=T,
               boundary=BOUNDARY[task], stns=stns)
    if task == "C":
        It = Vtrue[si, di]                             # (N, 1) true integers
        out["y_true"] = cls3(It[:, 0])
        # continuous truth used for latent scoring: the same jittered concentration as in the noise model
        out["I_true"] = It
        out["lc_true"] = (np.log(Ctrue[si, di]) - m_lat) / sd_lat     # (N, 1) standardised true log-conc.
    return out


def describe(t):
    sp = t["split"]
    return {f"n_{n}": int((sp == i).sum()) for i, n in enumerate(["train", "val", "test"])} | \
           {f"classes_{n}": np.bincount(t["y"][sp == i], minlength=K).tolist() for i, n in enumerate(["train", "val", "test"])}
