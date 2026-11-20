"""
ispu_core.py -- Metrological model of Indonesia's Air Pollution Standard Index (ISPU).

Implements the two regulatory transfer functions
  * KepMen LH No. KEP-45/MENLH/10/1997 (+ KEP-107/KABAPEDAL/11/1997 technical guide)
  * Permen LHK No. P.14/MENLHK/SETJEN/KUM.1/7/2020 (in force since 15 July 2020)
and the uncertainty / conformity-assessment quantities used in the paper:
  sub-index, integer rounding, category assignment, effective concentration thresholds,
  inversion of reported indices, specific risks (JCGM 106:2012), guard bands,
  multi-pollutant (max-operator) total risk, metrological confusion matrix, global risks.

All concentrations in ug/m3 (CO included, converted from mg/m3 for 1997).
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm

# ----------------------------------------------------------------------------------
# 1. Regulatory breakpoint tables
# ----------------------------------------------------------------------------------
# P.14/2020, Lampiran I: concentration upper limits for ISPU 50, 100, 200, 300 and the
# '>300' row. The '>300' row is mapped to ISPU 500 (0-500 scale). Averaging period in
# the table header: 24 h for every parameter.
IDX_2020 = np.array([0., 50., 100., 200., 300., 500.])
BP_2020 = {
    "PM10":  np.array([0., 50., 150., 350., 420., 500.]),
    "PM2.5": np.array([0., 15.5, 55.4, 150.4, 250.4, 500.]),
    "SO2":   np.array([0., 52., 180., 400., 800., 1200.]),
    "CO":    np.array([0., 4000., 8000., 15000., 30000., 45000.]),
    "O3":    np.array([0., 120., 235., 400., 800., 1000.]),
    "NO2":   np.array([0., 80., 200., 1130., 2260., 3000.]),
    "HC":    np.array([0., 45., 100., 215., 432., 648.]),
}
AVG_2020 = {p: "24 h" for p in BP_2020}

# KEP-45/1997 (KEP-107/1997 SI table): ISPU 50,100,200,300,400,500.
# PM10 24 h, SO2 24 h, CO 8 h (mg/m3 -> ug/m3), O3 1 h, NO2 1 h (undefined below ISPU 200).
IDX_1997 = np.array([0., 50., 100., 200., 300., 400., 500.])
BP_1997 = {
    "PM10": np.array([0., 50., 150., 350., 420., 500., 600.]),
    "SO2":  np.array([0., 80., 365., 800., 1600., 2100., 2620.]),
    "CO":   np.array([0., 5000., 10000., 17000., 34000., 46000., 57500.]),
    "O3":   np.array([0., 120., 235., 400., 800., 1000., 1200.]),
    "NO2":  np.array([np.nan, np.nan, np.nan, 1130., 2260., 3000., 3750.]),
}
AVG_1997 = {"PM10": "24 h", "SO2": "24 h", "CO": "8 h", "O3": "1 h", "NO2": "1 h"}

CATS = ["Baik", "Sedang", "Tidak Sehat", "Sangat Tidak Sehat", "Berbahaya"]
CATS_EN = ["Good", "Moderate", "Unhealthy", "Very unhealthy", "Hazardous"]
# Integer category boundaries: category k+1 starts at integer B_k + 1.
# 2020: Baik 1-50 | Sedang 51-100 | TS 101-200 | STS 201-300 | B >=301
CUT_2020 = np.array([50, 100, 200, 300])
# 1997: Baik 0-50 | Sedang 51-100 | TS 101-199 | STS 200-299 | B >=300
CUT_1997 = np.array([50, 100, 199, 299])

REGIMES = {
    "2020": dict(idx=IDX_2020, bp=BP_2020, cut=CUT_2020, avg=AVG_2020),
    "1997": dict(idx=IDX_1997, bp=BP_1997, cut=CUT_1997, avg=AVG_1997),
}

POLS = ["PM2.5", "PM10", "SO2", "CO", "O3", "NO2"]   # pollutants reported by Jakarta SPKU

# ----------------------------------------------------------------------------------
# 2. Transfer function, rounding, categories
# ----------------------------------------------------------------------------------
def _tables(p, regime="2020"):
    r = REGIMES[regime]
    return r["bp"][p], r["idx"]


def subindex(c, p, regime="2020"):
    """Unrounded sub-index I = (Ia-Ib)/(Xa-Xb)*(Xx-Xb)+Ib (continuous piecewise-linear).
    Values above the last breakpoint are extrapolated with the last slope."""
    X, I = _tables(p, regime)
    ok = ~np.isnan(X)
    X, I = X[ok], I[ok]
    c = np.asarray(c, float)
    out = np.interp(c, X, I)
    hi = c > X[-1]
    if np.any(hi):
        s = (I[-1] - I[-2]) / (X[-1] - X[-2])
        out = np.where(hi, I[-1] + s * (c - X[-1]), out)
    lo = c < X[0]
    if np.any(lo):
        out = np.where(lo, np.nan, out)
    return out


def inverse_subindex(I, p, regime="2020"):
    """Concentration corresponding to an (unrounded) sub-index value."""
    X, Ix = _tables(p, regime)
    ok = ~np.isnan(X)
    X, Ix = X[ok], Ix[ok]
    I = np.asarray(I, float)
    out = np.interp(I, Ix, X)
    hi = I > Ix[-1]
    if np.any(hi):
        s = (X[-1] - X[-2]) / (Ix[-1] - Ix[-2])
        out = np.where(hi, X[-1] + s * (I - Ix[-1]), out)
    lo = I < Ix[0]
    if np.any(lo):
        out = np.where(lo, np.nan, out)
    return out


def round_half_up(I):
    return np.floor(np.asarray(I, float) + 0.5)


def category_from_int(Iint, regime="2020"):
    """0..4 category from an integer index."""
    cut = REGIMES[regime]["cut"]
    return np.searchsorted(cut, np.asarray(Iint), side="left")


def slope(p, I_level, regime="2020", side="above"):
    """dI/dc (index units per ug/m3) of the segment just above/below index level."""
    X, I = _tables(p, regime)
    ok = ~np.isnan(X)
    X, I = X[ok], I[ok]
    k = np.searchsorted(I, I_level, side="right" if side == "above" else "left") - (0 if side == "above" else 1)
    k = int(np.clip(k, 1, len(I) - 1))
    return (I[k] - I[k - 1]) / (X[k] - X[k - 1])


def thresholds(p, regime="2020", convention="rounded"):
    """Concentration thresholds t_k separating category k from k+1.
    'nominal'  : breakpoint concentration of the boundary index (c > t => higher class)
    'rounded'  : smallest c whose ROUNDED index reaches the next class, f^-1(B + 0.5)."""
    cut = REGIMES[regime]["cut"]
    t = []
    for B in cut:
        if convention == "nominal":
            t.append(float(inverse_subindex(B, p, regime)))
        else:
            t.append(float(inverse_subindex(B + 0.5, p, regime)))
    return np.array(t)


def category_conc(c, p, regime="2020", convention="rounded"):
    t = thresholds(p, regime, convention)
    c = np.asarray(c, float)
    if convention == "nominal":
        return np.searchsorted(t, c, side="left")   # c == t stays in lower class
    return np.searchsorted(t, c, side="right")      # c == t moves up (rounding half up)

# ----------------------------------------------------------------------------------
# 3. Uncertainty scenarios
# ----------------------------------------------------------------------------------
# Relative standard uncertainties u_r = U_r/2 from the EU data-quality objectives
# (Directive 2008/50/EC, Annex I): fixed measurements PM 25 %, SO2/NO2/CO 15 %, O3 15 %;
# indicative measurements PM 50 %, SO2/NO2/CO 25 %, O3 30 %.
SCEN = {
    "S1": {"PM2.5": .125, "PM10": .125, "SO2": .075, "CO": .075, "O3": .075, "NO2": .075, "HC": .075},
    "S2": {"PM2.5": .25,  "PM10": .25,  "SO2": .125, "CO": .125, "O3": .15,  "NO2": .125, "HC": .125},
}
SCEN_LABEL = {"S1": "S1: reference-grade (EU DQO, fixed)", "S2": "S2: indicative / sensor-grade (EU DQO, indicative)"}


def logsd(u_r):
    """Std. deviation of ln(y/c) for a log-normal error with relative std. uncertainty u_r."""
    return np.sqrt(np.log1p(np.asarray(u_r, float) ** 2))

# ----------------------------------------------------------------------------------
# 4. Specific risks (JCGM 106:2012), single pollutant
# ----------------------------------------------------------------------------------
def class_probs_given_y(y, p, u_r, regime="2020"):
    """P(true class = k | measured y), flat prior on ln c, log-normal error.
    Returns array (..., 5)."""
    t = thresholds(p, regime, "rounded")
    s = logsd(u_r)
    y = np.atleast_1d(np.asarray(y, float))
    F = norm.cdf((np.log(t)[None, :] - np.log(y)[:, None]) / s)       # P(c < t_k | y)
    F = np.concatenate([np.zeros((len(y), 1)), F, np.ones((len(y), 1))], axis=1)
    return np.diff(F, axis=1)


def specific_risk(y, p, u_r, regime="2020"):
    """Probability that the true class differs from the class assigned to y."""
    P = class_probs_given_y(y, p, u_r, regime)
    k = category_conc(y, p, regime, "rounded")
    return 1 - P[np.arange(len(P)), k]


def specific_under_risk(y, p, u_r, regime="2020"):
    """Probability that the TRUE class is HIGHER than the class assigned to y
    (public under-warned; 'consumer's risk' of the public)."""
    P = class_probs_given_y(y, p, u_r, regime)
    k = category_conc(y, p, regime, "rounded")
    cum = np.cumsum(P, axis=1)
    return 1 - cum[np.arange(len(P)), k]


def indeterminate_zone(p, u_r, alpha=0.05, regime="2020"):
    """For each boundary t_k: measured-concentration interval [t e^{-z s}, t e^{z s}] in which
    the one-sided probability of a wrong class exceeds alpha; returned also in index units."""
    z = norm.ppf(1 - alpha)
    s = logsd(u_r)
    t = thresholds(p, regime, "rounded")
    lo, hi = t * np.exp(-z * s), t * np.exp(z * s)
    return t, lo, hi, subindex(lo, p, regime), subindex(hi, p, regime)

# ----------------------------------------------------------------------------------
# 5. Multi-pollutant (max operator)
# ----------------------------------------------------------------------------------
def total_under_risk(y_vec, pols, u_vec, regime="2020", rho=None, n_mc=0, rng=None):
    """P(true overall class > announced overall class | measured vector).
    Analytic for independent errors (product rule); MCM when rho (correlation matrix of
    ln-errors) is given."""
    Iann = max(float(round_half_up(subindex(y, p, regime))) for y, p in zip(y_vec, pols))
    kann = int(category_from_int(Iann, regime))
    if rho is None:
        prod = 1.0
        for y, p, u in zip(y_vec, pols, u_vec):
            t = thresholds(p, regime, "rounded")
            if kann >= len(t):
                continue
            prod *= norm.cdf((np.log(t[kann]) - np.log(y)) / logsd(u))
        return 1 - prod
    rng = np.random.default_rng(rng)
    s = np.array([logsd(u) for u in u_vec])
    L = np.linalg.cholesky(np.asarray(rho) * np.outer(s, s))
    z = rng.standard_normal((n_mc, len(pols))) @ L.T
    c = np.array(y_vec)[None, :] * np.exp(-z)            # posterior draws of true c
    ktrue = np.max(np.stack([category_conc(c[:, j], p, regime) for j, p in enumerate(pols)], 1), 1)
    return float(np.mean(ktrue > kann))

# ----------------------------------------------------------------------------------
# 6. Global risks and the metrological confusion matrix
# ----------------------------------------------------------------------------------
def confusion_matrix(prior_pdf, cgrid, p, u_r, regime="2020"):
    """M[j,k] = P(true class j, reported class k) for prior density on cgrid (trapezoid)."""
    t = thresholds(p, regime, "rounded")
    s = logsd(u_r)
    w = prior_pdf(cgrid)
    w = w / np.trapezoid(w, cgrid)
    # P(y < t_k | c): ln y = ln c + e
    F = norm.cdf((np.log(t)[None, :] - np.log(cgrid)[:, None]) / s)
    F = np.concatenate([np.zeros((len(cgrid), 1)), F, np.ones((len(cgrid), 1))], 1)
    Prep = np.diff(F, axis=1)                                    # P(reported k | c)
    jtrue = category_conc(cgrid, p, regime)
    M = np.zeros((5, 5))
    for j in range(5):
        m = jtrue == j
        if m.sum() < 2:
            # integrate with mask weights to keep trapezoid consistent
            pass
        M[j] = np.trapezoid((w * m)[:, None] * Prep, cgrid, axis=0)
    return M


def lognormal_pdf_from_mean(am, gsd):
    sg = np.log(gsd)
    mu = np.log(am) - 0.5 * sg ** 2
    return lambda c: np.exp(-(np.log(c) - mu) ** 2 / (2 * sg ** 2)) / (c * sg * np.sqrt(2 * np.pi))
