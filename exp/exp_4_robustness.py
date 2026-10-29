"""
Experiment 4
============

Robustness checks and the appendices, as independent modules.

A. Table A.1: Hill index of the Huber residual under the paper's screens,
   for 16 October and across the twelve sessions.
B. Alternative return lag, s = 10 seconds (Appendix A.2–A.3): the spot
   volatility of Figure 11 rebuilt from 10-second returns (Figures A.1 and
   A.2), and the integrated volatility at 1 and 10 seconds with the
   closed-form damping factor.
C. Time-varying ψ*_t (Appendix A.4, Figure A.3): the Table 3 standard error
   on windows of the most recent 100,000 trades, moved 1,000 trades at a time.
D. First-minute startup rule (Appendix A.5, Figure A.4): the presample
   swapped out by the clock (the paper's rule) or by trade count, on the six
   stocks of Figure B.1.
E. Simulated signatures (Appendix A.7, Table A.2, Figures A.5–A.6): Brownian
   and variance-gamma prices on the 16 October trade times, with and without
   Student-t noise, for raw prices and the median, Huber and linear filters.
   ``c_{t,s}`` is estimated on the noise-free Brownian paths for every case.
F. Cross-asset (Appendix B, Tables B.1–B.2, Figures B.1–B.2): NVDA and the
   54 other assets over the twelve sessions.

G. Sensitivity of the Table B.2 columns built on the linear spot path: the
   linear/Huber ratio against ``a``, the simulated Huber and linear damping
   factors, daily against pooled aggregation, a diurnal curve fit to the
   linear path; the Hill index with an arithmetic average over k on every
   asset; the SPY jump at the NVDA value of ``a``.

All median and Huber filters use the stratified estimator with B = 100 tail
draws (Remark 4(d)). Outputs go to ``results/`` with the prefix ``exp4_``;
the run log is ``results/exp4.log``. ``--modules`` selects modules (default
all); results of modules not run are kept in ``exp4_summary.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loader import load_day, previous_trading_dates
from data.generate_synthetic import brownian_at_trades, student_t_noise, variance_gamma_at_trades
from exp.common import RESULTS, jsonify, log_to_results, two_panel
from src import spot_pipeline as sp
from src.config import (
    AGGREGATION_CHECK_TICKER,
    ALTERNATIVE_HALF_LIFE_TRADES,
    ASSETS,
    CROSS_ASSET_WORKERS,
    DAMPING_CHECK_TICKERS,
    DAMPING_PATHS,
    DAMPING_WINDOW_LAGS,
    FIGURE_A4_DAYS,
    FIGURE_B1_TICKERS,
    HILL_MATCH_TOLERANCE,
    JUMP_CHECK_TICKER,
    JUMP_MATCH_TOLERANCE,
    LONG_LAG_S,
    NOISE_DF,
    NOISE_SCALE_CENTS,
    NVDA_A_PUBLISHED,
    NVDA_SEED,
    NVDA_TICKER,
    OCT16,
    PAPER_DATES,
    PRESAMPLE_OPENINGS,
    PRICE_HALF_LIFE_TRADES,
    PSI_STAR_LAGS,
    PSI_WINDOW_STEP,
    PSI_WINDOW_TRADES,
    RATIO_CHECK_A_GRID,
    RATIO_CHECK_TICKERS,
    REPRICING_HALF_WINDOW_S,
    REPRICING_RANGE_DOLLARS,
    SEED_BM,
    SEED_BM_NOISE,
    SEED_SIMULATED_FILTERS,
    SEED_VG,
    SEED_VG_NOISE,
    SESSION_SECONDS,
    SIMULATION_PATHS,
    SIMULATION_S,
    SIMULATION_SIGMA_ANNUAL,
    SIMULATION_WORKERS,
    SPOT_LAG_S,
    STARTUP_HORIZON_S,
    STARTUP_TICKERS,
    TABLE2_S,
    TABLE_B1_EDGES,
    TAIL_DRAWS,
    THINNING_STEP,
    VG_BETA,
)
from src.empirical_pipeline import opening_abs_residuals, run_pipeline
from src.hf_filters import damping_correction, exponential_huber, simulated_damping
from src.metrics import hill_tail_index, psi_star_cents, s_second_diff_stats, sample_acf, signature_vol

PREFIX = "exp4_"
MODULES = ("A", "B", "C", "D", "E", "F", "G")
NVDA_DATE = OCT16
SEED = NVDA_SEED
E_CACHE = Path(__file__).resolve().parent.parent / "data" / "cache" / "exp4_E_c.npz"


def _out(name: str) -> Path:
    return RESULTS / f"{PREFIX}{name}"


# Table B.2 as published: trades/s, outlier %, Hill, psi* cents, psi* bp,
# linear/Huber, signature 60s/1s, linear jump, ACF lag 1, ACF lag 2.
PAPER_B2 = {
    "NVDA": (69.63, 1.07, 1.43, 0.22, 0.16, 1.223, 0.98, 11.9, -0.08, 0.03),
    "TSLA": (44.4, 2.43, 1.83, 0.61, 0.28, 1.074, 1.02, 1.3, -0.02, 0.05),
    "AAPL": (23.94, 1.3, 2.1, 0.29, 0.13, 1.211, 0.91, 4.6, -0.09, 0.05),
    "SPY": (17.06, 0.92, 1.63, 0.35, 0.06, 1.322, 0.99, 137.6, 0.25, 0.2),
    "AMD": (17.0, 1.6, 1.85, 0.47, 0.3, 1.171, 0.94, 4.1, -0.06, 0.06),
    "AMZN": (15.6, 2.0, 2.03, 0.32, 0.17, 1.254, 0.92, 2.7, -0.07, 0.04),
    "MSFT": (14.64, 2.15, 1.68, 0.7, 0.17, 1.236, 0.89, 16.2, 0.01, 0.07),
    "AVGO": (10.14, 3.01, 1.9, 0.68, 0.38, 1.241, 0.83, 9.7, -0.03, 0.06),
    "MU": (9.19, 1.61, 2.01, 0.42, 0.38, 1.262, 0.89, 13.0, -0.01, 0.06),
    "INTC": (8.94, 0.96, 2.19, 0.12, 0.52, 1.143, 0.71, 2.6, 0.08, 0.03),
    "META": (8.74, 4.33, 2.51, 1.65, 0.29, 1.113, 0.72, 3.1, 0.01, 0.02),
    "GOOG": (8.25, 0.93, 1.65, 0.36, 0.21, 1.171, 0.94, 7.4, 0.02, 0.1),
    "COIN": (5.81, 5.03, 2.67, 1.74, 0.83, 1.089, 0.74, 1.1, 0.1, 0.04),
    "MRVL": (5.39, 2.1, 2.4, 0.43, 0.52, 1.268, 0.78, 12.3, 0.08, 0.08),
    "RIVN": (4.62, 0.69, 4.7, 0.1, 0.94, 1.118, 0.67, 1.9, 0.11, 0.04),
    "HOOD": (4.47, 1.3, 2.51, 0.19, 0.72, 1.183, 0.72, 1.8, 0.09, 0.07),
    "CELH": (4.43, 2.17, 2.28, 0.45, 1.33, 1.235, 0.71, 11.4, 0.07, 0.07),
    "QCOM": (4.33, 3.89, 2.0, 0.98, 0.57, 1.148, 0.73, 2.1, 0.09, 0.07),
    "PYPL": (4.3, 3.02, 2.38, 0.27, 0.33, 1.255, 0.88, 1.5, 0.04, 0.04),
    "CSCO": (3.96, 1.37, 1.91, 0.17, 0.3, 1.277, 0.67, 17.2, 0.13, 0.05),
    "PEP": (3.96, 2.11, 2.73, 0.48, 0.27, 1.318, 0.68, 13.4, 0.04, 0.07),
    "AMAT": (3.93, 3.77, 1.8, 0.91, 0.49, 1.191, 0.72, 25.5, 0.13, 0.09),
    "SBUX": (3.71, 1.61, 2.19, 0.34, 0.36, 1.172, 0.75, 6.7, 0.13, 0.07),
    "NFLX": (3.71, 5.15, 3.4, 3.31, 0.47, 1.072, 0.54, 1.6, 0.1, 0.01),
    "CRWD": (3.41, 5.36, 2.91, 2.17, 0.71, 1.102, 0.62, 1.2, 0.08, 0.03),
    "ADBE": (3.1, 4.89, 3.05, 2.86, 0.57, 1.115, 0.51, 1.4, 0.06, 0.02),
    "COST": (3.02, 5.6, 3.7, 3.82, 0.43, 1.016, 0.41, 1.4, 0.04, -0.01),
    "DKNG": (2.75, 2.01, 3.13, 0.25, 0.67, 1.189, 0.7, 1.1, 0.09, 0.09),
    "TXN": (2.69, 4.0, 2.51, 1.09, 0.54, 1.137, 0.7, 1.5, 0.16, 0.12),
    "PANW": (2.38, 4.95, 3.14, 2.69, 0.72, 1.133, 0.59, 1.6, 0.07, 0.02),
    "ABNB": (2.33, 3.48, 2.03, 0.84, 0.62, 1.149, 0.69, 8.4, 0.1, 0.07),
    "GILD": (2.25, 1.67, 3.07, 0.32, 0.36, 1.201, 0.75, 2.5, 0.15, 0.12),
    "DDOG": (2.11, 2.86, 3.18, 0.82, 0.65, 1.171, 0.7, 1.4, 0.15, 0.09),
    "ETSY": (1.88, 2.65, 2.98, 0.46, 0.87, 1.099, 0.64, 3.2, 0.12, 0.07),
    "LCID": (1.87, 0.03, 5.6, 0.08, 2.37, 1.004, 0.4, 1.1, 0.11, 0.0),
    "TTD": (1.61, 3.32, 2.78, 0.82, 0.69, 1.123, 0.69, 3.2, 0.15, 0.09),
    "ROKU": (1.51, 3.42, 2.63, 0.81, 1.05, 1.113, 0.68, 1.0, 0.13, 0.07),
    "TEAM": (1.49, 4.86, 2.91, 1.95, 1.03, 1.154, 0.68, 2.2, 0.11, 0.06),
    "INTU": (1.43, 5.13, 2.49, 4.12, 0.68, 1.115, 0.58, 9.3, 0.13, 0.04),
    "AMGN": (1.38, 4.16, 3.04, 1.82, 0.57, 1.023, 0.52, 3.9, 0.13, 0.05),
    "ISRG": (1.28, 4.79, 2.3, 3.19, 0.67, 1.049, 0.48, 2.5, 0.1, 0.02),
    "VRTX": (1.08, 5.11, 3.61, 3.95, 0.81, 1.044, 0.53, 1.8, 0.14, 0.05),
    "REGN": (0.91, 5.78, 2.25, 10.69, 1.06, 1.036, 0.53, 1.7, 0.1, 0.03),
    "CROX": (0.87, 4.91, 4.18, 1.35, 0.98, 1.052, 0.58, 1.4, 0.12, 0.06),
    "MELI": (0.83, 6.9, 3.71, 32.27, 1.58, 1.004, 0.38, 2.2, 0.01, -0.01),
    "CALM": (0.71, 4.69, 3.32, 1.13, 1.2, 0.982, 0.62, 0.9, 0.12, 0.09),
    "ORLY": (0.6, 7.11, 2.7, 12.75, 1.06, 1.071, 0.52, 1.0, 0.1, 0.06),
    "WING": (0.57, 6.32, 3.09, 6.2, 1.64, 1.011, 0.54, 1.2, 0.12, 0.05),
    "BKNG": (0.56, 8.67, 2.72, 70.21, 1.62, 1.081, 0.45, 1.3, 0.03, 0.0),
    "DUOL": (0.54, 6.53, 4.07, 5.08, 1.77, 1.02, 0.63, 1.2, 0.15, 0.08),
    "MIDD": (0.46, 5.01, 2.96, 1.52, 1.1, 0.967, 0.73, 1.8, 0.2, 0.11),
    "SAIA": (0.39, 5.96, 5.04, 11.51, 2.5, 1.014, 0.66, 1.0, 0.12, 0.05),
    "FIZZ": (0.22, 5.9, 2.53, 0.68, 1.47, 0.93, 0.75, 1.0, 0.2, 0.11),
    "LANC": (0.19, 5.35, 2.52, 3.29, 1.83, 0.972, 0.76, 1.0, 0.11, 0.01),
    "IPAR": (0.19, 5.42, 4.07, 2.29, 1.94, 0.974, 0.8, 1.6, 0.22, 0.1),
}

# Table B.1 as published: stocks, then lag-1 Huber 16 Oct / Huber 12 days /
# trades 16 Oct, then lag-2 in the same order.
PAPER_B1 = {
    "above 10": (7, -0.03, -0.06, -0.46, 0.05, 0.05, -0.01),
    "3 to 10": (19, 0.09, 0.08, -0.46, 0.03, 0.05, -0.02),
    "1 to 3": (15, 0.13, 0.13, -0.43, 0.06, 0.07, -0.03),
    "below 1": (13, 0.09, 0.12, -0.43, 0.01, 0.06, -0.03),
}


def _day(ticker: str, date: str, presample: np.ndarray | None = None, **kw):
    loaded = load_day(ticker, date)
    df = loaded.df
    res = run_pipeline(
        df["tau_s"].to_numpy(),
        df["log_price"].to_numpy(),
        tau_open=df["tau_open"].to_numpy(),
        presample_abs=presample,
        T60=loaded.T60,
        B=TAIL_DRAWS,
        seed=SEED + PAPER_DATES.index(date) if date in PAPER_DATES else SEED,
        **kw,
    )
    return loaded, res


def _presample(date: str, cache: dict, ticker: str = NVDA_TICKER) -> np.ndarray:
    parts = []
    for prev in previous_trading_dates(date, PRESAMPLE_OPENINGS):
        if (ticker, prev) not in cache:
            loaded = load_day(ticker, prev)
            cache[(ticker, prev)] = opening_abs_residuals(
                loaded.df["log_price"].to_numpy(),
                loaded.df["tau_open"].to_numpy(),
            )
        parts.append(cache[(ticker, prev)])
    return np.concatenate(parts) if parts else np.zeros(0)


def _window_range(tau: np.ndarray, px: np.ndarray, half: float) -> np.ndarray:
    """Range of ``px`` over the trades with ``|tau_i - tau_t| <= half``."""
    win = pd.Timedelta(seconds=half)
    back = pd.Series(px, index=pd.to_datetime(tau, unit="s")).rolling(win, closed="both")
    fwd = pd.Series(px[::-1], index=pd.to_datetime(-tau[::-1], unit="s")).rolling(win, closed="both")
    hi = np.maximum(back.max().to_numpy(), fwd.max().to_numpy()[::-1])
    lo = np.minimum(back.min().to_numpy(), fwd.min().to_numpy()[::-1])
    return hi - lo


# ---------------------------------------------------------------------------
# A. Table A.1
# ---------------------------------------------------------------------------

def module_a():
    """Table A.1 screens. Thresholds are held fixed when the half-life changes."""
    cache = {}
    per_day = []
    for date in PAPER_DATES:
        pre = _presample(date, cache)
        loaded, res = _day(NVDA_TICKER, date, pre)
        tau = loaded.df["tau_open"].to_numpy()
        base = hill_tail_index(res.resid_hub)
        standardized = hill_tail_index(res.resid_hub / np.maximum(res.delta, 1e-12))
        drop = _window_range(tau, np.exp(res.theta), REPRICING_HALF_WINDOW_S) > REPRICING_RANGE_DOLLARS
        no_reprice = hill_tail_index(res.resid_hub[~drop])
        thin = hill_tail_index(res.resid_hub[::THINNING_STEP])
        theta6 = exponential_huber(loaded.df["log_price"].to_numpy(), res.delta, res.m,
                                   halflife=ALTERNATIVE_HALF_LIFE_TRADES, B=TAIL_DRAWS, seed=SEED,
                                   delta0=float(res.delta_prev[0]))
        hill6 = hill_tail_index(loaded.df["log_price"].to_numpy() - theta6)
        row = {
            "date": date,
            "T": int(len(loaded.df)),
            "all_trades": base,
            "residual_over_delta": standardized,
            "repricing_removed": no_reprice,
            "every_10th": thin,
            "half_life_6": hill6,
            "repricing_share": float(drop.mean()),
        }
        per_day.append(row)
        print(f"A {date} T={row['T']} hill={base:.3f}", flush=True)

    def _rng(key):
        vals = [r[key] for r in per_day]
        return {"min": float(np.min(vals)), "max": float(np.max(vals))}
    oct16 = next(r for r in per_day if r["date"] == NVDA_DATE)
    keys = ("all_trades", "residual_over_delta", "repricing_removed", "every_10th", "half_life_6")
    return {"oct16": oct16, "twelve_day_range": {k: _rng(k) for k in keys}, "per_day": per_day}


# ---------------------------------------------------------------------------
# B. s = 10 seconds
# ---------------------------------------------------------------------------

def module_b(a_nvda: float | None = None):
    loaded, res = _day(NVDA_TICKER, NVDA_DATE, _presample(NVDA_DATE, {}))
    tau = loaded.df["tau_open"].to_numpy()
    lags = (SPOT_LAG_S, LONG_LAG_S)
    c_by = {s: damping_correction(tau, s, PRICE_HALF_LIFE_TRADES, DAMPING_WINDOW_LAGS) for s in lags}
    integrated = {}
    for name, path in (("huber", res.theta), ("linear", res.linear), ("trades", loaded.df["log_price"].to_numpy())):
        tbl = s_second_diff_stats(tau, path, s_list=lags, c_by_s=None if name == "trades" else c_by)
        integrated[name] = {str(s): {"annualized_vol_pct": r["annualized_vol"], "hill": r["hill"]} for s, r in tbl.items()}
    sp.ensure_filtered(NVDA_TICKER)
    if a_nvda is None:
        a_nvda = sp.spot_ticker(NVDA_TICKER)["a_qlike"]
    app = sp.appendix_a2_a3(NVDA_TICKER, a_nvda)
    i = PAPER_DATES.index(NVDA_DATE)
    sec = app["seconds"]
    two_panel([sec["linear_10"][i], sec["huber_10"][i]], [sec["linear_10"], sec["huber_10"]],
              ["linear filter, s = 10 s", "Huber filter, s = 10 s"], ["0.55", "tab:red"], _out("B_figure_a1_s10.png"))
    two_panel([sec["huber_1"][i], sec["huber_10"][i]], [sec["huber_1"], sec["huber_10"]],
              ["Huber filter, s = 1 s", "Huber filter, s = 10 s"], ["tab:red", "tab:blue"],
              _out("B_figure_a2_huber_s1_s10.png"))
    print("B", json.dumps(jsonify(app, drop={"seconds"}))[:400], flush=True)
    return {"integrated_closed_form_c": integrated, "spot": {k: v for k, v in app.items() if k != "seconds"}}


# ---------------------------------------------------------------------------
# C. Rolling psi*
# ---------------------------------------------------------------------------

def rolling_psi_star(psi: np.ndarray, inside: np.ndarray, window: int, step: int,
                     halflife: float = PRICE_HALF_LIFE_TRADES, head: int = PSI_STAR_LAGS):
    """``ψ* = sqrt(w'Γw) / share inside`` on windows ``[s, s + window)``.

    ``Γ`` is the autocovariance of the score within each window (lags 0 to
    ``head - 1``, demeaned by the window mean, divided by the window length),
    as in ``psi_star_cents``. Prefix sums make every window O(head).
    """
    psi = np.asarray(psi, dtype=np.float64)
    n = len(psi)
    starts = np.arange(0, n - window + 1, step)
    ends = starts + window
    P = np.concatenate([[0.0], np.cumsum(psi)])
    I = np.concatenate([[0.0], np.cumsum(inside.astype(np.float64))])
    mean = (P[ends] - P[starts]) / window
    gam = np.empty((len(starts), head))
    for k in range(head):
        prod = np.concatenate([[0.0], np.cumsum(psi[: n - k] * psi[k:])])
        s_pp = prod[ends - k] - prod[starts]
        s1 = P[ends - k] - P[starts]
        s2 = P[ends] - P[starts + k]
        gam[:, k] = (s_pp - mean * (s1 + s2) + (window - k) * mean * mean) / window
    lam = 2.0 ** (-1.0 / halflife)
    w = (1.0 - lam) * lam ** np.arange(head)
    idx = np.abs(np.subtract.outer(np.arange(head), np.arange(head)))
    quad = np.einsum("i,nij,j->n", w, gam[:, idx], w)
    slope = (I[ends] - I[starts]) / window
    return starts, ends, np.sqrt(np.maximum(quad, 0.0)) / slope


def module_c(window: int = PSI_WINDOW_TRADES, step: int = PSI_WINDOW_STEP):
    loaded, res = _day(NVDA_TICKER, NVDA_DATE, _presample(NVDA_DATE, {}))
    price = loaded.df["price"].to_numpy()
    tau = loaded.df["tau_open"].to_numpy()
    e = (price - np.exp(res.theta)) * 100.0
    d = res.delta_prev * price * 100.0
    psi = np.clip(e, -d, d)
    full = psi_star_cents(price, res.theta, res.delta_prev)["psi_star_cents"]
    starts, ends, ps = rolling_psi_star(psi, np.abs(e) <= d, window, step)
    t_end = tau[ends - 1]
    first_hour = t_end <= 3600.0
    second_half = t_end >= SESSION_SECONDS / 2.0
    delta_c = res.delta * price * 100.0
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(9.5 + t_end / 3600.0, ps, color="tab:red", lw=0.9, label=f"most recent {window:,} trades")
    ax.axhline(full, color="k", ls="--", lw=0.8, label=f"full day, {full:.2f} cents")
    ax.set_xlabel("time of day (hours)")
    ax.set_ylabel("ψ*_t (cents)")
    ax.set_title("NVDA 16 Oct 2024: Huber filter standard error on a rolling window (Figure A.3)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(_out("C_figure_a3_psi_rolling.png"), dpi=140)
    plt.close(fig)
    out = {
        "window": window, "step": step, "n_windows": int(len(ps)), "full_day_cents": float(full),
        "first_window_end_s": float(t_end[0]),
        "max_cents": float(ps.max()), "max_at_s": float(t_end[int(np.argmax(ps))]),
        "mean_first_hour": float(ps[first_hour].mean()) if first_hour.any() else float("nan"),
        "mean_second_half": float(ps[second_half].mean()),
        "delta_median_first_hour": float(np.median(delta_c[tau <= 3600.0])),
        "delta_median_second_half": float(np.median(delta_c[tau >= SESSION_SECONDS / 2.0])),
        "series": {"t_end_s": t_end[::10].tolist(), "psi_star_cents": ps[::10].tolist()},
    }
    out["ratio_second_half_to_first_hour"] = out["mean_second_half"] / out["mean_first_hour"]
    out["delta_ratio_second_half_to_first_hour"] = out["delta_median_second_half"] / out["delta_median_first_hour"]
    print("C", {k: v for k, v in out.items() if k != "series"}, flush=True)
    return out


# ---------------------------------------------------------------------------
# D. Startup rule of the first minute
# ---------------------------------------------------------------------------

def module_d(horizon: float = STARTUP_HORIZON_S):
    """Clock-time against trade-count swapping of the presample (Appendix A.5).

    Both rules share the seeds, so the median filter is identical and every
    difference comes from the scale. Only the first ``horizon`` seconds are
    filtered: nothing up to that point depends on later trades.
    """
    rows, panels = [], {}
    for ticker, paper_day in zip(STARTUP_TICKERS, FIGURE_A4_DAYS):
        cache = {}
        per_day = []
        for date in PAPER_DATES:
            loaded = load_day(ticker, date)
            df = loaded.df
            tau = df["tau_open"].to_numpy()
            n = min(int(np.sum(tau <= horizon)) + 1, len(tau))
            pre = _presample(date, cache, ticker)
            out = {}
            for mode in ("clock", "count"):
                res = run_pipeline(df["tau_s"].to_numpy()[:n], df["log_price"].to_numpy()[:n], tau_open=tau[:n],
                                   presample_abs=pre, T60=loaded.T60, B=TAIL_DRAWS,
                                   seed=SEED + PAPER_DATES.index(date), startup=mode)
                out[mode] = res
            px = df["price"].to_numpy()[: n - 1]
            dc = out["clock"].delta[: n - 1] * px * 100.0
            dk = out["count"].delta[: n - 1] * px * 100.0
            t60 = loaded.T60
            o_c = int((out["clock"].d_pos[:t60] + out["clock"].d_neg[:t60]).sum())
            o_k = int((out["count"].d_pos[:t60] + out["count"].d_neg[:t60]).sum())
            th = np.abs(np.exp(out["clock"].theta[: n - 1]) - np.exp(out["count"].theta[: n - 1])) * 100.0
            per_day.append({
                "date": date, "T60": int(t60),
                "delta0_cents": float(out["clock"].delta_prev[0] * px[0] * 100.0),
                "max_abs_delta_gap_cents": float(np.max(np.abs(dc - dk))),
                "count_delta_range_first_minute_cents": float(np.ptp(dk[:t60])) if t60 > 0 else 0.0,
                "outliers_first_minute_clock": o_c, "outliers_first_minute_count": o_k,
                "max_abs_huber_gap_cents": float(th.max()),
                "differs_after_first_minute": bool(np.any(np.abs(dc[t60:] - dk[t60:]) > 1e-12)),
            })
            panels[(ticker, date)] = (tau[: n - 1], dc, dk, per_day[-1]["delta0_cents"])
            print(f"D {ticker} {date} T60={t60} max gap {per_day[-1]['max_abs_delta_gap_cents']:.2f}c", flush=True)
        worst = max(per_day, key=lambda r: r["max_abs_delta_gap_cents"])
        rate = float(np.mean([load_day(ticker, d).cleaned_count for d in PAPER_DATES]) / SESSION_SECONDS)
        gaps = sorted((r["max_abs_delta_gap_cents"] for r in per_day), reverse=True)
        paper_gap = next(r["max_abs_delta_gap_cents"] for r in per_day if r["date"] == paper_day)
        rows.append({"ticker": ticker, "trades_per_second": rate, "day_most_moved": worst["date"],
                     "max_gap_cents": worst["max_abs_delta_gap_cents"], "paper_day": paper_day,
                     "paper_day_gap_cents": paper_gap, "paper_day_rank": gaps.index(paper_gap) + 1,
                     "per_day": per_day})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, row in zip(axes.ravel(), rows):
        t, dc, dk, d0 = panels[(row["ticker"], row["day_most_moved"])]
        ax.plot(t, dk, color="tab:red", lw=0.8, label="weight by trade count")
        ax.plot(t, dc, color="tab:blue", lw=1.0, label="weight by clock time")
        ax.axhline(d0, color="k", ls="--", lw=0.7, label="previous five openings, δ0")
        ax.set_title(f"{row['ticker']}, {row['day_most_moved']}, {row['trades_per_second']:.2g} trades/s", fontsize=9)
        ax.set_xlabel("seconds after the open")
        ax.set_ylabel("δ_t (cents)")
        ax.grid(True, linestyle=":", alpha=0.6)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(_out("D_figure_a4_startup.png"), dpi=140)
    plt.close(fig)
    return rows


# ---------------------------------------------------------------------------
# E. Simulated signatures
# ---------------------------------------------------------------------------

S_E = SIMULATION_S
FILTERS_E = ("raw", "median", "huber", "linear")
_E = {}


def _e_init():
    loaded = load_day(NVDA_TICKER, NVDA_DATE)
    tau = loaded.df["tau_open"].to_numpy()
    dt = np.maximum(np.diff(tau, prepend=tau[0]), 0.0)
    dt[0] = 0.0
    _E.update(tau=tau, dt=dt, T60=loaded.T60, open_px=float(loaded.df["price"].iloc[0]),
              sigma=SIMULATION_SIGMA_ANNUAL / np.sqrt(252.0 * SESSION_SECONDS))
    _E["omega"] = (NOISE_SCALE_CENTS / 100.0) / _E["open_px"]
    if E_CACHE.exists():
        with np.load(E_CACHE) as z:
            _E["c"] = {f: {s: z[f"{f}_{s}"].astype(np.float64) for s in S_E} for f in FILTERS_E if f != "raw"}


def _latent(kind: str, m: int):
    """Brownian or variance-gamma log-price at the trade times, and its increments."""
    if kind == "bm":
        return brownian_at_trades(_E["dt"], _E["sigma"], seed=SEED_BM + m)
    return variance_gamma_at_trades(_E["dt"], _E["sigma"], VG_BETA, seed=SEED_VG + m)


def _noise(kind: str, m: int) -> np.ndarray:
    seed = (SEED_BM_NOISE if kind == "bm" else SEED_VG_NOISE) + m
    return student_t_noise(len(_E["tau"]), _E["omega"], NOISE_DF, seed=seed)


def _paths(y: np.ndarray, m: int):
    res = run_pipeline(_E["tau"], y, tau_open=_E["tau"], T60=_E["T60"], halflife_trades=PRICE_HALF_LIFE_TRADES,
                       B=TAIL_DRAWS, seed=SEED_SIMULATED_FILTERS + m)
    return {"raw": y, "median": res.m, "huber": res.theta, "linear": res.linear}, res


def _stats(paths, res, truth: float, with_c: bool) -> dict:
    tau = _E["tau"]
    out = {"truth": truth, "delta_median_cents": float(np.median(res.delta) * _E["open_px"] * 100.0),
           "acf1": {f: float(sample_acf(np.diff(paths[f]), lags=(1,))[0]) for f in FILTERS_E}}
    out["plain"] = {f: [signature_vol(tau, paths[f], s) for s in S_E] for f in FILTERS_E}
    if with_c:
        out["corrected"] = {f: [signature_vol(tau, paths[f], s, None if f == "raw" else _E["c"][f][s]) for s in S_E]
                            for f in FILTERS_E}
    return out


def _e_base(m: int):
    """Pass 1: a noise-free Brownian path, its filtered paths (float32) and plain stats."""
    x, inc = _latent("bm", m)
    paths, res = _paths(x, m)
    truth = float(100.0 * np.sqrt(252.0 * np.sum(inc * inc)))
    st = _stats(paths, res, truth, with_c=False)
    return m, {f: np.asarray(v, dtype=np.float32) for f, v in paths.items()}, st


def _e_case(task):
    """Pass 2: one path of a case other than the noise-free Brownian one."""
    case, m = task
    kind = "bm" if case.startswith("bm") else "vg"
    x, inc = _latent(kind, m)
    y = x + _noise(kind, m) if case.endswith("noise") else x
    paths, res = _paths(y, m)
    truth = float(100.0 * np.sqrt(252.0 * np.sum(inc * inc)))
    st = _stats(paths, res, truth, with_c=True)
    if case == "vg":
        q = np.sort(inc * inc)[::-1]
        cum = np.cumsum(q) / q.sum()
        st["vg"] = {"n_increments_above_0.01_cents": int(np.sum(np.abs(inc) * _E["open_px"] * 100.0 > 0.01)),
                    "n_for_99pct_qv": int(np.searchsorted(cum, 0.99) + 1),
                    "n_for_50pct_qv": int(np.searchsorted(cum, 0.50) + 1)}
    return case, m, st


def _summarize_case(stats: list[dict]) -> dict:
    def avg(key, f):
        a = np.array([s[key][f] for s in stats], dtype=np.float64)
        return a.mean(axis=0).tolist()
    truth = np.array([s["truth"] for s in stats])
    out = {"n_paths": len(stats), "truth_mean": float(truth.mean()),
           "delta_median_cents": float(np.median([s["delta_median_cents"] for s in stats])),
           "acf1_mean": {f: float(np.mean([s["acf1"][f] for s in stats])) for f in FILTERS_E},
           "plain_mean": {f: avg("plain", f) for f in FILTERS_E},
           "plain_minus_truth_mean": {f: (np.array(avg("plain", f)) - truth.mean()).tolist() for f in FILTERS_E}}
    if "corrected" in stats[0]:
        out["corrected_mean"] = {f: avg("corrected", f) for f in FILTERS_E}
        diff = {f: np.array([np.array(s["corrected"][f]) - s["truth"] for s in stats]) for f in FILTERS_E}
        out["corrected_minus_truth_mean"] = {f: diff[f].mean(axis=0).tolist() for f in FILTERS_E}
    pdiff = {f: np.array([np.array(s["plain"][f]) - s["truth"] for s in stats]) for f in FILTERS_E}
    out["plain_minus_own_truth_mean"] = {f: pdiff[f].mean(axis=0).tolist() for f in FILTERS_E}
    out["plain_over_own_truth_mean"] = {f: np.mean([np.array(s["plain"][f]) / s["truth"] for s in stats], axis=0).tolist()
                                        for f in FILTERS_E}
    if "vg" in stats[0]:
        out["vg"] = {k: float(np.median([s["vg"][k] for s in stats])) for k in stats[0]["vg"]}
    return out


def module_e(n_paths: int = SIMULATION_PATHS, workers: int = SIMULATION_WORKERS):
    t0 = time.time()
    _e_init()
    tau = _E["tau"]
    n = len(tau)
    ref = {s: np.maximum(np.searchsorted(tau, tau - s, side="right") - 1, 0) for s in S_E}
    num = {s: np.zeros(n) for s in S_E}
    den = {f: {s: np.zeros(n) for s in S_E} for f in FILTERS_E if f != "raw"}
    kept, base_stats = {}, {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_e_init) as pool:
        for m, paths, st in pool.map(_e_base, range(n_paths)):
            x = paths["raw"].astype(np.float64)
            for s in S_E:
                u = ref[s]
                num[s] += (x - x[u]) ** 2
                for f in den:
                    p = paths[f].astype(np.float64)
                    den[f][s] += (p - p[u]) ** 2
            kept[m] = paths
            base_stats[m] = st
            if (m + 1) % 10 == 0:
                print(f"E base {m + 1}/{n_paths} {time.time() - t0:.0f}s", flush=True)
    c = {f: {s: np.where(den[f][s] > 0.0, num[s] / np.maximum(den[f][s], 1e-300), 1.0) for s in S_E} for f in den}
    E_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(E_CACHE, **{f"{f}_{s}": c[f][s].astype(np.float32) for f in c for s in S_E})
    _E["c"] = {f: {s: c[f][s].astype(np.float32).astype(np.float64) for s in S_E} for f in c}
    for m, paths in kept.items():
        base_stats[m]["corrected"] = {
            f: [signature_vol(tau, paths[f].astype(np.float64), s, None if f == "raw" else _E["c"][f][s]) for s in S_E]
            for f in FILTERS_E}
    del kept
    cases = {"bm": list(base_stats.values())}
    tasks = [(case, m) for case in ("bm_noise", "vg", "vg_noise") for m in range(n_paths)]
    with ProcessPoolExecutor(max_workers=workers, initializer=_e_init) as pool:
        for k, (case, m, st) in enumerate(pool.map(_e_case, tasks)):
            cases.setdefault(case, []).append(st)
            if (k + 1) % 20 == 0:
                print(f"E cases {k + 1}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    E_CACHE.unlink(missing_ok=True)
    summary = {"s": list(S_E), "sigma_annual": SIMULATION_SIGMA_ANNUAL, "vg_beta": VG_BETA,
               "noise_scale_cents": NOISE_SCALE_CENTS, "noise_df": NOISE_DF,
               "cases": {k: _summarize_case(v) for k, v in cases.items()}}
    _plot_e(summary)
    return summary


def _plot_e(summary):
    s = summary["s"]
    cs = summary["cases"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for ax, case, title in ((axes[0], "bm", "without noise"), (axes[1], "bm_noise", "with noise")):
        d = cs[case]
        ax.semilogx(s, d["corrected_mean"]["huber"], "r-", label="Huber price, with c_{t,s}")
        ax.semilogx(s, d["plain_mean"]["huber"], "g:", label="Huber price, c = 1")
        ax.semilogx(s, d["corrected_mean"]["linear"], "-", color="tab:orange", lw=0.8, label="linear, with c_{t,s}")
        ax.semilogx(s, d["corrected_mean"]["median"], "-", color="tab:green", lw=0.8, label="median, with c_{t,s}")
        ax.semilogx(s, cs["bm"]["plain_mean"]["raw"], color="tab:purple", label="simulated price")
        ax.axhline(100 * SIMULATION_SIGMA_ANNUAL, color="k", ls="--", lw=0.8, label="truth")
        ax.set_title(f"Brownian price, {title}")
        ax.set_xlabel("s (seconds)")
        ax.set_ylim(15, 60)
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
    axes[0].set_ylabel("annualized volatility (%)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(_out("E_figure_a5_brownian.png"), dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for ax, case, title in ((axes[0], "vg", "without noise"), (axes[1], "vg_noise", "with noise")):
        d = cs[case]
        ax.semilogx(s, d["corrected_minus_truth_mean"]["huber"], "r-", label="Huber price, with c_{t,s}")
        ax.semilogx(s, d["plain_minus_own_truth_mean"]["huber"], "g:", label="Huber price, c = 1")
        ax.semilogx(s, d["corrected_minus_truth_mean"]["linear"], "-", color="tab:orange", lw=0.8, label="linear, with c_{t,s}")
        ax.semilogx(s, cs["vg"]["plain_minus_own_truth_mean"]["raw"], color="tab:purple", label="simulated price")
        ax.axhline(0.0, color="k", ls="--", lw=0.8)
        ax.set_title(f"variance-gamma price, {title}")
        ax.set_xlabel("s (seconds)")
        ax.set_ylim(-30, 60)
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
    axes[0].set_ylabel("estimate minus truth (annualized volatility, %)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(_out("E_figure_a6_variance_gamma.png"), dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# F. Cross-asset
# ---------------------------------------------------------------------------

def _f_one(ticker: str) -> dict:
    sp.ensure_filtered(ticker)
    spot = sp.spot_ticker(ticker)
    row = sp.table_b2_row(ticker, spot)
    row["ticker"] = ticker
    row["paper"] = dict(zip(sp.B2_KEYS, PAPER_B2[ticker])) if ticker in PAPER_B2 else None
    row["per_day"] = spot["per_day"]
    sp._LIGHT.clear()
    return jsonify(row)


GROUPS = ("above 10", "3 to 10", "1 to 3", "below 1")


def _group(tps: float) -> str:
    hi, mid, lo = TABLE_B1_EDGES
    if tps > hi:
        return GROUPS[0]
    if tps >= mid:
        return GROUPS[1]
    if tps >= lo:
        return GROUPS[2]
    return GROUPS[3]


def _cross_section(rows: dict, source: str) -> dict:
    """Appendix B summary statements, on our rows (``replication``) or the
    published ones (``paper``). Group statements are over stocks, so SPY is
    left out of them, as in the text."""
    def val(t, k):
        return rows[t]["replication"][k] if source == "replication" else rows[t]["paper"][k]
    names = [t for t in rows if rows[t]["paper"] is not None]
    stocks = [t for t in names if t != "SPY"]
    others = [t for t in stocks if t != "NVDA"]
    tps = {t: val(t, "trades_per_second") for t in names}
    hill = {t: val(t, "hill") for t in names}
    fast = [t for t in stocks if _group(tps[t]) == GROUPS[0]]
    slow = [t for t in stocks if _group(tps[t]) == GROUPS[3]]
    under3 = [t for t in stocks if tps[t] < TABLE_B1_EDGES[1]]

    def med(group, k):
        v = [val(t, k) for t in group]
        v = [x for x in v if x == x]
        return float(np.median(v)) if v else float("nan")
    return {
        "hill_buckets_53_other_stocks": {
            "1-2": sum(1 <= hill[t] < 2 for t in others), "2-3": sum(2 <= hill[t] < 3 for t in others),
            "3-4": sum(3 <= hill[t] < 4 for t in others), ">=4": sum(hill[t] >= 4 for t in others),
            "<1": sum(hill[t] < 1 for t in others)},
        "hill_max_above_10_tps": float(max(hill[t] for t in fast)),
        "n_hill_at_least_4": sum(hill[t] >= 4 for t in names),
        "median_jump_above_10_tps": med(fast, "linear_jump"),
        "median_jump_below_1_tps": med(slow, "linear_jump"),
        "n_jump_at_least_3_of_54_stocks": sum(val(t, "linear_jump") >= 3 for t in stocks),
        "median_linear_over_huber_above_10_tps": med(fast, "linear_over_huber"),
        "median_linear_over_huber_below_1_tps": med(slow, "linear_over_huber"),
        "median_outlier_pct_above_10_tps": med(fast, "outlier_pct"),
        "median_outlier_pct_below_1_tps": med(slow, "outlier_pct"),
        "median_signature_above_10_tps": med(fast, "signature_60s_1s"),
        "median_signature_below_3_tps": med(under3, "signature_60s_1s"),
        "negative_acf1_12day_stocks": sorted(t for t in stocks if val(t, "acf_lag1") < 0),
    }


def _table_b1(rows: dict) -> dict:
    groups = {}
    for t, r in rows.items():
        if t == "SPY" or r["paper"] is None:
            continue
        groups.setdefault(_group(r["paper"]["trades_per_second"]), []).append(r)
    out = {}
    for g in PAPER_B1:
        rs = groups.get(g, [])
        if not rs:
            continue
        out[g] = {
            "stocks": len(rs),
            "lag1_huber_16oct": float(np.median([r["acf_16oct"]["huber"][0] for r in rs])),
            "lag1_huber_12day": float(np.median([r["replication"]["acf_lag1"] for r in rs])),
            "lag1_trades_16oct": float(np.median([r["acf_16oct"]["trades"][0] for r in rs])),
            "lag2_huber_16oct": float(np.median([r["acf_16oct"]["huber"][1] for r in rs])),
            "lag2_huber_12day": float(np.median([r["replication"]["acf_lag2"] for r in rs])),
            "lag2_trades_16oct": float(np.median([r["acf_16oct"]["trades"][1] for r in rs])),
            "paper": dict(zip(("stocks", "lag1_huber_16oct", "lag1_huber_12day", "lag1_trades_16oct",
                               "lag2_huber_16oct", "lag2_huber_12day", "lag2_trades_16oct"), PAPER_B1[g])),
        }
    return out


def _plot_f(rows: dict) -> None:
    six = [t for t in FIGURE_B1_TICKERS if t in rows]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, name in zip(axes.ravel(), six):
        sig = rows[name]["signature_12day"]
        for label, col in (("trades", "0.6"), ("linear", "tab:blue"), ("huber", "tab:red")):
            ax.loglog(sp.SIG_S, [sig[label][str(s)] for s in sp.SIG_S], color=col,
                      label={"trades": "raw trades", "linear": "linear filter", "huber": "Huber filter"}[label])
        ax.set_title(f"{name}, {rows[name]['replication']['trades_per_second']:.2g} trades/s")
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("s (seconds)")
    for ax in axes[:, 0]:
        ax.set_ylabel("annualized volatility (%)")
    fig.tight_layout()
    fig.savefig(_out("F_figure_b1_signatures.png"), dpi=140)
    plt.close(fig)

    keys = (("outlier_pct", "outliers (% of trades)", False), ("hill", "Hill index (absolute residual)", False),
            ("linear_over_huber", "linear / Huber spot volatility", False),
            ("signature_60s_1s", "signature, 1 min over 1 s", False),
            ("linear_jump", "linear jump at its largest residual (×)", True),
            ("acf_lag1", "ACF at lag 1, Huber filtered price", False))
    names = [t for t in rows if rows[t]["paper"] is not None]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (k, lab, logy) in zip(axes.ravel(), keys):
        x = [rows[t]["replication"]["trades_per_second"] for t in names]
        ax.scatter(x, [rows[t]["paper"][k] for t in names], s=14, facecolors="none", edgecolors="0.4", label="paper")
        ax.scatter(x, [rows[t]["replication"][k] for t in names], s=10, color="tab:red", label="replication")
        for t, col in (("NVDA", "tab:red"), ("SPY", "tab:blue")):
            if t in rows:
                ax.annotate(t, (rows[t]["replication"]["trades_per_second"], rows[t]["replication"][k]),
                            fontsize=7, color=col, xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_title(lab, fontsize=9)
        ax.set_xlabel("average trades per second")
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(_out("F_figure_b2_cross_section.png"), dpi=140)
    plt.close(fig)


def module_f(tickers=ASSETS, workers: int = CROSS_ASSET_WORKERS) -> dict:
    t0 = time.time()
    order = sorted(tickers, key=lambda t: -(PAPER_B2.get(t, (0.0,))[0]))
    rows = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(_f_one, order):
            rows[row["ticker"]] = row
            rep, pap = row["replication"], row["paper"] or {}
            print(f"F {row['ticker']} a={row['a_qlike']:.0f} ratio={rep['linear_over_huber']:.3f}/{pap.get('linear_over_huber')} "
                  f"jump={rep['linear_jump']:.1f}/{pap.get('linear_jump')} hill={rep['hill']:.2f}/{pap.get('hill')} "
                  f"{time.time() - t0:.0f}s", flush=True)
    rows = {t: rows[t] for t in ASSETS if t in rows}
    with _out("F_table_b2.json").open("w") as f:
        json.dump({t: {k: v for k, v in r.items() if k != "signature_per_day"} for t, r in rows.items()}, f, indent=1)
    _plot_f(rows)
    out = {"tickers": list(rows), "table_b1": _table_b1(rows),
           "cross_section": {"replication": _cross_section(rows, "replication"), "paper": _cross_section(rows, "paper")}}
    reps = [r for r in rows.values() if r["paper"] is not None]
    out["abs_gap_median"] = {k: float(np.median([abs(r["replication"][k] - r["paper"][k]) for r in reps]))
                             for k in sp.B2_KEYS}
    out["a_by_ticker"] = {t: {"a_trades": r["a_qlike"], "a_seconds": r["a_seconds"]} for t, r in rows.items()}
    out["paper_comparison"] = _paper_comparison(reps)
    return out


def _paper_comparison(reps: list) -> dict:
    """How the columns built on the linear spot path and the Hill index compare
    with the published Table B.2, row by row."""
    def col(k, src):
        return np.array([r[src][k] for r in reps], dtype=np.float64)
    ratio_r, ratio_p = col("linear_over_huber", "replication"), col("linear_over_huber", "paper")
    jump_r, jump_p = col("linear_jump", "replication"), col("linear_jump", "paper")
    hill_r, hill_p = col("hill", "replication"), col("hill", "paper")
    names = [r["ticker"] for r in reps]
    rel = jump_r / jump_p
    hill_by_group = {}
    for g in GROUPS:
        idx = [i for i, r in enumerate(reps) if r["ticker"] != "SPY" and _group(r["paper"]["trades_per_second"]) == g]
        gap = hill_r[idx] - hill_p[idx]
        hill_by_group[g] = {"stocks": len(idx), "n_above_tolerance": int(np.sum(gap > HILL_MATCH_TOLERANCE)),
                            "gap_min": float(gap.min()), "gap_max": float(gap.max()),
                            "gaps": {names[i]: float(hill_r[i] - hill_p[i]) for i in idx}}
    return {
        "rows": len(reps),
        "linear_over_huber": {"n_above_paper": int(np.sum(ratio_r > ratio_p)),
                              "median_gap": float(np.median(ratio_r - ratio_p)),
                              "corr": float(np.corrcoef(ratio_r, ratio_p)[0, 1])},
        "linear_jump": {"median_relative": float(np.median(rel)),
                        "share_within_tolerance": float(np.mean(np.abs(rel - 1.0) <= JUMP_MATCH_TOLERANCE)),
                        "tolerance": JUMP_MATCH_TOLERANCE,
                        "log_corr": float(np.corrcoef(np.log(jump_r), np.log(jump_p))[0, 1])},
        "hill": {"n_within_tolerance": int(np.sum(np.abs(hill_r - hill_p) <= HILL_MATCH_TOLERANCE)),
                 "tolerance": HILL_MATCH_TOLERANCE, "by_group": hill_by_group},
    }


# ---------------------------------------------------------------------------
# G. Sensitivity of the Table B.2 columns
# ---------------------------------------------------------------------------

def _known_a() -> dict:
    path = _out("F_table_b2.json")
    rows = json.loads(path.read_text()) if path.exists() else {}
    return {t: float(r["a_qlike"]) for t, r in rows.items()}


def _a_for(ticker: str, days, known: dict) -> float:
    if ticker in known:
        return known[ticker]
    return float(sp.select_mandelbrot_a(days)[0])


def _ratio_checks(known: dict) -> dict:
    out = {}
    for ticker in dict.fromkeys(RATIO_CHECK_TICKERS + (AGGREGATION_CHECK_TICKER,)):
        sp.ensure_filtered(ticker)
        days = sp.prepare(ticker)
        a_hat = _a_for(ticker, days, known)
        at_hat = sp.linear_over_huber(days, a_hat)
        row = {"a_qlike": a_hat, "pooled": at_hat["pooled"],
               "paper": PAPER_B2[ticker][5] if ticker in PAPER_B2 else None}
        if ticker in RATIO_CHECK_TICKERS:
            row["by_a"] = {str(a): sp.linear_over_huber(days, a)["pooled"] for a in RATIO_CHECK_A_GRID}
            vals = list(row["by_a"].values())
            row["by_a_range"] = [float(min(vals)), float(max(vals))]
        if ticker == AGGREGATION_CHECK_TICKER:
            row["daily"] = at_hat["daily"]
            row["median_of_daily"] = at_hat["median_of_daily"]
            row["mean_of_daily"] = at_hat["mean_of_daily"]
            sp.add_linear_diurnal(ticker, days)
            row["linear_path_diurnal"] = sp.linear_over_huber(days, a_hat, linear_level="level_linear")["pooled"]
        print(f"G ratio {ticker}: {json.dumps(jsonify(row, drop={'daily'}))}", flush=True)
        out[ticker] = row
        sp._LIGHT.clear()
    return out


def _damping_checks() -> dict:
    """Simulated Huber and linear damping factors at s = SPOT_LAG_S on 16 October."""
    out = {}
    for ticker in DAMPING_CHECK_TICKERS:
        sp.ensure_filtered(ticker)
        loaded = load_day(ticker, OCT16)
        df = loaded.df
        tau = df["tau_open"].to_numpy()
        res = run_pipeline(df["tau_s"].to_numpy(), df["log_price"].to_numpy(), tau_open=tau,
                           presample_abs=sp.presample_from_cache(ticker, OCT16), T60=loaded.T60,
                           B=TAIL_DRAWS, seed=sp.session_seed(ticker, OCT16))
        s_long = float(TABLE2_S[-1])
        vol = s_second_diff_stats(tau, res.theta, s_list=(s_long,),
                                  c_by_s={s_long: damping_correction(tau, s_long)})[s_long]["annualized_vol"]
        sigma = vol / 100.0 / np.sqrt(252.0 * SESSION_SECONDS)
        c = simulated_damping(tau, [SPOT_LAG_S], sigma, res.delta, halflife=PRICE_HALF_LIFE_TRADES,
                              n_paths=DAMPING_PATHS, seed=sp.session_seed(ticker, OCT16))
        c_h, c_l = c["huber"][SPOT_LAG_S], c["linear"][SPOT_LAG_S]
        vols = {name: s_second_diff_stats(tau, res.theta, s_list=(SPOT_LAG_S,), c_by_s={SPOT_LAG_S: cc})
                [SPOT_LAG_S]["annualized_vol"] for name, cc in (("huber", c_h), ("linear", c_l))}
        ok = np.isfinite(c_h) & np.isfinite(c_l) & (c_l > 0)
        out[ticker] = {"mean_c_huber": float(np.mean(c_h[ok])), "mean_c_linear": float(np.mean(c_l[ok])),
                       "mean_relative_gap": float(np.mean(c_h[ok]) / np.mean(c_l[ok]) - 1.0),
                       "median_pointwise_relative_gap": float(np.median(c_h[ok] / c_l[ok]) - 1.0),
                       "huber_vol_with_huber_c": vols["huber"], "huber_vol_with_linear_c": vols["linear"],
                       "vol_relative_gap": vols["huber"] / vols["linear"] - 1.0}
        print(f"G damping {ticker}: {out[ticker]}", flush=True)
    return out


def _hill_average_checks() -> dict:
    """Hill index of the 16 October Huber residual, log-uniform and arithmetic averages over k."""
    rows = {}
    for ticker in ASSETS:
        sp.ensure_filtered(ticker)
        y = load_day(ticker, OCT16).df["log_price"].to_numpy()
        with np.load(sp.cache_path(ticker, OCT16)) as f:
            r = y - np.asarray(f["theta"], dtype=np.float64)
        rows[ticker] = {"log_uniform": hill_tail_index(r), "arithmetic": hill_tail_index(r, average="arithmetic"),
                        "paper": PAPER_B2[ticker][2] if ticker in PAPER_B2 else None,
                        "paper_trades_per_second": PAPER_B2[ticker][0] if ticker in PAPER_B2 else None}
    by_group = {}
    for g in GROUPS:
        names = [t for t, r in rows.items() if t != "SPY" and r["paper"] is not None
                 and _group(r["paper_trades_per_second"]) == g]
        by_group[g] = {k: float(np.median([abs(rows[t][k] - rows[t]["paper"]) for t in names]))
                       for k in ("log_uniform", "arithmetic")}
    return {"rows": rows, "median_abs_gap_by_group": by_group}


def _jump_check(known: dict) -> dict:
    ticker = JUMP_CHECK_TICKER
    sp.ensure_filtered(ticker)
    days = sp.prepare(ticker)
    a_hat = _a_for(ticker, days, known)
    out = {"ticker": ticker, "a_qlike": a_hat, "at_a_qlike": sp.linear_jump_at(ticker, days, a_hat),
           "a_fixed": NVDA_A_PUBLISHED, "at_a_fixed": sp.linear_jump_at(ticker, days, NVDA_A_PUBLISHED)}
    sp._LIGHT.clear()
    print(f"G jump {ticker}: a={a_hat:.0f} {out['at_a_qlike']['ratio']:.1f}, "
          f"a={NVDA_A_PUBLISHED:.0f} {out['at_a_fixed']['ratio']:.1f}", flush=True)
    return out


def module_g() -> dict:
    known = _known_a()
    return {"ratio": _ratio_checks(known), "damping": _damping_checks(),
            "hill_average": _hill_average_checks(), "jump": _jump_check(known)}


def _save(path: Path, computed: dict) -> None:
    """Merge this run's modules into the file, which another run may have updated."""
    disk = json.loads(path.read_text()) if path.exists() else {}
    disk.update(jsonify(computed))
    path.write_text(json.dumps(disk, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Experiment 4: robustness checks and the appendices.")
    parser.add_argument("--modules", nargs="+", choices=MODULES, default=list(MODULES))
    args = parser.parse_args()
    log_to_results("exp4")
    path = _out("summary.json")
    summary = {}
    a_nvda = None
    runners = {"A": module_a, "C": module_c, "D": module_d, "E": module_e, "F": module_f, "G": module_g}
    for mod in ("A", "C", "D", "E", "F", "B", "G"):
        if mod not in args.modules:
            continue
        print(f"=== {mod} ===", flush=True)
        if mod == "B":
            summary["B"] = module_b(a_nvda if a_nvda is not None else _known_a().get(NVDA_TICKER))
        else:
            summary[mod] = runners[mod]()
        if mod == "F":
            a_nvda = summary["F"]["a_by_ticker"].get(NVDA_TICKER, {}).get("a_trades")
        _save(path, summary)
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
