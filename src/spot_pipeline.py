"""Multi-day filtering cache, diurnal spot volatility and the Table B.2 statistics.

Spot volatility needs every session of an asset at once: the diurnal curve of
a day is fit to the previous ten sessions, the opening background to the
previous five, and the Mandelbrot ``a`` is chosen by QLIKE over all twelve
paper days. Each session is therefore filtered once (the median and Huber
filters with the stratified estimator at B = 100, the scale filter and the
exact linear filter) and stored in ``CACHE``; everything below reads that
cache.

* ``Z`` is the 1-second pre-averaged variance increment with the closed-form
  damping factor of the 30-lag linear filter, which Appendix A.6 puts within
  1% of the Huber factor on average.
* The diurnal curve is a quadratic plus four sine-cosine pairs fit to the log
  of the ten-session average of the 10-second uniform estimator of
  Huber-filtered prices (Section 4.6).
* Spot variance divides a hyperbolic weighted mean of ``Z / diurnal`` by the
  same mean of the clock gaps and multiplies back by the diurnal level; the
  first 15 seconds blend the unadjusted filter towards the median
  first-minute variance of the previous five sessions.
* ``a`` minimises the QLIKE loss of ``σ̂(u − 60; a)`` against the 10-second
  uniform estimate ``σ̄(u)``, u from 75 s to the close, summed over the twelve
  sessions (Section 4.6 for NVDA, Appendix B for every other asset).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from data.data_loader import load_day

from .config import (
    ASSETS,
    DIURNAL_PRESAMPLE_DATES,
    DIURNAL_SESSIONS,
    JUMP_BASE_FROM_S,
    JUMP_BASE_TO_S,
    JUMP_MIN_TAU_S,
    JUMP_PEAK_FROM_S,
    JUMP_PEAK_TO_S,
    LONG_LAG_S,
    OCT16,
    PAPER_DATES,
    PRESAMPLE_DATES,
    PRESAMPLE_OPENINGS,
    SESSION_SECONDS,
    SESSION_SEED,
    SIGNATURE_S,
    SPOT_LAG_S,
    SPOT_SAMPLE_STEP,
    STARTUP_S,
    TAIL_DRAWS,
)
from .empirical_pipeline import opening_abs_residuals, run_pipeline
from .hf_filters import damping_correction
from .metrics import hill_tail_index, psi_star_cents, sample_acf, signature_vol
from .spot_vol import (
    HEXP_HALF_LIVES_S,
    annualized_percent,
    blend_open,
    clock_mixture_variance,
    diurnal_hyperbolic_spot,
    diurnal_variance,
    fit_diurnal,
    increments,
    onto_seconds,
    qlike,
    select_mandelbrot_a,
    trade_mixture_variance,
    uniform_spot_grid,
)

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache" / "filtered"
ALL_DATES = DIURNAL_PRESAMPLE_DATES + PRESAMPLE_DATES + PAPER_DATES
SIG_S = SIGNATURE_S

_LIGHT: dict = {}

_SCALAR_KEYS = ("signature", "acf", "acf_trades", "outlier_share", "hill", "psi_star_cents", "close_price",
                "t_star", "t_star_price", "t_star_size", "t_star_filtered", "open_delta_cents")


def previous(date: str, n: int) -> list[str]:
    i = ALL_DATES.index(date)
    return ALL_DATES[max(0, i - n) : i]


def session_seed(ticker: str, date: str) -> int:
    k = ASSETS.index(ticker) if ticker in ASSETS else len(ASSETS)
    return SESSION_SEED + 1000 * k + ALL_DATES.index(date)


def cache_path(ticker: str, date: str) -> Path:
    return CACHE / f"{ticker}_{date}.npz"


def _filter_one(ticker: str, date: str, presample: np.ndarray) -> np.ndarray:
    t0 = time.time()
    loaded = load_day(ticker, date)
    df = loaded.df
    y = df["log_price"].to_numpy()
    tau = df["tau_open"].to_numpy()
    price = df["price"].to_numpy()
    res = run_pipeline(df["tau_s"].to_numpy(), y, tau_open=tau, presample_abs=presample,
                       T60=loaded.T60, B=TAIL_DRAWS, seed=session_seed(ticker, date))
    c = damping_correction(tau, SPOT_LAG_S)
    z, _dt = increments(tau, res.theta, c=c)
    grid = uniform_spot_grid(tau, z)
    minute = float(z[tau <= STARTUP_S].sum() / STARTUP_S) if np.any(tau <= STARTUP_S) else float(np.nanmedian(grid))
    open_resid = opening_abs_residuals(y, tau)
    sig = np.zeros((3, len(SIG_S)), dtype=np.float64)
    for k, s in enumerate(SIG_S):
        cs = damping_correction(tau, float(s))
        sig[0, k] = signature_vol(tau, y, float(s))
        sig[1, k] = signature_vol(tau, res.linear, float(s), cs)
        sig[2, k] = signature_vol(tau, res.theta, float(s), cs)
    ps = psi_star_cents(price, res.theta, res.delta_prev)
    t_star = int(np.argmax(np.abs(res.resid_hub)))
    dest = cache_path(ticker, date)
    tmp = dest.with_name(dest.stem + "_tmp.npz")
    np.savez(
        tmp,
        tau=tau,
        theta=res.theta,
        linear=res.linear,
        grid=grid,
        minute_var=np.float64(minute),
        open_resid=open_resid,
        signature=sig,
        acf=sample_acf(np.diff(np.exp(res.theta)), lags=(1, 2)),
        acf_trades=sample_acf(np.diff(price), lags=(1, 2)),
        outlier_share=np.float64((res.d_pos + res.d_neg).mean()),
        hill=np.float64(hill_tail_index(res.resid_hub)),
        psi_star_cents=np.float64(ps["psi_star_cents"]),
        close_price=np.float64(price[-1]),
        t_star=np.int64(t_star),
        t_star_price=np.float64(price[t_star]),
        t_star_size=np.float64(df["size"].to_numpy()[t_star]),
        t_star_filtered=np.float64(np.exp(res.theta[t_star])),
        open_delta_cents=np.float64(res.delta_prev[0] * price[0] * 100.0),
    )
    tmp.replace(dest)
    print(f"filtered {ticker} {date} T={len(tau)} in {time.time() - t0:.0f}s", flush=True)
    return open_resid


def ensure_filtered(ticker: str) -> None:
    """Filter every missing session of ``ticker``, oldest first (the presample
    of a session is the opening residuals of the five before it)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    opens: dict[str, np.ndarray] = {}
    for date in ALL_DATES:
        dest = cache_path(ticker, date)
        if dest.exists() and dest.stat().st_size > 0:
            with np.load(dest) as z:
                opens[date] = np.array(z["open_resid"], dtype=np.float64, copy=True)
            continue
        parts = [opens[d] for d in previous(date, PRESAMPLE_OPENINGS) if len(opens.get(d, ())) > 0]
        pres = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)
        opens[date] = _filter_one(ticker, date, pres)


def light(ticker: str, date: str):
    key = (ticker, date)
    if key not in _LIGHT:
        with np.load(cache_path(ticker, date)) as z:
            _LIGHT[key] = (np.asarray(z["grid"], dtype=np.float64), float(z["minute_var"]))
    return _LIGHT[key]


def scalars(ticker: str, date: str) -> dict:
    with np.load(cache_path(ticker, date)) as z:
        return {k: z[k].copy() for k in _SCALAR_KEYS if k in z.files}


def coef_and_background(ticker: str, date: str):
    grids = np.vstack([light(ticker, d)[0] for d in previous(date, DIURNAL_SESSIONS)])
    count = np.sum(np.isfinite(grids), axis=0)
    mean = np.full(grids.shape[1], np.nan)
    ok = count > 0
    mean[ok] = np.nansum(grids[:, ok], axis=0) / count[ok]
    coef = fit_diurnal(mean)
    minutes = [m for _g, m in (light(ticker, d) for d in previous(date, PRESAMPLE_OPENINGS)) if np.isfinite(m) and m > 0]
    background = float(np.median(minutes)) if minutes else float(np.nanmean(grids[-1]))
    return coef, background


def prepare(ticker: str) -> list[dict]:
    """The twelve paper sessions with ``Z`` (Huber and linear), clock gaps,
    diurnal level, opening background and the QLIKE target."""
    days = []
    for date in PAPER_DATES:
        with np.load(cache_path(ticker, date)) as f:
            tau = np.asarray(f["tau"], dtype=np.float64)
            theta = np.asarray(f["theta"], dtype=np.float64)
            linear = np.asarray(f["linear"], dtype=np.float64)
            target = np.asarray(f["grid"], dtype=np.float64)
        coef, background = coef_and_background(ticker, date)
        c = damping_correction(tau, SPOT_LAG_S)
        z, dt = increments(tau, theta, c=c)
        z_lin, _ = increments(tau, linear, c=c)
        days.append({
            "date": date,
            "tau": tau,
            "Z": np.ascontiguousarray(z),
            "Z_lin": np.ascontiguousarray(z_lin),
            "dt": np.ascontiguousarray(dt),
            "level": np.ascontiguousarray(diurnal_variance(tau, coef)),
            "background": background,
            "target": target,
        })
    return days


def jump(tau, ann, t_star: int) -> dict:
    """Peak over the window −6 to +12 minutes over the median −15 to −3 minutes."""
    ts = float(tau[t_star])
    sec = onto_seconds(tau, ann)
    u = np.arange(len(sec), dtype=np.float64)
    before = (u >= ts + JUMP_BASE_FROM_S) & (u <= ts + JUMP_BASE_TO_S)
    around = (u >= ts + JUMP_PEAK_FROM_S) & (u <= ts + JUMP_PEAK_TO_S)
    base = float(np.nanmedian(sec[before]))
    peak = float(np.nanmax(sec[around]))
    return {"base_pct": base, "peak_pct": peak, "ratio": peak / base, "tau_s": ts}



def jump_trade(ticker: str, date: str = OCT16) -> int:
    """The trade behind the Table B.2 jump: the largest absolute Huber residual
    among trades at least 15 minutes after the open, so that the baseline
    window (15 to 3 minutes before the trade) lies inside the session."""
    with np.load(cache_path(ticker, date)) as f:
        tau = np.asarray(f["tau"], dtype=np.float64)
        theta = np.asarray(f["theta"], dtype=np.float64)
    y = load_day(ticker, date).df["log_price"].to_numpy()
    r = np.abs(y - theta)
    r[tau < JUMP_MIN_TAU_S] = -1.0
    return int(np.argmax(r))


def spot_ticker(ticker: str, keep: bool = False) -> dict:
    """QLIKE ``a``, the Huber and linear spot paths, the linear/Huber ratio and
    the 16 October jump for one asset."""
    t0 = time.time()
    days = prepare(ticker)
    a_hat, a_path = select_mandelbrot_a(days)
    print(f"{ticker} QLIKE a = {a_hat:.0f} trades in {time.time() - t0:.0f}s", flush=True)
    level_l, level_h, pointwise, per_day = [], [], [], []
    seconds_h, seconds_l = [], []
    jmp = None
    n_trades = 0
    for d in days:
        hub = diurnal_hyperbolic_spot(d["tau"], d["Z"], d["dt"], d["level"], d["background"], a_hat)
        lin = diurnal_hyperbolic_spot(d["tau"], d["Z_lin"], d["dt"], d["level"], d["background"], a_hat)
        idx = np.arange(SPOT_SAMPLE_STEP - 1, len(lin), SPOT_SAMPLE_STEP)
        ok = np.isfinite(lin[idx]) & np.isfinite(hub[idx]) & (lin[idx] > 0) & (hub[idx] > 0)
        level_l.append(np.sqrt(lin[idx][ok]))
        level_h.append(np.sqrt(hub[idx][ok]))
        pointwise.append(np.sqrt(lin[idx][ok] / hub[idx][ok]))
        integrated = float(100.0 * np.sqrt(252.0 * max(d["Z"].sum(), 0.0)))
        per_day.append({"date": d["date"], "T": int(len(d["tau"])), "integrated_huber_1s": integrated,
                        "median_spot_huber_pct": float(np.nanmedian(annualized_percent(hub)))})
        seconds_h.append(onto_seconds(d["tau"], annualized_percent(hub)))
        seconds_l.append(onto_seconds(d["tau"], annualized_percent(lin)))
        n_trades += len(d["tau"])
        if d["date"] == OCT16:
            sc = scalars(ticker, OCT16)
            k = jump_trade(ticker)
            jmp = jump(d["tau"], annualized_percent(lin), k)
            jump_h = jump(d["tau"], annualized_percent(hub), k)
            df16 = load_day(ticker, OCT16).df
            with np.load(cache_path(ticker, OCT16)) as f:
                filt = float(np.exp(f["theta"][k]))
            px = float(df16["price"].to_numpy()[k])
            jmp.update({"huber_ratio": jump_h["ratio"], "price": px, "size": float(df16["size"].to_numpy()[k]),
                        "filtered": filt, "residual_cents": (px - filt) * 100.0,
                        "largest_residual_of_day": {"tau_s": float(d["tau"][int(sc["t_star"])]),
                                                    "price": float(sc["t_star_price"]),
                                                    "filtered": float(sc["t_star_filtered"])}})
        if keep:
            d["hub"] = hub
    rate = n_trades / (len(PAPER_DATES) * SESSION_SECONDS)
    result = {
        "a_qlike": a_hat,
        "a_seconds": a_hat / rate,
        "a_path": a_path,
        "ratio": float(np.median(np.concatenate(level_l)) / np.median(np.concatenate(level_h))),
        "median_pointwise_ratio": float(np.median(np.concatenate(pointwise))),
        "trades_per_second": rate,
        "per_day": per_day,
        "jump": jmp,
        "seconds_huber": np.vstack(seconds_h),
        "seconds_linear": np.vstack(seconds_l),
    }
    if keep:
        result["prepared"] = days
    return result


B2_KEYS = ("trades_per_second", "outlier_pct", "hill", "psi_star_cents", "psi_star_bp",
           "linear_over_huber", "signature_60s_1s", "linear_jump", "acf_lag1", "acf_lag2")


def table_b2_row(ticker: str, spot: dict) -> dict:
    """One row of Table B.2, plus the 12-day signatures and the ACFs Table B.1 needs."""
    sc16 = scalars(ticker, OCT16)
    per = [scalars(ticker, d) for d in PAPER_DATES]
    sigs = np.stack([p["signature"] for p in per])
    acfs = np.stack([p["acf"] for p in per])
    i1, i60 = SIG_S.index(1.0), SIG_S.index(60.0)
    avg = np.sqrt(np.mean(sigs ** 2, axis=0))
    psi = float(sc16["psi_star_cents"])
    row = {
        "trades_per_second": spot["trades_per_second"],
        "outlier_pct": 100.0 * float(sc16["outlier_share"]),
        "hill": float(sc16["hill"]),
        "psi_star_cents": psi,
        "psi_star_bp": 1e4 * (psi / 100.0) / float(sc16["close_price"]),
        "linear_over_huber": spot["ratio"],
        "signature_60s_1s": float(avg[2, i60] / avg[2, i1]),
        "linear_jump": spot["jump"]["ratio"],
        "acf_lag1": float(np.median(acfs[:, 0])),
        "acf_lag2": float(np.median(acfs[:, 1])),
    }
    return {
        "replication": row,
        "a_qlike": spot["a_qlike"],
        "a_seconds": spot["a_seconds"],
        "jump": spot["jump"],
        "acf_16oct": {"huber": [float(v) for v in sc16["acf"]],
                      "trades": [float(v) for v in sc16.get("acf_trades", [np.nan, np.nan])]},
        "signature_12day": {name: dict(zip(map(str, SIG_S), avg[i].tolist()))
                            for i, name in enumerate(("trades", "linear", "huber"))},
        "signature_per_day": sigs.tolist(),
        "open_delta_cents_16oct": float(sc16["open_delta_cents"]),
    }


# ---------------------------------------------------------------------------
# Example 5 superposition (Figure 10)
# ---------------------------------------------------------------------------

def _mixture_variances(alpha, day, mode: str, rate: float):
    a = np.maximum(np.asarray(alpha, dtype=np.float64), 0.0)
    a = a / a.sum()
    half = np.asarray(HEXP_HALF_LIVES_S, dtype=np.float64)
    n15 = int(np.sum(day["tau"] < 15.0))
    ones = np.ones(n15)
    if mode == "clock":
        decay = np.log(2.0) / half
        var = clock_mixture_variance(day["Z"], day["dt"], a, decay, day["level"])
        early = clock_mixture_variance(day["Z"][:n15], day["dt"][:n15], a, decay, ones)
    else:
        lam = 2.0 ** (-1.0 / (half * rate))
        var = trade_mixture_variance(day["Z"], day["dt"], a, lam, day["level"])
        early = trade_mixture_variance(day["Z"][:n15], day["dt"][:n15], a, lam, ones)
    return blend_open(day["tau"], var, day["background"], early=early)


def _objective(alpha, prepared, mode: str, rate: float) -> float:
    if float(np.sum(np.maximum(alpha, 0.0))) <= 0.0:
        return 1e30
    return float(sum(qlike(d["tau"], _mixture_variances(alpha, d, mode, rate), d["target"]) for d in prepared))


def _simplex(step: float = 0.25) -> np.ndarray:
    n = int(round(1.0 / step))
    rows = []
    for i in range(n + 1):
        for j in range(n - i + 1):
            for k in range(n - i - j + 1):
                rows.append((i, j, k, n - i - j - k))
    return np.asarray(rows, dtype=np.float64) / n


def fit_mixture(prepared, mode: str, rate: float):
    """Mixture weights of the four components by QLIKE: a simplex grid, then Nelder–Mead."""
    best_a = np.full(4, 0.25)
    best_v = np.inf
    for row in _simplex(0.25):
        value = _objective(row, prepared, mode, rate)
        if value < best_v:
            best_v, best_a = value, row.copy()
    res = minimize(lambda x: _objective(x, prepared, mode, rate), best_a, method="Nelder-Mead",
                   options={"maxiter": 40, "xatol": 1e-2, "fatol": abs(best_v) * 1e-4})
    alpha = np.maximum(res.x, 0.0)
    alpha = alpha / alpha.sum()
    return alpha, float(_objective(alpha, prepared, mode, rate))


def hepx_paths(prepared, alpha, mode: str, rate: float):
    """Superposition spot paths on the one-second grid and their log correlation
    with the hyperbolic paths (``day["hub"]``)."""
    rows = []
    store = np.zeros(6)
    for day in prepared:
        var = _mixture_variances(alpha, day, mode, rate)
        a, b = var, day["hub"]
        m = np.isfinite(a) & np.isfinite(b) & (a > 0.0) & (b > 0.0)
        la, lb = np.log(a[m]), np.log(b[m])
        store += np.array([la.size, la.sum(), lb.sum(), la @ la, lb @ lb, la @ lb])
        rows.append(onto_seconds(day["tau"], annualized_percent(var)))
    n, sx, sy, sxx, syy, sxy = store
    corr = float((sxy - sx * sy / n) / np.sqrt((sxx - sx * sx / n) * (syy - sy * sy / n)))
    return np.vstack(rows), corr


# ---------------------------------------------------------------------------
# Appendix A.2–A.3: s = 10 seconds
# ---------------------------------------------------------------------------

def background_s(ticker: str, date: str, s: float) -> float:
    """``σ̃²`` at lag ``s``: median first-minute variance of the previous five sessions."""
    vals = []
    for prev in previous(date, PRESAMPLE_OPENINGS):
        with np.load(cache_path(ticker, prev)) as f:
            tau = np.asarray(f["tau"], dtype=np.float64)
            theta = np.asarray(f["theta"], dtype=np.float64)
        z, _ = increments(tau, theta, s=s, c=damping_correction(tau, s))
        m = tau <= STARTUP_S
        if m.any():
            vals.append(float(z[m].sum() / STARTUP_S))
    vals = [v for v in vals if np.isfinite(v) and v > 0.0]
    return float(np.median(vals)) if vals else float("nan")


def appendix_a2_a3(ticker: str, a_hat: float, s_long: float = LONG_LAG_S) -> dict:
    """The spot volatility of Figure 11 rebuilt from ``s = 10`` second returns,
    next to ``s = 1``.

    Same ``a``, the same diurnal curve (fit to the 1-second Huber returns) and
    each lag's own damping factor and opening background. A ratio "at the
    typical trade" is the Table B.2 one: the median of one path over the
    median of the other, every 300th trade of the twelve days. The median of
    the trade-by-trade ratio is kept as well.
    """
    labels = ("huber_1", "linear_1", "huber_10", "linear_10")
    samples = {k: [] for k in labels}
    seconds = {k: [] for k in labels}
    jumps = {}
    for date in PAPER_DATES:
        with np.load(cache_path(ticker, date)) as f:
            tau = np.asarray(f["tau"], dtype=np.float64)
            theta = np.asarray(f["theta"], dtype=np.float64)
            linear = np.asarray(f["linear"], dtype=np.float64)
        coef, bg1 = coef_and_background(ticker, date)
        level = diurnal_variance(tau, coef)
        bg_long = background_s(ticker, date, s_long)
        idx = np.arange(SPOT_SAMPLE_STEP - 1, len(tau), SPOT_SAMPLE_STEP)
        for lab, x, s, bg in (("huber_1", theta, SPOT_LAG_S, bg1), ("linear_1", linear, SPOT_LAG_S, bg1),
                              ("huber_10", theta, s_long, bg_long), ("linear_10", linear, s_long, bg_long)):
            z, dt = increments(tau, x, s=s, c=damping_correction(tau, s))
            ann = annualized_percent(diurnal_hyperbolic_spot(tau, z, dt, level, bg, a_hat))
            samples[lab].append(ann[idx])
            seconds[lab].append(onto_seconds(tau, ann))
            if date == OCT16 and lab.startswith("linear"):
                jumps[lab] = jump(tau, ann, jump_trade(ticker))
    s_ = {k: np.concatenate(v) for k, v in samples.items()}

    def ratios(a, b):
        m = np.isfinite(s_[a]) & np.isfinite(s_[b]) & (s_[a] > 0.0) & (s_[b] > 0.0)
        return {"of_medians": float(np.median(s_[a][m]) / np.median(s_[b][m])),
                "median_pointwise": float(np.median(s_[a][m] / s_[b][m]))}

    return {
        "a_trades": float(a_hat),
        "s_long": float(s_long),
        "linear_over_huber_s1": ratios("linear_1", "huber_1"),
        "linear_over_huber_s10": ratios("linear_10", "huber_10"),
        "huber_s10_over_s1": ratios("huber_10", "huber_1"),
        "jump_16oct_s1": jumps["linear_1"],
        "jump_16oct_s10": jumps["linear_10"],
        "seconds": {k: np.vstack(v) for k, v in seconds.items()},
    }


# ---------------------------------------------------------------------------
# Sensitivity of the linear/Huber ratio and the linear jump (exp_4 Module G)
# ---------------------------------------------------------------------------

def linear_over_huber(days, a: float, linear_level: str = "level") -> dict:
    """The Table B.2 ratio at a given ``a``: median linear spot volatility over
    the median Huber one, every ``SPOT_SAMPLE_STEP``-th trade, pooled over the
    days, plus the same ratio day by day. ``linear_level`` names the diurnal
    level used for the linear path (``"level"`` is the Huber-path fit)."""
    lin_s, hub_s, daily = [], [], []
    for d in days:
        hub = diurnal_hyperbolic_spot(d["tau"], d["Z"], d["dt"], d["level"], d["background"], a)
        lin = diurnal_hyperbolic_spot(d["tau"], d["Z_lin"], d["dt"], d[linear_level], d["background"], a)
        idx = np.arange(SPOT_SAMPLE_STEP - 1, len(lin), SPOT_SAMPLE_STEP)
        ok = np.isfinite(lin[idx]) & np.isfinite(hub[idx]) & (lin[idx] > 0) & (hub[idx] > 0)
        lv, hv = np.sqrt(lin[idx][ok]), np.sqrt(hub[idx][ok])
        lin_s.append(lv)
        hub_s.append(hv)
        daily.append(float(np.median(lv) / np.median(hv)))
    return {"pooled": float(np.median(np.concatenate(lin_s)) / np.median(np.concatenate(hub_s))),
            "daily": daily, "median_of_daily": float(np.median(daily)), "mean_of_daily": float(np.mean(daily))}


def add_linear_diurnal(ticker: str, days) -> None:
    """Add ``level_linear`` to each prepared day: the diurnal curve fit, as for
    the Huber path, to the linear path's own 10-second uniform estimates."""
    grids = {}
    for date in ALL_DATES:
        with np.load(cache_path(ticker, date)) as f:
            tau = np.asarray(f["tau"], dtype=np.float64)
            linear = np.asarray(f["linear"], dtype=np.float64)
        z, _ = increments(tau, linear, c=damping_correction(tau, SPOT_LAG_S))
        grids[date] = uniform_spot_grid(tau, z)
    for d in days:
        g = np.vstack([grids[p] for p in previous(d["date"], DIURNAL_SESSIONS)])
        count = np.sum(np.isfinite(g), axis=0)
        mean = np.full(g.shape[1], np.nan)
        ok = count > 0
        mean[ok] = np.nansum(g[:, ok], axis=0) / count[ok]
        d["level_linear"] = np.ascontiguousarray(diurnal_variance(d["tau"], fit_diurnal(mean)))


def linear_jump_at(ticker: str, days, a: float) -> dict:
    """The Table B.2 linear jump on 16 October at a given ``a``."""
    d = next(x for x in days if x["date"] == OCT16)
    lin = diurnal_hyperbolic_spot(d["tau"], d["Z_lin"], d["dt"], d["level"], d["background"], a)
    return jump(d["tau"], annualized_percent(lin), jump_trade(ticker))


def presample_from_cache(ticker: str, date: str) -> np.ndarray:
    """Opening residuals of the ``PRESAMPLE_OPENINGS`` sessions before ``date``, as stored by ``ensure_filtered``."""
    parts = []
    for prev in previous(date, PRESAMPLE_OPENINGS):
        with np.load(cache_path(ticker, prev)) as f:
            parts.append(np.asarray(f["open_resid"], dtype=np.float64))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)
