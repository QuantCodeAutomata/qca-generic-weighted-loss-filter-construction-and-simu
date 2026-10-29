"""Spot volatility of Section 4.5.2–4.6.

``Z_t`` is the pre-averaged variance increment. A weighted mean of ``Z``
divided by the same mean of the clock gaps is a variance per second. The
diurnal curve is a Fourier fit to the previous ten days' 10-second estimates,
and the open is blended towards the previous five first minutes. The
Mandelbrot parameter ``a`` of the hyperbolic weights is chosen per asset by
QLIKE (``select_mandelbrot_a``); ``A_TRADES`` is the paper's NVDA value.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from .config import (
    DIURNAL_FOURIER_PAIRS,
    HEXP_HALF_LIVES_S,
    MANDELBROT_A_GRID_POINTS,
    MANDELBROT_A_MAX,
    MANDELBROT_A_MIN,
    MANDELBROT_A_REFINE_STEPS,
    NVDA_A_PUBLISHED,
    OPEN_BLEND_S,
    QLIKE_HORIZON_S,
    QLIKE_START_S,
    SESSION_SECONDS,
    UNIFORM_SPOT_WINDOW_S,
)

A_TRADES = NVDA_A_PUBLISHED


def increments(tau: np.ndarray, x: np.ndarray, s: float = 1.0, c: np.ndarray | None = None):
    """``Z_t = (τ_t − τ_{t−1}) c_{t,s} r_{t,s}^2 / (τ_t − τ_{t,s})``.

    The reference is the opening print when no trade is at least ``s`` seconds
    earlier, and trades in the first ``s`` seconds after the open contribute
    nothing (Section 4.6 uses the opening-print return only once it is one
    second old). ``tau`` is in seconds after the open.
    """
    tau = np.asarray(tau, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = len(tau)
    if c is None:
        c = np.ones(n, dtype=np.float64)
    else:
        c = np.asarray(c, dtype=np.float64)
    idx = np.maximum(np.searchsorted(tau, tau - s, side="right") - 1, 0)
    lag = np.maximum(tau - tau[idx], 1e-12)
    dt = np.empty(n, dtype=np.float64)
    dt[0] = 0.0
    if n > 1:
        dt[1:] = np.diff(tau)
    z = dt * c * (x - x[idx]) ** 2 / lag
    z[tau < s] = 0.0
    z[0] = 0.0
    return z, dt


def uniform_spot_grid(
    tau: np.ndarray,
    z: np.ndarray,
    h: float = UNIFORM_SPOT_WINDOW_S,
    session: float = SESSION_SECONDS,
) -> np.ndarray:
    """``(1/h)`` times the sum of ``Z`` over ``(u−h, u]``, on integer seconds."""
    tau = np.asarray(tau, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    u = np.arange(0.0, session + 1.0, 1.0)
    prefix = np.cumsum(z)
    hi = np.searchsorted(tau, u, side="right") - 1
    lo = np.searchsorted(tau, u - h, side="right") - 1
    total = np.zeros(len(u), dtype=np.float64)
    valid = hi >= 0
    total[valid] = prefix[hi[valid]]
    has_lo = lo >= 0
    total[has_lo] -= prefix[lo[has_lo]]
    out = total / h
    out[hi <= lo] = np.nan
    return out


def fourier_design(u: np.ndarray, n_pairs: int = DIURNAL_FOURIER_PAIRS, session: float = SESSION_SECONDS) -> np.ndarray:
    """Intercept, linear, quadratic, and ``n_pairs`` sine-cosine pairs."""
    t = np.asarray(u, dtype=np.float64) / session
    cols = [np.ones_like(t), t, t * t]
    ang = 2.0 * np.pi * t
    for p in range(1, n_pairs + 1):
        cols.append(np.cos(p * ang))
        cols.append(np.sin(p * ang))
    return np.column_stack(cols)


def fit_diurnal(mean_var: np.ndarray, min_u: float = UNIFORM_SPOT_WINDOW_S) -> np.ndarray:
    """OLS of ``log`` of the 10-day average variance on the Fourier design."""
    u = np.arange(len(mean_var), dtype=np.float64)
    var = np.asarray(mean_var, dtype=np.float64)
    y = np.full(len(var), np.nan, dtype=np.float64)
    ok = var > 0.0
    y[ok] = np.log(var[ok])
    mask = (u >= min_u) & np.isfinite(y)
    coef, *_ = np.linalg.lstsq(fourier_design(u[mask]), y[mask], rcond=None)
    return coef


def diurnal_variance(tau: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """``exp(g(τ))``, the fitted variance level."""
    return np.exp(fourier_design(tau) @ np.asarray(coef, dtype=np.float64))


def _conv(x: np.ndarray, a: float) -> np.ndarray:
    """``sum_{j=0}^{t} x_{t−j} / (j+a)`` by one FFT."""
    n = len(x)
    kernel = 1.0 / (np.arange(n, dtype=np.float64) + a)
    size = 1 << int(math.ceil(math.log2(2 * n - 1)))
    prod = np.fft.rfft(np.ascontiguousarray(x), size) * np.fft.rfft(kernel, size)
    return np.fft.irfft(prod, size)[:n]


def hyperbolic_spot(z: np.ndarray, dt: np.ndarray, level: np.ndarray, a: float = A_TRADES) -> np.ndarray:
    """Hyperbolic trade-time weights. The normaliser cancels in the ratio."""
    level = np.asarray(level, dtype=np.float64)
    safe = np.maximum(level, 1e-300)
    num = _conv(np.asarray(z, dtype=np.float64) / safe, a)
    den = _conv(np.asarray(dt, dtype=np.float64), a)
    out = np.full(len(z), np.nan, dtype=np.float64)
    ok = (den > 0.0) & (num >= 0.0)
    out[ok] = level[ok] * num[ok] / den[ok]
    return out


@njit(cache=True)
def clock_mixture_variance(z, dt, alpha, decay, level):
    """Clock-time exponential components. ``decay`` is ``log(2) / half-life``."""
    n = z.shape[0]
    p = alpha.shape[0]
    sz = np.zeros(p)
    sd = np.zeros(p)
    s1 = np.zeros(p)
    out = np.empty(n)
    for t in range(n):
        lt = level[t]
        if lt <= 0.0:
            lt = 1e-300
        zt = z[t] / lt
        dtp = dt[t]
        num = 0.0
        den = 0.0
        for k in range(p):
            f = math.exp(-decay[k] * dtp)
            sz[k] = f * sz[k] + zt
            sd[k] = f * sd[k] + dtp
            s1[k] = f * s1[k] + 1.0
            num += alpha[k] * sz[k] / s1[k]
            den += alpha[k] * sd[k] / s1[k]
        out[t] = lt * num / den if den > 0.0 else np.nan
    return out


@njit(cache=True)
def trade_mixture_variance(z, dt, alpha, lam, level):
    """Trade-lag exponential components, ``lam[k]`` per trade."""
    n = z.shape[0]
    p = alpha.shape[0]
    sz = np.zeros(p)
    sd = np.zeros(p)
    s1 = np.zeros(p)
    out = np.empty(n)
    for t in range(n):
        lt = level[t]
        if lt <= 0.0:
            lt = 1e-300
        zt = z[t] / lt
        dtp = dt[t]
        num = 0.0
        den = 0.0
        for k in range(p):
            f = lam[k]
            sz[k] = f * sz[k] + zt
            sd[k] = f * sd[k] + dtp
            s1[k] = f * s1[k] + 1.0
            num += alpha[k] * sz[k] / s1[k]
            den += alpha[k] * sd[k] / s1[k]
        out[t] = lt * num / den if den > 0.0 else np.nan
    return out


def blend_open(tau: np.ndarray, spot: np.ndarray, background: float, horizon: float = OPEN_BLEND_S,
               early: np.ndarray | None = None) -> np.ndarray:
    """``σ̂²(u) = (15−u)/15 σ̃² + u/15 · filter`` for ``u < 15``.

    Section 4.6 blends the filter without the diurnal adjustment, which only
    starts at ``u = 15``. ``early`` is that unadjusted filter on the trades
    with ``tau < horizon``; without it the adjusted ``spot`` is blended.
    """
    out = np.asarray(spot, dtype=np.float64).copy()
    tau = np.asarray(tau, dtype=np.float64)
    m = tau < horizon
    w = tau[m] / horizon
    base = out[m] if early is None else np.asarray(early, dtype=np.float64)[: int(m.sum())]
    base = np.where(np.isfinite(base), base, background)
    out[m] = (1.0 - w) * background + w * base
    return out


def early_unadjusted(tau: np.ndarray, z: np.ndarray, dt: np.ndarray, a: float, horizon: float = OPEN_BLEND_S) -> np.ndarray:
    """Hyperbolic filter with no diurnal level on the trades before ``horizon``."""
    n = int(np.sum(np.asarray(tau, dtype=np.float64) < horizon))
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    return hyperbolic_spot(z[:n], dt[:n], np.ones(n, dtype=np.float64), a)


def diurnal_hyperbolic_spot(tau, z, dt, level, background, a: float = A_TRADES, horizon: float = OPEN_BLEND_S):
    """The Section 4.6 spot variance: diurnal filter from ``u = 15``, blend before."""
    refined = hyperbolic_spot(z, dt, level, a)
    return blend_open(tau, refined, background, horizon, early_unadjusted(tau, z, dt, a, horizon))


def annualized_percent(var_per_second: np.ndarray, session: float = SESSION_SECONDS) -> np.ndarray:
    """Daily-scaled spot volatility in percent per year.

    A constant per-second variance ``v`` integrates to ``v * session`` over the
    day, and ``100 sqrt(252)`` of that square root is the signature's unit.
    """
    v = np.maximum(np.asarray(var_per_second, dtype=np.float64), 0.0)
    return 100.0 * np.sqrt(252.0 * session * v)


def qlike(tau: np.ndarray, spot: np.ndarray, target: np.ndarray, lo: float = QLIKE_START_S,
          hi: float = SESSION_SECONDS) -> float:
    """∫ {σ̄²(u) / σ̂²(u−60) + log σ̂²(u−60)} du on integer seconds."""
    u = np.arange(lo, hi + 1.0)
    idx = np.searchsorted(np.asarray(tau, dtype=np.float64), u - QLIKE_HORIZON_S, side="right") - 1
    forecast = np.full(len(u), np.nan)
    ok = idx >= 0
    forecast[ok] = np.asarray(spot, dtype=np.float64)[idx[ok]]
    ui = np.minimum(u.astype(np.int64), len(target) - 1)
    tgt = np.asarray(target, dtype=np.float64)[ui]
    m = np.isfinite(forecast) & np.isfinite(tgt) & (forecast > 0.0) & (tgt > 0.0)
    if not np.any(m):
        return float("inf")
    return float(np.sum(tgt[m] / forecast[m] + np.log(forecast[m])))


def qlike_total(days, a: float) -> float:
    """QLIKE of ``σ̂(u−60; a)`` against ``σ̄(u)``, summed over ``days``.

    Each day is a dict with ``tau``, ``Z``, ``dt``, ``level`` (the diurnal
    variance at each trade), ``background`` (``σ̃²``) and ``target`` (the
    10-second uniform estimator on integer seconds).
    """
    total = 0.0
    for d in days:
        var = diurnal_hyperbolic_spot(d["tau"], d["Z"], d["dt"], d["level"], d["background"], a)
        total += qlike(d["tau"], var, d["target"])
    return float(total)


def select_mandelbrot_a(days, grid=None, n_refine: int = MANDELBROT_A_REFINE_STEPS):
    """Minimise ``qlike_total`` over ``a``: a log grid (extended downwards while
    its first point is the minimum), then golden section around the best point."""
    if grid is None:
        grid = np.geomspace(MANDELBROT_A_MIN, MANDELBROT_A_MAX, MANDELBROT_A_GRID_POINTS)
    grid = np.asarray(grid, dtype=np.float64)
    path = [(float(a), qlike_total(days, float(a))) for a in grid]
    values = np.array([v for _a, v in path])
    i = int(np.argmin(values))
    # A minimum on the first grid point is extended downwards (slow assets
    # can prefer fewer than ten trades of memory) until it is interior.
    while i == 0 and grid[0] > 1.0:
        lower = max(grid[0] / (grid[1] / grid[0]), 1.0)
        grid = np.concatenate([[lower], grid])
        path.insert(0, (float(lower), qlike_total(days, float(lower))))
        values = np.array([v for _a, v in path])
        i = int(np.argmin(values))
    lo = np.log(grid[max(i - 1, 0)])
    hi = np.log(grid[min(i + 1, len(grid) - 1)])
    g = (np.sqrt(5.0) - 1.0) / 2.0
    c = hi - g * (hi - lo)
    d = lo + g * (hi - lo)
    fc = qlike_total(days, float(np.exp(c)))
    fd = qlike_total(days, float(np.exp(d)))
    path += [(float(np.exp(c)), fc), (float(np.exp(d)), fd)]
    for _ in range(n_refine):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - g * (hi - lo)
            fc = qlike_total(days, float(np.exp(c)))
            path.append((float(np.exp(c)), fc))
        else:
            lo, c, fc = c, d, fd
            d = lo + g * (hi - lo)
            fd = qlike_total(days, float(np.exp(d)))
            path.append((float(np.exp(d)), fd))
    best = min(path, key=lambda r: r[1])
    return best[0], sorted(path)


def onto_seconds(tau: np.ndarray, values: np.ndarray, session: float = SESSION_SECONDS) -> np.ndarray:
    """Last value at or before each integer second."""
    u = np.arange(0.0, session + 1.0, 1.0)
    idx = np.searchsorted(np.asarray(tau, dtype=np.float64), u, side="right") - 1
    out = np.full(len(u), np.nan, dtype=np.float64)
    ok = idx >= 0
    out[ok] = np.asarray(values, dtype=np.float64)[idx[ok]]
    return out
