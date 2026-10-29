"""Loss, tracking-error and diagnostic metrics used across experiments."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import HILL_K_FRACTION_HIGH, HILL_K_FRACTION_LOW, TABLE1_K, TABLE2_S


def predictive_absolute_loss(Y: np.ndarray, theta: np.ndarray) -> float:
    """``(1/(T-1)) sum_{t=2}^T (|Y_t - theta_{t-1}| - |Y_t|)``."""
    Y = np.asarray(Y)
    theta = np.asarray(theta)
    T = len(Y)
    return float(np.mean(np.abs(Y[1:] - theta[: T - 1]) - np.abs(Y[1:])))


def oracle_absolute_loss(Y: np.ndarray, mu: np.ndarray) -> float:
    """Oracle benchmark using the true latent signal."""
    Y = np.asarray(Y)
    mu = np.asarray(mu)
    return float(np.mean(np.abs(Y[1:] - mu[1:]) - np.abs(Y[1:])))


def tracking_error(mu: np.ndarray, theta: np.ndarray) -> float:
    """``(1/(T-2)) sum_{t=3}^T |mu_t - theta_t|``."""
    mu = np.asarray(mu)
    theta = np.asarray(theta)
    return float(np.mean(np.abs(mu[2:] - theta[2:])))


def mean_abs_filter_gap(theta_a: np.ndarray, theta_b: np.ndarray) -> float:
    """Average ``|theta_t - theta_t^*|`` over the entire sample."""
    return float(np.mean(np.abs(np.asarray(theta_a) - np.asarray(theta_b))))


def hill_tail_index(
    x: np.ndarray,
    k_frac_low: float = HILL_K_FRACTION_LOW,
    k_frac_high: float = HILL_K_FRACTION_HIGH,
    *,
    frac_lo: float | None = None,
    frac_hi: float | None = None,
    average: str = "log",
) -> float:
    """Hill tail index over the largest 0.1% to 1% of observations.

    The paper says only that the Hill estimates are averaged over this range.
    ``average="log"`` (the default) weights them uniformly in log(k), the
    measure of a Hill plot; ``average="arithmetic"`` weights every k equally.
    Zeros are kept when counting the sample, because the percentage is of
    every observation.
    """
    if frac_lo is not None:
        k_frac_low = frac_lo
    if frac_hi is not None:
        k_frac_high = frac_hi
    x = np.abs(np.asarray(x, dtype=np.float64))
    x = x[np.isfinite(x)]
    # Zeros stay in the sample: they set n, so "0.1% to 1%" is a fraction of
    # every observation. They sit at the bottom of the order statistics and
    # do not enter the logs below.
    if len(x) < 100:
        return float("nan")
    x_sorted = np.sort(x)[::-1]
    n = len(x_sorted)
    k_lo = max(2, int(np.floor(n * k_frac_low)))
    k_hi = max(k_lo + 1, int(np.ceil(n * k_frac_high)))
    k_hi = min(k_hi, n - 1)
    while k_hi > k_lo and x_sorted[k_hi] <= 0.0:
        k_hi -= 1
    if x_sorted[k_lo] <= 0.0:
        return float("nan")
    logx = np.log(x_sorted[: k_hi + 1])
    csum = np.cumsum(logx)
    ks = np.arange(k_lo, k_hi + 1)
    gamma = csum[ks - 1] / ks - logx[ks]
    good = gamma > 0.0
    if not np.any(good):
        return float("nan")
    # Uniform average in log k: a Hill plot is read on a log threshold axis.
    w = 1.0 / ks[good] if average == "log" else np.ones(int(good.sum()))
    return float(np.sum(w / gamma[good]) / np.sum(w))


def sample_acf(x: np.ndarray, lags: Tuple[int, ...] = TABLE1_K) -> np.ndarray:
    """Sample autocorrelation at the supplied lags."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    out = np.empty(len(lags), dtype=np.float64)
    for i, h in enumerate(lags):
        if h >= len(x) or denom == 0.0:
            out[i] = float("nan")
        else:
            num = float(np.dot(x[h:], x[:-h]))
            out[i] = num / denom
    return out


def huber_score(residual: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Huber score ``psi(e) = clip(e, -delta, delta)``."""
    r = np.asarray(residual, dtype=np.float64)
    d = np.asarray(delta, dtype=np.float64)
    return np.clip(r, -d, d)


# Aliases for compatibility with experiment scripts
predictive_abs_loss = predictive_absolute_loss
tracking_error_abs = tracking_error
acf = sample_acf


def annualized_vol_from_rets(rets: np.ndarray, dt_seconds: float) -> float:
    """Annualise realised volatility assuming ``dt_seconds`` between returns.

    Uses 252 trading days * 6.5 hours * 3600 = 5,896,800 seconds per year.
    """
    rets = np.asarray(rets, dtype=np.float64)
    rets = rets[np.isfinite(rets)]
    if len(rets) < 2 or dt_seconds <= 0:
        return float("nan")
    var_per_sec = float(np.mean(rets * rets)) / dt_seconds
    return float(np.sqrt(var_per_sec * 252.0 * 6.5 * 3600.0))


def k_trade_diff_stats(x: np.ndarray, ks: Tuple[int, ...] = TABLE1_K) -> dict:
    """Table-1-style statistics for k-trade differences of a filtered path."""
    x = np.asarray(x, dtype=np.float64)
    out = {}
    for k in ks:
        if k >= len(x):
            out[int(k)] = {"mad_scaled": float("nan"),
                           "rms_scaled": float("nan"),
                           "acf1": float("nan"),
                           "hill": float("nan")}
            continue
        diffs = x[k:] - x[:-k]
        mad = float(np.mean(np.abs(diffs))) / np.sqrt(k)
        rms = float(np.sqrt(np.mean(diffs * diffs))) / np.sqrt(k)
        acf1 = float(sample_acf(np.diff(x), lags=(k,))[0]) if len(x) > k + 1 else float("nan")
        hill = hill_tail_index(diffs)
        out[int(k)] = {"mad_scaled": mad, "rms_scaled": rms, "acf1": acf1, "hill": hill}
    return out


def s_second_diff_stats(
    tau: np.ndarray,
    x: np.ndarray,
    s_list=TABLE2_S,
    c_by_s: Optional[dict] = None,
) -> dict:
    """Table 2 statistics for one log-price path.

    ``x`` is the log of the price (raw or filtered). The level panel is the
    mean of ``|exp(x_t) - exp(x_{t,s})| / sqrt(tau_t - tau_{t,s})`` in dollars.
    The volatility panel is equation (1) of the paper, in percent per year.
    When no trade is at least ``s`` seconds earlier the reference is the
    opening print. ``c_by_s[s]`` is the finite-sample factor; raw prices use 1.
    """
    tau = np.asarray(tau, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    price = np.exp(x)
    T = len(x)
    out = {}
    for s in s_list:
        s = float(s)
        idx_prev = np.searchsorted(tau, tau - s, side="right") - 1
        valid = idx_prev >= 0
        idx_use = np.where(valid, idx_prev, 0)
        log_diff = x - x[idx_use]
        px_diff = price - price[idx_use]
        lag = tau - tau[idx_use]
        lag = np.maximum(lag, 1e-12)
        # Opening reference is included, matching Table 2.
        mad = float(np.mean(np.abs(px_diff) / np.sqrt(lag)))
        dt = np.diff(tau, prepend=tau[0])
        dt[0] = 0.0
        dt = np.maximum(dt, 0.0)
        if c_by_s is not None and s in c_by_s:
            c = np.asarray(c_by_s[s], dtype=np.float64)
        elif c_by_s is not None and float(s) in c_by_s:
            c = np.asarray(c_by_s[float(s)], dtype=np.float64)
        else:
            c = 1.0
        incr = dt * c * (log_diff ** 2) / lag
        qv = float(np.sum(incr[1:]))
        # Equation (1) without its sqrt(23,400 / (23,400 - tau_1)) factor, about
        # 1.00002 for tau_1 near one second; Section 4.5.1 drops it as well.
        ann_vol = float(100.0 * np.sqrt(252.0) * np.sqrt(max(qv, 0.0)))
        hill = hill_tail_index(px_diff)
        # The opening-print fallback applies to X_{t,2s} as well, so every t
        # enters (dropping t without a trade 2s earlier moves the 1 min
        # correlations from 0.06 to 0.03 on 16 Oct).
        idx_use2 = np.maximum(np.searchsorted(tau, tau - 2.0 * s, side="right") - 1, 0)
        r1 = px_diff
        r2 = price[idx_use] - price[idx_use2]
        if T > 5 and r1.std() > 0 and r2.std() > 0:
            corr = float(np.corrcoef(r1, r2)[0, 1])
        else:
            corr = float("nan")
        finite_lag = lag[valid] if valid.any() else lag
        out[s] = {
            "mad_scaled": mad,
            "annualized_vol": ann_vol,
            "hill": hill,
            "adj_return_corr": corr,
            "mean_dt": float(np.mean(finite_lag)) if len(finite_lag) else float("nan"),
            "n_valid": int(T),
        }
    return out


def outlier_shares(residual: np.ndarray, delta_prev: np.ndarray) -> dict:
    """Return positive, negative and total outlier shares."""
    r = np.asarray(residual, dtype=np.float64)
    d = np.asarray(delta_prev, dtype=np.float64)
    pos = r > d
    neg = r < -d
    tot = len(r)
    return {
        "pos_share": float(pos.mean()) if tot else float("nan"),
        "neg_share": float(neg.mean()) if tot else float("nan"),
        "total_share": float((pos | neg).mean()) if tot else float("nan"),
    }


def signature_vol(tau: np.ndarray, x: np.ndarray, s: float, c: Optional[np.ndarray] = None) -> float:
    """Annualised ``100 sqrt(252) σ̂`` of equation (1) for one lag ``s``.

    ``X̂_{t,s}`` falls back to the opening print when no trade is ``s``
    seconds earlier; ``c`` is the damping factor (1 for raw trades).
    """
    tau = np.asarray(tau, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    idx = np.maximum(np.searchsorted(tau, tau - s, side="right") - 1, 0)
    lag = np.maximum(tau - tau[idx], 1e-12)
    dt = np.empty_like(tau)
    dt[0] = 0.0
    dt[1:] = np.maximum(np.diff(tau), 0.0)
    z = dt * (x - x[idx]) ** 2 / lag
    if c is not None:
        z = z * np.asarray(c, dtype=np.float64)
    return float(100.0 * np.sqrt(252.0 * max(float(z[1:].sum()), 0.0)))


def psi_star_cents(price: np.ndarray, theta: np.ndarray, delta_prev: np.ndarray,
                   halflife: int = 3, head: int = 30) -> dict:
    """Table 3's standard error ``ψ* = se(θ_t − X_t)`` of the Huber filter, cents.

    ``theta`` is the log filter and ``delta_prev`` the log threshold
    ``δ_{t−1}``. The score is the Huber ψ of the price residual with threshold
    ``δ_{t−1} p_t``; ``ψ* = sqrt(w'Γw) / E ψ'`` over the ``head`` lags of the
    exponential weights ``(1 − λ) λ^j`` (the tail mass beyond the head is
    left out), with ``E ψ'`` the share of residuals inside the band.
    """
    price = np.asarray(price, dtype=np.float64)
    e = (price - np.exp(np.asarray(theta, dtype=np.float64))) * 100.0
    d = np.asarray(delta_prev, dtype=np.float64) * price * 100.0
    psi = np.clip(e, -d, d)
    slope = float(np.mean(np.abs(e) <= d))
    lam = 2.0 ** (-1.0 / halflife)
    w = (1.0 - lam) * lam ** np.arange(head)
    z = psi - psi.mean()
    gam = np.array([np.dot(z[: len(z) - k], z[k:]) / len(z) for k in range(head)])
    G = gam[np.abs(np.subtract.outer(np.arange(head), np.arange(head)))]
    return {"psi_star_cents": float(np.sqrt(w @ G @ w) / slope), "slope": slope,
            "score_acf": gam / gam[0]}
