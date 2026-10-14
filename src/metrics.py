"""Loss, tracking-error and diagnostic metrics used across experiments."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


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
    k_frac_low: float = 0.001,
    k_frac_high: float = 0.01,
    *,
    frac_lo: float | None = None,
    frac_hi: float | None = None,
) -> float:
    """Hill tail index averaged over threshold fractions in ``[k_low, k_high]``.

    The paper averages Hill estimates over the largest 0.1% to 1% of
    observations.
    """
    if frac_lo is not None:
        k_frac_low = frac_lo
    if frac_hi is not None:
        k_frac_high = frac_hi
    x = np.abs(np.asarray(x, dtype=np.float64))
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) < 100:
        return float("nan")
    x_sorted = np.sort(x)[::-1]
    n = len(x_sorted)
    k_lo = max(2, int(np.floor(n * k_frac_low)))
    k_hi = max(k_lo + 1, int(np.ceil(n * k_frac_high)))
    k_hi = min(k_hi, n - 1)
    vals = []
    for k in range(k_lo, k_hi + 1):
        logs = np.log(x_sorted[:k]) - np.log(x_sorted[k])
        gamma = logs.mean()
        if gamma > 0:
            vals.append(1.0 / gamma)
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def sample_acf(x: np.ndarray, lags: Tuple[int, ...] = (1, 2, 5, 10, 25)) -> np.ndarray:
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


def k_trade_diff_stats(x: np.ndarray, ks: Tuple[int, ...] = (1, 2, 5, 10, 25)) -> dict:
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
    tau: np.ndarray, x: np.ndarray, s_list=(0.1, 1, 5, 10, 60, 900)
) -> dict:
    """Table-2-style statistics for s-second differences of a filtered path.

    For each lag ``s`` and each index ``t``, locate the most recent observation
    at least ``s`` seconds before ``tau[t]``. If none exists, use ``x[0]``.
    """
    tau = np.asarray(tau, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    T = len(x)
    out = {}
    for s in s_list:
        # idx_prev[t] is the largest i such that tau[i] <= tau[t] - s; else -1
        idx_prev = np.searchsorted(tau, tau - float(s), side="right") - 1
        valid = idx_prev >= 0
        x_prev = np.where(valid, x[np.clip(idx_prev, 0, T - 1)], x[0])
        diffs = x - x_prev
        dt = np.where(valid, tau - tau[np.clip(idx_prev, 0, T - 1)], np.nan)
        # average dt where defined
        mean_dt = float(np.nanmean(dt)) if np.isfinite(dt).any() else float("nan")
        mad = float(np.mean(np.abs(diffs))) / np.sqrt(max(mean_dt, 1e-12))
        ann_vol = annualized_vol_from_rets(diffs[valid], mean_dt) if np.isfinite(mean_dt) else float("nan")
        hill = hill_tail_index(diffs[valid])
        # adjacent-return correlation Corr(x_t - x_{t,s}, x_{t,s} - x_{t,2s})
        idx_prev2 = np.searchsorted(tau, tau - 2.0 * float(s), side="right") - 1
        valid2 = idx_prev2 >= 0
        both = valid & valid2
        if both.sum() > 5:
            r1 = x[both] - x[np.clip(idx_prev[both], 0, T - 1)]
            r2 = x[np.clip(idx_prev[both], 0, T - 1)] - x[np.clip(idx_prev2[both], 0, T - 1)]
            if r1.std() > 0 and r2.std() > 0:
                corr = float(np.corrcoef(r1, r2)[0, 1])
            else:
                corr = float("nan")
        else:
            corr = float("nan")
        out[float(s)] = {
            "mad_scaled": mad,
            "annualized_vol": ann_vol,
            "hill": hill,
            "adj_return_corr": corr,
            "mean_dt": mean_dt,
            "n_valid": int(valid.sum()),
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
