"""Five-filter pipeline from Section 4 of the paper.

Stages, all on log-prices:

1. Exponential median, half-life 3 trades, head ``W^{-1}(0.999)``.
2. Clock-time median of nonzero absolute residuals, half-life 4 minutes.
   The first minute blends the previous five openings (Section 4.6).
3. Huber filter, threshold ``delta = 2 * 1.4826 * b``, IRLS to 1e-12,
   each lag carrying ``delta`` of the previous trade.
4. Exact exponential mean with the same half-life (the linear benchmark).
5. Outlier indicators and the 13-second directional probability ``q_t``.

The median and Huber filters use the stratified estimator of Remark 4(d):
the head ``h = W_t^{-1}(0.999)`` (lags 0 to 29 at a 3-trade half-life) is exact
and ``B = 100`` lags beyond it are drawn with probability proportional to
their weights. Each Huber tail draw carries its own trade's threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .config import (
    OUTLIER_HALF_LIFE_S,
    PRICE_HALF_LIFE_TRADES,
    REGULAR_OPEN_SECONDS,
    SCALE_HALF_LIFE_S,
    STARTUP_S,
    STRATIFIED_HEAD_MASS,
    TAIL_DRAWS,
)
from .hf_filters import (
    clock_time_scale,
    directional_probability,
    exponential_huber,
    exponential_mean,
    exponential_median,
    threshold_from_scale,
)



@dataclass
class PipelineResult:
    tau: np.ndarray
    Y: np.ndarray
    m: np.ndarray
    resid_med: np.ndarray
    b: np.ndarray
    delta: np.ndarray
    theta: np.ndarray
    resid_hub: np.ndarray
    linear: np.ndarray
    T60: int
    delta_prev: np.ndarray
    d_pos: np.ndarray
    d_neg: np.ndarray
    q: np.ndarray
    meta: Dict = field(default_factory=dict)


def linear_exponential_filter(Y: np.ndarray, halflife: int = 3) -> np.ndarray:
    """Finite-sample exponential mean. Endpoints stay at the observations."""
    return exponential_mean(np.asarray(Y, dtype=np.float64), int(halflife))


def _tau_open_from(tau: np.ndarray) -> np.ndarray:
    tau = np.asarray(tau, dtype=np.float64)
    if len(tau) and float(tau[0]) > 30_000.0:
        return tau - REGULAR_OPEN_SECONDS
    return tau.copy()


def run_pipeline(
    tau: np.ndarray,
    Y: np.ndarray,
    *,
    halflife_trades: int = PRICE_HALF_LIFE_TRADES,
    B: int = TAIL_DRAWS,
    seed: int = 0,
    T60: Optional[int] = None,
    presample_abs: Optional[np.ndarray] = None,
    tau_open: Optional[np.ndarray] = None,
    scale_halflife_s: float = SCALE_HALF_LIFE_S,
    startup: str = "clock",
    median_tie: str = "exact",
) -> PipelineResult:
    """Run the price filters on one day of log-prices.

    ``B`` is the number of stratified tail draws per trade for the median and
    Huber filters (``B = 0`` keeps the exact head only). ``seed`` feeds
    ``np.random.default_rng``; the median and Huber filters get independent
    streams. ``startup="count"`` swaps the presample out by trade count
    instead of by the clock (Appendix A.5). ``median_tie="rounding"`` leaves a
    tie at one half in the median filter to floating-point rounding instead of
    breaking it towards the newest trade.
    """
    tau = np.asarray(tau, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if tau_open is None:
        tau_open = _tau_open_from(tau)
    else:
        tau_open = np.asarray(tau_open, dtype=np.float64)
    if T60 is None:
        T60 = int((tau_open <= STARTUP_S).sum())

    seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, size=2)
    m = exponential_median(Y, halflife_trades, B=B, seed=int(seeds[0]), head_mass=STRATIFIED_HEAD_MASS,
                           exact_tie=(median_tie == "exact"))
    resid_med = Y - m
    pre = (
        np.zeros(0, dtype=np.float64)
        if presample_abs is None
        else np.asarray(presample_abs, dtype=np.float64)
    )
    pre = pre[np.isfinite(pre) & (pre > 0.0)]
    # The order-statistic tree stores unnormalised weights exp(ln2 * tau / h).
    if np.log(2.0) / scale_halflife_s * float(np.max(tau_open)) > 700.0:
        raise ValueError("scale half-life too short for the session length (float overflow)")
    b = clock_time_scale(
        tau_open,
        np.abs(resid_med),
        pre,
        halflife_s=float(scale_halflife_s),
        startup_s=STARTUP_S,
        by_count=(startup == "count"),
        T60=int(T60),
    )
    delta = threshold_from_scale(b)
    # delta_0 is the presample-only scale (tau_0 = 0 in Section 4.6).
    if pre.size:
        b0 = float(np.sort(pre)[(pre.size + 1) // 2 - 1])
        delta0 = float(threshold_from_scale(np.array([b0]))[0])
    else:
        delta0 = float(delta[0])
    theta = exponential_huber(Y, delta, m, halflife_trades, B=B, seed=int(seeds[1]), delta0=delta0,
                              head_mass=STRATIFIED_HEAD_MASS)
    linear = exponential_mean(Y, halflife_trades)

    resid_hub = Y - theta
    delta_prev = np.empty_like(delta)
    delta_prev[0] = delta0
    if len(delta) > 1:
        delta_prev[1:] = delta[:-1]
    d_pos = (resid_hub > delta_prev).astype(np.float64)
    d_neg = (resid_hub < -delta_prev).astype(np.float64)
    q = directional_probability(
        tau_open,
        d_pos,
        d_neg,
        OUTLIER_HALF_LIFE_S,
        int(T60),
    )
    return PipelineResult(
        tau=tau,
        Y=Y,
        m=m,
        resid_med=resid_med,
        b=b,
        delta=delta,
        theta=theta,
        resid_hub=resid_hub,
        linear=linear,
        T60=int(T60),
        delta_prev=delta_prev,
        d_pos=d_pos,
        d_neg=d_neg,
        q=q,
        meta={
            "halflife_trades": int(halflife_trades),
            "scale_halflife_s": float(scale_halflife_s),
            "T60": int(T60),
            "n_presample": int(pre.size),
        },
    )


def opening_abs_residuals(
    log_price: np.ndarray,
    tau_open: np.ndarray,
    halflife: int = PRICE_HALF_LIFE_TRADES,
    *,
    minute_s: float = STARTUP_S,
    from_open_print: bool = False,
    B: int = 0,
    seed: int = 0,
) -> np.ndarray:
    """Nonzero absolute median-filter residuals from the first minute.

    The median filter is causal, but its last point is pinned to the price
    (``m_T = Y_T``), so it runs one trade past the minute and drops that trade.
    It keeps the exact head only (``B = 0``); the 0.1% tail mass leaves the
    presample median, and so ``delta_0``, unchanged on the paper's dates.
    ``minute_s``, ``from_open_print`` (count the minute from the opening print
    instead of 09:30) and ``B``/``seed`` vary that choice in the checks.
    """
    log_price = np.asarray(log_price, dtype=np.float64)
    tau_open = np.asarray(tau_open, dtype=np.float64)
    origin = float(tau_open[0]) if (from_open_print and len(tau_open)) else 0.0
    n = int(np.sum(tau_open - origin <= minute_s))
    if n < 3:
        return np.zeros(0, dtype=np.float64)
    y = log_price[: n + 1]
    r = np.abs(y - exponential_median(y, halflife, B=B, seed=seed))[:n]
    return r[r > 0.0]
