"""
Five-filter empirical pipeline for high-frequency trade data.

Stage 1  Median price filter m_t                  (abs-loss, 3-trade halflife)
Stage 2  Scale filter b_t and threshold delta_t   (abs-loss on |residuals|, 4-min halflife)
Stage 3  Huber price filter theta_t               (Huber loss with threshold delta_{t-j-1})
Stage 4  Benchmark paths (raw, linear, median)
Stage 5  Residual diagnostics (Tables 1/2/3 metrics)
Stage 6  Integrated volatility from Huber-filtered prices
Stage 7  Spot volatility with hyperbolic weights
Stage 8  Directional outlier probability q_t

Each filter uses the paper's stratified Monte Carlo estimator with the exact
head weights through the 99.9% cumulative-weight cutoff and B tail draws.

Design note
-----------
For exponential weights with half-life ``H`` trades we have
``lambda = 2^(-1/H)`` and the head cutoff ``h = W_t^{-1}(0.999)`` is a
small constant (around 29 for H=3), so the per-trade cost is O(h + B) and
the whole-day pass is O(T * (h + B)) which is tractable for T ~ 2 x 10^4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .losses import weighted_huber, weighted_median
from .weights import exponential_inv_cum_weight, exponential_weights


HALF_LIFE_TRADES_DEFAULT = 3
HALF_LIFE_SCALE_SECONDS = 240.0      # 4 minutes
HALF_LIFE_OUTLIER_SECONDS = 13.0


# ---------------------------------------------------------------------------
# Head-weight helpers (closed-form exponential)
# ---------------------------------------------------------------------------

def _exp_head(t: int, lam: float, p_head: float = 0.999) -> int:
    """Return ``h = W_t^{-1}(p_head)`` for the exponential weight scheme."""
    return exponential_inv_cum_weight(p_head, t, lam)


def _exp_head_weights(t: int, lam: float, h: int) -> Tuple[np.ndarray, float]:
    """Return the head weight vector w[0..h] and the tail mass 1 - sum(head)."""
    j = np.arange(h + 1, dtype=np.float64)
    n_t = (1.0 - lam ** t) / (1.0 - lam)
    head_w = (lam ** j) / n_t
    tail_mass = max(0.0, 1.0 - head_w.sum())
    return head_w, tail_mass


# ---------------------------------------------------------------------------
# Trade-time exponential filter (stratified)
# ---------------------------------------------------------------------------

def _stratified_trade_time_filter(
    Y: np.ndarray,
    *,
    loss: str,
    lam: float,
    B: int,
    rng: np.random.Generator,
    delta_path: Optional[np.ndarray] = None,
    median_interp: str = "lower",
    huber_init_fn: Optional[Callable[[int], float]] = None,
) -> np.ndarray:
    """Stratified exponential weighted-loss filter over trade-time lags.

    ``loss`` in {"absolute", "huber"}. For Huber, ``delta_path`` must be
    aligned so that ``delta_path[k]`` is the threshold to use at observation
    index k (we use delta_{t-j-1} i.e. shifted by one).
    """
    T = len(Y)
    theta = np.empty(T, dtype=np.float64)

    # Boundary conventions
    theta[0] = Y[0]
    if T >= 2:
        theta[-1] = Y[-1]

    for t_idx in range(1, T - 1):
        t = t_idx + 1  # 1-based "t" used in paper formulas; past observations are Y[:t]
        h = min(_exp_head(t, lam, 0.999), t - 1)
        # Window for lags j=0..h corresponds to Y[t-1], Y[t-2], ..., Y[t-1-h]
        head_idx = np.arange(t - 1, t - 2 - h, -1)
        head_y = Y[head_idx]
        head_w, tail_mass = _exp_head_weights(t, lam, h)

        if tail_mass > 0 and h < t - 1:
            # Analytic inverse-CDF sampling for the tail of a truncated exponential
            # with rate -log(lam) on {h+1,..., t-1}. We just sample lag indices
            # uniformly from the conditional distribution.
            # Cumulative weights on {h+1..t-1}: proportional to lam^j
            # Easiest: inverse CDF via truncated geometric sampling.
            # P(j = k | j > h) = lam^k (1 - lam) / (lam^{h+1} - lam^t)
            # Equivalently sample k = h + 1 + G where G is a truncated geometric
            # on {0,...,t-h-2}. Use rng.random -> search.
            u = rng.random(B)
            # Normalising constant for the conditional distribution
            denom = (lam ** (h + 1)) - (lam ** t)
            # The conditional cdf at k is: (lam^{h+1} - lam^{k+1}) / denom
            # Solve for k: lam^{k+1} = lam^{h+1} - u * denom
            # => k + 1 = log(lam^{h+1} - u*denom) / log(lam)
            val = (lam ** (h + 1)) - u * denom
            # Guard against roundoff
            val = np.clip(val, 1e-300, None)
            k_plus_one = np.log(val) / np.log(lam)
            tail_lags = np.ceil(k_plus_one - 1.0).astype(np.int64)
            tail_lags = np.clip(tail_lags, h + 1, t - 1)
            tail_idx = (t - 1) - tail_lags
            tail_y = Y[tail_idx]
            tail_wB = np.full(B, tail_mass / B, dtype=np.float64)
            y_aug = np.concatenate([head_y, tail_y])
            w_aug = np.concatenate([head_w, tail_wB])
        else:
            y_aug = head_y
            w_aug = head_w
            tail_idx = None

        if loss == "absolute":
            theta[t_idx] = weighted_median(y_aug, w_aug, interpolation=median_interp)
        elif loss == "huber":
            # Threshold per sampled observation
            d_head = (
                delta_path[np.clip(head_idx - 1, 0, len(delta_path) - 1)]
                if delta_path is not None
                else np.full(len(head_idx), 1.345, dtype=np.float64)
            )
            if tail_idx is not None:
                d_tail = delta_path[np.clip(tail_idx - 1, 0, len(delta_path) - 1)]
                d_aug = np.concatenate([d_head, d_tail])
            else:
                d_aug = d_head
            init = huber_init_fn(t_idx) if huber_init_fn is not None else None
            theta[t_idx] = weighted_huber(y_aug, w_aug, delta=d_aug, init=init, tol=1e-12, max_iter=50)
        else:
            raise ValueError(f"unknown loss: {loss}")

    return theta


# ---------------------------------------------------------------------------
# Clock-time exponential scale filter
# ---------------------------------------------------------------------------

def _clock_time_scale_filter(
    tau: np.ndarray,
    abs_resid: np.ndarray,
    *,
    halflife_s: float = HALF_LIFE_SCALE_SECONDS,
    head_window: float = None,
) -> np.ndarray:
    """Weighted median of |residuals| with clock-time exponential weights
    (half-life ``halflife_s`` seconds), zero residuals excluded.

    For speed we use a windowed approximation: at each trade we only consider
    residuals from the past ``head_window`` seconds (default 10 * halflife),
    which captures >99.9% of the cumulative weight.
    """
    if head_window is None:
        head_window = 10.0 * halflife_s  # 2^-10 ~ 1e-3
    T = len(abs_resid)
    b = np.empty(T, dtype=np.float64)
    # Two-pointer window
    lam_rate = np.log(2.0) / halflife_s
    j_start = 0
    for i in range(T):
        tau_i = tau[i]
        while tau_i - tau[j_start] > head_window and j_start < i:
            j_start += 1
        idx = np.arange(j_start, i + 1)
        yy = abs_resid[idx]
        mask = yy > 0.0
        if not mask.any():
            b[i] = b[i - 1] if i > 0 else 0.0
            continue
        yy = yy[mask]
        dt = tau_i - tau[idx[mask]]
        w = np.exp(-lam_rate * dt)
        w = w / w.sum()
        b[i] = weighted_median(yy, w, interpolation="lower")
    return b


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    tau: np.ndarray                 # seconds since midnight
    Y: np.ndarray                   # log prices
    m: np.ndarray                   # median filter
    resid_med: np.ndarray           # Y - m
    b: np.ndarray                   # scale filter
    delta: np.ndarray               # 2 * 1.4826 * b
    theta: np.ndarray               # Huber filter
    resid_hub: np.ndarray           # Y - theta
    linear: np.ndarray              # 3-trade-halflife linear filter
    T60: int
    d_pos: np.ndarray               # 1(resid > delta_prev)
    d_neg: np.ndarray               # 1(resid < -delta_prev)
    q: np.ndarray                   # directional outlier probability
    meta: Dict = field(default_factory=dict)


def linear_exponential_filter(Y: np.ndarray, halflife: int = 3) -> np.ndarray:
    """Causal exponentially weighted mean with half-life ``halflife`` trades.

    Uses the standard recursive form so this is O(T).
    """
    lam = 2.0 ** (-1.0 / halflife)
    out = np.empty_like(Y, dtype=np.float64)
    out[0] = Y[0]
    for t in range(1, len(Y)):
        out[t] = lam * out[t - 1] + (1.0 - lam) * Y[t]
    return out


def run_pipeline(
    tau: np.ndarray,
    Y: np.ndarray,
    *,
    halflife_trades: int = HALF_LIFE_TRADES_DEFAULT,
    B: int = 100,
    seed: int = 0,
    T60: Optional[int] = None,
) -> PipelineResult:
    """Run the five-filter pipeline on one day's log prices ``Y`` at times ``tau``."""
    tau = np.asarray(tau, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    lam = 2.0 ** (-1.0 / halflife_trades)
    T = len(Y)
    if T60 is None:
        T60 = int(((tau - tau[0]) <= 60.0).sum())

    # Stage 1: median price filter m_t
    m = _stratified_trade_time_filter(
        Y, loss="absolute", lam=lam, B=B, rng=rng, median_interp="lower"
    )

    # Stage 2: scale filter b_t and delta_t
    resid_med = Y - m
    # for t <= T60 we blend with previous openings; here we use the warm-up
    # convention of taking the running MAD of the first-minute residuals as b.
    abs_resid = np.abs(resid_med)
    b = np.zeros(T, dtype=np.float64)
    if T60 > 1:
        warm = abs_resid[:T60]
        warm = warm[warm > 0]
        if warm.size > 0:
            b_warm = float(np.median(warm))
        else:
            b_warm = 1e-6
        b[:T60] = b_warm
    # Rolling clock-time filter from T60 onward (use last-T60 residuals too)
    b_tail = _clock_time_scale_filter(
        tau[max(0, T60 - 1):],
        abs_resid[max(0, T60 - 1):],
        halflife_s=HALF_LIFE_SCALE_SECONDS,
    )
    b[T60:] = b_tail[1:] if T60 > 0 else b_tail
    # Guard against zeros
    b = np.where(b <= 0, np.median(abs_resid[abs_resid > 0]) if (abs_resid > 0).any() else 1e-6, b)
    delta = 2.0 * 1.4826 * b

    # Stage 3: Huber price filter theta_t (uses delta_{t-j-1})
    theta = _stratified_trade_time_filter(
        Y,
        loss="huber",
        lam=lam,
        B=B,
        rng=rng,
        delta_path=delta,
        huber_init_fn=lambda i: m[i],
    )

    # Stage 4: benchmark linear path
    linear = linear_exponential_filter(Y, halflife=halflife_trades)

    # Stage 5: Huber residuals and outlier indicators
    resid_hub = Y - theta
    delta_prev = np.concatenate([[delta[0]], delta[:-1]])
    d_pos = (resid_hub > delta_prev).astype(np.float64)
    d_neg = (resid_hub < -delta_prev).astype(np.float64)

    # Stage 8: directional outlier probability q_t
    q = _directional_outlier_prob(tau, d_pos, d_neg, halflife_s=HALF_LIFE_OUTLIER_SECONDS, T60=T60)

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
        T60=T60,
        d_pos=d_pos,
        d_neg=d_neg,
        q=q,
        meta={"halflife_trades": halflife_trades, "B": B, "seed": seed},
    )


def _directional_outlier_prob(
    tau: np.ndarray,
    d_pos: np.ndarray,
    d_neg: np.ndarray,
    *,
    halflife_s: float = HALF_LIFE_OUTLIER_SECONDS,
    T60: int = 0,
) -> np.ndarray:
    """Exponentially weighted clock-time Bernoulli filter of outlier indicators,
    returning ``q_t = p_t^+ / (p_t^+ + p_t^-)``.

    Implemented recursively via the continuous-time exponential filter.
    """
    T = len(tau)
    p_pos = np.zeros(T, dtype=np.float64)
    p_neg = np.zeros(T, dtype=np.float64)
    for t in range(1, T):
        dt = max(0.0, tau[t] - tau[t - 1])
        decay = 0.5 ** (dt / halflife_s)
        # standard EWMA update treating weight 1-decay to the current observation
        p_pos[t] = decay * p_pos[t - 1] + (1.0 - decay) * d_pos[t]
        p_neg[t] = decay * p_neg[t - 1] + (1.0 - decay) * d_neg[t]
    s = p_pos + p_neg
    q = np.where(s > 1e-12, p_pos / np.maximum(s, 1e-12), 0.5)
    # mask the warm-up region
    q[:T60] = np.nan
    return q
