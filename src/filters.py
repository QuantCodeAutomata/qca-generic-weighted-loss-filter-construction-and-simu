"""
Generic weighted-loss filter: exact, simulation, and stratified estimators.

Target filter (paper equation):

    theta_t = argmin_theta sum_{j=0}^{t-1} w_{t,j} * [ (1 - alpha) * E[L(theta, Y_1)]
                                                    + alpha * L(theta, Y_{t-j}) ]

For the empirical-application special case ``alpha = 1``:

    theta_t = argmin_theta sum_{j=0}^{t-1} w_{t,j} L(theta, Y_{t-j})

Three estimators are provided:
    1. ``exact_filter`` -- direct minimisation over the weighted empirical sample.
    2. ``simulated_filter`` -- Monte Carlo resample from the weighted empirical
       distribution ``F_t`` with ``B`` draws, then minimise the uniform-weight
       objective.
    3. ``stratified_filter`` -- for ``alpha = 1``, compute the heaviest
       weights exactly up to ``h = W_t^{-1}(0.999)`` and MC-sample only the tail.

``anchored_median_filter`` is the two-step anchored median of Section 2.4.2.
``simulated_filter_fast`` and ``stratified_exponential_filter`` are the same
estimators with the O(1) lag samplers of Section 3.4, used for the runtime
and Theorem 2 checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from .losses import (
    LOSSES,
    weighted_bernoulli_prob,
    weighted_huber,
    weighted_mean,
    weighted_median,
)


WeightFn = Callable[[int], np.ndarray]           # t -> weights of length t
LagSampler = Callable[[np.random.Generator, int, int], np.ndarray]  # (rng, t, B) -> lag indices
Solver = Callable[..., float]                    # solver signature depends on loss


# ---------------------------------------------------------------------------
# Default solvers
# ---------------------------------------------------------------------------

def _solver_for(loss_name: str, **kwargs) -> Solver:
    """Return a lambda(y, w) -> theta for the given loss."""
    if loss_name == "squared":
        return lambda y, w: weighted_mean(y, w)
    if loss_name == "absolute":
        interp = kwargs.get("interpolation", "hazen")
        return lambda y, w: weighted_median(y, w, interpolation=interp)
    if loss_name == "huber":
        # delta provided per-call via a vector matching y
        def _solve(y, w, delta=kwargs.get("delta"), init=None):
            return weighted_huber(y, w, delta=delta, init=init)
        return _solve
    if loss_name == "bernoulli":
        return lambda y, w: weighted_bernoulli_prob(y, w)
    raise ValueError(f"unknown loss: {loss_name}")


# ---------------------------------------------------------------------------
# Exact filter
# ---------------------------------------------------------------------------

def exact_filter(
    Y: np.ndarray,
    weight_fn: WeightFn,
    loss_name: str = "absolute",
    *,
    delta_path: Optional[np.ndarray] = None,
    interpolation: str = "hazen",
) -> np.ndarray:
    """Compute theta_t exactly for t = 1,...,T.

    The empirical window at time ``t`` is ``Y[0:t]`` indexed with
    lag ``j = t-1-i`` so that ``j = 0`` corresponds to ``Y[t-1]`` (current obs).
    Weights are those returned by ``weight_fn(t)`` with convention
    ``w[j]`` = weight on lag j.
    """
    Y = np.asarray(Y, dtype=np.float64)
    T = len(Y)
    theta = np.empty(T, dtype=np.float64)
    for t in range(1, T + 1):
        w = weight_fn(t)
        # lag j in weights maps to index t-1-j in Y
        y_win = Y[:t][::-1]  # position j of this array is Y[t-1-j]
        if loss_name == "huber":
            d = (
                delta_path[:t][::-1]
                if delta_path is not None
                else np.full(t, 1.345, dtype=np.float64)
            )
            theta[t - 1] = weighted_huber(y_win, w, delta=d)
        elif loss_name == "squared":
            theta[t - 1] = weighted_mean(y_win, w)
        elif loss_name == "absolute":
            theta[t - 1] = weighted_median(y_win, w, interpolation=interpolation)
        elif loss_name == "bernoulli":
            theta[t - 1] = weighted_bernoulli_prob(y_win, w)
        else:
            raise ValueError(f"unknown loss: {loss_name}")
    return theta


def anchored_median_filter(Y: np.ndarray, ell: int, alpha: float) -> np.ndarray:
    """Ex. 7(b) weighted median with ``F_t = (1-alpha) ECDF(Y_{1:T}) + alpha * rolling``.

    ``w_{t,j} = (1-alpha)/T + alpha 1(0<=j<=ell)/min(t,1+ell)`` for
    ``j in {t-T,...,t-1}``: the anchor covers the whole sample, future included.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must lie in [0, 1]")
    Y = np.asarray(Y, dtype=np.float64)
    T = len(Y)
    w_anchor = np.full(T, (1.0 - alpha) / T)
    theta = np.empty(T, dtype=np.float64)
    for t in range(1, T + 1):
        win = Y[max(0, t - 1 - ell):t]
        w_win = np.full(len(win), alpha / len(win))
        theta[t - 1] = weighted_median(np.concatenate([win, Y]),
                                       np.concatenate([w_win, w_anchor]))
    return theta


# ---------------------------------------------------------------------------
# Monte Carlo filter (general alpha)
# ---------------------------------------------------------------------------

def _sample_lags(
    rng: np.random.Generator, w: np.ndarray, B: int
) -> np.ndarray:
    """Draw ``B`` lag indices j in {0,...,t-1} with probability w[j]."""
    cdf = np.cumsum(w)
    cdf[-1] = 1.0  # guard rounding
    u = rng.random(B)
    return np.searchsorted(cdf, u, side="right").clip(0, len(w) - 1)


def simulated_filter(
    Y: np.ndarray,
    weight_fn: WeightFn,
    loss_name: str = "absolute",
    *,
    B: int = 500,
    alpha: float = 1.0,
    anchor_sampler: Optional[Callable[[np.random.Generator, int], np.ndarray]] = None,
    delta_path: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    interpolation: str = "hazen",
) -> np.ndarray:
    """Simulation-based estimator of ``theta_t`` using ``B`` Monte Carlo draws.

    For ``alpha < 1`` the mixture ``F_t = (1-alpha) F_{Y_1} + alpha * \\hat F_t``
    is sampled by drawing ``B_1 ~ Binomial(B, alpha)`` observations from the
    empirical distribution and ``B_2 = B - B_1`` from the anchor sampler.
    """
    Y = np.asarray(Y, dtype=np.float64)
    T = len(Y)
    if rng is None:
        rng = np.random.default_rng(0)

    theta = np.empty(T, dtype=np.float64)
    for t in range(1, T + 1):
        w = weight_fn(t)
        y_win = Y[:t][::-1]
        if alpha >= 1.0:
            lags = _sample_lags(rng, w, B)
            Y_star = y_win[lags]
            if loss_name == "huber":
                d_win = (
                    delta_path[:t][::-1]
                    if delta_path is not None
                    else np.full(t, 1.345, dtype=np.float64)
                )
                d_star = d_win[lags]
                theta[t - 1] = weighted_huber(
                    Y_star, np.ones(B) / B, delta=d_star
                )
            elif loss_name == "squared":
                theta[t - 1] = Y_star.mean()
            elif loss_name == "absolute":
                theta[t - 1] = weighted_median(
                    Y_star, np.ones(B) / B, interpolation=interpolation
                )
            elif loss_name == "bernoulli":
                theta[t - 1] = float(np.clip(Y_star.mean(), 1e-12, 1 - 1e-12))
            else:
                raise ValueError(f"unknown loss: {loss_name}")
        else:
            if anchor_sampler is None:
                raise ValueError("anchor_sampler is required when alpha < 1")
            B1 = int(rng.binomial(B, alpha))
            B2 = B - B1
            samples = []
            if B1 > 0:
                lags = _sample_lags(rng, w, B1)
                samples.append(y_win[lags])
            if B2 > 0:
                samples.append(anchor_sampler(rng, B2))
            Y_star = np.concatenate(samples) if samples else np.array([])
            if loss_name == "squared":
                theta[t - 1] = float(Y_star.mean())
            elif loss_name == "absolute":
                theta[t - 1] = weighted_median(
                    Y_star, np.ones_like(Y_star) / len(Y_star), interpolation=interpolation
                )
            elif loss_name == "huber":
                # delta per-draw: use delta at the sampled lag for in-sample,
                # fall back to median delta for anchor draws.
                d_win = (
                    delta_path[:t][::-1]
                    if delta_path is not None
                    else np.full(t, 1.345, dtype=np.float64)
                )
                d_star = np.empty(B, dtype=np.float64)
                if B1 > 0:
                    d_star[:B1] = d_win[lags]
                if B2 > 0:
                    d_star[B1:] = np.median(d_win)
                theta[t - 1] = weighted_huber(
                    Y_star, np.ones(B) / B, delta=d_star
                )
            elif loss_name == "bernoulli":
                theta[t - 1] = float(np.clip(Y_star.mean(), 1e-12, 1 - 1e-12))
            else:
                raise ValueError(f"unknown loss: {loss_name}")
    return theta


# ---------------------------------------------------------------------------
# Stratified filter (alpha = 1 only)
# ---------------------------------------------------------------------------

def stratified_filter(
    Y: np.ndarray,
    weight_fn: WeightFn,
    head_cutoff_fn: Callable[[int], int],
    loss_name: str = "absolute",
    *,
    B: int = 500,
    delta_path: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    interpolation: str = "hazen",
) -> np.ndarray:
    """Stratified Monte Carlo filter for ``alpha = 1``.

    The head (lags j=0,...,h) is retained exactly with its original weights
    and only the tail mass ``1 - W_t(h)`` is approximated by ``B`` MC draws
    with ``P(j = k | j > h) = w_{t,k} / (1 - W_t(h))``.

    The combined objective is minimised directly:
        dot L*(theta) = sum_{j=0}^h w_{t,j} L(theta, Y_{t-j})
                      + (1 - sum_{j=0}^h w_{t,j}) * (1/B) sum_b L(theta, Y_b*)

    which is equivalent to minimising ``sum_i w_i L(theta, y_i)`` over the
    augmented sample consisting of head values with their original weights
    and ``B`` tail draws each with weight ``(1 - W_t(h)) / B``.
    """
    Y = np.asarray(Y, dtype=np.float64)
    T = len(Y)
    if rng is None:
        rng = np.random.default_rng(0)

    theta = np.empty(T, dtype=np.float64)
    for t in range(1, T + 1):
        w = weight_fn(t)
        y_win = Y[:t][::-1]
        h = min(head_cutoff_fn(t), t - 1)
        head_y = y_win[: h + 1]
        head_w = w[: h + 1]
        tail_mass = max(0.0, 1.0 - head_w.sum())
        if tail_mass <= 1e-15 or h == t - 1:
            # entire window is "head"
            y_aug = head_y
            w_aug = head_w
            d_aug = None
            if loss_name == "huber":
                d_win = (
                    delta_path[:t][::-1]
                    if delta_path is not None
                    else np.full(t, 1.345, dtype=np.float64)
                )
                d_aug = d_win[: h + 1]
        else:
            tail_w = w[h + 1:].copy()
            tail_w /= tail_w.sum()
            tail_lags = _sample_lags(rng, tail_w, B) + (h + 1)
            tail_y = y_win[tail_lags]
            tail_wB = np.full(B, tail_mass / B, dtype=np.float64)
            y_aug = np.concatenate([head_y, tail_y])
            w_aug = np.concatenate([head_w, tail_wB])
            if loss_name == "huber":
                d_win = (
                    delta_path[:t][::-1]
                    if delta_path is not None
                    else np.full(t, 1.345, dtype=np.float64)
                )
                d_aug = np.concatenate([d_win[: h + 1], d_win[tail_lags]])

        if loss_name == "squared":
            theta[t - 1] = weighted_mean(y_aug, w_aug)
        elif loss_name == "absolute":
            theta[t - 1] = weighted_median(y_aug, w_aug, interpolation=interpolation)
        elif loss_name == "huber":
            theta[t - 1] = weighted_huber(y_aug, w_aug, delta=d_aug)
        elif loss_name == "bernoulli":
            theta[t - 1] = weighted_bernoulli_prob(y_aug, w_aug)
        else:
            raise ValueError(f"unknown loss: {loss_name}")
    return theta


# ---------------------------------------------------------------------------
# O(1)-per-t lag samplers (Section 3.4) and the matching fast estimators
# ---------------------------------------------------------------------------

def exponential_lag_sampler(lam: float) -> LagSampler:
    """Inverse-transform draws of ``j`` with ``P(j = k) ∝ lam**k``, ``k = lo..t-1``.

    The truncated geometric quantile is closed form, so a draw costs O(1) in t.
    """
    log_lam = np.log(lam)

    def draw(rng: np.random.Generator, t: int, B: int, lo: int = 0) -> np.ndarray:
        n = t - lo
        u = rng.random(B)
        g = np.floor(np.log1p(-u * (1.0 - lam ** n)) / log_lam).astype(np.int64)
        return lo + np.clip(g, 0, n - 1)

    return draw


def _inv_digamma(y: np.ndarray) -> np.ndarray:
    """Solve ``digamma(x) = y`` by Newton's method (Minka's starting point)."""
    from scipy.special import digamma, polygamma

    y = np.asarray(y, dtype=np.float64)
    x = np.where(y >= -2.22, np.exp(y) + 0.5, -1.0 / (y + 0.5772156649015329))
    for _ in range(6):
        x = x - (digamma(x) - y) / polygamma(1, x)
    return x


def hyperbolic_lag_sampler(a: float) -> LagSampler:
    """Draws with ``P(j = k) ∝ 1/(k + a)``, ``k = lo..t-1``, by the digamma quantile.

    ``W(k) ∝ Ψ(k + 1 + a) − Ψ(lo + a)``; the inverse digamma gives the lag
    in O(1), and one integer step either side fixes the rounding.
    """
    from scipy.special import digamma

    def draw(rng: np.random.Generator, t: int, B: int, lo: int = 0) -> np.ndarray:
        base = digamma(lo + a)
        span = digamma(t + a) - base
        u = rng.random(B)
        target = base + u * span
        k = np.ceil(_inv_digamma(target) - 1.0 - a).astype(np.int64)
        k = np.clip(k, lo, t - 1)
        # Smallest k with Ψ(k + 1 + a) >= target.
        up = digamma(k + 1 + a) < target
        k = np.where(up, k + 1, k)
        down = (k > lo) & (digamma(k + a) >= target)
        k = np.where(down, k - 1, k)
        return np.clip(k, lo, t - 1)

    return draw


def _solve_uniform(loss_name: str, y: np.ndarray, delta: Optional[np.ndarray], median: str) -> float:
    if loss_name == "squared":
        return float(y.mean())
    if loss_name == "absolute":
        if median == "lower":
            s = np.sort(y)
            return float(s[(len(s) + 1) // 2 - 1])
        return weighted_median(y, np.full(len(y), 1.0 / len(y)))
    if loss_name == "huber":
        return weighted_huber(y, np.full(len(y), 1.0 / len(y)), delta=delta)
    if loss_name == "bernoulli":
        return float(np.clip(y.mean(), 1e-12, 1 - 1e-12))
    raise ValueError(f"unknown loss: {loss_name}")


def simulated_filter_fast(
    Y: np.ndarray,
    sampler: LagSampler,
    loss_name: str = "absolute",
    *,
    B: int = 500,
    rng: Optional[np.random.Generator] = None,
    delta: float = 1.345,
    median: str = "lower",
) -> np.ndarray:
    """Definition 3 with ``alpha = 1``: ``B`` lag draws per ``t`` at O(B) cost.

    ``median="lower"`` takes the smallest value reaching half the draws (the
    convention of Theorem 2); ``"hazen"`` interpolates as in Remark 3.
    """
    Y = np.asarray(Y, dtype=np.float64)
    if rng is None:
        rng = np.random.default_rng(0)
    theta = np.empty(len(Y), dtype=np.float64)
    d = np.full(B, float(delta))
    for t in range(1, len(Y) + 1):
        lags = sampler(rng, t, B)
        theta[t - 1] = _solve_uniform(loss_name, Y[t - 1 - lags], d, median)
    return theta


def stratified_exponential_filter(
    Y: np.ndarray,
    lam: float,
    loss_name: str = "absolute",
    *,
    B: int = 100,
    p_head: float = 0.999,
    rng: Optional[np.random.Generator] = None,
    delta: float = 1.345,
    interpolation: str = "lower",
) -> np.ndarray:
    """Remark 4(d) for exponential weights without materialising ``w_{t,0:t-1}``.

    The head ``h = W_t^{-1}(p_head)`` comes from the closed-form quantile and
    the tail draws from ``exponential_lag_sampler``, so each ``t`` costs
    O(h + B), independent of ``t``.
    """
    from .weights import exponential_inv_cum_weight

    Y = np.asarray(Y, dtype=np.float64)
    if rng is None:
        rng = np.random.default_rng(0)
    draw = exponential_lag_sampler(lam)
    theta = np.empty(len(Y), dtype=np.float64)
    for t in range(1, len(Y) + 1):
        h = exponential_inv_cum_weight(p_head, t, lam)
        n_t = (1.0 - lam ** t) / (1.0 - lam)
        head_w = lam ** np.arange(h + 1) / n_t
        y_head = Y[t - 1 - np.arange(h + 1)]
        tail_mass = max(0.0, 1.0 - float(head_w.sum()))
        if h < t - 1 and tail_mass > 0.0:
            lags = draw(rng, t, B, lo=h + 1)
            y_aug = np.concatenate([y_head, Y[t - 1 - lags]])
            w_aug = np.concatenate([head_w, np.full(B, tail_mass / B)])
        else:
            y_aug, w_aug = y_head, head_w
        if loss_name == "squared":
            theta[t - 1] = weighted_mean(y_aug, w_aug)
        elif loss_name == "absolute":
            theta[t - 1] = weighted_median(y_aug, w_aug, interpolation=interpolation)
        elif loss_name == "huber":
            theta[t - 1] = weighted_huber(y_aug, w_aug, delta=float(delta))
        elif loss_name == "bernoulli":
            theta[t - 1] = weighted_bernoulli_prob(y_aug, w_aug)
        else:
            raise ValueError(f"unknown loss: {loss_name}")
    return theta


# ---------------------------------------------------------------------------
# Objective evaluators (useful for diagnostics)
# ---------------------------------------------------------------------------

def objective_value(
    theta: float,
    y_win: np.ndarray,
    w: np.ndarray,
    loss_name: str,
    delta_win: Optional[np.ndarray] = None,
) -> float:
    """Evaluate the weighted-objective ``Q_t(theta)`` for diagnostics."""
    loss = LOSSES[loss_name]["loss"]
    if loss_name == "huber":
        vals = loss(theta, y_win, delta_win)
    else:
        vals = loss(theta, y_win)
    return float(np.sum(w * vals))


@dataclass
class FilterTimings:
    """Simple container for runtime diagnostics."""
    T: int
    seconds: float

    @property
    def per_point_us(self) -> float:
        return 1e6 * self.seconds / max(1, self.T)
