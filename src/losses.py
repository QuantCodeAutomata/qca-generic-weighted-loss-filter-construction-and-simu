"""
Convex loss functions and loss-specific weighted-objective solvers.

Implements:
    - Squared loss:    L(theta, y) = 0.5 * (y - theta)^2
    - Absolute loss:   L(theta, y) = |y - theta| - |y|   (paper convention)
    - Huber loss:      L(theta, y) = H_delta(y - theta)
    - Bernoulli log-loss (for binary outcomes with sigmoid link)

For each loss we provide a vectorised ``L(theta, y)`` evaluator and a
weighted minimiser ``argmin_theta sum_j w_j L(theta, y_j)``:
    - squared  -> weighted mean (closed form)
    - absolute -> weighted median (custom, Hazen type-5 interpolation)
    - Huber    -> IRLS scalar solve
    - bernoulli-> weighted mean of binary outcomes (closed form on probability)
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Point-wise losses
# ---------------------------------------------------------------------------

def squared_loss(theta: float, y: np.ndarray) -> np.ndarray:
    """``0.5 (y - theta)^2``."""
    return 0.5 * (y - theta) ** 2


def absolute_loss(theta: float, y: np.ndarray) -> np.ndarray:
    """Paper convention ``|y - theta| - |y|`` -- a translation of the L1 loss
    that keeps expectations finite even for Cauchy-type Y."""
    return np.abs(y - theta) - np.abs(y)


def huber_loss(theta: float, y: np.ndarray, delta: float | np.ndarray) -> np.ndarray:
    """Huber loss ``H_delta(y - theta)``.

    ``delta`` may be a scalar or a length-``y`` vector (per-observation threshold).
    """
    r = y - theta
    absr = np.abs(r)
    d = np.asarray(delta, dtype=np.float64)
    quad = 0.5 * r * r
    lin = d * absr - 0.5 * d * d
    return np.where(absr <= d, quad, lin)


def bernoulli_logloss(theta: float, y: np.ndarray) -> np.ndarray:
    """Bernoulli log-loss with ``theta`` interpreted as a probability in (0, 1).

    Returns ``-[y log(theta) + (1-y) log(1-theta)]``.
    """
    eps = 1e-12
    p = float(np.clip(theta, eps, 1.0 - eps))
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


# ---------------------------------------------------------------------------
# Weighted minimisers (loss-specific solvers)
# ---------------------------------------------------------------------------

def weighted_mean(y: np.ndarray, w: np.ndarray) -> float:
    """Argmin of weighted squared loss: closed-form weighted mean."""
    s = w.sum()
    if s <= 0.0:
        raise ValueError("non-positive total weight")
    return float(np.sum(w * y) / s)


def weighted_median(
    y: np.ndarray,
    w: np.ndarray,
    *,
    collapse_ties: bool = True,
    interpolation: str = "hazen",
) -> float:
    """Weighted median via Hazen (type-5) interpolation.

    Mirrors R ``spatstat::weighted.median(type = 5, collapse = TRUE)``:
        1. Drop points with zero weight.
        2. Sort by ``y``.
        3. If ``collapse_ties`` is True, collapse repeated y's by summing weights.
        4. Compute cumulative weights normalized to sum to 1.
        5. Locate the median crossing at cumulative weight 0.5.
        6. If a bucket exactly straddles 0.5 (crosses) -> return that y.
        7. If a bucket boundary equals 0.5 (plateau) -> apply Hazen interpolation
           between the lower and upper bounds.

    When ``interpolation == 'lower'`` the smallest value whose cumulative weight
    is >= 0.5 is returned (the paper's "smallest value reaching half the total
    weight" convention used for the empirical median filter).
    """
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if y.shape != w.shape:
        raise ValueError("y and w must have equal shape")

    mask = w > 0.0
    if not mask.any():
        raise ValueError("all weights are zero")
    y = y[mask]
    w = w[mask]

    order = np.argsort(y, kind="mergesort")
    y = y[order]
    w = w[order]

    if collapse_ties:
        uniq, inv = np.unique(y, return_inverse=True)
        w_coll = np.zeros_like(uniq)
        np.add.at(w_coll, inv, w)
        y = uniq
        w = w_coll

    total = w.sum()
    w = w / total  # normalize
    cum = np.cumsum(w)

    if interpolation == "lower":
        # Smallest value whose cumulative weight >= 0.5.
        idx = int(np.searchsorted(cum, 0.5, side="left"))
        idx = min(idx, len(y) - 1)
        return float(y[idx])

    # Hazen / type-5 interpolation. ``cum[i]`` is upper cumulative at value y[i];
    # lower cumulative at value y[i] is cum[i-1] (or 0 for i=0).
    #
    # Find first index where cum > 0.5 (strict).  Call it ``u``.
    # The lower-bound value theta_lo has F_lo = cum[u-1] (or 0), F_up = cum[u].
    # The upper-bound value theta_up is y[u].
    # If F_lo < 0.5 < F_up (strict crossing), there is no ambiguity, return y[u].
    # If F_lo == 0.5 exactly (plateau), interpolate between y[u-1] and y[u]
    # using Hazen: theta = ((F_up - 0.5)/(F_up - F_lo)) * y[u-1]
    #                   + ((0.5 - F_lo)/(F_up - F_lo)) * y[u].
    # (The second term is zero in the plateau case; the first gives y[u-1].)
    # For practical floating purposes we treat near-plateau (|F_lo - 0.5|<eps)
    # as a plateau and interpolate.
    eps = 1e-12
    u = int(np.searchsorted(cum, 0.5, side="right"))
    if u >= len(y):
        u = len(y) - 1
    F_up = cum[u]
    F_lo = cum[u - 1] if u > 0 else 0.0
    # Plateau case: cumulative mass hits exactly 0.5 at the boundary
    # between y[u-1] and y[u]. Then theta_lo = y[u-1], theta_up = y[u],
    # and the formula weighs them according to the paper.
    if u > 0 and abs(cum[u - 1] - 0.5) < eps:
        theta_lo = y[u - 1]
        theta_up = y[u]
        # F_lo for the plateau: the mass up to but excluding y[u-1]; and
        # F_up is cum[u]. The Hazen formula collapses to a simple average
        # weighted by neighbouring mass.
        F_lo_plat = cum[u - 2] if u >= 2 else 0.0
        F_up_plat = cum[u]
        denom = F_up_plat - F_lo_plat
        if denom <= eps:
            return float(0.5 * (theta_lo + theta_up))
        w_lo = (F_up_plat - 0.5) / denom
        w_up = (0.5 - F_lo_plat) / denom
        return float(w_lo * theta_lo + w_up * theta_up)
    # Strict crossing within bucket u -> median value is y[u].
    return float(y[u])


def weighted_huber(
    y: np.ndarray,
    w: np.ndarray,
    delta: float | np.ndarray,
    *,
    init: Optional[float] = None,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> float:
    """Argmin of weighted Huber loss via scalar IRLS.

    For a convex scalar objective, IRLS iterates
        theta_{k+1} = sum_j w_j * c_j(theta_k) * y_j / sum_j w_j * c_j(theta_k)
    with ``c_j(theta) = 1`` if ``|y_j - theta| <= delta_j`` else ``delta_j/|y_j - theta|``.
    """
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    d = np.asarray(delta, dtype=np.float64)
    if d.ndim == 0:
        d = np.full(y.shape, float(d), dtype=np.float64)
    if not (y.shape == w.shape == d.shape):
        raise ValueError("y, w, delta must share shape (or delta scalar)")

    theta = float(init) if init is not None else weighted_median(y, w)
    for _ in range(max_iter):
        r = y - theta
        absr = np.abs(r)
        c = np.where(absr <= d, 1.0, d / np.maximum(absr, 1e-300))
        wc = w * c
        s = wc.sum()
        if s <= 0.0:
            break
        new_theta = float(np.sum(wc * y) / s)
        if abs(new_theta - theta) < tol:
            theta = new_theta
            break
        theta = new_theta
    return theta


def weighted_bernoulli_prob(y: np.ndarray, w: np.ndarray) -> float:
    """Argmin of weighted Bernoulli log-loss w.r.t. the probability parameter:
    the weighted mean of ``y in {0,1}``, clipped to (eps, 1-eps)."""
    s = w.sum()
    if s <= 0.0:
        raise ValueError("non-positive total weight")
    p = float(np.sum(w * y) / s)
    return float(np.clip(p, 1e-12, 1.0 - 1e-12))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

LOSSES = {
    "squared":   {"loss": squared_loss,   "solver": weighted_mean},
    "absolute":  {"loss": absolute_loss,  "solver": weighted_median},
    "huber":     {"loss": huber_loss,     "solver": weighted_huber},
    "bernoulli": {"loss": bernoulli_logloss, "solver": weighted_bernoulli_prob},
}
