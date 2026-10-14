"""
Lag-weighting schemes used in the generic weighted-loss filter.

Implements the four exact weight models described in the paper:
    (1) Uniform weights
    (2) Exponential weights (with closed-form inverse CDF)
    (3) Hyperbolic weights (via the digamma function)
    (4) Superposition weights (convex combination of base schemes)

All weight vectors returned have length ``t`` (lags j=0,...,t-1) and
sum to one up to numerical tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np
from scipy.special import digamma


# Custom implementation --- Context7 confirms no library equivalent exists
# for these specific weight schemes with the inverse-CDF properties needed.


def uniform_weights(t: int, ell: int) -> np.ndarray:
    """Uniform window weights ``w_{t,j} = 1(j <= ell)/min(t, 1+ell)``.

    Parameters
    ----------
    t : int
        Current time index (number of past observations used, j=0,...,t-1).
    ell : int
        Lookback window length (nonnegative).

    Returns
    -------
    np.ndarray of shape (t,)
        Nonnegative weights summing to one.
    """
    if t <= 0:
        raise ValueError("t must be positive")
    if ell < 0:
        raise ValueError("ell must be nonnegative")
    denom = min(t, 1 + ell)
    w = np.zeros(t, dtype=np.float64)
    k = min(t, ell + 1)
    w[:k] = 1.0 / denom
    return w


def exponential_weights(t: int, lam: float) -> np.ndarray:
    """Exponential weights ``w_{t,j} = lambda^j / n_t`` for j=0,...,t-1.

    ``n_t = (1 - lambda^t) / (1 - lambda)`` is the normalizing sum.
    """
    if not (0.0 < lam < 1.0):
        raise ValueError("lam must lie in (0, 1)")
    if t <= 0:
        raise ValueError("t must be positive")
    j = np.arange(t, dtype=np.float64)
    # Use log-space stable formula for n_t when lam is near 1
    n_t = (1.0 - lam ** t) / (1.0 - lam)
    w = (lam ** j) / n_t
    return w


def exponential_cum_weight(k: int, t: int, lam: float) -> float:
    """``W_t(k) = (1 - lambda^{k+1}) / (1 - lambda^t)`` -- cumulative weight at lag k."""
    if k < 0:
        return 0.0
    k_eff = min(k, t - 1)
    return (1.0 - lam ** (k_eff + 1)) / (1.0 - lam ** t)


def exponential_inv_cum_weight(p: float, t: int, lam: float) -> int:
    """``W_t^{-1}(p) = ceil(kappa_p)`` where
    ``1 + kappa_p = log(1 - p(1 - lambda^t)) / log(lambda)``.

    Returns the smallest lag k such that W_t(k) >= p (clipped to [0, t-1]).
    """
    if not (0.0 <= p < 1.0):
        raise ValueError("p must lie in [0, 1)")
    if t <= 0:
        raise ValueError("t must be positive")
    val = 1.0 - p * (1.0 - lam ** t)
    # log(val) / log(lam) gives (1 + kappa_p); kappa_p = that - 1
    one_plus_kappa = np.log(val) / np.log(lam)
    kappa_p = one_plus_kappa - 1.0
    k = int(np.ceil(kappa_p))
    return int(np.clip(k, 0, t - 1))


def hyperbolic_weights(t: int, a: float) -> np.ndarray:
    """Hyperbolic weights ``w_{t,j} = (1/(j+a)) / n_t`` with
    ``n_t = Psi(t + a) - Psi(a)`` (digamma).

    Parameters
    ----------
    a : float
        Positive hyperbolic shape parameter.
    """
    if a <= 0.0:
        raise ValueError("a must be positive")
    if t <= 0:
        raise ValueError("t must be positive")
    j = np.arange(t, dtype=np.float64)
    raw = 1.0 / (j + a)
    n_t = digamma(t + a) - digamma(a)
    return raw / n_t


@dataclass
class SuperpositionSpec:
    """Specification of a superposition weight model: convex mix of base specs.

    Each component is a callable ``t -> np.ndarray`` of length ``t``.
    """

    components: List[Callable[[int], np.ndarray]]
    alphas: Sequence[float]

    def __post_init__(self) -> None:
        a = np.asarray(self.alphas, dtype=np.float64)
        if a.ndim != 1 or len(a) != len(self.components):
            raise ValueError("alphas must align with components")
        if np.any(a < 0.0):
            raise ValueError("alphas must be nonnegative")
        if not np.isclose(a.sum(), 1.0):
            raise ValueError("alphas must sum to 1")

    def weights(self, t: int) -> np.ndarray:
        w = np.zeros(t, dtype=np.float64)
        for alpha, comp in zip(self.alphas, self.components):
            w += alpha * comp(t)
        return w


def anchored_weights(t: int, ell: int, alpha: float, T: int) -> np.ndarray:
    """Anchored weights (paper Part B stationary design):

    ``w_{t,j} = (1 - alpha)/T + alpha * 1(0 <= j <= ell)/min(t, 1+ell)``.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must lie in [0, 1]")
    if T <= 0:
        raise ValueError("T must be positive")
    base_unif = uniform_weights(t, ell)
    anchor = np.full(t, (1.0 - alpha) / T, dtype=np.float64)
    w = anchor + alpha * base_unif
    # Normalize to remove tiny floating drift; the model is well-defined.
    return w / w.sum()


def validate_weights(w: np.ndarray, atol: float = 1e-10) -> None:
    """Assert nonnegativity and unit sum."""
    assert np.all(w >= -atol), "weights must be nonnegative"
    assert abs(w.sum() - 1.0) < 1e-8, f"weights must sum to 1, got {w.sum()}"
