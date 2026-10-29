"""Synthetic data of the experiments.

* Example 7(a) and 7(b) signals in Cauchy noise (exp_1, exp_2);
* i.i.d. Cauchy draws for the runtime study (exp_1);
* Brownian and variance-gamma log-prices observed at a day's trade times,
  and the Student-t noise added to them (exp_4 Module E, Appendix A.7).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from src.config import EX7A_SIGNAL_SD, EX7B_PHI


def nonstationary_cauchy(
    T: int = 1000, sigma_eta: float = EX7A_SIGNAL_SD, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Example 7(a), the nonstationary signal (exp_1, exp_2 Parts A, C, D):

        mu_{t+1} = mu_t + 0.2 eta_t,  eta_t ~ N(0, 1),  mu_1 = 0
        epsilon_t ~ Cauchy(0, 1)
        Y_t      = mu_t + epsilon_t

    Returns ``(Y, mu)`` of length ``T``.
    """
    rng = np.random.default_rng(seed)
    eta = rng.standard_normal(T)
    mu = np.empty(T, dtype=np.float64)
    mu[0] = 0.0
    for t in range(1, T):
        mu[t] = mu[t - 1] + sigma_eta * eta[t - 1]
    epsilon = rng.standard_cauchy(T)
    Y = mu + epsilon
    return Y, mu


def stationary_cauchy(
    T: int = 1000, phi: float = EX7B_PHI, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Example 7(b), the stationary signal (exp_2 Part B):

        mu_{t+1} = phi mu_t + sqrt(1 - phi^2) eta_t,  mu_1 ~ N(0, 1)
        Y_t      = mu_t + epsilon_t,  epsilon_t ~ Cauchy(0, 1)
    """
    rng = np.random.default_rng(seed)
    sd_e = np.sqrt(1.0 - phi * phi)
    eta = rng.standard_normal(T)
    mu = np.empty(T, dtype=np.float64)
    mu[0] = rng.standard_normal()
    for t in range(1, T):
        mu[t] = phi * mu[t - 1] + sd_e * eta[t - 1]
    epsilon = rng.standard_cauchy(T)
    Y = mu + epsilon
    return Y, mu


def iid_cauchy(T: int, seed: int = 0) -> np.ndarray:
    """``T`` standard Cauchy draws (the exp_1 runtime study)."""
    return np.random.default_rng(seed).standard_cauchy(T)


def brownian_at_trades(dt: np.ndarray, sigma: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Brownian log-price at the trade times, ``sigma`` per square-root second.

    ``dt`` are the clock gaps between trades (``dt[0]`` is ignored: the path
    starts at zero on the opening print). Returns the path and its increments.
    """
    rng = np.random.default_rng(seed)
    inc = rng.normal(size=len(dt)) * sigma * np.sqrt(dt)
    inc[0] = 0.0
    return np.cumsum(inc), inc


def variance_gamma_at_trades(dt: np.ndarray, sigma: float, beta: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Variance-gamma log-price at the trade times (Appendix A.7).

    Increments ``U_t sqrt(Z(τ_t) − Z(τ_{t−1}))`` with gamma increments of shape
    ``σ²β Δτ`` and rate ``β``, the price in percent, so ``E Z(u) = σ² u``.
    """
    rng = np.random.default_rng(seed)
    s_pct = 100.0 * sigma
    g = rng.gamma(shape=np.maximum(s_pct * s_pct * beta * dt, 0.0), scale=1.0 / beta)
    inc = rng.standard_normal(len(dt)) * np.sqrt(g) / 100.0
    inc[0] = 0.0
    return np.cumsum(inc), inc


def student_t_noise(n: int, scale: float, df: float, seed: int) -> np.ndarray:
    """``scale`` times ``n`` Student-t draws with ``df`` degrees of freedom (Appendix A.7)."""
    return scale * np.random.default_rng(seed).standard_t(df, size=n)
