"""Synthetic data generators used in exp_1 and exp_2."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def nonstationary_cauchy(
    T: int = 1000, sigma_eta: float = 0.2, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Paper's Part A design:

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
    T: int = 1000, phi: float = 0.95, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Paper's Part B stationary design:

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


def brownian_log_price(
    n: int, sigma: float = 0.3, dt: float = 1.0 / (252 * 23400), seed: int = 0
) -> np.ndarray:
    """Simulate a Brownian-motion log-price path of length ``n`` with per-second
    variance scaled so annualised vol is ``sigma``. ``dt`` defaults to one second
    on a 252-day calendar."""
    rng = np.random.default_rng(seed)
    innov = rng.standard_normal(n) * sigma * np.sqrt(dt)
    return np.cumsum(innov)


def variance_gamma_log_price(
    n: int,
    sigma: float = 0.3,
    nu: float = 0.5,
    theta_drift: float = 0.0,
    dt: float = 1.0 / (252 * 23400),
    seed: int = 0,
) -> np.ndarray:
    """Simulate a variance-gamma log-price path via gamma-subordinated BM.

    ``X_t = theta_drift * G_t + sigma * W(G_t)`` with ``G_t`` a gamma
    subordinator having mean ``dt`` and variance ``nu * dt``.
    """
    rng = np.random.default_rng(seed)
    # Gamma shape k = dt / nu, scale = nu  (so mean = dt, variance = nu * dt)
    k = dt / nu
    g = rng.gamma(shape=k, scale=nu, size=n)
    w = rng.standard_normal(n) * np.sqrt(np.maximum(g, 0.0))
    increments = theta_drift * g + sigma * w
    return np.cumsum(increments)


def heavy_tail_microstructure_noise(
    n: int, scale: float = 1e-4, tail: float = 1.5, seed: int = 0
) -> np.ndarray:
    """Pareto-like symmetric heavy-tailed noise with configurable tail index."""
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    signs = rng.choice([-1.0, 1.0], size=n)
    # Pareto survival with tail index ``tail``: X = scale * (1 - u)^(-1/tail)
    mag = scale * (1.0 - u) ** (-1.0 / tail)
    return signs * mag
