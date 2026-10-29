"""Checks for the appendix building blocks of Experiment 4."""

import numpy as np

from src.hf_filters import clock_time_scale
from src.metrics import psi_star_cents


def test_count_startup_matches_clock_after_the_first_minute():
    rng = np.random.default_rng(0)
    tau = np.sort(rng.uniform(0.0, 300.0, 3000))
    tau[0] = 0.5
    a = np.abs(rng.standard_cauchy(3000)) * (rng.random(3000) > 0.4)
    pre = np.abs(rng.standard_normal(200)) + 0.1
    t60 = int(np.sum(tau <= 60.0))
    clock = clock_time_scale(tau, a, pre, 240.0, 60.0)
    count = clock_time_scale(tau, a, pre, 240.0, 60.0, True, t60)
    assert np.array_equal(clock[t60:], count[t60:])
    assert not np.array_equal(clock[:t60], count[:t60])
    assert count[0] == clock[0] or a[0] > 0.0


def test_rolling_psi_star_full_window_equals_table3_estimator():
    from exp.exp_4_robustness import rolling_psi_star

    rng = np.random.default_rng(1)
    n = 5000
    price = 100.0 + np.cumsum(rng.standard_normal(n) * 0.01)
    theta = np.log(price) + rng.standard_t(1.5, n) * 1e-5
    delta_prev = np.full(n, 3e-5)
    ref = psi_star_cents(price, theta, delta_prev)["psi_star_cents"]
    e = (price - np.exp(theta)) * 100.0
    d = delta_prev * price * 100.0
    _s, _e, ps = rolling_psi_star(np.clip(e, -d, d), np.abs(e) <= d, n, 1)
    assert np.isclose(ps[0], ref, rtol=1e-9)


def test_variance_gamma_quadratic_variation_has_the_target_mean():
    from exp import exp_4_robustness as e4

    tau = np.linspace(0.0, 23_400.0, 200_001)
    dt = np.diff(tau, prepend=tau[0])
    e4._E.update(tau=tau, dt=dt, sigma=0.40 / np.sqrt(252.0 * 23_400.0))
    qv = [np.sum(e4._latent("vg", m)[1] ** 2) for m in range(40)]
    ann = 100.0 * np.sqrt(252.0 * np.mean(qv))
    assert 35.0 < ann < 45.0
