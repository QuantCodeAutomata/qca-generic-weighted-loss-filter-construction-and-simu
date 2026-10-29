"""Unit tests for the three filter estimators."""

from __future__ import annotations

import time

import numpy as np

from src.filters import anchored_median_filter, exact_filter, simulated_filter, stratified_filter
from src.weights import (
    exponential_inv_cum_weight,
    exponential_weights,
    uniform_weights,
)


RNG_SEED = 1234


def test_exact_filter_output_shape():
    Y = np.arange(10, dtype=np.float64)
    theta = exact_filter(Y, lambda t: uniform_weights(t, 3), loss_name="squared")
    assert theta.shape == (10,)
    assert theta[0] == Y[0]  # t=1 collapses to a single-obs mean


def test_exact_filter_squared_matches_running_window_mean():
    Y = np.arange(1, 11, dtype=np.float64)
    theta = exact_filter(Y, lambda t: uniform_weights(t, 2), loss_name="squared")
    # for t >= 3 the window is the last 3 observations
    for t in range(3, 11):
        assert np.isclose(theta[t - 1], Y[t - 3 : t].mean())


def test_simulated_converges_to_exact_absolute():
    """Large-B simulated filter should match the exact weighted-median filter."""
    rng = np.random.default_rng(RNG_SEED)
    Y = rng.standard_normal(50)
    wfn = lambda t: exponential_weights(t, 0.8)
    theta_e = exact_filter(Y, wfn, loss_name="absolute")
    theta_s = simulated_filter(
        Y, wfn, loss_name="absolute", B=5000, rng=np.random.default_rng(RNG_SEED)
    )
    # Allow some noise given B is finite
    assert np.mean(np.abs(theta_e - theta_s)) < 0.3


def test_stratified_matches_exact_tightly_for_absolute():
    """Stratified estimator with 99.9% head mass should be very close to exact."""
    rng = np.random.default_rng(RNG_SEED)
    Y = rng.standard_normal(80)
    wfn = lambda t: exponential_weights(t, 0.8)
    hfn = lambda t: exponential_inv_cum_weight(0.999, t, 0.8)
    theta_e = exact_filter(Y, wfn, loss_name="absolute")
    theta_sr = stratified_filter(
        Y, wfn, hfn, loss_name="absolute", B=500, rng=np.random.default_rng(RNG_SEED)
    )
    assert np.mean(np.abs(theta_e - theta_sr)) < 0.2


def test_simulated_converges_to_exact_squared():
    rng = np.random.default_rng(RNG_SEED)
    Y = rng.standard_normal(60)
    wfn = lambda t: exponential_weights(t, 0.9)
    theta_e = exact_filter(Y, wfn, loss_name="squared")
    theta_s = simulated_filter(
        Y, wfn, loss_name="squared", B=10000, rng=np.random.default_rng(RNG_SEED)
    )
    assert np.mean(np.abs(theta_e - theta_s)) < 0.05


def test_simulation_improves_with_more_B():
    """Approximation error should decrease as B grows."""
    rng = np.random.default_rng(RNG_SEED)
    Y = rng.standard_normal(60)
    wfn = lambda t: exponential_weights(t, 0.9)
    theta_e = exact_filter(Y, wfn, loss_name="absolute")
    errs = []
    for B in [100, 2000]:
        theta_s = simulated_filter(
            Y, wfn, loss_name="absolute", B=B, rng=np.random.default_rng(RNG_SEED)
        )
        errs.append(np.mean(np.abs(theta_e - theta_s)))
    assert errs[1] <= errs[0] + 1e-3


def test_approximate_linear_time_scaling():
    """Full pass over T should scale approximately linearly for fixed B."""
    rng = np.random.default_rng(0)
    wfn = lambda t: exponential_weights(t, 0.9)
    hfn = lambda t: exponential_inv_cum_weight(0.999, t, 0.9)
    times = {}
    for T in [500, 1500]:
        Y = rng.standard_normal(T)
        t0 = time.perf_counter()
        stratified_filter(
            Y, wfn, hfn, loss_name="absolute", B=100, rng=np.random.default_rng(0)
        )
        times[T] = time.perf_counter() - t0
    # ratio of runtimes should be close to T ratio (within a factor of 5
    # given measurement noise on a tiny benchmark)
    ratio = times[1500] / max(times[500], 1e-9)
    assert ratio < 5 * (1500 / 500)


def test_anchored_median_endpoints():
    Y = np.random.default_rng(3).standard_cauchy(60)
    roll = exact_filter(Y, lambda t: uniform_weights(t, 4), loss_name="absolute")
    assert np.allclose(anchored_median_filter(Y, 4, 1.0), roll)
    assert np.allclose(anchored_median_filter(Y, 4, 0.0), np.median(Y))


def test_lag_samplers_follow_the_weights():
    from src.filters import exponential_lag_sampler, hyperbolic_lag_sampler
    from src.weights import exponential_weights, hyperbolic_weights

    rng = np.random.default_rng(0)
    for draw, w in ((exponential_lag_sampler(0.8), exponential_weights(60, 0.8)),
                    (hyperbolic_lag_sampler(3.0), hyperbolic_weights(60, 3.0))):
        k = draw(rng, 60, 400_000)
        freq = np.bincount(k, minlength=60) / len(k)
        assert np.abs(freq - w).max() < 3e-3
        tail = draw(rng, 60, 1000, lo=20)
        assert tail.min() >= 20 and tail.max() <= 59


def test_fast_stratified_matches_exact_for_squared_loss_in_expectation():
    from src.filters import stratified_exponential_filter
    from src.weights import exponential_weights

    rng = np.random.default_rng(1)
    Y = rng.standard_normal(300)
    exact = exact_filter(Y, lambda t: exponential_weights(t, 0.9), loss_name="squared")
    strat = stratified_exponential_filter(Y, 0.9, "squared", B=100, rng=np.random.default_rng(2))
    assert np.max(np.abs(exact - strat)) < 1e-2


def test_hf_median_tie_after_a_three_trade_jump():
    """With lam**3 = 1/2 the newest three trades win the exact tie, B or not."""
    from src.hf_filters import exponential_median

    for jump in (-0.5, 0.5):
        y = np.r_[np.zeros(400), np.full(3, jump), 0.0]
        for B in (0, 100):
            m = exponential_median(y, 3, B=B, seed=5)
            assert m[402] == jump
            assert m[401] == 0.0
