"""Integration tests for the empirical pipeline and metric helpers."""

from __future__ import annotations

import numpy as np

from src.empirical_pipeline import linear_exponential_filter, run_pipeline
from src.metrics import (
    hill_tail_index,
    k_trade_diff_stats,
    predictive_absolute_loss,
    s_second_diff_stats,
    tracking_error,
)
from data.generate_synthetic import nonstationary_cauchy


def test_predictive_absolute_loss_zero_for_perfect_predictor():
    Y = np.array([1.0, 2.0, 3.0, 4.0])
    # theta[t-1] exactly equals Y[t] for t = 2..T  (perfect one-step forecast)
    theta = np.array([2.0, 3.0, 4.0, 0.0])
    assert np.isclose(predictive_absolute_loss(Y, theta), -np.mean(np.abs(Y[1:])))


def test_tracking_error_zero_for_identity():
    mu = np.arange(10.0)
    assert tracking_error(mu, mu) == 0.0


def test_hill_tail_index_pareto_known_tail():
    rng = np.random.default_rng(0)
    # Pareto tail index 2.0: X = (1 - U)^(-1/2)
    u = rng.random(50000)
    x = (1.0 - u) ** (-0.5)
    hill = hill_tail_index(x, k_frac_low=0.001, k_frac_high=0.01)
    assert 1.5 < hill < 2.5


def test_k_trade_diff_stats_keys_and_values():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal(300) * 1e-3)
    stats = k_trade_diff_stats(x, ks=(1, 5))
    for k in (1, 5):
        s = stats[k]
        assert np.isfinite(s["mad_scaled"])
        assert np.isfinite(s["rms_scaled"])


def test_s_second_diff_stats_handles_all_seconds():
    tau = np.linspace(0, 100, 1000)
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal(1000) * 1e-3)
    stats = s_second_diff_stats(tau, x, s_list=(1, 10))
    for s in (1.0, 10.0):
        assert s in stats
        assert np.isfinite(stats[s]["annualized_vol"])


def test_linear_exponential_filter_recursion():
    Y = np.array([1.0, 2.0, 3.0, 4.0])
    out = linear_exponential_filter(Y, halflife=3)
    lam = 2.0 ** (-1 / 3)
    # Finite-sample weights, endpoints fixed at the observations.
    assert out[0] == Y[0]
    assert out[-1] == Y[-1]
    assert np.isclose(out[1], (Y[1] + lam * Y[0]) / (1.0 + lam))
    denom = 1.0 + lam + lam ** 2
    assert np.isclose(out[2], (Y[2] + lam * Y[1] + lam ** 2 * Y[0]) / denom)


def test_run_pipeline_on_synthetic_series_executes_cleanly():
    """The full five-filter pipeline must run end-to-end on a short series."""
    # Use Cauchy noise as a stress test
    Y, _ = nonstationary_cauchy(T=400, seed=42)
    tau = np.arange(len(Y), dtype=np.float64)  # 1 second between trades
    res = run_pipeline(tau, Y, halflife_trades=3, B=50, seed=7, T60=60)
    assert res.theta.shape == Y.shape
    assert res.m.shape == Y.shape
    assert res.delta.shape == Y.shape
    assert np.all(res.delta > 0)
    # The Huber path must remove most of the Cauchy noise.
    assert np.std(res.theta) < np.std(Y) * 0.5


def test_run_pipeline_median_boundaries():
    Y = np.linspace(0, 1, 50)
    tau = np.arange(50, dtype=np.float64)
    res = run_pipeline(tau, Y, halflife_trades=3, B=30, seed=0, T60=10)
    assert res.m[0] == Y[0]
    assert res.m[-1] == Y[-1]
    assert res.theta[0] == Y[0]
    assert res.theta[-1] == Y[-1]


def test_signature_vol_matches_table2_helper():
    from src.metrics import signature_vol

    rng = np.random.default_rng(4)
    tau = np.sort(rng.uniform(0.0, 500.0, 3000))
    x = np.cumsum(rng.standard_normal(3000) * 1e-4)
    for s in (1.0, 10.0):
        ref = s_second_diff_stats(tau, x, s_list=(s,))[s]["annualized_vol"]
        assert np.isclose(signature_vol(tau, x, s), ref)


def test_binomial_filter_is_a_normalised_clock_mean():
    from src.hf_filters import clock_binomial_filter

    tau = np.array([0.0, 1.0, 3.0, 4.0])
    d = np.array([1.0, 0.0, 1.0, 0.0])
    p = clock_binomial_filter(tau, d, 2.0)
    w = 2.0 ** (-(tau[-1] - tau) / 2.0)
    assert np.isclose(p[-1], np.sum(w * d) / np.sum(w))


def test_pipeline_tail_draws_are_reproducible():
    Y, _ = nonstationary_cauchy(T=500, seed=3)
    tau = np.arange(len(Y), dtype=np.float64) * 0.2
    a = run_pipeline(tau, Y, B=100, seed=9, T60=100)
    b = run_pipeline(tau, Y, B=100, seed=9, T60=100)
    assert np.array_equal(a.theta, b.theta)
    assert np.array_equal(a.m, b.m)
