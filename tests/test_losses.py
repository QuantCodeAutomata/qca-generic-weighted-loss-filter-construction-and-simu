"""Unit tests for loss functions and weighted solvers."""

from __future__ import annotations

import numpy as np
import pytest

from src.losses import (
    absolute_loss,
    bernoulli_logloss,
    huber_loss,
    squared_loss,
    weighted_bernoulli_prob,
    weighted_huber,
    weighted_mean,
    weighted_median,
)


def test_squared_loss_values():
    y = np.array([0.0, 1.0, 2.0])
    assert np.allclose(squared_loss(1.0, y), [0.5, 0.0, 0.5])


def test_absolute_loss_paper_convention():
    y = np.array([2.0, -2.0, 0.0])
    # L(theta, y) = |y - theta| - |y|  -> at theta=0 it is zero
    assert np.allclose(absolute_loss(0.0, y), 0.0)


def test_huber_loss_quadratic_and_linear_regions():
    y = np.array([0.0, 2.0])
    theta = 0.0
    delta = 1.0
    # residuals are 0 (quadratic region) and 2 (linear region)
    vals = huber_loss(theta, y, delta)
    assert np.isclose(vals[0], 0.0)
    assert np.isclose(vals[1], 1.0 * 2.0 - 0.5)  # delta*|r| - 0.5*delta^2


def test_bernoulli_logloss_known_values():
    y = np.array([1.0, 0.0])
    loss = bernoulli_logloss(0.5, y)
    assert np.allclose(loss, -np.log(0.5))


def test_weighted_mean_matches_numpy():
    y = np.random.default_rng(0).standard_normal(100)
    w = np.random.default_rng(1).random(100)
    assert np.isclose(weighted_mean(y, w), np.sum(w * y) / w.sum())


def test_weighted_median_uniform_matches_numpy_median_odd():
    y = np.array([1.0, 3.0, 7.0])
    w = np.ones(3) / 3.0
    assert np.isclose(weighted_median(y, w, interpolation="lower"), 3.0)


def test_weighted_median_hazen_plateau():
    # Equal weights on {0, 10}: Hazen interpolation gives midpoint 5.0
    y = np.array([0.0, 10.0])
    w = np.array([0.5, 0.5])
    out = weighted_median(y, w, interpolation="hazen")
    assert np.isclose(out, 5.0)


def test_weighted_median_hazen_interpolates_unequal_weights():
    # Jump midpoints 0.15, 0.45, 0.80 -> 2 + 0.05 / 0.35.
    y = np.array([1.0, 2.0, 3.0])
    w = np.array([0.3, 0.3, 0.4])
    assert np.isclose(weighted_median(y, w), 2.0 + 0.05 / 0.35)


def test_weighted_median_lower_selects_smallest():
    y = np.array([0.0, 10.0])
    w = np.array([0.5, 0.5])
    out = weighted_median(y, w, interpolation="lower")
    assert out == 0.0


def test_weighted_median_collapse_ties():
    y = np.array([1.0, 1.0, 2.0, 3.0])
    w = np.array([0.25, 0.25, 0.25, 0.25])
    # With ties collapsed, mass at 1 is 0.5, so lower interp returns 1
    out = weighted_median(y, w, interpolation="lower")
    assert out == 1.0


def test_weighted_median_zero_weight_dropped():
    y = np.array([1.0, 2.0, 1000.0])
    w = np.array([0.4, 0.6, 0.0])
    out = weighted_median(y, w, interpolation="lower")
    assert out == 2.0


def test_weighted_huber_matches_mean_in_quadratic_region():
    # If delta is huge the Huber solution equals the weighted mean.
    y = np.random.default_rng(0).standard_normal(50)
    w = np.ones(50) / 50
    theta_hub = weighted_huber(y, w, delta=1e6)
    theta_mean = weighted_mean(y, w)
    assert np.isclose(theta_hub, theta_mean, atol=1e-6)


def test_weighted_huber_robust_to_outliers():
    y = np.concatenate([np.zeros(99), np.array([1e6])])
    w = np.ones_like(y) / len(y)
    theta_hub = weighted_huber(y, w, delta=1.0)
    # outlier contributes a bounded influence
    assert abs(theta_hub) < 1.0


def test_weighted_bernoulli_prob_clipped():
    y = np.array([0.0, 0.0, 0.0])
    w = np.array([1.0, 1.0, 1.0])
    assert weighted_bernoulli_prob(y, w) == pytest.approx(1e-12)
    y1 = np.ones(3)
    assert weighted_bernoulli_prob(y1, w) == pytest.approx(1 - 1e-12)
