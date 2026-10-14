"""Unit tests for weight schemes."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import digamma

from src.weights import (
    SuperpositionSpec,
    anchored_weights,
    exponential_cum_weight,
    exponential_inv_cum_weight,
    exponential_weights,
    hyperbolic_weights,
    uniform_weights,
    validate_weights,
)


def test_uniform_weights_sum_to_one():
    w = uniform_weights(t=20, ell=5)
    assert w.shape == (20,)
    assert np.isclose(w.sum(), 1.0)
    # First (ell+1)=6 entries equal 1/6
    assert np.allclose(w[:6], 1.0 / 6.0)
    assert np.all(w[6:] == 0.0)


def test_uniform_weights_t_le_ell():
    w = uniform_weights(t=4, ell=10)
    assert w.shape == (4,)
    assert np.isclose(w.sum(), 1.0)
    assert np.allclose(w, 0.25)


def test_exponential_weights_formula():
    t, lam = 50, 0.9
    w = exponential_weights(t, lam)
    n_t = (1.0 - lam ** t) / (1.0 - lam)
    for j in [0, 5, 10, 25]:
        assert np.isclose(w[j], lam ** j / n_t)
    assert np.isclose(w.sum(), 1.0)


def test_exponential_cum_and_inv():
    t, lam = 100, 0.9
    for p in [0.1, 0.5, 0.9, 0.999]:
        k = exponential_inv_cum_weight(p, t, lam)
        cw = exponential_cum_weight(k, t, lam)
        # Smallest k such that W_t(k) >= p
        assert cw >= p - 1e-12
        if k > 0:
            assert exponential_cum_weight(k - 1, t, lam) < p + 1e-12


def test_hyperbolic_weights_sum_and_formula():
    t, a = 30, 5.0
    w = hyperbolic_weights(t, a)
    assert np.isclose(w.sum(), 1.0)
    n_t = digamma(t + a) - digamma(a)
    assert np.isclose(w[0], (1.0 / a) / n_t)


def test_anchored_weights_bounds():
    w = anchored_weights(t=100, ell=10, alpha=0.5, T=1000)
    assert np.all(w >= 0.0)
    assert np.isclose(w.sum(), 1.0)
    # Alpha=1 should reduce to uniform
    w1 = anchored_weights(t=100, ell=10, alpha=1.0, T=1000)
    w_unif = uniform_weights(100, 10)
    assert np.allclose(w1, w_unif)


def test_superposition_mix_sums_to_one():
    spec = SuperpositionSpec(
        components=[lambda t: uniform_weights(t, 5),
                    lambda t: exponential_weights(t, 0.9)],
        alphas=[0.3, 0.7],
    )
    w = spec.weights(20)
    assert np.isclose(w.sum(), 1.0)
    assert np.all(w >= 0.0)


def test_superposition_bad_alphas():
    with pytest.raises(ValueError):
        SuperpositionSpec(components=[lambda t: uniform_weights(t, 5)], alphas=[0.5])
    with pytest.raises(ValueError):
        SuperpositionSpec(
            components=[lambda t: uniform_weights(t, 5),
                        lambda t: exponential_weights(t, 0.9)],
            alphas=[0.3, 0.3],
        )


def test_validate_weights_rejects_negative():
    with pytest.raises(AssertionError):
        validate_weights(np.array([-0.1, 0.5, 0.6]))


def test_edge_case_t_equals_one():
    for wfn in [
        lambda: uniform_weights(1, 15),
        lambda: exponential_weights(1, 0.9),
        lambda: hyperbolic_weights(1, 5.0),
    ]:
        w = wfn()
        assert w.shape == (1,)
        assert np.isclose(w[0], 1.0)
