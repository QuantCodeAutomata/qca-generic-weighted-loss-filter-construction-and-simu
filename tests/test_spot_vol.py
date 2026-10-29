"""Small checks for the spot-volatility building blocks."""

import numpy as np

from src.spot_vol import (
    annualized_percent,
    blend_open,
    clock_mixture_variance,
    diurnal_variance,
    fit_diurnal,
    hyperbolic_spot,
    increments,
    trade_mixture_variance,
    uniform_spot_grid,
)


def test_increments_skip_the_first_second():
    tau = np.array([0.0, 0.4, 1.2])
    x = np.array([0.0, 0.0, 0.01])
    z, dt = increments(tau, x, s=1.0)
    assert z[1] == 0.0
    assert dt[2] == np.diff(tau)[-1]
    assert z[2] > 0.0


def test_uniform_grid_is_sum_over_h():
    tau = np.array([0.0, 0.5, 1.5])
    z = np.array([0.0, 0.2, 0.3])
    g = uniform_spot_grid(tau, z, h=1.0, session=2.0)
    assert np.isclose(g[1], 0.2)
    assert np.isclose(g[2], 0.3)


def test_hyperbolic_matches_direct_ratio():
    rng = np.random.default_rng(0)
    n = 25
    z = rng.random(n)
    z[0] = 0.0
    dt = rng.random(n)
    dt[0] = 0.0
    level = np.linspace(1.0, 2.0, n)
    a = 4.0
    spot = hyperbolic_spot(z, dt, level, a=a)
    t = 17
    j = np.arange(t + 1)
    num = np.sum((z[t - j] / level[t - j]) / (j + a))
    den = np.sum(dt[t - j] / (j + a))
    assert np.isclose(spot[t], level[t] * num / den)


def test_constant_variance_is_flat():
    tau = np.linspace(0.0, 100.0, 501)
    dt = np.diff(tau, prepend=tau[0])
    v = 2e-8
    z = v * dt
    z[0] = 0.0
    spot = hyperbolic_spot(z, dt, np.ones(len(z)), a=10.0)
    assert np.allclose(spot[50:], v, rtol=1e-8)
    # Same annualization as integrating a flat rate over the session.
    assert np.isclose(
        annualized_percent(np.array([v]), session=23_400.0)[0],
        100.0 * np.sqrt(252.0 * v * 23_400.0),
    )


def test_diurnal_fit_recovers_quadratic_log():
    u = np.arange(23_401, dtype=np.float64)
    t = u / 23_400.0
    var = np.exp(0.5 + 0.3 * t - 0.1 * t * t)
    coef = fit_diurnal(var, min_u=10.0)
    got = diurnal_variance(u[100:], coef)
    assert np.allclose(got, var[100:], rtol=1e-6)


def test_open_blend_starts_at_the_background():
    tau = np.array([0.0, 15.0])
    out = blend_open(tau, np.array([np.nan, 4.0]), background=2.0)
    assert out[0] == 2.0
    assert out[1] == 4.0


def test_one_component_mixture_matches_manual_weights():
    tau = np.array([0.0, 1.0, 2.0, 4.0])
    dt = np.array([0.0, 1.0, 1.0, 2.0])
    z = np.array([0.0, 0.1, 0.2, 0.4])
    level = np.ones(4)
    decay = np.array([np.log(2.0) / 2.0])
    out = clock_mixture_variance(z, dt, np.array([1.0]), decay, level)
    age = 4.0 - tau
    w = 2.0 ** (-age / 2.0)
    assert np.isclose(out[-1], np.sum(w * z) / np.sum(w * dt))

    lam = np.array([0.5])
    out_t = trade_mixture_variance(z, dt, np.array([1.0]), lam, level)
    j = np.arange(3, -1, -1)
    wt = lam[0] ** j
    assert np.isclose(out_t[-1], np.sum(wt * z) / np.sum(wt * dt))


def test_open_blend_uses_the_unadjusted_filter_before_15_seconds():
    from src.spot_vol import diurnal_hyperbolic_spot

    tau = np.linspace(0.0, 30.0, 61)
    dt = np.diff(tau, prepend=tau[0])
    v = 3e-8
    z = v * dt
    z[0] = 0.0
    level = np.full(len(tau), 4.0 * v)
    level[tau < 15.0] = 100.0 * v  # a diurnal level that must not enter u < 15
    out = diurnal_hyperbolic_spot(tau, z, dt, level, background=2.0 * v, a=5.0)
    i = int(np.searchsorted(tau, 7.5))
    w = tau[i] / 15.0
    raw = hyperbolic_spot(z[: i + 1], dt[: i + 1], np.ones(i + 1), a=5.0)[-1]
    assert np.isclose(out[i], (1.0 - w) * 2.0 * v + w * raw)


def test_qlike_selection_prefers_the_right_memory():
    from src.spot_vol import qlike_total, select_mandelbrot_a

    rng = np.random.default_rng(3)
    n = 40_000
    tau = np.cumsum(rng.exponential(0.5, n))
    tau -= tau[0]
    tau = tau[tau <= 2_000.0]
    n = len(tau)
    dt = np.diff(tau, prepend=tau[0])
    # Variance that switches level every 200 seconds.
    v = 1e-8 * np.where((tau // 200.0) % 2 == 0, 1.0, 6.0)
    z = v * dt * rng.chisquare(1, n)
    z[0] = 0.0
    day = {"tau": tau, "Z": z, "dt": dt, "level": np.ones(n), "background": 1e-8,
           "target": uniform_spot_grid(tau, z, session=2_000.0)}
    a_hat, path = select_mandelbrot_a([day], grid=np.geomspace(2.0, 20_000.0, 9), n_refine=6)
    vals = dict(path)
    assert qlike_total([day], a_hat) <= min(vals.values()) + 1e-9
    assert 2.0 < a_hat < 20_000.0
