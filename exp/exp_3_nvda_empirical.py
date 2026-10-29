"""
Experiment 3
============

NVDA, the twelve sessions of 1–16 October 2024, on the paper's cleaning of
the consolidated tape (16 October: T = 1,521,339). The five filters of
Section 4: median, noise scale (with the Section 4.6 startup from the
previous five openings), Huber threshold, Huber filter, and the directional
outlier probability ``q_t``.

Per session: Tables 1–3 (Table 2 with the Monte Carlo ``c_{t,s}`` of each
filter), the off-exchange share of trades and of outliers (Section 4.2), the
residual concentration and Hill checks of Section 4.4.1, the largest
one-trade moves (Section 4.4), the Table 3 confidence intervals, and the
Section 4.7 direction analysis (Figures 12 and 13, the ``q_t`` half-life by
predictive Bernoulli log-likelihood).

Across sessions: the threshold paths (Figures 6 and 7), the price snapshots
of Figure 8, the half-life selections of Section 4.3 (check loss of
``b_{t,60}``) and Section 4.4 (k-trade-ahead Huber forecast loss).

Stage ``spot``: the diurnal spot volatility (Sections 4.5.2–4.6), QLIKE
choice of ``a``, Figures 9, 10 and 11.

Stage ``sensitivity`` (16 October): the Table 1 Huber column under the
rounding tie rule of the median, the outlier count of the head-only filter
(B = 0), the opening threshold ``δ_0`` under other presample definitions,
and the Hill index on three subsample schemes.

Outputs go to ``results/`` with the prefix ``exp3_``: ``exp3_summary.json``
(stage ``days``), ``exp3_spot_summary.json``, ``exp3_sensitivity.json`` and
the figures; the run log is ``results/exp3.log``. ``--stages`` selects
stages (default: all).
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.data_loader import load_day, previous_trading_dates
from exp.common import RESULTS, jsonify, log_to_results, two_panel
from src import spot_pipeline as sp
from src.config import (
    CI_ALPHAS,
    DAMPING_PATHS,
    DAMPING_WINDOW_LAGS,
    DELTA0_MINUTE_LENGTHS_S,
    DELTA0_PUBLISHED_CENTS,
    DELTA0_SEEDS,
    DIRECTION_BIN_S,
    DIRECTION_GROUP_CENTS,
    EXP3_WORKERS,
    FIGURE6_HALF_LIVES_S,
    FIGURE8_WINDOW_LENGTH_S,
    FIGURE8_WINDOWS_S,
    FIGURE12_WINDOW_S,
    FORECAST_K,
    HILL_BLOCKS,
    HILL_SUBSAMPLE_SIZE,
    HILL_SUBSAMPLES,
    HILL_SYSTEMATIC_STEP,
    NVDA_SEED,
    NVDA_TICKER,
    OCT16,
    OFF_EXCHANGE_ID,
    PAPER_DATES,
    PRESAMPLE_OPENINGS,
    PRICE_HALF_LIFE_GRID,
    PRICE_HALF_LIFE_TRADES,
    PSI_STAR_LAGS,
    Q_HALF_LIFE_GRID_S,
    SCALE_HALF_LIFE_GRID_S,
    SCALE_HALF_LIFE_S,
    SCALE_LOSS_LAG_S,
    SESSION_SECONDS,
    STARTUP_S,
    TABLE1_K,
    TABLE2_S,
    TAIL_DRAWS,
)
from src.empirical_pipeline import opening_abs_residuals, run_pipeline
from src.hf_filters import (clock_time_scale, damping_correction, exponential_huber, outlier_predictive_loglik,
                            simulated_damping, threshold_from_scale)
from src.metrics import hill_tail_index, k_trade_diff_stats, psi_star_cents, s_second_diff_stats
from src.spot_vol import onto_seconds

PREFIX = "exp3_"
TICKER = NVDA_TICKER
STAGES = ("days", "spot", "sensitivity")


def _out(name: str):
    return RESULTS / f"{PREFIX}{name}"


def _head_weights() -> np.ndarray:
    lam = 2.0 ** (-1.0 / PRICE_HALF_LIFE_TRADES)
    return (1.0 - lam) * lam ** np.arange(PSI_STAR_LAGS)


def _clock(seconds_after_open: float) -> str:
    t = int(round(9.5 * 3600 + seconds_after_open))
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def _auction_print(df, i: int) -> dict:
    row = df.iloc[i]
    return {"price": float(row["price"]), "size": float(row["size"]), "tau_s": float(row["tau_s"])}


def _hill_random_subsamples(r: np.ndarray) -> list:
    rng = np.random.default_rng(NVDA_SEED)
    return [hill_tail_index(r[rng.choice(len(r), HILL_SUBSAMPLE_SIZE, replace=False)])
            for _ in range(HILL_SUBSAMPLES)]


def _presample(date: str) -> np.ndarray:
    parts = []
    for prev in previous_trading_dates(date, PRESAMPLE_OPENINGS):
        loaded = load_day(TICKER, prev)
        parts.append(
            opening_abs_residuals(
                loaded.df["log_price"].to_numpy(),
                loaded.df["tau_open"].to_numpy(),
            )
        )
    if not parts:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(parts)


def _cents_table(stats: dict) -> dict:
    """Price differences are in dollars; Table 1 prints cents."""
    out = {}
    for k, row in stats.items():
        out[str(k)] = {
            "mad_cents": row["mad_scaled"] * 100.0,
            "rms_cents": row["rms_scaled"] * 100.0,
            "acf": row["acf1"],
            "hill": row["hill"],
        }
    return out


def _table2(tau_open, logs, c_for) -> dict:
    out = {}
    for name, log_path in logs.items():
        raw = s_second_diff_stats(tau_open, log_path, s_list=TABLE2_S, c_by_s=c_for(name))
        out[name] = {
            str(s): {
                "mad_cents": row["mad_scaled"] * 100.0,
                "annualized_vol_pct": row["annualized_vol"],
                "hill": row["hill"],
                "adj_return_corr": row["adj_return_corr"],
            }
            for s, row in raw.items()
        }
    return out


def _table3(result, price) -> dict:
    """Table 3 residuals in price units, ``100 (p_t - X^_t)`` cents; scores and
    psi* use the same units (the Huber threshold is ``delta_{t-1} p_t``)."""
    w = _head_weights()
    n_lags = len(w)
    out = {}
    for name, x, kind in (("median", result.m, "sign"), ("huber", result.theta, "huber"),
                          ("linear", result.linear, "linear")):
        e = (price - np.exp(x)) * 100.0
        if kind == "huber":
            ps = psi_star_cents(price, x, result.delta_prev)
            acf, psi_star = ps["score_acf"], ps["psi_star_cents"]
        else:
            psi = np.sign(result.Y - x) if kind == "sign" else e  # exp(m) != p_t in floating point
            z = psi - psi.mean()
            gam = np.array([np.dot(z[: len(z) - k], z[k:]) / len(z) for k in range(n_lags)])
            acf = gam / gam[0]
            G = gam[np.abs(np.subtract.outer(np.arange(n_lags), np.arange(n_lags)))]
            psi_star = float(np.sqrt(w @ G @ w)) if kind == "linear" else float("nan")
        out[name] = {
            "mean_abs_resid_cents": float(np.mean(np.abs(e))),
            "sd_resid_cents": float(np.std(e)),
            "hill_score": float("inf") if kind != "linear" else hill_tail_index(e),
            "score_acf": {str(k): float(acf[k]) for k in TABLE1_K},
            "psi_star_cents": psi_star,
        }
    return out


def _exceed_share(e, d, eta, w) -> float:
    """``(1/T) Σ_t 1(g_t(η) > 0)``, ``g_t(η) = Σ_j w_j ψ(ε̂_{t−j} − η)`` (Section 4.4.1).

    ``d`` is the per-trade threshold of the score (``inf`` for the linear filter).
    """
    s = np.clip(e - eta, -d, d)
    g = np.convolve(s, w)[: len(s)]
    return float(np.mean(g > 0.0))


def _error_intervals(e, d, w, alphas=CI_ALPHAS) -> dict:
    """Quantiles of ``θ_t − X_t`` from ``P(θ_t − X_t > η) ≈ (1/T) Σ_t 1(g_t(η) > 0)``.

    Bisection on η for each tail probability; the probability is
    nonincreasing in η because the score is.
    """
    def solve(p):
        lo, hi = -5.0, 5.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if _exceed_share(e, d, mid, w) > p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    return {f"{a:.2f}": [solve(1.0 - a / 2.0), solve(a / 2.0)] for a in alphas}


def _section_441(result, price, exchange, tau) -> dict:
    """Venue shares (Section 4.2), residual concentration and the Hill checks of
    Section 4.4.1, the largest one-trade moves (Section 4.4) and the
    confidence intervals of Table 3."""
    r = result.resid_hub
    r2, ar = r * r, np.abs(r)
    order = np.argsort(r2)
    out_mask = (result.d_pos + result.d_neg) > 0
    off = exchange == OFF_EXCHANGE_ID
    sub = _hill_random_subsamples(r)
    keep = np.ones(len(r), dtype=bool)
    keep[order[-2:]] = False
    t_star = int(np.argmax(ar))
    d_lin = np.diff(np.exp(result.linear))
    d_hub = np.diff(np.exp(result.theta))
    k_hub = int(np.argmax(np.abs(d_hub)))
    w = _head_weights()
    e = (price - np.exp(result.theta)) * 100.0
    e_lin = (price - np.exp(result.linear)) * 100.0
    d = result.delta_prev * price * 100.0
    return {
        "off_exchange_share_trades": float(off.mean()),
        "off_exchange_share_outliers": float(off[out_mask].mean()) if out_mask.any() else float("nan"),
        "largest_share_sq": float(r2[order[-1]] / r2.sum()),
        "largest_share_abs": float(ar[order[-1]] / ar.sum()),
        "top1000_share_sq": float(r2[order[-1000:]].sum() / r2.sum()),
        "top1000_share_abs": float(ar[order[-1000:]].sum() / ar.sum()),
        "hill_without_two_largest": hill_tail_index(r[keep]),
        "hill_subsamples_25k": {"min": float(np.min(sub)), "max": float(np.max(sub))},
        "worst_trade": {"price": float(price[t_star]), "tau_s": float(tau[t_star]),
                        "linear_filter": float(np.exp(result.linear[t_star])),
                        "huber_filter": float(np.exp(result.theta[t_star])),
                        "linear_share_sq_1trade": float(d_lin[t_star - 1] ** 2 / np.sum(d_lin ** 2))},
        "huber_largest_move": {"share_sq_1trade": float(d_hub[k_hub] ** 2 / np.sum(d_hub ** 2)),
                               "tau_s": float(tau[k_hub + 1]), "cents": float(d_hub[k_hub] * 100.0)},
        "error_intervals_cents": {"huber": _error_intervals(e, d, w),
                                  "linear": _error_intervals(e_lin, np.full(len(e_lin), np.inf), w)},
    }


def _direction(date, tau, price, result) -> dict:
    """Section 4.7: where the outliers sit relative to the moving Huber price."""
    filt = np.exp(result.theta) * 100.0
    edges = np.arange(0.0, SESSION_SECONDS + 1e-9, DIRECTION_BIN_S)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        i0 = int(np.searchsorted(tau, lo, side="left"))
        i1 = int(np.searchsorted(tau, hi, side="left")) - 1
        if i1 <= i0:
            continue
        sl = slice(i0, i1 + 1)
        up = int(result.d_pos[sl].sum())
        dn = int(result.d_neg[sl].sum())
        if up + dn == 0:
            continue
        bins.append({"start_s": float(lo), "up": up, "down": dn, "prob_above": up / (up + dn),
                     "change_cents": float(filt[i1] - filt[i0])})
    x = np.array([b["change_cents"] for b in bins])
    y = np.array([b["prob_above"] for b in bins])
    groups = []
    g = DIRECTION_GROUP_CENTS
    for lo in np.arange(np.floor(x.min() / g) * g, x.max() + g, g):
        m = (x >= lo) & (x < lo + g)
        if m.sum() >= 2:
            groups.append({"lo_cents": float(lo), "n": int(m.sum()), "mean": float(y[m].mean()),
                           "se": float(y[m].std(ddof=1) / np.sqrt(m.sum()))})
    loglik = {str(h): outlier_predictive_loglik(tau, result.d_pos, result.d_neg, h, result.T60)
              for h in Q_HALF_LIFE_GRID_S}
    best = max(loglik, key=loglik.get)
    _plot_direction(date, tau, price, result, bins, groups, _out(f"{date}_figure12.png"))
    _plot_figure13(date, tau, result.q, bins, _out(f"{date}_figure13.png"))
    return {
        "n_bins": len(bins),
        "corr_prob_vs_change": float(np.corrcoef(x, y)[0, 1]) if len(bins) > 2 else float("nan"),
        "groups_10c": groups,
        "q_halflife_loglik": loglik,
        "q_halflife_best_s": float(best),
        "outliers_up": int(result.d_pos.sum()),
        "outliers_down": int(result.d_neg.sum()),
    }


def run_single_day(date: str) -> dict:
    print(f"\n--- {TICKER} {date} ---", flush=True)
    presample = _presample(date)
    loaded = load_day(TICKER, date)
    df = loaded.df
    Y = df["log_price"].to_numpy()
    price = df["price"].to_numpy()
    tau_open = df["tau_open"].to_numpy()
    result = run_pipeline(
        df["tau_s"].to_numpy(),
        Y,
        tau_open=tau_open,
        presample_abs=presample,
        T60=loaded.T60,
        halflife_trades=PRICE_HALF_LIFE_TRADES,
        B=TAIL_DRAWS,
        seed=NVDA_SEED,
    )
    cents = result.delta * price * 100.0
    hill = hill_tail_index(result.resid_hub)
    outlier_share = float((result.d_pos + result.d_neg).mean())
    paths = {
        "median": np.exp(result.m),
        "huber": np.exp(result.theta),
        "linear": np.exp(result.linear),
        "trades": price,
    }
    logs = {
        "median": result.m,
        "huber": result.theta,
        "linear": result.linear,
        "trades": Y,
    }
    table1 = {name: _cents_table(k_trade_diff_stats(px)) for name, px in paths.items()}
    c_lin = {float(s): damping_correction(tau_open, float(s), halflife=PRICE_HALF_LIFE_TRADES,
                                          window=DAMPING_WINDOW_LAGS) for s in TABLE2_S}
    table2_analytic = _table2(tau_open, logs, lambda n: None if n == "trades" else c_lin)
    vol15 = table2_analytic["huber"][str(float(TABLE2_S[-1]))]["annualized_vol_pct"] / 100.0
    sigma = vol15 / np.sqrt(252.0 * SESSION_SECONDS)
    c_mc = simulated_damping(tau_open, [float(s) for s in TABLE2_S], sigma, result.delta,
                             halflife=PRICE_HALF_LIFE_TRADES, n_paths=DAMPING_PATHS, seed=NVDA_SEED)
    table2 = _table2(tau_open, logs, lambda n: None if n == "trades" else c_mc[n])
    c_note = f"Monte Carlo c per filter, M={DAMPING_PATHS}"
    c1 = _table2(tau_open, {"huber": result.theta}, lambda n: None)["huber"]
    c_effect = {s: table2["huber"][s]["annualized_vol_pct"] / c1[s]["annualized_vol_pct"] - 1.0
                for s in ("1.0", "10.0")}

    _plot_prices(date, tau_open, price, result, _out(f"{date}_prices.png"))
    _plot_threshold(date, tau_open, cents, _out(f"{date}_threshold.png"))
    _plot_signature(date, table2, _out(f"{date}_signature.png"))
    _plot_q(date, tau_open, result.q, _out(f"{date}_q.png"))

    summary = {
        "date": date,
        "T": int(len(df)),
        "raw_count": int(loaded.raw_count),
        "T60": int(loaded.T60),
        "n_presample": int(presample.size),
        "opening_print": _auction_print(df, 0),
        "closing_print": _auction_print(df, -1),
        "hill_resid_hub": hill,
        "outlier_share": outlier_share,
        "mean_abs_resid_cents": float(np.mean(np.abs(result.resid_hub) * price * 100.0)),
        "delta_cents_open": float(cents[0]),
        "delta_cents_median": float(np.median(cents)),
        "median_resid_zero_share": float(np.mean(result.resid_med == 0.0)),
        "table1": table1,
        "table2": table2,
        "table2_c_method": c_note,
        "table2_analytic_linear_c": table2_analytic,
        "huber_c_effect": c_effect,
        "table3": _table3(result, price),
        "section_441": _section_441(result, price, df["exchange"].to_numpy(), tau_open),
        "direction": _direction(date, tau_open, price, result),
    }
    summary["halflife_scale_loss"] = _scale_halflife_loss(tau_open, result, presample)
    summary["halflife_price_loss"] = _price_halflife_loss(Y, result)
    summary["_delta_seconds"] = onto_seconds(tau_open, cents)
    if date == OCT16:
        summary["figure6"] = _figure6(tau_open, price, result, presample)
        _plot_figure8(tau_open, price, result, _out(f"{date}_figure8.png"))
    print(
        f"{date} T={summary['T']} Hill={hill:.3f} outliers={outlier_share:.4%} "
        f"mean|e|={summary['mean_abs_resid_cents']:.3f}c "
        f"delta open={cents[0]:.2f}c median={np.median(cents):.2f}c",
        flush=True,
    )
    return summary


def _scale_halflife_loss(tau, result, presample) -> dict:
    """Section 4.3: check loss of ``b_{t,60}`` (the scale at the latest trade at
    least a minute earlier) for the nonzero ``|ε̃_t|``, by half-life."""
    a = np.abs(result.resid_med)
    ref = np.searchsorted(tau, tau - SCALE_LOSS_LAG_S, side="right") - 1
    use = (ref >= 0) & (a > 0.0)
    pre = presample[np.isfinite(presample) & (presample > 0.0)]
    out = {}
    for h in SCALE_HALF_LIFE_GRID_S:
        b = clock_time_scale(tau, a, pre, halflife_s=h, startup_s=STARTUP_S)
        u = a[use] - b[ref[use]]
        out[str(h)] = float(np.sum(u * (0.5 - (u < 0.0))))
    return out


def _price_halflife_loss(Y, result) -> dict:
    """Section 4.4: ``Σ_t H_{δ_{t−k}}(Y_t − θ_{t−k})`` for each half-life and k,
    thresholds held at the day's path and the filter started from ``m_t``."""
    out = {}
    for h in PRICE_HALF_LIFE_GRID:
        theta = exponential_huber(Y, result.delta, result.m, halflife=h, B=TAIL_DRAWS, seed=NVDA_SEED + 7,
                                  delta0=float(result.delta_prev[0]))
        row = {}
        for k in FORECAST_K:
            u = Y[k:] - theta[:-k]
            d = result.delta[:-k]
            au = np.abs(u)
            row[str(k)] = float(np.sum(np.where(au <= d, 0.5 * u * u, d * au - 0.5 * d * d)))
        out[str(h)] = row
    return out


def _figure6(tau, price, result, presample) -> dict:
    """δ_t on one day for three scale half-lives, all starting from the presample."""
    a = np.abs(result.resid_med)
    pre = presample[np.isfinite(presample) & (presample > 0.0)]
    paths = {}
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for h, col in zip(FIGURE6_HALF_LIVES_S, ("0.6", "tab:red", "tab:blue")):
        d = threshold_from_scale(clock_time_scale(tau, a, pre, halflife_s=h, startup_s=STARTUP_S)) * price * 100.0
        sec = onto_seconds(tau, d)
        paths[str(h)] = {"start_cents": float(d[0]), "median_cents": float(np.median(d)),
                         "mad_of_changes": float(np.mean(np.abs(np.diff(sec[int(STARTUP_S):]))))}
        ax.plot(9.5 + np.arange(len(sec)) / 3600.0, sec, color=col, lw=0.8, label=f"half-life {h:g} s")
    ax.set_xlabel("time of day (hours)")
    ax.set_ylabel("δ_t (cents)")
    ax.set_title("NVDA 16 Oct 2024: Huber threshold for three scale half-lives (Figure 6)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(_out(f"{OCT16}_figure6.png"), dpi=140)
    plt.close(fig)
    base = paths[str(SCALE_HALF_LIFE_S)]["mad_of_changes"]
    for v in paths.values():
        v["movement_relative_to_4min"] = v["mad_of_changes"] / base
    return paths


def _plot_figure8(tau, price, result, path):
    """Full day and three 5-second windows: 13:00:08, 13:28:15 (with the median
    filter) and the window around the largest Huber residual."""
    t_star = int(np.argmax(np.abs(result.resid_hub)))
    w1, w2 = FIGURE8_WINDOWS_S
    L = FIGURE8_WINDOW_LENGTH_S
    windows = ((None, "full trading day"), (w1, f"{L:g} s window at {_clock(w1)}"),
               (w2, f"{L:g} s window at {_clock(w2)}, with the median filter"),
               (float(tau[t_star]) - 2.0, f"{L:g} s window around the ${price[t_star]:.2f} trade"))
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    for ax, (lo, title) in zip(axes, windows):
        if lo is None:
            m = slice(None, None, max(1, len(tau) // 30_000))
            x = 9.5 + tau[m] / 3600.0
        else:
            m = (tau >= lo) & (tau <= lo + L)
            x = tau[m] - lo
        ax.plot(x, price[m], ".", ms=2, color="0.65", label="trades")
        ax.plot(x, np.exp(result.theta[m]), color="tab:blue", lw=1.4 if lo is not None else 0.5, label="Huber filter")
        ax.plot(x, np.exp(result.linear[m]), color="tab:orange", lw=0.7 if lo is not None else 0.5, label="linear filter")
        if lo == w2:
            ax.plot(x, np.exp(result.m[m]), color="tab:green", lw=0.8, drawstyle="steps-post", label="median filter")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("time of day (hours)" if lo is None else "seconds into the window")
        ax.set_ylabel("price (dollars)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _halflife_tables(per_day) -> dict:
    scale = {h: sum(d["halflife_scale_loss"][str(h)] for d in per_day) for h in SCALE_HALF_LIFE_GRID_S}
    best_s = min(scale, key=scale.get)
    price = {k: {h: sum(d["halflife_price_loss"][str(h)][str(k)] for d in per_day) for h in PRICE_HALF_LIFE_GRID}
             for k in FORECAST_K}
    out = {
        "scale": {"pooled_loss_relative_to_min": {str(h): v / scale[best_s] for h, v in scale.items()},
                  "best_s": best_s},
        "price": {str(k): {"pooled_loss_relative_to_min": {str(h): v / min(row.values()) for h, v in row.items()},
                           "best_halflife": min(row, key=row.get),
                           "loss_3_over_2": row[3.0] / row[2.0]}
                  for k, row in price.items()},
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    xs = list(SCALE_HALF_LIFE_GRID_S)
    axes[0].semilogx(xs, [scale[h] / scale[best_s] for h in xs], "o-")
    axes[0].set_xlabel("scale half-life (seconds)")
    axes[0].set_ylabel("pooled check loss / minimum")
    axes[0].set_title("Section 4.3: noise-scale half-life")
    for k in FORECAST_K:
        row = price[k]
        axes[1].plot(list(PRICE_HALF_LIFE_GRID), [row[h] / min(row.values()) for h in PRICE_HALF_LIFE_GRID], "o-",
                     label=f"k = {k}")
    axes[1].set_xlabel("price half-life (trades)")
    axes[1].set_ylabel("pooled Huber forecast loss / minimum")
    axes[1].set_title("Section 4.4: price half-life")
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(_out("halflife_selection.png"), dpi=140)
    plt.close(fig)
    return out


def _plot_figure7(per_day) -> dict:
    rows = np.vstack([d["_delta_seconds"] for d in per_day])
    med = np.nanmedian(rows, axis=0)
    hours = 9.5 + np.arange(rows.shape[1]) / 3600.0
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for r in rows:
        ax.plot(hours, r, color="0.7", lw=0.5)
    ax.plot(hours, med, color="tab:red", lw=1.2, label="twelve-day median")
    ax.set_xlabel("time of day (hours)")
    ax.set_ylabel("δ_t (cents)")
    ax.set_title("NVDA, 12 sessions: Huber threshold through the day (Figure 7)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(_out("figure7_threshold_12days.png"), dpi=140)
    plt.close(fig)
    return {"median_first_minute_cents": float(np.nanmedian(med[:60])),
            "median_after_first_hour_cents": float(np.nanmedian(med[3600:])),
            "per_day_median_cents": [float(np.nanmedian(r[600:])) for r in rows]}


def spot_stage() -> dict:
    """Stage 7: diurnal spot volatility on the twelve sessions (Figures 9–11)."""
    sp.ensure_filtered(TICKER)
    res = sp.spot_ticker(TICKER, keep=True)
    i = PAPER_DATES.index(OCT16)
    two_panel([res["seconds_linear"][i], res["seconds_huber"][i]], [res["seconds_linear"], res["seconds_huber"]],
              ["linear filter", "Huber filter"], ["0.55", "tab:red"], _out("figure11_huber_linear.png"))
    sigs = np.stack([sp.scalars(TICKER, d)["signature"] for d in PAPER_DATES])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for k, name in enumerate(("trades", "linear filter", "Huber filter")):
        for day in sigs:
            axes[k].loglog(sp.SIG_S, day[k], color="0.75", lw=0.7)
        axes[k].loglog(sp.SIG_S, np.sqrt(np.mean(sigs[:, k] ** 2, axis=0)), color="tab:red", lw=1.6)
        axes[k].set_title(name)
        axes[k].set_xlabel("time lag s (seconds, log scale)")
        axes[k].grid(True, which="both", linestyle=":", alpha=0.6)
    axes[0].set_ylabel("annualized volatility (%, log scale)")
    fig.suptitle("NVDA, 12 sessions: individual days (grey) and twelve-day average (red), Figure 9")
    fig.tight_layout()
    fig.savefig(_out("figure9_signature.png"), dpi=140)
    plt.close(fig)
    rate = res["trades_per_second"]
    fitted, seconds = {}, {}
    for mode in ("trade", "clock"):
        alpha, loss = sp.fit_mixture(res["prepared"], mode, rate)
        seconds[mode], corr = sp.hepx_paths(res["prepared"], alpha, mode, rate)
        fitted[mode] = {"alpha": alpha.tolist(), "qlike": loss, "log_corr": corr}
        print(f"HExp {mode} alpha={np.round(alpha, 3).tolist()} log_corr={corr:.4f}", flush=True)
    two_panel([seconds["trade"][i], res["seconds_huber"][i]], [seconds["trade"], res["seconds_huber"]],
              ["HExp superposition", f"hyperbolic a={res['a_qlike']:.0f}"], ["0.55", "tab:red"],
              _out("figure10_hyperbolic_hepx.png"))
    avg = np.sqrt(np.mean(sigs ** 2, axis=0))
    return {
        "a_qlike": res["a_qlike"], "a_seconds": res["a_seconds"], "a_path": res["a_path"],
        "trades_per_second": rate, "linear_over_huber": res["ratio"],
        "median_pointwise_ratio": res["median_pointwise_ratio"], "jump_16oct": res["jump"],
        "per_day": res["per_day"],
        "hepx": {"rate_trades_per_second": rate, **fitted},
        "signature_12day": {name: dict(zip(map(str, sp.SIG_S), avg[k].tolist()))
                            for k, name in enumerate(("trades", "linear", "huber"))},
        "signature_raw_over_huber_0.1s_by_day": (sigs[:, 0, 0] / sigs[:, 2, 0]).tolist(),
    }


def _plot_prices(date, tau, price, result, path):
    step = max(1, len(price) // 25000)
    sl = slice(None, None, step)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(tau[sl], price[sl], ",", color="0.75", label="trades")
    ax.plot(tau[sl], np.exp(result.linear[sl]), lw=0.6, color="tab:orange", label="linear")
    ax.plot(tau[sl], np.exp(result.m[sl]), lw=0.6, color="tab:green", label="median")
    ax.plot(tau[sl], np.exp(result.theta[sl]), lw=0.6, color="tab:blue", label="Huber")
    ax.set_xlabel("seconds since 09:30 ET")
    ax.set_ylabel("NVDA price (dollars)")
    ax.set_title(f"NVDA {date}: trades, linear, median, Huber")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_threshold(date, tau, cents, path):
    step = max(1, len(cents) // 25000)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(tau[::step], cents[::step], lw=0.6, color="tab:red")
    ax.set_xlabel("seconds since 09:30 ET")
    ax.set_ylabel("Huber threshold (cents)")
    ax.set_title(f"NVDA {date}: delta_t")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_signature(date, table2, path):
    s_vals = [float(s) for s in TABLE2_S]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, marker in (("trades", "o"), ("linear", "s"), ("huber", "d")):
        ys = [table2[name][str(s)]["annualized_vol_pct"] for s in s_vals]
        ax.semilogx(s_vals, ys, marker=marker, label=name)
    ax.set_xlabel("lag s (seconds)")
    ax.set_ylabel("annualized volatility (percent)")
    ax.set_title(f"NVDA {date}: volatility signature")
    ax.legend()
    ax.grid(True, which="both", linestyle=":")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_direction(date, tau, price, result, bins, groups, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    lo, hi = FIGURE12_WINDOW_S
    m = (tau >= lo) & (tau <= hi)
    t_min = tau[m] / 60.0
    axes[0].plot(t_min, price[m], ",", color="0.6")
    up = m & (result.d_pos > 0)
    dn = m & (result.d_neg > 0)
    axes[0].plot(tau[up] / 60.0, price[up], ".", ms=3, color="tab:red", label="outliers above")
    axes[0].plot(tau[dn] / 60.0, price[dn], ".", ms=3, color="tab:blue", label="outliers below")
    axes[0].plot(t_min, np.exp(result.theta[m]), color="k", lw=0.7, label="Huber filter")
    axes[0].set_xlabel(f"minutes after 09:30 ({_clock(lo)[:5]} to {_clock(hi)[:5]})")
    axes[0].set_ylabel("price (dollars)")
    axes[0].legend(fontsize=8)
    x = [b["change_cents"] for b in bins]
    y = [b["prob_above"] for b in bins]
    axes[1].plot(x, y, ".", color="0.7")
    gx = [g["lo_cents"] + 5.0 for g in groups]
    axes[1].errorbar(gx, [g["mean"] for g in groups], yerr=[g["se"] for g in groups], color="tab:red", lw=1)
    axes[1].set_xlabel("2 min price change (cents)")
    axes[1].set_ylabel("2 min empirical cond prob")
    fig.suptitle(f"NVDA {date}: outliers and the robust price (Figure 12)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_figure13(date, tau, q, bins, path):
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(tau / 3600.0, q, lw=0.4, color="0.4", label="filtered cond prob q_t")
    ax.plot([(b["start_s"] + 60.0) / 3600.0 for b in bins], [b["prob_above"] for b in bins], ".",
            color="tab:red", label="2 min empirical cond prob")
    ax.axhline(0.5, color="k", ls="--", lw=0.7)
    ax.set_ylim(0, 1)
    ax.set_xlabel("hours after 09:30")
    ax.set_title(f"NVDA {date}: directional conditional probability (Figure 13)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_q(date, tau, q, path):
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(tau, q, lw=0.4)
    ax.axhline(0.5, color="k", ls="--", lw=0.7)
    ax.set_ylim(0, 1)
    ax.set_xlabel("seconds since 09:30 ET")
    ax.set_ylabel("q_t")
    ax.set_title(f"NVDA {date}: directional outlier probability")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _presample_variant(date: str, **kw) -> tuple[np.ndarray, list]:
    """Opening residuals of the previous openings under one presample definition
    (keywords of ``opening_abs_residuals``), and a per-day breakdown."""
    parts, days = [], []
    for prev in previous_trading_dates(date, PRESAMPLE_OPENINGS):
        df = load_day(TICKER, prev).df
        r = opening_abs_residuals(df["log_price"].to_numpy(), df["tau_open"].to_numpy(), **kw)
        p0 = float(df["price"].iloc[0])
        parts.append(r)
        days.append({"date": prev, "n": int(r.size), "open_price": p0,
                     "one_cent_atom": float(np.log1p(0.01 / p0)),
                     "share_at_one_cent": float(np.mean(np.isclose(r, np.log1p(0.01 / p0), rtol=1e-3)))})
    return np.concatenate(parts), days


def _delta0(pre: np.ndarray, price0: float) -> dict:
    pre = pre[np.isfinite(pre) & (pre > 0.0)]
    b0 = float(np.sort(pre)[(pre.size + 1) // 2 - 1])
    return {"n": int(pre.size), "b0": b0, "implied_atom_price": 0.01 / b0,
            "delta0_cents": float(threshold_from_scale(np.array([b0]))[0]) * price0 * 100.0}


def _table1_huber(result) -> dict:
    return _cents_table(k_trade_diff_stats(np.exp(result.theta)))


def sensitivity_stage() -> dict:
    """16 October: the median tie rule, the head-only filter, the opening
    threshold and the Hill subsample scheme."""
    date = OCT16
    loaded = load_day(TICKER, date)
    df = loaded.df
    Y, price = df["log_price"].to_numpy(), df["price"].to_numpy()
    tau, tau_open = df["tau_s"].to_numpy(), df["tau_open"].to_numpy()
    price0 = float(price[0])
    presample = _presample(date)

    def pipeline(**kw):
        return run_pipeline(tau, Y, tau_open=tau_open, presample_abs=presample, T60=loaded.T60,
                            halflife_trades=PRICE_HALF_LIFE_TRADES, seed=NVDA_SEED, **kw)

    out = {"date": date, "opening_price": price0}
    exact = pipeline(B=TAIL_DRAWS)
    rounding = pipeline(B=TAIL_DRAWS, median_tie="rounding")
    out["median_tie"] = {
        "exact": {"table1_huber": _table1_huber(exact), "outliers": int((exact.d_pos + exact.d_neg).sum())},
        "rounding": {"table1_huber": _table1_huber(rounding),
                     "outliers": int((rounding.d_pos + rounding.d_neg).sum())},
    }
    del rounding
    head_only = pipeline(B=0)
    out["head_only_B0"] = {"outliers": int((head_only.d_pos + head_only.d_neg).sum()),
                           "outliers_B100": out["median_tie"]["exact"]["outliers"]}
    del head_only
    print(f"tie rule: rms k=1 exact {out['median_tie']['exact']['table1_huber']['1']['rms_cents']:.3f}c, "
          f"rounding {out['median_tie']['rounding']['table1_huber']['1']['rms_cents']:.3f}c; "
          f"outliers B=0 {out['head_only_B0']['outliers']}", flush=True)

    base, per_day = _presample_variant(date)
    paper_b0 = DELTA0_PUBLISHED_CENTS / 100.0 / price0 / float(threshold_from_scale(np.array([1.0]))[0])
    delta0 = {"published_cents": DELTA0_PUBLISHED_CENTS, "published_implied_atom_price": 0.01 / paper_b0,
              "pipeline_cents": float(exact.delta_prev[0]) * price0 * 100.0,
              "baseline": _delta0(base, price0), "presample_days": per_day,
              "tail_draws_seeds": {}, "minute_from_open_print": None, "minute_length_s": {}}
    for seed in DELTA0_SEEDS:
        pre, _ = _presample_variant(date, B=TAIL_DRAWS, seed=seed)
        delta0["tail_draws_seeds"][str(seed)] = _delta0(pre, price0)
    pre, _ = _presample_variant(date, from_open_print=True)
    delta0["minute_from_open_print"] = _delta0(pre, price0)
    for length in DELTA0_MINUTE_LENGTHS_S:
        pre, _ = _presample_variant(date, minute_s=length)
        delta0["minute_length_s"][str(length)] = _delta0(pre, price0)
    seeds = [v["delta0_cents"] for v in delta0["tail_draws_seeds"].values()]
    delta0["tail_draws_range_cents"] = [float(min(seeds)), float(max(seeds))]
    out["delta0"] = delta0
    print(f"delta0: baseline {delta0['baseline']['delta0_cents']:.3f}c, seeds {min(seeds):.3f}-{max(seeds):.3f}c, "
          f"published {DELTA0_PUBLISHED_CENTS}c", flush=True)

    r = exact.resid_hub
    offsets = np.linspace(0, HILL_SYSTEMATIC_STEP, HILL_SUBSAMPLES, endpoint=False).astype(int)
    starts = np.linspace(0, len(r) - HILL_SUBSAMPLE_SIZE, HILL_BLOCKS).astype(int)
    schemes = {
        "random_without_replacement": _hill_random_subsamples(r),
        "systematic": [hill_tail_index(r[o::HILL_SYSTEMATIC_STEP]) for o in offsets],
        "contiguous_blocks": [hill_tail_index(r[a:a + HILL_SUBSAMPLE_SIZE]) for a in starts],
    }
    out["hill_subsamples"] = {
        "full_sample": hill_tail_index(r),
        "systematic_step": HILL_SYSTEMATIC_STEP,
        "subsample_size": HILL_SUBSAMPLE_SIZE,
        **{k: {"values": v, "min": float(np.min(v)), "max": float(np.max(v))} for k, v in schemes.items()},
    }
    for k, v in schemes.items():
        print(f"Hill {k}: {np.min(v):.3f}-{np.max(v):.3f}", flush=True)
    return out


def days_stage() -> dict:
    with ProcessPoolExecutor(max_workers=EXP3_WORKERS) as pool:
        per_day = list(pool.map(run_single_day, PAPER_DATES))
    hills = [s["hill_resid_hub"] for s in per_day]
    shares = [s["outlier_share"] for s in per_day]
    aggregate = {
        "dates": PAPER_DATES,
        "hill_resid_hub_mean": float(np.mean(hills)),
        "hill_resid_hub_range": [float(np.min(hills)), float(np.max(hills))],
        "outlier_share_mean": float(np.mean(shares)),
        "direction_corr_by_day": {s["date"]: s["direction"]["corr_prob_vs_change"] for s in per_day},
        "figure7": _plot_figure7(per_day),
        "halflife_selection": _halflife_tables(per_day),
    }
    for s in per_day:
        s.pop("_delta_seconds", None)
    aggregate["per_day"] = per_day
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    args = parser.parse_args()
    log_to_results("exp3")
    stages = {"days": (days_stage, "summary.json"), "spot": (spot_stage, "spot_summary.json"),
              "sensitivity": (sensitivity_stage, "sensitivity.json")}
    for name in STAGES:
        if name not in args.stages:
            continue
        print(f"\n=== Stage {name} ===", flush=True)
        fn, file = stages[name]
        result = jsonify(fn())
        with open(_out(file), "w") as f:
            json.dump(result, f, indent=2)
    print(f"\nResults saved to {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
