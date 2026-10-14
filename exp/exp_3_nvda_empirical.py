"""
Experiment 3
============

Empirical NVDA Replication: High-Frequency Robust Price Filtering and
Volatility Estimation.

Scope note
----------
The paper's full study uses 12 trading days of NVDA (~1.5M trades each). Here
we run the identical five-filter pipeline on a configurable window per day
(default: ``MAX_TRADES`` trades after cleaning for ``DATES`` dates). This keeps
the runtime reasonable while exercising every stage of the methodology:

* Stage 1  median price filter
* Stage 2  scale filter and threshold
* Stage 3  Huber price filter (stratified estimator)
* Stage 4  benchmark raw / linear / median paths
* Stage 5  residual diagnostics incl. Hill tail index
* Stage 6  volatility signature curves
* Stage 7  directional outlier probability q_t

Results saved under ``results/exp_3/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data_loader import load_day
from src.empirical_pipeline import run_pipeline
from src.metrics import (
    acf,
    hill_tail_index,
    k_trade_diff_stats,
    s_second_diff_stats,
)


OUT = Path(__file__).resolve().parent.parent / "results" / "exp_3"
OUT.mkdir(parents=True, exist_ok=True)

DATES = ["2024-10-14", "2024-10-15", "2024-10-16"]
MAX_TRADES_PER_DAY = int(__import__("os").environ.get("NVDA_MAX_TRADES", 15000))
TICKER = "NVDA"


def run_single_day(date: str, max_trades: int, seed: int = 7) -> Dict:
    print(f"\n--- {TICKER} {date} (max_trades={max_trades}) ---")
    loaded = load_day(TICKER, date, max_trades=max_trades)
    df = loaded.df
    Y = df["log_price"].values.astype(np.float64)
    tau = df["tau_s"].values.astype(np.float64)
    result = run_pipeline(tau, Y, halflife_trades=3, B=100, seed=seed, T60=loaded.T60)

    # --------- diagnostics ---------
    hill = {
        "resid_hub_hill": hill_tail_index(result.resid_hub, frac_lo=0.001, frac_hi=0.01),
        "resid_med_hill": hill_tail_index(result.resid_med, frac_lo=0.001, frac_hi=0.01),
    }
    outlier_share = float((result.d_pos + result.d_neg).mean())
    # Table-1 style (k-trade differences on the Huber path, converted to log)
    tbl1 = k_trade_diff_stats(result.theta, ks=(1, 2, 5, 10, 25))
    # Table-2 style (s-second differences)
    tbl2 = s_second_diff_stats(tau, result.theta, s_list=(0.1, 1, 5, 10, 60, 900))
    # Also for raw and linear so we can compare
    tbl2_raw = s_second_diff_stats(tau, Y, s_list=(0.1, 1, 5, 10, 60, 900))
    tbl2_lin = s_second_diff_stats(tau, result.linear, s_list=(0.1, 1, 5, 10, 60, 900))

    # --------- plots ---------
    _plot_price_paths(date, tau, Y, result, OUT / f"{date}_prices.png")
    _plot_threshold(date, tau, result.delta, OUT / f"{date}_threshold.png")
    _plot_signature(date, tbl2_raw, tbl2_lin, tbl2, OUT / f"{date}_signature.png")
    _plot_q(date, tau, result.q, OUT / f"{date}_q.png")

    summary = dict(
        date=date,
        T=int(len(Y)),
        raw_count=int(loaded.raw_count),
        cleaned_count=int(loaded.cleaned_count),
        T60=int(loaded.T60),
        hill=hill,
        outlier_share=outlier_share,
        median_delta_cents=float(np.median(result.delta) * np.median(df["price"]) * 100.0),
        table1=tbl1,
        table2_huber=tbl2,
        table2_raw=tbl2_raw,
        table2_linear=tbl2_lin,
        resid_mean_abs_cents=float(np.mean(np.abs(result.resid_hub)) * np.median(df["price"]) * 100.0),
    )
    return summary


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_price_paths(date, tau, Y, result, path):
    fig, ax = plt.subplots(figsize=(11, 5))
    px = np.exp(Y)
    ax.plot(tau, px, "o", ms=1, color="lightgray", label="raw trades", alpha=0.5)
    ax.plot(tau, np.exp(result.linear), lw=0.8, color="tab:orange", label="linear (3-trade HL)")
    ax.plot(tau, np.exp(result.m), lw=0.8, color="tab:green", label="median")
    ax.plot(tau, np.exp(result.theta), lw=0.8, color="tab:blue", label="Huber")
    ax.set_xlabel("seconds since midnight ET")
    ax.set_ylabel("NVDA price")
    ax.set_title(f"NVDA {date}: raw vs linear vs median vs Huber (3-trade halflife)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _plot_threshold(date, tau, delta, path):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(tau, delta, lw=0.7, color="tab:red")
    ax.set_xlabel("seconds since midnight ET")
    ax.set_ylabel(r"$\delta_t$ (log-price units)")
    ax.set_title(f"NVDA {date}: threshold path")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _plot_signature(date, tbl2_raw, tbl2_lin, tbl2_hub, path):
    s_vals = sorted(tbl2_raw.keys(), key=lambda x: float(x))
    def _vols(tbl):
        return [tbl[s]["annualized_vol"] for s in s_vals]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogx([float(s) for s in s_vals], _vols(tbl2_raw), "o-", label="raw")
    ax.semilogx([float(s) for s in s_vals], _vols(tbl2_lin), "s-", label="linear")
    ax.semilogx([float(s) for s in s_vals], _vols(tbl2_hub), "d-", label="Huber")
    ax.set_xlabel("lag s (seconds)")
    ax.set_ylabel("annualized volatility")
    ax.set_title(f"NVDA {date}: volatility signature")
    ax.legend(); ax.grid(True, which="both", linestyle=":")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _plot_q(date, tau, q, path):
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(tau, q, lw=0.5)
    ax.axhline(0.5, color="k", ls="--", lw=0.7)
    ax.set_ylim(0, 1)
    ax.set_xlabel("seconds since midnight ET")
    ax.set_ylabel(r"$q_t$")
    ax.set_title(f"NVDA {date}: directional outlier probability")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_summaries: List[Dict] = []
    for d in DATES:
        all_summaries.append(run_single_day(d, max_trades=MAX_TRADES_PER_DAY))

    # cross-day aggregate: avg Hill, avg outlier share
    hill_avg = float(np.mean([s["hill"]["resid_hub_hill"] for s in all_summaries
                              if s["hill"]["resid_hub_hill"] is not None]))
    out_share_avg = float(np.mean([s["outlier_share"] for s in all_summaries]))
    aggregate = {
        "dates": DATES,
        "max_trades_per_day": MAX_TRADES_PER_DAY,
        "hill_resid_hub_mean": hill_avg,
        "outlier_share_mean": out_share_avg,
        "per_day": all_summaries,
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(aggregate, f, indent=2, default=str)
    print(f"\nResults saved to {OUT}")


if __name__ == "__main__":
    main()
