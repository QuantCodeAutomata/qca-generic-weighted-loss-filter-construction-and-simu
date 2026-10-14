"""
Experiment 4
============

Robustness and cross-asset extension.

Modules implemented (reduced-scale versions of the paper's appendix):
  A. Tail-index robustness: alternative screens and half-lives for NVDA.
  B. Alternative volatility lags: s = 10 s vs s = 1 s on NVDA.
  C. Simulation-based volatility-signature experiment under Brownian and
     Brownian + heavy-tailed noise latent processes.
  D. Cross-asset mini-panel: tail index and outlier share for a handful of
     actively traded tickers.

Note: the full appendix uses 54 additional assets. To respect wall-clock
limits in the automation environment, module D ingests a small, documented
subset. All code paths generalise unchanged to larger universes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data_loader import load_day
from src.empirical_pipeline import run_pipeline
from src.metrics import (
    annualized_vol_from_rets,
    hill_tail_index,
    s_second_diff_stats,
)

OUT = Path(__file__).resolve().parent.parent / "results" / "exp_4"
OUT.mkdir(parents=True, exist_ok=True)

NVDA_DATE = "2024-10-16"
NVDA_MAX_TRADES = int(os.environ.get("NVDA_MAX_TRADES", 15000))

CROSS_ASSET_TICKERS = [t.strip() for t in os.environ.get(
    "EXP4_TICKERS", "TSLA,AAPL,SPY,AMD,MSFT"
).split(",") if t.strip()]
CROSS_ASSET_MAX_TRADES = int(os.environ.get("EXP4_MAX_TRADES", 5000))


# ---------------------------------------------------------------------------
# Module A: tail-index robustness
# ---------------------------------------------------------------------------

def module_a():
    loaded = load_day("NVDA", NVDA_DATE, max_trades=NVDA_MAX_TRADES)
    Y = loaded.df["log_price"].values.astype(np.float64)
    tau = loaded.df["tau_s"].values.astype(np.float64)

    variants = {}
    # Baseline (3-trade halflife)
    res3 = run_pipeline(tau, Y, halflife_trades=3, B=100, seed=1, T60=loaded.T60)
    variants["baseline_HL3"] = hill_tail_index(res3.resid_hub)
    # 6-trade halflife
    res6 = run_pipeline(tau, Y, halflife_trades=6, B=100, seed=1, T60=loaded.T60)
    variants["HL6"] = hill_tail_index(res6.resid_hub)
    # Residual / delta (standardized)
    r_std = res3.resid_hub / np.where(res3.delta > 0, res3.delta, 1.0)
    variants["baseline_HL3_standardized"] = hill_tail_index(r_std)
    # Thin by factor 10
    variants["baseline_HL3_thin10"] = hill_tail_index(res3.resid_hub[::10])
    # Remove repricing windows: trades near large filter jumps
    jumps = np.abs(np.diff(res3.theta, prepend=res3.theta[0]))
    keep = jumps < 0.001  # ~0.1% log jump
    variants["baseline_HL3_no_repricing"] = hill_tail_index(res3.resid_hub[keep])

    print("Module A tail-index variants:", variants)
    return variants


# ---------------------------------------------------------------------------
# Module B: alternative volatility lags
# ---------------------------------------------------------------------------

def module_b():
    loaded = load_day("NVDA", NVDA_DATE, max_trades=NVDA_MAX_TRADES)
    Y = loaded.df["log_price"].values.astype(np.float64)
    tau = loaded.df["tau_s"].values.astype(np.float64)
    res = run_pipeline(tau, Y, halflife_trades=3, B=100, seed=1, T60=loaded.T60)

    tbl_hub = s_second_diff_stats(tau, res.theta, s_list=(1, 10))
    tbl_raw = s_second_diff_stats(tau, Y, s_list=(1, 10))
    tbl_lin = s_second_diff_stats(tau, res.linear, s_list=(1, 10))
    out = {"huber": tbl_hub, "raw": tbl_raw, "linear": tbl_lin}
    print("Module B vol-lag comparison:", json.dumps(out, indent=2, default=str))
    return out


# ---------------------------------------------------------------------------
# Module C: simulated signature experiments
# ---------------------------------------------------------------------------

def _simulate_brownian_with_noise(T: int, dt: float, sigma: float, noise: str, rng):
    """Return (tau_sec, Y_log_price) for a Brownian latent price + optional noise."""
    tau = np.cumsum(rng.exponential(dt, size=T))
    inc = rng.standard_normal(T) * sigma * np.sqrt(np.diff(tau, prepend=tau[0]))
    latent = np.cumsum(inc)
    if noise == "none":
        micro = np.zeros(T)
    elif noise == "cauchy":
        micro = rng.standard_cauchy(T) * 1e-4
    else:
        raise ValueError(noise)
    return tau, latent + micro


def module_c(T: int = 4000):
    rng = np.random.default_rng(0)
    scenarios = {
        "brownian_no_noise":   _simulate_brownian_with_noise(T, 0.1, 2e-4, "none", rng),
        "brownian_cauchy":     _simulate_brownian_with_noise(T, 0.1, 2e-4, "cauchy", rng),
    }
    s_list = (0.1, 1, 5, 10, 60)
    out = {}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, (tau, Y) in scenarios.items():
        res = run_pipeline(tau, Y, halflife_trades=3, B=100, seed=1, T60=10)
        tbl_hub = s_second_diff_stats(tau, res.theta, s_list=s_list)
        tbl_raw = s_second_diff_stats(tau, Y, s_list=s_list)
        tbl_lin = s_second_diff_stats(tau, res.linear, s_list=s_list)
        out[name] = {"huber": tbl_hub, "raw": tbl_raw, "linear": tbl_lin}
        ax.semilogx([float(s) for s in s_list],
                    [tbl_raw[s]["annualized_vol"] for s in s_list],
                    "o--", label=f"{name} raw")
        ax.semilogx([float(s) for s in s_list],
                    [tbl_hub[s]["annualized_vol"] for s in s_list],
                    "d-", label=f"{name} Huber")
    ax.set_xlabel("lag s (seconds)")
    ax.set_ylabel("annualized vol")
    ax.set_title("Module C: simulated signature plots")
    ax.legend(fontsize=7); ax.grid(True, which="both", linestyle=":")
    fig.tight_layout(); fig.savefig(OUT / "C_signature.png", dpi=140); plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Module D: cross-asset mini-panel
# ---------------------------------------------------------------------------

def module_d():
    rows = []
    for tk in CROSS_ASSET_TICKERS:
        try:
            loaded = load_day(tk, NVDA_DATE, max_trades=CROSS_ASSET_MAX_TRADES)
            if len(loaded.df) < 100:
                print(f"[{tk}] skipped: too few trades")
                continue
            Y = loaded.df["log_price"].values.astype(np.float64)
            tau = loaded.df["tau_s"].values.astype(np.float64)
            res = run_pipeline(tau, Y, halflife_trades=3, B=100, seed=1, T60=loaded.T60)
            hill = hill_tail_index(res.resid_hub)
            out_share = float((res.d_pos + res.d_neg).mean())
            rows.append({
                "ticker": tk,
                "T": int(len(Y)),
                "hill_hub": hill,
                "outlier_share": out_share,
                "mean_abs_resid_bp": float(np.mean(np.abs(res.resid_hub)) * 1e4),
            })
            print(f"[{tk}] T={len(Y)}  Hill={hill}  outlier_share={out_share:.4f}")
        except Exception as e:
            print(f"[{tk}] skipped: {e!r}")

    # bar plot
    if rows:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar([r["ticker"] for r in rows],
               [r["hill_hub"] if r["hill_hub"] is not None else 0 for r in rows],
               color="tab:blue", alpha=0.8)
        ax.axhline(2.0, color="k", ls="--", lw=0.7, label="finite variance threshold (2)")
        ax.set_ylabel("Hill tail index (Huber residuals)")
        ax.set_title("Module D: cross-asset Huber-residual tail indices")
        ax.legend()
        fig.tight_layout(); fig.savefig(OUT / "D_hill.png", dpi=140); plt.close(fig)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    summary = {}
    print("=== Module A: tail-index robustness ===")
    summary["A"] = module_a()
    print("\n=== Module B: alternative vol lags ===")
    summary["B"] = module_b()
    print("\n=== Module C: simulated signature ===")
    summary["C"] = module_c()
    print("\n=== Module D: cross-asset mini-panel ===")
    summary["D"] = module_d()
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults written to {OUT}")


if __name__ == "__main__":
    main()
