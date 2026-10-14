"""
Experiment 2
============

Synthetic Replication of Weighted Median Filters Under Heavy-Tailed Noise.

Parts
-----
A. Rolling median under nonstationary Cauchy noise (uniform weights, varying ell).
B. Stationary anchored weighted median (anchored weights, varying alpha, ell).
C. Exponentially weighted median via lag sampling, exact vs. simulated for several B.

Outputs: path plots, loss/tracking-error curves, approximation-error plots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.filters import exact_filter, simulated_filter, stratified_filter
from src.metrics import predictive_abs_loss, tracking_error_abs
from src.synthetic import nonstationary_cauchy, stationary_cauchy
from src.weights import (
    anchored_weights,
    exponential_inv_cum_weight,
    exponential_weights,
    uniform_weights,
)

OUT = Path(__file__).resolve().parent.parent / "results" / "exp_2"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# A. Rolling median under nonstationary Cauchy noise
# ---------------------------------------------------------------------------

def part_a(T: int = 1000, n_reps: int = 10, ell_grid=None, seed_base: int = 100):
    if ell_grid is None:
        ell_grid = list(range(4, 51))

    # illustrative single-run paths
    Y, mu = nonstationary_cauchy(T=T, seed=seed_base)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(Y, color="lightgray", lw=0.6, label="observed Y_t")
    ax.plot(mu, color="black", lw=1, label=r"latent $\mu_t$")
    for ell, color in zip([1, 7, 15], ["tab:red", "tab:green", "tab:blue"]):
        theta = exact_filter(Y, lambda t, ell=ell: uniform_weights(t, ell), loss_name="absolute")
        ax.plot(theta, lw=1, color=color, label=f"ell={ell}")
    ax.set_ylim(np.nanpercentile(Y, 2), np.nanpercentile(Y, 98))
    ax.set_title("Part A: rolling median filter paths (nonstationary Cauchy)")
    ax.set_xlabel("t"); ax.set_ylabel("value"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "A_paths.png", dpi=140); plt.close(fig)

    # Averaged predictive-loss / tracking-error curves over reps
    pred = np.zeros((n_reps, len(ell_grid)))
    trk  = np.zeros((n_reps, len(ell_grid)))
    oracle = np.zeros(n_reps)
    for r in range(n_reps):
        Yr, mur = nonstationary_cauchy(T=T, seed=seed_base + r)
        oracle[r] = predictive_abs_loss(Yr, mur)
        for i, ell in enumerate(ell_grid):
            theta = exact_filter(Yr, lambda t, ell=ell: uniform_weights(t, ell), loss_name="absolute")
            pred[r, i] = predictive_abs_loss(Yr, theta)
            trk[r, i]  = tracking_error_abs(mur, theta)

    pred_mean = pred.mean(axis=0); trk_mean = trk.mean(axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(ell_grid, pred_mean, "o-", label="rolling median")
    ax1.axhline(oracle.mean(), color="k", ls="--", label="oracle")
    ax1.set_xlabel("ell"); ax1.set_ylabel("avg predictive abs-loss")
    ax1.set_title("Part A: predictive loss vs ell"); ax1.legend(); ax1.grid(True, linestyle=":")
    ax2.plot(ell_grid, trk_mean, "o-")
    ax2.set_xlabel("ell"); ax2.set_ylabel("avg |mu_t - theta_t|")
    ax2.set_title("Part A: tracking error vs ell"); ax2.grid(True, linestyle=":")
    fig.tight_layout(); fig.savefig(OUT / "A_curves.png", dpi=140); plt.close(fig)

    return {
        "ell_grid": ell_grid,
        "pred_mean": pred_mean.tolist(),
        "trk_mean": trk_mean.tolist(),
        "oracle_mean": float(oracle.mean()),
    }


# ---------------------------------------------------------------------------
# B. Stationary anchored weighted median
# ---------------------------------------------------------------------------

def part_b(T: int = 1000, n_reps: int = 8, ell_grid=None, alphas=(0.4, 0.6, 0.8, 1.0),
           seed_base: int = 200):
    if ell_grid is None:
        ell_grid = list(range(4, 51))

    pred = {a: np.zeros((n_reps, len(ell_grid))) for a in alphas}
    trk  = {a: np.zeros((n_reps, len(ell_grid))) for a in alphas}
    oracle = np.zeros(n_reps)

    for r in range(n_reps):
        Yr, mur = stationary_cauchy(T=T, phi=0.95, seed=seed_base + r)
        oracle[r] = predictive_abs_loss(Yr, mur)
        for a in alphas:
            for i, ell in enumerate(ell_grid):
                wfn = lambda t, ell=ell, a=a: anchored_weights(t, ell, a, T)
                theta = exact_filter(Yr, wfn, loss_name="absolute")
                pred[a][r, i] = predictive_abs_loss(Yr, theta)
                trk[a][r, i]  = tracking_error_abs(mur, theta)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for a in alphas:
        ax1.plot(ell_grid, pred[a].mean(axis=0), "o-", ms=3, label=f"alpha={a}")
        ax2.plot(ell_grid, trk[a].mean(axis=0),  "o-", ms=3, label=f"alpha={a}")
    ax1.axhline(oracle.mean(), color="k", ls="--", label="oracle")
    ax1.set_xlabel("ell"); ax1.set_ylabel("avg predictive abs-loss")
    ax1.set_title("Part B: predictive loss vs ell, by alpha"); ax1.legend(fontsize=8); ax1.grid(True, linestyle=":")
    ax2.set_xlabel("ell"); ax2.set_ylabel("avg |mu_t - theta_t|")
    ax2.set_title("Part B: tracking error vs ell, by alpha"); ax2.legend(fontsize=8); ax2.grid(True, linestyle=":")
    fig.tight_layout(); fig.savefig(OUT / "B_curves.png", dpi=140); plt.close(fig)

    # illustrative path for alpha=0.6 at ell in {1,7,15}
    Yr, mur = stationary_cauchy(T=T, phi=0.95, seed=seed_base)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(Yr, color="lightgray", lw=0.6, label="Y_t")
    ax.plot(mur, color="black", lw=1, label=r"$\mu_t$")
    for ell, c in zip([1, 7, 15], ["tab:red", "tab:green", "tab:blue"]):
        wfn = lambda t, ell=ell: anchored_weights(t, ell, 0.6, T)
        theta = exact_filter(Yr, wfn, loss_name="absolute")
        ax.plot(theta, lw=1, color=c, label=f"ell={ell}")
    ax.set_ylim(np.nanpercentile(Yr, 2), np.nanpercentile(Yr, 98))
    ax.set_title("Part B: anchored (alpha=0.6) weighted median paths")
    ax.set_xlabel("t"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "B_paths.png", dpi=140); plt.close(fig)

    out = {"ell_grid": ell_grid, "oracle_mean": float(oracle.mean())}
    for a in alphas:
        out[f"pred_mean_alpha_{a}"] = pred[a].mean(axis=0).tolist()
        out[f"trk_mean_alpha_{a}"]  = trk[a].mean(axis=0).tolist()
    return out


# ---------------------------------------------------------------------------
# C. Exponentially weighted median: exact vs. simulated vs. stratified
# ---------------------------------------------------------------------------

def part_c(T: int = 1000, lam: float = 0.9, Bs=(500, 5000, 50000), n_reps: int = 5,
           seed_base: int = 300):
    results: Dict[int, List[float]] = {B: [] for B in Bs}
    results_strat: Dict[int, List[float]] = {B: [] for B in Bs}
    for r in range(n_reps):
        Y, _ = nonstationary_cauchy(T=T, seed=seed_base + r)
        theta_e = exact_filter(Y, lambda t: exponential_weights(t, lam), loss_name="absolute")
        for B in Bs:
            theta_s = simulated_filter(
                Y, lambda t: exponential_weights(t, lam),
                loss_name="absolute", B=B, rng=np.random.default_rng(r * 97 + B),
            )
            results[B].append(float(np.mean(np.abs(theta_e - theta_s))))

            theta_sr = stratified_filter(
                Y,
                lambda t: exponential_weights(t, lam),
                lambda t: exponential_inv_cum_weight(0.999, t, lam),
                loss_name="absolute", B=B, rng=np.random.default_rng(r * 131 + B),
            )
            results_strat[B].append(float(np.mean(np.abs(theta_e - theta_sr))))
        print(f"rep {r}: naive {dict((B, results[B][-1]) for B in Bs)}  "
              f"strat {dict((B, results_strat[B][-1]) for B in Bs)}")

    # One illustrative path
    Y, mu = nonstationary_cauchy(T=T, seed=seed_base)
    theta_e = exact_filter(Y, lambda t: exponential_weights(t, lam), loss_name="absolute")
    theta_s = simulated_filter(Y, lambda t: exponential_weights(t, lam),
                               loss_name="absolute", B=500, rng=np.random.default_rng(0))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mu, color="black", lw=1, label=r"$\mu_t$")
    ax.plot(theta_e, color="tab:blue", lw=1, label="exact exp. wmedian")
    ax.plot(theta_s, color="tab:orange", lw=0.8, alpha=0.8, label="sim exp. wmedian (B=500)")
    ax.set_ylim(np.nanpercentile(Y, 2), np.nanpercentile(Y, 98))
    ax.set_title(f"Part C: exact vs simulated exp. weighted median (lam={lam})")
    ax.set_xlabel("t"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "C_paths.png", dpi=140); plt.close(fig)

    # Convergence plot
    naive_mean = {B: float(np.mean(results[B])) for B in Bs}
    strat_mean = {B: float(np.mean(results_strat[B])) for B in Bs}
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.loglog(list(Bs), [naive_mean[B] for B in Bs], "o-", label="naive sim")
    ax.loglog(list(Bs), [strat_mean[B] for B in Bs], "s-", label="stratified")
    ax.set_xlabel("B"); ax.set_ylabel("MAE vs exact")
    ax.set_title("Part C: approximation error vs B")
    ax.legend(); ax.grid(True, which="both", linestyle=":")
    fig.tight_layout(); fig.savefig(OUT / "C_convergence.png", dpi=140); plt.close(fig)

    return {"naive_mean": naive_mean, "strat_mean": strat_mean}


def main():
    summary = {}
    print("=== Part A ===")
    summary["A"] = part_a()
    print("\n=== Part B ===")
    summary["B"] = part_b()
    print("\n=== Part C ===")
    summary["C"] = part_c()
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults written to {OUT}")


if __name__ == "__main__":
    main()
