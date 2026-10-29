"""
Experiment 2
============

Synthetic Replication of Weighted Median Filters Under Heavy-Tailed Noise.

Parts
-----
A. Rolling median under nonstationary Cauchy noise (uniform weights, varying ell).
B. Stationary anchored weighted median (anchored weights, varying alpha, ell).
C. Exponentially weighted median via lag sampling (Figures 4 and 5): the
   simulated filter of Definition 3 against the exact filter for several B,
   both with the median convention of Theorem 2 (the smallest value at which
   the cumulative weight reaches one half). The frequency of an exact match
   is compared with Theorem 2's probability on paths of length 1,000.
   Figure 4 uses one path of length 5,000 and traces the predictive loss
   over lambda from 0.80 to 0.95; Figure 5 uses its first 1,000 points. The
   stratified estimator of Remark 4(d) is added for comparison, with both
   median conventions. Figure 3 draws Theorem 2's recovery probability in
   the stylized case of the paper.
D. Example 9 and Remark 3: the simulated rolling median when the median is
   not unique (odd window). Under the smallest-value convention the
   simulated median splits between the two middle values; computing the
   Hazen / type-5 median from the simulated lag frequencies recovers the
   interpolated median.

Outputs: path plots, loss/tracking-error curves, approximation-error plots,
written to ``results/`` with the prefix ``exp2_``; the run log is
``results/exp2.log``.
"""

from __future__ import annotations

import json
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import binom

from data.generate_synthetic import nonstationary_cauchy, stationary_cauchy
from exp.common import RESULTS, log_to_results
from src.config import (
    EX7B_PHI,
    EXP2_A_PATHS,
    EXP2_A_SEED,
    EXP2_B_ALPHAS,
    EXP2_B_PATH_ALPHA,
    EXP2_B_PATHS,
    EXP2_B_SEED,
    EXP2_C_B_GRID,
    EXP2_C_LAMBDA,
    EXP2_C_PATHS,
    EXP2_C_SEED,
    EXP2_D_B,
    EXP2_D_ELLS,
    EXP2_D_SEED,
    EXP2_ELL_GRID,
    EXP2_FIG3_B_MAX,
    EXP2_FIG3_B_TABLE,
    EXP2_FIG3_EPS,
    EXP2_FIG4_B,
    EXP2_FIG4_LAMBDA_GRID,
    EXP2_FIG4_T,
    EXP2_PATH_ELLS,
    EXP2_T,
)
from src.filters import (
    anchored_median_filter,
    exact_filter,
    exponential_lag_sampler,
    simulated_filter_fast,
    stratified_exponential_filter,
)
from src.losses import weighted_median
from src.metrics import predictive_abs_loss, tracking_error_abs
from src.weights import (
    exponential_inv_cum_weight,
    exponential_weights,
    uniform_weights,
)

PREFIX = "exp2_"


# ---------------------------------------------------------------------------
# A. Rolling median under nonstationary Cauchy noise
# ---------------------------------------------------------------------------

def part_a(T: int = EXP2_T, n_reps: int = EXP2_A_PATHS, ell_grid=EXP2_ELL_GRID, seed_base: int = EXP2_A_SEED):
    ell_grid = list(ell_grid)

    # illustrative single-run paths
    Y, mu = nonstationary_cauchy(T=T, seed=seed_base)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(Y, color="lightgray", lw=0.6, label="observed Y_t")
    ax.plot(mu, color="black", lw=1, label=r"latent $\mu_t$")
    for ell, color in zip(EXP2_PATH_ELLS, ["tab:red", "tab:green", "tab:blue"]):
        theta = exact_filter(Y, lambda t, ell=ell: uniform_weights(t, ell), loss_name="absolute")
        ax.plot(theta, lw=1, color=color, label=f"ell={ell}")
    ax.set_ylim(np.nanpercentile(Y, 2), np.nanpercentile(Y, 98))
    ax.set_title("Part A: rolling median filter paths (nonstationary Cauchy)")
    ax.set_xlabel("t"); ax.set_ylabel("value"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS / f"{PREFIX}A_paths.png", dpi=140); plt.close(fig)

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
    fig.tight_layout(); fig.savefig(RESULTS / f"{PREFIX}A_curves.png", dpi=140); plt.close(fig)

    return {
        "ell_grid": ell_grid,
        "pred_mean": pred_mean.tolist(),
        "trk_mean": trk_mean.tolist(),
        "oracle_mean": float(oracle.mean()),
    }


# ---------------------------------------------------------------------------
# B. Stationary anchored weighted median
# ---------------------------------------------------------------------------

def part_b(T: int = EXP2_T, n_reps: int = EXP2_B_PATHS, ell_grid=EXP2_ELL_GRID, alphas=EXP2_B_ALPHAS,
           seed_base: int = EXP2_B_SEED):
    ell_grid = list(ell_grid)

    pred = {a: np.zeros((n_reps, len(ell_grid))) for a in alphas}
    trk  = {a: np.zeros((n_reps, len(ell_grid))) for a in alphas}
    oracle = np.zeros(n_reps)

    for r in range(n_reps):
        Yr, mur = stationary_cauchy(T=T, phi=EX7B_PHI, seed=seed_base + r)
        oracle[r] = predictive_abs_loss(Yr, mur)
        for a in alphas:
            for i, ell in enumerate(ell_grid):
                theta = anchored_median_filter(Yr, ell, a)
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
    fig.tight_layout(); fig.savefig(RESULTS / f"{PREFIX}B_curves.png", dpi=140); plt.close(fig)

    Yr, mur = stationary_cauchy(T=T, phi=EX7B_PHI, seed=seed_base)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(Yr, color="lightgray", lw=0.6, label="Y_t")
    ax.plot(mur, color="black", lw=1, label=r"$\mu_t$")
    for ell, c in zip(EXP2_PATH_ELLS, ["tab:red", "tab:green", "tab:blue"]):
        theta = anchored_median_filter(Yr, ell, EXP2_B_PATH_ALPHA)
        ax.plot(theta, lw=1, color=c, label=f"ell={ell}")
    ax.set_ylim(np.nanpercentile(Yr, 2), np.nanpercentile(Yr, 98))
    ax.set_title(f"Part B: anchored (alpha={EXP2_B_PATH_ALPHA}) weighted median paths")
    ax.set_xlabel("t"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS / f"{PREFIX}B_paths.png", dpi=140); plt.close(fig)

    out = {"ell_grid": ell_grid, "oracle_mean": float(oracle.mean())}
    for a in alphas:
        out[f"pred_mean_alpha_{a}"] = pred[a].mean(axis=0).tolist()
        out[f"trk_mean_alpha_{a}"]  = trk[a].mean(axis=0).tolist()
    return out


# ---------------------------------------------------------------------------
# C. Exponentially weighted median: exact vs. simulated vs. stratified
# ---------------------------------------------------------------------------

def _theorem2_prob(Y: np.ndarray, theta: np.ndarray, lam: float, B: int) -> np.ndarray:
    """``P(θ*_t = θ_t | Y_{1:t})`` from Theorem 2, for every t."""
    k = int(np.ceil(B / 2.0)) - 1
    out = np.empty(len(Y))
    for t in range(1, len(Y) + 1):
        w = exponential_weights(t, lam)
        y = Y[:t][::-1]
        f_minus = float(w[y < theta[t - 1]].sum())
        f_at = float(w[y <= theta[t - 1]].sum())
        out[t - 1] = binom.cdf(k, B, min(f_minus, 1.0)) - binom.cdf(k, B, min(f_at, 1.0))
    return out


def part_c(T: int = EXP2_T, lam: float = EXP2_C_LAMBDA, Bs=EXP2_C_B_GRID, n_reps: int = EXP2_C_PATHS,
           seed_base: int = EXP2_C_SEED, lam_grid=EXP2_FIG4_LAMBDA_GRID, T_fig4: int = EXP2_FIG4_T,
           B_fig4: int = EXP2_FIG4_B):
    """Figure 4 is drawn on a path of length ``T_fig4`` (5,000 in the paper) and
    Figure 5 on its first 1,000 points; the error table uses ``n_reps`` paths
    of length ``T``."""
    lam_grid = list(lam_grid)
    draw = exponential_lag_sampler(lam)
    rows = {B: {"mae": [], "p_equal": [], "p_theorem2": []} for B in Bs}
    strat = {B: {"lower": [], "hazen": []} for B in Bs}
    for r in range(n_reps):
        Y, _ = nonstationary_cauchy(T=T, seed=seed_base + r)
        theta_e = exact_filter(Y, lambda t: exponential_weights(t, lam), loss_name="absolute",
                               interpolation="lower")
        theta_h = exact_filter(Y, lambda t: exponential_weights(t, lam), loss_name="absolute")
        for B in Bs:
            theta_s = simulated_filter_fast(Y, draw, "absolute", B=B, rng=np.random.default_rng(r * 97 + B))
            rows[B]["mae"].append(float(np.mean(np.abs(theta_e - theta_s))))
            rows[B]["p_equal"].append(float(np.mean(theta_e == theta_s)))
            rows[B]["p_theorem2"].append(float(np.mean(_theorem2_prob(Y, theta_e, lam, B))))
            for conv, ref in (("lower", theta_e), ("hazen", theta_h)):
                th = stratified_exponential_filter(Y, lam, "absolute", B=B, interpolation=conv,
                                                   rng=np.random.default_rng(r * 131 + B))
                strat[B][conv].append(float(np.mean(np.abs(ref - th))))
        print(f"rep {r}: " + "  ".join(f"B={B} mae={rows[B]['mae'][-1]:.4f} P(eq)={rows[B]['p_equal'][-1]:.3f} "
                                        f"thm2={rows[B]['p_theorem2'][-1]:.3f}" for B in Bs), flush=True)

    # Figure 4 on one path of length T_fig4, Figure 5 on its first 1,000 points
    # (the filter is causal, so those points do not depend on the rest).
    Y, mu = nonstationary_cauchy(T=T_fig4, seed=seed_base)
    theta_s = simulated_filter_fast(Y, draw, "absolute", B=B_fig4, rng=np.random.default_rng(0))
    loss_exact, loss_sim = [], []
    for lg in lam_grid:
        th_e = exact_filter(Y, lambda t, lg=lg: exponential_weights(t, lg), loss_name="absolute", interpolation="lower")
        th_s = simulated_filter_fast(Y, exponential_lag_sampler(lg), "absolute", B=B_fig4, rng=np.random.default_rng(1))
        loss_exact.append(predictive_abs_loss(Y, th_e))
        loss_sim.append(predictive_abs_loss(Y, th_s))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    yc = np.clip(Y, -30, 30)
    axes[0].plot(yc, ".", ms=2, color="0.6")
    axes[0].plot(theta_s, color="tab:red", lw=0.8)
    axes[0].set_title(f"Y_t and θ*_t, λ={lam}, B={B_fig4}")
    axes[1].plot(mu, ".", ms=2, color="0.4")
    axes[1].plot(theta_s, color="tab:red", lw=0.8)
    axes[1].set_title("μ_t and θ*_t")
    axes[2].plot(lam_grid, loss_sim, "o:", label="θ*_{t-1}")
    axes[2].plot(lam_grid, loss_exact, "s-", label="θ_{t-1}")
    axes[2].set_xlabel("λ")
    axes[2].set_title("predictive loss minus |Y_t|")
    axes[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS / f"{PREFIX}C_figure4.png", dpi=140); plt.close(fig)

    y5 = Y[:T]
    theta_e = exact_filter(y5, lambda t: exponential_weights(t, lam), loss_name="absolute", interpolation="lower")
    fig, axes = plt.subplots(1, len(Bs), figsize=(15, 3.8), sharey=True)
    for ax, B in zip(axes, Bs):
        th = simulated_filter_fast(y5, draw, "absolute", B=B, rng=np.random.default_rng(B))
        ax.plot(th - theta_e, ".", ms=2)
        ax.set_title(f"B = {B:,}")
        ax.set_xlabel("time")
    axes[0].set_ylabel("median filter error θ*_t − θ_t")
    fig.tight_layout(); fig.savefig(RESULTS / f"{PREFIX}C_figure5.png", dpi=140); plt.close(fig)

    naive = {B: {k: float(np.mean(v)) for k, v in rows[B].items()} for B in Bs}
    strat_mean = {B: {k: float(np.mean(v)) for k, v in strat[B].items()} for B in Bs}
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.loglog(list(Bs), [naive[B]["mae"] for B in Bs], "o-", label="simulated, lower median")
    ax.loglog(list(Bs), [strat_mean[B]["lower"] for B in Bs], "s-", label="stratified, lower median")
    ax.loglog(list(Bs), [strat_mean[B]["hazen"] for B in Bs], "^-", label="stratified, type 5")
    ax.set_xlabel("B"); ax.set_ylabel("mean |θ*_t − θ_t|")
    ax.set_title("Part C: approximation error vs B")
    ax.legend(fontsize=8); ax.grid(True, which="both", linestyle=":")
    fig.tight_layout(); fig.savefig(RESULTS / f"{PREFIX}C_convergence.png", dpi=140); plt.close(fig)
    return {"simulated": naive, "stratified": strat_mean, "lam_grid": lam_grid, "T_fig4": T_fig4,
            "loss_exact_by_lam": loss_exact, "loss_simulated_by_lam": loss_sim,
            "lam_best_exact": lam_grid[int(np.argmin(loss_exact))]}


# ---------------------------------------------------------------------------
# Figure 3 and Example 9
# ---------------------------------------------------------------------------

def figure3(eps=EXP2_FIG3_EPS, Bmax: int = EXP2_FIG3_B_MAX):
    """``P(θ*_t = θ_t)`` with ``F̂(θ−) = 0.5 − ε`` and ``F̂(θ) = 0.5 + ε``."""
    Bs = np.unique(np.round(np.geomspace(1, Bmax, 200)).astype(int))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    out = {}
    for e, style in zip(eps, ("k-", "r--", "b:")):
        k = np.ceil(Bs / 2.0) - 1
        p = binom.cdf(k, Bs, 0.5 - e) - binom.cdf(k, Bs, 0.5 + e)
        ax.semilogx(Bs, p, style, label=f"ε = {e}")
        out[str(e)] = {str(b): float(binom.cdf(np.ceil(b / 2) - 1, b, 0.5 - e) - binom.cdf(np.ceil(b / 2) - 1, b, 0.5 + e))
                       for b in EXP2_FIG3_B_TABLE}
    ax.set_xlabel("B")
    ax.set_ylabel("probability of recovery")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle=":")
    fig.tight_layout(); fig.savefig(RESULTS / f"{PREFIX}C_figure3.png", dpi=140); plt.close(fig)
    return out


def part_d(T: int = EXP2_T, ells=EXP2_D_ELLS, B: int = EXP2_D_B, seed: int = EXP2_D_SEED):
    """Example 9: simulate the rolling median with ``B`` uniform lag draws."""
    Y, _ = nonstationary_cauchy(T=T, seed=seed)
    rng = np.random.default_rng(seed)
    out = {}
    for ell in ells:
        lower_hit = upper_hit = 0
        hazen_err, n_used = [], 0
        for t in range(ell + 1, T + 1):
            win = Y[t - 1 - ell:t]
            srt = np.sort(win)
            lo, hi = srt[(ell + 1 + 1) // 2 - 1], srt[(ell + 2) // 2]
            lags = rng.integers(0, ell + 1, size=B)
            w_star = np.bincount(lags, minlength=ell + 1) / B
            vals = win[::-1]
            star_lower = weighted_median(vals, w_star, interpolation="lower")
            exact_hazen = weighted_median(vals, np.full(ell + 1, 1.0 / (ell + 1)))
            hazen_err.append(abs(weighted_median(vals, w_star) - exact_hazen))
            lower_hit += star_lower == srt[(ell + 1 + 1) // 2 - 1]
            if ell % 2 == 1:
                upper_hit += star_lower == hi
            n_used += 1
        out[str(ell)] = {"odd": bool(ell % 2), "B": B, "points": n_used,
                         "share_star_equals_lower_median": lower_hit / n_used,
                         "share_star_equals_upper_median": upper_hit / n_used if ell % 2 else None,
                         "mean_abs_hazen_from_frequencies_minus_exact_hazen": float(np.mean(hazen_err))}
        print(f"Example 9 ell={ell}: {out[str(ell)]}", flush=True)
    return out


def main():
    log_to_results("exp2")
    summary = {}
    print("=== Part A ===")
    summary["A"] = part_a()
    print("\n=== Part B ===")
    summary["B"] = part_b()
    print("\n=== Part C ===")
    summary["C"] = part_c()
    summary["C"]["figure3"] = figure3()
    print("\n=== Part D (Example 9) ===")
    summary["D"] = part_d()
    with open(RESULTS / f"{PREFIX}summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults written to {RESULTS}")


if __name__ == "__main__":
    main()
