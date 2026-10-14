"""
Experiment 1
============

Generic Weighted-Loss Filter Construction and Simulation Approximation
----------------------------------------------------------------------

Validates (a) exact vs. simulated vs. stratified filter agreement,
(b) convergence of simulation error with Monte Carlo size ``B``,
(c) approximately O(T) runtime scaling at fixed ``B``,
(d) applicability across convex losses (squared, absolute, Huber, Bernoulli)
    and weight schemes (uniform, exponential, hyperbolic, superposition).

All results are saved under ``results/exp_1/``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.filters import exact_filter, simulated_filter, stratified_filter
from src.synthetic import nonstationary_cauchy
from src.weights import (
    SuperpositionSpec,
    exponential_inv_cum_weight,
    exponential_weights,
    hyperbolic_weights,
    uniform_weights,
)


OUT = Path(__file__).resolve().parent.parent / "results" / "exp_1"
OUT.mkdir(parents=True, exist_ok=True)


def _mean_abs_gap(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


# ---------------------------------------------------------------------------
# A. Convergence over B
# ---------------------------------------------------------------------------

def convergence_over_B(T: int = 1000, seed: int = 0) -> Dict:
    """Approximation error vs. B for several loss / weight combinations."""
    Y, mu = nonstationary_cauchy(T=T, seed=seed)
    binary = (Y > 0).astype(np.float64)  # proxy binary outcome for Bernoulli

    def weight_fn_exp(lam=0.9):
        return lambda t: exponential_weights(t, lam)

    def weight_fn_unif(ell=15):
        return lambda t: uniform_weights(t, ell)

    def weight_fn_hyp(a=10):
        return lambda t: hyperbolic_weights(t, a)

    scenarios = [
        ("absolute_exp_lam0.9", "absolute", weight_fn_exp(0.9), Y),
        ("squared_unif_ell15",  "squared",  weight_fn_unif(15), Y),
        ("huber_hyp_a10",       "huber",    weight_fn_hyp(10),  Y),
        ("bernoulli_exp_lam0.9","bernoulli","" ,                binary),  # placeholder weights set below
    ]
    scenarios[-1] = ("bernoulli_exp_lam0.9", "bernoulli", weight_fn_exp(0.9), binary)

    Bs = [100, 500, 5000, 50000]
    rows = []
    for name, loss, wfn, series in scenarios:
        theta_exact = exact_filter(series, wfn, loss_name=loss)
        for B in Bs:
            theta_sim = simulated_filter(
                series, wfn, loss_name=loss, B=B, rng=np.random.default_rng(123)
            )
            gap = _mean_abs_gap(theta_exact, theta_sim)
            rows.append({"scenario": name, "loss": loss, "B": B, "mae": gap})
            print(f"{name:28s} B={B:>6d}  MAE={gap:.6f}")

    # Plot MAE vs B per scenario
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in sorted(set(r["scenario"] for r in rows)):
        xs = [r["B"] for r in rows if r["scenario"] == name]
        ys = [r["mae"] for r in rows if r["scenario"] == name]
        ax.loglog(xs, ys, marker="o", label=name)
    ax.set_xlabel("B (Monte Carlo draws)")
    ax.set_ylabel("MAE(exact, simulated)")
    ax.set_title(f"Convergence of simulated filter (T={T})")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle=":")
    fig.tight_layout()
    fig.savefig(OUT / "convergence_vs_B.png", dpi=140)
    plt.close(fig)

    return {"rows": rows, "Bs": Bs}


# ---------------------------------------------------------------------------
# B. Stratified vs exact
# ---------------------------------------------------------------------------

def stratified_vs_exact(T: int = 1000, seed: int = 0, lam: float = 0.9) -> Dict:
    Y, _ = nonstationary_cauchy(T=T, seed=seed)
    results = {}
    for loss in ["absolute", "squared", "huber"]:
        theta_e = exact_filter(Y, lambda t: exponential_weights(t, lam), loss_name=loss)
        theta_s = stratified_filter(
            Y,
            lambda t: exponential_weights(t, lam),
            lambda t: exponential_inv_cum_weight(0.999, t, lam),
            loss_name=loss,
            B=200,
            rng=np.random.default_rng(42),
        )
        gap = _mean_abs_gap(theta_e, theta_s)
        results[loss] = gap
        print(f"stratified vs exact  loss={loss:10s}  MAE={gap:.3e}")
    return results


# ---------------------------------------------------------------------------
# C. Runtime scaling in T (approximately O(T))
# ---------------------------------------------------------------------------

def runtime_scaling(Ts=(100, 300, 1000, 3000, 10000), B: int = 200, lam: float = 0.9) -> Dict:
    rows = []
    for T in Ts:
        rng = np.random.default_rng(0)
        Y = rng.standard_cauchy(T)
        t0 = time.perf_counter()
        _ = stratified_filter(
            Y,
            lambda t: exponential_weights(t, lam),
            lambda t: exponential_inv_cum_weight(0.999, t, lam),
            loss_name="absolute",
            B=B,
            rng=np.random.default_rng(0),
        )
        dt = time.perf_counter() - t0
        rows.append({"T": T, "seconds": dt, "per_point_us": 1e6 * dt / T})
        print(f"T={T:>6d}  total={dt:7.3f}s  per-pt={1e6*dt/T:7.1f} us")

    xs = [r["T"] for r in rows]
    ys = [r["seconds"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(xs, ys, "o-", label="stratified")
    # reference O(T) line through first point
    ref = ys[0] * (np.array(xs) / xs[0])
    ax.loglog(xs, ref, "k--", alpha=0.5, label="O(T) reference")
    ax.set_xlabel("T")
    ax.set_ylabel("seconds")
    ax.set_title("Runtime scaling of the stratified filter")
    ax.legend()
    ax.grid(True, which="both", linestyle=":")
    fig.tight_layout()
    fig.savefig(OUT / "runtime_scaling.png", dpi=140)
    plt.close(fig)
    return {"rows": rows}


# ---------------------------------------------------------------------------
# D. Multiple weight schemes with the same loss (illustrative paths)
# ---------------------------------------------------------------------------

def weight_scheme_showcase(T: int = 500, seed: int = 7) -> None:
    Y, mu = nonstationary_cauchy(T=T, seed=seed)
    specs = {
        "uniform(ell=15)":    lambda t: uniform_weights(t, 15),
        "exponential(lam=0.9)": lambda t: exponential_weights(t, 0.9),
        "hyperbolic(a=10)":   lambda t: hyperbolic_weights(t, 10),
    }
    # superposition: 50% exponential + 50% hyperbolic
    super_spec = SuperpositionSpec(
        components=[
            lambda t: exponential_weights(t, 0.9),
            lambda t: hyperbolic_weights(t, 10),
        ],
        alphas=[0.5, 0.5],
    )
    specs["superposition(0.5 exp + 0.5 hyp)"] = super_spec.weights

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(mu, color="black", alpha=0.6, lw=1, label=r"latent $\mu_t$")
    for name, wfn in specs.items():
        theta = exact_filter(Y, wfn, loss_name="absolute")
        ax.plot(theta, lw=1, label=name)
    ax.set_title("Weighted-median filter under four weight schemes (absolute loss)")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\theta_t$")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":")
    fig.tight_layout()
    fig.savefig(OUT / "weight_schemes.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    summary = {}
    print("=== Convergence over B ===")
    summary["convergence"] = convergence_over_B(T=1000, seed=1)
    print("\n=== Stratified vs exact ===")
    summary["stratified_vs_exact"] = stratified_vs_exact(T=1000, seed=1, lam=0.9)
    print("\n=== Runtime scaling ===")
    summary["runtime"] = runtime_scaling()
    print("\n=== Weight-scheme showcase ===")
    weight_scheme_showcase()

    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults written to {OUT}")


if __name__ == "__main__":
    main()
