"""
Experiment 1
============

Generic Weighted-Loss Filter Construction and Simulation Approximation
----------------------------------------------------------------------

Validates (a) exact vs. simulated vs. stratified filter agreement,
(b) convergence of simulation error with Monte Carlo size ``B``, measured as
    ``|θ_t − θ*_t|``, the objective gap ``Q_t(θ*_t) − Q_t(θ_t)`` and the
    change in one-step predictive loss,
(c) O(T) runtime at fixed ``B``: the lag is drawn by the closed-form
    quantile of the weights (Section 3.4), so each ``t`` costs O(B), against
    the O(T²) exact weighted median,
(d) applicability across convex losses (squared, absolute, Huber, Bernoulli)
    and weight schemes (uniform, exponential, hyperbolic, superposition),
(e) Theorem 1: for squared loss the simulated filter is conditionally
    unbiased and ``B Var(θ*_t | Y) → σ_t^2 = Σ_j w_{t,j} (Y_{t−j} − θ_t)^2``.

Results go to ``results/`` with the prefix ``exp1_``; the run log is
``results/exp1.log``.
"""

from __future__ import annotations

import json
import time
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.generate_synthetic import iid_cauchy, nonstationary_cauchy
from exp.common import RESULTS, log_to_results
from src.config import (
    EXP1_B_GRID,
    EXP1_HUBER_DELTA,
    EXP1_HYPERBOLIC_A,
    EXP1_LAG_SEED,
    EXP1_LAMBDA,
    EXP1_RUNTIME_B,
    EXP1_RUNTIME_EXACT_MAX_T,
    EXP1_RUNTIME_SEED,
    EXP1_RUNTIME_T,
    EXP1_SEED,
    EXP1_SHOWCASE_SEED,
    EXP1_SHOWCASE_T,
    EXP1_STRATIFIED_B,
    EXP1_STRATIFIED_SEED,
    EXP1_SUPERPOSITION_SHARES,
    EXP1_T,
    EXP1_UNIFORM_ELL,
    STRATIFIED_HEAD_MASS,
    THEOREM1_B,
    THEOREM1_REPS,
    THEOREM1_SEED,
    THEOREM1_TIMES,
)
from src.losses import LOSSES
from src.filters import (
    exact_filter,
    exponential_lag_sampler,
    objective_value,
    simulated_filter,
    simulated_filter_fast,
    stratified_exponential_filter,
    stratified_filter,
)
from src.weights import (
    SuperpositionSpec,
    exponential_inv_cum_weight,
    exponential_weights,
    hyperbolic_weights,
    uniform_weights,
)


PREFIX = "exp1_"


def _mean_abs_gap(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


HUBER_DELTA = EXP1_HUBER_DELTA


def _objectives(Y: np.ndarray, wfn, loss: str, theta: np.ndarray) -> np.ndarray:
    """``Q_t(θ_t) = Σ_j w_{t,j} L(θ_t, Y_{t−j})`` for every t; the gap
    ``Q_t(θ*_t) − Q_t(θ_t) ≥ 0`` measures the simulation error in objective units."""
    out = np.empty(len(Y))
    for t in range(1, len(Y) + 1):
        y_win = Y[:t][::-1]
        d = np.full(t, HUBER_DELTA) if loss == "huber" else None
        out[t - 1] = objective_value(float(theta[t - 1]), y_win, wfn(t), loss, d)
    return out


def _predictive_loss(Y: np.ndarray, theta: np.ndarray, loss: str) -> float:
    """``(1/(T−1)) Σ_{t≥2} L(θ_{t−1}, Y_t)``, the one-step-ahead loss."""
    vals = []
    for t in range(1, len(Y)):
        y = Y[t : t + 1]
        if loss == "huber":
            v = LOSSES[loss]["loss"](float(theta[t - 1]), y, HUBER_DELTA)
        else:
            v = LOSSES[loss]["loss"](float(theta[t - 1]), y)
        vals.append(float(v[0]))
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# A. Convergence over B
# ---------------------------------------------------------------------------

def convergence_over_B(T: int = EXP1_T, seed: int = EXP1_SEED) -> Dict:
    """Approximation error vs. B for several loss / weight combinations."""
    Y, mu = nonstationary_cauchy(T=T, seed=seed)
    binary = (Y > 0).astype(np.float64)  # proxy binary outcome for Bernoulli

    def weight_fn_exp(lam):
        return lambda t: exponential_weights(t, lam)

    def weight_fn_unif(ell):
        return lambda t: uniform_weights(t, ell)

    def weight_fn_hyp(a):
        return lambda t: hyperbolic_weights(t, a)

    lam, ell, a = EXP1_LAMBDA, EXP1_UNIFORM_ELL, EXP1_HYPERBOLIC_A
    scenarios = [
        (f"absolute_exp_lam{lam:g}", "absolute", weight_fn_exp(lam), Y),
        (f"squared_unif_ell{ell}", "squared", weight_fn_unif(ell), Y),
        (f"huber_hyp_a{a:g}", "huber", weight_fn_hyp(a), Y),
        (f"bernoulli_exp_lam{lam:g}", "bernoulli", weight_fn_exp(lam), binary),
    ]

    Bs = list(EXP1_B_GRID)
    rows = []
    for name, loss, wfn, series in scenarios:
        theta_exact = exact_filter(series, wfn, loss_name=loss)
        q_exact = _objectives(series, wfn, loss, theta_exact)
        pl_exact = _predictive_loss(series, theta_exact, loss)
        for B in Bs:
            theta_sim = simulated_filter(
                series, wfn, loss_name=loss, B=B, rng=np.random.default_rng(EXP1_LAG_SEED)
            )
            gap = _mean_abs_gap(theta_exact, theta_sim)
            obj_gap = float(np.mean(_objectives(series, wfn, loss, theta_sim) - q_exact))
            pl_gap = float(_predictive_loss(series, theta_sim, loss) - pl_exact)
            rows.append({"scenario": name, "loss": loss, "B": B, "mae": gap,
                         "objective_gap": obj_gap, "predictive_loss_gap": pl_gap,
                         "predictive_loss_exact": pl_exact})
            print(f"{name:28s} B={B:>6d}  MAE={gap:.6f}  dQ={obj_gap:.2e}  dPL={pl_gap:+.2e}")

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
    fig.savefig(RESULTS / f"{PREFIX}convergence_vs_B.png", dpi=140)
    plt.close(fig)

    return {"rows": rows, "Bs": Bs}


# ---------------------------------------------------------------------------
# B. Stratified vs exact
# ---------------------------------------------------------------------------

def stratified_vs_exact(T: int = EXP1_T, seed: int = EXP1_SEED, lam: float = EXP1_LAMBDA) -> Dict:
    Y, _ = nonstationary_cauchy(T=T, seed=seed)
    results = {}
    for loss in ["absolute", "squared", "huber"]:
        theta_e = exact_filter(Y, lambda t: exponential_weights(t, lam), loss_name=loss)
        theta_s = stratified_filter(
            Y,
            lambda t: exponential_weights(t, lam),
            lambda t: exponential_inv_cum_weight(STRATIFIED_HEAD_MASS, t, lam),
            loss_name=loss,
            B=EXP1_STRATIFIED_B,
            rng=np.random.default_rng(EXP1_STRATIFIED_SEED),
        )
        gap = _mean_abs_gap(theta_e, theta_s)
        results[loss] = gap
        print(f"stratified vs exact  loss={loss:10s}  MAE={gap:.3e}")
    return results


# ---------------------------------------------------------------------------
# C. Runtime scaling in T (approximately O(T))
# ---------------------------------------------------------------------------

def runtime_scaling(Ts=EXP1_RUNTIME_T, B: int = EXP1_RUNTIME_B, lam: float = EXP1_LAMBDA,
                    exact_max_T: int = EXP1_RUNTIME_EXACT_MAX_T) -> Dict:
    """Wall-clock of the O(1)-per-t estimators (Definition 3 and Remark 4(d))
    against the exact weighted median, which costs O(t) per point."""
    rows = []
    draw = exponential_lag_sampler(lam)
    for T in Ts:
        Y = iid_cauchy(T, seed=EXP1_RUNTIME_SEED)
        t0 = time.perf_counter()
        _ = stratified_exponential_filter(Y, lam, "absolute", B=B, rng=np.random.default_rng(EXP1_RUNTIME_SEED))
        t_strat = time.perf_counter() - t0
        t0 = time.perf_counter()
        _ = simulated_filter_fast(Y, draw, "absolute", B=B, rng=np.random.default_rng(EXP1_RUNTIME_SEED))
        t_sim = time.perf_counter() - t0
        row = {"T": T, "seconds_stratified": t_strat, "seconds_simulated": t_sim,
               "us_per_point_stratified": 1e6 * t_strat / T, "us_per_point_simulated": 1e6 * t_sim / T}
        if T <= exact_max_T:
            t0 = time.perf_counter()
            _ = exact_filter(Y, lambda t: exponential_weights(t, lam), loss_name="absolute", interpolation="lower")
            row["seconds_exact"] = time.perf_counter() - t0
            row["us_per_point_exact"] = 1e6 * row["seconds_exact"] / T
        rows.append(row)
        print(f"T={T:>7d}  stratified {1e6*t_strat/T:6.1f} us/pt  simulated {1e6*t_sim/T:6.1f} us/pt"
              + (f"  exact {row['us_per_point_exact']:8.1f} us/pt" if "us_per_point_exact" in row else ""))

    xs = [r["T"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, label in (("seconds_stratified", "stratified, h = W⁻¹(0.999)"), ("seconds_simulated", "simulated (Definition 3)")):
        ys = [r[key] for r in rows]
        ax.loglog(xs, ys, "o-", label=label)
    ex = [(r["T"], r["seconds_exact"]) for r in rows if "seconds_exact" in r]
    ax.loglog([e[0] for e in ex], [e[1] for e in ex], "s-", color="0.4", label="exact weighted median")
    ys = [r["seconds_stratified"] for r in rows]
    ax.loglog(xs, ys[0] * (np.array(xs) / xs[0]), "k--", alpha=0.5, label="O(T) reference")
    ax.set_xlabel("T")
    ax.set_ylabel("seconds")
    ax.set_title(f"Runtime, exponential median, B={B}")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle=":")
    fig.tight_layout()
    fig.savefig(RESULTS / f"{PREFIX}runtime_scaling.png", dpi=140)
    plt.close(fig)
    return {"rows": rows, "B": B}


def theorem1_check(T: int = EXP1_T, lam: float = EXP1_LAMBDA, B: int = THEOREM1_B, reps: int = THEOREM1_REPS,
                   seed: int = THEOREM1_SEED) -> Dict:
    """Monte Carlo over the lag draws at fixed data, squared loss."""
    Y, _ = nonstationary_cauchy(T=T, seed=seed)
    draw = exponential_lag_sampler(lam)
    rng = np.random.default_rng(seed)
    out = {}
    for t in THEOREM1_TIMES:
        w = exponential_weights(t, lam)
        y_win = Y[:t][::-1]
        theta = float(w @ y_win)
        sigma2 = float(w @ (y_win - theta) ** 2)
        sims = np.array([Y[t - 1 - draw(rng, t, B)].mean() for _ in range(reps)])
        out[str(t)] = {
            "theta": theta,
            "mean_theta_star": float(sims.mean()),
            "bias_over_se": float((sims.mean() - theta) / (sims.std(ddof=1) / np.sqrt(reps))),
            "B_var": float(B * sims.var(ddof=1)),
            "sigma2_t": sigma2,
            "ratio": float(B * sims.var(ddof=1) / sigma2),
        }
        print(f"t={t:5d} B*Var/sigma2={out[str(t)]['ratio']:.3f} bias/se={out[str(t)]['bias_over_se']:+.2f}")
    return {"B": B, "reps": reps, "lam": lam, "by_t": out}


# ---------------------------------------------------------------------------
# D. Multiple weight schemes with the same loss (illustrative paths)
# ---------------------------------------------------------------------------

def weight_scheme_showcase(T: int = EXP1_SHOWCASE_T, seed: int = EXP1_SHOWCASE_SEED) -> None:
    Y, mu = nonstationary_cauchy(T=T, seed=seed)
    lam, ell, a = EXP1_LAMBDA, EXP1_UNIFORM_ELL, EXP1_HYPERBOLIC_A
    specs = {
        f"uniform(ell={ell})": lambda t: uniform_weights(t, ell),
        f"exponential(lam={lam:g})": lambda t: exponential_weights(t, lam),
        f"hyperbolic(a={a:g})": lambda t: hyperbolic_weights(t, a),
    }
    share_exp, share_hyp = EXP1_SUPERPOSITION_SHARES
    super_spec = SuperpositionSpec(
        components=[
            lambda t: exponential_weights(t, lam),
            lambda t: hyperbolic_weights(t, a),
        ],
        alphas=[share_exp, share_hyp],
    )
    specs[f"superposition({share_exp:g} exp + {share_hyp:g} hyp)"] = super_spec.weights

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
    fig.savefig(RESULTS / f"{PREFIX}weight_schemes.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log_to_results("exp1")
    summary = {}
    print("=== Convergence over B ===")
    summary["convergence"] = convergence_over_B()
    print("\n=== Stratified vs exact ===")
    summary["stratified_vs_exact"] = stratified_vs_exact()
    print("\n=== Runtime scaling ===")
    summary["runtime"] = runtime_scaling()
    print("\n=== Theorem 1 ===")
    summary["theorem1"] = theorem1_check()
    print("\n=== Weight-scheme showcase ===")
    weight_scheme_showcase()

    with open(RESULTS / f"{PREFIX}summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults written to {RESULTS}")


if __name__ == "__main__":
    main()
