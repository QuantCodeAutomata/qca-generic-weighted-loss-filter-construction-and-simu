# Filtering without recursion: replication

Simon Donker van Heel and Neil Shephard (2026). *Filtering without recursion
and some of its uses in financial economics*. arXiv:2609.07207.
<https://arxiv.org/abs/2609.07207>

The paper defines a filter as the minimizer of a weighted sum of losses over
past observations, `θ_t = argmin_θ Σ_j w_{t,j} L(θ, Y_{t−j})`, instead of a
recursion. Any convex loss and any weight scheme can be used. Expensive
weights are handled by simulating lags (Definition 3) or by an exact head
plus a simulated tail (the stratified estimator, Remark 4(d)). The applied
part filters every trade of NVDA in October 2024 with a robust Huber filter,
and builds volatility estimates on the filtered prices.

This repository implements the methodology and re-runs the paper's tables
and figures on the same tape: Tables 1–3, A.1, B.1 and B.2 (all 55 assets),
Figures 3–13, A.1–A.6, B.1 and B.2. `results/RESULTS.md` compares every
number with the paper and lists what is not covered.

## Repository layout

```
.
├── data/                 # Data access, synthetic data, local caches
│   ├── data_loader.py       Massive REST ingestion + the paper's cleaning rules
│   ├── generate_synthetic.py  Cauchy-noise signals (Sections 2–3), Brownian and
│   │                          variance-gamma prices with Student-t noise (Appendix A.7)
│   └── cache/               Gitignored: trades/ (raw prints) and filtered/ (filtered sessions)
├── src/                  # Core library
│   ├── config.py            Every parameter of the four experiments, with its source in the paper
│   ├── losses.py            Convex-loss primitives + weighted solvers
│   ├── weights.py           Uniform / exponential / hyperbolic / superposition weights
│   ├── filters.py           Exact, anchored, simulated and stratified filters, O(1) lag samplers
│   ├── hf_filters.py        Trade-by-trade median / Huber / mean filters, clock-time scale,
│   │                        outlier probabilities, damping factor c_{t,s}
│   ├── empirical_pipeline.py  The five filters of Section 4 (median → scale → Huber → q_t)
│   ├── spot_vol.py          Diurnal curve, hyperbolic and HExp spot volatility, QLIKE choice of a
│   ├── spot_pipeline.py     Per-session filtering cache, spot volatility per asset, Table B.2 rows
│   └── metrics.py           Predictive loss, Hill index, ACF, Table 1/2/3 statistics, ψ*
├── exp/                  # Experiment drivers that write to results/
│   ├── exp_1_generic_filter.py   Convergence in B, Theorem 1, O(T) runtime, weight showcase
│   ├── exp_2_cauchy_medians.py   Rolling / anchored / exponential medians, Theorem 2, Figures 1–5, Example 9
│   ├── exp_3_nvda_empirical.py   NVDA, 12 sessions: the five filters, Tables 1–3, Figures 6–13,
│   │                             half-life choices, diurnal spot volatility, sensitivity checks
│   ├── exp_4_robustness.py       Appendices A and B: Table A.1, s = 10, rolling ψ*, startup rule,
│   │                             simulated signatures, 54 further assets (Tables B.1–B.2),
│   │                             sensitivity of the Table B.2 columns
│   └── common.py                 Logging to results/, plot and JSON helpers
├── tests/                # pytest unit + regression tests (no network)
├── results/              # Deliverables: expN_* plots and JSON, RESULTS.md
├── reproduce.sh          # Tests, then the four experiments
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

The empirical experiments read the trade tapes from `data/cache/trades/` and
fetch any missing session from the Massive REST API. Fetching needs a key:

```bash
export MASSIVE_API_KEY=your_massive_key_here
```

## Running

```bash
./reproduce.sh
```

runs `pytest -q` and then, from this directory,

```bash
python -m exp.exp_1_generic_filter
python -m exp.exp_2_cauchy_medians
python -m exp.exp_3_nvda_empirical
python -m exp.exp_4_robustness
```

All four run at the paper's configuration, set in `src/config.py`. Each
writes its figures and JSON summaries to `results/` with the prefix `expN_`.
Console output is also teed to `results/expN.log`, which is gitignored.
Parts can be rerun on their
own: `python -m exp.exp_3_nvda_empirical --stages spot sensitivity` (stages
`days`, `spot`, `sensitivity`) and `python -m exp.exp_4_robustness --modules
F G` (modules `A` to `G`); exp_4 merges the modules it runs into
`results/exp4_summary.json`.

Rough wall-clock on four cores with the tapes cached: exp_1 and exp_2 a few
minutes each; exp_3 about an hour and a half (twelve sessions, each with
100 Brownian paths for `c_{t,s}`, plus the spot-volatility and sensitivity
stages); exp_4 several hours (Module E filters 400 simulated days, Module F
filters 55 × 22 sessions).

## Methodology adherence

* **Target filter** — `θ_t = argmin_θ Σ_j w_{t,j} [(1−α)·E L(θ, Y_{t−j}) + α·L(θ, Y_{t−j})]`;
  `α = 1` is the empirical case.
* **Weights** follow the paper, including the closed-form exponential
  inverse CDF `W_t^{-1}(p) = ceil(log(1 − p(1 − λ^t))/log λ) − 1` and the
  digamma normalization `n_t = Ψ(t + a) − Ψ(a)` for hyperbolic weights.
  Lags are drawn in O(1) per draw by inverse CDF.
* **Weighted medians.** Sections 2–3 use Hazen / type-5 interpolation
  (Remark 3). Theorem 2 and the high-frequency median filter use the lower
  median. With a 3-trade half-life the three newest trades can hold exactly
  half of the weight; the tie goes to the side of the newest trade.
* **Stratified estimator** — exact head up to `h = W_t^{-1}(0.999)` (lags 0
  to 29 at a 3-trade half-life) plus `B = 100` tail draws per trade, as in
  the paper.
* **Empirical pipeline** — median filter (3-trade half-life) → clock-time
  scale of the nonzero median residuals (4-minute half-life, 60-second
  startup blended with the previous five openings) → threshold
  `δ_t = 2 · 1.4826 · b_t` → Huber filter by IRLS with lag-specific
  thresholds `δ_{t−j−1}` → directional outlier probability
  `q_t = p_t^+/(p_t^+ + p_t^−)` with a 13-second clock half-life, the
  half-life selected by predictive Bernoulli log-likelihood.
* **Volatility** — integrated volatility from s-second increments of the
  filtered price, each scaled up by the damping factor `c_{t,s}` (simulated
  per filter for Table 2, the closed form of the 30-lag linear filter
  elsewhere); spot volatility with Mandelbrot weights `1/(j + a)` in trade
  time on diurnally adjusted 1-second increments. `a` is chosen per asset by
  QLIKE: the spot estimate at `u − 60` seconds forecasts the 10-second
  uniform estimate at `u`.

## Data

* `data/cache/trades/` holds one pickle per ticker and session with every
  Massive print of the day `(timestamp, price, size, conditions, correction,
  exchange)`. The experiments use 22 sessions per asset: the 12 paper
  sessions (1–16 October 2024) and the 10 sessions before them, which supply
  the opening presample and the diurnal curve. NVDA and the 54 assets of
  Table B.2 need about 7 GB. It is filled on first use; delete it to force a
  refetch.
* `data/cache/filtered/` stores one filtered session per asset and day
  (`src/spot_pipeline.py`), reused by exp_3 (spot stage) and exp_4 Modules B,
  F and G (about 4.3 GB for 55 assets).
* Cleaning rules in `data/data_loader.py` mirror the paper: regular session
  only, drop corrected trades, drop extended-hours conditions, drop
  non-positive prices, the opening print is `t = 1`, the closing print is
  appended at 16:00:00. On 16 October 2024 this gives the paper's
  `T = 1,521,339` NVDA trades from 1,616,806 raw prints.

## Results

See [`results/RESULTS.md`](results/RESULTS.md) for every reproduced table
and figure, with the published values alongside.
