# Generic Weighted-Loss Filter: Construction and Simulation

Replication study of the paper's generic weighted-loss filter together with
its simulation-based and stratified estimators, applied to (i) heavy-tailed
synthetic series and (ii) high-frequency NVDA trade data from October 2024.

The repository implements the full methodology — exact filter, Monte Carlo
simulated filter, and stratified head-plus-tail estimator — across four
convex losses (squared, absolute, Huber, Bernoulli) and four weight schemes
(uniform, exponential, hyperbolic, superposition).

## Repository layout

```
.
├── data/                 # Raw-trade cache (gitignored); fetched via Massive REST
├── src/                  # Core library
│   ├── losses.py            Convex-loss primitives + weighted solvers
│   ├── weights.py           Uniform / exponential / hyperbolic / superposition / anchored
│   ├── filters.py           Exact, simulated, and stratified filter estimators
│   ├── synthetic.py         Cauchy-noise generators (Sections 2–3 of the paper)
│   ├── metrics.py           Predictive loss, tracking error, Hill, ACF, Table-1/2 stats
│   ├── data_loader.py       Massive REST ingestion + exact paper cleaning rules
│   └── empirical_pipeline.py  Five-stage NVDA pipeline (median → scale → Huber → q_t)
├── exp/                  # Experiment drivers that write to results/
│   ├── exp_1_generic_filter.py       Convergence, O(T) runtime, weight-scheme showcase
│   ├── exp_2_cauchy_medians.py       Rolling / anchored / exponential weighted-median
│   ├── exp_3_nvda_empirical.py       NVDA 2024-10-14..16 pipeline + diagnostics
│   └── exp_4_robustness.py           Tail-index, alt lags, simulated signatures, X-asset
├── tests/                # pytest unit + regression tests
├── results/              # Deliverables: plots, JSON summaries, RESULTS.md
├── run_experiments.py    # One-shot driver that executes all four experiments
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

Set the Massive REST key for the empirical experiments (3 and 4):

```bash
export MASSIVE_API_KEY=your_massive_key_here
```

## Running

```bash
# unit + regression tests
pytest -q

# individual experiments
python -m exp.exp_1_generic_filter
python -m exp.exp_2_cauchy_medians
python -m exp.exp_3_nvda_empirical     # needs MASSIVE_API_KEY
python -m exp.exp_4_robustness         # needs MASSIVE_API_KEY

# or all at once
python run_experiments.py
```

Environment knobs (used by exp_3 / exp_4 to keep wall-clock reasonable):

| Variable            | Default                 | Meaning                                       |
| ------------------- | ----------------------- | --------------------------------------------- |
| `NVDA_MAX_TRADES`   | `15000`                 | Max trades per day kept after cleaning        |
| `EXP4_TICKERS`      | `TSLA,AAPL,SPY,AMD,MSFT`| Comma-separated cross-asset tickers (exp_4 D) |
| `EXP4_MAX_TRADES`   | `5000`                  | Max trades per cross-asset day                |

## Methodology adherence

* **Target filter** — `theta_t = argmin_theta Q_t(theta)` with
  `Q_t(theta) = sum_j w_{t,j} [(1-α)·E L(θ, Y_{t-j}) + α·L(θ, Y_{t-j})]`.
  `α = 1` is the empirical special case used throughout exp_2/3.
* **Weight schemes** are implemented *verbatim* from the paper, including the
  closed-form exponential inverse-CDF
  `W_t^{-1}(p) = ceil(log(1-p·(1-λ^t))/log λ) − 1`
  and the digamma-based hyperbolic normalisation `n_t = Ψ(t+a) − Ψ(a)`.
* **Weighted median** uses the Hazen / `type=5` convention with duplicate
  collapsing (`src/losses.py::weighted_median`), matching the paper's
  convention exactly.
* **Stratified estimator** uses the exact head weights up to the 99.9 %
  cumulative-weight cutoff and simulates only the tail.
* **Empirical NVDA pipeline** follows the five stages precisely: median
  filter (3-trade half-life) → scale filter (4-minute clock-time half-life
  on non-zero residuals) → threshold `δ_t = 2·1.4826·b_t` → Huber filter
  via IRLS initialised at `m_t` → directional outlier probability
  `q_t = p_t^+/(p_t^+ + p_t^-)` with 13-s half-life.

## Scope note

The paper's full empirical study is 12 trading days of NVDA with
≈ 1.5 M trades/day. We expose `NVDA_MAX_TRADES` to let the pipeline run
end-to-end on representative windows (default 15 k trades × 3 days) inside
automation budgets. **All code paths generalise unchanged** to the full
sample; raise `NVDA_MAX_TRADES` to reproduce the paper's numbers.

Cross-asset extension (exp_4 D) likewise runs on a 5-ticker subset by
default and generalises to the appendix's 54-asset list via `EXP4_TICKERS`.

## Data handling

Raw trades are fetched from the Massive REST API (`list_trades`) and cached
to `data/raw/` as Parquet. This directory is **gitignored**; the repository
only commits derived results (`results/`) and small, deterministic test
fixtures. Cleaning rules in `src/data_loader.py` mirror the paper's rules:
drop corrected trades, drop extended-hours sale conditions, drop
non-positive prices, enforce opening-auction condition `O` as `t = 1` and
append closing-auction condition `6` at 16:00:00 ET.

## Results

See [`results/RESULTS.md`](results/RESULTS.md) for the compiled summary
of all four experiments.
