# Repository agent notes

Persistent context for anyone (human or AI) working on this repo.

## Scope

Replication of Simon Donker van Heel and Neil Shephard (2026), *Filtering
without recursion and some of its uses in financial economics*,
arXiv:2609.07207 (https://arxiv.org/abs/2609.07207): generic
weighted-loss filters with simulation-based and stratified estimators,
applied to synthetic Cauchy processes (exp_1, exp_2), to the October 2024
NVDA trade tape (exp_3), and to the paper's appendix checks and its 54
further assets (exp_4).

## Layout

* `src/`       – library code; tests cover each module. `src/config.py`
  holds every parameter of the experiments (module-level constants, each
  with its meaning, unit and source in the paper); drivers and library
  defaults read from it.
* `data/`      – `data_loader.py` (Massive trades + cleaning) and
  `generate_synthetic.py` (every simulated series). No loader or generator
  lives in `src/` or `exp/`.
* `exp/`       – scripts that *drive* src and write into `results/`.
* `tests/`     – pytest, pure library tests (no network).
* `results/`   – flat: `expN_*` PNGs and JSON, RESULTS.md. Every number in
  RESULTS.md comes from a file here. `expN.log` is written locally and
  gitignored.
* `data/cache/trades/` – LOCAL cache of Massive trade tapes (one pickle per
  ticker-day); gitignored.
* `data/cache/filtered/` – LOCAL cache of filtered sessions written by
  `src/spot_pipeline.py` (exp_3 spot stage, exp_4 B, F and G); gitignored,
  rebuilt on demand. Delete an asset's files after changing the filters.

## Conventions

* All numeric work in `np.float64`.
* `theta_t` always indexed `0..T-1`; `theta_{t-1}` is used for predictive
  loss, so `predictive_abs_loss(Y, theta)` evaluates `|Y_t - theta_{t-1}|`
  — subtle but tested in `tests/test_pipeline_and_metrics.py`.
* Two weighted-median conventions, both from the paper:
  * the generic filters of Sections 2–3 (exp_2 parts A/B) use Hazen /
    `type=5` interpolation with duplicate collapsing (Remark 3);
  * Theorem 2 and the high-frequency median filter use the *lower* median,
    the smallest value whose cumulative weight reaches one half.
* Exponential inverse-CDF `W_t^{-1}(p)` is used for the stratified head
  size — this is the fast path, do **not** replace with a search.
* Stratified estimator (Remark 4(d)): head `h = W_t^{-1}(0.999)` evaluated
  exactly, plus `B = 100` tail draws with `P(j | j > h) ∝ w_j`, each carrying
  `tail_mass / B`. Every empirical driver uses `B = 100`.
* RNG is always `np.random.default_rng(seed)` with explicit seeds passed
  down from the experiment driver. Numba kernels are seeded from that
  generator.
* Spot volatility memory `a` (Mandelbrot weights `1/(j + a)`) is selected
  **per asset** by QLIKE (Section 4.6, Appendix B). `NVDA_A_PUBLISHED = 563`
  is only the paper's published NVDA value; it is used only by the check
  that fixes `a` at it.
* Seeds of the cached sessions depend on the asset's position in
  `config.ASSETS`; append new assets at the end.

## Massive API usage

* `data/data_loader.py` wraps `massive.RESTClient.list_trades`.
* Reads key from `MASSIVE_API_KEY` env var — never hard-code.
* Every loader call reads `data/cache/trades/` first, so the experiments run offline
  once the tapes are cached. Delete a pickle to force a refetch.
* Cleaning rules follow the paper exactly:
  * regular session only (09:30:00–16:00:00 ET),
  * drop `correction != 0`,
  * drop extended-hours sale conditions (Massive ids 12, 13),
  * drop non-positive prices,
  * opening print (TAQ `O`, Massive 17) is `t = 1`,
  * closing print (TAQ `6`, Massive 8) is appended at 16:00:00.

## Running experiments

* exp_1 / exp_2 are fully synthetic — no API key needed.
* exp_3 / exp_4 need the cached tapes or `MASSIVE_API_KEY`.
* `./reproduce.sh` runs the tests and all four experiments at the paper's
  configuration; there are no environment knobs.
* `exp_3 --stages days spot sensitivity` and `exp_4 --modules A ... G`
  select parts. exp_4 merges the modules it runs into
  `results/exp4_summary.json`, so modules can be rerun one at a time.
* Each driver tees its stdout and stderr (worker processes included) into
  `results/expN.log`. That file is gitignored.

## Testing

* `pytest -q` must stay green.
* The test suite deliberately avoids any network call; empirical paths
  are exercised via synthetic trade grids.

## Gotchas

* When `λ` is very close to 1 the exponential normaliser `n_t` is
  computed via the stable `1 − λ^t` form; don't swap for a naive sum.
* Bernoulli log-loss clips probabilities to `[1e-12, 1 − 1e-12]`.
* IRLS for Huber uses a scalar inner loop *per lag* because each lag
  carries its own threshold `δ_{t-j-1}`; the opening print uses the
  presample-only `δ_0`.
* With a 3-trade half-life `λ^3 = 1/2`, so after a repricing the three
  newest trades hold exactly half the weight. The high-frequency median
  breaks that tie in favour of the side holding the newest trade (exact
  arithmetic, with a relative tolerance of 1e-12). Leaving the tie to
  floating-point rounding moves the lag-1 Huber rms of Table 1 from 0.435 to
  0.429 cents.
