# Repository agent notes

Persistent context for anyone (human or AI) working on this repo.

## Scope

Replication of a paper on **generic weighted-loss filters** with
simulation-based + stratified estimators, applied to synthetic Cauchy
processes (exp_2) and empirical NVDA trade data (exp_3/4).

## Layout

* `src/`       – library code; tests cover each module.
* `exp/`       – scripts that *drive* src and dump into `results/`.
* `tests/`     – pytest, pure library tests (no network).
* `results/`   – deliverable artefacts: PNGs, JSON summaries, RESULTS.md.
* `data/raw/`  – LOCAL cache for Massive REST responses; gitignored.

## Conventions

* All numeric work in `np.float64`.
* `theta_t` always indexed `0..T-1`; `theta_{t-1}` is used for predictive
  loss, so `predictive_abs_loss(Y, theta)` evaluates `|Y_t - theta_{t-1}|`
  — subtle but tested in `tests/test_pipeline_and_metrics.py`.
* Weighted median uses Hazen / `type=5` convention with duplicate
  collapsing (critical for matching the paper's non-unique-median rule).
* Exponential inverse-CDF `W_t^{-1}(p)` is used for the stratified head
  size — this is the fast path, do **not** replace with a search.
* RNG is always `np.random.default_rng(seed)` with explicit seeds passed
  down from the experiment driver.

## Massive API usage

* `src/data_loader.py` wraps `massive.RESTClient.list_trades`.
* Reads key from `MASSIVE_API_KEY` env var — never hard-code.
* Caches trades per `(ticker, date, max_trades)` as Parquet under
  `data/raw/`. Delete that directory to force a refetch.
* Cleaning rules follow the paper exactly:
  * regular session only (09:30:00–16:00:00 ET),
  * drop `correction != 0`,
  * drop extended-hours sale conditions,
  * drop non-positive prices,
  * opening-auction condition `O` is `t = 1`,
  * closing-auction condition `6` is appended at 16:00:00.

## Running experiments

* exp_1 / exp_2 are fully synthetic — no API key needed.
* exp_3 / exp_4 need `MASSIVE_API_KEY`.
* Wall-clock knobs: `NVDA_MAX_TRADES`, `EXP4_TICKERS`, `EXP4_MAX_TRADES`.

## Testing

* `pytest -q` must stay green.
* The test suite deliberately avoids any network call; empirical paths
  are exercised via synthetic trade grids.

## Gotchas

* When `λ` is very close to 1 the exponential normaliser `n_t` is
  computed via the stable `1 − λ^t` form; don't swap for a naive sum.
* Bernoulli log-loss clips probabilities to `[1e-12, 1 − 1e-12]`.
* IRLS for Huber uses a scalar inner loop *per lag* because each lag
  carries its own threshold `δ_{t-j-1}`.
