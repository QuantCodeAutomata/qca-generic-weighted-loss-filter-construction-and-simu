#!/usr/bin/env bash
# Runs the test suite and the four experiments at the paper's configuration
# (src/config.py). Each experiment writes its outputs to results/. The console
# log results/expN.log is gitignored. Experiments 3 and 4 need MASSIVE_API_KEY unless the trades
# are already cached under data/cache/trades/.
set -euo pipefail
cd "$(dirname "$0")"

PY="$(command -v python3 || command -v python)"

"$PY" -m pytest -q
"$PY" -m exp.exp_1_generic_filter
"$PY" -m exp.exp_2_cauchy_medians
"$PY" -m exp.exp_3_nvda_empirical
"$PY" -m exp.exp_4_robustness
