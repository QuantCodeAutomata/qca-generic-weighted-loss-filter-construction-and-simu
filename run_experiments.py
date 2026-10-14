"""Run all four experiments end-to-end.

Experiments 3 and 4 require ``MASSIVE_API_KEY`` and will be skipped with a
warning if the key is absent. Experiments 1 and 2 are fully synthetic.
"""
from __future__ import annotations

import os
import sys
import traceback


def _run(label: str, fn):
    print(f"\n{'=' * 72}\n== {label}\n{'=' * 72}")
    try:
        fn()
        print(f"[OK] {label}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {label}: {exc!r}")
        traceback.print_exc()
        return False


def main() -> int:
    from exp import exp_1_generic_filter, exp_2_cauchy_medians

    ok = True
    ok &= _run("Exp 1: generic weighted-loss filter", exp_1_generic_filter.main)
    ok &= _run("Exp 2: synthetic Cauchy medians", exp_2_cauchy_medians.main)

    if os.environ.get("MASSIVE_API_KEY"):
        from exp import exp_3_nvda_empirical, exp_4_robustness
        ok &= _run("Exp 3: NVDA empirical pipeline", exp_3_nvda_empirical.main)
        ok &= _run("Exp 4: robustness + cross-asset", exp_4_robustness.main)
    else:
        print("\n[skip] MASSIVE_API_KEY not set – exp_3 and exp_4 skipped.")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
