"""Helpers shared by the experiment scripts: the results directory, the run
log, JSON conversion and the two-panel spot-volatility plot."""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import PAPER_DATES, SESSION_SECONDS

RESULTS = Path(__file__).resolve().parent.parent / "results"


def log_to_results(name: str) -> Path:
    """Copy everything the run writes to stdout and stderr, including worker
    processes, to ``results/<name>.log``."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.log"
    sys.stdout.flush()
    sys.stderr.flush()
    tee = subprocess.Popen(["tee", str(path)], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())
    tee.stdin.close()
    sys.stdout = os.fdopen(os.dup(1), "w", buffering=1)

    def _close() -> None:
        sys.stdout.flush()
        sys.stderr.flush()
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        sys.stdout.close()
        tee.wait()

    atexit.register(_close)
    return path


def _stack(rows: np.ndarray):
    """Lay twelve one-second session paths end to end, in hours."""
    xs, ys = [], []
    for i, y in enumerate(rows):
        xs.append(i * (SESSION_SECONDS / 3600.0) + np.arange(len(y)) / 3600.0)
        y = y.copy()
        y[-1] = np.nan
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def two_panel(top, bottom, labels, colors, path: Path, ylabel: str = "annualized volatility (%)",
              day: str = "16 October 2024") -> None:
    """Top: one session on the one-second grid. Bottom: the twelve sessions."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharey=True)
    hours = np.arange(top[0].shape[0]) / 3600.0
    for y, lab, col in zip(top, labels, colors):
        axes[0].plot(hours, y, color=col, lw=0.7, label=lab)
    axes[0].set_title(day)
    axes[0].set_xlabel("hours after the open")
    for y, lab, col in zip(bottom, labels, colors):
        xs, ys = _stack(y)
        axes[1].plot(xs, ys, color=col, lw=0.4, label=lab)
    axes[1].set_title("1–16 October 2024")
    axes[1].set_xticks([i * SESSION_SECONDS / 3600.0 for i in range(len(PAPER_DATES))])
    axes[1].set_xticklabels([d[5:] for d in PAPER_DATES], rotation=45, ha="right", fontsize=7)
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        for handle in ax.legend(fontsize=8, loc="upper right").get_lines():
            handle.set_linewidth(1.5)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def jsonify(obj, drop=()):
    """Plain-JSON copy of nested results; keys in ``drop`` are left out."""
    if isinstance(obj, dict):
        return {str(k): jsonify(v, drop) for k, v in obj.items() if k not in drop}
    if isinstance(obj, (list, tuple)):
        return [jsonify(v, drop) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj
