"""Trade loader.

The paper's tape is the consolidated TAQ feed. Massive returns the same
prints: on 16 October 2024 the raw count is 1,616,806, and the cleaning
rules below reproduce the paper's kept count of 1,521,749 and the final
length T = 1,521,339.

Sale-condition ids are the SIP codes Massive publishes. For a Nasdaq listing
they line up with the TAQ characters used in the paper:

* 17  Market Center Opening Trade, TAQ ``O``  — trade 1
* 8   Closing Prints, TAQ ``6``                — appended at 16:00:00
* 12  Form T / extended hours                  — dropped
* 13  Extended hours, sold out of sequence     — dropped
"""

from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.config import (
    CLOSING_AUCTION_CONDITION,
    DIURNAL_PRESAMPLE_DATES,
    EXTENDED_HOURS_CONDITIONS,
    OPENING_AUCTION_CONDITION,
    PAPER_DATES,
    PRESAMPLE_DATES,
    REGULAR_CLOSE_SECONDS,
    REGULAR_OPEN_SECONDS,
    SESSION_SECONDS,
    STARTUP_S,
    TRADES_PAGE_LIMIT,
)

ET = ZoneInfo("America/New_York")
CACHE = Path(__file__).resolve().parent / "cache" / "trades"
EXTENDED_HOURS_CONDS = set(EXTENDED_HOURS_CONDITIONS)


@dataclass
class LoadedTrades:
    """One cleaned day."""

    ticker: str
    date: str
    df: pd.DataFrame
    T60: int
    raw_count: int
    cleaned_count: int


def _cache_path(ticker: str, date: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / f"{ticker}_{date}_fullday.pkl"


def _day_bounds_ns(date: str) -> tuple[int, int]:
    y, m, d = map(int, date.split("-"))
    start = int(datetime(y, m, d, 0, 0, 0, tzinfo=ET).timestamp() * 1e9)
    end = int(datetime(y, m, d, 23, 59, 59, 999000, tzinfo=ET).timestamp() * 1e9)
    return start, end


def _correction_flag(value) -> int:
    if value is None:
        return 0
    try:
        if pd.isna(value):
            return 0
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def fetch_raw_tuples(ticker: str, date: str, *, use_cache: bool = True) -> list:
    """Full-day prints as ``(ts_ns, price, size, conditions, correction, exchange)``."""
    path = _cache_path(ticker, date)
    if use_cache and path.exists() and path.stat().st_size > 0:
        with path.open("rb") as f:
            return pickle.load(f)

    from massive import RESTClient

    api_key = os.environ.get("MASSIVE_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable MASSIVE_API_KEY is not set")
    client = RESTClient(api_key)
    lo, hi = _day_bounds_ns(date)
    rows = []
    t0 = time.time()
    for trade in client.list_trades(
        ticker,
        timestamp_gte=lo,
        timestamp_lte=hi,
        limit=TRADES_PAGE_LIMIT,
        order="asc",
    ):
        rows.append(
            (
                int(trade.sip_timestamp or trade.participant_timestamp or 0),
                float(trade.price),
                float(trade.size or 0.0),
                tuple(int(c) for c in (trade.conditions or [])),
                _correction_flag(trade.correction),
                int(trade.exchange) if trade.exchange is not None else -1,
            )
        )
    with path.open("wb") as f:
        pickle.dump(rows, f, protocol=4)
    print(f"[{ticker} {date}] fetched {len(rows)} prints in {time.time() - t0:.1f}s")
    return rows


def clean_tuples(raw: list, *, date: str) -> LoadedTrades:
    """Apply the paper's cleaning rules and return the trade panel.

    Order matches the footnote on 16 October: regular-hours clock first,
    then extended-hours sale conditions, then corrections, then non-positive
    prices. The opening auction (condition 17) becomes trade 1. The closing
    auction (condition 8), which the tape stamps a fraction of a second after
    16:00, is appended at exactly 16:00:00.
    """
    raw_count = len(raw)
    midnight_ns = _day_bounds_ns(date)[0]
    kept = []
    closing = None  # (ts, price, size, conds, corr, exchange) with max size
    for row in raw:
        ts, price, size, conds, correction, exchange = row[:6]
        conds = tuple(int(c) for c in conds) if conds is not None else ()
        correction = _correction_flag(correction)
        tau = (int(ts) - midnight_ns) / 1e9
        tau_open = tau - REGULAR_OPEN_SECONDS
        cond_set = set(conds)
        if CLOSING_AUCTION_CONDITION in cond_set and tau_open >= SESSION_SECONDS - 1.0:
            if closing is None or size > closing[2]:
                closing = (ts, price, size, conds, correction, exchange, tau)
        if tau_open < 0.0 or tau_open > SESSION_SECONDS:
            continue
        if cond_set & EXTENDED_HOURS_CONDS:
            continue
        if correction != 0:
            continue
        if price <= 0.0:
            continue
        kept.append((int(ts), float(price), float(size), conds, correction, int(exchange), tau, tau_open))

    kept.sort(key=lambda r: r[0])
    open_at = 0
    for i, row in enumerate(kept):
        if OPENING_AUCTION_CONDITION in row[3]:
            open_at = i
            break
    kept = kept[open_at:]

    if closing is not None:
        ts, price, size, conds, correction, exchange, _tau = closing
        # Drop a copy already inside the session, then pin the auction at 16:00.
        kept = [r for r in kept if CLOSING_AUCTION_CONDITION not in r[3]]
        kept.append(
            (
                int(ts),
                float(price),
                float(size),
                conds,
                0,
                int(exchange),
                float(REGULAR_CLOSE_SECONDS),
                float(SESSION_SECONDS),
            )
        )

    if not kept:
        df = pd.DataFrame(
            columns=["ts_ns", "tau_s", "tau_open", "price", "size", "log_price", "conditions", "correction", "exchange"]
        )
        return LoadedTrades(ticker="", date=date, df=df, T60=0, raw_count=raw_count, cleaned_count=0)

    ts = np.array([r[0] for r in kept], dtype=np.int64)
    price = np.array([r[1] for r in kept], dtype=np.float64)
    size = np.array([r[2] for r in kept], dtype=np.float64)
    tau_s = np.array([r[6] for r in kept], dtype=np.float64)
    tau_open = np.array([r[7] for r in kept], dtype=np.float64)
    exchange = np.array([r[5] for r in kept], dtype=np.int64)
    correction = np.array([r[4] for r in kept], dtype=np.int64)
    df = pd.DataFrame(
        {
            "ts_ns": ts,
            "tau_s": tau_s,
            "tau_open": tau_open,
            "price": price,
            "size": size,
            "log_price": np.log(price),
            "conditions": [r[3] for r in kept],
            "correction": correction,
            "exchange": exchange,
        }
    )
    t60 = int((tau_open <= STARTUP_S).sum())
    return LoadedTrades(
        ticker="",
        date=date,
        df=df,
        T60=t60,
        raw_count=raw_count,
        cleaned_count=int(len(df)),
    )


def load_day(
    ticker: str,
    date: str,
    *,
    max_trades: Optional[int] = None,
    use_cache: bool = True,
) -> LoadedTrades:
    """Fetch (or read the cache) and clean one day.

    ``max_trades`` keeps a prefix of the cleaned panel. The paper's numbers
    use the full day (``max_trades=None``).
    """
    raw = fetch_raw_tuples(ticker, date, use_cache=use_cache)
    out = clean_tuples(raw, date=date)
    out.ticker = ticker
    if max_trades is not None and len(out.df) > max_trades:
        out.df = out.df.iloc[: int(max_trades)].reset_index(drop=True)
        out.cleaned_count = int(len(out.df))
        out.T60 = int((out.df["tau_open"].to_numpy() <= STARTUP_S).sum())
    return out


def previous_trading_dates(date: str, n: int = 5) -> list[str]:
    """The ``n`` paper-sample (or presample) sessions strictly before ``date``."""
    calendar = DIURNAL_PRESAMPLE_DATES + PRESAMPLE_DATES + PAPER_DATES
    if date not in calendar:
        raise KeyError(date)
    i = calendar.index(date)
    return calendar[max(0, i - n) : i]
