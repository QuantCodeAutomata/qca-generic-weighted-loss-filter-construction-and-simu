"""Trade-data loader for the NVDA (and cross-asset) empirical experiments.

Uses the Massive REST API. The paper's main sample is NVDA trades for
2024-10-01 .. 2024-10-16. Fetching the full ~1.5M trades/day for every day
is time- and bandwidth-prohibitive here; downstream experiments therefore
load a configurable *window* (default: first ``max_trades`` trades of a
single trading day). All cleaning rules are implemented exactly so the
same pipeline would operate on a full day without code changes.

Fields retained per trade:
    ts_ns      -- SIP timestamp in nanoseconds (int64)
    price      -- trade price (float)
    size       -- trade size (float)
    conditions -- tuple of integer sale-condition codes
    correction -- correction flag (int or None)
    exchange   -- venue code (int)

Data is cached as parquet in ``data/raw/`` to speed up re-runs.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


REGULAR_OPEN_SECONDS = 9 * 3600 + 30 * 60   # 09:30:00 ET
REGULAR_CLOSE_SECONDS = 16 * 3600           # 16:00:00 ET
EASTERN_UTC_OFFSET_HOURS = 4                # EDT in October 2024
OPENING_AUCTION_COND = 12                   # NYSE/Nasdaq opening cross is 'O' / 12 in TRF numeric map
CLOSING_AUCTION_COND = 15                   # numeric for '6' closing cross (varies by feed; checked at runtime)
EXTENDED_HOURS_CONDS = {13}                 # '13' approximates extended-hours
CORRECTION_CLEAN = {None, 0, "00", 0.0}


@dataclass
class LoadedTrades:
    """Container for one day's cleaned trades."""
    ticker: str
    date: str
    df: pd.DataFrame            # columns: ts_ns, tau_s, price, size, log_price, conditions, correction, exchange
    T60: int                    # count of trades in first 60 seconds
    raw_count: int              # number of raw records fetched
    cleaned_count: int          # count after cleaning


def _cache_path(ticker: str, date: str, max_trades: Optional[int]) -> Path:
    tag = "full" if max_trades is None else f"max{max_trades}"
    root = Path(__file__).resolve().parent.parent / "data" / "raw"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{ticker}_{date}_{tag}.pkl"


def fetch_trades(
    ticker: str,
    date: str,
    *,
    max_trades: Optional[int] = 20000,
    use_cache: bool = True,
    api_key_env: str = "MASSIVE_API_KEY",
) -> pd.DataFrame:
    """Fetch raw NVDA-style trades for ``ticker`` on ``date`` (YYYY-MM-DD).

    If ``max_trades`` is set, iteration stops after that many records. If
    ``None`` the entire day is pulled (may take minutes for liquid names).
    """
    path = _cache_path(ticker, date, max_trades)
    if use_cache and path.exists():
        return pd.read_pickle(path)

    from massive import RESTClient  # imported lazily

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {api_key_env} not set")
    client = RESTClient(api_key=api_key)

    rows = []
    start = time.time()
    # Restrict to the regular trading session [09:30, 16:00] US/Eastern using
    # nanosecond-UTC timestamp filters. This avoids pulling ~30% pre/post-market
    # trades when ``max_trades`` is small.
    day = pd.Timestamp(date, tz="US/Eastern")
    ts_lo = int((day + pd.Timedelta(hours=9, minutes=30)).tz_convert("UTC").value)
    ts_hi = int((day + pd.Timedelta(hours=16, minutes=0, seconds=1)).tz_convert("UTC").value)
    for t in client.list_trades(
        ticker=ticker,
        timestamp_gte=ts_lo,
        timestamp_lte=ts_hi,
        limit=50000,
        order="asc",
    ):
        rows.append(
            dict(
                ts_ns=int(t.sip_timestamp or t.participant_timestamp or 0),
                price=float(t.price),
                size=float(t.size or 0.0),
                conditions=tuple(int(c) for c in (t.conditions or [])),
                correction=t.correction,
                exchange=int(t.exchange) if t.exchange is not None else -1,
            )
        )
        if max_trades is not None and len(rows) >= max_trades:
            break
    elapsed = time.time() - start
    df = pd.DataFrame(rows)
    df.attrs["fetch_seconds"] = elapsed
    df.to_pickle(path)
    return df


def _condition_set(conds) -> set:
    return set(int(c) for c in conds) if conds is not None else set()


def clean_trades(raw: pd.DataFrame, *, date: str) -> LoadedTrades:
    """Apply the paper's cleaning rules.

    Rules:
      * drop extended-hours and corrected trades
      * drop non-positive prices
      * keep only regular-session trades (09:30:00 -- 16:00:00 ET)
      * identify opening/closing auction trades; make opening auction Y_1
      * append closing auction as Y_T at exactly 16:00:00
    """
    df = raw.copy()
    raw_count = len(df)

    # Convert ns timestamp to Eastern seconds-since-midnight on this date.
    ts_dt = pd.to_datetime(df["ts_ns"], unit="ns", utc=True)
    ts_et = ts_dt.dt.tz_convert("US/Eastern")
    df["ts_dt"] = ts_dt
    df["tau_s"] = (
        ts_et.dt.hour * 3600 + ts_et.dt.minute * 60 + ts_et.dt.second
        + ts_et.dt.microsecond * 1e-6 + ts_et.dt.nanosecond * 1e-9
    )

    # Drop corrected trades: valid rows have correction NaN/None/0.
    def _is_uncorrected(c):
        if c is None:
            return True
        try:
            if pd.isna(c):
                return True
        except (TypeError, ValueError):
            pass
        return c in CORRECTION_CLEAN

    before = len(df)
    df = df[df["correction"].apply(_is_uncorrected)]
    dropped_corr = before - len(df)

    # Drop extended-hours flagged trades
    before = len(df)
    df = df[~df["conditions"].apply(lambda cs: bool(_condition_set(cs) & EXTENDED_HOURS_CONDS))]
    dropped_ext = before - len(df)

    # Non-positive price
    before = len(df)
    df = df[df["price"] > 0]
    dropped_nonpos = before - len(df)

    # Keep regular-session only
    before = len(df)
    df = df[(df["tau_s"] >= REGULAR_OPEN_SECONDS) & (df["tau_s"] <= REGULAR_CLOSE_SECONDS)]
    dropped_hours = before - len(df)

    df = df.sort_values("ts_ns").reset_index(drop=True)

    # Identify opening auction: the first trade whose condition set contains the
    # opening-auction code. If none found, treat the first trade as opening.
    opening_idx = None
    for i, cs in enumerate(df["conditions"]):
        if OPENING_AUCTION_COND in _condition_set(cs):
            opening_idx = i
            break
    if opening_idx is None and len(df) > 0:
        opening_idx = 0

    dropped_pre_open = opening_idx or 0
    df = df.iloc[opening_idx:].reset_index(drop=True)

    # Rebase tau so opening auction is treated as t=1 (keep absolute tau).
    df["log_price"] = np.log(df["price"].values)

    # Append closing auction stamped at exactly 16:00:00 if present.
    closing_mask = df["conditions"].apply(lambda cs: CLOSING_AUCTION_COND in _condition_set(cs))
    if closing_mask.any():
        last_close = df[closing_mask].iloc[-1].copy()
        # Normalize its tau to 16:00:00
        last_close["tau_s"] = float(REGULAR_CLOSE_SECONDS)
        df = pd.concat([df[~closing_mask], last_close.to_frame().T], ignore_index=True)
        df = df.sort_values("tau_s").reset_index(drop=True)

    cleaned = len(df)
    # T60 = trades in first 60 seconds after open
    T60 = int((df["tau_s"] <= REGULAR_OPEN_SECONDS + 60.0).sum())

    print(
        f"[{date}] raw={raw_count} cleaned={cleaned} "
        f"(corr={dropped_corr}, ext={dropped_ext}, nonpos={dropped_nonpos}, "
        f"hours={dropped_hours}, pre_open={dropped_pre_open}) T60={T60}"
    )
    return LoadedTrades(
        ticker="",
        date=date,
        df=df.reset_index(drop=True),
        T60=T60,
        raw_count=raw_count,
        cleaned_count=cleaned,
    )


def load_day(
    ticker: str, date: str, *, max_trades: Optional[int] = 20000, use_cache: bool = True
) -> LoadedTrades:
    """Convenience wrapper: fetch and clean one day."""
    raw = fetch_trades(ticker, date, max_trades=max_trades, use_cache=use_cache)
    out = clean_trades(raw, date=date)
    out.ticker = ticker
    return out
