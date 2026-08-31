"""Data-loading utilities.

Fetches daily adjusted prices via the Massive API (env var ``MASSIVE_API_KEY``)
with a local Parquet cache under ``data/``. Yahoo Finance is used as a
fallback only if explicitly requested.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)


PORTFOLIOS: Dict[str, List[str]] = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "FDS", "NFLX", "ADBE", "CRM"],
    "Old Economy": ["XOM", "CVX", "JNJ", "PFE", "PG", "KO", "BA", "CAT", "VZ", "DUK"],
    "Finance":    ["JPM", "BAC", "WFC", "GS", "MS", "AXP", "USB", "BLK", "SCHW", "MET"],
    "Commodities":["SLB", "HAL", "EOG", "MPC", "NEM", "FCX", "BHP", "NUE", "MOS", "CF"],
}


# Paper's crisis windows — start dates chosen using only pre-crisis information.
# Documented as implementation assumptions in results/RESULTS.md.
CRISIS_STARTS: Dict[str, str] = {
    "GFC_2008":    "2008-09-15",  # Lehman collapse
    "Downgrade_2011": "2011-08-05",  # S&P U.S. downgrade
    "COVID_2020":  "2020-02-20",  # market peak before COVID drawdown
}


def _cache_path(ticker: str) -> Path:
    return DATA_DIR / f"{ticker}.parquet"


def _fetch_one_massive(ticker: str, from_: str, to: str) -> pd.Series:
    """Fetch a single ticker's daily closes from the Massive API."""
    from massive import RESTClient  # local import to keep tests light

    api_key = os.environ.get("MASSIVE_API_KEY") or os.environ.get("MASSIVE_TOKEN")
    if not api_key:
        raise RuntimeError("MASSIVE_API_KEY / MASSIVE_TOKEN env var not set.")
    client = RESTClient(api_key=api_key)
    records = []
    for a in client.list_aggs(
        ticker=ticker, multiplier=1, timespan="day",
        from_=from_, to=to, limit=50000,
    ):
        records.append((a.timestamp, a.close))
    if not records:
        raise RuntimeError(f"No data returned for {ticker}")
    ts, close = zip(*records)
    idx = pd.to_datetime(list(ts), unit="ms").tz_localize(None).normalize()
    s = pd.Series(list(close), index=idx, name=ticker, dtype="float64")
    return s.groupby(s.index).last().sort_index()


def load_prices(
    tickers: Iterable[str],
    start: str = "2007-06-01",
    end: str = "2025-05-31",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load a wide DataFrame of adjusted daily prices, cached to Parquet."""
    frames = []
    for t in tickers:
        p = _cache_path(t)
        s = None
        if use_cache and p.exists():
            try:
                s = pd.read_parquet(p)[t]
            except Exception:
                s = None
        if s is None:
            for attempt in range(3):
                try:
                    s = _fetch_one_massive(t, start, end)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
            s.to_frame().to_parquet(p)
        # trim to requested range
        s = s.loc[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
        frames.append(s)
    df = pd.concat(frames, axis=1).sort_index()
    return df


def portfolio_prices(name: str, **kwargs) -> pd.DataFrame:
    """Return a wide-format price panel for one of the named portfolios."""
    if name not in PORTFOLIOS:
        raise KeyError(f"Unknown portfolio: {name}")
    return load_prices(PORTFOLIOS[name], **kwargs)


def align_returns(prices: pd.DataFrame, kind: str = "log") -> pd.DataFrame:
    """Compute returns and drop rows where any series is missing.

    Missing-data policy: listwise deletion (documented assumption). For the
    Technology portfolio this drops all rows before META's IPO (May 2012).
    """
    from src.ewma import compute_returns

    rets = compute_returns(prices, kind=kind)
    return rets.dropna(how="any")
