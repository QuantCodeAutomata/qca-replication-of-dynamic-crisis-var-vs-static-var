"""Data-loading utilities.

Fetches daily adjusted prices via the Massive API (env var ``MASSIVE_API_KEY``)
with a local Parquet cache under ``data/``. Yahoo Finance fills missing history
when Massive does not cover the requested range (for example after a ticker
change). GOOGL, META, EOG and MPC are stored as Yahoo-adjusted series for the
full cache window and are not mixed with Massive.
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)


PORTFOLIOS: Dict[str, List[str]] = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "FDS", "NFLX", "ADBE", "CRM"],
    "Old Economy": ["XOM", "CVX", "JNJ", "PFE", "PG", "KO", "BA", "CAT", "VZ", "DUK"],
    "Finance":    ["JPM", "BAC", "WFC", "GS", "MS", "AXP", "USB", "BLK", "SCHW", "MET"],
    "Commodities":["SLB", "HAL", "EOG", "MPC", "NEM", "FCX", "BHP", "NUE", "MOS", "CF"],
}

# A cache cannot cover dates before a security existed. Treat a series that
# starts near its actual inception as complete rather than repeatedly trying
# Massive/Yahoo on every offline run. This is deliberately limited to the two
# paper constituents whose listing dates fall inside the requested 2007--2025
# range; ticker-change gaps (for example Massive META starting in 2021) still
# fail the coverage check and trigger the Yahoo supplement.
KNOWN_INCEPTION_DATES: Dict[str, str] = {
    "META": "2012-05-18",
    "MPC": "2011-06-24",
}

# These names are stored as Yahoo-adjusted closes for the full cache window
# (split-adjusted history; META includes the Facebook listing). Massive is
# not mixed in: GOOGL's Massive series starts only in 2014, and a splice
# onto Yahoo would insert a one-day return break.
YAHOO_ONLY_TICKERS = frozenset({"GOOGL", "META", "EOG", "MPC"})


# Crisis correlation peaks (paper Section 5) — the Brownian bridge *ends* here.
CRISIS_PEAKS: Dict[str, str] = {
    "GFC_2008":       "2008-09-15",  # Lehman collapse
    "Downgrade_2011": "2011-08-08",  # Black Monday 2011
    "COVID_2020":     "2020-03-16",  # peak EWMA correlation (paper table 5)
}

# Bridge-start / VaR as-of dates: 30 calendar days before each peak. This
# matches the paper's construction (fig. 3: COVID bridge runs 2020-02-14 ->
# 2020-03-17; section 5.5: "initialized at its historical February 2020
# correlation matrix"). The EWMA snapshot uses returns strictly before this
# date, so COVID's last observation is 2020-02-14, as in the paper.
CRISIS_STARTS: Dict[str, str] = {
    name: (pd.Timestamp(peak) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    for name, peak in CRISIS_PEAKS.items()
}


def _cache_path(ticker: str) -> Path:
    return DATA_DIR / f"{ticker}.parquet"


def _massive_api_key() -> str:
    """Resolve Massive API key from the environment, falling back to ``.env``."""
    api_key = os.environ.get("MASSIVE_API_KEY") or os.environ.get("MASSIVE_TOKEN")
    if api_key:
        return api_key
    try:
        from dotenv import dotenv_values
    except ImportError as e:
        raise RuntimeError(
            "MASSIVE_API_KEY / MASSIVE_TOKEN not in os.environ and python-dotenv "
            "is unavailable to read .env."
        ) from e
    values = dotenv_values(DATA_DIR.parent / ".env")
    api_key = values.get("MASSIVE_API_KEY") or values.get("MASSIVE_TOKEN")
    if not api_key:
        raise RuntimeError(
            "MASSIVE_API_KEY / MASSIVE_TOKEN not set in os.environ or .env."
        )
    return api_key.strip().strip('"').strip("'")


def _fetch_one_massive(ticker: str, from_: str, to: str) -> pd.Series:
    """Fetch a single ticker's daily closes from the Massive API."""
    from massive import RESTClient  # local import to keep tests light

    client = RESTClient(api_key=_massive_api_key())
    records = []
    for a in client.list_aggs(
        ticker=ticker, multiplier=1, timespan="day",
        from_=from_, to=to, adjusted=True, limit=50000,
    ):
        records.append((a.timestamp, a.close))
    if not records:
        raise RuntimeError(f"No data returned for {ticker}")
    ts, close = zip(*records)
    idx = pd.to_datetime(list(ts), unit="ms").tz_localize(None).normalize()
    s = pd.Series(list(close), index=idx, name=ticker, dtype="float64")
    return s.groupby(s.index).last().sort_index()


def _fetch_one_yahoo(ticker: str, from_: str, to: str) -> pd.Series:
    """Fetch one ticker's adjusted daily closes from Yahoo Finance."""
    import yfinance as yf  # local import to keep tests light

    # yfinance treats ``end`` as exclusive, while this module treats it as
    # inclusive.
    end_exclusive = (pd.Timestamp(to) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    data = yf.download(
        ticker,
        start=from_,
        end=end_exclusive,
        auto_adjust=True,
        progress=False,
        threads=False,
        timeout=60,
    )
    if data.empty or "Close" not in data:
        raise RuntimeError(f"No Yahoo data returned for {ticker}")

    close = data["Close"]
    # Recent yfinance versions return a one-column DataFrame for one ticker.
    if isinstance(close, pd.DataFrame):
        if ticker in close.columns:
            close = close[ticker]
        elif close.shape[1] == 1:
            close = close.iloc[:, 0]
        else:
            raise RuntimeError(f"Ambiguous Yahoo close columns for {ticker}")

    idx = pd.DatetimeIndex(pd.to_datetime(close.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    idx = idx.normalize()
    s = pd.Series(
        pd.to_numeric(close, errors="coerce").to_numpy(),
        index=idx,
        name=ticker,
        dtype="float64",
    ).dropna()
    if s.empty:
        raise RuntimeError(f"No valid Yahoo closes returned for {ticker}")
    return s.groupby(s.index).last().sort_index()


def _covers_requested_range(
    series: pd.Series,
    from_: str,
    to: str,
    ticker: str | None = None,
    tolerance_days: int = 7,
) -> bool:
    """Return whether a series reaches both requested range boundaries."""
    if series.empty:
        return False
    tolerance = pd.Timedelta(days=tolerance_days)
    required_start = pd.Timestamp(from_)
    if ticker in KNOWN_INCEPTION_DATES:
        required_start = max(
            required_start,
            pd.Timestamp(KNOWN_INCEPTION_DATES[ticker]),
        )
    return (
        series.index.min() <= required_start + tolerance
        and series.index.max() >= pd.Timestamp(to) - tolerance
    )


def _fetch_one_with_fallback(ticker: str, from_: str, to: str) -> pd.Series:
    """Fetch from Massive, supplementing insufficient history with Yahoo."""
    if ticker in YAHOO_ONLY_TICKERS:
        return _fetch_one_yahoo(ticker, from_, to)

    massive = None
    massive_error = None
    for attempt in range(3):
        try:
            massive = _fetch_one_massive(ticker, from_, to)
            break
        except Exception as exc:
            massive_error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)

    if massive is not None and _covers_requested_range(
        massive, from_, to, ticker=ticker
    ):
        return massive

    reason = (
        f"Massive returned only {massive.index.min().date()} to "
        f"{massive.index.max().date()}"
        if massive is not None
        else f"Massive failed: {massive_error}"
    )
    warnings.warn(
        f"{ticker}: {reason}; trying Yahoo Finance for missing history.",
        RuntimeWarning,
        stacklevel=2,
    )
    try:
        yahoo = _fetch_one_yahoo(ticker, from_, to)
    except Exception as yahoo_error:
        if massive is not None:
            warnings.warn(
                f"{ticker}: Yahoo fallback failed ({yahoo_error}); using the "
                "incomplete Massive series.",
                RuntimeWarning,
                stacklevel=2,
            )
            return massive
        raise RuntimeError(
            f"Both Massive and Yahoo failed for {ticker}: "
            f"Massive={massive_error}; Yahoo={yahoo_error}"
        ) from yahoo_error

    if massive is None:
        return yahoo
    # Keep Massive observations where available and use Yahoo only to fill dates
    # absent from Massive. This preserves the primary data source.
    return massive.combine_first(yahoo).sort_index().rename(ticker)


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
        if s is None or not _covers_requested_range(
            s, start, end, ticker=t
        ):
            fetched = _fetch_one_with_fallback(t, start, end)
            if s is not None and t not in YAHOO_ONLY_TICKERS:
                fetched = fetched.combine_first(s).sort_index().rename(t)
            s = fetched
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


def scenario_returns(
    prices: pd.DataFrame,
    scenario_date: str,
    kind: str = "log",
    min_history: int = 200,
) -> tuple[pd.DataFrame, List[str]]:
    """Build a no-look-ahead return panel for one historical scenario.

    Securities with fewer than ``min_history`` non-missing returns strictly
    before ``scenario_date`` are excluded; the remaining panel is listwise
    aligned. This shared helper keeps Experiment 1 on a consistent universe.
    """
    from src.ewma import compute_returns

    returns = compute_returns(prices, kind=kind)
    before = returns.index < pd.Timestamp(scenario_date)
    history_counts = returns.loc[before].notna().sum()
    eligible = history_counts[history_counts >= min_history].index.tolist()
    excluded = [ticker for ticker in prices.columns if ticker not in eligible]
    if not eligible:
        raise ValueError(
            f"No assets have {min_history} returns before {scenario_date}"
        )
    return returns.loc[before, eligible].dropna(how="any"), excluded


def align_returns(prices: pd.DataFrame, kind: str = "log") -> pd.DataFrame:
    """Compute returns and drop rows where any series is missing.

    Missing-data policy: listwise deletion. Yahoo can recover history hidden by
    ticker changes, but cannot create observations before a security's IPO.
    """
    from src.ewma import compute_returns

    rets = compute_returns(prices, kind=kind)
    return rets.dropna(how="any")
