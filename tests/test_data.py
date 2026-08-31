"""Unit tests for primary/fallback market-data loading."""
from __future__ import annotations

import pandas as pd
import pytest

from src import data


def _prices(ticker: str, start: str, periods: int) -> pd.Series:
    index = pd.bdate_range(start, periods=periods)
    return pd.Series(range(periods), index=index, name=ticker, dtype="float64")


def test_yahoo_only_tickers_skip_massive(monkeypatch):
    yahoo = _prices("EOG", "2024-01-02", 22)
    monkeypatch.setattr(data, "_fetch_one_yahoo", lambda *args: yahoo)

    def unexpected_massive(*args):
        raise AssertionError("Yahoo-only tickers must not call Massive")

    monkeypatch.setattr(data, "_fetch_one_massive", unexpected_massive)

    result = data._fetch_one_with_fallback("EOG", "2024-01-01", "2024-01-31")

    pd.testing.assert_series_equal(result, yahoo)


def test_massive_is_used_when_it_covers_requested_range(monkeypatch):
    massive = _prices("TEST", "2024-01-02", 22)
    monkeypatch.setattr(data, "_fetch_one_massive", lambda *args: massive)

    def unexpected_yahoo(*args):
        raise AssertionError("Yahoo should not be called")

    monkeypatch.setattr(data, "_fetch_one_yahoo", unexpected_yahoo)

    result = data._fetch_one_with_fallback("TEST", "2024-01-01", "2024-01-31")

    pd.testing.assert_series_equal(result, massive)


def test_yahoo_fills_history_missing_from_massive(monkeypatch):
    massive = _prices("TEST", "2024-01-15", 13)
    yahoo = _prices("TEST", "2024-01-02", 22)
    monkeypatch.setattr(data, "_fetch_one_massive", lambda *args: massive)
    monkeypatch.setattr(data, "_fetch_one_yahoo", lambda *args: yahoo)

    with pytest.warns(RuntimeWarning, match="trying Yahoo Finance"):
        result = data._fetch_one_with_fallback(
            "TEST", "2024-01-01", "2024-01-31"
        )

    assert result.index.min() == pd.Timestamp("2024-01-02")
    assert result.index.max() == pd.Timestamp("2024-01-31")
    pd.testing.assert_series_equal(result.loc[massive.index], massive)


def test_yahoo_is_used_when_massive_fails(monkeypatch):
    yahoo = _prices("TEST", "2024-01-02", 22)

    def failed_massive(*args):
        raise RuntimeError("Massive unavailable")

    monkeypatch.setattr(data, "_fetch_one_massive", failed_massive)
    monkeypatch.setattr(data, "_fetch_one_yahoo", lambda *args: yahoo)
    monkeypatch.setattr(data.time, "sleep", lambda *_: None)

    with pytest.warns(RuntimeWarning, match="Massive failed"):
        result = data._fetch_one_with_fallback(
            "TEST", "2024-01-01", "2024-01-31"
        )

    pd.testing.assert_series_equal(result, yahoo)


def test_incomplete_cache_is_refreshed_with_fallback(monkeypatch, tmp_path):
    cached = _prices("TEST", "2024-01-15", 13)
    complete = _prices("TEST", "2024-01-02", 22)
    cache_path = tmp_path / "TEST.parquet"
    cached.to_frame().to_parquet(cache_path)

    monkeypatch.setattr(data, "_cache_path", lambda ticker: cache_path)
    monkeypatch.setattr(
        data, "_fetch_one_with_fallback", lambda *args: complete
    )

    result = data.load_prices(
        ["TEST"], start="2024-01-01", end="2024-01-31"
    )

    assert result.index.min() == pd.Timestamp("2024-01-02")
    assert result.index.max() == pd.Timestamp("2024-01-31")


def test_known_ipo_cache_is_complete_without_network(monkeypatch, tmp_path):
    index = pd.bdate_range("2012-05-18", "2025-05-30")
    cached = pd.Series(range(len(index)), index=index, name="META", dtype="float64")
    cache_path = tmp_path / "META.parquet"
    cached.to_frame().to_parquet(cache_path)

    monkeypatch.setattr(data, "_cache_path", lambda ticker: cache_path)

    def unexpected_fetch(*args):
        raise AssertionError("A complete post-IPO cache should be used offline")

    monkeypatch.setattr(data, "_fetch_one_with_fallback", unexpected_fetch)

    result = data.load_prices(
        ["META"], start="2007-06-01", end="2025-05-31"
    )

    assert result.index.min() == pd.Timestamp("2012-05-18")


def test_ticker_change_gap_does_not_masquerade_as_ipo_coverage():
    massive_only_after_rename = _prices("META", "2021-06-30", 1000)

    assert not data._covers_requested_range(
        massive_only_after_rename,
        "2007-06-01",
        "2025-05-31",
        ticker="META",
    )
