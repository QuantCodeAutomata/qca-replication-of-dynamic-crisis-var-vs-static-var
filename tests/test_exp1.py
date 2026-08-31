"""Tests for scenario-specific historical portfolio composition."""
from __future__ import annotations

import numpy as np
import pandas as pd

from exp.exp_1_dynamic_vs_static_var import T, _returns_for_scenario
from src.data import CRISIS_PEAKS, CRISIS_STARTS


def test_scenario_universe_excludes_assets_without_200_prior_returns():
    index = pd.bdate_range("2007-01-01", periods=300)
    prices = pd.DataFrame(
        {
            "EXISTING": 100 * np.exp(np.arange(300) * 0.001),
            "LATE_IPO": np.r_[
                np.full(240, np.nan),
                50 * np.exp(np.arange(60) * 0.001),
            ],
        },
        index=index,
    )

    returns, excluded = _returns_for_scenario(
        prices, index[-1].strftime("%Y-%m-%d")
    )

    assert list(returns.columns) == ["EXISTING"]
    assert excluded == ["LATE_IPO"]
    assert returns.index.max() < index[-1]


def test_crisis_asof_is_peak_minus_30_calendar_days_and_horizon_is_21():
    assert T == 21
    for crisis, peak in CRISIS_PEAKS.items():
        expected = pd.Timestamp(peak) - pd.Timedelta(days=30)
        assert pd.Timestamp(CRISIS_STARTS[crisis]) == expected
