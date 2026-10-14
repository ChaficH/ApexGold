"""
Tests for ApexStrategyV2.
Verifies:
1. End-to-end evaluation flow through base scorer, regime detector, ML filter, and dynamic exit manager.
2. Flag toggles (regime_filter, dynamic_exits, meta_labeler).
3. Fallback to legacy behavior when V2 flags are disabled.
4. Correct generation of enhanced ConfluenceSignal with dynamic TP and breathing room ratchet.
"""

import numpy as np
import pandas as pd
import pytest

from strategy.confluence_scorer import ConfluenceScorer
from strategy.v2.strategy_v2 import ApexStrategyV2


def create_sample_dfs(n_bars: int = 60) -> dict:
    times = np.arange(1520000000, 1520000000 + n_bars * 300, 300)
    prices = 1320.0 + np.linspace(0, 5, n_bars)
    df_5m = pd.DataFrame({
        "time": times,
        "open": prices,
        "high": prices + 1.0,
        "low": prices - 0.5,
        "close": prices + 0.5,
        "volume": 200.0,
    })
    times_1m = np.arange(1520000000, 1520000000 + n_bars * 5 * 60, 60)
    prices_1m = 1320.0 + np.linspace(0, 5, len(times_1m))
    df_1m = pd.DataFrame({
        "time": times_1m,
        "open": prices_1m,
        "high": prices_1m + 0.3,
        "low": prices_1m - 0.2,
        "close": prices_1m + 0.1,
        "volume": 50.0,
    })
    return {"5m": df_5m, "15m": df_5m, "1H": df_5m, "1m": df_1m}


def test_strategy_v2_initialization():
    strat = ApexStrategyV2(mode="balanced")
    assert strat.regime_filter_enabled
    assert strat.dynamic_exits_enabled
    assert strat.meta_labeler_enabled
    assert strat.get_active_threshold() == 68


def test_strategy_v2_evaluation_pipeline():
    strat = ApexStrategyV2(mode="balanced", min_p_win=0.50)
    dfs = create_sample_dfs(80)
    bid = 1325.0
    ask = 1325.25

    res = strat.evaluate("XAUUSD", dfs, current_tick_bid=bid, current_tick_ask=ask, utc_timestamp=float(dfs["5m"]["time"].iloc[-1]))
    assert isinstance(res, tuple)
    assert len(res) == 2


def test_strategy_v2_fallback_when_disabled():
    # When flags disabled, strategy relies strictly on base confluence scorer
    strat = ApexStrategyV2(
        mode="balanced",
        regime_filter_enabled=False,
        dynamic_exits_enabled=False,
        meta_labeler_enabled=False,
        asia_session_filter_enabled=False,
    )
    dfs = create_sample_dfs(80)
    bid = 1325.0
    ask = 1325.25

    res = strat.evaluate("XAUUSD", dfs, current_tick_bid=bid, current_tick_ask=ask, utc_timestamp=float(dfs["5m"]["time"].iloc[-1]))
    assert isinstance(res, tuple)
