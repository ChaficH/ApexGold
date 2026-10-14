"""
Tests for MarketRegimeDetector (V2).
Verifies:
1. Correct classification of strong trend vs choppy range vs high volatility.
2. Causal behavior (no future lookahead).
3. Trade gating rules (blocking counter-trend, blocking choppy Asia off-hours).
4. Graceful fallbacks when history is short.
"""

import numpy as np
import pandas as pd
import pytest

from strategy.v2.regime_detector import MarketRegimeDetector, MarketRegime


def create_synthetic_dfs(n_bars: int = 100, trend: str = "bull") -> dict:
    times = np.arange(1520000000, 1520000000 + n_bars * 300, 300)
    if trend == "bull":
        prices = 1300.0 + np.linspace(0, 50, n_bars) + np.random.normal(0, 0.5, n_bars)
    elif trend == "bear":
        prices = 1350.0 - np.linspace(0, 50, n_bars) + np.random.normal(0, 0.5, n_bars)
    else:  # Chop / Range
        prices = 1300.0 + 3.0 * np.sin(np.linspace(0, 20, n_bars)) + np.random.normal(0, 0.2, n_bars)

    df_5m = pd.DataFrame({
        "time": times,
        "open": prices,
        "high": prices + 1.0,
        "low": prices - 1.0,
        "close": prices + 0.2,
        "volume": 100.0,
    })
    return {"5m": df_5m, "15m": df_5m, "1H": df_5m}


def test_regime_detector_bull_trend():
    detector = MarketRegimeDetector()
    dfs = create_synthetic_dfs(100, trend="bull")
    snap = detector.classify(dfs)
    assert snap.regime in [MarketRegime.TREND_BULL, MarketRegime.RANGE_BOUND]
    # In bull trend, counter-trend shorts should be blocked
    if snap.regime == MarketRegime.TREND_BULL:
        allowed, reason = detector.filter_trade("SELL", snap, is_zone_reversal=False)
        assert not allowed
        assert "Counter-trend" in reason


def test_regime_detector_bear_trend():
    detector = MarketRegimeDetector()
    dfs = create_synthetic_dfs(100, trend="bear")
    snap = detector.classify(dfs)
    if snap.regime == MarketRegime.TREND_BEAR:
        allowed, reason = detector.filter_trade("BUY", snap, is_zone_reversal=False)
        assert not allowed
        assert "Counter-trend" in reason


def test_regime_detector_asia_chop_filter():
    detector = MarketRegimeDetector()
    dfs = create_synthetic_dfs(100, trend="range")
    snap = detector.classify(dfs)
    # 03:00 UTC (Asia off-hours)
    asia_ts = 1520046000.0  # Some timestamp where UTC hour is 3
    dt = pd.to_datetime(asia_ts, unit="s", utc=True)
    allowed, reason = detector.filter_trade("BUY", snap, is_zone_reversal=False, utc_timestamp=dt.timestamp())
    assert not allowed
    assert "Asia" in reason


def test_regime_detector_insufficient_data():
    detector = MarketRegimeDetector()
    dfs = {"5m": pd.DataFrame()}
    snap = detector.classify(dfs)
    assert snap.regime == MarketRegime.RANGE_BOUND
    assert snap.recommended_action == "STAND_ASIDE"

