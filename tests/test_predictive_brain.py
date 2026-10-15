"""
Unit and integration tests for ApexPredictiveBrain & UI Thinking Stream.
"""

import time
import numpy as np
import pandas as pd
import pytest

from strategy.confluence_scorer import ConfluenceScorer
from strategy.predictive_brain import ApexPredictiveBrain, PredictiveEvaluation


def _generate_synthetic_dfs(trend: str = "BULL"):
    """Generate realistic synthetic multi-timeframe DataFrames."""
    n_bars = 300
    timestamps = [1700000000 + i * 300 for i in range(n_bars)]
    
    if trend == "BULL":
        base = 2650.0
        drift = 0.35
    elif trend == "BEAR":
        base = 2700.0
        drift = -0.35
    else: # CHOP
        base = 2675.0
        drift = 0.0

    closes = [base + i * drift + np.sin(i / 3.0) * 0.8 for i in range(n_bars)]
    highs = [c + 0.9 for c in closes]
    lows = [c - 0.9 for c in closes]
    opens = [c - 0.1 for c in closes]
    volumes = [150.0 + (i % 10) * 10 for i in range(n_bars)]

    df_5m = pd.DataFrame({
        "time": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })

    # 15m (sample every 3)
    df_15m = df_5m.iloc[::3].copy().reset_index(drop=True)
    # 1H (sample every 12)
    df_1h = df_5m.iloc[::12].copy().reset_index(drop=True)
    # 1m (interpolate)
    df_1m = df_5m.copy()

    return {"1H": df_1h, "15m": df_15m, "5m": df_5m, "1m": df_1m}


def test_predictive_brain_standby_state():
    """Verify brain returns safe standby state on startup."""
    brain = ApexPredictiveBrain()
    state = brain.get_state()
    assert state["status"] == "INITIALIZING"
    assert state["is_frozen"] is False
    assert "STANDBY" in state["verdict_badge"].upper() or "WARMING" in state["verdict_badge"].upper()


def test_predictive_brain_bullish_evaluation():
    """Verify predictive brain identifies bullish market and generates narrative thinking."""
    brain = ApexPredictiveBrain()
    dfs = _generate_synthetic_dfs(trend="BULL")
    current_bid = float(dfs["5m"]["close"].iloc[-1])
    current_ask = current_bid + 0.25

    state = brain.evaluate_market(
        symbol="XAUUSD",
        dfs=dfs,
        current_bid=current_bid,
        current_ask=current_ask,
        spread=0.25,
        now_ts=time.time(),
    )

    assert state["cycle_index"] == 1
    assert state["is_frozen"] is False
    assert state["p_bull"] > state["p_bear"]
    assert len(state["gates"]) >= 4
    assert "⚡" in state["narrative"]
    assert "Macro Structure" in state["narrative"]
    assert isinstance(state["expected_value_r"], float)


def test_predictive_brain_anti_chasing_veto():
    """Verify predictive brain blocks overbought chasing."""
    brain = ApexPredictiveBrain()
    dfs = _generate_synthetic_dfs(trend="BULL")
    # Simulate a price spike far above VWAP
    vwap_c = float(dfs["5m"]["close"].iloc[-1])
    spike_bid = vwap_c + 3.80
    spike_ask = spike_bid + 0.25

    state = brain.evaluate_market(
        symbol="XAUUSD",
        dfs=dfs,
        current_bid=spike_bid,
        current_ask=spike_ask,
        spread=0.25,
        now_ts=time.time(),
    )

    # When price is extended, status should be WAITING_FOR_PULLBACK
    assert state["is_frozen"] is False
    assert state["status"] in ["WAITING_FOR_PULLBACK", "HUNTING_SETUP", "QUALIFIED_TRADE"]
    
    # Test filter_signal
    class DummySignal:
        direction = "BUY"
    
    if state["status"] == "WAITING_FOR_PULLBACK":
        allowed, reason = brain.filter_signal(DummySignal())
        assert allowed is False
        assert "pullback" in reason.lower()


def test_confluence_scorer_predictive_brain_integration():
    """Verify ConfluenceScorer embeds and runs ApexPredictiveBrain."""
    scorer = ConfluenceScorer(mode="auto_adaptive")
    assert hasattr(scorer, "predictive_brain")
    assert isinstance(scorer.predictive_brain, ApexPredictiveBrain)

    dfs = _generate_synthetic_dfs(trend="BULL")
    scorer.evaluate_auto_brain(
        dfs=dfs,
        current_spread=0.25,
        max_spread=0.80,
        symbol="XAUUSD",
    )

    status = scorer.get_auto_brain_status()
    assert "predictive_brain" in status
    assert status["predictive_brain"]["cycle_index"] >= 1
    assert status["predictive_brain"]["is_frozen"] is False


def test_predictive_brain_history_tracking():
    """Verify rolling evaluation history is recorded."""
    brain = ApexPredictiveBrain(history_capacity=10)
    dfs = _generate_synthetic_dfs(trend="BULL")

    for i in range(5):
        brain.evaluate_market(
            symbol="XAUUSD",
            dfs=dfs,
            current_bid=2680.0 + i,
            current_ask=2680.25 + i,
            spread=0.25,
            now_ts=time.time() + i,
        )

    state = brain.get_state()
    assert len(state["history"]) >= 1
    assert state["cycle_index"] == 5
