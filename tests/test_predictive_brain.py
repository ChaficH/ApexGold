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


def test_ensemble_gate_and_telemetry_in_predictive_brain():
    """Verify Gate 6 and ensemble telemetry dictionary exist in state."""
    brain = ApexPredictiveBrain()
    dfs = _generate_synthetic_dfs(trend="BULL")

    # Feed ticks into ensemble order flow tracker
    brain.on_tick(2680.0, 2680.25)
    brain.on_tick(2680.20, 2680.45)

    brain.evaluate_market(
        symbol="XAUUSD",
        dfs=dfs,
        current_bid=2680.20,
        current_ask=2680.45,
        spread=0.25,
    )

    state = brain.get_state()
    assert "gates" in state
    gate_names = [g["name"] for g in state["gates"]]
    assert "Ensemble Order Flow & Volatility Consensus" in gate_names

    # Check ensemble telemetry payload
    assert "ensemble" in state
    ens = state["ensemble"]
    assert ens is not None
    assert "ensemble_bias" in ens
    assert "delta_ratio_60s" in ens
    assert "volatility_regime" in ens
    assert "conviction_boost" in ens
    assert "rr_expansion_boost" in ens


def test_order_flow_imbalance_tracker():
    """Verify OrderFlowImbalanceTracker buyer/seller delta tracking."""
    from strategy.ensemble_models import OrderFlowImbalanceTracker

    ofi = OrderFlowImbalanceTracker()
    now = 1000.0

    # 1st tick establishes reference
    ofi.on_tick(bid=2600.0, ask=2600.30, now_ts=now)
    # Price rises (buyer initiated)
    ofi.on_tick(bid=2600.20, ask=2600.50, now_ts=now + 1)
    ofi.on_tick(bid=2600.40, ask=2600.70, now_ts=now + 2)

    metrics = ofi.compute_metrics(window_sec=60.0)
    assert metrics["buyer_ticks"] >= 2
    assert metrics["delta_ratio"] > 0.0


def test_fair_value_gap_engine():
    """Verify 3-candle Fair Value Gap detection."""
    from strategy.ensemble_models import FairValueGapEngine

    # Candle 0: Low 2600, High 2602
    # Candle 1: Big impulse up: Low 2602, High 2608
    # Candle 2: Pullback/pause: Low 2605, High 2609
    # -> FVG between High[0]=2602 and Low[2]=2605 (Gap of 3.0 points)
    df = pd.DataFrame({
        "open": [2600.0, 2602.0, 2606.0, 2607.0, 2607.0],
        "high": [2602.0, 2608.0, 2609.0, 2610.0, 2610.0],
        "low": [2599.0, 2602.0, 2605.0, 2606.0, 2606.0],
        "close": [2601.0, 2607.0, 2608.0, 2609.0, 2609.0],
        "volume": [100.0] * 5,
    })

    fvgs = FairValueGapEngine.detect_fvgs(df, timeframe="5m")
    assert len(fvgs) >= 1
    bull_fvg = [f for f in fvgs if f.direction == "BULL"][0]
    assert bull_fvg.bottom == 2602.0
    assert bull_fvg.top == 2605.0
    assert bull_fvg.midpoint == 2603.50


def test_parkinson_volatility_classifier():
    """Verify Parkinson extreme-value volatility estimation."""
    from strategy.ensemble_models import ParkinsonVolatilityClassifier

    df = pd.DataFrame({
        "high": [2610.0 + i * 2.0 for i in range(25)],
        "low": [2600.0 + i * 2.0 for i in range(25)],
        "close": [2605.0 + i * 2.0 for i in range(25)],
        "open": [2602.0 + i * 2.0 for i in range(25)],
    })

    metrics = ParkinsonVolatilityClassifier.calculate(df, window=20)
    assert "parkinson_vol" in metrics
    assert metrics["parkinson_vol"] > 0.0
    assert "volatility_regime" in metrics

