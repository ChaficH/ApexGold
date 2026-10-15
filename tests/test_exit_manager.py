"""
Tests for DynamicExitManager (V2).
Verifies:
1. Dynamic volatility-aware TP computation.
2. Breathing-room ratchet behavior (does not trigger at premature +0.65R noise).
3. Monotonic stop advancement (never loosens).
"""

import pytest
from strategy.v2.exit_manager import DynamicExitManager


def test_dynamic_tp_calculation():
    mgr = DynamicExitManager(base_target_r=1.50)
    # Entry: 1300, SL: 1290 (Risk = 10 points)
    plan_buy = mgr.compute_exit_plan(entry_price=1300.0, initial_sl=1290.0, direction="BUY", atr_5m=2.0)
    # Target R should be within [1.30, 1.85]
    assert 1.30 <= plan_buy.target_r <= 1.85
    assert plan_buy.suggested_tp > 1300.0
    assert plan_buy.suggested_sl == 1290.0

    plan_sell = mgr.compute_exit_plan(entry_price=1300.0, initial_sl=1310.0, direction="SELL", atr_5m=2.0)
    assert plan_sell.suggested_tp < 1300.0
    assert plan_sell.suggested_sl == 1310.0


def test_ratchet_breathing_room():
    mgr = DynamicExitManager(be_trigger_r=0.90, be_lock_r=0.35)
    entry_p = 1300.0
    sl_p = 1290.0
    risk = 10.0

    # At +0.65R (premature legacy trigger): ratchet should NOT trigger yet
    stage, new_sl = mgr.evaluate_ratchet(0.65, 0, entry_p, sl_p, risk, "BUY")
    assert stage == 0
    assert new_sl == sl_p

    # At +0.90R: ratchet triggers stage 1, locking +0.35R ($3.50 -> 1303.50)
    stage, new_sl = mgr.evaluate_ratchet(0.92, 0, entry_p, sl_p, risk, "BUY")
    assert stage == 1
    assert new_sl == 1303.50

    # Never loosens: if price drops to 0.70R, stage and stop remain locked
    stage2, new_sl2 = mgr.evaluate_ratchet(0.70, stage, entry_p, new_sl, risk, "BUY")
    assert stage2 == 1
    assert new_sl2 == 1303.50


import time
from unittest.mock import MagicMock, AsyncMock, patch
import pandas as pd
from brokers.base import AccountInfo
from execution.gold_risk import GoldRiskEngine
from execution.trade_manager import ManagedPosition, TradeManager
from strategy.predictive_brain import ApexPredictiveBrain


@pytest.mark.anyio
async def test_profitable_trade_harvests_on_5m_structural_breakdown():
    """
    Verify that when a trade is in profit (+0.65R) and 1m/5m trends break down below VWAP,
    it dynamically harvests profit at market instead of waiting passively on SL/TP.
    """
    now = time.time()
    pos = ManagedPosition(
        ticket=99101,
        symbol="XAUUSD",
        direction="BUY",
        volume=0.01,
        open_price=2700.00,
        sl=2697.50,
        tp=2710.00,
        entry_time=now - 120.0,
        risk_r_dist=2.50,
        highest_r_reached=0.75,
        profit=1.62,
    )

    bear_prices = [2703.0 - i * 0.20 for i in range(30)]
    df_bear = pd.DataFrame({
        "open": [p + 0.1 for p in bear_prices],
        "high": [p + 0.25 for p in bear_prices],
        "low": [p - 0.25 for p in bear_prices],
        "close": [p - 0.1 for p in bear_prices],
        "volume": [100.0] * 30,
    })

    mock_hub = MagicMock()
    mock_hub.get_dataframe = AsyncMock(return_value=df_bear)

    mock_broker = MagicMock()
    mock_broker.get_account_info = AsyncMock(return_value=AccountInfo(balance=100.0, equity=101.62, margin=0.0, free_margin=101.62, currency="USD", leverage=1000))

    tm = TradeManager(
        broker=mock_broker,
        data_hub=mock_hub,
        news_reactor=MagicMock(_latest_velocity=1.0),
        risk_engine=MagicMock(base_budget_usd=100.0),
    )

    should_exit, reason = await tm._check_market_flip(
        pos=pos,
        floating_r=0.65,
        current_price=2701.62,
        now=now,
    )

    assert should_exit is True
    assert "Dynamic Profit Harvest" in reason
    assert pos.exit_evaluation == "HARVEST_PROFIT_NOW"
    assert "Structural Breakdown" in pos.exit_verdict_detail


@pytest.mark.anyio
async def test_profitable_trade_harvests_on_exhaustion_rejection_wick():
    """
    Verify that when floating profit is +0.65R and a huge rejection wick (>= 50%) appears
    with StochRSI overbought (>= 78) and red candle, profit is harvested immediately.
    """
    now = time.time()
    pos = ManagedPosition(
        ticket=99102,
        symbol="XAUUSD",
        direction="BUY",
        volume=0.01,
        open_price=2700.00,
        sl=2697.50,
        tp=2710.00,
        entry_time=now - 120.0,
        risk_r_dist=2.50,
        highest_r_reached=0.75,
        profit=1.70,
    )

    prices = [2701.0] * 29
    df = pd.DataFrame({
        "open": [p for p in prices] + [2701.80],
        "high": [p + 0.20 for p in prices] + [2703.50],
        "low": [p - 0.20 for p in prices] + [2701.20],
        "close": [p for p in prices] + [2701.60],
        "volume": [100.0] * 30,
    })

    mock_hub = MagicMock()
    mock_hub.get_dataframe = AsyncMock(return_value=df)

    mock_broker = MagicMock()
    mock_broker.get_account_info = AsyncMock(return_value=AccountInfo(balance=100.0, equity=101.70, margin=0.0, free_margin=101.70, currency="USD", leverage=1000))

    tm = TradeManager(
        broker=mock_broker,
        data_hub=mock_hub,
        news_reactor=MagicMock(_latest_velocity=1.0),
        risk_engine=MagicMock(base_budget_usd=100.0),
    )

    with patch("engine.indicators.TechnicalIndicators.calculate_stoch_rsi", return_value={"k": 85.0, "d": 82.0, "valid": True}):
        should_exit, reason = await tm._check_market_flip(
            pos=pos,
            floating_r=0.65,
            current_price=2701.62,
            now=now,
        )

    assert should_exit is True
    assert "Dynamic Profit Harvest" in reason
    assert pos.exit_evaluation == "HARVEST_PROFIT_NOW"
    assert "Liquidity Rejection" in pos.exit_verdict_detail


@pytest.mark.anyio
async def test_profitable_trade_harvests_on_high_water_mark_giveback():
    """
    Verify that when a runner hits +1.10R peak and gives back >= 35% of peak gains
    with opposing 1m candle, profit is harvested immediately to protect gains.
    """
    now = time.time()
    pos = ManagedPosition(
        ticket=99103,
        symbol="XAUUSD",
        direction="BUY",
        volume=0.01,
        open_price=2700.00,
        sl=2697.50,
        tp=2710.00,
        entry_time=now - 120.0,
        risk_r_dist=2.50,
        highest_r_reached=1.10,
        peak_r_giveback_pct=40.9,
        profit=1.62,
    )

    df_1m = pd.DataFrame({
        "open": [2701.80] * 15,
        "high": [2702.00] * 15,
        "low": [2701.40] * 15,
        "close": [2701.50] * 15,
        "volume": [100.0] * 15,
    })

    mock_hub = MagicMock()
    mock_hub.get_dataframe = AsyncMock(return_value=df_1m)

    mock_broker = MagicMock()
    mock_broker.get_account_info = AsyncMock(return_value=AccountInfo(balance=100.0, equity=101.62, margin=0.0, free_margin=101.62, currency="USD", leverage=1000))

    tm = TradeManager(
        broker=mock_broker,
        data_hub=mock_hub,
        news_reactor=MagicMock(_latest_velocity=1.0),
        risk_engine=MagicMock(base_budget_usd=100.0),
    )

    should_exit, reason = await tm._check_market_flip(
        pos=pos,
        floating_r=0.65,
        current_price=2701.62,
        now=now,
    )

    assert should_exit is True
    assert "Peak Profit Protection" in reason
    assert pos.exit_evaluation == "HARVEST_PROFIT_NOW"


@pytest.mark.anyio
async def test_profitable_trade_holds_and_expands_when_healthy():
    """
    Verify that when trend is healthy, no exhaustion wicks exist, and giveback is low,
    the bot evaluates HOLD_AND_EXPAND and lets the runner run.
    """
    now = time.time()
    pos = ManagedPosition(
        ticket=99104,
        symbol="XAUUSD",
        direction="BUY",
        volume=0.01,
        open_price=2700.00,
        sl=2697.50,
        tp=2710.00,
        entry_time=now - 120.0,
        risk_r_dist=2.50,
        highest_r_reached=0.85,
        peak_r_giveback_pct=5.8,
        profit=2.00,
    )

    bull_prices = [2700.0 + i * 0.3 for i in range(30)]
    df_bull = pd.DataFrame({
        "open": [p - 0.1 for p in bull_prices],
        "high": [p + 0.3 for p in bull_prices],
        "low": [p - 0.3 for p in bull_prices],
        "close": [p + 0.1 for p in bull_prices],
        "volume": [100.0] * 30,
    })

    mock_hub = MagicMock()
    mock_hub.get_dataframe = AsyncMock(return_value=df_bull)

    mock_broker = MagicMock()
    mock_broker.get_account_info = AsyncMock(return_value=AccountInfo(balance=100.0, equity=102.00, margin=0.0, free_margin=102.00, currency="USD", leverage=1000))

    tm = TradeManager(
        broker=mock_broker,
        data_hub=mock_hub,
        news_reactor=MagicMock(_latest_velocity=1.0),
        risk_engine=MagicMock(base_budget_usd=100.0),
    )

    should_exit, reason = await tm._check_market_flip(
        pos=pos,
        floating_r=0.80,
        current_price=2702.00,
        now=now,
    )

    assert should_exit is False
    assert reason == ""
    assert pos.exit_evaluation == "HOLD_AND_EXPAND"
    assert "Healthy Runner" in pos.exit_verdict_detail


def test_predictive_brain_narrative_includes_active_position_study():
    """
    Verify that predictive_brain renders the real-time active position health study
    into the narrative thinking stream for user transparency.
    """
    pb = ApexPredictiveBrain()
    now = time.time()
    prices = [2700.0 + i * 0.1 for i in range(30)]
    dfs = {
        "1H": pd.DataFrame({"open": prices, "high": [p + 1.0 for p in prices], "low": [p - 1.0 for p in prices], "close": prices, "volume": [100.0]*30}),
        "15m": pd.DataFrame({"open": prices, "high": [p + 0.5 for p in prices], "low": [p - 0.5 for p in prices], "close": prices, "volume": [100.0]*30}),
        "5m": pd.DataFrame({"open": prices, "high": [p + 0.3 for p in prices], "low": [p - 0.3 for p in prices], "close": prices, "volume": [100.0]*30}),
        "1m": pd.DataFrame({"open": prices, "high": [p + 0.1 for p in prices], "low": [p - 0.1 for p in prices], "close": prices, "volume": [100.0]*30}),
    }

    test_positions = [
        {
            "ticket": 993874683,
            "direction": "BUY",
            "volume": 0.01,
            "open_price": 4151.65,
            "r_multiple": 0.65,
            "highest_r_reached": 0.70,
            "peak_r_giveback_pct": 7.1,
            "profit": 1.65,
            "exit_evaluation": "HOLD_AND_EXPAND",
            "exit_verdict_detail": "Market structure intact. Holding for expansion targets.",
        }
    ]

    res = pb.evaluate_market(
        symbol="XAUUSD",
        dfs=dfs,
        current_bid=2703.0,
        current_ask=2703.25,
        spread=0.25,
        now_ts=now,
        active_positions=test_positions,
    )

    narrative = res.get("narrative", "")
    assert "REAL-TIME ACTIVE POSITION HEALTH & EXIT STUDY" in narrative
    assert "#993874683 BUY 0.01 LOTS @ 4151.65" in narrative
    assert "Floating: +0.65R (+$1.65)" in narrative
    assert "High-Water Mark: +0.70R (Giveback: 7.1%)" in narrative
    assert "[HOLD_AND_EXPAND]" in narrative

