"""
Automated Regression and Validation Tests for Strategy Intelligence:
- Dreiss Choppiness Index (CI) calculation and gating
- Market Structure (HH/HL/LH/LL, BOS, CHoCH)
- Liquidity Sweep & False Breakout Trap Detection
- Directional and Structural Anti-Revenge Filter
"""

import time
import pytest
import numpy as np
import pandas as pd

from engine.indicators import TechnicalIndicators
from strategy.confluence_scorer import ConfluenceScorer, ConfluenceSignal
from execution.trade_manager import TradeManager
from brokers.paper_sim import PaperSimAdapter
from engine.data_hub import DataHub
from engine.news_reactor import NewsReactor
from execution.gold_risk import GoldRiskEngine


def test_choppiness_index_detection():
    """Verify Choppiness Index correctly identifies high chop (CI > 60) and strong trend (CI < 40)."""
    # 1. Choppy data: rapid alternating consolidation range box
    osc = np.array([2350.0 + (1.0 if i % 2 == 0 else -1.0) for i in range(50)])
    high_chop = osc + 0.4
    low_chop = osc - 0.4
    close_chop = osc

    res_chop = TechnicalIndicators.calculate_choppiness_index(high_chop, low_chop, close_chop, period=14)
    assert res_chop["valid"] is True
    assert res_chop["ci"] >= 60.0
    assert res_chop["is_choppy"] is True

    # 2. Trending data: strong linear upward drift
    trend = np.linspace(2300.0, 2380.0, 50)
    high_trend = trend + 0.5
    low_trend = trend - 0.5
    close_trend = trend

    res_trend = TechnicalIndicators.calculate_choppiness_index(high_trend, low_trend, close_trend, period=14)
    assert res_trend["valid"] is True
    assert res_trend["ci"] <= 40.0
    assert res_trend["is_trending"] is True


def test_market_structure_hh_hl_and_bos():
    """Verify market structure identifies Higher Highs / Higher Lows and Bullish BOS."""
    # Create an upward zigzag series: 10, 20, 15, 25, 20, 30
    prices = [
        10.0, 12.0, 15.0, 18.0, 20.0, 17.0, 15.0, 16.0, 19.0, 22.0, 25.0,
        22.0, 20.0, 21.0, 24.0, 27.0, 30.0, 31.0
    ]
    h = np.array(prices) + 0.5
    l = np.array(prices) - 0.5
    c = np.array(prices)

    struct = TechnicalIndicators.detect_market_structure(h, l, c, window=2)
    assert struct["valid"] is True
    assert struct["is_bullish"] is True
    assert struct["structure"] == "BULLISH_HH_HL"
    assert struct["bos_bull"] is True


def test_market_structure_lh_ll_and_choch():
    """Verify market structure identifies Lower Highs / Lower Lows and Bearish BOS/CHoCH."""
    # Create a downward zigzag series: 30, 20, 25, 15, 20, 10
    prices = [
        30.0, 28.0, 25.0, 22.0, 20.0, 23.0, 25.0, 22.0, 19.0, 16.0, 15.0,
        18.0, 20.0, 17.0, 14.0, 11.0, 10.0, 9.0
    ]
    h = np.array(prices) + 0.5
    l = np.array(prices) - 0.5
    c = np.array(prices)

    struct = TechnicalIndicators.detect_market_structure(h, l, c, window=2)
    assert struct["valid"] is True
    assert struct["is_bearish"] is True
    assert struct["structure"] == "BEARISH_LH_LL"
    assert struct["bos_bear"] is True


def test_liquidity_trap_detector_bull_trap():
    """Verify Bull Trap detection when price spikes above resistance but rejects with long upper wick."""
    # Range of prior bars between 2345.00 and 2355.00
    rows = []
    base_p = 2350.0
    for i in range(15):
        rows.append({"open": base_p, "high": base_p + 2.0, "low": base_p - 2.0, "close": base_p + 0.5, "volume": 100})

    # Final candle: Spikes above range high (2352.0) to 2354.50, but closes at 2351.00 with large upper wick
    rows.append({
        "open": 2351.50,
        "high": 2354.50,  # +2.50 spike pierce
        "low": 2350.50,
        "close": 2351.00,  # Closes back below
        "volume": 250,
    })
    df_5m = pd.DataFrame(rows)

    traps = TechnicalIndicators.detect_liquidity_traps(df_5m, current_price=2351.00, is_gold=True)
    assert traps["valid"] is True
    assert traps["is_bull_trap"] is True
    assert traps["is_false_breakout"] is True
    assert "Bull Trap" in traps["trap_reason"]


def test_liquidity_trap_detector_bear_trap():
    """Verify Bear Trap detection when price spikes below support but rejects with long lower wick."""
    rows = []
    base_p = 2350.0
    for i in range(15):
        rows.append({"open": base_p, "high": base_p + 2.0, "low": base_p - 2.0, "close": base_p + 0.5, "volume": 100})

    # Final candle: Dips below range low (2348.0) to 2345.50, but springs back and closes at 2349.00 with large lower wick
    rows.append({
        "open": 2348.50,
        "high": 2349.50,
        "low": 2345.50,   # -2.50 drop pierce
        "close": 2349.00,  # Springs back above
        "volume": 250,
    })
    df_5m = pd.DataFrame(rows)

    traps = TechnicalIndicators.detect_liquidity_traps(df_5m, current_price=2349.00, is_gold=True)
    assert traps["valid"] is True
    assert traps["is_bear_trap"] is True
    assert traps["is_false_breakout"] is True
    assert "Bear Trap" in traps["trap_reason"]


@pytest.mark.anyio
async def test_anti_revenge_allows_opposite_direction_and_blocks_same_direction(tmp_path):
    """
    Verify intelligent anti-revenge:
    - Blocks entering the EXACT SAME losing trade in the same direction at the same price.
    - Freely permits OPPOSITE direction (trend reversal) after broker tick settlement!
    """
    db_path = str(tmp_path / "test_revenge.db")
    broker = PaperSimAdapter(initial_balance=200.0)
    await broker.connect()

    data_hub = DataHub(db_path=db_path)
    await data_hub.initialize()

    news_reactor = NewsReactor(enabled=False)
    risk_engine = GoldRiskEngine(budget_usd=200.0, risk_profile="balanced")

    tm = TradeManager(
        broker=broker,
        data_hub=data_hub,
        news_reactor=news_reactor,
        risk_engine=risk_engine,
        post_loss_buffer_seconds=0.1,    # Fast settlement for test
        anti_revenge_window_seconds=10.0, # 10s structural window
    )
    await tm.start()

    # Simulate a losing BUY trade at 2350.00
    now = time.time()
    tm._last_loss_time = now
    tm._last_loss_direction = "BUY"
    tm._last_loss_price = 2350.00
    tm._post_loss_cooldown_until = now - 1.0  # Buffer expired

    # 1. Candidate BUY trade at same level (2350.20) within anti_revenge_window
    buy_sig = ConfluenceSignal(
        symbol="XAUUSD",
        direction="BUY",
        score=75,
        tier="TIER_2_BALANCED",
        regime="ADAPTIVE_RUNNER",
        sub_regime="WAVE_RUNNER",
        target_rr=2.0,
        entry_price=2350.20,  # Near 2350.00
        suggested_sl=2347.50,
        suggested_tp=2355.00,
        risk_r_dist=2.70,
        be_lock_r=0.6,
        stall_exit_seconds=60.0,
        breakdown={},
        reason="Test",
        time=now,
    )

    can_buy, buy_reason = await tm.can_execute_signal(buy_sig)
    assert can_buy is False
    assert "Anti-Revenge Filter" in buy_reason
    assert "Blocked repeating losing BUY" in buy_reason

    # 2. Candidate SELL trade (OPPOSITE DIRECTION: Trend Reversal after fakeout!)
    sell_sig = ConfluenceSignal(
        symbol="XAUUSD",
        direction="SELL",
        score=78,
        tier="TIER_2_BALANCED",
        regime="ADAPTIVE_RUNNER",
        sub_regime="WAVE_RUNNER",
        target_rr=2.0,
        entry_price=2349.50,
        suggested_sl=2352.20,
        suggested_tp=2344.00,
        risk_r_dist=2.70,
        be_lock_r=0.6,
        stall_exit_seconds=60.0,
        breakdown={},
        reason="Test Reversal",
        time=now,
    )

    can_sell, sell_reason = await tm.can_execute_signal(sell_sig)
    assert can_sell is True
    assert sell_reason == "Execution approved"

    # 3. Candidate BUY trade at a structurally different price level (> $2.00 away on Gold)
    buy_higher_sig = ConfluenceSignal(
        symbol="XAUUSD",
        direction="BUY",
        score=75,
        tier="TIER_2_BALANCED",
        regime="ADAPTIVE_RUNNER",
        sub_regime="WAVE_RUNNER",
        target_rr=2.0,
        entry_price=2355.00,  # $5.00 away from 2350.00
        suggested_sl=2352.30,
        suggested_tp=2360.00,
        risk_r_dist=2.70,
        be_lock_r=0.6,
        stall_exit_seconds=60.0,
        breakdown={},
        reason="Test Level",
        time=now,
    )
    can_buy_high, high_reason = await tm.can_execute_signal(buy_higher_sig)
    assert can_buy_high is True
    assert high_reason == "Execution approved"

    # 4. Manual cooldown reset clears anti-revenge window
    tm.reset_cooldown()
    can_buy_after_reset, _ = await tm.can_execute_signal(buy_sig)
    assert can_buy_after_reset is True

    await tm.stop()
    await broker.disconnect()
