"""
Automated Regression and Validation Tests for ApexGold Smart Pre-News Trajectory & Free-Roll Engine,
1H Structural Wall Check, Mac MT5 Bridge HTTP Fallback, and Runtime Safety Traps.
"""

import asyncio
import time
import pytest
import pandas as pd
import numpy as np

from brokers.base import AccountInfo, Bar, OrderResult, OrderType, Position, PositionType, SymbolInfo, Tick
from brokers.mt5_mac_socket import MacSocketMT5Adapter
from engine.indicators import TechnicalIndicators
from engine.news_reactor import CalendarEvent, NewsReactor
from execution.gold_risk import GoldRiskEngine
from execution.trade_manager import ManagedPosition, TradeManager
from strategy.confluence_scorer import ConfluenceScorer
from ui.server import app, ctx


def create_gold_symbol(
    contract_size: float = 100.0,
    point: float = 0.01,
    digits: int = 2,
    spread: float = 0.25,
    volume_min: float = 0.01,
    volume_step: float = 0.01,
    volume_max: float = 100.0,
    ask: float = 2650.25,
    bid: float = 2650.00,
) -> SymbolInfo:
    return SymbolInfo(
        symbol="XAUUSD",
        digits=digits,
        point=point,
        spread=spread,
        tick_size=point,
        tick_value=point * contract_size,
        contract_size=contract_size,
        volume_min=volume_min,
        volume_max=volume_max,
        volume_step=volume_step,
        bid=bid,
        ask=ask,
    )


def create_account(balance: float = 10.0, leverage: int = 500) -> AccountInfo:
    return AccountInfo(
        balance=balance,
        equity=balance,
        margin=0.0,
        free_margin=balance,
        leverage=leverage,
        currency="USD",
    )


def test_forecast_vs_previous_numeric_parsing():
    """Verify numeric parsing of units (%, K, M, B) and fundamental Gold bias logic."""
    reactor = NewsReactor(enabled=True)

    # 1. Parse unit stripping
    assert reactor._parse_economic_num("0.3%") == 0.3
    assert reactor._parse_economic_num("215K") == 215000.0
    assert reactor._parse_economic_num("1.35M") == 1350000.0
    assert reactor._parse_economic_num("2.5B") == 2500000000.0
    assert reactor._parse_economic_num("-") is None

    # 2. Standard USD metrics (CPI, NFP, Retail Sales)
    # Cooling CPI (Forecast 0.2% < Previous 0.3%) -> Dovish -> BULLISH_GOLD
    cpi_event = CalendarEvent(
        id="cpi_1",
        title="US Core CPI m/m",
        currency="USD",
        impact="HIGH",
        impact_badge="🔴 HIGH",
        timestamp_utc=time.time() + 1800,
        local_time_hm="03:30 PM",
        local_time_hms="03:30:00 PM",
        local_date_str="2026-10-15",
        forecast="0.2%",
        previous="0.3%",
    )
    bias, reason = reactor.determine_fundamental_forecast_bias(cpi_event)
    assert bias == "BULLISH_GOLD"
    assert "Cooling USD" in reason

    # Hotter NFP (Forecast 220K > Previous 180K) -> Hawkish -> BEARISH_GOLD
    nfp_event = CalendarEvent(
        id="nfp_1",
        title="US Non-Farm Employment Change",
        currency="USD",
        impact="HIGH",
        impact_badge="🔴 HIGH",
        timestamp_utc=time.time() + 1800,
        local_time_hm="03:30 PM",
        local_time_hms="03:30:00 PM",
        local_date_str="2026-10-15",
        forecast="220K",
        previous="180K",
    )
    bias_nfp, _ = reactor.determine_fundamental_forecast_bias(nfp_event)
    assert bias_nfp == "BEARISH_GOLD"

    # 3. Inverse USD metrics (Jobless Claims, Unemployment Rate)
    # Higher Claims (Forecast 225K > Previous 210K) -> Weakening Labor -> BULLISH_GOLD
    claims_event = CalendarEvent(
        id="claims_1",
        title="US Unemployment Claims",
        currency="USD",
        impact="HIGH",
        impact_badge="🔴 HIGH",
        timestamp_utc=time.time() + 1800,
        local_time_hm="03:30 PM",
        local_time_hms="03:30:00 PM",
        local_date_str="2026-10-15",
        forecast="225K",
        previous="210K",
    )
    bias_claims, reason_claims = reactor.determine_fundamental_forecast_bias(claims_event)
    assert bias_claims == "BULLISH_GOLD"
    assert "Weakening Labor" in reason_claims


def test_evaluate_pre_news_trajectory_scoring():
    """Verify pre-news trajectory combines Order-Flow, Volume/Expansion, and Fundamental agreement."""
    reactor = NewsReactor(enabled=True)
    reactor._is_calendar_available = True
    now = time.time()

    # Event in 25 minutes (Zone A)
    event = CalendarEvent(
        id="cpi_traj",
        title="US Core CPI m/m",
        currency="USD",
        impact="HIGH",
        impact_badge="🔴 HIGH",
        timestamp_utc=now + 1500,  # 25 minutes ahead
        local_time_hm="03:30 PM",
        local_time_hms="03:30:00 PM",
        local_date_str="2026-10-15",
        forecast="0.2%",
        previous="0.3%",  # Cooling CPI -> Bullish Gold
    )
    reactor._all_events = [event]

    # Generate synthetic bullish price action for 15m, 5m, 1m with expanding bodies
    prices = [2650.0 + (i * 0.5) for i in range(30)]
    df_bull = pd.DataFrame({
        "open": [p - 0.35 for p in prices],
        "high": [p + 0.10 for p in prices],
        "low": [p - 0.40 for p in prices],
        "close": prices,  # Close near high, body is 0.35/0.50 = 70% of range
        "volume": [100.0] * 30,
    })

    dfs = {"15m": df_bull, "5m": df_bull, "1m": df_bull}
    traj = reactor.evaluate_pre_news_trajectory(symbol="XAUUSD", dfs=dfs, tick_velocity=0.25, current_utc=now)

    assert traj["active"] is True
    assert traj["fundamental_bias"] == "BULLISH_GOLD"
    assert traj["drift_direction"] == "BUY"
    assert traj["trajectory_bias"] == "BUY"
    assert traj["confidence_pct"] >= 78
    assert traj["order_flow_pts"] == 40
    assert traj["volume_pts"] == 30
    assert traj["fundamental_pts"] == 30


def test_confluence_scorer_zone_a_pre_news_trajectory_entry():
    """Zone A (T-45m to T-90s): Confidence >= 75% allows automatic entry with +12pt trajectory boost."""
    scorer = ConfluenceScorer(mode="auto_adaptive")
    reactor = NewsReactor(enabled=True)
    reactor._is_calendar_available = True
    now = time.time()

    event = CalendarEvent(
        id="cpi_entry",
        title="US Core CPI m/m",
        currency="USD",
        impact="HIGH",
        impact_badge="🔴 HIGH",
        timestamp_utc=now + 1200,  # 20 minutes ahead
        local_time_hm="03:30 PM",
        local_time_hms="03:30:00 PM",
        local_date_str="2026-10-15",
        forecast="0.2%",
        previous="0.4%",  # Bullish Gold
    )
    reactor._all_events = [event]

    prices = [2650.0 + (i * 0.5) for i in range(30)]
    df_bull = pd.DataFrame({
        "open": [p - 0.35 for p in prices],
        "high": [p + 0.10 for p in prices],
        "low": [p - 0.40 for p in prices],
        "close": prices,
        "volume": [100.0] * 30,
    })

    # Evaluate auto brain
    state = scorer.evaluate_auto_brain(
        symbol="XAUUSD",
        df_1h=df_bull,
        df_15m=df_bull,
        df_5m=df_bull,
        df_1m=df_bull,
        news_reactor=reactor,
        tick_velocity=0.25,
        now_ts=now,
    )

    assert state["sub_regime"] == "PRE_NEWS_TRAJECTORY"
    assert state["is_pre_news_trajectory"] is True
    assert state["dial_pct"] == 78.0
    assert state["active_threshold"] == 63
    assert state["trajectory_bias"] == "BUY"

    # Evaluate signals
    sig_buy = scorer.evaluate(
        symbol="XAUUSD",
        df_1h=df_bull,
        df_15m=df_bull,
        df_5m=df_bull,
        df_1m=df_bull,
        tick_velocity=0.25,
        utc_timestamp=now,
    )
    assert sig_buy is not None
    assert sig_buy.direction == "BUY"
    assert sig_buy.is_pre_news_trajectory is True
    assert sig_buy.breakdown.get("pre_news_trajectory") == 12


def test_confluence_scorer_1h_structural_wall_check():
    """1H structural wall check requires distance to nearest opposing swing high/low >= 1.8 * ATR_15m."""
    scorer = ConfluenceScorer(mode="auto_adaptive")

    c_1h = [2600.0 + i for i in range(25)]
    h_1h = [p + 1.0 for p in c_1h]
    l_1h = [p - 1.0 for p in c_1h]
    # Place a major swing high ceiling directly ahead at 2626.0 (only 1.0 point above current close 2625.0)
    h_1h[15] = 2626.0

    df_1h = pd.DataFrame({"open": c_1h, "high": h_1h, "low": l_1h, "close": c_1h, "volume": [100.0] * 25})

    # 15m ATR = 2.0. Required room = 1.8 * 2.0 = 3.6 points.
    df_15m = pd.DataFrame({
        "open": [2620.0] * 20,
        "high": [2622.0] * 20,
        "low": [2620.0] * 20,
        "close": [2621.0] * 20,
        "volume": [100.0] * 20,
    })

    # Distance to ceiling is 2626.0 - 2624.0 = 2.0 < 3.6 points -> Blocked!
    state = scorer.evaluate_auto_brain(
        symbol="XAUUSD",
        df_1h=df_1h,
        df_15m=df_15m,
        df_5m=df_15m,
        df_1m=df_15m,
    )
    # Since opposing wall is too close (< 1.8 * ATR), sub_regime does NOT qualify for WAVE_RUNNER
    assert state["sub_regime"] != "WAVE_RUNNER"


@pytest.mark.anyio
async def test_trade_manager_pre_news_freeroll_lock():
    """When news is within 90s, winning trades >= +0.40R lock SL to +0.20R guaranteed profit & expand TP to 4.0R."""
    from brokers.paper_sim import PaperSimAdapter
    broker = PaperSimAdapter()
    await broker.connect()

    reactor = NewsReactor(enabled=True)
    reactor._is_calendar_available = True
    now = time.time()
    broker._last_ticks["XAUUSD"] = Tick(symbol="XAUUSD", bid=2651.50, ask=2651.75, time=now, spread=0.25)

    # News event in 70 seconds (T-70s)
    event = CalendarEvent(
        id="cpi_freeroll",
        title="US Core CPI m/m",
        currency="USD",
        impact="HIGH",
        impact_badge="🔴 HIGH",
        timestamp_utc=now + 70,
        local_time_hm="03:30 PM",
        local_time_hms="03:30:00 PM",
        local_date_str="2026-10-15",
        forecast="0.2%",
        previous="0.3%",
    )
    reactor._all_events = [event]

    tm = TradeManager(broker=broker, news_reactor=reactor)
    tm._running = True

    # Open position with +0.50R profit
    pos = ManagedPosition(
        ticket=101,
        symbol="XAUUSD",
        direction="BUY",
        volume=0.01,
        open_price=2650.00,
        sl=2647.50,  # 2.50 risk distance
        tp=2655.00,
        entry_time=now - 60,
        risk_r_dist=2.50,
        current_price=2651.50,  # +1.50 profit = +0.60R
        profit=15.0,
    )
    tm._managed_positions[101] = pos

    broker_pos = Position(
        ticket=101,
        symbol="XAUUSD",
        type=PositionType.BUY,
        volume=0.01,
        open_price=2650.00,
        current_price=2651.50,
        sl=2647.50,
        tp=2655.00,
        profit=15.0,
    )
    broker._positions[101] = broker_pos

    # Run one management iteration
    asyncio.create_task(tm._management_loop())
    await asyncio.sleep(0.6)
    tm._running = False

    # Verify Free-Roll Lock:
    # Guaranteed offset = +0.20R = 0.50. Entry 2650 + spread 0.25 + 0.50 = 2650.75
    # TP expanded to 4.0R = 2650 + 10.0 = 2660.00
    assert pos.is_freeroll_locked is True
    assert pos.sl >= 2650.70
    assert pos.tp >= 2659.90


@pytest.mark.anyio
async def test_trade_manager_pre_news_t60s_scratch():
    """When news is within 60s, stalled trades (< +0.40R) are scratched to prevent spread blowout."""
    from brokers.paper_sim import PaperSimAdapter
    broker = PaperSimAdapter()
    await broker.connect()

    reactor = NewsReactor(enabled=True)
    reactor._is_calendar_available = True
    now = time.time()
    broker._last_ticks["XAUUSD"] = Tick(symbol="XAUUSD", bid=2650.25, ask=2650.50, time=now, spread=0.25)

    # Event in 45 seconds (T-45s)
    event = CalendarEvent(
        id="cpi_scratch",
        title="US Core CPI m/m",
        currency="USD",
        impact="HIGH",
        impact_badge="🔴 HIGH",
        timestamp_utc=now + 45,
        local_time_hm="03:30 PM",
        local_time_hms="03:30:00 PM",
        local_date_str="2026-10-15",
        forecast="0.2%",
        previous="0.3%",
    )
    reactor._all_events = [event]

    tm = TradeManager(broker=broker, news_reactor=reactor)
    tm._running = True

    # Open position stalled at +0.10R (< +0.40R)
    pos = ManagedPosition(
        ticket=102,
        symbol="XAUUSD",
        direction="BUY",
        volume=0.01,
        open_price=2650.00,
        sl=2647.50,
        tp=2655.00,
        entry_time=now - 60,
        risk_r_dist=2.50,
        current_price=2650.25,  # only +0.10R
        profit=2.5,
    )
    tm._managed_positions[102] = pos

    broker_pos = Position(
        ticket=102,
        symbol="XAUUSD",
        type=PositionType.BUY,
        volume=0.01,
        open_price=2650.00,
        current_price=2650.25,
        sl=2647.50,
        tp=2655.00,
        profit=2.5,
    )
    broker._positions[102] = broker_pos

    asyncio.create_task(tm._management_loop())
    await asyncio.sleep(0.35)
    tm._running = False

    # Position should have triggered close
    assert (pos.is_closing is True or pos.is_closed is True)
    assert "Pre-News T-60s Scratch" in (pos.close_reason or "")


@pytest.mark.anyio
async def test_mac_socket_adapter_http_queue_fallback():
    """If port 15555 cannot connect, MacSocketMT5Adapter seamlessly falls back to HTTP queue mode."""
    adapter = MacSocketMT5Adapter(host="127.0.0.1", port=15599, timeout_sec=0.2)
    connected = await adapter.connect()

    assert connected is True
    assert adapter._use_http_queue is True
    assert adapter.is_connected() is True

    # Order execution queues into server ctx
    ctx.mt5_pending_actions = []
    res = await adapter.open_order(
        symbol="XAUUSD",
        order_type=OrderType.BUY,
        volume=0.01,
        price=2650.00,
        sl=2647.50,
        tp=2655.00,
    )
    assert res.success is True
    assert len(ctx.mt5_pending_actions) == 1
    assert ctx.mt5_pending_actions[0]["action"] == "ORDER_SEND"
    assert ctx.mt5_pending_actions[0]["volume"] == 0.01

    await adapter.disconnect()


@pytest.mark.anyio
async def test_action_resume_is_properly_awaited():
    """action_resume endpoint properly awaits async resume_trading without returning an unawaited coroutine."""
    from ui.server import action_resume
    from brokers.paper_sim import PaperSimAdapter
    broker = PaperSimAdapter()
    await broker.connect()

    ctx.trade_manager = TradeManager(broker=broker)
    ctx.trade_manager.pause_trading(minutes=60)
    assert ctx.trade_manager.is_paused() is True

    resp = await action_resume()
    assert resp["status"] == "OK"
    assert ctx.trade_manager.is_paused() is False


@pytest.mark.anyio
async def test_toggle_pre_news_trajectory_endpoint():
    """Test 1-click toggle of pre-news trajectory engine."""
    from ui.server import action_toggle_pre_news_trajectory
    ctx.allow_pre_news_trajectory = True

    resp1 = await action_toggle_pre_news_trajectory()
    assert resp1["allow_pre_news_trajectory"] is False

    resp2 = await action_toggle_pre_news_trajectory()
    assert resp2["allow_pre_news_trajectory"] is True


def test_gold_risk_small_account_micro_sl_allowance():
    """On a $10 account, min_sl allows 1.50 distance so micro-scalp safely fits risk budget."""
    risk_engine = GoldRiskEngine(
        account_type="micro",
        budget_usd=10.00,
        risk_per_trade_pct=6.0,
        risk_profile="aggressive",
    )
    sym = create_gold_symbol(contract_size=100.0)
    acc = create_account(balance=10.00)

    # With min_sl = 1.50, 0.01 lot monetary risk = $1.50 + $0.25 spread = $1.75 <= $2.80 budget!
    lot, details = risk_engine.calculate_lot_size(sym, acc, sl_distance=1.50)

    assert lot == 0.01
    assert details["has_margin"] is True
    assert details["actual_monetary_risk"] == 1.75
    assert details["actual_monetary_risk"] <= details["risk_amount"]
