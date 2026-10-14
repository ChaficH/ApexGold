"""
Phase 5 Regression & Unit Tests for ApexGold API & Dashboard Safety.

Covers:
- Item 16: Secure MT5 bridge authentication and truthful system status reporting.
- Item 17: Sanitize UI commands and manual controls:
  * In-memory debouncing and rate limiting on execution endpoints.
  * Circuit breaker and risk guard gating on manual test trades.
  * Broker volume bounds and monetary risk budget enforcement on manual test trades.
  * Manual position protection against unauthorized modification.
  * Protective stop ratchet rule and broker stop-level distance constraints.
"""

import asyncio
import os
import sys
import time
from typing import Any, Dict, List, Optional
import pytest
from fastapi import HTTPException
from pydantic import BaseModel

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from brokers.base import (
    AbstractBrokerAdapter,
    AccountInfo,
    Bar,
    ClosedDeal,
    LimitOrder,
    OrderResult,
    OrderType,
    Position,
    PositionType,
    SymbolInfo,
    Tick,
)
from brokers.paper_sim import PaperSimAdapter
from execution.gold_risk import GoldRiskEngine
from execution.trade_manager import ManagedPosition, TradeManager
from ui.server import (
    ModifyPositionPayload,
    action_close_all,
    action_modify_position,
    action_test_trade,
    ctx,
    endpoint_rate_limiter,
    get_status,
    verify_bridge_token,
)


class MockBrokerAdapter(AbstractBrokerAdapter):
    """Mock broker adapter for server endpoint testing."""

    def __init__(self, initial_balance: float = 1000.0, is_paper: bool = True):
        self.balance = initial_balance
        self.equity = initial_balance
        self._positions: Dict[int, Position] = {}
        self._pending_orders: Dict[int, LimitOrder] = {}
        self._last_ticks: Dict[str, Tick] = {}
        self.is_paper = is_paper
        self.modify_calls: List[Dict[str, Any]] = []
        self._connected = True

    async def connect(self) -> bool:
        self._connected = True
        return True

    async def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    async def get_account_info(self) -> AccountInfo:
        return AccountInfo(
            balance=self.balance,
            equity=self.equity,
            margin=0.0,
            free_margin=self.balance,
            leverage=500,
            currency="USD",
        )

    async def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        return SymbolInfo(
            symbol=symbol,
            point=0.01,
            digits=2,
            spread=15,
            stops_level=20,  # 20 points = 0.20
            contract_size=100.0,
            volume_min=0.01,
            volume_max=10.0,
            volume_step=0.01,
            bid=2650.0,
            ask=2650.20,
        )

    async def get_tick(self, symbol: str) -> Optional[Tick]:
        return Tick(
            symbol=symbol,
            bid=2650.0,
            ask=2650.20,
            time=time.time(),
            spread=0.20,
        )

    async def get_rates(self, symbol: str, timeframe: str, count: int = 100) -> List[Bar]:
        return []

    async def open_order(
        self,
        symbol: str,
        order_type: OrderType,
        volume: float,
        price: float,
        sl: float,
        tp: float,
        comment: str = "",
    ) -> OrderResult:
        ticket = 99001
        pos = Position(
            ticket=ticket,
            symbol=symbol,
            type=PositionType.BUY if order_type in (OrderType.BUY, OrderType.BUY_LIMIT) else PositionType.SELL,
            volume=volume,
            open_price=price,
            sl=sl,
            tp=tp,
            open_time=time.time(),
            current_price=price,
        )
        self._positions[ticket] = pos
        return OrderResult(
            success=True,
            ticket=ticket,
            price=price,
            volume=volume,
        )

    async def modify_position(self, ticket: int, sl: float, tp: float) -> bool:
        self.modify_calls.append({"ticket": ticket, "sl": sl, "tp": tp})
        if ticket in self._positions:
            self._positions[ticket].sl = sl
            self._positions[ticket].tp = tp
        return True

    async def close_position(self, ticket: int, volume: Optional[float] = None) -> bool:
        self._positions.pop(ticket, None)
        return True

    async def cancel_order(self, ticket: int) -> bool:
        self._pending_orders.pop(ticket, None)
        return True

    async def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        return list(self._positions.values())

    async def close_all_positions(self, symbol: Optional[str] = None) -> int:
        c = len(self._positions)
        self._positions.clear()
        return c

    async def get_pending_orders(self, symbol: Optional[str] = None) -> List[LimitOrder]:
        return list(self._pending_orders.values())

    async def get_closed_deal(self, ticket: Any) -> Optional[ClosedDeal]:
        return None


class DummyDataHub:
    def get_latest_bars(self, symbol: str, timeframe: str, count: int = 50):
        return []

    def get_completed_bars(self, symbol: str, timeframe: str, count: int = 50):
        return []


class DummyNewsReactor:
    def is_in_high_impact_freeze(self, *args, **kwargs):
        return False, None

    def can_open_baseline_trade(self, symbol: str = "XAUUSD"):
        return True, ""

    def check_calendar_state(self):
        return {"status": "ACTIVE"}

    def get_complete_news_state(self):
        return {"calendar_status": "ACTIVE", "events": []}

    def get_feed_logs(self, limit: int = 15):
        return []


class DummyRequest:
    """Mock FastAPI Request object with headers, client, and query_params."""

    def __init__(self, headers: Optional[Dict[str, str]] = None, query_params: Optional[Dict[str, str]] = None):
        self.headers = headers or {}
        self.query_params = query_params or {}
        self.client = type("Client", (), {"host": "127.0.0.1"})()


# ==============================================================================
# 1. Bridge Authentication Tests
# ==============================================================================
def test_bridge_auth_rejects_missing_or_invalid_token():
    """Item 16: Incoming MT5 bridge requests are rejected when token is configured but missing/invalid."""
    ctx.bridge_token = "apex_secret_token_123"

    # Missing token in payload and headers
    with pytest.raises(HTTPException) as excinfo:
        verify_bridge_token(request=DummyRequest(), payload={"symbol": "XAUUSD"})
    assert excinfo.value.status_code == 401
    assert "Unauthorized" in excinfo.value.detail

    # Wrong token in payload
    with pytest.raises(HTTPException) as excinfo:
        verify_bridge_token(request=DummyRequest(), payload={"symbol": "XAUUSD", "token": "wrong_token"})
    assert excinfo.value.status_code == 401

    # Wrong token in header
    with pytest.raises(HTTPException) as excinfo:
        req = DummyRequest(headers={"X-Bridge-Token": "bad_token"})
        verify_bridge_token(request=req, payload={"symbol": "XAUUSD"})
    assert excinfo.value.status_code == 401


def test_bridge_auth_accepts_valid_token():
    """Item 16: Bridge requests are accepted with valid token in header, body, or query param."""
    ctx.bridge_token = "apex_secret_token_123"

    # Valid in payload
    verify_bridge_token(request=DummyRequest(), payload={"symbol": "XAUUSD", "token": "apex_secret_token_123"})

    # Valid in header
    req = DummyRequest(headers={"X-Bridge-Token": "apex_secret_token_123"})
    verify_bridge_token(request=req, payload={"symbol": "XAUUSD"})

    # Open bridge mode (empty token)
    ctx.bridge_token = ""
    verify_bridge_token(request=DummyRequest(), payload={"symbol": "XAUUSD"})


# ==============================================================================
# 2. Debouncing and Rate Limiting Tests
# ==============================================================================
@pytest.mark.anyio
async def test_endpoint_rate_limiter_debounces_rapid_calls():
    """Item 17: Rate limiter debounces rapid repeated invocations."""
    key = "test_rate_limiter_unique_key"
    first = await endpoint_rate_limiter.check(key, min_interval=0.3)
    assert first is True

    # Immediate second call must be blocked
    second = await endpoint_rate_limiter.check(key, min_interval=0.3)
    assert second is False

    # After interval elapses, next call is permitted
    await asyncio.sleep(0.35)
    third = await endpoint_rate_limiter.check(key, min_interval=0.3)
    assert third is True


@pytest.mark.anyio
async def test_action_test_trade_debouncing():
    """Item 17: Rapid clicks on 1-Click Test Trade are debounced."""
    broker = MockBrokerAdapter()
    risk = GoldRiskEngine(budget_usd=1000.0, risk_per_trade_pct=2.0)
    tm = TradeManager(
        broker=broker,
        data_hub=DummyDataHub(),
        news_reactor=DummyNewsReactor(),
        risk_engine=risk,
    )
    ctx.broker = broker
    ctx.risk_engine = risk
    ctx.trade_manager = tm
    ctx.active_symbol = "XAUUSD"
    ctx.live_trading_enabled = False

    # Force reset limiter timestamp
    endpoint_rate_limiter._last_call["test_trade"] = 0.0

    res1 = await action_test_trade(direction="BUY", volume=0.01)
    assert res1["status"] == "OK"

    # Immediate second click
    res2 = await action_test_trade(direction="BUY", volume=0.01)
    assert res2["status"] == "REJECTED"
    assert "debounced" in res2["message"].lower()


@pytest.mark.anyio
async def test_action_close_all_debouncing():
    """Item 17: Emergency close-all button is debounced to avoid thrashing."""
    broker = MockBrokerAdapter()
    risk = GoldRiskEngine(budget_usd=1000.0, risk_per_trade_pct=2.0)
    tm = TradeManager(
        broker=broker,
        data_hub=DummyDataHub(),
        news_reactor=DummyNewsReactor(),
        risk_engine=risk,
    )
    ctx.broker = broker
    ctx.trade_manager = tm

    endpoint_rate_limiter._last_call["close_all"] = 0.0
    res1 = await action_close_all()
    assert res1["status"] == "OK"

    res2 = await action_close_all()
    assert res2["status"] == "REJECTED"
    assert "debounced" in res2["message"].lower()


# ==============================================================================
# 3. Circuit Breaker & Risk Guard Gating on Manual Actions
# ==============================================================================
@pytest.mark.anyio
async def test_action_test_trade_blocked_by_circuit_breaker():
    """Item 17: 1-Click Test Trade is gated by active circuit breakers."""
    broker = MockBrokerAdapter()
    risk = GoldRiskEngine(budget_usd=1000.0, risk_per_trade_pct=2.0)
    tm = TradeManager(
        broker=broker,
        data_hub=DummyDataHub(),
        news_reactor=DummyNewsReactor(),
        risk_engine=risk,
        max_daily_drawdown_pct=3.0,
    )
    # Trip circuit breaker by simulating daily drawdown
    tm._daily_realized_pnl = -50.0  # -5.0% on $1000 account (limit is 3.0%)
    ctx.broker = broker
    ctx.trade_manager = tm
    ctx.risk_engine = risk

    endpoint_rate_limiter._last_call["test_trade"] = 0.0

    res = await action_test_trade(direction="BUY", volume=0.01)
    assert res["status"] == "REJECTED"
    assert "circuit breaker" in res["message"].lower() or "drawdown" in res["message"].lower()


@pytest.mark.anyio
async def test_action_test_trade_blocked_by_trading_pause():
    """Item 17: 1-Click Test Trade is blocked when manual pause is active."""
    broker = MockBrokerAdapter()
    risk = GoldRiskEngine(budget_usd=1000.0, risk_per_trade_pct=2.0)
    tm = TradeManager(
        broker=broker,
        data_hub=DummyDataHub(),
        news_reactor=DummyNewsReactor(),
        risk_engine=risk,
    )
    tm.pause_trading(minutes=30)
    ctx.broker = broker
    ctx.trade_manager = tm
    ctx.risk_engine = risk

    endpoint_rate_limiter._last_call["test_trade"] = 0.0

    res = await action_test_trade(direction="BUY", volume=0.01)
    assert res["status"] == "REJECTED"
    assert "paused" in res["message"].lower()


# ==============================================================================
# 4. Broker Volume Bounds and Monetary Risk Budget Enforcement
# ==============================================================================
@pytest.mark.anyio
async def test_action_test_trade_validates_volume_bounds():
    """Item 17: Test trade rejects volumes violating broker min/max bounds."""
    broker = MockBrokerAdapter()
    risk = GoldRiskEngine(budget_usd=1000.0, risk_per_trade_pct=2.0)
    tm = TradeManager(
        broker=broker,
        data_hub=DummyDataHub(),
        news_reactor=DummyNewsReactor(),
        risk_engine=risk,
    )
    ctx.broker = broker
    ctx.trade_manager = tm
    ctx.risk_engine = risk

    # Volume < min_lot (0.01 is min, test 0.005)
    endpoint_rate_limiter._last_call["test_trade"] = 0.0
    res_min = await action_test_trade(direction="BUY", volume=0.005)
    assert res_min["status"] == "REJECTED"
    assert "below broker minimum" in res_min["message"].lower()

    # Volume > max_lot (10.0 is max, test 20.0)
    endpoint_rate_limiter._last_call["test_trade"] = 0.0
    res_max = await action_test_trade(direction="BUY", volume=20.0)
    assert res_max["status"] == "REJECTED"
    assert "exceeds broker maximum" in res_max["message"].lower()


@pytest.mark.anyio
async def test_action_test_trade_rejects_excessive_monetary_risk():
    """Item 17: Test trade rejects order when monetary risk exceeds configured risk budget."""
    broker = MockBrokerAdapter(initial_balance=50.0)  # small balance: $50
    # Risk budget: 2% of $50 = $1.00
    risk = GoldRiskEngine(budget_usd=50.0, risk_per_trade_pct=2.0)
    tm = TradeManager(
        broker=broker,
        data_hub=DummyDataHub(),
        news_reactor=DummyNewsReactor(),
        risk_engine=risk,
    )
    ctx.broker = broker
    ctx.trade_manager = tm
    ctx.risk_engine = risk

    # A volume of 0.05 on Gold with 2.50 SL distance = $12.50 risk, which vastly exceeds $1.00 budget
    endpoint_rate_limiter._last_call["test_trade"] = 0.0
    res = await action_test_trade(direction="BUY", volume=0.05)
    assert res["status"] == "REJECTED"
    assert "monetary risk" in res["message"].lower() or "budget" in res["message"].lower()


# ==============================================================================
# 5. Manual Position Protection
# ==============================================================================
@pytest.mark.anyio
async def test_modify_position_protects_manual_positions_by_default():
    """Item 17: Modifying manually opened MT5 positions is blocked when manage_manual_positions=False."""
    broker = MockBrokerAdapter()
    ticket = 7701
    pos = Position(
        ticket=ticket,
        symbol="XAUUSD",
        type=PositionType.BUY,
        volume=0.02,
        open_price=2650.0,
        sl=2645.0,
        tp=2665.0,
        open_time=time.time(),
        current_price=2650.0,
        is_user_manual=True,  # User opened manually in MT5
    )
    broker._positions[ticket] = pos
    ctx.broker = broker
    ctx.trade_manager = None
    ctx.manage_manual_positions = False

    endpoint_rate_limiter._last_call[f"modify_pos_{ticket}"] = 0.0
    payload = ModifyPositionPayload(ticket=ticket, sl=2648.0, tp=2670.0)
    res = await action_modify_position(payload)

    assert res["status"] == "REJECTED"
    assert "manage_manual_positions=false" in res["message"].lower()

    # When explicit opt-in is enabled, modification is permitted
    ctx.manage_manual_positions = True
    endpoint_rate_limiter._last_call[f"modify_pos_{ticket}"] = 0.0
    res_opt_in = await action_modify_position(payload)
    assert res_opt_in["status"] == "OK"


# ==============================================================================
# 6. Protective Stop Ratchet & Broker Stop Constraints
# ==============================================================================
@pytest.mark.anyio
async def test_modify_position_rejects_loosening_protective_stop():
    """Item 17: Dashboard modify endpoint strictly forbids loosening existing stop loss."""
    broker = MockBrokerAdapter()
    ticket = 8801
    pos = Position(
        ticket=ticket,
        symbol="XAUUSD",
        type=PositionType.BUY,
        volume=0.01,
        open_price=2650.0,
        sl=2645.0,
        tp=2665.0,
        open_time=time.time(),
        current_price=2650.0,
        is_user_manual=False,
    )
    broker._positions[ticket] = pos
    ctx.broker = broker
    ctx.trade_manager = None
    ctx.manage_manual_positions = False

    # Attempt to loosen SL from 2645.0 down to 2640.0 on a BUY position
    endpoint_rate_limiter._last_call[f"modify_pos_{ticket}"] = 0.0
    payload = ModifyPositionPayload(ticket=ticket, sl=2640.0, tp=2665.0)
    res = await action_modify_position(payload)

    assert res["status"] == "REJECTED"
    assert "cannot loosen protective stop loss" in res["message"].lower()


@pytest.mark.anyio
async def test_modify_position_validates_sl_tp_sides_and_min_distance():
    """Item 17: Dashboard modify endpoint validates SL/TP sides and min distance."""
    broker = MockBrokerAdapter()
    ticket = 8802
    pos = Position(
        ticket=ticket,
        symbol="XAUUSD",
        type=PositionType.BUY,
        volume=0.01,
        open_price=2650.0,
        sl=2645.0,
        tp=2665.0,
        open_time=time.time(),
        current_price=2650.0,
        is_user_manual=False,
    )
    broker._positions[ticket] = pos
    ctx.broker = broker
    ctx.trade_manager = None
    ctx.manage_manual_positions = False

    # 1. SL above current price on BUY
    endpoint_rate_limiter._last_call[f"modify_pos_{ticket}"] = 0.0
    res1 = await action_modify_position(ModifyPositionPayload(ticket=ticket, sl=2655.0, tp=2670.0))
    assert res1["status"] == "REJECTED"
    assert "below current price" in res1["message"].lower()

    # 2. TP below current price on BUY
    endpoint_rate_limiter._last_call[f"modify_pos_{ticket}"] = 0.0
    res2 = await action_modify_position(ModifyPositionPayload(ticket=ticket, sl=2648.0, tp=2649.0))
    assert res2["status"] == "REJECTED"
    assert "above current price" in res2["message"].lower()

    # 3. SL distance below broker minimum (stops_level=20 -> min_dist=0.20)
    endpoint_rate_limiter._last_call[f"modify_pos_{ticket}"] = 0.0
    res3 = await action_modify_position(ModifyPositionPayload(ticket=ticket, sl=2649.95, tp=2670.0))
    assert res3["status"] == "REJECTED"
    assert "below broker min stop distance" in res3["message"].lower()


# ==============================================================================
# 7. Truthful System Status Reporting
# ==============================================================================
@pytest.mark.anyio
async def test_status_endpoint_reports_truthful_state_in_paper_mode():
    """Item 16: /api/status never claims live MT5 trading when in paper/scanner mode."""
    broker = PaperSimAdapter(initial_balance=500.0)
    ctx.broker = broker
    ctx.live_trading_enabled = False
    ctx.mt5_last_seen = asyncio.get_event_loop().time()  # Bridge is pinging

    status = await get_status()

    assert status["is_bridge_connected"] is True
    assert status["is_mt5_live"] is False
    assert status["live_trading_active"] is False
    assert status["is_paper_mode"] is True
    assert "PAPER" in status["broker_name"] or "SCANNER" in status["broker_name"]
    assert "LIVE" not in status["broker_name"]


@pytest.mark.anyio
async def test_status_endpoint_reports_live_only_when_explicitly_active():
    """Item 16: /api/status reports live trading only when live adapter and flag are enabled."""
    broker = MockBrokerAdapter(initial_balance=5000.0, is_paper=False)
    ctx.broker = broker
    ctx.live_trading_enabled = True
    ctx.mt5_last_seen = asyncio.get_event_loop().time()

    status = await get_status()

    assert status["is_bridge_connected"] is True
    assert status["is_mt5_live"] is True
    assert status["live_trading_active"] is True
    assert status["is_paper_mode"] is False
    assert "(LIVE)" in status["broker_name"]
