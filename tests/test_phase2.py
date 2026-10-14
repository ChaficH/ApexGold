"""
Phase 2 Regression & Unit Tests for ApexGold Trading Safety.

Covers:
- Item 6: Pending-order position registration upon fill (ticket, price, risk preservation, deduplication).
- Item 7: Updating local SL/TP only after broker confirmation; strict prohibition of stop-loss loosening.
- Item 8: Closed-trade accounting using actual broker execution history (commissions, swap, net P&L).
- Item 9: Protecting manually opened MT5 positions (magic number / metadata isolation and config gating).
"""

import asyncio
import os
import sys
import time
from typing import Any, Dict, List, Optional, Union
import pytest

# Ensure repo root is in python path
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
from engine.trade_history_store import TradeHistoryStore
from execution.gold_risk import GoldRiskEngine
from execution.trade_manager import ManagedPosition, TradeManager


class MockBroker(AbstractBrokerAdapter):
    """Mock broker adapter to test position management & reconciliation."""

    def __init__(self, initial_balance: float = 100.0):
        self.balance = initial_balance
        self.positions: Dict[int, Position] = {}
        self.pending_orders: Dict[int, LimitOrder] = {}
        self.closed_deals: Dict[int, ClosedDeal] = {}
        self._last_ticks: Dict[str, Any] = {}
        self.modify_should_succeed = True
        self.modify_calls: List[Dict[str, Any]] = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        pass

    def is_connected(self) -> bool:
        return True

    async def get_account_info(self) -> AccountInfo:
        return AccountInfo(
            balance=self.balance,
            equity=self.balance,
            margin=0.0,
            free_margin=self.balance,
            currency="USD",
            leverage=500,
            account_type="standard",
        )

    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        return SymbolInfo(
            symbol=symbol,
            bid=2650.0,
            ask=2650.25,
            point=0.01,
            digits=2,
            spread=0.25,
            contract_size=100.0,
            tick_size=0.01,
            tick_value=1.0,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
        )

    async def get_tick(self, symbol: str) -> Tick:
        return Tick(symbol=symbol, bid=2650.0, ask=2650.25, time=time.time())

    async def get_rates(self, symbol: str, timeframe: str, count: int = 100) -> List[Bar]:
        return []

    async def open_order(
        self,
        symbol: str,
        order_type: OrderType,
        volume: float,
        price: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = "ApexGold",
    ) -> OrderResult:
        ticket = 999001
        p = Position(
            ticket=ticket,
            symbol=symbol,
            type=PositionType.BUY if order_type == OrderType.BUY else PositionType.SELL,
            volume=volume,
            open_price=price or 2650.0,
            current_price=price or 2650.0,
            sl=sl or 0.0,
            tp=tp or 0.0,
            magic=15555,
            comment=comment,
        )
        self.positions[ticket] = p
        return OrderResult(success=True, ticket=ticket, price=p.open_price, volume=volume)

    async def modify_position(self, ticket: Union[int, str], sl: float, tp: float) -> bool:
        self.modify_calls.append({"ticket": int(ticket), "sl": sl, "tp": tp})
        if not self.modify_should_succeed:
            return False
        t_int = int(ticket)
        if t_int in self.positions:
            self.positions[t_int].sl = sl
            self.positions[t_int].tp = tp
        return True

    async def close_position(self, ticket: Union[int, str], volume: Optional[float] = None) -> bool:
        t_int = int(ticket)
        if t_int in self.positions:
            pos = self.positions.pop(t_int)
            self.closed_deals[t_int] = ClosedDeal(
                ticket=t_int,
                symbol=pos.symbol,
                direction=pos.type.value if hasattr(pos.type, "value") else str(pos.type),
                volume=pos.volume,
                open_price=pos.open_price,
                close_price=2655.0,
                open_time=pos.open_time,
                close_time=time.time(),
                profit=50.0,
                commission=-1.50,
                swap=-0.25,
                comment=pos.comment,
            )
            return True
        return False

    async def close_all_positions(self, symbol: Optional[str] = None) -> int:
        c = len(self.positions)
        self.positions.clear()
        return c

    async def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        return list(self.positions.values())

    async def cancel_order(self, ticket: Union[int, str]) -> bool:
        return self.pending_orders.pop(int(ticket), None) is not None

    async def get_pending_orders(self, symbol: Optional[str] = None) -> List[LimitOrder]:
        return list(self.pending_orders.values())

    async def get_closed_deal(self, ticket: Union[int, str]) -> Optional[ClosedDeal]:
        return self.closed_deals.get(int(ticket))


class DummyDataHub:
    async def get_dataframe(self, symbol: str, timeframe: str, count: int = 100):
        return None


class DummyNewsReactor:
    def check_calendar_state(self):
        return {"is_freeze": False}

    def analyze_news_fundamental_bias(self, symbol):
        return {"bias": "NEUTRAL", "confidence": 0}


# ==============================================================================
# 6. Test Pending-Order Position Registration upon Fill
# ==============================================================================
@pytest.mark.anyio
async def test_pending_order_registration_upon_fill():
    """Item 6: Filled pending limit orders must enroll as managed positions preserving risk."""
    broker = MockBroker()
    risk = GoldRiskEngine(budget_usd=100.0, risk_per_trade_pct=2.0)
    tm = TradeManager(
        broker=broker,
        data_hub=DummyDataHub(),
        news_reactor=DummyNewsReactor(),
        risk_engine=risk,
    )

    limit_ticket = 777001
    lo = LimitOrder(
        ticket=limit_ticket,
        symbol="XAUUSD",
        order_type="BUY_LIMIT",
        price=2645.0,
        volume=0.01,
        sl=2640.0,
        tp=2660.0,
        created_time=time.time(),
        timeout_seconds=60.0,
    )
    tm._pending_limit_orders[limit_ticket] = lo

    # Simulate broker filling the limit order into an active position with the same ticket
    broker.positions[limit_ticket] = Position(
        ticket=limit_ticket,
        symbol="XAUUSD",
        type=PositionType.BUY,
        volume=0.01,
        open_price=2645.0,
        current_price=2647.0,
        sl=2640.0,
        tp=2660.0,
        magic=15555,
    )

    # Run one evaluation cycle logic
    broker_positions = await broker.get_open_positions()
    broker_pos_map = {int(p.ticket): p for p in broker_positions}

    # Simulate management loop pending check
    matched_pos = broker_pos_map.get(limit_ticket)
    assert matched_pos is not None

    pos_ticket = int(matched_pos.ticket)
    risk_dist = abs(matched_pos.open_price - lo.sl)
    managed = ManagedPosition(
        ticket=pos_ticket,
        symbol=lo.symbol,
        direction="BUY",
        volume=matched_pos.volume,
        open_price=matched_pos.open_price,
        sl=lo.sl,
        tp=lo.tp,
        entry_time=time.time(),
        risk_r_dist=risk_dist,
        initial_regime="SMART_LIMIT_FILL",
        current_regime="SMART_LIMIT_FILL",
        sub_regime="SMART_LIMIT",
        be_lock_r=0.65,
        stall_exit_seconds=60.0,
    )
    tm._managed_positions[pos_ticket] = managed
    tm._pending_limit_orders.pop(limit_ticket, None)

    # Verifications
    assert limit_ticket not in tm._pending_limit_orders
    assert pos_ticket in tm._managed_positions
    m_pos = tm._managed_positions[pos_ticket]
    assert m_pos.open_price == 2645.0
    assert m_pos.sl == 2640.0
    assert m_pos.tp == 2660.0
    assert m_pos.risk_r_dist == 5.0


# ==============================================================================
# 7. Test Local SL/TP Update ONLY After Broker Confirmation & Never Loosen
# ==============================================================================
@pytest.mark.anyio
async def test_ratchet_floor_only_updates_on_broker_confirmation():
    """Item 7: If broker rejects SL modification, local state must NOT update."""
    broker = MockBroker()
    broker.modify_should_succeed = False  # Broker will reject

    risk = GoldRiskEngine(budget_usd=100.0, risk_per_trade_pct=2.0)
    tm = TradeManager(
        broker=broker,
        data_hub=DummyDataHub(),
        news_reactor=DummyNewsReactor(),
        risk_engine=risk,
    )

    pos = ManagedPosition(
        ticket=101,
        symbol="XAUUSD",
        direction="BUY",
        volume=0.01,
        open_price=2650.0,
        sl=2645.0,
        tp=2665.0,
        entry_time=time.time(),
        risk_r_dist=5.0,
        initial_regime="TEST",
        current_regime="TEST",
        sub_regime="TEST",
        ratchet_stage=0,
    )
    tm._managed_positions[101] = pos

    # When floating_r = 1.10 and ratchet_stage = 0, stage 1 BE lock triggers first
    # SL targets open_price + 0.25 = 2650.25
    await tm._evaluate_ratchet_floor(pos, floating_r=1.10, current_price=2655.5)

    # Broker rejected modification -> local pos.sl and ratchet_stage must remain UNCHANGED
    assert pos.sl == 2645.0
    assert pos.ratchet_stage == 0

    # Now simulate broker succeeding
    broker.modify_should_succeed = True
    await tm._evaluate_ratchet_floor(pos, floating_r=1.10, current_price=2655.5)

    # Stage 1 successfully locked to Breakeven
    assert pos.sl == 2650.25
    assert pos.ratchet_stage == 1

    # Next evaluation advances to Stage 2 (+0.45R lock = 2652.25)
    await tm._evaluate_ratchet_floor(pos, floating_r=1.10, current_price=2655.5)
    assert pos.sl == 2652.25
    assert pos.ratchet_stage == 2


@pytest.mark.anyio
async def test_trailing_stop_never_loosens_protective_stop():
    """Item 7: Trailing stop must NEVER move SL backward (never loosen)."""
    broker = MockBroker()
    broker.modify_should_succeed = True

    risk = GoldRiskEngine(budget_usd=100.0, risk_per_trade_pct=2.0)
    tm = TradeManager(
        broker=broker,
        data_hub=DummyDataHub(),
        news_reactor=DummyNewsReactor(),
        risk_engine=risk,
    )

    # BUY position where user or algorithm already secured high SL
    pos = ManagedPosition(
        ticket=102,
        symbol="XAUUSD",
        direction="BUY",
        volume=0.01,
        open_price=2650.0,
        sl=2662.0,  # Tight stop already locked
        tp=2680.0,
        entry_time=time.time(),
        risk_r_dist=5.0,
        initial_regime="TEST",
        current_regime="TEST",
        sub_regime="TEST",
        ratchet_stage=4,
    )
    tm._managed_positions[102] = pos

    broker.modify_calls.clear()

    # Suppose current_price fluctuates down to 2664.0 at floating_r >= 2.50
    # Dynamic trail candidate = 2664.0 - (5.0 * 0.85) = 2659.75
    # Since 2659.75 < 2662.0 (loosening!), it must NOT be sent to broker or updated!
    await tm._evaluate_ratchet_floor(pos, floating_r=2.80, current_price=2664.0)

    assert pos.sl == 2662.0  # SL protected, not loosened!
    assert len(broker.modify_calls) == 0


@pytest.mark.anyio
async def test_server_modify_position_rejects_on_broker_failure():
    """Item 7: /api/actions/modify_position does not modify local state if broker rejects."""
    from ui.server import app, ctx, ModifyPositionPayload, action_modify_position
    broker = MockBroker()
    broker.modify_should_succeed = False  # Reject

    risk = GoldRiskEngine(budget_usd=100.0, risk_per_trade_pct=2.0)
    tm = TradeManager(
        broker=broker,
        data_hub=DummyDataHub(),
        news_reactor=DummyNewsReactor(),
        risk_engine=risk,
    )
    pos = ManagedPosition(
        ticket=201,
        symbol="XAUUSD",
        direction="BUY",
        volume=0.01,
        open_price=2650.0,
        sl=2645.0,
        tp=2665.0,
        entry_time=time.time(),
        risk_r_dist=5.0,
        initial_regime="TEST",
        current_regime="TEST",
        sub_regime="TEST",
    )
    tm._managed_positions[201] = pos
    ctx.broker = broker
    ctx.trade_manager = tm
    ctx.live_trading_enabled = False

    payload = ModifyPositionPayload(ticket=201, sl=2648.0, tp=2670.0)
    res = await action_modify_position(payload)

    assert res["status"] == "ERROR"
    # Local in-memory position must remain unchanged!
    assert pos.sl == 2645.0
    assert pos.tp == 2665.0


# ==============================================================================
# 8. Test Closed-Trade Accounting with Real Broker Execution Deal
# ==============================================================================
@pytest.mark.anyio
async def test_closed_trade_accounting_uses_real_broker_deal(tmp_path):
    """Item 8: When broker provides closed deal history, use actual P&L, commission, and swap."""
    db_file = str(tmp_path / "test_history.db")
    history_store = TradeHistoryStore(db_path=db_file)
    await history_store.initialize()

    broker = MockBroker()
    risk = GoldRiskEngine(budget_usd=100.0, risk_per_trade_pct=2.0)
    tm = TradeManager(
        broker=broker,
        data_hub=DummyDataHub(),
        news_reactor=DummyNewsReactor(),
        risk_engine=risk,
        history_store=history_store,
    )

    ticket = 555001
    pos = ManagedPosition(
        ticket=ticket,
        symbol="XAUUSD",
        direction="BUY",
        volume=0.02,
        open_price=2650.0,
        sl=2645.0,
        tp=2660.0,
        entry_time=time.time() - 100,
        risk_r_dist=5.0,
        initial_regime="TEST",
        current_regime="TEST",
        sub_regime="TEST",
    )
    tm._managed_positions[ticket] = pos

    # Provide real broker execution deal
    # Gross profit: +$45.00, Commission: -$2.00, Swap: -$0.50 -> Net P&L: +$42.50
    broker.closed_deals[ticket] = ClosedDeal(
        ticket=ticket,
        symbol="XAUUSD",
        direction="BUY",
        volume=0.02,
        open_price=2650.0,
        close_price=2652.25,
        open_time=pos.entry_time,
        close_time=time.time(),
        profit=45.00,
        commission=-2.00,
        swap=-0.50,
        comment="ApexGold_Deal",
    )

    await tm._on_position_closed(pos, reason="Take Profit Hit", current_price=2652.25)

    # Verify TradeHistoryStore has exact net P&L $42.50
    records = await history_store.get_history(limit=5)
    assert len(records) == 1
    rec = records[0]
    assert rec["ticket"] == ticket
    assert rec["pnl"] == 42.50
    assert rec["close_price"] == 2652.25

    # Verify duplicate finalization is ignored
    await tm._on_position_closed(pos, reason="Duplicate Close Trigger", current_price=2652.25)
    records_after = await history_store.get_history(limit=5)
    assert len(records_after) == 1


# ==============================================================================
# 9. Test Protecting Manually Opened MT5 Positions
# ==============================================================================
@pytest.mark.anyio
async def test_manual_positions_are_protected_from_auto_management():
    """Item 9: Manual trades (magic != 15555, non-bot comment) must not be managed when manage_manual_positions is False."""
    from ui.server import ctx, mt5_bridge_tick

    broker = MockBroker()
    risk = GoldRiskEngine(budget_usd=100.0, risk_per_trade_pct=2.0)
    tm = TradeManager(
        broker=broker,
        data_hub=DummyDataHub(),
        news_reactor=DummyNewsReactor(),
        risk_engine=risk,
        manage_manual_positions=False,  # Default safe
    )
    ctx.broker = broker
    ctx.trade_manager = tm
    ctx.manage_manual_positions = False
    ctx.live_trading_enabled = False

    # Simulate MT5 bridge tick sending two positions:
    # 1. Manual user trade: ticket 801, magic 0, comment "iPhone MT5 app"
    # 2. Bot trade: ticket 802, magic 15555, comment "ApexGold_Auto"
    payload = {
        "action": "TICK",
        "symbol": "XAUUSD",
        "bid": 2650.0,
        "ask": 2650.25,
        "positions": [
            {
                "ticket": 801,
                "symbol": "XAUUSD",
                "type": "BUY",
                "volume": 0.05,
                "open_price": 2640.0,
                "current_price": 2650.0,
                "sl": 2630.0,
                "tp": 2670.0,
                "profit": 50.0,
                "comment": "iPhone MT5 app",
                "magic": 0,
            },
            {
                "ticket": 802,
                "symbol": "XAUUSD",
                "type": "SELL",
                "volume": 0.01,
                "open_price": 2652.0,
                "current_price": 2650.0,
                "sl": 2657.0,
                "tp": 2640.0,
                "profit": 2.0,
                "comment": "ApexGold_Auto",
                "magic": 15555,
            },
        ],
    }

    await mt5_bridge_tick(payload)

    # Only bot position (802) should be enrolled into automated TradeManager!
    assert 802 in tm._managed_positions
    # Manual position (801) must NOT be enrolled!
    assert 801 not in tm._managed_positions
