"""
Automated Regression Test Suite for ApexGold Blocker Fixes.

Verifies:
1. Safe post-order recovery matching:
   - Volume match alone or recent timestamp alone CANNOT match an unrelated trade.
   - Ambiguous recovery (multiple matches) halts trading, locks entries, and logs alert.
2. Broker state verification after attaching SL or issuing emergency close:
   - Verifies broker truth; if position remains unprotected or close fails, persists
     CRITICAL_CLOSE_FAILED and locks trading.
3. Strict risk calculation in execution/gold_risk.py:
   - Rejects non-positive contract_size, point, or price (no 4140.0 or 1.0/100.0 fallbacks).
   - Strictly enforces risk budget: actual monetary risk never exceeds budget (no 1.05 buffer).
4. API & WebSocket Security:
   - Explicit CORS origins (no wildcard "*").
   - Live bridge requires configured authentication token.
   - Runtime startup defaults to scanner mode.
   - WebSocket validates Origin and live token.
5. Persistent Candle Provenance in SQLite:
   - Synthetic bars retain is_synthetic=True and source='SYNTHETIC' across restart.
   - Synthetic data cannot silently masquerade as broker data.
"""

import asyncio
import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Union
import pytest
from fastapi import HTTPException
from starlette.websockets import WebSocketDisconnect

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
from engine.data_hub import DataHub
from engine.trade_history_store import TradeHistoryStore
from execution.gold_risk import GoldRiskEngine
from execution.trade_manager import ManagedPosition, TradeManager
from strategy.confluence_scorer import ConfluenceSignal
from ui.server import (
    ALLOWED_ORIGINS,
    ServerContext,
    action_close_position,
    ctx,
    verify_bridge_token,
    websocket_telemetry,
)


class MockBrokerForRecovery(AbstractBrokerAdapter):
    """Mock broker allowing precise control over positions and modify/close responses."""

    def __init__(self, initial_balance: float = 1000.0):
        self.balance = initial_balance
        self.equity = initial_balance
        self._positions: Dict[int, Position] = {}
        self.modify_success: bool = True
        self.close_success: bool = True
        self.close_removes_from_broker: bool = True

    async def connect(self) -> bool: return True
    async def disconnect(self) -> None: pass
    def is_connected(self) -> bool: return True

    async def get_account_info(self) -> AccountInfo:
        return AccountInfo(
            balance=self.balance,
            equity=self.equity,
            margin=0.0,
            free_margin=self.balance,
            currency="USD",
            leverage=500,
        )

    async def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        return SymbolInfo(
            symbol=symbol,
            point=0.01,
            digits=2,
            spread=0.25,
            contract_size=100.0,
            tick_size=0.01,
            tick_value=1.0,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            ask=2650.25,
            bid=2650.00,
        )

    async def get_tick(self, symbol: str) -> Optional[Tick]:
        return Tick(symbol=symbol, bid=2650.00, ask=2650.25, spread=0.25)

    async def get_rates(self, symbol: str, timeframe: str, count: int = 50) -> List[Bar]:
        return []

    async def open_order(self, *args, **kwargs) -> OrderResult:
        raise ConnectionResetError("Broker socket dropped")

    async def modify_position(self, ticket: Union[int, str], sl: float, tp: float) -> bool:
        t_int = int(ticket)
        if self.modify_success and t_int in self._positions:
            self._positions[t_int].sl = sl
            self._positions[t_int].tp = tp
            return True
        return False

    async def close_position(self, ticket: Union[int, str], volume: Optional[float] = None) -> bool:
        t_int = int(ticket)
        if self.close_success:
            if self.close_removes_from_broker:
                self._positions.pop(t_int, None)
            return True
        return False

    async def close_all_positions(self, symbol: Optional[str] = None) -> int:
        c = len(self._positions)
        if self.close_removes_from_broker:
            self._positions.clear()
        return c

    async def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        return list(self._positions.values())

    async def get_pending_orders(self, symbol: Optional[str] = None) -> List[LimitOrder]:
        return []

    async def cancel_order(self, ticket: Union[int, str]) -> bool:
        return True

    async def get_closed_deal(self, ticket: Union[int, str]) -> Optional[ClosedDeal]:
        return None


@pytest.fixture
def sample_signal():
    return ConfluenceSignal(
        symbol="XAUUSD",
        direction="BUY",
        score=85,
        tier="TIER_1_SNIPER",
        regime="TRENDING",
        sub_regime="BULL_MOMENTUM",
        target_rr=2.0,
        entry_price=2650.00,
        suggested_sl=2647.50,
        suggested_tp=2655.00,
        risk_r_dist=2.50,
        be_lock_r=1.0,
        stall_exit_seconds=120.0,
        breakdown={},
        reason="Sniper Confluence",
        time=time.time(),
        order_type="MARKET",
    )


# ==============================================================================
# 1. Post-Order Recovery: Strict Multi-Factor Matching & Ambiguity Lock
# ==============================================================================
@pytest.mark.anyio
async def test_post_order_recovery_rejects_unrelated_positions_with_matching_volume_or_recent_time(sample_signal):
    """
    Verify that an unrelated position that only shares the same volume (0.01)
    or only has a recent timestamp is NOT matched by post-order recovery.
    """
    mock_broker = MockBrokerForRecovery()
    # Unrelated position: same volume (0.01) but wrong direction (SELL) and distant price
    unrelated_pos = Position(
        ticket=55511,
        symbol="XAUUSD",
        type=PositionType.SELL,
        volume=0.01,
        open_price=2690.00,
        current_price=2690.00,
        sl=2695.00,
        tp=2680.00,
        open_time=time.time() - 5.0, # recent
        magic=99999, # different magic
    )
    mock_broker._positions[55511] = unrelated_pos

    tm = TradeManager(
        broker=mock_broker,
        data_hub=None,
        news_reactor=None,
        risk_engine=GoldRiskEngine(),
    )

    result = await tm._reconcile_post_order_failure(
        symbol="XAUUSD",
        direction="BUY",
        lot_size=0.01,
        target_entry=2650.00,
        padded_sl=2647.25,
        suggested_tp=2655.00,
        comment="ApexGold_Sniper",
        signal=sample_signal,
    )

    assert result is None, "Unrelated SELL position must not be matched to a BUY order!"
    assert 55511 not in tm._managed_positions


@pytest.mark.anyio
async def test_post_order_recovery_ambiguity_locks_trading_and_alerts(sample_signal):
    """
    If multiple unmanaged positions match the order specifications, recovery is ambiguous.
    The system must NOT guess: it must activate contradictory state, lock entries, and record an alert.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        store = TradeHistoryStore(db_path=db_path)
        await store.initialize()

        mock_broker = MockBrokerForRecovery()
        now = time.time()
        # Candidate 1
        pos1 = Position(
            ticket=7001,
            symbol="XAUUSD",
            type=PositionType.BUY,
            volume=0.01,
            open_price=2650.10,
            current_price=2650.10,
            sl=2647.25,
            tp=2655.00,
            open_time=now - 2.0,
            magic=15555,
        )
        # Candidate 2 (ambiguous duplicate)
        pos2 = Position(
            ticket=7002,
            symbol="XAUUSD",
            type=PositionType.BUY,
            volume=0.01,
            open_price=2650.15,
            current_price=2650.15,
            sl=2647.25,
            tp=2655.00,
            open_time=now - 1.0,
            magic=15555,
        )
        mock_broker._positions[7001] = pos1
        mock_broker._positions[7002] = pos2

        tm = TradeManager(
            broker=mock_broker,
            data_hub=None,
            news_reactor=None,
            risk_engine=GoldRiskEngine(),
            history_store=store,
        )

        result = await tm._reconcile_post_order_failure(
            symbol="XAUUSD",
            direction="BUY",
            lot_size=0.01,
            target_entry=2650.00,
            padded_sl=2647.25,
            suggested_tp=2655.00,
            comment="ApexGold",
            signal=sample_signal,
        )

        assert result is None
        assert tm._contradictory_state_active is True
        assert "Ambiguous post-order recovery" in tm._contradictory_state_reason
        assert tm.is_paused() is True

        events = await store.get_audit_events(limit=10)
        ambig_events = [e for e in events if e.get("event_type") == "AMBIGUOUS_RECOVERY_ALERT"]
        assert len(ambig_events) == 1
        assert "7001" in ambig_events[0]["details"]
        assert "7002" in ambig_events[0]["details"]

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


# ==============================================================================
# 2. Broker State Verification After Attaching SL and Emergency Close
# ==============================================================================
@pytest.mark.anyio
async def test_emergency_close_failure_persists_critical_audit_and_maintains_lock(sample_signal):
    """
    If a recovered position has sl=0.0 and modifying it fails, an emergency close is issued.
    If the broker query reveals that the position remains open (emergency close failed),
    the system must persist CRITICAL_CLOSE_FAILED and maintain the trading lock.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        store = TradeHistoryStore(db_path=db_path)
        await store.initialize()

        mock_broker = MockBrokerForRecovery()
        now = time.time()
        unprotected_pos = Position(
            ticket=8881,
            symbol="XAUUSD",
            type=PositionType.BUY,
            volume=0.01,
            open_price=2650.00,
            current_price=2650.00,
            sl=0.0, # Unprotected!
            tp=2655.00,
            open_time=now - 2.0,
            magic=15555,
        )
        mock_broker._positions[8881] = unprotected_pos
        mock_broker.modify_success = False # Modify fails!
        mock_broker.close_removes_from_broker = False # Emergency close fails to remove on broker!

        tm = TradeManager(
            broker=mock_broker,
            data_hub=None,
            news_reactor=None,
            risk_engine=GoldRiskEngine(),
            history_store=store,
        )

        result = await tm._reconcile_post_order_failure(
            symbol="XAUUSD",
            direction="BUY",
            lot_size=0.01,
            target_entry=2650.00,
            padded_sl=2647.25,
            suggested_tp=2655.00,
            comment="ApexGold",
            signal=sample_signal,
        )

        assert result is None
        assert tm._contradictory_state_active is True
        assert "CRITICAL: Emergency close failed" in tm._contradictory_state_reason
        assert tm.is_paused() is True

        events = await store.get_audit_events(limit=10)
        crit_events = [e for e in events if e.get("event_type") == "CRITICAL_CLOSE_FAILED"]
        assert len(crit_events) == 1
        assert "8881" in crit_events[0]["details"]

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


# ==============================================================================
# 3. Strict Risk Calculation & Metadata Rejection in gold_risk.py
# ==============================================================================
def test_risk_engine_rejects_non_positive_contract_size_and_price():
    """
    Verify that non-positive contract_size, point, or price are strictly rejected
    with lot_size=0.0 and no hardcoded 4140.0 or 1.0/100.0 fallbacks.
    """
    risk_engine = GoldRiskEngine(budget_usd=100.0, risk_per_trade_pct=2.0)
    account = AccountInfo(balance=100.0, equity=100.0, margin=0.0, free_margin=100.0, currency="USD", leverage=500)

    # 1. Invalid contract_size (<= 0)
    sym_invalid_contract = SymbolInfo(
        symbol="XAUUSD", bid=2649.75, ask=2650.00, point=0.01, digits=2, spread=0.25,
        contract_size=0.0, # INVALID!
        tick_size=0.01, tick_value=1.0, volume_min=0.01, volume_max=100.0, volume_step=0.01,
    )
    lot, details = risk_engine.calculate_lot_size(sym_invalid_contract, account, sl_distance=2.50)
    assert lot == 0.0
    assert details["has_margin"] is False
    assert "Invalid broker metadata" in details["warning"]

    # 2. Invalid price (ask <= 0) - should NOT fall back to 4140.0
    sym_invalid_ask = SymbolInfo(
        symbol="XAUUSD", bid=0.0, ask=0.0, point=0.01, digits=2, spread=0.25, # INVALID!
        contract_size=100.0,
        tick_size=0.01, tick_value=1.0, volume_min=0.01, volume_max=100.0, volume_step=0.01,
    )
    lot2, details2 = risk_engine.calculate_lot_size(sym_invalid_ask, account, sl_distance=2.50)
    assert lot2 == 0.0
    assert details2["has_margin"] is False
    assert "Invalid broker metadata" in details2["warning"]


def test_risk_engine_monetary_risk_never_exceeds_budget_without_buffer():
    """
    Verify that calculated monetary risk strictly never exceeds the risk budget.
    The previous 1.05 buffer (+5% over risk) has been completely removed.
    """
    account = AccountInfo(balance=100.0, equity=100.0, margin=0.0, free_margin=100.0, currency="USD", leverage=500)
    sym = SymbolInfo(
        symbol="XAUUSD", bid=2649.75, ask=2650.00, point=0.01, digits=2, spread=0.25,
        contract_size=100.0, tick_size=0.01, tick_value=1.0,
        volume_min=0.01, volume_max=100.0, volume_step=0.01,
    )

    # 0.01 lot with 2.50 SL costs $2.50 loss + $0.25 spread = $2.75.
    # $2.75 exceeds budget $2.00. Under previous 1.05 buffer, $2.00 * 1.05 = $2.10 (still exceeded).
    # But even if budget was $2.65, old code would have allowed $2.75 <= $2.65 * 1.05 ($2.78).
    # Now it must strictly be rejected!
    risk_engine_tight = GoldRiskEngine(budget_usd=132.50, risk_per_trade_pct=2.0) # Budget = $2.65
    lot, details = risk_engine_tight.calculate_lot_size(sym, account, sl_distance=2.50)
    assert lot == 0.0, "Must not allow 0.01 lot when min lot risk ($2.75) > budget ($2.65)!"
    assert details["has_margin"] is False
    assert "exceeds allowed risk budget" in details["warning"]


# ==============================================================================
# 4. API & WebSocket Security
# ==============================================================================
def test_cors_origins_are_explicit_not_wildcard():
    """Verify wildcard '*' is removed from CORS and explicit origins are configured."""
    assert "*" not in ALLOWED_ORIGINS
    assert "http://localhost:8000" in ALLOWED_ORIGINS
    assert "http://127.0.0.1:8000" in ALLOWED_ORIGINS


def test_server_context_runtime_defaults_to_scanner():
    """Verify ServerContext defaults to scanner mode on startup."""
    sc = ServerContext()
    assert sc.execution_mode == "scanner"
    assert sc.live_trading_enabled is False


def test_live_bridge_strictly_requires_token_when_live_active():
    """
    When live trading is enabled or active, accessing the bridge endpoint
    without a configured bridge token raises HTTPException(401).
    """
    ctx.live_trading_enabled = True
    ctx.bridge_token = ""
    os.environ.pop("APEX_BRIDGE_TOKEN", None)

    with pytest.raises(HTTPException) as exc_info:
        verify_bridge_token(request=None, payload={})
    assert exc_info.value.status_code == 401
    assert "Live bridge access requires configured bridge token" in exc_info.value.detail

    ctx.live_trading_enabled = False


@pytest.mark.anyio
async def test_websocket_rejects_unauthorized_foreign_origin():
    """Verify WebSocket connection rejects untrusted external origins with close code 1008."""
    class MockWebSocket:
        def __init__(self, origin: str):
            self.headers = {"origin": origin}
            self.closed_code = None
            self.closed_reason = None
            self.accepted = False

        async def accept(self):
            self.accepted = True

        async def close(self, code: int = 1000, reason: str = ""):
            self.closed_code = code
            self.closed_reason = reason

        async def receive_text(self):
            raise WebSocketDisconnect()

    untrusted_ws = MockWebSocket("http://evil-attacker.com")
    await websocket_telemetry(untrusted_ws)
    assert untrusted_ws.closed_code == 1008
    assert untrusted_ws.closed_reason == "Unauthorized origin"
    assert untrusted_ws.accepted is False


# ==============================================================================
# 5. Persistent Candle Provenance in SQLite Across Restart
# ==============================================================================
@pytest.mark.anyio
async def test_candle_provenance_persists_in_sqlite_across_restart():
    """
    Verify that synthetic bars and broker bars retain their is_synthetic and source
    provenance in SQLite after restart, and that exclude_synthetic=True safely isolates them.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        # Phase 1: Initialize DataHub and save a broker bar and a synthetic bar
        hub1 = DataHub(db_path=db_path)
        await hub1.initialize()

        now = int(time.time())
        broker_bar = Bar(
            time=now - 60, open=2650.0, high=2651.0, low=2649.0, close=2650.5,
            volume=100.0, is_synthetic=False, source="BROKER"
        )
        synthetic_bar = Bar(
            time=now, open=2650.5, high=2652.0, low=2650.0, close=2651.5,
            volume=50.0, is_synthetic=True, source="SYNTHETIC"
        )

        await hub1.append_bar("XAUUSD", "1m", broker_bar)
        await hub1.append_bar("XAUUSD", "1m", synthetic_bar)

        # Phase 2: Simulate restart by instantiating a brand-new DataHub on the same SQLite file
        hub2 = DataHub(db_path=db_path)
        await hub2.initialize()

        # Load all bars from database
        all_bars = await hub2.get_bars("XAUUSD", "1m", count=10, exclude_synthetic=False)
        assert len(all_bars) == 2
        # Verify provenance was preserved!
        b_bar = next(b for b in all_bars if b.time == broker_bar.time)
        s_bar = next(b for b in all_bars if b.time == synthetic_bar.time)
        assert b_bar.is_synthetic is False
        assert b_bar.source == "BROKER"
        assert s_bar.is_synthetic is True
        assert s_bar.source == "SYNTHETIC"

        # Exclude synthetic check
        broker_only = await hub2.get_bars("XAUUSD", "1m", count=10, exclude_synthetic=True)
        assert len(broker_only) == 1
        assert broker_only[0].time == broker_bar.time
        assert broker_only[0].is_synthetic is False

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_scalp_mode_configuration():
    """Verify that 'scalp' and 'scalping' transmission modes map to hyper scalping with threshold 52 and dial 92%."""
    from strategy.confluence_scorer import ConfluenceScorer
    scorer = ConfluenceScorer(threshold_balanced=68)

    # 1. Switch to scalp
    scorer.set_mode("scalp")
    assert scorer.mode == "hyper"
    assert scorer.get_active_threshold() == 52
    assert scorer.get_active_aggressiveness_pct() == 92.0

    # 2. Switch to scalping
    scorer.set_mode("scalping")
    assert scorer.mode == "hyper"
    assert scorer.get_active_threshold() == 52
    assert scorer.get_active_aggressiveness_pct() == 92.0


def test_slider_manual_mode_adaptation():
    """Verify that continuous dial adjustments apply manual mode and dynamic thresholds."""
    from strategy.confluence_scorer import ConfluenceScorer
    scorer = ConfluenceScorer(threshold_balanced=68)

    # 1. Dial 85%
    scorer.set_mode("manual", pct=85.0)
    assert scorer.mode == "manual"
    assert scorer.aggressiveness_pct == 85.0
    thresh_85 = scorer.get_active_threshold()
    assert 52 <= thresh_85 <= 60

    # 2. Dial 40%
    scorer.set_mode("manual", pct=40.0)
    assert scorer.mode == "manual"
    assert scorer.aggressiveness_pct == 40.0
    thresh_40 = scorer.get_active_threshold()
    assert 70 <= thresh_40 <= 76


@pytest.mark.anyio
async def test_micro_account_drawdown_protection_and_force_resume():
    """Verify micro-account ($9.18 balance) with positive net daily PnL (+$3.09) doesn't get locked, and force-resume authoritatively unblocks."""
    from unittest.mock import MagicMock, AsyncMock
    crashed_account = AccountInfo(
        balance=9.18,
        equity=9.18,
        margin=0.0,
        free_margin=9.18,
        currency="USD",
        leverage=1000,
    )
    mock_broker = MagicMock()
    mock_broker.get_account_info = AsyncMock(return_value=crashed_account)
    mock_broker.get_open_positions = AsyncMock(return_value=[])
    mock_broker.get_pending_orders = AsyncMock(return_value=[])

    mock_news = MagicMock()
    mock_news.can_open_baseline_trade.return_value = (True, "OK")

    tm = TradeManager(
        broker=mock_broker,
        data_hub=MagicMock(),
        news_reactor=mock_news,
        risk_engine=MagicMock(),
        max_daily_drawdown_pct=25.0,
    )
    # Scenario: Account started at $8.29, made +$5.28 profit, took two small invalidation losses (-$4.33).
    # Net daily realized PnL is +$3.09 (account is profitable!).
    tm._start_of_day_balance = 8.29
    tm._daily_realized_pnl = 3.09

    can_open, reason = await tm.can_open_manual_trade("XAUUSD", "BUY", 0.01)
    assert can_open is True
    assert reason == "OK"

    # Simulate pause and verify operator force resume
    tm._pause_until = time.time() + 3600
    assert tm.is_paused() is True

    resumed, msg = await tm.resume_trading(force=True)
    assert resumed is True
    assert "Trading resumed successfully" in msg
    assert tm.is_paused() is False
    assert tm._start_of_day_balance == 9.18


@pytest.mark.anyio
async def test_anti_hedging_mutual_exclusion():
    """Verify that opposite BUY and SELL trades on Gold cannot be opened simultaneously (Anti-Hedging Guard)."""
    from unittest.mock import MagicMock, AsyncMock
    open_sell = Position(
        ticket=12345,
        symbol="GOLD",
        type=PositionType.SELL,
        volume=0.01,
        open_price=4133.0,
        sl=4136.0,
        tp=4127.0,
        current_price=4132.5,
        profit=0.50,
        open_time=time.time(),
    )
    mock_broker = MagicMock()
    mock_broker.get_account_info = AsyncMock(return_value=AccountInfo(balance=100.0, equity=100.0, margin=0.0, free_margin=100.0, currency="USD", leverage=1000))
    mock_broker.get_open_positions = AsyncMock(return_value=[open_sell])
    mock_broker.get_pending_orders = AsyncMock(return_value=[])

    mock_news = MagicMock()
    mock_news.can_open_baseline_trade.return_value = (True, "OK")

    tm = TradeManager(
        broker=mock_broker,
        data_hub=MagicMock(),
        news_reactor=mock_news,
        risk_engine=MagicMock(),
    )
    tm._start_of_day_balance = 100.0

    # 1. Attempt manual BUY while SELL is active -> MUST BE REJECTED
    can_open, reason = await tm.can_open_manual_trade("GOLD", "BUY", 0.01)
    assert can_open is False
    assert "Anti-Hedging Guard" in reason
    assert "SELL position #12345 already active" in reason

    # 2. Attempt automated BUY signal while SELL is active -> MUST BE REJECTED
    buy_signal = ConfluenceSignal(
        symbol="GOLD",
        direction="BUY",
        score=85,
        tier="TIER_1_SNIPER",
        regime="TRENDING_EXPANSION",
        sub_regime="FAST_SCALP",
        target_rr=2.0,
        entry_price=4132.0,
        suggested_sl=4130.0,
        suggested_tp=4136.0,
        risk_r_dist=2.0,
        be_lock_r=0.65,
        stall_exit_seconds=60.0,
        breakdown={},
        reason="Test Buy",
        time=time.time(),
        order_type="MARKET",
    )
    can_exec, exec_reason = await tm.can_execute_signal(buy_signal)
    assert can_exec is False
    assert "Anti-Hedging Guard" in exec_reason


@pytest.mark.anyio
async def test_predictive_breakeven_scratch_on_reversal():
    """Verify that a trade in slight profit (+0.10R to +0.25R) scratches at breakeven when market turns adverse."""
    from unittest.mock import MagicMock, AsyncMock
    import pandas as pd
    from execution.trade_manager import ManagedPosition

    now = time.time()
    pos = ManagedPosition(
        ticket=8881,
        symbol="XAUUSD",
        direction="BUY",
        volume=0.01,
        open_price=2700.00,
        sl=2697.50,
        tp=2706.00,
        entry_time=now - 100.0, # 100s hold (>= 90s min hold)
        risk_r_dist=2.50,
        highest_r_reached=0.45, # reached substantial profit
        profit=0.20,
    )

    # 1m and 5m Bearish DataFrames
    bear_prices = [2705.0 - i * 0.3 for i in range(30)]
    df_bear = pd.DataFrame({
        "open": [p + 0.1 for p in bear_prices],
        "high": [p + 0.3 for p in bear_prices],
        "low": [p - 0.3 for p in bear_prices],
        "close": [p - 0.1 for p in bear_prices],
        "volume": [100.0] * 30,
    })

    mock_hub = MagicMock()
    mock_hub.get_dataframe = AsyncMock(return_value=df_bear)

    mock_broker = MagicMock()
    mock_broker.get_account_info = AsyncMock(return_value=AccountInfo(balance=10.0, equity=10.25, margin=0.0, free_margin=10.25, currency="USD", leverage=1000))

    tm = TradeManager(
        broker=mock_broker,
        data_hub=mock_hub,
        news_reactor=MagicMock(_latest_velocity=1.0),
        risk_engine=MagicMock(base_budget_usd=10.0),
        predictive_scratch_enabled=True,
        min_hold_time_seconds=90.0,
    )

    should_exit, reason = await tm._check_market_flip(
        pos=pos,
        floating_r=0.08,
        current_price=2700.20,
        now=now,
    )

    assert should_exit is True
    assert "Predictive Invalidation: Breakeven Scratch" in reason
    assert "Trend Bearish" in reason or "SuperTrend" in reason


@pytest.mark.anyio
async def test_trade_waits_it_out_and_does_not_close_early_on_noise():
    """Verify that trades wait it out (>= 90s hold time) and NEVER open and close directly on noise."""
    from unittest.mock import MagicMock, AsyncMock
    import pandas as pd
    from execution.trade_manager import ManagedPosition

    now = time.time()
    # Trade open for only 15 seconds
    pos = ManagedPosition(
        ticket=8882,
        symbol="XAUUSD",
        direction="BUY",
        volume=0.01,
        open_price=2700.00,
        sl=2697.50,
        tp=2706.00,
        entry_time=now - 15.0, # 15s hold (< 90s)
        risk_r_dist=2.50,
        highest_r_reached=0.0,
        profit=-0.50, # floating loss
    )

    bear_prices = [2702.0 - i * 0.25 for i in range(30)]
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
    mock_broker.get_account_info = AsyncMock(return_value=AccountInfo(balance=5.60, equity=5.10, margin=0.0, free_margin=5.10, currency="USD", leverage=1000))

    tm = TradeManager(
        broker=mock_broker,
        data_hub=mock_hub,
        news_reactor=MagicMock(_latest_velocity=1.0),
        risk_engine=MagicMock(base_budget_usd=5.60),
        predictive_scratch_enabled=True,
        min_hold_time_seconds=90.0,
    )

    should_exit, reason = await tm._check_market_flip(
        pos=pos,
        floating_r=-0.25,
        current_price=2699.37,
        now=now,
    )

    # Must wait it out! NEVER close directly
    assert should_exit is False
    assert reason == ""


@pytest.mark.anyio
async def test_high_certainty_deep_invalidation_closes_when_sure_to_lose():
    """Verify that deep loss (-0.75R) after >= 90s with confirmed 5m SuperTrend break closes early to save SL."""
    from unittest.mock import MagicMock, AsyncMock
    import pandas as pd
    from execution.trade_manager import ManagedPosition

    now = time.time()
    pos = ManagedPosition(
        ticket=8884,
        symbol="XAUUSD",
        direction="BUY",
        volume=0.01,
        open_price=2700.00,
        sl=2697.50,
        tp=2706.00,
        entry_time=now - 120.0, # 120s hold (>= 90s)
        risk_r_dist=2.50,
        highest_r_reached=0.0,
        profit=-1.87,
    )

    bear_prices = [2702.0 - i * 0.35 for i in range(30)]
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
    mock_broker.get_account_info = AsyncMock(return_value=AccountInfo(balance=5.60, equity=3.73, margin=0.0, free_margin=3.73, currency="USD", leverage=1000))

    tm = TradeManager(
        broker=mock_broker,
        data_hub=mock_hub,
        news_reactor=MagicMock(_latest_velocity=1.0),
        risk_engine=MagicMock(base_budget_usd=5.60),
        predictive_scratch_enabled=True,
        min_hold_time_seconds=90.0,
    )

    should_exit, reason = await tm._check_market_flip(
        pos=pos,
        floating_r=-0.75,
        current_price=2698.12,
        now=now,
    )

    assert should_exit is True
    assert "Confirmed Invalidation: 5m SuperTrend Bearish Break" in reason


@pytest.mark.anyio
async def test_predictive_does_not_cut_healthy_runners():
    """Verify that healthy winning trades are not cut by predictive invalidation and continue running."""
    from unittest.mock import MagicMock, AsyncMock
    import pandas as pd
    from execution.trade_manager import ManagedPosition

    now = time.time()
    pos = ManagedPosition(
        ticket=8883,
        symbol="XAUUSD",
        direction="BUY",
        volume=0.01,
        open_price=2700.00,
        sl=2697.50,
        tp=2706.00,
        entry_time=now - 120.0,
        risk_r_dist=2.50,
        highest_r_reached=0.85,
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
    mock_broker.get_account_info = AsyncMock(return_value=AccountInfo(balance=25.0, equity=27.0, margin=0.0, free_margin=27.0, currency="USD", leverage=1000))

    tm = TradeManager(
        broker=mock_broker,
        data_hub=mock_hub,
        news_reactor=MagicMock(_latest_velocity=1.0),
        risk_engine=MagicMock(base_budget_usd=25.0),
        predictive_scratch_enabled=True,
    )

    should_exit, reason = await tm._check_market_flip(
        pos=pos,
        floating_r=0.80,
        current_price=2702.00,
        now=now,
    )

    assert should_exit is False
    assert reason == ""


@pytest.mark.anyio
async def test_signals_execute_immediately_at_market(sample_signal):
    """Verify that bot signals execute immediately at MARKET (OrderType.BUY/SELL) rather than waiting on limits."""
    from unittest.mock import MagicMock, AsyncMock

    mock_broker = MagicMock()
    mock_broker.get_account_info = AsyncMock(return_value=AccountInfo(balance=5.60, equity=5.60, margin=0.0, free_margin=5.60, currency="USD", leverage=1000))
    mock_broker.get_open_positions = AsyncMock(return_value=[])
    mock_broker.get_pending_orders = AsyncMock(return_value=[])
    mock_broker.get_symbol_info = AsyncMock(return_value=SymbolInfo(
        symbol="GOLD", bid=4153.20, ask=4153.40, point=0.01, digits=2, spread=0.20,
        contract_size=100.0, tick_size=0.01, tick_value=1.0, volume_min=0.01, volume_max=100.0, volume_step=0.01
    ))
    mock_broker.open_order = AsyncMock(return_value=OrderResult(success=True, ticket=123456))

    mock_news = MagicMock()
    mock_news.can_open_baseline_trade.return_value = (True, "OK")

    risk_engine = GoldRiskEngine(budget_usd=5.60, account_type="micro", risk_profile="aggressive")

    tm = TradeManager(
        broker=mock_broker,
        data_hub=MagicMock(),
        news_reactor=mock_news,
        risk_engine=risk_engine,
    )

    # Even if signal has an order_type or limit_price, execute_trade must dispatch MARKET OrderType.BUY
    sample_signal.symbol = "GOLD"
    sample_signal.direction = "BUY"
    sample_signal.entry_price = 4153.40
    sample_signal.limit_price = 4153.26
    sample_signal.suggested_sl = 4151.90
    sample_signal.suggested_tp = 4156.40
    sample_signal.order_type = "BUY_LIMIT"

    res = await tm.execute_trade(sample_signal)
    assert res.success is True

    # Assert that open_order was called with OrderType.BUY (MARKET execution) and NOT OrderType.BUY_LIMIT
    mock_broker.open_order.assert_called_once()
    call_kwargs = mock_broker.open_order.call_args[1]
    assert call_kwargs["order_type"] == OrderType.BUY
    assert call_kwargs["price"] == 4153.40 # Market entry price, never waiting for 4153.26 pullback!


@pytest.mark.anyio
async def test_margin_feasibility_on_micro_account():
    """Verify that GoldRiskEngine accounts for margin correctly on micro balance with 1:1000 leverage."""
    sym = SymbolInfo(
        symbol="XAUUSD", bid=4153.20, ask=4153.40, point=0.01, digits=2, spread=0.20,
        contract_size=100.0, tick_size=0.01, tick_value=1.0, volume_min=0.01, volume_max=100.0, volume_step=0.01
    )
    # Case A: Free margin = $5.60, leverage = 1000. 0.01 lot requires (4153.40 * 0.01 * 100) / 1000 = $4.15.
    # $5.60 >= $4.15 -> has_margin must be True!
    acc_sufficient = AccountInfo(balance=5.60, equity=5.60, margin=0.0, free_margin=5.60, currency="USD", leverage=1000)
    engine = GoldRiskEngine(budget_usd=5.60, account_type="micro", risk_profile="aggressive")
    lot, details = engine.calculate_lot_size(sym, acc_sufficient, sl_distance=1.50)
    assert lot == 0.01
    assert details["has_margin"] is True
    assert details["margin_required"] == 4.15

    # Case B: Free margin = $2.50, leverage = 1000. Required $4.15 > $2.50.
    # Must report has_margin = False with explicit shortfall!
    acc_insufficient = AccountInfo(balance=2.50, equity=2.50, margin=0.0, free_margin=2.50, currency="USD", leverage=1000)
    lot_fail, details_fail = engine.calculate_lot_size(sym, acc_insufficient, sl_distance=1.50)
    assert lot_fail == 0.0
    assert details_fail["has_margin"] is False
    assert "Short by $1.65" in details_fail["warning"]



