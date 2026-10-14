"""
Automated Regression and Safety Test Suite for Phase 1.

Verifies:
1. Trade execution registers ManagedPosition without NameError (eff_bal fix).
2. Paper simulator is completely isolated from live MT5 execution queue.
3. Risk engine respects risk_per_trade_pct, prevents minimum-lot overexposure, and calculates monetary risk correctly.
4. Circuit breakers enforce daily drawdown, consecutive losses, open risk limits, and safe state reconstruction.
5. Order handling correctly maps pending orders and strictly prohibits fallback to unrelated positions.
"""

import asyncio
import os
import sys
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from brokers.base import (
    AccountInfo,
    Bar,
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
from strategy.confluence_scorer import ConfluenceSignal


# --------------------------------------------------------------------------
# Fixtures & Helpers
# --------------------------------------------------------------------------

@pytest.fixture
def sample_symbol_info():
    return SymbolInfo(
        symbol="XAUUSD",
        bid=2650.00,
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


@pytest.fixture
def sample_account_info():
    return AccountInfo(
        balance=100.0,
        equity=100.0,
        margin=0.0,
        free_margin=100.0,
        currency="USD",
        leverage=500,
        account_type="micro",
    )


@pytest.fixture
def sample_signal():
    return ConfluenceSignal(
        symbol="XAUUSD",
        direction="BUY",
        score=85,
        tier="TIER_1_SNIPER",
        regime="TRENDING_EXPANSION",
        sub_regime="BULL_MOMENTUM",
        target_rr=2.0,
        entry_price=2650.00,
        suggested_sl=2647.50,
        suggested_tp=2655.00,
        risk_r_dist=2.50,
        be_lock_r=1.0,
        stall_exit_seconds=120.0,
        breakdown={"M1": 25, "M5": 30, "M15": 30},
        reason="Sniper Confluence Alignment",
        time=time.time(),
        order_type="MARKET",
    )


# --------------------------------------------------------------------------
# 1. Regression Test: eff_bal NameError and ManagedPosition Registration
# --------------------------------------------------------------------------

def test_execute_trade_registers_position_without_name_error(
    sample_symbol_info, sample_account_info, sample_signal
):
    """
    Test that execute_trade() executes successfully, does not crash on 'eff_bal',
    and reliably enrolls the position into _managed_positions.
    """
    async def _run():
        mock_broker = MagicMock()
        account_200 = AccountInfo(
            balance=200.0,
            equity=200.0,
            margin=0.0,
            free_margin=200.0,
            currency="USD",
            leverage=500,
            account_type="micro",
        )
        mock_broker.get_account_info = AsyncMock(return_value=account_200)
        mock_broker.get_symbol_info = AsyncMock(return_value=sample_symbol_info)
        mock_broker.get_open_positions = AsyncMock(return_value=[])
        mock_broker.get_pending_orders = AsyncMock(return_value=[])
        mock_broker.get_tick = AsyncMock(return_value=Tick(symbol="XAUUSD", bid=2650.0, ask=2650.25))
        mock_broker.open_order = AsyncMock(
            return_value=OrderResult(
                success=True,
                ticket=999888,
                price=2650.25,
                volume=0.01,
                error=None,
            )
        )

        mock_news = MagicMock()
        mock_news.can_open_baseline_trade = MagicMock(return_value=(True, "OK"))
        mock_news._latest_velocity = 1.0

        risk_engine = GoldRiskEngine(budget_usd=200.0, risk_per_trade_pct=3.0)
        tm = TradeManager(
            broker=mock_broker,
            data_hub=MagicMock(),
            news_reactor=mock_news,
            risk_engine=risk_engine,
            max_daily_losses=4,
            max_daily_drawdown_pct=25.0,
        )

        result = await tm.execute_trade(sample_signal)

        assert result.success is True
        assert result.ticket == 999888
        # Ensure position is enrolled in position management
        assert 999888 in tm._managed_positions
        pos = tm._managed_positions[999888]
        assert pos.symbol == "XAUUSD"
        assert pos.ticket == 999888
        assert pos.volume == 0.01
        assert pos.be_lock_r > 0.0

    asyncio.run(_run())


# --------------------------------------------------------------------------
# 2. Complete Separation of Paper Trading and Live Trading
# --------------------------------------------------------------------------

def test_paper_simulator_has_no_mt5_bridge_queue():
    """Verify PaperSimAdapter contains no MT5 queue references and cannot forward trades."""
    async def _run():
        sim = PaperSimAdapter(initial_balance=100.0)
        await sim.connect()

        assert not hasattr(sim, "_mt5_queue")
        assert getattr(sim, "is_paper", False) is True

        # Open a simulated order
        res = await sim.open_order(
            symbol="XAUUSD",
            order_type=OrderType.BUY,
            volume=0.01,
            price=2650.00,
            sl=2647.50,
            tp=2655.00,
        )
        assert res.success is True
        assert res.ticket in sim._positions

        # Modify position
        mod_ok = await sim.modify_position(res.ticket, sl=2648.00, tp=2656.00)
        assert mod_ok is True
        assert sim._positions[res.ticket].sl == 2648.00

        # Close position
        close_ok = await sim.close_position(res.ticket)
        assert close_ok is True
        assert res.ticket not in sim._positions

        await sim.disconnect()

    asyncio.run(_run())


def test_server_blocks_paper_trades_from_entering_mt5_bridge():
    """Verify server bridge tick endpoint strictly returns NONE if broker is paper mode or live trading disabled."""
    async def _run():
        from ui.server import ctx, mt5_bridge_tick

        ctx.live_trading_enabled = False
        ctx.broker = PaperSimAdapter(initial_balance=100.0)
        ctx.mt5_pending_actions = [{"action": "ORDER_SEND", "symbol": "XAUUSD", "volume": 0.01}]

        # Tick poll from MT5
        action = await mt5_bridge_tick(payload={})
        assert action == {"action": "NONE"}
        # Verify unauthorized queue was purged
        assert len(ctx.mt5_pending_actions) == 0

    asyncio.run(_run())


# --------------------------------------------------------------------------
# 3. Position Sizing & Strict Risk Limits
# --------------------------------------------------------------------------

def test_risk_engine_rejects_trade_when_min_lot_exceeds_risk(sample_symbol_info):
    """
    On a $10 micro account with 2% risk ($0.20 budget), 0.01 lot of XAUUSD
    with 2.50 SL risks $2.50 (25% of account!).
    The engine must REJECT the trade rather than forcing 0.01 lot.
    """
    small_account = AccountInfo(
        balance=10.0,
        equity=10.0,
        margin=0.0,
        free_margin=10.0,
        currency="USD",
        leverage=500,
        account_type="micro",
    )
    risk_engine = GoldRiskEngine(budget_usd=10.0, risk_per_trade_pct=2.0)

    lot, details = risk_engine.calculate_lot_size(
        symbol_info=sample_symbol_info,
        account_info=small_account,
        sl_distance=2.50,
    )

    assert lot == 0.0
    assert details["has_margin"] is False
    assert "exceeds allowed risk budget" in details["warning"]


def test_risk_engine_sizes_correctly_when_budget_sufficient(sample_symbol_info):
    """
    On a $500 balance with 2% risk ($10.00 budget) and 2.50 SL distance:
    1 lot costs $250 + spread.
    Allowed lot should be approx 0.03 lots ($7.50 risk).
    """
    account = AccountInfo(
        balance=500.0,
        equity=500.0,
        margin=0.0,
        free_margin=500.0,
        currency="USD",
        leverage=500,
        account_type="micro",
    )
    risk_engine = GoldRiskEngine(budget_usd=500.0, risk_per_trade_pct=2.0)

    lot, details = risk_engine.calculate_lot_size(
        symbol_info=sample_symbol_info,
        account_info=account,
        sl_distance=2.50,
    )

    assert lot > 0.0
    assert details["has_margin"] is True
    # The monetary risk must be <= risk_amount * 1.05
    assert details["actual_monetary_risk"] <= details["risk_amount"] * 1.05


def test_effective_budget_never_exceeds_actual_balance():
    """Budget must never exceed actual balance even if base_budget_usd is higher."""
    risk_engine = GoldRiskEngine(budget_usd=100.0, compounding_enabled=True)
    eff = risk_engine.get_effective_budget(current_balance=25.0)
    assert eff == 25.0

    risk_engine_no_compound = GoldRiskEngine(budget_usd=100.0, compounding_enabled=False)
    eff_nc = risk_engine_no_compound.get_effective_budget(current_balance=25.0)
    assert eff_nc == 25.0


# --------------------------------------------------------------------------
# 4. Circuit Breakers Enforcement
# --------------------------------------------------------------------------

def test_circuit_breaker_blocks_trades_on_daily_drawdown(sample_signal):
    """Verify max_daily_drawdown_pct halts new trade executions."""
    async def _run():
        # Account started at $100, current equity dropped to $70 (30% drawdown)
        crashed_account = AccountInfo(
            balance=70.0,
            equity=70.0,
            margin=0.0,
            free_margin=70.0,
            currency="USD",
            leverage=500,
        )
        mock_broker = MagicMock()
        mock_broker.get_account_info = AsyncMock(return_value=crashed_account)

        tm = TradeManager(
            broker=mock_broker,
            data_hub=MagicMock(),
            news_reactor=MagicMock(),
            risk_engine=MagicMock(),
            max_daily_drawdown_pct=25.0, # 25% max
        )
        tm._start_of_day_balance = 100.0 # Started day with $100

        can_trade, reason = await tm.can_execute_signal(sample_signal)
        assert can_trade is False
        assert "Daily drawdown breached" in reason
        assert tm.is_paused() is True

        # Attempting to resume should be blocked while drawdown is still active
        resumed, msg = await tm.resume_trading()
        assert resumed is False
        assert "Daily drawdown" in msg

    asyncio.run(_run())


def test_circuit_breaker_blocks_trades_on_consecutive_losses():
    """Verify max_consecutive_losses halts trading."""
    mock_broker = MagicMock()
    tm = TradeManager(
        broker=mock_broker,
        data_hub=MagicMock(),
        news_reactor=MagicMock(),
        risk_engine=MagicMock(),
        max_consecutive_losses=3,
    )
    tm._consecutive_losses = 3
    assert tm.is_paused() is True


def test_circuit_breaker_blocks_excessive_open_risk(sample_signal, sample_symbol_info):
    """Verify max_total_open_risk_pct rejects trade if open positions already consume risk."""
    async def _run():
        acc = AccountInfo(balance=400.0, equity=400.0, free_margin=400.0, leverage=500)
        mock_broker = MagicMock()
        mock_broker.get_account_info = AsyncMock(return_value=acc)
        mock_broker.get_symbol_info = AsyncMock(return_value=sample_symbol_info)
        mock_broker.get_open_positions = AsyncMock(return_value=[])
        mock_broker.get_pending_orders = AsyncMock(return_value=[])
        mock_broker.get_tick = AsyncMock(return_value=Tick(symbol="XAUUSD", bid=2650.0, ask=2650.25))

        news = MagicMock()
        news.can_open_baseline_trade = MagicMock(return_value=(True, "OK"))

        risk_engine = GoldRiskEngine(budget_usd=400.0, risk_per_trade_pct=2.0)
        tm = TradeManager(
            broker=mock_broker,
            data_hub=MagicMock(),
            news_reactor=news,
            risk_engine=risk_engine,
            max_total_open_risk_pct=2.0,
        )

        # Inject an existing open managed position risking $7.00 (already winning in profit to satisfy pyramiding)
        tm._managed_positions[111] = ManagedPosition(
            ticket=111,
            symbol="XAUUSD",
            direction="BUY",
            volume=0.02,
            open_price=2645.0, # in solid profit relative to cur_tick 2650.0 (+2.0R)
            sl=2646.00,
            tp=2665.00,
            entry_time=time.time(),
            risk_r_dist=2.50,
            initial_regime="TRENDING_EXPANSION",
            current_regime="TRENDING_EXPANSION",
        )

        # Attempting another trade that pushes total risk > 5.0%
        can_trade, reason = await tm.can_execute_signal(sample_signal)
        assert can_trade is False
        assert "open risk limit reached" in reason

    asyncio.run(_run())


# --------------------------------------------------------------------------
# 5. Broker Order Handling & Verification
# --------------------------------------------------------------------------

def test_windows_mt5_adapter_pending_order_mapping():
    """Verify Windows MT5 adapter maps BUY_LIMIT to TRADE_ACTION_PENDING."""
    async def _run():
        from brokers.mt5_windows import WindowsMT5Adapter

        adapter = WindowsMT5Adapter()
        adapter._connected = True

        sent_requests = []

        def mock_order_send(req):
            sent_requests.append(req)
            res = MagicMock()
            res.retcode = 10008 # TRADE_RETCODE_PLACED
            res.order = 777666
            res.deal = 0
            res.price = req["price"]
            res.volume = req["volume"]
            res.comment = "Done"
            return res

        mock_mt5 = MagicMock()
        mock_mt5.order_send = mock_order_send
        mock_mt5.TRADE_ACTION_PENDING = 5
        mock_mt5.ORDER_TYPE_BUY_LIMIT = 2
        mock_mt5.ORDER_FILLING_RETURN = 2
        mock_mt5.ORDER_TIME_GTC = 0
        mock_mt5.TRADE_RETCODE_DONE = 10009
        mock_mt5.TRADE_RETCODE_PLACED = 10008

        with patch("brokers.mt5_windows.mt5", mock_mt5):
            adapter.get_tick = AsyncMock(return_value=Tick(symbol="XAUUSD", bid=2650.0, ask=2650.25))
            res = await adapter.open_order(
                symbol="XAUUSD",
                order_type=OrderType.BUY_LIMIT,
                volume=0.02,
                price=2645.00,
                sl=2640.00,
                tp=2660.00,
            )

            assert res.success is True
            assert res.ticket == 777666
            assert len(sent_requests) == 1
            req = sent_requests[0]
            assert req["action"] == mock_mt5.TRADE_ACTION_PENDING
            assert req["type"] == mock_mt5.ORDER_TYPE_BUY_LIMIT
            assert req["price"] == 2645.00

    asyncio.run(_run())


def test_windows_mt5_adapter_sell_limit_mapping():
    """Verify Windows MT5 adapter maps SELL_LIMIT to TRADE_ACTION_PENDING and ORDER_TYPE_SELL_LIMIT."""
    async def _run():
        from brokers.mt5_windows import WindowsMT5Adapter

        adapter = WindowsMT5Adapter()
        adapter._connected = True

        sent_requests = []

        def mock_order_send(req):
            sent_requests.append(req)
            res = MagicMock()
            res.retcode = 10008
            res.order = 888999
            res.deal = 0
            res.price = req["price"]
            res.volume = req["volume"]
            res.comment = "Done"
            return res

        mock_mt5 = MagicMock()
        mock_mt5.order_send = mock_order_send
        mock_mt5.TRADE_ACTION_PENDING = 5
        mock_mt5.ORDER_TYPE_SELL_LIMIT = 3
        mock_mt5.ORDER_FILLING_RETURN = 2
        mock_mt5.ORDER_TIME_GTC = 0
        mock_mt5.TRADE_RETCODE_DONE = 10009
        mock_mt5.TRADE_RETCODE_PLACED = 10008

        with patch("brokers.mt5_windows.mt5", mock_mt5):
            adapter.get_tick = AsyncMock(return_value=Tick(symbol="XAUUSD", bid=2650.0, ask=2650.25))
            res = await adapter.open_order(
                symbol="XAUUSD",
                order_type=OrderType.SELL_LIMIT,
                volume=0.01,
                price=2655.00,
                sl=2660.00,
                tp=2640.00,
            )

            assert res.success is True
            assert res.ticket == 888999
            assert len(sent_requests) == 1
            req = sent_requests[0]
            assert req["action"] == mock_mt5.TRADE_ACTION_PENDING
            assert req["type"] == mock_mt5.ORDER_TYPE_SELL_LIMIT
            assert req["price"] == 2655.00

    asyncio.run(_run())


def test_windows_mt5_adapter_refuses_to_modify_missing_position():
    """Verify modify_position fails safely and never modifies unrelated positions when ticket does not exist."""
    async def _run():
        from brokers.mt5_windows import WindowsMT5Adapter

        adapter = WindowsMT5Adapter()
        adapter._connected = True

        mock_mt5 = MagicMock()
        mock_mt5.positions_get = MagicMock(return_value=[]) # ticket not found!

        with patch("brokers.mt5_windows.mt5", mock_mt5):
            ok = await adapter.modify_position(ticket=999999, sl=2640.0, tp=2660.0)
            assert ok is False
            # Verify order_send was never called!
            mock_mt5.order_send.assert_not_called()

    asyncio.run(_run())


def test_risk_state_reconstruction_from_history():
    """Verify TradeManager reconstructs daily losses, daily realized pnl, and consecutive losses upon start."""
    async def _run():
        now = time.time()
        mock_history = MagicMock()
        mock_history.get_history = AsyncMock(return_value=[
            {"ticket": 1, "close_time": now - 3600, "pnl": 12.50},
            {"ticket": 2, "close_time": now - 1800, "pnl": -3.20},
            {"ticket": 3, "close_time": now - 900, "pnl": -2.80},
        ])

        mock_broker = MagicMock()
        mock_broker.get_account_info = AsyncMock(return_value=AccountInfo(
            balance=106.50, equity=106.50, free_margin=106.50, leverage=500
        ))

        tm = TradeManager(
            broker=mock_broker,
            data_hub=MagicMock(),
            news_reactor=MagicMock(),
            risk_engine=MagicMock(),
            history_store=mock_history,
        )

        await tm.sync_daily_risk_state()

        # Net P&L: 12.50 - 3.20 - 2.80 = 6.50
        assert round(tm._daily_realized_pnl, 2) == 6.50
        # 2 losing trades
        assert tm._daily_losses_count == 2
        # 2 consecutive losses at the end
        assert tm._consecutive_losses == 2
        # Start of day balance: 106.50 - 6.50 = 100.00
        assert round(tm._start_of_day_balance, 2) == 100.00

    asyncio.run(_run())


def test_cent_account_monetary_risk_calculation():
    """Verify cent account calculations handle USC units correctly."""
    cent_sym = SymbolInfo(
        symbol="XAUUSDc",
        bid=2650.00,
        ask=2650.25,
        point=0.01,
        digits=2,
        spread=0.25,
        contract_size=1.0, # 1.0 oz on Cent account
        tick_size=0.01,
        tick_value=0.01,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
    )
    risk_engine = GoldRiskEngine(account_type="cent", budget_usd=1000.0, risk_per_trade_pct=2.0)
    # Monetary risk for 0.10 lots with 3.00 SL: 0.10 * 300 ticks * 0.01 tick_value = 0.30 USC
    risk = risk_engine.calculate_monetary_risk(cent_sym, lot=0.10, sl_distance=3.00)
    assert risk > 0.0
    assert risk < 1.0

