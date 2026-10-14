"""
Regression & Unit Tests for Persistent Trade History Cleanup & Purge Guard.

Ensures:
- Historical closed MT5 trades never resurrect after Reset History or Delete Trade.
- Accounts for MT5 broker clock differences (e.g. UTC+2/3 ahead of local UTC).
- Permanently records purged tickets in SQLite so broker ticks cannot re-ingest them.
- Single trade deletion endpoint permanently removes the trade and recalibrates win rate.
"""

import asyncio
import os
import sqlite3
import sys
import tempfile
import time
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.trade_history_store import TradeHistoryStore
from ui.server import (
    DeleteTradePayload,
    action_clear_history,
    action_delete_trade,
    ctx,
)


@pytest.mark.anyio
async def test_clear_history_with_broker_clock_purge():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        store = TradeHistoryStore(db_path=db_path)
        await store.initialize()

        # Insert a closed deal with MT5 broker time ahead of local time
        broker_close_time = 1792010148.0
        ticket = 992702093
        await store.record_trade(
            ticket=ticket,
            symbol="GOLD",
            direction="BUY",
            volume=0.06,
            open_price=4088.25,
            close_price=4084.32,
            open_time=broker_close_time - 300,
            close_time=broker_close_time,
            pnl=-23.58,
            pnl_r=-1.96,
            regime="MT5_LIVE",
            sub_regime="REAL_DEAL",
            close_reason="MT5 Realized Loss (-$23.58)",
        )

        history = await store.get_history()
        assert len(history) == 1
        assert history[0]["ticket"] == ticket

        # User clicks [Reset History] with broker time captured from latest tick
        stats = await store.clear_history(broker_time=broker_close_time)
        assert stats["total_trades"] == 0
        assert stats["win_rate_pct"] == 0.0

        # Verify history table is empty
        history_after = await store.get_history()
        assert len(history_after) == 0

        # Verify ticket is in purged set
        assert ticket in store._purged_tickets
        assert store._cleared_cutoff_time >= broker_close_time

        # Attempt to re-ingest the exact same deal from MT5 closed_deals tick
        await store.record_trade(
            ticket=ticket,
            symbol="GOLD",
            direction="BUY",
            volume=0.06,
            open_price=4088.25,
            close_price=4084.32,
            open_time=broker_close_time - 300,
            close_time=broker_close_time,
            pnl=-23.58,
            pnl_r=-1.96,
            regime="MT5_LIVE",
            sub_regime="REAL_DEAL",
            close_reason="MT5 Realized Loss (-$23.58)",
        )

        # Must still be empty - ticket is purged and close_time <= cutoff
        history_recheck = await store.get_history()
        assert len(history_recheck) == 0

        # Also attempt another past deal with a different ticket but closed before cutoff
        past_ticket = 992702090
        await store.record_trade(
            ticket=past_ticket,
            symbol="GOLD",
            direction="SELL",
            volume=0.02,
            open_price=4085.00,
            close_price=4083.00,
            open_time=broker_close_time - 500,
            close_time=broker_close_time - 100,
            pnl=4.00,
            pnl_r=1.0,
            regime="MT5_LIVE",
            sub_regime="REAL_DEAL",
            close_reason="Take Profit",
        )
        assert len(await store.get_history()) == 0

        # A brand new deal closed AFTER the reset MUST be recorded normally
        new_ticket = 992702099
        new_close_time = store._cleared_cutoff_time + 10.0
        await store.record_trade(
            ticket=new_ticket,
            symbol="GOLD",
            direction="BUY",
            volume=0.01,
            open_price=4090.00,
            close_price=4095.00,
            open_time=new_close_time - 60,
            close_time=new_close_time,
            pnl=5.00,
            pnl_r=2.5,
            regime="MT5_LIVE",
            sub_regime="REAL_DEAL",
            close_reason="Take Profit",
        )
        fresh_history = await store.get_history()
        assert len(fresh_history) == 1
        assert fresh_history[0]["ticket"] == new_ticket

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


@pytest.mark.anyio
async def test_single_trade_deletion():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        store = TradeHistoryStore(db_path=db_path)
        await store.initialize()

        now = time.time()
        # Add two trades
        await store.record_trade(
            ticket=1001,
            symbol="GOLD",
            direction="BUY",
            volume=0.01,
            open_price=4080.0,
            close_price=4082.0,
            open_time=now - 200,
            close_time=now - 100,
            pnl=2.00,
            pnl_r=1.0,
            regime="BOT",
            sub_regime="SCALPING",
            close_reason="Take Profit",
        )
        await store.record_trade(
            ticket=1002,
            symbol="GOLD",
            direction="SELL",
            volume=0.02,
            open_price=4085.0,
            close_price=4080.0,
            open_time=now - 150,
            close_time=now - 50,
            pnl=10.00,
            pnl_r=2.5,
            regime="BOT",
            sub_regime="SCALPING",
            close_reason="Take Profit",
        )

        assert len(await store.get_history()) == 2

        # Delete ticket 1001
        res = await store.delete_trade(1001)
        assert res is True
        assert 1001 in store._purged_tickets

        # Check remaining
        history = await store.get_history()
        assert len(history) == 1
        assert history[0]["ticket"] == 1002

        # Re-adding ticket 1001 must be ignored
        await store.record_trade(
            ticket=1001,
            symbol="GOLD",
            direction="BUY",
            volume=0.01,
            open_price=4080.0,
            close_price=4082.0,
            open_time=now - 200,
            close_time=now - 100,
            pnl=2.00,
            pnl_r=1.0,
            regime="BOT",
            sub_regime="SCALPING",
            close_reason="Take Profit",
        )
        assert len(await store.get_history()) == 1

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


@pytest.mark.anyio
async def test_purged_tickets_persistence_across_restarts():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        # Session 1: Purge a ticket
        store1 = TradeHistoryStore(db_path=db_path)
        await store1.initialize()
        await store1.purge_ticket(888888)
        assert 888888 in store1._purged_tickets

        # Session 2: New instance on same database
        store2 = TradeHistoryStore(db_path=db_path)
        await store2.initialize()
        assert 888888 in store2._purged_tickets

        # Attempt to record trade with purged ticket
        await store2.record_trade(
            ticket=888888,
            symbol="GOLD",
            direction="BUY",
            volume=0.01,
            open_price=4080.0,
            close_price=4082.0,
            open_time=time.time() - 60,
            close_time=time.time(),
            pnl=5.00,
            pnl_r=2.0,
            regime="BOT",
            sub_regime="SCALPING",
            close_reason="Take Profit",
        )
        assert len(await store2.get_history()) == 0

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


@pytest.mark.anyio
async def test_api_action_delete_and_clear_history():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        store = TradeHistoryStore(db_path=db_path)
        await store.initialize()
        ctx.history_store = store
        ctx.latest_broker_time = 1792010200.0

        await store.record_trade(
            ticket=555,
            symbol="GOLD",
            direction="BUY",
            volume=0.01,
            open_price=4080.0,
            close_price=4078.0,
            open_time=1792010100.0,
            close_time=1792010150.0,
            pnl=-2.00,
            pnl_r=-1.0,
            regime="MT5",
            sub_regime="REAL",
            close_reason="SL",
        )
        assert len(await store.get_history()) == 1

        # Delete single trade via API
        del_resp = await action_delete_trade(DeleteTradePayload(ticket=555))
        assert del_resp["status"] == "OK"
        assert del_resp["ticket"] == 555
        assert len(await store.get_history()) == 0

        # Add another trade and clear all history via API
        await store.record_trade(
            ticket=777,
            symbol="GOLD",
            direction="SELL",
            volume=0.01,
            open_price=4080.0,
            close_price=4078.0,
            open_time=1792010250.0,
            close_time=1792010290.0,
            pnl=2.00,
            pnl_r=1.0,
            regime="MT5",
            sub_regime="REAL",
            close_reason="TP",
        )
        assert len(await store.get_history()) == 1

        clear_resp = await action_clear_history()
        assert clear_resp["status"] == "OK"
        assert len(await store.get_history()) == 0
        assert 777 in store._purged_tickets

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)
