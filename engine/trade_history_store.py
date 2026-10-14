"""
ApexGold Persistent Trade History & Performance Analytics Store.

Maintains SQLite persistence in data/market_history.db (table: trade_history)
so closed trades, win rate %, profit factor, and performance metrics
NEVER disappear when closing, restarting, or upgrading the application.
"""

import asyncio
from datetime import datetime, timezone
import logging
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("ApexGold.TradeHistoryStore")


class TradeHistoryStore:
    """
    Persistent SQLite Store for Completed Trades and Historical Win Rates.
    """

    def __init__(self, db_path: str = "data/market_history.db"):
        self.db_path = db_path
        self._lock = asyncio.Lock()
        self._cached_stats: Optional[Dict[str, Any]] = None
        self._cached_history: Optional[List[Dict[str, Any]]] = None
        self._last_calc_time: float = 0.0
        self._known_tickets: Set[int] = set()
        self._purged_tickets: Set[int] = set()
        self._cleared_cutoff_time: float = 0.0

    async def initialize(self) -> None:
        """Create trade_history SQLite table and indexes."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._init_db_schema)
        await self.get_stats(force_refresh=True)
        logger.info(f"TradeHistoryStore initialized SQLite table in {self.db_path} (cleared cutoff: {self._cleared_cutoff_time})")

    def _init_db_schema(self) -> None:
        """Create table synchronously in thread executor."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS meta_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
            """)
            cursor.execute("SELECT value FROM meta_settings WHERE key = 'cleared_cutoff_time';")
            row = cursor.fetchone()
            self._cleared_cutoff_time = float(row[0]) if row and row[0] else 0.0

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS purged_tickets (
                    ticket INTEGER PRIMARY KEY,
                    purged_at REAL NOT NULL
                );
            """)
            cursor.execute("SELECT ticket FROM purged_tickets;")
            self._purged_tickets = {int(r[0]) for r in cursor.fetchall() if r[0] is not None}

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_history (
                    ticket INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    volume REAL NOT NULL,
                    open_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    open_time REAL NOT NULL,
                    close_time REAL NOT NULL,
                    pnl REAL NOT NULL,
                    pnl_r REAL NOT NULL,
                    regime TEXT NOT NULL,
                    sub_regime TEXT NOT NULL,
                    close_reason TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trade_history_close_time 
                ON trade_history(close_time DESC);
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    details TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp
                ON audit_log(timestamp DESC);
            """)
            # Auto-purge any duplicate trades on startup
            # Safe deduplication: Only remove duplicate entries sharing the exact same ticket, or merge simulated placeholders
            cursor.execute("""
                DELETE FROM trade_history 
                WHERE ticket < 1000000 AND EXISTS (
                    SELECT 1 FROM trade_history t2 
                    WHERE t2.ticket >= 1000000 AND t2.symbol = trade_history.symbol 
                      AND t2.direction = trade_history.direction 
                      AND ABS(t2.open_price - trade_history.open_price) < 0.50
                );
            """)
            cursor.execute("SELECT ticket FROM trade_history;")
            self._known_tickets = {int(r[0]) for r in cursor.fetchall() if r[0] is not None}
            conn.commit()

    async def record_trade(
        self,
        ticket: int,
        symbol: str,
        direction: str,
        volume: float,
        open_price: float,
        close_price: float,
        open_time: float,
        close_time: float,
        pnl: float,
        pnl_r: float,
        regime: str,
        sub_regime: str,
        close_reason: str,
    ) -> None:
        """Save a finalized trade to SQLite with duplicate prevention."""
        t_int = int(ticket)
        if t_int in self._known_tickets:
            return
        if t_int in self._purged_tickets:
            return
        if float(close_time) <= self._cleared_cutoff_time:
            return

        async with self._lock:
            def _insert():
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    # Check for existing duplicate trade by ticket OR upgrade an unconfirmed simulated placeholder
                    cursor.execute("""
                        SELECT ticket FROM trade_history
                        WHERE ticket = ? 
                           OR (ticket < 1000000 AND symbol = ? AND direction = ? AND ABS(open_price - ?) < 0.50 AND ABS(open_time - ?) < 300)
                        LIMIT 1;
                    """, (int(ticket), str(symbol), str(direction).upper(), float(open_price), float(open_time)))
                    existing = cursor.fetchone()
                    if existing:
                        target_ticket = existing[0]
                        cursor.execute("""
                            UPDATE trade_history SET
                                ticket = ?,
                                close_price = ?,
                                close_time = ?,
                                pnl = ?,
                                pnl_r = ?,
                                close_reason = ?
                            WHERE ticket = ?;
                        """, (
                            int(ticket),
                            float(close_price),
                            float(close_time),
                            round(float(pnl), 2),
                            round(float(pnl_r), 2),
                            str(close_reason),
                            target_ticket,
                        ))
                    else:
                        cursor.execute("""
                            INSERT OR REPLACE INTO trade_history (
                                ticket, symbol, direction, volume, open_price, close_price,
                                open_time, close_time, pnl, pnl_r, regime, sub_regime, close_reason
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """, (
                            int(ticket),
                            str(symbol),
                            str(direction).upper(),
                            float(volume),
                            float(open_price),
                            float(close_price),
                            float(open_time),
                            float(close_time),
                            round(float(pnl), 2),
                            round(float(pnl_r), 2),
                            str(regime),
                            str(sub_regime),
                            str(close_reason),
                        ))
                    conn.commit()

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _insert)
            self._known_tickets.add(t_int)
            self._cached_stats = None
            self._cached_history = None
            logger.info(
                f"[TRADE PERSISTED] #{ticket} {direction} {volume} {symbol} "
                f"P&L: ${pnl:+.2f} ({pnl_r:+.2f}R) | Reason: {close_reason}"
            )

    async def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent completed trades sorted by close_time descending."""
        if self._cached_history is not None and len(self._cached_history) >= limit:
            return self._cached_history[:limit]

        async with self._lock:
            def _query():
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT ticket, symbol, direction, volume, open_price, close_price,
                               open_time, close_time, pnl, pnl_r, regime, sub_regime, close_reason
                        FROM trade_history
                        ORDER BY close_time DESC
                        LIMIT ?;
                    """, (limit,))
                    rows = cursor.fetchall()
                    result = []
                    for r in rows:
                        d = dict(r)
                        d["open_time_str"] = datetime.fromtimestamp(d["open_time"], tz=timezone.utc).strftime("%H:%M:%S")
                        d["close_time_str"] = datetime.fromtimestamp(d["close_time"], tz=timezone.utc).strftime("%H:%M:%S")
                        result.append(d)
                    return result

            loop = asyncio.get_running_loop()
            history = await loop.run_in_executor(None, _query)
            self._cached_history = history
            return history

    async def get_stats(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Compute lifetime and today's win rates, profit factor, and P&L metrics."""
        now = time.time()
        if not force_refresh and self._cached_stats is not None and (now - self._last_calc_time < 5.0):
            return self._cached_stats

        async with self._lock:
            def _calc():
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT pnl, pnl_r, close_time FROM trade_history;
                    """)
                    rows = cursor.fetchall()

                    total_trades = len(rows)
                    if total_trades == 0:
                        return {
                            "total_trades": 0,
                            "wins": 0,
                            "losses": 0,
                            "breakeven": 0,
                            "win_rate_pct": 0.0,
                            "total_pnl": 0.0,
                            "profit_factor": 1.0,
                            "avg_win": 0.0,
                            "avg_loss": 0.0,
                            "best_trade": 0.0,
                            "worst_trade": 0.0,
                            "today_trades": 0,
                            "today_pnl": 0.0,
                            "today_wins": 0,
                            "today_losses": 0,
                            "today_win_rate_pct": 0.0,
                        }

                    wins = 0
                    losses = 0
                    breakevens = 0
                    gross_profit = 0.0
                    gross_loss = 0.0
                    total_pnl = 0.0
                    best_trade = -999999.0
                    worst_trade = 999999.0

                    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
                    today_trades = 0
                    today_pnl = 0.0
                    today_wins = 0
                    today_losses = 0

                    for pnl, pnl_r, close_t in rows:
                        total_pnl += pnl
                        if pnl > best_trade: best_trade = pnl
                        if pnl < worst_trade: worst_trade = pnl

                        if pnl > 0.01:
                            wins += 1
                            gross_profit += pnl
                        elif pnl < -0.01:
                            losses += 1
                            gross_loss += abs(pnl)
                        else:
                            breakevens += 1

                        if close_t >= today_start:
                            today_trades += 1
                            today_pnl += pnl
                            if pnl > 0.01: today_wins += 1
                            elif pnl < -0.01: today_losses += 1

                    win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
                    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 1.0)
                    avg_win = (gross_profit / wins) if wins > 0 else 0.0
                    avg_loss = (gross_loss / losses) if losses > 0 else 0.0
                    today_wr = (today_wins / today_trades * 100.0) if today_trades > 0 else 0.0

                    return {
                        "total_trades": total_trades,
                        "wins": wins,
                        "losses": losses,
                        "breakeven": breakevens,
                        "win_rate_pct": round(win_rate, 1),
                        "total_pnl": round(total_pnl, 2),
                        "profit_factor": round(profit_factor, 2),
                        "avg_win": round(avg_win, 2),
                        "avg_loss": round(avg_loss, 2),
                        "best_trade": round(best_trade if best_trade != -999999.0 else 0.0, 2),
                        "worst_trade": round(worst_trade if worst_trade != 999999.0 else 0.0, 2),
                        "today_trades": today_trades,
                        "today_pnl": round(today_pnl, 2),
                        "today_wins": today_wins,
                        "today_losses": today_losses,
                        "today_win_rate_pct": round(today_wr, 1),
                    }

            loop = asyncio.get_running_loop()
            stats = await loop.run_in_executor(None, _calc)
            self._cached_stats = stats
            self._last_calc_time = now
            return stats

    async def clear_history(self, broker_time: Optional[float] = None) -> Dict[str, Any]:
        """Completely purge all recorded trade history and reset win rate / P&L statistics."""
        now_ts = time.time()
        async with self._lock:
            def _delete():
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    # 1. Capture all existing tickets and maximum close time in trade_history
                    cursor.execute("SELECT ticket, close_time FROM trade_history;")
                    rows = cursor.fetchall()
                    existing_tickets = [int(r[0]) for r in rows if r[0] is not None]
                    max_db_close = max([float(r[1]) for r in rows if r[1] is not None] + [0.0])

                    # Calculate effective cutoff covering both local time and MT5 broker clock
                    b_time = float(broker_time) if broker_time else 0.0
                    eff_cutoff = max(now_ts, max_db_close + 1.0, b_time + 1.0)

                    # 2. Record all existing tickets into purged_tickets table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS purged_tickets (
                            ticket INTEGER PRIMARY KEY,
                            purged_at REAL NOT NULL
                        );
                    """)
                    for t in existing_tickets:
                        cursor.execute("INSERT OR IGNORE INTO purged_tickets (ticket, purged_at) VALUES (?, ?);", (t, eff_cutoff))

                    # 3. Purge trade_history
                    cursor.execute("DELETE FROM trade_history;")
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS meta_settings (
                            key TEXT PRIMARY KEY,
                            value TEXT NOT NULL
                        );
                    """)
                    cursor.execute("INSERT OR REPLACE INTO meta_settings (key, value) VALUES ('cleared_cutoff_time', ?);", (str(eff_cutoff),))
                    conn.commit()
                    return existing_tickets, eff_cutoff

            loop = asyncio.get_running_loop()
            deleted_tickets, cutoff = await loop.run_in_executor(None, _delete)
            self._purged_tickets.update(deleted_tickets)
            self._cleared_cutoff_time = cutoff
            self._known_tickets.clear()
            self._cached_stats = None
            self._cached_history = None
            logger.info(f"[TRADE HISTORY CLEARED] Purged {len(deleted_tickets)} past tickets. Cutoff set to {cutoff}.")

        return await self.get_stats(force_refresh=True)

    async def delete_trade(self, ticket: int) -> bool:
        """Permanently delete an individual trade by ticket and add to purged set."""
        t_int = int(ticket)
        async with self._lock:
            def _del_one():
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM trade_history WHERE ticket = ?;", (t_int,))
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS purged_tickets (
                            ticket INTEGER PRIMARY KEY,
                            purged_at REAL NOT NULL
                        );
                    """)
                    cursor.execute("INSERT OR IGNORE INTO purged_tickets (ticket, purged_at) VALUES (?, ?);", (t_int, time.time()))
                    conn.commit()

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _del_one)
            self._purged_tickets.add(t_int)
            self._known_tickets.discard(t_int)
            self._cached_stats = None
            self._cached_history = None
            logger.info(f"[TRADE DELETED] Ticket #{t_int} deleted from SQLite and permanently purged.")
        return True

    async def purge_ticket(self, ticket: int) -> None:
        """Mark a ticket as purged so MT5 cannot ingest it into history."""
        t_int = int(ticket)
        self._purged_tickets.add(t_int)
        self._known_tickets.discard(t_int)
        loop = asyncio.get_running_loop()
        def _insert():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS purged_tickets (
                        ticket INTEGER PRIMARY KEY,
                        purged_at REAL NOT NULL
                    );
                """)
                cursor.execute("INSERT OR IGNORE INTO purged_tickets (ticket, purged_at) VALUES (?, ?);", (t_int, time.time()))
                cursor.execute("DELETE FROM trade_history WHERE ticket = ?;", (t_int,))
                conn.commit()
        await loop.run_in_executor(None, _insert)

    async def record_audit_event(self, event_type: str, details: str) -> None:
        """Persist safety, operational recovery, and emergency action audit log events."""
        now_ts = time.time()
        loop = asyncio.get_running_loop()

        def _insert():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO audit_log (event_type, details, timestamp) VALUES (?, ?, ?);",
                    (event_type, details, now_ts),
                )
                conn.commit()

        await loop.run_in_executor(None, _insert)
        logger.info(f"[AUDIT EVENT: {event_type}] {details}")

    async def get_audit_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent persistent audit events."""
        loop = asyncio.get_running_loop()

        def _fetch():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT event_type, details, timestamp, created_at FROM audit_log ORDER BY timestamp DESC LIMIT ?;",
                    (limit,),
                )
                return [dict(r) for r in cursor.fetchall()]

        return await loop.run_in_executor(None, _fetch)

