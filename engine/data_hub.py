"""
ApexGold Multi-Timeframe Data Hub & Local Time-Series Store.

Fetches, synchronizes, and caches 1m, 5m, 15m, and 1H historical candles
into a local SQLite store (data/market_history.db).
Ensures zero-latency multi-timeframe access for strategy scoring,
indicator calculations, and pattern similarity matching.
"""

import asyncio
import logging
import os
import sqlite3
import time
from typing import Dict, List, Optional

import pandas as pd

from brokers.base import AbstractBrokerAdapter, Bar

logger = logging.getLogger("ApexGold.DataHub")


class DataHub:
    """
    Central Time-Series & Historical Caching Engine.
    Maintains persistent SQLite storage and fast in-memory DataFrame buffers.
    """

    TIMEFRAMES = ["1m", "5m", "15m", "1H"]

    # Number of bars required for each timeframe warmup
    # 1H (90d) = ~2,160 bars; 15m (30d) = ~2,880 bars; 5m (14d) = ~4,032 bars; 1m (7d) = ~10,080 bars
    WARMUP_BAR_COUNTS = {
        "1H": 2160,
        "15m": 2880,
        "5m": 2500,
        "1m": 2000, # Conservative live buffer
    }

    def __init__(self, db_path: str = "data/market_history.db"):
        self.db_path = db_path
        self._memory_cache: Dict[str, Dict[str, List[Bar]]] = {} # symbol -> tf -> List[Bar]
        self._lock = asyncio.Lock()
        self._initialized = False
        self._last_db_save: Dict[str, float] = {}

    async def initialize(self) -> None:
        """Create database directory and SQLite tables with performance indexes."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._init_db_schema)
        self._initialized = True
        logger.info(f"DataHub initialized SQLite cache at {self.db_path}")

    def _init_db_schema(self) -> None:
        """Run SQLite table creation synchronously in executor."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candles (
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    time INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    PRIMARY KEY (symbol, timeframe, time)
                );
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_candles_lookup 
                ON candles(symbol, timeframe, time DESC);
            """)
            conn.commit()

    async def warmup(self, broker: AbstractBrokerAdapter, symbols: List[str]) -> None:
        """
        Pull historical data across 1m, 5m, 15m, and 1H for all active symbols.
        Stores them in SQLite and warms up memory cache.
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"Starting DataHub historical warmup for symbols: {symbols}...")

        for symbol in symbols:
            if symbol not in self._memory_cache:
                self._memory_cache[symbol] = {}

            for tf in self.TIMEFRAMES:
                needed = self.WARMUP_BAR_COUNTS.get(tf, 500)
                try:
                    # 1. Attempt to pull from broker
                    bars = await broker.get_rates(symbol, tf, count=needed)
                    if bars and len(bars) > 0:
                        await self._save_bars_to_db(symbol, tf, bars)
                        async with self._lock:
                            self._memory_cache[symbol][tf] = bars
                        logger.info(f"Loaded {len(bars)} bars for {symbol} [{tf}] from broker.")
                    else:
                        # 2. Try loading cached bars from SQLite
                        db_bars = await self._load_bars_from_db(symbol, tf, limit=needed)
                        if db_bars:
                            async with self._lock:
                                self._memory_cache[symbol][tf] = db_bars
                            logger.info(f"Loaded {len(db_bars)} bars for {symbol} [{tf}] from local DB.")
                        else:
                            logger.warning(f"No bars available for {symbol} [{tf}]")

                except Exception as e:
                    logger.error(f"Warmup failed for {symbol} [{tf}]: {e}")
                    # Attempt SQLite fallback
                    db_bars = await self._load_bars_from_db(symbol, tf, limit=needed)
                    if db_bars:
                        async with self._lock:
                            self._memory_cache[symbol][tf] = db_bars

    async def _save_bars_to_db(self, symbol: str, timeframe: str, bars: List[Bar]) -> None:
        """Batch insert/replace candles into SQLite."""
        if not bars:
            return

        records = [
            (symbol, timeframe, b.time, b.open, b.high, b.low, b.close, b.volume)
            for b in bars
        ]

        def _insert():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT OR REPLACE INTO candles (symbol, timeframe, time, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """, records)
                conn.commit()

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _insert)

    async def _load_bars_from_db(self, symbol: str, timeframe: str, limit: int = 1000) -> List[Bar]:
        """Load recent bars from SQLite."""
        def _query():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT time, open, high, low, close, volume
                    FROM candles
                    WHERE symbol = ? AND timeframe = ?
                    ORDER BY time DESC
                    LIMIT ?;
                """, (symbol, timeframe, limit))
                rows = cursor.fetchall()
                # Reverse to get chronological order (oldest to newest)
                return [
                    Bar(
                        time=int(r[0]),
                        open=float(r[1]),
                        high=float(r[2]),
                        low=float(r[3]),
                        close=float(r[4]),
                        volume=float(r[5]),
                    )
                    for r in reversed(rows)
                ]

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _query)

    async def append_bar(self, symbol: str, timeframe: str, bar: Bar) -> None:
        """
        Append a new closed candle or update the in-progress candle.
        Maintains memory cache and updates SQLite asynchronously.
        """
        async with self._lock:
            if symbol not in self._memory_cache:
                self._memory_cache[symbol] = {}
            if timeframe not in self._memory_cache[symbol]:
                self._memory_cache[symbol][timeframe] = []

            bars = self._memory_cache[symbol][timeframe]

            if bars and bars[-1].time == bar.time:
                # Update existing candle
                bars[-1] = bar
            else:
                bars.append(bar)
                # Keep cache capped to prevent memory leak
                max_bars = self.WARMUP_BAR_COUNTS.get(timeframe, 1000) + 200
                if len(bars) > max_bars:
                    self._memory_cache[symbol][timeframe] = bars[-max_bars:]

        # Asynchronously store to DB
        await self._save_bars_to_db(symbol, timeframe, [bar])

    async def inject_bars(self, symbol: str, timeframe: str, bars: List[Bar]) -> None:
        """
        Inject a sequence of authentic broker bars (e.g. from MT5).
        Merges with memory cache and updates SQLite database.
        """
        if not bars:
            return
        async with self._lock:
            if symbol not in self._memory_cache:
                self._memory_cache[symbol] = {}
            existing = self._memory_cache[symbol].get(timeframe, [])
            if existing:
                first_new_time = bars[0].time
                # Only keep existing bars if their price matches the new bars scale (within 15%)
                avg_new = sum(b.close for b in bars[:10]) / min(len(bars), 10)
                valid_existing = [
                    b for b in existing 
                    if b.time < first_new_time and abs(b.close - avg_new) / max(avg_new, 1.0) < 0.15
                ]
                merged = valid_existing + bars
                max_bars = max(self.WARMUP_BAR_COUNTS.get(timeframe, 1000) + 500, 3000)
                if len(merged) > max_bars:
                    merged = merged[-max_bars:]
                self._memory_cache[symbol][timeframe] = merged
            else:
                self._memory_cache[symbol][timeframe] = bars

        # Debounce DB writes: only write to DB every 30 seconds per timeframe to avoid SQLite contention
        now = time.time()
        db_key = f"{symbol}_{timeframe}"
        if now - self._last_db_save.get(db_key, 0.0) >= 30.0:
            self._last_db_save[db_key] = now
            await self._save_bars_to_db(symbol, timeframe, bars)

    async def get_bars(self, symbol: str, timeframe: str, count: int = 100) -> List[Bar]:
        """Fetch the last `count` bars for a symbol and timeframe."""
        async with self._lock:
            bars = self._memory_cache.get(symbol, {}).get(timeframe, [])
            if len(bars) >= count:
                return bars[-count:]

        # If not sufficient in memory, fetch from DB
        db_bars = await self._load_bars_from_db(symbol, timeframe, limit=count)
        if db_bars:
            async with self._lock:
                if symbol not in self._memory_cache:
                    self._memory_cache[symbol] = {}
                self._memory_cache[symbol][timeframe] = db_bars
            return db_bars[-count:]

        return bars

    async def get_dataframe(self, symbol: str, timeframe: str, count: int = 100) -> pd.DataFrame:
        """
        Retrieve candles formatted as a pandas DataFrame ready for indicators.
        Columns: ['time', 'open', 'high', 'low', 'close', 'volume'].
        """
        bars = await self.get_bars(symbol, timeframe, count=count)
        if not bars:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        data = {
            "time": [b.time for b in bars],
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
        }
        return pd.DataFrame(data)

    async def get_multi_timeframe_data(self, symbol: str, counts: Optional[Dict[str, int]] = None) -> Dict[str, pd.DataFrame]:
        """
        Returns aligned DataFrames for all 4 primary timeframes (1H, 15m, 5m, 1m).
        """
        default_counts = {
            "1H": 100,
            "15m": 120,
            "5m": 150,
            "1m": 100,
        }
        if counts:
            default_counts.update(counts)

        results: Dict[str, pd.DataFrame] = {}
        for tf in self.TIMEFRAMES:
            cnt = default_counts.get(tf, 100)
            results[tf] = await self.get_dataframe(symbol, tf, count=cnt)

        return results

