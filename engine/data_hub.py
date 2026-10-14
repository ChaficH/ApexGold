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
        """Run SQLite table creation and schema migration synchronously in executor."""
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
                    is_synthetic INTEGER DEFAULT 0,
                    source TEXT DEFAULT 'BROKER',
                    PRIMARY KEY (symbol, timeframe, time)
                );
            """)
            # Schema migration for existing databases
            cursor.execute("PRAGMA table_info(candles);")
            cols = [c[1] for c in cursor.fetchall()]
            if "is_synthetic" not in cols:
                cursor.execute("ALTER TABLE candles ADD COLUMN is_synthetic INTEGER DEFAULT 0;")
            if "source" not in cols:
                cursor.execute("ALTER TABLE candles ADD COLUMN source TEXT DEFAULT 'BROKER';")

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
        """Batch insert/replace candles into SQLite preserving synthetic and broker provenance."""
        if not bars:
            return

        records = [
            (
                symbol,
                timeframe,
                b.time,
                b.open,
                b.high,
                b.low,
                b.close,
                b.volume,
                1 if getattr(b, "is_synthetic", False) else 0,
                getattr(b, "source", "SYNTHETIC" if getattr(b, "is_synthetic", False) else "BROKER"),
            )
            for b in bars
        ]

        def _insert():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT OR REPLACE INTO candles (symbol, timeframe, time, open, high, low, close, volume, is_synthetic, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, records)
                conn.commit()

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _insert)

    async def _load_bars_from_db(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 1000,
        exclude_synthetic: bool = False
    ) -> List[Bar]:
        """Load recent bars from SQLite preserving candle provenance."""
        def _query():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                query = """
                    SELECT time, open, high, low, close, volume, is_synthetic, source
                    FROM candles
                    WHERE symbol = ? AND timeframe = ?
                """
                params = [symbol, timeframe]
                if exclude_synthetic:
                    query += " AND is_synthetic = 0 AND source != 'SYNTHETIC'"
                query += " ORDER BY time DESC LIMIT ?;"
                params.append(limit)

                cursor.execute(query, tuple(params))
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
                        is_synthetic=bool(r[6]),
                        source=str(r[7]),
                    )
                    for r in reversed(rows)
                ]

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _query)

    TF_SECONDS = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "1H": 3600,
        "1h": 3600,
        "4H": 14400,
        "1D": 86400,
    }

    async def append_bar(self, symbol: str, timeframe: str, bar: Bar) -> None:
        """
        Append a new closed candle or update the in-progress candle.
        Ensures strict timestamp deduplication and monotonic chronological sorting.
        """
        async with self._lock:
            if symbol not in self._memory_cache:
                self._memory_cache[symbol] = {}
            if timeframe not in self._memory_cache[symbol]:
                self._memory_cache[symbol][timeframe] = []

            bars = self._memory_cache[symbol][timeframe]

            if not bars or bar.time > bars[-1].time:
                bars.append(bar)
            elif bars[-1].time == bar.time:
                bars[-1] = bar
            else:
                # Out-of-order or duplicate timestamp in past
                updated = False
                for idx, existing_bar in enumerate(bars):
                    if existing_bar.time == bar.time:
                        bars[idx] = bar
                        updated = True
                        break
                if not updated:
                    bars.append(bar)
                    bars.sort(key=lambda b: b.time)

            # Keep cache capped to prevent memory leak
            max_bars = self.WARMUP_BAR_COUNTS.get(timeframe, 1000) + 200
            if len(bars) > max_bars:
                self._memory_cache[symbol][timeframe] = bars[-max_bars:]

        # Asynchronously store to DB
        await self._save_bars_to_db(symbol, timeframe, [bar])

    async def inject_bars(self, symbol: str, timeframe: str, bars: List[Bar]) -> None:
        """
        Inject a sequence of authentic broker bars (e.g. from MT5).
        Merges with memory cache, strictly deduplicates by timestamp,
        guarantees monotonic ordering, and updates SQLite database.
        """
        if not bars:
            return
        async with self._lock:
            if symbol not in self._memory_cache:
                self._memory_cache[symbol] = {}
            existing = self._memory_cache[symbol].get(timeframe, [])

            # Deduplicate by timestamp and maintain chronological order
            bar_map: Dict[int, Bar] = {b.time: b for b in existing}
            for b in bars:
                bar_map[b.time] = b

            merged = [bar_map[t] for t in sorted(bar_map.keys())]
            max_bars = max(self.WARMUP_BAR_COUNTS.get(timeframe, 1000) + 500, 3000)
            if len(merged) > max_bars:
                merged = merged[-max_bars:]
            self._memory_cache[symbol][timeframe] = merged

        # Debounce DB writes: only write to DB every 30 seconds per timeframe to avoid SQLite contention
        now = time.time()
        db_key = f"{symbol}_{timeframe}"
        if now - self._last_db_save.get(db_key, 0.0) >= 30.0:
            self._last_db_save[db_key] = now
            await self._save_bars_to_db(symbol, timeframe, bars)

    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        count: int = 100,
        completed_only: bool = False,
        now_ts: Optional[float] = None,
        exclude_synthetic: bool = False,
    ) -> List[Bar]:
        """
        Fetch the last `count` bars for a symbol and timeframe.
        If completed_only is True, excludes incomplete (in-progress) higher timeframe bars.
        If exclude_synthetic is True, excludes synthetic bars so they are never treated as broker data.
        """
        ref_time = now_ts or time.time()
        tf_sec = self.TF_SECONDS.get(timeframe, 60)

        async with self._lock:
            bars = self._memory_cache.get(symbol, {}).get(timeframe, [])
            if exclude_synthetic:
                bars = [b for b in bars if not getattr(b, "is_synthetic", False) and getattr(b, "source", "BROKER") != "SYNTHETIC"]
            if len(bars) >= count:
                res = bars[-count:]
                if completed_only:
                    res = [b for b in res if (b.time + tf_sec) <= ref_time]
                return res

        # If not sufficient in memory, fetch older bars from DB and merge without losing memory bars
        db_bars = await self._load_bars_from_db(symbol, timeframe, limit=count + 20, exclude_synthetic=exclude_synthetic)
        async with self._lock:
            mem_bars = self._memory_cache.get(symbol, {}).get(timeframe, [])
            if exclude_synthetic:
                mem_bars = [b for b in mem_bars if not getattr(b, "is_synthetic", False) and getattr(b, "source", "BROKER") != "SYNTHETIC"]
            bar_map = {b.time: b for b in db_bars}
            for b in mem_bars:
                bar_map[b.time] = b # in-memory bars take precedence

            merged = [bar_map[t] for t in sorted(bar_map.keys())]
            if not exclude_synthetic:
                if symbol not in self._memory_cache:
                    self._memory_cache[symbol] = {}
                self._memory_cache[symbol][timeframe] = merged

            res = merged[-count:]
            if completed_only:
                res = [b for b in res if (b.time + tf_sec) <= ref_time]
            return res

    async def get_dataframe(
        self,
        symbol: str,
        timeframe: str,
        count: int = 100,
        completed_only: bool = False,
        now_ts: Optional[float] = None,
        exclude_synthetic: bool = False,
    ) -> pd.DataFrame:
        """
        Retrieve candles formatted as a pandas DataFrame ready for indicators.
        Columns: ['time', 'open', 'high', 'low', 'close', 'volume', 'is_synthetic', 'source'].
        """
        bars = await self.get_bars(
            symbol,
            timeframe,
            count=count,
            completed_only=completed_only,
            now_ts=now_ts,
            exclude_synthetic=exclude_synthetic,
        )
        if not bars:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "is_synthetic", "source"])

        data = {
            "time": [b.time for b in bars],
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
            "is_synthetic": [getattr(b, "is_synthetic", False) for b in bars],
            "source": [getattr(b, "source", "BROKER") for b in bars],
        }
        return pd.DataFrame(data)

    async def get_multi_timeframe_data(
        self,
        symbol: str,
        counts: Optional[Dict[str, int]] = None,
        higher_tf_completed_only: bool = True,
        now_ts: Optional[float] = None,
        exclude_synthetic: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Returns aligned DataFrames for all 4 primary timeframes (1H, 15m, 5m, 1m).
        When higher_tf_completed_only is True, 1H and 15m candles exclude incomplete in-progress bars.
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
            completed_flag = higher_tf_completed_only if tf in ["1H", "15m"] else False
            results[tf] = await self.get_dataframe(
                symbol,
                tf,
                count=cnt,
                completed_only=completed_flag,
                now_ts=now_ts,
                exclude_synthetic=exclude_synthetic,
            )

        return results

    def is_stale(
        self,
        symbol: str,
        timeframe: str,
        max_age_seconds: Optional[float] = None,
        now_ts: Optional[float] = None
    ) -> bool:
        """Check if cached market data for symbol and timeframe is stale."""
        bars = self._memory_cache.get(symbol, {}).get(timeframe, [])
        if not bars:
            return True
        latest_time = bars[-1].time
        tf_sec = self.TF_SECONDS.get(timeframe, 60)
        allowed_age = max_age_seconds if max_age_seconds is not None else (tf_sec * 3.0)
        ref_time = now_ts or time.time()
        return (ref_time - latest_time) > allowed_age

