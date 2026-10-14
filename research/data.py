"""
ApexGold Quantitative Research Data Pipeline.
Reproducible multi-year historical data fetcher, tick-to-bar aggregator, validator,
and walk-forward chronological splitter.

Sources supported in priority order:
1. Local cached parquet in data/research/
2. MT5 historical exports / market_history.db SQLite
3. FX-Data public tick repository (https://github.com/FX-Data/FX-Data-XAUUSD-DS)
4. Yahoo Finance (GC=F Gold Futures 2024-2026 current regime)
"""

import os
import io
import time
import logging
import sqlite3
import datetime
from typing import Dict, List, Tuple, Optional, Any
import concurrent.futures

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger("ApexGold.ResearchData")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

RESEARCH_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "research"))
os.makedirs(RESEARCH_DATA_DIR, exist_ok=True)


def parse_tick_csv_chunk(csv_text: str) -> Optional[pd.DataFrame]:
    """
    Parse FX-Data tick CSV chunk.
    Format: timestamp, bid_scaled, ask_scaled, vol_bid, vol_ask
    Note: Prices are scaled / 100 in FX-Data; must multiply by 100.
    """
    if not csv_text or len(csv_text.strip()) == 0:
        return None
    try:
        df = pd.read_csv(
            io.StringIO(csv_text),
            header=None,
            names=["time_str", "bid_scaled", "ask_scaled", "vol_b", "vol_a"],
            usecols=["time_str", "bid_scaled", "ask_scaled"],
        )
        if df.empty:
            return None
        df["bid"] = df["bid_scaled"] * 100.0
        df["ask"] = df["ask_scaled"] * 100.0
        df["time"] = pd.to_datetime(df["time_str"], format="%Y.%m.%d %H:%M:%S.%f", utc=True)
        return df[["time", "bid", "ask"]]
    except Exception as e:
        logger.warning(f"Failed parsing tick csv chunk: {e}")
        return None


def fetch_file_content(url: str, session: requests.Session, retries: int = 3) -> Optional[str]:
    """Fetch raw file content with retry backoff."""
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=15)
            if r.status_code == 200:
                return r.text
            elif r.status_code == 404:
                return None
            time.sleep(0.5 * (attempt + 1))
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return None


def download_fxdata_month(year: int, month: int, max_workers: int = 15) -> pd.DataFrame:
    """
    Download all hourly tick files for a given month from FX-Data GitHub repo
    and aggregate into 1-minute OHLCV bars.
    """
    branch = f"XAUUSD-{year}"
    month_str = f"{month:02d}"
    api_url = f"https://api.github.com/repos/FX-Data/FX-Data-XAUUSD-DS/contents/XAUUSD/{year}/{month_str}?ref={branch}"
    
    logger.info(f"Fetching file listing for {year}-{month_str} from {api_url}...")
    headers = {"User-Agent": "ApexGold-Research/1.0"}
    r = requests.get(api_url, headers=headers, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to fetch directory listing for {year}-{month_str}: HTTP {r.status_code}")
    
    file_items = [f["name"] for f in r.json() if f["name"].endswith(".csv")]
    logger.info(f"Found {len(file_items)} tick files for {year}-{month_str}. Downloading concurrently...")

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=max_workers + 5, pool_maxsize=max_workers + 5)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    raw_urls = [
        (name, f"https://raw.githubusercontent.com/FX-Data/FX-Data-XAUUSD-DS/{branch}/XAUUSD/{year}/{month_str}/{name}")
        for name in file_items
    ]

    all_1m_bars = []

    def process_url(item):
        name, url = item
        text = fetch_file_content(url, session)
        if not text:
            return None
        ticks = parse_tick_csv_chunk(text)
        if ticks is None or ticks.empty:
            return None
        # Resample this hour's ticks directly into 1-minute bars
        ticks.set_index("time", inplace=True)
        res_bid = ticks["bid"].resample("1min").ohlc().dropna()
        res_ask = ticks["ask"].resample("1min").ohlc().dropna()
        res_count = ticks["bid"].resample("1min").count()

        common_idx = res_bid.index.intersection(res_ask.index)
        if common_idx.empty:
            return None
        res_bid = res_bid.loc[common_idx]
        res_ask = res_ask.loc[common_idx]
        res_count = res_count.loc[common_idx]

        combined = pd.DataFrame({
            "open": res_bid["open"],
            "high": res_bid["high"],
            "low": res_bid["low"],
            "close": res_bid["close"],
            "ask_open": res_ask["open"],
            "ask_high": res_ask["high"],
            "ask_low": res_ask["low"],
            "ask_close": res_ask["close"],
            "volume": res_count,
            "spread": (res_ask["close"] - res_bid["close"]).round(4),
        }).dropna(subset=["open", "high", "low", "close"])
        return combined

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(process_url, raw_urls):
            if res is not None and not res.empty:
                all_1m_bars.append(res)

    if not all_1m_bars:
        raise RuntimeError(f"No 1m bars successfully constructed for {year}-{month_str}")

    full_month_1m = pd.concat(all_1m_bars).sort_index()
    # Remove duplicate timestamps
    full_month_1m = full_month_1m[~full_month_1m.index.duplicated(keep="first")]
    logger.info(f"Aggregated {len(full_month_1m)} 1-minute bars for {year}-{month_str}.")
    return full_month_1m


def to_unix_seconds(idx: pd.DatetimeIndex) -> np.ndarray:
    """Universal resolution-agnostic conversion from DatetimeIndex to unix epoch seconds."""
    dtype_str = str(idx.dtype)
    if "us" in dtype_str:
        scale = 1_000_000
    elif "ms" in dtype_str:
        scale = 1_000
    elif "s" in dtype_str and "ns" not in dtype_str:
        scale = 1
    else:
        scale = 1_000_000_000
    return (idx.view("int64") // scale).astype(int)


def resample_1m_to_higher_timeframes(df_1m: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Resample 1-minute OHLCV bars into 5m, 15m, and 1H bars with strict completion alignment.
    Adds integer unix timestamp column 'time' for engine compatibility.
    """
    df = df_1m.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df.set_index("time", inplace=True)

    result = {}
    
    # Format 1m
    df_1m_out = df.dropna(subset=["open", "high", "low", "close"]).copy()
    df_1m_out["time"] = to_unix_seconds(df_1m_out.index)
    result["1m"] = df_1m_out.reset_index(drop=True)

    # 5m
    df_5m = pd.DataFrame({
        "open": df["open"].resample("5min", closed="left", label="left").first(),
        "high": df["high"].resample("5min", closed="left", label="left").max(),
        "low": df["low"].resample("5min", closed="left", label="left").min(),
        "close": df["close"].resample("5min", closed="left", label="left").last(),
        "volume": df["volume"].resample("5min", closed="left", label="left").sum(),
        "spread": df["spread"].resample("5min", closed="left", label="left").mean(),
    }).dropna()
    df_5m["time"] = to_unix_seconds(df_5m.index)
    result["5m"] = df_5m.reset_index(drop=True)

    # 15m
    df_15m = pd.DataFrame({
        "open": df["open"].resample("15min", closed="left", label="left").first(),
        "high": df["high"].resample("15min", closed="left", label="left").max(),
        "low": df["low"].resample("15min", closed="left", label="left").min(),
        "close": df["close"].resample("15min", closed="left", label="left").last(),
        "volume": df["volume"].resample("15min", closed="left", label="left").sum(),
        "spread": df["spread"].resample("15min", closed="left", label="left").mean(),
    }).dropna()
    df_15m["time"] = to_unix_seconds(df_15m.index)
    result["15m"] = df_15m.reset_index(drop=True)

    # 1H
    df_1h = pd.DataFrame({
        "open": df["open"].resample("1h", closed="left", label="left").first(),
        "high": df["high"].resample("1h", closed="left", label="left").max(),
        "low": df["low"].resample("1h", closed="left", label="left").min(),
        "close": df["close"].resample("1h", closed="left", label="left").last(),
        "volume": df["volume"].resample("1h", closed="left", label="left").sum(),
        "spread": df["spread"].resample("1h", closed="left", label="left").mean(),
    }).dropna()
    df_1h["time"] = to_unix_seconds(df_1h.index)
    result["1H"] = df_1h.reset_index(drop=True)

    return result


def fetch_2018_baseline_data(cache_dir: str = RESEARCH_DATA_DIR) -> Dict[str, pd.DataFrame]:
    """
    Download and cache the exact March-April 2018 XAUUSD dataset used for the known baseline.
    Caches as parquet in data/research/xauusd_2018_mar_apr_{tf}.parquet.
    """
    cache_prefix = os.path.join(cache_dir, "xauusd_2018_mar_apr")
    tfs = ["1m", "5m", "15m", "1H"]
    cached_files = {tf: f"{cache_prefix}_{tf}.parquet" for tf in tfs}

    if all(os.path.exists(path) for path in cached_files.values()):
        logger.info(f"Loading cached 2018 March-April dataset from {cache_dir}...")
        dfs = {tf: pd.read_parquet(path) for tf, path in cached_files.items()}
        validate_dataset(dfs)
        return dfs

    logger.info("Downloading March 2018 ticks...")
    m3 = download_fxdata_month(2018, 3)
    logger.info("Downloading April 2018 ticks...")
    m4 = download_fxdata_month(2018, 4)

    df_1m_all = pd.concat([m3, m4]).sort_index()
    df_1m_all = df_1m_all[~df_1m_all.index.duplicated(keep="first")]

    logger.info(f"Total 2018 Mar-Apr 1m bars: {len(df_1m_all)}. Resampling timeframes...")
    dfs = resample_1m_to_higher_timeframes(df_1m_all)
    validate_dataset(dfs)

    for tf, path in cached_files.items():
        dfs[tf].to_parquet(path, index=False)
        logger.info(f"Cached {tf} bars -> {path} ({len(dfs[tf])} rows)")

    return dfs


def fetch_recent_regime_data(cache_dir: str = RESEARCH_DATA_DIR) -> Dict[str, pd.DataFrame]:
    """
    Extract recent 2026 XAUUSD data from local MT5 SQLite cache (market_history.db)
    supplemented by continuous yfinance GC=F gold futures bars.
    """
    cache_prefix = os.path.join(cache_dir, "xauusd_recent_regime")
    tfs = ["1m", "5m", "15m", "1H"]
    cached_files = {tf: f"{cache_prefix}_{tf}.parquet" for tf in tfs}

    if all(os.path.exists(path) for path in cached_files.values()):
        logger.info(f"Loading cached recent regime dataset from {cache_dir}...")
        dfs = {tf: pd.read_parquet(path) for tf, path in cached_files.items()}
        validate_dataset(dfs)
        return dfs

    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "market_history.db"))
    dfs = {}
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        for tf in tfs:
            query = f"SELECT time, open, high, low, close, volume FROM candles WHERE symbol IN ('GOLD', 'XAUUSD', 'XAUUSD.m', 'XAUUSDm') AND timeframe = '{tf}' ORDER BY time ASC"
            df = pd.read_sql_query(query, conn)
            if not df.empty:
                df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
                df["spread"] = 0.25
                dfs[tf] = df
        conn.close()

    if len(dfs) == 4 and all(len(df) > 500 for df in dfs.values()):
        logger.info("Successfully extracted 2026 data from SQLite cache.")
        for tf, path in cached_files.items():
            dfs[tf].to_parquet(path, index=False)
        validate_dataset(dfs)
        return dfs

    # Fallback to yfinance GC=F
    logger.info("Fetching Gold data from Yahoo Finance GC=F...")
    import yfinance as yf
    ticker = yf.Ticker("GC=F")
    hist_1h = ticker.history(period="730d", interval="1h")
    hist_5m = ticker.history(period="60d", interval="5m")
    hist_15m = ticker.history(period="60d", interval="15m")
    hist_1m = ticker.history(period="7d", interval="1m")

    def format_yf(df):
        out = pd.DataFrame({
            "time": (df.index.astype("int64") // 10**9).astype(int),
            "open": df["Open"].values,
            "high": df["High"].values,
            "low": df["Low"].values,
            "close": df["Close"].values,
            "volume": df["Volume"].values,
            "spread": 0.25,
        }).dropna().drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
        return out

    dfs = {
        "1H": format_yf(hist_1h),
        "15m": format_yf(hist_15m),
        "5m": format_yf(hist_5m),
        "1m": format_yf(hist_1m),
    }

    for tf, path in cached_files.items():
        dfs[tf].to_parquet(path, index=False)
    validate_dataset(dfs)
    return dfs


def validate_dataset(dfs: Dict[str, pd.DataFrame]) -> bool:
    """
    Sanity checks on dataset:
    - Time column is monotonic increasing.
    - No NaN values in OHLC.
    - Low <= Open, High, Close and High >= Open, Low, Close.
    - Spread is positive.
    """
    for tf, df in dfs.items():
        if df.empty:
            raise ValueError(f"Dataset for timeframe {tf} is empty!")
        if "time" not in df.columns:
            raise ValueError(f"Timeframe {tf} missing 'time' column!")
        if not df["time"].is_monotonic_increasing:
            raise ValueError(f"Timeframe {tf} timestamps are not strictly monotonic!")
        
        # Check NaNs
        for col in ["open", "high", "low", "close"]:
            if df[col].isna().any():
                raise ValueError(f"Timeframe {tf} contains NaNs in {col}!")
        
        # Check High/Low validity
        if not (df["high"] >= df["low"]).all():
            raise ValueError(f"Timeframe {tf} contains bars where high < low!")
        
        # Spread sanity
        if "spread" in df.columns and (df["spread"] <= 0).any():
            logger.warning(f"Timeframe {tf} has non-positive spread values; resetting minimum to 0.10.")
            df["spread"] = df["spread"].clip(lower=0.10)

    logger.info("Dataset validation passed: monotonic, no NaNs, valid price envelopes.")
    return True


def split_train_holdout(
    dfs: Dict[str, pd.DataFrame],
    holdout_ratio: float = 0.20
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Strict chronological split:
    TRAIN/VALIDATE = first (1 - holdout_ratio) of chronological timeline.
    HOLD-OUT TEST = final holdout_ratio of chronological timeline (LOCKED).
    """
    primary_tf = "5m" if "5m" in dfs else "1H"
    df_primary = dfs[primary_tf]
    n_primary = len(df_primary)
    split_idx = int(n_primary * (1.0 - holdout_ratio))
    split_time = df_primary["time"].iloc[split_idx]

    train_dfs = {}
    holdout_dfs = {}

    for tf, df in dfs.items():
        train_dfs[tf] = df[df["time"] < split_time].copy().reset_index(drop=True)
        holdout_dfs[tf] = df[df["time"] >= split_time].copy().reset_index(drop=True)

    t_train_start = datetime.datetime.fromtimestamp(train_dfs[primary_tf]["time"].iloc[0], tz=datetime.timezone.utc)
    t_train_end = datetime.datetime.fromtimestamp(train_dfs[primary_tf]["time"].iloc[-1], tz=datetime.timezone.utc)
    t_hold_start = datetime.datetime.fromtimestamp(holdout_dfs[primary_tf]["time"].iloc[0], tz=datetime.timezone.utc)
    t_hold_end = datetime.datetime.fromtimestamp(holdout_dfs[primary_tf]["time"].iloc[-1], tz=datetime.timezone.utc)

    logger.info(f"Chronological split created:")
    logger.info(f"  TRAIN/VALIDATE: {t_train_start} to {t_train_end} ({len(train_dfs[primary_tf])} 5m bars)")
    logger.info(f"  HOLD-OUT TEST:  {t_hold_start} to {t_hold_end} ({len(holdout_dfs[primary_tf])} 5m bars, locked)")

    return train_dfs, holdout_dfs


def generate_walk_forward_folds(
    train_dfs: Dict[str, pd.DataFrame],
    train_bars: int = 1500,  # ~15 trading days
    test_bars: int = 400,    # ~4 trading days
    step_bars: int = 400,
    warmup_bars: int = 300,  # ~25 hours of warmup so indicators are fully initialized
) -> List[Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], float]]:
    """
    Generate rolling walk-forward folds on the TRAIN/VALIDATE partition.
    Includes causal warmup window for test folds so indicators (1H EMA50, ADX) are fully initialized.
    Never touches holdout data.
    """
    primary_df = train_dfs["5m"]
    total_bars = len(primary_df)
    folds = []

    start = 0
    while start + train_bars + test_bars <= total_bars:
        train_end = start + train_bars
        test_end = train_end + test_bars

        train_t_start = primary_df["time"].iloc[start]
        train_t_end = primary_df["time"].iloc[train_end - 1]
        test_t_start = primary_df["time"].iloc[train_end]
        test_t_end = primary_df["time"].iloc[test_end - 1]

        warmup_idx = max(0, train_end - warmup_bars)
        warmup_t_start = primary_df["time"].iloc[warmup_idx]

        fold_train = {}
        fold_test = {}
        for tf, df in train_dfs.items():
            fold_train[tf] = df[(df["time"] >= train_t_start) & (df["time"] <= train_t_end)].copy().reset_index(drop=True)
            fold_test[tf] = df[(df["time"] >= warmup_t_start) & (df["time"] <= test_t_end)].copy().reset_index(drop=True)

        folds.append((fold_train, fold_test, float(test_t_start)))
        start += step_bars

    logger.info(f"Generated {len(folds)} rolling walk-forward folds.")
    return folds


if __name__ == "__main__":
    logger.info("Initializing ApexGold data pipeline...")
    dfs_2018 = fetch_2018_baseline_data()
    train_dfs, holdout_dfs = split_train_holdout(dfs_2018)
    folds = generate_walk_forward_folds(train_dfs)
    print(f"Data pipeline ready! 2018 dataset loaded: {len(dfs_2018['5m'])} 5m bars, {len(folds)} walk-forward folds.")

