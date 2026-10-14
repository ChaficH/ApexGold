"""
ApexGold Technical Indicator Suite.

Implements high-performance multi-timeframe mathematical indicators including
EMA, RSI with hooking detection, ADX/DMI momentum, ATR volatility,
Swing High/Low detection, Room-to-Run structural analysis, and
1-minute sniper trigger pattern recognition (Engulfing & Liquidity Sweeps).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class TechnicalIndicators:
    """
    Mathematical and Price-Action Indicator Engine for ApexGold.
    All functions accept NumPy arrays, Pandas Series, or lists of float prices.
    """

    @staticmethod
    def _to_numpy(data: Union[pd.Series, np.ndarray, list]) -> np.ndarray:
        """Helper to convert inputs reliably into 1D float64 NumPy array."""
        if isinstance(data, pd.Series):
            return data.values.astype(np.float64)
        if isinstance(data, list):
            return np.array(data, dtype=np.float64)
        if isinstance(data, np.ndarray):
            return data.astype(np.float64)
        return np.array([float(x) for x in data], dtype=np.float64)

    @classmethod
    def ema(cls, data: Union[pd.Series, np.ndarray, list], period: int) -> np.ndarray:
        """
        Calculate Exponential Moving Average (EMA).
        """
        arr = cls._to_numpy(data)
        n = len(arr)
        if n < period or period <= 0:
            return np.full(n, np.nan)

        ema_vals = np.empty(n, dtype=np.float64)
        ema_vals[:period - 1] = np.nan
        # Seed first EMA with SMA
        ema_vals[period - 1] = np.mean(arr[:period])
        multiplier = 2.0 / (period + 1.0)

        for i in range(period, n):
            ema_vals[i] = (arr[i] - ema_vals[i - 1]) * multiplier + ema_vals[i - 1]

        return ema_vals

    @classmethod
    def ema_slope(cls, data: Union[pd.Series, np.ndarray, list], period: int, lookback: int = 3) -> float:
        """
        Compute the slope (directional momentum) of an EMA over recent bars.
        Positive = upward slope, Negative = downward slope.
        """
        ema_series = cls.ema(data, period)
        valid = ema_series[~np.isnan(ema_series)]
        if len(valid) < lookback + 1:
            return 0.0

        delta = valid[-1] - valid[-1 - lookback]
        # Return slope normalized per bar
        return float(delta / lookback)

    @classmethod
    def rsi(cls, close: Union[pd.Series, np.ndarray, list], period: int = 14) -> np.ndarray:
        """
        Wilder's Smoothed Relative Strength Index (RSI).
        """
        arr = cls._to_numpy(close)
        n = len(arr)
        if n <= period or period <= 0:
            return np.full(n, 50.0)

        deltas = np.diff(arr)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        rsi_vals = np.empty(n, dtype=np.float64)
        rsi_vals[:period] = np.nan

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        if avg_loss == 0.0:
            rsi_vals[period] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_vals[period] = 100.0 - (100.0 / (1.0 + rs))

        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss == 0.0:
                rsi_vals[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_vals[i + 1] = 100.0 - (100.0 / (1.0 + rs))

        return rsi_vals

    @classmethod
    def detect_rsi_hook(cls, rsi_series: np.ndarray) -> Dict[str, bool]:
        """
        Detects if RSI is hooking in trade direction:
        - Bullish hook: RSI was descending and has turned upwards below 55
        - Bearish hook: RSI was ascending and has turned downwards above 45
        """
        valid = rsi_series[~np.isnan(rsi_series)]
        if len(valid) < 3:
            return {"bullish_hook": False, "bearish_hook": False}

        r2, r1, r0 = valid[-3], valid[-2], valid[-1]

        bullish = (r1 <= r2) and (r0 > r1) and (r0 < 60.0)
        bearish = (r1 >= r2) and (r0 < r1) and (r0 > 40.0)

        return {"bullish_hook": bool(bullish), "bearish_hook": bool(bearish)}

    @classmethod
    def atr(
        cls,
        high: Union[pd.Series, np.ndarray, list],
        low: Union[pd.Series, np.ndarray, list],
        close: Union[pd.Series, np.ndarray, list],
        period: int = 14
    ) -> np.ndarray:
        """
        Average True Range (ATR) Wilder's smoothing.
        """
        h = cls._to_numpy(high)
        l = cls._to_numpy(low)
        c = cls._to_numpy(close)
        n = len(c)
        if n < period or period <= 0:
            return np.full(n, 0.0)

        tr = np.empty(n, dtype=np.float64)
        tr[0] = h[0] - l[0]
        for i in range(1, n):
            hl = h[i] - l[i]
            hc = abs(h[i] - c[i - 1])
            lc = abs(l[i] - c[i - 1])
            tr[i] = max(hl, hc, lc)

        atr_vals = np.empty(n, dtype=np.float64)
        atr_vals[:period - 1] = np.nan
        atr_vals[period - 1] = np.mean(tr[:period])

        for i in range(period, n):
            atr_vals[i] = (atr_vals[i - 1] * (period - 1) + tr[i]) / period

        return atr_vals

    @classmethod
    def adx(
        cls,
        high: Union[pd.Series, np.ndarray, list],
        low: Union[pd.Series, np.ndarray, list],
        close: Union[pd.Series, np.ndarray, list],
        period: int = 14
    ) -> Dict[str, np.ndarray]:
        """
        Average Directional Index (ADX) along with +DI and -DI.
        """
        h = cls._to_numpy(high)
        l = cls._to_numpy(low)
        c = cls._to_numpy(close)
        n = len(c)

        if n <= period:
            return {
                "adx": np.full(n, 20.0),
                "plus_di": np.full(n, 20.0),
                "minus_di": np.full(n, 20.0),
            }

        up_move = h[1:] - h[:-1]
        down_move = l[:-1] - l[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr = np.empty(n - 1, dtype=np.float64)
        for i in range(n - 1):
            hl = h[i + 1] - l[i + 1]
            hc = abs(h[i + 1] - c[i])
            lc = abs(l[i + 1] - c[i])
            tr[i] = max(hl, hc, lc)

        tr_smooth = np.zeros(n - 1)
        pdm_smooth = np.zeros(n - 1)
        mdm_smooth = np.zeros(n - 1)

        tr_smooth[period - 1] = np.sum(tr[:period])
        pdm_smooth[period - 1] = np.sum(plus_dm[:period])
        mdm_smooth[period - 1] = np.sum(minus_dm[:period])

        for i in range(period, n - 1):
            tr_smooth[i] = tr_smooth[i - 1] - (tr_smooth[i - 1] / period) + tr[i]
            pdm_smooth[i] = pdm_smooth[i - 1] - (pdm_smooth[i - 1] / period) + plus_dm[i]
            mdm_smooth[i] = mdm_smooth[i - 1] - (mdm_smooth[i - 1] / period) + minus_dm[i]

        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        dx = np.zeros(n - 1)

        for i in range(period - 1, n - 1):
            denom = tr_smooth[i] if tr_smooth[i] > 0 else 1e-9
            p_di = 100.0 * (pdm_smooth[i] / denom)
            m_di = 100.0 * (mdm_smooth[i] / denom)
            plus_di[i + 1] = p_di
            minus_di[i + 1] = m_di

            di_sum = p_di + m_di
            dx[i] = 100.0 * (abs(p_di - m_di) / di_sum) if di_sum > 0 else 0.0

        adx_vals = np.zeros(n)
        if (n - 1) >= (2 * period - 1):
            adx_start_idx = 2 * period - 2
            adx_vals[adx_start_idx + 1] = np.mean(dx[period - 1:adx_start_idx + 1])
            for i in range(adx_start_idx + 1, n - 1):
                adx_vals[i + 1] = (adx_vals[i] * (period - 1) + dx[i]) / period
        else:
            adx_vals[:] = 20.0

        return {
            "adx": adx_vals,
            "plus_di": plus_di,
            "minus_di": minus_di,
        }

    @classmethod
    def bollinger_bands(
        cls,
        close: Union[pd.Series, np.ndarray, list],
        period: int = 20,
        num_std: float = 2.0
    ) -> Dict[str, np.ndarray]:
        """
        Bollinger Bands calculation.
        """
        arr = cls._to_numpy(close)
        n = len(arr)
        if n < period:
            return {"middle": arr, "upper": arr, "lower": arr}

        series = pd.Series(arr)
        middle = series.rolling(period).mean().values
        std = series.rolling(period).std().values
        upper = middle + (num_std * std)
        lower = middle - (num_std * std)

        # Fill NaNs with current values
        middle = np.nan_to_num(middle, nan=arr[0])
        upper = np.nan_to_num(upper, nan=arr[0])
        lower = np.nan_to_num(lower, nan=arr[0])

        return {"middle": middle, "upper": upper, "lower": lower}

    @classmethod
    def vwap(
        cls,
        high: Union[pd.Series, np.ndarray, list],
        low: Union[pd.Series, np.ndarray, list],
        close: Union[pd.Series, np.ndarray, list],
        volume: Union[pd.Series, np.ndarray, list]
    ) -> np.ndarray:
        """
        Volume-Weighted Average Price (VWAP).
        """
        h = cls._to_numpy(high)
        l = cls._to_numpy(low)
        c = cls._to_numpy(close)
        v = cls._to_numpy(volume)
        # Avoid zero division
        v = np.where(v <= 0, 1.0, v)

        typical_price = (h + l + c) / 3.0
        cum_pv = np.cumsum(typical_price * v)
        cum_v = np.cumsum(v)
        return cum_pv / cum_v

    @classmethod
    def find_swings(
        cls,
        high: Union[pd.Series, np.ndarray, list],
        low: Union[pd.Series, np.ndarray, list],
        window: int = 5
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        Identifies structural swing highs and lows (fractal pivot points).
        A swing high has the highest price within window bars before and after.
        A swing low has the lowest price within window bars before and after.
        """
        h = cls._to_numpy(high)
        l = cls._to_numpy(low)
        n = len(h)

        swing_highs: List[Dict[str, float]] = []
        swing_lows: List[Dict[str, float]] = []

        if n < (2 * window + 1):
            return {"highs": swing_highs, "lows": swing_lows}

        for i in range(window, n - window):
            curr_h = h[i]
            is_high = True
            for w in range(1, window + 1):
                if h[i - w] >= curr_h or h[i + w] > curr_h:
                    is_high = False
                    break
            if is_high:
                swing_highs.append({"index": float(i), "price": round(float(curr_h), 4)})

            curr_l = l[i]
            is_low = True
            for w in range(1, window + 1):
                if l[i - w] <= curr_l or l[i + w] < curr_l:
                    is_low = False
                    break
            if is_low:
                swing_lows.append({"index": float(i), "price": round(float(curr_l), 4)})

        return {"highs": swing_highs, "lows": swing_lows}

    @classmethod
    def check_room_to_run(
        cls,
        entry_price: float,
        trade_direction: str,
        target_r_distance: float,
        swing_highs: List[Dict[str, float]],
        swing_lows: List[Dict[str, float]],
        safety_buffer: float = 0.20
    ) -> bool:
        """
        Evaluates 1H Macro "Room to Run".
        Returns True if no opposing major swing block exists before target distance.
        - BUY: Check if opposing swing high is between entry and (entry + target_r_dist - buffer)
        - SELL: Check if opposing swing low is between entry and (entry - target_r_dist + buffer)
        """
        direction = trade_direction.upper()

        if direction == "BUY":
            target_price = entry_price + target_r_distance
            for sh in swing_highs[-6:]:
                p = sh["price"]
                if p <= entry_price:
                    continue
                # Block only if an unbroken resistance ceiling sits directly in front of entry (< 60% of target)
                if entry_price < p <= (entry_price + target_r_distance * 0.60):
                    return False
            return True

        elif direction == "SELL":
            target_price = entry_price - target_r_distance
            for sl in swing_lows[-6:]:
                p = sl["price"]
                if p >= entry_price:
                    continue
                # Block only if an unbroken support floor sits directly in front of entry (< 60% of target)
                if (entry_price - target_r_distance * 0.60) <= p < entry_price:
                    return False
            return True

        return True

    @classmethod
    def detect_candle_trigger(
        cls,
        open_p: float,
        high_p: float,
        low_p: float,
        close_p: float,
        prev_open: float,
        prev_high: float,
        prev_low: float,
        prev_close: float,
        recent_swing_high: Optional[float] = None,
        recent_swing_low: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates 1-Minute Sniper Trigger Patterns:
        - Bullish / Bearish Engulfing
        - Liquidity Sweep / Pinbar Rejection (sweeping prior 5m high/low and rejecting)
        - Candle Body Momentum (> 55% of total candle range)
        """
        total_range = high_p - low_p
        if total_range <= 0:
            total_range = 1e-5

        body_size = abs(close_p - open_p)
        body_pct = (body_size / total_range) * 100.0

        is_bullish_close = close_p > open_p
        is_bearish_close = close_p < open_p

        # 1. Engulfing
        prev_body_size = abs(prev_close - prev_open)
        bullish_engulfing = (
            is_bullish_close and
            (prev_close <= prev_open) and
            (close_p >= prev_open) and
            (open_p <= prev_close) and
            (body_size > prev_body_size * 1.05)
        )

        bearish_engulfing = (
            is_bearish_close and
            (prev_close >= prev_open) and
            (close_p <= prev_open) and
            (open_p >= prev_close) and
            (body_size > prev_body_size * 1.05)
        )

        # 2. Liquidity Sweep & Pinbar Rejection
        upper_wick = high_p - max(open_p, close_p)
        lower_wick = min(open_p, close_p) - low_p

        # Bullish Sweep: Low swept below recent swing low, but rejected with long lower wick
        bullish_sweep = False
        if recent_swing_low is not None and low_p < recent_swing_low and close_p > recent_swing_low:
            if lower_wick >= (1.5 * body_size):
                bullish_sweep = True
        elif lower_wick >= (2.0 * body_size) and is_bullish_close:
            bullish_sweep = True

        # Bearish Sweep: High swept above recent swing high, but rejected with long upper wick
        bearish_sweep = False
        if recent_swing_high is not None and high_p > recent_swing_high and close_p < recent_swing_high:
            if upper_wick >= (1.5 * body_size):
                bearish_sweep = True
        elif upper_wick >= (2.0 * body_size) and is_bearish_close:
            bearish_sweep = True

        strong_momentum = body_pct >= 55.0

        return {
            "body_pct": round(body_pct, 2),
            "strong_momentum": strong_momentum,
            "bullish_engulfing": bullish_engulfing,
            "bearish_engulfing": bearish_engulfing,
            "bullish_sweep": bullish_sweep,
            "bearish_sweep": bearish_sweep,
            "bullish_trigger": bool(bullish_engulfing or bullish_sweep),
            "bearish_trigger": bool(bearish_engulfing or bearish_sweep),
        }

    @classmethod
    def detect_smc(cls, df: pd.DataFrame, swing_length: int = 5) -> Dict[str, Any]:
        """
        Compute Institutional Smart Money Concepts (SMC) via smartmoneyconcepts:
        - Fair Value Gaps (FVG): Imbalance zones
        - Institutional Order Blocks (OB): Supply / Demand institutional footprint
        - Break of Structure (BOS): Trend continuation
        - Change of Character (CHOCH): Early institutional trend reversal
        """
        result = {
            "bullish_fvg": False,
            "bearish_fvg": False,
            "unmitigated_bullish_ob": False,
            "unmitigated_bearish_ob": False,
            "recent_bullish_bos": False,
            "recent_bearish_bos": False,
            "recent_bullish_choch": False,
            "recent_bearish_choch": False,
            "fvg_top": None,
            "fvg_bottom": None,
            "ob_top": None,
            "ob_bottom": None,
        }
        if df is None or len(df) < 25:
            return result

        try:
            from smartmoneyconcepts import smc

            # Clean dataframe columns for SMC
            df_smc = df[["open", "high", "low", "close", "volume"]].copy()
            df_smc.columns = [c.lower() for c in df_smc.columns]

            # 1. Swing Highs and Lows
            shl = smc.swing_highs_lows(df_smc, swing_length=min(swing_length, max(2, len(df_smc) // 5)))

            # 2. Fair Value Gaps (FVG)
            fvg = smc.fvg(df_smc)
            if fvg is not None and len(fvg) > 0 and "FVG" in fvg.columns:
                recent_fvg = fvg.iloc[-5:]
                active_fvgs = recent_fvg[recent_fvg["FVG"].notna()]
                if not active_fvgs.empty:
                    last_fvg_row = active_fvgs.iloc[-1]
                    fvg_val = last_fvg_row["FVG"]
                    if fvg_val == 1.0:
                        result["bullish_fvg"] = True
                        result["fvg_top"] = float(last_fvg_row.get("Top", 0.0))
                        result["fvg_bottom"] = float(last_fvg_row.get("Bottom", 0.0))
                    elif fvg_val == -1.0:
                        result["bearish_fvg"] = True
                        result["fvg_top"] = float(last_fvg_row.get("Top", 0.0))
                        result["fvg_bottom"] = float(last_fvg_row.get("Bottom", 0.0))

            # 3. Order Blocks (OB)
            ob = smc.ob(df_smc, shl)
            if ob is not None and len(ob) > 0 and "OB" in ob.columns:
                recent_ob = ob.iloc[-15:]
                active_obs = recent_ob[recent_ob["OB"].notna()]
                for _, row in active_obs.iterrows():
                    is_unmitigated = pd.isna(row.get("MitigatedIndex")) or row.get("MitigatedIndex") == 0.0
                    if is_unmitigated:
                        if row["OB"] == 1.0:
                            result["unmitigated_bullish_ob"] = True
                            result["ob_top"] = float(row.get("Top", 0.0))
                            result["ob_bottom"] = float(row.get("Bottom", 0.0))
                        elif row["OB"] == -1.0:
                            result["unmitigated_bearish_ob"] = True
                            result["ob_top"] = float(row.get("Top", 0.0))
                            result["ob_bottom"] = float(row.get("Bottom", 0.0))

            # 4. Break of Structure (BOS) & Change of Character (CHOCH)
            bos = smc.bos_choch(df_smc, shl)
            if bos is not None and len(bos) > 0:
                recent_bos = bos.iloc[-8:]
                if "BOS" in recent_bos.columns:
                    bos_active = recent_bos[recent_bos["BOS"].notna()]
                    if not bos_active.empty:
                        last_bos_val = bos_active.iloc[-1]["BOS"]
                        if last_bos_val == 1.0:
                            result["recent_bullish_bos"] = True
                        elif last_bos_val == -1.0:
                            result["recent_bearish_bos"] = True

                if "CHOCH" in recent_bos.columns:
                    choch_active = recent_bos[recent_bos["CHOCH"].notna()]
                    if not choch_active.empty:
                        last_choch_val = choch_active.iloc[-1]["CHOCH"]
                        if last_choch_val == 1.0:
                            result["recent_bullish_choch"] = True
                        elif last_choch_val == -1.0:
                            result["recent_bearish_choch"] = True

        except Exception as e:
            pass

        return result

    @classmethod
    def calculate_vwap(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Institutional Volume-Weighted Average Price (VWAP) using 'ta' library.
        Returns VWAP level, direction relative to price, and slope.
        """
        if df is None or len(df) < 5 or not all(c in df.columns for c in ["high", "low", "close", "volume"]):
            return {"vwap": 0.0, "is_above": False, "is_below": False, "slope": 0.0, "valid": False}
        try:
            import ta
            vwap_series = ta.volume.VolumeWeightedAveragePrice(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                volume=df["volume"].fillna(1.0).clip(lower=1.0),
                window=min(len(df), 40)
            ).volume_weighted_average_price()
            val = float(vwap_series.iloc[-1])
            prev_val = float(vwap_series.iloc[-2]) if len(vwap_series) >= 2 else val
            curr_c = float(df["close"].iloc[-1])
            slope = val - prev_val
            return {
                "vwap": round(val, 2),
                "is_above": (curr_c >= val),
                "is_below": (curr_c < val),
                "slope": slope,
                "valid": not np.isnan(val) and val > 0,
            }
        except Exception:
            return {"vwap": 0.0, "is_above": False, "is_below": False, "slope": 0.0, "valid": False}

    @classmethod
    def calculate_kama(cls, df: pd.DataFrame, window: int = 10) -> Dict[str, Any]:
        """
        Kaufman's Adaptive Moving Average (KAMA) using 'ta' library.
        Automatically slows down during chop to prevent whipsaws and speeds up on trends.
        """
        if df is None or len(df) < window + 2 or "close" not in df.columns:
            return {"kama": 0.0, "is_above": False, "is_below": False, "valid": False}
        try:
            import ta
            kama_series = ta.momentum.KAMAIndicator(
                close=df["close"],
                window=window,
                pow1=2,
                pow2=30
            ).kama()
            val = float(kama_series.iloc[-1])
            curr_c = float(df["close"].iloc[-1])
            return {
                "kama": round(val, 2),
                "is_above": (curr_c >= val),
                "is_below": (curr_c < val),
                "valid": not np.isnan(val) and val > 0,
            }
        except Exception:
            return {"kama": 0.0, "is_above": False, "is_below": False, "valid": False}

    @classmethod
    def calculate_statistical_trend(cls, close_prices: Union[pd.Series, np.ndarray, list], window: int = 15) -> Dict[str, Any]:
        """
        Statistical Trend Quality & R-Squared using scipy.stats.linregress.
        R^2 >= 0.60 indicates genuine institutional trend persistence.
        R^2 < 0.25 indicates high-entropy random walk (chop).
        """
        arr = cls._to_numpy(close_prices)
        if len(arr) < window:
            return {"r_squared": 0.0, "slope": 0.0, "is_strong_trend": False, "is_random_walk": False, "valid": False}
        try:
            from scipy import stats
            subset = arr[-window:]
            x = np.arange(len(subset))
            res = stats.linregress(x, subset)
            r2 = float(res.rvalue ** 2) if not np.isnan(res.rvalue) else 0.0
            return {
                "r_squared": round(r2, 3),
                "slope": float(res.slope),
                "p_value": float(res.pvalue),
                "is_strong_trend": (r2 >= 0.60),
                "is_random_walk": (r2 < 0.25),
                "valid": True,
            }
        except Exception:
            return {"r_squared": 0.0, "slope": 0.0, "is_strong_trend": False, "is_random_walk": False, "valid": False}

    @classmethod
    def calculate_zscore(cls, close_prices: Union[pd.Series, np.ndarray, list], window: int = 20) -> Dict[str, Any]:
        """
        Statistical Z-Score of Price vs Moving Mean.
        Filters out buying at extreme tops (Z > +2.2) and selling at extreme bottoms (Z < -2.2).
        Identifies high-probability discount/premium pullback zones.
        """
        arr = cls._to_numpy(close_prices)
        if len(arr) < window:
            return {"zscore": 0.0, "is_extreme_overbought": False, "is_extreme_oversold": False, "is_bull_discount": False, "is_bear_premium": False, "valid": False}
        try:
            subset = arr[-window:]
            mean_val = np.mean(subset)
            std_val = np.std(subset)
            if std_val < 1e-6:
                return {"zscore": 0.0, "is_extreme_overbought": False, "is_extreme_oversold": False, "is_bull_discount": False, "is_bear_premium": False, "valid": True}
            z = float((subset[-1] - mean_val) / std_val)
            return {
                "zscore": round(z, 2),
                "is_extreme_overbought": (z >= 2.2),
                "is_extreme_oversold": (z <= -2.2),
                "is_bull_discount": (0.1 <= z <= 0.9),
                "is_bear_premium": (-0.9 <= z <= -0.1),
                "valid": True,
            }
        except Exception:
            return {"zscore": 0.0, "is_extreme_overbought": False, "is_extreme_oversold": False, "is_bull_discount": False, "is_bear_premium": False, "valid": False}

    @classmethod
    def calculate_bb_squeeze(cls, df: pd.DataFrame, window: int = 20) -> Dict[str, Any]:
        """
        Bollinger Band Squeeze and Volatility Expansion using 'ta' library.
        """
        if df is None or len(df) < window + 2 or "close" not in df.columns:
            return {"is_squeeze": False, "bandwidth": 0.0, "valid": False}
        try:
            import ta
            bb = ta.volatility.BollingerBands(close=df["close"], window=window, window_dev=2)
            bw = bb.bollinger_wband()
            val = float(bw.iloc[-1])
            is_squeeze = (val < 0.012)
            return {
                "is_squeeze": is_squeeze,
                "bandwidth": round(val, 4),
                "valid": not np.isnan(val),
            }
        except Exception:
            return {"is_squeeze": False, "bandwidth": 0.0, "valid": False}

    @classmethod
    def calculate_supertrend(
        cls,
        high: Union[pd.Series, np.ndarray, list],
        low: Union[pd.Series, np.ndarray, list],
        close: Union[pd.Series, np.ndarray, list],
        period: int = 10,
        multiplier: float = 2.5
    ) -> Dict[str, Any]:
        """
        Battle-Tested Quantitative ATR SuperTrend Algorithm.
        Dynamically adapts bands to market volatility and generates clear institutional trend regimes.
        Returns: direction (+1 for BULL, -1 for BEAR), trend line, and boolean state flags.
        """
        h = cls._to_numpy(high)
        l = cls._to_numpy(low)
        c = cls._to_numpy(close)
        n = len(c)
        if n < period + 2:
            return {"direction": 0, "trend_name": "NEUTRAL", "line": 0.0, "is_bull": False, "is_bear": False, "valid": False}

        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        atr = np.zeros(n, dtype=np.float64)
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        hl2 = (h + l) / 2.0
        bu = hl2 + (multiplier * atr)
        bl = hl2 - (multiplier * atr)
        fu = np.zeros(n, dtype=np.float64)
        fl = np.zeros(n, dtype=np.float64)
        trd = np.zeros(n, dtype=int)

        for i in range(period, n):
            fu[i] = bu[i] if (bu[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
            fl[i] = bl[i] if (bl[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
            if c[i] > fu[i - 1]:
                trd[i] = 1
            elif c[i] < fl[i - 1]:
                trd[i] = -1
            else:
                trd[i] = trd[i - 1]

        d = int(trd[-1])
        line = float(fl[-1] if d == 1 else fu[-1])
        return {
            "direction": d,
            "trend_name": "BULL" if d == 1 else ("BEAR" if d == -1 else "NEUTRAL"),
            "line": round(line, 2),
            "is_bull": (d == 1),
            "is_bear": (d == -1),
            "valid": True
        }

    @classmethod
    def calculate_stoch_rsi(
        cls,
        close: Union[pd.Series, np.ndarray, list],
        period: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3
    ) -> Dict[str, Any]:
        """
        Stochastic RSI Oscillator using 'ta' library.
        Identifies overbought tops (> 80) and oversold bottoms (< 20) to prevent knife-catching.
        """
        try:
            import ta
            s = pd.Series(cls._to_numpy(close))
            if len(s) < period + smooth_k + smooth_d + 2:
                return {"k": 50.0, "d": 50.0, "is_overbought": False, "is_oversold": False, "bullish_cross": False, "bearish_cross": False, "valid": False}

            stoch = ta.momentum.StochRSIIndicator(close=s, window=period, smooth1=smooth_k, smooth2=smooth_d)
            k_s = stoch.stochrsi_k()
            d_s = stoch.stochrsi_d()
            k = float(k_s.iloc[-1] * 100.0)
            d = float(d_s.iloc[-1] * 100.0)
            prev_k = float(k_s.iloc[-2] * 100.0)
            prev_d = float(d_s.iloc[-2] * 100.0)

            if np.isnan(k) or np.isnan(d):
                return {"k": 50.0, "d": 50.0, "is_overbought": False, "is_oversold": False, "bullish_cross": False, "bearish_cross": False, "valid": False}

            return {
                "k": round(k, 1),
                "d": round(d, 1),
                "is_overbought": (k >= 80.0),
                "is_oversold": (k <= 20.0),
                "bullish_cross": (k > d and prev_k <= prev_d and k < 55.0),
                "bearish_cross": (k < d and prev_k >= prev_d and k > 45.0),
                "valid": True
            }
        except Exception:
            return {"k": 50.0, "d": 50.0, "is_overbought": False, "is_oversold": False, "bullish_cross": False, "bearish_cross": False, "valid": False}

    @classmethod
    def get_4tf_trends(cls, dfs: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """
        Calculate mathematical directional bias for 1H, 15m, 5m, and 1m using SuperTrend + Moving Averages.
        Returns: Dict mapping each timeframe to 'BULL', 'BEAR', or 'NEUTRAL'.
        """
        trends = {"1H": "NEUTRAL", "15m": "NEUTRAL", "5m": "NEUTRAL", "1m": "NEUTRAL"}

        # 1H Macro Bias
        df_1h = dfs.get("1H")
        if df_1h is not None and len(df_1h) >= 20 and "close" in df_1h.columns:
            st_1h = cls.calculate_supertrend(df_1h["high"], df_1h["low"], df_1h["close"], period=10, multiplier=2.5)
            c = df_1h["close"].values
            ema50 = cls.ema(c, min(50, len(c)))[-1]
            if st_1h["valid"]:
                if st_1h["is_bull"] and c[-1] >= ema50 * 0.998:
                    trends["1H"] = "BULL"
                elif st_1h["is_bear"] and c[-1] <= ema50 * 1.002:
                    trends["1H"] = "BEAR"
                else:
                    trends["1H"] = st_1h["trend_name"]

        # 15m Session Wave Bias
        df_15m = dfs.get("15m")
        if df_15m is not None and len(df_15m) >= 20 and "close" in df_15m.columns:
            st_15m = cls.calculate_supertrend(df_15m["high"], df_15m["low"], df_15m["close"], period=10, multiplier=2.5)
            c = df_15m["close"].values
            ema20 = cls.ema(c, 20)[-1]
            if st_15m["valid"]:
                if st_15m["is_bull"] and c[-1] >= ema20 * 0.999:
                    trends["15m"] = "BULL"
                elif st_15m["is_bear"] and c[-1] <= ema20 * 1.001:
                    trends["15m"] = "BEAR"
                else:
                    trends["15m"] = st_15m["trend_name"]

        # 5m Setup Bias
        df_5m = dfs.get("5m")
        if df_5m is not None and len(df_5m) >= 20 and "close" in df_5m.columns:
            st_5m = cls.calculate_supertrend(df_5m["high"], df_5m["low"], df_5m["close"], period=10, multiplier=2.5)
            if st_5m["valid"]:
                trends["5m"] = st_5m["trend_name"]

        # 1m Trigger Bias
        df_1m = dfs.get("1m")
        if df_1m is not None and len(df_1m) >= 10 and "close" in df_1m.columns:
            st_1m = cls.calculate_supertrend(df_1m["high"], df_1m["low"], df_1m["close"], period=10, multiplier=2.0)
            c = df_1m["close"].values
            o = df_1m["open"].values
            if st_1m["valid"]:
                if st_1m["is_bull"] and c[-1] >= o[-1]:
                    trends["1m"] = "BULL"
                elif st_1m["is_bear"] and c[-1] <= o[-1]:
                    trends["1m"] = "BEAR"
                else:
                    trends["1m"] = st_1m["trend_name"]

        return trends

    @classmethod
    def detect_supply_demand_zones(
        cls,
        df_5m: Optional[pd.DataFrame],
        df_15m: Optional[pd.DataFrame],
        current_price: float,
        is_gold: bool = True
    ) -> Dict[str, Any]:
        """
        Institutional Multi-Timeframe Supply & Demand Zone + Liquidity Sweep Engine.
        Accurately identifies:
        - Major Demand Zones (Support floors / resting buy liquidity)
        - Major Supply Zones (Resistance ceilings / resting sell liquidity)
        - Session Range Bounds (High, Low, Mid-Range Equilibrium)
        - Bullish / Bearish Liquidity Sweeps (springs / upthrusts)
        """
        default_res = {
            "range_high": current_price + (5.0 if is_gold else 0.0050),
            "range_low": current_price - (5.0 if is_gold else 0.0050),
            "range_span": 10.0 if is_gold else 0.0100,
            "mid_range": current_price,
            "demand_floor": current_price - (5.0 if is_gold else 0.0050),
            "demand_ceiling": current_price - (2.5 if is_gold else 0.0025),
            "supply_floor": current_price + (2.5 if is_gold else 0.0025),
            "supply_ceiling": current_price + (5.0 if is_gold else 0.0050),
            "in_demand_zone": False,
            "in_supply_zone": False,
            "at_major_support": False,
            "at_major_resistance": False,
            "liquidity_sweep_bull": False,
            "liquidity_sweep_bear": False,
            "valid": False,
        }

        if df_5m is None or len(df_5m) < 15 or not all(c in df_5m.columns for c in ["high", "low", "close", "open"]):
            return default_res

        try:
            # Active session window: last 36 to 48 bars (3 to 4 hours)
            window = min(len(df_5m), 48)
            h_5m = df_5m["high"].values[-window:]
            l_5m = df_5m["low"].values[-window:]
            c_5m = df_5m["close"].values[-window:]
            o_5m = df_5m["open"].values[-window:]

            r_high = float(np.max(h_5m))
            r_low = float(np.min(l_5m))

            # If 15m is available, cluster with 15m swings
            if df_15m is not None and len(df_15m) >= 12:
                w15 = min(len(df_15m), 24)
                r_high = max(r_high, float(np.max(df_15m["high"].values[-w15:])))
                r_low = min(r_low, float(np.min(df_15m["low"].values[-w15:])))

            span = max(r_high - r_low, 3.0 if is_gold else 0.0030)
            mid = (r_high + r_low) / 2.0

            # Demand Zone is the lower 25% of the range, or within $2.20 of the absolute low on Gold
            zone_depth = max(span * 0.25, 2.20 if is_gold else 0.0018)
            demand_fl = r_low
            demand_ceil = r_low + zone_depth

            # Supply Zone is the upper 25% of the range, or within $2.20 of the absolute high on Gold
            supply_ceil = r_high
            supply_fl = r_high - zone_depth

            in_demand = (current_price <= demand_ceil)
            in_supply = (current_price >= supply_fl)

            at_major_support = (current_price <= demand_fl + (1.20 if is_gold else 0.0008))
            at_major_resistance = (current_price >= supply_ceil - (1.20 if is_gold else 0.0008))

            # Detect Liquidity Sweeps
            # Bullish Sweep: Candle dipped into or below demand floor but closed higher with strong lower wick / green body
            is_bull_sweep = False
            curr_l = l_5m[-1]
            prev_l = l_5m[-2] if len(l_5m) >= 2 else curr_l
            curr_c = c_5m[-1]
            curr_o = o_5m[-1]
            lower_wick = min(curr_o, curr_c) - curr_l
            body = abs(curr_c - curr_o)

            if (curr_l <= demand_fl + 0.80 or prev_l <= demand_fl + 0.80) and (curr_c > demand_fl):
                if lower_wick >= 1.2 * max(0.15, body) or (curr_c > curr_o):
                    is_bull_sweep = True

            # Bearish Sweep: Candle spiked into or above supply ceiling but closed lower with strong upper wick / red body
            is_bear_sweep = False
            curr_h = h_5m[-1]
            prev_h = h_5m[-2] if len(h_5m) >= 2 else curr_h
            upper_wick = curr_h - max(curr_o, curr_c)

            if (curr_h >= supply_ceil - 0.80 or prev_h >= supply_ceil - 0.80) and (curr_c < supply_ceil):
                if upper_wick >= 1.2 * max(0.15, body) or (curr_c < curr_o):
                    is_bear_sweep = True

            return {
                "range_high": round(r_high, 2 if is_gold else 5),
                "range_low": round(r_low, 2 if is_gold else 5),
                "range_span": round(span, 2 if is_gold else 5),
                "mid_range": round(mid, 2 if is_gold else 5),
                "demand_floor": round(demand_fl, 2 if is_gold else 5),
                "demand_ceiling": round(demand_ceil, 2 if is_gold else 5),
                "supply_floor": round(supply_fl, 2 if is_gold else 5),
                "supply_ceiling": round(supply_ceil, 2 if is_gold else 5),
                "in_demand_zone": in_demand,
                "in_supply_zone": in_supply,
                "at_major_support": at_major_support,
                "at_major_resistance": at_major_resistance,
                "liquidity_sweep_bull": is_bull_sweep,
                "liquidity_sweep_bear": is_bear_sweep,
                "valid": True,
            }
        except Exception:
            return default_res





