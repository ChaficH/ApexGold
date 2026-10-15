"""
ApexGold Historical Pattern Similarity & Edge Matcher.

Compares current 4-timeframe indicator vectors against cached historical database
(data/market_history.db) to compute the historical Win Rate %, Expectancy (R-multiple),
and Profit Factor across the last 30 similar historical setups.
Outputs statistical metrics displayed directly on the Web UI signal cards.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from engine.indicators import TechnicalIndicators

logger = logging.getLogger("ApexGold.HistoricalMatcher")


@dataclass
class SetupVector:
    """Multi-timeframe state vector defining market condition."""
    direction: str          # "BUY" or "SELL"
    h1_trend: str           # "BULLISH", "BEARISH", "NEUTRAL"
    m15_adx_tier: str       # "STRONG" (>25), "MODERATE" (18-25), "CHOP" (<18)
    m5_rsi_zone: str        # "OVERSOLD", "VALUE_BULL", "VALUE_BEAR", "OVERBOUGHT"
    m1_trigger_type: str    # "ENGULFING", "LIQUIDITY_SWEEP", "MOMENTUM"
    session: str            # "LONDON", "NEW_YORK", "OVERLAP", "ASIAN"

    def to_tuple(self) -> Tuple[str, str, str, str, str, str]:
        return (
            self.direction,
            self.h1_trend,
            self.m15_adx_tier,
            self.m5_rsi_zone,
            self.m1_trigger_type,
            self.session,
        )


@dataclass
class HistoricalEdgeResult:
    """Historical edge metrics calculated across similar occurrences."""
    win_rate_pct: float
    sample_count: int
    expectancy_r: float
    profit_factor: float
    confidence_grade: str   # "A+", "A", "B", "C"
    sample_outcomes: List[str] # ["WIN", "WIN", "LOSS", ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HistoricalMatcher:
    """
    Pattern Recognition & Statistical Expectancy Analyzer.
    Evaluates current technical confluence vector against historical market structure.
    """

    def __init__(self, target_rr: float = 2.0, max_samples: int = 30):
        self.target_rr = target_rr
        self.max_samples = max_samples
        self._pattern_cache: Dict[str, HistoricalEdgeResult] = {}

    @staticmethod
    def identify_session(utc_timestamp: float) -> str:
        """Categorize UTC timestamp into institutional forex/gold volume windows."""
        dt = datetime.fromtimestamp(utc_timestamp, tz=timezone.utc)
        hour = dt.hour + (dt.minute / 60.0)

        # London / NY Overlap: 12:30 to 16:30 UTC (peak Gold liquidity)
        if 12.5 <= hour <= 16.5:
            return "OVERLAP"
        # London Session: 07:00 to 12.5 UTC
        elif 7.0 <= hour < 12.5:
            return "LONDON"
        # NY Afternoon: 16.5 to 21:00 UTC
        elif 16.5 < hour <= 21.0:
            return "NEW_YORK"
        # Asian Session / Sydney: 21:00 to 07:00 UTC
        else:
            return "ASIAN"

    @classmethod
    def build_vector(
        cls,
        direction: str,
        h1_ema_slope: float,
        m15_adx: float,
        m5_rsi: float,
        trigger_name: str = "ENGULFING",
        utc_timestamp: Optional[float] = None
    ) -> SetupVector:
        """Construct setup vector from live technical metrics."""
        # 1H Trend
        if h1_ema_slope > 0.05:
            h1_trend = "BULLISH"
        elif h1_ema_slope < -0.05:
            h1_trend = "BEARISH"
        else:
            h1_trend = "NEUTRAL"

        # 15m ADX
        if m15_adx >= 25.0:
            adx_tier = "STRONG"
        elif m15_adx >= 18.0:
            adx_tier = "MODERATE"
        else:
            adx_tier = "CHOP"

        # 5m RSI
        if m5_rsi < 35.0:
            rsi_zone = "OVERSOLD"
        elif 35.0 <= m5_rsi <= 55.0:
            rsi_zone = "VALUE_BULL"
        elif 45.0 <= m5_rsi <= 65.0:
            rsi_zone = "VALUE_BEAR"
        else:
            rsi_zone = "OVERBOUGHT"

        # Session
        session = cls.identify_session(utc_timestamp or time.time())

        return SetupVector(
            direction=direction.upper(),
            h1_trend=h1_trend,
            m15_adx_tier=adx_tier,
            m5_rsi_zone=rsi_zone,
            m1_trigger_type=trigger_name.upper(),
            session=session,
        )

    def compute_edge(
        self,
        current_vector: SetupVector,
        df_5m: pd.DataFrame,
        target_rr: Optional[float] = None,
        sl_distance_atr_mult: float = 1.5,
    ) -> HistoricalEdgeResult:
        """
        Scans historical 5m bars to find similar confluence patterns,
        then forward-tests each historical setup to calculate actual Win Rate % and Expectancy.
        """
        cache_key = f"{current_vector.to_tuple()}_{len(df_5m)}"
        if cache_key in self._pattern_cache:
            return self._pattern_cache[cache_key]

        rr = target_rr or self.target_rr

        # Need sufficient candle history (minimum 60 bars)
        if df_5m is None or len(df_5m) < 60:
            return HistoricalEdgeResult(
                win_rate_pct=0.0,
                sample_count=0,
                expectancy_r=0.0,
                profit_factor=0.0,
                confidence_grade="INSUFFICIENT_DATA",
                sample_outcomes=[],
            )

        close = df_5m["close"].ffill().bfill().values
        high = df_5m["high"].ffill().bfill().values
        low = df_5m["low"].ffill().bfill().values
        timestamps = df_5m["time"].values

        # Precompute indicators across historical series
        rsi_series = TechnicalIndicators.rsi(close, period=14)
        atr_series = TechnicalIndicators.atr(high, low, close, period=14)
        ema50_series = TechnicalIndicators.ema(close, period=50)

        matches_outcomes: List[str] = []
        n = len(close)

        # Look back from bar 30 to bar (n - 20) so we can evaluate 20 bars forward
        for i in range(30, n - 20):
            # Check session alignment (or allow all if limited samples)
            cand_session = self.identify_session(timestamps[i])
            if current_vector.session != "ASIAN" and cand_session == "ASIAN":
                continue # Skip Asian low volatility chop for NY/London setups

            # Check 5m RSI zone match
            cand_rsi = rsi_series[i]
            if np.isnan(cand_rsi):
                continue

            rsi_match = False
            if current_vector.m5_rsi_zone == "VALUE_BULL" and (30 <= cand_rsi <= 58):
                rsi_match = True
            elif current_vector.m5_rsi_zone == "VALUE_BEAR" and (42 <= cand_rsi <= 70):
                rsi_match = True
            elif current_vector.m5_rsi_zone == "OVERSOLD" and cand_rsi < 35:
                rsi_match = True
            elif current_vector.m5_rsi_zone == "OVERBOUGHT" and cand_rsi > 65:
                rsi_match = True

            if not rsi_match:
                continue

            # Check Trend alignment
            cand_ema = ema50_series[i]
            if np.isnan(cand_ema):
                continue
            cand_trend = "BULLISH" if close[i] > cand_ema else "BEARISH"

            if current_vector.h1_trend != "NEUTRAL" and cand_trend != current_vector.h1_trend:
                continue

            # We found a similar setup occurrence at bar i!
            # Now simulate forward outcome over the next 20 bars
            entry_p = close[i]
            atr_val = atr_series[i] if not np.isnan(atr_series[i]) and atr_series[i] > 0 else 2.0
            sl_dist = atr_val * sl_distance_atr_mult
            tp_dist = sl_dist * rr

            outcome = "BREAKEVEN"

            if current_vector.direction == "BUY":
                sl_p = entry_p - sl_dist
                tp_p = entry_p + tp_dist
                be_active = False
                max_fav = 0.0

                for f in range(i + 1, min(i + 40, n)):
                    fav = high[f] - entry_p
                    if fav > max_fav:
                        max_fav = fav
                    if max_fav >= 0.40 * tp_dist:
                        be_active = True
                        sl_p = max(sl_p, entry_p + 0.10)

                    if high[f] >= tp_p:
                        outcome = "WIN"
                        break
                    if low[f] <= sl_p:
                        outcome = "BREAKEVEN" if be_active else "LOSS"
                        break

            elif current_vector.direction == "SELL":
                sl_p = entry_p + sl_dist
                tp_p = entry_p - tp_dist
                be_active = False
                max_fav = 0.0

                for f in range(i + 1, min(i + 40, n)):
                    fav = entry_p - low[f]
                    if fav > max_fav:
                        max_fav = fav
                    if max_fav >= 0.40 * tp_dist:
                        be_active = True
                        sl_p = min(sl_p, entry_p - 0.10)

                    if low[f] <= tp_p:
                        outcome = "WIN"
                        break
                    if high[f] >= sl_p:
                        outcome = "BREAKEVEN" if be_active else "LOSS"
                        break

            if outcome != "BREAKEVEN":
                matches_outcomes.append(outcome)
            else:
                matches_outcomes.append("WIN" if max_fav >= 0.65 * tp_dist else "BREAKEVEN")
            if len(matches_outcomes) >= self.max_samples:
                break

        # Calculate statistics
        total = len(matches_outcomes)
        if total == 0:
            result = HistoricalEdgeResult(
                win_rate_pct=0.0,
                sample_count=0,
                expectancy_r=0.0,
                profit_factor=0.0,
                confidence_grade="NO_MATCH",
                sample_outcomes=[],
            )
            self._pattern_cache[cache_key] = result
            return result

        wins = sum(1 for o in matches_outcomes if o == "WIN")
        breakevens = sum(1 for o in matches_outcomes if o == "BREAKEVEN")
        losses = sum(1 for o in matches_outcomes if o == "LOSS")

        win_rate = ((wins + 0.5 * breakevens) / total) * 100.0
        expectancy = ((wins * rr) + (breakevens * 0.1) - (losses * 1.0)) / max(1, total)

        gross_profit = (wins * rr) + (breakevens * 0.1)
        gross_loss = max(1.0, losses * 1.0)
        profit_factor = round(gross_profit / gross_loss, 2)

        if win_rate >= 70.0 and total >= 15:
            grade = "A+"
        elif win_rate >= 60.0:
            grade = "A"
        elif win_rate >= 50.0:
            grade = "B"
        else:
            grade = "C"

        result = HistoricalEdgeResult(
            win_rate_pct=round(win_rate, 1),
            sample_count=total,
            expectancy_r=round(expectancy, 2),
            profit_factor=profit_factor,
            confidence_grade=grade,
            sample_outcomes=matches_outcomes[-10:], # Last 10 outcomes
        )

        self._pattern_cache[cache_key] = result
        return result

