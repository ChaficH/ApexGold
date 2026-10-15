"""
ApexGold Quantitative Ensemble Model Suite.

Implements high-frequency institutional sub-models:
1. OrderFlowImbalanceTracker: Microstructural Tick Delta & Institutional Absorption.
2. FairValueGapEngine: 3-Candle Institutional Liquidity Gaps & Imbalance Retests.
3. ParkinsonVolatilityClassifier: Extreme-Value Parkinson Volatility & Breakout Expansion.
4. EnsembleQuantSuite: Composite multi-model voting matrix that dynamically scales
   directional conviction, target R:R expansion, and lot sizing.
"""

from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np
import pandas as pd
import time
import logging

logger = logging.getLogger("ApexGold.EnsembleModels")


@dataclass
class TickRecord:
    timestamp: float
    bid: float
    ask: float
    price_change: float
    is_buyer_initiated: bool


@dataclass
class FVGZone:
    timeframe: str
    direction: str          # "BULL" or "BEAR"
    top: float
    bottom: float
    midpoint: float
    created_time: float
    is_mitigated: bool = False


@dataclass
class EnsembleTelemetry:
    timestamp: float
    ensemble_bias: str      # "STRONG_BULL", "BULL", "NEUTRAL", "BEAR", "STRONG_BEAR"
    ensemble_score: float   # 0 to 100
    delta_ratio_60s: float  # -1.0 to +1.0
    absorption_state: str   # "BULL_ABSORPTION", "BEAR_ABSORPTION", "NEUTRAL"
    nearest_fvg: Optional[Dict[str, Any]]
    volatility_regime: str  # "EXPANSIVE_VOLATILITY", "COMPRESSED_VOLATILITY", "STABLE_VOLATILITY"
    parkinson_ratio: float
    conviction_boost: float # Multiplier for sizing (1.0 to 1.35)
    rr_expansion_boost: float # Target RR expansion (+0.0R to +1.5R)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OrderFlowImbalanceTracker:
    """
    Tracks microstructural order flow by calculating tick delta
    (buyer-initiated ticks hitting ask vs seller-initiated ticks hitting bid).
    Detects institutional absorption at support and resistance.
    """

    def __init__(self, max_ticks: int = 600):
        self.max_ticks = max_ticks
        self._ticks: deque = deque(maxlen=max_ticks)
        self._last_bid: float = 0.0
        self._last_ask: float = 0.0

    def on_tick(self, bid: float, ask: float, now_ts: Optional[float] = None) -> None:
        """Register a new live tick and classify initiation pressure."""
        now = now_ts or time.time()
        price_change = 0.0
        is_buyer = True

        if self._last_bid > 0.0:
            mid = (bid + ask) / 2.0
            last_mid = (self._last_bid + self._last_ask) / 2.0
            price_change = mid - last_mid

            if price_change > 0:
                is_buyer = True
            elif price_change < 0:
                is_buyer = False
            else:
                # Up-tick rule: if price identical, classify against previous ask
                is_buyer = (ask >= self._last_ask)

        self._ticks.append(
            TickRecord(
                timestamp=now,
                bid=bid,
                ask=ask,
                price_change=price_change,
                is_buyer_initiated=is_buyer,
            )
        )
        self._last_bid = bid
        self._last_ask = ask

    def compute_metrics(self, window_sec: float = 60.0) -> Dict[str, Any]:
        """Compute tick delta and institutional absorption over sliding window."""
        if not self._ticks:
            return {
                "tick_count": 0,
                "delta": 0,
                "delta_ratio": 0.0,
                "absorption_state": "NEUTRAL",
            }

        now = self._ticks[-1].timestamp
        cutoff = now - window_sec
        relevant = [t for t in self._ticks if t.timestamp >= cutoff]

        if not relevant:
            return {
                "tick_count": 0,
                "delta": 0,
                "delta_ratio": 0.0,
                "absorption_state": "NEUTRAL",
            }

        buyer_ticks = sum(1 for t in relevant if t.is_buyer_initiated)
        seller_ticks = len(relevant) - buyer_ticks
        delta = buyer_ticks - seller_ticks
        total = len(relevant)
        delta_ratio = round(delta / total, 3) if total > 0 else 0.0

        net_price_move = relevant[-1].bid - relevant[0].bid

        # Absorption Detection
        absorption = "NEUTRAL"
        if delta_ratio >= 0.25 and net_price_move >= -0.20:
            absorption = "BULL_ABSORPTION"  # Buyers strongly dominating and supporting price
        elif delta_ratio <= -0.25 and net_price_move <= 0.20:
            absorption = "BEAR_ABSORPTION"  # Sellers strongly dominating and depressing price

        return {
            "tick_count": total,
            "buyer_ticks": buyer_ticks,
            "seller_ticks": seller_ticks,
            "delta": delta,
            "delta_ratio": delta_ratio,
            "absorption_state": absorption,
        }


class FairValueGapEngine:
    """
    Identifies 3-candle institutional Fair Value Gaps (FVG) and liquidity imbalances.
    Detects whether price is testing or reacting to an institutional supply/demand gap.
    """

    @staticmethod
    def detect_fvgs(df: pd.DataFrame, timeframe: str = "5m", max_lookback: int = 30) -> List[FVGZone]:
        """Find active unmitigated FVGs in the last N candles."""
        fvgs = []
        if df is None or len(df) < 5:
            return fvgs

        h = df["high"].values
        l = df["low"].values
        t = df["time"].values if "time" in df.columns else np.zeros(len(df))
        n = len(df)
        start_idx = max(2, n - max_lookback)

        for i in range(start_idx, n):
            # Bullish FVG: Candle i-2 High < Candle i Low (Gap between high[i-2] and low[i])
            if l[i] > h[i - 2]:
                gap_top = float(l[i])
                gap_bottom = float(h[i - 2])
                if gap_top - gap_bottom >= 0.30:  # Minimum 30 cents gap on Gold
                    # Check if subsequent candles mitigated this gap
                    is_mitigated = False
                    for j in range(i + 1, n):
                        if l[j] <= gap_bottom:
                            is_mitigated = True
                            break
                    if not is_mitigated:
                        fvgs.append(
                            FVGZone(
                                timeframe=timeframe,
                                direction="BULL",
                                top=round(gap_top, 2),
                                bottom=round(gap_bottom, 2),
                                midpoint=round((gap_top + gap_bottom) / 2.0, 2),
                                created_time=float(t[i]) if (len(t) > i and t[i] != 0) else float(time.time()),
                                is_mitigated=False,
                            )
                        )

            # Bearish FVG: Candle i-2 Low > Candle i High (Gap between low[i-2] and high[i])
            elif h[i] < l[i - 2]:
                gap_top = float(l[i - 2])
                gap_bottom = float(h[i])
                if gap_top - gap_bottom >= 0.30:
                    is_mitigated = False
                    for j in range(i + 1, n):
                        if h[j] >= gap_top:
                            is_mitigated = True
                            break
                    if not is_mitigated:
                        fvgs.append(
                            FVGZone(
                                timeframe=timeframe,
                                direction="BEAR",
                                top=round(gap_top, 2),
                                bottom=round(gap_bottom, 2),
                                midpoint=round((gap_top + gap_bottom) / 2.0, 2),
                                created_time=float(t[i]) if (len(t) > i and t[i] != 0) else float(time.time()),
                                is_mitigated=False,
                            )
                        )

        return fvgs

    @staticmethod
    def evaluate_price_against_fvgs(current_price: float, fvgs: List[FVGZone]) -> Optional[Dict[str, Any]]:
        """Find the nearest relevant FVG and determine if price is testing it."""
        if not fvgs:
            return None

        nearest = None
        min_dist = float("inf")

        for fvg in fvgs:
            dist = abs(current_price - fvg.midpoint)
            if dist < min_dist:
                min_dist = dist
                nearest = fvg

        if nearest is None:
            return None

        is_inside = (nearest.bottom <= current_price <= nearest.top)
        return {
            "direction": nearest.direction,
            "top": nearest.top,
            "bottom": nearest.bottom,
            "midpoint": nearest.midpoint,
            "distance": round(min_dist, 2),
            "is_inside": is_inside,
            "timeframe": nearest.timeframe,
        }


class ParkinsonVolatilityClassifier:
    """
    Computes Extreme-Value Parkinson Volatility over high-low candle ranges:
    sigma_P = sqrt( 1 / (4 * ln(2) * N) * sum( ln(H_i / L_i)^2 ) )
    Detects volatility breakouts and compression regimes with mathematical rigor.
    """

    @staticmethod
    def calculate(df: pd.DataFrame, window: int = 14) -> Dict[str, Any]:
        """Calculate Parkinson volatility and classify current volatility regime."""
        if df is None or len(df) < window + 5:
            return {
                "parkinson_vol": 0.005,
                "parkinson_ratio": 1.0,
                "regime": "STABLE_VOLATILITY",
            }

        h = df["high"].values
        l = df["low"].values

        # Ensure valid positive prices
        valid = (h > 0) & (l > 0) & (h >= l)
        h = h[valid]
        l = l[valid]

        if len(h) < window + 5:
            return {
                "parkinson_vol": 0.005,
                "parkinson_ratio": 1.0,
                "regime": "STABLE_VOLATILITY",
            }

        # Avoid div by zero or negative logs
        ratios = np.maximum(h / np.maximum(l, 1e-6), 1.00001)
        log_hl_sq = np.log(ratios) ** 2

        factor = 1.0 / (4.0 * math.log(2.0))
        rolling_parkinson = np.sqrt(factor * np.convolve(log_hl_sq, np.ones(window) / window, mode="valid"))

        if len(rolling_parkinson) < 2:
            return {
                "parkinson_vol": 0.005,
                "parkinson_ratio": 1.0,
                "regime": "STABLE_VOLATILITY",
                "volatility_regime": "STABLE_VOLATILITY",
            }

        cur_vol = float(rolling_parkinson[-1])
        baseline_vol = float(np.mean(rolling_parkinson[-min(20, len(rolling_parkinson)):]))
        ratio = round(cur_vol / max(baseline_vol, 1e-6), 2)

        if ratio >= 1.35:
            regime = "EXPANSIVE_VOLATILITY"  # High momentum impulse
        elif ratio <= 0.70:
            regime = "COMPRESSED_VOLATILITY" # Coiling consolidation
        else:
            regime = "STABLE_VOLATILITY"

        return {
            "parkinson_vol": round(cur_vol, 5),
            "parkinson_ratio": ratio,
            "regime": regime,
            "volatility_regime": regime,
        }


class EnsembleQuantSuite:
    """
    Master Ensemble Architecture:
    Fuses Order Flow Imbalance, Fair Value Gap Support, and Parkinson Volatility
    with Multi-Timeframe Trend Consensus.
    """

    def __init__(self):
        self.flow_tracker = OrderFlowImbalanceTracker(max_ticks=600)
        self.fvg_engine = FairValueGapEngine()
        self.vol_classifier = ParkinsonVolatilityClassifier()
        self.latest_telemetry: Optional[EnsembleTelemetry] = None

    def on_tick(self, bid: float, ask: float, now_ts: Optional[float] = None) -> None:
        """Register live tick to order flow tracker."""
        self.flow_tracker.on_tick(bid, ask, now_ts)

    def evaluate_ensemble(
        self,
        symbol: str,
        dfs: Dict[str, pd.DataFrame],
        current_bid: float,
        macro_1h: str,
        macro_15m: str,
        macro_5m: str,
        kaufman_er: float,
    ) -> EnsembleTelemetry:
        """Evaluate ensemble sub-models and compute composite conviction scores."""
        now = time.time()

        # 1. Order Flow Metrics
        flow_metrics = self.flow_tracker.compute_metrics(window_sec=60.0)
        delta_ratio = flow_metrics["delta_ratio"]
        absorption = flow_metrics["absorption_state"]

        # 2. Fair Value Gaps
        df_5m = dfs.get("5m")
        df_15m = dfs.get("15m")
        fvgs_5m = self.fvg_engine.detect_fvgs(df_5m, timeframe="5m", max_lookback=25)
        fvgs_15m = self.fvg_engine.detect_fvgs(df_15m, timeframe="15m", max_lookback=25)
        all_fvgs = fvgs_5m + fvgs_15m
        fvg_info = self.fvg_engine.evaluate_price_against_fvgs(current_bid, all_fvgs)

        # 3. Parkinson Volatility
        vol_info = self.vol_classifier.calculate(df_5m, window=14)
        vol_regime = vol_info["regime"]
        vol_ratio = vol_info["parkinson_ratio"]

        # 4. Composite Scoring
        score_bull = 0.0
        score_bear = 0.0

        # Macro Trend Voting (Weight: 40%)
        if macro_1h == "BULL": score_bull += 20.0
        elif macro_1h == "BEAR": score_bear += 20.0

        if macro_15m == "BULL": score_bull += 15.0
        elif macro_15m == "BEAR": score_bear += 15.0

        if macro_5m == "BULL": score_bull += 10.0
        elif macro_5m == "BEAR": score_bear += 10.0

        # Order Flow Imbalance Voting (Weight: 25%)
        if delta_ratio >= 0.20:
            score_bull += 20.0 * min(1.0, delta_ratio * 2.0)
        elif delta_ratio <= -0.20:
            score_bear += 20.0 * min(1.0, abs(delta_ratio) * 2.0)

        if absorption == "BULL_ABSORPTION":
            score_bull += 10.0
        elif absorption == "BEAR_ABSORPTION":
            score_bear += 10.0

        # FVG Support Voting (Weight: 20%)
        if fvg_info:
            if fvg_info["direction"] == "BULL" and (fvg_info["is_inside"] or fvg_info["distance"] <= 1.5):
                score_bull += 15.0
            elif fvg_info["direction"] == "BEAR" and (fvg_info["is_inside"] or fvg_info["distance"] <= 1.5):
                score_bear += 15.0

        # Market Efficiency Multiplier (Weight: 15%)
        if kaufman_er >= 0.50:
            if score_bull > score_bear: score_bull += 15.0
            else: score_bear += 15.0

        # Bias determination
        total_score = max(score_bull, score_bear)
        if score_bull >= score_bear + 20.0 and score_bull >= 60.0:
            bias = "STRONG_BULL" if score_bull >= 78.0 else "BULL"
        elif score_bear >= score_bull + 20.0 and score_bear >= 60.0:
            bias = "STRONG_BEAR" if score_bear >= 78.0 else "BEAR"
        else:
            bias = "NEUTRAL"

        # Conviction boost for lot sizing (1.0 to 1.35)
        conviction_boost = 1.0
        if total_score >= 82.0 and absorption != "NEUTRAL":
            conviction_boost = 1.30
        elif total_score >= 70.0:
            conviction_boost = 1.15

        # Target RR Expansion Boost (+0.0R to +1.5R)
        rr_boost = 0.0
        if vol_regime == "EXPANSIVE_VOLATILITY" and total_score >= 75.0:
            rr_boost = 1.50 # Market is in massive institutional expansion - expand TP!
        elif total_score >= 80.0:
            rr_boost = 1.00

        telemetry = EnsembleTelemetry(
            timestamp=now,
            ensemble_bias=bias,
            ensemble_score=round(total_score, 1),
            delta_ratio_60s=delta_ratio,
            absorption_state=absorption,
            nearest_fvg=fvg_info,
            volatility_regime=vol_regime,
            parkinson_ratio=vol_ratio,
            conviction_boost=conviction_boost,
            rr_expansion_boost=rr_boost,
        )
        self.latest_telemetry = telemetry
        return telemetry
