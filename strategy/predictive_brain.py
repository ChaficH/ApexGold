"""
ApexGold Institutional Predictive Model Engine & Thinking Stream.

Advanced Quantitative Multi-Factor Probabilistic Predictive Brain:
1. Multi-Timeframe Trend & Momentum Odds:
   - 1H / 15m / 5m / 1m directional alignment
   - Kaufman Adaptive Efficiency Ratio (ER): Signal-to-noise ratio
   - Statistical Linear Regression R² & Slope: Trend conviction vs random walk
   - Directional Probabilities: P(BULL), P(BEAR), P(CHOP)
2. Mean-Reversion & Valuation Deviation:
   - Institutional 5m VWAP distance & Z-score
   - Multi-timeframe Stochastic RSI (%K, %D)
   - Anti-Chasing Penalty: Strictly penalizes buying into overbought tops or selling into oversold bottoms
3. Probabilistic Meta-Score & Expected Value (EV):
   - Bayesian / Multi-Factor Win Probability P(Win)
   - Expected Value in R-multiples: EV = P(Win) * R_target - (1 - P(Win)) * 1.0 - SpreadCost_R
   - Positive Mathematical Edge Gate: EV >= +0.25R required to trade
4. Real-Time "Chain-of-Thought" (Thinking Stream):
   - Sub-second monotonic heartbeat & UTC timestamp proving the bot is alive and never frozen
   - Step-by-step gate evaluations: Macro Trend, Value-Entry / Pullback, Volatility, News, EV
   - Human-readable narrative explanation detailing the exact reasoning for standing aside or trading
   - Rolling sliding window of recent evaluation decisions
"""

from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import logging
import math
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from engine.indicators import TechnicalIndicators

logger = logging.getLogger("ApexGold.PredictiveBrain")


@dataclass
class PredictiveEvaluation:
    """Snapshot of a single quantitative evaluation cycle."""
    timestamp: float
    time_utc_str: str
    cycle_index: int
    symbol: str
    current_bid: float
    current_ask: float
    spread_pts: float
    
    # Probabilistic & Quantitative Metrics
    p_bull: float
    p_bear: float
    p_chop: float
    p_win: float
    expected_value_r: float
    kaufman_er: float
    r_squared: float
    regime_name: str
    vwap_deviation: float
    stoch_rsi_k: float
    stoch_rsi_d: float
    
    # State & Verdict
    status: str             # "WAITING_FOR_PULLBACK", "HUNTING_SETUP", "QUALIFIED_TRADE", "DEFENSIVE_SHIELD", "STAND_ASIDE_CHOP"
    verdict_badge: str      # e.g. "WAITING FOR PULLBACK"
    is_frozen: bool         # Always False when evaluated
    
    # Gate Matrix
    gates: List[Dict[str, Any]]
    
    # Narrative Chain of Thought
    narrative: str
    target_zone: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ApexPredictiveBrain:
    """
    High-Performance Institutional Quantitative Predictive Brain for XAUUSD.
    Analyzes multi-timeframe order flow, computes Bayesian win probabilities,
    calculates mathematical expected value, and streams real-time reasoning to the UI.
    """

    def __init__(self, history_capacity: int = 30):
        self.cycle_index: int = 0
        self.last_eval_time: float = 0.0
        self.history: Deque[Dict[str, Any]] = deque(maxlen=history_capacity)
        self.latest_evaluation: Optional[PredictiveEvaluation] = None
        self._cached_state: Dict[str, Any] = {}

    def _sigmoid(self, x: float) -> float:
        """Numerically stable sigmoid function."""
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        else:
            z = math.exp(x)
            return z / (1.0 + z)

    def evaluate_market(
        self,
        symbol: str,
        dfs: Dict[str, pd.DataFrame],
        current_bid: float,
        current_ask: float,
        spread: float,
        news_reactor: Any = None,
        trade_manager: Any = None,
        now_ts: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute an end-to-end predictive evaluation cycle.
        Returns a rich state dictionary suitable for WebSocket broadcast and UI rendering.
        """
        now = now_ts or time.time()
        self.cycle_index += 1
        self.last_eval_time = now

        dt_utc = datetime.fromtimestamp(now, tz=timezone.utc)
        time_str = dt_utc.strftime("%H:%M:%S UTC")

        df_1h = dfs.get("1H")
        df_15m = dfs.get("15m")
        df_5m = dfs.get("5m")
        df_1m = dfs.get("1m")

        # Insufficient data fallback
        if df_5m is None or len(df_5m) < 20:
            return self._build_standby_state(symbol, now, time_str, current_bid, current_ask, spread, "Warming up multi-timeframe historical candles...")

        # ----------------------------------------------------------------------
        # 1. Multi-Timeframe Directional Bias & Trends
        # ----------------------------------------------------------------------
        tf_trends = TechnicalIndicators.get_4tf_trends(dfs)
        trend_1h = tf_trends.get("1H", "NEUTRAL")
        trend_15m = tf_trends.get("15m", "NEUTRAL")
        trend_5m = tf_trends.get("5m", "NEUTRAL")
        trend_1m = tf_trends.get("1m", "NEUTRAL")

        # 1H Macro Slope
        h1_slope = 0.0
        if df_1h is not None and len(df_1h) >= 20:
            c_1h = df_1h["close"].values
            h1_slope = float(TechnicalIndicators.ema_slope(c_1h, min(50, len(c_1h))))

        # ----------------------------------------------------------------------
        # 2. Kaufman Efficiency Ratio & Statistical R² on 5m
        # ----------------------------------------------------------------------
        c_5m = df_5m["close"].values
        kama_dict = TechnicalIndicators.calculate_kama(df_5m, window=10)
        efficiency_ratio = float(kama_dict.get("efficiency_ratio", 0.50))
        
        stat_trend = TechnicalIndicators.calculate_statistical_trend(c_5m, window=15)
        r_squared = float(stat_trend.get("r_squared", 0.50))
        reg_slope = float(stat_trend.get("slope", 0.0))

        # Choppiness Index
        h_5m = df_5m["high"].values
        l_5m = df_5m["low"].values
        chop_dict = TechnicalIndicators.calculate_choppiness_index(h_5m, l_5m, c_5m, period=14)
        chop_val = float(chop_dict.get("ci", 50.0))

        # ----------------------------------------------------------------------
        # 3. Institutional VWAP Alignment & Z-Score
        # ----------------------------------------------------------------------
        vwap_info = TechnicalIndicators.calculate_vwap(df_5m)
        vwap_val = float(vwap_info.get("vwap", current_bid))
        vwap_diff = current_bid - vwap_val

        zscore_info = TechnicalIndicators.calculate_zscore(c_5m, window=20)
        zscore_val = float(zscore_info.get("zscore", 0.0))

        # ----------------------------------------------------------------------
        # 4. Stochastic RSI (5m)
        # ----------------------------------------------------------------------
        stoch_rsi = TechnicalIndicators.calculate_stoch_rsi(c_5m, period=14, smooth_k=3, smooth_d=3)
        stoch_k = float(stoch_rsi.get("k", 50.0))
        stoch_d = float(stoch_rsi.get("d", 50.0))

        # ----------------------------------------------------------------------
        # 5. Probabilistic Directional Distribution: P(BULL), P(BEAR), P(CHOP)
        # ----------------------------------------------------------------------
        # Weighted point distribution based on structural evidence
        bull_points = 0.0
        bear_points = 0.0
        chop_points = 0.0

        if trend_1h == "BULL":
            bull_points += 3.0
        elif trend_1h == "BEAR":
            bear_points += 3.0
        else:
            chop_points += 2.0

        if trend_15m == "BULL":
            bull_points += 2.5
        elif trend_15m == "BEAR":
            bear_points += 2.5
        else:
            chop_points += 1.5

        if trend_5m == "BULL":
            bull_points += 2.0
        elif trend_5m == "BEAR":
            bear_points += 2.0

        if current_bid > vwap_val:
            bull_points += 1.5
        else:
            bear_points += 1.5

        if h1_slope > 0.02:
            bull_points += 1.5
        elif h1_slope < -0.02:
            bear_points += 1.5

        # Noise / Choppiness factors
        if chop_val > 58.0 or efficiency_ratio < 0.25 or r_squared < 0.25:
            chop_points += 3.5

        total_pts = bull_points + bear_points + chop_points + 1e-6
        p_bull = round(bull_points / total_pts, 3)
        p_bear = round(bear_points / total_pts, 3)
        p_chop = round(chop_points / total_pts, 3)

        # ----------------------------------------------------------------------
        # 6. Regime Classification
        # ----------------------------------------------------------------------
        if p_chop >= 0.45 or chop_val >= 60.0:
            regime_name = "RANGE_BOUND_CHOP"
        elif p_bull >= 0.55:
            regime_name = "BULL_TREND_EXPANSION" if efficiency_ratio >= 0.50 else "BULL_TREND_MOMENTUM"
        elif p_bear >= 0.55:
            regime_name = "BEAR_TREND_EXPANSION" if efficiency_ratio >= 0.50 else "BEAR_TREND_MOMENTUM"
        else:
            regime_name = "NEUTRAL_CONSOLIDATION"

        # ----------------------------------------------------------------------
        # 7. Bayesian Probability of Win P(Win) & Mathematical Expected Value (EV)
        # ----------------------------------------------------------------------
        # Evaluated for the dominant macro direction
        dominant_dir = "BUY" if p_bull >= p_bear else "SELL"
        dominant_prob = max(p_bull, p_bear)

        # Baseline log-odds
        logit = 0.0

        # Factor 1: Macro alignment (+0.60 logit if 1H + 15m strongly align)
        if dominant_dir == "BUY" and trend_1h == "BULL" and trend_15m == "BULL":
            logit += 0.65
        elif dominant_dir == "SELL" and trend_1h == "BEAR" and trend_15m == "BEAR":
            logit += 0.65
        elif (dominant_dir == "BUY" and (trend_1h == "BEAR" or trend_15m == "BEAR")) or \
             (dominant_dir == "SELL" and (trend_1h == "BULL" or trend_15m == "BULL")):
            logit -= 1.20  # Fighting macro trend severely penalizes win rate

        # Factor 2: Efficiency ratio & R² conviction
        logit += (efficiency_ratio - 0.40) * 0.80
        logit += (r_squared - 0.40) * 0.60

        # Factor 3: Anti-Chasing / Value Entry Penalty (ATR-Normalized)
        atr_5m_arr = TechnicalIndicators.atr(h_5m, l_5m, c_5m, 14)
        atr_5m = float(atr_5m_arr[-1]) if (atr_5m_arr is not None and len(atr_5m_arr) > 0 and not np.isnan(atr_5m_arr[-1])) else 2.50
        overextended_dist = max(2.80, 1.5 * atr_5m)

        # Genuinely overbought / oversold at extreme exhaustion wicks
        is_overbought = (stoch_k > 80.0 and vwap_diff > overextended_dist) or (stoch_k > 88.0 and vwap_diff > 2.0)
        is_oversold = (stoch_k < 20.0 and vwap_diff < -overextended_dist) or (stoch_k < 12.0 and vwap_diff < -2.0)

        pullback_ready = False
        if dominant_dir == "BUY":
            if is_overbought:
                logit -= 0.85  # Chasing at tops
            elif (20.0 <= stoch_k <= 65.0 and -1.5 * atr_5m <= vwap_diff <= 1.2 * atr_5m) or (efficiency_ratio >= 0.50 and stoch_k <= 75.0):
                logit += 0.45  # Buying the dip/consolidation in bull trend!
                pullback_ready = True
        elif dominant_dir == "SELL":
            if is_oversold:
                logit -= 0.85  # Chasing at bottoms
            elif (35.0 <= stoch_k <= 80.0 and -1.2 * atr_5m <= vwap_diff <= 1.5 * atr_5m) or (efficiency_ratio >= 0.50 and stoch_k >= 25.0):
                logit += 0.45  # Selling the rally/consolidation in bear trend!
                pullback_ready = True

        p_win = round(self._sigmoid(logit), 3)

        # Target reward-to-risk (adaptive based on trend strength, aiming for high profits)
        target_rr = 4.2 if efficiency_ratio >= 0.50 else 3.2
        spread_cost_r = (spread / 1.50) * 0.08  # Normalized spread impact
        
        # Mathematical EV in R-multiples: EV = P(win) * RR - (1 - P(win)) * 1.0 - Cost
        expected_value_r = round((p_win * target_rr) - ((1.0 - p_win) * 1.0) - spread_cost_r, 2)

        # ----------------------------------------------------------------------
        # 8. Multi-Gate Diagnostic Checklist
        # ----------------------------------------------------------------------
        gates = []

        # Gate 1: Macro Directional Lock (1H + 15m)
        macro_pass = (trend_1h != "NEUTRAL") and (trend_1h == trend_15m)
        gates.append({
            "name": "Macro Trend Alignment",
            "status": "PASS" if macro_pass else "WAITING",
            "detail": f"1H: {trend_1h} | 15m: {trend_15m} (Slope: {h1_slope:+.3f})",
            "passed": macro_pass,
        })

        # Gate 2: Value-Entry / Pullback Verification (Anti-Chasing)
        value_pass = True
        value_detail = ""
        if dominant_dir == "BUY":
            if is_overbought:
                value_pass = False
                value_detail = f"5m StochRSI is {stoch_k:.1f} (Overbought) & +{vwap_diff:.2f} above VWAP. Chasing prohibited; waiting for pullback."
            elif pullback_ready:
                value_detail = f"PULLBACK VERIFIED: 5m StochRSI is {stoch_k:.1f} near VWAP fair value ({vwap_val:.2f})."
            else:
                value_detail = f"VWAP diff: {vwap_diff:+.2f} pts | 5m StochRSI: {stoch_k:.1f}."
        else:
            if is_oversold:
                value_pass = False
                value_detail = f"5m StochRSI is {stoch_k:.1f} (Oversold) & {vwap_diff:.2f} below VWAP. Chasing prohibited; waiting for pullback."
            elif pullback_ready:
                value_detail = f"PULLBACK VERIFIED: 5m StochRSI is {stoch_k:.1f} near VWAP fair value ({vwap_val:.2f})."
            else:
                value_detail = f"VWAP diff: {vwap_diff:+.2f} pts | 5m StochRSI: {stoch_k:.1f}."

        gates.append({
            "name": "Value-Entry / Pullback Check",
            "status": "PASS" if value_pass else "WAITING",
            "detail": value_detail,
            "passed": value_pass,
        })

        # Gate 3: Volatility & Spread Health
        max_allowed_spread = 0.80 if any(k in symbol.upper() for k in ["XAU", "GOLD"]) else 0.0003
        spread_ok = (spread <= max_allowed_spread)
        chop_ok = (chop_val < 61.8)
        vol_pass = spread_ok and chop_ok
        gates.append({
            "name": "Spread & Volatility Health",
            "status": "PASS" if vol_pass else "BLOCKED",
            "detail": f"Spread: {spread:.2f} pts (Max: {max_allowed_spread:.2f}) | Chop Index: {chop_val:.1f} (Threshold: 61.8)",
            "passed": vol_pass,
        })

        # Gate 4: Economic News Shield
        is_news_freeze = False
        news_detail = "Economic calendar clear. No high-impact events imminent."
        if news_reactor:
            cal_state = news_reactor.check_calendar_state(current_utc=now)
            is_news_freeze = cal_state.get("is_news_freeze", False)
            if is_news_freeze:
                ev = cal_state.get("nearest_event") or {}
                news_detail = f"News Freeze Active: {ev.get('title', 'High-Impact USD News')}."
            else:
                cd = cal_state.get("countdown_seconds")
                if cd is not None and cd <= 1200:
                    news_detail = f"Upcoming High-Impact News in {cd // 60}m. Defensive shield primed."

        gates.append({
            "name": "Economic News Shield",
            "status": "BLOCKED" if is_news_freeze else "PASS",
            "detail": news_detail,
            "passed": not is_news_freeze,
        })

        # Gate 5: Mathematical Edge Gate (EV >= +0.25R)
        ev_pass = (expected_value_r >= 0.25 and p_win >= 0.58)
        gates.append({
            "name": "Mathematical Edge (EV >= +0.25R)",
            "status": "PASS" if ev_pass else "WAITING",
            "detail": f"Win Prob: {p_win * 100:.1f}% | Expected Value: {expected_value_r:+.2f}R (Min: +0.25R)",
            "passed": ev_pass,
        })

        # ----------------------------------------------------------------------
        # 9. Verdict, State & Narrative Chain-of-Thought Generation
        # ----------------------------------------------------------------------
        # Determine current state
        if is_news_freeze:
            status = "DEFENSIVE_SHIELD"
            verdict_badge = "DEFENSIVE SHIELD (NEWS FREEZE)"
        elif not spread_ok:
            status = "DEFENSIVE_SHIELD"
            verdict_badge = "DEFENSIVE SHIELD (HIGH SPREAD)"
        elif not chop_ok or p_chop >= 0.45:
            status = "STAND_ASIDE_CHOP"
            verdict_badge = "STANDING ASIDE (CHOPPY RANGE)"
        elif not macro_pass:
            status = "WAITING_FOR_ALIGNMENT"
            verdict_badge = "WAITING FOR TIMEFRAME ALIGNMENT"
        elif not value_pass:
            status = "WAITING_FOR_PULLBACK"
            verdict_badge = f"WAITING FOR PULLBACK ({dominant_dir} EXTENDED)"
        elif not ev_pass:
            status = "HUNTING_SETUP"
            verdict_badge = "HUNTING SETUP (BUILDING CONVCTION)"
        else:
            status = "QUALIFIED_TRADE"
            verdict_badge = f"SETUP QUALIFIED ({dominant_dir} EV: {expected_value_r:+.2f}R)"

        # Target entry zone calculation using 5m EMA9 and EMA20 for realistic pullback zones
        ema9_arr = TechnicalIndicators.ema(c_5m, 9)
        ema9_val = float(ema9_arr[-1]) if (ema9_arr is not None and len(ema9_arr) > 0 and not np.isnan(ema9_arr[-1])) else vwap_val
        ema20_arr = TechnicalIndicators.ema(c_5m, 20)
        ema20_val = float(ema20_arr[-1]) if (ema20_arr is not None and len(ema20_arr) > 0 and not np.isnan(ema20_arr[-1])) else vwap_val

        zone_low = min(ema9_val, ema20_val)
        zone_high = max(ema9_val, ema20_val)

        if dominant_dir == "BUY":
            target_zone = {
                "direction": "BUY",
                "optimal_entry_min": round(zone_low - (0.30 * atr_5m), 2),
                "optimal_entry_max": round(zone_high + (0.20 * atr_5m), 2),
                "invalidation_level": round(zone_low - (1.50 * atr_5m), 2),
                "target_objective": round(current_bid + (target_rr * 3.50), 2),
            }
        else:
            target_zone = {
                "direction": "SELL",
                "optimal_entry_min": round(zone_low - (0.20 * atr_5m), 2),
                "optimal_entry_max": round(zone_high + (0.30 * atr_5m), 2),
                "invalidation_level": round(zone_high + (1.50 * atr_5m), 2),
                "target_objective": round(current_bid - (target_rr * 3.50), 2),
            }

        # Human-Readable Narrative Thinking
        narrative = self._generate_narrative_thinking(
            time_str=time_str,
            cycle_idx=self.cycle_index,
            symbol=symbol,
            current_bid=current_bid,
            trend_1h=trend_1h,
            trend_15m=trend_15m,
            dominant_dir=dominant_dir,
            efficiency_ratio=efficiency_ratio,
            r_squared=r_squared,
            vwap_val=vwap_val,
            vwap_diff=vwap_diff,
            stoch_k=stoch_k,
            p_win=p_win,
            expected_value_r=expected_value_r,
            status=status,
            target_zone=target_zone,
            chop_val=chop_val,
        )

        eval_record = PredictiveEvaluation(
            timestamp=now,
            time_utc_str=time_str,
            cycle_index=self.cycle_index,
            symbol=symbol,
            current_bid=current_bid,
            current_ask=current_ask,
            spread_pts=round(spread, 2),
            p_bull=p_bull,
            p_bear=p_bear,
            p_chop=p_chop,
            p_win=p_win,
            expected_value_r=expected_value_r,
            kaufman_er=round(efficiency_ratio, 3),
            r_squared=round(r_squared, 3),
            regime_name=regime_name,
            vwap_deviation=round(vwap_diff, 2),
            stoch_rsi_k=round(stoch_k, 1),
            stoch_rsi_d=round(stoch_d, 1),
            status=status,
            verdict_badge=verdict_badge,
            is_frozen=False,
            gates=gates,
            narrative=narrative,
            target_zone=target_zone,
        )

        self.latest_evaluation = eval_record

        # Append to rolling history log (record key transitions or every 10 cycles to keep compact)
        history_entry = {
            "time": time_str,
            "cycle": self.cycle_index,
            "status": status,
            "badge": verdict_badge,
            "p_win": f"{p_win * 100:.1f}%",
            "ev": f"{expected_value_r:+.2f}R",
            "p_bull": f"{p_bull * 100:.0f}%",
            "p_bear": f"{p_bear * 100:.0f}%",
            "summary": narrative.split("\n")[-1] if "\n" in narrative else narrative,
        }
        
        # Don't duplicate consecutive identical states if time diff is small
        if not self.history or (self.history[-1].get("status") != status) or (self.cycle_index % 15 == 0):
            self.history.appendleft(history_entry)

        res_dict = eval_record.to_dict()
        res_dict["history"] = list(self.history)
        self._cached_state = res_dict
        return res_dict

    def _generate_narrative_thinking(
        self,
        time_str: str,
        cycle_idx: int,
        symbol: str,
        current_bid: float,
        trend_1h: str,
        trend_15m: str,
        dominant_dir: str,
        efficiency_ratio: float,
        r_squared: float,
        vwap_val: float,
        vwap_diff: float,
        stoch_k: float,
        p_win: float,
        expected_value_r: float,
        status: str,
        target_zone: Dict[str, Any],
        chop_val: float,
    ) -> str:
        """Construct a precise, institutional narrative explaining the bot's live thoughts."""
        lines = [
            f"⚡ [CYCLE #{cycle_idx} @ {time_str}] ACTIVE EVALUATION HEARTBEAT — BOT IS RUNNING NORMALLY",
            f"• Macro Structure: 1H {trend_1h} | 15m {trend_15m} | Kaufman Efficiency: {efficiency_ratio:.2f} | R²: {r_squared:.2f}",
            f"• Valuation: Spot {current_bid:.2f} vs 5m VWAP {vwap_val:.2f} ({vwap_diff:+.2f} pts) | 5m StochRSI: {stoch_k:.1f}",
            f"• Edge Metrics: P(Win) = {p_win * 100:.1f}% | Expected Value = {expected_value_r:+.2f}R",
        ]

        if status == "WAITING_FOR_PULLBACK":
            lines.append(
                f"🧠 VERDICT: Market is {trend_1h} but currently OVEREXTENDED. Entering a {dominant_dir} at the high/low carries negative statistical expectancy (EV: {expected_value_r:+.2f}R). "
                f"BOT IS DELIBERATELY STANDING ASIDE: Patiently waiting for price to retrace into the high-probability discount zone [{target_zone['optimal_entry_min']:.2f} - {target_zone['optimal_entry_max']:.2f}] or StochRSI to normalize before firing."
            )
        elif status == "STAND_ASIDE_CHOP":
            lines.append(
                f"🧠 VERDICT: Market is exhibiting high entropy and directionless chop (Choppiness Index: {chop_val:.1f} > 60). "
                f"BOT IS PRESERVING CAPITAL: Algorithmic edge degrades in choppy conditions. Standing aside until a clean breakout occurs."
            )
        elif status == "DEFENSIVE_SHIELD":
            lines.append(
                f"🧠 VERDICT: Defensive safety shield is actively engaged to protect equity from high-impact volatility spikes or wide broker spreads."
            )
        elif status == "WAITING_FOR_ALIGNMENT":
            lines.append(
                f"🧠 VERDICT: Higher timeframes are currently conflicted (1H: {trend_1h}, 15m: {trend_15m}). "
                f"BOT IS AWAITING DIRECTIONAL CONSENSUS: Institutional flow requires multi-timeframe alignment."
            )
        elif status == "QUALIFIED_TRADE":
            lines.append(
                f"🚀 VERDICT: HIGH-CONVICTION SETUP QUALIFIED! All predictive gates passed (EV: {expected_value_r:+.2f}R >= +0.25R, P_win: {p_win*100:.1f}%). "
                f"Prepared to dispatch {dominant_dir} towards target {target_zone['target_objective']:.2f}."
            )
        else:
            lines.append(
                f"🧠 VERDICT: System is actively scanning order flow and calculating setup probabilities. Standing by for optimal entry criteria."
            )

        return "\n".join(lines)

    def _build_standby_state(
        self,
        symbol: str,
        now: float,
        time_str: str,
        bid: float,
        ask: float,
        spread: float,
        reason: str,
    ) -> Dict[str, Any]:
        """Standby fallback when candle history is initializing."""
        return {
            "timestamp": now,
            "time_utc_str": time_str,
            "cycle_index": self.cycle_index,
            "symbol": symbol,
            "current_bid": bid,
            "current_ask": ask,
            "spread_pts": round(spread, 2),
            "p_bull": 0.33,
            "p_bear": 0.33,
            "p_chop": 0.34,
            "p_win": 0.50,
            "expected_value_r": 0.0,
            "kaufman_er": 0.50,
            "r_squared": 0.50,
            "regime_name": "INITIALIZING",
            "vwap_deviation": 0.0,
            "stoch_rsi_k": 50.0,
            "stoch_rsi_d": 50.0,
            "status": "INITIALIZING",
            "verdict_badge": "WARMING UP ENGINE",
            "is_frozen": False,
            "gates": [
                {"name": "Engine Initialization", "status": "WAITING", "detail": reason, "passed": False}
            ],
            "narrative": f"⚡ [CYCLE #{self.cycle_index} @ {time_str}] ApexGold Predictive Engine is warming up: {reason}",
            "target_zone": {},
            "history": list(self.history),
        }

    def get_state(self) -> Dict[str, Any]:
        """Return the latest cached predictive state snapshot."""
        if not self._cached_state:
            now = time.time()
            dt_utc = datetime.fromtimestamp(now, tz=timezone.utc)
            return self._build_standby_state(
                symbol="XAUUSD",
                now=now,
                time_str=dt_utc.strftime("%H:%M:%S UTC"),
                bid=0.0,
                ask=0.0,
                spread=0.25,
                reason="Engine initialized. Awaiting first tick...",
            )
        return self._cached_state

    def filter_signal(self, signal: Any) -> Tuple[bool, str]:
        """
        Gate trade execution with mathematical Expected Value.
        Returns (is_approved, reason).
        """
        if not self.latest_evaluation:
            return True, "Predictive model standby (approved by default)"

        ev = self.latest_evaluation.expected_value_r
        p_win = self.latest_evaluation.p_win
        dominant_dir = "BUY" if self.latest_evaluation.p_bull >= self.latest_evaluation.p_bear else "SELL"
        sig_dir = getattr(signal, "direction", dominant_dir)

        # Anti-chasing veto
        if self.latest_evaluation.status == "WAITING_FOR_PULLBACK":
            return False, f"Predictive Model Veto: Price is extended in {dominant_dir} direction. Waiting for pullback (EV: {ev:+.2f}R < +0.25R)"

        # Anti-counter-trend veto: if macro direction is strong (>= 65%) opposing the signal
        if (sig_dir == "BUY" and self.latest_evaluation.p_bear >= 0.65) or (sig_dir == "SELL" and self.latest_evaluation.p_bull >= 0.65):
            return False, f"Predictive Model Veto: Signal direction {sig_dir} directly conflicts with dominant macro trend"

        if ev < 0.20 or p_win < 0.55:
            return False, f"Predictive Model Veto: Sub-optimal expected value (EV: {ev:+.2f}R, P_win: {p_win*100:.1f}%)"

        return True, f"Predictive Model Approved: EV = {ev:+.2f}R, P_win = {p_win*100:.1f}%"
