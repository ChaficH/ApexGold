"""
ApexGold 100-Point Confluence Scoring Engine & Smart Auto-Pilot Market-Following Brain.

Features:
1. Flagship Default: AUTO_ADAPTIVE (Auto-reads market telemetry every 5s and auto-dials
   Aggressiveness percentage between 1% and 100%, moving the UI scroller live).
2. 3 Autonomous Sub-Regimes:
   - Sub-Regime 1: Fast Hit-and-Run Scalp (Auto-Dial 72%-85%, Min Score 62, BE +0.45R, 4s stall exit)
   - Sub-Regime 2: Institutional Wave Runner (Auto-Dial 55%-68%, Min Score 66, TP 2.5R-4.0R, lock +0.8R)
   - Sub-Regime 3: Chop / Pre-News Defensive Shield (Auto-Dial 25%-40%, Min Score 80, A+ sniper only)
3. 4 Manual Preset Gears:
   - CAUTIOUS (Smart Sniper — 35% | Min Score: 76/100)
   - STANDARD (Balanced — 55% | Min Score: 68/100)
   - MORE AGGRESSIVE (Active Momentum — 75% | Min Score: 60/100)
   - MOST AGGRESSIVE (Hyper Hit-and-Run — 92% | Min Score: 52/100)
4. Continuous 1%-100% Aggressiveness Scroller mapping mathematically to min score thresholds.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from engine.indicators import TechnicalIndicators

logger = logging.getLogger("ApexGold.ConfluenceScorer")


def pct_to_threshold(pct: float) -> int:
    """
    Map continuous aggressiveness percentage (1% to 100%) to min confluence score threshold (90 to 48).
    Calibrated across the 4 manual gear points:
      - 1%: 90 min score (ultra defensive)
      - 35%: 76 min score (Cautious / Smart Sniper)
      - 55%: 68 min score (Standard / Balanced)
      - 75%: 60 min score (More Aggressive / Active Momentum)
      - 92%: 52 min score (Most Aggressive / Hyper Hit-and-Run)
      - 100%: 48 min score (maximum action)
    """
    p = max(1.0, min(100.0, float(pct)))
    points = [
        (1.0, 90.0),
        (35.0, 76.0),
        (55.0, 68.0),
        (75.0, 60.0),
        (92.0, 52.0),
        (100.0, 48.0),
    ]
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        if x0 <= p <= x1:
            ratio = (p - x0) / (x1 - x0)
            return int(round(y0 + ratio * (y1 - y0)))
    return 68


def threshold_to_pct(thresh: int) -> float:
    """Map min confluence score threshold (48 to 90) back to aggressiveness percentage."""
    t = max(48, min(90, int(thresh)))
    points = [
        (90, 1.0),
        (76, 35.0),
        (68, 55.0),
        (60, 75.0),
        (52, 92.0),
        (48, 100.0),
    ]
    for i in range(len(points) - 1):
        y0, x0 = points[i]
        y1, x1 = points[i + 1]
        if y1 <= t <= y0:
            ratio = (y0 - t) / (y0 - y1)
            return round(x0 + ratio * (x1 - x0), 1)
    return 55.0


@dataclass
class ConfluenceSignal:
    """Trade execution signal produced by the 100-Point Scoring Engine."""
    symbol: str
    direction: str              # "BUY" or "SELL"
    score: int                  # 0 to 100
    tier: str                   # "TIER_1_SNIPER", "TIER_2_BALANCED", "TIER_3_AGGRESSIVE", "NONE"
    regime: str                 # "ADAPTIVE_RUNNER" or "ADAPTIVE_SCALP"
    sub_regime: str             # "FAST_SCALP", "WAVE_RUNNER", "NEWS_MOMENTUM_STUDY", or "DEFENSIVE_SHIELD"
    target_rr: float            # Expected risk-to-reward ratio (e.g. 1.2 or 3.0)
    entry_price: float
    suggested_sl: float
    suggested_tp: float
    risk_r_dist: float          # Distance in price between entry and SL
    be_lock_r: float            # Trigger to lock breakeven (+0.45R for fast scalp, +0.6R standard, +0.8R runner)
    stall_exit_seconds: float   # Timeout for stall exit (4.0s for fast scalp, 90s for runner)
    breakdown: Dict[str, int]   # Point breakdown for each timeframe
    reason: str
    time: float
    order_type: str = "MARKET"  # "MARKET", "BUY_LIMIT", or "SELL_LIMIT"
    limit_price: Optional[float] = None
    invalidation_price: Optional[float] = None
    historical_edge: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConfluenceScorer:
    """
    100-Point 'Smart Auto-Pilot' Market-Following Brain & Confluence Engine.
    Combines Macro Bias, Session Wave, Structure Setup, and 1m Execution Trigger.
    """

    def __init__(
        self,
        threshold_aggressive: int = 60,
        threshold_balanced: int = 68,
        threshold_sniper: int = 82,
        mode: str = "auto_adaptive", # "auto_adaptive" (DEFAULT), "cautious", "standard", "aggressive", "hyper", "manual"
        aggressiveness_pct: float = 65.0,
    ):
        self.threshold_aggressive = threshold_aggressive
        self.threshold_balanced = threshold_balanced
        self.threshold_sniper = threshold_sniper
        self.mode = mode.lower() if mode else "auto_adaptive"
        self.aggressiveness_pct = float(aggressiveness_pct)

        self.tf_trends: Dict[str, str] = {"1H": "NEUTRAL", "15m": "NEUTRAL", "5m": "NEUTRAL", "1m": "NEUTRAL"}

        # Auto-Adaptive Brain State
        self.auto_brain_state: Dict[str, Any] = {
            "is_auto": True if self.mode == "auto_adaptive" else False,
            "sub_regime": "WAVE_RUNNER",
            "dial_pct": 65.0,
            "active_threshold": 66,
            "pill_text": "🧠 AUTO-BRAIN INITIALIZING: Analyzing market structure and session velocity...",
            "reason": "Market Structure Baseline",
            "target_rr": 2.5,
            "be_lock_r": 0.6,
            "stall_exit_seconds": 90.0,
            "tf_trends": self.tf_trends,
            "last_update": 0.0,
        }

    def set_mode(self, mode_name: str, pct: Optional[float] = None) -> None:
        """Switch active transmission mode or dial a continuous manual percentage."""
        m = mode_name.lower().strip()
        if m in ["auto_adaptive", "auto", "smart_auto"]:
            self.mode = "auto_adaptive"
            self.auto_brain_state["is_auto"] = True
        elif m in ["cautious", "sniper", "tier_1"]:
            self.mode = "cautious"
            self.aggressiveness_pct = 35.0
            self.auto_brain_state["is_auto"] = False
        elif m in ["standard", "balanced", "tier_2"]:
            self.mode = "standard"
            self.aggressiveness_pct = 55.0
            self.auto_brain_state["is_auto"] = False
        elif m in ["aggressive", "more_aggressive", "tier_3"]:
            self.mode = "aggressive"
            self.aggressiveness_pct = 75.0
            self.auto_brain_state["is_auto"] = False
        elif m in ["hyper", "most_aggressive", "tier_4"]:
            self.mode = "hyper"
            self.aggressiveness_pct = 92.0
            self.auto_brain_state["is_auto"] = False
        elif m in ["manual", "custom"]:
            self.mode = "manual"
            self.auto_brain_state["is_auto"] = False
            if pct is not None:
                self.aggressiveness_pct = max(1.0, min(100.0, float(pct)))
        else:
            self.mode = "auto_adaptive"
            self.auto_brain_state["is_auto"] = True

        if pct is not None and self.mode == "manual":
            self.aggressiveness_pct = max(1.0, min(100.0, float(pct)))

        logger.info(f"[TRANSMISSION SHIFT] Active Mode: {self.mode.upper()} | Dial: {self.get_active_aggressiveness_pct():.1f}% | Required Score: {self.get_active_threshold()}/100")

    def get_active_threshold(self) -> int:
        """Return the required confluence score threshold based on active transmission mode."""
        if self.mode == "auto_adaptive":
            return int(self.auto_brain_state.get("active_threshold", 66))
        elif self.mode == "cautious":
            return 76
        elif self.mode == "standard":
            return 68
        elif self.mode == "aggressive":
            return 60
        elif self.mode == "hyper":
            return 52
        elif self.mode == "manual":
            return pct_to_threshold(self.aggressiveness_pct)
        return self.threshold_balanced

    def get_active_aggressiveness_pct(self) -> float:
        """Return active dial percentage (auto-adapted or manually selected)."""
        if self.mode == "auto_adaptive":
            return float(self.auto_brain_state.get("dial_pct", 65.0))
        elif self.mode == "cautious":
            return 35.0
        elif self.mode == "standard":
            return 55.0
        elif self.mode == "aggressive":
            return 75.0
        elif self.mode == "hyper":
            return 92.0
        return self.aggressiveness_pct

    def get_auto_brain_status(self) -> Dict[str, Any]:
        """Return complete snapshot of the Smart Auto-Pilot brain for UI broadcasting."""
        status = dict(self.auto_brain_state)
        status["mode"] = self.mode
        status["aggressiveness_pct"] = self.get_active_aggressiveness_pct()
        status["threshold"] = self.get_active_threshold()
        status["tf_trends"] = self.tf_trends
        return status

    def evaluate_auto_brain(
        self,
        dfs: Dict[str, pd.DataFrame],
        current_spread: float,
        max_spread: float,
        tick_velocity: float,
        news_reactor: Any,
        symbol: str = "XAUUSD",
        now_ts: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Sub-Regime Brain Evaluator (polled every 5 seconds).
        Dynamically classifies market environment into:
        1. DEFENSIVE_SHIELD (25%-40% Dial | Min Score 80)
        2. WAVE_RUNNER     (55%-68% Dial | Min Score 66)
        3. FAST_SCALP      (72%-85% Dial | Min Score 62)
        """
        now = now_ts or time.time()

        # Update real-time mathematical directional bias across all 4 timeframes
        self.tf_trends = TechnicalIndicators.get_4tf_trends(dfs)
        self.auto_brain_state["tf_trends"] = self.tf_trends

        # Rate-limit evaluation updates to once every 2.0s to avoid UI jitter
        if now - self.auto_brain_state.get("last_update", 0.0) < 2.0:
            return self.auto_brain_state


        df_1h = dfs.get("1H")
        df_15m = dfs.get("15m")
        df_5m = dfs.get("5m")
        df_1m = dfs.get("1m")

        dt = datetime.fromtimestamp(now, tz=timezone.utc)
        hour = dt.hour + (dt.minute / 60.0)
        is_london_ny_active = (7.0 <= hour <= 18.0) # High liquidity window

        # ----------------------------------------------------------------------
        # Sub-Regime 3: Chop / Pre-News Defensive Shield (25%-40% Aggressiveness)
        # Checked first: Protects account against news spikes, rollover, and high spread
        # ----------------------------------------------------------------------
        is_wide_spread = (current_spread >= 0.75 * max_spread) if max_spread > 0 else False
        is_dead_hours = (hour >= 21.0 or hour < 2.0) # Late Asian rollover chop

        is_near_major_news = False
        news_name = ""
        news_bias_direction = "NEUTRAL"
        news_confidence = 0.50
        news_summary = ""

        if news_reactor:
            cal_state = news_reactor.check_calendar_state()
            study = news_reactor.analyze_news_fundamental_bias(symbol)
            news_bias_direction = study.get("direction", "NEUTRAL")
            news_confidence = study.get("confidence", 0.50)
            news_summary = study.get("summary", "")

            if cal_state.get("is_news_freeze", False):
                is_near_major_news = True
                ev = cal_state.get("event") or {}
                news_name = ev.get("title", "High-Impact USD News")
            else:
                cd = cal_state.get("countdown_seconds")
                if cd is not None and cd <= 1200: # Within 20 min
                    is_near_major_news = True
                    ev = cal_state.get("event") or {}
                    news_name = ev.get("title", "Upcoming High-Impact USD News")

        # If news is active, but we have strong fundamental study clarity (confidence >= 70%):
        # DO NOT freeze! Dial up intelligent News Momentum Study mode!
        if is_near_major_news and news_confidence >= 0.70 and news_bias_direction in ["BUY", "SELL"] and not is_wide_spread:
            dial_pct = 76.0
            threshold = 64
            reason_str = f"News Momentum: {news_bias_direction} Bias on {news_name} ({news_summary})"
            pill_text = f"🧠 Auto-Brain: {dial_pct:.0f}% (News Momentum: {news_bias_direction} | Min: {threshold})"
            full_reason = f"News Momentum Study — {reason_str} | Threshold: {threshold}/100"
            self.auto_brain_state = {
                "is_auto": (self.mode == "auto_adaptive"),
                "sub_regime": "NEWS_MOMENTUM_STUDY",
                "dial_pct": dial_pct,
                "active_threshold": threshold,
                "pill_text": pill_text,
                "full_reason": full_reason,
                "reason": reason_str,
                "target_rr": 3.0,
                "be_lock_r": 0.5,
                "stall_exit_seconds": 60.0,
                "last_update": now,
                "news_bias": news_bias_direction,
                "news_confidence": news_confidence,
            }
            return self.auto_brain_state

        if is_near_major_news or is_wide_spread or is_dead_hours:
            # Defensive Shield Activated (only when no clear directional edge or wide spread)
            dial_pct = 32.0 if is_near_major_news else (28.0 if is_wide_spread else 36.0)
            threshold = 80
            short_reason = (
                f"Pre-News ({news_name[:16]}..)"
                if is_near_major_news else
                ("Wide Spread Guard" if is_wide_spread else "Rollover Guard")
            )
            reason_str = (
                f"Pre-News Shield ({news_name}) — Awaiting Catalyst"
                if is_near_major_news else
                ("Wide Spread Protection Active" if is_wide_spread else "Asian Rollover Low-Liquidity Shield")
            )
            pill_text = f"🧠 Auto-Brain: {dial_pct:.0f}% ({short_reason} | Min: {threshold})"
            full_reason = f"Chop/Pre-News Defensive Shield — {reason_str} | Threshold: {threshold}/100"

            self.auto_brain_state = {
                "is_auto": (self.mode == "auto_adaptive"),
                "sub_regime": "DEFENSIVE_SHIELD",
                "dial_pct": dial_pct,
                "active_threshold": threshold,
                "pill_text": pill_text,
                "full_reason": full_reason,
                "reason": reason_str,
                "target_rr": 2.0,
                "be_lock_r": 0.6,
                "stall_exit_seconds": 60.0,
                "last_update": now,
                "news_bias": news_bias_direction,
                "news_confidence": news_confidence,
            }
            return self.auto_brain_state

        # ----------------------------------------------------------------------
        # Check Multi-Timeframe Alignment & Momentum for Wave Runner vs Fast Scalp
        # ----------------------------------------------------------------------
        m15_adx = 20.0
        if df_15m is not None and len(df_15m) >= 20:
            h_15 = df_15m["high"].values
            l_15 = df_15m["low"].values
            c_15 = df_15m["close"].values
            adx_dict = TechnicalIndicators.adx(h_15, l_15, c_15, 14)
            val = adx_dict["adx"][-1]
            if not np.isnan(val):
                m15_adx = float(val)

        # Check 1H trend alignment
        h1_aligned = False
        h1_open_room = True
        if df_1h is not None and len(df_1h) >= 20:
            c_1h = df_1h["close"].values
            ema50 = TechnicalIndicators.ema(c_1h, min(50, len(c_1h)))[-1]
            slope = TechnicalIndicators.ema_slope(c_1h, min(50, len(c_1h)))
            h1_aligned = (abs(slope) > 0.05) or (c_1h[-1] > ema50 if slope > 0 else c_1h[-1] < ema50)

            swings_1h = TechnicalIndicators.find_swings(df_1h["high"].values, df_1h["low"].values, window=3)
            # Estimate room to run
            if swings_1h["highs"] and swings_1h["lows"]:
                recent_range = max(df_1h["high"].values[-5:]) - min(df_1h["low"].values[-5:])
                h1_open_room = recent_range >= 2.5

        # ----------------------------------------------------------------------
        # Sub-Regime 2: Institutional Wave Runner (55%-68% Aggressiveness)
        # Detected when 1H + 15m align, 15m ADX >= 22.0, and open room to run > 2.5R
        # ----------------------------------------------------------------------
        if h1_aligned and (m15_adx >= 22.0) and h1_open_room:
            dial_pct = 64.0 if m15_adx >= 25.0 else 58.0
            threshold = 66
            reason_str = f"Strong 1H/15m Trend Alignment (15m ADX {m15_adx:.1f} >= 22)"
            pill_text = f"🧠 Auto-Brain: {dial_pct:.0f}% (Wave Runner: Trend & ADX {m15_adx:.0f} | Min: {threshold})"
            full_reason = f"Institutional Wave Runner Mode — {reason_str} | Threshold: {threshold}/100"

            self.auto_brain_state = {
                "is_auto": (self.mode == "auto_adaptive"),
                "sub_regime": "WAVE_RUNNER",
                "dial_pct": dial_pct,
                "active_threshold": threshold,
                "pill_text": pill_text,
                "full_reason": full_reason,
                "reason": reason_str,
                "target_rr": 3.0,
                "be_lock_r": 0.8,
                "stall_exit_seconds": 90.0,
                "last_update": now,
            }
            return self.auto_brain_state

        # ----------------------------------------------------------------------
        # Sub-Regime 1: Fast Hit-and-Run Scalp (72%-85% Aggressiveness)
        # Detected during active liquidity with range bounds or high 1m velocity
        # ----------------------------------------------------------------------
        vel_mult = tick_velocity if tick_velocity > 0 else 1.0
        dial_pct = 78.0 if vel_mult >= 1.5 else (74.0 if is_london_ny_active else 72.0)
        threshold = 62
        reason_str = f"High Velocity Flow inside Box ({'London/NY Active' if is_london_ny_active else 'Velocity Surge'})"
        pill_text = f"🧠 Auto-Brain: {dial_pct:.0f}% (Fast Scalp: Flow Momentum | Min: {threshold})"
        full_reason = f"Fast Hit-and-Run Scalp Mode — {reason_str} | Threshold: {threshold}/100"

        self.auto_brain_state = {
            "is_auto": (self.mode == "auto_adaptive"),
            "sub_regime": "FAST_SCALP",
            "dial_pct": dial_pct,
            "active_threshold": threshold,
            "pill_text": pill_text,
            "full_reason": full_reason,
            "reason": reason_str,
            "target_rr": 1.2,
            "be_lock_r": 0.45,
            "stall_exit_seconds": 4.0,
            "last_update": now,
        }
        return self.auto_brain_state

    def evaluate(
        self,
        symbol: str,
        dfs: Dict[str, pd.DataFrame],
        current_tick_bid: float,
        current_tick_ask: float,
        is_impulse_active: bool = False,
        utc_timestamp: Optional[float] = None,
    ) -> Tuple[Optional[ConfluenceSignal], Optional[ConfluenceSignal]]:
        """
        Evaluate market data for both BUY and SELL directions.
        Returns (buy_signal, sell_signal).
        """
        now_ts = utc_timestamp or time.time()

        buy_signal = self._evaluate_direction(
            symbol=symbol,
            direction="BUY",
            dfs=dfs,
            current_price=current_tick_ask,
            is_impulse_active=is_impulse_active,
            utc_timestamp=now_ts,
        )

        sell_signal = self._evaluate_direction(
            symbol=symbol,
            direction="SELL",
            dfs=dfs,
            current_price=current_tick_bid,
            is_impulse_active=is_impulse_active,
            utc_timestamp=now_ts,
        )

        return buy_signal, sell_signal

    def _evaluate_direction(
        self,
        symbol: str,
        direction: str,
        dfs: Dict[str, pd.DataFrame],
        current_price: float,
        is_impulse_active: bool,
        utc_timestamp: float,
    ) -> Optional[ConfluenceSignal]:
        """Score a specific directional setup (0 to 100 points)."""
        df_1h = dfs.get("1H")
        df_15m = dfs.get("15m")
        df_5m = dfs.get("5m")
        df_1m = dfs.get("1m")

        if df_5m is None or len(df_5m) < 20 or df_1m is None or len(df_1m) < 10:
            return None

        breakdown = {
            "h1_macro": 0,
            "m15_wave": 0,
            "m5_setup": 0,
            "m1_trigger": 0,
            "smc_institutional": 0,
            "quant_alpha": 0,
        }
        reasons: List[str] = []

        # -------------------------------------------------------------
        # 1. 1-Hour (1H) Macro Bias & Room-to-Run (Max 20 Points)
        # -------------------------------------------------------------
        h1_ema50_aligned = False
        h1_room_to_run = True

        if df_1h is not None and len(df_1h) >= 20:
            c_1h = df_1h["close"].values
            h_1h = df_1h["high"].values
            l_1h = df_1h["low"].values

            ema50_1h = TechnicalIndicators.ema(c_1h, min(50, len(c_1h)))
            h1_ema_slope = TechnicalIndicators.ema_slope(c_1h, min(50, len(c_1h)))
            last_c_1h = c_1h[-1]
            last_ema50 = ema50_1h[-1] if not np.isnan(ema50_1h[-1]) else last_c_1h

            if direction == "BUY":
                if last_c_1h >= last_ema50 or h1_ema_slope > 0.0:
                    h1_ema50_aligned = True
                    breakdown["h1_macro"] += 10
                    reasons.append("1H EMA50 Bullish Flow (+10)")
                elif abs(last_c_1h - last_ema50) / last_c_1h < 0.002:
                    h1_ema50_aligned = True
                    breakdown["h1_macro"] += 8
                    reasons.append("1H EMA50 Support Bounce (+8)")
            else: # SELL
                if last_c_1h <= last_ema50 or h1_ema_slope < 0.0:
                    h1_ema50_aligned = True
                    breakdown["h1_macro"] += 10
                    reasons.append("1H EMA50 Bearish Flow (+10)")
                elif abs(last_c_1h - last_ema50) / last_c_1h < 0.002:
                    h1_ema50_aligned = True
                    breakdown["h1_macro"] += 8
                    reasons.append("1H EMA50 Resistance Reject (+8)")

            swings_1h = TechnicalIndicators.find_swings(h_1h, l_1h, window=3)
            atr_1h = TechnicalIndicators.atr(h_1h, l_1h, c_1h, period=14)[-1]
            est_target_dist = (atr_1h * 1.5) if not np.isnan(atr_1h) and atr_1h > 0 else 3.0

            h1_room_to_run = TechnicalIndicators.check_room_to_run(
                entry_price=current_price,
                trade_direction=direction,
                target_r_distance=est_target_dist,
                swing_highs=swings_1h["highs"],
                swing_lows=swings_1h["lows"],
            )

            if h1_room_to_run:
                breakdown["h1_macro"] += 10
                reasons.append("1H Room-to-Run Clear (+10)")
            else:
                reasons.append("1H Opposing S/R Wall Nearby (+0)")
        else:
            breakdown["h1_macro"] = 10
            h1_room_to_run = True
            reasons.append("1H Neutral Baseline (+10)")

        # -------------------------------------------------------------
        # 2. 15-Minute (15m) Session Wave & Momentum (Max 25 Points)
        # -------------------------------------------------------------
        m15_adx_val = 20.0
        m15_trend_aligned = False

        if df_15m is not None and len(df_15m) >= 20:
            c_15m = df_15m["close"].values
            h_15m = df_15m["high"].values
            l_15m = df_15m["low"].values

            ema20_15m = TechnicalIndicators.ema(c_15m, 20)[-1]
            ema50_15m = TechnicalIndicators.ema(c_15m, min(50, len(c_15m)))[-1]
            adx_dict_15m = TechnicalIndicators.adx(h_15m, l_15m, c_15m, 14)
            m15_adx_val = adx_dict_15m["adx"][-1] if not np.isnan(adx_dict_15m["adx"][-1]) else 20.0

            if direction == "BUY":
                if c_15m[-1] >= ema20_15m >= ema50_15m:
                    m15_trend_aligned = True
                    breakdown["m15_wave"] += 15
                    reasons.append("15m Strong Bull Wave (+15)")
                elif c_15m[-1] >= ema20_15m:
                    m15_trend_aligned = True
                    breakdown["m15_wave"] += 10
                    reasons.append("15m Above EMA20 (+10)")
            else: # SELL
                if c_15m[-1] <= ema20_15m <= ema50_15m:
                    m15_trend_aligned = True
                    breakdown["m15_wave"] += 15
                    reasons.append("15m Strong Bear Wave (+15)")
                elif c_15m[-1] <= ema20_15m:
                    m15_trend_aligned = True
                    breakdown["m15_wave"] += 10
                    reasons.append("15m Below EMA20 (+10)")

            if m15_trend_aligned:
                if m15_adx_val >= 22.0:
                    breakdown["m15_wave"] += 10
                    reasons.append(f"15m Trend & ADX Momentum ({m15_adx_val:.1f} > 22) (+10)")
                elif m15_adx_val >= 18.0:
                    breakdown["m15_wave"] += 6
                    reasons.append(f"15m Trend Active ({m15_adx_val:.1f} >= 18) (+6)")
            else:
                if m15_adx_val >= 25.0:
                    breakdown["m15_wave"] = max(0, breakdown["m15_wave"] - 8)
                    reasons.append(f"15m Strong Opposing ADX ({m15_adx_val:.1f} > 25) (-8 Penalty)")
                else:
                    reasons.append(f"15m ADX Chop/Neutral ({m15_adx_val:.1f}) (+0)")
        else:
            breakdown["m15_wave"] = 12
            reasons.append("15m Baseline (+12)")

        # -------------------------------------------------------------
        # 3. 5-Minute (5m) Setup Structure (Max 30 Points)
        # -------------------------------------------------------------
        c_5m = df_5m["close"].values
        h_5m = df_5m["high"].values
        l_5m = df_5m["low"].values

        ema20_5m = TechnicalIndicators.ema(c_5m, 20)[-1]
        ema50_5m = TechnicalIndicators.ema(c_5m, min(50, len(c_5m)))[-1]
        rsi_5m = TechnicalIndicators.rsi(c_5m, 14)[-1]

        # Setup Type A: Trend Pullback into EMA Value Zone / Trend Momentum Flow
        if direction == "BUY":
            if (ema50_5m * 0.998 <= c_5m[-1] <= ema20_5m * 1.002) and (38.0 <= rsi_5m <= 58.0):
                breakdown["m5_setup"] += 30
                reasons.append(f"5m Pullback into Value Zone (RSI {rsi_5m:.1f}) (+30)")
            elif (42.0 <= rsi_5m <= 72.0) and c_5m[-1] >= ema20_5m * 0.999:
                breakdown["m5_setup"] += 25
                reasons.append(f"5m Bullish Momentum Flow (RSI {rsi_5m:.1f}) (+25)")
            elif (35.0 <= rsi_5m <= 65.0):
                breakdown["m5_setup"] += 15
                reasons.append(f"5m General Bull Support Zone (RSI {rsi_5m:.1f}) (+15)")
        else: # SELL
            if (ema20_5m * 0.998 <= c_5m[-1] <= ema50_5m * 1.002) and (42.0 <= rsi_5m <= 62.0):
                breakdown["m5_setup"] += 30
                reasons.append(f"5m Pullback into Value Zone (RSI {rsi_5m:.1f}) (+30)")
            elif (28.0 <= rsi_5m <= 58.0) and c_5m[-1] <= ema20_5m * 1.001:
                breakdown["m5_setup"] += 25
                reasons.append(f"5m Bearish Momentum Flow (RSI {rsi_5m:.1f}) (+25)")
            elif (35.0 <= rsi_5m <= 65.0):
                breakdown["m5_setup"] += 15
                reasons.append(f"5m General Bear Resistance Zone (RSI {rsi_5m:.1f}) (+15)")

        # Setup Type B: Consolidation Box Breakout
        if breakdown["m5_setup"] < 30 and len(df_5m) >= 12:
            recent_high = np.max(h_5m[-10:-1])
            recent_low = np.min(l_5m[-10:-1])

            dt = datetime.fromtimestamp(utc_timestamp, tz=timezone.utc)
            hour = dt.hour + (dt.minute / 60.0)
            in_volume_window = (7.0 <= hour <= 11.5) or (12.5 <= hour <= 17.0)

            if direction == "BUY" and c_5m[-1] >= recent_high:
                pts = 30 if in_volume_window else 22
                breakdown["m5_setup"] = max(breakdown["m5_setup"], pts)
                reasons.append(f"5m Box Breakout (+{pts})")
            elif direction == "SELL" and c_5m[-1] <= recent_low:
                pts = 30 if in_volume_window else 22
                breakdown["m5_setup"] = max(breakdown["m5_setup"], pts)
                reasons.append(f"5m Box Breakout (+{pts})")

        # Setup Type C: Strong Trend Momentum Continuation
        if breakdown["m5_setup"] < 20:
            if direction == "BUY" and c_5m[-1] >= ema20_5m and rsi_5m >= 45.0:
                breakdown["m5_setup"] = max(breakdown["m5_setup"], 20)
                reasons.append(f"5m Trend Flow (RSI {rsi_5m:.1f}) (+20)")
            elif direction == "SELL" and c_5m[-1] <= ema20_5m and rsi_5m <= 55.0:
                breakdown["m5_setup"] = max(breakdown["m5_setup"], 20)
                reasons.append(f"5m Trend Flow (RSI {rsi_5m:.1f}) (+20)")

        # Exhaustion Guard: Avoid buying at extreme overbought or selling at extreme oversold
        if direction == "BUY" and rsi_5m > 76.0:
            breakdown["m5_setup"] = max(0, breakdown["m5_setup"] - 10)
            reasons.append(f"5m Exhaustion Warning: Overbought RSI {rsi_5m:.1f} (-10)")
        elif direction == "SELL" and rsi_5m < 24.0:
            breakdown["m5_setup"] = max(0, breakdown["m5_setup"] - 10)
            reasons.append(f"5m Exhaustion Warning: Oversold RSI {rsi_5m:.1f} (-10)")

        # -------------------------------------------------------------
        # 4. 1-Minute (1m) Sniper Execution Trigger (Max 25 Points)
        # -------------------------------------------------------------
        c_1m = df_1m["close"].values
        o_1m = df_1m["open"].values
        h_1m = df_1m["high"].values
        l_1m = df_1m["low"].values

        swings_5m = TechnicalIndicators.find_swings(h_5m, l_5m, window=3)
        last_sh_5m = swings_5m["highs"][-1]["price"] if swings_5m["highs"] else None
        last_sl_5m = swings_5m["lows"][-1]["price"] if swings_5m["lows"] else None

        trigger_data = TechnicalIndicators.detect_candle_trigger(
            open_p=o_1m[-1],
            high_p=h_1m[-1],
            low_p=l_1m[-1],
            close_p=c_1m[-1],
            prev_open=o_1m[-2],
            prev_high=h_1m[-2],
            prev_low=l_1m[-2],
            prev_close=c_1m[-2],
            recent_swing_high=last_sh_5m,
            recent_swing_low=last_sl_5m,
        )

        rsi_1m = TechnicalIndicators.rsi(c_1m, 14)
        rsi_hooks = TechnicalIndicators.detect_rsi_hook(rsi_1m)

        if direction == "BUY":
            if trigger_data["bullish_engulfing"]:
                breakdown["m1_trigger"] += 15
                reasons.append("1m Bullish Engulfing (+15)")
            elif trigger_data["bullish_sweep"]:
                breakdown["m1_trigger"] += 15
                reasons.append("1m Liquidity Sweep Rejection (+15)")
            elif is_impulse_active:
                breakdown["m1_trigger"] += 15
                reasons.append("High-Velocity Tick Impulse Breakout (+15)")
            elif c_1m[-1] > o_1m[-1]:
                pts = 12 if trigger_data["strong_momentum"] else 8
                breakdown["m1_trigger"] += pts
                reasons.append(f"1m Bullish Candle Flow (+{pts})")

            if rsi_hooks["bullish_hook"]:
                breakdown["m1_trigger"] += 10
                reasons.append("1m RSI Bullish Hook (+10)")
            elif rsi_1m[-1] >= 50.0:
                breakdown["m1_trigger"] += 8
                reasons.append("1m RSI Bullish Flow (+8)")

        else: # SELL
            if trigger_data["bearish_engulfing"]:
                breakdown["m1_trigger"] += 15
                reasons.append("1m Bearish Engulfing (+15)")
            elif trigger_data["bearish_sweep"]:
                breakdown["m1_trigger"] += 15
                reasons.append("1m Liquidity Sweep Rejection (+15)")
            elif is_impulse_active:
                breakdown["m1_trigger"] += 15
                reasons.append("High-Velocity Tick Impulse Breakout (+15)")
            elif c_1m[-1] < o_1m[-1]:
                pts = 12 if trigger_data["strong_momentum"] else 8
                breakdown["m1_trigger"] += pts
                reasons.append(f"1m Bearish Candle Flow (+{pts})")

            if rsi_hooks["bearish_hook"]:
                breakdown["m1_trigger"] += 10
                reasons.append("1m RSI Bearish Hook (+10)")
            elif rsi_1m[-1] <= 50.0:
                breakdown["m1_trigger"] += 8
                reasons.append("1m RSI Bearish Flow (+8)")

        breakdown["m1_trigger"] = min(25, breakdown["m1_trigger"])

        # -------------------------------------------------------------
        # 4b. Institutional Smart Money Concepts (SMC) via smartmoneyconcepts
        # -------------------------------------------------------------
        smc_5m = TechnicalIndicators.detect_smc(df_5m, swing_length=3) if df_5m is not None else {}
        smc_15m = TechnicalIndicators.detect_smc(df_15m, swing_length=5) if df_15m is not None else {}

        smc_pts = 0
        if direction == "BUY":
            if smc_5m.get("unmitigated_bullish_ob") or smc_15m.get("unmitigated_bullish_ob"):
                smc_pts += 8
                reasons.append("Institutional Bullish Order Block (OB) Retest (+8)")
            if smc_5m.get("bullish_fvg") or smc_15m.get("bullish_fvg"):
                smc_pts += 6
                reasons.append("Fair Value Gap (FVG) Imbalance Tap (+6)")
            if smc_5m.get("recent_bullish_choch") or smc_15m.get("recent_bullish_choch"):
                smc_pts += 6
                reasons.append("Institutional CHOCH Bullish Reversal (+6)")
            elif smc_5m.get("recent_bullish_bos") or smc_15m.get("recent_bullish_bos"):
                smc_pts += 4
                reasons.append("Break of Structure (BOS) Bullish Continuation (+4)")

            if smc_5m.get("unmitigated_bearish_ob"):
                smc_pts -= 8
                reasons.append("Opposing Bearish Supply Block Ahead (-8)")

        else: # SELL
            if smc_5m.get("unmitigated_bearish_ob") or smc_15m.get("unmitigated_bearish_ob"):
                smc_pts += 8
                reasons.append("Institutional Bearish Order Block (OB) Retest (+8)")
            if smc_5m.get("bearish_fvg") or smc_15m.get("bearish_fvg"):
                smc_pts += 6
                reasons.append("Fair Value Gap (FVG) Imbalance Tap (+6)")
            if smc_5m.get("recent_bearish_choch") or smc_15m.get("recent_bearish_choch"):
                smc_pts += 6
                reasons.append("Institutional CHOCH Bearish Reversal (+6)")
            elif smc_5m.get("recent_bearish_bos") or smc_15m.get("recent_bearish_bos"):
                smc_pts += 4
                reasons.append("Break of Structure (BOS) Bearish Continuation (+4)")

            if smc_5m.get("unmitigated_bullish_ob"):
                smc_pts -= 8
                reasons.append("Opposing Bullish Demand Block Below (-8)")

        breakdown["smc_institutional"] = min(20, max(-15, smc_pts))

        # -------------------------------------------------------------
        # 4c. Proven Open-Source Quantitative Alpha (ta & scipy)
        # - VWAP Institutional Benchmark (ta.volume.VolumeWeightedAveragePrice)
        # - Kaufman Adaptive Moving Average (ta.momentum.KAMAIndicator)
        # - Statistical Trend Quality & R^2 (scipy.stats.linregress)
        # - Z-Score Statistical Exhaustion & Mean-Reversion Pullback Filters
        # -------------------------------------------------------------
        quant_pts = 0
        vwap_data = TechnicalIndicators.calculate_vwap(df_5m)
        kama_data = TechnicalIndicators.calculate_kama(df_5m, window=10)
        stat_data = TechnicalIndicators.calculate_statistical_trend(df_5m["close"].values, window=15)
        zscore_data = TechnicalIndicators.calculate_zscore(df_5m["close"].values, window=20)

        if direction == "BUY":
            # 1. VWAP Confirmation
            if vwap_data.get("is_above"):
                quant_pts += 6
                reasons.append(f"Above Institutional VWAP ({vwap_data['vwap']:.2f}) (+6)")
            elif vwap_data.get("is_below"):
                quant_pts -= 6
                reasons.append("Below Institutional VWAP (-6 Penalty)")

            # 2. Kaufman Adaptive MA (KAMA) Trend Alignment
            if kama_data.get("is_above"):
                quant_pts += 4
                reasons.append("Above Kaufman Adaptive Moving Average (+4)")

            # 3. Statistical R^2 Trend Quality
            if stat_data.get("is_strong_trend") and stat_data.get("slope", 0) > 0:
                quant_pts += 6
                reasons.append(f"Statistically Verified Bull Trend (R² {stat_data['r_squared']:.2f} >= 0.60) (+6)")
            elif stat_data.get("is_random_walk"):
                quant_pts -= 5
                reasons.append(f"Statistical Chop / High Entropy Random Walk (R² {stat_data['r_squared']:.2f} < 0.25) (-5)")

            # 4. Statistical Z-Score Exhaustion & Discount Zones
            if zscore_data.get("is_extreme_overbought"):
                quant_pts -= 10
                reasons.append(f"Exhaustion Guard: Z-Score Overbought (+{zscore_data['zscore']:.1f}σ) - Blocked Buying Top (-10)")
            elif zscore_data.get("is_bull_discount"):
                quant_pts += 4
                reasons.append(f"Statistical Mean Retest Discount Zone (Z {zscore_data['zscore']:.1f}σ) (+4)")

        else: # SELL
            # 1. VWAP Confirmation
            if vwap_data.get("is_below"):
                quant_pts += 6
                reasons.append(f"Below Institutional VWAP ({vwap_data['vwap']:.2f}) (+6)")
            elif vwap_data.get("is_above"):
                quant_pts -= 6
                reasons.append("Above Institutional VWAP (-6 Penalty)")

            # 2. Kaufman Adaptive MA (KAMA) Trend Alignment
            if kama_data.get("is_below"):
                quant_pts += 4
                reasons.append("Below Kaufman Adaptive Moving Average (+4)")

            # 3. Statistical R^2 Trend Quality
            if stat_data.get("is_strong_trend") and stat_data.get("slope", 0) < 0:
                quant_pts += 6
                reasons.append(f"Statistically Verified Bear Trend (R² {stat_data['r_squared']:.2f} >= 0.60) (+6)")
            elif stat_data.get("is_random_walk"):
                quant_pts -= 5
                reasons.append(f"Statistical Chop / High Entropy Random Walk (R² {stat_data['r_squared']:.2f} < 0.25) (-5)")

            # 4. Statistical Z-Score Exhaustion & Premium Zones
            if zscore_data.get("is_extreme_oversold"):
                quant_pts -= 10
                reasons.append(f"Exhaustion Guard: Z-Score Oversold ({zscore_data['zscore']:.1f}σ) - Blocked Selling Bottom (-10)")
            elif zscore_data.get("is_bear_premium"):
                quant_pts += 4
                reasons.append(f"Statistical Mean Retest Premium Zone (Z {zscore_data['zscore']:.1f}σ) (+4)")

        breakdown["quant_alpha"] = min(20, max(-15, quant_pts))

        # Fundamental News Study Integration
        news_bias = self.auto_brain_state.get("news_bias")
        if news_bias and news_bias != "NEUTRAL":
            if news_bias == direction:
                breakdown["news_bias"] = 15
                reasons.append("Fundamental News Study Alignment (+15)")
            else:
                breakdown["news_bias"] = -20
                reasons.append("Opposing Fundamental News Bias (-20)")

        # -------------------------------------------------------------
        # Institutional Multi-Timeframe Trend Confluence Filter
        # -------------------------------------------------------------
        # 1. HARD COUNTER-TREND GATE: Never take a trade opposing BOTH 1H and 15m trends!
        if not h1_ema50_aligned and not m15_trend_aligned:
            logger.debug(f"[{symbol} {direction}] Filtered: Hard counter-trend opposing both 1H and 15m HTF.")
            return None

        # 2. HTF Divergence Penalty: If 1H or 15m is not aligned, apply heavy penalty
        htf_diverged = (not h1_ema50_aligned) or (not m15_trend_aligned)
        if htf_diverged:
            breakdown["htf_divergence"] = -25
            reasons.append("Higher-Timeframe Trend Divergence (-25 Penalty)")

        # Total Confluence Score
        total_score = sum(breakdown.values())
        if htf_diverged:
            # Strictly cap total score at 68 when HTF is not unanimous (NO SNIPER ALLOWED)
            total_score = min(total_score, 68)
        total_score = min(100, max(0, total_score))

        # Determine Tier: TIER_1_SNIPER requires unanimous HTF alignment!
        if total_score >= 82 and not htf_diverged:
            tier = "TIER_1_SNIPER"
        elif total_score >= 68:
            tier = "TIER_2_BALANCED"
        elif total_score >= 55:
            tier = "TIER_3_AGGRESSIVE"
        else:
            tier = "NONE"

        # Check against active threshold
        active_thresh = self.get_active_threshold()
        if total_score < active_thresh:
            return None # Does not cross active threshold

        # -------------------------------------------------------------
        # 5. Adaptive Regime Classification (SCALP vs RUNNER)
        # -------------------------------------------------------------
        is_gold = any(k in symbol.upper() for k in ["XAU", "GOLD"])
        if df_5m is not None and len(df_5m) >= 5:
            atr_5m = TechnicalIndicators.atr(df_5m["high"].values, df_5m["low"].values, df_5m["close"].values, 14)[-1]
            atr_buffer = (atr_5m * 1.5) if (not np.isnan(atr_5m) and atr_5m > 0) else (2.80 if is_gold else 0.0020)
            if direction == "BUY":
                recent_low = min(df_5m["low"].values[-5:])
                raw_sl_dist = max(current_price - (recent_low - (0.40 if is_gold else 0.0004)), atr_buffer)
            else:
                recent_high = max(df_5m["high"].values[-5:])
                raw_sl_dist = max((recent_high + (0.40 if is_gold else 0.0004)) - current_price, atr_buffer)
        else:
            if direction == "BUY":
                recent_low = min(l_1m[-5:])
                raw_sl_dist = current_price - (recent_low - (0.40 if is_gold else 0.0004))
            else:
                recent_high = max(h_1m[-5:])
                raw_sl_dist = (recent_high + (0.40 if is_gold else 0.0004)) - current_price

        min_sl = 2.50 if is_gold else 0.0015
        max_sl = 4.80 if is_gold else 0.0045
        risk_r_dist = float(np.clip(raw_sl_dist, min_sl, max_sl))

        # Sub-regime and trade parameters
        if self.mode == "auto_adaptive":
            sub_regime = self.auto_brain_state.get("sub_regime", "WAVE_RUNNER")
            target_rr = float(self.auto_brain_state.get("target_rr", 2.5))
            be_lock_r = float(self.auto_brain_state.get("be_lock_r", 0.65))
            stall_exit_seconds = float(self.auto_brain_state.get("stall_exit_seconds", 90.0))
        elif self.mode in ["hyper", "aggressive"]:
            sub_regime = "FAST_SCALP"
            target_rr = 1.8 if self.mode == "hyper" else 2.0
            be_lock_r = 0.65
            stall_exit_seconds = 45.0 if self.mode == "hyper" else 60.0
        elif self.mode == "cautious":
            sub_regime = "WAVE_RUNNER"
            target_rr = 2.5
            be_lock_r = 0.70
            stall_exit_seconds = 90.0
        else: # Standard / Manual
            sub_regime = "WAVE_RUNNER" if total_score >= 68 else "FAST_SCALP"
            target_rr = 2.2
            be_lock_r = 0.65
            stall_exit_seconds = 75.0

        if sub_regime == "FAST_SCALP":
            regime = "ADAPTIVE_SCALP"
        elif sub_regime == "DEFENSIVE_SHIELD":
            regime = "ADAPTIVE_RUNNER" if total_score >= 82 else "ADAPTIVE_SCALP"
        else:
            regime = "ADAPTIVE_RUNNER"

        # -------------------------------------------------------------
        # 5b. Smart Order Type Selection (MARKET vs LIMIT with Predictive Guard)
        # -------------------------------------------------------------
        order_type = "MARKET"
        limit_price = None
        invalidation_price = None

        ema9_1m = TechnicalIndicators.ema(c_1m, 9)
        ema9_1m_val = ema9_1m[-1] if len(ema9_1m) > 0 and not np.isnan(ema9_1m[-1]) else current_price
        
        # Limit Orders ONLY in WAVE_RUNNER mode when HTF is unanimously aligned and not near news/impulse
        if is_gold and not is_impulse_active and not htf_diverged and sub_regime == "WAVE_RUNNER":
            pullback_dist = abs(current_price - ema9_1m_val)
            if direction == "BUY" and current_price > (ema9_1m_val + 0.40):
                order_type = "BUY_LIMIT"
                limit_price = round(ema9_1m_val + 0.10, 2)
                invalidation_price = round(limit_price - 1.50, 2)
                reasons.append(f"Smart BUY_LIMIT @ {limit_price:.2f} (EMA 9 pullback)")
            elif direction == "SELL" and current_price < (ema9_1m_val - 0.40):
                order_type = "SELL_LIMIT"
                limit_price = round(ema9_1m_val - 0.10, 2)
                invalidation_price = round(limit_price + 1.50, 2)
                reasons.append(f"Smart SELL_LIMIT @ {limit_price:.2f} (EMA 9 pullback)")


        # Compute suggested SL and TP
        entry_p = limit_price if limit_price else current_price
        if direction == "BUY":
            suggested_sl = round(entry_p - risk_r_dist, 2 if is_gold else 5)
            suggested_tp = round(entry_p + (risk_r_dist * target_rr), 2 if is_gold else 5)
        else:
            suggested_sl = round(entry_p + risk_r_dist, 2 if is_gold else 5)
            suggested_tp = round(entry_p - (risk_r_dist * target_rr), 2 if is_gold else 5)

        full_reason = f"{tier} ({total_score}/100) -> [{sub_regime}] {regime} | " + ", ".join(reasons)

        return ConfluenceSignal(
            symbol=symbol,
            direction=direction,
            score=total_score,
            tier=tier,
            regime=regime,
            sub_regime=sub_regime,
            target_rr=target_rr,
            entry_price=round(entry_p, 2 if is_gold else 5),
            suggested_sl=suggested_sl,
            suggested_tp=suggested_tp,
            risk_r_dist=round(risk_r_dist, 2 if is_gold else 5),
            be_lock_r=be_lock_r,
            stall_exit_seconds=stall_exit_seconds,
            breakdown=breakdown,
            reason=full_reason,
            time=utc_timestamp,
            order_type=order_type,
            limit_price=limit_price,
            invalidation_price=invalidation_price,
        )
