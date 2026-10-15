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
    is_pre_news_trajectory: bool = False
    pre_news_trajectory_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SignalResult(tuple):
    """
    Two-tuple of (buy_signal, sell_signal) supporting tuple unpacking
    as well as direct property delegation to the primary active signal.
    """
    def __new__(cls, buy_signal: Optional[ConfluenceSignal], sell_signal: Optional[ConfluenceSignal]):
        return super().__new__(cls, (buy_signal, sell_signal))

    @property
    def buy(self) -> Optional[ConfluenceSignal]:
        return self[0]

    @property
    def sell(self) -> Optional[ConfluenceSignal]:
        return self[1]

    @property
    def primary(self) -> Optional[ConfluenceSignal]:
        return self[0] if self[0] is not None else self[1]

    def __getattr__(self, name: str) -> Any:
        p = self.primary
        if p is not None and hasattr(p, name):
            return getattr(p, name)
        raise AttributeError(f"'SignalResult' object has no attribute '{name}'")


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
        self.allow_pre_news_trajectory: bool = True

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
        elif m in ["hyper", "most_aggressive", "tier_4", "scalp", "scalping", "fast_scalp"]:
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
        elif self.mode in ["hyper", "scalp", "scalping", "fast_scalp"]:
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
        elif self.mode in ["hyper", "scalp", "scalping", "fast_scalp"]:
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
        dfs: Optional[Dict[str, pd.DataFrame]] = None,
        current_spread: float = 0.25,
        max_spread: float = 0.80,
        tick_velocity: float = 1.0,
        news_reactor: Any = None,
        symbol: str = "XAUUSD",
        now_ts: Optional[float] = None,
        df_1h: Optional[pd.DataFrame] = None,
        df_15m: Optional[pd.DataFrame] = None,
        df_5m: Optional[pd.DataFrame] = None,
        df_1m: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Sub-Regime Brain Evaluator (polled every 5 seconds).
        Dynamically classifies market environment into:
        1. DEFENSIVE_SHIELD (25%-40% Dial | Min Score 80)
        2. WAVE_RUNNER     (55%-68% Dial | Min Score 66)
        3. FAST_SCALP      (72%-85% Dial | Min Score 62)
        """
        now = now_ts or time.time()

        if dfs is None:
            dfs = {}
        if df_1h is not None:
            dfs["1H"] = df_1h
        if df_15m is not None:
            dfs["15m"] = df_15m
        if df_5m is not None:
            dfs["5m"] = df_5m
        if df_1m is not None:
            dfs["1m"] = df_1m

        # Update real-time mathematical directional bias across all 4 timeframes
        self.tf_trends = TechnicalIndicators.get_4tf_trends(dfs)
        self.auto_brain_state["tf_trends"] = self.tf_trends

        # Rate-limit evaluation updates to once every 2.0s to avoid UI jitter (skip if now_ts provided)
        last_up = self.auto_brain_state.get("last_update", 0.0)
        if now_ts is None and (now - last_up < 2.0):
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

        is_news_freeze = False
        if news_reactor:
            cal_state = news_reactor.check_calendar_state(current_utc=now)
            is_news_freeze = cal_state.get("is_news_freeze", False)
            study = news_reactor.analyze_news_fundamental_bias(symbol)
            news_bias_direction = study.get("direction", "NEUTRAL")
            news_confidence = study.get("confidence", 0.50)
            news_summary = study.get("summary", "")

            if is_news_freeze:
                is_near_major_news = True
                ev = cal_state.get("nearest_event") or {}
                news_name = ev.get("title", "High-Impact USD News")
            else:
                cd = cal_state.get("countdown_seconds")
                if cd is not None and cd <= 1200: # Within 20 min
                    is_near_major_news = True
                    ev = cal_state.get("nearest_event") or {}
                    news_name = ev.get("title", "Upcoming High-Impact USD News")

        # ----------------------------------------------------------------------
        # Strict High-Impact News Freeze Gate: HARD LOCK (Cannot be bypassed)
        # ----------------------------------------------------------------------
        if is_news_freeze:
            dial_pct = 10.0
            threshold = 95
            reason_str = f"High-Impact News Freeze Active: {news_name} — Entries Strictly Locked"
            pill_text = f"🛑 NEWS FREEZE: {news_name} | Entries Locked"
            full_reason = f"High-Impact News Safety Gate — {reason_str}"
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
                "is_news_freeze": True,
            }
            return self.auto_brain_state

        # ----------------------------------------------------------------------
        # Smart Pre-News Trajectory Window (Zone A: T-45m to T-90s)
        # Allows automated front-running entries if confidence >= 75%
        # ----------------------------------------------------------------------
        traj_data = {}
        if news_reactor and hasattr(news_reactor, "evaluate_pre_news_trajectory"):
            traj_data = news_reactor.evaluate_pre_news_trajectory(
                symbol=symbol,
                dfs={"15m": df_15m, "5m": df_5m, "1m": df_1m, "1H": df_1h},
                tick_velocity=tick_velocity,
                current_utc=now,
            )

        traj_active = traj_data.get("active", False)
        traj_conf = traj_data.get("confidence", 0.0)
        traj_bias = traj_data.get("trajectory_bias", "NEUTRAL")

        if self.allow_pre_news_trajectory and traj_active and traj_conf >= 0.75 and traj_bias in ("BUY", "SELL") and not is_wide_spread:
            dial_pct = 78.0
            threshold = 63
            ev_title = traj_data.get("event_title", news_name)
            fund_bias = traj_data.get("fundamental_bias", "NEUTRAL")
            reason_str = f"🎯 PRE_NEWS_TRAJECTORY: {traj_bias} on {ev_title} (Confidence {traj_data.get('confidence_pct', 78)}% | Fund: {fund_bias})"
            pill_text = f"🎯 PRE_NEWS_TRAJECTORY: {traj_bias} ({traj_data.get('confidence_pct', 78)}%) | Min: {threshold}"
            full_reason = f"Pre-News Trajectory Engine — {reason_str} | Threshold: {threshold}/100"
            self.auto_brain_state = {
                "is_auto": (self.mode == "auto_adaptive"),
                "sub_regime": "PRE_NEWS_TRAJECTORY",
                "dial_pct": dial_pct,
                "active_threshold": threshold,
                "pill_text": pill_text,
                "full_reason": full_reason,
                "reason": reason_str,
                "target_rr": 3.5,
                "be_lock_r": 0.40,
                "stall_exit_seconds": 60.0,
                "last_update": now,
                "news_bias": traj_bias,
                "news_confidence": traj_conf,
                "is_news_freeze": False,
                "is_pre_news_trajectory": True,
                "trajectory_bias": traj_bias,
                "trajectory_data": traj_data,
            }
            return self.auto_brain_state

        # If news is approaching (outside freeze) and we have strong fundamental clarity (confidence >= 70%):
        # Dial up intelligent News Momentum Study mode!
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
                "is_news_freeze": False,
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
            # 1H Structural Wall Check: distance to nearest opposing 1H swing high/low >= 1.8 * ATR_15m
            atr_15m_ref = 2.50
            if df_15m is not None and len(df_15m) >= 15:
                calc_atr = TechnicalIndicators.atr(df_15m["high"].values, df_15m["low"].values, df_15m["close"].values, 14)[-1]
                if not np.isnan(calc_atr) and calc_atr > 0:
                    atr_15m_ref = float(calc_atr)
            required_room = 1.8 * atr_15m_ref
            cur_p = c_1h[-1]

            if slope >= 0:
                opposing_highs = [sh["price"] for sh in swings_1h.get("highs", []) if sh["price"] > cur_p]
                h1_open_room = (min(opposing_highs) - cur_p >= required_room) if opposing_highs else True
            else:
                opposing_lows = [sl["price"] for sl in swings_1h.get("lows", []) if sl["price"] < cur_p]
                h1_open_room = (cur_p - max(opposing_lows) >= required_room) if opposing_lows else True

        # ----------------------------------------------------------------------
        # Sub-Regime 2: Institutional Wave Runner (55%-68% Aggressiveness)
        # Detected when 1H + 15m align, 15m ADX >= 22.0, and open room to run > 2.5R
        # ----------------------------------------------------------------------
        if h1_aligned and (m15_adx >= 22.0) and h1_open_room:
            dial_pct = 64.0 if m15_adx >= 25.0 else 58.0
            threshold = 74
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
                "target_rr": 2.8,
                "be_lock_r": 0.65,
                "stall_exit_seconds": 120.0,
                "last_update": now,
            }
            return self.auto_brain_state

        # ----------------------------------------------------------------------
        # Sub-Regime 1: Fast Hit-and-Run Scalp (72%-85% Aggressiveness)
        # Detected during active liquidity with range bounds or high 1m velocity
        # ----------------------------------------------------------------------
        vel_mult = tick_velocity if tick_velocity > 0 else 1.0
        dial_pct = 78.0 if vel_mult >= 1.5 else (74.0 if is_london_ny_active else 72.0)
        threshold = 75
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
            "target_rr": 2.2,
            "be_lock_r": 0.65,
            "stall_exit_seconds": 60.0,
            "last_update": now,
        }
        return self.auto_brain_state

    def evaluate(
        self,
        symbol: str,
        dfs: Optional[Dict[str, pd.DataFrame]] = None,
        current_tick_bid: Optional[float] = None,
        current_tick_ask: Optional[float] = None,
        is_impulse_active: bool = False,
        utc_timestamp: Optional[float] = None,
        df_1h: Optional[pd.DataFrame] = None,
        df_15m: Optional[pd.DataFrame] = None,
        df_5m: Optional[pd.DataFrame] = None,
        df_1m: Optional[pd.DataFrame] = None,
        tick_velocity: Optional[float] = None,
        **kwargs: Any,
    ) -> SignalResult:
        """
        Evaluate market data for both BUY and SELL directions.
        Returns SignalResult(buy_signal, sell_signal).
        """
        now_ts = utc_timestamp or time.time()

        if dfs is None:
            dfs = {}
        if df_1h is not None:
            dfs["1H"] = df_1h
        if df_15m is not None:
            dfs["15m"] = df_15m
        if df_5m is not None:
            dfs["5m"] = df_5m
        if df_1m is not None:
            dfs["1m"] = df_1m

        if current_tick_bid is None:
            ref_df = dfs.get("1m") if dfs.get("1m") is not None else (dfs.get("5m") if dfs.get("5m") is not None else dfs.get("15m"))
            if ref_df is not None and not ref_df.empty:
                current_tick_bid = float(ref_df["close"].iloc[-1])
            else:
                current_tick_bid = 2650.0
        if current_tick_ask is None:
            current_tick_ask = round(current_tick_bid + 0.25, 2)

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

        return SignalResult(buy_signal, sell_signal)

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

        # Hard Gate: Strictly block new trade evaluations during active high-impact news freeze
        if self.auto_brain_state.get("is_news_freeze", False):
            logger.debug(f"[{symbol} {direction}] Blocked: High-impact news freeze is active.")
            return None

        is_gold = any(k in symbol.upper() for k in ["XAU", "GOLD"])

        # =============================================================
        # 0. INSTITUTIONAL SUPPLY & DEMAND ZONE + LIQUIDITY TRAP ENGINE
        # =============================================================
        zones = TechnicalIndicators.detect_supply_demand_zones(
            df_5m=df_5m,
            df_15m=df_15m,
            current_price=current_price,
            is_gold=is_gold
        )
        traps = TechnicalIndicators.detect_liquidity_traps(
            df_5m=df_5m,
            current_price=current_price,
            is_gold=is_gold
        )

        # Execution 1m candle metrics
        c_1m = df_1m["close"].values
        o_1m = df_1m["open"].values
        h_1m = df_1m["high"].values
        l_1m = df_1m["low"].values
        is_green_1m = (c_1m[-1] > o_1m[-1])
        is_hammer_1m = (c_1m[-1] - l_1m[-1]) > 1.8 * max(0.05, abs(c_1m[-1] - o_1m[-1]))
        is_red_1m = (c_1m[-1] < o_1m[-1])
        is_star_1m = (h_1m[-1] - c_1m[-1]) > 1.8 * max(0.05, abs(c_1m[-1] - o_1m[-1]))

        # High-probability Zone Reversals at Range Extremes (including verified bear/bull traps)
        is_demand_reversal = (
            direction == "BUY" and
            (zones.get("in_demand_zone", False) or traps.get("is_bear_trap", False)) and
            (zones.get("liquidity_sweep_bull", False) or traps.get("is_bear_trap", False) or is_green_1m or is_hammer_1m)
        )
        is_supply_reversal = (
            direction == "SELL" and
            (zones.get("in_supply_zone", False) or traps.get("is_bull_trap", False)) and
            (zones.get("liquidity_sweep_bear", False) or traps.get("is_bull_trap", False) or is_red_1m or is_star_1m)
        )
        is_zone_reversal = is_demand_reversal or is_supply_reversal

        # STRICT INSTITUTIONAL ZONE BOUND PROTECTION:
        # Never sell into a Demand/Support Zone, never buy into a Supply/Resistance Zone!
        is_pre_news_aligned = (
            self.auto_brain_state.get("sub_regime") == "PRE_NEWS_TRAJECTORY" and
            self.auto_brain_state.get("trajectory_bias") == direction and
            self.auto_brain_state.get("is_pre_news_trajectory", False)
        )
        if direction == "SELL" and zones.get("in_demand_zone", False) and not traps.get("is_bull_trap", False) and not is_pre_news_aligned and not is_impulse_active:
            logger.debug(f"[{symbol} SELL] BLOCKED: Inside Demand Zone ({zones['demand_floor']:.2f} - {zones['demand_ceiling']:.2f}). No selling into support floor!")
            return None
        if direction == "BUY" and zones.get("in_supply_zone", False) and not traps.get("is_bear_trap", False) and not is_pre_news_aligned and not is_impulse_active:
            logger.debug(f"[{symbol} BUY] BLOCKED: Inside Supply Zone ({zones['supply_floor']:.2f} - {zones['supply_ceiling']:.2f}). No buying into ceiling!")
            return None

        # =============================================================
        # 1. NON-NEGOTIABLE HIGHER-TIMEFRAME (HTF) DIRECTIONAL ALIGNMENT
        # Institutional Rule #1: Trade strictly WITH the macro wind.
        # - If 1H is BULL and 15m is BULL -> ONLY BUY. Selling is strictly blocked.
        # - If 1H is BEAR and 15m is BEAR -> ONLY SELL. Buying is strictly blocked.
        # - If 1H != 15m (Conflicted/Chop) -> Stand aside unless confirmed Zone Extreme Reversal.
        # =============================================================
        tf_trends = TechnicalIndicators.get_4tf_trends(dfs)
        h1_trend = tf_trends.get("1H", "NEUTRAL")
        m15_trend = tf_trends.get("15m", "NEUTRAL")

        if h1_trend == "BULL" and m15_trend == "BULL" and direction == "SELL":
            logger.debug(f"[{symbol} SELL] BLOCKED: 1H & 15m Macro Trend is unanimously BULLISH. Strictly no selling into an institutional uptrend!")
            return None

        if h1_trend == "BEAR" and m15_trend == "BEAR" and direction == "BUY":
            logger.debug(f"[{symbol} BUY] BLOCKED: 1H & 15m Macro Trend is unanimously BEARISH. Strictly no buying into an institutional downtrend!")
            return None

        htf_conflicted = (h1_trend != "NEUTRAL" and m15_trend != "NEUTRAL" and h1_trend != m15_trend)
        if htf_conflicted and not is_zone_reversal and not is_pre_news_aligned and not is_impulse_active:
            logger.debug(f"[{symbol} {direction}] BLOCKED: 1H ({h1_trend}) and 15m ({m15_trend}) trends are in conflict. Standing aside until key zone extreme is reached.")
            return None

        # =============================================================
        # 2. PROVEN SYSTEMIC HARD FILTERS: Eliminate False Trend Knife-Catching
        # =============================================================

        # A. Battle-Tested 5m SuperTrend Directional Gate
        st_5m = TechnicalIndicators.calculate_supertrend(
            df_5m["high"], df_5m["low"], df_5m["close"], period=10, multiplier=2.5
        )
        if st_5m.get("valid"):
            if direction == "BUY" and st_5m.get("is_bear") and not is_demand_reversal:
                logger.debug(f"[{symbol} BUY] Blocked: 5m SuperTrend is BEAR ({st_5m.get('line')}).")
                return None
            if direction == "SELL" and st_5m.get("is_bull") and not is_supply_reversal:
                logger.debug(f"[{symbol} SELL] Blocked: 5m SuperTrend is BULL ({st_5m.get('line')}).")
                return None

        # B. 15m SuperTrend Macro Trend Gate
        st_15m = {"valid": False, "is_bull": False, "is_bear": False}
        if df_15m is not None and len(df_15m) >= 20:
            st_15m = TechnicalIndicators.calculate_supertrend(
                df_15m["high"], df_15m["low"], df_15m["close"], period=10, multiplier=2.5
            )
            if st_15m.get("valid") and not is_impulse_active and not is_zone_reversal:
                if direction == "BUY" and st_15m.get("is_bear"):
                    logger.debug(f"[{symbol} BUY] Blocked: 15m SuperTrend is BEAR ({st_15m.get('line')}).")
                    return None
                if direction == "SELL" and st_15m.get("is_bull"):
                    logger.debug(f"[{symbol} SELL] Blocked: 15m SuperTrend is BULL ({st_15m.get('line')}).")
                    return None

        # C. Stochastic RSI Momentum Exhaustion Gate (Never buy overbought, never sell oversold!)
        stoch_5m = TechnicalIndicators.calculate_stoch_rsi(df_5m["close"])
        if stoch_5m.get("valid") and not is_impulse_active and not is_pre_news_aligned:
            if direction == "BUY" and stoch_5m.get("is_overbought"):
                logger.debug(f"[{symbol} BUY] Blocked: 5m StochRSI Overbought (K={stoch_5m.get('k'):.1f} > 80). Waiting for pullback to value.")
                return None
            if direction == "SELL" and stoch_5m.get("is_oversold"):
                logger.debug(f"[{symbol} SELL] Blocked: 5m StochRSI Oversold (K={stoch_5m.get('k'):.1f} < 20). Waiting for bounce to resistance.")
                return None

        # D. Statistical Z-Score Exhaustion Gate (Never buy at extreme standard deviation tops or sell bottoms)
        zscore_data = TechnicalIndicators.calculate_zscore(df_5m["close"].values, window=20)
        if zscore_data.get("valid") and not is_impulse_active and not is_pre_news_aligned:
            if direction == "BUY" and zscore_data.get("is_extreme_overbought"):
                logger.debug(f"[{symbol} BUY] Blocked: Statistical Z-Score Overbought (+{zscore_data['zscore']:.1f}σ). Never buy the top!")
                return None
            if direction == "SELL" and zscore_data.get("is_extreme_oversold"):
                logger.debug(f"[{symbol} SELL] Blocked: Statistical Z-Score Oversold ({zscore_data['zscore']:.1f}σ). Never sell the bottom!")
                return None

        # E. 5m RSI Extended Filter (Anti-FOMO)
        rsi_5m_arr = TechnicalIndicators.rsi(df_5m["close"].values, 14)
        rsi_5m_val = float(rsi_5m_arr[-1]) if len(rsi_5m_arr) > 0 and not np.isnan(rsi_5m_arr[-1]) else 50.0
        if not is_impulse_active and not is_pre_news_aligned:
            if direction == "BUY" and rsi_5m_val > 68.0:
                logger.debug(f"[{symbol} BUY] Blocked: 5m RSI Overextended ({rsi_5m_val:.1f} > 68). Wait for pullback to value.")
                return None
            if direction == "SELL" and rsi_5m_val < 32.0:
                logger.debug(f"[{symbol} SELL] Blocked: 5m RSI Overextended ({rsi_5m_val:.1f} < 32). Wait for bounce to resistance.")
                return None

        # F. Immediate 1m Execution Candle Gate
        # Prevent selling into active bullish expansion candles or buying into red dumping candles
        if direction == "BUY":
            if not is_green_1m and not is_hammer_1m and not is_impulse_active:
                logger.debug(f"[{symbol} BUY] Blocked: 1m candle is red without hammer rejection.")
                return None
        else: # SELL
            if not is_red_1m and not is_star_1m and not is_impulse_active:
                logger.debug(f"[{symbol} SELL] Blocked: 1m candle is green without shooting star rejection.")
                return None

        # G. Institutional Liquidity Trap & False Breakout Gate
        # Strictly prevent buying the top of a bull trap or selling the bottom of a bear trap
        if traps.get("valid"):
            if direction == "BUY" and traps.get("is_bull_trap"):
                logger.debug(f"[{symbol} BUY] Blocked: Active Bull Trap / Liquidity Sweep top ({traps.get('trap_reason')}).")
                return None
            if direction == "SELL" and traps.get("is_bear_trap"):
                logger.debug(f"[{symbol} SELL] Blocked: Active Bear Trap / Liquidity Sweep bottom ({traps.get('trap_reason')}).")
                return None

        breakdown = {
            "h1_macro": 0,
            "m15_wave": 0,
            "m5_setup": 0,
            "m1_trigger": 0,
            "smc_institutional": 0,
            "quant_alpha": 0,
            "market_structure": 0,
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
            # 1H Structural Wall Check distance: 1.8 * ATR_15m
            atr_15m_ref = 2.50
            if df_15m is not None and len(df_15m) >= 15:
                calc_atr = TechnicalIndicators.atr(df_15m["high"].values, df_15m["low"].values, df_15m["close"].values, 14)[-1]
                if not np.isnan(calc_atr) and calc_atr > 0:
                    atr_15m_ref = float(calc_atr)
            est_target_dist = 1.8 * atr_15m_ref

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

        # Setup Type B: Consolidation Box Breakout (Only in open room, never breaking into boundary walls!)
        if breakdown["m5_setup"] < 30 and len(df_5m) >= 12:
            recent_high = np.max(h_5m[-11:-1])
            recent_low = np.min(l_5m[-11:-1])

            dt = datetime.fromtimestamp(utc_timestamp, tz=timezone.utc)
            hour = dt.hour + (dt.minute / 60.0)
            in_volume_window = (7.0 <= hour <= 11.5) or (12.5 <= hour <= 17.0)

            if direction == "BUY" and c_5m[-1] >= recent_high and not zones.get("in_supply_zone", False):
                pts = 30 if in_volume_window else 22
                breakdown["m5_setup"] = max(breakdown["m5_setup"], pts)
                reasons.append(f"5m Box Breakout (+{pts})")
            elif direction == "SELL" and c_5m[-1] <= recent_low and not zones.get("in_demand_zone", False):
                pts = 30 if in_volume_window else 22
                breakdown["m5_setup"] = max(breakdown["m5_setup"], pts)
                reasons.append(f"5m Box Breakout (+{pts})")

        # Setup Type D: Institutional Demand / Supply Zone Reversal & Liquidity Sweep
        if is_demand_reversal:
            pts = 35 if (zones.get("liquidity_sweep_bull") or traps.get("is_bear_trap")) else 28
            breakdown["m5_setup"] = max(breakdown["m5_setup"], pts)
            trap_extra = f" / Bear Trap Sweep ({traps.get('trap_reason')})" if traps.get("is_bear_trap") else ""
            reasons.append(f"Institutional Demand Zone Bounce / Liquidity Sweep ({zones['demand_floor']:.2f}){trap_extra} (+{pts})")
        elif is_supply_reversal:
            pts = 35 if (zones.get("liquidity_sweep_bear") or traps.get("is_bull_trap")) else 28
            breakdown["m5_setup"] = max(breakdown["m5_setup"], pts)
            trap_extra = f" / Bull Trap Sweep ({traps.get('trap_reason')})" if traps.get("is_bull_trap") else ""
            reasons.append(f"Institutional Supply Zone Reject / Liquidity Sweep ({zones['supply_ceiling']:.2f}){trap_extra} (+{pts})")

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

            # 5. SuperTrend & StochRSI Quantitative Edge
            if st_5m.get("is_bull"):
                quant_pts += 5
                reasons.append("5m SuperTrend Bullish (+5)")
            if stoch_5m.get("bullish_cross"):
                quant_pts += 4
                reasons.append(f"5m StochRSI Bullish Cross ({stoch_5m.get('k', 50):.1f}/{stoch_5m.get('d', 50):.1f}) (+4)")

            # 6. Dreiss Choppiness Index (Freqtrade standard)
            ci_5m = TechnicalIndicators.calculate_choppiness_index(df_5m["high"], df_5m["low"], df_5m["close"], period=14)
            if ci_5m.get("valid"):
                if ci_5m.get("is_trending"):
                    quant_pts += 6
                    reasons.append(f"Choppiness Index Trending (CI {ci_5m['ci']:.1f} < 40) (+6)")
                elif ci_5m.get("is_choppy"):
                    if not is_zone_reversal:
                        quant_pts -= 6
                        reasons.append(f"Choppiness Index Chop Warning (CI {ci_5m['ci']:.1f} >= 60) (-6 Penalty)")
                    else:
                        reasons.append(f"Choppiness Index Range Box (CI {ci_5m['ci']:.1f}) (Neutral)")

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

            # 5. SuperTrend & StochRSI Quantitative Edge
            if st_5m.get("is_bear"):
                quant_pts += 5
                reasons.append("5m SuperTrend Bearish (+5)")
            if stoch_5m.get("bearish_cross"):
                quant_pts += 4
                reasons.append(f"5m StochRSI Bearish Cross ({stoch_5m.get('k', 50):.1f}/{stoch_5m.get('d', 50):.1f}) (+4)")

            # 6. Dreiss Choppiness Index (Freqtrade standard)
            ci_5m = TechnicalIndicators.calculate_choppiness_index(df_5m["high"], df_5m["low"], df_5m["close"], period=14)
            if ci_5m.get("valid"):
                if ci_5m.get("is_trending"):
                    quant_pts += 6
                    reasons.append(f"Choppiness Index Trending (CI {ci_5m['ci']:.1f} < 40) (+6)")
                elif ci_5m.get("is_choppy"):
                    if not is_zone_reversal:
                        quant_pts -= 6
                        reasons.append(f"Choppiness Index Chop Warning (CI {ci_5m['ci']:.1f} >= 60) (-6 Penalty)")
                    else:
                        reasons.append(f"Choppiness Index Range Box (CI {ci_5m['ci']:.1f}) (Neutral)")

        breakdown["quant_alpha"] = min(20, max(-15, quant_pts))

        # -------------------------------------------------------------
        # 4d. Market Structure (HH/HL/LH/LL, BOS, CHoCH)
        # -------------------------------------------------------------
        struct_5m = TechnicalIndicators.detect_market_structure(df_5m["high"], df_5m["low"], df_5m["close"], window=3)
        ms_pts = 0
        if struct_5m.get("valid"):
            if direction == "BUY":
                if struct_5m.get("is_bullish"):
                    ms_pts += 8
                    reasons.append("5m Bullish Market Structure HH/HL (+8)")
                if struct_5m.get("bos_bull"):
                    ms_pts += 6
                    reasons.append("5m Bullish BOS Continuation (+6)")
                if struct_5m.get("choch_bull"):
                    ms_pts += 8
                    reasons.append("5m Bullish CHoCH Reversal (+8)")
                if struct_5m.get("is_bearish") and not is_zone_reversal:
                    ms_pts -= 8
                    reasons.append("5m Opposing Bearish Structure LH/LL (-8)")
            else: # SELL
                if struct_5m.get("is_bearish"):
                    ms_pts += 8
                    reasons.append("5m Bearish Market Structure LH/LL (+8)")
                if struct_5m.get("bos_bear"):
                    ms_pts += 6
                    reasons.append("5m Bearish BOS Continuation (+6)")
                if struct_5m.get("choch_bear"):
                    ms_pts += 8
                    reasons.append("5m Bearish CHoCH Reversal (+8)")
                if struct_5m.get("is_bullish") and not is_zone_reversal:
                    ms_pts -= 8
                    reasons.append("5m Opposing Bullish Structure HH/HL (-8)")

        breakdown["market_structure"] = min(20, max(-12, ms_pts))

        # Fundamental News Study Integration
        news_bias = self.auto_brain_state.get("news_bias")
        if news_bias and news_bias != "NEUTRAL":
            if news_bias == direction:
                breakdown["news_bias"] = 15
                reasons.append("Fundamental News Study Alignment (+15)")
            else:
                breakdown["news_bias"] = -20
                reasons.append("Opposing Fundamental News Bias (-20)")

        # Smart Pre-News Trajectory Integration (Zone A)
        sub_regime_active = self.auto_brain_state.get("sub_regime")
        traj_bias_active = self.auto_brain_state.get("trajectory_bias")
        is_pre_news_active = self.auto_brain_state.get("is_pre_news_trajectory", False)
        if sub_regime_active == "PRE_NEWS_TRAJECTORY" and traj_bias_active in ("BUY", "SELL"):
            if traj_bias_active == direction:
                breakdown["pre_news_trajectory"] = 12
                reasons.append("🎯 Pre-News Trajectory Alignment Boost (+12)")
            else:
                breakdown["pre_news_trajectory"] = -25
                reasons.append("Counter Pre-News Trajectory Penalty (-25)")

        # -------------------------------------------------------------
        # Institutional Multi-Timeframe Trend Confluence Filter
        # -------------------------------------------------------------
        # 1. HARD COUNTER-TREND GATE: Never take a trade opposing BOTH 1H and 15m trends UNLESS it's a confirmed Zone Reversal!
        if not h1_ema50_aligned and not m15_trend_aligned:
            if is_zone_reversal:
                breakdown["htf_divergence"] = -8
                reasons.append("Institutional Zone Reversal (Bypassed HTF Divergence) (-8)")
            else:
                logger.debug(f"[{symbol} {direction}] Filtered: Hard counter-trend opposing both 1H and 15m HTF.")
                return None

        # 2. HTF Divergence Penalty: If 1H or 15m is not aligned, apply heavy penalty
        htf_diverged = (not h1_ema50_aligned) or (not m15_trend_aligned)
        if htf_diverged:
            if is_zone_reversal:
                breakdown["htf_divergence"] = -5
                reasons.append("Zone Reversal Divergence Offset (-5)")
            else:
                breakdown["htf_divergence"] = -25
                reasons.append("Higher-Timeframe Trend Divergence (-25 Penalty)")

        # Total Confluence Score
        total_score = sum(breakdown.values())
        if htf_diverged and not is_zone_reversal:
            # Strictly cap total score at 68 when HTF is not unanimous (NO SNIPER ALLOWED)
            total_score = min(total_score, 68)
        total_score = min(100, max(0, total_score))

        # Determine Tier: TIER_1_SNIPER requires unanimous HTF alignment OR Confirmed Zone Reversal!
        if total_score >= 82 and (not htf_diverged or is_zone_reversal):
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
        if is_demand_reversal and zones.get("valid"):
            # Set SL safely below the Demand Floor
            raw_sl_dist = max(current_price - (zones["demand_floor"] - (0.50 if is_gold else 0.0005)), 2.50 if is_gold else 0.0015)
        elif is_supply_reversal and zones.get("valid"):
            # Set SL safely above the Supply Ceiling
            raw_sl_dist = max((zones["supply_ceiling"] + (0.50 if is_gold else 0.0005)) - current_price, 2.50 if is_gold else 0.0015)
        elif df_5m is not None and len(df_5m) >= 5:
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

        is_small_acc = getattr(self, "account_balance", None) is not None and self.account_balance <= 35.0
        min_sl = 2.50 if (is_gold and is_small_acc) else (2.80 if is_gold else 0.0015)
        max_sl = 3.80 if (is_gold and is_small_acc) else (4.50 if is_gold else 0.0035)
        risk_r_dist = float(np.clip(raw_sl_dist, min_sl, max_sl))

        # Sub-regime and trade parameters
        if self.mode == "auto_adaptive":
            sub_regime = self.auto_brain_state.get("sub_regime", "WAVE_RUNNER")
            target_rr = float(self.auto_brain_state.get("target_rr", 2.6))
            be_lock_r = float(self.auto_brain_state.get("be_lock_r", 0.65))
            stall_exit_seconds = float(self.auto_brain_state.get("stall_exit_seconds", 90.0))
        elif self.mode in ["hyper", "aggressive", "scalp", "scalping", "fast_scalp"]:
            sub_regime = "FAST_SCALP"
            target_rr = 1.8 if self.mode in ["hyper", "scalp", "scalping", "fast_scalp"] else 2.2
            be_lock_r = 0.50 if self.mode in ["hyper", "scalp", "scalping", "fast_scalp"] else 0.65
            stall_exit_seconds = 45.0 if self.mode in ["hyper", "scalp", "scalping", "fast_scalp"] else 60.0
        elif self.mode == "cautious":
            sub_regime = "WAVE_RUNNER"
            target_rr = 2.8
            be_lock_r = 0.65
            stall_exit_seconds = 120.0
        elif self.mode == "manual":
            if self.aggressiveness_pct >= 70.0:
                sub_regime = "FAST_SCALP"
                target_rr = 1.8 if self.aggressiveness_pct >= 85.0 else 2.0
                be_lock_r = 0.50 if self.aggressiveness_pct >= 85.0 else 0.60
                stall_exit_seconds = 45.0 if self.aggressiveness_pct >= 85.0 else 60.0
            else:
                sub_regime = "WAVE_RUNNER"
                target_rr = 2.5
                be_lock_r = 0.65
                stall_exit_seconds = 90.0
        else: # Standard
            sub_regime = "WAVE_RUNNER" if total_score >= 68 else "FAST_SCALP"
            target_rr = 2.4
            be_lock_r = 0.65
            stall_exit_seconds = 90.0

        if sub_regime == "FAST_SCALP":
            regime = "ADAPTIVE_SCALP"
        elif sub_regime == "DEFENSIVE_SHIELD":
            regime = "ADAPTIVE_RUNNER" if total_score >= 82 else "ADAPTIVE_SCALP"
        else:
            regime = "ADAPTIVE_RUNNER"

        # -------------------------------------------------------------
        # 5b. Always Execute Confluence Signals at MARKET (No Waiting on Limits)
        # -------------------------------------------------------------
        order_type = "MARKET"
        limit_price = None
        invalidation_price = None


        # Compute suggested SL and TP
        entry_p = limit_price if limit_price else current_price
        if direction == "BUY":
            suggested_sl = round(entry_p - risk_r_dist, 2 if is_gold else 5)
            calc_tp = entry_p + (risk_r_dist * target_rr)
            if is_demand_reversal and zones.get("valid") and zones["mid_range"] > entry_p:
                calc_tp = max(calc_tp, zones["mid_range"])
            suggested_tp = round(calc_tp, 2 if is_gold else 5)
        else:
            suggested_sl = round(entry_p + risk_r_dist, 2 if is_gold else 5)
            calc_tp = entry_p - (risk_r_dist * target_rr)
            if is_supply_reversal and zones.get("valid") and zones["mid_range"] < entry_p:
                calc_tp = min(calc_tp, zones["mid_range"])
            suggested_tp = round(calc_tp, 2 if is_gold else 5)

        prefix = "🎯 PRE_NEWS_TRAJECTORY " if is_pre_news_active else ""
        full_reason = f"{prefix}{tier} ({total_score}/100) -> [{sub_regime}] {regime} | " + ", ".join(reasons)

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
            is_pre_news_trajectory=is_pre_news_active,
            pre_news_trajectory_confidence=float(self.auto_brain_state.get("news_confidence", 0.0)) if is_pre_news_active else 0.0,
        )

    def calculate_planned_limits(
        self,
        symbol: str,
        dfs: Dict[str, pd.DataFrame],
        current_tick_bid: float,
        current_tick_ask: float,
        now_ts: Optional[float] = None,
        account_balance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Institutional Planned Limits Engine.
        Continuously calculates resting BUY_LIMIT and SELL_LIMIT setups at institutional
        Supply/Demand zones and SuperTrend structural support/resistance.
        Provides full mathematical transparency (entry, SL, TP, R:R, score, invalidation, and reasons)
        so the bot displays exact planned calculations and never 'just sits there' during chop.
        """
        now = now_ts or time.time()
        is_gold = any(k in symbol.upper() for k in ["XAU", "GOLD"])
        mid_price = round((current_tick_bid + current_tick_ask) / 2.0, 2 if is_gold else 5)

        df_1h = dfs.get("1H")
        df_15m = dfs.get("15m")
        df_5m = dfs.get("5m")
        df_1m = dfs.get("1m")

        if df_5m is None or len(df_5m) < 15 or df_1m is None or len(df_1m) < 5:
            return {
                "symbol": symbol,
                "mid_price": mid_price,
                "planned_buy": None,
                "planned_sell": None,
                "recalculated_at": now,
            }

        # 1. Structural Zones & Indicators
        zones = TechnicalIndicators.detect_supply_demand_zones(df_5m, df_15m, mid_price, is_gold)
        traps = TechnicalIndicators.detect_liquidity_traps(df_5m, mid_price, is_gold)
        st_5m = TechnicalIndicators.calculate_supertrend(df_5m["high"], df_5m["low"], df_5m["close"], 10, 2.5) if len(df_5m) >= 20 else {}
        st_15m = TechnicalIndicators.calculate_supertrend(df_15m["high"], df_15m["low"], df_15m["close"], 10, 2.5) if df_15m is not None and len(df_15m) >= 20 else {}

        # ATR calculation for risk distance
        atr_series = TechnicalIndicators.atr(df_5m["high"], df_5m["low"], df_5m["close"], 14)
        atr_val = float(atr_series[-1]) if len(atr_series) > 0 and not np.isnan(atr_series[-1]) else (2.20 if is_gold else 0.0020)
        atr_val = max(1.20 if is_gold else 0.0010, min(4.50 if is_gold else 0.0045, atr_val))

        # -------------------------------------------------------------
        # 2. Institutional BUY LIMIT Calculation
        # -------------------------------------------------------------
        cand_buy_level = current_tick_ask - (atr_val * 0.75)
        buy_source = "ATR Pullback Level"
        if st_5m.get("is_bull") and st_5m.get("line", 0) < (current_tick_ask - 0.40):
            cand_buy_level = float(st_5m["line"])
            buy_source = f"5m SuperTrend Support (${cand_buy_level:.2f})"
        elif zones.get("valid") and zones.get("demand_ceiling", 0) < (current_tick_ask - 0.40):
            cand_buy_level = float(zones["demand_ceiling"])
            buy_source = f"Demand Ceiling (${cand_buy_level:.2f})"

        buy_limit_price = min(round(cand_buy_level, 2 if is_gold else 5), round(current_tick_ask - 0.45, 2 if is_gold else 5))
        buy_dist = round(current_tick_ask - buy_limit_price, 2 if is_gold else 5)

        # SL placed below Demand floor or 1.1x ATR
        dem_floor = float(zones.get("demand_floor", buy_limit_price - atr_val)) if zones.get("valid") else (buy_limit_price - atr_val)
        raw_buy_sl = min(buy_limit_price - (atr_val * 1.1), dem_floor - (0.50 if is_gold else 0.0005))
        acc_bal = account_balance if account_balance is not None else getattr(self, "account_balance", None)
        is_small_acc = (acc_bal is not None and acc_bal <= 35.0)
        min_sl_dist = 1.80 if is_gold else 0.0015
        max_sl_dist = 2.60 if (is_gold and is_small_acc) else (3.50 if is_gold else 0.0040)
        buy_sl = round(max(buy_limit_price - max_sl_dist, min(buy_limit_price - min_sl_dist, raw_buy_sl)), 2 if is_gold else 5)
        buy_risk = round(buy_limit_price - buy_sl, 2 if is_gold else 5)

        # TP placed at Mid-Range or 1.8R/2.2R
        target_rr = 2.0
        mid_target = float(zones.get("mid_range", buy_limit_price + (buy_risk * target_rr)))
        cand_tp = max(buy_limit_price + (buy_risk * 1.8), mid_target)
        if st_15m.get("line", 0) > (buy_limit_price + 1.50):
            cand_tp = min(cand_tp, float(st_15m["line"]) - (0.30 if is_gold else 0.0003))
        buy_tp = round(max(buy_limit_price + (buy_risk * 1.5), cand_tp), 2 if is_gold else 5)
        buy_reward = round(buy_tp - buy_limit_price, 2 if is_gold else 5)
        buy_rr = round(buy_reward / buy_risk, 2) if buy_risk > 0 else target_rr

        buy_invalidation = round(buy_sl - (0.50 if is_gold else 0.0005), 2 if is_gold else 5)

        # Confluence / Readiness Score
        b_score = 52
        if st_5m.get("is_bull"):
            b_score += 15
        if zones.get("valid") and abs(buy_limit_price - zones.get("demand_ceiling", 0)) < 2.0:
            b_score += 12
        if buy_rr >= 2.0:
            b_score += 8
        if traps.get("is_bear_trap"):
            b_score += 15
        if st_15m.get("is_bull"):
            b_score += 10
        b_score = min(95, max(30, b_score))

        buy_status = "WAITING_PULLBACK"
        if buy_dist <= 3.50 and b_score >= 60:
            buy_status = "READY_TO_STAGE"

        planned_buy = {
            "symbol": symbol,
            "direction": "BUY",
            "order_type": "BUY_LIMIT",
            "limit_price": float(buy_limit_price),
            "current_price": float(current_tick_ask),
            "distance_pts": float(buy_dist),
            "sl": float(buy_sl),
            "tp": float(buy_tp),
            "risk_pts": float(buy_risk),
            "reward_pts": float(buy_reward),
            "target_rr": float(buy_rr),
            "invalidation_price": float(buy_invalidation),
            "score": int(b_score),
            "source": buy_source,
            "status": buy_status,
            "reason": f"BUY_LIMIT @ {buy_limit_price:.2f} ({buy_source}) | SL: {buy_sl:.2f} (${buy_risk:.2f}) | TP: {buy_tp:.2f} (${buy_reward:.2f}, {buy_rr:.1f}R) | Inval < {buy_invalidation:.2f}",
        }

        # -------------------------------------------------------------
        # 3. Institutional SELL LIMIT Calculation
        # -------------------------------------------------------------
        cand_sell_level = current_tick_bid + (atr_val * 0.75)
        sell_source = "ATR Pullback Level"
        if st_15m.get("is_bear") and st_15m.get("line", 0) > (current_tick_bid + 0.40):
            cand_sell_level = float(st_15m["line"])
            sell_source = f"15m SuperTrend Resistance (${cand_sell_level:.2f})"
        elif zones.get("valid") and zones.get("supply_floor", 0) > (current_tick_bid + 0.40):
            cand_sell_level = float(zones["supply_floor"])
            sell_source = f"Supply Floor (${cand_sell_level:.2f})"

        sell_limit_price = max(round(cand_sell_level, 2 if is_gold else 5), round(current_tick_bid + 0.45, 2 if is_gold else 5))
        sell_dist = round(sell_limit_price - current_tick_bid, 2 if is_gold else 5)

        sup_ceiling = float(zones.get("supply_ceiling", sell_limit_price + atr_val)) if zones.get("valid") else (sell_limit_price + atr_val)
        raw_sell_sl = max(sell_limit_price + (atr_val * 1.1), sup_ceiling + (0.60 if is_gold else 0.0005))
        sell_sl = round(min(sell_limit_price + max_sl_dist, max(sell_limit_price + min_sl_dist, raw_sell_sl)), 2 if is_gold else 5)
        sell_risk = round(sell_sl - sell_limit_price, 2 if is_gold else 5)

        mid_target_sell = float(zones.get("mid_range", sell_limit_price - (sell_risk * target_rr)))
        cand_sell_tp = min(sell_limit_price - (sell_risk * 1.8), mid_target_sell)
        if st_5m.get("line", 0) < (sell_limit_price - 1.50):
            cand_sell_tp = max(cand_sell_tp, float(st_5m["line"]) + (0.30 if is_gold else 0.0003))
        sell_tp = round(min(sell_limit_price - (sell_risk * 1.5), cand_sell_tp), 2 if is_gold else 5)
        sell_reward = round(sell_limit_price - sell_tp, 2 if is_gold else 5)
        sell_rr = round(sell_reward / sell_risk, 2) if sell_risk > 0 else target_rr

        sell_invalidation = round(sell_sl + (0.50 if is_gold else 0.0005), 2 if is_gold else 5)

        s_score = 52
        if st_15m.get("is_bear"):
            s_score += 15
        if zones.get("valid") and abs(sell_limit_price - zones.get("supply_floor", 0)) < 2.0:
            s_score += 12
        if sell_rr >= 2.0:
            s_score += 8
        if traps.get("is_bull_trap"):
            s_score += 15
        if st_5m.get("is_bear"):
            s_score += 10
        s_score = min(95, max(30, s_score))

        sell_status = "WAITING_RETEST"
        if sell_dist <= 3.50 and s_score >= 60:
            sell_status = "READY_TO_STAGE"

        planned_sell = {
            "symbol": symbol,
            "direction": "SELL",
            "order_type": "SELL_LIMIT",
            "limit_price": float(sell_limit_price),
            "current_price": float(current_tick_bid),
            "distance_pts": float(sell_dist),
            "sl": float(sell_sl),
            "tp": float(sell_tp),
            "risk_pts": float(sell_risk),
            "reward_pts": float(sell_reward),
            "target_rr": float(sell_rr),
            "invalidation_price": float(sell_invalidation),
            "score": int(s_score),
            "source": sell_source,
            "status": sell_status,
            "reason": f"SELL_LIMIT @ {sell_limit_price:.2f} ({sell_source}) | SL: {sell_sl:.2f} (${sell_risk:.2f}) | TP: {sell_tp:.2f} (${sell_reward:.2f}, {sell_rr:.1f}R) | Inval > {sell_invalidation:.2f}",
        }

        res = {
            "symbol": symbol,
            "mid_price": float(mid_price),
            "planned_buy": planned_buy,
            "planned_sell": planned_sell,
            "zones": {
                "demand_floor": float(zones.get("demand_floor", 0.0)) if zones.get("demand_floor") is not None else None,
                "demand_ceiling": float(zones.get("demand_ceiling", 0.0)) if zones.get("demand_ceiling") is not None else None,
                "supply_floor": float(zones.get("supply_floor", 0.0)) if zones.get("supply_floor") is not None else None,
                "supply_ceiling": float(zones.get("supply_ceiling", 0.0)) if zones.get("supply_ceiling") is not None else None,
                "mid_range": float(zones.get("mid_range", 0.0)) if zones.get("mid_range") is not None else None,
            },
            "recalculated_at": float(now),
        }
        self.latest_planned_limits = res
        return res

    def create_signal_from_planned_limit(self, planned_limit: Dict[str, Any]) -> ConfluenceSignal:
        """Construct full ConfluenceSignal from a planned limit dictionary."""
        direction = planned_limit["direction"]
        order_type = planned_limit["order_type"]
        limit_price = planned_limit["limit_price"]
        sl = planned_limit["sl"]
        tp = planned_limit["tp"]
        risk_dist = planned_limit["risk_pts"]
        rr = planned_limit["target_rr"]
        score = planned_limit["score"]
        inval = planned_limit["invalidation_price"]
        reason = planned_limit["reason"]

        tier = "TIER_2_BALANCED" if score >= 68 else "TIER_3_AGGRESSIVE"
        return ConfluenceSignal(
            symbol=planned_limit.get("symbol", "GOLD"),
            direction=direction,
            score=score,
            tier=tier,
            regime="ADAPTIVE_LIMIT_STAGING",
            sub_regime="ZONE_BOUND_SETUP",
            target_rr=rr,
            entry_price=limit_price,
            suggested_sl=sl,
            suggested_tp=tp,
            risk_r_dist=risk_dist,
            be_lock_r=0.60,
            stall_exit_seconds=90.0,
            breakdown={"limit_setup": score},
            reason=f"🎯 PLANNED {order_type} | {reason}",
            time=time.time(),
            order_type=order_type,
            limit_price=limit_price,
            invalidation_price=inval,
        )
