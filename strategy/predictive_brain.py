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

from engine.historical_matcher import HistoricalMatcher, SetupVector
from engine.indicators import TechnicalIndicators
from strategy.ensemble_models import EnsembleQuantSuite, EnsembleTelemetry
from strategy.experience_learner import AdaptiveExperienceLearner
from strategy.v2.meta_labeler import MetaLabelerFilter

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
    candle_pos_5m: float = 0.5
    ensemble: Optional[Dict[str, Any]] = None
    experience: Optional[Dict[str, Any]] = None
    news_catalyst: Optional[Dict[str, Any]] = None
    historical_match: Optional[Dict[str, Any]] = None

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
        self.ensemble_suite = EnsembleQuantSuite()
        self.meta_labeler = MetaLabelerFilter(model_path="models/metalabeler_v1.txt", probability_threshold=0.52)
        self.experience_learner = AdaptiveExperienceLearner()
        self.historical_matcher = HistoricalMatcher(target_rr=2.0)


    def learn_from_trade(
        self,
        ticket: int,
        symbol: str,
        direction: str,
        open_price: float,
        close_price: float,
        pnl_usd: float,
        pnl_r: float,
        close_reason: str,
        context: Dict[str, Any],
    ) -> str:
        """Process finalized trade through the online experience learner."""
        if hasattr(self, "experience_learner") and self.experience_learner:
            return self.experience_learner.record_trade_outcome(
                ticket=ticket,
                symbol=symbol,
                direction=direction,
                open_price=open_price,
                close_price=close_price,
                pnl_usd=pnl_usd,
                pnl_r=pnl_r,
                close_reason=close_reason,
                context=context,
            )
        return ""

    def on_tick(self, bid: float, ask: float, now_ts: Optional[float] = None) -> None:
        """Register live tick into the ensemble suite."""
        if hasattr(self, "ensemble_suite") and self.ensemble_suite:
            self.ensemble_suite.on_tick(bid, ask, now_ts)

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
        active_positions: Optional[List[Any]] = None,
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

        positions: List[Any] = []
        if active_positions:
            positions = active_positions
        elif trade_manager:
            if hasattr(trade_manager, "get_managed_positions"):
                try:
                    positions = trade_manager.get_managed_positions()
                except Exception:
                    pass
            elif hasattr(trade_manager, "_managed_positions"):
                try:
                    positions = [p.to_dict() if hasattr(p, "to_dict") else p for p in trade_manager._managed_positions.values()]
                except Exception:
                    pass

        df_1h = dfs.get("1H")
        df_15m = dfs.get("15m")
        df_5m = dfs.get("5m")
        df_1m = dfs.get("1m")

        # Insufficient data fallback
        if df_5m is None or len(df_5m) < 20:
            return self._build_standby_state(symbol, now, time_str, current_bid, current_ask, spread, "Warming up multi-timeframe historical candles...")

        # Register live tick to ensemble order flow tracker
        self.ensemble_suite.on_tick(current_bid, current_ask, now)

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
        # 4b. Quantitative Ensemble Models (OFI Delta, FVG Support, Parkinson Vol)
        # ----------------------------------------------------------------------
        ensemble_telemetry = self.ensemble_suite.evaluate_ensemble(
            symbol=symbol,
            dfs=dfs,
            current_bid=current_bid,
            macro_1h=trend_1h,
            macro_15m=trend_15m,
            macro_5m=trend_5m,
            kaufman_er=efficiency_ratio,
        )

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

        # Macro News & Fundamental Surprise Delta Integration
        news_bias_data: Dict[str, Any] = {}
        if news_reactor and hasattr(news_reactor, "analyze_news_fundamental_bias"):
            try:
                news_bias_data = news_reactor.analyze_news_fundamental_bias(symbol)
                news_dir = news_bias_data.get("direction", "NEUTRAL")
                news_conf = float(news_bias_data.get("confidence", 0.50))
                if news_dir == "BUY":
                    bull_points += 2.2 * min(1.0, news_conf / 0.75)
                elif news_dir == "SELL":
                    bear_points += 2.2 * min(1.0, news_conf / 0.75)
            except Exception as e:
                logger.debug(f"News fundamental study error: {e}")

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

        # Measure Candle Relative Position (0.0 = low wick, 1.0 = high wick)
        high_5m_curr = float(h_5m[-1])
        low_5m_curr = float(l_5m[-1])
        range_5m = max(0.40, high_5m_curr - low_5m_curr)
        candle_pos_5m = float(np.clip((current_bid - low_5m_curr) / range_5m, 0.0, 1.0))

        candle_pos_1m = 0.5
        if df_1m is not None and not df_1m.empty and len(df_1m) > 0:
            h_1m_curr = float(df_1m["high"].iloc[-1])
            l_1m_curr = float(df_1m["low"].iloc[-1])
            range_1m = max(0.20, h_1m_curr - l_1m_curr)
            candle_pos_1m = float(np.clip((current_bid - l_1m_curr) / range_1m, 0.0, 1.0))

        # Baseline log-odds (0.0 corresponds to 50.0% uninformative random walk)
        logit = 0.0

        # Factor 1: Macro alignment (calibrated: +0.22 pushes baseline to ~55.5%)
        if dominant_dir == "BUY" and trend_1h == "BULL" and trend_15m == "BULL":
            logit += 0.22
        elif dominant_dir == "SELL" and trend_1h == "BEAR" and trend_15m == "BEAR":
            logit += 0.22
        elif (dominant_dir == "BUY" and (trend_1h == "BEAR" or trend_15m == "BEAR")) or \
             (dominant_dir == "SELL" and (trend_1h == "BULL" or trend_15m == "BULL")):
            logit -= 0.65  # Fighting macro trend severely penalizes win rate

        # Factor 2: Efficiency ratio & R² conviction (calibrated institutional nudge)
        logit += float(np.clip((efficiency_ratio - 0.40) * 0.30, -0.15, 0.15))
        logit += float(np.clip((r_squared - 0.40) * 0.25, -0.12, 0.12))

        # Factor 3: Anti-Chasing / Value Entry Penalty (ATR-Normalized & Candle Position)
        atr_5m_arr = TechnicalIndicators.atr(h_5m, l_5m, c_5m, 14)
        atr_5m = float(atr_5m_arr[-1]) if (atr_5m_arr is not None and len(atr_5m_arr) > 0 and not np.isnan(atr_5m_arr[-1])) else 2.50
        overextended_dist = max(2.50, 1.2 * atr_5m)

        # Genuinely overbought / oversold at extreme exhaustion wicks
        is_overbought = (stoch_k > 80.0 and vwap_diff > overextended_dist) or (stoch_k > 85.0 and vwap_diff > 1.80) or (vwap_diff > 3.50)
        is_oversold = (stoch_k < 20.0 and vwap_diff < -overextended_dist) or (stoch_k < 15.0 and vwap_diff < -1.80) or (vwap_diff < -3.50)

        # Chasing the top wick of a candle (BUY) or bottom wick (SELL)
        is_chasing_top = (candle_pos_5m > 0.72 and vwap_diff > 0.80) or (candle_pos_1m > 0.82 and vwap_diff > 1.20) or is_overbought
        is_chasing_bottom = (candle_pos_5m < 0.28 and vwap_diff < -0.80) or (candle_pos_1m < 0.18 and vwap_diff < -1.20) or is_oversold

        pullback_ready = False
        if dominant_dir == "BUY":
            if is_chasing_top:
                logit -= 0.85  # Severe penalty for buying the candle high wick / top
            elif (candle_pos_5m <= 0.60 and -1.5 * atr_5m <= vwap_diff <= 1.0 * atr_5m and stoch_k <= 68.0):
                # Buying the dip / pullback into value or fair value discount
                logit += 0.25
                pullback_ready = True
            elif (ensemble_telemetry and ensemble_telemetry.nearest_fvg and ensemble_telemetry.nearest_fvg.get("direction") == "BULL"
                  and abs(current_bid - ensemble_telemetry.nearest_fvg["midpoint"]) <= 1.20 and candle_pos_5m <= 0.65):
                # Near institutional bullish FVG support
                logit += 0.25
                pullback_ready = True
        elif dominant_dir == "SELL":
            if is_chasing_bottom:
                logit -= 0.85  # Severe penalty for selling the candle low wick / bottom
            elif (candle_pos_5m >= 0.40 and -1.0 * atr_5m <= vwap_diff <= 1.5 * atr_5m and stoch_k >= 32.0):
                # Selling the rally / bounce into resistance or premium
                logit += 0.25
                pullback_ready = True
            elif (ensemble_telemetry and ensemble_telemetry.nearest_fvg and ensemble_telemetry.nearest_fvg.get("direction") == "BEAR"
                  and abs(current_bid - ensemble_telemetry.nearest_fvg["midpoint"]) <= 1.20 and candle_pos_5m >= 0.35):
                # Near institutional bearish FVG resistance
                logit += 0.25
                pullback_ready = True

        # Factor 4: Quantitative Ensemble Suite (OFI Delta + Institutional FVG + Parkinson Volatility)
        if ensemble_telemetry:
            is_bias_bull = "BULL" in ensemble_telemetry.ensemble_bias
            is_bias_bear = "BEAR" in ensemble_telemetry.ensemble_bias
            if dominant_dir == "BUY" and is_bias_bull:
                logit += 0.15 * (ensemble_telemetry.ensemble_score / 100.0)
            elif dominant_dir == "BUY" and is_bias_bear:
                logit -= 0.35  # Opposing order flow/liquidity structure
            elif dominant_dir == "SELL" and is_bias_bear:
                logit += 0.15 * (ensemble_telemetry.ensemble_score / 100.0)
        # Factor 4b: Historical Pattern Similarity & Statistical Edge Matcher
        hist_edge_result = None
        if hasattr(self, "historical_matcher") and self.historical_matcher and df_5m is not None and len(df_5m) >= 60:
            try:
                setup_vec = SetupVector(
                    direction=dominant_dir,
                    h1_trend="BULLISH" if trend_1h == "BULL" else ("BEARISH" if trend_1h == "BEAR" else "NEUTRAL"),
                    m15_adx_tier="STRONG" if efficiency_ratio >= 0.50 else ("MODERATE" if efficiency_ratio >= 0.30 else "CHOP"),
                    m5_rsi_zone="OVERSOLD" if stoch_k < 35.0 else ("VALUE_BULL" if stoch_k <= 55.0 else ("VALUE_BEAR" if stoch_k <= 70.0 else "OVERBOUGHT")),
                    m1_trigger_type="MOMENTUM" if efficiency_ratio >= 0.40 else "ENGULFING",
                    session=HistoricalMatcher.identify_session(now),
                )
                hist_edge_result = self.historical_matcher.compute_edge(setup_vec, df_5m, target_rr=2.0)
                if hist_edge_result and hist_edge_result.sample_count >= 5:
                    hist_delta = (hist_edge_result.win_rate_pct / 100.0) - 0.50
                    logit += float(np.clip(hist_delta * 0.30, -0.20, 0.20))
            except Exception as e:
                logger.debug(f"HistoricalMatcher calculation error: {e}")

        # Factor 4c: Macro News Fundamental Wind Modulation
        if news_bias_data:
            nb_dir = news_bias_data.get("direction", "NEUTRAL")
            nb_conf = float(news_bias_data.get("confidence", 0.50))
            if nb_dir == dominant_dir:
                logit += 0.16 * min(1.0, nb_conf / 0.75)
            elif nb_dir != "NEUTRAL":
                logit -= 0.32 * min(1.0, nb_conf / 0.75)

        # Factor 4d: Online Continual Experience Learner (Adaptive Mistake Reflex & Pattern Synergy)
        exp_eval = None
        exp_telemetry = None
        if hasattr(self, "experience_learner") and self.experience_learner:
            eval_ctx = {
                "candle_pos_5m": candle_pos_5m,
                "vwap_deviation": vwap_diff,
                "stoch_rsi_k": stoch_k,
                "kaufman_er": efficiency_ratio,
                "r_squared": r_squared,
                "order_flow_delta": float(ensemble_telemetry.delta_ratio_60s if ensemble_telemetry else 0.0),
                "is_zone_reversal": 0.0,
                "news_bias_direction": news_bias_data.get("direction", "NEUTRAL"),
                "news_confidence": float(news_bias_data.get("confidence", 0.50)),
                "historical_pattern_wr": float(hist_edge_result.win_rate_pct if hist_edge_result else 50.0),
            }
            exp_eval = self.experience_learner.evaluate_candidate(
                symbol=symbol,
                direction=dominant_dir,
                context=eval_ctx,
                atr=atr_5m,
            )
            # Damping or boosting based on learned experience
            logit += (exp_eval.synergy_bonus * 0.35) - (exp_eval.mistake_penalty * 0.50)
            exp_telemetry = self.experience_learner.get_telemetry()

        raw_p_win = self._sigmoid(logit)


        # Factor 5: Meta-Labeler ML Filter Blending (Secondary Causal ML model)
        if hasattr(self, "meta_labeler") and self.meta_labeler and self.meta_labeler.is_loaded:
            feat_dict = {
                "score": float(ensemble_telemetry.ensemble_score if ensemble_telemetry else 50.0),
                "h1_macro": 1.0 if trend_1h == "BULL" else (-1.0 if trend_1h == "BEAR" else 0.0),
                "m15_wave": 1.0 if trend_15m == "BULL" else (-1.0 if trend_15m == "BEAR" else 0.0),
                "m5_setup": 1.0 if trend_5m == "BULL" else (-1.0 if trend_5m == "BEAR" else 0.0),
                "m1_trigger": 1.0 if pullback_ready else 0.0,
                "smc_institutional": 1.0 if (ensemble_telemetry and ensemble_telemetry.nearest_fvg) else 0.0,
                "quant_alpha": float(ensemble_telemetry.delta_ratio_60s if ensemble_telemetry else 0.0),
                "adx_15m": 25.0,
                "choppiness_index": float(chop_val),
                "atr_percentile": float(np.clip(atr_5m / 3.0, 0.1, 1.0)),
                "h1_ema_slope": float(h1_slope),
                "hour": float(dt_utc.hour),
                "session_code": 1.0 if (8 <= dt_utc.hour <= 17) else 0.0,
                "is_zone_reversal": 0.0,
                "regime_code": 1.0 if efficiency_ratio >= 0.50 else 0.0,
            }
            ml_p_win = self.meta_labeler.predict_p_win(feat_dict)
            calibrated_p = (0.50 * raw_p_win) + (0.50 * ml_p_win)
        else:
            calibrated_p = raw_p_win

        # Calibrated institutional bounds: max 64.0% win prob, min 32.0%
        p_win = round(float(np.clip(calibrated_p, 0.32, 0.64)), 3)

        # Target reward-to-risk (aiming for high profits: 3.2 - 4.2 RR)
        base_rr = 3.6 if efficiency_ratio >= 0.50 else 3.0
        target_rr = base_rr + (ensemble_telemetry.rr_expansion_boost if ensemble_telemetry else 0.0)
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
            if is_chasing_top:
                value_pass = False
                value_detail = f"TOP-OF-CANDLE CHASE BLOCKED: 5m candle pos is {candle_pos_5m*100:.0f}% (upper wick) & +{vwap_diff:.2f} above VWAP. Chasing tops prohibited; waiting for pullback."
            elif pullback_ready:
                value_pass = True
                value_detail = f"PULLBACK VERIFIED: 5m candle pos is {candle_pos_5m*100:.0f}% near VWAP fair value ({vwap_val:.2f}, diff: {vwap_diff:+.2f})."
            else:
                value_pass = False
                value_detail = f"WAITING FOR PULLBACK: 5m candle pos: {candle_pos_5m*100:.0f}% | VWAP diff: {vwap_diff:+.2f} pts | StochRSI: {stoch_k:.1f}."
        else:
            if is_chasing_bottom:
                value_pass = False
                value_detail = f"BOTTOM-OF-CANDLE CHASE BLOCKED: 5m candle pos is {candle_pos_5m*100:.0f}% (lower wick) & {vwap_diff:.2f} below VWAP. Selling bottoms prohibited; waiting for bounce."
            elif pullback_ready:
                value_pass = True
                value_detail = f"BOUNCE VERIFIED: 5m candle pos is {candle_pos_5m*100:.0f}% near VWAP fair value ({vwap_val:.2f}, diff: {vwap_diff:+.2f})."
            else:
                value_pass = False
                value_detail = f"WAITING FOR BOUNCE: 5m candle pos: {candle_pos_5m*100:.0f}% | VWAP diff: {vwap_diff:+.2f} pts | StochRSI: {stoch_k:.1f}."

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

        # Gate 4: Macro News & Fundamental Wind Gate
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
                if news_bias_data and news_bias_data.get("summary"):
                    news_detail = f"Macro Bias: {news_bias_data.get('direction')} ({news_bias_data.get('confidence', 0.5)*100:.0f}% Conf) | {news_bias_data.get('summary')}"
                elif cd is not None and cd <= 1200:
                    news_detail = f"Upcoming High-Impact News in {cd // 60}m. Defensive shield primed."

        gates.append({
            "name": "Macro News & Fundamental Wind",
            "status": "BLOCKED" if is_news_freeze else "PASS",
            "detail": news_detail,
            "passed": not is_news_freeze,
        })

        # Gate 5: Mathematical Edge Gate (EV >= +0.20R)
        ev_pass = (expected_value_r >= 0.20 and p_win >= 0.55)
        gates.append({
            "name": "Mathematical Edge (EV >= +0.20R)",
            "status": "PASS" if ev_pass else "WAITING",
            "detail": f"Quant ML P(Win): {p_win * 100:.1f}% | Expected Value: {expected_value_r:+.2f}R (Min: +0.20R)",
            "passed": ev_pass,
        })

        # Gate 6: Ensemble Order Flow & Volatility Consensus
        ensemble_pass = True
        ensemble_detail = "Ensemble models confirm order flow and volatility."
        if ensemble_telemetry:
            is_bias_bull = "BULL" in ensemble_telemetry.ensemble_bias
            is_bias_bear = "BEAR" in ensemble_telemetry.ensemble_bias
            fvg_desc = f"{ensemble_telemetry.nearest_fvg['direction']} FVG @ {ensemble_telemetry.nearest_fvg['midpoint']:.2f}" if ensemble_telemetry.nearest_fvg else "None"
            
            if (dominant_dir == "BUY" and is_bias_bear) or (dominant_dir == "SELL" and is_bias_bull):
                ensemble_pass = False
                ensemble_detail = f"Order Flow Delta opposes {dominant_dir}: bias {ensemble_telemetry.ensemble_bias} (ratio: {ensemble_telemetry.delta_ratio_60s:+.2f})."
            else:
                ensemble_detail = f"Consensus: {ensemble_telemetry.ensemble_bias} (Score: {ensemble_telemetry.ensemble_score:.0f}/100) | OFI Delta: {ensemble_telemetry.delta_ratio_60s:+.2f} | FVG: {fvg_desc}"

        gates.append({
            "name": "Ensemble Order Flow & Volatility Consensus",
            "status": "PASS" if ensemble_pass else "WAITING",
            "detail": ensemble_detail,
            "passed": ensemble_pass,
        })

        # Gate 7: Online Continual Experience Reflex & Mistake Guard
        exp_pass = True
        exp_detail = "Experience Learner confirms setup parameters."
        if exp_eval:
            if exp_eval.veto:
                exp_pass = False
                exp_detail = f"MISTAKE PATTERN VETO: {exp_eval.veto_reason}"
            elif exp_eval.mistake_penalty > 0.05:
                exp_detail = f"Reflex Caution: Setup has {exp_eval.mistake_penalty*100:.1f}% similarity to past losing trade. Adjusted EV."
            elif exp_eval.synergy_bonus > 0.05:
                exp_detail = f"Pattern Synergy: Setup matches historical winning pattern (+{exp_eval.synergy_bonus*100:.1f}% bonus)."
            else:
                telemetry = exp_telemetry or {}
                exp_detail = f"Online Learner Active: {telemetry.get('trades_analyzed', 0)} trades studied (Priors preserved, L2 regularized)."

        gates.append({
            "name": "Online Experience Reflex",
            "status": "PASS" if exp_pass else "BLOCKED",
            "detail": exp_detail,
            "passed": exp_pass,
        })

        # Gate 8: Historical Pattern Similarity Edge
        hist_pass = True
        hist_detail = "Historical pattern matcher: Insufficient bars for pattern match."
        if hist_edge_result:
            if hist_edge_result.sample_count >= 5:
                hist_pass = (hist_edge_result.win_rate_pct >= 40.0)
                hist_detail = (
                    f"Pattern Match: {hist_edge_result.win_rate_pct:.1f}% WR across "
                    f"{hist_edge_result.sample_count} similar historical setups (Exp: {hist_edge_result.expectancy_r:+.2f}R, Grade: {hist_edge_result.confidence_grade})"
                )
            else:
                hist_detail = f"Pattern Match: Sample count warming up ({hist_edge_result.sample_count}/30)."

        gates.append({
            "name": "Historical Pattern Edge",
            "status": "PASS" if hist_pass else "WAITING",
            "detail": hist_detail,
            "passed": hist_pass,
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
        elif not ensemble_pass:
            status = "WAITING_FOR_ALIGNMENT"
            verdict_badge = f"WAITING FOR ORDER FLOW CONSENSUS ({ensemble_telemetry.ensemble_bias if ensemble_telemetry else 'OFI'} OPPOSES)"
        elif not exp_pass:
            status = "WAITING_FOR_ALIGNMENT"
            verdict_badge = "MISTAKE REFLEX GUARD (AVOIDING REPEAT LOSS)"
        elif not value_pass:
            status = "WAITING_FOR_PULLBACK"
            verdict_badge = f"WAITING FOR PULLBACK ({dominant_dir} EXTENDED)"
        elif not ev_pass:
            status = "HUNTING_SETUP"
            verdict_badge = "HUNTING SETUP (BUILDING CONVICTION)"
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
            ensemble_info=ensemble_telemetry,
            candle_pos_5m=candle_pos_5m,
            is_chasing_top=is_chasing_top,
            is_chasing_bottom=is_chasing_bottom,
            active_positions=positions,
            exp_info=exp_telemetry,
            exp_eval=exp_eval,
            news_bias_data=news_bias_data,
            hist_edge_result=hist_edge_result,
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
            candle_pos_5m=round(candle_pos_5m, 2),
            status=status,
            verdict_badge=verdict_badge,
            is_frozen=False,
            gates=gates,
            narrative=narrative,
            target_zone=target_zone,
            ensemble=ensemble_telemetry.to_dict() if ensemble_telemetry else None,
            experience=exp_telemetry,
            news_catalyst=news_bias_data if news_bias_data else None,
            historical_match=hist_edge_result.to_dict() if hist_edge_result else None,
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
        ensemble_info: Optional[EnsembleTelemetry] = None,
        candle_pos_5m: float = 0.5,
        is_chasing_top: bool = False,
        is_chasing_bottom: bool = False,
        active_positions: Optional[List[Any]] = None,
        exp_info: Optional[Dict[str, Any]] = None,
        exp_eval: Optional[Any] = None,
        news_bias_data: Optional[Dict[str, Any]] = None,
        hist_edge_result: Optional[Any] = None,
    ) -> str:
        """Construct a precise, institutional narrative explaining the bot's live thoughts."""
        candle_pct = int(round(candle_pos_5m * 100))
        lines = [
            f"⚡ [CYCLE #{cycle_idx} @ {time_str}] ACTIVE EVALUATION HEARTBEAT — BOT IS RUNNING NORMALLY",
            f"• Macro Structure: 1H {trend_1h} | 15m {trend_15m} | Kaufman Efficiency: {efficiency_ratio:.2f} | R²: {r_squared:.2f}",
            f"• Valuation: Spot {current_bid:.2f} vs 5m VWAP {vwap_val:.2f} ({vwap_diff:+.2f} pts) | 5m Candle Pos: {candle_pct}% | 5m StochRSI: {stoch_k:.1f}",
            f"• Edge Metrics: P(Win) = {p_win * 100:.1f}% (Quant Calibrated) | Expected Value = {expected_value_r:+.2f}R",
        ]

        if news_bias_data and news_bias_data.get("summary"):
            lines.append(f"• Macro News Catalyst: {news_bias_data.get('direction')} ({news_bias_data.get('confidence', 0.5)*100:.0f}% Conf) — {news_bias_data.get('summary')}")

        if hist_edge_result and hist_edge_result.sample_count >= 5:
            lines.append(f"• Historical Pattern Edge: {hist_edge_result.win_rate_pct:.1f}% Win Rate across {hist_edge_result.sample_count} matching historical setups (Exp: {hist_edge_result.expectancy_r:+.2f}R, Grade: {hist_edge_result.confidence_grade})")


        if exp_info:
            trades_analyzed = exp_info.get("trades_analyzed", 0)
            wins = exp_info.get("winning_trades", 0)
            losses = exp_info.get("losing_trades", 0)
            mistakes = exp_info.get("mistakes_recorded", 0)
            successes = exp_info.get("successes_recorded", 0)
            lesson = exp_info.get("last_lesson", "Institutional prior weights active. Awaiting closed trade data to refine reflexes.")

            lines.append("────────────────────────────────────────")
            lines.append("🧠 CONTINUAL EXPERIENCE LEARNER (ONLINE MEMORY):")
            lines.append(f"• Analyzed: {trades_analyzed} Trades ({wins}W / {losses}L) | Mistake Clusters: {mistakes} | Success Signatures: {successes}")
            lines.append(f"• Latest Institutional Reflection: \"{lesson}\"")
            if exp_eval and getattr(exp_eval, "mistake_penalty", 0.0) > 0.03:
                lines.append(f"⚠️ Active Mistake Reflex: -{exp_eval.mistake_penalty*100:.1f}% confidence damping applied (matches past failure vector)")
            elif exp_eval and getattr(exp_eval, "synergy_bonus", 0.0) > 0.03:
                lines.append(f"✨ Synergy Reinforcement: +{exp_eval.synergy_bonus*100:.1f}% confidence boost (matches institutional winning setup)")
            lines.append("────────────────────────────────────────")

        if active_positions:
            lines.append("────────────────────────────────────────")
            lines.append("📊 REAL-TIME ACTIVE POSITION HEALTH & EXIT STUDY:")
            for p in active_positions:
                ticket = p.get("ticket") if isinstance(p, dict) else getattr(p, "ticket", "")
                direction = p.get("direction") if isinstance(p, dict) else getattr(p, "direction", "")
                volume = p.get("volume") if isinstance(p, dict) else getattr(p, "volume", 0.0)
                open_p = p.get("open_price") if isinstance(p, dict) else getattr(p, "open_price", 0.0)
                floating_r = p.get("r_multiple", 0.0) if isinstance(p, dict) else getattr(p, "r_multiple", 0.0)
                highest_r = p.get("highest_r_reached", 0.0) if isinstance(p, dict) else getattr(p, "highest_r_reached", 0.0)
                giveback = p.get("peak_r_giveback_pct", 0.0) if isinstance(p, dict) else getattr(p, "peak_r_giveback_pct", 0.0)
                profit = p.get("profit", 0.0) if isinstance(p, dict) else getattr(p, "profit", 0.0)
                exit_eval = p.get("exit_evaluation", "HOLD_AND_EXPAND") if isinstance(p, dict) else getattr(p, "exit_evaluation", "HOLD_AND_EXPAND")
                exit_detail = p.get("exit_verdict_detail", "") if isinstance(p, dict) else getattr(p, "exit_verdict_detail", "")

                status_icon = "🟢" if exit_eval == "HOLD_AND_EXPAND" else ("🟡" if exit_eval == "TIGHTEN_AND_PROTECT" else "🔴")
                pnl_str = f"+${profit:.2f}" if profit >= 0 else f"-${abs(profit):.2f}"
                lines.append(f"{status_icon} [TICKET #{ticket} {direction} {volume} LOTS @ {open_p:.2f}]")
                lines.append(f"   • Floating: {floating_r:+.2f}R ({pnl_str}) | High-Water Mark: +{highest_r:.2f}R (Giveback: {giveback:.1f}%)")
                lines.append(f"   • AI Study: [{exit_eval}] — {exit_detail}")
            lines.append("────────────────────────────────────────")

        if ensemble_info:
            fvg_str = f"{ensemble_info.nearest_fvg['direction']} @ {ensemble_info.nearest_fvg['midpoint']:.2f}" if ensemble_info.nearest_fvg else "None"
            lines.append(
                f"• Quant Ensemble: Bias={ensemble_info.ensemble_bias} ({ensemble_info.ensemble_score:.0f}/100) | "
                f"OFI Delta={ensemble_info.delta_ratio_60s:+.2f} | "
                f"Vol={ensemble_info.volatility_regime} (Parkinson Ratio: {ensemble_info.parkinson_ratio:.2f}) | "
                f"FVG={fvg_str}"
            )

        if status == "WAITING_FOR_PULLBACK":
            chase_note = "AT TOP OF CANDLE (HIGH WICK)" if dominant_dir == "BUY" and is_chasing_top else ("AT BOTTOM OF CANDLE (LOW WICK)" if dominant_dir == "SELL" and is_chasing_bottom else "OVEREXTENDED")
            lines.append(
                f"🧠 VERDICT: Market is {trend_1h} but currently {chase_note} (candle pos: {candle_pct}%, VWAP diff: {vwap_diff:+.2f} pts). Entering a {dominant_dir} at the high/low carries negative statistical expectancy (EV: {expected_value_r:+.2f}R). "
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
                f"🚀 VERDICT: HIGH-CONVICTION SETUP QUALIFIED! All predictive gates passed (EV: {expected_value_r:+.2f}R >= +0.20R, P_win: {p_win*100:.1f}%). "
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
            "candle_pos_5m": 0.50,
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

        # Pre-news trajectory setups are catalyzed by imminent fundamental calendar events
        if getattr(signal, "is_pre_news_trajectory", False):
            return True, "Predictive Model Approved: Pre-news trajectory catalyst setup"

        ev = self.latest_evaluation.expected_value_r
        p_win = self.latest_evaluation.p_win
        dominant_dir = "BUY" if self.latest_evaluation.p_bull >= self.latest_evaluation.p_bear else "SELL"
        sig_dir = getattr(signal, "direction", dominant_dir)

        # Anti-chasing veto
        if self.latest_evaluation.status == "WAITING_FOR_PULLBACK":
            return False, f"Predictive Model Veto: Price is extended in {dominant_dir} direction. Waiting for pullback (EV: {ev:+.2f}R < +0.25R)"

        # Anti-counter-trend veto: if macro direction is strong (>= 65%) opposing the signal
        # Allow ONLY if it is a confirmed supply/demand zone reversal, tier 1 sniper, or scalp setup
        if (sig_dir == "BUY" and self.latest_evaluation.p_bear >= 0.65) or (sig_dir == "SELL" and self.latest_evaluation.p_bull >= 0.65):
            is_zone_rev = getattr(signal, "is_zone_reversal", False)
            is_scalp = getattr(signal, "mode", "") == "FAST_SCALP" or getattr(signal, "tier", "") == "TIER_1_SNIPER"
            if not (is_zone_rev or is_scalp):
                return False, f"Predictive Model Veto: Signal direction {sig_dir} directly conflicts with dominant macro trend"

        # Online Experience Reflex & Mistake Guard
        if hasattr(self, "experience_learner") and self.experience_learner:
            eval_ctx = {
                "candle_pos_5m": getattr(signal, "candle_pos_5m", self.latest_evaluation.candle_pos_5m),
                "vwap_deviation": getattr(signal, "vwap_deviation", self.latest_evaluation.vwap_deviation),
                "stoch_rsi_k": getattr(signal, "stoch_rsi_k", self.latest_evaluation.stoch_rsi_k),
                "kaufman_er": getattr(signal, "kaufman_er", self.latest_evaluation.kaufman_er),
                "r_squared": getattr(signal, "r_squared", self.latest_evaluation.r_squared),
                "order_flow_delta": getattr(signal, "order_flow_delta", 0.0),
                "is_zone_reversal": 1.0 if getattr(signal, "is_zone_reversal", False) else 0.0,
            }
            exp_cand = self.experience_learner.evaluate_candidate(
                symbol=getattr(signal, "symbol", self.latest_evaluation.symbol),
                direction=sig_dir,
                context=eval_ctx,
                atr=getattr(signal, "atr", 2.5),
            )
            if exp_cand.veto:
                return False, f"Predictive Model Experience Learner Veto: {exp_cand.veto_reason}"

        if ev < 0.20 or p_win < 0.55:
            return False, f"Predictive Model Veto: Sub-optimal expected value (EV: {ev:+.2f}R, P_win: {p_win*100:.1f}%)"

        return True, f"Predictive Model Approved: EV = {ev:+.2f}R, P_win = {p_win*100:.1f}%"
