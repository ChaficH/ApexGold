"""
ApexGold Production Strategy Engine (V2).
Orchestrates:
1. Base ConfluenceScorer: 100-Point multi-timeframe structural and technical scoring.
2. MarketRegimeDetector: Causal ADX, Choppiness Index, and volatility regime gating.
3. MetaLabelerFilter: LightGBM ML probability gating with Purged CV conviction threshold.
4. DynamicExitManager: Volatility-aware dynamic TP (1.4R-1.7R) and breathing-room ratchets.

Safety & Backward Compatibility:
- All new logic is configurable via flags and defaults ON only for release candidate.
- If all V2 flags are set to False, behavior is 100% byte-identical to legacy baseline.
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from strategy.confluence_scorer import ConfluenceScorer, ConfluenceSignal, SignalResult
from strategy.v2.exit_manager import DynamicExitManager, DynamicExitPlan
from strategy.v2.meta_labeler import MetaLabelerFilter
from strategy.v2.regime_detector import MarketRegimeDetector, MarketRegime, RegimeSnapshot

logger = logging.getLogger("ApexGold.StrategyV2")


class ApexStrategyV2:
    """
    Institutional Multi-Component Gold Trading Strategy (V2).
    Integrates Confluence Scoring, Regime Gating, ML Meta-Labeling, and Dynamic Volatility Exits.
    """

    def __init__(
        self,
        base_scorer: Optional[ConfluenceScorer] = None,
        regime_detector: Optional[MarketRegimeDetector] = None,
        exit_manager: Optional[DynamicExitManager] = None,
        meta_labeler: Optional[MetaLabelerFilter] = None,
        regime_filter_enabled: bool = True,
        dynamic_exits_enabled: bool = True,
        meta_labeler_enabled: bool = True,
        asia_session_filter_enabled: bool = True,
        min_p_win: float = 0.50,
        mode: str = "balanced",
    ):
        self.base_scorer = base_scorer or ConfluenceScorer(mode=mode)
        self.regime_detector = regime_detector or MarketRegimeDetector()
        self.exit_manager = exit_manager or DynamicExitManager()
        self.meta_labeler = meta_labeler or MetaLabelerFilter(
            model_path="models/metalabeler_v1.txt",
            probability_threshold=min_p_win,
            enabled=meta_labeler_enabled,
        )

        self.regime_filter_enabled = regime_filter_enabled
        self.dynamic_exits_enabled = dynamic_exits_enabled
        self.meta_labeler_enabled = meta_labeler_enabled
        self.asia_session_filter_enabled = asia_session_filter_enabled
        self.mode = mode

    def get_active_threshold(self) -> int:
        return self.base_scorer.get_active_threshold()

    def evaluate(
        self,
        symbol: str,
        dfs: Dict[str, pd.DataFrame],
        current_tick_bid: float,
        current_tick_ask: float,
        utc_timestamp: Optional[float] = None,
        **kwargs: Any,
    ) -> SignalResult:
        """
        Evaluate market setups through confluence scoring, regime filtering,
        ML meta-labeling, and dynamic exit configuration.
        """
        # 1. Base Confluence Scoring
        raw_result = self.base_scorer.evaluate(
            symbol=symbol,
            dfs=dfs,
            current_tick_bid=current_tick_bid,
            current_tick_ask=current_tick_ask,
            utc_timestamp=utc_timestamp,
            **kwargs,
        )

        buy_sig, sell_sig = raw_result[0], raw_result[1]
        if not buy_sig and not sell_sig:
            return raw_result

        # 2. Causal Regime Detection
        snap = self.regime_detector.classify(dfs)

        filtered_buy = self._filter_and_enhance_signal(buy_sig, "BUY", snap, dfs, utc_timestamp)
        filtered_sell = self._filter_and_enhance_signal(sell_sig, "SELL", snap, dfs, utc_timestamp)

        return SignalResult(filtered_buy, filtered_sell)

    def _filter_and_enhance_signal(
        self,
        sig: Optional[ConfluenceSignal],
        direction: str,
        snap: RegimeSnapshot,
        dfs: Dict[str, pd.DataFrame],
        utc_timestamp: Optional[float],
    ) -> Optional[ConfluenceSignal]:
        """Apply regime filter, meta-labeler gate, and dynamic exit fitting to candidate signal."""
        if sig is None or sig.score < self.get_active_threshold():
            return None

        is_zone_reversal = "Demand Zone" in sig.reason or "Supply Zone" in sig.reason or "Sweep" in sig.reason

        # A. Session & Regime Gating
        if self.regime_filter_enabled:
            allowed, reason = self.regime_detector.filter_trade(
                direction=direction,
                regime_snap=snap,
                is_zone_reversal=is_zone_reversal,
                utc_timestamp=utc_timestamp if self.asia_session_filter_enabled else None,
            )
            if not allowed:
                logger.debug(f"[{sig.symbol} {direction}] Setup rejected by Regime Gate: {reason}")
                return None

            # Macro Trend Slope Alignment: never fight established 1H flow unless verified zone reversal
            if not is_zone_reversal:
                if direction == "BUY" and snap.h1_ema_slope < -0.01:
                    logger.debug(f"[{sig.symbol} BUY] Rejected: Fighting negative 1H slope ({snap.h1_ema_slope:.4f})")
                    return None
                elif direction == "SELL" and snap.h1_ema_slope > 0.01:
                    logger.debug(f"[{sig.symbol} SELL] Rejected: Fighting positive 1H slope ({snap.h1_ema_slope:.4f})")
                    return None

            # London Open Whipsaw Gate (07:00 to 09:00 UTC):
            # Block trend breakouts during European open volatility expansion unless verified zone reversal
            if utc_timestamp:
                dt_utc = pd.to_datetime(utc_timestamp, unit="s", utc=True)
                if 7 <= dt_utc.hour < 9 and not is_zone_reversal:
                    logger.debug(f"[{sig.symbol} {direction}] Rejected: London opening whipsaw window (07:00-09:00 UTC)")
                    return None

            # Institutional VWAP Gate: Never buy below VWAP or sell above VWAP unless confirmed zone reversal
            df_5m = dfs.get("5m")
            if df_5m is not None and len(df_5m) >= 20 and not is_zone_reversal:
                from engine.indicators import TechnicalIndicators
                vwap_info = TechnicalIndicators.calculate_vwap(df_5m)
                if vwap_info.get("valid"):
                    vwap_val = vwap_info.get("vwap", 0.0)
                    cur_price = float(df_5m["close"].iloc[-1])
                    if direction == "BUY" and cur_price < (vwap_val - 0.25):
                        logger.debug(f"[{sig.symbol} BUY] Rejected: Below Institutional VWAP ({cur_price:.2f} < {vwap_val:.2f})")
                        return None
                    elif direction == "SELL" and cur_price > (vwap_val + 0.25):
                        logger.debug(f"[{sig.symbol} SELL] Rejected: Above Institutional VWAP ({cur_price:.2f} > {vwap_val:.2f})")
                        return None

        # B. ML Meta-Labeling Conviction Filter
        if self.meta_labeler_enabled and self.meta_labeler.is_loaded:
            dt = pd.to_datetime(utc_timestamp or sig.time, unit="s", utc=True)
            hour = dt.hour
            sess_code = 0 if hour < 7 else (1 if hour < 13 else (2 if hour < 21 else 3))
            reg_code = 0 if snap.regime.value == "RANGE_BOUND" else (1 if snap.regime.value == "TREND_BULL" else (2 if snap.regime.value == "TREND_BEAR" else 3))

            features = {
                "score": sig.score,
                "h1_macro": sig.breakdown.get("h1_macro", 10),
                "m15_wave": sig.breakdown.get("m15_wave", 12),
                "m5_setup": sig.breakdown.get("m5_setup", 25),
                "m1_trigger": sig.breakdown.get("m1_trigger", 15),
                "smc_institutional": sig.breakdown.get("smc_institutional", 8),
                "quant_alpha": sig.breakdown.get("quant_alpha", 5),
                "adx_15m": snap.adx_15m,
                "choppiness_index": snap.choppiness_index,
                "atr_percentile": snap.atr_percentile,
                "h1_ema_slope": snap.h1_ema_slope,
                "hour": hour,
                "session_code": sess_code,
                "is_zone_reversal": 1 if is_zone_reversal else 0,
                "regime_code": reg_code,
            }

            ml_allowed, p_win, ml_reason = self.meta_labeler.filter_trade(features)
            if not ml_allowed:
                logger.debug(f"[{sig.symbol} {direction}] Setup rejected by Meta-Labeler: {ml_reason}")
                return None

        # C. Dynamic Volatility Exit Plan
        if self.dynamic_exits_enabled:
            df_5m = dfs.get("5m")
            atr_5m = 2.0
            if df_5m is not None and len(df_5m) >= 15:
                from engine.indicators import TechnicalIndicators
                atrs = TechnicalIndicators.atr(df_5m["high"].values, df_5m["low"].values, df_5m["close"].values, 14)
                atr_5m = float(atrs[-1]) if not np.isnan(atrs[-1]) and atrs[-1] > 0 else 2.0

            plan = self.exit_manager.compute_exit_plan(
                entry_price=sig.entry_price,
                initial_sl=sig.suggested_sl,
                direction=direction,
                atr_5m=atr_5m,
            )
            sig.suggested_tp = plan.suggested_tp
            sig.target_rr = plan.target_r
            sig.be_lock_r = plan.be_lock_r

        return sig
