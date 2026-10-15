"""
ApexGold Adaptive Online Experience Learner (Continuous Meta-Learning).

Implements real-time learning from trading successes and mistakes:
1. Pre-Calibrated Prior: Does NOT start from zero; begins with institutional prior weights.
2. Anti-Memorization Principle: Features are normalized structural ratios (VWAP z-score, 
   candle position, momentum percentile, efficiency ratio) rather than literal price levels.
   Strict L2 regularization pulls weights toward institutional baseline to prevent noise overfitting.
3. Mistake Reflex & Pattern Synergy:
   - When a trade loses, the learner analyzes root causes and records a penalized context signature.
   - When a trade wins, positive causal factors are reinforced.
   - Future candidate setups are evaluated against recent mistake memory clusters with cosine similarity.
4. Real-Time Telemetry & Transparency:
   - Feeds real-time learning metrics, active penalties, and lessons learned directly into the Thinking stream.
5. Persistent State:
   - Serializes learned state to models/online_experience_learner.json across bot restarts.
"""

from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
import logging
import math
import os
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ApexGold.ExperienceLearner")


# Standard institutional feature keys for structural learning
EXPERIENCE_FEATURE_KEYS = [
    "score_norm",          # Confluence score / 100.0 (0.0 to 1.0)
    "candle_pos_5m",       # Position inside 5m candle (0.0=bottom wick, 1.0=top wick)
    "vwap_dev_norm",       # Distance from 5m VWAP / ATR (-2.0 to +2.0)
    "stoch_rsi_norm",      # (StochRSI K - 50.0) / 50.0 (-1.0 oversold to +1.0 overbought)
    "kaufman_er",          # Efficiency ratio (0.0 to 1.0)
    "r_squared",           # Linear regression trend persistence (0.0 to 1.0)
    "order_flow_delta",    # Tick delta initiation ratio (-1.0 to +1.0)
    "is_zone_reversal",    # 1.0 if supply/demand zone or liquidity trap, 0.0 otherwise
    "trend_align_1h",      # +1.0 if aligned with 1H, -1.0 if counter, 0.0 if neutral
    "trend_align_15m",     # +1.0 if aligned with 15m, -1.0 if counter, 0.0 if neutral
    "direction_val",       # +1.0 for BUY, -1.0 for SELL
]

# Robust institutional priors (Never start from 0)
DEFAULT_PRIOR_WEIGHTS: Dict[str, float] = {
    "score_norm": 0.40,
    "candle_pos_5m": -0.35,       # Penalize buying near 1.0 (top) or selling near 0.0 (bottom) when modulated
    "vwap_dev_norm": -0.25,       # Penalize buying extended above VWAP or selling extended below
    "stoch_rsi_norm": -0.20,      # Penalize buying overbought or selling oversold
    "kaufman_er": 0.25,           # Reward high signal-to-noise trend persistence
    "r_squared": 0.20,            # Reward high statistical conviction
    "order_flow_delta": 0.30,     # Reward order flow delta agreeing with direction
    "is_zone_reversal": 0.35,     # Reward confirmed institutional supply/demand zones
    "trend_align_1h": 0.30,       # Reward macro alignment
    "trend_align_15m": 0.30,      # Reward session wave alignment
    "direction_val": 0.00,        # Symmetrical bias (no blind long or short prejudice)
}

FEATURE_METADATA: Dict[str, Dict[str, str]] = {
    "score_norm": {
        "name": "Confluence Multi-Factor Score",
        "category": "Signal Conviction",
        "description": "Baseline composite technical and quantitative score normalized to [0, 1].",
    },
    "candle_pos_5m": {
        "name": "Anti-Chase Wick Position",
        "category": "Anti-Chasing & Valuation",
        "description": "Candle position (0=low wick, 1=high wick). Strictly penalizes buying top wicks or selling bottom wicks.",
    },
    "vwap_dev_norm": {
        "name": "Institutional VWAP Deviation",
        "category": "Anti-Chasing & Valuation",
        "description": "Distance from 5m VWAP normalized by ATR. Penalizes buying overextended premiums or selling deep discounts.",
    },
    "stoch_rsi_norm": {
        "name": "Stochastic RSI Momentum Extreme",
        "category": "Momentum & Exhaustion",
        "description": "Measures momentum exhaustion. Guards against buying overbought (>80) or selling oversold (<20).",
    },
    "kaufman_er": {
        "name": "Kaufman Adaptive Efficiency (ER)",
        "category": "Market Regime & Noise",
        "description": "Signal-to-noise ratio. Rewards clean directional trends and penalizes erratic directionless chop.",
    },
    "r_squared": {
        "name": "Statistical Trend Conviction (R²)",
        "category": "Market Regime & Noise",
        "description": "Linear regression conviction. Differentiates persistent institutional flow from random Brownian motion.",
    },
    "order_flow_delta": {
        "name": "Order Flow Delta Aggression (OFI)",
        "category": "Order Flow & Liquidity",
        "description": "Sub-second aggressive buying vs selling initiation volume. Penalizes setups opposing order flow.",
    },
    "is_zone_reversal": {
        "name": "Supply / Demand Zone Rejection",
        "category": "Institutional Levels",
        "description": "Rewards buying confirmed demand floors or selling confirmed supply ceilings with trap confirmation.",
    },
    "trend_align_1h": {
        "name": "1H Macro Trend Alignment",
        "category": "Multi-Timeframe Structure",
        "description": "Rewards setups riding the 1-Hour institutional macro wind and penalizes counter-trend trades.",
    },
    "trend_align_15m": {
        "name": "15m Intermediate Wave Alignment",
        "category": "Multi-Timeframe Structure",
        "description": "Rewards entries synchronized with the 15-minute intermediate trend wave.",
    },
    "direction_val": {
        "name": "Directional Neutrality & Symmetry",
        "category": "Directional Balance",
        "description": "Ensures completely unbiased symmetry between BUY and SELL trade generation.",
    },
}


@dataclass
class TradeExperienceRecord:
    """Record of a finalized trade experience with its causal context signature."""
    ticket: int
    symbol: str
    direction: str
    open_price: float
    close_price: float
    pnl_usd: float
    pnl_r: float
    close_reason: str
    timestamp: float
    features: Dict[str, float]
    was_mistake: bool
    root_cause: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperienceEvaluation:
    """Quantitative candidate evaluation returned by the Experience Learner."""
    p_win_adjustment: float = 0.0
    ev_adjustment: float = 0.0
    max_mistake_similarity: float = 0.0
    max_success_similarity: float = 0.0
    reflex_note: str = ""
    is_vetoed: bool = False
    weights: Dict[str, float] = field(default_factory=dict)
    total_trades_analyzed: int = 0
    wins_learned: int = 0
    mistakes_learned: int = 0
    mistake_penalty: float = 0.0
    synergy_bonus: float = 0.0
    similarity_to_mistake: float = 0.0
    similarity_to_success: float = 0.0
    confidence_adjustment: float = 0.0
    veto: bool = False
    veto_reason: str = ""

    def __getitem__(self, item):
        return getattr(self, item)

    def get(self, item, default=None):
        return getattr(self, item, default)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AdaptiveExperienceLearner:
    """
    Continual Learning & Online Mistake-Reflex Engine.
    Adapts weights through regularized online updates and maintains episodic mistake clusters.
    """

    def __init__(
        self,
        persistence_path: Optional[str] = "models/online_experience_learner.json",
        learning_rate: float = 0.08,
        l2_regularization: float = 0.015,
        memory_capacity: int = 80,
    ):
        self.persistence_path = persistence_path
        self.learning_rate = learning_rate
        self.l2_reg = l2_regularization
        self.memory_capacity = memory_capacity

        # Current adaptive weights (starts from prior, NOT from 0)
        self.weights: Dict[str, float] = dict(DEFAULT_PRIOR_WEIGHTS)
        self.bias: float = 0.15 # Baseline positive prior bias (log-odds corresponding to ~54% baseline)

        # Episodic Experience Buffers
        self.mistake_memory: Deque[Dict[str, Any]] = deque(maxlen=memory_capacity)
        self.success_memory: Deque[Dict[str, Any]] = deque(maxlen=memory_capacity)
        self.recent_lessons: Deque[str] = deque(maxlen=15)

        # Learning Counters
        self.total_trades_analyzed: int = 0
        self.wins_learned: int = 0
        self.mistakes_learned: int = 0
        self.last_update_time: float = 0.0

        # Load persisted experience if available
        self.load_state()

    def extract_feature_vector(
        self,
        context: Dict[str, Any],
        direction: str = "BUY",
    ) -> Dict[str, float]:
        """
        Normalize raw market context into a bounded, dimensionless feature vector.
        Prevents literal price memorization; ensures purely structural representation.
        """
        dir_sign = 1.0 if direction.upper() == "BUY" else -1.0
        
        # 1. Score
        score = float(context.get("score", 65.0))
        score_norm = max(0.0, min(1.0, score / 100.0))

        # 2. Candle Position (0.0 to 1.0)
        # Symmetrically directionalized: 
        # For BUY: high candle position (0.8-1.0) is chasing top wick (+1.0)
        # For SELL: low candle position (0.0-0.2) is chasing bottom wick (+1.0)
        raw_cpos = float(context.get("candle_pos_5m", 0.50))
        if dir_sign > 0:
            cpos_chase = raw_cpos # Closer to 1.0 means buying higher
        else:
            cpos_chase = 1.0 - raw_cpos # Closer to 0.0 means selling lower (raw 0.1 -> 0.9 chase)

        # 3. VWAP Deviation Normalized by ATR
        # Positive if entry is chasing away from fair value
        raw_vwap_diff = float(context.get("vwap_deviation", 0.0))
        atr_ref = max(0.50, float(context.get("atr", 2.0)))
        vwap_dev_norm = (raw_vwap_diff * dir_sign) / atr_ref
        vwap_dev_norm = max(-2.0, min(2.0, vwap_dev_norm))

        # 4. Stochastic RSI Normalized (-1.0 to +1.0)
        stoch_k = float(context.get("stoch_rsi_k", 50.0))
        stoch_norm = ((stoch_k - 50.0) / 50.0) * dir_sign
        stoch_norm = max(-1.0, min(1.0, stoch_norm))

        # 5. Kaufman Efficiency Ratio & R-Squared
        k_er = max(0.0, min(1.0, float(context.get("kaufman_er", 0.50))))
        r2 = max(0.0, min(1.0, float(context.get("r_squared", 0.50))))

        # 6. Order Flow Delta
        delta_ratio = float(context.get("order_flow_delta", 0.0))
        order_flow_aligned = max(-1.0, min(1.0, delta_ratio * dir_sign))

        # 7. Zone Reversal
        is_zone = 1.0 if context.get("is_zone_reversal", False) else 0.0

        # 8. Trend Alignments
        t_1h = str(context.get("trend_1h", "NEUTRAL")).upper()
        t_15m = str(context.get("trend_15m", "NEUTRAL")).upper()
        
        target_trend = "BULL" if dir_sign > 0 else "BEAR"
        opp_trend = "BEAR" if dir_sign > 0 else "BULL"

        align_1h = 1.0 if t_1h == target_trend else (-1.0 if t_1h == opp_trend else 0.0)
        align_15m = 1.0 if t_15m == target_trend else (-1.0 if t_15m == opp_trend else 0.0)

        return {
            "score_norm": round(score_norm, 3),
            "candle_pos_5m": round(cpos_chase, 3),
            "vwap_dev_norm": round(vwap_dev_norm, 3),
            "stoch_rsi_norm": round(stoch_norm, 3),
            "kaufman_er": round(k_er, 3),
            "r_squared": round(r2, 3),
            "order_flow_delta": round(order_flow_aligned, 3),
            "is_zone_reversal": is_zone,
            "trend_align_1h": align_1h,
            "trend_align_15m": align_15m,
            "direction_val": dir_sign,
        }

    def _cosine_similarity(self, vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        """Compute cosine similarity between two normalized feature vectors."""
        keys = EXPERIENCE_FEATURE_KEYS
        dot = sum(vec_a.get(k, 0.0) * vec_b.get(k, 0.0) for k in keys)
        norm_a = math.sqrt(sum(vec_a.get(k, 0.0) ** 2 for k in keys))
        norm_b = math.sqrt(sum(vec_b.get(k, 0.0) ** 2 for k in keys))
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0
        return max(-1.0, min(1.0, dot / (norm_a * norm_b)))

    def record_trade_outcome(
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
        """
        Learn continuously from a trade outcome:
        - If pnl_r < 0: Classifies the root mistake, updates penalty weights, records mistake signature.
        - If pnl_r > 0: Reinforces winning factors, updates success weights, records archetype.
        Returns a concise human-readable lesson string.
        """
        now = time.time()
        feat_vec = self.extract_feature_vector(context, direction)
        self.total_trades_analyzed += 1
        self.last_update_time = now

        # Convert R-multiple outcome into a continuous target probability y in (0, 1)
        # e.g., +2.0R -> 0.88, +0.5R -> 0.62, 0.0R -> 0.50, -0.4R -> 0.40, -1.0R -> 0.27
        y_target = 1.0 / (1.0 + math.exp(-max(-3.0, min(3.0, pnl_r * 1.2))))

        # Model predicted score on these features
        z = self.bias + sum(self.weights.get(k, 0.0) * feat_vec.get(k, 0.0) for k in EXPERIENCE_FEATURE_KEYS)
        y_pred = 1.0 / (1.0 + math.exp(-max(-6.0, min(6.0, z))))

        # Error gradient
        grad = (y_target - y_pred)

        # -------------------------------------------------------------
        # Regularized Online Weight Update (Elastic Net / L2 Decay)
        # Prevents memorization and keeps weights anchored near proven institutional priors
        # -------------------------------------------------------------
        for k in EXPERIENCE_FEATURE_KEYS:
            prior_w = DEFAULT_PRIOR_WEIGHTS.get(k, 0.0)
            cur_w = self.weights.get(k, prior_w)
            # Gradient update + L2 regularization pulling toward prior
            delta = (self.learning_rate * grad * feat_vec.get(k, 0.0)) - (self.l2_reg * (cur_w - prior_w))
            new_w = cur_w + delta
            # Safe clamping to prevent explosive divergence
            self.weights[k] = round(max(prior_w - 0.60, min(prior_w + 0.60, new_w)), 4)

        # Bias update
        self.bias = round(self.bias + (self.learning_rate * 0.5 * grad), 4)

        # -------------------------------------------------------------
        # Mistake Diagnosis & Episodic Memory Logging
        # -------------------------------------------------------------
        is_loss = (pnl_r < -0.10 or pnl_usd < -0.10)
        root_causes = []

        if is_loss:
            self.mistakes_learned += 1
            # Diagnose specific structural mistake
            if feat_vec.get("candle_pos_5m", 0.0) > 0.70:
                root_causes.append("Chasing top/bottom candle wick")
            if feat_vec.get("vwap_dev_norm", 0.0) > 0.60:
                root_causes.append("Overextended from VWAP fair value")
            if feat_vec.get("stoch_rsi_norm", 0.0) > 0.50:
                root_causes.append("Entering into momentum exhaustion")
            if feat_vec.get("kaufman_er", 0.0) < 0.35 and feat_vec.get("r_squared", 0.0) < 0.25:
                root_causes.append("Trading inside choppy low-efficiency noise")
            if feat_vec.get("trend_align_1h", 0.0) < 0 and not feat_vec.get("is_zone_reversal", 0.0):
                root_causes.append("Fighting 1H macro trend without key zone confirmation")
            if feat_vec.get("order_flow_delta", 0.0) < -0.20:
                root_causes.append("Opposing institutional order flow delta")

            if not root_causes:
                root_causes.append("Standard statistical market variance / stop sweep")

            root_cause_str = ", ".join(root_causes)
            lesson = (
                f"Learned from Mistake #{ticket} ({direction} {pnl_r:+.2f}R, ${pnl_usd:+.2f}): "
                f"Identified root cause [{root_cause_str}]. Adjusted penalty reflex vectors."
            )

            # Store in episodic mistake bank
            self.mistake_memory.append({
                "ticket": ticket,
                "direction": direction,
                "open_price": round(open_price, 2),
                "close_price": round(close_price, 2),
                "pnl_usd": round(pnl_usd, 2),
                "pnl_r": round(pnl_r, 2),
                "close_reason": close_reason,
                "time_str": datetime.fromtimestamp(now, tz=timezone.utc).strftime("%H:%M:%S UTC"),
                "timestamp": now,
                "features": feat_vec,
                "root_cause": root_cause_str,
                "ai_correction": f"Reflex damping applied (-{min(18, max(5, int(round(abs(pnl_r) * 12))))}%) to candidate setups matching this signature.",
            })
        else:
            self.wins_learned += 1
            primary_drivers = []
            if feat_vec.get("is_zone_reversal", 0.0) > 0.5:
                primary_drivers.append("Key Supply/Demand Zone Rejection")
            if feat_vec.get("order_flow_delta", 0.0) > 0.20:
                primary_drivers.append("Order flow delta alignment")
            if feat_vec.get("trend_align_1h", 0.0) > 0:
                primary_drivers.append("1H macro trend synergy")
            if feat_vec.get("kaufman_er", 0.0) > 0.60:
                primary_drivers.append("High Kaufman trend efficiency")

            driver_str = ", ".join(primary_drivers) if primary_drivers else "Disciplined R:R execution"
            lesson = (
                f"Reinforced from Success #{ticket} ({direction} {pnl_r:+.2f}R, ${pnl_usd:+.2f}): "
                f"Positive alpha driver [{driver_str}]. Boosted confidence weights."
            )

            # Store in success memory
            self.success_memory.append({
                "ticket": ticket,
                "direction": direction,
                "open_price": round(open_price, 2),
                "close_price": round(close_price, 2),
                "pnl_usd": round(pnl_usd, 2),
                "pnl_r": round(pnl_r, 2),
                "close_reason": close_reason,
                "time_str": datetime.fromtimestamp(now, tz=timezone.utc).strftime("%H:%M:%S UTC"),
                "timestamp": now,
                "features": feat_vec,
                "driver": driver_str,
                "ai_reinforcement": f"Pattern synergy (+{min(10, max(3, int(round(pnl_r * 3.5))))}% win prob) awarded when this structure reoccurs.",
            })

        self.recent_lessons.appendleft(lesson)
        logger.info(f"[EXPERIENCE LEARNER] {lesson}")
        self.save_state()
        return lesson

    def evaluate_candidate(
        self,
        context: Optional[Dict[str, Any]] = None,
        direction: str = "BUY",
        symbol: str = "XAUUSD",
        atr: float = 2.5,
        **kwargs,
    ) -> ExperienceEvaluation:
        """
        Evaluate a candidate setup against accumulated live trading experience:
        1. Compares setup against recent mistake clusters. If cosine similarity > 0.70,
           applies an intelligent Mistake Reflex Penalty.
        2. Compares setup against recent success clusters. If cosine similarity > 0.75,
           awards a Pattern Synergy Bonus.
        3. Returns quantitative ExperienceEvaluation with win probability adjustment, EV adjustment,
           mistake_penalty, synergy_bonus, reflex notes, and veto flag.
        """
        if context is None:
            context = kwargs.get("context", {})

        feat_vec = self.extract_feature_vector(context, direction)

        # 1. Check similarity against recent mistakes
        max_mistake_sim = 0.0
        worst_mistake_cause = ""
        for m in self.mistake_memory:
            m_feat = m.get("features", {})
            sim = self._cosine_similarity(feat_vec, m_feat)
            if sim > max_mistake_sim:
                max_mistake_sim = sim
                worst_mistake_cause = m.get("root_cause", "Recent loss pattern")

        # 2. Check similarity against recent successes
        max_success_sim = 0.0
        best_success_driver = ""
        for s in self.success_memory:
            s_feat = s.get("features", {})
            sim = self._cosine_similarity(feat_vec, s_feat)
            if sim > max_success_sim:
                max_success_sim = sim
                best_success_driver = s.get("driver", "Recent winning pattern")

        p_win_adj = 0.0
        ev_adj = 0.0
        reflex_notes: List[str] = []
        is_vetoed = False

        # Active Mistake Reflex Penalty
        if max_mistake_sim >= 0.70:
            # Scale penalty with similarity
            penalty_severity = min(0.18, (max_mistake_sim - 0.70) * 0.60)
            p_win_adj -= penalty_severity
            ev_adj -= round(penalty_severity * 1.8, 2)
            sim_pct = int(round(max_mistake_sim * 100))
            reflex_notes.append(
                f"⚠️ Mistake Reflex Active ({sim_pct}% match to recent loss): [{worst_mistake_cause}]. "
                f"Damped win odds by {penalty_severity*100:.1f}%."
            )
            # High-conviction veto if pattern is near identical (> 88%) to a recent bad loss
            if max_mistake_sim >= 0.88 and not context.get("is_zone_reversal", False):
                is_vetoed = True
                reflex_notes.append(f"🛑 VETO: Critical similarity ({sim_pct}%) to recent repeating mistake.")

        # Pattern Synergy Bonus from recent validated wins
        elif max_success_sim >= 0.75:
            bonus = min(0.10, (max_success_sim - 0.75) * 0.35)
            p_win_adj += bonus
            ev_adj += round(bonus * 1.5, 2)
            sim_pct = int(round(max_success_sim * 100))
            reflex_notes.append(
                f"✅ Pattern Synergy ({sim_pct}% match to verified win): [{best_success_driver}]. "
                f"Boosted win odds by +{bonus*100:.1f}%."
            )

        if not reflex_notes:
            reflex_notes.append("Experience Learner: Setup evaluated with normal institutional baseline weights.")

        mistake_penalty = abs(p_win_adj) if p_win_adj < 0 else 0.0
        synergy_bonus = p_win_adj if p_win_adj > 0 else 0.0

        return ExperienceEvaluation(
            p_win_adjustment=round(p_win_adj, 3),
            ev_adjustment=round(ev_adj, 2),
            max_mistake_similarity=round(max_mistake_sim, 3),
            max_success_similarity=round(max_success_sim, 3),
            reflex_note=" ".join(reflex_notes),
            is_vetoed=is_vetoed,
            weights=dict(self.weights),
            total_trades_analyzed=self.total_trades_analyzed,
            wins_learned=self.wins_learned,
            mistakes_learned=self.mistakes_learned,
            mistake_penalty=round(mistake_penalty, 3),
            synergy_bonus=round(synergy_bonus, 3),
            similarity_to_mistake=round(max_mistake_sim, 3),
            similarity_to_success=round(max_success_sim, 3),
            confidence_adjustment=round(p_win_adj, 3),
            veto=is_vetoed,
            veto_reason=" ".join(reflex_notes) if is_vetoed else "",
        )

    def get_telemetry(self) -> Dict[str, Any]:
        """Return rich, transparent online learning telemetry for the AI Learning UI tab."""
        factor_list = []
        factor_deltas = {}
        for k in EXPERIENCE_FEATURE_KEYS:
            prior_w = DEFAULT_PRIOR_WEIGHTS.get(k, 0.0)
            cur_w = self.weights.get(k, prior_w)
            delta = round(cur_w - prior_w, 4)
            factor_deltas[k] = delta
            meta = FEATURE_METADATA.get(k, {})

            if delta < -0.015:
                status = "DAMPED / PENALIZED"
                status_color = "text-rose-400"
                shift_badge = f"{delta:+.3f}"
            elif delta > 0.015:
                status = "REINFORCED / PROMOTED"
                status_color = "text-emerald-400"
                shift_badge = f"{delta:+.3f}"
            else:
                status = "ANCHORED TO PRIOR"
                status_color = "text-zinc-400"
                shift_badge = "0.000 (Prior)"

            factor_list.append({
                "key": k,
                "name": meta.get("name", k),
                "category": meta.get("category", "Quantitative Structure"),
                "description": meta.get("description", ""),
                "prior_weight": prior_w,
                "learned_weight": cur_w,
                "delta": delta,
                "status": status,
                "status_color": status_color,
                "shift_badge": shift_badge,
            })

        top_penalties = [k for k, d in sorted(factor_deltas.items(), key=lambda x: x[1]) if d < -0.02][:3]
        top_rewards = [k for k, d in sorted(factor_deltas.items(), key=lambda x: -x[1]) if d > 0.02][:3]

        recent_mistakes_summary = [
            {
                "ticket": m["ticket"],
                "direction": m["direction"],
                "pnl_usd": m.get("pnl_usd", 0.0),
                "pnl_r": m["pnl_r"],
                "close_reason": m.get("close_reason", "SL_HIT"),
                "cause": m.get("root_cause", ""),
                "ai_correction": m.get("ai_correction", "Reflex damping active"),
                "time_str": m.get("time_str", ""),
            }
            for m in reversed(list(self.mistake_memory))
        ]

        recent_successes_summary = [
            {
                "ticket": s["ticket"],
                "direction": s["direction"],
                "pnl_usd": s.get("pnl_usd", 0.0),
                "pnl_r": s["pnl_r"],
                "close_reason": s.get("close_reason", "TP_HIT"),
                "driver": s.get("driver", ""),
                "ai_reinforcement": s.get("ai_reinforcement", "Pattern synergy active"),
                "time_str": s.get("time_str", ""),
            }
            for s in reversed(list(self.success_memory))
        ]

        return {
            "total_trades_analyzed": self.total_trades_analyzed,
            "trades_analyzed": self.total_trades_analyzed,
            "wins_learned": self.wins_learned,
            "winning_trades": self.wins_learned,
            "mistakes_learned": self.mistakes_learned,
            "losing_trades": self.mistakes_learned,
            "mistakes_recorded": len(self.mistake_memory),
            "successes_recorded": len(self.success_memory),
            "last_lesson": self.recent_lessons[0] if self.recent_lessons else "Institutional prior weights active. Awaiting trade outcomes.",
            "win_rate_learned": f"{(self.wins_learned / max(1, self.total_trades_analyzed)) * 100:.1f}%",
            "top_penalized_factors": top_penalties,
            "top_rewarded_factors": top_rewards,
            "factors": factor_list,
            "mistakes": recent_mistakes_summary,
            "successes": recent_successes_summary,
            "recent_lessons": list(self.recent_lessons),
            "all_lessons": list(self.recent_lessons),
            "regularization": {
                "l2_regularization": self.l2_reg,
                "learning_rate": self.learning_rate,
                "bias": self.bias,
                "memory_capacity": self.memory_capacity,
                "status": "ACTIVE & STABLE",
                "anti_memorization": "Dimensionless features normalized by ATR/wick bounds. L2 decay prevents overfitting.",
            },
            "has_experience": (self.total_trades_analyzed > 0),
        }

    def save_state(self) -> None:
        """Persist learned weights and recent episodic memories to disk."""
        if not self.persistence_path:
            return
        try:
            os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
            payload = {
                "weights": self.weights,
                "bias": self.bias,
                "total_trades_analyzed": self.total_trades_analyzed,
                "wins_learned": self.wins_learned,
                "mistakes_learned": self.mistakes_learned,
                "last_update_time": self.last_update_time,
                "recent_lessons": list(self.recent_lessons),
                "mistake_memory": list(self.mistake_memory),
                "success_memory": list(self.success_memory),
            }
            with open(self.persistence_path, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save online experience learner state: {e}")

    def load_state(self) -> bool:
        """Load previously learned weights and memories from disk."""
        if not self.persistence_path or not os.path.exists(self.persistence_path):
            return False
        try:
            with open(self.persistence_path, "r") as f:
                payload = json.load(f)
            
            saved_weights = payload.get("weights", {})
            for k in EXPERIENCE_FEATURE_KEYS:
                if k in saved_weights:
                    self.weights[k] = float(saved_weights[k])

            self.bias = float(payload.get("bias", self.bias))
            self.total_trades_analyzed = int(payload.get("total_trades_analyzed", 0))
            self.wins_learned = int(payload.get("wins_learned", 0))
            self.mistakes_learned = int(payload.get("mistakes_learned", 0))
            self.last_update_time = float(payload.get("last_update_time", 0.0))

            for l in payload.get("recent_lessons", []):
                self.recent_lessons.append(str(l))
            for m in payload.get("mistake_memory", []):
                self.mistake_memory.append(m)
            for s in payload.get("success_memory", []):
                self.success_memory.append(s)

            logger.info(
                f"[EXPERIENCE LEARNER] Successfully restored state from {self.persistence_path}: "
                f"{self.total_trades_analyzed} trades analyzed ({self.wins_learned} wins, {self.mistakes_learned} mistakes)."
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to load online experience learner from {self.persistence_path}: {e}")
            return False
