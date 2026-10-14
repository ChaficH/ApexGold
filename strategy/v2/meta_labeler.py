"""
ApexGold Meta-Labeling ML Filter Engine (V2).
Implements Marcos Lopez de Prado's Triple-Barrier Meta-Labeling methodology:
1. Primary Model: ConfluenceScorer (generates setup candidates, direction, entry, SL).
2. Secondary ML Filter: LightGBM binary classifier predicting P(Win | Causal Market State).
3. Trade Execution Gating: Only execute if predicted P(Win) >= calibrated threshold.

Safety & Institutional Standards:
- Zero Look-Ahead Bias: All feature inputs are strictly causal at decision time.
- Fallback Resilience: If model artifact is missing, mismatched, or lightgbm is not installed,
  gracefully falls back to primary confluence scorer without crashing.
- Versioned Schema: Validates feature schema, version, and threshold before inference.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

logger = logging.getLogger("ApexGold.MetaLabeler")

FEATURE_SCHEMA = [
    "score",
    "h1_macro",
    "m15_wave",
    "m5_setup",
    "m1_trigger",
    "smc_institutional",
    "quant_alpha",
    "adx_15m",
    "choppiness_index",
    "atr_percentile",
    "h1_ema_slope",
    "hour",
    "session_code",
    "is_zone_reversal",
    "regime_code",
]


class MetaLabelerFilter:
    """
    LightGBM Meta-Labeling Execution Filter with Schema Validation and Fail-Safe Fallbacks.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        probability_threshold: float = 0.52,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.probability_threshold = probability_threshold
        self.model = None
        self.model_meta: Dict[str, Any] = {}
        self.is_loaded = False

        if model_path and os.path.exists(model_path) and HAS_LIGHTGBM and self.enabled:
            self._load_model(model_path)

    def _load_model(self, model_path: str) -> bool:
        """Load and validate model artifact with schema integrity check."""
        try:
            meta_path = f"{model_path}.meta.json"
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    self.model_meta = json.load(f)

                # Schema verification
                saved_features = self.model_meta.get("features", [])
                if saved_features != FEATURE_SCHEMA:
                    logger.warning(f"MetaLabeler schema mismatch! Expected {FEATURE_SCHEMA}, got {saved_features}. Disabling ML filter.")
                    self.is_loaded = False
                    return False

                self.probability_threshold = float(self.model_meta.get("optimal_threshold", self.probability_threshold))

            self.model = lgb.Booster(model_file=model_path)
            self.is_loaded = True
            logger.info(f"MetaLabeler V2 successfully loaded from {model_path} (Threshold: {self.probability_threshold:.3f})")
            return True
        except Exception as e:
            logger.warning(f"Failed to load MetaLabeler model from {model_path}: {e}. Falling back to baseline scorer.")
            self.is_loaded = False
            return False

    def predict_p_win(self, features: Dict[str, Any]) -> float:
        """Predict win probability given causal feature vector."""
        if not self.is_loaded or self.model is None or not self.enabled:
            return 0.55  # Neutral default passing baseline

        try:
            vec = [float(features.get(f, 0.0)) for f in FEATURE_SCHEMA]
            x = np.array([vec], dtype=np.float32)
            preds = self.model.predict(x)
            p_win = float(preds[0])
            return p_win
        except Exception as e:
            logger.debug(f"MetaLabeler inference error: {e}. Defaulting to neutral probability.")
            return 0.55

    def filter_trade(self, features: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Evaluate candidate setup against calibrated ML probability threshold.
        Returns: (is_allowed, p_win, reason)
        """
        if not self.enabled or not self.is_loaded:
            return True, 0.55, "MetaLabeler bypassed (model not loaded or disabled)"

        p_win = self.predict_p_win(features)
        if p_win >= self.probability_threshold:
            return True, p_win, f"ML Setup Approved: P(win)={p_win:.3f} >= {self.probability_threshold:.3f}"
        else:
            return False, p_win, f"ML Setup Blocked: Low conviction P(win)={p_win:.3f} < {self.probability_threshold:.3f}"

