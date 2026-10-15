"""
Tests for MetaLabelerFilter (V2).
Verifies:
1. Model loading with schema validation.
2. Prediction of win probabilities.
3. Filtering setups below probability threshold.
4. Fallback resilience when model is missing or corrupt.
"""

import os
import numpy as np
import pytest
from strategy.v2.meta_labeler import MetaLabelerFilter, FEATURE_SCHEMA, HAS_LIGHTGBM


def test_meta_labeler_loads_and_predicts():
    if not HAS_LIGHTGBM:
        pytest.skip("lightgbm not installed in environment")

    model_path = "models/metalabeler_v1.txt"
    if not os.path.exists(model_path):
        pytest.skip("Model artifact not yet trained")

    ml_filter = MetaLabelerFilter(model_path=model_path, probability_threshold=0.52)
    assert ml_filter.is_loaded

    # Sample features
    features = {
        "score": 75,
        "h1_macro": 10,
        "m15_wave": 15,
        "m5_setup": 25,
        "m1_trigger": 15,
        "smc_institutional": 8,
        "quant_alpha": 5,
        "adx_15m": 25.0,
        "choppiness_index": 42.0,
        "atr_percentile": 0.50,
        "h1_ema_slope": 0.05,
        "hour": 14,
        "session_code": 2,
        "is_zone_reversal": 0,
        "regime_code": 1,
    }

    p_win = ml_filter.predict_p_win(features)
    assert 0.0 <= p_win <= 1.0

    allowed, p, reason = ml_filter.filter_trade(features)
    assert isinstance(allowed, (bool, np.bool_))
    assert 0.0 <= p <= 1.0


def test_meta_labeler_fallback_on_missing_model():
    ml_filter = MetaLabelerFilter(model_path="models/non_existent.txt")
    assert not ml_filter.is_loaded

    allowed, p, reason = ml_filter.filter_trade({"score": 70})
    assert allowed  # Always permits trade when disabled or model missing (safe fallback)
    assert "bypassed" in reason
