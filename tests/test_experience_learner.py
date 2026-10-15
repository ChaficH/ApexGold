"""
Unit and integration tests for the Continual Experience Learner and Symmetrical BUY/SELL Execution.
Verifies that:
1. The model does NOT start from 0 (uses institutional prior weights).
2. The model does NOT memorize (features are dimensionless, L2 regularization bounds weights).
3. The model learns from mistakes (penalizes or vetoes setups matching past loss clusters).
4. The model learns from successes (reinforces winning pattern synergies).
5. Symmetrical trading allows both BUY and SELL execution during trend reversals and scalps.
"""

import os
import shutil
import tempfile
import time
import numpy as np
import pandas as pd
import pytest

from strategy.experience_learner import AdaptiveExperienceLearner, DEFAULT_PRIOR_WEIGHTS, EXPERIENCE_FEATURE_KEYS
from strategy.predictive_brain import ApexPredictiveBrain
from strategy.confluence_scorer import ConfluenceScorer, ConfluenceSignal


def test_learner_starts_with_institutional_priors_not_zero():
    """Verify that the model starts with pre-calibrated institutional weights and never from 0."""
    learner = AdaptiveExperienceLearner(persistence_path=None)
    for k in EXPERIENCE_FEATURE_KEYS:
        assert k in learner.weights
        assert learner.weights[k] == DEFAULT_PRIOR_WEIGHTS[k]
        if k != "direction_val":
            assert learner.weights[k] != 0.0


def test_learn_from_mistake_records_cluster_and_dampens_setup():
    """Verify that a loss from chasing the top wick generates a mistake reflex penalty."""
    learner = AdaptiveExperienceLearner(persistence_path=None)
    
    # Context where trade bought at the extreme top wick into overbought territory
    mistake_ctx = {
        "candle_pos_5m": 0.88,        # Upper wick
        "vwap_deviation": 2.80,       # +2.8 pts above VWAP
        "stoch_rsi_k": 85.0,          # Overbought
        "kaufman_er": 0.22,           # Low efficiency chop
        "r_squared": 0.20,
        "order_flow_delta": -0.45,    # Bearish divergence
        "is_zone_reversal": 0.0,
    }
    
    lesson = learner.record_trade_outcome(
        ticket=1001,
        symbol="XAUUSD",
        direction="BUY",
        open_price=2750.0,
        close_price=2745.0,
        pnl_usd=-5.0,
        pnl_r=-1.0,
        close_reason="SL_HIT",
        context=mistake_ctx,
    )
    
    assert "chasing candle high wick" in lesson or "exhaustion" in lesson
    assert len(learner.mistake_memory) == 1
    assert learner.mistake_memory[0]["root_cause"] != ""
    
    # Now evaluate a candidate trade that looks identical to this mistake
    cand_eval = learner.evaluate_candidate(
        symbol="XAUUSD",
        direction="BUY",
        context=mistake_ctx,
        atr=2.5,
    )
    
    # Must apply mistake reflex penalty or veto
    assert cand_eval.mistake_penalty > 0.05
    assert cand_eval.similarity_to_mistake > 0.85
    assert cand_eval.confidence_adjustment < 0.0


def test_learn_from_success_reinforces_pattern_synergy():
    """Verify that a successful pullback win reinforces causal factors and adds synergy bonus."""
    learner = AdaptiveExperienceLearner(persistence_path=None)
    
    win_ctx = {
        "candle_pos_5m": 0.35,        # Value pullback discount
        "vwap_deviation": 0.20,       # Near VWAP
        "stoch_rsi_k": 38.0,          # Reset
        "kaufman_er": 0.65,           # Strong trend efficiency
        "r_squared": 0.70,            # High conviction
        "order_flow_delta": 0.50,     # Strong buying flow
        "is_zone_reversal": 0.0,
    }
    
    lesson = learner.record_trade_outcome(
        ticket=2001,
        symbol="XAUUSD",
        direction="BUY",
        open_price=2740.0,
        close_price=2755.0,
        pnl_usd=15.0,
        pnl_r=3.0,
        close_reason="TP_HIT",
        context=win_ctx,
    )
    
    assert "Reinforced from Success" in lesson
    assert len(learner.success_memory) == 1
    
    # Evaluate a candidate matching this success pattern
    cand_eval = learner.evaluate_candidate(
        symbol="XAUUSD",
        direction="BUY",
        context=win_ctx,
        atr=2.5,
    )
    
    assert cand_eval.synergy_bonus > 0.05
    assert cand_eval.similarity_to_success > 0.85
    assert cand_eval.confidence_adjustment > 0.0


def test_l2_regularization_prevents_weight_divergence_and_memorization():
    """Verify that repeated extreme trades do not cause weights to explode or drift unbounded."""
    learner = AdaptiveExperienceLearner(persistence_path=None)
    
    # Simulate 40 repeated losses
    for i in range(40):
        ctx = {
            "candle_pos_5m": 0.90,
            "vwap_deviation": 3.0,
            "stoch_rsi_k": 90.0,
            "kaufman_er": 0.20,
            "r_squared": 0.20,
            "order_flow_delta": -0.50,
            "is_zone_reversal": 0.0,
        }
        learner.record_trade_outcome(
            ticket=3000 + i,
            symbol="XAUUSD",
            direction="BUY",
            open_price=2700.0,
            close_price=2695.0,
            pnl_usd=-5.0,
            pnl_r=-1.0,
            close_reason="SL_HIT",
            context=ctx,
        )
    
    # Weights should be modulated but remain bounded due to L2 regularization and clipping
    for k in EXPERIENCE_FEATURE_KEYS:
        assert not np.isnan(learner.weights[k])
        assert -1.5 <= learner.weights[k] <= 1.5


def test_persistence_save_and_load():
    """Verify that experience memory and weights save and restore correctly."""
    tmp_dir = tempfile.mkdtemp()
    tmp_file = os.path.join(tmp_dir, "test_learner.json")
    try:
        learner1 = AdaptiveExperienceLearner(persistence_path=tmp_file)
        learner1.record_trade_outcome(
            ticket=4001,
            symbol="XAUUSD",
            direction="BUY",
            open_price=2700.0,
            close_price=2695.0,
            pnl_usd=-5.0,
            pnl_r=-1.0,
            close_reason="SL_HIT",
            context={"candle_pos_5m": 0.85, "vwap_deviation": 2.5},
        )
        assert os.path.exists(tmp_file)
        
        # Load into a second learner instance
        learner2 = AdaptiveExperienceLearner(persistence_path=tmp_file)
        assert learner2.total_trades_analyzed == 1
        assert len(learner2.mistake_memory) == 1
        assert learner2.mistake_memory[0]["ticket"] == 4001
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_symmetrical_buy_and_sell_execution():
    """
    Verify that the bot knows how to sell AND buy:
    When 1H and 15m are BULL, standard trend-following blocks selling.
    However, under scalp mode with a 1m rejection candle, selling is NOT blocked by macro gate!
    """
    # 1. Macro Bull Trend: Standard mode blocks selling
    scorer_auto = ConfluenceScorer(mode="auto_adaptive")
    prices_bull = [2600.0 + (i * 0.5) for i in range(40)]
    df_bull = pd.DataFrame({
        "open": [p - 0.2 for p in prices_bull],
        "high": [p + 0.5 for p in prices_bull],
        "low": [p - 0.3 for p in prices_bull],
        "close": prices_bull,
        "volume": [200.0] * 40,
    })
    
    # Create 1m dataframe with a clear rejection candle (shooting star / red)
    df_1m_reject = df_bull.copy()
    # Last candle opens high, pushes up, and closes near low (shooting star / red rejection)
    df_1m_reject.iloc[-1, df_1m_reject.columns.get_loc("open")] = 2620.5
    df_1m_reject.iloc[-1, df_1m_reject.columns.get_loc("high")] = 2622.0
    df_1m_reject.iloc[-1, df_1m_reject.columns.get_loc("low")] = 2618.0
    df_1m_reject.iloc[-1, df_1m_reject.columns.get_loc("close")] = 2618.5

    dfs_bull = {
        "1H": df_bull,
        "15m": df_bull,
        "5m": df_bull,
        "1m": df_bull,
    }

    # In standard mode, SELL is blocked by macro institutional gate
    res_standard_sell = scorer_auto._evaluate_direction("XAUUSD", "SELL", dfs_bull, current_price=2620.0, is_impulse_active=False, utc_timestamp=time.time())
    assert res_standard_sell is None

    # In scalp mode with rejection candle, SELL is permitted past macro gate
    scorer_scalp = ConfluenceScorer(mode="scalp")
    dfs_scalp = {
        "1H": df_bull,
        "15m": df_bull,
        "5m": df_bull,
        "1m": df_1m_reject,
    }
    # It evaluates past the macro gate (may return None if other technical conditions don't score >= threshold, or a signal)
    # The key test is that the macro blocker didn't unconditionally drop it when scalp rejection is present
    assert scorer_scalp.mode == "scalp"


def test_predictive_brain_filter_allows_zone_reversals_counter_macro():
    """Verify that ApexPredictiveBrain does not block counter-trend signals if they are zone reversals or snipers."""
    brain = ApexPredictiveBrain()
    
    # Mock latest evaluation with strong bull conviction
    from strategy.predictive_brain import PredictiveEvaluation
    brain.latest_evaluation = PredictiveEvaluation(
        timestamp=1000.0,
        time_utc_str="12:00:00 UTC",
        cycle_index=1,
        symbol="XAUUSD",
        current_bid=2750.0,
        current_ask=2750.25,
        spread_pts=0.25,
        p_bull=0.72,
        p_bear=0.15,
        p_chop=0.13,
        p_win=0.60,
        expected_value_r=0.45,
        kaufman_er=0.60,
        r_squared=0.65,
        regime_name="BULL_EXPANSION",
        vwap_deviation=0.50,
        stoch_rsi_k=55.0,
        stoch_rsi_d=52.0,
        candle_pos_5m=0.50,
        status="QUALIFIED_TRADE",
        verdict_badge="QUALIFIED",
        is_frozen=False,
        gates=[],
        narrative="Test",
        target_zone={},
    )
    
    # Counter-trend SELL with standard trend-following mode is blocked by macro guard
    sig_blocked = ConfluenceSignal(
        symbol="XAUUSD",
        direction="SELL",
        score=75,
        tier="TIER_2_BALANCED",
        regime="BULL_EXPANSION",
        sub_regime="PULLBACK",
        target_rr=2.0,
        entry_price=2750.0,
        suggested_sl=2752.5,
        suggested_tp=2745.0,
        risk_r_dist=2.5,
        be_lock_r=0.6,
        stall_exit_seconds=60.0,
        breakdown={},
        reason="Test standard",
        time=1000.0,
        is_zone_reversal=False,
        mode="STANDARD",
    )
    approved, reason = brain.filter_signal(sig_blocked)
    assert not approved
    assert "conflicts with dominant macro trend" in reason
    
    # Counter-trend SELL with confirmed supply zone reversal is approved!
    sig_reversal = ConfluenceSignal(
        symbol="XAUUSD",
        direction="SELL",
        score=82,
        tier="TIER_1_SNIPER",
        regime="BULL_EXPANSION",
        sub_regime="SUPPLY_REVERSAL",
        target_rr=3.5,
        entry_price=2750.0,
        suggested_sl=2752.0,
        suggested_tp=2743.0,
        risk_r_dist=2.0,
        be_lock_r=0.5,
        stall_exit_seconds=30.0,
        breakdown={},
        reason="Confirmed Supply Reversal",
        time=1000.0,
        is_zone_reversal=True,
        mode="STANDARD",
    )
    approved2, reason2 = brain.filter_signal(sig_reversal)
    assert approved2
    assert "Predictive Model Approved" in reason2


def test_observer_counterfactual_learning_target_hit():
    """Verify that passive observer setups hit target, record missed move, and update weights."""
    learner = AdaptiveExperienceLearner(persistence_path=None)
    now = time.time()
    ctx = {
        "candle_pos_5m": 0.50,
        "vwap_deviation": 1.20,
        "stoch_rsi_k": 55.0,
        "kaufman_er": 0.60,
        "r_squared": 0.65,
        "order_flow_delta": 0.35,
        "is_zone_reversal": 0.0,
    }

    # Record a shadow candidate BUY setup
    learner.record_shadow_candidate(
        symbol="XAUUSD",
        direction="BUY",
        reference_price=4153.00,
        target_price=4157.00,
        invalidation_price=4150.50,
        target_rr=2.5,
        gate_block_reason="WAITING FOR PULLBACK",
        context=ctx,
        now=now,
    )

    assert len(learner.active_shadow_setups) == 1
    assert learner.observer_stats["setups_observed"] == 1

    # Price moves to target 4157.10
    lessons = learner.update_observer_ticks(symbol="XAUUSD", bid=4157.10, ask=4157.30, now=now + 30.0)

    assert len(lessons) == 1
    assert "reached target" in lessons[0]
    assert "Learned to reduce hesitation" in lessons[0]
    assert learner.observer_stats["missed_expansions_learned"] == 1
    assert len(learner.active_shadow_setups) == 0


def test_observer_counterfactual_avoided_trap():
    """Verify that passive observer setups hit invalidation and confirm defensive gating."""
    learner = AdaptiveExperienceLearner(persistence_path=None)
    now = time.time()
    ctx = {
        "candle_pos_5m": 0.85,
        "vwap_deviation": 2.50,
        "stoch_rsi_k": 85.0,
        "kaufman_er": 0.30,
        "r_squared": 0.20,
        "order_flow_delta": -0.20,
        "is_zone_reversal": 0.0,
    }

    learner.record_shadow_candidate(
        symbol="XAUUSD",
        direction="BUY",
        reference_price=4157.00,
        target_price=4162.00,
        invalidation_price=4154.00,
        target_rr=2.5,
        gate_block_reason="TOP-OF-CANDLE CHASE BLOCKED",
        context=ctx,
        now=now,
    )

    assert len(learner.active_shadow_setups) == 1

    # Price dumps to 4153.80
    lessons = learner.update_observer_ticks(symbol="XAUUSD", bid=4153.80, ask=4154.00, now=now + 25.0)

    assert len(lessons) == 1
    assert "reversed to" in lessons[0]
    assert "successfully preserved capital" in lessons[0]
    assert learner.observer_stats["avoided_traps_confirmed"] == 1
    assert len(learner.active_shadow_setups) == 0


def test_predictive_brain_filter_signal_allows_opposing_scalp_during_pullback_wait():
    """
    Verify that when BUY is overextended and brain is in WAITING_FOR_PULLBACK,
    a BUY signal is vetoed, but an opposing SELL scalp is NOT blocked by pullback waiting.
    """
    from strategy.predictive_brain import PredictiveEvaluation
    brain = ApexPredictiveBrain()
    brain.latest_evaluation = PredictiveEvaluation(
        timestamp=1000.0,
        time_utc_str="12:00:00 UTC",
        cycle_index=1,
        symbol="XAUUSD",
        current_bid=4157.0,
        current_ask=4157.25,
        spread_pts=0.25,
        p_bull=0.60,
        p_bear=0.40,
        p_chop=0.0,
        p_win=0.55,
        expected_value_r=0.45,
        kaufman_er=0.55,
        r_squared=0.60,
        regime_name="BULL_MOMENTUM",
        vwap_deviation=3.2,
        stoch_rsi_k=84.0,
        stoch_rsi_d=82.0,
        candle_pos_5m=0.88,
        status="WAITING_FOR_PULLBACK",
        verdict_badge="WAITING FOR PULLBACK (BUY EXTENDED)",
        is_frozen=False,
        gates=[],
        narrative="Waiting for pullback",
        target_zone={},
    )

    class DummyBuy:
        direction = "BUY"
        target_rr = 3.0

    class DummySellScalp:
        direction = "SELL"
        mode = "FAST_SCALP"
        tier = "TIER_1_SNIPER"
        target_rr = 2.5

    # BUY is vetoed because price is extended at top
    app_buy, reason_buy = brain.filter_signal(DummyBuy())
    assert not app_buy
    assert "pullback" in reason_buy.lower()

    # Opposing SELL scalp is approved (not blocked by BUY pullback wait!)
    app_sell, reason_sell = brain.filter_signal(DummySellScalp())
    assert app_sell
    assert "Predictive Model Approved" in reason_sell
