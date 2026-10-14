"""
Tests for DynamicExitManager (V2).
Verifies:
1. Dynamic volatility-aware TP computation.
2. Breathing-room ratchet behavior (does not trigger at premature +0.65R noise).
3. Monotonic stop advancement (never loosens).
"""

import pytest
from strategy.v2.exit_manager import DynamicExitManager


def test_dynamic_tp_calculation():
    mgr = DynamicExitManager(base_target_r=1.50)
    # Entry: 1300, SL: 1290 (Risk = 10 points)
    plan_buy = mgr.compute_exit_plan(entry_price=1300.0, initial_sl=1290.0, direction="BUY", atr_5m=2.0)
    # Target R should be within [1.30, 1.85]
    assert 1.30 <= plan_buy.target_r <= 1.85
    assert plan_buy.suggested_tp > 1300.0
    assert plan_buy.suggested_sl == 1290.0

    plan_sell = mgr.compute_exit_plan(entry_price=1300.0, initial_sl=1310.0, direction="SELL", atr_5m=2.0)
    assert plan_sell.suggested_tp < 1300.0
    assert plan_sell.suggested_sl == 1310.0


def test_ratchet_breathing_room():
    mgr = DynamicExitManager(be_trigger_r=0.90, be_lock_r=0.35)
    entry_p = 1300.0
    sl_p = 1290.0
    risk = 10.0

    # At +0.65R (premature legacy trigger): ratchet should NOT trigger yet
    stage, new_sl = mgr.evaluate_ratchet(0.65, 0, entry_p, sl_p, risk, "BUY")
    assert stage == 0
    assert new_sl == sl_p

    # At +0.90R: ratchet triggers stage 1, locking +0.35R ($3.50 -> 1303.50)
    stage, new_sl = mgr.evaluate_ratchet(0.92, 0, entry_p, sl_p, risk, "BUY")
    assert stage == 1
    assert new_sl == 1303.50

    # Never loosens: if price drops to 0.70R, stage and stop remain locked
    stage2, new_sl2 = mgr.evaluate_ratchet(0.70, stage, entry_p, new_sl, risk, "BUY")
    assert stage2 == 1
    assert new_sl2 == 1303.50

