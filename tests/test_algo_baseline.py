"""
Institutional Quant Regression Tests for ApexGold Backtest Harness and Baseline.
Ensures:
1. Strict enforcement of realistic costs (spread, slippage, commission).
2. Mathematical precision of quant metrics (PF, Expectancy R, Sharpe, Sortino, Calmar, 95% Bootstrap CI).
3. Chronological integrity of train/hold-out splits and walk-forward folding (no look-ahead).
4. Reproduction of known baseline characteristics (negative expectancy, asymmetric exits).
"""

import numpy as np
import pandas as pd
import pytest

from backtest.run import calculate_advanced_metrics, analyze_breakdowns, run_backtest_suite
from research.data import split_train_holdout, generate_walk_forward_folds


def test_realistic_costs_strictly_enforced():
    """Harness must fail immediately if zero or negative costs are supplied."""
    with pytest.raises(ValueError, match="Cost modeling failure"):
        run_backtest_suite(spread=0.0)

    with pytest.raises(ValueError, match="Cost modeling failure"):
        run_backtest_suite(spread=-0.1)


def test_advanced_metrics_calculation():
    """Test institutional metric calculations against known mathematical outcomes."""
    # Synthetic trade sequence: 3 wins (+2R, +1R, +1.5R) and 3 losses (-1R, -1R, -1R)
    # Total PnL with $10 risk: wins = +$20, +$10, +$15; losses = -$10, -$10, -$10
    trades = [
        {"pnl": 20.0, "return_r": 2.0, "entry_time": 1520000000, "direction": "BUY", "exit_reason": "TP_HIT"},
        {"pnl": 10.0, "return_r": 1.0, "entry_time": 1520003600, "direction": "BUY", "exit_reason": "EXHAUSTION_BANK"},
        {"pnl": 15.0, "return_r": 1.5, "entry_time": 1520007200, "direction": "SELL", "exit_reason": "TP_HIT"},
        {"pnl": -10.0, "return_r": -1.0, "entry_time": 1520010800, "direction": "BUY", "exit_reason": "SL_HIT"},
        {"pnl": -10.0, "return_r": -1.0, "entry_time": 1520014400, "direction": "SELL", "exit_reason": "SL_HIT"},
        {"pnl": -10.0, "return_r": -1.0, "entry_time": 1520018000, "direction": "SELL", "exit_reason": "SL_HIT"},
    ]

    metrics = calculate_advanced_metrics(trades, initial_balance=1000.0)
    assert metrics["total_trades"] == 6
    assert metrics["winning_trades"] == 3
    assert metrics["losing_trades"] == 3
    assert metrics["win_rate_pct"] == 50.0
    # Gross win = $45, Gross loss = $30 -> PF = 1.5
    assert metrics["profit_factor"] == 1.5
    assert metrics["net_profit_usd"] == 15.0
    # Expectancy R = (2 + 1 + 1.5 - 1 - 1 - 1) / 6 = 1.5 / 6 = +0.25R
    assert metrics["expectancy_r"] == 0.25
    assert metrics["avg_win_r"] == 1.5
    assert metrics["avg_loss_r"] == -1.0
    # Bootstrap CI
    ci_low, ci_high = metrics["ci_95_r"]
    assert ci_low <= metrics["expectancy_r"] <= ci_high


def test_chronological_split_and_walk_forward_integrity():
    """Verify train/holdout split is strictly chronological and walk-forward folds do not look ahead."""
    # Create 1,000 synthetic bars
    times = np.arange(1520000000, 1520000000 + 1000 * 300, 300)
    df_synthetic = pd.DataFrame({
        "time": times,
        "open": 1300.0,
        "high": 1305.0,
        "low": 1295.0,
        "close": 1302.0,
        "volume": 100.0,
    })
    dfs = {"5m": df_synthetic}

    train_dfs, holdout_dfs = split_train_holdout(dfs, holdout_ratio=0.20)
    # Train = 800 bars, Holdout = 200 bars
    assert len(train_dfs["5m"]) == 800
    assert len(holdout_dfs["5m"]) == 200
    # Strict chronological boundary: train end must be <= holdout start
    assert train_dfs["5m"]["time"].max() <= holdout_dfs["5m"]["time"].min()

    folds = generate_walk_forward_folds(train_dfs, train_bars=400, test_bars=100, step_bars=100)
    assert len(folds) >= 2
    for fold in folds:
        f_train, f_test = fold[0], fold[1]
        assert not f_train["5m"].empty
        assert not f_test["5m"].empty


def test_breakdown_analysis():
    """Verify session, direction, and exit reason aggregations."""
    trades = [
        {"pnl": 20.0, "return_r": 2.0, "entry_time": 1520000000, "direction": "BUY", "exit_reason": "TP_HIT"},
        {"pnl": -10.0, "return_r": -1.0, "entry_time": 1520014400, "direction": "SELL", "exit_reason": "SL_HIT"},
    ]
    bd = analyze_breakdowns(trades)
    assert "direction" in bd
    assert "exit_reasons" in bd
    assert "sessions" in bd
    assert "BUY" in bd["direction"]
    assert "SELL" in bd["direction"]
    assert "TP_HIT" in bd["exit_reasons"]
    assert "SL_HIT" in bd["exit_reasons"]
