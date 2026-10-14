"""
Targeted Walk-Forward Gate Evaluation.
"""

import sys
import numpy as np
import pandas as pd
from backtest.engine import BacktestEngine
from backtest.run import calculate_advanced_metrics
from research.data import fetch_2018_baseline_data, split_train_holdout, generate_walk_forward_folds
from strategy.v2.exit_manager import DynamicExitManager
from strategy.v2.strategy_v2 import ApexStrategyV2

dfs = fetch_2018_baseline_data()
train_dfs, holdout_dfs = split_train_holdout(dfs, holdout_ratio=0.20)
folds = generate_walk_forward_folds(train_dfs)

engine = BacktestEngine(
    initial_balance=1000.0,
    risk_per_trade_pct=1.0,
    spread_points=0.25,
    slippage_points=0.05,
    commission_per_lot=7.0,
)

for p_thresh in [0.52, 0.53, 0.54]:
    strat = ApexStrategyV2(
        mode="balanced",
        regime_filter_enabled=True,
        dynamic_exits_enabled=True,
        meta_labeler_enabled=True,
        asia_session_filter_enabled=True,
        min_p_win=p_thresh,
    )
    strat.exit_manager = DynamicExitManager(
        base_target_r=1.45,
        be_trigger_r=0.85,
        be_lock_r=0.40,
        trail_trigger_r=1.20,
        trail_lock_r=0.80,
    )

    all_oos = []
    fold_pnls = []

    for idx, fold_entry in enumerate(folds):
        f_train, f_test, test_start_ts = fold_entry
        res = engine._simulate("XAUUSD", strat, f_test.get("1H"), f_test.get("15m"), f_test["5m"], f_test.get("1m"), start_trade_time=test_start_ts)
        fold_pnl = sum(t["pnl"] for t in res.trades)
        fold_pnls.append(fold_pnl)
        all_oos.extend(res.trades)

    m = calculate_advanced_metrics(all_oos, 1000.0)
    prof_folds = sum(1 for p in fold_pnls if p > 0)
    fold_pct = round((prof_folds / len(folds)) * 100.0, 1)

    print(f"P={p_thresh} -> Trades={m['total_trades']} WR={m['win_rate_pct']}% PF={m['profit_factor']} Exp={m['expectancy_r']:+.3f}R PnL=${m['net_profit_usd']:+.2f} Folds={prof_folds}/{len(folds)} ({fold_pct}%) DD={m['max_drawdown_pct']}% CI={m['ci_95_r']}", flush=True)

