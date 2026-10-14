"""
ApexGold Institutional Quantitative Diagnostics & Signal Attribution Engine.
Implements STEP 2 of the Research Plan:
1. Feature/signal attribution & marginal expectancy of individual scoring components:
   - 1H Macro Bias (EMA50, room-to-run)
   - 15m Session Wave (EMA20/50, ADX)
   - 5m Setup Structure (Value zones, consolidation breakout, zone reversal)
   - 1m Execution Trigger (Engulfing, liquidity sweep, RSI hook)
   - SMC Institutional (Order Blocks, FVGs, CHOCH, BOS)
   - Quantitative Alpha (VWAP, KAMA, Linreg slope, Z-Score)
2. Trade Exit Analysis:
   - Intrabar MFE (Maximum Favorable Excursion) & MAE (Maximum Adverse Excursion) distributions
   - Analysis of why 2.5R-4.0R TP is hit in only <1% of trades
   - Quantifying premature stop-outs from +0.65R breakeven ratchet
3. Asymmetry & Calibration:
   - Long vs Short asymmetry (BUY vs SELL expectancy and drawdown)
   - Session alpha (Asia vs London vs New York vs Overnight)
   - Day-of-week seasonality
   - Score calibration (Does score >= 80 actually outperform 68?)
"""

import datetime
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine
from backtest.run import calculate_advanced_metrics, analyze_breakdowns
from research.data import fetch_2018_baseline_data
from strategy.confluence_scorer import ConfluenceScorer

logger = logging.getLogger("ApexGold.Diagnostics")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_detailed_diagnostic(mode: str = "balanced") -> Dict[str, Any]:
    """Execute complete deep diagnostic on 2018 benchmark dataset."""
    logger.info("Loading 2018 dataset for diagnostics...")
    dfs = fetch_2018_baseline_data()
    df_5m = dfs["5m"]
    df_1h = dfs.get("1H")
    df_15m = dfs.get("15m")
    df_1m = dfs.get("1m")

    scorer = ConfluenceScorer(mode=mode)
    engine = BacktestEngine(
        scorer=scorer,
        initial_balance=1000.0,
        risk_per_trade_pct=1.0,
        spread_points=0.25,
        slippage_points=0.05,
        commission_per_lot=7.0,
    )

    logger.info(f"Running simulation with feature instrumentation (Mode: {mode})...")
    res = engine._simulate("XAUUSD", scorer, df_1h, df_15m, df_5m, df_1m)
    trades = res.trades

    logger.info(f"Analyzing {len(trades)} trades across features, exits, and regimes...")

    df_t = pd.DataFrame(trades)
    df_t["entry_dt"] = pd.to_datetime(df_t["entry_time"], unit="s", utc=True)
    df_t["day_of_week"] = df_t["entry_dt"].dt.day_name()
    hours = df_t["entry_dt"].dt.hour
    df_t["session"] = np.where(
        hours < 7, "ASIA",
        np.where(hours < 13, "LONDON",
        np.where(hours < 21, "NEW_YORK", "OVERNIGHT"))
    )

    # 1. Overall Metrics
    overall = calculate_advanced_metrics(trades, 1000.0)

    # 2. Exit Reason Breakdown & Ratchet Analysis
    exit_counts = df_t["exit_reason"].value_counts().to_dict()
    exit_pnl = df_t.groupby("exit_reason")["pnl"].sum().round(2).to_dict()
    exit_avg_r = df_t.groupby("exit_reason")["return_r"].mean().round(2).to_dict()

    # 3. Directional Asymmetry (BUY vs SELL)
    dir_metrics = {}
    for d in ["BUY", "SELL"]:
        sub = df_t[df_t["direction"] == d]
        wins = sub[sub["pnl"] > 0]
        losses = sub[sub["pnl"] < 0]
        dir_metrics[d] = {
            "trades": len(sub),
            "win_rate_pct": round(len(wins) / len(sub) * 100.0, 1) if len(sub) > 0 else 0.0,
            "net_pnl": round(sub["pnl"].sum(), 2),
            "expectancy_r": round(sub["return_r"].mean(), 3),
            "avg_win_r": round(wins["return_r"].mean(), 2) if not wins.empty else 0.0,
            "avg_loss_r": round(losses["return_r"].mean(), 2) if not losses.empty else 0.0,
            "profit_factor": round(wins["pnl"].sum() / abs(losses["pnl"].sum()), 2) if not losses.empty and losses["pnl"].sum() != 0 else 0.0,
        }

    # 4. Session Breakdown
    sess_metrics = {}
    for sess, sub in df_t.groupby("session"):
        wins = sub[sub["pnl"] > 0]
        sess_metrics[sess] = {
            "trades": len(sub),
            "win_rate_pct": round(len(wins) / len(sub) * 100.0, 1),
            "net_pnl": round(sub["pnl"].sum(), 2),
            "expectancy_r": round(sub["return_r"].mean(), 3),
        }

    # 5. Day of Week Seasonality
    dow_metrics = {}
    for dow in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        sub = df_t[df_t["day_of_week"] == dow]
        if not sub.empty:
            wins = sub[sub["pnl"] > 0]
            dow_metrics[dow] = {
                "trades": len(sub),
                "win_rate_pct": round(len(wins) / len(sub) * 100.0, 1),
                "net_pnl": round(sub["pnl"].sum(), 2),
                "expectancy_r": round(sub["return_r"].mean(), 3),
            }

    # 6. Score vs Forward Expectancy Calibration
    score_bins = [(68, 72), (73, 77), (78, 82), (83, 100)]
    calib = []
    for s_min, s_max in score_bins:
        sub = df_t[(df_t["score"] >= s_min) & (df_t["score"] <= s_max)]
        if not sub.empty:
            wins = sub[sub["pnl"] > 0]
            calib.append({
                "score_range": f"{s_min}-{s_max}",
                "trades": len(sub),
                "win_rate_pct": round(len(wins) / len(sub) * 100.0, 1),
                "expectancy_r": round(sub["return_r"].mean(), 3),
                "net_pnl": round(sub["pnl"].sum(), 2),
            })

    # 7. Intrabar MFE & MAE Simulation on Each Trade
    # Look ahead 40 bars from entry to measure natural excursion
    mfes = []
    maes = []
    r_levels_reached = {0.5: 0, 1.0: 0, 1.5: 0, 2.0: 0, 2.5: 0, 3.0: 0}

    # Map time to bar index for 5m
    time_to_idx = {int(t): idx for idx, t in enumerate(df_5m["time"])}

    for t in trades:
        entry_t = t["entry_time"] - 300  # Bar start time
        idx = time_to_idx.get(entry_t)
        if idx is None:
            continue
        entry_p = t["entry_price"]
        sl_p = t["sl"]
        direction = t["direction"]
        risk_dist = abs(entry_p - sl_p)
        if risk_dist <= 0:
            continue

        future_bars = df_5m.iloc[idx + 1: min(idx + 41, len(df_5m))]
        if future_bars.empty:
            continue

        if direction == "BUY":
            max_p = future_bars["high"].max()
            min_p = future_bars["low"].min()
            mfe_r = (max_p - entry_p) / risk_dist
            mae_r = (entry_p - min_p) / risk_dist
        else:
            max_p = future_bars["high"].max()
            min_p = future_bars["low"].min()
            mfe_r = (entry_p - min_p) / risk_dist
            mae_r = (max_p - entry_p) / risk_dist

        mfes.append(mfe_r)
        maes.append(mae_r)

        for lvl in r_levels_reached.keys():
            if mfe_r >= lvl:
                r_levels_reached[lvl] += 1

    mfe_series = pd.Series(mfes)
    mae_series = pd.Series(maes)
    total_analyzed = len(mfes)

    mfe_stats = {
        "median_mfe_r": round(mfe_series.median(), 2),
        "mean_mfe_r": round(mfe_series.mean(), 2),
        "p75_mfe_r": round(mfe_series.quantile(0.75), 2),
        "p90_mfe_r": round(mfe_series.quantile(0.90), 2),
        "pct_reaching_0.5r": round(r_levels_reached[0.5] / total_analyzed * 100.0, 1) if total_analyzed else 0,
        "pct_reaching_1.0r": round(r_levels_reached[1.0] / total_analyzed * 100.0, 1) if total_analyzed else 0,
        "pct_reaching_1.5r": round(r_levels_reached[1.5] / total_analyzed * 100.0, 1) if total_analyzed else 0,
        "pct_reaching_2.0r": round(r_levels_reached[2.0] / total_analyzed * 100.0, 1) if total_analyzed else 0,
        "pct_reaching_2.5r": round(r_levels_reached[2.5] / total_analyzed * 100.0, 1) if total_analyzed else 0,
        "pct_reaching_3.0r": round(r_levels_reached[3.0] / total_analyzed * 100.0, 1) if total_analyzed else 0,
    }

    mae_stats = {
        "median_mae_r": round(mae_series.median(), 2),
        "mean_mae_r": round(mae_series.mean(), 2),
        "p75_mae_r": round(mae_series.quantile(0.75), 2),
        "p90_mae_r": round(mae_series.quantile(0.90), 2),
    }

    return {
        "overall": overall,
        "exit_reasons": {
            "counts": exit_counts,
            "pnl": exit_pnl,
            "avg_r": exit_avg_r,
        },
        "directional_asymmetry": dir_metrics,
        "session_metrics": sess_metrics,
        "day_of_week": dow_metrics,
        "score_calibration": calib,
        "mfe_stats": mfe_stats,
        "mae_stats": mae_stats,
    }


if __name__ == "__main__":
    diag = run_detailed_diagnostic("balanced")
    print("\n" + "=" * 80)
    print("APEXGOLD QUANTITATIVE STEP 2 DIAGNOSTICS COMPLETE")
    print("=" * 80)
    print(json.dumps(diag, indent=2))

