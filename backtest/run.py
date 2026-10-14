"""
ApexGold Quantitative Research & Backtest Runner CLI.
Usage:
    python -m backtest.run --from 2018-03-01 --to 2018-04-30 --mode balanced --report
    python -m backtest.run --from 2018-03-01 --to 2018-04-30 --mode balanced --walk-forward --report

Produces:
- Trades CSV and equity curve
- Profit Factor, Expectancy (R), Win Rate, Avg Win/Loss R, Max Drawdown
- Sharpe, Sortino, Calmar ratios
- Exit reason breakdown, Long vs Short split, Session & Regime breakdown
- 95% Bootstrap Confidence Interval on Expectancy
- QuantStats HTML tear sheets (when --report is specified)
"""

import argparse
import datetime
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine, BacktestResult, BacktestTrade
from research.data import (
    fetch_2018_baseline_data,
    fetch_recent_regime_data,
    split_train_holdout,
    generate_walk_forward_folds,
    RESEARCH_DATA_DIR,
)
from strategy.confluence_scorer import ConfluenceScorer

try:
    import quantstats as qs
    HAS_QUANTSTATS = True
except ImportError:
    HAS_QUANTSTATS = False

logger = logging.getLogger("ApexGold.BacktestRun")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def calculate_advanced_metrics(trades: List[Dict[str, Any]], initial_balance: float = 1000.0) -> Dict[str, Any]:
    """Calculate institutional quant metrics on simulated trades."""
    if not trades:
        return {
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "expectancy_r": 0.0,
            "avg_win_r": 0.0,
            "avg_loss_r": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "ci_95_r": (0.0, 0.0),
        }

    df_t = pd.DataFrame(trades)
    pnls = df_t["pnl"].values
    returns_r = df_t["return_r"].values
    
    total_trades = len(df_t)
    wins = df_t[df_t["pnl"] > 0]
    losses = df_t[df_t["pnl"] < 0]
    
    win_rate = (len(wins) / total_trades) * 100.0 if total_trades > 0 else 0.0
    gross_win = wins["pnl"].sum() if not wins.empty else 0.0
    gross_loss = abs(losses["pnl"].sum()) if not losses.empty else 0.0
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)
    
    avg_win_r = round(wins["return_r"].mean(), 2) if not wins.empty else 0.0
    avg_loss_r = round(losses["return_r"].mean(), 2) if not losses.empty else 0.0
    expectancy_r = round(returns_r.mean(), 3) if total_trades > 0 else 0.0

    # Equity curve & Max Drawdown
    cumulative_pnl = np.cumsum(pnls)
    equity = initial_balance + cumulative_pnl
    peaks = np.maximum.accumulate(equity)
    drawdowns_usd = peaks - equity
    drawdowns_pct = (drawdowns_usd / peaks) * 100.0
    max_dd_pct = round(np.max(drawdowns_pct), 2) if len(drawdowns_pct) > 0 else 0.0

    # Sharpe, Sortino, Calmar (annualized assuming 252 trading days, ~5 trades/day)
    trades_per_year = 252 * 5
    if np.std(pnls) > 0:
        sharpe = round((np.mean(pnls) / np.std(pnls)) * np.sqrt(trades_per_year), 2)
    else:
        sharpe = 0.0

    downside_pnls = pnls[pnls < 0]
    if len(downside_pnls) > 0 and np.std(downside_pnls) > 0:
        sortino = round((np.mean(pnls) / np.std(downside_pnls)) * np.sqrt(trades_per_year), 2)
    else:
        sortino = 0.0

    net_profit = cumulative_pnl[-1] if len(cumulative_pnl) > 0 else 0.0
    calmar = round((net_profit / max(1.0, np.max(drawdowns_usd))), 2) if max_dd_pct > 0 else 0.0

    # Bootstrap 95% Confidence Interval on Expectancy (1,000 resamples)
    np.random.seed(42)
    boot_means = []
    n = len(returns_r)
    for _ in range(1000):
        sample = np.random.choice(returns_r, size=n, replace=True)
        boot_means.append(sample.mean())
    ci_lower = round(float(np.percentile(boot_means, 2.5)), 3)
    ci_upper = round(float(np.percentile(boot_means, 97.5)), 3)

    return {
        "total_trades": total_trades,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": pf,
        "net_profit_usd": round(net_profit, 2),
        "expectancy_r": expectancy_r,
        "avg_win_r": avg_win_r,
        "avg_loss_r": avg_loss_r,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown_pct": max_dd_pct,
        "ci_95_r": (ci_lower, ci_upper),
    }


def analyze_breakdowns(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Breakdown trade performance across direction, session, and exit reasons."""
    if not trades:
        return {}

    df = pd.DataFrame(trades)
    
    # Assign trading sessions based on UTC entry hour
    # Asia: 00:00 - 07:00 UTC, London: 07:00 - 13:00 UTC, NY: 13:00 - 21:00 UTC, Overnight: 21:00 - 24:00 UTC
    df["entry_dt"] = pd.to_datetime(df["entry_time"], unit="s", utc=True)
    hours = df["entry_dt"].dt.hour
    df["session"] = np.where(
        hours < 7, "ASIA",
        np.where(hours < 13, "LONDON",
        np.where(hours < 21, "NEW_YORK", "OVERNIGHT"))
    )

    # Direction breakdown
    dir_summary = {}
    for d in ["BUY", "SELL"]:
        sub = df[df["direction"] == d]
        if not sub.empty:
            wins = (sub["pnl"] > 0).sum()
            dir_summary[d] = {
                "trades": len(sub),
                "win_rate_pct": round((wins / len(sub)) * 100.0, 1),
                "pnl_usd": round(sub["pnl"].sum(), 2),
                "expectancy_r": round(sub["return_r"].mean(), 3),
            }

    # Exit reason breakdown
    exit_summary = {}
    for reason, group in df.groupby("exit_reason"):
        exit_summary[reason] = {
            "count": len(group),
            "pct_of_total": round((len(group) / len(df)) * 100.0, 1),
            "avg_r": round(group["return_r"].mean(), 2),
            "pnl_usd": round(group["pnl"].sum(), 2),
        }

    # Session breakdown
    session_summary = {}
    for sess, group in df.groupby("session"):
        session_summary[sess] = {
            "trades": len(group),
            "win_rate_pct": round(((group["pnl"] > 0).sum() / len(group)) * 100.0, 1),
            "pnl_usd": round(group["pnl"].sum(), 2),
            "expectancy_r": round(group["return_r"].mean(), 3),
        }

    return {
        "direction": dir_summary,
        "exit_reasons": exit_summary,
        "sessions": session_summary,
    }


def generate_quantstats_report(trades: List[Dict[str, Any]], output_path: str = "docs/quantstats_tearsheet.html"):
    """Generate professional QuantStats performance tear sheet."""
    if not HAS_QUANTSTATS or not trades:
        logger.warning("QuantStats not available or no trades to report.")
        return

    try:
        df_t = pd.DataFrame(trades)
        df_t["entry_dt"] = pd.to_datetime(df_t["entry_time"], unit="s", utc=True).dt.tz_localize(None)
        daily_pnl = df_t.set_index("entry_dt")["pnl"].resample("D").sum()
        # Convert dollar PnL to daily percentage return based on $1,000 base
        daily_returns = daily_pnl / 1000.0
        daily_returns = daily_returns.fillna(0.0)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        qs.reports.html(
            daily_returns,
            output=output_path,
            title="ApexGold Institutional Strategy Tear Sheet",
            download_filename=output_path,
        )
        logger.info(f"QuantStats tear sheet saved to {output_path}")
    except Exception as e:
        logger.warning(f"QuantStats HTML generation failed: {e}")


def run_backtest_suite(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    mode: str = "balanced",
    walk_forward: bool = False,
    report: bool = False,
    dataset: str = "2018",
    initial_balance: float = 1000.0,
    risk_pct: float = 1.0,
    spread: float = 0.25,
    slippage: float = 0.05,
    commission: float = 7.0,
    scorer_override: Optional[ConfluenceScorer] = None,
) -> Dict[str, Any]:
    """Execute complete backtest suite matching CLI specification."""
    # Enforce realistic cost modeling: fail if zero costs configured
    if spread <= 0 or commission < 0:
        raise ValueError("Cost modeling failure: Spread and commission must be strictly positive!")

    logger.info(f"Loading dataset '{dataset}'...")
    if dataset == "recent":
        dfs = fetch_recent_regime_data()
    else:
        dfs = fetch_2018_baseline_data()

    # Filter date range if specified
    if start_date:
        ts_start = int(datetime.datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc).timestamp())
        dfs = {tf: df[df["time"] >= ts_start].reset_index(drop=True) for tf, df in dfs.items()}
    if end_date:
        ts_end = int(datetime.datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc).timestamp()) + 86400
        dfs = {tf: df[df["time"] <= ts_end].reset_index(drop=True) for tf, df in dfs.items()}

    scorer = scorer_override or ConfluenceScorer(mode=mode)
    engine = BacktestEngine(
        scorer=scorer,
        initial_balance=initial_balance,
        risk_per_trade_pct=risk_pct,
        spread_points=spread,
        slippage_points=slippage,
        commission_per_lot=commission,
    )

    if walk_forward:
        logger.info("Running Walk-Forward cross-validation across rolling folds...")
        train_dfs, holdout_dfs = split_train_holdout(dfs, holdout_ratio=0.20)
        folds = generate_walk_forward_folds(train_dfs)

        fold_results = []
        all_oos_trades = []

        for idx, (f_train, f_test) in enumerate(folds):
            res_fold = engine._simulate("XAUUSD", scorer, f_test.get("1H"), f_test.get("15m"), f_test["5m"], f_test.get("1m"))
            fold_metrics = calculate_advanced_metrics(res_fold.trades, initial_balance=initial_balance)
            fold_results.append({
                "fold": idx + 1,
                "trades": len(res_fold.trades),
                "profit_factor": fold_metrics["profit_factor"],
                "win_rate": fold_metrics["win_rate_pct"],
                "expectancy_r": fold_metrics["expectancy_r"],
                "pnl_usd": fold_metrics["net_profit_usd"],
            })
            all_oos_trades.extend(res_fold.trades)

        aggregate_metrics = calculate_advanced_metrics(all_oos_trades, initial_balance=initial_balance)
        breakdowns = analyze_breakdowns(all_oos_trades)
        profitable_folds = sum(1 for f in fold_results if f["pnl_usd"] > 0)
        fold_pass_rate = round((profitable_folds / len(fold_results)) * 100.0, 1) if fold_results else 0.0

        output = {
            "mode": mode,
            "type": "WALK_FORWARD_VALIDATION",
            "folds_count": len(fold_results),
            "profitable_folds": profitable_folds,
            "fold_pass_rate_pct": fold_pass_rate,
            "aggregate_metrics": aggregate_metrics,
            "fold_details": fold_results,
            "breakdowns": breakdowns,
            "trades": all_oos_trades,
        }
    else:
        logger.info(f"Running full historical simulation (Mode: {mode.upper()})...")
        res = engine._simulate("XAUUSD", scorer, dfs.get("1H"), dfs.get("15m"), dfs["5m"], dfs.get("1m"))
        metrics = calculate_advanced_metrics(res.trades, initial_balance=initial_balance)
        breakdowns = analyze_breakdowns(res.trades)

        output = {
            "mode": mode,
            "type": "FULL_SIMULATION",
            "metrics": metrics,
            "breakdowns": breakdowns,
            "trades": res.trades,
            "equity_curve": res.equity_curve,
        }

    # Save outputs if report requested
    if report:
        os.makedirs("docs", exist_ok=True)
        trades_df = pd.DataFrame(output.get("trades", []))
        if not trades_df.empty:
            trades_df.to_csv("docs/trades.csv", index=False)
            logger.info(f"Saved {len(trades_df)} simulated trades to docs/trades.csv")
            generate_quantstats_report(output.get("trades", []), output_path="docs/quantstats_tearsheet.html")

    return output


def print_summary_report(output: Dict[str, Any]):
    """Pretty-print backtest summary report to stdout."""
    print("\n" + "=" * 80)
    print(f"APEXGOLD QUANTITATIVE BACKTEST REPORT — {output.get('mode', 'BALANCED').upper()}")
    print("=" * 80)

    if output.get("type") == "WALK_FORWARD_VALIDATION":
        agg = output["aggregate_metrics"]
        print(f"Evaluation Type:       Walk-Forward Cross-Validation ({output['folds_count']} folds)")
        print(f"Profitable Folds:      {output['profitable_folds']}/{output['folds_count']} ({output['fold_pass_rate_pct']}%)")
        print(f"Total OOS Trades:      {agg['total_trades']}")
        print(f"Win Rate:              {agg['win_rate_pct']}% ({agg['winning_trades']}W - {agg['losing_trades']}L)")
        print(f"Profit Factor:         {agg['profit_factor']}")
        print(f"Net Profit (USD):      ${agg['net_profit_usd']:.2f}")
        print(f"Expectancy (R/trade):  {agg['expectancy_r']:.3f}R")
        print(f"Bootstrap 95% CI (R):  [{agg['ci_95_r'][0]:.3f}R, {agg['ci_95_r'][1]:.3f}R]")
        print(f"Sharpe Ratio:          {agg['sharpe_ratio']} | Sortino: {agg['sortino_ratio']} | Calmar: {agg['calmar_ratio']}")
        print(f"Max Drawdown:          {agg['max_drawdown_pct']}%")
    else:
        m = output["metrics"]
        print(f"Evaluation Type:       Full Historical Simulation")
        print(f"Total Trades:          {m['total_trades']}")
        print(f"Win Rate:              {m['win_rate_pct']}% ({m['winning_trades']}W - {m['losing_trades']}L)")
        print(f"Profit Factor:         {m['profit_factor']}")
        print(f"Net Profit (USD):      ${m['net_profit_usd']:.2f}")
        print(f"Expectancy (R/trade):  {m['expectancy_r']:.3f}R")
        print(f"Avg Win / Loss (R):    +{m['avg_win_r']:.2f}R / {m['avg_loss_r']:.2f}R")
        print(f"Bootstrap 95% CI (R):  [{m['ci_95_r'][0]:.3f}R, {m['ci_95_r'][1]:.3f}R]")
        print(f"Sharpe Ratio:          {m['sharpe_ratio']} | Sortino: {m['sortino_ratio']} | Calmar: {m['calmar_ratio']}")
        print(f"Max Drawdown:          {m['max_drawdown_pct']}%")

    bk = output.get("breakdowns", {})
    if "direction" in bk:
        print("\n--- Directional Split ---")
        for d, v in bk["direction"].items():
            print(f"  {d:4s}: {v['trades']} trades | WR: {v['win_rate_pct']}% | PnL: ${v['pnl_usd']:.2f} | Exp: {v['expectancy_r']:.3f}R")

    if "exit_reasons" in bk:
        print("\n--- Exit Reason Breakdown ---")
        for r, v in bk["exit_reasons"].items():
            print(f"  {r:16s}: {v['count']:3d} trades ({v['pct_of_total']:4.1f}%) | Avg R: {v['avg_r']:+.2f}R | PnL: ${v['pnl_usd']:+.2f}")

    if "sessions" in bk:
        print("\n--- Session Performance ---")
        for s, v in bk["sessions"].items():
            print(f"  {s:10s}: {v['trades']:3d} trades | WR: {v['win_rate_pct']}% | PnL: ${v['pnl_usd']:+.2f} | Exp: {v['expectancy_r']:.3f}R")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="ApexGold Quantitative Backtest Harness")
    parser.add_argument("--from", dest="start_date", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--to", dest="end_date", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--mode", type=str, default="balanced", choices=["cautious", "balanced", "aggressive", "hyper"])
    parser.add_argument("--walk-forward", action="store_true", help="Run rolling walk-forward cross validation")
    parser.add_argument("--report", action="store_true", help="Export trades.csv and QuantStats HTML tearsheet")
    parser.add_argument("--dataset", type=str, default="2018", choices=["2018", "recent"])
    parser.add_argument("--initial-balance", type=float, default=1000.0)
    parser.add_argument("--risk-pct", type=float, default=1.0)
    parser.add_argument("--spread", type=float, default=0.25)
    parser.add_argument("--slippage", type=float, default=0.05)
    parser.add_argument("--commission", type=float, default=7.0)

    args = parser.parse_args()

    results = run_backtest_suite(
        start_date=args.start_date,
        end_date=args.end_date,
        mode=args.mode,
        walk_forward=args.walk_forward,
        report=args.report,
        dataset=args.dataset,
        initial_balance=args.initial_balance,
        risk_pct=args.risk_pct,
        spread=args.spread,
        slippage=args.slippage,
        commission=args.commission,
    )

    print_summary_report(results)


if __name__ == "__main__":
    main()

