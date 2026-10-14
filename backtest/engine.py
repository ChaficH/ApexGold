"""
ApexGold Multi-Timeframe Historical Backtesting Engine.

Runs the exact 100-Point Confluence Scorer and Adaptive Chameleon Trade Management
across cached multi-timeframe historical data with:
- Zero fabricated results (returns explicit insufficient-data error when data is lacking).
- Zero look-ahead bias (strict completion alignment on higher-timeframe candles).
- Conservative intrabar execution (assumes stop-loss hit when SL and TP are both touched in the same bar).
- True execution costs (spread, slippage, round-trip commissions, lot size constraints via GoldRiskEngine).
- Independent scorer instances to prevent mutating live strategy modes.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from brokers.base import AccountInfo, SymbolInfo
from engine.data_hub import DataHub
from engine.indicators import TechnicalIndicators
from execution.gold_risk import GoldRiskEngine
from strategy.confluence_scorer import ConfluenceScorer, ConfluenceSignal

logger = logging.getLogger("ApexGold.BacktestEngine")


@dataclass
class BacktestTrade:
    """Historical trade simulation record."""
    ticket: int
    entry_time: int
    entry_time_str: str
    exit_time: int
    exit_time_str: str
    symbol: str
    direction: str
    volume: float
    entry_price: float
    exit_price: float
    sl: float
    tp: float
    pnl: float
    return_r: float
    regime: str
    score: int
    exit_reason: str
    trading_cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestResult:
    """Full backtest performance report."""
    symbol: str
    initial_balance: float
    final_equity: float
    net_profit: float
    return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    max_drawdown_usd: float
    max_drawdown_pct: float
    equity_curve: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    avg_win_usd: float = 0.0
    avg_loss_usd: float = 0.0
    avg_return_r: float = 0.0
    trading_costs_usd: float = 0.0
    rejected_trades: int = 0
    insufficient_data: bool = False
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BacktestEngine:
    """
    High-Speed Multi-Timeframe Backtest Runner.
    Simulates real execution with spread, slippage, commission, and Chameleon trade management.
    Never invents or fabricates data.
    """

    def __init__(
        self,
        scorer: Optional[ConfluenceScorer] = None,
        initial_balance: float = 10.0,
        risk_per_trade_pct: float = 2.0,
        spread_points: float = 0.25,  # $0.25 on Gold
        slippage_points: float = 0.05,
        commission_per_lot: float = 7.0,  # $7 per standard lot round-trip ($0.07 per 0.01 lot)
    ):
        self.scorer = scorer or ConfluenceScorer(mode="standard")
        self.initial_balance = initial_balance
        self.risk_per_trade_pct = risk_per_trade_pct
        self.spread_points = spread_points
        self.slippage_points = slippage_points
        self.commission_per_lot = commission_per_lot

    async def run(
        self,
        data_hub: DataHub,
        symbol: str = "XAUUSD",
        days: int = 30,
        mode: str = "balanced",
    ) -> BacktestResult:
        """
        Execute multi-timeframe backtest over the specified number of historical days.
        Uses an isolated ConfluenceScorer instance to avoid mutating the live trading mode.
        """
        # Independent scorer instance: prevents mutating live strategy settings
        bt_scorer = ConfluenceScorer(
            threshold_aggressive=getattr(self.scorer, "threshold_aggressive", 60),
            threshold_balanced=getattr(self.scorer, "threshold_balanced", 68),
            threshold_sniper=getattr(self.scorer, "threshold_sniper", 82),
            mode=mode.lower() if mode else "balanced",
            aggressiveness_pct=getattr(self.scorer, "aggressiveness_pct", 55.0),
        )
        logger.info(f"Starting {days}-day backtest on {symbol} (Mode: {bt_scorer.mode.upper()})...")

        # Pull historical data from DataHub
        bars_1h = await data_hub.get_dataframe(symbol, "1H", count=days * 24)
        bars_15m = await data_hub.get_dataframe(symbol, "15m", count=days * 24 * 4)
        bars_5m = await data_hub.get_dataframe(symbol, "5m", count=days * 24 * 12)
        bars_1m = await data_hub.get_dataframe(symbol, "1m", count=min(10080, days * 24 * 60))

        # Check for insufficient data: NEVER return fabricated / fallback trades!
        if bars_5m is None or len(bars_5m) < 50:
            logger.warning(f"Insufficient historical data in DataHub for backtest on {symbol}. Returning explicit status.")
            return BacktestResult(
                symbol=symbol,
                initial_balance=self.initial_balance,
                final_equity=self.initial_balance,
                net_profit=0.0,
                return_pct=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate_pct=0.0,
                profit_factor=0.0,
                max_drawdown_usd=0.0,
                max_drawdown_pct=0.0,
                equity_curve=[{"time": int(time.time()), "equity": round(self.initial_balance, 2)}],
                trades=[],
                avg_win_usd=0.0,
                avg_loss_usd=0.0,
                avg_return_r=0.0,
                trading_costs_usd=0.0,
                rejected_trades=0,
                insufficient_data=True,
                error_message="Insufficient historical candle data. Please fetch more history from MT5 before running a backtest.",
            )

        return self._simulate(symbol, bt_scorer, bars_1h, bars_15m, bars_5m, bars_1m)

    def _simulate(
        self,
        symbol: str,
        bt_scorer: ConfluenceScorer,
        df_1h: Optional[pd.DataFrame],
        df_15m: Optional[pd.DataFrame],
        df_5m: pd.DataFrame,
        df_1m: Optional[pd.DataFrame],
    ) -> BacktestResult:
        """
        Chronological multi-timeframe simulation.
        Enforces:
        - Strict higher-timeframe completion (no look-ahead bias).
        - Conservative intrabar handling when SL and TP are both inside one bar range.
        - GoldRiskEngine sizing and trade rejection.
        - Real execution costs (spread, slippage, commission).
        - Chronological single-position constraint (max_open_positions: 1).
        """
        balance = self.initial_balance
        peak_equity = balance
        max_drawdown_usd = 0.0
        max_drawdown_pct = 0.0
        rejected_trades = 0
        total_trading_costs = 0.0

        is_gold = any(k in symbol.upper() for k in ["XAU", "GOLD"])
        contract_size = 100.0 if is_gold else 100000.0
        point = 0.01 if is_gold else 0.00001
        digits = 2 if is_gold else 5

        risk_engine = GoldRiskEngine(
            budget_usd=self.initial_balance,
            risk_per_trade_pct=self.risk_per_trade_pct,
            compounding_enabled=True,
        )

        equity_curve: List[Dict[str, Any]] = [
            {"time": int(df_5m["time"].iloc[0]), "equity": round(balance, 2)}
        ]
        trades: List[BacktestTrade] = []
        ticket_counter = 1

        n_bars = len(df_5m)
        i = 50  # Start after warmup window

        while i < n_bars - 1:
            current_bar = df_5m.iloc[i]
            bar_start_time = float(current_bar["time"])
            decision_time = bar_start_time + 300.0  # Decision made at 5m close

            # -------------------------------------------------------------
            # Prevent Look-Ahead Bias: Only use completed higher-TF candles
            # -------------------------------------------------------------
            hist_5m = df_5m.iloc[:i + 1]

            if df_1h is not None and not df_1h.empty:
                hist_1h = df_1h[df_1h["time"] + 3600.0 <= decision_time]
            else:
                hist_1h = hist_5m

            if df_15m is not None and not df_15m.empty:
                hist_15m = df_15m[df_15m["time"] + 900.0 <= decision_time]
            else:
                hist_15m = hist_5m

            if df_1m is not None and not df_1m.empty:
                hist_1m = df_1m[df_1m["time"] + 60.0 <= decision_time]
            else:
                hist_1m = hist_5m

            # Require minimum history on higher timeframes
            if len(hist_1h) < 5 or len(hist_15m) < 10:
                i += 1
                continue

            dfs = {
                "1H": hist_1h,
                "15m": hist_15m,
                "5m": hist_5m,
                "1m": hist_1m,
            }

            bid = float(current_bar["close"])
            ask = round(bid + self.spread_points, digits)

            buy_sig, sell_sig = bt_scorer.evaluate(
                symbol=symbol,
                dfs=dfs,
                current_tick_bid=bid,
                current_tick_ask=ask,
                utc_timestamp=decision_time,
            )

            signal = buy_sig if (buy_sig and (not sell_sig or buy_sig.score >= sell_sig.score)) else sell_sig

            if signal and signal.score >= bt_scorer.get_active_threshold():
                direction = signal.direction
                raw_entry = float(signal.entry_price)
                sl_p = float(signal.suggested_sl)
                tp_p = float(signal.suggested_tp)
                regime = signal.regime

                # Realistic Entry with Spread & Slippage
                if direction == "BUY":
                    entry_p = round(ask + self.slippage_points, digits)
                else:
                    entry_p = round(bid - self.slippage_points, digits)

                risk_dist = abs(entry_p - sl_p)
                if risk_dist <= point:
                    i += 1
                    continue

                # Position Sizing via GoldRiskEngine
                sym_info = SymbolInfo(
                    symbol=symbol,
                    bid=bid,
                    ask=ask,
                    point=point,
                    digits=digits,
                    spread=self.spread_points,
                    contract_size=contract_size,
                    tick_size=point,
                    tick_value=point * contract_size,
                    volume_min=0.01,
                    volume_max=100.0,
                    volume_step=0.01,
                )

                acc_info = AccountInfo(
                    balance=balance,
                    equity=balance,
                    margin=0.0,
                    free_margin=balance,
                    currency="USD",
                )
                lot_size, risk_meta = risk_engine.calculate_lot_size(
                    symbol_info=sym_info,
                    account_info=acc_info,
                    sl_distance=risk_dist,
                )
                has_margin = risk_meta.get("has_margin", True)

                # Reject trade if minimum lot exceeds permitted risk budget
                if not has_margin or lot_size <= 0.0:
                    rejected_trades += 1
                    i += 1
                    continue

                # Monetary risk budget for this trade
                risk_usd = risk_dist * lot_size * contract_size

                # Trading cost calculation (Commission + Slippage + Spread)
                comm = round(lot_size * self.commission_per_lot, 2)
                slip_cost = round(self.slippage_points * lot_size * contract_size, 2)
                trade_costs = round(comm + slip_cost, 2)
                total_trading_costs += trade_costs

                # -------------------------------------------------------------
                # Forward Simulation over Subsequent Bars
                # -------------------------------------------------------------
                exit_p = entry_p
                exit_time = decision_time
                exit_reason = "TIME_EXIT"
                ratchet_stage = 0
                highest_r = 0.0
                f = i + 1

                for f in range(i + 1, min(i + 40, n_bars)):
                    bar_f = df_5m.iloc[f]
                    h_f = float(bar_f["high"])
                    l_f = float(bar_f["low"])
                    c_f = float(bar_f["close"])
                    bar_f_time = float(bar_f["time"])
                    exit_time = bar_f_time + 300.0

                    # Calculate current R reached
                    if direction == "BUY":
                        cur_r = (h_f - entry_p) / risk_dist
                    else:
                        cur_r = (entry_p - l_f) / risk_dist

                    if cur_r > highest_r:
                        highest_r = cur_r

                    # Ratchet Floor: One-Way Stop Protection (never loosens)
                    if ratchet_stage < 1 and cur_r >= 0.65:
                        ratchet_stage = 1
                        new_sl = round(entry_p + self.spread_points if direction == "BUY" else entry_p - self.spread_points, digits)
                        sl_p = max(sl_p, new_sl) if direction == "BUY" else min(sl_p, new_sl)

                    elif ratchet_stage < 2 and cur_r >= 1.00:
                        ratchet_stage = 2
                        new_sl = round(entry_p + (0.45 * risk_dist) if direction == "BUY" else entry_p - (0.45 * risk_dist), digits)
                        sl_p = max(sl_p, new_sl) if direction == "BUY" else min(sl_p, new_sl)

                    elif ratchet_stage < 3 and cur_r >= 1.50:
                        ratchet_stage = 3
                        new_sl = round(entry_p + (0.85 * risk_dist) if direction == "BUY" else entry_p - (0.85 * risk_dist), digits)
                        sl_p = max(sl_p, new_sl) if direction == "BUY" else min(sl_p, new_sl)

                    # ---------------------------------------------------------
                    # Check Intrabar SL / TP Hits with Conservative Collision
                    # ---------------------------------------------------------
                    sl_hit = (l_f <= sl_p) if direction == "BUY" else (h_f >= sl_p)
                    tp_hit = (h_f >= tp_p) if direction == "BUY" else (l_f <= tp_p)

                    # Conservative Resolution: If BOTH SL and TP are reached in the same bar,
                    # assume the adverse outcome (Stop Loss was hit first)
                    if sl_hit and tp_hit:
                        exit_p = sl_p
                        exit_reason = "BREAKEVEN_LOCK" if ratchet_stage >= 1 else "SL_HIT"
                        break
                    elif sl_hit:
                        exit_p = sl_p
                        exit_reason = "BREAKEVEN_LOCK" if ratchet_stage >= 1 else "SL_HIT"
                        break
                    elif tp_hit:
                        exit_p = tp_p
                        exit_reason = "TP_HIT"
                        break

                    # Exhaustion bank in Runner mode if stalled past 15 bars
                    if regime == "ADAPTIVE_RUNNER" and (f - i) > 15 and highest_r >= 1.2:
                        exit_p = c_f
                        exit_reason = "EXHAUSTION_BANK"
                        break

                # Calculate Net P&L (Real Gross P&L minus trading costs)
                if direction == "BUY":
                    gross_pnl = (exit_p - entry_p) * lot_size * contract_size
                else:
                    gross_pnl = (entry_p - exit_p) * lot_size * contract_size

                net_pnl = round(gross_pnl - trade_costs, 2)
                return_r = round(net_pnl / max(0.01, risk_usd), 2)
                balance = round(balance + net_pnl, 2)

                # Drawdown tracking
                if balance > peak_equity:
                    peak_equity = balance
                dd_usd = round(peak_equity - balance, 2)
                dd_pct = round((dd_usd / peak_equity) * 100.0, 2) if peak_equity > 0 else 0.0

                if dd_usd > max_drawdown_usd:
                    max_drawdown_usd = dd_usd
                if dd_pct > max_drawdown_pct:
                    max_drawdown_pct = dd_pct

                trades.append(BacktestTrade(
                    ticket=ticket_counter,
                    entry_time=int(decision_time),
                    entry_time_str=datetime.fromtimestamp(decision_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
                    exit_time=int(exit_time),
                    exit_time_str=datetime.fromtimestamp(exit_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
                    symbol=symbol,
                    direction=direction,
                    volume=lot_size,
                    entry_price=round(entry_p, digits),
                    exit_price=round(exit_p, digits),
                    sl=round(sl_p, digits),
                    tp=round(tp_p, digits),
                    pnl=net_pnl,
                    return_r=return_r,
                    regime=regime,
                    score=signal.score,
                    exit_reason=exit_reason,
                    trading_cost=trade_costs,
                ))
                ticket_counter += 1

                equity_curve.append({
                    "time": int(exit_time),
                    "equity": round(balance, 2),
                })

                # Chronological Single-Position Constraint:
                # Step past the completed trade before evaluating next signal
                i = f + 1
            else:
                i += 1

        # Summary Metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        losing_trades = sum(1 for t in trades if t.pnl < 0)
        win_rate = round((winning_trades / total_trades) * 100.0, 1) if total_trades > 0 else 0.0

        wins_usd = [t.pnl for t in trades if t.pnl > 0]
        losses_usd = [t.pnl for t in trades if t.pnl < 0]

        gross_profit = sum(wins_usd)
        gross_loss = abs(sum(losses_usd))
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

        avg_win_usd = round(sum(wins_usd) / len(wins_usd), 2) if wins_usd else 0.0
        avg_loss_usd = round(sum(losses_usd) / len(losses_usd), 2) if losses_usd else 0.0
        avg_return_r = round(sum(t.return_r for t in trades) / total_trades, 2) if total_trades > 0 else 0.0

        net_profit = round(balance - self.initial_balance, 2)
        return_pct = round((net_profit / self.initial_balance) * 100.0, 1)

        return BacktestResult(
            symbol=symbol,
            initial_balance=self.initial_balance,
            final_equity=balance,
            net_profit=net_profit,
            return_pct=return_pct,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate_pct=win_rate,
            profit_factor=profit_factor,
            max_drawdown_usd=max_drawdown_usd,
            max_drawdown_pct=max_drawdown_pct,
            avg_win_usd=avg_win_usd,
            avg_loss_usd=avg_loss_usd,
            avg_return_r=avg_return_r,
            trading_costs_usd=round(total_trading_costs, 2),
            rejected_trades=rejected_trades,
            insufficient_data=False,
            equity_curve=equity_curve,
            trades=[t.to_dict() for t in trades],
        )
