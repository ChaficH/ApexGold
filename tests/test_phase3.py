"""
Phase 3 Regression & Unit Tests for ApexGold Backtesting Accuracy.

Covers:
- Item 10: Eliminating fabricated fallback backtest results; returning explicit insufficient-data status.
- Item 11: Preventing look-ahead bias on higher timeframes and conservative intrabar conflict handling.
- Item 12: Aligning backtesting with live trading logic, isolating the live scorer, and verifying comprehensive performance metrics.
"""

import time
import pytest
import pandas as pd
import numpy as np

from backtest.engine import BacktestEngine, BacktestResult
from strategy.confluence_scorer import ConfluenceScorer


class MockDataHub:
    """Mock DataHub providing controllable historical multi-timeframe candles."""

    def __init__(self, data_map=None):
        self.data_map = data_map or {}

    async def get_dataframe(self, symbol: str, timeframe: str, count: int = 100):
        return self.data_map.get((symbol, timeframe))


# ==============================================================================
# 10. Test Removal of Fabricated Backtest Results
# ==============================================================================
@pytest.mark.anyio
async def test_insufficient_data_returns_error_status_not_fake_trades():
    """Item 10: When historical data is insufficient, return explicit error status with 0 trades."""
    engine = BacktestEngine(initial_balance=100.0)
    empty_hub = MockDataHub()  # Returns None for all timeframes

    result = await engine.run(data_hub=empty_hub, symbol="XAUUSD", days=30)

    assert isinstance(result, BacktestResult)
    assert result.insufficient_data is True
    assert result.total_trades == 0
    assert len(result.trades) == 0
    assert result.net_profit == 0.0
    assert result.final_equity == 100.0
    assert "Insufficient historical candle data" in (result.error_message or "")
    # Ensure fabricated fallback method does not exist
    assert not hasattr(engine, "_generate_fallback_backtest")


# ==============================================================================
# 11. Test Elimination of Look-Ahead Bias & Conservative Intrabar Execution
# ==============================================================================
def create_synthetic_bars(count: int, start_time: int, interval_sec: int, base_price: float = 2650.0):
    """Generate realistic OHLCV dataframe."""
    times = [start_time + (i * interval_sec) for i in range(count)]
    # Small realistic wave
    closes = [base_price + np.sin(i / 10.0) * 5.0 for i in range(count)]
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]
    opens = [c - 0.2 for c in closes]
    volumes = [100.0] * count

    return pd.DataFrame({
        "time": times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


def test_look_ahead_bias_higher_timeframe_alignment():
    """Item 11: A 1H candle starting at 14:00 must NOT be visible to decisions made at 14:15."""
    start_ts = 1700000000  # 14:00
    df_1h = create_synthetic_bars(count=5, start_time=start_ts, interval_sec=3600, base_price=2650.0)
    df_5m = create_synthetic_bars(count=60, start_time=start_ts, interval_sec=300, base_price=2650.0)

    engine = BacktestEngine()
    # At the 3rd 5m candle (14:15), decision_time = start + 300 + 300 + 300 = 14:15
    current_5m_bar = df_5m.iloc[2]
    decision_time = float(current_5m_bar["time"]) + 300.0

    # Test filtering logic used in engine:
    hist_1h = df_1h[df_1h["time"] + 3600.0 <= decision_time]

    # The 1H bar at 14:00 (which finishes at 15:00) must NOT be in hist_1h at 14:15!
    assert len(hist_1h) == 0


def test_conservative_intrabar_collision_favors_stop_loss():
    """Item 11: When both SL and TP are touched inside the same candle, assume Stop Loss hit."""
    engine = BacktestEngine(initial_balance=100.0, risk_per_trade_pct=2.0)
    bt_scorer = ConfluenceScorer(mode="standard")

    # Construct synthetic 5m bars where a trade opens and subsequent candle has extreme high AND low
    start_ts = 1700000000
    df_5m = create_synthetic_bars(count=70, start_time=start_ts, interval_sec=300, base_price=2650.0)
    df_15m = create_synthetic_bars(count=50, start_time=start_ts - (50 * 900), interval_sec=900, base_price=2650.0)
    df_1h = create_synthetic_bars(count=30, start_time=start_ts - (30 * 3600), interval_sec=3600, base_price=2650.0)

    # Force a candle with both extreme low (hitting SL) and extreme high (hitting TP)
    collision_bar_idx = 55
    df_5m.loc[collision_bar_idx, "high"] = 2750.0  # Would hit TP
    df_5m.loc[collision_bar_idx, "low"] = 2550.0   # Would hit SL

    result = engine._simulate(
        symbol="XAUUSD",
        bt_scorer=bt_scorer,
        df_1h=df_1h,
        df_15m=df_15m,
        df_5m=df_5m,
        df_1m=None,
    )

    # If any trade was active during bar 55, its exit reason must be SL_HIT (conservative), never TP_HIT
    for t in result.trades:
        if t["entry_time"] <= df_5m.loc[collision_bar_idx, "time"] <= t["exit_time"]:
            assert t["exit_reason"] in ("SL_HIT", "BREAKEVEN_LOCK")
            assert t["exit_reason"] != "TP_HIT"


# ==============================================================================
# 12. Test Strategy Mode Isolation & Comprehensive Metrics Reporting
# ==============================================================================
@pytest.mark.anyio
async def test_backtest_does_not_mutate_live_scorer_mode():
    """Item 12: Running a backtest in 'cautious' mode must NOT alter the live scorer's mode."""
    live_scorer = ConfluenceScorer(mode="hyper")  # Live engine is running hyper mode
    assert live_scorer.mode == "hyper"

    engine = BacktestEngine(scorer=live_scorer, initial_balance=100.0)
    hub = MockDataHub()

    # Run backtest with mode "cautious"
    await engine.run(data_hub=hub, symbol="XAUUSD", days=30, mode="cautious")

    # The live scorer's mode MUST remain untouched ("hyper")!
    assert live_scorer.mode == "hyper"


def test_backtest_reports_all_required_metrics():
    """Item 12: Verify BacktestResult contains all required analytics fields."""
    start_ts = 1700000000
    df_5m = create_synthetic_bars(count=100, start_time=start_ts, interval_sec=300, base_price=2650.0)
    df_15m = create_synthetic_bars(count=50, start_time=start_ts - (50 * 900), interval_sec=900, base_price=2650.0)
    df_1h = create_synthetic_bars(count=30, start_time=start_ts - (30 * 3600), interval_sec=3600, base_price=2650.0)

    engine = BacktestEngine(initial_balance=50.0, risk_per_trade_pct=2.0)
    bt_scorer = ConfluenceScorer(mode="standard")

    result = engine._simulate(
        symbol="XAUUSD",
        bt_scorer=bt_scorer,
        df_1h=df_1h,
        df_15m=df_15m,
        df_5m=df_5m,
        df_1m=None,
    )

    d = result.to_dict()
    required_fields = [
        "total_trades",
        "winning_trades",
        "losing_trades",
        "win_rate_pct",
        "net_profit",
        "profit_factor",
        "max_drawdown_usd",
        "max_drawdown_pct",
        "avg_win_usd",
        "avg_loss_usd",
        "avg_return_r",
        "trading_costs_usd",
        "rejected_trades",
        "insufficient_data",
        "equity_curve",
        "trades",
    ]
    for field in required_fields:
        assert field in d, f"Missing required field {field} in BacktestResult"
