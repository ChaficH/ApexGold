# ApexGold Quantitative Research & Experimentation Log

Autonomous Quant Research and Continuous Engineering Log in branch `algo-upgrade`.
Tracks baseline reproduction, diagnostic findings, and experimental iterations evaluated against strict institutional acceptance gates.

---

## 1. Acceptance Gates (Production Release Criteria)

Every released candidate MUST pass all gates out-of-sample:
1. **Profit Factor (PF)**: $\ge 1.30$ across rolling walk-forward folds (net of spread, slippage, commissions).
2. **Expectancy ($R$)**: $> +0.15R$ net of costs, with bootstrap 95% Confidence Interval lower bound $> 0$.
3. **Maximum Drawdown**: $\le 15.0\%$ on 1% risk per trade.
4. **Win Rate / Payoff**: Win Rate $\ge 50\%$ OR Average Win / Average Loss ratio $\ge 1.80$.
5. **Fold Consistency**: $\ge 70\%$ of rolling walk-forward folds profitable.
6. **Held-Out Test**: Passes the locked 20% hold-out test chronologically evaluated once at candidate release.
7. **Regression Integrity**: All existing unit tests pass, zero regressions in live broker/bridge paths.

---

## 2. Benchmark Baseline (Measured on Real XAUUSD Ticks Mar-Apr 2018)

- **Dataset**: FX-Data Dukascopy/Dukascopy-DS high-density XAUUSD ticks aggregated into 1m (55,479 bars), 5m (11,098 bars), 15m (3,700 bars), and 1H (925 bars).
- **Execution Cost Parameters**: 1% risk per trade, \$0.25 fixed spread, \$0.05 slippage, \$7.00/standard lot round-trip commission (\$0.07/0.01 lot).
- **Harness Verification**: Exact reproduction confirmed via `python -m backtest.run --from 2018-03-01 --to 2018-04-30 --mode balanced`.

### Baseline Performance Table

| Metric | Balanced Mode (Standard) | Aggressive Mode (Threshold 60) | Acceptance Gate | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Total Trades** | 285 | 396 | $\ge 100$ | Sample OK |
| **Win Rate** | 44.2% (126W - 159L) | 38.1% (151W - 245L) | $\ge 50.0\%$ | ❌ FAIL |
| **Profit Factor** | **0.60** | **0.34** | $\ge 1.30$ | ❌ FAIL |
| **Net Profit (USD)** | **-\$284.82** | **-\$543.36** | $> \$0$ | ❌ FAIL |
| **Expectancy ($R$/trade)** | **-0.145R** | **-0.281R** | $> +0.15R$ | ❌ FAIL |
| **Bootstrap 95% CI ($R$)** | **[-0.232R, -0.051R]** | [-0.372R, -0.190R] | Lower bound $> 0$ | ❌ FAIL |
| **Avg Win / Avg Loss ($R$)** | +0.48R / -0.64R | +0.42R / -0.71R | $\ge 1.80$ | ❌ FAIL |
| **Max Drawdown** | **29.89%** | **54.34%** | $\le 15.0\%$ | ❌ FAIL |
| **Sharpe / Sortino** | -6.59 / -9.98 | -12.14 / -15.20 | $> 1.0$ | ❌ FAIL |

### Baseline Exit Breakdown (Balanced Mode)

| Exit Reason | Trade Count | % of Total | Avg Return ($R$) | Net PnL (USD) | Diagnosis |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BREAKEVEN_LOCK** | 87 | 30.5% | +0.14R | +\$89.02 | Triggered at +0.65R, sets SL to +0.12R; normal noise stops out trade prematurely |
| **SL_HIT** | 95 | 33.3% | -1.05R | -\$685.23 | Catastrophic loss driver; 1 out of 3 trades takes a full adverse stop |
| **TIME_EXIT** | 64 | 22.5% | -0.04R | -\$19.92 | Stalled trades after 40 bars exit near flat |
| **EXHAUSTION_BANK** | 37 | 13.0% | +1.19R | +\$303.97 | Only profitable automated exit; banks runners stalling past 15 bars |
| **TP_HIT** | 2 | 0.7% | +2.30R | +\$27.34 | Unrealistic 2.5R–4.0R target hit in only 1 of 142 trades |

### Directional & Session Breakdown (Balanced Mode)

| Segment | Trades | Win Rate | Net PnL (USD) | Expectancy ($R$) | Key Observation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BUY** | 135 | 43.0% | -\$161.95 | -0.184R | Heavy underperformance; knife-catching macro pullbacks |
| **SELL** | 150 | 45.3% | -\$122.87 | -0.109R | Modestly better than BUY but still negative |
| **ASIA** (00-07 UTC) | 63 | 33.3% | -\$97.77 | -0.223R | Low liquidity, erratic mean-reverting chop |
| **LONDON** (07-13 UTC) | 87 | 44.8% | -\$136.60 | -0.240R | Fakeouts, opening drive reversals trigger full stops |
| **NEW_YORK** (13-21 UTC) | 113 | 46.9% | -\$90.05 | -0.104R | Highest volume window, tighter spread execution |
| **OVERNIGHT** (21-24 UTC) | 22 | 59.1% | +\$39.60 | +0.248R | Predictable low-volatility drift |

---

## 3. Step 2 Diagnostics & Root Cause Analysis

### 1. The "Premature Ratchet Trap" (MFE vs Breakeven)
- Intrabar MFE distribution reveals that **60.4% of trades reach $\ge +0.50R$**, and **50.5% reach $\ge +1.50R$**.
- However, when a trade reaches $+0.65R$, the current ratchet instantly moves the stop loss to `entry_price + spread` ($+0.12R$).
- Gold normal intrabar volatility causes immediate noise pullbacks that trigger `BREAKEVEN_LOCK` (87 trades, 30.5%). After commission and slippage, the trader extracts negligible alpha ($+0.14R$) while forfeiting the 50.5% probability that the move extends to $+1.50R$.

### 2. The Unrealistic Fixed Target Trap
- Baseline Take-Profit targets are hardcoded to $2.5R - 4.0R$.
- In fast intraday 5m gold trading, only **0.7% of trades (2 out of 285)** ever reach 2.5R before either stalling or retracing.
- Capping targets at realistic dynamic ATR/volatility levels ($1.2R - 1.6R$) aligns with natural market excursions where $51.2\%$ of trades reach $+1.0R$.

### 3. Lack of Regime Conditioning
- The confluence score sums technical points linearly regardless of whether the market is trending, mean-reverting in a range, or expanding into volatility.
- Trades taken in choppy range conditions (e.g. Asia morning, dead London midday) enter trend-breakout setups that inevitably get trapped.

### 4. Non-Calibrated Score Thresholds
- Empirical calibration shows that trades scoring **73-82 have worse expectancy (-0.35R)** than trades scoring **68-72 (-0.10R)**.
- High scores often represent indicator "over-crowding" at the end of extended moves (exhaustion tops/bottoms) rather than genuine continuation momentum.

---

## 4. Experiment Roadmap & Gate Tracking

| Exp ID | Hypothesis & Method | Target Metrics | OOS PF | Expectancy ($R$) | Max DD | Gate Decision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-00** | Baseline (Balanced 2018 benchmark) | N/A | 0.60 | -0.145R | 29.89% | Baseline Ref |
| **EXP-01** | **Regime Detection Filter**: ADX/volatility-state + HMM regime gating (Trend continuation only in trend regimes, Range sweep only in range regimes, stand aside in low-quality chop). | PF $\ge 1.00$, Exp $> -0.05R$ | TBD | TBD | TBD | Pending |
| **EXP-02** | **Dynamic Volatility Stops & Realized Exits**: ATR-fitted dynamic TP (1.2R-1.6R), widened breakeven trigger (delay lock until +0.90R to allow breathing room), and session-based time stops. | PF $\ge 1.20$, Exp $> +0.05R$ | TBD | TBD | TBD | Pending |
| **EXP-03** | **Directional & Session Gating**: Eliminate low-liquidity Asia chop entries and filter counter-trend setups opposing 1H/15m macro regime. | PF $\ge 1.30$, Exp $> +0.10R$ | TBD | TBD | TBD | Pending |
| **EXP-04** | **Meta-Labeling ML Filter (LightGBM)**: Causal triple-barrier classification with purged cross-validation, trading only high-probability setups ($P(\text{win}) \ge \theta$). | PF $\ge 1.30$, Exp $> +0.15R$, 95% CI $> 0$ | TBD | TBD | TBD | Pending |

---
