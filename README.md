# 🏆 ApexGold — Autonomous Multi-Timeframe Algorithmic Trading Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![MetaTrader 5](https://img.shields.io/badge/MetaTrader-5-red.svg)](https://www.metatrader5.com/)
[![Tests](https://img.shields.io/badge/Tests-99%20Passed-emerald.svg)]()
[![License](https://img.shields.io/badge/License-Proprietary-yellow.svg)](LICENSE)

ApexGold is an institutional-grade, fully autonomous quantitative trading platform and real-time Web Dashboard built in Python and MQL5. Engineered specifically for **Gold (`XAUUSD` / `GOLD` / `XAUUSDc` / `XAUUSDm`)** and high-liquidity Forex majors (`EURUSD`, `GBPUSD`, `USDJPY`), it is designed to safely scale accounts of all sizes—from micro capital (**$10.00** demo/cent accounts) to standard portfolios—using strict asymmetric risk management, multi-timeframe trend alignment, and real-time market-following execution.

---

## ⚡ Core Features & Strategy Architecture

### 1. Adaptive "Chameleon" Execution Engine (250ms Loop)
- **Dynamic Setup Classification**: Automatically classifies trades into `⚡ FAST_SCALP` ($1.2\text{R}-2.2\text{R}$ target) or `🚀 WAVE_RUNNER` ($2.5\text{R}-4.0\text{R}$ target) based on 1H/15m institutional volume expansion.
- **Multi-Stage Volatility Ratchet SL Floor**:
  - $+0.45\text{R}/+0.65\text{R} \to$ Locks Breakeven with spread buffer (risk-free trade).
  - $+1.00\text{R} \to$ Locks guaranteed $+0.45\text{R}$ profit floor.
  - $+1.50\text{R} \to$ Locks guaranteed $+0.85\text{R}$ profit floor.
  - $+2.00\text{R} \to$ Locks guaranteed $+1.35\text{R}$ profit floor.
  - $+2.50\text{R}+ \to$ Structural trailing behind dynamic swing levels.
- **Dynamic Take-Profit Expansion**: Surges with strong continuous momentum automatically push TP out to $4.0\text{R}-5.0\text{R}$ to let institutional winners run.
- **Market Flip & Early Stall Exits**: Cuts losses early ($-0.15\text{R}$ to $-0.65\text{R}$) if higher timeframe structure flips, or banks cash when tick velocity stalls (4.0s stall exit in fast scalps).
- **Inline SL/TP Controls in Web UI**: Real-time position card with quick nudge chips (`SL +$1`, `SL -$1`, `TP +$2`, `Lock BE`) and direct `Save SL/TP to MT5` synchronization.

### 2. Institutional Planned Limits Engine
- **Calculations While Waiting**: Eliminates dead waiting by calculating institutional `BUY_LIMIT` and `SELL_LIMIT` levels at key 5m/15m SuperTrend support/resistance zones, displayed directly in the dashboard cards.
- **Autonomous & Manual Staging**:
  - Automatically places limit orders on MT5 in `full_auto` mode when confluence criteria are met.
  - Manual 1-click `Stage` button to queue pending orders directly to MT5.
- **Predictive Invalidation & Drift Cancellation**:
  - Auto-cancels limits if price drifts $> \$2.50$ away from the limit price.
  - Predictive invalidation cancels the pending order BEFORE fill if support/resistance breaks.
  - Prioritizes live market signals: auto-clears pending limits when a high-confluence market signal fires with 0 open positions.

### 3. Smart Pre-News Trajectory & Free-Roll Engine
Replaces passive, blind 15-minute news blackouts with an intelligent 3-zone system:
- **Zone A ($T-45\text{m}$ to $T-90\text{s}$ — Pre-News Trajectory Window)**:
  - Evaluates fundamental forecast vs. previous data (e.g. CPI cooling vs. heating USD).
  - Measures institutional order-flow drift (15m, 5m, 1m EMA alignment) and candle expansion.
  - When **Trajectory Confidence $\ge 78\%$** and confluence score $\ge 63$, automatically enters the trade with a `+12pt` trajectory alignment boost (`🎯 PRE_NEWS_TRAJECTORY`).
- **Zone B ($T-90\text{s}$ to $T-0\text{s}$ — Final Spread-Spike Buffer)**:
  - Freezes new raw market entries right before release as liquidity providers widen spreads.
  - **$T-90\text{s}$ Free-Roll Lock**: If an open position is winning $\ge +0.40\text{R}$, locks Stop Loss to $\text{Entry} + \text{Spread} + 0.20\text{R}$ guaranteed profit and expands Take Profit to $4.0\text{R}$ (completely risk-free runner).
  - **$T-60\text{s}$ Pre-News Scratch**: If an open position is stalled or negative ($< +0.40\text{R}$), auto-closes before release to prevent two-way spread blowout.
- **Zone C ($T+15\text{s}$ to $T+15\text{m}$ — Post-Release Impulse Momentum)**:
  - Once live broker spread normalizes, rides post-news breakout waves with a `+15pt` momentum boost.

### 4. 100-Point Confluence Scoring Engine & Market Structure
- **4-Timeframe Analysis**:
  - **1-Hour (1H)**: Macro institutional trend, EMA 50 slope, and 1H structural wall clearance ($1.8 \times \text{ATR}_{15m}$).
  - **15-Minute (15m)**: Session wave direction, EMA 20/50 alignment, and ADX momentum.
  - **5-Minute (5m)**: Setup structure, value-zone pullbacks, and consolidation box breakouts.
  - **1-Minute (1m)**: Precision execution trigger, sweeps, and candle momentum.
- **Quantitative & SMC Edge**:
  - Institutional Volume-Weighted Average Price (**VWAP**) & Kaufman Adaptive Moving Average (**KAMA**).
  - Statistical Trend Quality ($R^2 \ge 0.60$) & Z-Score Mean Retest levels.
  - Dreiss Choppiness Index (CI) to gate consolidation chop ($\text{CI} \ge 60$).
  - Smart Money Concepts (**SMC**): Liquidity Sweeps, Order Block Imbalance, and Bull/Bear Trap detection.

### 5. Adaptive Position Sizing ($10 to $10,000+)
- **Micro-Account Scaling ($10–$35)**: Specifically calibrated for small balances ($10.00). Broker minimum lot ($0.01$ lot) has an expanded risk budget ($45\%$ / $\$4.50$) under aggressive profile and a $60\%$ initial open risk envelope, allowing $0.01$ Gold trades to execute without hitting artificial risk blocks.
- **Compounding Growth**: As account balance doubles ($10 \to 20 \to 40 \to 70 \to 140$), lot sizing dynamically compounds based on fractional risk equity.
- **Zero Daily Loss Limit**: `max_daily_losses: 0` disables arbitrary trade count limits, allowing the bot to run continuously while protecting capital via the $25\%$ daily drawdown circuit breaker.
- **Losing Streak Throttle**: Progressively throttles risk budget by $20\%$ (1 loss), $40\%$ (2 losses), and $50\%$ (3+ losses), resetting to $100\%$ on the next win.

---

## 🎛️ Transmission Gears & Aggressiveness Scroller

The Web Dashboard features a 4-speed transmission gear system and an anti-jitter aggressiveness slider:

| Gear | Mode | Target / Min Score | Characteristics |
| :--- | :--- | :--- | :--- |
| **Sniper** | `cautious` | 35% Scroller (Min 76) | Patient sniper entries. High confluence, strict filters. |
| **Balanced** | `standard` | 55% Scroller (Min 68) | Balanced swing and wave setups. |
| **Momentum** | `aggressive` | 75% Scroller (Min 60) | Active momentum scalping. Fast entries on pullbacks. |
| **Hyper Scalp** | `hyper` | 92% Scroller (Min 52) | High-frequency scalping. Relaxes 15m/1H hard blocks. |
| **Auto-Pilot** | `auto_adaptive` | Auto-Dialed (1%–100%) | **Flagship AI Brain**: Evaluates real-time volatility, ATR, and spread every 5s to auto-adjust aggressiveness. |

---

## 🎮 Execution Modes & 1-Click UI Buttons

| Mode | Behavior |
| :--- | :--- |
| **`scanner`** (Read-Only) | Live technical scanner and confluence analyzer. Computes scores and limits without placing orders. |
| **`confirm`** (Semi-Auto) | Emits audio/visual alerts and stages 1-click execution cards. Requires 1 user click to execute. |
| **`full_auto`** (Autonomous) | **Zero manual input required.** Sizes, calculates SL/TP, and submits orders to MT5 autonomously. |

### Dashboard 1-Click Direct Buttons:
- **`⚡ BUY 0.01` / `⚡ SELL 0.01`**: Instant market orders sent directly to MT5 via the bridge.
- **`Stage`**: Instantly stages the calculated institutional `BUY_LIMIT` or `SELL_LIMIT`.
- **`CLOSE ALL`**: Emergency 1-click close for all active MT5 positions.
- **`Lock BE`**: Moves Stop Loss to entry + spread buffer on all winning positions.

---

## 🛠️ System Requirements

- **Operating System**: macOS (12.0+ Monterey, Ventura, Sonoma, Sequoia) or Windows (10 / 11) or Linux (Ubuntu 22.04+)
- **Python**: Version 3.10, 3.11, or 3.12 (Python 3.11/3.12 recommended)
- **MetaTrader 5**: Desktop terminal (demo or live account)

---

## 💻 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/ChaficH/ApexGold.git
cd ApexGold
git checkout Chafic_Branch
```

### 2. Set Up Virtual Environment

#### On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### On Windows:
```cmd
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔑 Bridge Authentication Token

To protect your live trading account, the MetaTrader 5 bridge uses a shared secret token:
- **Default Token:** Both `config.yaml` and `mql5/ApexBridge.mq5` are pre-configured with the matching default token:
  ```yaml
  bridge_token: "apex_secret_2026"
  ```
  **You do not need to do anything if you keep this default!** The EA and Python server connect seamlessly out-of-the-box.

---

## 🍎 How to Run Live Trading on macOS (Wine / CrossOver MT5)

MetaTrader 5 on macOS runs in an Apple Wine/CrossOver container. The bridge operates via high-speed local WebRequest (`http://127.0.0.1:8000/api/mt5_bridge/tick`) with automated queue fallback.

### Step 1: Configure `config.yaml` for Live Execution
Open [`config.yaml`](config.yaml) and ensure:
```yaml
broker:
  active_broker: "mt5_mac_socket"
  bridge_token: "apex_secret_2026"

trading:
  primary_symbol: "XAUUSD"          # or GOLD / XAUUSDc / XAUUSDm
  live_trading_enabled: true         # Enable live execution
  execution_mode: "full_auto"       # "full_auto" or "confirm"
  strategy_mode: "auto_adaptive"    # AI Auto-Pilot
  account_type: "micro"
  budget_usd: 10.00
  risk_profile: "aggressive"
  risk_per_trade_pct: 6.0
  max_open_positions: 2
  manage_manual_positions: true
  max_daily_losses: 0               # 0 = disabled (unlimited)
```

### Step 2: Enable WebRequest in MetaTrader 5 (One-Time Setup)
1. In MetaTrader 5, open **Tools $\to$ Options** (or press `Ctrl + O` / `Cmd + ,`).
2. Click the **Expert Advisors** tab.
3. Check **"Allow Algo Trading"**.
4. Check **"Allow WebRequest for listed URL"**.
5. Click the green `+` icon and add:
   ```text
   http://127.0.0.1:8000
   ```
6. Click **OK**.
7. Ensure the **"Algo Trading"** button on the main MT5 top toolbar is green / active.

### Step 3: Install & Compile the Bridge EA in MT5
1. Inside MT5, open **MetaEditor** (press `F4`).
2. In the Navigator tree on the left, right-click `Experts` $\to$ **Open Folder**.
3. Copy [`mql5/ApexBridge.mq5`](mql5/ApexBridge.mq5) into that `Experts` folder.
4. In MetaEditor, open `ApexBridge.mq5` and press **Compile** (`F7`). Ensure output shows `0 errors, 0 warnings`.
5. Return to MT5, find `ApexBridge` in the Navigator tree under **Expert Advisors**, and drag it onto your active Gold chart (`XAUUSD` or `GOLD`).
6. In the EA Inputs dialog:
   - `InpBridgeMode` = `BRIDGE_WEBREQUEST`
   - `InpServerUrl` = `http://127.0.0.1:8000/api/mt5_bridge/tick`
   - `InpBridgeToken` = `apex_secret_2026`
7. Click **OK**. The top-right corner of the MT5 chart will show a smiling face icon.

### Step 4: Start the ApexGold Engine
In your macOS terminal:
```bash
python3 main.py
```
Open your browser to:
```text
http://127.0.0.1:8000
```
The status banner will display:
```text
🟢 MT5 MAC WINE BRIDGE (LIVE) | Connected | Execution Mode: FULL_AUTO
```

---

## 🪟 How to Run Live Trading on Windows

### Option A: Native MetaTrader 5 Python Connection (Recommended for Windows)

1. **Install the native MetaTrader 5 library:**
   ```cmd
   pip install MetaTrader5
   ```

2. **Configure `config.yaml`:**
   ```yaml
   broker:
     active_broker: "mt5_windows"
     windows_mt5:
       terminal_path: ""       # Leave empty for default installation
       login: 12345678         # Your MT5 account login number
       password: ""            # Your MT5 account password
       server: ""              # Broker server (e.g. "Exness-Real10")
   trading:
     live_trading_enabled: true
     execution_mode: "full_auto"
   ```

3. **Launch ApexGold:**
   ```cmd
   python main.py
   ```

### Option B: ApexBridge EA via WebRequest (Works on both Windows & Mac)
Follow the exact same 4 steps as the macOS setup above.

---

## 🧪 Running the Automated Test Suite

ApexGold contains **99 automated regression, integration, and risk-control tests**:

```bash
# Run the complete test suite
python3 -m pytest -q

# Run only the smart pre-news & free-roll engine tests
python3 -m pytest tests/test_smart_pre_news_engine.py -q

# Run static compilation check across all modules
python3 -m py_compile main.py brokers/*.py engine/*.py execution/*.py strategy/*.py ui/*.py tests/*.py
```

---

## 📜 License & Open-Source Compliance

- **License**: ApexGold is proprietary software licensed under the terms of the [`LICENSE`](LICENSE) agreement (Copyright (c) 2026 Chafic Hakim. All Rights Reserved).
- **IP & GPL-3.0 Compliance Audit**: A comprehensive analysis of all researched mathematical indicators (such as the Dreiss Choppiness Index and fractional sizing models) and their clean-room implementation status is documented in [`docs/OPEN_SOURCE_RESEARCH.md`](docs/OPEN_SOURCE_RESEARCH.md). All runtime dependencies in `requirements.txt` are permissively licensed (MIT, BSD, Apache-2.0).

---

## 📂 Repository Structure

```text
├── LICENSE                      # Formal Proprietary Software License & Terms of Use
├── config.yaml                  # Master settings (Risk, Broker, News, Tokens, Execution)
├── requirements.txt             # Python dependencies (Permissive open source only)
├── main.py                      # FastAPI server, WebSockets & autonomous trading loop
├── README.md                    # Setup, architecture & operations manual
├── docs/
│   └── OPEN_SOURCE_RESEARCH.md  # Open-source research, mathematical formulas & GPL-3.0 IP audit
├── mql5/
│   └── ApexBridge.mq5           # MQL5 Bridge EA with WebRequest & TCP socket support
├── brokers/
│   ├── base.py                  # AbstractBrokerAdapter & unified data classes
│   ├── paper_sim.py             # Built-in paper trading simulator
│   ├── mt5_mac_socket.py        # High-speed macOS/Wine & HTTP queue bridge adapter
│   └── mt5_windows.py           # Native Windows MetaTrader5 adapter
├── engine/
│   ├── data_hub.py              # Multi-timeframe bar fetcher & SQLite cache
│   ├── indicators.py            # Technical indicators, 4-TF trends & SMC math
│   ├── news_reactor.py          # Pre-news trajectory analyzer, calendar & velocity shield
│   └── trade_history_store.py   # Closed deals, statistics & win-rate tracking
├── strategy/
│   └── confluence_scorer.py     # 100-Point Confluence Scorer, Auto-Pilot & Planned Limits
├── execution/
│   ├── gold_risk.py             # Contract calculator, adaptive lot sizer & micro-account scaling
│   └── trade_manager.py         # 250ms Chameleon loop, T-90s Free-Roll lock & ratchet SL
├── tests/                       # 99 automated test suites
└── ui/
    ├── server.py                # REST API endpoints, token verification & 4 FPS WebSocket hub
    └── index.html               # Dark-mode TradingView Lightweight Charts Web UI
```

---

## 🛡️ Risk Disclaimer

*Trading foreign exchange and commodities on margin carries a high level of risk and may not be suitable for all investors. Leverage can work against you as well as for you. Past performance is not indicative of future results. Always test thoroughly in demo or paper trading environments before deploying real capital.*
