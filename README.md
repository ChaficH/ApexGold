# 🏆 ApexGold — Autonomous Multi-Timeframe Algorithmic Trading Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![MetaTrader 5](https://img.shields.io/badge/MetaTrader-5-red.svg)](https://www.metatrader5.com/)
[![License](https://img.shields.io/badge/License-Proprietary-yellow.svg)]()

ApexGold is an institutional-grade, fully automated quantitative trading engine and real-time Web Dashboard built in Python and MQL5. Engineered specifically for **Gold (`XAUUSD` / `XAUUSDc` / `XAUUSDm`)** and high-liquidity Forex majors (`EURUSD`, `GBPUSD`, `USDJPY`), it is designed to safely scale accounts of all sizes—from micro capital (**$10.00** demo/cent accounts) to standard portfolios—using strict asymmetric risk management, multi-timeframe trend alignment, and real-time market-following execution.

---

## ⚡ Core Features & Architecture

### 1. Adaptive "Chameleon" Execution Engine (250ms Loop)
- **Dynamic Classification**: Automatically tags trade setups as `⚡ FAST_SCALP` ($1.2\text{R}$ target) or `🚀 WAVE_RUNNER` ($2.5\text{R}-4.0\text{R}$ target) based on 1H/15m institutional volume expansion.
- **Multi-Stage Volatility Ratchet SL**:
  - $+0.65\text{R} \to$ Locks Breakeven with spread buffer.
  - $+1.00\text{R} \to$ Locks guaranteed $+0.45\text{R}$ profit floor.
  - $+1.50\text{R} \to$ Locks guaranteed $+0.85\text{R}$ profit floor.
  - $+2.00\text{R} \to$ Locks guaranteed $+1.35\text{R}$ profit floor.
  - $+2.50\text{R}+ \to$ Structural trailing behind dynamic swing levels.
- **Dynamic Take-Profit Expansion**: If market surges into target with high momentum and low counter-wicks, pushes TP out $+1.25\text{R}$ to capture extended institutional waves.
- **Stall & Reversal Exits**: Bank profits immediately upon detecting adverse flow spikes or structural rejection wicks while deep in profit.

### 2. 100-Point Confluence Scoring Engine & Trend Alignment
- **4-Timeframe Analysis**:
  - **1-Hour (1H)**: Macro institutional trend, EMA 50 slope, and structural room-to-run ($20\text{ pts}$).
  - **15-Minute (15m)**: Session wave direction, EMA 20/50 alignment, and ADX momentum ($25\text{ pts}$).
  - **5-Minute (5m)**: Setup structure, value-zone pullbacks, and box breakouts ($30\text{ pts}$).
  - **1-Minute (1m)**: Precision sniper execution trigger, sweeps, and engulfing signals ($25\text{ pts}$).
- **Hard Counter-Trend Block**: Strictly prohibits trades that fight both 1H and 15m trends.
- **HTF Divergence Penalty**: If 1H and 15m diverge, applies a $-25$ penalty and caps max score at $68/100$, preventing counter-trend Sniper signals.
- **Quantitative & SMC Confluence**:
  - Institutional Volume-Weighted Average Price (**VWAP**) confirmation.
  - Kaufman's Adaptive Moving Average (**KAMA**) noise filtering via `package:ta`.
  - Statistical Trend Quality & $R^2$ regression via `scipy.stats.linregress`.
  - Smart Money Concepts (**SMC**): Order Block (OB) retests, Fair Value Gaps (FVG), and CHOCH/BOS structures via `smartmoneyconcepts`.

### 3. Strict Exposure & Small-Account Protection
- **Committed Exposure Guard**: Total exposure ($\text{Open Positions} + \text{Pending Orders}$) is strictly enforced. Accounts under $\$150$ are hard-capped at **1 single committed order**.
- **Zero Order Flooding / Stacking**: Pending limits cannot be stacked at adjacent ticks.
- **75-Second Limit Timeout & Drift Protection**: Unfilled limit orders auto-cancel after 75s or if price drifts $> \$2.50$ away.
- **Post-Loss Circuit Breaker**: Activates a 5-minute (300s) cooling-off period after any stop-loss closure to prevent revenge trading (pauses for 30 minutes after 2 consecutive losses).
- **Volatility-Adaptive Stop Loss**: Never artificially squashes Gold SL into market noise; maintains full ATR breathing room ($\$3.50-\$5.50$).

### 4. Millisecond News & Tick-Velocity Reactor
- **Timezone-Aware Economic Calendar**: Live countdown to CPI, NFP, FOMC, and interest rate releases.
- **Fundamental News Study**: Pre-news lockout ($T-180\text{s}$) prevents opening into red-folder volatility, while locking profitable trades to Breakeven so they ride releases risk-free.
- **Breaking News Sentiment**: Ingests real-time financial headlines (CNBC, MarketWatch) to calculate fundamental market sentiment.

### 5. Institutional Dark-Mode Web Dashboard
- **TradingView Lightweight Charts**: Smooth real-time candle streaming (1:1 tick sync with MT5).
- **Interactive SL/TP Controls**: Inline editing for open positions and pending limit orders with quick $+1$, $-1$, and Lock BE chips.
- **Live 4-TF Indicators**: Real-time directional bias badges (`1H: ▲`, `15m: ▲`, `5m: ▼`, `1m: ▲`).
- **Telemetry**: 4 FPS WebSocket feed providing instant equity, margin, spread, and algorithmic state.

---

## 🛠️ System Requirements

- **Operating System**: macOS (12.0+ Monterey, Ventura, Sonoma, Sequoia) or Windows (10 / 11) or Linux (Ubuntu 22.04+)
- **Python**: Version 3.10, 3.11, or 3.12 (Python 3.10+ recommended)
- **MetaTrader 5**: Desktop terminal (optional if running in built-in Paper Simulator mode)

---

## 💻 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/ChaficH/ApexGold.git
cd ApexGold
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

## 🚀 Running the Trading Engine

### Step 1: Launch the Server
```bash
# macOS / Linux
python3 main.py

# Windows
python main.py
```

### Step 2: Access the Dashboard
Open your web browser and navigate to:
```text
http://127.0.0.1:8000
```

The system will start immediately in **Paper Simulator** mode with simulated/live historical data. You can inspect the charts, run backtests, test signals, and adjust strategy gears directly from the UI.

---

## 🍎 Connecting MetaTrader 5 on macOS (Wine / CrossOver)

MetaTrader 5 on macOS runs inside an Apple Wine/CrossOver container. ApexGold includes a zero-latency local HTTP WebRequest bridge (**`ApexBridge.mq5`**) designed specifically for macOS Wine compatibility.

### 3-Step Setup:

1. **Allow Local WebRequest in MT5**:
   - In MT5, open **Tools $\to$ Options** (or press `Ctrl + O` / `Cmd + ,`).
   - Click the **Expert Advisors** tab.
   - Check **"Allow WebRequest for listed URL"**.
   - Click the green `+` icon and add:
     ```text
     http://127.0.0.1:8000
     ```
   - Check **"Allow Algo Trading"** in the dialog AND on the main MT5 top toolbar.

2. **Compile the Expert Advisor in Wine**:
   - Open **MetaEditor** (press `F4` inside MT5).
   - In the Navigator tree on the left, right-click `Experts` $\to$ **Open Folder**.
   - Copy [`mql5/ApexBridge.mq5`](file:///Users/chaficalhakim/Desktop/Trading%20script/mql5/ApexBridge.mq5) into the `Experts` directory.
   - In MetaEditor, open `ApexBridge.mq5` and press **Compile** (`F7`).
   - Confirm the output window displays `0 errors, 0 warnings`.

3. **Attach to Chart**:
   - In MT5, drag `ApexBridge` onto an active Gold chart (`XAUUSD`, `GOLD`, or `XAUUSDc`).
   - In the Inputs tab:
     - `InpBridgeMode` = `BRIDGE_WEBREQUEST`
     - `InpServerUrl` = `http://127.0.0.1:8000/api/mt5_bridge/tick`
   - In `config.yaml`, ensure:
     ```yaml
     broker:
       active_broker: "mt5_mac_socket"
     ```
   - The dashboard top status will change to:
     ```text
     MT5 MAC WINE BRIDGE (LIVE) 🟢
     ```

---

## 🪟 Connecting MetaTrader 5 on Windows

On Windows, you have two flexible connection options:

### Option A: Native MetaTrader 5 Python Adapter
1. Ensure the Windows `MetaTrader5` package is installed:
   ```cmd
   pip install MetaTrader5
   ```
2. In `config.yaml`, configure:
   ```yaml
   broker:
     active_broker: "mt5_windows"
     windows_mt5:
       terminal_path: ""       # Leave blank for default path or specify terminal64.exe
       login: 12345678         # Your MT5 account number
       password: "YourPassword"
       server: "Exness-Real10" # Your broker server name
   ```
3. Run `python main.py`. The engine will connect directly to the running MT5 terminal via native IPC.

### Option B: ApexBridge EA WebRequest
Works identically to the macOS setup above:
1. Allow `http://127.0.0.1:8000` in MT5 under **Tools $\to$ Options $\to$ Expert Advisors**.
2. Compile and attach `mql5/ApexBridge.mq5` to your chart.
3. In `config.yaml`, set `active_broker: "mt5_mac_socket"`.

---

## ⚙️ Configuration Reference (`config.yaml`)

```yaml
trading:
  primary_symbol: "XAUUSD"          # Active symbol (XAUUSD, GOLD, EURUSD, etc.)
  execution_mode: "full_auto"       # "full_auto", "confirm" (1-click), or "scanner"
  strategy_mode: "auto_adaptive"    # "auto_adaptive", "cautious", "standard", "aggressive", "hyper"
  account_type: "micro"             # "micro", "cent", "standard", "paper"
  budget_usd: 10.00                 # Base capital sizing
  risk_per_trade_pct: 2.0           # Fractional risk percentage per trade
  compounding_enabled: true         # Automatically scales sizing with account balance growth
  max_open_positions: 1             # Strict focus on 1 high-probability trade at a time

strategy:
  thresholds:
    aggressive: 60                  # Minimum score for aggressive scalp
    balanced: 68                    # Minimum score for standard entry
    sniper: 82                      # Minimum score for Tier 1 Sniper entry

news_reactor:
  enabled: true
  pre_news_freeze_seconds: 180      # Stop opening new trades 3m before Red Folder news
  lock_breakeven_on_open_trades: true # Move profitable trades to BE before news release
```

---

## 📂 Repository Structure

```text
├── config.yaml                  # Master settings (Risk, Broker, News, Scoring)
├── requirements.txt             # Python dependencies
├── main.py                      # FastAPI server, WebSockets & trading loop
├── README.md                    # Setup, architecture & operations manual
├── .gitignore                   # Ignores databases, cache, logs & local state
├── mql5/
│   └── ApexBridge.mq5           # MQL5 Bridge EA with WebRequest & TCP socket support
├── brokers/
│   ├── base.py                  # AbstractBrokerAdapter & unified data classes
│   ├── paper_sim.py             # Built-in paper trading simulator
│   ├── mt5_mac_socket.py        # High-speed macOS/Wine & TCP bridge adapter
│   └── mt5_windows.py           # Native Windows MetaTrader5 adapter
├── engine/
│   ├── data_hub.py              # Multi-timeframe bar fetcher & SQLite cache
│   ├── indicators.py            # Technical indicators, 4-TF trends & SMC math
│   ├── news_reactor.py          # Economic calendar, headlines & velocity shield
│   ├── historical_matcher.py    # Pattern matching & statistical expectancy
│   └── trade_history_store.py   # Closed deals, statistics & win-rate tracking
├── strategy/
│   └── confluence_scorer.py     # 100-Point Confluence Scorer & Chameleon classifier
├── execution/
│   ├── gold_risk.py             # Contract calculator, lot sizer & spread buffer
│   └── trade_manager.py         # 250ms Chameleon management loop & ratchet SL
├── backtest/
│   └── engine.py                # Multi-timeframe backtester & replay engine
├── telemetry/
│   └── telegram_bot.py          # Mobile alerts for signals & executions
└── ui/
    ├── server.py                # REST API endpoints & 4 FPS WebSocket broadcaster
    └── index.html               # Dark-mode TradingView Lightweight Charts Web UI
```

---

## 🛡️ Risk Disclaimer

*Trading foreign exchange and commodities on margin carries a high level of risk and may not be suitable for all investors. Leverage can work against you as well as for you. Past performance is not indicative of future results. Always test thoroughly in demo or paper trading environments before deploying real capital.*
