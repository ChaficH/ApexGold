# 🏆 ApexGold — Autonomous Multi-Timeframe Algorithmic Trading Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![MetaTrader 5](https://img.shields.io/badge/MetaTrader-5-red.svg)](https://www.metatrader5.com/)
[![License](https://img.shields.io/badge/License-Proprietary-yellow.svg)]()

ApexGold is an institutional-grade, fully autonomous quantitative trading platform and real-time Web Dashboard built in Python and MQL5. Engineered specifically for **Gold (`XAUUSD` / `XAUUSDc` / `XAUUSDm`)** and high-liquidity Forex majors (`EURUSD`, `GBPUSD`, `USDJPY`), it is designed to safely scale accounts of all sizes—from micro capital (**$10.00** demo/cent accounts) to standard portfolios—using strict asymmetric risk management, multi-timeframe trend alignment, and real-time market-following execution.

---

## ⚡ Core Features & Strategy Architecture

### 1. Adaptive "Chameleon" Execution Engine (250ms Loop)
- **Dynamic Classification**: Automatically morphs trade setups between `⚡ FAST_SCALP` ($1.2\text{R}-2.2\text{R}$ target) and `🚀 WAVE_RUNNER` ($2.5\text{R}-4.0\text{R}$ target) based on 1H/15m institutional volume expansion.
- **Multi-Stage Volatility Ratchet SL Floor**:
  - $+0.45\text{R}/+0.65\text{R} \to$ Locks Breakeven with spread buffer.
  - $+1.00\text{R} \to$ Locks guaranteed $+0.45\text{R}$ profit floor.
  - $+1.50\text{R} \to$ Locks guaranteed $+0.85\text{R}$ profit floor.
  - $+2.00\text{R} \to$ Locks guaranteed $+1.35\text{R}$ profit floor.
  - $+2.50\text{R}+ \to$ Structural trailing behind dynamic swing levels.
- **Dynamic Take-Profit Expansion**: Surges with high trend momentum push TP out to $4.0\text{R}-5.0\text{R}$ to capture runaway institutional extensions.
- **Stall & Early Invalidation Exits**: Banks profits immediately upon detecting adverse order-flow stall (4.0s tick velocity stall in scalps) or structural invalidation.

### 2. Smart Pre-News Trajectory & Free-Roll Engine
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

### 3. 100-Point Confluence Scoring Engine & Market Structure
- **4-Timeframe Analysis**:
  - **1-Hour (1H)**: Macro institutional trend, EMA 50 slope, and 1H structural wall clearance ($1.8 \times \text{ATR}_{15m}$).
  - **15-Minute (15m)**: Session wave direction, EMA 20/50 alignment, and ADX momentum.
  - **5-Minute (5m)**: Setup structure, value-zone pullbacks, and consolidation box breakouts.
  - **1-Minute (1m)**: Precision execution trigger, sweeps, and candle momentum.
- **Proven SMC & Quantitative Edge**:
  - Institutional Volume-Weighted Average Price (**VWAP**) & Kaufman Adaptive Moving Average (**KAMA**).
  - Statistical Trend Quality ($R^2 \ge 0.60$) & Z-Score Mean Retest levels.
  - Dreiss Choppiness Index (CI) to gate consolidation chop ($\text{CI} \ge 60$).
  - Smart Money Concepts (**SMC**): Liquidity Sweeps, Order Block Imbalance, and Bull/Bear Trap detection.

### 4. Adaptive Position Sizing ($10 to $10,000+)
- **Mathematical Lot Allocation**: Calculates lot size based on exact MT5 balance, equity, leverage, contract size ($100\text{ oz}$ on standard gold), and ATR-based stop-loss distance.
- **Micro-Account Safe ($10–$25)**: For accounts $\le \$25$, allows a calibrated minimum stop loss distance of $\$1.50$, allowing a $0.01$ lot trade to fit within the risk budget without exceeding risk limits.
- **Zero Risk Exceedance**: Rounding always truncates downward; calculated monetary risk strictly never exceeds the configured risk budget.

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

## 🔑 Bridge Authentication Token (How to Set or Change)

To protect your live trading account, the MetaTrader 5 bridge uses a shared secret token:
- **Default Token:** Both `config.yaml` and `mql5/ApexBridge.mq5` are **already pre-configured** with the matching default token:
  ```yaml
  bridge_token: "apex_secret_2026"
  ```
  **You do not need to do anything if you keep this default!** The EA and Python server connect seamlessly out-of-the-box.

- **To Change the Token (Optional Custom Password):**
  1. Open [`config.yaml`](config.yaml) and update:
     ```yaml
     broker:
       bridge_token: "my_custom_secret_123"
     ```
  2. In MetaTrader 5, right-click the `ApexBridge` EA on your chart $\to$ **Properties (F7)** $\to$ **Inputs**:
     - Set `InpBridgeToken` = `"my_custom_secret_123"`
  3. Click **OK**.

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
  primary_symbol: "XAUUSD"          # or XAUUSDc / XAUUSDm depending on your broker
  live_trading_enabled: true         # Enabled for live execution
  execution_mode: "full_auto"       # "full_auto" for autonomous entry, or "confirm" for 1-click UI confirmation
  budget_usd: 10.00                 # Sized automatically to your account
  risk_per_trade_pct: 2.5
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
5. Return to MT5, find `ApexBridge` in the Navigator tree under **Expert Advisors**, and drag it onto your active Gold (`XAUUSD`) chart.
6. In the EA Inputs dialog:
   - `InpBridgeMode` = `BRIDGE_WEBREQUEST`
   - `InpServerUrl` = `http://127.0.0.1:8000/api/mt5_bridge/tick`
   - `InpBridgeToken` = `apex_secret_2026`
7. Click **OK**. The top-right corner of the MT5 chart will show a smiling face icon.

### Step 4: Start the ApexGold Engine
In your macOS terminal:
```bash
python main.py
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

On Windows, you can choose between native IPC connection (recommended) or the WebRequest EA bridge:

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
       terminal_path: ""       # Leave empty to use default installation, or set path to terminal64.exe
       login: 12345678         # Your MT5 account login number (optional if MT5 is already logged in)
       password: ""            # Your MT5 account password
       server: ""              # Broker server (e.g. "Exness-Real10", "ICMarkets-Demo")

   trading:
     live_trading_enabled: true
     execution_mode: "full_auto"
   ```

3. **Open MetaTrader 5:**
   Launch your desktop MT5 terminal and log into your account. Make sure **"Algo Trading"** is toggled on in the top toolbar.

4. **Launch ApexGold:**
   ```cmd
   python main.py
   ```
   The engine connects directly to the running MT5 terminal via native Windows IPC.

---

### Option B: ApexBridge EA via WebRequest (Works on both Windows & Mac)

Follow the exact same steps as the macOS setup above:
1. In MT5: **Tools $\to$ Options $\to$ Expert Advisors** $\to$ Allow WebRequest to `http://127.0.0.1:8000`.
2. Attach `mql5/ApexBridge.mq5` to your Gold chart.
3. In `config.yaml`, set:
   ```yaml
   broker:
     active_broker: "mt5_mac_socket"
     bridge_token: "apex_secret_2026"
   trading:
     live_trading_enabled: true
     execution_mode: "full_auto"
   ```
4. Run:
   ```cmd
   python main.py
   ```

---

## 🎮 Execution Modes Explained

You can set the execution mode in `config.yaml` or toggle it directly live in the web dashboard:

| Execution Mode | How It Works |
| :--- | :--- |
| **`scanner`** (Safe Default) | Read-only market analysis. Computes live scores, charts, and signals without placing any orders. |
| **`confirm`** (1-Click Semi-Auto) | Emits real-time sound alerts and displays a 1-click **Execute Trade** button on the UI dashboard and Telegram. Requires 1 manual click per setup. |
| **`full_auto`** (Fully Autonomous) | **Zero manual intervention.** When confluence threshold is reached, automatically sizes, calculates SL/TP, and submits orders to MT5 autonomously. |

---

## 🧪 Running the Full Test Suite

ApexGold contains 95 automated regression, integration, and risk-control tests:

```bash
# Run the complete test suite
python -m pytest tests/ -v

# Run only the smart pre-news & free-roll engine tests
python -m pytest tests/test_smart_pre_news_engine.py -v

# Run static compilation check across all modules
python -m py_compile main.py backtest/*.py brokers/*.py engine/*.py execution/*.py strategy/*.py ui/*.py tests/*.py
```

---

## 📂 Repository Structure

```text
├── config.yaml                  # Master settings (Risk, Broker, News, Tokens, Execution)
├── requirements.txt             # Python dependencies
├── main.py                      # FastAPI server, WebSockets & autonomous trading loop
├── README.md                    # Setup, architecture & operations manual
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
│   ├── news_reactor.py          # Pre-news trajectory analyzer, economic calendar & velocity shield
│   └── trade_history_store.py   # Closed deals, statistics & win-rate tracking
├── strategy/
│   └── confluence_scorer.py     # 100-Point Confluence Scorer, Zone A rules & 1H structural wall
├── execution/
│   ├── gold_risk.py             # Contract calculator, adaptive lot sizer & small-account risk cap
│   └── trade_manager.py         # 250ms Chameleon loop, T-90s Free-Roll lock & T-60s scratch
├── tests/                       # 95 automated test suites
└── ui/
    ├── server.py                # REST API endpoints, token verification & 4 FPS WebSocket hub
    └── index.html               # Dark-mode TradingView Lightweight Charts Web UI
```

---

## 🛡️ Risk Disclaimer

*Trading foreign exchange and commodities on margin carries a high level of risk and may not be suitable for all investors. Leverage can work against you as well as for you. Past performance is not indicative of future results. Always test thoroughly in demo or paper trading environments before deploying real capital.*
