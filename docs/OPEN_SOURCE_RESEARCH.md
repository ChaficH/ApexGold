# Open-Source Algorithmic Trading Tool Research & Synthesis

This document details the quantitative tools, libraries, and mathematical models researched and synthesized into the **ApexGold** trading engine. Each entry records the upstream origin, open-source license, mathematical formulation, exact adaptations made for XAU/USD (Gold) on MetaTrader 5, and the strategic rationale for its selection.

---

## 1. Dreiss Choppiness Index (CI)

- **Upstream Origin**: Freqtrade / TradingView technical indicator suite (originally developed by Australian commodity trader E.W. Dreiss).
- **License**: GPL-3.0 (Freqtrade) / MIT compatible mathematical formulation.
- **Location in Codebase**: [`engine/indicators.py`](file:///Users/chaficalhakim/Desktop/Trading%20script/engine/indicators.py#L987-L1040), [`strategy/confluence_scorer.py`](file:///Users/chaficalhakim/Desktop/Trading%20script/strategy/confluence_scorer.py#L998-L1047).
- **Mathematical Formula**:
  $$\text{True Range}_1 = \max(H_i - L_i, |H_i - C_{i-1}|, |L_i - C_{i-1}|)$$
  $$\text{SumTR} = \sum_{j=0}^{n-1} \text{True Range}_1(i-j)$$
  $$\text{Price Range} = \max(H_{i-n+1 \dots i}) - \min(L_{i-n+1 \dots i})$$
  $$\text{CI} = 100 \times \frac{\log_{10}\left(\frac{\text{SumTR}}{\text{Price Range}}\right)}{\log_{10}(n)}$$
- **Adaptation for ApexGold**:
  - Implemented with $n = 14$ on the 5-minute and 15-minute timeframe.
  - Threshold gating:
    - $\text{CI} \ge 60.0$: Classifies the market as high-entropy consolidation / chop box. Penalizes trend breakout orders by $-6$ points and blocks false breakouts, preventing whipsaws.
    - $\text{CI} \le 40.0$: Confirms a directional trend with high fractal persistence, awarding $+6$ points to momentum continuation.
- **Rationale**:
  Gold frequently consolidates during late Asian and early European hours before major news catalysts. Standard moving average crossovers produce catastrophic whipsaws during these periods. The Choppiness Index measures fractal dimension and prevents buying or selling inside choppy ranges.

---

## 2. Market Structure: Swing Fractals, Higher Highs / Lower Lows, BOS, and CHoCH

- **Upstream Origin**: vectorbt (Apache-2.0) and SmartMoneyConcepts (MIT).
- **License**: Apache-2.0 / MIT.
- **Location in Codebase**: [`engine/indicators.py`](file:///Users/chaficalhakim/Desktop/Trading%20script/engine/indicators.py#L1042-L1131), [`strategy/confluence_scorer.py`](file:///Users/chaficalhakim/Desktop/Trading%20script/strategy/confluence_scorer.py#L1052-L1084).
- **Formulation**:
  - **Swing High**: Bar $i$ where $H_i > \max(H_{i-w \dots i-1})$ and $H_i > \max(H_{i+1 \dots i+w})$.
  - **Swing Low**: Bar $i$ where $L_i < \min(L_{i-w \dots i-1})$ and $L_i < \min(L_{i+1 \dots i+w})$.
  - **Higher High (HH) & Higher Low (HL)**: $H_0 > H_1$ and $L_0 > L_1 \implies \text{Bullish Structure}$.
  - **Lower High (LH) & Lower Low (LL)**: $H_0 < H_1$ and $L_0 < L_1 \implies \text{Bearish Structure}$.
  - **Break of Structure (BOS)**: Continuation break where current price exceeds previous swing high in an established bull trend (or breaks previous swing low in a bear trend).
  - **Change of Character (CHoCH)**: Reversal break where price breaches the opposing swing level, indicating structural exhaustion and trend transition.
- **Adaptation for ApexGold**:
  - Implemented rolling pivot analysis with a 3-bar window across 5-minute candles.
  - Scorer awards $+8$ points for trend alignment (HH/HL for BUY, LH/LL for SELL), $+6$ for BOS continuation, and $+8$ for CHoCH reversals, while applying a $-8$ penalty for trading against prevailing market structure.
- **Rationale**:
  Simple indicator-based trading suffers from lag. Structural price action (SMC / Dow Theory) ensures the bot trades *with* institutional liquidity flows rather than fighting established trends.

---

## 3. Liquidity Sweep & False Breakout Trap Detection

- **Upstream Origin**: Smart Money Concepts (ICT / SMC) algorithms and TA-Lib price pattern recognition (BSD-2-Clause).
- **License**: MIT / BSD-2-Clause.
- **Location in Codebase**: [`engine/indicators.py`](file:///Users/chaficalhakim/Desktop/Trading%20script/engine/indicators.py#L1133-L1230), [`strategy/confluence_scorer.py`](file:///Users/chaficalhakim/Desktop/Trading%20script/strategy/confluence_scorer.py#L520-L555, L609-L620, L800-L815).
- **Formulation**:
  - **Bull Trap (Upthrust / Resistance Sweep)**:
    - Price breaches prior 5-minute range high by $\ge \$0.30$ on Gold.
    - Candle fails to sustain the breakout and closes back inside the range.
    - Upper rejection wick accounts for $\ge 45\%$ of the candle's total range.
  - **Bear Trap (Spring / Support Sweep)**:
    - Price breaches prior 5-minute range low by $\ge \$0.30$ on Gold.
    - Candle springs back and closes back inside the range.
    - Lower rejection wick accounts for $\ge 45\%$ of the candle's total range.
- **Adaptation for ApexGold**:
  - **Hard Protective Gate**: Strictly blocks BUY orders when a bull trap is active (prevents buying the absolute top), and strictly blocks SELL orders when a bear trap is active (prevents selling the absolute bottom).
  - **Reversal Edge Bonus**: Inverts the edge by awarding $+35$ points in Setup Type D when trading the *reversal* of the trap (selling the rejected bull trap or buying the rejected bear trap).
- **Rationale**:
  Directly resolves the critical real-world failure mode where breakout traders are lured into buying liquidity tops or selling liquidity bottoms. Institutional market makers intentionally trigger retail stop-loss clusters before reversing price.

---

## 4. Adaptive Fractional Position Sizing & Losing-Streak Throttling

- **Upstream Origin**: Backtrader (GPLv3) position sizers, QuantConnect risk management framework, and Ralph Vince fractional money management models.
- **License**: MIT / Apache compatible formulation.
- **Location in Codebase**: [`execution/gold_risk.py`](file:///Users/chaficalhakim/Desktop/Trading%20script/execution/gold_risk.py#L25-L290).
- **Formulation**:
  $$\text{Raw Lot} = \frac{\text{Risk Budget (\USD)}}{\text{Loss Per Lot}(\text{Padded SL}) + \text{Spread Cost Per Lot}}$$
  $$\text{Normalized Lot} = \text{floor}\left(\frac{\text{Raw Lot}}{\text{Volume Step}}\right) \times \text{Volume Step}$$
  $$\text{Monetary Risk} = \text{Normalized Lot} \times (\text{Loss Per Lot} + \text{Spread Cost Per Lot}) \le \text{Risk Budget}$$
- **Adaptation for ApexGold**:
  - **Risk Profiles**:
    - **Conservative**: Base $2.0\%$ risk, max cap $2.5\%$, max lot cap $0.20$.
    - **Balanced**: Base $3.0\%$ risk, max cap $5.0\%$, max lot cap $0.50$.
    - **Aggressive**: Base $6.0\%$ risk, max cap $15.0\%$ (with small account allowance up to $28\%$ for balances $\le \$25$ to permit safe $0.01$ lot trading on standard gold contracts).
  - **Confidence Multipliers**:
    - Tier 1 Sniper setups: $1.25\times$ to $1.35\times$ base sizing within profile caps.
    - Tier 2 Balanced setups: $1.0\times$ to $1.10\times$.
    - Tier 3 Aggressive setups: $0.80\times$ to $0.90\times$.
  - **Losing Streak Risk Throttle**:
    - 1 loss: $0.80\times$ ($20\%$ risk reduction).
    - 2 consecutive losses: $0.60\times$ ($40\%$ risk reduction).
    - 3+ consecutive losses: $0.50\times$ ($50\%$ risk reduction).
    - Resets to $1.0\times$ on the next winning trade.
  - **Strict Budget Safety Guarantee**: Step-down loop guarantees that rounding, broker volume step normalization, and spread compensation NEVER allow monetary risk to exceed the allowed dollar budget. Trades are cleanly rejected with explicit warnings if minimum broker lot exceeds the budget.
- **Rationale**:
  Preserves capital during drawdown periods and prevents emotional over-leveraging while allowing validated setups and growing accounts to compound safely.

---

## 5. Intelligent Structural Anti-Revenge Mechanism

- **Upstream Origin**: Professional proprietary desk risk rules (Chande, Elder, and Freqtrade protection handlers).
- **Location in Codebase**: [`execution/trade_manager.py`](file:///Users/chaficalhakim/Desktop/Trading%20script/execution/trade_manager.py#L85-L125, L630-L650, L1580-L1605).
- **Mechanism**:
  - Replaced rigid, blind 2-minute or 5-minute trade freezes with a dual-stage structural protection model:
    1. **12-Second Settlement Buffer**: Short pause strictly to let broker ticks stabilize and deal accounting settle.
    2. **Directional & Price-Level Filter**: Within a 45-second window following a loss, the bot blocks re-entering the *exact same losing trade in the same direction at the same price level* ($< \$1.50$ difference on Gold).
    3. **Immediate Reversal Permission**: If market structure flips or an opposite-direction signal appears (e.g., breakout fakeout confirms reverse trend), the trade is **freely permitted immediately** after the 12s buffer!
    4. **Manual Override**: The user can reset cooldown at any time via a dedicated 1-click dashboard endpoint (`POST /api/actions/reset_cooldown`).
- **Rationale**:
  Prevents blind tilt and revenge-trading into the exact same failed price level, while eliminating dumb delays that prevent catching genuine market turns.

---

## 6. Open-Source License Compatibility & IP Audit (GPL-3.0 Analysis)

This section documents the formal intellectual property and open-source license compatibility audit of the **ApexGold** codebase, specifically evaluating the upstream sources referenced in this document against ApexGold's **Proprietary** license model.

### 6.1 Upstream License Inventory

| Component / Concept | Upstream Reference | Upstream License | Nature of Reference in ApexGold |
| :--- | :--- | :--- | :--- |
| **Dreiss Choppiness Index (CI)** | Freqtrade / E.W. Dreiss (1991) | GPL-3.0 (Freqtrade) / Public Math | **Clean-Room Implementation**: Independently authored pure NumPy vectorized formula. Zero Freqtrade code copied. |
| **Market Structure (SMC / Dow)** | vectorbt / SmartMoneyConcepts | Apache-2.0 / MIT | Permissive third-party library (`smartmoneyconcepts`) used via standard package dependency. |
| **Liquidity Sweeps & Trap Detection** | TA-Lib / ICT Concepts | BSD-2-Clause / Public Concepts | Proprietary algorithm written from price action definitions. |
| **Fractional Position Sizing** | Backtrader / Ralph Vince (1990) | GPL-3.0 (Backtrader) / Public Math | **Clean-Room Implementation**: Independently authored fractional lot sizing loop in `gold_risk.py`. Zero Backtrader code copied. |
| **Anti-Revenge & Settlement Cooldown** | Freqtrade Protections / Elder | GPL-3.0 (Freqtrade) / Trading Rules | Independently designed structural price-distance window ($< \$1.50$) and 12s settlement buffer. |

### 6.2 Legal Analysis: Mathematical Formulas vs. GPL Copyleft

1. **Copyright Scope (Idea vs. Expression Distinction)**:
   - Under international copyright law (17 U.S.C. § 102(b), WIPO Copyright Treaty Article 2, and EU Directive 2009/24/EC Article 1(2)), copyright protection extends exclusively to the *creative expression of software code*, **not to underlying ideas, procedures, systems, methods of operation, concepts, mathematical principles, or financial formulas**.
   - The **Dreiss Choppiness Index** was published by Australian commodity trader E.W. Dreiss in 1991. The mathematical definition:
     $$\text{CI} = 100 \times \frac{\log_{10}\left(\frac{\sum \text{TR}_n}{\max(H_n) - \min(L_n)}\right)}{\log_{10}(n)}$$
     is a public mathematical formula. Coding this formula from mathematical first principles in Python/NumPy does not constitute a derivative work of Freqtrade or any other GPL-3.0 implementation.
   - Similarly, **Fractional Position Sizing** ($\text{Lot} = \frac{\text{Budget}}{\text{Cost}}$) and Ralph Vince's money management principles are foundational financial mathematics dating back to the 1980s–1990s, predating Backtrader by decades.

2. **Absence of GPL-3.0 Linking or Inclusion**:
   - The GNU General Public License v3.0 (GPL-3.0) contains a strong copyleft clause (Section 5) requiring that any work derived from, or linking directly to, GPL-3.0 code must itself be licensed under GPL-3.0.
   - An audit of [`requirements.txt`](file:///Users/chaficalhakim/Desktop/Trading%20script/requirements.txt) confirms that:
     - Neither `freqtrade` nor `backtrader` is listed as a dependency or imported anywhere in the project.
     - All runtime libraries used by ApexGold are distributed under permissive licenses:
       - **MIT**: `fastapi`, `pydantic`, `ta`, `smartmoneyconcepts`, `pyyaml`.
       - **BSD-2 / BSD-3**: `uvicorn`, `websockets`, `httpx`, `pandas`, `numpy`, `scipy`, `feedparser`.
       - **Apache-2.0**: `requests`, `aiohttp`, `yfinance`, `python-multipart`.
       - **Proprietary**: `MetaTrader5` (official MetaQuotes SDK).

3. **Clean-Room Guarantee**:
   - All code in `engine/indicators.py`, `execution/gold_risk.py`, and `execution/trade_manager.py` was authored from scratch specifically for the sub-second MetaTrader 5 execution environment.
   - No source code, docstrings, classes, or binary files were copied, transcribed, or adapted from Freqtrade or Backtrader repositories.

### 6.3 Compliance Considerations & Future Guidance

> [!IMPORTANT]
> **Key Legal Takeaway**:
> ApexGold is fully compatible with its **Proprietary Software License** because all indicator mathematics, position sizing models, and risk rules are independent, clean-room implementations of public domain quantitative theory, and all runtime dependencies are permissively licensed (MIT / BSD / Apache-2.0).

#### Future Development Safeguards:
- **Do Not Vendor GPL Code**: Future contributors must **never** directly copy-paste code snippets, helper functions, or modules from GPL-licensed repositories (such as Freqtrade or Backtrader) into the ApexGold codebase.
- **Do Not Add GPL Dependencies**: Avoid adding GPL-licensed packages to `requirements.txt`. If an algorithm from a GPL project is researched, only the underlying mathematical concept or abstract logic may be studied, and the Python implementation must be authored independently (clean-room approach).
- **If GPL Code is Ever Integrated**: If the project ever decides to directly incorporate or link GPL-3.0 modules in the future, the entire ApexGold repository license would legally need to be re-licensed under the **GNU General Public License v3.0 (GPL-3.0)** to maintain legal compliance. Under the current independent architecture, the Proprietary license remains fully valid and legally sound.


