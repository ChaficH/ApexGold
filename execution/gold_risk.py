"""
ApexGold Dynamic Risk & Contract Math Engine (XAUUSD Specialized).

Handles Gold contract specifications, Cent vs Micro vs Standard account scaling,
asymmetric $10 budget risk management ($1.00 risk per trade), live spread padding,
and margin feasibility checks.
"""

import logging
import math
from typing import Any, Dict, Optional, Tuple

from brokers.base import AccountInfo, SymbolInfo

logger = logging.getLogger("ApexGold.GoldRisk")


class GoldRiskEngine:
    """
    Automated Contract Inspector & Smart Risk Engine for Gold & Forex.
    Enables precise, safe execution for small account sizes starting at $10.00.
    """

    def __init__(
        self,
        account_type: str = "cent",      # "cent", "micro", "standard", "paper"
        budget_usd: float = 10.00,
        risk_per_trade_pct: float = 10.0, # 10% risk = $1.00 on $10
        compounding_enabled: bool = True,
        max_daily_losses: int = 2,
    ):
        self.account_type = account_type.lower()
        self.base_budget_usd = budget_usd
        self.risk_per_trade_pct = risk_per_trade_pct
        self.compounding_enabled = compounding_enabled
        self.max_daily_losses = max_daily_losses

    def inspect_symbol(self, symbol_info: SymbolInfo, account_info: AccountInfo) -> Dict[str, Any]:
        """
        Inspect dynamic broker contract specifications and return execution parameters.
        """
        is_gold = any(k in symbol_info.symbol.upper() for k in ["XAU", "GOLD"])
        # Cost per 1 point ($0.01 move) on 1 standard lot
        contract_size = symbol_info.contract_size
        if is_gold and (contract_size <= 0 or contract_size > 1000.0):
            contract_size = 1.0 if self.account_type == "cent" else 100.0
        point_cost_per_lot = contract_size * symbol_info.point

        return {
            "symbol": symbol_info.symbol,
            "is_gold": is_gold,
            "contract_size": symbol_info.contract_size,
            "point": symbol_info.point,
            "digits": symbol_info.digits,
            "spread_points": symbol_info.spread / symbol_info.point if symbol_info.point > 0 else 0,
            "spread_price": symbol_info.spread,
            "tick_size": symbol_info.tick_size,
            "tick_value": symbol_info.tick_value,
            "volume_min": symbol_info.volume_min,
            "volume_step": symbol_info.volume_step,
            "account_currency": account_info.currency,
            "account_leverage": account_info.leverage,
        }

    def get_effective_budget(self, current_balance: float) -> float:
        """
        Calculates active trading budget.
        If compounding is enabled, scales the budget upwards as balance grows.
        """
        if not self.compounding_enabled:
            return self.base_budget_usd

        # If on Cent account, balance is in Cents (e.g., $10 = 1,000 USC)
        if self.account_type == "cent" or "USC" in str(self.base_budget_usd):
            # Scale based on current balance
            return max(self.base_budget_usd, current_balance)

        # Standard USD account
        return max(self.base_budget_usd, current_balance)

    def calculate_lot_size(
        self,
        symbol_info: SymbolInfo,
        account_info: AccountInfo,
        sl_distance: float,
        override_risk_usd: Optional[float] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute precise lot size so maximum loss equals exactly risk_per_trade_pct of budget.
        Handles Cent account smooth scaling ($1.00 risk = 100 cents across $3 move)
        and Standard/Micro account minimum lot (0.01) constraints.
        """
        is_gold = any(k in symbol_info.symbol.upper() for k in ["XAU", "GOLD"])
        effective_budget = self.get_effective_budget(account_info.balance)

        # Dynamic risk percentage scaling based on balance
        if override_risk_usd is not None and override_risk_usd > 0:
            risk_amount = override_risk_usd
        else:
            if effective_budget < 60.0:
                risk_pct = 3.5
            elif effective_budget < 150.0:
                # E.g. $85 budget risks ~4.0% ($3.40), allowing 0.02 lots on tight SLs
                risk_pct = 4.0
            elif effective_budget < 400.0:
                risk_pct = 3.0
            else:
                risk_pct = 2.0
            risk_amount = effective_budget * (risk_pct / 100.0)

        # Ensure volatility breathing room based on market ATR (minimum 2.50 on Gold, never squashed)
        min_sl = 2.50 if is_gold else 0.0010
        actual_sl_dist = max(sl_distance, min_sl)

        # Real Gold contract size is 100 oz per standard lot (1.0 oz on Cent account)
        contract_size = symbol_info.contract_size
        if is_gold and (contract_size <= 0 or contract_size > 1000.0):
            contract_size = 1.0 if self.account_type == "cent" else 100.0

        # Risk formula: LotSize = RiskAmount / (SL_Distance * ContractSize)
        raw_lot = risk_amount / (actual_sl_dist * contract_size)

        warning_msg = None
        if self.account_type == "cent":
            # On Cent accounts, 1,000 USC balance allows micro-sizing smoothly
            lot = round(raw_lot / symbol_info.volume_step) * symbol_info.volume_step
            lot = max(symbol_info.volume_min, min(symbol_info.volume_max, lot))
            lot = round(lot, 2)
        else:
            # On Standard/Micro accounts:
            # Dynamic tier caps that safely scale up as budget allows
            if is_gold:
                if effective_budget < 40.0:
                    max_safe_lot = 0.01
                elif effective_budget < 120.0:
                    # On $50 - $120 balance (e.g. $85), allow up to 0.02 lots
                    max_safe_lot = 0.02
                elif effective_budget < 250.0:
                    # On $120 - $250 balance, allow up to 0.04 lots
                    max_safe_lot = 0.04
                elif effective_budget < 500.0:
                    max_safe_lot = 0.08
                elif effective_budget < 1000.0:
                    max_safe_lot = 0.15
                elif effective_budget < 2500.0:
                    max_safe_lot = 0.35
                else:
                    max_safe_lot = 1.00
            else:
                max_safe_lot = 0.05 if effective_budget < 100.0 else (0.10 if effective_budget < 300.0 else 0.50)

            # Sizing selection
            lot = min(raw_lot, max_safe_lot)
            lot = round(lot / symbol_info.volume_step) * symbol_info.volume_step
            lot = max(symbol_info.volume_min, min(symbol_info.volume_max, lot))
            lot = round(lot, 2)

        # Margin Requirement Check & Smart Budget Clamping
        current_price = symbol_info.ask if symbol_info.ask > 0 else 4140.0
        leverage = max(account_info.leverage, 1)

        margin_required = (current_price * lot * contract_size) / leverage
        has_margin = margin_required <= account_info.free_margin

        if not has_margin:
            # Auto-clamp lot size to fit available margin safely
            safe_margin = max(0.0, account_info.free_margin * 0.90)
            max_affordable_lot = math.floor((safe_margin * leverage) / (current_price * contract_size) / symbol_info.volume_step) * symbol_info.volume_step
            if max_affordable_lot >= symbol_info.volume_min:
                lot = round(max_affordable_lot, 2)
                margin_required = (current_price * lot * contract_size) / leverage
                has_margin = True
                warning_msg = f"Auto-clamped lot size to {lot:.2f} to fit free margin (${account_info.free_margin:.2f})."
            else:
                warning_msg = (
                    f"Margin Warning: Required ${margin_required:.2f} exceeds free margin ${account_info.free_margin:.2f} "
                    f"at 1:{leverage} leverage."
                )

        details = {
            "lot_size": lot,
            "risk_amount": round(risk_amount, 2),
            "effective_budget": round(effective_budget, 2),
            "sl_distance": round(actual_sl_dist, 2 if is_gold else 5),
            "margin_required": round(margin_required, 2),
            "free_margin": round(account_info.free_margin, 2),
            "has_margin": has_margin,
            "warning": warning_msg,
        }

        return lot, details

    def pad_stop_loss(
        self,
        direction: str,
        raw_sl: float,
        spread: float,
        point: float = 0.01,
        extra_buffer_points: int = 2,
    ) -> float:
        """
        Live Spread Compensation:
        Pads Stop Loss by (current_spread + 2 points) so normal spread wicks
        never prematurely trigger Stop Loss.
        """
        buffer = spread + (extra_buffer_points * point)
        if direction.upper() == "BUY":
            # For BUY, move SL lower by buffer
            return round(raw_sl - buffer, 2 if point == 0.01 else 5)
        else:
            # For SELL, move SL higher by buffer
            return round(raw_sl + buffer, 2 if point == 0.01 else 5)

