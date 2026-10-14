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
        Calculates active trading budget safely.
        Budget can NEVER exceed the actual available balance.
        """
        safe_bal = max(0.0, current_balance)
        if not self.compounding_enabled:
            return min(self.base_budget_usd, safe_bal)
        return safe_bal

    def calculate_monetary_risk(
        self,
        symbol_info: SymbolInfo,
        lot: float,
        sl_distance: float,
    ) -> float:
        """Calculate total monetary risk including spread for a given lot and SL distance."""
        is_gold = any(k in symbol_info.symbol.upper() for k in ["XAU", "GOLD"])
        contract_size = symbol_info.contract_size
        if is_gold and (contract_size <= 0 or contract_size > 1000.0):
            contract_size = 1.0 if self.account_type == "cent" else 100.0

        if symbol_info.tick_size > 0 and symbol_info.tick_value > 0:
            loss_per_lot = (sl_distance / symbol_info.tick_size) * symbol_info.tick_value
            spread_cost = (symbol_info.spread / symbol_info.tick_size) * symbol_info.tick_value
        else:
            loss_per_lot = sl_distance * contract_size
            spread_cost = symbol_info.spread * contract_size

        return round(lot * (loss_per_lot + spread_cost), 2)

    def calculate_lot_size(
        self,
        symbol_info: SymbolInfo,
        account_info: AccountInfo,
        sl_distance: float,
        override_risk_usd: Optional[float] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute precise lot size so maximum loss equals configured risk_per_trade_pct.
        Rejects trades when minimum lot size exceeds permitted risk.
        Never automatically increases risk above the configured risk budget.
        """
        is_gold = any(k in symbol_info.symbol.upper() for k in ["XAU", "GOLD"])
        effective_budget = self.get_effective_budget(account_info.balance)

        if override_risk_usd is not None and override_risk_usd > 0:
            risk_amount = override_risk_usd
        else:
            risk_amount = effective_budget * (self.risk_per_trade_pct / 100.0)

        # Minimum SL distance for volatility breathing room
        min_sl = 2.50 if is_gold else 0.0010
        actual_sl_dist = max(sl_distance, min_sl)

        contract_size = symbol_info.contract_size
        if is_gold and (contract_size <= 0 or contract_size > 1000.0):
            contract_size = 1.0 if self.account_type == "cent" else 100.0

        # Calculate loss and execution costs per 1.0 standard lot
        if symbol_info.tick_size > 0 and symbol_info.tick_value > 0:
            ticks = actual_sl_dist / symbol_info.tick_size
            loss_per_lot = ticks * symbol_info.tick_value
            spread_cost_per_lot = (symbol_info.spread / symbol_info.tick_size) * symbol_info.tick_value
        else:
            loss_per_lot = actual_sl_dist * contract_size
            spread_cost_per_lot = symbol_info.spread * contract_size

        cost_per_lot = loss_per_lot + spread_cost_per_lot
        if cost_per_lot <= 0:
            return 0.0, {
                "lot_size": 0.0,
                "has_margin": False,
                "warning": "Invalid SL distance or contract specification.",
                "risk_amount": round(risk_amount, 2),
                "effective_budget": round(effective_budget, 2),
            }

        raw_lot = risk_amount / cost_per_lot

        v_step = symbol_info.volume_step if symbol_info.volume_step > 0 else 0.01
        v_min = symbol_info.volume_min if symbol_info.volume_min > 0 else 0.01
        v_max = symbol_info.volume_max if symbol_info.volume_max > 0 else 100.0

        # Step down to avoid exceeding risk
        lot = math.floor(raw_lot / v_step) * v_step
        min_lot_risk = v_min * cost_per_lot

        # Strict risk check: If minimum volume risk exceeds risk budget, reject the trade
        if lot < v_min:
            if min_lot_risk > (risk_amount * 1.05):
                warning_msg = (
                    f"Minimum lot ({v_min}) monetary risk (${min_lot_risk:.2f}) exceeds allowed "
                    f"risk budget (${risk_amount:.2f}, {self.risk_per_trade_pct}% of ${effective_budget:.2f}). "
                    f"Trade rejected for capital safety."
                )
                logger.warning(warning_msg)
                return 0.0, {
                    "lot_size": 0.0,
                    "risk_amount": round(risk_amount, 2),
                    "min_lot_risk": round(min_lot_risk, 2),
                    "effective_budget": round(effective_budget, 2),
                    "sl_distance": round(actual_sl_dist, 2 if is_gold else 5),
                    "margin_required": 0.0,
                    "free_margin": round(account_info.free_margin, 2),
                    "has_margin": False,
                    "warning": warning_msg,
                }
            else:
                lot = v_min

        lot = min(v_max, lot)
        lot = round(lot, 2)

        # Margin Requirement Check
        current_price = symbol_info.ask if symbol_info.ask > 0 else 4140.0
        leverage = max(account_info.leverage, 1)
        margin_required = (current_price * lot * contract_size) / leverage
        has_margin = margin_required <= account_info.free_margin

        warning_msg = None
        if not has_margin:
            safe_margin = max(0.0, account_info.free_margin * 0.90)
            max_affordable_lot = math.floor((safe_margin * leverage) / (current_price * contract_size) / v_step) * v_step
            if max_affordable_lot >= v_min and (max_affordable_lot * cost_per_lot <= risk_amount * 1.05):
                lot = round(max_affordable_lot, 2)
                margin_required = (current_price * lot * contract_size) / leverage
                has_margin = True
                warning_msg = f"Auto-clamped lot size to {lot:.2f} to fit free margin (${account_info.free_margin:.2f})."
            else:
                warning_msg = (
                    f"Margin Warning: Required ${margin_required:.2f} exceeds free margin ${account_info.free_margin:.2f} "
                    f"at 1:{leverage} leverage."
                )
                return 0.0, {
                    "lot_size": 0.0,
                    "risk_amount": round(risk_amount, 2),
                    "effective_budget": round(effective_budget, 2),
                    "sl_distance": round(actual_sl_dist, 2 if is_gold else 5),
                    "margin_required": round(margin_required, 2),
                    "free_margin": round(account_info.free_margin, 2),
                    "has_margin": False,
                    "warning": warning_msg,
                }

        actual_monetary_risk = round(lot * cost_per_lot, 2)

        details = {
            "lot_size": lot,
            "risk_amount": round(risk_amount, 2),
            "actual_monetary_risk": actual_monetary_risk,
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

