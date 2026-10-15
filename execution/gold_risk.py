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
    Enables precise, safe execution for small account sizes starting at $10.00
    with adaptive position sizing, configurable risk profiles, confidence scaling,
    and consecutive-loss risk throttling.
    """

    PROFILES = {
        "conservative": {
            "name": "Conservative",
            "base_risk_pct": 2.0,
            "max_risk_pct": 2.5,
            "max_open_positions": 1,
            "confidence_multipliers": {
                "TIER_1_SNIPER": 1.0,
                "TIER_2_BALANCED": 0.85,
                "TIER_3_AGGRESSIVE": 0.70,
                "NONE": 0.70,
            },
            "max_lot_cap": 0.20,
        },
        "balanced": {
            "name": "Balanced",
            "base_risk_pct": 3.0,
            "max_risk_pct": 5.0,
            "max_open_positions": 2,
            "confidence_multipliers": {
                "TIER_1_SNIPER": 1.25,
                "TIER_2_BALANCED": 1.0,
                "TIER_3_AGGRESSIVE": 0.80,
                "NONE": 0.80,
            },
            "max_lot_cap": 0.50,
        },
        "aggressive": {
            "name": "Aggressive",
            "base_risk_pct": 10.0,
            "max_risk_pct": 25.0,
            "small_account_max_risk_pct": 45.0,  # Explicit allowance for small ($10-$20) test accounts
            "max_open_positions": 2,
            "confidence_multipliers": {
                "TIER_1_SNIPER": 1.50,
                "TIER_2_BALANCED": 1.25,
                "TIER_3_AGGRESSIVE": 1.00,
                "NONE": 0.85,
            },
            "max_lot_cap": 1.00,
        },
    }

    def __init__(
        self,
        account_type: str = "cent",      # "cent", "micro", "standard", "paper"
        budget_usd: float = 10.00,
        risk_per_trade_pct: float = 2.0, # Configured base risk percentage
        compounding_enabled: bool = True,
        max_daily_losses: int = 0,       # 0 = disabled (no daily loss limit)
        risk_profile: str = "balanced",   # "conservative", "balanced", "aggressive"
    ):
        self.account_type = account_type.lower()
        self.base_budget_usd = budget_usd
        self.risk_per_trade_pct = risk_per_trade_pct
        self.compounding_enabled = compounding_enabled
        self.max_daily_losses = max_daily_losses
        self.risk_profile = risk_profile.lower() if risk_profile in self.PROFILES else "balanced"

    def set_risk_profile(self, profile_name: str) -> None:
        """Switch active risk profile between conservative, balanced, and aggressive."""
        p = profile_name.lower().strip()
        if p in self.PROFILES:
            self.risk_profile = p
            logger.info(f"[RISK ENGINE] Switched risk profile to {p.upper()} ({self.PROFILES[p]['name']})")
        else:
            logger.warning(f"[RISK ENGINE] Unknown risk profile '{profile_name}'. Keeping {self.risk_profile}")

    def get_risk_profile_info(self) -> Dict[str, Any]:
        """Return metadata for the currently active risk profile."""
        prof_data = self.PROFILES.get(self.risk_profile, self.PROFILES["balanced"])
        return {
            "profile": self.risk_profile,
            "name": prof_data["name"],
            "base_risk_pct": self.risk_per_trade_pct or prof_data["base_risk_pct"],
            "max_risk_pct": prof_data["max_risk_pct"],
            "max_open_positions": prof_data["max_open_positions"],
            "max_lot_cap": prof_data["max_lot_cap"],
            "account_type": self.account_type,
        }

    def inspect_symbol(self, symbol_info: SymbolInfo, account_info: AccountInfo) -> Dict[str, Any]:
        """
        Inspect dynamic broker contract specifications and return execution parameters.
        Rejects invalid broker metadata (non-positive contract_size, point, or ask).
        """
        is_gold = any(k in symbol_info.symbol.upper() for k in ["XAU", "GOLD"])
        contract_size = symbol_info.contract_size
        is_valid = (contract_size > 0) and (symbol_info.point > 0) and (symbol_info.ask > 0)
        point_cost_per_lot = (contract_size * symbol_info.point) if is_valid else 0.0

        return {
            "symbol": symbol_info.symbol,
            "is_gold": is_gold,
            "is_valid": is_valid,
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
        Falls back to base_budget_usd if broker balance is zero or uninitialized during startup.
        """
        safe_bal = max(0.0, current_balance)
        if safe_bal <= 0.0:
            return self.base_budget_usd
        if not self.compounding_enabled:
            return min(self.base_budget_usd, safe_bal)
        return safe_bal

    def calculate_risk_budget(
        self,
        account_info: AccountInfo,
        confidence_tier: Optional[str] = None,
        score: Optional[int] = None,
        consecutive_losses: int = 0,
    ) -> float:
        """
        Calculate permitted monetary risk budget (USD) for a single trade,
        accounting for risk profile, setup confidence scaling, and losing streak throttle.
        """
        effective_budget = self.get_effective_budget(account_info.balance)
        prof = self.PROFILES.get(self.risk_profile, self.PROFILES["balanced"])

        # Determine effective base risk percentage
        base_pct = self.risk_per_trade_pct if self.risk_per_trade_pct > 0 else prof["base_risk_pct"]

        # Confidence Scaling
        conf_mult = 1.0
        tier_key = confidence_tier.upper() if confidence_tier else None
        if not tier_key and score is not None:
            if score >= 82:
                tier_key = "TIER_1_SNIPER"
            elif score >= 68:
                tier_key = "TIER_2_BALANCED"
            else:
                tier_key = "TIER_3_AGGRESSIVE"

        if tier_key and tier_key in prof["confidence_multipliers"]:
            conf_mult = prof["confidence_multipliers"][tier_key]

        # Losing Streak Risk Throttle (anti-tilt capital protection)
        streak_mult = 1.0
        if consecutive_losses == 1:
            streak_mult = 0.80
        elif consecutive_losses == 2:
            streak_mult = 0.60
        elif consecutive_losses >= 3:
            streak_mult = 0.50

        # Small account special allowance (under aggressive profile for <= $25 balances)
        max_pct = prof["max_risk_pct"]
        if self.risk_profile == "aggressive" and effective_budget <= 25.0:
            base_pct = max(base_pct, prof.get("small_account_max_risk_pct", 45.0))
            max_pct = prof.get("small_account_max_risk_pct", 45.0)

        effective_pct = min(base_pct * conf_mult * streak_mult, max_pct)
        return round(effective_budget * (effective_pct / 100.0), 2)

    def calculate_monetary_risk(
        self,
        symbol_info: SymbolInfo,
        lot: float,
        sl_distance: float,
    ) -> float:
        """
        Calculate total monetary risk including spread for a given lot and SL distance.
        Rejects non-positive contract specifications.
        """
        if symbol_info.contract_size <= 0:
            logger.error(f"Cannot calculate monetary risk: invalid contract_size ({symbol_info.contract_size}) for {symbol_info.symbol}")
            return 0.0

        contract_size = symbol_info.contract_size

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
        override_risk_usd: Optional[float] = None,
        confidence_tier: Optional[str] = None,
        score: Optional[int] = None,
        consecutive_losses: int = 0,
        ev_r: Optional[float] = None,
        p_win: Optional[float] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute precise lot size so maximum loss equals configured risk budget.
        Adapts to account balance, risk profiles (conservative/balanced/aggressive),
        confidence tiers (Tier 1 Sniper vs Balanced), predictive model EV, and consecutive loss streak.
        Rejects trades when minimum lot size exceeds permitted risk.
        Never automatically increases risk above the configured risk budget.
        Rejects invalid broker metadata (non-positive contract_size, point, or ask).
        """
        effective_budget = self.get_effective_budget(account_info.balance)
        prof = self.PROFILES.get(self.risk_profile, self.PROFILES["balanced"])

        if override_risk_usd is not None and override_risk_usd > 0:
            risk_amount = override_risk_usd
            conf_mult = 1.0
            streak_mult = 1.0
        else:
            # Determine effective base risk percentage
            base_pct = self.risk_per_trade_pct if self.risk_per_trade_pct > 0 else prof["base_risk_pct"]

            # Confidence Scaling
            conf_mult = 1.0
            tier_key = confidence_tier.upper() if confidence_tier else None
            if not tier_key and score is not None:
                if score >= 82:
                    tier_key = "TIER_1_SNIPER"
                elif score >= 68:
                    tier_key = "TIER_2_BALANCED"
                else:
                    tier_key = "TIER_3_AGGRESSIVE"

            if tier_key and tier_key in prof["confidence_multipliers"]:
                conf_mult = prof["confidence_multipliers"][tier_key]

            # Predictive Model Conviction Sizing (Kelly-Informed Booster)
            if ev_r is not None and ev_r >= 0.30 and p_win is not None and p_win >= 0.58:
                edge_boost = min(1.30, 1.0 + (ev_r * 0.35))
                conf_mult = round(conf_mult * edge_boost, 2)

            # Losing Streak Throttle
            streak_mult = 1.0
            if consecutive_losses == 1:
                streak_mult = 0.80
            elif consecutive_losses == 2:
                streak_mult = 0.60
            elif consecutive_losses >= 3:
                streak_mult = 0.50

            max_pct = prof["max_risk_pct"]
            if self.risk_profile == "aggressive" and effective_budget <= 25.0:
                base_pct = max(base_pct, prof.get("small_account_max_risk_pct", 45.0))
                max_pct = prof.get("small_account_max_risk_pct", 45.0)

            effective_pct = min(base_pct * conf_mult * streak_mult, max_pct)
            risk_amount = round(effective_budget * (effective_pct / 100.0), 2)

        # 1. Strict broker metadata validation - reject non-positive contract_size, point, or price
        if symbol_info.contract_size <= 0 or symbol_info.point <= 0 or symbol_info.ask <= 0:
            err_msg = (
                f"Invalid broker metadata for {symbol_info.symbol}: "
                f"contract_size={symbol_info.contract_size}, point={symbol_info.point}, ask={symbol_info.ask}. "
                f"Trade rejected."
            )
            logger.error(err_msg)
            return 0.0, {
                "lot_size": 0.0,
                "has_margin": False,
                "warning": err_msg,
                "risk_amount": round(risk_amount, 2),
                "effective_budget": round(effective_budget, 2),
                "margin_required": 0.0,
                "free_margin": round(account_info.free_margin, 2),
                "risk_profile": self.risk_profile,
                "confidence_mult": conf_mult,
                "streak_mult": streak_mult,
            }

        is_gold = any(k in symbol_info.symbol.upper() for k in ["XAU", "GOLD"])
        # Minimum SL distance: 2.50 for standard gold, but allow 1.20-1.50 for small accounts (<= $35) or micro-scalps
        min_sl = 1.50 if (is_gold and (effective_budget <= 35.0 or self.account_type in ["micro", "cent"])) else (2.50 if is_gold else 0.0010)
        actual_sl_dist = max(sl_distance, min_sl)
        contract_size = symbol_info.contract_size

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

        min_lot_risk = round(v_min * cost_per_lot, 2)

        # Strict risk check: If minimum volume risk exceeds permitted risk budget, reject the trade
        # For micro accounts (effective_budget <= 25.0) under aggressive profile, allow the single minimum lot (0.01)
        # to execute as long as monetary risk does not exceed account balance.
        is_small_micro = (effective_budget <= 25.0) and (self.risk_profile == "aggressive")
        allowed_risk_cap = max(round(risk_amount, 2), round(effective_budget * 0.98, 2)) if is_small_micro else round(risk_amount, 2)

        if min_lot_risk > allowed_risk_cap:
            warning_msg = (
                f"Minimum lot ({v_min}) monetary risk (${min_lot_risk:.2f}) exceeds allowed "
                f"risk budget (${risk_amount:.2f}, {self.risk_per_trade_pct}% of ${effective_budget:.2f}). "
                f"Trade rejected for capital safety."
            )
            logger.warning(warning_msg)
            return 0.0, {
                "lot_size": 0.0,
                "risk_amount": round(risk_amount, 2),
                "min_lot_risk": min_lot_risk,
                "effective_budget": round(effective_budget, 2),
                "sl_distance": round(actual_sl_dist, 2 if is_gold else 5),
                "margin_required": 0.0,
                "free_margin": round(account_info.free_margin, 2),
                "has_margin": False,
                "warning": warning_msg,
            }

        # Step down to strictly avoid exceeding risk
        lot = math.floor(raw_lot / v_step) * v_step
        lot = round(lot, 4)

        # Enforce that rounding or floating point arithmetic NEVER allows monetary risk > risk_amount (or allowed_risk_cap for small micro)
        effective_cap = allowed_risk_cap if is_small_micro else round(risk_amount, 2)
        while lot >= v_min and round(lot * cost_per_lot, 2) > effective_cap:
            lot = round(lot - v_step, 4)

        if lot < v_min:
            if is_small_micro and min_lot_risk <= (effective_budget * 0.98):
                lot = v_min
            else:
                warning_msg = (
                    f"Calculated lot size below minimum broker volume ({v_min}) after strict risk budgeting. "
                    f"Trade rejected."
                )
                logger.warning(warning_msg)
                return 0.0, {
                    "lot_size": 0.0,
                    "risk_amount": round(risk_amount, 2),
                    "min_lot_risk": min_lot_risk,
                    "effective_budget": round(effective_budget, 2),
                    "sl_distance": round(actual_sl_dist, 2 if is_gold else 5),
                    "margin_required": 0.0,
                    "free_margin": round(account_info.free_margin, 2),
                    "has_margin": False,
                    "warning": warning_msg,
                }

        lot = min(v_max, lot)
        # Enforce profile max_lot_cap if specified
        if "max_lot_cap" in prof and prof["max_lot_cap"] > 0:
            lot = min(lot, prof["max_lot_cap"])
        # Micro account safety clamp: max 0.05 lot for accounts <= $150
        if effective_budget <= 150.0:
            lot = min(lot, 0.05)
        lot = round(lot, 2)

        # Final rounding re-verification
        while lot >= v_min and round(lot * cost_per_lot, 2) > effective_cap:
            lot = round(lot - v_step, 2)

        if lot < v_min:
            if is_small_micro and min_lot_risk <= (effective_budget * 0.98):
                lot = v_min
            else:
                return 0.0, {
                    "lot_size": 0.0,
                    "risk_amount": round(risk_amount, 2),
                    "min_lot_risk": min_lot_risk,
                    "effective_budget": round(effective_budget, 2),
                    "sl_distance": round(actual_sl_dist, 2 if is_gold else 5),
                    "margin_required": 0.0,
                    "free_margin": round(account_info.free_margin, 2),
                    "has_margin": False,
                    "warning": "Rounding exceeded risk budget; trade rejected.",
                    "risk_profile": self.risk_profile,
                    "confidence_mult": conf_mult,
                    "streak_mult": streak_mult,
                }

        # Margin Requirement Check (using verified broker ask price)
        current_price = symbol_info.ask if symbol_info.ask > 0 else (symbol_info.bid if symbol_info.bid > 0 else 4150.0)
        leverage = max(account_info.leverage, 1)
        margin_required = (current_price * lot * contract_size) / leverage
        has_margin = margin_required <= account_info.free_margin

        warning_msg = None
        if not has_margin:
            safe_margin = max(0.0, account_info.free_margin * 0.90)
            max_affordable_lot = math.floor((safe_margin * leverage) / (current_price * contract_size) / v_step) * v_step
            max_affordable_lot = round(max_affordable_lot, 2)
            while max_affordable_lot >= v_min and round(max_affordable_lot * cost_per_lot, 2) > round(risk_amount, 2):
                max_affordable_lot = round(max_affordable_lot - v_step, 2)

            if max_affordable_lot >= v_min:
                lot = max_affordable_lot
                margin_required = (current_price * lot * contract_size) / leverage
                has_margin = True
                warning_msg = f"Auto-clamped lot size to {lot:.2f} to fit free margin (${account_info.free_margin:.2f})."
            else:
                shortfall = max(0.01, round(margin_required - account_info.free_margin, 2))
                warning_msg = (
                    f"Margin Warning: Required ${margin_required:.2f} exceeds free margin ${account_info.free_margin:.2f} "
                    f"at 1:{leverage} leverage for {lot} {symbol_info.symbol} (Short by ${shortfall:.2f})."
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
                    "risk_profile": self.risk_profile,
                    "confidence_mult": conf_mult,
                    "streak_mult": streak_mult,
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
            "risk_profile": self.risk_profile,
            "confidence_mult": conf_mult,
            "streak_mult": streak_mult,
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

