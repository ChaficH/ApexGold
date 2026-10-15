"""
ApexGold Adaptive "Chameleon" Trade Lifecycle Manager & Transmission Controller.

Monitors open positions every 250ms to enforce:
1. Dynamic Regime Morphing:
   - Auto-Upgrade: Scalp -> Big Wave Runner on Institutional Momentum Surge (>2.2x velocity, expanding candles).
   - Auto-Downgrade: Runner -> Instant Market Profit Bank on Intraday Exhaustion/Rejection.
2. Fast Hit-and-Run Scalp Logic:
   - Early Breakeven lock at +0.45R.
   - 4-Second Tick Velocity Stall Exit ("Take the money and run" when order flow dies).
3. Multi-Stage Ratchet Stop-Loss Floor (+0.45R/+0.6R Breakeven, +0.5R/+0.8R Locks, +1.0R Locks, +2.0R Dynamic Trailing).
4. Dead-Trade Scratch Timer (closes stalled trades after 12 minutes to free capital).
5. USD Correlation Guard & Daily Drawdown Circuit Breaker.
"""

import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from brokers.base import (
    AbstractBrokerAdapter,
    OrderResult,
    OrderType,
    Position,
    PositionType,
)
from engine.data_hub import DataHub
from engine.indicators import TechnicalIndicators
from engine.news_reactor import NewsReactor
from execution.gold_risk import GoldRiskEngine
from strategy.confluence_scorer import ConfluenceSignal

logger = logging.getLogger("ApexGold.TradeManager")


@dataclass
class ManagedPosition:
    """State tracking for an active position undergoing adaptive management."""
    ticket: Union[int, str]
    symbol: str
    direction: str              # "BUY" or "SELL"
    volume: float
    open_price: float
    sl: float
    tp: float
    entry_time: float
    risk_r_dist: float          # Distance in price representing 1.0R
    initial_regime: str = "ADAPTIVE_RUNNER"         # "ADAPTIVE_SCALP" or "ADAPTIVE_RUNNER"
    current_regime: str = "ADAPTIVE_RUNNER"         # May morph in real time
    sub_regime: str = "WAVE_RUNNER" # "FAST_SCALP", "WAVE_RUNNER", "DEFENSIVE_SHIELD"
    be_lock_r: float = 0.60     # +0.45R for fast scalp, +0.6R standard
    stall_exit_seconds: float = 90.0 # 4.0s for fast scalp, 90.0s for runner
    highest_r_reached: float = 0.0
    ratchet_stage: int = 0      # 0=init, 1=BE, 2=+0.5R/+0.8R, 3=+1.0R, 4=trailing (+2.0R+)
    entry_velocity: float = 1.0
    stall_start_time: float = 0.0
    last_mod_time: float = 0.0
    close_reason: Optional[str] = None
    is_closed: bool = False
    profit: float = 0.0
    current_price: float = 0.0
    origin: str = "BOT (AUTO)"
    is_user_manual: bool = False
    is_freeroll_locked: bool = False
    is_closing: bool = False
    close_trigger_price: float = 0.0
    r_multiple: float = 0.0
    exit_evaluation: str = "HOLD_AND_EXPAND"     # "HOLD_AND_EXPAND", "TIGHTEN_AND_PROTECT", "HARVEST_PROFIT_NOW"
    exit_verdict_detail: str = "Market structure intact. Holding for expansion."
    peak_r_giveback_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TradeManager:
    """
    Sub-Second Adaptive Trade Lifecycle & Position Risk Controller.
    Executes trades and runs a continuous 250ms 'Follow-the-Market' management loop.
    """

    def __init__(
        self,
        broker: AbstractBrokerAdapter,
        data_hub: Optional[DataHub] = None,
        news_reactor: Optional[NewsReactor] = None,
        risk_engine: Optional[GoldRiskEngine] = None,
        correlation_guard: bool = True,
        max_daily_losses: int = 0,       # 0 = disabled (no daily loss limit)
        max_consecutive_losses: int = 0, # 0 = disabled (no consecutive loss pause unless configured)
        max_daily_drawdown_pct: float = 25.0,
        max_total_open_risk_pct: float = 10.0,
        max_open_positions: int = 1,
        history_store: Optional[Any] = None,
        manage_manual_positions: bool = False,
        post_loss_buffer_seconds: float = 12.0,
        anti_revenge_window_seconds: float = 45.0,
        predictive_scratch_enabled: bool = True,
        micro_budget_loss_cap_pct: float = 15.0,
        max_mitigation_loss_r: float = 0.40,
        min_hold_time_seconds: float = 90.0,
        scorer: Optional[Any] = None,
        circuit_breaker_enabled: bool = True,
    ):
        self.broker = broker
        self.data_hub = data_hub
        self.news_reactor = news_reactor if news_reactor is not None else NewsReactor(enabled=False)
        self.risk_engine = risk_engine if risk_engine is not None else GoldRiskEngine()
        self.correlation_guard = correlation_guard
        self.circuit_breaker_enabled = bool(circuit_breaker_enabled)
        self.max_daily_losses = max_daily_losses if self.circuit_breaker_enabled else 0
        self.max_consecutive_losses = max_consecutive_losses if self.circuit_breaker_enabled else 0
        self.max_daily_drawdown_pct = max_daily_drawdown_pct if self.circuit_breaker_enabled else 0.0
        self.max_total_open_risk_pct = max_total_open_risk_pct
        self.max_open_positions = max_open_positions
        self.history_store = history_store
        self.manage_manual_positions = manage_manual_positions
        self.post_loss_buffer_seconds = float(post_loss_buffer_seconds) if self.circuit_breaker_enabled else 0.0
        self.anti_revenge_window_seconds = float(anti_revenge_window_seconds)
        self.predictive_scratch_enabled = bool(predictive_scratch_enabled)
        self.micro_budget_loss_cap_pct = float(micro_budget_loss_cap_pct)
        self.max_mitigation_loss_r = float(max_mitigation_loss_r)
        self.min_hold_time_seconds = float(min_hold_time_seconds)
        self.scorer = scorer
        self.magic_number: int = 15555

        self._managed_positions: Dict[Union[int, str], ManagedPosition] = {}
        self._pending_limit_orders: Dict[Union[int, str], Any] = {}
        self._recorded_tickets: set = set()
        self._management_task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = asyncio.Lock()
        self._order_execution_lock = asyncio.Lock()

        # Strict Single-Execution Lock & Smart Cooldown
        self._is_order_in_flight: bool = False
        self._last_order_time: float = 0.0
        self._min_cooldown_seconds: float = 8.0
        self._post_loss_cooldown_until: float = 0.0
        self._last_loss_direction: Optional[str] = None
        self._last_loss_price: float = 0.0
        self._last_loss_time: float = 0.0

        # Circuit breaker metrics
        self._daily_losses_count = 0
        self._consecutive_losses = 0
        self._daily_realized_pnl = 0.0
        self._pause_until = 0.0
        self._start_of_day_balance: float = 0.0
        self._today_date_str: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Operational State Reconciliation & Contradictory Safety Guards
        self._contradictory_state_active: bool = False
        self._contradictory_state_reason: str = ""

    async def sync_daily_risk_state(self) -> None:
        """
        Reconstruct daily risk metrics from persistent trade history and broker account info.
        Prevents restart from resetting loss counters or daily drawdown limits.
        """
        now_dt = datetime.now(timezone.utc)
        self._today_date_str = now_dt.strftime("%Y-%m-%d")
        start_of_day_dt = datetime(now_dt.year, now_dt.month, now_dt.day, tzinfo=timezone.utc)
        start_of_day_ts = start_of_day_dt.timestamp()

        if self.history_store:
            try:
                trades = await self.history_store.get_history(limit=200)
                today_trades = [t for t in trades if t.get("close_time", 0.0) >= start_of_day_ts]
                today_trades.sort(key=lambda x: x.get("close_time", 0.0))

                self._daily_realized_pnl = sum(t.get("pnl", 0.0) for t in today_trades)
                self._daily_losses_count = sum(1 for t in today_trades if t.get("pnl", 0.0) < -0.05)

                cons = 0
                for t in reversed(today_trades):
                    p = t.get("pnl", 0.0)
                    if p < -0.05:
                        cons += 1
                    elif p > 0.05:
                        break
                self._consecutive_losses = cons
                logger.info(
                    f"Reconstructed daily risk state: {len(today_trades)} trades today, "
                    f"P&L: ${self._daily_realized_pnl:.2f}, Daily losses: {self._daily_losses_count}, "
                    f"Consecutive losses: {self._consecutive_losses}"
                )
            except Exception as e:
                logger.warning(f"Could not reconstruct risk state from history store: {e}")

        try:
            acc = await self.broker.get_account_info()
            if acc and getattr(acc, "balance", 0.0) > 0.0:
                if self._start_of_day_balance <= 0.0:
                    self._start_of_day_balance = max(1.0, acc.balance - self._daily_realized_pnl)
                logger.info(f"Start-of-day balance set to: ${self._start_of_day_balance:.2f} (Current: ${acc.balance:.2f})")
            else:
                logger.info(f"Broker account balance not yet available on startup (Current: ${getattr(acc, 'balance', 0.0):.2f}). Waiting for broker sync.")
        except Exception as e:
            logger.warning(f"Could not fetch account info for risk state sync: {e}")

    async def reconcile_broker_state(self) -> Dict[str, Any]:
        """
        Reconcile local trade manager state with broker state on startup, reconnection,
        or recovery. Enrolls bot-owned positions and pending orders, verifies protective
        stop losses, and detects contradictory or unmanaged risk.
        """
        async with self._lock:
            try:
                broker_positions = await self.broker.get_open_positions()
                broker_orders = await self.broker.get_pending_orders()
            except Exception as e:
                logger.error(f"[RECONCILE ERROR] Failed to fetch broker state: {e}")
                self._contradictory_state_active = True
                self._contradictory_state_reason = f"Broker query failed during state reconciliation: {e}"
                return {"status": "ERROR", "error": str(e), "is_contradictory": True}

            broker_pos_by_ticket = {}
            for p in broker_positions:
                if str(p.ticket).isdigit():
                    broker_pos_by_ticket[int(p.ticket)] = p

            broker_orders_by_ticket = {}
            for o in broker_orders:
                if str(o.ticket).isdigit():
                    broker_orders_by_ticket[int(o.ticket)] = o

            enrolled_positions = []
            unprotected_positions = []

            for ticket, p in broker_pos_by_ticket.items():
                is_manual = getattr(p, "is_user_manual", False)
                magic = getattr(p, "magic", 0)
                comment = getattr(p, "comment", "")
                is_bot_owned = (magic == 15555) or ("ApexGold" in comment) or ("Paper" in comment)

                # Skip manual positions if policy dictates not to manage them
                if is_manual and not self.manage_manual_positions:
                    continue
                if not is_bot_owned and not self.manage_manual_positions:
                    continue

                # Verify protective stop loss
                if p.sl <= 0.0:
                    unprotected_positions.append(ticket)
                    logger.warning(f"[RECONCILE WARNING] Active position #{ticket} has no protective stop loss!")

                if ticket not in self._managed_positions:
                    # Enroll into managed positions
                    direction_str = p.type.value if hasattr(p.type, "value") else str(p.type)
                    sl_dist = abs(p.open_price - p.sl) if p.sl > 0 else 2.50
                    m = ManagedPosition(
                        ticket=ticket,
                        symbol=p.symbol,
                        direction=direction_str.upper(),
                        volume=p.volume,
                        open_price=p.open_price,
                        sl=p.sl,
                        tp=p.tp,
                        entry_time=p.open_time or time.time(),
                        risk_r_dist=sl_dist,
                        initial_regime="RECOVERED",
                        current_regime="RECOVERED",
                        sub_regime="RECOVERED",
                        is_user_manual=is_manual,
                        current_price=getattr(p, "current_price", p.open_price),
                    )
                    self._managed_positions[ticket] = m
                    enrolled_positions.append(ticket)
                    logger.info(f"[RECONCILE] Enrolled active position #{ticket} into live management (SL={p.sl}, TP={p.tp}).")

            # Check if any local managed positions no longer exist on broker
            disappeared_tickets = []
            for ticket in list(self._managed_positions.keys()):
                if ticket not in broker_pos_by_ticket:
                    disappeared_tickets.append(ticket)
                    pos = self._managed_positions.pop(ticket)
                    logger.info(f"[RECONCILE] Position #{ticket} closed while offline. Reconciling closed deal...")
                    deal = await self.broker.get_closed_deal(ticket)
                    if deal and self.history_store:
                        await self.history_store.record_trade(
                            ticket=ticket,
                            symbol=deal.symbol,
                            direction=deal.direction,
                            volume=deal.volume,
                            open_price=deal.open_price,
                            close_price=deal.close_price,
                            open_time=deal.open_time,
                            close_time=deal.close_time,
                            pnl=round(deal.profit + getattr(deal, "commission", 0.0) + getattr(deal, "swap", 0.0), 2),
                            pnl_r=0.0,
                            regime=pos.initial_regime,
                            sub_regime=pos.sub_regime,
                            close_reason="Closed While Offline",
                        )

            # Reconcile pending limit orders
            for ticket, o in broker_orders_by_ticket.items():
                if ticket not in self._pending_limit_orders:
                    self._pending_limit_orders[ticket] = o
                    logger.info(f"[RECONCILE] Enrolled pending order #{ticket} into live tracking.")

            # If there are unprotected positions, flag contradictory state
            if unprotected_positions:
                self._contradictory_state_active = True
                self._contradictory_state_reason = f"Position(s) #{unprotected_positions} missing protective stop loss"
                if self.history_store:
                    await self.history_store.record_audit_event("UNPROTECTED_POSITION_ALERT", self._contradictory_state_reason)
            else:
                self._contradictory_state_active = False
                self._contradictory_state_reason = ""

            return {
                "status": "OK",
                "enrolled": enrolled_positions,
                "disappeared": disappeared_tickets,
                "unprotected": unprotected_positions,
                "is_contradictory": self._contradictory_state_active,
            }

    async def _reconcile_post_order_failure(
        self,
        symbol: str,
        direction: str,
        lot_size: float,
        target_entry: float,
        padded_sl: float,
        suggested_tp: float,
        comment: str,
        signal: ConfluenceSignal,
    ) -> Optional[ManagedPosition]:
        """
        Post-order failure recovery:
        Queries broker to verify whether an order was accepted despite dispatch errors or timeouts.
        Verifies protective SL/TP, enforces emergency close policy if protection cannot be confirmed,
        and logs audit trail.
        """
        try:
            await asyncio.sleep(0.5)
            open_positions = await self.broker.get_open_positions()
            now = time.time()
            matching_candidates: List[Position] = []

            for p in open_positions:
                ticket_int = int(p.ticket) if str(p.ticket).isdigit() else p.ticket
                if ticket_int in self._managed_positions:
                    continue
                p_dir = p.type.value if hasattr(p.type, "value") else str(p.type)

                # Strict multi-factor matching:
                # 1. Symbol match
                if p.symbol != symbol:
                    continue
                # 2. Direction match
                if p_dir.upper() != direction.upper():
                    continue
                # 3. Exact volume match
                if abs(p.volume - lot_size) >= 0.001:
                    continue
                # 4. Recent open time within 30s
                p_open_time = getattr(p, "open_time", now) or now
                if abs(now - p_open_time) > 30.0:
                    continue
                # 5. Price proximity (within 1.5% or $15 for gold)
                threshold = max(target_entry * 0.015, 15.0)
                if abs(p.open_price - target_entry) > threshold:
                    continue
                # 6. Magic number or comment match if present
                p_magic = getattr(p, "magic", 0)
                if p_magic not in (0, self.magic_number):
                    continue
                p_comment = getattr(p, "comment", "")
                if comment and p_comment and (comment not in p_comment) and ("ApexGold" not in p_comment):
                    continue

                matching_candidates.append(p)

            if len(matching_candidates) > 1:
                # Ambiguous recovery: multiple unmanaged candidate positions matched!
                self._contradictory_state_active = True
                cand_tickets = [c.ticket for c in matching_candidates]
                self._contradictory_state_reason = f"Ambiguous post-order recovery: found {len(matching_candidates)} candidate positions matching order params: {cand_tickets}"
                self.pause_trading(minutes=1440)
                logger.critical(f"[POST-ORDER RECOVERY AMBIGUITY] {self._contradictory_state_reason}. Blocking all new entries!")
                if self.history_store:
                    await self.history_store.record_audit_event(
                        "AMBIGUOUS_RECOVERY_ALERT",
                        self._contradictory_state_reason
                    )
                return None

            if len(matching_candidates) == 0:
                logger.info("[POST-ORDER RECOVERY] Confirmed no ghost position was opened by broker.")
                return None

            match_pos = matching_candidates[0]
            ticket = int(match_pos.ticket)
            logger.warning(f"[POST-ORDER RECOVERY] Unregistered position #{ticket} detected on broker. Verifying protective stop loss...")

            # Protection verification & post-action broker query
            sl_confirmed = (match_pos.sl > 0.0)
            if not sl_confirmed:
                logger.warning(f"[POST-ORDER RECOVERY] Position #{ticket} has no confirmed SL. Attempting to attach protective SL={padded_sl}...")
                mod_ok = await self.broker.modify_position(ticket, sl=padded_sl, tp=suggested_tp)

                # Query broker state again to verify actual result of modification
                refreshed_positions = await self.broker.get_open_positions()
                refreshed_pos = next(
                    (p for p in refreshed_positions if (int(p.ticket) if str(p.ticket).isdigit() else p.ticket) == ticket),
                    None
                )

                if refreshed_pos is not None and refreshed_pos.sl > 0.0:
                    sl_confirmed = True
                    match_pos = refreshed_pos
                else:
                    # EMERGENCY POLICY: Position remains unprotected on broker!
                    logger.critical(f"[EMERGENCY POLICY] Could not verify/attach protective stop loss to position #{ticket}! Closing immediately.")
                    close_ok = await self.broker.close_position(ticket)

                    # Query broker state again to verify actual result of emergency close
                    post_close_positions = await self.broker.get_open_positions()
                    still_open = any(
                        (int(p.ticket) if str(p.ticket).isdigit() else p.ticket) == ticket
                        for p in post_close_positions
                    )

                    self._contradictory_state_active = True
                    self.pause_trading(minutes=1440)

                    if still_open:
                        self._contradictory_state_reason = f"CRITICAL: Emergency close failed for unprotected position #{ticket}! Position is still open on broker."
                        logger.critical(f"[EMERGENCY POLICY FAILURE] {self._contradictory_state_reason}")
                        if self.history_store:
                            await self.history_store.record_audit_event(
                                "CRITICAL_CLOSE_FAILED",
                                self._contradictory_state_reason
                            )
                    else:
                        self._contradictory_state_reason = f"Emergency close executed on unprotected position #{ticket}"
                        logger.warning(f"[EMERGENCY POLICY] {self._contradictory_state_reason}")
                        if self.history_store:
                            await self.history_store.record_audit_event(
                                "EMERGENCY_CLOSE",
                                f"Closed position #{ticket} because protective stop-loss could not be verified."
                            )
                    return None

            # Register recovered position safely into management
            recovered_managed = ManagedPosition(
                ticket=ticket,
                symbol=symbol,
                direction=direction,
                volume=match_pos.volume if match_pos.volume > 0 else lot_size,
                open_price=match_pos.open_price if match_pos.open_price > 0 else target_entry,
                sl=match_pos.sl if match_pos.sl > 0 else padded_sl,
                tp=match_pos.tp if match_pos.tp > 0 else suggested_tp,
                entry_time=getattr(match_pos, "open_time", now) or now,
                risk_r_dist=abs(target_entry - padded_sl),
                initial_regime=signal.regime,
                current_regime=signal.regime,
                sub_regime=signal.sub_regime,
                be_lock_r=signal.be_lock_r,
                stall_exit_seconds=signal.stall_exit_seconds,
            )
            async with self._lock:
                self._managed_positions[ticket] = recovered_managed

            if self.history_store:
                await self.history_store.record_audit_event(
                    "POST_ORDER_RECOVERY",
                    f"Recovered position #{ticket} ({symbol} {direction} {match_pos.volume}) with confirmed SL={match_pos.sl}"
                )

            logger.info(f"[POST-ORDER RECOVERY] Successfully enrolled recovered position #{ticket} into Chameleon management.")
            return recovered_managed

        except Exception as e:
            logger.critical(f"[POST-ORDER RECOVERY ERROR] Error while recovering unconfirmed order: {e}", exc_info=True)
            return None

    async def start(self) -> None:
        """Start the 250ms live position management loop."""
        self._running = True
        await self.sync_daily_risk_state()
        await self.reconcile_broker_state()
        if self._management_task is None or self._management_task.done():
            self._management_task = asyncio.create_task(self._management_loop())
        logger.info("TradeManager 250ms Follow-the-Market loop started.")

    async def stop(self) -> None:
        """Gracefully terminate background management."""
        self._running = False
        if self._management_task and not self._management_task.done():
            self._management_task.cancel()
            try:
                await self._management_task
            except asyncio.CancelledError:
                pass
        logger.info("TradeManager stopped.")

    def pause_trading(self, minutes: int = 60) -> None:
        """Temporarily pause automated trade executions."""
        if not self.circuit_breaker_enabled:
            logger.info("pause_trading called but circuit breaker is disabled (trading is always enabled). Ignoring pause.")
            self._pause_until = 0.0
            return
        self._pause_until = time.time() + (minutes * 60)
        logger.info(f"Trading paused for {minutes} minutes until {datetime.fromtimestamp(self._pause_until, tz=timezone.utc)}")

    async def resume_trading(self, force: bool = False) -> Tuple[bool, str]:
        """
        Immediately resume trading and reset circuit breaker if daily drawdown allows.
        When force=True (manual operator override from UI/API), clears pause and updates baseline.
        """
        acc = await self.broker.get_account_info()
        eff_base = self._start_of_day_balance if self._start_of_day_balance > 0 else (acc.balance if acc else 100.0)
        is_micro = eff_base <= 35.0
        allowed_dd_pct = 50.0 if is_micro else self.max_daily_drawdown_pct

        if self.circuit_breaker_enabled and not force and acc and self._start_of_day_balance > 0 and self.max_daily_drawdown_pct > 0.0:
            eq_dd_pct = ((self._start_of_day_balance - acc.equity) / self._start_of_day_balance) * 100.0
            real_dd_pct = ((-self._daily_realized_pnl) / self._start_of_day_balance) * 100.0
            if self._daily_realized_pnl >= 0 and acc.equity >= (self._start_of_day_balance * 0.90):
                max_dd = 0.0
            else:
                max_dd = max(eq_dd_pct, real_dd_pct)

            if max_dd >= allowed_dd_pct:
                msg = f"Cannot resume: Daily drawdown ({max_dd:.1f}% >= {allowed_dd_pct:.1f}%) still breached!"
                logger.warning(msg)
                return False, msg

        rec = await self.reconcile_broker_state()
        if not force and rec.get("is_contradictory", False):
            msg = f"Cannot resume: Unresolved contradictory broker state ({self._contradictory_state_reason})."
            logger.warning(msg)
            return False, msg

        self._pause_until = 0.0
        self._daily_losses_count = 0
        self._consecutive_losses = 0
        self._post_loss_cooldown_until = 0.0
        self._last_order_time = 0.0
        self._last_loss_time = 0.0
        self._contradictory_state_active = False
        self._contradictory_state_reason = ""

        if force and acc and getattr(acc, "balance", 0.0) > 0:
            self._start_of_day_balance = acc.balance
            self._daily_realized_pnl = 0.0
            logger.info(f"Operator force-resumed trading: baseline balance reset to ${self._start_of_day_balance:.2f}.")
        else:
            logger.info("Trading resumed and daily loss counter reset.")
        return True, "Trading resumed successfully"

    def is_paused(self) -> bool:
        """Check if circuit breaker or manual pause is active."""
        if getattr(self, "_contradictory_state_active", False):
            return True
        if not self.circuit_breaker_enabled:
            return False
        if time.time() < self._pause_until:
            return True
        if self.max_daily_losses > 0 and self._daily_losses_count >= self.max_daily_losses:
            return True
        if self.max_consecutive_losses > 0 and self._consecutive_losses >= self.max_consecutive_losses:
            return True
        return False

    def is_circuit_breaker_tripped(self) -> Tuple[bool, str]:
        """Check whether any circuit breaker condition is active and return status with description."""
        if not self.circuit_breaker_enabled:
            return False, ""
        now = time.time()
        if now < self._pause_until:
            rem = int(self._pause_until - now)
            return True, f"Trading paused ({rem}s remaining)"
        if self.max_daily_losses > 0 and self._daily_losses_count >= self.max_daily_losses:
            return True, f"Max daily losses reached ({self._daily_losses_count}/{self.max_daily_losses})"
        if self.max_consecutive_losses > 0 and self._consecutive_losses >= self.max_consecutive_losses:
            return True, f"Max consecutive losses reached ({self._consecutive_losses}/{self.max_consecutive_losses})"
        if self.max_daily_drawdown_pct > 0.0:
            eff_base = self._start_of_day_balance if self._start_of_day_balance > 0 else (getattr(self.risk_engine, "base_budget_usd", 0.0) or 100.0)
            is_micro = eff_base <= 35.0
            allowed_dd_pct = 75.0 if is_micro else self.max_daily_drawdown_pct
            if eff_base > 0 and self._daily_realized_pnl < 0:
                real_dd_usd = max(0.0, -self._daily_realized_pnl)
                real_dd_pct = (real_dd_usd / eff_base) * 100.0
                if real_dd_pct >= allowed_dd_pct:
                    return True, f"Daily realized drawdown limit breached ({real_dd_pct:.1f}% >= {allowed_dd_pct:.1f}%)"
        if now < self._post_loss_cooldown_until:
            rem = int(self._post_loss_cooldown_until - now)
            return True, f"Post-loss buffer active ({rem}s remaining)"
        return False, ""

    async def can_open_manual_trade(self, symbol: str, direction: str, volume: float = 0.01) -> Tuple[bool, str]:
        """
        Validate whether a manual or UI trade can be safely opened against
        circuit breakers, drawdown limits, exposure, and news freeze.
        """
        acc = await self.broker.get_account_info()
        eff_balance = acc.balance if acc else 100.0
        self._reset_daily_metrics_if_new_day(current_balance=eff_balance)
        if eff_balance > 0 and self._start_of_day_balance <= 0.0:
            self._start_of_day_balance = max(1.0, eff_balance - self._daily_realized_pnl)

        if self.circuit_breaker_enabled:
            cb_tripped, cb_reason = self.is_circuit_breaker_tripped()
            if cb_tripped:
                return False, f"Circuit Breaker active: {cb_reason}"

            if getattr(self, "_contradictory_state_active", False):
                return False, f"Reconciliation Alert: Unresolved contradictory broker state ({self._contradictory_state_reason}). New orders blocked."

            if acc and getattr(acc, "equity", 0.0) > 0 and self._start_of_day_balance > 1.0 and self.max_daily_drawdown_pct > 0.0:
                is_micro = eff_balance <= 35.0
                allowed_dd_pct = 75.0 if is_micro else self.max_daily_drawdown_pct
                
                eq_dd_usd = max(0.0, self._start_of_day_balance - acc.equity)
                eq_dd_pct = (eq_dd_usd / self._start_of_day_balance) * 100.0
                real_dd_usd = max(0.0, -self._daily_realized_pnl)
                real_dd_pct = (real_dd_usd / self._start_of_day_balance) * 100.0
                
                # Net daily drawdown only applies if net equity or realized balance is actually below start of day
                if self._daily_realized_pnl >= 0 and acc.equity >= (self._start_of_day_balance * 0.90):
                    max_dd = 0.0
                else:
                    max_dd = max(eq_dd_pct, real_dd_pct)

                if max_dd >= allowed_dd_pct:
                    now_dt = datetime.now(timezone.utc)
                    tomorrow_dt = (now_dt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                    sec_to_midnight = max(60, int((tomorrow_dt - now_dt).total_seconds()))
                    self._pause_until = time.time() + sec_to_midnight
                    return False, f"Circuit Breaker: Daily drawdown breached ({max_dd:.1f}% >= {allowed_dd_pct:.1f}%). Orders halted until 00:00 UTC."
        else:
            if getattr(self, "_contradictory_state_active", False):
                return False, f"Reconciliation Alert: Unresolved contradictory broker state ({self._contradictory_state_reason}). New orders blocked."

        if self._is_order_in_flight:
            return False, "Order already in flight (waiting for broker execution)"

        if self.news_reactor:
            can_open_news = True
            news_reason = ""
            if hasattr(self.news_reactor, "can_open_baseline_trade"):
                res = self.news_reactor.can_open_baseline_trade()
                if isinstance(res, tuple) and len(res) == 2:
                    can_open_news, news_reason = res
            if not can_open_news:
                return False, f"News Reactor: {news_reason}"

        active_positions = await self.broker.get_open_positions()
        managed_open = [p for p in self._managed_positions.values() if not p.is_closed]
        all_positions_count = max(len(managed_open), len(active_positions))

        # Strict Anti-Hedging Guard:
        # Never allow opening BUY and SELL simultaneously on the same symbol or Gold asset
        is_gold = any(k in symbol.upper() for k in ["XAU", "GOLD"])
        target_dir = direction.upper().strip()

        for p in active_positions:
            p_sym = p.symbol.upper()
            if (is_gold and any(k in p_sym for k in ["XAU", "GOLD"])) or (p_sym == symbol.upper()):
                p_dir = (p.type.value if hasattr(p.type, "value") else str(p.type)).upper()
                if p_dir != target_dir:
                    return False, f"Anti-Hedging Guard: Open {p_dir} position #{p.ticket} already active. Simultaneous BUY & SELL (hedging) is strictly forbidden."

        for m in managed_open:
            m_sym = m.symbol.upper()
            if (is_gold and any(k in m_sym for k in ["XAU", "GOLD"])) or (m_sym == symbol.upper()):
                if m.direction.upper() != target_dir:
                    return False, f"Anti-Hedging Guard: Managed {m.direction} position #{m.ticket} already active. Simultaneous BUY & SELL (hedging) is strictly forbidden."

        # Small account guard: accounts under $150 strictly limited to 1 active position
        if eff_balance < 150.0 and all_positions_count >= 1:
            return False, f"Account balance (${eff_balance:.2f}) under $150 strictly limited to 1 active position at a time."

        if self.max_open_positions > 0 and all_positions_count >= self.max_open_positions:
            return False, f"Maximum open positions reached ({all_positions_count}/{self.max_open_positions})"

        return True, "OK"

    def _reset_daily_metrics_if_new_day(self, current_balance: Optional[float] = None) -> None:
        """Reset circuit breaker counters at 00:00 UTC."""
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today_str != self._today_date_str:
            logger.info(f"New trading day ({today_str}): resetting daily counters.")
            self._daily_losses_count = 0
            self._consecutive_losses = 0
            self._daily_realized_pnl = 0.0
            self._pause_until = 0.0
            self._contradictory_state_active = False
            self._contradictory_state_reason = ""
            self._today_date_str = today_str
            if current_balance is not None and current_balance > 0:
                self._start_of_day_balance = current_balance

    async def _release_in_flight_lock(self, delay_sec: float = 2.5) -> None:
        """Release in-flight lock after broker communication delay."""
        await asyncio.sleep(delay_sec)
        self._is_order_in_flight = False

    async def can_execute_signal(self, signal: ConfluenceSignal) -> Tuple[bool, str]:
        """
        Verify circuit breakers, in-flight locks, news freeze, and USD correlation guard before placing order.
        Strictly guarantees NO multiple trades can ever open simultaneously.
        """
        acc = await self.broker.get_account_info()
        eff_balance = acc.balance if acc else 100.0
        self._reset_daily_metrics_if_new_day(current_balance=eff_balance)
        if eff_balance > 0 and self._start_of_day_balance <= 0.0:
            self._start_of_day_balance = max(1.0, eff_balance - self._daily_realized_pnl)

        now = time.time()
        if getattr(self, "_contradictory_state_active", False):
            return False, f"Reconciliation Alert: Unresolved contradictory broker state ({self._contradictory_state_reason}). New orders blocked."

        if self.circuit_breaker_enabled:
            if self.is_paused():
                if now < self._pause_until:
                    rem_s = int(self._pause_until - now)
                    return False, f"Trading paused (Cooldown: {rem_s}s remaining. Use Resume button to unblock)"
                daily_str = f"Daily losses: {self._daily_losses_count}/{self.max_daily_losses}" if self.max_daily_losses > 0 else "Daily losses: Unlimited"
                return False, f"Trading paused ({daily_str}, Consecutive: {self._consecutive_losses}/{self.max_consecutive_losses})"

            # 0. Daily Drawdown Circuit Breaker Enforcement
            if acc and getattr(acc, "equity", 0.0) > 0 and self._start_of_day_balance > 1.0 and self.max_daily_drawdown_pct > 0.0:
                is_micro = eff_balance <= 35.0
                allowed_dd_pct = 75.0 if is_micro else self.max_daily_drawdown_pct

                eq_dd_usd = max(0.0, self._start_of_day_balance - acc.equity)
                eq_dd_pct = (eq_dd_usd / self._start_of_day_balance) * 100.0
                real_dd_usd = max(0.0, -self._daily_realized_pnl)
                real_dd_pct = (real_dd_usd / self._start_of_day_balance) * 100.0

                # Net daily drawdown only applies if net equity or realized balance is actually below start of day
                if self._daily_realized_pnl >= 0 and acc.equity >= (self._start_of_day_balance * 0.90):
                    max_dd = 0.0
                else:
                    max_dd = max(eq_dd_pct, real_dd_pct)

                if max_dd >= allowed_dd_pct:
                    now_dt = datetime.now(timezone.utc)
                    tomorrow_dt = (now_dt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                    sec_to_midnight = max(60, int((tomorrow_dt - now_dt).total_seconds()))
                    self._pause_until = time.time() + sec_to_midnight
                    return False, f"Circuit Breaker: Daily drawdown breached ({max_dd:.1f}% >= {allowed_dd_pct:.1f}%). New orders halted until 00:00 UTC."

            # 1. Post-Loss Tick Buffer (Broker deal and tick settlement period)
            now = time.time()
            if now < self._post_loss_cooldown_until:
                rem = int(self._post_loss_cooldown_until - now)
                return False, f"Post-Loss Buffer ({rem}s remaining) - Letting broker ticks stabilize."

        # 1b. Smart Directional & Structural Anti-Revenge Guard:
        # Blocks blindly re-entering the EXACT SAME losing trade in the same direction at the same level.
        # Intentionally permits high-probability opposite reversals and structural shifts immediately!
        if signal and (now - self._last_loss_time < self.anti_revenge_window_seconds):
            if signal.direction == self._last_loss_direction:
                is_gold = any(k in signal.symbol.upper() for k in ["XAU", "GOLD"])
                if abs(signal.entry_price - self._last_loss_price) < (1.50 if is_gold else 0.0012):
                    rem = int(self.anti_revenge_window_seconds - (now - self._last_loss_time))
                    return False, f"Anti-Revenge Filter: Blocked repeating losing {signal.direction} at same level ({signal.entry_price:.2f}) without structural shift ({rem}s remaining)."

        # 1c. In-Flight Execution Mutex Check
        if self._is_order_in_flight:
            return False, "Order already in flight (waiting for broker execution)"

        # 2. Minimum Cooldown Between Executions (10 seconds)
        elapsed = now - self._last_order_time
        if elapsed < self._min_cooldown_seconds:
            rem = int(self._min_cooldown_seconds - elapsed)
            return False, f"Order cooldown active ({rem}s remaining)"

        # 3. News Reactor Check
        can_open_news, news_reason = self.news_reactor.can_open_baseline_trade()
        if not can_open_news:
            return False, news_reason

        # 4. Strict Small-Account Exposure Guard (Positions + Pending Orders)
        acc = await self.broker.get_account_info()
        eff_balance = acc.balance if acc else 100.0

        active_positions = await self.broker.get_open_positions()
        managed_open = [p for p in self._managed_positions.values() if not p.is_closed]
        all_positions_count = max(len(managed_open), len(active_positions))

        # Strict Anti-Hedging Guard:
        # Never allow opening BUY and SELL simultaneously on the same symbol or Gold asset
        is_gold_signal = any(k in signal.symbol.upper() for k in ["XAU", "GOLD"])
        target_dir = signal.direction.upper().strip()

        for p in active_positions:
            p_sym = p.symbol.upper()
            if (is_gold_signal and any(k in p_sym for k in ["XAU", "GOLD"])) or (p_sym == signal.symbol.upper()):
                p_dir = (p.type.value if hasattr(p.type, "value") else str(p.type)).upper()
                if p_dir != target_dir:
                    return False, f"Anti-Hedging Guard: Conflicting open {p_dir} trade #{p.ticket} vs signal ({target_dir}). Simultaneous BUY & SELL (hedging) is strictly forbidden."

        for m in managed_open:
            m_sym = m.symbol.upper()
            if (is_gold_signal and any(k in m_sym for k in ["XAU", "GOLD"])) or (m_sym == signal.symbol.upper()):
                if m.direction.upper() != target_dir:
                    return False, f"Anti-Hedging Guard: Conflicting managed {m.direction} trade #{m.ticket} vs signal ({target_dir}). Simultaneous BUY & SELL (hedging) is strictly forbidden."

        # Check pending orders across all layers (TradeManager, PaperBroker, and MT5 Bridge)
        import inspect
        from ui.server import ctx as server_ctx
        if hasattr(self.broker, "get_pending_orders"):
            p_res = self.broker.get_pending_orders()
            broker_pending = await p_res if inspect.isawaitable(p_res) else (p_res or [])
        else:
            broker_pending = []

        mt5_pending = getattr(server_ctx, "mt5_pending_orders", []) or []
        tm_pending = list(self._pending_limit_orders.values())
        total_pending_count = max(len(tm_pending), len(broker_pending), len(mt5_pending))

        # Auto-clear stale pending limits (> 60s old or price drifted > $2.00 away)
        if total_pending_count > 0:
            cur_tick = await self.broker.get_tick(signal.symbol)
            cur_mid = ((cur_tick.bid + cur_tick.ask) / 2.0) if cur_tick and cur_tick.bid > 0 else signal.entry_price
            
            stale_tickets_to_cancel = []
            for lo in list(self._pending_limit_orders.values()):
                if (now - lo.created_time > 60.0) or (abs(cur_mid - lo.price) > 2.0):
                    stale_tickets_to_cancel.append(lo.ticket)
            
            for o in mt5_pending:
                o_ticket = o.get("ticket")
                o_price = float(o.get("price", 0.0))
                if o_price > 0 and abs(cur_mid - o_price) > 2.0 and o_ticket:
                    stale_tickets_to_cancel.append(int(o_ticket))

            if stale_tickets_to_cancel:
                logger.info(f"Auto-clearing {len(stale_tickets_to_cancel)} stale pending limit orders before new signal: {stale_tickets_to_cancel}")
                for st in set(stale_tickets_to_cancel):
                    await self.broker.cancel_order(st)
                    self._pending_limit_orders.pop(st, None)
                if hasattr(self.broker, "get_pending_orders"):
                    p_res2 = self.broker.get_pending_orders()
                    broker_pending = await p_res2 if inspect.isawaitable(p_res2) else (p_res2 or [])
                else:
                    broker_pending = []
                mt5_pending = getattr(server_ctx, "mt5_pending_orders", []) or []
                total_pending_count = max(len(self._pending_limit_orders), len(broker_pending), len(mt5_pending))

        total_exposure = all_positions_count + total_pending_count
        is_limit = getattr(signal, "order_type", "MARKET") in ["BUY_LIMIT", "SELL_LIMIT"]

        # Exposure limit: respect configured max_open_positions (up to 2 for micro accounts)
        if eff_balance < 150.0:
            allowed_max = min(self.max_open_positions, 2 if self.max_open_positions > 1 else 1)
        elif eff_balance < 300.0:
            allowed_max = 2 if (signal.score >= 82 and (signal.tier == "TIER_1_SNIPER")) else 1
        else:
            allowed_max = 3 if (signal.score >= 82 and (signal.tier == "TIER_1_SNIPER")) else 1

        if total_exposure >= allowed_max:
            # If 0 open positions and an incoming high-confluence MARKET order is blocked by a pending limit,
            # auto-clear the pending limit order to prioritize immediate market execution
            if all_positions_count == 0 and not is_limit and total_pending_count > 0:
                logger.info(f"Auto-clearing pending limit order(s) to execute high-probability market signal {signal.direction}")
                for p_t in list(self._pending_limit_orders.keys()):
                    await self.broker.cancel_order(p_t)
                    self._pending_limit_orders.pop(p_t, None)
                total_pending_count = 0
                total_exposure = all_positions_count + total_pending_count
            if total_exposure >= allowed_max:
                return False, f"Maximum exposure reached ({total_exposure}/{allowed_max}) including {total_pending_count} pending order(s) for balance ${eff_balance:.2f}"

        # Stacking limit orders is strictly forbidden:
        if is_limit and total_pending_count > 0:
            return False, f"Pending limit order already active for {signal.symbol}. Stacking duplicate limit orders is forbidden."


        # 5. Smart Multi-Position Risk Guard (Pyramiding ONLY allowed for >= $150 balance with existing trade in solid profit)
        if all_positions_count >= allowed_max:
            return False, f"Maximum open positions reached ({all_positions_count}/{allowed_max}) for balance ${eff_balance:.2f}."

            for p in active_positions:
                p_sym = p.symbol.upper()
                if (is_gold_signal and any(k in p_sym for k in ["XAU", "GOLD"])) or (p.symbol == signal.symbol):
                    p_profit = getattr(p, "profit", 0.0)
                    if p_profit <= 0.0:
                        return False, f"Existing {p.symbol} trade #{p.ticket} not in profit (${p_profit:.2f}). Pyramiding only allowed on winning trades!"

            for m in self._managed_positions.values():
                if not m.is_closed:
                    m_sym = m.symbol.upper()
                    if (is_gold_signal and any(k in m_sym for k in ["XAU", "GOLD"])) or (m.symbol == signal.symbol):
                        # If existing trade is not at least +0.80R in profit with SL secured, forbid new entry
                        cur_tick = await self.broker.get_tick(m.symbol)
                        if cur_tick and cur_tick.bid > 0:
                            curr_p = cur_tick.bid if m.direction == "BUY" else cur_tick.ask
                            dist = (curr_p - m.open_price) if m.direction == "BUY" else (m.open_price - curr_p)
                            float_r = dist / max(m.risk_r_dist, 0.01)
                            if float_r < 0.80:
                                return False, f"Existing {m.symbol} trade #{m.ticket} only at {float_r:.2f}R. Must reach at least +0.80R with SL secured before adding!"

        # 6. Total Open Risk Guard
        if eff_balance > 0:
            total_open_risk_usd = 0.0
            for p in self._managed_positions.values():
                if not p.is_closed:
                    sym_info = await self.broker.get_symbol_info(p.symbol)
                    pos_risk = self.risk_engine.calculate_monetary_risk(
                        symbol_info=sym_info,
                        lot=p.volume,
                        sl_distance=p.risk_r_dist
                    )
                    total_open_risk_usd += pos_risk

            cand_sym_info = await self.broker.get_symbol_info(signal.symbol)
            cand_risk = self.risk_engine.calculate_monetary_risk(
                symbol_info=cand_sym_info,
                lot=cand_sym_info.volume_min,
                sl_distance=signal.risk_r_dist
            )
            total_projected_risk_pct = ((total_open_risk_usd + cand_risk) / eff_balance) * 100.0
            if eff_balance <= 100.0 and total_open_risk_usd == 0.0:
                pass  # Always permit single minimum lot entry on micro/small accounts
            else:
                allowed_max_risk = max(self.max_total_open_risk_pct, 45.0) if eff_balance <= 35.0 else (max(self.max_total_open_risk_pct, 30.0) if eff_balance <= 150.0 else self.max_total_open_risk_pct)
                if total_projected_risk_pct > allowed_max_risk:
                    return False, f"Total open risk limit reached ({total_projected_risk_pct:.1f}% > {allowed_max_risk:.1f}% allowed)"

        return True, "Execution approved"

    async def execute_trade(self, signal: ConfluenceSignal) -> OrderResult:
        """
        Execute live market order with dynamically calculated lot size, padded SL, and TP.
        Enrolls position into ManagedPosition for real-time 250ms Chameleon monitoring.
        Guaranteed thread-safe and immune to concurrent order collisions.
        """
        async with self._order_execution_lock:
            can_trade, reason = await self.can_execute_signal(signal)
            if not can_trade:
                logger.warning(f"Trade rejected: {reason}")
                return OrderResult(success=False, error=reason)

            # Acquire execution lock immediately
            self._is_order_in_flight = True
            self._last_order_time = time.time()

            try:
                account_info = await self.broker.get_account_info()
                symbol_info = await self.broker.get_symbol_info(signal.symbol)
                # All bot trade signals execute immediately at MARKET to prevent missed runs
                order_type = OrderType.BUY if signal.direction == "BUY" else OrderType.SELL
                target_entry = signal.entry_price

                # Volatility-Adaptive Stop Loss
                is_gold = any(k in signal.symbol.upper() for k in ["XAU", "GOLD"])
                raw_sl = signal.suggested_sl

                # Pad Stop Loss against live spread FIRST so sizing accounts for full padded distance
                padded_sl = self.risk_engine.pad_stop_loss(
                    direction=signal.direction,
                    raw_sl=raw_sl,
                    spread=symbol_info.spread,
                    point=symbol_info.point,
                    extra_buffer_points=2,
                )
                padded_sl_dist = abs(target_entry - padded_sl)

                # Calculate exact lot size with recalculated padded risk distance and predictive brain conviction
                lot_size, risk_details = self.risk_engine.calculate_lot_size(
                    symbol_info=symbol_info,
                    account_info=account_info,
                    sl_distance=padded_sl_dist,
                    confidence_tier=getattr(signal, "tier", None),
                    score=getattr(signal, "score", None),
                    consecutive_losses=self._consecutive_losses,
                    ev_r=getattr(signal, "predictive_ev", None),
                    p_win=getattr(signal, "predictive_p_win", None),
                )

                if not risk_details.get("has_margin", False) or lot_size <= 0.0:
                    msg = f"Trade Rejected: {risk_details.get('warning', 'Margin/Risk check failed')}"
                    logger.error(msg)
                    return OrderResult(success=False, error=msg)

                order_comment = f"ApexGold_{signal.sub_regime[:4]}"

                # Dispatch order to broker
                res = await self.broker.open_order(
                    symbol=signal.symbol,
                    order_type=order_type,
                    volume=lot_size,
                    price=target_entry,
                    sl=padded_sl,
                    tp=signal.suggested_tp,
                    comment=order_comment,
                )

                if res.success and res.ticket is not None:
                    try:
                        eff_bal = account_info.balance if account_info else 100.0
                        active_be_r = 0.35 if (is_gold and eff_bal < 50.0) else signal.be_lock_r

                        # Register position for sub-second Chameleon adaptive management
                        managed = ManagedPosition(
                            ticket=int(res.ticket),
                            symbol=signal.symbol,
                            direction=signal.direction,
                            volume=res.volume if res.volume > 0 else lot_size,
                            open_price=res.price if res.price > 0 else target_entry,
                            sl=padded_sl,
                            tp=signal.suggested_tp,
                            entry_time=time.time(),
                            risk_r_dist=padded_sl_dist,
                            initial_regime=signal.regime,
                            current_regime=signal.regime,
                            sub_regime=signal.sub_regime,
                            be_lock_r=active_be_r,
                            stall_exit_seconds=signal.stall_exit_seconds,
                            highest_r_reached=0.0,
                            entry_velocity=getattr(self.news_reactor, "_latest_velocity", 1.0) or 1.0,
                        )
                        async with self._lock:
                            self._managed_positions[managed.ticket] = managed

                        logger.info(
                            f"[ADAPTIVE TRADE INITIATED] #{managed.ticket} {signal.direction} {lot_size} {signal.symbol} "
                            f"@ {res.price:.2f} | Sub-Regime: {signal.sub_regime} -> {signal.regime} | Risk: {padded_sl_dist:.2f} | TP: {signal.suggested_tp:.2f}"
                        )
                    except Exception as reg_err:
                        logger.critical(f"Critical: Failed to enroll position #{res.ticket} into management: {reg_err}", exc_info=True)
                        # Emergency fallback: always ensure basic ManagedPosition is registered
                        fallback_managed = ManagedPosition(
                            ticket=int(res.ticket),
                            symbol=signal.symbol,
                            direction=signal.direction,
                            volume=res.volume if res.volume > 0 else lot_size,
                            open_price=res.price if res.price > 0 else target_entry,
                            sl=padded_sl,
                            tp=signal.suggested_tp,
                            entry_time=time.time(),
                            risk_r_dist=padded_sl_dist,
                            initial_regime=signal.regime,
                            current_regime=signal.regime,
                            sub_regime=signal.sub_regime,
                            be_lock_r=signal.be_lock_r,
                            stall_exit_seconds=signal.stall_exit_seconds,
                        )
                        async with self._lock:
                            self._managed_positions[fallback_managed.ticket] = fallback_managed

                if not res.success or res.ticket is None:
                    logger.warning(f"[ORDER UNCONFIRMED] Broker returned non-success ({res.error}). Checking if order was accepted anyway...")
                    recovered = await self._reconcile_post_order_failure(
                        symbol=signal.symbol,
                        direction=signal.direction,
                        lot_size=lot_size,
                        target_entry=target_entry,
                        padded_sl=padded_sl,
                        suggested_tp=signal.suggested_tp,
                        comment=order_comment,
                        signal=signal,
                    )
                    if recovered:
                        return OrderResult(success=True, ticket=recovered.ticket, price=recovered.open_price, volume=recovered.volume)

                return res

            except Exception as exec_err:
                logger.error(f"[EXECUTION EXCEPTION] Order dispatch raised error: {exec_err}. Initiating post-order reconciliation...")
                recovered = await self._reconcile_post_order_failure(
                    symbol=signal.symbol,
                    direction=signal.direction,
                    lot_size=lot_size,
                    target_entry=target_entry,
                    padded_sl=padded_sl,
                    suggested_tp=signal.suggested_tp,
                    comment=order_comment,
                    signal=signal,
                )
                if recovered:
                    return OrderResult(success=True, ticket=recovered.ticket, price=recovered.open_price, volume=recovered.volume)
                raise
            finally:
                asyncio.create_task(self._release_in_flight_lock(delay_sec=2.5))

    async def _management_loop(self) -> None:
        """
        250ms Sub-Second 'Follow-the-Market' Position Evaluator.
        Performs real-time Ratchet Locks, Scalp -> Runner Upgrades,
        Fast Scalp 4s Stall Exits, Runner -> Instant Cash Downgrades, and Dead-Trade Scratching.
        """
        while self._running:
            try:
                await asyncio.sleep(0.25)

                async with self._lock:
                    if not self._managed_positions and not self._pending_limit_orders:
                        continue
                    active_tickets = list(self._managed_positions.keys())
                    pending_tickets = list(self._pending_limit_orders.keys())

                broker_positions = await self.broker.get_open_positions()
                broker_pos_map = {int(p.ticket): p for p in broker_positions if str(p.ticket).isdigit()}

                now = time.time()

                # -------------------------------------------------------------
                # Monitor Pending Limit Orders (Predictive Cancellation & Expiry)
                # -------------------------------------------------------------
                for p_ticket in pending_tickets:
                    async with self._lock:
                        lo = self._pending_limit_orders.get(p_ticket)
                    if not lo:
                        continue

                    # If order filled by broker into an active position, enroll into managed positions!
                    matched_pos = broker_pos_map.get(p_ticket)
                    if not matched_pos:
                        for bp in broker_positions:
                            bp_t = int(bp.ticket) if str(bp.ticket).isdigit() else bp.ticket
                            if bp_t not in self._managed_positions and bp.symbol == lo.symbol:
                                lo_dir = "BUY" if "BUY" in str(lo.order_type).upper() else "SELL"
                                bp_dir = bp.type.value if hasattr(bp.type, "value") else str(bp.type).upper()
                                if lo_dir == bp_dir and abs(bp.open_price - lo.price) <= 2.0:
                                    matched_pos = bp
                                    break

                    if matched_pos:
                        pos_ticket = int(matched_pos.ticket) if str(matched_pos.ticket).isdigit() else matched_pos.ticket
                        async with self._lock:
                            # Avoid duplicate registration
                            if pos_ticket not in self._managed_positions:
                                risk_dist = abs(matched_pos.open_price - lo.sl) if lo.sl > 0 else (abs(lo.price - lo.sl) if lo.sl > 0 else 2.50)
                                lo_dir = "BUY" if "BUY" in str(lo.order_type).upper() else "SELL"
                                managed = ManagedPosition(
                                    ticket=pos_ticket,
                                    symbol=lo.symbol,
                                    direction=lo_dir,
                                    volume=matched_pos.volume if matched_pos.volume > 0 else lo.volume,
                                    open_price=matched_pos.open_price if matched_pos.open_price > 0 else lo.price,
                                    sl=lo.sl,
                                    tp=lo.tp,
                                    entry_time=time.time(),
                                    risk_r_dist=risk_dist,
                                    initial_regime="SMART_LIMIT_FILL",
                                    current_regime="SMART_LIMIT_FILL",
                                    sub_regime="SMART_LIMIT",
                                    be_lock_r=0.65,
                                    stall_exit_seconds=min(getattr(lo, "timeout_seconds", 60.0), 60.0),
                                    highest_r_reached=0.0,
                                )
                                self._managed_positions[pos_ticket] = managed
                                logger.info(
                                    f"[PENDING ORDER FILLED & ENROLLED] Limit #{p_ticket} executed as Managed Position #{pos_ticket} "
                                    f"@ {matched_pos.open_price:.2f} | SL: {lo.sl:.2f} | TP: {lo.tp:.2f}"
                                )
                            self._pending_limit_orders.pop(p_ticket, None)
                        continue

                    tick = await self.broker.get_tick(lo.symbol)
                    if not tick:
                        continue

                    # A. Timeout Expiration (75 seconds)
                    effective_timeout = min(lo.timeout_seconds, 75.0)
                    if now - lo.created_time > effective_timeout:
                        logger.info(f"[LIMIT TIMEOUT] Order #{p_ticket} expired after {effective_timeout}s. Cancelling.")
                        await self.broker.cancel_order(p_ticket)
                        async with self._lock:
                            self._pending_limit_orders.pop(p_ticket, None)
                        continue

                    # B. Price Drift Cancellation (if price moves > $2.50 away from limit entry)
                    curr_mid = (tick.bid + tick.ask) / 2.0
                    drift = abs(curr_mid - lo.price)
                    if drift > 2.50:
                        logger.info(f"[LIMIT DRIFT CANCEL] Order #{p_ticket} price drifted ${drift:.2f} away from limit {lo.price:.2f}. Cancelling.")
                        await self.broker.cancel_order(p_ticket)
                        async with self._lock:
                            self._pending_limit_orders.pop(p_ticket, None)
                        continue

                    # C. Predictive Invalidation Guard
                    if lo.invalidation_price > 0:
                        should_cancel = False
                        cancel_reason = ""
                        if lo.order_type == "BUY_LIMIT" and tick.bid <= lo.invalidation_price:
                            should_cancel = True
                            cancel_reason = f"Price broke support {lo.invalidation_price:.2f} (current bid: {tick.bid:.2f})"
                        elif lo.order_type == "SELL_LIMIT" and tick.ask >= lo.invalidation_price:
                            should_cancel = True
                            cancel_reason = f"Price broke resistance {lo.invalidation_price:.2f} (current ask: {tick.ask:.2f})"

                        if should_cancel:
                            logger.info(f"[PREDICTIVE LIMIT CANCEL] Order #{p_ticket} {lo.order_type} canceled BEFORE fill: {cancel_reason}")
                            await self.broker.cancel_order(p_ticket)
                            async with self._lock:
                                self._pending_limit_orders.pop(p_ticket, None)
                            continue

                # Audit MT5 pending orders from server_ctx if any drifted
                from ui.server import ctx as s_ctx
                mt5_orders = getattr(s_ctx, "mt5_pending_orders", []) or []
                if mt5_orders:
                    for mo in list(mt5_orders):
                        mo_tick = mo.get("ticket")
                        mo_price = float(mo.get("price", 0.0))
                        mo_sym = str(mo.get("symbol", "XAUUSD"))
                        if mo_tick and mo_price > 0:
                            t_info = await self.broker.get_tick(mo_sym)
                            if t_info and t_info.bid > 0:
                                t_mid = (t_info.bid + t_info.ask) / 2.0
                                if abs(t_mid - mo_price) > 3.0:
                                    logger.info(f"[MT5 STALE LIMIT CANCEL] Order #{mo_tick} price drifted ${abs(t_mid - mo_price):.2f} away. Cancelling from MT5.")
                                    await self.broker.cancel_order(int(mo_tick))


                for ticket in active_tickets:
                    async with self._lock:
                        if ticket not in self._managed_positions:
                            continue
                        pos = self._managed_positions[ticket]

                    if getattr(pos, "is_user_manual", False) and not self.manage_manual_positions:
                        # Skip automated management of user manual trades
                        continue

                    # 1. Check if position was closed at broker level (SL or TP hit, or after close_position)
                    if ticket not in broker_pos_map:
                        # Check if broker assigned a real ticket to this order
                        matched_real_ticket = None
                        for real_t, bp in broker_pos_map.items():
                            if real_t in self._managed_positions:
                                continue  # Strictly prevent stealing another position's claimed ticket!
                            if bp.symbol == pos.symbol and (bp.type.value if hasattr(bp.type, "value") else str(bp.type)) == pos.direction:
                                if abs(bp.open_price - pos.open_price) < 1.0 or (now - pos.entry_time < 20.0):
                                    matched_real_ticket = real_t
                                    break
                        if matched_real_ticket is not None:
                            async with self._lock:
                                self._managed_positions.pop(ticket, None)
                                pos.ticket = matched_real_ticket
                                self._managed_positions[matched_real_ticket] = pos
                            live_p = broker_pos_map[matched_real_ticket]
                        elif now - pos.entry_time < 15.0 and not getattr(pos, "is_closing", False):
                            # Order is still in-flight or waiting for MT5 handshake
                            continue
                        else:
                            # Position was closed on broker (SL, TP, or confirmed external closure)
                            tick = await self.broker.get_tick(pos.symbol)
                            cur_p = (tick.bid if pos.direction == "BUY" else tick.ask) if tick and tick.bid > 0 else (getattr(pos, "close_trigger_price", None) or pos.open_price)
                            close_p = cur_p
                            reason = getattr(pos, "close_reason", None) or "Broker Closed (SL/TP)"
                            if not getattr(pos, "close_reason", None):
                                if pos.sl > 0:
                                    if pos.direction == "BUY" and cur_p <= pos.sl + 0.60:
                                        close_p = pos.sl
                                        reason = "Stop Loss Hit"
                                    elif pos.direction == "SELL" and cur_p >= pos.sl - 0.60:
                                        close_p = pos.sl
                                        reason = "Stop Loss Hit"
                                if pos.tp > 0:
                                    if pos.direction == "BUY" and cur_p >= pos.tp - 0.60:
                                        close_p = pos.tp
                                        reason = "Take Profit Hit"
                                    elif pos.direction == "SELL" and cur_p <= pos.tp + 0.60:
                                        close_p = pos.tp
                                        reason = "Take Profit Hit"

                            await self._on_position_closed(pos, reason=reason, current_price=close_p)
                            continue
                    else:
                        live_p = broker_pos_map[ticket]

                    current_price = live_p.current_price
                    pos.current_price = current_price
                    pos.profit = getattr(live_p, "profit", 0.0)

                    # Compute floating R-multiple
                    if pos.direction == "BUY":
                        floating_dist = current_price - pos.open_price
                    else:
                        floating_dist = pos.open_price - current_price

                    floating_r = floating_dist / max(pos.risk_r_dist, 0.01)
                    pos.r_multiple = round(floating_r, 2)

                    if floating_r > pos.highest_r_reached:
                        pos.highest_r_reached = round(floating_r, 2)

                    if pos.highest_r_reached > 0.0:
                        pos.peak_r_giveback_pct = round(max(0.0, (pos.highest_r_reached - floating_r) / pos.highest_r_reached * 100.0), 1)
                    else:
                        pos.peak_r_giveback_pct = 0.0

                    # If position is already in-flight closing, don't execute additional rules
                    if getattr(pos, "is_closing", False):
                        continue

                    # =============================================================
                    # INTELLIGENT RULE 1: Market Flip & Reversal Detection
                    # Cuts losses early (-0.15R to -0.65R) or banks profits if market turns!
                    # =============================================================
                    should_exit_flip, flip_reason = await self._check_market_flip(pos, floating_r, current_price, now)
                    if should_exit_flip:
                        logger.warning(
                            f"[MARKET FLIP ACTION] #{ticket} {pos.direction} exiting now: {flip_reason}!"
                        )
                        pos.is_closing = True
                        pos.close_reason = flip_reason
                        pos.close_trigger_price = current_price
                        await self.broker.close_position(ticket)
                        if getattr(self.broker, "_mt5_queue", None) is None:
                            await self._on_position_closed(pos, reason=flip_reason, current_price=current_price)
                        continue

                    # =============================================================
                    # INTELLIGENT RULE 2: Smart Pre-News Free-Roll Engine & Protection Lock
                    # 1. T-90s Free-Roll Lock: When news is within 90s, if floating profit >= +0.40R:
                    #    Lock SL to (Entry + Spread + 0.20R) and expand TP to 4.0R (Free-Roll Runner)!
                    # 2. T-60s Pre-News Scratch: When news is within 60s, if stalled (< +0.40R):
                    #    Scratch/close to prevent broker spread blowout & slippage at release (T-0s).
                    # 3. If in modest profit (+0.15R to +0.39R), lock Breakeven so it rides safely.
                    # 4. If fundamental news surprise severely opposes trade and position is in deep loss, exit.
                    # =============================================================
                    cal_state = self.news_reactor.check_calendar_state() if self.news_reactor else {}
                    is_news_freeze = cal_state.get("is_news_freeze", False) or cal_state.get("is_freeze", False)
                    countdown_sec = cal_state.get("countdown_seconds")

                    is_gold = any(k in pos.symbol.upper() for k in ["XAU", "GOLD"])
                    spread_buf = 0.25 if is_gold else 0.0002

                    # Check upcoming news proximity
                    is_near_release = is_news_freeze or (countdown_sec is not None and 0 < countdown_sec <= 90)

                    if is_near_release:
                        # Case A: T-90s Free-Roll Lock on strong winning trades (>= +0.40R)
                        if floating_r >= 0.40 and not getattr(pos, "is_freeroll_locked", False):
                            guaranteed_offset = 0.20 * pos.risk_r_dist
                            if pos.direction == "BUY":
                                new_sl = round(pos.open_price + spread_buf + guaranteed_offset, 2 if is_gold else 5)
                                new_tp = round(pos.open_price + (5.0 * pos.risk_r_dist), 2 if is_gold else 5)
                                is_tighter = (new_sl > pos.sl)
                            else:
                                new_sl = round(pos.open_price - spread_buf - guaranteed_offset, 2 if is_gold else 5)
                                new_tp = round(pos.open_price - (5.0 * pos.risk_r_dist), 2 if is_gold else 5)
                                is_tighter = (new_sl < pos.sl if pos.sl > 0 else True)

                            if is_tighter:
                                ok = await self.broker.modify_position(pos.ticket, sl=new_sl, tp=new_tp)
                                if ok:
                                    pos.sl = new_sl
                                    pos.tp = new_tp
                                    pos.is_freeroll_locked = True
                                    pos.ratchet_stage = max(pos.ratchet_stage, 2)
                                    pos.last_mod_time = time.time()
                                    logger.info(
                                        f"[PRE-NEWS FREE-ROLL LOCK] #{ticket} {pos.direction} locked SL to +0.20R profit floor ({new_sl}) "
                                        f"and expanded TP to 5.0R ({new_tp}) before upcoming news!"
                                    )
                                else:
                                    logger.warning(f"[PRE-NEWS FREE-ROLL FAILED] Broker rejected modify for #{ticket}")

                        # Case B: T-60s Pre-News Scratch on stalled trades (< +0.40R)
                        elif countdown_sec is not None and 0 < countdown_sec <= 60 and floating_r < 0.40:
                            reason_scratch = f"Pre-News T-60s Scratch (Stalled {floating_r:.2f}R < +0.40R)"
                            logger.info(f"[PRE-NEWS SCRATCH] Position #{ticket} closed at T-{countdown_sec}s: {reason_scratch}")
                            pos.is_closing = True
                            pos.close_reason = reason_scratch
                            pos.close_trigger_price = current_price
                            await self.broker.close_position(ticket)
                            if getattr(self.broker, "_mt5_queue", None) is None:
                                await self._on_position_closed(pos, reason=reason_scratch, current_price=current_price)
                            continue

                        # Case C: Modest profit breakeven lock (between +0.15R and +0.40R)
                        elif floating_r >= 0.15 and pos.ratchet_stage < 1 and not getattr(pos, "is_freeroll_locked", False):
                            new_sl = round(pos.open_price + spread_buf if pos.direction == "BUY" else pos.open_price - spread_buf, 2 if is_gold else 5)
                            is_tighter = (new_sl > pos.sl) if pos.direction == "BUY" else (new_sl < pos.sl if pos.sl > 0 else True)
                            if is_tighter:
                                ok = await self.broker.modify_position(pos.ticket, sl=new_sl, tp=pos.tp)
                                if ok:
                                    pos.sl = new_sl
                                    pos.ratchet_stage = 1
                                    pos.last_mod_time = time.time()
                                    logger.info(f"[CALCULATED NEWS PROTECTION] #{ticket} locked SL to Breakeven ({new_sl}) to ride news safely without closing!")
                                else:
                                    logger.warning(f"[CALCULATED NEWS PROTECTION FAILED] Broker rejected SL modify to {new_sl} for #{ticket}")

                        # Case D: Fundamental news bias check
                        bias_data = self.news_reactor.analyze_news_fundamental_bias(pos.symbol) if self.news_reactor else {}
                        bias = bias_data.get("bias", "NEUTRAL")
                        conf = bias_data.get("confidence", 0)

                        if bias == pos.direction:
                            # News surprise confirms this position direction - let it run!
                            logger.info(f"[NEWS BIAS ALIGNED] #{ticket} {pos.direction} matches fundamental news surprise ({conf}% confidence). Position kept open!")
                        elif bias != "NEUTRAL" and bias != pos.direction and conf >= 75 and floating_r < -0.70:
                            # Severe opposing fundamental release and already deep underwater
                            reason_news = f"Calculated News Exit (Opposing {bias} surprise {conf}%)"
                            logger.warning(f"[CALCULATED NEWS EXIT] #{ticket} closed: {reason_news}")
                            pos.is_closing = True
                            pos.close_reason = reason_news
                            pos.close_trigger_price = current_price
                            await self.broker.close_position(ticket)
                            if getattr(self.broker, "_mt5_queue", None) is None:
                                await self._on_position_closed(pos, reason=reason_news, current_price=current_price)
                            continue

                    # =============================================================
                    # INTELLIGENT RULE 3: Dead-Trade Scratch Timer
                    # If trade sits flat between -0.3R and +0.3R for > 12 minutes (720s)
                    # =============================================================
                    duration_sec = now - pos.entry_time
                    if duration_sec > 720 and (-0.30 <= floating_r <= 0.30):
                        logger.info(f"[DEAD_SETUP SCRATCH] Position #{ticket} stalled for {int(duration_sec/60)}m. Scratching.")
                        pos.is_closing = True
                        pos.close_reason = "Dead-Trade Scratch"
                        pos.close_trigger_price = current_price
                        await self.broker.close_position(ticket)
                        if getattr(self.broker, "_mt5_queue", None) is None:
                            await self._on_position_closed(pos, reason="Dead-Trade Scratch", current_price=current_price)
                        continue

                    # =============================================================
                    # INTELLIGENT RULE 4: Dynamic Take-Profit Expansion ("Let Winners Run")
                    # If price approaches TP with strong continuous momentum, push TP further out!
                    # =============================================================
                    if floating_r >= 0.70:
                        await self._evaluate_dynamic_tp(pos, floating_r, current_price)

                    # =============================================================
                    # INTELLIGENT RULE 5: Multi-Stage Volatility Trailing Stop-Loss Floor
                    # Tightens risk early, locks breakeven, and ratchets guaranteed profit floors!
                    # =============================================================
                    await self._evaluate_ratchet_floor(pos, floating_r, current_price)

                    # =============================================================
                    # INTELLIGENT RULE 6: Fast Scalp Velocity Stall Profit Lock
                    # Tightens SL to guaranteed profit floor instead of cutting runners short!
                    # =============================================================
                    if pos.sub_regime == "FAST_SCALP" and floating_r >= 1.50 and (now - pos.entry_time >= 90.0):
                        current_vel = self.news_reactor._latest_velocity or 0.0
                        is_velocity_dead = (current_vel < 0.50 * max(0.5, pos.entry_velocity))

                        if is_velocity_dead:
                            if pos.stall_start_time == 0.0:
                                pos.stall_start_time = now
                            elif now - pos.stall_start_time >= max(60.0, pos.stall_exit_seconds):
                                # Rather than dumping the runner, tighten stop loss to lock +1.05R guaranteed cash!
                                # This protects the gain while leaving the trade active to hit high TP (+3.5R - 6.0R+).
                                is_gold = any(k in pos.symbol.upper() for k in ["XAU", "GOLD"])
                                min_stall_floor = pos.risk_r_dist * 1.05
                                target_sl = round(pos.open_price + min_stall_floor if pos.direction == "BUY" else pos.open_price - min_stall_floor, 2 if is_gold else 5)
                                is_tighter = (pos.direction == "BUY" and target_sl > pos.sl) or (pos.direction == "SELL" and (pos.sl <= 0 or target_sl < pos.sl))
                                if is_tighter:
                                    ok = await self.broker.modify_position(pos.ticket, sl=target_sl, tp=pos.tp)
                                    if ok:
                                        pos.sl = target_sl
                                        pos.last_mod_time = now
                                        logger.info(
                                            f"[FAST SCALP PROFIT LOCK] Velocity stalled at +{floating_r:.2f}R. "
                                            f"Tightened SL to {target_sl:.2f} (Locking +1.05R profit floor) to let winner run to TP!"
                                        )
                                # Only if stalled for an extreme duration (>= 180s) and profit actively drops below +0.90R:
                                if now - pos.stall_start_time >= 180.0 and floating_r < 0.90:
                                    stall_reason = f"Fast Scalp Momentum Decay Bank (+{floating_r:.2f}R)"
                                    logger.info(
                                        f"[FAST SCALP MOMENTUM DECAY EXIT] Extended stall (>=180s) and profit decayed. "
                                        f"Banking cash now at +{floating_r:.2f}R for #{ticket}!"
                                    )
                                    pos.is_closing = True
                                    pos.close_reason = stall_reason
                                    pos.close_trigger_price = current_price
                                    await self.broker.close_position(ticket)
                                    if getattr(self.broker, "_mt5_queue", None) is None:
                                        await self._on_position_closed(pos, reason=stall_reason, current_price=current_price)
                                    continue
                        else:
                            pos.stall_start_time = 0.0

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in TradeManager loop: {e}", exc_info=True)
                await asyncio.sleep(1.0)

    async def _check_market_flip(
        self,
        pos: ManagedPosition,
        floating_r: float,
        current_price: float,
        now: float
    ) -> Tuple[bool, str]:
        """
        Intelligent Market Flip, Breakeven Scratch & Budget-Scaled Early Mitigation Cut Engine.
        Continuously studies live market structure, 1m/5m multi-timeframe trends, momentum collapse,
        and opposing volatility/order-flow:
        1. Breakeven Scratch: If trade was in profit (+0.05R+ or highest_r >= 0.15R) or is hovering flat,
           and live market studies show setup is failing/reversing, immediately scratch at breakeven/micro-gain.
        2. Early Mitigation Cut: If trade is floating slightly negative (-0.08R to -0.40R, loss within
           budget percentage limit) and structure breaks against it, close early to save 70%+ of SL risk.
        3. Asymmetric Deep Invalidation: If trade reaches <= -0.50R with severe adverse shock or SuperTrend break,
           cut before full -1.0R / -2.0R SL.
        """
        # 1. Enforce minimum hold time: wait it out, never open and close trades directly on noise!
        hold_time = now - pos.entry_time
        min_hold = getattr(self, "min_hold_time_seconds", 90.0)
        if hold_time < min_hold and floating_r < 0.40 and pos.highest_r_reached < 0.70:
            pos.exit_evaluation = "HOLD_AND_EXPAND"
            pos.exit_verdict_detail = f"Trade stabilizing ({int(hold_time)}s / {int(min_hold)}s min hold). Allowing setup to breathe."
            return False, ""

        if not getattr(self, "predictive_scratch_enabled", True):
            # Fallback check if predictive scratch disabled
            if floating_r > -0.75:
                pos.exit_evaluation = "HOLD_AND_EXPAND"
                pos.exit_verdict_detail = "Predictive scratch disabled; holding position."
                return False, ""

        # 2. Extract Account Equity & Calculate Floating USD
        eff_balance = 100.0
        if hasattr(self.broker, "get_account_info"):
            try:
                acc = await self.broker.get_account_info()
                if acc and getattr(acc, "balance", 0.0) > 0:
                    eff_balance = acc.balance
            except Exception:
                pass
        if eff_balance == 100.0 and self._start_of_day_balance > 0:
            eff_balance = self._start_of_day_balance
        elif eff_balance == 100.0 and hasattr(self, "risk_engine") and getattr(self.risk_engine, "base_budget_usd", 0.0) > 0:
            eff_balance = self.risk_engine.base_budget_usd

        is_gold = any(k in pos.symbol.upper() for k in ["XAU", "GOLD"])
        lot_mult = 100.0 if is_gold else 100000.0
        if hasattr(pos, "profit") and pos.profit != 0.0:
            floating_usd = pos.profit
        else:
            price_diff = (current_price - pos.open_price) if pos.direction == "BUY" else (pos.open_price - current_price)
            floating_usd = price_diff * pos.volume * lot_mult

        # 3. Multi-Timeframe Technical & Market Studies
        dfs: Dict[str, Any] = {}
        if self.data_hub:
            try:
                for tf, cnt in [("1m", 30), ("5m", 30), ("15m", 25)]:
                    df_tf = await self.data_hub.get_dataframe(pos.symbol, tf, count=cnt)
                    if df_tf is not None and len(df_tf) >= 8:
                        dfs[tf] = df_tf
            except Exception as e:
                logger.debug(f"Could not load dataframes for market study: {e}")

        trend_1m = "NEUTRAL"
        trend_5m = "NEUTRAL"
        if dfs:
            try:
                trends = TechnicalIndicators.get_4tf_trends(dfs)
                trend_1m = trends.get("1m", "NEUTRAL")
                trend_5m = trends.get("5m", "NEUTRAL")
            except Exception:
                pass

        # Check 5m SuperTrend
        st_5m = {}
        df_5m = dfs.get("5m")
        if df_5m is not None and len(df_5m) >= 15:
            try:
                st_5m = TechnicalIndicators.calculate_supertrend(
                    df_5m["high"], df_5m["low"], df_5m["close"], period=10, multiplier=2.5
                )
            except Exception:
                pass

        # Check 5m VWAP
        vwap_val = 0.0
        if df_5m is not None and len(df_5m) >= 5:
            try:
                vwap_res = TechnicalIndicators.calculate_vwap_and_distance(df_5m)
                vwap_val = float(vwap_res.get("vwap", 0.0))
            except Exception:
                pass

        # Check 5m StochRSI
        stoch_val = 50.0
        if df_5m is not None and len(df_5m) >= 15:
            try:
                stoch_res = TechnicalIndicators.calculate_stoch_rsi(df_5m["close"], 14, 3, 3)
                if isinstance(stoch_res, dict):
                    stoch_val = float(stoch_res.get("k", 50.0))
            except Exception:
                pass

        # Check 5m Candle metrics
        c5, o5, h5, l5 = current_price, current_price, current_price, current_price
        upper_wick_5m, lower_wick_5m = 0.0, 0.0
        if df_5m is not None and len(df_5m) > 0:
            c5 = float(df_5m["close"].iloc[-1])
            o5 = float(df_5m["open"].iloc[-1])
            h5 = float(df_5m["high"].iloc[-1])
            l5 = float(df_5m["low"].iloc[-1])
            rng5 = max(0.01, h5 - l5)
            upper_wick_5m = (h5 - max(o5, c5)) / rng5
            lower_wick_5m = (min(o5, c5) - l5) / rng5

        # Check 1m Structure & Candle metrics
        df_1m = dfs.get("1m")
        c1, o1, h1, l1 = current_price, current_price, current_price, current_price
        upper_wick_1m, lower_wick_1m = 0.0, 0.0
        if df_1m is not None and len(df_1m) > 0:
            c1 = float(df_1m["close"].iloc[-1])
            o1 = float(df_1m["open"].iloc[-1])
            h1 = float(df_1m["high"].iloc[-1])
            l1 = float(df_1m["low"].iloc[-1])
            rng1 = max(0.01, h1 - l1)
            upper_wick_1m = (h1 - max(o1, c1)) / rng1
            lower_wick_1m = (min(o1, c1) - l1) / rng1

        # Check Velocity Shock
        vel = getattr(self.news_reactor, "_latest_velocity", 1.0) or 1.0

        # Check Order Flow / OFI Delta from Scorer / PredictiveBrain
        ofi_delta = 0.0
        if hasattr(self, "scorer") and self.scorer:
            pb = getattr(self.scorer, "predictive_brain", None)
            if pb and hasattr(pb, "latest_evaluation") and pb.latest_evaluation:
                ens = getattr(pb.latest_evaluation, "ensemble", None)
                if isinstance(ens, dict):
                    ofi_delta = float(ens.get("delta_ratio_60s", 0.0))

        pnl_note = f"+${floating_usd:.2f}" if floating_usd >= 0 else f"-${abs(floating_usd):.2f}"

        # 4. Decision Rules:
        # =====================================================================
        # RULE GROUP 1: Dynamic Profit Harvesting & Trade Health Study
        # "Study if the trade can hold more or close" - don't rely passively on SL/TP!
        # =====================================================================
        is_profitable = (floating_r >= 0.50 or pos.highest_r_reached >= 0.85)

        if is_profitable:
            # Trigger C4: Peak Profit Protection (Surrendered >= 35% of peak profit)
            if pos.highest_r_reached >= 0.85 and pos.peak_r_giveback_pct >= 35.0 and floating_r >= 0.20:
                is_reversing_1m = (c1 < o1 or trend_1m == "BEAR") if pos.direction == "BUY" else (c1 > o1 or trend_1m == "BULL")
                if is_reversing_1m:
                    pos.exit_evaluation = "HARVEST_PROFIT_NOW"
                    pos.exit_verdict_detail = (
                        f"Peak Profit Protection: Surrendered {pos.peak_r_giveback_pct:.0f}% of peak "
                        f"(+{pos.highest_r_reached:.2f}R down to +{floating_r:.2f}R | {pnl_note})"
                    )
                    return True, f"Dynamic Profit Harvest: {pos.exit_verdict_detail}"

            # Trigger C1: Multi-Timeframe Structural Trend Breakdown & VWAP Loss
            if floating_r >= 0.50:
                if pos.direction == "BUY":
                    is_vwap_lost = (vwap_val > 0 and c5 < vwap_val and c5 < o5)
                    is_supertrend_lost = bool(st_5m.get("valid") and st_5m.get("is_bear"))
                    if (trend_1m == "BEAR" and trend_5m == "BEAR" and is_vwap_lost) or (is_supertrend_lost and c1 < o1):
                        pos.exit_evaluation = "HARVEST_PROFIT_NOW"
                        pos.exit_verdict_detail = (
                            f"Structural Breakdown: 1m & 5m Bearish below VWAP "
                            f"({c5:.2f} < {vwap_val:.2f} | +{floating_r:.2f}R, {pnl_note})"
                        )
                        return True, f"Dynamic Profit Harvest: {pos.exit_verdict_detail}"
                else: # SELL
                    is_vwap_lost = (vwap_val > 0 and c5 > vwap_val and c5 > o5)
                    is_supertrend_lost = bool(st_5m.get("valid") and st_5m.get("is_bull"))
                    if (trend_1m == "BULL" and trend_5m == "BULL" and is_vwap_lost) or (is_supertrend_lost and c1 > o1):
                        pos.exit_evaluation = "HARVEST_PROFIT_NOW"
                        pos.exit_verdict_detail = (
                            f"Structural Breakdown: 1m & 5m Bullish above VWAP "
                            f"({c5:.2f} > {vwap_val:.2f} | +{floating_r:.2f}R, {pnl_note})"
                        )
                        return True, f"Dynamic Profit Harvest: {pos.exit_verdict_detail}"

            # Trigger C2: Severe Exhaustion Rejection Wick at Liquidity Pool
            if floating_r >= 0.60:
                if pos.direction == "BUY":
                    max_wick = max(upper_wick_5m, upper_wick_1m)
                    if (upper_wick_5m >= 0.50 or upper_wick_1m >= 0.60) and c1 < o1 and stoch_val >= 78.0:
                        pos.exit_evaluation = "HARVEST_PROFIT_NOW"
                        pos.exit_verdict_detail = (
                            f"Liquidity Rejection: {max_wick*100:.0f}% upper wick & StochRSI overbought "
                            f"({stoch_val:.1f}) | +{floating_r:.2f}R, {pnl_note}"
                        )
                        return True, f"Dynamic Profit Harvest: {pos.exit_verdict_detail}"
                else: # SELL
                    max_wick = max(lower_wick_5m, lower_wick_1m)
                    if (lower_wick_5m >= 0.50 or lower_wick_1m >= 0.60) and c1 > o1 and stoch_val <= 22.0:
                        pos.exit_evaluation = "HARVEST_PROFIT_NOW"
                        pos.exit_verdict_detail = (
                            f"Liquidity Rejection: {max_wick*100:.0f}% lower wick & StochRSI oversold "
                            f"({stoch_val:.1f}) | +{floating_r:.2f}R, {pnl_note}"
                        )
                        return True, f"Dynamic Profit Harvest: {pos.exit_verdict_detail}"

            # Trigger C3: Institutional Order Flow Delta Inversion
            if floating_r >= 0.60:
                if pos.direction == "BUY" and ofi_delta <= -0.40 and c1 < o1:
                    pos.exit_evaluation = "HARVEST_PROFIT_NOW"
                    pos.exit_verdict_detail = (
                        f"Order Flow Inversion: Aggressive seller delta ({ofi_delta:+.2f}) on red bar "
                        f"| +{floating_r:.2f}R, {pnl_note}"
                    )
                    return True, f"Dynamic Profit Harvest: {pos.exit_verdict_detail}"
                elif pos.direction == "SELL" and ofi_delta >= 0.40 and c1 > o1:
                    pos.exit_evaluation = "HARVEST_PROFIT_NOW"
                    pos.exit_verdict_detail = (
                        f"Order Flow Inversion: Aggressive buyer delta ({ofi_delta:+.2f}) on green bar "
                        f"| +{floating_r:.2f}R, {pnl_note}"
                    )
                    return True, f"Dynamic Profit Harvest: {pos.exit_verdict_detail}"

            # Study shows trade can hold: check if we should tighten or hold for expansion
            if floating_r >= 0.40:
                wick_warn = (upper_wick_5m >= 0.35 or upper_wick_1m >= 0.40) if pos.direction == "BUY" else (lower_wick_5m >= 0.35 or lower_wick_1m >= 0.40)
                stoch_warn = (stoch_val >= 75.0) if pos.direction == "BUY" else (stoch_val <= 25.0)
                if wick_warn or stoch_warn:
                    pos.exit_evaluation = "TIGHTEN_AND_PROTECT"
                    pos.exit_verdict_detail = (
                        f"Momentum Deceleration: Opposing wick / StochRSI ({stoch_val:.1f}). "
                        f"Tightening ratchet stop floor behind price to lock gains."
                    )
                else:
                    pos.exit_evaluation = "HOLD_AND_EXPAND"
                    pos.exit_verdict_detail = (
                        f"Healthy Runner: Multi-timeframe trend supportive (1m {trend_1m} / 5m {trend_5m}). "
                        f"No exhaustion detected; holding runner for high targets (Peak: +{pos.highest_r_reached:.2f}R)."
                    )

        # =====================================================================
        # RULE GROUP 2: Breakeven Scratch (Protecting Capital on Fading Trades)
        # =====================================================================
        can_scratch_be = (
            pos.highest_r_reached >= 0.40 and
            -0.05 <= floating_r <= 0.15
        )
        if can_scratch_be:
            is_reversal = False
            rev_reasons = []
            if pos.direction == "BUY":
                if trend_1m == "BEAR" and trend_5m == "BEAR":
                    is_reversal = True
                    rev_reasons.append("1m & 5m Trend Bearish")
                elif st_5m.get("valid") and st_5m.get("is_bear"):
                    is_reversal = True
                    rev_reasons.append("5m SuperTrend Bearish Break")
            else:
                if trend_1m == "BULL" and trend_5m == "BULL":
                    is_reversal = True
                    rev_reasons.append("1m & 5m Trend Bullish")
                elif st_5m.get("valid") and st_5m.get("is_bull"):
                    is_reversal = True
                    rev_reasons.append("5m SuperTrend Bullish Break")

            if is_reversal:
                pos.exit_evaluation = "HARVEST_PROFIT_NOW"
                pos.exit_verdict_detail = f"Breakeven Scratch: {', '.join(rev_reasons)} | {floating_r:+.2f}R, {pnl_note}"
                return True, f"Predictive Invalidation: Breakeven Scratch ({', '.join(rev_reasons)} | {floating_r:+.2f}R, {pnl_note})"

        # =====================================================================
        # RULE GROUP 3: High-Certainty Adverse Invalidation (Only close when 100% sure to lose)
        # =====================================================================
        if floating_r <= -0.70:
            if st_5m.get("valid"):
                if pos.direction == "BUY" and st_5m.get("is_bear"):
                    pos.exit_evaluation = "HARVEST_PROFIT_NOW"
                    pos.exit_verdict_detail = f"Confirmed Invalidation: 5m SuperTrend Bearish Break ({floating_r:.2f}R)"
                    return True, f"Confirmed Invalidation: 5m SuperTrend Bearish Break ({floating_r:.2f}R)"
                elif pos.direction == "SELL" and st_5m.get("is_bull"):
                    pos.exit_evaluation = "HARVEST_PROFIT_NOW"
                    pos.exit_verdict_detail = f"Confirmed Invalidation: 5m SuperTrend Bullish Break ({floating_r:.2f}R)"
                    return True, f"Confirmed Invalidation: 5m SuperTrend Bullish Break ({floating_r:.2f}R)"
            if vel >= 3.5 * max(0.5, pos.entry_velocity):
                if pos.direction == "BUY" and c1 < pos.open_price:
                    pos.exit_evaluation = "HARVEST_PROFIT_NOW"
                    pos.exit_verdict_detail = f"Confirmed Invalidation: Severe Adverse Velocity Shock ({vel:.1f}x | {floating_r:.2f}R)"
                    return True, f"Confirmed Invalidation: Severe Adverse Velocity Shock ({vel:.1f}x | {floating_r:.2f}R)"
                elif pos.direction == "SELL" and c1 > pos.open_price:
                    pos.exit_evaluation = "HARVEST_PROFIT_NOW"
                    pos.exit_verdict_detail = f"Confirmed Invalidation: Severe Adverse Velocity Shock ({vel:.1f}x | {floating_r:.2f}R)"
                    return True, f"Confirmed Invalidation: Severe Adverse Velocity Shock ({vel:.1f}x | {floating_r:.2f}R)"

        # Default trade health study status when open and holding
        if not is_profitable:
            if floating_r >= -0.20:
                pos.exit_evaluation = "HOLD_AND_EXPAND"
                pos.exit_verdict_detail = f"Trade developing ({floating_r:+.2f}R). Structure intact, allowing setup breathing room."
            else:
                pos.exit_evaluation = "HOLD_AND_EXPAND"
                pos.exit_verdict_detail = f"Underlying structure holding ({floating_r:+.2f}R). Not invalidating unless 100% confirmed loss."

        return False, ""

    async def _evaluate_dynamic_tp(
        self,
        pos: ManagedPosition,
        floating_r: float,
        current_price: float
    ) -> None:
        """
        Dynamic Take-Profit Expansion ("Let Winners Run").
        When price surges towards the target with strong trend velocity,
        dynamically push TP further out to capture massive institutional runs.
        """
        if pos.risk_r_dist <= 0:
            return

        current_target_r = abs(pos.tp - pos.open_price) / pos.risk_r_dist
        # If we have reached >= 60% of current TP or >= 1.50R, and target is under 10.0R:
        if (floating_r >= 0.60 * current_target_r or floating_r >= 1.50) and current_target_r < 10.0:
            # Check 1m and 5m momentum
            df_1m = await self.data_hub.get_dataframe(pos.symbol, "1m", count=5)
            df_5m = await self.data_hub.get_dataframe(pos.symbol, "5m", count=5)

            is_trending_strong = False
            if df_1m is not None and len(df_1m) >= 2:
                c1 = df_1m["close"].iloc[-1]
                o1 = df_1m["open"].iloc[-1]
                h1 = df_1m["high"].iloc[-1]
                l1 = df_1m["low"].iloc[-1]
                rng1 = max(0.01, h1 - l1)

                if pos.direction == "BUY" and c1 > o1:
                    upper_wick = h1 - c1
                    if upper_wick / rng1 < 0.40: # Clean bull candle with little opposing wick
                        is_trending_strong = True
                elif pos.direction == "SELL" and c1 < o1:
                    lower_wick = c1 - l1
                    if lower_wick / rng1 < 0.40:
                        is_trending_strong = True

            # If 5m also supports trend direction, confirm expansion
            if df_5m is not None and len(df_5m) >= 2:
                c5 = df_5m["close"].iloc[-1]
                o5 = df_5m["open"].iloc[-1]
                if pos.direction == "BUY" and c5 > o5:
                    is_trending_strong = True
                elif pos.direction == "SELL" and c5 < o5:
                    is_trending_strong = True

            if is_trending_strong:
                # Expand TP by 1.5R to 2.0R to capture high profits
                expansion_step = 2.0 if current_target_r >= 4.0 else 1.5
                new_target_r = round(min(10.0, current_target_r + expansion_step), 1)
                is_gold = any(k in pos.symbol.upper() for k in ["XAU", "GOLD"])
                if pos.direction == "BUY":
                    new_tp = round(pos.open_price + (pos.risk_r_dist * new_target_r), 2 if is_gold else 5)
                else:
                    new_tp = round(pos.open_price - (pos.risk_r_dist * new_target_r), 2 if is_gold else 5)

                if new_tp != pos.tp:
                    old_tp = pos.tp
                    ok = await self.broker.modify_position(pos.ticket, sl=pos.sl, tp=new_tp)
                    if ok:
                        pos.tp = new_tp
                        pos.current_regime = "ADAPTIVE_RUNNER"
                        pos.last_mod_time = time.time()
                        logger.info(
                            f"[DYNAMIC TP EXPANSION] #{pos.ticket} surging strongly (+{floating_r:.2f}R)! "
                            f"Extended TP from {old_tp:.2f} to {new_tp:.2f} (+{new_target_r:.1f}R) to capture maximum profit!"
                        )
                    else:
                        logger.warning(f"[DYNAMIC TP EXPANSION FAILED] Broker rejected TP modify to {new_tp} for #{pos.ticket}")

    async def _evaluate_ratchet_floor(self, pos: ManagedPosition, floating_r: float, current_price: float) -> None:
        """
        Enforce One-Way Ratchet Floor on Broker Stop Loss:
        - Allows trade ample room to breathe during early stage pullbacks.
        - +0.80R+ -> Breakeven (Entry + Spread Buffer)
        - +1.00R+ -> Lock +0.45R Guaranteed Profit Floor
        - +1.50R+ -> Lock +0.85R Guaranteed Profit Floor
        - +2.00R+ -> Lock +1.35R Guaranteed Profit Floor
        - +2.50R+ -> Volatility-Adaptive Structural Trailing behind price
        - +4.00R+ -> High-Profit Institutional Runner Trailing (Tight protection behind price)
        """
        is_gold = any(k in pos.symbol.upper() for k in ["XAU", "GOLD"])
        spread_buf = 0.25 if is_gold else 0.0002
        new_sl = None

        candidate_stage = pos.ratchet_stage

        # Stage 1: Breakeven Lock at +0.80R+ (gives trade full breathing room)
        # Or at +0.50R if AI Study detects deceleration and evaluates TIGHTEN_AND_PROTECT
        trigger_be = 0.50 if getattr(pos, "exit_evaluation", "") == "TIGHTEN_AND_PROTECT" else max(0.80, pos.be_lock_r)
        if pos.ratchet_stage < 1 and floating_r >= trigger_be:
            candidate_stage = 1
            if pos.direction == "BUY":
                new_sl = round(pos.open_price + spread_buf, 2 if is_gold else 5)
            else:
                new_sl = round(pos.open_price - spread_buf, 2 if is_gold else 5)

        # Stage 2: Lock Profit at +1.00R (Locks +0.45R) (or +0.75R if TIGHTEN_AND_PROTECT)
        elif pos.ratchet_stage < 2 and (floating_r >= 1.00 or (floating_r >= 0.75 and getattr(pos, "exit_evaluation", "") == "TIGHTEN_AND_PROTECT")):
            candidate_stage = 2
            lock_dist = pos.risk_r_dist * 0.45
            if pos.direction == "BUY":
                new_sl = round(pos.open_price + lock_dist, 2 if is_gold else 5)
            else:
                new_sl = round(pos.open_price - lock_dist, 2 if is_gold else 5)

        # Stage 3: Lock Profit at +1.50R (Locks +0.85R)
        elif pos.ratchet_stage < 3 and floating_r >= 1.50:
            candidate_stage = 3
            lock_dist = pos.risk_r_dist * 0.85
            if pos.direction == "BUY":
                new_sl = round(pos.open_price + lock_dist, 2 if is_gold else 5)
            else:
                new_sl = round(pos.open_price - lock_dist, 2 if is_gold else 5)

        # Stage 4: Lock Profit at +2.00R (Locks +1.35R)
        elif pos.ratchet_stage < 4 and floating_r >= 2.00:
            candidate_stage = 4
            lock_dist = pos.risk_r_dist * 1.35
            if pos.direction == "BUY":
                new_sl = round(pos.open_price + lock_dist, 2 if is_gold else 5)
            else:
                new_sl = round(pos.open_price - lock_dist, 2 if is_gold else 5)

        # Stage 5: Dynamic Structural Trailing at +2.50R to +4.00R
        elif floating_r >= 2.50 and floating_r < 4.00:
            candidate_stage = 5
            trail_dist = pos.risk_r_dist * 0.85
            min_lock = pos.risk_r_dist * 1.60
            if pos.direction == "BUY":
                candidate_sl = max(pos.open_price + min_lock, round(current_price - trail_dist, 2 if is_gold else 5))
                if candidate_sl > pos.sl:
                    new_sl = round(candidate_sl, 2 if is_gold else 5)
            else:
                candidate_sl = min(pos.open_price - min_lock, round(current_price + trail_dist, 2 if is_gold else 5))
                if pos.sl <= 0 or candidate_sl < pos.sl:
                    new_sl = round(candidate_sl, 2 if is_gold else 5)

        # Stage 6: High-Profit Institutional Runner Trailing (+4.00R+)
        elif floating_r >= 4.00:
            candidate_stage = 6
            trail_dist = pos.risk_r_dist * 0.70
            min_lock = pos.risk_r_dist * 2.50  # Guarantee at least +2.50R profit!
            if pos.direction == "BUY":
                candidate_sl = max(pos.open_price + min_lock, round(current_price - trail_dist, 2 if is_gold else 5))
                if candidate_sl > pos.sl:
                    new_sl = round(candidate_sl, 2 if is_gold else 5)
            else:
                candidate_sl = min(pos.open_price - min_lock, round(current_price + trail_dist, 2 if is_gold else 5))
                if pos.sl <= 0 or candidate_sl < pos.sl:
                    new_sl = round(candidate_sl, 2 if is_gold else 5)

        if new_sl is not None:
            # Enforce protective stop: NEVER loosen an existing stop loss!
            is_tightening = False
            if pos.direction == "BUY" and new_sl > pos.sl:
                is_tightening = True
            elif pos.direction == "SELL" and (pos.sl <= 0 or new_sl < pos.sl):
                is_tightening = True

            if is_tightening:
                ok = await self.broker.modify_position(pos.ticket, sl=new_sl, tp=pos.tp)
                if ok:
                    pos.sl = new_sl
                    pos.ratchet_stage = candidate_stage
                    pos.last_mod_time = time.time()
                    logger.info(f"[RATCHET SL UPDATED] #{pos.ticket} SL moved to {new_sl:.2f} (Stage {candidate_stage})")
                else:
                    logger.warning(f"[RATCHET SL REJECTED] Broker rejected SL modify to {new_sl:.2f} for #{pos.ticket}")

    async def _on_position_closed(
        self,
        pos: ManagedPosition,
        reason: str,
        current_price: Optional[float] = None
    ) -> None:
        """Handle position closure, record mathematically exact P&L, update persistent SQLite history, and adjust circuit breaker."""
        async with self._lock:
            self._managed_positions.pop(pos.ticket, None)

        pos_id = str(pos.ticket)
        if pos_id in self._recorded_tickets:
            logger.debug(f"Position #{pos_id} already finalized and recorded. Skipping duplicate record.")
            return
        self._recorded_tickets.add(pos_id)

        pos.is_closed = True
        pos.close_reason = reason
        now = time.time()

        deal = None
        if hasattr(self.broker, "get_closed_deal"):
            try:
                deal = await self.broker.get_closed_deal(pos.ticket)
            except Exception as e:
                logger.warning(f"Error fetching closed deal for #{pos.ticket}: {e}")

        is_gold = any(k in pos.symbol.upper() for k in ["XAU", "GOLD"])
        contract_size = 100.0 if is_gold else 100000.0

        if deal is not None:
            close_p = deal.close_price
            # Realized net P&L including broker commissions and swap
            pnl = round(deal.profit + deal.commission + deal.swap, 2)
            monetary_risk = max(pos.risk_r_dist * pos.volume * contract_size, 0.01)
            pnl_r = round(pnl / monetary_risk, 2)
            logger.info(
                f"[BROKER EXECUTION DEAL] #{pos.ticket} executed deal confirmed: Close: {close_p:.2f} | "
                f"Net Realized: ${pnl:+.2f} (Profit: ${deal.profit:.2f}, Comm: ${deal.commission:.2f}, Swap: ${deal.swap:.2f})"
            )
        else:
            close_p = current_price if (current_price and current_price > 0) else getattr(pos, "close_trigger_price", None)
            if close_p is None:
                tick = await self.broker.get_tick(pos.symbol)
                if tick and tick.bid > 0:
                    close_p = tick.bid if pos.direction == "BUY" else tick.ask
                elif "Stop Loss" in reason or "SL" in reason:
                    close_p = pos.sl if pos.sl > 0 else pos.open_price
                elif "Take Profit" in reason or "TP" in reason:
                    close_p = pos.tp if pos.tp > 0 else pos.open_price
                else:
                    close_p = pos.open_price

            # If close_p ended up identical to open_price but pos.profit exists, reconstruct true exit price
            if abs(close_p - pos.open_price) < 0.01 and abs(getattr(pos, "profit", 0.0)) > 0.05:
                price_delta = pos.profit / (pos.volume * contract_size)
                close_p = round(pos.open_price + price_delta if pos.direction == "BUY" else pos.open_price - price_delta, 2 if is_gold else 5)

            if pos.direction == "BUY":
                floating_dist = close_p - pos.open_price
            else:
                floating_dist = pos.open_price - close_p

            pnl_r = round(floating_dist / max(pos.risk_r_dist, 0.01), 2)
            # Mathematically exact dollar P&L
            pnl = round(floating_dist * pos.volume * contract_size, 2)

        # Deterministic ticket integer
        if str(pos.ticket).isdigit():
            ticket_int = int(pos.ticket)
        else:
            ticket_int = abs(hash((pos.symbol, pos.direction, round(pos.open_price, 2), int(pos.entry_time)))) % 100000000

        # Persistent storage in SQLite trade_history table
        if self.history_store:
            try:
                await self.history_store.record_trade(
                    ticket=ticket_int,
                    symbol=pos.symbol,
                    direction=pos.direction,
                    volume=pos.volume,
                    open_price=pos.open_price,
                    close_price=close_p,
                    open_time=pos.entry_time,
                    close_time=now,
                    pnl=pnl,
                    pnl_r=pnl_r,
                    regime=pos.current_regime,
                    sub_regime=pos.sub_regime,
                    close_reason=reason,
                )
            except Exception as e:
                logger.error(f"Error persisting trade to history store: {e}")

        acc = await self.broker.get_account_info()
        logger.info(
            f"[POSITION FINALIZED] #{pos.ticket} closed ({reason}) | P&L: ${pnl:+.2f} ({pnl_r:+.2f}R) | "
            f"Open: {pos.open_price:.2f} -> Close: {close_p:.2f}. Balance: {acc.balance:.2f} {acc.currency}"
        )

        # Daily realized P&L and circuit breaker metrics
        self._daily_realized_pnl += pnl
        is_loss = (pnl < -0.05)
        if is_loss:
            self._daily_losses_count += 1
            self._consecutive_losses += 1
            self._last_loss_direction = pos.direction
            self._last_loss_price = pos.open_price
            self._last_loss_time = now

            if self.circuit_breaker_enabled:
                # Intelligent post-loss settlement buffer
                self._post_loss_cooldown_until = now + self.post_loss_buffer_seconds
                self._last_order_time = now + 8.0
                logger.warning(
                    f"[POST-LOSS BUFFER] Trade #{pos.ticket} closed with loss (${pnl:.2f}). "
                    f"Consecutive losses: {self._consecutive_losses}/{self.max_consecutive_losses}, "
                    f"Daily losses: {self._daily_losses_count}/{self.max_daily_losses if self.max_daily_losses > 0 else 'Unlimited'}. Buffer active for {self.post_loss_buffer_seconds}s."
                )
                if self.max_consecutive_losses > 0 and self._consecutive_losses >= self.max_consecutive_losses:
                    self.pause_trading(minutes=15)
                    logger.error(f"[CIRCUIT BREAKER ACTIVATED] Max consecutive losses hit ({self._consecutive_losses}). Auto-trading paused 15m.")
                elif self.max_daily_losses > 0 and self._daily_losses_count >= self.max_daily_losses:
                    self.pause_trading(minutes=30)
                    logger.error(f"[CIRCUIT BREAKER ACTIVATED] Max daily losses hit ({self._daily_losses_count}). Auto-trading paused 30m.")
            else:
                self._post_loss_cooldown_until = 0.0
                self._pause_until = 0.0
                logger.info(
                    f"[TRADE CLOSED] Trade #{pos.ticket} closed with loss (${pnl:.2f}). "
                    f"Circuit breaker is disabled — trading remains always enabled."
                )
        elif pnl > 0.05:
            self._consecutive_losses = 0
            logger.info(f"[PROFITABLE TRADE] Trade #{pos.ticket} closed with profit (${pnl:.2f}). Consecutive loss counter reset.")

    def reset_cooldown(self) -> None:
        """Instantly clear post-loss cooldown, loss counters, and circuit breaker."""
        self._post_loss_cooldown_until = 0.0
        self._pause_until = 0.0
        self._last_order_time = 0.0
        self._last_loss_time = 0.0
        self._daily_losses_count = 0
        self._consecutive_losses = 0
        self._contradictory_state_active = False
        self._contradictory_state_reason = ""
        logger.info("[COOLDOWN RESET] Cooldown and circuit breaker manually reset.")

    async def emergency_close_all(self) -> int:
        """1-Click Emergency Button: Closes all positions immediately."""
        logger.warning("[EMERGENCY ACTION] Closing all active positions across broker!")
        count = await self.broker.close_all_positions()
        async with self._lock:
            self._managed_positions.clear()

        # Query broker state again and verify actual result
        remaining = await self.broker.get_open_positions()
        if remaining:
            rem_tickets = [p.ticket for p in remaining]
            self._contradictory_state_active = True
            self._contradictory_state_reason = f"CRITICAL: Emergency close all left {len(remaining)} position(s) open on broker: {rem_tickets}"
            self.pause_trading(minutes=1440)
            logger.critical(f"[EMERGENCY ACTION FAILURE] {self._contradictory_state_reason}")
            if self.history_store:
                await self.history_store.record_audit_event("CRITICAL_CLOSE_FAILED", self._contradictory_state_reason)
        return count

    async def emergency_lock_breakeven_all(self) -> int:
        """1-Click Emergency Action: Locks SL to Breakeven on all profitable positions."""
        logger.info("[EMERGENCY ACTION] Moving SL to Breakeven on all profitable positions...")
        positions = await self.broker.get_open_positions()
        count = 0
        spread_buf = 0.25

        for p in positions:
            if p.profit > 0:
                if p.type == PositionType.BUY:
                    new_sl = round(p.open_price + spread_buf, 2)
                else:
                    new_sl = round(p.open_price - spread_buf, 2)
                ok = await self.broker.modify_position(p.ticket, sl=new_sl, tp=p.tp)
                if ok:
                    count += 1
        return count

    def get_managed_positions(self) -> List[Dict[str, Any]]:
        """Return snapshot of managed positions for Web UI."""
        return [p.to_dict() for p in self._managed_positions.values()]

    def get_daily_status(self) -> Dict[str, Any]:
        """Return daily risk metrics and circuit breaker status."""
        now = time.time()
        return {
            "daily_losses": self._daily_losses_count,
            "max_daily_losses": self.max_daily_losses,
            "consecutive_losses": self._consecutive_losses,
            "max_consecutive_losses": self.max_consecutive_losses,
            "daily_realized_pnl": round(self._daily_realized_pnl, 2),
            "max_daily_drawdown_pct": self.max_daily_drawdown_pct,
            "start_of_day_balance": round(self._start_of_day_balance, 2),
            "circuit_breaker_enabled": self.circuit_breaker_enabled,
            "is_paused": self.is_paused(),
            "pause_seconds_left": max(0, int(self._pause_until - now)) if self.circuit_breaker_enabled else 0,
            "post_loss_cooldown_seconds_left": max(0, int(self._post_loss_cooldown_until - now)) if self.circuit_breaker_enabled else 0,
        }

    async def execute_planned_limit(
        self,
        planned_limit: Dict[str, Any],
        scorer: Any,
    ) -> OrderResult:
        """
        Execute an institutional planned BUY_LIMIT or SELL_LIMIT order.
        Verifies no duplicate pending order is already active for this direction.
        """
        if not planned_limit or not planned_limit.get("limit_price"):
            return OrderResult(success=False, error="Invalid planned limit payload")

        limit_type = planned_limit.get("order_type", "BUY_LIMIT")

        # Check existing pending limit orders to avoid duplicates and hedging
        limit_dir = "BUY" if "BUY" in limit_type.upper() else "SELL"
        opposing_type = "SELL_LIMIT" if limit_dir == "BUY" else "BUY_LIMIT"

        async with self._lock:
            for lo in self._pending_limit_orders.values():
                lo_ot = getattr(lo, "order_type", "")
                if lo_ot == limit_type:
                    return OrderResult(success=False, error=f"{limit_type} already active on broker (#{getattr(lo, 'ticket', '')})")
                if lo_ot == opposing_type:
                    return OrderResult(success=False, error=f"Hedging forbidden: Opposing {opposing_type} already active on broker (#{getattr(lo, 'ticket', '')})")

        from ui.server import ctx as s_ctx
        mt5_orders = getattr(s_ctx, "mt5_pending_orders", []) or []
        for mo in mt5_orders:
            mo_type = str(mo.get("type", "")).upper()
            if limit_type in mo_type:
                return OrderResult(success=False, error=f"{limit_type} already pending on MT5 (#{mo.get('ticket')})")
            if opposing_type in mo_type:
                return OrderResult(success=False, error=f"Hedging forbidden: Opposing {opposing_type} already pending on MT5 (#{mo.get('ticket')})")

        active_positions = await self.broker.get_open_positions()
        is_gold = any(k in planned_limit.get("symbol", "").upper() for k in ["XAU", "GOLD"])
        for p in active_positions:
            p_sym = p.symbol.upper()
            if (is_gold and any(k in p_sym for k in ["XAU", "GOLD"])) or (p_sym == planned_limit.get("symbol", "").upper()):
                p_dir = (p.type.value if hasattr(p.type, "value") else str(p.type)).upper()
                if p_dir != limit_dir:
                    return OrderResult(success=False, error=f"Hedging forbidden: Active {p_dir} position #{p.ticket} already open on {p.symbol}.")

        sig = scorer.create_signal_from_planned_limit(planned_limit)
        return await self.execute_trade(sig)

    async def reconcile_planned_limits(
        self,
        planned_limits: Dict[str, Any],
        current_bid: float,
        current_ask: float,
    ) -> List[int]:
        """
        Recalculate and remove pending limit orders if market conditions invalidate them.
        If price has drifted away, broken invalidation, or the newly recalculated optimal
        institutional level has shifted away by > $1.50, cancels the order on MT5.
        """
        cancelled_tickets: List[int] = []
        now = time.time()
        p_buy = planned_limits.get("planned_buy") if planned_limits else None
        p_sell = planned_limits.get("planned_sell") if planned_limits else None

        async with self._lock:
            current_pending = list(self._pending_limit_orders.items())

        for p_ticket, lo in current_pending:
            order_type = getattr(lo, "order_type", "")
            o_price = float(getattr(lo, "price", 0.0))
            cancel_reason = None

            # 1. Check recalculation shift
            if order_type == "BUY_LIMIT" and p_buy:
                new_opt_price = float(p_buy.get("limit_price", 0.0))
                if abs(new_opt_price - o_price) >= 1.50:
                    cancel_reason = f"Recalculated Buy Limit level shifted from {o_price:.2f} to {new_opt_price:.2f}"
            elif order_type == "SELL_LIMIT" and p_sell:
                new_opt_price = float(p_sell.get("limit_price", 0.0))
                if abs(new_opt_price - o_price) >= 1.50:
                    cancel_reason = f"Recalculated Sell Limit level shifted from {o_price:.2f} to {new_opt_price:.2f}"

            # 2. Check drift from current market
            mid_now = (current_bid + current_ask) / 2.0
            if abs(mid_now - o_price) > 3.50:
                cancel_reason = f"Price drifted {abs(mid_now - o_price):.2f} pts away from limit {o_price:.2f}"

            # 3. Check invalidation level
            inval = float(getattr(lo, "invalidation_price", 0.0))
            if inval > 0:
                if order_type == "BUY_LIMIT" and current_bid <= inval:
                    cancel_reason = f"Market breached Buy Invalidation level {inval:.2f}"
                elif order_type == "SELL_LIMIT" and current_ask >= inval:
                    cancel_reason = f"Market breached Sell Invalidation level {inval:.2f}"

            # 4. Timeout (90s)
            c_time = float(getattr(lo, "created_time", now))
            if (now - c_time) > 90.0:
                cancel_reason = "Order timed out after 90s without fill"

            if cancel_reason:
                logger.info(f"[LIMIT RECALCULATED & REMOVED] Order #{p_ticket} {order_type} @ {o_price:.2f} removed: {cancel_reason}")
                ok = await self.broker.cancel_order(p_ticket)
                if ok:
                    cancelled_tickets.append(int(p_ticket))
                    async with self._lock:
                        self._pending_limit_orders.pop(p_ticket, None)

        return cancelled_tickets
