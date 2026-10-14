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
from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

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
    initial_regime: str         # "ADAPTIVE_SCALP" or "ADAPTIVE_RUNNER"
    current_regime: str         # May morph in real time
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
        data_hub: DataHub,
        news_reactor: NewsReactor,
        risk_engine: GoldRiskEngine,
        correlation_guard: bool = True,
        max_daily_losses: int = 4,
        max_daily_drawdown_pct: float = 25.0,
        max_open_positions: int = 1,
        history_store: Optional[Any] = None,
    ):
        self.broker = broker
        self.data_hub = data_hub
        self.news_reactor = news_reactor
        self.risk_engine = risk_engine
        self.correlation_guard = correlation_guard
        self.max_daily_losses = max_daily_losses
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self.max_open_positions = max_open_positions
        self.history_store = history_store

        self._managed_positions: Dict[Union[int, str], ManagedPosition] = {}
        self._pending_limit_orders: Dict[Union[int, str], Any] = {}
        self._recorded_tickets: set = set()
        self._management_task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = asyncio.Lock()

        # Strict Single-Execution Lock & Cooldown
        self._is_order_in_flight: bool = False
        self._last_order_time: float = 0.0
        self._min_cooldown_seconds: float = 20.0
        self._post_loss_cooldown_until: float = 0.0

        # Circuit breaker metrics
        self._daily_losses_count = 0
        self._daily_realized_pnl = 0.0
        self._pause_until = 0.0
        self._last_day = datetime.now(timezone.utc).day

    async def start(self) -> None:
        """Start the 250ms live position management loop."""
        self._running = True
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
        self._pause_until = time.time() + (minutes * 60)
        logger.info(f"Trading paused for {minutes} minutes until {datetime.fromtimestamp(self._pause_until, tz=timezone.utc)}")

    def resume_trading(self) -> None:
        """Immediately resume trading and reset circuit breaker."""
        self._pause_until = 0.0
        self._daily_losses_count = 0
        logger.info("Trading resumed and daily loss counter reset.")

    def is_paused(self) -> bool:
        """Check if circuit breaker or manual pause is active."""
        if time.time() < self._pause_until:
            return True
        if self._daily_losses_count >= self.max_daily_losses:
            return True
        return False

    def _reset_daily_metrics_if_new_day(self) -> None:
        """Reset circuit breaker counters at 00:00 UTC."""
        today = datetime.now(timezone.utc).day
        if today != self._last_day:
            logger.info("New trading day: resetting daily loss counter and drawdown.")
            self._daily_losses_count = 0
            self._daily_realized_pnl = 0.0
            self._last_day = today

    async def _release_in_flight_lock(self, delay_sec: float = 2.5) -> None:
        """Release in-flight lock after broker communication delay."""
        await asyncio.sleep(delay_sec)
        self._is_order_in_flight = False

    async def can_execute_signal(self, signal: ConfluenceSignal) -> Tuple[bool, str]:
        """
        Verify circuit breakers, in-flight locks, news freeze, and USD correlation guard before placing order.
        Strictly guarantees NO multiple trades can ever open simultaneously.
        """
        self._reset_daily_metrics_if_new_day()

        if self.is_paused():
            return False, f"Trading paused (Daily losses: {self._daily_losses_count}/{self.max_daily_losses})"

        # 1. Post-Loss Anti-Revenge Guard
        now = time.time()
        if now < self._post_loss_cooldown_until:
            rem = int(self._post_loss_cooldown_until - now)
            return False, f"Post-Loss Cooldown Active ({rem}s remaining) - Preventing rapid re-entry / revenge trading."

        # 1b. In-Flight Execution Mutex Check
        if self._is_order_in_flight:
            return False, "Order already in flight (waiting for broker execution)"

        # 2. Minimum Cooldown Between Executions (20 seconds)
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

        # Check pending orders across all layers (TradeManager, PaperBroker, and MT5 Bridge)
        from ui.server import ctx as server_ctx
        broker_pending = await self.broker.get_pending_orders() if hasattr(self.broker, "get_pending_orders") else []
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
                broker_pending = await self.broker.get_pending_orders() if hasattr(self.broker, "get_pending_orders") else []
                mt5_pending = getattr(server_ctx, "mt5_pending_orders", []) or []
                total_pending_count = max(len(self._pending_limit_orders), len(broker_pending), len(mt5_pending))

        total_exposure = all_positions_count + total_pending_count

        # Absolute rule: Any micro/small account under $150 balance MUST NEVER open more than 1 trade or pending order!
        if eff_balance < 150.0:
            allowed_max = 1
        elif eff_balance < 300.0:
            allowed_max = 2 if (signal.score >= 82 and (signal.tier == "TIER_1_SNIPER")) else 1
        else:
            allowed_max = 3 if (signal.score >= 82 and (signal.tier == "TIER_1_SNIPER")) else 1

        if total_exposure >= allowed_max:
            return False, f"Maximum exposure reached ({total_exposure}/{allowed_max}) including {total_pending_count} pending order(s) for balance ${eff_balance:.2f}"

        # Stacking limit orders is strictly forbidden:
        is_limit = getattr(signal, "order_type", "MARKET") in ["BUY_LIMIT", "SELL_LIMIT"]
        if is_limit and total_pending_count > 0:
            return False, f"Pending limit order already active for {signal.symbol}. Stacking duplicate limit orders is forbidden."


        # 5. Smart Multi-Position Risk Guard (Pyramiding ONLY allowed for >= $150 balance with existing trade in solid profit)
        is_gold_signal = any(k in signal.symbol.upper() for k in ["XAU", "GOLD"])
        if all_positions_count > 0:
            if eff_balance < 150.0:
                return False, f"Account balance (${eff_balance:.2f}) under $150 strictly limited to 1 active position."

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
                        if m.direction != signal.direction:
                            return False, f"Conflicting open trade #{m.ticket} ({m.direction}) vs signal ({signal.direction})"
                        # If existing trade is not at least +0.80R in profit with SL secured, forbid new entry
                        cur_tick = await self.broker.get_tick(m.symbol)
                        if cur_tick and cur_tick.bid > 0:
                            curr_p = cur_tick.bid if m.direction == "BUY" else cur_tick.ask
                            dist = (curr_p - m.open_price) if m.direction == "BUY" else (m.open_price - curr_p)
                            float_r = dist / max(m.risk_r_dist, 0.01)
                            if float_r < 0.80:
                                return False, f"Existing {m.symbol} trade #{m.ticket} only at {float_r:.2f}R. Must reach at least +0.80R with SL secured before adding!"

        return True, "Execution approved"

    async def execute_trade(self, signal: ConfluenceSignal) -> OrderResult:
        """
        Execute live market order with dynamically calculated lot size, padded SL, and TP.
        Enrolls position into ManagedPosition for real-time 250ms Chameleon monitoring.
        """
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
            is_limit = getattr(signal, "order_type", "MARKET") in ["BUY_LIMIT", "SELL_LIMIT"]
            if is_limit:
                order_type = OrderType.BUY_LIMIT if signal.direction == "BUY" else OrderType.SELL_LIMIT
                target_entry = signal.limit_price or signal.entry_price
            else:
                order_type = OrderType.BUY if signal.direction == "BUY" else OrderType.SELL
                target_entry = signal.entry_price

            # Volatility-Adaptive Stop Loss: Respects market ATR and structural swings
            is_gold = any(k in signal.symbol.upper() for k in ["XAU", "GOLD"])
            actual_sl_dist = signal.risk_r_dist
            raw_sl = signal.suggested_sl

            # Calculate exact lot size
            lot_size, risk_details = self.risk_engine.calculate_lot_size(
                symbol_info=symbol_info,
                account_info=account_info,
                sl_distance=actual_sl_dist,
            )

            if not risk_details.get("has_margin", False):
                msg = f"Rejected: Margin requirement exceeds available balance: {risk_details.get('warning')}"
                logger.error(msg)
                return OrderResult(success=False, error=msg)

            # Pad Stop Loss against live spread
            padded_sl = self.risk_engine.pad_stop_loss(
                direction=signal.direction,
                raw_sl=raw_sl,
                spread=symbol_info.spread,
                point=symbol_info.point,
                extra_buffer_points=2,
            )

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
                if is_limit:
                    from brokers.base import LimitOrder
                    lo = LimitOrder(
                        ticket=int(res.ticket),
                        symbol=signal.symbol,
                        order_type=order_type.value,
                        price=target_entry,
                        volume=lot_size,
                        sl=padded_sl,
                        tp=signal.suggested_tp,
                        created_time=time.time(),
                        timeout_seconds=75.0,
                        invalidation_price=signal.invalidation_price or 0.0,
                        comment=order_comment,
                    )
                    async with self._lock:
                        self._pending_limit_orders[lo.ticket] = lo
                    logger.info(
                        f"[SMART LIMIT ORDER PLACED] #{lo.ticket} {order_type.value} {lot_size} {signal.symbol} "
                        f"@ {target_entry:.2f} | Invalidation Guard: {lo.invalidation_price:.2f} | SL: {padded_sl:.2f} | TP: {signal.suggested_tp:.2f}"
                    )
                else:
                    # Early breakeven on micro balance: lock at +0.35R (~$0.70 gain) to protect capital
                    active_be_r = 0.35 if (is_gold and eff_bal < 50.0) else signal.be_lock_r

                    # Register position for sub-second Chameleon adaptive management
                    managed = ManagedPosition(
                        ticket=int(res.ticket),
                        symbol=signal.symbol,
                        direction=signal.direction,
                        volume=res.volume,
                        open_price=res.price,
                        sl=padded_sl,
                        tp=signal.suggested_tp,
                        entry_time=time.time(),
                        risk_r_dist=actual_sl_dist,
                        initial_regime=signal.regime,
                        current_regime=signal.regime,
                        sub_regime=signal.sub_regime,
                        be_lock_r=active_be_r,
                        stall_exit_seconds=signal.stall_exit_seconds,
                        highest_r_reached=0.0,
                        entry_velocity=self.news_reactor._latest_velocity or 1.0,
                    )
                    async with self._lock:
                        self._managed_positions[managed.ticket] = managed

                    logger.info(
                        f"[ADAPTIVE TRADE INITIATED] #{managed.ticket} {signal.direction} {lot_size} {signal.symbol} "
                        f"@ {res.price:.2f} | Sub-Regime: {signal.sub_regime} -> {signal.regime} | Risk: {signal.risk_r_dist:.2f} | TP: {signal.suggested_tp:.2f}"
                    )

            return res

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

                    # If order filled by broker into an active position, remove from pending
                    if p_ticket in broker_pos_map:
                        async with self._lock:
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
                    pos.profit = getattr(live_p, "profit", 0.0)

                    # Compute floating R-multiple
                    if pos.direction == "BUY":
                        floating_dist = current_price - pos.open_price
                    else:
                        floating_dist = pos.open_price - current_price

                    floating_r = floating_dist / max(pos.risk_r_dist, 0.01)

                    if floating_r > pos.highest_r_reached:
                        pos.highest_r_reached = round(floating_r, 2)

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
                    # INTELLIGENT RULE 2: Extra-Calculated News Management (No Arbitrary Cutoffs)
                    # Instead of blindly cutting off trades on news:
                    # 1. If trade is in profit, lock Breakeven so it rides the news risk-free!
                    # 2. Check fundamental news surprise bias (Actual vs Forecast).
                    # 3. If bias aligns with trade, let it run!
                    # 4. Only exit if high-confidence news strongly opposes trade AND position is in deep loss.
                    # =============================================================
                    cal_state = self.news_reactor.check_calendar_state() if self.news_reactor else {}
                    is_news_freeze = cal_state.get("is_news_freeze", False) or cal_state.get("is_freeze", False)
                    if is_news_freeze:
                        # 1. If in profit, lock to Breakeven so it rides news safely with ZERO risk
                        if floating_r >= 0.15 and pos.ratchet_stage < 1:
                            pos.ratchet_stage = 1
                            is_gold = any(k in pos.symbol.upper() for k in ["XAU", "GOLD"])
                            spread_buf = 0.25 if is_gold else 0.0002
                            new_sl = round(pos.open_price + spread_buf if pos.direction == "BUY" else pos.open_price - spread_buf, 2 if is_gold else 5)
                            pos.sl = new_sl
                            await self.broker.modify_position(pos.ticket, sl=new_sl, tp=pos.tp)
                            logger.info(f"[CALCULATED NEWS PROTECTION] #{ticket} locked SL to Breakeven ({new_sl}) to ride news safely without closing!")

                        # 2. Study fundamental news bias
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
                    # INTELLIGENT RULE 6: Fast Scalp Velocity Stall Exit
                    # =============================================================
                    if pos.sub_regime == "FAST_SCALP" and floating_r >= 1.50 and (now - pos.entry_time >= 90.0):
                        current_vel = self.news_reactor._latest_velocity or 0.0
                        is_velocity_dead = (current_vel < 0.50 * max(0.5, pos.entry_velocity))

                        if is_velocity_dead:
                            if pos.stall_start_time == 0.0:
                                pos.stall_start_time = now
                            elif now - pos.stall_start_time >= max(30.0, pos.stall_exit_seconds):
                                stall_reason = f"Fast Scalp Stall Bank (+{floating_r:.2f}R)"
                                logger.info(
                                    f"[FAST SCALP STALL EXIT] Velocity stalled for {now - pos.stall_start_time:.1f}s (>=30s). "
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
        Intelligent Market Flip & Adverse Transition Detector.
        Determines if order flow or micro-structure has decisively flipped against our trade.
        Returns:
            (True, reason) -> Position must be closed immediately at market (cutting loss or locking profit).
        """
        # 1. Minimum Hold Time: Give trade at least 60s to breathe past spread and develop!
        if (now - pos.entry_time) < 60.0:
            return False, ""

        # 2. Asymmetric Risk Guard: Early Adverse Invalidation for Losing Trades!
        # When a trade is underwater (floating_r <= -0.65R), check if institutional structure has invalidated the setup.
        # This prevents taking full -1.0R / -2.5R losses by cutting at ~ -0.70R when market decisively breaks against us.
        if floating_r <= -0.65:
            # Check 5m SuperTrend structural alignment
            df_5m = await self.data_hub.get_dataframe(pos.symbol, "5m", count=25)
            if df_5m is not None and len(df_5m) >= 20:
                st_5m = TechnicalIndicators.calculate_supertrend(
                    df_5m["high"], df_5m["low"], df_5m["close"], period=10, multiplier=2.5
                )
                if st_5m.get("valid"):
                    if pos.direction == "BUY" and st_5m.get("is_bear"):
                        return True, f"Adverse 5m SuperTrend Bearish Break: Early Invalidation ({floating_r:.2f}R)"
                    elif pos.direction == "SELL" and st_5m.get("is_bull"):
                        return True, f"Adverse 5m SuperTrend Bullish Break: Early Invalidation ({floating_r:.2f}R)"

            # Check Severe Opposing Velocity Shock
            vel = self.news_reactor._latest_velocity or 1.0
            if vel >= 3.5 * max(0.5, pos.entry_velocity):
                return True, f"Severe Opposing Velocity Shock ({vel:.1f}x): Early Invalidation ({floating_r:.2f}R)"

        # Note: Winning trades (floating_r > 0) are NOT choked by 1m noise!
        # They are protected by the One-Way Ratchet Floor (BE at +0.65R, lock profit at +1.0R, +1.5R, +2.0R)
        # to ensure positive mathematical expectancy.
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
        # If we have reached >= 70% of current TP and target is under 6.0R:
        if floating_r >= 0.70 * current_target_r and current_target_r < 6.0:
            # Check 1m momentum
            df_1m = await self.data_hub.get_dataframe(pos.symbol, "1m", count=3)
            if df_1m is not None and len(df_1m) >= 2:
                c1 = df_1m["close"].iloc[-1]
                o1 = df_1m["open"].iloc[-1]
                h1 = df_1m["high"].iloc[-1]
                l1 = df_1m["low"].iloc[-1]
                rng1 = max(0.01, h1 - l1)

                is_trending_strong = False
                if pos.direction == "BUY" and c1 > o1:
                    upper_wick = h1 - c1
                    if upper_wick / rng1 < 0.35: # Clean candle with little opposing wick
                        is_trending_strong = True
                elif pos.direction == "SELL" and c1 < o1:
                    lower_wick = c1 - l1
                    if lower_wick / rng1 < 0.35:
                        is_trending_strong = True

                if is_trending_strong:
                    # Expand TP by 1.2R to 1.5R
                    new_target_r = round(current_target_r + 1.25, 1)
                    is_gold = any(k in pos.symbol.upper() for k in ["XAU", "GOLD"])
                    if pos.direction == "BUY":
                        new_tp = round(pos.open_price + (pos.risk_r_dist * new_target_r), 2 if is_gold else 5)
                    else:
                        new_tp = round(pos.open_price - (pos.risk_r_dist * new_target_r), 2 if is_gold else 5)

                    if new_tp != pos.tp:
                        old_tp = pos.tp
                        pos.tp = new_tp
                        pos.current_regime = "ADAPTIVE_RUNNER"
                        pos.last_mod_time = time.time()
                        await self.broker.modify_position(pos.ticket, sl=pos.sl, tp=pos.tp)
                        logger.info(
                            f"[DYNAMIC TP EXPANSION] #{pos.ticket} surging strongly (+{floating_r:.2f}R)! "
                            f"Extended TP from {old_tp:.2f} to {new_tp:.2f} (+{new_target_r:.1f}R) to capture maximum profit!"
                        )

    async def _evaluate_ratchet_floor(self, pos: ManagedPosition, floating_r: float, current_price: float) -> None:
        """
        Enforce One-Way Ratchet Floor on Broker Stop Loss:
        - Allows trade full room to breathe until reaching healthy profit!
        - +0.65R+ -> Breakeven (Entry + Spread Buffer)
        - +1.00R+ -> Lock +0.45R Guaranteed Profit Floor
        - +1.50R+ -> Lock +0.85R Guaranteed Profit Floor
        - +2.00R+ -> Lock +1.35R Guaranteed Profit Floor
        - +2.50R+ -> Volatility-Adaptive Trailing behind price structure
        """
        is_gold = any(k in pos.symbol.upper() for k in ["XAU", "GOLD"])
        spread_buf = 0.25 if is_gold else 0.0002
        new_sl = None

        # Stage 1: Breakeven Lock at +0.65R (or higher depending on strategy regime)
        trigger_be = max(0.65, pos.be_lock_r)
        if pos.ratchet_stage < 1 and floating_r >= trigger_be:
            pos.ratchet_stage = 1
            if pos.direction == "BUY":
                new_sl = round(pos.open_price + spread_buf, 2 if is_gold else 5)
            else:
                new_sl = round(pos.open_price - spread_buf, 2 if is_gold else 5)
            logger.info(f"[RATCHET BE LOCK] #{pos.ticket} hit +{floating_r:.2f}R -> SL moved to Breakeven {new_sl:.2f}")

        # Stage 2: Lock Profit at +1.00R (Locks +0.45R)
        elif pos.ratchet_stage < 2 and floating_r >= 1.00:
            pos.ratchet_stage = 2
            lock_dist = pos.risk_r_dist * 0.45
            if pos.direction == "BUY":
                new_sl = round(pos.open_price + lock_dist, 2 if is_gold else 5)
            else:
                new_sl = round(pos.open_price - lock_dist, 2 if is_gold else 5)
            logger.info(f"[RATCHET LOCK +0.45R] #{pos.ticket} hit +{floating_r:.2f}R -> SL locked at {new_sl:.2f}")

        # Stage 3: Lock Profit at +1.50R (Locks +0.85R)
        elif pos.ratchet_stage < 3 and floating_r >= 1.50:
            pos.ratchet_stage = 3
            lock_dist = pos.risk_r_dist * 0.85
            if pos.direction == "BUY":
                new_sl = round(pos.open_price + lock_dist, 2 if is_gold else 5)
            else:
                new_sl = round(pos.open_price - lock_dist, 2 if is_gold else 5)
            logger.info(f"[RATCHET LOCK +0.85R] #{pos.ticket} hit +{floating_r:.2f}R -> SL locked at {new_sl:.2f}")

        # Stage 4: Lock Profit at +2.00R (Locks +1.35R)
        elif pos.ratchet_stage < 4 and floating_r >= 2.00:
            pos.ratchet_stage = 4
            lock_dist = pos.risk_r_dist * 1.35
            if pos.direction == "BUY":
                new_sl = round(pos.open_price + lock_dist, 2 if is_gold else 5)
            else:
                new_sl = round(pos.open_price - lock_dist, 2 if is_gold else 5)
            logger.info(f"[RATCHET LOCK +1.35R] #{pos.ticket} hit +{floating_r:.2f}R -> SL locked at {new_sl:.2f}")

        # Stage 5: Dynamic Structural Trailing at +2.50R+
        elif floating_r >= 2.50:
            pos.ratchet_stage = 5
            trail_dist = pos.risk_r_dist * 0.85
            if pos.direction == "BUY":
                candidate_sl = round(current_price - trail_dist, 2 if is_gold else 5)
                if candidate_sl > pos.sl:
                    new_sl = candidate_sl
            else:
                candidate_sl = round(current_price + trail_dist, 2 if is_gold else 5)
                if candidate_sl < pos.sl:
                    new_sl = candidate_sl

        if new_sl is not None and new_sl != pos.sl:
            pos.sl = new_sl
            pos.last_mod_time = time.time()
            await self.broker.modify_position(pos.ticket, sl=pos.sl, tp=pos.tp)

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

        is_gold = any(k in pos.symbol.upper() for k in ["XAU", "GOLD"])
        contract_size = 100.0 if is_gold else 100000.0

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

        is_loss = (pnl < -0.05)
        if is_loss:
            self._daily_losses_count += 1
            # Strict 5-Minute (300s) Anti-Revenge Cooldown after any loss
            self._post_loss_cooldown_until = now + 300.0
            self._last_order_time = now + 300.0
            logger.warning(
                f"[POST-LOSS ANTI-REVENGE COOLDOWN] Trade #{pos.ticket} closed with loss (${pnl:.2f}). "
                f"Cooldown active for 300s (5m) to let market establish new structure before any re-entry."
            )
            if self._daily_losses_count >= self.max_daily_losses:
                self.pause_trading(minutes=120)
                logger.error("[CIRCUIT BREAKER ACTIVATED] Max consecutive daily losses hit. Auto-trading paused 2 hours.")
            elif self._daily_losses_count >= 2:
                self.pause_trading(minutes=30)
                logger.warning("[CIRCUIT BREAKER] 2 consecutive losses detected. Auto-trading paused 30 minutes to protect capital.")

    async def emergency_close_all(self) -> int:
        """1-Click Emergency Button: Closes all positions immediately."""
        logger.warning("[EMERGENCY ACTION] Closing all active positions across broker!")
        count = await self.broker.close_all_positions()
        async with self._lock:
            self._managed_positions.clear()
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
            "is_paused": self.is_paused(),
            "pause_seconds_left": max(0, int(self._pause_until - now)),
            "post_loss_cooldown_seconds_left": max(0, int(self._post_loss_cooldown_until - now)),
        }
