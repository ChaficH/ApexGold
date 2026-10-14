"""
ApexGold Paper Simulation Broker Adapter.

Provides a full local simulated broker environment fed by live market feeds (yfinance)
and a sub-millisecond realistic tick generator.
Allows instant testing of the entire trading platform, UI, auto-execution,
and risk engine on macOS and Windows without requiring MetaTrader 5.
"""

import asyncio
import logging
import math
import random
import time
from typing import Dict, List, Optional, Union

import numpy as np

from brokers.base import (
    AbstractBrokerAdapter,
    AccountInfo,
    Bar,
    ClosedDeal,
    OrderResult,
    OrderType,
    Position,
    PositionType,
    SymbolInfo,
    Tick,
)

logger = logging.getLogger("ApexGold.PaperSim")


class PaperSimAdapter(AbstractBrokerAdapter):
    """
    Production-grade Paper Trading Simulator.
    Simulates real broker behavior including spread, slippage, margin requirement,
    dynamic P&L calculation, and automated SL/TP triggering.
    """

    # Symbol symbol-to-yfinance mapping
    YF_TICKERS = {
        "XAUUSD": "GC=F",       # COMEX Gold Futures or XAUUSD=X
        "XAUUSDc": "GC=F",
        "XAUUSDm": "GC=F",
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USDJPY": "USDJPY=X",
    }

    def __init__(
        self,
        initial_balance: float = 10.0,
        currency: str = "USD",
        leverage: int = 500,
        account_type: str = "cent",
        spread_points: float = 25.0, # 0.25 on Gold
    ):
        self._initial_balance = float(initial_balance)
        self.balance = float(initial_balance)
        self.currency = currency
        self.leverage = leverage
        self.account_type = account_type
        self.spread_points = spread_points
        self._connected = False

        self._ticket_counter = 100000
        self._positions: Dict[int, Position] = {}
        self._pending_orders: Dict[int, Any] = {}
        self._closed_positions: List[Dict] = []
        self._closed_deals: Dict[int, ClosedDeal] = {}
        self._last_ticks: Dict[str, Tick] = {}
        self._bar_cache: Dict[str, Dict[str, List[Bar]]] = {} # symbol -> tf -> List[Bar]
        
        # Base realistic default prices if offline or weekend
        self._base_prices = {
            "XAUUSD": 4163.50,
            "XAUUSDc": 4163.50,
            "XAUUSDm": 4163.50,
            "EURUSD": 1.1195,
            "GBPUSD": 1.3265,
            "USDJPY": 158.45,
        }

        self._tick_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self.is_paper: bool = True

    async def connect(self) -> bool:
        """Initialize the simulator and start the live price generator."""
        self._connected = True
        logger.info(
            f"PaperSim connected! Balance: {self.balance:.2f} {self.currency} "
            f"({self.account_type.upper()} account, 1:{self.leverage})"
        )
        # Seed initial ticks
        for symbol, base_price in self._base_prices.items():
            spread = 0.25 if "XAU" in symbol else 0.00015
            self._last_ticks[symbol] = Tick(
                symbol=symbol,
                bid=round(base_price, 2 if "XAU" in symbol else 5),
                ask=round(base_price + spread, 2 if "XAU" in symbol else 5),
                time=time.time(),
                spread=round(spread, 5),
            )

        # Seed latest live real market prices asynchronously
        async def _fetch_initial_ticks():
            try:
                import yfinance as yf
                tickers_to_fetch = [("XAUUSD", "GC=F"), ("EURUSD", "EURUSD=X"), ("GBPUSD", "GBPUSD=X"), ("USDJPY", "USDJPY=X")]
                loop = asyncio.get_running_loop()
                for sym, yf_t in tickers_to_fetch:
                    t_data = await loop.run_in_executor(None, lambda y=yf_t: yf.Ticker(y).history(period="1d", interval="1m"))
                    if t_data is not None and not t_data.empty:
                        last_p = float(t_data["Close"].iloc[-1])
                        spread = 0.25 if "XAU" in sym else 0.00015
                        self._base_prices[sym] = last_p
                        if sym == "XAUUSD":
                            self._base_prices["XAUUSDc"] = last_p
                            self._base_prices["XAUUSDm"] = last_p
                            self._last_ticks["XAUUSDc"] = Tick(symbol="XAUUSDc", bid=round(last_p, 2), ask=round(last_p + spread, 2), time=time.time(), spread=spread)
                            self._last_ticks["XAUUSDm"] = Tick(symbol="XAUUSDm", bid=round(last_p, 2), ask=round(last_p + spread, 2), time=time.time(), spread=spread)
                        self._last_ticks[sym] = Tick(
                            symbol=sym,
                            bid=round(last_p, 2 if "XAU" in sym else 5),
                            ask=round(last_p + spread, 2 if "XAU" in sym else 5),
                            time=time.time(),
                            spread=spread,
                        )
                        logger.info(f"[LIVE REAL FEED] Seeded {sym} from {yf_t} @ {last_p:.2f}")
            except Exception as e:
                logger.debug(f"Initial tick fetch error: {e}")

        asyncio.create_task(_fetch_initial_ticks())

        # Start continuous tick & SL/TP monitoring loop
        if self._tick_task is None or self._tick_task.done():
            self._tick_task = asyncio.create_task(self._simulation_loop())

        return True

    async def disconnect(self) -> None:
        """Stop simulator background tasks."""
        self._connected = False
        if self._tick_task and not self._tick_task.done():
            self._tick_task.cancel()
            try:
                await self._tick_task
            except asyncio.CancelledError:
                pass
        logger.info("PaperSim disconnected.")

    def is_connected(self) -> bool:
        return self._connected

    def _get_contract_spec(self, symbol: str) -> Dict:
        """Return contract specification based on symbol and account type."""
        is_gold = any(k in symbol.upper() for k in ["XAU", "GOLD"])
        if is_gold:
            digits = 2
            point = 0.01
            # Standard Gold is 100 oz per lot. On Cent account, 1 lot = 1 oz
            contract_size = 1.0 if self.account_type == "cent" else 100.0
            tick_size = 0.01
            tick_value = 1.0 if self.account_type != "cent" else 0.01
            vol_min = 0.01
            vol_step = 0.01
        else:
            digits = 5
            point = 0.00001
            contract_size = 1000.0 if self.account_type == "cent" else 100000.0
            tick_size = 0.00001
            tick_value = 1.0 if self.account_type != "cent" else 0.01
            vol_min = 0.01
            vol_step = 0.01

        return {
            "digits": digits,
            "point": point,
            "contract_size": contract_size,
            "tick_size": tick_size,
            "tick_value": tick_value,
            "volume_min": vol_min,
            "volume_max": 100.0,
            "volume_step": vol_step,
        }

    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """Return current symbol quotes and contract dimensions."""
        spec = self._get_contract_spec(symbol)
        tick = await self.get_tick(symbol)
        return SymbolInfo(
            symbol=symbol,
            bid=tick.bid,
            ask=tick.ask,
            point=spec["point"],
            digits=spec["digits"],
            spread=round(tick.ask - tick.bid, spec["digits"]),
            contract_size=spec["contract_size"],
            tick_size=spec["tick_size"],
            tick_value=spec["tick_value"],
            volume_min=spec["volume_min"],
            volume_max=spec["volume_max"],
            volume_step=spec["volume_step"],
        )

    async def get_tick(self, symbol: str) -> Tick:
        """Get the latest real-time tick."""
        if symbol not in self._last_ticks:
            base = self._base_prices.get(symbol, 2385.50)
            spread = 0.25 if "XAU" in symbol else 0.00015
            self._last_ticks[symbol] = Tick(
                symbol=symbol,
                bid=base,
                ask=base + spread,
                time=time.time(),
                spread=spread,
            )
        return self._last_ticks[symbol]

    def _calculate_account_info_unlocked(self) -> AccountInfo:
        """Compute live equity, margin, and free margin without acquiring lock."""
        if getattr(self, "_override_free_margin", None) is not None:
            return AccountInfo(
                balance=round(self.balance, 2),
                equity=round(getattr(self, "equity", self.balance), 2),
                margin=round(getattr(self, "_override_margin", 0.0), 2),
                free_margin=round(self._override_free_margin, 2),
                currency=self.currency,
                leverage=self.leverage,
                account_type=self.account_type,
            )

        floating_pnl = sum(p.profit for p in self._positions.values())
        equity = round(self.balance + floating_pnl, 2)
        used_margin = 0.0

        for p in self._positions.values():
            spec = self._get_contract_spec(p.symbol)
            # Margin = (Price * Volume * ContractSize) / Leverage
            margin_req = (p.open_price * p.volume * spec["contract_size"]) / self.leverage
            used_margin += margin_req

        free_margin = max(0.0, equity - used_margin)

        return AccountInfo(
            balance=round(self.balance, 2),
            equity=equity,
            margin=round(used_margin, 2),
            free_margin=round(free_margin, 2),
            currency=self.currency,
            leverage=self.leverage,
            account_type=self.account_type,
        )

    async def get_account_info(self) -> AccountInfo:
        """Compute live equity, margin, and free margin."""
        async with self._lock:
            return self._calculate_account_info_unlocked()

    async def get_rates(self, symbol: str, timeframe: str, count: int = 100) -> List[Bar]:
        """
        Return historical OHLCV bars.
        Attempts yfinance first; falls back to high-fidelity synthetic price generation
        to guarantee zero downtime and immediate offline operation.
        """
        tf_clean = timeframe.upper().replace("M", "m").replace("H", "h")
        cache_key = f"{symbol}_{tf_clean}"

        # 1. Try pulling from live yfinance
        bars = await self._fetch_yfinance_or_cache(symbol, tf_clean, count)
        if bars and len(bars) > 0:
            return bars[-count:] if len(bars) >= count else bars

        # 2. Synthetic realistic candle generator fallback (only if offline)
        return self._generate_synthetic_bars(symbol, tf_clean, count)

    async def _fetch_yfinance_or_cache(self, symbol: str, timeframe: str, count: int) -> Optional[List[Bar]]:
        """Fetch live candle history from yfinance asynchronously."""
        try:
            import yfinance as yf
            yf_ticker = self.YF_TICKERS.get(symbol, "GC=F")
            
            # Map timeframe
            interval_map = {
                "1M": "1m", "1m": "1m",
                "5M": "5m", "5m": "5m",
                "15M": "15m", "15m": "15m",
                "1H": "60m", "1h": "60m", "H1": "60m",
            }
            yf_interval = interval_map.get(timeframe, "5m")

            period_map = {
                "1m": "7d",      # 1m is 7 days max on Yahoo Finance
                "5m": "60d",     # 5m is 60 days
                "15m": "60d",    # 15m is 60 days
                "60m": "730d",   # 1H is 730 days
                "1h": "730d",
            }
            yf_period = period_map.get(yf_interval, "60d")

            # Run in thread pool to avoid blocking async event loop
            loop = asyncio.get_running_loop()
            df = await loop.run_in_executor(
                None,
                lambda: yf.Ticker(yf_ticker).history(period=yf_period, interval=yf_interval)
            )

            if df is not None and not df.empty:
                bars: List[Bar] = []
                for idx, row in df.iterrows():
                    ts = int(idx.timestamp())
                    bars.append(Bar(
                        time=ts,
                        open=round(float(row["Open"]), 2 if "XAU" in symbol else 5),
                        high=round(float(row["High"]), 2 if "XAU" in symbol else 5),
                        low=round(float(row["Low"]), 2 if "XAU" in symbol else 5),
                        close=round(float(row["Close"]), 2 if "XAU" in symbol else 5),
                        volume=round(float(row.get("Volume", 100)), 2),
                    ))
                if bars:
                    # Calibrate / shift the bars to match the live MT5 broker price exactly if active
                    existing_tick = self._last_ticks.get(symbol) or self._last_ticks.get("XAUUSD")
                    if existing_tick and existing_tick.bid > 0:
                        diff = existing_tick.bid - bars[-1].close
                        for b in bars:
                            b.open = round(b.open + diff, 2 if "XAU" in symbol else 5)
                            b.high = round(b.high + diff, 2 if "XAU" in symbol else 5)
                            b.low = round(b.low + diff, 2 if "XAU" in symbol else 5)
                            b.close = round(b.close + diff, 2 if "XAU" in symbol else 5)
                    else:
                        last_c = bars[-1].close
                        spread = 0.25 if "XAU" in symbol else 0.00015
                        self._last_ticks[symbol] = Tick(
                            symbol=symbol,
                            bid=round(last_c, 2 if "XAU" in symbol else 5),
                            ask=round(last_c + spread, 2 if "XAU" in symbol else 5),
                            time=time.time(),
                            spread=spread,
                        )
                    return bars
        except Exception as e:
            logger.debug(f"yfinance fetch failed for {symbol} ({e}), using generator.")
        return None

    def _generate_synthetic_bars(self, symbol: str, timeframe: str, count: int) -> List[Bar]:
        """Generate statistically realistic random walk candles around base price."""
        tf_seconds = {
            "1m": 60, "5m": 300, "15m": 900, "1h": 3600, "60m": 3600,
        }.get(timeframe.lower(), 300)

        now = int(time.time())
        existing_tick = self._last_ticks.get(symbol) or self._last_ticks.get("XAUUSD")
        if existing_tick and existing_tick.bid > 0:
            current_price = existing_tick.bid
        else:
            current_price = self._base_prices.get(symbol, 4141.50)
        volatility = 1.2 if "XAU" in symbol else 0.0008

        bars: List[Bar] = []
        price = current_price - (count * 0.1 * volatility)

        for i in range(count):
            bar_time = now - ((count - i) * tf_seconds)
            change = np.random.normal(0.02, volatility)
            open_p = price
            close_p = price + change
            high_p = max(open_p, close_p) + abs(np.random.normal(0, volatility * 0.5))
            low_p = min(open_p, close_p) - abs(np.random.normal(0, volatility * 0.5))
            volume = float(random.randint(50, 500))

            bars.append(Bar(
                time=bar_time,
                open=round(open_p, 2 if "XAU" in symbol else 5),
                high=round(high_p, 2 if "XAU" in symbol else 5),
                low=round(low_p, 2 if "XAU" in symbol else 5),
                close=round(close_p, 2 if "XAU" in symbol else 5),
                volume=volume,
                is_synthetic=True,
            ))
            price = close_p

        return bars

    async def open_order(
        self,
        symbol: str,
        order_type: OrderType,
        volume: float,
        price: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = "ApexGold"
    ) -> OrderResult:
        """Simulate execution with slippage and margin validation."""
        async with self._lock:
            tick = await self.get_tick(symbol)
            spec = self._get_contract_spec(symbol)

            # Limit order handling
            if order_type in [OrderType.BUY_LIMIT, OrderType.SELL_LIMIT]:
                limit_target = price if price is not None and price > 0 else (tick.ask - 0.50 if order_type == OrderType.BUY_LIMIT else tick.bid + 0.50)
                self._ticket_counter += 1
                ticket = self._ticket_counter

                from brokers.base import LimitOrder
                lo = LimitOrder(
                    ticket=ticket,
                    symbol=symbol,
                    order_type=order_type.value,
                    price=round(limit_target, spec["digits"]),
                    volume=round(volume, 2),
                    sl=round(sl, spec["digits"]) if sl else 0.0,
                    tp=round(tp, spec["digits"]) if tp else 0.0,
                    created_time=time.time(),
                    timeout_seconds=180.0,
                    invalidation_price=0.0,
                    comment=comment,
                )
                self._pending_orders[ticket] = lo
                logger.info(
                    f"[SIM LIMIT PLACED] #{ticket} {order_type.value} {volume} {symbol} @ {lo.price:.2f} "
                    f"| SL: {lo.sl:.2f} | TP: {lo.tp:.2f} | Comment: {comment}"
                )

                return OrderResult(
                    success=True,
                    ticket=ticket,
                    price=lo.price,
                    volume=volume,
                    error=None,
                )

            # Fill price: Buy fills on Ask, Sell fills on Bid
            fill_price = tick.ask if order_type == OrderType.BUY else tick.bid

            # Add minor slippage (0.01 to 0.05 on Gold)
            slippage = random.uniform(0.01, 0.03) if "XAU" in symbol else 0.00003
            if order_type == OrderType.BUY:
                fill_price = round(fill_price + slippage, spec["digits"])
            else:
                fill_price = round(fill_price - slippage, spec["digits"])

            # Check margin requirement
            margin_req = (fill_price * volume * spec["contract_size"]) / self.leverage
            account = self._calculate_account_info_unlocked()
            if margin_req > account.free_margin:
                msg = f"Insufficient margin: required {margin_req:.2f}, free {account.free_margin:.2f}"
                logger.warning(msg)
                return OrderResult(success=False, error=msg)

            self._ticket_counter += 1
            ticket = self._ticket_counter

            pos = Position(
                ticket=ticket,
                symbol=symbol,
                type=PositionType.BUY if order_type == OrderType.BUY else PositionType.SELL,
                volume=round(volume, 2),
                open_price=fill_price,
                current_price=fill_price,
                sl=round(sl, spec["digits"]) if sl else 0.0,
                tp=round(tp, spec["digits"]) if tp else 0.0,
                profit=0.0,
                open_time=time.time(),
                magic=15555,
                comment=comment,
            )

            self._positions[ticket] = pos
            logger.info(
                f"[SIM EXECUTION] #{ticket} {order_type.value} {volume} {symbol} @ {fill_price:.2f} "
                f"| SL: {pos.sl:.2f} | TP: {pos.tp:.2f} | Comment: {comment}"
            )

            return OrderResult(
                success=True,
                ticket=ticket,
                price=fill_price,
                volume=volume,
                error=None,
            )

    async def modify_position(
        self,
        ticket: Union[int, str],
        sl: float,
        tp: float
    ) -> bool:
        """Modify position SL and TP."""
        async with self._lock:
            ticket_int = int(ticket)
            if ticket_int in self._positions:
                pos = self._positions[ticket_int]
                spec = self._get_contract_spec(pos.symbol)
                pos.sl = round(sl, spec["digits"])
                pos.tp = round(tp, spec["digits"])
                logger.info(f"[SIM MODIFY] #{ticket} New SL: {pos.sl:.2f}, New TP: {pos.tp:.2f}")
                return True
            return True

    async def close_position(
        self,
        ticket: Union[int, str],
        volume: Optional[float] = None
    ) -> bool:
        """Close position and realize P&L."""
        async with self._lock:
            ticket_int = int(ticket)
            if ticket_int not in self._positions:
                return True

            pos = self._positions.pop(ticket_int)
            self.balance = round(self.balance + pos.profit, 2)
            
            record = {
                "ticket": ticket_int,
                "symbol": pos.symbol,
                "type": pos.type.value,
                "volume": pos.volume,
                "open_price": pos.open_price,
                "close_price": pos.current_price,
                "profit": pos.profit,
                "close_time": time.time(),
                "comment": pos.comment,
            }
            self._closed_positions.append(record)
            self._closed_deals[ticket_int] = ClosedDeal(
                ticket=ticket_int,
                symbol=pos.symbol,
                direction=pos.type.value if hasattr(pos.type, "value") else str(pos.type),
                volume=pos.volume,
                open_price=pos.open_price,
                close_price=pos.current_price,
                open_time=pos.open_time,
                close_time=time.time(),
                profit=pos.profit,
                commission=0.0,
                swap=0.0,
                comment=pos.comment,
            )
            logger.info(
                f"[SIM CLOSED] #{ticket_int} PnL: {pos.profit:+.2f} {self.currency} | "
                f"New Balance: {self.balance:.2f} {self.currency}"
            )
            return True

    async def close_all_positions(self, symbol: Optional[str] = None) -> int:
        """Close all positions matching symbol (or all if None)."""
        tickets_to_close = []
        async with self._lock:
            for t, p in self._positions.items():
                if symbol is None or p.symbol == symbol:
                    tickets_to_close.append(t)

        count = 0
        for t in tickets_to_close:
            if await self.close_position(t):
                count += 1
        return count

    async def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Return snapshot of open positions."""
        async with self._lock:
            if symbol is None:
                return list(self._positions.values())
            return [p for p in self._positions.values() if p.symbol == symbol]

    async def cancel_order(self, ticket: Union[int, str]) -> bool:
        """Cancel a pending simulated limit order."""
        async with self._lock:
            int_ticket = int(ticket) if str(ticket).isdigit() else ticket
            canceled = False
            if int_ticket in self._pending_orders:
                del self._pending_orders[int_ticket]
                logger.info(f"[SIM LIMIT DELETED] #{ticket} cancelled.")
                canceled = True

            return canceled

    async def get_pending_orders(self, symbol: Optional[str] = None) -> List[Any]:
        """Return snapshot of active pending limit orders."""
        async with self._lock:
            if symbol is None:
                return list(self._pending_orders.values())
            return [o for o in self._pending_orders.values() if o.symbol == symbol]

    async def get_closed_deal(self, ticket: Union[int, str]) -> Optional[ClosedDeal]:
        """Retrieve real executed closed deal history for ticket."""
        async with self._lock:
            return self._closed_deals.get(int(ticket))

    async def _simulation_loop(self) -> None:
        """
        Sub-second tick engine.
        Updates prices, evaluates floating P&L, checks for SL/TP hits, and auto-closes.
        """
        while self._connected:
            try:
                await asyncio.sleep(0.1) # 100ms cycle

                # Jitter prices slightly to simulate active order flow
                for symbol in list(self._last_ticks.keys()):
                    tick = self._last_ticks[symbol]
                    spec = self._get_contract_spec(symbol)
                    
                    # Random small price wiggle
                    vol = 0.05 if "XAU" in symbol else 0.00005
                    delta = random.choice([-1, 1]) * random.uniform(0.00, vol)
                    new_bid = round(tick.bid + delta, spec["digits"])
                    spread = 0.25 if "XAU" in symbol else 0.00015
                    new_ask = round(new_bid + spread, spec["digits"])

                    self._last_ticks[symbol] = Tick(
                        symbol=symbol,
                        bid=new_bid,
                        ask=new_ask,
                        time=time.time(),
                        spread=spread,
                    )

                # Evaluate pending limit orders for fills
                to_fill_limits: List[Any] = []
                async with self._lock:
                    for ticket, lo in list(self._pending_orders.items()):
                        tick = self._last_ticks.get(lo.symbol)
                        if not tick:
                            continue
                        if lo.order_type == "BUY_LIMIT" and tick.ask <= lo.price:
                            to_fill_limits.append(lo)
                        elif lo.order_type == "SELL_LIMIT" and tick.bid >= lo.price:
                            to_fill_limits.append(lo)

                for lo in to_fill_limits:
                    async with self._lock:
                        if lo.ticket in self._pending_orders:
                            del self._pending_orders[lo.ticket]
                            spec = self._get_contract_spec(lo.symbol)
                            pos = Position(
                                ticket=lo.ticket,
                                symbol=lo.symbol,
                                type=PositionType.BUY if lo.order_type == "BUY_LIMIT" else PositionType.SELL,
                                volume=lo.volume,
                                open_price=lo.price,
                                current_price=lo.price,
                                sl=lo.sl,
                                tp=lo.tp,
                                profit=0.0,
                                open_time=time.time(),
                                magic=15555,
                                comment=lo.comment or "ApexGold_LimitFill",
                            )
                            self._positions[lo.ticket] = pos
                            logger.info(f"[SIM LIMIT FILLED] #{lo.ticket} {lo.order_type} filled @ {lo.price:.2f}")

                # Update positions and test SL/TP triggers
                to_close_sl_tp: List[tuple] = [] # (ticket, reason, close_price)

                async with self._lock:
                    for ticket, pos in self._positions.items():
                        tick = self._last_ticks.get(pos.symbol)
                        if not tick:
                            continue

                        spec = self._get_contract_spec(pos.symbol)
                        
                        if pos.type == PositionType.BUY:
                            pos.current_price = tick.bid
                            # Profit = (Bid - OpenPrice) * Volume * ContractSize
                            pnl = (tick.bid - pos.open_price) * pos.volume * spec["contract_size"]
                            pos.profit = round(pnl, 2)

                            # Check TP
                            if pos.tp > 0 and tick.bid >= pos.tp:
                                to_close_sl_tp.append((ticket, "TP_HIT", pos.tp))
                            # Check SL
                            elif pos.sl > 0 and tick.bid <= pos.sl:
                                to_close_sl_tp.append((ticket, "SL_HIT", pos.sl))

                        elif pos.type == PositionType.SELL:
                            pos.current_price = tick.ask
                            # Profit = (OpenPrice - Ask) * Volume * ContractSize
                            pnl = (pos.open_price - tick.ask) * pos.volume * spec["contract_size"]
                            pos.profit = round(pnl, 2)

                            # Check TP
                            if pos.tp > 0 and tick.ask <= pos.tp:
                                to_close_sl_tp.append((ticket, "TP_HIT", pos.tp))
                            # Check SL
                            elif pos.sl > 0 and tick.ask >= pos.sl:
                                to_close_sl_tp.append((ticket, "SL_HIT", pos.sl))

                # Process triggered SL/TP outside main position lock
                for ticket, reason, price in to_close_sl_tp:
                    logger.info(f"[SIM SL/TP TRIGGER] Position #{ticket} hit {reason} @ {price:.2f}")
                    await self.close_position(ticket)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in PaperSim loop: {e}", exc_info=True)
                await asyncio.sleep(1.0)
