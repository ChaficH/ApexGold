"""
ApexGold Windows MetaTrader 5 Broker Adapter.

Uses the official MetaTrader5 Python package for direct IPC memory connection
to MetaTrader 5 on Windows OS.
Includes platform detection and graceful fallback messaging when run on macOS/Linux.
"""

import asyncio
import logging
import platform
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

from brokers.base import (
    AbstractBrokerAdapter,
    AccountInfo,
    Bar,
    ClosedDeal,
    LimitOrder,
    OrderResult,
    OrderType,
    Position,
    PositionType,
    SymbolInfo,
    Tick,
)

logger = logging.getLogger("ApexGold.MT5Windows")

# Safe import for MetaTrader5 package
mt5 = None
IS_WINDOWS = platform.system().lower() == "windows"
if IS_WINDOWS:
    try:
        import MetaTrader5 as mt5
    except ImportError:
        logger.warning("MetaTrader5 package not installed. Run: pip install MetaTrader5")


class WindowsMT5Adapter(AbstractBrokerAdapter):
    """
    Direct Windows MT5 Terminal Adapter.
    Communicates directly with local MetaTrader 5 via official Windows DLLs/IPC.
    """

    # Timeframe mapping to MT5 constants
    TIMEFRAME_MAP = {}
    if mt5 is not None:
        TIMEFRAME_MAP = {
            "1M": mt5.TIMEFRAME_M1,
            "1m": mt5.TIMEFRAME_M1,
            "M1": mt5.TIMEFRAME_M1,
            "5M": mt5.TIMEFRAME_M5,
            "5m": mt5.TIMEFRAME_M5,
            "M5": mt5.TIMEFRAME_M5,
            "15M": mt5.TIMEFRAME_M15,
            "15m": mt5.TIMEFRAME_M15,
            "M15": mt5.TIMEFRAME_M15,
            "1H": mt5.TIMEFRAME_H1,
            "1h": mt5.TIMEFRAME_H1,
            "H1": mt5.TIMEFRAME_H1,
        }

    def __init__(
        self,
        terminal_path: Optional[str] = None,
        login: int = 0,
        password: str = "",
        server: str = "",
        timeout_ms: int = 60000,
        account_type: str = "cent",
    ):
        self.terminal_path = terminal_path
        self.login = login
        self.password = password
        self.server = server
        self.timeout_ms = timeout_ms
        self.account_type = account_type
        self._connected = False

    async def connect(self) -> bool:
        """Initialize MetaTrader 5 terminal connection."""
        if not IS_WINDOWS:
            msg = "WindowsMT5Adapter is only supported on Windows. On macOS, use MacSocketMT5Adapter with ApexBridge.mq5."
            logger.error(msg)
            raise RuntimeError(msg)

        if mt5 is None:
            msg = "MetaTrader5 package is not installed. Run: pip install MetaTrader5"
            logger.error(msg)
            raise ImportError(msg)

        init_kwargs = {"timeout": self.timeout_ms}
        if self.terminal_path:
            init_kwargs["path"] = self.terminal_path

        # Initialize MT5 in an executor to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        init_ok = await loop.run_in_executor(None, lambda: mt5.initialize(**init_kwargs))
        if not init_ok:
            error_code, error_desc = mt5.last_error()
            logger.error(f"MT5 initialize failed: [{error_code}] {error_desc}")
            return False

        # Login if credentials provided
        if self.login > 0:
            login_ok = await loop.run_in_executor(
                None,
                lambda: mt5.login(
                    login=self.login,
                    password=self.password,
                    server=self.server,
                    timeout=self.timeout_ms,
                )
            )
            if not login_ok:
                error_code, error_desc = mt5.last_error()
                logger.error(f"MT5 login failed: [{error_code}] {error_desc}")
                mt5.shutdown()
                return False

        self._connected = True
        acc = mt5.account_info()
        acc_dict = acc._asdict() if acc else {}
        logger.info(
            f"Connected to Windows MT5 successfully! Account: {acc_dict.get('login')} | "
            f"Server: {acc_dict.get('server')} | Balance: {acc_dict.get('balance')} {acc_dict.get('currency')}"
        )
        return True

    async def disconnect(self) -> None:
        """Gracefully shut down MT5 connection."""
        if mt5 and self._connected:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, mt5.shutdown)
            self._connected = False
            logger.info("Windows MT5 disconnected.")

    def is_connected(self) -> bool:
        """Verify MT5 terminal and account connectivity."""
        if not self._connected or mt5 is None:
            return False
        terminal_info = mt5.terminal_info()
        return terminal_info is not None and terminal_info.connected

    async def get_account_info(self) -> AccountInfo:
        """Retrieve live MT5 account equity and margin."""
        if not self.is_connected():
            raise ConnectionError("Not connected to Windows MT5")

        loop = asyncio.get_running_loop()
        acc = await loop.run_in_executor(None, mt5.account_info)
        if acc is None:
            raise RuntimeError(f"Failed to fetch account info: {mt5.last_error()}")

        return AccountInfo(
            balance=float(acc.balance),
            equity=float(acc.equity),
            margin=float(acc.margin),
            free_margin=float(acc.margin_free),
            currency=str(acc.currency),
            leverage=int(acc.leverage),
            account_type=self.account_type,
        )

    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """Retrieve symbol trading specifications from MT5."""
        if not self.is_connected():
            raise ConnectionError("Not connected to Windows MT5")

        loop = asyncio.get_running_loop()
        info = await loop.run_in_executor(None, lambda: mt5.symbol_info(symbol))
        if info is None:
            # Attempt to select symbol into Market Watch
            await loop.run_in_executor(None, lambda: mt5.symbol_select(symbol, True))
            info = await loop.run_in_executor(None, lambda: mt5.symbol_info(symbol))
            if info is None:
                raise ValueError(f"Symbol {symbol} not found in MT5: {mt5.last_error()}")

        return SymbolInfo(
            symbol=info.name,
            bid=float(info.bid),
            ask=float(info.ask),
            point=float(info.point),
            digits=int(info.digits),
            spread=float(info.spread * info.point),
            contract_size=float(info.trade_contract_size),
            tick_size=float(info.trade_tick_size),
            tick_value=float(info.trade_tick_value),
            volume_min=float(info.volume_min),
            volume_max=float(info.volume_max),
            volume_step=float(info.volume_step),
        )

    async def get_tick(self, symbol: str) -> Tick:
        """Retrieve latest bid/ask tick."""
        if not self.is_connected():
            raise ConnectionError("Not connected to Windows MT5")

        loop = asyncio.get_running_loop()
        tick = await loop.run_in_executor(None, lambda: mt5.symbol_info_tick(symbol))
        if tick is None:
            raise RuntimeError(f"Failed to fetch tick for {symbol}: {mt5.last_error()}")

        spread = round(tick.ask - tick.bid, 5)
        return Tick(
            symbol=symbol,
            bid=float(tick.bid),
            ask=float(tick.ask),
            time=float(tick.time_msc) / 1000.0 if hasattr(tick, "time_msc") and tick.time_msc > 0 else float(tick.time),
            spread=spread,
        )

    async def get_rates(self, symbol: str, timeframe: str, count: int = 100) -> List[Bar]:
        """Fetch historical OHLCV bars from MT5."""
        if not self.is_connected():
            raise ConnectionError("Not connected to Windows MT5")

        tf_const = self.TIMEFRAME_MAP.get(timeframe, mt5.TIMEFRAME_M5 if mt5 else 5)
        loop = asyncio.get_running_loop()
        rates = await loop.run_in_executor(
            None,
            lambda: mt5.copy_rates_from_pos(symbol, tf_const, 0, count)
        )

        if rates is None or len(rates) == 0:
            raise RuntimeError(f"Failed to get rates for {symbol} {timeframe}: {mt5.last_error()}")

        bars: List[Bar] = []
        for r in rates:
            bars.append(Bar(
                time=int(r["time"]),
                open=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
                volume=float(r["tick_volume"]),
            ))
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
        """Send market or pending limit order to MT5 with attached SL and TP."""
        if not self.is_connected():
            return OrderResult(success=False, error="Not connected to Windows MT5")

        tick = await self.get_tick(symbol)
        loop = asyncio.get_running_loop()

        # Correct Order Type & Action Mapping
        is_pending = False
        if order_type == OrderType.BUY_LIMIT:
            mt5_action = mt5.TRADE_ACTION_PENDING
            mt5_order_type = mt5.ORDER_TYPE_BUY_LIMIT
            exec_price = price if (price is not None and price > 0) else (tick.ask - 0.50)
            is_pending = True
        elif order_type == OrderType.SELL_LIMIT:
            mt5_action = mt5.TRADE_ACTION_PENDING
            mt5_order_type = mt5.ORDER_TYPE_SELL_LIMIT
            exec_price = price if (price is not None and price > 0) else (tick.bid + 0.50)
            is_pending = True
        elif order_type == OrderType.BUY:
            mt5_action = mt5.TRADE_ACTION_DEAL
            mt5_order_type = mt5.ORDER_TYPE_BUY
            exec_price = price if (price is not None and price > 0) else tick.ask
        elif order_type == OrderType.SELL:
            mt5_action = mt5.TRADE_ACTION_DEAL
            mt5_order_type = mt5.ORDER_TYPE_SELL
            exec_price = price if (price is not None and price > 0) else tick.bid
        else:
            return OrderResult(success=False, error=f"Unsupported order type: {order_type}")

        # Dynamic filling mode negotiation
        if is_pending:
            filling_type = mt5.ORDER_FILLING_RETURN
        else:
            sym_info = await loop.run_in_executor(None, lambda: mt5.symbol_info(symbol))
            filling_mode = getattr(sym_info, "filling_mode", 0) if sym_info else 0
            if filling_mode & 1:
                filling_type = mt5.ORDER_FILLING_FOK
            elif filling_mode & 2:
                filling_type = mt5.ORDER_FILLING_IOC
            else:
                filling_type = mt5.ORDER_FILLING_RETURN

        request = {
            "action": mt5_action,
            "symbol": symbol,
            "volume": float(volume),
            "type": mt5_order_type,
            "price": float(exec_price),
            "deviation": 20,
            "magic": 15555,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,
        }

        if sl is not None and sl > 0:
            request["sl"] = float(sl)
        if tp is not None and tp > 0:
            request["tp"] = float(tp)

        result = await loop.run_in_executor(None, lambda: mt5.order_send(request))

        if result is None:
            return OrderResult(success=False, error=f"MT5 order_send returned None: {mt5.last_error()}")

        if result.retcode not in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
            error_msg = f"Order failed [retcode={result.retcode}]: {result.comment}"
            logger.error(error_msg)
            return OrderResult(success=False, error=error_msg)

        ticket = result.order or result.deal
        logger.info(
            f"[WINDOWS MT5 ORDER OK] Deal #{result.deal} / Order #{result.order} "
            f"{order_type.value} {volume} {symbol} @ {result.price:.2f}"
        )
        return OrderResult(
            success=True,
            ticket=ticket,
            price=float(result.price if result.price > 0 else exec_price),
            volume=float(result.volume if result.volume > 0 else volume),
            error=None,
        )

    async def modify_position(
        self,
        ticket: Union[int, str],
        sl: float,
        tp: float
    ) -> bool:
        """Modify existing position SL/TP in MT5. Never modifies an unrelated position."""
        if not self.is_connected():
            return False

        ticket_int = int(ticket)
        loop = asyncio.get_running_loop()

        # Strict identity verification: ensure ticket actually exists in open positions
        positions = await loop.run_in_executor(
            None,
            lambda: mt5.positions_get(ticket=ticket_int)
        )
        if not positions:
            logger.error(f"[WINDOWS MT5 MODIFY REJECTED] Position #{ticket_int} not found. Refusing to modify unrelated positions.")
            return False

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket_int,
            "sl": float(sl),
            "tp": float(tp),
        }

        result = await loop.run_in_executor(None, lambda: mt5.order_send(request))

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"[WINDOWS MT5 MODIFY OK] #{ticket} SL: {sl:.2f}, TP: {tp:.2f}")
            return True

        err = result.comment if result else str(mt5.last_error())
        logger.error(f"[WINDOWS MT5 MODIFY FAILED] #{ticket}: {err}")
        return False

    async def close_position(
        self,
        ticket: Union[int, str],
        volume: Optional[float] = None
    ) -> bool:
        """Close an active position by sending opposing deal."""
        if not self.is_connected():
            return False

        loop = asyncio.get_running_loop()
        ticket_int = int(ticket)
        positions = await loop.run_in_executor(
            None,
            lambda: mt5.positions_get(ticket=ticket_int)
        )

        if not positions:
            logger.warning(f"Position #{ticket_int} not found to close.")
            return False

        pos = positions[0]
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        close_volume = float(volume) if volume is not None and volume > 0 else float(pos.volume)

        tick = await self.get_tick(pos.symbol)
        close_price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket_int,
            "symbol": pos.symbol,
            "volume": close_volume,
            "type": close_type,
            "price": float(close_price),
            "deviation": 20,
            "magic": 15555,
            "comment": "ApexGold Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = await loop.run_in_executor(None, lambda: mt5.order_send(request))
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"[WINDOWS MT5 CLOSED] #{ticket_int} volume {close_volume} @ {close_price:.2f}")
            return True

        err = result.comment if result else str(mt5.last_error())
        logger.error(f"[WINDOWS MT5 CLOSE FAILED] #{ticket_int}: {err}")
        return False

    async def close_all_positions(self, symbol: Optional[str] = None) -> int:
        """Close all active MT5 positions."""
        positions = await self.get_open_positions(symbol)
        closed_count = 0
        for p in positions:
            if await self.close_position(p.ticket):
                closed_count += 1
        return closed_count

    async def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Fetch all currently open positions from MT5."""
        if not self.is_connected():
            return []

        loop = asyncio.get_running_loop()
        raw_positions = await loop.run_in_executor(None, mt5.positions_get)
        if raw_positions is None:
            return []

        result: List[Position] = []
        for p in raw_positions:
            if symbol and p.symbol != symbol:
                continue
            pos_type = PositionType.BUY if p.type == mt5.POSITION_TYPE_BUY else PositionType.SELL
            result.append(Position(
                ticket=int(p.ticket),
                symbol=str(p.symbol),
                type=pos_type,
                volume=float(p.volume),
                open_price=float(p.price_open),
                current_price=float(p.price_current),
                sl=float(p.sl),
                tp=float(p.tp),
                profit=float(p.profit),
                open_time=float(p.time),
                magic=int(p.magic),
                comment=str(p.comment),
            ))
        return result

    async def cancel_order(self, ticket: Union[int, str]) -> bool:
        """Cancel a pending MT5 limit/stop order."""
        if not self.is_connected():
            return False

        ticket_int = int(ticket)
        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": ticket_int,
        }

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: mt5.order_send(request))

        if result and result.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
            logger.info(f"[WINDOWS MT5 CANCEL OK] Pending order #{ticket_int} deleted.")
            return True

        err = result.comment if result else str(mt5.last_error())
        logger.error(f"[WINDOWS MT5 CANCEL FAILED] #{ticket_int}: {err}")
        return False

    async def get_pending_orders(self, symbol: Optional[str] = None) -> List[Any]:
        """Fetch all active pending limit orders from MT5."""
        if not self.is_connected():
            return []

        from brokers.base import LimitOrder
        loop = asyncio.get_running_loop()
        raw_orders = await loop.run_in_executor(None, mt5.orders_get)
        if not raw_orders:
            return []

        orders: List[LimitOrder] = []
        for o in raw_orders:
            if symbol and o.symbol != symbol:
                continue
            if o.type == mt5.ORDER_TYPE_BUY_LIMIT:
                ord_type = "BUY_LIMIT"
            elif o.type == mt5.ORDER_TYPE_SELL_LIMIT:
                ord_type = "SELL_LIMIT"
            else:
                continue

            orders.append(LimitOrder(
                ticket=int(o.ticket),
                symbol=str(o.symbol),
                order_type=ord_type,
                price=float(o.price_open),
                volume=float(o.volume_current),
                sl=float(o.sl),
                tp=float(o.tp),
                created_time=float(o.time_setup),
                comment=str(o.comment),
            ))
        return orders

    async def get_closed_deal(self, ticket: Union[int, str]) -> Optional[ClosedDeal]:
        """Fetch executed closed deal from MT5 history if available."""
        if not self.is_connected() or mt5 is None:
            return None

        loop = asyncio.get_running_loop()
        def _get_deal():
            t_int = int(ticket) if str(ticket).isdigit() else 0
            if t_int <= 0:
                return None
            now_dt = datetime.now()
            from_dt = now_dt - timedelta(days=7)
            deals = mt5.history_deals_get(position=t_int)
            if not deals:
                deals = mt5.history_deals_get(from_dt, now_dt, position=t_int)
            if not deals:
                return None

            # Find closing deal (DEAL_ENTRY_OUT or last deal)
            exit_deal = None
            for d in reversed(deals):
                if getattr(d, "entry", None) == 1:
                    exit_deal = d
                    break
            if not exit_deal:
                exit_deal = deals[-1]

            entry_deal = deals[0]
            direction = "BUY" if getattr(entry_deal, "type", 0) == 0 else "SELL"
            return ClosedDeal(
                ticket=t_int,
                symbol=str(getattr(exit_deal, "symbol", "")),
                direction=direction,
                volume=float(getattr(exit_deal, "volume", 0.0)),
                open_price=float(getattr(entry_deal, "price", 0.0)),
                close_price=float(getattr(exit_deal, "price", 0.0)),
                open_time=float(getattr(entry_deal, "time", 0.0)),
                close_time=float(getattr(exit_deal, "time", 0.0)),
                profit=float(getattr(exit_deal, "profit", 0.0)),
                commission=float(getattr(exit_deal, "commission", 0.0)),
                swap=float(getattr(exit_deal, "swap", 0.0)),
                comment=str(getattr(exit_deal, "comment", "")),
            )

        try:
            return await loop.run_in_executor(None, _get_deal)
        except Exception as e:
            logger.warning(f"Error retrieving closed deal for #{ticket}: {e}")
            return None

