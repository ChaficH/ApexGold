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
from typing import Dict, List, Optional, Union

from brokers.base import (
    AbstractBrokerAdapter,
    AccountInfo,
    Bar,
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
        """Send market order to MT5 with attached SL and TP."""
        if not self.is_connected():
            return OrderResult(success=False, error="Not connected to Windows MT5")

        tick = await self.get_tick(symbol)
        mt5_order_type = mt5.ORDER_TYPE_BUY if order_type == OrderType.BUY else mt5.ORDER_TYPE_SELL
        exec_price = tick.ask if order_type == OrderType.BUY else tick.bid
        if price is not None and price > 0:
            exec_price = price

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": mt5_order_type,
            "price": float(exec_price),
            "deviation": 20,
            "magic": 15555,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if sl is not None and sl > 0:
            request["sl"] = float(sl)
        if tp is not None and tp > 0:
            request["tp"] = float(tp)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: mt5.order_send(request))

        if result is None:
            return OrderResult(success=False, error=f"MT5 order_send returned None: {mt5.last_error()}")

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            error_msg = f"Order failed [retcode={result.retcode}]: {result.comment}"
            logger.error(error_msg)
            return OrderResult(success=False, error=error_msg)

        logger.info(
            f"[WINDOWS MT5 ORDER FILLED] Deal #{result.deal} / Order #{result.order} "
            f"{order_type.value} {volume} {symbol} @ {result.price:.2f}"
        )
        return OrderResult(
            success=True,
            ticket=result.order,
            price=float(result.price),
            volume=float(result.volume),
            error=None,
        )

    async def modify_position(
        self,
        ticket: Union[int, str],
        sl: float,
        tp: float
    ) -> bool:
        """Modify existing position SL/TP in MT5."""
        if not self.is_connected():
            return False

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": int(ticket),
            "sl": float(sl),
            "tp": float(tp),
        }

        loop = asyncio.get_running_loop()
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

