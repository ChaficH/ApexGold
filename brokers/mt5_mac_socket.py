"""
ApexGold macOS/Cross-Platform High-Speed Socket MT5 Adapter.

Connects directly via TCP socket (127.0.0.1:15555) to the ApexBridge.mq5 Expert Advisor
running inside MetaTrader 5 on macOS (Wine/CrossOver/native) or Windows.
Provides sub-2ms execution, tick streaming, historical data fetching,
and position management with zero third-party bridge fees.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union

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

logger = logging.getLogger("ApexGold.MacSocketMT5")


class MacSocketMT5Adapter(AbstractBrokerAdapter):
    """
    Sub-millisecond TCP Socket Client for ApexBridge.mq5.
    Sends newline-delimited JSON RPC requests and parses real-time responses.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 15555,
        timeout_sec: float = 5.0,
        account_type: str = "cent",
    ):
        self.host = host
        self.port = port
        self.timeout_sec = timeout_sec
        self.account_type = account_type

        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._connected = False
        self._request_id = 0
        self._lock = asyncio.Lock()
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._listen_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """Establish TCP connection to ApexBridge.mq5 socket server."""
        try:
            logger.info(f"Connecting to ApexBridge.mq5 at {self.host}:{self.port}...")
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=self.timeout_sec,
            )
            self._connected = True

            # Start background reader task
            if self._listen_task is None or self._listen_task.done():
                self._listen_task = asyncio.create_task(self._socket_reader_loop())

            # Perform handshake ping
            pong = await self._send_command({"action": "PING"})
            if pong.get("status") == "OK":
                logger.info("Successfully connected to ApexBridge.mq5 server!")
                return True
            else:
                logger.warning(f"ApexBridge ping returned non-OK: {pong}")
                return True
        except Exception as e:
            logger.error(
                f"Could not connect to ApexBridge.mq5 on {self.host}:{self.port} ({e}). "
                "Ensure MT5 is running and ApexBridge EA is attached to any chart."
            )
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Close socket connection cleanly."""
        self._connected = False
        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass

        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None
            self._reader = None
        logger.info("ApexBridge.mq5 socket disconnected.")

    def is_connected(self) -> bool:
        """Check if TCP socket is active."""
        return self._connected and self._writer is not None and not self._writer.is_closing()

    async def _send_command(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a JSON RPC command and await matched response.
        Thread/task safe with unique req_id tracking.
        """
        if not self.is_connected():
            raise ConnectionError(f"ApexBridge socket disconnected ({self.host}:{self.port})")

        async with self._lock:
            self._request_id += 1
            req_id = self._request_id
            payload["req_id"] = req_id

            loop = asyncio.get_running_loop()
            future: asyncio.Future = loop.create_future()
            self._pending_requests[req_id] = future

            msg = (json.dumps(payload) + "\n").encode("utf-8")
            self._writer.write(msg)
            await self._writer.drain()

        try:
            response = await asyncio.wait_for(future, timeout=self.timeout_sec)
            return response
        except asyncio.TimeoutError:
            self._pending_requests.pop(req_id, None)
            raise TimeoutError(f"ApexBridge request timed out after {self.timeout_sec}s: {payload.get('action')}")
        except Exception as e:
            self._pending_requests.pop(req_id, None)
            raise e

    async def _socket_reader_loop(self) -> None:
        """Continuously reads incoming newline-delimited JSON messages from MT5."""
        while self._connected and self._reader:
            try:
                line = await self._reader.readline()
                if not line:
                    logger.warning("ApexBridge socket reached EOF. Reconnecting...")
                    self._connected = False
                    break

                text = line.decode("utf-8").strip()
                if not text:
                    continue

                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    logger.error(f"Malformed JSON from ApexBridge: {text}")
                    continue

                req_id = data.get("req_id")
                if req_id in self._pending_requests:
                    fut = self._pending_requests.pop(req_id)
                    if not fut.done():
                        fut.set_result(data)
                else:
                    logger.debug(f"Received unsolicited broadcast from ApexBridge: {data}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ApexBridge socket reader: {e}")
                self._connected = False
                break

    async def get_account_info(self) -> AccountInfo:
        """Retrieve real-time account details from MT5 via EA."""
        res = await self._send_command({"action": "ACCOUNT_INFO"})
        if res.get("status") != "OK":
            raise RuntimeError(f"ApexBridge ACCOUNT_INFO error: {res.get('error')}")

        d = res.get("data", {})
        return AccountInfo(
            balance=float(d.get("balance", 0.0)),
            equity=float(d.get("equity", 0.0)),
            margin=float(d.get("margin", 0.0)),
            free_margin=float(d.get("free_margin", 0.0)),
            currency=str(d.get("currency", "USD")),
            leverage=int(d.get("leverage", 500)),
            account_type=self.account_type,
        )

    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """Fetch symbol quotes and contract details from MT5 via EA."""
        res = await self._send_command({"action": "SYMBOL_INFO", "symbol": symbol})
        if res.get("status") != "OK":
            raise RuntimeError(f"ApexBridge SYMBOL_INFO error: {res.get('error')}")

        d = res.get("data", {})
        point = float(d.get("point", 0.01))
        spread_raw = float(d.get("spread", 0.0))
        # If spread in points, convert to price distance
        spread = round(spread_raw * point, int(d.get("digits", 2))) if spread_raw > 1.0 else spread_raw

        return SymbolInfo(
            symbol=str(d.get("symbol", symbol)),
            bid=float(d.get("bid", 0.0)),
            ask=float(d.get("ask", 0.0)),
            point=point,
            digits=int(d.get("digits", 2)),
            spread=spread,
            contract_size=float(d.get("contract_size", 100.0)),
            tick_size=float(d.get("tick_size", point)),
            tick_value=float(d.get("tick_value", 1.0)),
            volume_min=float(d.get("volume_min", 0.01)),
            volume_max=float(d.get("volume_max", 100.0)),
            volume_step=float(d.get("volume_step", 0.01)),
        )

    async def get_tick(self, symbol: str) -> Tick:
        """Fetch latest live tick for symbol."""
        res = await self._send_command({"action": "TICK", "symbol": symbol})
        if res.get("status") != "OK":
            # Fall back to SYMBOL_INFO if TICK is not separate
            s_info = await self.get_symbol_info(symbol)
            return Tick(
                symbol=symbol,
                bid=s_info.bid,
                ask=s_info.ask,
                time=time.time(),
                spread=s_info.spread,
            )

        d = res.get("data", {})
        bid = float(d.get("bid", 0.0))
        ask = float(d.get("ask", 0.0))
        return Tick(
            symbol=symbol,
            bid=bid,
            ask=ask,
            time=float(d.get("time", time.time())),
            spread=round(ask - bid, 5),
        )

    async def get_rates(self, symbol: str, timeframe: str, count: int = 100) -> List[Bar]:
        """Fetch historical bars via MT5 CopyRates."""
        # Normalize timeframe: 1m -> M1, 5m -> M5, 15m -> M15, 1H -> H1
        tf_map = {
            "1m": "M1", "1M": "M1", "M1": "M1",
            "5m": "M5", "5M": "M5", "M5": "M5",
            "15m": "M15", "15M": "M15", "M15": "M15",
            "1h": "H1", "1H": "H1", "H1": "H1", "60m": "H1",
        }
        ea_tf = tf_map.get(timeframe, "M5")

        res = await self._send_command({
            "action": "GET_RATES",
            "symbol": symbol,
            "timeframe": ea_tf,
            "count": int(count),
        })

        if res.get("status") != "OK":
            raise RuntimeError(f"ApexBridge GET_RATES error: {res.get('error')}")

        raw_bars = res.get("data", [])
        bars: List[Bar] = []
        for b in raw_bars:
            bars.append(Bar(
                time=int(b.get("time", 0)),
                open=float(b.get("open", 0.0)),
                high=float(b.get("high", 0.0)),
                low=float(b.get("low", 0.0)),
                close=float(b.get("close", 0.0)),
                volume=float(b.get("volume", 0.0)),
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
        """Send market order to MT5 via ApexBridge socket."""
        payload = {
            "action": "ORDER_SEND",
            "symbol": symbol,
            "type": order_type.value,
            "volume": float(volume),
            "comment": comment,
        }
        if price is not None and price > 0:
            payload["price"] = float(price)
        if sl is not None and sl > 0:
            payload["sl"] = float(sl)
        if tp is not None and tp > 0:
            payload["tp"] = float(tp)

        res = await self._send_command(payload)
        if res.get("status") != "OK":
            err = res.get("error", "Unknown execution error")
            logger.error(f"[ApexBridge ORDER FAILED] {err}")
            return OrderResult(success=False, error=err)

        d = res.get("data", {})
        ticket = d.get("ticket") or d.get("order")
        fill_price = float(d.get("price", 0.0))
        fill_vol = float(d.get("volume", volume))

        logger.info(f"[ApexBridge ORDER OK] #{ticket} {order_type.value} {fill_vol} {symbol} @ {fill_price:.2f}")
        return OrderResult(
            success=True,
            ticket=ticket,
            price=fill_price,
            volume=fill_vol,
            error=None,
        )

    async def modify_position(
        self,
        ticket: Union[int, str],
        sl: float,
        tp: float
    ) -> bool:
        """Modify position SL/TP in MT5 via ApexBridge."""
        payload = {
            "action": "ORDER_MODIFY",
            "ticket": int(ticket),
            "sl": float(sl),
            "tp": float(tp),
        }
        res = await self._send_command(payload)
        if res.get("status") == "OK":
            logger.info(f"[ApexBridge MODIFY OK] #{ticket} SL: {sl:.2f}, TP: {tp:.2f}")
            return True
        logger.error(f"[ApexBridge MODIFY FAILED] #{ticket}: {res.get('error')}")
        return False

    async def close_position(
        self,
        ticket: Union[int, str],
        volume: Optional[float] = None
    ) -> bool:
        """Close position in MT5 via ApexBridge."""
        payload = {
            "action": "ORDER_CLOSE",
            "ticket": int(ticket),
        }
        if volume is not None and volume > 0:
            payload["volume"] = float(volume)

        res = await self._send_command(payload)
        if res.get("status") == "OK":
            logger.info(f"[ApexBridge CLOSE OK] #{ticket}")
            return True
        logger.error(f"[ApexBridge CLOSE FAILED] #{ticket}: {res.get('error')}")
        return False

    async def close_all_positions(self, symbol: Optional[str] = None) -> int:
        """Close all positions via ApexBridge."""
        payload = {"action": "CLOSE_ALL"}
        if symbol:
            payload["symbol"] = symbol

        res = await self._send_command(payload)
        if res.get("status") == "OK":
            count = int(res.get("data", {}).get("closed_count", 0))
            logger.info(f"[ApexBridge CLOSE ALL OK] Closed {count} positions.")
            return count

        # Fallback to manual iteration if server returns single error
        positions = await self.get_open_positions(symbol)
        closed = 0
        for p in positions:
            if await self.close_position(p.ticket):
                closed += 1
        return closed

    async def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Retrieve all currently active positions from MT5."""
        payload = {"action": "GET_POSITIONS"}
        if symbol:
            payload["symbol"] = symbol

        res = await self._send_command(payload)
        if res.get("status") != "OK":
            logger.error(f"[ApexBridge GET_POSITIONS FAILED]: {res.get('error')}")
            return []

        raw_positions = res.get("data", [])
        positions: List[Position] = []
        for p in raw_positions:
            pos_type = PositionType.BUY if p.get("type", "BUY") == "BUY" else PositionType.SELL
            positions.append(Position(
                ticket=int(p.get("ticket", 0)),
                symbol=str(p.get("symbol", "")),
                type=pos_type,
                volume=float(p.get("volume", 0.0)),
                open_price=float(p.get("open_price", 0.0)),
                current_price=float(p.get("current_price", 0.0)),
                sl=float(p.get("sl", 0.0)),
                tp=float(p.get("tp", 0.0)),
                profit=float(p.get("profit", 0.0)),
                open_time=float(p.get("open_time", time.time())),
                magic=int(p.get("magic", 15555)),
                comment=str(p.get("comment", "")),
            ))
        return positions

