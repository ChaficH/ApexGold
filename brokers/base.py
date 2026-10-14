"""
ApexGold Broker Base Interface & Common Data Structures.

Provides standard cross-platform data representations and the abstract base class
(AbstractBrokerAdapter) that all broker adapters (PaperSim, Windows MT5, Mac Socket MT5)
must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Dict, List, Optional, Union, Any
import time


class OrderType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    BUY_LIMIT = "BUY_LIMIT"
    SELL_LIMIT = "SELL_LIMIT"


class PositionType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Tick:
    """Live price tick representation."""
    symbol: str
    bid: float
    ask: float
    time: float = field(default_factory=time.time)
    spread: float = 0.0

    def __post_init__(self):
        if self.spread == 0.0 and self.ask >= self.bid:
            self.spread = round(self.ask - self.bid, 5)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Bar:
    """Standardized OHLCV candle representation."""
    time: int            # Unix timestamp in seconds
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    is_synthetic: bool = False
    source: str = "BROKER"

    def __post_init__(self):
        if self.is_synthetic and self.source == "BROKER":
            self.source = "SYNTHETIC"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Position:
    """Active open market position."""
    ticket: Union[int, str]
    symbol: str
    type: PositionType
    volume: float
    open_price: float
    current_price: float
    sl: float = 0.0
    tp: float = 0.0
    profit: float = 0.0
    open_time: float = field(default_factory=time.time)
    magic: int = 15555
    comment: str = ""
    is_user_manual: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value if isinstance(self.type, PositionType) else str(self.type)
        return d


@dataclass
class SymbolInfo:
    """Instrument contract specification and live quotes."""
    symbol: str
    bid: float
    ask: float
    point: float = 0.01
    digits: int = 2
    spread: float = 0.0
    contract_size: float = 100.0  # 100 oz per standard lot on Gold
    tick_size: float = 0.01
    tick_value: float = 1.0       # $ value of 1 tick on 1 lot
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    stops_level: int = 0

    @property
    def min_lot(self) -> float:
        return self.volume_min

    @property
    def max_lot(self) -> float:
        return self.volume_max

    @property
    def step_lot(self) -> float:
        return self.volume_step

    def __post_init__(self):
        if self.spread == 0.0 and self.ask >= self.bid:
            self.spread = round(self.ask - self.bid, self.digits)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AccountInfo:
    """Broker account equity and balance details."""
    balance: float
    equity: float
    margin: float = 0.0
    free_margin: float = 0.0
    currency: str = "USD"
    leverage: int = 500
    account_type: str = "cent"  # "cent", "micro", "standard", "paper"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LimitOrder:
    """Pending limit order representation with predictive invalidation guard."""
    ticket: Union[int, str]
    symbol: str
    order_type: str        # "BUY_LIMIT" or "SELL_LIMIT"
    price: float           # target entry price
    volume: float
    sl: float = 0.0
    tp: float = 0.0
    created_time: float = field(default_factory=time.time)
    timeout_seconds: float = 120.0
    invalidation_price: float = 0.0 # Price threshold that invalidates the limit setup
    comment: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OrderResult:
    """Result of an order execution request."""
    success: bool
    ticket: Optional[Union[int, str]] = None
    price: float = 0.0
    volume: float = 0.0
    error: Optional[str] = None

@dataclass
class ClosedDeal:
    """Historical closed deal record from broker."""
    ticket: Union[int, str]
    symbol: str
    direction: str
    volume: float
    open_price: float
    close_price: float
    open_time: float
    close_time: float
    profit: float
    commission: float = 0.0
    swap: float = 0.0
    comment: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AbstractBrokerAdapter(ABC):
    """
    Abstract interface for all broker connections.
    Enables unified operation across Paper Simulation, Native Windows MT5,
    and High-Speed Mac Socket MT5 Bridge.
    """

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the broker terminal or server."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Gracefully close broker connection."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if connection is active and healthy."""
        pass

    @abstractmethod
    async def get_account_info(self) -> AccountInfo:
        """Retrieve current balance, equity, and margin details."""
        pass

    @abstractmethod
    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """Fetch real-time specifications and quotes for symbol."""
        pass

    @abstractmethod
    async def get_tick(self, symbol: str) -> Tick:
        """Fetch the latest bid/ask tick for symbol."""
        pass

    @abstractmethod
    async def get_rates(self, symbol: str, timeframe: str, count: int = 100) -> List[Bar]:
        """
        Fetch historical OHLCV bars.
        Timeframe can be '1m'/'M1', '5m'/'M5', '15m'/'M15', '1H'/'H1'.
        """
        pass

    @abstractmethod
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
        """Execute a market order with attached SL and TP."""
        pass

    @abstractmethod
    async def modify_position(
        self,
        ticket: Union[int, str],
        sl: float,
        tp: float
    ) -> bool:
        """Modify existing position Stop Loss and Take Profit levels."""
        pass

    @abstractmethod
    async def close_position(
        self,
        ticket: Union[int, str],
        volume: Optional[float] = None
    ) -> bool:
        """Close an open position (full or partial)."""
        pass

    @abstractmethod
    async def close_all_positions(self, symbol: Optional[str] = None) -> int:
        """Close all active positions. Returns number of closed positions."""
        pass

    @abstractmethod
    async def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Retrieve all currently active positions."""
        pass

    async def cancel_order(self, ticket: Union[int, str]) -> bool:
        """Cancel a pending limit or stop order."""
        return False

    async def get_pending_orders(self, symbol: Optional[str] = None) -> List[LimitOrder]:
        """Retrieve all currently active pending orders."""
        return []

    async def get_closed_deal(self, ticket: Union[int, str]) -> Optional[ClosedDeal]:
        """Retrieve real executed closed deal history for ticket if supported."""
        return None

