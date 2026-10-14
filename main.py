"""
ApexGold Master Application & Trading Loop Orchestrator.

Starts the FastAPI Web Dashboard & WebSocket Broadcaster, initializes broker adapters,
warms up the Multi-Timeframe Data Hub, activates the Millisecond News Reactor,
and runs the autonomous Chameleon algorithmic trading loop.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
import os
import signal
import sys
import time
from typing import Optional
import yaml

import uvicorn

from backtest.engine import BacktestEngine
from brokers.base import AbstractBrokerAdapter, OrderType
from brokers.mt5_mac_socket import MacSocketMT5Adapter
from brokers.mt5_windows import WindowsMT5Adapter
from brokers.paper_sim import PaperSimAdapter
from engine.data_hub import DataHub
from engine.news_reactor import NewsReactor
from engine.trade_history_store import TradeHistoryStore
from execution.gold_risk import GoldRiskEngine
from execution.trade_manager import TradeManager
from strategy.confluence_scorer import ConfluenceScorer
from telemetry.telegram_bot import TelegramNotifier
from ui.server import app, broadcast_loop, ctx

# Configure unified logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ApexGold.Main")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load settings from YAML configuration file."""
    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} not found. Using defaults.")
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


async def create_broker_adapter(cfg: dict) -> AbstractBrokerAdapter:
    """Instantiate broker adapter according to configuration or OS environment."""
    broker_cfg = cfg.get("broker", {})
    trading_cfg = cfg.get("trading", {})
    live_trading_enabled = trading_cfg.get("live_trading_enabled", False)
    active = broker_cfg.get("active_broker", "paper_sim").lower()
    account_type = trading_cfg.get("account_type", "micro")

    # Explicit live-trading opt-in requirement:
    if active in ["mt5_mac_socket", "mt5_windows"] and not live_trading_enabled:
        logger.warning(
            f"Active broker is configured as '{active}', but 'live_trading_enabled' is False. "
            f"Enforcing safe PaperSimAdapter operation to protect real funds. "
            f"To enable live execution, set 'live_trading_enabled: true' in config.yaml."
        )
        active = "paper_sim"

    if active == "mt5_mac_socket":
        mac_cfg = broker_cfg.get("mac_socket", {})
        adapter = MacSocketMT5Adapter(
            host=mac_cfg.get("host", "127.0.0.1"),
            port=int(mac_cfg.get("port", 15555)),
            timeout_sec=float(mac_cfg.get("timeout_sec", 5.0)),
            account_type=account_type,
        )
        connected = await adapter.connect()
        if connected:
            return adapter
        logger.warning("Could not connect to MacSocketMT5Adapter. Falling back cleanly and safely to PaperSimAdapter.")

    elif active == "mt5_windows":
        win_cfg = broker_cfg.get("windows_mt5", {})
        try:
            adapter = WindowsMT5Adapter(
                terminal_path=win_cfg.get("terminal_path", None) or None,
                login=int(win_cfg.get("login", 0)),
                password=str(win_cfg.get("password", "")),
                server=str(win_cfg.get("server", "")),
                timeout_ms=int(win_cfg.get("timeout_ms", 60000)),
                account_type=account_type,
            )
            connected = await adapter.connect()
            if connected:
                return adapter
        except Exception as e:
            logger.warning(f"Windows MT5 connection error ({e}). Falling back cleanly and safely to PaperSimAdapter.")

    # Default Out-of-the-box Pure Paper Simulator
    sim_cfg = broker_cfg.get("paper_sim", {})
    adapter = PaperSimAdapter(
        initial_balance=float(sim_cfg.get("initial_balance", 10.0)),
        currency=str(sim_cfg.get("currency", "USD")),
        leverage=int(sim_cfg.get("leverage", 500)),
        account_type=account_type,
        spread_points=float(sim_cfg.get("spread_points", 25.0)),
    )
    await adapter.connect()
    return adapter


async def autonomous_trading_loop(
    broker: AbstractBrokerAdapter,
    data_hub: DataHub,
    news_reactor: NewsReactor,
    trade_manager: TradeManager,
    scorer: ConfluenceScorer,
    telegram: TelegramNotifier,
    primary_symbol: str = "XAUUSD",
):
    """
    Sub-Second Autonomous Algorithmic Execution Loop.
    Continuously ingests ticks, updates historical candles, scores confluences,
    and executes trades automatically when conditions align.
    """
    logger.info(f"Autonomous Trading Loop ACTIVE on {primary_symbol}!")
    last_candle_check = 0.0
    loop_start_time = time.time()

    while True:
        try:
            await asyncio.sleep(0.1) # 100ms cycle
            now = time.time()

            # Wait up to 6s on startup for MT5 bridge handshake to identify active broker symbol (e.g. GOLD vs XAUUSD)
            if hasattr(broker, "_mt5_queue") and broker._mt5_queue is not None:
                if ctx.mt5_last_seen == 0.0 and (now - loop_start_time < 6.0):
                    await asyncio.sleep(0.2)
                    continue

            cur_sym = ctx.active_symbol or primary_symbol

            # 1. Fetch live tick from broker
            tick = await broker.get_tick(cur_sym)
            if not tick:
                continue

            # 2. Feed tick into NewsReactor
            tick_metrics = news_reactor.on_tick(tick)

            # 3. Periodically evaluate closed candles & high-velocity breakout triggers
            # Check every 2 seconds or immediately on high-velocity impulse
            should_evaluate = (now - last_candle_check >= 2.0) or tick_metrics.get("is_impulse_active", False)

            if should_evaluate:
                last_candle_check = now

                # Pull multi-timeframe data
                dfs = await data_hub.get_multi_timeframe_data(cur_sym)
                if not dfs.get("5m") is not None or len(dfs["5m"]) < 20:
                    # Fallback to base symbol if specific broker suffix is still warming up
                    if cur_sym != primary_symbol:
                        dfs = await data_hub.get_multi_timeframe_data(primary_symbol)
                    if not dfs.get("5m") is not None or len(dfs["5m"]) < 20:
                        continue

                # Refresh Smart Auto-Pilot Brain
                sym_info = await broker.get_symbol_info(cur_sym)
                cur_spread = sym_info.spread if sym_info else 0.25
                max_spread = 0.80 if any(k in cur_sym.upper() for k in ["XAU", "GOLD"]) else 0.0003
                scorer.evaluate_auto_brain(
                    dfs=dfs,
                    current_spread=cur_spread,
                    max_spread=max_spread,
                    tick_velocity=tick_metrics.get("velocity_factor", 1.0),
                    news_reactor=news_reactor,
                    symbol=cur_sym,
                    now_ts=now,
                )

                # Run 100-Point Confluence Scoring Engine
                buy_sig, sell_sig = scorer.evaluate(
                    symbol=cur_sym,
                    dfs=dfs,
                    current_tick_bid=tick.bid,
                    current_tick_ask=tick.ask,
                    is_impulse_active=tick_metrics.get("is_impulse_active", False),
                    utc_timestamp=now,
                )

                # Prioritize highest scoring signal
                candidate_signal = None
                if buy_sig and sell_sig:
                    candidate_signal = buy_sig if buy_sig.score >= sell_sig.score else sell_sig
                else:
                    candidate_signal = buy_sig or sell_sig

                if candidate_signal:
                    threshold = scorer.get_active_threshold()
                    if candidate_signal.score >= threshold:
                        mode = ctx.execution_mode

                        if mode == "full_auto":
                            # Auto-Execute Live Order
                            res = await trade_manager.execute_trade(candidate_signal)
                            if res.success:
                                await telegram.notify_order_filled(
                                    ticket=int(res.ticket) if str(res.ticket).isdigit() else 0,
                                    symbol=candidate_signal.symbol,
                                    direction=candidate_signal.direction,
                                    volume=res.volume,
                                    price=res.price,
                                    sl=candidate_signal.suggested_sl,
                                    tp=candidate_signal.suggested_tp,
                                    regime=candidate_signal.regime,
                                )
                        elif mode == "confirm":
                            # Send 1-Click Alert
                            logger.info(f"[1-CLICK CONFIRM REQUIRED] {candidate_signal.direction} {primary_symbol} ({candidate_signal.score}/100)")
                            await telegram.notify_signal(
                                symbol=candidate_signal.symbol,
                                direction=candidate_signal.direction,
                                score=candidate_signal.score,
                                tier=candidate_signal.tier,
                                regime=candidate_signal.regime,
                                entry=candidate_signal.entry_price,
                                sl=candidate_signal.suggested_sl,
                                tp=candidate_signal.suggested_tp,
                            )
                        elif mode == "scanner":
                            logger.info(f"[SCANNER FLOW] {candidate_signal.direction} {primary_symbol} Score: {candidate_signal.score}/100 ({candidate_signal.regime})")

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in autonomous trading loop: {e}", exc_info=True)
            await asyncio.sleep(1.0)


@asynccontextmanager
async def lifespan(app_instance):
    """FastAPI startup and shutdown lifecycle manager."""
    cfg = load_config("config.yaml")
    ctx.config = cfg

    trading_cfg = cfg.get("trading", {})
    ctx.active_symbol = trading_cfg.get("primary_symbol", "XAUUSD")
    ctx.live_trading_enabled = bool(trading_cfg.get("live_trading_enabled", False))
    ctx.execution_mode = trading_cfg.get("execution_mode", "scanner")
    ctx.strategy_mode = trading_cfg.get("strategy_mode", "auto_adaptive")

    # 1. Connect Broker (Strict Paper/Live separation)
    broker = await create_broker_adapter(cfg)
    ctx.broker = broker

    # 2. Initialize Data Hub & SQLite Store
    data_cfg = cfg.get("data_hub", {})
    data_hub = DataHub(db_path=data_cfg.get("database_path", "data/market_history.db"))
    await data_hub.initialize()
    ctx.data_hub = data_hub

    # 2b. Initialize Persistent Trade History & Performance Analytics Store
    history_store = TradeHistoryStore(db_path=data_cfg.get("database_path", "data/market_history.db"))
    await history_store.initialize()
    ctx.history_store = history_store

    # Warmup historical bars in background
    symbols_to_warm = trading_cfg.get("allowed_symbols", ["XAUUSD"])
    asyncio.create_task(data_hub.warmup(broker, symbols_to_warm))

    # 3. Start News Reactor
    news_cfg = cfg.get("news_reactor", {})
    news_reactor = NewsReactor(
        enabled=news_cfg.get("enabled", True),
        pre_news_freeze_seconds=news_cfg.get("pre_news_freeze_seconds", 180),
        post_news_momentum_window_sec=news_cfg.get("post_news_momentum_window_sec", 300),
        post_news_cooldown_seconds=news_cfg.get("post_news_cooldown_seconds", 3),
        max_spread_multiplier=news_cfg.get("max_spread_multiplier", 2.5),
        tick_velocity_window_ms=news_cfg.get("tick_velocity_window_ms", 800),
        impulse_tick_density_mult=news_cfg.get("tick_density_multiplier", 3.0),
        tracked_currencies=news_cfg.get("currencies", ["USD", "XAU"]),
    )
    await news_reactor.start()
    ctx.news_reactor = news_reactor

    # 4. Initialize Risk Engine
    risk_engine = GoldRiskEngine(
        account_type=trading_cfg.get("account_type", "cent"),
        budget_usd=float(trading_cfg.get("budget_usd", 10.0)),
        risk_per_trade_pct=float(trading_cfg.get("risk_per_trade_pct", 10.0)),
        compounding_enabled=trading_cfg.get("compounding_enabled", True),
        max_daily_losses=int(trading_cfg.get("max_daily_losses", 4)),
    )
    ctx.risk_engine = risk_engine

    # 5. Initialize Trade Manager & Chameleon Loop
    trade_manager = TradeManager(
        broker=broker,
        data_hub=data_hub,
        news_reactor=news_reactor,
        risk_engine=risk_engine,
        correlation_guard=trading_cfg.get("correlation_guard", True),
        max_daily_losses=int(trading_cfg.get("max_daily_losses", 4)),
        max_daily_drawdown_pct=float(trading_cfg.get("max_daily_drawdown_pct", 25.0)),
        max_open_positions=int(trading_cfg.get("max_open_positions", 1)),
        history_store=history_store,
    )
    await trade_manager.start()
    ctx.trade_manager = trade_manager

    # 6. Initialize Confluence Scorer
    strat_cfg = cfg.get("strategy", {})
    thresh = strat_cfg.get("thresholds", {})
    scorer = ConfluenceScorer(
        threshold_aggressive=int(thresh.get("aggressive", 60)),
        threshold_balanced=int(thresh.get("balanced", 68)),
        threshold_sniper=int(thresh.get("sniper", 82)),
        mode=ctx.strategy_mode,
    )
    ctx.scorer = scorer

    # 7. Initialize Backtest Engine
    ctx.backtester = BacktestEngine(scorer=scorer, initial_balance=float(trading_cfg.get("budget_usd", 10.0)))

    # 8. Initialize Telegram Notifier
    tg_cfg = cfg.get("telegram", {})
    telegram = TelegramNotifier(
        bot_token=tg_cfg.get("bot_token", ""),
        chat_id=tg_cfg.get("chat_id", ""),
        enabled=tg_cfg.get("enabled", False),
    )
    ctx.telegram = telegram

    # 9. Launch Background Broadcaster & Autonomous Trading Loops
    broadcaster_task = asyncio.create_task(broadcast_loop())
    trading_task = asyncio.create_task(
        autonomous_trading_loop(
            broker=broker,
            data_hub=data_hub,
            news_reactor=news_reactor,
            trade_manager=trade_manager,
            scorer=scorer,
            telegram=telegram,
            primary_symbol=ctx.active_symbol,
        )
    )

    logger.info("=" * 60)
    logger.info(" APEXGOLD TRADING PLATFORM IS FULLY OPERATIONAL!")
    logger.info(" Web Dashboard: http://127.0.0.1:8000")
    logger.info(f" Active Broker: {broker.__class__.__name__}")
    logger.info(f" Primary Symbol: {ctx.active_symbol} | Mode: {ctx.execution_mode.upper()} ({ctx.strategy_mode.upper()})")
    logger.info("=" * 60)

    yield

    # Clean shutdown
    logger.info("Initiating ApexGold shutdown...")
    broadcaster_task.cancel()
    trading_task.cancel()
    await trade_manager.stop()
    await news_reactor.stop()
    await broker.disconnect()
    logger.info("ApexGold shutdown complete.")


# Attach lifespan to FastAPI app
app.router.lifespan_context = lifespan


def main():
    """CLI launcher entrypoint."""
    cfg = load_config("config.yaml")
    ui_cfg = cfg.get("ui", {})
    host = ui_cfg.get("host", "127.0.0.1")
    port = int(ui_cfg.get("port", 8000))

    uvicorn.run("main:app", host=host, port=port, log_level="info", reload=False)


if __name__ == "__main__":
    main()

