"""
ApexGold Real-Time Web UI Server & WebSocket Broadcaster.

FastAPI backend serving the institutional dark-mode TradingView Lightweight Charts dashboard,
managing WebSocket telemetry streams (4 FPS), REST action endpoints, Smart Auto-Pilot Brain
telemetry, and the macOS Wine MT5 WebRequest bridge (/api/mt5_bridge/tick).
"""

import asyncio
from datetime import datetime, timezone
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backtest.engine import BacktestEngine
from brokers.base import AbstractBrokerAdapter, PositionType
from engine.data_hub import DataHub
from engine.news_reactor import NewsReactor
from execution.gold_risk import GoldRiskEngine
from execution.trade_manager import TradeManager
from strategy.confluence_scorer import ConfluenceScorer
from telemetry.telegram_bot import TelegramNotifier

logger = logging.getLogger("ApexGold.UIServer")

app = FastAPI(title="ApexGold Trading Engine", version="1.0.0")

ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
custom_origins = os.getenv("APEX_ALLOWED_ORIGINS", "")
if custom_origins:
    ALLOWED_ORIGINS.extend([orig.strip() for orig in custom_origins.split(",") if orig.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Server Global State Context
# ------------------------------------------------------------------------------
class ServerContext:
    def __init__(self):
        self.broker: Optional[AbstractBrokerAdapter] = None
        self.data_hub: Optional[DataHub] = None
        self.news_reactor: Optional[NewsReactor] = None
        self.risk_engine: Optional[GoldRiskEngine] = None
        self.trade_manager: Optional[TradeManager] = None
        self.scorer: Optional[ConfluenceScorer] = None
        self.backtester: Optional[BacktestEngine] = None
        self.telegram: Optional[TelegramNotifier] = None
        self.config: Dict[str, Any] = {}
        self.active_symbol = "XAUUSD"
        self.execution_mode = "scanner"     # scanner (SAFE RUNTIME DEFAULT), confirm, full_auto
        self.strategy_mode = "auto_adaptive" # auto_adaptive (FLAGSHIP DEFAULT), cautious, standard, aggressive, hyper, manual
        self.aggressiveness_pct = 65.0      # 1% to 100% continuous dial
        
        self.history_store: Optional[Any] = None
        self.live_trading_enabled: bool = False
        self.manage_manual_positions: bool = False
        self.bridge_token: str = ""
        self.allow_pre_news_trajectory: bool = True
        self.latest_broker_time: float = 0.0
        
        # macOS Wine MT5 Bridge Pending Queue
        self.mt5_pending_actions: List[Dict[str, Any]] = []
        self.mt5_last_seen = 0.0
        self.mt5_algo_trading_allowed = True
        self.planned_limits: Dict[str, Any] = {}
        self.auto_limits_enabled: bool = False
        self.predictive_brain: Optional[Any] = None

    def is_live_trading_active(self) -> bool:
        """Returns True only when live trading is explicitly enabled on a real MT5 broker adapter."""
        if not self.live_trading_enabled:
            return False
        if not self.broker:
            return False
        if getattr(self.broker, "is_paper", False):
            return False
        from brokers.paper_sim import PaperSimAdapter
        if isinstance(self.broker, PaperSimAdapter):
            return False
        return True

ctx = ServerContext()


# ------------------------------------------------------------------------------
# Execution Rate Limiter & Debouncer
# ------------------------------------------------------------------------------
class EndpointRateLimiter:
    """In-memory rate limiter and debouncer for sensitive execution endpoints."""
    def __init__(self):
        self._last_call: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def check(self, key: str, min_interval: float = 0.5) -> bool:
        async with self._lock:
            now = time.time()
            last = self._last_call.get(key, 0.0)
            if (now - last) < min_interval:
                return False
            self._last_call[key] = now
            return True

endpoint_rate_limiter = EndpointRateLimiter()


def verify_bridge_token(request: Optional[Request] = None, payload: Optional[Dict[str, Any]] = None):
    """
    Authenticate incoming bridge requests via header, payload, or query parameter.
    If live trading is enabled or active, a non-empty bridge token is strictly required.
    """
    required = (getattr(ctx, "bridge_token", "") or os.environ.get("APEX_BRIDGE_TOKEN", "")).strip()

    # Requirement: Live bridge use strictly requires configured authentication
    if ctx.is_live_trading_active() or ctx.live_trading_enabled:
        if not required:
            client_host = request.client.host if (request and request.client) else "unknown"
            logger.critical(f"[BRIDGE AUTH ERROR] Live trading is enabled but no bridge authentication token is configured! Rejecting bridge access from {client_host}.")
            raise HTTPException(
                status_code=401,
                detail="Live bridge access requires configured bridge token (bridge_token in config or APEX_BRIDGE_TOKEN env)"
            )

    if not required:
        return  # Bridge authentication not configured in scanner / paper mode

    header_token = (request.headers.get("X-Bridge-Token") or request.headers.get("x-bridge-token")) if request else None
    payload_token = payload.get("token") if payload and isinstance(payload, dict) else None
    query_token = request.query_params.get("token") if request else None
    token = header_token or payload_token or query_token
    if not token or token != required:
        client_host = request.client.host if (request and request.client) else "unknown"
        logger.warning(f"[BRIDGE AUTH REJECTED] Unauthorized bridge request from {client_host}")
        raise HTTPException(status_code=401, detail="Unauthorized bridge access: invalid or missing bridge token")



# ------------------------------------------------------------------------------
# WebSocket Connection Hub
# ------------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        if not self.active_connections:
            return
        dead = set()
        data_text = json.dumps(message)
        for ws in self.active_connections:
            try:
                await ws.send_text(data_text)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.active_connections.discard(ws)

ws_manager = ConnectionManager()


# ------------------------------------------------------------------------------
# HTML Dashboard Entrypoint
# ------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the single-page dark-mode dashboard."""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>ApexGold Dashboard loading... Please ensure ui/index.html exists.</h2>")


# ------------------------------------------------------------------------------
# REST API Endpoints
# ------------------------------------------------------------------------------
@app.get("/api/status")
async def get_status():
    """Return live system state snapshot."""
    acc = await ctx.broker.get_account_info() if ctx.broker else None
    tick = await ctx.broker.get_tick(ctx.active_symbol) if ctx.broker else None
    is_bridge_connected = (asyncio.get_event_loop().time() - ctx.mt5_last_seen < 3.0)
    is_live = ctx.is_live_trading_active() and is_bridge_connected

    if is_live:
        broker_name = "MT5 MAC WINE BRIDGE (LIVE)"
    elif is_bridge_connected:
        broker_name = "MT5 BRIDGE (SCANNER / PAPER MODE)"
    elif ctx.broker and getattr(ctx.broker, "is_paper", False):
        broker_name = "PAPER_SIM ACTIVE (SIMULATED)"
    elif ctx.broker and ctx.broker.is_connected():
        broker_name = f"{ctx.broker.__class__.__name__} (CONNECTED)"
    else:
        broker_name = "DISCONNECTED"

    if hasattr(ctx.broker, "_positions") and ctx.broker._positions:
        positions = []
        for p in ctx.broker._positions.values():
            mp = ctx.trade_manager._managed_positions.get(p.ticket) if ctx.trade_manager else None
            positions.append({
                "ticket": p.ticket,
                "symbol": p.symbol,
                "direction": p.type.value if hasattr(p.type, "value") else str(p.type),
                "volume": p.volume,
                "open_price": p.open_price,
                "current_price": getattr(p, "current_price", p.open_price),
                "sl": p.sl,
                "tp": p.tp,
                "profit": getattr(p, "profit", 0.0),
                "r_multiple": getattr(mp, "r_multiple", 0.0) if mp else 0.0,
                "ratchet_stage": getattr(mp, "ratchet_stage", 0) if mp else 0,
                "highest_r_reached": getattr(mp, "highest_r_reached", 0.0) if mp else 0.0,
                "peak_r_giveback_pct": getattr(mp, "peak_r_giveback_pct", 0.0) if mp else 0.0,
                "exit_evaluation": getattr(mp, "exit_evaluation", "HOLD_AND_EXPAND") if mp else "HOLD_AND_EXPAND",
                "exit_verdict_detail": getattr(mp, "exit_verdict_detail", "Market structure intact. Holding for expansion.") if mp else "Market structure intact.",
                "comment": getattr(p, "comment", ""),
                "is_user_manual": getattr(p, "is_user_manual", False),
                "origin": "USER (MT5)" if getattr(p, "is_user_manual", False) else "BOT (AUTO)",
            })
    else:
        positions = ctx.trade_manager.get_managed_positions() if ctx.trade_manager else []
    daily_stats = ctx.trade_manager.get_daily_status() if ctx.trade_manager else {}
    cb_tripped, cb_reason = ctx.trade_manager.is_circuit_breaker_tripped() if ctx.trade_manager else (False, "")
    cal_state = ctx.news_reactor.check_calendar_state() if ctx.news_reactor else {}
    news_full = ctx.news_reactor.get_complete_news_state() if ctx.news_reactor else {}
    auto_brain = ctx.scorer.get_auto_brain_status() if ctx.scorer else {}

    raw_pending = getattr(ctx, "mt5_pending_orders", [])
    if not raw_pending and ctx.trade_manager and hasattr(ctx.trade_manager, "_pending_limit_orders"):
        raw_pending = [o.to_dict() for o in ctx.trade_manager._pending_limit_orders.values()]

    pending_orders = []
    for o in raw_pending:
        if isinstance(o, dict):
            t = str(o.get("type") or o.get("order_type") or "PENDING").upper()
            pending_orders.append({
                "ticket": int(o["ticket"]) if str(o.get("ticket", "")).isdigit() else o.get("ticket"),
                "symbol": o.get("symbol", ctx.active_symbol),
                "type": t,
                "order_type": t,
                "volume": float(o.get("volume", 0.01)),
                "price": float(o.get("price", 0.0)),
                "sl": float(o.get("sl", 0.0)),
                "tp": float(o.get("tp", 0.0)),
                "comment": str(o.get("comment", "")),
            })

    return {
        "active_symbol": ctx.active_symbol,
        "execution_mode": ctx.execution_mode,
        "strategy_mode": ctx.scorer.mode if ctx.scorer else ctx.strategy_mode,
        "aggressiveness_pct": ctx.scorer.get_active_aggressiveness_pct() if ctx.scorer else ctx.aggressiveness_pct,
        "threshold": ctx.scorer.get_active_threshold() if ctx.scorer else 68,
        "auto_brain": auto_brain,
        "is_bridge_connected": is_bridge_connected,
        "is_mt5_live": is_live,
        "live_trading_active": is_live,
        "live_trading_enabled": ctx.live_trading_enabled,
        "is_paper_mode": not is_live,
        "broker_type": ctx.broker.__class__.__name__ if ctx.broker else "None",
        "account_type": getattr(ctx.risk_engine, "account_type", "micro") if ctx.risk_engine else "micro",
        "risk_profile": getattr(ctx.risk_engine, "risk_profile", "balanced") if ctx.risk_engine else "balanced",
        "risk_profile_info": ctx.risk_engine.get_risk_profile_info() if ctx.risk_engine else {},
        "manage_manual_positions": ctx.manage_manual_positions,
        "circuit_breaker_enabled": getattr(ctx.trade_manager, "circuit_breaker_enabled", False) if ctx.trade_manager else False,
        "circuit_breaker_tripped": cb_tripped,
        "circuit_breaker_reason": cb_reason,
        "mt5_algo_trading_allowed": getattr(ctx, "mt5_algo_trading_allowed", True),
        "broker_connected": True if (is_bridge_connected or (ctx.broker and ctx.broker.is_connected())) else False,
        "broker_name": broker_name,
        "account": acc.to_dict() if acc else {},
        "tick": tick.to_dict() if tick else {},
        "positions": positions,
        "pending_orders": pending_orders,
        "daily_stats": daily_stats,
        "news": cal_state,
        "news_state": news_full,
        "tf_trends": auto_brain.get("tf_trends", {"1H": "NEUTRAL", "15m": "NEUTRAL", "5m": "NEUTRAL", "1m": "NEUTRAL"}),
        "feed_logs": ctx.news_reactor.get_feed_logs(15) if ctx.news_reactor else [],
        "allow_pre_news_trajectory": getattr(ctx, "allow_pre_news_trajectory", True),
        "auto_limits_enabled": getattr(ctx, "auto_limits_enabled", False),
        "last_mt5_error": getattr(ctx, "last_mt5_error", ""),
        "planned_limits": getattr(ctx, "planned_limits", {}),
        "predictive_brain": ctx.scorer.predictive_brain.get_state() if (ctx.scorer and hasattr(ctx.scorer, "predictive_brain")) else {},
        "pre_news_trajectory": ctx.news_reactor.evaluate_pre_news_trajectory(
            symbol=ctx.active_symbol,
            dfs={
                tf: (ctx.data_hub.get_df(ctx.active_symbol, tf) if (ctx.data_hub and hasattr(ctx.data_hub, "get_df")) else None)
                for tf in ["1m", "5m", "15m", "1H"]
            },
            tick_velocity=getattr(ctx.news_reactor, "_latest_velocity", 0.0),
        ) if (ctx.news_reactor and hasattr(ctx.news_reactor, "evaluate_pre_news_trajectory")) else {},
    }


@app.get("/api/predictive_brain")
async def get_predictive_brain():
    """Return real-time quantitative predictive model thinking state."""
    if ctx.scorer and hasattr(ctx.scorer, "predictive_brain"):
        return ctx.scorer.predictive_brain.get_state()
    return {}


@app.get("/api/news")
async def get_news_state():
    """Return comprehensive timezone-aware economic calendar and breaking headlines."""
    if not ctx.news_reactor:
        return {}
    return ctx.news_reactor.get_complete_news_state()


@app.get("/api/candles")
async def get_candles(symbol: str = "XAUUSD", timeframe: str = "1m", count: int = 800):
    """Fetch OHLCV candles formatted for TradingView Lightweight Charts."""
    if not ctx.data_hub:
        return []
    target_sym = symbol or ctx.active_symbol or "XAUUSD"
    bars = await ctx.data_hub.get_bars(target_sym, timeframe, count=count)
    if not bars or len(bars) < 50:
        alt_sym = "XAUUSD" if target_sym != "XAUUSD" else (ctx.active_symbol or "XAUUSD.m")
        alt_bars = await ctx.data_hub.get_bars(alt_sym, timeframe, count=count)
        if alt_bars and len(alt_bars) > len(bars):
            bars = alt_bars
    if (not bars or len(bars) < 20) and ctx.broker:
        rates = await ctx.broker.get_rates(target_sym, timeframe, count=count)
        if rates:
            asyncio.create_task(ctx.data_hub.warmup(ctx.broker, [target_sym]))
            bars = rates[-count:] if len(rates) >= count else rates
    return [
        {
            "time": b.time,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
        }
        for b in bars
    ]


class SettingsPayload(BaseModel):
    execution_mode: Optional[str] = None
    strategy_mode: Optional[str] = None
    aggressiveness_pct: Optional[float] = None
    active_symbol: Optional[str] = None
    budget_usd: Optional[float] = None
    risk_pct: Optional[float] = None
    account_type: Optional[str] = None
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    telegram_enabled: Optional[bool] = None
    auto_limits_enabled: Optional[bool] = None


@app.post("/api/settings")
async def update_settings(payload: SettingsPayload):
    """Dynamically modify runtime settings without restarting server."""
    if payload.execution_mode:
        ctx.execution_mode = payload.execution_mode.lower()
    
    if payload.auto_limits_enabled is not None:
        ctx.auto_limits_enabled = bool(payload.auto_limits_enabled)

    if payload.strategy_mode or payload.aggressiveness_pct is not None:
        if payload.strategy_mode:
            target_mode = payload.strategy_mode.lower()
        elif payload.aggressiveness_pct is not None:
            target_mode = "manual"
        else:
            target_mode = ctx.scorer.mode if ctx.scorer else ctx.strategy_mode

        ctx.strategy_mode = target_mode
        if payload.aggressiveness_pct is not None:
            ctx.aggressiveness_pct = float(payload.aggressiveness_pct)
        if ctx.scorer:
            ctx.scorer.set_mode(target_mode, pct=payload.aggressiveness_pct)

    if payload.active_symbol:
        ctx.active_symbol = payload.active_symbol.upper()
    if payload.budget_usd and ctx.risk_engine:
        ctx.risk_engine.base_budget_usd = payload.budget_usd
    if payload.risk_pct and ctx.risk_engine:
        ctx.risk_engine.risk_per_trade_pct = payload.risk_pct
    if payload.account_type and ctx.risk_engine:
        ctx.risk_engine.account_type = payload.account_type.lower()
    if payload.telegram_token is not None and ctx.telegram:
        ctx.telegram.configure(
            bot_token=payload.telegram_token,
            chat_id=payload.telegram_chat_id or ctx.telegram.chat_id,
            enabled=payload.telegram_enabled if payload.telegram_enabled is not None else ctx.telegram.enabled,
        )
    return {
        "status": "OK",
        "message": "Settings updated successfully.",
        "strategy_mode": ctx.scorer.mode if ctx.scorer else ctx.strategy_mode,
        "aggressiveness_pct": ctx.scorer.get_active_aggressiveness_pct() if ctx.scorer else ctx.aggressiveness_pct,
        "threshold": ctx.scorer.get_active_threshold() if ctx.scorer else 68,
    }


@app.post("/api/actions/close_all")
async def action_close_all():
    """1-Click Emergency Button: Close all active positions."""
    if not await endpoint_rate_limiter.check("close_all", min_interval=0.5):
        return {"status": "REJECTED", "message": "Emergency close all debounced. Please wait before retrying."}
    if not ctx.trade_manager:
        raise HTTPException(status_code=500, detail="TradeManager not ready")
    closed = await ctx.trade_manager.emergency_close_all()
    if ctx.broker and hasattr(ctx.broker, "close_all_positions"):
        await ctx.broker.close_all_positions()
    return {"status": "OK", "closed_count": closed}


@app.post("/api/actions/lock_breakeven")
async def action_lock_breakeven():
    """1-Click Emergency Action: Move SL to Breakeven on all winning positions."""
    if not await endpoint_rate_limiter.check("lock_be", min_interval=1.0):
        return {"status": "REJECTED", "message": "Lock breakeven debounced. Please wait before retrying."}
    if not ctx.trade_manager:
        raise HTTPException(status_code=500, detail="TradeManager not ready")
    count = await ctx.trade_manager.emergency_lock_breakeven_all()
    return {"status": "OK", "modified_count": count}


@app.post("/api/actions/pause")
async def action_pause(minutes: int = 60):
    """Pause auto-execution for specified duration."""
    if not ctx.trade_manager:
        raise HTTPException(status_code=500, detail="TradeManager not ready")
    ctx.trade_manager.pause_trading(minutes=minutes)
    return {"status": "OK", "message": f"Paused for {minutes} minutes."}


@app.post("/api/actions/resume")
async def action_resume():
    """Resume automated trade executions."""
    if not ctx.trade_manager:
        raise HTTPException(status_code=500, detail="TradeManager not ready")
    ok, msg = await ctx.trade_manager.resume_trading(force=True)
    return {"status": "OK" if ok else "BLOCKED", "message": msg}


@app.post("/api/actions/reset_cooldown")
async def action_reset_cooldown():
    """Instantly clear post-loss cooldown and circuit breaker."""
    if ctx.trade_manager:
        ctx.trade_manager.reset_cooldown()
        await ctx.trade_manager.resume_trading(force=True)
    return {"status": "OK", "message": "Anti-revenge cooldown and circuit breaker cleared."}


@app.post("/api/actions/toggle_pre_news_trajectory")
async def action_toggle_pre_news_trajectory():
    """Toggle smart pre-news trajectory front-running on/off."""
    current = getattr(ctx, "allow_pre_news_trajectory", True)
    ctx.allow_pre_news_trajectory = not current
    if ctx.scorer:
        ctx.scorer.allow_pre_news_trajectory = ctx.allow_pre_news_trajectory
    state_str = "ENABLED" if ctx.allow_pre_news_trajectory else "DISABLED"
    logger.info(f"[PRE-NEWS ENGINE] Smart Pre-News Trajectory is now {state_str}.")
    return {
        "status": "OK",
        "allow_pre_news_trajectory": ctx.allow_pre_news_trajectory,
        "message": f"Pre-news trajectory engine {state_str}.",
    }


@app.post("/api/actions/risk_profile")
async def action_set_risk_profile(payload: Dict[str, Any]):
    """Switch active risk profile (conservative, balanced, aggressive)."""
    profile = str(payload.get("profile", "")).lower()
    if ctx.risk_engine:
        ctx.risk_engine.set_risk_profile(profile)
        return {
            "status": "OK",
            "profile": ctx.risk_engine.risk_profile,
            "info": ctx.risk_engine.get_risk_profile_info(),
        }
    raise HTTPException(status_code=500, detail="Risk engine not ready")


@app.get("/api/history")
async def get_trade_history(limit: int = 50):
    """Retrieve persistent completed trade history from SQLite."""
    if not ctx.history_store:
        from engine.trade_history_store import TradeHistoryStore
        ctx.history_store = TradeHistoryStore()
        await ctx.history_store.initialize()
    return await ctx.history_store.get_history(limit=limit)


@app.get("/api/stats")
async def get_trade_stats():
    """Retrieve persistent lifetime and daily win rates and performance analytics."""
    if not ctx.history_store:
        from engine.trade_history_store import TradeHistoryStore
        ctx.history_store = TradeHistoryStore()
        await ctx.history_store.initialize()
    return await ctx.history_store.get_stats()


@app.post("/api/actions/clear_history")
async def action_clear_history():
    """Purge all recorded trade history and reset win rate / P&L statistics."""
    if not ctx.history_store:
        from engine.trade_history_store import TradeHistoryStore
        ctx.history_store = TradeHistoryStore()
        await ctx.history_store.initialize()
    broker_t = getattr(ctx, "latest_broker_time", None)
    fresh_stats = await ctx.history_store.clear_history(broker_time=broker_t)
    if ctx.trade_manager:
        ctx.trade_manager._daily_losses_count = 0
        ctx.trade_manager._daily_realized_pnl = 0.0
    return {"status": "OK", "message": "History cleared and win rate reset to 0%", "stats": fresh_stats}


class DeleteTradePayload(BaseModel):
    ticket: int


@app.post("/api/actions/delete_trade")
async def action_delete_trade(payload: DeleteTradePayload):
    """Permanently delete an individual trade from history and mark ticket as purged."""
    if not ctx.history_store:
        from engine.trade_history_store import TradeHistoryStore
        ctx.history_store = TradeHistoryStore()
        await ctx.history_store.initialize()
    ticket = int(payload.ticket)
    await ctx.history_store.delete_trade(ticket)
    fresh_stats = await ctx.history_store.get_stats(force_refresh=True)
    return {"status": "OK", "ticket": ticket, "stats": fresh_stats}


@app.post("/api/actions/close_position")
async def action_close_position(ticket: int):
    """1-Click Emergency Action: Close a specific active position by ticket."""
    if not await endpoint_rate_limiter.check(f"close_pos_{ticket}", min_interval=0.5):
        return {"status": "REJECTED", "message": f"Close request for position #{ticket} debounced. Please wait."}
    if not ctx.broker:
        raise HTTPException(status_code=500, detail="Broker not ready")
    success = await ctx.broker.close_position(ticket)
    # Re-query broker state to verify actual closure (only on synchronous brokers)
    if not getattr(ctx.broker, "_use_http_queue", False):
        open_positions = await ctx.broker.get_open_positions()
        still_open = any((int(p.ticket) if str(p.ticket).isdigit() else p.ticket) == ticket for p in open_positions)
        if not success or still_open:
            if ctx.trade_manager:
                ctx.trade_manager._contradictory_state_active = True
                ctx.trade_manager._contradictory_state_reason = f"CRITICAL: Emergency close failed for position #{ticket}! Still open on broker."
                ctx.trade_manager.pause_trading(minutes=1440)
                if ctx.history_store:
                    await ctx.history_store.record_audit_event("CRITICAL_CLOSE_FAILED", ctx.trade_manager._contradictory_state_reason)
            return {"status": "ERROR", "message": f"Broker failed to close position #{ticket}", "ticket": ticket}
    if ctx.trade_manager and hasattr(ctx.trade_manager, "_managed_positions"):
        ctx.trade_manager._managed_positions.pop(ticket, None)
    return {"status": "OK", "ticket": ticket}


@app.post("/api/actions/cancel_limit")
async def action_cancel_limit(ticket: int):
    """1-Click Action: Cancel a pending Limit Order."""
    if not await endpoint_rate_limiter.check(f"cancel_limit_{ticket}", min_interval=0.5):
        return {"status": "REJECTED", "message": f"Cancel order #{ticket} debounced. Please wait."}
    if not ctx.broker:
        raise HTTPException(status_code=500, detail="Broker not ready")
    success = await ctx.broker.cancel_order(ticket)
    if ctx.trade_manager and hasattr(ctx.trade_manager, "_pending_limit_orders"):
        ctx.trade_manager._pending_limit_orders.pop(ticket, None)
    if ctx.is_live_trading_active() and ctx.mt5_pending_actions is not None:
        ctx.mt5_pending_actions.append({"action": "ORDER_CANCEL", "ticket": ticket})
    if not success:
        return {"status": "ERROR", "message": f"Broker failed to cancel order #{ticket}", "ticket": ticket}
    return {"status": "OK", "ticket": ticket}


@app.post("/api/actions/cancel_all_limits")
async def action_cancel_all_limits():
    """1-Click Action: Cancel all active pending Limit Orders."""
    if not await endpoint_rate_limiter.check("cancel_all_limits", min_interval=1.0):
        return {"status": "REJECTED", "message": "Cancel all limits debounced. Please wait."}
    count = 0
    if ctx.trade_manager and hasattr(ctx.trade_manager, "_pending_limit_orders"):
        for t in list(ctx.trade_manager._pending_limit_orders.keys()):
            if ctx.broker:
                await ctx.broker.cancel_order(t)
            ctx.trade_manager._pending_limit_orders.pop(t, None)
            count += 1
    if hasattr(ctx, "mt5_pending_orders") and ctx.mt5_pending_orders:
        for o in ctx.mt5_pending_orders:
            t = o.get("ticket")
            if t and ctx.is_live_trading_active() and ctx.mt5_pending_actions is not None:
                ctx.mt5_pending_actions.append({"action": "ORDER_CANCEL", "ticket": int(t)})
                count += 1
        ctx.mt5_pending_orders = []
    return {"status": "OK", "canceled_count": count, "cancelled_count": count}


class ModifyPositionPayload(BaseModel):
    ticket: int
    sl: float
    tp: float


@app.post("/api/actions/modify_position")
async def action_modify_position(payload: ModifyPositionPayload):
    """1-Click Action: Dynamically modify Stop Loss and Take Profit for an active position."""
    ticket = int(payload.ticket)
    sl = round(float(payload.sl), 2)
    tp = round(float(payload.tp), 2)

    # 1. Debounce rapid button clicks per ticket
    if not await endpoint_rate_limiter.check(f"modify_pos_{ticket}", min_interval=0.5):
        return {
            "status": "REJECTED",
            "message": f"Modify position #{ticket} debounced. Please wait before retrying.",
            "ticket": ticket
        }

    # 2. Locate active position across broker and trade manager
    pos = None
    if ctx.broker and hasattr(ctx.broker, "_positions") and ticket in ctx.broker._positions:
        pos = ctx.broker._positions[ticket]
    elif ctx.trade_manager and hasattr(ctx.trade_manager, "_managed_positions") and ticket in ctx.trade_manager._managed_positions:
        pos = ctx.trade_manager._managed_positions[ticket]

    if pos is None:
        return {
            "status": "REJECTED",
            "message": f"Position #{ticket} not found in active positions.",
            "ticket": ticket
        }

    # 3. Protect manually opened positions if manage_manual_positions is False
    if not ctx.manage_manual_positions:
        is_manual = getattr(pos, "is_user_manual", False)
        if is_manual:
            logger.warning(f"[MODIFY REJECTED] Manual position #{ticket} modification blocked because manage_manual_positions is False.")
            return {
                "status": "REJECTED",
                "message": f"Modification of manual position #{ticket} is blocked by safety policy (manage_manual_positions=False).",
                "ticket": ticket,
            }

    # 4. Validate SL/TP constraints against broker stop levels and protective ratchet rule
    cur_p = pos.current_price if (hasattr(pos, "current_price") and pos.current_price and pos.current_price > 0) else pos.open_price
    p_dir = getattr(pos, "direction", None)
    if not p_dir:
        p_dir = pos.type.value if hasattr(pos.type, "value") else str(pos.type).upper()
    else:
        p_dir = p_dir.upper()

    sym_info = await ctx.broker.get_symbol_info(pos.symbol) if ctx.broker else None
    min_dist = (getattr(sym_info, "stops_level", 0) * sym_info.point) if (sym_info and getattr(sym_info, "stops_level", 0) > 0) else (0.20 if "XAU" in pos.symbol.upper() else 0.0002)

    if "BUY" in p_dir:
        if sl > 0:
            if sl >= cur_p:
                return {"status": "REJECTED", "message": f"SL ({sl}) for BUY must be below current price ({cur_p}).", "ticket": ticket}
            if (cur_p - sl) < min_dist:
                return {"status": "REJECTED", "message": f"SL distance ({cur_p - sl:.2f}) is below broker min stop distance ({min_dist:.2f}).", "ticket": ticket}
            if pos.sl > 0 and sl < pos.sl:
                return {"status": "REJECTED", "message": f"Cannot loosen protective stop loss (New SL {sl} < current SL {pos.sl}).", "ticket": ticket}
        if tp > 0:
            if tp <= cur_p:
                return {"status": "REJECTED", "message": f"TP ({tp}) for BUY must be above current price ({cur_p}).", "ticket": ticket}
            if (tp - cur_p) < min_dist:
                return {"status": "REJECTED", "message": f"TP distance ({tp - cur_p:.2f}) is below broker min stop distance ({min_dist:.2f}).", "ticket": ticket}
    else:  # SELL
        if sl > 0:
            if sl <= cur_p:
                return {"status": "REJECTED", "message": f"SL ({sl}) for SELL must be above current price ({cur_p}).", "ticket": ticket}
            if (sl - cur_p) < min_dist:
                return {"status": "REJECTED", "message": f"SL distance ({sl - cur_p:.2f}) is below broker min stop distance ({min_dist:.2f}).", "ticket": ticket}
            if pos.sl > 0 and sl > pos.sl:
                return {"status": "REJECTED", "message": f"Cannot loosen protective stop loss (New SL {sl} > current SL {pos.sl}).", "ticket": ticket}
        if tp > 0:
            if tp >= cur_p:
                return {"status": "REJECTED", "message": f"TP ({tp}) for SELL must be below current price ({cur_p}).", "ticket": ticket}
            if (cur_p - tp) < min_dist:
                return {"status": "REJECTED", "message": f"TP distance ({cur_p - tp:.2f}) is below broker min stop distance ({min_dist:.2f}).", "ticket": ticket}

    logger.info(f"[ACTION] Modify position #{ticket}: SL={sl}, TP={tp}")

    # 5. Send modification to broker first
    broker_ok = False
    if ctx.broker:
        broker_ok = await ctx.broker.modify_position(ticket, sl=sl, tp=tp)

    # 6. Update local state ONLY upon successful confirmation
    if broker_ok:
        if ctx.trade_manager and hasattr(ctx.trade_manager, "_managed_positions"):
            if ticket in ctx.trade_manager._managed_positions:
                m_pos = ctx.trade_manager._managed_positions[ticket]
                m_pos.sl = sl
                m_pos.tp = tp
                m_pos.last_mod_time = time.time()
    else:
        return {"status": "ERROR", "message": f"Broker rejected position #{ticket} modification", "ticket": ticket}

    # 7. Queue command for live MT5 Bridge only if live trading is active
    if ctx.is_live_trading_active() and ctx.mt5_pending_actions is not None:
        ctx.mt5_pending_actions.append({
            "action": "ORDER_MODIFY",
            "ticket": ticket,
            "sl": sl,
            "tp": tp,
        })

    return {"status": "OK", "ticket": ticket, "sl": sl, "tp": tp}


class ModifyOrderPayload(BaseModel):
    ticket: int
    price: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None


@app.post("/api/actions/modify_order")
async def action_modify_order(payload: ModifyOrderPayload):
    """1-Click Action: Dynamically modify Price, SL, and TP for a pending order."""
    ticket = int(payload.ticket)
    price = round(float(payload.price), 2) if payload.price is not None else 0.0
    sl = round(float(payload.sl), 2) if payload.sl is not None else 0.0
    tp = round(float(payload.tp), 2) if payload.tp is not None else 0.0

    if not await endpoint_rate_limiter.check(f"modify_order_{ticket}", min_interval=0.5):
        return {"status": "REJECTED", "message": f"Modify order #{ticket} debounced. Please wait before retrying."}

    logger.info(f"[ACTION] Modify pending order #{ticket}: Price={price}, SL={sl}, TP={tp}")

    if ctx.trade_manager and hasattr(ctx.trade_manager, "_pending_limit_orders"):
        if ticket in ctx.trade_manager._pending_limit_orders:
            o = ctx.trade_manager._pending_limit_orders[ticket]
            if price > 0: o.price = price
            if sl > 0: o.sl = sl
            if tp > 0: o.tp = tp

    if ctx.is_live_trading_active() and ctx.mt5_pending_actions is not None:
        ctx.mt5_pending_actions.append({
            "action": "ORDER_MODIFY",
            "ticket": ticket,
            "price": price,
            "sl": sl,
            "tp": tp,
        })

    return {"status": "OK", "ticket": ticket, "price": price, "sl": sl, "tp": tp}


@app.get("/api/planned_limits")
async def get_planned_limits():
    """Return live institutional planned BUY_LIMIT and SELL_LIMIT calculations."""
    return getattr(ctx, "planned_limits", {})


class StageLimitPayload(BaseModel):
    direction: str = "BUY"


@app.post("/api/actions/stage_limit")
async def action_stage_limit(payload: StageLimitPayload):
    """1-Click Action: Manually stage the currently calculated institutional limit order."""
    dir_clean = payload.direction.upper().strip()
    limits = getattr(ctx, "planned_limits", {})
    key = "planned_buy" if dir_clean == "BUY" else "planned_sell"
    target_limit = limits.get(key)
    if not target_limit:
        return {"status": "ERROR", "message": f"No active planned {dir_clean} limit setup calculated"}
    if not ctx.trade_manager or not ctx.scorer:
        return {"status": "ERROR", "message": "Trade manager or scorer not ready"}
    res = await ctx.trade_manager.execute_planned_limit(target_limit, ctx.scorer)
    if res.success:
        return {"status": "OK", "ticket": res.ticket, "price": res.price, "message": f"Limit order #{res.ticket} staged successfully @ {res.price}"}
    return {"status": "ERROR", "message": res.error or "Failed to stage limit order"}


@app.post("/api/actions/toggle_auto_limits")
async def action_toggle_auto_limits():
    """Toggle autonomous limits engine on/off."""
    ctx.auto_limits_enabled = not getattr(ctx, "auto_limits_enabled", True)
    logger.info(f"[ACTION] Toggled autonomous limit staging to {ctx.auto_limits_enabled}")
    return {"status": "OK", "auto_limits_enabled": ctx.auto_limits_enabled}


@app.post("/api/backtest")
async def run_backtest(days: int = 30, mode: str = "auto_adaptive"):
    """Trigger 30-day historical multi-timeframe backtest on active symbol."""
    if not ctx.backtester or not ctx.data_hub:
        raise HTTPException(status_code=500, detail="Backtest engine not initialized")
    result = await ctx.backtester.run(data_hub=ctx.data_hub, symbol=ctx.active_symbol, days=days, mode=mode)
    return result.to_dict()


# ------------------------------------------------------------------------------
# macOS Wine MT5 WebRequest Bridge Endpoints
# ------------------------------------------------------------------------------
@app.post("/api/mt5_bridge/tick")
async def mt5_bridge_tick(payload: Dict[str, Any], request: Request = None):
    """
    Ultra-Fast Bridge Endpoint for MetaTrader 5 on macOS Wine.
    Receives live ticks, forming candles, and positions from ApexBridge.mq5 and
    immediately returns pending trade/modify/close orders in sub-2ms.
    """
    verify_bridge_token(request, payload)
    now_loop = asyncio.get_event_loop().time()
    was_disconnected = (now_loop - ctx.mt5_last_seen) > 10.0 and (ctx.mt5_last_seen > 0)
    ctx.mt5_last_seen = now_loop
    if was_disconnected and ctx.trade_manager:
        logger.info("[MT5 RECONNECTED] MetaTrader 5 bridge reconnected. Triggering broker state reconciliation...")
        asyncio.create_task(ctx.trade_manager.reconcile_broker_state())
    
    sym = payload.get("symbol", ctx.active_symbol)
    bid = float(payload.get("bid", 0.0))
    ask = float(payload.get("ask", 0.0))
    spread = float(payload.get("spread", round(ask - bid, 2)))
    balance = float(payload.get("balance", 0.0))
    equity = float(payload.get("equity", 0.0))
    margin = float(payload.get("margin", 0.0))
    free_margin = float(payload.get("free_margin", 0.0))
    leverage = int(payload.get("leverage", 500))
    positions_raw = payload.get("positions", [])
    cur_bar = payload.get("cur_bar")

    ctx.mt5_algo_trading_allowed = bool(payload.get("algo_trading_allowed", True))
    
    last_cmd_result = payload.get("last_cmd_result")
    if last_cmd_result and isinstance(last_cmd_result, dict):
        if last_cmd_result.get("status") == "ERROR":
            err = last_cmd_result.get("error", "Unknown MT5 Error")
            ctx.last_mt5_error = err
            logger.error(f"[MT5 BRIDGE EXECUTION ERROR] {err}")
        elif last_cmd_result.get("status") == "OK":
            ticket = last_cmd_result.get("ticket")
            ctx.last_mt5_error = ""
            logger.info(f"[MT5 BRIDGE EXECUTION SUCCESS] Order #{ticket} filled on MT5")

    if sym and sym != ctx.active_symbol:
        ctx.active_symbol = sym

    if ctx.broker:
        ctx.broker._active_broker_symbol = sym

    if ctx.broker and bid > 0 and ask > 0:
        from brokers.base import Tick
        now_ts = float(payload.get("time", time.time()))
        if now_ts < 1000000000:
            now_ts = time.time()
        else:
            ctx.latest_broker_time = max(getattr(ctx, "latest_broker_time", 0.0), now_ts)

        tick_obj = Tick(
            symbol=sym,
            bid=bid,
            ask=ask,
            time=now_ts,
            spread=spread,
        )
        ctx.broker._last_ticks[sym] = tick_obj
        ctx.broker._last_ticks[ctx.active_symbol] = tick_obj
        ctx.broker._last_ticks["XAUUSD"] = tick_obj

        if hasattr(ctx.broker, "_base_prices"):
            mid = (bid + ask) / 2.0
            ctx.broker._base_prices[sym] = mid
            ctx.broker._base_prices["XAUUSD"] = mid
            ctx.broker._base_prices[ctx.active_symbol] = mid

        if balance > 0:
            if hasattr(ctx.broker, "balance"):
                ctx.broker.balance = balance
            if ctx.risk_engine:
                ctx.risk_engine.base_budget_usd = balance
            if ctx.trade_manager:
                if ctx.trade_manager._start_of_day_balance <= 1.0:
                    ctx.trade_manager._start_of_day_balance = max(1.0, balance - ctx.trade_manager._daily_realized_pnl)
        if equity > 0 and hasattr(ctx.broker, "equity"):
            ctx.broker.equity = equity
        if hasattr(ctx.broker, "_override_free_margin") and free_margin > 0:
            ctx.broker._override_free_margin = free_margin
            ctx.broker._override_margin = margin
        if leverage > 0 and hasattr(ctx.broker, "leverage"):
            ctx.broker.leverage = leverage

    # Synchronize open positions with MT5
    if isinstance(positions_raw, list):
        if len(positions_raw) == 0:
            # MT5 has no open positions: check if any previously open position closed in loss
            if ctx.trade_manager and hasattr(ctx.trade_manager, "_managed_positions"):
                for m in list(ctx.trade_manager._managed_positions.values()):
                    if getattr(m, "profit", 0.0) < -0.05:
                        now_ts = time.time()
                        ctx.trade_manager._daily_losses_count += 1
                        ctx.trade_manager._post_loss_cooldown_until = now_ts + 15.0
                        ctx.trade_manager._last_order_time = now_ts + 10.0
                        ctx.trade_manager._last_loss_direction = getattr(m, "direction", None)
                        ctx.trade_manager._last_loss_price = getattr(m, "open_price", 0.0)
                        ctx.trade_manager._last_loss_time = now_ts
                        logger.warning(
                            f"[MT5 SL HIT DETECTED] Position #{m.ticket} closed with loss (${m.profit:.2f}). "
                            f"Post-loss buffer active for 15s. Directional anti-revenge set for {getattr(m, 'direction', '')}."
                        )
                ctx.trade_manager._managed_positions.clear()
            if hasattr(ctx.broker, "_positions"):
                ctx.broker._positions.clear()
        else:
            from brokers.base import Position, PositionType
            synced_positions = {}
            for p in positions_raw:
                raw_ticket = p.get("ticket")
                ticket = int(raw_ticket) if str(raw_ticket).isdigit() else raw_ticket
                p_sym = p.get("symbol", sym)
                raw_t = str(p.get("type", "")).upper().strip()
                p_type = PositionType.BUY if raw_t in ("BUY", "0", "POSITION_TYPE_BUY") else PositionType.SELL
                p_vol = float(p.get("volume", 0.01))
                p_open = float(p.get("open_price", bid))
                p_curr = float(p.get("current_price", bid))
                p_sl = float(p.get("sl", 0.0))
                p_tp = float(p.get("tp", 0.0))
                p_pnl = float(p.get("profit", 0.0))
                p_comm = str(p.get("comment", ""))
                p_magic = int(p.get("magic", 0)) if str(p.get("magic", "")).isdigit() else 0
                is_bot_trade = (p_magic == 15555) or p_comm.startswith("ApexGold") or p_comm.startswith("Chameleon")
                is_manual = not is_bot_trade

                pos_obj = Position(
                    ticket=ticket,
                    symbol=p_sym,
                    type=p_type,
                    volume=p_vol,
                    open_price=p_open,
                    current_price=p_curr,
                    sl=p_sl,
                    tp=p_tp,
                    profit=p_pnl,
                    comment=p_comm,
                    magic=p_magic,
                    is_user_manual=is_manual,
                )
                synced_positions[ticket] = pos_obj

            if hasattr(ctx.broker, "_positions"):
                ctx.broker._positions = synced_positions

            # Auto-enroll real MT5 positions into TradeManager if permitted
            if ctx.trade_manager:
                from execution.trade_manager import ManagedPosition
                for t_key, pos_obj in synced_positions.items():
                    if pos_obj.is_user_manual and not ctx.manage_manual_positions:
                        # Strictly protect manual MT5 trades: do NOT enroll into automated position management!
                        logger.debug(f"[MANUAL POSITION PROTECTED] #{t_key} opened manually by user. Skipping automated management.")
                        continue

                    if t_key not in ctx.trade_manager._managed_positions:
                        dist = abs(pos_obj.open_price - pos_obj.sl) if pos_obj.sl > 0 else 2.50
                        ctx.trade_manager._managed_positions[t_key] = ManagedPosition(
                            ticket=t_key,
                            symbol=pos_obj.symbol,
                            direction=pos_obj.type.value if hasattr(pos_obj.type, "value") else str(pos_obj.type),
                            volume=pos_obj.volume,
                            open_price=pos_obj.open_price,
                            sl=pos_obj.sl,
                            tp=pos_obj.tp,
                            entry_time=time.time(),
                            risk_r_dist=dist,
                            initial_regime="ADAPTIVE_MANAGED",
                            current_regime="ADAPTIVE_MANAGED",
                            sub_regime="WAVE_RUNNER",
                            stall_exit_seconds=60.0,
                            profit=pos_obj.profit,
                            current_price=pos_obj.current_price,
                            origin="USER (MT5)" if pos_obj.is_user_manual else "BOT (AUTO)",
                            is_user_manual=pos_obj.is_user_manual,
                        )
                    else:
                        m_pos = ctx.trade_manager._managed_positions[t_key]
                        if m_pos.is_user_manual and not ctx.manage_manual_positions:
                            continue
                        if pos_obj.sl > 0:
                            m_pos.sl = pos_obj.sl
                        if pos_obj.tp > 0:
                            m_pos.tp = pos_obj.tp
                        m_pos.profit = pos_obj.profit
                        m_pos.current_price = pos_obj.current_price

                # Purge closed MT5 positions from _managed_positions
                stale_keys = [
                    k for k in ctx.trade_manager._managed_positions.keys()
                    if str(k).isdigit() and int(k) > 200000 and int(k) not in synced_positions
                ]
                for sk in stale_keys:
                    closed_m = ctx.trade_manager._managed_positions.pop(sk, None)
                    if closed_m and getattr(closed_m, "profit", 0.0) < -0.05:
                        now_ts = time.time()
                        ctx.trade_manager._daily_losses_count += 1
                        ctx.trade_manager._post_loss_cooldown_until = now_ts + 15.0
                        ctx.trade_manager._last_order_time = now_ts + 10.0
                        ctx.trade_manager._last_loss_direction = getattr(closed_m, "direction", None)
                        ctx.trade_manager._last_loss_price = getattr(closed_m, "open_price", 0.0)
                        ctx.trade_manager._last_loss_time = now_ts
                        logger.warning(
                            f"[MT5 SL HIT DETECTED] Position #{closed_m.ticket} closed with loss (${closed_m.profit:.2f}). "
                            f"Post-loss buffer active for 15s. Directional anti-revenge set for {getattr(closed_m, 'direction', '')}."
                        )

    # Ingest real closed deals directly from MetaTrader 5 history
    closed_deals = payload.get("closed_deals")
    if closed_deals and isinstance(closed_deals, list) and ctx.history_store:
        for deal in closed_deals:
            try:
                # Prefer pos_id (position ticket) over deal ticket to avoid duplicate deal entries
                raw_d_ticket = deal.get("pos_id") or deal.get("ticket")
                if not raw_d_ticket:
                    continue
                d_ticket = int(raw_d_ticket) if str(raw_d_ticket).isdigit() else int(time.time())
                if hasattr(ctx.history_store, "_known_tickets") and d_ticket in ctx.history_store._known_tickets:
                    continue
                if hasattr(ctx.history_store, "_purged_tickets") and d_ticket in ctx.history_store._purged_tickets:
                    continue

                d_close_t = float(deal.get("close_time", time.time()))
                if d_close_t > 1000000000:
                    ctx.latest_broker_time = max(getattr(ctx, "latest_broker_time", 0.0), d_close_t)

                # Ignore deals that closed before history was cleared by the user
                if hasattr(ctx.history_store, "_cleared_cutoff_time") and d_close_t <= ctx.history_store._cleared_cutoff_time:
                    continue

                d_sym = str(deal.get("symbol", sym))
                d_dir = "BUY" if str(deal.get("direction", "")).upper() in ("BUY", "0") else "SELL"
                d_vol = float(deal.get("volume", 0.01))
                d_open = float(deal.get("open_price", 0.0))
                d_close = float(deal.get("close_price", 0.0))
                d_open_t = float(deal.get("open_time", 0.0))
                d_pnl = float(deal.get("profit", 0.0))
                d_comm = float(deal.get("commission", 0.0))
                d_swap = float(deal.get("swap", 0.0))
                net_pnl = round(d_pnl + d_comm + d_swap, 2)

                is_gold = any(k in d_sym.upper() for k in ["XAU", "GOLD"])
                if d_dir == "BUY":
                    dist = d_close - d_open
                else:
                    dist = d_open - d_close
                pnl_r = round(dist / 2.0, 2) if is_gold else round(dist / 0.0020, 2)

                comment = str(deal.get("comment", ""))
                if "[sl]" in comment.lower():
                    close_reason = "MT5 Broker Stop Loss Hit"
                elif "[tp]" in comment.lower():
                    close_reason = "MT5 Broker Take Profit Hit"
                elif net_pnl > 0:
                    close_reason = f"MT5 Realized Profit (+${net_pnl:.2f})"
                else:
                    close_reason = f"MT5 Realized Loss (-${abs(net_pnl):.2f})"

                asyncio.create_task(ctx.history_store.record_trade(
                    ticket=d_ticket,
                    symbol=d_sym,
                    direction=d_dir,
                    volume=d_vol,
                    open_price=d_open if d_open > 0 else d_close,
                    close_price=d_close,
                    open_time=d_open_t if d_open_t > 0 else (d_close_t - 60),
                    close_time=d_close_t,
                    pnl=net_pnl,
                    pnl_r=pnl_r,
                    regime="MT5_LIVE",
                    sub_regime="REAL_DEAL",
                    close_reason=close_reason,
                ))
            except Exception as e:
                logger.debug(f"Closed deal parse error: {e}")

    # Synchronize active pending limit/stop orders from MT5
    raw_orders = payload.get("orders")
    if raw_orders is not None and isinstance(raw_orders, list):
        ctx.mt5_pending_orders = raw_orders
    if cur_bar and isinstance(cur_bar, dict) and ctx.data_hub:
        from brokers.base import Bar
        b_time = int(cur_bar.get("time", 0))
        if b_time > 0:
            bar_obj = Bar(
                time=b_time,
                open=float(cur_bar.get("open", bid)),
                high=float(cur_bar.get("high", max(bid, ask))),
                low=float(cur_bar.get("low", min(bid, ask))),
                close=float(cur_bar.get("close", bid)),
                volume=float(cur_bar.get("volume", 1.0)),
            )
            asyncio.create_task(ctx.data_hub.append_bar(sym, "1m", bar_obj))
            if sym != "XAUUSD":
                asyncio.create_task(ctx.data_hub.append_bar("XAUUSD", "1m", bar_obj))

    # Synchronize 1:1 broker historical bars across all timeframes from MT5
    chart_tf = payload.get("chart_tf")
    if chart_tf:
        ctx.mt5_chart_tf = chart_tf

    tf_mappings = [
        ("rates_m1", "1m"),
        ("rates_m5", "5m"),
        ("rates_m15", "15m"),
        ("rates_h1", "1H"),
    ]
    for tf_key, tf_code in tf_mappings:
        raw_list = payload.get(tf_key) or (payload.get("rates", []) if tf_code == "1m" else [])
        if raw_list and isinstance(raw_list, list) and len(raw_list) > 5 and ctx.data_hub:
            from brokers.base import Bar
            p_bars = [
                Bar(
                    time=int(r[0]),
                    open=float(r[1]),
                    high=float(r[2]),
                    low=float(r[3]),
                    close=float(r[4]),
                    volume=float(r[5]) if len(r) > 5 else 1.0,
                )
                for r in raw_list
                if isinstance(r, (list, tuple)) and len(r) >= 5
            ]
            if p_bars:
                asyncio.create_task(ctx.data_hub.inject_bars(sym, tf_code, p_bars))
                if sym != "XAUUSD":
                    asyncio.create_task(ctx.data_hub.inject_bars("XAUUSD", tf_code, p_bars))
                if sym != ctx.active_symbol:
                    asyncio.create_task(ctx.data_hub.inject_bars(ctx.active_symbol, tf_code, p_bars))

    next_action = {"action": "NONE"}
    if not ctx.is_live_trading_active():
        # Hard separation: Paper mode or disabled live trading NEVER sends actions to MT5
        if ctx.mt5_pending_actions:
            ctx.mt5_pending_actions.clear()
        return next_action

    if ctx.mt5_pending_actions:
        next_action = ctx.mt5_pending_actions.pop(0)

    return next_action


@app.post("/api/actions/test_trade")
async def action_test_trade(direction: str = "BUY", volume: float = 0.01):
    """1-Click Test Button: Safely executes trade after debouncing, risk engine checks, and broker constraint validation."""
    # 1. Debounce rapid button clicks (1.0s interval)
    if not await endpoint_rate_limiter.check("test_trade", min_interval=1.0):
        return {"status": "REJECTED", "message": "Duplicate trade request debounced. Please wait before retrying."}

    direction_clean = direction.upper().strip()
    if direction_clean not in ("BUY", "SELL"):
        return {"status": "REJECTED", "message": f"Invalid order direction '{direction}'. Must be BUY or SELL."}

    if volume <= 0:
        return {"status": "REJECTED", "message": f"Volume must be positive, got {volume}."}

    sym = ctx.active_symbol or "XAUUSD"

    # 2. Check circuit breakers & exposure via TradeManager
    if ctx.trade_manager:
        can_open, cb_reason = await ctx.trade_manager.can_open_manual_trade(sym, direction_clean, volume)
        if not can_open:
            logger.warning(f"[TEST TRADE REJECTED] {cb_reason}")
            return {"status": "REJECTED", "message": f"Trade rejected by Risk Guard / Circuit Breaker: {cb_reason}"}

    # 3. Validate volume and broker constraints
    sym_info = await ctx.broker.get_symbol_info(sym) if ctx.broker else None
    if sym_info:
        if volume < sym_info.min_lot:
            return {"status": "REJECTED", "message": f"Volume {volume} is below broker minimum {sym_info.min_lot}."}
        if volume > sym_info.max_lot:
            return {"status": "REJECTED", "message": f"Volume {volume} exceeds broker maximum {sym_info.max_lot}."}

    tick = await ctx.broker.get_tick(sym) if ctx.broker else None
    cur_p = (tick.ask if direction_clean == "BUY" else tick.bid) if tick else 2650.0
    sl_dist = 2.50 if any(k in sym.upper() for k in ["XAU", "GOLD"]) else 0.0025
    sl = round(cur_p - sl_dist, 2) if direction_clean == "BUY" else round(cur_p + sl_dist, 2)
    tp = round(cur_p + (sl_dist * 2.0), 2) if direction_clean == "BUY" else round(cur_p - (sl_dist * 2.0), 2)

    # 4. Validate monetary risk against configured risk budget
    if ctx.risk_engine and ctx.broker:
        acc = await ctx.broker.get_account_info()
        if acc:
            is_small_acc = (acc.balance <= 35.0) or (getattr(ctx.risk_engine, "account_type", "") in ["micro", "cent"])
            if is_small_acc and volume <= (sym_info.min_lot if sym_info else 0.01):
                # Always allow the operator's direct 1-Click trade for the broker minimum lot
                pass
            else:
                budget = ctx.risk_engine.calculate_risk_budget(acc)
                monetary_risk = ctx.risk_engine.calculate_monetary_risk(sym_info, volume, abs(cur_p - sl))
                max_allowed = max(budget, (acc.balance * 0.95) if is_small_acc else budget)
                if monetary_risk > max_allowed:
                    return {
                        "status": "REJECTED",
                        "message": f"Monetary risk (${monetary_risk:.2f}) exceeds allowed budget (${max_allowed:.2f}).",
                    }

    # 5. Execution
    if not ctx.is_live_trading_active():
        # Execute safely in paper simulator without dispatching to MT5
        if ctx.broker:
            from brokers.base import OrderType
            order_type = OrderType.BUY if direction_clean == "BUY" else OrderType.SELL
            res = await ctx.broker.open_order(
                symbol=sym,
                order_type=order_type,
                volume=volume,
                price=cur_p,
                sl=sl,
                tp=tp,
                comment="Paper_TestTrade",
            )
            if res and res.ticket:
                return {
                    "status": "OK",
                    "message": f"Simulated {direction_clean} {volume} executed on {ctx.broker.__class__.__name__} (Live MT5 trading disabled)",
                    "ticket": res.ticket,
                }
            else:
                return {"status": "ERROR", "message": f"Paper trade execution failed: {getattr(res, 'comment', 'Unknown error')}"}
        return {"status": "ERROR", "message": "Broker not available"}

    order_cmd = {
        "action": "ORDER_SEND",
        "symbol": sym,
        "type": direction_clean,
        "volume": float(volume),
        "sl": sl,
        "tp": tp,
        "comment": "ApexGold_1Click",
    }
    ctx.mt5_pending_actions.append(order_cmd)
    logger.info(f"[1-CLICK ACTION] Queued MT5 Live Order: {order_cmd}")
    return {"status": "OK", "message": f"Dispatched {direction_clean} {volume} to MT5", "order": order_cmd}


@app.post("/api/mt5_bridge/rates")
async def mt5_bridge_rates(payload: Dict[str, Any], request: Request = None):
    """
    Ingest authentic historical OHLCV bars directly from MetaTrader 5.
    Synchronizes the Web Dashboard candles to match the MT5 broker 100%.
    """
    verify_bridge_token(request, payload)
    sym = payload.get("symbol", ctx.active_symbol)
    tf = payload.get("timeframe", "1m")
    bars_data = payload.get("bars", [])
    if not bars_data or not ctx.data_hub:
        return {"status": "OK", "count": 0}

    from brokers.base import Bar
    bars = [
        Bar(
            time=int(b["time"]),
            open=float(b["open"]),
            high=float(b["high"]),
            low=float(b["low"]),
            close=float(b["close"]),
            volume=float(b.get("volume", 1.0)),
        )
        for b in bars_data
        if "time" in b and "open" in b and "close" in b
    ]

    if bars:
        await ctx.data_hub.inject_bars(sym, tf, bars)
        if sym != "XAUUSD":
            await ctx.data_hub.inject_bars("XAUUSD", tf, bars)
        if sym != ctx.active_symbol:
            await ctx.data_hub.inject_bars(ctx.active_symbol, tf, bars)
        logger.info(f"Synchronized {len(bars)} {tf} broker bars from MT5 for {sym} (Latest: {bars[-1].close})")

    return {"status": "OK", "count": len(bars)}


# ------------------------------------------------------------------------------
# WebSocket Real-Time Broadcaster (4 FPS)
# ------------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_telemetry(websocket: WebSocket):
    """Sub-second telemetry stream delivering live charts, score gauge, and Chameleon state."""
    # Origin verification for CSRF / cross-site WebSocket hijacking protection
    origin = websocket.headers.get("origin")
    if origin:
        normalized_origin = origin.rstrip("/")
        allowed_normalized = [o.rstrip("/") for o in ALLOWED_ORIGINS]
        if normalized_origin not in allowed_normalized:
            logger.warning(f"[WEBSOCKET REJECTED] Connection rejected from unauthorized origin: {origin}")
            await websocket.close(code=1008, reason="Unauthorized origin")
            return

    # If live trading is active and a bridge token is configured, enforce token
    required = (getattr(ctx, "bridge_token", "") or os.environ.get("APEX_BRIDGE_TOKEN", "")).strip()
    is_trusted_origin = origin and (origin.rstrip("/") in [o.rstrip("/") for o in ALLOWED_ORIGINS])
    if ctx.is_live_trading_active() and required and not is_trusted_origin:
        token = websocket.query_params.get("token") or websocket.headers.get("x-bridge-token")
        if token != required:
            logger.warning("[WEBSOCKET REJECTED] Missing or invalid authentication token for live trading telemetry stream.")
            await websocket.close(code=1008, reason="Authentication required for live telemetry stream")
            return

    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


async def broadcast_loop():
    """Continuously broadcast 4 FPS telemetry state to all connected dashboard UIs."""
    while True:
        try:
            await asyncio.sleep(0.25)
            if not ws_manager.active_connections or not ctx.broker:
                continue

            # Assemble real-time telemetry packet
            tick = await ctx.broker.get_tick(ctx.active_symbol)
            is_bridge_connected = (asyncio.get_event_loop().time() - ctx.mt5_last_seen < 3.0)
            if hasattr(ctx.broker, "_positions") and ctx.broker._positions:
                positions = [
                    {
                        "ticket": p.ticket,
                        "symbol": p.symbol,
                        "direction": p.type.value if hasattr(p.type, "value") else str(p.type),
                        "volume": p.volume,
                        "open_price": p.open_price,
                        "current_price": p.current_price,
                        "sl": p.sl,
                        "tp": p.tp,
                        "profit": p.profit,
                        "ratchet_stage": 0,
                        "comment": getattr(p, "comment", ""),
                        "is_user_manual": getattr(p, "is_user_manual", False),
                        "origin": "USER (MT5)" if getattr(p, "is_user_manual", False) else "BOT (AUTO)",
                    }
                    for p in ctx.broker._positions.values()
                ]
            else:
                positions = ctx.trade_manager.get_managed_positions() if ctx.trade_manager else []
            acc = await ctx.broker.get_account_info()
            cal_state = ctx.news_reactor.check_calendar_state() if ctx.news_reactor else {}

            # Multi-timeframe trend status & live score
            dfs = await ctx.data_hub.get_multi_timeframe_data(ctx.active_symbol) if ctx.data_hub else {}

            # Update Smart Auto-Pilot Brain
            if ctx.scorer and dfs:
                sym_info = await ctx.broker.get_symbol_info(ctx.active_symbol)
                cur_spread = sym_info.spread if sym_info else 0.25
                max_spread = 0.80 if any(k in ctx.active_symbol.upper() for k in ["XAU", "GOLD"]) else 0.0003
                tick_vel = ctx.news_reactor._latest_velocity if ctx.news_reactor else 1.0

                ctx.scorer.evaluate_auto_brain(
                    dfs=dfs,
                    current_spread=cur_spread,
                    max_spread=max_spread,
                    tick_velocity=tick_vel,
                    news_reactor=ctx.news_reactor,
                    symbol=ctx.active_symbol,
                    trade_manager=ctx.trade_manager,
                )

            buy_sig, sell_sig = (None, None)
            if ctx.scorer and dfs:
                buy_sig, sell_sig = ctx.scorer.evaluate(
                    symbol=ctx.active_symbol,
                    dfs=dfs,
                    current_tick_bid=tick.bid,
                    current_tick_ask=tick.ask,
                    is_impulse_active=ctx.news_reactor._is_impulse_active if ctx.news_reactor else False,
                )

            active_sig = buy_sig or sell_sig
            score = active_sig.score if active_sig else 0
            tier = active_sig.tier if active_sig else "NONE"
            regime = active_sig.regime if active_sig else "WAITING"
            sub_regime = active_sig.sub_regime if active_sig else (ctx.scorer.auto_brain_state.get("sub_regime", "WAVE_RUNNER") if ctx.scorer else "WAVE_RUNNER")
            breakdown = active_sig.breakdown if active_sig else {"h1_macro": 0, "m15_wave": 0, "m5_setup": 0, "m1_trigger": 0}

            is_bridge_connected = (asyncio.get_event_loop().time() - ctx.mt5_last_seen < 3.0)
            is_live = ctx.is_live_trading_active() and is_bridge_connected
            if is_live:
                broker_name = "MT5 MAC WINE BRIDGE (LIVE)"
            elif is_bridge_connected:
                broker_name = "MT5 BRIDGE (SCANNER / PAPER MODE)"
            elif ctx.broker and getattr(ctx.broker, "is_paper", False):
                broker_name = "PAPER_SIM ACTIVE (SIMULATED)"
            elif ctx.broker and ctx.broker.is_connected():
                broker_name = f"{ctx.broker.__class__.__name__} (CONNECTED)"
            else:
                broker_name = "DISCONNECTED"

            cb_tripped, cb_reason = ctx.trade_manager.is_circuit_breaker_tripped() if ctx.trade_manager else (False, "")

            raw_pending = getattr(ctx, "mt5_pending_orders", [])
            if not raw_pending and ctx.trade_manager and hasattr(ctx.trade_manager, "_pending_limit_orders"):
                raw_pending = [o.to_dict() for o in ctx.trade_manager._pending_limit_orders.values()]

            pending_orders = []
            for o in raw_pending:
                if isinstance(o, dict):
                    t = str(o.get("type") or o.get("order_type") or "PENDING").upper()
                    pending_orders.append({
                        "ticket": int(o["ticket"]) if str(o.get("ticket", "")).isdigit() else o.get("ticket"),
                        "symbol": o.get("symbol", ctx.active_symbol),
                        "type": t,
                        "order_type": t,
                        "volume": float(o.get("volume", 0.01)),
                        "price": float(o.get("price", 0.0)),
                        "sl": float(o.get("sl", 0.0)),
                        "tp": float(o.get("tp", 0.0)),
                        "comment": str(o.get("comment", "")),
                    })

            packet = {
                "type": "TELEMETRY",
                "time": tick.time,
                "symbol": ctx.active_symbol,
                "bid": tick.bid,
                "ask": tick.ask,
                "spread": tick.spread,
                "broker_name": broker_name,
                "broker_connected": True if (is_bridge_connected or (ctx.broker and ctx.broker.is_connected())) else False,
                "is_bridge_connected": is_bridge_connected,
                "is_mt5_live": is_live,
                "live_trading_active": is_live,
                "live_trading_enabled": ctx.live_trading_enabled,
                "is_paper_mode": not is_live,
                "broker_type": ctx.broker.__class__.__name__ if ctx.broker else "None",
                "account_type": getattr(ctx.risk_engine, "account_type", "micro") if ctx.risk_engine else "micro",
                "manage_manual_positions": ctx.manage_manual_positions,
                "circuit_breaker_enabled": getattr(ctx.trade_manager, "circuit_breaker_enabled", False) if ctx.trade_manager else False,
                "circuit_breaker_tripped": cb_tripped,
                "circuit_breaker_reason": cb_reason,
                "mt5_algo_trading_allowed": getattr(ctx, "mt5_algo_trading_allowed", True),
                "chart_tf": getattr(ctx, "mt5_chart_tf", "1H"),
                "account": acc.to_dict(),
                "score": score,
                "tier": tier,
                "regime": regime,
                "sub_regime": sub_regime,
                "breakdown": breakdown,
                "signal": active_sig.to_dict() if active_sig else None,
                "positions": positions,
                "pending_orders": pending_orders,
                "stats": await ctx.history_store.get_stats() if ctx.history_store else {},
                "history": await ctx.history_store.get_history(25) if ctx.history_store else [],
                "strategy_mode": ctx.scorer.mode if ctx.scorer else ctx.strategy_mode,
                "aggressiveness_pct": ctx.scorer.get_active_aggressiveness_pct() if ctx.scorer else ctx.aggressiveness_pct,
                "threshold": ctx.scorer.get_active_threshold() if ctx.scorer else 68,
                "auto_brain": ctx.scorer.get_auto_brain_status() if ctx.scorer else {},
                "tf_trends": ctx.scorer.tf_trends if ctx.scorer else {"1H": "NEUTRAL", "15m": "NEUTRAL", "5m": "NEUTRAL", "1m": "NEUTRAL"},
                "news": cal_state,

                "news_state": ctx.news_reactor.get_complete_news_state() if ctx.news_reactor else {},
                "daily_stats": ctx.trade_manager.get_daily_status() if ctx.trade_manager else {},
                "feed_logs": ctx.news_reactor.get_feed_logs(10) if ctx.news_reactor else [],
                "planned_limits": getattr(ctx, "planned_limits", {}),
                "predictive_brain": ctx.scorer.predictive_brain.get_state() if (ctx.scorer and hasattr(ctx.scorer, "predictive_brain")) else {},
            }

            await ws_manager.broadcast(packet)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in broadcast loop: {e}")
            await asyncio.sleep(1.0)
