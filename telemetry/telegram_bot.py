"""
ApexGold Telegram Bot Telemetry & Notification Hook.

Sends real-time alerts for signals, Chameleon morphing, ratchet locks,
order fills, and circuit breaker events directly to your phone.
Configurable dynamically from the Web UI.
"""

import asyncio
import logging
from typing import Any, Dict, Optional
import aiohttp

logger = logging.getLogger("ApexGold.Telegram")


class TelegramNotifier:
    """
    Asynchronous Telegram Alert Broadcaster.
    Guaranteed non-blocking with automatic failure suppression.
    """

    def __init__(
        self,
        bot_token: str = "",
        chat_id: str = "",
        enabled: bool = False,
    ):
        self.bot_token = bot_token.strip()
        self.chat_id = chat_id.strip()
        self.enabled = enabled and bool(self.bot_token) and bool(self.chat_id)

    def configure(self, bot_token: str, chat_id: str, enabled: bool) -> None:
        """Update Telegram credentials on-the-fly from Web UI."""
        self.bot_token = bot_token.strip()
        self.chat_id = chat_id.strip()
        self.enabled = enabled and bool(self.bot_token) and bool(self.chat_id)
        logger.info(f"Telegram notifier updated: enabled={self.enabled}")

    async def send_message(self, text: str) -> bool:
        """Send formatted HTML/Markdown notification asynchronously."""
        if not self.enabled:
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=5) as resp:
                    if resp.status == 200:
                        return True
                    else:
                        body = await resp.text()
                        logger.warning(f"Telegram send failed [{resp.status}]: {body}")
                        return False
        except Exception as e:
            logger.warning(f"Telegram notification error: {e}")
            return False

    async def notify_signal(self, symbol: str, direction: str, score: int, tier: str, regime: str, entry: float, sl: float, tp: float) -> None:
        """Broadcast high-probability trade signal."""
        icon = "🟢 BUY" if direction.upper() == "BUY" else "🔴 SELL"
        regime_icon = "🚀 RUNNER" if "RUNNER" in regime else "⚡ SCALP"
        msg = (
            f"<b>🏆 ApexGold Signal Alert</b>\n"
            f"<b>Asset:</b> {symbol} | {icon}\n"
            f"<b>Confluence Score:</b> {score}/100 ({tier})\n"
            f"<b>Initial Regime:</b> {regime_icon} ({regime})\n"
            f"<b>Entry Price:</b> {entry:.2f}\n"
            f"<b>Stop Loss:</b> {sl:.2f}\n"
            f"<b>Take Profit:</b> {tp:.2f}\n"
        )
        asyncio.create_task(self.send_message(msg))

    async def notify_order_filled(self, ticket: int, symbol: str, direction: str, volume: float, price: float, sl: float, tp: float, regime: str) -> None:
        """Broadcast live order fill."""
        icon = "🟢" if direction.upper() == "BUY" else "🔴"
        msg = (
            f"<b>⚡ Order Executed #{ticket}</b>\n"
            f"{icon} <b>{direction} {volume} {symbol}</b> @ {price:.2f}\n"
            f"<b>Mode:</b> {regime}\n"
            f"<b>SL:</b> {sl:.2f} | <b>TP:</b> {tp:.2f}\n"
        )
        asyncio.create_task(self.send_message(msg))

    async def notify_morph(self, ticket: int, old_regime: str, new_regime: str, reason: str) -> None:
        """Broadcast Chameleon regime morphing."""
        msg = (
            f"<b>🦎 Chameleon Morph Alert!</b>\n"
            f"<b>Position:</b> #{ticket}\n"
            f"<b>Regime Change:</b> {old_regime} ➡️ <b>{new_regime}</b>\n"
            f"<b>Reason:</b> {reason}\n"
        )
        asyncio.create_task(self.send_message(msg))

    async def notify_trade_closed(self, ticket: int, symbol: str, pnl: float, reason: str, balance: float) -> None:
        """Broadcast position closed."""
        pnl_icon = "💰" if pnl >= 0 else "🛑"
        msg = (
            f"<b>{pnl_icon} Position Closed #{ticket}</b>\n"
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>Realized PnL:</b> {'+' if pnl >= 0 else ''}{pnl:.2f}\n"
            f"<b>Exit Reason:</b> {reason}\n"
            f"<b>New Account Balance:</b> {balance:.2f}\n"
        )
        asyncio.create_task(self.send_message(msg))

