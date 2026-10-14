"""
ApexGold Timezone-Aware Live News & Volatility Command Center.

Features:
1. Automatic Local Timezone Detection (e.g. Asia/Beirut UTC+03:00).
2. Full Economic Calendar Schedule:
   - 🔴 Major (High Impact): CPI, NFP, FOMC, Powell, PPI, Retail Sales
   - 🟠 Medium & 🟡 Minor: Consumer Sentiment, Trade Balance, Claims, Auctions
   - 🔥 "Happening Right Now" Live Release State Machine (T-2m to T+15m)
   - ⏳ Hero Countdown to Next Major News
   - 📅 Today's & Tomorrow's Chronological Timeline in Local Time
3. Live Breaking Financial Headlines Stream (15s poller) with Automated Gold Sentiment Scoring
   (🟢 BULLISH GOLD, 🔴 BEARISH GOLD, ⚪ NEUTRAL).
4. Sub-50ms Tick Velocity & Spread Blowout Shield.
"""

import asyncio
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
import email.utils
import json
import logging
import os
import re
import time
from typing import Any, Deque, Dict, List, Optional, Tuple
import urllib.request
import xml.etree.ElementTree as ET

import aiohttp
from dateutil import parser as dt_parser

from brokers.base import Tick

logger = logging.getLogger("ApexGold.NewsReactor")


def detect_system_timezone():
    """
    Detect the host machine's exact local IANA timezone and UTC offset.
    e.g., 'Asia/Beirut', UTC+03:00
    """
    iana_name = "UTC"
    # 1. Try reading /etc/localtime symlink on macOS/Linux
    try:
        if os.path.exists("/etc/localtime"):
            real_path = os.path.realpath("/etc/localtime")
            parts = real_path.split("zoneinfo/")
            if len(parts) > 1:
                iana_name = parts[1]
    except Exception:
        pass

    # 2. Get local datetime with system timezone
    now_local = datetime.now().astimezone()
    local_tz = now_local.tzinfo

    if iana_name == "UTC" and local_tz is not None:
        iana_name = str(local_tz)

    offset_str = now_local.strftime("%z")
    if offset_str:
        formatted_offset = f"UTC{offset_str[:3]}:{offset_str[3:]}"
    else:
        formatted_offset = "UTC+00:00"

    return iana_name, local_tz, formatted_offset


@dataclass
class CalendarEvent:
    """Full-featured economic calendar event."""
    id: str
    title: str
    currency: str
    impact: str              # "HIGH", "MEDIUM", "LOW"
    impact_badge: str        # "🔴 HIGH", "🟠 MED", "🟡 LOW"
    timestamp_utc: float
    local_time_hm: str       # e.g., "03:30 PM"
    local_time_hms: str      # e.g., "03:30:00 PM"
    local_date_str: str      # e.g., "2026-10-14"
    forecast: str = ""
    previous: str = ""
    actual: str = ""
    status: str = "UPCOMING" # "LIVE NOW", "UPCOMING", "PASSED"
    relative_time: str = ""  # e.g., "in 02h 45m"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BreakingHeadline:
    """Live breaking financial headline with Gold sentiment tag."""
    title: str
    source: str
    url: str
    timestamp_utc: float
    local_time_str: str      # e.g., "12:45:10 PM"
    relative_str: str        # e.g., "45s ago"
    sentiment: str           # "BULLISH_GOLD", "BEARISH_GOLD", "NEUTRAL"
    sentiment_badge: str     # "🟢 BULLISH GOLD", "🔴 BEARISH GOLD", "⚪ NEUTRAL"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NewsReactor:
    """
    Timezone-Aware Economic Calendar & Millisecond Volatility Command Center.
    """

    FOREX_FACTORY_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    
    # Financial RSS endpoints for 15-second breaking stream
    RSS_FEED_URLS = [
        ("CNBC Finance", "https://www.cnbc.com/id/10000664/device/rss/rss.html"),
        ("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
    ]

    # Keyword lists for automated Gold Sentiment Scoring
    BULLISH_GOLD_KEYWORDS = [
        "dollar drops", "dollar slides", "dollar falls", "dollar slumps", "weaker dollar", "dollar weakness",
        "rate cut", "rate cuts", "dovish", "fed eases", "easing cycle",
        "inflation surge", "cpi rises", "inflation heats", "inflation jump",
        "geopolitical", "safe-haven", "safe haven", "escalat", "war", "middle east",
        "gold rallies", "gold surges", "gold hits record", "gold jumps", "bullion",
        "yields plunge", "yields drop", "yields fall", "treasury yields tumble",
        "recession fear", "debt crisis", "bank crisis", "fed pause",
    ]

    BEARISH_GOLD_KEYWORDS = [
        "dollar surges", "dollar jumps", "dollar rallies", "dollar gains", "strong dollar", "dollar strength",
        "rate hike", "hawkish", "fed delays", "higher for longer",
        "strong jobs", "nfp beat", "payrolls surge", "job growth",
        "inflation cools", "cpi drops", "inflation slows", "disinflation",
        "yields rise", "yields surge", "yields jump", "treasury yields climb",
        "economic resilience", "retail sales beat", "gdp strong",
    ]

    def __init__(
        self,
        enabled: bool = True,
        pre_news_freeze_seconds: int = 90,
        post_news_momentum_window_sec: int = 300,
        post_news_cooldown_seconds: int = 3,
        max_spread_multiplier: float = 2.5,
        tick_velocity_window_ms: int = 800,
        impulse_tick_density_mult: float = 3.0,
        tracked_currencies: Optional[List[str]] = None,
        block_on_calendar_unavailable: bool = False,
        post_news_freeze_seconds: int = 15,
    ):
        self.enabled = enabled
        self.pre_news_freeze_seconds = pre_news_freeze_seconds
        self.post_news_freeze_seconds = post_news_freeze_seconds
        self.post_news_momentum_window_sec = post_news_momentum_window_sec
        self.post_news_cooldown_seconds = post_news_cooldown_seconds
        self.max_spread_multiplier = max_spread_multiplier
        self.tick_velocity_window_ms = tick_velocity_window_ms
        self.impulse_tick_density_mult = impulse_tick_density_mult
        self.tracked_currencies = tracked_currencies or ["USD", "XAU", "EUR", "GBP"]
        self.block_on_calendar_unavailable = block_on_calendar_unavailable

        # Calendar and feed availability
        self._is_calendar_available: bool = False
        self.calendar_status: str = "INITIALIZING"
        self.headlines_status: str = "INITIALIZING"

        # Timezone Setup
        self.iana_tz_name, self.local_tz, self.utc_offset_str = detect_system_timezone()
        logger.info(f"NewsReactor detected local timezone: {self.iana_tz_name} ({self.utc_offset_str})")

        # Full Calendar Events (Major, Medium, Minor)
        self._all_events: List[CalendarEvent] = []
        self._major_events: List[CalendarEvent] = []
        self._last_calendar_fetch = 0.0

        # Breaking Live Headlines
        self._headlines: Deque[BreakingHeadline] = deque(maxlen=60)
        self._seen_headline_titles = set()

        # Background Tasks
        self._calendar_task: Optional[asyncio.Task] = None
        self._headlines_task: Optional[asyncio.Task] = None

        # Tick Velocity & Volatility Engine
        self._tick_history: Deque[Tuple[float, float, float, float]] = deque(maxlen=5000)
        self._spread_history: Deque[float] = deque(maxlen=300)
        self._avg_spread = 0.25
        self._latest_velocity = 0.0
        self._latest_density_ratio = 1.0
        self._is_impulse_active = False
        self._is_spread_blown_out = False
        self._feed_logs: Deque[Dict[str, Any]] = deque(maxlen=50)

    async def start(self) -> None:
        """Start async calendar refresh and 15s breaking headlines poller."""
        if not self.enabled:
            return

        logger.info("NewsReactor starting Timezone-Aware Calendar & Headlines Stream...")
        # Initial immediate fetch
        await self.refresh_calendar()
        await self.fetch_breaking_headlines()

        if self._calendar_task is None or self._calendar_task.done():
            self._calendar_task = asyncio.create_task(self._calendar_loop())
        if self._headlines_task is None or self._headlines_task.done():
            self._headlines_task = asyncio.create_task(self._headlines_loop())

    async def stop(self) -> None:
        """Gracefully stop tasks."""
        for t in [self._calendar_task, self._headlines_task]:
            if t and not t.done():
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        logger.info("NewsReactor stopped.")

    def format_to_local(self, timestamp_utc: float) -> Tuple[str, str, str]:
        """Convert UTC timestamp into (HH:MM AM/PM, HH:MM:SS AM/PM, YYYY-MM-DD) in local time."""
        dt_local = datetime.fromtimestamp(timestamp_utc, tz=self.local_tz)
        return (
            dt_local.strftime("%I:%M %p"),
            dt_local.strftime("%I:%M:%S %p"),
            dt_local.strftime("%Y-%m-%d"),
        )

    def get_market_session_status(self) -> Dict[str, Any]:
        """
        Evaluate institutional forex session status & countdown to next open in local time.
        """
        now_utc = datetime.now(timezone.utc)
        hour_utc = now_utc.hour + (now_utc.minute / 60.0)

        # Session intervals in UTC:
        # London: 07:00 to 16:30 UTC
        # New York: 12:30 to 21:00 UTC
        # Overlap: 12:30 to 16:30 UTC
        # Asian: 21:00 to 07:00 UTC
        if 12.5 <= hour_utc <= 16.5:
            current_session = "LONDON & NEW YORK OVERLAP (PEAK GOLD VOLUME)"
            status_color = "EMERALD"
            countdown_msg = "London Closes in " + self._format_duration(16.5 - hour_utc)
        elif 7.0 <= hour_utc < 12.5:
            current_session = "LONDON SESSION ACTIVE"
            status_color = "BLUE"
            countdown_msg = "New York Opens in " + self._format_duration(12.5 - hour_utc)
        elif 16.5 < hour_utc <= 21.0:
            current_session = "NEW YORK AFTERNOON SESSION"
            status_color = "PURPLE"
            countdown_msg = "NY Closes in " + self._format_duration(21.0 - hour_utc)
        else:
            current_session = "ASIAN / PACIFIC SESSION (CONSOLIDATION)"
            status_color = "AMBER"
            # Next London open is at 07:00 UTC
            hours_to_london = (7.0 - hour_utc) if hour_utc < 7.0 else (24.0 - hour_utc + 7.0)
            countdown_msg = "London Opens in " + self._format_duration(hours_to_london)

        local_time_now = datetime.now(self.local_tz).strftime("%I:%M:%S %p")

        return {
            "current_session": current_session,
            "session_badge_color": status_color,
            "session_countdown": countdown_msg,
            "local_time_now": local_time_now,
            "iana_tz": self.iana_tz_name,
            "utc_offset": self.utc_offset_str,
        }

    @staticmethod
    def _format_duration(hours_float: float) -> str:
        total_seconds = int(max(0.0, hours_float * 3600))
        h = total_seconds // 3600
        m = (total_seconds % 3600) // 60
        return f"{h:02d}h {m:02d}m"

    async def _calendar_loop(self) -> None:
        """Poll ForexFactory calendar every 30 minutes."""
        while self.enabled:
            try:
                await asyncio.sleep(1800)
                await self.refresh_calendar()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in calendar refresh loop: {e}")
                await asyncio.sleep(60.0)

    async def refresh_calendar(self) -> None:
        """Pull weekly calendar from ForexFactory JSON endpoint."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                async with session.get(self.FOREX_FACTORY_URL, headers=headers, timeout=10) as resp:
                    if resp.status == 200:
                        raw_data = await resp.json()
                        self._parse_full_calendar(raw_data)
                        self._last_calendar_fetch = time.time()
                        self._is_calendar_available = True
                        self.calendar_status = "AVAILABLE"
                        logger.info(f"Loaded {len(self._all_events)} events ({len(self._major_events)} Major) into NewsReactor.")
                        return
                    else:
                        logger.warning(f"ForexFactory calendar endpoint returned HTTP {resp.status}.")
        except Exception as e:
            logger.warning(f"Could not load live economic calendar ({e}). Feed marked UNAVAILABLE. Fake schedule generation is disabled.")

        self._is_calendar_available = False
        self.calendar_status = "UNAVAILABLE"
        self._all_events = []
        self._major_events = []

    def _parse_full_calendar(self, raw_events: List[Dict[str, Any]]) -> None:
        """Parse raw calendar items into timezone-converted CalendarEvent objects."""
        now_utc = time.time()
        parsed_all: List[CalendarEvent] = []
        parsed_major: List[CalendarEvent] = []

        for idx, item in enumerate(raw_events):
            country = str(item.get("country", "")).strip().upper()
            if country not in self.tracked_currencies:
                continue

            raw_impact = str(item.get("impact", "")).strip().lower()
            if raw_impact in ["high", "red"]:
                impact_level = "HIGH"
                impact_badge = "🔴 HIGH"
            elif raw_impact in ["medium", "orange"]:
                impact_level = "MEDIUM"
                impact_badge = "🟠 MED"
            elif raw_impact in ["low", "yellow"]:
                impact_level = "LOW"
                impact_badge = "🟡 LOW"
            else:
                continue # Skip holidays / non-economic

            date_str = item.get("date", "")
            try:
                dt = dt_parser.parse(date_str)
                ts = dt.astimezone(timezone.utc).timestamp()
            except Exception:
                continue

            # Keep events from 24h ago to 7 days ahead
            if ts < (now_utc - 86400) or ts > (now_utc + 7 * 86400):
                continue

            hm, hms, date_d = self.format_to_local(ts)

            # Determine relative string
            diff = ts - now_utc
            if abs(diff) <= 900:
                rel_str = "LIVE NOW"
                status = "LIVE NOW"
            elif diff < -900:
                rel_str = f"{abs(int(diff // 60))}m ago"
                status = "PASSED"
            else:
                h = int(diff // 3600)
                m = int((diff % 3600) // 60)
                rel_str = f"in {h:02d}h {m:02d}m" if h > 0 else f"in {m}m"
                status = "UPCOMING"

            event = CalendarEvent(
                id=f"ev_{idx}_{int(ts)}",
                title=str(item.get("title", "Economic Release")),
                currency=country,
                impact=impact_level,
                impact_badge=impact_badge,
                timestamp_utc=ts,
                local_time_hm=hm,
                local_time_hms=hms,
                local_date_str=date_d,
                forecast=str(item.get("forecast", "")),
                previous=str(item.get("previous", "")),
                actual=str(item.get("actual", "")),
                status=status,
                relative_time=rel_str,
            )

            parsed_all.append(event)
            if impact_level == "HIGH":
                parsed_major.append(event)

        parsed_all.sort(key=lambda x: x.timestamp_utc)
        parsed_major.sort(key=lambda x: x.timestamp_utc)

        self._all_events = parsed_all
        self._major_events = parsed_major

    def _populate_fallback_calendar(self) -> None:
        """Create structured representative calendar if network is offline."""
        now = time.time()
        sample_data = [
            ("US Core CPI m/m", "USD", "HIGH", "🔴 HIGH", 7200, "0.3%", "0.2%"),
            ("US Core Retail Sales m/m", "USD", "HIGH", "🔴 HIGH", 14400, "0.4%", "0.1%"),
            ("US Unemployment Claims", "USD", "HIGH", "🔴 HIGH", 28800, "215K", "219K"),
            ("Fed Chair Powell Speaks", "USD", "HIGH", "🔴 HIGH", 43200, "-", "-"),
            ("US Michigan Consumer Sentiment", "USD", "MEDIUM", "🟠 MED", 10800, "69.0", "68.2"),
            ("US Housing Starts", "USD", "LOW", "🟡 LOW", 5400, "1.36M", "1.35M"),
        ]

        parsed: List[CalendarEvent] = []
        for idx, (title, cur, imp, badge, offset, fcast, prev) in enumerate(sample_data):
            ts = now + offset
            hm, hms, date_d = self.format_to_local(ts)
            ev = CalendarEvent(
                id=f"fb_{idx}",
                title=title,
                currency=cur,
                impact=imp,
                impact_badge=badge,
                timestamp_utc=ts,
                local_time_hm=hm,
                local_time_hms=hms,
                local_date_str=date_d,
                forecast=fcast,
                previous=prev,
                actual="",
                status="UPCOMING",
                relative_time=f"in {int(offset//3600)}h {int((offset%3600)//60)}m",
            )
            parsed.append(ev)

        self._all_events = parsed
        self._major_events = [e for e in parsed if e.impact == "HIGH"]

    async def _headlines_loop(self) -> None:
        """Poll live financial RSS feeds every 15 seconds."""
        while self.enabled:
            try:
                await asyncio.sleep(15)
                await self.fetch_breaking_headlines()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in headlines loop: {e}")
                await asyncio.sleep(15.0)

    async def fetch_breaking_headlines(self) -> None:
        """Fetch news headlines from financial RSS feeds and score Gold sentiment."""
        loop = asyncio.get_running_loop()

        def _fetch_rss():
            headlines_found: List[BreakingHeadline] = []
            for source_name, url in self.RSS_FEED_URLS:
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        root = ET.fromstring(resp.read())
                        items = root.findall(".//item")
                        for item in items[:8]: # Top 8 from each source
                            title_el = item.find("title")
                            link_el = item.find("link")
                            pub_date_el = item.find("pubDate")

                            title = title_el.text.strip() if title_el is not None and title_el.text else ""
                            if not title or title in self._seen_headline_titles:
                                continue

                            link = link_el.text.strip() if link_el is not None and link_el.text else "#"
                            
                            ts = time.time()
                            if pub_date_el is not None and pub_date_el.text:
                                try:
                                    dt = dt_parser.parse(pub_date_el.text)
                                    ts = dt.astimezone(timezone.utc).timestamp()
                                except Exception:
                                    pass

                            sentiment, badge = self.score_gold_sentiment(title)
                            hm, hms, _ = self.format_to_local(ts)

                            # Relative age
                            diff = max(0, int(time.time() - ts))
                            rel_str = f"{diff}s ago" if diff < 60 else f"{diff//60}m ago"

                            hl = BreakingHeadline(
                                title=title,
                                source=source_name,
                                url=link,
                                timestamp_utc=ts,
                                local_time_str=hms,
                                relative_str=rel_str,
                                sentiment=sentiment,
                                sentiment_badge=badge,
                            )
                            headlines_found.append(hl)
                            self._seen_headline_titles.add(title)
                except Exception as e:
                    logger.debug(f"RSS fetch failed for {source_name}: {e}")

            return headlines_found

        new_items = await loop.run_in_executor(None, _fetch_rss)
        for item in reversed(new_items):
            self._headlines.appendleft(item)

        if self._headlines:
            self.headlines_status = "AVAILABLE"
        else:
            self.headlines_status = "UNAVAILABLE"

    def score_gold_sentiment(self, text: str) -> Tuple[str, str]:
        """
        Calculates Gold (XAUUSD) Sentiment Score from headline keywords:
        - Bullish Gold: USD weakness, rate cuts, inflation, geopolitical escalation
        - Bearish Gold: USD strength, rate hikes, hawkish Fed, surging yields
        """
        lower = text.lower()
        bull_score = sum(1 for kw in self.BULLISH_GOLD_KEYWORDS if kw in lower)
        bear_score = sum(1 for kw in self.BEARISH_GOLD_KEYWORDS if kw in lower)

        if bull_score > bear_score:
            return "BULLISH_GOLD", "🟢 BULLISH GOLD"
        elif bear_score > bull_score:
            return "BEARISH_GOLD", "🔴 BEARISH GOLD"
        return "NEUTRAL", "⚪ NEUTRAL"

    def _seed_fallback_headlines(self) -> None:
        """Seed representative financial headlines with sentiment tags."""
        now = time.time()
        samples = [
            ("Middle East geopolitical tensions spur safe-haven bids across Gold and crude oil", "Reuters", -45),
            ("US Dollar steadies as Treasury yields hold near 4.10% ahead of CPI release", "CNBC Finance", -120),
            ("Federal Reserve speakers reiterate data-dependent stance on upcoming rate cuts", "MarketWatch", -240),
            ("Gold consolidates near session highs as investors await US retail sales momentum", "FinancialJuice", -360),
        ]
        for t, src, diff in samples:
            ts = now + diff
            sent, badge = self.score_gold_sentiment(t)
            hm, hms, _ = self.format_to_local(ts)
            self._headlines.append(BreakingHeadline(
                title=t,
                source=src,
                url="#",
                timestamp_utc=ts,
                local_time_str=hms,
                relative_str=f"{abs(diff)}s ago",
                sentiment=sent,
                sentiment_badge=badge,
            ))

    def get_complete_news_state(self, current_utc: Optional[float] = None) -> Dict[str, Any]:
        """
        Generate complete real-time payload:
        1. Local Timezone info & Market Sessions
        2. 'Happening Right Now' Active Alert (T-freeze to T+cooldown)
        3. Hero Next Major News Countdown
        4. Full Timeline Table (Major + Minor)
        5. Breaking Live Headlines Stream with Gold Sentiment
        6. Explicit Calendar Availability Status
        """
        now = current_utc if current_utc is not None else time.time()
        session_info = self.get_market_session_status()

        # Update dynamic relative times and status on all events
        for ev in self._all_events:
            diff = ev.timestamp_utc - now
            if -900 <= diff <= 120:
                ev.status = "LIVE NOW"
                ev.relative_time = "LIVE NOW"
            elif diff < -900:
                ev.status = "PASSED"
                ev.relative_time = f"{abs(int(diff // 60))}m ago"
            else:
                ev.status = "UPCOMING"
                h = int(diff // 3600)
                m = int((diff % 3600) // 60)
                ev.relative_time = f"in {h:02d}h {m:02d}m" if h > 0 else f"in {m}m"

        # 1. State Machine & High-Impact Freeze Detection
        happening_now_event: Optional[Dict[str, Any]] = None
        happening_mode = "NONE" # "PRE_NEWS_SHIELD_LOCK", "POST_NEWS_FREEZE_LOCK", "POST_NEWS_MOMENTUM_CAPTURE", "NONE"
        is_news_freeze = False

        for ev in self._all_events:
            diff = ev.timestamp_utc - now # positive = future, negative = past
            if ev.impact == "HIGH":
                # Strict freeze window before and after high-impact events
                if -self.post_news_freeze_seconds <= diff <= self.pre_news_freeze_seconds:
                    is_news_freeze = True
                    happening_now_event = ev.to_dict()
                    if diff >= 0:
                        happening_mode = "PRE_NEWS_SHIELD_LOCK"
                    elif -self.post_news_cooldown_seconds <= diff < 0:
                        happening_mode = "WHIPSAW_SPIKE_COOLING"
                    else:
                        happening_mode = "POST_NEWS_FREEZE_LOCK"
                    break
            elif -900 <= diff <= 120:
                happening_now_event = ev.to_dict()
                happening_mode = "POST_RELEASE_SETTLING"

        # 2. Hero Next Major News Countdown
        next_major_event: Optional[Dict[str, Any]] = None
        countdown_seconds: Optional[int] = None
        countdown_formatted = "None Scheduled"

        major_events_list = self._major_events if self._major_events else [e for e in self._all_events if e.impact == "HIGH"]
        for ev in major_events_list:
            diff = ev.timestamp_utc - now
            if diff > 0:
                next_major_event = ev.to_dict()
                countdown_seconds = int(diff)
                h = int(diff // 3600)
                m = int((diff % 3600) // 60)
                s = int(diff % 60)
                countdown_formatted = f"{h:02d}h {m:02d}m {s:02d}s"
                break

        # 3. Today & Tomorrow Chronological Timeline
        today_str = datetime.now(self.local_tz).strftime("%Y-%m-%d")
        tomorrow_str = (datetime.now(self.local_tz) + timedelta(days=1)).strftime("%Y-%m-%d")
        timeline = [
            ev.to_dict() for ev in self._all_events
            if ev.local_date_str in [today_str, tomorrow_str]
        ]

        # 4. Breaking Headlines
        headlines_list = []
        for hl in list(self._headlines)[:20]:
            diff = max(0, int(now - hl.timestamp_utc))
            hl_dict = hl.to_dict()
            hl_dict["relative_str"] = f"{diff}s ago" if diff < 60 else f"{diff//60}m ago"
            headlines_list.append(hl_dict)

        return {
            "timezone": session_info,
            "calendar_status": self.calendar_status,
            "is_calendar_available": self._is_calendar_available,
            "headlines_status": self.headlines_status,
            "happening_now": {
                "active": happening_now_event is not None,
                "event": happening_now_event,
                "mode": happening_mode,
            },
            "next_major_news": {
                "event": next_major_event,
                "countdown_seconds": countdown_seconds,
                "countdown_formatted": countdown_formatted,
            },
            "timeline": timeline,
            "headlines": headlines_list,
            "is_news_freeze": is_news_freeze,
        }

    def check_calendar_state(self, current_utc: Optional[float] = None) -> Dict[str, Any]:
        """Backward-compatible fast calendar state check for trading loop."""
        state = self.get_complete_news_state(current_utc=current_utc)
        happening = state.get("happening_now", {})
        next_maj = state.get("next_major_news", {})
        
        is_freeze = state.get("is_news_freeze", False)
        is_momentum = happening.get("mode") == "POST_NEWS_MOMENTUM_CAPTURE"

        nearest_ev = happening.get("event") or next_maj.get("event")
        cd = next_maj.get("countdown_seconds")

        msg = "Normal Market Flow"
        if is_freeze:
            ev_title = nearest_ev.get("title", "High-Impact Release") if nearest_ev else "Release"
            msg = f"[HIGH-IMPACT NEWS FREEZE] '{ev_title}' active! Entries locked."
        elif is_momentum:
            ev_title = nearest_ev.get("title", "Release") if nearest_ev else "Release"
            msg = f"[POST-NEWS MOMENTUM] Sniping institutional wave on '{ev_title}'."
        elif nearest_ev:
            msg = f"Next: {nearest_ev.get('title')} ({nearest_ev.get('currency')}) in {next_maj.get('countdown_formatted')}"
        elif not self._is_calendar_available:
            msg = "Economic calendar feed unavailable."

        return {
            "is_news_freeze": is_freeze,
            "is_post_news_momentum": is_momentum,
            "should_lock_breakeven": is_freeze,
            "nearest_event": nearest_ev,
            "countdown_seconds": cd,
            "message": msg,
            "calendar_status": self.calendar_status,
            "is_calendar_available": self._is_calendar_available,
        }

    def on_tick(self, tick: Tick, atr_15m: float = 2.50) -> Dict[str, Any]:
        """Track sub-50ms tick velocity and spread blowout shield."""
        now = tick.time if tick.time > 0 else time.time()
        self._tick_history.append((now, tick.bid, tick.ask, tick.spread))
        self._spread_history.append(tick.spread)

        if len(self._spread_history) >= 10:
            self._avg_spread = float(sum(self._spread_history) / len(self._spread_history))

        spread_ratio = (tick.spread / self._avg_spread) if self._avg_spread > 0 else 1.0
        self._is_spread_blown_out = (spread_ratio >= self.max_spread_multiplier)

        # 800ms Displacement
        target_ts = now - (self.tick_velocity_window_ms / 1000.0)
        reference_bid = tick.bid
        for t_time, t_bid, _, _ in reversed(self._tick_history):
            if t_time <= target_ts:
                reference_bid = t_bid
                break

        price_delta_800ms = abs(tick.bid - reference_bid)
        effective_atr = max(atr_15m, 0.50)
        self._latest_velocity = round(price_delta_800ms / effective_atr, 4)

        # Density in last 1.0s vs 30s average
        t_1s_ago = now - 1.0
        ticks_1s = sum(1 for t in self._tick_history if t[0] >= t_1s_ago)
        t_30s_ago = now - 30.0
        ticks_30s = sum(1 for t in self._tick_history if t[0] >= t_30s_ago)
        avg_density = (ticks_30s / 30.0) if ticks_30s > 0 else 1.0

        self._latest_density_ratio = round(ticks_1s / max(avg_density, 0.5), 2)

        self._is_impulse_active = (
            (self._latest_density_ratio >= self.impulse_tick_density_mult) and
            (self._latest_velocity >= 0.20) and
            (not self._is_spread_blown_out)
        )

        return {
            "tick_velocity": self._latest_velocity,
            "density_ratio": self._latest_density_ratio,
            "avg_spread": round(self._avg_spread, 3),
            "spread_ratio": round(spread_ratio, 2),
            "is_impulse_active": self._is_impulse_active,
            "is_spread_blown_out": self._is_spread_blown_out,
        }

    def _parse_economic_num(self, val_str: str) -> Optional[float]:
        """Safely parse numbers with units like %, K, M, B from calendar strings."""
        if not val_str or str(val_str).strip() in ["-", "", "None"]:
            return None
        s = str(val_str).strip()
        mult = 1.0
        if s.endswith("%"):
            mult = 1.0
        elif s.upper().endswith("K"):
            mult = 1e3
        elif s.upper().endswith("M"):
            mult = 1e6
        elif s.upper().endswith("B"):
            mult = 1e9
        cleaned = re.sub(r"[^\d\.\-+]", "", s)
        try:
            return float(cleaned) * mult
        except ValueError:
            return None

    def determine_fundamental_forecast_bias(self, event: CalendarEvent) -> Tuple[str, str]:
        """
        Determine fundamental forecast bias for Gold (XAUUSD) based on Forecast vs. Previous:
        - For standard USD strength/inflation metrics (CPI, Core CPI, PPI, NFP, ADP, GDP, Retail Sales, ISM, PMI, Fed Rate, JOLTS):
          * Forecast < Previous (Cooling USD / Dovish) -> BULLISH_GOLD
          * Forecast > Previous (Hotter USD / Hawkish) -> BEARISH_GOLD
        - For inverse USD metrics (Unemployment Rate, Initial Jobless Claims):
          * Forecast > Previous (Weakening Labor) -> BULLISH_GOLD
          * Forecast < Previous (Tightening Labor) -> BEARISH_GOLD
        """
        fcast_num = self._parse_economic_num(event.forecast)
        prev_num = self._parse_economic_num(event.previous)
        t_lower = event.title.lower()

        if fcast_num is None or prev_num is None:
            return "NEUTRAL", f"No forecast/previous numbers available for '{event.title}'"

        diff = fcast_num - prev_num

        # Check inverse USD metrics first
        inverse_metrics = [
            "unemployment rate",
            "jobless claims",
            "initial claims",
            "continuing claims",
            "unemployment claims",
            "claims",
        ]
        is_inverse = any(k in t_lower for k in inverse_metrics)

        # Standard USD metrics
        standard_metrics = [
            "cpi", "core cpi", "ppi", "nfp", "non-farm", "payrolls",
            "adp", "gdp", "retail sales", "ism", "pmi", "fed rate",
            "interest rate", "jolts", "employment change",
        ]
        is_standard = any(k in t_lower for k in standard_metrics)

        if is_inverse:
            if diff > 0:
                return "BULLISH_GOLD", f"Weakening Labor: Forecast {event.forecast} > Previous {event.previous} -> Bullish Gold"
            elif diff < 0:
                return "BEARISH_GOLD", f"Tightening Labor: Forecast {event.forecast} < Previous {event.previous} -> Bearish Gold"
            return "NEUTRAL", f"Forecast {event.forecast} matches Previous {event.previous}"

        if is_standard or True:
            # Default to standard USD strength logic if non-inverse
            if diff < 0:
                return "BULLISH_GOLD", f"Cooling USD: Forecast {event.forecast} < Previous {event.previous} -> Bullish Gold"
            elif diff > 0:
                return "BEARISH_GOLD", f"Hotter USD: Forecast {event.forecast} > Previous {event.previous} -> Bearish Gold"
            return "NEUTRAL", f"Forecast {event.forecast} matches Previous {event.previous}"

    def evaluate_pre_news_trajectory(
        self,
        symbol: str = "XAUUSD",
        dfs: Optional[Dict[str, Any]] = None,
        tick_velocity: float = 0.0,
        current_utc: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Smart Pre-News Trajectory & Drift Analyzer (T-45m to T-90s window).
        Combines:
        1. 40% Weight — Institutional Order-Flow Drift: 15m, 5m, 1m price action + EMA slopes.
        2. 30% Weight — Tick Accumulation & Volume: tick velocity and candle body expansion (> 60% of range).
        3. 30% Weight — Fundamental Forecast Agreement: technical drift matches fundamental bias.
        """
        now = current_utc if current_utc is not None else time.time()
        upcoming_events = [
            ev for ev in self._all_events
            if 0 < (ev.timestamp_utc - now) <= 2700  # within 45 minutes
            and ev.impact in ("HIGH", "MEDIUM")
            and ev.currency in ("USD", "XAU")
        ]

        if not upcoming_events:
            return {
                "active": False,
                "confidence": 0.0,
                "confidence_pct": 0,
                "trajectory_bias": "NEUTRAL",
                "drift_direction": "NONE",
                "fundamental_bias": "NEUTRAL",
                "event": None,
                "seconds_to_event": 0,
                "reasons": ["No upcoming High/Medium USD/Gold news release within 45 minutes"],
            }

        nearest_ev = min(upcoming_events, key=lambda x: x.timestamp_utc)
        seconds_to_event = int(nearest_ev.timestamp_utc - now)

        # 1. Fundamental Forecast Bias
        fundamental_bias, fund_reason = self.determine_fundamental_forecast_bias(nearest_ev)

        # 2. 40% Weight — Institutional Order-Flow Drift (15m, 5m, 1m)
        order_flow_pts = 10
        drift_direction = "NEUTRAL"
        timeframe_signals = {}

        if dfs:
            from engine.indicators import TechnicalIndicators
            tf_votes = []
            for tf in ["15m", "5m", "1m"]:
                df = dfs.get(tf)
                if df is not None and len(df) >= 10:
                    c = df["close"].values
                    ema9 = TechnicalIndicators.ema(c, min(9, len(c)))[-1]
                    ema21 = TechnicalIndicators.ema(c, min(21, len(c)))[-1]
                    slope = TechnicalIndicators.ema_slope(c, min(14, len(c)))
                    if c[-1] > ema9 and c[-1] > ema21 and slope > 0.01:
                        tf_votes.append("BUY")
                        timeframe_signals[tf] = "BUY"
                    elif c[-1] < ema9 and c[-1] < ema21 and slope < -0.01:
                        tf_votes.append("SELL")
                        timeframe_signals[tf] = "SELL"
                    else:
                        timeframe_signals[tf] = "NEUTRAL"

            buy_votes = sum(1 for v in tf_votes if v == "BUY")
            sell_votes = sum(1 for v in tf_votes if v == "SELL")

            if buy_votes == 3:
                order_flow_pts = 40
                drift_direction = "BUY"
            elif sell_votes == 3:
                order_flow_pts = 40
                drift_direction = "SELL"
            elif buy_votes >= 2 and sell_votes == 0:
                order_flow_pts = 28
                drift_direction = "BUY"
            elif sell_votes >= 2 and buy_votes == 0:
                order_flow_pts = 28
                drift_direction = "SELL"
            elif buy_votes > sell_votes:
                order_flow_pts = 20
                drift_direction = "BUY"
            elif sell_votes > buy_votes:
                order_flow_pts = 20
                drift_direction = "SELL"
            else:
                order_flow_pts = 12
                drift_direction = "NEUTRAL"
        else:
            # Fallback if DataFrames not passed
            order_flow_pts = 25
            drift_direction = "BUY" if fundamental_bias == "BULLISH_GOLD" else ("SELL" if fundamental_bias == "BEARISH_GOLD" else "NEUTRAL")

        # 3. 30% Weight — Tick Accumulation & Candle Body Expansion (> 60% of range)
        volume_pts = 10
        body_expanded = False
        if dfs:
            # Check 1m and 5m recent candle bodies
            for tf_check in ["1m", "5m"]:
                df_check = dfs.get(tf_check)
                if df_check is not None and len(df_check) >= 3:
                    c = df_check["close"].values
                    o = df_check["open"].values
                    h = df_check["high"].values
                    l = df_check["low"].values
                    recent_bodies = [abs(c[i] - o[i]) for i in range(-3, 0)]
                    recent_ranges = [max(h[i] - l[i], 0.01) for i in range(-3, 0)]
                    ratios = [b / r for b, r in zip(recent_bodies, recent_ranges)]
                    if any(r >= 0.60 for r in ratios):
                        body_expanded = True
                        break

        effective_vel = tick_velocity if tick_velocity > 0 else self._latest_velocity
        is_velocity_elevated = effective_vel >= 0.15 or self._latest_density_ratio >= 1.25

        if body_expanded and is_velocity_elevated:
            volume_pts = 30
        elif body_expanded or is_velocity_elevated:
            volume_pts = 22
        else:
            volume_pts = 12

        # 4. 30% Weight — Fundamental Forecast Agreement
        fundamental_pts = 10
        fundamental_agreed = False
        if fundamental_bias in ("BULLISH_GOLD", "BEARISH_GOLD"):
            expected_dir = "BUY" if fundamental_bias == "BULLISH_GOLD" else "SELL"
            if drift_direction == expected_dir:
                fundamental_pts = 30
                fundamental_agreed = True
            elif drift_direction == "NEUTRAL":
                fundamental_pts = 20
                fundamental_agreed = True
            else:
                # Contradiction: technical drift opposing fundamental forecast
                fundamental_pts = 5
                fundamental_agreed = False
        else:
            # Neutral fundamental forecast
            fundamental_pts = 20
            fundamental_agreed = True

        total_pts = min(100, max(0, order_flow_pts + volume_pts + fundamental_pts))
        confidence = round(total_pts / 100.0, 2)

        trajectory_bias = drift_direction if (drift_direction in ("BUY", "SELL") and confidence >= 0.50) else (
            "BUY" if fundamental_bias == "BULLISH_GOLD" and confidence >= 0.70 else (
                "SELL" if fundamental_bias == "BEARISH_GOLD" and confidence >= 0.70 else "NEUTRAL"
            )
        )

        reasons = [
            f"Upcoming event: {nearest_ev.title} ({nearest_ev.currency}) in {seconds_to_event}s",
            fund_reason,
            f"Order-Flow Drift: {drift_direction} ({order_flow_pts}/40 pts)",
            f"Volume/Expansion: {'Expanded (>60%)' if body_expanded else 'Moderate'} ({volume_pts}/30 pts)",
            f"Fundamental Agreement: {'Matched' if fundamental_agreed else 'Divergent'} ({fundamental_pts}/30 pts)",
            f"Total Trajectory Confidence: {total_pts}% -> Bias: {trajectory_bias}",
        ]

        return {
            "active": True,
            "confidence": confidence,
            "confidence_pct": total_pts,
            "trajectory_bias": trajectory_bias,
            "drift_direction": drift_direction,
            "fundamental_bias": fundamental_bias,
            "event": nearest_ev.to_dict(),
            "event_title": nearest_ev.title,
            "event_impact": nearest_ev.impact,
            "seconds_to_event": seconds_to_event,
            "forecast": nearest_ev.forecast,
            "previous": nearest_ev.previous,
            "order_flow_pts": order_flow_pts,
            "volume_pts": volume_pts,
            "fundamental_pts": fundamental_pts,
            "reasons": reasons,
        }


    def analyze_news_fundamental_bias(self, symbol: str = "XAUUSD") -> Dict[str, Any]:
        """
        Deep fundamental study of economic calendar and real-time breaking headlines:
        1. Analyzes surprise delta (Actual vs. Forecast) on US CPI, NFP, Retail Sales, Claims.
        2. Aggregates breaking financial headlines sentiment (Bullish vs. Bearish Gold/USD).
        3. Delivers actionable directional bias (BUY / SELL) instead of pure defensive freeze.
        """
        now = time.time()
        is_gold = any(k in symbol.upper() for k in ["XAU", "GOLD"])

        bias_direction = "NEUTRAL"
        confidence = 0.50
        summary_parts = []
        recent_surprise = None

        # 1. Economic Calendar Surprise Study (Look back 2 hours, look forward 30 mins)
        active_events = [
            ev for ev in self._all_events
            if -7200 <= (ev.timestamp_utc - now) <= 1800 and ev.currency in ["USD", "XAU"]
        ]

        for ev in active_events:
            actual_num = self._parse_economic_num(ev.actual)
            fcast_num = self._parse_economic_num(ev.forecast)
            prev_num = self._parse_economic_num(ev.previous)

            t_lower = ev.title.lower()

            # A. Inflation releases (CPI, Core CPI, PPI, PCE)
            if any(k in t_lower for k in ["cpi", "ppi", "pce", "inflation"]):
                if actual_num is not None and fcast_num is not None:
                    diff = actual_num - fcast_num
                    recent_surprise = {"event": ev.title, "actual": actual_num, "forecast": fcast_num, "diff": diff}
                    if diff > 0.05: # Hotter inflation -> USD stronger -> Gold drops
                        bias_direction = "SELL" if is_gold else "BUY"
                        confidence = 0.85
                        summary_parts.append(f"{ev.title} beat forecast ({ev.actual} vs {ev.forecast}) -> Bearish Gold")
                    elif diff < -0.05: # Cooling inflation -> Fed cuts -> Gold surges
                        bias_direction = "BUY" if is_gold else "SELL"
                        confidence = 0.88
                        summary_parts.append(f"{ev.title} came in cooler ({ev.actual} vs {ev.forecast}) -> Bullish Gold")

            # B. Employment releases (Non-Farm Payrolls, Unemployment Rate, ADP)
            elif any(k in t_lower for k in ["non-farm", "payrolls", "nfp", "employment change"]):
                if actual_num is not None and fcast_num is not None:
                    diff = actual_num - fcast_num
                    recent_surprise = {"event": ev.title, "actual": actual_num, "forecast": fcast_num, "diff": diff}
                    if diff > 15.0: # Strong jobs -> hawkish Fed -> Bearish Gold
                        bias_direction = "SELL" if is_gold else "BUY"
                        confidence = 0.85
                        summary_parts.append(f"Jobs beat strongly ({ev.actual} vs {ev.forecast}) -> Strong USD")
                    elif diff < -15.0: # Weak jobs -> dovish Fed -> Bullish Gold
                        bias_direction = "BUY" if is_gold else "SELL"
                        confidence = 0.88
                        summary_parts.append(f"Jobs missed ({ev.actual} vs {ev.forecast}) -> Rate cut impulse")

            # C. Unemployment Claims
            elif "claims" in t_lower:
                if actual_num is not None and fcast_num is not None:
                    diff = actual_num - fcast_num
                    if diff > 5.0: # Higher claims -> weaker labor -> Bullish Gold
                        bias_direction = "BUY" if is_gold else "SELL"
                        confidence = 0.75
                        summary_parts.append(f"Jobless claims surged ({ev.actual} vs {ev.forecast}) -> Weak USD")
                    elif diff < -5.0:
                        bias_direction = "SELL" if is_gold else "BUY"
                        confidence = 0.75
                        summary_parts.append(f"Jobless claims fell ({ev.actual} vs {ev.forecast}) -> Strong USD")

            # D. Retail Sales / GDP
            elif any(k in t_lower for k in ["retail sales", "gdp"]):
                if actual_num is not None and fcast_num is not None:
                    diff = actual_num - fcast_num
                    if diff > 0.1:
                        bias_direction = "SELL" if is_gold else "BUY"
                        confidence = 0.75
                        summary_parts.append(f"{ev.title} beat expectations ({ev.actual} vs {ev.forecast})")
                    elif diff < -0.1:
                        bias_direction = "BUY" if is_gold else "SELL"
                        confidence = 0.75
                        summary_parts.append(f"{ev.title} contracted ({ev.actual} vs {ev.forecast})")

        # 2. Breaking Headlines Sentiment Analysis
        bullish_hl = 0
        bearish_hl = 0
        recent_headlines = list(self._headlines)[-25:]
        for hl in recent_headlines:
            if hl.sentiment == "BULLISH_GOLD":
                bullish_hl += 1
            elif hl.sentiment == "BEARISH_GOLD":
                bearish_hl += 1

        total_sentiment_hl = bullish_hl + bearish_hl
        sentiment_ratio = (bullish_hl - bearish_hl) / max(total_sentiment_hl, 1)

        if bias_direction == "NEUTRAL" and total_sentiment_hl >= 3:
            if sentiment_ratio >= 0.35:
                bias_direction = "BUY" if is_gold else "SELL"
                confidence = 0.72
                summary_parts.append(f"Breaking headlines heavily Bullish Gold ({bullish_hl} pos vs {bearish_hl} neg)")
            elif sentiment_ratio <= -0.35:
                bias_direction = "SELL" if is_gold else "BUY"
                confidence = 0.72
                summary_parts.append(f"Breaking headlines heavily Bearish Gold ({bearish_hl} neg vs {bullish_hl} pos)")

        if not summary_parts:
            summary_parts.append("Economic calendar and headlines balanced; standard technical structure applies.")

        return {
            "direction": bias_direction,
            "confidence": round(confidence, 2),
            "summary": "; ".join(summary_parts),
            "headline_bullish": bullish_hl,
            "headline_bearish": bearish_hl,
            "recent_surprise": recent_surprise,
        }

    def can_open_baseline_trade(self) -> Tuple[bool, str]:
        """
        Smart News Entry Gate:
        Enforces strict high-impact news freeze and spread blowout shield.
        During confirmed high-impact economic news freeze windows, new entries are strictly locked.
        """
        if self._is_spread_blown_out:
            return False, f"Spread blown out ({self.max_spread_multiplier}x normal). Broker slippage shield active."

        if not self._is_calendar_available:
            if self.block_on_calendar_unavailable:
                return False, "Economic calendar feed unavailable (strict safety lock enabled)."

        state = self.check_calendar_state()
        if state.get("is_news_freeze", False):
            nearest = state.get("nearest_event") or {}
            ev_title = nearest.get("title", "High-Impact Release")
            return False, f"High-impact news freeze active for '{ev_title}'. New entries strictly locked."

        warning_str = " (Warning: Economic calendar feed unavailable)" if not self._is_calendar_available else ""
        return True, f"Trading conditions nominal.{warning_str}"

    def get_feed_logs(self, count: int = 20) -> List[Dict[str, Any]]:
        """Return latest activity logs."""
        return list(self._feed_logs)[-count:]
