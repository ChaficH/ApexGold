"""
ApexGold Automated Regression & Verification Tests for Phase 4:
Market Data, Indicators, and News Reactor Safety.

Verifies:
1. Item 13: Candle deduplication, monotonic chronological ordering, completed-bar filtering, stale check, and synthetic tag isolation.
2. Item 14: Economic calendar unavailability reporting (no fabricated events), user-configurable safety blocking, hard high-impact news freeze, and no fake headlines.
3. Item 15: Indicator calculations NaN resilience, sparse history safety, and elimination of fabricated historical pattern edge.
"""

import os
import sys
import time
import pytest
import numpy as np
import pandas as pd
from unittest.mock import AsyncMock, patch

# Ensure repo root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from brokers.base import Bar, Tick
from brokers.paper_sim import PaperSimAdapter
from engine.data_hub import DataHub
from engine.historical_matcher import HistoricalMatcher, SetupVector
from engine.indicators import TechnicalIndicators
from engine.news_reactor import CalendarEvent, NewsReactor


# ==============================================================================
# Item 13: Data Hub & Market Data Synchronization Tests
# ==============================================================================

@pytest.mark.anyio
async def test_data_hub_append_bar_deduplication_and_monotonic_ordering(tmp_path):
    """Verify append_bar updates identical timestamps and maintains strictly sorted order."""
    db_file = str(tmp_path / "test_data.db")
    hub = DataHub(db_path=db_file)
    await hub.initialize()

    # Append out-of-order bars and duplicate timestamps
    base_ts = 1700000000
    b1 = Bar(time=base_ts + 60, open=2000.0, high=2005.0, low=1999.0, close=2004.0, volume=10.0)
    b2 = Bar(time=base_ts + 180, open=2004.0, high=2008.0, low=2003.0, close=2007.0, volume=15.0)
    b3 = Bar(time=base_ts + 120, open=2002.0, high=2006.0, low=2001.0, close=2005.0, volume=12.0) # Arrives out of order
    b1_update = Bar(time=base_ts + 60, open=2000.0, high=2005.0, low=1999.0, close=2004.5, volume=11.0) # Duplicate timestamp update

    await hub.append_bar("XAUUSD", "1m", b1)
    await hub.append_bar("XAUUSD", "1m", b2)
    await hub.append_bar("XAUUSD", "1m", b3)
    await hub.append_bar("XAUUSD", "1m", b1_update)

    bars = await hub.get_bars("XAUUSD", "1m", count=10)
    assert len(bars) == 3, f"Expected 3 deduplicated bars, got {len(bars)}"
    timestamps = [b.time for b in bars]
    assert timestamps == [base_ts + 60, base_ts + 120, base_ts + 180], "Bars are not in monotonic sorted order"
    assert bars[0].close == 2004.5, "Bar at base_ts + 60 was not updated with latest value"


@pytest.mark.anyio
async def test_data_hub_inject_bars_deduplicates_and_sorts(tmp_path):
    """Verify inject_bars deduplicates mixed sequences and enforces chronological order."""
    db_file = str(tmp_path / "test_data.db")
    hub = DataHub(db_path=db_file)
    await hub.initialize()

    base_ts = 1700000000
    batch1 = [
        Bar(time=base_ts + 300, open=2000.0, high=2005.0, low=1995.0, close=2002.0, volume=5.0),
        Bar(time=base_ts + 600, open=2002.0, high=2007.0, low=2001.0, close=2006.0, volume=8.0),
    ]
    batch2 = [
        Bar(time=base_ts, open=1995.0, high=2001.0, low=1994.0, close=2000.0, volume=10.0), # Earlier
        Bar(time=base_ts + 300, open=2000.0, high=2005.0, low=1995.0, close=2003.0, volume=7.0), # Duplicate with updated close
        Bar(time=base_ts + 900, open=2006.0, high=2010.0, low=2005.0, close=2009.0, volume=12.0),
    ]

    await hub.inject_bars("XAUUSD", "5m", batch1)
    await hub.inject_bars("XAUUSD", "5m", batch2)

    bars = await hub.get_bars("XAUUSD", "5m", count=10)
    assert len(bars) == 4
    times = [b.time for b in bars]
    assert times == [base_ts, base_ts + 300, base_ts + 600, base_ts + 900]
    # Check update
    assert bars[1].close == 2003.0


@pytest.mark.anyio
async def test_data_hub_completed_only_filtering_excludes_incomplete_higher_tf(tmp_path):
    """Verify higher-timeframe candles (1H, 15m) in-progress are filtered out when completed_only=True."""
    db_file = str(tmp_path / "test_data.db")
    hub = DataHub(db_path=db_file)
    await hub.initialize()

    # 1H candles: Started at 14:00 (1700000000) and 15:00 (1700003600)
    h1_bar1 = Bar(time=1700000000, open=2000.0, high=2010.0, low=1998.0, close=2005.0, volume=50.0)
    h1_bar2 = Bar(time=1700003600, open=2005.0, high=2015.0, low=2004.0, close=2012.0, volume=20.0)

    await hub.inject_bars("XAUUSD", "1H", [h1_bar1, h1_bar2])

    # Case A: At 15:30 (1700003600 + 1800), bar2 is still in-progress (needs 3600s, ends at 1700007200)
    now_ts = 1700003600 + 1800
    completed_bars = await hub.get_bars("XAUUSD", "1H", count=10, completed_only=True, now_ts=now_ts)
    assert len(completed_bars) == 1
    assert completed_bars[0].time == 1700000000

    # Case B: At 16:01 (1700007260), bar2 has completed
    now_ts_completed = 1700007260
    all_completed = await hub.get_bars("XAUUSD", "1H", count=10, completed_only=True, now_ts=now_ts_completed)
    assert len(all_completed) == 2


@pytest.mark.anyio
async def test_data_hub_stale_detection(tmp_path):
    """Verify is_stale returns True when latest candle exceeds allowed age."""
    db_file = str(tmp_path / "test_data.db")
    hub = DataHub(db_path=db_file)
    await hub.initialize()

    current_ts = 1700001000
    # 5m candle from 15 minutes ago (900 seconds ago, 3x timeframe)
    old_bar = Bar(time=current_ts - 1000, open=2000.0, high=2005.0, low=1999.0, close=2002.0)
    await hub.append_bar("XAUUSD", "5m", old_bar)

    assert hub.is_stale("XAUUSD", "5m", max_age_seconds=600, now_ts=current_ts) is True

    # Now add fresh bar from 60 seconds ago
    fresh_bar = Bar(time=current_ts - 60, open=2002.0, high=2004.0, low=2001.0, close=2003.0)
    await hub.append_bar("XAUUSD", "5m", fresh_bar)
    assert hub.is_stale("XAUUSD", "5m", max_age_seconds=600, now_ts=current_ts) is False


def test_synthetic_bars_tagged_in_paper_sim():
    """Verify PaperSimAdapter synthetic bars are explicitly tagged with is_synthetic=True."""
    adapter = PaperSimAdapter()
    bars = adapter._generate_synthetic_bars("XAUUSD", "5m", count=10)
    assert len(bars) == 10
    for b in bars:
        assert b.is_synthetic is True, "Synthetic bar was not tagged with is_synthetic=True"


# ==============================================================================
# Item 14: News Reactor & Economic Calendar Safety Tests
# ==============================================================================

@pytest.mark.anyio
async def test_news_reactor_reports_unavailable_and_does_not_invent_fake_events():
    """Verify that if calendar feed fails, it reports UNAVAILABLE and does NOT make fake events."""
    reactor = NewsReactor(enabled=True)
    
    # Mock a network failure when hitting the calendar URL
    with patch("aiohttp.ClientSession.get", side_effect=Exception("Connection refused")):
        await reactor.refresh_calendar()

    assert reactor.calendar_status == "UNAVAILABLE"
    assert reactor._is_calendar_available is False
    assert len(reactor._all_events) == 0, "Fake calendar events were fabricated when network was offline!"
    assert len(reactor._major_events) == 0


def test_news_reactor_block_on_calendar_unavailable_option():
    """Verify block_on_calendar_unavailable option strictly controls trade allowance."""
    # Subtest A: Safety blocking enabled
    reactor_strict = NewsReactor(enabled=True, block_on_calendar_unavailable=True)
    reactor_strict._is_calendar_available = False
    reactor_strict.calendar_status = "UNAVAILABLE"
    can_open, reason = reactor_strict.can_open_baseline_trade()
    assert can_open is False
    assert "Economic calendar feed unavailable" in reason

    # Subtest B: Safety blocking disabled (proceeds with clear warning)
    reactor_permissive = NewsReactor(enabled=True, block_on_calendar_unavailable=False)
    reactor_permissive._is_calendar_available = False
    reactor_permissive.calendar_status = "UNAVAILABLE"
    can_open_perm, reason_perm = reactor_permissive.can_open_baseline_trade()
    assert can_open_perm is True
    assert "Warning" in reason_perm


def test_high_impact_news_freeze_blocks_trade_during_window():
    """Verify verified HIGH-impact news event blocks trades during the configured 15-minute freeze window."""
    reactor = NewsReactor(
        enabled=True,
        pre_news_freeze_seconds=900,  # 15 minutes
        post_news_freeze_seconds=900, # 15 minutes
    )
    reactor._is_calendar_available = True
    reactor.calendar_status = "AVAILABLE"

    event_time = 1700000000
    high_impact_event = CalendarEvent(
        id="cpi_1",
        title="US Core CPI m/m",
        currency="USD",
        impact="HIGH",
        impact_badge="🔴 HIGH",
        timestamp_utc=event_time,
        local_time_hm="03:30 PM",
        local_time_hms="03:30:00 PM",
        local_date_str="2026-10-14",
    )
    reactor._all_events = [high_impact_event]
    reactor._major_events = [high_impact_event]

    # Test 1: 5 minutes BEFORE release (event_time - 300) -> Must be FROZEN
    with patch("time.time", return_value=event_time - 300):
        state = reactor.check_calendar_state(current_utc=event_time - 300)
        assert state["is_news_freeze"] is True
        can_open, msg = reactor.can_open_baseline_trade()
        assert can_open is False
        assert "High-impact news freeze active" in msg

    # Test 2: 5 minutes AFTER release (event_time + 300) -> Must be FROZEN
    with patch("time.time", return_value=event_time + 300):
        state = reactor.check_calendar_state(current_utc=event_time + 300)
        assert state["is_news_freeze"] is True
        can_open, msg = reactor.can_open_baseline_trade()
        assert can_open is False
        assert "High-impact news freeze active" in msg

    # Test 3: 30 minutes BEFORE release (event_time - 1800) -> OUTSIDE window, should allow trading
    with patch("time.time", return_value=event_time - 1800):
        state = reactor.check_calendar_state(current_utc=event_time - 1800)
        assert state["is_news_freeze"] is False
        can_open, msg = reactor.can_open_baseline_trade()
        assert can_open is True


@pytest.mark.anyio
async def test_no_fake_headlines_seeded_when_rss_offline():
    """Verify fetch_breaking_headlines does not fabricate headlines when RSS fails."""
    reactor = NewsReactor(enabled=True)
    with patch("urllib.request.urlopen", side_effect=Exception("Network down")):
        await reactor.fetch_breaking_headlines()

    assert len(reactor._headlines) == 0
    assert reactor.headlines_status == "UNAVAILABLE"


# ==============================================================================
# Item 15: Indicator NaN Resilience & Historical Matcher Tests
# ==============================================================================

def test_supertrend_nan_resilience_and_safe_output():
    """Verify SuperTrend handles NaNs, sparse history, and empty inputs gracefully."""
    # Empty inputs
    st_empty = TechnicalIndicators.calculate_supertrend([], [], [])
    assert st_empty["valid"] is False
    assert st_empty["direction"] == 0

    # Short history (< period + 2)
    st_short = TechnicalIndicators.calculate_supertrend([10, 11, 12], [9, 10, 11], [10, 11, 12], period=10)
    assert st_short["valid"] is False

    # Inputs containing NaNs
    h = [np.nan, 2005.0] + [2000.0 + i for i in range(25)]
    l = [np.nan, 1995.0] + [1990.0 + i for i in range(25)]
    c = [np.nan, 2002.0] + [1995.0 + i for i in range(25)]
    st_nan = TechnicalIndicators.calculate_supertrend(h, l, c, period=10)
    assert st_nan["valid"] is True
    assert not np.isnan(st_nan["line"])
    assert st_nan["direction"] in [-1, 1]


def test_stoch_rsi_sparse_history_and_nan_handling():
    """Verify StochRSI handles sparse history and NaNs without raising IndexError."""
    # Very short series
    res_short = TechnicalIndicators.calculate_stoch_rsi([100.0, 101.0, 102.0], period=14)
    assert res_short["valid"] is False
    assert res_short["k"] == 50.0

    # Series with initial NaNs (needs >= 32 bars for StochRSI d to compute)
    data = [np.nan, np.nan] + [2000.0 + np.sin(i) * 10 for i in range(45)]
    res = TechnicalIndicators.calculate_stoch_rsi(data, period=14)
    assert res["valid"] is True
    assert not np.isnan(res["k"])
    assert not np.isnan(res["d"])


def test_vwap_kama_zscore_indicators_nan_handling():
    """Verify quantitative indicators (VWAP, KAMA, Z-Score) handle NaNs safely."""
    # Dataframe with NaN rows
    df = pd.DataFrame({
        "high": [np.nan, 2005.0, 2010.0, 2008.0, 2012.0, 2015.0, 2014.0],
        "low": [np.nan, 1995.0, 2000.0, 1998.0, 2002.0, 2005.0, 2004.0],
        "close": [np.nan, 2002.0, 2007.0, 2005.0, 2010.0, 2012.0, 2011.0],
        "volume": [np.nan, 100.0, 150.0, 120.0, 200.0, 250.0, 180.0],
    })

    vwap = TechnicalIndicators.calculate_vwap(df)
    assert vwap["valid"] is True
    assert not np.isnan(vwap["vwap"])

    kama = TechnicalIndicators.calculate_kama(df, window=4)
    assert isinstance(kama, dict)

    zscore = TechnicalIndicators.calculate_zscore(df["close"].values, window=5)
    assert zscore["valid"] is True
    assert not np.isnan(zscore["zscore"])


def test_historical_matcher_returns_zero_on_sparse_history_and_no_match():
    """Verify HistoricalMatcher returns 0% win rate and NO_DATA / NO_MATCH instead of fake 65%."""
    matcher = HistoricalMatcher(target_rr=2.0)
    vec = SetupVector(
        direction="BUY",
        h1_trend="BULLISH",
        m15_adx_tier="STRONG",
        m5_rsi_zone="VALUE_BULL",
        m1_trigger_type="ENGULFING",
        session="LONDON",
    )

    # 1. Sparse history (< 60 bars)
    df_short = pd.DataFrame({
        "time": [1700000000 + i * 300 for i in range(20)],
        "close": [2000.0 + i for i in range(20)],
        "high": [2005.0 + i for i in range(20)],
        "low": [1995.0 + i for i in range(20)],
    })
    res_sparse = matcher.compute_edge(vec, df_short)
    assert res_sparse.sample_count == 0
    assert res_sparse.win_rate_pct == 0.0
    assert res_sparse.confidence_grade == "INSUFFICIENT_DATA"
    assert res_sparse.sample_outcomes == []

    # 2. Sufficient history but 0 pattern matches
    # Create market structure completely opposing the vector (e.g. constant bear trend with high RSI)
    df_no_match = pd.DataFrame({
        "time": [1700000000 + i * 300 for i in range(100)],
        "close": [2000.0 - i * 5 for i in range(100)],
        "high": [2005.0 - i * 5 for i in range(100)],
        "low": [1995.0 - i * 5 for i in range(100)],
    })
    res_no_match = matcher.compute_edge(vec, df_no_match)
    assert res_no_match.sample_count == 0
    assert res_no_match.win_rate_pct == 0.0
    assert res_no_match.confidence_grade == "NO_MATCH"
    assert res_no_match.sample_outcomes == []
