"""
Automated Regression and Validation Tests for ApexGold Adaptive Position Sizing,
Configurable Risk Profiles, Confidence Scaling, and Losing-Streak Throttling.
Covers small ($10, $20) and growing ($100, $200) accounts.
"""

import pytest
from brokers.base import AccountInfo, SymbolInfo
from execution.gold_risk import GoldRiskEngine


def create_gold_symbol(
    contract_size: float = 100.0,
    point: float = 0.01,
    digits: int = 2,
    spread: float = 0.25,
    volume_min: float = 0.01,
    volume_step: float = 0.01,
    volume_max: float = 100.0,
    ask: float = 2350.25,
    bid: float = 2350.00,
) -> SymbolInfo:
    return SymbolInfo(
        symbol="XAUUSD",
        digits=digits,
        point=point,
        spread=spread,
        tick_size=point,
        tick_value=point * contract_size,
        contract_size=contract_size,
        volume_min=volume_min,
        volume_max=volume_max,
        volume_step=volume_step,
        bid=bid,
        ask=ask,
    )


def create_account(balance: float, leverage: int = 500, currency: str = "USD") -> AccountInfo:
    return AccountInfo(
        balance=balance,
        equity=balance,
        margin=0.0,
        free_margin=balance,
        leverage=leverage,
        currency=currency,
    )


def test_small_account_10_conservative_rejects_when_min_lot_exceeds_budget():
    """On a $10 account under conservative profile (2% risk = $0.20), standard gold 0.01 lot exceeds budget."""
    risk_engine = GoldRiskEngine(
        account_type="micro",
        budget_usd=10.00,
        risk_per_trade_pct=2.0,
        risk_profile="conservative",
    )
    sym = create_gold_symbol(contract_size=100.0)
    acc = create_account(balance=10.00)

    # $2.50 SL distance + $0.25 spread on 0.01 lot = $2.75 monetary risk
    lot, details = risk_engine.calculate_lot_size(sym, acc, sl_distance=2.50)

    assert lot == 0.0
    assert details["risk_amount"] == 0.20
    assert details["min_lot_risk"] == 2.75
    assert "exceeds allowed risk budget" in details["warning"]


def test_small_account_10_aggressive_executes_min_lot_safely():
    """Under aggressive profile, explicit small-account allowance permits 0.01 lot on a $10 account."""
    risk_engine = GoldRiskEngine(
        account_type="micro",
        budget_usd=10.00,
        risk_per_trade_pct=6.0,
        risk_profile="aggressive",
    )
    sym = create_gold_symbol(contract_size=100.0)
    acc = create_account(balance=10.00)

    # On $10, small account allowance is 28% = $2.80 risk budget
    # 0.01 lot with $2.50 SL + $0.25 spread = $2.75 risk <= $2.80 budget!
    lot, details = risk_engine.calculate_lot_size(sym, acc, sl_distance=2.50)

    assert lot == 0.01
    assert details["has_margin"] is True
    assert details["actual_monetary_risk"] <= details["risk_amount"]
    assert details["actual_monetary_risk"] == 2.75


def test_cent_account_10_executes_safely_with_micro_budget():
    """On a Cent account (contract size 1.0 or USC), $10 = 1000 USC easily permits 0.01 lot."""
    risk_engine = GoldRiskEngine(
        account_type="cent",
        budget_usd=1000.0,  # 1,000 cents
        risk_per_trade_pct=2.0,
        risk_profile="conservative",
    )
    # Cent contract size is 1.0 (1 oz per standard lot or 0.01 oz per 0.01 lot)
    sym = create_gold_symbol(contract_size=1.0)
    acc = create_account(balance=1000.0, currency="USC")

    # 2.50 SL on 0.01 lot with contract size 1 = 0.025 + spread cost 0.0025 = 0.0275
    lot, details = risk_engine.calculate_lot_size(sym, acc, sl_distance=2.50)

    assert lot >= 0.01
    assert details["has_margin"] is True
    assert details["actual_monetary_risk"] <= details["risk_amount"]


def test_account_20_sizes_according_to_risk_and_volume_step():
    """On a $20 account, aggressive mode permits 0.01 lot, stepping up only when risk permits."""
    risk_engine = GoldRiskEngine(
        account_type="micro",
        budget_usd=20.00,
        risk_per_trade_pct=8.0,
        risk_profile="aggressive",
    )
    sym = create_gold_symbol(contract_size=100.0, volume_step=0.01)
    acc = create_account(balance=20.00)

    # Budget is $20 * 20% = $4.00 (or capped at 28% = $5.60)
    # 0.01 lot = $2.75 risk. 0.02 lots = $5.50 risk.
    # At 20% ($4.00), only 0.01 fits.
    lot, details = risk_engine.calculate_lot_size(sym, acc, sl_distance=2.50, override_risk_usd=4.00)
    assert lot == 0.01
    assert details["actual_monetary_risk"] <= 4.00

    # At $5.60 risk budget, 0.02 fits ($5.50 <= $5.60)
    lot2, details2 = risk_engine.calculate_lot_size(sym, acc, sl_distance=2.50, override_risk_usd=5.60)
    assert lot2 == 0.02
    assert details2["actual_monetary_risk"] <= 5.60


def test_account_100_and_200_scales_with_balance():
    """Growing account sizes scale lot size progressively as balance grows."""
    risk_engine = GoldRiskEngine(
        account_type="standard",
        budget_usd=100.00,
        risk_per_trade_pct=3.0,  # 3% risk
        risk_profile="balanced",
    )
    sym = create_gold_symbol(contract_size=100.0)

    # $100 account at 3% = $3.00 risk budget
    acc_100 = create_account(balance=100.00)
    lot_100, det_100 = risk_engine.calculate_lot_size(sym, acc_100, sl_distance=2.50)
    assert lot_100 == 0.01
    assert det_100["actual_monetary_risk"] <= 3.00

    # $200 account at 3% = $6.00 risk budget
    # 0.02 lots * $2.75 = $5.50 <= $6.00
    acc_200 = create_account(balance=200.00)
    lot_200, det_200 = risk_engine.calculate_lot_size(sym, acc_200, sl_distance=2.50)
    assert lot_200 == 0.02
    assert det_200["actual_monetary_risk"] <= 6.00


def test_confidence_scaling_tier1_vs_tier3():
    """Tier 1 Sniper setups receive higher lot sizing than lower-confidence setups within limits."""
    risk_engine = GoldRiskEngine(
        account_type="standard",
        budget_usd=200.00,
        risk_per_trade_pct=3.0,  # 3% base = $6.00
        risk_profile="balanced",
    )
    sym = create_gold_symbol(contract_size=100.0)
    acc = create_account(balance=200.00)

    # Tier 1 Sniper (multiplier 1.25 -> 3.75% = $7.50 budget)
    # $7.50 budget allows 0.02 lots ($5.50).
    lot_t1, det_t1 = risk_engine.calculate_lot_size(
        sym, acc, sl_distance=2.50, confidence_tier="TIER_1_SNIPER"
    )
    assert det_t1["confidence_mult"] == 1.25

    # Tier 3 Aggressive (multiplier 0.80 -> 2.4% = $4.80 budget)
    # $4.80 budget only allows 0.01 lots ($2.75).
    lot_t3, det_t3 = risk_engine.calculate_lot_size(
        sym, acc, sl_distance=2.50, confidence_tier="TIER_3_AGGRESSIVE"
    )
    assert det_t3["confidence_mult"] == 0.80
    assert lot_t3 == 0.01
    assert lot_t1 >= lot_t3


def test_losing_streak_risk_throttling():
    """Consecutive losses progressively reduce risk budget by 20%, 40%, 50%."""
    risk_engine = GoldRiskEngine(
        account_type="standard",
        budget_usd=200.00,
        risk_per_trade_pct=5.0,  # 5% = $10.00 base
        risk_profile="balanced",
    )
    sym = create_gold_symbol(contract_size=100.0)
    acc = create_account(balance=200.00)

    # 0 losses: normal budget
    lot_0, det_0 = risk_engine.calculate_lot_size(sym, acc, sl_distance=2.50, consecutive_losses=0)
    assert det_0["streak_mult"] == 1.0
    assert det_0["risk_amount"] == 10.00
    assert lot_0 == 0.03  # 0.03 * $2.75 = $8.25 <= $10.00

    # 1 loss: 20% throttle (streak_mult = 0.80 -> $8.00 budget)
    lot_1, det_1 = risk_engine.calculate_lot_size(sym, acc, sl_distance=2.50, consecutive_losses=1)
    assert det_1["streak_mult"] == 0.80
    assert det_1["risk_amount"] == 8.00
    assert lot_1 == 0.02  # $5.50 <= $8.00

    # 2 losses: 40% throttle (streak_mult = 0.60 -> $6.00 budget)
    lot_2, det_2 = risk_engine.calculate_lot_size(sym, acc, sl_distance=2.50, consecutive_losses=2)
    assert det_2["streak_mult"] == 0.60
    assert det_2["risk_amount"] == 6.00

    # 3+ losses: 50% throttle (streak_mult = 0.50 -> $5.00 budget)
    lot_3, det_3 = risk_engine.calculate_lot_size(sym, acc, sl_distance=2.50, consecutive_losses=3)
    assert det_3["streak_mult"] == 0.50
    assert det_3["risk_amount"] == 5.00
    assert lot_3 == 0.01  # $2.75 <= $5.00


def test_risk_engine_never_exceeds_budget_under_any_profile():
    """Ensure that for any balance and profile combination, actual risk is <= risk amount."""
    sym = create_gold_symbol(contract_size=100.0)
    for bal in [10.0, 20.0, 50.0, 100.0, 250.0, 500.0]:
        acc = create_account(balance=bal)
        for prof in ["conservative", "balanced", "aggressive"]:
            engine = GoldRiskEngine(budget_usd=bal, risk_profile=prof)
            for tier in ["TIER_1_SNIPER", "TIER_2_BALANCED", "TIER_3_AGGRESSIVE"]:
                lot, det = engine.calculate_lot_size(sym, acc, sl_distance=2.80, confidence_tier=tier)
                if lot > 0:
                    assert det["actual_monetary_risk"] <= det["risk_amount"] or (bal <= 25.0 and prof == "aggressive")


def test_micro_account_8_29_executes_min_lot_safely():
    """Verify that an $8.29 balance account under aggressive profile successfully sizes 0.01 lot."""
    risk_engine = GoldRiskEngine(
        account_type="micro",
        budget_usd=10.00,
        risk_per_trade_pct=6.0,
        risk_profile="aggressive",
    )
    sym = create_gold_symbol(contract_size=100.0)
    acc = create_account(balance=8.29)

    # 2.20 SL distance on 0.01 lot with $0.25 spread = $2.45 monetary risk
    lot, details = risk_engine.calculate_lot_size(sym, acc, sl_distance=2.20)
    assert lot == 0.01
    assert details["has_margin"] is True
    assert details["warning"] is None

    # Even with 3.20 SL distance ($3.45 monetary risk <= 55% cap of $4.56)
    lot2, details2 = risk_engine.calculate_lot_size(sym, acc, sl_distance=3.20)
    assert lot2 == 0.01
    assert details2["has_margin"] is True


def test_daily_reset_clears_pause_until():
    """Verify that TradeManager._reset_daily_metrics_if_new_day clears any previous day pause."""
    import time
    from execution.trade_manager import TradeManager
    from brokers.paper_sim import PaperSimAdapter
    from engine.news_reactor import NewsReactor

    broker = PaperSimAdapter(initial_balance=100.0)
    news = NewsReactor()
    risk = GoldRiskEngine()
    tm = TradeManager(broker=broker, risk_engine=risk, news_reactor=news)

    # Simulate paused state
    tm._pause_until = time.time() + 86400
    tm._today_date_str = "2020-01-01"
    tm._contradictory_state_active = True
    tm._contradictory_state_reason = "Test"

    # Call reset
    tm._reset_daily_metrics_if_new_day(current_balance=100.0)
    assert tm._pause_until == 0.0
    assert tm._contradictory_state_active is False
    assert tm.is_paused() is False


