from datetime import date
from decimal import Decimal as D

import pytest

from shop.loyalty import award_points, record_award
from shop.loyalty import Earning, RewardAccount


TODAY = date(2026, 11, 1)


def test_activity_on_first_day_of_window_earns_bonus() -> None:
    account = RewardAccount("customer", "standard", (Earning(date(2026, 10, 2), 100, "purchase"),))
    award = award_points(account, D("40.00"), TODAY)
    assert (award.base_points, award.bonus_points, award.total_points) == (40, 10, 50)


def test_activity_before_window_does_not_earn_bonus() -> None:
    account = RewardAccount("customer", "standard", (Earning(date(2026, 10, 1), 100, "purchase"),))
    assert award_points(account, D("40.00"), TODAY).bonus_points == 0


def test_gold_multiplier_truncates_partial_currency_units() -> None:
    account = RewardAccount("customer", "gold")
    award = award_points(account, D("10.99"), TODAY)
    assert (award.base_points, award.total_points) == (20, 20)


def test_recording_award_keeps_original_account_unchanged() -> None:
    account = RewardAccount("customer", "silver")
    award = award_points(account, D("20.00"), TODAY)
    updated = record_award(account, award, TODAY)
    assert account.earnings == ()
    assert updated.earnings[0].points == 30


def test_negative_purchase_is_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        award_points(RewardAccount("customer", "standard"), D("-1"), TODAY)
