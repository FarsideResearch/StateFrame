"""Tier multipliers and rolling activity rules."""

from datetime import date, timedelta
from decimal import Decimal, ROUND_FLOOR

from .models import Earning, RewardAccount

WINDOW_DAYS = 30
ACTIVE_THRESHOLD = 100
MULTIPLIERS = {
    "standard": Decimal("1"),
    "silver": Decimal("1.5"),
    "gold": Decimal("2"),
}


def tier_multiplier(tier: str) -> Decimal:
    """Look up a validated account tier's earning rate."""
    return MULTIPLIERS[tier]


def whole_points(amount: Decimal) -> int:
    """Spend earns points only for whole currency units."""
    return int(amount.to_integral_value(rounding=ROUND_FLOOR))


def recent_earnings(earnings: tuple[Earning, ...], on: date) -> tuple[Earning, ...]:
    """Include the full 30-day lookback period, including its first day."""
    start = on - timedelta(days=WINDOW_DAYS)
    return tuple(item for item in earnings if start < item.earned_on <= on)


def activity_points(account: RewardAccount, on: date) -> int:
    return sum(item.points for item in recent_earnings(account.earnings, on))


def base_award(amount: Decimal, tier: str) -> int:
    return int(
        (Decimal(whole_points(amount)) * tier_multiplier(tier)).to_integral_value(
            rounding=ROUND_FLOOR
        )
    )


def activity_bonus(account: RewardAccount, on: date, base: int) -> int:
    """Active customers receive a quarter of their base award."""
    recent_points = sum(item.points for item in recent_earnings(account.earnings, on))
    if recent_points < ACTIVE_THRESHOLD:
        return 0
    return base // 4


def validate_earning_dates(account: RewardAccount, on: date) -> None:
    """A ledger cannot use future activity to qualify for today's bonus."""
    if any(item.earned_on > on for item in account.earnings):
        raise ValueError("reward ledger contains future earnings")


def next_activity_threshold(account: RewardAccount, on: date) -> int:
    """Points needed before the current lookback period earns a bonus."""
    return max(0, ACTIVE_THRESHOLD - activity_points(account, on))
