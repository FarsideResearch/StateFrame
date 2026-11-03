"""Public reward award entry point."""

from datetime import date
from decimal import Decimal

from .models import Award, Earning, RewardAccount
from .policy import activity_bonus, base_award, validate_earning_dates


def _validate_purchase(account: RewardAccount, amount: Decimal, on: date) -> None:
    if amount.is_nan() or amount.is_infinite():
        raise ValueError("purchase amount must be finite")
    if amount < 0:
        raise ValueError("purchase amount cannot be negative")
    validate_earning_dates(account, on)


def award_points(account: RewardAccount, amount: Decimal, on: date) -> Award:
    """Calculate points for a completed purchase without changing the account."""
    _validate_purchase(account, amount, on)
    base = base_award(amount, account.tier)
    bonus = activity_bonus(account, on, base)
    return Award(
        customer_id=account.customer_id,
        base_points=base,
        bonus_points=bonus,
        total_points=base + bonus,
        purchase_amount=amount,
    )


def record_award(account: RewardAccount, award: Award, on: date) -> RewardAccount:
    """Return a new account ledger after a successful checkout."""
    if award.customer_id != account.customer_id:
        raise ValueError("award belongs to another customer")
    entry = Earning(earned_on=on, points=award.total_points, source="purchase")
    return account.with_earning(entry)


def award_and_record(
    account: RewardAccount, amount: Decimal, on: date
) -> tuple[Award, RewardAccount]:
    """Convenience operation for a paid order and its new ledger state."""
    award = award_points(account, amount, on)
    return award, record_award(account, award, on)


def points_for_orders(
    account: RewardAccount, amounts: tuple[Decimal, ...], on: date
) -> tuple[Award, ...]:
    """Evaluate purchases in order so later ones see earlier activity."""
    current = account
    awards: list[Award] = []
    for amount in amounts:
        award, current = award_and_record(current, amount, on)
        awards.append(award)
    return tuple(awards)


def current_points(account: RewardAccount) -> int:
    """Return the complete recorded earning history for display."""
    return account.lifetime_points
