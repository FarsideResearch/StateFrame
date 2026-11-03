"""Customer reward accounting."""


# --- models ---
from dataclasses import dataclass
from datetime import date
from decimal import Decimal


@dataclass(frozen=True)
class Earning:
    earned_on: date
    points: int
    source: str

    def __post_init__(self) -> None:
        if self.points < 0:
            raise ValueError("earned points cannot be negative")
        if not self.source:
            raise ValueError("earning source is required")


@dataclass(frozen=True)
class RewardAccount:
    customer_id: str
    tier: str
    earnings: tuple[Earning, ...] = ()

    def __post_init__(self) -> None:
        if not self.customer_id:
            raise ValueError("customer id is required")
        if self.tier not in {"standard", "silver", "gold"}:
            raise ValueError("unknown reward tier")

    def with_earning(self, earning: Earning) -> "RewardAccount":
        return RewardAccount(
            customer_id=self.customer_id,
            tier=self.tier,
            earnings=self.earnings + (earning,),
        )

    @property
    def lifetime_points(self) -> int:
        return sum(earning.points for earning in self.earnings)

    @property
    def earning_count(self) -> int:
        return len(self.earnings)


@dataclass(frozen=True)
class Award:
    customer_id: str
    base_points: int
    bonus_points: int
    total_points: int
    purchase_amount: Decimal

    def __post_init__(self) -> None:
        if self.total_points != self.base_points + self.bonus_points:
            raise ValueError("award points do not match components")

    @property
    def has_activity_bonus(self) -> bool:
        return self.bonus_points > 0


# --- policy ---
from datetime import date, timedelta
from decimal import Decimal, ROUND_FLOOR


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


# --- awards ---
from datetime import date
from decimal import Decimal



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
