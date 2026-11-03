"""Reward account and earning records."""

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
