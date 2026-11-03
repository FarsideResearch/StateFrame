"""Value objects used by the pricing calculation."""

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class PriceLine:
    sku: str
    unit_price: Decimal
    quantity: int

    def __post_init__(self) -> None:
        if not self.sku:
            raise ValueError("sku is required")
        if self.unit_price < 0:
            raise ValueError("unit price cannot be negative")
        if self.quantity < 1:
            raise ValueError("quantity must be positive")

    @property
    def extended_price(self) -> Decimal:
        return self.unit_price * self.quantity

    @property
    def label(self) -> str:
        """A stable cart-line label for receipts and diagnostics."""
        return f"{self.quantity} × {self.sku}"


@dataclass(frozen=True)
class Coupon:
    code: str
    minimum_spend: Decimal
    percentage: Decimal
    eligible_skus: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("coupon code is required")
        if self.minimum_spend < 0:
            raise ValueError("minimum spend cannot be negative")
        if not Decimal("0") <= self.percentage <= Decimal("100"):
            raise ValueError("percentage must be between zero and 100")

    def applies_to(self, sku: str) -> bool:
        return not self.eligible_skus or sku in self.eligible_skus


@dataclass(frozen=True)
class PriceBreakdown:
    subtotal: Decimal
    discount: Decimal
    total: Decimal
    applied_coupon: str | None

    def __post_init__(self) -> None:
        if self.discount < 0 or self.discount > self.subtotal:
            raise ValueError("invalid discount")
        if self.total != self.subtotal - self.discount:
            raise ValueError("total does not match components")

    @property
    def has_discount(self) -> bool:
        return self.discount > 0
