"""Order pricing and promotions."""


# --- models ---
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


# --- rules ---
from decimal import Decimal, ROUND_HALF_UP


CENT = Decimal("0.01")


def money(value: Decimal) -> Decimal:
    """Round a nonnegative amount at the transaction boundary."""
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def eligible_subtotal(lines: tuple[PriceLine, ...], coupon: Coupon) -> Decimal:
    """Find the portion of a cart a coupon may discount."""
    if not coupon.eligible_skus:
        return sum((line.extended_price for line in lines), Decimal("0"))
    return sum(
        (line.extended_price for line in lines if coupon.applies_to(line.sku)),
        Decimal("0"),
    )


def cart_subtotal(lines: tuple[PriceLine, ...]) -> Decimal:
    """Keep threshold evaluation independent of eligible item filtering."""
    return sum((line.extended_price for line in lines), Decimal("0"))


def meets_minimum(subtotal: Decimal, coupon: Coupon) -> bool:
    """A threshold is inclusive in the published promotion terms."""
    return subtotal >= coupon.minimum_spend


def coupon_discount(lines: tuple[PriceLine, ...], coupon: Coupon) -> Decimal:
    """Compute the saving for an offer already known to meet its threshold."""
    eligible = eligible_subtotal(lines, coupon)
    return money(eligible * coupon.percentage / Decimal("100"))


def best_coupon(
    lines: tuple[PriceLine, ...], coupons: tuple[Coupon, ...]
) -> tuple[str | None, Decimal]:
    """Choose the largest saving; stable input order breaks ties."""
    selected: str | None = None
    saving = Decimal("0.00")
    for coupon in coupons:
        if not meets_minimum(cart_subtotal(lines), coupon):
            continue
        candidate = coupon_discount(lines, coupon)
        if candidate > saving:
            selected, saving = coupon.code, candidate
    return selected, saving


def net_total(lines: tuple[PriceLine, ...], discount: Decimal) -> Decimal:
    """Validate a promotion cannot produce a negative payable amount."""
    subtotal = money(cart_subtotal(lines))
    if discount < 0 or discount > subtotal:
        raise ValueError("discount exceeds cart subtotal")
    return subtotal - discount


# --- engine ---
from decimal import Decimal
from typing import Iterable



def _materialize_lines(lines: Iterable[PriceLine]) -> tuple[PriceLine, ...]:
    result = tuple(lines)
    if not result:
        raise ValueError("at least one price line is required")
    return result


def _materialize_coupons(coupons: Iterable[Coupon]) -> tuple[Coupon, ...]:
    result = tuple(coupons)
    codes = [coupon.code for coupon in result]
    if len(codes) != len(set(codes)):
        raise ValueError("coupon codes must be unique")
    return result


def _validate_cart_currency(lines: tuple[PriceLine, ...]) -> None:
    """Checkout prices must already be expressed in whole cents."""
    for line in lines:
        if line.unit_price != money(line.unit_price):
            raise ValueError(f"fractional-cent price for {line.sku}")


def _validate_coupon_codes(coupons: tuple[Coupon, ...]) -> None:
    """Avoid ambiguous receipts when two offers share a public code."""
    for coupon in coupons:
        if coupon.code != coupon.code.strip():
            raise ValueError("coupon codes cannot have surrounding whitespace")


def _breakdown(
    cart: tuple[PriceLine, ...],
    coupon_code: str | None,
    discount: Decimal,
) -> PriceBreakdown:
    """Create one internally consistent monetary result."""
    return PriceBreakdown(
        subtotal=money(cart_subtotal(cart)),
        discount=discount,
        total=net_total(cart, discount),
        applied_coupon=coupon_code,
    )


def price_order(
    lines: Iterable[PriceLine], coupons: Iterable[Coupon] = ()
) -> PriceBreakdown:
    """Price a cart and apply the single most valuable eligible coupon."""
    cart = _materialize_lines(lines)
    offers = _materialize_coupons(coupons)
    _validate_cart_currency(cart)
    _validate_coupon_codes(offers)
    coupon_code, discount = best_coupon(cart, offers)
    return _breakdown(cart, coupon_code, discount)
