"""Public pricing entry point."""

from decimal import Decimal
from typing import Iterable

from .models import Coupon, PriceBreakdown, PriceLine
from .rules import best_coupon, cart_subtotal, money, net_total


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
