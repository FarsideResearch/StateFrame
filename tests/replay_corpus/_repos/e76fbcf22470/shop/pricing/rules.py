"""Promotion selection and monetary rounding rules."""

from decimal import Decimal, ROUND_HALF_UP

from .models import Coupon, PriceLine

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
    return subtotal > coupon.minimum_spend


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
