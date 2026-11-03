from decimal import Decimal as D

import pytest

from shop.pricing import price_order
from shop.pricing import Coupon, PriceLine


def test_coupon_applies_at_exact_minimum_spend() -> None:
    result = price_order(
        [PriceLine("MUG", D("20.00"), 1)],
        [Coupon("WELCOME", D("20.00"), D("10"))],
    )
    assert (result.discount, result.total, result.applied_coupon) == (
        D("2.00"), D("18.00"), "WELCOME"
    )


def test_below_threshold_does_not_discount() -> None:
    result = price_order(
        [PriceLine("MUG", D("19.99"), 1)],
        [Coupon("WELCOME", D("20.00"), D("10"))],
    )
    assert result.discount == D("0.00")
    assert result.applied_coupon is None


def test_best_coupon_respects_eligible_skus() -> None:
    result = price_order(
        [PriceLine("MUG", D("30.00"), 1), PriceLine("BOOK", D("20.00"), 1)],
        [
            Coupon("MUGS", D("10"), D("20"), frozenset({"MUG"})),
            Coupon("ALL", D("10"), D("10")),
        ],
    )
    assert result.applied_coupon == "MUGS"
    assert result.total == D("44.00")


def test_duplicate_coupon_codes_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        price_order(
            [PriceLine("MUG", D("5.00"), 1)],
            [Coupon("X", D("0"), D("5")), Coupon("X", D("0"), D("10"))],
        )
