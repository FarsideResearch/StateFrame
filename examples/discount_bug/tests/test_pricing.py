import pytest

from shop.pricing import apply_discount, order_total


def test_no_discount_keeps_price():
    assert apply_discount(80.0, 0) == 80.0


def test_quarter_off():
    assert apply_discount(80.0, 25) == 60.0


def test_full_discount_is_free():
    assert apply_discount(19.99, 100) == 0.0


def test_order_total_applies_discount_once():
    assert order_total([10.0, 30.0], 10) == 36.0


def test_rejects_out_of_range_percent():
    with pytest.raises(ValueError):
        apply_discount(10.0, 120)
