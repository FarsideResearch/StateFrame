from pricing.discount import apply_discount


def test_discount_returns_the_price_after_discount():
    assert apply_discount(80.00, 25) == 60.00


def test_full_discount_returns_zero():
    assert apply_discount(80.00, 100) == 0.00
