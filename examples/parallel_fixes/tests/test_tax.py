from pricing.tax import calculate_tax


def test_tax_rate_is_a_percentage():
    assert calculate_tax(100.00, 10) == 10.00
