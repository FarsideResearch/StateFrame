from pricing.checkout import total_with_tax


def test_checkout_adds_tax_from_the_declared_interface():
    assert total_with_tax(100.00, 10) == 110.00
