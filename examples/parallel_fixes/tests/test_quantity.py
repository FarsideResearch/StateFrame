from pricing.quantity import line_total


def test_line_total_includes_quantity():
    assert line_total(3.25, 4) == 13.00
