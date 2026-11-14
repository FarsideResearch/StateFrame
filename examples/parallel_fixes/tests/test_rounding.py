from pricing.rounding import round_cents


def test_rounding_keeps_cents():
    assert round_cents(10.129) == 10.13
