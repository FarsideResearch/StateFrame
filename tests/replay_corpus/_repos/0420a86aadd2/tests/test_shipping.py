from decimal import Decimal as D

import pytest

from shop.shipping.models import Destination, Parcel
from shop.shipping.quotes import quote_shipping


BUSINESS = Destination("10001", zone=2, residential=False)


def test_bulky_light_carton_uses_carrier_dimensional_divisor() -> None:
    quote = quote_shipping(BUSINESS, [Parcel(D("12"), D("12"), D("10"), D("2"))])
    assert quote.billable_pounds == 9
    assert quote.total == D("15.35")


def test_dense_carton_uses_actual_weight() -> None:
    quote = quote_shipping(BUSINESS, [Parcel(D("4"), D("4"), D("4"), D("5.2"))])
    assert quote.billable_pounds == 6
    assert quote.total == D("11.75")


def test_residential_express_has_surcharge() -> None:
    destination = Destination("90210", zone=1, residential=True)
    quote = quote_shipping(
        destination, [Parcel(D("2"), D("2"), D("2"), D("1"))], "express"
    )
    assert quote.total == D("14.50")


def test_empty_shipment_is_rejected() -> None:
    with pytest.raises(ValueError, match="parcel"):
        quote_shipping(BUSINESS, [])
