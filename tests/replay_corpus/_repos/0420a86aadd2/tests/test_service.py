from datetime import date
from decimal import Decimal as D

import pytest

from shop.inventory.models import StockLot
from shop.invoicing.models import TaxJurisdiction
from shop.loyalty.models import RewardAccount
from shop.service import CheckoutRequest, checkout
from shop.shipping.models import Destination, Parcel


def order(**changes: object) -> CheckoutRequest:
    values = {
        "order_id": "ORDER-101",
        "sku": "MUG",
        "quantity": 1,
        "unit_price": D("20.00"),
        "ship_on": date(2026, 11, 5),
        "destination": Destination("10001", 1, False),
        "parcel": Parcel(D("2"), D("2"), D("2"), D("1")),
        "jurisdiction": TaxJurisdiction("LOCAL", D("0.05")),
        "account": RewardAccount("customer", "standard"),
    }
    values.update(changes)
    return CheckoutRequest(**values)


def test_checkout_records_complete_receipt() -> None:
    request = order()
    receipt = checkout(
        request, (StockLot("lot-a", "MUG", date(2027, 1, 1), 2),)
    )
    assert receipt.reservation.complete
    assert receipt.shipping.total == D("5.00")
    assert receipt.invoice.total == D("26.00")
    assert receipt.award.total_points == 26
    assert receipt.account.earnings[0].points == 26


def test_checkout_rejects_unavailable_stock() -> None:
    with pytest.raises(ValueError, match="insufficient stock"):
        checkout(order(), ())


def test_checkout_requires_order_id() -> None:
    with pytest.raises(ValueError, match="order id"):
        order(order_id="")
