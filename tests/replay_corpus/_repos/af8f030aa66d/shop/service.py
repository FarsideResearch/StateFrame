"""Checkout composition boundary for independent domain packages."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from .inventory import Reservation, StockLot, StockRequest
from .inventory import reserve_stock
from .invoicing import create_invoice
from .invoicing import Invoice, InvoiceLine, TaxJurisdiction
from .loyalty import award_points, record_award
from .loyalty import Award, RewardAccount
from .pricing import price_order
from .pricing import Coupon, PriceBreakdown, PriceLine
from .shipping import Destination, Parcel, ShippingQuote
from .shipping import quote_shipping


@dataclass(frozen=True)
class CheckoutRequest:
    order_id: str
    sku: str
    quantity: int
    unit_price: Decimal
    ship_on: date
    destination: Destination
    parcel: Parcel
    jurisdiction: TaxJurisdiction
    account: RewardAccount
    coupon: Coupon | None = None

    def __post_init__(self) -> None:
        if not self.order_id:
            raise ValueError("order id is required")


@dataclass(frozen=True)
class CheckoutReceipt:
    price: PriceBreakdown
    reservation: Reservation
    shipping: ShippingQuote
    invoice: Invoice
    award: Award
    account: RewardAccount


def checkout(request: CheckoutRequest, lots: tuple[StockLot, ...]) -> CheckoutReceipt:
    """Price, reserve, ship, invoice, and record an order."""
    price = price_order(
        (PriceLine(request.sku, request.unit_price, request.quantity),),
        (request.coupon,) if request.coupon else (),
    )
    reservation = reserve_stock(
        StockRequest(request.sku, request.quantity, request.ship_on), lots
    )
    shipping = quote_shipping((request.parcel,), request.destination)
    invoice = create_invoice(
        request.order_id,
        (InvoiceLine("Merchandise", 1, price.total),),
        request.jurisdiction,
        shipping.total,
    )
    award = award_points(request.account, invoice.total, request.ship_on)
    updated_account = record_award(request.account, award, request.ship_on)
    return CheckoutReceipt(price, reservation, shipping, invoice, award, updated_account)
