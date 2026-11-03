"""Public invoice builder."""

from decimal import Decimal
from typing import Iterable

from .models import Invoice, InvoiceLine, TaxJurisdiction
from .taxes import (
    calculate_tax,
    merchandise_amount,
    money,
    validate_currency_lines,
    validate_tax_base,
)


def _lines(lines: Iterable[InvoiceLine]) -> tuple[InvoiceLine, ...]:
    result = tuple(lines)
    if not result:
        raise ValueError("invoice requires a merchandise line")
    validate_currency_lines(result)
    validate_tax_base(result)
    return result


def _validate_shipping(shipping: Decimal) -> None:
    if not shipping.is_finite():
        raise ValueError("shipping must be finite")
    if shipping < 0 or shipping != money(shipping):
        raise ValueError("shipping must be a nonnegative cent amount")


def _validate_number(number: str) -> None:
    if not number.strip():
        raise ValueError("invoice number is required")
    if number != number.strip():
        raise ValueError("invoice number cannot have surrounding whitespace")


def create_invoice(
    number: str,
    lines: Iterable[InvoiceLine],
    jurisdiction: TaxJurisdiction,
    shipping: Decimal = Decimal("0.00"),
) -> Invoice:
    """Produce an immutable invoice from completed checkout charges."""
    _validate_number(number)
    _validate_shipping(shipping)
    items = _lines(lines)
    merchandise = money(merchandise_amount(items))
    tax = calculate_tax(items, jurisdiction)
    return Invoice(
        number=number,
        merchandise=merchandise,
        shipping=shipping,
        tax=tax,
        total=merchandise + shipping + tax,
    )


def invoice_balance(invoice: Invoice, paid: Decimal) -> Decimal:
    """Calculate the unpaid amount for a payment or refund display."""
    if paid < 0:
        raise ValueError("paid amount cannot be negative")
    if paid != money(paid):
        raise ValueError("paid amount must use cents")
    return invoice.total - paid
