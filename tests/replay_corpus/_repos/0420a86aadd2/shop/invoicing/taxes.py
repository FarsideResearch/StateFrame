"""Jurisdiction tax calculation and money rounding."""

from decimal import Decimal, ROUND_HALF_EVEN, ROUND_HALF_UP

from .models import InvoiceLine, TaxJurisdiction

CENT = Decimal("0.01")


def money(amount: Decimal) -> Decimal:
    """Round charges using ordinary commercial half-up rounding."""
    return amount.quantize(CENT, rounding=ROUND_HALF_UP)


def taxable_amount(lines: tuple[InvoiceLine, ...]) -> Decimal:
    """Shipping is excluded and only taxable merchandise is included."""
    return sum((line.amount for line in lines if is_taxable(line)), Decimal("0"))


def is_taxable(line: InvoiceLine) -> bool:
    """Keep exemption checks in one place for summaries and calculation."""
    return line.taxable


def merchandise_amount(lines: tuple[InvoiceLine, ...]) -> Decimal:
    return sum((line.amount for line in lines), Decimal("0"))


def calculate_tax(
    lines: tuple[InvoiceLine, ...], jurisdiction: TaxJurisdiction
) -> Decimal:
    """Round once after applying a jurisdiction rate to taxable goods."""
    raw_tax = taxable_amount(lines) * jurisdiction.rate
    return raw_tax.quantize(CENT, rounding=ROUND_HALF_EVEN)


def validate_currency_lines(lines: tuple[InvoiceLine, ...]) -> None:
    """Reject fractional cent prices before recording an invoice."""
    for line in lines:
        if line.unit_price != money(line.unit_price):
            raise ValueError("invoice line prices must use cents")


def taxable_line_count(lines: tuple[InvoiceLine, ...]) -> int:
    """Expose which line items are within the jurisdiction's tax base."""
    return sum(1 for line in lines if is_taxable(line))


def validate_tax_base(lines: tuple[InvoiceLine, ...]) -> None:
    """Guard arithmetic assumptions before a tax amount is computed."""
    if any(line.amount < 0 for line in lines):
        raise ValueError("taxable amounts cannot be negative")
    if any(not line.amount.is_finite() for line in lines):
        raise ValueError("invoice amounts must be finite")


def line_tax_preview(
    line: InvoiceLine, jurisdiction: TaxJurisdiction
) -> Decimal:
    """Display estimate only; invoice tax rounds after aggregation."""
    if not line.taxable:
        return Decimal("0.00")
    return money(line.amount * jurisdiction.rate)
