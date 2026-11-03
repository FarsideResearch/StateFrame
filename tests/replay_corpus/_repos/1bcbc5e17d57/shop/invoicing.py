"""Customer invoice creation."""


# --- models ---
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class InvoiceLine:
    description: str
    quantity: int
    unit_price: Decimal
    taxable: bool = True

    def __post_init__(self) -> None:
        if not self.description:
            raise ValueError("description is required")
        if self.quantity < 1 or self.unit_price < 0:
            raise ValueError("invalid invoice line")

    @property
    def amount(self) -> Decimal:
        return self.unit_price * self.quantity

    @property
    def display_quantity(self) -> str:
        return f"{self.quantity} × {self.description}"


@dataclass(frozen=True)
class TaxJurisdiction:
    code: str
    rate: Decimal

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("jurisdiction code is required")
        if not Decimal("0") <= self.rate <= Decimal("1"):
            raise ValueError("tax rate must be between zero and one")


@dataclass(frozen=True)
class Invoice:
    number: str
    merchandise: Decimal
    shipping: Decimal
    tax: Decimal
    total: Decimal

    def __post_init__(self) -> None:
        if not self.number:
            raise ValueError("invoice number is required")
        if min(self.merchandise, self.shipping, self.tax) < 0:
            raise ValueError("invoice amounts cannot be negative")
        if self.total != self.merchandise + self.shipping + self.tax:
            raise ValueError("invoice total does not match components")

    @property
    def pretax_total(self) -> Decimal:
        return self.merchandise + self.shipping

    @property
    def is_tax_free(self) -> bool:
        return self.tax == 0


# --- taxes ---
from decimal import Decimal, ROUND_HALF_EVEN, ROUND_HALF_UP


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
    return money(raw_tax)


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


# --- builder ---
from decimal import Decimal
from typing import Iterable



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
