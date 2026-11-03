"""Invoice line items and tax jurisdiction values."""

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
