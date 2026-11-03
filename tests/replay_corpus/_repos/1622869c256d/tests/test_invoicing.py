from decimal import Decimal as D

import pytest

from shop.invoicing.builder import create_invoice
from shop.invoicing.models import InvoiceLine, TaxJurisdiction


TAX = TaxJurisdiction("LOCAL", D("0.05"))


def test_half_cent_tax_rounds_up_on_customer_invoice() -> None:
    invoice = create_invoice("INV-1", [InvoiceLine("Notebook", 1, D("0.10"))], TAX)
    assert (invoice.tax, invoice.total) == (D("0.01"), D("0.11"))


def test_tax_excludes_nontaxable_line_and_shipping() -> None:
    invoice = create_invoice(
        "INV-2",
        [InvoiceLine("Book", 1, D("10.00"), False), InvoiceLine("Pen", 1, D("20.00"))],
        TAX,
        D("4.00"),
    )
    assert (invoice.tax, invoice.total) == (D("1.00"), D("35.00"))


def test_tax_rounds_once_on_aggregate_taxable_amount() -> None:
    invoice = create_invoice(
        "INV-3", [InvoiceLine("A", 1, D("0.06")), InvoiceLine("B", 1, D("0.06"))], TAX
    )
    assert invoice.tax == D("0.01")


def test_fractional_cent_price_is_rejected() -> None:
    with pytest.raises(ValueError, match="cents"):
        create_invoice("INV-4", [InvoiceLine("Pen", 1, D("0.105"))], TAX)
