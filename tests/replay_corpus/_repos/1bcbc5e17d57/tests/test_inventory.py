from datetime import date

import pytest

from shop.inventory import StockLot, StockRequest
from shop.inventory import InsufficientStock, reserve_stock


SHIP = date(2026, 11, 5)


def test_reservation_uses_soonest_expiring_lot_first() -> None:
    lots = [
        StockLot("late", "TEA", date(2027, 3, 1), 10),
        StockLot("early", "TEA", date(2026, 12, 1), 10),
    ]
    result = reserve_stock(StockRequest("TEA", 3, SHIP), lots)
    assert [(a.lot_id, a.quantity) for a in result.allocations] == [("early", 3)]


def test_reservation_skips_expired_and_unrelated_lots() -> None:
    lots = [
        StockLot("old", "TEA", SHIP, 9),
        StockLot("other", "COFFEE", date(2027, 1, 1), 9),
        StockLot("valid", "TEA", date(2027, 1, 1), 4),
    ]
    result = reserve_stock(StockRequest("TEA", 4, SHIP), lots)
    assert result.complete
    assert result.allocations[0].lot_id == "valid"


def test_reservation_rejects_short_stock() -> None:
    with pytest.raises(InsufficientStock, match="TEA"):
        reserve_stock(
            StockRequest("TEA", 4, SHIP),
            [StockLot("one", "TEA", date(2027, 1, 1), 3)],
        )


def test_reservation_can_span_multiple_lots() -> None:
    lots = [
        StockLot("a", "TEA", date(2027, 1, 1), 2),
        StockLot("b", "TEA", date(2027, 1, 1), 3),
    ]
    result = reserve_stock(StockRequest("TEA", 5, SHIP), lots)
    assert result.fulfilled == 5
    assert {a.lot_id for a in result.allocations} == {"a", "b"}
