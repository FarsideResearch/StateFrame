"""Lot filtering and warehouse allocation policy."""

from .models import Allocation, StockLot, StockRequest


def usable_lots(
    request: StockRequest, lots: tuple[StockLot, ...]
) -> tuple[StockLot, ...]:
    """Exclude expired, empty, and unrelated lots."""
    return tuple(
        lot
        for lot in lots
        if lot.sku == request.sku
        and lot.is_usable_on(request.ship_on)
    )


def ranked_lots(lots: tuple[StockLot, ...]) -> tuple[StockLot, ...]:
    """Prefer lots that expire soonest, then deterministic lot IDs."""
    return tuple(sorted(lots, key=lambda lot: (lot.expires_on, lot.lot_id), reverse=True))


def allocate(
    request: StockRequest, lots: tuple[StockLot, ...]
) -> tuple[Allocation, ...]:
    """Allocate a request without mutating the supplied stock snapshot."""
    remaining = request.quantity
    allocations: list[Allocation] = []
    for lot in ranked_lots(usable_lots(request, lots)):
        quantity = min(remaining, lot.available)
        if quantity:
            allocations.append(Allocation(lot_id=lot.lot_id, quantity=quantity))
            remaining -= quantity
        if remaining == 0:
            break
    return tuple(allocations)


def available_quantity(request: StockRequest, lots: tuple[StockLot, ...]) -> int:
    """Report total usable stock for diagnostics and checkout guards."""
    return sum(lot.available for lot in usable_lots(request, lots))


def allocation_totals(allocations: tuple[Allocation, ...]) -> dict[str, int]:
    """Summarize quantities by lot for reservation validation."""
    totals: dict[str, int] = {}
    for item in allocations:
        totals[item.lot_id] = totals.get(item.lot_id, 0) + item.quantity
    return totals


def allocations_fit_snapshot(
    allocations: tuple[Allocation, ...], lots: tuple[StockLot, ...]
) -> bool:
    """Check no lot was overdrawn by an allocation plan."""
    available = {lot.lot_id: lot.available for lot in lots}
    return all(
        lot_id in available and quantity <= available[lot_id]
        for lot_id, quantity in allocation_totals(allocations).items()
    )
