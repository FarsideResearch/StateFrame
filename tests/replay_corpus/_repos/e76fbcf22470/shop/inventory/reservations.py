"""Public stock reservation entry point."""

from typing import Iterable

from .models import Reservation, StockLot, StockRequest
from .selection import allocate, allocations_fit_snapshot, available_quantity


class InsufficientStock(ValueError):
    """The requested quantity is unavailable at dispatch time."""


def _validate_lot_ids(lots: tuple[StockLot, ...]) -> None:
    ids = [lot.lot_id for lot in lots]
    if len(ids) != len(set(ids)):
        raise ValueError("lot ids must be unique in a stock snapshot")


def _validate_allocation(
    reservation: Reservation, snapshot: tuple[StockLot, ...]
) -> None:
    if not reservation.complete:
        raise RuntimeError("allocation did not satisfy a validated request")
    if not allocations_fit_snapshot(reservation.allocations, snapshot):
        raise RuntimeError("allocation exceeded available stock")


def remaining_stock(
    lots: Iterable[StockLot], reservation: Reservation
) -> tuple[StockLot, ...]:
    """Return a new stock snapshot after a reservation is committed."""
    from dataclasses import replace

    used = {item.lot_id: item.quantity for item in reservation.allocations}
    return tuple(
        replace(lot, available=lot.available - used.get(lot.lot_id, 0))
        for lot in lots
    )


def reserve_and_apply(
    request: StockRequest, lots: Iterable[StockLot]
) -> tuple[Reservation, tuple[StockLot, ...]]:
    """Reserve against one snapshot and return its post-order stock state."""
    snapshot = tuple(lots)
    reservation = reserve_stock(request, snapshot)
    return reservation, remaining_stock(snapshot, reservation)


def reserve_stock(request: StockRequest, lots: Iterable[StockLot]) -> Reservation:
    """Reserve complete orders from an immutable stock snapshot."""
    snapshot = tuple(lots)
    _validate_lot_ids(snapshot)
    if available_quantity(request, snapshot) < request.quantity:
        raise InsufficientStock(f"insufficient stock for {request.sku}")
    allocations = allocate(request, snapshot)
    reservation = Reservation(
        sku=request.sku,
        requested=request.quantity,
        allocations=allocations,
    )
    _validate_allocation(reservation, snapshot)
    return reservation
