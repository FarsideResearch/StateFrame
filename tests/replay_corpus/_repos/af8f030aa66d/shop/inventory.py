"""Warehouse reservation workflow."""


# --- models ---
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class StockLot:
    lot_id: str
    sku: str
    expires_on: date
    available: int

    def __post_init__(self) -> None:
        if not self.lot_id or not self.sku:
            raise ValueError("lot id and sku are required")
        if self.available < 0:
            raise ValueError("available quantity cannot be negative")

    def is_usable_on(self, ship_on: date) -> bool:
        """A lot expiring on dispatch day is not safe to ship."""
        return self.available > 0 and self.expires_on > ship_on


@dataclass(frozen=True)
class StockRequest:
    sku: str
    quantity: int
    ship_on: date

    def __post_init__(self) -> None:
        if not self.sku:
            raise ValueError("sku is required")
        if self.quantity < 1:
            raise ValueError("quantity must be positive")


@dataclass(frozen=True)
class Allocation:
    lot_id: str
    quantity: int

    def __post_init__(self) -> None:
        if not self.lot_id or self.quantity < 1:
            raise ValueError("allocation must name a lot and positive quantity")


@dataclass(frozen=True)
class Reservation:
    sku: str
    requested: int
    allocations: tuple[Allocation, ...]

    @property
    def fulfilled(self) -> int:
        return sum(item.quantity for item in self.allocations)

    @property
    def complete(self) -> bool:
        return self.fulfilled == self.requested

    @property
    def lot_count(self) -> int:
        return len(self.allocations)


# --- selection ---
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


# --- reservations ---
from typing import Iterable



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
