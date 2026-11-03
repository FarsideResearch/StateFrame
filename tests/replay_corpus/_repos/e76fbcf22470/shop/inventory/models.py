"""Inventory lots and reservation records."""

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
