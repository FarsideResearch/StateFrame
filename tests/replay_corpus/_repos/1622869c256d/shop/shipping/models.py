"""Parcel, destination, and rate quote values."""

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class Parcel:
    length_in: Decimal
    width_in: Decimal
    height_in: Decimal
    weight_lb: Decimal

    def __post_init__(self) -> None:
        if min(self.length_in, self.width_in, self.height_in, self.weight_lb) <= 0:
            raise ValueError("parcel dimensions and weight must be positive")

    @property
    def volume_cubic_in(self) -> Decimal:
        return self.length_in * self.width_in * self.height_in

    @property
    def longest_side_in(self) -> Decimal:
        return max(self.length_in, self.width_in, self.height_in)

    @property
    def girth_in(self) -> Decimal:
        sides = sorted((self.length_in, self.width_in, self.height_in))
        return 2 * (sides[0] + sides[1])


@dataclass(frozen=True)
class Destination:
    postal_code: str
    zone: int
    residential: bool = True

    def __post_init__(self) -> None:
        if not self.postal_code.strip():
            raise ValueError("postal code is required")
        if not 1 <= self.zone <= 8:
            raise ValueError("zone must be between one and eight")


@dataclass(frozen=True)
class ShippingQuote:
    service_level: str
    billable_pounds: int
    base_charge: Decimal
    residential_surcharge: Decimal
    total: Decimal

    def __post_init__(self) -> None:
        if self.billable_pounds < 1:
            raise ValueError("billable pounds must be positive")
        if self.total != self.base_charge + self.residential_surcharge:
            raise ValueError("shipping total does not match components")

    @property
    def has_surcharge(self) -> bool:
        return self.residential_surcharge > 0
