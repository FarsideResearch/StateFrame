"""Parcel rate calculation."""


# --- models ---
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


# --- rates ---
from decimal import Decimal, ROUND_CEILING


DIMENSIONAL_DIVISOR = Decimal("166")
ZONE_INCREMENT = Decimal("0.75")
RESIDENTIAL_FEE = Decimal("2.50")
MAX_LENGTH_IN = Decimal("48")
MAX_LENGTH_AND_GIRTH_IN = Decimal("108")


def pounds_rounded_up(pounds: Decimal) -> int:
    return int(pounds.to_integral_value(rounding=ROUND_CEILING))


def billable_weight(parcel: Parcel) -> int:
    """Carrier bills the higher of actual and dimensional weight."""
    dimensional = parcel.volume_cubic_in / DIMENSIONAL_DIVISOR
    return max(pounds_rounded_up(parcel.weight_lb), pounds_rounded_up(dimensional))


def combined_billable_weight(parcels: tuple[Parcel, ...]) -> int:
    """Each carton rounds separately before shipment charges are summed."""
    return sum(billable_weight(parcel) for parcel in parcels)


def base_rate(zone: int, pounds: int, service_level: str) -> Decimal:
    """Small deterministic rate table suitable for domestic checkout."""
    if service_level == "ground":
        opening, additional = Decimal("5.00"), Decimal("1.20")
    elif service_level == "express":
        opening, additional = Decimal("12.00"), Decimal("2.80")
    else:
        raise ValueError("unsupported service level")
    return opening + (zone - 1) * ZONE_INCREMENT + (pounds - 1) * additional


def residential_fee(destination: Destination) -> Decimal:
    return RESIDENTIAL_FEE if destination.residential else Decimal("0.00")


def validate_carrier_limits(parcel: Parcel) -> None:
    """Reject cartons the small-parcel carrier will not accept."""
    if parcel.longest_side_in > MAX_LENGTH_IN:
        raise ValueError("parcel exceeds maximum length")
    if parcel.longest_side_in + parcel.girth_in > MAX_LENGTH_AND_GIRTH_IN:
        raise ValueError("parcel exceeds combined length and girth")


def shipment_surcharge(destination: Destination, carton_count: int) -> Decimal:
    """Residential surcharge applies once per shipment, not carton."""
    if carton_count < 1:
        raise ValueError("shipment must contain at least one carton")
    return residential_fee(destination)


def zone_distance(destination: Destination) -> int:
    """Count zone steps from the origin zone for rate explanations."""
    return destination.zone - 1


# --- quotes ---
from typing import Iterable



def _validate_parcels(parcels: Iterable[Parcel]) -> tuple[Parcel, ...]:
    cartons = tuple(parcels)
    if not cartons:
        raise ValueError("at least one parcel is required")
    for carton in cartons:
        validate_carrier_limits(carton)
    return cartons


def _validate_service_level(service_level: str) -> str:
    """Normalize display input before asking the rate table for a price."""
    normalized = service_level.strip().lower()
    if normalized not in {"ground", "express"}:
        raise ValueError("unsupported service level")
    return normalized


def compare_services(
    destination: Destination, parcels: Iterable[Parcel]
) -> tuple[ShippingQuote, ShippingQuote]:
    """Offer both rates for a shipment without consuming a generator twice."""
    cartons = _validate_parcels(parcels)
    return (
        quote_shipping(destination, cartons, "ground"),
        quote_shipping(destination, cartons, "express"),
    )


def quote_shipping(
    destination: Destination,
    parcels: Iterable[Parcel],
    service_level: str = "ground",
) -> ShippingQuote:
    """Quote a domestic shipment from a destination and its cartons."""
    # Keep validation before billable-weight arithmetic so invalid cartons
    # produce a domain error rather than a misleading carrier rate.
    cartons = _validate_parcels(parcels)
    service_level = _validate_service_level(service_level)
    pounds = combined_billable_weight(cartons)
    charge = base_rate(destination.zone, pounds, service_level)
    surcharge = shipment_surcharge(destination, len(cartons))
    return ShippingQuote(
        service_level=service_level,
        billable_pounds=pounds,
        base_charge=charge,
        residential_surcharge=surcharge,
        total=charge + surcharge,
    )
