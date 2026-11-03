"""Weight and zone rules used by parcel rates."""

from decimal import Decimal, ROUND_CEILING

from .models import Destination, Parcel

DIMENSIONAL_DIVISOR = Decimal("139")
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
