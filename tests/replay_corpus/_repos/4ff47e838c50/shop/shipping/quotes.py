"""Public shipping quote entry point."""

from typing import Iterable

from .models import Destination, Parcel, ShippingQuote
from .rates import (
    base_rate,
    combined_billable_weight,
    shipment_surcharge,
    validate_carrier_limits,
)


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
