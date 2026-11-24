"""Closed numeric and fixed-vocabulary telemetry details."""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from fractions import Fraction
import re
from math import isfinite
from types import MappingProxyType

from .. import __version__ as _ENGINE_VERSION


_PROVIDERS = frozenset(
    "azure cohere google groq lmstudio local llamacpp mistral ollama openai "
    "openai-compatible vllm".split()
)
_VERSION = re.compile(
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-(?:0|[1-9][0-9]*|[0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9][0-9]*|[0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?\Z"
)
_MODEL_ID = re.compile(r"[0-9a-f]{32}\Z")
_MAX_MICRO_UNITS = (1 << 63) - 1

# This mapping is the sole source of optional attribute names.
_ATTRIBUTE_NAMES = MappingProxyType({
    "request_model": "gen_ai.request.model",
    "response_model": "gen_ai.response.model",
    "provider_name": "gen_ai.provider.name",
    "engine_version": "service.version",
    "cost_value": "stateframe.cost.microunits",
    "cost_kind": "stateframe.cost.kind",
})
_BASE_FIELDS = ("request_model", "response_model", "provider_name", "engine_version")
_COST_FIELDS = ("cost_value", "cost_kind")
ATTRIBUTE_NAMES = frozenset(_ATTRIBUTE_NAMES.values())


def valid_version(value: object) -> bool:
    return type(value) is str and _VERSION.fullmatch(value) is not None


def valid_attribute(key: str, value: object) -> bool:
    """Recheck the closed detail allowlist at store and wire boundaries."""
    if key in {_ATTRIBUTE_NAMES["request_model"], _ATTRIBUTE_NAMES["response_model"]}:
        return type(value) is str and _MODEL_ID.fullmatch(value) is not None and value != "0" * 32
    if key == _ATTRIBUTE_NAMES["provider_name"]:
        return type(value) is str and value in _PROVIDERS | {"other"}
    if key == _ATTRIBUTE_NAMES["engine_version"]:
        return valid_version(value)
    if key == _ATTRIBUTE_NAMES["cost_value"]:
        return type(value) is int and 0 <= value <= _MAX_MICRO_UNITS
    if key == _ATTRIBUTE_NAMES["cost_kind"]:
        return type(value) is str and value in {"reported", "estimated"}
    return False


@dataclass(frozen=True, slots=True)
class Levels:
    include_cost: bool = False


def model_id(key: bytes, value: object) -> str | None:
    """Replace a model label with its installation-keyed identity hash."""
    if type(value) is not str or not value:
        return None
    try:
        source = value.encode("utf-8", "strict")
    except UnicodeError:
        return None
    # derive bounds its inputs; prehashing lets the identity function cover any
    # valid model label without retaining the label in a record or frame.
    import hashlib
    from .identity import derive

    return derive(key, "model", (hashlib.sha256(source).digest(),), 16).hex()


def provider_name(value: object) -> str:
    if type(value) is not str or not value.isascii() or len(value) > 32:
        return "other"
    lowered = value.casefold()
    return lowered if lowered in _PROVIDERS else "other"


def _cost_number(value: object) -> Fraction | None:
    if type(value) is int and value >= 0:
        return Fraction(value)
    if type(value) is float and isfinite(value) and value >= 0:
        return Fraction(str(value))
    return None


def _round_units(number: Fraction) -> int | None:
    units, remainder = divmod(number.numerator, number.denominator)
    units += int(2 * remainder >= number.denominator)
    return units if units <= _MAX_MICRO_UNITS else None


def _price_rates(table: object) -> tuple[Fraction, Fraction, int] | None:
    if type(table) is tuple and len(table) == 2:
        table = {"input_per_million": table[0], "output_per_million": table[1]}
    if type(table) is not dict:
        return None
    candidate = table
    if set(candidate) not in ({"input", "output"}, {"input_per_million", "output_per_million"}):
        if len(candidate) != 1 or type(next(iter(candidate.values()))) is not dict:
            return None
        candidate = next(iter(candidate.values()))
    keys = set(candidate)
    if keys == {"input", "output"}:
        first, second, multiplier = candidate["input"], candidate["output"], 1_000_000
    elif keys == {"input_per_million", "output_per_million"}:
        first, second, multiplier = candidate["input_per_million"], candidate["output_per_million"], 1
    else:
        return None
    input_rate, output_rate = _cost_number(first), _cost_number(second)
    if input_rate is None or output_rate is None:
        return None
    return input_rate, output_rate, multiplier


def cost_attributes(reported: object, table: object, usage: object) -> tuple[tuple[str, object], ...]:
    amount = _cost_number(reported)
    units = _round_units(amount * 1_000_000) if amount is not None else None
    if units is not None:
        return ((_ATTRIBUTE_NAMES["cost_value"], units), (_ATTRIBUTE_NAMES["cost_kind"], "reported"))
    rates = _price_rates(table)
    if (rates is None or type(usage) is not tuple or len(usage) != 2
            or any(type(number) is not int or number < 0 for number in usage)):
        return ()
    input_rate, output_rate, multiplier = rates
    units = _round_units((usage[0] * input_rate + usage[1] * output_rate) * multiplier)
    return () if units is None else (
        (_ATTRIBUTE_NAMES["cost_value"], units), (_ATTRIBUTE_NAMES["cost_kind"], "estimated"),
    )


def _record_fields(record: object) -> set[str]:
    if not is_dataclass(record) or isinstance(record, type):
        return set()
    params = getattr(type(record), "__dataclass_params__", None)
    if params is None or not params.frozen:
        return set()
    try:
        return {field.name for field in fields(record)}
    except (TypeError, AttributeError):
        return set()


def detail_attribute_keys(levels: Levels) -> tuple[str, ...]:
    keys = (*_BASE_FIELDS, *(_COST_FIELDS if levels.include_cost is True else ()))
    return tuple(_ATTRIBUTE_NAMES[key] for key in keys)


def describe(levels: Levels) -> tuple[str, ...]:
    return detail_attribute_keys(levels)


def extra_attributes(levels: Levels, record: object, key: bytes) -> tuple[tuple[str, object], ...]:
    available = _record_fields(record)

    def read(name: str) -> object:
        if name not in available:
            return None
        try:
            return object.__getattribute__(record, name)
        except (AttributeError, TypeError):
            return None

    values: dict[str, object] = {}
    for field_name in ("request_model", "response_model"):
        value = model_id(key, read(field_name))
        if value is not None:
            values[field_name] = value
    values["provider_name"] = provider_name(read("provider"))
    version = read("engine_version")
    if version == _ENGINE_VERSION and valid_version(version):
        values["engine_version"] = version
    if levels.include_cost is True:
        costs = dict(cost_attributes(read("reported_cost"), read("price_table"), read("usage")))
        for field_name in _COST_FIELDS:
            attribute = _ATTRIBUTE_NAMES[field_name]
            if attribute in costs:
                values[field_name] = costs[attribute]
    keys = (*_BASE_FIELDS, *(_COST_FIELDS if levels.include_cost is True else ()))
    return tuple((_ATTRIBUTE_NAMES[name], values[name]) for name in keys if name in values)
