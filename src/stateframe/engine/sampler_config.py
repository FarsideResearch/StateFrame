"""Validated optional sampling controls for model request bodies."""
from __future__ import annotations

import math
import os
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass


_FLOAT_FIELDS = ("top_p", "min_p")
_INT_FIELDS = ("top_k",)
VARIANT_ENV = "STATEFRAME_TOKENLAB_SAMPLER_VARIANT"
SAMPLER_EVENT = "tokenlab_v17_sampler_sent"


def _bounded_float(name: str, raw: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number in [0, 1]") from error
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return value


def _nonnegative_int(name: str, raw: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a nonnegative integer") from error
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def variant_mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = (source.get(VARIANT_ENV, "off") or "off").strip().lower()
    if value not in {"off", "v17t6"}:
        raise ValueError(f"{VARIANT_ENV} must be off or v17t6")
    return value


@dataclass(frozen=True)
class SamplerSettings:
    """The optional fields that may be added to a request."""

    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None

    def request_fields(self) -> dict[str, float | int]:
        fields: dict[str, float | int] = {}
        if self.top_p is not None:
            fields["top_p"] = self.top_p
        if self.top_k is not None:
            fields["top_k"] = self.top_k
        if self.min_p is not None:
            fields["min_p"] = self.min_p
        return fields

    def identity(self) -> dict[str, object]:
        return {
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
        }


def from_environment(environ: Mapping[str, str] | None = None) -> SamplerSettings:
    """Read and validate the opt-in environment fields.

    An absent field stays absent. An explicitly supplied empty or malformed
    value is rejected so a configured run cannot silently use a server default.
    """
    source = os.environ if environ is None else environ
    values: dict[str, object] = {}
    for field in _FLOAT_FIELDS:
        raw = source.get(f"REACT_{field.upper()}")
        values[field] = None if raw is None else _bounded_float(
            f"REACT_{field.upper()}", raw)
    for field in _INT_FIELDS:
        raw = source.get(f"REACT_{field.upper()}")
        values[field] = None if raw is None else _nonnegative_int(
            f"REACT_{field.upper()}", raw)
    return SamplerSettings(**values)


def apply_to_request(body: MutableMapping[str, object],
                     settings: SamplerSettings | None = None) -> MutableMapping[str, object]:
    """Add configured sampler fields in place and return the same body."""
    if not isinstance(body, MutableMapping):
        raise TypeError("request body must be mutable")
    selected = from_environment() if settings is None else settings
    if variant_mode() == "v17t6":
        source = os.environ
        body["temperature"] = _bounded_float("REACT_TEMP", source.get("REACT_TEMP", "0.6"))
        selected = SamplerSettings(
            top_p=selected.top_p if selected.top_p is not None else 0.95,
            top_k=selected.top_k,
            min_p=selected.min_p if selected.min_p is not None else 0.05,
        )
    fields = selected.request_fields()
    if fields:
        body.update(fields)
    return body


def journal_sent(capture: object, body: Mapping[str, object], *, phase: str = "turn",
                 attempt: int | None = None) -> None:
    """Journal the exact sampler fields present in an opted-in provider body."""
    variant = variant_mode()
    if variant == "off":
        return
    journal = getattr(capture, "journal", None)
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        return
    fields = {name: body[name] for name in ("temperature", "top_p", "min_p") if name in body}
    emit(SAMPLER_EVENT, producer="sampler_config", variant=variant, phase=phase,
         attempt=attempt, fields=fields, request_id=getattr(capture, "request_id", None),
         dispatch=False)


def effective_identity(temperature: float | int | None,
                       settings: SamplerSettings | None = None) -> dict[str, object]:
    """Return a JSON-safe record of the requested sampler, including unset fields."""
    selected = from_environment() if settings is None else settings
    identity = {"temperature": temperature, **selected.identity()}
    variant = variant_mode()
    if variant != "off":
        identity["variant"] = variant
    return identity
