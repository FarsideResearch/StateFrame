"""Clause-grounded, isolated behavioral witnesses.

This is a host observation organ, not an oracle.  A case records the observed
shape/value projection before its optional expectation runs, and one bad case
cannot prevent later independent cases from executing.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from kernel_gap_switches import emit, truth_enabled


SWITCH = "STATEFRAME_GAPS_WITNESSES"
PRODUCER = "kernel_gap_witnesses"
MAX_CASES = 6
MAX_PROJECTION_CHARS = 600


def _bounded(value: object, limit: int = MAX_PROJECTION_CHARS) -> str:
    try:
        text = json.dumps(value, sort_keys=True, default=repr, ensure_ascii=False)
    except (TypeError, ValueError):
        text = repr(value)
    return text[:limit] + ("..." if len(text) > limit else "")


def observation_shape(value: object) -> object:
    """Describe shape without pretending it proves semantic behavior."""
    if isinstance(value, Mapping):
        return {str(key): observation_shape(child) for key, child in sorted(
            value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "length": len(value),
                "items": [observation_shape(item) for item in value[:4]]}
    if value is None or isinstance(value, (str, int, float, bool)):
        return {"type": type(value).__name__, "value": value}
    return {"type": type(value).__name__}


@dataclass(frozen=True)
class WitnessCase:
    case_id: str
    requirement_id: str
    surface: str
    clause: str
    signature: str
    fixture: object
    variation: str
    expected_projection: object | None = None
    expect: Callable[[object], bool] | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class WitnessResult:
    case_id: str
    status: str
    observed_shape: object | None = None
    observed_projection: str = ""
    error_type: str = ""
    error_message: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "status": self.status,
            "observed_shape": self.observed_shape,
            "observed_projection": self.observed_projection,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class WitnessSuite:
    enabled: bool
    revision: str
    results: tuple[WitnessResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {"enabled": self.enabled, "revision": self.revision,
                "results": [result.to_dict() for result in self.results]}


def run_witnesses(cases: Sequence[WitnessCase], invoke: Callable[[object], object], *,
                  revision: str, journal: object | None = None) -> WitnessSuite:
    """Run bounded independent cases and continue after every local failure."""
    if not truth_enabled(SWITCH):
        return WitnessSuite(False, str(revision), ())
    results: list[WitnessResult] = []
    for case in tuple(cases)[:MAX_CASES]:
        try:
            observed = invoke(case.fixture)
        except Exception as error:  # noqa: BLE001 - isolate one witness case
            result = WitnessResult(
                case.case_id, "OBSERVATION_ERROR", error_type=type(error).__name__,
                error_message=str(error)[:240],
            )
        else:
            shape = observation_shape(observed)
            projection = _bounded(observed)
            try:
                if case.expected_projection is not None and observed != case.expected_projection:
                    status = "COUNTEREXAMPLE"
                elif case.expect is not None and not case.expect(observed):
                    status = "COUNTEREXAMPLE"
                else:
                    status = "VERIFIED_OBSERVATION"
                result = WitnessResult(case.case_id, status, shape, projection)
            except Exception as error:  # noqa: BLE001 - isolate a faulty assertion
                result = WitnessResult(
                    case.case_id, "CHECK_ERROR", shape, projection,
                    type(error).__name__, str(error)[:240],
                )
        results.append(result)
        emit(
            journal,
            "gaps_witness_case",
            producer=PRODUCER,
            enabled_value=True,
            witness_case=case.case_id,
            requirement=case.requirement_id,
            surface=case.surface,
            variation=case.variation,
            status=result.status,
            observed_shape=result.observed_shape,
            observed_projection=result.observed_projection,
            error_type=result.error_type,
        )
    return WitnessSuite(True, str(revision), tuple(results))


def render_witness_suite(suite: WitnessSuite) -> str:
    """Render observations as a bounded fact block for an existing frame."""
    if not suite.enabled:
        return ""
    payload = _bounded(suite.to_dict(), 1800)
    return "[Behavioral witness observations]\n" + payload


__all__ = [
    "MAX_CASES", "SWITCH", "WitnessCase", "WitnessResult", "WitnessSuite",
    "observation_shape", "render_witness_suite", "run_witnesses",
]
