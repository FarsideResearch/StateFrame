"""Registry of guarded fault handlers: which operation each handler guards and what faults are expected.

Data only. Importing this module changes no bytes and no behavior. The contract runner and the
guard tests read HANDLERS; kernel_fault_disposition.record_fault reads it to classify a fault.
Taxonomy: A fail-closed at op boundary, B typed degradation (needs fallback_id), C narrowed except,
D reviewed broad boundary. Unknown requiredness is UNCLASSIFIED, never guessed as optional.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

TAXONOMY = ("A", "B", "C", "D")
TIERS = ("T0", "T1", "T2", "T3")
REQUIREDNESS = ("required", "optional", "UNCLASSIFIED")
DISPOSITIONS = ("fail_closed", "degraded", "narrowed", "boundary")


@dataclass(frozen=True)
class HandlerSpec:
    handler_id: str
    module: str
    function: str
    op: str
    taxonomy: str
    tier: str
    requiredness: str
    expected: tuple[str, ...]
    dispositions: tuple[str, ...]
    fallback_id: str | None
    owner: str
    contract_tests: tuple[str, ...]
    policy_version: int
    # Work item the fault blocks (e.g. "engine_run"), carried on every fault row so a triage view can
    # tie a fault to the unit of work it stopped. None when the handler guards no tracked work item.
    required_work_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("handler_id", "module", "function", "op", "owner"):
            _require_text(getattr(self, name), name)
        if self.taxonomy not in TAXONOMY:
            raise ValueError(f"{self.handler_id}: taxonomy {self.taxonomy!r} not in {TAXONOMY}")
        if self.tier not in TIERS:
            raise ValueError(f"{self.handler_id}: tier {self.tier!r} not in {TIERS}")
        if self.requiredness not in REQUIREDNESS:
            raise ValueError(f"{self.handler_id}: requiredness {self.requiredness!r} not in {REQUIREDNESS}")
        _require_tuple_of_text(self.expected, "expected", self.handler_id)
        _require_tuple_of_text(self.contract_tests, "contract_tests", self.handler_id)
        if not isinstance(self.dispositions, tuple) or not self.dispositions:
            raise ValueError(f"{self.handler_id}: dispositions must be a non-empty tuple")
        for disposition in self.dispositions:
            if disposition not in DISPOSITIONS:
                raise ValueError(f"{self.handler_id}: disposition {disposition!r} not in {DISPOSITIONS}")
        if self.taxonomy == "B":
            _require_text(self.fallback_id, "fallback_id")
        elif self.fallback_id is not None:
            raise ValueError(f"{self.handler_id}: fallback_id is only allowed for taxonomy B")
        if isinstance(self.policy_version, bool) or not isinstance(self.policy_version, int) \
                or self.policy_version < 1:
            raise ValueError(f"{self.handler_id}: policy_version must be an int >= 1")
        if self.required_work_id is not None:
            _require_text(self.required_work_id, "required_work_id")


def _require_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text, got {value!r}")


def _require_tuple_of_text(value: object, name: str, handler_id: str) -> None:
    if not isinstance(value, tuple):
        raise ValueError(f"{handler_id}: {name} must be a tuple")
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{handler_id}: {name} entries must be non-empty text")


# Product registry. Each live guard site adds its row (plus inject hook and record_fault call).
# Tests use register_for_tests, never this dict.
HANDLERS: dict[str, HandlerSpec] = {
    # Catch-all for an engine run that ends ERR:* (probes/react_repair.py work). Every exception here is unexpected.
    "react_repair.run_exception": HandlerSpec(
        handler_id="react_repair.run_exception", module="probes.react_repair", function="work",
        op="react_repair.run", taxonomy="D", tier="T0", requiredness="required", expected=(),
        dispositions=("boundary",), fallback_id=None, owner="engine",
        contract_tests=(
            "tests/test_engine_error_reason.py::test_catch_all_returns_class_and_message_and_records_reason_row",
        ),
        policy_version=1, required_work_id="engine_run",
    ),
}

_TEST_HANDLERS: dict[str, HandlerSpec] = {}


def lookup_handler(handler_id: str) -> HandlerSpec | None:
    spec = HANDLERS.get(handler_id)
    return spec if spec is not None else _TEST_HANDLERS.get(handler_id)


@contextmanager
def register_for_tests(spec: HandlerSpec) -> Iterator[HandlerSpec]:
    if not isinstance(spec, HandlerSpec):
        raise TypeError(f"register_for_tests needs a HandlerSpec, got {type(spec).__name__}")
    if spec.handler_id in HANDLERS or spec.handler_id in _TEST_HANDLERS:
        raise ValueError(f"{spec.handler_id}: already registered")
    _TEST_HANDLERS[spec.handler_id] = spec
    try:
        yield spec
    finally:
        _TEST_HANDLERS.pop(spec.handler_id, None)
