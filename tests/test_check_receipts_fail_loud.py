"""High-risk receipt failures must reach callers before any misleading wire fact."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError, dataclass
from types import SimpleNamespace

import pytest

from kernel_check_receipts import (
    _check_execution_context,
    _next_check_execution_ordinal,
    check_execution_binding_mode,
    classify_check_outcome,
    current_wire_revision,
    record_check_output,
    retrieve_check_output,
)


class BrokenFiles(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 1

    def items(self):
        raise OSError("product files unavailable")


def _unavailable_binding(message: str):
    def binding():
        raise OSError(message)

    return binding


def test_wire_revision_binding_property_failure_is_visible() -> None:
    class Runtime:
        @property
        def binding(self):
            raise RuntimeError("binding property unavailable")

    with pytest.raises(RuntimeError, match="binding property unavailable"):
        current_wire_revision(Runtime())


def test_wire_revision_binding_call_failure_is_visible() -> None:
    runtime = SimpleNamespace(binding=_unavailable_binding("binding unavailable"))
    with pytest.raises(OSError, match="binding unavailable"):
        current_wire_revision(runtime)


def test_wire_revision_mapping_failure_is_visible() -> None:
    runtime = SimpleNamespace(binding={"product_files": BrokenFiles()})
    with pytest.raises(OSError, match="product files unavailable"):
        current_wire_revision(runtime)


def test_receipt_lookup_refuses_failed_live_revision() -> None:
    runtime = SimpleNamespace(
        binding=_unavailable_binding("live revision unavailable"),
        _check_output_receipts={"check-receipt:old": {
            "revision": "sha256:old", "selector": "tests/test_old.py", "stdout": "stale output",
        }},
    )
    with pytest.raises(OSError, match="live revision unavailable"):
        retrieve_check_output(runtime, "check-receipt:old")


def test_binding_mode_property_failure_is_visible() -> None:
    class Runtime:
        @property
        def stateframe_check_execution_binding_mode(self):
            raise RuntimeError("binding mode unavailable")

    with pytest.raises(RuntimeError, match="binding mode unavailable"):
        check_execution_binding_mode(Runtime())


def test_execution_context_property_failure_is_visible() -> None:
    class Runtime:
        @property
        def _stateframe_check_execution_contexts(self):
            raise RuntimeError("execution context unavailable")

    with pytest.raises(RuntimeError, match="execution context unavailable"):
        _check_execution_context(Runtime(), "check-execution:one")


def test_execution_ordinal_assignment_failure_is_visible() -> None:
    @dataclass(frozen=True)
    class Journal:
        check_receipt_count: int = 1
        check_execution_allocations: int = 1

    with pytest.raises(FrozenInstanceError):
        _next_check_execution_ordinal(Journal())


def test_malformed_exit_code_cannot_be_classified_as_pass() -> None:
    with pytest.raises(ValueError, match="CHECK_RECEIPT_INVALID_EXIT_CODE"):
        classify_check_outcome("PASS", outcome="PASS", exit_code="not-an-exit-code")


def test_malformed_collected_count_cannot_be_classified_as_pass() -> None:
    with pytest.raises(ValueError, match="CHECK_RECEIPT_INVALID_COLLECTED_TESTS"):
        classify_check_outcome("PASS", outcome="PASS", collected_tests="not-a-count")


def test_record_refuses_failed_pinned_revision() -> None:
    runtime = SimpleNamespace(
        stateframe_check_execution_binding_mode="on",
        _stateframe_check_execution_contexts={
            "check-execution:one": {"binding": {"product_files": BrokenFiles()}},
        },
    )
    with pytest.raises(OSError, match="product files unavailable"):
        record_check_output(
            runtime, selector="tests/test_one.py", status="PASS", outcome="PASS",
            execution_id="check-execution:one", revision="sha256:legacy",
        )
