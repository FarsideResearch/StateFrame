"""Completion evidence failures must not become a current, permissive snapshot."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import kernel_completion_policy as policy
import pytest


def _journaled_runtime(**fields: object) -> tuple[SimpleNamespace, list[tuple[str, dict[str, object]]]]:
    events: list[tuple[str, dict[str, object]]] = []
    journal = SimpleNamespace(emit=lambda kind, **data: events.append((kind, data)))
    return SimpleNamespace(journal=journal, **fields), events


def _unavailable(message: str):
    def fail():
        raise OSError(message)

    return fail


def _assert_failure_event(events: list[tuple[str, dict[str, object]]], code: str) -> None:
    assert events == [("unified_stop_evidence_failure", {
        "producer": "kernel_completion_policy", "dispatch": False,
        "reason_code": code,
    })]


def test_public_verdict_failure_is_visible_before_request() -> None:
    runtime, events = _journaled_runtime(public_test_verdict=_unavailable("verdict I/O failed"))

    with pytest.raises(RuntimeError, match="VERDICT_UNAVAILABLE") as caught:
        policy.collect_clauses(runtime)

    assert isinstance(caught.value.__cause__, OSError)
    _assert_failure_event(events, "VERDICT_UNAVAILABLE")


def test_declared_product_path_resolution_failure_is_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, events = _journaled_runtime(
        owner=SimpleNamespace(root=tmp_path, task_row={}),
        scope={"declared_paths": ["src/product.py"]},
    )
    resolve = Path.resolve

    def fail_product_path(path: Path, *args: object, **kwargs: object) -> Path:
        if path == tmp_path / "src/product.py":
            raise OSError("product path I/O failed")
        return resolve(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setenv(policy.DECLARED_PATH_DONE_GATE_ENV, "on")
        patch.setattr(Path, "resolve", fail_product_path)
        with pytest.raises(RuntimeError, match="DECLARED_PATH_UNAVAILABLE") as caught:
            policy.collect_clauses(runtime)

    assert isinstance(caught.value.__cause__, OSError)
    _assert_failure_event(events, "DECLARED_PATH_UNAVAILABLE")


def test_declared_owner_root_resolution_failure_is_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, events = _journaled_runtime(
        owner=SimpleNamespace(root=tmp_path, task_row={}),
        scope={"declared_paths": ["src/product.py"]},
    )
    resolve = Path.resolve

    def fail_owner_root(path: Path, *args: object, **kwargs: object) -> Path:
        if path == tmp_path:
            raise OSError("owner root I/O failed")
        return resolve(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setenv(policy.DECLARED_PATH_DONE_GATE_ENV, "on")
        patch.setattr(Path, "resolve", fail_owner_root)
        with pytest.raises(RuntimeError, match="DECLARED_ROOT_UNAVAILABLE") as caught:
            policy.collect_clauses(runtime)

    assert isinstance(caught.value.__cause__, OSError)
    _assert_failure_event(events, "DECLARED_ROOT_UNAVAILABLE")


def test_snapshot_binding_failure_is_visible_before_hashing() -> None:
    runtime, events = _journaled_runtime(binding=_unavailable("binding I/O failed"))

    with pytest.raises(RuntimeError, match="BINDING_UNAVAILABLE") as caught:
        policy.snapshot_id(runtime, (policy.ClauseAssessment("tests.public", policy.SATISFIED),))

    assert isinstance(caught.value.__cause__, OSError)
    _assert_failure_event(events, "BINDING_UNAVAILABLE")
