"""Fail-loud checks: broad except sites in the test-run span, check receipt, check truth and authored lifecycle modules.

Scope: kernel_authored_check_lifecycle.py, kernel_check_receipts.py, kernel_check_truth_adapter.py and
kernel_test_run_spans.py. Every broad except in the four modules is triaged in the verdict table kept with the
T8 test-gap plan. Sites that swallow on a path a request takes and return a typed outcome, or
that sit in advisory telemetry whose contract is "must never replace the check result", get a pin test here.

No default-path CRASH-HIDER exists in this unit. The opt-in CRASH-HIDER sites are fixed and pinned here with the
gating switches ON: a check execution binding that cannot be pinned raises in 'on' mode (journaled in 'shadow'), a
failed vacuity journal projection downgrades the certificate to UNKNOWN, and a pinned selector scope that cannot be
applied is no longer reported as applied.

Inventory guard: every broad except in the four modules must be listed in the triaged set below, so a new
swallow cannot land untriaged.
"""
from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"


# --- kernel_check_receipts ---------------------------------------------------------------------------


class _UnreadableRuntime:
    """Runtime whose product revision property raises, like a broken binding accessor."""

    @property
    def product_revision(self) -> str:
        raise RuntimeError("forced: product revision unavailable")


def test_wire_revision_property_failure_fails_closed_to_unknown() -> None:
    """Site kernel_check_receipts:156. A failing revision accessor yields UNKNOWN, which the stale-wire
    consumers (kernel_greenfact, kernel_request_tool_surface) treat as unverified rather than trusting a revision."""
    from kernel_check_receipts import current_wire_revision

    assert current_wire_revision(_UnreadableRuntime()) == "UNKNOWN"


class _MemoModeRuntime:
    @property
    def stateframe_check_receipt_memo_mode(self) -> str:
        raise RuntimeError("forced: memo mode unavailable")


def test_receipt_memo_mode_failure_defaults_to_off() -> None:
    """Site kernel_check_receipts:1380. A failing mode read falls back to the 'off' default (no memo reuse)."""
    from kernel_check_receipts import _check_receipt_memo_mode

    assert _check_receipt_memo_mode(_MemoModeRuntime()) == "off"


def test_shadow_emit_failure_never_fails_receipt_recording() -> None:
    """Site kernel_check_receipts:754. Shadow telemetry is advisory: a failing journal emit must not raise."""
    from kernel_check_receipts import _emit_shadow_mismatch

    class Journal:
        def emit(self, *_args: object, **_fields: object) -> None:
            raise OSError("forced: journal unavailable")

    runtime = SimpleNamespace(journal=Journal())
    _emit_shadow_mismatch(runtime, kind="check_receipt_enrichment_dedup_shadow_mismatch",
                          execution_id="check-execution:one", field="emission",
                          legacy_value="emit", proposed_value="suppress")


class _BrokenItems(dict):
    """Receipt mapping whose items view raises, forcing the comparison fallback."""

    def items(self):  # type: ignore[override]
        raise RuntimeError("forced: receipt items unreadable")


def test_receipt_content_compare_failure_reports_changed() -> None:
    """Site kernel_check_receipts:794. A comparison failure returns False, so the receipt is rewritten
    (generation bump) instead of being suppressed as unchanged. This is the safe direction."""
    from kernel_check_receipts import _receipt_content_equal

    broken = _BrokenItems({"status": "PASS"})
    assert _receipt_content_equal(broken, {"status": "PASS"}) is False


# --- kernel_test_run_spans --------------------------------------------------------------------------


def test_profile_event_failure_never_replaces_completed_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Site kernel_test_run_spans:294. A profile that cannot build its event finalizes quietly."""
    from kernel_test_run_spans import TestRunProfile

    profile = TestRunProfile("on")

    def broken_event() -> dict[str, object]:
        raise RuntimeError("forced: event build failed")

    monkeypatch.setattr(profile, "_event", broken_event)
    profile.finalize(None)


def test_profile_publish_failure_never_replaces_completed_check() -> None:
    """Site kernel_test_run_spans:314. A journal that rejects the profile event does not fail finalize."""
    from kernel_test_run_spans import TestRunProfile

    class Journal:
        def emit(self, *_args: object, **_fields: object) -> None:
            raise OSError("forced: journal rejected event")

    profile = TestRunProfile("on", runtime=SimpleNamespace(journal=Journal()))
    profile.finalize(None)


def test_loaded_runtime_lookup_failure_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Site kernel_test_run_spans:339. A failing active-runtime getter yields no runtime, never an import."""
    import sys

    from kernel_test_run_spans import _loaded_runtime

    def broken_get() -> object:
        raise RuntimeError("forced: active runtime getter failed")

    fake_pilot = SimpleNamespace(ACTIVE_RUNTIME=SimpleNamespace(get=broken_get))
    monkeypatch.setitem(sys.modules, "kernel_pilot", fake_pilot)
    assert _loaded_runtime() is None


def test_profile_scope_fails_open_when_profile_cannot_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """Site kernel_test_run_spans:362. When the profile cannot start the check still runs with no profile."""
    import kernel_test_run_spans as spans

    monkeypatch.setenv(spans.MODE_ENV, "on")

    def refusing_profile(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("forced: monotonic observer unavailable")

    monkeypatch.setattr(spans, "TestRunProfile", refusing_profile)
    with spans.profile_scope(runtime=None) as profile:
        assert profile is None


def test_profile_cleanup_failure_does_not_mask_check_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Site kernel_test_run_spans:376. A failure while finalizing the outer profile leaves the check result intact."""
    import kernel_test_run_spans as spans

    monkeypatch.setenv(spans.MODE_ENV, "on")

    def broken_finalize(self: object, runtime: object = None) -> None:
        raise RuntimeError("forced: finalize failed")

    monkeypatch.setattr(spans.TestRunProfile, "finalize", broken_finalize)
    with spans.profile_scope(runtime=None) as profile:
        assert profile is not None


# --- kernel_authored_check_lifecycle ----------------------------------------------------------------


def _fake_check(identity: str = "authored:one") -> object:
    """Frozen check shell with the attributes execute_check reads; avoids freezing real executables."""
    from kernel_authored_check_lifecycle import FrozenCheck

    check = FrozenCheck.__new__(FrozenCheck)
    object.__setattr__(check, "identity", identity)
    return check


@pytest.fixture
def lifecycle(monkeypatch: pytest.MonkeyPatch) -> Iterator[object]:
    import kernel_authored_check_lifecycle as lifecycle_module

    monkeypatch.setattr(lifecycle_module.FrozenCheck, "artifact",
                        property(lambda self: {"language": "fake"}))
    monkeypatch.setattr(lifecycle_module, "_execution_files",
                        lambda check, snapshot: {"main.py": b"print('ok')"})
    monkeypatch.setattr(lifecycle_module, "_snapshot_complete", lambda check, snapshot: (True, ""))
    yield lifecycle_module


_SNAPSHOT = {"revision": "sha256:rev", "side": "candidate", "binding_stable": True}


def test_adapter_exception_becomes_typed_unrunnable(lifecycle: object) -> None:
    """Site kernel_authored_check_lifecycle:520. An adapter exception is a typed UNRUNNABLE result with a reason."""

    def crashing_run(artifact: object, payload: object, **_kwargs: object) -> object:
        raise RuntimeError("forced: adapter crashed")

    result = lifecycle.execute_check(_fake_check(), _SNAPSHOT, SimpleNamespace(run=crashing_run))
    assert result.status == lifecycle.UNRUNNABLE
    assert result.reason == "adapter_error:RuntimeError"


def test_adapter_positional_retry_exception_becomes_typed_unrunnable(lifecycle: object) -> None:
    """Site kernel_authored_check_lifecycle:518. The TypeError retry path also turns a crash into UNRUNNABLE."""
    calls: list[str] = []

    def legacy_signature(artifact: object, payload: object, *args: object, **kwargs: object) -> object:
        calls.append("call")
        if kwargs:
            raise TypeError("forced: keyword signature unsupported")
        raise ValueError("forced: adapter crashed on positional retry")

    result = lifecycle.execute_check(_fake_check(), _SNAPSHOT, SimpleNamespace(run=legacy_signature))
    assert calls, "adapter was never invoked"
    assert result.status == lifecycle.UNRUNNABLE
    assert result.reason == "adapter_error:ValueError"


# --- opt-in switches ON: check execution binding and vacuity journal truth -----------------------------


class _RecordingJournal:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def emit(self, kind: str, **fields: object) -> None:
        self.events.append((kind, fields))

    def kinds(self) -> list[str]:
        return [kind for kind, _fields in self.events]


class _ContextsUnwritableRuntime:
    """Runtime that refuses to store the pinned execution contexts, like a frozen or slotted owner."""

    def __init__(self, mode: str) -> None:
        object.__setattr__(self, "stateframe_check_execution_binding_mode", mode)
        object.__setattr__(self, "journal", _RecordingJournal())

    def __setattr__(self, name: str, value: object) -> None:
        if name == "_stateframe_check_execution_contexts":
            raise RuntimeError("forced: contexts cannot be stored")
        object.__setattr__(self, name, value)


def test_binding_on_registration_failure_fails_loud() -> None:
    """Site kernel_check_truth_adapter pin (binding on). The pinned binding is the contract in 'on' mode, so a
    failed registration must raise a typed error instead of letting the receipt fall back to the current revision."""
    from kernel_check_truth_adapter import pin_check_execution_context

    runtime = _ContextsUnwritableRuntime("on")
    with pytest.raises(RuntimeError) as info:
        pin_check_execution_context(runtime, "check-execution:one", binding={"head": "abc"},
                                    selector="tests/test_a.py")
    assert type(info.value).__name__ == "CheckExecutionBindingError"
    assert isinstance(info.value.__cause__, RuntimeError)
    assert runtime.journal.kinds() == ["check_execution_binding_register_failed"]
    fields = runtime.journal.events[0][1]
    assert fields["execution_id"] == "check-execution:one"
    assert fields["mode"] == "on"
    assert fields["error_type"] == "RuntimeError"


def test_binding_shadow_registration_failure_is_reported_not_raised() -> None:
    """Regression: in 'shadow' mode the legacy revision path stays authoritative, so the failure is journaled and the
    check launch continues."""
    from kernel_check_truth_adapter import pin_check_execution_context

    runtime = _ContextsUnwritableRuntime("shadow")
    pin_check_execution_context(runtime, "check-execution:one", binding={"head": "abc"},
                                selector="tests/test_a.py")
    assert runtime.journal.kinds() == ["check_execution_binding_register_failed"]
    assert runtime.journal.events[0][1]["mode"] == "shadow"


def test_binding_failure_event_emit_error_never_masks_the_typed_error(caplog: pytest.LogCaptureFixture) -> None:
    """Regression: a broken journal must not replace the typed binding error with its own exception; the lost
    event is logged instead."""
    from kernel_check_truth_adapter import pin_check_execution_context

    class BrokenJournal:
        def emit(self, *_args: object, **_fields: object) -> None:
            raise OSError("forced: journal unavailable")

    runtime = _ContextsUnwritableRuntime("on")
    object.__setattr__(runtime, "journal", BrokenJournal())
    with pytest.raises(RuntimeError) as info:
        pin_check_execution_context(runtime, "check-execution:one", binding={"head": "abc"},
                                    selector="tests/test_a.py")
    assert type(info.value).__name__ == "CheckExecutionBindingError"
    assert "could not journal check_execution_binding_register_failed" in caplog.text


def test_binding_on_registration_success_still_pins_context() -> None:
    """Regression: the happy path keeps the detached binding and normalized scope."""
    from kernel_check_truth_adapter import pin_check_execution_context

    runtime = SimpleNamespace(stateframe_check_execution_binding_mode="on", journal=_RecordingJournal())
    pin_check_execution_context(runtime, "check-execution:one", binding={"head": "abc"},
                                selector="tests/test_a.py", selector_scope=[" tests/test_a.py::t1 ", ""])
    context = runtime._stateframe_check_execution_contexts["check-execution:one"]
    assert context == {"binding": {"head": "abc"}, "selector_scope": ("tests/test_a.py::t1",)}
    assert runtime.journal.events == []


_PINNED_SCOPE = ("tests/test_a.py::t1", "tests/test_a.py::t2")


def _record(runtime: object, **extra: object) -> dict[str, object]:
    from kernel_check_receipts import record_check_output

    return record_check_output(
        runtime, selector="tests/test_a.py", status="PASS", stdout="2 passed in 0.01s",
        exit_code=0, execution_id="check-execution:one", **extra)


def _pinned_runtime() -> SimpleNamespace:
    return SimpleNamespace(
        stateframe_check_execution_binding_mode="on",
        journal=_RecordingJournal(),
        _stateframe_check_execution_contexts={
            "check-execution:one": {"binding": {}, "selector_scope": _PINNED_SCOPE},
        },
    )


def _stored(runtime: SimpleNamespace) -> dict[str, object]:
    rows = [row for row in runtime._check_output_receipts.values()
            if row.get("execution_id") == "check-execution:one"]
    assert len(rows) == 1
    return rows[0]


def test_binding_on_scope_fallback_does_not_claim_pinned_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """Site kernel_check_receipts pinned-scope fallback (binding on). When the pinned scope cannot be applied the
    receipt falls back to the legacy scope; it must not report the pinned scope as applied and overwrite the
    scope an earlier view of the same execution already settled."""
    import kernel_check_receipts as receipts

    runtime = _pinned_runtime()
    _record(runtime)
    assert _stored(runtime)["selector_scope"] == list(_PINNED_SCOPE)

    real = receipts._selector_scope

    def failing_on_pinned(selector: str, value: object) -> tuple[str, ...]:
        if value == _PINNED_SCOPE:
            raise RuntimeError("forced: pinned scope unreadable")
        return real(selector, value)

    monkeypatch.setattr(receipts, "_selector_scope", failing_on_pinned)
    _record(runtime)
    assert _stored(runtime)["selector_scope"] == list(_PINNED_SCOPE)
    assert "check_execution_binding_scope_failed" in runtime.journal.kinds()


def test_binding_on_pinned_scope_still_applied_on_success() -> None:
    """Regression: a readable pinned scope is still applied to a later view of the same execution."""
    runtime = _pinned_runtime()
    _record(runtime)
    _record(runtime)
    assert _stored(runtime)["selector_scope"] == list(_PINNED_SCOPE)
    assert "check_execution_binding_scope_failed" not in runtime.journal.kinds()


class _VacuityJournal(_RecordingJournal):
    def __init__(self, fail: bool) -> None:
        super().__init__()
        self.fail = fail

    def cached_records(self) -> list[dict[str, object]]:
        if self.fail:
            raise RuntimeError("forced: journal projection unavailable")
        return []


@pytest.fixture
def vacuity_on(monkeypatch: pytest.MonkeyPatch) -> object:
    """Vacuity sentinel and journal truth on; the check output alone yields a PASS certificate."""
    import kernel_vacuity_sentinel as sentinel
    import kernel_vacuity_truth as truth

    def passing_certificate(*_args: object, **_kwargs: object) -> object:
        return sentinel.ObservationCertificate(
            1, 1, 1, 0, False, sentinel.PASS, "1 executed, all passed", ("tests/test_a.py",), "key")

    monkeypatch.setattr(sentinel, "certificate_from_output", passing_certificate)
    monkeypatch.setattr(truth, "enabled", lambda *_args, **_kwargs: True)
    return sentinel


def _vacuity_runtime(journal: object) -> SimpleNamespace:
    return SimpleNamespace(stateframe_vacuity_sentinel_mode="on", journal=journal,
                           binding=lambda: {"head": "abc"}, _check_output_receipts={})


def test_vacuity_journal_failure_does_not_keep_completed_certificate(vacuity_on: object) -> None:
    """Site kernel_check_truth_adapter journal projection (vacuity truth on). Without journal truth the certificate
    cannot see a change since the check or a never-run gate, so it must not be presented as a current PASS."""
    from kernel_check_truth_adapter import _refresh_vacuity_view

    runtime = _vacuity_runtime(_VacuityJournal(fail=True))
    _refresh_vacuity_view(runtime, {"receipt_id": "r1", "exit_code": 0, "selector": "tests/test_a.py"})
    certificate = runtime._vacuity_certificate
    assert certificate.status == vacuity_on.UNKNOWN
    assert "PASS" in certificate.reason and "journal" in certificate.reason
    assert runtime.journal.kinds() == ["check_truth_vacuity_journal_failed"]
    assert runtime.journal.events[0][1]["error_type"] == "RuntimeError"


def test_vacuity_current_view_failure_does_not_keep_completed_certificate(
        vacuity_on: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same site, failure inside the journal projection itself."""
    from kernel_check_truth_adapter import _refresh_vacuity_view

    def broken_view(*_args: object, **_kwargs: object) -> object:
        raise KeyError("forced: projection failed")

    monkeypatch.setattr(vacuity_on, "current_view", broken_view)
    runtime = _vacuity_runtime(_VacuityJournal(fail=False))
    _refresh_vacuity_view(runtime, {"receipt_id": "r1", "exit_code": 0, "selector": "tests/test_a.py"})
    assert runtime._vacuity_certificate.status == vacuity_on.UNKNOWN
    assert runtime.journal.kinds() == ["check_truth_vacuity_journal_failed"]
    assert runtime.journal.events[0][1]["error_type"] == "KeyError"


def test_vacuity_journal_success_still_projects_truth(vacuity_on: object) -> None:
    """Regression: a working journal projection still replaces the output certificate with journal truth."""
    from kernel_check_truth_adapter import _refresh_vacuity_view

    runtime = _vacuity_runtime(_VacuityJournal(fail=False))
    _refresh_vacuity_view(runtime, {"receipt_id": "r1", "exit_code": 0, "selector": "tests/test_a.py"})
    assert runtime._vacuity_certificate.status == vacuity_on.GATE_NEVER_RAN
    assert runtime.journal.events == []



def test_binding_off_registration_is_untouched() -> None:
    """OFF identity: with binding off nothing is stored, raised or journaled, even on an unwritable runtime."""
    from kernel_check_truth_adapter import pin_check_execution_context

    runtime = _ContextsUnwritableRuntime("off")
    assert pin_check_execution_context(runtime, "check-execution:one", binding={"head": "abc"},
                                       selector="tests/test_a.py") is None
    assert runtime.journal.events == []
    assert not hasattr(runtime, "_stateframe_check_execution_contexts")


def test_vacuity_sentinel_off_never_reads_the_journal() -> None:
    """OFF identity: with the vacuity sentinel off no certificate is built and the journal is never read."""
    from kernel_check_truth_adapter import _refresh_vacuity_view

    journal = _VacuityJournal(fail=True)
    runtime = SimpleNamespace(stateframe_vacuity_sentinel_mode="off", journal=journal,
                              binding=lambda: {"head": "abc"}, _check_output_receipts={})
    _refresh_vacuity_view(runtime, {"receipt_id": "r1", "exit_code": 0, "selector": "tests/test_a.py"})
    assert not hasattr(runtime, "_vacuity_certificate")
    assert journal.events == []


@pytest.mark.parametrize("binding_mode", ["off", "on"])
@pytest.mark.parametrize("journal_truth", [False, True])
def test_switch_pairs_binding_x_vacuity_truth(
        binding_mode: str, journal_truth: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pairs 00/01/10/11 of check execution binding x vacuity journal truth, with both failures forced: each switch
    changes only its own outcome, and only its own event is journaled."""
    import kernel_vacuity_sentinel as sentinel
    import kernel_vacuity_truth as truth
    from kernel_check_truth_adapter import _refresh_vacuity_view, pin_check_execution_context

    monkeypatch.setattr(sentinel, "certificate_from_output", lambda *_a, **_k: sentinel.ObservationCertificate(
        1, 1, 1, 0, False, sentinel.PASS, "1 executed, all passed", ("tests/test_a.py",), "key"))
    monkeypatch.setattr(truth, "enabled", lambda *_args, **_kwargs: journal_truth)
    runtime = _ContextsUnwritableRuntime(binding_mode)
    journal = _VacuityJournal(fail=True)
    object.__setattr__(runtime, "journal", journal)
    object.__setattr__(runtime, "stateframe_vacuity_sentinel_mode", "on")
    object.__setattr__(runtime, "binding", lambda: {"head": "abc"})
    object.__setattr__(runtime, "_check_output_receipts", {})

    if binding_mode == "on":
        with pytest.raises(RuntimeError, match="could not pin"):
            pin_check_execution_context(runtime, "check-execution:one", binding={"head": "abc"},
                                        selector="tests/test_a.py")
    else:
        pin_check_execution_context(runtime, "check-execution:one", binding={"head": "abc"},
                                    selector="tests/test_a.py")
    _refresh_vacuity_view(runtime, {"receipt_id": "r1", "exit_code": 0, "selector": "tests/test_a.py"})

    expected_status = sentinel.UNKNOWN if journal_truth else sentinel.PASS
    assert runtime._vacuity_certificate.status == expected_status
    expected_events = (["check_execution_binding_register_failed"] if binding_mode == "on" else []) + (
        ["check_truth_vacuity_journal_failed"] if journal_truth else [])
    assert journal.kinds() == expected_events

# --- inventory guard ----------------------------------------------------------------------------------


def _broad_except_sites(paths: tuple[Path, ...]) -> set[tuple[str, int]]:
    broad = re.compile(r"^\s*except\s*(?:\(|Exception\b|BaseException\b|:)")
    found: set[tuple[str, int]] = set()
    for path in paths:
        for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not broad.match(text):
                continue
            if re.match(r"^\s*except\s*\(", text) and not re.search(r"Exception|BaseException", text):
                continue  # narrow tuple
            found.add((path.stem, number))
    return found


_MODULE_PATHS = (
    ENGINE / "kernel_authored_check_lifecycle.py",
    ENGINE / "kernel_check_receipts.py",
    ENGINE / "kernel_check_truth_adapter.py",
    ENGINE / "kernel_test_run_spans.py",
)

# No pilot sites live in these modules (checked against tests/test_t8_failloud_pilot.py).
_PILOT: set[tuple[str, int]] = set()

# Every broad except in the modules. The census lists 25 silent sites; receipts 1247 (re-raise, census LOUD),
# truth adapter 405 (raise on a non-callable projection) and spans 525 (terminal stats fallback, child template) are
# triaged here too. Opt-in fixes: truth adapter 69 (raises in binding 'on'), 414 (UNKNOWN certificate), receipts 934
# (honest pinned-scope flag); receipts 775 is the journal emit those fixes report through (logs if the journal fails).
_TRIAGED: set[tuple[str, int]] = {
    ("kernel_authored_check_lifecycle", 70), ("kernel_authored_check_lifecycle", 518),
    ("kernel_authored_check_lifecycle", 520),
    ("kernel_check_receipts", 156), ("kernel_check_receipts", 213), ("kernel_check_receipts", 754),
    ("kernel_check_receipts", 775), ("kernel_check_receipts", 794), ("kernel_check_receipts", 934),
    ("kernel_check_receipts", 1247), ("kernel_check_receipts", 1317), ("kernel_check_receipts", 1380),
    ("kernel_check_receipts", 1393), ("kernel_check_receipts", 1409),
    ("kernel_check_truth_adapter", 69), ("kernel_check_truth_adapter", 95),
    ("kernel_check_truth_adapter", 408), ("kernel_check_truth_adapter", 417),
    ("kernel_test_run_spans", 251), ("kernel_test_run_spans", 294), ("kernel_test_run_spans", 301),
    ("kernel_test_run_spans", 314), ("kernel_test_run_spans", 321), ("kernel_test_run_spans", 339),
    ("kernel_test_run_spans", 362), ("kernel_test_run_spans", 376), ("kernel_test_run_spans", 435),
    ("kernel_test_run_spans", 525), ("kernel_test_run_spans", 565),
}


def test_checks_broad_excepts_are_triaged() -> None:
    found = _broad_except_sites(_MODULE_PATHS)
    expected = _PILOT | _TRIAGED
    assert found == expected, (
        f"untriaged: {sorted(found - expected)}; stale: {sorted(expected - found)}")


def test_checks_tags_point_at_broad_except_sites() -> None:
    """Any T8-<MODULE>-<line> tag in this file must name a real broad except line in its module."""
    broad = re.compile(r"^\s*except\s*(?:\(|Exception\b|BaseException\b|:)")
    source = Path(__file__).read_text(encoding="utf-8")
    for module, line in re.findall(r"T8-([A-Z0-9_]+)-(\d+)", source):
        path = ENGINE / f"{module.lower()}.py"
        lines = path.read_text(encoding="utf-8").splitlines()
        assert broad.match(lines[int(line) - 1]), f"T8-{module}-{line} does not point at a broad except"
