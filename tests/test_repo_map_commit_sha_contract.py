"""Fail-open commit lookup contract for repo-map intake caching."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ENGINE = SRC / "stateframe" / "engine"
_WarningCall = tuple[tuple[object, ...], dict[str, object]]


@pytest.fixture(scope="module")
def repo_map_modules():
    for path in (str(SRC), str(ENGINE)):
        if path not in sys.path:
            sys.path.insert(0, path)

    from stateframe import ts_grammars

    ts_grammars.install_as_language_pack()
    return (
        importlib.import_module("mc_repo_intel_vendor.repo_map"),
        importlib.import_module("kernel_git_view"),
        importlib.import_module("kernel_repo_intel"),
    )


class _Journal:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def emit(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))


def _install_query_failure(monkeypatch: pytest.MonkeyPatch, git_view: object,
                           error: BaseException) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise error

    monkeypatch.setattr(git_view, "legacy_git_query", fail)


def _capture_warnings(monkeypatch: pytest.MonkeyPatch, repo_map: object) -> list[_WarningCall]:
    calls: list[_WarningCall] = []
    monkeypatch.setattr(
        repo_map.logger, "warning",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    return calls


def _assert_failure_record(journal: _Journal, warning_calls: list[_WarningCall],
                           exc_type: str, errno: int | None) -> None:
    assert len(warning_calls) == 1
    assert "commit SHA lookup failed" in warning_calls[0][0][0]
    assert len(journal.events) == 1
    event, fields = journal.events[0]
    assert event == "repo_map_commit_sha_unavailable"
    assert set(fields) == {"reason", "exc_type", "errno"}
    assert fields == {
        "reason": "commit_sha_lookup_failed",
        "exc_type": exc_type,
        "errno": errno,
    }


def test_oserror_returns_empty_sha_and_records_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repo_map_modules,
) -> None:
    repo_map, git_view, _kernel_repo_intel = repo_map_modules
    error = OSError(13, "unavailable")
    _install_query_failure(monkeypatch, git_view, error)
    warning_calls = _capture_warnings(monkeypatch, repo_map)
    journal = _Journal()

    assert repo_map.get_current_commit_sha(str(tmp_path), journal=journal) == ""

    _assert_failure_record(journal, warning_calls, type(error).__name__, 13)


def test_valueerror_returns_empty_sha_and_records_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repo_map_modules,
) -> None:
    repo_map, git_view, _kernel_repo_intel = repo_map_modules
    error = ValueError("invalid git view")
    _install_query_failure(monkeypatch, git_view, error)
    warning_calls = _capture_warnings(monkeypatch, repo_map)
    journal = _Journal()

    assert repo_map.get_current_commit_sha(str(tmp_path), journal=journal) == ""

    _assert_failure_record(journal, warning_calls, "ValueError", None)


def test_git_object_error_returns_empty_sha_and_records_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repo_map_modules,
) -> None:
    repo_map, git_view, _kernel_repo_intel = repo_map_modules
    error = git_view.GitObjectError("git object unavailable")
    _install_query_failure(monkeypatch, git_view, error)
    warning_calls = _capture_warnings(monkeypatch, repo_map)
    journal = _Journal()

    assert repo_map.get_current_commit_sha(str(tmp_path), journal=journal) == ""

    _assert_failure_record(journal, warning_calls, "GitObjectError", None)


def test_success_returns_first_forty_sha_characters_without_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repo_map_modules,
) -> None:
    repo_map, git_view, _kernel_repo_intel = repo_map_modules
    sha = "a" * 40
    monkeypatch.setattr(git_view, "legacy_git_query", lambda *_args, **_kwargs: sha + "extra")
    warning_calls = _capture_warnings(monkeypatch, repo_map)
    journal = _Journal()

    assert repo_map.get_current_commit_sha(str(tmp_path), journal=journal) == sha
    assert warning_calls == []
    assert journal.events == []


@pytest.mark.parametrize(
    "error_name",
    [
        pytest.param("exec-session-error", id="exec-session-error"),
        pytest.param("other-runtime-error", id="other-runtime-error"),
        pytest.param("probe-cleanup-error", id="probe-cleanup-error"),
    ],
)
def test_other_exceptions_propagate_without_warning_or_event(
    error_name: str, monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path, repo_map_modules,
) -> None:
    repo_map, git_view, _kernel_repo_intel = repo_map_modules
    error_type: type[BaseException] = {
        "exec-session-error": importlib.import_module("kernel_exec_session").ExecSessionError,
        "other-runtime-error": RuntimeError,
        "probe-cleanup-error": importlib.import_module("kernel_safe_probe").ProbeCleanupError,
    }[error_name]
    error = error_type("secret detail")
    _install_query_failure(monkeypatch, git_view, error)
    warning_calls = _capture_warnings(monkeypatch, repo_map)
    journal = _Journal()

    with pytest.raises(error_type):
        repo_map.get_current_commit_sha(str(tmp_path), journal=journal)

    assert warning_calls == []
    assert journal.events == []


def test_hostile_exception_rendering_and_metaclass_are_not_used(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repo_map_modules,
) -> None:
    repo_map, git_view, _kernel_repo_intel = repo_map_modules
    render_calls: list[str] = []

    class _HostileOSError(OSError):
        def __str__(self) -> str:
            render_calls.append("str")
            raise AssertionError("exception string was rendered")

        def __repr__(self) -> str:
            render_calls.append("repr")
            raise AssertionError("exception repr was rendered")

        @property
        def errno(self) -> int:
            render_calls.append("errno-property")
            raise AssertionError("overridden errno property was read")

    class _HostileMeta(type):
        @property
        def __name__(cls) -> str:
            raise AssertionError("metaclass __name__ was read")

        @property
        def __mro__(cls) -> tuple[type, ...]:
            raise AssertionError("metaclass __mro__ was read")

    class _HostileMetaValueError(ValueError, metaclass=_HostileMeta):
        pass

    warning_calls = _capture_warnings(monkeypatch, repo_map)
    journal = _Journal()
    error = _HostileOSError(23, "secret")
    _install_query_failure(monkeypatch, git_view, error)

    assert repo_map.get_current_commit_sha(str(tmp_path), journal=journal) == ""
    _assert_failure_record(journal, warning_calls, "_HostileOSError", 23)
    assert render_calls == []

    warning_calls.clear()
    journal.events.clear()
    meta_error = _HostileMetaValueError("secret")
    _install_query_failure(monkeypatch, git_view, meta_error)
    assert repo_map.get_current_commit_sha(str(tmp_path), journal=journal) == ""
    _assert_failure_record(journal, warning_calls, "_HostileMetaValueError", None)


def test_path_named_exception_class_uses_invalid_type_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repo_map_modules,
) -> None:
    repo_map, git_view, _kernel_repo_intel = repo_map_modules
    path_named_error = type("../secret", (ValueError,), {})("hidden")
    _install_query_failure(monkeypatch, git_view, path_named_error)
    warning_calls = _capture_warnings(monkeypatch, repo_map)
    journal = _Journal()

    assert repo_map.get_current_commit_sha(str(tmp_path), journal=journal) == ""

    _assert_failure_record(journal, warning_calls, "<invalid>", None)


def test_missing_explicit_journal_uses_active_runtime_journal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repo_map_modules,
) -> None:
    repo_map, git_view, _kernel_repo_intel = repo_map_modules
    error = ValueError("invalid git view")
    _install_query_failure(monkeypatch, git_view, error)
    warning_calls = _capture_warnings(monkeypatch, repo_map)
    journal = _Journal()
    pilot = importlib.import_module("kernel_pilot")
    token = pilot.ACTIVE_RUNTIME.set(SimpleNamespace(journal=journal))
    try:
        assert repo_map.get_current_commit_sha(str(tmp_path)) == ""
    finally:
        pilot.ACTIVE_RUNTIME.reset(token)

    _assert_failure_record(journal, warning_calls, "ValueError", None)


def test_no_active_runtime_keeps_warning_without_journal_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repo_map_modules,
) -> None:
    repo_map, git_view, _kernel_repo_intel = repo_map_modules
    _install_query_failure(monkeypatch, git_view, ValueError("invalid git view"))
    warning_calls = _capture_warnings(monkeypatch, repo_map)
    pilot = importlib.import_module("kernel_pilot")
    token = pilot.ACTIVE_RUNTIME.set(None)
    try:
        assert repo_map.get_current_commit_sha(str(tmp_path)) == ""
    finally:
        pilot.ACTIVE_RUNTIME.reset(token)

    assert len(warning_calls) == 1


def test_intake_orientation_forwards_explicit_journal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repo_map_modules,
) -> None:
    _repo_map, _git_view, kernel_repo_intel = repo_map_modules
    journal = _Journal()
    observed: list[object] = []

    def current_commit(_root: str, *, journal: object | None = None) -> str:
        observed.append(journal)
        return "a" * 40

    monkeypatch.setattr(kernel_repo_intel, "get_current_commit_sha", current_commit)
    monkeypatch.setattr(
        kernel_repo_intel, "build_repo_map_with_stats",
        lambda *_args, **_kwargs: ("", {"source": "test"}),
    )
    monkeypatch.setattr(kernel_repo_intel, "_ensure_focus_entry", lambda *_args: "")
    monkeypatch.setattr(kernel_repo_intel, "bound_repo_map", lambda *_args, **_kwargs: "bounded map")
    monkeypatch.setattr(kernel_repo_intel, "_journal_repo_map_receipt", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(kernel_repo_intel, "_emit_for_root", lambda *_args, **_kwargs: None)
    nav_basics = importlib.import_module("kernel_nav_basics")
    monkeypatch.setattr(nav_basics, "render_intake_orientation", lambda *_args, **_kwargs: "orientation")

    assert kernel_repo_intel.intake_orientation(tmp_path, journal=journal) == "orientation"
    assert observed == [journal]


def test_unguarded_control_fails_the_empty_on_failure_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repo_map_modules,
) -> None:
    repo_map, git_view, _kernel_repo_intel = repo_map_modules
    error = OSError(5, "unavailable")
    _install_query_failure(monkeypatch, git_view, error)
    monkeypatch.setattr(repo_map.logger, "warning", lambda *_args, **_kwargs: None)

    def unguarded_lookup(workspace_dir: str) -> str:
        result = git_view.legacy_git_query(Path(workspace_dir), ["rev-parse", "HEAD"])
        return result.strip()[:40] if result else ""

    def satisfies_empty_on_failure(lookup: Callable[[str], str]) -> bool:
        try:
            return lookup(str(tmp_path)) == ""
        except (OSError, ValueError, git_view.GitObjectError):
            return False

    assert satisfies_empty_on_failure(repo_map.get_current_commit_sha)
    assert not satisfies_empty_on_failure(unguarded_lookup)


def test_ordinary_sink_errors_do_not_replace_empty_sha(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repo_map_modules,
) -> None:
    repo_map, git_view, _kernel_repo_intel = repo_map_modules
    _install_query_failure(monkeypatch, git_view, ValueError("invalid git view"))

    def fail_sink(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("sink unavailable")

    class _FailingJournal:
        def emit(self, *_args: object, **_kwargs: object) -> None:
            fail_sink()

    monkeypatch.setattr(repo_map.logger, "warning", fail_sink)
    assert repo_map.get_current_commit_sha(str(tmp_path), journal=_FailingJournal()) == ""


def test_logger_base_exception_propagates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repo_map_modules,
) -> None:
    repo_map, git_view, _kernel_repo_intel = repo_map_modules
    _install_query_failure(monkeypatch, git_view, ValueError("invalid git view"))
    journal = _Journal()

    def interrupt(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(repo_map.logger, "warning", interrupt)
    with pytest.raises(KeyboardInterrupt):
        repo_map.get_current_commit_sha(str(tmp_path), journal=journal)
    assert journal.events == []


def test_journal_base_exception_propagates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repo_map_modules,
) -> None:
    repo_map, git_view, _kernel_repo_intel = repo_map_modules
    _install_query_failure(monkeypatch, git_view, ValueError("invalid git view"))
    monkeypatch.setattr(repo_map.logger, "warning", lambda *_args, **_kwargs: None)

    class _ExitJournal:
        def emit(self, *_args: object, **_kwargs: object) -> None:
            raise SystemExit(7)

    with pytest.raises(SystemExit) as error:
        repo_map.get_current_commit_sha(str(tmp_path), journal=_ExitJournal())
    assert error.value.code == 7
