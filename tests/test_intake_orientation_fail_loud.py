from __future__ import annotations

import builtins
import errno
import importlib
import inspect
import json
import logging
import pathlib
import sys
import tempfile
from typing import Callable

import pytest

from stateframe.engine.probes import react_repair
from stateframe.scripted_endpoint import Script, serve
from stateframe.task_row import build_row
from stateframe.workcopy import create
from _journal import measurement_events


MARKER = "[NAVBASICS intake orientation]"
_EVENT_ENVELOPE_FIELDS = {
    "schema", "kind", "run_id", "task_id", "event_id", "sequence", "timestamp_ns",
    "code_hash", "release_id", "harness_version",
}
_INTAKE_FAILURE_FIELDS = {"exc_type", "step", "errno"}


class _StopAfterFirstRequest(BaseException):
    """Stop the real worker after the scripted endpoint records request 1."""


class _RaisingNameMeta(type):
    def __getattribute__(cls, name: str) -> object:
        if name == "__name__":
            raise KeyboardInterrupt
        return super().__getattribute__(name)


class _HostileDiagnosticError(Exception, metaclass=_RaisingNameMeta):
    def __str__(self) -> str:
        raise KeyboardInterrupt

    def __repr__(self) -> str:
        raise KeyboardInterrupt


class _HostileNamePropertyMeta(type):
    @property
    def __name__(cls) -> str:
        raise KeyboardInterrupt


class _HostileNamePropertyError(Exception, metaclass=_HostileNamePropertyMeta):
    pass


_HOSTILE_MRO_PROPERTY_CALLS: list[bool] = []


class _HostileMroPropertyMeta(type):
    @property
    def __mro__(cls) -> tuple[type, ...]:
        _HOSTILE_MRO_PROPERTY_CALLS.append(True)
        raise KeyboardInterrupt


class _HostileMroPropertyError(Exception, metaclass=_HostileMroPropertyMeta):
    pass


class _RaisingErrnoOSError(OSError):
    @property
    def errno(self) -> int:
        self.errno_property_calls += 1
        raise KeyboardInterrupt


class _CaptureJournal:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def emit(self, kind: str, **fields: object) -> None:
        self.events.append((kind, fields))


def _fixture_row(tmp_path: pathlib.Path, task_id: str) -> dict[str, object]:
    repository = tmp_path / "fixture-repo"
    (repository / "app").mkdir(parents=True)
    (repository / "tests").mkdir()
    (repository / "app" / "__init__.py").write_text("", encoding="utf-8")
    (repository / "app" / "answer.py").write_text(
        "def answer():\n    return 1\n", encoding="utf-8")
    (repository / "tests" / "test_answer.py").write_text(
        "from app.answer import answer\n\n"
        "def test_answer():\n    assert answer() == 2\n",
        encoding="utf-8",
    )
    work_copy = create(repository, tmp_path / "fixture-copy")
    return build_row(
        work_copy,
        task="Fix answer so the public test passes",
        target="app/answer.py",
        task_id=task_id,
    )


def _run_to_first_request(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    task_id: str,
    run_name: str = "run",
    row_mutator: Callable[[dict[str, object]], None] | None = None,
    runtime_mutator: Callable[[object], None] | None = None,
    expected_error: type[Exception] | None = None,
) -> tuple[dict[str, object] | None, list[dict[str, object]], list[dict[str, object]], pathlib.Path]:
    run_dir = tmp_path / run_name / "run"
    journal_parent = run_dir / "worker" / "measurement-runs"
    journal_parent.mkdir(parents=True)
    private_temp = tmp_path / run_name / "worker-tmp"
    private_temp.mkdir(parents=True)
    monkeypatch.setattr(tempfile, "tempdir", str(private_temp))
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / run_name / "stateframe-home"))
    # The legacy converge adapter interprets argv[1] as a task count at import.
    # Calling the worker directly from pytest must not hand it pytest's `-q`.
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])

    runtime_module = importlib.import_module("kernel_stateframe_runtime")
    runtime_type = runtime_module.StateFrameRuntime
    original_init = runtime_type.__init__

    def scoped_init(self: object, owner: object, **kwargs: object) -> None:
        kwargs["output"] = journal_parent
        original_init(self, owner, **kwargs)
        if runtime_mutator is not None:
            runtime_mutator(self)

    monkeypatch.setattr(runtime_type, "__init__", scoped_init)
    monkeypatch.setattr(react_repair, "MAX_STEPS", 1)

    script = Script({"model": "scripted", "steps": [], "tokenize": {"mode": "ok"}})
    server = serve(script)
    monkeypatch.setattr(
        react_repair, "API",
        f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
    )
    monkeypatch.setattr(react_repair, "MODEL", "scripted")
    monkeypatch.setattr(react_repair, "KEY", "")

    original_ask_tools = react_repair.ask_tools

    def stop_after_request(*args: object, **kwargs: object) -> object:
        original_ask_tools(*args, **kwargs)
        raise _StopAfterFirstRequest

    monkeypatch.setattr(react_repair, "ask_tools", stop_after_request)
    row = _fixture_row(tmp_path / run_name, task_id)
    if row_mutator is not None:
        row_mutator(row)
    pilot = importlib.import_module("kernel_pilot")
    measurement = importlib.import_module("kernel_measurement")
    identity = importlib.import_module("kernel_verdict_identity")
    for context_var in (
        pilot.ACTIVE_PARENT,
        pilot.ACTIVE_RUNTIME,
        measurement.ACTIVE_REQUEST,
        identity._ACTIVE_PROVIDER,
    ):
        context_var.set(None)
    try:
        if expected_error is None:
            with pytest.raises(_StopAfterFirstRequest):
                react_repair.work(row)
        else:
            with pytest.raises(expected_error):
                react_repair.work(row)
    finally:
        server.shutdown()
        server.server_close()
        for context_var in (
            pilot.ACTIVE_PARENT,
            pilot.ACTIVE_RUNTIME,
            measurement.ACTIVE_REQUEST,
            identity._ACTIVE_PROVIDER,
        ):
            context_var.set(None)

    if expected_error is not None:
        assert not script.requests, "worker should fail before making request 1"
        return None, measurement_events(run_dir), script.requests, run_dir

    assert len(script.requests) == 1, "the scripted endpoint must record request 1 exactly once"
    body = script.requests[0]
    messages = body.get("messages")
    assert isinstance(messages, list)
    user = next(message for message in messages if message.get("role") == "user")
    assert isinstance(user.get("content"), str)
    return body, measurement_events(run_dir), script.requests, run_dir


def _install_failure(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    step: str,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> object:
        raise error

    if step == "import":
        original_import = builtins.__import__

        def fail_nav_import(name: str, *args: object, **kwargs: object) -> object:
            frame = inspect.currentframe()
            caller = frame.f_back if frame is not None else None
            if (name == "kernel_nav_basics" and caller is not None
                    and caller.f_code is react_repair.work.__code__
                    and "_orientation" in caller.f_locals):
                raise error
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fail_nav_import)
    elif step == "selectors":
        original_selectors = react_repair._public_check_ids

        def fail_selectors(*args: object, **kwargs: object) -> object:
            frame = inspect.currentframe()
            caller = frame.f_back if frame is not None else None
            if (caller is not None and caller.f_code is react_repair.work.__code__
                    and caller.f_locals.get("_intake_step") == "selectors"):
                raise error
            return original_selectors(*args, **kwargs)

        monkeypatch.setattr(react_repair, "_public_check_ids", fail_selectors)
    elif step == "target_path":
        original_relative_to = pathlib.Path.relative_to

        def fail_target_path(path: pathlib.Path, *args: object, **kwargs: object) -> pathlib.Path:
            frame = inspect.currentframe()
            caller = frame.f_back if frame is not None else None
            if (caller is not None and caller.f_code is react_repair.work.__code__
                    and caller.f_locals.get("_intake_step") == "target_path"):
                raise error
            return original_relative_to(path, *args, **kwargs)

        monkeypatch.setattr(pathlib.Path, "relative_to", fail_target_path)
    elif step == "intake_orientation":
        repo_intel = importlib.import_module("kernel_repo_intel")
        monkeypatch.setattr(repo_intel, "intake_orientation", fail)
    else:
        raise AssertionError(f"unsupported injected step {step}")


def _assert_failure_evidence(
    body: dict[str, object],
    events: list[dict[str, object]],
    warning_records: list[logging.LogRecord],
    *,
    exc_type: str,
    step: str,
    input_message: str,
    errno_value: int | None = None,
) -> None:
    messages = body["messages"]
    assert isinstance(messages, list)
    user = next(message for message in messages if message.get("role") == "user")
    content = user.get("content")
    assert isinstance(content, str)
    assert MARKER not in content

    failures = [event for event in events if event.get("kind") == "intake_orientation_failed"]
    assert len(failures) == 1, f"expected one intake_orientation_failed event, found {len(failures)}"
    failure = failures[0]
    failure_fields = {key: value for key, value in failure.items()
                      if key not in _EVENT_ENVELOPE_FIELDS}
    assert failure_fields == {
        "exc_type": exc_type,
        "step": step,
        "errno": errno_value,
    }
    serialized_event = json.dumps(failure_fields)
    assert "/" not in serialized_event
    assert input_message not in serialized_event

    warnings = [record for record in warning_records
                if record.getMessage().startswith("NAVBASICS intake orientation failed:")]
    assert len(warnings) == 1, f"expected one NAVBASICS warning, found {len(warnings)}"
    warning = warnings[0]
    assert warning.levelno == logging.WARNING
    assert warning.args == (exc_type, step, errno_value)
    diagnostic_record = json.dumps({
        "msg": warning.msg,
        "args": warning.args,
        "rendered": warning.getMessage(),
    })
    assert "/" not in diagnostic_record
    assert input_message not in diagnostic_record


@pytest.mark.parametrize(("error_type", "step"), [
    (ImportError, "import"),
    (OSError, "selectors"),
    (FileNotFoundError, "selectors"),
    (TypeError, "target_path"),
    (ValueError, "intake_orientation"),
])
def test_orientation_failures_are_logged_and_journaled_once(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error_type: type[Exception],
    step: str,
) -> None:
    input_message = (
        f"diagnostic points at /tmp/intake-private.py and {tmp_path}/private/host-file.py")
    error = error_type(input_message)
    _install_failure(monkeypatch, error, step)
    caplog.set_level(logging.WARNING, logger=react_repair.__name__)

    body, events, _requests, _run_dir = _run_to_first_request(
        tmp_path, monkeypatch, task_id=f"intake-fail-{error_type.__name__}-{step}")

    assert body is not None
    _assert_failure_evidence(
        body, events, caplog.records,
        exc_type=error_type.__name__, step=step, input_message=input_message,
    )


@pytest.mark.parametrize("message", [
    "/tmp/my secret.py",
    "file:///tmp/my secret.py",
    r"C:\Users\mike\secret.py",
    r"\\server\share\my secret.py",
    "prefix/tmp/private",
])
def test_intake_diagnostics_never_include_exception_path_text(
    message: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    journal = _CaptureJournal()
    caplog.set_level(logging.WARNING, logger=react_repair.__name__)

    react_repair._record_intake_orientation_failure(
        ValueError(message), "intake_orientation", journal)

    assert len(journal.events) == 1
    kind, fields = journal.events[0]
    assert kind == "intake_orientation_failed"
    assert set(fields) == _INTAKE_FAILURE_FIELDS
    assert fields == {"exc_type": "ValueError", "step": "intake_orientation", "errno": None}
    serialized_event = json.dumps(fields)
    assert "/" not in serialized_event
    assert message not in serialized_event

    warning = next(
        record for record in caplog.records
        if record.getMessage().startswith("NAVBASICS intake orientation failed:")
    )
    diagnostic_record = json.dumps({
        "msg": warning.msg,
        "args": warning.args,
        "rendered": warning.getMessage(),
    })
    assert "/" not in diagnostic_record
    assert message not in diagnostic_record


@pytest.mark.parametrize("class_name", [
    "/tmp/private-exception-text.py",
    "Private Exception",
    "A" * 101,
    "PrívateException",
], ids=["path", "spaces", "overlong", "non-ascii"])
def test_intake_recorder_rejects_unsafe_exception_type_names(
    class_name: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    journal = _CaptureJournal()
    caplog.set_level(logging.WARNING, logger=react_repair.__name__)
    error_type = type(class_name, (ValueError,), {})

    react_repair._record_intake_orientation_failure(
        error_type("secret"), "intake_orientation", journal)

    assert journal.events == [(
        "intake_orientation_failed",
        {"exc_type": "<invalid>", "step": "intake_orientation", "errno": None},
    )]
    event_record = json.dumps(journal.events)
    assert class_name not in event_record

    warning = next(
        record for record in caplog.records
        if record.getMessage().startswith("NAVBASICS intake orientation failed:")
    )
    assert warning.args == ("<invalid>", "intake_orientation", None)
    diagnostic_record = json.dumps({
        "msg": warning.msg,
        "args": warning.args,
        "rendered": warning.getMessage(),
    })
    assert class_name not in diagnostic_record


def test_intake_recorder_never_renders_hostile_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    journal = _CaptureJournal()
    caplog.set_level(logging.WARNING, logger=react_repair.__name__)

    react_repair._record_intake_orientation_failure(
        _HostileDiagnosticError("secret"), "intake_orientation", journal)

    assert journal.events == [(
        "intake_orientation_failed",
        {"exc_type": "_HostileDiagnosticError", "step": "intake_orientation", "errno": None},
    )]
    warning = next(
        record for record in caplog.records
        if record.getMessage().startswith("NAVBASICS intake orientation failed:")
    )
    assert warning.args == ("_HostileDiagnosticError", "intake_orientation", None)


def test_intake_recorder_bypasses_hostile_metaclass_name_property() -> None:
    journal = _CaptureJournal()

    react_repair._record_intake_orientation_failure(
        _HostileNamePropertyError("secret"), "intake_orientation", journal)

    assert journal.events == [(
        "intake_orientation_failed",
        {"exc_type": "_HostileNamePropertyError", "step": "intake_orientation", "errno": None},
    )]


def test_intake_recorder_bypasses_hostile_metaclass_mro_property() -> None:
    journal = _CaptureJournal()
    _HOSTILE_MRO_PROPERTY_CALLS.clear()

    react_repair._record_intake_orientation_failure(
        _HostileMroPropertyError("secret"), "intake_orientation", journal)

    assert journal.events == [(
        "intake_orientation_failed",
        {"exc_type": "_HostileMroPropertyError", "step": "intake_orientation", "errno": None},
    )]
    assert _HOSTILE_MRO_PROPERTY_CALLS == []


def test_intake_recorder_reads_oserror_errno_from_base_descriptor() -> None:
    journal = _CaptureJournal()

    react_repair._record_intake_orientation_failure(
        OSError(errno.ENOENT, "x"), "intake_orientation", journal)

    assert journal.events == [(
        "intake_orientation_failed",
        {"exc_type": "FileNotFoundError", "step": "intake_orientation", "errno": 2},
    )]


def test_intake_recorder_handles_overridden_oserror_errno_property() -> None:
    journal = _CaptureJournal()
    error = _RaisingErrnoOSError(errno.ENOENT, "x")
    error.errno_property_calls = 0

    react_repair._record_intake_orientation_failure(
        error,
        "intake_orientation",
        journal,
    )

    assert len(journal.events) == 1
    kind, fields = journal.events[0]
    assert kind == "intake_orientation_failed"
    assert set(fields) == _INTAKE_FAILURE_FIELDS
    assert fields["exc_type"] == "_RaisingErrnoOSError"
    assert fields["step"] == "intake_orientation"
    assert fields["errno"] == 2
    assert error.errno_property_calls == 0


def test_intake_recorder_propagates_keyboard_interrupt_from_logger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = _CaptureJournal()

    def interrupt(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(react_repair.logger, "warning", interrupt)

    with pytest.raises(KeyboardInterrupt):
        react_repair._record_intake_orientation_failure(
            ValueError("secret"), "intake_orientation", journal)

    assert journal.events == []


def test_intake_recorder_propagates_system_exit_from_journal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _SystemExitJournal:
        def emit(self, _kind: str, **_fields: object) -> None:
            raise SystemExit(7)

    monkeypatch.setattr(react_repair.logger, "warning", lambda *_args, **_kwargs: None)

    with pytest.raises(SystemExit) as error:
        react_repair._record_intake_orientation_failure(
            ValueError("secret"), "intake_orientation", _SystemExitJournal())

    assert error.value.code == 7


def test_success_flush_happens_after_fixture_seed_with_identity_on(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATEFRAME_VERDICT_IDENTITY", "on")
    repo_intel = importlib.import_module("kernel_repo_intel")
    monkeypatch.setattr(
        repo_intel,
        "get_current_commit_sha",
        lambda *_args, **_kwargs: "test-commit",
    )
    monkeypatch.setattr(
        repo_intel,
        "build_repo_map_with_stats",
        lambda *_args, **_kwargs: ("", {"source": "test"}),
    )

    identity_module = importlib.import_module("kernel_verdict_identity")
    pending_journal = importlib.import_module("kernel_silent_off_organs").PendingWiringJournal
    original_flush = pending_journal.flush
    flush_code_hashes: list[str] = []
    flush_organ_names: list[tuple[object, ...]] = []

    def capture_flush_code_hash(self: object, journal: object) -> int:
        provider = getattr(journal, "_verdict_identity_provider")
        flush_code_hashes.append(provider.current().code_hash)
        flush_organ_names.append(tuple(
            fields.get("organ") for kind, fields in getattr(self, "_events")
            if kind == "organ_wiring_state"
        ))
        return original_flush(self, journal)

    monkeypatch.setattr(pending_journal, "flush", capture_flush_code_hash)

    def seed_initial_source(row: dict[str, object]) -> None:
        row["_fixture_initial_src"] = "def answer():\n    return 0\n"

    def neutralized_recorder(*_args: object, **_kwargs: object) -> None:
        return None

    def capture_identity(state: dict[str, object]) -> Callable[[object], None]:
        def capture(runtime: object) -> None:
            provider = getattr(runtime, "verdict_identity_provider")
            assert provider is not None
            state["provider"] = provider
            state["before_seed"] = provider.current().code_hash

        return capture

    neutralized_identity: dict[str, object] = {}

    # Neutralizing the failure recorder isolates successful request preparation;
    # the buffered journal flush must observe the fixture-seeded source in both runs.
    with monkeypatch.context() as neutralized_patch:
        neutralized_patch.setattr(
            react_repair, "_record_intake_orientation_failure", neutralized_recorder)
        neutralized_body, neutralized_events, _requests, _run_dir = _run_to_first_request(
            tmp_path, neutralized_patch, task_id="intake-orientation-success",
            run_name="neutralized",
            row_mutator=seed_initial_source,
            runtime_mutator=capture_identity(neutralized_identity),
        )

    changed_identity: dict[str, object] = {}

    body, events, _requests, _run_dir = _run_to_first_request(
        tmp_path, monkeypatch, task_id="intake-orientation-success", run_name="changed",
        row_mutator=seed_initial_source,
        runtime_mutator=capture_identity(changed_identity),
    )

    assert neutralized_body is not None and body is not None
    assert json.dumps(body).encode() == json.dumps(neutralized_body).encode()
    messages = body["messages"]
    assert isinstance(messages, list)
    user = next(message for message in messages if message.get("role") == "user")
    content = user.get("content")
    assert isinstance(content, str) and MARKER in content
    assert not any(event.get("kind") == "intake_orientation_failed" for event in events)
    assert not any(event.get("kind") == "intake_orientation_failed"
                   for event in neutralized_events)

    def identity_projection(
        journal_events: list[dict[str, object]],
    ) -> tuple[tuple[object, ...], ...]:
        return tuple((
            event.get("kind"),
            event.get("identity_hash"),
            event.get("code_hash"),
            event.get("release_id"),
            event.get("harness_version"),
        ) for event in journal_events)

    assert identity_projection(events) == identity_projection(neutralized_events)
    assert len(flush_code_hashes) == 2
    assert flush_organ_names == [
        ("expectation_ledger", "composed_cognition", "corpus_fixes"),
        ("expectation_ledger", "composed_cognition", "corpus_fixes"),
    ]

    for journal_events, identity_state, flush_code_hash, buffered_organs in (
        (neutralized_events, neutralized_identity, flush_code_hashes[0], flush_organ_names[0]),
        (events, changed_identity, flush_code_hashes[1], flush_organ_names[1]),
    ):
        provider = identity_state["provider"]
        before_seed = identity_state["before_seed"]
        assert isinstance(provider, identity_module.RunIdentityProvider)
        assert flush_code_hash != before_seed
        wiring_events = [event for event in journal_events
                         if event.get("organ") in buffered_organs]
        assert tuple(event.get("organ") for event in wiring_events) == buffered_organs
        assert [event.get("code_hash") for event in wiring_events] == [
            flush_code_hash,
            flush_code_hash,
            flush_code_hash,
        ]


def test_intake_failure_is_journaled_before_post_construction_oserror(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_failure(monkeypatch, ValueError("intake failed at:/tmp/private.py"), "intake_orientation")
    runtime_module = importlib.import_module("kernel_stateframe_runtime")
    runtime_type = runtime_module.StateFrameRuntime
    original_binding = runtime_type.binding

    def fail_after_construction(self: object, *args: object, **kwargs: object) -> object:
        if getattr(self, "_force_post_runtime_binding_failure", False):
            raise OSError("forced post-construction failure")
        return original_binding(self, *args, **kwargs)

    monkeypatch.setattr(runtime_type, "binding", fail_after_construction)

    def make_generic(row: dict[str, object]) -> None:
        row["runner_kind"] = "generic"
        row["test_command"] = []

    def fail_first_binding(runtime: object) -> None:
        setattr(runtime, "_force_post_runtime_binding_failure", True)

    body, events, requests, _run_dir = _run_to_first_request(
        tmp_path,
        monkeypatch,
        task_id="intake-failure-before-binding-oserror",
        row_mutator=make_generic,
        runtime_mutator=fail_first_binding,
        expected_error=OSError,
    )

    assert body is None and requests == []
    failures = [event for event in events if event.get("kind") == "intake_orientation_failed"]
    assert len(failures) == 1
    failure_fields = {key: value for key, value in failures[0].items()
                      if key not in _EVENT_ENVELOPE_FIELDS}
    assert failure_fields == {
        "exc_type": "ValueError",
        "step": "intake_orientation",
        "errno": None,
    }


def test_planted_suppressed_event_makes_failure_assertion_fail(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    error = ValueError("planted failure")
    _install_failure(monkeypatch, error, "intake_orientation")

    def suppress_failure_event(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(react_repair, "_record_intake_orientation_failure", suppress_failure_event)
    caplog.set_level(logging.WARNING, logger=react_repair.__name__)
    body, events, _requests, _run_dir = _run_to_first_request(
        tmp_path, monkeypatch, task_id="intake-orientation-planted-control")

    with pytest.raises(AssertionError, match="expected one intake_orientation_failed event"):
        _assert_failure_evidence(
            body, events, caplog.records,
            exc_type="ValueError", step="intake_orientation",
            input_message="planted failure",
        )
