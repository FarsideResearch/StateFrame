"""The runtime owns one host-authorship ledger per step and every capture refers to it."""
from __future__ import annotations

import ast
import inspect
import itertools
from types import SimpleNamespace

import pytest

import kernel_host_authorship as authorship
import kernel_measurement
import kernel_stateframe_runtime as runtime_module
from kernel_stateframe_runtime import StateFrameRuntime

ENTRY_POINTS = ("prepare", "_prepare_lean_plain_loop", "_prepare_continuation_boundary")


_SERIAL = itertools.count()


class _Stop(Exception):
    pass


def _runtime(tmp_path, step=3, **fields):
    runtime = StateFrameRuntime.__new__(StateFrameRuntime)
    runtime.owner = SimpleNamespace(current_step=step, task_row={})
    runtime.snapshot_id = "snapshot"
    runtime.endpoint_profile = None
    # One journal directory per runtime: a journal holds its directory's writer lease.
    home = tmp_path / f"journal-{next(_SERIAL)}"
    home.mkdir()
    runtime.journal = kernel_measurement.AuditJournal(home, f"host-authorship-{step}", {})
    runtime.system_outcome = None
    for name, value in fields.items():
        setattr(runtime, name, value)
    return runtime


def _stop(*_args, **_kwargs):
    raise _Stop


def _functions():
    tree = ast.parse(inspect.getsource(runtime_module))
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            found.setdefault(node.name, node)
    return found


def _calls(node, name):
    return [item for item in ast.walk(node) if isinstance(item, ast.Call)
            and getattr(item.func, "id", getattr(item.func, "attr", None)) == name]


@pytest.mark.parametrize("mode", ["shadow", "on"])
def test_ledger_opens_before_first_authored_row(tmp_path, monkeypatch, mode):
    monkeypatch.setenv(authorship.MODE_ENV, mode)
    # The open call is the first statement of each entry point, ahead of any row the host writes.
    functions = _functions()
    for name in ENTRY_POINTS:
        body = functions[name].body
        first = body[1] if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) else body[0]
        assert isinstance(first, ast.Expr) and _calls(first, "open_step"), name
    # And it has run before the first thing each entry point does next.
    import kernel_host_cache
    import kernel_lean
    runtime = _runtime(tmp_path, step=3)
    monkeypatch.setattr(kernel_host_cache, "clear_request_cache", _stop)
    with pytest.raises(_Stop):
        runtime.prepare([], uncapped=[], tools=[], provider_request=False)
    ledger = authorship.ledger_of(runtime)
    assert ledger is not None and ledger.step == 3 and ledger.rows == {}
    monkeypatch.setattr(kernel_lean, "plain_tool_surface", _stop)
    lean = _runtime(tmp_path, step=4)
    with pytest.raises(_Stop):
        lean._prepare_lean_plain_loop([], [], [], provider_request=True)
    assert authorship.ledger_of(lean).step == 4
    # A nested preparation of the same step keeps the object; the next step replaces it.
    runtime.continuation = None
    assert runtime._prepare_continuation_boundary([]) is None
    assert authorship.ledger_of(runtime) is ledger
    runtime.owner.current_step = 5
    assert runtime._prepare_continuation_boundary([]) is None
    assert authorship.ledger_of(runtime) is not ledger and authorship.ledger_of(runtime).step == 5


def _continuation(**flags):
    return SimpleNamespace(
        engine=SimpleNamespace(state=SimpleNamespace(active_id="root")),
        prepare=lambda _uncapped: None, _unresolved_effect_details=lambda _identity: flags.get("details", []),
        _dispatchable_repair=lambda _detail: False, completion_finding=lambda: "finding",
        completed=flags.get("completed", False), terminal_blocked=False)


def test_every_capture_of_a_step_holds_the_runtime_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv(authorship.MODE_ENV, "shadow")
    runtime = _runtime(tmp_path, step=6, continuation=_continuation(completed=True))
    done = runtime._prepare_continuation_boundary([])
    ledger = authorship.ledger_of(runtime)
    assert done.overflow == {"kind": "CONTINUATION_COMPLETE"} and done.capture._host_authorship is ledger
    runtime.continuation = _continuation(details=[{"status": "unknown", "tool": "file_create"}])
    runtime._continuation_prepare_guard_count = 0
    blocked = runtime._prepare_continuation_boundary([])
    assert blocked.overflow["kind"] == "CONTINUATION_BLOCKED" and blocked.capture._host_authorship is ledger
    main = runtime_module._prepare_request_capture(runtime)
    assert main._host_authorship is ledger and ledger.request_id == main.request_id
    unavailable = runtime_module._unavailable_frame_request(runtime, "reason")
    assert unavailable.capture._host_authorship is ledger
    # The main capture helper opens the step itself when no entry point ran first.
    late = _runtime(tmp_path, step=9)
    assert runtime_module._prepare_request_capture(late)._host_authorship is authorship.ledger_of(late)
    assert authorship.ledger_of(late).step == 9
    # Every capture construction in the module is bound: none can be added without a ledger.
    functions = _functions()
    sites = {name: len(_calls(node, "RequestCapture")) for name, node in functions.items()
             if _calls(node, "RequestCapture")}
    assert sites == {"_prepare_request_capture": 1, "_maybe_bind_quantity": 1,
                     "_prepare_lean_plain_loop": 1, "_prepare_continuation_boundary": 2, "prepare": 2}
    for name, count in sites.items():
        helper = {"_prepare_request_capture": "bind", "_maybe_bind_quantity": "_private_host_authorship"}.get(
            name, "_bind_host_authorship")
        assert len(_calls(functions[name], helper)) == count, name


def test_off_leaves_runtime_and_captures_untouched(tmp_path, monkeypatch):
    built = []
    monkeypatch.setattr(authorship.HostAuthorship, "__init__", lambda self, *a, **k: built.append(self))
    for value in (None, "off"):
        if value is None:
            monkeypatch.delenv(authorship.MODE_ENV, raising=False)
        else:
            monkeypatch.setenv(authorship.MODE_ENV, value)
        runtime = _runtime(tmp_path, step=6, continuation=_continuation(completed=True))
        before = set(vars(runtime))
        done = runtime._prepare_continuation_boundary([])
        main = runtime_module._prepare_request_capture(runtime)
        aux = runtime_module._private_host_authorship(
            kernel_measurement.RequestCapture(runtime.journal, 6, "snapshot"))
        runtime_module._mark_host_authorship(runtime, "prepare.final", [{"role": "user", "content": "x"}])
        assert done.capture._host_authorship is None and main._host_authorship is None
        assert aux._host_authorship is None
        assert "_host_authorship" not in vars(runtime)
        assert set(vars(runtime)) - before <= {"_continuation_prepare_guard_key", "_continuation_prepare_guard_count",
                                               "continuation_terminal"}
    assert built == []


def test_auxiliary_capture_gets_private_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv(authorship.MODE_ENV, "on")
    runtime = _runtime(tmp_path, step=7)
    parent = authorship.open_step(runtime, 7)
    capture = runtime_module._private_host_authorship(
        kernel_measurement.RequestCapture(runtime.journal, 7, "snapshot"))
    private = capture._host_authorship
    assert isinstance(private, authorship.HostAuthorship) and private is not parent
    assert private.private and private.step == 7 and private.request_id == capture.request_id
    assert authorship.ledger_of(runtime) is parent  # the parent's ledger is not replaced or shared
    authorship.author_row(parent, "user", authorship.builder().host("[Panel]"), kind="panel", site="t")
    assert authorship.row_kind(private, {"role": "user", "content": "[Panel]"}) is None
    # The auxiliary sender's capture factory goes through the private helper.
    lambdas = [node for node in ast.walk(_functions()["_maybe_bind_quantity"]) if isinstance(node, ast.Lambda)
               and _calls(node, "RequestCapture")]
    assert len(lambdas) == 1 and _calls(lambdas[0], "_private_host_authorship")


def test_pass_marks_are_recorded_in_shadow_only(tmp_path, monkeypatch):
    rows = [{"role": "user", "content": "x"}]
    for mode, expected in (("shadow", ["prepare.final"]), ("on", [])):
        monkeypatch.setenv(authorship.MODE_ENV, mode)
        runtime = _runtime(tmp_path, step=8)
        ledger = authorship.open_step(runtime, 8)
        runtime_module._mark_host_authorship(runtime, "prepare.final", rows)
        assert [tag for tag, _keys in ledger.marks] == expected
    functions = _functions()
    assert len(_calls(functions["prepare"], "_mark_host_authorship")) >= 4
    assert len(_calls(functions["_prepare_lean_plain_loop"], "_mark_host_authorship")) >= 2


@pytest.mark.parametrize("mode", ["off", "on"])
def test_public_review_capture_gets_private_ledger(tmp_path, monkeypatch, mode):
    import kernel_public_review
    import kernel_request_kind
    monkeypatch.setenv(authorship.MODE_ENV, mode)
    runtime = _runtime(tmp_path, step=11)
    parent = authorship.open_step(runtime, 11)
    seen = []

    def stop_at_bind(capture, purpose):
        seen.append((capture, purpose))
        raise _Stop

    monkeypatch.setattr(kernel_request_kind, "bind_auxiliary", stop_at_bind)
    review = kernel_public_review.PublicReview.__new__(kernel_public_review.PublicReview)
    review.runtime = runtime
    with pytest.raises(_Stop):
        review._model([{"role": "user", "content": "review"}], "review", "http://endpoint", "key")
    (capture, purpose), = seen
    assert purpose == "public_review"
    if mode == "off":
        assert parent is None and capture._host_authorship is None
        assert "_host_authorship" not in vars(runtime)
        return
    private = capture._host_authorship
    assert isinstance(private, authorship.HostAuthorship) and private.private
    assert private is not parent and authorship.ledger_of(runtime) is parent
    assert private.step == 11 and private.request_id == capture.request_id
