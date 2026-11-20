"""A failed durable handoff must terminate without releasing its slot."""
from __future__ import annotations

import ast
import errno
import json
import os
from pathlib import Path
import signal
import stat
import threading
from types import SimpleNamespace

from stateframe import _child, parallel_resources
from stateframe.parallel_worker import kernel_start_identity


ENDPOINT = "http://127.0.0.1:8080/v1"
SLOT = 3
INCARNATION = 7
HANDOFF_JOIN_SECONDS = 15


def test_persistent_directory_fsync_failure_is_bounded_and_fails_closed(
        tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "worker"
    process_dir = run_dir / "proc" / str(INCARNATION)
    process_dir.mkdir(parents=True)
    pid = os.getpid()
    identity = {"pid": pid, "pgid": pid, "kernel_identity": kernel_start_identity(pid)}
    (process_dir / "process.json").write_text(
        json.dumps({"incarnation": INCARNATION, **identity}), encoding="utf-8")

    lease_root = tmp_path / "slot-leases"
    monkeypatch.setattr(parallel_resources, "LEASE_ROOT", lease_root)
    monkeypatch.setenv("SF_SUBFRAME_INCARNATION", str(INCARNATION))
    monkeypatch.delenv("SF_SUBFRAME_SLOT_CLAIM", raising=False)
    monkeypatch.chdir(run_dir)
    lease = parallel_resources.lease_slot(ENDPOINT, [SLOT])
    assert lease is not None

    real_fsync = os.fsync

    def directory_fsync_fails(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError(errno.EIO, "injected directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(parallel_resources.os, "fsync", directory_fsync_fails)
    real_publish = parallel_resources._publish_slot_handoff
    outcomes: list[str] = []
    result: dict[str, BaseException | None] = {}

    def counted_publish(*args: object) -> None:
        try:
            real_publish(*args)
        except BaseException as error:  # noqa: BLE001 - retain the result of every attempt
            outcomes.append(type(error).__name__)
            raise
        outcomes.append("ok")

    monkeypatch.setattr(parallel_resources, "_publish_slot_handoff", counted_publish)

    def run_handoff() -> None:
        try:
            lease.handoff_to_host()
            result["outcome"] = None
        except BaseException as error:  # noqa: BLE001 - assert the terminal typed failure
            result["outcome"] = error

    worker = threading.Thread(target=run_handoff, daemon=True)
    worker.start()
    try:
        worker.join(HANDOFF_JOIN_SECONDS)
        assert not worker.is_alive(), "persistent directory fsync failure must terminate handoff"

        outcome = result.get("outcome")
        failure_type = getattr(parallel_resources, "SlotHandoffDurabilityError", None)
        assert isinstance(failure_type, type)
        assert isinstance(outcome, failure_type), f"expected typed durability failure, got {outcome!r}"
        assert outcome.slot == SLOT
        assert outcome.attempts == len(outcomes)
        assert outcome.attempts <= 16
        assert outcome.last_error.__class__ is OSError

        marker = lease.path.with_name(f"{lease.path.stem}.handoff.json")
        assert marker.is_file(), "failed durability retries must leave the visible claim in place"
        assert lease._descriptor >= 0, "exhaustion must retain the slot flock"
        competing_lease = parallel_resources.lease_slot(ENDPOINT, [SLOT])
        assert competing_lease is None, "a competing worker must not acquire the unresolved slot"
    finally:
        if worker.is_alive():
            # A red run uses the old unbounded loop; let it observe a successful barrier before exit.
            monkeypatch.setattr(parallel_resources.os, "fsync", real_fsync)
            worker.join(timeout=6)
        if not worker.is_alive():
            lease.close()


def test_child_cleanup_reports_typed_handoff_failure_without_closing_lease() -> None:
    tree = ast.parse(Path(_child.__file__).read_text(encoding="utf-8"))
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "_main")
    outer = next(node for node in main.body
                 if isinstance(node, ast.Try)
                 and any(isinstance(child, ast.Try) for child in node.finalbody))
    handoff_error = parallel_resources.SlotHandoffDurabilityError(
        SLOT, 8, OSError(errno.EIO, "injected directory fsync failure"))

    class Session:
        containment_verified = False

        def stop_all(self):
            return SimpleNamespace(ok=False, detail="owned descendant remains")

        def close(self):
            return None

    class Lease:
        close_calls = 0
        handoff_calls = 0

        def close(self):
            self.close_calls += 1

        def handoff_to_host(self):
            self.handoff_calls += 1
            raise handoff_error

    lease = Lease()
    events: list[tuple[tuple[object, ...], dict[str, object]]] = []
    namespace = {
        "mutation_runtime": None,
        "hook_wiring": None,
        "session": Session(),
        "slot_lease": lease,
        "result": {"result": "SUBMITTED"},
        "status": 0,
        "previous_term": signal.getsignal(signal.SIGTERM),
        "signal": signal,
        "emit": lambda *args, **kwargs: events.append((args, kwargs)),
        "SlotHandoffDurabilityError": parallel_resources.SlotHandoffDurabilityError,
        "_reason_log": _child._reason_log,
    }
    exec(compile(ast.Module(body=outer.finalbody, type_ignores=[]),
                 "<child cleanup>", "exec"), namespace)

    assert lease.handoff_calls == 1
    assert lease.close_calls == 0, "typed handoff failure must retain the slot lease"
    assert namespace["result"] == {
        "result": "ENGINE_ERROR",
        "error": "SlotHandoffDurabilityError",
    }
    assert namespace["status"] == 1
    assert any(kwargs.get("error") == "SlotHandoffDurabilityError" for _, kwargs in events)
