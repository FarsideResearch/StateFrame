"""Engine child process: runs one StateFrame task from a prepared run directory.

Started by ``stateframe run`` as ``<python> -B _child.py <run-dir>``. It uses
stdlib-only package helpers and the vendored engine directory named in the spec.

What it adds around the engine, mirroring the reference runner's ``checked_ask``:
* the declared environment (product profile + run settings) is installed before any
  engine import, after purging inherited engine keys;
* every provider-bound request goes through the frame-only presentation preflight; a frame
  that cannot be built fails the request (no silent history fallback);
* every physical chat send is recorded byte-for-byte (request identity JSON and the raw
  response) under ``wire/``, and a one-line turn log is printed for the parent to stream.
The API key is read from the environment variable named in the spec and is never written.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
from contextlib import suppress
from pathlib import Path

if __package__:
    from .cache_metrics import normalize
    from .engine import kernel_reason_log as _reason_log
    from .engine.kernel_safe_paths import safe_home
    from .replay_tape import (
        CheckpointUnfaithful,
        Replayer,
        ReplayUnfaithful,
        mask_volatile,
    )
    from .skills import SkillError, skills_digest, verify_records
    from .state_dirs import private_directory
else:  # Executed by path under the selected project interpreter.
    # The interpreter may also have another StateFrame installation. Package
    # imports used by subframes must belong to this child and its adjacent engine.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cache_metrics import normalize
    from engine import kernel_reason_log as _reason_log
    from engine.kernel_safe_paths import safe_home
    from replay_tape import (
        CheckpointUnfaithful,
        Replayer,
        ReplayUnfaithful,
        mask_volatile,
    )
    from skills import SkillError, skills_digest, verify_records
    from state_dirs import private_directory

# The wrappers' own state (send counters, the id stream position, ...). An engine checkpoint saves
# these by value and a checkpoint restore refills them in place (kernel_checkpoint participants), so
# the re-installed wrappers continue exactly where the interrupted attempt's wrappers were.
HOST_STATE: dict[str, dict] = {}
# A resumed run's time left at its last completed step ({"seconds": float, "started": bool}).
_RESUMED_TIME: dict[str, object] = {}



def recorded_stable_wire_decision(recorded: bytes | None) -> str | None:
    """Stable-wire decision a replayed send must reproduce, from its recorded request.

    A projection was "applied". A plain request is a stable-wire fallback ("plain")
    only when stable wire is on for this run; otherwise stable wire never ran. Calling
    it "plain" there switched replay onto the stable-wire path, which skips final
    source ordering, so the first live request after replay laid out its source
    windows differently from the request that was in flight.
    """
    if recorded is None:
        return None
    try:
        request = json.loads(recorded)
    except (ValueError, RecursionError):
        return None
    messages = request.get("messages") if isinstance(request, dict) else None
    if not isinstance(messages, list):
        return None
    from kernel_stable_wire import is_stable_wire_projection, mode as stable_wire_mode
    # Inspect recorded structure without consulting the replay decision provider.
    if is_stable_wire_projection(messages, recorded=True):
        return "applied"
    return "plain" if stable_wire_mode() == "on" else None


def _failed_send_record(send: int, wire: bytes, layout: dict[str, object],
                        stable_record: dict[str, object],
                        transport_attempts: list[dict[str, object]], error: BaseException,
                        endpoint_identity: dict[str, str] | None) -> dict[str, object]:
    """Build the durable sends.jsonl row for an unsuccessful physical send."""
    record: dict[str, object] = {
        "send": send,
        "request_bytes": len(wire),
        "request_sha256": hashlib.sha256(wire).hexdigest(),
        **layout,
        **stable_record,
        "transport_attempts": transport_attempts,
        "error": type(error).__name__,
    }
    if endpoint_identity is not None:
        record["endpoint_identity"] = dict(endpoint_identity)
    return record


def _start_resumed_deadline() -> None:
    """Once replay is over, give the run the time it had left, counted from now (once)."""
    seconds = _RESUMED_TIME.get("seconds")
    if seconds is None or _RESUMED_TIME.get("started"):
        return
    _RESUMED_TIME["started"] = True
    from kernel_read_fixes import resume_sandbox_deadline
    resume_sandbox_deadline(float(seconds))
    emit("resume_time_left_started", seconds_left=float(seconds))
CHECKPOINT_RESTORE_ENV = "STATEFRAME_CHECKPOINT_RESTORE"
RUNNER_HOME_IDENTITY_ENV = "STATEFRAME_RUNNER_HOME_IDENTITY"
# New-process authority, deliberately outside HOST_STATE/checkpoint participants.
_RUNNER_HOME_IDENTITY_CAPTURED = False
_RUNNER_HOME_IDENTITY_PIN: str | None = None

ENGINE_PREFIXES = ("DS_", "KERNEL_", "STATEFRAME_", "REACT_", "PILOT_", "UNIT_PLAN_")
LOG_PREFIX = "[sf]"


def emit(kind: str, **fields: object) -> None:
    """One machine-readable progress line for the parent (stdout)."""
    print(LOG_PREFIX + " " + json.dumps({"kind": kind, **fields}, ensure_ascii=False,
                                         default=str, separators=(",", ":")), flush=True)


def atomic_bytes(path: Path, data: bytes) -> None:
    """Durably publish exact bytes without exposing a partially written snapshot."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def emit_run_env_events(run: Path) -> None:
    """Forward host provisioning events to the durable engine log stream."""
    try:
        document = json.loads((run / "env-seed.json").read_text())
    except (OSError, ValueError):
        return
    if not isinstance(document, dict) or not isinstance(document.get("events", ()), list):
        return
    for event in document.get("events", ()):
        if isinstance(event, dict) and event.get("event") == "env_seed_quarantined":
            fields = {name: event[name] for name in ("key", "quarantine")
                      if isinstance(event.get(name), str)}
            emit("env_seed_quarantined", **fields)


def emit_resume_pending_check(spec: dict, resume: dict | None) -> None:
    """Record a routed resume that dropped an unproven in-flight check, on the child's event stream.

    Kept out of the engine's task row: anything there can reach request bytes, and a changed
    request would break the replay this resume depends on.
    """
    check = resume.get("pending_check") if spec.get("routes") and isinstance(resume, dict) else None
    if isinstance(check, dict):
        emit("routes_resume_pending_check_skipped", **check)


def _runner_home_identity_pin() -> str | None:
    """Capture inherited authority once, before any run-spec environment is installed.

    Interpreter wrappers may bootstrap the environment before main(). Retain that
    original pin privately, including its absence; later installs cannot replace it.
    """
    global _RUNNER_HOME_IDENTITY_CAPTURED, _RUNNER_HOME_IDENTITY_PIN
    inherited = os.environ.pop(RUNNER_HOME_IDENTITY_ENV, None)
    if not _RUNNER_HOME_IDENTITY_CAPTURED:
        _RUNNER_HOME_IDENTITY_PIN = inherited
        _RUNNER_HOME_IDENTITY_CAPTURED = True
    return _RUNNER_HOME_IDENTITY_PIN


def install_environment(spec: dict) -> None:
    _runner_home_identity_pin()
    checkpoint_off = os.environ.get("STATEFRAME_CHECKPOINT", "").strip().lower() in {"off", "0"}
    routed_keys: dict[str, str] = {}
    if isinstance(spec.get("routes"), dict):
        for settings in spec["routes"].get("roles", {}).values():
            name = settings.get("api_key_env") if isinstance(settings, dict) else None
            if isinstance(name, str) and name in os.environ:
                routed_keys[name] = os.environ[name]
    for key in list(os.environ):
        if key.startswith(ENGINE_PREFIXES) and not key.startswith("STATEFRAME_TASK_"):
            del os.environ[key]
    os.environ.pop("SF_PROMPT_CACHE_PIN_DEGRADED_REASON", None)
    os.environ.update({str(k): str(v) for k, v in spec["environment"].items()})
    # Legacy fixture authority is never available to runner-created children.
    os.environ.pop("STATEFRAME_CHECKPOINT_LEGACY_DIRECT_ENGINE", None)
    os.environ.pop(RUNNER_HOME_IDENTITY_ENV, None)
    if checkpoint_off:
        os.environ["STATEFRAME_CHECKPOINT"] = "off"
    os.environ.update(routed_keys)


def _install_primary_api_key_env(spec: dict) -> None:
    """Expose the run's primary credential variable name to checkpoint provenance."""
    name = spec.get("api_key_env")
    if isinstance(name, str) and name:
        os.environ["STATEFRAME_PRIMARY_API_KEY_ENV"] = name
    else:
        os.environ.pop("STATEFRAME_PRIMARY_API_KEY_ENV", None)


def _selected_resume_checkpoint(resume: dict | None) -> dict | None:
    checkpoint = resume.get("checkpoint") if resume is not None else None
    if not isinstance(checkpoint, dict) or checkpoint.get("status") == "skipped":
        return None
    return checkpoint


def configure_subframe_slot(run: Path, leased_slot: int, *, pinned: bool,
                            replayer: Replayer | None) -> int | None:
    """Preserve a recorded pin on replay; use the current endpoint policy for live sends."""
    historical_slot = run / "parallel-slot.json"
    if historical_slot.exists() or historical_slot.is_symlink():
        if historical_slot.is_symlink() or not historical_slot.is_file():
            raise RuntimeError("subframe historical slot artifact is unsafe")
        try:
            with historical_slot.open("rb") as source:
                recorded = source.read(257)
            if len(recorded) > 256:
                raise RuntimeError("subframe historical slot artifact exceeds 256 bytes")
            slot_record = json.loads(recorded)
        except (OSError, UnicodeError, ValueError) as error:
            raise RuntimeError("subframe historical slot artifact is unreadable or invalid") from error
        if not isinstance(slot_record, dict) or set(slot_record) not in ({"slot"}, {"slot", "pinned"}):
            raise RuntimeError("subframe historical slot artifact has invalid fields")
        old_slot = slot_record.get("slot")
        if type(old_slot) is not int or old_slot < 0:
            raise RuntimeError("subframe historical slot artifact is invalid")
        historical_pinned = slot_record.get("pinned", True)
        if type(historical_pinned) is not bool:
            raise RuntimeError("subframe historical slot pin status is invalid")
    else:
        if replayer is not None:
            raise RuntimeError("subframe historical slot artifact is missing for replay")
        old_slot = leased_slot
        historical_pinned = pinned
        with historical_slot.open("x", encoding="utf-8") as output:
            output.write(json.dumps({"slot": old_slot, "pinned": historical_pinned}))
            output.flush()
            os.fsync(output.fileno())
    live_slot = leased_slot if pinned else None
    replaying = replayer is not None and (replayer.active or replayer.pending is not None)
    active_slot = (old_slot if historical_pinned else None) if replaying else live_slot
    if active_slot is None:
        os.environ.pop("REACT_SLOT_ID", None)
    else:
        os.environ["REACT_SLOT_ID"] = str(active_slot)
    return live_slot


def _install_direct_opener() -> None:
    if __package__:
        from .engine.kernel_redirect_policy import (
            CHILD_DIRECT_OPENER_ATTRIBUTE, build_credential_scoped_opener,
        )
    else:
        from engine.kernel_redirect_policy import (
            CHILD_DIRECT_OPENER_ATTRIBUTE, build_credential_scoped_opener,
        )
    opener = build_credential_scoped_opener(proxy_mode="direct")
    setattr(opener, CHILD_DIRECT_OPENER_ATTRIBUTE, True)
    urllib.request.install_opener(opener)


def _usage(response: object) -> dict[str, object]:
    response = response if isinstance(response, dict) else {}
    usage = response.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    counts = normalize(usage, response.get("timings"))
    completion = usage.get("completion_tokens")
    return {"prompt_tokens": counts["total"],
            "completion_tokens": completion if type(completion) is int and completion >= 0 else None,
            "cached_tokens": counts["cached"]}


def _actions(response: object) -> list[str]:
    """Bounded diagnostics only; no normalization of the native response body."""
    if not isinstance(response, dict):
        return ["malformed response: expected object"]
    choices = response.get("choices", [])
    if not isinstance(choices, list):
        return ["malformed choices: expected array"]
    out = []
    # Bound both iteration and rendered output, even for an oversized malformed batch.
    for choice in choices[:16]:
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            out.append("malformed message: expected object")
            continue
        calls = choice["message"].get("tool_calls", [])
        if not isinstance(calls, list):
            out.append("malformed tool_calls: expected array")
            continue
        for call in calls[:16]:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict):
                out.append("malformed function: expected object")
                continue
            name = function.get("name")
            if not isinstance(name, str):
                out.append("malformed function name: expected string")
                continue
            name = name[:60]
            raw = function.get("arguments")
            try:
                arguments = json.loads(raw) if isinstance(raw, str) else None
            except (ValueError, RecursionError):
                arguments = None
            if not isinstance(arguments, dict):
                out.append(f"{name}: malformed arguments (expected JSON object)")
                continue
            target = arguments.get("path") or arguments.get("pattern") or arguments.get("selector") or ""
            if target and not isinstance(target, str):
                out.append(f"{name}: malformed target (expected string)")
            else:
                out.append(f"{name}({target[:60]})" if target else name)
        if len(calls) > 16:
            out.append("actions truncated")
    truncated = len(out) > 16 or len(choices) > 16
    return out[:16] + (["actions truncated"] if truncated else [])


class SubframeBudgetExhausted(BaseException):
    """A subframe has no reserved physical sends left."""


def install_wire_recorder(run: Path, chat_url: str, replayer: Replayer | None = None,
                          routed_endpoints: set[str] | None = None, hook_wiring=None,
                          *, send_ledger=None, send_cap: int | None = None) -> None:
    """Record every physical chat send: exact identity bytes, response bytes, usage.

    On ``stateframe resume`` a send the interrupted attempt already completed is answered from its
    recorded response (``replayer``), and is recorded here like any other send, marked ``replayed``.
    """
    import kernel_bounded_http

    if replayer is not None:
        # Replay is the only path that needs these engine imports. The recorded
        # request bytes decide the old order; current meter and switch do not.
        from kernel_cache_layout_candidate import (
            install_replay_layout_decision_provider,
            install_replay_stable_wire_decision_provider,
            install_replay_stable_wire_windows_provider,
            install_replay_stable_wire_reserves_provider,
        )
        from kernel_cache_layout_view import FINAL_LAYOUT_NAME
        from kernel_stateframe_final_message import AUTHORITY_OVERLAY_HEADER
        from kernel_stable_wire import recorded_window_layout

        def replay_layout_decision() -> str | None:
            recorded = replayer.request_for_layout_decision()
            if recorded is None:
                return None
            try:
                request = json.loads(recorded)
            except (ValueError, RecursionError):
                return None
            messages = request.get("messages") if isinstance(request, dict) else None
            final = messages[-1] if isinstance(messages, list) and messages else None
            if not isinstance(final, dict):
                return None
            content = final.get("content")
            has_overlay = isinstance(content, str) and AUTHORITY_OVERLAY_HEADER in content
            has_layout_name = final.get("name") == FINAL_LAYOUT_NAME
            if has_overlay and has_layout_name:
                return "layout"
            if not has_overlay and not has_layout_name:
                return "plain"
            return None

        install_replay_layout_decision_provider(replay_layout_decision)

        def replay_stable_wire_decision() -> str | None:
            return recorded_stable_wire_decision(replayer.request_for_layout_decision())

        install_replay_stable_wire_decision_provider(replay_stable_wire_decision)

        def replay_stable_wire_windows() -> tuple[list[str], int] | None:
            recorded = replayer.request_for_layout_decision()
            if recorded is None:
                return None
            try:
                request = json.loads(recorded)
            except (ValueError, RecursionError):
                return None
            messages = request.get("messages") if isinstance(request, dict) else None
            return recorded_window_layout(messages)

        install_replay_stable_wire_windows_provider(replay_stable_wire_windows)
        install_replay_stable_wire_reserves_provider(replayer.reserves_for_layout_decision)

    wire_dir = run / "wire"
    wire_dir.mkdir(exist_ok=True)
    original = kernel_bounded_http.request_bytes
    counter = {"n": 0}
    HOST_STATE["wire"] = counter
    wire_lock = threading.Lock()

    def route_identity(wire: bytes, endpoint: str) -> dict[str, str]:
        try:
            from .endpoint import normalize_base_url
        except ImportError:
            from stateframe.endpoint import normalize_base_url
        try:
            body = json.loads(wire)
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError("subframe model request is not valid JSON") from error
        model = body.get("model") if isinstance(body, dict) else None
        if not isinstance(model, str) or not model:
            raise ValueError("subframe model request lacks a model identity")
        return {"url": normalize_base_url(endpoint), "model": model}

    def final_layout_record(_wire: bytes) -> dict[str, str]:
        from kernel_measurement import ACTIVE_REQUEST

        capture = ACTIVE_REQUEST.get()
        record = getattr(capture, "_final_cache_layout_record", None)
        record = record if isinstance(record, dict) else {}
        result: dict[str, str] = {}
        # The engine capture is authoritative; inferring from message markers
        # would add metadata to legacy plain sends.
        layout = record.get("final_layout")
        # Omit the plain layout value to preserve the original 0.2.2 shape.
        if isinstance(layout, str) and layout in {"applied", "failed_open"}:
            result["final_layout"] = layout
            reason = record.get("final_layout_reason")
            if isinstance(reason, str) and reason:
                result["final_layout_reason"] = reason

        stable_wire = record.get("stable_wire")
        if isinstance(stable_wire, str) and stable_wire in {"applied", "plain", "failed_open"}:
            result["stable_wire"] = stable_wire
            reason = record.get("stable_wire_reason")
            if isinstance(reason, str) and reason:
                result["stable_wire_reason"] = reason
        return result

    def recorded(wire: bytes, endpoint: str, key: str, **kwargs):
        route = kwargs.pop("route_meta", None)
        if send_ledger is None and routed_endpoints is None and endpoint != chat_url:
            return original(wire, endpoint, key, **kwargs)
        if routed_endpoints is not None:
            if endpoint not in routed_endpoints or not isinstance(route, dict):
                raise ValueError("routed send has no authorized destination record")
            if route.get("endpoint") != endpoint:
                raise ValueError("routed send destination differs from route authority")
            try:
                wire_model = json.loads(wire).get("model")
            except (ValueError, AttributeError) as error:
                raise ValueError("routed send has no valid model field") from error
            if wire_model != route.get("model"):
                raise ValueError("routed send model differs from route authority")
        with wire_lock:
            if replayer is not None and replayer.strict and replayer.diverged is not None:
                raise CheckpointUnfaithful("This recording can't be replayed by this version. Start a fresh run. "
                    "an earlier rebuilt request differed; nothing more is sent")
            if replayer is not None and replayer.route_mismatch:
                raise CheckpointUnfaithful("This recording can't be replayed by this version. Start a fresh run. "
                    "an earlier routed replay differed; nothing more is sent")
            counter["n"] += 1
            index = counter["n"]
            stem = wire_dir / f"{index:04d}"
            from kernel_measurement import ACTIVE_REQUEST
            from kernel_stable_wire_record import validate_reserves_record

            reserves = validate_reserves_record(
                getattr(ACTIVE_REQUEST.get(), "_stable_wire_reserves_record", None))
            stable_record = {"stable_wire_reserves": reserves} if reserves is not None else {}
            if reserves is not None:
                # fsync before publishing the request, so a pending send has its trim authority.
                atomic_bytes(stem.with_suffix(".stable-wire.json"), json.dumps(
                    reserves, sort_keys=True, separators=(",", ":")).encode())
            identity = route_identity(wire, endpoint) if send_ledger is not None else None
            if identity is not None:
                identity_path = wire_dir / f"{index:04d}.endpoint.json"
                encoded_identity = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
                try:
                    with identity_path.open("xb") as output:
                        output.write(encoded_identity)
                        output.flush()
                        os.fsync(output.fileno())
                except FileExistsError:
                    if identity_path.is_symlink() or identity_path.read_bytes() != encoded_identity:
                        raise ValueError("subframe wire endpoint identity changed for an existing send") from None
            request_path = stem.with_suffix(".request.json")
            if routed_endpoints is not None:
                atomic_bytes(stem.with_suffix(".route.json"), json.dumps(
                    route, sort_keys=True, separators=(",", ":")).encode())
            if identity is not None and request_path.exists():
                if request_path.is_symlink() or request_path.read_bytes() != wire:
                    raise ValueError("subframe wire request changed for an existing send")
            elif routed_endpoints is not None:
                atomic_bytes(request_path, wire)
            else:
                atomic_bytes(request_path, wire)
            # The live countdown is not on the wire; a resume reads it from this private sidecar.
            from kernel_read_fixes import sandbox_time_left
            time_left = sandbox_time_left()
            if time_left is not None:
                atomic_bytes(stem.with_suffix(".time-left.json"), json.dumps(
                    time_left, sort_keys=True, separators=(",", ":")).encode())
            started = time.monotonic()
            replayed = None
            # The abandoned single-worker retry path can overlap sends; keep replay state serial too.
            if replayer is not None and (replayer.active or replayer.pending is not None):
                replayed = replayer.answer(wire, route=route, endpoint_identity=identity)
                if routed_endpoints is not None and replayer.pending_verified is not None:
                    atomic_bytes(run / "resume-pending-verified.json", json.dumps(
                        replayer.pending_verified, sort_keys=True, separators=(",", ":")).encode())
                if replayed is None and replayer.diverged is not None:
                    atomic_bytes(run / "resume-divergence.json", json.dumps(
                        {**replayer.diverged, "send": index, "checkpoint": replayer.strict}, indent=1).encode())
                    if routed_endpoints is not None and replayer.route_mismatch:
                        raise CheckpointUnfaithful("This recording can't be replayed by this version. Start a fresh run. "
                            f"send {index}: {replayer.diverged['reason']}")
                    if identity is not None:
                        from kernel_tool_tape import ToolReplayRefused
                        raise ToolReplayRefused(f"send {index}: {replayer.diverged['reason']}")
                    if replayer.strict:
                        emit("checkpoint_unfaithful", send=index, **{k: v for k, v in replayer.diverged.items()
                                                                     if k != "first_difference"})
                        atomic_bytes(run / "checkpoint-refused.json", json.dumps(
                            {"error": "CheckpointUnfaithful", "send": index,
                             "reason": replayer.diverged["reason"]}, indent=1).encode())
                        raise CheckpointUnfaithful("This recording can't be replayed by this version. Start a fresh run. "
                            f"send {index}: {replayer.diverged['reason']}")
                    if replayer.pending_mismatch:
                        # No answer exists for this send. Retire replay authority before
                        # handing the rebuilt request to the live transport.
                        fields = {k: v for k, v in replayer.diverged.items()
                                  if k != "first_difference"}
                        from kernel_measurement import ACTIVE_REQUEST
                        capture = ACTIVE_REQUEST.get()
                        journal = getattr(capture, "journal", None)
                        if journal is not None:
                            journal.emit("replay_pending_discarded", send=index, **fields)
                        emit("replay_pending_discarded", send=index, **fields)
                        from kernel_tool_tape import end_replay
                        end_replay()
                        _start_resumed_deadline()
                    else:
                        raise ReplayUnfaithful("This recording can't be replayed by this version. Start a fresh run. "
                            f"send {index}: {replayer.diverged['reason']}")
                elif replayed is not None and not replayer.active:
                    emit("replay_done", send=index, replayed=replayer.cursor)
                    # The next request is the first one built live: it shows the resumed time left.
                    _start_resumed_deadline()
                if replayer.strict and replayer.diverged is None and not replayer.active \
                        and replayer.pending is None \
                        and not getattr(replayer, "verified_emitted", False):
                    replayer.verified_emitted = True
                    emit("checkpoint_verified", send=index, replayed=replayer.cursor)
            elif replayer is not None:
                from kernel_tool_tape import end_replay
                end_replay()
                _start_resumed_deadline()
            if (replayer is not None and replayed is None and replayer.diverged is None
                    and not replayer.active and replayer.pending is None):
                from kernel_tool_tape import end_replay
                end_replay()
                _start_resumed_deadline()
            replay_position = replayer.cursor if replayer is not None else None
            replay_total = len(replayer.tape) if replayer is not None else None
        authorized_ids: list[str] = []
        def before_attempt(attempt_index: int) -> None:
            if send_cap is None:
                raise SubframeBudgetExhausted("BUDGET_EXHAUSTED: subframe physical send allowance is exhausted")
            send_id = (f"{os.environ.get('SF_SUBFRAME_INCARNATION', '0')}:{os.getpid()}:"
                       f"{index}:{attempt_index}:"
                       f"{hashlib.sha256(wire).hexdigest()[:20]}")
            if not send_ledger.authorize_with_cap(send_id, send_cap):
                raise SubframeBudgetExhausted("BUDGET_EXHAUSTED: subframe physical send allowance is exhausted")
            authorized_ids.append(send_id)

        def settle_attempts(attempts: list[dict[str, object]], usage: dict[str, object]) -> None:
            for position, send_id in enumerate(authorized_ids[:len(attempts)]):
                send_ledger.settle(send_id, usage if position == len(attempts) - 1 else {})

        def settle_after_error(send: int, attempts: list[dict[str, object]]) -> None:
            if send_ledger is None:
                return
            try:
                settle_attempts(attempts, {})
            except BaseException as settlement_error:  # noqa: BLE001 - preserve the active send error
                error_name = type(settlement_error).__name__
                # Best-effort reporting must not replace the transport or budget exception.
                with suppress(BaseException), wire_lock:
                    atomic_bytes(wire_dir / f"{send:04d}.settle-error.txt",
                                 (error_name + "\n").encode("utf-8"))
                with suppress(BaseException):
                    emit("send_settlement_error", send=send, error=error_name)

        transport_attempts: list[dict[str, object]] = []
        try:
            response = (replayed if replayed is not None else original(
                wire, endpoint, key, record_attempts=transport_attempts.extend,
                **({"before_attempt": before_attempt} if send_ledger is not None else {}),
                **({**kwargs, "route_meta": route} if routed_endpoints is not None else kwargs)))
        except Exception as error:
            settle_after_error(index, transport_attempts)
            detail = getattr(error, "error_detail", None)
            detail = detail if isinstance(detail, str) and detail else None
            error_path = wire_dir / f"{index:04d}.error.txt"
            error_text = type(error).__name__ + (f": {detail}" if detail else "") + "\n"
            if transport_attempts:
                with wire_lock, (run / "sends.jsonl").open("a") as stream:
                    atomic_bytes(error_path, error_text.encode("utf-8"))
                    stream.write(json.dumps(_failed_send_record(
                        index, wire, final_layout_record(wire), stable_record,
                        transport_attempts, error, identity)) + "\n")
            else:
                with wire_lock:
                    atomic_bytes(error_path, error_text.encode("utf-8"))
            emit("send_error", send=index, error=type(error).__name__,
                 **({"detail": detail} if detail else {}))
            raise
        except SubframeBudgetExhausted:
            settle_after_error(index, transport_attempts)
            with wire_lock:
                atomic_bytes(wire_dir / f"{index:04d}.error.txt", b"SubframeBudgetExhausted\n")
            raise
        if hook_wiring is not None:
            from kernel_measurement import ACTIVE_REQUEST
            capture = ACTIVE_REQUEST.get()
            request_id = getattr(capture, "request_id", None)
            hook_wiring.sent(wire, str(request_id) if request_id is not None else f"send:{index}")
        elapsed = time.monotonic() - started
        with wire_lock:
            atomic_bytes(stem.with_suffix(".response.json"), response)
        try:
            parsed = json.loads(response)
        except (ValueError, RecursionError):
            parsed = None
        usage = _usage(parsed)
        if send_ledger is not None and replayed is None:
            settle_attempts(transport_attempts, usage)
        try:
            body = json.loads(wire)
        except (ValueError, RecursionError):
            body = None
        messages = body.get("messages") if isinstance(body, dict) else None
        record = {"send": index, "request_bytes": len(wire),
                  "request_sha256": hashlib.sha256(wire).hexdigest(),
                  "messages": len(messages) if isinstance(messages, list) else None,
                  **final_layout_record(wire),
                  **stable_record,
                  "seconds": round(elapsed, 3), **usage, "actions": _actions(parsed)}
        if identity is not None:
            record["endpoint_identity"] = identity
        if replayed is not None:
            record.update(replayed=True, replay_position=replay_position, replay_total=replay_total)
        if transport_attempts:
            record["transport_attempts"] = transport_attempts
        with wire_lock, (run / "sends.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        emit("send", **record)
        return response

    kernel_bounded_http.request_bytes = recorded


def install_token_count_warning() -> None:
    """Report fallback once per child, including failures before checked_ask."""
    from kernel_request_diet import RequestTokenMeter

    original = RequestTokenMeter.measure
    if getattr(original, "_stateframe_fallback_warning", False):
        return
    warned = {"done": False}
    HOST_STATE["token_warning"] = warned

    def measured(self, body):
        result = original(self, body)
        if result[1] == "calibrated_bytes" and not warned["done"]:
            warned["done"] = True
            reason = self.unavailable_reason()
            # No tokenizer configured (no_url/disabled) is by design: INFO. A configured one that failed: WARN.
            by_design = reason in {"no_url", "disabled"}
            _reason_log.emit_reason("fallback", "tokenizer_unavailable_unconfigured" if by_design else "tokenizer_unavailable",
                                    "engine_child", "token count fell back to calibrated bytes", reason,
                                    level="INFO" if by_design else "WARN")
            emit("token_counting_unavailable", reason=reason)
        return result

    measured._stateframe_fallback_warning = True
    RequestTokenMeter.measure = measured


def install_checked_ask(rr, spec: dict, run: Path, hook_wiring=None, *,
                        live_slot: int | None = None,
                        replayer: Replayer | None = None,
                        managed_slot: bool = False) -> None:
    """The reference runner's provider callback contract, minus its release fences."""
    original_ask = rr.ask_tools
    requests_dir = run / "requests"
    requests_dir.mkdir(exist_ok=True)
    counter = {"n": 0}
    HOST_STATE["requests"] = counter
    engine_spec = {"environment": spec["environment"], "public_row": spec["public_row"],
                   "presentation_arm": "frame"}

    def checked_ask(messages, **kwargs):
        from kernel_pilot import ACTIVE_RUNTIME
        from capability_request_preflight import preflight_presentation_or_degrade
        from kernel_model_policy import journal_effective_policy

        runtime = ACTIVE_RUNTIME.get()
        if runtime is None:
            from kernel_request_kind import RequestKindError
            raise RequestKindError("primary_runtime_missing")
        runtime.declared_environment = dict(spec["environment"])
        runtime.environment_sources = dict(spec.get("environment_sources", {}))
        runtime.served_model = spec["model"]
        journal_effective_policy(runtime)
        tools = kwargs.get("tools") or []
        capture = getattr(runtime, "_active_capture", None)

        def record_preflight(messages, tools, presentation, *, bound_capture=None, bound_runtime=None):
            if presentation.get("degraded") or presentation.get("presentation_degraded"):
                raise RuntimeError("frame-only preflight degraded to history; refusing (product is frame-only)")
            counter["n"] += 1
            index = counter["n"]
            from kernel_measurement import ACTIVE_REQUEST
            capture = bound_capture if bound_capture is not None else ACTIVE_REQUEST.get()
            request_runtime = bound_runtime if bound_runtime is not None else runtime
            turn = getattr(capture, "step", None)
            if turn is None:
                turn = getattr(request_runtime, "current_step", None)
            record = {"request": index, "turn": turn,
                      "request_id": getattr(request_runtime, "current_request_id", None),
                      "messages": messages, "tools": tools,
                      "presentation": {k: v for k, v in presentation.items()
                                       if isinstance(v, (str, int, float, bool)) or v is None}}
            atomic_bytes(requests_dir / f"{index:04d}.json",
                         json.dumps(record, ensure_ascii=False, default=str, indent=1).encode("utf-8"))
            emit("request", request=index, turn=turn, messages=len(messages),
                 roles="".join(m.get("role", "?")[0] for m in messages if isinstance(m, dict)),
                 tools=[t.get("function", {}).get("name") for t in tools if isinstance(t, dict)])

        from kernel_request_kind import classify, RequestKindError
        if capture is None:
            raise RequestKindError("primary_capture_missing")
        if classify(capture, runtime, {"messages": messages, "tools": tools}):
            def bound_gate(record, bound_capture, bound_runtime):
                from capability_request_preflight import PRECOMPOSED_HOST_TEXT_SCHEMA
                from kernel_request_seal import RequestSealError, log_seal_violation
                body = record.body
                receipt = {
                    **dict(getattr(bound_capture, "release_precomposed_host_text", {}) or {}),
                    "schema": PRECOMPOSED_HOST_TEXT_SCHEMA,
                    "request_id": bound_capture.request_id,
                    "body_digest": record.body_digest,
                    "allocation_digest": record.allocation.digest,
                }
                final_messages, presentation = preflight_presentation_or_degrade(
                    body["messages"], engine_spec, bound_runtime, tools=body.get("tools", []),
                    frame_request_allocation=record.allocation, precomposed_host_text=receipt,
                )
                if final_messages != body["messages"]:
                    log_seal_violation("CLI preflight mutated the sealed request")
                    raise RequestSealError("CLI preflight mutated the sealed request")
                record_preflight(final_messages, body.get("tools", []), presentation,
                                 bound_capture=bound_capture, bound_runtime=bound_runtime)

            from kernel_request_kind import require_adapter
            require_adapter(capture, runtime, messages, gate=bound_gate,
                            host_preparer=prepare_request_host_text, adapter="cli")
        else:
            extra = {}
            if "frame_request_allocation" in preflight_parameters:
                extra["frame_request_allocation"] = getattr(capture, "frame_request_allocation", None)
            messages, presentation = preflight_presentation_or_degrade(
                messages, engine_spec, runtime, tools=tools, **extra)
            record_preflight(messages, tools, presentation)
        if replayer is None or (not replayer.active and replayer.pending is None):
            if live_slot is not None:
                os.environ["REACT_SLOT_ID"] = str(live_slot)
            elif managed_slot:
                os.environ.pop("REACT_SLOT_ID", None)
        return original_ask(messages, **kwargs)

    import inspect
    from capability_request_preflight import preflight_presentation_or_degrade as _preflight
    preflight_parameters = set(inspect.signature(_preflight).parameters)
    # The engine asks the wrapper for its input preparation before runtime.prepare qualifies
    # source locations. The reference runner's only input step is the host-capacity notice, which the
    # product never enables, so preparation is the identity.
    def prepare_request_host_text(messages, *, runtime, tools,
                                   frame_request_allocation=None, framefit5_inputs=None):
        from capability_request_preflight import prepare_release_host_text
        if hook_wiring is not None:
            panel = hook_wiring.panel(runtime)
            if panel is not None:
                messages = [*messages, panel]
        return prepare_release_host_text(
            messages, runtime, tools=tools,
            frame_request_allocation=frame_request_allocation,
            framefit5_inputs=framefit5_inputs,
        )

    checked_ask.prepare_request_host_text = prepare_request_host_text
    checked_ask.prepare_request_inputs = lambda messages: messages
    rr.ask_tools = checked_ask


CAPTURE_SKIP_PARTS = {"__pycache__", ".pytest_cache", ".test-tmp"}


def _copy_shared_context_file(source: Path, destination: Path, *, mode: int) -> None:
    """Clone immutable snapshot bytes to a private inode, reflinking blocks when supported."""
    import shutil

    if sys.platform == "linux":
        created = False
        try:
            import fcntl

            source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                source_stat = os.fstat(source_fd)
                destination_fd = os.open(
                    destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                created = True
                try:
                    fcntl.ioctl(destination_fd, 0x40049409, source_fd)
                    os.fchmod(destination_fd, mode)
                    os.utime(destination, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns))
                    return
                finally:
                    os.close(destination_fd)
            finally:
                os.close(source_fd)
        except FileExistsError as error:
            raise RuntimeError("shared subframe context destination appeared during cloning") from error
        except OSError:
            if created:
                destination.unlink(missing_ok=True)
    shutil.copy2(source, destination)
    destination.chmod(mode)


def install_repo_context(rr, parent: Path, *, shared_snapshot: Path | None = None,
                         snapshot_manifest_digest: str | None = None,
                         source_manifest_digest: str | None = None,
                         excluded_paths: tuple[str, ...] = ()) -> None:
    """Put the whole repository into the engine's sandbox, not only the declared files.

    The engine's multi-file branch builds a fresh root from the declared files and public tests.
    Before its public tests are written, copy every other file of the run's read-only parent
    snapshot in, so imports, fixtures and configs the checks need are there. Declared files are
    already written by the engine and are never overwritten here.
    """
    import shutil

    shared_manifest = None
    if shared_snapshot is not None:
        if not isinstance(snapshot_manifest_digest, str) or not isinstance(source_manifest_digest, str):
            raise RuntimeError("shared subframe context lacks authenticated manifest identities")
        try:
            try:
                from .parallel_worker import read_snapshot_manifest
            except ImportError:
                from stateframe.parallel_worker import read_snapshot_manifest
            shared_manifest = read_snapshot_manifest(
                shared_snapshot, snapshot_manifest_digest, verify_contents=False,
                verify_inventory=False)
        except (OSError, ImportError, RuntimeError, ValueError) as error:
            raise RuntimeError(f"shared subframe context snapshot is invalid: {error}") from error
        if shared_manifest.source_digest != source_manifest_digest:
            raise RuntimeError("shared subframe context source digest differs from the unit start")
        for relative in shared_manifest.directories:
            directory = Path(shared_snapshot) / relative
            if (directory.is_symlink() or not directory.is_dir()
                    or stat.S_IMODE(directory.stat().st_mode) != 0o555):
                raise RuntimeError(f"shared subframe context directory is unsafe: {relative}")
        if any(not isinstance(path, str) or not path or path.startswith("/") or "\\" in path
               or any(part in {"", ".", ".."} for part in path.split("/"))
               for path in excluded_paths):
            raise RuntimeError("shared subframe context exclusions contain an unsafe path")
    excluded = frozenset(excluded_paths)

    original = rr._materialize_public_tests
    done: dict[str, Path] = {}
    HOST_STATE["repo_context"] = done

    def materialize(root, fileset, row):
        raw_root = Path(root)
        if shared_manifest is not None and raw_root.is_symlink():
            raise RuntimeError("shared subframe context sandbox root is a symlink")
        root = raw_root.resolve()
        if done and done["root"] != root:
            raise RuntimeError("engine switched sandbox roots mid-run")
        if not done:
            if shared_manifest is not None:
                shared_root = Path(shared_snapshot)
                for relative, original_mode in shared_manifest.files.items():
                    rel = Path(relative)
                    if relative in excluded or CAPTURE_SKIP_PARTS & set(rel.parts):
                        continue
                    source = shared_root / rel
                    destination = root / rel
                    try:
                        source_mode = source.lstat().st_mode
                    except OSError as error:
                        raise RuntimeError(
                            f"shared subframe context file is missing: {relative}") from error
                    if not stat.S_ISREG(source_mode) or stat.S_IMODE(source_mode) != 0o444:
                        raise RuntimeError(f"shared subframe context file is unsafe: {relative}")
                    destination_parent = root
                    for part in rel.parts[:-1]:
                        destination_parent = destination_parent / part
                        if destination_parent.is_symlink():
                            raise RuntimeError(
                                "shared subframe context destination contains a symlink")
                        if destination_parent.exists():
                            if not destination_parent.is_dir():
                                raise RuntimeError(
                                    "shared subframe context destination parent is not a directory")
                        else:
                            destination_parent.mkdir()
                    if destination.is_symlink():
                        raise RuntimeError(
                            "shared subframe context destination is a symlink")
                    if destination.exists():
                        continue
                    try:
                        destination.parent.resolve().relative_to(root)
                    except ValueError as error:
                        raise RuntimeError(
                            "shared subframe context destination escaped its sandbox") from error
                    ancestor = destination.parent
                    while ancestor != root:
                        if ancestor.is_symlink():
                            raise RuntimeError(
                                "shared subframe context destination contains a symlink")
                        ancestor = ancestor.parent
                    if original_mode & 0o111:
                        # Executable helpers retain their source mode in the private sandbox.
                        shutil.copy2(source, destination)
                        destination.chmod(original_mode | stat.S_IWUSR)
                        continue
                    # Clone immutable snapshot bytes without exposing a mutable hardlink.
                    # Match the old private-context mode derived from the read-only snapshot.
                    _copy_shared_context_file(
                        source, destination,
                        mode=stat.S_IMODE(source_mode) | stat.S_IWUSR)
                done["shared_snapshot"] = shared_root
                done["shared_snapshot_digest"] = shared_manifest.digest
            else:
                for source in sorted(parent.rglob("*")):
                    rel = source.relative_to(parent)
                    if source.is_symlink() or not source.is_file() or CAPTURE_SKIP_PARTS & set(rel.parts):
                        continue
                    destination = root / rel
                    if destination.exists():
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                    # The engine rewrites public tests in its own sandbox copy; a read-only source file (A10
                    # fixtures ship protected tests as 0444) must not make that copy unwritable. Sandbox only.
                    destination.chmod(destination.stat().st_mode | stat.S_IWUSR)
            done["root"] = root
        return original(root, fileset, row)

    rr._materialize_public_tests = materialize


def install_compact_capture(parent: Path) -> None:
    """Capture only declared or changed files at each proposal boundary.

    The engine's capture stores every file of the sandbox, base64, per proposal. With the whole
    repository in the sandbox that is the repo size times the number of steps. The product only
    needs what differs from the parent (and the declared files), which is what it applies back.
    """
    import base64
    import difflib
    import kernel_bridge_evidence as evidence

    def parent_bytes(rel: str) -> bytes | None:
        path = parent / rel
        return path.read_bytes() if path.is_file() and not path.is_symlink() else None

    def capture_candidate(destination, root, row, proposals):
        if not destination:
            return
        folder = Path(destination).resolve()
        root = Path(root)
        if folder.is_relative_to(root.resolve()):
            raise ValueError("candidate evidence must remain outside the worker root")
        folder.mkdir(parents=True, exist_ok=True)
        declared = set((row.get("files") or {}).keys())
        files: dict[str, bytes] = {}
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root)
            if CAPTURE_SKIP_PARTS & set(rel.parts) or path.is_symlink() or not path.is_file():
                continue
            name = rel.as_posix()
            data = path.read_bytes()
            if name in declared or parent_bytes(name) != data:
                files[name] = data
        deleted = sorted(name for name in declared if parent_bytes(name) is not None
                         and not (root / name).exists())
        target = str(row.get("path") or "")
        candidate = files.get(target, b"").decode("utf-8", errors="replace")
        before = (parent_bytes(target) or b"").decode("utf-8", errors="replace")
        record = {"proposals": proposals, "target": target, "capture": "stateframe-compact/1",
                  "files_base64": {n: base64.b64encode(d).decode("ascii") for n, d in files.items()},
                  "deleted_files": deleted,
                  "file_hashes": {n: hashlib.sha256(d).hexdigest() for n, d in files.items()},
                  "patch": "".join(difflib.unified_diff(
                      before.splitlines(keepends=True), candidate.splitlines(keepends=True),
                      fromfile=target + ".parent", tofile=target))}
        out = folder / f"{proposals:04d}.json"
        encoded = json.dumps(record, sort_keys=True, indent=1) + "\n"
        if out.exists():
            if out.read_text() != encoded:
                raise ValueError("candidate boundary was captured twice with different contents")
            return
        with out.open("x") as handle:
            handle.write(encoded)

    evidence.capture_candidate = capture_candidate


def install_test_log(rr) -> None:
    original = rr.run_tests

    def logged(*args, **kwargs):
        observation = original(*args, **kwargs)
        first = str(observation).strip().splitlines()[:1]
        emit("tests", summary=(first[0][:160] if first else ""))
        return observation

    rr.run_tests = logged


def engine_step_limit(max_turns: object) -> int | None:
    """The engine's MAX_STEPS for a product turn limit: 0 means no step limit.

    The unbounded engine represents "no cap" as None (kernel_step_limits.parse_step_limit); a literal 0 there is
    zero fuel, i.e. a run that takes no turns. Use the engine's own parser when it has one; the runner only
    sends 0 to an engine that declares STEPS_UNBOUNDED.
    """
    turns = int(max_turns)
    if turns != 0:
        return turns
    try:
        from kernel_step_limits import parse_step_limit
    except ImportError:
        return None
    return parse_step_limit("unbounded")


def install_run_identity(seed: object) -> None:
    """Make the engine's run-scoped host ids (uuid4: run ids, journal handles) a stream fixed by the run's seed.

    Ids stay unique per run (the runner draws a random seed for each run), but the SAME run started again
    from the same seed gets the same host ids in the same order. Provider-visible ids do not use this
    stream: they derive from the task identity, step and content (kernel_wire_identity.wire_run_seed).
    """
    import uuid

    if not isinstance(seed, str) or len(seed) != 32 or any(c not in "0123456789abcdef" for c in seed):
        return  # a run made before seeds existed keeps fresh random ids
    import threading

    key = bytes.fromhex(seed)
    state = {"n": 0}
    HOST_STATE["run_identity"] = state
    lock = threading.Lock()

    def uuid4() -> uuid.UUID:
        with lock:
            state["n"] += 1
            number = state["n"]
        return uuid.UUID(bytes=hashlib.sha256(key + number.to_bytes(8, "big")).digest()[:16], version=4)

    uuid.uuid4 = uuid4


def install_check_recorder(run: Path, replayer: Replayer | None, recorded_dir: Path | None) -> None:
    """Record every sandboxed check's output; on resume, give a replayed check its recorded output.

    A re-executed check prints different wall-clock durations ("1 failed in 0.03s" vs "0.04s"), and the engine
    hashes the raw output into its typed state, so the rebuilt request would differ from the recorded one for no
    real reason. While the replay is still faithful, a re-executed check whose output equals the recorded one apart
    from durations and run ids is handed the RECORDED output: the same result, with the timings the original run
    really measured. Any other difference keeps the new output, so the replay notices it and continues live.
    """
    import dataclasses
    import zlib

    # Fail open: this only serves a later resume, so it must never break the run it records.
    try:
        import kernel_execution_policy as policy

        folder = run / "checks"
        folder.mkdir(exist_ok=True)
        original = policy.run_sandboxed
    except Exception as error:  # noqa: BLE001 - recorder off, run unaffected
        emit("check_recorder_off", reason=f"{type(error).__name__}: {error}"[:200])
        return
    counter = {"n": 0, "on": True}
    # This belongs to the new attempt's replay controller, not the saved run state. Restoring
    # the old value (false on the original run) would lose recorded check timings in the tail.
    substitution = {"enabled": recorded_dir is not None}
    HOST_STATE["checks"] = counter

    def masked(text: str) -> str:
        return mask_volatile(text) if isinstance(text, str) else text

    def recorded_check(index: int) -> dict | None:
        path = None if recorded_dir is None else recorded_dir / f"{index:04d}.json.z"
        if path is None or not path.is_file():
            return None
        try:
            return json.loads(zlib.decompress(path.read_bytes()))
        except (OSError, ValueError, zlib.error):
            return None

    def run_sandboxed(command, root, **kwargs):
        result = original(command, root, **kwargs)  # a real check error propagates exactly as before
        if not counter["on"]:
            return result
        try:
            return record_and_match(command, result)
        except Exception as error:  # noqa: BLE001 - disk full, unexpected result shape: pass the result through
            counter["on"] = False
            emit("check_recorder_off", reason=f"{type(error).__name__}: {error}"[:200])
            return result

    def record_and_match(command, result):
        counter["n"] += 1
        index = counter["n"]
        completed = result.completed
        record = {"check": index, "argv": [str(part) for part in command],
                  "returncode": completed.returncode,
                  "stdout": completed.stdout if isinstance(completed.stdout, str) else None,
                  "stderr": completed.stderr if isinstance(completed.stderr, str) else None}
        if substitution["enabled"] and replayer is not None and replayer.active:
            previous = recorded_check(index)
            if (previous is not None and previous.get("argv") == record["argv"]
                    and previous.get("returncode") == record["returncode"]
                    and masked(previous.get("stdout")) == masked(record["stdout"])
                    and masked(previous.get("stderr")) == masked(record["stderr"])):
                if (previous.get("stdout"), previous.get("stderr")) != (record["stdout"], record["stderr"]):
                    completed = subprocess.CompletedProcess(completed.args, completed.returncode,
                                                            previous.get("stdout"), previous.get("stderr"))
                    result = dataclasses.replace(result, completed=completed)
                    record.update(stdout=previous.get("stdout"), stderr=previous.get("stderr"),
                                  recorded_timings=True)
            else:
                substitution["enabled"] = False  # the replay's state differs from here on; never patch again
        atomic_bytes(folder / f"{index:04d}.json.z",
                     zlib.compress(json.dumps(record).encode("utf-8")))
        return result

    policy.run_sandboxed = run_sandboxed


ENGINE_LOCK = ".engine.lock"


def hold_engine_lock(run: Path):
    """One engine per run directory, held for this process's lifetime (released by the OS on any exit)."""
    import fcntl

    handle = (run / ENGINE_LOCK).open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError("another engine is already running on this run directory") from None
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()) + "\n")
    handle.flush()
    return handle


def watch_parent() -> None:
    """Stop when the runner that started us is gone (killed with SIGKILL, laptop crash recovery).

    The engine runs in its own session so the runner can stop its whole process group; an orphaned
    engine would otherwise keep calling the model and editing its sandbox with nobody recording the run.
    """
    import signal
    import threading

    parent = os.getppid()

    def loop() -> None:
        while True:
            time.sleep(1.0)
            if os.getppid() != parent:
                try:
                    # Only our own session's group (the runner starts us with start_new_session);
                    # never a group we merely belong to, such as a test runner's.
                    if os.getpgrp() == os.getpid():
                        _reason_log.emit_kill(
                            "engine_child", target="engine process group", pid=os.getpid(),
                            why=f"parent runner pid {parent} is gone",
                            signal_name="SIGKILL", level="ERROR", cause_key="parent_gone")
                        os.killpg(os.getpgrp(), signal.SIGKILL)
                finally:
                    os._exit(137)

    threading.Thread(target=loop, name="stateframe-parent-watch", daemon=True).start()


CHECKPOINT_KEY = Path(".stateframe") / "checkpoint.key"  # under the user's home
KEY_LINK_SETTLE_S = 2.0  # how long a reader waits out publish()'s temporary second link


def checkpoint_key() -> bytes | None:
    """This install's checkpoint signing key: 32 random bytes in a 0600 file of a 0700 directory
    owned by this user, outside every run directory. Created on first use; None if unusable
    (then no checkpoint is written, and resume replays). The directory and the key are opened without
    following symlinks and checked by fstat on the open descriptors, so an existing directory that is
    a symlink or another user's is refused, not trusted. Owned loose modes are repaired."""
    import secrets
    import stat as stat_module

    home = safe_home()
    if home is None:
        emit("checkpoint_key_unusable", reason="home directory unknown")
        return None
    path = home / CHECKPOINT_KEY

    def publish(replace: bool, directory_fd: int) -> None:
        # Complete key in a private temp file first, then link/replace it into place: a crash can never
        # leave a short key behind, and a concurrent reader never sees a partial one.
        temp = ".checkpoint.key." + secrets.token_hex(16)
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow,
                     0o600, dir_fd=directory_fd)
        try:
            with os.fdopen(fd, "wb") as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(secrets.token_bytes(32))
                stream.flush()
                os.fsync(stream.fileno())
            if replace:
                os.replace(temp, CHECKPOINT_KEY.name,
                           src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
            else:
                try:
                    os.link(temp, CHECKPOINT_KEY.name,
                            src_dir_fd=directory_fd, dst_dir_fd=directory_fd,
                            follow_symlinks=False)
                except FileExistsError:
                    pass  # another process won the race: use its key
        finally:
            try:
                os.unlink(temp, dir_fd=directory_fd)
            except FileNotFoundError:
                pass

    def settled_fstat(fd: int) -> os.stat_result:
        # publish() links its temp file to the key name and unlinks the temp right after, so a
        # concurrent reader can briefly see two links. Wait that window out; a key that stays
        # hard-linked past it is still refused by the caller.
        info = os.fstat(fd)
        waited = 0.0
        while info.st_nlink > 1 and waited < KEY_LINK_SETTLE_S:
            time.sleep(0.02)
            waited += 0.02
            info = os.fstat(fd)
        return info

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_fd = key_fd = None
    try:
        private_directory(path.parent)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow)
        directory_info = os.fstat(directory_fd)
        if (not stat_module.S_ISDIR(directory_info.st_mode)
                or directory_info.st_uid != os.getuid()
                or stat_module.S_IMODE(directory_info.st_mode) != 0o700):
            emit("checkpoint_key_unusable", reason="key directory is not a private directory of this user")
            return None
        try:
            key_fd = os.open(CHECKPOINT_KEY.name, os.O_RDONLY | nofollow, dir_fd=directory_fd)
        except FileNotFoundError:
            publish(replace=False, directory_fd=directory_fd)
            key_fd = os.open(CHECKPOINT_KEY.name, os.O_RDONLY | nofollow, dir_fd=directory_fd)
        info = settled_fstat(key_fd)
        if not stat_module.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
            emit("checkpoint_key_unusable", reason="key file is not a private file of this user")
            return None
        if stat_module.S_IMODE(info.st_mode) != 0o600:
            os.fchmod(key_fd, 0o600)
        if info.st_size < 32:
            # No checkpoint can be signed with a short key, so nothing is lost.
            os.close(key_fd)
            key_fd = None
            print("stateframe: checkpoint key was incomplete (an earlier crash?); replaced it", file=sys.stderr)
            publish(replace=True, directory_fd=directory_fd)
            key_fd = os.open(CHECKPOINT_KEY.name, os.O_RDONLY | nofollow, dir_fd=directory_fd)
            info = settled_fstat(key_fd)
            if not stat_module.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
                emit("checkpoint_key_unusable", reason="key file is not a private file of this user")
                return None
        chunks = []
        while True:
            chunk = os.read(key_fd, 4096)
            if not chunk:
                break
            chunks.append(chunk)
        key = b"".join(chunks)
    except OSError as error:
        emit("checkpoint_key_unusable", reason=type(error).__name__)
        return None
    finally:
        for fd in (key_fd, directory_fd):
            if fd is not None:
                os.close(fd)
    if len(key) < 32:
        emit("checkpoint_key_unusable", reason="key file is incomplete")
        return None
    return key


def register_host_state(*, required: bool, secret: str = "") -> None:
    """Hand the wrappers' state to the engine's checkpoint (an older engine has none: skip)."""
    try:
        import kernel_checkpoint
    except ImportError:
        if required:
            raise
        return
    saving = os.environ.get("STATEFRAME_CHECKPOINT", "on").strip().lower() not in {"off", "0"}
    if saving and hasattr(kernel_checkpoint, "register_signing_key"):
        key = checkpoint_key()
        if key is not None:
            kernel_checkpoint.register_signing_key(key)
    for name, state in HOST_STATE.items():
        kernel_checkpoint.register_participant(name, state)
    if secret and hasattr(kernel_checkpoint, "register_secret"):
        kernel_checkpoint.register_secret(secret)


def _error_fields(error: BaseException) -> dict[str, str]:
    """error_type/error_message for an engine terminal receipt. Empty when the exception has no text.

    The message is whitespace-normalized and capped at 600 characters, matching the engine-result envelope.
    outcomes.engine_error_reason reads these to name the cause.
    """
    message = " ".join(str(error).split())[:600]
    if not message:
        return {}
    return {"error_type": type(error).__name__, "error_message": message}


def _engine_error_terminal(kind: str, error: BaseException) -> dict[str, object]:
    """Terminal receipt for an ENGINE_ERROR crash: the exception class plus its message when it has one."""
    return {"result": kind, "error": type(error).__name__, **_error_fields(error)}


def _finish_mutation_runtime(mutation_runtime, result: dict, status: int, emit_event):
    """Finish an enabled mutation runtime without turning host refusal into run failure."""
    try:
        try:
            mutation_runtime.invoke_hook(event="session_end", facts={"result": result.get("result")})
        finally:
            mutation_runtime.close()
    except BaseException as error:
        code = getattr(type(error), "code", None)
        if code == "MUTATION_PLATFORM_UNQUALIFIED":
            emit_event("mutation_refused", code=code, detail=str(error)[:300])
        else:
            emit_event("engine_error", error=type(error).__name__, detail=str(error)[:300])
            result, status = {"result": "MUTATION_CONTAINMENT", "error": str(error)[:300],
                              **_error_fields(error)}, 1
    return result, status


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: _child.py <run-dir>", file=sys.stderr)
        return 2
    startup_home_identity = _runner_home_identity_pin()
    run = Path(argv[1]).resolve(strict=True)
    watch_parent()
    lock = hold_engine_lock(run)  # noqa: F841 - held until exit
    spec = json.loads((run / "spec.json").read_text())
    emit_run_env_events(run)
    resume = spec.get("resume") if isinstance(spec.get("resume"), dict) else None
    checkpoint = _selected_resume_checkpoint(resume)
    recorded_mode = str(spec.get("environment", {}).get("STATEFRAME_CHECKPOINT", "")).strip().lower()
    if (os.environ.get("STATEFRAME_CHECKPOINT", "").strip().lower() in {"off", "0"}
            or recorded_mode in {"off", "0"}):
        checkpoint = None
    if checkpoint is not None:
        # Checkpoint resume: the engine restores its state as of send ``sends``; only the recorded
        # exchanges after it are replayed, strictly (any difference refuses before sending).
        tail = [item for item in resume.get("tape") or [] if int(item["send"]) > int(checkpoint["sends"])]
        replayer = Replayer(tail, resume.get("model_swap"), strict=True, pending=checkpoint.get("pending"),
                            route_changes=resume.get("route_changes"))
    else:
        replayer = (Replayer(resume.get("tape") or [], resume.get("model_swap"),
                             pending=resume.get("pending"), route_changes=resume.get("route_changes"))
                    if resume is not None else None)
    install_environment(spec)
    # Keep all engine urlopen calls direct, including those outside the primary transport.
    _install_direct_opener()
    _install_primary_api_key_env(spec)
    if checkpoint is not None:
        os.environ[CHECKPOINT_RESTORE_ENV] = str(checkpoint["dir"])
    install_run_identity(spec.get("run_identity_seed"))  # before any engine import
    seconds_left = resume.get("seconds_left") if resume is not None else None
    if isinstance(seconds_left, (int, float)) and not isinstance(seconds_left, bool):
        _RESUMED_TIME["seconds"] = float(seconds_left)
        if replayer is None or (not replayer.active and replayer.pending is None):
            _start_resumed_deadline()  # nothing to replay: the run is live from its first request
    key = os.environ.get(spec["api_key_env"], "") if spec.get("api_key_env") else ""
    try:
        skill_records = spec.get("skills", [])
        skill_digest = spec.get("skills_digest", skills_digest([]))
        if "skills" in spec:
            skills_file = Path(spec["environment"]["STATEFRAME_SKILL_SETTINGS_FILE"])
            skill_settings = json.loads(skills_file.read_text(encoding="utf-8"))
            if (skill_settings.get("schema") != "stateframe-skill-settings/1"
                    or skill_settings.get("skills") != skill_records
                    or skill_settings.get("skills_digest") != skill_digest
                    or skill_settings.get("measurements") != spec.get("skills_measurements")):
                raise SkillError("host skill settings file differs from the run spec")
        verify_records(skill_records, skill_digest, spec.get("skills_measurements"))
    except (SkillError, OSError, KeyError, ValueError) as error:
        _reason_log.emit_reason("invariant", "spec_skills_unusable", "engine_child",
                                "engine refused to start: skill records unusable",
                                f"{type(error).__name__}: {str(error)[:300]}", level="ERROR")
        emit("engine_error", error=type(error).__name__, detail=str(error)[:300])
        return 1
    engine = Path(spec["engine_dir"]).resolve(strict=True)
    # Repo intel asks tree_sitter_language_pack for parsers; serve them from the per-language grammar packages.
    try:
        try:
            from .ts_grammars import install_as_language_pack
        except ImportError:
            from ts_grammars import install_as_language_pack
        install_as_language_pack()
    except Exception:  # noqa: BLE001 - repo map AST is optional; the engine falls back to its text scan
        pass
    sys.path.insert(0, str(engine))
    try:
        import kernel_checkpoint
    except ImportError:
        pass  # An older engine may not ship checkpoint support.
    else:
        kernel_checkpoint.configure_product_host()
        kernel_checkpoint.configure_runner_home_identity_pin(startup_home_identity)
        try:
            # Register the installation key before any model/tool activity, including no-key runs.
            kernel_checkpoint.verify_stateframe_home_identity(run)
        except kernel_checkpoint.CheckpointError:
            emit("engine_error", error="CheckpointError", detail="STATEFRAME_HOME_IDENTITY_CHANGED")
            return 1
    unit_spec = None
    send_ledger = None
    slot_lease = None
    live_slot = None
    send_cap = None
    if "unit_spec" in spec:
        from kernel_parallel_records import UnitSpec
        from kernel_unit_policy import UnitPolicy, install as install_unit_policy
        from kernel_check_permits import install as install_check_permits
        try:
            from .parallel_resources import (SendLedger, SlotHandoffDurabilityError, SlotPoolError,
                                             idle_slots, lease_slot, parse_slot_pool, slot_candidates)
        except ImportError:
            from stateframe.parallel_resources import (SendLedger, SlotHandoffDurabilityError, SlotPoolError,
                                                       idle_slots, lease_slot, parse_slot_pool, slot_candidates)
        unit_spec = UnitSpec.from_json(spec["unit_spec"])
        if Path(str(spec.get("parallel_engine_python", ""))).resolve() != Path(sys.executable).resolve():
            raise RuntimeError("subframe interpreter differs from the recorded parent engine interpreter")
        permit_root = Path(str(spec["parallel_permit_root"]))
        install_check_permits(permit_root, unit_spec.limits.check_permits)
        install_unit_policy(UnitPolicy(unit_spec))
        send_ledger = SendLedger(Path(str(spec["parallel_send_ledger"])))
        send_cap = unit_spec.limits.unit_max_requests
        endpoint = str(spec["parallel_parent_endpoint"])
        available = idle_slots(endpoint)
        try:
            pool = parse_slot_pool(spec.get("parallel_slots"), source="--parallel-slots in worker spec")
        except SlotPoolError as error:
            raise RuntimeError(f"invalid subframe slot pool: {error}") from error
        prompt_cache_slot = spec.get("prompt_cache_slot")
        parent_slot = (prompt_cache_slot.get("slot") if isinstance(prompt_cache_slot, dict)
                       else None)
        candidates = slot_candidates(pool, available, parent_slot,
                                     fallback_capacity=unit_spec.limits.max_concurrency)
        slot_lease = lease_slot(endpoint, candidates)
        if slot_lease is None:
            raise RuntimeError("RESOURCES_UNAVAILABLE: no subframe model slot lease is available")
        live_slot = configure_subframe_slot(run, slot_lease.slot,
                                            # A configured pool coordinates local concurrency even
                                            # when the provider has no proven native slot API. Only
                                            # /slots discovery authorizes provider slot fields.
                                            pinned=available is not None,
                                            replayer=replayer)
    elif spec.get("parallel_mode", "off") != "off":
        from kernel_check_permits import install as install_check_permits
        limits = spec.get("parallel_limits")
        if not isinstance(limits, dict) or type(limits.get("check_permits")) is not int:
            raise RuntimeError("parent subframe limits are missing or invalid")
        install_check_permits(run / "parallel" / "check-permits", limits["check_permits"])
    routed_endpoints = None
    if spec.get("routes") is not None:
        import kernel_frame_routes
        base = spec["chat_url"].removesuffix("/chat/completions")
        environment = spec.get("environment", {})
        baseline = {"endpoint": base, "model": spec["model"],
                    "api_key_env": spec.get("api_key_env"),
                    "temperature": float(environment.get("REACT_TEMP", 1.0)),
                    "max_tokens": int(environment.get("REACT_MAX_OUTPUT_TOKENS", 24000)),
                    "timeout_s": float(environment.get("REACT_REQUEST_TIMEOUT_S", 1800)),
                    "slot": (spec.get("prompt_cache_slot") or {}).get("slot"),
                    "context_tokens": int(environment.get("STATEFRAME_CONTEXT_TOKENS", 65536))}
        kernel_frame_routes.install(spec["routes"], baseline=baseline)
        # Same rule as stateframe.routes.uses_other_model (this module cannot import the package):
        # decide2's built-in "decision-tier" alias is served by the default model.
        if os.environ.get("LLAMA_CPP_TOKENIZE_URL") and any(
                settings.get("model") not in {spec["model"], "decision-tier" if role == "decide2" else spec["model"]}
                and not settings.get("tokenizer")
                for role, settings in spec["routes"]["roles"].items()):
            raise ValueError("LLAMA_CPP_TOKENIZE_URL conflicts with routed models; set each role tokenizer")
        routed_endpoints = {spec["chat_url"]}
        routed_endpoints.update(
            str(settings["endpoint"]).rstrip("/") + "/chat/completions"
            for settings in spec["routes"]["roles"].values())
        HOST_STATE["frame_routes_last_action"] = kernel_frame_routes.LAST_CLOSED_ACTION
        HOST_STATE["frame_routes_accounting"] = kernel_frame_routes.ACCOUNTING
    from kernel_provider_dialect import seed_run_capability_cache
    seed_run_capability_cache(spec.get("provider_capabilities"))
    from kernel_custom_tools import CustomToolError, load_tool_files
    from kernel_mcp_tools import load_mcp_files
    from kernel_tool_settings import ToolSettingsError, load_tool_settings
    from kernel_tool_tape import ToolTapeError, configure_replay
    hook_wiring = None
    try:
        settings = load_tool_settings()
        if settings is not None:
            if settings.as_dict() != spec.get("tool_settings"):
                raise ToolSettingsError("host tool settings file differs from the run spec")
            builtin_names = frozenset(("grep", "list_files", "read", "edit", "str_replace", "insert_before",
                "authored_check", "file_create", "run_tests", "read_check_receipt", "recall", "done",
                "shell", "job_wait", "job_output", "job_stop", "file_delete", "file_move", "git", "web_fetch", "git_commit", "git_branch", "skill_load",
                *(("mutate",) if "mutations" in spec else ())))
            custom = load_tool_files([Path(item["path"]) for item in settings.tools_files],
                                     builtin_names=builtin_names, shell_network=settings.shell_network)
            if custom.digest != spec.get("custom_tools_digest"):
                raise CustomToolError("custom tool definition or executable changed since host validation")
            mcp_servers = load_mcp_files([Path(item["path"]) for item in settings.mcp_files],
                                         shell_network=settings.shell_network)
        if "hooks" in spec:
            from kernel_hooks import HookSet, verify_bundle
            from kernel_hook_wiring import HookWiring
            from kernel_tool_tape import replay_mode_active

            if type(spec["hooks"]) is not dict:
                raise ValueError("run hook spec must be an object")
            hook_spec = dict(spec["hooks"])
            hook_spec.pop("start_context", None)
            hooks = HookSet.from_spec(hook_spec)
            verify_bundle(hook_spec["bundle_manifest"], run / "hook-bundle")
            hook_wiring = HookWiring(run, hooks, spec, replay=replay_mode_active)
            HOST_STATE["hooks"] = hook_wiring.state
        configure_replay(run, active=bool(replayer and replayer.tape))
        if unit_spec is None and spec.get("parallel_mode", "off") != "off":
            from kernel_parallel import install as install_parallel
            install_parallel(run, spec)
            if spec.get("environment", {}).get("STATEFRAME_PARALLEL_SUGGEST") == "on":
                from kernel_parallel_suggest import install_host_context
                install_host_context(spec)
    except (ToolSettingsError, CustomToolError, ToolTapeError, SkillError, OSError, KeyError, ValueError) as error:
        _reason_log.emit_reason("invariant", "engine_setup_failed", "engine_child",
                                "engine refused to start: tool or skill setup failed",
                                f"{type(error).__name__}: {str(error)[:300]}", level="ERROR")
        emit("engine_error", error=type(error).__name__, detail=str(error)[:300])
        return 1
    sys.argv[:] = [sys.argv[0]]  # engine modules read argv as legacy CLI input; give them none
    output = run / "worker"
    for name in ("", "results", "scratch"):
        (output / name).mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(output / "scratch")
    from kernel_exec_session import ExecSession
    from kernel_shell_tool import configure as configure_shell
    session = ExecSession(run)
    import signal
    previous_term = signal.getsignal(signal.SIGTERM)

    def terminate_shell_run(signum, frame):
        raise KeyboardInterrupt("engine received SIGTERM")

    signal.signal(signal.SIGTERM, terminate_shell_run)
    started = time.monotonic()
    result: object = None
    mutation_runtime = None
    try:
        if "mutations" in spec:
            if __package__:
                from . import mutate_host
            else:
                import mutate_host
            mutation_runtime = mutate_host.start(spec, run)
        if settings is not None:
            configure_shell(run, settings, spec, session)
            from kernel_shell_tool import service as shell_service
            from kernel_agent_tools import configure as configure_agent_tools
            configure_agent_tools(shell_service(), custom, mcp_servers,
                                  resume_active=resume is not None)
        from kernel_skills import configure as configure_skills
        configure_skills(skill_records, skill_digest, spec.get("chat_url"), auth_key=key or None)
        from kernel_project_instructions import configure as configure_project_instructions
        configure_project_instructions(spec.get("project_instructions"))
        from probes import react_repair as rr
        if hook_wiring is not None:
            rr._hook_wiring = hook_wiring
        from kernel_protocol_spec import publish_public_spec

        publish_public_spec(output / "protocol-spec.json")
        rr.POC = output
        rr.OUT = output / "rows.jsonl"
        rr.MAX_STEPS = engine_step_limit(spec["max_turns"])
        rr.API = spec["chat_url"]
        rr.MODEL = spec["model"]
        rr.KEY = key
        rr.REPOS["build"] = Path(spec["parent_snapshot"])
        if unit_spec is None:
            install_repo_context(rr, Path(spec["parent_snapshot"]))
        else:
            shared_snapshot = run.parents[1] / "snapshots" / f"g{unit_spec.start_generation}"
            public_tests = spec["public_row"].get("public_tests", {})
            public_test_paths = tuple(public_tests) if isinstance(public_tests, dict) else ()
            excluded_paths = tuple(sorted(set(unit_spec.contract.owns)
                                          | set(unit_spec.contract.creates)
                                          | set(unit_spec.contract.check.source_paths)
                                          | set(public_test_paths)))
            install_repo_context(
                rr, Path(spec["parent_snapshot"]), shared_snapshot=shared_snapshot,
                snapshot_manifest_digest=str(spec["parallel_snapshot_manifest_digest"]),
                source_manifest_digest=unit_spec.start_manifest_digest,
                excluded_paths=excluded_paths,
            )
        install_compact_capture(Path(spec["parent_snapshot"]))
        install_wire_recorder(run, spec["chat_url"], replayer, routed_endpoints, hook_wiring,
                              send_ledger=send_ledger, send_cap=send_cap)
        previous_checks = Path(resume["previous_attempt"]) / "checks" if resume is not None else None
        install_check_recorder(run, replayer, previous_checks)
        install_token_count_warning()
        install_checked_ask(rr, spec, run, hook_wiring,
                            live_slot=live_slot, replayer=replayer,
                            managed_slot=unit_spec is not None)
        install_test_log(rr)
        register_host_state(required=checkpoint is not None, secret=key)
        if routed_endpoints is not None:
            import kernel_checkpoint
            for settings in spec["routes"]["roles"].values():
                name = settings.get("api_key_env")
                if isinstance(name, str) and os.environ.get(name):
                    kernel_checkpoint.register_secret(os.environ[name])
        row = json.loads(json.dumps(spec["public_row"]))
        row.update(_skip_db=True, _bridge_ordinary_brief=row["msg"],
                   _bridge_capture=str(output / "candidates"),
                   information_policy={"information_policy_version": "information-policy/1",
                                       "initial_oracle_symptom": "OFF",
                                       "live_grader_feedback": "FINAL_ONLY",
                                       "target_disclosure": "PUBLIC_PATHS_UNLOCALIZED"})
        emit_resume_pending_check(spec, resume)
        provider_records = spec.get("provider_capabilities")
        if isinstance(provider_records, dict) and provider_records:
            emit("provider_dialects", endpoints=[
                {"origin": origin, **record}
                for origin, record in sorted(provider_records.items())
                if isinstance(origin, str) and isinstance(record, dict)
            ])
        emit("engine_start", turns=rr.MAX_STEPS,
             **({"resume_attempt": resume.get("attempt"), "replay_sends": len(replayer.tape)}
                if resume is not None and replayer is not None else {}))
        result = rr.work(row)
        status = 0
    except BaseException as error:  # noqa: BLE001 - report every engine failure as a typed record
        atomic_bytes(run / "child-traceback.txt", traceback.format_exc().encode("utf-8"))
        _reason_log.emit_reason("invariant", "engine_crashed", "engine_child",
                                "engine stopped with an error",
                                f"{type(error).__name__}: {' '.join(str(error).split())[:300]}", level="ERROR")
        emit("engine_error", error=type(error).__name__, detail=" ".join(str(error).split())[:300])
        recovery_type = None
        if unit_spec is None and spec.get("parallel_mode", "off") != "off":
            from kernel_parallel import RecoveryRequired as recovery_type
        recovery_required = recovery_type is not None and isinstance(error, recovery_type)
        if hook_wiring is not None:
            from kernel_hook_exec import HookContainmentError
            hook_containment = isinstance(error, HookContainmentError)
        else:
            hook_containment = False
        kind = ("RECOVERY_REQUIRED" if recovery_required else
                "BUDGET_EXHAUSTED" if isinstance(error, SubframeBudgetExhausted) else
                "HOOK_CONTAINMENT" if hook_containment else "ENGINE_ERROR")
        emit("recovery_required" if recovery_required else "engine_error",
             error=type(error).__name__, detail=" ".join(str(error).split())[:300])
        result = (_engine_error_terminal(kind, error) if kind == "ENGINE_ERROR"
                  else {"result": kind, "error": type(error).__name__})
        if recovery_required:
            result["reason_code"] = "RECOVERY_REQUIRED"
            result["error_message"] = " ".join(str(error).split())[:300]
        status = 1
    finally:
        if mutation_runtime is not None:
            result, status = _finish_mutation_runtime(mutation_runtime, result, status, emit)
        if hook_wiring is not None:
            try:
                hook_wiring.close()
            except BaseException as error:  # a failed hook cleanup is terminal
                _reason_log.emit_reason("invariant", "hook_cleanup_failed", "engine_child",
                                        "hook containment could not be verified at cleanup",
                                        f"{type(error).__name__}: {str(error)[:300]}", level="ERROR")
                emit("engine_error", error=type(error).__name__, detail=str(error)[:300])
                result, status = {"result": "HOOK_CONTAINMENT", "error": str(error)[:300]}, 1
        try:
            cleanup = session.stop_all()
            if not cleanup.ok:
                raise RuntimeError(f"CLEANUP_FAILED: {cleanup.detail}")
        except (RuntimeError, OSError) as error:
            _reason_log.emit_reason("invariant", "shell_cleanup_failed", "engine_child",
                                    "shell sessions did not stop cleanly", str(error)[:300], level="ERROR")
            emit("engine_error", error="ShellCleanupFailed", detail=str(error)[:300])
            result, status = {"result": "ENGINE_ERROR", "error": "ShellCleanupFailed"}, 1
        finally:
            try:
                session.close()
            except (RuntimeError, OSError) as error:
                _reason_log.emit_reason("invariant", "shell_cleanup_failed", "engine_child",
                                        "shell sessions did not stop cleanly", str(error)[:300], level="ERROR")
                emit("engine_error", error="ShellCleanupFailed", detail=str(error)[:300])
                result, status = {"result": "ENGINE_ERROR", "error": "ShellCleanupFailed"}, 1
            if slot_lease is not None:
                containment_verified = bool(getattr(session, "containment_verified", False))
                if hasattr(session, "_process") and session._process is None:
                    containment_verified = True
                try:
                    if containment_verified:
                        slot_lease.close()
                    else:
                        slot_lease.handoff_to_host()
                except SlotHandoffDurabilityError as error:
                    emit("engine_error", error=type(error).__name__, detail=str(error)[:300])
                    result, status = {"result": "ENGINE_ERROR", "error": type(error).__name__}, 1
                except (RuntimeError, OSError) as error:
                    emit("engine_error", error="SlotLeaseCleanupFailed", detail=str(error)[:300])
                    result, status = {"result": "ENGINE_ERROR", "error": "SlotLeaseCleanupFailed"}, 1
            signal.signal(signal.SIGTERM, previous_term)
    elapsed = time.monotonic() - started
    atomic_bytes(run / "engine-result.json", json.dumps(
        {"result": result, "seconds": round(elapsed, 3)}, default=str, indent=1).encode("utf-8"))
    emit("engine_end", result=(result or {}).get("result") if isinstance(result, dict) else str(result),
         seconds=round(elapsed, 1))
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
