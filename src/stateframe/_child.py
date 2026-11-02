"""Engine child process: runs one StateFrame task from a prepared run directory.

Started by ``stateframe run`` as ``<python> -B _child.py <run-dir>``. It imports nothing
from the ``stateframe`` package (so it can run under the user's own project interpreter),
only stdlib-only sibling helpers and the vendored engine directory named in the spec.

What it adds around the engine, mirroring the lab release runner's ``checked_ask``:
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
import time
import traceback
import urllib.request
from pathlib import Path

if __package__:
    from .cache_metrics import normalize
    from .replay_tape import CheckpointUnfaithful, Replayer, mask_volatile
    from .skills import SkillError, skills_digest, verify_records
else:  # Executed by path under the selected project interpreter.
    from cache_metrics import normalize
    from replay_tape import CheckpointUnfaithful, Replayer, mask_volatile
    from skills import SkillError, skills_digest, verify_records

# The wrappers' own state (send counters, the id stream position, ...). An engine checkpoint saves
# these by value and a checkpoint restore refills them in place (kernel_checkpoint participants), so
# the re-installed wrappers continue exactly where the interrupted attempt's wrappers were.
HOST_STATE: dict[str, dict] = {}
CHECKPOINT_RESTORE_ENV = "STATEFRAME_CHECKPOINT_RESTORE"

ENGINE_PREFIXES = ("DS_", "KERNEL_", "STATEFRAME_", "REACT_", "PILOT_", "UNIT_PLAN_")
LOG_PREFIX = "[sf]"


def emit(kind: str, **fields: object) -> None:
    """One machine-readable progress line for the parent (stdout)."""
    print(LOG_PREFIX + " " + json.dumps({"kind": kind, **fields}, ensure_ascii=False,
                                         default=str, separators=(",", ":")), flush=True)


def emit_resume_pending_check(spec: dict, resume: dict | None) -> None:
    """Record a routed resume that dropped an unproven in-flight check, on the child's event stream.

    Kept out of the engine's task row: anything there can reach request bytes, and a changed
    request would break the replay this resume depends on.
    """
    check = resume.get("pending_check") if spec.get("routes") and isinstance(resume, dict) else None
    if isinstance(check, dict):
        emit("routes_resume_pending_check_skipped", **check)


def install_environment(spec: dict) -> None:
    routed_keys: dict[str, str] = {}
    if isinstance(spec.get("routes"), dict):
        for settings in spec["routes"].get("roles", {}).values():
            name = settings.get("api_key_env") if isinstance(settings, dict) else None
            if isinstance(name, str) and name in os.environ:
                routed_keys[name] = os.environ[name]
    for key in list(os.environ):
        if key.startswith(ENGINE_PREFIXES) and not key.startswith("STATEFRAME_TASK_"):
            del os.environ[key]
    os.environ.update({str(k): str(v) for k, v in spec["environment"].items()})
    os.environ.update(routed_keys)


def _install_direct_opener() -> None:
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))


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


def install_wire_recorder(run: Path, chat_url: str, replayer: Replayer | None = None,
                          routed_endpoints: set[str] | None = None, hook_wiring=None) -> None:
    """Record every physical chat send: exact identity bytes, response bytes, usage.

    On ``stateframe resume`` a send the interrupted attempt already completed is answered from its
    recorded response (``replayer``), and is recorded here like any other send, marked ``replayed``.
    """
    import kernel_bounded_http

    wire_dir = run / "wire"
    wire_dir.mkdir(exist_ok=True)
    original = kernel_bounded_http.request_bytes
    counter = {"n": 0}
    HOST_STATE["wire"] = counter

    def atomic_bytes(path: Path, data: bytes) -> None:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
            temp = Path(stream.name)
            stream.write(data)
        try:
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)

    def recorded(wire: bytes, endpoint: str, key: str, **kwargs):
        route = kwargs.pop("route_meta", None)
        if routed_endpoints is None and endpoint != chat_url:
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
        if replayer is not None and replayer.strict and replayer.diverged is not None:
            raise CheckpointUnfaithful("an earlier rebuilt request differed; nothing more is sent")
        if replayer is not None and replayer.route_mismatch:
            raise CheckpointUnfaithful("an earlier routed replay differed; nothing more is sent")
        counter["n"] += 1
        index = counter["n"]
        stem = wire_dir / f"{index:04d}"
        if routed_endpoints is not None:
            atomic_bytes(stem.with_suffix(".route.json"), json.dumps(
                route, sort_keys=True, separators=(",", ":")).encode())
            atomic_bytes(stem.with_suffix(".request.json"), wire)
        else:
            stem.with_suffix(".request.json").write_bytes(wire)
        started = time.monotonic()
        replayed = None
        # An in-flight request to verify keeps the replayer in the path even with no tail left to answer.
        if replayer is not None and (replayer.active or replayer.pending is not None):
            replayed = replayer.answer(wire, route=route)
            if routed_endpoints is not None and replayer.pending_verified is not None:
                atomic_bytes(run / "resume-pending-verified.json", json.dumps(
                    replayer.pending_verified, sort_keys=True, separators=(",", ":")).encode())
            if replayed is None and replayer.diverged is not None:
                (run / "resume-divergence.json").write_text(json.dumps(
                    {**replayer.diverged, "send": index, "checkpoint": replayer.strict}, indent=1))
                if routed_endpoints is not None and replayer.route_mismatch:
                    raise CheckpointUnfaithful(f"send {index}: {replayer.diverged['reason']}")
                if (run / "tool-tape.jsonl").is_file() or hook_wiring is not None:
                    from kernel_tool_tape import ToolReplayRefused
                    raise ToolReplayRefused(f"send {index}: {replayer.diverged['reason']}")
                if replayer.strict:
                    emit("checkpoint_unfaithful", send=index, **{k: v for k, v in replayer.diverged.items()
                                                                 if k != "first_difference"})
                    (run / "checkpoint-refused.json").write_text(json.dumps(
                        {"error": "CheckpointUnfaithful", "send": index, "reason": replayer.diverged["reason"]},
                        indent=1))
                    raise CheckpointUnfaithful(f"send {index}: {replayer.diverged['reason']}")
                emit("replay_diverged", send=index, **{k: v for k, v in replayer.diverged.items()
                                                       if k != "first_difference"})
            elif replayed is not None and not replayer.active:
                emit("replay_done", send=index, replayed=replayer.cursor)
            if replayer.strict and replayer.diverged is None and not replayer.active \
                    and replayer.pending is None \
                    and not getattr(replayer, "verified_emitted", False):
                replayer.verified_emitted = True
                emit("checkpoint_verified", send=index, replayed=replayer.cursor)
        elif replayer is not None:
            from kernel_tool_tape import end_replay
            end_replay()
        if (replayer is not None and replayed is None and replayer.diverged is None
                and not replayer.active and replayer.pending is None):
            from kernel_tool_tape import end_replay
            end_replay()
        transport_attempts: list[dict[str, object]] = []
        try:
            response = (replayed if replayed is not None else original(
                wire, endpoint, key, record_attempts=transport_attempts.extend,
                **({**kwargs, "route_meta": route} if routed_endpoints is not None else kwargs)))
        except Exception as error:
            (wire_dir / f"{index:04d}.error.txt").write_text(type(error).__name__ + "\n")
            if transport_attempts:
                with (run / "sends.jsonl").open("a") as stream:
                    stream.write(json.dumps({"send": index, "request_bytes": len(wire),
                                             "request_sha256": hashlib.sha256(wire).hexdigest(),
                                             "transport_attempts": transport_attempts,
                                             "error": type(error).__name__}) + "\n")
            emit("send_error", send=index, error=type(error).__name__)
            raise
        if hook_wiring is not None:
            from kernel_measurement import ACTIVE_REQUEST
            capture = ACTIVE_REQUEST.get()
            request_id = getattr(capture, "request_id", None)
            hook_wiring.sent(wire, str(request_id) if request_id is not None else f"send:{index}")
        elapsed = time.monotonic() - started
        stem.with_suffix(".response.json").write_bytes(response)
        try:
            parsed = json.loads(response)
        except (ValueError, RecursionError):
            parsed = None
        usage = _usage(parsed)
        try:
            body = json.loads(wire)
        except (ValueError, RecursionError):
            body = None
        messages = body.get("messages") if isinstance(body, dict) else None
        record = {"send": index, "request_bytes": len(wire),
                  "request_sha256": hashlib.sha256(wire).hexdigest(),
                  "messages": len(messages) if isinstance(messages, list) else None,
                  "seconds": round(elapsed, 3), **usage, "actions": _actions(parsed)}
        if replayed is not None:
            record.update(replayed=True, replay_position=replayer.cursor, replay_total=len(replayer.tape))
        if transport_attempts:
            record["transport_attempts"] = transport_attempts
        with (run / "sends.jsonl").open("a") as stream:
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
            emit("token_counting_unavailable")
        return result

    measured._stateframe_fallback_warning = True
    RequestTokenMeter.measure = measured


def install_checked_ask(rr, spec: dict, run: Path, hook_wiring=None) -> None:
    """The release runner's provider callback contract, minus the lab release fences."""
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
            (requests_dir / f"{index:04d}.json").write_text(
                json.dumps(record, ensure_ascii=False, default=str, indent=1))
            emit("request", request=index, turn=turn, messages=len(messages),
                 roles="".join(m.get("role", "?")[0] for m in messages if isinstance(m, dict)),
                 tools=[t.get("function", {}).get("name") for t in tools if isinstance(t, dict)])

        from kernel_request_kind import classify, RequestKindError
        if capture is None:
            raise RequestKindError("primary_capture_missing")
        if classify(capture, runtime, {"messages": messages, "tools": tools}):
            def bound_gate(record, bound_capture, bound_runtime):
                from capability_request_preflight import PRECOMPOSED_HOST_TEXT_SCHEMA
                from kernel_request_seal import RequestSealError
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
        return original_ask(messages, **kwargs)

    import inspect
    from capability_request_preflight import preflight_presentation_or_degrade as _preflight
    preflight_parameters = set(inspect.signature(_preflight).parameters)
    # The engine asks the wrapper for its input preparation before runtime.prepare qualifies
    # source locations. The lab runner's only input step is the host-capacity notice, which the
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


def install_repo_context(rr, parent: Path) -> None:
    """Put the whole repository into the engine's sandbox, not only the declared files.

    The engine's multi-file branch builds a fresh root from the declared files and public tests.
    Before its public tests are written, copy every other file of the run's read-only parent
    snapshot in, so imports, fixtures and configs the checks need are there. Declared files are
    already written by the engine and are never overwritten here.
    """
    import shutil

    original = rr._materialize_public_tests
    done: dict[str, Path] = {}
    HOST_STATE["repo_context"] = done

    def materialize(root, fileset, row):
        root = Path(root).resolve()
        if done and done["root"] != root:
            raise RuntimeError("engine switched sandbox roots mid-run")
        if not done:
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
    """Make the engine's run-scoped ids (uuid4: run ids, receipts, handles) a stream fixed by the run's seed.

    Ids stay unique per run (the runner draws a random seed for each run), but the SAME run started again
    from the same seed gets the same ids in the same order. ``stateframe resume`` relies on that: the replayed
    requests then match the recorded ones byte for byte, including the receipts and hashes derived from ids.
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
    counter = {"n": 0, "substituting": recorded_dir is not None, "on": True}
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
        if counter["substituting"] and replayer is not None and replayer.active:
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
                counter["substituting"] = False  # the replay's state differs from here on; never patch again
        (folder / f"{index:04d}.json.z").write_bytes(zlib.compress(json.dumps(record).encode("utf-8")))
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
                        os.killpg(os.getpgrp(), signal.SIGKILL)
                finally:
                    os._exit(137)

    threading.Thread(target=loop, name="stateframe-parent-watch", daemon=True).start()


CHECKPOINT_KEY = Path(".stateframe") / "checkpoint.key"  # under the user's home


def checkpoint_key() -> bytes | None:
    """This install's checkpoint signing key: 32 random bytes in a 0600 file of a 0700 directory
    owned by this user, outside every run directory. Created on first use; None if unusable
    (then no checkpoint is written, and resume replays). The directory and the key are opened without
    following symlinks and checked by fstat on the open descriptors, so an existing directory that is
    a symlink, another user's, or group/other accessible is refused, not trusted."""
    import secrets
    import stat as stat_module

    try:
        home = Path.home()  # RuntimeError when HOME is unset and the uid has no passwd entry (containers)
    except (RuntimeError, KeyError, OSError) as error:
        emit("checkpoint_key_unusable", reason="home directory unknown: " + type(error).__name__)
        return None
    path = home / CHECKPOINT_KEY

    def publish(replace: bool) -> None:
        # Complete key in a private temp file first, then link/replace it into place: a crash can never
        # leave a short key behind, and a concurrent reader never sees a partial one.
        fd, temp = tempfile.mkstemp(prefix=".checkpoint.key.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(secrets.token_bytes(32))
                stream.flush()
                os.fsync(stream.fileno())
            if replace:
                os.replace(temp, path)
            else:
                try:
                    os.link(temp, path)
                except FileExistsError:
                    pass  # another process won the race: use its key
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    def private(info: os.stat_result, kind) -> bool:
        return kind(info.st_mode) and info.st_uid == os.getuid() and not info.st_mode & 0o077

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_fd = key_fd = None
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow)
        if not private(os.fstat(directory_fd), stat_module.S_ISDIR):
            emit("checkpoint_key_unusable", reason="key directory is not a private directory of this user")
            return None
        if not os.path.lexists(path):
            publish(replace=False)
        info = os.lstat(path)
        if stat_module.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and info.st_size < 32:
            # No checkpoint can be signed with a short key (the engine refuses one), so nothing is lost.
            print("stateframe: checkpoint key was incomplete (an earlier crash?); replaced it", file=sys.stderr)
            publish(replace=True)
        key_fd = os.open(CHECKPOINT_KEY.name, os.O_RDONLY | nofollow, dir_fd=directory_fd)
        if not private(os.fstat(key_fd), stat_module.S_ISREG):
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
    return key if len(key) >= 32 else None


def register_host_state(*, required: bool, secret: str = "") -> None:
    """Hand the wrappers' state to the engine's checkpoint (an older engine has none: skip)."""
    try:
        import kernel_checkpoint
    except ImportError:
        if required:
            raise
        return
    saving = os.environ.get("STATEFRAME_CHECKPOINT", "off").strip().lower() not in {"", "off", "0"}
    if saving and hasattr(kernel_checkpoint, "register_signing_key"):
        key = checkpoint_key()
        if key is not None:
            kernel_checkpoint.register_signing_key(key)
    for name, state in HOST_STATE.items():
        kernel_checkpoint.register_participant(name, state)
    if secret and hasattr(kernel_checkpoint, "register_secret"):
        kernel_checkpoint.register_secret(secret)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: _child.py <run-dir>", file=sys.stderr)
        return 2
    run = Path(argv[1]).resolve(strict=True)
    watch_parent()
    lock = hold_engine_lock(run)  # noqa: F841 - held until exit
    spec = json.loads((run / "spec.json").read_text())
    resume = spec.get("resume") if isinstance(spec.get("resume"), dict) else None
    checkpoint = resume.get("checkpoint") if resume is not None and isinstance(resume.get("checkpoint"), dict) else None
    if checkpoint is not None:
        # Checkpoint resume: the engine restores its state as of send ``sends``; only the recorded
        # exchanges after it are replayed, strictly (any difference refuses before sending).
        tail = [item for item in resume.get("tape") or [] if int(item["send"]) > int(checkpoint["sends"])]
        replayer = Replayer(tail, strict=True, pending=checkpoint.get("pending"),
                            route_changes=resume.get("route_changes"))
    else:
        replayer = (Replayer(resume.get("tape") or [], resume.get("model_swap"),
                             pending=resume.get("pending"), route_changes=resume.get("route_changes"))
                    if resume is not None else None)
    install_environment(spec)
    # Keep all engine urlopen calls direct, including those outside the primary transport.
    _install_direct_opener()
    if checkpoint is not None:
        os.environ[CHECKPOINT_RESTORE_ENV] = str(checkpoint["dir"])
    install_run_identity(spec.get("run_identity_seed"))  # before any engine import
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
                "shell", "job_wait", "job_output", "job_stop", "file_delete", "file_move", "git", "web_fetch", "skill_load"))
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
    except (ToolSettingsError, CustomToolError, ToolTapeError, SkillError, OSError, KeyError, ValueError) as error:
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
    try:
        if settings is not None:
            configure_shell(run, settings, spec, session)
            from kernel_shell_tool import service as shell_service
            from kernel_agent_tools import configure as configure_agent_tools
            configure_agent_tools(shell_service(), custom, mcp_servers,
                                  resume_active=resume is not None)
        from kernel_skills import configure as configure_skills
        configure_skills(skill_records, skill_digest, spec.get("chat_url"))
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
        install_repo_context(rr, Path(spec["parent_snapshot"]))
        install_compact_capture(Path(spec["parent_snapshot"]))
        install_wire_recorder(run, spec["chat_url"], replayer, routed_endpoints, hook_wiring)
        previous_checks = Path(resume["previous_attempt"]) / "checks" if resume is not None else None
        install_check_recorder(run, replayer, previous_checks)
        install_token_count_warning()
        install_checked_ask(rr, spec, run, hook_wiring)
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
        emit("engine_start", turns=rr.MAX_STEPS,
             **({"resume_attempt": resume.get("attempt"), "replay_sends": len(replayer.tape)}
                if resume is not None and replayer is not None else {}))
        result = rr.work(row)
        status = 0
    except BaseException as error:  # noqa: BLE001 - report every engine failure as a typed record
        (run / "child-traceback.txt").write_text(traceback.format_exc())
        emit("engine_error", error=type(error).__name__, detail=" ".join(str(error).split())[:300])
        if hook_wiring is not None:
            from kernel_hook_exec import HookContainmentError
            kind = "HOOK_CONTAINMENT" if isinstance(error, HookContainmentError) else "ENGINE_ERROR"
        else:
            kind = "ENGINE_ERROR"
        result = {"result": kind, "error": type(error).__name__}
        status = 1
    finally:
        if hook_wiring is not None:
            try:
                hook_wiring.close()
            except BaseException as error:  # a failed hook cleanup is terminal
                emit("engine_error", error=type(error).__name__, detail=str(error)[:300])
                result, status = {"result": "HOOK_CONTAINMENT", "error": str(error)[:300]}, 1
        try:
            cleanup = session.stop_all()
            if not cleanup.ok:
                raise RuntimeError(f"CLEANUP_FAILED: {cleanup.detail}")
        except (RuntimeError, OSError) as error:
            emit("engine_error", error="ShellCleanupFailed", detail=str(error)[:300])
            result, status = {"result": "ENGINE_ERROR", "error": "ShellCleanupFailed"}, 1
        finally:
            try:
                session.close()
            except (RuntimeError, OSError) as error:
                emit("engine_error", error="ShellCleanupFailed", detail=str(error)[:300])
                result, status = {"result": "ENGINE_ERROR", "error": "ShellCleanupFailed"}, 1
            signal.signal(signal.SIGTERM, previous_term)
    elapsed = time.monotonic() - started
    (run / "engine-result.json").write_text(json.dumps(
        {"result": result, "seconds": round(elapsed, 3)}, default=str, indent=1))
    emit("engine_end", result=(result or {}).get("result") if isinstance(result, dict) else str(result),
         seconds=round(elapsed, 1))
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
