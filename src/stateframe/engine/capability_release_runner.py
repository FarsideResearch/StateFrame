"""Two-arm working-source completeness diagnostic within fresh StateFrames; no private grading."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping
from types import SimpleNamespace

from capability_runtime_release import (
    activate_release,
    check_release_admission,
    verify_release,
)

ROOT = Path(__file__).resolve().parent
SOURCE = Path(__file__).resolve().parent
_CONFIG_PREFIXES = ("DS_", "KERNEL_", "STATEFRAME_", "REACT_")
HOST_TASK_POLICY_PREFIX = "STATEFRAME_TASK_"  # host launch policy, survives the config purge


def _resume_descriptor(run: Path, follow_up: str | None):
    """Select one retained v29 session and bind resume to its trusted journal head."""
    from kernel_continuation_restart import WorkerResume

    parent = run / "worker" / "continuation-workspaces"
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("resume requires the retained continuation workspace")
    sessions = []
    for directory in sorted(parent.iterdir()):
        initialized = directory / "initialized.json"
        if directory.is_dir() and not directory.is_symlink() and initialized.is_file():
            sessions.append(directory)
    if len(sessions) != 1:
        raise ValueError("resume requires exactly one retained session")
    directory = sessions[0]
    try:
        initialized = json.loads((directory / "initialized.json").read_text())
        journal_path = Path(initialized["journal_path"])
        lines = journal_path.read_bytes().splitlines()
        head = json.loads(lines[-1]).get("record_hash") if lines else None
    except (KeyError, OSError, TypeError, ValueError, UnicodeDecodeError) as error:
        raise ValueError("resume journal head is unreadable") from error
    if not isinstance(head, str) or not head.startswith("sha256:"):
        raise ValueError("resume journal has no trusted head")
    return WorkerResume(
        workspace=directory,
        expected_journal_head=head,
        follow_up=follow_up,
        request_id=uuid.uuid4().hex if follow_up is not None else None,
    )


def prepare_request(messages, plan):
    """Add the opt-in host receipt at the same boundary as runner preflight."""
    if not getattr(plan, "enabled", False):
        return messages
    notice = getattr(plan, "rendered_notice", "")
    if not notice or any(message.get("role") == "system"
                         and isinstance(message.get("content"), str)
                         and (message["content"] == notice
                              or message["content"].startswith(notice + "\n"))
                         for message in messages if isinstance(message, dict)):
        return messages
    return [{"role": "system", "content": notice}, *messages]


def _capacity_plan(spec):
    """Keep the default runner free of capacity imports and host probing."""
    environment = spec.get("environment", {})
    environment = environment if isinstance(environment, Mapping) else {}
    raw = str(environment.get("CAPABILITY_HOST_CAPACITY", "off") or "off").strip().lower()
    if raw == "off":
        return SimpleNamespace(enabled=False)
    from capability_host_capacity import build_plan
    return build_plan(spec)



def save(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def runtime_record(run: Path) -> Path:
    """Bind an explicit runtime identity without replacing older evidence."""
    spec = json.loads((run / "spec.json").read_text())
    if "CAPABILITY_ATTEMPT_DEADLINE_MONOTONIC" in spec["environment"]:
        raise ValueError("spec may not override supervisor deadline")
    raw = spec.get("runtime_identity_path", "/work/evidence/minicpm-runtime.json")
    if not isinstance(raw, str) or not Path(raw).is_absolute():
        raise ValueError("absolute runtime identity path required")
    path = Path(raw)
    if path.is_symlink() or not path.is_file():
        raise ValueError("runtime identity must be a regular file")
    if "runtime_identity_path" in spec:
        record = json.loads(path.read_text())
        environment = spec["environment"]
        fields = {
            "DS_MODEL": "model", "REACT_QUANTIZATION": "quantization",
            "REACT_TOKENIZER": "tokenizer_sha256",
            "REACT_CHAT_TEMPLATE": "chat_template_sha256",
            "REACT_SERVER_BUILD": "runtime_build",
            "REACT_ENVIRONMENT_REVISION": "server_configuration_sha256",
        }
        if any(not isinstance(record.get(field), str) or not record[field]
               or environment.get(key) != record[field] for key, field in fields.items()):
            raise ValueError("run environment differs from runtime identity")
        slots = record.get("slots")
        slot = environment.get("REACT_SLOT_ID")
        if (type(slots) is not int or slots < 1 or not isinstance(slot, str)
                or not slot.isdigit() or not 0 <= int(slot) < slots):
            raise ValueError("run slot outside recorded server capacity")
    return path.resolve(strict=True)


def source_hashes(run: Path) -> dict[str, str]:
    manifest = verify_release(ROOT)
    spec = json.loads((run / "spec.json").read_text())
    run_environment = spec.get("environment")
    run_environment = run_environment if isinstance(run_environment, dict) else {}
    gate_values = (
        os.environ.get("STATEFRAME_MECHANISM_ADMISSION", "off"),
        run_environment.get("STATEFRAME_MECHANISM_ADMISSION", "off"),
    )
    # The admission module is optional on the historical path.  Do not import
    # it merely to discover that both the chain and leaf explicitly stayed off.
    if any(str(value).strip().lower() != "off" for value in gate_values):
        from kernel_mechanism_admission import require_declaration, resolve_policy

        policy = resolve_policy(run_environment=run_environment)
        if policy.enabled:
            require_declaration(policy)
            check_release_admission(ROOT, required_switches=policy.required_switches)
    paths = [ROOT / name for name in manifest["files"]]
    paths += [ROOT / "release-manifest.json", run / "spec.json",
              run.parent / "protocol.json", runtime_record(run)]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _install_declared_environment(spec: Mapping[str, object]) -> dict[str, str]:
    """Install the immutable run environment before importing runtime consumers."""

    raw_environment = spec.get("environment")
    if not isinstance(raw_environment, Mapping):
        raise ValueError("run spec has no environment object")
    environment: dict[str, str] = {}
    for key, value in raw_environment.items():
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"run environment value is not scalar: {key}")
        environment[str(key)] = str(value)
    # Host task-execution policy (STATEFRAME_TASK_NETWORK / _ENV_KEYS / _PYTEST_PLUGINS) is declared
    # by the worker LAUNCH (docker --env), never by a spec.  10-06 batch 837: this purge was the hop
    # that dropped it (source_env had 256 STATEFRAME_* keys and zero STATEFRAME_TASK_*), so the
    # sandbox lost --share-net and the capability canary could never prove runtime policy.  Host
    # policy is authoritative: a spec may restate it verbatim but never override or invent it.
    conflicting = sorted(
        key for key, value in environment.items()
        if key.startswith(HOST_TASK_POLICY_PREFIX) and os.environ.get(key) != value
    )
    if conflicting:
        raise ValueError("run spec conflicts with host task-execution policy: " + ", ".join(conflicting))
    for key in list(os.environ):
        if key.startswith(_CONFIG_PREFIXES) and not key.startswith(HOST_TASK_POLICY_PREFIX):
            del os.environ[key]
    os.environ.update(environment)
    expected = {key: value for key, value in environment.items() if key.startswith(_CONFIG_PREFIXES)}
    actual = {key: value for key, value in os.environ.items()
              if key.startswith(_CONFIG_PREFIXES) and not key.startswith(HOST_TASK_POLICY_PREFIX)}
    if actual != {k: v for k, v in expected.items() if not k.startswith(HOST_TASK_POLICY_PREFIX)}:
        raise ValueError("process environment differs from declared run environment")
    return environment


def _bind_endpoint_size(run: Path, *, resume: bool) -> None:
    """Pin resolved slot limits before a resumed worker can prepare requests."""
    from kernel_endpoint_size import resolve_endpoint_size

    size = resolve_endpoint_size()
    selected = size.run_config()
    path = run / "endpoint-size.json"

    def mismatch(recorded: object) -> ValueError:
        values = recorded if isinstance(recorded, Mapping) else {}
        context = values.get("context_tokens")
        budget = values.get("prompt_budget")
        configured = values.get("configured")
        return ValueError(
            "resume endpoint size differs from recorded run config: "
            f"STATEFRAME_CONTEXT_TOKENS={selected['context_tokens']}, "
            f"STATEFRAME_PROMPT_BUDGET={selected['prompt_budget']}, "
            f"configured={selected['configured']}; recorded limit "
            f"context={context if type(context) is int else 'invalid'}, "
            f"budget={budget if type(budget) is int else 'invalid'}, "
            f"configured={configured if type(configured) is bool else 'invalid'}. "
            "Set both variables to the recorded values and setting "
            "state, or start a new run.")

    if resume:
        try:
            recorded = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            raise ValueError("resume has no valid recorded endpoint size") from error
        if recorded != selected:
            raise mismatch(recorded)
        if line := size.startup_line():
            print(line, file=sys.stderr)
        return
    try:
        with path.open("x") as stream:
            json.dump(selected, stream, sort_keys=True)
            stream.write("\n")
    except FileExistsError:
        recorded = json.loads(path.read_text())
        if recorded != selected:
            raise mismatch(recorded)
    if line := size.startup_line():
        print(line, file=sys.stderr)


def _served_model_identity(spec: Mapping[str, object]) -> str:
    """Resolve the served checkpoint label independently from request alias."""

    environment = spec.get("environment")
    if isinstance(environment, Mapping):
        explicit = environment.get("REACT_SERVED_MODEL")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()[:160]
    raw_path = spec.get("runtime_identity_path")
    if not isinstance(raw_path, str) or not raw_path:
        return ""
    path = Path(raw_path)
    if not path.is_file() or path.is_symlink():
        return ""
    try:
        identity = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(identity, Mapping):
        return ""
    explicit = identity.get("served_model")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()[:160]
    server_config = identity.get("server_config")
    model_path = server_config.get("model_path") if isinstance(server_config, Mapping) else None
    if isinstance(model_path, str) and model_path.strip():
        return model_path.replace("\\", "/").rsplit("/", 1)[-1][:160]
    return str(identity.get("model", ""))[:160]


def _postflight_effective_config(run: Path, spec: Mapping[str, object]) -> None:
    """Reject a completed run whose durable config record differs from its spec."""

    environment = spec.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError("postflight cannot validate a missing spec environment")
    expected_keys = {
        str(key) for key in environment
        if str(key).startswith(("STATEFRAME_", "DS_"))
    }
    expected_sources = spec.get("environment_sources")
    expected_sources = expected_sources if isinstance(expected_sources, Mapping) else {}
    expected = {
        key: {
            "value": str(environment[key]),
            "source": str(expected_sources.get(key, "spec.json")),
        }
        for key in sorted(expected_keys)
    }
    journals = sorted((run / "worker" / "measurement-runs").glob("*/events.jsonl"))
    events: list[dict[str, object]] = []
    from kernel_measurement import resolve_journal_blob_refs

    for path in journals:
        blob_cache: dict[str, object] = {}
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                value = resolve_journal_blob_refs(json.loads(line), path, blob_cache)
                if isinstance(value, dict) and value.get("kind") == "model_policy_resolved":
                    events.append(value)
    if len(events) != 1:
        raise ValueError(f"effective config postflight expected one record, found {len(events)}")
    # Host task policy (STATEFRAME_TASK_*) is declared by the docker launch, not the spec: the purge
    # above lets it survive and the kernel records it from the process environment.  Compare the
    # SAME allowlist here, or every run ends in FINALIZATION_ERROR after a clean finish (batch 857:
    # 60 requests, RESPONSE_BUDGET_EXHAUSTED, then this raise; no salvage grade).  Any other key
    # that drifts, appears, or disappears still raises.
    effective = events[0].get("effective_config")
    if isinstance(effective, Mapping):
        effective = {key: value for key, value in effective.items()
                     if not str(key).startswith(HOST_TASK_POLICY_PREFIX)}
    if effective != expected:
        raise ValueError("effective config differs from declared spec environment")
    policy = events[0].get("policy")
    if not isinstance(policy, Mapping) or policy.get("requested_model") != str(environment.get("DS_MODEL", "")):
        raise ValueError("effective model identity differs from declared DS_MODEL")



def _row_has_target(row: Mapping[str, object]) -> bool:
    entry = row.get("entry")
    return (any(isinstance(row.get(key), str) and str(row.get(key)).strip()
                for key in ("path", "rel"))
            or (isinstance(entry, Mapping) and isinstance(entry.get("path"), str)
                and bool(entry["path"].strip())))


def _public_solver_row_from_spec(spec: Mapping[str, object]) -> dict[str, object]:
    """Project only ``public_row`` into the solver; evaluator state is opaque."""

    public = spec.get("public_row")
    if not isinstance(public, Mapping):
        raise ValueError("release spec requires a public_row mapping")
    from capability_study_support import public_parent_row
    from kernel_aha11_native_honesty import public_solver_row

    return public_solver_row(public_parent_row(dict(public)))


def _refuse_unresolvable_target(row: Mapping[str, object]) -> None:
    """Audit 2026-10-05 #3: refuse in one line before the runtime starts.

    The retained continuation session needs a resolved target (it records it
    for resume); without one batches 889/894 died in <15 s with an untyped
    AttributeError.  Targetless (localize-first) runs without continuation stay
    legal: the frame series and S1 localize path handle ``target is None``.
    """
    from kernel_continuation_runtime import enabled as continuation_enabled
    if continuation_enabled() and not _row_has_target(row):
        raise ValueError("live spec has no resolvable public target; refusing before the runtime "
                         "starts (declared public paths and row target are both absent)")


def check_request_contract(messages, tools, public_row: Mapping[str, object], *,
                           require_edit_scope: bool = True):
    """Mode-aware provider-request contract; raises before any provider call.

    Pure (no run directory, no runtime) so the cut gate's scripted first-turn replays can run
    the exact same check in every mode the release supports: a release whose first request
    violates its mode's contract must fail the cut, not die on the GPU (10-13: r2BG LEAN died
    before request 1 on the full-mode EditAuthority assertion).
    Returns ``(edit_tool_function_or_None, system_text)``.
    """
    edit = next((tool["function"] for tool in tools or []
                 if tool["function"]["name"] == "edit"), None)
    system = "\n".join(message.get("content") or "" for message in messages
                       if message["role"] == "system")
    from kernel_lean import PLAIN_SYSTEM_PROMPT, enabled as lean_enabled
    if lean_enabled():
        # LEAN deliberately sends the plain system prompt (react_repair + kernel_lean); the
        # full-mode EditAuthority line is not part of its contract.
        # The runtime may append exactly one canonical continuation-limits line
        # (kernel_stateframe_runtime: "[CONTINUATION LIMITS | ...]", single line); nothing else.
        tail = system[len(PLAIN_SYSTEM_PROMPT):] if system.startswith(PLAIN_SYSTEM_PROMPT) else None
        if not (tail == "" or (
                isinstance(tail, str) and tail.startswith("\n\n[CONTINUATION LIMITS |")
                and "\n" not in tail[2:] and "\r" not in tail)):
            raise ValueError("LEAN request system message is not the plain system prompt")
    elif edit is not None:
        from kernel_edit_affordance import contract
        entry = public_row.get("entry") or {}
        scope = contract(multi_file=bool(public_row.get("files")),
                         entry_path=entry.get("path"), entry_fn=entry.get("fn"),
                         target_fn=public_row.get("fn", ""))
        if scope is None and require_edit_scope:
            raise ValueError("release study requires the truthful edit contract")
        # Offline harnesses filter KERNEL_* (no KERNEL_EDIT_AFFORDANCE), so no scope exists there;
        # the live runner always declares it and keeps the refusal above.
        if scope is not None:
            # r38 single EditAuthority: the tool points at the authority, the system carries its
            # one line, and the legacy scope sentence must be gone (scope rendered once).
            from kernel_edit_affordance import AUTHORITY_EDIT_DESCRIPTION, AUTHORITY_SYSTEM_LINE
            assert edit["description"].startswith(AUTHORITY_EDIT_DESCRIPTION)
            assert AUTHORITY_SYSTEM_LINE in system
            assert scope["instruction"] not in system
    assert "complete top-level function, not a class" not in system
    assert "edit replaces a complete top-level function only" not in system
    return edit, system


AGENT_LOOP_ENV = "STATEFRAME_AGENT_LOOP"


def agent_loop_selection(spec: Mapping[str, object]) -> str:
    """Run-pinned engine choice from the declared spec environment; default ``full``."""
    environment = spec.get("environment")
    raw = environment.get(AGENT_LOOP_ENV, "full") if isinstance(environment, Mapping) else "full"
    value = str(raw or "full").strip().lower()
    if value not in {"full", "plain"}:
        raise ValueError(f"{AGENT_LOOP_ENV} must be 'full' or 'plain', not {raw!r}")
    return value


def _child_plain(run: Path, spec: Mapping[str, object], fence, *, resume: bool,
                 follow_up: str | None) -> None:
    """Plain-loop engine: release fence, public row, serialized checks, real candidate export."""
    environment = spec["environment"]
    true_resume = str(environment.get("STATEFRAME_TRUE_RESUME",
                                      os.environ.get("STATEFRAME_TRUE_RESUME", "off"))).strip().lower()
    if resume or follow_up is not None or true_resume == "on":
        raise ValueError("STATEFRAME_AGENT_LOOP=plain does not support resume; refusing before launch")
    if str(environment.get("CAPABILITY_HOST_CAPACITY", "off") or "off").strip().lower() != "off":
        raise ValueError("STATEFRAME_AGENT_LOOP=plain does not support CAPABILITY_HOST_CAPACITY; "
                         "refusing before launch")
    sys.path.insert(0, str(ROOT))
    sys.argv[:] = [sys.argv[0]]
    output = run / "worker"
    output.mkdir()
    (output / "results").mkdir()
    (output / "scratch").mkdir()
    tempfile.tempdir = str(output / "scratch")
    from capability_study_worker import unscored_solver_result
    from kernel_aha11_native_honesty import build_honest_release_receipt
    from kernel_plain_host import run_plain
    from kernel_plain_loop import SYSTEM, TOOLS

    row = _public_solver_row_from_spec(spec)
    parent_source = str(row.get("parent_src", ""))
    parent_revision = "sha256:" + hashlib.sha256(parent_source.encode("utf-8")).hexdigest()
    save(run / ("honest-release-input-" + uuid.uuid4().hex + ".json"),
         build_honest_release_receipt(row, source_revision=parent_revision, candidate_revision="unobserved",
                                      parent_status="UNCOVERED", candidate_status="UNCOVERED",
                                      terminal_status="INCOMPLETE",
                                      limitations=("candidate execution has not started",)))
    terminal_smoke = os.environ.get("LAUNCH_SMOKE_TERMINAL", "0") == "1"
    if terminal_smoke:
        os.environ["PYTHONPATH"] = str(ROOT)
    row.update(_skip_db=True, _bridge_capture=(None if terminal_smoke else str(output / "candidates")))
    save(run / "tool-contract-preflight.json", {
        "checked_before_provider": True, "agent_loop": "plain", "tools": TOOLS,
        "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest()})

    def before_request(_body: dict) -> None:
        verify_release(ROOT)
        with (run / "release-imports.jsonl").open("a") as audit:
            audit.write(json.dumps({"imports": fence.snapshot()}) + "\n")

    result = run_plain(row, output, before_request=before_request)
    _postflight_plain_config(run, spec)
    save(run / "first-outcome.json", unscored_solver_result(result))
    save(run / ("honest-release-receipt-" + uuid.uuid4().hex + ".json"),
         build_honest_release_receipt(
             row, source_revision=parent_revision, candidate_revision=str(result["candidate_revision"]),
             parent_status=str(result["parent_status"]), candidate_status=str(result["candidate_status"]),
             terminal_status=str(result["result"]),
             limitations=("plain engine: parent behaviour was not executed; public checks only",)))
    verify_release(ROOT)
    save(run / "release-end.json", {"release": str(ROOT), "imports": fence.snapshot(), "resume": False,
                                    "agent_loop": "plain"})


def _postflight_plain_config(run: Path, spec: Mapping[str, object]) -> None:
    """The plain engine has no runtime journal: compare the live process env to the spec."""
    environment = spec.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError("postflight cannot validate a missing spec environment")
    expected = {str(key): str(value) for key, value in environment.items()
                if str(key).startswith(("STATEFRAME_", "DS_")) and not str(key).startswith(HOST_TASK_POLICY_PREFIX)}
    actual = {key: value for key, value in os.environ.items()
              if key.startswith(("STATEFRAME_", "DS_")) and not key.startswith(HOST_TASK_POLICY_PREFIX)}
    save(run / "worker" / "plain" / "effective-config.json",
         {"agent_loop": "plain", "expected": expected, "actual_matches": actual == expected,
          "served_model": _served_model_identity(spec)})
    if actual != expected:
        raise ValueError("effective config differs from declared spec environment")


def _make_checked_ask(original_ask, *, run, spec, fence, capacity_plan):
    """Build the release request wrapper used at the live provider callback."""
    def checked_ask(messages, **kwargs):
        messages = prepare_request(messages, capacity_plan)
        verify_release(ROOT)
        with (run / "release-imports.jsonl").open("a") as audit:
            audit.write(json.dumps({"imports": fence.snapshot()}) + "\n")
        tools = kwargs.get("tools") or []
        # The green-lock terminal menu legitimately withholds `edit` (done +
        # run_tests only); the edit contract is checked whenever edit is offered.
        edit, system = check_request_contract(messages, tools, spec["public_row"])
        receipt = run / "tool-contract-preflight.json"
        if edit is not None and not receipt.exists():
            save(receipt, {"checked_before_provider": True,
                           "edit_schema": edit, "system_sha256":
                           hashlib.sha256(system.encode()).hexdigest()})
        from kernel_pilot import ACTIVE_RUNTIME
        runtime = ACTIVE_RUNTIME.get()
        if runtime is None:
            from kernel_request_kind import RequestKindError
            raise RequestKindError("primary_runtime_missing")
        runtime.declared_environment = dict(spec["environment"])
        runtime.environment_sources = dict(spec.get("environment_sources", {}))
        runtime.served_model = _served_model_identity(spec)
        from kernel_model_policy import journal_effective_policy
        journal_effective_policy(runtime)
        capture = getattr(runtime, "_active_capture", None)

        def bound_gate(record, capture, runtime):
            from capability_request_preflight import preflight_presentation_or_degrade, PRECOMPOSED_HOST_TEXT_SCHEMA
            from kernel_request_seal import RequestSealError
            messages = record.body["messages"]
            tools = record.body.get("tools", [])
            precomposed_host_text = {
                **dict(getattr(capture, "release_precomposed_host_text", {}) or {}),
                "request_id": capture.request_id,
                "body_digest": record.body_digest,
                "allocation_digest": record.allocation.digest,
                "schema": PRECOMPOSED_HOST_TEXT_SCHEMA,
            }
            messages, presentation = preflight_presentation_or_degrade(
                messages, spec, runtime, tools=tools,
                frame_request_allocation=capture.frame_request_allocation,
                precomposed_host_text=(
                    precomposed_host_text
                    if isinstance(precomposed_host_text, Mapping) else None
                ),
            )
            presentation["request_id"] = getattr(runtime, "current_request_id", None)
            presentation["provider_message_sha256"] = hashlib.sha256(
                json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
                .encode("utf-8")
            ).hexdigest()
            if capacity_plan.enabled:
                presentation = {**presentation, "host_capacity": capacity_plan.receipt}
            receipt = run / "scope-preflight.json"
            if not receipt.exists():
                save(receipt, presentation)
            with (run / "presentation-preflight.jsonl").open("a") as audit:
                audit.write(json.dumps(presentation) + "\n")
            if messages != record.body["messages"]:
                raise RequestSealError("release preflight mutated the sealed request")

        if capture is None:
            from kernel_request_kind import RequestKindError
            raise RequestKindError("primary_capture_missing")
        from kernel_request_kind import require_adapter
        primary = require_adapter(
            capture, runtime, messages, gate=bound_gate,
            host_preparer=prepare_request_host_text, adapter="release",
        )
        if not primary:
            from capability_request_preflight import preflight_presentation_or_degrade
            messages, _presentation = preflight_presentation_or_degrade(
                messages, spec, runtime, tools=tools,
                frame_request_allocation=getattr(capture, "frame_request_allocation", None),
            )
        from kernel_release_host_prefix import bind_release_prefix_boundary
        bind_release_prefix_boundary(capture, messages, tools)
        return original_ask(messages, **kwargs)

    # The runner consumes capacity inputs before runtime.prepare. Final host-text
    # composition is handed to FrameFit5 preparation so allocation captures it.
    def prepare_enclosing_host_inputs(messages):
        prepare_host_inputs = getattr(original_ask, "prepare_request_inputs", None)
        if callable(prepare_host_inputs):
            messages = prepare_host_inputs(messages)
            if not isinstance(messages, list) or any(
                    not isinstance(message, Mapping) for message in messages):
                raise ValueError("host request input preparation returned invalid messages")
        return messages

    def prepare_request_inputs(messages):
        # Enclosing inventory is a generated provider contribution, not an
        # original conversation system prompt. Collect it on the projected
        # draft below, after semantic extraction and before the shared fit.
        return prepare_request(messages, capacity_plan)

    def prepare_request_host_text(
            messages, *, runtime, tools, frame_request_allocation=None,
            framefit5_inputs=None):
        from capability_request_preflight import prepare_release_host_text
        # Runtime projection may replace incoming system instructions. Restore
        # enclosing-host inputs on the projected candidate before allocation,
        # using the same idempotent input hook as the runner's initial prepare.
        messages = prepare_enclosing_host_inputs(messages)
        messages = prepare_request(messages, capacity_plan)
        return prepare_release_host_text(
            messages, runtime, tools=tools,
            frame_request_allocation=frame_request_allocation,
            framefit5_inputs=framefit5_inputs,
        )

    checked_ask.prepare_request_inputs = prepare_request_inputs
    checked_ask.prepare_request_host_text = prepare_request_host_text
    return checked_ask


def child(run: Path, *, resume: bool = False, follow_up: str | None = None) -> None:
    spec = json.loads((run / "spec.json").read_text())
    if not isinstance(spec, Mapping):
        raise ValueError("run spec must be an object")
    _install_declared_environment(spec)
    _bind_endpoint_size(run, resume=resume)
    fence = activate_release(ROOT, SOURCE)
    if "CAPABILITY_ATTEMPT_DEADLINE_MONOTONIC" in spec["environment"]:
        raise ValueError("spec may not override supervisor deadline")
    if spec["public_row"].get("repo") != "build":
        raise ValueError("runtime release supports synthetic build tasks only")
    engine = agent_loop_selection(spec)
    start_name = "resume-start-" + uuid.uuid4().hex + ".json" if resume else "release-start.json"
    save(run / start_name, {"release": str(ROOT), "imports": fence.snapshot(), "resume": resume,
                            **({"agent_loop": engine} if engine == "plain" else {})})
    if engine == "plain":
        # Dispatch BEFORE react_repair, true-resume or the runtime request wrapper: the full
        # engine below stays byte-for-byte untouched for every run that does not opt in.
        _child_plain(run, spec, fence, resume=resume, follow_up=follow_up)
        return
    requested_true_resume = resume or str(
        spec["environment"].get("STATEFRAME_TRUE_RESUME", os.environ.get("STATEFRAME_TRUE_RESUME", "off"))
    ).strip().lower() == "on"
    if requested_true_resume:
        os.environ["STATEFRAME_TRUE_RESUME"] = "on"
        from kernel_true_resume import activate
        activate()
    sys.path.insert(0, str(ROOT))
    sys.argv[:] = [sys.argv[0]]
    from probes import react_repair as rr
    from capability_study_support import install_test_serialization
    from capability_study_worker import unscored_solver_result
    capacity_plan = _capacity_plan(spec)

    output = run / "worker"
    output.mkdir()
    (output / "results").mkdir()
    (output / "scratch").mkdir()
    from kernel_protocol_spec import publish_public_spec
    publish_public_spec(output / "protocol-spec.json")
    rr.POC = output
    rr.OUT = output / "rows.jsonl"
    from kernel_step_limits import parse_step_limit
    rr.MAX_STEPS = parse_step_limit(spec["steps"])
    tempfile.tempdir = str(output / "scratch")
    test_lock = Path(os.environ.get("CAPABILITY_TEST_LOCK", "/work/study-pytest.lock"))
    install_test_serialization(rr, test_lock, run / "test-lock.jsonl")
    if capacity_plan.enabled:
        save(run / "worker-host-capacity.json", capacity_plan.receipt)
        from capability_host_capacity import install_audit_admission
        install_audit_admission(rr, capacity_plan)
    from capability_capacity_preflight import install as install_capacity
    from kernel_measurement import RequestCapture
    install_capacity(spec, RequestCapture)
    original_ask = rr.ask_tools

    rr.ask_tools = _make_checked_ask(
        original_ask, run=run, spec=spec, fence=fence,
        capacity_plan=capacity_plan,
    )
    from kernel_aha11_native_honesty import build_honest_release_receipt
    row = _public_solver_row_from_spec(spec)
    # ``spec.evaluator`` is host-only.  Copying its location into this row
    # makes reference-fix symbols an oracle for the solver's first action and
    # makes a public module look localized before any public evidence exists.
    # Public declared paths remain in ``files``/``allowed_product_paths``.
    parent_source = str(row.get("parent_src", ""))
    parent_revision = "sha256:" + hashlib.sha256(parent_source.encode("utf-8")).hexdigest()
    save(
        run / ("honest-release-input-" + uuid.uuid4().hex + ".json"),
        build_honest_release_receipt(
            row,
            source_revision=parent_revision,
            candidate_revision="unobserved",
            parent_status="UNCOVERED",
            candidate_status="UNCOVERED",
            terminal_status="INCOMPLETE",
            limitations=("candidate execution has not started",),
        ),
    )
    _refuse_unresolvable_target(row)
    terminal_smoke = os.environ.get("LAUNCH_SMOKE_TERMINAL", "0") == "1"
    if terminal_smoke:
        # The smoke supervisor uses a root-level sitecustomize hook for provider
        # wiring. Do not leak that hook into the fixture's real test subprocess.
        os.environ["PYTHONPATH"] = str(ROOT)
    row.update(_skip_db=True, _bridge_ordinary_brief=row["msg"],
               _bridge_capture=(None if terminal_smoke else str(output / "candidates")),
               information_policy={"information_policy_version": "information-policy/1",
                   "initial_oracle_symptom": "ON" if terminal_smoke else "OFF",
                   "live_grader_feedback": "DIAGNOSTIC" if terminal_smoke else "FINAL_ONLY",
                         "target_disclosure": "PUBLIC_PATHS_UNLOCALIZED"})
    descriptor = _resume_descriptor(run, follow_up) if resume else None
    result = rr.work(row, resume=descriptor)
    _postflight_effective_config(run, spec)
    outcome_name = "resume-outcome-" + uuid.uuid4().hex + ".json" if resume else "first-outcome.json"
    save(run / outcome_name, unscored_solver_result(result))
    result_mapping = result if isinstance(result, Mapping) else {}
    parent_status = str(result_mapping.get("parent_status", "UNCOVERED"))
    candidate_status = str(result_mapping.get("candidate_status", "UNCOVERED"))
    terminal_status = str(result_mapping.get("result", result_mapping.get("outcome", "INCOMPLETE")))
    limitations = () if parent_status != "UNCOVERED" and candidate_status != "UNCOVERED" else (
        "terminal result did not provide parent/candidate observations",
    )
    save(
        run / ("honest-release-receipt-" + uuid.uuid4().hex + ".json"),
        build_honest_release_receipt(
            row,
            source_revision=parent_revision,
            candidate_revision=str(result_mapping.get("candidate_revision", "unobserved")),
            parent_status=parent_status,
            candidate_status=candidate_status,
            terminal_status=terminal_status,
            clause_ids=result_mapping.get("clause_ids", ()),
            limitations=limitations,
        ),
    )
    verify_release(ROOT)
    end_name = "resume-end-" + uuid.uuid4().hex + ".json" if resume else "release-end.json"
    save(run / end_name, {"release": str(ROOT), "imports": fence.snapshot(), "resume": resume})


# Attempt wall-clock deadline (both arms). Was 2400s: correct JobQueue candidates were reached
# and then cut off by the deadline twice on 09-24 (131 hjqon3-924082, 132 stk6-924102) under
# uncapped reasoning + 4 concurrent runs per llama slot group. Timed-out runs are SCORED on their final grade
# (CORRECT_UNFINISHED when the code is correct); only harness crashes are excluded (timeouts.sh).
ATTEMPT_DEADLINE_SECONDS = 14400  # Allow up to four hours for a verified repair.


def supervise(run: Path, *, resume: bool = False, follow_up: str | None = None) -> None:
    frozen = source_hashes(run)
    save(run / "source-freeze.json", frozen)
    process = None
    timed_out = False
    capacity_refusal = None
    admission = None
    attempt = "resume-" + uuid.uuid4().hex if resume else "first"
    with (run / (attempt + ".log")).open("x") as log:
        from protected_tree_guard import guarded_command
        child_args = [str(Path(__file__)), str(run), "child"]
        if resume:
            child_args.extend(["--resume", *( [follow_up] if follow_up is not None else [] )])
        command = guarded_command([sys.executable, "-B", "protected_tree_guard.py", "--inside",
            *child_args],
            roots=[ROOT, SOURCE / "results", SOURCE / "results.SNAPSHOT-20260920"])
        spec = json.loads((run / "spec.json").read_text())
        if "CAPABILITY_ATTEMPT_DEADLINE_MONOTONIC" in spec["environment"]:
            raise ValueError("spec may not override supervisor deadline")
        class _CapacityRefusalSentinel(Exception):
            pass

        capacity_refusal_type = _CapacityRefusalSentinel
        capacity_plan = _capacity_plan(spec)
        if capacity_plan.enabled:
            from capability_host_capacity import (
                CapacityRefusal, captured_process_group, terminate_process_group,
            )
            capacity_refusal_type = CapacityRefusal
        if capacity_plan.enabled:
            save(run / "admission-host-capacity.json", capacity_plan.receipt)
        deadline_enabled = spec["environment"].get("KERNEL_PUBLIC_REVIEW_DEADLINE", "0") == "1"
        attempt_deadline = time.monotonic() + ATTEMPT_DEADLINE_SECONDS
        child_environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
        child_environment.pop("CAPABILITY_ATTEMPT_DEADLINE_MONOTONIC", None)
        if deadline_enabled:
            child_environment["CAPABILITY_ATTEMPT_DEADLINE_MONOTONIC"] = str(attempt_deadline)
        try:
            if capacity_plan.enabled:
                admission_timeout = (max(0, attempt_deadline - time.monotonic())
                                      if deadline_enabled else ATTEMPT_DEADLINE_SECONDS)
                admission = capacity_plan.acquire_worker(timeout_s=admission_timeout)
            process = subprocess.Popen(command, cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env=child_environment)
            process_group = (captured_process_group(process)
                             if capacity_plan.enabled else None)
            save(run / (attempt + "-process.json"), {"pid": process.pid, "started": time.time()})
            try:
                process.wait(timeout=max(0, attempt_deadline - time.monotonic()) if deadline_enabled else ATTEMPT_DEADLINE_SECONDS)
            except subprocess.TimeoutExpired:
                timed_out = True
                if capacity_plan.enabled:
                    terminate_process_group(process, pgid=process_group)
                else:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
        except capacity_refusal_type as exc:
            capacity_refusal = str(exc)
        finally:
            if admission is not None:
                admission.release()
    save(run / (attempt + "-completion.json"), {
        "outcomes": [{"stage": "resume" if resume else "first", "exit_code": process.returncode if process is not None else None,
                      "timed_out": timed_out, **({"infrastructure": "capacity_refused",
                                                   "reason": capacity_refusal}
                                                  if capacity_refusal is not None else {})}],
        "source_drift": [name for name, sha in frozen.items()
                         if not Path(name).is_file()
                         or hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha],
        "scope": "Developmental native/frame diagnostic; private and posthoc grading pending"})


if __name__ == "__main__":
    if len(sys.argv) in {3, 4} and sys.argv[1] == "--resume":
        run = Path(sys.argv[2]).resolve(strict=True)
        if run.is_relative_to(ROOT):
            raise SystemExit("Experiment outputs must be outside the immutable release")
        supervise(run, resume=True, follow_up=sys.argv[3] if len(sys.argv) == 4 else None)
        raise SystemExit(0)
    if len(sys.argv) in {4, 5} and sys.argv[2:4] == ["child", "--resume"]:
        run = Path(sys.argv[1]).resolve(strict=True)
        if run.is_relative_to(ROOT):
            raise SystemExit("Experiment outputs must be outside the immutable release")
        child(run, resume=True, follow_up=sys.argv[4] if len(sys.argv) == 5 else None)
        raise SystemExit(0)
    if len(sys.argv) not in {2, 3}:
        raise SystemExit("Expected absolute run directory and optional child")
    run = Path(sys.argv[1]).resolve(strict=True)
    if run.is_relative_to(ROOT):
        raise SystemExit("Experiment outputs must be outside the immutable release")
    if len(sys.argv) == 3:
        if sys.argv[2] != "child":
            raise SystemExit("Invalid stage")
        child(run)
    else:
        supervise(run)
