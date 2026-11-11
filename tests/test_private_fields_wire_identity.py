"""Private non-wire state must never reach a model-facing id or digest.

Structural rule: changing only host environment identity in a real engine run
must leave every provider request unchanged (except the clock countdown).
To add a case, extend the recorded replay so the new identity site is exercised,
assert its marker is present in the captured wire, and keep the planted leak
control sensitive to the new path.

Covered id-sites: check execution/check/test receipt IDs, check evidence refs and
their rendered wire fact, unified-stop snapshot ID and rendered authority block,
request/content/evidence wire IDs, and action receipt wire ID. Add a site by
returning its actual production output from a case-output helper and running the shared perturbation
loop over the private inputs that can reach that site. Keep a planted leak control.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
import re
import subprocess
import textwrap
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import kernel_check_receipts as check_receipts
import kernel_completion_policy as completion_policy
import kernel_countdown_slot
import kernel_exec_session as exec_session
import kernel_exec_supervisor as exec_supervisor
import kernel_execution_policy as execution_policy
import kernel_stateframe_runtime as stateframe_runtime
import kernel_verdict
import kernel_wire_identity
import kernel_wire_truth_facts
import pytest
from kernel_measurement import canonical_bytes

from tests._journal import measurement_events
from tests import test_allocation_replay_corpus as corpus

SELECTOR = "tests/test_private_fields_wire_identity.py"
BINDING = {"product_revision": "sha256:stable-product"}
_ABSENT = object()


def _structural_bootstrap(identity: str, *, planted: bool) -> str:
    lines = [
        "original_install_run_identity = host.install_run_identity",
        "def install_with_environment(seed):",
        "    original_install_run_identity(seed)",
        "    import kernel_shell_tool",
        f"    kernel_shell_tool.ShellTool.identity = lambda self: {identity!r}",
    ]
    if planted:
        lines.extend((
            "    import kernel_wire_identity",
            "    kernel_wire_identity.WIRE_ENV_IDENTITY_FIELDS = frozenset()",
        ))
    lines.append("host.install_run_identity = install_with_environment")
    return "\n".join(lines)


def _structural_request_bodies(run_dir: Path) -> tuple[bytes, ...]:
    wire_dir = run_dir / "wire"
    paths = sorted(wire_dir.glob("*.request.json"))
    assert len(paths) == 3, f"expected three provider sends, found {len(paths)}"
    return tuple(path.read_bytes() for path in paths)


def _structural_environment_revision(run_dir: Path, expected: str) -> str:
    def values(value: object):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "environment_revision" and isinstance(item, str):
                    yield item
                yield from values(item)
        elif isinstance(value, list):
            for item in value:
                yield from values(item)

    revisions = set(values(measurement_events(run_dir)))
    assert expected in revisions, f"bound revision absent from journal: {revisions}"
    return expected


def _structural_intention_ids(run_dir: Path) -> tuple[str, ...]:
    # The short replay journals its live intention even when the current gear
    # omits the continuation block from the provider frame.
    ids = set(re.findall(r"intention:[0-9a-f]{20}",
                         json.dumps(measurement_events(run_dir))))
    assert ids, "real replay did not create a continuation intention"
    return tuple(sorted(ids))


def _structural_check_execution_ids(run_dir: Path) -> tuple[str, ...]:
    ids = {
        identity
        for row in measurement_events(run_dir)
        for identity in row.get("canonical_check_execution_ids", ())
        if isinstance(identity, str) and identity.startswith("check-execution:")
    }
    assert ids, "real replay did not execute a check"
    return tuple(sorted(ids))


def _structural_wire_ids(bodies: tuple[bytes, ...]) -> tuple[tuple[str, ...], ...]:
    # The full-body comparison below is the authority. This projection makes
    # model-facing identity drift explicit in a failure, including embedded facts.
    pattern = re.compile(
        r'(?:unified|wire-truth|intention|check):[0-9a-f]{20,64}'
        r'|native_[0-9a-f]{20}(?:_[0-9]+)?'
    )
    ids = tuple(tuple(pattern.findall(body.decode("utf-8"))) for body in bodies)
    contents = (
        message["content"]
        for body in bodies
        for message in json.loads(body)["messages"]
        if isinstance(message.get("content"), str)
    )
    joined = "\n".join(contents)
    assert re.search(
        r"\[WIRE_TRUTH_FACT [^\]]+\] check_execution_ref: "
        r"[0-9a-f]{20}(?::[0-9]+)?(?:,[0-9a-f]{20}(?::[0-9]+)?)*",
        joined,
    ), "replay did not produce a model-facing check execution ref"
    assert "RECEIPTS" in joined
    assert re.search(r"\[Unified stop authority\]\nsnapshot=unified:[0-9a-f]{20,64}", joined)
    assert any(value.startswith("unified:") for row in ids for value in row)
    assert any(value.startswith("wire-truth:") for row in ids for value in row)
    return ids


@pytest.mark.host_sandbox
def test_structural_host_environment_identity_never_reaches_model(
        tmp_path: Path) -> None:

    def run(name: str, identity: str, *, planted: bool = False):
        capture_dir = tmp_path / name
        capture_dir.mkdir()
        outcome = corpus._run_case(
            capture_dir, "clean-meter-22file-r1",
            stop_after_provider_requests=3,
            extra_bootstrap=_structural_bootstrap(identity, planted=planted),
        )
        run_dir = capture_dir / "run"
        assert outcome.endpoint_requests == 3, outcome.evidence[-4000:]
        assert (run_dir / "capture-limit-reached").read_text(encoding="ascii") == "3"
        bodies = _structural_request_bodies(run_dir)
        revision = _structural_environment_revision(run_dir, identity)
        return (bodies, revision, _structural_intention_ids(run_dir),
                _structural_check_execution_ids(run_dir))

    with ThreadPoolExecutor(max_workers=3) as captures:
        first_future = captures.submit(run, "environment-a", "host-environment-a")
        second_future = captures.submit(run, "environment-b", "host-environment-b")
        planted_future = captures.submit(
            run, "environment-planted", "host-environment-b", planted=True)
        first, first_revision, first_intentions, first_checks = first_future.result()
        second, second_revision, second_intentions, second_checks = second_future.result()
        planted, planted_revision, _, _ = planted_future.result()
    assert first_revision != second_revision
    assert first_intentions == second_intentions
    assert first_checks == second_checks
    assert len(first) == len(second)
    assert _structural_wire_ids(first) == _structural_wire_ids(second)
    def canonical(body: bytes) -> object:
        return kernel_countdown_slot.canonical_countdown(json.loads(body))
    canonical_first = tuple(map(canonical, first))
    canonical_second = tuple(map(canonical, second))
    assert canonical_first == canonical_second

    def countdown_slots(body: object) -> list[str]:
        return re.findall(
            r"sandbox_seconds_remaining=([^;]+);",
            json.dumps(body, sort_keys=True, ensure_ascii=False),
        )

    for capture_name, capture in (
        ("environment-a", canonical_first),
        ("environment-b", canonical_second),
        ("environment-planted", tuple(map(canonical, planted))),
    ):
        for body in capture:
            assert all(
                slot == kernel_countdown_slot.CANONICAL_TIME_SLOT
                for slot in countdown_slots(body)
            ), f"{capture_name} request retained a live countdown slot"

    assert planted_revision == second_revision
    assert len(planted) == len(first)
    _structural_wire_ids(planted)
    canonical_planted = tuple(map(canonical, planted))
    assert canonical_planted != canonical_first, (
        "planted environment leak did not change provider requests")


def _field_is_private_or_audit(name: str, metadata: object = None,
                               documentation: object = None) -> bool:
    normalized = name.lower()
    if name.startswith("_") or any(token in normalized for token in (
            "private", "audit", "non_wire", "memory_limit", "cgroup",
            "fallback_reason", "kill_reason")):
        return True
    if isinstance(metadata, Mapping):
        if metadata.get("wire") is False or metadata.get("public") is False:
            return True
        if any(bool(metadata.get(key)) for key in ("private", "audit", "non_wire")):
            return True
        documentation = documentation or metadata.get("doc") or metadata.get("description")
    text = str(documentation or "").lower()
    return any(token in text for token in ("private", "audit", "non-wire", "non wire"))


def _dataclass_record_types() -> tuple[type, ...]:
    modules = (check_receipts, exec_session, execution_policy, kernel_verdict,
               stateframe_runtime)
    found: list[type] = []
    for module in modules:
        for candidate in vars(module).values():
            if (inspect.isclass(candidate) and dataclasses.is_dataclass(candidate)
                    and candidate.__module__ == module.__name__):
                name = candidate.__name__.lower()
                if any(token in name for token in ("check", "receipt", "execution", "policy", "result")):
                    found.append(candidate)
    return tuple(dict.fromkeys(found))


def _literal_mapping_keys(function: object) -> set[str]:
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    except (OSError, TypeError, IndentationError, SyntaxError):
        return set()
    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys.update(key.value for key in node.keys
                        if isinstance(key, ast.Constant) and isinstance(key.value, str))
        elif (isinstance(node, ast.Subscript)
              and isinstance(node.value, ast.Name) and node.value.id == "private"
              and isinstance(node.slice, ast.Constant)
              and isinstance(node.slice.value, str)):
            keys.add(node.slice.value)
    return keys


def _private_instance_fields(record_type: type) -> set[str]:
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(record_type)))
    except (OSError, TypeError, IndentationError, SyntaxError):
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign) else [])
        for target in targets:
            for item in ast.walk(target):
                if (isinstance(item, ast.Attribute) and isinstance(item.value, ast.Name)
                        and item.value.id == "self"
                        and _field_is_private_or_audit(item.attr)):
                    names.add(item.attr)
    return names


def _emitted_fields(function: object, event_kind: str) -> set[str]:
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    except (OSError, TypeError, IndentationError, SyntaxError):
        return set()
    fields: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "emit" or not node.args:
            continue
        first = node.args[0]
        if not isinstance(first, ast.Constant) or first.value != event_kind:
            continue
        fields.update(keyword.arg for keyword in node.keywords if keyword.arg)
        for keyword in node.keywords:
            if keyword.arg is not None:
                continue
            fields.update(_dict_keys(keyword.value))
    return fields


def _dict_keys(node: ast.AST) -> set[str]:
    keys: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Dict):
            keys.update(key.value for key in child.keys
                        if isinstance(key, ast.Constant) and isinstance(key.value, str))
    return keys


def _private_field_inventory() -> tuple[set[str], dict[str, dataclasses.Field]]:
    names: set[str] = set()
    definitions: dict[str, dataclasses.Field] = {}
    for record_type in _dataclass_record_types():
        for field in dataclasses.fields(record_type):
            definitions.setdefault(field.name, field)
            if _field_is_private_or_audit(
                    field.name, field.metadata, getattr(field, "doc", None)):
                names.add(field.name)
    names.update(_private_instance_fields(exec_supervisor.ManagedProcess))

    builders = (
        check_receipts.record_check_output,
        stateframe_runtime.StateFrameRuntime.test_observation,
        stateframe_runtime.StateFrameRuntime._record_test_completion_receipt,
    )
    for builder in builders:
        try:
            names.update(name for name in inspect.signature(builder).parameters
                         if _field_is_private_or_audit(name))
        except (TypeError, ValueError):
            pass
        names.update(name for name in _literal_mapping_keys(builder)
                     if _field_is_private_or_audit(name))
    return names, definitions


def _values_for(field_name: str, definition: dataclasses.Field | None) -> tuple[object, ...]:
    if field_name == "memory_limit_mechanism":
        return "rlimit_data+rss_supervisor", "cgroup_memory_max"
    if field_name == "memory_limit_killed":
        return False, True
    if field_name == "memory_limit_reason":
        return ("aggregate resident-memory limit exceeded",
                "private sample threshold exceeded")
    if field_name == "cgroup_leaf_pattern":
        return "applied", "failed:foreign_process"
    if field_name == "_stateframe_memo_generation":
        return 2, 19
    if field_name == "_receipt_identity_content":
        return {"private": "first"}, {"private": "second"}
    if field_name == "cgroup":
        return Path("/sys/fs/cgroup/stateframe-a"), Path("/sys/fs/cgroup/stateframe-b")
    if "path" in field_name:
        return "/sys/fs/cgroup/stateframe-a", "/sys/fs/cgroup/stateframe-b"
    if "fallback" in field_name or "reason" in field_name:
        return "fallback-a", "fallback-b"
    annotation = getattr(definition, "type", None)
    if annotation is bool:
        return False, True
    if annotation is int:
        return 2, 19
    return f"private-{field_name}-a", f"private-{field_name}-b"


def _base_dataclass_records() -> tuple[object, ...]:
    cleanup = exec_session.CleanupReport(True, 0, (), "")
    exec_result = exec_session.ExecResult(0, False, b"", False, 0, 0.0, cleanup)
    capabilities = execution_policy.CapabilityReport(
        "Linux", "bwrap", True, True, True, True, True, True)
    policy_result = execution_policy.PolicyResult(
        completed=subprocess.CompletedProcess(["true"], 0, "", ""),
        timed_out=False, process_group_killed=False, output_truncated=False,
        capabilities=capabilities,
    )
    return exec_result, policy_result


def _typed_value(field_name: str, value: object) -> object:
    if value is _ABSENT:
        return value
    for record in _base_dataclass_records():
        field_names = {field.name for field in dataclasses.fields(record)}
        if field_name in field_names:
            record = dataclasses.replace(record, **{field_name: value})
            return getattr(record, field_name)
    return value


def _supervisor_status(cgroup: object = None) -> dict[str, object]:
    """Project process status, which must not serialize the private cgroup path."""
    process = object.__new__(exec_supervisor.ManagedProcess)
    process.exit_code = 0
    process.cleanup = None
    process.timed_out = False
    process.collector = SimpleNamespace(total_bytes=0)
    process.cursor = 0
    process.started = 1.0
    process.ended = 1.0
    process.memory_limit_mechanism = None
    process.cgroup = cgroup
    return process.status()


def _case_outputs(
        field_name: str, value: object, monkeypatch: pytest.MonkeyPatch, *,
        completion_bindings: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    value = _typed_value(field_name, value)

    def emit(kind: str, **fields: object) -> dict[str, object]:
        return {"kind": kind, **fields}

    journal = SimpleNamespace(
        task_id="private-wire-task", run_id="a" * 32,
        check_receipt_count=0, check_execution_allocations=0, emit=emit,
    )
    owner = SimpleNamespace(current_step=7, f2p_ids=[SELECTOR], _receipts=[])
    runtime = SimpleNamespace(
        owner=owner, revision=BINDING["product_revision"], journal=journal,
        test_receipts=[], host_check_receipts=[], _check_output_receipts={},
        micrograph_wire=None,
    )
    runtime.binding = lambda: dict(
        completion_bindings.get("binding", BINDING) if completion_bindings else BINDING)
    runtime._test_invalidation_receipt_fields = lambda _event, _owner, _runtime: {}

    check_kwargs: dict[str, object] = {}
    check_builder_params = inspect.signature(check_receipts.record_check_output).parameters
    if field_name in check_builder_params and value is not _ABSENT:
        check_kwargs[field_name] = value
    receipt = check_receipts.record_check_output(
        runtime, selector=SELECTOR, status="PASS", outcome="PASS", sequence=7,
        revision=BINDING["product_revision"], exit_code=0, stdout="passed\n",
        execution_complete=True, **check_kwargs,
    )
    private_receipt = runtime._check_output_receipts[receipt["receipt_id"]]
    if field_name == "_stateframe_memo_generation":
        if value is _ABSENT:
            private_receipt.pop(field_name, None)
        else:
            private_receipt[field_name] = value
    public_receipt = check_receipts._public_receipt(private_receipt, journal)
    runtime.host_check_receipts[:] = [public_receipt]

    completion_bindings = completion_bindings or {}
    test_event: dict[str, object] = {
        "kind": "test_subprocess_completed", "producer": "test_subprocess",
        "selectors": [SELECTOR], "selector_scope": [SELECTOR], "exit_code": 0,
        "status": "PASS", "outcome": "PASS", "revision": BINDING["product_revision"],
        "binding": dict(completion_bindings.get("binding", BINDING)),
        "after_binding": dict(completion_bindings.get("after_binding", BINDING)),
        "binding_stable": True, "sequence": 7, "likely_initiator": None,
    }
    test_event_fields = _emitted_fields(
        stateframe_runtime.StateFrameRuntime.test_observation,
        "test_subprocess_completed",
    )
    if field_name in test_event_fields or field_name == "memory_limit_mechanism":
        if field_name == "memory_limit_killed":
            if value is True:
                test_event.update({
                    "memory_limit_killed": True,
                    "memory_limit_reason": "aggregate resident-memory limit exceeded",
                    "likely_initiator": "aggregate_rss_supervisor",
                })
        elif field_name == "memory_limit_reason":
            if value is not _ABSENT:
                test_event.update({
                    "memory_limit_killed": True,
                    "memory_limit_reason": value,
                    "likely_initiator": "aggregate_rss_supervisor",
                })
        elif value is not _ABSENT:
            test_event[field_name] = value

    import kernel_readiness_authority
    monkeypatch.setattr(kernel_readiness_authority, "runtime_enabled", lambda _runtime: False)
    stateframe_runtime.StateFrameRuntime._record_test_completion_receipt(
        runtime, test_event, (public_receipt,))
    if field_name == "_receipt_identity_content":
        if value is _ABSENT:
            runtime.test_receipts[-1].pop(field_name, None)
        else:
            runtime.test_receipts[-1][field_name] = value

    evidence = kernel_verdict._event_ref_material(runtime.test_receipts[-1])
    check_binding = completion_bindings.get("binding", BINDING)
    check = kernel_wire_truth_facts._current_public_check(runtime, check_binding)
    check_ref = check[1]
    rendered = kernel_wire_truth_facts.render_fact_messages(
        {"check_execution_ref": check_ref})
    return {
        "check_receipt_id": receipt["receipt_id"],
        "check_execution_id": receipt["execution_id"],
        "test_receipt_id": runtime.test_receipts[-1]["receipt_id"],
        "public_receipt": public_receipt,
        "public_evidence": evidence,
        "dependency_identity": hashlib.sha256(canonical_bytes(evidence)).hexdigest(),
        "check_execution_ref": check_ref,
        "rendered_wire_fact": rendered,
        "supervisor_status": _supervisor_status(
            value if field_name == "cgroup" else None),
    }


def _field_leaks(
        fields: set[str], case_outputs: Callable[[str, object], dict[str, object]],
        values_for: Callable[[str], tuple[object, ...]],
) -> dict[str, list[str]]:
    leaks: dict[str, list[str]] = {}
    for field_name in sorted(fields):
        baseline = case_outputs(field_name, _ABSENT)
        baseline_bytes = {name: canonical_bytes(value) for name, value in baseline.items()}
        changed_outputs: set[str] = set()
        for value in values_for(field_name):
            variant = case_outputs(field_name, value)
            changed_outputs.update(
                name for name, item in variant.items()
                if canonical_bytes(item) != baseline_bytes[name]
            )
        if changed_outputs:
            leaks[field_name] = sorted(changed_outputs)
    return leaks


def _private_field_leaks(fields: set[str], definitions: dict[str, dataclasses.Field],
                         monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    return _field_leaks(
        fields, lambda field, value: _case_outputs(field, value, monkeypatch),
        lambda field: _values_for(field, definitions.get(field)),
    )


def _assert_no_private_wire_leaks(
        fields: set[str], definitions: dict[str, dataclasses.Field],
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaks = _private_field_leaks(fields, definitions, monkeypatch)
    assert not leaks, "private fields changed model-facing bytes: " + "; ".join(
        f"{field} -> {','.join(outputs)}" for field, outputs in sorted(leaks.items()))


def test_private_fields_do_not_change_check_wire_identity(monkeypatch):
    fields, definitions = _private_field_inventory()
    # The diff stores the cgroup path as ManagedProcess.cgroup. Leaf-pattern
    # failure status carries the fallback reason, with no separate reason field.
    required_new_fields = {
        "memory_limit_mechanism", "memory_limit_killed",
        "memory_limit_reason", "cgroup_leaf_pattern", "cgroup",
    }
    assert required_new_fields <= fields, (
        "private field inventory missed new execution fields: "
        + ", ".join(sorted(required_new_fields - fields)))
    assert set(kernel_wire_identity.WIRE_AUDIT_ONLY_FIELDS) == {
        "memory_limit_mechanism", "memory_limit_killed",
        "memory_limit_reason", "cgroup_leaf_pattern",
        "memory_limit_samples_skipped", "memory_limit_sample_skip_reason",
        "memory_limit_stop_cause",
    }
    assert set(kernel_wire_identity.WIRE_AUDIT_ONLY_FIELDS) <= fields
    assert {"_stateframe_memo_generation", "_receipt_identity_content"} <= fields
    # This value is producer-computed from the receipt's intentional identity
    # inputs and drives duplicate-receipt ordinals. Mutating it alone fabricates
    # a receipt state production never creates, so do not perturb it independently.
    _assert_no_private_wire_leaks(
        fields - {"_receipt_identity_content"}, definitions, monkeypatch)


def test_checker_detects_a_planted_private_digest_input(monkeypatch):
    _fields, definitions = _private_field_inventory()
    original = kernel_verdict._event_ref_material

    def safe_private_projection(row):
        material = original(row)
        material.pop("memory_limit_mechanism", None)
        return material

    monkeypatch.setattr(kernel_verdict, "_event_ref_material", safe_private_projection)
    assert not _private_field_leaks(
        {"memory_limit_mechanism"}, definitions, monkeypatch)

    def planted_private_input(row):
        material = safe_private_projection(row)
        if "memory_limit_mechanism" in row:
            material["memory_limit_mechanism"] = row["memory_limit_mechanism"]
        return material

    monkeypatch.setattr(kernel_verdict, "_event_ref_material", planted_private_input)
    leaks = _private_field_leaks(
        {"memory_limit_mechanism"}, definitions, monkeypatch)
    assert "check_execution_ref" in leaks.get("memory_limit_mechanism", [])
    with pytest.raises(AssertionError, match="memory_limit_mechanism"):
        _assert_no_private_wire_leaks(
            {"memory_limit_mechanism"}, definitions, monkeypatch)


_HOST_BINDING_FIELDS = ("env_identity", "package_versions", "host_paths", "timings")
_HOST_BINDING_VALUES = {
    "env_identity": "tool-cache-a",
    "package_versions": {"pytest": "8.0", "python": "3.14"},
    "host_paths": ["/host/a/site-packages", "/host/a/tool-cache"],
    "timings": {"prepare_ms": 12, "test_ms": 27},
}


def _unified_stop_outputs(field_name: str, value: object) -> dict[str, object]:
    binding: dict[str, object] = {
        **BINDING, "artifact": "sha256:stable-product",
        "environment": "host-a", "environment_revision": "env-a",
        "policy_identity": "policy-a", "authorization_scope": "scope-a",
        "tenant_identity": "tenant-a", "test_harness_revision": "harness-a",
    }
    key, _, nested = field_name.partition(".")
    if nested:
        host = dict(_HOST_BINDING_VALUES)
        if value is not _ABSENT:
            host[nested] = value
        binding[key] = host
    elif value is not _ABSENT:
        binding[key] = value
    runtime = SimpleNamespace(binding=lambda: binding)
    clauses = completion_policy.collect_clauses(runtime)
    rendered = completion_policy.render(runtime)
    assert rendered.startswith(completion_policy.RENDER_HEADER)
    return {
        "unified_stop_snapshot_id": completion_policy.snapshot_id(runtime, clauses),
        "rendered_unified_stop_authority": rendered,
    }


def _host_binding_values(field_name: str) -> tuple[object, ...]:
    if "." not in field_name:
        return "host-b", "host-c"
    nested = field_name.partition(".")[2]
    variants: dict[str, tuple[object, ...]] = {
        "env_identity": ("tool-cache-b", "tool-cache-c"),
        "package_versions": ({"pytest": "8.1", "python": "3.14"},
                             {"pytest": "8.0", "python": "3.15"}),
        "host_paths": (["/host/b/site-packages", "/host/a/tool-cache"],
                       ["/host/a/site-packages", "/host/b/tool-cache"]),
        "timings": ({"prepare_ms": 13, "test_ms": 27},
                    {"prepare_ms": 12, "test_ms": 28}),
    }
    return variants[nested]


def _unified_stop_leaks() -> dict[str, list[str]]:
    # Both keys are present in StateFrameRuntime.binding()'s rich projection;
    # the plain projection exposes environment. Nested values exercise every
    # host datum carried by an environment revision, independent of encoding.
    keys = {"environment", "environment_revision"}
    fields = keys | {f"{key}.{part}" for key in keys for part in _HOST_BINDING_FIELDS}
    return _field_leaks(fields, _unified_stop_outputs, _host_binding_values)


def test_private_binding_does_not_change_unified_stop_wire_identity() -> None:
    assert {"environment", "environment_revision"} <= completion_policy.NON_WIRE_ENVIRONMENT_BINDING_KEYS
    assert not _unified_stop_leaks()


def test_unified_stop_guard_detects_planted_environment_leak(monkeypatch) -> None:
    assert not _unified_stop_leaks()
    monkeypatch.setattr(completion_policy, "NON_WIRE_ENVIRONMENT_BINDING_KEYS", frozenset())
    leaks = _unified_stop_leaks()
    assert "unified_stop_snapshot_id" in leaks["environment"]
    assert "rendered_unified_stop_authority" in leaks["environment_revision.env_identity"]


_COMPLETION_ENV_BINDING_FIELDS = (
    "environment",
    "environment_revision",
)


def _check_reference_environment_outputs(
        field_name: str, value: object, monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    bindings: dict[str, dict[str, object]] = {
        name: {
            **BINDING,
            "environment": "host-a",
            "environment_revision": "env-a",
        }
        for name in ("binding", "after_binding")
    }
    if value is not _ABSENT:
        for binding in bindings.values():
            binding[field_name] = value

    outputs = _case_outputs(
        "completion_environment_identity", _ABSENT, monkeypatch,
        completion_bindings=bindings,
    )
    return {
        "check_execution_ref": outputs["check_execution_ref"],
        "rendered_wire_fact": outputs["rendered_wire_fact"],
    }


def _check_reference_environment_values(field_name: str) -> tuple[str, str]:
    if field_name == "environment":
        return "host-b", "host-c"
    if field_name == "environment_revision":
        return "env-b", "env-c"
    raise AssertionError(f"unexpected completion binding field: {field_name}")


def _check_reference_environment_leaks(
        monkeypatch: pytest.MonkeyPatch,
) -> dict[str, list[str]]:
    return _field_leaks(
        set(_COMPLETION_ENV_BINDING_FIELDS),
        lambda field, value: _check_reference_environment_outputs(
            field, value, monkeypatch),
        _check_reference_environment_values,
    )


def test_private_completion_binding_does_not_change_check_wire_identity(monkeypatch):
    assert {"environment", "environment_revision"} <= (
        kernel_wire_identity.WIRE_ENV_IDENTITY_FIELDS)
    assert not _check_reference_environment_leaks(monkeypatch)


def test_check_reference_guard_detects_planted_environment_leak(monkeypatch):
    assert not _check_reference_environment_leaks(monkeypatch)
    monkeypatch.setattr(kernel_wire_identity, "WIRE_ENV_IDENTITY_FIELDS", frozenset())
    leaks = _check_reference_environment_leaks(monkeypatch)
    assert "check_execution_ref" in leaks["environment"]
    assert "rendered_wire_fact" in leaks["environment_revision"]


def test_wire_identity_content_drops_environment_and_audit_identity_recursively():
    content = {
        "binding": {
            "environment": "host-a",
            "environment_revision": "env-a",
            "product_revision": "product-a",
            "memory_limit_mechanism": "rlimit_data+rss_supervisor",
            "memory_limit_killed": False,
            "memory_limit_reason": "private sample threshold exceeded",
            "cgroup_leaf_pattern": "applied",
        },
        "after_binding": [{
            "environment": "host-b",
            "environment_revision": "env-b",
            "product_revision": "product-a",
            "memory_limit_killed": True,
        }],
        "nested": {"deeper": [{
            "environment_revision": "env-c",
            "cgroup_leaf_pattern": "failed:foreign_process",
        }]},
    }

    public_content = kernel_wire_identity.wire_identity_content(content)

    assert public_content == {
        "binding": {"product_revision": "product-a"},
        "after_binding": [{"product_revision": "product-a"}],
        "nested": {"deeper": [{}]},
    }
    changed_product_content = {
        "binding": {
            "environment": "host-a",
            "environment_revision": "env-a",
            "product_revision": "product-b",
        },
        "after_binding": content["after_binding"],
        "nested": content["nested"],
    }
    assert public_content != kernel_wire_identity.wire_identity_content(
        changed_product_content)


def _wire_id_outputs(field_name: str, value: object) -> dict[str, object]:
    journal = SimpleNamespace(task_id="private-wire-task", run_id="host-run-a",
                              timestamp="host-time-a", record_hash="host-hash-a")
    content = {"messages": [{"role": "user", "content": "Check candidate"}],
               "audit": {"timestamp": "host-time-a", "record_hash": "host-hash-a",
                         "journal_id": "host-journal-a", "tool_call_id": "call-a",
                         "request_id": "host-request-a", "receipt_id": "host-receipt-a"}}
    if value is not _ABSENT:
        content["audit"][field_name] = value
        if field_name == "timestamp":
            journal.timestamp = value
        elif field_name == "record_hash":
            journal.record_hash = value
        elif field_name == "journal_id":
            journal.journal_id = value
        elif field_name == "tool_call_id":
            journal.tool_call_id = value
    public_content = kernel_wire_identity.wire_identity_content(content)
    return {
        "content_wire_id": kernel_wire_identity.content_wire_id(
            journal, "request-content", 7, public_content),
        "seeded_content_wire_id": kernel_wire_identity.seeded_content_wire_id(
            "task:private-wire-task", "check-evidence", 7, public_content),
        "request_wire_id": kernel_wire_identity.request_wire_id(
            journal, 7, public_content, purpose="decision"),
        "action_receipt_wire_id": kernel_wire_identity.action_receipt_wire_id(
            journal, request_id="request-a", step=7,
            tool_call_id=content["audit"]["tool_call_id"],
            action_kind="edit", fingerprint="semantic-edit-a",
            target_path="src/app.py"),
    }


def test_private_request_and_action_receipt_metadata_do_not_change_wire_ids() -> None:
    fields = {"timestamp", "record_hash", "journal_id", "tool_call_id",
              "request_id", "receipt_id"}
    assert fields <= kernel_wire_identity._WIRE_IDENTITY_METADATA_KEYS
    assert not _field_leaks(fields, _wire_id_outputs,
                            lambda field: _values_for(field, None))
    journal = SimpleNamespace(task_id="private-wire-task")
    baseline = kernel_wire_identity.request_wire_id(journal, 7, {"content": "a"})
    assert baseline != kernel_wire_identity.request_wire_id(journal, 7, {"content": "b"})
    action = {"request_id": baseline, "step": 7, "tool_call_id": "call-a",
              "action_kind": "edit", "target_path": "src/app.py"}
    assert (kernel_wire_identity.action_receipt_wire_id(
        journal, **action, fingerprint="semantic-edit-a") !=
        kernel_wire_identity.action_receipt_wire_id(
            journal, **action, fingerprint="semantic-edit-b"))

def test_only_rss_kill_initiator_is_normalized_from_wire_identity():
    from kernel_wire_identity import wire_identity_content

    assert wire_identity_content({
        "likely_initiator": "aggregate_rss_supervisor",
    }) == {"likely_initiator": None}
    assert wire_identity_content({
        "likely_initiator": "wall_timeout",
    }) == {"likely_initiator": "wall_timeout"}

def test_continuation_wire_dependencies_drop_both_host_identity_sets(monkeypatch):
    import kernel_continuation_runtime

    monkeypatch.setattr(
        kernel_continuation_runtime, "binding_versions", lambda binding: dict(binding))
    private_fields = (
        set(kernel_wire_identity.WIRE_ENV_IDENTITY_FIELDS)
        | set(kernel_wire_identity.WIRE_AUDIT_ONLY_FIELDS)
    )
    binding = {field: f"private-{field}" for field in private_fields}
    binding["product_revision"] = "product-a"

    assert kernel_continuation_runtime._wire_dependencies(binding) == {
        "product_revision": "product-a",
    }


def test_wire_identity_content_drops_environment_identity_recursively():
    content = {
        "binding": {
            "environment": "host-a",
            "environment_revision": "env-a",
            "product_revision": "product-a",
        },
        "after_binding": [{
            "environment": "host-b",
            "environment_revision": "env-b",
            "product_revision": "product-a",
        }],
        "nested": {"deeper": [{"environment_revision": "env-c"}]},
    }

    public_content = kernel_wire_identity.wire_identity_content(content)

    assert public_content == {
        "binding": {"product_revision": "product-a"},
        "after_binding": [{"product_revision": "product-a"}],
        "nested": {"deeper": [{}]},
    }
    changed_product_content = {
        "binding": {
            "environment": "host-a",
            "environment_revision": "env-a",
            "product_revision": "product-b",
        },
        "after_binding": content["after_binding"],
        "nested": content["nested"],
    }
    assert public_content != kernel_wire_identity.wire_identity_content(
        changed_product_content)
