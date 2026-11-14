from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

from stateframe import routes
import kernel_frame_allocation_capture as allocation_capture
import kernel_frame_allocation_validation as allocation_validation
import kernel_completion_owner
import kernel_stateframe_f4
import kernel_stateframe_framefit5
from kernel_frame_optional_allocation import FrameAllocationReplanNeeded, identity
from kernel_stateframe_framefit5 import FrameFit5Error, scope_typed_state_for_unit
from kernel_stateframe_envelope import STATE_HEADER, serialize_stateframe_payload
from kernel_completion_owner import RequestActionContractError, assert_request_action_contract
from kernel_parallel_records import default_subframe_token_cap
from kernel_request_budget import ENDPOINT_PROMPT_TOKEN_BUDGET, HARD_MAX_REQUEST_BYTES
import kernel_unit_policy
from probes import react_repair
import stateframe.parallel_worker as parallel_worker
from stateframe.parallel_worker import _child_environment


def _bytes(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode("utf-8"))


def _planted_unit() -> SimpleNamespace:
    contract = {
        "id": "small_adapter",
        "outcome": "The parser rejects malformed values.",
        "owns": ["pkg/reader.py", "pkg/writer.py"],
        "creates": [],
        "reads": ["pkg/contract.py"],
        "requirement_ids": ["req:bounded-values"],
        "check": {
            "command_kind": "pytest",
            "selectors": ["tests/test_adapter.py"],
            "source_paths": ["tests/test_adapter.py"],
        },
        "exports": [{"id": "parse", "symbol": "parse", "signature": "parse(text)"}],
        "consumes": [{"export_id": "contract", "symbol": "Record", "path": "pkg/other.py"}],
    }
    spec = {
        "schema": "stateframe-unit-spec/3",
        "unit_id": "small_adapter",
        "contract": contract,
        "requirement_clauses": [{
            "id": "req:bounded-values",
            "text": "Reject values outside the contract bounds.",
            "refs": [],
            "source": "planner",
        }],
    }
    return SimpleNamespace(to_json=lambda: spec)


def _oversized_state() -> dict[str, object]:
    sites = []
    paths = ("pkg/reader.py", "pkg/writer.py") + ("pkg/other.py",) * 13
    for index in range(15):
        sites.append({
            "schema": "stateframe-framefit4-open-edit-site/1",
            "source": "remaining_sites_inventory",
            "path": paths[index % len(paths)],
            "line": index + 1,
            "literal": "name",
            "revision": "sha256:" + "a" * 64,
            "content_sha256": "b" * 64,
            "current_bytes": (f"SOURCE_{index}\n" + "x" * (500 if index < 2 else 3500)),
            "current_line": "name = value",
            "lines": {"start": 1, "end": 80},
            "read_this_request": True,
        })
    return {
        "schema": "stateframe-framefit5/1",
        "mode": "FRAME_ONLY",
        "task": {"source": "original_task_message", "sha256": "c" * 64},
        "next_site": sites[0],
        "open_sites": sites,
        "read_facts": [{"path": "pkg/other.py", "status": "UNCHANGED"}],
        "applied_edits": [],
        "last_verdict": {"schema": "verdict/1", "line": "parent verdict", "status": None},
        "action_ledger": {
            "schema": "ledger/1",
            "reads": [{"path": "pkg/other.py"}],
            "edits": [],
            "applied_edits": [],
            "last_test": None,
            "superseded_test_count": 0,
            "unclassified_actions": [{
                "kind": "parallel_disposition", "call_id": "parent-parallel-call",
                "path": "pkg/other.py",
            }],
            "latest_parallel_disposition": {"call_id": "parent-parallel-call"},
        },
        "established_facts": [{"name": "parent", "value": "parent-parallel-call"}],
        "prior_user_context": [{"content": "parent parallel action"}],
        "owner_protocol": {
            "schema": "owner/1", "admitted_actions": ["read", "edit"],
            "evidence_pin": "unit-local-evidence-pin " + "p" * 3000,
        },
        "open_work": {"open_edit_site_count": 15, "next_site_available": True},
    }


def test_subframe_typed_state_is_contract_scoped_and_fits_default_cap():
    unit = _planted_unit()
    original = _oversized_state()
    before_frame = {
        "schema": "stateframe-f4/1",
        "rev": "sha256:revision",
        "frame_hygiene": {"schema": "stateframe-frame-hygiene/1"},
        "state": {"framefit5": original},
    }
    scoped = scope_typed_state_for_unit(original, unit)
    after_frame = {**before_frame, "state": {"framefit5": scoped}}
    before_bytes = _bytes(before_frame)
    after_bytes = _bytes(after_frame)
    # A StateFrame is JSON text inside the request's message content, so the
    # surrounding request adds escaping overhead beyond its raw frame bytes.
    estimated_after_request = math.ceil((41_000 + after_bytes * 1.2) * 0.3193)
    before_top_key_bytes = {key: _bytes({key: value}) for key, value in before_frame.items()}
    after_top_key_bytes = {key: _bytes({key: value}) for key, value in after_frame.items()}
    print("PLANTED_FRAME_METRICS " + json.dumps({
        "unit": "small_adapter",
        "before_frame_bytes": before_bytes,
        "after_frame_bytes": after_bytes,
        "before_top_key_bytes": before_top_key_bytes,
        "after_top_key_bytes": after_top_key_bytes,
        "estimated_after_request_tokens": estimated_after_request,
    }, sort_keys=True))

    frame_text = json.dumps(after_frame, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert before_bytes > after_bytes
    assert estimated_after_request <= default_subframe_token_cap(ENDPOINT_PROMPT_TOKEN_BUDGET)
    assert "pkg/other.py" not in frame_text
    assert "parent-parallel-call" not in frame_text
    assert [site["path"] for site in scoped["open_sites"]] == [
        "pkg/reader.py", "pkg/writer.py",
    ]
    assert scoped["next_site"] == scoped["open_sites"][0]
    assert scoped["open_work"]["open_edit_site_count"] == 2
    assert "latest_parallel_disposition" not in scoped["action_ledger"]
    scope = scoped["subframe_scope"]
    assert scope["unit_id"] == "small_adapter"
    assert scope["contract"]["owns"] == ["pkg/reader.py", "pkg/writer.py"]
    assert scope["requirements"][0]["id"] == "req:bounded-values"
    assert scope["contract"]["check"]["selectors"] == ["tests/test_adapter.py"]
    assert scoped["owner_protocol"]["evidence_pin"].startswith("unit-local-evidence-pin ")


def _terminal_owner_contract(terminal_line: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    tool = {"type": "function", "function": {
        "name": "read", "description": "read",
        "parameters": {"type": "object", "properties": {}},
    }}
    schemas = [tool]
    digest = kernel_completion_owner._digest(schemas)
    contract: dict[str, object] = {
        "request_id": "child-terminal-line",
        "route": "read",
        "terminal_line": terminal_line,
        "admitted_tool_schemas": schemas,
        "advertised_actions": ["read"],
        "tool_schema_digest": digest,
        "owner_blocks": {"menu": {"digest": digest}},
    }
    contract["decision_digest"] = kernel_completion_owner._contract_decision_digest(contract)
    return schemas, contract


def _messages_with_frame(frame: dict[str, object]) -> list[dict[str, object]]:
    payload = {"state": {"framefit5": frame}}
    return [
        {"role": "system", "content": "System"},
        {"role": "user", "content": STATE_HEADER + "\n" + json.dumps(payload)},
    ]


def test_unit_projection_preserves_child_terminal_owner_line_exactly():
    unit = _planted_unit()
    terminal_line = (
        "Final step: call done now to submit this revision "
        "(uncertified; open: [uncertifiable host coverage])."
    )
    child_frame = scope_typed_state_for_unit(_oversized_state(), unit)
    child_protocol = {
        "schema": "stateframe-framefit5-owner/1",
        "route": "read",
        "admitted_actions": ["read"],
        "terminal_line": terminal_line,
    }
    child_frame["owner_protocol"] = child_protocol

    rescoped = scope_typed_state_for_unit(child_frame, unit)

    assert rescoped["owner_protocol"] == child_protocol
    schemas, contract = _terminal_owner_contract(terminal_line)
    assert_request_action_contract(
        None, _messages_with_frame(rescoped), schemas,
        request_id="child-terminal-line", frozen_contract=contract,
    )


@pytest.mark.parametrize("authority", ["altered", "missing"])
def test_final_request_contract_rejects_altered_or_missing_terminal_owner_line(authority):
    unit = _planted_unit()
    terminal_line = (
        "Final step: call done now to submit this revision "
        "(uncertified; open: [uncertifiable host coverage])."
    )
    child_frame = scope_typed_state_for_unit(_oversized_state(), unit)
    child_frame["owner_protocol"] = {
        "schema": "stateframe-framefit5-owner/1",
        "route": "read",
        "admitted_actions": ["read"],
        "terminal_line": terminal_line,
    }
    child_frame = scope_typed_state_for_unit(child_frame, unit)
    protocol = child_frame["owner_protocol"]
    assert isinstance(protocol, dict)
    if authority == "altered":
        protocol["terminal_line"] = terminal_line + " changed"
    else:
        protocol.pop("terminal_line")
    schemas, contract = _terminal_owner_contract(terminal_line)

    with pytest.raises(
            RequestActionContractError,
            match="serialized terminal owner line was removed or changed"):
        assert_request_action_contract(
            None, _messages_with_frame(child_frame), schemas,
            request_id="child-terminal-line", frozen_contract=contract,
        )


def test_unit_projection_preserves_only_contract_edit_sites():
    unit = _planted_unit()
    unit.to_json()["contract"]["owns"] = ["pkg/reader.py"]
    unit.to_json()["contract"]["creates"] = ["pkg/writer.py"]
    owned = {"path": "pkg/reader.py", "line": 1}
    created = {"path": "pkg/writer.py", "line": 2}
    read_only = {"path": "pkg/contract.py", "line": 3}
    sibling = {"path": "pkg/other.py", "line": 4}
    state = {**_oversized_state(),
             "open_sites": [sibling, owned, read_only, created],
             "next_site": sibling}

    projected = scope_typed_state_for_unit(state, unit)

    assert projected["open_sites"] == [owned, created]
    assert projected["next_site"] == owned
    assert projected["open_work"]["open_edit_site_count"] == 2
    assert projected["open_work"]["next_site_available"] is True


def test_unit_projection_rejects_malformed_host_edit_sites():
    state = {**_oversized_state(), "open_sites": [{"line": 1}]}
    with pytest.raises(FrameFit5Error, match="malformed edit site"):
        scope_typed_state_for_unit(state, _planted_unit())


def test_unit_projection_expands_location_sites_before_scoping(monkeypatch, tmp_path):
    import kernel_open_site_capacity

    owned = {"path": "pkg/reader.py", "line": 1}
    sibling = {"path": "pkg/other.py", "line": 2}
    state = {**_oversized_state(),
             "site_inventory_encoding": kernel_open_site_capacity.LOCATION_MODE,
             "open_sites": [owned, sibling], "next_site": owned}
    monkeypatch.setattr(kernel_open_site_capacity, "expand_location_inventory",
                        lambda frame, root: [owned, sibling] if root == tmp_path else [])

    projected = scope_typed_state_for_unit(state, _planted_unit(), root=tmp_path)

    assert projected["open_sites"] == [owned]
    assert projected["next_site"] == owned
    with pytest.raises(FrameFit5Error, match="no workspace authority"):
        scope_typed_state_for_unit(state, _planted_unit())


def test_allocation_authority_uses_unit_scoped_sites_and_ledger(monkeypatch, tmp_path):
    unit = _planted_unit()
    owned_site = {"path": "pkg/reader.py", "line": 1, "revision": "sha256:reader"}
    sibling_site = {"path": "pkg/other.py", "line": 1, "revision": "sha256:sibling"}
    context_frame = {
        "schema": "stateframe-framefit5/1",
        "mode": "FRAME_ONLY",
        "task": {"source": "original_task_message", "sha256": "c" * 64},
        "next_site": owned_site,
        "open_sites": [owned_site, sibling_site],
        "read_facts": [{"path": "pkg/other.py", "status": "UNCHANGED"}],
        "applied_edits": [],
        "last_verdict": {},
        "action_ledger": {
            "schema": "ledger/1",
            "reads": [{"path": "pkg/other.py"}],
            "edits": [],
            "applied_edits": [],
            "last_test": None,
            "superseded_test_count": 0,
            "unclassified_actions": [{"kind": "parallel_disposition", "call_id": "parent-call"}],
        },
        "owner_protocol": {"schema": "stateframe-framefit5-owner/1",
                           "route": "ACTION", "admitted_actions": ["read", "edit"]},
    }
    projected = scope_typed_state_for_unit(context_frame, unit)
    request_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": STATE_HEADER + "\n" + json.dumps({
            "state": {"framefit5": projected},
        }, separators=(",", ":"))},
    ]
    workspace_authorization = {"scope": "workspace", "generation": "1"}
    runtime = SimpleNamespace(
        binding=lambda: {"product_revision": "sha256:revision"},
        _request_source_closure={"workspace_authorization": workspace_authorization},
        _framefit_action_context={"framefit5": context_frame},
        frame_ceiling=100_000,
        owner=SimpleNamespace(root=tmp_path),
    )
    monkeypatch.setattr(kernel_unit_policy, "active", lambda: SimpleNamespace(spec=unit))
    monkeypatch.setattr(kernel_stateframe_framefit5, "_next_sites",
                        lambda _runtime, _context: [owned_site, sibling_site])
    monkeypatch.setattr("kernel_frame_source_snapshot.frame_source_snapshot",
                        lambda _runtime, _plan: None)

    site_id = identity(owned_site)
    site_inventory = {
        "inventory_identity": [site_id],
        "selected_identities": [site_id],
        "open_sites": [site_id],
        "next_site": site_id,
        "encoding_mode": "full_inventory",
        "sites": [owned_site],
    }
    capacity = {
        "frame_ceiling": runtime.frame_ceiling,
        "token_ceiling": ENDPOINT_PROMPT_TOKEN_BUDGET,
        "host_byte_ceiling": HARD_MAX_REQUEST_BYTES,
        "template_environment": allocation_validation.template_environment(),
        "tool_schema_digest": "sha256:tools",
        "system_messages_digest": identity([request_messages[0]]),
    }
    allocation = SimpleNamespace(
        validate=lambda: None,
        identity={
            "product_revision": "sha256:revision",
            "workspace_authorization": workspace_authorization,
            "ledger_digest": identity(projected["action_ledger"]),
        },
        source_layout={"source_plan": {}},
        capacity=capacity,
        site_inventory=site_inventory,
        optional_heads={"panel": None, "dependency_revisions": {}},
    )

    allocation_validation.validate_allocation_authority(
        allocation, runtime, request_messages, check_layout=False,
    )

    tampered = json.loads(request_messages[1]["content"][len(STATE_HEADER) + 1:])
    tampered["state"]["framefit5"]["owner_protocol"]["route"] = "UNAUTHORIZED"
    altered = [request_messages[0], {**request_messages[1],
               "content": STATE_HEADER + "\n" + json.dumps(tampered, separators=(",", ":"))}]
    with pytest.raises(FrameAllocationReplanNeeded, match="fields=\\['owner_protocol'\\]"):
        allocation_validation.validate_allocation_authority(
            allocation, runtime, altered, check_layout=False,
        )

    tampered_ledger = json.loads(request_messages[1]["content"][len(STATE_HEADER) + 1:])
    tampered_ledger["state"]["framefit5"]["action_ledger"]["reads"].append(
        {"path": "pkg/reader.py", "call_id": "candidate-only-read"},
    )
    altered_ledger = [request_messages[0], {**request_messages[1],
                      "content": STATE_HEADER + "\n" + json.dumps(
                          tampered_ledger, separators=(",", ":"))}]
    with pytest.raises(FrameAllocationReplanNeeded, match="unit-scoped serialized ledger"):
        allocation_validation.validate_allocation_authority(
            allocation, runtime, altered_ledger, check_layout=False,
        )

    allocation.site_inventory = {
        **site_inventory,
        "inventory_identity": [site_id, identity(sibling_site)],
        "selected_identities": [site_id, identity(sibling_site)],
        "open_sites": [site_id, identity(sibling_site)],
        "sites": [owned_site, sibling_site],
    }
    with pytest.raises(FrameAllocationReplanNeeded, match="unit-scoped allocation inventory"):
        allocation_validation.validate_allocation_authority(
            allocation, runtime, request_messages, check_layout=False,
        )


def test_scoped_allocation_capture_binds_host_projected_ledger(monkeypatch, tmp_path):
    unit = _planted_unit()
    host_frame = {
        "schema": "stateframe-framefit5/1",
        "mode": "FRAME_ONLY",
        "task": {"source": "original_task_message", "sha256": "c" * 64},
        "next_site": None,
        "open_sites": [],
        "read_facts": [],
        "applied_edits": [],
        "last_verdict": {},
        "action_ledger": {
            "schema": "ledger/1",
            "reads": [{"path": "pkg/reader.py"}, {"path": "pkg/other.py"}],
            "edits": [],
            "applied_edits": [],
            "last_test": None,
            "superseded_test_count": 0,
            "unclassified_actions": [{
                "kind": "parallel_disposition", "call_id": "parent-call",
            }],
        },
    }
    host_projection = scope_typed_state_for_unit(host_frame, unit)
    candidate_ledger = {
        **host_projection["action_ledger"],
        "reads": [
            *host_projection["action_ledger"]["reads"],
            {"path": "pkg/reader.py", "call_id": "candidate-only-read"},
        ],
    }
    candidate_frame = {**host_projection, "action_ledger": candidate_ledger}
    assert candidate_ledger != host_projection["action_ledger"]
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task: update the reader."},
        {"role": "user", "content": serialize_stateframe_payload({
            "schema": "stateframe-f4/1", "state": {"framefit5": candidate_frame},
        })},
    ]
    body = {"messages": messages, "tools": []}
    optional_heads = {
        "candidates": [], "retained_identities": [], "allowance": 0,
        "omitted_count": 0, "deduplicated_count": 0,
        "render_policy_identity": "sha256:render-policy",
    }
    site_inventory = {
        "inventory_identity": [], "selected_identities": [], "open_sites": [],
        "next_site": None, "encoding_mode": "full_inventory", "sites": [],
    }
    monkeypatch.setattr(kernel_unit_policy, "active", lambda: SimpleNamespace(spec=unit))
    monkeypatch.setattr(kernel_stateframe_f4, "latest_complete_tool_exchange", lambda _messages: [])
    monkeypatch.setattr(allocation_capture, "_capture_carrier_roles",
                        lambda *_args, **_kwargs: {})
    monkeypatch.setattr(allocation_capture, "_obligations_and_locations",
                        lambda *_args, **_kwargs: ([], {}))
    monkeypatch.setattr(allocation_capture, "selection_from_frame",
                        lambda *_args: (optional_heads, site_inventory))
    monkeypatch.setattr("kernel_request_source_window.source_delivery_satisfied",
                        lambda _receipt: True)
    runtime = SimpleNamespace(
        binding=lambda: {"product_revision": "sha256:revision"},
        _framefit_action_context={"framefit5": host_frame},
        owner=SimpleNamespace(root=tmp_path),
        frame_ceiling=100_000,
    )
    capture = SimpleNamespace(request_id="request", _framefit5_newest_user_suffix=())
    authorization = {"scope": "workspace", "generation": "1"}

    allocation = allocation_capture.capture_candidate(
        runtime=runtime, capture=capture, body=body, source_plan={},
        receipt={"status": "COMPLETE", "proof_status": "VERIFIED",
                 "workspace_authorization": authorization},
        sources={}, optional_inputs=(optional_heads, site_inventory),
        byte_ceiling=100_000, token_ceiling=ENDPOINT_PROMPT_TOKEN_BUDGET,
        measurement_identity="strict-test",
    )

    assert allocation.identity["ledger_digest"] == identity(host_projection["action_ledger"])


def test_worker_environment_never_inherits_parent_force_manifest(monkeypatch, tmp_path):
    monkeypatch.setenv("STATEFRAME_PARALLEL_FORCE_MANIFEST", str(tmp_path / "forced.json"))
    environment = _child_environment({"environment": {}})
    assert "STATEFRAME_PARALLEL_FORCE_MANIFEST" not in environment


def test_worker_environment_carries_recorded_home_identity_pin_not_ambient(monkeypatch):
    pin = '{"home_st_dev":1,"home_st_ino":2,"key_st_ctime_ns":3,"key_st_dev":1,"key_st_ino":4,"realpath":"/h"}'
    monkeypatch.setenv("STATEFRAME_RUNNER_HOME_IDENTITY", "ambient-must-not-leak")
    assert "STATEFRAME_RUNNER_HOME_IDENTITY" not in _child_environment({"environment": {}})
    environment = _child_environment({"environment": {"STATEFRAME_RUNNER_HOME_IDENTITY": pin}})
    assert environment["STATEFRAME_RUNNER_HOME_IDENTITY"] == pin


@pytest.mark.parametrize("override", [None, "unit", "unit.patch"])
def test_worker_environment_forwards_effective_route_credentials(monkeypatch, override):
    credentials = {"SF_TEST_PARENT_KEY": "fake-parent", "SF_TEST_READ_KEY": "fake-read",
                   "SF_TEST_UNIT_KEY": "fake-unit", "SF_TEST_PATCH_KEY": "fake-patch"}
    default = {"endpoint": "http://127.0.0.1:8080/v1", "model": "test",
               "api_key_env": "SF_TEST_PARENT_KEY"}
    entries = ["read.api_key_env=SF_TEST_READ_KEY"]
    if override:
        entries.append("unit.api_key_env=SF_TEST_UNIT_KEY")
    if override == "unit.patch":
        entries.append("unit.patch.api_key_env=SF_TEST_PATCH_KEY")
    # An unused class must neither leak its credential nor prevent this worker's launch.
    entries.append("unit.other.api_key_env=SF_TEST_UNUSED_KEY")
    recorded = routes.resolve(default, cli=entries,
                              environ={**credentials, "SF_TEST_UNUSED_KEY": "fake-unused"}).public()
    for name, value in credentials.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("SF_TEST_UNUSED_KEY", raising=False)
    spec = {"api_key_env": "SF_TEST_PARENT_KEY", "routes": recorded,
            "environment": {"STATEFRAME_ROUTE_SCOPE": "unit", "STATEFRAME_ROUTE_UNIT_CLASS": "patch"}}

    environment = _child_environment(spec)

    expected = {"SF_TEST_PARENT_KEY", "SF_TEST_READ_KEY"} if override is None else {
        "SF_TEST_PARENT_KEY", "SF_TEST_PATCH_KEY" if override == "unit.patch" else "SF_TEST_UNIT_KEY"}
    assert set(credentials) & environment.keys() == expected
    assert all(environment[name] == credentials[name] for name in expected)
    assert "SF_TEST_UNUSED_KEY" not in environment
    assert all(value not in json.dumps(spec) for value in credentials.values())


@pytest.mark.parametrize("override", ["read", "unit", "unit.patch"])
def test_worker_refuses_missing_route_credential_before_spawn(monkeypatch, tmp_path, override):
    recorded = routes.resolve(
        {"endpoint": "http://127.0.0.1:8080/v1", "model": "test", "api_key_env": None},
        cli=[f"{override}.api_key_env=SF_TEST_MISSING_KEY"],
        environ={"SF_TEST_MISSING_KEY": "fake-missing"}).public()
    monkeypatch.delenv("SF_TEST_MISSING_KEY", raising=False)

    def unexpected_spawn(*args, **kwargs):
        pytest.fail("worker spawned before checking its route credential")

    monkeypatch.setattr(parallel_worker.subprocess, "Popen", unexpected_spawn)

    with pytest.raises(parallel_worker.WorkerLaunchError, match="SF_TEST_MISSING_KEY is not set"):
        _launch_environment_recording_worker(
            tmp_path, environment={"STATEFRAME_ROUTE_SCOPE": "unit", "STATEFRAME_ROUTE_UNIT_CLASS": "patch"},
            recorded_routes=recorded)
    assert not (tmp_path / "worker" / "proc").exists()


def test_forced_manifest_consumer_is_disabled_under_unit_policy(monkeypatch, tmp_path):
    manifest = str(tmp_path / "forced.json")
    monkeypatch.setenv("STATEFRAME_PARALLEL_FORCE_MANIFEST", manifest)
    monkeypatch.setattr(kernel_unit_policy, "_installed", None)
    assert react_repair._forced_parallel_manifest_for_parent() == manifest

    monkeypatch.setattr(kernel_unit_policy, "_installed", SimpleNamespace())
    assert react_repair._forced_parallel_manifest_for_parent() is None


def test_forced_manifest_reader_uses_current_home(monkeypatch, tmp_path):
    for index, home in enumerate((tmp_path / "first-home", tmp_path / "second-home")):
        home.mkdir()
        (home / "forced.json").write_text(f'{{"index":{index}}}', encoding="utf-8")
        monkeypatch.setenv("HOME", str(home))
        assert react_repair._read_forced_parallel_manifest("~/forced.json") == (
            {"index": index}, f'{{"index":{index}}}', None)


_PARENT_ONLY_ENV = {
    "STATEFRAME_PARALLEL_FORCE_MANIFEST": "parent-only-manifest.json",
    "STATEFRAME_PARENT_BUDGET_CAP": "parent-only-budget",
    "STATEFRAME_PARENT_ONLY_SWITCH": "parent-only-switch",
}


def _launch_environment_recording_worker(tmp_path: Path, *, environment: dict[str, str] | None = None,
                                         api_key_env: str | None = None,
                                         recorded_routes: dict[str, object] | None = None):
    package = tmp_path / "fake" / "stateframe"
    engine_dir = package / "engine"
    engine_dir.mkdir(parents=True)
    (package / "_child.py").write_text(
        "import json, os, sys, time\n"
        "from pathlib import Path\n"
        "run_dir = Path(sys.argv[1])\n"
        "(run_dir / 'child-environment.json').write_text(json.dumps(dict(os.environ), sort_keys=True))\n"
        "time.sleep(60)\n"
    )
    run_dir = tmp_path / "worker"
    run_dir.mkdir()
    (run_dir / "spec.json").write_text(json.dumps({
        "schema": "stateframe-run/1",
        "public_row": {},
        "engine_dir": str(engine_dir),
        "parallel_engine_python": sys.executable,
        "api_key_env": api_key_env,
        "routes": recorded_routes,
        "environment": environment or {},
    }))
    return parallel_worker.launch_worker(run_dir, 1)


def _recorded_child_environment(handle) -> dict[str, str]:
    record = handle.run_dir / "child-environment.json"
    deadline = time.monotonic() + 5
    while not record.is_file() and handle.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert record.is_file(), f"worker child did not record its environment: {handle.log_path.read_text()}"
    value = json.loads(record.read_text())
    assert isinstance(value, dict)
    assert all(isinstance(key, str) and isinstance(item, str) for key, item in value.items())
    return value


def _assert_parent_only_environment_absent(environment: dict[str, str]) -> None:
    leaked = sorted(set(_PARENT_ONLY_ENV) & environment.keys())
    assert not leaked, f"parent-only environment reached worker child: {', '.join(leaked)}"


def _stop_environment_recording_worker(handle) -> None:
    if handle.poll() is None:
        handle.cancel("test cleanup")
    code = handle.reap(timeout=10)
    if code is None:
        handle.escalate(grace=0)
        code = handle.reap(timeout=10)
    assert code is not None, f"worker child did not stop: {handle.log_path.read_text()}"


@pytest.mark.host_sandbox
def test_real_worker_child_environment_drops_parent_only_stateframe_vars(monkeypatch, tmp_path):
    for name, value in _PARENT_ONLY_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("OPENAI_TEST_API_KEY", "synthetic-provider-key")
    handle = _launch_environment_recording_worker(
        tmp_path,
        environment={"STATEFRAME_TEST_PYTHON": sys.executable, "PYTHONHASHSEED": "17"},
        api_key_env="OPENAI_TEST_API_KEY",
    )
    try:
        environment = _recorded_child_environment(handle)
        _assert_parent_only_environment_absent(environment)
        assert environment["STATEFRAME_TEST_PYTHON"] == sys.executable
        assert environment["PYTHONHASHSEED"] == "17"
        assert environment["OPENAI_TEST_API_KEY"] == "synthetic-provider-key"
        assert environment["SF_SUBFRAME_INCARNATION"] == "1"
    finally:
        _stop_environment_recording_worker(handle)


@pytest.mark.host_sandbox
def test_real_worker_environment_guard_detects_planted_allowlist_leak(monkeypatch, tmp_path):
    for name, value in _PARENT_ONLY_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        parallel_worker,
        "CHILD_ENV_ALLOWLIST",
        parallel_worker.CHILD_ENV_ALLOWLIST | {"STATEFRAME_PARALLEL_FORCE_MANIFEST"},
    )
    handle = _launch_environment_recording_worker(tmp_path)
    try:
        environment = _recorded_child_environment(handle)
        with pytest.raises(AssertionError, match="STATEFRAME_PARALLEL_FORCE_MANIFEST"):
            _assert_parent_only_environment_absent(environment)
    finally:
        _stop_environment_recording_worker(handle)


_PINNED_LIMITS = "[CONTINUATION LIMITS | stateframe-continuation-limits/1] model_decisions_remaining=12"
_LIVE_LIMITS = "[CONTINUATION LIMITS | stateframe-continuation-limits/1] model_decisions_remaining=11"


def _owner_limits(limits: str | None) -> dict:
    owner = {"schema": "stateframe-framefit5-owner/1"}
    if limits is not None:
        owner["continuation_limits"] = limits
    return {"mode": "FRAME_ONLY", "owner_protocol": owner}


def test_scoped_state_keeps_pinned_limits_over_live_counter_without_delivered_line():
    scoped = scope_typed_state_for_unit(
        _owner_limits(_PINNED_LIMITS), _planted_unit(), continuation_limits=_LIVE_LIMITS)
    assert scoped["owner_protocol"]["continuation_limits"] == _PINNED_LIMITS


def test_scoped_state_uses_live_counter_when_nothing_pinned():
    scoped = scope_typed_state_for_unit(
        _owner_limits(None), _planted_unit(), continuation_limits=_LIVE_LIMITS)
    assert scoped["owner_protocol"]["continuation_limits"] == _LIVE_LIMITS


def test_scoped_state_prefers_delivered_host_line():
    delivered = [{"role": "user", "content": "x\n" + _LIVE_LIMITS}]
    scoped = scope_typed_state_for_unit(
        _owner_limits(_PINNED_LIMITS), _planted_unit(), delivered_windows=delivered,
        continuation_limits="")
    assert scoped["owner_protocol"]["continuation_limits"] == _LIVE_LIMITS
