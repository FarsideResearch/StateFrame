"""Subframe contract and send-budget proofs through the shipped profile and wire."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from barrier_endpoint import BarrierEndpoint
from e2e_support import (
    assert_shipped_wire,
    assert_subframe_surface,
    parallel_call_roots,
    parent_wire_files,
    read_json,
    result_records,
    run_cli,
    subframe_directories,
    subframe_wire_files,
    summary_for,
    tool_step,
    wire_message_text,
)


pytestmark = pytest.mark.host_sandbox


def _contract(unit_id: str, owned: str, *, exports: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "id": unit_id,
        "outcome": f"Update {owned}.",
        "owns": [owned],
        "creates": [],
        "exports": exports or [],
        "consumes": [],
        "reads": [],
        "requirements": [],
        "check": {"tests": []},
    }


def _repo(root: Path, files: dict[str, str]) -> Path:
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pkg/__init__.py").write_text("")
    for path, content in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    (root / "tests/test_smoke.py").write_text("def test_smoke():\n    assert True\n")
    return root


_LEGACY_UNIT_TERM = re.compile(r"(?<![a-z])units?(?![a-z])", re.IGNORECASE)


def _uses_legacy_unit_term(value: str) -> bool:
    # Whole-word match: other tools' text legitimately says "unittest runners".
    return _LEGACY_UNIT_TERM.search(value) is not None


def test_subframe_naming_guard_detects_planted_legacy_text() -> None:
    assert _uses_legacy_unit_term("unit_blocked")
    assert _uses_legacy_unit_term("TOO_MANY_UNITS")
    assert _uses_legacy_unit_term("Each unit starts loaded")
    assert not _uses_legacy_unit_term("pytest or unittest runners")


def test_unknown_non_python_export_cannot_commit_without_a_passing_check(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    repo = _repo(tmp_path / "repo", {
        "pkg/api.ts": "export function readValue(value: number): number { return value; }\n",
        "pkg/other.py": "VALUE = 0\n",
    })
    expected = {"kind": "function", "params": [], "returns": None, "decorators": []}
    parent = [
        tool_step("parallel", {"subframes": [
            _contract("api", "pkg/api.ts", exports=[{
                "id": "api.readValue", "path": "pkg/api.ts", "symbol": "readValue",
                "signature": expected,
            }]),
            _contract("other", "pkg/other.py"),
        ]}),
        tool_step("done", {"summary": "The unsupported signature was kept unverified."}),
    ]
    subframes = {
        "api": [
            tool_step("str_replace", {"path": "pkg/api.ts", "old_str": "return value;",
                                       "new_str": "return value + 1;"}),
            tool_step("done", {"summary": "The API edit is ready for contract review."}),
        ],
        "other": [
            tool_step("str_replace", {"path": "pkg/other.py", "old_str": "VALUE = 0",
                                       "new_str": "VALUE = 1"}),
            tool_step("done", {"summary": "The independent edit is ready."}),
        ],
    }
    run_dir = tmp_path / "run"
    with BarrierEndpoint(parent, subframes) as endpoint:
        completed = run_cli(isolated_engine_python, endpoint.url, repo, run_dir,
                            "Update both files while preserving the API contract.", parallel="2")
        endpoint.assert_consumed()
    assert (run_dir / "summary.json").is_file(), completed.stdout + completed.stderr

    result, = result_records(run_dir).values()
    api = next(item for item in result.outcomes if item.unit_id == "api")
    assert (api.status, api.reason) == ("REFUSED", "UNVERIFIABLE_CONTRACT")
    call_root, = parallel_call_roots(run_dir)
    verdict = read_json(call_root / "refused-contracts" / "api.json")["verdicts"][0]
    assert (verdict["status"], verdict["expected"], verdict["actual"]) == ("UNKNOWN", expected, None)
    other = next(item for item in result.outcomes if item.unit_id == "other")
    assert (other.status, other.txn_id) == ("COMMITTED_UNVERIFIED", 1)
    assert other.check == "NOT_RUN"
    assert result.txn_order == (1,)
    assert (run_dir / "workspace/pkg/api.ts").read_text() == (
        "export function readValue(value: number): number { return value; }\n")
    parent_payloads = assert_shipped_wire(parent_wire_files(run_dir))
    # The fan-out disposition rides in the action ledger's protected latest_parallel_disposition.
    receipts = [ledger["latest_parallel_disposition"]["sections"]["disposition"]
                for ledger in (payload["state"]["framefit5"]["action_ledger"]
                               for payload in parent_payloads)
                if isinstance(ledger.get("latest_parallel_disposition"), dict)]
    assert len(receipts) == 1
    assert ("Plan other: owns=pkg/other.py; check=NOT_RUN; consumes=-; "
            "status=COMMITTED_UNVERIFIED; check_status=NOT_RUN") in receipts[0]["result_summary"]
    children = subframe_directories(run_dir)
    assert set(children) == {"api", "other"}
    for unit_id, directory in children.items():
        requests = subframe_wire_files(directory)
        assert len(requests) >= 1
        assert_shipped_wire(requests, task_marker=f"Subframe {unit_id}:")
        for request in requests:
            assert_subframe_surface(request)


def test_unoffered_delete_tool_is_refused_without_removing_an_unowned_file(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    repo = _repo(tmp_path / "repo", {
        "pkg/owned.py": "VALUE = 0\n",
        "pkg/protected.py": "KEEP = True\n",
    })
    parent = [
        tool_step("parallel", {"subframes": [_contract("edit", "pkg/owned.py")]}),
        tool_step("done", {"summary": "The forbidden file operation had no effect."}),
    ]
    subframes = {
        "edit": [tool_step("file_delete", {"path": "pkg/protected.py"}, required=True)],
    }
    subframes["edit"][0]["force"] = True
    run_dir = tmp_path / "run"
    with BarrierEndpoint(parent, subframes) as endpoint:
        completed = run_cli(isolated_engine_python, endpoint.url, repo, run_dir,
                            "Keep subframe tools inside the declared file policy.", parallel="2",
                            arguments=("--parallel-min-subframes", "1"))
        endpoint.assert_consumed()
    assert (run_dir / "summary.json").is_file(), completed.stdout + completed.stderr

    result, = result_records(run_dir).values()
    assert result.admission.status == "ADMITTED"
    assert (run_dir / "workspace/pkg/protected.py").read_text() == "KEEP = True\n"
    assert all("pkg/protected.py" not in [delta.path for delta in outcome.files]
               for outcome in result.outcomes)
    children = subframe_directories(run_dir)
    assert set(children) == {"edit"}
    requests = subframe_wire_files(children["edit"])
    assert_shipped_wire(requests, task_marker="Subframe edit:")
    assert all("file_delete" not in json.dumps(read_json(path)["tools"])
               for path in requests)
    for path in requests:
        assert_subframe_surface(path)
    wire_text = "\n".join(wire_message_text(path) for path in requests)
    assert "SCOPE_VIOLATION" in wire_text


def test_send_budget_preserves_spent_held_free_and_parent_reserve_at_each_ledger_transition(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    repo = _repo(tmp_path / "repo", {"pkg/a.py": "VALUE = 0\n", "pkg/b.py": "VALUE = 0\n"})
    units = [_contract("a", "pkg/a.py"), _contract("b", "pkg/b.py")]
    parent = [
        tool_step("parallel", {"subframes": units}),
        tool_step("done", {"summary": "Both bounded subframes used their send allowance."}),
    ]
    subframes = {
        unit_id: [
            tool_step("str_replace", {"path": f"pkg/{unit_id}.py", "old_str": "VALUE = 0",
                                       "new_str": "VALUE = 1"}),
            tool_step("done", {"summary": f"{unit_id} is complete."}),
        ]
        for unit_id in ("a", "b")
    }
    run_dir = tmp_path / "run"
    arguments = ("--parallel-subframe-max-requests", "4", "--parallel-parent-reserve", "2")
    with BarrierEndpoint(parent, subframes) as endpoint:
        completed = run_cli(isolated_engine_python, endpoint.url, repo, run_dir,
                            "Use bounded subframe sends while retaining parent turns.",
                            parallel="2", arguments=arguments, max_turns=13)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        endpoint.assert_consumed()

    result, = result_records(run_dir).values()
    assert result.admission.status == "ADMITTED"
    run_spec = read_json(run_dir / "spec.json")
    assert run_spec["profile"] == "r2hc3"
    assert run_spec["parallel_limits"]["unit_max_requests"] == 4
    assert run_spec["parallel_limits"]["parent_reserve_requests"] == 2
    summary = read_json(run_dir / "summary.json")
    assert summary["requests"] == 2
    assert summary["parallel"]["totals"]["requests"] == 4
    assert result.txn_order == (1, 2)
    assert all(item.usage.physical_sends == 2 for item in result.outcomes)

    intent = read_json(run_dir / "parallel" / result.call_id / "intent.json")
    assert intent["admission"]["status"] == "ADMITTED"
    assert intent["reservations"] == {"a": 4, "b": 4}
    expected_local = [(0, 1, 3), (1, 0, 3), (1, 1, 2), (2, 0, 2)]
    for unit_id in ("a", "b"):
        ledger = run_dir / "parallel" / result.call_id / "units" / unit_id / "sends-ledger.jsonl"
        records = [json.loads(line) for line in ledger.read_text().splitlines()]
        assert [row["state"] for row in records] == [
            "AUTHORIZED", "SETTLED", "AUTHORIZED", "SETTLED"]
        assert len({row["send_id"] for row in records}) == 2
        actual_local = []
        spent = held = 0
        for row in records:
            if row["state"] == "AUTHORIZED":
                held += 1
            else:
                spent += 1
                held -= 1
            free = intent["reservations"][unit_id] - spent - held
            actual_local.append((spent, held, free))
        assert actual_local == expected_local
        assert all(min(counters) >= 0 for counters in actual_local)
        assert all(sum(counters) == 4 for counters in actual_local)
    assert sum(intent["reservations"].values()) + run_spec["parallel_limits"]["parent_reserve_requests"] == 10
    assert all(len((run_dir / "parallel" / result.call_id / "units" / unit_id /
                    "sends-ledger.jsonl").read_text().splitlines()) == 4
               for unit_id in ("a", "b"))
    assert_shipped_wire(parent_wire_files(run_dir))
    children = subframe_directories(run_dir)
    assert set(children) == {"a", "b"}
    for unit_id, directory in children.items():
        assert_shipped_wire(subframe_wire_files(directory), task_marker=f"Subframe {unit_id}:")


def test_second_parallel_call_uses_remaining_run_wide_parent_allowance(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    repo = _repo(tmp_path / "repo", {"pkg/a.py": "VALUE = 0\n", "pkg/b.py": "VALUE = 0\n"})
    units = [_contract("a", "pkg/a.py"), _contract("b", "pkg/b.py")]
    parent = [
        tool_step("parallel", {"subframes": units}),
        tool_step("parallel", {"subframes": units}),
        tool_step("done", {"summary": "The second call was refused by the remaining allowance."}),
    ]
    subframes = {
        unit_id: [
            tool_step("str_replace", {"path": f"pkg/{unit_id}.py", "old_str": "VALUE = 0",
                                       "new_str": "VALUE = 1"}),
            tool_step("done", {"summary": f"{unit_id} is complete."}),
        ] for unit_id in ("a", "b")
    }
    run_dir = tmp_path / "run"
    arguments = ("--parallel-subframe-max-requests", "4", "--parallel-parent-reserve", "2")
    with BarrierEndpoint(parent, subframes) as endpoint:
        completed = run_cli(isolated_engine_python, endpoint.url, repo, run_dir,
                            "Use two bounded subframes, then check a second call's allowance.",
                            parallel="2", arguments=arguments, max_turns=13)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        endpoint.assert_consumed()

    results = list(result_records(run_dir).values())
    assert len(results) == 2
    admitted, = (result for result in results if result.admission.status == "ADMITTED")
    refused, = (result for result in results if result.admission.status == "REFUSED")
    assert [(item.unit_id, item.usage.physical_sends) for item in admitted.outcomes] == [
        ("a", 2), ("b", 2)]
    assert (refused.admission.status, refused.admission.refusal.reason) == (
        "REFUSED", "RESOURCES_UNAVAILABLE")
    budget = read_json(run_dir / "parallel/budget.json")
    assert (budget["limit"], budget["parent_steps_spent"], budget["subframe_sends_authorized"],
            budget["held_reservations"], budget["free"], budget["parent_reserve"]) == (
                13, 2, 4, 0, 5, 2)
    assert sum(budget[key] for key in ("parent_steps_spent", "subframe_sends_authorized",
                                        "held_reservations", "free", "parent_reserve")) == 13
    assert all(budget[key] >= 0 for key in ("parent_steps_spent", "subframe_sends_authorized",
                                            "held_reservations", "free", "parent_reserve"))
    assert not (run_dir / "parallel" / refused.call_id / "units").exists()
    assert_shipped_wire(parent_wire_files(run_dir))
    for unit_id, directory in subframe_directories(run_dir).items():
        assert_shipped_wire(subframe_wire_files(directory), task_marker=f"Subframe {unit_id}:")


def test_parent_turn_reservation_refuses_before_any_subframe_launch(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    repo = _repo(tmp_path / "repo", {"pkg/a.py": "VALUE = 0\n", "pkg/b.py": "VALUE = 0\n"})
    parent = [
        tool_step("parallel", {"subframes": [_contract("a", "pkg/a.py"), _contract("b", "pkg/b.py")]}),
        tool_step("done", {"summary": "The insufficient budget was refused before launch."}),
    ]
    run_dir = tmp_path / "run"
    with BarrierEndpoint(parent, {}) as endpoint:
        completed = run_cli(isolated_engine_python, endpoint.url, repo, run_dir,
                            "Refuse subframes when their maximum sends would consume parent turns.",
                            parallel="2", max_turns=2)
        # The refused fan-out changed nothing, so the host rejects the parent's done
        # (empty diff, no check) and the two-turn allowance is spent: a loud stop, not a pass.
        assert completed.returncode == 1, completed.stdout + completed.stderr
        endpoint.assert_consumed()
    assert summary_for(run_dir)["engine_result"] == "RESPONSE_BUDGET_EXHAUSTED"

    result, = result_records(run_dir).values()
    refusal = result.admission.refusal
    assert refusal is not None
    assert refusal.reason == "RESOURCES_UNAVAILABLE"
    assert "subframe_max_requests=12" in refusal.detail
    assert "parent_reserve_requests=4" in refusal.detail
    # required counts the subframes' own sends (2 x 12); the reserve is named beside it.
    assert "required=24" in refusal.detail
    assert "free=0" in refusal.detail
    assert result.txn_order == ()
    assert not list(parallel_call_roots(run_dir)[0].glob("units/*/unit-spec.json"))
    assert all(row["marker"] == "parent" for row in endpoint.requests)
    assert_shipped_wire(parent_wire_files(run_dir))


def test_model_facing_wire_names_use_subframe_term(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    repo = _repo(tmp_path / "repo", {"pkg/a.py": "VALUE = 0\n", "pkg/b.py": "VALUE = 0\n"})
    parent = [
        tool_step("parallel", {"subframes": [_contract("a", "pkg/a.py"), _contract("b", "pkg/b.py")]}),
        tool_step("done", {"summary": "The subframe naming check completed."}),
    ]
    subframes = {
        unit_id: [
            tool_step("str_replace", {"path": f"pkg/{unit_id}.py", "old_str": "VALUE = 0",
                                       "new_str": "VALUE = 1"}),
            tool_step("done", {"summary": f"Subframe {unit_id} completed."}),
        ]
        for unit_id in ("a", "b")
    }
    run_dir = tmp_path / "run"
    with BarrierEndpoint(parent, subframes) as endpoint:
        completed = run_cli(isolated_engine_python, endpoint.url, repo, run_dir,
                            "Update these independent modules in separate subframes.", parallel="2")
        assert completed.returncode == 0, completed.stdout + completed.stderr
        endpoint.assert_consumed()

    violations: list[str] = []
    parent_paths = parent_wire_files(run_dir)
    assert_shipped_wire(parent_paths)
    parent_parallel_schemas = 0
    for path in parent_paths:
        tools = read_json(path).get("tools")
        assert isinstance(tools, list)
        for tool in tools:
            if not isinstance(tool, dict) or not isinstance(tool.get("function"), dict):
                continue
            function = tool["function"]
            if function.get("name") in {"parallel", "subframe_blocked"}:
                assert not _uses_legacy_unit_term(json.dumps(function))
            for field in ("name", "description"):
                value = function.get(field)
                if isinstance(value, str) and _uses_legacy_unit_term(value):
                    violations.append(f"parent schema {field} {value!r}")
            if function.get("name") == "parallel":
                parent_parallel_schemas += 1
                parameters = function.get("parameters")
                properties = parameters.get("properties") if isinstance(parameters, dict) else None
                if not isinstance(properties, dict) or "subframes" not in properties:
                    violations.append("parallel schema is missing the 'subframes' argument")
                if isinstance(properties, dict) and "units" in properties:
                    violations.append("parallel schema exposes the 'units' argument")
    assert parent_parallel_schemas > 0

    children = subframe_directories(run_dir)
    assert set(children) == {"a", "b"}
    for subframe_id, directory in children.items():
        child_paths = subframe_wire_files(directory)
        assert child_paths
        task_contents = [message["content"]
                         for path in child_paths
                         for message in read_json(path).get("messages", [])
                         if isinstance(message, dict) and isinstance(message.get("content"), str)
                         and f"Subframe {subframe_id}:" in message["content"]]
        assert task_contents, f"subframe {subframe_id} task was not present in wire bytes"
        assert all(not _uses_legacy_unit_term(content) for content in task_contents)
        for path in child_paths:
            assert_shipped_wire([path], task_marker=f"Subframe {subframe_id}:")
            tools = read_json(path).get("tools")
            assert isinstance(tools, list)
            for tool in tools:
                if not isinstance(tool, dict) or not isinstance(tool.get("function"), dict):
                    continue
                function = tool["function"]
                for field in ("name", "description"):
                    value = function.get(field)
                    if isinstance(value, str) and _uses_legacy_unit_term(value):
                        violations.append(f"subframe schema {field} {value!r}")
    result, = result_records(run_dir).values()
    assert result.observation
    assert not _uses_legacy_unit_term(result.observation), result.observation
    # The parent wire carries the observation as the protected fan-out disposition (a compact
    # result_summary), so each recorded plan line must appear there rather than the verbatim text.
    plan_lines = [line for line in result.observation.splitlines() if line.startswith("Plan ")]
    assert plan_lines
    wire_text = "\n".join(wire_message_text(path) for path in parent_paths)
    for line in plan_lines:
        assert line in wire_text, f"recorded plan line is absent from the saved parent wire: {line}"
    assert not violations, "; ".join(violations)


def test_subframe_blocked_tool_uses_subframe_terminal(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    repo = _repo(tmp_path / "repo", {"pkg/a.py": "VALUE = 0\n", "pkg/b.py": "VALUE = 0\n"})
    parent = [
        tool_step("parallel", {"subframes": [_contract("a", "pkg/a.py"), _contract("b", "pkg/b.py")]}),
        tool_step("done", {"summary": "Both subframes reported their contract blockers."}),
    ]
    run_dir = tmp_path / "run"
    subframes = {
        subframe_id: [tool_step("subframe_blocked", {
            "reason_code": "OTHER", "detail": "The subframe cannot complete its contract."})]
        for subframe_id in ("a", "b")
    }
    with BarrierEndpoint(parent, subframes) as endpoint:
        run_cli(isolated_engine_python, endpoint.url, repo, run_dir,
                "Run two subframes that report blocked contracts.", parallel="2")
        endpoint.assert_consumed()

    children = subframe_directories(run_dir)
    assert set(children) == {"a", "b"}
    for directory in children.values():
        terminal = read_json(directory / "engine-result.json")["result"]
        assert terminal["result"] == "SUBFRAME_BLOCKED"
        assert terminal["reason_code"] == "OTHER"
        assert terminal["detail"] == "The subframe cannot complete its contract."
        requests = subframe_wire_files(directory)
        assert requests
        from stateframe.resume import ResumeRefused, assess
        with pytest.raises(ResumeRefused, match="subframe run already finished blocked: OTHER"):
            assess(directory)
        assert read_json(directory / "engine-result.json")["result"] == terminal
        assert subframe_wire_files(directory) == requests
        assert any(
            tool["function"]["name"] == "subframe_blocked"
            for request in requests
            for tool in read_json(request)["tools"]
            if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
        )
    result, = result_records(run_dir).values()
    assert "Subframe a: BLOCKED" in result.observation
    assert "Subframe b: BLOCKED" in result.observation
    assert not _uses_legacy_unit_term(result.observation)
