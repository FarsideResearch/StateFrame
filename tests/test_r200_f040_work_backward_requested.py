"""Outcome milestones qualify against public brief evidence on the provider wire."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.pinned_tree import pinned_tree


ROOT = Path(__file__).resolve().parents[1]
BASE = "6e3b096cf3059f759138a88f7fdf46acd3d75baa"
SWITCH = "STATEFRAME_OUTCOME_MILESTONES"
BRIEF = (
    "Make percentage discounts return the discounted price. "
    "Allow shoppers to see prices rounded to two decimal places."
)


_PROVIDER_DRIVER = """
import io
import json
import shutil
import sys
from pathlib import Path

import test_checkpoint_b_resume_fidelity as fixture

work, mode, brief = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
repo = work / "repo"
shutil.copytree(fixture.ROOT / "examples/discount_bug", repo)
profile = fixture._write_profile(work / "profile.json")
document = json.loads(profile.read_text())
document["switches"]["STATEFRAME_HOST_CLOSURE"] = "off"
if mode != "unset":
    document["switches"]["STATEFRAME_OUTCOME_MILESTONES"] = mode
profile.write_text(json.dumps(document))
launcher = fixture._write_launcher(work / "launch.py")
start_server = fixture.serve
requests = []

def capture_raw_requests(script):
    server = start_server(script)

    class RawHandler(server.RequestHandlerClass):
        def do_POST(self):
            if self.path.endswith("/chat/completions"):
                stream = self.rfile
                raw = stream.read(int(self.headers["Content-Length"]))
                requests.append(raw)
                self.rfile = io.BytesIO(raw)
                try:
                    super().do_POST()
                finally:
                    self.rfile = stream
            else:
                super().do_POST()

    server.RequestHandlerClass = RawHandler
    return server

fixture.serve = capture_raw_requests
assert_restored = fixture._assert_checkpoint_restored

def assert_resumed(run_dir, sends):
    resume = json.loads((run_dir / "spec.json").read_text())["resume"]
    if "checkpoint" in resume:
        assert_restored(run_dir, sends)
    else:
        assert len(resume["tape"]) == sends
        assert resume["pending"]["send"] == sends + 1
        assert not (run_dir / "resume-divergence.json").exists()

fixture._assert_checkpoint_restored = assert_resumed
script = fixture._script_document("discount_bug")
if mode == "on":
    script["steps"][1] = {
        "tool": "work", "required": True, "arguments": {"ops": [
            {"op": "add_step", "goal_id": "goal-0", "text": "Wire the checkout caller."},
            {"op": "add_step", "goal_id": "goal-0",
             "text": "User action: Apply a discount | Expected result: Checkout shows the discounted price"},
            {"op": "add_field", "record_id": "step-3", "key": "outcome_milestone",
             "value": {"source_clauses": ["obl-000"], "prerequisites": ["step-2"]}},
        ]},
    }
interrupted = fixture._run_interrupted(
    "discount_bug", repo, work / "run", launcher, profile, work / "home",
    completed_actions=2, task=brief, script_document=script,
)
assert len(requests) == 4, len(requests)
assert requests[2] == interrupted.pending_request
assert requests[3] == interrupted.resumed_request
for index, raw in enumerate(requests):
    (work / f"provider-{index}.json").write_bytes(raw)
"""


def _provider_requests(tree, work, mode):
    work.mkdir()
    env = {name: value for name, value in os.environ.items()
           if name in {"PATH", "LANG", "TMPDIR"}}
    env["PYTHONPATH"] = os.pathsep.join(str(tree / path) for path in (
        "src", "src/stateframe/engine", "tests",
    ))
    result = subprocess.run(
        [sys.executable, "-c", _PROVIDER_DRIVER, str(work), mode, BRIEF],
        cwd=tree, env=env, capture_output=True, text=True, timeout=480, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return tuple((work / f"provider-{index}.json").read_bytes() for index in range(4))


@pytest.mark.host_sandbox
def test_f040_work_backward_requested_off_request_bytes_match_pinned_099(tmp_path, tmp_path_factory):
    base = pinned_tree(BASE, tmp_path_factory)
    expected = _provider_requests(base, tmp_path / "base", "unset")
    assert all(b"[StateFrame]" in wire and b"[OUTCOME MILESTONES]" not in wire
               for wire in expected)
    assert b"str_replace" in expected[2] and b"100 - percent" in expected[2]
    assert expected[2] == expected[3]
    for mode in ("unset", "off"):
        assert _provider_requests(ROOT, tmp_path / mode, mode) == expected


@pytest.mark.host_sandbox
def test_f040_disabled_wire_matches_pinned_after_capacity_retry(
        tmp_path, tmp_path_factory, monkeypatch):
    import test_capacity_400_acceptance as capacity_fixture

    base = pinned_tree(BASE, tmp_path_factory)

    def retry_case(tree, work):
        with monkeypatch.context() as patcher:
            patcher.setattr(capacity_fixture, "ROOT", tree)
            return capacity_fixture._run_case(
                work, (capacity_fixture.CAPACITY_ERROR,),
                task=f"{BRIEF} Fix app.py so VALUE is 7.",
            )

    expected = retry_case(base, tmp_path / "base-capacity-retry")
    actual = retry_case(ROOT, tmp_path / "current-capacity-retry")
    assert len(expected.requests) == len(actual.requests) == 2
    assert BRIEF.encode() in expected.requests[0]
    assert len(expected.requests[1]) < len(expected.requests[0])
    assert actual.requests == expected.requests


def test_f040_work_backward_requested_switch_defaults_off(monkeypatch):
    from kernel_obligation_seed import outcome_milestones_enabled
    from stateframe import profiles, runner

    monkeypatch.delenv(SWITCH, raising=False)
    assert not outcome_milestones_enabled()
    assert SWITCH not in profiles.load(profiles.DEFAULT)
    assert runner.OPT_IN_SWITCHES[SWITCH] == ("off", "on")
    assert SWITCH not in runner.resolve_opt_in_switches()
    for setting in ("off", "on"):
        monkeypatch.setenv(SWITCH, setting)
        assert outcome_milestones_enabled() is (setting == "on")
        assert runner.resolve_opt_in_switches()[SWITCH] == setting


def test_f040_health_check_endpoint_is_a_public_outcome(monkeypatch):
    from kernel_obligation_seed import outcome_milestone_sources

    monkeypatch.setenv("STATEFRAME_OBLIGATION_SEED", "on")
    brief = "Add a health check endpoint. Allow admins to download audit logs."
    assert outcome_milestone_sources(brief) == {
        "obl-000": "Add a health check endpoint",
        "obl-001": "Allow admins to download audit logs",
    }
    assert not outcome_milestone_sources(
        "Check that the audit endpoint responds. Add a health check endpoint."
    )


def test_f040_user_facing_test_command_is_a_public_outcome(monkeypatch):
    from kernel_obligation_seed import outcome_milestone_sources

    monkeypatch.setenv("STATEFRAME_OBLIGATION_SEED", "on")
    brief = (
        "Add a CLI command to run tests. "
        "Add a CLI command to export results as JSON."
    )
    assert outcome_milestone_sources(brief) == {
        "obl-000": "Add a CLI command to run tests",
        "obl-001": "Add a CLI command to export results as JSON",
    }


def test_f040_verification_clause_does_not_count_as_a_public_outcome(monkeypatch):
    from kernel_obligation_seed import outcome_milestone_sources

    monkeypatch.setenv("STATEFRAME_OBLIGATION_SEED", "on")
    brief = (
        "Add a CLI command to export results as JSON. "
        "Add tests for the CLI command."
    )
    assert outcome_milestone_sources(brief) == {}


def test_f040_regression_tests_do_not_count_as_a_second_public_outcome(monkeypatch):
    from kernel_obligation_seed import outcome_milestone_sources

    monkeypatch.setenv("STATEFRAME_OBLIGATION_SEED", "on")
    brief = (
        "Add a CLI command to run tests. "
        "Add regression tests to verify the CLI can run tests."
    )
    assert outcome_milestone_sources(brief) == {}


def test_f040_cli_command_to_list_failing_tests_is_a_public_outcome(monkeypatch):
    from kernel_obligation_seed import outcome_milestone_sources

    monkeypatch.setenv("STATEFRAME_OBLIGATION_SEED", "on")
    brief = (
        "Add a CLI command to list failing tests. "
        "Add a CLI command to export results as JSON."
    )
    assert outcome_milestone_sources(brief) == {
        "obl-000": "Add a CLI command to list failing tests",
        "obl-001": "Add a CLI command to export results as JSON",
    }


def test_f040_regression_test_for_failing_test_flag_does_not_create_second_outcome(monkeypatch):
    from kernel_obligation_seed import outcome_milestone_sources

    monkeypatch.setenv("STATEFRAME_OBLIGATION_SEED", "on")
    brief = (
        "Add a CLI flag to list failing tests. "
        "Add a regression test that the flag lists only failing tests in the CLI output."
    )
    assert outcome_milestone_sources(brief) == {}
    # The test's assertion may itself describe the CLI behavior. Keep it
    # verification work even when that wording resembles a public runner.
    assert outcome_milestone_sources(
        "Add a CLI flag to list failing tests. "
        "Add a regression test to verify the CLI flag lists only failing tests."
    ) == {}


def test_f040_panel_budget_reserves_tokens_for_pending_source(monkeypatch):
    import kernel_request_budget
    from kernel_stateframe_framefit5 import assemble_framefit5_request
    from kernel_work_record import OUTCOME_MILESTONE_PREFIX, WORK_TOOL, WorkRecordStore
    from test_framefit_off_literal_suffix import EXCHANGE, STATE, SYSTEM

    monkeypatch.setenv("STATEFRAME_OBLIGATION_SEED", "on")
    monkeypatch.setenv("STATEFRAME_WORK_RECORD", "on")
    record = WorkRecordStore(BRIEF)
    prerequisite = record.apply([{
        "op": "add_step", "goal_id": "goal-0", "text": "Wire the checkout caller.",
    }])["created"][0]
    outcome = record.apply([{
        "op": "add_step", "goal_id": "goal-0",
        "text": "User action: Apply a discount | Expected result: Checkout shows the discounted price",
    }])["created"][0]
    record.apply([{
        "op": "add_field", "record_id": outcome, "key": "outcome_milestone",
        "value": {"source_clauses": ["obl-000"], "prerequisites": [prerequisite]},
    }])
    source = "PENDING REQUIRED SOURCE: checkout price formatting"
    capture = SimpleNamespace(
        request_id="f040-source-token-reserve", _frame_candidate_counter=None,
        _frame_candidate_measurement=None, _capacity_retry_limits=None,
    )
    obligation = SimpleNamespace(identity=SimpleNamespace(
        path="checkout.py", byte_start=0, byte_end=len(source.encode("utf-8")),
    ))
    runtime = SimpleNamespace(
        owner=SimpleNamespace(information_policy=SimpleNamespace(ordinary_brief=BRIEF)),
        work_record=record, frame_ceiling=8_000, _active_capture=capture,
        working_set_state=SimpleNamespace(active_obligations=lambda: (obligation,)),
    )
    contract = {"admitted_tool_schemas": [WORK_TOOL]}

    class ScriptedEightKMeter:
        def measure(self, body):
            serialized = json.dumps(body, ensure_ascii=False)
            has_panel = OUTCOME_MILESTONE_PREFIX in serialized
            has_source = source in serialized
            if has_panel and has_source:
                return 8_004, "tokenize"
            if has_panel:
                return 7_504, "tokenize"
            if has_source:
                return 500, "tokenize"
            return 7_004, "tokenize"

    monkeypatch.setattr(kernel_request_budget, "_request_token_meter",
                        lambda _runtime, _capture: ScriptedEightKMeter())
    monkeypatch.delenv(SWITCH, raising=False)
    task = {"role": "user", "content": f"Task: {BRIEF}"}
    prepared = [SYSTEM, task, STATE]
    transcript = [SYSTEM, task, *EXCHANGE]
    baseline = assemble_framefit5_request(
        prepared, transcript, runtime=runtime, owner_contract=contract,
        sources={"checkout.py": source},
    )
    monkeypatch.setenv(SWITCH, "on")
    enabled = assemble_framefit5_request(
        prepared, transcript, runtime=runtime, owner_contract=contract,
        sources={"checkout.py": source},
    )
    assert enabled == baseline
    assert OUTCOME_MILESTONE_PREFIX not in json.dumps(enabled, ensure_ascii=False)


def test_f040_panel_fit_prices_rendered_required_source_window(monkeypatch):
    import hashlib
    import kernel_request_budget
    from kernel_stateframe_framefit5 import assemble_framefit5_request
    from kernel_work_record import OUTCOME_MILESTONE_PREFIX, WORK_TOOL, WorkRecordStore
    from test_framefit_off_literal_suffix import EXCHANGE, STATE, SYSTEM

    monkeypatch.setenv("STATEFRAME_OBLIGATION_SEED", "on")
    monkeypatch.setenv("STATEFRAME_WORK_RECORD", "on")
    monkeypatch.setenv("STATEFRAME_EVICTFIX", "off")
    monkeypatch.setenv(SWITCH, "on")
    source = "".join(f"line_{index} = {index}\n" for index in range(1, 41))
    source_bytes = source.encode("utf-8")
    record = WorkRecordStore(BRIEF)
    prerequisite = record.apply([{
        "op": "add_step", "goal_id": "goal-0", "text": "Wire the checkout caller.",
    }])["created"][0]
    outcome = record.apply([{
        "op": "add_step", "goal_id": "goal-0",
        "text": "User action: Apply a discount | Expected result: Checkout shows the discounted price",
    }])["created"][0]
    record.apply([{
        "op": "add_field", "record_id": outcome, "key": "outcome_milestone",
        "value": {"source_clauses": ["obl-000"], "prerequisites": [prerequisite]},
    }])
    obligation = SimpleNamespace(identity=SimpleNamespace(
        path="shop/pricing.py", revision="sha256:" + hashlib.sha256(source_bytes).hexdigest(),
        first_line=1, last_line=40, byte_start=0, byte_end=len(source_bytes),
    ))
    capture = SimpleNamespace(
        request_id="f040-rendered-source-budget", _frame_candidate_counter=None,
        _frame_candidate_measurement=None, _capacity_retry_limits=None,
    )
    runtime = SimpleNamespace(
        owner=SimpleNamespace(information_policy=SimpleNamespace(ordinary_brief=BRIEF)),
        work_record=record, working_set_state=SimpleNamespace(
            active_obligations=lambda: (obligation,),
        ), frame_ceiling=8_000, _active_capture=capture,
    )
    contract = {"admitted_tool_schemas": [WORK_TOOL]}
    task = {"role": "user", "content": f"Task: {BRIEF}"}
    prepared = [SYSTEM, task, STATE]
    transcript = [SYSTEM, task, *EXCHANGE]

    class SourceWindowMeter:
        def measure(self, body):
            serialized = json.dumps(body, ensure_ascii=False)
            panel = OUTCOME_MILESTONE_PREFIX in serialized
            rendered_window = "[Model source window]" in serialized
            source_present = "line_1 = 1" in serialized
            if panel and rendered_window:
                return 8_345, "tokenize"
            if panel and source_present:
                return 8_000, "tokenize"
            if rendered_window:
                return 7_893, "tokenize"
            if source_present:
                return 7_500, "tokenize"
            return 7_000, "tokenize"

    monkeypatch.setattr(kernel_request_budget, "_request_token_meter",
                        lambda _runtime, _capture: SourceWindowMeter())
    messages = assemble_framefit5_request(
        prepared, transcript, runtime=runtime, owner_contract=contract,
        sources={"shop/pricing.py": source},
    )
    assert OUTCOME_MILESTONE_PREFIX not in json.dumps(messages, ensure_ascii=False)


def test_f040_panel_fit_reserves_fitter_read_summary(monkeypatch):
    import hashlib
    import kernel_request_budget
    from kernel_stateframe_framefit5 import assemble_framefit5_request
    from kernel_work_record import OUTCOME_MILESTONE_PREFIX, WORK_TOOL, WorkRecordStore
    from test_framefit_off_literal_suffix import EXCHANGE, STATE, SYSTEM

    monkeypatch.setenv("STATEFRAME_OBLIGATION_SEED", "on")
    monkeypatch.setenv("STATEFRAME_WORK_RECORD", "on")
    monkeypatch.setenv("STATEFRAME_EVICTFIX", "off")
    monkeypatch.setenv(SWITCH, "on")
    source = "".join(f"line_{index} = {index}\n" for index in range(1, 41))
    source_bytes = source.encode("utf-8")
    read_summary = f"kept 1 recent reads; about {len(source)} chars of source fit"
    record = WorkRecordStore(BRIEF)
    prerequisite = record.apply([{
        "op": "add_step", "goal_id": "goal-0", "text": "Wire the checkout caller.",
    }])["created"][0]
    outcome = record.apply([{
        "op": "add_step", "goal_id": "goal-0",
        "text": "User action: Apply a discount | Expected result: Checkout shows the discounted price",
    }])["created"][0]
    record.apply([{
        "op": "add_field", "record_id": outcome, "key": "outcome_milestone",
        "value": {"source_clauses": ["obl-000"], "prerequisites": [prerequisite]},
    }])
    obligation = SimpleNamespace(identity=SimpleNamespace(
        path="shop/pricing.py", revision="sha256:" + hashlib.sha256(source_bytes).hexdigest(),
        first_line=1, last_line=40, byte_start=0, byte_end=len(source_bytes),
    ))
    capture = SimpleNamespace(
        request_id="f040-fitter-read-summary-budget", _frame_candidate_counter=None,
        _frame_candidate_measurement=None, _capacity_retry_limits=None,
    )
    runtime = SimpleNamespace(
        owner=SimpleNamespace(information_policy=SimpleNamespace(ordinary_brief=BRIEF)),
        work_record=record, working_set_state=SimpleNamespace(
            active_obligations=lambda: (obligation,),
        ), frame_ceiling=8_192, _active_capture=capture,
    )
    contract = {"admitted_tool_schemas": [WORK_TOOL]}
    task = {"role": "user", "content": f"Task: {BRIEF}"}
    prepared = [SYSTEM, task, STATE]
    transcript = [SYSTEM, task, *EXCHANGE]

    class ReadSummaryMeter:
        def measure(self, body):
            serialized = json.dumps(body, ensure_ascii=False)
            panel = OUTCOME_MILESTONE_PREFIX in serialized
            window = "[Model source window]" in serialized
            summary = read_summary in serialized
            if panel and window and summary:
                return 8_193, "tokenize"
            if panel and window:
                return 8_152, "tokenize"
            if window and summary:
                return 7_760, "tokenize"
            if window:
                return 7_500, "tokenize"
            return 7_000, "tokenize"

    monkeypatch.setattr(kernel_request_budget, "_request_token_meter",
                        lambda _runtime, _capture: ReadSummaryMeter())
    messages = assemble_framefit5_request(
        prepared, transcript, runtime=runtime, owner_contract=contract,
        sources={"shop/pricing.py": source},
    )
    assert OUTCOME_MILESTONE_PREFIX not in json.dumps(messages, ensure_ascii=False)


def test_f040_split_prerequisite_preserves_advisory_milestone(monkeypatch):
    from kernel_obligation_seed import outcome_milestone_sources
    from kernel_verdict import open_work_ids
    from kernel_work_record import OUTCOME_MILESTONE_PREFIX, WorkRecordStore, render_work_record

    monkeypatch.setenv("STATEFRAME_OBLIGATION_SEED", "on")
    monkeypatch.setenv("STATEFRAME_WORK_RECORD", "on")
    monkeypatch.setenv(SWITCH, "on")
    record = WorkRecordStore(BRIEF)
    prerequisite = record.apply([{
        "op": "add_step", "goal_id": "goal-0", "text": "Wire the checkout caller.",
    }])["created"][0]
    outcome = record.apply([{
        "op": "add_step", "goal_id": "goal-0",
        "text": "User action: Apply a discount | Expected result: Checkout shows the discounted price",
    }])["created"][0]
    record.apply([{
        "op": "add_field", "record_id": outcome, "key": "outcome_milestone",
        "value": {"source_clauses": ["obl-000"], "prerequisites": [prerequisite]},
    }])
    clauses = outcome_milestone_sources(BRIEF)

    before = render_work_record(record, compact=True, outcome_sources=clauses)
    assert OUTCOME_MILESTONE_PREFIX in before
    assert outcome not in open_work_ids(record)

    split = record.apply([{
        "op": "split_step", "id": prerequisite,
        "into": ["Locate the checkout caller.", "Wire the price formatter."],
    }])
    children = split["created"]
    after = render_work_record(record, compact=True, outcome_sources=clauses)
    assert OUTCOME_MILESTONE_PREFIX in after
    panel = json.loads(after.removeprefix(OUTCOME_MILESTONE_PREFIX))
    assert [item["id"] for item in panel["prerequisites"]] == children
    assert [item["text"] for item in panel["prerequisites"]] == [
        "Locate the checkout caller.", "Wire the price formatter.",
    ]
    assert outcome not in open_work_ids(record)


def test_f040_hides_panel_when_stable_head_lists_uncallable_work(monkeypatch):
    from kernel_stateframe_framefit5 import assemble_framefit5_request
    from kernel_work_record import OUTCOME_MILESTONE_PREFIX, WORK_TOOL, WorkRecordStore
    from test_framefit_off_literal_suffix import EXCHANGE, STATE, SYSTEM

    monkeypatch.setenv("STATEFRAME_OBLIGATION_SEED", "on")
    monkeypatch.setenv("STATEFRAME_WORK_RECORD", "on")
    monkeypatch.setenv(SWITCH, "on")
    record = WorkRecordStore(BRIEF)
    prerequisite = record.apply([{
        "op": "add_step", "goal_id": "goal-0", "text": "Wire the checkout caller.",
    }])["created"][0]
    outcome = record.apply([{
        "op": "add_step", "goal_id": "goal-0",
        "text": "User action: Apply a discount | Expected result: Checkout shows the discounted price",
    }])["created"][0]
    record.apply([{
        "op": "add_field", "record_id": outcome, "key": "outcome_milestone",
        "value": {"source_clauses": ["obl-000"], "prerequisites": [prerequisite]},
    }])
    runtime = SimpleNamespace(
        owner=SimpleNamespace(information_policy=SimpleNamespace(ordinary_brief=BRIEF)),
        work_record=record, frame_ceiling=40_000,
    )
    read_tool = {"type": "function", "function": {
        "name": "read", "description": "Read source files.",
        "parameters": {"type": "object", "properties": {}, "required": [],
                       "additionalProperties": False},
    }}
    contract = {
        "admitted_tool_schemas": [read_tool, WORK_TOOL],
        "advertised_actions": ["read", "work"],
        "callable_actions": ["read"],
    }
    task = {"role": "user", "content": f"Task: {BRIEF}"}
    prepared = [SYSTEM, task, STATE]
    transcript = [SYSTEM, task, *EXCHANGE]

    messages = assemble_framefit5_request(
        prepared, transcript, runtime=runtime, owner_contract=contract,
    )
    assert OUTCOME_MILESTONE_PREFIX not in messages[-1]["content"]


def test_f040_work_backward_requested_panel_does_not_displace_reasoning(monkeypatch):
    import kernel_request_budget
    from kernel_stateframe_framefit5 import assemble_framefit5_request
    from kernel_work_record import OUTCOME_MILESTONE_PREFIX, WORK_TOOL, WorkRecordStore
    from test_framefit_off_literal_suffix import EXCHANGE, STATE, SYSTEM

    monkeypatch.setenv("STATEFRAME_OBLIGATION_SEED", "on")
    monkeypatch.setenv("STATEFRAME_WORK_RECORD", "on")
    record = WorkRecordStore(BRIEF)
    prerequisite = record.apply([{
        "op": "add_step", "goal_id": "goal-0", "text": "Wire the checkout caller.",
    }])["created"][0]
    outcome = record.apply([{
        "op": "add_step", "goal_id": "goal-0",
        "text": "User action: Apply a discount | Expected result: Checkout shows the discounted price",
    }])["created"][0]
    record.apply([{
        "op": "add_field", "record_id": outcome, "key": "outcome_milestone",
        "value": {"source_clauses": ["obl-000"], "prerequisites": [prerequisite]},
    }])
    notes = tuple(f"Reasoning note {index}: preserve the verified decision {index}."
                  for index in range(1, 5))
    system = {**SYSTEM, "content": SYSTEM["content"] + "\n\n" + "\n".join(notes)}
    task = {"role": "user", "content": f"Task: {BRIEF}"}
    prepared = [system, task, STATE]
    transcript = [system, task, *EXCHANGE]

    class ScriptedEightKMeter:
        def measure(self, body):
            serialized = json.dumps(body, ensure_ascii=False)
            return (7_950 if OUTCOME_MILESTONE_PREFIX in serialized else 7_900, "tokenize")

    runtime = SimpleNamespace(
        owner=SimpleNamespace(information_policy=SimpleNamespace(ordinary_brief=BRIEF)),
        work_record=record,
        frame_ceiling=8_000,
        _active_capture=SimpleNamespace(
            request_id="f040-scripted-eight-k", _frame_candidate_counter=None,
            _frame_candidate_measurement=None,
        ),
    )
    contract = {"admitted_tool_schemas": [WORK_TOOL]}
    meter = ScriptedEightKMeter()
    baseline = assemble_framefit5_request(
        prepared, transcript, runtime=runtime, owner_contract=contract,
    )
    monkeypatch.setattr("kernel_context.token_estimate", lambda _body: 7_900)
    monkeypatch.setattr(kernel_request_budget, "_request_token_meter",
                        lambda _runtime, _capture: meter)
    monkeypatch.setenv("STATEFRAME_EVICTFIX", "on")
    monkeypatch.setenv(SWITCH, "on")
    enabled = assemble_framefit5_request(
        prepared, transcript, runtime=runtime, owner_contract=contract,
    )
    body = {"messages": enabled, "tools": contract["admitted_tool_schemas"]}
    assert meter.measure({"messages": baseline, "tools": contract["admitted_tool_schemas"]})[0] <= 8_000
    assert meter.measure(body)[0] <= 8_000
    assert all(note in json.dumps(enabled, ensure_ascii=False) for note in notes)
    assert OUTCOME_MILESTONE_PREFIX not in json.dumps(enabled, ensure_ascii=False)


def test_f040_milestone_reassembly_keeps_frozen_provider_model(monkeypatch):
    import kernel_request_budget
    from kernel_frame_allocation_capture import FrozenRequestCounter
    from kernel_stateframe_framefit5 import assemble_framefit5_request
    from kernel_work_record import OUTCOME_MILESTONE_PREFIX, WORK_TOOL, WorkRecordStore
    from test_framefit_off_literal_suffix import EXCHANGE, STATE, SYSTEM

    monkeypatch.setenv("STATEFRAME_OBLIGATION_SEED", "on")
    monkeypatch.setenv("STATEFRAME_WORK_RECORD", "on")
    record = WorkRecordStore(BRIEF)
    prerequisite = record.apply([{
        "op": "add_step", "goal_id": "goal-0", "text": "Wire the checkout caller.",
    }])["created"][0]
    outcome = record.apply([{
        "op": "add_step", "goal_id": "goal-0",
        "text": "User action: Apply a discount | Expected result: Checkout shows the discounted price",
    }])["created"][0]
    record.apply([{
        "op": "add_field", "record_id": outcome, "key": "outcome_milestone",
        "value": {"source_clauses": ["obl-000"], "prerequisites": [prerequisite]},
    }])
    runtime = SimpleNamespace(
        owner=SimpleNamespace(information_policy=SimpleNamespace(ordinary_brief=BRIEF)),
        work_record=record,
        frame_ceiling=40_000,
        journal=SimpleNamespace(manifest={"model": "fixture-model"}),
    )
    task = {"role": "user", "content": f"Task: {BRIEF}"}
    prepared = [SYSTEM, task, STATE]
    transcript = [SYSTEM, task, *EXCHANGE]
    contract = {"admitted_tool_schemas": [WORK_TOOL]}

    class ModelBoundMeter:
        _request_id = "f040-model-reassembly"
        _evictfix_token_pair = None

        @staticmethod
        def tokenizer_identity(body):
            return f"model:{body.get('model', '<missing>')}"

    meter = ModelBoundMeter()

    def assemble(rows):
        return assemble_framefit5_request(
            rows, transcript, runtime=runtime, owner_contract=contract,
        )

    monkeypatch.delenv(SWITCH, raising=False)
    baseline = assemble(prepared)
    frozen_body = {"model": "fixture-model", "messages": baseline, "tools": [WORK_TOOL]}
    capture = SimpleNamespace(
        request_id="f040-model-reassembly",
        _frame_candidate_counter=FrozenRequestCounter.capture(
            frozen_body, meter, 200, "calibrated_bytes", "f040-model-reassembly",
        ),
        _capacity_retry_limits=None,
    )
    runtime._active_capture = capture
    monkeypatch.setattr(kernel_request_budget, "_request_token_meter",
                        lambda _runtime, _capture: meter)
    monkeypatch.setenv(SWITCH, "on")

    first = assemble(prepared)
    reassembled = assemble(first)
    assert OUTCOME_MILESTONE_PREFIX in first[-1]["content"]
    assert reassembled == first


def test_f040_work_backward_requested_on_path_is_bounded_and_idempotent(monkeypatch):
    from kernel_context import token_estimate
    from kernel_greenfact import GREENFACT_TEXT
    from kernel_obligation_seed import outcome_milestone_sources
    from kernel_stateframe_framefit5 import VOLATILE_TAIL_MARKER, assemble_framefit5_request
    from kernel_verdict import open_work_ids
    from kernel_work_record import (
        OUTCOME_MILESTONE_PREFIX, WORK_INTENT_TOOL, WORK_TOOL, WorkRecordStore,
        open_obligations,
    )
    from test_framefit_off_literal_suffix import EXCHANGE, STATE, SYSTEM

    monkeypatch.setenv("STATEFRAME_OBLIGATION_SEED", "on")
    monkeypatch.setenv("STATEFRAME_WORK_RECORD", "on")
    record = WorkRecordStore(BRIEF)
    clauses = outcome_milestone_sources(BRIEF)
    first, second = clauses

    def step(text):
        reply = record.apply([{"op": "add_step", "goal_id": "goal-0", "text": text}])
        return reply["created"][0]

    prerequisite = step("Wire the price formatter to the checkout caller.")
    outcome = step("User action: Apply a discount | Expected result: Checkout shows the new price")
    later = step("User action: View a rounded price | Expected result: Two decimal places appear")
    for step_id, clause_id in ((outcome, first), (later, second)):
        record.apply([{"op": "add_field", "record_id": step_id, "key": "outcome_milestone",
                       "value": {"source_clauses": [clause_id], "prerequisites": [prerequisite]}}])
    runtime = SimpleNamespace(
        owner=SimpleNamespace(information_policy=SimpleNamespace(ordinary_brief=BRIEF)),
        work_record=record, frame_ceiling=40_000,
    )
    task = {"role": "user", "content": f"Task: {BRIEF}"}
    transcript = [SYSTEM, task, *EXCHANGE]
    prepared = [SYSTEM, task, STATE]
    contract = {"admitted_tool_schemas": [WORK_TOOL]}

    def assemble(rows):
        return assemble_framefit5_request(rows, transcript, runtime=runtime, owner_contract=contract)

    monkeypatch.delenv(SWITCH, raising=False)
    baseline = assemble(prepared)
    monkeypatch.setenv(SWITCH, "on")
    enabled = assemble(prepared)
    assert enabled[:-1] == baseline[:-1]
    assert assemble(enabled) == enabled
    tail = enabled[-1]["content"]
    assert tail.count(OUTCOME_MILESTONE_PREFIX) == 1
    fact = tail.split(OUTCOME_MILESTONE_PREFIX)[1]
    assert len((OUTCOME_MILESTONE_PREFIX + fact).encode()) <= 4096
    panel = json.loads(fact)
    assert "earliest point a real user" in panel["rule"]
    assert "first outcome that needs it" in panel["rule"]
    assert "existing checks to verify completion" in panel["rule"]
    assert panel["source_clauses"] == clauses
    assert panel["active_outcome"]["id"] == outcome
    assert panel["active_outcome"]["source_clauses"] == [first]
    assert panel["prerequisites"] == [{
        "id": prerequisite, "status": "todo", "text": "Wire the price formatter to the checkout caller.",
    }]
    assert later not in fact

    evidence = {"type": "text", "text": "Immutable source and reasoning evidence."}
    structured = [*baseline[:-1], {"role": "user", "content": [
        {"type": "text", "text": VOLATILE_TAIL_MARKER + "\n\n"},
        evidence,
        {"type": "text", "text": GREENFACT_TEXT},
    ]}]
    mounted = assemble(structured)
    assert mounted[-1]["content"][1] == evidence
    assert mounted[-1]["content"][-1]["text"] == GREENFACT_TEXT
    assert assemble(mounted) == mounted

    obligations = open_obligations(record)
    work_ids = open_work_ids(record)
    assert prerequisite in work_ids
    assert outcome not in work_ids and later not in work_ids
    record.apply([{"op": "set_status", "id": outcome, "status": "done"}])
    assert open_obligations(record) == obligations
    assert open_work_ids(record) == work_ids
    advanced = assemble(enabled)
    assert json.loads(advanced[-1]["content"].split(OUTCOME_MILESTONE_PREFIX)[1])[
        "active_outcome"]["source_clauses"] == [second]

    runtime.frame_ceiling = token_estimate({"messages": baseline, "tools": []})
    assert assemble(advanced) == baseline
    runtime.frame_ceiling = 40_000
    runtime.owner.information_policy.ordinary_brief = "Make checkout show the new price."
    assert assemble(advanced) == baseline
    runtime.owner.information_policy.ordinary_brief = BRIEF
    runtime.stateframe_host_closure_mode = "on"
    contract["admitted_tool_schemas"] = [WORK_INTENT_TOOL]
    closed = assemble(advanced)
    assert closed[:-1] == baseline[:-1]
    assert OUTCOME_MILESTONE_PREFIX not in closed[-1]["content"]
    runtime.stateframe_host_closure_mode = "off"
    runtime.work_record = None
    assert assemble(advanced) == baseline

    runtime.work_record = record
    record.apply([{"op": "set_status", "id": later, "status": "done"}])
    invalid = step("User action: View checkout | Expected result: A total appears")
    record.apply([{"op": "add_field", "record_id": invalid, "key": "outcome_milestone",
                   "value": {"source_clauses": ["obl-999"], "prerequisites": [invalid]}}])
    assert assemble(advanced) == baseline
    assert invalid in open_work_ids(record)
    assert not outcome_milestone_sources("\n".join(
        "Add export format %d." % index for index in range(13)
    ))


@pytest.mark.host_sandbox
def test_f040_milestone_survives_a_sealed_provider_send_and_resume(tmp_path):
    first, post_read, planned, resumed = _provider_requests(ROOT, tmp_path / "enabled", "on")
    prefix = "[OUTCOME MILESTONES] "
    for wire in (first, post_read, planned, resumed):
        tail = json.loads(wire)["messages"][-1]["content"]
        assert tail.count(prefix) == 1
    for wire in (planned, resumed):
        tail = json.loads(wire)["messages"][-1]["content"]
        panel = json.loads(next(line.removeprefix(prefix) for line in tail.splitlines()
                                if line.startswith(prefix)))
        assert panel["active_outcome"]["source_clauses"] == ["obl-000"]
        assert panel["prerequisites"][0]["text"] == "Wire the checkout caller."
    assert planned == resumed
