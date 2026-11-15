"""Model-facing subframe corrections must lead to a usable contract."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from stateframe.engine.kernel_parallel import ParallelInputError, _complete_arguments, _request
from stateframe.engine.kernel_parallel_admission import DirectorySnapshot


def _repo(tmp_path):
    (tmp_path / "a.py").write_text("VALUE = 1\n")
    (tmp_path / "b.py").write_text("VALUE = 2\n")
    return DirectorySnapshot(tmp_path, 0)


def _ordinary():
    return {"subframes": [
        {"id": "a", "outcome": "Change a", "owns": ["a.py"], "creates": []},
        {"id": "b", "outcome": "Change b", "owns": ["b.py"], "creates": []},
    ]}


def _terse():
    return {"subframes": [
        {"id": "a", "goal": "Change a", "owns": ["a.py"]},
        {"id": "b", "goal": "Change b", "owns": ["b.py"]},
    ]}


def _parse(args, snapshot, *, expand):
    if expand:
        args, actions = _complete_arguments(args, snapshot)
    else:
        actions = ()
    return _request(args, "call", "2"), args, actions


def _fourth_wire_call():
    fixture = Path(__file__).parent / "fixtures/parallel/l50_full_contract_0004.json"
    return json.loads(fixture.read_text(encoding="utf-8"))


def test_fourth_wire_call_uses_first_requirement_and_records_expansions(tmp_path):
    args = _fourth_wire_call()
    assert len(args["subframes"]) == 3
    assert all(set(unit) == {"id", "owns", "reads", "requirements", "check", "model_class"}
               for unit in args["subframes"])

    request, completed, actions = _parse(args, _repo(tmp_path), expand=True)

    # The parsed request orders units canonically, so pair everything by subframe id, not by position.
    parsed_by_id = {unit.id: unit for unit in request.units}
    expanded_by_id = {unit["id"]: unit for unit in completed["subframes"]}
    assert set(parsed_by_id) == set(expanded_by_id) == {unit["id"] for unit in args["subframes"]}
    for original in args["subframes"]:
        parsed, expanded = parsed_by_id[original["id"]], expanded_by_id[original["id"]]
        assert parsed.outcome == original["requirements"][0]
        assert parsed.creates == ()
        assert expanded["outcome"] == original["requirements"][0]
        assert expanded["creates"] == []
        assert {(action["field"], action["reason_code"], action.get("source"))
                for action in actions if action["subframe"] == parsed.id
                and action["field"] in {"outcome", "creates"}} == {
                    ("outcome", "REQUIREMENT_SENTENCE", "requirements[0]"),
                    ("creates", "OPTIONAL_FIELD_DEFAULTED", "default"),
                }


def test_fourth_wire_call_expansions_reach_parent_journal(tmp_path, monkeypatch):
    import stateframe.engine.kernel_parallel as parallel
    from test_parallel_dispatch import setup_context

    args = _fourth_wire_call()
    run, _, context = setup_context(tmp_path, monkeypatch)

    class NoLaunchHost:
        def submit(self, op, scope, payload):
            pytest.fail("a refused admission launched a subframe")

        def poll(self, result_id):
            pytest.fail("a refused admission polled a subframe")

        def cancel(self, scope):
            pytest.fail("a refused admission cancelled a subframe")

    result = parallel.execute_parallel(context, args, host=NoLaunchHost(),
                                       store_root=run / "parallel" / "parallel_1")
    assert result.admission.status == "REFUSED"
    recorded = [value for name, value in context.runtime.journal.events
                if name == "parallel_plan_completed" and value.get("field") in {"outcome", "creates"}]
    assert {(value["subframe"], value["field"], value["reason_code"], value["source"])
            for value in recorded} == {
                (unit["id"], "outcome", "REQUIREMENT_SENTENCE", "requirements[0]")
                for unit in args["subframes"]
            } | {
                (unit["id"], "creates", "OPTIONAL_FIELD_DEFAULTED", "default")
                for unit in args["subframes"]
            }


def test_missing_outcome_and_requirements_has_literal_repair_hint(tmp_path):
    args = _ordinary()
    first = args["subframes"][0]
    first.pop("outcome")
    first.pop("requirements", None)
    with pytest.raises(ParallelInputError) as refusal:
        _parse(args, _repo(tmp_path), expand=True)
    assert refusal.value.field == "outcome"
    assert refusal.value.hint == 'add "outcome": "<one sentence>"'

    first.update(json.loads("{" + refusal.value.hint.removeprefix("add ") + "}"))
    request, _, _ = _parse(args, _repo(tmp_path), expand=True)
    assert request.units[0].outcome == "<one sentence>"


@pytest.mark.parametrize("invalid", ["", 3, None])
def test_present_invalid_outcome_is_never_replaced_by_requirement(tmp_path, invalid):
    args = _fourth_wire_call()
    args["subframes"][0]["outcome"] = invalid
    with pytest.raises(ParallelInputError) as refusal:
        _parse(args, _repo(tmp_path), expand=True)
    assert refusal.value.field == "outcome"


@pytest.mark.parametrize("dropped", ("id", "owns", "reads", "requirements", "check", "model_class"))
def test_fourth_wire_call_field_drop_hint_moves_to_acceptance_or_new_site(tmp_path, dropped):
    args = _fourth_wire_call()
    original = args["subframes"][0].pop(dropped)
    snapshot = _repo(tmp_path)
    try:
        request, _, _ = _parse(args, snapshot, expand=True)
    except ParallelInputError as first:
        assert first.hint
        if dropped in {"id", "owns", "requirements"}:
            assert first.hint.startswith("add ")
        if first.hint.startswith("add "):
            args["subframes"][0].update(json.loads("{" + first.hint[4:] + "}"))
        else:
            assert first.field == dropped
            args["subframes"][0][dropped] = original
        try:
            request, _, _ = _parse(args, snapshot, expand=True)
        except ParallelInputError as second:
            assert (second.unit_id, second.field) != (first.unit_id, first.field)
        else:
            assert len(request.units) == 3
    else:
        assert len(request.units) == 3


def test_distinct_terse_outcome_is_preserved_with_goal_requirement(tmp_path):
    args = _terse()
    args["subframes"][0]["outcome"] = "The corrected value is returned"
    request, completed, actions = _parse(args, _repo(tmp_path), expand=True)
    first = request.units[0]
    assert first.outcome == "The corrected value is returned"
    assert completed["subframes"][0]["requirements"] == ["Change a"]
    assert first.requirement_ids == (
        "planner:b1fdea9ca740",  # sha256("Change a")[:12]
    )
    assert not any(action["subframe"] == "a" and action["field"] == "outcome"
                   for action in actions)
    assert any(action["field"] == "requirements" and action["source"] == "goal"
               for action in actions)


def test_terse_hint_example_defaults_every_omitted_field(tmp_path):
    args = _terse()
    request, completed, actions = _parse(args, _repo(tmp_path), expand=True)
    assert [(unit.id, unit.outcome, unit.owns, unit.creates)
            for unit in request.units] == [
                ("a", "Change a", ("a.py",), ()),
                ("b", "Change b", ("b.py",), ()),
            ]
    assert [unit["creates"] for unit in completed["subframes"]] == [[], []]
    assert [(action["subframe"], action["field"], action["action"])
            for action in actions if action["field"] == "creates"] == [
                ("a", "creates", "expanded_field"),
                ("b", "creates", "expanded_field"),
            ]


def test_explicit_terse_checks_and_exports_bypass_unneeded_test_scan(tmp_path):
    snapshot = _repo(tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_broken.py").write_text("def broken(:\n")
    args = _terse()
    with pytest.raises(ParallelInputError) as refusal:
        _parse(args, snapshot, expand=True)
    assert refusal.value.field == "check"
    assert refusal.value.hint == "provide check.tests and exports explicitly for each subframe"
    for unit in args["subframes"]:
        unit["check"] = {"tests": []}
        unit["exports"] = []
    request, _, _ = _parse(args, snapshot, expand=True)
    assert [(unit.id, unit.check.selectors, unit.exports) for unit in request.units] == [
        ("a", (), ()), ("b", (), ()),
    ]


def test_admission_count_hints_point_in_the_right_direction(tmp_path):
    from stateframe.engine.kernel_parallel import _refusal_location
    from stateframe.engine.kernel_parallel_admission import admit
    from stateframe.engine.kernel_parallel_records import ParallelLimits

    snapshot = _repo(tmp_path)
    limits = ParallelLimits(min_units=2, max_units=2)
    one = _request({"subframes": _ordinary()["subframes"][:1]}, "one", "2")
    few = admit(one, snapshot, (), limits, protected=lambda _: False, case_insensitive=False)
    assert few.refusal is not None and few.refusal.reason == "TOO_FEW_UNITS"
    assert _refusal_location(few.refusal, one)[2] == (
        "add subframes until the named minimum is met, or lower that minimum")

    three = _ordinary()["subframes"] + [
        {"id": "c", "outcome": "Change c", "owns": ["c.py"], "creates": []}]
    many_request = _request({"subframes": three}, "many", "2")
    many = admit(many_request, snapshot, (), limits, protected=lambda _: False,
                 case_insensitive=False)
    assert many.refusal is not None and many.refusal.reason == "TOO_MANY_UNITS"
    assert _refusal_location(many.refusal, many_request)[2] == (
        "remove subframes until the named maximum is met, or raise that maximum")


@pytest.mark.parametrize("state", ("complete", "pending_result", "pending_intent"))
def test_recorded_goal_conflict_refusal_replays_with_original_words(tmp_path, monkeypatch, state):
    import hashlib
    import json

    import stateframe.engine.kernel_parallel as parallel
    from stateframe.engine.kernel_parallel_records import (
        AdmissionResult, ParallelMeasured, ParallelResult, Refusal, canonical_json,
    )
    from stateframe.engine.kernel_parallel_txn import manifest_digest
    from stateframe.engine.kernel_tool_tape import ToolReplayRefused, ToolTape, configure_replay
    from test_parallel_dispatch import setup_context

    run, root, context = setup_context(tmp_path, monkeypatch)
    args = _terse()
    args["subframes"][0]["outcome"] = "The corrected value is returned"
    raw_call_id = "parallel_1"
    call_root = run / "parallel" / raw_call_id
    call_root.mkdir(parents=True)
    current_manifest = manifest_digest(root)
    environment = str(context.runtime.binding().get("environment_revision") or "disabled")
    error = ("subframe a outcome: conflicts with goal; fix: "
             "make outcome equal goal or omit outcome")
    observation = "REFUSED INVALID_ARGUMENTS: " + error
    refusal = Refusal("INVALID_ARGUMENTS", "subframe a outcome: conflicts with goal",
                      (), (), (), "make outcome equal goal or omit outcome")
    admission = AdmissionResult("REFUSED", refusal, (), (), (), 0, current_manifest,
                                context.limits)
    result = ParallelResult(raw_call_id, admission, (), (), "NOT_RUN", (), "NOT_RUN",
                            observation, ParallelMeasured(0.0, 0.0, 0.0, 0, 0.0))
    if state != "pending_intent":
        (call_root / "result.json").write_bytes(canonical_json(result.to_json()))
    tape = ToolTape(run)
    tape.intent(call_id=raw_call_id, tool="parallel",
                args={"schema": "stateframe-parallel-input-refusal/1", "arguments": args},
                replay_class="local",
                capability_digest=hashlib.sha256(b"stateframe-subframe-input/1").hexdigest(),
                definition_digest=hashlib.sha256(canonical_json(args)).hexdigest(),
                env_identity=environment, pre_manifest_digest=current_manifest)
    if state == "complete":
        tape.complete(call_id=raw_call_id, exit_code=0, observation=observation,
                      post_manifest_digest=current_manifest, env_identity=environment)
    else:
        parallel._service[1]["parallel_recovery"] = {
            "schema": "stateframe-parallel-input-refusal/1", "call_id": raw_call_id,
            "arguments": args,
        }
    configure_replay(run, active=True)

    class NoLaunchHost:
        def submit(self, op, scope, payload):
            pytest.fail("recorded refusal launched a subframe")

        def poll(self, result_id):
            pytest.fail("recorded refusal polled a subframe")

        def cancel(self, scope):
            pytest.fail("recorded refusal cancelled a subframe")

    replay = parallel.execute_parallel(context, args, host=NoLaunchHost(), store_root=call_root)
    assert replay == result
    assert json.loads((call_root / "result.json").read_bytes()) == result.to_json()
    assert ToolTape(run).lookup(raw_call_id).observation == observation
    (call_root / "result.json").write_bytes(canonical_json(
        {**result.to_json(), "observation": "REFUSED INVALID_ARGUMENTS: altered"}))
    with pytest.raises(ToolReplayRefused, match="observation changed"):
        parallel.execute_parallel(context, args, host=NoLaunchHost(), store_root=call_root)


def test_pending_fourth_wire_call_replays_historical_refusal_before_result_seal(tmp_path, monkeypatch):
    import hashlib

    import stateframe.engine.kernel_parallel as parallel
    from stateframe.engine.kernel_parallel_records import canonical_json
    from stateframe.engine.kernel_parallel_txn import manifest_digest
    from stateframe.engine.kernel_tool_tape import ToolTape, configure_replay
    from test_parallel_dispatch import setup_context

    args = _fourth_wire_call()
    files = {unit["owns"][0]: "VALUE = 1\n" for unit in args["subframes"]}
    run, root, context = setup_context(tmp_path, monkeypatch, files=files)
    call_id = context.runtime.current_tool_call_id
    tape = ToolTape(run)
    tape.intent(call_id=call_id, tool="parallel",
                args={"schema": "stateframe-parallel-input-refusal/1", "arguments": args},
                replay_class="local",
                capability_digest=hashlib.sha256(b"stateframe-subframe-input/1").hexdigest(),
                definition_digest=hashlib.sha256(canonical_json(args)).hexdigest(),
                env_identity=str(context.runtime.binding().get("environment_revision") or "disabled"),
                pre_manifest_digest=manifest_digest(root))
    parallel._service[1]["parallel_recovery"] = {
        "schema": "stateframe-parallel-input-refusal/1", "call_id": call_id,
        "arguments": args,
    }
    configure_replay(run, active=True)

    class NoLaunchHost:
        def submit(self, op, scope, payload):
            pytest.fail("recorded refusal launched a subframe")

        def poll(self, result_id):
            pytest.fail("recorded refusal polled a subframe")

        def cancel(self, scope):
            pytest.fail("recorded refusal cancelled a subframe")

    result = parallel.execute_parallel(context, args, host=NoLaunchHost(),
                                       store_root=run / "parallel" / call_id)
    expected = ("REFUSED INVALID_ARGUMENTS: subframe fix_pricing outcome: "
                "expected nonempty text; fix: provide outcome as text")
    assert result.observation == expected
    assert result.admission.refusal is not None
    assert result.admission.refusal.hint == "provide outcome as text"
    assert ToolTape(run).lookup(call_id).observation == expected


# Each repair uses the value the hint asks for, keeping all other model input.
# ``None`` means remove the field. The parser, rather than a mock, decides
# whether the repaired request is accepted.
_NEAR_MISSES = (
    ("id", None, "a"), ("id", 3, "a"),
    ("outcome", None, "Change a"), ("outcome", "", "Change a"),
    ("owns", None, ["a.py"]), ("owns", "a.py", ["a.py"]),
    ("creates", None, []), ("creates", "new.py", []),
    ("reads", "a.py", []), ("requirements", "condition", []),
    ("requirements", ["x"] * 9, []),
    ("requirements", ["x" * 301], ["x"]),
    ("exports", {}, []), ("exports", [7], []),
    ("exports", [{"id": "x"}], []),
    ("consumes", {}, []), ("consumes", [7], []),
    ("consumes", [{"export_id": 3}], []),
    ("check", {}, {"tests": []}),
    ("check", {"tests": [""]}, {"tests": []}),
    ("check", {"tests": "test_a.py"}, {"tests": []}),
    ("model_class", "", "default"),
    ("bogus", 1, None),
)


@pytest.mark.parametrize("field,bad,repair", _NEAR_MISSES)
def test_ordinary_refusal_hint_repairs_without_repeating_site(tmp_path, field, bad, repair):
    args = _ordinary()
    if bad is None:
        args["subframes"][0].pop(field, None)
    else:
        args["subframes"][0][field] = bad
    snapshot = _repo(tmp_path)
    seen = set()
    for _ in range(5):
        try:
            request, _, _ = _parse(args, snapshot, expand=False)
        except ParallelInputError as error:
            site = (error.unit_id, error.field)
            assert site not in seen, (site, error.hint)
            seen.add(site)
            assert error.hint
            if error.field == field or error.field.startswith(field + "[") or error.field.startswith(field + "."):
                if repair is None:
                    args["subframes"][0].pop(field, None)
                else:
                    args["subframes"][0][field] = copy.deepcopy(repair)
            else:
                pytest.fail(f"hint for {field} led to unrelated refusal: {error}")
        else:
            assert request.units[0].id == "a"
            break
    else:
        pytest.fail(f"hint chain did not end: {seen}")


@pytest.mark.parametrize("field,bad,repair", (
    ("goal", "", "Change a"),
    ("goal", 42, "Change a"),
    ("outcome", "", "The corrected value is returned"),
    ("creates", "new.py", []),
    ("check", {}, {"tests": []}),
))
def test_terse_refusal_hint_repairs_without_repeating_site(tmp_path, field, bad, repair):
    args = _terse()
    args["subframes"][0][field] = bad
    snapshot = _repo(tmp_path)
    seen = set()
    for _ in range(5):
        try:
            request, _, _ = _parse(args, snapshot, expand=True)
        except ParallelInputError as error:
            site = (error.unit_id, error.field)
            assert site not in seen, (site, error.hint)
            seen.add(site)
            assert error.hint
            assert error.field == field or error.field.startswith(field + ".")
            args["subframes"][0][field] = copy.deepcopy(repair)
        else:
            assert request.units[0].id == "a"
            break
    else:
        pytest.fail(f"hint chain did not end: {seen}")
