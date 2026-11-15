"""Adapter tests: a declared scenario yields exact provider request bytes and a journal from a model-free run."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import enginecheck
from enginecheck import HarnessError, Need, Scenario, VacuousScenario, run, run_pair

READ_THEN_DONE = [
    {"tool": "read", "required": True, "arguments": {"path": "README.md"}},
    {"tool": "done", "required": True, "arguments": {"summary": "Read the README."}},
]
FIXTURE_FILES = {
    "README.md": b"Enginecheck adapter fixture.\n",
    "empty.txt": b"",
    "crlf.txt": b"one\r\ntwo\r\n",
    "binary.bin": b"\xff\x00a\r\n",
}


def _scenario(**overrides: object) -> Scenario:
    values: dict[str, object] = {
        "files": {"README.md": FIXTURE_FILES["README.md"]},
        "task": "Read the README.",
        "steps": READ_THEN_DONE,
    }
    values.update(overrides)
    return Scenario(**values)  # type: ignore[arg-type]


def _wire_files(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "wire").glob("*.request.json"), key=lambda p: int(p.name.split(".")[0]))


def test_two_step_scenario_yields_provider_requests_and_exit_code(tmp_path: Path) -> None:
    trace = run(_scenario(), tmp_path)

    assert len(trace.requests) >= 2
    for raw in trace.requests:
        decoded = json.loads(raw)
        assert isinstance(decoded, dict) and "messages" in decoded
    assert trace.request_json == [json.loads(raw) for raw in trace.requests]
    assert trace.requests == [path.read_bytes() for path in _wire_files(trace.run_dir)]
    assert isinstance(trace.exit_code, int)


def test_same_scenario_twice_gives_byte_identical_requests(tmp_path: Path) -> None:
    first = run(_scenario(), tmp_path / "first")
    second = run(_scenario(), tmp_path / "second")

    assert first.requests == second.requests


def test_need_that_cannot_hold_raises_vacuous_scenario(tmp_path: Path) -> None:
    impossible = Need("the run reads a file that does not exist", lambda trace: False)
    with pytest.raises(VacuousScenario, match="the run reads a file that does not exist"):
        run(_scenario(needs=[impossible]), tmp_path)


def test_scenario_files_are_byte_exact_in_repo(tmp_path: Path) -> None:
    trace = run(_scenario(files=FIXTURE_FILES), tmp_path)

    for name, content in FIXTURE_FILES.items():
        assert (trace.repo / name).read_bytes() == content, name
    assert len(trace.requests) >= 2


def test_run_pair_applies_switch_only_to_the_on_side(tmp_path: Path) -> None:
    off, on = run_pair(_scenario(), tmp_path, {"STATEFRAME_PAGE_LOG": "on"})

    assert on.switches["STATEFRAME_PAGE_LOG"] == "on"
    assert "STATEFRAME_PAGE_LOG" not in off.switches
    assert off.run_dir != on.run_dir


def test_run_pair_off_side_strips_inherited_engine_switch(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("STATEFRAME_X_PROBE", "1")
    engine_envs: list[dict[str, str]] = []
    real_run_monitored = enginecheck.run_monitored

    def recording_run_monitored(command, **kwargs):
        engine_envs.append(dict(kwargs["env"]))
        return real_run_monitored(command, **kwargs)

    monkeypatch.setattr(enginecheck, "run_monitored", recording_run_monitored)
    off, on = run_pair(_scenario(), tmp_path, {"STATEFRAME_X_PROBE": "2"})

    assert "STATEFRAME_X_PROBE" not in off.switches
    assert "STATEFRAME_X_PROBE" not in engine_envs[0]
    assert on.switches["STATEFRAME_X_PROBE"] == "2"
    assert engine_envs[1]["STATEFRAME_X_PROBE"] == "2"


def test_trace_switches_report_the_env_passed_to_the_engine(monkeypatch, tmp_path: Path) -> None:
    engine_envs: list[dict[str, str]] = []
    real_run_monitored = enginecheck.run_monitored

    def recording_run_monitored(command, **kwargs):
        engine_envs.append(dict(kwargs["env"]))
        return real_run_monitored(command, **kwargs)

    monkeypatch.setattr(enginecheck, "run_monitored", recording_run_monitored)
    trace = run(_scenario(switches={"STATEFRAME_X_PROBE": "3"}), tmp_path)

    engine_switches = {name: value for name, value in engine_envs[0].items()
                       if name.startswith("STATEFRAME_") and name != "STATEFRAME_HOME"}
    assert trace.switches == engine_switches == {"STATEFRAME_X_PROBE": "3"}


def test_unstartable_engine_raises_harness_error(monkeypatch, tmp_path: Path) -> None:
    def failing_run_monitored(command, **kwargs):
        raise FileNotFoundError("no such interpreter")

    monkeypatch.setattr(enginecheck, "run_monitored", failing_run_monitored)
    with pytest.raises(HarnessError) as info:
        run(_scenario(), tmp_path)
    assert isinstance(info.value.__cause__, FileNotFoundError)


def test_engine_that_sends_no_request_raises_harness_error(tmp_path: Path) -> None:
    with pytest.raises(HarnessError):
        run(_scenario(task=""), tmp_path)


def test_wire_files_sort_numerically_past_four_digits(tmp_path: Path) -> None:
    wire = tmp_path / "wire"
    wire.mkdir()
    for stem in ("9999", "10000", "0001"):
        (wire / f"{stem}.request.json").write_bytes(b"{}")

    names = [path.name for path in enginecheck._request_paths(tmp_path)]

    assert names == ["0001.request.json", "9999.request.json", "10000.request.json"]
