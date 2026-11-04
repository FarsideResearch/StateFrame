"""Tests for keeping recorded corpus engine runs out of xdist workers."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import serial_lane_plugin

pytest_plugins = ["pytester"]


TESTS = Path(__file__).resolve().parent
_COLLECTED_PROPERTY = "serial_lane_collected"


class _FakeItem:
    def __init__(self, *, serial_lane: bool = False):
        self.markers = {"serial_lane": pytest.mark.serial_lane} if serial_lane else {}
        self.user_properties: list[tuple[str, object]] = []
        self.added_markers = []

    def get_closest_marker(self, name: str):
        return self.markers.get(name)

    def add_marker(self, marker):
        self.added_markers.append(marker)


def _prepare_pytester(pytester, ledger: Path) -> None:
    pytester._monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(
            part for part in (str(TESTS), os.environ.get("PYTHONPATH", "")) if part
        ),
    )
    pytester._monkeypatch.setenv("SF_SERIAL_LANE_LEDGER", str(ledger))
    # Ensure the child ignores inherited worker variables and inspects its own pytest config.
    pytester._monkeypatch.setenv("PYTEST_XDIST_WORKER", "outer-worker")


def test_serial_lane_items_are_tagged_and_skipped_for_xdist():
    serial_item = _FakeItem(serial_lane=True)
    regular_item = _FakeItem()
    config = SimpleNamespace(option=SimpleNamespace(numprocesses=2), workerinput={})

    serial_lane_plugin.collect_serial_lane_items([serial_item, regular_item], config)

    assert serial_item.user_properties == [(_COLLECTED_PROPERTY, True)]
    assert len(serial_item.added_markers) == 1
    skip = serial_item.added_markers[0]
    assert skip.name == "skip"
    assert skip.kwargs["reason"] == (
        "serial_lane: 1 corpus cases deferred to the serial step"
    )
    assert regular_item.user_properties == []
    assert regular_item.added_markers == []
    assert config._serial_lane_collected == 1
    assert config._serial_lane_deferred == 1


def test_inherited_worker_environment_does_not_defer_plain_session(monkeypatch):
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "outer-worker")
    serial_item = _FakeItem(serial_lane=True)
    config = SimpleNamespace(option=SimpleNamespace(numprocesses=None))

    serial_lane_plugin.collect_serial_lane_items([serial_item], config)

    assert serial_item.user_properties == [(_COLLECTED_PROPERTY, True)]
    assert serial_item.added_markers == []
    assert config._serial_lane_collected == 1
    assert config._serial_lane_deferred == 0


def test_terminal_summary_reports_controller_deferred_count():
    lines = []
    terminal = SimpleNamespace(write_line=lines.append)
    config = SimpleNamespace(
        _serial_lane_deferred=0,
        _serial_lane_deferred_nodeids={"a", "b", "c"},
    )

    serial_lane_plugin.terminal_summary(terminal, 0, config)

    assert lines == ["serial_lane: 3 corpus cases deferred to the serial step"]


def test_controller_counts_only_collection_tagged_reports(monkeypatch):
    config = SimpleNamespace(
        option=SimpleNamespace(numprocesses=2),
        _serial_lane_reported_nodeids=set(),
        _serial_lane_deferred_nodeids=set(),
        _serial_lane_ran_nodeids=set(),
    )
    monkeypatch.setattr(serial_lane_plugin, "_SERIAL_LANE_CONFIG", config)
    tagged = [(_COLLECTED_PROPERTY, True)]
    reports = [
        SimpleNamespace(
            nodeid="test_cases.py::test_serial_a",
            keywords={"serial_lane": True},
            user_properties=tagged,
            when="setup",
            skipped=True,
        ),
        SimpleNamespace(
            nodeid="test_cases.py::test_serial_b",
            keywords={"serial_lane": True},
            user_properties=tagged,
            when="setup",
            skipped=True,
        ),
        SimpleNamespace(
            nodeid="test_cases.py::test_serial_a",
            keywords={"serial_lane": True},
            user_properties=tagged,
            when="setup",
            skipped=True,
        ),
        SimpleNamespace(
            nodeid="test_cases.py::test_serial_c",
            keywords={"serial_lane": True},
            user_properties=tagged,
            when="call",
            skipped=False,
        ),
        SimpleNamespace(
            nodeid="test_cases.py::test_runtime_marker",
            keywords={"serial_lane": True},
            user_properties=[],
            when="call",
            skipped=False,
        ),
    ]

    for report in reports:
        serial_lane_plugin.record_logreport(report)

    assert config._serial_lane_reported_nodeids == {
        "test_cases.py::test_serial_a",
        "test_cases.py::test_serial_b",
        "test_cases.py::test_serial_c",
    }
    assert config._serial_lane_deferred_nodeids == {
        "test_cases.py::test_serial_a",
        "test_cases.py::test_serial_b",
    }
    assert config._serial_lane_ran_nodeids == {"test_cases.py::test_serial_c"}


def test_terminal_summary_omits_empty_deferred_count():
    lines = []
    terminal = SimpleNamespace(write_line=lines.append)
    config = SimpleNamespace(
        _serial_lane_deferred=0,
        _serial_lane_deferred_nodeids=set(),
    )

    serial_lane_plugin.terminal_summary(terminal, 0, config)

    assert lines == []


def test_marked_serial_lane_case_runs_in_plain_pytest(pytester):
    pytester.makepyfile(
        test_cases="""
import pytest


@pytest.mark.serial_lane
def test_serial_case():
    pass
"""
    )
    ledger = pytester.path / "serial-lane.jsonl"
    _prepare_pytester(pytester, ledger)

    plain = pytester.runpytest_subprocess(
        "-p", "conftest", "-p", "no:xdist", "-p", "no:cacheprovider", "-q", "test_cases.py"
    )
    plain.assert_outcomes(passed=1)
    assert json.loads(ledger.read_text(encoding="utf-8")) == {
        "collected": 1,
        "deferred": 0,
        "ran": 1,
    }


def test_serial_lane_xdist_regression_with_real_controller_and_ledger(pytester):
    pytester.makepyfile(
        test_cases="""
import pytest


@pytest.mark.serial_lane
def test_serial_a():
    pass


@pytest.mark.serial_lane
def test_serial_b():
    pass


def test_regular():
    pass
"""
    )
    ledger = pytester.path / "serial-lane.jsonl"
    _prepare_pytester(pytester, ledger)

    distributed = pytester.runpytest_subprocess(
        "-p", "conftest", "-p", "no:cacheprovider", "-n", "2", "-q", "test_cases.py"
    )
    distributed.assert_outcomes(passed=1, skipped=2)
    summary_line = "serial_lane: 2 corpus cases deferred to the serial step"
    assert distributed.stdout.str().splitlines().count(summary_line) == 1

    checker_env = os.environ.copy()
    checker = subprocess.run(
        [sys.executable, "-m", "serial_lane_ledger", str(ledger)],
        check=False,
        capture_output=True,
        text=True,
        env=checker_env,
        timeout=10,
    )
    assert checker.returncode == 1
    assert checker.stderr == (
        "serial_lane ledger failure: 2 corpus cases collected, 0 ran across "
        "1 pytest sessions (2 deferred)\n"
    )

    serial = pytester.runpytest_subprocess(
        "-p", "conftest", "-p", "no:xdist", "-m", "serial_lane", "-q", "test_cases.py"
    )
    serial.assert_outcomes(passed=2, deselected=1)

    checker = subprocess.run(
        [sys.executable, "-m", "serial_lane_ledger", str(ledger)],
        check=False,
        capture_output=True,
        text=True,
        env=checker_env,
        timeout=10,
    )
    assert checker.returncode == 0, checker.stderr
    assert checker.stdout == (
        "serial_lane ledger: 4 corpus cases collected, 2 ran, 2 deferred "
        "across 2 pytest sessions\n"
    )


def test_xdist_runtime_marker_does_not_count_as_serial_lane_run(pytester):
    pytester.makepyfile(
        test_cases="""
import pytest


@pytest.mark.serial_lane
def test_serial_a():
    pass


@pytest.mark.serial_lane
def test_serial_b():
    pass


def test_runtime_marker(request):
    request.node.add_marker(pytest.mark.serial_lane)
"""
    )
    ledger = pytester.path / "serial-lane-runtime-marker.jsonl"
    _prepare_pytester(pytester, ledger)

    distributed = pytester.runpytest_subprocess(
        "-p", "conftest", "-p", "no:cacheprovider", "-n", "2", "-q", "test_cases.py"
    )
    distributed.assert_outcomes(passed=1, skipped=2)
    summary_line = "serial_lane: 2 corpus cases deferred to the serial step"
    assert distributed.stdout.str().splitlines().count(summary_line) == 1
    assert json.loads(ledger.read_text(encoding="utf-8")) == {
        "collected": 2,
        "deferred": 2,
        "ran": 0,
    }

    checker = subprocess.run(
        [sys.executable, "-m", "serial_lane_ledger", str(ledger)],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=10,
    )
    assert checker.returncode == 1
    assert checker.stderr == (
        "serial_lane ledger failure: 2 corpus cases collected, 0 ran across "
        "1 pytest sessions (2 deferred)\n"
    )
