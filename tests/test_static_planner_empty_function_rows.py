"""Static planning keeps explicit targets whose implementations are still stubs."""

from __future__ import annotations

import unit_planner_v2
import pytest


@pytest.mark.parametrize(
    "body",
    (
        "    pass\n",
        "    ...\n",
        "    \"\"\"Implementation pending.\"\"\"\n",
        "    raise NotImplementedError\n",
    ),
)
def test_existing_spec_target_gets_unit_when_tests_do_not_reference_it(
    tmp_path, monkeypatch, body: str
) -> None:
    source = tmp_path / "pkg" / "worker.py"
    source.parent.mkdir(parents=True)
    source.write_text("def load_value():\n" + body, encoding="utf-8")
    monkeypatch.setenv(unit_planner_v2.MODE_ENV, "on")

    row = {
        "fn": "pkg/worker.py",
        "path": "pkg/worker.py",
        "sha": "stub-target",
        "msg": "Implement def load_value.",
        "test_src": (
            "from pkg.worker import load_value\n\n"
            "def test_contract():\n"
            "    assert True\n"
        ),
    }

    plan = unit_planner_v2.derive_with_index(row, tmp_path)

    assert plan is not None
    assert [(unit.symbol, unit.files) for unit in plan.units] == [
        ("load_value", ("pkg/worker.py",)),
    ]


def test_stub_target_in_large_source_file_gets_unit(tmp_path, monkeypatch) -> None:
    source = tmp_path / "pkg" / "worker.py"
    source.parent.mkdir(parents=True)
    source.write_text("# filler\n" * 150_000 + "def load_value():\n    pass\n", encoding="utf-8")
    monkeypatch.setenv(unit_planner_v2.MODE_ENV, "on")

    row = {
        "fn": "pkg/worker.py",
        "path": "pkg/worker.py",
        "sha": "large-stub-target",
        "msg": "Implement def load_value.",
        "test_src": (
            "from pkg.worker import load_value\n\n"
            "def test_contract():\n"
            "    assert True\n"
        ),
    }

    plan = unit_planner_v2.derive_with_index(row, tmp_path)

    assert plan is not None
    assert [unit.symbol for unit in plan.units] == ["load_value"]


def test_missing_spec_target_still_produces_no_unit(tmp_path, monkeypatch) -> None:
    source = tmp_path / "pkg" / "worker.py"
    source.parent.mkdir(parents=True)
    source.write_text("def helper():\n    return 1\n", encoding="utf-8")
    monkeypatch.setenv(unit_planner_v2.MODE_ENV, "on")

    row = {
        "fn": "pkg/worker.py",
        "path": "pkg/worker.py",
        "sha": "missing-target",
        "msg": "Implement def load_value.",
        "test_src": (
            "from pkg.worker import load_value\n\n"
            "def test_contract():\n"
            "    assert True\n"
        ),
    }

    assert unit_planner_v2.derive_with_index(row, tmp_path) is None
