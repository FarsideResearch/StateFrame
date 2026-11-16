"""Golden observation strings for the legacy ``str_replace`` executor.

The edit-protocol work routes the ``legacy_safe`` front end through the shared
batch validator and ``EditAuthority`` when ``STATEFRAME_BOUND_EDIT=on``.  That
refactor is only admissible if every observation the model sees today stays
byte-identical.  This suite pins today's bytes for the executor outcomes the
refactor touches (not found, N places, parse break, success) under both switch
values, plus the bound-target refusals that only exist when the switch is on.

The golden file is ``tests/fixtures/legacy_safe_observations.json``.  Regenerate
it deliberately with ``LEGACY_SAFE_GOLDEN_WRITE=1`` after an intentional prose
change; an unintended diff is a parity break.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib

import pytest

from kernel_bound_edit import bind_target
from probes.react_repair import str_replace_edit

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "legacy_safe_observations.json"
SCHEMA = "legacy-safe-observations/1"
FILE_NAME = "mod.py"
SOURCE = (
    "def alpha():\n"
    "    return 1\n"
    "\n"
    "\n"
    "def beta():\n"
    "    return 2\n"
)
# Switches that alter the executor's prose or side effects; pinned so the golden
# describes exactly one configuration.
_PINNED_OFF = (
    "STATEFRAME_EDIT_GUARD",
    "STATEFRAME_EMPIRICAL_FIXES",
    "STATEFRAME_EFFECT_BOUNDARY",
    "STATEFRAME_REPRO_FIRST_GATE",
    "STATEFRAME_GREEN_LOCK",
    "STATEFRAME_DESTRUCTIVE_EDIT_GUARD",
    "STATEFRAME_SCOPE_WIDEN_GUARD",
    "STATEFRAME_WORKSPACE_TRUTH",
    "STATEFRAME_TOKENLAB_NONPY_EDITS",
)

# case id -> (bound_edit_mode, target, old_str, new_str, expect_mutation)
# target: "bound" binds the current file, "stale" binds then edits the file,
# None passes no target_ref.
CASES: dict[str, tuple[str, str | None, str, str, bool]] = {
    "off_not_found": ("off", None, "    return 3\n", "    return 30\n", False),
    "off_n_places": ("off", None, "    return ", "    return 9 + ", False),
    "off_parse_break": ("off", None, "    return 2\n", "    return 2 )\n", False),
    "off_success": ("off", None, "    return 1\n", "    return 10\n", True),
    "off_empty_old": ("off", None, "", "    return 10\n", False),
    "off_noop": ("off", None, "    return 1\n", "    return 1\n", False),
    "off_missing_file": ("off", "missing", "    return 1\n", "    return 10\n", False),
    "on_not_found": ("on", "bound", "    return 3\n", "    return 30\n", False),
    "on_n_places": ("on", "bound", "    return ", "    return 9 + ", False),
    "on_parse_break": ("on", "bound", "    return 2\n", "    return 2 )\n", False),
    "on_success": ("on", "bound", "    return 1\n", "    return 10\n", True),
    "on_missing_target": ("on", None, "    return 1\n", "    return 10\n", False),
    "on_stale_target": ("on", "stale", "    return 1\n", "    return 10\n", False),
}


def _pin_environment(monkeypatch, bound_mode: str) -> None:
    for name in _PINNED_OFF:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("STATEFRAME_BOUND_EDIT", bound_mode)


def _bind(root: pathlib.Path, path: pathlib.Path) -> dict[str, object]:
    raw = path.read_bytes()
    return bind_target(
        path=path.relative_to(root).as_posix(), symbol="",
        revision="sha256:" + hashlib.sha256(raw).hexdigest(),
        source=raw.decode("utf-8"),
    )


def _run_case(case_id: str, tmp_path: pathlib.Path, monkeypatch) -> tuple[str, bytes | None]:
    bound_mode, target, old, new, _expect_mutation = CASES[case_id]
    _pin_environment(monkeypatch, bound_mode)
    root = tmp_path / case_id
    root.mkdir()
    path = root / FILE_NAME
    target_ref = None
    if target == "missing":
        return str_replace_edit(path, old, new, root=root), None
    path.write_text(SOURCE, encoding="utf-8")
    if target in {"bound", "stale"}:
        target_ref = _bind(root, path)
    if target == "stale":
        path.write_text(SOURCE + "\n# drift\n", encoding="utf-8")
    observation = str_replace_edit(path, old, new, target_ref=target_ref, root=root)
    return observation, path.read_bytes()


def _golden() -> dict[str, str]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert payload.get("schema") == SCHEMA
    return payload["observations"]


@pytest.fixture(scope="module")
def regenerated_golden(tmp_path_factory):
    """Optionally rewrite the golden from today's executor before comparing."""
    if os.environ.get("LEGACY_SAFE_GOLDEN_WRITE") != "1":
        return None
    monkeypatch = pytest.MonkeyPatch()
    try:
        tmp_path = tmp_path_factory.mktemp("legacy-safe-golden")
        observations = {case_id: _run_case(case_id, tmp_path, monkeypatch)[0] for case_id in CASES}
    finally:
        monkeypatch.undo()
    FIXTURE.write_text(
        json.dumps({"schema": SCHEMA, "observations": observations}, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return observations


def test_golden_covers_exactly_the_pinned_cases(regenerated_golden):
    assert set(_golden()) == set(CASES)


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_observation_bytes_match_golden(case_id, tmp_path, monkeypatch, regenerated_golden):
    observation, after = _run_case(case_id, tmp_path, monkeypatch)
    assert observation == _golden()[case_id]
    expect_mutation = CASES[case_id][4]
    if after is None:
        return
    if expect_mutation:
        assert after != SOURCE.encode("utf-8")
        assert b"return 10" in after
    elif CASES[case_id][1] == "stale":
        assert after == (SOURCE + "\n# drift\n").encode("utf-8")
    else:
        assert after == SOURCE.encode("utf-8")


def test_refusals_never_mutate_and_say_so():
    """Every pinned refusal is a REJECTED/NO-OP observation; only successes apply."""
    golden = _golden()
    for case_id, (_mode, _target, _old, _new, expect_mutation) in CASES.items():
        text = golden[case_id]
        if expect_mutation:
            assert text.startswith("str_replace applied.")
        else:
            assert text.startswith(("str_replace REJECTED:", "str_replace NO-OP:"))
            assert "applied." not in text


def test_bound_mode_keeps_legacy_prose_for_shared_outcomes():
    """The control arm: switching the bound target on must not change these strings."""
    golden = _golden()
    for outcome in ("not_found", "n_places", "parse_break", "success"):
        assert golden[f"on_{outcome}"] == golden[f"off_{outcome}"]


def test_golden_file_unchanged_by_routing():
    """The routing change must not need a golden regen: this sha is e91a2a5b's fixture."""
    import hashlib
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == "5a2fc757e8c420ca418eb7b516cfbd5e41a08a1f641dc9ed3bddc37e749246e4"
