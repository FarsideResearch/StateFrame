"""FU-070-ALLOC: the child edit gate judges the request that was sent, not the pre-render list."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel_source_view import file_floor_messages

PATH = "owned_source.py"
SOURCE = "".join(f"VALUE_{line:04d} = {line}\n" for line in range(2100))
TASK = {"role": "user", "content": "Task: inspect the owned source."}


def _prepared(pre_render: list[dict], sent: list[dict] | None) -> SimpleNamespace:
    """A prepared request whose pre-render list lacks the source window; ``sent`` is the wire."""
    qualified = None if sent is None else SimpleNamespace(body={"messages": sent})
    capture = SimpleNamespace(request_id="req-sent", qualified_request=qualified)
    return SimpleNamespace(messages=tuple(pre_render), capture=capture)


def _admit(prepared: SimpleNamespace):
    from probes.react_repair import _admit_sent_edit

    return _admit_sent_edit(prepared, path=PATH, source=SOURCE.encode("utf-8"))


def test_edit_admitted_from_complete_sent_wire_when_pre_render_has_no_pages() -> None:
    pages = file_floor_messages(PATH, SOURCE)
    prepared = _prepared([TASK], [TASK, *pages])
    assert _admit(prepared) == (True, "", "")


def test_edit_refused_when_sent_wire_is_missing_a_middle_page() -> None:
    pages = file_floor_messages(PATH, SOURCE)
    # pages[0..2] are the three source pages and pages[3] is the map line; drop the middle page.
    partial = [TASK, pages[0], pages[2], pages[3]]
    ok, reason, _detail = _admit(_prepared([TASK], partial))
    assert ok is False
    assert reason == "incomplete_view"


def test_edit_refused_when_only_one_page_of_several_is_sent() -> None:
    pages = file_floor_messages(PATH, SOURCE)
    ok, reason, _detail = _admit(_prepared([TASK], [TASK, pages[0], pages[3]]))
    assert ok is False
    assert reason == "incomplete_view"


def test_edit_fails_closed_without_a_qualified_sent_wire() -> None:
    pages = file_floor_messages(PATH, SOURCE)
    # The pre-render list is complete, but no sent wire exists: must refuse, never admit on pre-render.
    ok, reason, _detail = _admit(_prepared([TASK, *pages], None))
    assert ok is False
    assert reason == "incomplete_view"


def _unit(owns: set[str], creates: set[str] = frozenset()) -> SimpleNamespace:
    contract = SimpleNamespace(owns=tuple(sorted(owns)), creates=tuple(sorted(creates)))
    return SimpleNamespace(spec=SimpleNamespace(contract=contract))


def _owned_file(tmp_path: Path) -> Path:
    target = tmp_path / PATH
    target.write_text(SOURCE, encoding="utf-8")
    return target


def test_gate_scope_reads_allocator_enabled_not_the_raw_environment(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import kernel_read_receipts
    from probes.react_repair import _unit_edit_needs_sent_view

    owned = _owned_file(tmp_path)
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "off")
    monkeypatch.setattr(kernel_read_receipts, "enabled", lambda: True)
    assert _unit_edit_needs_sent_view(_unit({PATH}), op="edit", path=PATH, file=owned) is True
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    monkeypatch.setattr(kernel_read_receipts, "enabled", lambda: False)
    assert _unit_edit_needs_sent_view(_unit({PATH}), op="edit", path=PATH, file=owned) is False


def test_gate_never_runs_with_the_allocator_off(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from probes.react_repair import _unit_edit_needs_sent_view

    owned = _owned_file(tmp_path)
    monkeypatch.delenv("STATEFRAME_WORKING_SET_ALLOCATOR", raising=False)
    assert _unit_edit_needs_sent_view(_unit({PATH}), op="edit", path=PATH, file=owned) is False
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    assert _unit_edit_needs_sent_view(_unit({PATH}), op="edit", path=PATH, file=owned) is True


def test_gate_skips_missing_and_symlinked_owned_paths(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from probes.react_repair import _unit_edit_needs_sent_view

    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    owned = _owned_file(tmp_path)
    link = tmp_path / "linked.py"
    link.symlink_to(owned)
    unit = _unit({PATH, "linked.py", "absent.py"})
    assert _unit_edit_needs_sent_view(unit, op="edit", path="linked.py", file=link) is False
    assert _unit_edit_needs_sent_view(
        unit, op="edit", path="absent.py", file=tmp_path / "absent.py") is False
    assert _unit_edit_needs_sent_view(unit, op="edit", path="", file=owned) is False


def test_created_path_edit_is_outside_the_gate_by_documented_scope(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import kernel_unit_policy
    from probes.react_repair import _unit_edit_needs_sent_view

    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    owned = _owned_file(tmp_path)
    created = tmp_path / "new_module.py"
    created.write_text("VALUE = 1\n", encoding="utf-8")
    unit = _unit({PATH}, creates={"new_module.py"})
    assert _unit_edit_needs_sent_view(unit, op="edit", path="new_module.py", file=created) is False
    assert _unit_edit_needs_sent_view(unit, op="create", path=PATH, file=owned) is False
    assert _unit_edit_needs_sent_view(unit, op="edit", path=PATH, file=owned) is True
    # The scope rests on this: a unit edits only through tools anchored on current bytes.
    # A new unit edit tool must revisit the created-path exclusion (CHANGELOG, helper docstring).
    assert kernel_unit_policy._MUTATING_TOOLS == {"str_replace", "insert_before", "file_create"}


def test_edit_gate_refusal_path_never_dereferences_capture_directly() -> None:
    # The refusal journal line runs exactly when the sent wire is missing,
    # which includes prepared.capture being None; a direct attribute access
    # there turned a clean SOURCE_NOT_VERIFIED refusal into AttributeError.
    import pathlib
    import re

    import probes.react_repair as rr

    text = pathlib.Path(rr.__file__).read_text(encoding="utf-8")
    block = text[text.index('"edit_admission_refused"'):][:400]
    assert "_prepared.capture." not in block
    assert re.search(r"getattr\(getattr\(_prepared, \"capture\", None\)", block)
    assert rr._admit_sent_edit(SimpleNamespace(messages=(), capture=None),
                            path="a.py", source=b"x\n")[0] is False
