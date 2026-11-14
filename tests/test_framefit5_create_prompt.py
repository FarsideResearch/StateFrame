"""Regression coverage for targetless FrameFit units and their created product."""

from probes import react_repair


def test_framefit_targetless_task_keeps_full_prompt(monkeypatch):
    monkeypatch.setenv("STATEFRAME_FRAMEFIT", "on")
    content = react_repair._initial_task_content(
        "Subframe create: Create pkg/new.py.",
        "Create only the declared file.",
        "Visible check: tests/test_create.py.",
    )

    assert content == (
        "Task: Subframe create: Create pkg/new.py.\n\n"
        "Create only the declared file.\n\n"
        "Visible check: tests/test_create.py.\nBegin."
    )


def test_non_framefit_prompt_keeps_existing_label(monkeypatch):
    monkeypatch.setenv("STATEFRAME_FRAMEFIT", "off")

    assert react_repair._initial_task_content("Repair value.", "Inspect pkg/a.py.", "") == (
        "Bug report: Repair value.\n\nInspect pkg/a.py.\nBegin."
    )


def test_declared_create_is_captured_as_product(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/new.py").write_text("VALUE = 1\n")
    (tmp_path / "pkg/unrelated.py").write_text("VALUE = 2\n")
    row = {
        "tier": "USER-REPO",
        "scoped_edits": True,
        "files": {},
        "allowed_product_paths": [],
        "new_file_roots": ["pkg/new.py"],
    }

    assert react_repair._product_source_paths(
        tmp_path, react_repair.FileSet.from_row(row), row,
    ) == ("pkg/new.py",)
