from pathlib import Path
from types import SimpleNamespace

from stateframe.engine.kernel_effects import Effect, Entry
from stateframe.engine.kernel_shell_tool import ShellReply, publish_effects


def test_off_mode_shell_directory_is_not_tracked_as_created_file(tmp_path, monkeypatch):
    from stateframe.engine import kernel_green_lock

    monkeypatch.setattr(kernel_green_lock, "note_applied_product_edit", lambda _path: None)
    receipts = []
    bolt = SimpleNamespace(root=tmp_path, product_edits=0, last_test_exit=0,
                           _last_green_revision="green",
                           receipt=lambda *args, **kwargs: receipts.append((args, kwargs)))
    runtime = SimpleNamespace(target=Path(tmp_path / "existing.py"))
    effects = (Effect("new-directory", "created", None, Entry("dir", 0o755, None, 0)),
               Effect("new-file.py", "created", None, Entry("file", 0o644, "a" * 64, 1)))
    created = set()

    publish_effects(ShellReply("created", effects), runtime, bolt, created, name="shell")

    assert created == {str(tmp_path / "new-file.py")}
    assert receipts[0][1]["changed_paths"] == ("new-directory", "new-file.py")
    assert receipts[0][1]["effects"] == (
        {"path": "new-directory", "change": "created"},
        {"path": "new-file.py", "change": "created"},
    )
