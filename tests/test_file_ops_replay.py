"""Native file mutations replay from a fresh tree and refuse divergent history."""
import pytest

from kernel_file_ops import FileOperationReplay, file_delete, file_move
from kernel_tool_tape import configure_replay, tape_for_run, ToolReplayRefused


def begin(root, operation, *, identity="env", authority=None):
    arguments = ({"source": "old.py", "destination": "new.py"}
                 if operation == "file_move" else {"path": "old.py"})
    return FileOperationReplay(root, operation, arguments, "call-1",
                               authority=authority or {"editable": ["old.py"]},
                               environment_identity=identity)


def mutate(root, operation):
    options = dict(protected=lambda _: False, editable=lambda _: True)
    if operation == "file_move":
        file_move(root, "old.py", "new.py", new_file_allowed=lambda _: True, **options)
    else:
        file_delete(root, "old.py", **options)


@pytest.mark.parametrize("operation", ["file_move", "file_delete"])
@pytest.mark.parametrize("difference", [None, "pre-state", "post-state", "environment", "authority", "pending"])
def test_native_operation_tape_replays_or_refuses(tmp_path, operation, difference):
    run = tmp_path / "run"
    run.mkdir()
    original, replay = tmp_path / "original", tmp_path / "replay"
    for root in (original, replay):
        root.mkdir()
        (root / "old.py").write_text("VALUE = 1\n")
    configure_replay(run, active=False)
    first = begin(original, operation)
    mutate(original, operation)
    if difference != "pending":
        assert first.complete("APPLIED", "original observation") == "original observation"
    configure_replay(run, active=True)
    if difference == "pre-state":
        (replay / "extra.py").write_text("extra")
    if difference in {"pre-state", "environment", "authority", "pending"}:
        with pytest.raises(ToolReplayRefused):
            begin(replay, operation, identity="changed" if difference == "environment" else "env",
                  authority={"editable": []} if difference == "authority" else None)
        assert (replay / "old.py").exists()
        return
    again = begin(replay, operation)
    mutate(replay, operation)
    if difference == "post-state":
        (replay / "extra.py").write_text("extra")
        with pytest.raises(ToolReplayRefused, match="workspace manifest"):
            again.complete("APPLIED", "new observation")
    else:
        assert again.complete("APPLIED", "new observation") == "original observation"
        assert not (replay / "old.py").exists()
        assert (replay / "new.py").exists() == (operation == "file_move")
        assert tape_for_run().lookup("call-1").completed
