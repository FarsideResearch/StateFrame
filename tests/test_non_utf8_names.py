"""File names that are not valid UTF-8 are left alone instead of ending the run."""
import os
import sys
from pathlib import Path

import pytest

from stateframe import workcopy

sys.path.insert(0, str(Path(workcopy.__file__).parent / "engine"))
import kernel_effects  # noqa: E402
import kernel_shell_tool  # noqa: E402

BAD = os.fsdecode(b"bad-\xff.py")
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="needs arbitrary byte file names")


@pytest.mark.parametrize("admit", [False, True])
def test_start_copy_leaves_the_name_out(tmp_path, admit):
    repo = tmp_path / "repo"
    (repo / os.fsdecode(b"dir-\xfe")).mkdir(parents=True)
    (repo / "keep.py").write_text("A = 1\n")
    (repo / "café.py").write_text("B = 1\n")
    (repo / BAD).write_text("C = 1\n")
    (repo / os.fsdecode(b"dir-\xfe") / "in.py").write_text("D = 1\n")
    work = workcopy.create(repo, tmp_path / "copy", admit_snapshot=admit)
    assert set(work.baseline) == {"keep.py", "café.py"}
    (work.copy / "keep.py").write_text("A = 2\n")
    assert workcopy.apply_in_place(work) == ["keep.py"]
    assert (repo / BAD).read_text() == "C = 1\n"
    assert (repo / os.fsdecode(b"dir-\xfe") / "in.py").read_text() == "D = 1\n"


@pytest.mark.parametrize("name,allowed", [(BAD, False), ("café.py", True), ("\\xff.py", True)])
def test_shell_created_name_is_restored(name, allowed):
    policy = kernel_effects.WorkspaceEffectPolicy(lambda _: False, lambda _: True, lambda _: True)
    entry = kernel_effects.Entry("file", 0o644, "0" * 64, 1)
    ok, reason = policy.authorize(kernel_effects.Effect(name, "created", None, entry))
    assert ok is allowed
    if not allowed:
        assert reason == "file names must be valid UTF-8"


def test_shell_summary_prints_the_name_escaped():
    assert kernel_shell_tool.bounded_summary(BAD) == "bad-\\udcff.py"
    for text in ("plain", "café", "x" * 4097):
        assert kernel_shell_tool.bounded_summary(text).startswith(text[:4096])


@pytest.mark.parametrize("tool,arguments", [
    ("file_create", {"path": BAD, "content": "x"}), ("file_move", {"source": "a.py", "destination": BAD}),
    ("git", {"action": "show", "path": BAD})])
def test_tool_arguments_with_the_name_have_a_digest(tool, arguments):
    from kernel_frame_series import canonical_argument_digest

    escaped = {key: value.replace(BAD, "bad-\\udcff.py") for key, value in arguments.items()}
    assert canonical_argument_digest(tool, arguments) != canonical_argument_digest(tool, escaped)


def test_tool_tape_keeps_arguments_with_the_name(tmp_path):
    import kernel_tool_tape

    kernel_tool_tape.ToolTape(tmp_path).intent(
        call_id="call-1", tool="shell", args={"command": f"cat {BAD}"}, replay_class="observation",
        capability_digest="capability", definition_digest=None, env_identity="disabled",
        pre_manifest_digest="manifest")
    restored = kernel_tool_tape.ToolTape(tmp_path).lookup("call-1")
    assert restored is not None and restored.args == {"command": f"cat {BAD}"}


@pytest.mark.parametrize("child", [False, True])
def test_directory_renamed_to_the_name_is_restored(tmp_path, child):
    root = tmp_path / "workspace"
    (root / "source").mkdir(parents=True)
    if child:
        (root / "source" / "in.py").write_text("A = 1\n")
    before = kernel_effects.scan(root, exclude=lambda _: False)
    store = kernel_effects.ContentStore(tmp_path / "store")
    store.capture(root, before)
    os.rename(root / "source", root / os.fsdecode(b"dir-\xfe"))
    after = kernel_effects.scan(root, exclude=lambda _: False)
    policy = kernel_effects.WorkspaceEffectPolicy(lambda _: False, lambda _: True, lambda _: True)
    kernel_effects.reconcile(root, before, after, policy, store, None)
    assert (root / "source").is_dir()
    assert not child or (root / "source" / "in.py").read_text() == "A = 1\n"
    assert os.listdir(root) == ["source"]
