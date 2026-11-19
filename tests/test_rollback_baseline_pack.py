import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

import kernel_effects
from kernel_effects import (
    ContentStore,
    EffectError,
    WorkspaceEffectPolicy,
    manifest_digest,
    reconcile,
    scan,
)
from kernel_shell_tool import ShellTool, ToolSettings


def _manifest(root):
    return scan(root, exclude=lambda _: False)


def _mixed_tree(root):
    nested = root / "nested" / "deeper"
    nested.mkdir(parents=True)
    (root / "nested").chmod(0o750)
    nested.chmod(0o711)
    for name, data, mode in (
        ("plain", b"\x00original\xff", 0o640),
        ("executable", b"\x00original\xff", 0o751),
        ("nested/empty", b"", 0o600),
        ("nested/deeper/data", b"different\n", 0o644),
    ):
        path = root / name
        path.write_bytes(data)
        path.chmod(mode)
    (root / "shortcut").symlink_to("nested/deeper/data")
    return root


def _bind(run, workspace, parent):
    run.mkdir(exist_ok=True)
    tool = ShellTool(run, ToolSettings(run_env=False), {"parent_snapshot": str(parent)},
                     SimpleNamespace(configure=lambda _: None))
    tool.bind(workspace, {"files": {}}, set())
    return tool


def _pack_root(run, manifest):
    return run / "shell-baseline" / manifest_digest(manifest)


def _reopen(run, manifest):
    return kernel_effects.BaselineStore(run / "shell-baseline", manifest)


def _tree_bytes_and_modes(root):
    tree = {}
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            data = os.readlink(os.fsencode(path))
        elif stat.S_ISREG(mode):
            data = path.read_bytes()
        else:
            data = None
        tree[path.relative_to(root)] = (mode, data)
    return tree


def _rollback(workspace, baseline, baseline_store, store):
    # Pre-call bytes differ from the pinned baseline, so rollback must use the baseline store.
    (workspace / "plain").write_bytes(b"pre-call bytes")
    before = _manifest(workspace)
    store.capture(workspace, before)
    for name in ("plain", "executable", "nested/empty", "nested/deeper/data"):
        (workspace / name).write_bytes(b"changed")
        (workspace / name).chmod(0o666)
    (workspace / "nested").chmod(0o700)
    (workspace / "created-link").symlink_to("plain")
    policy = WorkspaceEffectPolicy(lambda _: True, None, lambda _: True, baseline)
    result = reconcile(workspace, before, _manifest(workspace), policy, store, baseline_store)
    return result, _tree_bytes_and_modes(workspace)


def test_shell_rollback_matches_per_object_store(tmp_path):
    parent = _mixed_tree(tmp_path / "parent")
    legacy_tree = _mixed_tree(tmp_path / "legacy-workspace")
    packed_tree = _mixed_tree(tmp_path / "packed-workspace")
    original = _manifest(parent)
    legacy = ContentStore(tmp_path / "legacy")
    legacy.capture(parent, original)
    run = tmp_path / "run"
    tool = _bind(run, packed_tree, parent)

    assert tool.baseline == original
    assert _rollback(packed_tree, original, tool.baseline_store, tool.store) == _rollback(
        legacy_tree, original, legacy, ContentStore(tmp_path / "per-call"))
    assert _tree_bytes_and_modes(packed_tree) == _tree_bytes_and_modes(parent)

    root = _pack_root(run, original)
    assert {path.name for path in root.iterdir()} == {"objects.pack", "index.json"}
    index = json.loads((root / "index.json").read_bytes())
    pack = (root / "objects.pack").read_bytes()
    assert len(index) == 3
    assert len(pack) == len(b"\x00original\xffdifferent\n")
    for digest, (offset, length, mode) in index.items():
        assert pack[offset:offset + length] == legacy.get(digest)
        assert any(e.digest == digest and e.mode == mode for e in original.values())
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in root.iterdir())
    assert not any(path.is_file() for path in (run / "shell-baseline" / "cas").rglob("*"))


@pytest.mark.parametrize("when", ["before", "after"])
@pytest.mark.parametrize("operation", [
    "pack write", "pack fsync", "index write", "index fsync",
    "staging directory fsync", "publish", "store directory fsync", "run directory fsync",
])
def test_baseline_publication_survives_interrupted_write(tmp_path, monkeypatch, operation, when):
    parent = _mixed_tree(tmp_path / "parent")
    workspace = _mixed_tree(tmp_path / "workspace")
    manifest = _manifest(parent)
    run = tmp_path / "run"
    root = _pack_root(run, manifest)
    events = []
    real_stream = kernel_effects._stream_source
    real_dump = json.dump
    real_fsync = os.fsync
    real_replace = os.replace
    injected = False

    def observe(name, action, *args, **kwargs):
        nonlocal injected
        events.append(name)
        if name == operation and when == "before":
            injected = True
            raise OSError("injected write failure")
        result = action(*args, **kwargs)
        if name == operation and when == "after":
            injected = True
            raise OSError("injected write failure")
        return result

    def sync(fd):
        info = os.fstat(fd)
        if stat.S_ISDIR(info.st_mode):
            if os.path.samestat(info, run.stat()):
                name = "run directory fsync"
            elif os.path.samestat(info, root.parent.stat()):
                name = "store directory fsync"
            else:
                name = "staging directory fsync"
        else:
            name = "index fsync" if "index write" in events else "pack fsync"
        observe(name, real_fsync, fd)

    def replace(src, dst):
        assert Path(dst) == root
        assert "staging directory fsync" in events
        assert "index fsync" in events
        assert "pack fsync" in events
        observe("publish", real_replace, src, dst)

    with monkeypatch.context() as patch:
        patch.setattr(kernel_effects, "_stream_source",
                      lambda *args: observe("pack write", real_stream, *args))
        patch.setattr(json, "dump", lambda *args, **kwargs:
                      observe("index write", real_dump, *args, **kwargs))
        patch.setattr(os, "fsync", sync)
        patch.setattr(os, "replace", replace)
        with pytest.raises(OSError, match="injected write failure"):
            _bind(run, workspace, parent)
    assert injected
    published = operation in {"store directory fsync", "run directory fsync"} or (
        operation == "publish" and when == "after"
    )
    assert root.exists() == published
    if published:
        reopened = _reopen(run, manifest)
        for name, entry in manifest.items():
            if entry.kind == "file":
                assert reopened.get(entry.digest) == (parent / name).read_bytes()

    # A fresh binding can finish after any interruption, including after the rename.
    tool = _bind(run, workspace, parent)
    for name, entry in manifest.items():
        if entry.kind == "file":
            assert tool.baseline_store.get(entry.digest) == (parent / name).read_bytes()


def test_legacy_baseline_still_resolves(tmp_path):
    parent = _mixed_tree(tmp_path / "parent")
    manifest = _manifest(parent)
    run = tmp_path / "run"
    legacy = ContentStore(run / "shell-baseline")
    legacy.capture(parent, manifest)
    reopened = _reopen(run, manifest)
    for name, entry in manifest.items():
        if entry.kind == "file":
            assert reopened.get(entry.digest) == (parent / name).read_bytes()
    assert not _pack_root(run, manifest).exists()


def test_packed_baseline_checks_content_integrity(tmp_path):
    parent = _mixed_tree(tmp_path / "parent")
    run = tmp_path / "run"
    tool = _bind(run, parent, parent)
    root = _pack_root(run, tool.baseline)
    digest = tool.baseline["plain"].digest
    offset = json.loads((root / "index.json").read_bytes())[digest][0]
    pack = root / "objects.pack"
    pack.chmod(0o644)
    with pack.open("r+b") as output:
        output.seek(offset)
        output.write(b"!")
    with pytest.raises(EffectError, match="corrupt"):
        _reopen(run, tool.baseline).get(digest)
