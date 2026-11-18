"""ContentStore.capture fast path for objects already present in the CAS.

RED on the pre-change tree: a second capture of unchanged files still creates a temp file,
fsyncs it and walks the link path even though the object already exists. GREEN: a known digest
only hashes the live source (same drift checks) and verifies the stored object.

The parity test pins, per edge case, the exception type/errno/message and the resulting CAS
tree (paths, modes, bytes) to what the pre-change implementation produced.
"""

import hashlib
import os
import tempfile
from pathlib import Path

import pytest

from kernel_effects import ContentStore, EffectError, scan


def _workspace(root: Path, files: dict[str, bytes]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (root / name).write_bytes(data)
    return root


def _spy(monkeypatch: pytest.MonkeyPatch, owner: object, name: str) -> list[str]:
    calls: list[str] = []
    real = getattr(owner, name)

    def spy(*args: object, **kwargs: object) -> object:
        calls.append(name)
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, name, spy)
    return calls


def test_known_object_capture_skips_temp_fsync_and_link(tmp_path, monkeypatch):
    root = _workspace(tmp_path / "workspace", {"a.txt": b"alpha", "b.txt": b"beta"})
    store = ContentStore(tmp_path / "run")
    manifest = scan(root, exclude=lambda _: False)
    store.capture(root, manifest)

    fsyncs = _spy(monkeypatch, os, "fsync")
    temps = _spy(monkeypatch, tempfile, "mkstemp")
    links = _spy(monkeypatch, os, "link")
    store.capture(root, manifest)

    assert (fsyncs, temps, links) == ([], [], [])
    assert store.get(manifest["a.txt"].digest) == b"alpha"
    assert not list(store.root.rglob(".cas-*"))


# --- parity: outcomes pinned to the pre-change implementation -------------------------------

PRE_CAPTURED = {
    "existing_ok",
    "existing_corrupt",
    "existing_symlink",
    "existing_source_drift",
    "existing_source_and_object_corrupt",
}


def _run_scenario(name: str, tmp_path: Path) -> tuple[tuple, list]:
    root = _workspace(tmp_path / "workspace", {"a.txt": b"alpha", "b.txt": b"beta"})
    store = ContentStore(tmp_path / "run")
    manifest = scan(root, exclude=lambda _: False)
    obj = store._path(manifest["a.txt"].digest)
    if name in PRE_CAPTURED:
        store.capture(root, manifest)

    if name in {"existing_corrupt", "existing_source_and_object_corrupt"}:
        obj.chmod(0o644)
        obj.write_bytes(b"omega")  # same size, wrong bytes
    if name == "existing_symlink":
        obj.unlink()
        obj.symlink_to(root / "b.txt")
    if name in {"new_source_drift", "existing_source_drift", "existing_source_and_object_corrupt"}:
        (root / "a.txt").write_bytes(b"ALPHA")  # same size, different bytes
    if name == "source_size_changed":
        with open(root / "a.txt", "ab") as handle:
            handle.write(b"!")
    if name == "source_hardlinked":
        os.link(root / "a.txt", tmp_path / "alias.txt")
    if name == "source_swapped_symlink":
        (root / "a.txt").unlink()
        (root / "a.txt").symlink_to(root / "b.txt")

    try:
        store.capture(root, manifest)
        outcome: tuple = ("ok",)
    except EffectError as error:
        outcome = ("EffectError", str(error))
    except OSError as error:
        outcome = ("OSError", error.errno)

    cas: list = []
    for path in sorted(store.root.rglob("*")):
        rel = path.relative_to(store.root).as_posix()
        if path.is_symlink():
            cas.append((rel, "symlink"))
        elif path.is_dir():
            cas.append((rel, "dir"))
        else:
            cas.append(
                (rel, oct(path.stat().st_mode & 0o777), hashlib.sha256(path.read_bytes()).hexdigest())
            )
    return outcome, cas


# Expected values were recorded on the pre-change implementation (71c6e4e1).
GOLDEN: dict[str, tuple[tuple, list]] = {
    'new_ok': (       ('ok',),
        [       ('8e', 'dir'),
                (       '8e/d3f6ad685b959ead7022518e1af76cd816f8e8ec7ccdda1ed4018e8f2223f8',
                        '0o444',
                        '8ed3f6ad685b959ead7022518e1af76cd816f8e8ec7ccdda1ed4018e8f2223f8'),
                ('f4', 'dir'),
                (       'f4/4e64e75f3948e9f73f8dfa94721c4ce8cbb4f265c4790c702b2d41cfbf2753',
                        '0o444',
                        'f44e64e75f3948e9f73f8dfa94721c4ce8cbb4f265c4790c702b2d41cfbf2753')]),
    'new_source_drift': (       ('EffectError', 'workspace changed while capturing a.txt'),
        [       ('8e', 'dir'),
                ('f4', 'dir'),
                (       'f4/4e64e75f3948e9f73f8dfa94721c4ce8cbb4f265c4790c702b2d41cfbf2753',
                        '0o444',
                        'f44e64e75f3948e9f73f8dfa94721c4ce8cbb4f265c4790c702b2d41cfbf2753')]),
    'existing_ok': (       ('ok',),
        [       ('8e', 'dir'),
                (       '8e/d3f6ad685b959ead7022518e1af76cd816f8e8ec7ccdda1ed4018e8f2223f8',
                        '0o444',
                        '8ed3f6ad685b959ead7022518e1af76cd816f8e8ec7ccdda1ed4018e8f2223f8'),
                ('f4', 'dir'),
                (       'f4/4e64e75f3948e9f73f8dfa94721c4ce8cbb4f265c4790c702b2d41cfbf2753',
                        '0o444',
                        'f44e64e75f3948e9f73f8dfa94721c4ce8cbb4f265c4790c702b2d41cfbf2753')]),
    'existing_corrupt': (       (       'EffectError',
                'CAS object 8ed3f6ad685b959ead7022518e1af76cd816f8e8ec7ccdda1ed4018e8f2223f8 is '
                'corrupt'),
        [       ('8e', 'dir'),
                (       '8e/d3f6ad685b959ead7022518e1af76cd816f8e8ec7ccdda1ed4018e8f2223f8',
                        '0o644',
                        '304b4a90a76a1cbe4c112e074b30e75181f54df43d60f883597457844293b341'),
                ('f4', 'dir'),
                (       'f4/4e64e75f3948e9f73f8dfa94721c4ce8cbb4f265c4790c702b2d41cfbf2753',
                        '0o444',
                        'f44e64e75f3948e9f73f8dfa94721c4ce8cbb4f265c4790c702b2d41cfbf2753')]),
    'existing_symlink': (       ('OSError', 40),
        [       ('8e', 'dir'),
                ('8e/d3f6ad685b959ead7022518e1af76cd816f8e8ec7ccdda1ed4018e8f2223f8', 'symlink'),
                ('f4', 'dir'),
                (       'f4/4e64e75f3948e9f73f8dfa94721c4ce8cbb4f265c4790c702b2d41cfbf2753',
                        '0o444',
                        'f44e64e75f3948e9f73f8dfa94721c4ce8cbb4f265c4790c702b2d41cfbf2753')]),
    'existing_source_drift': (       ('EffectError', 'workspace changed while capturing a.txt'),
        [       ('8e', 'dir'),
                (       '8e/d3f6ad685b959ead7022518e1af76cd816f8e8ec7ccdda1ed4018e8f2223f8',
                        '0o444',
                        '8ed3f6ad685b959ead7022518e1af76cd816f8e8ec7ccdda1ed4018e8f2223f8'),
                ('f4', 'dir'),
                (       'f4/4e64e75f3948e9f73f8dfa94721c4ce8cbb4f265c4790c702b2d41cfbf2753',
                        '0o444',
                        'f44e64e75f3948e9f73f8dfa94721c4ce8cbb4f265c4790c702b2d41cfbf2753')]),
    'existing_source_and_object_corrupt': (       ('EffectError', 'workspace changed while capturing a.txt'),
        [       ('8e', 'dir'),
                (       '8e/d3f6ad685b959ead7022518e1af76cd816f8e8ec7ccdda1ed4018e8f2223f8',
                        '0o644',
                        '304b4a90a76a1cbe4c112e074b30e75181f54df43d60f883597457844293b341'),
                ('f4', 'dir'),
                (       'f4/4e64e75f3948e9f73f8dfa94721c4ce8cbb4f265c4790c702b2d41cfbf2753',
                        '0o444',
                        'f44e64e75f3948e9f73f8dfa94721c4ce8cbb4f265c4790c702b2d41cfbf2753')]),
    'source_size_changed': (       ('EffectError', 'workspace changed while capturing a.txt'),
        [       ('f4', 'dir'),
                (       'f4/4e64e75f3948e9f73f8dfa94721c4ce8cbb4f265c4790c702b2d41cfbf2753',
                        '0o444',
                        'f44e64e75f3948e9f73f8dfa94721c4ce8cbb4f265c4790c702b2d41cfbf2753')]),
    'source_hardlinked': (       ('EffectError', 'workspace changed while capturing a.txt'),
        [       ('f4', 'dir'),
                (       'f4/4e64e75f3948e9f73f8dfa94721c4ce8cbb4f265c4790c702b2d41cfbf2753',
                        '0o444',
                        'f44e64e75f3948e9f73f8dfa94721c4ce8cbb4f265c4790c702b2d41cfbf2753')]),
    'source_swapped_symlink': (       ('OSError', 40),
        [       ('f4', 'dir'),
                (       'f4/4e64e75f3948e9f73f8dfa94721c4ce8cbb4f265c4790c702b2d41cfbf2753',
                        '0o444',
                        'f44e64e75f3948e9f73f8dfa94721c4ce8cbb4f265c4790c702b2d41cfbf2753')]),
}


@pytest.mark.parametrize(
    "name",
    [
        "new_ok",
        "new_source_drift",
        "existing_ok",
        "existing_corrupt",
        "existing_symlink",
        "existing_source_drift",
        "existing_source_and_object_corrupt",
        "source_size_changed",
        "source_hardlinked",
        "source_swapped_symlink",
    ],
)
def test_capture_outcomes_match_pre_change_behaviour(tmp_path, name):
    assert _run_scenario(name, tmp_path) == GOLDEN[name]
