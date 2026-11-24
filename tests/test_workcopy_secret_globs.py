"""Secret path exclusions preserve source in the copy and mutation inputs."""
from __future__ import annotations

import errno
import subprocess
import sys
import tracemalloc
from pathlib import Path

import pytest

from kernel_mutate_inputs import input_path_admitted

from stateframe.workcopy import WorkCopyError, create, frame_source_path_allowed


KEPT = (
    "secret_binding.py", "secrets_manager.py", "credential_store.py", "id_utils.py",
    "app/models/secret_binding.py", ".env.schema.ts", "id_rsa_tools.go",
    "id_dsa_config.rs", "id_ecdsa_loader.java", "id_ed25519_check.rb", ".env.setup.js",
    "id_rsa_codec.c", "id_rsa_codec.H", "secret_helper.sh",
    "credential_store/module.py", ".envrc.example",
)
EXCLUDED = (
    ".env", ".env.local", "server.pem", "id_rsa", "id_ed25519.pub",
    # Data/config files that name a secret or credential stay out even when the name is not a fixed rule.
    "client_secret.json", "credentials.yaml", "aws_credentials.ini", "credentials.example.json",
    "credentials.json", ".netrc", "secrets.yaml", "config/app.secret",
    "server.key", "id_dsa", "id_ecdsa", "server.p12", "server.pfx", ".npmrc", ".pypirc",
    "credentials", ".git-credentials", "secrets.json", "secrets.yml", "secrets.toml",
    "config/app.secrets", "nested/.secrets/module.py", "nested/.SECRETS/other.py",
    "nested/SERVER.PEM",
)


def _write(repo: Path, path: str, content: str = "value = 42\n") -> None:
    file = repo / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(content, encoding="utf-8")


def test_workcopy_filters_secret_material(tmp_path):
    repo = tmp_path / "repo"
    for path in (*KEPT, *EXCLUDED):
        _write(repo, path)
    copy = create(repo, tmp_path / "copy")

    assert set(KEPT) - copy.baseline.keys() == set()
    assert set(EXCLUDED) & copy.baseline.keys() == set()
    assert all((copy.copy / path).is_file() and frame_source_path_allowed(path) for path in KEPT)
    assert all(not (copy.copy / path).exists() and not frame_source_path_allowed(path) for path in EXCLUDED)


def test_workcopy_accepts_more_than_5000_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    for index in range(5_001):
        (repo / f"file-{index:05d}.txt").touch()

    copy = create(repo, tmp_path / "copy")

    assert len(copy.baseline) == 5_001


def test_repository_test_reads_a_twenty_mib_file_while_large_sources_stay_out_of_text(tmp_path):
    import hashlib

    repo = tmp_path / "repo"
    repo.mkdir()
    payload = repo / "payload.bin"
    with payload.open("wb") as stream:
        stream.truncate(20 << 20)
    generated = repo / "generated.py"
    generated.write_bytes(b"# generated source\n" + b"x" * (3 << 20))
    (repo / "test_payload_reader.py").write_text(
        "from pathlib import Path\nassert len(Path('payload.bin').read_bytes()) == 20 << 20\n",
        encoding="utf-8",
    )

    copy = create(repo, tmp_path / "copy", admit_snapshot=False)

    assert set(copy.baseline) == {"payload.bin", "generated.py", "test_payload_reader.py"}
    assert copy.baseline["payload.bin"] == hashlib.sha256(bytes(1 << 20) * 20).hexdigest()
    assert "payload.bin" not in copy.text_files()
    assert "generated.py" not in copy.text_files()
    from stateframe.runner import candidate_digest
    delivered = {}
    candidate_digest(copy.copy, file_digests=delivered)
    assert delivered["payload.bin"] == copy.baseline["payload.bin"]
    result = subprocess.run(
        ["bash", "-c", "ulimit -v 16000000; exec \"$1\" test_payload_reader.py", "repo-test", sys.executable],
        cwd=copy.copy, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_large_text_changes_are_loaded_for_diff_but_unchanged_files_are_not(tmp_path, monkeypatch):
    from stateframe import workcopy

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "generated.py").write_bytes(b"x" * (3 << 20))
    (repo / "unchanged.txt").write_bytes(b"steady\n" * (1 << 19))
    copy = create(repo, tmp_path / "copy", admit_snapshot=False)
    target = copy.copy / "generated.py"
    unchanged = copy.copy / "unchanged.txt"
    with target.open("r+b") as stream:
        stream.write(b"y")

    original_read_bytes = Path.read_bytes
    large_reads = []

    def reject_large_read(path):
        if path == unchanged:
            raise AssertionError("unchanged large files must not be loaded")
        if path == target:
            large_reads.append(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_large_read)
    assert workcopy.changed_files(copy) == {"generated.py": "modified"}
    assert "generated.py" in workcopy.unified_diff(copy)
    assert copy.text_files()["generated.py"].startswith("y")
    assert large_reads


def test_copying_a_three_hundred_mib_file_keeps_python_memory_flat(tmp_path, monkeypatch):
    from stateframe import workcopy

    repo = tmp_path / "repo"
    repo.mkdir()
    with (repo / "large.bin").open("wb") as stream:
        stream.truncate(300 << 20)

    def refuse_clone(_source, _target):
        raise OSError(errno.EIO, "clone refused")

    monkeypatch.setattr(workcopy, "_reflink", refuse_clone)
    tracemalloc.start()
    copy = create(repo, tmp_path / "copy", admit_snapshot=False)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert "large.bin" in copy.baseline
    assert (copy.copy / "large.bin").stat().st_size == 300 << 20
    assert peak < 8 << 20


@pytest.mark.parametrize("admit_snapshot", [None, False])
def test_workcopy_accepts_empty_repository(tmp_path, admit_snapshot):
    repo = tmp_path / "repo"
    repo.mkdir()

    copy = create(repo, tmp_path / "copy", admit_snapshot=admit_snapshot)

    assert copy.copy.is_dir()
    assert copy.baseline == {}


def test_workcopy_removes_partial_copy_on_enospc(tmp_path, monkeypatch):
    from stateframe import workspace_start_snapshot

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "answer.py").write_text("VALUE = 1\n")
    destination = tmp_path / "copy"
    copy_file = workspace_start_snapshot._Copies.file

    def fail_after_partial(self, source, path, info, *, data=None, writable=False, before_write=None):
        if path == destination / "answer.py":
            with path.open("xb") as output:
                output.write(b"V")
            raise OSError(errno.ENOSPC, "No space left on device")
        return copy_file(self, source, path, info, data=data, writable=writable,
                         before_write=before_write)

    monkeypatch.setattr(workspace_start_snapshot._Copies, "file", fail_after_partial)
    with pytest.raises(WorkCopyError, match="repository copy needs 20 bytes"):
        create(repo, destination, admit_snapshot=False)
    assert not destination.exists()


def test_mutation_inputs_filter_secret_material(tmp_path):
    repo = tmp_path / "repo"
    for path in (*KEPT, *EXCLUDED):
        _write(repo, path)

    assert [path for path in KEPT if not input_path_admitted(repo, path, object())] == []
    assert [path for path in EXCLUDED if input_path_admitted(repo, path, object())] == []


def test_workcopy_test_can_import_secret_binding(tmp_path):
    repo = tmp_path / "repo"
    _write(repo, "pkg/__init__.py", "")
    _write(repo, "pkg/secret_binding.py")
    test_file = "test_binding.py"
    _write(repo, test_file, "from pkg.secret_binding import value\n\nassert value == 42\n")

    copy = create(repo, tmp_path / "copy")
    result = subprocess.run(
        [sys.executable, "-E", test_file], cwd=copy.copy, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert (copy.copy / "pkg/secret_binding.py").is_file()
