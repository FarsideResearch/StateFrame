"""Secret path exclusions preserve source in the copy and mutation inputs."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from kernel_mutate_inputs import input_path_admitted

from stateframe.workcopy import create, frame_source_path_allowed


KEPT = (
    "secret_binding.py", "secrets_manager.py", "credential_store.py", "id_utils.py",
    "app/models/secret_binding.py", ".env.schema.ts", "id_rsa_tools.go",
    "id_dsa_config.rs", "id_ecdsa_loader.java", "id_ed25519_check.rb", ".env.setup.js",
    "id_rsa_codec.c", "id_rsa_codec.H", "credentials.example.json", "secret_helper.sh",
    "credential_store/module.py", ".envrc.example",
)
EXCLUDED = (
    ".env", ".env.local", "server.pem", "id_rsa", "id_ed25519.pub",
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
