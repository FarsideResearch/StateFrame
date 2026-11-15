"""Checkpoint build identity must agree across product entry points and detect code changes."""
from __future__ import annotations

from pathlib import Path
import contextvars
import json
import subprocess
import sys
import types

import pytest

from _timeouts import scaled
from stateframe import resume
from stateframe.engine import kernel_checkpoint as checkpoint


@pytest.fixture
def product_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    product = tmp_path / "stateframe"
    engine = product / "engine"
    engine.mkdir(parents=True)
    (engine / "core.py").write_text("VALUE = 1\n", encoding="utf-8")
    (product / "_child.py").write_text("CHILD = 1\n", encoding="utf-8")
    (product / "helper.py").write_text("HELPER = 1\n", encoding="utf-8")
    launcher = tmp_path / "launcher"
    launcher.mkdir()
    (launcher / "cli.py").write_text("CLI = 1\n", encoding="utf-8")
    host = types.ModuleType("_checkpoint_test_host")
    host.__file__ = str(product / "_child.py")
    monkeypatch.setitem(sys.modules, host.__name__, host)
    monkeypatch.setattr(checkpoint, "HOST_MODULE", host.__name__)
    monkeypatch.setattr(checkpoint, "_ENGINE_ROOT", str(engine))
    monkeypatch.setattr(checkpoint, "_FINGERPRINT", None)
    monkeypatch.setattr(checkpoint, "_PRODUCT_HOST", False)
    monkeypatch.setattr(checkpoint, "_HOST_IDENTITY_BOUND", False)
    return product, engine, launcher, host


@pytest.mark.parametrize("entrypoint", ["cli", "missing-file", "missing-module"])
def test_product_fingerprint_agrees_for_writer_and_host_validation(
        product_tree, monkeypatch: pytest.MonkeyPatch, entrypoint: str) -> None:
    _product, _engine, launcher, host = product_tree
    checkpoint.configure_product_host()
    writer = checkpoint.engine_fingerprint()
    if entrypoint == "cli":
        host.__file__ = str(launcher / "cli.py")
    elif entrypoint == "missing-file":
        del host.__file__
    else:
        monkeypatch.delitem(sys.modules, host.__name__)
    monkeypatch.setattr(checkpoint, "_FINGERPRINT", None)
    assert checkpoint.engine_fingerprint() == writer


@pytest.mark.parametrize("changed_file", ["helper.py", "engine/core.py"])
def test_host_validation_still_refuses_a_real_product_code_change(
        product_tree, monkeypatch: pytest.MonkeyPatch, changed_file: str) -> None:
    product, _engine, launcher, host = product_tree
    checkpoint.configure_product_host()
    host.__file__ = str(launcher / "cli.py")
    meta = {"python": list(sys.version_info[:3]),
            "engine_files_sha256": checkpoint.engine_fingerprint(), "entries": []}
    monkeypatch.setattr(checkpoint, "_authenticate_state", lambda _path: (meta, b"state"))
    assert checkpoint.validate(product / "snapshot") == meta
    (product / changed_file).write_text("VALUE = 2\n", encoding="utf-8")
    monkeypatch.setattr(checkpoint, "_FINGERPRINT", None)
    with pytest.raises(checkpoint.CheckpointError, match="another engine build"):
        checkpoint.validate(product / "snapshot")


def test_product_startup_replaces_a_cached_embedding_identity(
        product_tree, monkeypatch: pytest.MonkeyPatch) -> None:
    product, _engine, launcher, host = product_tree
    host.__file__ = str(launcher / "cli.py")
    embedding = checkpoint.engine_fingerprint()
    checkpoint.configure_product_host()
    product_identity = checkpoint.engine_fingerprint()
    assert product_identity != embedding
    host.__file__ = str(product / "_child.py")
    monkeypatch.setattr(checkpoint, "_FINGERPRINT", None)
    assert checkpoint.engine_fingerprint() == product_identity
    checkpoint.configure_product_host()
    assert checkpoint.engine_fingerprint() == product_identity


def _save_select_restore(run: Path, declaration: str) -> tuple[Path, str]:
    value = "saved frame value"
    checkpoint._STATE["setup"] = {"code": sys._getframe().f_code, "line": 0, "funcs": frozenset()}
    if declaration == "before-save":
        checkpoint.configure_product_host()
    fingerprint = checkpoint.engine_fingerprint()
    saved = checkpoint.save(sys._getframe(), 1, run)
    assert checkpoint.authenticate(saved)["engine_files_sha256"] == fingerprint
    if declaration == "between-save-and-select":
        checkpoint.configure_product_host()
    selected = resume.select_checkpoint(run, [{"send": 1}], run / "wire")
    assert selected is not None and selected["dir"] == str(saved)
    value = "changed after save"
    restored = checkpoint.restore(sys._getframe(), saved, run)
    assert value == "saved frame value"
    assert restored["engine_files_sha256"] == fingerprint
    return saved, fingerprint


@pytest.mark.parametrize("declaration", ["before-save", "between-save-and-select", "embedding"])
def test_same_process_save_select_restore_preserves_host_identity(
        product_tree, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declaration: str) -> None:
    _product, _engine, launcher, host = product_tree
    host.__file__ = str(launcher / "cli.py")
    monkeypatch.setenv(checkpoint.LEGACY_DIRECT_ENGINE_ENV, "1")
    monkeypatch.setattr(checkpoint, "_RUNNER_HOME_IDENTITY_CONFIGURED", False)
    for name in ("_STATE", "_SCANS", "_PARTICIPANTS", "_POOL_KNOWN"):
        monkeypatch.setattr(checkpoint, name, {})
    for name in ("_SECRETS", "_SIGNING_KEY"):
        monkeypatch.setattr(checkpoint, name, [])
    monkeypatch.setattr(checkpoint, "_SECRET_GENERATION", 0)
    monkeypatch.setattr(checkpoint, "_VERIFIED_POOL_IDENTITIES", set())
    monkeypatch.setattr(checkpoint, "_is_state_module", lambda *_args: False)
    monkeypatch.setattr(checkpoint, "_context_items", lambda: [])
    monkeypatch.setattr(checkpoint.contextvars, "copy_context", contextvars.Context)
    monkeypatch.setattr(checkpoint, "install_continuous_clock", lambda _saved: None)
    checkpoint.register_signing_key(b"host-identity-round-trip-key-0123456789")
    run = tmp_path / "run"
    run.mkdir()

    saved, fingerprint = _save_select_restore(run, declaration)

    # Also recompute: retaining cached bytes while switching host ownership is unsafe.
    monkeypatch.setattr(checkpoint, "_FINGERPRINT", None)
    assert checkpoint.engine_fingerprint() == fingerprint
    assert checkpoint.validate(saved)["engine_files_sha256"] == fingerprint
    if declaration == "embedding":
        (launcher / "cli.py").write_text("CLI = 2\n", encoding="utf-8")
        monkeypatch.setattr(checkpoint, "_FINGERPRINT", None)
        with pytest.raises(checkpoint.CheckpointError, match="another engine build"):
            checkpoint.validate(saved)


def test_standalone_engine_still_fingerprints_its_host(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = tmp_path / "engine"
    engine.mkdir()
    (engine / "core.py").write_text("VALUE = 1\n", encoding="utf-8")
    host_dir = tmp_path / "embedding"
    host_dir.mkdir()
    host_file = host_dir / "app.py"
    host_file.write_text("HOST = 1\n", encoding="utf-8")
    host = types.ModuleType("_checkpoint_test_host")
    host.__file__ = str(host_file)
    monkeypatch.setitem(sys.modules, host.__name__, host)
    monkeypatch.setattr(checkpoint, "HOST_MODULE", host.__name__)
    monkeypatch.setattr(checkpoint, "_ENGINE_ROOT", str(engine))
    monkeypatch.setattr(checkpoint, "_FINGERPRINT", None)
    monkeypatch.setattr(checkpoint, "_PRODUCT_HOST", False)
    monkeypatch.setattr(checkpoint, "_HOST_IDENTITY_BOUND", False)
    original = checkpoint.engine_fingerprint()
    host_file.write_text("HOST = 2\n", encoding="utf-8")
    monkeypatch.setattr(checkpoint, "_FINGERPRINT", None)
    assert checkpoint.engine_fingerprint() != original


def test_installed_embedding_host_change_refuses_signed_checkpoint_in_fresh_process(
        product_tree, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _product, engine, launcher, host = product_tree
    host_file = launcher / "cli.py"
    host.__file__ = str(host_file)
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setenv(checkpoint.LEGACY_DIRECT_ENGINE_ENV, "1")
    monkeypatch.setattr(checkpoint, "_RUNNER_HOME_IDENTITY_CONFIGURED", False)
    monkeypatch.setattr(checkpoint, "_STATE", {})
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    monkeypatch.setattr(checkpoint, "_SECRET_GENERATION", 0)
    monkeypatch.setattr(checkpoint, "_SIGNING_KEY", [])
    monkeypatch.setattr(checkpoint, "_VERIFIED_POOL_IDENTITIES", set())
    monkeypatch.setattr(checkpoint, "_POOL_KNOWN", {})
    signing_key = b"embedding-checkpoint-test-key-0123456789"
    checkpoint.register_signing_key(signing_key)
    context = {name: {} for name in ("hostfns", "participants", "framecells", "framefuncs", "cvars")}
    monkeypatch.setattr(checkpoint, "collect", lambda _frame: ({"frame": {}}, context))
    saved = checkpoint.save(None, 1, run)
    original = checkpoint.validate(saved)["engine_files_sha256"]
    # Import the actual installed engine API in another interpreter; the synthetic
    # package has the normal _child.py sibling, but the host belongs to an embedding.
    code = "\n".join([
        "import json, sys, types",
        "from pathlib import Path",
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[1] / 'src')!r})",
        "from stateframe.engine import kernel_checkpoint as checkpoint",
        f"host = types.ModuleType({host.__name__!r})",
        f"host.__file__ = {str(host_file)!r}",
        "sys.modules[host.__name__] = host",
        "checkpoint.HOST_MODULE = host.__name__",
        f"checkpoint._ENGINE_ROOT = {str(engine)!r}",
        f"checkpoint.register_signing_key({signing_key!r})",
        "print(json.dumps({'fingerprint': checkpoint.engine_fingerprint()}), flush=True)",
        "try:",
        f"    checkpoint.validate(Path({str(saved)!r}))",
        "except checkpoint.CheckpointError as error:",
        "    print(str(error))",
        "    sys.exit(7)",
    ])

    def resume():
        return subprocess.run([sys.executable, "-I", "-B", "-c", code],
                              capture_output=True, text=True, timeout=scaled(30))

    unchanged = resume()
    assert unchanged.returncode == 0, unchanged.stderr + unchanged.stdout
    assert json.loads(unchanged.stdout.splitlines()[0])["fingerprint"] == original
    host_file.write_text("CLI = 2\n", encoding="utf-8")
    changed = resume()
    assert json.loads(changed.stdout.splitlines()[0])["fingerprint"] != original
    assert changed.returncode == 7, changed.stderr + changed.stdout
    assert "another engine build" in changed.stdout
