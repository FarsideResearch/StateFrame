"""Checkpoint build identity must agree across product entry points and detect code changes."""
from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest

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
    return product, engine, launcher, host


@pytest.mark.parametrize("entrypoint", ["cli", "missing-file", "missing-module"])
def test_product_fingerprint_agrees_for_writer_and_host_validation(
        product_tree, monkeypatch: pytest.MonkeyPatch, entrypoint: str) -> None:
    _product, _engine, launcher, host = product_tree
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
    host.__file__ = str(launcher / "cli.py")
    meta = {"python": list(sys.version_info[:3]),
            "engine_files_sha256": checkpoint.engine_fingerprint(), "entries": []}
    monkeypatch.setattr(checkpoint, "_authenticate_state", lambda _path: (meta, b"state"))
    assert checkpoint.validate(product / "snapshot") == meta
    (product / changed_file).write_text("VALUE = 2\n", encoding="utf-8")
    monkeypatch.setattr(checkpoint, "_FINGERPRINT", None)
    with pytest.raises(checkpoint.CheckpointError, match="another engine build"):
        checkpoint.validate(product / "snapshot")


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
    original = checkpoint.engine_fingerprint()
    host_file.write_text("HOST = 2\n", encoding="utf-8")
    monkeypatch.setattr(checkpoint, "_FINGERPRINT", None)
    assert checkpoint.engine_fingerprint() != original
