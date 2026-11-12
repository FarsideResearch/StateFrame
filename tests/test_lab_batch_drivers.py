"""Extracted batch commands retain their empty-input and output contracts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType


LAB = Path(__file__).resolve().parents[1] / "bench/lab"


def _driver(name: str) -> ModuleType:
    path = LAB / f"{name}_driver.py"
    spec = importlib.util.spec_from_file_location(f"lab_{name}_driver", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def test_swebench_adapter_empty_corpus(tmp_path: Path) -> None:
    driver = _driver("swebench_adapter")
    source = tmp_path / "instances.jsonl"
    output = tmp_path / "rows.jsonl"
    skipped = tmp_path / "skipped.json"
    source.write_text("", encoding="utf-8")

    assert driver.main([
        "--source", str(source), "--output", str(output),
        "--skipped-output", str(skipped), "--cache-dir", str(tmp_path / "cache"),
    ]) == 0
    assert output.read_text(encoding="utf-8") == ""
    assert json.loads(skipped.read_text(encoding="utf-8"))["considered"] == 0


def test_swebench_env_empty_rows(tmp_path: Path) -> None:
    driver = _driver("swebench_env")
    rows = tmp_path / "rows.jsonl"
    output = tmp_path / "enriched.jsonl"
    proof = tmp_path / "proof.json"
    rows.write_text("", encoding="utf-8")

    assert driver.main([
        "--rows", str(rows), "--cache-dir", str(tmp_path / "envs"),
        "--output", str(output), "--proof-output", str(proof),
    ]) == 0
    assert output.read_text(encoding="utf-8") == ""
    assert proof.is_file()


def test_realtask_empty_candidates(tmp_path: Path, monkeypatch) -> None:
    driver = _driver("realtask_benchmark")
    monkeypatch.setattr(driver, "extract_dev_tasks", lambda *args, **kwargs: ())
    outputs = {
        name: tmp_path / f"{name}.jsonl"
        for name in ("task", "validation", "pending", "controller")
    }

    assert driver.main([
        "--repo", str(tmp_path),
        "--task-output", str(outputs["task"]),
        "--validation-output", str(outputs["validation"]),
        "--pending-output", str(outputs["pending"]),
        "--controller-output", str(outputs["controller"]),
        "--snapshot-dir", str(tmp_path / "snapshots"),
    ]) == 2
    assert all(path.read_text(encoding="utf-8") == "" for path in outputs.values())
