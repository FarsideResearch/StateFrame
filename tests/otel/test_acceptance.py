from __future__ import annotations

import ast
from fnmatch import fnmatchcase
import importlib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

import pytest

from stateframe import __version__
from stateframe.otel import DETAIL_FLAGS
from stateframe.otel import config


ROOT = Path(__file__).resolve().parents[2]


def _registry() -> dict[str, object]:
    return json.loads((ROOT / "src/stateframe/otel/switch-registry.json").read_text(encoding="utf-8"))


def test_registry_matches_actual_config_behavior():
    registry = _registry()
    names = {row["name"] for row in registry["switches"]}
    enabled = config.resolve(config.Settings(), {"OTEL_TRACES_EXPORTER": "otlp"})
    disabled = config.resolve(config.Settings(), {"OTEL_TRACES_EXPORTER": "none"})

    assert registry["schema"] == "stateframe-otel-switches/1"
    assert {"OTEL_SDK_DISABLED", "OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_TRACES_EXPORTER"} <= names
    assert all("sensitive" in row and "support" in row for row in registry["switches"])
    assert not any("content" in row["name"].lower() for row in registry["switches"])
    assert enabled.traces is not None
    assert disabled.traces is None


def test_qualification_claims_match_route_signal_codec():
    registry = _registry()
    claims = registry.get("qualifications", [])

    assert claims == []
    assert registry.get("unqualified_routes")
    for route in registry["unqualified_routes"]:
        assert route["qualified"] is False
        assert set(route["signals"]) == {"traces"}
        assert set(route["protocols"]) <= {"http/protobuf", "http/json"}


def test_offline_wheel_manifest_help_and_smoke(tmp_path):
    sys.path.insert(0, str(ROOT / "_build"))
    try:
        backend = importlib.import_module("sf_backend")
        wheel_name = backend.build_wheel(str(tmp_path))
    finally:
        sys.path.remove(str(ROOT / "_build"))

    wheel = tmp_path / wheel_name
    with zipfile.ZipFile(wheel) as package:
        names = set(package.namelist())
        entry_points = next(name for name in names if name.endswith(".dist-info/entry_points.txt"))
        entry_data = package.read(entry_points).decode("utf-8")
        assert "stateframe-otel = stateframe.otel.cli:main" in entry_data
        assert "stateframe/otel/manifest.json" in names
        assert json.loads(package.read("stateframe/otel/manifest.json"))["source_baseline"] == __version__
        assert "stateframe/otel/cli.py" in names
        assert "stateframe/otel/switch-registry.json" in names

    env = dict(os.environ)
    env["PYTHONPATH"] = str(wheel)
    entry = "from stateframe.otel.cli import main; raise SystemExit(main())"
    result = subprocess.run([sys.executable, "-c", entry, "--help"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "usage: stateframe-otel {export,status,purge} ...\n"


def test_otel_registry_host_only_and_docs_coverage():
    manifest = json.loads((ROOT / "src/stateframe/otel/manifest.json").read_text(encoding="utf-8"))
    package = ROOT / "src/stateframe/otel"
    actual = {
        str(path.relative_to(package)).replace(os.sep, "/")
        for path in package.rglob("*")
        if path.is_file() and path.suffix in {".py", ".json"} and path != package / "manifest.json"
    }

    assert manifest["schema"] == "stateframe-otel-manifest/1"
    assert manifest["version"] == 1
    assert manifest["source_baseline"] == "package-version"
    assert set(manifest["files"]) == actual
    assert manifest["files"] == {name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                                 for name in sorted(actual)}
    guide = (ROOT / "docs/guide/observability.md").read_text(encoding="utf-8")
    flags = {"--" + flag.replace("_", "-") for flag in DETAIL_FLAGS}
    environment = {"OTEL_STATEFRAME_" + flag.upper() for flag in DETAIL_FLAGS}
    for path in package.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_bytes())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                flags.update(arg.value for arg in node.args if isinstance(arg, ast.Constant)
                             and isinstance(arg.value, str) and arg.value.startswith("--"))
            if (isinstance(node, ast.Constant) and type(node.value) is str
                    and re.fullmatch(r"OTEL_[A-Z0-9_]*[A-Z0-9]", node.value)):
                environment.add(node.value)
    rows = _registry()["switches"]
    registry_text = " ".join(f"{row['name']} {row['cli'] or ''}" for row in rows)
    flag_pattern = r"(?<![\w-])--[a-z][a-z0-9]*(?:-[a-z0-9]+)*(?![\w-])"
    registered = set(re.findall(flag_pattern, registry_text))
    assert flags <= registered, flags - registered
    documented = set(re.findall(flag_pattern, guide))
    assert flags <= documented, flags - documented
    env_patterns = set()
    for row in rows:
        for field in (row["name"], row["env"] or ""):
            for option in re.findall(r"OTEL_[A-Z0-9_*]+", field):
                env_patterns.add(option)
                env_patterns.add(option.replace("OTLP_TRACES_", "OTLP_"))
    unregistered = {name for name in environment
                    if not any(fnmatchcase(name, pattern) for pattern in env_patterns)}
    assert not unregistered, unregistered


def test_otel_hygiene_rejects_unregistered_environment_switch(tmp_path, monkeypatch):
    package = tmp_path / "src/stateframe/otel"
    shutil.copytree(ROOT / "src/stateframe/otel", package)
    shutil.copytree(ROOT / "docs/guide", tmp_path / "docs/guide")
    config = package / "config.py"
    config.write_bytes(config.read_bytes() + b'\n_REVIEW_FIXTURE = "OTEL_NEW_REVIEW_SWITCH"\n')
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"] = {name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                         for name in manifest["files"]}
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)

    with pytest.raises(AssertionError, match="OTEL_NEW_REVIEW_SWITCH"):
        test_otel_registry_host_only_and_docs_coverage()


def test_source_distribution_contains_observability_guide_assets():
    from _build import sf_backend

    for file in ("docs/guide/observability.md", "docs/assets/observability-dashboard.json",
                 "scripts/otel_demo.sh"):
        assert sf_backend._sdist_includes(Path(file)), file


@pytest.mark.parametrize("verb", ["run", "resume", "chat"])
def test_native_verbs_are_refused(verb, capsys):
    from stateframe.otel import cli

    assert cli.main([verb, "--help"]) == 2
    assert capsys.readouterr().err == "stateframe-otel: unsupported_setting\n"
