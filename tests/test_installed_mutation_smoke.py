"""The installed release hook checks the host-specific mutation contract."""
from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import sys
import textwrap

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "installed_smoke.sh"


def _stub_venv(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").symlink_to(Path(sys.executable))
    stub = tmp_path / "stateframe_stub.py"
    stub.write_text(textwrap.dedent("""
        import json
        import os
        import platform
        import sys

        mode = os.environ["STUB_MODE"]
        args = sys.argv[1:]
        if os.environ.get("PYTHONPATH"):
            print("PYTHONPATH was not cleared", file=sys.stderr)
            raise SystemExit(90)
        if not os.environ.get("STATEFRAME_HOME") or not os.path.isdir(os.environ["STATEFRAME_HOME"]):
            print("STATEFRAME_HOME is not a fresh directory", file=sys.stderr)
            raise SystemExit(91)
        if args == ["mutations", "list"]:
            with open(os.environ["STUB_LIST_MARKER"], "w", encoding="utf-8") as marker:
                marker.write("called")
            print("[]")
            raise SystemExit(0)
        if args != ["mutations", "selftest", "--json"]:
            print("unexpected arguments: " + repr(args), file=sys.stderr)
            raise SystemExit(92)
        is_linux = platform.system() == "Linux"
        if mode == "traceback":
            if is_linux:
                print("Traceback (most recent call last):\\n  RuntimeError: fixture")
            else:
                print("MUTATION_PLATFORM_UNSUPPORTED\\nTraceback (most recent call last):")
            raise SystemExit(4)
        if is_linux:
            if mode == "ok":
                print(json.dumps({"ok": True, "checks": [{"name": "positive", "ok": True}]}))
            else:
                print(json.dumps({"ok": False, "checks": [
                    {"name": "file_facts", "ok": False, "detail": "fixture missing"}]}))
            raise SystemExit(0)
        if mode == "ok":
            print("MUTATION_PLATFORM_UNSUPPORTED: mutations are Linux-only")
            raise SystemExit(4)
        print(json.dumps({"ok": False, "checks": [{"name": "platform", "ok": False,
                                                     "detail": "unsupported"}]}))
        raise SystemExit(0)
    """), encoding="utf-8")
    binary = bin_dir / "stateframe"
    binary.write_text(
        "#!/bin/sh\nexec " + shlex.quote(sys.executable) + " " + shlex.quote(str(stub)) + ' "$@"\n',
        encoding="utf-8",
    )
    binary.chmod(0o755)
    return binary


@pytest.mark.parametrize("mode", ["ok", "false", "traceback"], ids=["ok", "false-json", "traceback"])
def test_installed_smoke_stub_cli_current_os_branch(tmp_path, mode):
    stateframe = _stub_venv(tmp_path)
    list_marker = tmp_path / "mutations-list-called"
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=tmp_path,
        env={**os.environ, "STATEFRAME_BIN": str(stateframe), "STUB_MODE": mode,
             "PYTHONPATH": str(tmp_path / "must-be-unset"),
             "STUB_LIST_MARKER": str(list_marker)},
        capture_output=True,
        text=True,
        timeout=10,
    )
    output = result.stdout + result.stderr
    if mode == "ok":
        assert result.returncode == 0, output
        assert "PASS: installed mutation smoke" in output
        if not sys.platform.startswith("linux"):
            assert list_marker.read_text(encoding="utf-8") == "called"
    else:
        assert result.returncode != 0
        assert "FAIL: installed mutation smoke:" in output
        if mode == "false" and sys.platform.startswith("linux"):
            assert "file_facts: fixture missing" in output
