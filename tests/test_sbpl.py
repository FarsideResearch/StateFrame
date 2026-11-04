"""SBPL quoting guards, including real Seatbelt probes when available."""
from __future__ import annotations

import subprocess
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from kernel_sbpl import sbpl_path_regex, sbpl_regex_literal, sbpl_string
from kernel_test_isolation import _darwin_xcrun_filter
from protected_tree_guard import guarded_command


@pytest.mark.parametrize(("path", "quoted"), [
    ("/tmp/plain", '"/tmp/plain"'),
    ("/tmp/has space", '"/tmp/has space"'),
    ('/tmp/has"quote', '"/tmp/has\\"quote"'),
    ("/tmp/has\\backslash", '"/tmp/has\\\\backslash"'),
    ("/tmp/café", '"/tmp/café"'),
    ("/tmp/emoji😀", '"/tmp/emoji😀"'),
])
def test_sbpl_string_quoting(path, quoted):
    assert sbpl_string(path) == quoted


@pytest.mark.parametrize("bad", ["\0", "\n", "\r", "\t", "\x7f", "\ud800"])
def test_sbpl_string_refuses_unsafe_characters(bad):
    with pytest.raises(ValueError):
        sbpl_string("/tmp/" + bad)
    with pytest.raises(ValueError):
        sbpl_regex_literal("^/tmp/" + bad + "$")


def test_sbpl_regex_uses_raw_backslashes_and_ere_escaping():
    assert sbpl_regex_literal(r"^/tmp/dot\.dir$") == r'#"^/tmp/dot\.dir$"'
    assert sbpl_path_regex("/tmp/dot.dir", "/xcrun_db-[^/]+$") == (
        r'#"^/tmp/dot\.dir/xcrun_db-[^/]+$"')
    assert sbpl_path_regex("/tmp/a.[]()*+?{}|^$\\b", "$") == (
        r'#"^/tmp/a\.\[\]\(\)\*\+\?\{\}\|\^\$\\b$"')
    assert sbpl_path_regex("/tmp/plain_dir", "/xcrun_db-[^/]+$") == (
        '#"^/tmp/plain_dir/xcrun_db-[^/]+$"')


def test_sbpl_raw_regex_refuses_quote():
    with pytest.raises(ValueError, match="double quote"):
        sbpl_regex_literal('^/tmp/has"quote$')


@pytest.fixture
def seatbelt():
    if sys.platform != "darwin":
        pytest.skip("live SBPL probes require macOS")
    executable = shutil.which("sandbox-exec")
    if executable is None:
        pytest.fail("macOS sandbox-exec is unavailable")
    true_executable = shutil.which("true")
    assert true_executable is not None, "true executable is unavailable"
    probe = subprocess.run([executable, "-p", "(version 1) (allow default)",
                            true_executable], capture_output=True, text=True, timeout=10)
    if probe.returncode != 0:
        pytest.fail(f"sandbox-exec capability probe failed (exit {probe.returncode}): "
                    f"{probe.stderr.strip()}")
    return executable


def _touch(command: list[str], path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([*command, "/usr/bin/touch", str(path)],
                          capture_output=True, text=True, timeout=10)


@pytest.mark.host_sandbox
@pytest.mark.parametrize(("name", "sibling"), [
    ("dot.dir", "dotXdir"),
    ("dash-dir", "dashXdir"),
    ("plus+dir", "plusXdir"),
    ("plain_dir", "plainXdir"),
])
def test_xcrun_regex_matches_only_one_cache_file(seatbelt, monkeypatch, name, sibling):
    import kernel_test_isolation as isolation

    with tempfile.TemporaryDirectory(prefix="sf-sbpl-", dir="/private/tmp") as temporary:
        base = Path(temporary)
        directory = base / name
        directory.mkdir()
        sibling_dir = base / sibling
        sibling_dir.mkdir()
        (directory / "xcrun_db-x").mkdir()
        monkeypatch.setattr(isolation, "_darwin_xcrun_cache_paths", lambda: [directory])
        filter_rule = _darwin_xcrun_filter()
        profile = ("(version 1) (allow default) "
                   f"(deny file-write* (require-not (require-any {filter_rule} "
                   '(literal "/dev/null"))))')
        command = [seatbelt, "-p", profile]
        allowed = directory / "xcrun_db-abc"
        assert _touch(command, allowed).returncode == 0
        assert allowed.is_file()
        for denied in (directory / "xcrun_db-", directory / "xcrun_db-x/child",
                       directory / "prefix-xcrun_db-a", directory / "unrelated",
                       sibling_dir / "xcrun_db-a"):
            result = _touch(command, denied)
            assert result.returncode != 0, (denied, result.stderr)
            assert not denied.exists()


@pytest.mark.host_sandbox
@pytest.mark.parametrize("name", ["has space", 'has"quote', "has\\backslash", "café", "emoji😀"])
def test_sbpl_string_allows_and_protects_exact_path(seatbelt, monkeypatch, name):
    import protected_tree_guard as guard

    with tempfile.TemporaryDirectory(prefix="sf-sbpl-", dir="/private/tmp") as temporary:
        base = Path(temporary)
        directory = base / name
        directory.mkdir()
        sibling = base / "sibling"
        sibling.mkdir()
        child = directory / "child"
        outside = sibling / "child"
        allow_profile = ("(version 1) (allow default) "
                         f"(deny file-write* (require-not (subpath {sbpl_string(str(directory))})))")
        assert _touch([seatbelt, "-p", allow_profile], child).returncode == 0
        assert _touch([seatbelt, "-p", allow_profile], outside).returncode != 0
        child.unlink()
        monkeypatch.setattr(guard.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(guard.shutil, "which", lambda name: seatbelt)
        deny_command = guarded_command([], roots=(directory,))
        assert _touch(deny_command, child).returncode != 0
        assert not child.exists()
        assert _touch(deny_command, outside).returncode == 0
