"""The Linux check sandbox must let a shared-library Python find libpython.

A Python built with --enable-shared under /usr/local (the official python images, pyenv
--enable-shared) loads /usr/local/lib/libpython*.so through the loader cache. Without the
cache every in-run check exited 127 before pytest started.
"""
import kernel_test_isolation as isolation


def _linux(monkeypatch, *, cache_present: bool):
    monkeypatch.setattr(isolation, "backend", lambda: "/usr/bin/bwrap")
    monkeypatch.setattr(isolation.platform, "system", lambda: "Linux")
    real_isfile = isolation.os.path.isfile
    monkeypatch.setattr(isolation.os.path, "isfile",
                        lambda path: cache_present if path == "/etc/ld.so.cache" else real_isfile(path))


def _binds(command):
    return [command[i:i + 3] for i, part in enumerate(command) if part in ("--ro-bind", "--bind")]


def test_check_sandbox_binds_only_the_loader_cache_from_etc(monkeypatch, tmp_path):
    _linux(monkeypatch, cache_present=True)
    command, _ = isolation.isolated_command(["/bin/echo", "ok"], tmp_path, {})
    etc = [bind for bind in _binds(command[:command.index("--")]) if bind[1].startswith("/etc")]
    assert etc == [["--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache"]]


def test_check_sandbox_skips_a_missing_loader_cache(monkeypatch, tmp_path):
    _linux(monkeypatch, cache_present=False)
    command, _ = isolation.isolated_command(["/bin/echo", "ok"], tmp_path, {})
    assert "/etc/ld.so.cache" not in command[:command.index("--")]
