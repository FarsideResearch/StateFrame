"""Shared run environment seeds must remain immutable across runs."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from _timeouts import scaled
from stateframe.engine import kernel_env_seed
from stateframe.engine import kernel_run_env
from stateframe.engine.kernel_run_env import create_run_env


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    site = source / "lib/python3.11/site-packages/demo-1.0.dist-info"
    site.mkdir(parents=True)
    (site / "METADATA").write_text("Name: demo\nVersion: 1.0\n")
    (site / "RECORD").write_text("demo.py,,\n")
    package = site.parent / "demo.py"
    package.write_text("value = 1\n")
    (source / "bin").mkdir()
    (source / "bin/python").symlink_to(sys.executable)
    (source / "bin/tool").write_text(f"#!{source}/bin/python\n")
    (source / "pyvenv.cfg").write_text(f"home = {source}\n")
    return source


def test_seed_reuse_and_private_relocation(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = _source(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    one = kernel_env_seed.clone(source, first, source / "bin/python", None, ())
    two = kernel_env_seed.clone(source, second, source / "bin/python", None, ())
    assert one == two
    assert len(list((tmp_path / "home/env-seeds").glob("[0-9a-f]" * 64))) == 1
    assert (first / "bin/tool").stat().st_ino != (one / "bin/tool").stat().st_ino
    assert (first / "lib/python3.11/site-packages/demo.py").read_text() == "value = 1\n"
    assert json.loads((one / "seed.json").read_text())["files"]


def test_seed_quarantines_tampered_file(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = _source(tmp_path)
    seed = kernel_env_seed.clone(source, tmp_path / "first", source / "bin/python", None, ())
    target = seed / "lib/python3.11/site-packages/demo.py"
    target.chmod(0o600)
    target.write_text("corrupt\n")
    events = []
    rebuilt = kernel_env_seed.clone(source, tmp_path / "second", source / "bin/python", None, (),
                                    event=events.append)
    assert rebuilt == seed
    assert target.read_text() == "value = 1\n"
    assert list(seed.parent.glob("quarantine-*"))
    assert events and events[0]["event"] == "env_seed_quarantined"


def test_run_env_quarantine_uses_engine_event_stream(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(kernel_run_env, "_run", lambda *_args, **_kwargs: "pip fixture")
    source = _source(tmp_path)
    first = tmp_path / "first"
    create_run_env(first, source / "bin/python")
    seed = Path(json.loads((first / "env-seed.json").read_text())["path"])
    changed = seed / "lib/python3.11/site-packages/demo.py"
    changed.chmod(0o600)
    changed.write_text("corrupt\n")

    second = tmp_path / "second"
    create_run_env(second, source / "bin/python")
    metadata = json.loads((second / "env-seed.json").read_text())
    assert metadata["events"][0]["event"] == "env_seed_quarantined"
    assert not (second / "checkpoints" / "events.jsonl").exists()


def test_child_emits_run_env_seed_events(tmp_path, capsys):
    from stateframe import _child

    (tmp_path / "env-seed.json").write_text(json.dumps({
        "path": "/seed", "events": [{"event": "env_seed_quarantined", "key": "digest"}]}))
    _child.emit_run_env_events(tmp_path)
    line = capsys.readouterr().out.strip()
    assert line.startswith("[sf] ")
    assert json.loads(line[5:]) == {"kind": "env_seed_quarantined", "key": "digest"}


def test_source_env_edit_changes_seed_key(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = _source(tmp_path)
    first = kernel_env_seed.clone(source, tmp_path / "first", source / "bin/python", None, ())
    (source / "lib/python3.11/site-packages/demo.py").write_text("value = 2\n")
    second = kernel_env_seed.clone(source, tmp_path / "second", source / "bin/python", None, ())
    assert first != second
    assert (tmp_path / "second/lib/python3.11/site-packages/demo.py").read_text() == "value = 2\n"


def test_copy_switch_uses_legacy_copy_path(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_ENV_SEED", "copy")
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(kernel_run_env, "_run", lambda *_args, **_kwargs: "pip fixture")
    source = _source(tmp_path)
    run = create_run_env(tmp_path / "run", source / "bin/python")
    assert run.kind == "clone"
    assert (run.bin_dir / "tool").read_text() == f"#!{run.root}/bin/python\n"
    assert not (tmp_path / "run/env-seed.json").exists()
    assert not (tmp_path / "home/env-seeds").exists()


def test_concurrent_seed_builders_publish_one_seed(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = _source(tmp_path)
    code = ("import sys; from pathlib import Path; "
            "from stateframe.engine.kernel_env_seed import clone; "
            "source = Path(sys.argv[1]); "
            "clone(source, Path(sys.argv[2]), source / 'bin/python', None, ())")
    processes = [subprocess.Popen([sys.executable, "-c", code, str(source),
                                   str(tmp_path / f"run-{index}")],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                 for index in range(2)]
    for process in processes:
        stdout, stderr = process.communicate(timeout=scaled(60))
        assert process.returncode == 0, stdout + stderr
    seeds = list((tmp_path / "home/env-seeds").glob("[0-9a-f]" * 64))
    assert len(seeds) == 1
    assert not list(seeds[0].parent.glob(".*.tmp-*"))


def test_seed_preserves_empty_directories_and_directory_symlinks(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = _source(tmp_path)
    (source / "lib/empty").mkdir()
    (source / "lib/alias").symlink_to("empty", target_is_directory=True)
    run = tmp_path / "run"
    seed = kernel_env_seed.clone(source, run, source / "bin/python", None, ())
    assert (run / "lib/empty").is_dir()
    assert (run / "lib/alias").is_symlink()
    assert os.readlink(run / "lib/alias") == "empty"
    assert kernel_env_seed.clone(source, tmp_path / "again", source / "bin/python", None, ()) == seed
    assert not list(seed.parent.glob("quarantine-*"))


@pytest.mark.skipif(sys.platform != "linux", reason="hardlink fallback is Linux specific")
def test_fallback_hardlink_is_read_only_and_run_replacement_is_private(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(kernel_env_seed, "_reflink", lambda *_args: False)
    source = _source(tmp_path)
    run = tmp_path / "run"
    seed = kernel_env_seed.clone(source, run, source / "bin/python", None, ())
    relative = "lib/python3.11/site-packages/demo.py"
    shared = run / relative
    assert shared.stat().st_ino == (seed / relative).stat().st_ino
    assert not shared.stat().st_mode & 0o222
    if os.geteuid() != 0:
        result = subprocess.run([sys.executable, "-c", "import pathlib,sys; pathlib.Path(sys.argv[1]).open('a').write('bad')", str(shared)],
                                capture_output=True, text=True)
        assert result.returncode != 0 and "PermissionError" in result.stderr
    subprocess.run(["sed", "-i", "s/value = 1/value = 2/", str(shared)], check=True)
    assert shared.read_text() == "value = 2\n"
    assert (seed / relative).read_text() == "value = 1\n"
    replacement = shared.with_name("replacement")
    replacement.write_text("changed\n")
    replacement.replace(shared)
    assert shared.read_text() == "changed\n"
    assert (seed / relative).read_text() == "value = 1\n"


@pytest.mark.host_sandbox
def test_run_env_relocates_script_shebang_from_seed(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = tmp_path / "source"
    subprocess.run([sys.executable, "-m", "venv", str(source)], check=True)
    run = create_run_env(tmp_path / "run", source / "bin/python")
    script = run.bin_dir / "pip"
    head = script.read_bytes().splitlines()[:2]
    # pip writes a /bin/sh exec launcher instead of a direct shebang when the path is too long.
    launcher = head[1] if head[0] == b"#!/bin/sh" else head[0][2:]
    assert os.fsencode(str(run.bin_dir)) + b"/python" in launcher
    assert os.fsencode(str(source)) not in b"\n".join(head)
    assert (tmp_path / "run/env-seed.json").is_file()


def test_clone_across_filesystems_copies_instead_of_linking(tmp_path, monkeypatch):
    """A seed store on another filesystem (EXDEV) still yields a private, complete run env."""
    import errno

    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(kernel_env_seed, "_reflink", lambda source, target: False)

    def cross_device(source, target, *args, **kwargs):
        raise OSError(errno.EXDEV, "Invalid cross-device link", str(source), None, str(target))

    monkeypatch.setattr(kernel_env_seed.os, "link", cross_device)
    source = _source(tmp_path)
    run = tmp_path / "run"
    seed = kernel_env_seed.clone(source, run, source / "bin/python", None, ())
    module = run / "lib/python3.11/site-packages/demo.py"
    assert module.read_text() == "value = 1\n"
    assert module.stat().st_ino != (seed / "lib/python3.11/site-packages/demo.py").stat().st_ino


def test_source_digest_is_framed_and_sees_content_mode_links_and_names(tmp_path):
    root = tmp_path / "src"
    (root / "lib").mkdir(parents=True)
    (root / "lib/a.py").write_text("a")
    (root / "bin").mkdir()
    # Link targets are digest data only (never executed); kept off /usr/bin so the portable-executables guard stays quiet.
    (root / "bin/python").symlink_to("/opt/seed-target/python3")
    seen = {kernel_env_seed._source_digest(root)}
    for change in (lambda: (root / "lib/a.py").write_text("b"),
                   lambda: (root / "lib/a.py").chmod(0o600),
                   lambda: ((root / "bin/python").unlink(), (root / "bin/python").symlink_to("/opt/seed-target/python")),
                   lambda: (root / "lib/a.py").rename(root / "lib/b.py"),
                   # Shifting bytes between a name and its content must not collide (framed entries).
                   lambda: (root / "lib/ab").write_text("")):
        change()
        seen.add(kernel_env_seed._source_digest(root))
    assert len(seen) == 6
