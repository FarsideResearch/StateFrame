"""Retry and publication behavior for the offline installer."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "install.sh"


FAKE_PYTHON_BODY = r'''import os
import sys
import time
from pathlib import Path


args = sys.argv[1:]
tool_name = Path(sys.argv[0]).name


def record(label, value):
    trace = os.environ.get("FAKE_INSTALL_TRACE")
    if trace:
        with open(trace, "a", encoding="utf-8") as stream:
            stream.write(f"{label}\t{value}\n")


def wait_for_release(marker_dir, prefix):
    marker_dir = Path(marker_dir)
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / f"{prefix}-{os.getpid()}"
    marker.touch()
    while not (marker_dir / "release").exists():
        time.sleep(0.01)


if tool_name == "stateframe":
    if args == ["--version"]:
        barrier = os.environ.get("FAKE_VERSION_BARRIER")
        if barrier:
            wait_for_release(barrier, "ready")
        print("stateframe 0.8.19")
        raise SystemExit(0)
    if args == ["--help"]:
        print("usage: stateframe [--version|--help]")
        raise SystemExit(0)

if args and args[0] == "-c":
    code = args[1]
    if "sys.version_info[0]" in code:
        print("3.11")
        raise SystemExit(0)
    if "sys.version_info[:3]" in code:
        print("3.11.9")
        raise SystemExit(0)
    if "pathlib, stateframe" in code:
        venv = Path(sys.argv[0]).parent.parent
        print(venv / "lib" / "python3.11" / "site-packages" / "stateframe")
        raise SystemExit(0)
    raise SystemExit(0)

if args and args[0].endswith("build_wheelhouse.py"):
    print("demo-dependency==1.0")
    raise SystemExit(0)

if args[:2] == ["-m", "venv"]:
    if len(args) == 3 and args[2] == "--help":
        raise SystemExit(0)
    venv = Path(args[2])
    record("venv", venv)
    venv.mkdir(parents=True, exist_ok=True)
    (venv / "pyvenv.cfg").write_text("home = synthetic\n", encoding="utf-8")
    binary = venv / "bin"
    binary.mkdir(parents=True, exist_ok=True)
    python = binary / "python"
    if not python.exists():
        try:
            python.symlink_to(Path(__file__).resolve())
        except FileExistsError:
            pass
    raise SystemExit(0)

if args[:2] == ["-m", "pip"]:
    if len(args) == 3 and args[2] == "--version":
        raise SystemExit(0)
    if "install" in args and args[-1] == os.environ.get("FAKE_REPO_ROOT"):
        venv = Path(sys.argv[0]).parent.parent
        package = venv / "lib" / "python3.11" / "site-packages" / "stateframe"
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("__version__ = '0.8.19'\n", encoding="utf-8")
        cli = venv / "bin" / "stateframe"
        if not cli.exists():
            try:
                cli.symlink_to(Path(__file__).resolve())
            except FileExistsError:
                pass
    raise SystemExit(0)

if args and args[0] == "-":
    script = sys.stdin.read()
    if len(args) == 2:
        raise SystemExit(0)
    if len(args) == 3 and "os.rename" in script:
        if os.environ.get("FAKE_FAIL_RENAME") == "1":
            print("synthetic rename failure", file=sys.stderr)
            raise SystemExit(1)
        try:
            os.rename(args[1], args[2])
        except OSError as error:
            print(f"could not rename incomplete environment: {error}", file=sys.stderr)
            raise SystemExit(1)
        raise SystemExit(0)
    if len(args) == 3 and "os.symlink" in script:
        try:
            os.symlink(args[1], args[2])
        except FileExistsError:
            raise SystemExit(3)
        except OSError as error:
            print(f"could not create install link: {error}", file=sys.stderr)
            raise SystemExit(1)
        raise SystemExit(0)
    if os.environ.get("FAKE_FAIL_OPTIONAL") == "1":
        print("synthetic optional validation failure", file=sys.stderr)
        raise SystemExit(1)
    pause_at = os.environ.get("FAKE_PAUSE_OPTIONAL")
    if pause_at:
        marker = Path(pause_at)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        release = Path(os.environ["FAKE_RELEASE_OPTIONAL"])
        while not release.exists():
            time.sleep(0.01)
    raise SystemExit(0)

raise SystemExit(0)
'''


FAKE_UNAME = """#!/bin/sh
case "$1" in
  -s) printf 'Linux\\n' ;;
  -m) printf 'x86_64\\n' ;;
  *) exit 1 ;;
esac
"""


FAKE_MKDIR = r'''#!/bin/sh
match=0
for argument in "$@"; do
  if [ "$argument" = "$SF_HOME" ]; then
    match=1
  fi
done
if [ "$match" = 1 ] && [ "${FAKE_SYNC_HOME_MKDIR:-}" = 1 ]; then
  /bin/mkdir "$@" || exit $?
  marker="$FAKE_SYNC_DIR/mkdir-$$"
  : > "$marker"
  while :; do
    set -- "$FAKE_SYNC_DIR"/mkdir-*
    if [ "$#" -ge 2 ]; then
      break
    fi
    /bin/sleep 0.01
  done
  exit 0
fi
exec /bin/mkdir "$@"
'''


def _make_doubles(base: Path, *, synchronize_publishers: bool = False) -> tuple[Path, Path]:
    bin_dir = base / "fake-bin"
    bin_dir.mkdir(parents=True)
    fake_python = bin_dir / "python-fake"
    fake_python.write_text(f"#!{sys.executable}\n" + FAKE_PYTHON_BODY, encoding="utf-8")
    fake_python.chmod(0o755)

    uname = bin_dir / "uname"
    uname.write_text(FAKE_UNAME, encoding="utf-8")
    uname.chmod(0o755)
    bwrap = bin_dir / "bwrap"
    bwrap.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    bwrap.chmod(0o755)

    if synchronize_publishers:
        mkdir = bin_dir / "mkdir"
        mkdir.write_text(FAKE_MKDIR, encoding="utf-8")
        mkdir.chmod(0o755)
    return bin_dir, fake_python


def _base_environment(
    home: Path,
    sf_home: Path,
    bin_dir: Path,
    fake_python: Path,
    trace: Path,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "SF_HOME": str(sf_home),
            "PYTHON": str(fake_python),
            "PATH": str(bin_dir) + os.pathsep + env.get("PATH", ""),
            "FAKE_INSTALL_TRACE": str(trace),
            "FAKE_REPO_ROOT": str(ROOT),
        }
    )
    return env


def _run_install(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(INSTALLER), "--offline"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _environment_dirs(sf_home: Path) -> list[Path]:
    environments: set[Path] = set()
    for current, directories, files in os.walk(sf_home, followlinks=False):
        if "stateframe" in files and Path(current).name == "bin":
            environments.add(Path(current).parent)
    return sorted(environments)


def _tree_snapshot(root: Path) -> tuple[tuple[str, str, int, bytes | str | None], ...]:
    snapshot: list[tuple[str, str, int, bytes | str | None]] = []
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in sorted(directories + files):
            path = current_path / name
            relative = str(path.relative_to(root))
            mode = stat.S_IMODE(path.lstat().st_mode)
            if path.is_symlink():
                snapshot.append((relative, "link", mode, os.readlink(path)))
            elif path.is_dir():
                snapshot.append((relative, "dir", mode, None))
            else:
                snapshot.append((relative, "file", mode, path.read_bytes()))
    return tuple(sorted(snapshot))


def _points_into(public_path: Path, environment: Path) -> bool:
    if not (public_path.exists() or public_path.is_symlink()):
        return False
    resolved = public_path.resolve()
    target = environment.resolve()
    return resolved == target or target in resolved.parents


def _user_shim(home: Path) -> tuple[Path, bytes]:
    shim = home / ".local" / "bin" / "stateframe"
    shim.parent.mkdir(parents=True)
    content = b"#!/bin/sh\nprintf 'user command\\n'\n"
    shim.write_bytes(content)
    shim.chmod(0o755)
    return shim, content


def test_failed_offline_install_can_be_retried_without_replacing_paths(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    sf_home = tmp_path / "private-install"
    trace = tmp_path / "install-trace.txt"
    shim, shim_bytes = _user_shim(home)
    bin_dir, fake_python = _make_doubles(tmp_path)
    env = _base_environment(home, sf_home, bin_dir, fake_python, trace)

    failed_env = dict(env, FAKE_FAIL_OPTIONAL="1")
    first = _run_install(failed_env)
    failed_environments = _environment_dirs(sf_home)
    assert failed_environments, first.stdout + first.stderr
    failed_tree = failed_environments[0]
    failed_snapshot = _tree_snapshot(failed_tree)
    failed_was_published = _points_into(sf_home / "venv", failed_tree)
    failed_was_linked = _points_into(shim, failed_tree)

    retry = _run_install(env)
    final_snapshot = _tree_snapshot(failed_tree) if failed_tree.exists() else ()
    retry_environments = _environment_dirs(sf_home)
    stable_cli = sf_home / "venv" / "bin" / "stateframe"
    shim_after = shim.read_bytes() if shim.exists() else b""
    failed_still_published = _points_into(sf_home / "venv", failed_tree)
    failed_still_linked = _points_into(shim, failed_tree)

    assert first.returncode != 0, first.stdout + first.stderr
    assert "synthetic optional validation failure" in first.stderr
    assert retry.returncode == 0, retry.stdout + retry.stderr
    assert failed_snapshot == final_snapshot, "retry changed the earlier attempt tree"
    assert shim_after == shim_bytes, "installer changed the existing user command"
    assert not failed_was_published, "failed attempt was exposed at the stable environment path"
    assert not failed_was_linked, "user command resolves to the failed attempt"
    assert not failed_still_published, "retry left the failed attempt exposed at the stable path"
    assert not failed_still_linked, "user command resolves to the failed attempt after retry"
    assert stable_cli.is_file(), "successful retry did not publish the stable command"
    assert len(retry_environments) >= 2, "retry did not retain the earlier attempt"


def test_fresh_install_publishes_a_working_stable_cli(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    sf_home = tmp_path / "private-install"
    trace = tmp_path / "install-trace.txt"
    bin_dir, fake_python = _make_doubles(tmp_path)
    env = _base_environment(home, sf_home, bin_dir, fake_python, trace)

    result = _run_install(env)
    stable_cli = sf_home / "venv" / "bin" / "stateframe"
    version = subprocess.run(
        [str(stable_cli), "--version"], cwd=ROOT, env=env, capture_output=True, text=True, check=False
    ) if stable_cli.exists() else None
    help_result = subprocess.run(
        [str(stable_cli), "--help"], cwd=ROOT, env=env, capture_output=True, text=True, check=False
    ) if stable_cli.exists() else None

    assert result.returncode == 0, result.stdout + result.stderr
    assert stable_cli.is_file(), "successful install did not expose the stable CLI"
    assert version is not None and version.returncode == 0, "stable CLI version check failed"
    assert "stateframe 0.8.19" in version.stdout
    assert help_result is not None and help_result.returncode == 0, "stable CLI help failed"
    assert "usage: stateframe" in help_result.stdout


def test_simultaneous_publishers_build_separate_attempts(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    sf_home = tmp_path / "private-install"
    trace = tmp_path / "install-trace.txt"
    shim, shim_bytes = _user_shim(home)
    bin_dir, fake_python = _make_doubles(tmp_path, synchronize_publishers=True)
    barrier = tmp_path / "publish-barrier"
    barrier.mkdir()
    env = _base_environment(home, sf_home, bin_dir, fake_python, trace)
    env.update(
        {
            "FAKE_SYNC_HOME_MKDIR": "1",
            "FAKE_SYNC_DIR": str(barrier),
            "FAKE_VERSION_BARRIER": str(barrier),
        }
    )

    bash = shutil.which("bash")
    assert bash is not None
    publishers = [
        subprocess.Popen(
            [bash, str(INSTALLER), "--offline"],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    try:
        while len(list(barrier.glob("ready-*"))) < 2:
            if any(process.poll() is not None for process in publishers):
                break
            time.sleep(0.01)
    finally:
        (barrier / "release").touch()
    outputs = [process.communicate() for process in publishers]
    return_codes = [process.returncode for process in publishers]

    targets = [
        Path(line.split("\t", 1)[1])
        for line in trace.read_text(encoding="utf-8").splitlines()
        if line.startswith("venv\t")
    ]
    stable_cli = sf_home / "venv" / "bin" / "stateframe"

    assert sum(code == 0 for code in return_codes) >= 1, "no publisher completed: " + repr(outputs)
    assert len(targets) == 2, "both publishers should reach environment creation: " + repr(outputs)
    assert len(set(targets)) == 2, "publishers shared one environment tree"
    assert stable_cli.is_file(), "no complete command was published: " + repr(outputs)
    assert shim.read_bytes() == shim_bytes, "publisher changed the existing user command"


def test_directory_without_venv_marker_is_preserved(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    sf_home = tmp_path / "private-install"
    venv = sf_home / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_bytes(b"partial environment marker\n")
    before = _tree_snapshot(venv)

    bin_dir, fake_python = _make_doubles(tmp_path)
    trace = tmp_path / "install-trace.txt"
    env = _base_environment(home, sf_home, bin_dir, fake_python, trace)
    result = _run_install(env)
    after = _tree_snapshot(venv) if venv.exists() else ()

    assert result.returncode != 0, "installer accepted an unprovenanced partial environment"
    assert after == before, "installer removed or changed the partial environment"
    assert str(venv) in result.stderr, result.stdout + result.stderr
    assert "SF_HOME" in result.stderr, result.stdout + result.stderr


def test_legacy_incomplete_environment_is_moved_byte_for_byte_and_retry_succeeds(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    sf_home = tmp_path / "private-install"
    legacy = sf_home / "venv"
    (legacy / "bin").mkdir(parents=True)
    (legacy / "pyvenv.cfg").write_bytes(b"home = previous interpreter\ninclude-system-site-packages = false\n")
    (legacy / "lib" / "python3.11" / "site-packages").mkdir(parents=True)
    (legacy / "lib" / "python3.11" / "site-packages" / "marker.bin").write_bytes(b"\x00old\xfftree\n")
    original_tree = _tree_snapshot(legacy)
    original_config_inode = (legacy / "pyvenv.cfg").stat().st_ino

    shim, shim_bytes = _user_shim(home)
    bin_dir, fake_python = _make_doubles(tmp_path)
    trace = tmp_path / "install-trace.txt"
    env = _base_environment(home, sf_home, bin_dir, fake_python, trace)

    result = _run_install(env)
    moved = sorted(sf_home.glob(".venv-incomplete.*/venv"))
    stable_cli = sf_home / "venv" / "bin" / "stateframe"
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert len(moved) == 1, f"expected one preserved legacy environment, found {moved}"
    assert _tree_snapshot(moved[0]) == original_tree, "moved legacy environment changed"
    assert (moved[0] / "pyvenv.cfg").stat().st_ino == original_config_inode, "legacy tree was copied instead of renamed"
    assert f"preserved incomplete environment at {moved[0]}" in output
    assert output.count(str(moved[0])) == 1, "preservation path should be reported once"
    assert stable_cli.is_file(), "successful retry did not publish the stable command"
    assert shim.read_bytes() == shim_bytes, "retry changed the existing user command"


@pytest.mark.parametrize(
    "target_kind",
    ["complete_environment", "symlink", "dangling_symlink", "file", "directory_without_cfg"],
)
def test_other_existing_targets_are_refused_and_untouched(tmp_path: Path, target_kind: str) -> None:
    home = tmp_path / "home"
    home.mkdir()
    sf_home = tmp_path / "private-install"
    sf_home.mkdir()
    venv = sf_home / "venv"

    if target_kind == "complete_environment":
        (venv / "bin").mkdir(parents=True)
        (venv / "pyvenv.cfg").write_bytes(b"existing complete environment\n")
        (venv / "bin" / "stateframe").write_bytes(b"existing command\n")
    elif target_kind == "symlink":
        target = sf_home / "kept-target"
        target.mkdir()
        (target / "marker").write_bytes(b"symlink target\n")
        venv.symlink_to(target, target_is_directory=True)
    elif target_kind == "dangling_symlink":
        venv.symlink_to(sf_home / "missing-target", target_is_directory=True)
    elif target_kind == "file":
        venv.write_bytes(b"existing regular file\n")
    else:
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_bytes(b"directory without venv marker\n")

    before = _tree_snapshot(sf_home)
    bin_dir, fake_python = _make_doubles(tmp_path)
    trace = tmp_path / "install-trace.txt"
    env = _base_environment(home, sf_home, bin_dir, fake_python, trace)

    result = _run_install(env)

    assert result.returncode != 0, f"installer accepted {target_kind}: {result.stdout}{result.stderr}"
    assert result.stderr == (
        f"stateframe install: install target already exists: {venv} "
        "(choose a distinct SF_HOME to preserve the existing environment and incomplete material)\n"
    )
    assert _tree_snapshot(sf_home) == before, f"installer changed {target_kind} target"


def test_failed_legacy_rename_leaves_original_environment_untouched(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    sf_home = tmp_path / "private-install"
    legacy = sf_home / "venv"
    (legacy / "bin").mkdir(parents=True)
    (legacy / "pyvenv.cfg").write_bytes(b"home = previous interpreter\n")
    (legacy / "marker").write_bytes(b"keep me\n")
    before = _tree_snapshot(legacy)

    bin_dir, fake_python = _make_doubles(tmp_path)
    trace = tmp_path / "install-trace.txt"
    env = _base_environment(home, sf_home, bin_dir, fake_python, trace)
    env["FAKE_FAIL_RENAME"] = "1"

    result = _run_install(env)

    assert result.returncode != 0, "installer continued after the legacy rename failed"
    assert "could not preserve incomplete install target" in result.stderr
    assert str(legacy) in result.stderr
    assert _tree_snapshot(legacy) == before, "failed rename changed the original environment"
    assert not list(sf_home.glob(".venv-attempt.*")), "installer continued after the failed rename"
