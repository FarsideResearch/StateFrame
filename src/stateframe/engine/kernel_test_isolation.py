"""Shared OS boundary for all restricted v46 execution.

Path checks on read/grep are insufficient: model-created code can open host
files. Restricted policies therefore execute children with only the selected
workspace and installed interpreter/libraries readable, with no network.

This builder is intentionally shared by the central execution policy and the
typed observation probe. Callers must still use the policy's capability check
and resource-limited launcher; this function never supplies an unsandboxed
fallback.
"""
from __future__ import annotations

import json
from functools import lru_cache
import os
from pathlib import Path
import platform
import re
import shutil
import site
import subprocess
import sys

if __package__:
    from .kernel_test_python import test_python
else:
    from kernel_test_python import test_python


# Host task policy is read LAZILY at sandbox-build time, never frozen at import: a module imported
# before the host policy is visible (or across a config purge) must still honour it (10-06, batch 837).
def host_task_network() -> str:
    return os.environ.get("STATEFRAME_TASK_NETWORK", "")


def host_task_pytest_plugins() -> str:
    return os.environ.get("STATEFRAME_TASK_PYTEST_PLUGINS", "")


def _declared_relay_dir(env: dict[str, str]) -> Path | None:
    raw = env.get("STATEFRAME_TASK_RELAY_SOCKET_DIR")
    if not raw:
        return None
    candidate = Path(raw)
    if (
        candidate.parent != Path("/tmp")
        or candidate.is_symlink()
        or not candidate.is_dir()
        or not candidate.name.startswith("stage6-pg-")
    ):
        raise RuntimeError("invalid Stage 6 relay directory")
    return candidate


# Standard install locations, used when PATH does not list the sandbox binary (an IDE, cron or a
# minimal-PATH launcher must still get the sandbox, never a silent "unavailable").
_SANDBOX_DIRS = ("/usr/bin", "/bin", "/usr/local/bin", "/usr/sbin")


def sandbox_executable(name: str) -> str | None:
    """Resolve sandbox-exec/bwrap from PATH, then from the standard system locations."""
    found = shutil.which(name)
    if found:
        return found
    for directory in _SANDBOX_DIRS:
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def backend() -> str:
    name = "sandbox-exec" if platform.system() == "Darwin" else "bwrap"
    executable = sandbox_executable(name)
    if executable is None:
        raise RuntimeError(f"restricted information policy requires {name}; no unsandboxed fallback")
    return executable


@lru_cache(maxsize=16)
def _bwrap_userns_guard_supported(executable: str) -> bool:
    """Probe one installed binary, not each command, for the paired guard flags."""
    try:
        result = subprocess.run([executable, "--help"], capture_output=True, text=True,
                                timeout=5, check=False,
                                env={"PATH": os.defpath, "LANG": "C", "LC_ALL": "C"})
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("cannot inspect bubblewrap namespace protections") from error
    if result.returncode != 0:
        raise RuntimeError("cannot inspect bubblewrap namespace protections")
    return "--disable-userns" in result.stdout and "--assert-userns-disabled" in result.stdout


def _interpreter_roots(selected: Path) -> set[str]:
    """Admit only installed Python prefixes, including virtualenvs and PATH Pythons."""
    roots: set[str] = set()
    for binary in (selected, selected.resolve()):
        prefix = binary.parent.parent
        if prefix == Path.home() or prefix in {
                Path("/"), Path("/Users"), Path("/home"), Path("/opt"), Path("/tmp"),
                Path("/private/tmp")}:
            continue
        library = prefix / "lib"
        if (prefix / "pyvenv.cfg").is_file() or any(
                path.is_dir() for path in library.glob("python*/site-packages")):
            roots.add(str(prefix))
    return roots


def isolated_command(command: list[str], root: Path, env: dict[str, str], *,
                     readonly_workspace: bool = False) -> tuple[list[str], dict[str, str]]:
    executable = backend()
    root = root.resolve()
    scratch = root / ".test-tmp"
    scratch.mkdir(exist_ok=True)
    # Second filter layer: keep it in sync with kernel_execution_policy._safe_environment,
    # which also admits host-declared task keys (STATEFRAME_TASK_ENV_KEYS). 10-03: this
    # layer re-stripped DATABASE_URL and every AHA check still died with pytest exit 4.
    if __package__:  # lazy: kernel_execution_policy imports this module at load time
        from .kernel_execution_policy import _host_task_env_keys
    else:
        from kernel_execution_policy import _host_task_env_keys
    allowed = {
        "PATH", "LANG", "LC_ALL", "PYTHONHASHSEED", "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
        "STATEFRAME_TASK_RELAY_SOCKET_DIR",
    }
    allowed |= _host_task_env_keys()
    clean = {key: value for key, value in env.items() if key in allowed}
    clean.update(HOME=str(scratch), TMPDIR=str(scratch), PYTHONDONTWRITEBYTECODE="1",
                 PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    # HOST-declared: a real project's pytest.ini can require installed plugins
    # (pytest-cov addopts, pytest-asyncio). The image's installed plugins are host
    # supplied, not model supplied. 10-03: forcing autoload off made every AHA check
    # exit 4 ("unrecognized arguments: --cov=...").
    if host_task_pytest_plugins() == "installed":
        clean.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)
    relay_dir = _declared_relay_dir(clean)
    selected = Path(test_python())
    resolved_selected = selected.resolve()
    interpreter_roots = (_interpreter_roots(selected)
                         if selected != Path(sys.executable) else set())
    if platform.system() == "Darwin":
        readable = {"/System", "/usr", "/Library/Apple",
                    sys.prefix, sys.base_prefix, str(Path(sys.executable).parent),
                    str(Path(sys.prefix).resolve()), str(Path(sys.base_prefix).resolve()),
                    str(Path(sys.executable).resolve().parent), str(root)}
        if selected != Path(sys.executable):
            # A selected executable can sit directly under HOME. Permit the file,
            # not its entire parent directory, before adding verified Python roots.
            readable.update((str(selected), str(resolved_selected)))
            readable.update(interpreter_roots)
        # Homebrew's launcher and dynamically linked runtime cross opt/Cellar
        # symlinks; these are installed code directories, not its var/data tree.
        if relay_dir is not None:
            readable.add(str(relay_dir))
        if any(str(p).startswith("/opt/homebrew/")
               for p in (Path(sys.executable), Path(sys.executable).resolve())):
            readable.update({"/opt/homebrew/Cellar", "/opt/homebrew/opt", "/opt/homebrew/lib"})
        readable.update(str(Path(value).resolve()) for value in site.getsitepackages())
        rules = " ".join(f"(subpath {json.dumps(value)})" for value in sorted(readable))
        ancestors = {str(parent) for value in readable for parent in Path(value).parents}
        metadata = " ".join(f"(literal {json.dumps(value)})" for value in sorted(ancestors))
        devices = '(subpath "/dev")'
        # Keep ordinary interpreter Mach/IPC operations while restricting its
        # information-bearing filesystem and network capabilities explicitly.
        writable = scratch if readonly_workspace else root
        profile = ("(version 1) (allow default) (deny network*) "
                   # The launcher reads ancestor directories while resolving its
                   # executable. Permit those exact directories, not their children.
                   f"(deny file-read-data (require-not (require-any {rules} {metadata} {devices}))) "
                   f"(deny file-read-metadata (require-not (require-any {rules} {metadata} {devices}))) "
                   f"(deny file-write* (require-not (require-any (subpath {json.dumps(str(writable))}) {devices})))")
        return [executable, "-p", profile, *command], clean
    # No host home, repository, private result tree, sockets or audit directory is mounted.
    arguments = [executable, "--die-with-parent", "--unshare-all", "--new-session",
                 "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    # The HOST (never the model) may declare that checks need the container's own
    # private network, e.g. a stage6 worker whose tests reach a per-run Postgres on an
    # unpublished Docker bridge. Only then share the net namespace and expose the two
    # name-resolution files (nothing else from /etc). 10-03: without this, AHA checks
    # failed to connect to their database inside the sandbox.
    if host_task_network() == "container":
        arguments.append("--share-net")
        for path in ("/etc/hosts", "/etc/resolv.conf"):
            if os.path.exists(path):
                arguments.extend(["--ro-bind", path, path])
    # Do not mount host /etc: deployed images may place credentials and service
    # configuration there. Python and its dynamic libraries are covered by the
    # explicit runtime prefixes below; DNS is unavailable by design.
    for value in ("/usr", "/bin", "/lib", "/lib64"):
        if Path(value).exists():
            arguments.extend(["--ro-bind", value, value])
    prefixes = {str(Path(sys.prefix).resolve()), str(Path(sys.base_prefix).resolve())}
    if selected != Path(sys.executable):
        prefixes.update((str(selected), str(resolved_selected)))
        prefixes.update(interpreter_roots)
    mounted: list[Path] = []
    for value in sorted(prefixes):
        if value.startswith("/usr") or value in {"/", "/home", "/root"}:
            continue
        # A path inside an already read-only-mounted prefix (a venv's bin/python inside the
        # venv) is visible already; binding it again makes bwrap create a mount point inside a
        # read-only mount and fail ("Can't create file at ...").
        if any(Path(value).is_relative_to(parent) for parent in mounted):
            continue
        arguments.extend(["--ro-bind", value, value])
        mounted.append(Path(value))
    if relay_dir is not None:
        # /tmp is a fresh sandbox tmpfs; expose only this run's relay directory.
        arguments.extend(["--bind", str(relay_dir), str(relay_dir)])
    arguments.extend(["--ro-bind" if readonly_workspace else "--bind", str(root), str(root)])
    if readonly_workspace:
        arguments.extend(["--bind", str(scratch), str(scratch)])
    arguments.extend(["--chdir", str(root), "--", *command])
    return arguments, clean


def _darwin_user_temp_dir() -> Path:
    """Some Python builds omit Darwin's confstr names; getconf exposes them."""
    try:
        directory = os.confstr("CS_DARWIN_USER_TEMP_DIR")
    except ValueError:
        try:
            result = subprocess.run(
                ["/usr/bin/getconf", "DARWIN_USER_TEMP_DIR"], check=True,
                capture_output=True, text=True, timeout=5, cwd="/",
                env={"PATH": os.defpath, "LANG": "C", "LC_ALL": "C"})
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError("cannot determine Darwin user temporary directory via getconf") from error
        directory = result.stdout.strip()
    if not directory or not Path(directory).is_absolute():
        raise RuntimeError("cannot determine absolute Darwin user temporary directory")
    return Path(directory)


def _darwin_xcrun_filter() -> str:
    """Developer shims use confstr's cache directory even with a private TMPDIR."""
    if sys.platform != "darwin":
        return ""
    directory = _darwin_user_temp_dir()
    # Anchor both ends and exclude slashes: never grant the directory or children
    # of a cache-named directory. Admit the canonical /private path and its alias.
    paths = {directory, directory.resolve(strict=True)}
    return " ".join('(regex #' + json.dumps("^" + re.escape(str(path)) +
                                          "/xcrun_db-[^/]+$") + ')'
                    for path in sorted(paths))


def capability_command(command, cap, *, platform_backend=None, shared_namespace=False,
                       single_process=False, private_pid_unit=False):
    """Build a shell/job boundary from explicit authority, never host test policy.

    shared_namespace is private to the Linux session supervisor: keep its PID/net
    namespaces while creating a per-command mount and user namespace. A none
    capability still gets a fresh, disconnected network namespace.
    """
    if not cap.sandboxed:
        return list(command), dict(cap.env)
    selected = platform_backend or ("sandbox-exec" if platform.system() == "Darwin" else "bwrap")
    if selected not in {"sandbox-exec", "bwrap"}:
        raise ValueError("execution backend must be sandbox-exec or bwrap")
    executable = sandbox_executable(selected)
    if executable is None:
        raise RuntimeError(f"execution capability requires {selected}; no unsandboxed fallback")
    reads = set(cap.read_roots) | set(cap.write_roots) | {cap.workspace,
            Path(__file__).with_name("sf_launch.py").resolve()}
    writes = set(cap.write_roots) | ({cap.workspace} if cap.workspace_writable else set())
    if selected == "sandbox-exec":
        cache_filter = _darwin_xcrun_filter()
        def filters(paths):
            return " ".join(f"(subpath {json.dumps(str(p))})" for p in sorted(paths))
        ancestors = {p for root in reads for p in root.parents}
        metadata = " ".join(f"(literal {json.dumps(str(p))})" for p in sorted(ancestors))
        devices = '(literal "/dev/null") (literal "/dev/zero") (literal "/dev/random") (literal "/dev/urandom") (literal "/dev/fd")'
        profile = ("(version 1) (allow default) "
                   f"(deny file-read* (require-not (require-any {filters(reads)} {metadata} {devices} {cache_filter}))) "
                   f"(deny file-write* (require-not (require-any {filters(writes)} {cache_filter} (literal \"/dev/null\")))) ")
        if single_process:
            profile += "(deny process-fork) "
        if cap.network != "full":
            profile += "(deny network*) "
        if cap.network == "loopback":
            profile += ('(allow network-inbound (local ip "localhost:*")) '
                        '(allow network-outbound (remote ip "localhost:*")) ')
        return [executable, "-p", profile, *command], dict(cap.env)
    arguments = [executable, "--die-with-parent"]
    if shared_namespace:
        arguments += ["--unshare-user", "--unshare-ipc", "--unshare-uts"]
        if cap.network == "none":
            arguments += ["--unshare-net"]
        # Each command gets its own PID namespace and a procfs of only its own processes:
        # the supervisor's root and fds stay hidden, while tools that read /proc (df, ps,
        # free, mount) work. The supervisor still sees every nested descendant in its table,
        # and orphans die with the command's namespace instead of reparenting to it.
        arguments += ["--unshare-pid", "--proc", "/proc"]
    else:
        arguments += ["--unshare-all", "--new-session", "--proc", "/proc"]
        if private_pid_unit:
            arguments += ["--as-pid-1"]
        if cap.network == "full":
            arguments += ["--share-net"]
    # Both paths created a user namespace above; refuse further user namespaces
    # when this binary provides the guard. Older binaries keep their existing contract.
    if _bwrap_userns_guard_supported(executable):
        arguments += ["--disable-userns", "--assert-userns-disabled"]
    arguments += ["--dev", "/dev", "--cap-drop", "ALL"]
    # Merged-/usr distributions make /lib, /bin, ... symlinks. Capability roots are resolved,
    # so recreate those links or the ELF loader (/lib/ld-linux-*.so) is missing in the sandbox.
    for link in ("/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32"):
        if os.path.islink(link):
            arguments += ["--symlink", os.readlink(link), link]
    # The loader finds libraries outside /lib and /usr/lib (e.g. /usr/local/lib/libpython*.so)
    # only through its cache; it holds library paths, nothing secret. The rest of /etc stays out.
    if os.path.isfile("/etc/ld.so.cache"):
        arguments += ["--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache"]
    # Parent mounts first, narrower mounts last; writable roots are explicit.
    # Bind resolved targets, then recreate uncovered aliases. Binding a symlink
    # follows it and can hide a required directory link or create the wrong tree.
    if __package__:
        from .kernel_exec_capability import executable_symlinks
    else:
        from kernel_exec_capability import executable_symlinks
    links = executable_symlinks(tuple(reads))
    mounts = {p.resolve(strict=True): "--ro-bind" for p in reads}
    mounts.update({p: "--bind" for p in writes})
    for path in sorted(mounts, key=lambda p: (len(p.parts), str(p))):
        arguments += [mounts[path], str(path), str(path)]
    system_links = {Path(p) for p in ("/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32")
                    if os.path.islink(p)}
    for path, target in sorted(links.items(), key=lambda item: (len(item[0].parts), str(item[0]))):
        if path not in system_links and not any(parent in mounts for parent in path.parents):
            arguments += ["--symlink", target, str(path)]
    # bwrap builds the root as a writable tmpfs and creates missing parents there; without this a
    # write outside every declared root "succeeds" into throwaway memory. Bind mounts keep their mode.
    arguments += ["--remount-ro", "/"]
    command = [os.path.realpath(sys.executable), "-I", str(Path(__file__).with_name("sf_launch.py").resolve()),
               "--network-none" if cap.network == "none" else "--process-filter",
               json.dumps(dict(cap.env), separators=(",", ":")), *command]
    arguments += ["--chdir", str(cap.workspace), "--", *command]
    return arguments, dict(cap.env)
