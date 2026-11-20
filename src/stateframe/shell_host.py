"""Host-owned shell provisioning and confirmation transport."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

from . import _STARTUP_SPEC_KEY, _startup_optimization_enabled
from .engine.kernel_exec_capability import (
    SYSTEM_SHELL_PATHS, PathInventoryStamps, current_path_authority_fingerprint,
    path_authority_fingerprint, path_directory_stamp, path_executable_roots,
)
from .engine.kernel_run_env import RunEnvError, create_run_env
from .storage_gc import _write_json_atomic


def _path_directories(raw_path: list[str] | None = None, *,
                      stamps: PathInventoryStamps | None = None) -> list[Path]:
    directories = {}
    for value in os.get_exec_path() if raw_path is None else raw_path:
        path = Path(value)
        if not value or not path.is_absolute():
            continue
        if not path.is_dir():
            if stamps is not None:
                stamps.skipped_target(path, directory=True)
            continue
        directory = path.resolve(strict=True)
        if not directory.is_dir():
            if stamps is not None:
                stamps.complete = False
            continue
        if stamps is not None:
            path_directory_stamp(path, directory, stamps)
        directories[directory] = None
    return list(directories)


_PATH_CACHE_KIND = "shell-path-inventory"
_PATH_CACHE_MAX_AGE = 3600
_PATH_CACHE_FACTS_LIMIT = 4 << 20


def _probe_store():
    from importlib import import_module

    name = f"{__package__}.send1_probe_cache"
    try:
        return import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        return None


def _path_snapshot(scanned, inputs, fingerprint):
    return {
        "directories": list(map(str, scanned)),
        "roots": sorted({str(root) for roots in scanned.values() for root in roots}),
        "inputs": inputs,
        "fingerprint": fingerprint,
    }


def _snapshot_for_stamps(scanned, stamps):
    selectors = {kind: sorted(paths) for kind, paths in stamps.items()}
    fingerprint = path_authority_fingerprint(stamps)
    if current_path_authority_fingerprint(selectors) == fingerprint:
        return _path_snapshot(scanned, selectors, fingerprint)
    return None


def _unpack_path_inventory(facts):
    import base64
    import zlib

    if not isinstance(facts, dict) or facts.keys() != {"inventory"}:
        return None
    data = facts["inventory"]
    if not isinstance(data, str) or len(data) > _PATH_CACHE_FACTS_LIMIT:
        return None
    decompressor = zlib.decompressobj()
    payload = decompressor.decompress(base64.b64decode(data, validate=True),
                                      _PATH_CACHE_FACTS_LIMIT + 1)
    if not decompressor.eof or decompressor.unused_data or len(payload) > _PATH_CACHE_FACTS_LIMIT:
        return None
    return json.loads(payload)


def _cached_path_inventory(facts, candidates, directories):
    facts = _unpack_path_inventory(facts)
    if not isinstance(facts, dict) or facts.keys() != {"by_directory", "stamps"}:
        return None
    entries, stamps = facts["by_directory"], facts["stamps"]
    if (not isinstance(entries, list) or not isinstance(stamps, dict)
            or not all(isinstance(paths, dict) for paths in stamps.values())):
        return None
    allowed = dict.fromkeys(map(str, candidates))
    scanned = {}
    for entry in entries:
        if not isinstance(entry, list) or len(entry) != 2:
            return None
        directory, roots = entry
        if (not isinstance(directory, str) or directory not in allowed or directory in scanned
                or not isinstance(roots, list) or not all(
                    isinstance(path, str) and path.startswith("/") and "\0" not in path
                    for path in roots) or roots != sorted(set(roots))):
            return None
        scanned[directory] = roots
    if (list(scanned) != [path for path in allowed if path in scanned]
            or any(str(directory) not in scanned for directory in directories)):
        return None
    snapshot = _snapshot_for_stamps(scanned, stamps)
    if snapshot is not None:
        return {Path(directory): roots for directory, roots in scanned.items()}, snapshot
    return None


def _publish_path_inventory(store, fingerprint, scanned, stamps):
    import base64
    import zlib

    payload = json.dumps({
        "by_directory": [[str(path), sorted(map(str, roots))] for path, roots in scanned.items()],
        "stamps": stamps,
    }, ensure_ascii=True, separators=(",", ":")).encode()
    if len(payload) <= _PATH_CACHE_FACTS_LIMIT:
        facts = {"inventory": base64.b64encode(zlib.compress(payload, level=1)).decode("ascii")}
        store.publish(_PATH_CACHE_KIND, fingerprint, facts)


def _path_inventory(system_paths, *, store=None, policy=(), seed=False):
    raw_path = os.get_exec_path()
    directories = _path_directories(raw_path)
    candidates = (*directories, *system_paths)
    if store is not None:
        from .send1_probe_cache import reads_available
        from .startup_cache_writes import defer

        if not seed and not reads_available():
            defer(_PATH_CACHE_KIND, lambda: _path_inventory(
                system_paths, store=store, policy=policy, seed=True))
            store = None
    host = store.host_fingerprint() if store is not None else None
    if seed and host is None:
        return None
    if host is not None:
        import zlib

        fingerprint = path_authority_fingerprint({
            "version": 1, "host": host, "path": os.environ.get("PATH"), "lookup": raw_path,
            "directories": list(map(str, directories)), "system": list(map(str, system_paths)),
            "platform": sys.platform, "uid": [os.getuid(), os.geteuid()], "policy": policy,
        })
        try:
            facts = None if seed else store.lookup(_PATH_CACHE_KIND, fingerprint, max_age=_PATH_CACHE_MAX_AGE)
        except OSError:
            facts = None
        try:
            cached = _cached_path_inventory(facts, candidates, directories)
        except (OSError, TypeError, ValueError, RecursionError, zlib.error):
            cached = None
        if cached is not None:
            return directories, *cached
        try:
            stamps = PathInventoryStamps()
            observed = _path_directories(raw_path, stamps=stamps)
            scanned, _ = path_executable_roots((*observed, *system_paths), stamps=stamps)
            snapshot = _snapshot_for_stamps(scanned, stamps.values) if stamps.complete else None
        except OSError:
            # Preserve the eager scan's treatment of unavailable metadata and races.
            snapshot = None
        if snapshot is not None:
            if observed == directories:
                try:
                    defer(_PATH_CACHE_KIND, lambda: _publish_path_inventory(
                        store, fingerprint, scanned, stamps.values))
                except OSError:
                    if seed:
                        raise
            return observed, scanned, snapshot
        if seed:
            return None
        directories = _path_directories(raw_path)
        candidates = (*directories, *system_paths)
    scanned, inputs = path_executable_roots(candidates)
    return directories, scanned, _path_snapshot(scanned, sorted(inputs), path_authority_fingerprint(inputs))


def _live_path_authority() -> tuple[list[str], list[str]]:
    from .engine.kernel_exec_capability import executable_read_roots

    directories, roots = [], set()
    for value in os.get_exec_path():
        if not value or not Path(value).is_absolute() or not Path(value).is_dir():
            continue
        directory = Path(value).resolve(strict=True)
        if not directory.is_dir():
            continue
        if str(directory) not in directories:
            directories.append(str(directory))
        for executable in directory.iterdir():
            if executable.is_file() and os.access(executable, os.X_OK):
                roots.update(map(str, executable_read_roots(executable)))
    return directories, sorted(roots)


def path_authority() -> tuple[list[str], list[str]]:
    """Freeze PATH directories and resolved executable locations before execution."""
    store = _probe_store() if _startup_optimization_enabled() else None
    if store is None:
        return _live_path_authority()
    directories, _, snapshot = _path_inventory((), store=store)
    return list(map(str, directories)), snapshot["roots"]


def provision(options, run_dir: Path, spec: dict) -> None:
    from .runner import use_run_env_python

    settings = options.tool_settings
    if getattr(options, "allow_source_tree", False):
        spec["allow_source_tree"] = True
    if "shell_path_dirs" not in spec:
        store = (_probe_store() if spec.get(_STARTUP_SPEC_KEY) is True
                 and not getattr(options, "_resuming", False) else None)
        if store is None:
            spec["shell_path_dirs"], spec["shell_read_roots"] = _live_path_authority()
            spec["shell_read_roots"] = sorted(set(spec["shell_read_roots"]) |
                                             {str(Path(p).resolve(strict=True)) for p in settings.shell_read})
        else:
            with store.launch_scope(True, getattr(options, "_stateframe_home_identity", None)):
                directories, scanned, snapshot = _path_inventory(
                    SYSTEM_SHELL_PATHS, store=store,
                    policy=(settings.shell, settings.host_authorization is None, settings.shell_network))
            spec["shell_path_dirs"] = list(map(str, directories))
            spec["shell_read_roots"] = sorted(
                {str(root) for directory in directories for root in scanned.get(directory, ())} |
                {str(Path(p).resolve(strict=True)) for p in settings.shell_read})
            spec["shell_path_snapshot"] = snapshot
    if settings.run_env and settings.shell != "off":
        source = spec.setdefault("run_env_source_python", options._test_python)
        workspace_objects = spec.get("environment", {}).get("STATEFRAME_WORKSPACE_OBJECTS") == "on"
        try:
            environment = create_run_env(
                run_dir, Path(source), source_repo=Path(spec["repo"]),
                authorized_read_roots=tuple(Path(path).resolve(strict=True) for path in settings.shell_read),
                allow_source_tree=getattr(options, "allow_source_tree", False),
                # Object checkpoint trees cannot retain hardlinks that reach outside the run.
                allow_seed_hardlinks=not workspace_objects)
        except RunEnvError as error:
            print(f"run environment off: {error}; dependency installs will not persist for checks",
                  file=sys.stderr)
            settings = replace(settings, run_env=False)
            options.tool_settings = settings
            spec["tool_settings"] = settings.as_dict()
            settings_file = run_dir / "tool-settings.json"
            _write_json_atomic(settings_file, settings.as_dict(), mode=0o600)
        else:
            use_run_env_python(options, environment)
            spec["run_env"] = {"root": str(environment.root), "kind": environment.kind,
                               "env": dict(environment.env),
                               "external_import_roots": [str(path) for path in environment.external_import_roots]}
            spec["shell_read_roots"] = sorted(set(spec["shell_read_roots"]) |
                                              {str(path) for path in environment.external_import_roots})
            spec["capabilities"]["test_python"] = str(environment.python)
    _write_json_atomic(run_dir / "spec.json", spec)


def answer_pending(run_dir: Path, confirm=None) -> None:
    """Only the host sees the TTY; an unattended run always declines."""
    request = run_dir / "shell-ask.json"
    answer = run_dir / "shell-answer.json"
    if not request.exists():
        return
    value = json.loads(request.read_text())
    call_id = value["call_id"]
    if answer.exists() and json.loads(answer.read_text()).get("call_id") == call_id:
        return
    approved = bool(confirm(value["command"])) if confirm is not None else False
    temporary = run_dir / "shell-answer.pending"
    temporary.write_text(json.dumps({"call_id": call_id, "approved": approved}))
    temporary.replace(answer)
