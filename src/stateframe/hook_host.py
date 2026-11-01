"""Host-side hook configuration, project trust, and runner lifecycle."""
from __future__ import annotations

import json
import os
import re
import secrets
import shlex
import stat
from dataclasses import asdict, dataclass
from pathlib import Path

from .engine.kernel_hook_exec import (
    HookContainmentError,
    HookReplayRefused,
    HookRunner,
    HookTape,
)
from .engine.kernel_hooks import (
    HookConfigError,
    HookSet,
    PanelEntry,
    create_bundle as _create_bundle,
    event_id,
    hookset_trust_digest,
    load_hook_files,
    load_project_hook_file,
    merge_hooksets,
    verify_bundle,
    verify_sources,
)

__all__ = ["HookContainmentError", "HookRunRefused", "HookTrustError", "HookRecordsError",
           "ProjectHookSelection", "resolve_project_hooks", "trust_project_hooks", "load_hooks",
           "create_bundle", "refuse_run_dir_under_roots", "hooks_from_spec", "run_start_hooks",
           "resume_start_hooks", "run_session_end", "hook_record_summary", "verify_resume_hooks",
           "rebuild_start_context"]

TRUST_SCHEMA = "stateframe-hook-trust/1"
_TRUST_RELATIVE = Path(".stateframe") / "trusted-hooks.json"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_NETWORK_LEVEL = {"none": 0, "loopback": 1, "full": 2}


class HookTrustError(ValueError):
    """The project hook trust store is malformed or cannot be updated safely."""


class HookRunRefused(RuntimeError):
    """A runner-side gating hook refused the task before the engine started."""


class HookRecordsError(RuntimeError):
    """The attempt-local hook record projection cannot be read safely."""


@dataclass(frozen=True)
class ProjectHookSelection:
    repo: Path
    hooks: HookSet | None
    digest: str | None
    message: str | None


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"invalid JSON constant {value}")


def _stateframe_dir(*, create: bool) -> Path | None:
    directory = Path.home() / ".stateframe"
    try:
        info = directory.lstat()
    except FileNotFoundError:
        if not create:
            return None
        try:
            directory.mkdir(mode=0o700, parents=True)
        except FileExistsError:
            pass
        except OSError as error:
            raise HookTrustError(f"cannot create hook trust directory {directory}: {error}") from error
        try:
            info = directory.lstat()
        except OSError as error:
            raise HookTrustError(f"cannot inspect hook trust directory {directory}: {error}") from error
    except OSError as error:
        raise HookTrustError(f"cannot inspect hook trust directory {directory}: {error}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise HookTrustError(f"hook trust path is not a real directory: {directory}")
    return directory


def _trust_path(*, create_directory: bool = False) -> Path | None:
    directory = _stateframe_dir(create=create_directory)
    return directory / "trusted-hooks.json" if directory is not None else None


def _validate_store(value: object) -> list[dict[str, str]]:
    if (type(value) is not dict or set(value) != {"schema", "entries"}
            or value.get("schema") != TRUST_SCHEMA or type(value.get("entries")) is not list):
        raise HookTrustError("trusted-hooks.json has an invalid stateframe-hook-trust/1 document")
    if len(value["entries"]) > 4096:
        raise HookTrustError("trusted-hooks.json contains too many entries")
    entries: list[dict[str, str]] = []
    identities: set[tuple[str, str]] = set()
    for entry in value["entries"]:
        if (type(entry) is not dict or set(entry) != {"repo", "file", "digest"}
                or any(type(entry.get(key)) is not str or not entry[key]
                       for key in ("repo", "file", "digest"))):
            raise HookTrustError("trusted-hooks.json contains an invalid entry")
        repo, file_path, digest = entry["repo"], entry["file"], entry["digest"]
        if (not Path(repo).is_absolute() or not Path(file_path).is_absolute()
                or os.path.normpath(repo) != repo or os.path.normpath(file_path) != file_path
                or not _DIGEST.fullmatch(digest)):
            raise HookTrustError("trusted-hooks.json contains a non-canonical path or digest")
        identity = (repo, file_path)
        if identity in identities:
            raise HookTrustError("trusted-hooks.json contains a duplicate trust entry")
        identities.add(identity)
        entries.append({"repo": repo, "file": file_path, "digest": digest})
    return entries


def _read_trust_store() -> list[dict[str, str]]:
    path = _trust_path()
    if path is None:
        return []
    try:
        info = path.lstat()
    except FileNotFoundError:
        return []
    except OSError as error:
        raise HookTrustError(f"cannot inspect hook trust store {path}: {error}") from error
    if (stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) & 0o077):
        raise HookTrustError(f"hook trust store must be a private regular file: {path}")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            opened = os.fstat(descriptor)
            if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1
                    or (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino)):
                raise HookTrustError(f"hook trust store changed while opening: {path}")
            with os.fdopen(descriptor, "rb", closefd=False) as source:
                raw = source.read((1 << 20) + 1)
        finally:
            os.close(descriptor)
        if len(raw) > 1 << 20:
            raise HookTrustError("trusted-hooks.json exceeds 1 MiB")
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_pairs,
                              parse_constant=_reject_constant)
    except HookTrustError:
        raise
    except (OSError, UnicodeError, ValueError) as error:
        raise HookTrustError(f"trusted-hooks.json is damaged: {error}") from error
    return _validate_store(document)


def _write_trust_store(entries: list[dict[str, str]]) -> None:
    if len(entries) > 4096:
        raise HookTrustError("trusted-hooks.json contains too many entries")
    path = _trust_path(create_directory=True)
    if path is None:
        raise HookTrustError("cannot locate hook trust store")
    payload = (json.dumps({"schema": TRUST_SCHEMA, "entries": entries}, ensure_ascii=False,
                          indent=1, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    directory = path.parent
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                         | getattr(os, "O_NOFOLLOW", 0))
    temporary = ".trusted-hooks-" + secrets.token_hex(12) + ".tmp"
    try:
        try:
            current = os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            current = None
        if current is not None and (not stat.S_ISREG(current.st_mode) or current.st_nlink != 1):
            raise HookTrustError(f"hook trust store is not a regular file: {path}")
        file_descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                                  | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=descriptor)
        try:
            os.fchmod(file_descriptor, 0o600)
            with os.fdopen(file_descriptor, "wb", closefd=False) as stream:
                stream.write(payload)
                stream.flush()
            os.fsync(file_descriptor)
        finally:
            os.close(file_descriptor)
        os.replace(temporary, path.name, src_dir_fd=descriptor, dst_dir_fd=descriptor)
        os.fsync(descriptor)
    except HookTrustError:
        raise
    except OSError as error:
        raise HookTrustError(f"cannot update hook trust store {path}: {error}") from error
    finally:
        try:
            os.unlink(temporary, dir_fd=descriptor)
        except FileNotFoundError:
            pass
        finally:
            os.close(descriptor)


def _project_path(repo: Path, *, strict_directory: bool = True) -> tuple[Path, Path | None]:
    try:
        root = Path(repo).resolve(strict=True)
    except OSError as error:
        raise HookConfigError(f"repository unavailable: {repo}: {error}") from error
    if not root.is_dir():
        raise HookConfigError(f"repository is not a directory: {root}")
    directory = root / ".stateframe"
    try:
        directory_info = directory.lstat()
    except FileNotFoundError:
        return root, None
    except OSError as error:
        if not strict_directory and isinstance(error, PermissionError):
            return root, None
        raise HookConfigError(f"project hook path unavailable: {directory}: {error}") from error
    if stat.S_ISLNK(directory_info.st_mode) or not stat.S_ISDIR(directory_info.st_mode):
        if not strict_directory:
            return root, None
        raise HookConfigError(f"project hook directory must be a real directory: {directory}")
    path = directory / "hooks.json"
    try:
        info = path.lstat()
    except FileNotFoundError:
        return root, None
    except OSError as error:
        if not strict_directory and isinstance(error, PermissionError):
            return root, None
        raise HookConfigError(f"project hook file unavailable: {path}: {error}") from error
    if (stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
        raise HookConfigError(f"project hook file must be a regular singly linked file: {path}")
    return root, path


def resolve_project_hooks(repo: Path, *, shell_network: str,
                          no_project_hooks: bool = False) -> ProjectHookSelection:
    """Resolve project trust once, using the original repository identity."""
    try:
        root = Path(repo).resolve(strict=True)
    except OSError as error:
        raise HookConfigError(f"repository unavailable: {repo}: {error}") from error
    if no_project_hooks:
        return ProjectHookSelection(root, None, None, None)
    root, path = _project_path(root, strict_directory=False)
    if path is None:
        return ProjectHookSelection(root, None, None, None)
    records = _read_trust_store()
    repo_text, file_text = str(root), str(path)
    trusted = [entry["digest"] for entry in records
               if entry["repo"] == repo_text and entry["file"] == file_text]
    if not trusted:
        return ProjectHookSelection(root, None, None, _untrusted_message(root, path))
    try:
        candidate = load_project_hook_file(root, shell_network="full")
        if candidate is None:
            return ProjectHookSelection(root, None, None, _untrusted_message(root, path))
        digest = hookset_trust_digest(candidate)
    except HookConfigError:
        return ProjectHookSelection(root, None, None, _untrusted_message(root, path))
    if digest not in trusted:
        return ProjectHookSelection(root, None, None, _untrusted_message(root, path))
    if shell_network not in _NETWORK_LEVEL:
        raise HookConfigError("shell_network: invalid network mode")
    if any(_NETWORK_LEVEL[hook.network] > _NETWORK_LEVEL[shell_network]
           for hook in candidate.definitions):
        raise HookConfigError(f"trusted project hooks require a higher --shell-network than {shell_network}")
    try:
        selected = load_project_hook_file(root, shell_network=shell_network)
    except HookConfigError:
        return ProjectHookSelection(root, None, None, _untrusted_message(root, path))
    if selected is None or hookset_trust_digest(selected) != digest:
        return ProjectHookSelection(root, None, None, _untrusted_message(root, path))
    return ProjectHookSelection(root, selected, digest,
                                f"hooks: project file trusted (digest {digest[:12]}...)")


def _untrusted_message(repo: Path, path: Path) -> str:
    return (f"project hooks not loaded: {path} is not trusted (or changed); review it, then run: "
            f"stateframe hooks trust --repo {shlex.quote(os.fspath(repo))}")


def _definition_lines(hooks: HookSet, file_path: Path, digest: str) -> list[str]:
    source_digest = next((item["sha256"] for item in hooks.files if item["path"] == str(file_path)), None)
    lines = [f"project hook file: {file_path}", f"hookset sha256: {digest}"]
    if source_digest:
        lines.append(f"file sha256: {source_digest}")
    for hook in hooks.definitions:
        details = {
            "event": hook.event,
            "tools": list(hook.tools),
            "paths": list(hook.paths),
            "changed": hook.changed,
            "argv": list(hook.argv),
            "network": hook.network,
            "writes": hook.writes,
            "digest": hook.definition_digest(),
        }
        lines.append(f"- {hook.name}: {json.dumps(details, ensure_ascii=False, sort_keys=True)}")
        for item in hook.files:
            lines.append(f"  file {json.dumps(item.relative)} sha256={item.sha256}")
    if not hooks.definitions:
        lines.append("- (no hooks)")
    return lines


def trust_project_hooks(repo: Path, *, remove: bool = False, printer=print) -> str | None:
    """Trust the exact project hook digest, or withdraw trust for its path."""
    root, path = _project_path(repo)
    if path is None:
        if remove:
            expected = str(root / ".stateframe" / "hooks.json")
            records = _read_trust_store()
            filtered = [entry for entry in records
                        if (entry["repo"], entry["file"]) != (str(root), expected)]
            if len(filtered) != len(records):
                _write_trust_store(filtered)
                printer(f"removed project hook trust for {expected}")
            else:
                printer(f"no project hook trust found for {expected}")
            return None
        raise HookConfigError(f"project hook file not found: {root / _TRUST_RELATIVE.parent / 'hooks.json'}")
    records = _read_trust_store()
    if remove:
        filtered = [entry for entry in records
                    if (entry["repo"], entry["file"]) != (str(root), str(path))]
        if len(filtered) != len(records):
            _write_trust_store(filtered)
            printer(f"removed project hook trust for {path}")
        else:
            printer(f"no project hook trust found for {path}")
        return None
    hooks = load_project_hook_file(root, shell_network="full")
    if hooks is None:
        raise HookConfigError(f"project hook file not found: {path}")
    digest = hookset_trust_digest(hooks)
    kept = [entry for entry in records
            if (entry["repo"], entry["file"]) != (str(root), str(path))]
    kept.append({"repo": str(root), "file": str(path), "digest": digest})
    kept.sort(key=lambda entry: (entry["repo"], entry["file"], entry["digest"]))
    _write_trust_store(kept)
    for line in _definition_lines(hooks, path, digest):
        printer(line)
    printer(f"trusted project hooks for {root}; trust store: {Path.home() / _TRUST_RELATIVE}")
    return digest


def load_hooks(paths: tuple[Path, ...], *, repo: Path, shell_network: str,
               project: ProjectHookSelection) -> HookSet | None:
    """Load trusted project hooks followed by explicit hook files, if enabled."""
    if not paths and project.hooks is None:
        return None
    explicit = load_hook_files(paths, shell_network=shell_network, repo=repo)
    hooks = merge_hooksets(project.hooks, explicit)
    if project.hooks is not None:
        current_digest = hookset_trust_digest(project.hooks)
        if current_digest != project.digest:
            raise HookConfigError("trusted project hook selection changed before prepare")
    # This verifies the config, executable, and bundled resource pins before the run directory is created.
    verify_sources(hooks.as_spec({"files": [], "directories": []}))
    return hooks


def create_bundle(hooks: HookSet, bundle_dir: Path) -> dict:
    return _create_bundle(hooks, bundle_dir)


def refuse_run_dir_under_roots(run_dir: Path, hooks: HookSet) -> None:
    candidate = Path(run_dir).resolve(strict=False)
    for root in hooks.read_roots():
        try:
            resolved = Path(root).resolve(strict=False)
            inside = candidate.is_relative_to(resolved)
        except (OSError, RuntimeError) as error:
            raise HookConfigError(f"cannot resolve hook read root {root}: {error}") from error
        if inside:
            raise HookConfigError(f"run directory is under hook read root {resolved}: {candidate}")


def hooks_from_spec(value: object) -> tuple[HookSet, list[dict[str, str]]]:
    if type(value) is not dict:
        raise HookConfigError("invalid pinned hook spec")
    core_keys = {"files", "definitions", "digest", "bundle_manifest"}
    if set(value) not in (core_keys, core_keys | {"start_context"}):
        raise HookConfigError("invalid pinned hook spec fields")
    start_context = value.get("start_context", [])
    if type(start_context) is not list:
        raise HookConfigError("hook start_context must be a list")
    entries: list[dict[str, str]] = []
    for entry in start_context:
        if (type(entry) is not dict or set(entry) != {"kind", "hook", "event", "label", "text"}
                or any(type(entry.get(key)) is not str for key in ("kind", "hook", "event", "label", "text"))
                or entry["kind"] != "start" or entry["event"] not in ("session_start", "prompt_submit")):
            raise HookConfigError("hook start_context contains an invalid panel entry")
        entries.append(dict(entry))
    core = {key: value[key] for key in core_keys}
    return HookSet.from_spec(core), entries


def _new_runner(run_dir: Path, hooks_spec: dict, workspace: Path, *, shell_network: str,
                run_seed: str, run_id: str, version: str, replay=lambda: False) -> HookRunner:
    hooks, _ = hooks_from_spec(hooks_spec)
    verify_bundle(hooks_spec["bundle_manifest"], run_dir / "hook-bundle")
    return HookRunner(run_dir, hooks, bundle_dir=run_dir / "hook-bundle", workspace=workspace,
                      shell_network=shell_network, run_seed=run_seed, run_id=run_id, version=version,
                      replay=replay)


def _display_event_result(result, event: str, printer) -> None:
    for invocation in result.runs:
        if invocation.outcome.error_code:
            message = json.dumps(invocation.outcome.reason, ensure_ascii=False)
            printer(f"hooks: {invocation.hook} {event} error {invocation.outcome.error_code}: {message}")


def run_start_hooks(run_dir: Path, hooks_spec: dict, workspace: Path, *, task: str, repo: Path,
                    shell_network: str, run_seed: str, run_id: str, version: str,
                    printer=print) -> list[dict[str, str]]:
    runner = _new_runner(run_dir, hooks_spec, workspace, shell_network=shell_network,
                         run_seed=run_seed, run_id=run_id, version=version)
    entries: list[PanelEntry] = []
    try:
        started = runner.run_event("session_start", boundary="session_start",
                                   data={"task": task, "repo": str(repo)}, label="")
        _display_event_result(started, "session_start", printer)
        entries.extend(started.entries)
        submitted = runner.run_event("prompt_submit", boundary="prompt_submit",
                                     data={"prompt": task}, label="")
        _display_event_result(submitted, "prompt_submit", printer)
        if submitted.decision == "deny":
            raise HookRunRefused(f"{submitted.deny_code or 'HOOK_DENIED'}: {submitted.deny_hook}: {submitted.reason}")
        entries.extend(submitted.entries)
        return [asdict(entry) for entry in entries]
    finally:
        runner.close()


def _start_event_records(run_dir: Path, hooks_spec: dict, *, run_seed: str) -> dict[str, tuple[bool, tuple]]:
    """Return whether each runner start event is complete and its ordered definitions."""
    hooks, _ = hooks_from_spec(hooks_spec)
    tape = HookTape(run_dir)
    progress: dict[str, tuple[bool, tuple]] = {}
    for event in ("session_start", "prompt_submit"):
        expected = tuple((index, hook) for index, hook in enumerate(hooks.definitions)
                         if hook.event == event)
        completed = []
        missing = []
        for index, hook in expected:
            record_id = event_id(run_seed, event, index)
            record = tape.lookup(record_id)
            if record is None:
                missing.append(hook)
                continue
            identity = record.get("identity")
            if (type(identity) is not dict or identity.get("event") != event
                    or identity.get("hook") != hook.name):
                raise HookConfigError(f"hook {hook.name} has a mismatched runner record; start a new run")
            if not record.get("completed"):
                raise HookConfigError(f"hook {hook.name} may or may not have run; start a new run")
            result = record.get("result")
            outcome = result.get("outcome") if type(result) is dict else None
            if (type(outcome) is not dict or outcome.get("decision") not in ("continue", "deny", "error")
                    or type(outcome.get("context")) is not str
                    or type(outcome.get("reason")) is not str):
                raise HookConfigError(f"hook {hook.name} has a malformed {event} record")
            completed.append(record)
        if completed and missing:
            raise HookConfigError(
                f"hook {missing[0].name} may or may not have run; start a new run"
            )
        progress[event] = (not missing, expected)
    start_complete, start_expected = progress["session_start"]
    prompt_complete, prompt_expected = progress["prompt_submit"]
    if start_expected and not start_complete and prompt_expected and prompt_complete:
        raise HookConfigError("prompt_submit hook record exists before session_start; start a new run")
    return progress


def _entries_from_tape(run_dir: Path, hooks_spec: dict, *, run_seed: str,
                       event: str, expected: tuple) -> list[dict[str, str]]:
    tape = HookTape(run_dir)
    entries: list[dict[str, str]] = []
    for index, hook in expected:
        record = tape.lookup(event_id(run_seed, event, index))
        if record is None or not record.get("completed"):
            raise HookConfigError(f"hook {hook.name} has no completed {event} record; start a new run")
        result = record.get("result")
        outcome = result.get("outcome") if type(result) is dict else None
        if type(outcome) is not dict or type(outcome.get("context")) is not str:
            raise HookConfigError(f"hook {hook.name} has a malformed {event} record")
        if outcome["context"]:
            entries.append(asdict(PanelEntry("start", hook.name, event, "", outcome["context"])))
    return entries


def resume_start_hooks(run_dir: Path, hooks_spec: dict, workspace: Path, *, task: str, repo: Path,
                       shell_network: str, run_seed: str, run_id: str, version: str,
                       printer=print) -> list[dict[str, str]]:
    """Finish start hooks that were not reached before an interrupted first prepare."""
    progress = _start_event_records(run_dir, hooks_spec, run_seed=run_seed)
    entries: list[dict[str, str]] = []
    for event, data in (("session_start", {"task": task, "repo": str(repo)}),
                        ("prompt_submit", {"prompt": task})):
        complete, expected = progress[event]
        if not expected:
            continue
        hook_runner = _new_runner(
            run_dir,
            hooks_spec,
            workspace,
            shell_network=shell_network,
            run_seed=run_seed,
            run_id=run_id,
            version=version,
            replay=(lambda: True) if complete else (lambda: False),
        )
        try:
            if complete:
                entries.extend(_entries_from_tape(run_dir, hooks_spec, run_seed=run_seed,
                                                  event=event, expected=expected))
                result = hook_runner.run_event(event, boundary=event, data=data, label="")
                _display_event_result(result, event, printer)
                # The event result validates the tape identity and provides the original decision.
                if event == "prompt_submit" and result.decision == "deny":
                    raise HookRunRefused(
                        f"{result.deny_code or 'HOOK_DENIED'}: {result.deny_hook}: {result.reason}"
                    )
            else:
                result = hook_runner.run_event(event, boundary=event, data=data, label="")
                _display_event_result(result, event, printer)
                if event == "prompt_submit" and result.decision == "deny":
                    raise HookRunRefused(
                        f"{result.deny_code or 'HOOK_DENIED'}: {result.deny_hook}: {result.reason}"
                    )
                entries.extend(asdict(entry) for entry in result.entries)
        except HookReplayRefused as error:
            raise HookConfigError(str(error)) from error
        finally:
            hook_runner.close()
    return entries


def run_session_end(run_dir: Path, hooks_spec: dict, workspace: Path, *, data: dict[str, object],
                    shell_network: str, run_seed: str, run_id: str, version: str,
                    printer=print) -> None:
    runner = _new_runner(run_dir, hooks_spec, workspace, shell_network=shell_network,
                         run_seed=run_seed, run_id=run_id, version=version)
    try:
        result = runner.run_event("session_end", boundary="session_end", data=data, label="")
        _display_event_result(result, "session_end", printer)
        for entry in result.entries:
            printer(f"hooks: {entry.hook} session_end output: {json.dumps(entry.text, ensure_ascii=False)}")
    finally:
        runner.close()


def hook_record_summary(run_dir: Path) -> dict[str, object]:
    path = Path(run_dir) / "hooks" / "records.jsonl"
    if not path.exists():
        return {"runs": 0, "denied": 0, "vetoes": 0, "errors": 0, "last_error": None}
    if path.is_symlink() or not path.is_file():
        raise HookRecordsError(f"hook records path is not a regular file: {path}")
    runs = denied = vetoes = errors = 0
    last_error = None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise HookRecordsError(f"cannot read hook records {path}: {error}") from error
    for number, line in enumerate(lines, start=1):
        if not line:
            continue
        try:
            item = json.loads(line, object_pairs_hook=_strict_pairs, parse_constant=_reject_constant)
        except ValueError as error:
            raise HookRecordsError(f"hook records line {number} is damaged: {error}") from error
        if (type(item) is not dict or type(item.get("hook")) is not str
                or type(item.get("event")) is not str or type(item.get("decision")) is not str
                or (item.get("error_code") is not None and type(item.get("error_code")) is not str)):
            raise HookRecordsError(f"hook records line {number} has an invalid shape")
        runs += 1
        denied += item["event"] in ("prompt_submit", "pre_tool") and item["decision"] == "deny"
        vetoes += item["event"] == "stop" and item["decision"] == "deny"
        if item.get("error_code") is not None:
            errors += 1
            last_error = f"{item['hook']}: {item['error_code']}"
    return {"runs": runs, "denied": denied, "vetoes": vetoes, "errors": errors,
            "last_error": last_error}


def verify_resume_hooks(run_dir: Path, hooks_value: object, *, run_seed: str) -> None:
    """Verify pinned hook sources, bundle and durable completion before attempt archival."""
    hooks, published_context = hooks_from_spec(hooks_value)
    if type(run_seed) is not str or not run_seed:
        raise HookConfigError("run spec lacks a hook identity seed")
    try:
        tape = HookTape(run_dir)
    except (OSError, ValueError) as error:
        raise HookConfigError(f"hook tape is damaged: {error}") from error
    pending = tape.pending()
    if pending:
        record = tape.lookup(pending[0])
        identity = record.get("identity") if isinstance(record, dict) else None
        name = identity.get("hook") if isinstance(identity, dict) else None
        if type(name) is not str or not name:
            raise HookConfigError("hook tape has a pending record without a hook name; start a new run")
        raise HookConfigError(f"hook {name} may or may not have run; start a new run")
    core = {key: hooks_value[key] for key in ("files", "definitions", "digest", "bundle_manifest")}
    try:
        verify_sources(core)
    except HookConfigError:
        raise
    try:
        verify_bundle(hooks_value["bundle_manifest"], Path(run_dir) / "hook-bundle")
    except HookConfigError as error:
        raise HookConfigError(f"hook definition, executable or file changed: {error}") from error
    progress = _start_event_records(run_dir, hooks_value, run_seed=run_seed)
    for event, (complete, expected) in progress.items():
        if "start_context" in hooks_value and expected and not complete:
            raise HookConfigError(f"hook {expected[0][1].name} may or may not have run; start a new run")
    if "start_context" in hooks_value:
        recorded_context = []
        for event in ("session_start", "prompt_submit"):
            complete, expected = progress[event]
            if expected and complete:
                recorded_context.extend(_entries_from_tape(
                    run_dir, hooks_value, run_seed=run_seed, event=event, expected=expected
                ))
        if published_context != recorded_context:
            raise HookConfigError("published hook start_context does not match its runner records; start a new run")
    for index, hook in enumerate(hooks.definitions):
        if hook.event != "prompt_submit":
            continue
        record = tape.lookup(event_id(run_seed, "prompt_submit", index))
        result = record.get("result") if isinstance(record, dict) else None
        outcome = result.get("outcome") if isinstance(result, dict) else None
        if type(outcome) is not dict:
            continue
        decision = outcome.get("decision")
        blocked = decision == "deny" or decision == "error" and hook.on_error == "block"
        if blocked:
            reason = outcome.get("reason")
            detail = reason if type(reason) is str and reason else "no reason supplied"
            raise HookConfigError(
                f"hook {hook.name} refused prompt submission: {detail}; start a new run"
            )


def rebuild_start_context(run_dir: Path, hooks_value: object, *, run_seed: str) -> list[dict[str, str]] | None:
    """Recreate unpublished context from complete records; None means start hooks never finished."""
    if type(hooks_value) is not dict:
        raise HookConfigError("invalid pinned hook spec")
    _, published = hooks_from_spec(hooks_value)
    if "start_context" in hooks_value:
        return published
    try:
        progress = _start_event_records(run_dir, hooks_value, run_seed=run_seed)
    except HookConfigError:
        raise
    except (OSError, ValueError) as error:
        raise HookConfigError(f"hook tape is damaged: {error}") from error
    if any(expected and not complete for complete, expected in progress.values()):
        return None
    entries: list[dict[str, str]] = []
    for event in ("session_start", "prompt_submit"):
        _complete, expected = progress[event]
        if expected:
            entries.extend(_entries_from_tape(run_dir, hooks_value, run_seed=run_seed,
                                              event=event, expected=expected))
    return entries
