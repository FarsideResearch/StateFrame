"""Host-owned shell provisioning and confirmation transport."""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

from .engine.kernel_run_env import RunEnvError, create_run_env


def path_authority() -> tuple[list[str], list[str]]:
    """Freeze PATH directories and resolved executable locations before execution."""
    from .engine.kernel_exec_capability import executable_read_roots
    directories, roots = [], set()
    for value in os.get_exec_path():
        if not value or not Path(value).is_absolute():
            continue
        if not Path(value).is_dir():
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


def provision(options, run_dir: Path, spec: dict) -> None:
    from .runner import use_run_env_python

    settings = options.tool_settings
    if "shell_path_dirs" not in spec:
        spec["shell_path_dirs"], spec["shell_read_roots"] = path_authority()
        spec["shell_read_roots"] = sorted(set(spec["shell_read_roots"]) |
                                         {str(Path(p).resolve(strict=True)) for p in settings.shell_read})
    if settings.run_env and settings.shell != "off":
        source = spec.setdefault("run_env_source_python", options._test_python)
        try:
            environment = create_run_env(
                run_dir, Path(source), source_repo=Path(spec["repo"]),
                authorized_read_roots=tuple(Path(path).resolve(strict=True) for path in settings.shell_read))
        except RunEnvError as error:
            print(f"run environment off: {error}; dependency installs will not persist for checks")
            settings = replace(settings, run_env=False)
            options.tool_settings = settings
            spec["tool_settings"] = settings.as_dict()
            settings_file = run_dir / "tool-settings.json"
            settings_file.write_text(json.dumps(settings.as_dict(), indent=1, sort_keys=True))
        else:
            use_run_env_python(options, environment)
            spec["run_env"] = {"root": str(environment.root), "kind": environment.kind,
                               "env": dict(environment.env),
                               "external_import_roots": [str(path) for path in environment.external_import_roots]}
            spec["shell_read_roots"] = sorted(set(spec["shell_read_roots"]) |
                                              {str(path) for path in environment.external_import_roots})
            spec["capabilities"]["test_python"] = str(environment.python)
    (run_dir / "spec.json").write_text(json.dumps(spec, indent=1, sort_keys=True))


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
