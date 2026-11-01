import json
import subprocess
import sys

import pytest

from kernel_hooks import (
    HookConfigError,
    HookSet,
    create_bundle,
    load_hook_files,
    resolve_argv,
    verify_bundle,
    verify_sources,
)


def test_bundle_modes_layout_and_resume_pin(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    directory = tmp_path / "definitions"
    directory.mkdir()
    (directory / "sub").mkdir()
    script = directory / "sub" / "guard.py"
    script.write_text("print('ok')\n")
    data = directory / "sub" / "resource.txt"
    data.write_text("resource")
    config = directory / "hooks.json"
    config.write_text(
        json.dumps(
            {
                "schema": "stateframe-hooks/1",
                "hooks": [
                    {
                        "name": "guard",
                        "event": "pre_tool",
                        "argv": [sys.executable, "sub/guard.py"],
                        "files": ["sub/guard.py", "sub/resource.txt"],
                    }
                ],
            }
        )
    )
    hooks = load_hook_files([config], shell_network="none", repo=repo)
    bundle = tmp_path / "run" / "hook-bundle"
    manifest = create_bundle(hooks, bundle)
    assert (bundle / "0/sub/guard.py").stat().st_mode & 0o777 == 0o400
    assert (bundle / "0/sub/resource.txt").stat().st_mode & 0o777 == 0o400
    assert (bundle / "0/sub").stat().st_mode & 0o777 == 0o500
    assert resolve_argv(hooks.definitions[0], bundle)[1] == str(
        bundle / "0/sub/guard.py"
    )
    verify_bundle(manifest, bundle)
    spec = hooks.as_spec(manifest)
    assert HookSet.from_spec(spec).digest == hooks.digest
    verify_sources(spec)
    script.write_text("print('changed')\n")
    with pytest.raises(HookConfigError, match="changed"):
        verify_sources(spec)
    assert (bundle / "0/sub/guard.py").read_text() == "print('ok')\n"


def test_executable_bundle_and_separate_source_indexes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    configs = []
    for index in range(2):
        folder = tmp_path / f"hook{index}"
        folder.mkdir()
        script = folder / "run.sh"
        script.write_text(f"#!/bin/sh\necho {index}\n")
        script.chmod(0o700)
        config = folder / "hooks.json"
        config.write_text(
            json.dumps(
                {
                    "schema": "stateframe-hooks/1",
                    "hooks": [
                        {
                            "name": f"h{index}",
                            "event": "session_start",
                            "argv": ["run.sh"],
                            "files": ["run.sh"],
                        }
                    ],
                }
            )
        )
        configs.append(config)
    hooks = load_hook_files(configs, shell_network="none", repo=repo)
    bundle = tmp_path / "run" / "hook-bundle"
    manifest = create_bundle(hooks, bundle)
    assert (bundle / "0/run.sh").stat().st_mode & 0o777 == 0o500
    assert (bundle / "1/run.sh").stat().st_mode & 0o777 == 0o500
    assert (bundle / "0/run.sh").read_bytes() != (bundle / "1/run.sh").read_bytes()
    verify_bundle(manifest, bundle)
    (bundle / "0/run.sh").chmod(0o700)
    with pytest.raises(HookConfigError, match="bundle"):
        verify_bundle(manifest, bundle)


def test_shebang_interpreter_is_pinned(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    interpreter = tmp_path / "interpreter"
    interpreter.write_text('#!/bin/sh\nexec /bin/sh "$@"\n')
    interpreter.chmod(0o700)
    folder = tmp_path / "definitions"
    folder.mkdir()
    script = folder / "script"
    script.write_text(f"#!{interpreter}\necho hello\n")
    script.chmod(0o700)
    config = folder / "hooks.json"
    config.write_text(
        json.dumps(
            {
                "schema": "stateframe-hooks/1",
                "hooks": [
                    {
                        "name": "script",
                        "event": "session_start",
                        "argv": ["script"],
                        "files": ["script"],
                    }
                ],
            }
        )
    )
    hooks = load_hook_files([config], shell_network="none", repo=repo)
    bundle = tmp_path / "run" / "hook-bundle"
    manifest = create_bundle(hooks, bundle)
    verify_bundle(manifest, bundle)
    interpreter.write_text("#!/bin/sh\nexit 1\n")
    with pytest.raises(HookConfigError, match="changed"):
        verify_sources(hooks.as_spec(manifest))


def test_bundled_env_shebang_is_refused(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    script = tmp_path / "script"
    script.write_text("#!/usr/bin/env python3\nprint('unsafe')\n")
    script.chmod(0o700)
    config = tmp_path / "hooks.json"
    config.write_text(json.dumps({"schema": "stateframe-hooks/1", "hooks": [
        {"name": "script", "event": "session_start", "argv": ["script"], "files": ["script"]}
    ]}))
    with pytest.raises(HookConfigError, match="use an absolute interpreter path"):
        load_hook_files([config], shell_network="none", repo=repo)


def test_external_env_shebang_is_refused(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    executable = tmp_path / "external-script"
    executable.write_text("#!/usr/bin/env python3\nprint('unsafe')\n")
    executable.chmod(0o700)
    config_dir = tmp_path / "definitions"
    config_dir.mkdir()
    config = config_dir / "hooks.json"
    config.write_text(json.dumps({"schema": "stateframe-hooks/1", "hooks": [
        {"name": "script", "event": "session_start", "argv": [str(executable)]}
    ]}))
    with pytest.raises(HookConfigError, match="use an absolute interpreter path"):
        load_hook_files([config], shell_network="none", repo=repo)


def test_bundled_sibling_import_and_resource(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    folder = tmp_path / "definitions"
    folder.mkdir()
    (folder / "main.py").write_text("import helper\nprint(helper.read())\n")
    (folder / "helper.py").write_text(
        "from pathlib import Path\ndef read(): return Path(__file__).with_name('resource.txt').read_text()\n"
    )
    (folder / "resource.txt").write_text("bundled")
    config = folder / "hooks.json"
    config.write_text(
        json.dumps(
            {
                "schema": "stateframe-hooks/1",
                "hooks": [
                    {
                        "name": "script",
                        "event": "session_start",
                        "argv": [sys.executable, "main.py"],
                        "files": ["main.py", "helper.py", "resource.txt"],
                    }
                ],
            }
        )
    )
    hooks = load_hook_files([config], shell_network="none", repo=repo)
    bundle = tmp_path / "run" / "hook-bundle"
    create_bundle(hooks, bundle)
    argv = resolve_argv(hooks.definitions[0], bundle)
    result = subprocess.run(argv, capture_output=True, text=True, check=True)
    assert result.stdout == "bundled\n"
