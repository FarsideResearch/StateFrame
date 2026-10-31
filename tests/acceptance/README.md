# Agent-tool acceptance suite

End-to-end scenarios for the agent tools. Each test starts the public `stateframe run` CLI against a scripted
local endpoint (`python -m stateframe.scripted_endpoint`), then checks the recorded model requests, the tool
replies, the working copy, the summary, the final diff and the original repository.

Run the fixture checks anywhere:

```bash
python -m pytest -q tests/acceptance/test_fixture_integrity.py
```

The end-to-end tests are marked `host_sandbox`: they start the real execution sandbox (`sandbox-exec` on macOS,
`bwrap` on Linux) and local sockets, so run them on a host where both are available:

```bash
python -m pytest -q tests/acceptance -m host_sandbox
```

Run one pytest process at a time: several of these scenarios run the engine's own checks with pytest.

The fixture repositories under `fixtures/` are excluded from collection. The `host_sandbox` marker is registered
in `tests/conftest.py`.

`fixtures/scripts/pytest_completion.script.json` drives a scenario where the agent runs `python -m pytest`
through the shell; it is meant to be run directly with the scripted endpoint rather than from inside pytest.

| Module | Covers |
|---|---|
| `test_shell_workflows.py` | Environment inspection, exploration, builds and non-pytest checks, formatters, local dependency installs into the run environment, scripts and code generation, shell effects next to file tools |
| `test_effects_policy.py` | Read-only command smuggling, writes outside the workspace, protected tests restored, edit-scope restore, symlinks, output floods, bounded process trees, stdin, ANSI output |
| `test_additional_edges.py` | Dangerous flags and environment assignments in read-only mode, empty and multi-line commands, path escapes, binary output, private web addresses, custom tool network limits |
| `test_git_web.py` | Git history, excluded secret paths, non-Git repositories, web fetch refusals |
| `test_jobs_custom.py` | Background dev server with waits and stop, `done` stopping a running job, custom tools, ask mode without a terminal |
| `test_resume_completion.py` | Kill and resume with taped shell and file operations, external replay refusal, volatile output, completion ownership, disabled tools, prompt budget with every tool enabled |
| `test_fixture_integrity.py` | Fixture and script validity, old run specs refused |

Cases that need a live external service (network package installs, public web pages) are outside this offline
suite.
