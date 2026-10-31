# Code map

A diagram of how these fit together is in [ARCHITECTURE.md](../../ARCHITECTURE.md#code-map).

## Product package (`src/stateframe/`, standard library only)

| Module | Owns |
|---|---|
| `cli.py` | The `run`, `resume`, `inspect` and `chat` commands and their flags. |
| `runner.py` | One run: preflight, run directory, spec, driving the engine child, before/after checks, the outcome. |
| `test_interpreter.py` | Choosing the Python that runs your tests (`--test-python`, project venv, `python3` on PATH). |
| `_child.py` | The engine child process: installs the run's environment and runs one task. |
| `workcopy.py` | The safe working copy and `--in-place` apply. |
| `task_row.py` | Turning a repository plus a task into the engine's task description. |
| `outer_checks.py` | Running your checks in a fresh, sandboxed interpreter with a deadline. |
| `endpoint.py` | Endpoint configuration, API key from an environment variable, server probe. |
| `resume.py`, `replay_tape.py` | Deciding whether a stopped run can continue; replaying its recorded responses. |
| `inspect_run.py` | Reading a run directory's recorded requests and frames. |
| `chat.py` | The interactive session (`/diff`, `/undo`, `/apply`). |
| `outcomes.py` | Turning engine evidence into the printed outcome. |
| `checks.py` | Strict checks for acceptance tests over non-code output (reports, JSON, CSV). |
| `scripted_endpoint.py` | A scripted OpenAI-compatible server for tests and offline demos. |
| `profiles/` | Switch profiles and their loader. |
| `ts_grammars.py` | The tree-sitter grammar packages behind the repo map. |

## Engine (`src/stateframe/engine/`)

The engine is a flat package of modules, each owning one mechanism; `release-manifest.json` lists every file it ships. Entry points worth knowing:

| Module | Owns |
|---|---|
| `probes/react_repair.py` | The agent loop and the tool definitions (`TOOLS`: `grep`, `list_files`, `read`, `edit`, `str_replace`, `insert_before`, `authored_check`, `file_create`, `run_tests`, `read_check_receipt`, `recall`, `done`) and their handlers. |
| `kernel_stateframe_runtime.py` | `StateFrameRuntime`: building, fitting and qualifying each request (`prepare`) and recording what each action observed. |
| `kernel_frame_*.py` | The frame's sections and how they are budgeted. |
| `kernel_checkpoint.py` | Saving and loading signed checkpoints. |
| `kernel_execution_policy.py`, `kernel_test_isolation.py` | Sandboxed command and test execution. |
| `kernel_git_view.py` | Pinned, sanitized Git history observations and run-baseline diffs. |
| `kernel_web_fetch.py` | Direct-IP, SSRF-checked web observations with bounded text extraction. |
| `kernel_release_switch_registry.py` | Which `STATEFRAME_*` switches this engine reads; a setting with no reader is refused. |
| `mc_repo_intel_vendor/` | The repo map: syntax-aware outlines of functions and classes. |
