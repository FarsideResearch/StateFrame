# Code map

A diagram of how these fit together is in [ARCHITECTURE.md](../../ARCHITECTURE.md#code-map).

## Product package (`src/stateframe/`, standard library only)

| Module | Owns |
|---|---|
| `cli.py` | The `run`, `resume`, `inspect` and `chat` commands and their flags. |
| `skills.py` | Skill directory parsing, installation, path resolution and pinned digest checks. |
| `instructions.py` | Project instructions: AGENTS.md / CLAUDE.md discovery, size bound and resume digest pin. |
| `runner.py` | One run: preflight, run directory, spec, driving the engine child, before/after checks, the outcome. |
| `hook_host.py` | Project hook trust and loading, pinned bundles, runner lifecycle events, terminal records and resume checks. |
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
| `probes/react_repair.py` | The agent loop and the tool definitions (`TOOLS`: `grep`, `list_files`, `read`, `edit`, `str_replace`, `insert_before`, `authored_check`, `file_create`, `file_delete`, `file_move`, `shell`, `run_tests`, `read_check_receipt`, `recall`, `done`) and their handlers. |
| `kernel_file_ops.py`, `kernel_done_is_submit_authority.py` | Guarded file removal and moves, including refusal of executable moves; versioned sealed candidate writes and deletions. |
| `kernel_hooks.py` | Hook definitions, strict loading, bundle pins, matchers, payloads, output handling and the optional request panel. |
| `kernel_hook_exec.py`, `kernel_hook_driver.py` | Sandboxed hook execution, process and stream bounds, durable replay records, scoped formatter effects and raw output files. |
| `kernel_hook_wiring.py` | Child event seams, pre-tool decisions, post-tool notes, stop gates and checkpointed hook state. |
| `kernel_stateframe_runtime.py` | `StateFrameRuntime`: building, fitting and qualifying each request (`prepare`) and recording what each action observed. |
| `kernel_code_bookmarks.py`, `kernel_read_lockout.py` | Current-failure source bookmarks and optional bounded read restriction after unchanged reads. |
| `kernel_frame_*.py` | The frame's sections and how they are budgeted. |
| `kernel_checkpoint.py` | Saving and loading signed checkpoints. |
| `kernel_shell_tool.py`, `kernel_exec_session.py` | Foreground shell admission, per-run execution, effects, confirmation and replay. |
| `kernel_execution_reply_budget.py` | Fit shell, job and custom-command reply delivery against the measured whole request without changing the taped observation. |
| `kernel_shell_policy.py`, `kernel_tool_settings.py` | Command policy, output normalization and host-issued tool settings. |
| `kernel_exec_capability.py`, `kernel_exec_supervisor.py`, `kernel_exec_tracker.py`, `sf_launch.py` | Per-command authority, executable symlink and shebang chain admission, supervised process lifetime and resource limits. |
| `kernel_effects.py`, `kernel_run_env.py` | Bounded full workspace effect reconciliation, including executable-bit restoration, and run-local dependency environments. |
| `kernel_agent_schemas.py`, `kernel_agent_tools.py`, `kernel_agent_jobs.py`, `kernel_job_output.py` | Schemas, per-run tool registration and Git/web adapters, background jobs, frozen job snapshots, bounded output and schema budget admission. |
| `kernel_skills.py` | Bounded skill catalog and user-role body loading. |
| `kernel_project_instructions.py` | The framed project-instructions block appended to the end of the engine system message. |
| `kernel_custom_tools.py`, `kernel_mcp_tools.py`, `kernel_mcp_driver.py`, `kernel_tool_tape.py` | Trusted command and MCP stdio definitions, bounded server exchanges and durable tool replay records. |
| `kernel_execution_policy.py`, `kernel_test_isolation.py` | Sandboxed command and test execution. |
| `kernel_git_view.py` | Pinned, sanitized Git history observations, commit-rooted `show`, run-baseline diffs and offline Git queries for repository indexing. |
| `kernel_plain_host.py`, `kernel_plain_loop.py` | The alternate plain loop: foreground shell only from the new tool set; selected skills and Git, web, custom and MCP settings are refused. |
| `kernel_safe_probe.py`, `kernel_ship_tree_smoke.py` | Sandboxed, read-only repository-code diagnostics and provisioning, including imports before the first request and after edits; cleanup failures stop the run. |
| `kernel_web_fetch.py` | Direct-IP, SSRF-checked web observations with bounded text extraction. |
| `kernel_release_switch_registry.py` | Which `STATEFRAME_*` switches this engine reads; a setting with no reader is refused. |
| `mc_repo_intel_vendor/` | The repo map: syntax-aware outlines of functions and classes. |
