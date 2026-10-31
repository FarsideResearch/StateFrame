# Changelog

## 0.2.1 (2026-10-31)

Correctness fixes in what the model is shown. This release does not fix the re-reading stall on large multi-file tasks described in KNOWN_LIMITATIONS.md; that work continues.

- Public-check status is tied to the current state of the whole product. Earlier, a run whose checks failed after edits to files other than the starting file could be shown "not run" instead of "failed".
- Edit suggestions never name a protected test, which the agent cannot edit. Files created or moved during the run stay eligible.
- A red-test confrontation without a target symbol is skipped instead of raising an error.

## 0.2.0 (2026-10-31)

- Add bounded foreground shell commands and read-only background jobs with explicit command and network policy. Host execution needs a CLI flag and prints a warning.
- Add guarded file deletion and moves, read-only sanitized Git history, opt-in public web fetch and explicitly trusted JSON command tools.
- Record tool intents and results for resume. Reproducible local effects are compared on replay; external effects require explicit replay authorization. Run-local dependency environments keep installs out of the source environment.
- Fit shell, job and custom-command reply delivery to the measured request budget while retaining the full taped observation within collector limits. Bound workspace reconciliation to 16 MiB per file, 256 MiB of regular-file data and 20,000 entries.
- Refuse and restore executable-bit changes, which the sealed candidate does not carry. Restrict the plain loop to foreground shell from the new tool set.
- On Linux, each shell command gets its own process namespace and `/proc`, so `df`, `ps`, `free` and `mount` work while the supervising process stays hidden.
- Document the sandbox limits and tool contracts in [Agent tools](docs/guide/tools.md). Live checks for this release are in the README.

### Security

Earlier releases ran some repository code outside the execution sandbox: imports before the first request, post-edit import checks, obligation checks, the startup check of the selected test interpreter, and host Git calls that consulted repository configuration. In 0.2.0 these operations run inside the execution sandbox. Repository indexing Git queries use a sanitized, offline Git view.

### Fixes

- Linux: runs no longer stop at their first check when the user already owns more than 64 processes (most desktop sessions). The check sandbox capped the user's total process count instead of the processes it creates.
- Linux: the check sandbox no longer fails to start when the test interpreter lives inside a mounted environment directory.
- The sandbox binary is found in its standard location when PATH does not list it (for example when started from an IDE or a scheduler).
- A transient fork failure while starting the check sandbox on a loaded machine is retried briefly instead of stopping the run.
- macOS: shell commands that use developer tools (make, git, cc) run inside the sandbox instead of opening the Command Line Tools installer.
- A check command whose program is not installed (for example `make`) now stops the run with "check command not found" instead of "sandbox unavailable".

## 0.1.0 preview

First preview release.

- Run history is kept in spool files on disk instead of in memory, and the ids the engine puts in its requests (request, receipt and check ids, run aliases, set ordering) no longer come from random values. Both are always on. Volatile test and tool output (durations, memory addresses, run-directory paths) is normalized before the model sees it, with the raw output kept in the run journal. Two runs on the same input sent identical requests on the measured 40-request workload apart from three clock fields and an edit-receipt ID; see KNOWN_LIMITATIONS.md for what can still differ. Disk use and memory on long runs for this build are not measured yet (KNOWN_LIMITATIONS.md).
- Resume replays the recorded run and checks the next request byte for byte before continuing live. Checkpoints for a faster restore are built in but off by default in this release (a checkpoint restore was not faithful in our tests; resume then replays). Measured on the previous engine: +0.6% wall time, and a run killed at request 150 resumed in 31 s with the next request byte-identical to the original. Checkpoints take a fixed disk base per run that grows with repository size (about 1.1 GB with two kept checkpoints on a large repository; see KNOWN_LIMITATIONS.md). `STATEFRAME_CHECKPOINT=save` turns checkpoints on.
- Frame-only engine: each request is system instructions, tools, the task, a typed state frame and the newest exchange. No chat history is sent.
- `stateframe run`, `stateframe chat` and `stateframe inspect`.
- Works on a copy of your repository; `--in-place` applies a passing result. Protected tests cannot be edited by the agent.
- No time limit and no step limit by default (stop with Ctrl-C, or set `--timeout` / `--max-turns`).
- One-command install with pinned, hash-checked dependencies from PyPI; `--offline` uses the bundled wheelhouse.
- Syntax-aware repo map and outlines for 12 languages from individual MIT grammar packages, on by default; structure outlines (ids and headings, selectors, keys, tables) for HTML, CSS, JSON, TOML and YAML from five more grammar packages (17 in total).
- Local-only endpoints by default.
- `examples/brief_to_report` and `stateframe.checks`: a recipe for file-based tasks whose acceptance is a pytest file over output files (reports, JSON, CSV). On the build just before this release, 11 of 11 short live runs of report, data-file and plan-document tasks passed their acceptance tests; `stateframe.checks` itself is not yet measured as task acceptance.
- Measured on the preview engine (re-measure on this release pending): the small-fix, multi-file create and chat demos pass; 13 of 14 recorded real tasks complete with no crashes (the 14th is a long task that stops with a clear refusal once its task state no longer fits the prompt budget); a real multi-file bug fix in a production Python codebase was solved live and passed hidden tests; the large-file refusal from the previous build is fixed; a 7-file feature passed all of its protected tests and 17 of 18 independent edge-case tests.
