# Changelog

## 0.2.8 (2026-11-01)

- Add opt-in project instructions: `--instructions` gives the agent the repository's `AGENTS.md` (else `CLAUDE.md`), and `--instructions=FILE` a chosen file. The text is bounded to 6,144 bytes with a visible truncation note, appended once to the end of the system prompt so the request head stays cache-stable, framed as untrusted repository content that cannot change tools, the sandbox or engine rules, and pinned by SHA-256 so resume refuses a changed or missing file. Without the flag, requests are unchanged byte for byte.
- Measured: on Qwen3.6-35B-A3B the three bundled examples passed 9 of 9 runs with and without a realistic `AGENTS.md` (27 and 28 model requests), so the feature ships off by default.

## 0.2.7 (2026-11-01)

- Use non-pytest `--test-cmd` commands as the engine's whole-command check, with exit-code results and bounded failure output in the frame.
- Discover common Python, JavaScript, TypeScript, Go, Rust and shell test files, plus configured check resources, for edit protection and change reporting.
- Keep pytest selector checks and their existing behavior for pytest commands.
- On macOS, allow the host-declared check executable through the existing sandbox read policy when it is outside Python's installed roots.
- Keep generic checks available through worker projection, edits and completion; label a successful generic run "verified by check command".
- Allow repositories without tests: use an explicit `--check-cmd`, a declared build or type check, or an agent-written check; label runs with no evidence `unverified: review the diff`.
- Change test protection from 0.2's lock on all pre-existing tests to `--protect-tests off` by default. Use `--protect-tests all` for the old behavior or `task` to lock judging tests. Report every changed test separately and qualify results when tests appear weakened.
- Let a run stop with structured evidence when a judging test is reported wrong, without claiming verification.
- Classify test verification from the command actually run; lock command-selected and initially failing task tests; detect Python suite-wide skips and assertion comparison changes; accept wrong-test evidence for strict pytest runs.
- Selected skills never bypass `task` or `all` test protection. With protection `off`, skill-driven test edits are reported and possible weakening qualifies the result, just like other test edits.
- Start generic runs at a source file when an empty package initializer would otherwise prevent the first request from being allocated.
- Keep no-flag request bytes identical to 0.2.6; advertise structured wrong-test evidence on explicit protection selection in editable modes and generic runs.

## 0.2.6 (2026-11-01)

- Add opt-in lifecycle hooks for start context, tool guards and notes, submission gates, and terminal notifications. Hooks run in the sandbox, use explicit configuration or digest-trusted project files, and replay completed results from the run tape. See [Hooks](docs/guide/hooks.md).
- `pre_tool` and `post_tool` hooks also see `skill_load` and MCP tool calls. An MCP call is checked against the arguments it will actually send, including a `why` field the server declares.
- With no hooks configured, the request bytes, tool tape and journal event kinds match 0.2.5; the child does not import the hook modules.
- On Linux, hook network mode `loopback` is a private network namespace, as it is for shell commands. A hook that must reach a service on the host's loopback needs `full`.
- Measured so far: two live runs on Qwen3.6-35B-A3B from a clean install, with hooks on all six events, were each submitted in 4 requests with their checks passing. In both, every event fired and the stop hook's one veto reached the model. This is a small sample on one small task.

## 0.2.5 (2026-11-01)

- Model requests never go through an HTTP proxy. A proxy set in a profile, or a macOS system proxy, could receive a local run's requests, including the task and API key.

## 0.2.4 (2026-11-01)

- Add explicitly selected reusable skills, a bounded catalog and on-demand instruction loading. Optional command definitions require separate trust and use the existing sandbox and replay path.
- Pin selected skill bytes for resume and refuse changed or missing bundles before replay.
- Add opt-in MCP stdio tool servers with sandboxed discovery and calls, bounded schemas and protocol messages, and tape-aware replay.
- Keep the protocol client in the engine, contain each server through a private Linux PID namespace or macOS process-creation denial, and bound model-facing omission notes to fixed reason codes.
- Harden Linux namespace isolation when supported by bubblewrap and account for nested PID namespace descendants in server resource limits.
- On Linux, a busy desktop (hundreds of threads owned by the same user, for example a browser) could make every sandboxed check fail with "Creating new namespace failed: Resource temporarily unavailable". The process limit is now set after the sandbox starts, relative to the real user's current thread count, so other programs' threads no longer block it. Sandboxed commands keep a hard task limit.

## 0.2.3 (2026-11-01)

- Requests to a stock llama-server failed with HTTP 415 once they reached 32 KiB; compression is now off by default. If compression is enabled for a compatible relay and a server refuses it, the request is sent once more without compression.
- Resume now tells releases apart: the engine fingerprint it compares is refreshed for this release (0.2.1 and 0.2.2 carried the 0.2.0 value, so a run started on an older release was admitted and then fell back to replaying its recorded requests).

## 0.2.2 (2026-11-01)

- Multi-file requests now offer current product code before repeated-read and navigation bookkeeping. Coverage lines show which source ranges are in view and whether a file changed since its last read when that comparison is available. Small files of at most 60 lines and 2 KiB can be shown whole when the request budget allows. Code bookmarks keep the failing test and related product source in view when the current check has a usable source location.
- A completed batch of proved unchanged reads costs one no-progress watch unit instead of one per file; repeated unchanged batches still end `STALLED`. `STATEFRAME_CODE_BOOKMARKS` is on by default; setting it off restores 0.2.1 behavior byte for byte. `STATEFRAME_READ_LOCKOUT` remains an independent, default-off switch with a two-request maximum. See the running guide and known limitations.
- Measured so far: all five live runs of the previously stalled multi-file task completed and were submitted, with all 14 tests passing (three on an earlier build of this release in 2, 4 and 26 requests; two on the release code in 7 and 23 requests). This is a small sample on one task.
- Safety and determinism for the new source view: whole files only add to what 0.2.1 would show (a large file never displaces smaller recent windows), every source shown in a request passes the same secret-path filter as the initial working copy (a secret-named file created mid-run is never shown), resume restores the request counter that numbers repeated reads, and the opt-in read lockout classifies tools by capability so a dynamically added reading tool cannot bypass it. An offline rebuild showed all 11 product files whole, up from 1 of 11, while the request fell from 85,664 to 77,017 bytes under the 82,026-byte ceiling.

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
