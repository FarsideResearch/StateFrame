# Changelog

## Unreleased

- Source bookmarks now number delivered pages on the shipped request path. Repeated unchanged source reads can return a notice, and a completed all-unchanged batch costs one no-progress watch unit. The optional read restriction remains off by default and is capped at two requests; see the running guide and known limitations.

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
