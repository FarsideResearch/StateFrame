# Known limitations

## Supported environment

This preview is supported on macOS on Apple silicon and on Linux x86_64 (tested on Ubuntu 22.04.5, kernel 6.8, Python 3.11: install, the three demos, the bubblewrap sandbox with no network and no writes outside the run, and `kill -9` then resume all pass). Long runs (200-300 requests) and the speed and memory numbers were measured on macOS only; Linux is tested end to end on short runs. Checkpoints were measured on macOS; on Linux, resume is verified through full replay. Linux arm64 preview runs are untested. Ubuntu 24.04 and later restrict unprivileged user namespaces, which bubblewrap needs; that release is untested, and if the sandbox cannot start there the usual setting is `sudo sysctl kernel.apparmor_restrict_unprivileged_userns=0` (or an AppArmor profile that allows `bwrap`). Docker is not supported. Use Python 3.11 or newer with venv and ensurepip (Python 3.14.1 is not supported: the pinned networkx 3.6.1 excludes that one release; other 3.14.x releases work). The installer downloads pinned, hash-checked dependencies from PyPI; `./install.sh --offline` uses the bundled wheelhouse instead, without schema validation or the syntax-aware repo map.

The new command containment path was exercised in a Linux aarch64 container. Its x86_64 path has not been verified for this tool release. That is separate from the earlier Linux x86_64 preview checks above.

## Request and model capacity

Request body compression is off by default because stock llama-server does not accept gzip request bodies. Enable `STATEFRAME_HTTP_GZIP=on` or `STATEFRAME_PLAIN_GZIP=on` only in a custom profile for a relay that inflates them. In the default run mode, a compressed request refused with HTTP 415 is resent once without compression within the same deadline. If the run deadline stops a request while that uncompressed resend is in flight, the error is recorded but the two attempts are not listed separately.

The engine has a **33,344-token prompt budget** and requires a model server context slot that supports **65,536 tokens**. The slot is server capacity, not a recommendation for cache RAM. Set cache RAM sized to your machine. These are the sizes we built and measured with: a single 48 GB GPU (RTX A6000 class) running Qwen3.6-35B-A3B, a mixture-of-experts model (35B total, about 3B active per token), 4-bit UD-Q4_K_XL, under llama.cpp, with one server sharing a 262,144-token cache across 4 slots (`-np 4 -c 262144 --kv-unified`), so 65,536 tokens per run. A stock 24 GB card was not tested. They are the defaults; both can be changed (see Context size in docs/guide/running.md). Only the default 65,536-token slot is measured on long runs; other sizes (32,768 and 131,072 included) are tested on short scripted runs only.

Large tasks or repositories can require more state and source context than fits. In that case, the engine may stop before completing the task.

- Very long tasks (roughly 25 or more model turns) and larger multi-file changes (around 10 files or more) can stop with a refusal when the task state no longer fits the prompt budget. Nothing is lost: the partial result is saved.
- Shell, job and custom-command reply delivery shrinks against the measured whole request when a populated frame nears the prompt limit. The full taped observation remains on disk, subject to the command collector's output cap. If the authoritative minimum still cannot fit, the request is refused.
- Selected skill descriptions are visible in the typed frame. Skill bodies are delivered only after `skill_load`, with at most two resident bodies. Other files in a skill directory are not shown to the model. Bodies may be omitted to fit the prompt budget, then loaded again. Skill text is user content, not a security boundary.

## Agent tools

- Shell, delete, move, read-only Git, opt-in web fetch, background jobs, explicitly configured command tools and opt-in MCP stdio tool servers are available. MCP HTTP transport, resources, prompts, sampling and persistent sessions are not. On macOS, stdio servers must be single-process programs; wrappers that spawn children need a direct final-program argv. See [Agent tools](docs/guide/tools.md).
- MCP tools and configured command tools are offered while the agent explores, but late phases (after the first edit, while verifying, and when closing out) narrow the tool menu to a fixed set, and these tools can drop out of it. A lookup the agent needs after it has started editing may therefore be unavailable for the rest of the run. Ask for such lookups early in the task.
- For shell commands and background jobs on macOS, a process deliberately double-forked and orphaned between supervisor sweeps can outlive a run. Its sandbox profile continues to restrict filesystem and network access. Aggregate resident memory is polled rather than enforced as a hard cap, so usage can overshoot between samples.
- Linux checks and commands set a finite task limit from real-UID threads visible inside their PID namespace plus headroom. Checks and commands also use a per-command cgroup `pids.max` when a delegated writable v2 subtree is available, with supervisor descendant counting for commands. Hidden tasks and tasks created during the `/proc` scan can still make the UID-wide limit refuse safe work or allow an overshoot when no cgroup is available; inherited lower hard limits can also prevent startup.
- Linux has one network placement per execution session. A full-network session cannot later use the same shared loopback namespace; a narrower command gets its own loopback namespace and cannot reach a service in the full-network namespace.
- Background jobs are stopped at completion and are not relaunched on resume. A running job from the stopped attempt appears as `lost on restart` when the run continues.
- External operations, including full-network shell calls and custom tools marked `external`, are never repeated automatically on resume. `--replay-external` repeats them knowingly; the command may repeat an outside effect. An incomplete tool operation refuses resume because its outside effect cannot be inferred from the run record.
- Network package installs need `--shell-network full`. The default loopback mode cannot reach package indexes. A run-local install does not update the project's dependency declaration. If a selected virtual environment cannot be safely cloned or remapped, the run environment is disabled with a message.
- Web fetch is off by default, accepts public HTTP(S) text only and blocks private addresses, including redirects to them. Its untrusted-content label is not a prompt-injection security boundary.
- Shell output is neither revision-bound source-read evidence nor engine check evidence. Use `read` and `run_tests` respectively. A command that exits zero or runs pytest itself does not complete a task.
- Custom command definitions require a finite positive timeout no greater than 600 seconds; larger values fail at load.
- Executable-bit changes are refused and restored, including creation of executable files. Shell deletion of an executable is refused; `file_delete` supports intentional removal, while `file_move` refuses executable sources. The sealed candidate does not carry executable-mode changes.
- Workspace reconciliation stops without promoting command effects when any regular file exceeds 16 MiB, aggregate regular-file bytes exceed 256 MiB, or the tree exceeds 20,000 entries. These bounds also apply before execution when the snapshot is captured.
- The shipped product profile uses the composed loop. The plain loop accepts foreground shell but refuses background jobs and Git, web or custom tool settings before execution. Lean mode is not supported by the product profile.

## Lifecycle hooks

Hooks always use the execution sandbox. There is no host-execution option or unsandboxed fallback; a sandbox startup failure is a hook error. Windows is unsupported. Linux desktop notifications cannot reach the user's D-Bus session from the sandbox; use an explicitly configured webhook instead. On Linux, network mode `loopback` gives a hook its own private loopback, so a service listening on the host's loopback needs `full`. The macOS notification and webhook samples are examples and are not exercised by the automated end-to-end tests.

If a process is killed while a hook has a durable intent but no completion record, StateFrame cannot know whether an outside effect occurred and refuses to resume that run. Completed hook decisions are served from the run tape. A hook script is pinned, but a hook that deliberately runs project code from the mutable working copy can be influenced by the agent's edits to that code. The `[USER HOOKS]` label describes the text's source; it does not make that text a prompt-injection boundary. See [Lifecycle hooks](docs/guide/hooks.md) for the write, network and size limits.

## Checks

Ordinary non-verbose `go test` output reports package success without a positive test count, so it receives the weaker `verified by check command` label. Use `go test -v` to include named passing tests for test verification.

Go and Rust command paths are implemented but their toolchains are not exercised on this project's automated gates. A repository with no tests can still finish from a passing declared build or type check, an agent-written check with a failing parent and passing candidate, or an unverified result when no check exists. Structured wrong-test evidence is advertised on an explicit `--protect-tests task|off` selection, or `all` for a generic runner; omitting the new flags preserves the 0.2.6 request schema.

When tests exist, the default check uses pytest test files (`--check` names them), and its engine checks support per-test selectors. A non-pytest `--test-cmd` uses one whole-command result by exit code, with no per-test selector or collection evidence. A command that cannot start, exits 126 or 127, or times out cannot pass. Without tests or an explicit command, StateFrame tries a declared build or type check, then an agent-written check. With no check, the result is `unverified: review the diff`; it is not a test pass. Checks run under the existing sandbox. A generic check needs a usable Python interpreter for StateFrame itself, but that interpreter does not need pytest. The check can inspect output files, not only code (see "Tasks whose output is files" in the README). Test weakening detection for JavaScript, TypeScript, Go, Rust and shell is heuristic and can miss changes in meaning.

In a repository with more than 200 source files, a run must name the files it may change with `--target` and `--edit`; the rest of the repository stays available as read-only context.

## Optional analysis

- **Syntax-aware repo map: on by default** for 12 languages (Python, JavaScript, TypeScript, TSX, Go, Rust, Java, Ruby, PHP, C, C++, C#), each from its own MIT grammar package. For HTML, CSS, JSON, TOML and YAML files the engine reads structure (ids and headings, selectors, keys, tables) from five more grammar packages. Other languages, including Vue, use the built-in text-scan repo map. `./install.sh --no-repomap` skips it.
- **Named working sets are not available to the model in this release.** The engine keeps its own set of files it is working on, but the tool that lets the model name and manage working sets is not offered in its requests yet.
- **Offline install** (`--offline`): schema validation and the syntax-aware repo map are OFF; the engine skips schema validation and uses the text-scan repo map.

## Files dropping out of long runs

To stay inside the prompt budget, the engine evicts file contents the model read earlier. Usually the task state says which file was evicted. On long runs that read many files, some evictions carry no notice. Measured on one long task: 0 of 15 checks unannounced at request 11, 30% at request 30, 45% at request 61. The model can read the file again. `STATEFRAME_EVICTFIX=on` (opt-in, off by default) keeps more recent reads in the frame, but in this release it makes host time per turn grow on long runs, and we have not measured an edit gain from it.

## Experimental switches

`STATEFRAME_MICROGRAPH=on` is experimental and off by default. With it off, requests are byte-identical to a build without it. We have not measured whether turning it on helps, so we make no quality claims for it.

## Models

This release has an early five-run result on the previously stalled multi-file refactor: Qwen3.6-35B-A3B submitted with all 14 tests passing in all five runs. On 0.2.1, Qwen3.6-35B-A3B solved the HTTP transport task (graded as a correct candidate), while the same refactor ended in a no-progress stop. DeepSeek V4 Flash solved the refactor on that build in 2 of 2 runs, with all 14 tests passing. These small samples are smoke tests, not success rates. On an earlier release, Flash also solved the HTTP transport task and solved a harder multi-step task in 1 of 2 runs (hidden tests); the other run exhausted its request budget mid-fix. Repeated reads can still occur on a larger single-file task. Other models are untested.

## Per-step models

- A routed run's in-flight request (the one that had no reply when the run stopped) is checked against its rebuilt copy on the first resume. On later resumes it is checked only if the previous resume recorded that its own check passed. Otherwise, for example when a resumed run stops again before its check, StateFrame resumes by full replay without an in-flight check, prints a line saying so and records it in the run.
- Routed runs with checkpoints enabled: resume replays; checkpoint restore of routed runs is not supported. Checkpoints are still written if `STATEFRAME_CHECKPOINT=save` is set, but a routed resume never uses them.
- If only a route record was written before a stop, the original send had not started. Resume checks the route, then sends the rebuilt request live. Replay also masks standalone 32-character hex values, test durations and object addresses; a meaningful change with one of those shapes can compare equal.
- Roles are assigned mechanically. The engine cannot tell reading from editing before the reply, because the tools it offers do not depend on what the step is for. So an edit step first goes to the model of the role it was assigned, usually `read` (or `plan` on the first request and after a failed test run); if that reply writes files it is thrown away unrun and the step is asked again of the `edit` model. Each edit step costs one extra request on that first model.
- Shell commands are not escalated. A `shell` call counts as a reading step and runs on whichever role's model asked for it, even when the command writes files, so a `read` model can change the working copy through `shell`. The sandbox and effect checks still apply and record every change. Use `--shell read-only` to prevent this.
- Switching models between requests resets the server's prompt cache for that request, so a routed run reuses less cache than a single-model run.
- The token allowance for chat-template overhead is calibrated on Qwen3.6-35B-A3B. Each role's tokens are counted on its own server, but a model with a much larger template can come close to its slot on the largest frames.
- Every role's server slot must be at least the run's context size (65,536 tokens by default). A small model served with a smaller slot is refused at start when its server reports the slot size; a server that reports none is not checked.
- Routed roles speak the OpenAI-compatible chat API. There is no Anthropic or other provider-specific adapter yet.
- Routing makes a run cheaper only when the `read` model is good at the tool protocol. In our first live check (the discount_bug example, reading on MiniCPM5-1B, everything else on Qwen3.6-35B-A3B), both runs passed the tests. The routed run needed 38 requests against 3, because most of the small model's replies made no tool call or ran to their token limit, and it took over 20 times as long. Try a read model on a short task first, and cap `read.max_tokens`.
- Cost figures are estimates from the rates you set, using the token counts each server reports. A server that reports no usage leaves its cost unknown.

## Multi-file source and repeated reads

With `STATEFRAME_CODE_BOOKMARKS` on by default, the frame offers current editable product source before repeated-read and navigation bookkeeping. Each editable product path has a coverage line showing which source ranges are in view and whether the source changed since its last read when that comparison is available. Files of at most 60 lines and 2 KiB can be shown whole when the request budget allows. Very large repositories still get selected windows, not every whole file. Bookmarks use a current failing check to keep its test code and related product source in view, but need a usable source location. Setup and collection failures, and failures without a `file:line`, do not activate them. Set `STATEFRAME_CODE_BOOKMARKS=off` to restore 0.2.1 behavior byte for byte.

Measured so far on one multi-file task that had stopped on every previously observed build: all five current live runs completed and were submitted, with all 14 tests passing (three on an earlier build of this release, two on the release code). This is an early result from five runs, not a general success rate. In an offline rebuild of one recorded request, the frame showed 11 of 11 product files whole, up from 1 of 11; complete request size fell from 85,664 to 77,017 bytes, below the 82,026-byte ceiling.

A completed batch made up entirely of unchanged reads costs one no-progress watch unit, rather than one per file. Mixed or unproved batches keep per-action accounting. If the model continues to repeat unchanged reads, the run still ends with `STALLED` and preserves its partial changes for inspection.

The optional `STATEFRAME_READ_LOCKOUT=on` restriction is off by default. When enabled, it can hide source-reading routes for up to two requests after unchanged reads, while a current failing check, editable bookmarked source, and an edit and check route are available. If the no-progress watch is one step from its limit, it can stop the run before the restriction takes effect.

## Memory on long runs

This release moves run history out of memory (see "Disk use for run history"). Measured on a release-candidate build in short runs: the engine's memory footprint held at about 1.7 GB from request 10 to 60 (1,723 to 1,782 MB), growing about 0.8 to 1.1 MB per request between requests 20 and 40. The 1.7 GB is a fixed base, mostly repository analysis, and grows with repository size. Runs past about 60 requests on this build are still being measured. Measured on the previous engine (before this change), on one long task: memory grew about 12 to 30 MB per model request, about 11 GB at 240 requests. Measure footprint (Activity Monitor "Memory" or `top -stats mem`), not `ps` rss, which leaves out compressed memory.

## Engine time per turn on long runs

The prompt stays the same size every turn, and on this release's build the engine's own time per turn stays about flat on runs up to about 60 requests. Older engines grew on long runs. Measured on an older engine, in one live long run: median engine time per request went from 8.0 s early in the run to 19.3 s by request 200. On a 55-turn replay of the build this release is cut from (same engine code, comments cleaned), median engine time per turn went from 15.2 s to 16.9 s, and on a 60-request live run the engine's own time between responses did not trend up (about -4 ms per request). Runs past about 60 turns on this build are still being measured.

## Disk use for run history

The engine keeps each run's request history in spool files on disk instead of in memory. Once the run journal is open they sit in the run directory (default `~/.stateframe/runs/<repo>-<time>`, or the folder you pass with `--run-dir`); before that, briefly, in a private `sf-history-run-*` folder under `/private/tmp` (or `/tmp`). It grows with the run; see Requirements in README.md for free disk. The files are opened and then unlinked, so `ls` and `du` do not show them while the run is open; `df` does. They are released when the run closes. A run that is killed can leave an empty `sf-history-run-*` folder behind, which is safe to delete. Checkpoint copies of the history are kept in `<run-dir>/checkpoints/journal-cache` until you remove the run directory. Uninstalling StateFrame does not remove run directories; delete them yourself when you no longer need to inspect or resume them.

This behavior is always on in this release and has no setting.

Measured on a large repository without checkpoints: a run directory uses about 3 MB per request, about 550-750 MB for a 209-request run. The largest parts are the run's journal and history (about 200 MB, needed by resume and replay), saved candidate snapshots (about 130 MB), copies of the repository (about 30-75 MB each) and the recorded requests (about 23 MB). Checkpoints come on top while the run is live (see "Resuming a stopped run"; off by default).

## Identical runs and request bytes

The engine normalizes volatile test and tool output (test-runner durations, memory addresses in tracebacks, run-directory paths) before the model sees it; the raw output stays in the run journal. On the measured 40-request workload, two runs on the same input sent byte-identical requests on 40 of 40 requests, apart from three clock fields. In live runs the ID the engine gives each applied edit (its edit receipt) is not yet deterministic, so requests after an edit can differ in that one field. Tool output the engine does not recognize as volatile can still differ between runs. This is not full determinism. The model's own output is not deterministic, so a live run can take a different path. On purpose, the engine still shows the model some real clock readings: the time in the environment panel, how long a job has run, and when a navigation handle expires. A few rarely used paths still use random ids, the ones that continue an interrupted intention, fork work, or run several attempts side by side, so runs that take those paths can differ in request bytes.

## Prompt cache reuse

Prompt cache reuse is modest. The engine rebuilds a fixed-size frame for every request instead of appending to a history, so the server can reuse less of the previous request than a chat-history agent does. Measured: about 25% of prompt tokens reused on a real multi-step task (about 94% when a prompt repeats exactly). Each request therefore pays to process most of its prompt again. A cache-friendly frame layout is in the code but not active on the default path in this release; it did not improve reuse enough in testing.

## Token counting

- Without `LLAMA_CPP_TOKENIZE_URL`, tokens are counted on the model server. If that server is remote, each count is a network round trip and adds noticeable time per turn. Co-locate a tokenizer (QUICKSTART.md).
- The engine counts tokens through a `/tokenize` endpoint (llama.cpp servers have one; other servers are untested; set `LLAMA_CPP_TOKENIZE_URL` to use a separate llama.cpp tokenizer). With a server that has none (for example Ollama or hosted APIs) and no separate tokenizer, the engine falls back to a conservative estimate, and a run can then stop before its first request because even the minimum frame looks too big (`ERR:FrameFit5ImpossibleMinimum`).
- If the tokenizer stops answering mid-run, the engine falls back to an estimate that can undercount, so a request can exceed the server's context slot and be rejected. Keep the tokenizer up for the whole run.

## Endpoint locality

Remote endpoints are rejected by default. An explicit override can enable them and warns that source code will leave the machine. A loopback endpoint may also forward elsewhere, so local-only endpoint checks cannot verify the server's upstream route.

## Internal identifiers

Internal identifiers keep development names (profile keys, module filenames, schema IDs).

## Editing files during a run

If files in the working copy change while a request is being prepared (for example, you edit a file mid-run), the engine refuses that request rather than send stale content. Avoid editing the working copy during a run.

## Slow or shared hardware

Each model request has a fixed 600-second timeout. If the model server is slow or shares its GPU with other work, a long answer can take more than 600 seconds. The request is then retried, and the server may finish the abandoned attempt anyway, using GPU time while the retry runs. On slow hardware, give the model server the GPU to itself during a run.

## Time and step limits

There is no time limit by default: a run continues until the engine finishes, and you can stop it any time with Ctrl-C. Pass `--timeout <seconds>` to set one. Each model request still has its own timeout, so a hung model server cannot hang the run.

There is no step limit by default either. Pass `--max-turns N` to stop after N model turns.

## Resuming a stopped run

- On this release resume replays the whole recorded run, and checkpoints are off by default: in our tests a checkpoint restore did not rebuild the next request faithfully (resume refused it and replayed, which gave the correct result). The replay takes time in proportion to the run's length.
- Catch-up re-runs reproducible local tool work and checks, but serves recorded observations for Git, web and jobs. It takes time in proportion to the run's length, though it takes no model time. Shell-enabled resumes rebuild the run environment and replay from the beginning.
- A rebuilt request that differs from its record may write `resume-divergence.json` and continue live. A tool-tape mismatch, incomplete operation or external operation without `--replay-external` instead refuses resume. On an earlier preview engine we saw request divergence from hash-seed-dependent ordering (fixed in this release) and from journal event-id drift (fixed in later engines).
- Resume needs the same StateFrame version that started the run, and the same model name (or `--model` with the name the server now lists). It refuses a run made by another version, a run that already submitted its result, and a run whose records were changed or damaged.
- Resume keeps the run's original `--max-turns`: passing a different value is refused, except for a run that stopped at its step limit.
- A run that stopped at its step limit resumes only with a larger `--max-turns`, and then restarts live from request 1: the step limit is part of the first request, so neither the checkpoint nor the recorded requests can be reused. `stateframe resume` prints a WARNING when this happens.
- A chat task stopped with Ctrl-C can be resumed with `stateframe resume <its run directory>`; the chat session itself does not pick the result up automatically.
- If you turn checkpoints on: checkpoint resume is used only when you resume with the same `--max-turns`, endpoint and model name. Changing the endpoint or `--model` (for a renamed server model) resumes by replay; changing `--max-turns` restarts the run live from request 1 (see above).
- Checkpoint disk use has a fixed base per run that grows with repository size: on a large test repository each checkpoint held about 430-460 MB of engine state (mostly repository-analysis structures) plus about 0.13 MB per request, so with the two newest kept a run used about 1.1 GB, briefly about 1.7 GB while a new checkpoint is written before the oldest is removed. Checkpoints are off by default; leave them off if disk is tight (resume then replays the whole run). When a run finishes for good (its result was submitted, or it stopped on an error that would only repeat, so resume would refuse it), StateFrame removes its checkpoints and prints how much it freed; `--keep-checkpoints` (or `STATEFRAME_KEEP_CHECKPOINTS=1`) keeps them. Stopped, cancelled, killed and crashed runs always keep theirs. An earlier engine's figure of about 1.1 MB per request was only the per-request increment, not the total. `du` reports more because it counts the copy-on-write file clones each checkpoint keeps; real use is lower. Each extra kept checkpoint (`STATEFRAME_CHECKPOINT_KEEP`) adds one engine-state file, measured at 45 MB at 130 requests and 79 MB at 286.
- Checkpoints are tied to the engine build and Python version that wrote them. After an upgrade, an old run resumes by replay.
- Loading a checkpoint runs code that was saved with it. Only checkpoints signed with your own install key, owned by you and not writable by others are loaded. Treat a run directory like source code: do not resume one you got from someone else.
- After a checkpoint resume, the time the run spent stopped does not count against an engine time budget; the engine's clock continues from the checkpoint.
- Checkpoint saving needs APFS clones or a POSIX fork; where neither works it saves more slowly or not at all, and resume replays.

## License

StateFrame is licensed under the GNU Affero General Public License v3.0 only (AGPL-3.0-only), or under a separate commercial license from Farside Research LLC (see COMMERCIAL-LICENSE.md). Third-party dependencies keep their own licenses; see THIRD_PARTY_LICENSES.md.
