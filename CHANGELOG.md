# Changelog

## Unreleased

- Under `--protect-tests task/all`, creating files that change test collection (`conftest.py`, test-package `__init__.py`, pytest config, `sitecustomize.py`/`.pth`) is now refused; before, a new `tests/__init__.py` could import product code at collection time and change the test result.
- Fixed: the queue-wait warning now reads sealed run journals (`events.jsonl.xz`), so finished runs report provider queue waits again; the reader also satisfies the journal shim guard.
- Tests: the engine-child lookup in the resume acceptance tests asks `ps` for untruncated arguments (`ps -ww`). On macOS, `ps` cut the command line to the terminal width under captured pytest output, so the `_child.py` suffix disappeared and `test_acceptance_widened_run_resumes_by_replay` failed on an untouched checkout. Test-only; no product change.
- Tests: `test_routes_cli`'s resume test installed the wire recorder without monkeypatching, so later tests in the same process inherited it. It now restores it, `HOST_STATE` is isolated in five more test files, an autouse fixture names any test that leaves an installer attribute behind, and a forced-order subprocess test reproduces the original leak. Test-only; no product change.
- Tests: a guard requires every Linux bwrap launch-argv builder to give the command its own PID namespace (`--unshare-pid` or `--unshare-all`), so an orphaned descendant always dies with its namespace instead of escaping tracking. Builders are found mechanically; a reviewed exemption covers the release runner's write guard. Test-only; no product change.
- Tests: helpers that start subprocesses without inherited `STATEFRAME_*` switches now give them a temporary `STATEFRAME_HOME`, so test runs no longer pile env seeds into the real `~/.stateframe`; a guard keeps every such helper honest.
- Tests: the tokenizer end-to-end test reads sealed measurement journals (`events.jsonl.xz`), so it checks real tokenizer measurements again instead of finding none.

## 0.3.29 (2026-11-03)

- Finished runs take far less disk. A fixed scripted run with a run environment went from 35.8 MB to 0.9 MB, and the bundled examples shrink about 3x before any environment is counted. Measurement journals are sealed as `events.jsonl.xz`, their blobs go into one append-only pack, preflight `requests/` records go into `requests.pack.xz`, and checkpoints and the run environment are removed once the result cannot be resumed. Interrupted, killed and stopped runs keep everything resume needs. `wire/` and `spec.json` are never touched, and request bytes are unchanged.
- Changed: the layout of a finished run directory. Scripts that read its files directly need updating. Once a run cannot be resumed, `env/` is removed and `env-manifest.json` records what it held; `worker/measurement-runs/<id>/events.jsonl` becomes `events.jsonl.xz`; loose `blobs/*.zlib` become `blobs/pack-0001.sfpk` with an index; `requests/*.json` become `requests.pack.xz`. Pass `--keep-run-env` (or set `STATEFRAME_KEEP_RUN_ENV=on`) to keep `env/`, and `--keep-checkpoints` to keep checkpoints. `stateframe inspect` reads both layouts, and so does `stateframe rehydrate`, which rebuilds `env/`. Interrupted, killed and stopped runs keep the old loose files until they finish.
- Run environments are cloned from a shared read-only seed under `~/.stateframe/env-seeds/` (copy-on-write where the filesystem allows it, hard links for package files on ext4). `STATEFRAME_ENV_SEED=copy` restores plain copies.
- New `stateframe rehydrate <run>` rebuilds a removed environment from its seed or from `env-manifest.json`, says loudly when the result is not exact, and lists every difference. `--keep-run-env` or `STATEFRAME_KEEP_RUN_ENV=on` keeps the environment instead.
- New `stateframe gc` previews and, with `--apply`, reclaims storage in older runs. It reads both layouts and never changes unfinished runs unless asked. It prunes shared seeds only when it scans the default runs home, or with `--prune-seeds`.
- Fixed: a run environment refused while it was being prepared stayed on disk (about 33 MB per run) although the run went on without it.
- Fixed: finished-run checkpoints could stay on disk. The engine leaves candidate workspaces inside checkpoints read-only, so removing them failed with EACCES (permission denied), and the error was only printed as a warning. Checkpoint and environment removal (at finish and in `stateframe gc`) now makes directories writable first. File modes are never changed, so files hard-linked into a shared seed keep theirs.
- Fixed: the `LATEST` checkpoint pointer could name a checkpoint directory that was not complete yet. Checkpoint pooling now also works on Linux.
- Run directories started on an older release cannot be resumed after upgrading, because resume needs the engine release that started the run. `stateframe resume` refuses them with: "cannot resume: this run was made by engine release <old>, the installed engine is <new>; resume replays the run on the same engine, so it needs the release that started it". Resume such a run with the release that started it. Reading still works on them: `stateframe inspect`, request records and measurement journals read them unchanged, and `stateframe gc` reclaims their space without changing what reads back. `stateframe rehydrate` refuses them, with "<run> has no env-manifest.json; it still has its environment in env/, so there is nothing to rehydrate" while the old environment is still there, or "<run> has no env-manifest.json, so its environment cannot be rebuilt; runs made before StateFrame recorded one have no manifest" once it is gone.
- See [Run storage](docs/guide/storage.md) for formats, switches and recovery.

## 0.3.28 (2026-11-03)

- Runs no longer stop mid-task with `FrameRequestAllocationError` (`closure_units_undelivered`, with or without `final_wire_proof_missing` or `closure_omissions_unreconciled`) when the last fit before sending drops a selected source page. StateFrame now defers that page behind its read route, the same way it already did earlier in allocation, and proves the rebuilt request again. Deferral is bounded to one per selected file; other proof failures still refuse.
- On a 22-file repository, v0.3.10 refused mid-run in 3 of 3 live runs; with this change, 0 of 3 refused.
- New test guard: every recorded allocation refusal we have (34 runs) is replayed without a model and must finish. A deliberately broken deferral makes it fail.
- When nothing is dropped, request bytes are unchanged.

## 0.3.27 (2026-11-03)

- A model request waiting for a busy server slot now stays alive past its per-request timeout when `/slots` or `/health` returns any HTTP status, including 401, 404, 501, and 503. The extra wait ends at the run deadline or `STATEFRAME_QUEUE_WAIT_MAX_S` (1800 seconds by default). A response that has started still has its per-request deadline. Each extended send records `provider_queue_wait`, and the run prints one queue warning.
- Request-body uploads now have a progress watchdog (`STATEFRAME_SEND_STALL_S`, 90 seconds by default). A stalled upload closes its connection and enters the ordinary frozen-byte transport retry policy, with one `send_stall` journal event per failed attempt. Healthy probes extend queue waits only after the entire body was sent. The sending process decides whether the body finished before the per-request deadline, at its last socket write, and the caller waits for that verdict within the unchanged hard deadline, so a busy machine no longer fails a request whose upload finished in time.
- In-flight probes use the request's selected endpoint and API key, never follow redirects, and are bounded by the request's hard deadline. Post-failure health recovery uses that same destination, including for routed requests. Model requests without a configured key send the same headers as before.
- Measured in one contended live run against a shared server: requests rode out 28 queue waits, the longest 193 seconds, with no transport errors, where the previous release failed all 4 requests in about 80 seconds without a response. A request whose upload stalled was cut off after 15 seconds of no progress, and its retry on a new connection succeeded 18 seconds after the stall began. The previous release stayed blocked on that write for the full 40-second test window.
- A profile that sets `STATEFRAME_SEND_STALL_S` or `STATEFRAME_QUEUE_WAIT_MAX_S` is refused by earlier versions ("unconsumed STATEFRAME settings").
- Health probes remain separate from model sends and do not change the request body, wire tape, or replay records.

## 0.3.26 (2026-11-03)

- Add powerups: an installable package of a skill, command tools, lifecycle hooks and an optional MCP definition, selected once and kept. `stateframe powerups install|enable|disable|list|show|remove|trust|promote` manage them; `--powerup NAME[=MODE]|PATH`, `--powerup-set NAME.KEY=VALUE` and `--no-powerups` adjust one `run` or `chat`. Each powerup is off, offered (only its skill catalog line is shown), on, or auto (on when its file and task rules match), per user, profile, trusted project or command. Selected parts enter the existing skill, tool, hook and MCP loaders with the same limits as hand-written ones, so with `--shell-sandbox off` powerup command tools run on the host like any shell tool; user `--tools` and `--mcp` take the tool schema allowance first.
- A broken, changed or over-budget powerup is skipped with a one-line notice and recorded in `spec.json`; the run continues. The spec pins each powerup's digest, and resume refuses a changed one with `POWERUP_CHANGED`. An unreadable powerup state file turns every powerup off for that run. Powerup digests now use a new format, so after upgrading every enabled or trusted powerup is skipped until it is reviewed and trusted again. An enabled powerup prints `powerup NAME disabled: re-trust required: pinned with an older digest format; run 'stateframe powerups enable NAME' after review`; a trusted project powerup prints `powerup NAME disabled: re-trust required: project powerup changed or pinned with an older digest format; run 'stateframe powerups trust --repo REPO' after review`. Resuming a run recorded before the upgrade stops with `POWERUP_CHANGED: NAME: pinned with an older digest format; re-trust required`. Installed powerups start off until enabled.
- Three working examples ship off by default in `examples/powerups/`: `py-format` (ruff format after edits), `test-gate` (a configured check before submission) and `repo-grep` (a bounded read-only search command). When ruff is missing, `py-format` does nothing and prints one notice; when its check program is missing, `test-gate` refuses submission. With no powerup selected, request bytes and the run spec are unchanged.

## 0.3.25 (2026-11-03)

- Shell-enabled runs now resume from a checkpoint instead of replaying from the start. Two things kept this from working. Every save after a shell step failed with `cannot pickle 'socket' object`, and three failures in a row turned saving off. Resume also never offered a checkpoint to a run with the shell on. The open execution session is now saved without its supervisor process, control socket and log, and a restored run starts a fresh session the next time it runs a command. A checkpoint already copies the run environment and shell scratch state, and none is taken while a background job is running. On a scripted run killed at request 3, resume took 13-14 s from the checkpoint against 18 s by full replay; the gap grows with the run's length. Run vitals now report `checkpoint_failures` and `checkpoint_saving_disabled`.

## 0.3.24 (2026-11-03)

- Token counting now works against a llama.cpp server started with `--api-key`. StateFrame called `/tokenize` without the key, the server refused it, and every request was sized by an 82 KB byte estimate instead of real token counts. That estimate is tighter than the token budget, so source the model had just read could be left out of the next request. The key is now sent to `/tokenize` when it is on the same origin as the model endpoint, and never to a different host or across a redirect.
- When token counting does fall back, the `token_counting_unavailable` event now says why: `unauthorized`, `http_error`, `connection`, `timeout`, `bad_response`, `no_url` or `disabled`.
- Against a server without a key, request bytes are unchanged.

## 0.3.23 (2026-11-03)

- A shell command that only creates directories (for example `mkdir -p pkg/batch`) no longer leaves its effect unresolved. Directories are not recorded as file changes, so the effect stayed `unknown` and every later tool call except `done` was refused with "no typed repair action on the live menu", a dead end present since 0.3.0. The effect now settles as applied from its journaled shell receipt when it belongs to the same run and effect and every recorded path is a valid workspace-relative path. As before, the exit code does not decide this.
- A new test (`tests/test_refusal_names_live_action.py`) checks that every recorded refusal names an action offered on the next request's tool menu. Fences that can still leave no such action are listed by name in `KNOWN_DEAD_END_FENCES`.

## 0.3.22 (2026-11-03)

- Fix (macOS, regression in 0.3.15-0.3.21): background jobs and execution sessions whose process started a child in a new session (setsid) ended the run with ENGINE_ERROR "descendants survived SIGKILL; signal permission denied", because cleanup never signalled that detached child. Cleanup now stops it with a signal the kernel delivers only if the process's PID generation still matches, so a reused PID is never signalled, and the cleanup error names the real cause instead of always saying "signal permission denied". If that kernel signal is unavailable, cleanup re-checks the process identity immediately before a PID signal and labels that path. Tests in `test_launch_unit_kill.py` that asserted the setsid child survives on macOS encoded this bug and now assert no survivor on every platform. Cleanup also keeps adopting members of the launch process group after the leader exits (members born before the leader was reaped), so a fork burst whose parents die between scans no longer leaves an untracked process holding the output pipe ("output pipe held by untracked descendant").

## 0.3.21 (2026-11-03)

- Stage 2 fixes `chains.tree_hash` (NUL and lone-surrogate collisions), `behavior_bridge._files_hash`, `kernel_state_owner_inventory._source_revision_vector`, `swebench_env._dependency_manifest_digest`, the expert-gap closure freshness check, `kernel_remaining_sites.current_workspace_revision`, `long_horizon_protocol._staged_digest`, and both workspace-truth fact revisions. The digest guard now has an empty allowlist and detects NUL-tagged ticket payloads.
- All normal-input request bytes remain unchanged, including multi-file owner inventory, remaining-sites, chain, workspace-truth and staged-task values. NUL or surrogate inputs use typed digests. Cache keys and closure freshness now also compare ordered dependency paths, so the compatible no-terminator revision residual cannot reuse a fact or plan for a different path list.
- Persisted closure plans carry `dependency_digest_format=stateframe.dependency_revision/2` and host-only `dependency_paths`; old plans rebuild. SWE-bench manifest digests carry `stateframe.swebench_dependencies/2` and old ready/failed markers or repair contracts are refused. Staged task records carry `task_digest_format=stateframe.staged_task/2` and old records fail validation.
- The replay acceptance helper requests untruncated `ps` command lines. On this Linux host the default width hid `_child.py`, so its child discovery failed even though the engine had reached the paused request.

## 0.3.20 (2026-11-03)

- Add `STATEFRAME_FINAL_CACHE_LAYOUT` (opt-in, off by default): each request places the typed state frame after the stable carrier rows and moves current authority into the final message, so a local server can reuse more of the previous prompt. Measured live on Qwen3.6-35B-A3B with llama.cpp over 20 runs, the server reused 34% more cached prompt tokens per request from request 2 on (6,164 against 4,595), and the cached share rose from 20.9% to 25.7%. A request that would not fit the prompt envelope in this layout is sent in the normal layout instead of stopping the run. It cannot be combined with the page log (`STATEFRAME_PAGE_LOG` in a custom profile). See [Running](docs/guide/running.md#prompt-cache-current-authority-last).

## 0.3.19 (2026-11-03)

- Three tree digests that could give two different trees the same value in normal use are fixed, and three more no longer confuse a missing or unreadable file with real content. One shared helper, `kernel_tree_digest.py`, records each path with an explicit type (regular file, missing, unreadable, symlink with its target, directory) instead of concatenating paths and bytes or hashing a sentinel string such as `<missing>`. Each site hashes under its own versioned domain. Fully fixed: the get-sure workspace fingerprint (`a.py`="x" matched `a.pyx`=""), the ship-tree smoke digest (same pair, and a missing file matched a file containing `<missing>`), and the realtask snapshot digest (a symlink `x -> y` matched a file `x` containing `L:y`). Missing-file case fixed: `_revision12` (missing matched `<missing>`), `dependency_revision` and the source-closure revision (missing matched `<unavailable>`), and the coordinated-change tree revision (missing matched an empty file).
- Request bytes are unchanged. `_revision12`, `dependency_revision`, the source-closure revision and the coordinated-change revision reach model-visible text (fact block headers, `dependency_revision=` rows, batch tool observations), so they keep the 0.3.18 value whenever every listed path is a readable regular file, and only a missing or unreadable file now hashes under its own typed encoding. That 0.3.18 encoding has no terminator after each file's content, so two different path lists can still give the same value (for example `a`="bc", `d`="e" and `a`="b", `cd`="e"); see KNOWN_LIMITATIONS.md.
- Persisted formats: realtask task rows, snapshot metadata and controller records now carry `digest_format=stateframe.realtask_tree/2`, and a row or snapshot made by 0.3.18 is refused with a rebuild message instead of being compared. Ship-tree smoke results and baselines carry a format marker, so a result from an older run is recomputed, not reused. The get-sure fingerprint is only compared within one call.
- A new test guard scans the engine for hashes fed unframed path and content bytes and fails on any new one; the remaining sites are pinned by name until stage 2 (see KNOWN_LIMITATIONS.md). The guard cannot see the compatible encoding's remaining case above; a dedicated test pins it instead.

## 0.3.18 (2026-11-03)

- The dependency-license guard's planted-import check writes its probe into a temporary package instead of the shipped `src/stateframe` tree, so a parallel worker (`pytest -n`) scanning the shipped package no longer sees the probe and fails. A new test runs the planted and shipped scans concurrently.

## 0.3.17 (2026-11-03)

- Release-comparison tests run each compared tree under its own interpreter and check which `stateframe` package it imports before comparing bytes. The engine's child process drops `PYTHONPATH`, so a shared interpreter could import an installed copy instead of the tree under test.
- `test_hook_wiring_parity` and the edit-scope parity test pin the stable tool head off on both sides; since 0.3.13 the base release defaults it on, and only the candidate side had been pinned.
- Tests that compare request bytes with an older release must now state every switch whose default has changed since then. `tests/tools/changed_defaults.py` lists those switches (so far the stable tool head, from 0.3.13), and `tests/test_cross_version_pins.py` fails if a comparison helper is called without one. A planted omission is checked to fail.
- The request comparison tool (`tests/tools/compare_no_mcp_requests.py`) now compares with the same 0.2.3 tree and settings as its test, so its byte-identical result holds from the command line; `--current-defaults` shows how today's defaults differ instead.

## 0.3.16 (2026-11-03)

- Two slow acceptance tests (`test_e19_volatile_shell_observation_is_byte_identical_on_resume` and `test_no_skills_wire_matches_023`) no longer time out on loaded test hosts. Their subprocess timeouts are now 1.5 times the slowest wall time measured on the slowest gate host under heavy load, recorded next to the measurement in `tests/acceptance/_support.py`, instead of a fixed 240 seconds. A guard test fails if either test goes back to a bare timeout or a measurement goes unused.
- `test_existing_discount_requests_match_untouched_0_2_8` compares the 0.2.8 request bytes with the stable tool head switched off again; since 0.3.13 turned it on by default, the test had compared against the new default and failed. The shared byte-capture helper now requires every caller to state the stable tool head setting, so a default change cannot slip into a byte comparison unnoticed.
- The launcher test for the process-count limit (`test_launcher_session_umask_limits_and_fds`) no longer fails when other processes start or stop during the test, or when the inherited soft limit is lower than the ceiling the launcher asks for.
- This release changes tests only; the installed engine is identical to 0.3.15.

## 0.3.15 (2026-11-03)

- Process-group cleanup now uses the leader identity captured when the process starts. A reused PID cannot authorize a later group signal, including when cleanup runs without a descendant monitor. Request bytes are unchanged.
- Outer checks now clean their own check group and acknowledge cleanup. Run, hook and check cleanup stop surviving group members before reaping an exited leader; Linux retains the launch identity when pidfds are unavailable.
- Check, job, environment and release cleanup keep the launcher unreaped until group escalation finishes. Job readers cannot reap the leader during requested cleanup. A missing launch record or descendant family reports unverified containment, and a forced kill is recorded as a timeout. Outer checks acknowledge process cleanup before removing temporary files. On macOS, detached descendants are reported as survivors when they cannot be signalled safely.

## 0.3.14 (2026-11-03)

- The stable tool head now lists the run's full set of offerable tools from the first request, so the tool list no longer grows on request 2 and the server can reuse the cached tools, system prompt and task from request 2 on. The current step's callable menu still controls what can run, and a tool first seen later is still appended. Measured live on Qwen3.6-35B-A3B with llama.cpp, one run of each bundled example: request 2 hit the cache in every run, and the cached share of prompt tokens from request 2 on rose to 37-47%, from 18-22% without the seed. The seed lists five tools earlier than before (`authored_check`, `decide`, `insert_before`, `predict`, `work`), which makes the cached head about 680 tokens larger on every request. In 15 sampled first requests with those five listed, the model called none of them, the same as without them.

## 0.3.13 (2026-11-03)

- The stable tool head is now on by default: each request lists the tools offered so far in the run, in first-seen order, so the server can reuse the cached tools, system prompt and task. The current step's callable menu still controls what can run. Set `STATEFRAME_STABLE_TOOL_HEAD=off` in the shell or a custom profile to restore per-step tool lists and the previous request bytes.
- Measured on Qwen3.6-35B-A3B with llama.cpp over 30 runs (three bundled examples and two harder multi-file tasks, three runs each way): uncached prompt tokens per run fell 25% and wall time per run 15%, every run with the stable tool head submitted, and hidden-test scores were no worse. See [Running](docs/guide/running.md#prompt-cache-stable-tool-head) for the per-task table.

## 0.3.12 (2026-11-03)

- Add `STATEFRAME_REQUIREMENT_RECHECK` (opt-in, off by default): the first time the model says it is done, StateFrame answers once with the task's own sentences as a numbered checklist and asks the model to check each one before finishing. It never asks with fewer than 2 steps left or when the task has fewer than 2 sentences, and it makes no extra model call. On one harder multi-file task it raised the hidden-test score from 2 of 9 to 8 of 9 in 2 of 3 runs; on a task every arm already passed it roughly doubled the requests. With the switch off, request bytes are unchanged.
- Correct the project-instructions note in KNOWN_LIMITATIONS: with more runs, the process-guidance `AGENTS.md` showed no reliable gain.

## 0.3.11 (2026-11-02)

- Scoped runs now refuse edits to existing product files outside their editable set before changing the workspace. The refusal names the file and how to widen the scope; new product files remain allowed under the existing rules.
- The host warns and marks a run unsuccessful if a sealed candidate contains changed existing files it cannot apply. Shell writes restored for being outside scope now say REFUSED in the model reply.
- Checkpoints now remain available when StateFrame starts under a permissive umask. Its own state directories and checkpoint key are created with private permissions; existing owned directories and keys with loose permissions are tightened safely.
- A run whose checkpoints are disabled now prints one warning on stderr with the reason. Normal runs keep the same request bytes and output.

## 0.3.10 (2026-11-02)

- New opt-in `--edit-scope widen` (or `STATEFRAME_EDIT_SCOPE_WIDEN=on`): edit and file-operation tools can admit up to 24 existing product files beyond a scoped run's starting set, in action order and across repository directories. Tests, named checks, hook inputs, links and special files remain refused. Each admission records its pre-edit digest in the durable journal; the host requires and rechecks it before carrying writes, deletions and moves into the result. New-file roots stay fixed; shell cannot initiate widening.
- Before this, a `str_replace` on such a file could appear to apply and then be left out of the result; with widening on it is kept.
- Runs with widening on resume by replay rather than from a checkpoint. With the option off, request bytes are unchanged.

## 0.3.9 (2026-11-02)

- Add `STATEFRAME_GIT_COMMIT` (opt-in, off by default): the agent can call `git_commit(message)` and `git_branch(name)` to split its work into reviewable commits. Commits are snapshots in the run folder; your repository and the working copy are never touched, and secret paths are never included. After the run, each branch is written as a `git format-patch` series under `<run-dir>/commits/<branch>/`, ready for `git am`. Commits are deterministic, an unchanged commit is refused, and resume serves recorded commits instead of committing again.
- With the switch off, the tools are not offered and request bytes are unchanged.

## 0.3.8 (2026-11-02)

- Add `STATEFRAME_SHELL_STATE` (opt-in, off by default): the shell keeps its working directory and exported variables between foreground calls, so `cd sub && export X=1` followed by `pwd; echo $X` prints `sub` and `1`. A directory outside the workspace or one that no longer exists falls back to the workspace root, shell-maintained and `STATEFRAME_*` variables are never carried, and a timed-out call keeps the previous state. The state is part of each call's replay record, so resume reproduces it and refuses a call started from a different state. Background jobs and custom or MCP tools are unchanged.
- With the switch off, shell commands, request bytes and the tool tape are unchanged.

## 0.3.7 (2026-11-02)

- A repository with more than 200 source files no longer needs `--target`/`--edit` when the task or its chosen checks point at the files. StateFrame chooses up to 12 editable files: those the task names, then the modules imported by the checks you chose (`--check`, or test files named in `--test-cmd`), then one more import hop. It prints the list at startup. With nothing to go on, or with more than 8 selected test files, it stops and asks for the flags as before. `--target` and `--edit` still decide when given.
- The scope is fixed for the run; a file outside it stays read-only context. If the choice misses the file the fix needs, rerun with `--edit`.
- Runs that started before this release build the same task and send the same request bytes.

## 0.3.6 (2026-11-02)

- MCP servers on a busy Linux host no longer fail discovery with "cannot identify private MCP PID namespace for supervision". After `initialize`, StateFrame waited at most 0.2 s for its resource monitor to find the server's private PID namespace, and one monitor sample walks the whole host process table. The wait is now 2 s, scaled up to 16 s when the host's load exceeds its CPU count, and never past the call's deadline. It still ends at once if the monitor fails or the server exits.
- Request bytes are unchanged.

## 0.3.5 (2026-11-02)

- Fix runs stopping mid-run with `FrameRequestAllocationError ... failures=source_delivery_partial` after the agent re-read a file that an earlier, fuller request had deferred. The old deferral notice stays in the request's stable prefix, so the source proof still saw a declared deferral, skipped the file because it was now complete, found nothing left to defer and returned before marking the delivery complete. The request was refused with no actual source missing. A delivery whose deferred files are all complete and verified on the final wire is now marked complete, the same way a delivery without a deferral notice already was.
- Measured on a recorded failing run replayed through a scripted endpoint at the default budget: 0.3.4 refused the 7th request; this release sent all 37 and finished. Runs that never hit the case send the same requests byte for byte.

## 0.3.4 (2026-11-02)

- A Linux sandbox no longer drops a read-only mount that crosses a host mount boundary. Since 0.2.12 a read-only path beneath a read-only-mounted parent is not mounted again; a recursive bind can omit unbindable submounts, so the child now keeps its own mount when the host has a mount point between the two. Paths under the sandbox's own `/dev` and `/proc` keep theirs too, and if the host mount table cannot be read, nothing is pruned.
- The test interpreter is no longer probed a second time when it is the exact path the engine interpreter probe already proved imports pytest. Any other path, including a symlink to the same binary, is still probed.
- Scanning PATH directories for executables takes one `lstat` per entry plus a `stat` only for links, and admits exactly what it did before: a regular file after following links that the user can execute. Unreadable, vanished or dangling entries are skipped.
- Stopping a sandboxed command no longer risks signalling an unrelated process group. The group kill now runs only when the launcher is still the same live process (same start time, not a zombie) and still leads its group; if its PID was reused, the group is left alone.
- Request bytes are unchanged.

## 0.3.1 (2026-11-02)

- Fix runs on mid-size repositories failing before their first request with `FrameRequestAllocationError ... failures=closure_omissions_unreconciled`. Above about 12 candidate source files, the bounded source closure stops scanning and records the next file as "closure file limit reached". The final source proof counted that unopened file as undelivered source and refused the whole run. A file the walk never reached is now a scan-scope fact: the run proceeds, and the file stays listed in the frame's file inventory and repository map, where the agent can read it. The journal keeps the omission with its continuation handle. An omission on a path whose units were selected for delivery still refuses the request, as do unavailable and out-of-scope sources.
- Requests on repositories under the closure limit are unchanged byte for byte.
- Known limitation: on such repositories a longer run can still stop mid-run with `closure_units_undelivered` or `source_delivery_partial` once earlier read pages no longer fit. Measured live on a 22-file repository: 0.2.8 and 0.3.0 sent 0 requests; this release sent between 6 and 25 in four runs before stopping. See KNOWN_LIMITATIONS.md.

## 0.3.0 (2026-11-02)

- Checkpoint saves now check free space before capture and skip low-disk saves without pausing the engine. The automatic floor is at least 1 GiB and three times the latest state size; `STATEFRAME_CHECKPOINT_MIN_FREE_BYTES` overrides it, including `0` to disable the check. Saving resumes after space recovers, and skip events stay out of model requests.
- Mid-request resume now verifies the in-flight request byte for byte on both full replay and checkpoint restore. Tokenizer fallback telemetry no longer advances the journal sequence, which could shift check IDs and edit receipts when tokenizer availability changed between attempts. An in-flight mismatch now stops before a live send.
- Checkpoint restore now preserves the resumed attempt's replay control, complete recorded tail, and check timing. The token meter had tracked journal subscriptions by `id()`, which changes on restore; duplicate subscriptions shifted edit receipt IDs. Tracking journal objects preserves the IDs, and kill-and-resume checks verified byte-identical next requests.
- On Python 3.11 and 3.12, checkpoint restore did not reach the running engine. Before Python 3.13 a frame's locals are a snapshot, so the saved values went into a copy and the engine continued with the new attempt's setup objects; on Linux this stopped the run with "workspace ownership changed". Restore now writes the saved values into the suspended frame, and the workspace lease is re-derived from the original run folder and locked again.
- Checkpoints are saved at every step by default (was every 15 steps), so resume usually replays nothing. Each save pauses the engine about 0.2 s and is written in the background. At request 110 of a 120-request run, resume reached the next request in 47 s, versus 72 s with the old cadence.
- Checkpoints are on by default. Set `STATEFRAME_CHECKPOINT=off` to opt out; an unfaithful restore still falls back to full replay.
- Runs with hooks resume by replay and never restore a checkpoint. Restoring could re-run hooks that had already fired between the checkpoint and the interruption. A resume with hooks configured records `resume.checkpoint` as skipped with reason `hooks_configured`, writes a `checkpoint_skipped` event and prints one line saying it is replaying instead. Checkpoint restore applies only to `--shell off` runs.
- Runs with per-step models also resume by replay and never restore a checkpoint. The resume records `routes_configured` as the skip reason, unless hooks are configured too, and keeps the request and route checks during replay.
- Measured from a clean install on Qwen3.6-35B-A3B, killed at the second model request: without hooks, the checkpoint restored and verified, the resumed request was byte-identical, and resume reached it in 23.2 s versus 27.8 s by full replay. With hooks on all six events, resume replayed and the resumed request was byte-identical.

## 0.2.13 (2026-11-02)

- Add `STATEFRAME_STABLE_TOOL_HEAD` (opt-in, off by default): one append-only tool list per run, so the server can reuse the cached tools, system prompt and task between requests. Only the current step's tools can run; a listed but unavailable tool gets the usual not-offered reply, and a short note names the callable tools. With the switch off, request bytes are unchanged.
- Measured so far on Qwen3.6-35B-A3B (llama.cpp): about 6k cached prompt tokens per request after the step where a new tool first appears (about 20% of the prompt on multi-file tasks, up to about 40% on small ones), against 0-3% reuse without the switch. Task outcomes matched in 3 paired runs on two tasks; the effect on total run time is not established. This is a small sample.

## 0.2.12 (2026-11-02)

- Start faster: the engine ships its switch registry instead of parsing the engine source at every start, cutting process start to the first model request from about 90 s to about 45 s on a loaded Linux host. A fingerprint of the declared files guards the shipped registry; on any drift the engine falls back to the scan.
- Keep editable test files out of the starting-file hint for file-creation tasks. When a task names a new output, use a public file beside it or a declared new-file target without inventing a source file. Source-window budgeting skips non-public targets while the public-path validator stays strict.
- Sweep runnable Quickstart and README commands and every bundled example from a clean install, with a broken-release positive control and a release-routine script.
- The startup check of the model server (`/v1/models`, and `/slots` on llama.cpp) tries once more when the server takes longer than 10 seconds to answer, so a run no longer fails to start because a busy or shared server stalled one request. An HTTP error or a refused connection still stops the run at once.
- The startup check of the engine and test interpreters tries once more, with twice the time limit (60 s, then 120 s), when it times out, so a run on a heavily loaded machine no longer fails with a healthy interpreter. If both attempts time out, the message says it timed out under load; an interpreter that exits with an error is reported at once as broken or incomplete.
- Each execution supervisor request carries an id, and late replies to earlier requests are discarded. A reply that arrived after the timeout can no longer corrupt the next call ("startup probe could not run safely: Extra data"). The supervisor timeout is `STATEFRAME_EXEC_RPC_SECONDS` (default 15) scaled by host load per CPU (1 to 8).
- System executable directories are scanned once per process instead of once per capability build. This removes 10 to 30 s of startup CPU on Linux, and the resulting capability is unchanged. `STATEFRAME_STARTUP_PROBE_SECONDS` (default 60, at most 300) sets the interpreter startup probe limit.
- Sandboxed Linux commands now start in about 0.5 s instead of about 3 s. The inner sandbox no longer re-binds every executable on PATH under the already read-only /usr (7 mounts instead of about 425). The mount view and access set are unchanged, and a containment check confirms nothing escapes.
- Linux: checks no longer fail with exit 127 when the test Python is a shared-library build under `/usr/local` (the official python container images, pyenv `--enable-shared`). The check sandbox now exposes the loader cache, as the shell sandbox already did, so `libpython` is found; the rest of `/etc` stays out.
- Tests that kill engine processes now verify each process's start identity and command before signalling, and refuse invalid, own and own-group targets.

### Security

- On Linux, an MCP server whose launch ran past the MCP timeout while its sandbox was still being set up could outlive the timeout and keep running with no supervisor (no wall-clock or memory limit), though still confined by its sandbox namespaces and dropped capabilities. Affected: 0.2.4 through 0.2.10, MCP servers on Linux only. The shell tool was not affected, and neither was macOS, where the MCP server runs as the launched process itself and cannot fork.
- Fix: a timed-out or stopped sandbox launch is now killed as a whole unit (the launcher, its process group and every tracked descendant), and cleanup fails instead of reporting success when any member survives. The same cleanup now covers the check and probe runners and the execution supervisor's shutdown.

## 0.2.9 (2026-11-01)

- Add optional per-step models. Each request is assigned a role (`read`, `plan`, `edit`, `finish`, `decide2`, `unit`) from what the engine already knows, and each role can have its own endpoint, model, sampling, limits, slot, tokenizer and cost rates. Set them with `--route ROLE.FIELD=VALUE`, `STATEFRAME_ROUTE_<ROLE>_<FIELD>`, a TOML or JSON routes file, or the profile. `stateframe config init` writes a commented example.
- With no route settings, requests are byte-identical to 0.2.8.
- A reply that edits files or says done from another role's model is discarded unrun and the same step is asked again of the `edit` or `finish` model, from the same frame.
- Each role's server is checked at start. Remote endpoints need a per-role opt-in (or the global one), including on redirects. Routed requests never go through an HTTP proxy.
- Runs print their effective routes at start and per-role sends, discarded replies, tokens and estimated cost at the end.
- Resume records each request's route before it is sent and verifies it on replay. Changing a route on resume is refused; `stateframe resume --route` may move an endpoint or rename a model.
- The request in flight when a routed run stopped is checked against its rebuilt copy on the first resume, and on a later resume only when the previous resume recorded that its own check passed. Otherwise the run resumes by full replay without that check and says so. Routed runs never restore from a checkpoint; they replay.
- `shell` calls are not escalated: they run on whichever role's model asked for them, including commands that write files. `--shell read-only` prevents that.
- Workspace-writing custom, skill and MCP tools escalate to the `edit` model like built-in edits.
- After a failed test run on the current code, the next request goes to the `plan` model.

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
