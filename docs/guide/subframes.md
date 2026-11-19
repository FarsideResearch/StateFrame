# Subframes

Subframes let the parent split independent file work into separate, bounded engine runs. They are parallel processing
inside one StateFrame task, not independently steered agents: there are no child chat histories, model-written briefs,
reports, or messages between subframes. The parent owns the only authoritative state, workspace, and journal. A
subframe's proposal becomes parent state only after the host verifies it and publishes a transaction. The model is fork-join: each subframe's workspace is a snapshot forked from the parent at dispatch, and its
result is joined back only through that verified transaction. Subframes never see another subframe's work in progress,
only the committed interface projections of producers they depend on.

## When to use them

Use `parallel` when a change has at least two distinct pieces that each need their own read, edit, and check cycle,
and their owned paths do not overlap. Small edits that fit in one response usually do not justify the extra setup.
A dependency is allowed when its consumer names the producer's export. The consumer starts
only after that producer has committed, and receives the committed interface projection, never work in progress.

For example, two modules with separate files and tests can be suitable. A change where every piece edits the same
shared module is not. Keep tightly coupled work together, especially when its interfaces are still being decided.
Long dependency chains leave little work that can run at the same time.

Subframes do not promise a general speedup. They add process, workspace-copy, request, and verification costs.
Concurrent prompts also share the server's capacity, so large frames or a busy endpoint can make a run slower.

## Enable subframes

Use the same endpoint and model settings as a normal run, then add `--parallel`:

```bash
stateframe run --repo path/to/repo --task "Fix the parser and formatter, with tests for each." --parallel auto
```

`--parallel off` is the default. `auto` lets the engine size concurrency; a positive integer sets the maximum number
of subframes active at once. For example, `--parallel 2` allows up to two active subframes. `--parallel 1` uses the
same subframe validation and publication process but runs at most one at a time; the default minimum plan size is
still two subframes.

Enabling the feature only offers the `parallel` tool to the model. The model decides whether to call it, so a run
with zero subframes is a valid outcome. Enabling it changes the offered tool menu even if the model never uses it.
The shipped default profile already supports subframes; no extra profile flag is needed.

### Opt-in parallel selection

Set `STATEFRAME_PARALLEL_SUGGEST=on` with `--parallel auto` or `--parallel N` to let the host recommend fan-out only when it finds at least two unambiguous, disjoint Python source and test pairs. The default is `off`, which preserves ordinary request bytes. With a recommendation, the parent request shows revision-bound outlines for those pairs, a fact naming each pair, and a concrete `parallel` call near the top of the instructions. Each subframe still receives its full owned source and test when dispatched. The host records its decision, backend, pressure axis, and reason in `engine.log`; it never forces a tool call.

The input axis fires when the combined pair sources, estimated with the request budget's 2.46 bytes per token fallback, exceed `STATEFRAME_PARALLEL_SUGGEST_INPUT_SHARE_PCT` of the smaller parent frame or prompt budget (default 50%). The output axis fires when estimated edits exceed two maximum model responses. The estimate is `max(12, source lines × edit fraction) + max(8, test lines × edit fraction)` per pair, multiplied by tokens per edit line. Defaults are 25% and 12 tokens per line. These are selection estimates, not edit limits.

| Setting | Default | Meaning |
|---|---|---|
| `STATEFRAME_PARALLEL_SUGGEST_MIN_UNITS` | `2` | Minimum independent pairs. |
| `STATEFRAME_PARALLEL_SUGGEST_INPUT_SHARE_PCT` | `50` | Strict input share threshold, 1 to 100. |
| `STATEFRAME_PARALLEL_SUGGEST_EDIT_FRACTION_PCT` | `25` | Expected changed line share, 1 to 100. |
| `STATEFRAME_PARALLEL_SUGGEST_TOKENS_PER_EDIT_LINE` | `12` | Emitted token estimate per changed line. |
| `STATEFRAME_PARALLEL_SUGGEST_BACKEND` | `auto` | `auto`, `hosted`, `local`, or `unknown`. |

`auto` recognizes a proven llama.cpp provider as local. Other providers remain unknown because a strict API dialect alone cannot distinguish a hosted service from a rented GPU. Declare `hosted` when the endpoint has effectively unlimited model concurrency. Hosted endpoints can recommend at two pairs with no pressure axis, subject to the configured `--parallel` concurrency. Local endpoints reuse `/slots` discovery and require at least two free subframe slots plus either pressure axis. The parent's pinned slot is excluded. Discovery failure uses the unknown rule: at least two configured slots and a pressure axis. No recommendation oversubscribes the configured concurrency.

## Contract and admission

The parent calls `parallel` with a plan of subframe contracts, normally at least two. You describe the task at the
CLI; the model supplies these contracts. The engine can fill checks, reads, exports, and dependencies when they
follow unambiguously from the plan and existing source. The expanded plan is checked before anything launches.

The model-facing fields are:

| Field | Meaning |
|---|---|
| `id` | Unique identifier, up to 24 lowercase letters, digits, or underscores, starting with a letter. |
| `goal` | A short description of the work, up to 300 characters. Required; `outcome` is accepted as an older spelling. |
| `owns` | Existing files this subframe may change. |
| `creates` | New files it must create. |
| `exports` | Interfaces it promises, with an id, path, symbol, and optional signature. Existing Python signatures can be inferred from source. |
| `consumes` | An existing symbol or another subframe's export id; cross-subframe consumes create dependency edges. |
| `reads` | Supporting source paths that are neither owned nor another subframe's paths. |
| `requirements` | Up to eight acceptance conditions or exact parent requirement ids, each up to 300 characters. |
| `check.tests` | Frozen test selectors for this subframe. An empty list means no subframe check. |
| `model_class` | Optional routing class; `default` when omitted. |

Before launching anything, the engine checks normalized in-repository paths, protected files, links and aliases,
exclusive ownership, create targets, export and consume resolution, approved reads, dependency cycles, source size,
and configured caps. The complete first request is measured before its first provider send. The coupling scan
looks for undeclared references between paths. A detected
undeclared edge refuses admission; a scan with incomplete coverage is recorded as `UNKNOWN`, not as proof of
independence. The engine does not invent missing interfaces or silently drop owned source to fit a frame.

`owns` names existing files the subframe may edit. `creates` names only paths that do not exist yet; if a target already
exists, admission refuses the plan with `CREATE_EXISTS` and the model must retry with that path under `owns`.
Name the independent outcomes and their files in the task so the model can plan the split. Each subframe receives
its owned files and selected test source on its first request.

Each subframe sees its complete owned source on its first request, its declared support, and only committed dependency
interfaces. Other producers' source bodies do not appear in its model-facing request; diagnostics can withhold those
bodies while raw evidence remains in the run record. Its tool menu is restricted to its owned work,
declared creation, reads, and checks. Deletes and moves remain parent operations. A check that writes files is marked
`INVALID` and cannot change the proposal. The subframe seals a proposal; the parent validates scope, fresh base digests,
interfaces, checks and composition before committing. A passing subframe check is not a passing final task check.
After the last commit, the parent runs final verification and reports integration status separately from each
subframe's status.

## CLI and profile settings

The defaults below apply unless a profile or shell setting overrides them. CLI limit flags take precedence over
those settings. Limits apply to each `parallel` call unless described as per-subframe or run-wide.

| CLI flag | Default | Meaning |
|---|---|---|
| `--parallel off\|auto\|N` | `off` | Offer the tool with automatic concurrency or up to N active subframes. N must be positive. |
| `--parallel-slots LIST` | Unrestricted pool | Restrict slot leases to comma-separated server slot ids, for example `2,5`. |
| `--parallel-max-subframes N` | 6 | Maximum number of subframes in one plan, including those waiting to start. |
| `--parallel-min-subframes N` | 2 | Minimum number of subframes in one plan. |
| `--parallel-max-paths-per-subframe N` | 6 | Maximum combined count of existing owned paths, new paths, and declared supporting reads per subframe. |
| `--parallel-max-paths-total N` | 24 | Maximum combined count of those paths across the plan. A supporting read used by two subframes counts in each. |
| `--parallel-subframe-tokens N` | 26,675 with the shipped budget | First-request token ceiling per subframe. Derived as 80% of the run's prompt budget, rounded down (at least one token). |
| `--parallel-max-owned-bytes N` | 24,576 bytes (24 KiB) | Maximum total size of existing owned source files per subframe. |
| `--parallel-subframe-max-requests N` | 12 | Maximum physical provider attempts per subframe, including transport retries. The default can be reduced to fit the run's remaining allowance. |
| `--parallel-parent-reserve N` | 4 | Requests kept available for the parent rather than reserved for subframes under a finite run allowance. |
| `--parallel-check-permits N` | 1 to 4, CPU-dependent | Maximum checks running at once under the run's shared check-permit pool. Computed as CPU count divided by four, rounded down and bounded to 1 through 4. |
| `--parallel-max N` | 4 | Concurrency ceiling for both automatic and explicit concurrency. This is separate from the total number of subframes in a plan. |

Selected test-source paths are validated and contribute to request size. They are not added separately to the path
counts above unless they are also declared as owned, new, or supporting-read paths.

`auto` probes idle slots at the configured model endpoints and respects `--parallel-max` and any slot pool. When
`/slots` is unavailable, it uses the configured ceiling, further limited by an explicit slot pool. Slot leases are
cooperative: they coordinate participating StateFrame runs but do not reserve hardware against other clients.

The corresponding profile or shell settings are:

| CLI flag | Setting |
|---|---|
| `--parallel-slots` | `STATEFRAME_PARALLEL_SLOTS` |
| `--parallel-max-subframes` | `STATEFRAME_PARALLEL_MAX_SUBFRAMES` |
| `--parallel-min-subframes` | `STATEFRAME_PARALLEL_MIN_SUBFRAMES` |
| `--parallel-max-paths-per-subframe` | `STATEFRAME_PARALLEL_MAX_PATHS_PER_SUBFRAME` |
| `--parallel-max-paths-total` | `STATEFRAME_PARALLEL_MAX_PATHS_TOTAL` |
| `--parallel-subframe-tokens` | `STATEFRAME_PARALLEL_SUBFRAME_REQUEST_TOKENS` |
| `--parallel-max-owned-bytes` | `STATEFRAME_PARALLEL_MAX_OWNED_SOURCE_BYTES` |
| `--parallel-subframe-max-requests` | `STATEFRAME_PARALLEL_SUBFRAME_MAX_REQUESTS` |
| `--parallel-parent-reserve` | `STATEFRAME_PARALLEL_PARENT_RESERVE_REQUESTS` |
| `--parallel-check-permits` | `STATEFRAME_PARALLEL_CHECK_PERMITS` |
| `--parallel-max` | `STATEFRAME_PARALLEL_MAX_CONCURRENCY` |

`STATEFRAME_PARALLEL_DROP_TOOL_AFTER_TERMINAL` is on by default for runs with `--parallel`: after a verified, complete
fan-out, that exact plan cannot be dispatched again in the same run. The `parallel` schema stays available, so a
different plan can still be dispatched. Set it to `off` to disable the re-dispatch guard. It has no effect when `--parallel` is off.

### Behavioral interface documentation

`STATEFRAME_SUBFRAME_INTERFACE_BEHAVIOR` is off by default; set it to `on` in the shell or a custom profile to
include multiline Python docstrings in consumed interface projections. With it unset or `off`, interface records,
digests, and child task text retain their existing bytes and first-line documentation.

When enabled, each nonempty docstring becomes a labeled JSON envelope with `text` and `truncated` fields in the
existing `doc_first_line` wire field. This avoids adding record fields or changing the wire schema. The entire
envelope, including the label, JSON escaping, and truncation marker, is capped at 2,048 UTF-8 bytes per interface.
It contains a deterministic prefix of the cleaned docstring, with Unicode and control characters JSON-escaped.
The adjacent path, definition line, and SHA-256 digest identify the resolved source; re-exports use the defining
file, and methods use their own documentation and definition line. Missing docs remain empty, and non-Python
projections are unchanged. Recorded projections retain their text when deserialized regardless of the current switch.

The projection labels documentation as untrusted source data, not instructions or proof of behavior. Quoting keeps
embedded newlines and instruction-like text inside the envelope. The provider is never executed and its full source
remains withheld unless the contract grants a supporting read. The excerpt travels in the child task and counts
toward the existing first-request budget; requirements are retained, and an oversized request is refused before
provider send. Documentation cannot establish undocumented return shapes, exceptions, side effects, or sentinels.
Use explicit requirements, checks, or declared supporting reads for those contracts.

### Validation and request allowances

With subframes enabled, limit values must be positive integers. Invalid limits are rejected at run start rather
than silently clamped:

- Maximum subframes cannot exceed 1,024, and minimum subframes cannot exceed that configured maximum.
- Paths per subframe cannot exceed total paths.
- The first-request token cap cannot exceed the run's prompt budget. The default is `prompt_budget * 80 // 100`,
  so the shipped 33,344-token budget yields 26,675 tokens.
- Explicit `--parallel N` cannot exceed `--parallel-max` or the number of ids in an explicit slot pool.
- A slot pool must be nonempty and contain unique nonnegative integers.

The run's request allowance is shared between parent work and subframe provider attempts. Before launch, the engine
reserves a request allowance for every subframe and keeps the parent reserve available. With a finite allowance and
no explicit subframe request cap, it reduces the default cap to the smaller of 12 and the available allowance divided
by the number of subframes (rounded down). If this gives fewer than three requests per subframe, admission is refused.
An explicitly configured cap is reserved as requested; if it cannot fit, admission is refused instead of reducing it.

`STATEFRAME_PARALLEL_FANOUT_CONTINUATION` defaults to `on` for runs using `--parallel`. It keeps subframe-specific
continuation guidance and current source/test windows available to the parent; `off` omits that section.
With `--parallel off`, the runner removes parallel settings, so they cannot enable the feature on their own.

### Host timing and process safety

On Linux the host reads process liveness from `/proc`. Other platforms, or a restricted `/proc`, use `ps`
with one retry. An inspection timeout leaves a seal pending for the next tick; the host never seals while
process-group liveness is unknown or a group member remains alive. Temporary supervision failures retain
the worker and its slot instead of cancelling it. Channel service rotates across attempts and pending
sequences, guaranteeing progress even when advancement or directory scans exceed the soft 0.5 second budget.
Up to eight attempts receive service per tick, each with at least one pending request when available.

These shell settings accept positive finite seconds and are recorded for enabled subframe runs:

| Setting | Default base seconds | Purpose |
|---|---:|---|
| `STATEFRAME_PARALLEL_INSPECT_SECONDS` | 15 | Each fallback `ps` or macOS memory-pressure probe, with one retry |
| `STATEFRAME_PARALLEL_CLEANUP_SECONDS` | 30 | Termination grace, post-kill reap, containment sweeps and output drain |
| `STATEFRAME_PARALLEL_PREPARE_SECONDS` | 120 | Snapshot clone subprocess; preparation thread joins allow preparation plus cleanup time, at least 130 seconds |
| `STATEFRAME_PARALLEL_WATCHDOG_GRACE_SECONDS` | 60 | Extra host allowance beyond an explicitly configured worker run-wall budget |
| `STATEFRAME_PARALLEL_HOST_RESPONSE_SECONDS` | 300 | Maximum wait for a host answer, including an asynchronous seal result |

A subframe worker with no run wall in its spec is capped at four hours, counted from the moment it holds its model slot.

Each base allowance is multiplied by the one-minute host load divided by logical CPU count, bounded from
1 through 8, with each resulting wait capped below the platform polling limit of 2,147,483 seconds.
Missing load information uses the base. The parent run deadline still takes precedence over
the host response allowance. Pending seal acknowledgments do not restart that allowance. An unanswered
request ends with a host-response timeout naming the operation and unit. Increase the response setting
for unusually slow storage or sustained host pressure. Cancellation grace starts after SIGTERM delivery;
time spent inspecting processes cannot consume it. A failed signal inspection remains retryable.

### First-request token measurement

The first subframe request uses an exact `/tokenize` count when that endpoint can measure the final body. Otherwise it
uses a versioned byte profile selected by exact provider and model identity. The measured profile for DeepInfra's
`deepseek-ai/DeepSeek-V4-Flash-0731` is `deepinfra/deepseek-ai/deepseek-v4-flash-0731@v1` and estimates
the serialized request size in bytes multiplied by 0.3193, rounded up, with no fixed residual. Its confidence is
labeled `default`, rather than an exact token count. Other providers and models
use the generic conservative ratio: the larger of one third token per byte and the maximum measured profile ratio
with configured headroom. At the default headroom, that is one third token per byte. Image inputs are currently
reported as unknown and refused before sending.

| Profile switch | Default | Accepted values |
|---|---:|---|
| `STATEFRAME_PARALLEL_BYTE_TOKEN_CALIBRATION` | `auto` | `auto`, `generic`, or the exact versioned profile id above |
| `STATEFRAME_PARALLEL_BYTE_TOKEN_HEADROOM_PCT` | 10% | Whole percent from 0 through 50 |
| `STATEFRAME_PARALLEL_USAGE_FEEDBACK` | `observe` | `observe` or `off` |

The published profile's 10% margin is a floor: headroom values below 10 do not lower its bound, while higher values
raise it up to the configured maximum. Usage feedback records observed request/usage pairs for later calibration, but
it does not change a subframe cap estimate. The `off` setting disables those observations; there is no feedback mode
that admits a request based on a lower run-level estimate. The measurement record includes the selected calibration,
measurement method, confidence, serialized byte count, estimate, and cap. The selected calibration and values are
saved for replay.

Each physical provider attempt consumes one subframe request allowance. The transport authorizes the attempt in a
durable record before sending, including an uncompressed retry after gzip HTTP 415. An attempt interrupted before
its result is recorded still counts as spent on resume.

## Where to see results

Each actual request is saved as `wire/NNNN.request.json`. Inspect the parent run and each subframe run directory to
review the bytes actually sent, including the subframe task text and restricted tool schemas.

The parent observation starts with `Subframe results:` and records each subframe's status, changed files, contract
verdict, check status, request count, and any refusal or cancellation reason. It ends with the final check and
integration result. Timing measurements stay in the saved results rather than the model's tool reply.

The next parent request retains a bounded summary of the latest call, including subframe identities, statuses,
changed files, reasons, and verification results. Full results and refusal evidence remain under
`parallel/<call>/receipt/`; the model can page through them with `read_check_receipt`.

The run summary (`summary.json`) has a `parallel` section containing per-call results, subframe outcomes, usage
totals, and measurements:

| Summary location | What to look for |
|---|---|
| `parallel.totals.subframes` | Number of subframe outcomes recorded across calls. A refused plan can launch none. |
| `parallel.totals.requests` | Total recorded subframe requests. |
| `parallel.measured.peak_active_subframes` | Highest measured overlap across calls. It can be lower than the requested concurrency. |
| `parallel.calls` and `parallel.subframes` | Per-call integration results and per-subframe status, reason, usage, and timing. |
| `parallel.degraded` | Calls unable to use subframes because of context or resource limits. |
| `parallel.errors` | Unreadable or invalid result artifacts that prevented complete summary reporting. |

Totals also include physical sends, prompt/completion tokens, and initial-frame token measurements when recorded.
Measurements include wall time, estimated serial time, their ratio, queue wait, and supervisor tick latency when
available. Those measurements describe the recorded run, not what another run will achieve.

The `parallel` section is included when the run has parallel artifacts. If the tool was never called, it can be absent.
Do not treat its absence as proof that the flag was ignored.

Under the run directory, `parallel/<call>/result.json` holds the full call result. Each
`parallel/<call>/units/<subframe>/` directory holds that subframe's specification, request records, and process logs.
Its launch log is under `proc/<incarnation>/engine.log`. Use `stateframe inspect <run-dir> --send N --frame` to
review a recorded parent frame. See [Run storage](storage.md) for the overall run-directory layout.

Per-subframe status and integration status answer different questions. `COMMITTED` means that subframe's proposal
was accepted with a passing composition check, not that the whole task passed. Final integration can be `VERIFIED`,
`CHECKED_WITH_UNKNOWNS` (a passing final check with incomplete coupling-scan coverage), `FAILED`, `UNVERIFIED`, or
`NOT_RUN`. Review both the final check and the overall run result before applying the candidate. A failed subframe
does not undo other subframes' committed changes inside the run.

## Refusal and outcome codes

Admission refusals reject the plan before launching it. Later failures apply to individual subframes; the parent
receives the reason and can revise the plan or continue the work itself. Codes in saved results sometimes use
`UNIT` where the displayed message uses `SUBFRAME`; both spellings are listed below.

| Code | Cause | What to do |
|---|---|---|
| `ADMISSION_REFUSED` | Fallback summary reason when admission was refused without a specific reason. | Read the call result and recorded tool reply for details before retrying. |
| `INVALID_ARGUMENTS` | The model supplied malformed fields, values, or selectors in the plan. | Follow the reply's named field and repair hint; the parent can resubmit a corrected plan. |
| `TOO_FEW_UNITS` / `TOO_FEW_SUBFRAMES` | Plan has fewer subframes than the configured minimum. | Use serial work, add independent work, or lower `--parallel-min-subframes`. |
| `TOO_MANY_UNITS` / `TOO_MANY_SUBFRAMES` | Plan exceeds the configured maximum count. | Reduce the plan or raise `--parallel-max-subframes` within the 1,024 ceiling. |
| `TOO_MANY_PATHS` | A subframe has no owned or new path, or a per-subframe or total path count is exceeded. | Give it work paths, reduce scope, or raise the path limit named in the detail. |
| `OWNERSHIP_OVERLAP` | Owned or new paths overlap, including a file and a descendant path. | Assign each path to one subframe; keep shared edits together. |
| `UNSAFE_PATH` | A path is protected, missing where it must exist, not normalized, linked, or not a regular file; also used for invalid or duplicate subframe identities. | Correct the named path or identity. Use repository-relative paths and one owner for a missing directory subtree. |
| `CREATE_EXISTS` | A path declared as new already exists, either at admission or when publishing a proposal. | Put an existing file under `owns`; if it appeared later, replan against the current files. |
| `READ_NOT_ALLOWED` | A read or test-source path is unsafe, missing, or overlaps ownership that it may not read. | Correct supporting reads and test selectors; use a declared interface for another subframe's work. |
| `UNKNOWN_REQUIREMENT` | A contract refers to a requirement id not present in the parent requirements. | Use an existing requirement id or a plain acceptance condition. |
| `EXPORT_OUTSIDE_OWNERSHIP` | An export points outside its producer's owned or new files. | Move the export to the subframe that owns that path. |
| `DUPLICATE_EXPORT` | Export ids repeat, or more than one export names the same path and symbol. | Declare each interface once with a unique id. |
| `UNRESOLVED_CONSUME` | A dependency cannot be found, names conflicting export details, or refers to the consumer itself. | Name an existing definition or the correct producer export. |
| `AMBIGUOUS_CONSUME` | An interface has multiple possible definitions or cannot be resolved statically. | Use an unambiguous definition, simplify the interface, or keep the work together. |
| `UNSUPPORTED_CONTRACT` | A cross-subframe non-Python dependency has no usable check recipe. | Supply a check for the consumer or keep the dependent work in one subframe. |
| `UNDECLARED_COUPLING` | Source inspection finds a dependency missing from the declared interfaces, at admission or after edits. | Declare the producer/consumer relationship or combine the coupled work. |
| `DEPENDENCY_CYCLE` | Declared dependencies form a cycle. | Combine those pieces or change the dependency direction so no cycle remains. |
| `MISSING_SIGNATURE` | A recognized repair-hint code for a missing Python export signature. The current admission code has no path that emits it. | If encountered in a record, supply the canonical signature named by the hint; omitted signatures are not automatically refused. |
| `UNIT_CONTEXT_TOO_LARGE` / `SUBFRAME_CONTEXT_TOO_LARGE` | Existing owned source exceeds the byte limit, or the complete first request exceeds the token cap. | Check which limit is named. Split the subframe or raise `--parallel-max-owned-bytes` or `--parallel-subframe-tokens`; the token cap must fit the prompt budget. |
| `RESOURCES_UNAVAILABLE` | Request reservations do not fit, no usable model slots are available, or workspace preparation fails disk/memory checks. | Read the detail. Reduce count or reservations, increase `--max-turns`, wait for slots, or free the named host resource. |
| `TINY_PLAN` | Advisory for a plan whose subframes each own one file of at most 4 KiB, have one check, and declare no new files or supporting reads. It does not refuse admission. | Consider doing the small edits in the parent. |
| `RESOLVED_REQUEST_CAP` | Informational record of the per-subframe request allowance granted at admission. | Compare it with the requested/default cap when investigating early exhaustion. No repair is required. |
| `STALE_BASE` | A proposal's base file no longer matches the current parent file. | Replan from current source; do not force the stale proposal into the candidate. |
| `MISSING_EXPORT` / `SIGNATURE_MISMATCH` | A proposal omits a promised interface or changes its declared Python signature. | Correct the implementation or revise the interface contract before retrying. |
| `UNRESOLVED_IMPORT` | Contract verification finds an unresolved local import in the composed files. | Correct the import or include the missing dependency in the plan. |
| `COMPOSITION_CHECK_FAILED` | The check for a proposal composed with current parent files did not pass. | Inspect its failure or invalid-check detail and fix the implementation or check. |
| `UNVERIFIABLE_CONTRACT` | Static verification is unknown and no passing composition check establishes the contract. | Supply a meaningful check or simplify the interface. |
| `PRODUCER_FAILED` | A dependency producer did not commit, so its consumer is cancelled without starting. | Fix the producer first, then retry the dependent work. |

Outcome statuses describe what happened to a proposal:

| Status | Cause | What to do |
|---|---|---|
| `COMMITTED` | Accepted into the parent candidate with a passing composition check. | Review final integration and the whole-run result. |
| `COMMITTED_UNVERIFIED` | Accepted into the parent candidate without a passing composition check. | Obtain verification before relying on the change. |
| `REFUSED` | The plan or an individual subframe result was not accepted. | Read the accompanying reason and use the code table above. |
| `BLOCKED` | The subframe reported that its contract prevents completion. Reasons include `NEEDS_UNOWNED_FILE`, `INTERFACE_WRONG`, `CONTRACT_IMPOSSIBLE`, `CHECK_WRONG`, or `OTHER`. | Read the detail and revise ownership, interface, requirements, or check as appropriate. |
| `BUDGET_EXHAUSTED` | The subframe spent its physical provider-attempt allowance. | Inspect retries and progress; reduce scope or start a run with a larger subframe cap and enough total requests. |
| `CANCELLED` | Cancellation was requested, the supervisor closed, or a producer failed. | Check the reason; repair a failed dependency or resume an interrupted parent run when permitted. |
| `TIMED_OUT` | The subframe process reached its deadline. | Inspect its log; reduce scope or use an appropriate run timeout. |
| `ENGINE_ERROR` | Launch, execution, or result handling failed. | Inspect the subframe process log and call result; resolve the named error before retrying. |
| `RECOVERY_REQUIRED` | An interrupted durable decision or publication must be reconciled before work can continue. | Resume the parent run; keep its records intact. |

### Edits from a subframe that stopped early

A subframe that ends without submitting (for example `HARNESS_CONTRACT` or `BUDGET_EXHAUSTED` after it already
edited a file) is recorded as `REFUSED` or with its stop status, and its edits do not reach the parent candidate.
The parent keeps the newest readable capture of that subframe's files at `parallel/<call>/salvage/<subframe>.json`
and journals `parallel_unit_salvaged` with its digest and the changed files (at most 64 names, with the full count in
`files_total`). A capture is kept when any file differs from the bytes the subframe started from or a file was
deleted, not only when its main target file changed. The same is done for a subframe whose submission the parent did
not accept. By default nothing reads that file; it is there for you to inspect.

Set `STATEFRAME_PARALLEL_SALVAGE_ADOPT=on` to let the parent offer the kept edits of a subframe that never submitted to the ordinary commit path.
A submission that was already refused is kept but not offered again. The
default is `off`, which leaves request bytes, outcomes, and records unchanged. With it on, the parent compares the
capture with the snapshot the subframe started from, builds a proposal from the owned files that changed, and runs
it through the same gates as a submitted proposal: the parent file must still match the subframe's starting bytes
(`STALE_BASE` otherwise), contracts are verified, and the subframe's check is rerun on the composed files
(`COMPOSITION_CHECK_FAILED` otherwise). The final check then covers it like any other commit. Nothing is bypassed,
and a subframe's own check result is never trusted for salvaged edits.

| Result | What the parent sees | What to do |
|---|---|---|
| Adopted | The subframe's status becomes `COMMITTED` or `COMMITTED_UNVERIFIED`, and a `Subframe <id> salvage:` line names why it stopped, the adopted files, the transaction, and the capture's path and SHA-256. | Review it as a commit from a subframe that never declared itself done. |
| Not adopted | The subframe keeps its stop status. The salvage line gives the refusal reason, the files that had changed, and the capture's path and SHA-256. | Use the reason: redo the work on current files after `STALE_BASE`, fix the edit after a failed check, or revise ownership after `SCOPE_VIOLATION`. |

Adoption is also refused with `SALVAGE_INVALID` (the capture is missing, altered, or malformed),
`SALVAGE_UNSUPPORTED` (the subframe deleted a file), `SALVAGE_INCOMPLETE` (a declared new file was never created), or
`SALVAGE_EMPTY` (no owned file differs from the starting snapshot). Each verdict is also written to
`parallel/<call>/salvage/<subframe>.adoption.json` and added to the `parallel_unit_salvaged` journal entry.

Check statuses are separate: `PASS`, `FAIL`, `INVALID`, or `NOT_RUN`. An `INVALID` check may have written files,
failed isolation or cleanup, or been unable to acquire a shared check permit. It is not passing evidence.

## Resume and recovery

Resume the parent run with `stateframe resume <run-dir>`. StateFrame records the plan before side effects, replays
completed calls without launching subframes or making new provider calls, and reconciles interrupted publications
so that each accepted proposal is applied once. Recovery stops verified old subframe processes and can continue an
unfinished subframe from its existing run directory. Checkpoint restore must reproduce the next parent request
before continuing. Keep the whole parent run directory, including its parallel records.

Recovery records the child process's kernel birth identity at launch. On Linux this is the boot ID and process start
tick; on macOS it is the kernel process start time with microsecond precision. Resume compares that exact identity
before sending a signal. If a process record predates this identity, resume refuses to signal it and asks for manual
inspection. A reused process-group number whose recorded processes have exited is fenced as stale.

If the leader exits while its subframe descendants remain, resume names the surviving PIDs and refuses automatic
recovery when it cannot prove their recorded identities. Inspect those PIDs with `ps -p <pid> -o pid,ppid,pgid,command`
and the run's `parallel/<call>/units/<subframe>/proc/<incarnation>/process.json`. Once you have confirmed they belong
to that run, stop only those PIDs with the `kill -TERM <pid> ...` command shown in the error. If they do not exit,
repeat with `kill -KILL <pid> ...` after checking their identities again, then retry `stateframe resume <run-dir>`.

## Limits and current gaps

- Slot leases are cooperative advisory leases. They coordinate StateFrame runs that use the same lease protocol; they
  do not reserve hardware against unrelated server clients.
- Diagnostic projection is fail-closed whenever the subframe's view withholds a file or a withheld frame appears: a
  subframe then sees exit codes, counts, node ids, exception type names and the locations of visible frames, while
  message bodies, captured stdout/stderr/log and parametrize ids are replaced by `[... withheld]` markers, and a
  command-kind check shows only its exit status and counts. Full assertion detail is kept only when the workspace
  holds no withheld files and every frame is owned.
- Deletes and moves are not available inside a subframe.
- Non-Python export signatures are not statically verified; their contract verdict remains `UNKNOWN` and needs a
  passing composition check where required.
- The coupling scan is bounded to 2,000 files, 16 MiB of source, and a 10-second scan deadline. Incomplete coverage
  is recorded as `UNKNOWN`; it does not establish independence.
- Large prompts share server capacity and can make concurrency slower.
- Without a parent test command or subframe check selectors, the final check is `NOT_RUN`. If coupling-scan
  coverage is unknown, integration is `UNVERIFIED`.
- Workspace preparation requires free disk space for two repository-sized copies plus 1 GiB and at least 20% free
  memory. An unavailable resource probe also prevents preparation.

## Troubleshooting

| Symptom | What to check |
|---|---|
| Zero subframes | First check whether the model called `parallel`. If it did not, the run may proceed normally. If it did, read the admission reason in `parallel/<call>/result.json`. |
| Fewer active subframes than requested | N is a ceiling. Dependency order, idle slots, slot leases, preparation resources, or check permits can reduce overlap. Look at peak active subframes and queue wait in the summary. |
| Context refusal even though owned files are small | The first request includes instructions, tools, checks, supporting source, and dependency interfaces as well as owned source. Read the token measurement, reduce declared scope, or raise the token cap within the prompt budget. |
| Subframes exhaust requests early | Physical attempts include transport retries, and admission may reduce the default allowance. Check `RESOLVED_REQUEST_CAP`, request counts, physical sends, and the total run allowance. |
| Every subframe check passes but integration fails | Inspect the final check and composed interfaces. Separate checks do not establish that the combined changes work together. |
| A check is `INVALID` | Read its invalid-check detail. Keep checks from changing product files and resolve isolation, cleanup, or shared-permit failures before trusting the result. |
| Resume refuses to stop old processes | Follow the identity-inspection steps above. Do not signal a PID based only on its number or delete recovery records to bypass the check. |

For other run failures, see [Troubleshooting](troubleshooting.md).
