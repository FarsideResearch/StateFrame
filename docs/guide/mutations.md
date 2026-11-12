# Mutation units

Mutation units are small Python helpers with declared JSON input and output. Use one for a bounded job such as reading a file’s structure or comparing snapshots, so the model can inspect a compact result instead of repeating a long sequence of tool calls.

StateFrame runs each unit in an isolated, resource-bounded process. A unit receives admitted input snapshots, not repository paths. It cannot access the network or write to the workspace directly. A network-enabled unit asks the host broker to fetch public HTTP(S); a write-enabled unit returns proposals for the host to validate and apply. Mutation results are untrusted observations, not test receipts or proof that a task is correct.

## Enable mutations

Run and chat accept `--mutations off|prebuilt|authored`. Mutations default to off. Choose a level for one run:

~~~sh
stateframe run --repo . --task "Summarize data/scores.csv" \
  --mutations prebuilt --mutation-read 'data/**'
~~~

The level can also come from `STATEFRAME_MUTATIONS` in the shell or from the selected profile’s `switches` object. Its values are `off`, `prebuilt`, and `authored`. Precedence is explicit `--mutations`, environment, profile, then `off`. Use `--mutations off` to override an enabled environment or profile value.

In an existing profile, add this key to its `switches` object:

~~~text
"STATEFRAME_MUTATIONS": "prebuilt"
~~~

~~~sh
export STATEFRAME_MUTATIONS=prebuilt
stateframe run --repo . --task "Summarize data/scores.csv"
~~~

`stateframe mutations enable-level LEVEL` prints the environment setting and equivalent run and chat commands. With `--profile PROFILE`, it validates the selected profile and includes it in the printed commands. It does not edit profiles.

| Level | Available behavior |
|---|---|
| `off` | No mutation tool, mutation frame, or mutation runtime import. Request construction stays unchanged. |
| `prebuilt` | Shipped helpers with source digests checked at admission. |
| `authored` | Prebuilt helpers plus authoring and repair episodes. |

Mutation execution is Linux-only in this release. On another platform, requesting an enabled level or attaching a mutation hook is refused with `MUTATION_PLATFORM_UNSUPPORTED: mutations are Linux-only in this release; macOS is planned`. The management commands `list`, `show`, and `doctor` remain available and report the platform limit. Other `stateframe mutations` subcommands, including `selftest`, are refused.

## Grants and access

Enabling a level does not grant network or workspace writes. Each capability is a separate, host-enforced grant. Unit source, output, repair, retention, and promotion cannot increase these grants.

| Capability | Run or chat option | Effect |
|---|---|---|
| Public web | `--mutation-network public-web`; repeat `--mutation-domain DOMAIN` to add allowed domains | Network requests go through the host broker. A supplied domain list restricts broker destinations. The run also requires `--shell-network full`. That setting applies to the ordinary shell tool too; the mutation domain list does not constrain shell traffic. |
| Workspace proposals | Repeat `--mutation-write GLOB` | Units may propose edits to matching workspace paths. The host checks the full proposal set, edit scope, protected paths, parent identities, and base digests before applying it after a successful terminal result and verified cleanup. This does not change the model’s ordinary edit-tool grant. |
| Input snapshots | Repeat `--mutation-read GLOB` | Adds a path restriction for files a unit may snapshot. With no mutation read globs, the run’s ordinary read scope applies. |

Mutation read and write grants match repository-relative paths. A path without `*` or `?` matches only that exact filename; brackets are literal. `*` matches zero or more characters within one directory, and `?` matches one. A complete `**` path segment can span directories, so `data/**` includes files directly in `data/` and below it. The run's ordinary declared read paths use the same matching rules for mutation snapshots.

Inputs are copied into read-only snapshots before execution. A request can admit at most 16 singly linked regular files, each no larger than 4 MiB, with a 16 MiB total input cap. Paths must be relative and at most 4,096 characters; an optional `tail_bytes` snapshot is also capped at 4 MiB. Common credential path names and the `.git` and `.stateframe` directories are excluded as defense in depth. These filters inspect paths, not secrets embedded in ordinary file contents.

A write grant covers unit proposals only. The mutation guard refuses test paths it recognizes, including `test/**`, `tests/**`, `test_*.py`, and `conftest.py`, plus paths protected by the run. It also refuses interpreter or test startup files such as `pytest.ini`, `tox.ini`, `setup.cfg`, `pyproject.toml`, `noxfile.py`, `sitecustomize.py`, `usercustomize.py`, and `*.pth`.

The broker accepts requests up to 16 KiB, URLs up to 2,048 UTF-8 bytes, response bodies up to 8 MiB, and deadlines up to 20 seconds. The `Accept` value and combined response headers are each limited to 8 KiB. A domain allowlist can contain at most 64 entries. A unit invocation has at most four payload phases; exceeding that returns `MUTATION_PROTOCOL`. Custom `SSL_CERT_FILE` and `SSL_CERT_DIR` settings are not passed to the broker, so a private CA configured only through those variables may fail verification.

## What the model sees

When enabled, StateFrame adds one fixed `mutate` tool. Its fields are `op`, optional `id`, and optional `args`. The operation is selected from `catalog`, `use`, `enter`, `submit`, `leave`, `watch`, `unwatch`, `inspect`, and `retain`. The `args` value is a JSON object encoded as a string, up to 16,000 UTF-8 bytes.

The normal mutation panel shows the active level, ready unit names, latest result, and optional observer output. During an authoring episode, the frame shows its goal, remaining sends, latest diagnostic, and a short draft-source preview. Catalog details are requested on demand with `mutate(op="catalog")`, rather than copied into every request.

While an authoring episode is open, callable tools are narrowed to mutation and read-oriented operations. Direct effect tools are withheld. Mutation replies are marked as untrusted observations and cannot satisfy a check, completion requirement, or done gate.

## Prebuilt catalog

Use `stateframe mutations list` to list the shipped catalog and `stateframe mutations show UNIT` to inspect a unit’s contract and provenance. The host checks catalog digests again when admitting a unit.

| Unit | What it does |
|---|---|
| `file_facts` | Report bounded file size, line count, encoding, newline, BOM, and binary facts. |
| `structured_peek` | Preview CSV, TSV, JSON, JSONL, TOML, or INI structure. |
| `archive_list` | List ZIP or tar members and preview one bounded text member without extraction. |
| `sqlite_peek` | Read SQLite schema and up to twenty rows per table from a main-file snapshot. |
| `snapshot_diff` | Compare two bounded snapshots and return stable hashes plus changed lines. |
| `text_grep_struct` | Find matches using a bounded safe-regex subset, with line and column. |
| `log_watch` | Count literal patterns in a bounded log tail and classify rotation or truncation. |
| `html_extract` | Fetch HTML through the host broker and extract bounded text, links, and tables. Requires the public-web grant. |

## Worked example: preview a CSV

Suppose `data/scores.csv` contains:

~~~csv
name,score
Ada,10
Lin,9
~~~

Start a run with the prebuilt level and a mutation read restriction for that directory:

~~~sh
stateframe run --repo . --task "Summarize data/scores.csv" \
  --mutations prebuilt --mutation-read 'data/**'
~~~

The model can call the `mutate` tool with these fields:

~~~json
{
  "op": "use",
  "id": "structured_peek",
  "args": "{\"input\":{\"input_id\":\"in0\",\"format\":\"csv\"},\"inputs\":[\"data/scores.csv\"]}"
}
~~~

StateFrame admits the path, snapshots its bytes as `in0`, and passes the JSON input to the helper. For the sample above, the observation includes:

~~~text
[untrusted mutation observation structured_peek r1 sha256:<12 hex> in:sha256:<12 hex>]
{"columns":["name","score"],"error":"","error_column":0,"error_line":0,"format":"csv","row_count":2,"rows":[["Ada","10"],["Lin","9"]],"top_level_type":"table","truncated":false}
~~~

The digest values vary by unit source and input. The result is still an observation, so use ordinary checks and review for correctness.

## Author and repair a unit

Authoring is available at the `authored` level. Open an episode with a goal of at most 512 characters and up to 16 input paths, each at most 256 characters:

~~~text
mutate(op="enter", id="byte_counter", args="{\"goal\":\"count bytes in data/sample.bin\",\"inputs\":[\"data/sample.bin\"]}")
~~~

The episode admits those paths under the same read grants and snapshot limits as `use`. Input `i` is named `in<i>`. The handler must define `handle(input, ctx)` and read admitted files through `ctx.input_bytes(input_id)` or `ctx.input_text(input_id)`; an unambiguous input file name is accepted in place of its ID.

For a three-byte `data/sample.bin` containing `abc`, this short-form submission smoke-tests a byte counter:

~~~text
mutate(op="submit", args="{\"code\":\"def handle(input, ctx):\\n    return {'bytes': len(ctx.input_bytes(input['input_id']))}\\n\",\"expect\":{\"bytes\":3}}")
~~~

The short form accepts `code`, optional `summary`, `input_schema`, `output_schema`, `expect`, and `inputs`. With no episode inputs, the smoke input list is empty unless `submit` supplies `inputs`. A later `use` call without `inputs` reads the files the unit was smoke-tested on. The full `{"manifest": {...}, "files": {...}}` form supports explicit schemas and multiple modules. A bundle may contain one to eight Python modules, must include at least one smoke row with an expected object result, and must fit within 64 KiB including its manifest. Each input or output schema is limited to 8 KiB and nesting depth 8. A summary is limited to 2,048 UTF-8 bytes; a reader can list up to 32 unique format names, each at most 32 characters. Static validation and smoke execution run in the mutation sandbox. A smoke pass proves only the supplied example.

A repair submits a new immutable revision whose parent is the previous source digest. The short form sets the revision and parent from the current unit automatically. A refusal includes a bounded diagnostic and, when possible, the next revision and remaining sends so the unit can be corrected. Diagnostics are limited to 1,200 characters, escape control characters, and hide host paths; sandbox paths may remain visible. Static errors identify source locations, handler errors include a bounded recent trace and stderr tail, and a smoke mismatch identifies the first differing result path.

A unit never receives raw network or workspace write authority. It can request public fetches through the host broker and return edit proposals in its terminal result. A later fetch phase reruns the handler from its start with the same admitted inputs and recorded responses available, so CPU work may repeat. Smoke validation never applies edit proposals.

## Budgets

Budgets are durable across the whole run and resume. The run and chat commands do not expose budget overrides; they use the defaults below. The maximum column is the budget validator’s ceiling, not a CLI tuning setting.

| Budget | Default | Validator ceiling |
|---|---:|---:|
| Authoring episodes | 3 | 6 |
| Revisions per unit / total | 3 / 8 | 5 / 16 |
| Mutation-frame sends per episode / run | 4 / 10 | 4 / 20 |
| Payload invocations | 40 | 200 |
| Payload CPU / wall time | 60 / 120 seconds | 300 / 600 seconds |
| Authored source bytes | 256 KiB | 1 MiB |
| Output bytes | 64 KiB | 256 KiB |
| Broker fetches / fetched bytes | 10 / 8 MiB | 50 / 32 MiB |
| Broker CPU / wall time | 60 / 120 seconds | 300 / 600 seconds |
| Committed edit files / bytes | 16 / 256 KiB | 64 / 1 MiB |
| Journal bytes | 2 MiB | 8 MiB |

Each edit proposal set can contain at most 16 files. One mutation result is limited to 64 KiB. A payload is capped at 256 MiB of memory, 120 seconds of running time, and 300 seconds of CPU time. Its wall clock starts when the sandbox is released to run the payload. Sandbox startup has a separate 10-second window, scaled by the existing execution supervisor timeout setting and host load. A startup timeout returns `REFUSED MUTATION_STARTUP_TIMEOUT` after cleanup; it does not count as a payload timeout. Reaching a run budget returns a `REFUSED MUTATION_BUDGET` observation and disables mutations for the rest of that run. The main task can continue.

For an edit proposal, the host checks the grant, edit scope, protected paths, parent directory identity, and base SHA-256 for the entire set before writing. It records pre and post bytes durably before the first replacement, checks each target again just before replacing it, and attempts to roll back its replacements after an ordinary write failure. An interrupted commit is reconciled on resume; an unprovable rollback blocks recovery.

## Watches

`watch` schedules an offline, read-only unit at committed turn boundaries. The run must also have no mutation network or write grant. Its `args` object requires `input`, `inputs`, `every_turns`, and `max_ticks`; the two schedule values must be from 1 to 30. Each tick snapshots the inputs again and records file rotation, truncation, or deletion. Repeated output is coalesced. Use `unwatch` with the watch ID to stop it.

An interrupted run records unfinished watches as stopped. Resume replays completed ticks from the tape without launching them again.

## Retention, promotion, and hooks

Units are available in the current run after admission. Retaining an authored unit with `stateframe mutations retain RUN UNIT --scope project|user` stores its source bundle under a content digest and adds a project or user reference. Project scope uses the resolved repository path and its first Git commit; run project-scoped management commands from the same repository path. `stateframe mutations remove UNIT --scope project|user` removes the reference. `stateframe mutations prune` sweeps abandoned runtime roots and collects objects with no remaining reference. Add `--dry-run` to preview cleanup.

Retained units are available for listing, promotion, and hook attachment, but this release does not automatically load them as general-purpose `mutate(use)` units in later runs. Use `stateframe mutations list --scope user` or `stateframe mutations list --scope project` to inspect retained units.

The run/chat flags `--mutation-retain run|project|user` and `--mutation-audit source|digest` are accepted and recorded in the run policy, but the runtime does not use them to automatically persist units or filter journal source. To persist an admitted unit, use the explicit `stateframe mutations retain` command. The model’s `mutate(op="retain")` operation records a run-audit marker only.

`stateframe mutations promote UNIT [--out DIR]` exports a prebuilt or retained unit as a disabled powerup and invokes the installed powerup consumer. The output directory must not exist. If the consumer does not advertise and validate isolated-unit support, the command refuses; it does not leave an export directory as a fallback.

A repeated `--mutation-hook UNIT:EVENT` attaches a retained offline, read-only unit to `post_tool` or `session_end`. Hooks require the `authored` level. Observers receive limited host facts, not tool arguments, output, or model context. Each trigger is limited to one second, all observers share a ten-second run budget, output is bounded to 512 bytes, and a unit is disabled after three failures. Hooks cannot veto the run, and a mutation call does not trigger a mutation hook. A Linux host that fails qualification records `platform_unqualified`, disables that observer, and continues the ordinary run.

A mutation interrupted by a crash is refused on resume with `MUTATION_INTERRUPTED`. Rerun the mutation if its result is still needed.

Hook triggers write `HOOK_TRIGGER_INTENT` and finish with `HOOK_TRIGGER_COMPLETE`, `HOOK_FAILURE`, or `HOOK_REFUSED`; resume writes `HOOK_TRIGGER_INTERRUPTED` for an unfinished trigger, and disabled observers write `HOOK_DISABLED`. A disabled observer stays disabled when its retained unit is attached again during resume. A host qualification failure writes `HOOK_REFUSED` once per affected hook and event in that attempt with code `MUTATION_PLATFORM_UNQUALIFIED`, and that record counts as the trigger's completion on resume.

## Journal, replay, and cache behavior

Mutation evidence is stored under `<run-dir>/mutations/`, with events in `journal.jsonl` and content-addressed evidence blobs in `evidence/`. The tool tape is authoritative for model-visible mutation results. A completed call replays its recorded observation without rerunning the payload. The journal holds the unit, broker, budget, hook, and edit evidence needed to rebuild state and reconcile effects.

Mutation-enabled resume reconciles mutation evidence first and replays recorded model exchanges. It does not restore engine checkpoints. Committed edit postimages are carried into the fresh replay workspace, preserving them for later checks and the final candidate. Unknown external fetch outcomes normally refuse resume with `MUTATION_BROKER_UNKNOWN`. `stateframe resume RUN --replay-external` accepts the lost mutation fetch outcome as interrupted; it does not dispatch that fetch again. A prepared workspace edit is recovered only when recorded file and directory identities and digests still match.

With mutations off, request bytes and tool order remain unchanged. When enabled, requests include the `mutate` tool and mutation frame. Host qualification is cached under `<state-home>/mutations/qualified.json`. State home is `STATEFRAME_HOME` when set, otherwise `$HOME/.stateframe`; set `STATEFRAME_HOME` if a home directory cannot be determined. Qualification is bound to the kernel, architecture, Python runtime, and mutation closure; a changed binding requires qualification again. `stateframe mutations selftest` runs a fresh qualification check.

## Safety and limits

Linux mutation payloads run under bubblewrap and a seccomp boundary. They cannot fork, exec, or create threads. The sandbox mounts the interpreter, standard library, and required shared libraries; it excludes `site-packages`. It exposes `/etc/ld.so.cache` read-only when present so the dynamic loader can locate shared libraries; the rest of `/etc` stays unmounted. If a standard-library extension needs a missing library, qualification reports the exclusion and a unit importing it fails on import.

The host broker is a separate disposable process and has network access only when the public-web grant is active. It mounts resolver files and Python's compiled OpenSSL CA file and directory read-only, validates every DNS answer, and retries only vetted addresses within the fetch deadline. A write proposal is applied only by the host after terminal success and verified cleanup. If Linux qualification fails, the run can still start, but mutation calls answer `REFUSED MUTATION_PLATFORM_UNQUALIFIED` rather than falling back to a weaker boundary. Use `stateframe mutations doctor` for host facts and `stateframe mutations selftest` to run qualification probes. A run start or resume sweeps that run’s runtime root; `prune` attempts cleanup across stale roots. Unprovable runtime identities are left unresolved and reported. Recovery after an interrupted run has not been demonstrated.

Mutation results do not replace ordinary tests, review, or completion checks. A smoke pass proves protocol behavior for its fixture, not general correctness.

## CLI reference

Run and chat accept:

~~~text
--mutations off|prebuilt|authored
--mutation-network none|public-web                 default: none
--mutation-domain DOMAIN                           repeatable
--mutation-write GLOB                              repeatable
--mutation-read GLOB                               repeatable
--mutation-retain run|project|user                 default: run
--mutation-audit source|digest                     default: source
--mutation-hook UNIT:post_tool|session_end          repeatable
~~~

`--mutation-domain`, `--mutation-write`, `--mutation-read`, and `--mutation-hook` default to no entries. The exact hook events are `post_tool` and `session_end`. The mutation network grant requires the separate `--shell-network full` setting.

Management commands:

~~~text
stateframe mutations list [--run RUN | --scope user|project] [--json]
stateframe mutations show UNIT [--json]
stateframe mutations enable-level off|prebuilt|authored [--profile PROFILE]
stateframe mutations selftest [--json]
stateframe mutations doctor [--json]
stateframe mutations retain RUN UNIT --scope project|user
stateframe mutations remove UNIT --scope project|user
stateframe mutations promote UNIT [--out DIR]
stateframe mutations prune [--dry-run] [--json]
~~~

With no selector, `list` shows prebuilt units. `list --run` shows units admitted by that run; `list --scope` shows retained units for the current repository or user scope.

## Troubleshooting

| Message or symptom | What to check |
|---|---|
| `MUTATION_PLATFORM_UNSUPPORTED` | Mutation execution is Linux-only in this release. Only `list`, `show`, and `doctor` are available on other platforms. |
| `MUTATION_PLATFORM_UNQUALIFIED` | Run `stateframe mutations doctor` and `stateframe mutations selftest`. Check the reported bwrap, seccomp, namespace, and Python runtime facts. |
| Mutations are not offered | Check the selected level, `STATEFRAME_MUTATIONS`, profile value, and any explicit `--mutations off`. Use `stateframe mutations list` to see available prebuilt units. |
| `--mutation-network public-web` is refused | Add `--shell-network full`. Check the broker domain list and fetch budgets. |
| `REFUSED MUTATION_BUDGET` | The run reached a durable invocation, time, byte, send, journal, or edit limit. Resume does not reset charges. |
| Resume reports `MUTATION_BROKER_UNKNOWN` | Inspect the run evidence. If you accept the unknown fetch result as interrupted, resume with `--replay-external`; the fetch is not retried. |
| A retained unit cannot be used with `mutate(use)` | Retained units are not automatically loaded as general-purpose units in later runs. Attach an eligible unit as a hook or promote it through a supported powerup consumer. |
| A mutation hook does not attach | Check that the unit is retained, offline, read-only, and named with `--mutation-hook UNIT:post_tool` or `--mutation-hook UNIT:session_end`; hooks require `--mutations authored`. |
| `stateframe mutations promote UNIT` refuses | The installed powerup consumer must advertise and validate isolated-unit mutation support. |
| `stateframe mutations prune --dry-run` is refused | The installed runtime sweeper must support dry-run. The CLI refuses before cleanup if it does not. |

## Refusal codes

Mutation tool calls return a refusal observation with one of these codes. Run or resume may also stop with a recovery code when saved evidence cannot be reconciled safely.

| Code | Meaning |
|---|---|
| `MUTATION_ADMISSION` | A unit or its requested inputs failed admission checks. |
| `MUTATION_BAD_ARGUMENTS` | An operation, field, or JSON argument is missing or invalid. |
| `MUTATION_BUDGET` | A durable run budget or a per-call output limit was reached. |
| `MUTATION_CATALOG` | The prebuilt catalog could not be validated. |
| `MUTATION_DIGEST` | A unit source digest did not match its recorded value. |
| `MUTATION_EDIT` | An edit proposal failed path, scope, or base-state validation. |
| `MUTATION_EXECUTION` | A unit failed, timed out, exceeded output limits, or replayed a recorded failure. |
| `MUTATION_STARTUP_TIMEOUT` | The sandbox did not reach payload release within its startup window. The launch is stopped and cleaned up. |
| `MUTATION_FRAME` | The operation is not allowed in the current authoring episode. |
| `MUTATION_GRANT` | The requested network, write, or retention capability was not granted. |
| `MUTATION_LEVEL` | The operation requires the `authored` level. |
| `MUTATION_PARENT` | A repair does not identify the current unit revision as its parent. |
| `MUTATION_PLATFORM_UNQUALIFIED` | Linux sandbox qualification failed. Doctor output may identify a failed or timed-out qualification check. |
| `MUTATION_PROTOCOL` | A unit exceeded four payload phases or did not return a valid terminal result. |
| `MUTATION_SCOPE` | The workspace needed by the operation is unavailable. |
| `MUTATION_UNKNOWN` | The requested watch ID is not active. |
| `MUTATION_WATCH` | A watch could not take a tick, often because an input changed or became unavailable. |
| `MUTATION_UNAVAILABLE` | The mutation runtime was not available when the tool call arrived. |
| `MUTATION_PLATFORM_UNSUPPORTED` | Mutation work was requested on a platform other than Linux. |
| `MUTATION_CONTAINMENT` | Cleanup or an earlier launch could not be verified, so execution or recovery stops. |
| `MUTATION_INTERRUPTED` | A mutation or watch was interrupted. Resume serves this refusal instead of running the operation again. |
| `MUTATION_BROKER_UNKNOWN` | A public fetch started but its outcome was not recorded. Resume normally refuses; `--replay-external` accepts it as interrupted without retrying the fetch. |
| `MUTATION_BROKER_INCONSISTENT` | Saved public-fetch evidence is inconsistent. |
| `MUTATION_CLEANUP_UNKNOWN` | A committed edit lacks verified cleanup evidence. |
| `MUTATION_EFFECT_INDETERMINATE` | Saved edit evidence cannot prove whether workspace changes are safe to keep. |
| `MUTATION_EVIDENCE_TAMPERED` | Saved mutation evidence is invalid or does not match its recorded result. |
| `MUTATION_TAPE_CORRUPT` | The recorded mutation call sequence is malformed or conflicting. |
| `MUTATION_LEDGER_CORRUPT` | Saved launch records are invalid. |
| `MUTATION_LEDGER_WITHOUT_INTENT` | A saved launch record has no matching mutation call. |
| `MUTATION_JOURNAL_WITHOUT_INTENT` | A saved mutation event has no matching call. |

Qualification details such as `MUTATION_PLATFORM_UNQUALIFIED_FAILURE` and `MUTATION_PLATFORM_UNQUALIFIED_TIMEOUT` identify which sandbox check failed; the overall mutation refusal is `MUTATION_PLATFORM_UNQUALIFIED`.
