# Running StateFrame

## Commands

| Command | Does |
|---|---|
| `stateframe run --repo R --task "..."` | Runs one task on a safe copy of `R`. |
| `stateframe resume <run-dir>` | Continues a stopped run (Ctrl-C, `kill -9`, crash, lost server). |
| `stateframe inspect <run-dir> --send N --frame` | Shows the frame, messages or exact bytes of request `N`. |
| `stateframe chat --repo R` | Interactive session: tasks, follow-ups, `/diff`, `/undo`, `/apply`. |
| `stateframe hooks trust --repo R` | Review and trust the exact `.stateframe/hooks.json` configuration for repository `R`. |

## Common flags

- `--endpoint`, `--model` (or `STATEFRAME_ENDPOINT`, `STATEFRAME_MODEL`): the server and model. Loopback only unless `STATEFRAME_ALLOW_REMOTE_ENDPOINT=1`.
- `--check PATH` (repeatable): pytest files selected for the default check; for a generic command, extra existing check files to protect.
- `--test-cmd COMMAND`: run repository tests before, during and after a task. Without it, StateFrame uses `python -m pytest -q` when pytest files exist. A non-pytest command passes only on exit 0. Missing commands, exits 126/127 and timeouts do not pass.
- `--check-cmd COMMAND`: an explicit non-test command to verify a task, such as `npm run build`. It uses the same sandbox, output bounds and exit-code rules.
- `--runner-kind auto|pytest|generic`: detect pytest from the command by default, or choose the kind explicitly for a wrapper. Only pytest supports per-test selectors; generic commands run as a whole.
- `--protect-tests off|task|all`: `off` (default) permits test edits and reports them; `task` locks tests named by the command or selectors and tests failing at the start; `all` locks every pre-existing test as in 0.2. For pytest, `task` and `all` also refuse new collection hooks, test-package initializers, pytest configuration, and Python startup hooks that could change the check result. If a generic command has no explicit test file selector, `task` locks every existing test file, including when the starting check passes. The result line reports this fallback.
- `--protected-test-glob GLOB`: add a repository-relative pattern for test or check resources. Repeat as needed; it is locked when protection applies.
- `--target`, `--edit`: in repositories over 200 source files, the files the change may touch; the rest is read-only context. Without them, StateFrame chooses up to 12 files: the ones the task names, then the modules imported by the checks you chose (`--check`, or test files named in `--test-cmd`), and one more import hop. It prints the list at startup. It stops and asks for the flags when nothing points at a file, or when more than 8 test files are selected (`pytest tests/` says nothing about where the fix lives).
- `--edit-scope widen` (or `STATEFRAME_EDIT_SCOPE_WIDEN=on`, in the environment or a profile): in a scoped run, an edit, file deletion or move can add another existing product file to the editable set. By default, attempts to change an existing file outside the editable set return `NOT_EDITABLE` before writing, with a hint to rerun using `--edit PATH` or to enable widening. The file must be a regular file inside the repository with no symlink or hard link, and cannot be a test, named check or hook input. The starting directory does not limit existing files; new-file roots stay fixed. At most 24 files are admitted, in action order. Each admission records the pre-edit digest in the durable journal. The host requires that record and rechecks protection and identity before applying a changed or deleted file. A sealed candidate containing an unapplied change to an existing file is reported as an error. First shell writes outside scope remain refused. Off by default; with it off, requests are unchanged. A run with widening on resumes by replay, never from a checkpoint.
- `--in-place`: apply a passing result to your repository.
- `--run-dir`: where to keep the run.
- `--max-turns N`, `--timeout S`: optional caps; by default there are none.
- `STATEFRAME_QUEUE_WAIT_MAX_S` (environment or profile): finite positive cap in seconds for a silent request whose selected endpoint keeps answering `/slots` or `/health`. Default 1800 seconds, shortened by the run deadline. The ordinary per-request cap resumes once a response starts. Queue waits are journaled and summarized with one warning line.
- `STATEFRAME_SEND_STALL_S` (environment or profile): maximum time with no request-body bytes sent, from 10 to 600 seconds; default 90. A stalled upload is aborted and can retry on a new connection under the ordinary transport policy. Queue-wait extension starts only after the full body is sent. Each send stall records sent and total bytes, stalled seconds, and the endpoint origin without credentials.
- `--mutations off|prebuilt|authored` plus `--mutation-network`, repeatable `--mutation-domain`, `--mutation-read`, `--mutation-write` and `--mutation-hook UNIT:EVENT`, `--mutation-retain` and `--mutation-audit`: opt into sandboxed mutation units and their separate grants. They are off by default. See [Mutation units](mutations.md).
- `--test-env KEY=VALUE`: environment for your tests (not for engine settings).
- `--pytest-plugins installed`: let pytest load installed plugins.
- `--shell off|read-only|standard|unrestricted` (default `standard`): command policy. `--shell-network none|loopback|full` defaults to `loopback`.
- `--shell-ask` / `--no-shell-ask`: confirmation for commands outside the inspection and user allow sets. Default off for `run`, on for `chat`.
- `--shell-allow PROGRAM`, `--shell-deny PROGRAM`, `--shell-read PATH`: repeatable program and readable-root settings.
- `--allow-source-tree` (run): authorize a StateFrame editable source tree found in the `--test-python` environment. StateFrame validates the `src/stateframe` layout and adds that tree as a read-only shell root for the run.
- `--shell-sandbox on|off` (default `on`): `off` requires `--shell unrestricted`, prints a host-access warning and is recorded as host authorization.
- `--run-env` / `--no-run-env`: run-local dependency environment; default on when shell is enabled, otherwise off.
- `STATEFRAME_ENV_SEED_MAX_GB` (environment): soft limit for the shared environment seed store in decimal GB, default 10. Set it to `0` for unbounded storage. Eviction removes old seeds after a successful clone or during `stateframe gc`; rehydration rebuilds from the recorded distributions instead of copying when its seed is gone.
- `--git-tool` / `--no-git-tool`: read-only Git tool, on by default when the source has Git metadata.
- `--web` / `--no-web`: web fetch, off by default. `--web-allow DOMAIN` repeats to restrict hosts.
- `--tools FILE`: repeatable trusted custom command definitions in `stateframe-tools/1` JSON.
- `--hooks FILE`: repeatable lifecycle hook definitions in `stateframe-hooks/1` JSON. Project hooks at `.stateframe/hooks.json` require `stateframe hooks trust --repo R`; `--no-project-hooks` skips them for one `run` or `chat` command.
- `--skill NAME_OR_PATH`: repeatable selected skill bundle. `--trust-skill-tools` offers its optional command definitions. See [Skills](skills.md).
- `--powerup NAME[=MODE]|PATH`: repeatable powerup selection for this command; MODE is `off`, `offer`, `on` or `auto`. `--powerup-set NAME.KEY=VALUE` overrides one setting, and `--no-powerups` turns every powerup off. See [Powerups](powerups.md).
- `--instructions[=FILE]`: project instructions, off by default. With no value, the repository's `AGENTS.md`, else `CLAUDE.md`; with a value, that file. `--no-instructions` states the default explicitly. See [Project instructions](#project-instructions).
- On `resume`, `--replay-external` explicitly repeats recorded external operations before continuing.

Hook files named with `--hooks` are explicitly trusted by that command. Project hook files need a separate digest trust step, and edits to the file or its pinned scripts require trust again. The [hooks guide](hooks.md) describes the event contract and the sandbox boundary.

## Hosted and OpenAI-compatible endpoints

StateFrame chooses the dialect for each selected endpoint, including routed endpoints. It makes bounded `GET /props` and `GET /slots` requests to that endpoint's origin using the request's API key and does not follow redirects. A 200 `/props` response identifies llama.cpp only when it has `default_generation_settings` plus `n_ctx` or `model_path`. A populated `/slots` array with distinct nonnegative integer IDs, positive integer `n_ctx`, and boolean `is_processing` also proves llama.cpp when `/props` is missing or unrecognized. Without either proof or the explicit `STATEFRAME_PROVIDER=llama.cpp` override, StateFrame selects `openai-strict`. The first run line and final summary show each chosen dialect, the reason, and how reasoning controls were handled.

`STATEFRAME_PROVIDER=llama.cpp` is the existing operator override for a trusted llama.cpp-compatible endpoint whose `/props` cannot be reached or classified. It selects the llama.cpp request shape even when `/props` does not prove it. Slot features still require a successful `/slots` response; StateFrame never invents slot 0.

For `openai-strict`, the final request is limited to the documented [OpenAI Chat Completions create schema](https://github.com/openai/openai-python/blob/main/src/openai/types/chat/completion_create_params.py), including nested messages, tools, and JSON schemas. llama.cpp-only fields such as `reasoning_budget_tokens`, `reasoning_mode`, `chat_template_kwargs`, `cache_prompt`, `id_slot`, `top_k`, `min_p`, samplers, and grammar are removed. `n_probs` becomes `logprobs: true` and `top_logprobs`. Set `REACT_REASONING_EFFORT` in the shell or profile to `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max` only when the selected provider accepts OpenAI's `reasoning_effort` field; StateFrame sends that setting and otherwise omits reasoning controls. A profile with global reasoning-off is refused unless `REACT_REASONING_EFFORT=none` explicitly confirms endpoint support.

The llama.cpp `/tokenize` payload is sent only after that tokenizer origin proves the llama.cpp dialect. A strict generation endpoint uses StateFrame's calibrated byte-based estimate unless `LLAMA_CPP_TOKENIZE_URL` points to a separate tokenizer origin that proves llama.cpp; in that case the tokenization request goes to that origin.

If a provider rejects a request with a recognized context-limit HTTP 400, StateFrame refits that request to smaller byte and token limits and retries once if time and an attempt remain within the existing deadline. When the provider reports a usable context limit, later requests to the same endpoint, model and credential use that lower limit too. An unrelated HTTP 400 is not retried this way. If the protected minimum will not fit, or the smaller request also exceeds the provider's context limit, the run stops with a capacity reason and retains its partial result.

Features that depend on llama.cpp capabilities stop before the model request when those capabilities are unavailable. This includes explicitly selected prompt-cache reuse, KV chunk reuse and its slot save/restore/erase calls, self-plan prefill, true prompt continuation, routed static slots, history-compaction cache/slot, and public-review `cache_prompt`. The shipped profile's automatic prompt-cache pin degrades when `/slots` is unproven: StateFrame disables the optimization, prints a `cache: prompt-cache pin dropped` notice, records one `provider_feature_degraded` journal event, and shows `prompt-cache slot: degraded` in the final summary. Explicit `--slot` and route slot selections still require proof and refuse before dispatch. History compaction with `cache_prompt=true` also needs an explicitly selected slot reported by `/slots`. Study capacity preflight is restricted to a loopback llama.cpp endpoint. Template-conformance calls to `/apply-template` and `/tokenize` require the selected endpoint to prove the llama.cpp dialect. The error names the feature and missing proof. The scripted endpoint answers llama.cpp's `/props` shape by default for local identity runs; its optional `/slots` response is enabled only when a script supplies slots. The three bundled examples supply a reported 65,536-token slot for local identity checks.

## Project instructions

`--instructions` reads the repository's `AGENTS.md` (or `CLAUDE.md` when there is no `AGENTS.md`) from the repository root; `--instructions=FILE` reads that file instead, which may live outside the repository. Write the value with `=` (or put the flag last) so the next argument is not taken as the file name.

- The file must be a regular file of at most 1 MiB. An in-repository symlink is followed only when its target stays inside the repository, and a file the working copy excludes (for example a secret-named path) is never shown.
- At most 6,144 bytes are shown, cut at a line boundary, with a visible note giving the shown and total byte counts. The model can read the rest of a file inside the repository with its normal read tools.
- The text is appended once to the end of the engine's system message, after every engine rule and before the task, so the stable request head (and the server's prompt cache) is unchanged from one request to the next. Hook start context, the skill catalog and MCP tools keep their usual places.
- In the last-resort compaction for an otherwise unfittable request, the engine shortens its whole system message to one line, and the instructions are dropped with it.
- The text is framed as untrusted repository content. It cannot grant or remove tools, widen the sandbox or network, change protected tests or completion checks, or override engine rules.
- The run records the file's SHA-256. `resume` refuses before replay when the file changed or is missing.
- Without `--instructions`, or with it when no file exists, requests are byte for byte the same as without this feature.
- The plain loop and lean mode do not support project instructions: an explicit `--instructions=FILE` is refused and a discovered file is skipped with a note.

## Which Python runs your tests

The engine runs on StateFrame's own Python. Pytest runs on your project's: `--test-python PATH` if given, else the repository's `.venv/`, `venv/` or `env/`, else `python3` on PATH. StateFrame prints the choice and stops if that Python cannot import pytest. A generic check does not import pytest. `--python PATH` sets one interpreter for both the engine and Python checks.

With no tests or explicit command, StateFrame uses the first declared check it finds: Python project metadata selects `python -m compileall -q` over product Python sources; `package.json` selects `typecheck`, then `build`, then `lint` using the lockfile's package manager or npm; `go.mod` selects `go build ./...`; `Cargo.toml` selects `cargo check`; a Makefile selects `check`, then `build`. If none exists, an agent-written Python check can provide evidence only when it fails on the original and passes on the candidate in the check sandbox. With no check at all, the result says `unverified: review the diff`. Result labels distinguish tests, check commands, builds and agent-written checks. An explicit `--test-cmd` in a repository with no discovered test files is labeled a check command, not a test pass.

Test discovery covers existing `test_*.py`, `*_test.py`, `conftest.py`, `*.test.*`, `*.spec.*`, `*_test.go`, `test*.sh`, `check*.sh`, and files under `tests/`, `test/` or `__tests__/`. In `all` mode, a script directly named in a generic command, `package.json`, `Makefile`, `Cargo.toml`, and Vitest, Jest and Playwright config files are protected too. Pytest's `all` mode retains its 0.2 protection. In `off` and `task`, each changed test is reported separately. Potentially removed assertions or tests, constant-true assertions, dropped Python predicate checks, new skip/xfail/only/todo markers and loosened comparisons are flagged by file and test. Python is checked with syntax analysis; other languages use a line heuristic, which can miss semantic changes. A run with a weakening finding includes it in the result line and summary.

For example, `stateframe run --repo ~/code/shop --task "Fix the failing check" --test-cmd 'npm test' --protect-tests all --protected-test-glob 'scripts/verify-*'` uses the package test command and locks matching scripts alongside the defaults. The command must be executable in the check sandbox. Its exit code decides whether the check passed; the result says `verified by tests` only when the completed command prints a recognized positive test-count receipt. Otherwise it says `verified by check command`. Per-test selectors are available only with pytest. In generic mode, or with pytest and `--protect-tests task` or `off`, an agent can report a wrong judging test with its path, name, assertion, reason and observed behavior; the run stops without claiming a pass. The pytest `all` done-tool schema remains frozen and does not advertise `test_wrong`.

Example, a project with its own virtual environment somewhere else:

```bash
stateframe run --repo ~/code/shop --task "..." --test-python ~/venvs/shop/bin/python
```

## Context size

| Variable | Default | What it does |
|---|---|---|
| `STATEFRAME_CONTEXT_TOKENS` | 65,536 | The server slot one run uses. |
| `STATEFRAME_PROMPT_BUDGET` | 33,344 | The most prompt tokens one request may use. |

Set them in the shell you start StateFrame from (or in a custom profile). With neither set, nothing changes.

**If you set only `STATEFRAME_CONTEXT_TOKENS`, the prompt budget scales with it automatically (about 51%: 33,344 / 65,536, rounded down); the rest of the slot is reserved for the model's reply.**

| Slot (`STATEFRAME_CONTEXT_TOKENS`) | Prompt budget |
|---|---|
| 32,768 | 16,672 |
| 65,536 | 33,344 |
| 131,072 | 66,688 |

```bash
STATEFRAME_CONTEXT_TOKENS=32768 stateframe run --repo ... --task "..."
```

The run prints `prompt budget: 16,672 tokens; 32,768-token server slot required (budget derived from slot at 33344/65536)`.

Watch for:

- **The server's slot must equal the context size exactly.** The engine reads the slot size from llama.cpp's `/slots` (or `/props`) and refuses a run whose server reports any other size, larger too: `STATEFRAME_CONTEXT_TOKENS=32768 requires a live server slot of 32768 tokens; observed 65536. Set the server slot to 32768 or unset the context setting.` A server that reports no slot size cannot be used with these settings.
- **llama.cpp `--kv-unified` reports the total cache as every slot's size**, so it never matches a per-run size; with these settings, start the server without `--kv-unified`: `-c` = slot x `--parallel` (for example `-np 4 -c 131072` for four 32,768-token runs). Leaving both settings unset works with either layout.
- **An explicit `STATEFRAME_PROMPT_BUDGET` can only be lowered**, never above the derived value, because the rest of the slot is the reply reserve: `STATEFRAME_PROMPT_BUDGET=20000 exceeds the maximum 16672 tokens for STATEFRAME_CONTEXT_TOKENS=32768 after reserving 16096 tokens. Set it to at most 16672 or unset it.`
- Values must be positive whole numbers: `STATEFRAME_CONTEXT_TOKENS=lots is invalid; minimum 1 token. Set it to a positive integer or unset it.`
- **A resumed run keeps the sizes it started with.** A different value on resume is refused: `STATEFRAME_CONTEXT_TOKENS=32768 differs from the run's start (unset); a resumed run keeps the context size it started with. Unset it to resume`.

Only the default 65,536-token slot is measured on long runs; other sizes are tested on short scripted runs.

## Per-step models

By default one endpoint and one model handle every request, and requests are byte-identical to a build without routing. You can instead send each kind of step to its own model: a small local model for reading, a stronger one for writing code. Each request is still built from the typed state frame, so switching models mid-run loses nothing; the frame is the handoff.

Requests are sorted into roles mechanically, from what the engine already knows, never by asking a model:

| Role | Which requests |
|---|---|
| `read` | The working default: navigating, reading, searching, running checks. |
| `plan` | The first request of a run, and the request right after a failed test run on the current code. |
| `edit` | Any step whose reply writes or changes files. |
| `finish` | A step whose reply says the work is done. |
| `decide2` | The engine's small internal decision request (rare). |
| `unit`, `unit.<class>` | Requests made inside parallel work units. |

The engine cannot tell before a reply whether a step will edit, because the tools it offers do not depend on the step's purpose. So `edit` and `finish` work by escalation: when a reply from another role's model edits files or says done, that reply is thrown away unrun, and the same step is asked again, with the same frame, of the `edit` (or `finish`) model. That costs one extra cheap request per edit step. Escalation never lowers a step: a write wins over done, done over read.

Start from the commented example:

```bash
stateframe config init     # writes ~/.stateframe/routes.toml with every line commented out
```

Uncomment only what you need. Anything you leave out inherits the run's `--endpoint`, `--model` and other settings:

```toml
# Reading on a small local model; everything else stays on the default server.
[roles.read]
endpoint = "http://127.0.0.1:18601/v1"
model = "minicpm5-1b"
slot = 0
cost_input_per_mtok = 0.0
cost_output_per_mtok = 0.0

[roles.edit]
cost_input_per_mtok = 0.60
cost_cached_input_per_mtok = 0.06
cost_output_per_mtok = 2.40
```

Settings are read, highest first, from `--route ROLE.FIELD=VALUE` (repeatable), `STATEFRAME_ROUTE_<ROLE>_<FIELD>` variables, a routes file (`--routes PATH`, `STATEFRAME_ROUTES`, or `~/.stateframe/routes.toml`), and the profile. TOML and JSON files both work. Unknown roles, unknown fields, wrong types and out-of-range numbers stop the run before it starts, with a message that names the setting.

```bash
stateframe run --repo ... --task "..." --route read.endpoint=http://127.0.0.1:18601/v1 --route read.model=minicpm5-1b
```

Watch for:

- **Each role's server is checked at start**, as far as it reports: its model must be among the models it lists, a pinned `slot` must be one of the slots it reports, and its slot must be at least the run's context size (65,536 tokens by default). A server that lists no models or reports no slots is not checked on that point.
- **Mixing local and remote is per role.** A remote endpoint needs `allow_remote = true` on that role (or `STATEFRAME_ALLOW_REMOTE_ENDPOINT=1` for all) and prints a warning that your code will leave the machine. Routed requests may follow redirects. A redirect to a remote host is refused unless that role allows remote endpoints; a redirect to a different scheme, host or port drops authentication, API key and cookie headers, including on later hops back to the original origin. Redirects keep the original request deadline. Routed requests and tokenizer calls ignore `HTTP_PROXY` and system proxy settings, so they go directly to the selected address or an allowed redirect destination. Caller-supplied proxy credentials are removed from direct requests.
- **Shell commands are not escalated.** A `shell` call counts as a `read` step, so it runs on whichever role's model asked for it, including commands that write files. A `read` model can therefore change the working copy through `shell` without the `edit` model seeing that step. The sandbox and the effect checks still apply and record every change. To keep a small read model from writing this way, run with `--shell read-only`.
- **Each role counts tokens on its own server** unless you set its `tokenizer`. If `LLAMA_CPP_TOKENIZE_URL` is set and a role uses another model, the run is refused until that role names its own `tokenizer` (or you unset the variable), so one model's counts never size another model's prompt.
- **A small read model must still follow the tool protocol.** Most requests in a routed run are `read` requests, so a model that answers without a tool call, or rambles to its token limit, costs more time than it saves. In one check, a 1B read model turned a 3-request fix into a 38-request one. Cap `read.max_tokens` (for example 2048) and try your read model on a short task before a long one.
- **Keys are named, never stored.** `api_key_env` names the variable that holds a role's key; the value is never written to the run directory.
- **The run prints its effective routes at start**, and a per-role table at the end: sends, discarded replies, prompt, cached and completion tokens, and an estimated cost from your rates (unknown when you give none).
- **A resumed run keeps its routes.** Changing a role's model, sampling or limits on resume is refused and names the setting. `stateframe resume --route` may move a role to a new endpoint address or rename its model, for when a server moves. Every replayed request must go to the role, endpoint and model it was recorded with, or resume stops before sending anything. The request that had no reply when the run stopped is checked against its rebuilt copy on the first resume; on later resumes only if the previous resume recorded that its own check passed. Otherwise, for example when a resumed run stops again before its check, StateFrame resumes by full replay without that check and prints a line saying so. If a rebuilt request's bytes differ, replay goes live early from that point, as it does without routes, still on the recorded route.
- **Routed runs do not restore checkpoints.** With checkpoints enabled, a routed run still writes them, but its resume always replays the recorded run.

## Profiles and switches

`STATEFRAME_HTTP_GZIP` and `STATEFRAME_PLAIN_GZIP` are `off` by default. Set either to `on` in a custom profile only when the endpoint is a relay that inflates gzip request bodies. A stock llama-server rejects them with HTTP 415. In the default run mode, a compressed request refused with HTTP 415 is resent once without compression within the original deadline.

A profile is a JSON file (`"schema": "stateframe-product-profile/1"`) whose `switches` map sets the engine's `STATEFRAME_*` switches for a run. The default is the one profile shipped in `src/stateframe/profiles/`. `--profile NAME_OR_PATH` uses another. Many switches ship disabled; turning one on is a profile edit. Engine settings come only from the profile: the engine child drops inherited `STATEFRAME_*` variables, `--test-env` refuses them, and the engine refuses any switch it has no reader for. Host-issued shell, network, sandbox, web and custom-tool authority comes from CLI flags, not profile switches. The shipped product profile uses the composed loop; lean mode is not supported by that profile. The plain loop refuses Git, web and custom-tool settings before execution and supports foreground shell only.

### Multi-file source and repeated reads

`STATEFRAME_CODE_BOOKMARKS` is on by default. Each request offers current editable product source before repeated-read and navigation bookkeeping. Files of at most 60 lines and 2 KiB may be shown whole when the request budget allows; larger tasks get selected windows. A coverage line for every editable product path says which lines are in view and whether that source changed since its last read when the comparison is available. When a current failing check has a usable source location, code bookmarks keep its test code and related product source in view. Set the switch to `off` in the profile to restore the 0.2.1 behavior byte for byte.

A completed batch made entirely of proved unchanged reads costs one no-progress watch unit instead of one per file. Mixed or unproved batches keep per-action accounting. Repeated unchanged batches still reach `STALLED`.

`STATEFRAME_READ_LOCKOUT=on` is an independent optional profile switch; it is off by default and requires bookmarks. After a completed batch of unchanged reads, the current failing check must still apply, editable bookmarked source must be visible, and an edit and check route must be available. The next request then hides source-reading routes. The restriction lasts at most two requests and releases sooner when new evidence appears or the repair route disappears. If the no-progress watch is one step from its limit, the run can stop before this restriction takes effect.

### Prompt cache: stable tool head

`STATEFRAME_STABLE_TOOL_HEAD` is on when unset. Set it to `off` in the shell that starts StateFrame or in a custom profile to restore per-step tool lists. Invalid values are refused.

The first request lists the run's already-filtered tool menu plus its registered offerable schemas. The current request's callable menu stays first and keeps its existing order; seed-only schemas follow by name. This makes the listing stable from request 1 when the run's offerable set is fixed. Schemas that become available later and were not in the initial set are appended in first-seen order. Only the current step's callable menu can run; a listed but unavailable action gets the usual "not offered for the current state" reply, and a short note names the currently callable actions. There is no separate seed switch. With `STATEFRAME_STABLE_TOOL_HEAD=off`, requests keep the per-step menu and the request bytes of releases before the stable tool head.

Measured live on Qwen3.6-35B-A3B served by llama.cpp, one run of each bundled example with the seed: request 2 hit the cache in every run, and the cached share of prompt tokens from request 2 on was 37-47%, against 18-22% for the same examples without the seed. The seed lists five tools earlier than before (`authored_check`, `decide`, `insert_before`, `predict`, `work`), so the cached head is about 680 tokens larger on every request (5,938 tokens instead of 5,257). To check that listing not-yet-callable tools early does not invite calls to them, the first request of five recorded runs was sampled three times with the five tools listed and three times without (Qwen3.6-35B-A3B, recorded settings): the model called none of them in any of the 15 samples in either case, and its first actions were the same.

The following live comparison is from the stable tool head before request-1 seeding, not a measurement of the initial seed. It covers the three bundled examples and two harder multi-file tasks scored with hidden tests, with three runs per task in each setting (30 runs, none void). Uncached prompt tokens per request is the server's `prompt_n` summed over a run and divided by its requests; wall time is per successful task.

| Task | Submitted (off / on) | Uncached prompt tokens per request (off / on) | Wall time per successful task (off / on) |
|---|---|---|---|
| discount_bug | 3 of 3 / 3 of 3 | 10,013 / 7,699 (-23%) | 208 s / 184 s |
| brief_to_report | 3 of 3 / 3 of 3 | 11,358 / 9,380 (-17%) | 330 s / 289 s |
| tiny_site | 3 of 3 / 3 of 3 | 11,627 / 10,682 (-8%) | 356 s / 288 s |
| multi-file task 1 | 3 of 3 / 3 of 3 | 21,498 / 18,775 (-13%) | 1,436 s / 1,459 s |
| multi-file task 2 | 2 of 3 / 3 of 3 | 23,206 / 21,027 (-9%) | 3,846 s / 1,941 s |

Over all 30 runs, uncached prompt tokens per run fell 25% (84,714 to 63,166) and wall time per run 15% (979 s to 832 s). After the first request, 14-22% of each prompt was served from the cache with the switch on; with it off, 10 of 15 runs had none. Hidden-test scores were no worse (26 against 22 of 42 passed). With three runs per task, read the outcome and wall-time columns as "not worse" rather than as a measured gain; the token reduction appeared in every task.

### Prompt cache: current authority last

`STATEFRAME_FINAL_CACHE_LAYOUT` (on by default; set it to `off` in the shell you start StateFrame from, or in a custom profile, to get the previous request order and bytes) reorders each request so that the parts that change on every step come last. The typed state frame moves after the stable carrier rows, and current authority rows (continuation limits, stop and completion authority, evidence pins, menu and read contracts) move into the final message with a small overlay that names them. The model sees the same content; only its position changes, and the request is checked to rebuild the original frame exactly before it is sent. If the reordered request would not fit the measured prompt envelope, StateFrame sends the normally fitted request for that step and records a `final_cache_layout_failed_open` event instead of stopping the run. The same happens for any request the layout cannot reorder while it is only on by default; with `STATEFRAME_FINAL_CACHE_LAYOUT=on` set explicitly, such a request stops the run with an error instead, so a layout problem is never hidden from someone who asked for it. It cannot be combined with the page log, which a custom profile turns on with `STATEFRAME_PAGE_LOG`: with the layout only on by default, the page log wins and each request records `final_cache_layout_failed_open` with reason `page_log_on`; a run that sets both explicitly stops with an error when its first request is built.

The fitter measures the layout's token overhead before allocating the request. If the final arrangement grows more than that estimate, it can reallocate against the measured growth in up to two bounded retries. A request that cannot fit even after refitting still uses the fallback above. With the default layout, source windows keep their first-delivered order across requests; newly selected windows append at the end, and evicted windows are removed without changing the order of the survivors. Failure and bookmark priorities still determine which windows fit under budget. This order is saved for resume, and an ordinary fallback request keeps it too. This keeps more of the source prefix reusable without enabling the experimental stable wire switch.

Measured live on Qwen3.6-35B-A3B served by llama.cpp (one server, 8 slots), across the three bundled examples and two harder multi-file tasks with two runs per task in each setting (20 runs, none void): from request 2 on, the server reused 6,164 cached prompt tokens per request with the switch on against 4,595 with it off (+34%), and the cached share of prompt tokens rose from 20.9% to 25.7%. The arms ran their own trajectories, and runs with the switch on sent 9% larger prompts, so uncached tokens per request are not comparable between them. 10 of 10 runs submitted with the switch off and 8 of 10 with it on: one stopped on a prompt-envelope error that now falls back as described above, and one repeated an unresolved shell effect until its deadline, which is unrelated to the layout. A token-exact replay of recorded requests, which over-counts cached tokens by about 6% for both layouts, estimates 7.8% fewer uncached prompt tokens per request on the same trajectories.

### Prompt cache: stable source windows

`STATEFRAME_STABLE_WIRE=on` is an opt-in projection on top of the default-on final cache layout; explicitly setting `STATEFRAME_FINAL_CACHE_LAYOUT=off` makes the first request refuse. The default is off, so existing request bytes stay unchanged. Within an epoch, delivered standard and model-projected source windows keep their exact bytes and order, and new windows append. A `[Source windows status]` message follows `[StateFrame volatile]`, identifies current windows, and lists every retained window under `Not current, do not edit from these:`. That status text contains no em dash or en dash. Source coverage, `[Host fact]`, and each guidance row follow the status as separate messages. The frame splits into one stable `[StateFrame]` message followed by `[StateFrame volatile]`, which carries the whole action ledger, headcarry, recorded events, open sites, and code bookmarks when present. Other frame fields and lists keep their existing rendering and order.

`STATEFRAME_STABLE_WIRE_HEADROOM` sets the positive integer token allowance for retained but not current windows in the request; it defaults to 16000. Values below 6000 are refused because measured compactions every 2-3 requests erase the gain. When retained-window headroom exceeds the setting, request structure or stable frame content changes, or the full prompt envelope no longer fits with retained windows counted, the next epoch contains only the current windows in their first-seen order. Stable frame content is included in the structural signature because that whole message must remain byte-identical within an epoch. An offline comparison on recorded runs measured 16000 ahead of 12000, with a 10.3% fresh-request reduction versus 7.9% and zero compactions versus one. If a fresh current-only request still does not fit, StateFrame sends the final-layout input unchanged and records `stable_wire_failed_open`. A malformed or unprovable projection refuses instead of sending altered bytes. Epoch state is checkpointed only while stable wire is on. This switch remains off by default. Experimental; in live testing it can end a run with StableWireError when a delivered source window is re-rendered. A fix is planned; leave it off.

### Prompt cache: recommended llama.cpp server settings

For hybrid models such as Qwen3.6-35B-A3B, llama.cpp reuses the prompt cache only up to its last saved checkpoint, not up to the longest matching prefix. Start the server with `-cms 256 -cram 0` (keep `--ctx-checkpoints 32`). In a replay of four recorded runs of one task against one llama.cpp build b11044 server, these settings raised the cached share of prompt tokens by 2.7 to 8.6 points over the same server with `--ctx-checkpoints 32` alone; with the default spacing only the system and tool head (about 5,900 tokens) was reused. `-cms 1024` gained less (1.5 to 7.2 points). The server's `/props` does not report these settings, so StateFrame cannot check them for you: if the cached share of later requests stays near the size of the system and tool head, the server is probably still on the default checkpoint spacing.

### Closing check: requirement recheck

`STATEFRAME_REQUIREMENT_RECHECK=on` (off by default; set it in the shell you start StateFrame from, or in a custom profile) answers the model's first `done` of a run with the task's own sentences as a numbered checklist (at most 20, each cut at 240 characters) and asks it to check each one against its changes before finishing. It refuses once per run only, never when fewer than 2 model steps are left, and never when the task has fewer than 2 sentences. It makes no extra model call; the second `done` submits as usual. The run journal records a `requirement_recheck_refused` event when it fires. Request bytes are unchanged when the switch is off.

Measured on Qwen3.6-35B-A3B on two harder multi-file tasks scored with hidden tests (3 runs per arm): on the first task, 2 of 3 recheck runs passed 8 of 9 hidden tests and one passed 2 of 9, against 2 of 9 in all 3 runs without it; on the second task every run passed with or without it. The check costs extra steps: over the 6 runs, 53 model requests and about 1.36M prompt tokens with the switch, against 27 and 0.62M without it, most of that on the task that already passed.

### Shell: persistent directory and variables

`STATEFRAME_SHELL_STATE=on` (off by default) keeps the shell's working directory and exported variables between foreground `shell` calls, so `cd sub && export X=1` followed by `pwd; echo $X` prints `sub` and `1`. Each call starts in the directory the previous call ended in and with the variables it exported or unset; a reply names the directory when it is not the workspace root. A directory outside the workspace, or one a later command removed, falls back to the workspace root. Variables the shell maintains itself (`PWD`, `OLDPWD`, `SHLVL`, `_`) and `STATEFRAME_*` names are never carried; at most 256 variables and 64 KiB are kept. A call that times out keeps the previous state. Background jobs and custom or MCP tools still start at the workspace root with the base environment. The state is part of each call's replay record, so resume reproduces it. With the switch off, commands, request bytes and the tool tape are unchanged.

### Commits you can review

`STATEFRAME_GIT_COMMIT=on` (off by default) gives the agent two tools. `git_commit(message)` records the current working copy as a commit on the current branch, and `git_branch(name)` starts a new branch from the current branch's latest commit. Neither changes files or touches your repository: each commit is a snapshot in the run folder, and secret paths are never included. After the run, StateFrame writes one `git format-patch` series per branch to `<run-dir>/commits/<branch>/` and prints where it is. Apply a series to your checkout with `git am <run-dir>/commits/main/*.patch`. The first patch applies on top of the state the run started from. Commits are deterministic (fixed author `StateFrame` and timestamps), a commit with no change is refused, and a run holds at most 200 commits. Resume serves each recorded commit from the run folder instead of committing again. With the switch off, the tools are not offered and request bytes are unchanged.

## Checkpoints and resume

A run started with `--timeout` keeps its time limit when you resume it: it gets the time that was left when it stopped, counted from the end of replay, so replaying recorded steps spends none of it. `stateframe resume --timeout S` sets a new limit instead.

Checkpoints are on by default, with a save attempted at every engine step boundary and the two newest kept; a save is skipped while the previous writer is running. They are signed with a per-install key in `~/.stateframe/checkpoint.key`. Checkpoint restore applies to runs with and without the shell; runs with hooks or per-step models resume by replay and never restore a checkpoint. Resume compares the raw bytes of every request after the restored checkpoint and the first in-flight request with their recorded bytes before going live. An explicit model name swap may change only the leading top-level `model` field. The run's time-left countdown (`sandbox_seconds_remaining`) is compared at its recorded value, since a rebuilt request is made later; every other byte must match. Checkpoint restore admits only classified audit or deterministically rehydratable journal tails; an unknown or unreconciled state-changing tail blocks checkpoint restore. If no checkpoint qualifies or a request differs, resume falls back to full replay. Full replay also verifies the in-flight request and stops before a live send if it differs. Set `STATEFRAME_CHECKPOINT=off` to skip selection, restore and later saves on resume. See "Checkpoint settings" in the README for other switches and measured disk use. When a run finishes for good (its result was submitted, or it stopped on an error that would only repeat, so resume would refuse it), StateFrame removes its checkpoints and prints how much it freed; `--keep-checkpoints` (or `STATEFRAME_KEEP_CHECKPOINTS=1`) keeps them. It also removes the run environment (`--keep-run-env` keeps it) and compacts the journal, blobs and preflight requests; see [Run storage](storage.md) for the finished-run layout. Stopped, cancelled, killed and crashed runs keep their newest snapshots; archived attempts retain diagnostics without snapshot payloads.

## Determinism scope

Identical inputs give identical requests on the measured workload apart from three clock fields (the time in the environment panel, how long a job has run, when a navigation handle expires). Tool output the engine does not recognize as volatile can still differ. See KNOWN_LIMITATIONS.md.


## Agent tools

The default shell mode runs bounded foreground commands and can start read-only
background jobs in the composed loop. The sandbox and post-command effect checks
enforce filesystem authority; the standard-mode program filter is a convenience.
Executable-bit changes are restored, and reconciliation stops if a file exceeds
16 MiB, aggregate regular-file bytes exceed 256 MiB, or the tree exceeds 20,000
entries. Shell, job and custom-command output delivery shrinks to fit the measured
request while the full taped observation remains on disk within collector limits.
Shell output does not count as a source read or an engine check. Network installs
need `--shell-network full`, and installs in the run environment last only for that
run. See [Agent tools and execution safety](tools.md) for each tool, its arguments,
policy, refusals and resume behavior.

### Foreground shell settings

`--shell standard` enables bounded foreground commands by default. The sandbox
is the security boundary; its program filter is a convenience, and interpreters
and build scripts can run other programs within that boundary. `--shell read-only`
uses an inspection grammar, a read-only workspace, scratch-only writes and no
network. `--shell unrestricted` removes the filter. `--shell off` hides the tool.
Host execution requires explicit `--shell unrestricted --shell-sandbox off`.

Networking defaults to loopback. Use `--shell-network none` to disconnect it or
`--shell-network full` for network installs. `--shell-allow`, `--shell-deny` and
`--shell-read` extend the host policy. Chat answers `--shell-ask` requests on its
TTY; unattended runs decline. Confirmation expires after 120 seconds.

The default run environment clones the selected virtual environment or creates
a system-site environment from the selected base interpreter. Installs affect
this run only; update dependency declarations to keep them. Engine checks and
final host checks use the same environment. `--no-run-env` disables provisioning.
Shell output grants neither source-read credit nor test evidence: use native
reads and `run_tests` for those. Authorized product changes invalidate earlier
reads and checks; protected tests are restored from the immutable start snapshot.
Plain-loop mode retains its declared-product-only edit/export contract.

A shell-enabled resume restores the environment and scratch state from a
checkpoint when one qualifies (it copies the run directory and saves the
execution session unstarted; no checkpoint is taken while a background job
runs), and otherwise replays from the beginning. Local calls caught up after the
restore point, or during a full replay, must reproduce their
exit code, workspace and environment identity before their recorded output is
served. Full-network and host calls require `--replay-external`; incomplete calls
refuse resume. Raw retained output and truncation counters remain in the run
directory. The foreground shell tool does not launch background shell commands; read-only background jobs use the separate job tools.
