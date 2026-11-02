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
- `--protect-tests off|task|all`: `off` (default) permits test edits and reports them; `task` locks tests named by the command or selectors and tests failing at the start; `all` locks every pre-existing test as in 0.2. If a generic command has no explicit test file selector, `task` locks every existing test file, including when the starting check passes. The result line reports this fallback.
- `--protected-test-glob GLOB`: add a repository-relative pattern for test or check resources. Repeat as needed; it is locked when protection applies.
- `--target`, `--edit`: in repositories over 200 source files, the files the change may touch; the rest is read-only context.
- `--in-place`: apply a passing result to your repository.
- `--run-dir`: where to keep the run.
- `--max-turns N`, `--timeout S`: optional caps; by default there are none.
- `--test-env KEY=VALUE`: environment for your tests (not for engine settings).
- `--pytest-plugins installed`: let pytest load installed plugins.
- `--shell off|read-only|standard|unrestricted` (default `standard`): command policy. `--shell-network none|loopback|full` defaults to `loopback`.
- `--shell-ask` / `--no-shell-ask`: confirmation for commands outside the inspection and user allow sets. Default off for `run`, on for `chat`.
- `--shell-allow PROGRAM`, `--shell-deny PROGRAM`, `--shell-read PATH`: repeatable program and readable-root settings.
- `--shell-sandbox on|off` (default `on`): `off` requires `--shell unrestricted`, prints a host-access warning and is recorded as host authorization.
- `--run-env` / `--no-run-env`: run-local dependency environment; default on when shell is enabled, otherwise off.
- `--git-tool` / `--no-git-tool`: read-only Git tool, on by default when the source has Git metadata.
- `--web` / `--no-web`: web fetch, off by default. `--web-allow DOMAIN` repeats to restrict hosts.
- `--tools FILE`: repeatable trusted custom command definitions in `stateframe-tools/1` JSON.
- `--hooks FILE`: repeatable lifecycle hook definitions in `stateframe-hooks/1` JSON. Project hooks at `.stateframe/hooks.json` require `stateframe hooks trust --repo R`; `--no-project-hooks` skips them for one `run` or `chat` command.
- `--skill NAME_OR_PATH`: repeatable selected skill bundle. `--trust-skill-tools` offers its optional command definitions. See [Skills](skills.md).
- `--instructions[=FILE]`: project instructions, off by default. With no value, the repository's `AGENTS.md`, else `CLAUDE.md`; with a value, that file. `--no-instructions` states the default explicitly. See [Project instructions](#project-instructions).
- On `resume`, `--replay-external` explicitly repeats recorded external operations before continuing.

Hook files named with `--hooks` are explicitly trusted by that command. Project hook files need a separate digest trust step, and edits to the file or its pinned scripts require trust again. The [hooks guide](hooks.md) describes the event contract and the sandbox boundary.

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
- **Mixing local and remote is per role.** A remote endpoint needs `allow_remote = true` on that role (or `STATEFRAME_ALLOW_REMOTE_ENDPOINT=1` for all) and prints a warning that your code will leave the machine. A redirect to a remote host is refused unless that role allows remote endpoints. Routed requests and tokenizer calls ignore `HTTP_PROXY` and system proxy settings, so they go only to the address you set.
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

## Checkpoints and resume

Resume replays the recorded run by default. Checkpoints for a faster restore are off by default in this release (`STATEFRAME_CHECKPOINT=save` turns them on: about one every 5 requests, the two newest kept, signed with a per-install key in `~/.stateframe/checkpoint.key`). See "Checkpoint settings" in the README for the switches. When a run finishes for good (its result was submitted, or it stopped on an error that would only repeat, so resume would refuse it), StateFrame removes its checkpoints and prints how much it freed; `--keep-checkpoints` (or `STATEFRAME_KEEP_CHECKPOINTS=1`) keeps them. Stopped, cancelled, killed and crashed runs always keep theirs.

## Determinism scope

Identical inputs give identical requests on the measured workload apart from three clock fields (the time in the environment panel, how long a job has run, when a navigation handle expires) and an edit-receipt ID. Tool output the engine does not recognize as volatile can still differ. See KNOWN_LIMITATIONS.md.


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

Shell-enabled resumes rebuild the environment and scratch state and replay from
the beginning, even when a checkpoint exists. Local calls must reproduce their
exit code, workspace and environment identity before their recorded output is
served. Full-network and host calls require `--replay-external`; incomplete calls
refuse resume. Raw retained output and truncation counters remain in the run
directory. The foreground shell tool does not launch background shell commands; read-only background jobs use the separate job tools.
