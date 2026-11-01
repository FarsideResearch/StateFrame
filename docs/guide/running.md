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
- `--check PATH` (repeatable): the pytest files that decide pass or fail; default every `test_*.py` / `*_test.py`.
- `--test-cmd`: the outer before/after command only; it does not replace the engine's checks.
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

The engine runs on StateFrame's own Python. Your tests run on your project's: `--test-python PATH` if given, else the repository's `.venv/`, `venv/` or `env/`, else `python3` on PATH. StateFrame never falls back to its own environment silently, prints the choice at the start of each run, and stops with a clear message if the chosen Python cannot import pytest. `--python PATH` sets one interpreter for both the engine and your tests.

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
