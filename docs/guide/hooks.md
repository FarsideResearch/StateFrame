# Lifecycle hooks

Hooks are user-configured commands that StateFrame runs at six boundaries of a run. They can add bounded context, deny an admitted action, report on a completed tool call, require another pass before submission, or print a final status. Hooks run in the execution sandbox with a capability separate from shell commands.

Start with the examples in [`examples/hooks/`](../../examples/hooks/). The main configuration exercises start context, shell guarding, changed-file formatting and syntax checks, and a passing-check gate:

```bash
stateframe run --repo ~/code/shop --task "Fix the pricing bug." \
  --hooks examples/hooks/hooks.json
```

The bundled scripts use an absolute `/usr/bin/python3` shebang. Use a Python 3.11+ interpreter at an absolute path on your machine if that path is unavailable or too old. The separate notification examples are described below.

## Events

| Event | When it runs | Can block? | Where output goes |
|---|---|---|---|
| `session_start` | Once after StateFrame creates the working copy and before baseline checks. | No | Start context in the optional `[USER HOOKS]` panel. |
| `prompt_submit` | Once after `session_start`, with the task text, before baseline checks and the child process. | Yes, refuses the run with status 2. | Context joins start context; a denial prints its reason. |
| `pre_tool` | After tool admission and before the tool has any effect. | Yes, refuses that action. | A denial is returned as `REFUSED HOOK_DENIED` in the tool reply. |
| `post_tool` | After the tool reply and effect reconciliation are final, including refused or failed actions. | No | A note is added to the next request. |
| `stop` | Before each submission, including model `done` and host submission routes. | Yes, can ask the model to continue. | A veto appears as `REFUSED HOOK_STOP`; repeated vetoes can end as `HOOK_BLOCKED`. |
| `session_end` | Once when the final run outcome is known, after `--in-place` application. | No | Escaped text is printed in the terminal. |

Hooks run in file order. A hook never causes another hook event. In chat, each task is one run and gets its own lifecycle.

## Configuration and command contract

Files are UTF-8 JSON with schema `stateframe-hooks/1`. Unknown keys, duplicate JSON keys, invalid types, and invalid paths are rejected before the first model request. `--hooks FILE` explicitly names a file and can be repeated on `run` and `chat`. `stateframe resume` uses the run's pinned definitions.

```json
{
  "schema": "stateframe-hooks/1",
  "hooks": [
    {
      "name": "no-push",
      "event": "pre_tool",
      "tools": ["shell"],
      "argv": ["guard.py"],
      "files": ["guard.py"],
      "timeout": 10
    }
  ]
}
```

The main fields are:

| Field | Meaning and limits |
|---|---|
| `name` | Unique identifier, 1 to 48 lowercase letters, digits, `_` or `-`, starting with a letter. |
| `event` | One of the six event names above. |
| `argv` | 1 to 64 strings, passed as separate arguments without a shell. The executable is absolute or a bundled `files` entry. `{paths}` is permitted only as a whole argument on `post_tool`. |
| `files` | Up to 32 regular files relative to the definition file's directory. Each is at most 1 MiB and all files for a hook total at most 4 MiB. Symlinks, traversal and absolute paths are rejected. Files are snapshotted for the run. |
| `tools` | Exact names or `fnmatch` globs for `pre_tool` and `post_tool`; defaults to `*`. `done` never matches. |
| `changed` | `true` limits a `post_tool` hook to calls that applied workspace changes. `writes: "changed"` implies this filter. |
| `paths` | `fnmatch` globs over repository-relative action paths. For `pre_tool`, these are admitted path arguments. For `post_tool`, these are applied changed paths. |
| `timeout` | Whole seconds from 1 through 590; default 30. |
| `network` | `none` by default, or `loopback` or `full`. It cannot exceed the run's `--shell-network` setting. |
| `writes` | `none` by default, or `changed` for scoped post-tool formatting. |
| `on_error` | `block` by default or `continue`, for the gating events `prompt_submit`, `pre_tool` and `stop`. |
| `max_blocks` | Stop event limit from 1 through 10; default 3. |

A hook receives one JSON object on standard input using schema `stateframe-hook-input/1`. Common fields identify the event, hook and stable event id. The `data` object carries only the event facts: admitted arguments for `pre_tool`, the final reply and changed paths for `post_tool`, latest check facts for `stop`, and outcome, apply status and changed paths for `session_end`. It does not include model reasoning, transcript, environment variables, run spec or endpoint key.

The exit and output rules are:

- Exit 0 continues. Empty stdout is allowed. Plain text is context for `session_start`, `prompt_submit` and `post_tool`; `session_end` text is printed to the terminal. A stdout value beginning with `{` must be exactly one JSON object with only `decision`, `reason` and `context` fields.
- Exit 2 denies `prompt_submit`, `pre_tool` or `stop`; the bounded stderr text is the reason. Exit 2 on an observing event is an error.
- Any other exit status, signal, timeout, invalid JSON or sandbox startup failure is a hook error. Exit 1 does not mean deny. Gating errors block by default and are reported as `HOOK_FAILED`; `on_error: "continue"` explicitly allows a gate error to continue.
- Process cleanup failures and workspace changes that cannot be restored are terminal containment failures. An `on_error` setting cannot override containment.

Each file can contain at most 1 MiB of JSON. A run can load at most 32 hooks total and 8 per event. Arguments are limited to 64 KiB. Hook input is capped at 256 KiB and retained output at 32 KiB per stream. Context is capped at 2 KiB per hook and the entire request panel at 8 KiB and 2,048 measured tokens, whichever limit binds first.

## Matchers and formatter writes

Tool matchers use exact names or `fnmatch` globs. `pre_tool` path matchers see only the admitted action's path arguments. `post_tool` path matchers see only applied, existing paths changed by that action. Deleted paths appear in JSON event data but are never passed to `{paths}`. When the matched write set is empty, the formatter hook is skipped.

`writes: "changed"` is allowed only for `post_tool`, with network `none` or `loopback`. StateFrame grants writes only to the existing regular files that the triggering action changed and that passed the hook's path filters. The formatter may change their bytes while preserving each file's full mode. New files, deletions, renames, cache directories, executable-bit changes, protected tests, hook inputs and paths outside that set are restored and reported. A restoration failure stops the run. Authorized formatter changes become ordinary file receipts and are applied from recorded bytes on resume without running the formatter again.

Hooks can make a run stricter or supply bounded text. They cannot approve an action the engine refused, change tool arguments or replies, manufacture a check result, widen network or filesystem authority, change the tool menu, mark work complete, or write arbitrary workspace files.

## Project hook trust

A repository may keep one hook file at `.stateframe/hooks.json`. The file is not loaded just because it exists. Review it and then trust its exact digest:

```bash
stateframe hooks trust --repo ~/code/shop
stateframe run --repo ~/code/shop --task "..."
```

Trust covers the JSON file, each pinned executable and every declared `files` entry. A changed or untrusted file is not loaded; StateFrame prints a notice and continues without its hooks. Trust the changed files again after reviewing them. Use `--no-project-hooks` to skip project hooks for one `run` or `chat` command, or `stateframe hooks trust --repo PATH --remove` to withdraw trust.

The hook file and its bundled scripts are snapshotted and protected from edits during a run. Project trust is resolved against the original repository, not the generated working copy. Hooks do not come from profiles or environment variables.

## Sandbox and network

Hooks always run inside StateFrame's execution sandbox with a minimal environment, a fresh scratch home, bounded time and output, and no read access to the run's records or endpoint key. There is no host-execution option and no unsandboxed fallback. The declared network mode can be `none`, `loopback` or `full`, but never broader than `--shell-network`. On Linux, `loopback` is a private network namespace, as it is for commands: a hook that must reach a service listening on the host's loopback needs `full`. `writes: "changed"` cannot use full network.

The entry point is pinned, but a hook can deliberately run project code from the working copy, such as `make lint` or a project test command. The agent can edit that code during the run, so such a check can be influenced by the edits. Use bundled hook scripts when the hook's own code must stay fixed.

Hook output is bounded, sanitized and labeled in the `[USER HOOKS]` panel. That label describes provenance; it is not a prompt-injection boundary. The engine's dispatch, check and completion rules remain authoritative.

## Replay and resume

Every hook invocation is recorded in `hook-tape.jsonl` before its result is published. A completed hook is served from that record during replay and is not executed again, including network notifications. Formatter post-bytes are re-applied from the run's recorded effect store and checked against their pre-digests. Request replay for a hook-enabled run is strict: a changed hook result or request refuses replay instead of silently switching to a live request.

A durable hook intent without a completion record may have produced an outside effect. StateFrame refuses to resume that run; reconcile the effect and start a new run. Changing a pinned definition, executable or bundled file also prevents resume.

When no hook source is enabled, the child does not import hook modules, the run spec gains no hook keys or files, and the request bytes, tool tape and journal event kinds match the base release. The raw parity proof compares a run with an untouched checkout of that release.

## Examples and platform limits

`examples/hooks/hooks.json` is the main configuration. The CLI end-to-end test exercises its session context, formatter, syntax check and passing-check gate under the default shipped profile. Its shell text guard is provided as a reviewable example but is not exercised by that test. `macos-notification.json` and `webhook-notification.json` are unverified examples; review the command and destination before using them.

The macOS notification example calls `/usr/bin/osascript`. Linux desktop notification buses are not reachable from the sandbox; use a webhook there. Windows is unsupported. Killing a run while a hook is executing can leave an incomplete intent and make the run unresumable. See [Known limitations](../../KNOWN_LIMITATIONS.md#lifecycle-hooks).

## Troubleshooting

| Symptom | What to check |
|---|---|
| `project hooks not loaded` | Review `.stateframe/hooks.json` and its files, then run `stateframe hooks trust --repo PATH`. |
| `REFUSED HOOK_DENIED` | The named gating hook exited 2. Read the refusal reason and check its matcher and input. |
| `REFUSED HOOK_FAILED` | A gating hook errored or returned an invalid result. Inspect `hooks/records.jsonl` and its bounded `hooks/out/` files. |
| Hook output is absent from a request | Check whether the hook ran, whether its matcher selected the event, and whether the optional panel was trimmed. `hooks/delivery.jsonl` records panel delivery. |
| `HOOK_BLOCKED` | The stop hook exceeded its block limit. Inspect its reason and the `blocks_so_far` value. The candidate is not applied. |
| `HOOK_CONTAINMENT` | Cleanup or restoration could not be confirmed. The run is terminal and must be inspected before starting a new run. |
| Resume says a hook may or may not have run | Its intent has no completion. Reconcile any outside effect and start a new run. |
