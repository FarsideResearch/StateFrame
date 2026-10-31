# Agent tools and execution safety

StateFrame offers a tool only when its run settings and the current request allow it. A disabled tool is absent from the menu. Tool output is an observation, not a claim that a file was read or a check passed. Use `read` for source-read evidence and `run_tests` for engine check evidence.

The existing file and check tools are `grep`, `list_files`, `read`, `edit`, `str_replace`, `insert_before`, `authored_check`, `file_create`, `run_tests`, `read_check_receipt`, `recall` and `done`. Some runs also offer probe and whole-module operations. The following tools add command execution, file housekeeping and outside observations. Each accepts an optional `why` string.

## Shell and jobs

`shell(command, timeout_seconds?, background?)` runs `/bin/sh -c` in the run's execution session. `command` is required, nonempty text of at most 16,384 characters. A foreground call defaults to 120 seconds; `timeout_seconds` accepts 1 through 600. `background: true` starts a job, with a 600-second default and the same 1 through 600 range. At most four jobs may run at once. The shell tool is offered unless `--shell off` is set.

A foreground reply shows the command (first 200 characters), exit code or timeout, elapsed time, shell policy, network mode and bounded merged stdout and stderr. It also lists authorized file effects, restored unauthorized effects and a changed environment identity when applicable. ANSI escapes are removed, carriage-return progress lines are collapsed and run, home and scratch paths are normalized. Binary output gets a byte-count notice. Text output initially keeps up to 6 KiB from each end with an omitted-byte count. Near the prompt limit, delivery shrinks further against a measurement of the whole request. The status and effect summary remain intact, and the tool tape retains the full observation; process output itself is retained only up to the supervisor's output cap. The shorter delivery states both its omitted observation bytes and bytes already omitted from process output. A zero exit code, including from a shell-run test command, does not satisfy the engine's completion checks.

Background jobs receive a read-only working copy and a writable job scratch directory. They cannot generate product files. A successful launch replies `started job j1: <command>`. Job controls remain available while a job exists, including after checks turn green:

| Tool | Arguments | Result |
|---|---|---|
| `job_wait` | `job`, `until: exit\|output\|endpoint`; optional literal `text` for output, TCP `port` (1 through 65535) for endpoint, `timeout_seconds` (1 through 600, default 120) | `met`, `timeout` or `exited`, the exit code, elapsed time and a bounded output tail. Endpoint means a connection succeeded, not that this job owns the port. |
| `job_output` | `job`, optional `tail_lines` (1 through 200, default 200) | Current state, exit code and retained tail. |
| `job_stop` | `job` | Stops the supervised job and reports the number of processes stopped. |

The frame carries a `jobs` section with IDs, shortened commands, state, exit codes, recent events and bounded new output. One sequenced snapshot is used throughout each logical request, including retries. The retained output ring is 1 MiB; older bytes can be lost and are counted. `done` stops remaining jobs before completion.

Shell refusals use `REFUSED <CODE>: <reason>`: `SHELL_OFF`, `READ_ONLY_POLICY`, `DENIED_PROGRAM`, `ASK_DECLINED`, `NETWORK_OFF`, `JOB_LIMIT`, `SANDBOX_UNAVAILABLE`, `TIMEOUT` and `CLEANUP_FAILED`. `NETWORK_OFF` names `--shell-network full` for detected dependency-install commands. `TIMEOUT` includes partial output. `CLEANUP_FAILED` stops the run because effects cannot safely be inspected. Job controls use `UNKNOWN_JOB` and `BAD_CONDITION`; invalid background timeouts use `DENIED_PROGRAM`.

## Command policy

| `--shell` mode | Workspace | Commands |
|---|---|---|
| `off` | No shell tool | Default run environment is also off. |
| `read-only` | Read-only; scratch writes only; network none | Inspection programs and restricted pipelines, `&&`, `;` and `/dev/null` redirection. Expansion, command substitution, file-writing options and other syntax are refused. |
| `standard` (default) | Writable under reconciliation | Sandbox boundary plus a convenience program filter. `sudo`, `doas`, `su`, `ssh`, `scp` and `git` are denied; `curl`, `wget` and `nc` are also denied with network none. Interpreters and build scripts may run other programs inside the sandbox. |
| `unrestricted` | Writable under reconciliation | No program filter. The same sandbox and effect policy still apply. |

`--shell-network none|loopback|full` defaults to `loopback`. Use `full` for network dependency installs. `--shell-allow PROGRAM` suppresses an ask prompt for that program; it does not override the standard-mode deny set or `--shell-deny PROGRAM`. Both flags repeat. `--shell-read PATH` adds an explicit read-only root. The policy is applied to the command text before `/bin/sh` runs it; it cannot constrain what an admitted interpreter or build script launches. `git` has a separate read-only tool. The shipped product profile uses the composed loop. The plain loop allows foreground shell but refuses background jobs before execution and requires Git, web and custom tools to be disabled. Lean mode is not supported by the product profile.

`--shell-ask` asks before a program outside the inspection and `--shell-allow` sets. It defaults off for `run` and on for `chat`; `--no-shell-ask` changes that. Chat reads the answer from its terminal. An unattended run declines, and a request expires after 120 seconds. Decisions are recorded for resume.

`--shell-sandbox off` requires `--shell unrestricted` and prints `WARNING: SHELL SANDBOX OFF; COMMANDS CAN ACCESS THE HOST`. This is host execution, with access beyond the declared roots. It is authorized only by a CLI flag and recorded in the run. Tool authority cannot be supplied by a profile switch.

## Sandbox and workspace effects

The host issues a separate capability for shell, jobs, Git, custom tools and probes. It contains the allowed read and write roots, network mode, complete child environment and wall, CPU, memory, file, descriptor, process and output limits. Ordinary shell commands can write the working copy, run environment and scratch home. Background jobs and Git see a read-only working copy. Read access also includes required system libraries and installed runtime roots. Executables discovered on the host PATH grant read authority for those executable files and their symlink and absolute shebang chains, not for their parent directories. macOS developer tool shims such as `make`, `git` and `cc` work within this boundary. A missing sandbox refuses execution; it never silently falls back to host mode.

On macOS, `sandbox-exec` enforces the declared filesystem and network profile. A supervisor tracks descendants by parent, process group and session and stops them before effects are reconciled. A process deliberately double-forked and orphaned between supervisor sweeps can outlive the run, while remaining in its sandbox profile. Aggregate resident memory is polled, so the memory limit can be exceeded between samples and is not a hard cap.

On Linux, bubblewrap supplies a persistent session with PID and network namespaces and a supervisor at PID 1. Per-command mounts enforce the declared roots. When the installed bubblewrap supports `--disable-userns` and `--assert-userns-disabled`, each command also blocks creation of further user namespaces. Older versions continue to run without that protection; use a newer bubblewrap where untrusted commands can create namespaces. Network `none` also blocks socket syscalls; loopback is private to the session. A full-network session cannot switch to shared loopback placement later in the same session; narrower per-command loopback uses a separate namespace and cannot reach services in the full-network namespace. The sandbox needs working unprivileged user namespaces. Linux command containment was exercised in an aarch64 container; x86_64 command containment has not been verified in this work.

After an arbitrary command becomes quiescent, StateFrame rehashes and checks the working copy. Allowed creations, edits and deletions produce effect receipts and invalidate stale reads and checks. Executable-bit changes, including creating an executable, are refused and restored because the sealed candidate does not carry that metadata. Shell removal of an executable is refused; use `file_delete` for an intentional removal. A protected test, its inode alias, a file outside `--edit` scope in a scoped repository, or a forbidden link or special file is restored to its pre-call state. Protected tests use the immutable run-start baseline. A failed restoration stops the run. New directories are allowed only as part of authorized product effects. Reconciliation refuses to promote effects if a regular file exceeds 16 MiB, total regular-file bytes exceed 256 MiB, or the workspace exceeds 20,000 entries. The limits apply to both the pre-call snapshot and post-call scan. This policy applies to shell and workspace-writing custom tools, not only native edit tools.

## File housekeeping

`file_delete(path)` removes one regular file, including an executable. `file_move(source, destination)` moves one non-executable regular file and creates missing destination directories. Both paths are repository-relative. Neither tool follows a symlink or permits edits to protected tests. The source must be editable; a destination must satisfy the new-file policy and must not already exist. A case-only rename is handled on case-insensitive filesystems. Deletions and moves enter the sealed candidate, so they can be checked and applied like writes.

On success the model sees `file_delete applied. <path>` or `file_move applied. <source>, <destination>` and an edit receipt. A refusal uses `REFUSED <CODE>: <reason>`, for example `REFUSED DESTINATION_EXISTS: ...`; an operating-system failure is reported as `file_move FAILED: <exception type>: ...`. Refusal codes are `PROTECTED_TEST`, `PATH_NOT_FOUND`, `OUTSIDE_WORKSPACE`, `DESTINATION_EXISTS`, `NOT_EDITABLE`, `NEW_FILE_FORBIDDEN`, `SYMLINK`, `IS_DIRECTORY` and `UNSUPPORTED_MODE`. A non-directory ancestor produces `PATH_NOT_FOUND`, a special source produces `NOT_EDITABLE`, an identical source and destination produce `DESTINATION_EXISTS`, and an executable move produces `UNSUPPORTED_MODE`.

## Git and web observations

The read-only `git` tool is offered for a Git source repository unless `--no-git-tool` is set. It accepts `action: status|diff|log|show|blame`; optional `against: baseline|HEAD` for status or diff, `path`, `rev`, `max_count` (1 through 50) and an inclusive `start_line`/`end_line` pair for blame. A status or diff is scoped to copied product paths. History uses a private sanitized Git view with pinned HEAD and no repository configuration, remote helper, external diff, pager or lazy fetch. `show` peels revisions to commits before reading their trees; raw blob and tree object IDs are refused. Excluded paths are refused or filtered. Internal repo-indexing Git queries also use an offline sanitized view, so remote-derived cache keys are unavailable. Git output gives no source-read credit. Refusals are `NOT_A_GIT_REPO`, `EXCLUDED_PATH`, `BAD_REVISION`, `MISSING_OBJECT` and `BAD_ARGUMENT`. The tool is absent when the source is not a Git repository.

`web_fetch(url, max_chars?)` is off by default; enable it with `--web`. `max_chars` accepts 1 through 12,000 and defaults to 12,000. `--web-allow DOMAIN` is a repeatable optional host allow list. Each URL and redirect is checked; private, loopback, link-local, metadata and other non-public addresses are refused. Requests use a vetted IP without ambient proxies, with the original hostname for TLS verification. Only HTTP and HTTPS on ports 80 and 443 are allowed. There are at most five redirects, a 20-second total deadline, a 2 MiB transferred-body limit and an 8 MiB decoded limit. Readable text, HTML, JSON and XML are accepted. The model sees extracted text in an `UNTRUSTED EXTERNAL CONTENT` block, capped with byte counts. That label is a warning to the model, not a prompt-injection boundary. Refusals are `WEB_REFUSED_ADDRESS`, `WEB_REFUSED_SCHEME`, `WEB_REFUSED_PORT`, `WEB_DOMAIN_NOT_ALLOWED`, `WEB_TOO_LARGE`, `WEB_CONTENT_TYPE`, `WEB_TIMEOUT` and `WEB_HTTP_ERROR`.

## Custom command tools

`--tools FILE` explicitly trusts a JSON file and can be repeated. The file must use `stateframe-tools/1`. For example, this definition offers a bounded, read-only command that prints a selected label:

```json
{
  "schema": "stateframe-tools/1",
  "tools": [
    {
      "name": "show_label",
      "description": "Print a label for this run.",
      "parameters": {
        "type": "object",
        "properties": {
          "label": {"type": "string", "maxLength": 40, "pattern": "[A-Za-z0-9_]{1,40}"}
        },
        "required": ["label"]
      },
      "argv": ["/usr/bin/printf", "%s\\n", "{label}"],
      "network": "none",
      "writes": "none",
      "replay": "local",
      "timeout_seconds": 30
    }
  ]
}
```

Run with `stateframe run ... --tools path/to/tools.json`. Names must match `^[a-z][a-z0-9_]{1,40}$` and cannot collide with a built-in or another file. Each tool has at most eight typed properties (`string`, `integer`, `number`, `boolean`, with optional enum); strings need `maxLength` at most 1,000. The description is at most 300 characters. An `argv` placeholder occupies a whole item and refers to a required property; values are passed as separate arguments, without shell interpolation. An executable such as `sh -c` can interpret those arguments itself, so the definition's author owns that risk. The executable and definition are hashed for the run. The combined custom schemas must fit 1,500 measured tokens, or load fails with a named error.

`network` defaults to `none` and cannot exceed `--shell-network`. `writes` defaults to `none`, which mounts the working copy read-only; `workspace` allows reconciled effects. `replay` defaults to `external`; use `local` only for a reproducible command. A full-network tool must use external replay. `timeout_seconds` is required and must be finite, greater than zero and at most 600; invalid values fail at load before execution. Custom replies have bounded shell-style output and shrink further when the whole request approaches its budget. Invalid arguments use `BAD_ARGUMENTS`; a changed executable uses `TOOL_CHANGED`. Invalid definitions fail at load before a model request. With no tokenizer, schema admission uses a recorded byte estimate rather than a measured token count.

## MCP servers

`--mcp FILE` is a repeatable opt-in for trusted local MCP stdio servers. The file is strict JSON. For example:

```json
{
  "schema": "stateframe-mcp/1",
  "servers": [{
    "name": "local_docs",
    "trust": "I reviewed this local server and its executable.",
    "argv": ["/absolute/path/to/server", "--fixed-option"],
    "network": "none",
    "writes": "none",
    "replay": "external",
    "processes": "single",
    "timeout_seconds": 30,
    "tools": ["search"]
  }]
}
```

Each server needs a unique name, a nonempty trust statement, fixed argv and a finite timeout of at most 600 seconds. The optional `tools` list limits discovery; without it, all listed tools are considered. The executable and definition are hashed before the run and checked again on resume. StateFrame initializes the server and lists its tools inside an execution sandbox before offering any to the model. Invalid tool names, collisions, unsupported schemas and tools beyond the shared 1,500-token custom plus MCP schema allowance are omitted. The run records engine-owned omission codes and safe names. The model sees at most 16 entries in a 1,000-byte note; remaining entries are counted. Function descriptions are written by the engine and label server descriptions as quoted, bounded, untrusted metadata. Property descriptions receive the same label; neither can authorize actions. Server results are also untrusted content. Put the interpreter in argv explicitly (for example `["/usr/bin/python3", "/path/server.py"]`): a `#!/usr/bin/env` script looks up its interpreter through PATH inside the sandbox and may not find it.

Each discovery and tool call starts a fresh server inside the capability sandbox; the protocol client stays in the engine process. This boundary remains in force even when shell host execution is explicitly authorized. The call is supervised throughout protocol reads and writes. Aggregate resident memory, process count and wall time are checked for the server's containment unit; a limit breach ends the call with a refusal. These limits are polled, so a brief excess between samples remains possible. `processes` defaults to `single`. On macOS, the server sandbox denies process creation. Set `processes` to `spawn` for wrappers that need children on Linux; its private PID namespace is torn down after the call. macOS refuses `spawn`, so point argv at the final program, for example `node /path/dist/index.js` instead of `npx`. `network` defaults to `none` and cannot exceed `--shell-network`; `writes` defaults to `none`, making the working copy read-only. Workspace writes use the same effect reconciliation as custom commands. `replay` defaults to `external`. Resume refuses to initialize or list any external-class server until `--replay-external` is given, even if no tool call was taped. `local` re-executes and compares workspace state; `observation` serves the tape and is allowed only with no network and no workspace writes. A full-network server must use external replay. MCP arguments, including a declared `why` field, are sent as advertised; the engine does not add its own `why` field to MCP schemas. Stateful servers that need a persistent session are unsupported. Only stdio tools/list and tools/call are supported; server requests cannot direct the client to execute other work.

## Run environment, resume and prompt budget

`--run-env` is on by default when shell is enabled; `--no-run-env` disables it. StateFrame clones a selected project virtual environment or creates a run-local system-site environment from the selected base interpreter. Python and package caches for this run live under the run directory. Editable installs pointing into the source repository are remapped to the working copy. If safe remapping or cloning fails, StateFrame reports the problem and continues with run environment off. Installs affect this run only: add dependencies to the project declaration to keep them. Engine and final host checks use the run environment when it is available. A changed environment identity makes earlier check evidence stale. Network installs need `--shell-network full`.

Shell, job, Git, web and custom calls have durable intent and completion records in `tool-tape.jsonl`, bound to arguments, capability, definition, environment and workspace state. Native file operations follow the existing edit replay path. Resume treats taped calls by replay class:

| Class | Calls | Resume |
|---|---|---|
| Observation | Git, web, read-only shell, job starts and controls, job frame snapshots | Serve the recorded observation without re-execution. Jobs are never relaunched; a running job becomes `lost on restart` when live work resumes. |
| Local | Sandboxed shell with none or loopback network, custom tools marked `local` | Re-execute on the fresh working copy. Exit code, resulting workspace manifest and environment identity must match before recorded output is served. |
| External | Full-network shell, host-mode shell, custom tools marked `external` | Refuse automatic replay. `stateframe resume <run-dir> --replay-external` explicitly repeats them, then applies the local comparison. Repetition can repeat an outside effect. |

An intent without completion means the operation may have happened; resume refuses and asks for reconciliation and a new run. Tool divergence refuses resume, including checkpoint fallback. Shell-enabled resume rebuilds the run environment and replays from the start. Live jobs make checkpoints ineligible. The menu, tool schemas, job section, effects and newest exchange all count against the same request budget. Execution reply delivery shrinks against the measured whole request, while the original taped observation remains on disk. If the required minimum will not fit, StateFrame refuses the request rather than silently dropping authoritative state.
