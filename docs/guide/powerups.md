# Powerups

A powerup packages a skill, command tools, lifecycle hooks, and optionally an MCP server definition. StateFrame resolves it on the host at run start. The selected parts enter the same skill, tool, hook, and MCP loaders used by `--skill`, `--tools`, `--hooks`, and `--mcp`. A powerup adds no engine capability.

## Package

Create a directory whose name matches `name`. It needs `powerup.toml` and may contain `SKILL.md`, `commands.json`, `hooks.json`, `mcp.json`, and their scripts. `commands.json` uses `stateframe-tools/1`; `hooks.json` uses `stateframe-hooks/1`; `mcp.json` uses `stateframe-mcp/1`. Do not use a skill `tools.json` in a powerup.

```toml
schema = "stateframe-powerup/1"
name = "py-format"
version = "1.0.0"
summary = "Format changed Python files with ruff after each edit."
default = "off"

[offer]
any_files = ["pyproject.toml", "*.py"]
all_of = []
task_matches = []
loops = ["composed"]

[activate]
skill = true
commands = true
hooks = true
mcp = true

[settings.line_length]
type = "integer"
default = 100
min = 40
max = 200
```

The manifest is strict: unknown keys, unsafe patterns, unbounded settings, links, special files, and a source tree over 1 MiB are refused. `install.json` is reserved for installation provenance. Each part is also checked by its usual loader. A skill description must match the manifest summary. Setting types are `string`, `integer`, `number`, `boolean`, and `enum`. Numeric settings require `min` and `max`; strings require `max_length`; enums require `values`. A command or hook can use a setting only as a whole argv element such as `"{setting:line_length}"`. StateFrame renders the chosen values into a content-addressed copy under `$STATEFRAME_HOME/powerup-cache/`.

All three examples find Python on `PATH`. The `py-format` and `test-gate` hooks use `python3` as the first argv element. StateFrame resolves that name at load and pins the resolved interpreter, so a later `PATH` change does not swap it. The `repo-grep` command uses `#!/usr/bin/env python3` in its bundled script. When you write a powerup, never hard-code `/usr/bin/python3`: hooks need an absolute interpreter or a PATH name in argv, and command scripts may use `#!/usr/bin/env python3`. Hooks refuse env shebangs, which is why the hook examples name `python3` in argv instead.

## Discovery, trust, and modes

Discovery prefers an explicit `--powerup PATH`, then a trusted `<repo>/.stateframe/powerups/<name>/`, then `$STATEFRAME_HOME/powerups/<name>/`. The default home is `~/.stateframe`. Bundled examples are not discovered automatically. `stateframe powerups trust --repo PATH` pins the project state file and project powerup trees in the same trust store used for project hooks. Changed project content is not loaded until trusted again.

The effective mode and settings come from the manifest, then `$STATEFRAME_HOME/powerups.json` global state, its selected profile section, trusted project `.stateframe/powerups.json`, and CLI overrides. A project state file can set `on` only for a powerup shipped by that project. `--no-powerups` wins over every source. Installed powerups start `off`, even if their manifest requests another default; `enable` is the trust act and pins the reviewed tree digest. An active installed unit without a pinned digest is skipped until enabled through the CLI.

`off` attaches nothing. `offer` attaches only `SKILL.md`: the model sees its catalog line and can call `skill_load`, but command schemas and command names are absent. `on` attaches the parts allowed by `[activate]`. `auto` becomes `on` only when the offer rules match. `any_files` matches at least one repository-relative path, `task_matches` requires one regex match in the task, and either match can offer the unit. `all_of` requires a match for every listed glob. The scan is bounded at 20,000 directory entries. Rules are evaluated once at run start, and `loops` limits compatible agent loops.

User-supplied tools and MCP servers take the shared 1,500-token schema allowance first. Powerups append in name order and are skipped greedily when they would exceed the allowance or the six-powerup-tool limit. User skills and hooks likewise keep their original order. A broken, changed, or over-budget powerup prints `powerup NAME disabled: REASON` and is recorded in `spec.json` under `powerups.skipped`; other units can continue. If the user or trusted project state file cannot be read, no powerup is attached for that run and the skip is recorded as `state_invalid`. The spec has no `powerups` key when no unit is active or skipped. Gating powerup hooks without `on_error` get `continue` in the rendered copy. Authors can explicitly set `block`.

The spec pins both the source tree digest and rendered directory digest. Resume checks the rendered directory before replay and refuses a missing or changed copy with `POWERUP_CHANGED`. Tree digests hash one canonical, versioned JSON list of every entry's type, path, mode and content digest. Powerup parts get exactly the limits a hand-written tool, hook or MCP server would get in the same run, and a powerup cannot widen them. Hooks and MCP servers always run in the sandbox. Command tools follow the run's shell setting: with `--shell unrestricted --shell-sandbox off`, powerup command tools run unsandboxed with full host access as the user running StateFrame, the same as every other shell tool in that run. Review its command argv, network and writes, hook events, bundled scripts, and MCP trust statement before enabling.

### Upgrading from an earlier powerups build

The digest format changed, so pins made before the upgrade are never accepted. After upgrading, review each powerup and trust it again. Until you do, the run continues without it and prints one of these notices:

- Enabled powerup: `powerup NAME disabled: re-trust required: pinned with an older digest format; run 'stateframe powerups enable NAME' after review`
- Trusted project powerup: `powerup NAME disabled: re-trust required: project powerup changed or pinned with an older digest format; run 'stateframe powerups trust --repo REPO' after review`
- Trusted project `powerups.json` that changed after it was trusted: `powerup project state disabled: re-trust required: PATH changed since it was trusted; run 'stateframe powerups trust --repo REPO' after review`, followed by `powerups disabled for this run: fix or remove the invalid state file`. No powerup is attached until it is trusted again.
- Resuming a run recorded before the upgrade: `POWERUP_CHANGED: NAME: pinned with an older digest format; re-trust required`. Start a new run instead.

## Isolated attachment contract

Every active command, hook, and MCP server has a `stateframe-isolated-unit-attachment/1` record in the pinned powerup entry. Each record has a unit digest, an executor, a capability ceiling and its digest, a replay policy, and a command or hook binding. The canonical schema is in `mutations/DESIGN.md`, section 8.2. Ordinary `commands.json`, `hooks.json`, and `mcp.json` parts use the `shell` executor. A `units.json` file may declare a `mutation` executor and its binding. `mutation` units are passed only to a registered executor. `register_executor("mutation", handler)` registers an adapter before the run; `attach_units` keeps its existing call signature. Without that adapter, the entire unit is skipped with `executor_unavailable`. A mutation attachment is never converted into command argv.

Any command or MCP server with `network` set to `loopback` or `full` must declare `replay: "external"`. A networked hook must have a matching external replay attachment in `units.json`, including the hook name, event, and network ceiling. A local or observation replay declaration on a networked part causes the powerup to be skipped. The run's shell network limit still applies.

Installed and promoted powerups always start off, regardless of the manifest default. Installation validates every part and stores provenance in `install.json`. A promoted hook may observe `post_tool` or `session_end`. Its `prompt_submit`, `pre_tool`, and `stop` bindings are skipped until `powerups enable NAME --trust-gating` records trust for the exact installed tree digest. A promoted `session_start` hook is never attached. Changing the tree invalidates both enablement and gating trust.

When an active command or MCP powerup needs schema budgeting, a temporary repository snapshot is used to discover user MCP servers. The run later discovers them in its actual workspace and run directory. These results cannot be reused safely because server startup can depend on those paths; sharing one result would require a change to the engine discovery boundary. Hook-only powerups do not start user MCP servers for budgeting. Runs without active powerups perform no optional budget discovery.

## Commands

```bash
stateframe powerups list [--profile PROFILE]
stateframe powerups show NAME [--profile PROFILE]  # effective values and their sources
stateframe powerups install DIR [--force]
stateframe powerups remove NAME
stateframe powerups enable NAME [--mode on|offer|auto] [--profile PROFILE] [--set KEY=VALUE] [--trust-gating]
stateframe powerups disable NAME [--profile PROFILE]
stateframe powerups trust --repo REPO [--remove]
stateframe powerups promote RUN_DIR MUTATION_ID --name NAME
stateframe run ... --powerup NAME=offer --powerup-set NAME.KEY=VALUE
stateframe chat ... --powerup PATH --no-powerups
```

Promotion reads a recorded mutation attach unit through a reader that verifies its run record and digest. It validates the generated package through the same installation path, records `promoted from mutation ID in RUN_DIR` in `install.json`, and leaves the installation off for review.

A promoted package can carry `units.json`, a list of isolated attachment records. The host validates each declared executor, binding, capability ceiling, replay policy, and ceiling digest. A registered executor receives `(options, unit)` and attaches only the declared isolated unit.

Working examples live in [`examples/powerups/`](../../examples/powerups/): `py-format` formats changed Python files when ruff is installed, `test-gate` checks before submission, and `repo-grep` adds a bounded read-only search command. Each example has its own README and ships off by default.
