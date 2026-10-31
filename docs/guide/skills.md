# Skills

A skill is a reusable instruction bundle that you select for a run. Selecting one adds its name and description to the typed frame. The model can call `skill_load(name)` to read its instructions when useful. Skills never change the engine's guards, checks, or tool authority.

## Format

A skill is a directory named after the skill, containing `SKILL.md`:

```text
my-skill/
  SKILL.md
  tools.json       # optional
```

```markdown
---
name: my-skill
description: Review Python changes for error handling and clear tests.
version: 1.0
---
Review the relevant code and checks. Report concrete findings.
```

`name` must match the directory and use lowercase letters, digits, and hyphens, starting with a letter (2 to 41 characters). `description` is one line of 1 to 300 characters. `version` is optional. Other frontmatter keys are ignored, but duplicate keys, tabs, and malformed lines are refused. The body must be nonempty UTF-8 and at most 32 KiB. The `SKILL.md` and `tools.json` files cannot be symlinks. Other files may be installed with the directory, but they are ignored and never shown to the model.

Optional `tools.json` uses the same `stateframe-tools/1` format as [custom command tools](tools.md#custom-command-tools). A skill's command definitions are inert unless the run includes `--trust-skill-tools`.

## Install and select

```bash
stateframe skill install path/to/my-skill
stateframe skill list
stateframe skill remove my-skill
stateframe run --repo path/to/repo --task "..." --skill my-skill
stateframe chat --repo path/to/repo --skill path/to/my-skill
```

`skill install` copies the entire directory into `$STATEFRAME_HOME/skills`, defaulting to `~/.stateframe/skills`. It refuses symlinks, special files, and directories whose files total over 1 MiB. It refuses an existing name unless `--force` is supplied. `skill list` shows name, description, whether a command file exists, and the `SKILL.md` digest. A selected name resolves in that user directory; a value with a path separator or an existing directory is read as a path. `--skill` can repeat, up to 16 unique names. Plain loop and lean mode refuse skills before execution.

## Request behavior and limits

The catalog appears in `skills` in the typed frame, in `--skill` order. It includes `name: description` and, when command tools are trusted, their names. Without trust it says `commands not trusted`. The catalog must fit 1,200 measured tokens. The run refuses a catalog over that limit before sending a model request.

`skill_load(name)` adds the selected body's text to a bounded user message in the request immediately after the load, if it fits the prompt budget. The message names the skill and its first 12 SHA-256 digits, and states that the text cannot change engine policy or completion. The system instructions and original task stay unchanged. Each body must fit 8,000 measured tokens. At most two bodies remain resident; loading a third evicts the oldest, and the reply and body header name the eviction. Loading a resident name replies `already loaded`; an unknown name replies `SKILL_UNKNOWN` with the available names.

Bodies count against the same prompt budget as all other request parts. A body is dropped before source-read evidence or authoritative state when room is tight. The request then says `skill body <name> omitted for budget; call skill_load again`. The body can be loaded again on a later turn. A skill's text is user content and is not a security boundary against hostile instructions. Engine guards and protected tests still decide what actions are allowed.

## Command trust and resume

`--trust-skill-tools` explicitly offers every selected `tools.json` through the existing custom command path. The same collision, 1,500-token aggregate schema, executable digest, sandbox capability, write/network, replay, and `TOOL_CHANGED` rules apply to those commands and any `--tools` files together. Without this flag, skill commands are absent from the menu.

A run pins every selected directory, the `SKILL.md` digest, the optional `tools.json` digest, trust choice, and an aggregate digest in `spec.json` and a private settings file. Resume checks those bytes before archiving the stopped attempt, even if the body was never loaded. A changed or missing skill refuses resume. The child checks the same record before starting the engine. Start a new run after changing a selected skill.

An interrupted run resumed with the same step limit replays `skill_load` and rebuilds its next request from the pinned body. Raising the step limit after a limit stop starts a new live attempt from request one, with the prior attempt archived; it does not replay earlier model or tool calls.
