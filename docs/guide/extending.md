# Extending StateFrame

## Change behavior with a profile

Copy the default profile (the JSON file in `src/stateframe/profiles/`), edit values under `switches`, and run with `--profile path/to/my_profile.json`. Keep `"schema": "stateframe-product-profile/1"`. A switch the engine does not read is refused at the first request (`unconsumed STATEFRAME settings`), so a typo fails loudly instead of being ignored.

## Add a command tool

For a trusted local executable, write a `stateframe-tools/1` JSON file and pass it with `--tools FILE`. Define its typed parameters, argument template, network and workspace-write authority, timeout and replay class. The timeout must be finite, positive and at most 600 seconds; invalid definitions fail before the first request. No engine edit is needed. The file and executable are checked at run start, and a changed executable refuses later calls. The [agent tools guide](tools.md#custom-command-tools) has a complete example and the limits. Use `replay: external` for commands whose outside effects cannot safely be repeated. This tool set requires the composed loop in the shipped product profile; the plain loop refuses custom-tool settings.

## Add an engine tool

A new engine-owned capability still needs code. Work on an unpacked copy of the package and install it from there (`pip install -e <dir>` or `./install.sh`). A tool has a schema and a handler, often in `src/stateframe/engine/probes/react_repair.py`:

1. its schema in `TOOLS` (name, description, parameters), which is what the model is offered;
2. its handler in the agent loop, where the tool name is dispatched and its result is recorded.

Also register its admission, receipt and completion contracts so the tool survives menu filtering and its result reaches the next frame. Add a test that runs it end to end (the scripted endpoint below makes that possible without a model). Code-executing tools need an explicit execution capability and sandbox boundary; follow the existing agent-tool adapters. For workspace-writing commands, reconciliation refuses files over 16 MiB, more than 256 MiB of regular-file data or more than 20,000 entries. The sealed candidate cannot carry executable-bit changes, so those effects are restored. Execution replies are fitted to the measured request budget; retain engine-owned status and effect boundaries when adding a new reply. New engine modules belong in `release-manifest.json`.

## Add an engine switch

Read the switch in engine code with a literal name, for example `os.environ.get("STATEFRAME_MY_SWITCH", "off")`, in a file listed in the engine's `release-manifest.json` (`src/stateframe/engine/release-manifest.json`; add your new module there). The switch registry finds readers by reading the source, so a name built at run time is not seen. Then set it in a profile.

## Tests

- The package does not ship its own test suite. Test your changes with pytest in your checkout, and use the pieces below.
- Offline end-to-end runs use `scripted_endpoint.py`: a script file lists tool calls in order (`{"model": "scripted", "steps": [{"tool": "read", "arguments": {...}}, ...]}`), and `python -m stateframe.scripted_endpoint --script s.json --port 0` serves them as an OpenAI-compatible endpoint. Add `"tokenize": {"mode": "ok"}` so it also answers `/tokenize` the way llama.cpp does; without it the engine counts tokens by a conservative estimate and a run can fail to fit (see troubleshooting). `examples/*.script.json` are working scripts.
- Acceptance tests for non-code output can use `stateframe.checks` (`schema_validity`, `closed_set_membership`, `citation_span_grounding`, `deterministic_recomputation`, and `checks.require(...)`).
