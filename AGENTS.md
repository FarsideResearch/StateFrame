# StateFrame: orientation for AI coding assistants and new developers

StateFrame is an Open Agent Engine for local OpenAI-compatible model servers, built and measured on software engineering work. It never resends chat history: every model request is rebuilt from the system instructions, the tool definitions, the task, a typed state frame and the newest exchange, inside a fixed 33,344-token budget on a 65,536-token server slot.

## Repository map

| Path | What it is |
|---|---|
| `src/stateframe/` | The product package: CLI, run driver, working copy, resume, inspect, chat. Standard library only. |
| `src/stateframe/engine/` | The engine. It builds and qualifies every request, runs the tools, keeps the typed state and decides when a task is done. It runs in its own child process. Shell effects are reconciled with 16 MiB per-file, 256 MiB aggregate and 20,000-entry bounds. |
| `src/stateframe/profiles/` | Switch profiles: the engine settings a run uses (the one JSON file shipped there is the default). |
| `examples/` | Small tasks with tests, plus scripted model responses for offline demos. |
| `install.sh`, `wheelhouse/` | One-command install with pinned, hash-checked dependencies. |
| `docs/guide/` | The guide: concepts, request lifecycle, code map, running, agent tools, extending, troubleshooting, design rationale. |

## Install, run, test

```bash
./install.sh                                   # macOS arm64 or Linux x86_64 (bubblewrap); Python 3.11+ (not 3.14.1)
export STATEFRAME_ENDPOINT=http://127.0.0.1:8080/v1
export STATEFRAME_MODEL=my-local-model
stateframe run --repo examples/discount_bug --task "Fix the pricing code so the tests pass."
stateframe inspect <run-dir> --send 2 --frame   # the exact frame the model saw
```

The installer puts StateFrame in `~/.stateframe/venv` and, if the name is free, links `~/.local/bin/stateframe` to it. `./install.sh --offline` installs from the bundled wheels only; it skips the optional schema-validation and repo-map extras, which the install summary reports as OFF.

No model server? Start `python -m stateframe.scripted_endpoint --script examples/discount_bug.script.json --port 0` in the background; it prints a URL that answers with recorded tool calls (the example scripts also answer `/tokenize`); run with `--endpoint <that URL> --model scripted` (the URL already ends in `/v1`) to exercise the whole pipeline offline.

Your repository's tests run with your project's Python (`.venv/`, `venv/`, `env/`, else `python3` on PATH); `--test-python PATH` overrides it.

The shipped product profile uses the composed loop; lean mode is not supported by that profile. The plain loop allows foreground shell but refuses Git, web, custom tools and background jobs. Command replies shrink to the measured request budget, with omitted byte counts and the full taped observation retained within collector limits. Executable-bit effects are refused and restored because the sealed candidate does not transport them.

## Where to go next

- [ARCHITECTURE.md](ARCHITECTURE.md): diagrams of the request lifecycle, the frame, completion, resume, history and the code map.
- [Concepts](docs/guide/concepts.md): the typed state frame, requests, receipts, checkpoints, disk-backed history.
- [How a request is built](docs/guide/request-lifecycle.md), end to end.
- [Code map](docs/guide/code-map.md): which module owns what.
- [Running](docs/guide/running.md): CLI, profiles, flags, test interpreter, resume, determinism scope.
- [Agent tools](docs/guide/tools.md): command, job, file, Git, web and custom tool contracts and safety.
- [Extending](docs/guide/extending.md): tools, profiles, tests.
- [Troubleshooting](docs/guide/troubleshooting.md): common errors and what they mean.
- [Design rationale](docs/guide/design.md): why it is built this way.
- Limits: [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md). License: AGPL-3.0-only or commercial ([COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)); contributions need the [CLA](CLA.md).
