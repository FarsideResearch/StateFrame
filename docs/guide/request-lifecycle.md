# How a run and a request are built, end to end

Diagram: [ARCHITECTURE.md](../../ARCHITECTURE.md#request-lifecycle).

1. **`stateframe run`** (`stateframe/cli.py` -> `stateframe/runner.py`) checks the interpreter, probes the endpoint (the model must be served, the slot must hold 65,536 tokens), chooses the test interpreter, and creates the run directory (default `~/.stateframe/runs/<repo>-<time>`).
2. **Working copy.** `stateframe/workcopy.py` copies your repository into the run directory. Your original is only written with `--in-place`, and only after checks pass.
3. **Task row.** `stateframe/task_row.py` turns the working copy and your task into the engine's task description: the files, the starting file hint, the check files.
4. **Before checks.** Your checks run once on the copy to record the starting point (`stateframe/outer_checks.py`, sandboxed, with your project's Python).
5. **Engine child.** `stateframe/_child.py` starts the engine in its own process with the run's spec (`spec.json`: profile switches, model, endpoint, limits).
6. **Each model turn**, inside the engine:
   - the engine builds the typed state frame from its current state (check results, files in play, recent action results, what is owed);
   - it fits frame and source into the 33,344-token budget, trimming lower-priority detail if needed;
   - it qualifies the request: complete, within budget, addressed to an allowed endpoint;
   - it sends exactly one sealed request body and records its bytes under `wire/`;
   - it runs the tool the model asked for in a sandbox (no network) and records the result.
7. **Done.** When the model calls `done`, the engine checks the result and stores it as the final candidate.
8. **After checks.** StateFrame runs your checks again on that stored candidate, prints the diff and an outcome line, and with `--in-place` applies it to your repository.
9. **Inspect.** `stateframe inspect <run-dir>` reads `wire/NNNN.request.json`, so what you see is what the model saw.
