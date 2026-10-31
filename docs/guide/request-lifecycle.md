# How a run and a request are built, end to end

Diagram: [ARCHITECTURE.md](../../ARCHITECTURE.md#request-lifecycle).

1. **`stateframe run`** (`stateframe/cli.py` -> `stateframe/runner.py`) checks the interpreter, probes the endpoint (the model must be served, the slot must hold 65,536 tokens), chooses the test interpreter, and creates the run directory (default `~/.stateframe/runs/<repo>-<time>`).
2. **Working copy.** `stateframe/workcopy.py` copies your repository into the run directory. Your original is only written with `--in-place`, and only after checks pass.
3. **Task row.** `stateframe/task_row.py` turns the working copy and your task into the engine's task description: the files, the starting file hint, the check files.
4. **Before checks.** Your checks run once on the copy to record the starting point (`stateframe/outer_checks.py`, sandboxed, with your project's Python).
5. **Engine child.** `stateframe/_child.py` starts the engine in its own process with the run's spec (`spec.json`: profile switches, model, endpoint, limits).
6. **Each model turn**, inside the engine:
   - the engine builds the typed state frame from its current state (check results, files in play, recent action results, what is owed);
   - it fits frame, source and the newest execution reply into the 33,344-token budget, trimming lower-priority detail or execution output delivery if needed while retaining engine-owned status and effects;
   - it qualifies the request: complete, within budget, addressed to an allowed endpoint;
   - it sends exactly one sealed request body and records its bytes under `wire/`;
   - it admits the tool against the current menu and its host-issued capability, then runs commands in the configured sandbox and network mode;
   - for shell, job, Git, web and custom calls it records a tool-tape intent and completion for replay. Native file operations follow the edit replay path. Authorized workspace effects become receipts, invalidate stale reads and checks, and appear in the next frame. Executable-bit changes and other unauthorized effects are restored. Reconciliation refuses to promote effects if its 16 MiB per-file, 256 MiB aggregate or 20,000-entry bound is exceeded. Job state is snapshotted once for the request.
7. **Done.** When the model calls `done`, the engine checks the result and stores it as the final candidate.
8. **After checks.** StateFrame runs your checks again on that stored candidate, prints the diff and an outcome line, and with `--in-place` applies it to your repository.
9. **Inspect.** `stateframe inspect <run-dir>` reads `wire/NNNN.request.json`, so what you see is what the model saw.

The tool schemas, bounded observations and job section all use the request's prompt budget. Shell, job and custom-command output delivery shrinks to the measured remaining room; the full taped observation stays in the run directory within collector limits. The shipped product profile uses the composed loop. The plain loop refuses Git, web, custom commands and background jobs before execution; lean mode is unsupported by the product profile. [Agent tools](tools.md) lists arguments, policies and replay classes.
