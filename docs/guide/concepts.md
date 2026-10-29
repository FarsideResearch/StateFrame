# Concepts

Diagrams for everything on this page: [ARCHITECTURE.md](../../ARCHITECTURE.md).

## Typed state frame instead of chat history

A chat agent resends the whole conversation every turn, so the prompt grows with the task. StateFrame does not. Each request carries:

1. the system instructions,
2. the tool definitions,
3. the task,
4. a **typed state frame**: named sections holding what the task needs right now (current check results, the files and code in play, what each recent action did, what is still owed),
5. the **newest exchange**: the model's last action and its result.

Older turns are not sent. Their content already lives in the typed state, and the full record stays in the run directory, where `stateframe inspect` shows it.

## What decides what goes in the frame

The engine, not the model. Before each send it fits the frame and the selected source code into the prompt budget (33,344 tokens by default, the size it was built and measured with; see [Context size](running.md#context-size)). When something has to give, it trims lower-priority detail; it never summarizes away the authoritative state, and it always keeps a ledger of what was read. If a task cannot fit what it needs, the engine stops with a clear reason instead of sending a broken request.

## Receipts and ledgers

Every action the engine runs (a read, an edit, a test run) produces a host-recorded result. Facts the engine measured, such as which tests pass, which files were read and what is still owed, are handed to the model plainly in the frame. The model decides what to do; the engine owns the bookkeeping. The task ledger is schema-validated.

## Behavior is controlled by code

The engine owns the tool menu, the guards, the decision that a task is done, the test requirements and exactly what the model sees each turn. The model cannot step outside those rails. You change behavior the way you change software: with profiles and switches that are versioned and testable (see [running.md](running.md)). Many switches ship disabled by default; they are still part of the engine.

## Done means checked

The model can say it is done; the engine decides. A candidate is accepted only after the engine's checks pass, protected tests cannot be edited by the agent, and StateFrame runs your checks again on the stored result before reporting an outcome.

## Run history on disk

The run's history lives in the run directory on disk, not in process memory, and the engine reads back what it needs. This is always on. Disk use grows with the run; see Requirements in the README.

## Checkpoints and resume

`stateframe resume <run-dir>` replays the recorded model responses (no model calls) up to where the run stopped, checks that the next request it rebuilds matches the recorded one byte for byte, and continues live. The engine can also save signed checkpoints of its state for a faster restore (`STATEFRAME_CHECKPOINT=save`); they are off by default in this release because a checkpoint restore was not faithful in our tests. Resume checks every restore, and anything wrong with a checkpoint falls back to the full replay, which is always correct and just slower.

## Determinism

Request ids are not random, and volatile test and tool output (durations, memory addresses in tracebacks, run-directory paths) is normalized before the model sees it, with the raw output kept in the run journal. Identical inputs give identical requests on the measured workload apart from three clock fields and an edit-receipt ID; this is not full determinism, because the model's own output is not deterministic. See KNOWN_LIMITATIONS.md.

## Why this matters

The README's "Why StateFrame" section lists the benefits, each marked measured or by design.
