# Run storage and retention

StateFrame keeps run records needed for inspect and resume in each run directory. The engine's authoritative model exchanges remain under `wire/`; retention never changes those files.

## What a run stores

- `wire/` contains the exact requests and responses exchanged with the model. It is never compacted by finish-time cleanup or `stateframe gc`.
- `requests/NNNN.json` contains preflight request records. These are records made before the provider request and are not necessarily the same bytes as `wire/`. At finish, StateFrame stores their exact original bytes in `requests.pack.xz`, a compressed tar archive, then removes the loose files. `stateframe inspect`, summaries and test helpers read both formats. Older runs with loose records remain supported.
- `worker/measurement-runs/*/events.jsonl` is sealed as `events.jsonl.xz` (about 12x smaller), and the journal's compressed blobs are appended to `blobs/pack-0001.sfpk` with a rebuildable index. Every reader goes through one journal store that reads both the sealed and the loose layout, so older runs need no conversion.
- `checkpoints/` and `env/` are removed once the terminal engine result cannot be resumed. A stopped, cancelled, killed, crashed, budget-stopped or interrupted run keeps resume state.
- `env-manifest.json` records the removed environment's Python version and implementation, platform and machine, interpreter paths, sorted installed distributions with their RECORD hashes, editable/direct URL metadata, and file and byte counts. `env-sites.json` stays in place.
- If a run environment is refused while it is being prepared (for example an import path that needs `--shell-read`), the run continues without it and the partial copy is removed.

Finish-time cleanup is best effort. If one storage action fails, StateFrame logs a warning and preserves the run result.

## Retention switches

Finished runs drop checkpoints and the run-local Python environment by default. Keep the environment with `--keep-run-env` on `run` or `resume`, or set `STATEFRAME_KEEP_RUN_ENV=on`. The default is `off`. Checkpoint retention continues to use `--keep-checkpoints` or `STATEFRAME_KEEP_CHECKPOINTS=1`.

`STATEFRAME_ENV_SEED=auto|copy` is read on the host while preparing a run environment. `auto` (the default) keeps one read-only copy of each source environment under `~/.stateframe/env-seeds/` and clones runs from it: a copy-on-write clone on APFS and on Linux filesystems with reflinks, hard links for package files on ext4, and private copies for anything a run may write. `copy` always copies the source environment and does not create a seed reference.

## Inspect or reclaim old run storage

`stateframe gc` previews storage that can be reclaimed. With no roots it scans `~/.stateframe/runs`; pass one or more run directories or run trees to scan elsewhere. By default it considers runs at least 14 days old and keeps all newer runs. Use `--older-than DAYS` to change the age threshold and `--keep-last N` to retain the newest N runs per root.

```bash
stateframe gc
stateframe gc --older-than 30 --keep-last 10
stateframe gc --apply
stateframe gc /path/to/run --apply
```

`--apply` removes checkpoints and run environments from finished runs and seals measurement and preflight records. A live `.run.lock` or `.engine.lock` skips that run.

Environment seeds are shared by every run, including runs kept outside `~/.stateframe/runs` with `--run-dir`. `gc` with no roots prunes a seed only when no scanned run references it and it has not been used for `--older-than` days; each new run touches the seed it clones from. With explicit roots, `gc` leaves seeds alone unless you add `--prune-seeds`. Old `quarantine-*` seeds go under the same rules. A pruned seed costs nothing but time: the next run rebuilds it, and `stateframe rehydrate` rebuilds from the manifest instead. Unfinished runs are never changed unless `--include-unfinished` is passed; applying that option prints that resume will be lost. The JSON view is available with `--json`.

The dry-run table estimates reclaimed bytes per run for environments, checkpoints, measurement journals, blobs and preflight requests. These are storage estimates; hard-linked or copy-on-write files can make physical disk savings differ.

## Rebuild a removed environment

Use the saved manifest to rebuild `env/`:

```bash
stateframe rehydrate <run-dir>
stateframe rehydrate <run-dir> --dry-run
stateframe rehydrate <run-dir> --offline
```

Rehydration first looks for the run's seed and restores it exactly when it is still valid. Otherwise, or if the seed cannot be restored, it says so and creates a virtual environment with the recorded interpreter, which must match the saved Python version and implementation. It installs each recorded distribution at its exact version, and local or editable projects (such as an editable StateFrame checkout) from their recorded directory. Offline mode disables package indexes and searches local wheel caches and wheelhouse directories.

The result is compared with the manifest by name, version and package contents. Rows pip writes differently on every install (bytecode, its own bookkeeping files, launcher scripts) are ignored, and an editable project is compared by its source location, since its code lives there. Any other difference is printed as a mismatch and the command exits nonzero; the rebuilt environment remains available for inspection. Rehydration refuses to overwrite an existing `env/`.

Exact rebuilds need the recorded interpreter and either package-index access or the required wheels locally. Platform-specific packages may not be available from a different host.
