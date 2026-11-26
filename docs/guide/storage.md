# Run storage and retention

StateFrame keeps run records needed for inspect and resume in each run directory. The engine's authoritative model exchanges remain under `wire/`; retention never changes those files.

## Installation home

`STATEFRAME_HOME` selects the installation home, defaulting to `~/.stateframe`. Keep it outside the run directory and repository; a home inside either tree, including through a symlink, is refused before the run starts. The installation home may contain run directories. StateFrame pins the home directory and `identity.key` at startup, verifies that pin before checkpointing or publishing exports, and refuses missing or replaced identities with `STATEFRAME_HOME_IDENTITY_CHANGED`. The identity key is excluded from checkpoints, exported patches and request bundles even when no provider API key is configured.

Direct engine integrations without a runner spec must explicitly set `STATEFRAME_CHECKPOINT_LEGACY_DIRECT_ENGINE=1` to use legacy checkpoint fixtures. This setting never relaxes a runner startup pin or disables byte protection for an existing installation key.

## What a run stores

- `wire/` contains the exact requests and responses exchanged with the model. It is never compacted by finish-time cleanup or `stateframe gc`.
- `summary.json` records the run outcome. Its `problems` section counts refusals, kills and timeouts by cause with the first occurrence of each. For a successful `--in-place` promotion, `apply-receipt.json` records the applied file paths and is durably published before `summary.json`. If summary publication is interrupted after changes are applied, inspect this receipt to identify the promoted files.
- `requests/NNNN.json` contains preflight request records. These are records made before the provider request and are not necessarily the same bytes as `wire/`. At finish, StateFrame stores their exact original bytes in `requests.pack.xz`, a compressed tar archive, then removes the loose files. `stateframe inspect`, summaries and test helpers read both formats. Older runs with loose records remain supported.
- `worker/measurement-runs/*/events.jsonl` is sealed as `events.jsonl.xz` (about 12x smaller), and the journal's compressed blobs are appended to `blobs/pack-0001.sfpk` with a rebuildable index. Every reader goes through one journal store that reads both the sealed and the loose layout, so older runs need no conversion.
- `checkpoints/` and `env/` are removed once the terminal engine result cannot be resumed. A stopped, cancelled, killed, crashed, budget-stopped or interrupted run keeps resume state.
- While a run is active, checkpoint snapshots keep audit-journal suffixes in `checkpoints/journal-cache/`; restore reassembles the exact `events.jsonl` bytes. A truncated or rewritten journal starts a fresh chunk lineage. New `state.pkl` files use zlib level 1; StateFrame verifies stored bytes before bounded decompression, then verifies the raw pickle digest. A checkpoint written by a different StateFrame release is refused by name rather than restored, so finish or restart a run across an upgrade.
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

Environment seeds are shared by every run, including runs kept outside `~/.stateframe/runs` with `--run-dir`. `gc` with no roots prunes a seed only when no scanned run references it and it has not been used for `--older-than` days; each new run touches the seed it clones from. With explicit roots, age-based pruning is off unless you add `--prune-seeds`. Old `quarantine-*` seeds go under the same age rules. After pruning, `gc` also enforces `STATEFRAME_ENV_SEED_MAX_GB` for any roots, including explicit ones. The cap may evict a seed referenced by a finished run, which rehydrates from its manifest when needed. Dry-run reports planned cap eviction without deleting seeds. A pruned seed costs nothing but time: the next run rebuilds it, and `stateframe rehydrate` rebuilds from the manifest instead. Unfinished runs are never changed unless `--include-unfinished` is passed; applying that option prints that resume will be lost. The JSON view is available with `--json`.

The dry-run table estimates reclaimed bytes per run for environments, checkpoints, measurement journals, blobs and preflight requests. These are storage estimates; hard-linked or copy-on-write files can make physical disk savings differ.

## Measure one run

Use `run_allocation` on a finished run or at a barrier where the engine and checkpoint writer have finished writing. Supply the seed directory recorded in `env-seed.json` if it still exists. For a cold-start experiment with a dedicated, initially empty seed store, pass the whole store to include its sidecar files. Omit `seed_root` when there is no seed.

```python
import json
from pathlib import Path
from stateframe.storage_gc import run_allocation

report = run_allocation(Path("/path/to/run"), seed_root=Path("/path/to/seed"))
print(json.dumps(report, indent=2))
```

Each top-level directory has a bucket, including a zero-byte entry when its tree contains no regular files. The `.` bucket contains regular files directly under the run root, including the spec, summary and packed preflight records. Logical bytes count every regular-file name. Allocated bytes use `st_blocks × 512`, counting each device and inode once per root. The root files are visited first, then directories in name order, so a hard link spanning two buckets is allocated to the first. Symlinks, directory metadata and sparse holes carry no file allocation in this metric. Distinct reflink inodes each contribute their filesystem-reported blocks, which can include shared extents.

`run.attributable_bytes` is the warm cost of the run after subtracting inodes shared with the seed. `seed.allocated_bytes` gives the separate cold seed charge. Their sum, `combined_allocated_bytes`, counts all regular-file allocations across both roots once. A seed on another device can require private copies, increasing the run's attributable bytes. User-kept environments and checkpoints are charged for as long as they remain. Interrupted runs retain their recovery state.

The five-send storage regression gates the library fixture at **100,000,000 attributable bytes**. It checks the prepared run, each of four published checkpoints during the next provider request, and the finished run. An overage fails the regression. Run execution, retention and resume follow their usual policies.

### Five-send reference measurements

Both fixtures use the bundled pricing edit and the same five scripted calls over loopback (read the test, read the implementation, edit, run the tests, finish). The library fixture adds all 17 Python source files from the bundled `packaging` 26.0 wheel, for 20 materialized source and test files totaling 260,886 bytes. The lookup-table fixture adds 128 generated Python modules with 512 fixed-width rows each, for 131 files totaling 8,456,570 bytes. Both have a project `.venv` built with the host interpreter's bundled pip and the repository's pytest, pluggy, iniconfig, packaging and Pygments wheels.

The reference host ran Linux x86-64 with bubblewrap, ext4 and Python 3.14.4. The private-copy experiment disabled workspace reflinks and refused both reflinks and hard links from the seed. It simulates the copy path on a single device. APFS, Linux reflink filesystems and an actual cross-device mount were outside this measurement.

| Attributable allocation (decimal MB) | Prepared | Send 2 | Send 3 | Send 4 | Send 5 | Retained | Cold seed separately |
|---|---:|---:|---:|---:|---:|---:|---:|
| Library, default | 8.1 | 36.4 | 42.2 | 45.4 | 49.0 | 6.4 | 20.5 |
| Library, private copy | 21.2 | 49.2 | 55.0 | 58.2 | 61.8 | 6.4 | 20.5 |
| Lookup tables, default | 33.6 | 200.1 | 279.9 | 316.3 | 393.0 | 168.8 | 20.5 |
| Lookup tables, private copy | 46.7 | 213.2 | 293.0 | 329.4 | 406.1 | 168.8 | 20.5 |

The lookup-table fixture is a separate stress experiment. It exceeds the target on the reference host. Its default fifth send holds 261.2 MB in checkpoints, 97.3 MB in the worker tree and 8.9 MB in the immutable parent. After normal cleanup the worker tree retains 141.8 MB, including 67.8 MB of candidate captures.

Run `python -m pytest -s tests/test_everyday_run_disk.py` to see the library measurements by bucket. To reproduce the lookup-table overage in both copy modes, run the command below. It applies the same limit at every barrier and exits with a budget assertion when the fixture exceeds it.

```bash
SF_TEST_DISK_LOOKUP_TABLES=1 python -m pytest -s tests/test_everyday_run_disk.py -k tables
```

In a separate interruption of the library fixture, the engine restored the first checkpoint and reissued the same 63,637 request bytes at send two. Stopping there left 67.9 MB attributable to the run, including its recovery state.

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

## Journal durability

The continuation journal fsyncs every row before `emit` returns, with one exception: `turn_timing` and `action_envelope_metrics` rows written before the first `continuation_worker_checkpoint` row exists in that journal wait for the next fsync (any later row, `sync()`, or `close()`). Once a checkpoint exists, every row, telemetry included, is fsynced immediately, so a lost unsynced suffix can never add or remove a post-checkpoint tail row and the resume decision cannot change. Receipts, seals, mutations, `request_submitted`, tool dispatch, `journal_observer_failed` and the checkpoint itself are always durable on return,. After SIGKILL nothing written is lost; after power loss only pre-checkpoint timing rows since the last durable row can be missing, as a clean suffix of the hash chain, and a torn final line is rejected on resume. Set `SF_JOURNAL_GROUP_COMMIT=0` to fsync every row.
