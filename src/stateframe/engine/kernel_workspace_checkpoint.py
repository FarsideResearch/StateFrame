"""Signed, object-backed engine checkpoint boundaries."""
from __future__ import annotations

import hmac
import io
import os
import pickle
import shutil
import sys
import tempfile
import weakref
import zlib
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

if __package__:
    from . import kernel_checkpoint as engine
    from . import kernel_workspace_candidate as records
else:
    import kernel_checkpoint as engine
    import kernel_workspace_candidate as records

if TYPE_CHECKING:
    from kernel_workspace_objects import ObjectLease
    from kernel_workspace_refs import Ref, RootKey

SCHEMA = "stateframe-engine-checkpoint/2"
STATE_SCHEMA = "stateframe-checkpoint-state/1"
META_SCHEMA = "stateframe-checkpoint-meta/1"
BLOB_PAGE_SCHEMA = "stateframe-checkpoint-blobs/1"
_FIELDS = frozenset({"schema", "namespace", "tree", "state", "run_tree", "restore_meta",
                     "step", "sends", "seq", "python", "engine_files_sha256", "signature"})
_CONTROL_LIMIT = records.OBJECT_LIMIT
_PAGE_SIZE = 256
_MAX_PAYLOAD_BYTES = 1 << 30


@dataclass(frozen=True)
class Checkpoint:
    tree: str
    state: Ref
    run_tree: str
    restore_meta: Ref


def _checkpoint(value: dict) -> tuple[Checkpoint, tuple[Ref, ...]]:
    from kernel_workspace_refs import Ref

    if (set(value) != _FIELDS or value["schema"] != SCHEMA
            or any(type(value[key]) is not int or value[key] < 0 for key in ("step", "sends", "seq"))
            or not isinstance(value["python"], list) or len(value["python"]) != 3
            or any(type(part) is not int or part < 0 for part in value["python"])):
        raise ValueError("invalid workspace checkpoint header")
    for name in ("namespace", "engine_files_sha256", "signature"):
        records.digest(value[name])
    checkpoint = Checkpoint(records.digest(value["tree"]), records.typed_ref(value["state"], "state"),
                            records.digest(value["run_tree"]), records.typed_ref(value["restore_meta"], "meta"))
    return checkpoint, (Ref("tree", checkpoint.tree), checkpoint.state,
                        Ref("tree", checkpoint.run_tree), checkpoint.restore_meta)


def _checkpoint_record(lease: ObjectLease, checkpoint: Checkpoint, *,
                       step: int, sends: int, seq: int) -> dict:
    value = {
        "schema": SCHEMA, "namespace": records._namespace(lease._store),
        "tree": checkpoint.tree, "state": records.ref_json(checkpoint.state),
        "run_tree": checkpoint.run_tree, "restore_meta": records.ref_json(checkpoint.restore_meta),
        "step": step, "sends": sends, "seq": seq, "python": list(sys.version_info[:3]),
        "engine_files_sha256": engine.engine_fingerprint(bind_host=True),
    }
    value["signature"] = engine._signature(value, checkpoint.state.oid)
    return value


def save_checkpoint(lease: ObjectLease, key: RootKey, checkpoint: Checkpoint, *,
                    run: Path, step: int, sends: int, seq: int) -> Path:
    """Publish a signed checkpoint and only then advance the run's discoverable head."""
    root = run / "checkpoints"
    final = root / key.key
    value = _checkpoint_record(lease, checkpoint, step=step, sends=sends, seq=seq)
    _, edges = _checkpoint(value)
    if final.exists():
        if records.read_mirror(final / engine.META_FILE) != records.canonical(value):
            raise engine.CheckpointError("workspace checkpoint conflicts with an existing boundary")
        records.read_committed(lease, key, value, edges)
        return final
    previous = engine.publication_candidates(root) if engine._atomic_checkpoints() else None
    if previous is None:
        previous = engine._checkpoints(root)
    with tempfile.TemporaryDirectory(prefix=".workspace-save-", dir=root) as temporary:
        stage = Path(temporary) / "boundary"
        stage.mkdir(mode=0o700)
        records.write_mirror(stage / engine.META_FILE, records.canonical(value))
        records.commit_record(lease, key, value, edges)
        engine._publish_checkpoint(root, stage, final, run)
    _retire_pruned(lease, run, previous)
    return final


def _retire_pruned(lease: ObjectLease, run: Path, candidates: list[Path] | tuple[Path, ...]) -> None:
    from kernel_workspace_refs import read_root, retire_root

    for directory in candidates:
        if not directory.exists():
            key = records._key(run, "checkpoint", directory.name)
            ref = read_root(lease, key)
            if ref is not None:
                retire_root(lease, key, expected=ref)


def _load_checkpoint(run: Path, directory: Path, lease: ObjectLease) -> tuple[dict, Checkpoint]:
    if (directory.parent != run / "checkpoints" or not engine._owned_private(directory.parent)
            or not engine._owned_private(directory)):
        raise ValueError("invalid workspace checkpoint directory")
    value = records.decode(records.read_mirror(directory / engine.META_FILE))
    checkpoint, edges = _checkpoint(value)
    if value["namespace"] != records._namespace(lease._store):
        raise ValueError("workspace checkpoint namespace differs from its run")
    if not hmac.compare_digest(value["signature"], engine._signature(value, checkpoint.state.oid)):
        raise engine.CheckpointError("checkpoint signature does not verify")
    records.read_committed(lease, records._key(run, "checkpoint", directory.name), value, edges)
    return value, checkpoint


def read_checkpoint(run: Path, directory: Path) -> Checkpoint:
    from kernel_workspace_refs import open_run_store

    with open_run_store(run).lease() as lease:
        return _load_checkpoint(run, directory, lease)[1]


def _refs(value: object, edges: tuple[Ref, ...], *, field: str, kind: str) -> tuple[Ref, ...]:
    if not isinstance(value, dict) or not isinstance(value.get(field), list):
        raise ValueError("invalid checkpoint edge list")  # noqa: TRY004
    refs = tuple(records.typed_ref(item, kind) for item in value[field])
    if refs != edges:
        raise ValueError("checkpoint edge list does not match its payload")
    return refs


def _state(lease: ObjectLease, ref: Ref) -> tuple[Ref, Ref | None]:
    value, edges = records.read_envelope(lease, ref, STATE_SCHEMA)
    if not isinstance(value, dict) or set(value) != {"control", "blobs"}:
        raise ValueError("invalid semantic checkpoint state")
    control = records.typed_ref(value["control"], "blob")
    page = records.typed_ref(value["blobs"], "meta") if value["blobs"] is not None else None
    if edges != ((control,) if page is None else (control, page)):
        raise ValueError("checkpoint state edges do not match its payload")
    return control, page


def _blob_page(lease: ObjectLease, ref: Ref) -> tuple[tuple[Ref, ...], Ref | None]:
    value, edges = records.read_envelope(lease, ref, BLOB_PAGE_SCHEMA)
    if not isinstance(value, dict) or set(value) != {"blobs", "previous"} or not isinstance(value["blobs"], list):
        raise ValueError("invalid checkpoint object page")
    previous = records.typed_ref(value["previous"], "meta") if value["previous"] is not None else None
    blobs = _refs(value, edges if previous is None else edges[:-1], field="blobs", kind="blob")
    if not 1 <= len(blobs) <= _PAGE_SIZE or (previous is not None and (not edges or edges[-1] != previous)):
        raise ValueError("invalid checkpoint object page edges")
    return blobs, previous


def _metadata(lease: ObjectLease, ref: Ref) -> tuple[dict, tuple[Ref, ...]]:
    value, edges = records.read_envelope(lease, ref, META_SCHEMA)
    if (not isinstance(value, dict) or set(value) != {"workspace", "run", "mount", "clocks"}
            or not isinstance(value["clocks"], dict)
            or set(value["clocks"]) != {"monotonic", "perf_counter"}
            or any(type(clock) not in (int, float) for clock in value["clocks"].values())):
        raise ValueError("invalid workspace checkpoint restore metadata")
    records.relative(value["mount"])
    expected = tuple(records.typed_ref(value[key], "meta") for key in ("run", "workspace"))
    if expected != edges:
        raise ValueError("invalid workspace checkpoint restore metadata edges")
    return value, expected


def record_edges(lease: ObjectLease, ref: Ref):
    """Validate each typed checkpoint or candidate envelope before exposing its edges."""
    if ref.kind == "record":
        value, edges = records.read_envelope(lease, ref, records.RECORD_SCHEMA)
        if not isinstance(value, dict):
            raise ValueError("invalid workspace boundary payload")
        schema = value.get("schema")
        if schema == SCHEMA:
            expected = _checkpoint(value)[1]
        elif schema == records.SCHEMA:
            expected = records._candidate(value)[1]
        else:
            raise ValueError("unknown workspace boundary schema")
        if expected != edges:
            raise ValueError("workspace boundary edges do not match its payload")
        yield from edges
    elif ref.kind == "state":
        yield from (edge for edge in _state(lease, ref) if edge is not None)
    elif ref.kind == "meta":
        schema = records.decode(records.read_object(lease, ref)).get("schema")
        if schema == META_SCHEMA:
            yield from _metadata(lease, ref)[1]
        elif schema == BLOB_PAGE_SCHEMA:
            blobs, previous = _blob_page(lease, ref)
            yield from blobs
            if previous is not None:
                yield previous
        elif schema == records.PROJECTION_SCHEMA:
            records._projection(lease, ref)
            yield from records.read_envelope(lease, ref, schema)[1]
        elif schema == records.SEAL_SCHEMA:
            records._seal_projection(lease, ref)
        else:
            raise ValueError("unknown workspace record metadata schema")
    else:
        raise ValueError("unexpected workspace record reference kind")


class _StateCapture(engine._Pickler):
    """Keep large immutable payloads out of the compressed control graph."""

    def __init__(self, stream, ctx, lease, prior):
        super().__init__(stream, ctx)
        self.lease = lease
        self.prior = prior
        self.leaves = {}
        self.references = {}
        self.pending = {}
        self.pending_bytes = 0

    def publish_pending(self):
        """Publish small leaves in bounded batches before the state root is committed."""
        if not self.pending:
            return
        from kernel_workspace_objects import ObjectRef

        self.lease.publish_many(
            (ObjectRef(oid, len(data)), lambda data=data: io.BytesIO(data))
            for oid, data in self.pending.items()
        )
        self.pending.clear()
        self.pending_bytes = 0

    def persistent_id(self, obj):
        existing = super().persistent_id(obj)
        if existing is not None or type(obj) not in (bytes, str) or len(obj) < 256:
            return existing
        from kernel_workspace_refs import Ref

        identity = id(obj)
        if identity in self.leaves:
            return ("workspace-text" if type(obj) is str else "workspace-bytes",
                    self.leaves[identity][2], self.leaves[identity][1].oid)
        previous = self.prior.get(identity)
        if (previous is not None and previous[0] is obj and previous[2] is not None
                and _blob_identity(self.lease, previous[1]) == previous[2]):
            ref = previous[1]
            stamp = previous[2]
        else:
            data = obj.encode("utf-8", "surrogatepass") if type(obj) is str else obj
            engine._check_registered_secrets(data)
            if len(data) <= records.OBJECT_LIMIT:
                ref = Ref("blob", sha256(data).hexdigest())
                if ref.oid not in self.pending:
                    if self.pending and self.pending_bytes + len(data) > 32 << 20:
                        self.publish_pending()
                    self.pending[ref.oid] = data
                    self.pending_bytes += len(data)
                if self.pending_bytes >= 32 << 20:
                    self.publish_pending()
                stamp = None
            else:
                storage = self.lease.publish_stream(io.BytesIO(data), size=len(data), max_bytes=_MAX_PAYLOAD_BYTES)
                ref = Ref("blob", storage.digest)
                stamp = _blob_identity(self.lease, ref)
        ordinal = len(self.leaves)
        self.leaves[identity] = obj, ref, ordinal, stamp
        self.references.setdefault(ref.oid, ref)
        return ("workspace-text" if type(obj) is str else "workspace-bytes", ordinal, ref.oid)


def _blob_identity(lease: ObjectLease, ref: Ref) -> tuple[int, ...] | None:
    # Only a successfully authenticated publication can populate the memo.
    # Keep that attestation tied to the same immutable private file.
    try:
        info = os.stat(ref.oid, dir_fd=lease._check(), follow_symlinks=False)
    except FileNotFoundError:
        return None
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


class _StateRestore(engine._Unpickler):
    def __init__(self, stream, ctx, lease, blobs):
        super().__init__(stream, ctx)
        self.lease = lease
        self.blobs = blobs
        self.loaded = {}

    def persistent_load(self, pid):
        if pid[0] not in ("workspace-bytes", "workspace-text"):
            return super().persistent_load(pid)
        if len(pid) != 3:
            raise engine.CheckpointError("invalid checkpoint payload reference")
        _, ordinal, oid = pid
        if type(ordinal) is not int or ordinal < 0 or oid not in self.blobs:
            raise engine.CheckpointError("checkpoint refers to an unauthenticated byte payload")
        existing = self.loaded.get(ordinal)
        if existing is None:
            leaf = self.lease.resolve(oid, max_bytes=_MAX_PAYLOAD_BYTES)
            output = io.BytesIO()
            self.lease.copy_verified(leaf, output, max_bytes=_MAX_PAYLOAD_BYTES)
            data = output.getvalue()
            if pid[0] == "workspace-text":
                try:
                    value = data.decode("utf-8", "surrogatepass")
                except UnicodeDecodeError as error:
                    raise engine.CheckpointError("invalid checkpoint text payload") from error
            else:
                value = data
            existing = oid, pid[0], value
            self.loaded[ordinal] = existing
        if existing[:2] != (oid, pid[0]):
            raise engine.CheckpointError("checkpoint aliases disagree on a byte payload")
        return existing[2]


def _save_state(lease: ObjectLease, payload: dict, ctx: dict, previous: dict) -> tuple[Ref, dict]:
    control = io.BytesIO()
    writer = engine._StateWriter(control)
    capture = _StateCapture(writer, ctx, lease, previous)
    capture.dump(payload)
    capture.publish_pending()
    if writer.count > _CONTROL_LIMIT:
        raise engine.CheckpointError("checkpoint control state exceeds its object budget")
    writer.finish()
    control_ref = records.publish(lease, "blob", control.getvalue())
    refs = tuple(capture.references[oid] for oid in sorted(capture.references))
    page = None
    for start in range(0, len(refs), _PAGE_SIZE):
        group = refs[start:start + _PAGE_SIZE]
        page = records.publish_envelope(lease, "meta", BLOB_PAGE_SCHEMA,
                                        {"blobs": [records.ref_json(ref) for ref in group],
                                         "previous": records.ref_json(page) if page else None},
                                        group + ((page,) if page else ()))
    state = records.publish_envelope(lease, "state", STATE_SCHEMA,
                                     {"control": records.ref_json(control_ref),
                                      "blobs": records.ref_json(page) if page else None},
                                     (control_ref,) + ((page,) if page else ()))
    # Hold every large immutable leaf for one successful capture generation.
    # The writer fences the projection before it becomes the next memo; a
    # fixed cap would repeatedly encode unchanged history beyond that cap.
    retained = {}
    for identity, (obj, ref, _, stamp) in capture.leaves.items():
        if sys.getsizeof(obj) < 4096:
            continue
        # Small leaves were batched, so their file did not exist when first
        # encountered. Stamp only after publication has completed.
        stamp = stamp if stamp is not None else _blob_identity(lease, ref)
        if stamp is None:
            raise engine.CheckpointError("checkpoint payload was not published")
        retained[identity] = obj, ref, stamp
    return state, retained


def _read_state(lease: ObjectLease, ref: Ref) -> tuple[bytes, set[str]]:
    control, page = _state(lease, ref)
    seen = set()
    blobs = set()
    while page is not None:
        if page.oid in seen or len(seen) >= 4096:
            raise ValueError("checkpoint byte payload pages exceed the graph budget")
        seen.add(page.oid)
        group, page = _blob_page(lease, page)
        for leaf in group:
            lease.resolve(leaf.oid, max_bytes=_MAX_PAYLOAD_BYTES)
            blobs.add(leaf.oid)
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(records.read_object(lease, control), _CONTROL_LIMIT + 1)
    except zlib.error as error:
        raise ValueError("checkpoint control state is damaged") from error
    if not decoder.eof or decoder.unused_data or len(raw) > _CONTROL_LIMIT:
        raise ValueError("checkpoint control state exceeds its object budget or is damaged")
    return raw, blobs


def validate_directory(directory: Path, *, compatible: bool) -> dict:
    from kernel_workspace_refs import open_run_store

    run = directory.parent.parent
    try:
        with open_run_store(run).lease() as lease:
            value, checkpoint = _load_checkpoint(run, directory, lease)
            if compatible:
                _compatible(value)
            from kernel_workspace_tree_io import lookup_ref

            if (lookup_ref(lease, checkpoint.tree, ()) is None
                    or lookup_ref(lease, checkpoint.run_tree, ()) is None):
                raise ValueError("workspace checkpoint tree root is unavailable")
            _metadata(lease, checkpoint.restore_meta)
            _read_state(lease, checkpoint.state)
            return value
    except (OSError, ValueError, pickle.PickleError) as error:
        raise engine.CheckpointError("workspace checkpoint could not be verified") from error


def _compatible(value: dict) -> None:
    if value["python"] != list(sys.version_info[:3]):
        raise engine.CheckpointError("checkpoint was written by another Python version")
    if value["engine_files_sha256"] != engine.engine_fingerprint(bind_host=True):
        raise engine.CheckpointError("checkpoint was written by another engine build")


def _workspace(frame, run: Path) -> Path:
    actual = frame.f_locals.get("root")
    if not isinstance(actual, Path):
        actual = run / "workspace"
    path = actual.relative_to(run)
    if not path.parts or path.parts[0] in engine.SKIP_RUN_ENTRIES:
        raise engine.CheckpointError("workspace checkpoint mount is outside the admitted run tree")
    return actual


class _RunSide:
    """Reuse committed run objects and observe the mutable tail via filesystem events."""

    def __init__(self, run: Path, mount: Path, home: Path, store):
        from kernel_workspace_feed import FeedUpdate, WorkspaceFeed
        from kernel_workspace_refs import RUN_BINDING
        from kernel_workspace_state import WorkspaceFence, WorkspaceState

        self.run = run
        self.mount = os.fsencode(mount.as_posix())
        self.home = home
        self.home_subtree = home if home.is_relative_to(run) else None
        skip = frozenset(map(os.fsencode, engine.SKIP_RUN_ENTRIES | {RUN_BINDING}))
        mount_path = self.mount
        home_path = (os.fsencode(home.relative_to(run).as_posix())
                     if self.home_subtree is not None else None)

        # The workspace mount is fenced separately. Keep its subtree out of
        # both the run-tail observation and the change feed.
        def include(path: bytes) -> bool:
            return (path.split(b"/", 1)[0] not in skip
                    and path != mount_path and not path.startswith(mount_path + b"/")
                    and (home_path is None or
                         (path != home_path and not path.startswith(home_path + b"/"))))

        self._in_scope = include
        self.links: dict[tuple[bytes, ...], bool] = {}
        private = Path(tempfile.mkdtemp(prefix="checkpoint-fence-", dir=store._path))
        class RunFeed(WorkspaceFeed):
            def poll(self, **kwargs) -> FeedUpdate:
                update = super().poll(**kwargs)
                if update.requires_sweep:
                    return update
                return FeedUpdate(paths=tuple(path for path in update.paths if include(path)))

        feed = RunFeed(run, enabled=True, include_directory=include)
        self._cleanup = weakref.finalize(self, _close_run_side, feed, private, os.getpid())
        try:
            self.state = WorkspaceState(WorkspaceFence.create(run, private / "fence"), feed=feed)
        except BaseException:
            self._cleanup()
            raise

    def close(self) -> None:
        self.state._reset_spool()
        self._cleanup()

    def _select(self, parts, _is_directory: bool) -> bool:
        if not self._in_scope(b"/".join(parts)):
            return False
        path = self.run.joinpath(*map(os.fsdecode, parts))
        linked = path.is_symlink()
        excluded = engine._snapshot_path_excluded(
            path, self.home, self.home_subtree, set(), is_symlink=linked,
        )
        if linked:
            self.links[parts] = excluded
        else:
            self.links.pop(parts, None)
        return not excluded

    def changed_links(self) -> list[bytes]:
        # Targets outside the run can redirect an unchanged link into or out
        # of the protected home without generating an event under the run.
        changed = []
        for parts, excluded in tuple(self.links.items()):
            path = self.run.joinpath(*map(os.fsdecode, parts))
            if not path.is_symlink():
                self.links.pop(parts)
                changed.append(b"/".join(parts))
            elif engine._real_path(path).is_relative_to(self.home) != excluded:
                changed.append(b"/".join(parts))
        return changed

    def pin(self):
        return self.state.pin("checkpoint-run", coverage=lambda: (self.mount, self._select),
                              recheck=self.changed_links())


def _run_side(run, mount, home, store) -> _RunSide:
    key = (os.getpid(), run, mount, home, store._objects_identity, store._path)
    prior = engine._STATE.get("object_run_side")
    if prior is not None and prior[0] == key:
        return prior[1]
    if prior is not None:
        prior[1].close()
        engine._STATE.pop("object_run_side", None)
    side = _RunSide(run, mount, home, store)
    engine._STATE["object_run_side"] = key, side
    return side


def _close_run_side(feed, private: Path, owner: int) -> None:
    try:
        feed.close()
    finally:
        if owner == os.getpid():
            shutil.rmtree(private, ignore_errors=True)


def capture(frame, step: int, run: Path) -> Path:
    """Pin both tree projections and semantic state before publishing their objects."""
    from kernel_request_auth_identity import _installation_home
    from kernel_workspace_consumers import consumers_for
    from kernel_workspace_refs import open_run_store

    stage = "admission"
    try:
        if not engine._SIGNING_KEY:
            raise engine.CheckpointError("no checkpoint signing key registered by the host")
        actual = _workspace(frame, run)
        consumers = consumers_for(actual)
        if consumers is None:
            raise engine.CheckpointError("checkpoint has no fenced workspace binding")
        identity = engine._verify_stateframe_home_identity(run)
        engine._STATE["protected_home_identity"] = identity
        generation = engine._SECRET_GENERATION
        store = open_run_store(run)
        resolved = run.resolve(strict=True)
        mount = actual.relative_to(run)
        home = engine._real_path(Path(identity["realpath"]) if identity else _installation_home())
        root = run / "checkpoints"
        engine._private_dir(root)
        key = records._key(run, "checkpoint", f"{step:04d}-{os.getpid()}")
        memo_key = (resolved, mount, home, store._objects_identity, records._namespace(store),
                    generation, tuple(engine._SECRETS))
        with ExitStack() as stack:
            lock = stack.enter_context((root / ".checkpoint.lock").open("a+b"))
            engine.fcntl.flock(lock, engine.fcntl.LOCK_EX | engine.fcntl.LOCK_NB)
            if (root / key.key).exists():
                raise engine.CheckpointError("checkpoint already exists at this boundary")
            stage = "workspace"
            live = stack.enter_context(consumers.state.pin("checkpoint-workspace"))
            live.check_retention()
            stage = "run_tail"
            side = _run_side(resolved, mount, home, store)
            side_snapshot = stack.enter_context(side.pin())
            side_snapshot.check_retention()

            stage = "semantic_state"
            clocks = {"monotonic": engine.time.monotonic(), "perf_counter": engine.time.perf_counter()}
            payload, ctx = engine.collect(frame)
            payload["frame"]["step"] = step - 1
            sends = engine._sends(run)

            lease = stack.enter_context(store.lease())
            state, next_memo = _save_state(lease, payload, ctx, _state_memo.get(memo_key, {}))
            stage = "objects"
            tree, workspace_meta = live.persist_with_metadata(lease)
            run_tree, run_meta = side_snapshot.persist_with_metadata(lease)
            restore_meta = records.publish_envelope(
                lease, "meta", META_SCHEMA,
                {"workspace": records.ref_json(workspace_meta), "run": records.ref_json(run_meta),
                 "mount": mount.as_posix(), "clocks": clocks},
                (run_meta, workspace_meta),
            )
            stage = "workspace"
            live.check_quiescent()
            stage = "run_tail"
            side_snapshot.check_quiescent()
            if side.changed_links():
                side.state.invalidate()
                raise engine.CheckpointError("checkpoint run link policy changed")
            stage = "publication"
            engine._verify_stateframe_home_identity(run)
            if engine._SECRET_GENERATION != generation:
                raise engine.CheckpointError("STATEFRAME_HOME_IDENTITY_CHANGED")
            seq = engine._next_seq(root)
            final = save_checkpoint(
                lease, key, Checkpoint(tree, state, run_tree, restore_meta),
                run=run, step=step, sends=sends, seq=seq,
            )
            engine._STATE["next_seq"] = seq + 1
        _state_memo.clear()
        _state_memo[memo_key] = next_memo
        return final
    except Exception as error:
        if getattr(error, "boundary_recovery_stop", False):
            raise
        raise engine.ObjectCaptureError(stage, engine._failure_reason(error)) from error


_state_memo: dict = {}


ACTIVATION = ".workspace-activation.json"


def _activation_names(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > 4096:
        raise ValueError("invalid workspace activation entry list")
    for name in value:
        if len(records.relative(name)) != 1 or name in engine.SKIP_RUN_ENTRIES:
            raise ValueError("workspace activation contains a protected or invalid entry")
    if value != sorted(set(value)):
        raise ValueError("invalid workspace activation entry order")
    return tuple(value)


def _move_entry(source: Path, destination: Path) -> None:
    if os.path.lexists(destination):
        raise engine.CheckpointError("workspace activation destination is occupied")
    source.rename(destination)
    engine._fsync_dir(source.parent)
    engine._fsync_dir(destination.parent)


def recover_activation(run: Path) -> None:
    """Roll an interrupted multi-entry restore back before archiving or execution."""
    root = run / "checkpoints"
    marker = root / ACTIVATION
    if not os.path.lexists(marker):
        return
    value = records.decode(records.read_mirror(marker, limit=256 << 10))
    if (set(value) != {"run", "stage", "prior", "old", "new", "signature"}
            or value["run"] != os.fspath(run.resolve(strict=True))):
        raise ValueError("invalid workspace activation record")
    signature = records.digest(value["signature"])
    if not hmac.compare_digest(signature, engine._signature(value, "workspace-activation")):
        raise ValueError("workspace activation record authentication failed")
    for field, prefix in (("stage", ".workspace-restore-"), ("prior", "setup-discard.")):
        path = value[field]
        if not isinstance(path, str) or not path.startswith(prefix) or len(records.relative(path)) != 1:
            raise ValueError("invalid workspace activation directory")
    incoming = root / value["stage"] / "run"
    prior = root / value["prior"]
    if any(not path.is_dir() or not engine._owned_private(path)
           for path in (root, incoming.parent, incoming, prior)):
        raise ValueError("workspace activation directory is unavailable")
    for name in _activation_names(value["new"]):
        if not os.path.lexists(incoming / name):
            _move_entry(run / name, incoming / name)
    for name in _activation_names(value["old"]):
        if os.path.lexists(prior / name):
            _move_entry(prior / name, run / name)
    marker.unlink()
    engine._fsync_dir(root)


@contextmanager
def _activate(run: Path, incoming: Path, prior: Path):
    root = run / "checkpoints"
    marker = root / ACTIVATION
    old = sorted(item.name for item in run.iterdir() if item.name not in engine.SKIP_RUN_ENTRIES)
    new = sorted(item.name for item in incoming.iterdir())
    _activation_names(old)
    _activation_names(new)
    value = {"run": os.fspath(run.resolve(strict=True)), "stage": incoming.parent.name,
             "prior": prior.name, "old": old, "new": new}
    value["signature"] = engine._signature(value, "workspace-activation")
    records.write_mirror(marker, records.canonical(value))
    try:
        for name in old:
            _move_entry(run / name, prior / name)
        for name in new:
            _move_entry(incoming / name, run / name)
        yield
    except BaseException:
        recover_activation(run)
        raise
    else:
        marker.unlink()
        engine._fsync_dir(root)


def restore_checkpoint(run: Path, directory: Path, frame) -> dict:
    from kernel_workspace_refs import open_run_store
    from kernel_workspace_tree_io import CheckoutPolicy, checkout_tree

    try:
        with open_run_store(run).lease() as lease:
            value, checkpoint = _load_checkpoint(run, directory, lease)
            _compatible(value)
            meta, (run_meta, workspace_meta) = _metadata(lease, checkpoint.restore_meta)
            control, blobs = _read_state(lease, checkpoint.state)
            policy = CheckoutPolicy(True, True, 0o7777, _MAX_PAYLOAD_BYTES)
            root = run / "checkpoints"
            recover_activation(run)
            stage = Path(tempfile.mkdtemp(prefix=".workspace-restore-", dir=root))
            try:
                incoming = stage / "run"
                checkout_tree(lease, checkpoint.run_tree, incoming, metadata=run_meta, policy=policy)
                mount = incoming / meta["mount"]
                if not mount.parent.is_dir() or mount.parent.resolve() != mount.parent:
                    raise engine.CheckpointError("checkpoint workspace mount parent is unavailable")
                checkout_tree(lease, checkpoint.tree, mount, metadata=workspace_meta, policy=policy)
                # Preserve the setup tree outside the staged snapshot. Reducers
                # that reopen process resources need the final logical paths.
                prior = Path(tempfile.mkdtemp(prefix="setup-discard.", dir=root))
                with _activate(run, incoming, prior):
                    from kernel_workspace_consumers import rebind_after_checkpoint_activation

                    rebind_after_checkpoint_activation(run, run / meta["mount"])
                    cells, funcs = engine._frame_maps(frame)
                    ctx = {"framecells": cells, "framefuncs": funcs}
                    engine._SAVE.update(load_root=root, load_run_dir=run)
                    try:
                        payload = _StateRestore(io.BytesIO(control), ctx, lease, blobs).load()
                        engine._apply_restored_state(frame, payload)
                    finally:
                        engine._SAVE.clear()
                engine.install_continuous_clock(meta["clocks"])
            finally:
                if not (root / ACTIVATION).exists():
                    shutil.rmtree(stage)
            return value
    except (OSError, ValueError, pickle.PickleError) as error:
        raise engine.CheckpointError("workspace checkpoint could not be restored") from error
