"""Immutable proposal records and their legacy export projection."""
from __future__ import annotations

import difflib
import hashlib
import io
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if __package__:
    from . import kernel_checkpoint as checkpoint
else:
    import kernel_checkpoint as checkpoint

if TYPE_CHECKING:
    from kernel_workspace_objects import ObjectLease
    from kernel_workspace_refs import Ref, RootKey
    from kernel_workspace_tree import Directory

SCHEMA = "stateframe-candidate-trees/1"
RECORD_SCHEMA = "stateframe-record/1"
PROJECTION_SCHEMA = "stateframe-candidate-projection/1"
RECORD_LIMIT = 16 << 10
OBJECT_LIMIT = 16 << 20
_DIGEST = re.compile(r"[a-f0-9]{64}")
_REF_KINDS = frozenset({"tree", "node", "blob", "state", "meta", "record"})
_FIELDS = frozenset({"schema", "proposals", "namespace", "parent_tree", "new_tree", "projection"})


@dataclass(frozen=True)
class Candidate:
    parent_tree: str
    new_tree: str


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _unique_fields(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate workspace record field")
        result[key] = value
    return result


def decode(raw: bytes) -> dict:
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_fields)
    if not isinstance(value, dict) or canonical(value) != raw:
        raise ValueError("workspace record is not canonical")
    return value


def digest(value: object) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError("invalid workspace object ID")
    return value


def typed_ref(value: object, kind: str | None = None) -> Ref:
    from kernel_workspace_refs import Ref

    if (not isinstance(value, list) or len(value) != 2
            or not isinstance(value[0], str) or value[0] not in _REF_KINDS
            or (kind is not None and kind != value[0])):
        raise ValueError("invalid workspace object reference")
    return Ref(value[0], digest(value[1]))


def ref_json(ref: Ref) -> list[str]:
    return [ref.kind, ref.oid]


def publish(lease: ObjectLease, kind: str, raw: bytes) -> Ref:
    from kernel_workspace_objects import ObjectRef
    from kernel_workspace_refs import Ref

    object_ref = ObjectRef(hashlib.sha256(raw).hexdigest(), len(raw))
    lease.publish_many(((object_ref, lambda: io.BytesIO(raw)),))
    return Ref(kind, object_ref.digest)


def read_object(lease: ObjectLease, ref: Ref, *, limit: int = OBJECT_LIMIT) -> bytes:
    stored = lease.resolve(ref.oid, max_bytes=limit)
    if stored.size <= OBJECT_LIMIT:
        return lease.read(stored)
    output = io.BytesIO()
    lease.copy_verified(stored, output, max_bytes=limit)
    return output.getvalue()


def publish_envelope(lease: ObjectLease, kind: str, schema: str, payload: object,
                     edges: tuple[Ref, ...] = ()) -> Ref:
    raw = canonical({"schema": schema, "payload": payload, "edges": [ref_json(ref) for ref in edges]})
    checkpoint._check_registered_secrets(raw)
    return publish(lease, kind, raw)


def read_envelope(lease: ObjectLease, ref: Ref, schema: str) -> tuple[object, tuple[Ref, ...]]:
    value = decode(read_object(lease, ref))
    if set(value) != {"schema", "payload", "edges"} or value["schema"] != schema:
        raise ValueError("invalid workspace object envelope")
    if not isinstance(value["edges"], list):
        raise ValueError("invalid workspace object edges")  # noqa: TRY004
    return value["payload"], tuple(typed_ref(edge) for edge in value["edges"])


def read_mirror(path: Path, *, limit: int = RECORD_LIMIT) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or info.st_mode & 0o022 or info.st_size > limit):
            raise ValueError("workspace record mirror is not a bounded private file")
        raw = stream.read(limit + 1)
    if len(raw) != info.st_size:
        raise ValueError("workspace record mirror changed while reading")
    return raw


def write_mirror(path: Path, raw: bytes) -> None:
    """Expose only a complete mirror after the namespace root has committed."""
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".workspace-record-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if read_mirror(path) != raw:
                raise ValueError("workspace boundary was captured twice with different contents") from None
        checkpoint._fsync_dir(path.parent)
    finally:
        os.unlink(temporary)


def _namespace(store) -> str:
    return digest(store._path.name)


def _key(run: Path, kind: str, name: str) -> RootKey:
    from kernel_workspace_refs import RootKey, workspace_run_id

    return RootKey(workspace_run_id(run), kind, name)


def read_committed(lease: ObjectLease, key: RootKey, value: dict,
                   edges: tuple[Ref, ...]) -> None:
    from kernel_workspace_refs import read_root

    ref = read_root(lease, key)
    if ref is None or ref.kind != "record":
        raise ValueError("workspace boundary has no committed root")
    payload, recorded_edges = read_envelope(lease, ref, RECORD_SCHEMA)
    if payload != value or recorded_edges != edges:
        raise ValueError("workspace mirror differs from the committed root")


def commit_record(lease: ObjectLease, key: RootKey, value: dict,
                  edges: tuple[Ref, ...]) -> None:
    from kernel_workspace_refs import commit_root

    ref = publish_envelope(lease, "record", RECORD_SCHEMA, value, edges)
    commit_root(lease, key, ref, expected=None)


def _candidate(value: dict) -> tuple[Candidate, tuple[Ref, ...]]:
    from kernel_workspace_refs import Ref

    if (set(value) != _FIELDS or value["schema"] != SCHEMA
            or type(value["proposals"]) is not int or value["proposals"] < 0):
        raise ValueError("invalid workspace candidate record")
    digest(value["namespace"])
    candidate = Candidate(digest(value["parent_tree"]), digest(value["new_tree"]))
    return candidate, (Ref("tree", candidate.parent_tree), Ref("tree", candidate.new_tree),
                       typed_ref(value["projection"], "meta"))


def save_candidate(lease: ObjectLease, key: RootKey, value: Candidate, *,
                   projection: Ref, destination: Path) -> None:
    """Commit a proposal tree pair and publish its small run-side mirror."""
    namespace = _namespace(lease._store)
    record = {"schema": SCHEMA, "proposals": _number(destination), "namespace": namespace,
              "parent_tree": value.parent_tree, "new_tree": value.new_tree,
              "projection": ref_json(projection)}
    _, edges = _candidate(record)
    if destination.exists() and read_mirror(destination) != canonical(record):
        raise ValueError("workspace proposal conflicts with an existing boundary")
    commit_record(lease, key, record, edges)
    write_mirror(destination, canonical(record))


def _number(boundary: Path) -> int:
    if boundary.suffix != ".json" or not re.fullmatch(r"[0-9]{1,16}", boundary.stem):
        raise ValueError("invalid workspace proposal boundary")
    return int(boundary.stem)


def _open_candidate(run: Path, boundary: Path, lease: ObjectLease, namespace: str):
    if (boundary.parent != run / "worker" / "candidates"
            or not checkpoint._owned_private(boundary.parent)
            or not checkpoint._owned_private(boundary.parent.parent)):
        raise ValueError("invalid workspace candidate directory")
    value = decode(read_mirror(boundary))
    candidate, edges = _candidate(value)
    if value["namespace"] != namespace or value["proposals"] != _number(boundary):
        raise ValueError("workspace candidate does not match its boundary")
    read_committed(lease, _key(run, "proposal", boundary.stem), value, edges)
    return candidate, edges[-1]


def read_candidate(run: Path, boundary: Path, *, parent: Path) -> Candidate:
    from kernel_workspace_refs import open_run_store

    store = open_run_store(run)
    with store.lease() as lease:
        candidate, _projection = _open_candidate(run, boundary, lease, _namespace(store))
    return candidate


def relative(name: object) -> tuple[bytes, ...]:
    from kernel_workspace_tree import validate_name

    if not isinstance(name, str):
        raise ValueError("invalid workspace relative path")  # noqa: TRY004
    result = tuple(os.fsencode(part) for part in name.split("/"))
    if len(result) > 256:
        raise ValueError("workspace relative path is too deep")
    for part in result:
        validate_name(part)
    return result


def _file_ref(lease: ObjectLease, node: Ref | None) -> Ref | None:
    from kernel_workspace_tree_io import child_refs

    if node is not None:
        return next((ref for ref in child_refs(lease, node) if ref.kind == "blob"), None)
    return None


def _projection(lease: ObjectLease, ref: Ref) -> dict:
    value, edges = read_envelope(lease, ref, PROJECTION_SCHEMA)
    if (not isinstance(value, dict) or set(value) != {"variant", "target", "declared"}
            or not isinstance(value["target"], str) or not isinstance(value["declared"], list)
            or value["variant"] not in ("compact", "full")):
        raise ValueError("invalid workspace candidate projection")
    for name in value["declared"]:
        relative(name)
    if value["declared"] != sorted(set(value["declared"])):
        raise ValueError("invalid workspace candidate declaration set")
    if value["target"]:
        relative(value["target"])
    if value["variant"] == "full":
        if len(edges) != 1 or edges[0].kind != "tree":
            raise ValueError("full candidate projection lacks its empty comparison tree")
    elif edges:
        raise ValueError("compact candidate projection has unexpected edges")
    return value


def _export(lease: ObjectLease, candidate: Candidate, projection: Ref,
            *, max_bytes: int | None = None) -> dict[str, bytes | None]:
    from kernel_workspace_tree_io import iter_changes, lookup_ref

    view = _projection(lease, projection)
    full = view["variant"] == "full"
    before = candidate.parent_tree
    if full:
        before = read_envelope(lease, projection, PROJECTION_SCHEMA)[1][0].oid
    selection: dict[str, Ref | None] = {}
    for change in iter_changes(lease, before, candidate.new_tree):
        current = _file_ref(lease, change.after)
        if current is not None and (full or current != _file_ref(lease, change.before)):
            selection["/".join(os.fsdecode(part) for part in change.path)] = current
    if not full:
        for name in view["declared"]:
            path = relative(name)
            present = lookup_ref(lease, candidate.new_tree, path)
            current = _file_ref(lease, present)
            if current is not None:
                selection[name] = current
            elif present is None and _file_ref(lease, lookup_ref(lease, before, path)) is not None:
                selection[name] = None
    result: dict[str, bytes | None] = {}
    size = 0
    for name, ref in sorted(selection.items()):
        value = read_object(lease, ref, limit=max_bytes or (256 << 20)) if ref is not None else None
        if max_bytes is not None:
            size += 4 * ((len(value or b"") + 2) // 3) + len(os.fsencode(name))
            if size > max_bytes:
                raise ValueError("workspace candidate exceeds salvage byte limit")
        result[name] = value
    return result


def export_candidate(run: Path, boundary: Path, *, parent: Path) -> dict[str, bytes | None]:
    from kernel_workspace_refs import open_run_store

    store = open_run_store(run)
    with store.lease() as lease:
        candidate, projection = _open_candidate(run, boundary, lease, _namespace(store))
        return _export(lease, candidate, projection)


def salvage_candidate(run: Path, boundary: Path, destination: Path) -> dict[str, object] | None:
    from kernel_workspace_refs import open_run_store
    from kernel_workspace_tree_io import lookup_ref

    store = open_run_store(run)
    with store.lease() as lease:
        candidate, projection = _open_candidate(run, boundary, lease, _namespace(store))
        view = _projection(lease, projection)
        target = view["target"]
        if not target:
            return None
        files = _export(lease, candidate, projection, max_bytes=OBJECT_LIMIT)
        old = _file_ref(lease, lookup_ref(lease, candidate.parent_tree, relative(target)))
        before = read_object(lease, old) if old else b""
        after = files.get(target) or b""
        patch = "".join(difflib.unified_diff(
            before.decode("utf-8", "replace").splitlines(keepends=True),
            after.decode("utf-8", "replace").splitlines(keepends=True),
            fromfile=target + ".parent", tofile=target,
        ))
        if not patch.strip():
            return None
        # The salvage mirror keeps the original typed tree roots alive after unit cleanup.
        call_root = destination.parent.parent
        value = decode(read_mirror(boundary))
        _, edges = _candidate(value)
        commit_record(lease, _key(run, "parallel-salvage", destination.stem), value, edges)
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw = canonical(value)
        write_mirror(destination, raw)
    return {"status": "PRESERVED", "path": str(destination.relative_to(call_root)),
            "source": boundary.name, "sha256": hashlib.sha256(raw).hexdigest(),
            "files": sorted(name for name, value in files.items() if value is not None)}


def _filter_proposal_tree(tree: Directory, excluded: frozenset[bytes]) -> Directory:
    from kernel_workspace_tree import Directory, children, replace

    for name, entry in children(tree):
        if name in excluded:
            tree = replace(tree, name, None)
        elif isinstance(entry, Directory):
            filtered = _filter_proposal_tree(entry, excluded)
            if filtered is not entry:
                tree = replace(tree, name, filtered)
    return tree


class _ProposalLinks:
    def __init__(self) -> None:
        self._tree: Directory | None = None
        self._paths: set[tuple[bytes, ...]] = set()

    def verify(self, root: Path, tree: Directory, excluded: frozenset[bytes], home: Path) -> None:
        from kernel_workspace_state import WorkspaceUnavailable
        from kernel_workspace_tree import Symlink, diff, directory

        # Resolving an unchanged symlink can still reach a changed target,
        # including a target outside the filtered observation.
        previous = self._tree if self._tree is not None else directory(tree.mode, ())
        for change in diff(previous, tree):
            if isinstance(change.after, Symlink):
                self._paths.add(change.path)
            elif isinstance(change.before, Symlink):
                self._paths.discard(change.path)
        self._tree = tree
        for path in self._paths:
            if excluded.isdisjoint(path):
                resolved = Path(os.path.realpath(root.joinpath(*map(os.fsdecode, path))))
                if resolved.is_relative_to(home):
                    raise WorkspaceUnavailable("workspace snapshot violates retention policy")


class ProposalCapture:
    """Hold the frozen parent and the admitted projection for an attempt."""

    def __init__(self, run: Path, parent: Path, *, compact: bool):
        self.run = run
        self.parent = parent
        self.compact = compact
        self._projection_key = None
        self._projection_ref = None
        self._parent_state = None
        self._candidate_state = None
        self._private = None
        self._retained: tuple[Path, Candidate, Ref] | None = None
        self._parent_scope: tuple[Ref, frozenset[bytes]] | None = None
        self._parent_links = _ProposalLinks()
        self._live_links = _ProposalLinks()

    def _frozen_state(self, store):
        from kernel_workspace_consumers import consumers_for
        from kernel_workspace_state import WorkspaceFence, WorkspaceState

        if self._parent_state is None:
            consumers = consumers_for(self.parent)
            if consumers is not None:
                self._parent_state = consumers.state
            else:
                self._private = tempfile.TemporaryDirectory(prefix="proposal-parent-", dir=store._path)
                private = Path(self._private.name)
                self._parent_state = WorkspaceState(WorkspaceFence.create(self.parent, private / "fence"))
        return self._parent_state

    def _verify_retained(self, store, lease: ObjectLease) -> Candidate | None:
        from kernel_workspace_refs import Ref
        from kernel_workspace_tree_io import read_entry

        if self._retained is None:
            return None
        path, candidate, projection = self._retained
        if _open_candidate(self.run, path, lease, _namespace(store)) != (candidate, projection):
            raise ValueError("retained workspace proposal changed")
        for oid in {candidate.parent_tree, candidate.new_tree}:
            read_entry(lease, Ref("tree", oid))
        read_object(lease, projection)
        return candidate

    def __call__(self, destination: str, root: Path, row: dict, proposals: int) -> None:
        if not destination:
            return
        from contextlib import ExitStack

        from kernel_workspace_consumers import consumers_for
        from kernel_workspace_refs import Ref, open_run_store, read_root
        from kernel_workspace_state import WorkspaceState, WorkspaceUnavailable, _installation_home, _secret_policy
        from kernel_workspace_tree import directory
        from kernel_workspace_tree_io import write_tree

        folder = Path(destination).resolve()
        if folder != self.run / "worker" / "candidates" or folder.is_relative_to(root.resolve()):
            raise ValueError("workspace candidate directory is outside its run boundary")
        checkpoint._private_dir(folder.parent)
        checkpoint._private_dir(folder)
        store = open_run_store(self.run)
        consumers = consumers_for(root)
        if consumers is None:
            raise WorkspaceUnavailable("candidate has no fenced workspace binding")
        if self.compact:
            from stateframe._child import CAPTURE_SKIP_PARTS

            skips = CAPTURE_SKIP_PARTS
            target = str(row.get("path") or "")
            declared = tuple(sorted(row.get("files") or {}))
        else:
            skips = {"__pycache__", ".pytest_cache", ".test-tmp"}
            if row.get("files") is not None:
                from build_tasks.multifile_row import normalize_repo_path

                target = normalize_repo_path(row["path"], field="candidate multi-file target")
                if target not in row["files"]:
                    raise ValueError("candidate target must be a declared multi-file source")
            else:
                target = "app/" + row["rel"]
            declared = ()
        policy = _secret_policy()
        home_setting = _installation_home()
        projection_key = target, declared, policy
        excluded = frozenset(os.fsencode(part) for part in skips)

        def select(path, _is_directory):
            return not any(part in excluded for part in path)

        def coverage():
            return ("candidate", excluded), select

        admitted = consumers.admitted_proposal_parent(self.parent) if self.compact else None
        live_state = consumers.proposal_state(excluded) if self.compact else consumers.state
        if live_state is consumers.state:
            # Checkpoints pin full coverage. Keep filtered proposals on their own index.
            if self._candidate_state is None or self._candidate_state.fence is not live_state.fence:
                self._candidate_state = WorkspaceState(live_state.fence)
            live_state = self._candidate_state
        with ExitStack() as stack:
            if admitted is not None:
                parent_state, parent_key, parent_ref = admitted
                parent = stack.enter_context(parent_state.pin("candidate-parent"))
            else:
                parent = stack.enter_context(self._frozen_state(store).pin("candidate-parent", coverage=coverage))
            live = stack.enter_context(live_state.pin("candidate", coverage=coverage))
            lease = stack.enter_context(store.lease())
            if self.compact:
                home = Path(os.path.realpath(home_setting))
                self._parent_links.verify(self.parent, parent.tree, excluded, home)
                self._live_links.verify(live_state.fence.root, live.tree, excluded, home)
            retained = self._verify_retained(store, lease)
            parent_scope = None
            if admitted is not None:
                if (read_root(lease, parent_key) != parent_ref
                        or parent.tree.digest.hex() != parent_ref.oid):
                    raise WorkspaceUnavailable("admitted proposal parent is unavailable")
                before = parent.persist(lease)
                parent_scope = parent_ref, excluded
                if retained is not None and self._parent_scope == parent_scope:
                    before = retained.parent_tree
                else:
                    filtered = _filter_proposal_tree(parent.tree, excluded)
                    if filtered is not parent.tree:
                        before = write_tree(lease, filtered, previous=before)
            else:
                before = parent.persist(lease)
            after = live.persist(lease)
            if self._projection_key != projection_key:
                if target:
                    relative(target)
                for name in declared:
                    relative(name)
                edges = () if self.compact else (Ref("tree", write_tree(lease, directory(0o755, ()))),)
                variant = "compact" if self.compact else "full"
                projection = publish_envelope(
                    lease, "meta", PROJECTION_SCHEMA,
                    {"target": target, "declared": list(declared), "variant": variant}, edges)
            else:
                projection = self._projection_ref
            if _secret_policy() != policy or _installation_home() != home_setting:
                raise WorkspaceUnavailable("proposal retention policy changed during capture")
            path = folder / f"{proposals:04d}.json"
            candidate = Candidate(before, after)
            save_candidate(lease, _key(self.run, "proposal", path.stem), candidate,
                           projection=projection, destination=path)
            self._projection_key, self._projection_ref = projection_key, projection
            if self.compact:
                self._retained = path, candidate, projection
                self._parent_scope = parent_scope


SEAL_SCHEMA = "stateframe-candidate-seal/1"


def _source_tree(lease: ObjectLease, sources: dict):
    from kernel_workspace_tree import Directory, File, directory
    from kernel_workspace_tree_io import write_tree

    paths: dict = {}
    contents = {}
    for name, content in sources.items():
        if content is None:
            continue
        path = relative(name)
        body = content.encode("utf-8") if isinstance(content, str) else content
        owner = paths
        for part in path[:-1]:
            owner = owner.setdefault(part, {})
            if not isinstance(owner, dict):
                raise ValueError("candidate file overlaps a directory")  # noqa: TRY004
        if path[-1] in owner:
            raise ValueError("candidate file overlaps another entry")
        checksum = hashlib.sha256(body).digest()
        contents.setdefault(checksum, body)
        owner[path[-1]] = File(0o644, len(body), checksum)
    from kernel_workspace_objects import ObjectRef

    entries = []
    for checksum, body in contents.items():
        checkpoint._check_registered_secrets(body)
        if len(body) > OBJECT_LIMIT:
            stored = lease.publish_stream(io.BytesIO(body), size=len(body), max_bytes=1 << 30)
            if stored.digest != checksum.hex():
                raise ValueError("candidate source changed during publication")
        else:
            entries.append((ObjectRef(checksum.hex(), len(body)), lambda body=body: io.BytesIO(body)))
    lease.publish_many(entries)

    def freeze(mapping) -> Directory:
        return directory(0o755, ((key, freeze(value) if isinstance(value, dict) else value)
                                 for key, value in mapping.items()))

    return write_tree(lease, freeze(paths))


def save_seal(destination: Path, artifact_digest: str, sources: dict,
              parent_sources: dict) -> None:
    from kernel_workspace_refs import open_run_store

    run = destination.parents[4]
    if destination.parents[2] != run / "worker" / "results":
        raise ValueError("workspace seal is outside its run boundary")
    oid = digest(artifact_digest.removeprefix("sha256:"))
    if destination.name != f"{oid}.json":
        raise ValueError("workspace seal artifact name is invalid")
    store = open_run_store(run)
    with store.lease() as lease:
        from kernel_workspace_refs import Ref

        before = _source_tree(lease, {name: parent_sources[name] for name in sources if name in parent_sources})
        after = _source_tree(lease, sources)
        selection = [[name, "delete" if source is None else "binary" if isinstance(source, bytes) else "text"]
                     for name, source in sorted(sources.items())]
        projection = publish_envelope(
            lease, "meta", SEAL_SCHEMA, {"artifact_digest": artifact_digest, "selection": selection})
        value = {"schema": SCHEMA, "proposals": 0, "namespace": _namespace(store),
                 "parent_tree": before, "new_tree": after, "projection": ref_json(projection)}
        edges = (Ref("tree", before), Ref("tree", after), projection)
        if destination.exists() and read_mirror(destination) != canonical(value):
            raise ValueError("workspace seal artifact conflicts with the existing record")
        commit_record(lease, _key(run, "seal", oid), value, edges)
        write_mirror(destination, canonical(value))


def _seal_projection(lease: ObjectLease, ref: Ref) -> dict:
    value, edges = read_envelope(lease, ref, SEAL_SCHEMA)
    if (edges or not isinstance(value, dict) or set(value) != {"artifact_digest", "selection"}
            or not isinstance(value["artifact_digest"], str)
            or not value["artifact_digest"].startswith("sha256:")
            or not isinstance(value["selection"], list) or not value["selection"]):
        raise ValueError("invalid workspace seal projection")
    digest(value["artifact_digest"].removeprefix("sha256:"))
    names = []
    for item in value["selection"]:
        if not isinstance(item, list) or len(item) != 2 or item[1] not in ("binary", "text", "delete"):
            raise ValueError("invalid workspace seal entry")
        relative(item[0])
        names.append(item[0])
    if names != sorted(set(names)):
        raise ValueError("invalid workspace seal path order")
    return value


def export_seal(run: Path, path: Path, expected: str) -> dict[str, bytes | None]:
    from kernel_done_is_submit_authority import artifact_digest
    from kernel_workspace_refs import open_run_store
    from kernel_workspace_tree_io import lookup_ref

    store = open_run_store(run)
    with store.lease() as lease:
        value = decode(read_mirror(path))
        candidate, edges = _candidate(value)
        if value["namespace"] != _namespace(store) or value["proposals"] != 0:
            raise ValueError("workspace seal does not match its run")
        read_committed(lease, _key(run, "seal", path.stem), value, edges)
        projection = _seal_projection(lease, edges[-1])
        if projection["artifact_digest"] != expected:
            raise ValueError("workspace seal projection does not match the submission")
        selected: dict[str, str | bytes | None] = {}
        files: dict[str, bytes | None] = {}
        for name, kind in projection["selection"]:
            entry = lookup_ref(lease, candidate.new_tree, relative(name))
            if kind == "delete":
                if entry is not None:
                    raise ValueError("workspace seal deletion has a tree entry")
                selected[name] = files[name] = None
                continue
            blob = _file_ref(lease, entry)
            if blob is None:
                raise ValueError("workspace seal write has no file entry")
            raw = read_object(lease, blob, limit=256 << 20)
            files[name] = raw
            selected[name] = raw.decode("utf-8") if kind == "text" else raw
        if artifact_digest(selected) != expected:
            raise ValueError("workspace seal content does not match the submission digest")
        return files
