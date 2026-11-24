"""Host-owned projections of one fenced workspace observation."""
from __future__ import annotations

import fcntl
import os
import secrets
import stat
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from hashlib import sha256
from io import TextIOWrapper
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, cast
from weakref import WeakSet, WeakValueDictionary, finalize

from kernel_workspace_state import WorkspaceFence, WorkspaceState, WorkspaceUnavailable
from kernel_workspace_tree import Directory, File, Symlink, children, lookup

if TYPE_CHECKING:
    from kernel_ship_tree_smoke import SmokeProjection
    from kernel_workspace_objects import ObjectLease, WorkspaceObjects
    from kernel_workspace_feed import WorkspaceFeed
    from kernel_workspace_refs import Ref, RootKey
    from kernel_workspace_state import WorkspaceSnapshot
    from kernel_workspace_tree_io import CheckoutPolicy

T = TypeVar("T")


def _file_digests(tree: Directory, paths: Sequence[str]) -> dict[str, str | None] | None:
    from kernel_green_lock import is_product_path

    files: dict[str, str | None] = {}
    for path in paths:
        raw = str(path).replace("\\", "/").strip()
        while raw.startswith("./"):
            raw = raw[2:]
        parts = Path(raw).parts
        if not parts or raw.startswith("/") or ".." in parts or not is_product_path(raw):
            return None
        current = tree
        for index, part in enumerate(parts):
            entry = lookup(current, (os.fsencode(part),))
            if entry is None:
                files[raw] = None
                break
            if index == len(parts) - 1 and isinstance(entry, File):
                files[raw] = "sha256:" + entry.sha256.hex()
                break
            if not isinstance(entry, Directory):
                return None
            current = entry
        else:
            return None
    return files or None


def _has_links(tree: Directory) -> bool:
    pending = [tree]
    while pending:
        for _, entry in children(pending.pop()):
            if isinstance(entry, Symlink):
                return True
            if isinstance(entry, Directory):
                pending.append(entry)
    return False


class _RunWorkspaceState(WorkspaceState):
    """Keep separate coverage indexes on the same writer fence."""

    def __init__(self, fence: WorkspaceFence, views: WeakSet[_RunWorkspaceState],
                 label: str, *, feed: WorkspaceFeed | None = None):
        super().__init__(fence, feed=feed)
        self._views = views
        views.add(self)
        self._work_id = sha256(f"{self._work_id}:{label}".encode()).hexdigest()
        if feed is not None:
            finalize(self, feed.close)

    @contextmanager
    def begin_write(self, paths: Iterable[str | bytes] | None = None) -> Iterator[None]:
        from kernel_workspace_index import _footprint

        self._check_process()
        if isinstance(paths, (str, bytes)):
            paths = (paths,)
        changed = (frozenset(b"/".join(parts) for parts in _footprint(paths))
                   if paths is not None else None)
        with self.fence.writer() as (previous, epoch), ExitStack() as stack:
            pending = []
            for view in tuple(self._views):
                stack.enter_context(view._mutex)
                earlier = view._changes_since_pin(previous)
                view._pending = epoch, None
                if earlier is not None and changed is not None:
                    pending.append((view, earlier | changed))
            yield
            for view, footprint in pending:
                view._pending = epoch, footprint


class WorkspaceConsumers:
    """Share projections only across observations supplied by the host fence.

    The owner must hold this binding for its attempt and register it only for
    fully observed roots whose live writers participate in that fence.
    """

    def __init__(self, state: WorkspaceState, *, shell: ShellWorkspace | None = None,
                 checkout: tuple[WorkspaceObjects, CheckoutPolicy] | None = None,
                 admitted_parent: str | None = None):
        self.state = state
        self.shell = shell
        self.checkout = checkout
        self.admitted_parent = admitted_parent
        self._tree: bytes | None = None
        self._digests: OrderedDict[tuple[str, ...], dict[str, str | None] | None] = OrderedDict()
        self._smoke: OrderedDict[tuple[object, ...], object] = OrderedDict()
        self._links: bool | None = None
        self._ship_tree: SmokeProjection | None = None
        self._candidate_state: tuple[frozenset[bytes], WorkspaceState] | None = None
        self._admitted_parent: tuple[WorkspaceState, RootKey, Ref] | None = None

    def ship_tree(self) -> SmokeProjection:
        if self._ship_tree is None:
            from kernel_ship_tree_smoke import SmokeProjection

            self._ship_tree = SmokeProjection(self)
        return self._ship_tree

    def smoke_epoch(self) -> bytes:
        with self.state.fence._lock(fcntl.LOCK_SH) as (_, epoch):
            return epoch

    @contextmanager
    def smoke_read(self, *, background: bool = False) -> Iterator[WorkspaceSnapshot]:
        boundary = "ship-tree-proof"
        # The bootstrap reader must not hold the foreground index mutex. Both
        # readers still use the same host writer fence.
        state = WorkspaceState(self.state.fence) if background else self.state
        with state.pin(boundary) as observation:
            yield observation
            if not observation.validate_for_use(boundary):
                raise WorkspaceUnavailable("smoke workspace observation changed")
    def proposal_state(self, excluded: frozenset[bytes]) -> WorkspaceState:
        if not isinstance(self.state, _RunWorkspaceState):
            return self.state
        if self._candidate_state is None or self._candidate_state[0] != excluded:
            from kernel_workspace_feed import WorkspaceFeed

            feed = WorkspaceFeed(self.state.fence.root, enabled=True,
                                 include_directory=lambda path: excluded.isdisjoint(path.split(b"/")))
            scoped = _RunWorkspaceState(self.state.fence, self.state._views, "proposal", feed=feed)
            self._candidate_state = excluded, scoped
        return self._candidate_state[1]

    def admitted_proposal_parent(self, path: Path) -> tuple[WorkspaceState, RootKey, Ref] | None:
        from kernel_workspace_state import _installation_home, _secret_policy

        admitted = self._admitted_parent
        if admitted is not None:
            state, _, _ = admitted
            # A different policy must scan the filtered paths. Filtering can
            # also change whether all aliases of a hardlinked file are covered.
            if (path == state.fence.root and state.fence.root != self.state.fence.root
                    and state._home_setting == _installation_home()
                    and state._secrets == _secret_policy() and not state._leaders):
                return admitted
        return None

    def _observe(self, tree: Directory) -> None:
        if tree.digest != self._tree:
            self._tree = tree.digest
            self._digests.clear()
            self._smoke.clear()
            self._links = None

    def digests(self, paths: Sequence[str]) -> dict[str, str | None] | None:
        boundary = "green-lock"
        with self.state.pin(boundary) as observation:
            self._observe(observation.tree)
            key = tuple(paths)
            if key not in self._digests:
                self._digests[key] = _file_digests(observation.tree, key)
                if len(self._digests) > 8:
                    self._digests.popitem(last=False)
            self._digests.move_to_end(key)
            result = self._digests[key]
            if not observation.validate_for_use(boundary):
                raise WorkspaceUnavailable("product workspace observation changed")
            return dict(result) if result is not None else None

    def smoke_projection(self, boundary: str, key: tuple[object, ...],
                         resolve: Callable[[], tuple[T, bool]]) -> T:
        with self.state.pin(boundary) as observation:
            self._observe(observation.tree)
            if self._links is None:
                self._links = _has_links(observation.tree)
            cache_key = (boundary, *key)
            if cache_key in self._smoke:
                self._smoke.move_to_end(cache_key)
                result = cast(T, self._smoke[cache_key])
            else:
                result, internal_only = resolve()
                if internal_only and not self._links:
                    self._smoke[cache_key] = result
                    if len(self._smoke) > 12:
                        self._smoke.popitem(last=False)
            if not observation.validate_for_use(boundary):
                raise WorkspaceUnavailable("smoke workspace observation changed")
            return result

    def source_snapshot(self, paths: Sequence[str]) -> dict[str, str | None] | None:
        """Project startup text from the bytes already observed by the live owner."""
        from kernel_workspace_state import _spool_source

        boundary = "initial-product-sources"
        with self.state.pin(boundary) as observation:
            spool = self.state._spool
            if spool is None:
                return None
            if spool.data is not None:
                spool.data.flush()
            sources: dict[str, str | None] = {}
            text_by_digest: dict[bytes, str] = {}
            for path in paths:
                relative = Path(path)
                if relative.is_absolute() or ".." in relative.parts:
                    return None
                parts = tuple(map(os.fsencode, relative.parts))
                entry = observation.tree
                for part in parts:
                    if not isinstance(entry, Directory):
                        entry = None
                        break
                    entry = lookup(entry, (part,))
                    if isinstance(entry, Symlink):
                        # The legacy reader follows declared links.
                        return None
                if self.state._home is not None and (
                        self.state.fence.root / relative).is_relative_to(self.state._home):
                    return None
                if not isinstance(entry, File):
                    sources[path] = None
                    continue
                if entry.sha256 not in text_by_digest:
                    position = spool.positions.get(entry.sha256)
                    if position is None:
                        # Rejected or previously published bytes may not be in this spool.
                        return None
                    with TextIOWrapper(_spool_source(spool.data, *position),
                                       errors="ignore") as source:
                        text_by_digest[entry.sha256] = source.read()
                sources[path] = text_by_digest[entry.sha256]
            if not observation.validate_for_use(boundary):
                raise WorkspaceUnavailable("startup workspace observation changed")
            return sources


_bindings: WeakValueDictionary[Path, WorkspaceConsumers] = WeakValueDictionary()


def bind_consumers(root: Path, consumers: WorkspaceConsumers) -> None:
    root = Path(os.path.abspath(root))
    if root != consumers.state.fence.root:
        raise ValueError("consumer root differs from workspace fence")
    _bindings[root] = consumers


def consumers_for(root: Path) -> WorkspaceConsumers | None:
    return _bindings.get(Path(os.path.abspath(root)))


def checkout_snapshot(run: Path, key: RootKey, destination: Path, *,
                      policy: CheckoutPolicy) -> None:
    """Materialize a committed snapshot into an independent private checkout."""
    from kernel_workspace_refs import open_run_store, read_root
    from kernel_workspace_tree_io import checkout_tree

    with open_run_store(run).lease() as lease:
        ref = read_root(lease, key)
        if ref is None or ref.kind != "tree":
            raise WorkspaceUnavailable("committed workspace snapshot unavailable")
        checkout_tree(lease, ref.oid, destination, metadata=None, policy=policy)


def _effect_entry(entry):
    from hashlib import sha256
    from kernel_effects import Entry
    from kernel_workspace_tree import Special

    if isinstance(entry, File):
        if entry.hardlinked:
            return Entry("other", entry.mode, None, entry.size)
        return Entry("file", entry.mode, entry.sha256.hex(), entry.size)
    if isinstance(entry, Directory):
        return Entry("dir", entry.mode, None, 0)
    if isinstance(entry, Symlink):
        return Entry("symlink", entry.mode, sha256(entry.target).hexdigest(), len(entry.target))
    if isinstance(entry, Special):
        return Entry("other", entry.mode, None, entry.size)
    return None


def _relative(path: tuple[bytes, ...]) -> str:
    return "/".join(map(os.fsdecode, path))


def _included(path: tuple[bytes, ...], exclude: Callable[[str], bool]) -> bool:
    return all(not exclude(_relative(path[:index])) for index in range(1, len(path) + 1))


def _live_manifest(tree: Directory, exclude: Callable[[str], bool]):
    from kernel_effects import EffectError

    entries = {}
    pending = [((), iter(children(tree)))]
    while pending:
        parent, iterator = pending[-1]
        row = next(iterator, None)
        if row is None:
            pending.pop()
            continue
        name, entry = row
        path = (*parent, name)
        relative = _relative(path)
        if exclude(relative):
            continue
        if isinstance(entry, File):
            if entry.size < 0:
                raise EffectError("workspace file has an invalid size")
        if isinstance(entry, Directory):
            pending.append((path, iter(children(entry))))
        entries[relative] = _effect_entry(entry)
    return entries


def _live_changes(before: Directory, after: Directory, exclude: Callable[[str], bool]):
    from kernel_effects import classify_effect
    from kernel_workspace_tree import diff

    changes = []
    for row in diff(before, after):
        if row.path and _included(row.path, exclude):
            effect = classify_effect(_relative(row.path), _effect_entry(row.before),
                                     _effect_entry(row.after))
            if effect is not None:
                changes.append(effect)
    return sorted(changes, key=lambda effect: effect.path)


class EffectBlobs:
    """Bounded reads from the workspace store for already-authorized rollback."""

    def __init__(self, store):
        self.store = store

    def get(self, digest: str) -> bytes:
        from kernel_effects import EffectError, MAX_FILE_BYTES

        try:
            with self.store.lease() as lease:
                return lease.read(lease.resolve(digest, max_bytes=MAX_FILE_BYTES))
        except (OSError, ValueError) as error:
            raise EffectError("workspace effect object unavailable") from error

    def copy_to(self, digest: str, output) -> None:
        from kernel_effects import EffectError
        from kernel_workspace_objects import MAX_STREAM_BYTES

        try:
            with self.store.lease() as lease:
                lease.copy_verified(lease.resolve(digest, max_bytes=MAX_STREAM_BYTES), output,
                                    max_bytes=MAX_STREAM_BYTES)
        except (OSError, ValueError) as error:
            raise EffectError(f"workspace effect object unavailable: {error}") from error


def _stored_effect_entry(entry):
    from hashlib import sha256
    from kernel_effects import Entry

    if entry.kind == "file" and not entry.hardlinked:
        return Entry("file", entry.mode, entry.sha256.hex(), entry.size)
    if entry.kind == "dir":
        return Entry("dir", entry.mode, None, 0)
    if entry.kind == "symlink":
        return Entry("symlink", entry.mode, sha256(entry.target).hexdigest(), len(entry.target))
    return Entry("other", entry.mode, None, entry.size)


def effect_manifest(lease: ObjectLease, tree: str, *, exclude: Callable[[str], bool]):
    from kernel_effects import EffectError
    from kernel_workspace_tree_io import iter_entries

    result = {}
    for path, _, entry in iter_entries(lease, tree):
        if not path or not _included(path, exclude):
            continue
        if entry.kind == "file":
            if entry.size < 0:
                raise EffectError("workspace file has an invalid size")
        result[_relative(path)] = _stored_effect_entry(entry)
    return result


def effect_changes(lease: ObjectLease, before: str, after: str, *, exclude: Callable[[str], bool]):
    from kernel_effects import classify_effect
    from kernel_workspace_tree_io import iter_changes, read_entry

    changes = []
    for row in iter_changes(lease, before, after):
        if not row.path or not _included(row.path, exclude):
            continue
        old = _stored_effect_entry(read_entry(lease, row.before)) if row.before else None
        new = _stored_effect_entry(read_entry(lease, row.after)) if row.after else None
        effect = classify_effect(_relative(row.path), old, new)
        if effect is not None:
            changes.append(effect)
    yield from sorted(changes, key=lambda effect: effect.path)


def product_digests(lease: ObjectLease, tree: str,
                    paths: Sequence[str]) -> dict[str, str | None] | None:
    from kernel_green_lock import is_product_path
    from kernel_workspace_tree_io import lookup_ref, read_entry

    files: dict[str, str | None] = {}
    for path in paths:
        raw = str(path).replace("\\", "/").strip()
        while raw.startswith("./"):
            raw = raw[2:]
        parts = Path(raw).parts
        if not parts or raw.startswith("/") or ".." in parts or not is_product_path(raw):
            return None
        relative = tuple(map(os.fsencode, parts))
        for length in range(1, len(relative) + 1):
            ref = lookup_ref(lease, tree, relative[:length])
            if ref is None:
                files[raw] = None
                break
            entry = read_entry(lease, ref)
            if length == len(relative) and entry.kind == "file":
                files[raw] = "sha256:" + entry.sha256.hex()
                break
            if entry.kind != "dir":
                return None
        else:
            return None
    return files or None


class ShellWorkspace:
    """Tree-backed foreground transaction for an exclusively owned workcopy.

    The host supplies the authenticated parent pin and a coverage policy for
    effect accounting. G1's snapshot publication is the admission boundary for
    retained bytes, including environment and secret policy.
    """

    def __init__(self, state, store, *, parent_key, coverage=None, load_parent=None):
        self.state = state
        self.store = store
        self.parent_key = parent_key
        self._attempt = secrets.token_hex(16)
        self.coverage = coverage
        self.blobs = EffectBlobs(store)
        self._manifest_cache = OrderedDict()
        self.load_parent = load_parent
        self.parent_paths: tuple[str, ...] | None = None

    def baseline(self):
        from kernel_effects import EffectError
        from kernel_workspace_refs import read_root

        try:
            if self.load_parent is not None:
                manifest = self.load_parent(self)
                self.load_parent = None
                self.parent_paths = None
                return manifest
            with self.store.lease() as lease:
                ref = read_root(lease, self.parent_key)
                if ref is None or ref.kind != "tree":
                    raise EffectError("shell parent tree unavailable")
                return effect_manifest(lease, ref.oid, exclude=lambda _: False)
        except (OSError, ValueError) as error:
            raise EffectError("shell parent tree unavailable") from error

    def _manifest(self, tree: Directory):
        # The effect manifest covers entries beneath the root, whose own mode
        # can differ between the immutable parent and the private worker.
        key = tree._page.digest if tree._page is not None else None
        if key not in self._manifest_cache:
            self._manifest_cache[key] = _live_manifest(tree, lambda _: False)
            if len(self._manifest_cache) > 3:
                self._manifest_cache.popitem(last=False)
        self._manifest_cache.move_to_end(key)
        return self._manifest_cache[key]

    def _key(self, kind: str, call_id: str):
        from hashlib import sha256
        from kernel_workspace_refs import RootKey

        # Tape call IDs repeat on replay. Replay can have a different private
        # tree (for example a rebuilt cache) with the same legacy identity.
        identity = sha256(f"{self._attempt}\0{call_id}".encode()).hexdigest()
        return RootKey(self.parent_key.run_id, kind, identity)

    def _commit(self, boundary, snapshot, call_id, kind):
        from kernel_effects import EffectError
        from kernel_workspace_refs import Ref, commit_root

        key = self._key(kind, call_id)
        try:
            with self.store.lease() as lease:
                root = snapshot.persist(lease)
                if not snapshot.validate_for_use(boundary):
                    raise EffectError("shell workspace observation changed")
                commit_root(lease, key, Ref("tree", root), expected=None)
        except (OSError, ValueError) as error:
            raise EffectError("shell effect root unavailable") from error

    def before(self, call_id: str):
        boundary = "shell-pre"
        with self.state.pin(boundary, coverage=self.coverage) as snapshot:
            manifest = self._manifest(snapshot.tree)
            self._commit(boundary, snapshot, call_id, boundary)
            return snapshot.tree, manifest

    def scan(self):
        boundary = "shell-observe"
        with self.state.pin(boundary, coverage=self.coverage) as snapshot:
            if not snapshot.validate_for_use(boundary):
                from kernel_effects import EffectError
                raise EffectError("shell workspace observation changed")
            return self._manifest(snapshot.tree)

    def execution(self, capability):
        return self.state.begin_write(
            () if capability.sandboxed and not capability.workspace_writable else None,
        )

    def settle(self, before, manifest, policy, call_id):
        from kernel_effects import reconcile_effects

        with self.state.pin("shell-raw-post", coverage=self.coverage) as snapshot:
            after_tree = snapshot.tree
            after = self._manifest(after_tree)

        def final():
            with self.state.pin("shell-settled", coverage=self.coverage) as snapshot:
                return self._manifest(snapshot.tree)

        result = reconcile_effects(
            self.state.fence.root, manifest, after, policy, self.blobs, self.blobs,
            changes=_live_changes(before, after_tree, lambda _: False),
            observe_final=final, restore_scope=self.state.begin_write,
        )
        # Keep raw post-command entries ephemeral. Publish only after compensation
        # has been verified while holding a pin for the exact settled tree.
        boundary = "shell-post"
        with self.state.pin(boundary, coverage=self.coverage) as snapshot:
            if self._manifest(snapshot.tree) is not result.manifest:
                from kernel_effects import EffectError
                raise EffectError("settled shell workspace observation changed")
            self._commit(boundary, snapshot, call_id, boundary)
        return result


def read_snapshot_file(run: Path, key: RootKey, path: str) -> bytes:
    """Read one regular file through an authenticated workspace snapshot."""
    from kernel_workspace_refs import open_run_store, read_root
    from kernel_workspace_tree_io import lookup_ref, read_entry

    parts = Path(path).parts
    if not parts or Path(path).is_absolute() or ".." in parts:
        raise ValueError("invalid snapshot file path")
    with open_run_store(run).lease() as lease:
        root = read_root(lease, key)
        if root is None or root.kind != "tree":
            raise WorkspaceUnavailable("committed workspace snapshot unavailable")
        ref = lookup_ref(lease, root.oid, tuple(map(os.fsencode, parts)))
        node = read_entry(lease, ref) if ref is not None else None
        if node is None or node.kind != "file":
            raise WorkspaceUnavailable("snapshot file unavailable")
        blob = lease.resolve(node.sha256.hex(), max_bytes=node.size)
        if blob.size != node.size:
            raise WorkspaceUnavailable("snapshot file size changed")
        from io import BytesIO

        output = BytesIO()
        lease.copy_verified(blob, output, max_bytes=node.size)
        return output.getvalue()


def smoke_roots(lease: ObjectLease, tree: str, *, root: Path, paths: Sequence[str],
                pythonpath: str, resolver_id: str) -> tuple[str, ...]:
    """Resolve a bound live view only while it is pinned to the requested tree."""
    from kernel_ship_tree_smoke import _source_roots
    from kernel_workspace_refs import Ref
    from kernel_workspace_tree_io import read_entry

    root = Path(root).resolve()
    consumers = consumers_for(root)
    if consumers is None:
        raise WorkspaceUnavailable("smoke workspace binding unavailable")
    boundary = "ship-tree-pinned-roots"
    with consumers.state.pin(boundary) as snapshot:
        if snapshot.tree.digest.hex() != tree or not snapshot.validate_for_use(boundary):
            raise WorkspaceUnavailable("smoke workspace tree differs from the requested tree")
        # Verify the referenced root exists in the associated immutable store.
        if read_entry(lease, Ref("tree", tree)).kind != "dir":
            raise WorkspaceUnavailable("smoke workspace root is invalid")
        return consumers.smoke_projection(
            boundary, (tuple(paths), pythonpath, resolver_id),
            lambda: _source_roots(root, paths, pythonpath),
        )


def _filter_checkout(root: Path, tree: Directory, ignore, parent=()):
    from kernel_workspace_tree import replace

    original = tuple(children(tree))
    names = [os.fsdecode(name) for name, _ in original]
    omitted = set(ignore(str(root.joinpath(*map(os.fsdecode, parent))), names))
    removed = []
    for name, entry in original:
        path = (*parent, name)
        if os.fsdecode(name) in omitted:
            tree = replace(tree, name, None)
            removed.append(path)
        elif isinstance(entry, Directory):
            updated, descendants = _filter_checkout(root, entry, ignore, path)
            if updated is not entry:
                tree = replace(tree, name, updated)
            removed.extend(descendants)
    return tree, removed


def copy_pinned_workspace(source: Path, destination: Path, *, ignore=None) -> bool:
    """Materialize an authorized snapshot for an isolated tool needing a directory.

    The host supplies the store, a complete admission policy and checkout
    limits with the binding. The returned directory has its own writable inodes.
    """
    consumers = consumers_for(source)
    if consumers is None or consumers.checkout is None:
        return False
    from kernel_workspace_tree_io import checkout_tree, write_restore_meta, write_tree

    store, policy = consumers.checkout
    boundary = "consumer-checkout"
    with consumers.state.pin(boundary) as snapshot, store.lease() as lease:
        tree, metadata = snapshot.persist_with_metadata(lease)
        if ignore is not None:
            projected, dropped = _filter_checkout(Path(source), snapshot.tree, ignore)
            if dropped:
                tree = write_tree(lease, projected, previous=tree)
                metadata = write_restore_meta(lease, (), previous=metadata, removed=dropped)
        if not snapshot.validate_for_use(boundary):
            raise WorkspaceUnavailable("workspace checkout observation changed")
        checkout_tree(lease, tree, destination, metadata=metadata, policy=policy)
    return True


def _parent_inventory(parent: Path, excluded: tuple[bytes, ...] | None):
    """Record the private parent identity without opening any file payload."""
    from kernel_workspace_index import _check_stat, _open_directory, _open_root, _signature

    entries = {}

    def visit(fd, prefix):
        before = os.fstat(fd)
        with os.scandir(fd) as listing:
            for child in listing:
                name = os.fsencode(child.name)
                path = (*prefix, name)
                if path == excluded:
                    continue
                info = child.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    with _open_directory(fd, name) as directory:
                        _check_stat(info, os.fstat(directory))
                        visit(directory, path)
                else:
                    entries[path] = _signature(info)
        _check_stat(before, os.fstat(fd))
        entries[prefix] = _signature(before)

    with _open_root(tuple(map(os.fsencode, parent.parts[1:]))) as fd:
        visit(fd, ())
    return entries


def _deferred_parent(store, parent_key, parent: Path, private: Path):
    import tempfile

    from kernel_workspace_index import _signature
    from kernel_workspace_refs import Ref, commit_root, read_root
    from kernel_workspace_state import WorkspaceFence, _installation_home
    from kernel_workspace_tree_io import read_entry

    with store.lease() as lease:
        retained = read_root(lease, parent_key)
    if retained is not None:
        if retained.kind != "tree":
            raise WorkspaceUnavailable("shell parent tree unavailable")

        def load(shell):
            with store.lease() as lease:
                if read_root(lease, parent_key) != retained:
                    raise WorkspaceUnavailable("retained shell parent changed")
                key = read_entry(lease, retained)._page
                if key in shell._manifest_cache:
                    return shell._manifest_cache[key]
                return effect_manifest(lease, retained.oid, exclude=lambda _: False)

        return load, None

    home = _installation_home().resolve()
    if parent == home or parent.is_relative_to(home / "workspace-state"):
        raise WorkspaceUnavailable("workspace parent overlaps private storage")
    excluded = (tuple(map(os.fsencode, home.relative_to(parent).parts))
                if home.is_relative_to(parent) else None)
    inventory = _parent_inventory(parent, excluded)

    def load(shell):
        boundary = "shell-baseline"
        with tempfile.TemporaryDirectory(prefix="parent-", dir=private) as scratch:
            initial = WorkspaceState(WorkspaceFence.create(parent, Path(scratch) / "fence"))
            with initial.pin(boundary) as snapshot, store.lease() as lease:
                observed = {path: _signature(row.info) for path, row in initial._index.rows.items()}
                if initial._home != home or observed != inventory:
                    raise WorkspaceUnavailable("shell parent changed after binding")
                manifest = shell._manifest(snapshot.tree)
                ref = Ref("tree", snapshot.persist(lease))
                if not snapshot.validate_for_use(boundary):
                    raise WorkspaceUnavailable("workspace parent changed during capture")
                commit_root(lease, parent_key, ref, expected=None)
                return manifest

    return load, tuple(_relative(path) for path in inventory if path)


def bind_run_consumers(run: Path, root: Path, *, parent: Path,
                       excluded: Path | None = None,
                       defer_parent: bool = False) -> WorkspaceConsumers:
    """Bind this private child workspace to an authenticated host run."""
    import tempfile
    from kernel_workspace_feed import WorkspaceFeed
    from kernel_workspace_refs import Ref, RootKey, commit_root, open_run_store, read_root, workspace_run_id
    from kernel_workspace_tree_io import CheckoutPolicy

    root = Path(root).resolve(strict=True)
    run = Path(run).resolve(strict=True)
    if not root.is_relative_to(run) or root == run:
        raise WorkspaceUnavailable("workspace consumer root is outside the run")
    store = open_run_store(run)
    parent_key = RootKey(workspace_run_id(run), "shell-baseline", "parent")
    private = tempfile.TemporaryDirectory(prefix="consumer-fence-", dir=store._path)
    admitted_parent = None
    retained_parent = None
    load_parent = None
    parent_paths = None
    try:
        # A retained rollback parent stays authoritative after its live directory disappears.
        parent = Path(parent).resolve()
        if not parent.is_relative_to(run) or parent == run:
            raise WorkspaceUnavailable("workspace parent is outside the run")
        if defer_parent:
            load_parent, parent_paths = _deferred_parent(store, parent_key, parent, Path(private.name))
            with store.lease() as lease:
                retained_parent = read_root(lease, parent_key)
        else:
            with store.lease() as lease:
                retained_parent = read_root(lease, parent_key)
        if not defer_parent and retained_parent is None:
            initial = WorkspaceState(WorkspaceFence.create(parent, Path(private.name) / "parent"))
            boundary = "shell-baseline"
            with initial.pin(boundary) as snapshot, store.lease() as lease:
                ref = Ref("tree", snapshot.persist(lease))
                if not snapshot.validate_for_use(boundary):
                    raise WorkspaceUnavailable("workspace parent changed during capture")
                commit_root(lease, parent_key, ref, expected=None)
                retained_parent = ref
            admitted_parent = initial, parent_key, ref

        def select(parts, _is_directory):
            path = root.joinpath(*map(os.fsdecode, parts))
            return excluded is None or not path.is_relative_to(excluded)

        coverage = None if excluded is None or not excluded.is_relative_to(root) else (
            lambda: (("shell-effects", excluded), select)
        )
        fence = WorkspaceFence.create(root, Path(private.name) / "live")
        views: WeakSet[_RunWorkspaceState] = WeakSet()
        state = _RunWorkspaceState(fence, views, "live", feed=WorkspaceFeed(root, enabled=True))
        shell_state = state if coverage is None else _RunWorkspaceState(
            fence, views, "shell", feed=WorkspaceFeed(root, enabled=True,
                include_directory=lambda path: select(tuple(path.split(b"/")), True)))
        shell = ShellWorkspace(shell_state, store, parent_key=parent_key, coverage=coverage,
                               load_parent=load_parent)
        shell.parent_paths = parent_paths
        if parent == root and parent_paths is not None:
            # Without a separate host parent, retain the live start before edits.
            shell.baseline()
        policy = _default_checkout_policy()
        consumers = WorkspaceConsumers(state, shell=shell, checkout=(store, policy),
                                       admitted_parent=retained_parent.oid if retained_parent else None)
        consumers._admitted_parent = admitted_parent
        consumers.private_directory = private
        bind_consumers(root, consumers)
        return consumers
    except BaseException:
        private.cleanup()
        raise


def _default_checkout_policy() -> CheckoutPolicy:
    from kernel_workspace_tree_io import CheckoutPolicy

    return CheckoutPolicy(True, True, 0o7777, 256 << 20)


def rebind_after_checkpoint_activation(run: Path, root: Path) -> None:
    """Bind a fresh index after an object restore replaces the workcopy inode."""
    from kernel_shell_tool import service

    tool = service()
    if tool is None or tool.root != root:
        return
    if tool.workspace_consumers is None:
        raise WorkspaceUnavailable("checkpoint workspace has no host consumer binding")
    excluded = tool.environment.root if tool.environment is not None else None
    consumers = bind_run_consumers(run, root, parent=run / "parent", excluded=excluded)
    tool.workspace_consumers = consumers
    tool.workspace_effects = consumers.shell


def write_scope(path: Path):
    """Reserve the owner fence for a native edit whose destination is known."""
    from contextlib import nullcontext

    absolute = Path(os.path.abspath(path))
    for parent in absolute.parents:
        consumers = _bindings.get(parent)
        if consumers is not None:
            return consumers.state.begin_write((absolute.relative_to(parent).as_posix(),))
    return nullcontext()
