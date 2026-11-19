"""Build an ordered panel manifest from committed deltas and scoped facts.

The host stages a canonical layout (an array of [slot, variant] pairs). It
exposes the ordered variant catalog at CATALOG_KEY and one selected
[encoding, ordered_value(value)] row per slot_key via Facts. An empty slot
page represents absence. Each page certifies membership and its producer's
actual upstream reads, including any receipt rows or negative lookups.

Hold Publication.transaction from fold through render and head promotion.
If validation fails, discard the staged result and rebuild with previous=None
against a verified basis and its complete fact snapshot.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping

import kernel_turn_panel_store as store
import kernel_turn_reducers as reducers
from kernel_turn_contracts import (
    Artifact, Basis, Context, Corrupt, Dep, Event, Facts, Key, Page, Panels, Publication, Ref,
    Stale, Unavailable,
)
from kernel_turn_reducers import ordered_value as ordered_value


CATALOG_KEY = Key("query-membership", b'["panel-catalog"]')


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return (os.environ if environ is None else environ).get("STATEFRAME_TURN_PANELS", "off") == "on"


def slot_key(slot: str, variant: str) -> Key:
    return Key("query-membership", store.canonical(["panel", slot, variant]))


def _namespace(ctx: Context) -> str:
    return store.sha256(store.canonical([ctx.principal, ctx.run, ctx.profile_digest]))



class PanelCapacity(Unavailable):
    """A live recipe exceeds the port's fact-page bound."""


class ProjectionFacts:
    """Read host-owned decision recipes only when a panel slot needs them."""

    def __init__(self, publication: Publication, slots: tuple[str, ...]):
        self.publication = publication
        self.catalog = [[slot, "live"] for slot in slots]
        self.slots = {slot_key(slot, "live"): slot for slot in slots}
        self.revisions = dict.fromkeys(slots, "absent")
        self.project: Callable[[], object] | None = None
        self.selected: str | None = None
        self.basis: Basis | None = None

    def _revision(self, key: Key) -> str:
        if key == CATALOG_KEY:
            return "decision-layout"
        try:
            return self.revisions[self.slots[key]]
        except KeyError as error:
            raise Unavailable("unknown live panel fact") from error

    def lookup(self, at: Basis, key: Key, cursor: bytes | None, limit: int) -> Page:
        if at != self.basis or cursor is not None:
            raise Stale("live panel lookup used a stale basis")
        revision = self._revision(key)
        if key == CATALOG_KEY:
            rows = self.catalog
        elif revision == "absent":
            rows = []
        elif self.slots[key] == self.selected and self.project is not None:
            rows = [["json", ordered_value(self.project())]]
        else:
            raise Stale("live panel recipe is outside its projection boundary")
        payload = store.canonical(rows)
        if len(rows) > limit or len(payload) > store.MAX_FACT_BYTES:
            raise PanelCapacity("live decision recipe exceeds the fact-page bound")
        ref = self.publication.stage(store.FACT_PAGE, payload)
        return Page(Artifact(ref, (Dep(key, revision),)), None)

    def current(self, at: Basis, deps: tuple[Dep, ...]) -> bool:
        if at != self.basis:
            raise Stale("live panel validation used a stale basis")
        return all(self._revision(dep.key) == dep.revision for dep in deps)

    def build(self, parent, stop) -> None:
        raise Unavailable("live decision panels use point facts")

def restore_artifact(ctx: Context, *, publication: Publication, ref: Ref) -> Artifact:
    """Recover a panel root's dependencies from its committed manifest ref."""
    manifest = store.decode_manifest(publication, _namespace(ctx), ref)
    return Artifact(ref, store.artifact_deps(manifest))


class _PanelPort:
    def __init__(self, ctx: Context, *, facts: Facts, publication: Publication,
                 incremental: bool):
        self._facts = facts
        self._publication = publication
        self._incremental = incremental
        self._attempt = ctx.attempt
        self._namespace = _namespace(ctx)

    def _lookup(self, at: Basis, key: Key, limit: int) -> tuple[list[object], tuple[Dep, ...], str]:
        page = self._facts.lookup(at, key, None, limit)
        if page.next_cursor is not None:
            raise Unavailable("panel fact page is incomplete")
        deps = tuple(sorted(page.data.deps))
        if (not any(dep.key == key for dep in deps)
                or any(not dep.revision for dep in deps)
                or len({dep.key for dep in deps}) != len(deps)):
            raise Corrupt("panel fact page has invalid dependencies or missing membership")
        value = store.load_metadata(self._publication, page.data.value, store.FACT_PAGE,
                                    store.MAX_FACT_BYTES)
        if not isinstance(value, list) or len(value) > limit:
            raise Corrupt("invalid panel fact page")
        return value, deps, page.data.value.sha256

    def _update_slot(self, at: Basis, name: store.SlotName, previous: store.Slot | None) -> store.Slot:
        rows, deps, identity = self._lookup(at, slot_key(*name), 1)
        if previous is not None and previous.input_digest == identity:
            return store.Slot(previous.ref, deps, identity)
        if not rows:
            return store.Slot(None, deps, identity)
        row = rows[0]
        if (not isinstance(row, list) or len(row) != 2 or not isinstance(row[0], str)
                or row[0] not in store.FRAGMENTS):
            raise Corrupt("invalid panel variant recipe")
        encoding, value = row
        fragment = reducers.encode_fragment(encoding, value)
        ref = self._publication.stage(store.FRAGMENTS[encoding], fragment)
        return store.Slot(ref, deps, identity)

    def fold(self, previous: Artifact | None, at: Basis, events: Iterator[Event]) -> Artifact:
        old = (store.read_manifest(self._publication, self._namespace, previous)
               if previous is not None else None)
        delta = reducers.reduce_events(old.basis.journal if old else None, at.journal, events,
                                       old.last_event if old else None)
        reuse = old if self._incremental else None
        verify_all = delta.verify_dependencies
        verify_step = False
        if reuse is not None:
            prior = reuse.basis
            verify_all |= (reuse.attempt != self._attempt or
                           (prior.parent, prior.live, prior.epoch, prior.authority)
                           != (at.parent, at.live, at.epoch, at.authority))
            verify_step = prior.step != at.step

        def current(deps: tuple[Dep, ...]) -> bool:
            check = deps if verify_all else tuple(
                dep for dep in deps if dep.key in delta.changed
                or (verify_step and dep.key.family == "step"))
            return not check or self._facts.current(at, check)

        if reuse is not None and current(reuse.catalog_deps):
            order, catalog_deps = tuple(reuse.slots), reuse.catalog_deps
        else:
            catalog, catalog_deps, _ = self._lookup(at, CATALOG_KEY, store.MAX_SLOTS)
            order = store.slot_order(catalog, unique=True)
            if not order:
                raise Unavailable("panel catalog has no admitted variants")
        slots = {}
        for name in order:
            old_slot = reuse.slots.get(name) if reuse is not None else None
            slots[name] = (old_slot if old_slot is not None and current(old_slot.deps)
                           else self._update_slot(at, name, old_slot))
        manifest = store.Manifest(at, self._attempt, delta.last_event, catalog_deps, slots)
        deps = store.artifact_deps(manifest)
        if len({dep.key for dep in deps}) != len(deps):
            raise Stale("panel facts disagree at the frozen basis")
        if previous is not None and reuse is not None and manifest == reuse:
            return previous
        return store.write_manifest(self._publication, self._namespace, manifest)

    def render(self, manifest: Artifact, layout: Ref) -> bytes:
        stored = store.read_manifest(self._publication, self._namespace, manifest)
        selection = store.slot_order(store.load_metadata(
            self._publication, layout, store.LAYOUT, store.MAX_FACT_BYTES,
        ), unique=False)
        fragments = []
        for name in selection:
            slot = stored.slots.get(name)
            if slot is None or slot.ref is None:
                raise Stale("layout refers to an absent panel variant")
            fragments.append(self._publication.load(slot.ref))
        return b"[" + b", ".join(fragments) + b"]"


def open_port(ctx: Context, *, facts: Facts, publication: Publication) -> Panels:
    return _PanelPort(ctx, facts=facts, publication=publication, incremental=enabled())
