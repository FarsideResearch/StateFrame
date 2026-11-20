"""Composition root for the frozen, run-local turn interfaces."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
import json
import os
from importlib import import_module
from pathlib import Path

from kernel_turn_contracts import (
    Basis, Context, Live, Ports, Ref, Snapshot, Stale, Unavailable, Workspace,
)


class WorkspaceLive:
    """Combine the workspace observation with a host authority fence.

    The coordinator supplies an authority observer that serializes run-owned
    policy, route and journal changes until the submission callback returns.
    """

    def __init__(self, authority: Callable[[Snapshot], AbstractContextManager[Basis]]) -> None:
        self._authority = authority
        self._workspace: Workspace | None = None

    @contextmanager
    def bound(self, workspace: Workspace) -> Iterator[None]:
        if self._workspace is not None:
            raise Unavailable("live authority is already bound")
        self._workspace = workspace
        try:
            yield
        finally:
            self._workspace = None

    @contextmanager
    def observing(self, scope: Ref) -> Iterator[Basis]:
        workspace = self._workspace
        if workspace is None:
            raise Unavailable("live authority has no workspace")
        with workspace.pin(scope) as snapshot, self._authority(snapshot) as basis:
            if (basis.live, basis.epoch) != (snapshot.root, snapshot.epoch):
                raise Stale("host authority belongs to another workspace observation")
            yield basis


def _factory(name: str):
    module = "kernel_turn_" + name
    try:
        return import_module(module).open_port
    except ModuleNotFoundError as error:
        if error.name != module:
            raise
        raise Unavailable(f"required turn adapter is unavailable: {name}") from error



class DecisionPanels:
    """Run-local panel roots derived from a verified decision input observation."""

    def __init__(self, directory: Path, *, task: str, parent: str):
        from kernel_turn_panel_store import canonical, sha256
        from kernel_turn_panels import ProjectionFacts
        from kernel_turn_publication import GENESIS

        # Panel events belong to this projection. They do not advance the
        # composed run's authority or continuation-journal cursor.
        attempt = os.urandom(16).hex()
        ctx = Context(str(os.geteuid()), attempt, attempt, directory,
                      sha256(canonical([task, parent, "decision-panels"])))
        self.publication = _factory("publication")(ctx)
        self.facts = ProjectionFacts(self.publication, ("frame", "settled"))
        self.panels = _factory("panels")(ctx, facts=self.facts, publication=self.publication)
        self.parent = parent
        self.tip = GENESIS
        self.manifest = None
        self.head = None
        self.layouts = {}

    def render(self, slot: str, revision: str, *, live: str, epoch: int, step: int,
               authority: str, verify: Callable[[], bool],
               project: Callable[[], object]) -> object:
        from kernel_turn_contracts import Commit, Event, Head
        from kernel_turn_panel_store import LAYOUT, canonical, sha256
        from kernel_turn_panels import slot_key
        from kernel_turn_publication import record_event

        facts, publication = self.facts, self.publication
        previous = facts.revisions[slot]
        changed = previous != revision
        tip, head, manifest = self.tip, self.head, self.manifest
        events = ()
        try:
            with publication.transaction():
                if publication.head() != head or not verify():
                    raise Stale("decision panel inputs changed before projection")
                if changed:
                    payload = canonical([slot, revision, live, epoch, authority, step])
                    ref = publication.stage("turn/event/1", payload)
                    next_head = Head(tip.seq + 1, sha256(canonical([tip, ref])))
                    event = Event(tip, next_head, "decision_panel", (slot_key(slot, "live"),), ref)
                    record_event(publication, event)
                    tip, events = next_head, (event,)
                at = Basis(self.parent, live, epoch, tip, step, authority)
                facts.basis, facts.selected, facts.project = at, slot, project
                facts.revisions[slot] = revision
                # The only new event is the transaction's own verified next row.
                # Its predecessor is the last successfully committed local root.
                if changed:
                    manifest = self.panels.fold(manifest, at, iter(events))
                layout = self.layouts.get(slot)
                if layout is None:
                    layout = publication.stage(LAYOUT, canonical([[slot, "live"]]))
                result, = json.loads(self.panels.render(manifest, layout))
                if not verify():
                    raise Stale("decision panel inputs changed during projection")
                if changed:
                    head = publication.commit(Commit(at, head, tip.digest,
                                                     (("decision", manifest.value),)))
        except BaseException:
            facts.revisions[slot] = previous
            raise
        finally:
            facts.basis = facts.selected = facts.project = None
        self.tip, self.head, self.manifest = tip, head, manifest
        self.layouts[slot] = layout
        return result

@contextmanager
def open_run(ctx: Context, *, live: Live) -> Iterator[Ports]:
    """Compose admitted services in dependency order for one run attempt."""
    with ExitStack() as stack:
        def open_adapter(name: str, **providers):
            port = _factory(name)(ctx, **providers)
            close = getattr(port, "close", None)
            if callable(close):
                stack.callback(close)
            return port

        publication = open_adapter("publication")
        workspace = open_adapter("workspace", publication=publication)
        if isinstance(live, WorkspaceLive):
            stack.enter_context(live.bound(workspace))
        meter = open_adapter("meter")
        facts = open_adapter("facts", workspace=workspace, publication=publication)
        panels = open_adapter("panels", facts=facts, publication=publication)
        proofs = open_adapter("proofs", workspace=workspace, facts=facts, publication=publication)
        launch = open_adapter("launch", workspace=workspace, facts=facts,
                              publication=publication)
        overlap = open_adapter("overlap", facts=facts, publication=publication)
        stack.callback(overlap.cancel)
        qualification = open_adapter(
            "qualification", live=live, panels=panels, proofs=proofs,
            meter=meter, publication=publication,
        )
        yield Ports(workspace, panels, proofs, launch, facts, qualification,
                    publication, overlap, meter)
