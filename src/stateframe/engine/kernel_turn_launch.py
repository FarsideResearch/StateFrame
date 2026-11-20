"""Admit and retain the parent used by the bootstrap request."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from hashlib import sha256
from pathlib import Path
from threading import Event, Lock, Thread

if __package__:
    from .kernel_turn_contracts import (
        Artifact, Context, Corrupt, Facts, Launch, Ports, Publication, Ref, Snapshot, Stale,
        Unavailable, Workspace,
    )
else:
    from kernel_turn_contracts import (
        Artifact, Context, Corrupt, Facts, Launch, Ports, Publication, Ref, Snapshot, Stale,
        Unavailable, Workspace,
    )

SWITCH = "STATEFRAME_TURN_MINIMAL_LAUNCH"
_DIGEST = re.compile(r"[0-9a-f]{64}")
_NAVIGATION = frozenset({"list_files", "read"})


def enabled(environment: Mapping[str, str]) -> bool:
    value = environment.get(SWITCH, "off").strip().lower()
    if value not in {"on", "off"}:
        raise ValueError(f"{SWITCH} accepts on or off")
    return value == "on"


def encode(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def allows_navigation(name: object) -> bool:
    return isinstance(name, str) and name in _NAVIGATION


def navigation_tools(tools: list[dict]) -> list[dict]:
    return [
        tool for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
        and allows_navigation(tool["function"].get("name"))
    ]


def _task_record(publication: Publication, ref: Ref) -> dict:
    if ref.codec != "turn/task/1":
        raise Corrupt("incompatible launch task record")
    data = publication.load(ref)
    if len(data) != ref.size or sha256(data).hexdigest() != ref.sha256:
        raise Corrupt("launch task record content changed")
    try:
        record = json.loads(data)
    except (ValueError, RecursionError) as error:
        raise Corrupt("invalid launch task record") from error
    if not isinstance(record, dict):
        raise Corrupt("invalid launch task record")
    return record


def _bootstrap_inputs(record: dict) -> tuple[list, list, dict, dict, list]:
    messages = record.get("messages")
    task = record.get("msg")
    scope = record.get("scope")
    preflight = record.get("preflight")
    head = record.get("head")
    step = record.get("step")
    authority = record.get("authority")
    if (not isinstance(task, str) or not task.strip()
            or not isinstance(messages, list) or not messages
            or not all(isinstance(message, dict) for message in messages)):
        raise Unavailable("launch requires the task and its prepared messages")
    if not isinstance(scope, dict) or not scope:
        raise Unavailable("launch requires the authorized scope")
    if not isinstance(preflight, dict) or not preflight:
        raise Unavailable("launch requires a completed preflight")
    if (not isinstance(head, list) or len(head) != 2 or type(head[0]) is not int or head[0] < 0
            or not isinstance(head[1], str) or not _DIGEST.fullmatch(head[1])
            or type(step) is not int or step < 0
            or not isinstance(authority, str) or not _DIGEST.fullmatch(authority)):
        raise Corrupt("invalid launch basis")
    offered = record.get("tools")
    if not isinstance(offered, list):
        raise Unavailable("launch requires a qualified navigation menu")
    tools = navigation_tools(offered)
    if not tools:
        raise Unavailable("launch requires an authorized navigation tool")
    return head, messages, scope, preflight, tools


class _Launch:
    def __init__(self, workspace: Workspace, facts: Facts, publication: Publication):
        self._workspace = workspace
        self._facts = facts
        self._publication = publication
        self._parent: Snapshot | None = None
        self._started = False
        self._build_lock = Lock()

    def bootstrap(self, source: Path, policy: Ref, task: Ref) -> tuple[Snapshot, Artifact]:
        if self._parent is not None:
            raise Stale("launch already has an admitted parent")
        if policy.codec != "turn/admission-policy/1":
            raise Corrupt("incompatible launch admission policy")
        record = _task_record(self._publication, task)
        head, messages, scope, preflight, tools = _bootstrap_inputs(record)
        parent = self._workspace.admit(source, policy)
        with self._workspace.retain(parent.root) as retained:
            if retained.root != parent.root:
                raise Stale("launch retained a different parent")
            slots = [
                [name, self._publication.stage("turn/slot-fragment/1", encode(data)), []]
                for name, data in (("messages", messages), ("scope", scope),
                                   ("preflight", preflight), ("tools", tools))
            ]
            manifest = {
                "basis": {
                    "parent": parent.root, "live": parent.root, "epoch": parent.epoch,
                    "journal": head, "step": record["step"], "authority": record["authority"],
                },
                "head": head,
                "slots": slots,
                "layout": [[name, "bootstrap"] for name, _, _ in slots],
                "producer": "turn/launch/1",
                "escape_layer": "json",
            }
            frame = self._publication.stage("turn/slot-manifest/1", encode(manifest))
        self._parent = parent
        return parent, Artifact(frame, ())

    @contextmanager
    def start_build(self, parent: Snapshot) -> Iterator[None]:
        if self._parent != parent:
            raise Stale("launch build requires this run's unused admitted parent")
        with self._workspace.retain(parent.root) as retained:
            if retained.root != parent.root:
                raise Stale("launch retained a different parent")
            with self._build_lock:
                if self._started:
                    raise Stale("launch build requires this run's unused admitted parent")
                self._started = True
            stop = Event()
            worker = Thread(target=self._build, args=(parent, stop), name="stateframe-launch-facts")
            try:
                worker.start()
            except RuntimeError as error:
                _report_build_failure(error)
                yield
                return
            try:
                yield
            finally:
                stop.set()
                worker.join()

    def _build(self, parent: Snapshot, stop: Event) -> None:
        try:
            self._facts.build(parent, stop.is_set)
        except Exception as error:  # Optional facts can always be obtained on demand.
            _report_build_failure(error)


def _report_build_failure(error: Exception) -> None:
    logging.getLogger(__name__).warning("Background fact build failed (%s)", type(error).__name__)


def open_port(ctx: Context, *, workspace: Workspace, facts: Facts, publication: Publication) -> Launch:
    """Bind the run's admission, facts and publication services."""
    return _Launch(workspace, facts, publication)


class HostLaunch:
    """Stream the prepared bootstrap into the child host's qualification path."""

    def __init__(self, ports: Ports, source: Path, spec: dict, lifetime: ExitStack):
        self.ports = ports
        self._source = source
        self._spec = spec
        self._lifetime = lifetime
        self._parent: Snapshot | None = None

    @property
    def admitted(self) -> bool:
        return self._parent is not None

    def bootstrap(self, policy: Ref, prepared: Mapping[str, object]) -> tuple[Snapshot, Artifact]:
        """Under a publication transaction, stage the prepared task and queue ingestion.

        The caller can render the frame in the same transaction. Keep the enclosing
        host context alive through materialization and release the transaction before
        any potentially blocking workspace reconciliation or worker teardown.
        """
        launch = self._spec["turn_launch"]
        row = self._spec["public_row"]
        scope = {key: row[key] for key in (
            "entry", "allowed_product_paths", "new_file_roots",
            "public_check_paths", "scoped_edits", "scope_source",
        ) if key in row}
        if row.get("scoped_edits"):
            scope["editable_paths"] = list(row["files"])
        preflight = prepared.get("preflight")
        if not isinstance(preflight, dict):
            raise Unavailable("launch requires current preflight and limits")
        task = self.ports.publication.stage("turn/task/1", encode({
            **prepared, "msg": row["msg"], "scope": scope,
            "preflight": {**preflight, "baseline": launch["baseline"]},
        }))
        parent, frame = self.ports.launch.bootstrap(self._source, policy, task)
        self._lifetime.enter_context(self.ports.launch.start_build(parent))
        self._parent = parent
        return parent, frame
