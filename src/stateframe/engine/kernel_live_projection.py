"""Causal live adapter for the tested StateFrame fold, with a public-only view.

This module performs no inference or grading. The caller supplies completed
public actions and trusted subprocess receipts, never a gold task row.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from pathlib import Path, PurePosixPath

from kernel_measurement import digest
from kernel_read_working_set import evictfix_enabled
from kernel_non_product_paths import NON_PRODUCT_PARTS, is_non_product  # noqa: F401 (re-export)
from probes.stateframe_shadow import Turn, _refresh_derived, fold_turn, initial_frame
from kernel_test_artifact_policy import discover_test_artifact_policy
from kernel_verdict import NO_HOST_CHECKS, SATISFIED, UNRESOLVED, Verdict, evaluate_required_tests


_READ_PAGE_IDENTITY = re.compile(
    r"^\[read page\] path=(.*?) revision=(sha256:[0-9a-f]{64}) "
    r"lines=(\d+)-(\d+)/(\d+) chars=(\d+) "
    r"truncated=(?:true|false) decoded_with_replacement=(?:true|false)$",
    re.IGNORECASE,
)


def _compact_live_read(item: Mapping[str, object]) -> dict[str, object]:
    """Project only identity from a host read receipt and a short useful note."""
    target = item.get("target")
    snapshot = item.get("snapshot")
    snapshot = snapshot if isinstance(snapshot, Mapping) else {}
    captured = snapshot.get("content")
    captured = captured if isinstance(captured, Mapping) else {}
    raw = captured.get("value")
    raw = raw if isinstance(raw, str) else ""
    row: dict[str, object] = {}
    if isinstance(target, str) and target:
        row["path"] = target
    header, separator, remainder = raw.partition("\n")
    match = _READ_PAGE_IDENTITY.fullmatch(header) if separator else None
    identity = None
    if match is not None:
        identity = (match.group(1), match.group(2).lower(),
                    *map(int, match.group(3, 4, 5, 6)), None, False)
    elif separator:
        # Match the production pager's complete header contract, rather than
        # assuming the older short test receipt is what the host emits.
        from kernel_request_source_window import _READ_PAGE_HEADER

        production = _READ_PAGE_HEADER.fullmatch(header.removesuffix("\r"))
        if production is not None:
            fields = production.groupdict()
            start, end = int(fields["start"]), int(fields["end"])
            if (start == int(fields["effective_start"])
                    and end == int(fields["effective_end"])
                    and int(fields["start_column"]) == 0):
                identity = (fields["path"], fields["revision"].lower(), start, end,
                            int(fields["total"]), int(fields["chars"]),
                            int(fields["bytes"]), True)
    if identity is not None and identity[0] == target:
        _path, revision, start, end, total, chars, byte_count, production_header = identity
        _notice, second_separator, content = remainder.partition("\n")
        page_content = content[:chars]
        complete = (captured.get("complete") is True
                    and len(page_content) == chars
                    and (production_header or len(content) == chars)
                    and (byte_count is None
                         or len(page_content.encode("utf-8")) == byte_count))
        clipped = (captured.get("complete") is False
                   and captured.get("status") == "observed_but_omitted"
                   and len(content) <= chars)
        if second_separator and 0 < start <= end <= total and (complete or clipped):
            row["ranges"] = [[start, end]]
            row["revision"] = revision
            raw = page_content
    summary = next((line.strip() for line in raw.splitlines() if line.strip()), "")
    row["summary"] = summary[:160] if summary else "Prior read recorded; revalidate source before editing."
    return row


def is_harness_path(path: Path) -> bool:
    """Shared lexical role used by workspace binding and public review inputs."""
    return ("tests" in path.parts or path.name.startswith("test_")
            or path.name in {'conftest.py', 'pytest.ini', 'pyproject.toml',
                             'setup.cfg', 'tox.ini', 'pytest.py'}
            or '_pytest' in path.parts)


def workspace_binding(
        root: Path, environment: str, *, command: object = (),
        run_write_set: tuple[str, ...] | list[str] = (),
        _entries: list[tuple[tuple[str, ...], str, bool]] | None = None) -> dict:
    """Bind all public workspace files, including siblings and test helpers.

    L112 host-time fix: identical to hashing every file on every call (the former
    ``sorted(root.rglob("*"))`` loop, kept as a reference in the unit tests), but
    (1) directories whose every descendant ``is_non_product`` are not walked, and
    (2) a file is rehashed only when its lstat signature changed (see
    :func:`_file_value`).  The result must always equal a full rehash.
    """
    policy = discover_test_artifact_policy(root, command=command)
    policy_key: object = policy  # frozen dataclass: memo is keyed by policy VALUE
    try:
        hash(policy)
    except TypeError:
        policy_key = None
    product, harness = {}, {}
    for parts, path, is_link in (_entries if _entries is not None
                                 else _workspace_entries(Path(root))):
        key = "/".join(parts)
        non_product, is_test = _path_roles(key, parts)
        if non_product:  # .git/.test-tmp + coverage/caches (audit #2)
            continue
        if _policy_matches(policy, policy_key, key):
            continue
        if is_link:
            # Link identity is included without reading an escaped target.
            value = "symlink:" + os.readlink(path)
        else:
            value = _file_value(path)
            if value is None:  # vanished or no longer a regular file mid-walk
                continue
        (harness if is_test else product)[key] = value
    return {"product_revision": digest(product), "test_harness_revision": digest(harness),
            "environment_revision": environment, "product_files": product,
            "harness_files": harness}


# Lexical role memos.  ``is_non_product`` and ``is_harness_path`` are pure functions
# of the relative path string, and ``TestArtifactPolicy.matches`` of (policy value,
# path string), so these never go stale; they only skip ~600k PurePosixPath builds.
_ROLE_MEMO: dict[str, tuple[bool, bool]] = {}
_POLICY_MEMO: dict[tuple[object, str], bool] = {}
_ROLE_MEMO_MAX = 200_000


def _path_roles(key: str, parts: tuple[str, ...]) -> tuple[bool, bool]:
    roles = _ROLE_MEMO.get(key)
    if roles is None:
        relative = PurePosixPath(*parts)
        roles = (is_non_product(relative), is_harness_path(relative))
        if len(_ROLE_MEMO) >= _ROLE_MEMO_MAX:
            _ROLE_MEMO.clear()
        _ROLE_MEMO[key] = roles
    return roles


def _policy_matches(policy: object, policy_key: object, key: str) -> bool:
    if policy_key is None:
        return policy.matches(key)
    memo_key = (policy_key, key)
    matched = _POLICY_MEMO.get(memo_key)
    if matched is None:
        matched = bool(policy.matches(key))
        if len(_POLICY_MEMO) >= _ROLE_MEMO_MAX:
            _POLICY_MEMO.clear()
        _POLICY_MEMO[memo_key] = matched
    return matched


def _prunable_directory(parts: tuple[str, ...]) -> bool:
    """True when EVERY descendant of this directory ``is_non_product``.

    ``is_non_product`` tests path components (any hidden/non-product part, a
    root-only first part, an adjacent pair) plus the final name.  All component
    tests on ``parts`` carry over to every descendant; the neutral child name
    ``"a"`` cannot trigger the name test or start a new pair, so a True here
    holds for the whole subtree.
    """
    return is_non_product(PurePosixPath(*parts, "a"))


def _workspace_entries(root: Path) -> list[tuple[tuple[str, ...], str, bool]]:
    """Regular files and symlinks under ``root`` in ``sorted(root.rglob("*"))`` order.

    Like ``Path.rglob`` (3.13+ default ``recurse_symlinks=False``) symlinked
    directories are reported as links and never descended.  Directories that
    are wholly non-product are pruned (their files are discarded anyway).
    """
    found: list[tuple[tuple[str, ...], str, bool]] = []
    # Absolute paths: the digest memo is keyed by path and must not alias across cwds.
    stack: list[tuple[str, tuple[str, ...]]] = [(str(root.absolute()), ())]
    while stack:
        directory, prefix = stack.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = list(iterator)
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            continue
        for entry in entries:
            parts = (*prefix, entry.name)
            try:
                if entry.is_symlink():
                    found.append((parts, entry.path, True))
                elif entry.is_dir(follow_symlinks=False):
                    if not _prunable_directory(parts):
                        stack.append((entry.path, parts))
                elif entry.is_file(follow_symlinks=False):
                    found.append((parts, entry.path, False))
            except OSError:
                continue
    found.sort(key=lambda item: item[0])  # PosixPath ordering compares parts
    return found


# Per-process digest memo: absolute path -> (lstat signature, "sha256:<hex>").
# Any change to inode, device, size, mtime_ns or ctime_ns (ctime moves on every
# write, even when a tool restores mtime) forces a rehash.  Entries younger than
# _RACY_NS are never memoized (git's racy-timestamp rule), and an entry is
# stored only when the file's signature is unchanged across the read.
_DIGEST_MEMO: dict[str, tuple[tuple[int, int, int, int, int], str]] = {}
_DIGEST_MEMO_MAX = 200_000
_RACY_NS = 2_000_000_000


def _stat_signature(result: os.stat_result) -> tuple[int, int, int, int, int]:
    return (result.st_ino, result.st_dev, result.st_size, result.st_mtime_ns, result.st_ctime_ns)


def _file_value(path: str) -> str | None:
    try:
        before = os.lstat(path)
    except OSError:
        return None
    if not stat.S_ISREG(before.st_mode):
        return None
    signature = _stat_signature(before)
    memo = _DIGEST_MEMO.get(path)
    if memo is not None and memo[0] == signature:
        return memo[1]
    with open(path, "rb") as stream:
        value = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
    try:
        after = _stat_signature(os.lstat(path))
    except OSError:
        after = None
    if (after == signature
            and time.time_ns() - max(before.st_mtime_ns, before.st_ctime_ns) > _RACY_NS):
        if len(_DIGEST_MEMO) >= _DIGEST_MEMO_MAX:
            _DIGEST_MEMO.clear()
        _DIGEST_MEMO[path] = (signature, value)
    else:
        _DIGEST_MEMO.pop(path, None)
    return value


def _freeze_files(value: dict[str, str]) -> MappingProxyType:
    """Freeze one validated projection so consumers cannot mutate its view."""
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class WorkspaceView:
    """Runtime-owned immutable binding for one mutation epoch."""

    root: str
    scope_identity: str
    mutation_epoch: int
    boundary: str
    product_revision: str
    test_harness_revision: str
    environment_revision: str
    policy_identity: str
    authorization_scope: str
    tenant_identity: str
    product_files: Mapping[str, str]
    harness_files: Mapping[str, str]

    @property
    def root_revision(self) -> str:
        return self.product_revision

    def as_binding(self) -> dict[str, object]:
        """Return the historical mutable-shaped binding at an API boundary."""
        return {
            "product_revision": self.product_revision,
            "test_harness_revision": self.test_harness_revision,
            "environment_revision": self.environment_revision,
            "product_files": dict(self.product_files),
            "harness_files": dict(self.harness_files),
        }

    def identity(self) -> dict[str, object]:
        """Return the complete host verification identity without mutable files."""
        return {
            "root": self.root,
            "scope_identity": self.scope_identity,
            "mutation_epoch": self.mutation_epoch,
            "product_revision": self.product_revision,
            "test_harness_revision": self.test_harness_revision,
            "environment_revision": self.environment_revision,
            "policy_identity": self.policy_identity,
            "authorization_scope": self.authorization_scope,
            "tenant_identity": self.tenant_identity,
        }


class WorkspaceViewStore:
    """Reconcile a workspace once after a host mutation or unsafe boundary."""

    _UNSAFE_BOUNDARIES = frozenset({
        "resume", "test", "subprocess", "external", "policy", "authorization",
        "environment", "tenant", "run",
    })

    def __init__(self, root: Path, environment: str, *, scope_identity: str = "",
                 policy_identity: str = "", authorization_scope: str = "",
                 tenant_identity: str = "") -> None:
        self.root = Path(root).resolve()
        self.environment = str(environment)
        self.scope_identity = str(scope_identity)
        self.policy_identity = str(policy_identity or "UNKNOWN_POLICY")
        self.authorization_scope = str(authorization_scope or self.scope_identity or "UNKNOWN_SCOPE")
        self.tenant_identity = str(tenant_identity or "UNKNOWN_TENANT")
        self.mutation_epoch = 0
        self._view: WorkspaceView | None = None
        self._dirty = True
        self._last_boundary: str | None = None
        self._last_boundary_epoch = -1
        self._last_context: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
        self._entry_signatures: tuple[tuple[object, ...], ...] = ()

    @staticmethod
    def _context(command: object, run_write_set: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
        def normalize(value: object) -> tuple[str, ...]:
            if isinstance(value, str):
                return (value,)
            if isinstance(value, (list, tuple, set, frozenset)):
                return tuple(sorted(str(item) for item in value))
            return (str(value),) if value else ()
        return normalize(command), normalize(run_write_set)

    def update_context(self, *, environment: str | None = None,
                       scope_identity: str | None = None,
                       policy_identity: str | None = None,
                       authorization_scope: str | None = None,
                       tenant_identity: str | None = None) -> None:
        changed = False
        if environment is not None and str(environment) != self.environment:
            self.environment = str(environment)
            changed = True
        if scope_identity is not None and str(scope_identity) != self.scope_identity:
            self.scope_identity = str(scope_identity)
            changed = True
        if policy_identity is not None and str(policy_identity) != self.policy_identity:
            self.policy_identity = str(policy_identity)
            changed = True
        if (authorization_scope is not None
                and str(authorization_scope) != self.authorization_scope):
            self.authorization_scope = str(authorization_scope)
            changed = True
        if tenant_identity is not None and str(tenant_identity) != self.tenant_identity:
            self.tenant_identity = str(tenant_identity)
            changed = True
        if changed:
            self.note_mutation((), "context_changed")

    def note_mutation(self, changed_paths: object = (), outcome: object = None) -> int:
        """Invalidate all derived views after a host-visible mutation."""
        has_paths = bool(changed_paths)
        if has_paths or outcome is not None:
            self.mutation_epoch += 1
            self._dirty = True
        return self.mutation_epoch

    @staticmethod
    def _entry_signature(entries: list[tuple[tuple[str, ...], str, bool]]) -> tuple[tuple[object, ...], ...]:
        values: list[tuple[object, ...]] = []
        for parts, path, is_link in entries:
            try:
                stat_result = os.lstat(path)
                stat_signature: object = _stat_signature(stat_result)
                link_target = os.readlink(path) if is_link else None
            except OSError:
                stat_signature, link_target = "MISSING", None
            values.append((parts, is_link, stat_signature, link_target))
        return tuple(values)

    def current_view(self, boundary: object = None, *, command: object = (),
                     run_write_set: object = ()) -> WorkspaceView:
        boundary_name = str(boundary or "ordinary")
        context = self._context(command, run_write_set)
        if context != self._last_context:
            self._dirty = True
        reconcile_entries: list[tuple[tuple[str, ...], str, bool]] | None = None
        if self._view is not None and boundary_name in self._UNSAFE_BOUNDARIES and (
                boundary_name != self._last_boundary or self._last_boundary_epoch != self.mutation_epoch):
            # Reconcile once at the host-controlled boundary. The same entries
            # feed binding construction so an external write does not cause a
            # second consumer-specific walk.
            reconcile_entries = _workspace_entries(self.root)
            if self._entry_signature(reconcile_entries) != self._entry_signatures:
                self.note_mutation((), {"boundary": boundary_name, "reason": "external_change"})
        if self._view is None or self._dirty:
            entries = reconcile_entries if reconcile_entries is not None else _workspace_entries(self.root)
            binding = workspace_binding(
                self.root, self.environment, command=context[0], run_write_set=context[1],
                _entries=entries)
            self._view = WorkspaceView(
                root=str(self.root), scope_identity=self.scope_identity,
                mutation_epoch=self.mutation_epoch, boundary=boundary_name,
                product_revision=str(binding["product_revision"]),
                test_harness_revision=str(binding["test_harness_revision"]),
                environment_revision=str(binding["environment_revision"]),
                policy_identity=self.policy_identity,
                authorization_scope=self.authorization_scope,
                tenant_identity=self.tenant_identity,
                product_files=_freeze_files(binding["product_files"]),
                harness_files=_freeze_files(binding["harness_files"]),
            )
            self._entry_signatures = self._entry_signature(entries)
            self._dirty = False
            self._last_context = context
        # An unchanged reconciliation still crosses the host boundary. Record
        # it so returning to a prior unsafe boundary cannot reuse stale state.
        self._last_boundary = boundary_name
        self._last_boundary_epoch = self.mutation_epoch
        return self._view


class LiveProjection:
    def __init__(self, *, task_id: str, symbol: str, path: str, brief: str,
                 selectors: list[str], binding: dict):
        public = {"fn": symbol, "rel": path, "msg": brief,
                  "messages": [{"role": "user", "content": brief}],
                  "required_selectors": list(selectors),
                  **{key: binding[key] for key in (
                      "product_revision", "test_harness_revision", "environment_revision")}}
        self.frame = initial_frame(public)
        self.frame["task_id"] = task_id
        self.frame["objective"]["public_requirements"] = brief
        self.last_step = -1
        self.seen_trace_events: set[str] = set()
        self._message_cursor = 0
        self._last_assistant_index: int | None = None
        self._trace_cursor = 0
        self._pending_trace_positions: list[int] = []
        self.model_claims: list[dict] = []
        self.initial_harness_revision = binding['test_harness_revision']
        self.initial_harness_files = dict(binding['harness_files'])
        # STATEFRAME_HYPOTHESES_NOT_FACTS: the product revision each step's request
        # was built against (first binding seen for that step, before its tool ran).
        self.initial_product_revision = binding["product_revision"]
        self.revision_by_step: dict[int, str] = {}

    def update(self, messages: list[dict], trace: list[dict], step: int,
               binding: dict, test_receipts: list[dict]) -> dict:
        """Fold the previous complete exchange, including refusal/continue paths."""
        from kernel_hypothesis_projection import enabled as hypothesis_binding_enabled
        bind_hypotheses = hypothesis_binding_enabled()
        if bind_hypotheses and isinstance(step, int):
            self.revision_by_step.setdefault(step, binding["product_revision"])
        if len(messages) < self._message_cursor:
            self._message_cursor = 0
            self._last_assistant_index = None
        for index in range(self._message_cursor, len(messages)):
            if messages[index].get("role") == "assistant":
                self._last_assistant_index = index
        self._message_cursor = len(messages)
        if len(trace) < self._trace_cursor:
            self._trace_cursor = 0
            self._pending_trace_positions = []
        if self._last_assistant_index is not None and step > self.last_step:
            index = self._last_assistant_index
            assistant = copy.deepcopy(messages[index])
            calls = assistant.get("tool_calls") or []
            if calls:
                fn = calls[0].get("function", {})
                malformed = False
                try:
                    arguments = json.loads(fn.get("arguments") or "{}")
                except (ValueError, TypeError):
                    arguments = {}
                    malformed = True
                if not isinstance(arguments, dict):
                    arguments = {}
                    malformed = True
                rejected = any(str(message.get('content', '')).startswith(
                                   ('[step-contract]', '[information-policy]'))
                               for message in messages[index + 1:])
                if malformed or rejected:
                    fn['name'] = 'invalid_proposal'
                    arguments = {'proposal_status': 'rejected_before_dispatch'}
                rationale = arguments.get("why") or arguments.get("reason")
                if isinstance(rationale, str) and rationale.strip():
                    self.model_claims.append({"step": step, "claim": rationale[:600],
                                              "status": "UNVERIFIED_MODEL_CLAIM"})
                # Model-supplied revision claims can never bind a verdict.
                if not malformed and not rejected:
                    arguments.update({key: binding[key] for key in (
                        "product_revision", "test_harness_revision", "environment_revision")})
                fn["arguments"] = json.dumps(arguments)
            events = []
            positions = [*self._pending_trace_positions,
                         *range(self._trace_cursor, len(trace))]
            pending_positions: list[int] = []
            for position in positions:
                raw = trace[position]
                event = dict(raw)
                event_turn = event.get("turn")
                if type(event_turn) is not int and type(event.get("step")) is int:
                    event_turn = event["step"] - 1
                if type(event_turn) is not int or event_turn < 0:
                    continue
                if event_turn >= step:
                    pending_positions.append(position)
                    continue
                if "event_id" in event:
                    event_id = event["event_id"]
                else:
                    event_id = digest({"position": position, "event": event})
                    event["event_id"] = event_id
                if event_id in self.seen_trace_events:
                    continue
                event["step"] = event_turn + 1
                events.append(event)
                self.seen_trace_events.add(event["event_id"])
            self._trace_cursor = len(trace)
            self._pending_trace_positions = pending_positions
            self.frame = fold_turn(self.frame, Turn(
                index=max(0, step - 1), position=step,
                assistant=assistant, observations=copy.deepcopy(messages[index + 1:]),
                kernel_events=events))
            self.last_step = step
            if bind_hypotheses:
                from kernel_hypothesis_projection import bind_hypotheses as bind_records
                bind_records(self.frame.get("hypotheses"), self.revision_by_step,
                             self.initial_product_revision)
        # Text that resembles pytest output is not authoritative. Certification
        # is reconstructed exclusively from parent-observed subprocess receipts.
        facts = self.frame["facts"]
        facts["verdicts"], facts["test_observations"] = [], []
        for receipt in test_receipts:
            bound = receipt.get("binding", {})
            for selector in receipt.get("selectors", []):
                identity = digest({"receipt": receipt["receipt_id"], "selector": selector})
                turn = receipt.get("step", 0)
                verdict = {"receipt_id": identity, "turn": turn, "selector": selector,
                           "exit": receipt["exit_code"] if receipt.get("binding_stable") is True else None}
                facts["verdicts"].append(verdict)
                facts["test_observations"].append({**verdict, "binding": {
                    key: {"status": "bound", "value": bound[key]} if bound.get(key)
                    else {"status": "unavailable", "value": None}
                    for key in ("product_revision", "test_harness_revision", "environment_revision")}})
        facts["tests_run"] = len(facts["verdicts"])
        current = {key: {"status": "bound", "value": binding[key]} for key in (
            "product_revision", "test_harness_revision", "environment_revision")}
        self.frame.setdefault("action", {})["revision_evidence"] = current
        self.frame["world"].update(current)
        _refresh_derived(self.frame)
        authority = evaluate_required_tests(
            binding, self.frame["world"].get("required_selectors", ()), test_receipts)
        effective_authority = authority
        self.frame["world"]["valid_product_verification"] = authority.state == SATISFIED
        certification = dict(self.frame["world"].get("certification", {}))
        status = {"SATISFIED": "certified", "VIOLATED": "failed",
                  "CONFLICT": "conflict", "UNRESOLVED": "not_observed",
                  NO_HOST_CHECKS: "no_host_checks"}[authority.state]
        if authority.state == "UNRESOLVED" and any(
                reason.startswith(("STALE_OR_UNSTABLE_BINDING", "BINDING_UNSTABLE:"))
                for reason in authority.reason_codes):
            status = "stale"
        certification.update({
            "status": status,
            "state": authority.state,
            "policy_version": authority.policy_version,
            "supporting_evidence": list(authority.supporting_evidence),
            "reason_codes": list(authority.reason_codes),
        })
        if authority.state == "UNRESOLVED":
            certification.setdefault("missing_selectors", [
                reason.split(":", 1)[1] for reason in authority.reason_codes
                if reason.startswith("MISSING_SELECTOR:")])
        self.frame["world"]["certification"] = certification
        current_harness = binding['harness_files']
        original_changed = any(current_harness.get(name) != value
                               for name, value in self.initial_harness_files.items())
        unsafe_additions = [name for name in current_harness if name not in self.initial_harness_files
                            and not (Path(name).name.startswith('test_') and name.endswith('.py'))]
        if original_changed or unsafe_additions:
            effective_authority = Verdict(
                UNRESOLVED, authority.supporting_evidence,
                ["TASK_OWNED_PUBLIC_HARNESS_CHANGED"])
            self.frame['world']['certification'] = {
                'status': 'unavailable', 'reason': 'task_owned_public_harness_changed',
                'current_harness_checks_are_not_original_acceptance_evidence': True}
            self.frame['world']['valid_product_verification'] = False
        self.frame["world"]["completion_verdict"] = effective_authority.to_dict()
        self.frame["artifact_revisions"] = dict(binding["product_files"])
        return copy.deepcopy(self.frame)

    def public_view(self) -> dict:
        """Bound semantic state, with omission counts and no dialogue history."""
        frame = self.frame
        facts, world = frame["facts"], frame["world"]
        # The offline resume rubric assumes source cannot be resolved. A live
        # request actually includes the current working module, so exporting that
        # rubric would falsely tell the model that edit prerequisites are absent.
        certified = world["valid_product_verification"]
        next_action = ("review_public_requirements_then_done" if certified
                       else "revise_using_the_failing_test_observation" if facts["last_test_exit"] == 1
                       else "test_the_changed_source" if facts["edits_applied"]
                       else "use_current_source_and_requirements_to_choose_a_testable_change")
        readiness = {"current_source": "included_in_Working_source",
                     "edit": "available; replacement must parse and match the tool contract",
                     "str_replace": "available; old_str must match exactly once",
                     "verification": "current_public_checks_passed" if certified else "required",
                     "private_grader": "unknown; never exposed to the solver"}
        view = {"schema": "live-stateframe-rich/1", "task_id": frame["task_id"],
                'required_public_checks': list(world.get('required_selectors', [])),
                "objective": frame["objective"], "phase": frame["phase"],
                "artifact_revisions": frame.get("artifact_revisions", {}),
                "hypotheses": frame["hypotheses"][-8:],
                "hypotheses_omitted": max(0, len(frame["hypotheses"]) - 8),
                "hypothesis_state": frame["hypothesis_state"],
                "model_claims": self.model_claims[-4:],
                "model_claims_omitted": max(0, len(self.model_claims) - 4),
                "verification": world["certification"],
                "valid_product_verification": world["valid_product_verification"],
                # The raw receipt stream is deliberately not copied into this
                # compact model-facing view.  Carry the host-computed shared
                # verdict so lean/operational consumers cannot fall back to
                # prose or an initiator-specific status field.
                "completion_verdict": copy.deepcopy(world.get("completion_verdict", {})),
                "resume": {"recommended_next_action": next_action, "readiness": readiness},
                "pending": {"status": "no_concurrent_tool_execution"},
                "world": {"revision_binding_state": world["revision_binding_state"],
                          "test_world": world["test_world"],
                          "last_test": world["last_test"]},
                "obligations": {"public_requirements": "review_against_objective",
                                "current_public_checks": "passed" if certified else "not_yet_verified"},
                "negative_work": frame["negative_work"][-8:],
                "negative_work_omitted": max(0, len(frame["negative_work"]) - 8),
                "read_memory": (
                    [_compact_live_read(item) for item in facts["read_observations"][-3:]]
                    if evictfix_enabled() else [
                        {"target": item.get("target"), "turn": item.get("turn"),
                         "status": "HISTORICAL_OBSERVATION; reread if source changed",
                         "content": str(item.get("snapshot", {}).get("content", {}))[:900]}
                        for item in facts["read_observations"][-3:]
                    ]),
                "read_memory_omitted": max(0, len(facts["read_observations"]) - 3),
                "facts": {"tests_run": facts["tests_run"],
                          "last_test_exit": facts["last_test_exit"],
                          "files_inspected": facts["files_inspected"],
                          "recent_edits": facts["edits_applied"][-4:]}}
        from kernel_hypothesis_projection import HOST_REVISION_FIELD, enabled as hypothesis_binding_enabled
        if hypothesis_binding_enabled():
            revision = world.get("product_revision")
            view[HOST_REVISION_FIELD] = {
                "domain": "product_revision",
                "value": revision.get("value") if isinstance(revision, dict) else None,
                "initial": self.initial_product_revision}
        return view
