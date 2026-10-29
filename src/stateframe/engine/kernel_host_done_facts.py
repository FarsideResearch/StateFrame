"""Host-owned completion facts.

The completion claim is model-facing input, not completion evidence.  This
module keeps the two records separate: ``claims`` are retained for diagnosis,
while :class:`DoneFacts` is reconstructed from the workspace and typed runner
receipts.  The acceptance predicate never reads a claim.

The module is deliberately usable without a ``StateFrameRuntime`` instance so
the release boundary and small offline probes can exercise the authority in
isolation.
"""
from __future__ import annotations

import copy
import difflib
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path


MODE_ENV = "STATEFRAME_HOST_DONE_FACTS"
SCHEMA = "stateframe-host-done-facts/1"
DONE_FACTS_SCHEMA = SCHEMA

_PASS = frozenset({"PASS", "PASSED", "GREEN", "SUCCESS", "SATISFIED", "OK"})
_FAIL = frozenset({"FAIL", "FAILED", "RED", "ERROR", "REFUSED", "BLOCKED"})
_IGNORED_PARTS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", ".test-tmp", ".tox",
})


def host_done_facts_mode() -> str:
    """Return the strict opt-in value for this lane's authority."""
    value = os.environ.get(MODE_ENV, "off")
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def mode() -> str:
    return host_done_facts_mode()


def enabled() -> bool:
    return host_done_facts_mode() == "on"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _copy(value: object) -> object:
    try:
        return copy.deepcopy(value)
    except (TypeError, ValueError):
        return str(value)


def _json_safe(value: object) -> object:
    """Keep diagnostic claims serializable without granting them authority."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return str(value)


def _as_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (bytes, bytearray)):
        return (bytes(value).decode("utf-8", "replace"),)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _unique_text(values: Sequence[object] | None) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        values = (_as_text(values),)
    found: list[str] = []
    for value in values or ():
        text = _as_text(value)
        if text and text not in found:
            found.append(text)
    return tuple(found)


def _content_bytes(value: object) -> bytes:
    """Accept both our byte snapshots and small test-friendly snapshot maps."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8", "surrogateescape")
    if isinstance(value, Mapping):
        for key in ("content", "bytes", "data", "text"):
            if key in value:
                return _content_bytes(value[key])
    return repr(value).encode("utf-8", "backslashreplace")


def _path_is_ignored(relative: Path) -> bool:
    return any(part in _IGNORED_PARTS for part in relative.parts)


def workspace_snapshot(root: str | os.PathLike[str]) -> dict[str, bytes]:
    """Capture regular workspace files as a relative-path/content map.

    The snapshot deliberately excludes VCS and test-cache directories.  The
    runtime writes its measurement output outside the task workspace; this
    small exclusion set also prevents runner cache churn from becoming a
    solver edit.
    """
    base = Path(root)
    if not base.exists() or not base.is_dir():
        return {}
    result: dict[str, bytes] = {}
    for current, dirs, files in os.walk(base, topdown=True, followlinks=False):
        current_path = Path(current)
        dirs[:] = sorted(
            name for name in dirs
            if name not in _IGNORED_PARTS and not (current_path / name).is_symlink()
        )
        for name in sorted(files):
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                continue
            try:
                relative = path.relative_to(base)
            except ValueError:
                continue
            if _path_is_ignored(relative):
                continue
            try:
                result[relative.as_posix()] = path.read_bytes()
            except OSError:
                # An unreadable path cannot be represented as a false edit.
                # The host's test/check receipt remains the evidence gate.
                continue
    return {relative: result[relative] for relative in sorted(result, key=Path)}


snapshot_workspace = workspace_snapshot


def _lines(content: bytes) -> list[str]:
    return content.decode("utf-8", "surrogateescape").splitlines(keepends=True)


def _line_change(before: bytes, after: bytes) -> tuple[int, int]:
    old = _lines(before)
    new = _lines(after)
    added = deleted = 0
    for tag, old_start, old_end, new_start, new_end in difflib.SequenceMatcher(
            a=old, b=new, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        deleted += old_end - old_start
        added += new_end - new_start
    # Binary files do not have useful text opcodes.  Represent a changed binary
    # as one changed line so it still appears in the typed workspace diff.
    if before and after and not old and not new:
        return 1, 1
    return added, deleted


def _snapshot_revision(snapshot: Mapping[str, object]) -> str:
    rows = [(str(path), hashlib.sha256(_content_bytes(content)).hexdigest())
            for path, content in sorted(snapshot.items(), key=lambda item: str(item[0]))]
    return "workspace:" + _digest(rows)


def _revision(value: object) -> str | None:
    if isinstance(value, Mapping):
        for key in ("product_revision", "current_revision", "workspace_revision",
                    "revision", "artifact"):
            if value.get(key) is not None:
                candidate = _revision(value[key])
                if candidate:
                    return candidate
        return None
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        text = str(value)
        return text if text else None
    return _digest(value)


@dataclass(frozen=True)
class FileDelta:
    path: str
    status: str
    added_lines: int
    deleted_lines: int
    before_lines: int
    after_lines: int
    before_sha256: str | None = None
    after_sha256: str | None = None

    @property
    def net_lines(self) -> int:
        return self.added_lines - self.deleted_lines

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "status": self.status,
            "added_lines": self.added_lines,
            "deleted_lines": self.deleted_lines,
            "net_lines": self.net_lines,
            "before_lines": self.before_lines,
            "after_lines": self.after_lines,
            "before_sha256": self.before_sha256,
            "after_sha256": self.after_sha256,
        }


def diff_snapshots(before: Mapping[str, object], after: Mapping[str, object]) -> tuple[FileDelta, ...]:
    """Return deterministic typed deltas for the host workspace diff."""
    old = {str(path): _content_bytes(content) for path, content in before.items()}
    new = {str(path): _content_bytes(content) for path, content in after.items()}
    deltas: list[FileDelta] = []
    for path in sorted(set(old) | set(new)):
        before_bytes = old.get(path, b"")
        after_bytes = new.get(path, b"")
        if path in old and path in new and before_bytes == after_bytes:
            continue
        added, deleted = _line_change(before_bytes, after_bytes)
        deltas.append(FileDelta(
            path=path,
            status="added" if path not in old else "deleted" if path not in new else "modified",
            added_lines=added,
            deleted_lines=deleted,
            before_lines=len(_lines(before_bytes)) if path in old else 0,
            after_lines=len(_lines(after_bytes)) if path in new else 0,
            before_sha256=(hashlib.sha256(before_bytes).hexdigest() if path in old else None),
            after_sha256=(hashlib.sha256(after_bytes).hexdigest() if path in new else None),
        ))
    return tuple(deltas)


diff_workspace = diff_snapshots


@dataclass(frozen=True)
class CheckResult:
    status: str
    selectors: tuple[str, ...] = ()
    revision: str | None = None
    current: bool = False
    exit_code: int | None = None
    receipt_id: str | None = None
    source: str = "runner"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "result": self.status,
            "selectors": list(self.selectors),
            "revision": self.revision,
            "current": self.current,
            "exit_code": self.exit_code,
            "receipt_id": self.receipt_id,
            "source": self.source,
        }


def _selectors(row: Mapping[str, object]) -> tuple[str, ...]:
    values: list[object] = []
    for key in ("selectors", "public_selectors", "tests"):
        values.extend(_sequence(row.get(key)))
    for key in ("selector", "test_selector", "test", "name"):
        if row.get(key) is not None:
            values.append(row[key])
    return _unique_text(values)


def _status(row: Mapping[str, object]) -> str:
    exit_code = row.get("exit_code", row.get("returncode"))
    if isinstance(exit_code, bool):
        exit_code = int(exit_code)
    if isinstance(exit_code, int):
        return "PASS" if exit_code == 0 else "FAIL"
    for key in ("status", "result_class", "outcome", "result", "verdict"):
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip().upper()
        if text in _PASS or text.endswith("_PASS") or text.endswith("_GREEN"):
            return "PASS"
        if text in _FAIL or text.endswith("_FAIL") or text.endswith("_RED"):
            return "FAIL"
    return "UNKNOWN"


def _row_revision(row: Mapping[str, object]) -> str | None:
    for key in ("after_binding", "binding", "current", "revision_info"):
        value = row.get(key)
        candidate = _revision(value)
        if candidate:
            return candidate
    for key in ("current_revision", "workspace_revision", "product_revision", "revision",
                "artifact"):
        candidate = _revision(row.get(key))
        if candidate:
            return candidate
    return None


def _receipt_id(row: Mapping[str, object]) -> str | None:
    for key in ("receipt_id", "event_id", "id"):
        if row.get(key) is not None:
            return str(row[key])
    return None


def _is_current(row: Mapping[str, object], current_revision: str | None) -> bool:
    if row.get("binding_stable") is False or row.get("stale") is True:
        return False
    candidate = _row_revision(row)
    if candidate is not None and current_revision is not None:
        return candidate == current_revision
    # A typed receipt without a revision is still runner-owned evidence for
    # the revision at which the runtime emitted it.  Explicit revision data
    # above takes precedence and catches stale checks.
    return True


def _normalise_receipts(receipts: object) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    for item in _sequence(receipts):
        if not isinstance(item, Mapping):
            continue
        nested = item.get("receipt")
        if isinstance(nested, Mapping) and not any(
                key in item for key in ("exit_code", "selectors", "result_class", "status")):
            rows.append(nested)
        else:
            rows.append(item)
    return tuple(rows)


def _selector_covers(required: str, observed: str) -> bool:
    if required == observed:
        return True
    required_file = required.split("::", 1)[0]
    observed_file = observed.split("::", 1)[0]
    return required_file == observed or required == observed_file or (
        observed.startswith(required + "::") or required.startswith(observed + "::"))


def _claims(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return _json_safe(dict(value))  # type: ignore[return-value]
    if value is None:
        return {}
    return {"summary": _json_safe(value)}


@dataclass(frozen=True)
class DoneFacts:
    """A typed, runner-derived completion record.

    ``claims`` is intentionally present for audit/debugging, but is never read
    by :meth:`_rejection_reasons` or any acceptance helper.
    """

    current_revision: str
    changed_files: tuple[str, ...] = ()
    file_deltas: tuple[FileDelta, ...] = ()
    per_file_line_deltas: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    last_check_result: CheckResult | None = None
    public_tests_required: tuple[str, ...] = ()
    public_tests_ran: tuple[str, ...] = ()
    public_tests_passed: tuple[str, ...] = ()
    unresolved_gating_items: tuple[str, ...] = ()
    empty_diff: bool = True
    evidence_digest: str = ""
    claims: Mapping[str, object] = field(default_factory=dict)
    rejection_reasons: tuple[str, ...] = ()
    accepted: bool = field(init=False)

    def __post_init__(self) -> None:
        deltas = tuple(self.file_deltas)
        changed = tuple(self.changed_files) or tuple(delta.path for delta in deltas)
        line_deltas = {
            str(path): {
                "added_lines": int(values.get("added_lines", values.get("added", 0))),
                "deleted_lines": int(values.get("deleted_lines", values.get("deleted", 0))),
                "net_lines": int(values.get(
                    "net_lines", values.get("net", values.get("added_lines", 0) -
                    values.get("deleted_lines", 0)))),
            }
            for path, values in self.per_file_line_deltas.items()
            if isinstance(values, Mapping)
        }
        if not line_deltas:
            line_deltas = {
                delta.path: {"added_lines": delta.added_lines,
                             "deleted_lines": delta.deleted_lines,
                             "net_lines": delta.net_lines}
                for delta in deltas
            }
        unresolved = _unique_text(self.unresolved_gating_items)
        passed = _unique_text(self.public_tests_passed)
        if not passed and self.last_check_result is not None:
            if self.last_check_result.current and self.last_check_result.status == "PASS":
                passed = self.last_check_result.selectors
        reasons = self._rejection_reasons(
            empty_diff=bool(self.empty_diff),
            last_check=self.last_check_result,
            required=self.public_tests_required,
            ran=self.public_tests_ran,
            passed=passed,
            unresolved=unresolved,
        )
        if not self.evidence_digest:
            evidence = {
                "current_revision": self.current_revision,
                "changed_files": list(changed),
                "per_file_line_deltas": line_deltas,
                "last_check_result": (self.last_check_result.to_dict()
                                       if self.last_check_result else None),
                "public_tests_required": list(self.public_tests_required),
                "public_tests_ran": list(self.public_tests_ran),
                "public_tests_passed": list(passed),
                "unresolved_gating_items": list(unresolved),
                "empty_diff": bool(self.empty_diff),
            }
            evidence_digest = _digest(evidence)
        else:
            evidence_digest = str(self.evidence_digest)
        object.__setattr__(self, "changed_files", changed)
        object.__setattr__(self, "file_deltas", deltas)
        object.__setattr__(self, "per_file_line_deltas", line_deltas)
        object.__setattr__(self, "unresolved_gating_items", unresolved)
        object.__setattr__(self, "public_tests_passed", passed)
        object.__setattr__(self, "rejection_reasons", reasons)
        object.__setattr__(self, "evidence_digest", evidence_digest)
        object.__setattr__(self, "claims", _claims(self.claims))
        object.__setattr__(self, "accepted", not reasons)

    @staticmethod
    def _rejection_reasons(*, empty_diff: bool, last_check: CheckResult | None,
                           required: Sequence[str], ran: Sequence[str],
                           passed: Sequence[str],
                           unresolved: Sequence[str]) -> tuple[str, ...]:
        reasons: list[str] = []
        if empty_diff:
            reasons.append("EMPTY_DIFF")
        if required:
            if last_check is None:
                reasons.append("CHECK_MISSING")
            elif not last_check.current:
                reasons.append("CHECK_NOT_CURRENT")
            elif last_check.status != "PASS":
                reasons.append("CHECK_FAILED")
        pass_selectors = tuple(passed)
        missing = [selector for selector in required
                   if not any(_selector_covers(selector, observed) for observed in pass_selectors)]
        if missing:
            reasons.append("PUBLIC_TESTS_MISSING")
        if unresolved:
            reasons.append("UNRESOLVED_GATING_ITEMS")
        return tuple(reasons)

    @property
    def is_acceptable(self) -> bool:
        return self.accepted

    @property
    def host_check_state(self) -> str:
        return "REQUIRED_PUBLIC_CHECKS" if self.public_tests_required else "NO_HOST_CHECKS"

    @property
    def changed_file_paths(self) -> tuple[str, ...]:
        return self.changed_files

    @property
    def line_deltas(self) -> Mapping[str, Mapping[str, int]]:
        return self.per_file_line_deltas

    @property
    def last_check(self) -> CheckResult | None:
        return self.last_check_result

    @property
    def unresolved_gates(self) -> tuple[str, ...]:
        return self.unresolved_gating_items

    def to_dict(self) -> dict[str, object]:
        line_deltas = {path: dict(values)
                       for path, values in self.per_file_line_deltas.items()}
        return {
            "schema": SCHEMA,
            "current_revision": self.current_revision,
            "changed_files": list(self.changed_files),
            "changed_file_paths": list(self.changed_files),
            "file_deltas": [delta.to_dict() for delta in self.file_deltas],
            "per_file_line_deltas": line_deltas,
            "line_deltas": copy.deepcopy(line_deltas),
            "last_check_result": (self.last_check_result.to_dict()
                                   if self.last_check_result is not None else None),
            "public_tests_required": list(self.public_tests_required),
            "public_tests_ran": list(self.public_tests_ran),
            "public_tests_passed": list(self.public_tests_passed),
            "host_check_state": self.host_check_state,
            "no_host_checks": not bool(self.public_tests_required),
            "unresolved_gating_items": list(self.unresolved_gating_items),
            "empty_diff": self.empty_diff,
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "evidence_digest": self.evidence_digest,
            # Explicitly labelled non-authoritative at the serialization edge.
            "claims": _copy(self.claims),
        }

    as_dict = to_dict

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "DoneFacts":
        raw_deltas = value.get("file_deltas")
        deltas: list[FileDelta] = []
        for item in _sequence(raw_deltas):
            if not isinstance(item, Mapping):
                continue
            deltas.append(FileDelta(
                path=_as_text(item.get("path")),
                status=_as_text(item.get("status")) or "modified",
                added_lines=int(item.get("added_lines", item.get("added", 0)) or 0),
                deleted_lines=int(item.get("deleted_lines", item.get("deleted", 0)) or 0),
                before_lines=int(item.get("before_lines", 0) or 0),
                after_lines=int(item.get("after_lines", 0) or 0),
                before_sha256=(str(item["before_sha256"])
                               if item.get("before_sha256") is not None else None),
                after_sha256=(str(item["after_sha256"])
                              if item.get("after_sha256") is not None else None),
            ))
        raw_check = value.get("last_check_result")
        check = None
        if isinstance(raw_check, Mapping):
            exit_code = raw_check.get("exit_code")
            check = CheckResult(
                status=_status(raw_check),
                selectors=_unique_text(_sequence(raw_check.get("selectors"))),
                revision=_revision(raw_check.get("revision")),
                current=bool(raw_check.get("current")),
                exit_code=(int(exit_code) if isinstance(exit_code, int) else None),
                receipt_id=(str(raw_check["receipt_id"])
                            if raw_check.get("receipt_id") is not None else None),
                source=_as_text(raw_check.get("source")) or "runner",
            )
        raw_lines = value.get("per_file_line_deltas", value.get("line_deltas", {}))
        line_deltas = raw_lines if isinstance(raw_lines, Mapping) else {}
        raw_changed = value.get("changed_files", value.get("changed_file_paths", ()))
        changed_values = [item.get("path") if isinstance(item, Mapping) else item
                          for item in _sequence(raw_changed)]
        changed = _unique_text(changed_values)
        if not changed and deltas:
            changed = tuple(delta.path for delta in deltas)
        required = _unique_text(_sequence(value.get("public_tests_required")))
        ran = _unique_text(_sequence(value.get("public_tests_ran")))
        passed = _unique_text(_sequence(value.get("public_tests_passed")))
        return cls(
            current_revision=_as_text(value.get("current_revision")) or "unknown",
            changed_files=changed,
            file_deltas=tuple(deltas),
            per_file_line_deltas=line_deltas,
            last_check_result=check,
            public_tests_required=required,
            public_tests_ran=ran,
            public_tests_passed=passed,
            unresolved_gating_items=_unique_text(
                _sequence(value.get("unresolved_gating_items"))),
            empty_diff=bool(value.get("empty_diff", not changed)),
            evidence_digest=_as_text(value.get("evidence_digest")),
            claims=_claims(value.get("claims")),
        )

    from_dict = from_mapping


def compute_done_facts(
    root: str | os.PathLike[str],
    baseline: Mapping[str, object] | str | os.PathLike[str] | None = None,
    *,
    current_revision: object = None,
    edit_receipts: object = (),
    test_receipts: object = (),
    required_public_tests: Sequence[object] = (),
    public_tests: Sequence[object] = (),
    unresolved_gating_items: Sequence[object] = (),
    claims: object = None,
    runner_evidence: Mapping[str, object] | None = None,
) -> DoneFacts:
    """Build facts solely from a workspace snapshot and typed host receipts.

    ``edit_receipts`` is accepted as an audit input for API symmetry, but the
    authoritative changed-file calculation is always the workspace diff.  A
    caller may pass a mapping of runner evidence to make release-boundary use
    explicit; neither that mapping nor ``claims`` is consulted for acceptance.
    """
    del edit_receipts  # The filesystem is the source of truth for changed files.
    evidence = runner_evidence if isinstance(runner_evidence, Mapping) else {}
    if not test_receipts:
        test_receipts = evidence.get("test_receipts", evidence.get("checks", ()))
    if not unresolved_gating_items:
        unresolved_gating_items = _sequence(evidence.get("unresolved_gating_items"))
    if claims is None and "claims" in evidence:
        claims = evidence.get("claims")
    after = workspace_snapshot(root)
    if baseline is None:
        before: Mapping[str, object] = {}
    elif isinstance(baseline, (str, os.PathLike)):
        before = workspace_snapshot(baseline)
    else:
        before = baseline
    deltas = diff_snapshots(before, after)
    revision = _revision(current_revision) or _snapshot_revision(after)
    required = _unique_text(required_public_tests)
    if not required:
        required = _unique_text(public_tests)
    rows = _normalise_receipts(test_receipts)
    checks: list[CheckResult] = []
    current_rows: list[CheckResult] = []
    for row in rows:
        check = CheckResult(
            status=_status(row),
            selectors=_selectors(row),
            revision=_row_revision(row) or revision,
            current=_is_current(row, revision),
            exit_code=(int(row["exit_code"]) if isinstance(row.get("exit_code"), int)
                       else None),
            receipt_id=_receipt_id(row),
            source=_as_text(row.get("source")) or "runner",
        )
        checks.append(check)
        if check.current:
            current_rows.append(check)
    last_check = current_rows[-1] if current_rows else (checks[-1] if checks else None)
    public_ran: list[str] = []
    public_passed: list[str] = []
    for check in current_rows:
        for selector in check.selectors:
            if selector not in public_ran:
                public_ran.append(selector)
            if check.status == "PASS" and selector not in public_passed:
                public_passed.append(selector)
    line_deltas = {
        delta.path: {"added_lines": delta.added_lines,
                     "deleted_lines": delta.deleted_lines,
                     "net_lines": delta.net_lines}
        for delta in deltas
    }
    return DoneFacts(
        current_revision=revision,
        changed_files=tuple(delta.path for delta in deltas),
        file_deltas=deltas,
        per_file_line_deltas=line_deltas,
        last_check_result=last_check,
        public_tests_required=required,
        public_tests_ran=tuple(public_ran),
        public_tests_passed=tuple(public_passed),
        unresolved_gating_items=_unique_text(unresolved_gating_items),
        empty_diff=not bool(deltas),
        claims=_claims(claims),
    )


build_done_facts = compute_done_facts
derive_done_facts = compute_done_facts
from_runner_evidence = compute_done_facts


def claims_record(*, summary: object = None, claim: object = None,
                  evidence_refs: object = ()) -> dict[str, object]:
    """Return the diagnostic claim envelope without making it authoritative."""
    return {
        "summary": _json_safe(summary),
        "claim": _json_safe(claim),
        "evidence_refs": _json_safe(list(_sequence(evidence_refs))),
    }


def decorate_outcome(result: Mapping[str, object]) -> dict[str, object]:
    """Attach a host-authoritative completion view to a release outcome."""
    decorated = dict(result)
    done_seal = result.get("done_seal")
    if (isinstance(done_seal, Mapping)
            and done_seal.get("schema") == "stateframe-done-seal/1"
            and done_seal.get("stored") is True):
        decorated["completion"] = {
            "accepted": True,
            "authority": "done_seal",
            "schema": SCHEMA,
            "rejection_reasons": [],
            "evidence_digest": done_seal.get("artifact_digest"),
        }
        decorated["completion_accepted"] = True
        decorated["completion_authority"] = "done_seal"
        return decorated
    if "claims" not in decorated:
        claim_values = {key: result[key] for key in ("summary", "claim", "evidence_refs")
                        if key in result}
        if claim_values:
            decorated["claims"] = _claims(claim_values)
    raw = result.get("done_facts")
    if isinstance(raw, Mapping):
        facts = DoneFacts.from_mapping(raw)
        decorated["done_facts"] = facts.to_dict()
        submission_blocked = (
            result.get("accepted") is False
            or str(result.get("result", result.get("outcome", ""))).upper()
            in {"INCOMPLETE", "RUN_INCOMPLETE", "AUTHORITY_FAILURE"}
        )
        completion_accepted = bool(facts.accepted and not submission_blocked)
        completion = {
            "accepted": completion_accepted,
            "done_facts_accepted": facts.accepted,
            "authority": "host_done_facts",
            "schema": SCHEMA,
            "rejection_reasons": list(facts.rejection_reasons),
            "evidence_digest": facts.evidence_digest,
        }
        if submission_blocked:
            completion["rejection_reasons"].append("SUBMISSION_NOT_ACCEPTED")
    else:
        completion = {
            "accepted": False,
            "authority": "host_done_facts",
            "schema": SCHEMA,
            "rejection_reasons": ["DONE_FACTS_MISSING"],
        }
    decorated["completion"] = completion
    decorated["completion_accepted"] = bool(completion["accepted"])
    decorated["completion_authority"] = "host_done_facts"
    return decorated


def outcome_uses_done_facts(result: Mapping[str, object]) -> bool:
    return (isinstance(result.get("completion"), Mapping)
            and result["completion"].get("authority") == "host_done_facts")
