"""Explicit startup wiring for the three L126E silent-off organs.

Each organ has a ``STATEFRAME_*`` switch. Values ``1/true/yes/on/enabled``
request ON; ``0/false/no/off/disabled`` request OFF. Composed cognition is
default-OFF pending its ON/OFF ladder; the expectation and corpus switches
retain their existing default-ON request behavior.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


EffectiveState = Literal["on", "off", "failed"]

# Defaults preserve live intent where it was already active. Composed cognition
# stays opt-in until the requested ON/OFF ladder qualifies it for default-on.
_SWITCHES: dict[str, tuple[str, bool]] = {
    "composed_cognition": ("STATEFRAME_COMPOSED_COGNITION", False),
    "expectation_ledger": ("STATEFRAME_EXPECTATION_LEDGER", True),
    "corpus_fixes": ("STATEFRAME_CORPUS_FIXES", True),
}
_ON_VALUES = frozenset({"1", "true", "yes", "on", "enabled"})
_OFF_VALUES = frozenset({"0", "false", "no", "off", "disabled"})


@dataclass(frozen=True)
class OrganWiring:
    """Startup result paired with its one append-only wiring-state event."""

    value: object | None
    requested: bool
    effective: EffectiveState
    reason: str | None = None
    error_type: str | None = None


def _requested(organ: str) -> tuple[bool, str | None]:
    env_name, default = _SWITCHES[organ]
    raw = os.environ.get(env_name)
    if raw is None:
        return default, None
    normalized = raw.strip().casefold()
    if normalized in _ON_VALUES:
        return True, None
    if normalized in _OFF_VALUES:
        return False, None
    return False, "invalid_flag_value"


class PendingWiringJournal:
    """Buffer organ wiring events until the run journal exists, then replay them in order.

    10-08 (r2 ea37b0ac, live 930/931 crashed at startup with zero requests): react_repair wires its organs ~230 lines
    BEFORE StateFrameRuntime (and its journal) is constructed, so `measurement.journal` raised AttributeError on None.
    Wiring writes here; react_repair calls flush(measurement.journal) right after the runtime exists.
    """

    def __init__(self) -> None:
        self._events: list[tuple[str, dict[str, object]]] = []
        self._target: object = None

    def emit(self, kind: str, **fields: object) -> None:
        if self._target is not None:
            self._target.emit(kind, **fields)  # type: ignore[attr-defined]
        else:
            self._events.append((kind, dict(fields)))

    def flush(self, journal: object) -> int:
        """Replay buffered events into `journal`; later emits go straight through. Returns events replayed."""
        if not callable(getattr(journal, "emit", None)):
            raise TypeError("flush requires an append-only journal")
        events, self._events = self._events, []
        for kind, fields in events:
            journal.emit(kind, **fields)  # type: ignore[attr-defined]
        self._target = journal
        return len(events)


def _emit_state(
    journal: object,
    *,
    organ: str,
    requested: bool,
    effective: EffectiveState,
    reason: str | None = None,
    error_type: str | None = None,
) -> None:
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        raise TypeError("organ wiring requires an append-only journal")
    emit(
        "organ_wiring_state",
        organ=organ,
        requested=requested,
        effective=effective,
        reason=reason,
        error_type=error_type,
    )


def record_wiring_failure(
    journal: object,
    *,
    organ: str,
    operation: str,
    error: Exception,
) -> None:
    """Journal runtime organ failures that happen after startup wiring."""
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        raise TypeError("organ failure requires an append-only journal")
    emit(
        "organ_wiring_failure",
        organ=organ,
        operation=operation,
        reason="operation_failed",
        error_type=type(error).__name__,
    )


def stable_unit_id(task_row: Mapping[str, object], function: str) -> str:
    """Bind a composed unit to task, source, path, and symbol, including file tasks."""
    identity = {
        "task_id": str(task_row.get("task_id") or task_row.get("id") or ""),
        "source": str(task_row.get("sha") or ""),
        "path": str(task_row.get("rel") or task_row.get("path") or ""),
        "function": str(function or "").strip(),
    }
    if not any(identity[key] for key in ("task_id", "source", "path")):
        raise ValueError("composed cognition needs a stable task or source identity")
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"react-repair:{digest[:24]}"


def expectation_subject(function: str, task_row: Mapping[str, object]) -> str:
    """Use the target path as a stable subject for file-level tasks without a symbol."""
    symbol = str(function or "").strip()
    if symbol:
        return symbol
    return str(task_row.get("rel") or task_row.get("path") or "suite").strip() or "suite"


def _build_composed_cognition(
    task_row: Mapping[str, object], function: str, bolt: object,
) -> object:
    from composed_cognition import (
        AuthorizedAction,
        CandidateSource,
        ComposedCognition,
        RunScope,
    )
    from detreason.uledger import ReductionAction, ULedger, URow, UNKNOWN_ATOM
    import composed_cognition as composed_module

    source_sha = str(task_row["sha"])
    path = str(task_row.get("rel") or task_row.get("path") or "repair target")
    target = str(function or "").strip() or path
    baseline = ULedger(source_sha[:8])
    baseline.open(
        URow(
            id="u:baseline",
            utype=UNKNOWN_ATOM,
            claim=f"{target} repair baseline",
            u=0.8,
            reduction_actions=[ReductionAction("u-baseline", "cheap", 0.6)],
        )
    )
    names = tuple(getattr(bolt, "f2p_names", ())[:4])
    atoms = names or (f"{target} oracle must pass",)
    return ComposedCognition(
        scope=RunScope(
            mission_id=source_sha[:8],
            unit_id=stable_unit_id(task_row, function),
            task_id="repair",
            workspace_id="workspace",
            sandbox_id="sandbox",
            source_revision="run",
            env_id="env",
            oracle_id="oracle",
        ),
        acceptance_atoms=atoms,
        baseline_ledger=baseline,
        hypotheses=(
            composed_module.mg.Hypothesis("h1", f"the defect is inside {target} itself"),
            composed_module.mg.Hypothesis(
                "h2", f"the defect is in a caller/callee contract of {target}"
            ),
            composed_module.mg.Hypothesis("h3", "the failing evidence is being misread"),
        ),
        guard_anchors=(
            AuthorizedAction(
                "probe.target",
                CandidateSource.GUARD_ANCHOR,
                f"Re-read {path}:{target} against the oracle's demanded values",
                {"h1": "target", "h2": "caller", "h3": "target"},
                ordinal=1,
            ),
            AuthorizedAction(
                "probe.callers",
                CandidateSource.GUARD_ANCHOR,
                f"Inspect the call sites of {target} one hop out",
                {"h1": "target", "h2": "caller", "h3": "caller"},
                ordinal=2,
            ),
        ),
    )


def wire_composed_cognition(
    task_row: Mapping[str, object],
    function: str,
    bolt: object,
    journal: object,
) -> OrganWiring:
    requested, flag_reason = _requested("composed_cognition")
    if not requested:
        reason = flag_reason or "disabled_by_flag"
        _emit_state(
            journal, organ="composed_cognition", requested=False,
            effective="off", reason=reason,
        )
        return OrganWiring(None, False, "off", reason=reason)
    try:
        composed = _build_composed_cognition(task_row, function, bolt)
    except Exception as error:  # noqa: BLE001 - startup wiring must remain fail-open
        _emit_state(
            journal, organ="composed_cognition", requested=True,
            effective="failed", reason="initialization_failed",
            error_type=type(error).__name__,
        )
        return OrganWiring(
            None, True, "failed", reason="initialization_failed",
            error_type=type(error).__name__,
        )
    _emit_state(
        journal, organ="composed_cognition", requested=True, effective="on",
    )
    return OrganWiring(composed, True, "on")


def wire_expectation_ledger(journal: object) -> OrganWiring:
    requested, flag_reason = _requested("expectation_ledger")
    if not requested:
        reason = flag_reason or "disabled_by_flag"
        _emit_state(
            journal, organ="expectation_ledger", requested=False,
            effective="off", reason=reason,
        )
        return OrganWiring(None, False, "off", reason=reason)
    try:
        from expectation_ledger import ExpectationLedger

        ledger = ExpectationLedger()
    except Exception as error:  # noqa: BLE001 - startup wiring must remain fail-open
        _emit_state(
            journal, organ="expectation_ledger", requested=True,
            effective="failed", reason="initialization_failed",
            error_type=type(error).__name__,
        )
        return OrganWiring(
            None, True, "failed", reason="initialization_failed",
            error_type=type(error).__name__,
        )
    _emit_state(
        journal, organ="expectation_ledger", requested=True, effective="on",
    )
    return OrganWiring(ledger, True, "on")


def _release_declares_path(release_root: Path, relative_path: str) -> bool:
    """Check the actual release file list, then source declarations and TREES."""
    release_manifest_path = release_root / "release-manifest.json"
    if release_manifest_path.is_file():
        manifest = json.loads(release_manifest_path.read_text())
        files = manifest.get("files") if isinstance(manifest, Mapping) else None
        return isinstance(files, Mapping) and relative_path in files

    declaration_path = release_root / "release_declarations.json"
    if not declaration_path.is_file():
        return False
    declaration = json.loads(declaration_path.read_text())
    if not isinstance(declaration, Mapping):
        raise ValueError("release declarations must be a mapping")

    resource_patterns = [
        value for value in declaration.get("data_resources", ())
        if isinstance(value, str)
    ]
    package_trees = [
        value for value in declaration.get("package_trees", ())
        if isinstance(value, str)
    ]
    if any(fnmatch.fnmatchcase(relative_path, pattern) for pattern in resource_patterns):
        return True
    if any(
        relative_path == tree.rstrip("/")
        or relative_path.startswith(tree.rstrip("/") + "/")
        for tree in package_trees
    ):
        return True

    module_patterns = [
        value for value in declaration.get("python_modules", ())
        if isinstance(value, str)
    ]
    module_match = any(
        fnmatch.fnmatchcase(relative_path, pattern) for pattern in module_patterns
    )
    excluded = any(
        fnmatch.fnmatchcase(relative_path, pattern)
        for pattern in declaration.get("excluded_paths", ())
        if isinstance(pattern, str)
    )
    runtime_exception = any(
        fnmatch.fnmatchcase(relative_path, pattern)
        for pattern in declaration.get("runtime_exceptions", ())
        if isinstance(pattern, str)
    )
    if module_match and (not excluded or runtime_exception):
        return True

    # Runtime releases include Python files only from these declared trees.
    try:
        from capability_runtime_release import TREES
    except ImportError:
        TREES = ()
    in_runtime_tree = any(relative_path.startswith(f"{tree}/") for tree in TREES)
    return in_runtime_tree and relative_path.endswith(".py")


def wire_corpus_fixes(
    release_root: Path,
    task_root: Path,
    task_row: Mapping[str, object],
    resume: object | None,
    journal: object,
) -> OrganWiring:
    requested, flag_reason = _requested("corpus_fixes")
    if not requested:
        reason = flag_reason or "disabled_by_flag"
        _emit_state(journal, organ="corpus_fixes", requested=False,
                    effective="off", reason=reason)
        return OrganWiring(0, False, "off", reason=reason)
    try:
        # This package-qualified path is the name recorded by the immutable release.
        from probes import apply_corpus_fixes

        if resume is None:
            fix = apply_corpus_fixes.corpus_fix_for(task_row)
            if fix is not None and not _release_declares_path(
                release_root, str(fix.source_relpath)
            ):
                # The loader ships, but the matching oracle payload does not. Keep
                # hidden corpus test oracles out of releases and report that plainly.
                _emit_state(
                    journal, organ="corpus_fixes", requested=True,
                    effective="off", reason="not_shipped",
                )
                return OrganWiring(0, True, "off", reason="not_shipped")
            applied = apply_corpus_fixes.apply_corpus_fixes(task_root, task_row)
        else:
            applied = 0
    except Exception as error:  # noqa: BLE001 - startup wiring must remain fail-open
        _emit_state(
            journal, organ="corpus_fixes", requested=True,
            effective="failed", reason="initialization_failed",
            error_type=type(error).__name__,
        )
        return OrganWiring(
            0, True, "failed", reason="initialization_failed",
            error_type=type(error).__name__,
        )
    _emit_state(journal, organ="corpus_fixes", requested=True, effective="on")
    return OrganWiring(applied, True, "on")


def observe_expectation_suite(
    ledger: object,
    result: object,
    *,
    journal: object,
    environment: str,
    subject: str,
    selector: str,
    turn: int,
    workspace_revision: int,
) -> object:
    """Observe a suite and make post-start failures visible in the journal."""
    observe = getattr(ledger, "observe_suite_result", None)
    if not callable(observe):
        error = TypeError("expectation ledger has no observe_suite_result method")
        record_wiring_failure(
            journal, organ="expectation_ledger", operation="observe", error=error,
        )
        raise error
    try:
        return observe(
            result,
            environment=environment,
            subject=subject,
            selector=selector,
            turn=turn,
            workspace_revision=workspace_revision,
        )
    except Exception as error:
        record_wiring_failure(
            journal, organ="expectation_ledger", operation="observe", error=error,
        )
        raise
