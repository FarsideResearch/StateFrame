"""Opt-in, host-verified locate/act scaffolding.

Bounded locate is deliberately a guidance scaffold, not completion authority.
The model may propose a location, but the host owns the only receipt that can
say the proposed path and span existed at the observed revision.  The module
is independent of coding tools: a small adapter can translate a tool call into
the generic records defined here.

The default is OFF.  Importing this module, or constructing an OFF session,
does not emit events, change a frame, or alter an existing action.  An active
session emits ``bounded_locate_fired`` for every transition/action it takes so
an offline batch can distinguish a configured-but-unplugged scaffold from one
that actually ran.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
try:
    from enum import StrEnum
except ImportError:  # Python 3.9 compatibility for the v46 runner.
    class StrEnum(str, Enum):
        """Small compatibility equivalent for Python versions before 3.11."""

        def __str__(self) -> str:
            return self.value
from pathlib import Path, PurePosixPath
from types import MappingProxyType


LOCATE_SWITCH_ENV = "STATEFRAME_BOUNDED_LOCATE"
MODEL_ENV = "STATEFRAME_BOUNDED_LOCATE_MODEL"
READ_BUDGET_ENV = "STATEFRAME_BOUNDED_LOCATE_READ_BUDGET"
TOKEN_BUDGET_ENV = "STATEFRAME_BOUNDED_LOCATE_TOKEN_BUDGET"
BUDGETS_ENV = "STATEFRAME_BOUNDED_LOCATE_BUDGETS"


class LocatePhase(StrEnum):
    OFF = "off"
    LOCATE = "locate"
    PROBE_REQUIRED = "probe_required"
    ACT = "act"


OFF = LocatePhase.OFF
LOCATE = LocatePhase.LOCATE
PROBE_REQUIRED = LocatePhase.PROBE_REQUIRED
ACT = LocatePhase.ACT


class BoundedLocateError(RuntimeError):
    """Base error for a host locate/act policy refusal."""


class LocatePhaseError(BoundedLocateError):
    """The requested operation is not valid in the current phase."""


class LocateBudgetExceeded(BoundedLocateError):
    """A locate read or token budget was exhausted."""


class EditSiteVerificationError(ValueError):
    """The host could not verify a model-proposed edit site."""


_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_READ_TOOLS = frozenset({"read", "grep", "list", "ls", "recall_evidence", "recall",
                         "recall_generated_artifact", "read_ticket"})
_EDIT_TOOLS = frozenset({
    "edit", "str_replace", "file_create", "replace_module", "write", "replace",
})


def lexical_token_count(value: object) -> int:
    """Count bounded lexical tokens for accounting, without provider coupling."""

    return len(_TOKEN.findall(str(value or "")))


def _text(value: object, *, limit: int = 512) -> str:
    return " ".join(str(value or "").split())[:limit]


def _strict_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _refs(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        raw = tuple(value)
    elif isinstance(value, Iterable) and not isinstance(value, Mapping):
        raw = tuple(value)
    else:
        raw = (value,)
    result: list[str] = []
    for item in raw:
        text = _text(item, limit=240)
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _jsonable(value: object) -> object:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())
    return _text(value, limit=800)


def _digest(value: object) -> str:
    encoded = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _model_name(model: object = None) -> str:
    if model is None or not str(model).strip():
        model = os.environ.get(MODEL_ENV) or os.environ.get("DS_MODEL") or "default"
    return str(model).strip()


def model_family(model: object = None) -> str:
    """Map provider names to stable ablation cohorts, without model knowledge."""

    value = _model_name(model).lower()
    if any(token in value for token in (
        "small", "mini", "nano", "flash", "qwen", "llama", "mistral",
    )):
        return "small"
    if any(token in value for token in (
        "large", "frontier",
    )):
        return "large"
    return "medium"


def bounded_locate_enabled(model: object = None) -> bool:
    """Resolve the per-model opt-in switch; the compatibility default is OFF.

    ``STATEFRAME_BOUNDED_LOCATE=on`` enables the scaffold for the selected
    model.  A comma-separated value (for example ``small,medium``) enables
    only the named model/family, which makes ablation explicit and cheap.
    """

    raw = os.environ.get(LOCATE_SWITCH_ENV, "off").strip().lower()
    if raw in {"", "0", "off", "false", "no"}:
        return False
    if raw in {"1", "on", "true", "yes", "all"}:
        return True
    selected = _model_name(model).lower()
    family = model_family(selected)
    names = {item.strip() for item in raw.split(",") if item.strip()}
    if not names:
        raise ValueError(f"{LOCATE_SWITCH_ENV} must be off, on, or a model list")
    return selected in names or family in names


def enabled(model: object = None) -> bool:
    """Compatibility spelling for callers that use the generic switch API."""

    return bounded_locate_enabled(model)


def mode(model: object = None) -> str:
    return "on" if bounded_locate_enabled(model) else "off"


@dataclass(frozen=True)
class LocateBudget:
    """Separate read-count and lexical-token ceilings for one model cohort."""

    read_limit: int
    token_limit: int

    def __post_init__(self) -> None:
        _strict_int(self.read_limit, "read_limit", minimum=1)
        _strict_int(self.token_limit, "token_limit", minimum=1)

    @property
    def max_reads(self) -> int:
        return self.read_limit

    @property
    def read_budget(self) -> int:
        return self.read_limit

    @property
    def max_tokens(self) -> int:
        return self.token_limit

    @property
    def token_budget(self) -> int:
        return self.token_limit

    def to_dict(self) -> dict[str, int]:
        return {"read_limit": self.read_limit, "token_limit": self.token_limit}


DEFAULT_MODEL_BUDGETS = MappingProxyType({
    # These are deliberately modest guidance defaults, not completion bounds.
    "small": LocateBudget(8, 2400),
    "medium": LocateBudget(12, 4000),
    "large": LocateBudget(20, 7000),
})


def _env_positive(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        value = int(raw.strip())
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _budget_from_mapping(value: object, fallback: LocateBudget) -> LocateBudget:
    if not isinstance(value, Mapping):
        raise ValueError("bounded locate budget entries must be mappings")
    reads = next((value[key] for key in ("read_limit", "read_budget", "reads") if key in value), fallback.read_limit)
    tokens = next((value[key] for key in ("token_limit", "token_budget", "tokens") if key in value), fallback.token_limit)
    return LocateBudget(
        _strict_int(reads, "read_limit", minimum=1),
        _strict_int(tokens, "token_limit", minimum=1),
    )


def budget_for_model(model: object = None) -> LocateBudget:
    """Resolve a model-specific budget, with explicit environment overrides."""

    selected = _model_name(model)
    family = model_family(selected)
    fallback = DEFAULT_MODEL_BUDGETS[family]
    raw = os.environ.get(BUDGETS_ENV)
    if raw is not None:
        try:
            configured = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"{BUDGETS_ENV} must be a JSON object") from error
        if not isinstance(configured, Mapping):
            raise ValueError(f"{BUDGETS_ENV} must be a JSON object")
        selected_value = configured.get(selected, configured.get(family, configured.get("default")))
        if selected_value is not None:
            fallback = _budget_from_mapping(selected_value, fallback)
    reads = _env_positive(READ_BUDGET_ENV)
    tokens = _env_positive(TOKEN_BUDGET_ENV)
    return LocateBudget(reads or fallback.read_limit, tokens or fallback.token_limit)


@dataclass(frozen=True)
class EditSpan:
    """One-based inclusive line span with optional zero-based columns."""

    start_line: int
    end_line: int
    start_column: int = 0
    end_column: int | None = None

    def __post_init__(self) -> None:
        _strict_int(self.start_line, "start_line", minimum=1)
        _strict_int(self.end_line, "end_line", minimum=1)
        _strict_int(self.start_column, "start_column", minimum=0)
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        if self.end_column is not None:
            _strict_int(self.end_column, "end_column", minimum=0)
            if self.end_line == self.start_line and self.end_column <= self.start_column:
                raise ValueError("same-line span must have positive width")

    @classmethod
    def coerce(cls, value: object) -> "EditSpan":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            start_line = value.get("start_line", value.get("line"))
            end_line = value.get("end_line", value.get("end", start_line))
            return cls(
                start_line=int(start_line),
                end_line=int(end_line),
                start_column=int(value.get("start_column", value.get("column", 0))),
                end_column=(None if value.get("end_column") is None
                            else int(value["end_column"])),
            )
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            values = tuple(value)
            if len(values) == 2:
                return cls(int(values[0]), int(values[1]))
            if len(values) == 3:
                return cls(int(values[0]), int(values[1]), int(values[2]))
            if len(values) == 4:
                return cls(int(values[0]), int(values[1]), int(values[2]), int(values[3]))
        raise ValueError("span must be a mapping, EditSpan, or 2-4 item sequence")

    def to_dict(self) -> dict[str, int | None]:
        return {
            "start_line": self.start_line,
            "end_line": self.end_line,
            "start_column": self.start_column,
            "end_column": self.end_column,
        }


@dataclass(frozen=True)
class EditSiteHypothesis:
    """Model proposal: path/span plus references to observations it used."""

    path: str
    span: EditSpan | Mapping[str, object] | Sequence[object]
    evidence_refs: tuple[str, ...] | Sequence[object] = ()

    def __post_init__(self) -> None:
        path = str(self.path).replace("\\", "/").strip()
        if not path:
            raise ValueError("EditSiteHypothesis.path is required")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "span", EditSpan.coerce(self.span))
        object.__setattr__(self, "evidence_refs", _refs(self.evidence_refs))

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "span": self.span.to_dict(),
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True)
class EditSiteReceipt:
    """Host-typed receipt produced only after span and evidence validation."""

    hypothesis: EditSiteHypothesis
    source_revision: str
    span_digest: str
    receipt_id: str
    verified: bool = True
    host_authority: str = "HOST_OBSERVED"

    @property
    def path(self) -> str:
        return self.hypothesis.path

    @property
    def span(self) -> EditSpan:
        return self.hypothesis.span  # type: ignore[return-value]

    @property
    def evidence_refs(self) -> tuple[str, ...]:
        return self.hypothesis.evidence_refs

    @property
    def host_verified(self) -> bool:
        return self.verified

    def to_dict(self) -> dict[str, object]:
        return {
            "receipt_id": self.receipt_id,
            "path": self.path,
            "span": self.span.to_dict(),
            "evidence_refs": list(self.evidence_refs),
            "source_revision": self.source_revision,
            "span_digest": self.span_digest,
            "verified": self.verified,
            "host_verified": self.verified,
            "host_authority": self.host_authority,
        }


@dataclass(frozen=True)
class DiscriminatingProbe:
    """Typed question request emitted when locate cannot spend another read."""

    probe_id: str
    basis_revision: str
    subject_path: str
    question: str
    choices: tuple[str, ...] = ("narrow_existing_site", "different_site", "act_without_more_reads")
    forced: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "probe_id": self.probe_id,
            "basis_revision": self.basis_revision,
            "subject_path": self.subject_path,
            "question": self.question,
            "choices": list(self.choices),
            "forced": self.forced,
        }


@dataclass(frozen=True)
class ReadReceipt:
    """Host observation used to charge a locate budget and seed evidence refs."""

    path: str
    read_index: int
    token_count: int
    evidence_ref: str
    basis_revision: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "read_index": self.read_index,
            "token_count": self.token_count,
            "evidence_ref": self.evidence_ref,
            "basis_revision": self.basis_revision,
        }


def _safe_relative(path: object) -> str:
    raw = str(path or "").replace("\\", "/").strip()
    if not raw or raw.startswith("/"):
        raise EditSiteVerificationError("edit site path must be a non-empty relative path")
    parsed = PurePosixPath(raw)
    if ".." in parsed.parts:
        raise EditSiteVerificationError("edit site path may not escape the workspace")
    normalized = parsed.as_posix().lstrip("./")
    if not normalized or normalized == ".":
        raise EditSiteVerificationError("edit site path is empty")
    return normalized


def _source_for(
    hypothesis: EditSiteHypothesis,
    *,
    root: str | os.PathLike[str] | Path | None,
    source_loader: Callable[[str], str] | Mapping[str, str] | None,
) -> tuple[str, str]:
    relative = _safe_relative(hypothesis.path)
    if source_loader is not None:
        try:
            source = (source_loader(relative) if callable(source_loader)
                      else source_loader[relative])
        except (KeyError, OSError, UnicodeError, TypeError) as error:
            raise EditSiteVerificationError(f"host source unavailable for {relative}") from error
        if not isinstance(source, str):
            raise EditSiteVerificationError(f"host source for {relative} is not text")
        return relative, source
    if root is None:
        raise EditSiteVerificationError("host root or source_loader is required")
    workspace = Path(root).resolve()
    candidate = (workspace / relative).resolve()
    try:
        inside = candidate.is_relative_to(workspace)
    except AttributeError:  # pragma: no cover - retained for older Python workers
        inside = str(candidate).startswith(str(workspace) + os.sep)
    if not inside or not candidate.is_file():
        raise EditSiteVerificationError(f"host cannot verify source path {relative}")
    try:
        source = candidate.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise EditSiteVerificationError(f"host cannot read source path {relative}") from error
    return relative, source


def verify_edit_site(
    hypothesis: EditSiteHypothesis,
    *,
    root: str | os.PathLike[str] | Path | None = None,
    source_loader: Callable[[str], str] | Mapping[str, str] | None = None,
    known_evidence_refs: Iterable[object] | None = None,
) -> EditSiteReceipt:
    """Verify a proposal against current host source and known host evidence."""

    if not isinstance(hypothesis, EditSiteHypothesis):
        hypothesis = EditSiteHypothesis(**dict(hypothesis))  # type: ignore[arg-type]
    refs = set(_refs(known_evidence_refs)) if known_evidence_refs is not None else None
    if not hypothesis.evidence_refs:
        raise EditSiteVerificationError("edit site requires at least one evidence_ref")
    if refs is not None and not set(hypothesis.evidence_refs).issubset(refs):
        unknown = sorted(set(hypothesis.evidence_refs) - refs)
        raise EditSiteVerificationError("unknown host evidence refs: " + ", ".join(unknown))
    relative, source = _source_for(hypothesis, root=root, source_loader=source_loader)
    span = hypothesis.span
    lines = source.splitlines(keepends=True)
    if not lines:
        raise EditSiteVerificationError(f"host source path {relative} is empty")
    if span.end_line > len(lines):
        raise EditSiteVerificationError(
            f"edit span ends at line {span.end_line}, but {relative} has {len(lines)} lines"
        )
    start_text = lines[span.start_line - 1].rstrip("\r\n")
    end_text = lines[span.end_line - 1].rstrip("\r\n")
    if span.start_column > len(start_text):
        raise EditSiteVerificationError("edit span start_column is outside the source line")
    if span.end_column is not None and span.end_column > len(end_text):
        raise EditSiteVerificationError("edit span end_column is outside the source line")
    selected = lines[span.start_line - 1:span.end_line]
    selected[0] = selected[0][span.start_column:]
    if span.end_column is not None:
        selected[-1] = selected[-1][:span.end_column]
    selected_text = "".join(selected)
    if not selected_text:
        raise EditSiteVerificationError("edit span selects no source text")
    revision = "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()
    span_digest = "sha256:" + hashlib.sha256(selected_text.encode("utf-8")).hexdigest()
    receipt_id = "edit-site:" + _digest({
        "path": relative,
        "span": span.to_dict(),
        "evidence_refs": list(hypothesis.evidence_refs),
        "source_revision": revision,
    })[:24]
    normalized = EditSiteHypothesis(relative, span, hypothesis.evidence_refs)
    return EditSiteReceipt(normalized, revision, span_digest, receipt_id)


class BoundedLocateSession:
    """State machine for one locate/act episode."""

    def __init__(
        self,
        *,
        root: str | os.PathLike[str] | Path | None = None,
        model: object = None,
        budget: LocateBudget | None = None,
        journal: object | None = None,
        question_engine: object | None = None,
        probe_callback: Callable[[DiscriminatingProbe], object] | None = None,
        known_evidence_refs: Iterable[object] = (),
        source_loader: Callable[[str], str] | Mapping[str, str] | None = None,
        enabled: bool | None = None,
        basis_revision: object = "",
    ) -> None:
        self.model = _model_name(model)
        self.enabled = bounded_locate_enabled(self.model) if enabled is None else bool(enabled)
        self.root = Path(root).resolve() if root is not None else None
        self.budget = budget or budget_for_model(self.model)
        self.journal = journal
        self.question_engine = question_engine
        self.probe_callback = probe_callback
        self.source_loader = source_loader
        self.basis_revision = str(basis_revision or "")
        self.phase = LOCATE if self.enabled else OFF
        self.reads_used = 0
        self.tokens_used = 0
        self._started = False
        self._reads_closed = False
        self._observations: list[ReadReceipt] = []
        self._evidence_refs = set(_refs(known_evidence_refs))
        self._receipt: EditSiteReceipt | None = None
        self._probe: DiscriminatingProbe | None = None
        self._probe_delivery: object = None
        self._probe_answer: object = None

    @property
    def receipt(self) -> EditSiteReceipt | None:
        return self._receipt

    @property
    def pending_probe(self) -> DiscriminatingProbe | None:
        return self._probe if self.phase is PROBE_REQUIRED else None

    @property
    def observations(self) -> tuple[ReadReceipt, ...]:
        return tuple(self._observations)

    def _emit(self, action: str, **fields: object) -> None:
        if not self.enabled:
            return
        payload = {
            "mechanism": "bounded_locate",
            "action": action,
            "fired": True,
            "phase": self.phase.value,
            "model": self.model,
            "reads_used": self.reads_used,
            "tokens_used": self.tokens_used,
            "budget": self.budget.to_dict(),
            **fields,
        }
        emitter = getattr(self.journal, "emit", None)
        if callable(emitter):
            emitter("bounded_locate_fired", **payload)
        elif callable(self.journal):
            self.journal("bounded_locate_fired", **payload)

    def start(self) -> dict[str, object]:
        if not self.enabled:
            return {}
        if not self._started:
            self._started = True
            self._emit("locate_started", basis_revision=self.basis_revision)
        return self.projection()

    def _ensure_started(self) -> None:
        if self.enabled and not self._started:
            self.start()

    def _blocked(self, reason: str, *, tool: str = "") -> str:
        self._emit("action_blocked", tool=tool, reason=reason)
        return f"[bounded-locate] REFUSED: {reason}"

    def _read_path(self, path: object) -> str:
        raw = str(path or "").replace("\\", "/").strip()
        if self.root is not None:
            try:
                candidate = Path(raw)
                if candidate.is_absolute():
                    candidate = candidate.resolve().relative_to(self.root)
                raw = candidate.as_posix()
            except (OSError, ValueError):
                pass
        return raw.lstrip("./") or "<unknown>"

    def _force_probe(self, *, reason: str, path: str = "") -> DiscriminatingProbe:
        if self._probe is not None:
            return self._probe
        self.phase = PROBE_REQUIRED
        self._probe = DiscriminatingProbe(
            probe_id="locate-probe:" + _digest({
                "model": self.model,
                "basis_revision": self.basis_revision,
                "path": path,
                "reads_used": self.reads_used,
                "tokens_used": self.tokens_used,
            })[:24],
            basis_revision=self.basis_revision,
            subject_path=path,
            question=(
                "Which observed location most discriminates the next edit? "
                "Choose a known site and provide an EditSiteHypothesis; do not read another file."
            ),
        )
        self._emit("budget_exhausted", reason=reason, subject_path=path)
        self._emit("forced_discriminating_probe", probe=self._probe.to_dict(), reason=reason)
        callback = self.probe_callback
        if callback is None:
            candidate = self.question_engine
            if callable(candidate):
                callback = candidate
            for name in ("force_discriminating_probe", "ask_discriminating", "ask"):
                if callback is not None:
                    break
                method = getattr(candidate, name, None)
                if callable(method):
                    callback = method
                    break
        if callback is not None:
            try:
                self._probe_delivery = callback(self._probe)
            except Exception as error:  # pragma: no cover - foreign question engines
                self._probe_delivery = {
                    "status": "QUESTION_ENGINE_ERROR",
                    "error_type": type(error).__name__,
                }
        elif self.question_engine is not None:
            self._probe_delivery = {"status": "PENDING_QUESTION_ENGINE"}
        return self._probe

    def record_read(
        self,
        path: object,
        *,
        token_count: int | None = None,
        content: object = None,
        evidence_ref: object = None,
        basis_revision: object = None,
    ) -> ReadReceipt | None:
        """Charge one host-observed read or force a probe at the ceiling."""

        if not self.enabled:
            return None
        self._ensure_started()
        path_text = self._read_path(path)
        if self.phase is ACT or self._reads_closed:
            self._emit("observation_after_locate_close", path=path_text, phase=self.phase.value)
        if content is not None:
            tokens = lexical_token_count(content)
        elif token_count is None:
            tokens = 1
        else:
            tokens = _strict_int(token_count, "token_count", minimum=0)
        exceeds_reads = self.reads_used + 1 > self.budget.read_limit
        exceeds_tokens = self.tokens_used + tokens > self.budget.token_limit
        if exceeds_reads or exceeds_tokens:
            reason = "read_limit" if exceeds_reads else "token_limit"
            self._emit("observation_budget_exceeded", reason=reason, path=path_text)
        self.reads_used += 1
        self.tokens_used += tokens
        reference = _text(evidence_ref, limit=240) or f"bounded-read:{self.reads_used}"
        self._evidence_refs.add(reference)
        observation = ReadReceipt(
            path_text, self.reads_used, tokens, reference,
            str(basis_revision if basis_revision is not None else self.basis_revision),
        )
        self._observations.append(observation)
        self._emit("read_observed", observation=observation.to_dict())
        return observation

    def observe_action_receipt(
        self, receipt: Mapping[str, object], *, content: object = None,
        token_count: int | None = None,
    ) -> ReadReceipt | None:
        """Coding-pack seam: turn an existing action receipt into a read charge."""

        if not self.enabled or not isinstance(receipt, Mapping):
            return None
        kind = str(receipt.get("kind", receipt.get("tool", ""))).lower()
        if kind not in _READ_TOOLS:
            return None
        path = receipt.get("target_path") or receipt.get("path") or receipt.get("intent") or "<read>"
        try:
            return self.record_read(
                path,
                token_count=token_count,
                content=content,
                evidence_ref=receipt.get("receipt_id"),
                basis_revision=receipt.get("workspace_revision", self.basis_revision),
            )
        except LocateBudgetExceeded:
            return None

    def answer_probe(self, answer: object) -> object:
        if not self.enabled:
            return None
        if self.phase is not PROBE_REQUIRED or self._probe is None:
            raise LocatePhaseError("no forced discriminating probe is pending")
        if answer is None or (isinstance(answer, str) and not answer.strip()):
            raise ValueError("probe answer is required")
        self._probe_answer = _jsonable(answer)
        self._reads_closed = True
        self.phase = LOCATE
        self._emit("probe_answered", probe_id=self._probe.probe_id, answer=self._probe_answer)
        return answer

    def submit_hypothesis(self, hypothesis: EditSiteHypothesis) -> EditSiteReceipt:
        """Host-verify a typed hypothesis and close locate for the act phase."""

        if not self.enabled:
            raise LocatePhaseError("bounded locate is off")
        self._ensure_started()
        if self.phase is PROBE_REQUIRED:
            raise LocatePhaseError("answer the forced discriminating probe first")
        if self.phase is ACT and self._receipt is not None:
            return self._receipt
        if self.phase is not LOCATE:
            raise LocatePhaseError(f"cannot submit an edit site in phase {self.phase.value}")
        try:
            receipt = verify_edit_site(
                hypothesis,
                root=self.root,
                source_loader=self.source_loader,
                known_evidence_refs=self._evidence_refs,
            )
        except (EditSiteVerificationError, TypeError, ValueError) as error:
            self._emit("hypothesis_rejected", reason=str(error), hypothesis=_jsonable(hypothesis))
            if isinstance(error, EditSiteVerificationError):
                raise
            raise EditSiteVerificationError(str(error)) from error
        self._receipt = receipt
        self.phase = ACT
        self._emit("edit_site_receipt", receipt=receipt.to_dict())
        return receipt

    def _creates_new_file(self, path: object) -> bool:
        """True when `path` names a file that does not exist yet inside the session root."""
        if self.root is None or not str(path or "").strip():
            return False
        try:
            candidate = Path(str(path))
            candidate = (candidate if candidate.is_absolute() else self.root / candidate).resolve()
            candidate.relative_to(self.root)
        except (OSError, ValueError):
            return False
        return not candidate.exists()

    def before_action(self, tool: object, arguments: Mapping[str, object] | None = None) -> str | None:
        """Return host feedback for an optional coding-tool dispatch hook."""

        if not self.enabled:
            return None
        self._ensure_started()
        name = str(tool or "").strip().lower()
        args = arguments if isinstance(arguments, Mapping) else {}
        if name in _READ_TOOLS:
            return None
        if name in _EDIT_TOOLS:
            if name == "file_create" and self._creates_new_file(args.get("path")):
                # WIP rule: creating a planned sibling that does not exist yet is never a locate violation.
                self._emit("new_file_admitted", tool=name, path=self._read_path(args.get("path")))
                return None
            if self.phase is not ACT or self._receipt is None:
                return self._blocked(
                    "submit a host-verified EditSiteHypothesis before acting", tool=name)
            requested = args.get("path")
            if requested:
                try:
                    requested_path = _safe_relative(requested)
                except EditSiteVerificationError:
                    requested_path = str(requested)
                if requested_path != self._receipt.path:
                    return self._blocked(
                        f"act on receipt path {self._receipt.path}, not {requested_path}", tool=name)
        return None

    def projection(self) -> dict[str, object]:
        if not self.enabled:
            return {}
        return {
            "schema": "bounded-locate/1",
            "enabled": True,
            "phase": self.phase.value,
            "model": self.model,
            "budget": self.budget.to_dict(),
            "reads_used": self.reads_used,
            "tokens_used": self.tokens_used,
            "reads_remaining": max(0, self.budget.read_limit - self.reads_used),
            "tokens_remaining": max(0, self.budget.token_limit - self.tokens_used),
            "observed_reads": [item.to_dict() for item in self._observations],
            "evidence_refs": sorted(self._evidence_refs),
            "edit_site_receipt": self._receipt.to_dict() if self._receipt is not None else None,
            "pending_probe": self._probe.to_dict() if self.phase is PROBE_REQUIRED and self._probe else None,
            "probe_delivery": _jsonable(self._probe_delivery),
            "guidance": (
                "locate: submit EditSiteHypothesis(path, span, evidence_refs)"
                if self.phase is LOCATE else
                "act: use the host-verified edit-site receipt"
                if self.phase is ACT else
                "answer the forced discriminating probe; do not read another file"
            ),
        }


# Friendly aliases for callers that name the mechanism as a phase rather than a session.
BoundedLocate = BoundedLocateSession
EditSiteReceiptVerificationError = EditSiteVerificationError


__all__ = [
    "ACT", "OFF", "LOCATE", "PROBE_REQUIRED", "LocatePhase", "BoundedLocateError",
    "LocatePhaseError", "LocateBudgetExceeded", "EditSiteVerificationError",
    "EditSiteReceiptVerificationError", "LocateBudget", "DEFAULT_MODEL_BUDGETS",
    "EditSpan", "EditSiteHypothesis", "EditSiteReceipt", "ReadReceipt",
    "DiscriminatingProbe", "BoundedLocateSession", "BoundedLocate", "lexical_token_count",
    "bounded_locate_enabled", "enabled", "mode", "model_family", "budget_for_model",
    "verify_edit_site", "LOCATE_SWITCH_ENV", "MODEL_ENV", "READ_BUDGET_ENV",
    "TOKEN_BUDGET_ENV", "BUDGETS_ENV",
]
