"""Pure System One distribution calibration and immutable artifact identity.

This module owns probability mechanics only.  It does not choose an action,
apply a loss specification, emit a policy decision, or produce an evaluation
report.  Inputs are complete, same-position score vectors.  A caller must not
pass an observed top-k subset and describe it as a calibrated distribution.

Support is always declared by the caller and is stored in the artifact.  A
site below that declaration remains ``UNCALIBRATED`` and ``SHADOW_ONLY``;
calibration output is still useful for diagnostics but carries no authority.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Iterable


CALIBRATION_SCHEMA = "sys1-calibration/1"
UNCALIBRATED = "UNCALIBRATED"
CALIBRATED = "CALIBRATED"
SHADOW_ONLY = "SHADOW_ONLY"
_QUESTION_TYPES = frozenset({"CHOICE", "BOOLEAN", "SCORE"})
_EPSILON = 1e-300


class CalibrationError(ValueError):
    """Base error for malformed calibration inputs or artifacts."""


class CalibrationIdentityError(CalibrationError):
    """Raised when an artifact or score vector is bound to another identity."""


def _labels(value: Sequence[object], field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise CalibrationError(f"{field} must be a sequence of labels")
    result = tuple(_required_text(item, f"{field}[]") for item in value)
    if not result:
        raise CalibrationError(f"{field} must not be empty")
    if len(set(result)) != len(result):
        raise CalibrationError(f"{field} must not contain duplicates")
    return result


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise CalibrationError(f"{field} must be finite")
    return result


def _positive(value: object, field: str) -> float:
    result = _finite(value, field)
    if result <= 0:
        raise CalibrationError(f"{field} must be positive")
    return result


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CalibrationError(f"{field} must be a non-negative integer")
    return value


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or "\x00" in value:
        raise CalibrationError(f"{field} must be a non-empty canonical string")
    return value


def _freeze(value: object, field: str) -> object:
    """Convert JSON-like identity data into a deterministic immutable value."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _finite(value, field)
    if isinstance(value, Mapping):
        frozen: list[tuple[str, object]] = []
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise CalibrationError(f"{field} mapping keys must be non-empty strings")
            frozen.append((key, _freeze(item, f"{field}.{key}")))
        return tuple(sorted(frozen, key=lambda pair: pair[0]))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze(item, f"{field}[]") for item in value)
    raise CalibrationError(f"{field} must be JSON-like")


def _thaw(value: object) -> object:
    if isinstance(value, tuple):
        if value and all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in value):
            return {item[0]: _thaw(item[1]) for item in value}
        return [_thaw(item) for item in value]
    return value


def _thaw_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, tuple):
        raise CalibrationError("frozen mapping is invalid")
    return {item[0]: _thaw(item[1]) for item in value if isinstance(item, tuple) and len(item) == 2}


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CalibrationKey:
    """The complete identity of a score population and its serving regime."""

    site: str
    schema: str
    menu: object
    levels: object
    evidence: str
    procedure: str
    prompt: str
    model: str
    tokenizer: str
    server_domain: str
    settings: object

    def __post_init__(self) -> None:
        for name in ("site", "schema", "evidence", "procedure", "prompt", "model", "tokenizer", "server_domain"):
            object.__setattr__(self, name, _required_text(getattr(self, name), f"key.{name}"))
        if not isinstance(self.menu, Sequence) or isinstance(self.menu, (str, bytes, bytearray)):
            raise CalibrationError("key.menu must be a sequence")
        if not self.menu:
            raise CalibrationError("key.menu must not be empty")
        if not isinstance(self.levels, Sequence) or isinstance(self.levels, (str, bytes, bytearray)):
            raise CalibrationError("key.levels must be a sequence")
        if not isinstance(self.settings, Mapping):
            raise CalibrationError("key.settings must be a mapping")
        frozen_menu = _freeze(self.menu, "key.menu")
        frozen_levels = _freeze(self.levels, "key.levels")
        for name, value in (("menu", frozen_menu), ("levels", frozen_levels)):
            if isinstance(value, tuple) and len({ _canonical_json(_thaw(item)) for item in value }) != len(value):
                raise CalibrationError(f"key.{name} must not contain duplicate entries")
        object.__setattr__(self, "menu", frozen_menu)
        object.__setattr__(self, "levels", frozen_levels)
        object.__setattr__(self, "settings", _freeze(self.settings, "key.settings"))
        _definition_ids(frozen_menu, "key.menu")
        if frozen_levels:
            _definition_ids(frozen_levels, "key.levels")

    @property
    def site_id(self) -> str:
        return self.site

    @property
    def question_schema(self) -> str:
        return self.schema

    @property
    def menu_signature(self) -> object:
        return self.menu

    @property
    def level_definitions(self) -> object:
        return self.levels

    @property
    def evidence_selector(self) -> str:
        return self.evidence

    @property
    def procedure_version(self) -> str:
        return self.procedure

    @property
    def prompt_version(self) -> str:
        return self.prompt

    @property
    def model_identity(self) -> str:
        return self.model

    @property
    def tokenizer_identity(self) -> str:
        return self.tokenizer

    @property
    def score_domain(self) -> str:
        return self.server_domain

    @property
    def serving_settings(self) -> object:
        return self.settings

    def to_dict(self) -> dict[str, object]:
        return {
            "site": self.site,
            "schema": self.schema,
            "menu": _thaw(self.menu),
            "levels": _thaw(self.levels),
            "evidence": self.evidence,
            "procedure": self.procedure,
            "prompt": self.prompt,
            "model": self.model,
            "tokenizer": self.tokenizer,
            "server_domain": self.server_domain,
            "settings": _thaw_mapping(self.settings),
        }

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CalibrationKey":
        required = {
            "site", "schema", "menu", "levels", "evidence", "procedure", "prompt",
            "model", "tokenizer", "server_domain", "settings",
        }
        if not isinstance(value, Mapping):
            raise CalibrationError("calibration key must be a mapping")
        missing = sorted(required - set(value))
        extra = sorted(set(value) - required)
        if missing or extra:
            raise CalibrationError(f"calibration key fields mismatch; missing={missing}, extra={extra}")
        return cls(**{name: value[name] for name in required})


@dataclass(frozen=True, slots=True)
class SupportRequirement:
    """Caller-declared support gates; values are not universal safety claims."""

    minimum_labels: int
    minimum_per_class: int = 1
    minimum_independent_labels: int = 0

    def __post_init__(self) -> None:
        _nonnegative_int(self.minimum_labels, "support.minimum_labels")
        _nonnegative_int(self.minimum_per_class, "support.minimum_per_class")
        _nonnegative_int(self.minimum_independent_labels, "support.minimum_independent_labels")
        if self.minimum_labels == 0:
            raise CalibrationError("support.minimum_labels must be positive")

    def to_dict(self) -> dict[str, int]:
        return {
            "minimum_labels": self.minimum_labels,
            "minimum_per_class": self.minimum_per_class,
            "minimum_independent_labels": self.minimum_independent_labels,
        }


@dataclass(frozen=True, slots=True)
class SupportReport:
    count: int
    independent_count: int
    per_class: tuple[tuple[str, int], ...]
    supported: bool
    reasons: tuple[str, ...]


def _support_report(
    targets: Sequence[str], requirement: SupportRequirement, expected_labels: Sequence[str],
    independent_label_count: int | None,
) -> SupportReport:
    counts: dict[str, int] = {label: 0 for label in expected_labels}
    for target in targets:
        counts[target] = counts.get(target, 0) + 1
    independent = len(targets) if independent_label_count is None else _nonnegative_int(
        independent_label_count, "independent_label_count"
    )
    reasons: list[str] = []
    if len(targets) < requirement.minimum_labels:
        reasons.append("insufficient_labels")
    if independent < requirement.minimum_independent_labels:
        reasons.append("insufficient_independent_labels")
    if any(counts.get(label, 0) < requirement.minimum_per_class for label in expected_labels):
        reasons.append("insufficient_per_class_support")
    return SupportReport(
        count=len(targets),
        independent_count=independent,
        per_class=tuple(sorted(counts.items())),
        supported=not reasons,
        reasons=tuple(reasons),
    )


def logsumexp(values: Iterable[float]) -> float:
    """Compute log(sum(exp(values))) without overflow or shift sensitivity."""

    numbers = tuple(_finite(value, "score") for value in values)
    if not numbers:
        raise CalibrationError("at least one score is required")
    maximum = max(numbers)
    return maximum + math.log(sum(math.exp(value - maximum) for value in numbers))


def _score_items(scores: object, labels: Sequence[str] | None) -> list[tuple[str, float]]:
    if isinstance(scores, Mapping):
        if not scores:
            raise CalibrationError("complete score vector cannot be empty")
        if any(not isinstance(label, str) or not label for label in scores):
            raise CalibrationError("score labels must be non-empty strings")
        if labels is None:
            ordered = list(scores.keys())
        else:
            ordered = list(_labels(labels, "labels"))
            if set(ordered) != set(scores):
                raise CalibrationError("score vector must contain exactly the declared labels")
        return [(label, _finite(scores[label], f"scores[{label}]") ) for label in ordered]
    if isinstance(scores, Sequence) and not isinstance(scores, (str, bytes, bytearray)):
        if labels is None:
            raise CalibrationError("sequence scores require an exact label sequence")
        ordered = list(_labels(labels, "labels"))
        if len(ordered) != len(scores):
            raise CalibrationError("sequence scores and labels must be a non-empty one-to-one vector")
        return [(label, _finite(value, f"scores[{label}]")) for label, value in zip(ordered, scores)]
    raise CalibrationError("scores must be a mapping or numeric sequence")


def normalize_complete_vector(
    scores: Mapping[str, float] | Sequence[float],
    labels: Sequence[str] | None = None,
    *,
    expected_labels: Sequence[str] | None = None,
) -> dict[str, float]:
    """Softmax one complete same-position vector with stable logsumexp.

    ``labels``/``expected_labels`` are required for sequence inputs and, when
    supplied for mappings, enforce exact set equality.  There is deliberately
    no missing-option or observed-subset branch.
    """

    if labels is not None and expected_labels is not None and tuple(labels) != tuple(expected_labels):
        raise CalibrationError("labels and expected_labels disagree")
    declared = expected_labels if expected_labels is not None else labels
    items = _score_items(scores, declared)
    normalizer = logsumexp(value for _, value in items)
    result = {label: math.exp(value - normalizer) for label, value in items}
    # Correct the final rounding residue without changing the complete support.
    last = items[-1][0]
    result[last] = 1.0 - sum(value for label, value in result.items() if label != last)
    if result[last] < 0.0 or not math.isfinite(result[last]):
        raise CalibrationError("normalization produced an invalid probability")
    return result


def _complete_probability_vector(
    probabilities: Mapping[str, float] | Sequence[float],
    labels: Sequence[str] | None = None,
) -> dict[str, float]:
    """Validate a complete probability vector without renormalizing a subset."""

    items = _score_items(probabilities, labels)
    if any(value < 0.0 for _, value in items):
        raise CalibrationError("probabilities must be non-negative")
    total = sum(value for _, value in items)
    if not math.isclose(total, 1.0, rel_tol=1e-8, abs_tol=1e-8):
        raise CalibrationError("complete probability vector must sum to one")
    result = {label: value for label, value in items}
    last = items[-1][0]
    result[last] = 1.0 - sum(value for label, value in result.items() if label != last)
    if result[last] < 0.0:
        raise CalibrationError("probability vector has invalid rounding residue")
    return result


def apply_temperature(
    scores: Mapping[str, float] | Sequence[float], temperature: float,
    labels: Sequence[str] | None = None,
) -> dict[str, float]:
    """Apply positive scalar temperature to the complete raw score vector."""

    value = _positive(temperature, "temperature")
    items = _score_items(scores, labels)
    maximum = max(score for _, score in items)
    scaled = {label: (score - maximum) / value for label, score in items}
    return normalize_complete_vector(scaled, labels=[label for label, _ in items])


@dataclass(frozen=True, slots=True)
class TemperatureFit:
    temperature: float
    lower: float
    upper: float
    boundary: str | None
    nll: float
    label_count: int

    def __post_init__(self) -> None:
        temperature = _positive(self.temperature, "temperature_fit.temperature")
        lower = _positive(self.lower, "temperature_fit.lower")
        upper = _positive(self.upper, "temperature_fit.upper")
        if lower >= upper or not lower <= temperature <= upper:
            raise CalibrationError("temperature fit bounds are invalid")
        if self.boundary not in (None, "lower", "upper"):
            raise CalibrationError("temperature fit boundary is invalid")
        _finite(self.nll, "temperature_fit.nll")
        _nonnegative_int(self.label_count, "temperature_fit.label_count")

    @property
    def at_boundary(self) -> bool:
        return self.boundary is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "temperature": self.temperature,
            "lower": self.lower,
            "upper": self.upper,
            "boundary": self.boundary,
            "nll": self.nll,
            "label_count": self.label_count,
        }


def _probability_to_scores(value: object, labels: Sequence[str] | None) -> dict[str, float]:
    items = _score_items(value, labels)
    probabilities = [score for _, score in items]
    if any(score < 0.0 for score in probabilities):
        raise CalibrationError("probabilities must be non-negative")
    total = sum(probabilities)
    if not math.isclose(total, 1.0, rel_tol=1e-8, abs_tol=1e-8):
        raise CalibrationError("probability vector must already be complete and sum to one")
    return {label: math.log(max(score, _EPSILON)) for label, score in items}


def _row_scores(record: object, label_override: object = None) -> tuple[dict[str, float], str, str | None]:
    if isinstance(record, Sequence) and not isinstance(record, (str, bytes, bytearray, Mapping)):
        if len(record) not in (2, 3):
            raise CalibrationError("training tuple must be (scores, label[, independent_id])")
        raw_scores = record[0]
        target = record[1]
        independent_id = record[2] if len(record) == 3 else None
    elif isinstance(record, Mapping):
        target = record.get("label", record.get("truth", record.get("target", record.get("y"))))
        if target is None:
            raise CalibrationError("training row requires a label")
        independent_id = record.get("independent_id", record.get("label_id"))
        raw_scores = None
        for name in ("scores", "raw_scores", "logits", "logprobs"):
            if name in record:
                raw_scores = record[name]
                break
        if raw_scores is None and "probabilities" in record:
            labels = record.get("labels", record.get("option_ids"))
            raw_scores = _probability_to_scores(record["probabilities"], labels if isinstance(labels, Sequence) else None)
        if raw_scores is None:
            raise CalibrationError("training row requires scores or probabilities")
        labels = record.get("labels", record.get("option_ids"))
        if labels is not None and (isinstance(labels, (str, bytes, bytearray)) or not isinstance(labels, Sequence)):
            raise CalibrationError("row labels must be a sequence")
    else:
        raise CalibrationError("training row must be a mapping or tuple")
    if label_override is not None:
        target = label_override
    if isinstance(target, bool) or not isinstance(target, (str, int)):
        raise CalibrationError("training label must be a string or integer option id")
    target_text = str(target)
    labels_value = locals().get("labels") if isinstance(record, Mapping) else None
    scores = dict(_score_items(raw_scores, labels_value))
    if target_text not in scores:
        raise CalibrationError("training label is absent from the complete score vector")
    if independent_id is not None and not isinstance(independent_id, (str, int)):
        raise CalibrationError("independent_id must be a string or integer")
    return scores, target_text, None if independent_id is None else str(independent_id)


def _training_rows(records: Sequence[object], labels: Sequence[object] | None = None) -> tuple[list[tuple[dict[str, float], str, str | None]], tuple[str, ...]]:
    if isinstance(records, (str, bytes, bytearray)):
        raise CalibrationError("training records must be a sequence")
    if labels is not None and len(records) != len(labels):
        raise CalibrationError("records and labels must have equal lengths")
    rows = [_row_scores(record, labels[index] if labels is not None else None) for index, record in enumerate(records)]
    if not rows:
        raise CalibrationError("at least one labelled score vector is required")
    expected = tuple(rows[0][0])
    if any(set(row[0]) != set(expected) for row in rows[1:]):
        raise CalibrationError("all training vectors must have the same complete label set")
    return rows, expected


def _temperature_nll(rows: Sequence[tuple[dict[str, float], str, str | None]], temperature: float) -> float:
    value = _positive(temperature, "temperature")
    total = 0.0
    for scores, target, _ in rows:
        maximum = max(scores.values())
        scaled = [(score - maximum) / value for score in scores.values()]
        total += logsumexp(scaled) - (scores[target] - maximum) / value
    return total / len(rows)


def fit_temperature_report(
    records: Sequence[object], labels: Sequence[object] | None = None,
    *, bounds: tuple[float, float] = (0.05, 20.0),
) -> TemperatureFit:
    """Fit bounded positive temperature and report whether a boundary won."""

    rows, _ = _training_rows(records, labels)
    if len(bounds) != 2:
        raise CalibrationError("temperature bounds must contain lower and upper")
    lower = _positive(bounds[0], "temperature lower bound")
    upper = _positive(bounds[1], "temperature upper bound")
    if lower >= upper:
        raise CalibrationError("temperature bounds must be ordered")
    log_lower, log_upper = math.log(lower), math.log(upper)
    grid_size = 256
    grid = [log_lower + (log_upper - log_lower) * index / grid_size for index in range(grid_size + 1)]
    values = [_temperature_nll(rows, math.exp(value)) for value in grid]
    best = min(range(len(grid)), key=lambda index: (values[index], index))
    if best == 0:
        temperature = lower
        boundary = "lower"
    elif best == grid_size:
        temperature = upper
        boundary = "upper"
    else:
        left, right = grid[best - 1], grid[best + 1]
        golden = (1.0 + math.sqrt(5.0)) / 2.0
        for _ in range(96):
            first = right - (right - left) / golden
            second = left + (right - left) / golden
            if _temperature_nll(rows, math.exp(first)) <= _temperature_nll(rows, math.exp(second)):
                right = second
            else:
                left = first
        temperature = math.exp((left + right) / 2.0)
        boundary = None
    return TemperatureFit(temperature, lower, upper, boundary, _temperature_nll(rows, temperature), len(rows))


def fit_temperature(
    records: Sequence[object], labels: Sequence[object] | None = None,
    *, bounds: tuple[float, float] = (0.05, 20.0),
) -> float:
    """Compatibility scalar view of :func:`fit_temperature_report`."""

    return fit_temperature_report(records, labels, bounds=bounds).temperature


def _binary_rows(samples: Sequence[object]) -> tuple[list[tuple[float, int, str | None]], int]:
    result: list[tuple[float, int, str | None]] = []
    ids: set[str] = set()
    for sample in samples:
        if isinstance(sample, Mapping):
            # Deliberately do not accept a self-reported ``confidence`` field;
            # this lane calibrates host-observed score values only.
            raw_score = sample.get("score", sample.get("q"))
            raw_label = sample.get("label", sample.get("acceptable", sample.get("truth")))
            raw_id = sample.get("independent_id", sample.get("label_id"))
        elif isinstance(sample, Sequence) and not isinstance(sample, (str, bytes, bytearray)) and len(sample) in (2, 3):
            raw_score, raw_label = sample[0], sample[1]
            raw_id = sample[2] if len(sample) == 3 else None
        else:
            raise CalibrationError("binary sample must be a mapping or tuple")
        score = _finite(raw_score, "isotonic score")
        if not 0.0 <= score <= 1.0:
            raise CalibrationError("isotonic scores must be in [0, 1]")
        if isinstance(raw_label, bool):
            label = int(raw_label)
        elif isinstance(raw_label, int) and raw_label in (0, 1):
            label = raw_label
        else:
            raise CalibrationError("binary labels must be boolean, 0, or 1")
        ident = None if raw_id is None else _required_text(str(raw_id), "independent_id")
        if ident is not None:
            ids.add(ident)
        result.append((score, label, ident))
    if not result:
        raise CalibrationError("at least one binary label is required")
    return result, (len(ids) if ids else len(result))


def _pav(samples: Sequence[tuple[float, int, str | None]]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    grouped: list[list[float]] = []
    for score, label, _ in sorted(samples, key=lambda row: row[0]):
        if grouped and grouped[-1][0] == score:
            grouped[-1][1] += label
            grouped[-1][2] += 1.0
        else:
            grouped.append([score, float(label), 1.0])
    blocks: list[list[float]] = []
    for score, positive, count in grouped:
        blocks.append([score, score, positive, count])
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            left_mean, right_mean = left[2] / left[3], right[2] / right[3]
            if left_mean <= right_mean:
                break
            blocks[-2] = [left[0], right[1], left[2] + right[2], left[3] + right[3]]
            blocks.pop()
    return tuple(block[1] for block in blocks), tuple(block[2] / block[3] for block in blocks)


@dataclass(frozen=True, slots=True)
class IsotonicCalibrator:
    kind: str
    status: str
    thresholds: tuple[float, ...]
    values: tuple[float, ...]
    label_count: int
    independent_label_count: int
    tail_policy: str = "constant"

    def __post_init__(self) -> None:
        _required_text(self.kind, "isotonic.kind")
        if self.status not in (CALIBRATED, UNCALIBRATED):
            raise CalibrationError("isotonic status is invalid")
        if len(self.thresholds) != len(self.values) or (self.status == CALIBRATED and not self.thresholds):
            raise CalibrationError("isotonic knots must be paired and non-empty when calibrated")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in self.thresholds + self.values):
            raise CalibrationError("isotonic knots must be finite probabilities")
        if any(self.thresholds[index] > self.thresholds[index + 1] for index in range(len(self.thresholds) - 1)):
            raise CalibrationError("isotonic thresholds must be sorted")
        if any(self.values[index] > self.values[index + 1] for index in range(len(self.values) - 1)):
            raise CalibrationError("isotonic values must be monotone")
        if self.tail_policy != "constant":
            raise CalibrationError("isotonic tail policy is unsupported")
        _nonnegative_int(self.label_count, "isotonic.label_count")
        _nonnegative_int(self.independent_label_count, "isotonic.independent_label_count")

    @property
    def qualified(self) -> bool:
        return self.status == CALIBRATED

    def predict(self, score: float) -> float:
        if not self.qualified:
            raise CalibrationError("isotonic calibrator is unqualified and shadow-only")
        value = _finite(score, "isotonic prediction score")
        index = bisect.bisect_left(self.thresholds, value)
        index = min(max(index, 0), len(self.values) - 1)
        return min(1.0, max(0.0, self.values[index]))

    @property
    def p_selected_acceptable(self) -> "IsotonicCalibrator":
        return self

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "status": self.status,
            "thresholds": list(self.thresholds),
            "values": list(self.values),
            "label_count": self.label_count,
            "independent_label_count": self.independent_label_count,
            "tail_policy": self.tail_policy,
        }


def fit_binary_isotonic(
    samples: Sequence[object], *, support: SupportRequirement,
    kind: str = "p_true", independent_label_count: int | None = None,
) -> IsotonicCalibrator:
    """Fit PAV only after explicit binary and independent-label support."""

    rows, inferred_independent = _binary_rows(samples)
    declared_independent = inferred_independent if independent_label_count is None else independent_label_count
    report = _support_report(
        [str(label) for _, label, _ in rows], support, ("0", "1"), declared_independent,
    )
    if not report.supported:
        return IsotonicCalibrator(kind, UNCALIBRATED, (), (), report.count, report.independent_count)
    thresholds, values = _pav(rows)
    return IsotonicCalibrator(kind, CALIBRATED, thresholds, values, report.count, report.independent_count)


def fit_selected_acceptability(
    samples: Sequence[object], *, support: SupportRequirement,
    independent_label_count: int | None = None,
) -> IsotonicCalibrator:
    """Fit the separate top-label target named ``p_selected_acceptable``."""

    return fit_binary_isotonic(
        samples, support=support, kind="p_selected_acceptable",
        independent_label_count=independent_label_count,
    )


def expected_rank(probabilities: Mapping[str, float], levels: Mapping[str, float] | Sequence[float]) -> float:
    """Return the SCORE expectation, explicitly an expected rank/value."""

    labels = tuple(probabilities)
    distribution = _complete_probability_vector(probabilities, labels)
    if isinstance(levels, Mapping):
        if set(levels) != set(labels):
            raise CalibrationError("level values must cover the full SCORE distribution")
        values = [_finite(levels[label], f"levels[{label}]") for label in labels]
    else:
        if isinstance(levels, (str, bytes, bytearray)) or len(levels) != len(labels):
            raise CalibrationError("level values must align with the full SCORE distribution")
        values = [_finite(value, "level value") for value in levels]
    return sum(distribution[label] * value for label, value in zip(labels, values))


def ordinal_tail_probabilities(
    probabilities: Mapping[str, float], levels: Sequence[str],
) -> dict[str, float]:
    """Return coherent ``P(level >= cutoff)`` tails from the full distribution."""

    labels = tuple(levels)
    if not labels or len(set(labels)) != len(labels) or set(labels) != set(probabilities):
        raise CalibrationError("ordinal levels must exactly cover the SCORE distribution")
    distribution = _complete_probability_vector(probabilities, labels)
    return {
        label: sum(distribution[item] for item in labels[index:])
        for index, label in enumerate(labels)
    }


@dataclass(frozen=True, slots=True)
class OrdinalTailCalibrator:
    level_ids: tuple[str, ...]
    maps: tuple[IsotonicCalibrator, ...]
    status: str
    heldout_baseline_rps: float
    heldout_calibrated_rps: float

    def __post_init__(self) -> None:
        level_ids = _labels(self.level_ids, "ordinal.level_ids")
        object.__setattr__(self, "level_ids", level_ids)
        if len(level_ids) < 2 or len(self.maps) != len(level_ids) - 1:
            raise CalibrationError("ordinal tail maps must cover every non-terminal cutoff")
        if self.status != CALIBRATED or any(not item.qualified for item in self.maps):
            raise CalibrationError("only a qualified ordinal tail artifact may be constructed")
        baseline = _finite(self.heldout_baseline_rps, "heldout_baseline_rps")
        calibrated = _finite(self.heldout_calibrated_rps, "heldout_calibrated_rps")
        if baseline < 0.0 or calibrated < 0.0:
            raise CalibrationError("ordinal RPS values must be non-negative")
        if calibrated >= baseline:
            raise CalibrationError("ordinal tails require strict held-out improvement")

    def apply(self, probabilities: Mapping[str, float]) -> dict[str, float]:
        distribution = _complete_probability_vector(probabilities, self.level_ids)
        raw_tails = ordinal_tail_probabilities(distribution, self.level_ids)
        predicted: list[float] = []
        for level_id, calibrator in zip(self.level_ids[1:], self.maps):
            predicted.append(calibrator.predict(raw_tails[level_id]))
        coherent = []
        previous = 1.0
        for value in predicted:
            value = min(previous, max(0.0, min(1.0, value)))
            coherent.append(value)
            previous = value
        bounds = [1.0, *coherent, 0.0]
        result = {level_id: bounds[index] - bounds[index + 1] for index, level_id in enumerate(self.level_ids)}
        return _complete_probability_vector(result, self.level_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "level_ids": list(self.level_ids),
            "maps": [item.to_dict() for item in self.maps],
            "status": self.status,
            "heldout_baseline_rps": self.heldout_baseline_rps,
            "heldout_calibrated_rps": self.heldout_calibrated_rps,
        }


def _record_distribution(record: object, labels: Sequence[str]) -> tuple[dict[str, float], str]:
    if not isinstance(record, Mapping):
        raise CalibrationError("ordinal records must be mappings")
    raw_label = record.get("label", record.get("truth", record.get("target")))
    if raw_label is None:
        raise CalibrationError("ordinal record requires a label")
    target = str(raw_label)
    if target not in labels:
        if isinstance(raw_label, int) and 0 <= raw_label < len(labels):
            target = labels[raw_label]
        else:
            raise CalibrationError("ordinal label is absent from levels")
    if "probabilities" in record:
        probabilities = _probability_to_scores(record["probabilities"], labels)
        probabilities = normalize_complete_vector(probabilities, expected_labels=labels)
    else:
        raw = record.get("scores", record.get("raw_scores", record.get("logits")))
        if raw is None:
            raise CalibrationError("ordinal record requires probabilities or scores")
        probabilities = normalize_complete_vector(raw, expected_labels=labels)
    return probabilities, target


def _rps(distribution: Mapping[str, float], target_index: int, levels: Sequence[str]) -> float:
    cumulative = 0.0
    result = 0.0
    for index in range(len(levels) - 1):
        cumulative += distribution[levels[index]]
        observed = 1.0 if target_index <= index else 0.0
        result += (cumulative - observed) ** 2
    return result / max(1, len(levels) - 1)


def fit_ordinal_tails(
    training_records: Sequence[Mapping[str, object]],
    heldout_records: Sequence[Mapping[str, object]],
    *,
    levels: Sequence[str],
    support: SupportRequirement,
    independent_label_count: int | None = None,
    heldout_independent_label_count: int | None = None,
    minimum_improvement: float = 0.0,
) -> OrdinalTailCalibrator | None:
    """Fit coherent ordinal tails only when a separate unseen test set improves RPS."""

    level_ids = tuple(_required_text(label, "ordinal level") for label in levels)
    if len(level_ids) < 2 or len(set(level_ids)) != len(level_ids):
        raise CalibrationError("ordinal levels must be unique and contain at least two levels")
    if _finite(minimum_improvement, "minimum_improvement") < 0:
        raise CalibrationError("minimum_improvement must be non-negative")
    train = [_record_distribution(row, level_ids) for row in training_records]
    holdout = [_record_distribution(row, level_ids) for row in heldout_records]
    if not train or not holdout:
        return None
    targets = [target for _, target in train]
    report = _support_report(targets, support, level_ids, independent_label_count)
    if not report.supported:
        return None
    holdout_count = len(holdout) if heldout_independent_label_count is None else _nonnegative_int(
        heldout_independent_label_count, "heldout_independent_label_count"
    )
    if holdout_count < support.minimum_independent_labels:
        return None
    maps: list[IsotonicCalibrator] = []
    for cutoff in range(1, len(level_ids)):
        samples = [
            (
                sum(probabilities[level_id] for level_id in level_ids[cutoff:]),
                int(target in level_ids[cutoff:]),
            )
            for probabilities, target in train
        ]
        maps.append(
            fit_binary_isotonic(
                samples,
                support=SupportRequirement(1, 1, 0),
                kind=f"ordinal_tail:{level_ids[cutoff]}",
            )
        )
    baseline = 0.0
    calibrated = 0.0
    for probabilities, target in holdout:
        target_index = level_ids.index(target)
        baseline += _rps(probabilities, target_index, level_ids)
        raw_tails = ordinal_tail_probabilities(probabilities, level_ids)
        tail_values = [calibrator.predict(raw_tails[level_ids[index]]) for index, calibrator in enumerate(maps, 1)]
        coherent: list[float] = []
        previous = 1.0
        for value in tail_values:
            value = min(previous, max(0.0, min(1.0, value)))
            coherent.append(value)
            previous = value
        bounds = [1.0, *coherent, 0.0]
        candidate = {level_ids[index]: bounds[index] - bounds[index + 1] for index in range(len(level_ids))}
        calibrated += _rps(candidate, target_index, level_ids)
    baseline /= len(holdout)
    calibrated /= len(holdout)
    if not calibrated < baseline - float(minimum_improvement):
        return None
    return OrdinalTailCalibrator(level_ids, tuple(maps), CALIBRATED, baseline, calibrated)


@dataclass(frozen=True, slots=True)
class CalibratedDistribution:
    probabilities_items: tuple[tuple[str, float], ...]
    status: str
    shadow_only: bool
    question_type: str
    temperature: float
    key_digest: str
    artifact_id: str
    p_selected_acceptable_value: float | None = None

    def __post_init__(self) -> None:
        if self.status not in (CALIBRATED, UNCALIBRATED):
            raise CalibrationError("distribution status is invalid")
        if self.shadow_only != (self.status != CALIBRATED):
            raise CalibrationError("distribution shadow_only must follow status")
        question_type = _required_text(self.question_type, "distribution.question_type").upper()
        if question_type not in _QUESTION_TYPES:
            raise CalibrationError("distribution.question_type is invalid")
        object.__setattr__(self, "question_type", question_type)
        _positive(self.temperature, "distribution.temperature")
        _required_text(self.key_digest, "distribution.key_digest")
        _required_text(self.artifact_id, "distribution.artifact_id")
        if not self.probabilities_items:
            raise CalibrationError("distribution must contain a complete vector")
        if any(not isinstance(item, tuple) or len(item) != 2 for item in self.probabilities_items):
            raise CalibrationError("distribution probabilities must be label/value pairs")
        labels = _labels(tuple(item[0] for item in self.probabilities_items), "distribution.labels")
        probabilities = tuple(_finite(item[1], f"distribution.probabilities[{label}]") for label, item in zip(labels, self.probabilities_items))
        if any(value < 0.0 for value in probabilities) or not math.isclose(sum(probabilities), 1.0, rel_tol=1e-8, abs_tol=1e-8):
            raise CalibrationError("distribution probabilities must be non-negative and sum to one")
        if self.p_selected_acceptable_value is not None:
            acceptable = _finite(self.p_selected_acceptable_value, "distribution.p_selected_acceptable")
            if not 0.0 <= acceptable <= 1.0:
                raise CalibrationError("distribution.p_selected_acceptable must be in [0, 1]")

    @property
    def probabilities(self) -> dict[str, float]:
        return dict(self.probabilities_items)

    @property
    def p_selected_acceptable(self) -> float | None:
        return self.p_selected_acceptable_value

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "probabilities": self.probabilities,
            "status": self.status,
            "shadow_only": self.shadow_only,
            "question_type": self.question_type,
            "temperature": self.temperature,
            "key_digest": self.key_digest,
            "artifact_id": self.artifact_id,
        }
        if self.p_selected_acceptable_value is not None:
            result["p_selected_acceptable"] = self.p_selected_acceptable_value
        return result


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    schema: str
    version: str
    key_digest: str
    artifact_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "schema": self.schema,
            "version": self.version,
            "key_digest": self.key_digest,
            "artifact_id": self.artifact_id,
        }


def _isotonic_from_mapping(value: object) -> IsotonicCalibrator | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise CalibrationIdentityError("isotonic artifact must be a mapping")
    required = ("kind", "status", "thresholds", "values", "label_count", "independent_label_count")
    if any(name not in value for name in required):
        raise CalibrationIdentityError("isotonic artifact fields are incomplete")
    try:
        thresholds = tuple(float(item) for item in value["thresholds"])
        values = tuple(float(item) for item in value["values"])
        return IsotonicCalibrator(
            value["kind"], value["status"], thresholds, values,
            value["label_count"], value["independent_label_count"], value.get("tail_policy", "constant"),
        )
    except (CalibrationError, TypeError, ValueError) as exc:
        raise CalibrationIdentityError("isotonic artifact fields are invalid") from exc


def _ordinal_from_mapping(value: object) -> OrdinalTailCalibrator | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise CalibrationIdentityError("ordinal artifact must be a mapping")
    required = ("level_ids", "maps", "status", "heldout_baseline_rps", "heldout_calibrated_rps")
    if any(name not in value for name in required):
        raise CalibrationIdentityError("ordinal artifact fields are incomplete")
    maps_value = value.get("maps")
    if not isinstance(maps_value, Sequence) or isinstance(maps_value, (str, bytes, bytearray)):
        raise CalibrationIdentityError("ordinal artifact maps are invalid")
    level_value = value["level_ids"]
    if not isinstance(level_value, Sequence) or isinstance(level_value, (str, bytes, bytearray)):
        raise CalibrationIdentityError("ordinal artifact level IDs are invalid")
    maps = tuple(_isotonic_from_mapping(item) for item in maps_value)
    if any(item is None for item in maps):
        raise CalibrationIdentityError("ordinal artifact contains an empty map")
    try:
        return OrdinalTailCalibrator(
            tuple(level_value), tuple(item for item in maps if item is not None),
            value["status"], value["heldout_baseline_rps"], value["heldout_calibrated_rps"],
        )
    except (CalibrationError, TypeError, ValueError) as exc:
        raise CalibrationIdentityError("ordinal artifact fields are invalid") from exc


def _definition_ids(definitions: object, field: str) -> tuple[str, ...]:
    if not isinstance(definitions, tuple) or not definitions:
        return ()
    result: list[str] = []
    for item in definitions:
        thawed = _thaw(item)
        if isinstance(thawed, str):
            identifier = thawed
        elif isinstance(thawed, Mapping):
            found = False
            identifier = None
            for name in ("id", "option_id", "level_id", "level", "semantic_id"):
                if name in thawed:
                    identifier = thawed[name]
                    found = True
                    break
            if not found:
                return ()
        else:
            return ()
        if not isinstance(identifier, str) or not identifier.strip():
            raise CalibrationError(f"{field} semantic IDs must be non-empty strings")
        if identifier != identifier.strip() or "\x00" in identifier:
            raise CalibrationError(f"{field} semantic IDs must be canonical strings")
        result.append(identifier)
    if len(set(result)) != len(result):
        raise CalibrationError(f"{field} contains duplicate semantic IDs")
    return _labels(result, field)


def _level_ids_from_key(key: CalibrationKey) -> tuple[str, ...]:
    return _definition_ids(key.levels, "key.levels") or _definition_ids(key.menu, "key.menu")


@dataclass(frozen=True, slots=True)
class CalibrationArtifact:
    key: CalibrationKey
    question_type: str
    status: str
    shadow_only: bool
    label_count: int
    independent_label_count: int
    support: SupportRequirement
    temperature_fit: TemperatureFit
    binary_isotonic: IsotonicCalibrator | None = None
    binary_positive_label: str | None = None
    selected_acceptability: IsotonicCalibrator | None = None
    ordinal_tails: OrdinalTailCalibrator | None = None
    artifact_version: str = "1"

    def __post_init__(self) -> None:
        question_type = _required_text(self.question_type, "artifact.question_type").upper()
        if question_type not in _QUESTION_TYPES:
            raise CalibrationError("artifact.question_type is invalid")
        object.__setattr__(self, "question_type", question_type)
        if self.status not in (CALIBRATED, UNCALIBRATED):
            raise CalibrationError("artifact status is invalid")
        if self.shadow_only != (self.status != CALIBRATED):
            raise CalibrationError("shadow_only must follow calibration status")
        _nonnegative_int(self.label_count, "artifact.label_count")
        _nonnegative_int(self.independent_label_count, "artifact.independent_label_count")
        _required_text(self.artifact_version, "artifact.artifact_version")
        if self.binary_isotonic is not None and self.status != CALIBRATED:
            raise CalibrationError("shadow artifacts cannot contain active binary isotonic calibration")
        if self.binary_isotonic is not None and self.question_type != "BOOLEAN":
            raise CalibrationError("binary isotonic calibration requires a BOOLEAN artifact")
        if self.selected_acceptability is not None and self.status != CALIBRATED:
            raise CalibrationError("shadow artifacts cannot contain active acceptability calibration")
        if self.selected_acceptability is not None and self.question_type != "CHOICE":
            raise CalibrationError("selected acceptability calibration requires a CHOICE artifact")
        if self.ordinal_tails is not None and self.status != CALIBRATED:
            raise CalibrationError("shadow artifacts cannot contain active ordinal tails")
        if self.ordinal_tails is not None and self.question_type != "SCORE":
            raise CalibrationError("ordinal tails require a SCORE artifact")
        for name, calibrator in (
            ("binary_isotonic", self.binary_isotonic),
            ("selected_acceptability", self.selected_acceptability),
        ):
            if calibrator is not None and not calibrator.qualified:
                raise CalibrationError(f"{name} must be qualified before being stored")
        if self.binary_isotonic is not None and not self.binary_positive_label:
            raise CalibrationError("binary positive label is required with binary isotonic calibration")
        if self.binary_isotonic is None and self.binary_positive_label is not None:
            raise CalibrationError("binary positive label requires binary isotonic calibration")
        if self.binary_positive_label is not None:
            _required_text(self.binary_positive_label, "artifact.binary_positive_label")

    @property
    def key_digest(self) -> str:
        return self.key.digest

    @property
    def identity(self) -> ArtifactIdentity:
        return ArtifactIdentity(CALIBRATION_SCHEMA, self.artifact_version, self.key_digest, self.artifact_id)

    @property
    def artifact_id(self) -> str:
        return _digest(self._body())

    def _body(self) -> dict[str, object]:
        return {
            "schema": CALIBRATION_SCHEMA,
            "version": self.artifact_version,
            "key": self.key.to_dict(),
            "key_digest": self.key_digest,
            "question_type": self.question_type,
            "status": self.status,
            "shadow_only": self.shadow_only,
            "label_count": self.label_count,
            "independent_label_count": self.independent_label_count,
            "support": self.support.to_dict(),
            "temperature_fit": self.temperature_fit.to_dict(),
            "binary_isotonic": None if self.binary_isotonic is None else self.binary_isotonic.to_dict(),
            "binary_positive_label": self.binary_positive_label,
            "selected_acceptability": None if self.selected_acceptability is None else self.selected_acceptability.to_dict(),
            "ordinal_tails": None if self.ordinal_tails is None else self.ordinal_tails.to_dict(),
        }

    def to_dict(self) -> dict[str, object]:
        body = self._body()
        body["artifact_id"] = self.artifact_id
        return body

    def assert_compatible(self, key: CalibrationKey, artifact_id: str | None = None) -> None:
        if key.digest != self.key_digest:
            raise CalibrationIdentityError("calibration key digest does not match artifact")
        if artifact_id is not None and artifact_id != self.artifact_id:
            raise CalibrationIdentityError("calibration artifact identity does not match")

    def apply(
        self, scores: Mapping[str, float] | Sequence[float], *, key: CalibrationKey | None = None,
        labels: Sequence[str] | None = None, selected: str | None = None,
    ) -> CalibratedDistribution:
        if key is not None:
            self.assert_compatible(key)
        declared = _level_ids_from_key(self.key)
        expected = _labels(labels, "labels") if labels is not None else declared
        if declared and set(expected) != set(declared):
            raise CalibrationIdentityError("score labels do not match the artifact menu/levels")
        if declared and isinstance(scores, Mapping):
            try:
                actual = _labels(tuple(scores), "score labels")
            except CalibrationError as exc:
                raise CalibrationIdentityError("score labels do not match the artifact menu/levels") from exc
            if set(actual) != set(declared):
                raise CalibrationIdentityError("score labels do not match the artifact menu/levels")
        raw = normalize_complete_vector(scores, expected_labels=expected or None)
        scaled = apply_temperature(scores, self.temperature_fit.temperature, labels=list(raw))
        if self.ordinal_tails is not None:
            scaled = self.ordinal_tails.apply(scaled)
        if self.question_type.upper() == "BOOLEAN":
            if len(scaled) != 2:
                raise CalibrationError("BOOLEAN artifact requires exactly two complete labels")
            names = list(scaled)
            if self.binary_isotonic is not None:
                if self.binary_positive_label not in scaled:
                    raise CalibrationError("binary positive label is absent from the vector")
                positive = self.binary_isotonic.predict(scaled[self.binary_positive_label])
                scaled[self.binary_positive_label] = positive
                other = names[0] if names[1] == self.binary_positive_label else names[1]
                scaled[other] = 1.0 - positive
            else:
                positive = scaled[names[0]]
                scaled[names[1]] = 1.0 - positive
        acceptable: float | None = None
        if self.selected_acceptability is not None:
            choice = selected or max(scaled, key=scaled.get)
            if choice not in scaled:
                raise CalibrationError("selected option is absent from the complete vector")
            acceptable = self.selected_acceptability.predict(scaled[choice])
        return CalibratedDistribution(
            tuple((label, scaled[label]) for label in scaled),
            self.status,
            self.shadow_only,
            self.question_type,
            self.temperature_fit.temperature,
            self.key_digest,
            self.artifact_id,
            acceptable,
        )

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object], *, expected_artifact_id: str | None = None,
    ) -> "CalibrationArtifact":
        if not isinstance(value, Mapping):
            raise CalibrationIdentityError("artifact must be a mapping")
        data = dict(value)
        supplied_id = data.pop("artifact_id", None)
        if not isinstance(supplied_id, str):
            raise CalibrationIdentityError("artifact_id is required")
        if expected_artifact_id is not None and supplied_id != expected_artifact_id:
            raise CalibrationIdentityError("artifact_id does not match the expected release identity")
        if _digest(data) != supplied_id:
            raise CalibrationIdentityError("artifact identity digest mismatch")
        if data.get("schema") != CALIBRATION_SCHEMA:
            raise CalibrationIdentityError("artifact schema mismatch")
        key_value = data.get("key")
        key = CalibrationKey.from_mapping(key_value) if isinstance(key_value, Mapping) else None
        if key is None or data.get("key_digest") != key.digest:
            raise CalibrationIdentityError("artifact key identity mismatch")
        support_value = data.get("support")
        temp_value = data.get("temperature_fit")
        if not isinstance(support_value, Mapping) or not isinstance(temp_value, Mapping):
            raise CalibrationIdentityError("artifact fit metadata is incomplete")
        required_support = ("minimum_labels", "minimum_per_class", "minimum_independent_labels")
        required_fit = ("temperature", "lower", "upper", "nll", "label_count")
        required_artifact = (
            "question_type", "status", "shadow_only", "label_count", "independent_label_count", "version",
        )
        if any(name not in support_value for name in required_support) or any(name not in temp_value for name in required_fit):
            raise CalibrationIdentityError("artifact fit metadata is incomplete")
        if any(name not in data for name in required_artifact) or not isinstance(data["shadow_only"], bool):
            raise CalibrationIdentityError("artifact fields are incomplete or mistyped")
        binary_positive_label = data.get("binary_positive_label")
        if binary_positive_label is not None and not isinstance(binary_positive_label, str):
            raise CalibrationIdentityError("binary_positive_label is invalid")
        try:
            support = SupportRequirement(**{name: support_value[name] for name in required_support})
            fit = TemperatureFit(
                temp_value["temperature"], temp_value["lower"], temp_value["upper"],
                temp_value.get("boundary"), temp_value["nll"], temp_value["label_count"],
            )
            artifact = cls(
                key=key,
                question_type=data["question_type"],
                status=data["status"],
                shadow_only=data["shadow_only"],
                label_count=data["label_count"],
                independent_label_count=data["independent_label_count"],
                support=support,
                temperature_fit=fit,
                artifact_version=data["version"],
                binary_isotonic=_isotonic_from_mapping(data.get("binary_isotonic")),
                binary_positive_label=binary_positive_label,
                selected_acceptability=_isotonic_from_mapping(data.get("selected_acceptability")),
                ordinal_tails=_ordinal_from_mapping(data.get("ordinal_tails")),
            )
        except (CalibrationError, TypeError, ValueError) as exc:
            raise CalibrationIdentityError("artifact fields are invalid") from exc
        if artifact.artifact_id != supplied_id:
            raise CalibrationIdentityError("artifact identity does not match decoded artifact")
        return artifact


def fit_calibration_artifact(
    key: CalibrationKey,
    records: Sequence[object],
    *,
    question_type: str,
    support: SupportRequirement,
    independent_label_count: int | None = None,
    temperature_bounds: tuple[float, float] = (0.05, 20.0),
    binary_isotonic_samples: Sequence[object] | None = None,
    binary_positive_label: str | None = None,
    selected_acceptability_samples: Sequence[object] | None = None,
    selected_acceptability_support: SupportRequirement | None = None,
    ordinal_training_records: Sequence[Mapping[str, object]] | None = None,
    ordinal_holdout_records: Sequence[Mapping[str, object]] | None = None,
    ordinal_support: SupportRequirement | None = None,
    ordinal_independent_label_count: int | None = None,
    ordinal_holdout_independent_label_count: int | None = None,
) -> CalibrationArtifact:
    """Fit the numeric artifact; policy and evaluation remain external."""

    if not isinstance(key, CalibrationKey):
        raise CalibrationError("key must be a CalibrationKey")
    question_type = _required_text(question_type, "question_type").upper()
    if question_type not in _QUESTION_TYPES:
        raise CalibrationError("question_type must be CHOICE, BOOLEAN, or SCORE")
    rows, labels = _training_rows(records)
    declared_labels = _level_ids_from_key(key)
    if declared_labels and set(declared_labels) != set(labels):
        raise CalibrationError("training vectors do not match the key's complete menu/levels")
    targets = [target for _, target, _ in rows]
    support_report = _support_report(targets, support, labels, independent_label_count)
    if support_report.supported:
        temperature_fit = fit_temperature_report(records, bounds=temperature_bounds)
        status = CALIBRATED
        binary = None
        selected_map = None
        ordinal = None
        if binary_isotonic_samples is not None:
            if binary_positive_label is None:
                raise CalibrationError("binary_positive_label is required for binary isotonic calibration")
            binary = fit_binary_isotonic(binary_isotonic_samples, support=support, kind="p_true")
            if not binary.qualified:
                binary = None
        if selected_acceptability_samples is not None:
            if selected_acceptability_support is None:
                raise CalibrationError("selected acceptability support must be declared")
            selected_map = fit_selected_acceptability(
                selected_acceptability_samples, support=selected_acceptability_support,
            )
            if not selected_map.qualified:
                selected_map = None
        if ordinal_training_records is not None or ordinal_holdout_records is not None:
            if ordinal_training_records is None or ordinal_holdout_records is None or ordinal_support is None:
                raise CalibrationError("ordinal tails require training, unseen test set, and support declarations")
            level_ids = _level_ids_from_key(key)
            if not level_ids:
                raise CalibrationError("ordinal tails require level ids in the calibration key")
            ordinal = fit_ordinal_tails(
                ordinal_training_records, ordinal_holdout_records, levels=level_ids,
                support=ordinal_support, independent_label_count=ordinal_independent_label_count,
                heldout_independent_label_count=ordinal_holdout_independent_label_count,
            )
        return CalibrationArtifact(
            key, question_type, status, False, len(rows), support_report.independent_count,
            support, temperature_fit, binary, binary_positive_label if binary is not None else None,
            selected_map, ordinal,
        )
    lower = _positive(temperature_bounds[0], "temperature lower bound")
    upper = _positive(temperature_bounds[1], "temperature upper bound")
    if lower >= upper:
        raise CalibrationError("temperature bounds must be ordered")
    shadow_temperature = min(max(1.0, lower), upper)
    shadow_fit = TemperatureFit(shadow_temperature, lower, upper, None, 0.0, len(rows))
    return CalibrationArtifact(
        key, question_type, UNCALIBRATED, True, len(rows), support_report.independent_count,
        support, shadow_fit,
    )


# Explicitly named alias for callers that use the report's per-site wording.
fit_site_calibration = fit_calibration_artifact


__all__ = [
    "ArtifactIdentity", "CALIBRATED", "CALIBRATION_SCHEMA", "CalibrationArtifact",
    "CalibrationError", "CalibrationIdentityError", "CalibrationKey", "CalibratedDistribution",
    "IsotonicCalibrator", "OrdinalTailCalibrator", "SHADOW_ONLY", "SupportReport",
    "SupportRequirement", "TemperatureFit", "UNCALIBRATED", "apply_temperature",
    "expected_rank", "fit_binary_isotonic", "fit_calibration_artifact", "fit_ordinal_tails",
    "fit_selected_acceptability", "fit_site_calibration", "fit_temperature",
    "fit_temperature_report", "logsumexp", "normalize_complete_vector",
    "ordinal_tail_probabilities",
]
