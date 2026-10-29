"""Per-site calibration and cost-aware thresholding for System One decisions.

This module is deliberately standalone: it uses only the Python standard
library and does not import Frame Lab, runtime, or evaluator code.  A labelled
record may contain a full probability distribution (the preferred form) or a
scalar ``raw_p`` plus ``correct`` for compact replays.

The public boundary is :func:`fit_site`.  Sites with fewer than
``MIN_LABELS`` labelled outcomes remain shadow-only, retain raw probabilities,
and always report ``UNCALIBRATED_PROPOSE`` when evaluated.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path


MIN_LABELS = 50
DEFAULT_TEMPERATURE_BOUNDS = (0.05, 20.0)
DEFAULT_TREND_WINDOW = 100
UNCALIBRATED = "UNCALIBRATED"
CALIBRATED = "CALIBRATED"
UNCALIBRATED_PROPOSE = "UNCALIBRATED_PROPOSE"
PROPOSE = "PROPOSE"
ABSTAIN = "ABSTAIN"
_ABSTAIN_IDS = {"ABSTAIN", "abstain"}
_PROBABILITY_EPSILON = 1e-300


@dataclass(frozen=True)
class CostPair:
    """Costs used to compare acting with falling back.

    ``wrong_act`` is the cost of taking the selected action when its label is
    wrong.  ``fallback`` is the cost of abstaining and using the site's
    fallback mechanism.  Correct actions have zero incremental cost.
    """

    wrong_act: float
    fallback: float

    def __post_init__(self) -> None:
        _finite_nonnegative(self.wrong_act, "wrong_act", strictly_positive=True)
        _finite_nonnegative(self.fallback, "fallback")

    @property
    def cost_threshold(self) -> float:
        """Return the probability at which acting beats fallback in expectation."""

        return _clamp01(1.0 - (self.fallback / self.wrong_act))

    @property
    def wrong_act_cost(self) -> float:
        """Compatibility name for JSON/configuration callers."""

        return self.wrong_act

    @property
    def fallback_cost(self) -> float:
        """Compatibility name for JSON/configuration callers."""

        return self.fallback

    def to_dict(self) -> dict[str, float]:
        return {
            "wrong_act": self.wrong_act,
            "fallback": self.fallback,
            "cost_threshold": self.cost_threshold,
        }


@dataclass(frozen=True)
class LabelledDecision:
    """Canonical labelled input accepted by the calibration store."""

    site_id: str
    probabilities: Mapping[str, float]
    label: str
    decision_id: str = ""
    prediction: str | None = None
    sequence: int | float | str | None = None


@dataclass(frozen=True)
class SiteCalibration:
    """Fitted policy and current metrics for one decision site."""

    site_id: str
    temperature: float
    tau: float
    cost_pair: CostPair
    label_count: int
    status: str
    authoritative: bool
    shadow_only: bool
    metrics: Mapping[str, float | None] = field(default_factory=dict)

    @property
    def T_site(self) -> float:
        """Spec spelling for the fitted site temperature."""

        return self.temperature

    @property
    def tau_site(self) -> float:
        """Spec spelling for the fitted site threshold."""

        return self.tau

    @property
    def calibration_status(self) -> str:
        return self.status

    @property
    def is_authoritative(self) -> bool:
        return self.authoritative

    def evaluate(
        self,
        probabilities: Mapping[str, float],
        *,
        selected: str | None = None,
        legal_mass: float = 1.0,
    ) -> dict[str, object]:
        """Score one new distribution without granting authority to shadow sites."""

        raw = _normalise_probabilities(probabilities)
        if not isinstance(legal_mass, (int, float)) or isinstance(legal_mass, bool):
            raise ValueError("legal_mass must be a finite number in [0, 1]")
        if not math.isfinite(float(legal_mass)) or not 0.0 <= float(legal_mass) <= 1.0:
            raise ValueError("legal_mass must be a finite number in [0, 1]")
        calibrated = calibrate_probabilities(raw, self.temperature)
        choice = selected if selected is not None else _argmax(calibrated)
        if choice not in calibrated:
            raise ValueError("selected option is absent from probabilities")
        raw_p = raw[choice]
        calibrated_p = calibrated[choice]

        if not self.authoritative:
            disposition = UNCALIBRATED_PROPOSE
        elif float(legal_mass) < 0.9 or _is_abstain(choice):
            disposition = ABSTAIN
        elif calibrated_p >= self.tau:
            disposition = PROPOSE
        else:
            disposition = ABSTAIN
        return {
            "raw_p": raw_p,
            "calibrated_p": calibrated_p,
            "T_site": self.temperature,
            "tau_site": self.tau,
            "selected": choice,
            "legal_mass": float(legal_mass),
            "disposition": disposition,
            "status": self.status,
            "authoritative": self.authoritative,
            "shadow_only": self.shadow_only,
        }

    def to_dict(self) -> dict[str, object]:
        metrics = {key: self.metrics[key] for key in sorted(self.metrics)}
        return {
            "site_id": self.site_id,
            "label_count": self.label_count,
            "status": self.status,
            "authoritative": self.authoritative,
            "shadow_only": self.shadow_only,
            "temperature": self.temperature,
            "T_site": self.temperature,
            "tau": self.tau,
            "tau_site": self.tau,
            "cost_pair": self.cost_pair.to_dict(),
            "cost_threshold": self.cost_pair.cost_threshold,
            "metrics": metrics,
        }


@dataclass(frozen=True)
class _Observation:
    site_id: str
    probabilities: dict[str, float]
    label: str
    prediction: str | None
    decision_id: str
    sequence: int | float | str | None


def fit_temperature(
    records: Sequence[object],
    labels: Sequence[object] | None = None,
    *,
    bounds: tuple[float, float] = DEFAULT_TEMPERATURE_BOUNDS,
) -> float:
    """Fit one temperature by minimizing labelled multiclass log loss.

    ``records`` normally contains mappings or :class:`LabelledDecision`
    objects.  For compact numerical replays, callers may pass a sequence of
    probability mappings (or numeric rows) and the matching ``labels``.
    """

    observations = _training_observations(records, labels)
    if not observations:
        raise ValueError("at least one labelled record is required")
    lower, upper = bounds
    if (
        not isinstance(lower, (int, float))
        or isinstance(lower, bool)
        or not isinstance(upper, (int, float))
        or isinstance(upper, bool)
        or not math.isfinite(float(lower))
        or not math.isfinite(float(upper))
        or lower <= 0
        or lower >= upper
    ):
        raise ValueError("temperature bounds must be finite, positive, and ordered")

    log_lower = math.log(float(lower))
    log_upper = math.log(float(upper))
    grid_size = 128
    grid = [
        log_lower + (log_upper - log_lower) * index / grid_size
        for index in range(grid_size + 1)
    ]
    values = [_negative_log_likelihood(observations, math.exp(value)) for value in grid]
    best_index = min(range(len(values)), key=values.__getitem__)
    if best_index == 0:
        return float(lower)
    if best_index == grid_size:
        return float(upper)

    left = grid[best_index - 1]
    right = grid[best_index + 1]
    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
    for _ in range(96):
        first = right - (right - left) / golden_ratio
        second = left + (right - left) / golden_ratio
        first_value = _negative_log_likelihood(observations, math.exp(first))
        second_value = _negative_log_likelihood(observations, math.exp(second))
        if first_value < second_value:
            right = second
        else:
            left = first
    return math.exp((left + right) / 2.0)


def calibrate_probabilities(
    probabilities: Mapping[str, float], temperature: float
) -> dict[str, float]:
    """Apply temperature scaling and return a normalized distribution."""

    if (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or not math.isfinite(float(temperature))
        or temperature <= 0
    ):
        raise ValueError("temperature must be a finite positive number")
    raw = _normalise_probabilities(probabilities)
    logits = {
        key: math.log(max(value, _PROBABILITY_EPSILON)) / float(temperature)
        for key, value in raw.items()
    }
    maximum = max(logits.values())
    weights = {key: math.exp(value - maximum) for key, value in logits.items()}
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def choose_tau(
    records: Sequence[object],
    cost_pair: CostPair | Mapping[str, object],
    *,
    temperature: float = 1.0,
) -> float:
    """Choose a labelled threshold while respecting the cost break-even floor.

    Candidate thresholds are the cost break-even plus observed calibrated
    scores.  Empirical ties select the higher threshold, which keeps the
    policy conservative without ever allowing a threshold below the stated
    cost pair.
    """

    pair = _coerce_cost_pair(cost_pair)
    observations = _coerce_observations(records)
    floor = pair.cost_threshold
    if not observations:
        return 1.0

    scored: list[tuple[float, bool, bool]] = []
    for observation in observations:
        calibrated = calibrate_probabilities(observation.probabilities, temperature)
        selected = observation.prediction or _argmax(calibrated)
        if selected not in calibrated:
            raise ValueError("prediction is absent from probabilities")
        scored.append(
            (calibrated[selected], selected == observation.label, not _is_abstain(selected))
        )

    candidates = {floor, 1.0}
    for score, _, actionable in scored:
        if actionable and score >= floor:
            candidates.add(score)
            if score < 1.0:
                candidates.add(math.nextafter(score, 1.0))

    def realised_cost(threshold: float) -> float:
        total = 0.0
        for score, correct, actionable in scored:
            acts = actionable and score >= threshold
            total += pair.wrong_act if acts and not correct else 0.0
            total += pair.fallback if not acts else 0.0
        return total

    return min(sorted(candidates), key=lambda threshold: (realised_cost(threshold), -threshold))


def fit_site(
    site_id: str,
    records: Sequence[object],
    cost_pair: CostPair | Mapping[str, object],
    *,
    min_labels: int = MIN_LABELS,
) -> SiteCalibration:
    """Fit one site's temperature and threshold, preserving shadow semantics."""

    if not isinstance(site_id, str) or not site_id.strip():
        raise ValueError("site_id must be a nonempty string")
    if type(min_labels) is not int or min_labels <= 0:
        raise ValueError("min_labels must be a positive integer")
    required_labels = max(MIN_LABELS, min_labels)
    pair = _coerce_cost_pair(cost_pair)
    observations = _coerce_observations(records, default_site_id=site_id)
    if any(observation.site_id != site_id for observation in observations):
        raise ValueError("fit_site received records for another site")
    label_count = len(observations)
    authoritative = label_count >= required_labels
    if authoritative:
        temperature = fit_temperature(observations)
        tau = choose_tau(observations, pair, temperature=temperature)
        status = CALIBRATED
    else:
        # Raw p is retained by contract until the site has enough labels.
        temperature = 1.0
        tau = 1.0
        status = UNCALIBRATED
    metrics = _metrics(observations, temperature=temperature, tau=tau)
    return SiteCalibration(
        site_id=site_id,
        temperature=temperature,
        tau=tau,
        cost_pair=pair,
        label_count=label_count,
        status=status,
        authoritative=authoritative,
        shadow_only=not authoritative,
        metrics=metrics,
    )


def export_trends(
    records: Sequence[object] | Mapping[str, Sequence[object]],
    cost_pairs: CostPair | Mapping[str, CostPair | Mapping[str, object]],
    *,
    min_labels: int = MIN_LABELS,
    window: int = DEFAULT_TREND_WINDOW,
) -> dict[str, object]:
    """Build deterministic per-site metrics for the Frame Lab UI."""

    if type(min_labels) is not int or min_labels <= 0:
        raise ValueError("min_labels must be a positive integer")
    required_labels = max(MIN_LABELS, min_labels)
    if type(window) is not int or window <= 0:
        raise ValueError("window must be a positive integer")
    grouped = _group_records(records)
    sites: list[dict[str, object]] = []
    for site_id in sorted(grouped):
        pair = _cost_pair_for_site(cost_pairs, site_id)
        observations = grouped[site_id]
        calibration = fit_site(site_id, observations, pair, min_labels=min_labels)
        trend: list[dict[str, object]] = []
        for end in range(1, len(observations) + 1):
            sample = observations[max(0, end - window) : end]
            point_temperature = calibration.temperature if end >= required_labels else 1.0
            point_tau = calibration.tau if end >= required_labels else 1.0
            point = _metrics(sample, temperature=point_temperature, tau=point_tau)
            point["label_count"] = end
            point["temperature"] = point_temperature
            point["tau"] = point_tau
            point["status"] = calibration.status if end >= required_labels else UNCALIBRATED
            point["authoritative"] = calibration.authoritative and end >= required_labels
            trend.append(point)
        site_payload = calibration.to_dict()
        site_payload["trend"] = trend
        site_payload["latest"] = trend[-1] if trend else _metrics([], temperature=1.0, tau=1.0)
        sites.append(site_payload)
    return {
        "schema_version": "decision-calibration/v1",
        "min_labels": required_labels,
        "trend_window": window,
        "sites": sites,
    }


def export_trends_json(
    records: Sequence[object] | Mapping[str, Sequence[object]],
    cost_pairs: CostPair | Mapping[str, CostPair | Mapping[str, object]],
    *,
    destination: str | Path | None = None,
    min_labels: int = MIN_LABELS,
    window: int = DEFAULT_TREND_WINDOW,
) -> str:
    """Serialize :func:`export_trends`; optionally write the UI artifact."""

    payload = export_trends(records, cost_pairs, min_labels=min_labels, window=window)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    if destination is not None:
        Path(destination).write_text(encoded, encoding="utf-8")
    return encoded


# Singular aliases keep callers from depending on the plural export wording.
export_trend = export_trends
export_trend_json = export_trends_json


def _training_observations(
    records: Sequence[object], labels: Sequence[object] | None
) -> list[_Observation]:
    if labels is None:
        return _coerce_observations(records)
    if len(records) != len(labels):
        raise ValueError("records and labels must have equal lengths")
    result: list[_Observation] = []
    for index, (raw_probabilities, raw_label) in enumerate(zip(records, labels)):
        if isinstance(raw_probabilities, (int, float)) and not isinstance(raw_probabilities, bool):
            probability = _probability_scalar(raw_probabilities, "raw_p")
            probabilities = {"selected": probability, "__other__": 1.0 - probability}
            if isinstance(raw_label, bool):
                label = "selected" if raw_label else "__other__"
            else:
                label = _label_from_row(raw_label, probabilities)
        else:
            probabilities = _probability_row(raw_probabilities)
            label = _label_from_row(raw_label, probabilities)
        result.append(_Observation("", probabilities, label, None, str(index), index))
    return result


def _probability_row(value: object) -> dict[str, float]:
    if isinstance(value, Mapping):
        return _normalise_probabilities(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _normalise_probabilities({str(index): item for index, item in enumerate(value)})
    raise ValueError("probability row must be a mapping or numeric sequence")


def _label_from_row(value: object, probabilities: Mapping[str, float]) -> str:
    if isinstance(value, bool):
        raise ValueError("boolean labels require a scalar raw_p/correct record")
    if isinstance(value, int) and not isinstance(value, bool):
        candidate = str(value)
    elif isinstance(value, str):
        candidate = value
    else:
        raise ValueError("labels must be option ids or integer option indexes")
    if candidate not in probabilities:
        raise ValueError("label is absent from probability row")
    return candidate


def _coerce_observations(
    records: Sequence[object], *, default_site_id: str = ""
) -> list[_Observation]:
    if isinstance(records, (str, bytes, bytearray)):
        raise ValueError("records must be a sequence of labelled records")
    result: list[_Observation] = []
    for index, record in enumerate(records):
        result.append(_coerce_observation(record, default_site_id=default_site_id, index=index))
    return result


def _coerce_observation(record: object, *, default_site_id: str, index: int) -> _Observation:
    if isinstance(record, _Observation):
        site_id = record.site_id or default_site_id
        return _Observation(
            site_id,
            dict(record.probabilities),
            record.label,
            record.prediction,
            record.decision_id or str(index),
            record.sequence,
        )
    if isinstance(record, LabelledDecision):
        site_id = record.site_id or default_site_id
        return _Observation(
            site_id,
            _normalise_probabilities(record.probabilities),
            _label_from_row(record.label, record.probabilities),
            record.prediction,
            record.decision_id or str(index),
            record.sequence,
        )
    if not isinstance(record, Mapping):
        raise ValueError("each labelled record must be a mapping or LabelledDecision")

    site_value = record.get("site_id", record.get("site", default_site_id))
    site_id = site_value if isinstance(site_value, str) else default_site_id
    if not site_id and default_site_id:
        site_id = default_site_id
    decision_id = record.get("decision_id", record.get("id", str(index)))
    decision_text = decision_id if isinstance(decision_id, str) else str(decision_id)
    prediction = _first_string(record, ("prediction", "predicted", "answer", "choice"))
    sequence = record.get("sequence", record.get("labelled_at", record.get("timestamp")))

    raw_probabilities = _first_present(
        record, ("probabilities", "raw_probabilities", "probs", "distribution", "letter_mass")
    )
    if raw_probabilities is not None:
        probabilities = _normalise_probabilities(raw_probabilities)
        raw_label = _first_present(record, ("label", "truth", "outcome", "correct_label", "y"))
        if isinstance(raw_label, bool):
            correct = raw_label
            selected = prediction or _argmax(probabilities)
            raw_label = selected if correct else _first_other(probabilities, selected)
        label = _label_from_row(raw_label, probabilities)
        return _Observation(site_id, probabilities, label, prediction, decision_text, sequence)

    raw_probability = _first_present(
        record, ("raw_p", "raw_probability", "p", "probability", "confidence", "score")
    )
    if raw_probability is None:
        raise ValueError("record needs probabilities or raw_p")
    probability = _probability_scalar(raw_probability, "raw_p")
    selected = prediction or "selected"
    correct_value = _first_present(record, ("correct", "is_correct"))
    raw_label = _first_present(record, ("label", "truth", "outcome", "y"))
    if isinstance(raw_label, bool) and correct_value is None:
        correct_value = raw_label
    elif type(raw_label) is int and raw_label in {0, 1} and correct_value is None:
        correct_value = bool(raw_label)
    elif isinstance(raw_label, str) and raw_label.lower() in {"correct", "incorrect"} and correct_value is None:
        correct_value = raw_label.lower() == "correct"
    if not isinstance(correct_value, bool):
        raise ValueError("scalar raw_p records require boolean correct")
    probabilities = {selected: probability, "__other__": 1.0 - probability}
    label = selected if correct_value else "__other__"
    return _Observation(site_id, probabilities, label, selected, decision_text, sequence)


def _group_records(
    records: Sequence[object] | Mapping[str, Sequence[object]],
) -> dict[str, list[_Observation]]:
    grouped: dict[str, list[_Observation]] = {}
    if isinstance(records, Mapping) and not _looks_like_record(records):
        for site_id, site_records in records.items():
            if not isinstance(site_id, str) or not isinstance(site_records, Sequence):
                raise ValueError("site record groups must map string ids to sequences")
            observations = _coerce_observations(site_records, default_site_id=site_id)
            if any(item.site_id not in {"", site_id} for item in observations):
                raise ValueError("group contains a record for another site")
            grouped[site_id] = [
                _Observation(site_id, item.probabilities, item.label, item.prediction,
                             item.decision_id, item.sequence)
                for item in observations
            ]
        return grouped
    for observation in _coerce_observations(records):
        if not observation.site_id:
            raise ValueError("export records require site_id")
        grouped.setdefault(observation.site_id, []).append(observation)
    return grouped


def _looks_like_record(value: Mapping[object, object]) -> bool:
    record_keys = {
        "site_id", "site", "probabilities", "raw_probabilities", "raw_p", "raw_probability",
        "p", "probability", "confidence", "label", "truth", "correct", "is_correct",
    }
    return any(key in value for key in record_keys)


def _cost_pair_for_site(
    cost_pairs: CostPair | Mapping[str, CostPair | Mapping[str, object]], site_id: str
) -> CostPair:
    if isinstance(cost_pairs, CostPair):
        return cost_pairs
    if isinstance(cost_pairs, Mapping) and {"wrong_act", "fallback"}.issubset(cost_pairs):
        return _coerce_cost_pair(cost_pairs)
    if not isinstance(cost_pairs, Mapping) or site_id not in cost_pairs:
        raise ValueError(f"no cost pair registered for site {site_id!r}")
    return _coerce_cost_pair(cost_pairs[site_id])


def _coerce_cost_pair(value: CostPair | Mapping[str, object]) -> CostPair:
    if isinstance(value, CostPair):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("cost_pair must be CostPair or a mapping")
    wrong = value.get("wrong_act", value.get("wrong_act_cost"))
    fallback = value.get("fallback", value.get("fallback_cost"))
    if wrong is None or fallback is None:
        raise ValueError("cost_pair needs wrong_act and fallback")
    return CostPair(float(wrong), float(fallback))


def _metrics(
    observations: Sequence[_Observation], *, temperature: float, tau: float
) -> dict[str, float | None]:
    if not observations:
        return {"accuracy": None, "ece": None, "coverage": 0.0, "covered_accuracy": None}
    predictions: list[tuple[float, bool, bool]] = []
    for observation in observations:
        calibrated = calibrate_probabilities(observation.probabilities, temperature)
        selected = observation.prediction or _argmax(calibrated)
        confidence = calibrated[selected]
        predictions.append((confidence, selected == observation.label, not _is_abstain(selected)))
    accuracy = sum(correct for _, correct, _ in predictions) / len(predictions)
    ece = _ece(predictions)
    covered = [correct for confidence, correct, actionable in predictions
               if actionable and confidence >= tau]
    coverage = len(covered) / len(predictions)
    covered_accuracy = sum(covered) / len(covered) if covered else None
    return {
        "accuracy": accuracy,
        "ece": ece,
        "coverage": coverage,
        "covered_accuracy": covered_accuracy,
    }


def _ece(predictions: Sequence[tuple[float, bool, bool]]) -> float:
    total = len(predictions)
    error = 0.0
    for bucket in range(5):
        lower = bucket / 5.0
        upper = (bucket + 1) / 5.0
        members = [
            (confidence, correct)
            for confidence, correct, _ in predictions
            if lower <= confidence < upper or (bucket == 4 and confidence == 1.0)
        ]
        if members:
            mean_confidence = sum(item[0] for item in members) / len(members)
            accuracy = sum(item[1] for item in members) / len(members)
            error += len(members) / total * abs(accuracy - mean_confidence)
    return error


def _negative_log_likelihood(observations: Sequence[_Observation], temperature: float) -> float:
    total = 0.0
    for observation in observations:
        logits = {
            key: math.log(max(value, _PROBABILITY_EPSILON)) / temperature
            for key, value in observation.probabilities.items()
        }
        maximum = max(logits.values())
        log_partition = maximum + math.log(
            sum(math.exp(value - maximum) for value in logits.values())
        )
        total += log_partition - logits[observation.label]
    return total / len(observations)


def _normalise_probabilities(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("probabilities must be a nonempty mapping")
    result: dict[str, float] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("probability option ids must be nonempty strings")
        result[key] = _probability_scalar(raw, f"probability for {key}")
    total = sum(result.values())
    if total <= 0:
        raise ValueError("probabilities must have positive mass")
    return {key: probability / total for key, probability in result.items()}


def _probability_scalar(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return float(value)


def _finite_nonnegative(value: object, name: str, *, strictly_positive: bool = False) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or (value <= 0 if strictly_positive else value < 0)
    ):
        qualifier = "positive" if strictly_positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")


def _first_present(record: Mapping[str, object], names: Sequence[str]) -> object | None:
    for name in names:
        if name in record:
            return record[name]
    return None


def _first_string(record: Mapping[str, object], names: Sequence[str]) -> str | None:
    value = _first_present(record, names)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{names[0]} must be a nonempty string")
    return value


def _argmax(probabilities: Mapping[str, float]) -> str:
    return max(probabilities, key=probabilities.__getitem__)


def _first_other(probabilities: Mapping[str, float], selected: str) -> str:
    for key in probabilities:
        if key != selected:
            return key
    raise ValueError("a boolean correctness label needs at least two options")


def _is_abstain(option: str) -> bool:
    return option in _ABSTAIN_IDS or option.upper() == "ABSTAIN"


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
