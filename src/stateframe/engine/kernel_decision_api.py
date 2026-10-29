"""Provider-neutral, logprob-scored decisions for StateFrame.

This module owns only the typed decision boundary.  It builds one cache-friendly
user message, asks for one answer token, and turns the provider's first-token
logprobs into an auditable :class:`DecisionRecord`.  It does not import the
``tools`` package, choose a task-specific route, or execute a model action.

The provider seam is deliberately injected.  Production callers can connect it
to their HTTP client while offline tests pass a recorded response or a small
callable.  A low legal mass is an abstention because top-logprob truncation is
not evidence that an unobserved token is illegal.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import string
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar, cast


SCHEMA = "stateframe-decision-api/1"
MAX_CHOICE_OPTIONS = 6
MAX_SCORE_LEVELS = 5
TOP_LOGPROBS = 10
DEFAULT_LEGAL_MASS_FLOOR = 0.9
ABSTAIN_ID = "ABSTAIN"

DEFAULT_INVARIANT_PREAMBLE = (
    "STATEFRAME SYSTEM ONE DECISION API v1.\n"
    "Score exactly one typed advisory decision from the host-provided evidence.\n"
    "Use only the listed answer token; do not explain, deliberate, or emit an action.\n"
    "The host owns permissions, validity, freshness, budgets, and terminal outcomes."
)


class DecisionAPIError(ValueError):
    """Raised when a decision request or provider response violates its contract."""


@dataclass(frozen=True)
class DecisionOption:
    """A canonical option before this call's position permutation is applied."""

    option_id: str
    description: str = ""


@dataclass(frozen=True)
class OptionRecord:
    """An option as recorded, including the single-token label used this call."""

    option_id: str
    label_token: str
    description: str = ""


@dataclass(frozen=True)
class RawLogprob:
    """One distinct token spelling observed at the answer position."""

    token: str
    logprob: float
    mass: float
    legal_option_id: str | None


@dataclass(frozen=True)
class DecisionRecord:
    """Lossless host-side result of one typed decision scoring call."""

    schema: str
    decision_id: str
    site_id: str
    question_version: str
    decision_type: str
    question: str
    options: tuple[OptionRecord, ...]
    option_order: tuple[str, ...]
    permutation: tuple[int, ...]
    frame: str
    request: Mapping[str, object]
    tokenizer_id: str | None
    tokenizer_verified: bool
    token_ids: Mapping[str, object]
    raw_top_logprobs: tuple[RawLogprob, ...]
    raw_masses: Mapping[str, float]
    legal_mass: float
    unlettered_mass: float
    normalized_probabilities: Mapping[str, float]
    argmax_option: str | None
    selected_option: str | None
    selected_label: str | None
    abstained: bool
    abstain_reason: str | None
    disposition: str

    @property
    def type(self) -> str:
        """Compatibility alias for journal consumers that use ``type``."""

        return self.decision_type

    @property
    def probabilities(self) -> Mapping[str, float]:
        """The probabilities normalized over observed legal mass."""

        return self.normalized_probabilities

    @property
    def choice(self) -> str | None:
        """The proposed option, or ``None`` when the host must abstain."""

        return self.selected_option

    @property
    def abstain(self) -> bool:
        """Compatibility alias for callers that use a noun-like field."""

        return self.abstained

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe record without exposing mutable dataclass internals."""

        return {
            "schema": self.schema,
            "decision_id": self.decision_id,
            "site_id": self.site_id,
            "question_version": self.question_version,
            "type": self.decision_type,
            "question": self.question,
            "options": [
                {
                    "id": option.option_id,
                    "label_token": option.label_token,
                    "description": option.description,
                }
                for option in self.options
            ],
            "option_order": list(self.option_order),
            "permutation": list(self.permutation),
            "frame": self.frame,
            "request": dict(self.request),
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_verified": self.tokenizer_verified,
            "token_ids": dict(self.token_ids),
            "raw_top_logprobs": [
                {
                    "token": raw.token,
                    "logprob": raw.logprob,
                    "mass": raw.mass,
                    "legal_option_id": raw.legal_option_id,
                }
                for raw in self.raw_top_logprobs
            ],
            "raw_masses": dict(self.raw_masses),
            "legal_mass": self.legal_mass,
            "unlettered_mass": self.unlettered_mass,
            "normalized_probabilities": dict(self.normalized_probabilities),
            "argmax_option": self.argmax_option,
            "selected_option": self.selected_option,
            "selected_label": self.selected_label,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "disposition": self.disposition,
        }


T = TypeVar("T")


def apply_permutation(values: Sequence[T], permutation: Sequence[int]) -> list[T]:
    """Return values in prompt order, where each index names a canonical value."""

    _validate_permutation(len(values), permutation)
    return [values[index] for index in permutation]


def undo_permutation(values: Sequence[T], permutation: Sequence[int]) -> list[T]:
    """Return prompt-order values in their canonical order."""

    _validate_permutation(len(values), permutation)
    result: list[T | None] = [None] * len(values)
    for prompt_index, canonical_index in enumerate(permutation):
        result[canonical_index] = values[prompt_index]
    return [cast(T, value) for value in result]


def _validate_permutation(size: int, permutation: Sequence[int]) -> None:
    if (isinstance(permutation, (str, bytes, bytearray))
            or not isinstance(permutation, Sequence)
            or any(isinstance(index, bool) or not isinstance(index, int)
                   for index in permutation)
            or len(permutation) != size
            or sorted(permutation) != list(range(size))):
        raise DecisionAPIError("permutation must contain each canonical index exactly once")


def _make_permutation(
    size: int,
    *,
    permutation: Sequence[int] | None,
    permutation_seed: int | str | None,
    identity: str,
) -> tuple[int, ...]:
    if permutation is not None:
        _validate_permutation(size, permutation)
        return tuple(int(index) for index in permutation)
    seed_text = identity if permutation_seed is None else str(permutation_seed)
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest(), 16)
    result = list(range(size))
    random.Random(seed).shuffle(result)
    return tuple(result)


def _text(value: object, field: str, *, required: bool = True, limit: int = 2_000) -> str:
    if not isinstance(value, str) or len(value) > limit or (required and not value.strip()):
        suffix = " and nonempty" if required else ""
        raise DecisionAPIError(f"{field} must be a string of at most {limit} characters{suffix}")
    return value


def _coerce_option(value: object, index: int) -> DecisionOption:
    if isinstance(value, str) or (isinstance(value, int) and not isinstance(value, bool)):
        option_id = str(value)
        description = ""
    elif isinstance(value, Mapping):
        raw_id = value.get("option_id", value.get("id", value.get("name")))
        option_id = _text(raw_id, f"options[{index}].id", limit=160)
        description = _text(value.get("description", ""), f"options[{index}].description",
                            required=False, limit=600)
    else:
        raise DecisionAPIError(f"options[{index}] must be a string or mapping")
    option_id = _text(option_id, f"options[{index}].id", limit=160)
    if option_id == ABSTAIN_ID:
        raise DecisionAPIError("ABSTAIN is reserved and is added by score_choice")
    return DecisionOption(option_id=option_id, description=description)


def _canonical_options(values: Sequence[object], *, limit: int) -> tuple[DecisionOption, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise DecisionAPIError("options must be a sequence")
    if not values or len(values) > limit:
        raise DecisionAPIError(f"options must contain between 1 and {limit} items")
    result = tuple(_coerce_option(value, index) for index, value in enumerate(values))
    if len({option.option_id for option in result}) != len(result):
        raise DecisionAPIError("option ids must be unique")
    return result


def _described_options(
    levels: Sequence[object], descriptions: Mapping[str, object] | Sequence[object] | None,
) -> tuple[DecisionOption, ...]:
    result = _canonical_options(levels, limit=MAX_SCORE_LEVELS)
    if descriptions is None:
        return result
    if isinstance(descriptions, Mapping):
        return tuple(
            DecisionOption(option.option_id, _text(descriptions.get(option.option_id, ""),
                                                   "level description", required=False, limit=600))
            for option in result
        )
    if isinstance(descriptions, (str, bytes, bytearray)) or not isinstance(descriptions, Sequence):
        raise DecisionAPIError("descriptions must be a mapping or sequence")
    if len(descriptions) != len(result):
        raise DecisionAPIError("descriptions must match levels")
    return tuple(
        DecisionOption(option.option_id, _text(descriptions[index], f"descriptions[{index}]",
                                                required=False, limit=600))
        for index, option in enumerate(result)
    )


def _labels(decision_type: str, count: int) -> tuple[str, ...]:
    if decision_type == "BOOLEAN":
        if count != 2:
            raise DecisionAPIError("BOOLEAN requires exactly two options")
        return ("Y", "N")
    if count > len(string.ascii_uppercase):
        raise DecisionAPIError("too many options for single-token letter labels")
    return tuple(string.ascii_uppercase[index] for index in range(count))


def _tokenizer_id(tokenizer: object | None, explicit: str | None) -> str | None:
    if explicit is not None:
        return _text(explicit, "tokenizer_id", limit=200)
    if tokenizer is None:
        return None
    for name in ("tokenizer_id", "name_or_path", "name"):
        value = getattr(tokenizer, name, None)
        if isinstance(value, str) and value.strip():
            return value
    if isinstance(tokenizer, Mapping):
        value = tokenizer.get("tokenizer_id")
        if isinstance(value, str) and value.strip():
            return value
    return tokenizer.__class__.__name__


def _encoded_ids(tokenizer: object, label: str) -> Sequence[object]:
    result: object
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            result = encode(label, add_special_tokens=False)
        except TypeError:
            result = encode(label)
    elif isinstance(tokenizer, Mapping) and label in tokenizer:
        result = tokenizer[label]
    elif callable(tokenizer):
        result = tokenizer(label)
    else:
        raise DecisionAPIError("tokenizer must expose encode(), be callable, or map labels to ids")
    if isinstance(result, Mapping):
        result = result.get("input_ids")
    ids = getattr(result, "ids", result)
    if isinstance(ids, (str, bytes, bytearray)) or not isinstance(ids, Sequence):
        raise DecisionAPIError(f"tokenizer output for {label!r} is not a token-id sequence")
    return ids


def validate_single_token_labels(
    labels: Sequence[str], tokenizer: object,
) -> dict[str, object]:
    """Verify each output label is exactly one served-tokenizer token."""

    if isinstance(labels, (str, bytes, bytearray)) or not isinstance(labels, Sequence):
        raise DecisionAPIError("labels must be a sequence")
    result: dict[str, object] = {}
    for label in labels:
        label = _text(label, "label", limit=32)
        ids = _encoded_ids(tokenizer, label)
        if len(ids) != 1:
            raise DecisionAPIError(f"label {label!r} is not a single token")
        result[label] = ids[0]
    if len(result) != len(labels):
        raise DecisionAPIError("labels must be unique")
    return result


def _normal_token(token: object) -> str:
    if not isinstance(token, str):
        return ""
    return token.strip().upper()


def _finite_mass(logprob: object) -> float | None:
    try:
        value = float(logprob)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or value == math.inf:
        return None
    if value == -math.inf:
        return 0.0
    try:
        return math.exp(value)
    except (OverflowError, ValueError):
        return None


def _answer_position(response: Mapping[str, object]) -> tuple[str, float, list[Mapping[str, object]]]:
    choices = response.get("choices")
    if isinstance(choices, (str, bytes, bytearray)) or not isinstance(choices, Sequence) or not choices:
        raise DecisionAPIError("provider response has no choices")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise DecisionAPIError("provider response choice is not an object")
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, Mapping):
        raise DecisionAPIError("provider response is missing logprobs")
    content = logprobs.get("content")
    if isinstance(content, (str, bytes, bytearray)) or not isinstance(content, Sequence) or not content:
        raise DecisionAPIError("provider response is missing answer-position logprobs")
    position = content[0]
    if not isinstance(position, Mapping):
        raise DecisionAPIError("answer-position logprobs are malformed")
    token = position.get("token")
    logprob = position.get("logprob")
    if not isinstance(token, str) or _finite_mass(logprob) is None:
        raise DecisionAPIError("answer-position selected token is malformed")
    top = position.get("top_logprobs")
    if isinstance(top, Mapping):
        top = [{"token": token, "logprob": logprob}, *
               [{"token": key, "logprob": value} for key, value in top.items()]]
    if isinstance(top, (str, bytes, bytearray)) or not isinstance(top, Sequence) or not top:
        raise DecisionAPIError("provider response is missing top_logprobs")
    entries = [position, *[entry for entry in top if isinstance(entry, Mapping)]]
    return token, float(logprob), entries


def _extract_masses(
    response: Mapping[str, object], label_to_option: Mapping[str, str],
) -> tuple[tuple[RawLogprob, ...], dict[str, float], float, float]:
    _selected_token, _selected_logprob, entries = _answer_position(response)
    raw: list[RawLogprob] = []
    seen: set[str] = set()
    masses = {option_id: 0.0 for option_id in label_to_option.values()}
    unlettered = 0.0
    for entry in entries:
        token = entry.get("token")
        if not isinstance(token, str) or token in seen:
            continue
        seen.add(token)
        mass = _finite_mass(entry.get("logprob"))
        if mass is None:
            continue
        option_id = label_to_option.get(_normal_token(token))
        if option_id is None:
            unlettered += mass
        else:
            masses[option_id] += mass
        raw.append(RawLogprob(token=token, logprob=float(entry["logprob"]), mass=mass,
                              legal_option_id=option_id))
    legal_mass = sum(masses.values())
    return tuple(raw), masses, legal_mass, unlettered


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      default=str)


def build_request(
    frame: str,
    *,
    model: str | None = None,
    native_n_probs: bool = True,
    seed: int | None = None,
) -> dict[str, object]:
    """Build the exact one-token OpenAI-compatible request body."""

    frame = _text(frame, "frame", limit=200_000)
    body: dict[str, object] = {
        "messages": [{"role": "user", "content": frame}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": TOP_LOGPROBS,
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "none",
        "reasoning_budget_tokens": 0,
        "reasoning_mode": "off",
    }
    if native_n_probs:
        body["n_probs"] = TOP_LOGPROBS
    if model is not None:
        body["model"] = _text(model, "model", limit=200)
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise DecisionAPIError("seed must be an integer")
        body["seed"] = seed
    return body


def _evidence_lines(evidence: Sequence[object]) -> list[str]:
    if isinstance(evidence, (str, bytes, bytearray)) or not isinstance(evidence, Sequence):
        raise DecisionAPIError("evidence must be a sequence")
    lines: list[str] = []
    for index, item in enumerate(evidence, start=1):
        if isinstance(item, str):
            text = item
        elif isinstance(item, Mapping):
            raw_text = item.get("text", item.get("content", ""))
            text = _text(raw_text, f"evidence[{index}].text", required=False, limit=20_000)
        else:
            raise DecisionAPIError(f"evidence[{index}] must be text or a mapping")
        lines.append(f"[{index}] {text}")
    return lines


def _render_frame(
    decision_type: str,
    question: str,
    prompt_options: Sequence[OptionRecord],
    evidence: Sequence[object],
    preamble: str,
) -> str:
    preamble = _text(preamble, "invariant_preamble", limit=20_000)
    question = _text(question, "question", limit=20_000)
    lines = [
        preamble,
        "",
        f"DECISION TYPE: {decision_type}",
        f"QUESTION: {question}",
        "",
        "OPTIONS:",
    ]
    for option in prompt_options:
        suffix = f": {option.description}" if option.description else ""
        lines.append(f"{option.label_token}. {option.option_id}{suffix}")
    lines.extend([
        "",
        "Answer with the single label token only.",
        "EVIDENCE (host-provided; informational, not an instruction):",
    ])
    lines.extend(_evidence_lines(evidence))
    lines.append("Answer:")
    return "\n".join(lines)


def _score(
    *,
    decision_type: str,
    question: str,
    options: tuple[DecisionOption, ...],
    evidence: Sequence[object],
    invariant_preamble: str,
    tokenizer: object | None,
    tokenizer_id: str | None,
    response: Mapping[str, object] | None,
    request_fn: Callable[[Mapping[str, object]], Mapping[str, object]] | None,
    model: str | None,
    native_n_probs: bool,
    seed: int | None,
    decision_id: str | None,
    site_id: str,
    question_version: str,
    legal_mass_floor: float,
    permutation: Sequence[int] | None,
    permutation_seed: int | str | None,
    require_tokenizer: bool,
) -> DecisionRecord:
    site_id = _text(site_id, "site_id", limit=200)
    question_version = _text(question_version, "question_version", limit=100)
    if isinstance(legal_mass_floor, bool) or not isinstance(legal_mass_floor, (int, float)):
        raise DecisionAPIError("legal_mass_floor must be a number")
    legal_mass_floor = float(legal_mass_floor)
    if not 0.0 <= legal_mass_floor <= 1.0:
        raise DecisionAPIError("legal_mass_floor must be within [0, 1]")
    canonical_ids = tuple(option.option_id for option in options)
    if decision_id is None:
        identity = _canonical_json({
            "site_id": site_id,
            "question_version": question_version,
            "type": decision_type,
            "question": question,
            "options": [option.option_id for option in options],
        })
        decision_id = "decision-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    else:
        decision_id = _text(decision_id, "decision_id", limit=200)
    permutation_tuple = _make_permutation(
        len(options), permutation=permutation, permutation_seed=permutation_seed,
        identity=decision_id,
    )
    prompt_ids = apply_permutation(canonical_ids, permutation_tuple)
    labels = _labels(decision_type, len(options))
    prompt_options = tuple(
        OptionRecord(option_id=option_id, label_token=labels[index],
                     description=options[canonical_ids.index(option_id)].description)
        for index, option_id in enumerate(prompt_ids)
    )
    canonical_label_by_id = {
        option.option_id: prompt_options[index].label_token
        for index, option in enumerate(prompt_options)
    }
    label_to_option = {label: option_id for option_id, label in canonical_label_by_id.items()}
    token_ids: dict[str, object] = {}
    if tokenizer is not None:
        token_ids = validate_single_token_labels(tuple(label_to_option), tokenizer)
    elif require_tokenizer:
        raise DecisionAPIError("a served tokenizer is required for label verification")
    frame = _render_frame(decision_type, question, prompt_options, evidence, invariant_preamble)
    request = build_request(frame, model=model, native_n_probs=native_n_probs, seed=seed)
    if response is None:
        if request_fn is None:
            raise DecisionAPIError("provide response or request_fn; the kernel does not create an implicit client")
        response = request_fn(request)
    if not isinstance(response, Mapping):
        raise DecisionAPIError("provider response must be a mapping")
    raw_top, prompt_masses, legal_mass, unlettered_mass = _extract_masses(response, label_to_option)
    raw_masses = {option_id: float(prompt_masses.get(option_id, 0.0)) for option_id in canonical_ids}
    if legal_mass > 0.0:
        probabilities = {option_id: mass / legal_mass for option_id, mass in raw_masses.items()}
    else:
        probabilities = {option_id: 0.0 for option_id in canonical_ids}
    prompt_probability = [probabilities[option_id] for option_id in prompt_ids]
    argmax_option = prompt_ids[prompt_probability.index(max(prompt_probability))] if legal_mass > 0 else None
    argmax_label = canonical_label_by_id.get(argmax_option) if argmax_option else None
    if legal_mass < legal_mass_floor:
        selected_option = None
        selected_label = None
        abstain_reason = "legal_mass_below_floor"
    elif argmax_option == ABSTAIN_ID:
        selected_option = None
        selected_label = argmax_label
        abstain_reason = "abstain_option_won"
    else:
        selected_option = argmax_option
        selected_label = argmax_label
        abstain_reason = None
    return DecisionRecord(
        schema=SCHEMA,
        decision_id=decision_id,
        site_id=site_id,
        question_version=question_version,
        decision_type=decision_type,
        question=question,
        options=tuple(
            OptionRecord(option_id=option.option_id,
                         label_token=canonical_label_by_id[option.option_id],
                         description=option.description)
            for option in options
        ),
        option_order=tuple(prompt_ids),
        permutation=permutation_tuple,
        frame=frame,
        request=request,
        tokenizer_id=_tokenizer_id(tokenizer, tokenizer_id),
        tokenizer_verified=tokenizer is not None,
        token_ids=token_ids,
        raw_top_logprobs=raw_top,
        raw_masses=raw_masses,
        legal_mass=legal_mass,
        unlettered_mass=unlettered_mass,
        normalized_probabilities=probabilities,
        argmax_option=argmax_option,
        selected_option=selected_option,
        selected_label=selected_label,
        abstained=selected_option is None,
        abstain_reason=abstain_reason,
        disposition="ABSTAIN" if selected_option is None else "PROPOSE",
    )


def score_choice(
    question: str,
    options: Sequence[object],
    *,
    evidence: Sequence[object] = (),
    invariant_preamble: str = DEFAULT_INVARIANT_PREAMBLE,
    tokenizer: object | None = None,
    tokenizer_id: str | None = None,
    response: Mapping[str, object] | None = None,
    request_fn: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
    model: str | None = None,
    native_n_probs: bool = True,
    seed: int | None = None,
    decision_id: str | None = None,
    site_id: str = "default",
    question_version: str = "v1",
    legal_mass_floor: float = DEFAULT_LEGAL_MASS_FLOOR,
    permutation: Sequence[int] | None = None,
    permutation_seed: int | str | None = None,
    require_tokenizer: bool = False,
    include_abstain: bool = True,
) -> DecisionRecord:
    """Score one choice among host-provided options plus an optional ABSTAIN."""

    canonical = list(_canonical_options(options, limit=MAX_CHOICE_OPTIONS))
    if include_abstain:
        canonical.append(DecisionOption(ABSTAIN_ID, "No listed option has enough support."))
    return _score(
        decision_type="CHOICE", question=question, options=tuple(canonical), evidence=evidence,
        invariant_preamble=invariant_preamble, tokenizer=tokenizer, tokenizer_id=tokenizer_id,
        response=response, request_fn=request_fn, model=model, native_n_probs=native_n_probs,
        seed=seed, decision_id=decision_id, site_id=site_id, question_version=question_version,
        legal_mass_floor=legal_mass_floor, permutation=permutation,
        permutation_seed=permutation_seed, require_tokenizer=require_tokenizer,
    )


def score_score(
    question: str,
    levels: Sequence[object] = ("1", "2", "3", "4", "5"),
    *,
    descriptions: Mapping[str, object] | Sequence[object] | None = None,
    evidence: Sequence[object] = (),
    invariant_preamble: str = DEFAULT_INVARIANT_PREAMBLE,
    tokenizer: object | None = None,
    tokenizer_id: str | None = None,
    response: Mapping[str, object] | None = None,
    request_fn: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
    model: str | None = None,
    native_n_probs: bool = True,
    seed: int | None = None,
    decision_id: str | None = None,
    site_id: str = "default",
    question_version: str = "v1",
    legal_mass_floor: float = DEFAULT_LEGAL_MASS_FLOOR,
    permutation: Sequence[int] | None = None,
    permutation_seed: int | str | None = None,
    require_tokenizer: bool = False,
) -> DecisionRecord:
    """Score ordered levels; callers may compute an expected level downstream."""

    return _score(
        decision_type="SCORE", question=question,
        options=_described_options(levels, descriptions), evidence=evidence,
        invariant_preamble=invariant_preamble, tokenizer=tokenizer, tokenizer_id=tokenizer_id,
        response=response, request_fn=request_fn, model=model, native_n_probs=native_n_probs,
        seed=seed, decision_id=decision_id, site_id=site_id, question_version=question_version,
        legal_mass_floor=legal_mass_floor, permutation=permutation,
        permutation_seed=permutation_seed, require_tokenizer=require_tokenizer,
    )


def score_boolean(
    question: str,
    *,
    evidence: Sequence[object] = (),
    invariant_preamble: str = DEFAULT_INVARIANT_PREAMBLE,
    tokenizer: object | None = None,
    tokenizer_id: str | None = None,
    response: Mapping[str, object] | None = None,
    request_fn: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
    model: str | None = None,
    native_n_probs: bool = True,
    seed: int | None = None,
    decision_id: str | None = None,
    site_id: str = "default",
    question_version: str = "v1",
    legal_mass_floor: float = DEFAULT_LEGAL_MASS_FLOOR,
    permutation: Sequence[int] | None = None,
    permutation_seed: int | str | None = None,
    require_tokenizer: bool = False,
) -> DecisionRecord:
    """Score whether a host-provided statement is true using Y/N tokens."""

    return _score(
        decision_type="BOOLEAN", question=question,
        options=(DecisionOption("true", "The statement is true."),
                 DecisionOption("false", "The statement is false.")),
        evidence=evidence, invariant_preamble=invariant_preamble, tokenizer=tokenizer,
        tokenizer_id=tokenizer_id, response=response, request_fn=request_fn, model=model,
        native_n_probs=native_n_probs, seed=seed, decision_id=decision_id, site_id=site_id,
        question_version=question_version, legal_mass_floor=legal_mass_floor,
        permutation=permutation, permutation_seed=permutation_seed,
        require_tokenizer=require_tokenizer,
    )


__all__ = [
    "ABSTAIN_ID",
    "DEFAULT_INVARIANT_PREAMBLE",
    "DEFAULT_LEGAL_MASS_FLOOR",
    "DecisionAPIError",
    "DecisionOption",
    "DecisionRecord",
    "OptionRecord",
    "RawLogprob",
    "apply_permutation",
    "build_request",
    "score_boolean",
    "score_choice",
    "score_score",
    "undo_permutation",
    "validate_single_token_labels",
]
