"""Tokenizer-qualified, single-token labels for the System One prompt facade.

This module deliberately stops at the prompt boundary.  It records exact
token IDs and UTF-8 bytes for a label, but it does not inspect or extract model
scores.  A tokenizer adapter may be a recorded fixture, a tokenizer object, or
another provider-neutral callable; no transport package is imported here.
"""

from __future__ import annotations

import hashlib
import json
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast


NEUTRAL_LABELS: tuple[str, ...] = tuple(string.ascii_uppercase[:8])
"""The only labels this protocol qualifies: neutral ASCII ``A`` through ``H``."""


class LabelQualificationError(ValueError):
    """Raised when labels cannot be proven safe at the answer boundary."""


class TokenizerLike(Protocol):
    """Minimal optional tokenizer surface accepted by :func:`qualify_labels`."""

    def tokenize(self, text: str, **kwargs: object) -> object:
        """Return IDs plus exact pieces/bytes for ``text``."""


@dataclass(frozen=True)
class TokenizedText:
    """Exact tokenizer observation used for qualification.

    ``token_bytes`` are bytes emitted by each token, not a re-encoding of a
    normalized token string.  Keeping them explicit prevents punctuation or
    whitespace stripping from turning a different token into a valid label.
    """

    token_ids: tuple[int, ...]
    token_bytes: tuple[bytes, ...]
    special_tokens: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.token_bytes):
            raise LabelQualificationError(
                "token IDs and token bytes have different lengths"
            )
        if len(self.token_ids) != len(self.special_tokens):
            raise LabelQualificationError(
                "token IDs and special-token flags have different lengths"
            )


@dataclass(frozen=True)
class QualifiedLabel:
    """One neutral label proven to be one exact non-special tokenizer token."""

    label: str
    token_id: int
    token_bytes: bytes


@dataclass(frozen=True)
class LabelQualification:
    """Immutable qualification receipt for one tokenizer/template boundary."""

    labels: tuple[QualifiedLabel, ...]
    answer_boundary: str
    boundary_sha256: str
    template_fingerprint: str
    tokenizer_fingerprint: str
    boundary_token_ids: tuple[int, ...]

    @property
    def token_ids(self) -> dict[str, int]:
        """Return the qualified ``label -> token ID`` mapping."""

        return {item.label: item.token_id for item in self.labels}

    @property
    def label_token_map(self) -> dict[str, int]:
        """Explicit alias for transport adapters consuming label-token metadata."""

        return self.token_ids

    @property
    def token_bytes(self) -> dict[str, bytes]:
        """Return the qualified ``label -> exact token bytes`` mapping."""

        return {item.label: item.token_bytes for item in self.labels}

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe receipt without losing byte identity."""

        return {
            "labels": [
                {
                    "label": item.label,
                    "token_id": item.token_id,
                    "token_bytes_hex": item.token_bytes.hex(),
                }
                for item in self.labels
            ],
            "answer_boundary": self.answer_boundary,
            "boundary_sha256": self.boundary_sha256,
            "template_fingerprint": self.template_fingerprint,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "boundary_token_ids": list(self.boundary_token_ids),
        }


def fingerprint_bytes(value: bytes) -> str:
    """Return the SHA-256 fingerprint used for exact wire bytes."""

    return hashlib.sha256(value).hexdigest()


def fingerprint_json(value: object) -> str:
    """Return a stable SHA-256 fingerprint for JSON-compatible metadata."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return fingerprint_bytes(encoded)


def _as_text(value: object, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise LabelQualificationError(f"{field} must be text")
    if not allow_empty and not value:
        raise LabelQualificationError(f"{field} must not be empty")
    if "\x00" in value:
        raise LabelQualificationError(f"{field} contains NUL")
    return value


def _as_id(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LabelQualificationError(f"{field} must be an integer token ID")
    if value < 0:
        raise LabelQualificationError(f"{field} must be non-negative")
    return value


def _bytes_value(value: object, field: str) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        numbers = list(value)
        if any(
            isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 255
            for item in numbers
        ):
            raise LabelQualificationError(f"{field} contains invalid byte values")
        return bytes(cast(list[int], numbers))
    raise LabelQualificationError(f"{field} is not a byte sequence")


def _sequence(value: object, field: str) -> list[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise LabelQualificationError(f"{field} must be a sequence")
    return list(value)


def _mapping_field(
    payload: Mapping[str, object], names: tuple[str, ...]
) -> object | None:
    for name in names:
        if name in payload:
            return payload[name]
    return None


def _decode_one(tokenizer: object, token_id: int) -> bytes | None:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        return None
    for value in ([token_id], (token_id,)):
        try:
            result = decode(value, skip_special_tokens=False)
        except TypeError:
            try:
                result = decode(value)
            except (TypeError, ValueError, KeyError):
                continue
        except (ValueError, KeyError):
            continue
        if isinstance(result, bytes):
            return result
        if isinstance(result, str):
            return result.encode("utf-8")
    return None


def _decode_ids(tokenizer: object, ids: Sequence[int]) -> tuple[bytes, ...] | None:
    decoded: list[bytes] = []
    for token_id in ids:
        value = _decode_one(tokenizer, token_id)
        if value is None:
            return None
        decoded.append(value)
    return tuple(decoded)


def _parse_tokenization(tokenizer: object, raw: object, text: str) -> TokenizedText:
    payload: Mapping[str, object] | None = raw if isinstance(raw, Mapping) else None
    if payload is not None:
        ids_raw = _mapping_field(payload, ("token_ids", "input_ids", "ids", "tokens"))
        bytes_raw = _mapping_field(payload, ("token_bytes", "tokens_bytes", "bytes"))
        pieces_raw = _mapping_field(
            payload, ("pieces", "tokens_string", "token_strings")
        )
        special_raw = _mapping_field(payload, ("special_tokens_mask", "special"))
        if ids_raw is None and isinstance(payload.get("encoding"), Mapping):
            return _parse_tokenization(tokenizer, payload["encoding"], text)
    else:
        ids_raw = raw
        bytes_raw = None
        pieces_raw = None
        special_raw = None

    ids_values = _sequence(ids_raw, f"tokenization for {text!r}.ids")
    ids = tuple(_as_id(value, f"tokenization for {text!r}.ids") for value in ids_values)

    if bytes_raw is not None:
        byte_values = _sequence(bytes_raw, f"tokenization for {text!r}.token_bytes")
        token_bytes = tuple(
            _bytes_value(value, f"tokenization for {text!r}.token_bytes[{index}]")
            for index, value in enumerate(byte_values)
        )
    elif pieces_raw is not None:
        pieces = _sequence(pieces_raw, f"tokenization for {text!r}.pieces")
        token_bytes = tuple(
            _as_text(value, f"tokenization for {text!r}.pieces[{index}]").encode(
                "utf-8"
            )
            for index, value in enumerate(pieces)
        )
    else:
        decoded = _decode_ids(tokenizer, ids)
        if decoded is None:
            raise LabelQualificationError(
                f"tokenizer returned IDs for {text!r} without exact token bytes or decode()"
            )
        token_bytes = decoded

    if len(token_bytes) != len(ids):
        raise LabelQualificationError(
            f"tokenization for {text!r} has mismatched IDs and bytes"
        )

    if special_raw is None:
        special_tokens = (False,) * len(ids)
    else:
        special_values = _sequence(
            special_raw, f"tokenization for {text!r}.special_tokens_mask"
        )
        if len(special_values) != len(ids):
            raise LabelQualificationError(
                f"tokenization for {text!r} has invalid special-token mask"
            )
        normalized_special: list[bool] = []
        for index, value in enumerate(special_values):
            if type(value) is bool:
                normalized_special.append(value)
            elif type(value) is int and value in (0, 1):
                normalized_special.append(bool(value))
            else:
                raise LabelQualificationError(
                    f"tokenization for {text!r}.special_tokens_mask[{index}] "
                    "must be boolean or 0/1"
                )
        special_tokens = tuple(normalized_special)

    return TokenizedText(ids, token_bytes, special_tokens)


def _invoke_tokenizer(tokenizer: object, text: str) -> TokenizedText:
    raw: object
    tokenize = getattr(tokenizer, "tokenize", None)
    if callable(tokenize):
        attempts = (
            {"add_special_tokens": False, "with_pieces": True},
            {"add_special": False, "with_pieces": True},
            {"add_special_tokens": False},
            {},
        )
        for kwargs in attempts:
            try:
                raw = tokenize(text, **kwargs)
                return _parse_tokenization(tokenizer, raw, text)
            except TypeError:
                continue
            except (KeyError, ValueError) as error:
                raise LabelQualificationError(
                    f"tokenizer cannot qualify {text!r}: {type(error).__name__}"
                ) from error
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        for kwargs in ({"add_special_tokens": False}, {}):
            try:
                raw = encode(text, **kwargs)
                return _parse_tokenization(tokenizer, raw, text)
            except TypeError:
                continue
            except (KeyError, ValueError) as error:
                raise LabelQualificationError(
                    f"tokenizer cannot qualify {text!r}: {type(error).__name__}"
                ) from error
    if isinstance(tokenizer, Mapping):
        records = tokenizer.get("tokenizations")
        if isinstance(records, Mapping) and text in records:
            return _parse_tokenization(tokenizer, records[text], text)
        if text in tokenizer:
            return _parse_tokenization(tokenizer, tokenizer[text], text)
    if callable(tokenizer):
        try:
            raw = tokenizer(text)
        except (KeyError, TypeError, ValueError) as error:
            raise LabelQualificationError(
                f"tokenizer cannot qualify {text!r}: {type(error).__name__}"
            ) from error
        return _parse_tokenization(tokenizer, raw, text)
    raise LabelQualificationError(
        "tokenizer must expose tokenize()/encode(), be callable, or contain recorded tokenizations"
    )


def _identity(tokenizer: object, explicit: str | None) -> str:
    if explicit is not None:
        return _as_text(explicit, "tokenizer_fingerprint")
    if isinstance(tokenizer, Mapping):
        for key in ("tokenizer_fingerprint", "tokenizer_id", "name_or_path", "name"):
            value = tokenizer.get(key)
            if isinstance(value, str) and value:
                return value
    for name in ("tokenizer_fingerprint", "tokenizer_id", "name_or_path", "name"):
        value = getattr(tokenizer, name, None)
        if isinstance(value, str) and value:
            return value
    return fingerprint_json(
        {
            "adapter_module": type(tokenizer).__module__,
            "adapter_name": type(tokenizer).__qualname__,
        }
    )


def _special_token_ids(tokenizer: object) -> frozenset[int]:
    candidates: object | None = None
    if isinstance(tokenizer, Mapping):
        for key in ("special_token_ids", "all_special_ids"):
            if key in tokenizer:
                candidates = tokenizer[key]
                break
    if candidates is None:
        for name in ("special_token_ids", "all_special_ids"):
            value = getattr(tokenizer, name, None)
            if value is not None:
                candidates = value
                break
    if candidates is None or isinstance(candidates, (str, bytes, bytearray)):
        return frozenset()
    if not isinstance(candidates, Sequence):
        return frozenset()
    return frozenset(
        value
        for value in candidates
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
    )


def _label_sequence(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise LabelQualificationError(
            "labels must be a sequence of neutral ASCII labels"
        )
    labels = tuple(value)
    if not labels or len(labels) > len(NEUTRAL_LABELS):
        raise LabelQualificationError(
            "labels must contain between one and eight labels"
        )
    normalized: list[str] = []
    for index, label in enumerate(labels):
        if not isinstance(label, str) or label not in NEUTRAL_LABELS:
            raise LabelQualificationError(
                f"labels[{index}] must be one of neutral ASCII A-H; got {label!r}"
            )
        normalized.append(label)
    if len(set(normalized)) != len(normalized):
        raise LabelQualificationError("labels must be unique")
    return tuple(normalized)


def _is_label_sequence(value: object) -> bool:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        return False
    return all(isinstance(item, str) for item in value)


def one_token_grammar(labels: Sequence[str]) -> str:
    """Return an exact grammar with no optional whitespace or wrapper output."""

    selected = _label_sequence(labels)
    return "root ::= " + " | ".join(json.dumps(label) for label in selected)


def qualify_labels(
    first: object,
    second: object | None = None,
    *,
    answer_boundary: str | bytes | None = None,
    template_fingerprint: str | None = None,
    tokenizer_fingerprint: str | None = None,
) -> LabelQualification:
    """Qualify neutral labels against the exact bytes at a real answer boundary.

    Both ``qualify_labels(labels, tokenizer, ...)`` and
    ``qualify_labels(tokenizer, labels, ...)`` are accepted for compatibility;
    omitting ``second`` means all A-H labels.  The boundary is concatenated with
    each label without inserting a separator, so a leading-space token cannot
    pass by normalization.
    """

    if second is None:
        tokenizer = first
        labels = NEUTRAL_LABELS
    elif _is_label_sequence(first) and not _is_label_sequence(second):
        labels = first
        tokenizer = second
    else:
        tokenizer = first
        labels = second
    selected = _label_sequence(labels)
    if answer_boundary is None:
        raise LabelQualificationError(
            "answer_boundary is required for label qualification"
        )
    if isinstance(answer_boundary, bytes):
        try:
            boundary = answer_boundary.decode("utf-8")
        except UnicodeDecodeError as error:
            raise LabelQualificationError(
                "answer_boundary must be valid UTF-8"
            ) from error
        boundary_bytes = answer_boundary
    else:
        boundary = _as_text(answer_boundary, "answer_boundary", allow_empty=True)
        boundary_bytes = boundary.encode("utf-8")

    boundary_tokens = _invoke_tokenizer(tokenizer, boundary)
    if b"".join(boundary_tokens.token_bytes) != boundary_bytes:
        raise LabelQualificationError(
            "tokenizer boundary bytes do not exactly round-trip the answer boundary"
        )
    qualified: list[QualifiedLabel] = []
    seen_ids: set[int] = set()
    special_ids = _special_token_ids(tokenizer)
    for label in selected:
        label_bytes = label.encode("ascii")
        standalone = _invoke_tokenizer(tokenizer, label)
        if len(standalone.token_ids) != 1:
            raise LabelQualificationError(f"label {label!r} is not exactly one token")
        if standalone.special_tokens[0] or standalone.token_ids[0] in special_ids:
            raise LabelQualificationError(f"label {label!r} uses a special token")
        if standalone.token_bytes[0] != label_bytes:
            raise LabelQualificationError(
                f"label {label!r} does not round-trip to exact ASCII bytes"
            )
        token_id = standalone.token_ids[0]
        if token_id in seen_ids:
            raise LabelQualificationError(f"labels share token ID {token_id}")
        seen_ids.add(token_id)

        boundary_plus_label = _invoke_tokenizer(tokenizer, boundary + label)
        prefix_size = len(boundary_tokens.token_ids)
        if (
            boundary_plus_label.token_ids[:prefix_size] != boundary_tokens.token_ids
            or boundary_plus_label.token_bytes[:prefix_size]
            != boundary_tokens.token_bytes
            or boundary_plus_label.special_tokens[:prefix_size]
            != boundary_tokens.special_tokens
        ):
            raise LabelQualificationError(
                f"label {label!r} changes tokenization or bytes before the answer boundary"
            )
        if b"".join(boundary_plus_label.token_bytes) != boundary_bytes + label_bytes:
            raise LabelQualificationError(
                f"label {label!r} does not exactly round-trip at the answer boundary"
            )
        suffix_ids = boundary_plus_label.token_ids[prefix_size:]
        suffix_bytes = boundary_plus_label.token_bytes[prefix_size:]
        suffix_special = boundary_plus_label.special_tokens[prefix_size:]
        if len(suffix_ids) != 1 or suffix_ids[0] != token_id:
            raise LabelQualificationError(
                f"label {label!r} is not one token at the real answer boundary"
            )
        if suffix_special[0] or suffix_bytes[0] != label_bytes:
            raise LabelQualificationError(
                f"label {label!r} has different bytes at the real answer boundary"
            )
        qualified.append(QualifiedLabel(label, token_id, label_bytes))

    return LabelQualification(
        labels=tuple(qualified),
        answer_boundary=boundary,
        boundary_sha256=fingerprint_bytes(boundary_bytes),
        template_fingerprint=(
            _as_text(template_fingerprint, "template_fingerprint")
            if template_fingerprint is not None
            else fingerprint_bytes(boundary_bytes)
        ),
        tokenizer_fingerprint=_identity(tokenizer, tokenizer_fingerprint),
        boundary_token_ids=boundary_tokens.token_ids,
    )


qualify_neutral_labels = qualify_labels


__all__ = [
    "LabelQualification",
    "LabelQualificationError",
    "NEUTRAL_LABELS",
    "QualifiedLabel",
    "TokenizedText",
    "fingerprint_bytes",
    "fingerprint_json",
    "one_token_grammar",
    "qualify_labels",
    "qualify_neutral_labels",
]
