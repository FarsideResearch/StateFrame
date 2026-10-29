"""Product-localized failure cards built from existing failure evidence."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from kernel_gap_switches import emit, truth_enabled


SWITCH = "STATEFRAME_GAPS_FAILURE_CARDS"
PRODUCER = "kernel_gap_failure_cards"
MAX_TEXT = 420


def _text(value: object, limit: int = MAX_TEXT) -> str:
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, Mapping):
        text = ", ".join(f"{key}={child}" for key, child in value.items())
    else:
        text = str(value or "")
    return text[:limit] + ("..." if len(text) > limit else "")


@dataclass(frozen=True)
class FailureCard:
    classification: str
    asserted_behavior: str
    observed_difference: str
    test_location: str
    product_path: str
    product_symbol: str
    confidence: str
    recovery_actions: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "classification": self.classification,
            "asserted_behavior": self.asserted_behavior,
            "observed_difference": self.observed_difference,
            "test_location": self.test_location,
            "product_path": self.product_path,
            "product_symbol": self.product_symbol,
            "confidence": self.confidence,
            "recovery_actions": list(self.recovery_actions),
        }


def _product_frame(failure: Mapping[str, object]) -> Mapping[str, object] | None:
    frames = failure.get("product_frames", failure.get("frames", ()))
    if isinstance(frames, Mapping):
        return frames
    if isinstance(frames, Sequence) and not isinstance(frames, (str, bytes, bytearray)):
        for frame in frames:
            if isinstance(frame, Mapping) and (frame.get("path") or frame.get("symbol")):
                return frame
    return None


def build_failure_card(failure: Mapping[str, object], *, journal: object | None = None,
                       symbol_index: Mapping[str, str] | None = None) -> FailureCard | None:
    if not truth_enabled(SWITCH):
        return None
    product = _product_frame(failure)
    error_class = _text(failure.get("error_class", failure.get("exception", "")), 80).lower()
    import_error = "import" in error_class or "module" in error_class
    test_location = _text(failure.get("test_location", failure.get("location", "")), 180)
    product_path = _text((product or {}).get("path", ""), 180)
    product_symbol = _text(
        failure.get("product_symbol", (product or {}).get("symbol", "")), 140)
    if symbol_index and product_symbol and product_symbol in symbol_index:
        product_path = _text(symbol_index[product_symbol], 180)
    asserted = _text(failure.get("asserted_behavior", failure.get("expected", "")))
    observed = _text(failure.get("observed_difference", failure.get("observed", "")))
    if import_error and not product_path:
        classification = "harness_or_fixture_import"
        confidence = "low"
        actions = ("inspect_collection_fixture", "read_import_traceback", "run_focused_check")
    elif import_error and product_path:
        target = _text(failure.get("import_target", ""), 120)
        classification = "wrong_import_target" if target and target != product_path else "missing_product_definition"
        confidence = "medium"
        actions = ("inspect_product_importer", "read_product_symbol", "run_focused_witness")
    elif product_path:
        classification = "product_behavior_mismatch"
        confidence = "medium" if product_symbol else "low"
        actions = ("read_product_source", "inspect_data_transformation", "run_focused_witness")
    else:
        classification = "unlocalized_failure"
        confidence = "low"
        actions = ("read_failure_context", "locate_product_surface", "run_focused_witness")
    card = FailureCard(classification, asserted, observed, test_location, product_path,
                       product_symbol, confidence, actions)
    emit(journal, "gaps_failure_card", producer=PRODUCER, enabled_value=True,
         classification=card.classification, test_location=card.test_location,
         product_path=card.product_path, product_symbol=card.product_symbol,
         confidence=card.confidence, recovery_action_count=len(card.recovery_actions))
    return card


def render_failure_card(card: FailureCard | None) -> str:
    if card is None:
        return ""
    return "[Product-localized failure]\n" + json.dumps(card.to_dict(), sort_keys=True,
                                                            separators=(",", ":"))


__all__ = ["FailureCard", "PRODUCER", "SWITCH", "build_failure_card", "render_failure_card"]
