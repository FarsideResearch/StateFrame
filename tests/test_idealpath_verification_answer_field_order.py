"""The ideal-path final verification answer asks for evidence before the outcome enum."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

kernel_idealpath = importlib.import_module("kernel_idealpath")

EXPECTED_REQUIRED = [
    "artifact_ref", "changes", "verification_refs",
    "unresolved_row_ids", "residual_risks", "limitations", "outcome",
]


def test_s7_outcome_enum_is_last_in_properties_and_required() -> None:
    schema = kernel_idealpath._answer_schema("S7")
    properties = list(schema["properties"])
    assert schema["required"] == EXPECTED_REQUIRED
    assert properties == EXPECTED_REQUIRED[:-1] + ["finish_proposal", "outcome"]
    assert properties[-1] == "outcome"
    assert schema["properties"]["outcome"]["enum"] == [
        "VERIFIED_FIXED", "NOT_REPRODUCED", "INCOMPLETE",
    ]


def test_s7_answer_fields_table_ends_with_outcome() -> None:
    assert kernel_idealpath._ANSWER_FIELDS["S7"][-1] == "outcome"
