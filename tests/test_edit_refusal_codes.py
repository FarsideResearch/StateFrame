"""kernel_edit_refusal_codes: closed catalog, handle-free public text, stable fingerprints,
and a prose -> code map that leaves today's observations untouched."""
from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from kernel_edit_refusal_codes import (  # noqa: E402
    CODE_COUNT, HANDLE_PATTERN, PUBLIC_TEXT, REFUSAL_CODES, Refusal, RefusalCode,
    classify_legacy_refusal, fingerprint, path_digest, public_text,
)

EXPECTED = [
    "STALE_HANDLE", "SOURCE_CHANGED", "UNCOVERED_SOURCE", "OLD_NOT_FOUND",
    "AMBIGUOUS_MATCH", "DISPLAY_MARKER", "OVERLAPPING_EDITS", "MIXED_FILE_BATCH",
    "EMPTY_OLD_TEXT", "TOO_MANY_EDITS", "PARSE_BREAK", "NOT_EDITABLE",
]


def test_catalog_is_closed_and_ordered():
    assert [code.value for code in REFUSAL_CODES] == EXPECTED
    assert len(RefusalCode) == CODE_COUNT == 12
    assert set(PUBLIC_TEXT) == set(RefusalCode)
    with pytest.raises(ValueError):
        RefusalCode("SOMETHING_ELSE")
    assert RefusalCode("OLD_NOT_FOUND") is RefusalCode.OLD_NOT_FOUND
    assert all(isinstance(code.value, str) and code.value == code.name for code in RefusalCode)


@pytest.mark.parametrize("code", list(RefusalCode))
def test_every_code_has_public_text_without_handle_spelling(code):
    text = public_text(code)
    assert text and text == PUBLIC_TEXT[code]
    assert HANDLE_PATTERN.search(text) is None
    assert "q42h1" not in text and "handle id" not in text.lower()
    assert text.endswith(".")


def test_public_text_rejects_strings_and_unknown_values():
    with pytest.raises(ValueError):
        public_text("OLD_NOT_FOUND")  # type: ignore[arg-type]


def test_fingerprint_is_sha256_of_code_path_digest_and_old_text():
    expected = hashlib.sha256(
        b"OLD_NOT_FOUND|" + path_digest("src/a.py").encode() + b"|def f():\n").hexdigest()
    assert fingerprint(RefusalCode.OLD_NOT_FOUND, "src/a.py", "def f():\n") == expected
    assert path_digest("src/a.py") == hashlib.sha256(b"src/a.py").hexdigest()


def test_fingerprint_stable_across_handle_renames_and_request_ordinals():
    one = Refusal.build(RefusalCode.OLD_NOT_FOUND, edit_index=0, path="src/a.py",
                        old_text="x = 1\n")
    two = Refusal.build(RefusalCode.OLD_NOT_FOUND, edit_index=3, path="src/a.py",
                        old_text="x = 1\n")
    assert one.fingerprint == two.fingerprint
    for handle in ("q42h1", "q43h1", "q9h7"):
        assert handle not in one.fingerprint
        assert HANDLE_PATTERN.search(one.public_text) is None
    assert HANDLE_PATTERN.search(one.as_dict()["public_text"]) is None


def test_fingerprint_separates_code_path_and_old_text():
    base = fingerprint(RefusalCode.OLD_NOT_FOUND, "src/a.py", "x")
    assert fingerprint(RefusalCode.AMBIGUOUS_MATCH, "src/a.py", "x") != base
    assert fingerprint(RefusalCode.OLD_NOT_FOUND, "src/b.py", "x") != base
    assert fingerprint(RefusalCode.OLD_NOT_FOUND, "src/a.py", "y") != base
    # No delimiter confusion: path and old_text cannot shift bytes into each other.
    assert (fingerprint(RefusalCode.OLD_NOT_FOUND, "a|b", "c")
            != fingerprint(RefusalCode.OLD_NOT_FOUND, "a", "b|c"))
    with pytest.raises(ValueError):
        fingerprint(RefusalCode.OLD_NOT_FOUND, "", "x")
    with pytest.raises(ValueError):
        fingerprint(RefusalCode.OLD_NOT_FOUND, "src/a.py", None)  # type: ignore[arg-type]


def test_refusal_validates_fields():
    good = Refusal.build(RefusalCode.PARSE_BREAK, edit_index=1, path="p.py", old_text="a")
    assert good.as_dict() == {
        "code": "PARSE_BREAK", "edit_index": 1, "fingerprint": good.fingerprint,
        "public_text": PUBLIC_TEXT[RefusalCode.PARSE_BREAK]}
    with pytest.raises(ValueError):
        Refusal(code="PARSE_BREAK", edit_index=0, fingerprint=good.fingerprint,  # type: ignore[arg-type]
                public_text="x.")
    with pytest.raises(ValueError):
        Refusal(code=RefusalCode.PARSE_BREAK, edit_index=-1, fingerprint=good.fingerprint,
                public_text="x.")
    with pytest.raises(ValueError):
        Refusal(code=RefusalCode.PARSE_BREAK, edit_index=True, fingerprint=good.fingerprint,
                public_text="x.")
    with pytest.raises(ValueError):
        Refusal(code=RefusalCode.PARSE_BREAK, edit_index=0, fingerprint="abc",
                public_text="x.")
    with pytest.raises(ValueError):
        Refusal(code=RefusalCode.PARSE_BREAK, edit_index=0, fingerprint=good.fingerprint,
                public_text="Handle q42h1 is stale.")


# Today's prose refusals, byte-for-byte as the executors emit them.
LEGACY = [
    ("str_replace REJECTED: old_str not found. Copy the exact current text "
     "(check whitespace) — `read` the region first.", RefusalCode.OLD_NOT_FOUND),
    ("str_replace REJECTED: old_str matches 3 places. Include more surrounding lines "
     "so it is unique.", RefusalCode.AMBIGUOUS_MATCH),
    ("str_replace REJECTED: the file would not parse afterwards (invalid syntax line 4). "
     "Nothing was changed.", RefusalCode.PARSE_BREAK),
    ("edit REJECTED: syntax error at line 9", RefusalCode.PARSE_BREAK),
    ("str_replace REJECTED: old_str is required and must be non-empty: copy the exact "
     "current text to replace. Nothing was changed.", RefusalCode.EMPTY_OLD_TEXT),
    ("str_replace REJECTED: the marker is not file content; re-read lines 1-40. ",
     RefusalCode.DISPLAY_MARKER),
    ("insert_before REJECTED: protected public test paths are immutable",
     RefusalCode.NOT_EDITABLE),
    ("str_replace REJECTED: pkg is a directory, not a file.", RefusalCode.NOT_EDITABLE),
    ("STALE_SOURCE: src/a.py no longer matches site-1", RefusalCode.SOURCE_CHANGED),
    ("STALE_REVISION: file revision changed for src/a.py", RefusalCode.SOURCE_CHANGED),
    ("STALE_SPAN: site-1 is outside current source", RefusalCode.UNCOVERED_SOURCE),
    ("AMBIGUOUS[0]: site site-1 was selected twice", RefusalCode.AMBIGUOUS_MATCH),
    ("AMBIGUOUS_ORDER: insertion boundary src/a.py:12 needs unique order",
     RefusalCode.AMBIGUOUS_MATCH),
    ('{"status": "REJECTED", "code": "OLD_TEXT_MISMATCH"}', RefusalCode.OLD_NOT_FOUND),
]


@pytest.mark.parametrize("observation, code", LEGACY)
def test_legacy_prose_maps_to_catalog_code(observation, code):
    assert classify_legacy_refusal(observation) is code


def test_legacy_prose_outside_catalog_is_unmapped_and_untouched():
    for observation in ("str_replace applied.", "", "str_replace NO-OP: old_str and "
                        "new_str are identical", None):
        assert classify_legacy_refusal(observation) is None  # type: ignore[arg-type]
    text = LEGACY[0][0]
    classify_legacy_refusal(text)
    assert text == LEGACY[0][0]


def test_legacy_executor_prose_still_present_in_engine_source():
    """The map reads today's bytes; the executors must still emit them unchanged."""
    executor = (ENGINE / "probes" / "react_repair.py").read_text()
    for needle in ("old_str not found. Copy the exact current text",
                   "old_str matches {n} places. Include more",
                   "the file would not parse afterwards",
                   "old_str is required and must be non-empty"):
        assert needle in executor, needle
    authority = (ENGINE / "kernel_revision_bound_edit.py").read_text()
    for needle in ("STALE_SOURCE:", "STALE_REVISION:", "STALE_SPAN:", "AMBIGUOUS["):
        assert needle in authority, needle


def test_module_is_pure_and_imports_no_engine_code():
    tree = ast.parse((ENGINE / "kernel_edit_refusal_codes.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert imported == {"__future__", "enum", "hashlib", "re", "dataclasses"}
    assert not any(name.startswith(("kernel_", "probes", "stateframe")) for name in imported)
    source = (ENGINE / "kernel_edit_refusal_codes.py").read_text()
    assert "os.environ" not in source and "open(" not in source


CATALOG_IMPORTERS = {
    "kernel_edit_refusal_codes.py",
    "kernel_edit_source_batch.py",
    "kernel_revision_bound_edit.py",
    "kernel_edit_handles.py",
    "kernel_edit_repair_request.py",
    "kernel_edit_source_tool.py",
    "kernel_edit_repair_hook.py",  # builds repair requests from refusal codes
    "kernel_legacy_safe_edit.py",  # maps legacy str_replace refusals onto the same codes
}


def test_only_edit_protocol_modules_import_the_catalog():
    """Only the edit-protocol modules (validator, authority, handles, repair, executors) may depend on the catalog."""
    for path in ENGINE.rglob("*.py"):
        if path.name in CATALOG_IMPORTERS or "__pycache__" in path.parts:
            continue
        assert "kernel_edit_refusal_codes" not in path.read_text(), path
