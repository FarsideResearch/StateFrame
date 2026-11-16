"""kernel_legacy_safe_edit: the validator-backed single edit reproduces today's str_replace prose byte for byte
and composes exactly what src.replace(old, new, 1) would, through EditAuthority."""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENGINE = ROOT / "src" / "stateframe" / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from kernel_edit_refusal_codes import RefusalCode  # noqa: E402
from kernel_legacy_safe_edit import SYNTHETIC_HANDLE, python_parse_check, single_edit  # noqa: E402

GOLDEN = json.loads((ROOT / "tests" / "fixtures" / "legacy_safe_observations.json").read_text("utf-8"))["observations"]
SOURCE = ("def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n").encode("utf-8")


def run(old, new, source=SOURCE):
    return single_edit(relative_path="mod.py", source=source, old_text=old, new_text=new, root="/")


def test_not_found_prose_matches_golden():
    out = run("    return 3\n", "    return 30\n")
    assert out.observation == GOLDEN["off_not_found"] and out.candidate is None
    assert out.codes == (RefusalCode.OLD_NOT_FOUND,)


def test_n_places_prose_matches_golden_with_real_count():
    out = run("    return ", "    return 9 + ")
    assert out.observation == GOLDEN["off_n_places"]
    assert out.codes == (RefusalCode.AMBIGUOUS_MATCH,)


def test_parse_break_prose_matches_golden():
    out = run("    return 2\n", "    return 2 )\n")
    assert out.observation == GOLDEN["off_parse_break"]
    assert out.codes == (RefusalCode.PARSE_BREAK,)


def test_success_composes_exactly_like_str_replace_once():
    out = run("    return 1\n", "    return 10\n")
    assert out.observation is None and out.codes == ()
    assert out.candidate == SOURCE.decode().replace("    return 1\n", "    return 10\n", 1).encode()


@pytest.mark.parametrize("prefix", [b"\xef\xbb\xbf", b""])
def test_bom_and_crlf_bytes_outside_the_edit_are_preserved(prefix):
    source = prefix + SOURCE.replace(b"\n", b"\r\n")
    out = run("    return 1\r\n", "    return 10\r\n", source=source)
    assert out.candidate is not None
    assert out.candidate.startswith(prefix) and b"\r\n" in out.candidate and b"return 10" in out.candidate
    assert out.candidate.count(b"\n") == source.count(b"\n")


def test_synthetic_handle_never_leaks_into_prose():
    for old, new in [("    return 3\n", "x"), ("    return ", "y"), ("    return 2\n", "    return 2 )\n")]:
        obs = run(old, new).observation
        assert obs is not None and SYNTHETIC_HANDLE not in obs and "q0" not in obs


def test_parse_check_detail_format():
    assert python_parse_check(b"def f(:\n    pass\n") is not None
    assert python_parse_check(b"x = 1\n") is None
    detail = python_parse_check(b"def alpha():\n    return 1\n\n\ndef beta():\n    return 2 )\n")
    assert detail == "unmatched ')' line 6"
