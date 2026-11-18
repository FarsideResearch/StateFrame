"""Fail-loud T8-ORACLE: broad except sites in the witness, oracle, check-selection and function-index modules.

Scope: src/stateframe/engine/check_selection.py, func_index.py, intent_oracle.py and witness_replay.py. None of the
four reads a STATEFRAME_* switch, so every site is on the shipped path.

Loudness bar: a raise, a WARNING-or-higher record from the owning module, or a typed outcome that a reader checks.
Each forcing test forces the guarded failure and asserts one of those. Tags T8-<MODULE>-<line> name the guarded site.

Verdicts (line: class). CRASH-HIDER, now fixed and asserted loud: check_selection 200, func_index 633,
intent_oracle 773 and 894, witness_replay 842, 1032 and 1151 (1032 also records the failure on the surviving
evidence rows). ALREADY-COVERED or DEGRADE-OK pins: check_selection 123 (prints and logs),
intent_oracle 847, 865, 871, 948, witness_replay 371, 519, 523, 981, 985 (519/523 are the documented
"no opinion" fallback of _span_facts). intent_oracle 871 keeps SystemExit from candidate code as a typed
outcome; KeyboardInterrupt (Ctrl-C, and SIGTERM in the engine child) re-raises before it. DEAD: intent_oracle 219.
PARSER-BOUNDARY: intent_oracle 239 and 338. AMBIGUOUS (no test, owner question): func_index 436,
intent_oracle 364 and 368, witness_replay 936 and 1271. Wiring: check_selection imports the corpus fix module by
its package-qualified name (probes.apply_corpus_fixes), never by the bare name that needs probes/ on sys.path.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
MODULES = ("check_selection", "func_index", "intent_oracle", "witness_replay")
BROAD = re.compile(r"^\s*except\s*(?:\(|Exception\b|BaseException\b|:)")

# No pilot sample site lives in these four modules.
PILOT: set[tuple[str, int]] = set()
TRIAGED: set[tuple[str, int]] = {
    ("check_selection", 123), ("check_selection", 200),
    ("func_index", 436), ("func_index", 633),
    ("intent_oracle", 219), ("intent_oracle", 239), ("intent_oracle", 338), ("intent_oracle", 364),
    ("intent_oracle", 368), ("intent_oracle", 773), ("intent_oracle", 847), ("intent_oracle", 865),
    ("intent_oracle", 871), ("intent_oracle", 894), ("intent_oracle", 948),
    ("witness_replay", 371), ("witness_replay", 519), ("witness_replay", 523), ("witness_replay", 842),
    ("witness_replay", 936), ("witness_replay", 981), ("witness_replay", 985), ("witness_replay", 1032),
    ("witness_replay", 1151), ("witness_replay", 1271),
}


def _loud(caplog: pytest.LogCaptureFixture, module_name: str) -> bool:
    """True when the module emitted a WARNING-or-higher record during the call."""
    return any(record.levelno >= logging.WARNING and module_name in record.name
               for record in caplog.records)


def test_oracle_broad_excepts_are_triaged() -> None:
    found: set[tuple[str, int]] = set()
    for stem in MODULES:
        path = ENGINE / f"{stem}.py"
        for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not BROAD.match(text):
                continue
            if re.match(r"^\s*except\s*\(", text) and not re.search(r"Exception|BaseException", text):
                continue  # narrow tuple
            found.add((path.stem, number))
    assert found == PILOT | TRIAGED, (
        f"untriaged: {sorted(found - (PILOT | TRIAGED))}; stale: {sorted((PILOT | TRIAGED) - found)}")


def test_oracle_tags_point_at_broad_except_sites() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    tags = re.findall(r"T8-([A-Z0-9_]+)-(\d+)", source)
    assert tags
    for module, line in tags:
        lines = (ENGINE / (module.lower() + ".py")).read_text(encoding="utf-8").splitlines()
        assert BROAD.match(lines[int(line) - 1]), (
            f"{module}:{line} is no longer a broad except; retag or promote the test")


# --- check_selection --------------------------------------------------------------------------------


def test_corpus_overlay_failure_prints_a_visible_line(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                      capsys: pytest.CaptureFixture[str],
                                                      caplog: pytest.LogCaptureFixture) -> None:
    """Site 123 is ALREADY-COVERED (print): the overlay failure names the task and the exception type, and logs."""
    import check_selection as selection
    from probes import apply_corpus_fixes as corpus

    def refusing(root: Path, d: dict[str, str]) -> int:
        raise RuntimeError("forced: overlay failed")

    monkeypatch.setattr(corpus, "apply_corpus_fixes", refusing)
    caplog.set_level(logging.WARNING)
    assert selection.overlay_fix_tests(tmp_path, {"fn": "f_one", "repo": "swe"}) == 0
    assert "[corpus-fix] overlay failed for f_one: RuntimeError" in capsys.readouterr().out
    assert _loud(caplog, "check_selection")


def test_fail_to_pass_corpus_fix_failure_is_loud(monkeypatch: pytest.MonkeyPatch,
                                                 caplog: pytest.LogCaptureFixture) -> None:
    """T8-CHECK_SELECTION-200: a failing corpus lookup still falls back to the commit selector, but logs it."""
    import check_selection as selection
    from probes import apply_corpus_fixes as corpus

    def refusing(d: dict[str, str]) -> None:
        raise RuntimeError("forced: corpus lookup failed")

    monkeypatch.setattr(corpus, "corpus_fix_for", refusing)
    caplog.set_level(logging.WARNING)
    assert selection.fail_to_pass_ids({"repo": "swe", "fn": "f_one", "sha": ""}) == []
    assert _loud(caplog, "check_selection"), "corpus lookup failed but the selector fell back silently"
    assert any("RuntimeError" in record.getMessage() for record in caplog.records)


def test_check_selection_never_imports_the_bare_corpus_module() -> None:
    """Wiring: the bare name resolves only after probes/ is put on sys.path by another module."""
    source = (ENGINE / "check_selection.py").read_text(encoding="utf-8")
    assert not re.search(r"^\s*(?:from\s+apply_corpus_fixes\s+import|import\s+apply_corpus_fixes\b)",
                         source, re.M), "check_selection imports apply_corpus_fixes by its bare name"
    assert "from probes import apply_corpus_fixes" in source


def test_corpus_fix_module_resolves_in_fresh_engine_process() -> None:
    code = ("import importlib.util, sys; "
            "sys.exit(0 if importlib.util.find_spec('probes.apply_corpus_fixes') else 1)")
    proc = subprocess.run(  # noqa: S603 - fixed argv, repo interpreter, no shell
        [sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": str(ENGINE)},
        capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0, "probes.apply_corpus_fixes is not importable from the engine path alone"


def test_registered_corpus_fix_selector_used_without_bare_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: with the bare module name unimportable, the registered selector still wins."""
    import check_selection as selection
    from probes import apply_corpus_fixes as corpus

    monkeypatch.setitem(sys.modules, "apply_corpus_fixes", None)  # bare import now raises ImportError
    monkeypatch.setattr(corpus, "corpus_fix_for",
                        lambda d: SimpleNamespace(selector="tests/test_corpus_fix.py::test_contract"))
    assert selection.fail_to_pass_ids({"repo": "swe", "fn": "f_one", "sha": ""}) == [
        "tests/test_corpus_fix.py::test_contract"]


# --- func_index -------------------------------------------------------------------------------------


def test_unreadable_repo_file_is_loud(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                      caplog: pytest.LogCaptureFixture) -> None:
    """T8-FUNC_INDEX-633: an unreadable file is still skipped, but one WARNING names it."""
    import func_index

    (tmp_path / "alpha.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    (tmp_path / "beta.py").write_text("def b():\n    return 2\n", encoding="utf-8")
    real_read = Path.read_text

    def flaky(self: Path, *args: object, **kwargs: object) -> str:
        if self.name == "beta.py":
            raise OSError("forced: unreadable file")
        return real_read(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", flaky)
    caplog.set_level(logging.WARNING)
    files = func_index.load_repo_files(tmp_path)
    assert sorted(files) == ["alpha.py"]
    assert _loud(caplog, "func_index"), f"beta.py dropped without a trace: {sorted(files)}"
    assert any("beta.py (OSError)" in record.getMessage() for record in caplog.records)


def test_readable_repo_loads_without_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Regression: the skip report never fires on a clean tree."""
    import func_index

    (tmp_path / "alpha.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    caplog.set_level(logging.WARNING)
    assert sorted(func_index.load_repo_files(tmp_path)) == ["alpha.py"]
    assert not _loud(caplog, "func_index")


@pytest.mark.parametrize(("unreadable", "truncated"), [(0, False), (5, False), (6, True)])
def test_unreadable_file_report_boundaries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                           caplog: pytest.LogCaptureFixture, unreadable: int,
                                           truncated: bool) -> None:
    """Boundary: one WARNING per call, at most five names listed, the count always exact."""
    import func_index

    (tmp_path / "keep.py").write_text("def k():\n    return 0\n", encoding="utf-8")
    bad = {f"bad_{index}.py" for index in range(unreadable)}
    for name in bad:
        (tmp_path / name).write_text("def b():\n    return 1\n", encoding="utf-8")
    real_read = Path.read_text

    def flaky(self: Path, *args: object, **kwargs: object) -> str:
        if self.name in bad:
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "forced: undecodable")
        return real_read(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", flaky)
    caplog.set_level(logging.WARNING)
    assert sorted(func_index.load_repo_files(tmp_path)) == ["keep.py"]
    messages = [r.getMessage() for r in caplog.records
                if r.levelno >= logging.WARNING and "func_index" in r.name]
    if not unreadable:
        assert messages == []
        return
    assert len(messages) == 1
    assert f"skipped {unreadable} unreadable file(s)" in messages[0]
    assert messages[0].count("(UnicodeDecodeError)") == min(unreadable, 5)
    assert messages[0].endswith(" ...") is truncated


# --- intent_oracle ----------------------------------------------------------------------------------


def test_unloadable_candidate_source_yields_no_signal() -> None:
    """Site 847 is DEGRADE-OK: a failed exec returns None, the documented 'no signal' value."""
    import intent_oracle as oracle

    assert oracle._load("def f(:\n    pass\n", "f") is None
    assert callable(oracle._load("def f(x):\n    return x\n", "f"))


def test_candidate_exception_is_a_typed_outcome() -> None:
    """Site 865 is ALREADY-COVERED: the exception type is carried on Outcome.exc."""
    import intent_oracle as oracle

    def boom() -> None:
        raise ValueError("forced")

    outcome = oracle.run_call(boom, SimpleNamespace(args=(), as_kwargs=dict))
    assert outcome.ok is False and outcome.exc == "ValueError"


def test_candidate_system_exit_is_a_typed_outcome() -> None:
    """Site 871 is ALREADY-COVERED (typed exc): SystemExit from candidate code is an outcome, not an exit."""
    import intent_oracle as oracle

    def leave() -> None:
        raise SystemExit(3)

    outcome = oracle.run_call(leave, SimpleNamespace(args=(), as_kwargs=dict))
    assert outcome.ok is False and outcome.exc == "SystemExit"


class _EngineTerminated(KeyboardInterrupt):
    """Shape of the engine's SIGTERM exception (a KeyboardInterrupt subclass)."""


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, _EngineTerminated])
def test_candidate_keyboard_interrupt_stops_the_run(interrupt: type[BaseException]) -> None:
    """Site 871 owner decision: Ctrl-C or engine SIGTERM during a candidate call re-raises, trace restored."""
    import intent_oracle as oracle

    def interrupted() -> None:
        raise interrupt()

    before = sys.gettrace()
    with pytest.raises(interrupt):
        oracle.run_call(interrupted, SimpleNamespace(args=(), as_kwargs=dict))
    assert sys.gettrace() is before


def test_broken_expectation_miner_is_loud(monkeypatch: pytest.MonkeyPatch,
                                          caplog: pytest.LogCaptureFixture) -> None:
    """T8-INTENT_ORACLE-773: a broken miner is skipped so the rest still run, and the skip is logged."""
    import intent_oracle as oracle

    def broken(fn: object, doc: str) -> list[object]:
        raise RuntimeError("forced: miner failed")

    monkeypatch.setattr(oracle, "_EXTRACTORS", (broken, *oracle._EXTRACTORS))
    caplog.set_level(logging.WARNING)
    source = "def f(x):\n    '''Return x.\n\n    >>> f(2)\n    2\n    '''\n    return x\n"
    assert oracle.extract(source), "the healthy miners must still contribute their expectations"
    assert _loud(caplog, "intent_oracle"), "a miner failed and the expectation list shrank silently"
    assert any("broken" in record.getMessage() for record in caplog.records)


def test_raising_expectation_check_is_loud(caplog: pytest.LogCaptureFixture) -> None:
    """T8-INTENT_ORACLE-894: a raising check still abstains (None), and the cause is logged."""
    import intent_oracle as oracle

    def boom(outs: list[object]) -> bool:
        raise RuntimeError("forced: check failed")

    caplog.set_level(logging.WARNING)
    verdicts = oracle.evaluate("def f(x):\n    return x\n", "f", [SimpleNamespace(calls=[], check=boom)])
    assert verdicts == {0: None}
    assert _loud(caplog, "intent_oracle"), "an authored check raised and was recorded as an abstention silently"


def test_discriminator_failure_reports_not_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    """Site 948 is DEGRADE-OK: the empty result carries applied=False and keeps every candidate."""
    import intent_oracle as oracle

    def refusing(*args: object) -> dict[str, object]:
        raise RuntimeError("forced: discrimination failed")

    monkeypatch.setattr(oracle, "_discriminate", refusing)
    cands = [object(), object()]
    result = oracle.discriminate(cands, "def f(x):\n    return x\n", ["def f(x):", "    return x"])
    assert result["applied"] is False
    assert result["survivors"] == cands and result["killed"] == []


# --- witness_replay ---------------------------------------------------------------------------------


def test_witness_row_runtime_error_is_a_typed_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Site 371 is DEGRADE-OK: a runtime error becomes the ('ERR', type) key, distinct from every OK key."""
    import witness_replay as wr

    def refusing(node: object, env: object) -> object:
        raise ValueError("forced")

    monkeypatch.setattr(wr, "safe_eval", refusing)
    assert wr._row_key(None, {}) == ("ERR", "ValueError")


def test_unrunnable_witness_call_returns_sentinel() -> None:
    """Sites 981 and 985 are DEGRADE-OK: both a raising call and an unprintable result return _UNRUNNABLE."""
    import witness_replay as wr

    def boom(*args: object) -> None:
        raise ValueError("forced")

    class Unprintable:
        def __repr__(self) -> str:
            raise ValueError("forced")

    assert wr._call_capture(boom, ()) is wr._UNRUNNABLE
    assert wr._call_capture(lambda *args: Unprintable(), ()) is wr._UNRUNNABLE


def test_missing_mutations_module_is_no_opinion(monkeypatch: pytest.MonkeyPatch) -> None:
    """T8-WITNESS_REPLAY-519 (DEGRADE-OK): mutations unavailable -> None, the documented "no opinion"."""
    import witness_replay as wr

    monkeypatch.setitem(sys.modules, "mutations", None)
    assert wr._span_facts(["x = a < b"]) is None


def test_span_facts_runtime_failure_is_no_opinion(monkeypatch: pytest.MonkeyPatch) -> None:
    """T8-WITNESS_REPLAY-523 (DEGRADE-OK): a span_facts crash -> None, so the witness does not apply."""
    import mutations
    import witness_replay as wr

    calls: list[object] = []

    def refusing(lines: object) -> object:
        calls.append(lines)
        raise RuntimeError("forced: span facts failed")

    monkeypatch.setattr(mutations, "span_facts", refusing)
    assert wr._span_facts(["x = a < b"]) is None
    assert calls == [["x = a < b"]]


def test_witness_probe_failure_is_loud(monkeypatch: pytest.MonkeyPatch,
                                       caplog: pytest.LogCaptureFixture) -> None:
    """T8-WITNESS_REPLAY-842: a probe bug still falls through to the boolean judges, and is logged."""
    import witness_replay as wr

    def refusing(*args: object) -> object:
        raise RuntimeError("forced: probe failed")

    monkeypatch.setattr(wr, "_await_witness", refusing)
    caplog.set_level(logging.WARNING)
    span = ["def f(x):", "    return x < 1"]
    assert wr.family_witness(span, (1, "    return x < 1", "    return x > 1")) == (False, False, "")
    assert _loud(caplog, "witness_replay"), "a witness probe crashed and the family silently did not apply"


def test_intent_oracle_discrimination_failure_is_loud(monkeypatch: pytest.MonkeyPatch,
                                                      caplog: pytest.LogCaptureFixture) -> None:
    """T8-WITNESS_REPLAY-1032: survivors stay (fail-open); the failure is logged and recorded on the rows."""
    import intent_oracle
    import witness_replay as wr

    def refusing(*args: object, **kwargs: object) -> object:
        raise RuntimeError("forced: discrimination failed")

    monkeypatch.setattr(intent_oracle, "discriminate", refusing)
    caplog.set_level(logging.WARNING)
    muts = [(1, "    return x", "    return -x"), (1, "    return x", "    return 0"),
            (1, "    return x", "    return x + 1")]
    survivors = [muts[0], muts[2]]
    evidence = [{"survived": True, "reason": "r0"}, {"survived": False, "reason": "r1"},
                {"survived": True, "reason": "r2"}]
    out_survivors, out_evidence = wr._apply_intent_oracle(
        ["def f(x):", "    return x"], muts, survivors, evidence)
    assert _loud(caplog, "witness_replay"), "the intent oracle failed and the certificate omitted it silently"
    assert out_survivors == survivors
    assert [row.get("intent_oracle_error") for row in out_evidence] == ["RuntimeError", None, "RuntimeError"]
    assert out_evidence[0]["intent_oracle"] is None and "intent_oracle" not in out_evidence[1]
    assert [row["reason"] for row in out_evidence] == ["r0", "r1", "r2"]
    assert "intent_oracle" not in evidence[0], "caller evidence rows must not be mutated in place"
    assert json.loads(json.dumps(out_evidence)) == out_evidence, "certificate rows must stay JSON round-trippable"


def test_mirror_prefilter_failure_is_loud(monkeypatch: pytest.MonkeyPatch,
                                          caplog: pytest.LogCaptureFixture) -> None:
    """T8-WITNESS_REPLAY-1151: a pre-filter bug keeps the candidate ('not a mirror'), and is logged."""
    import witness_replay as wr

    def refusing(old: str, new: str) -> str:
        raise RuntimeError("forced: pre-filter failed")

    monkeypatch.setattr(wr, "_default_idiom_mirror", refusing)
    caplog.set_level(logging.WARNING)
    assert wr.is_mirror_candidate("bool_flip", (3, "if x == y:", "if x != y:")) == (False, "")
    assert _loud(caplog, "witness_replay"), "the mirror pre-filter crashed and reported 'not a mirror' silently"
