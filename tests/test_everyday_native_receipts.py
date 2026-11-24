from __future__ import annotations

import random
import re

import pytest

from stateframe.test_runners import (
    _parsed_passes,
    _runner_for_command,
    command_runs_tests,
    verification_evidence_for_tests,
)


GO_COMMAND = ("go", "test", "-v", "./...")
GO_MULTI_PACKAGE = """\
=== RUN   TestAddition
--- PASS: TestAddition (0.00s)
PASS
ok      example.test/math 0.003s
?       example.test/empty [no test files]
"""

CARGO_COMMAND = ("cargo", "test")
CARGO_MULTI_TARGET = """\
     Running unittests src/lib.rs (target/debug/deps/sample-0000000000000000)

running 2 tests
test arithmetic::adds ... ok
test arithmetic::subtracts ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
   Doc-tests sample

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
"""

_ANSI = re.compile(r"\x1b\[[\d;]*m")
_CARGO_RESULT = re.compile(
    r"^test result: (ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored; "
    r"(\d+) measured; (\d+) filtered out; finished in \S+$")
_CARGO_EVENT = re.compile(r"\btest\s+.+?\s+\.\.\.\s+.*?(ok|ignored|FAILED)(?:,\s*.+)?$")


def _base_parsed_passes(runner, output):
    """Copied base Go/Cargo receipt branches as a regression oracle."""
    lines = [line.strip() for line in _ANSI.sub("", output).splitlines() if line.strip()]
    if not lines:
        return 0
    if runner == "go":
        if (not re.fullmatch(r"ok\s+\S+(?:\s+.*)?", lines[-1])
                or any(re.match(r"(?:--- FAIL:|FAIL\b)", line) for line in lines)
                or len(lines) < 3 or lines[-2] != "PASS"):
            return 0
        start = max((index for index, line in enumerate(lines[:-2]) if re.match(
            r"(?:ok\s|FAIL\b|\?\s)", line)), default=-1) + 1
        package_lines = lines[start:-2]
        started = {match.group(1) for line in package_lines if (match := re.fullmatch(
            r"=== RUN\s+(\S+)", line))}
        return sum(bool(match and match.group(1) in started) for line in package_lines
                   if (match := re.fullmatch(r"--- PASS: (\S+)(?:\s+.*)?", line)))
    if runner != "cargo":
        return 0
    match = re.fullmatch(
        r"test result: ok\. (\d+) passed; (\d+) failed; "
        r"(\d+) ignored; (\d+) measured; (\d+) filtered out; finished in \S+", lines[-1]
    )
    if match is None:
        return 0
    return int(match.group(1)) if int(match.group(2)) == 0 else 0


def _ansi(line):
    return f"\x1b[32m{line}\x1b[0m"


def _cargo_unit(target, passed, *, ignored=0, failed=0, ansi=False):
    total = passed + ignored + failed
    lines = [f"Running tests/{target}", f"running {total} tests"]
    lines.extend(f"test {target}::case_{index:04d} ... ok" for index in range(passed))
    lines.extend(f"test {target}::skip_{index:04d} ... ignored" for index in range(ignored))
    lines.extend(f"test {target}::fail_{index:04d} ... FAILED" for index in range(failed))
    status = "FAILED" if failed else "ok"
    lines.append(
        f"test result: {status}. {passed} passed; {failed} failed; {ignored} ignored; "
        "0 measured; 0 filtered out; finished in 0.00s"
    )
    if ansi:
        lines = [_ansi(line) for line in lines]
    return "\n".join(lines) + "\n"


def _go_unit(passed, *, ansi=False):
    lines = []
    for index in range(passed):
        name = f"Test{index:04d}"
        lines.extend((f"=== RUN   {name}", f"--- PASS: {name} (0.00s)"))
    lines.extend(("PASS", "ok example.test/math 0.003s"))
    if ansi:
        lines = [_ansi(line) for line in lines]
    return "\n".join(lines) + "\n"


def _tail_4000(output):
    return output.encode("utf-8")[-4000:].decode("utf-8", errors="ignore")


def _visible_cargo_issue(output):
    lines = [line.strip() for line in _ANSI.sub("", output).splitlines() if line.strip()]
    if any(line.startswith("test result: FAILED") or
           ((match := _CARGO_EVENT.search(line)) and match.group(1) == "FAILED")
           for line in lines):
        return True
    unit_start = 0
    for end, line in enumerate(lines):
        if not line.startswith("test result:"):
            continue
        result = _CARGO_RESULT.fullmatch(line)
        if result is None:
            return True
        unit = lines[unit_start:end]
        unit_start = end + 1
        runs = [re.fullmatch(r"running (\d+) tests?", row) for row in unit]
        runs = [int(match.group(1)) for match in runs if match]
        events = [match.group(1) for row in unit
                  if (match := _CARGO_EVENT.search(row))]
        passed, failed, ignored = (int(result.group(index)) for index in (2, 3, 4))
        actual = {name: events.count(name) for name in ("ok", "ignored", "FAILED")}
        if runs:
            if (len(runs) != 1 or runs[0] != passed + failed + ignored
                    or len(events) != runs[0] or actual != {
                        "ok": passed, "ignored": ignored, "FAILED": failed
                    }):
                return True
        elif any(actual[name] > expected for name, expected in (
                ("ok", passed), ("ignored", ignored), ("FAILED", failed))):
            return True
    return False


def _random_cargo_output(rng):
    units = []
    for index in range(rng.randint(1, 3)):
        passed = rng.randint(0, 6)
        ignored = rng.randint(0, 2)
        total = passed + ignored
        lines = [f"Running tests/target_{index}", f"running {total} tests"]
        lines.extend(f"test t{index}::p{case} ... ok" for case in range(passed))
        lines.extend(f"test t{index}::i{case} ... ignored" for case in range(ignored))
        lines.append(
            f"test result: ok. {passed} passed; 0 failed; {ignored} ignored; "
            "0 measured; 0 filtered out; finished in 0.00s"
        )
        if rng.randrange(2):
            insert = rng.randrange(2, len(lines))
            lines.insert(insert, rng.choice((
                "warning: unused variable: `sample`",
                "test generated::slow has been running for over 60 seconds",
                "captured output between test events",
                "",
            )))
        event_indexes = [index for index, line in enumerate(lines)
                         if line.startswith("test t") and " ... " in line]
        if event_indexes:
            event_index = rng.choice(event_indexes)
            if rng.randrange(3) == 0:
                lines[event_index] = lines[event_index].replace(
                    " ... ", " ... printed without a newline")
            elif rng.randrange(2):
                lines[event_index] = "captured output: " + lines[event_index]
            elif lines[event_index].endswith(" ... ignored"):
                lines[event_index] += ", reason"
        if rng.randrange(2):
            lines = [_ansi(line) for line in lines]
        units.extend(lines)
    output = "\n".join(units) + "\n"
    return _tail_4000(output) if rng.randrange(2) else output


@pytest.mark.parametrize(("command", "output"), [
    (GO_COMMAND, GO_MULTI_PACKAGE),
    (CARGO_COMMAND, CARGO_MULTI_TARGET),
])
def test_successful_native_runs_preserve_positive_units(command, output):
    assert command_runs_tests("generic", command, set(), output=output)
    assert verification_evidence_for_tests(
        "tests", "generic", command, output, passed=True, declared=True
    ) == ("tests", "verified by tests", True)


@pytest.mark.parametrize(("command", "output"), [
    (GO_COMMAND, """\
PASS
ok example.test/math 0.003s
"""),
    (GO_COMMAND, """\
=== RUN   TestExpected
--- PASS: TestDifferent (0.00s)
PASS
ok example.test/math 0.003s
"""),
    (GO_COMMAND, """\
--- PASS: TestAddition (0.00s)
PASS
ok example.test/math 0.003s
"""),
    (GO_COMMAND, """\
--- PASS: TestAddition (0.00s)
=== RUN   TestAddition
PASS
ok example.test/math 0.003s
"""),
    (GO_COMMAND, """\
{"Action":"pass","Package":"example.test/math","Test":"TestAddition"}
"""),
    (CARGO_COMMAND, """\
running 2 tests
test arithmetic::adds ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
"""),
    (CARGO_COMMAND, """\
running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
"""),
])
def test_native_pseudo_footers_and_empty_units_do_not_verify(command, output):
    assert not command_runs_tests("generic", command, set(), output=output)
    assert verification_evidence_for_tests(
        "tests", "generic", command, output, passed=True, declared=True
    ) == ("command", None, False)


@pytest.mark.parametrize(("command", "output"), [
    (GO_COMMAND, GO_MULTI_PACKAGE + "--- FAIL: TestLater (0.00s)\nFAIL example.test/later [build failed]\n"),
    (CARGO_COMMAND, CARGO_MULTI_TARGET.replace(
        "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s",
        "test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s",
    )),
    (CARGO_COMMAND, CARGO_MULTI_TARGET + "test result: FAILED. incomplete footer\n"),
])
def test_failure_after_a_positive_unit_cannot_verify(command, output):
    assert not command_runs_tests("generic", command, set(), output=output)


def test_process_status_and_invocation_recognition_remain_authoritative():
    assert verification_evidence_for_tests(
        "tests", "generic", GO_COMMAND, GO_MULTI_PACKAGE, passed=False, declared=True
    ) == ("command", None, False)
    assert not command_runs_tests(
        "generic", ("go", "env", "test"), set(), output=GO_MULTI_PACKAGE
    )
    assert _runner_for_command("generic", ("go", "test", "./..."), None) == "go"
    assert _runner_for_command("generic", ("go", "test", "-test.v", "./..."), None) == "go"
    assert not command_runs_tests(
        "generic", ("go", "test", "./..."), set(), output="ok example.test/math 0.003s\n"
    )
    assert not command_runs_tests(
        "generic", ("go", "test", "-json", "./..."), set(),
        output='{"Action":"pass","Package":"example.test/math"}\n'
    )
    assert command_runs_tests(
        "generic", ("go", "test", "-v=true", "./..."), set(), output=GO_MULTI_PACKAGE
    )


@pytest.mark.parametrize(("command", "output"), [
    (GO_COMMAND, GO_MULTI_PACKAGE),
    (CARGO_COMMAND, CARGO_MULTI_TARGET),
])
def test_trailing_noise_does_not_change_go_but_cargo_uses_its_complete_footer(command, output):
    observed = command_runs_tests("generic", command, set(), output=output + "x" * 5000)
    assert observed is (command == CARGO_COMMAND)


@pytest.mark.parametrize("test_count", [20, 150, 300, 2000])
@pytest.mark.parametrize("unit_count", [1, 3])
def test_cargo_counts_survive_full_and_4000_byte_tail(test_count, unit_count):
    output = "".join(
        _cargo_unit(f"target_{index}", test_count, ansi=True)
        for index in range(unit_count)
    )
    expected = test_count * unit_count

    assert _base_parsed_passes("cargo", output) == test_count
    assert _parsed_passes("cargo", output) == expected
    assert expected >= _base_parsed_passes("cargo", output)

    tail = _tail_4000(output)
    base_count = _base_parsed_passes("cargo", tail)
    assert base_count == test_count
    assert _parsed_passes("cargo", tail) >= base_count
    if len(output.encode("utf-8")) <= 4000:
        assert _parsed_passes("cargo", tail) == expected


def test_go_large_verbose_receipts_preserve_base_counts_for_full_and_tail():
    output = _go_unit(300, ansi=True)
    assert _base_parsed_passes("go", output) == 300
    assert _parsed_passes("go", output) == 300

    tail = _tail_4000(output)
    base_count = _base_parsed_passes("go", tail)
    assert 0 < base_count < 300
    assert _parsed_passes("go", tail) >= base_count


def test_cargo_footer_only_tail_matches_base_without_hiding_visible_issues():
    footer_only = _cargo_unit("target_0", 3, ansi=True).splitlines()[-1] + "\n"
    assert _base_parsed_passes("cargo", footer_only) == 3
    assert _parsed_passes("cargo", footer_only) == 3

    contradictory = "\n".join([
        "test target::case_0000 ... ok",
        "test target::case_0001 ... ok",
        "test result: ok. 1 passed; 0 failed; 0 ignored; "
        "0 measured; 0 filtered out; finished in 0.00s",
    ])
    assert _parsed_passes("cargo", contradictory) == 0
    malformed_event = "\n".join([
        "test target::case_0000 ... uncertain",
        "test result: ok. 1 passed; 0 failed; 0 ignored; "
        "0 measured; 0 filtered out; finished in 0.00s",
    ])
    assert _base_parsed_passes("cargo", malformed_event) == 1
    assert _parsed_passes("cargo", malformed_event) == 1
    event_before_start = "test target::fake ... ok\n" + _cargo_unit("target", 1)
    assert _parsed_passes("cargo", event_before_start) == 0

    failed = _cargo_unit("target_0", 3, failed=1, ansi=True)
    assert _parsed_passes("cargo", failed) == 0


def test_cargo_earlier_positive_units_survive_a_later_empty_unit():
    output = (
        _cargo_unit("target_0", 2)
        + _cargo_unit("target_1", 1)
        + _cargo_unit("target_2", 0)
    )
    assert _parsed_passes("cargo", output) == 3

    mismatched_later = _cargo_unit("target_0", 2) + _cargo_unit("target_1", 2).replace(
        "test result: ok. 2 passed; 0 failed;", "test result: ok. 1 passed; 0 failed;", 1
    )
    assert _parsed_passes("cargo", mismatched_later) == 0


def test_cargo_generated_outputs_never_lose_base_evidence_without_visible_issues():
    rng = random.Random(271828)
    for _ in range(10000):
        output = _random_cargo_output(rng)
        base_count = _base_parsed_passes("cargo", output)
        parsed_count = _parsed_passes("cargo", output)
        assert not _visible_cargo_issue(output), output
        assert parsed_count >= base_count, output


def test_cargo_nocapture_warnings_slow_notices_and_ignored_reasons_are_noise():
    lines = _cargo_unit("target", 2, ignored=1).splitlines()
    lines[2] = "test target::case_0000 ... output without a trailing spaceok"
    lines[3] = "captured output: test target::case_0001 ... ok"
    lines[4] += ", reason"
    lines[3:3] = [
        "",
        "warning: unused variable: `sample`",
        "test target::slow has been running for over 60 seconds",
        "captured output between test events",
    ]
    lines.append("warning: compiler emitted a diagnostic")
    output = "\n".join(lines) + "\n"

    assert _base_parsed_passes("cargo", output) == 0
    assert _parsed_passes("cargo", output) == 2


def test_cargo_doc_test_names_with_spaces_are_counted():
    output = _cargo_unit("doc-tests", 1).replace(
        "test doc-tests::case_0000 ... ok", "test src/lib.rs - f (line 12) ... ok"
    )
    assert _parsed_passes("cargo", output) == 1


@pytest.mark.parametrize(("case", "output"), [
    ("visible FAILED event", "test target::case ... FAILED\n" +
     _cargo_unit("target", 1).splitlines()[-1] + "\n"),
    ("running/header/footer mismatch", "running 2 tests\n" +
     "test target::case ... ok\n" + _cargo_unit("target", 1).splitlines()[-1] + "\n"),
    ("event/footer category mismatch", "running 2 tests\n" +
     "test target::case ... ok\ntest target::skip ... ignored\n" +
     _cargo_unit("target", 2).splitlines()[-1] + "\n"),
    ("event/footer mismatch", "test target::a ... ok\ntest target::b ... ok\n" +
     _cargo_unit("target", 1).splitlines()[-1] + "\n"),
])
def test_cargo_head_minus_one_differences_are_only_visible_failures_or_contradictions(
        case, output):
    assert _base_parsed_passes("cargo", output) > 0, case
    assert _parsed_passes("cargo", output) == 0, case
    assert _visible_cargo_issue(output), case


@pytest.mark.parametrize(("body", "passed"), [
    ("running 5 tests\n.....\n", 5),
    ("running 1 test\ntest a ... hello\nok\n", 1),
    ("running 2 tests\nhello\ntest a ... ok\nno newlinetest b ... ok\n", 2),
    ("running 1 test\ntest a ... ok <0.001s>\n", 1),
    ("running 2 tests\ntest a ... ok\n", 2),
])
def test_cargo_hidden_event_lines_keep_the_footer_count(body, passed):
    footer = (f"test result: ok. {passed} passed; 0 failed; 0 ignored; 0 measured; "
              "0 filtered out; finished in 0.00s\n")
    assert _parsed_passes("cargo", body + footer) == passed
