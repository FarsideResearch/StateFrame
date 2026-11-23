"""Sentence spans keep dotted references intact and retain exact boundaries."""
from __future__ import annotations

from hashlib import sha256
from itertools import product
from types import SimpleNamespace

import pytest

from stateframe.engine import kernel_oracle_grounding as grounding
from stateframe.engine.kernel_oracle_grounding import extract_ticket_spans
from stateframe.profiles import DEFAULT, load


def _clauses(ticket: str) -> tuple[str, ...]:
    return tuple(clause for _start, _end, clause in extract_ticket_spans(ticket))


def _normalized(value: str) -> str:
    return " ".join(value.split())


def test_sentence_closing_abbreviation_splits_only_before_capitalized_sentence():
    cases = {
        "sentence-closing abbreviation": (
            "Use examples etc. Next.", ("Use examples etc.", "Next.")
        ),
        "continuation after abbreviation": (
            "Use examples etc. carefully.", ("Use examples etc. carefully.",)
        ),
        "abbreviation in continuation": (
            "Use examples etc. i.e. details remain.",
            ("Use examples etc. i.e. details remain.",),
        ),
        "i.e. before capital": ("Use examples i.e. Next.", ("Use examples i.e. Next.",)),
        "vs. before capital": ("Use examples vs. Next.", ("Use examples vs. Next.",)),
        "cf. before capital": ("Use examples cf. Next.", ("Use examples cf. Next.",)),
        "Mr. before capital": ("Ask Mr. Next.", ("Ask Mr. Next.",)),
    }
    mismatches = {
        name: (expected, _clauses(ticket))
        for name, (ticket, expected) in cases.items()
        if _clauses(ticket) != expected
    }

    assert not mismatches, f"unexpected abbreviation boundaries: {mismatches}"


def test_internal_period_table_does_not_create_false_sentence_boundaries():
    code_block = "```text\nmodule.part = 1\n```"
    cases = {
        "module path": ("Call module.submodule.run() now.", ("Call module.submodule.run() now.",)),
        "method call": ("Call object.method() now.", ("Call object.method() now.",)),
        "version": ("Use version 3.14.2 now.", ("Use version 3.14.2 now.",)),
        "file name": ("Keep archive.tar.gz intact.", ("Keep archive.tar.gz intact.",)),
        "abbreviations": ("Use e.g. JSON, i.e. structured data.",
                          ("Use e.g. JSON, i.e. structured data.",)),
        "other common abbreviations": (
            "Use formats incl. JSON, a.k.a. structured output.",
            ("Use formats incl. JSON, a.k.a. structured output.",),
        ),
        "URL": ("Open https://docs.example.test/a.b now.",
                ("Open https://docs.example.test/a.b now.",)),
        "double quoted text": ('Store "ready." as text.', ('Store "ready." as text.',)),
        "single quoted text": ("Store 'ready.' as text.", ("Store 'ready.' as text.",)),
        "backticked text": ("Store `file.part` as text.", ("Store `file.part` as text.",)),
        "fenced code": (f"Read this:\n{code_block}\nThen stop.",
                        ("Read this:", code_block, "Then stop.")),
        "non-ASCII": ("Keep café 東京 intact.", ("Keep café 東京 intact.",)),
        "CRLF soft wrap": ("Keep this sentence\r\non one line.",
                           ("Keep this sentence on one line.",)),
        "empty": ("", ()),
    }
    mismatches = {
        name: (expected, _clauses(ticket))
        for name, (ticket, expected) in cases.items()
        if _clauses(ticket) != expected
    }

    assert not mismatches, f"unexpected clause boundaries: {mismatches}"


def test_terminal_table_keeps_sentence_endings_and_closing_quotes_together():
    cases = {
        "period": ("Use streams. Reject stale rows.", ("Use streams.", "Reject stale rows.")),
        "question mark": ("Ready? Continue.", ("Ready?", "Continue.")),
        "ellipsis": ("Wait... Then proceed.", ("Wait...", "Then proceed.")),
        "double quote": ('Return "ready." Then stop.', ('Return "ready."', "Then stop.")),
        "single quote": ("Return 'ready.' Then stop.", ("Return 'ready.'", "Then stop.")),
        "backtick": ("Return `ready.` Then stop.", ("Return `ready.`", "Then stop.")),
        "curly quote": ("Return “ready.” Then stop.", ("Return “ready.”", "Then stop.")),
        "curly single quote": ("Return ‘ready.’ Then stop.", ("Return ‘ready.’", "Then stop.")),
        "unbalanced backtick": ("Keep `an inline marker. Then stop.",
                                ("Keep `an inline marker.", "Then stop.")),
        "apostrophe": ("Don't stop here. Continue.", ("Don't stop here.", "Continue.")),
        "post-terminal closers": (
            "Return the value?!)]}'\" Next request.",
            ("Return the value?!)]}'\"", "Next request."),
        ),
    }
    mismatches = {
        name: (expected, _clauses(ticket))
        for name, (ticket, expected) in cases.items()
        if _clauses(ticket) != expected
    }

    assert not mismatches, f"unexpected terminal handling: {mismatches}"


def test_span_offsets_match_exact_source_characters_across_unicode_and_crlf():
    ticket = "Call naïve.module.part().\r\nThen stop."
    first_end = ticket.index("\r\n")
    second_start = first_end + len("\r\n")
    expected = (
        (0, first_end, ticket[:first_end]),
        (second_start, len(ticket), ticket[second_start:]),
    )

    assert extract_ticket_spans(ticket) == expected


def test_large_dotted_reference_is_processed_as_one_sentence():
    path = ".".join(["module"] * 14_286) + ".run()"
    ticket = f"Call {path}."
    assert len(ticket) >= 100_000

    spans = extract_ticket_spans(ticket)

    assert len(spans) == 1
    assert spans[0] == (0, len(ticket), ticket)


def test_generated_spans_are_ordered_non_overlapping_and_reproduce_sentences():
    references = (
        "module.client.run()",
        "version 4.11.2",
        "archive.tar.gz",
        "e.g. structured output",
        "naïve.élève",
        "https://docs.example.test/a.b",
    )
    prefixes = ("Use café", "東京では")
    newlines = ("\n", "\r\n")
    failures: list[str] = []

    for prefix, reference, newline in product(prefixes, references, newlines):
        first = f"{prefix} {reference} as input."
        ticket = f"{first}{newline}Continue."
        spans = extract_ticket_spans(ticket)
        cursor = 0
        ordered = True
        exact = True
        for start, end, clause in spans:
            if not (cursor <= start < end <= len(ticket)):
                ordered = False
            if ticket[cursor:start].strip():
                exact = False
            if _normalized(ticket[start:end]) != _normalized(clause):
                exact = False
            cursor = end
        if ticket[cursor:].strip():
            exact = False
        reproduced = _normalized(" ".join(clause for _start, _end, clause in spans))
        if not ordered or not exact or reproduced != _normalized(ticket):
            failures.append(f"{reference!r}: ordered={ordered}, exact={exact}, reproduced={reproduced!r}")
        if tuple(clause for _start, _end, clause in spans) != (first, "Continue."):
            failures.append(f"{reference!r}: incorrect sentence clauses")

    assert not failures, "span property violations: " + "; ".join(failures[:6])


@pytest.mark.parametrize("reference", (
    r"..\src\settings.py", "src/v1./settings.py", "cafe\u0301.module.run()",
))
def test_relative_paths_and_decomposed_names_keep_internal_periods(reference):
    first = f"Read {reference}."
    ticket = first + "\r\nThen stop."

    assert extract_ticket_spans(ticket) == (
        (0, len(first), first),
        (len(first) + 2, len(ticket), "Then stop."),
    )


@pytest.mark.parametrize("delimiters", (("“", "”"), ("‘", "’"), ("`", "`")))
def test_url_terminal_period_is_recognized_before_paired_delimiters(delimiters):
    opening, closing = delimiters
    first = f"Return {opening}https://example.test/path.{closing}"

    assert _clauses(first + " Then stop.") == (first, "Then stop.")


def test_url_query_marker_before_a_comma_is_not_a_sentence_end():
    ticket = "Use https://example.test/path?, then stop."

    assert _clauses(ticket) == (ticket,)


def test_balanced_url_parenthesis_keeps_internal_period():
    first = "Read https://example.test/fn(arg.) now."

    assert _clauses(first + " Then stop.") == (first, "Then stop.")


@pytest.mark.parametrize("first, quoted", (
    ("Use users' data.", "Return 'ready. pending' as text."),
    ('Use 6" screws.', 'Return "ready. pending" as text.'),
))
def test_word_suffix_marks_do_not_open_quotes_across_sentences(first, quoted):
    assert _clauses(f"{first} {quoted} Next.") == (first, quoted, "Next.")


@pytest.mark.parametrize("delimiters", (("“", "”"), ('"', '"')))
def test_adjacent_opening_quote_stays_with_its_own_sentence(delimiters):
    opening, closing = delimiters
    first = f"Say {opening}Ready.{closing}"
    second = f"{opening}Continue.{closing}"

    assert _clauses(first + second) == (first, second)


def test_escaped_backtick_does_not_open_inline_code():
    first = r"Show \` as a character."
    second = "Store `ready. pending` as text."

    assert _clauses(f"{first} {second}") == (first, second)


@pytest.mark.parametrize("member", (
    "$app.$store.run()", "this.#handler()", "client.open().close()", "items[0].name",
))
def test_member_access_does_not_end_a_sentence(member):
    first = f"Call {member}."

    assert _clauses(first + " Then stop.") == (first, "Then stop.")


@pytest.mark.parametrize("path", ("dir/..", r"C:\work\.", "../../../."))
def test_terminal_dot_path_segments_preserve_exact_source_spans(path):
    first = f"Resolve {path} before writing."
    ticket = first + "\r\nThen stop."

    assert extract_ticket_spans(ticket) == (
        (0, len(first), first),
        (len(first) + 2, len(ticket), "Then stop."),
    )
    assert _clauses(f"Resolve {path}! Then stop.") == (f"Resolve {path}!", "Then stop.")


def test_parent_path_followed_by_sentence_period():
    first = "Resolve dir/..."

    assert _clauses(first + " Then stop.") == (first, "Then stop.")


def test_bracket_inside_closing_quote_stays_with_sentence():
    first = "Report “The file is missing (check the path.)”"

    assert _clauses(first + " Then stop.") == (first, "Then stop.")


def test_url_apostrophe_does_not_open_a_quote():
    first = "Use https://example.test/'path."
    second = "Then store 'ready. pending' as text."

    assert _clauses(f"{first} {second} Next.") == (first, second, "Next.")


@pytest.mark.parametrize("opening, closing", (("«", "»"), ("「", "」")))
def test_directional_quotes_keep_internal_and_terminal_periods(opening, closing):
    first = f"Store {opening}ready. pending{closing} as text."
    second = f"Return {opening}ready.{closing}"

    assert _clauses(f"{first} {second} Next.") == (first, second, "Next.")


def test_quote_spacing_stays_inside_the_terminal_source_span():
    first = "Report «\u202fReady.\u202f»"
    ticket = first + " Then stop."

    assert extract_ticket_spans(ticket) == (
        (0, len(first), "Report « Ready. »"),
        (len(first) + 1, len(ticket), "Then stop."),
    )


def test_quoted_terminal_can_precede_an_uncased_sentence():
    first = "Report “期限切れ.”"
    second = "東京の担当へ連絡する."

    assert _clauses(f"{first} {second}") == (first, second)


def test_no_is_a_response_when_it_does_not_introduce_a_number():
    first = "Answer “No.”"
    second = "Check No. 4 only."

    assert _clauses(f"{first} {second}") == (first, second)


@pytest.mark.parametrize("delimiter", ('"', '`'))
def test_long_internal_punctuation_run_has_linear_scan_cost(delimiter):
    class CountedText(str):
        reads = 0

        def __getitem__(self, index):
            if isinstance(index, int):
                self.reads += 1
            return super().__getitem__(index)

    first = f"Store {delimiter}ready{'!?' * 512} pending{delimiter} as text."
    text = CountedText(first + " Then stop.")

    assert grounding._sentence_ranges(text) == ((0, len(first)), (len(first), len(text)))
    assert text.reads < 20 * len(text)


def _clause_request(ticket):
    from kernel_prompt_compact import compact_clause_block, set_collision_scope
    from kernel_request_evidence_closure import serialized_request_bytes
    from kernel_stateframe_f4 import HANDOFF_HEADER, compose_messages
    from kernel_stateframe_get_sure import build_table, render_model_block

    prefix = [{"role": "system", "content": "Follow the task."},
              {"role": "user", "content": f"Task: {ticket}"}]
    block = "\n".join(line for line in render_model_block(build_table(ticket)).splitlines()
                      if not line.startswith("Your last conclusion:"))
    set_collision_scope([*(message["content"] for message in prefix), block])
    block = compact_clause_block(block, task_present=True)
    messages = compose_messages(
        prefix=prefix, path="", working="", state={},
        clause_blocks=[{"role": "user", "content": block}],
        handoff={"role": "user", "content": f"{HANDOFF_HEADER}\nContinue."},
        exchanges=[({"role": "assistant", "content": "Reading."},
                    {"role": "tool", "content": "OK"})],
    )
    return messages, serialized_request_bytes(messages)


def test_clause_request_changes_only_the_generated_user_message(monkeypatch):
    import kernel_prompt_compact
    from kernel_request_evidence_closure import serialized_request_bytes
    from kernel_stateframe_f4 import CLAUSE_HEADER

    monkeypatch.setattr(kernel_prompt_compact, "_COLLIDING", set())
    monkeypatch.setattr(kernel_prompt_compact, "_PENDING", {})
    first = 'Return “https://example.test/path.”'
    ticket = first + " Then stop."
    unchanged_ticket = "Use streams. Reject stale rows."
    profile = load(DEFAULT)
    switches = ("STATEFRAME_ORACLE_GROUNDING", "STATEFRAME_POST_GREEN_WORK_MENU")
    monkeypatch.setenv("STATEFRAME_TOKENLAB_GET_SURE_HINT_DROP",
                       profile["STATEFRAME_TOKENLAB_GET_SURE_HINT_DROP"])
    for switch in switches:
        monkeypatch.setenv(switch, "off")
    inactive, inactive_wire = _clause_request(ticket)
    _, inactive_ordinary_wire = _clause_request(unchanged_ticket)

    for switch in switches:
        assert profile[switch] == "on"
        monkeypatch.setenv(switch, profile[switch])
    active, active_wire = _clause_request(ticket)
    _, active_ordinary_wire = _clause_request(unchanged_ticket)

    # Captured from the same request composition at 7d706332.
    assert sha256(inactive_wire).hexdigest() == "44f8cd02589a05b7652642c876c67b4eac9c2ffc7f24ae3704baa74560d5b046"
    assert sha256(active_wire).hexdigest() != "fb39e866ddd8abd9037a426ab66a1359b97b008a1444eeeb69e64a85335a7dfb"
    assert active_ordinary_wire == inactive_ordinary_wire
    assert sha256(active_ordinary_wire).hexdigest() == "837c3a5122f6b27100fc4a1e8e6524f42bd62ac6af3562148cf969b32568fd0b"
    unchanged_wire = serialized_request_bytes([*active[:2], *active[3:]])
    assert sha256(unchanged_wire).hexdigest() == "4452237ca9ae3315b26919ae4a01c502b928a0f6706471512554b053d7c23517"
    assert active[:2] == inactive[:2]
    assert active[3:] == inactive[3:]
    assert active[2]["role"] == "user"
    assert active[2]["content"].startswith(CLAUSE_HEADER)
    assert f"C1 UNPROBED: {first}\n" in active[2]["content"]
    assert active[2]["content"].endswith("C2 UNPROBED: Then stop.")


def test_sentence_closing_abbreviation_changes_only_the_appended_findings_message(monkeypatch):
    from kernel_request_evidence_closure import serialized_request_bytes
    from kernel_stateframe_runtime import build_working_view

    ticket = "Use examples etc. Next."
    prefix = [
        {"role": "system", "content": "Follow the task."},
        {"role": "user", "content": f"Task: {ticket}"},
        {"role": "assistant", "content": "Reading."},
        {"role": "tool", "content": "OK"},
    ]
    profile = load(DEFAULT)
    for switch in ("STATEFRAME_FRAME_CLARITY", "STATEFRAME_DISPLAY_TRUTH"):
        monkeypatch.setenv(switch, profile[switch])
    monkeypatch.setenv("STATEFRAME_EDIT_GUARD", "off")
    frame = {"task_brief": ticket}
    monkeypatch.setenv("STATEFRAME_ORACLE_GROUNDING", "off")
    inactive = build_working_view("findings", frame)
    monkeypatch.setenv("STATEFRAME_ORACLE_GROUNDING", profile["STATEFRAME_ORACLE_GROUNDING"])
    active = build_working_view("findings", frame)
    inactive_messages = [*prefix, inactive]
    active_messages = [*prefix, active]

    assert serialized_request_bytes(active_messages) != serialized_request_bytes(inactive_messages)
    assert active_messages[:-1] == inactive_messages[:-1] == prefix
    assert active["role"] == "user"
    assert 'C1: "Use examples etc."' in active["content"]
    assert 'C2: "Next."' in active["content"]


def test_grounding_changes_only_the_optional_history_state_tail(monkeypatch):
    from kernel_request_evidence_closure import serialized_request_bytes
    from kernel_stateframe_hstate_plus import apply_tail, tail_blocks

    first = "Call $app.$store.run()."
    ticket = first + " Then stop."
    prefix = [
        {"role": "system", "content": "Follow the task."},
        {"role": "user", "content": f"Task: {ticket}"},
        {"role": "assistant", "content": "Reading."},
        {"role": "tool", "content": "OK"},
    ]
    profile = load(DEFAULT)
    assert profile["STATEFRAME_HSTATE_PLUS"] == "off"
    monkeypatch.setenv("STATEFRAME_REQUIREMENT_ALIGNMENT", profile["STATEFRAME_REQUIREMENT_ALIGNMENT"])

    def messages(grounding_mode, *, tail_enabled):
        for switch in ("STATEFRAME_ORACLE_GROUNDING", "STATEFRAME_POST_GREEN_WORK_MENU"):
            monkeypatch.setenv(switch, grounding_mode)
        runtime = SimpleNamespace(requirements_ledger={"ticket": ticket})
        blocks = tail_blocks(runtime, {}, ())
        return apply_tail(prefix, blocks, enabled=tail_enabled, placement="append").messages

    inactive = messages("off", tail_enabled=True)
    active = messages("on", tail_enabled=True)
    absent = messages("on", tail_enabled=False)
    # Captured from the same request composition at 7d706332.
    assert sha256(serialized_request_bytes(inactive)).hexdigest() == "46106bf1430e62c5092df4ca16839cf44a083601ba93197981267140a8ba30e1"
    assert sha256(serialized_request_bytes(active)).hexdigest() != "0ae61b6a751abbef91794c8172c04fb556832fe747584f6e96812ae538920e09"
    assert sha256(serialized_request_bytes(absent)).hexdigest() == "1d6be64b73b790da805ceb828055e7065b4d7cecf712bad8dd4386a7b2337f32"
    assert active[:-1] == inactive[:-1] == absent == tuple(prefix)
    assert active[-1]["role"] == "user"
    assert f"C1 | targets: target unresolved | {first}\n" in active[-1]["content"]
    assert "C2 | targets: target unresolved | Then stop.\n" in active[-1]["content"]


def test_grounding_changes_the_appended_working_view_findings(monkeypatch):
    from kernel_request_evidence_closure import serialized_request_bytes
    from kernel_stateframe_runtime import build_working_view

    first = "Limit $app.$store.run() to 10 seconds."
    ticket = first + " Then stop."
    prefix = [
        {"role": "system", "content": "Follow the task."},
        {"role": "user", "content": f"Task: {ticket}"},
        {"role": "assistant", "content": "Reading."},
        {"role": "tool", "content": "OK"},
    ]
    frame = {"task_brief": ticket}
    profile = load(DEFAULT)
    for switch in ("STATEFRAME_FRAME_CLARITY", "STATEFRAME_DISPLAY_TRUTH"):
        monkeypatch.setenv(switch, profile[switch])
    monkeypatch.setenv("STATEFRAME_EDIT_GUARD", "off")
    monkeypatch.setenv("STATEFRAME_ORACLE_GROUNDING", "off")
    inactive = build_working_view("findings", frame)
    monkeypatch.setenv("STATEFRAME_ORACLE_GROUNDING", profile["STATEFRAME_ORACLE_GROUNDING"])
    active = build_working_view("findings", frame)

    # Captured from the same request composition at 7d706332.
    inactive_wire = serialized_request_bytes([*prefix, inactive])
    active_wire = serialized_request_bytes([*prefix, active])
    assert sha256(inactive_wire).hexdigest() == "6e3e577f0f5113629cb620d5129b61e33716ee1e291126fcc6dc2b821733558b"
    assert sha256(active_wire).hexdigest() != "a118130fb26de7ce82d55fe0353bbdba1e0a3328bb9ef843df8835e6555a91cc"
    assert active["role"] == "user"
    assert f'C1: "{first}"\nC2: "Then stop."\n' in active["content"]
