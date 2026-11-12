"""A fitted body can need more final-layout headroom than its input did."""

from __future__ import annotations

import copy
import json
import math
import re
from types import SimpleNamespace

import pytest

import kernel_cache_layout_candidate as layout
import kernel_frame_allocation_capture as allocation
import kernel_request_budget as request_budget
import kernel_request_source_window as source_window
import kernel_stateframe_framefit5 as framefit5
from kernel_context import token_estimate
from kernel_request_diet import RequestTokenMeter
from kernel_cache_layout_view import CacheLayoutError, semantic_messages
from kernel_stateframe_final_message import AUTHORITY_OVERLAY_HEADER


def _body() -> dict[str, object]:
    limits = "[CONTINUATION LIMITS | stateframe-continuation-limits/1] remaining=2"
    evidence = "[EVIDENCEPIN]" + " ".join(f"proof{number:03d}" for number in range(60)) + "[/EVIDENCEPIN]"
    payload = {"state": {"framefit5": {
        "mode": framefit5.PRODUCT_MODE,
        "owner_protocol": {"schema": "stateframe-framefit5-owner/1",
                           "continuation_limits": limits, "evidence_pin": evidence},
        "next_site": {"path": "pkg/a.py"},
        "last_verdict": {"line": "LAST FAIL"},
        "action_ledger": {"reads": [], "edits": []},
    }}}
    return {"messages": [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task: fix it"},
        {"role": "user", "content": "[StateFrame]\n" + json.dumps(payload, sort_keys=True, separators=(",", ":"))},
        {"role": "user", "_request_budget_section": "optional",
         "content": "[COMPLETION OWNER]\n" + evidence},
        {"role": "user", "content": limits},
        {"role": "user", "content": "[WIRE VOLATILE TAIL]"},
    ], "tools": []}


def _tokenize(text: str) -> int:
    # Fixed subword approximation: four-character word pieces, separate digits
    # and punctuation. A sanity check of the sample gave 2.43 chars/token for
    # plain JSON and 2.02 for punctuation-heavy layout JSON. A direct check of
    # http://127.0.0.1:18593/tokenize was refused while the tunnel was down;
    # this test never needs that endpoint.
    return sum(math.ceil(len(part) / 4) if part[0].isalpha() else 1
               for part in re.findall(r"[A-Za-z_]+|\d+|[^\w\s]", text))


def test_final_layout_rechecks_growth_after_fit(monkeypatch, tmp_path):
    monkeypatch.delenv("STATEFRAME_FINAL_CACHE_LAYOUT", raising=False)
    monkeypatch.setenv("STATEFRAME_PAGE_LOG", "off")
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "off")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("value = 1\n")
    body = _body()
    meter = RequestTokenMeter(tokenizer=_tokenize, template_overhead_tokens=0)
    original_plain, method = meter.measure(body)
    original_layout, _ = meter.measure(layout._project_final_cache_layout(body))
    original_growth = original_layout - original_plain
    assert method == "tokenize" and original_growth > 0
    ceiling = original_layout - 80

    # The meter and endpoint cap normally come from the host connection. These
    # stubs bind the fixed tokenizer and a synthetic ceiling; fitting stays real.
    monkeypatch.setattr(request_budget, "_request_token_meter",
                        lambda *_args, **_kwargs: meter)
    monkeypatch.setattr(request_budget, "ENDPOINT_PROMPT_TOKEN_BUDGET", ceiling)
    monkeypatch.setattr(request_budget, "_lean_request_limits",
                        lambda *_args, **_kwargs: (533_504, ceiling))
    # The synthetic runtime has no workspace authorization record. Supply the
    # empty source proof; the real verifier otherwise reports that host fact as
    # unverified before the fit and layout can be exercised.
    monkeypatch.setattr(source_window, "verify_final_source_delivery",
                        lambda *_args, **_kwargs: {"status": "COMPLETE"})
    monkeypatch.setattr(source_window, "source_delivery_satisfied",
                        lambda receipt: receipt["status"] == "COMPLETE")

    def run_once():
        journal = SimpleNamespace(events=[])
        journal.emit = lambda kind, **fields: journal.events.append((kind, fields))
        capture = SimpleNamespace(
            request_id="growth-after-fit", journal=journal,
            _framefit5_semantic_messages=copy.deepcopy(body["messages"]),
            _framefit5_task_message=body["messages"][1],
            _request_source_closure_override={"units": [], "paths": [],
                "revisions": {}, "deferred_paths": [], "source_contents": {}},
            request_source_final_delivery=None,
        )
        runtime = SimpleNamespace(
            frame_ceiling=40_000, owner=SimpleNamespace(root=tmp_path),
            _code_bookmark_carry=(), _empirical_frame_sources={},
            _framefit_action_context={},
        )
        meter.bind_request(capture.request_id, capture.request_id)
        fitted = allocation.finalize_frame_candidate(
            runtime, capture, body, frame_counter=token_estimate)
        fitted_body = copy.deepcopy(fitted)
        fitted_plain, _ = allocation.final_request_tokens(fitted, meter, capture)
        fitted_projection = layout._project_final_cache_layout(fitted)
        _assert_single_order_wire_format(fitted_projection)
        fitted_layout, _ = allocation.final_request_tokens(fitted_projection, meter, capture)
        assert original_plain > ceiling - original_growth
        assert fitted_plain <= ceiling - original_growth
        assert len(fitted["messages"]) < len(body["messages"]), (
            f"original={original_plain} reserve={original_growth} ceiling={ceiling} "
            f"fitted={fitted_plain} layout={fitted_layout} "
            f"optional={fitted['messages'][3].get('content', '')[:80]}")
        assert fitted_layout - fitted_plain > original_growth
        assert fitted_layout - ceiling == 8
        assert fitted_plain > ceiling - (fitted_layout - fitted_plain)
        rendered = layout.render_final_cache_candidate(
            capture, runtime, fitted, meter=meter,
            inputs=capture._frame_candidate_inputs)
        rendered_tokens, _ = allocation.final_request_tokens(rendered, meter, capture)
        return (json.dumps(rendered, sort_keys=True, separators=(",", ":")).encode(),
                rendered, rendered_tokens, journal.events, fitted_plain, fitted_layout,
                fitted_body)

    first = run_once()
    second = run_once()
    wire, rendered, final_tokens, events, fitted_plain, fitted_layout, fitted_body = first
    assert wire == second[0]
    failures = [fields for kind, fields in events if kind == "final_cache_layout_failed_open"]
    assert len(failures) == 1, failures
    assert failures[0]["reason"] == "exceeds_measured_envelope"
    assert rendered == fitted_body
    assert all(message.get("name") != "stateframe_final_cache_1"
               for message in rendered["messages"])
    assert final_tokens == fitted_plain


def _assert_single_order_wire_format(body):
    marker = AUTHORITY_OVERLAY_HEADER + "\n"
    content = body["messages"][-1]["content"]
    prefix, separator, encoded = content.partition(marker)
    assert separator == marker
    overlay, consumed = json.JSONDecoder().raw_decode(encoded)
    order = overlay["order"]
    assert isinstance(order, dict)
    assert set(order) == {"prefix_length", "original_prefix_count", "frame_index", "moved"}

    malformed = copy.deepcopy(body)
    malformed_overlay = copy.deepcopy(overlay)
    malformed_overlay["order"] = [order[key] for key in (
        "prefix_length", "original_prefix_count", "frame_index", "moved")]
    malformed["messages"][-1]["content"] = (
        prefix + marker + json.dumps(malformed_overlay, sort_keys=True, separators=(",", ":"))
        + encoded[consumed:])
    with pytest.raises(CacheLayoutError, match="order metadata is malformed"):
        semantic_messages(malformed["messages"])


def _case_body(authority_copies, optional_context_words):
    body = _body()
    if authority_copies:
        frame = body["messages"][2]
        payload = json.loads(frame["content"].split("\n", 1)[1])
        payload["state"].update({
            f"item_{index}_authority": body["messages"][3]["content"]
            for index in range(authority_copies)
        })
        frame["content"] = "[StateFrame]\n" + json.dumps(
            payload, sort_keys=True, separators=(",", ":"))
    # Each row is smaller than the completion-owner row, so the first fit
    # discards that larger optional row while leaving these for a retry.
    body["messages"][3:3] = [
        {"role": "user", "_request_budget_section": "optional",
         "content": f"[Reference context {index}] " + " ".join(
             f"ctx{word:03d}" for word in range(word_count))}
        for index, word_count in enumerate(optional_context_words)
    ]
    return body


@pytest.mark.parametrize(
    ("authority_copies", "optional_context_words", "ceiling_rule", "expected_overage"),
    [
        pytest.param(0, (), "layout_minus_80", 8, id="existing-plus-8"),
        pytest.param(0, (5,), "layout_minus_94", 22, id="growth-plus-22"),
        pytest.param(10, (60, 60), "plain_minus_22", 300, id="growth-plus-300"),
        pytest.param(25, (66, 66, 66, 66), "plain_minus_20", 658, id="growth-plus-658"),
    ],
)
def test_final_layout_reserve_growth_and_order_wire_format(
        monkeypatch, tmp_path, authority_copies, optional_context_words,
        ceiling_rule, expected_overage):
    monkeypatch.delenv("STATEFRAME_FINAL_CACHE_LAYOUT", raising=False)
    monkeypatch.setenv("STATEFRAME_PAGE_LOG", "off")
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "off")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("value = 1\n")
    body = _case_body(authority_copies, optional_context_words)

    meter = RequestTokenMeter(tokenizer=_tokenize, template_overhead_tokens=0)
    original_plain, method = meter.measure(body)
    original_layout, _ = meter.measure(layout._project_final_cache_layout(body))
    original_growth = original_layout - original_plain
    assert method == "tokenize"
    if ceiling_rule == "layout_minus_80":
        ceiling = original_layout - 80
    elif ceiling_rule == "layout_minus_94":
        ceiling = original_layout - 94
    elif ceiling_rule == "plain_minus_22":
        ceiling = original_plain - 22
    else:
        assert ceiling_rule == "plain_minus_20"
        ceiling = original_plain - 20

    monkeypatch.setattr(request_budget, "_request_token_meter",
                        lambda *_args, **_kwargs: meter)
    monkeypatch.setattr(request_budget, "ENDPOINT_PROMPT_TOKEN_BUDGET", ceiling)
    monkeypatch.setattr(request_budget, "_lean_request_limits",
                        lambda *_args, **_kwargs: (533_504, ceiling))
    monkeypatch.setattr(source_window, "verify_final_source_delivery",
                        lambda *_args, **_kwargs: {"status": "COMPLETE"})
    monkeypatch.setattr(source_window, "source_delivery_satisfied",
                        lambda receipt: receipt["status"] == "COMPLETE")

    def run_once():
        journal = SimpleNamespace(events=[])
        journal.emit = lambda kind, **fields: journal.events.append((kind, fields))
        capture = SimpleNamespace(
            request_id="growth-after-fit", journal=journal,
            _framefit5_semantic_messages=copy.deepcopy(body["messages"]),
            _framefit5_task_message=body["messages"][1],
            _request_source_closure_override={"units": [], "paths": [],
                "revisions": {}, "deferred_paths": [], "source_contents": {}},
            request_source_final_delivery=None,
        )
        runtime = SimpleNamespace(
            frame_ceiling=40_000, owner=SimpleNamespace(root=tmp_path),
            _code_bookmark_carry=(), _empirical_frame_sources={},
            _framefit_action_context={},
        )
        meter.bind_request(capture.request_id, capture.request_id)
        allocation_passes = 0
        finalize = allocation.finalize_frame_candidate

        def record_allocation(*args, **kwargs):
            nonlocal allocation_passes
            allocation_passes += 1
            return finalize(*args, **kwargs)

        # A retry may call the full allocator instead of the late trim callback.
        with monkeypatch.context() as pass_patch:
            pass_patch.setattr(allocation, "finalize_frame_candidate", record_allocation)
            fitted = allocation.finalize_frame_candidate(
                runtime, capture, body, frame_counter=token_estimate)
            fitted_body = copy.deepcopy(fitted)
            fitted_plain, _ = allocation.final_request_tokens(fitted, meter, capture)
            fitted_projection = layout._project_final_cache_layout(fitted)
            fitted_layout, _ = allocation.final_request_tokens(
                fitted_projection, meter, capture)
            _assert_single_order_wire_format(fitted_projection)
            assert len(fitted["messages"]) < len(body["messages"])
            assert fitted_layout - fitted_plain > original_growth
            minimum_body = copy.deepcopy(fitted)
            minimum_body["messages"] = [
                message for message in minimum_body["messages"]
                if message.get("_request_budget_section") != "optional"
            ]
            protected_minimum, _ = meter.measure(minimum_body)
            minimum_layout, _ = meter.measure(
                layout._project_final_cache_layout(minimum_body))
            real_growth = minimum_layout - protected_minimum
            assert real_growth == fitted_layout - fitted_plain
            rendered = layout.render_final_cache_candidate(
                capture, runtime, fitted, meter=meter,
                inputs=capture._frame_candidate_inputs)
        assert 1 <= allocation_passes <= 3
        rendered_tokens, _ = allocation.final_request_tokens(rendered, meter, capture)
        return (json.dumps(rendered, sort_keys=True, separators=(",", ":")).encode(),
                rendered, rendered_tokens, journal.events, fitted_plain, fitted_layout,
                fitted_body, protected_minimum, real_growth, allocation_passes)

    first = run_once()
    second = run_once()
    (wire, rendered, final_tokens, events, fitted_plain, fitted_layout,
     fitted_body, protected_minimum, real_growth, allocation_passes) = first
    overage = fitted_layout - ceiling
    print(f"authority_copies={authority_copies} projected={fitted_layout} "
          f"ceiling={ceiling} projection_overshoot={overage:+d} "
          f"growth={real_growth} minimum={protected_minimum} "
          f"available={ceiling - real_growth}")
    assert overage == expected_overage
    assert wire == second[0]
    failures = [fields for kind, fields in events if kind == "final_cache_layout_failed_open"]
    if expected_overage == 8:
        assert ceiling - real_growth < protected_minimum
        assert len(failures) == 1, failures
        assert failures[0]["reason"] == "exceeds_measured_envelope"
        assert rendered == fitted_body
        assert all(message.get("name") != "stateframe_final_cache_1"
                   for message in rendered["messages"])
        assert final_tokens == fitted_plain
        return
    assert ceiling - real_growth >= protected_minimum
    assert not failures, (f"expected_pre_fix_overshoot={expected_overage:+d} "
                          f"projection_overshoot={overage:+d} plain={original_plain} "
                          f"prefit_growth={original_growth} fitted_plain={fitted_plain} "
                          f"fitted_growth={real_growth} "
                          f"projected={fitted_layout} ceiling={ceiling} failures={failures}")
    assert 2 <= allocation_passes <= 3
    assert rendered["messages"][-1]["name"] == "stateframe_final_cache_1"
    assert final_tokens <= ceiling
    _assert_single_order_wire_format(rendered)


@pytest.mark.parametrize(
    ("case", "authority_copies", "optional_context_words"),
    [
        pytest.param("existing-plus-8", 0, (), id="existing-plus-8"),
        pytest.param("growth-plus-22", 0, (5,), id="growth-plus-22"),
        pytest.param("growth-plus-300", 10, (60, 60), id="growth-plus-300"),
        pytest.param("growth-plus-658", 25, (66, 66, 66, 66), id="growth-plus-658"),
    ],
)
def test_final_layout_order_has_one_wire_format(
        case, authority_copies, optional_context_words):
    body = _case_body(authority_copies, optional_context_words)
    _assert_single_order_wire_format(layout._project_final_cache_layout(body))
    print(f"{case}: object order; list order rejected")
