"""Compatibility checks for the request-local projection memo."""
from __future__ import annotations

import copy
import json
import pickle

import pytest


def _body():
    payload = {"state": {"framefit5": {
        "owner_protocol": {
            "schema": "stateframe-framefit5-owner/1",
            "continuation_limits": "[CONTINUATION LIMITS | stateframe-continuation-limits/1] remaining=2",
            "evidence_pin": "[EVIDENCEPIN]current[/EVIDENCEPIN]",
        },
        "next_site": {"path": "pkg/a.py"},
        "cacheable_rows": [{
            "evidence_pin": "[EVIDENCEPIN]list-original[/EVIDENCEPIN]",
            "label": "stable",
        }],
        "last_verdict": {"line": "LAST FAIL"},
        "action_ledger": {"reads": [], "edits": []},
    }}}
    return {"messages": [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task: fix it"},
        {"role": "user", "content": "[StateFrame]\n" + json.dumps(
            payload, sort_keys=True, separators=(",", ":"))},
        {"role": "user", "content": "[Current source: pkg/a.py]\nrevision=sha256:123\nsource bytes"},
        {"role": "user", "content": "[CONTINUATION LIMITS | stateframe-continuation-limits/1] remaining=2"},
        {"role": "user", "content": "[WIRE VOLATILE TAIL]"},
    ], "tools": []}


def _type_tree(value):
    if type(value) is dict:
        return (dict, tuple((key, _type_tree(value[key])) for key in sorted(value)))
    if type(value) is list:
        return (list, tuple(_type_tree(child) for child in value))
    return type(value)


def _transaction(function):
    try:
        from kernel_working_set_cache import transaction
    except ImportError:
        return function()
    return transaction(function)()


def test_retained_authority_in_cacheable_list_fails_on_projection_miss_and_hit(monkeypatch):
    import kernel_cache_layout_candidate as candidate
    import kernel_stateframe_page_log as page_log

    original = page_log._partition_authority
    partitions = []

    def retain_authority_in_list(value, path=(), *, final=False):
        cacheable, fields = original(value, path, final=final)
        if final and not path:
            partitions.append(None)
            cacheable["state"]["framefit5"]["cacheable_rows"][0]["evidence_pin"] = (
                "[EVIDENCEPIN]retained[/EVIDENCEPIN]")
        return cacheable, fields

    monkeypatch.setattr(page_log, "_partition_authority", retain_authority_in_list)
    body = _body()
    errors = []

    def check_twice():
        for _ in range(2):
            try:
                candidate._project_final_cache_layout(copy.deepcopy(body))
            except ValueError as error:
                errors.append(str(error))
            else:
                errors.append(None)

    _transaction(check_twice)
    # The base has no memo and partitions twice. A memoized implementation
    # partitions once, then must apply the same fail-closed check to its hit.
    memoized = "projection_row" in candidate._project_final_cache_layout.__code__.co_names
    assert len(partitions) == (1 if memoized else 2)
    assert errors == ["final cache layout retained frame authority"] * 2


def test_inspect_layout_returns_fresh_plain_payloads_on_miss_and_hit(monkeypatch):
    import kernel_cache_layout_candidate as candidate
    import kernel_cache_layout_view as layout_view

    source = _body()
    expected = json.loads(source["messages"][2]["content"].split("\n", 1)[1])
    messages = candidate._checked_final_cache_projection(copy.deepcopy(source))["messages"]
    restored_calls = []
    original_restore = getattr(layout_view, "_restore_and_partition", None)
    if original_restore is not None:
        def counted_restore(*args, **kwargs):
            restored_calls.append(None)
            return original_restore(*args, **kwargs)

        monkeypatch.setattr(layout_view, "_restore_and_partition", counted_restore)

    def exercise():
        miss = layout_view.inspect_layout(messages)[1]
        hit = layout_view.inspect_layout(messages)[1]
        expected_types = _type_tree(expected)
        assert _type_tree(miss) == expected_types
        assert _type_tree(hit) == expected_types
        assert miss == expected and expected == miss
        assert hit == expected and expected == hit

        shallow = copy.copy(miss)
        deep = copy.deepcopy(miss)
        pickled = pickle.loads(pickle.dumps(miss))
        json_copy = json.loads(json.dumps(miss))
        for copied in (shallow, deep, pickled, json_copy):
            assert _type_tree(copied) == expected_types
            assert copied == expected and expected == copied

        deep["state"]["framefit5"]["action_ledger"]["reads"].append("deep-copy-only")
        after_deepcopy = layout_view.inspect_layout(messages)[1]
        assert after_deepcopy == expected

        miss["state"]["framefit5"]["action_ledger"]["reads"].append("caller-only")
        miss["state"]["framefit5"]["next_site"]["path"] = "caller-only.py"
        assert miss["state"]["framefit5"]["action_ledger"]["reads"] == ["caller-only"]
        assert miss["state"]["framefit5"]["next_site"]["path"] == "caller-only.py"
        later_hit = layout_view.inspect_layout(messages)[1]
        assert _type_tree(later_hit) == expected_types
        assert later_hit == expected

        hit["state"]["framefit5"]["action_ledger"]["edits"].append("caller-only")
        final_hit = layout_view.inspect_layout(messages)[1]
        assert _type_tree(final_hit) == expected_types
        assert final_hit == expected

    _transaction(exercise)
    if original_restore is not None:
        assert len(restored_calls) == 1
