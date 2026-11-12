"""Exact token admission must use the same envelope for protected authority."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from kernel_frame_allocation_capture import FrameFit5ImpossibleMinimum
from kernel_request_budget import HARD_MAX_REQUEST_BYTES
from kernel_stateframe_capacity import protected_bytes, protected_limit
from kernel_stateframe_action_ledger import build_action_context
from tests import test_narrow_stall as narrow


class ExactMeter:
    _request_id = None
    measure_method = "tokenize"
    template_overhead_tokens = 0

    def measure(self, _body):
        return 1_000, "tokenize"

    def tokenizer_identity(self, _body):
        return "exact-fixture"

    def journal_fields(self, _method):
        return {"tokenizer_calls": 0}

    def bind_measurement(self, _body, **_kwargs):
        return SimpleNamespace()


class OutageMeter(ExactMeter):
    def __init__(self):
        self.calls = 0

    def measure(self, _body):
        self.calls += 1
        return (1_000, "tokenize") if self.calls == 1 else (1_000, "calibrated_bytes")


def test_exact_counted_protected_minimum_uses_verified_byte_envelope(tmp_path, monkeypatch):
    import kernel_request_budget

    root = tmp_path / "workspace"
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pkg/__init__.py").write_text("VALUE = 1\n")
    (root / "tests/test_app.py").write_text("def test_app(): assert True\n")
    messages = [{"role": "user", "content": "Task: Fix value.\n" + "x" * 90_000}]
    runtime = narrow._runtime_with_product(root)
    runtime.frame_ceiling = 100_000
    context = build_action_context(messages, workspace_root=root)
    context["journal_receipt_records"] = []
    monkeypatch.setattr(kernel_request_budget, "_request_token_meter",
                        lambda *_args, **_kwargs: ExactMeter())

    try:
        body = narrow._qualified_fixture_request(
            runtime, {"verification": {"revision": "current", "status": "FAIL"}},
            messages, context, baseline=False, monkeypatch=monkeypatch,
        )
    except FrameFit5ImpossibleMinimum as error:
        detail = error.detail
        assert detail["measure_method"] == "tokenize"
        assert detail["prompt_tokens"] <= detail["max_prompt_tokens"]
        assert detail["request_bytes"] <= detail["max_request_bytes"]
        assert detail["protected_request_bytes"] > detail["protected_byte_limit"]
        pytest.fail("exact-counted request refused by the legacy protected byte limit")

    assert protected_bytes(body) > HARD_MAX_REQUEST_BYTES
    assert protected_bytes(body) <= protected_limit(body, exact_tokenizer=True)


def test_frozen_exact_label_does_not_admit_after_tokenizer_outage(tmp_path, monkeypatch):
    import kernel_request_budget

    root = tmp_path / "workspace"
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pkg/__init__.py").write_text("VALUE = 1\n")
    (root / "tests/test_app.py").write_text("def test_app(): assert True\n")
    messages = [{"role": "user", "content": "Task: Fix value.\n" + "x" * 90_000}]
    runtime = narrow._runtime_with_product(root)
    runtime.frame_ceiling = 100_000
    context = build_action_context(messages, workspace_root=root)
    context["journal_receipt_records"] = []
    meter = OutageMeter()
    monkeypatch.setattr(kernel_request_budget, "_request_token_meter",
                        lambda *_args, **_kwargs: meter)

    with pytest.raises(FrameFit5ImpossibleMinimum) as error:
        narrow._qualified_fixture_request(
            runtime, {"verification": {"revision": "current", "status": "FAIL"}},
            messages, context, baseline=False, monkeypatch=monkeypatch,
        )

    assert meter.calls > 1
    assert error.value.detail["measure_method"] == "tokenize"


def test_below_cap_request_keeps_legacy_bytes(tmp_path, monkeypatch):
    import kernel_request_budget

    root = tmp_path / "workspace"
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pkg/__init__.py").write_text("VALUE = 1\n")
    (root / "tests/test_app.py").write_text("def test_app(): assert True\n")
    messages = [{"role": "user", "content": "Task: Fix value.\n" + "x" * 1_000}]
    runtime = narrow._runtime_with_product(root)
    runtime.frame_ceiling = 100_000
    context = build_action_context(messages, workspace_root=root)
    context["journal_receipt_records"] = []
    monkeypatch.setattr(kernel_request_budget, "_request_token_meter",
                        lambda *_args, **_kwargs: ExactMeter())

    body = narrow._qualified_fixture_request(
        runtime, {"verification": {"revision": "current", "status": "FAIL"}},
        messages, context, baseline=False, monkeypatch=monkeypatch,
    )

    assert protected_bytes(body) < HARD_MAX_REQUEST_BYTES
    wire = json.dumps(body).replace(str(root), "<ROOT>").encode()
    assert hashlib.sha256(wire).hexdigest() == "6862d8ec24e03d8cdd74d090736da5bab6867896a71e7df5839e37e0b2e2b4fa"
