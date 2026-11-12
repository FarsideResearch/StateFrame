"""User-visible engine diagnostics use the product prefix consistently."""

from pathlib import Path

from expectation_ledger import SuiteStatus, run_typed_suite
from kernel_liveness import CensusReport, _emit_receipt, render_report


ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"


def test_shipped_diagnostics_do_not_use_legacy_prefix():
    legacy_prefix = "[kernel-" + "wiring]"
    remaining = [
        path.relative_to(ENGINE).as_posix()
        for path in ENGINE.rglob("*.py")
        if path.name != "kernel_semantics.py" and legacy_prefix in path.read_text(encoding="utf-8")
    ]
    assert remaining == []


def test_expectation_suite_failure_uses_product_prefix(tmp_path, caplog):
    def fail(*_args, **_kwargs):
        raise RuntimeError("runner unavailable")

    result = run_typed_suite(tmp_path, run=fail)

    assert result.status is SuiteStatus.INFRA_ERROR
    assert "[stateframe] typed suite runner failed: RuntimeError" in caplog.text


def test_liveness_receipt_failure_and_report_use_product_prefix(capsys):
    def fail(_receipt):
        raise RuntimeError("sink unavailable")

    errors = _emit_receipt({}, receipt_path=None, receipt_sink=fail)
    report = CensusReport(
        run_id="test", status="INCOMPLETE", organs=(), digest="digest",
        receipt={}, errors=errors,
    )

    assert errors == ("receipt_sink_failed:RuntimeError",)
    assert "[stateframe] liveness census receipt sink failed: RuntimeError" in capsys.readouterr().out
    assert "[stateframe] census finalization errors: receipt_sink_failed:RuntimeError" in render_report(report)
