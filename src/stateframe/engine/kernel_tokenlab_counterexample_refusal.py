"""Concrete evidence for a refused ``done`` decision."""
from __future__ import annotations

import os
import json
import copy
from collections.abc import Mapping, Sequence


MODE_ENV = "STATEFRAME_TOKENLAB_COUNTEREXAMPLE_REFUSAL"
SCHEMA = "stateframe-v19-counterexample-refusal/1"
EVENT = "tokenlab_v19_counterexample_refusal"
PRODUCER = "kernel_tokenlab_counterexample_refusal"


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(runtime: object | None = None) -> bool:
    configured = getattr(runtime, "stateframe_tokenlab_counterexample_refusal_mode", None)
    if configured is not None:
        return str(configured).strip().lower() == "on"
    return mode() == "on"


def _row(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _basis(row: Mapping[str, object], check: Mapping[str, object]) -> str:
    direct = [str(row.get("oracle_basis") or row.get("refusal_oracle_basis") or "").upper().replace("-", "_"),
              str(check.get("oracle_basis", "") or "").upper().replace("-", "_")]
    if any(value in {"PUBLIC_TEST", "PUBLIC_EXAMPLE", "PUBLIC_INPUT"} for value in direct):
        return "PUBLIC_EXAMPLE"
    if "TICKET" in direct:
        return "TICKET"
    values = [row.get("oracle_basis"), check.get("oracle_basis")]
    provenance = check.get("provenance")
    if isinstance(provenance, Mapping):
        values.extend((provenance.get("oracle_basis"), provenance.get("oracle_source")))
        sources = provenance.get("oracle_sources")
        if isinstance(sources, Sequence) and not isinstance(sources, (str, bytes, bytearray)):
            values.extend(sources)
    selection = check.get("selection")
    if isinstance(selection, Sequence) and not isinstance(selection, (str, bytes, bytearray)):
        values.extend(selection)
    text = " ".join(str(value or "").upper() for value in values)
    if "PUBLIC_EXAMPLE" in text or "PUBLIC EXAMPLE" in text or "PUBLIC_INPUT" in text:
        return "PUBLIC_EXAMPLE"
    if "TICKET" in text or "BRIEF" in text:
        return "TICKET"
    if "PRESERVED_BEHAVIOR" in text or "COMPATIBILITY" in text:
        return "TICKET"
    return str(next((value for value in values if value not in (None, "")), "ADVISORY"))


def _entry_point(check: Mapping[str, object]) -> str:
    return str(check.get("entry_point", check.get("target", "")) or "")


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _symbol(entry_point: str, row: Mapping[str, object]) -> str:
    symbol = str(row.get("target_symbol", "") or "")
    if symbol:
        return symbol.rsplit(".", 1)[-1]
    return entry_point.rsplit(":", 1)[-1].rsplit(".", 1)[-1] or "call"


def _call_text(check: Mapping[str, object], row: Mapping[str, object]) -> str:
    entry_point = _entry_point(check)
    symbol = _symbol(entry_point, row)
    fixture = check.get("fixture")
    if isinstance(fixture, Mapping) and isinstance(fixture.get("calls"), Sequence):
        calls = fixture.get("calls")
        call = next((item for item in calls if isinstance(item, Mapping)), {})
        args = call.get("args", ())
        kwargs = call.get("kwargs", {})
    elif isinstance(fixture, Mapping):
        args = fixture.get("args", ())
        kwargs = fixture.get("kwargs", {})
    else:
        args, kwargs = (), {}
    pieces = [_json(value)
              for value in args] if isinstance(args, Sequence) and not isinstance(args, (str, bytes, bytearray)) else []
    if isinstance(kwargs, Mapping):
        pieces.extend(f"{key}={_json(value)}"
                      for key, value in kwargs.items())
    return f"{symbol}({', '.join(pieces)})"


def _result_text(row: Mapping[str, object], receipt: Mapping[str, object]) -> str:
    counterexample = row.get("counterexample", receipt.get("counterexample"))
    def find_exception(value: object) -> tuple[str, object] | None:
        if isinstance(value, Mapping):
            exception = value.get("exception")
            if isinstance(exception, Mapping):
                exception_type = str(exception.get("type", exception.get("exception_type", "")) or "")
                if exception_type:
                    return exception_type, exception.get("message", exception.get("error", ""))
            exception_type = str(value.get("exception_type", "") or "")
            if exception_type:
                return exception_type, value.get("message", value.get("error", ""))
            for child in value.values():
                found = find_exception(child)
                if found is not None:
                    return found
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for child in value:
                found = find_exception(child)
                if found is not None:
                    return found
        return None

    found_exception = find_exception(counterexample) or find_exception(receipt)
    if found_exception is not None:
        exception_type, message = found_exception
        return f"{exception_type}{(' ' + str(message)) if message else ''}"
    observed = row.get("observed", receipt.get("observed"))
    if observed not in (None, ""):
        return repr(observed)
    return str(row.get("expected", "counterexample"))


def enrich_row(row: Mapping[str, object], check: Mapping[str, object] | None = None,
               receipt: Mapping[str, object] | None = None) -> dict[str, object]:
    """Attach a reproducible call/result to one counterexample row."""
    result = dict(row)
    check_row = _row(check)
    receipt_row = _row(receipt)
    basis = _basis(result, check_row)
    call = _call_text(check_row, result)
    result.update({
        "entry_point": _entry_point(check_row),
        "fixture": check_row.get("fixture"),
        "selection": check_row.get("selection", check_row.get("selector")),
        "counterexample": copy.deepcopy(
            receipt_row.get("counterexample", check_row.get("counterexample"))),
        "refusal_oracle_basis": basis,
        "refusal_call": call,
        "refusal_result": _result_text(result, receipt_row),
    })
    return result


def _rank(value: object) -> int:
    return {"PUBLIC_EXAMPLE": 3, "TICKET": 2}.get(str(value), 1)


def _prepared_rows(rows: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    prepared: list[dict[str, object]] = []
    for row in rows:
        value = dict(row)
        if not value.get("refusal_call"):
            value = enrich_row(value, check=value, receipt=value)
        prepared.append(value)
    return tuple(prepared)


def render_evidence(rows: Sequence[Mapping[str, object]]) -> str:
    prepared = _prepared_rows(rows)
    if not prepared:
        return ""
    selected = max(prepared, key=lambda row: _rank(row.get("refusal_oracle_basis")))
    basis = str(selected.get("refusal_oracle_basis", "ADVISORY"))
    return (f"counterexample evidence ({basis}): "
            f"{selected.get('refusal_call', 'call')} -> "
            f"{selected.get('refusal_result', 'counterexample')}")


def emit_refusal(runtime: object | None, rows: Sequence[Mapping[str, object]],
                 detail: str = "") -> None:
    prepared = _prepared_rows(rows)
    if not prepared:
        return
    selected = max(prepared, key=lambda row: _rank(row.get("refusal_oracle_basis")))
    journal = getattr(runtime, "journal", None)
    emitter = getattr(journal, "emit", None)
    if not callable(emitter):
        return
    try:
        emitter(EVENT, producer=PRODUCER, event_schema=SCHEMA, dispatch=False,
                 check_ref=str(selected.get("check_id", "")),
                 oracle_basis=str(selected.get("refusal_oracle_basis", "ADVISORY")),
                 call=str(selected.get("refusal_call", "")),
                 result=str(selected.get("refusal_result", "")), detail=detail)
    except Exception:  # noqa: BLE001 - refusal telemetry must not alter authority
        return


__all__ = ["EVENT", "MODE_ENV", "PRODUCER", "SCHEMA", "enabled", "emit_refusal",
           "enrich_row", "mode", "render_evidence"]
