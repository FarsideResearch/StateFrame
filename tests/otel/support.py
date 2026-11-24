"""Small valid records shared by exporter unit tests."""
from stateframe.otel.identity import derive
from stateframe.otel.types import AttrKey, SafeAttr, SafeSpan, Signal, SignalBatch


def synthetic_batch() -> SignalBatch:
    key = b"k" * 32
    seed = b"s" * 16
    run_id = derive(key, "run", (seed,), 16)
    trace_id = derive(key, "trace", (seed,), 16)
    attrs = tuple(sorted((
        SafeAttr(AttrKey.RUN_ID, run_id.hex()),
        SafeAttr(AttrKey.SOURCE, "record"),
        SafeAttr(AttrKey.TIMING, "publication"),
        SafeAttr(AttrKey.COMPLETE, True),
    ), key=lambda item: str(item.key)))
    span = SafeSpan(
        "stateframe.invocation", trace_id, derive(key, "support-span", (seed,), 8), None,
        1_730_000_000_000_000_000, 1_730_000_000_000_000_000,
        attrs, (), (), 0, 0, 0, False,
    )
    return SignalBatch(Signal.TRACES, derive(key, "resource", (), 16), (span,))
