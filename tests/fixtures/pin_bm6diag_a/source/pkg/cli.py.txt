from .batch import run_batch as _run_batch
from .worker import Worker


def run_one(item, work):
    return Worker().process_one(item, work)


def run_batch(items, work, *, stop_on_error=False):
    results = _run_batch(items, work, stop_on_error=stop_on_error)
    return [{"index": r.index, "item": r.item, "ok": r.ok, "value": r.value, "error": r.error} for r in results]
