from .batch import run_batch as _run_batch


def run_one(item, work):
    return work(item)


def run_batch(items, work, *, stop_on_error=False):
    return _run_batch(items, work, stop_on_error=stop_on_error)
