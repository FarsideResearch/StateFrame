from .service import run_one
from .batch import run_batch as _run_batch


class Worker:
    def process_one(self, item, work):
        return run_one(item, work)

    def process_batch(self, items, work, *, stop_on_error=False):
        return _run_batch(items, work, stop_on_error=stop_on_error)
