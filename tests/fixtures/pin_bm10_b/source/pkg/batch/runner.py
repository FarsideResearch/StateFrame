from pkg import config

from .contracts import BatchResult


_UNSET = object()


def run_batch(items, work, *, stop_on_error=_UNSET):
    if stop_on_error is _UNSET:
        stop_on_error = config.STOP_ON_ERROR

    results = []
    for i, item in enumerate(items):
        try:
            value = work(item)
            results.append(BatchResult(index=i, item=item, ok=True, value=value, error=None))
        except Exception as exc:
            results.append(BatchResult(index=i, item=item, ok=False, value=None, error=f"{type(exc).__name__}: {exc}"))
            if stop_on_error:
                break
    return results
