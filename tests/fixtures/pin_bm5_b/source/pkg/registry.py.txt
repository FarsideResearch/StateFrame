import importlib

_CAPABILITIES = {"single": "single-item"}


def lookup(name):
    return _CAPABILITIES.get(name)


def register_batch():
    batch = importlib.import_module("pkg.batch")
    _CAPABILITIES["batch"] = batch.run_batch
