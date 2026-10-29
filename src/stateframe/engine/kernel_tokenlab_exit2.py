"""Truthful classification for exit-2 import failures.

The ordinary pytest renderer has to remain byte-compatible when this lane is
off.  When enabled, only an import failure that points into the editable
product tree is relabeled.  Test files, site-packages, and an unbound harness
remain collection/plumbing failures.
"""
from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path


MODE_ENV = "STATEFRAME_TOKENLAB_TRUTHFUL_EXIT2"
SCHEMA = "stateframe-v19-truthful-exit2/1"
EVENT = "tokenlab_v19_truthful_exit2"
PRODUCER = "kernel_tokenlab_exit2"

_IMPORT_NAME = re.compile(
    r"cannot import name [\"'](?P<name>[A-Za-z_]\w*)[\"']\s+from\s+[\"'](?P<module>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)[\"']",
    re.IGNORECASE,
)
_ATTRIBUTE_NAME = re.compile(
    r"module [\"'](?P<module>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)[\"']\s+has no attribute [\"'](?P<name>[A-Za-z_]\w*)[\"']",
    re.IGNORECASE,
)
_TRACEBACK_FILE = re.compile(r"File [\"'](?P<path>[^\"']+)[\"']")
_FROM_IMPORT = re.compile(
    r"from\s+(?P<module>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s+import\s+(?P<name>[A-Za-z_]\w*)"
)


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


def _text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return " ".join(f"{key}={child}" for key, child in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return " ".join(_text(item) for item in value)
    return str(value or "")


def _relative(path: str, root: str | Path | None) -> str:
    raw = path.replace("\\", "/")
    if root is None:
        return raw.lstrip("./").lstrip("/")
    try:
        return Path(raw).resolve().relative_to(Path(root).resolve()).as_posix()
    except (OSError, ValueError):
        return raw.lstrip("./")


def _is_product_path(path: str, root: str | Path | None) -> bool:
    relative = _relative(path, root)
    parts = tuple(item for item in relative.split("/") if item)
    if not parts or any(item in {"tests", "test", "site-packages", ".venv", "venv",
                              "dist-packages", "__pycache__"} for item in parts):
        return False
    if any(item.startswith(".") for item in parts):
        return False
    return parts[-1].endswith(".py") or "app" in parts


def _module_from_path(path: str, root: str | Path | None) -> str:
    relative = _relative(path, root)
    parts = list(Path(relative).with_suffix("").parts)
    if "app" in parts:
        parts = parts[parts.index("app"):]
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) or Path(path).stem


def _failure_parts(blob: str, traceback_path: str) -> tuple[str, str] | None:
    match = _IMPORT_NAME.search(blob) or _ATTRIBUTE_NAME.search(blob)
    if match:
        return str(match.group("module")), str(match.group("name"))
    match = _FROM_IMPORT.search(blob)
    if match and traceback_path:
        return str(match.group("module")), str(match.group("name"))
    if traceback_path:
        module = _module_from_path(traceback_path, None)
        return module, "<missing symbol>"
    return None


def classify(blob: object, *, root: str | Path | None = None,
             targets: Sequence[object] = (), fn: object = "",
             rel_hint: object = "") -> dict[str, object] | None:
    """Return a product-import label only for an editable product traceback."""
    text = _text(blob)
    if not re.search(r"\b(?:ImportError|ModuleNotFoundError|AttributeError)\b", text,
                    re.IGNORECASE):
        return None
    paths = [str(rel_hint)] if rel_hint else []
    paths.extend(str(value) for value in targets if str(value).strip())
    paths.extend(match.group("path") for match in _TRACEBACK_FILE.finditer(text))
    product_path = next((path for path in paths if _is_product_path(path, root)), "")
    if not product_path:
        return None
    parts = _failure_parts(text, product_path)
    if parts is None:
        return None
    module, name = parts
    relative_path = _relative(product_path, root)
    return {
        "label": f"import error in YOUR code: {module} does not define {name}",
        "module": module,
        "missing_symbol": name,
        "path": relative_path,
        "function": str(fn or ""),
        "target_count": len(tuple(targets)),
    }


def emit(runtime: object | None, detail: Mapping[str, object]) -> None:
    journal = getattr(runtime, "journal", None)
    emitter = getattr(journal, "emit", None)
    if not callable(emitter):
        return
    try:
        emitter(EVENT, producer=PRODUCER, event_schema=SCHEMA, dispatch=False,
                 label=str(detail.get("label", "")),
                 module=str(detail.get("module", "")),
                 missing_symbol=str(detail.get("missing_symbol", "")),
                 path=str(detail.get("path", "")),
                 function=str(detail.get("function", "")))
    except Exception:  # noqa: BLE001 - diagnostics must not change test rendering
        return


__all__ = ["EVENT", "MODE_ENV", "PRODUCER", "SCHEMA", "classify", "emit", "enabled", "mode"]
