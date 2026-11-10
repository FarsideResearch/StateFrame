"""Worker-local cache keys for guards that reuse whole source-tree scans."""
from __future__ import annotations

from pathlib import Path


def source_tree_signature(root: Path) -> tuple[tuple[str, int, int], ...]:
    """Capture every Python path, modification time, and size under a scan root."""
    signature = []
    for path in sorted(root.rglob("*.py")):
        stat = path.stat()
        signature.append((str(path.resolve()), stat.st_mtime_ns, stat.st_size))
    return tuple(signature)
