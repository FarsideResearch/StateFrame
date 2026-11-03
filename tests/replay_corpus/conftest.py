"""Keep embedded repository fixtures out of StateFrame test discovery."""
from pathlib import Path


def pytest_ignore_collect(collection_path: Path, config: object) -> bool:
    del config
    parts = collection_path.parts
    if "replay_corpus" not in parts:
        return False
    embedded_paths = parts[parts.index("replay_corpus") + 1:]
    return any(part in {"repo", "_repos"} for part in embedded_paths)
