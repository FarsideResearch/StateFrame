"""Fixture repositories are run by StateFrame, not the outer pytest process."""
from pathlib import Path


collect_ignore = [str(Path(__file__).parent / "fixtures")]
