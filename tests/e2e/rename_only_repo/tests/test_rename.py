from pathlib import Path


def test_module_renamed():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "pkg" / "old.py").exists()
    assert (root / "pkg" / "new.py").read_text() == "VALUE = 42\n"
