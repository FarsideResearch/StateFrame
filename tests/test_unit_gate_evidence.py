import hashlib
import json
from types import SimpleNamespace

from unit_gate_evidence import collection_command, pass_snapshot, source_identity


def test_source_identity_scans_over_20000_entries_and_256_mib(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for index in range(20_001):
        (root / f"empty-{index:05d}.bin").touch()
    file_size = 16 << 20
    for index in range(17):
        path = root / f"sparse-{index:02d}.bin"
        with path.open("wb") as stream:
            stream.truncate(file_size)

    identity = source_identity(root)

    assert len(identity) == 64


def test_source_identity_streams_a_20_mib_repository_file(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    path = root / "archive.bin"
    size = 20 << 20
    with path.open("wb") as output:
        output.truncate(size)

    identity = source_identity(root)

    assert len(identity) == 64


def test_unit_gate_hashes_a_40_mib_product_then_allows_its_deletion(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    path = root / "generated.bin"
    block = b"g" * (1 << 20)
    expected = hashlib.sha256()
    with path.open("wb") as output:
        for _ in range(40):
            output.write(block)
            expected.update(block)
    unit = SimpleNamespace(files=("generated.bin",), gate_tests=())

    product, _test_sources, inventory = pass_snapshot(
        unit, root, SimpleNamespace(unit_gate_runs=()), "source identity",
    )
    path.unlink()

    assert len(product) == 64
    assert product == hashlib.sha256(json.dumps(
        [("generated.bin", expected.hexdigest())], separators=(",", ":"),
    ).encode()).hexdigest()
    assert inventory is None
    assert not path.exists()


def test_collection_finds_a_gate_in_a_20_mib_test_file(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "test_runner.py").write_text("def test_runner():\n    pass\n")
    large = root / "test_large.py"
    line = "# " + "x" * 1022 + "\n"
    with large.open("w", encoding="utf-8") as output:
        output.write("def target_case():\n    pass\n")
        for _ in range((20 << 20) // len(line)):
            output.write(line)
    plan = SimpleNamespace(
        units=(SimpleNamespace(gate_tests=("target_case",), files=()),),
        public_gate_paths=(),
    )

    command, _identity = collection_command(
        ["python3", "-m", "pytest", "-q"], ["test_runner.py"], root, plan,
    )

    assert "test_runner.py" in command
    assert "test_large.py" in command


def test_unrelated_broken_test_file_does_not_stop_collection(tmp_path):
    from types import SimpleNamespace

    import unit_gate_evidence

    (tmp_path / "test_good.py").write_text("def test_wanted():\n    pass\n")
    (tmp_path / "test_broken.py").write_text("def test_other(:\n  '''\n")
    plan = SimpleNamespace(units=(SimpleNamespace(gate_tests=("test_wanted",)),), public_gate_paths=())
    command, _ = unit_gate_evidence.collection_command(["python", "-m", "pytest"], ["test_good.py"], tmp_path, plan)
    assert "test_good.py" in command and "test_broken.py" not in command
