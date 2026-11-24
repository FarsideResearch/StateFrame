from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest

from stateframe.task_row import TaskRowError, build_row
from stateframe.workcopy import create


OLD_TEXT_SUFFIXES = {
    ".py", ".pyi", ".txt", ".md", ".rst", ".toml", ".cfg", ".ini", ".json",
    ".yaml", ".yml", ".csv", ".sql", ".html", ".css", ".js", ".ts", ".sh", ".in",
}
OLD_GENERIC_EXTRA_SUFFIXES = {
    ".jsx", ".tsx", ".mjs", ".cjs", ".mts", ".cts", ".go", ".rs", ".c", ".h",
    ".cc", ".cpp", ".hpp",
}
OLD_ENTRY_SUFFIXES = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".c", ".cc", ".cpp", ".sh",
}
OLD_GENERIC_NAMES = {"Makefile", "makefile", "GNUmakefile", "Cargo.toml"}
TARGET_SUFFIXES = (
    ".vue", ".tf", ".tfvars", ".hcl", ".java", ".kt", ".go", ".rs", ".c", ".h",
    ".cc", ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift", ".scala", ".jsx", ".tsx",
    ".mjs", ".cjs", ".scss", ".less", ".svelte", ".proto", ".graphql", ".gradle", ".xml",
    ".ps1", ".bat", ".lua", ".dart", ".r", ".jl", ".pl", ".ex", ".exs",
)


def _make_work(root: Path, files: dict[str, str | bytes]):
    repo = root / "repo"
    for relative, content in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
    return create(repo, root / "copy")


@pytest.mark.parametrize(("runner_kind", "target"), (
    ("generic", "infra/main.tf"), ("pytest", "infra/main.tf"),
    ("generic", "web/App.vue"), ("pytest", "web/App.vue"),
    ("generic", "src/Main.java"), ("pytest", "src/Main.java"),
    ("generic", "Dockerfile"), ("pytest", "Dockerfile"),
    ("pytest", "Makefile"), ("generic", "CMakeLists.txt"), ("pytest", "CMakeLists.txt"),
))
def test_existing_source_targets_are_admitted_for_each_runner(tmp_path, runner_kind, target):
    contents = {
        "infra/main.tf": 'resource "example" "main" {}\n',
        "web/App.vue": "<template><main /></template>\n",
        "src/Main.java": "class Main {}\n",
        "Dockerfile": "FROM scratch\n",
        "Makefile": "all:\n\t@true\n",
        "CMakeLists.txt": "project(example)\n",
    }
    contents["tests/test_check.py"] = "def test_check():\n    assert True\n"
    work = _make_work(tmp_path, contents)
    command = (("pytest", "-q", "tests/test_check.py")
               if runner_kind == "pytest" else ("true",))

    row = build_row(
        work, task="Update the selected source.", checks=["tests/test_check.py"],
        target=target, runner_kind=runner_kind, test_command=command, protect_tests="all",
    )

    assert row["entry"]["path"] == target
    assert row["files"][target] == {"parent_src": contents[target]}


def test_explicit_edit_path_admits_new_source_suffix_on_demand(tmp_path):
    work = _make_work(tmp_path, {
        "infra/main.tf": 'resource "example" "main" {}\n',
        "tests/test_check.py": "def test_check():\n    assert True\n",
    })

    row = build_row(
        work, task="Update the selected source.", checks=["tests/test_check.py"],
        edit_paths=["infra/main.tf"], runner_kind="generic", test_command=("true",),
        protect_tests="all",
    )

    assert row["entry"]["path"] == "infra/main.tf"
    assert set(row["files"]) == {"infra/main.tf"}


def test_product_source_table_covers_requested_suffixes_and_names():
    from stateframe.task_row import TARGET, is_product_source

    for suffix in TARGET_SUFFIXES:
        assert is_product_source("src/file" + suffix, mode=TARGET), suffix
    for name in ("Dockerfile", "Makefile", "CMakeLists.txt"):
        assert is_product_source(name, mode=TARGET), name
    assert not is_product_source("credentials.tf", mode=TARGET)
    assert not is_product_source("src/file.bin", mode=TARGET)


def test_new_source_types_do_not_expand_default_task_rows():
    from stateframe.task_row import GENERIC, PYTEST, TARGET, is_product_source

    old_generic = OLD_TEXT_SUFFIXES | OLD_GENERIC_EXTRA_SUFFIXES
    for suffix in TARGET_SUFFIXES:
        if suffix not in old_generic:
            assert not is_product_source("src/file" + suffix, mode=GENERIC), suffix
        if suffix not in OLD_TEXT_SUFFIXES:
            assert not is_product_source("src/file" + suffix, mode=PYTEST), suffix
    assert is_product_source("Dockerfile", mode=TARGET)
    assert not is_product_source("Dockerfile", mode=GENERIC)


def test_legacy_task_row_request_digest_is_unchanged():
    suffixes = OLD_TEXT_SUFFIXES | OLD_GENERIC_EXTRA_SUFFIXES
    files = {"src/file" + suffix: "value\n" for suffix in suffixes}
    files.update({name: "value\n" for name in OLD_GENERIC_NAMES})

    class MemoryWork:
        baseline = files

        def text_files(self, wanted=None, large=None):
            return files

    options = dict(task="Update the repository source.", runner_kind="generic", protect_tests="off")
    row = build_row(
        MemoryWork(), **options,
    )
    body = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    target_row = build_row(MemoryWork(), target="src/file.py", **options)
    target_body = json.dumps(
        target_row, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()

    assert len(row["files"]) == 32
    assert row["entry"]["path"] == "src/file.c"
    assert hashlib.sha256(body).hexdigest() == (
        "8011be954576c25fb361c23f609543d0f8f160c6c8e9ecd268d432d272caa3fd"
    )
    assert hashlib.sha256(target_body).hexdigest() == (
        "0bf7249ae0f445647bb035afd87d8a3c779c1bc843731edd871627bb5e41954d"
    )


def test_old_predicates_match_400_deterministic_file_sets():
    from stateframe.task_row import ENTRY, GENERIC, PYTEST, TARGET, is_product_source

    old_generic_suffixes = OLD_TEXT_SUFFIXES | OLD_GENERIC_EXTRA_SUFFIXES
    candidates = [f"src/file-{index}{suffix}" for index, suffix in enumerate(
        sorted(old_generic_suffixes | set(TARGET_SUFFIXES) | {".bin", ".pyc", ".tf"}))]
    candidates.extend(("Makefile", "makefile", "GNUmakefile", "Cargo.toml", "Dockerfile",
                       "CMakeLists.txt", "src/__init__.py"))
    rng = random.Random(0x5A17)

    for _ in range(400):
        paths = rng.sample(candidates, k=rng.randrange(len(candidates) + 1))
        expected_pytest = {path for path in paths
                           if Path(path).suffix in OLD_TEXT_SUFFIXES
                           or Path(path).name in {"Makefile", "setup.py"}}
        expected_generic = {path for path in paths
                            if Path(path).suffix in old_generic_suffixes
                            or Path(path).name in OLD_GENERIC_NAMES}
        expected_entry = {path for path in paths
                          if Path(path).suffix in OLD_ENTRY_SUFFIXES
                          and not path.endswith("/__init__.py")}
        assert {path for path in paths if is_product_source(path, mode=PYTEST)} == expected_pytest
        assert {path for path in paths if is_product_source(path, mode=GENERIC)} == expected_generic
        assert expected_generic <= {
            path for path in paths if is_product_source(path, mode=TARGET)
        }
        assert {path for path in paths if is_product_source(path, mode=ENTRY)
                and not path.endswith("/__init__.py")} == expected_entry


@pytest.mark.parametrize("path,content", (
    ("infra/main.tf", b"\x00\xff"),
    ("infra/credentials.tf", "placeholder\n"),
))
def test_binary_and_secret_like_targets_remain_rejected(tmp_path, path, content):
    work = _make_work(tmp_path, {"README.md": "Repository notes.\n", path: content})

    with pytest.raises(TaskRowError, match="not a product source file"):
        build_row(work, task="Update the selected source.", target=path,
                  runner_kind="generic", protect_tests="off")


def test_large_targeted_source_is_admitted_and_untargeted_is_not(tmp_path):
    text = "<template>x</template>\n" * 60_000
    work = _make_work(tmp_path, {"README.md": "Repository notes.\n", "web/App.vue": text})
    assert "web/App.vue" in work.large_files

    row = build_row(work, task="Update the selected source.", target="web/App.vue",
                    runner_kind="generic", protect_tests="off")
    assert row["files"]["web/App.vue"] == {"parent_src": text}
    plain = build_row(work, task="Update the notes.", runner_kind="generic", protect_tests="off")
    assert "web/App.vue" not in json.dumps(plain)
