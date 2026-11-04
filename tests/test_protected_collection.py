"""Collection-time write protection for guarded pytest runs."""

import pytest

from stateframe.test_runners import collection_change_reason, pytest_plugin_modules


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("conftest.py", "conftest.py"),
        ("pkg/conftest.py", "conftest.py"),
        ("tests/conftest.py", "conftest.py"),
        ("__init__.py", "__init__.py"),
        ("tests/__init__.py", "__init__.py"),
        ("suite/__init__.py", "__init__.py"),
        ("suite/sub/__init__.py", "__init__.py"),
        ("pkg/__init__.py", None),
        ("pytest.ini", "pytest.ini"),
        ("tests/pyproject.toml", "pyproject.toml"),
        ("sitecustomize.py", "sitecustomize.py"),
        ("pkg/usercustomize.py", "usercustomize.py"),
        ("pkg/entry.pth", "entry.pth"),
        ("pytest_custom.py", "pytest_custom.py"),
        ("pkg/my_plugin.py", "my_plugin.py"),
        ("pkg/my_plugin/__init__.py", "__init__.py"),
        ("pkg/new_module.py", None),
    ],
)
def test_collection_creation_classification(path, expected):
    reason = collection_change_reason(
        path,
        kind="pytest",
        protected_tests=("suite/sub/test_judge.py",),
        test_files=("tests/test_other.py",),
        pytest_plugins=("pkg.my_plugin",),
        created=True,
    )
    assert (expected in reason if expected else reason is None)


def test_existing_pyproject_only_blocks_pytest_section_changes():
    kwargs = dict(kind="pytest", protected_tests=("tests/test_app.py",),
                  test_files=("tests/test_app.py",), created=False)
    original = '[project]\nname = "app"\n[tool.pytest.ini_options]\naddopts = "-q"\n'
    product_edit = original.replace('name = "app"', 'name = "new-app"')
    pytest_edit = original.replace('addopts = "-q"', 'addopts = "-x"')
    assert collection_change_reason("pyproject.toml", before=original,
                                    after=product_edit, **kwargs) is None
    assert "pyproject.toml" in collection_change_reason(
        "pyproject.toml", before=original, after=pytest_edit, **kwargs)
    assert collection_change_reason("setup.cfg", before="[metadata]\nname = app\n",
                                    after="[metadata]\nname = new-app\n", **kwargs) is None
    assert "setup.cfg" in collection_change_reason(
        "setup.cfg", before="[metadata]\nname = app\n",
        after="[metadata]\nname = app\n[tool:pytest]\naddopts = -x\n", **kwargs)


def test_create_tox_ini_with_pytest_section_is_protected():
    reason = collection_change_reason(
        "tox.ini", kind="pytest", created=True,
        after="[tox]\nenvlist = py311\n[pytest]\naddopts = -q\n")
    assert reason is not None and "tox.ini" in reason


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("[tox]\nenvlist = py311\n",
         "[tox]\nenvlist = py311\n[pytest]\naddopts = -q\n"),
        ("[pytest]\naddopts = -q\n", "[pytest]\naddopts = -x\n"),
    ],
    ids=("add-pytest-section", "alter-pytest-section"),
)
def test_modify_tox_ini_pytest_section_is_protected(before, after):
    reason = collection_change_reason(
        "tox.ini", kind="pytest", created=False, before=before, after=after)
    assert reason is not None and "tox.ini" in reason


def test_modify_tox_ini_only_outside_pytest_section_is_allowed():
    before = "[tox]\nenvlist = py311\n\n[pytest]\naddopts = -q\n"
    after = "[tox]\nenvlist = py312\n\n[pytest]\naddopts = -q\n"
    assert collection_change_reason(
        "tox.ini", kind="pytest", created=False, before=before, after=after) is None


def test_create_root_setup_cfg_with_pytest_section_is_protected():
    reason = collection_change_reason(
        "setup.cfg", kind="pytest", created=True,
        after="[metadata]\nname = app\n[tool:pytest]\naddopts = -q\n")
    assert reason is not None and "setup.cfg" in reason


def test_create_root_pyproject_toml_is_protected():
    reason = collection_change_reason(
        "pyproject.toml", kind="pytest", created=True,
        after='[project]\nname = "app"\n')
    assert reason is not None and "pyproject.toml" in reason


def test_modify_pytest_ini_is_protected():
    reason = collection_change_reason(
        "pytest.ini", kind="pytest", created=False,
        before="[pytest]\naddopts = -q\n",
        after="[pytest]\naddopts = -x\n")
    assert reason is not None and "pytest.ini" in reason


def test_guard_is_inactive_for_off_and_generic():
    for kind, mode in (("pytest", "off"), ("generic", "all")):
        assert collection_change_reason("tests/__init__.py", kind=kind,
                                        protect_tests=mode,
                                        protected_tests=("tests/test_app.py",),
                                        created=True) is None


def test_existing_test_initializer_is_protected_but_product_initializer_is_not():
    kwargs = dict(kind="pytest", protected_tests=("tests/test_app.py",), created=False)
    assert "__init__.py" in collection_change_reason("tests/__init__.py", **kwargs)
    assert collection_change_reason("pkg/__init__.py", **kwargs) is None


def test_pytest_plugin_names_are_read_from_literal_declarations(tmp_path):
    test = tmp_path / "tests/test_app.py"
    test.parent.mkdir()
    test.write_text('pytest_plugins = ("pkg.my_plugin", "other")\n')
    assert pytest_plugin_modules(tmp_path, ("tests/test_app.py",)) == (
        "other", "pkg.my_plugin")
    assert "my_plugin.py" in collection_change_reason(
        "pkg/my_plugin.py", kind="pytest", created=True,
        pytest_plugins=pytest_plugin_modules(tmp_path, ("tests/test_app.py",)))
    test.write_text('pytest_plugins = BASE + ["pkg.dynamic_plugin"]\n')
    assert pytest_plugin_modules(tmp_path, ("tests/test_app.py",)) == (
        "*", "pkg.dynamic_plugin")
    assert collection_change_reason(
        "pkg/unknown.py", kind="pytest", created=True,
        pytest_plugins=pytest_plugin_modules(tmp_path, ("tests/test_app.py",)))
