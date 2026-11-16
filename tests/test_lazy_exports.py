"""Package exports must stay lazy and explain missing optional dependencies."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import weightslab as wl

ULTRALYTICS_EXPORTS = (
    "WLAwareTrainer",
    "WLAwareSegmentationTrainer",
    "WLAwareDataset",
    "WLAwareSegmentationDataset",
)


@pytest.fixture
def uncached_exports(monkeypatch):
    names = (*ULTRALYTICS_EXPORTS, "compare_predictions")
    for name in names:
        monkeypatch.delitem(vars(wl), name, raising=False)
    yield
    # __getattr__ caches successful lookups; never leak a fake export to another test.
    for name in names:
        vars(wl).pop(name, None)


@pytest.mark.parametrize("name", ULTRALYTICS_EXPORTS)
def test_ultralytics_exports_resolve_lazily_and_cache(name, monkeypatch, uncached_exports):
    expected = object()
    importer = Mock(return_value=SimpleNamespace(**{name: expected}))
    monkeypatch.setattr(wl.importlib, "import_module", importer)

    assert name in dir(wl)
    assert name in wl.__all__
    assert getattr(wl, name) is expected
    assert getattr(wl, name) is expected
    importer.assert_called_once_with(".integrations.ultralytics", "weightslab")


@pytest.mark.parametrize("name", ULTRALYTICS_EXPORTS)
def test_missing_ultralytics_explains_how_to_install(name, monkeypatch, uncached_exports):
    original = ModuleNotFoundError("No module named 'ultralytics'", name="ultralytics")
    monkeypatch.setattr(wl.importlib, "import_module", Mock(side_effect=original))

    with pytest.raises(ImportError) as raised:
        getattr(wl, name)

    assert f"weightslab.{name}" in str(raised.value)
    assert "pip install 'weightslab[ultralytics]'" in str(raised.value)
    assert raised.value.__cause__ is original
    assert name not in vars(wl)


@pytest.mark.parametrize("name", ("WLAwareTrainer", "compare_predictions"))
def test_unrelated_import_errors_are_not_masked(name, monkeypatch, uncached_exports):
    original = ModuleNotFoundError("No module named 'torch'", name="torch")
    monkeypatch.setattr(wl.importlib, "import_module", Mock(side_effect=original))

    with pytest.raises(ImportError) as raised:
        getattr(wl, name)

    assert raised.value is original


def test_comparison_export_is_preserved(monkeypatch, uncached_exports):
    expected = object()
    importer = Mock(return_value=SimpleNamespace(compare_predictions=expected))
    monkeypatch.setattr(wl.importlib, "import_module", importer)

    assert wl.compare_predictions is expected
    importer.assert_called_once_with(".diagnostics", "weightslab")


def test_plain_import_does_not_load_torch_or_ultralytics():
    # A fresh interpreter matters: the main test process may already have torch loaded.
    script = textwrap.dedent("""
        import sys

        class RejectHeavyImports:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'torch', 'ultralytics'}:
                    raise AssertionError(f'Plain import tried to load {fullname}')

        sys.meta_path.insert(0, RejectHeavyImports())
        import weightslab as wl
        assert 'WLAwareTrainer' in dir(wl)
        assert 'WLAwareTrainer' not in vars(wl)
        assert 'torch' not in sys.modules
        assert 'ultralytics' not in sys.modules
    """)
    env = dict(os.environ, WEIGHTSLAB_OPENCODE_AUTOINSTALL="0", WL_NO_TELEMETRY="1",
               WEIGHTSLAB_SKIP_SECURE_INIT="true", WEIGHTSLAB_SUPPRESS_BANNER="1",
               WEIGHTSLAB_LOG_TO_FILE="false", WEIGHTSLAB_LOG_LEVEL="ERROR")
    result = subprocess.run([sys.executable, "-c", script], env=env,
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
