"""wl.serve()'s automatic Weights Studio (weightslab/ui/embedded.py).

The suite turns the automatic UI off for everything else (tests/conftest.py,
WEIGHTSLAB_SERVE_UI=0); these tests turn it back on for themselves.
"""
import socket
import urllib.error
import urllib.request
from unittest import mock

import pytest

from weightslab.ui import embedded
from weightslab.ui import server as ui_server


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def ui_env(monkeypatch, tmp_path):
    """Automatic UI on, no Studio around, plain HTTP, loopback, private state."""
    monkeypatch.setenv("WEIGHTSLAB_SERVE_UI", "1")
    monkeypatch.delenv("WEIGHTSLAB_UI_ORIGIN", raising=False)   # restored after the test
    monkeypatch.setenv("GRPC_TLS_ENABLED", "0")
    monkeypatch.setenv("WEIGHTSLAB_UI_HOST", "127.0.0.1")
    monkeypatch.setenv("WEIGHTSLAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(ui_server, "has_static_assets", lambda: True)
    embedded.stop()
    yield tmp_path
    embedded.stop()


def _answers(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=5).close()
    except urllib.error.HTTPError:
        pass                                   # any HTTP status: the server is up
    except OSError:
        return False
    return True


def test_serves_the_ui_once_and_reuses_it(ui_env):
    port = _free_port()
    url = embedded.start(50999, ui_port=port, open_browser=False)

    assert url == f"http://127.0.0.1:{port}"
    assert embedded.is_running() and embedded.url() == url
    assert _answers(url + "/")
    # A second wl.serve() in the same process (a notebook cell run twice).
    assert embedded.start(50999, ui_port=_free_port(), open_browser=False) == url


def test_never_takes_the_backend_port(ui_env):
    backend_port = _free_port()
    url = embedded.start(backend_port, ui_port=backend_port, open_browser=False)
    assert url is not None
    assert not url.endswith(f":{backend_port}")


def test_off_with_weightslab_serve_ui_0(ui_env, monkeypatch):
    monkeypatch.setenv("WEIGHTSLAB_SERVE_UI", "0")
    assert embedded.start(50999, ui_port=_free_port(), open_browser=False) is None
    assert not embedded.is_running()


def test_not_started_from_inside_a_studio_session(ui_env, monkeypatch):
    monkeypatch.setenv("WEIGHTSLAB_UI_ORIGIN", "http://127.0.0.1:8080")
    assert embedded.start(50999, ui_port=_free_port(), open_browser=False) is None


def test_uses_a_running_weightslab_start_for_this_backend(ui_env):
    from weightslab.utils.active_experiment import record_ui_experiment

    experiment = ui_env / "experiment"
    experiment.mkdir()
    # Recorded with this (live) pid, as `weightslab start` records its own.
    record_ui_experiment(experiment, ui_port=8123, backend_port=50999)

    assert embedded.start(50999, ui_port=_free_port(), open_browser=False) is None
    # A UI pointed at ANOTHER backend does not show this one.
    assert embedded.start(51000, ui_port=_free_port(), open_browser=False) is not None


def test_not_started_when_the_ui_could_not_reach_a_tls_backend(ui_env, monkeypatch):
    monkeypatch.setattr(embedded, "_tls_settings", lambda: (None, None, False))
    assert embedded.start(50999, ui_port=_free_port(), open_browser=False) is None


def test_port_preference(monkeypatch):
    monkeypatch.delenv("WEIGHTSLAB_UI_PORT", raising=False)
    monkeypatch.setattr(embedded, "_hyperparams", lambda: None)
    assert embedded._preferred_port(None) == 8080
    monkeypatch.setenv("WEIGHTSLAB_UI_PORT", "8200")
    assert embedded._preferred_port(None) == 8200
    monkeypatch.setattr(embedded, "_hyperparams", lambda: {"ui_port": 8300})
    assert embedded._preferred_port(None) == 8300
    assert embedded._preferred_port(8400) == 8400
    assert embedded._preferred_port(0) == 8300                  # not a port: next source


def test_live_ui_for_backend(monkeypatch, tmp_path):
    from weightslab.utils import active_experiment as ae

    monkeypatch.setenv("WEIGHTSLAB_STATE_DIR", str(tmp_path / "state"))
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    assert ae.live_ui_for_backend(50051) is None
    ae.record_ui_experiment(experiment, ui_port=8080, backend_port=50051)
    assert ae.live_ui_for_backend(50051)["ui_port"] == 8080
    assert ae.live_ui_for_backend(50052) is None


# --------------------------------------------------------------- wl.serve()

@pytest.fixture
def serve_stubs(monkeypatch):
    """wl.serve() with the backend, CLI and kernel stubbed out."""
    from weightslab import src as wl_src

    monkeypatch.setenv("WEIGHTSLAB_DISABLE_EMBEDDED_KERNEL", "1")
    monkeypatch.delenv("WEIGHTSLAB_UI_ORIGIN", raising=False)
    monkeypatch.setattr(wl_src, "grpc_serve", mock.Mock())
    monkeypatch.setattr(wl_src, "cli_serve", mock.Mock())
    monkeypatch.setattr(wl_src, "_running_in_colab", lambda: False)
    start = mock.Mock(return_value="http://localhost:8080")
    monkeypatch.setattr(embedded, "start", start)
    return wl_src, start


def test_serve_starts_the_ui_for_its_backend(serve_stubs):
    wl_src, start = serve_stubs
    wl_src.serve(serving_cli=False, grpc_port=50777, ui_port=9100, ui_open_browser=False)

    start.assert_called_once()
    args, kwargs = start.call_args
    assert args == (50777,)
    assert kwargs["backend_host"] == "localhost"                # 0.0.0.0 bind -> dial loopback
    assert kwargs["ui_port"] == 9100
    assert kwargs["open_browser"] is False


@pytest.mark.parametrize("serve_kwargs", [
    {"serving_ui": False},
    {"serving_grpc": False},
])
def test_serve_without_the_ui(serve_stubs, serve_kwargs):
    wl_src, start = serve_stubs
    wl_src.serve(serving_cli=False, **serve_kwargs)
    start.assert_not_called()


def test_serve_does_not_start_the_ui_in_colab(serve_stubs, monkeypatch):
    wl_src, start = serve_stubs
    monkeypatch.setattr(wl_src, "_running_in_colab", lambda: True)
    wl_src.serve(serving_cli=False)
    start.assert_not_called()


def test_serve_survives_a_ui_that_fails(serve_stubs):
    wl_src, start = serve_stubs
    start.side_effect = RuntimeError("boom")
    wl_src.serve(serving_cli=False)                             # no exception
    wl_src.grpc_serve.assert_called_once()


def test_a_process_never_registers_with_its_own_ui(monkeypatch):
    """Its own UI server would kill it at exit (_TrackedProcesses.shutdown)."""
    from weightslab import src as wl_src

    monkeypatch.setenv("WEIGHTSLAB_UI_ORIGIN", "http://127.0.0.1:8080")
    monkeypatch.setattr(embedded, "is_running", lambda: True)
    thread = mock.Mock()
    monkeypatch.setattr(wl_src.threading, "Thread", thread)
    wl_src._register_pid_with_ui_server()
    thread.assert_not_called()
