"""Weights Studio served from the training process itself, by ``wl.serve()``.

``weightslab start`` runs the UI as its own process: it serves the bundled SPA
and proxies gRPC-Web to the backend that ``wl.serve()`` brings up. Having to
remember that second command (in a second terminal) was the most common reason
a first run "had no UI". So ``wl.serve()`` now starts the same server itself, in
a daemon thread of the training process -- the UI comes up with the backend and
goes away with it.

It stays out of the way when it is not wanted or not useful:

* ``wl.serve(serving_ui=False)``, or ``WEIGHTSLAB_SERVE_UI=0`` in the
  environment (no code change, e.g. for CI or a sweep);
* a ``weightslab start`` is already running for this backend's port (recorded
  in the active-experiment marker), or this process was launched from a
  Weights Studio session (``WEIGHTSLAB_UI_ORIGIN`` is set) -- the Studio is
  already open, a second one would only add a stray browser tab;
* (decided by ``wl.serve()``) no gRPC backend is served, or the process runs in
  Google Colab, where the browser is on another machine: there the way in is
  still ``serving_bore=True`` + ``weightslab start`` / ``weightslab tunnel``
  locally.

TLS follows the backend's own decision (``grpc_tls_enabled`` in the config,
else ``GRPC_TLS_ENABLED``, which the import-time cert check sets), so the two
ends always agree. Host and port follow ``weightslab start``'s rules:
``WEIGHTSLAB_UI_HOST`` (default ``0.0.0.0``); ``ui_port`` argument, else
``ui_port`` in the config, else ``WEIGHTSLAB_UI_PORT``, else 8080, falling back
to a free port when that one is busy.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

_FALSE_VALUES = ("0", "false", "no", "off")
_DEFAULT_UI_PORT = 8080

_lock = threading.Lock()
_httpd: Any = None
_url: Optional[str] = None


def disabled_by_env() -> bool:
    """True when ``WEIGHTSLAB_SERVE_UI`` turns the automatic UI off."""
    return os.environ.get("WEIGHTSLAB_SERVE_UI", "1").strip().lower() in _FALSE_VALUES


def is_running() -> bool:
    """Whether this process is serving Weights Studio itself."""
    return _httpd is not None


def url() -> Optional[str]:
    """The URL of the UI this process serves, or None."""
    return _url


def _coerce_port(value: Any) -> Optional[int]:
    try:
        port = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return port if 0 < port < 65536 else None


def _hyperparams() -> Any:
    try:
        from weightslab.backend.ledgers import get_hyperparams
        return get_hyperparams()
    except Exception:  # noqa: BLE001 -- no hyperparameters registered
        return None


def _config_get(config: Any, key: str) -> Any:
    try:
        return config.get(key) if config is not None else None
    except Exception:  # noqa: BLE001 -- not a mapping
        return None


def _preferred_port(ui_port: Optional[int]) -> int:
    """``ui_port`` argument > ``ui_port`` in the config > env > 8080."""
    for candidate in (ui_port,
                      _config_get(_hyperparams(), "ui_port"),
                      os.environ.get("WEIGHTSLAB_UI_PORT")):
        port = _coerce_port(candidate)
        if port is not None:
            return port
    return _DEFAULT_UI_PORT


def _tls_settings() -> Tuple[Optional[str], Optional[str], bool]:
    """``(certs_dir, auth_token, usable)`` matching the backend's TLS choice.

    ``usable`` is False when the backend serves TLS but no certificate
    directory is found for the UI to dial it with (custom ``GRPC_TLS_*`` paths
    outside the WeightsLab certs dir): the UI could not reach it.
    """
    from weightslab.trainer.trainer_services import _resolve_bool_setting

    if not _resolve_bool_setting(_hyperparams(), "grpc_tls_enabled", "GRPC_TLS_ENABLED", "0"):
        return None, None, True
    from weightslab.security import CertAuthManager

    manager = CertAuthManager.from_env_or_default()
    if not manager.has_valid_certs():
        return None, None, False
    try:
        token = manager.get_or_create_auth_token()
    except Exception:  # noqa: BLE001 -- token is optional
        token = None
    return str(manager.certs_dir), token, True


def _ui_already_serving(backend_port: int) -> Optional[str]:
    """Why a UI for this backend already exists, or None."""
    origin = (os.environ.get("WEIGHTSLAB_UI_ORIGIN") or "").strip()
    if origin:
        return f"this process was launched from Weights Studio ({origin})"
    try:
        from weightslab.utils.active_experiment import live_ui_for_backend
        entry = live_ui_for_backend(backend_port)
    except Exception:  # noqa: BLE001 -- advisory record only
        entry = None
    if entry:
        port = entry.get("ui_port")
        where = f"http://localhost:{port}" if port else "another `weightslab start`"
        return f"`weightslab start` already serves this backend at {where}"
    return None


def start(
    backend_port: int,
    *,
    backend_host: str = "localhost",
    experiment_dir: Optional[str] = None,
    ui_port: Optional[int] = None,
    open_browser: bool = True,
) -> Optional[str]:
    """Serve Weights Studio from this process, proxying to ``backend_port``.

    Returns the UI's URL, or None when it was not started (see the module
    docstring for when). Idempotent: a second call returns the same URL.
    Never raises -- a UI that cannot start must not stop a training run.
    """
    global _httpd, _url
    with _lock:
        if _httpd is not None:
            return _url
        if disabled_by_env():
            logger.info("WEIGHTSLAB_SERVE_UI is off: not starting Weights Studio "
                        "(run `weightslab start` to open it).")
            return None
        reason = _ui_already_serving(int(backend_port))
        if reason:
            logger.info(f"Not starting another Weights Studio: {reason}.")
            return None
        try:
            from weightslab.ui import server as ui_server
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not load the Weights Studio server ({exc}); "
                           "run `weightslab start` to open the UI.")
            return None
        if not ui_server.has_static_assets():
            logger.warning("This install has no bundled Weights Studio assets, so "
                           "wl.serve() does not start the UI. See `weightslab start`.")
            return None

        certs_dir, auth_token, tls_usable = _tls_settings()
        if not tls_usable:
            logger.warning(
                "The backend serves TLS, but no WeightsLab certificates were found "
                "for the UI to connect with, so wl.serve() does not start the UI. "
                "Run `weightslab start --certs` with the matching certificates.")
            return None

        ui_host = os.environ.get("WEIGHTSLAB_UI_HOST", "0.0.0.0")
        port = _preferred_port(ui_port)
        if port == int(backend_port):
            port = 0                       # never take the backend's own port
        probe_host = "127.0.0.1" if ui_host in ("0.0.0.0", "::", "") else ui_host
        bind_port = ui_server.find_free_port(port, host=probe_host)
        if port and bind_port != port:
            logger.info(f"UI port {port} is busy; serving Weights Studio on {bind_port}.")
        try:
            httpd = ui_server.serve_ui(
                ui_host=ui_host,
                ui_port=bind_port,
                backend_host=backend_host,
                backend_port=int(backend_port),
                open_browser=open_browser,
                certs_dir=certs_dir,
                grpc_auth_token=auth_token,
                block=False,
                experiment_dir=experiment_dir,
            )
        except Exception as exc:  # noqa: BLE001 -- never break the training run
            logger.warning(f"Could not start Weights Studio ({exc}); "
                           "run `weightslab start` to open the UI.")
            return None

        scheme = "https" if str(os.environ.get("WEIGHTSLAB_UI_ORIGIN", "")).startswith("https") else "http"
        display_host = "localhost" if ui_host in ("0.0.0.0", "::", "") else ui_host
        _httpd = httpd
        _url = f"{scheme}://{display_host}:{httpd.server_address[1]}"
        return _url


def stop() -> None:
    """Stop the UI this process serves (no-op when there is none)."""
    global _httpd, _url
    with _lock:
        httpd, _httpd, _url = _httpd, None, None
    if httpd is None:
        return
    try:
        httpd.shutdown()
        httpd.server_close()
    except Exception:  # noqa: BLE001
        pass
    if os.environ.get("WEIGHTSLAB_UI_ORIGIN", "").endswith(f":{httpd.server_address[1]}"):
        os.environ.pop("WEIGHTSLAB_UI_ORIGIN", None)
