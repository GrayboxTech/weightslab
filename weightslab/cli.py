"""WeightsLab command-line interface.

Docker-free. Commands:
  * ``weightslab start``            — run the native UI server (bundled SPA +
                                      gRPC-Web proxy), the pip-native replacement
                                      for the old Docker/Envoy stack.
  * ``weightslab start example``    — run a bundled PyTorch example.
  * ``weightslab se``               — set up the secure environment (TLS certs +
                                      gRPC auth token) in $WEIGHTSLAB_CERTS_DIR.
  * ``weightslab cli``              — attach a terminal to a running experiment.
  * ``weightslab tunnel``           — forward a remote gRPC backend to a local port.
"""

import argparse
import os
import stat
import subprocess
import sys
import logging
from pathlib import Path
from typing import Optional, Sequence

import yaml

from weightslab.security import CertAuthManager
from weightslab.tunnel import DEFAULT_LISTEN_PORT

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Capture whether WEIGHTSLAB_CERTS_DIR was already in the shell env when this
# process started (set by the user) vs. injected later by our own code.
_CERTS_DIR_IN_ORIGINAL_ENV: bool = "WEIGHTSLAB_CERTS_DIR" in os.environ


def _coerce_valid_port(value) -> Optional[int]:
    """Convert a value to a valid TCP port, or None when invalid."""
    try:
        port = int(value)
    except (TypeError, ValueError):
        return None
    return port if 1 <= port <= 65535 else None


def _lookup_nested(mapping: dict, path: Sequence[str]):
    """Safely resolve a nested dict path and return None on missing keys."""
    cur = mapping
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _load_ui_port_from_experiment_config(explicit_path: Optional[str] = None) -> Optional[int]:
    """Read a preferred UI port from an experiment config file when available.

    Resolution order:
      1) explicit_path (if provided)
      2) $WEIGHTSLAB_EXPERIMENT_CONFIG
      3) ./config.yaml, ./config.yml, ./experiment_config.yaml, ./experiment_config.yml
    """
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    env_cfg = os.getenv("WEIGHTSLAB_EXPERIMENT_CONFIG", "").strip()
    if env_cfg:
        candidates.append(Path(env_cfg))
    cwd = Path.cwd()
    candidates.extend([
        cwd / "config.yaml",
        cwd / "config.yml",
        cwd / "experiment_config.yaml",
        cwd / "experiment_config.yml",
    ])

    first_existing = next((p for p in candidates if p.exists() and p.is_file()), None)
    if first_existing is None:
        return None

    try:
        payload = yaml.safe_load(first_existing.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug(f"Could not parse experiment config {first_existing}: {exc}")
        return None

    if not isinstance(payload, dict):
        return None

    candidate_values = [
        payload.get("ui_port"),
        payload.get("studio_port"),
        _lookup_nested(payload, ("studio", "port")),
        _lookup_nested(payload, ("weightslab", "ui_port")),
        _lookup_nested(payload, ("weightslab", "studio", "port")),
    ]
    for val in candidate_values:
        coerced = _coerce_valid_port(val)
        if coerced is not None:
            return coerced
    return None


def _resolve_ui_port(args) -> tuple[int, str]:
    """Resolve preferred UI port source: arg > config > env > default."""
    arg_port = _coerce_valid_port(getattr(args, "port", None))
    if arg_port is not None:
        return arg_port, "--port"

    config_port = _load_ui_port_from_experiment_config(getattr(args, "config", None))
    if config_port is not None:
        return config_port, "experiment config"

    env_port = _coerce_valid_port(os.getenv("WL_LAST_UI_PORT"))
    if env_port is not None:
        return env_port, "WL_LAST_UI_PORT"

    compat_env_port = _coerce_valid_port(os.getenv("WEIGHTSLAB_UI_PORT"))
    if compat_env_port is not None:
        return compat_env_port, "WEIGHTSLAB_UI_PORT"

    return 50051, "default"


def _persist_certs_dir(certs_dir_str: str) -> None:
    """Persist WEIGHTSLAB_CERTS_DIR so future terminals and the training backend find it.

    Windows — runs `setx` (permanent user env) and prints the PS one-liner for
               the current session.
    Linux/macOS — appends an export line to ~/.bashrc (idempotent) and prints the
               source command for the current session.
    """
    export_line = f'export WEIGHTSLAB_CERTS_DIR="{certs_dir_str}"'
    if _is_windows():
        result = subprocess.run(
            ["setx", "WEIGHTSLAB_CERTS_DIR", certs_dir_str],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if result.returncode == 0:
            logger.info(" WEIGHTSLAB_CERTS_DIR saved permanently via setx (new terminals will have it)")
        else:
            logger.warning(f"setx failed — set it manually: setx WEIGHTSLAB_CERTS_DIR \"{certs_dir_str}\"")
        logger.info(f" Current terminal (PowerShell): $env:WEIGHTSLAB_CERTS_DIR = \"{certs_dir_str}\"")
    else:
        bashrc = Path.home() / ".bashrc"
        try:
            existing = bashrc.read_text(encoding="utf-8") if bashrc.exists() else ""
            if export_line not in existing:
                with open(bashrc, "a", encoding="utf-8") as f:
                    f.write(f"\n# Added by weightslab\n{export_line}\n")
                logger.info(f" WEIGHTSLAB_CERTS_DIR appended to {bashrc} (new terminals will have it)")
            else:
                logger.info(f" WEIGHTSLAB_CERTS_DIR already in {bashrc}")
        except OSError as e:
            logger.warning(f"Could not write to {bashrc}: {e}")
            logger.info(f" Add manually: {export_line}")
        logger.info(f" Current terminal: source ~/.bashrc (or open a new terminal)")


def _banner() -> str:
    """Return the WeightsLab ASCII banner, or a plain title if art is unavailable."""
    try:
        from weightslab.art import _BANNER
        return _BANNER
    except Exception:
        return "WeightsLab"


_DESCRIPTION = (
    _banner()
    + "\nWeightsLab — Inspect, Edit, and Evolve Neural Networks\n"
    + "Run the Weights Studio UI (Docker-free), bundled examples, and the secure "
    + "(TLS + gRPC auth) environment."
)

_EPILOG = """\
commands:
  se                       Set up the secure environment: generate TLS
                           certificates + a gRPC auth token in
                           ~/.weightslab-certs. Then set WEIGHTSLAB_CERTS_DIR
                           (the single source of truth) so the backend + new
                           shells find them.
                             --force-certs   regenerate even if certs exist

  start                    Start the Weights Studio UI natively — no Docker.
                           Serves the bundled SPA and proxies gRPC-Web to a
                           running backend, all from one Python process.
                           UNSECURED (HTTP) by default.
                             --port PORT           UI HTTP port (default 50051)
                             --config FILE         experiment config file used to read ui_port
                             --backend-port PORT   backend gRPC port (default 50051)
                             --backend-host HOST   backend gRPC host (default localhost)
                             --no-browser          don't open a browser
                             --certs               HTTPS + mTLS from $WEIGHTSLAB_CERTS_DIR

  start example            Run a bundled PyTorch example (foreground; stop with
                           Ctrl+C). Installs the example's requirements first,
                           without prompting. Defaults to classification:
                             --cls      classification example (default)
                             --seg      segmentation example
                             --det      detection example
                             --clus     clustering example
                             --gen      generation example
                             --3d_det   3D LiDAR point-cloud detection example
                             --2d_det   2D LiDAR point-cloud detection example

  cli                      Open an interactive terminal connected to a
                           currently-running experiment (pause/resume, status,
                           evaluate, agent query, etc.). Auto-discovers the
                           running experiment; the experiment must be serving
                           the CLI (e.g. wl.serve(serving_cli=True)).
                             --port PORT   connect to a specific CLI port
                             --host HOST   connect to a specific host (default: localhost)

  tunnel                   Forward a REMOTE gRPC backend (e.g. a Colab run
                           behind `ngrok tcp 50051`) to a LOCAL port so
                           `weightslab start` can proxy to it. Raw TCP, so the
                           backend must be plaintext (the default).
                             ENDPOINT          remote host:port (e.g. bore.pub:12345)
                                               (default: $WEIGHTSLAB_TUNNEL_ENDPOINT)
                             --listen-port N   local port to expose (default 50051)
                             --listen-host H   interface to bind (default: auto)
                             --remote-port N   remote port, if not in ENDPOINT

examples:
  weightslab se                       # one-time secure setup (then export WEIGHTSLAB_CERTS_DIR)
  weightslab se --force-certs         # regenerate the certs
    weightslab start                    # launch the UI (unsecured HTTP, default) at :50051
  weightslab start --certs            # launch the UI over HTTPS (needs `weightslab se` first)
  weightslab start --port 9000        # launch the UI on a custom port
  weightslab start --backend-port 50052   # proxy to a backend on a custom gRPC port
  weightslab start example            # run the classification demo (default)
  weightslab start example --seg      # run the segmentation demo
  weightslab start example --det      # run the detection demo
  weightslab start example --3d_det   # run the 3D LiDAR detection demo
  weightslab cli                      # connect a terminal to the running experiment
  weightslab cli --port 60000         # connect to a specific CLI port
  weightslab tunnel bore.pub:12345    # expose a remote (Colab) backend at localhost:50051
"""


def _get_cert_script() -> Path:
    """Get the generate-certs-auth-token.sh script path."""
    return Path(__file__).parent / 'ui' / 'utils' / 'generate-certs-auth-token.sh'


def _get_cert_script_ps1() -> Path:
    """Get the generate-certs-auth-token.ps1 script path."""
    return Path(__file__).parent / 'ui' / 'utils' / 'generate-certs-auth-token.ps1'


def _is_windows() -> bool:
    """Check if running on Windows."""
    return sys.platform == 'win32'


def _make_executable(path) -> None:
    """Add the execute bit to a file so it can be run directly (POSIX only).

    pip installs the bundled ``.sh`` scripts as package data *without* the execute
    bit, and they are committed/shipped non-executable. ``_run_shell_script`` runs
    them as a command (``bash -c "VAR=... '/path/script.sh'"``), which requires
    the execute bit — otherwise bash fails with "Permission denied" until the user
    ``chmod +x`` by hand. This grants u+x,g+x,o+x (e.g. 0644 -> 0755) and is a
    best-effort no-op on Windows, where the POSIX execute bit is not used.
    """
    if _is_windows():
        return
    try:
        mode = os.stat(path).st_mode
        os.chmod(path, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError as exc:
        logger.debug(f"Could not set execute bit on {path}: {exc}")


def _ensure_scripts_executable() -> None:
    """Make the bundled cert-generation shell script executable (POSIX only).

    Covers generate-certs-auth-token.sh under ``weightslab/ui/utils`` so a freshly
    pip-installed package can run it without the user having to ``chmod +x`` first.
    No-op on Windows.
    """
    if _is_windows():
        return
    utils_dir = Path(__file__).parent / 'ui' / 'utils'
    try:
        scripts = list(utils_dir.rglob('*.sh'))
    except OSError as exc:
        logger.debug(f"Could not enumerate bundled scripts under {utils_dir}: {exc}")
        return
    for script in scripts:
        _make_executable(script)


def _run_powershell_script(script_path: str, args: list = None, env_vars: dict = None) -> int:
    """Run a PowerShell script and return exit code."""
    if not _is_windows():
        logger.error("PowerShell certificate generation requires Windows")
        return 1

    cmd = [
        'powershell',
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-File', script_path
    ]

    if args:
        cmd.extend(args)

    try:
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)
        result = subprocess.run(cmd, env=env)
        return result.returncode
    except Exception as e:
        logger.error(f"Failed to run script: {e}")
        return 1


def _convert_to_git_bash_path(win_path: str) -> str:
    """Convert Windows path to Git Bash compatible format."""
    # Normalize backslashes to forward slashes explicitly: Path(...).as_posix()
    # does NOT convert '\' on POSIX hosts (e.g. Linux CI runners), so a Windows
    # input path would keep its separators there.
    p = str(win_path).replace("\\", "/")
    # Convert C:/Users/... to /mnt/c/Users/... for Git Bash
    if len(p) > 1 and p[1] == ':':
        drive = p[0].lower()
        rest = p[2:]
        return f"/mnt/{drive}{rest}"
    return p


def _run_shell_script(script_path: str, args: list = None, env_vars: dict = None) -> int:
    """Run a shell script using bash with proper environment variable passing."""

    try:
        # Fix line endings in the file before running
        with open(script_path, 'rb') as f:
            script_bytes = f.read()

        # Ensure Unix line endings
        fixed_bytes = script_bytes.replace(b'\r\n', b'\n').replace(b'\r', b'\n')

        # Write back if needed
        if fixed_bytes != script_bytes:
            with open(script_path, 'wb') as f:
                f.write(fixed_bytes)

        # We invoke the script as a command below (bash -c "VAR=... 'script'"),
        # which needs the execute bit. pip installs it without one, so add it here
        # (no-op on Windows). Uses the on-disk path before any Git Bash conversion.
        _make_executable(script_path)

        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)
            # Debug: log all environment variables being passed
            for key, value in env_vars.items():
                logger.info(f"Passing env var: {key}={value}")

        # Build bash command - pass Windows path directly, script will handle conversion
        # # Process path to ensure it's compatible with bash, especially on Windows
        if _is_windows() and '\\' in script_path:
            script_path = script_path.replace("\\", "/") # Ensure path is Unix-style for bash
            script_path = _convert_to_git_bash_path(script_path)
            logger.info(f"Converted script path for bash: {script_path}")
        logger.info(f"Running shell script: {script_path} with args: {args} and env_vars: {env_vars}")

        # Build environment variable assignments for bash command
        env_assignments = ' '.join([f"{k}='{v}'" for k, v in env_vars.items()]) if env_vars else ""

        if env_assignments:
            # Pass env vars directly in bash command using -c flag
            # This works on Windows (Git Bash), Linux, and macOS
            logger.info('Using env assignments')
            bash_script_cmd = f"{env_assignments} '{script_path}'"
            if args:
                bash_script_cmd += " " + " ".join(f"'{arg}'" for arg in args)
            bash_cmd = ['bash', '-c', bash_script_cmd]
        else:
            logger.info('Not using env assignments')
            bash_cmd = ['bash', '-x', str(script_path)]
            if args:
                bash_cmd.extend(args)

        result = subprocess.run(bash_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.stdout:
            for line in result.stdout.splitlines():
                logger.info(line)
        return result.returncode
    except FileNotFoundError:
        logger.error(f"Script file not found: {script_path}")
        return 1
    except Exception as e:
        logger.error(f"Failed to run script: {e}")
        return 1


def _generate_certs_with_fallback(force_certs: bool = False, certs_dir=None) -> int:
    """Try shell script first, fall back to PowerShell on Windows if it fails.

    ``certs_dir`` is forwarded to the generation scripts as ``WEIGHTSLAB_CERTS_DIR``
    so certs land in the single source-of-truth directory (the scripts default to
    ``~/.weightslab-certs`` when it is not provided). The shell script receives a
    POSIX absolute path (``/mnt/c/...`` on Windows/WSL); PowerShell receives the
    host-native path (``C:/...``).
    """
    env_vars = None
    if certs_dir is not None:
        # Shell scripts (bash/WSL) need a POSIX-style absolute path.
        if _is_windows():
            bash_certs_dir = _convert_to_git_bash_path(str(certs_dir))
        else:
            bash_certs_dir = str(certs_dir)
        env_vars = {'WEIGHTSLAB_CERTS_DIR': bash_certs_dir}

    cert_script = str(_get_cert_script())
    if not Path(cert_script).exists():
        logger.warning(f"Shell script not found: {cert_script}")
    else:
        script_args = []
        if force_certs:
            script_args.append('--force-create-certs')

        logger.info("Attempting certificate generation with shell script...")
        exit_code = _run_shell_script(cert_script, script_args, env_vars)
        if exit_code == 0:
            return 0
        logger.warning(f"Shell script failed (exit code {exit_code})")

    # Fallback to PowerShell on Windows
    if _is_windows():
        logger.info("Falling back to PowerShell for certificate generation...")
        cert_script_ps1 = str(_get_cert_script_ps1())
        if not Path(cert_script_ps1).exists():
            logger.error(f"PowerShell script not found: {cert_script_ps1}")
            return 1

        script_args = []
        if force_certs:
            script_args.append('-ForceCreateCerts')

        exit_code = _run_powershell_script(cert_script_ps1, script_args, env_vars)
        return exit_code
    else:
        logger.error("Neither shell nor PowerShell script could generate certificates")
        return 1


def ui_secure_environment(args):
    """`weightslab se`: create a certs directory with certs + gRPC token.

    The directory is the single source of truth — WEIGHTSLAB_CERTS_DIR is exported
    for this process and the user is asked to export it globally. Everything else
    (TLS on/off, auth on/off) is derived from the files in that directory by the
    backend and `weightslab start --certs`, so this command sets no other env.
    """
    logger.info("Setting up secure environment...")
    # Bundled .sh scripts ship without the execute bit (pip strips it); make them
    # runnable so the cert-generation script below doesn't fail on "Permission denied".
    _ensure_scripts_executable()

    force_certs = getattr(args, "force_certs", False)
    no_auth = getattr(args, "no_auth", False)
    certs_dir = getattr(args, "certs_dir", None)
    if certs_dir:
        # Absolute path so Windows Python, WSL bash and the server agree on location.
        certs_dir = str(Path(certs_dir).resolve())

    # Resolve the target directory first (no filesystem work in __init__), so we
    # can point the generation scripts at it via WEIGHTSLAB_CERTS_DIR.
    manager = CertAuthManager(certs_dir=certs_dir, enable_auth=not no_auth)

    exit_code = _generate_certs_with_fallback(force_certs=force_certs, certs_dir=manager.certs_dir)
    if exit_code != 0:
        logger.error("Certificate generation failed")
        sys.exit(1)

    manager.certs_dir.mkdir(parents=True, exist_ok=True)
    manager.get_or_create_auth_token()

    # Export ONLY the single source of truth for this process.
    os.environ["WEIGHTSLAB_CERTS_DIR"] = str(manager.certs_dir)

    logger.info(" Certificates generated successfully")
    logger.info(" gRPC auth token created")
    logger.info(f" Certs and token stored in: {manager.certs_dir}")
    logger.info(f" WEIGHTSLAB_CERTS_DIR exported for this process: {manager.certs_dir}")
    logger.info("Then launch the secured UI with: weightslab start --certs")
    # Keep this the FINAL output so the user can't miss the action they must take.
    logger.warning("")
    logger.warning(" ACTION REQUIRED — set WEIGHTSLAB_CERTS_DIR globally so new shells "
                   "and the training backend find these certs (single source of truth):")
    logger.warning(f" (bash) echo 'export WEIGHTSLAB_CERTS_DIR=\"{manager.certs_dir}\"' >> ~/.bashrc && source ~/.bashrc")
    logger.warning(f" (Windows) setx WEIGHTSLAB_CERTS_DIR \"{manager.certs_dir}\"")


# Bundled PyTorch examples, keyed by the CLI flag (e.g. --cls -> wl-classification).
# kind -> (dir_name, label, category) where category is the examples/ subfolder.
_EXAMPLES = {
    "cls": ("wl-classification", "classification", "PyTorch"),
    "seg": ("wl-segmentation", "segmentation", "PyTorch"),
    "det": ("wl-detection", "detection", "PyTorch"),
    "clus": ("wl-clustering", "clustering", "PyTorch"),
    "gen": ("wl-generation", "generation", "PyTorch"),
    "3d_det": ("wl-3d-lidar-detection", "3D LiDAR detection", "Usecases"),
    "2d_det": ("wl-2d-lidar-detection", "2D LiDAR detection", "Usecases"),
}
_DEFAULT_EXAMPLE = "cls"


def _get_example_dir(name: str = "wl-classification", category: str = "PyTorch") -> Path:
    """Path to a bundled example directory (under examples/<category>/<name>)."""
    return Path(__file__).parent / 'examples' / category / name


def _install_example_requirements(example_dir: Path) -> None:
    """Install an example's requirements non-interactively, if a file is present.

    Looks for requirements.txt (then requirements.in) in the example directory and
    runs `pip install -r` with the current interpreter and `--no-input` so it never
    prompts. Non-fatal: a failure is logged and the example is still attempted, so a
    transient install hiccup doesn't block a run where deps are already satisfied.
    """
    for fname in ("requirements.txt", "requirements.in"):
        req = example_dir / fname
        if not req.exists():
            continue
        logger.info(f"Installing example requirements (non-interactive): {req}")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(req),
                 "--no-input", "--disable-pip-version-check"],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            logger.warning(
                f"Failed to install requirements ({req}): {exc}. "
                "Continuing — the example may still run if deps are already installed."
            )
        return # only the first matching requirements file is used


def example_start(args):
    """`weightslab start example [--cls|--seg|--clus|--gen]`: run a bundled example.

    Defaults to the classification (cls) example. First installs the example's
    requirements (if a requirements file is present) without prompting, then runs
    its main.py with the current Python interpreter from its own directory so it
    resolves its sibling config.yaml. Runs in the foreground (serves until Ctrl+C).
    """
    kind = getattr(args, "example_kind", None) or _DEFAULT_EXAMPLE
    dir_name, label, category = _EXAMPLES.get(kind, _EXAMPLES[_DEFAULT_EXAMPLE])

    example_dir = _get_example_dir(dir_name, category)
    main_py = example_dir / "main.py"
    if not main_py.exists():
        logger.error(f"{label.capitalize()} example not found: {main_py}")
        sys.exit(1)

    # Install the example's own requirements first, without any interaction.
    _install_example_requirements(example_dir)

    logger.info(f"Starting the WeightsLab {label} ({kind}) example...")
    logger.info(f" {main_py}")
    logger.info("In another terminal, launch the UI with: weightslab start")
    logger.info("Then open the URL printed by `weightslab start` — stop the example with Ctrl+C.")
    if not _CERTS_DIR_IN_ORIGINAL_ENV:
        manager = CertAuthManager.from_env_or_default()
        if manager.has_valid_certs():
            logger.warning(
                "WEIGHTSLAB_CERTS_DIR is not set in your shell environment. "
                "TLS will work this session (certs found at default location) "
                "but may not persist across terminals."
            )
            _persist_certs_dir(str(manager.certs_dir))
    try:
        env = os.environ.copy()
        env['WEIGHTSLAB_SUPPRESS_BANNER'] = '1'
        result = subprocess.run([sys.executable, str(main_py)], cwd=str(example_dir), env=env)
    except KeyboardInterrupt:
        logger.info("Example stopped.")
        return
    if result.returncode != 0:
        sys.exit(result.returncode)


def cli_connect(args):
    """`weightslab cli [--port N] [--host H]`: open an interactive terminal
    connected to a currently-running experiment's CLI server.

    With no --port, auto-discovers the running experiment (the backend advertises
    its actual port on startup). Pass --port to target a specific server.
    """
    try:
        import weightslab.backend.cli as cli_backend
    except Exception as exc:
        logger.error(f"Could not load the WeightsLab CLI client: {exc}")
        sys.exit(1)

    port = getattr(args, "port", None)
    host = getattr(args, "host", None)
    exit_code = cli_backend.cli_connect(cli_port=port, cli_host=host)
    sys.exit(exit_code)


def ui_start_native(args):
    """`weightslab start`: launch the Weights Studio UI natively (no Docker).

    Serves the pre-built SPA vendored in this package and proxies gRPC-Web to a
    running backend (started by ``wl.serve()``), all from one pure-Python HTTP
    server — like ``tensorboard``, the UI ships in the wheel.

    Unsecured HTTP by default. Pass ``--certs`` to serve HTTPS + mTLS to the
    backend, derived solely from cert-file presence in $WEIGHTSLAB_CERTS_DIR.
    """
    try:
        from weightslab.ui import server as ui_server
    except Exception as exc:  # pragma: no cover - import guard
        logger.error(f"Could not load the WeightsLab UI server: {exc}")
        sys.exit(1)

    ui_host = getattr(args, "host", None) or os.getenv("WEIGHTSLAB_UI_HOST", "0.0.0.0")
    preferred_ui_port, ui_port_source = _resolve_ui_port(args)
    ui_port = preferred_ui_port
    backend_host = (getattr(args, "backend_host", None)
                    or os.getenv("GRPC_BACKEND_HOST", "localhost"))
    backend_port = (getattr(args, "backend_port", None)
                    or int(os.getenv("GRPC_BACKEND_PORT", "50051")))
    open_browser = not getattr(args, "no_browser", False)

    # Never bind the UI server on the same TCP port as the backend gRPC target.
    # If `weightslab start` grabs backend_port first, the training process cannot
    # start its gRPC server and the UI appears disconnected.
    if ui_port == backend_port:
        logger.warning(
            f"Requested UI port {ui_port} matches backend gRPC port {backend_port}; "
            "choosing a random free UI port to avoid collision."
        )
        ui_port = 0

    # TLS/auth: single source of truth is cert-file presence in the certs dir.
    # Only consulted when the user explicitly opts in with --certs.
    certs_dir = None
    grpc_auth_token = None
    if getattr(args, "certs", False):
        manager = CertAuthManager.from_env_or_default()
        if manager.has_valid_certs():
            certs_dir = str(manager.certs_dir)
            try:
                grpc_auth_token = manager.get_or_create_auth_token()
            except Exception:
                grpc_auth_token = None
        else:
            logger.warning(
                f"--certs requested but no valid certs in {manager.certs_dir}. "
                "Run `weightslab se` first. Falling back to unsecured HTTP."
            )

    if not ui_server.has_static_assets():
        logger.warning(
            "No bundled UI assets found in this install. The gRPC-Web proxy will "
            "still run, but the web page is unavailable. Build the frontend and "
            "vendor it into this package (weights_studio: `ui/utils/build-and-deploy.sh`, "
            "then `weightslab/ui/utils/sync-frontend.sh`)."
        )

    # If the preferred UI port is taken, fall back to a free one.
    probe_host = "127.0.0.1" if ui_host in ("0.0.0.0", "::", "") else ui_host
    actual_port = ui_server.find_free_port(ui_port, host=probe_host)
    if actual_port != ui_port:
        logger.warning(f"Port {ui_port} is busy; using {actual_port} instead.")
        ui_port = actual_port
    logger.info(f"UI port source: {ui_port_source} (preferred {preferred_ui_port}, using {ui_port})")
    os.environ["WL_LAST_UI_PORT"] = str(ui_port)

    ui_server.serve_ui(
        ui_host=ui_host,
        ui_port=ui_port,
        backend_host=backend_host,
        backend_port=backend_port,
        open_browser=open_browser,
        certs_dir=certs_dir,
        grpc_auth_token=grpc_auth_token,
        block=True,
    )


def _add_ui_server_flags(p: argparse.ArgumentParser) -> None:
    """Attach flags for the native (Docker-free) UI server."""
    p.add_argument('--port', type=int, default=None,
                   help='UI HTTP port (default: config ui_port, else $WL_LAST_UI_PORT, else 50051)')
    p.add_argument('--config', default=None,
                   help='Experiment config file to read ui_port from (yaml/yml)')
    p.add_argument('--host', default=None,
                   help='UI bind host (default: 0.0.0.0)')
    p.add_argument('--backend-host', dest='backend_host', default=None,
                   help='Backend gRPC host to proxy to (default: localhost)')
    p.add_argument('--backend-port', dest='backend_port', type=int, default=None,
                   help='Backend gRPC port to proxy to (default: 50051)')
    p.add_argument('--no-browser', dest='no_browser', action='store_true',
                   help='Do not open the web browser automatically')
    p.add_argument('--certs', action='store_true',
                   help='Serve HTTPS + mTLS to the backend using TLS certs from '
                        '$WEIGHTSLAB_CERTS_DIR (default: unsecured HTTP). '
                        'Run `weightslab se` first to generate them.')


def _add_example_kind_flags(p: argparse.ArgumentParser) -> None:
    """Attach the mutually-exclusive example-kind flags (default: classification)."""
    group = p.add_mutually_exclusive_group()
    group.add_argument("--cls", action="store_const", dest="example_kind", const="cls",
                       help="Run the classification example (default)")
    group.add_argument("--seg", action="store_const", dest="example_kind", const="seg",
                       help="Run the segmentation example")
    group.add_argument("--det", action="store_const", dest="example_kind", const="det",
                       help="Run the detection example")
    group.add_argument("--clus", action="store_const", dest="example_kind", const="clus",
                       help="Run the clustering example")
    group.add_argument("--gen", action="store_const", dest="example_kind", const="gen",
                       help="Run the generation example")
    group.add_argument("--3d_det", action="store_const", dest="example_kind", const="3d_det",
                       help="Run the 3D LiDAR point-cloud detection example")
    group.add_argument("--2d_det", action="store_const", dest="example_kind", const="2d_det",
                       help="Run the 2D LiDAR point-cloud detection example")
    p.set_defaults(example_kind=_DEFAULT_EXAMPLE)


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser (banner + detailed command reference).

    The CLI is intentionally minimal — exactly these commands:
        weightslab --help | -h | help
        weightslab se [--force-certs]
        weightslab start [--port PORT] [--config FILE] [--backend-port PORT] [--certs]
        weightslab start example [--cls|--seg|--det|--clus|--gen|--3d_det|--2d_det]
        weightslab cli [--port PORT] [--host HOST]
        weightslab tunnel ENDPOINT
    """
    parser = argparse.ArgumentParser(
        prog="weightslab",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="{se,start,cli,tunnel,help}")

    # weightslab se [--force-certs] [certs_dir]
    se_parser = sub.add_parser("se", help="Set up the secure environment (TLS certs + gRPC auth token)")
    se_parser.add_argument('--force-certs', action='store_true', help='Regenerate certificates even if they already exist')
    se_parser.add_argument('certs_dir', nargs='?', default=None,
                           help='Custom directory for certs/token (default: $WEIGHTSLAB_CERTS_DIR or ~/.weightslab-certs)')

    # weightslab cli [--port N] [--host H]
    cli_parser = sub.add_parser(
        "cli", help="Open an interactive terminal connected to the running experiment")
    cli_parser.add_argument('--port', type=int, default=None,
                            help='CLI server port (default: auto-discover the running experiment)')
    cli_parser.add_argument('--host', default=None,
                            help='CLI server host (default: localhost)')

    # weightslab tunnel ENDPOINT [--listen-port N] [--listen-host H] [--remote-port N]
    tunnel_parser = sub.add_parser(
        "tunnel",
        help="Forward a remote gRPC backend (e.g. a Colab run via `ngrok tcp 50051`) "
             "to a local port so `weightslab start` can proxy to it")
    tunnel_parser.add_argument(
        'endpoint', nargs='?', default=None,
        help="Remote backend endpoint host:port (e.g. bore.pub:12345). "
             "A tcp:// prefix is accepted. Default: $WEIGHTSLAB_TUNNEL_ENDPOINT.")
    tunnel_parser.add_argument(
        '--listen-port', '-p', type=int, default=DEFAULT_LISTEN_PORT,
        help=f"Local port to expose (default: {DEFAULT_LISTEN_PORT} — the port `weightslab start` proxies to)")
    tunnel_parser.add_argument(
        '--listen-host', default=None,
        help="Local interface to bind (default: 127.0.0.1 on Windows/macOS, 0.0.0.0 on Linux)")
    tunnel_parser.add_argument(
        '--remote-port', type=int, default=None,
        help="Remote port, if not included in ENDPOINT")

    # weightslab start [--port N ...]        -> native (Docker-free) UI server
    # weightslab start example [--cls|--seg|--clus|--gen]  -> bundled example
    start_parser = sub.add_parser(
        "start",
        help="Start the Weights Studio UI natively (no Docker); "
             "or `start example` to run a bundled example")
    # Flags on the bare `start` command launch the native UI server.
    _add_ui_server_flags(start_parser)
    start_sub = start_parser.add_subparsers(dest="start_target")
    example_parser = start_sub.add_parser(
        "example", help="Start a bundled PyTorch example (default: classification)")
    _add_example_kind_flags(example_parser)

    # Tolerate the swapped order: `weightslab example start [flags]` (and bare
    # `weightslab example`) behave exactly like `weightslab start example`. Hidden
    # from --help on purpose (argparse.SUPPRESS) — a forgiving fallback.
    example_alias = sub.add_parser("example", help=argparse.SUPPRESS)
    example_alias_sub = example_alias.add_subparsers(dest="example_action")
    example_alias_start = example_alias_sub.add_parser(
        "start", help="Start a bundled PyTorch example (default: classification)")
    _add_example_kind_flags(example_alias_start)

    sub.add_parser("help", help="Show this help message")

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "help" or args.command is None:
        parser.print_help()
    elif args.command == "cli":
        cli_connect(args)
    elif args.command == "tunnel":
        from weightslab.tunnel import tunnel_connect
        tunnel_connect(args)
    elif args.command == "se":
        ui_secure_environment(args)
    elif args.command == "start":
        if getattr(args, "start_target", None) == "example":
            example_start(args)
        else:
            # Bare `weightslab start` -> launch the native (Docker-free) UI.
            ui_start_native(args)
    elif args.command == "example":
        # Alias for `start example` — tolerate the swapped subcommand order.
        example_start(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
