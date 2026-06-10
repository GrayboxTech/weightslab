"""Weights Studio Docker management with secure TLS."""

import argparse
import os
import re
import shutil
import subprocess
import sys
import socket
import logging
from pathlib import Path

from weightslab.security import CertAuthManager

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Docker resources owned by the bundled stack. Cleanup is scoped strictly to
# these names — we never run a global `docker system prune`.
_FRONTEND_IMAGE = "graybx/weightslab"
_STACK_CONTAINERS = ("weights_studio_envoy", "weights_studio_frontend")

# TLS/auth env vars that are *derived* from cert-file presence in
# WEIGHTSLAB_CERTS_DIR. WEIGHTSLAB_CERTS_DIR is the single source of truth; these
# are computed by the deploy pipeline (build-and-deploy.sh + the compose `auto`
# logic) from the actual files. They are stripped before launching so a stale or
# pre-set value can never override the file-based decision.
_DERIVED_DEPLOY_ENV_VARS = (
    "GRPC_TLS_ENABLED",
    "GRPC_TLS_REQUIRE_CLIENT_AUTH",
    "GRPC_TLS_CERT_FILE",
    "GRPC_TLS_KEY_FILE",
    "GRPC_TLS_CA_FILE",
    "ENVOY_UPSTREAM_TLS",
    "ENVOY_DOWNSTREAM_TLS",
    "WS_SERVER_PROTOCOL",
    "VITE_SERVER_PROTOCOL",
    "VITE_DEV_SERVER_HTTPS",
    "WL_ENABLE_GRPC_AUTH_TOKEN",
    "VITE_WL_ENABLE_GRPC_AUTH_TOKEN",
    "GRPC_AUTH_TOKEN",
    "VITE_GRPC_AUTH_TOKEN",
)


def _strip_derived_deploy_env() -> None:
    """Drop derived TLS/auth env vars so the deploy pipeline decides solely from
    cert-file presence in WEIGHTSLAB_CERTS_DIR (the single source of truth)."""
    for key in _DERIVED_DEPLOY_ENV_VARS:
        os.environ.pop(key, None)


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
    + "Manage the Weights Studio UI, its Docker stack, and the secure "
    + "(TLS + gRPC auth) environment."
)

_EPILOG = """\
commands:
  se                       Set up the secure environment: generate TLS
                           certificates + a gRPC auth token in
                           ~/.weightslab-certs and export WEIGHTSLAB_CERTS_DIR.
                             --force-certs   regenerate even if certs exist

  ui launch                Generate certificates if missing, purge stale
                           weightslab/weights_studio Docker resources, then
                           build & start the UI stack.
                             --no-certs      run unsecured (HTTP, no auth)

  start example            Run the bundled classification (cls) example
                           (foreground; stop with Ctrl+C).

examples:
  weightslab se                       # one-time secure setup
  weightslab se --force-certs         # regenerate the certs
  weightslab ui launch                # certs-if-missing + clean + launch
  weightslab ui launch --no-certs     # unsecured launch (HTTP)
  weightslab start example            # run the classification demo
"""


def _get_compose_file():
    """Return the path to the bundled docker-compose.yml."""
    # return files("weightslab.ui.docker") / "docker-compose.yml"
    return Path(__file__).parent / 'ui' / 'docker' / 'docker-compose.yml'


def _get_envoy_config():
    """Return the path to the bundled envoy.yaml."""
    # return files("weightslab.ui.envoy") / "envoy.yaml"
    return Path(__file__).parent / 'ui' / 'docker' / 'envoy.yaml'


def _get_bootstrap_script() -> Path:
    """Get the bootstrap-secure.ps1 script path."""
    return Path(__file__).parent / 'ui' / 'docker' / 'utils' / 'build-and-deploy.sh'


def _get_cert_script() -> Path:
    """Get the generate-certs-auth-token.sh script path."""
    return Path(__file__).parent / 'ui' / 'docker' / 'utils' / 'generate-certs-auth-token.sh'


def _get_cert_script_ps1() -> Path:
    """Get the generate-certs-auth-token.ps1 script path."""
    return Path(__file__).parent / 'ui' / 'docker' / 'utils' / 'generate-certs-auth-token.ps1'


def _is_windows() -> bool:
    """Check if running on Windows."""
    return sys.platform == 'win32'


def _check_docker():
    """Verify that docker is installed and the daemon is running."""
    if shutil.which("docker") is None:
        logger.error(
            "Docker is required but not found on your PATH.\n"
            "Install it from: https://docs.docker.com/get-docker/"
        )
        sys.exit(1)

    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        logger.error(
            "Docker is installed but the daemon is not running.\n"
            "Start it with: Docker Desktop or 'sudo systemctl start docker'"
        )
        sys.exit(1)


def _compose_cmd(compose_file, envoy_config, action):
    """Build and run a docker compose command."""
    env = os.environ.copy()
    env["WS_ENVOY_CONFIG"] = str(envoy_config)

    cmd = ["docker", "compose", "-f", str(compose_file)] + action
    result = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    if result.stdout:
        for line in result.stdout.splitlines():
            logger.info(line)

    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def _test_backend_connection(host: str = '127.0.0.1', port: int = 50051, timeout: float = 5.0) -> bool:
    """Test if backend gRPC server is reachable."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        logger.debug(f"Backend connection test failed: {e}")
        return False


def _run_powershell_script(script_path: str, args: list = None, env_vars: dict = None) -> int:
    """Run a PowerShell script and return exit code."""
    if not _is_windows():
        logger.error("Secure launch requires Windows with PowerShell")
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


def _to_docker_host_path(path) -> str:
    """Return a path that Docker Desktop can bind-mount.

    On Windows, normalize WSL (/mnt/c/...) and Git Bash (/c/...) forms — and
    native paths that Path() mangled into \\mnt\\c\\... — to C:/... form.
    Docker Desktop cannot resolve /mnt-style sources and silently mounts an
    empty directory, which crashes Envoy on missing certs. On Linux/macOS the
    path is returned unchanged.
    """
    p = str(path).replace('\\', '/')
    if _is_windows():
        m = re.match(r'^/mnt/([a-zA-Z])/(.*)$', p) or re.match(r'^/([a-zA-Z])/(.*)$', p)
        if m:
            return f"{m.group(1).upper()}:/{m.group(2)}"
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

        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)
            # Debug: log all environment variables being passed
            for key, value in env_vars.items():
                logger.info(f"Passing env var: {key}={value}")

        # Build bash command - pass Windows path directly, script will handle conversion
        # # Process path to ensure it's compatible with bash, especially on Windows
        if _is_windows() and '\\' in script_path:
            script_path = script_path.replace("\\", "/")  # Ensure path is Unix-style for bash
            script_path = _convert_to_git_bash_path(script_path)
            logger.info(f"Converted script path for bash: {script_path}")
        logger.info(f"Running shell script: {script_path} with args: {args} and env_vars: {env_vars}")

        # Build environment variable assignments for bash command
        env_assignments = ' '.join([f"{k}='{v}'" for k, v in env_vars.items()]) if env_vars else ""
        cwd_path = str(Path.home())

        if env_assignments:
            # Pass env vars directly in bash command using -c flag
            # This works on Windows (Git Bash), Linux, and macOS
            logger.info('Using env assignments')
            bash_script_cmd = f"{env_assignments} '{script_path}'"
            if args:
                bash_script_cmd += " " + " ".join(f"'{arg}'" for arg in args)
            else:
                bash_script_cmd += " " + cwd_path
            bash_cmd = ['bash', '-c', bash_script_cmd]
        else:
            logger.info('Not using env assignments')
            bash_cmd = ['bash', '-x', str(script_path), cwd_path]
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
    ``~/.weightslab-certs`` when it is not provided). The host-native path
    (``C:/...`` on Windows) is understood by both Git Bash and PowerShell.
    """
    env_vars = None
    if certs_dir is not None:
        env_vars = {'WEIGHTSLAB_CERTS_DIR': _to_docker_host_path(certs_dir)}

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


def _ensure_certificates(manager: CertAuthManager, force_certs: bool = False) -> bool:
    """Generate certs + auth token in ``manager.certs_dir`` if missing (or forced).

    Does NOT export any TLS/auth env vars: the launch pipeline derives TLS purely
    from cert-file presence in ``WEIGHTSLAB_CERTS_DIR`` (the single source of
    truth). Returns True if certs are present afterwards, False otherwise.
    """
    if manager.has_valid_certs() and not force_certs:
        logger.info(f"✓ Using existing certificates in {manager.certs_dir}")
        manager.get_or_create_auth_token()
        return True

    logger.info(
        "Regenerating certificates (--force-certs)..."
        if force_certs
        else "No certificates found — generating them now..."
    )
    manager.certs_dir.mkdir(parents=True, exist_ok=True)
    exit_code = _generate_certs_with_fallback(force_certs=force_certs, certs_dir=manager.certs_dir)
    if exit_code != 0:
        logger.warning("Certificate generation failed — continuing in unsecured mode")
        return False

    manager.get_or_create_auth_token()
    logger.info(f"✓ Certificates ready in {manager.certs_dir}")
    return manager.has_valid_certs()


def _remove_docker_image(image: str) -> None:
    """Remove a Docker image (all tags) if present; no-op when absent."""
    result = subprocess.run(
        ["docker", "images", "-q", image],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    image_ids = sorted(set(result.stdout.split()))
    if not image_ids:
        return
    logger.info(f"Removing cached image '{image}' ({len(image_ids)} ref(s))...")
    subprocess.run(
        ["docker", "rmi", "-f", *image_ids],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _clean_stale_docker_resources() -> None:
    """Remove leftover weightslab / weights_studio Docker state before a launch.

    Stale containers, the compose network, anonymous volumes, a cached frontend
    image, and a leftover generated ``.env`` can each silently break a fresh
    launch (an old image served instead of a rebuild, or an empty cert mount
    that crashes Envoy). This is scoped STRICTLY to weightslab/weights_studio
    resources — it never runs a global ``docker system prune``.
    """
    logger.info("Cleaning stale Docker resources (weightslab/weights_studio only)...")
    compose_file = _get_compose_file()
    envoy_config = _get_envoy_config()

    # 1. Tear down any existing stack: containers + default network + anon volumes.
    try:
        _compose_cmd(compose_file, envoy_config, ["down", "--remove-orphans", "--volumes"])
    except subprocess.CalledProcessError as exc:
        logger.debug(f"'compose down' returned non-zero (nothing to remove?): {exc}")

    # 2. Force-remove leftover named containers started outside this compose project.
    for container in _STACK_CONTAINERS:
        subprocess.run(
            ["docker", "rm", "-f", container],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    # 3. Drop the cached frontend image so a fresh one is built/pulled.
    _remove_docker_image(_FRONTEND_IMAGE)

    # 4. Remove the generated .env so stale values don't leak into compose.
    env_file = Path(compose_file).parent / ".env"
    if env_file.exists():
        try:
            env_file.unlink()
            logger.info(f"Removed stale env file: {env_file}")
        except OSError as exc:
            logger.debug(f"Could not remove {env_file}: {exc}")


def ui_launch(args):
    """Generate certs if missing, purge stale Docker state, then build & start the UI.

    Flags (all optional, read defensively so legacy callers still work):
      --no-certs     skip certificate generation/usage (HTTP, no gRPC auth)
      --no-auth      TLS without a gRPC auth token
      --force-certs  regenerate certificates even if they already exist
      --no-clean     skip the stale Docker resource cleanup step
      --dev          use the dev compose overlay
    """
    _check_docker()

    no_certs = getattr(args, "no_certs", False)
    no_auth = getattr(args, "no_auth", False)
    force_certs = getattr(args, "force_certs", False)
    no_clean = getattr(args, "no_clean", False)

    manager = CertAuthManager.from_env_or_default(enable_auth=not no_auth)

    if no_certs:
        logger.info("⚠ --no-certs: launching in unsecured mode (HTTP, no gRPC auth)")
        # Ensure the certs dir exists (empty is fine) so the compose bind-mount
        # has a real, deterministic source. Envoy/nginx run plaintext and ignore
        # whatever is (or isn't) inside it.
        manager.certs_dir.mkdir(parents=True, exist_ok=True)
    else:
        _ensure_certificates(manager, force_certs=force_certs)

    # Remove stale weightslab/weights_studio Docker resources that could prevent
    # a clean launch. Scoped strictly to our own resources (see helper docstring).
    if no_clean:
        logger.info("Skipping stale Docker resource cleanup (--no-clean)")
    else:
        _clean_stale_docker_resources()

    # Single source of truth: the deploy pipeline (build-and-deploy.sh + the
    # compose `auto` logic) derives TLS/auth solely from cert-file presence in
    # WEIGHTSLAB_CERTS_DIR. Strip any pre-set/derived TLS env so it cannot
    # override that file-based decision.
    _strip_derived_deploy_env()

    # Run bootstrap script to setup environment
    bootstrap_script = str(_get_bootstrap_script())
    if Path(bootstrap_script).exists():
        logger.info("Running bootstrap script...")
        logger.info(f"Bootstrap script path: {bootstrap_script}")
        # Convert Windows path to Unix-style for bash
        certs_dir_str = str(manager.certs_dir)
        if _is_windows() and '\\' in certs_dir_str:
            certs_dir_str = _convert_to_git_bash_path(certs_dir_str)
            logger.info(f"Converted path to Unix-style: {certs_dir_str}")

        # Calculate WEIGHTSLAB_ROOT (parent of weightslab package)
        weightslab_root = str(Path(__file__).parent.parent)
        if _is_windows() and '\\' in weightslab_root:
            weightslab_root = _convert_to_git_bash_path(weightslab_root)

        # Docker Desktop (Windows) bind mounts need a host-native path
        # (e.g. C:/Users/...), NOT the /mnt/c Unix path used for the bash
        # script's own file checks. as_posix() yields C:/... on Windows and a
        # normal /abs/path on Linux/macOS — correct for docker compose on every
        # platform. The bash script writes this into .env for the compose mount.
        certs_dir_host = _to_docker_host_path(manager.certs_dir)

        # Always pass the real certs dir so the compose bind-mount has a valid
        # source. With --no-certs we add --unsecure, which forces Envoy/nginx to
        # plaintext (mounted files, if any, are ignored) without leaving the mount
        # source empty.
        bootstrap_env_vars = {
            'WEIGHTSLAB_CERTS_DIR': certs_dir_str,
            'WEIGHTSLAB_CERTS_DIR_HOST': certs_dir_host,
            'WEIGHTSLAB_ROOT': weightslab_root
        }
        script_args = []
        if no_certs:
            script_args.append('--unsecure')
        if getattr(args, "dev", False):
            script_args.append('--dev')
        logger.info(f"WEIGHTSLAB_CERTS_DIR={bootstrap_env_vars['WEIGHTSLAB_CERTS_DIR']}")
        logger.info(f"WEIGHTSLAB_CERTS_DIR_HOST={bootstrap_env_vars['WEIGHTSLAB_CERTS_DIR_HOST']}")
        logger.info(f"WEIGHTSLAB_ROOT={bootstrap_env_vars['WEIGHTSLAB_ROOT']}")
        exit_code = _run_shell_script(bootstrap_script, script_args, bootstrap_env_vars)
        if exit_code != 0:
            logger.warning(f"Bootstrap script exited with code {exit_code}, continuing anyway...")
    else:
        logger.warning(f"Bootstrap script not found: {bootstrap_script}")

    # docker compose gives an exported env var precedence over .env. Force the
    # host-native certs path here (on every path, including --no-certs) so our
    # compose call below bind-mounts a real folder — a stray /mnt/c or empty
    # value would mount an empty/invalid dir and can crash Envoy.
    os.environ["WEIGHTSLAB_CERTS_DIR"] = _to_docker_host_path(manager.certs_dir)

    _compose_cmd(
        _get_compose_file(),
        _get_envoy_config(),
        ["up", "-d", "--pull", "always"],
    )
    port = os.environ.get("VITE_PORT", "5173")
    # HTTPS iff cert files actually exist in WEIGHTSLAB_CERTS_DIR — the same
    # file-presence rule the deploy pipeline applies.
    secured = (not no_certs) and manager.has_valid_certs()
    protocol = "https" if secured else "http"
    logger.info(f"Weights Studio UI is running at: {protocol}://localhost:{port}")


def ui_secure_environment(args):
    """`weightslab se`: create a certs directory with certs + gRPC token.

    The directory is the single source of truth — WEIGHTSLAB_CERTS_DIR is exported
    for this process and the user is asked to export it globally. Everything else
    (TLS on/off, auth on/off) is derived from the files in that directory by the
    backend and the deploy pipeline, so this command does not set any other env.
    """
    logger.info("Setting up secure environment...")

    force_certs = getattr(args, "force_certs", False)
    no_auth = getattr(args, "no_auth", False)
    certs_dir = getattr(args, "certs_dir", None)

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

    logger.info("✓ Certificates generated successfully")
    logger.info("✓ gRPC auth token created")
    logger.info(f"✓ Certs and token stored in: {manager.certs_dir}")
    logger.info(f"✓ WEIGHTSLAB_CERTS_DIR exported for this process: {manager.certs_dir}")
    logger.info("")
    logger.info("Export it globally so new shells and the training backend find it:")
    logger.info(f"   (bash)    echo 'export WEIGHTSLAB_CERTS_DIR=\"{manager.certs_dir}\"' >> ~/.bashrc && source ~/.bashrc")
    logger.info(f"   (Windows) setx WEIGHTSLAB_CERTS_DIR \"{manager.certs_dir}\"")
    logger.info("Then launch the UI with: weightslab ui launch")


def _get_example_dir(name: str = "ws-classification") -> Path:
    """Path to a bundled PyTorch example directory."""
    return Path(__file__).parent / 'examples' / 'PyTorch' / name


def example_start(args):
    """`weightslab example start`: run the bundled classification (cls) example.

    Runs the example's main.py with the current Python interpreter, from its own
    directory so it resolves its sibling config.yaml. Runs in the foreground
    (the script serves until you stop it with Ctrl+C).
    """
    example_dir = _get_example_dir("ws-classification")
    main_py = example_dir / "main.py"
    if not main_py.exists():
        logger.error(f"Classification example not found: {main_py}")
        sys.exit(1)

    logger.info("Starting the WeightsLab classification (cls) example...")
    logger.info(f"   {main_py}")
    logger.info("In another terminal, launch the UI with: weightslab ui launch")
    logger.info("Then open https://localhost:5173 — stop the example with Ctrl+C.")
    try:
        result = subprocess.run([sys.executable, str(main_py)], cwd=str(example_dir))
    except KeyboardInterrupt:
        logger.info("Example stopped.")
        return
    if result.returncode != 0:
        sys.exit(result.returncode)


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser (banner + detailed command reference).

    The CLI is intentionally minimal — exactly these commands:
        weightslab --help | -h | help
        weightslab se [--force-certs]
        weightslab ui launch [--no-certs]
        weightslab start example
    """
    parser = argparse.ArgumentParser(
        prog="weightslab",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # weightslab se [--force-certs]
    se_parser = sub.add_parser("se", help="Set up the secure environment (TLS certs + gRPC auth token)")
    se_parser.add_argument('--force-certs', action='store_true', help='Regenerate certificates even if they already exist')

    # weightslab ui launch [--no-certs]
    ui_parser = sub.add_parser("ui", help="Manage the Weights Studio UI")
    ui_sub = ui_parser.add_subparsers(dest="action")
    launch_ui_parser = ui_sub.add_parser(
        "launch", help="Generate certs if missing, clean stale Docker state, then launch the UI")
    launch_ui_parser.add_argument('--no-certs', action='store_true', help='Run unsecured (HTTP, no gRPC auth); do not generate or use certs')

    # weightslab start example
    start_parser = sub.add_parser("start", help="Start a bundled WeightsLab resource")
    start_sub = start_parser.add_subparsers(dest="start_target")
    start_sub.add_parser("example", help="Start the bundled classification (cls) example")

    sub.add_parser("help", help="Show this help message")

    return parser, ui_parser, start_parser


def main():
    parser, ui_parser, start_parser = _build_parser()
    args = parser.parse_args()

    if args.command == "help" or args.command is None:
        parser.print_help()
    elif args.command == "se":
        ui_secure_environment(args)
    elif args.command == "ui":
        if getattr(args, "action", None) == "launch":
            ui_launch(args)
        else:
            ui_parser.print_help()
    elif args.command == "start":
        if getattr(args, "start_target", None) == "example":
            example_start(args)
        else:
            start_parser.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
