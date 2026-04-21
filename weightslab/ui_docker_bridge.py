"""Weights Studio Docker management with secure TLS."""

import argparse
import os
import shutil
import subprocess
import sys
import socket
import time
import logging
from pathlib import Path
from importlib.resources import files

from weightslab.security import CertAuthManager

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def _get_compose_file():
    """Return the path to the bundled docker-compose.yml."""
    return files("weightslab.ui.docker") / "docker-compose.yml"


def _get_envoy_config():
    """Return the path to the bundled envoy.yaml."""
    return files("weightslab.ui.envoy") / "envoy.yaml"


def _get_bootstrap_script() -> Path:
    """Get the bootstrap-secure.ps1 script path."""
    return Path(__file__).parent / 'ui' / 'docker' / 'utils' / 'bootstrap-secure.ps1'


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
    subprocess.run(cmd, env=env, check=True)


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


def _run_powershell_script(script_path: str, args: list = None) -> int:
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
        result = subprocess.run(cmd, env=os.environ.copy())
        return result.returncode
    except Exception as e:
        logger.error(f"Failed to run script: {e}")
        return 1


def ui_launch(args):
    """Pull images and start UI containers."""
    _check_docker()
    _compose_cmd(
        _get_compose_file(),
        _get_envoy_config(),
        ["up", "-d", "--pull", "always"],
    )
    port = os.environ.get("VITE_PORT", "5173")
    logger.info(f"Weights Studio UI is running at: https://localhost:{port}")


def ui_launch_secure(args):
    """Launch with secured TLS and gRPC auth."""
    logger.info("Launching Weights Studio with secured TLS...")

    # Initialize certificates and auth
    manager = CertAuthManager.from_env_or_default(enable_auth=not args.no_auth)

    success, msg = manager.initialize(force_certs=args.force_certs)
    if not success:
        logger.error(f"Failed to initialize certificates: {msg}")
        sys.exit(1)

    logger.info(msg)

    # Prepare bootstrap script arguments
    script_args = []
    if args.force_certs:
        script_args.append('-force_create_certs')
    if args.no_auth:
        script_args.append('-no_auth_token')
    if args.dev:
        script_args.append('-dev')

    # Run bootstrap script
    bootstrap_script = str(_get_bootstrap_script())
    if not Path(bootstrap_script).exists():
        logger.error(f"Bootstrap script not found: {bootstrap_script}")
        sys.exit(1)

    logger.info("Running bootstrap script...")
    exit_code = _run_powershell_script(bootstrap_script, script_args)
    if exit_code != 0:
        logger.error(f"Bootstrap script failed with exit code {exit_code}")
        sys.exit(exit_code)

    logger.info("✓ Docker stack launched successfully")

    # Test communication if requested
    if args.test:
        time.sleep(2)
        logger.info("Testing backend-UI communication...")
        if _test_backend_connection():
            logger.info("✓ Backend server is reachable")
        else:
            logger.warning(
                "Backend server not reachable at localhost:50051. "
                "Ensure backend is running."
            )


def ui_stop(args):
    """Stop UI containers (keeps images)."""
    _check_docker()
    _compose_cmd(
        _get_compose_file(),
        _get_envoy_config(),
        ["stop"],
    )
    logger.info("Weights Studio UI stopped.")


def ui_drop(args):
    """Stop and remove containers, networks, and images."""
    _check_docker()
    _compose_cmd(
        _get_compose_file(),
        _get_envoy_config(),
        ["down", "--rmi", "all"],
    )
    logger.info("Weights Studio UI containers and images removed.")


def docker_launch_secure(args):
    """Launch Docker with PowerShell bootstrap (Windows only)."""
    ui_launch_secure(args)


def docker_stop(args):
    """Stop Docker containers."""
    ui_stop(args)


def docker_info(args):
    """Show Docker configuration."""
    manager = CertAuthManager.from_env_or_default()

    logger.info("=== Weights Studio Docker Configuration ===")
    logger.info(f"Certificates directory: {manager.certs_dir}")
    logger.info(f"Certificates exist: {manager.has_valid_certs()}")
    logger.info(f"Auth enabled: {manager.enable_auth}")

    logger.info("\n=== Environment Variables ===")
    tls_env = manager.setup_tls_environment()
    auth_env = manager.setup_auth_environment()

    for key, value in {**tls_env, **auth_env}.items():
        logger.info(f"{key}={value}")


def main():
    parser = argparse.ArgumentParser(
        prog="weightslab docker",
        description="WeightsLab Docker Management",
    )
    sub = parser.add_subparsers(dest="command")

    # Legacy UI commands
    ui_parser = sub.add_parser("ui", help="Manage the Weights Studio UI (legacy)")
    ui_sub = ui_parser.add_subparsers(dest="action")
    ui_sub.add_parser("launch", help="Pull images and start the UI")
    ui_sub.add_parser("stop", help="Stop the UI containers")
    ui_sub.add_parser("drop", help="Stop containers and remove images")

    # Secure Docker commands
    docker_parser = sub.add_parser("docker", help="Manage Docker with secure TLS")
    docker_sub = docker_parser.add_subparsers(dest="action")

    launch_parser = docker_sub.add_parser("launch", help="Launch with secured TLS")
    launch_parser.add_argument('--dev', action='store_true', help='Use dev configuration')
    launch_parser.add_argument('--force-certs', action='store_true', help='Regenerate certs')
    launch_parser.add_argument('--no-auth', action='store_true', help='Disable auth token')
    launch_parser.add_argument('--test', action='store_true', help='Test backend connection')

    docker_sub.add_parser("stop", help="Stop Docker containers")
    docker_sub.add_parser("info", help="Show configuration")

    sub.add_parser("help", help="Show this help message")

    args = parser.parse_args()

    ui_actions = {
        "launch": ui_launch,
        "stop": ui_stop,
        "drop": ui_drop,
    }

    docker_actions = {
        "launch": docker_launch_secure,
        "stop": docker_stop,
        "info": docker_info,
    }

    if args.command == "help" or args.command is None:
        parser.print_help()
    elif args.command == "ui" and args.action in ui_actions:
        ui_actions[args.action](args)
    elif args.command == "ui":
        ui_parser.print_help()
    elif args.command == "docker" and args.action in docker_actions:
        docker_actions[args.action](args)
    elif args.command == "docker":
        docker_parser.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
