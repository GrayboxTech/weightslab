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
    p = Path(win_path).as_posix()
    # Convert C:/Users/... to /c/Users/... for Git Bash
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

        if env_assignments:
            # Pass env vars directly in bash command using -c flag
            # This works on Windows (Git Bash), Linux, and macOS
            bash_script_cmd = f"{env_assignments} '{script_path}'"
            if args:
                bash_script_cmd += " " + " ".join(f"'{arg}'" for arg in args)
            bash_cmd = ['bash', '-c', bash_script_cmd]
        else:
            bash_cmd = ['bash', str(script_path)]
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


def _generate_certs_with_fallback(force_certs: bool = False) -> int:
    """Try shell script first, fall back to PowerShell on Windows if it fails."""
    cert_script = str(_get_cert_script())
    if not Path(cert_script).exists():
        logger.warning(f"Shell script not found: {cert_script}")
    else:
        script_args = []
        if force_certs:
            script_args.append('--force-create-certs')

        logger.info("Attempting certificate generation with shell script...")
        exit_code = _run_shell_script(cert_script, script_args)
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

        exit_code = _run_powershell_script(cert_script_ps1, script_args)
        return exit_code
    else:
        logger.error("Neither shell nor PowerShell script could generate certificates")
        return 1


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


def _convert_to_git_bash_path(win_path: str) -> str:
    """Convert Windows path to Git Bash compatible format."""
    p = Path(win_path).as_posix()
    # Convert C:/Users/... to /c/Users/... for Git Bash
    if len(p) > 1 and p[1] == ':':
        drive = p[0].lower()
        rest = p[2:]
        return f"/{drive}{rest}"
    return p


def _run_shell_script(script_path: str, args: list = None) -> int:
    """Run a shell script using bash -c with inline execution."""
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

        # Convert back to string for piping
        script_content = fixed_bytes.decode('utf-8')

        # Use bash -c to run the script with arguments
        # Build the command that sets up $1, $2, etc. before executing script
        if args:
            args_setup = ' '.join(f'"{arg}"' for arg in args)
            bash_cmd = f'set -- {args_setup}; eval "{script_content}"'
        else:
            bash_cmd = script_content

        result = subprocess.run(['bash', '-c', bash_cmd], env=os.environ.copy())
        return result.returncode
    except FileNotFoundError:
        logger.error(f"Script file not found: {script_path}")
        return 1
    except Exception as e:
        logger.error(f"Failed to run script: {e}")
        return 1


def _generate_certs_with_fallback(force_certs: bool = False) -> int:
    """Try shell script first, fall back to PowerShell on Windows if it fails."""
    cert_script = str(_get_cert_script())
    if not Path(cert_script).exists():
        logger.warning(f"Shell script not found: {cert_script}")
    else:
        script_args = []
        if force_certs:
            script_args.append('--force-create-certs')

        logger.info("Attempting certificate generation with shell script...")
        exit_code = _run_shell_script(cert_script, script_args)
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

        exit_code = _run_powershell_script(cert_script_ps1, script_args)
        return exit_code
    else:
        logger.error("Neither shell nor PowerShell script could generate certificates")
        return 1


def ui_launch(args):
    """Pull images and start UI containers."""
    _check_docker()

    # Run bootstrap script to setup environment
    manager = CertAuthManager.from_env_or_default(enable_auth=True)
    bootstrap_script = str(_get_bootstrap_script())
    if Path(bootstrap_script).exists():
        logger.info("Running bootstrap script...")
        logger.info(f"Bootstrap script path: {bootstrap_script}")
        # Convert Windows path to Unix-style for bash
        certs_dir_str = str(manager.certs_dir)
        if _is_windows() and '\\' in certs_dir_str:
            certs_dir_str = _convert_to_git_bash_path(certs_dir_str)
            logger.info(f"Converted path to Unix-style: {certs_dir_str}")
        bootstrap_env_vars = {
            'WEIGHTSLAB_CERTS_DIR': certs_dir_str
        }
        logger.info(f"WEIGHTSLAB_CERTS_DIR={bootstrap_env_vars['WEIGHTSLAB_CERTS_DIR']}")
        exit_code = _run_shell_script(bootstrap_script, [], bootstrap_env_vars)
        if exit_code != 0:
            logger.warning(f"Bootstrap script exited with code {exit_code}, continuing anyway...")
    else:
        logger.warning(f"Bootstrap script not found: {bootstrap_script}")

    _compose_cmd(
        _get_compose_file(),
        _get_envoy_config(),
        ["up", "-d", "--pull", "always"],
    )
    port = os.environ.get("VITE_PORT", "5173")
    logger.info(f"Weights Studio UI is running at: https://localhost:{port}")


def ui_secure_environment(args):
    """Generate TLS certificates and gRPC auth token (one-time setup)."""
    logger.info("Setting up secure environment...")

    exit_code = _generate_certs_with_fallback(force_certs=args.force_certs)
    if exit_code != 0:
        logger.error("Certificate generation failed")
        sys.exit(1)

    manager = CertAuthManager(certs_dir=args.certs_dir, enable_auth=not args.no_auth)

    # Ensure certs directory exists
    manager.certs_dir.mkdir(parents=True, exist_ok=True)

    # Get or create auth token
    manager.get_or_create_auth_token()

    # Set environment variables
    env_vars = manager.setup_tls_environment()
    env_vars.update(manager.setup_auth_environment())

    for key, value in env_vars.items():
        os.environ[key] = value

    logger.info("✓ Certificates generated successfully")
    logger.info("✓ gRPC auth token created")
    logger.info(f"✓ Certs and token stored in: {manager.certs_dir}")


def ui_docker_secure_environment(args):
    """Setup secure environment AND launch Docker (both in one command)."""
    logger.info("Setting up secure environment...")

    exit_code = _generate_certs_with_fallback(force_certs=args.force_certs)
    if exit_code == 0:
        manager = CertAuthManager.from_env_or_default(enable_auth=not args.no_auth)

        # Ensure certs directory exists
        manager.certs_dir.mkdir(parents=True, exist_ok=True)

        # Set environment variables
        env_vars = manager.setup_tls_environment()
        env_vars.update(manager.setup_auth_environment())

        for key, value in env_vars.items():
            os.environ[key] = value

        logger.info("✓ Certificates and auth token generated successfully")
        logger.info(f"✓ Certs and token stored in: {manager.certs_dir}")
    else:
        logger.warning("Certificate generation failed")
        logger.warning("Launching Docker in unsecured mode...")

    # Launch Docker regardless of secure setup success (non-blocking fallback)
    logger.info("\nLaunching Docker stack...")
    ui_launch_secure(args)


def ui_launch_secure(args):
    """Launch with secured TLS and gRPC auth (if certs exist)."""
    logger.info("Launching Weights Studio...")

    # Force unsecured mode if --unsecure flag is set
    if hasattr(args, 'unsecure') and args.unsecure:
        logger.info("⚠ Forcing unsecured mode (HTTP, no auth)")
        ui_launch(args)
        return

    # Check for existing certificates (no generation)
    manager = CertAuthManager.from_env_or_default(enable_auth=not args.no_auth)
    success, msg = manager.check_and_apply()

    if not success:
        logger.warning(f"Secure certs not found — falling back to unsecured mode")
        logger.warning("To set up security, run: weightslab ui se")
        ui_launch(args)
        return

    logger.info("✓ Secure environment configured")

    # Prepare bootstrap script arguments
    script_args = []
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
    logger.info(f"Bootstrap script path: {bootstrap_script}")
    logger.info(f"Bootstrap script args: {script_args}")
    # Convert Windows path to Unix-style for bash
    certs_dir_str = str(manager.certs_dir)
    if _is_windows() and '\\' in certs_dir_str:
        certs_dir_str = _convert_to_git_bash_path(certs_dir_str)
        logger.info(f"Converted path to Unix-style: {certs_dir_str}")
    bootstrap_env_vars = {
        'WEIGHTSLAB_CERTS_DIR': certs_dir_str
    }
    logger.info(f"WEIGHTSLAB_CERTS_DIR={bootstrap_env_vars['WEIGHTSLAB_CERTS_DIR']}")
    exit_code = _run_shell_script(bootstrap_script, script_args, bootstrap_env_vars)
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
        prog="weightslab",
        description="WeightsLab Docker Management",
    )
    sub = parser.add_subparsers(dest="command")

    # Top-level secure environment command (shortcut)
    se_parser = sub.add_parser("se", aliases=["secured_environment"],
                               help="Setup secure environment (TLS certs and gRPC auth token)")
    se_parser.add_argument('certs_dir', nargs='?', default=None, help='Directory to store certificates (default: ~/.weightslab-certs)')
    se_parser.add_argument('--no-auth', action='store_true', help='Skip gRPC auth token generation')
    se_parser.add_argument('--force-certs', action='store_true', help='Regenerate certs even if they exist')

    # UI commands
    ui_parser = sub.add_parser("ui", help="Manage Weights Studio UI and Docker")
    ui_sub = ui_parser.add_subparsers(dest="action")

    # Legacy UI-only commands
    ui_sub.add_parser("launch", help="Pull images and start the UI (legacy)")
    ui_sub.add_parser("stop", help="Stop the UI containers (legacy)")
    ui_sub.add_parser("drop", help="Stop containers and remove images (legacy)")

    # Docker commands under UI
    docker_parser = ui_sub.add_parser("docker", help="Manage Docker with UI")
    docker_sub = docker_parser.add_subparsers(dest="docker_action")

    # Docker secure environment + launch
    docker_se_parser = docker_sub.add_parser("se", aliases=["secured_environment"],
                                             help="Setup secure env + launch Docker (one command)")
    docker_se_parser.add_argument('--no-auth', action='store_true', help='Skip gRPC auth token generation')
    docker_se_parser.add_argument('--force-certs', action='store_true', help='Regenerate certs even if they exist')
    docker_se_parser.add_argument('--dev', action='store_true', help='Use dev configuration')
    docker_se_parser.add_argument('--test', action='store_true', help='Test backend connection')

    # Docker launch only
    launch_parser = docker_sub.add_parser("launch", help="Launch Docker (use existing certs or unsecured)")
    launch_parser.add_argument('--dev', action='store_true', help='Use dev configuration')
    launch_parser.add_argument('--no-auth', action='store_true', help='Disable auth token')
    launch_parser.add_argument('--unsecure', action='store_true', help='Force HTTP mode without certs/auth')
    launch_parser.add_argument('--test', action='store_true', help='Test backend connection')

    # Docker info
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
        "se": ui_docker_secure_environment,
        "secured_environment": ui_docker_secure_environment,
    }

    if args.command == "help" or args.command is None:
        parser.print_help()
    elif args.command in ("se", "secured_environment"):
        ui_secure_environment(args)
    elif args.command == "ui":
        if args.action == "docker":
            if hasattr(args, 'docker_action') and args.docker_action in docker_actions:
                docker_actions[args.docker_action](args)
            else:
                docker_parser.print_help()
        elif args.action in ui_actions:
            ui_actions[args.action](args)
        else:
            ui_parser.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
