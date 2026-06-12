"""Weights Studio Docker management with secure TLS."""

import argparse
import os
import re
import shutil
import stat
import subprocess
import sys
import socket
import logging
from pathlib import Path

from weightslab.security import CertAuthManager

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Capture whether WEIGHTSLAB_CERTS_DIR was already in the shell env when this
# process started (set by the user) vs. injected later by our own code.
_CERTS_DIR_IN_ORIGINAL_ENV: bool = "WEIGHTSLAB_CERTS_DIR" in os.environ

# Docker resources owned by the bundled stack. Cleanup is scoped strictly to
# these names — we never run a global `docker system prune`.
_FRONTEND_IMAGE = "graybx/weightslab"
_STACK_CONTAINERS = ("weights_studio_envoy", "weights_studio_frontend")

# Cached Docker Compose base command. Resolved once per process to either the
# v2 plugin (``["docker", "compose"]``) or the legacy v1 standalone binary
# (``["docker-compose"]``) — see _detect_compose_cmd(). None until first probe.
_COMPOSE_BASE_CMD = None

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


def _persist_certs_dir(certs_dir_str: str) -> None:
    """Persist WEIGHTSLAB_CERTS_DIR so future terminals and the training backend find it.

    Windows  — runs `setx` (permanent user env) and prints the PS one-liner for
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
            logger.info("✓ WEIGHTSLAB_CERTS_DIR saved permanently via setx (new terminals will have it)")
        else:
            logger.warning(f"setx failed — set it manually: setx WEIGHTSLAB_CERTS_DIR \"{certs_dir_str}\"")
        logger.info(f"  Current terminal (PowerShell): $env:WEIGHTSLAB_CERTS_DIR = \"{certs_dir_str}\"")
    else:
        bashrc = Path.home() / ".bashrc"
        try:
            existing = bashrc.read_text(encoding="utf-8") if bashrc.exists() else ""
            if export_line not in existing:
                with open(bashrc, "a", encoding="utf-8") as f:
                    f.write(f"\n# Added by weightslab\n{export_line}\n")
                logger.info(f"✓ WEIGHTSLAB_CERTS_DIR appended to {bashrc} (new terminals will have it)")
            else:
                logger.info(f"✓ WEIGHTSLAB_CERTS_DIR already in {bashrc}")
        except OSError as e:
            logger.warning(f"Could not write to {bashrc}: {e}")
            logger.info(f"  Add manually: {export_line}")
        logger.info(f"  Current terminal: source ~/.bashrc  (or open a new terminal)")


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
                           ~/.weightslab-certs. Then set WEIGHTSLAB_CERTS_DIR
                           (the single source of truth) so the backend + new
                           shells find them.
                             --force-certs   regenerate even if certs exist

  ui launch                Purge stale weightslab/weights_studio Docker
                           resources, then build & start the UI stack.
                           UNSECURED (HTTP) by default — no certs generated.
                             --certs         generate (if missing) + use TLS
                                             certs + gRPC auth (HTTPS)

  start example            Run a bundled PyTorch example (foreground; stop with
                           Ctrl+C). Installs the example's requirements first,
                           without prompting. Defaults to classification:
                             --cls     classification example (default)
                             --seg     segmentation example
                             --det     detection example
                             --clus    clustering example
                             --gen     generation example
                             --3d_det  3D LiDAR point-cloud detection example
                             --2d_det  2D LiDAR point-cloud detection example

examples:
  weightslab se                       # one-time secure setup (then export WEIGHTSLAB_CERTS_DIR)
  weightslab se --force-certs         # regenerate the certs
  weightslab ui launch                # clean + launch (unsecured HTTP, default)
  weightslab ui launch --certs        # secured launch (HTTPS + gRPC auth)
  weightslab start example            # run the classification demo (default)
  weightslab start example --seg      # run the segmentation demo
  weightslab start example --det      # run the detection demo
  weightslab start example --3d_det   # run the 3D LiDAR detection demo
  weightslab start example --2d_det   # run the 2D LiDAR detection demo
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
        # Non-fatal: e.g. a root-owned system install the user can't chmod. They
        # would not have been able to chmod it manually either; surface as debug.
        logger.debug(f"Could not set execute bit on {path}: {exc}")


def _ensure_scripts_executable() -> None:
    """Make every bundled shell script executable (POSIX only).

    Covers build-and-deploy.sh, generate-certs-auth-token.sh and the other ``.sh``
    files under ``weightslab/ui`` so a freshly pip-installed package can run them
    without the user having to ``chmod +x`` first. No-op on Windows.
    """
    if _is_windows():
        return
    ui_dir = Path(__file__).parent / 'ui'
    try:
        scripts = list(ui_dir.rglob('*.sh'))
    except OSError as exc:
        logger.debug(f"Could not enumerate bundled scripts under {ui_dir}: {exc}")
        return
    for script in scripts:
        _make_executable(script)


def _detect_compose_cmd():
    """Return the base command for Docker Compose, preferring v2 over v1.

    Docker Compose ships in two forms:
      * v2 — the ``docker compose`` CLI plugin (bundled with Docker Desktop and
        the recommended install). Invoked as two words: ``docker compose``.
      * v1 — the legacy standalone ``docker-compose`` binary (one word, hyphen).

    We probe v2 first (``docker compose version``) then fall back to v1
    (``docker-compose version``) so ``weightslab ui launch`` works with either.
    Returns ``["docker", "compose"]``, ``["docker-compose"]``, or None if neither
    is available.
    """
    # v2: the `docker compose` plugin.
    try:
        result = subprocess.run(
            ["docker", "compose", "version"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return ["docker", "compose"]
    except FileNotFoundError:
        # docker itself is missing; _check_docker() surfaces the user-facing error.
        pass

    # v1: the legacy standalone `docker-compose` binary.
    if shutil.which("docker-compose") is not None:
        try:
            result = subprocess.run(
                ["docker-compose", "version"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if result.returncode == 0:
                return ["docker-compose"]
        except FileNotFoundError:
            pass

    return None


def _compose_base_cmd():
    """Cached Docker Compose base command (v2 ``docker compose``, else v1 ``docker-compose``)."""
    global _COMPOSE_BASE_CMD
    if _COMPOSE_BASE_CMD is None:
        _COMPOSE_BASE_CMD = _detect_compose_cmd()
    return _COMPOSE_BASE_CMD


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

    # If WEIGHTSLAB_CERTS_DIR isn't set, fall back to the default certs dir when
    # it actually holds a full cert set + gRPC token. This gives docker compose a
    # valid host-native bind-mount source (so Envoy gets its certs) even when the
    # user never exported the var — and covers compose calls that run before
    # ui_launch sets it (e.g. the stale-resource cleanup `compose down`).
    if not env.get("WEIGHTSLAB_CERTS_DIR"):
        default_mgr = CertAuthManager()
        if default_mgr.has_valid_certs() and default_mgr.token_file.exists():
            host_path = _to_docker_host_path(default_mgr.certs_dir)
            env["WEIGHTSLAB_CERTS_DIR"] = host_path
            logger.info(
                f"WEIGHTSLAB_CERTS_DIR not set — using default certs dir for docker: {host_path}"
            )

    base = _compose_base_cmd()
    if base is None:
        logger.error(
            "Docker Compose is required but was not found.\n"
            "Install Compose v2 (recommended) — the `docker compose` CLI plugin, "
            "bundled with Docker Desktop: https://docs.docker.com/compose/install/\n"
            "The legacy v1 `docker-compose` binary also works if it is on your PATH."
        )
        sys.exit(1)

    action = list(action)
    # Compose v1 (`docker-compose`) has no `up --pull <policy>` flag (it is
    # v2-only). Emulate it: `pull` first (best-effort — some services only build
    # locally), then `up` without the flag. v2 supports it inline, so leave it.
    if base == ["docker-compose"] and action and action[0] == "up" and "--pull" in action:
        i = action.index("--pull")
        del action[i:i + 2]  # drop '--pull' and its policy value (e.g. 'always')
        logger.info("Docker Compose v1 detected — pulling images before 'up'...")
        pull_result = subprocess.run(
            base + ["-f", str(compose_file), "pull"],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        if pull_result.stdout:
            for line in pull_result.stdout.splitlines():
                logger.info(line)

    cmd = base + ["-f", str(compose_file)] + action
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
            script_path = script_path.replace("\\", "/")  # Ensure path is Unix-style for bash
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
        # _to_docker_host_path gives C:/... which WSL bash treats as a relative
        # path, creating the certs dir under cwd instead of ~/.weightslab-certs.
        # Use the /mnt/c/... form so the path is absolute inside WSL.
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


_DEV_CA_SUBJECT = "weightslab-dev-ca"


def _install_ca_trust(ca_file: Path) -> None:
    """Install the dev CA into the OS trust store so browsers trust the HTTPS UI.

    Idempotent and safe to call on every launch. Platform behavior:
      * Windows — adds to the CurrentUser\\Root store via the .NET X509Store API
        (silent, no prompt).
      * macOS   — adds to the login keychain (may show a one-time auth prompt).
      * Linux   — installs into the system trust store via sudo (one-time prompt)
        and, best-effort, the user's NSS DB so Chrome/Firefox trust it too.

    A failure here is non-fatal: TLS still works, the browser just shows a
    self-signed warning until the CA is trusted manually.
    """
    if not ca_file.exists():
        logger.warning(f"CA file not found, skipping trust install: {ca_file}")
        return

    if _is_windows():
        ps = (
            "$ErrorActionPreference='Stop';"
            f"$caPath='{ca_file}';"
            "$certObj=New-Object System.Security.Cryptography.X509Certificates.X509Certificate2($caPath);"
            "$store=New-Object System.Security.Cryptography.X509Certificates.X509Store('Root','CurrentUser');"
            "$store.Open('ReadWrite');"
            "try{"
            # Already trusted (same thumbprint)? Nothing to do — avoids a fragile Remove.
            "$match=$store.Certificates|Where-Object{$_.Thumbprint -eq $certObj.Thumbprint};"
            "if(-not $match){"
            # Drop stale same-subject CAs from a previous rotation (best-effort).
            f"$stale=$store.Certificates|Where-Object{{$_.Subject -eq 'CN={_DEV_CA_SUBJECT}'}};"
            "foreach($c in $stale){try{$store.Remove($c)}catch{}};"
            "$store.Add($certObj);"
            "}"
            "}finally{$store.Close()}"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        if result.returncode == 0:
            logger.info("✓ Dev CA trusted in Windows CurrentUser\\Root store (restart browser to apply)")
        else:
            logger.warning(f"Could not auto-trust dev CA: {result.stderr.strip()}")
        return

    if sys.platform == "darwin":
        check = subprocess.run(
            ["security", "find-certificate", "-c", _DEV_CA_SUBJECT],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if check.returncode == 0:
            logger.info("✓ Dev CA already trusted (macOS keychain)")
            return
        logger.info("Installing dev CA into macOS login keychain (may prompt)...")
        subprocess.run(
            ["security", "add-trusted-cert", "-r", "trustRoot",
             "-k", os.path.expanduser("~/Library/Keychains/login.keychain-db"), str(ca_file)],
        )
        return

    # Linux: system trust store (curl/openssl) + best-effort NSS (browsers).
    system_ca = Path("/usr/local/share/ca-certificates/weightslab-dev-ca.crt")
    try:
        already = system_ca.exists() and system_ca.read_bytes() == ca_file.read_bytes()
    except OSError:
        already = False
    if not already:
        logger.info("Installing dev CA into the Linux system trust store (may prompt for sudo)...")
        subprocess.run(["sudo", "cp", str(ca_file), str(system_ca)])
        subprocess.run(["sudo", "update-ca-certificates"])
    else:
        logger.info("✓ Dev CA already in Linux system trust store")

    # Browsers use their own NSS DB; add it there too if certutil is available.
    if shutil.which("certutil"):
        nssdb = Path.home() / ".pki" / "nssdb"
        nssdb.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["certutil", "-A", "-n", _DEV_CA_SUBJECT, "-t", "C,,",
             "-i", str(ca_file), "-d", f"sql:{nssdb}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


def _ensure_certificates(manager: CertAuthManager, force_certs: bool = False) -> bool:
    """Generate certs + auth token in ``manager.certs_dir`` if missing (or forced).

    Does NOT export any TLS/auth env vars: the launch pipeline derives TLS purely
    from cert-file presence in ``WEIGHTSLAB_CERTS_DIR`` (the single source of
    truth). Returns True if certs are present afterwards, False otherwise.
    """
    if manager.has_any_credentials() and not force_certs:
        logger.info(f"✓ Using existing credentials in {manager.certs_dir}")
        manager.get_or_create_auth_token()
        # Ensure the CA is trusted even when reusing certs from a prior run that
        # was generated via bash (which does not install OS trust).
        _install_ca_trust(manager.ca_file)
        return manager.has_valid_certs()

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
    _install_ca_trust(manager.ca_file)
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
    """Purge stale Docker state, then build & start the UI.

    By default the UI launches UNSECURED (HTTP, no gRPC auth) — no certificates
    are generated. Pass ``--certs`` to generate (if missing) and use TLS certs +
    a gRPC auth token. Existing certs in WEIGHTSLAB_CERTS_DIR are always honored
    (file presence is the single source of truth) and are never deleted here.

    Flags (all optional, read defensively so legacy callers still work):
      --certs        generate (if missing) and use TLS certs + gRPC auth (HTTPS)
      --force-certs  with --certs, regenerate certificates even if they exist
      --no-clean     skip the stale Docker resource cleanup step
      --dev          use the dev compose overlay
    """
    _check_docker()
    # pip installs the bundled .sh scripts without the execute bit; make them
    # runnable so the user never has to `chmod +x` before `weightslab ui launch`.
    _ensure_scripts_executable()

    use_certs = getattr(args, "certs", False)
    no_auth = getattr(args, "no_auth", False)
    force_certs = getattr(args, "force_certs", False)
    no_clean = getattr(args, "no_clean", False)
    certs_dir_arg = getattr(args, "certs_dir", None)

    # A custom certs dir (positional arg) takes precedence over the env var /
    # default. Otherwise fall back to $WEIGHTSLAB_CERTS_DIR or ~/.weightslab-certs.
    if certs_dir_arg:
        # Resolve to an absolute path so Windows Python, WSL bash and docker all
        # agree on the location (a relative path resolves differently per shell).
        certs_dir_arg = str(Path(certs_dir_arg).resolve())
        manager = CertAuthManager(certs_dir=certs_dir_arg, enable_auth=not no_auth)
        logger.info(f"Using custom certs directory: {manager.certs_dir}")
    else:
        manager = CertAuthManager.from_env_or_default(enable_auth=not no_auth)

    if use_certs:
        _ensure_certificates(manager, force_certs=force_certs)
    else:
        # Default: do NOT generate certs. Existing certs in WEIGHTSLAB_CERTS_DIR
        # are still honored (file-presence is the single source of truth); this
        # never deletes anything. Ensure the dir exists for the compose bind-mount.
        logger.info("Launching WITHOUT cert generation (default; HTTP). "
                    "Pass --certs for secured HTTPS + gRPC auth.")
        try:
            manager.certs_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            os.makedirs(str(manager.certs_dir), exist_ok=True)
            logger.warning('Fail to create weightslab-certs directory!')

    # Single source of truth: TLS is on iff cert files exist in the dir.
    secured = manager.has_valid_certs()

    # Set the correct certs path in the process env NOW so that _compose_cmd
    # passes it to docker compose — this overrides any stale/malformed value
    # in a leftover .env file and prevents the "too many colons" bind-mount error.
    os.environ["WEIGHTSLAB_CERTS_DIR"] = _to_docker_host_path(manager.certs_dir)

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
        # source. When there are no certs we add --unsecure, which forces
        # Envoy/nginx to plaintext (mounted files, if any, are ignored) without
        # leaving the mount source empty.
        bootstrap_env_vars = {
            'WEIGHTSLAB_CERTS_DIR': certs_dir_str,
            'WEIGHTSLAB_CERTS_DIR_HOST': certs_dir_host,
            'WEIGHTSLAB_ROOT': weightslab_root
        }
        # On Windows the bootstrap runs in WSL/Git-Bash where `docker` usually
        # isn't on PATH; the compose up below (run from Windows Python) handles
        # the deploy. Tell the script to only write .env and skip docker ops so
        # it doesn't emit a spurious "build failed" error.
        if _is_windows():
            bootstrap_env_vars['WEIGHTSLAB_SKIP_DOCKER_OPS'] = '1'
        script_args = []
        if not secured:
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
    # host-native certs path here (on every path, secured or not) so our
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
    secured = manager.has_valid_certs()
    protocol = "https" if secured else "http"
    logger.info(f"Weights Studio UI is running at: {protocol}://localhost:{port}")
    if secured:
        certs_dir_str = str(manager.certs_dir)
        # Persist first (it logs its own lines), so the export/setx reminder
        # below stays the FINAL output and can't be missed.
        # Persist when a custom dir was given (it won't match the default, so the
        # backend must be told where to look) or when the var isn't already set.
        if certs_dir_arg or not _CERTS_DIR_IN_ORIGINAL_ENV:
            _persist_certs_dir(certs_dir_str)
        # The backend and any new shell must point at the same certs dir, or
        # they'll mismatch the UI's TLS/auth. Keep this the last thing printed.
        logger.warning("")
        logger.warning("⚠ ACTION REQUIRED — TLS is ON. Set WEIGHTSLAB_CERTS_DIR so the "
                       "training backend and new terminals use the same certificates:")
        logger.warning(f"   (bash)    export WEIGHTSLAB_CERTS_DIR=\"{certs_dir_str}\"")
        logger.warning(f"   (Windows) setx WEIGHTSLAB_CERTS_DIR \"{certs_dir_str}\"")
    else:
        logger.info("UI is running UNSECURED (HTTP, no gRPC auth). "
                    "Re-run with `weightslab ui launch --certs` for TLS.")


def ui_secure_environment(args):
    """`weightslab se`: create a certs directory with certs + gRPC token.

    The directory is the single source of truth — WEIGHTSLAB_CERTS_DIR is exported
    for this process and the user is asked to export it globally. Everything else
    (TLS on/off, auth on/off) is derived from the files in that directory by the
    backend and the deploy pipeline, so this command does not set any other env.
    """
    logger.info("Setting up secure environment...")
    # Bundled .sh scripts ship without the execute bit (pip strips it); make them
    # runnable so the cert-generation script below doesn't fail on "Permission denied".
    _ensure_scripts_executable()

    force_certs = getattr(args, "force_certs", False)
    no_auth = getattr(args, "no_auth", False)
    certs_dir = getattr(args, "certs_dir", None)
    if certs_dir:
        # Absolute path so Windows Python, WSL bash and docker agree on location.
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

    logger.info("✓ Certificates generated successfully")
    logger.info("✓ gRPC auth token created")
    logger.info(f"✓ Certs and token stored in: {manager.certs_dir}")
    logger.info(f"✓ WEIGHTSLAB_CERTS_DIR exported for this process: {manager.certs_dir}")
    logger.info("Then launch the secured UI with: weightslab ui launch --certs")
    # Keep this the FINAL output so the user can't miss the action they must take.
    logger.warning("")
    logger.warning("⚠ ACTION REQUIRED — set WEIGHTSLAB_CERTS_DIR globally so new shells "
                   "and the training backend find these certs (single source of truth):")
    logger.warning(f"   (bash)    echo 'export WEIGHTSLAB_CERTS_DIR=\"{manager.certs_dir}\"' >> ~/.bashrc && source ~/.bashrc")
    logger.warning(f"   (Windows) setx WEIGHTSLAB_CERTS_DIR \"{manager.certs_dir}\"")


# Bundled PyTorch examples, keyed by the CLI flag (e.g. --cls -> ws-classification).
# Each value is (directory name under examples/PyTorch, human-readable label).
# kind -> (dir_name, label, category) where category is the examples/ subfolder.
_EXAMPLES = {
    "cls": ("ws-classification", "classification", "PyTorch"),
    "seg": ("ws-segmentation", "segmentation", "PyTorch"),
    "det": ("ws-detection", "detection", "PyTorch"),
    "clus": ("ws-clustering", "clustering", "PyTorch"),
    "gen": ("ws-generation", "generation", "PyTorch"),
    "3d_det": ("ws-3d-lidar-detection", "3D LiDAR detection", "Usecases"),
    "2d_det": ("ws-2d-lidar-detection", "2D LiDAR detection", "Usecases"),
}
_DEFAULT_EXAMPLE = "cls"


def _get_example_dir(name: str = "ws-classification", category: str = "PyTorch") -> Path:
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
        return  # only the first matching requirements file is used


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
    logger.info(f"   {main_py}")
    logger.info("In another terminal, launch the UI with: weightslab ui launch")
    logger.info(f"Then open http://localhost:5173 — stop the example with Ctrl+C.")
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
        weightslab ui launch [--certs]
        weightslab start example [--cls|--seg|--det|--clus|--gen|--3d_det|--2d_det]
    """
    parser = argparse.ArgumentParser(
        prog="weightslab",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # metavar lists only the documented commands; the `example` alias is accepted
    # but intentionally omitted here (and help=SUPPRESS'd below) so it stays hidden.
    sub = parser.add_subparsers(dest="command", metavar="{se,ui,start,help}")

    # weightslab se [--force-certs] [certs_dir]
    se_parser = sub.add_parser("se", help="Set up the secure environment (TLS certs + gRPC auth token)")
    se_parser.add_argument('--force-certs', action='store_true', help='Regenerate certificates even if they already exist')
    se_parser.add_argument('certs_dir', nargs='?', default=None,
                           help='Custom directory for certs/token (default: $WEIGHTSLAB_CERTS_DIR or ~/.weightslab-certs)')

    # weightslab ui launch [--certs] [certs_dir]
    ui_parser = sub.add_parser("ui", help="Manage the Weights Studio UI")
    ui_sub = ui_parser.add_subparsers(dest="action")
    launch_ui_parser = ui_sub.add_parser(
        "launch", help="Clean stale Docker state, then launch the UI (unsecured by default; --certs for TLS)")
    launch_ui_parser.add_argument('--certs', action='store_true', help='Generate (if missing) and use TLS certs + gRPC auth token (secured HTTPS). Default: unsecured HTTP.')
    launch_ui_parser.add_argument('certs_dir', nargs='?', default=None,
                                  help='Custom directory for certs/token (default: $WEIGHTSLAB_CERTS_DIR or ~/.weightslab-certs)')

    # weightslab start example [--cls|--seg|--clus|--gen]
    start_parser = sub.add_parser("start", help="Start a bundled WeightsLab resource")
    start_sub = start_parser.add_subparsers(dest="start_target")
    example_parser = start_sub.add_parser(
        "example", help="Start a bundled PyTorch example (default: classification)")
    _add_example_kind_flags(example_parser)

    # Tolerate the swapped order: `weightslab example start [flags]` (and bare
    # `weightslab example`) behave exactly like `weightslab start example`. Hidden
    # from --help on purpose (argparse.SUPPRESS) — it's a forgiving fallback, not a
    # documented command.
    example_alias = sub.add_parser("example", help=argparse.SUPPRESS)
    example_alias_sub = example_alias.add_subparsers(dest="example_action")
    example_alias_start = example_alias_sub.add_parser(
        "start", help="Start a bundled PyTorch example (default: classification)")
    _add_example_kind_flags(example_alias_start)

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
    elif args.command == "example":
        # Alias for `start example` — tolerate the swapped subcommand order
        # (`weightslab example start [flags]`) and the bare `weightslab example`.
        example_start(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
