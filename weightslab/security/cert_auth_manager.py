"""Manage secure certificates and gRPC auth tokens for weightslab."""

import os
import re
import logging
import subprocess
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def _normalize_native_path(path: str) -> str:
    """Normalize a WSL/Git-Bash style path to a native path on Windows.

    On Windows, a value like ``/mnt/c/Users/...`` or ``/c/Users/...`` (from a
    shell that set WEIGHTSLAB_CERTS_DIR) would otherwise be taken literally by
    pathlib and resolve to ``C:\\mnt\\c\\Users\\...`` — splitting certs and the
    auth token across two directories. Convert those forms back to ``C:\\...``.
    No-op on Linux/macOS and for already-native paths.
    """
    if os.name != 'nt':
        return path
    p = str(path).replace('\\', '/')
    m = re.match(r'^/mnt/([a-zA-Z])/(.*)$', p) or re.match(r'^/([a-zA-Z])/(.*)$', p)
    if m:
        return f"{m.group(1).upper()}:/{m.group(2)}"
    return path


def _get_user_profile() -> str:
    """Get the user's home directory.

    On Windows, prefer the native profile (USERPROFILE / expanduser) and ignore
    HOME: shells like Git Bash and WSL set HOME to a Unix-style path
    (e.g. /mnt/c/Users/... or /c/Users/...) that is not a valid Windows
    filesystem path and corrupts both cert storage and Docker bind mounts.
    """
    if os.name == 'nt':
        return os.environ.get('USERPROFILE') or os.path.expanduser('~')
    return os.environ.get('HOME') or os.path.expanduser('~')


def _generate_hex_token(byte_count: int = 32) -> str:
    """Generate a strong hex token using secure random."""
    import secrets
    return secrets.token_hex(byte_count)


class CertAuthManager:
    """Manages TLS certificates and gRPC auth tokens for secure communication."""

    def __init__(self, certs_dir: Optional[str] = None, enable_auth: bool = True):
        """
        Initialize the certificate and auth manager.

        Args:
            certs_dir: Directory to store certificates. Defaults to ~/.weightslab-certs
            enable_auth: Whether to enable gRPC auth tokens
        """
        self.enable_auth = enable_auth
        if certs_dir is None:
            self.certs_dir = Path(_get_user_profile()) / '.weightslab-certs'
        else:
            # Normalize WSL/Git-Bash paths (/mnt/c/..., /c/...) to native form on
            # Windows so certs and the auth token never split across directories.
            self.certs_dir = Path(_normalize_native_path(str(certs_dir)))

        self.cert_file = self.certs_dir / 'backend-server.crt'
        self.key_file = self.certs_dir / 'backend-server.key'
        self.ca_file = self.certs_dir / 'ca.crt'
        self.token_file = self.certs_dir / '.grpc_auth_token'

    def has_valid_certs(self) -> bool:
        """Check if the full certificate set exists (all 3 files)."""
        return (
            self.cert_file.exists() and
            self.key_file.exists() and
            self.ca_file.exists()
        )

    def has_any_credentials(self) -> bool:
        """Return True if a complete cert set exists (token alone does not count)."""
        return self.has_valid_certs()

    def clear_credentials(self) -> int:
        """Remove all generated certs, keys and the auth token from the certs dir.

        Used for unsecured launches so the training backend (which derives TLS
        from cert-file presence) does not pick up stale certs. Returns the number
        of files removed. The directory itself is kept (empty) for bind mounts.
        """
        names = [
            'backend-server.crt', 'backend-server.key',
            'envoy-server.crt', 'envoy-server.key',
            'envoy-client.crt', 'envoy-client.key',
            'ca.crt', 'ca.srl', '.grpc_auth_token',
        ]
        removed = 0
        for name in names:
            f = self.certs_dir / name
            try:
                if f.exists():
                    f.unlink()
                    removed += 1
            except OSError as e:
                logger.warning(f"Could not remove {f}: {e}")
        if removed:
            logger.info(f"Removed {removed} credential file(s) from {self.certs_dir}")
        return removed

    def generate_certs(self, force: bool = False) -> Tuple[bool, str]:
        """
        Generate TLS certificates using the bootstrap script.

        Args:
            force: Force regeneration even if certs exist

        Returns:
            Tuple of (success, message)
        """
        if self.has_valid_certs() and not force:
            logger.info(f"Certificates already exist in {self.certs_dir}")
            return True, f"Certificates already exist in {self.certs_dir}"

        # Try to use PowerShell to generate certs
        try:
            script_dir = Path(__file__).parent.parent / 'ui' / 'docker' / 'utils'
            generate_script = script_dir / 'generate-certs.ps1'

            if not generate_script.exists():
                return False, f"Certificate generation script not found at {generate_script}"

            cmd = ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(generate_script)]
            if force:
                cmd.append('-force_create_certs')

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                cwd=str(script_dir)
            )

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout
                logger.error(f"Certificate generation failed: {error_msg}")
                return False, f"Certificate generation failed: {error_msg}"

            logger.info(f"Certificates generated successfully in {self.certs_dir}")
            return True, "Certificates generated successfully"

        except Exception as e:
            logger.error(f"Certificate generation exception: {e}")
            return False, str(e)

    def get_or_create_auth_token(self) -> str:
        """
        Get existing auth token or create a new one.

        Returns:
            The auth token as a hex string
        """
        if not self.enable_auth:
            return ""

        # The token file in this certs_dir is the single source of truth — check
        # it FIRST. (Do not short-circuit on os.environ['GRPC_AUTH_TOKEN']: it may
        # hold a different dir's token applied at import, which would prevent a
        # token from ever being written into a custom certs_dir.)
        if self.token_file.exists():
            try:
                with open(self.token_file, 'r') as f:
                    token = f.read().strip()
                    if token:
                        logger.debug("Using existing gRPC auth token from file")
                        return token
            except Exception as e:
                logger.warning(f"Could not read existing token: {e}")

        # No token file in this dir yet. Reuse an env-provided token if present
        # (so a value already exported stays consistent) — otherwise mint one —
        # and materialize it as a file so the dir is self-contained.
        token = os.environ.get('GRPC_AUTH_TOKEN', '').strip() or _generate_hex_token()

        # Save token
        try:
            self.certs_dir.mkdir(parents=True, exist_ok=True)
            with open(self.token_file, 'w') as f:
                f.write(token)
            # Restrict permissions on token file (Unix-like)
            try:
                os.chmod(self.token_file, 0o600)
            except Exception:
                pass  # Windows doesn't support chmod
            logger.info(f"Wrote gRPC auth token to {self.token_file}")
        except Exception as e:
            logger.error(f"Could not save token: {e}")

        return token

    def setup_tls_environment(self) -> dict:
        """
        Set up TLS environment variables.

        Returns:
            Dictionary of environment variables to set
        """
        env_vars = {
            'GRPC_TLS_ENABLED': '1',
            'GRPC_TLS_REQUIRE_CLIENT_AUTH': '1',
            'GRPC_TLS_CERT_FILE': str(self.cert_file),
            'GRPC_TLS_KEY_FILE': str(self.key_file),
            'GRPC_TLS_CA_FILE': str(self.ca_file),
            'WEIGHTSLAB_CERTS_DIR': str(self.certs_dir),
            'ENVOY_DOWNSTREAM_TLS': 'on',
            'ENVOY_UPSTREAM_TLS': 'on',
            'WS_SERVER_PROTOCOL': 'https',
            'VITE_SERVER_PROTOCOL': 'https',
        }

        return env_vars

    def setup_auth_environment(self) -> dict:
        """
        Set up gRPC auth token environment variables.

        Returns:
            Dictionary of environment variables to set
        """
        env_vars = {}

        if self.enable_auth:
            token = self.get_or_create_auth_token()
            env_vars['WL_ENABLE_GRPC_AUTH_TOKEN'] = '1'
            env_vars['VITE_WL_ENABLE_GRPC_AUTH_TOKEN'] = '1'
            env_vars['GRPC_AUTH_TOKEN'] = token
            env_vars['VITE_GRPC_AUTH_TOKEN'] = token
        else:
            env_vars['WL_ENABLE_GRPC_AUTH_TOKEN'] = '0'
            env_vars['VITE_WL_ENABLE_GRPC_AUTH_TOKEN'] = '0'

        return env_vars

    def check_and_apply(self) -> Tuple[bool, str]:
        """
        Check for existing certs/token and apply environment variables.
        Does NOT generate certs — returns failure if not found.

        Returns:
            Tuple of (success, message)
        """
        if not self.has_valid_certs():
            return False, f"No certs found in {self.certs_dir}. Run: weightslab ui se"

        env_vars = self.setup_tls_environment()
        env_vars.update(self.setup_auth_environment())

        for key, value in env_vars.items():
            os.environ[key] = value

        logger.info("Secure environment applied from existing certs")
        return True, "Secure environment applied from existing certs"

    def setup_secure_environment(self, force_certs: bool = False) -> Tuple[bool, str]:
        """
        Generate TLS certificates and auth token.
        Called explicitly by 'weightslab ui se' command.

        Args:
            force_certs: Force regenerate certificates

        Returns:
            Tuple of (success, message)
        """
        # Ensure certs directory exists
        self.certs_dir.mkdir(parents=True, exist_ok=True)

        # Generate certs
        success, msg = self.generate_certs(force=force_certs)
        if not success:
            return False, msg

        # Set environment variables
        env_vars = self.setup_tls_environment()
        env_vars.update(self.setup_auth_environment())

        for key, value in env_vars.items():
            os.environ[key] = value

        logger.info("Secure environment created successfully")
        return True, "Secure environment created successfully"

    def initialize(self, force_certs: bool = False) -> Tuple[bool, str]:
        """
        Backward compatibility: calls check_and_apply() for check-only behavior.
        Use setup_secure_environment() to explicitly generate certs.

        Args:
            force_certs: Ignored (kept for backward compat)

        Returns:
            Tuple of (success, message)
        """
        return self.check_and_apply()

    @staticmethod
    def from_env_or_default(enable_auth: bool = True) -> 'CertAuthManager':
        """Create manager using environment variables or defaults."""
        certs_dir = os.environ.get('WEIGHTSLAB_CERTS_DIR')
        if certs_dir is not None:
            certs_dir = certs_dir.strip().strip("'\"")
        return CertAuthManager(certs_dir=certs_dir, enable_auth=enable_auth)
