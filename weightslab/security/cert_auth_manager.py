"""Manage secure certificates and gRPC auth tokens for weightslab."""

import os
import logging
import subprocess
import json
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def _get_user_profile() -> str:
    """Get user profile directory."""
    return os.environ.get('USERPROFILE') or os.path.expanduser('~')


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
            self.certs_dir = Path(certs_dir)

        self.cert_file = self.certs_dir / 'backend-server.crt'
        self.key_file = self.certs_dir / 'backend-server.key'
        self.ca_file = self.certs_dir / 'ca.crt'
        self.token_file = self.certs_dir / '.grpc_auth_token'

    def has_valid_certs(self) -> bool:
        """Check if certificates exist and are valid."""
        return (
            self.cert_file.exists() and
            self.key_file.exists() and
            self.ca_file.exists()
        )

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

        # Check if token is already set in environment (from bootstrap script)
        env_token = os.environ.get('GRPC_AUTH_TOKEN', '').strip()
        if env_token:
            logger.debug("Using GRPC_AUTH_TOKEN from environment")
            return env_token

        # Check if token exists in file
        if self.token_file.exists():
            try:
                with open(self.token_file, 'r') as f:
                    token = f.read().strip()
                    if token:
                        logger.debug("Using existing gRPC auth token from file")
                        return token
            except Exception as e:
                logger.warning(f"Could not read existing token: {e}")

        # Generate new token
        token = _generate_hex_token()

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
            logger.info(f"Generated new gRPC auth token")
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

    def initialize(self, force_certs: bool = False) -> Tuple[bool, str]:
        """
        Initialize certificates and auth tokens.

        Args:
            force_certs: Force regenerate certificates

        Returns:
            Tuple of (success, message)
        """
        # Ensure certs directory exists
        self.certs_dir.mkdir(parents=True, exist_ok=True)

        # Generate certs if needed
        success, msg = self.generate_certs(force=force_certs)
        if not success:
            return False, msg

        # Get or create auth token
        token = self.get_or_create_auth_token()

        # Set environment variables
        env_vars = self.setup_tls_environment()
        env_vars.update(self.setup_auth_environment())

        for key, value in env_vars.items():
            os.environ[key] = value

        logger.info("TLS and auth environment initialized")
        return True, "TLS and auth initialized successfully"

    @staticmethod
    def from_env_or_default(enable_auth: bool = True) -> 'CertAuthManager':
        """Create manager using environment variables or defaults."""
        certs_dir = os.environ.get('WEIGHTSLAB_CERTS_DIR')
        return CertAuthManager(certs_dir=certs_dir, enable_auth=enable_auth)
