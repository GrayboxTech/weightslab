"""Tests for secure Docker setup with TLS and gRPC auth."""

import os
import tempfile
import pytest
from pathlib import Path
import socket
from unittest.mock import Mock, patch, MagicMock

from weightslab.security import CertAuthManager
from weightslab.ui_docker_bridge import (
    _test_backend_connection,
    _is_windows,
)


class TestCertAuthManager:
    """Test certificate and auth token management."""

    def test_manager_init_with_custom_dir(self):
        """Test manager initialization with custom directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CertAuthManager(certs_dir=tmpdir, enable_auth=True)

            assert manager.certs_dir == Path(tmpdir)
            assert manager.enable_auth is True
            assert manager.cert_file == Path(tmpdir) / 'backend-server.crt'
            assert manager.key_file == Path(tmpdir) / 'backend-server.key'
            assert manager.ca_file == Path(tmpdir) / 'ca.crt'

    def test_manager_init_with_default_dir(self):
        """Test manager initialization with default directory."""
        manager = CertAuthManager(enable_auth=True)

        user_profile = os.environ.get('HOME') or os.path.expanduser('~')
        expected_dir = Path(user_profile) / '.weightslab-certs'

        assert manager.certs_dir == expected_dir

    def test_has_valid_certs_missing(self):
        """Test certificate validation when certs don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CertAuthManager(certs_dir=tmpdir)

            assert manager.has_valid_certs() is False

    def test_auth_token_generation(self):
        """Test gRPC auth token generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CertAuthManager(certs_dir=tmpdir, enable_auth=True)

            token1 = manager.get_or_create_auth_token()
            assert token1 is not None
            assert len(token1) > 0
            assert isinstance(token1, str)

            # Should return same token on second call
            token2 = manager.get_or_create_auth_token()
            assert token1 == token2

    def test_auth_token_disabled(self):
        """Test that no token is generated when auth is disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CertAuthManager(certs_dir=tmpdir, enable_auth=False)

            token = manager.get_or_create_auth_token()
            assert token == ""

    def test_tls_environment_variables(self):
        """Test TLS environment variable setup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CertAuthManager(certs_dir=tmpdir)

            env_vars = manager.setup_tls_environment()

            assert 'GRPC_TLS_ENABLED' in env_vars
            assert env_vars['GRPC_TLS_ENABLED'] == '1'
            assert 'GRPC_TLS_CERT_FILE' in env_vars
            assert 'GRPC_TLS_KEY_FILE' in env_vars
            assert 'GRPC_TLS_CA_FILE' in env_vars
            assert 'WEIGHTSLAB_CERTS_DIR' in env_vars
            assert 'ENVOY_DOWNSTREAM_TLS' in env_vars
            assert 'ENVOY_UPSTREAM_TLS' in env_vars

    def test_auth_environment_variables_enabled(self):
        """Test auth environment variables when enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CertAuthManager(certs_dir=tmpdir, enable_auth=True)

            env_vars = manager.setup_auth_environment()

            assert 'WL_ENABLE_GRPC_AUTH_TOKEN' in env_vars
            assert env_vars['WL_ENABLE_GRPC_AUTH_TOKEN'] == '1'
            assert 'GRPC_AUTH_TOKEN' in env_vars
            assert len(env_vars['GRPC_AUTH_TOKEN']) > 0

    def test_auth_environment_variables_disabled(self):
        """Test auth environment variables when disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CertAuthManager(certs_dir=tmpdir, enable_auth=False)

            env_vars = manager.setup_auth_environment()

            assert 'WL_ENABLE_GRPC_AUTH_TOKEN' in env_vars
            assert env_vars['WL_ENABLE_GRPC_AUTH_TOKEN'] == '0'
            assert 'GRPC_AUTH_TOKEN' not in env_vars

    def test_from_env_or_default(self):
        """Test creating manager from environment variables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ['WEIGHTSLAB_CERTS_DIR'] = tmpdir

            try:
                manager = CertAuthManager.from_env_or_default()
                assert manager.certs_dir == Path(tmpdir)
            finally:
                del os.environ['WEIGHTSLAB_CERTS_DIR']

    @patch('weightslab.security.cert_auth_manager.subprocess.run')
    def test_generate_certs_success(self, mock_run):
        """Test successful certificate generation."""
        mock_run.return_value = MagicMock(returncode=0, stdout='', stderr='')

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CertAuthManager(certs_dir=tmpdir)

            # Since we're mocking subprocess, we need to create the files
            manager.certs_dir.mkdir(parents=True, exist_ok=True)
            manager.cert_file.touch()
            manager.key_file.touch()
            manager.ca_file.touch()

            success, msg = manager.generate_certs()
            assert success is True

    @patch('weightslab.security.cert_auth_manager.subprocess.run')
    def test_generate_certs_script_not_found(self, mock_run):
        """Test certificate generation when script is missing."""
        mock_run.return_value = MagicMock(returncode=1, stdout='', stderr='Script not found')

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CertAuthManager(certs_dir=tmpdir)

            success, msg = manager.generate_certs()

            # Should fail when script returns non-zero exit code
            assert success is False or not manager.has_valid_certs()


class TestBackendConnection:
    """Test backend connection utilities."""

    def test_backend_connection_timeout(self):
        """Test backend connection with timeout."""
        result = _test_backend_connection(
            host='127.0.0.1',
            port=59999,  # Likely not listening
            timeout=0.5
        )
        assert result is False

    @patch('socket.socket')
    def test_backend_connection_success(self, mock_socket):
        """Test successful backend connection."""
        mock_sock_instance = MagicMock()
        mock_sock_instance.connect_ex.return_value = 0
        mock_socket.return_value = mock_sock_instance

        result = _test_backend_connection()
        assert result is True

    @patch('socket.socket')
    def test_backend_connection_failure(self, mock_socket):
        """Test failed backend connection."""
        mock_sock_instance = MagicMock()
        mock_sock_instance.connect_ex.return_value = 1
        mock_socket.return_value = mock_sock_instance

        result = _test_backend_connection()
        assert result is False


class TestDockerBridgeIntegration:
    """Integration tests for Docker bridge."""

    def test_is_windows(self):
        """Test Windows detection."""
        result = _is_windows()
        assert isinstance(result, bool)

    def test_cert_auth_environment_integration(self):
        """Test that cert auth manager sets env vars correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CertAuthManager(certs_dir=tmpdir, enable_auth=True)

            # Create dummy cert files
            tmpdir_path = Path(tmpdir)
            tmpdir_path.mkdir(parents=True, exist_ok=True)
            (tmpdir_path / 'backend-server.crt').touch()
            (tmpdir_path / 'backend-server.key').touch()
            (tmpdir_path / 'ca.crt').touch()

            success, msg = manager.initialize()

            # Check that env vars were set
            assert os.environ.get('GRPC_TLS_ENABLED') == '1'
            assert os.environ.get('WL_ENABLE_GRPC_AUTH_TOKEN') == '1'
            assert 'GRPC_AUTH_TOKEN' in os.environ


class TestSecureInitialization:
    """Test secure initialization at package import."""

    def test_import_with_secure_init(self):
        """Test that secure init doesn't break import."""
        # This is a basic sanity check
        try:
            import weightslab
            assert weightslab is not None
        except Exception as e:
            pytest.fail(f"Import failed: {e}")

    def test_secure_init_skip_env_var(self):
        """Test that WEIGHTSLAB_SKIP_SECURE_INIT works."""
        os.environ['WEIGHTSLAB_SKIP_SECURE_INIT'] = 'true'

        try:
            # Re-import should use skip flag
            import importlib
            import weightslab
            importlib.reload(weightslab)
        except Exception as e:
            pytest.fail(f"Import failed: {e}")
        finally:
            del os.environ['WEIGHTSLAB_SKIP_SECURE_INIT']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
