"""Tests for secure environment setup on the native (Docker-free) stack."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from weightslab.security import CertAuthManager


class TestCertAuthManager:
    """Certificate + auth token manager tests."""

    def test_manager_init_with_custom_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CertAuthManager(certs_dir=tmpdir, enable_auth=True)
            assert manager.certs_dir == Path(tmpdir)
            assert manager.enable_auth is True
            assert manager.cert_file == Path(tmpdir) / "backend-server.crt"
            assert manager.key_file == Path(tmpdir) / "backend-server.key"
            assert manager.ca_file == Path(tmpdir) / "ca.crt"

    def test_has_valid_certs_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CertAuthManager(certs_dir=tmpdir)
            assert manager.has_valid_certs() is False

    def test_auth_token_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CertAuthManager(certs_dir=tmpdir, enable_auth=True)
            token1 = manager.get_or_create_auth_token()
            token2 = manager.get_or_create_auth_token()
            assert token1
            assert token1 == token2

    def test_auth_token_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CertAuthManager(certs_dir=tmpdir, enable_auth=False)
            assert manager.get_or_create_auth_token() == ""

    def test_tls_environment_variables_no_legacy_envoy_vars(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CertAuthManager(certs_dir=tmpdir)
            env_vars = manager.setup_tls_environment()

            assert env_vars["GRPC_TLS_ENABLED"] == "1"
            assert env_vars["GRPC_TLS_REQUIRE_CLIENT_AUTH"] == "1"
            assert "GRPC_TLS_CERT_FILE" in env_vars
            assert "GRPC_TLS_KEY_FILE" in env_vars
            assert "GRPC_TLS_CA_FILE" in env_vars
            assert env_vars["WEIGHTSLAB_CERTS_DIR"] == tmpdir

            assert "ENVOY_DOWNSTREAM_TLS" not in env_vars
            assert "ENVOY_UPSTREAM_TLS" not in env_vars
            assert "VITE_SERVER_PROTOCOL" not in env_vars

    def test_auth_environment_variables_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CertAuthManager(certs_dir=tmpdir, enable_auth=True)
            env_vars = manager.setup_auth_environment()

            assert env_vars["WL_ENABLE_GRPC_AUTH_TOKEN"] == "1"
            assert "GRPC_AUTH_TOKEN" in env_vars
            assert "VITE_WL_ENABLE_GRPC_AUTH_TOKEN" not in env_vars
            assert "VITE_GRPC_AUTH_TOKEN" not in env_vars

    def test_auth_environment_variables_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CertAuthManager(certs_dir=tmpdir, enable_auth=False)
            env_vars = manager.setup_auth_environment()

            assert env_vars["WL_ENABLE_GRPC_AUTH_TOKEN"] == "0"
            assert "GRPC_AUTH_TOKEN" not in env_vars

    def test_from_env_or_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["WEIGHTSLAB_CERTS_DIR"] = tmpdir
            try:
                manager = CertAuthManager.from_env_or_default()
                assert manager.certs_dir == Path(tmpdir)
            finally:
                del os.environ["WEIGHTSLAB_CERTS_DIR"]

    @patch("weightslab.security.cert_auth_manager.subprocess.run")
    def test_generate_certs_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CertAuthManager(certs_dir=tmpdir)
            success, _ = manager.generate_certs()
            assert success is True

    @patch("weightslab.security.cert_auth_manager.subprocess.run")
    @patch("weightslab.security.cert_auth_manager.Path.exists")
    def test_generate_certs_script_missing(self, mock_exists, mock_run):
        # First exists call is for cert file checks (False), then script path (False).
        mock_exists.return_value = False
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CertAuthManager(certs_dir=tmpdir)
            success, msg = manager.generate_certs()
            assert success is False
            assert "not found" in msg.lower()


class TestSecureInitialization:
    """Import-time secure initialization sanity checks."""

    def test_import_with_secure_init(self):
        try:
            import weightslab
            assert weightslab is not None
        except Exception as exc:
            pytest.fail(f"Import failed: {exc}")

    def test_secure_init_skip_env_var(self):
        os.environ["WEIGHTSLAB_SKIP_SECURE_INIT"] = "true"
        try:
            import importlib
            import weightslab

            importlib.reload(weightslab)
        except Exception as exc:
            pytest.fail(f"Import failed: {exc}")
        finally:
            del os.environ["WEIGHTSLAB_SKIP_SECURE_INIT"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
