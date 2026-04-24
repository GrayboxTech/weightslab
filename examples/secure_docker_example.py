#!/usr/bin/env python
"""
Example: Running WeightsLab with Secure Docker Setup

This example demonstrates:
1. Automatic certificate and auth token initialization
2. Secure backend-UI communication
3. Using the Docker bridge to launch containers
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_automatic_initialization():
    """
    Example 1: Automatic Secure Initialization

    When you import weightslab, certificates and auth tokens are
    automatically created if they don't exist.
    """
    logger.info("=" * 60)
    logger.info("Example 1: Automatic Secure Initialization")
    logger.info("=" * 60)

    # Simply importing weightslab triggers secure initialization
    import weightslab as wl
    from weightslab.security import CertAuthManager

    # Verify that certs were created
    manager = CertAuthManager.from_env_or_default()

    logger.info(f"Certificates directory: {manager.certs_dir}")
    logger.info(f"Certificates exist: {manager.has_valid_certs()}")

    if manager.has_valid_certs():
        logger.info("✓ Certificates are ready for secure communication")
    else:
        logger.warning("Certificates not found - manual setup required")

    # Show environment variables set by initialization
    tls_env = manager.setup_tls_environment()
    logger.info("\nTLS Environment Variables:")
    for key, value in tls_env.items():
        logger.info(f"  {key} = {value}")


def example_2_programmatic_docker_launch():
    """
    Example 2: Programmatic Docker Launch

    Launch Docker containers with secure TLS from Python code.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Programmatic Docker Launch")
    logger.info("=" * 60)

    from weightslab.ui_docker_bridge import initialize_weightslab_docker

    logger.info("Initializing Docker deployment...")

    try:
        success = initialize_weightslab_docker(
            force_certs=False,  # Don't regenerate if they exist
            enable_auth=True     # Use authentication
        )

        if success:
            logger.info("✓ Docker initialization successful")
            logger.info("  You can now launch Docker with:")
            logger.info("  python -m weightslab.ui_docker_bridge docker launch")
        else:
            logger.error("✗ Docker initialization failed")
            return False

    except Exception as e:
        logger.error(f"Docker initialization error: {e}")
        return False

    return True


def example_3_verify_cert_auth_manager():
    """
    Example 3: Using CertAuthManager Directly

    Detailed control over certificate and auth token management.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: CertAuthManager Direct Usage")
    logger.info("=" * 60)

    from weightslab.security import CertAuthManager

    # Create manager with default settings
    manager = CertAuthManager.from_env_or_default(enable_auth=True)

    logger.info(f"Working with certs in: {manager.certs_dir}")

    # Check certificate status
    if manager.has_valid_certs():
        logger.info("✓ Certificates exist and are valid")
    else:
        logger.info("Certificates need to be generated")

        logger.info("Generating certificates...")
        success, msg = manager.generate_certs(force=False)

        if success:
            logger.info(f"✓ {msg}")
        else:
            logger.error(f"✗ {msg}")
            return False

    # Get or create auth token
    logger.info("Setting up authentication...")
    token = manager.get_or_create_auth_token()
    logger.info(f"✓ Auth token ready ({len(token)} chars)")

    # Get all environment variables needed
    logger.info("\nEnvironment variables to set:")
    tls_env = manager.setup_tls_environment()
    auth_env = manager.setup_auth_environment()

    all_env = {**tls_env, **auth_env}
    for key, value in sorted(all_env.items()):
        if 'TOKEN' in key or 'KEY' in key:
            # Don't print sensitive values
            logger.info(f"  {key} = ****")
        else:
            logger.info(f"  {key} = {value}")

    logger.info("\nSetting up secure environment...")
    success, msg = manager.setup_secure_environment()
    if not success:
        logger.error(f"✗ Failed to setup secure environment: {msg}")
        return False
    logger.info(f"✓ {msg}")

    return True


def example_4_test_backend_connection():
    """
    Example 4: Test Backend-UI Communication

    Verify that the backend server is reachable.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Example 4: Test Backend Connection")
    logger.info("=" * 60)

    from weightslab.ui_docker_bridge import _test_backend_connection

    logger.info("Testing backend server connectivity...")
    logger.info("  Target: 127.0.0.1:50051")
    logger.info("  Timeout: 5.0 seconds")

    is_reachable = _test_backend_connection(
        host='127.0.0.1',
        port=50051,
        timeout=5.0
    )

    if is_reachable:
        logger.info("✓ Backend server is reachable and responsive")
    else:
        logger.warning("✗ Backend server not responding")
        logger.warning("  Make sure to start the backend before the UI:")
        logger.warning("  python -m weightslab.backend.grpc_server")

    return is_reachable


def example_5_environment_variables():
    """
    Example 5: Environment Variables and Configuration

    Show how environment variables control the secure setup.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Example 5: Environment Variables")
    logger.info("=" * 60)

    import os

    logger.info("Current secure configuration:")

    key_vars = [
        'WEIGHTSLAB_CERTS_DIR',
        'GRPC_TLS_ENABLED',
        'GRPC_TLS_REQUIRE_CLIENT_AUTH',
        'WL_ENABLE_GRPC_AUTH_TOKEN',
        'WEIGHTSLAB_SKIP_SECURE_INIT',
        'WEIGHTSLAB_LOG_LEVEL',
    ]

    for key in key_vars:
        value = os.environ.get(key, '(not set)')
        logger.info(f"  {key:35} = {value}")

    logger.info("\nYou can customize by setting environment variables:")
    logger.info("  export WEIGHTSLAB_CERTS_DIR=/custom/path")
    logger.info("  export GRPC_BACKEND_PORT=50052")
    logger.info("  export WEIGHTSLAB_SKIP_SECURE_INIT=true")


def example_6_complete_workflow():
    """
    Example 6: Complete Workflow

    Shows the typical sequence of operations to run a secure setup.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Example 6: Complete Workflow")
    logger.info("=" * 60)

    from weightslab.security import CertAuthManager

    logger.info("Step 1: Setup secure environment")
    manager = CertAuthManager.from_env_or_default(enable_auth=True)
    success, msg = manager.setup_secure_environment(force_certs=False)
    if not success:
        logger.error(f"  Failed: {msg}")
        return False
    logger.info(f"  Result: {msg}")

    logger.info("\nStep 2: Verify certificates")
    if manager.has_valid_certs():
        logger.info("  ✓ Certificates are valid")
    else:
        logger.info("  ✗ Certificates are missing")
        return False

    logger.info("\nStep 3: Setup environment")
    env_vars = {**manager.setup_tls_environment(), **manager.setup_auth_environment()}
    logger.info(f"  Set {len(env_vars)} environment variables")

    logger.info("\nStep 4: Ready for communication")
    logger.info("  Backend and UI can now communicate securely")
    logger.info("  Launch with: python -m weightslab.ui_docker_bridge docker launch")

    return True


def main():
    """Run all examples."""
    logger.info("\nWeightsLab Secure Docker Setup Examples\n")

    # Show which examples to run
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num == '1':
            example_1_automatic_initialization()
        elif example_num == '2':
            example_2_programmatic_docker_launch()
        elif example_num == '3':
            example_3_verify_cert_auth_manager()
        elif example_num == '4':
            example_4_test_backend_connection()
        elif example_num == '5':
            example_5_environment_variables()
        elif example_num == '6':
            example_6_complete_workflow()
        else:
            logger.info("Usage: python secure_docker_example.py [1-6]")
    else:
        # Run all examples
        try:
            example_1_automatic_initialization()
            example_3_verify_cert_auth_manager()
            example_4_test_backend_connection()
            example_5_environment_variables()
            example_6_complete_workflow()

            logger.info("\n" + "=" * 60)
            logger.info("All examples completed successfully!")
            logger.info("=" * 60)
            logger.info("\nNext steps:")
            logger.info("1. Start your training script (auto-creates certs)")
            logger.info("2. Launch UI: python -m weightslab.ui_docker_bridge docker launch")
            logger.info("3. Access at: https://localhost:8080")
            logger.info("\nFor more details, see SECURE_DOCKER_SETUP.md")

        except Exception as e:
            logger.error(f"Example execution failed: {e}", exc_info=True)
            return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
