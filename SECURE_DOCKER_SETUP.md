# Secure WeightsLab Docker Setup with TLS and gRPC Authentication

This guide explains how to set up and run WeightsLab with secured Docker containers using TLS certificates and gRPC authentication tokens.

## Overview

The secure setup provides:

- **TLS Encryption**: All backend-UI communication is encrypted with TLS certificates
- **gRPC Authentication**: Mutual authentication using secure tokens
- **Automatic Initialization**: Certs and tokens are auto-generated on first run
- **Bidirectional Setup**: Works whether backend or UI starts first

## Quick Start

### 1. Basic Setup (Automatic)

When you run WeightsLab for the first time, certificates and auth tokens are automatically created:

```python
import weightslab as wl

# This will automatically initialize secure certs if not already done
model = wl.watch_or_edit(my_model)
optimizer = wl.watch_or_edit(my_optimizer)
# ... your training code
```

### 2. Launch Docker with Secure TLS

#### Using Python CLI:

```python
from weightslab.ui_docker_bridge import initialize_weightslab_docker

# Initialize and launch
initialize_weightslab_docker(enable_auth=True)
```

#### Using Command Line:

```bash
# Launch with TLS and auth
python -m weightslab.ui_docker_bridge docker launch

# Launch with dev configuration
python -m weightslab.ui_docker_bridge docker launch --dev

# Launch and test backend communication
python -m weightslab.ui_docker_bridge docker launch --test

# Force regenerate certificates
python -m weightslab.ui_docker_bridge docker launch --force-certs

# Disable auth token
python -m weightslab.ui_docker_bridge docker launch --no-auth
```

### 3. Workflow: Backend First

When you start WeightsLab training first:

```bash
# 1. Start your experiment (auto-creates certs in ~/.weightslab-certs)
python your_training_script.py

# 2. In another terminal, launch Docker with same certs
python -m weightslab.ui_docker_bridge docker launch --test

# Docker will use environment variables set by bootstrap-secure.ps1
# Backend and UI communicate securely
```

### 4. Workflow: UI First

When you start Docker UI first:

```powershell
# 1. Run bootstrap-secure.ps1 (sets env vars, creates certs)
# Linux/Mac: Run the UI with manual env setup
# Or from Python:
from weightslab.ui_docker_bridge import initialize_weightslab_docker
initialize_weightslab_docker()

# 2. Environment variables are exported with cert paths
# GRPC_TLS_CERT_FILE, GRPC_TLS_KEY_FILE, GRPC_TLS_CA_FILE
# GRPC_AUTH_TOKEN are all set

# 3. In the same shell, start your training
python your_training_script.py

# Backend uses the same certs and auth token from environment
```

## File Locations

### Certificates Directory

Default location: `~/.weightslab-certs/`

Contains:
- `backend-server.crt` - Backend TLS certificate
- `backend-server.key` - Backend TLS private key
- `ca.crt` - Certificate authority
- `.grpc_auth_token` - Secure auth token (readable by owner only)
- `envoy-server.crt` - Envoy proxy certificate
- `envoy-server.key` - Envoy proxy key

Custom location via environment variable:
```bash
export WEIGHTSLAB_CERTS_DIR=/custom/path/to/certs
```

## Environment Variables

The system automatically sets these variables:

```bash
# TLS Configuration
GRPC_TLS_ENABLED=1
GRPC_TLS_REQUIRE_CLIENT_AUTH=1
GRPC_TLS_CERT_FILE=~/.weightslab-certs/backend-server.crt
GRPC_TLS_KEY_FILE=~/.weightslab-certs/backend-server.key
GRPC_TLS_CA_FILE=~/.weightslab-certs/ca.crt
WEIGHTSLAB_CERTS_DIR=~/.weightslab-certs

# Proxy Configuration
ENVOY_DOWNSTREAM_TLS=on
ENVOY_UPSTREAM_TLS=on

# Protocol Configuration
WS_SERVER_PROTOCOL=https
VITE_SERVER_PROTOCOL=https

# Authentication
WL_ENABLE_GRPC_AUTH_TOKEN=1
GRPC_AUTH_TOKEN=<secure-hex-token>
VITE_GRPC_AUTH_TOKEN=<secure-hex-token>
```

## Python API

### CertAuthManager

Manage certificates and auth tokens programmatically:

```python
from weightslab.security import CertAuthManager

# Create manager with default settings
manager = CertAuthManager()

# Or with custom certificate directory
manager = CertAuthManager(certs_dir='/custom/path', enable_auth=True)

# Check if certificates exist
if manager.has_valid_certs():
    print("Certificates already exist")
else:
    print("Certificates need to be generated")

# Generate certificates (or skip if they exist)
success, msg = manager.generate_certs(force=False)
if success:
    print(msg)
else:
    print(f"Error: {msg}")

# Get or create auth token
token = manager.get_or_create_auth_token()
print(f"Auth token: {token}")

# Initialize everything at once
success, msg = manager.initialize(force_certs=False)
if success:
    # Environment variables are now set
    print("Ready to communicate securely")

# Get environment variables for manual setup
tls_env = manager.setup_tls_environment()
auth_env = manager.setup_auth_environment()

for key, value in {**tls_env, **auth_env}.items():
    import os
    os.environ[key] = value
```

## Docker Bridge Commands

### Launch Variations

```bash
# Standard launch with TLS
python -m weightslab.ui_docker_bridge docker launch

# Development mode (live reload, debug)
python -m weightslab.ui_docker_bridge docker launch --dev

# Test communication after launch
python -m weightslab.ui_docker_bridge docker launch --test

# Force regenerate certificates
python -m weightslab.ui_docker_bridge docker launch --force-certs

# Disable authentication (not recommended for production)
python -m weightslab.ui_docker_bridge docker launch --no-auth

# Show current configuration
python -m weightslab.ui_docker_bridge docker info

# Stop containers
python -m weightslab.ui_docker_bridge docker stop

# Legacy UI commands (without TLS)
python -m weightslab.ui_docker_bridge ui launch
python -m weightslab.ui_docker_bridge ui stop
python -m weightslab.ui_docker_bridge ui drop
```

## Configuration Options

### Disable Automatic Initialization

If you want to skip automatic cert/auth initialization:

```bash
export WEIGHTSLAB_SKIP_SECURE_INIT=true
python your_script.py
```

### Custom Log Directory

Certificates are stored in `~/.weightslab-certs` by default. To use a custom location:

```bash
export WEIGHTSLAB_CERTS_DIR=/path/to/certs
```

This works across both backend and UI - they'll use the same certificates.

### Disable gRPC Authentication

To run without auth tokens (not recommended):

```bash
python -m weightslab.ui_docker_bridge docker launch --no-auth
```

Or programmatically:

```python
from weightslab.security import CertAuthManager

manager = CertAuthManager(enable_auth=False)
manager.initialize()
```

## Troubleshooting

### Backend and UI Not Communicating

1. **Check certificates exist**:
   ```bash
   ls -la ~/.weightslab-certs/
   ```

2. **Verify environment variables**:
   ```python
   import os
   print(f"GRPC_TLS_ENABLED: {os.environ.get('GRPC_TLS_ENABLED')}")
   print(f"GRPC_AUTH_TOKEN: {os.environ.get('GRPC_AUTH_TOKEN')}")
   ```

3. **Test backend connectivity**:
   ```bash
   python -m weightslab.ui_docker_bridge docker launch --test
   ```

### Port Already in Use

If port 50051 (backend) or 8080 (Envoy proxy) is already in use:

```bash
# Check what's using the port
netstat -ano | grep 50051

# Or use a different port via environment variable
export GRPC_BACKEND_PORT=50052
```

### Certificate Generation Fails on PowerShell

Ensure PowerShell execution policy allows scripts:

```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Windows Path Issues

If you get path-related errors, ensure paths use forward slashes in Python or escape backslashes:

```python
from pathlib import Path
certs_dir = Path.home() / '.weightslab-certs'  # Recommended
```

## Complete Example: Two-Terminal Setup

### Terminal 1: Start Training with Secure Setup

```bash
cd ~/my_weightslab_project
python train.py
```

Contents of `train.py`:
```python
import weightslab as wl  # Auto-initializes secure certs

model = wl.watch_or_edit(MyModel())
optimizer = wl.watch_or_edit(MyOptimizer())

# Training loop
for epoch in range(10):
    # ... training code
    pass
```

### Terminal 2: Launch UI in Same Environment

```bash
cd ~/my_weightslab_project

# Ensure same cert directory (optional, uses default ~/.weightslab-certs)
export WEIGHTSLAB_CERTS_DIR=~/.weightslab-certs

# Launch UI with TLS
python -m weightslab.ui_docker_bridge docker launch --test
```

Both processes now use:
- Same TLS certificates
- Same gRPC auth token
- Secure encrypted communication

## Security Notes

### Best Practices

1. **Protect certificate directory**: The `.weightslab-certs` directory contains private keys
   ```bash
   chmod 700 ~/.weightslab-certs
   chmod 600 ~/.weightslab-certs/*
   ```

2. **Rotate tokens periodically**:
   ```bash
   rm ~/.weightslab-certs/.grpc_auth_token
   # Next run will generate a new token
   ```

3. **Never commit certs to version control**:
   ```bash
   echo ".weightslab-certs/" >> .gitignore
   ```

4. **Use environment variables for sensitive data**:
   ```bash
   export WEIGHTSLAB_CERTS_DIR=/secure/location
   ```

### Production Considerations

- Use proper certificate management (e.g., Let's Encrypt for public services)
- Implement certificate rotation policies
- Use mutual TLS (mTLS) for all services
- Monitor certificate expiration
- Implement audit logging

## API Reference

### CertAuthManager Class

```python
class CertAuthManager:
    def __init__(self, certs_dir: Optional[str] = None, enable_auth: bool = True)
    
    # Properties
    certs_dir: Path
    enable_auth: bool
    cert_file: Path
    key_file: Path
    ca_file: Path
    token_file: Path
    
    # Methods
    def has_valid_certs() -> bool
    def generate_certs(force: bool = False) -> Tuple[bool, str]
    def get_or_create_auth_token() -> str
    def setup_tls_environment() -> dict
    def setup_auth_environment() -> dict
    def initialize(force_certs: bool = False) -> Tuple[bool, str]
    
    @staticmethod
    def from_env_or_default(enable_auth: bool = True) -> CertAuthManager
```

### Docker Bridge Functions

```python
def initialize_weightslab_docker(
    force_certs: bool = False,
    enable_auth: bool = True
) -> bool
    """Initialize Docker deployment from Python"""

def _test_backend_connection(
    host: str = '127.0.0.1',
    port: int = 50051,
    timeout: float = 5.0
) -> bool
    """Test if backend gRPC server is reachable"""
```

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review environment variables with `docker info` command
3. Check logs in `~/.weightslab_logs/`
4. Open an issue on GitHub with:
   - Python version
   - OS (Windows/Linux/Mac)
   - Error message and logs
   - Commands that failed
