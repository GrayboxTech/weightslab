# WeightsLab Secure Docker - Quick Start Guide

Get started with secured WeightsLab Docker containers in 5 minutes.

## Prerequisites

- Python 3.8+
- Docker Desktop or Docker engine
- Windows with PowerShell (for bootstrap script) OR Linux/Mac with equivalent setup

## 30-Second Setup

### Option A: Backend First (Recommended)

```bash
# Terminal 1: Start your training (auto-creates certs)
python your_training_script.py

# Terminal 2: Launch Docker with secure TLS
python -m weightslab.ui_docker_bridge docker launch --test
```

### Option B: UI First

```bash
# Terminal 1: Launch secure Docker first
python -m weightslab.ui_docker_bridge docker launch --test

# Terminal 2: Start training in the same environment
python your_training_script.py
```

## What Happens Automatically

When you import weightslab or launch Docker:

1. ✓ Generates TLS certificates (if needed) in `~/.weightslab-certs/`
2. ✓ Creates secure gRPC auth tokens
3. ✓ Sets up all environment variables
4. ✓ Backend and UI communicate securely over HTTPS with mutual authentication

## Verify It's Working

```bash
# Test connection
python -m weightslab.ui_docker_bridge docker launch --test

# Check configuration
python -m weightslab.ui_docker_bridge docker info

# View logs
ls ~/.weightslab_logs/
```

## Common Commands

```bash
# Launch with TLS (production)
python -m weightslab.ui_docker_bridge docker launch

# Launch dev mode (live reload)
python -m weightslab.ui_docker_bridge docker launch --dev

# Regenerate certificates
python -m weightslab.ui_docker_bridge docker launch --force-certs

# Stop containers
python -m weightslab.ui_docker_bridge docker stop

# Show configuration
python -m weightslab.ui_docker_bridge docker info
```

## Typical Workflow

```python
# 1. Your training script (train.py)
import weightslab as wl  # Auto-creates certs!

model = wl.watch_or_edit(MyModel())
optimizer = wl.watch_or_edit(MyOptimizer())

for epoch in range(10):
    # training code
    pass
```

```bash
# 2. Start training
python train.py

# 3. In another terminal, launch UI
python -m weightslab.ui_docker_bridge docker launch --test

# 4. Open browser to https://localhost:8080
```

## File Locations

```
~/.weightslab-certs/          # All certificates and tokens
├── backend-server.crt        # Backend TLS certificate
├── backend-server.key        # Backend TLS key
├── ca.crt                    # Certificate authority
└── .grpc_auth_token         # Authentication token

~/.weightslab_logs/          # Training logs
```

## Troubleshooting

### Containers not starting?
```bash
# Check Docker is running
docker ps

# See detailed errors
python -m weightslab.ui_docker_bridge docker launch --test
```

### Backend and UI not connecting?
```bash
# Verify backend is running
python -m weightslab.ui_docker_bridge docker launch --test

# Check environment variables are set
import os
print(os.environ.get('GRPC_AUTH_TOKEN'))
```

### Reset everything?
```bash
# Stop Docker
python -m weightslab.ui_docker_bridge docker stop

# Regenerate all certs
rm -rf ~/.weightslab-certs
python -m weightslab.ui_docker_bridge docker launch --force-certs
```

## Configuration

### Custom certificate directory
```bash
export WEIGHTSLAB_CERTS_DIR=/custom/path
python -m weightslab.ui_docker_bridge docker launch
```

### Skip auto-init (optional)
```bash
export WEIGHTSLAB_SKIP_SECURE_INIT=true
python your_script.py
```

### Disable auth (not recommended)
```bash
python -m weightslab.ui_docker_bridge docker launch --no-auth
```

## Python API

```python
from weightslab.security import CertAuthManager

# Programmatic setup
manager = CertAuthManager()
success, msg = manager.initialize()

# Or with custom directory
manager = CertAuthManager(certs_dir='/path/to/certs')
manager.initialize(force_certs=True)

# Launch Docker from Python
from weightslab.ui_docker_bridge import initialize_weightslab_docker
initialize_weightslab_docker(force_certs=False, enable_auth=True)
```

## Next Steps

1. **Run examples**: `python examples/secure_docker_example.py`
2. **Read full guide**: See `SECURE_DOCKER_SETUP.md`
3. **Check logs**: `tail -f ~/.weightslab_logs/*`
4. **Access UI**: `https://localhost:8080` (after Docker launches)

## Security

- ✓ TLS encrypted communication
- ✓ Mutual authentication with tokens
- ✓ Certificates in user home directory
- ✓ Auto-rotation support

⚠️ Keep `~/.weightslab-certs/` private (contains private keys)

## Support

For detailed information, see:
- `SECURE_DOCKER_SETUP.md` - Complete documentation
- `examples/secure_docker_example.py` - Working examples
- `weightslab/tests/test_secure_docker.py` - Test cases
