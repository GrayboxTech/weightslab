# Weights Studio Docker Deployment Guide

This Docker setup automatically adapts to the availability of TLS certificates and gRPC auth tokens. All scenarios use a **single port (5173)** that works with HTTP or HTTPS depending on certificate availability.

## Quick Start

### Scenario 1: HTTP Only (No Certs, No Auth) - Development/Testing

Fastest way to get running without certificates:

```bash
# Copy the default environment
cp .env.example .env

# Ensure certificates directory is empty
rm -rf ../envoy/certs/*

# Run with production compose
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Access at: http://localhost:5173
```

The container will:
- Listen on port 443 in HTTP mode
- Disable gRPC auth token validation
- Host port 5173 maps to container port 443

### Scenario 2: HTTPS + gRPC Auth - Secure Production

```bash
# Generate certificates and auth token (one-time setup)
weightslab ui se

# Copy environment with auth enabled
cp .env.example .env
cat >> .env << 'EOF'
VITE_WL_ENABLE_GRPC_AUTH_TOKEN=1
EOF

# Run with production compose
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Access at: https://localhost:5173
# Frontend will validate gRPC auth token on API calls
```

The container will:
- Detect certificates in `/etc/nginx/certs/`
- Enable HTTPS on port 443
- Host port 5173 maps to container port 443 (HTTPS)
- Pass gRPC auth token to frontend

### Scenario 3: HTTPS Only (No Auth)

```bash
# Generate certificates without auth
weightslab ui se --no-auth

# Copy environment with HTTPS, no auth
cp .env.example .env

# Run with production compose
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Access at: https://localhost:5173
```

The container will:
- Detect certificates
- Enable HTTPS on port 443
- Disable gRPC auth token validation
- Host port 5173 maps to container port 443 (HTTPS)

## How It Works

### Dynamic Configuration

The Docker image includes an entrypoint script (`nginx-entrypoint.sh`) that:

1. **Checks for certificates** at container startup:
   - Looks for `/etc/nginx/certs/envoy-server.crt` and `.key`

2. **Generates appropriate nginx server block**:
   - Single server listening on port 443
   - If certs found → Enable `ssl` directive + certificate paths
   - If certs missing → Plain HTTP on port 443

3. **Validates the config** and starts nginx

### Volume Mounts

The `docker-compose.yml` mounts:

```yaml
volumes:
  - ../envoy/certs:/etc/nginx/certs:ro  # Optional - uses HTTP if empty or missing
```

The mount path is **optional**:
- If `../envoy/certs/` exists but is empty → HTTP on port 443
- If certificates files exist → HTTPS on port 443
- If directory doesn't exist → Docker creates it (empty) → HTTP on port 443

### Certificate Files

When certificates are available, they must be at:

```
weightslab/ui/envoy/certs/
├── envoy-server.crt
└── envoy-server.key
```

Generate with:

```bash
weightslab ui se [--force-certs] [--no-auth]
```

## Environment Variables

### Frontend Configuration

| Variable | Default | Effect |
|----------|---------|--------|
| `VITE_WL_ENABLE_GRPC_AUTH_TOKEN` | `0` | Enable gRPC auth validation on frontend |
| `VITE_GRPC_AUTH_TOKEN` | empty | Auth token sent with gRPC calls |
| `WS_SERVER_HOST` | `localhost` | Backend server address |
| `WS_SERVER_PORT` | `8080` | Backend server port |
| `WS_SERVER_PROTOCOL` | `https` | Backend protocol (http/https) |

### Docker Configuration

| Variable | Default | Effect |
|----------|---------|--------|
| `VITE_PORT` | `5173` | Host port (container always uses 443) |
| `GRPC_BACKEND_PORT` | `50051` | Backend gRPC port for Envoy |
| `ENVOY_PORT` | `8080` | Envoy gRPC-Web proxy port |
| `NODE_ENV` | `production` | Node environment |

## Health Checks

The nginx config includes a health endpoint at `/health`:

```bash
# HTTP mode (no certs)
curl http://localhost:5173/health
# Output: healthy

# HTTPS mode (with certs)
curl https://localhost:5173/health
# Output: healthy (ignoring cert warnings in curl: curl -k)
```

## Troubleshooting

### Container won't start

1. Check logs:
   ```bash
   docker compose logs weights_studio
   ```

2. Common issues:
   - Certs directory doesn't exist → Create: `mkdir -p ../envoy/certs`
   - Port conflict → Check `docker ps` or change `VITE_HTTP_PORT`/`VITE_HTTPS_PORT`

### HTTPS not working

1. Verify certificates exist:
   ```bash
   ls -la ../envoy/certs/
   ```

2. Check nginx accepted the config:
   ```bash
   docker exec weights_studio_frontend nginx -t
   ```

### gRPC calls failing

1. Check Envoy is running:
   ```bash
   docker ps | grep envoy
   ```

2. Verify backend is reachable:
   ```bash
   docker exec weights_studio_frontend curl -v http://weights_studio_envoy:8080/
   ```

3. Validate auth token if enabled:
   - Frontend sends `Authorization: Bearer <token>` header
   - Backend must accept the same token

## Production Deployment

### Pre-deployment Checklist

- [ ] Generate TLS certificates (recommended for production)
- [ ] Set gRPC auth token (if exposing externally)
- [ ] Configure `WS_SERVER_HOST` to point to actual backend
- [ ] Test health endpoints
- [ ] Verify backend connectivity
- [ ] Review nginx logs after startup

### Recommended Configuration

```bash
# Production with full security
weightslab ui se --force-certs  # Generate fresh certs

# Set environment
cat > .env << 'EOF'
VITE_WL_ENABLE_GRPC_AUTH_TOKEN=1
WS_SERVER_HOST=backend.production.internal
WS_SERVER_PORT=8080
WS_SERVER_PROTOCOL=https
NODE_ENV=production
EOF

# Deploy
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Scaling

To run multiple instances:

1. Change container names and ports per instance:
   ```bash
   VITE_HTTP_PORT=5174 VITE_HTTPS_PORT=5175 docker compose ... up -d
   ```

2. Use a load balancer in front
3. Share the same certificates and auth token across instances
4. Ensure all point to the same backend

## Docker Image

### Building

```bash
# From weightslab/ui/ directory
docker build -t weights_studio_frontend:latest .
```

### Publishing

```bash
docker tag weights_studio_frontend:latest graybx/weightslab:latest
docker push graybx/weightslab:latest
```

Then update `docker-compose.yml`:
```yaml
services:
  weights_studio:
    image: graybx/weightslab:latest
```
