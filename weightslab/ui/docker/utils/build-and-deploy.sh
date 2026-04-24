#!/bin/sh
# Auto-detect TLS and tokens, then build production image
# set -e

# Parse command line arguments
DEV=false

for arg in "$@"; do
  case "$arg" in
    --dev)
      DEV=true
      ;;
  esac
done

# Check for --unsecure flag
FORCE_UNSECURE=0
if [ "$1" = "--unsecure" ]; then
    echo "⚠ Forcing UNSECURE mode (HTTP, no auth)"
    FORCE_UNSECURE=1
    shift
fi

WEIGHTSLAB_CERTS_DIR="${HOME}/.weightslab-certs"
TOKEN_FILE="${WEIGHTSLAB_CERTS_DIR}/.grpc_auth_token"

echo "DEBUG: HOME=$HOME"
echo "DEBUG: WEIGHTSLAB_CERTS_DIR=$WEIGHTSLAB_CERTS_DIR"
echo "DEBUG: Checking for certs..."
ls -la "$WEIGHTSLAB_CERTS_DIR" 2>/dev/null || echo "DEBUG: WEIGHTSLAB_CERTS_DIR not found or empty"

# Detect TLS
if [ "$FORCE_UNSECURE" = "1" ]; then
    echo "✗ UNSECURE mode: HTTP (no TLS)"
    VITE_DEV_SERVER_HTTPS=0
    VITE_SERVER_PROTOCOL=http
elif [ -f "$WEIGHTSLAB_CERTS_DIR/envoy-server.crt" ] && [ -f "$WEIGHTSLAB_CERTS_DIR/envoy-server.key" ]; then
    echo "✓ TLS certificates found - building with HTTPS support"
    VITE_DEV_SERVER_HTTPS=1
    VITE_SERVER_PROTOCOL=https
else
    echo "✗ TLS certificates not found - building for HTTP mode"
    VITE_DEV_SERVER_HTTPS=0
    VITE_SERVER_PROTOCOL=http
fi

# Detect gRPC token
VITE_WL_ENABLE_GRPC_AUTH_TOKEN=0
VITE_GRPC_AUTH_TOKEN=""

if [ "$FORCE_UNSECURE" = "1" ]; then
    echo "✗ UNSECURE mode: auth disabled"
elif [ -f "$TOKEN_FILE" ]; then
    echo "✓ gRPC token found - enabling auth"
    VITE_WL_ENABLE_GRPC_AUTH_TOKEN=1
    VITE_GRPC_AUTH_TOKEN=$(cat "$TOKEN_FILE")
else
    echo "✗ gRPC token not found - auth disabled"
fi

# Write environment variables to .env file for docker compose
echo "Writing environment variables to .env..."
cat > .env << EOF
VITE_DEV_SERVER_HTTPS=$VITE_DEV_SERVER_HTTPS
VITE_SERVER_PROTOCOL=$VITE_SERVER_PROTOCOL
VITE_WL_ENABLE_GRPC_AUTH_TOKEN=$VITE_WL_ENABLE_GRPC_AUTH_TOKEN
VITE_GRPC_AUTH_TOKEN=$VITE_GRPC_AUTH_TOKEN
EOF

# Build and deploy
if [ "$DEV" = "true" ]; then
    # echo "Building development image..."
    # docker compose -f ../docker-compose.yml -f ../docker-compose.dev.yml build \
    # --build-arg VITE_DEV_SERVER_HTTPS="$VITE_DEV_SERVER_HTTPS" \
    # --build-arg VITE_WL_ENABLE_GRPC_AUTH_TOKEN="$VITE_WL_ENABLE_GRPC_AUTH_TOKEN" \
    # --build-arg VITE_GRPC_AUTH_TOKEN="$VITE_GRPC_AUTH_TOKEN" \
    # --build-arg VITE_SERVER_PROTOCOL="$VITE_SERVER_PROTOCOL" \
    # "$@"

    echo "✓ Dev build complete!"

    # Deploy (docker compose automatically reads .env)
    echo "Deploying containers..."
    docker compose -f ../docker-compose.yml -f ../docker-compose.dev.yml down
    docker compose -f ../docker-compose.yml -f ../docker-compose.dev.yml up -d

    echo "✓ Deployment to development complete!"
else
    # echo "Building production image..."
    # docker compose -f ../docker-compose.yml -f ../docker-compose.prod.yml build \
    # --build-arg VITE_DEV_SERVER_HTTPS="$VITE_DEV_SERVER_HTTPS" \
    # --build-arg VITE_WL_ENABLE_GRPC_AUTH_TOKEN="$VITE_WL_ENABLE_GRPC_AUTH_TOKEN" \
    # --build-arg VITE_GRPC_AUTH_TOKEN="$VITE_GRPC_AUTH_TOKEN" \
    # --build-arg VITE_SERVER_PROTOCOL="$VITE_SERVER_PROTOCOL" \
    # "$@"

    # echo "✓ Production build complete!"

    # Deploy (docker compose automatically reads .env)
    echo "Deploying containers..."
    docker compose -f ../docker-compose.yml -f ../docker-compose.prod.yml down
    docker compose -f ../docker-compose.yml -f ../docker-compose.prod.yml up -d

    echo "✓ Deployment to production complete!"
fi
