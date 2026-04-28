#!/bin/sh
# Auto-detect TLS and tokens, then build production image
# set -e

# Set default WEIGHTSLAB_CERTS_DIR (can be overridden by environment variable)
WEIGHTSLAB_CERTS_DIR="${WEIGHTSLAB_CERTS_DIR:-${HOME}/.weightslab-certs}"

# Parse command line arguments
DEV=false
FORCE_UNSECURE=0

for arg in "$@"; do
  case "$arg" in
    --dev)
      DEV=true
      ;;
    --unsecure|--unsecured)
      echo "⚠ Forcing UNSECURE mode (HTTP, no auth)"
      FORCE_UNSECURE=1
      ;;
  esac
done

# Check if WEIGHTSLAB_CERTS_DIR was explicitly set to empty string (before applying defaults)
# This is how the E2E tests disable certs: WEIGHTSLAB_CERTS_DIR=""
if [ "$FORCE_UNSECURE" = "1" ]; then
    echo "⚠ --unsecured flag provided: disabling certs and auth"
    WEIGHTSLAB_CERTS_DIR=""
elif [ "${WEIGHTSLAB_CERTS_DIR+x}" ] && [ -z "$WEIGHTSLAB_CERTS_DIR" ]; then
    echo "⚠ WEIGHTSLAB_CERTS_DIR explicitly set to empty - forcing UNSECURE mode"
    FORCE_UNSECURE=1
else
    # Respect WEIGHTSLAB_CERTS_DIR from environment variable, with fallback to home directory
    WEIGHTSLAB_CERTS_DIR="${WEIGHTSLAB_CERTS_DIR:-${HOME}/.weightslab-certs}"
fi

TOKEN_FILE="${WEIGHTSLAB_CERTS_DIR}/.grpc_auth_token"

echo "DEBUG: HOME=$HOME"
echo "DEBUG: WEIGHTSLAB_CERTS_DIR='$WEIGHTSLAB_CERTS_DIR' (empty: $([ -z "$WEIGHTSLAB_CERTS_DIR" ] && echo "yes" || echo "no"))"
echo "DEBUG: FORCE_UNSECURE=$FORCE_UNSECURE"
echo "DEBUG: WEIGHTSLAB_CERTS_DIR env var was set: $([ "${WEIGHTSLAB_CERTS_DIR+x}" ] && echo "yes" || echo "no")"
if [ -n "$WEIGHTSLAB_CERTS_DIR" ]; then
    echo "DEBUG: Checking for certs..."
    ls -la "$WEIGHTSLAB_CERTS_DIR" 2>/dev/null || echo "DEBUG: WEIGHTSLAB_CERTS_DIR path does not exist or is empty"
else
    echo "DEBUG: WEIGHTSLAB_CERTS_DIR is empty, skipping cert check"
fi

# Respect environment variables if explicitly set (for E2E tests)
VITE_DEV_SERVER_HTTPS="${VITE_DEV_SERVER_HTTPS:-unset}"
VITE_SERVER_PROTOCOL="${VITE_SERVER_PROTOCOL:-unset}"
ENVOY_UPSTREAM_TLS="${ENVOY_UPSTREAM_TLS:-unset}"
ENVOY_DOWNSTREAM_TLS="${ENVOY_DOWNSTREAM_TLS:-unset}"

# Detect TLS (unless overridden by environment)
if [ "$FORCE_UNSECURE" = "1" ]; then
    echo "✗ UNSECURE mode: HTTP (no TLS)"
    VITE_DEV_SERVER_HTTPS=0
    VITE_SERVER_PROTOCOL=http
    ENVOY_UPSTREAM_TLS=off
    ENVOY_DOWNSTREAM_TLS=off
elif [ "$VITE_SERVER_PROTOCOL" != "unset" ]; then
    # Environment variables are already set, use them
    echo "ℹ Using environment-provided protocol settings"
    [ "$VITE_DEV_SERVER_HTTPS" = "unset" ] && VITE_DEV_SERVER_HTTPS=0
    [ "$ENVOY_UPSTREAM_TLS" = "unset" ] && ENVOY_UPSTREAM_TLS=off
    [ "$ENVOY_DOWNSTREAM_TLS" = "unset" ] && ENVOY_DOWNSTREAM_TLS=off
elif [ -f "$WEIGHTSLAB_CERTS_DIR/envoy-server.crt" ] && [ -f "$WEIGHTSLAB_CERTS_DIR/envoy-server.key" ]; then
    echo "✓ TLS certificates found - building with HTTPS support"
    VITE_DEV_SERVER_HTTPS=1
    VITE_SERVER_PROTOCOL=https
    ENVOY_UPSTREAM_TLS=on
    ENVOY_DOWNSTREAM_TLS=on
else
    echo "✗ TLS certificates not found - building for HTTP mode"
    VITE_DEV_SERVER_HTTPS=0
    VITE_SERVER_PROTOCOL=http
    ENVOY_UPSTREAM_TLS=off
    ENVOY_DOWNSTREAM_TLS=off
fi

# Detect gRPC token
VITE_WL_ENABLE_GRPC_AUTH_TOKEN="${VITE_WL_ENABLE_GRPC_AUTH_TOKEN:-unset}"
VITE_GRPC_AUTH_TOKEN="${VITE_GRPC_AUTH_TOKEN:-}"

if [ "$FORCE_UNSECURE" = "1" ]; then
    echo "✗ UNSECURE mode: auth disabled"
    VITE_WL_ENABLE_GRPC_AUTH_TOKEN=0
    VITE_GRPC_AUTH_TOKEN=""
elif [ "$VITE_WL_ENABLE_GRPC_AUTH_TOKEN" != "unset" ]; then
    # Environment variable is already set, use it
    echo "ℹ Using environment-provided auth settings"
    true
elif [ -f "$TOKEN_FILE" ]; then
    echo "✓ gRPC token found - enabling auth"
    VITE_WL_ENABLE_GRPC_AUTH_TOKEN=1
    VITE_GRPC_AUTH_TOKEN=$(cat "$TOKEN_FILE")
else
    echo "✗ gRPC token not found - auth disabled"
    VITE_WL_ENABLE_GRPC_AUTH_TOKEN=0
    VITE_GRPC_AUTH_TOKEN=""
fi

# Write environment variables to .env file for docker compose
echo "Writing environment variables to .env..."
# Note: .env must be in the docker directory where docker-compose.yml is located
ENV_FILE="../.env"

# Convert WEIGHTSLAB_CERTS_DIR to absolute path for docker-compose compatibility
if [ -n "$WEIGHTSLAB_CERTS_DIR" ] && [ ! -z "$WEIGHTSLAB_CERTS_DIR" ]; then
    # Resolve to absolute path
    WEIGHTSLAB_CERTS_DIR_ABSOLUTE="$(cd "$WEIGHTSLAB_CERTS_DIR" 2>/dev/null && pwd)" || WEIGHTSLAB_CERTS_DIR_ABSOLUTE="$WEIGHTSLAB_CERTS_DIR"
fi

if [ -z "$WEIGHTSLAB_CERTS_DIR" ]; then
    # Unsecured mode: explicitly disable all TLS and auth variables
    cat > "$ENV_FILE" << EOF
VITE_DEV_SERVER_HTTPS=0
VITE_SERVER_PROTOCOL=http
VITE_WL_ENABLE_GRPC_AUTH_TOKEN=0
VITE_GRPC_AUTH_TOKEN=
WEIGHTSLAB_CERTS_DIR=
ENVOY_UPSTREAM_TLS=off
ENVOY_DOWNSTREAM_TLS=off
WS_SERVER_PROTOCOL=http
GRPC_BACKEND_PORT=${GRPC_BACKEND_PORT:-50051}
EOF
else
    # Secured mode: include certs
    cat > "$ENV_FILE" << EOF
VITE_DEV_SERVER_HTTPS=$VITE_DEV_SERVER_HTTPS
VITE_SERVER_PROTOCOL=$VITE_SERVER_PROTOCOL
VITE_WL_ENABLE_GRPC_AUTH_TOKEN=$VITE_WL_ENABLE_GRPC_AUTH_TOKEN
VITE_GRPC_AUTH_TOKEN=$VITE_GRPC_AUTH_TOKEN
WEIGHTSLAB_CERTS_DIR=$WEIGHTSLAB_CERTS_DIR_ABSOLUTE
ENVOY_UPSTREAM_TLS=$ENVOY_UPSTREAM_TLS
ENVOY_DOWNSTREAM_TLS=$ENVOY_DOWNSTREAM_TLS
WS_SERVER_PROTOCOL=$VITE_SERVER_PROTOCOL
GRPC_BACKEND_PORT=${GRPC_BACKEND_PORT:-50051}
EOF
fi
echo "✓ .env file written to $ENV_FILE"
cat "$ENV_FILE"

# Check if image already exists
IMAGE_NAME="weights_studio_frontend"
if docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo "ℹ Image '$IMAGE_NAME' already exists - skipping build"
    SKIP_BUILD=true
else
    echo "ℹ Image '$IMAGE_NAME' not found - will build"
    SKIP_BUILD=false
fi

# Build and deploy
if [ "$DEV" = "true" ]; then
    if [ "$SKIP_BUILD" = "false" ]; then
        echo "Building development image (single image, configuration at runtime)..."
        docker compose -f ../docker-compose.yml -f ../docker-compose.dev.yml build

        echo "✓ Dev build complete!"
    else
        echo "✓ Skipped dev build (image already exists)"
    fi

    # Deploy (docker compose automatically reads .env)
    echo "Deploying containers..."
    docker compose -f ../docker-compose.yml -f ../docker-compose.dev.yml down
    docker compose -f ../docker-compose.yml -f ../docker-compose.dev.yml up -d --force-recreate

    echo "✓ Deployment to development complete!"
else
    if [ "$SKIP_BUILD" = "false" ]; then
        echo "Building production image (single image, configuration at runtime)..."
        # Build with defaults - configuration happens at runtime via .env
        docker compose -f ../docker-compose.yml -f ../docker-compose.prod.yml build

        BUILD_STATUS=$?
        if [ $BUILD_STATUS -ne 0 ]; then
            echo "✗ Production build failed with status $BUILD_STATUS"
            exit $BUILD_STATUS
        fi

        echo "✓ Production build complete!"
    else
        echo "✓ Skipped production build (image already exists)"
    fi

    # Deploy (docker compose automatically reads .env)
    echo "Deploying containers..."
    echo "ℹ Stopping existing containers..."
    docker compose -f ../docker-compose.yml -f ../docker-compose.prod.yml down

    echo "ℹ Starting containers..."
    docker compose -f ../docker-compose.yml -f ../docker-compose.prod.yml up -d --force-recreate

    UP_STATUS=$?
    if [ $UP_STATUS -ne 0 ]; then
        echo "✗ Container startup failed with status $UP_STATUS"
        echo "ℹ Checking container logs..."
        docker compose -f ../docker-compose.yml -f ../docker-compose.prod.yml logs --tail=50 || true
        exit $UP_STATUS
    fi

    echo "✓ Deployment to production complete!"
    echo "ℹ Running containers:"
    docker compose -f ../docker-compose.yml -f ../docker-compose.prod.yml ps || true
fi
