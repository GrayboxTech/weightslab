#!/bin/sh
# Auto-detect TLS and tokens, then build production image
# set -e

# DEBUG: Print all environment variables to diagnose issues
echo "===== Environment Variables Received ====="
env | grep -E 'WEIGHTSLAB|VITE|ENVOY|HOME' || true
echo "=========================================="

# Ensure environment variables are available (important when called from Python subprocess)
echo "Init: WEIGHTSLAB_CERTS_DIR='$WEIGHTSLAB_CERTS_DIR' (received from env)"
WEIGHTSLAB_CERTS_DIR="${WEIGHTSLAB_CERTS_DIR:-$HOME/.weightslab-certs}"
export WEIGHTSLAB_CERTS_DIR
echo "After default: WEIGHTSLAB_CERTS_DIR='$WEIGHTSLAB_CERTS_DIR'"

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
elif [ ! -d "$WEIGHTSLAB_CERTS_DIR" ]; then
    # WEIGHTSLAB_CERTS_DIR doesn't exist, try converting Windows path to Unix-style
    CONVERTED_PATH=""
    if echo "$WEIGHTSLAB_CERTS_DIR" | grep -q '\\'; then
        # Path contains backslashes - likely Windows path, convert to Unix-style (for Git Bash)
        # Convert C:\path\to\dir -> /c/path/to/dir
        CONVERTED_PATH=$(echo "$WEIGHTSLAB_CERTS_DIR" | sed 's/^\([A-Za-z]\):\\/\/\L\1\//; s/\\/\//g')
        echo "Detected Windows path, converting to Unix-style: $CONVERTED_PATH"

        if [ -d "$CONVERTED_PATH" ]; then
            echo "✓ Found converted path at '$CONVERTED_PATH'"
            WEIGHTSLAB_CERTS_DIR="$CONVERTED_PATH"
        else
            echo "✗ Converted path not found at '$CONVERTED_PATH'"
            # Continue with next fallback
            WEIGHTSLAB_CERTS_DIR=""
        fi
    else
        # Not a Windows path, try default ~/.weightslab-certs
        DEFAULT_CERTS_DIR="$HOME/.weightslab-certs"
        if [ -d "$DEFAULT_CERTS_DIR" ]; then
            echo "WEIGHTSLAB_CERTS_DIR not found at '$WEIGHTSLAB_CERTS_DIR', using default: $DEFAULT_CERTS_DIR"
            WEIGHTSLAB_CERTS_DIR="$DEFAULT_CERTS_DIR"
        else
            echo "Default certs directory not found ($DEFAULT_CERTS_DIR), continuing without certs"
            WEIGHTSLAB_CERTS_DIR=""
        fi
    fi
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
    echo "Using environment-provided protocol settings"
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

# Export all environment variables for docker compose
export VITE_DEV_SERVER_HTTPS
export VITE_SERVER_PROTOCOL
export ENVOY_UPSTREAM_TLS
export ENVOY_DOWNSTREAM_TLS

# Detect gRPC token
VITE_WL_ENABLE_GRPC_AUTH_TOKEN="${VITE_WL_ENABLE_GRPC_AUTH_TOKEN:-unset}"
VITE_GRPC_AUTH_TOKEN="${VITE_GRPC_AUTH_TOKEN:-}"

if [ "$FORCE_UNSECURE" = "1" ]; then
    echo "✗ UNSECURE mode: auth disabled"
    VITE_WL_ENABLE_GRPC_AUTH_TOKEN=0
    VITE_GRPC_AUTH_TOKEN=""
elif [ "$VITE_WL_ENABLE_GRPC_AUTH_TOKEN" != "unset" ]; then
    # Environment variable is already set, use it
    echo "Using environment-provided auth settings"
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

# Export auth variables for docker compose
export VITE_WL_ENABLE_GRPC_AUTH_TOKEN
export VITE_GRPC_AUTH_TOKEN

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$DOCKER_DIR/.env"

# Write environment variables to .env file for docker compose
echo "Writing environment variables to .env..."
# Note: .env must be in the docker directory where docker-compose.yml is located

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

# Convert WEIGHTSLAB_CERTS_DIR to absolute path for docker-compose compatibility
if [ -n "$WEIGHTSLAB_CERTS_DIR" ] && [ ! -z "$WEIGHTSLAB_CERTS_DIR" ]; then
    # Resolve to absolute path
    WEIGHTSLAB_CERTS_DIR_ABSOLUTE="$(cd "$WEIGHTSLAB_CERTS_DIR" 2>/dev/null && pwd)" || WEIGHTSLAB_CERTS_DIR_ABSOLUTE="$WEIGHTSLAB_CERTS_DIR"
fi

# Check if image already exists
IMAGE_NAME="graybx/weightslab"
if docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo "Image '$IMAGE_NAME' already exists - skipping build"
    SKIP_BUILD=true
else
    echo "Image '$IMAGE_NAME' not found - will build"
    SKIP_BUILD=false
fi

# Build and deploy
if [ "$DEV" = "true" ]; then
    if [ "$SKIP_BUILD" = "false" ]; then
        echo "Building development image (single image, configuration at runtime)..."
        docker compose -f $DOCKER_DIR/docker-compose.yml -f $DOCKER_DIR/docker-compose.dev.yml build

        echo "✓ Dev build complete!"
    else
        echo "✓ Skipped dev build (image already exists)"
    fi

    # Deploy (docker compose automatically reads .env)
    echo "Deploying containers..."
    docker compose -f $DOCKER_DIR/docker-compose.yml -f $DOCKER_DIR/docker-compose.dev.yml down
    docker compose -f $DOCKER_DIR/docker-compose.yml -f $DOCKER_DIR/docker-compose.dev.yml up -d --force-recreate

    echo "✓ Deployment to development complete!"
else
    if [ "$SKIP_BUILD" = "false" ]; then
        echo "Building production image (single image, configuration at runtime)..."
        # Build with defaults - configuration happens at runtime via .env
        docker compose -f $DOCKER_DIR/docker-compose.yml build

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
    echo "Stopping existing containers..."
    docker compose -f $DOCKER_DIR/docker-compose.yml down

    echo "Starting containers..."
    docker compose -f $DOCKER_DIR/docker-compose.yml up -d --force-recreate

    UP_STATUS=$?
    if [ $UP_STATUS -ne 0 ]; then
        echo "✗ Container startup failed with status $UP_STATUS"
        echo "Checking container logs..."
        docker compose -f $DOCKER_DIR/docker-compose.yml logs --tail=50 || true
        exit $UP_STATUS
    fi

    echo "✓ Deployment to production complete!"
    echo "Running containers:"
    docker compose -f $DOCKER_DIR/docker-compose.yml ps || true
fi
