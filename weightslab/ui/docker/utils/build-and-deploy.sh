#!/bin/bash
# Auto-detect TLS and tokens, then build production image

# Print all environment variables to diagnose issues
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
      echo "Forcing UNSECURE mode (HTTP, no auth)"
      FORCE_UNSECURE=1
      ;;
  esac
done

# Check if WEIGHTSLAB_CERTS_DIR was explicitly set to empty string (before applying defaults)
# This is how the E2E tests disable certs: WEIGHTSLAB_CERTS_DIR=""
if [ "$FORCE_UNSECURE" = "1" ]; then
    # Force HTTP / no auth, but KEEP WEIGHTSLAB_CERTS_DIR: the compose bind-mount
    # needs a real (possibly empty) source directory. TLS/auth are forced off in
    # the derivation block below regardless of any files in it.
    echo "--unsecured flag provided: forcing HTTP / no auth (certs dir kept for bind-mount only)"
elif [ "${WEIGHTSLAB_CERTS_DIR+x}" ] && [ -z "$WEIGHTSLAB_CERTS_DIR" ]; then
    echo "WEIGHTSLAB_CERTS_DIR explicitly set to empty - forcing UNSECURE mode"
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
            echo "Found converted path at '$CONVERTED_PATH'"
            WEIGHTSLAB_CERTS_DIR="$CONVERTED_PATH"
        else
            echo "Converted path not found at '$CONVERTED_PATH'"
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
if [ -n "$WEIGHTSLAB_CERTS_DIR" ]; then
    ls -la "$WEIGHTSLAB_CERTS_DIR" 2>/dev/null || echo "WEIGHTSLAB_CERTS_DIR path does not exist or is empty"
else
    echo "WEIGHTSLAB_CERTS_DIR is empty, skipping cert check"
fi

# ---------------------------------------------------------------------------
# Single source of truth: TLS + gRPC auth are derived SOLELY from the presence
# of cert files in WEIGHTSLAB_CERTS_DIR. Any inherited VITE_*/ENVOY_* values are
# intentionally ignored and recomputed here, so a stale or pre-set env var can
# never force HTTPS without certs (which would crash Envoy on an empty mount) or
# vice versa. `--unsecure` (or an empty WEIGHTSLAB_CERTS_DIR) forces everything
# off.
# ---------------------------------------------------------------------------
if [ "$FORCE_UNSECURE" = "1" ] || [ -z "$WEIGHTSLAB_CERTS_DIR" ]; then
    echo "Unsecured mode: HTTP, no auth (no certs directory)"
    VITE_DEV_SERVER_HTTPS=0
    VITE_SERVER_PROTOCOL=http
    ENVOY_UPSTREAM_TLS=off
    ENVOY_DOWNSTREAM_TLS=off
    VITE_WL_ENABLE_GRPC_AUTH_TOKEN=0
    VITE_GRPC_AUTH_TOKEN=""
else
    # Downstream HTTPS (browser <-> Envoy): needs the Envoy server cert + key.
    if [ -f "$WEIGHTSLAB_CERTS_DIR/envoy-server.crt" ] && [ -f "$WEIGHTSLAB_CERTS_DIR/envoy-server.key" ]; then
        echo "Envoy server cert found in $WEIGHTSLAB_CERTS_DIR - enabling HTTPS"
        VITE_DEV_SERVER_HTTPS=1
        VITE_SERVER_PROTOCOL=https
        ENVOY_DOWNSTREAM_TLS=on
    else
        echo "No Envoy server cert in $WEIGHTSLAB_CERTS_DIR - HTTP (no downstream TLS)"
        VITE_DEV_SERVER_HTTPS=0
        VITE_SERVER_PROTOCOL=http
        ENVOY_DOWNSTREAM_TLS=off
    fi

    # Upstream mTLS (Envoy <-> backend gRPC): needs the Envoy client cert + CA.
    if [ -f "$WEIGHTSLAB_CERTS_DIR/envoy-client.crt" ] && [ -f "$WEIGHTSLAB_CERTS_DIR/envoy-client.key" ] && [ -f "$WEIGHTSLAB_CERTS_DIR/ca.crt" ]; then
        echo "Envoy client cert + CA found - enabling upstream TLS"
        ENVOY_UPSTREAM_TLS=on
    else
        echo "No Envoy client cert/CA - upstream plaintext"
        ENVOY_UPSTREAM_TLS=off
    fi

    # gRPC auth token.
    if [ -f "$TOKEN_FILE" ]; then
        echo "gRPC token found - enabling auth"
        VITE_WL_ENABLE_GRPC_AUTH_TOKEN=1
        VITE_GRPC_AUTH_TOKEN=$(cat "$TOKEN_FILE")
    else
        echo "gRPC token not found - auth disabled"
        VITE_WL_ENABLE_GRPC_AUTH_TOKEN=0
        VITE_GRPC_AUTH_TOKEN=""
    fi
fi

# Export all derived variables for docker compose
export VITE_DEV_SERVER_HTTPS
export VITE_SERVER_PROTOCOL
export ENVOY_UPSTREAM_TLS
export ENVOY_DOWNSTREAM_TLS
export VITE_WL_ENABLE_GRPC_AUTH_TOKEN
export VITE_GRPC_AUTH_TOKEN

# Get weightslab root from environment variable or derive from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "$WEIGHTSLAB_ROOT" ]; then
    # WEIGHTSLAB_ROOT is explicitly set - use it
    echo "Using WEIGHTSLAB_ROOT from environment: $WEIGHTSLAB_ROOT"
    DOCKER_DIR="$WEIGHTSLAB_ROOT/weightslab/ui/docker"
else
    # Default: derive from script location (script is at weightslab/ui/docker/utils/)
    DOCKER_DIR="$(dirname "$SCRIPT_DIR")"
fi

ENV_FILE="$DOCKER_DIR/.env"
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "DOCKER_DIR: $DOCKER_DIR"

# Write environment variables to .env file for docker compose
echo "Writing environment variables to .env..."
# Note: .env must be in the docker directory where docker-compose.yml is located

# Convert WEIGHTSLAB_CERTS_DIR to absolute path for docker-compose compatibility
if [ -n "$WEIGHTSLAB_CERTS_DIR" ] && [ ! -z "$WEIGHTSLAB_CERTS_DIR" ]; then
    # Resolve to absolute path
    WEIGHTSLAB_CERTS_DIR_ABSOLUTE="$(cd "$WEIGHTSLAB_CERTS_DIR" 2>/dev/null && pwd)" || WEIGHTSLAB_CERTS_DIR_ABSOLUTE="$WEIGHTSLAB_CERTS_DIR"
fi

# Path written into .env for the docker compose bind mount. Prefer the
# host-native path provided by the Python launcher (e.g. C:/Users/... on
# Windows/Docker Desktop): the Unix path above (/mnt/c/...) is only valid for
# this script's own file checks, NOT for a Docker Desktop bind mount, which
# silently mounts an empty dir for /mnt paths and crashes Envoy on missing certs.
CERTS_DIR_FOR_ENV="${WEIGHTSLAB_CERTS_DIR_HOST:-$WEIGHTSLAB_CERTS_DIR_ABSOLUTE}"

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
WEIGHTSLAB_CERTS_DIR=$CERTS_DIR_FOR_ENV
ENVOY_UPSTREAM_TLS=$ENVOY_UPSTREAM_TLS
ENVOY_DOWNSTREAM_TLS=$ENVOY_DOWNSTREAM_TLS
WS_SERVER_PROTOCOL=$VITE_SERVER_PROTOCOL
GRPC_BACKEND_PORT=${GRPC_BACKEND_PORT:-50051}
EOF
fi
echo ".env file written to $ENV_FILE"
cat "$ENV_FILE"

# Convert WEIGHTSLAB_CERTS_DIR to absolute path for docker-compose compatibility
if [ -n "$WEIGHTSLAB_CERTS_DIR" ] && [ ! -z "$WEIGHTSLAB_CERTS_DIR" ]; then
    # Resolve to absolute path
    WEIGHTSLAB_CERTS_DIR_ABSOLUTE="$(cd "$WEIGHTSLAB_CERTS_DIR" 2>/dev/null && pwd)" || WEIGHTSLAB_CERTS_DIR_ABSOLUTE="$WEIGHTSLAB_CERTS_DIR"
fi

# All cert/token file checks above used the Unix path (e.g. /mnt/c/...) so they
# resolve under this shell. But docker compose gives an exported shell env var
# precedence over the .env file, and Docker Desktop cannot bind-mount a
# /mnt-style source (it silently mounts an empty dir, crashing Envoy on missing
# certs). Re-export the host-native path so the compose bind mount matches .env.
if [ -n "$CERTS_DIR_FOR_ENV" ]; then
    export WEIGHTSLAB_CERTS_DIR="$CERTS_DIR_FOR_ENV"
    echo "Exporting host-native WEIGHTSLAB_CERTS_DIR for docker compose: $WEIGHTSLAB_CERTS_DIR"
fi

# Check if image already exists
IMAGE_NAME="graybx/weightslab"
# if docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
#     echo "Image '$IMAGE_NAME' already exists - skipping build"
#     SKIP_BUILD=true
# else
#     echo "Image '$IMAGE_NAME' not found - will build"
#     SKIP_BUILD=false
# fi
SKIP_BUILD=false

# When invoked from the Python launcher on a host where docker runs outside this
# shell (e.g. Windows: docker lives on the host, not in WSL), the launcher does
# `docker compose pull/up` itself. In that case this script only needs to write
# the .env above — skip all docker operations here to avoid noisy failures.
if [ -n "$WEIGHTSLAB_SKIP_DOCKER_OPS" ] && [ "$WEIGHTSLAB_SKIP_DOCKER_OPS" != "0" ]; then
    echo "Skipping docker build/deploy in shell (handled by the launcher)."
    exit 0
fi

# Build and deploy
if [ "$DEV" = "true" ]; then
    echo "Skipped dev build (image already exists)"
    # Deploy (docker compose automatically reads .env)
    echo "Deploying containers..."
    docker compose -f $DOCKER_DIR/docker-compose.yml -f $DOCKER_DIR/docker-compose.dev.yml down
    docker compose -f $DOCKER_DIR/docker-compose.yml -f $DOCKER_DIR/docker-compose.dev.yml up -d --force-recreate

    echo "Deployment to development complete!"
else
    if [ "$SKIP_BUILD" = "false" ]; then
        echo "Building production image (single image, configuration at runtime)..."
        # Build with defaults - configuration happens at runtime via .env
        docker compose -f $DOCKER_DIR/docker-compose.yml build --no-cache

        BUILD_STATUS=$?
        if [ $BUILD_STATUS -ne 0 ]; then
            echo "Production build failed with status $BUILD_STATUS"
            exit $BUILD_STATUS
        fi

        echo "Production build complete!"
    else
        echo "Skipped production build (image already exists)"
    fi

    # Deploy (docker compose automatically reads .env)
    echo "Deploying containers..."
    echo "Stopping existing containers..."
    docker compose -f $DOCKER_DIR/docker-compose.yml down

    echo "Starting containers..."
    docker compose -f $DOCKER_DIR/docker-compose.yml up -d --force-recreate

    UP_STATUS=$?
    if [ $UP_STATUS -ne 0 ]; then
        echo "Container startup failed with status $UP_STATUS"
        echo "Checking container logs..."
        docker compose -f $DOCKER_DIR/docker-compose.yml logs --tail=50 || true
        exit $UP_STATUS
    fi

    echo "Deployment to production complete!"
    echo "Running containers:"
    docker compose -f $DOCKER_DIR/docker-compose.yml ps || true
fi
