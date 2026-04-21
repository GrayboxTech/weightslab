#!/usr/bin/env bash
set -euo pipefail

# Parse command line arguments
NO_AUTH_TOKEN=false
FORCE_CREATE_CERTS=false
NO_CERT=false
DEV=false

for arg in "$@"; do
  case "$arg" in
    --no-auth-token)
      NO_AUTH_TOKEN=true
      ;;
    --force-create-certs)
      FORCE_CREATE_CERTS=true
      ;;
    --no-cert)
      NO_CERT=true
      ;;
    --dev)
      DEV=true
      ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is required but was not found in PATH." >&2
  exit 1
fi

if [ -z "${SCRIPT_DIR:-}" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
DOCKER_UTILS_DIR="${SCRIPT_DIR}"
DOCKER_DIR="$(cd "${DOCKER_UTILS_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${DOCKER_DIR}/.." && pwd)"

if [ "$NO_CERT" = false ]; then
  echo "[1/3] Generating development certificates..."
  CERT_ARGS=""
  if [ "$FORCE_CREATE_CERTS" = true ]; then
    CERT_ARGS="--force-create-certs"
  fi
  "${SCRIPT_DIR}/generate-certs.sh" $CERT_ARGS
else
  echo "[1/3] Skipping certificate generation (--no-cert flag set)..."
fi

echo "[2/3] Exporting secure backend defaults for current shell..."
export WEIGHTSLAB_CERTS_DIR="${HOME}/.weightslab-certs"
export GRPC_TLS_ENABLED=1
export GRPC_TLS_REQUIRE_CLIENT_AUTH=1
export GRPC_TLS_CERT_FILE="${WEIGHTSLAB_CERTS_DIR}/backend-server.crt"
export GRPC_TLS_KEY_FILE="${WEIGHTSLAB_CERTS_DIR}/backend-server.key"
export GRPC_TLS_CA_FILE="${WEIGHTSLAB_CERTS_DIR}/ca.crt"
export ENVOY_DOWNSTREAM_TLS=on
export ENVOY_UPSTREAM_TLS=on
export WS_SERVER_PROTOCOL=https
export VITE_SERVER_PROTOCOL=https
export VITE_DEV_SERVER_HTTPS=1
export VITE_DEV_SERVER_CERT_FILE=/app/envoy/certs/envoy-server.crt
export VITE_DEV_SERVER_KEY_FILE=/app/envoy/certs/envoy-server.key

# Set WL_ENABLE_GRPC_AUTH_TOKEN based on --no-auth-token flag (default enabled)
if [ "$NO_AUTH_TOKEN" = true ]; then
  export WL_ENABLE_GRPC_AUTH_TOKEN=0
  export VITE_WL_ENABLE_GRPC_AUTH_TOKEN=0
else
  export WL_ENABLE_GRPC_AUTH_TOKEN=1
  export VITE_WL_ENABLE_GRPC_AUTH_TOKEN=1
fi

if [ "${WL_ENABLE_GRPC_AUTH_TOKEN:-0}" = "1" ]; then
  if [ -z "${GRPC_AUTH_TOKEN:-}" ]; then
    # SHA-256 hex token generated from CSPRNG bytes.
    GRPC_AUTH_TOKEN="$(head -c 32 /dev/urandom | sha256sum | awk '{print $1}')"
    export GRPC_AUTH_TOKEN
    echo "Generated GRPC_AUTH_TOKEN for this shell session."
  else
    echo "Using existing GRPC_AUTH_TOKEN from current shell session."
  fi
  export VITE_GRPC_AUTH_TOKEN="${GRPC_AUTH_TOKEN}"
else
  unset GRPC_AUTH_TOKEN || true
  unset VITE_GRPC_AUTH_TOKEN || true
  echo "GRPC auth token disabled (--no-auth-token flag set)."
fi

if [ "$NO_CERT" = false ]; then
  echo "[3/3] Starting secured Envoy + frontend stack..."
  if [ "$DEV" = true ]; then
    docker compose -f "${DOCKER_DIR}/docker-compose.yml" -f "${DOCKER_DIR}/docker-compose.dev.yml" up -d --force-recreate envoy
    docker compose -f "${DOCKER_DIR}/docker-compose.yml" -f "${DOCKER_DIR}/docker-compose.dev.yml" up -d --force-recreate weights_studio
    echo "Secure dev stack is up."
  else
    docker compose -f "${DOCKER_DIR}/docker-compose.yml" -f "${DOCKER_DIR}/docker-compose.prod.yml" up -d --force-recreate envoy
    docker compose -f "${DOCKER_DIR}/docker-compose.yml" -f "${DOCKER_DIR}/docker-compose.prod.yml" up -d --force-recreate weights_studio
    echo "Secure prod stack is up."
  fi
else
  echo "[3/3] Skipping stack startup (--no-cert flag set, only updating auth token)..."
fi

echo "Backend TLS env exported in this shell. Start backend from this shell to use them."
if [ "${WL_ENABLE_GRPC_AUTH_TOKEN:-0}" = "1" ]; then
  echo "GRPC_AUTH_TOKEN is set in this shell for backend gRPC auth."
fi

echo ""
echo "Available flags:"
echo "  --no-auth-token          Disable gRPC auth token"
echo "  --force-create-certs     Recreate certificates even if they exist"
echo "  --no-cert                Skip certificate generation/copying, only update auth token"
echo "  --dev                    Use dev docker-compose (default: prod)"
