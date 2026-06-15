#!/usr/bin/env bash
# =============================================================================
# DinD entrypoint: start the inner docker daemon, launch the UI stack, then run
# the classification example in the foreground (it serves the gRPC backend).
# =============================================================================
set -euo pipefail

# Set WEIGHTSLAB_TLS=1 (via the trainer service environment) for HTTPS + mTLS.
WEIGHTSLAB_TLS="${WEIGHTSLAB_TLS:-0}"
# Single source of truth for certs — ui-launch generates here, the backend reads here.
export WEIGHTSLAB_CERTS_DIR="${WEIGHTSLAB_CERTS_DIR:-$HOME/.weightslab-certs}"

# --- GPU visibility check (best-effort) --------------------------------------
# Show the GPU the container can see. Non-fatal: on a CPU-only host (or one
# without the NVIDIA Container Toolkit) this just prints a notice and training
# falls back to CPU (config device: auto).
echo "[entrypoint] Checking GPU visibility (nvidia-smi)..."
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    nvidia-smi -L
else
    echo "[entrypoint] No NVIDIA GPU visible (nvidia-smi unavailable) — running on CPU."
fi

echo "[entrypoint] Starting inner docker daemon (DinD)..."
# The inner daemon manages the Envoy + frontend containers. It needs --privileged
# (set on the trainer service in docker-compose.yml).
dockerd >/var/log/dockerd.log 2>&1 &

echo "[entrypoint] Waiting for inner docker daemon to become ready..."
for _ in $(seq 1 60); do
    if docker info >/dev/null 2>&1; then
        break
    fi
    sleep 1
done
if ! docker info >/dev/null 2>&1; then
    echo "[entrypoint] ERROR: inner dockerd did not start. Last log lines:" >&2
    tail -n 40 /var/log/dockerd.log >&2 || true
    exit 1
fi
echo "[entrypoint] Inner docker daemon is up."

# The bundled deploy script would otherwise run a full 'docker compose build
# --no-cache' of the frontend. We only need the published image, so tell it to
# just write the .env and skip docker ops — 'ui launch' then pulls + starts the
# stack itself via 'docker compose up --pull always'.
export WEIGHTSLAB_SKIP_DOCKER_OPS=1

if [ "${WEIGHTSLAB_TLS}" = "1" ]; then
    echo "[entrypoint] TLS mode — launching SECURED UI (HTTPS + mTLS) into ${WEIGHTSLAB_CERTS_DIR}..."
    weightslab ui launch --certs
    # --certs only configures Envoy + the frontend. The backend gRPC server needs
    # TLS turned on too, or Envoy's upstream mTLS handshake fails (503s). Point it
    # at the same cert dir (holds backend-server.crt/key + ca.crt).
    export GRPC_TLS_ENABLED=1
    export GRPC_TLS_CERT_DIR="${WEIGHTSLAB_CERTS_DIR}"
    SCHEME="https"
else
    echo "[entrypoint] HTTP mode — launching UNSECURED UI..."
    weightslab ui launch
    SCHEME="http"
fi

echo "[entrypoint] Starting the classification (cls) example — serves gRPC on :${GRPC_BACKEND_PORT:-50051}."
echo "[entrypoint] Open the UI at ${SCHEME}://localhost:5173 once training is serving."
# Foreground: keeps the container alive while the backend serves the browser.
exec weightslab start example  # --3d_det
