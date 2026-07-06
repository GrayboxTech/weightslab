#!/usr/bin/env bash
# =============================================================================
# Self-contained siblings entrypoint — runs entirely from inside the container,
# with NO host bind mounts and NO path alignment.
#
# Flow:
#   1. render Envoy's plaintext config from the installed weightslab package,
#   2. push it into a named volume *over the docker socket* (no host path),
#   3. bring up Envoy + frontend (ui-compose.yml) on the host daemon,
#   4. run the example (serves the gRPC backend, published to the host).
# =============================================================================
set -euo pipefail

GRPC_BACKEND_PORT="${GRPC_BACKEND_PORT:-50051}"
ENVOY_VOLUME="wl_envoy_cfg"
NGINX_CERTS_VOLUME="wl_nginx_certs"
UI_COMPOSE="/opt/ui/ui-compose.yml"
UI_PROJECT="weightslab_ui"

# Set WEIGHTSLAB_TLS=1 (via the trainer service environment) for HTTPS + mTLS.
WEIGHTSLAB_TLS="${WEIGHTSLAB_TLS:-0}"
export WEIGHTSLAB_CERTS_DIR="${WEIGHTSLAB_CERTS_DIR:-$HOME/.weightslab-certs}"

# --- GPU visibility check (best-effort) --------------------------------------
echo "[entrypoint] Checking GPU visibility (nvidia-smi)..."
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    nvidia-smi -L
else
    echo "[entrypoint] No NVIDIA GPU visible (nvidia-smi unavailable) — running on CPU."
fi

# --- Reach the host docker daemon --------------------------------------------
if ! docker info >/dev/null 2>&1; then
    echo "[entrypoint] ERROR: cannot reach the host docker daemon." >&2
    echo "             Did you mount /var/run/docker.sock? (see docker-compose.yml)" >&2
    exit 1
fi

# Locate the installed package's Envoy templates. find_spec does NOT execute
# weightslab/__init__.py, so it avoids the import-time logging banner that would
# otherwise pollute this command substitution.
PKG="$(python -c 'import importlib.util as u; print(u.find_spec("weightslab").submodule_search_locations[0])')"

# --- 0) Start from clean named volumes ---------------------------------------
# Tear down any prior UI stack first so the volumes are free to recreate.
docker compose -p "${UI_PROJECT}" -f "${UI_COMPOSE}" down --remove-orphans >/dev/null 2>&1 || true
docker rm -f weights_studio_envoy weights_studio_frontend >/dev/null 2>&1 || true
docker volume rm "${ENVOY_VOLUME}" "${NGINX_CERTS_VOLUME}" >/dev/null 2>&1 || true
docker volume create "${ENVOY_VOLUME}" >/dev/null
docker volume create "${NGINX_CERTS_VOLUME}" >/dev/null

# Helper: pipe a file from this container into a named volume over the socket
# (no host path involved). Usage: stage_file <volume> <dest-path-in-volume> <src>
stage_file() { docker run --rm -i -v "$1:/vol" busybox sh -c "cat > /vol/$2" < "$3"; }
# Helper: copy named files from WEIGHTSLAB_CERTS_DIR into <volume>/<subdir> via tar.
# The generated key files are 0600/root; Envoy + nginx run as non-root, so make
# them readable (dev certs) — otherwise Envoy crashes with "unable to read file".
stage_certs() { local vol="$1" sub="$2"; shift 2
    tar -C "${WEIGHTSLAB_CERTS_DIR}" -cf - "$@" \
      | docker run --rm -i -v "${vol}:/vol" busybox \
          sh -c "mkdir -p /vol/${sub} && tar -C /vol/${sub} -xf - && chmod -R a+rX /vol/${sub}"; }

if [ "${WEIGHTSLAB_TLS}" = "1" ]; then
    echo "[entrypoint] TLS mode — generating certs and staging the mTLS Envoy config..."
    # Generate ca.crt, envoy-server.*, envoy-client.*, backend-server.* + token.
    weightslab se
    ENVOY_TMPL="${PKG}/ui/envoy/envoy.yaml"   # FULL mTLS template
    [ -f "${ENVOY_TMPL}" ] || { echo "[entrypoint] ERROR: ${ENVOY_TMPL} not found" >&2; exit 1; }
    sed "s/__GRPC_BACKEND_PORT__/${GRPC_BACKEND_PORT}/g" "${ENVOY_TMPL}" > /tmp/envoy.generated.yaml
    stage_file "${ENVOY_VOLUME}" "envoy.yaml" /tmp/envoy.generated.yaml
    # Envoy's certs (downstream server + upstream client + CA) -> /etc/envoy/certs
    stage_certs "${ENVOY_VOLUME}" "certs" ca.crt envoy-server.crt envoy-server.key envoy-client.crt envoy-client.key
    # The frontend (nginx) HTTPS cert -> /etc/nginx/certs (triggers HTTPS mode)
    stage_certs "${NGINX_CERTS_VOLUME}" "." envoy-server.crt envoy-server.key
    export WS_SERVER_PROTOCOL=https
    # Backend gRPC TLS (else Envoy's upstream mTLS fails against a plaintext backend).
    export GRPC_TLS_ENABLED=1
    export GRPC_TLS_CERT_DIR="${WEIGHTSLAB_CERTS_DIR}"
    SCHEME="https"
else
    echo "[entrypoint] HTTP mode — staging the plaintext Envoy config..."
    ENVOY_TMPL="${PKG}/ui/envoy/envoy.downstream_upstream_plaintext.yaml"
    [ -f "${ENVOY_TMPL}" ] || { echo "[entrypoint] ERROR: ${ENVOY_TMPL} not found" >&2; exit 1; }
    sed "s/__GRPC_BACKEND_PORT__/${GRPC_BACKEND_PORT}/g" "${ENVOY_TMPL}" > /tmp/envoy.generated.yaml
    stage_file "${ENVOY_VOLUME}" "envoy.yaml" /tmp/envoy.generated.yaml
    # nginx certs volume left empty -> frontend serves HTTP.
    export WS_SERVER_PROTOCOL=http
    SCHEME="http"
fi
echo "[entrypoint] Staged Envoy config (backend port ${GRPC_BACKEND_PORT}, scheme ${SCHEME})."

# --- Bring up Envoy + frontend as siblings on the host daemon ----------------
echo "[entrypoint] Starting Envoy + frontend (bind-mount-free) on the host daemon..."
WS_SERVER_PROTOCOL="${WS_SERVER_PROTOCOL}" \
    docker compose -p "${UI_PROJECT}" -f "${UI_COMPOSE}" up -d --pull always

echo "[entrypoint] UI is up — open ${SCHEME}://localhost:5173 once training is serving."

# --- Run the example — serves gRPC on :GRPC_BACKEND_PORT (published) ----------
echo "[entrypoint] Starting the classification (cls) example — serves gRPC on :${GRPC_BACKEND_PORT}."
exec weightslab start example --cls
