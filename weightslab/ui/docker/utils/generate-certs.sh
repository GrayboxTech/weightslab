#!/usr/bin/env bash
set -euo pipefail

TRUST_CA="${TRUST_CA:-1}"
FORCE_CREATE="${FORCE_CREATE:-0}"

# Parse command line arguments
for arg in "$@"; do
  if [ "$arg" = "--force-create-certs" ]; then
    FORCE_CREATE=1
  fi
done

if ! command -v openssl >/dev/null 2>&1; then
  echo "Error: openssl is required but was not found in PATH." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use user's default certificate directory
USER_CERT_DIR="${HOME}/.weightslab-certs"

mkdir -p "${USER_CERT_DIR}"

# Check if certs already exist in user directory
CERTS_EXIST=0
if [ -f "${USER_CERT_DIR}/ca.crt" ] && [ -f "${USER_CERT_DIR}/envoy-server.crt" ] && [ -f "${USER_CERT_DIR}/backend-server.crt" ]; then
  CERTS_EXIST=1
fi

# If certs exist and not forcing recreation, skip generation
if [ "$CERTS_EXIST" = "1" ] && [ "$FORCE_CREATE" = "0" ]; then
  echo "Using existing certificates from ${USER_CERT_DIR}..."
  exit 0
fi

if [ "$FORCE_CREATE" = "1" ]; then
  echo "Force creating new certificates (--force-create-certs)..."
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

echo "Generating local dev CA..."
openssl genrsa -out "${TMP_DIR}/ca.key" 4096
openssl req -x509 -new -nodes -key "${TMP_DIR}/ca.key" -sha256 -days 825 \
  -subj "/CN=weightslab-dev-ca" \
  -out "${TMP_DIR}/ca.crt"

cat > "${TMP_DIR}/envoy-server.ext" << 'EOF'
subjectAltName = DNS:localhost,IP:127.0.0.1,IP:0:0:0:0:0:0:0:1
extendedKeyUsage = serverAuth
EOF

cat > "${TMP_DIR}/backend-server.ext" << 'EOF'
subjectAltName = DNS:localhost,DNS:host.docker.internal,IP:127.0.0.1,IP:0:0:0:0:0:0:0:1
extendedKeyUsage = serverAuth
EOF

cat > "${TMP_DIR}/envoy-client.ext" << 'EOF'
extendedKeyUsage = clientAuth
EOF

echo "Generating Envoy HTTPS server cert..."
openssl genrsa -out "${TMP_DIR}/envoy-server.key" 2048
openssl req -new -key "${TMP_DIR}/envoy-server.key" -subj "/CN=localhost" -out "${TMP_DIR}/envoy-server.csr"
openssl x509 -req -in "${TMP_DIR}/envoy-server.csr" -CA "${TMP_DIR}/ca.crt" -CAkey "${TMP_DIR}/ca.key" \
  -CAcreateserial -out "${TMP_DIR}/envoy-server.crt" -days 825 -sha256 -extfile "${TMP_DIR}/envoy-server.ext"

echo "Generating backend gRPC server cert..."
openssl genrsa -out "${TMP_DIR}/backend-server.key" 2048
openssl req -new -key "${TMP_DIR}/backend-server.key" -subj "/CN=host.docker.internal" -out "${TMP_DIR}/backend-server.csr"
openssl x509 -req -in "${TMP_DIR}/backend-server.csr" -CA "${TMP_DIR}/ca.crt" -CAkey "${TMP_DIR}/ca.key" \
  -CAcreateserial -out "${TMP_DIR}/backend-server.crt" -days 825 -sha256 -extfile "${TMP_DIR}/backend-server.ext"

echo "Generating Envoy mTLS client cert..."
openssl genrsa -out "${TMP_DIR}/envoy-client.key" 2048
openssl req -new -key "${TMP_DIR}/envoy-client.key" -subj "/CN=envoy-client" -out "${TMP_DIR}/envoy-client.csr"
openssl x509 -req -in "${TMP_DIR}/envoy-client.csr" -CA "${TMP_DIR}/ca.crt" -CAkey "${TMP_DIR}/ca.key" \
  -CAcreateserial -out "${TMP_DIR}/envoy-client.crt" -days 825 -sha256 -extfile "${TMP_DIR}/envoy-client.ext"

# Install to user's default directory (for reuse)
install -m 0644 "${TMP_DIR}/ca.crt" "${USER_CERT_DIR}/ca.crt"
install -m 0644 "${TMP_DIR}/envoy-server.crt" "${USER_CERT_DIR}/envoy-server.crt"
install -m 0600 "${TMP_DIR}/envoy-server.key" "${USER_CERT_DIR}/envoy-server.key"
install -m 0644 "${TMP_DIR}/envoy-client.crt" "${USER_CERT_DIR}/envoy-client.crt"
install -m 0600 "${TMP_DIR}/envoy-client.key" "${USER_CERT_DIR}/envoy-client.key"
install -m 0644 "${TMP_DIR}/backend-server.crt" "${USER_CERT_DIR}/backend-server.crt"
install -m 0600 "${TMP_DIR}/backend-server.key" "${USER_CERT_DIR}/backend-server.key"

if [[ "${TRUST_CA}" == "1" ]]; then
  CA_SRC="${USER_CERT_DIR}/ca.crt"
  OS_NAME="$(uname -s)"
  if [[ "${OS_NAME}" == "Darwin" ]]; then
    if security add-trusted-cert -d -r trustRoot -k "$HOME/Library/Keychains/login.keychain-db" "${CA_SRC}" >/dev/null 2>&1; then
      echo "CA trusted in macOS login keychain."
    else
      echo "Warning: could not trust CA automatically on macOS."
      echo "Run manually: security add-trusted-cert -d -r trustRoot -k \"$HOME/Library/Keychains/login.keychain-db\" \"${CA_SRC}\""
    fi
  elif [[ "${OS_NAME}" == "Linux" ]]; then
    if command -v update-ca-certificates >/dev/null 2>&1; then
      if sudo cp "${CA_SRC}" /usr/local/share/ca-certificates/weightslab-dev-ca.crt && sudo update-ca-certificates >/dev/null 2>&1; then
        echo "CA trusted in Linux system trust store."
      else
        echo "Warning: could not trust CA automatically on Linux."
        echo "Run manually: sudo cp \"${CA_SRC}\" /usr/local/share/ca-certificates/weightslab-dev-ca.crt && sudo update-ca-certificates"
      fi
    else
      echo "Warning: update-ca-certificates not found; install CA manually for your distro/browser."
    fi
  else
    echo "Warning: automatic CA trust is not implemented for OS '${OS_NAME}'."
  fi
else
  echo "Skipping CA trust installation (TRUST_CA=${TRUST_CA})."
fi

echo "Dev certificates generated successfully."
echo "- Certs location: ${USER_CERT_DIR}"
