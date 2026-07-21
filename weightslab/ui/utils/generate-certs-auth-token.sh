#!/bin/bash

set -e

# Git Bash / MSYS rewrites arguments that look like POSIX paths into Windows
# paths. That corrupts the openssl "-subj /CN=..." values, but the "-out /tmp/.."
# paths DO need conversion — so exclude only the subject strings. Harmless no-op
# on Linux/macOS.
export MSYS2_ARG_CONV_EXCL="/CN=;/C=;/O=;/OU="

SKIP_TRUST=false
FORCE_CREATE_CERTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-trust)
            SKIP_TRUST=true
            shift
            ;;
        --force-create-certs)
            FORCE_CREATE_CERTS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if ! command -v openssl &> /dev/null; then
    echo "openssl is required but was not found in PATH."
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Single source of truth: WEIGHTSLAB_CERTS_DIR decides where certs live.
# Default to ~/.weightslab-certs when not provided.
USER_CERT_DIR="${WEIGHTSLAB_CERTS_DIR:-$HOME/.weightslab-certs}"
echo "Using certs directory: $USER_CERT_DIR"

mkdir -p "$USER_CERT_DIR"

CERTS_EXIST=false
if [ -f "$USER_CERT_DIR/ca.crt" ] && \
   [ -f "$USER_CERT_DIR/ui-server.crt" ] && \
   [ -f "$USER_CERT_DIR/backend-server.crt" ]; then
    CERTS_EXIST=true
fi

if [ "$CERTS_EXIST" = true ] && [ "$FORCE_CREATE_CERTS" = false ]; then
    echo "Using existing certificates from $USER_CERT_DIR..."
    exit 0
fi

if [ "$FORCE_CREATE_CERTS" = true ]; then
    echo "Force creating new certificates (--force-create-certs)..."
fi

TMP_DIR=$(mktemp -d)

cleanup() {
    if [ -d "$TMP_DIR" ]; then
        rm -rf "$TMP_DIR"
    fi
}

trap cleanup EXIT

echo "Generating local dev CA..."
openssl genrsa -out "$TMP_DIR/ca.key" 4096
openssl req -x509 -new -nodes -key "$TMP_DIR/ca.key" -sha256 -days 825 \
    -subj "/CN=weightslab-dev-ca" \
    -out "$TMP_DIR/ca.crt"

cat > "$TMP_DIR/ui-server.ext" << 'EOF'
subjectAltName = DNS:localhost,IP:127.0.0.1,IP:0:0:0:0:0:0:0:1
extendedKeyUsage = serverAuth
EOF

cat > "$TMP_DIR/backend-server.ext" << 'EOF'
subjectAltName = DNS:localhost,IP:127.0.0.1,IP:0:0:0:0:0:0:0:1
extendedKeyUsage = serverAuth
EOF

cat > "$TMP_DIR/ui-client.ext" << 'EOF'
extendedKeyUsage = clientAuth
EOF

echo "Generating UI HTTPS server cert..."
openssl genrsa -out "$TMP_DIR/ui-server.key" 2048
openssl req -new -key "$TMP_DIR/ui-server.key" -subj "/CN=localhost" -out "$TMP_DIR/ui-server.csr"
openssl x509 -req -in "$TMP_DIR/ui-server.csr" -CA "$TMP_DIR/ca.crt" -CAkey "$TMP_DIR/ca.key" \
    -CAcreateserial -out "$TMP_DIR/ui-server.crt" -days 825 -sha256 -extfile "$TMP_DIR/ui-server.ext"

echo "Generating backend gRPC server cert..."
openssl genrsa -out "$TMP_DIR/backend-server.key" 2048
openssl req -new -key "$TMP_DIR/backend-server.key" -subj "/CN=localhost" -out "$TMP_DIR/backend-server.csr"
openssl x509 -req -in "$TMP_DIR/backend-server.csr" -CA "$TMP_DIR/ca.crt" -CAkey "$TMP_DIR/ca.key" \
    -CAcreateserial -out "$TMP_DIR/backend-server.crt" -days 825 -sha256 -extfile "$TMP_DIR/backend-server.ext"

echo "Generating UI mTLS client cert..."
openssl genrsa -out "$TMP_DIR/ui-client.key" 2048
openssl req -new -key "$TMP_DIR/ui-client.key" -subj "/CN=ui-client" -out "$TMP_DIR/ui-client.csr"
openssl x509 -req -in "$TMP_DIR/ui-client.csr" -CA "$TMP_DIR/ca.crt" -CAkey "$TMP_DIR/ca.key" \
    -CAcreateserial -out "$TMP_DIR/ui-client.crt" -days 825 -sha256 -extfile "$TMP_DIR/ui-client.ext"

cp -f "$TMP_DIR/ca.crt" "$USER_CERT_DIR/ca.crt"
cp -f "$TMP_DIR/ui-server.crt" "$USER_CERT_DIR/ui-server.crt"
cp -f "$TMP_DIR/ui-server.key" "$USER_CERT_DIR/ui-server.key"
cp -f "$TMP_DIR/ui-client.crt" "$USER_CERT_DIR/ui-client.crt"
cp -f "$TMP_DIR/ui-client.key" "$USER_CERT_DIR/ui-client.key"
cp -f "$TMP_DIR/backend-server.crt" "$USER_CERT_DIR/backend-server.crt"
cp -f "$TMP_DIR/backend-server.key" "$USER_CERT_DIR/backend-server.key"

if [ "$SKIP_TRUST" = false ]; then
    echo "Note: CA trust installation requires platform-specific commands (macOS: security add-trusted-cert, Linux: system-dependent)"
    echo "Consider running: sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain $USER_CERT_DIR/ca.crt (macOS)"
    echo "Or for Linux, consult your distribution's certificate management tools."
else
    echo "Skipping CA trust installation (--skip-trust)."
fi

echo "Dev certificates generated successfully."
echo "- Certs location: $USER_CERT_DIR"
