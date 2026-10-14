#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD=("$PYTHON_BIN")
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=(python3)
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD=(python)
elif command -v py >/dev/null 2>&1; then
  PYTHON_CMD=(py -3)
else
  echo "[weightslab-docs] ERROR: no Python interpreter found in PATH."
  echo "[weightslab-docs] Set PYTHON_BIN explicitly and retry."
  exit 1
fi

echo "[weightslab-docs] Using Python: ${PYTHON_CMD[*]}"

need_docs_install=0
if ! "${PYTHON_CMD[@]}" -c "import sphinx" >/dev/null 2>&1; then
  need_docs_install=1
fi
if ! "${PYTHON_CMD[@]}" -c "import furo, myst_parser, sphinx_design, sphinxcontrib.mermaid" >/dev/null 2>&1; then
  need_docs_install=1
fi

if [ "$need_docs_install" -eq 1 ]; then
  echo "[weightslab-docs] Installing docs requirements..."
  "${PYTHON_CMD[@]}" -m pip install -r docs/requirements.txt
fi

if ! "${PYTHON_CMD[@]}" -c "import weightslab" >/dev/null 2>&1; then
  echo "[weightslab-docs] Installing local package (editable mode)..."
  "${PYTHON_CMD[@]}" -m pip install -e .
fi

echo "[weightslab-docs] Building HTML docs..."
"${PYTHON_CMD[@]}" -m sphinx -b html docs docs/_build/html

HTML_DIR="$ROOT_DIR/docs/_build/html"
INDEX_HTML="$HTML_DIR/index.html"

echo "[weightslab-docs] Build complete:"
echo "  $INDEX_HTML"

# ---------------------------------------------------------------------------
# Serve the built docs over HTTP.
#
# Opening index.html as a file:// URL breaks anything the browser treats as a
# cross-origin request (the search index, some of the JS assets). A local HTTP
# server gives the docs the same origin they have in production.
#
#   WEIGHTSLAB_DOCS_NO_SERVE=1  build only, do not serve
#   WEIGHTSLAB_DOCS_NO_OPEN=1   serve, but do not open a browser
#   WEIGHTSLAB_DOCS_HOST        bind address      (default 127.0.0.1)
#   WEIGHTSLAB_DOCS_PORT        preferred port    (default 8000)
# ---------------------------------------------------------------------------

if [[ "${WEIGHTSLAB_DOCS_NO_SERVE:-0}" == "1" ]]; then
  echo "[weightslab-docs] Serving disabled (WEIGHTSLAB_DOCS_NO_SERVE=1). Open manually:"
  echo "  $INDEX_HTML"
  exit 0
fi

DOCS_HOST="${WEIGHTSLAB_DOCS_HOST:-127.0.0.1}"
DOCS_PORT_REQUESTED="${WEIGHTSLAB_DOCS_PORT:-8000}"

# Pick the requested port, or the first free one above it.
DOCS_PORT="$("${PYTHON_CMD[@]}" - "$DOCS_HOST" "$DOCS_PORT_REQUESTED" <<'PY' || true
import socket
import sys

host, start = sys.argv[1], int(sys.argv[2])
for port in range(start, start + 50):
    sock = socket.socket()
    try:
        sock.bind((host, port))
    except OSError:
        continue
    finally:
        sock.close()
    print(port)
    break
PY
)"

if [[ -z "$DOCS_PORT" ]]; then
  echo "[weightslab-docs] ERROR: no free port in ${DOCS_PORT_REQUESTED}..$((DOCS_PORT_REQUESTED + 49)) on $DOCS_HOST."
  echo "[weightslab-docs] Set WEIGHTSLAB_DOCS_PORT to a free port and retry."
  exit 1
fi

if [[ "$DOCS_PORT" != "$DOCS_PORT_REQUESTED" ]]; then
  echo "[weightslab-docs] Port $DOCS_PORT_REQUESTED is busy, using $DOCS_PORT instead."
fi

DOCS_URL="http://${DOCS_HOST}:${DOCS_PORT}/index.html"

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[weightslab-docs] Serving $HTML_DIR at $DOCS_URL"
"${PYTHON_CMD[@]}" -m http.server "$DOCS_PORT" --bind "$DOCS_HOST" --directory "$HTML_DIR" >/dev/null 2>&1 &
SERVER_PID=$!

# Wait for the socket to accept connections before pointing a browser at it.
server_ready=0
for _ in $(seq 1 100); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    break
  fi
  if "${PYTHON_CMD[@]}" - "$DOCS_HOST" "$DOCS_PORT" <<'PY' >/dev/null 2>&1
import socket
import sys

with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=0.5):
    pass
PY
  then
    server_ready=1
    break
  fi
  sleep 0.1
done

if [ "$server_ready" -ne 1 ]; then
  echo "[weightslab-docs] ERROR: the docs server did not come up on $DOCS_URL."
  exit 1
fi

open_url() {
  local url="$1"
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$url" >/dev/null 2>&1 || true
  elif command -v open >/dev/null 2>&1; then
    open "$url" >/dev/null 2>&1 || true
  elif command -v cmd.exe >/dev/null 2>&1; then
    cmd.exe /c start "" "$url" >/dev/null 2>&1 || true
  elif command -v powershell.exe >/dev/null 2>&1; then
    powershell.exe -NoProfile -Command "Start-Process '$url'" >/dev/null 2>&1 || true
  else
    echo "[weightslab-docs] Could not auto-open a browser on this shell."
  fi
}

if [[ "${WEIGHTSLAB_DOCS_NO_OPEN:-0}" == "1" ]]; then
  echo "[weightslab-docs] Auto-open disabled (WEIGHTSLAB_DOCS_NO_OPEN=1). Open manually:"
  echo "  $DOCS_URL"
else
  echo "[weightslab-docs] Opening $DOCS_URL in your browser..."
  open_url "$DOCS_URL"
fi

echo "[weightslab-docs] Press Ctrl+C to stop the server."
wait "$SERVER_PID"
