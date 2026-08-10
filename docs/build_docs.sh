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

INDEX_HTML="$ROOT_DIR/docs/_build/html/index.html"

echo "[weightslab-docs] Build complete:"
echo "  $INDEX_HTML"

if [[ "${WEIGHTSLAB_DOCS_NO_OPEN:-0}" == "1" ]]; then
  echo "[weightslab-docs] Auto-open disabled (WEIGHTSLAB_DOCS_NO_OPEN=1)."
  exit 0
fi

echo "[weightslab-docs] Opening docs index in your browser..."

if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$INDEX_HTML" >/dev/null 2>&1 || true
elif command -v open >/dev/null 2>&1; then
  open "$INDEX_HTML" >/dev/null 2>&1 || true
elif command -v cmd.exe >/dev/null 2>&1; then
  cmd.exe /c start "" "$INDEX_HTML" >/dev/null 2>&1 || true
elif command -v powershell.exe >/dev/null 2>&1; then
  if command -v wslpath >/dev/null 2>&1; then
    WIN_INDEX_HTML="$(wslpath -w "$INDEX_HTML")"
  else
    WIN_INDEX_HTML="$INDEX_HTML"
  fi
  powershell.exe -NoProfile -Command "Start-Process '$WIN_INDEX_HTML'" >/dev/null 2>&1 || true
else
  echo "[weightslab-docs] Could not auto-open browser on this shell. Open manually:"
  echo "  $INDEX_HTML"
fi
