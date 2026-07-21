#!/bin/sh
# Vendor the built Weights Studio SPA into this package so `weightslab start`
# can serve it (and so it ships inside the wheel). Docker-free.
#
# The frontend build lives in the weights_studio repo (its ./dist, gitignored).
# This script builds it there (unless SKIP_BUILD=1) and copies dist/ into
# weightslab/ui/static/ here.
#
# Usage:
#   weightslab/ui/utils/sync-frontend.sh
#   WEIGHTS_STUDIO_DIR=../weights_studio SKIP_BUILD=1 weightslab/ui/utils/sync-frontend.sh
#
# Env:
#   WEIGHTS_STUDIO_DIR  Path to the weights_studio repo (default: ../weights_studio
#                       relative to the weightslab repo root).
#   SKIP_BUILD=1        Reuse the existing dist/ instead of rebuilding.

set -e

# weightslab repo root = three levels up from this script (weightslab/ui/utils/).
PKG_STATIC_DIR="$(cd "$(dirname "$0")/.." && pwd)/static"
WL_REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
STUDIO_DIR="${WEIGHTS_STUDIO_DIR:-$WL_REPO_ROOT/../weights_studio}"

if [ ! -d "$STUDIO_DIR" ]; then
    echo "ERROR: weights_studio repo not found at: $STUDIO_DIR" >&2
    echo "  Set WEIGHTS_STUDIO_DIR to the weights_studio repo root." >&2
    exit 1
fi
STUDIO_DIR="$(cd "$STUDIO_DIR" && pwd)"

if [ "${SKIP_BUILD:-0}" = "1" ]; then
    echo "[sync-frontend] SKIP_BUILD=1 -> reusing existing dist/"
else
    echo "[sync-frontend] building SPA in $STUDIO_DIR ..."
    ( cd "$STUDIO_DIR" && npm run build )
fi

if [ ! -d "$STUDIO_DIR/dist" ]; then
    echo "ERROR: no dist/ at $STUDIO_DIR/dist (build first)" >&2
    exit 1
fi

echo "[sync-frontend] vendoring dist/ -> $PKG_STATIC_DIR"
rm -rf "$PKG_STATIC_DIR"
mkdir -p "$PKG_STATIC_DIR"
cp -R "$STUDIO_DIR/dist/." "$PKG_STATIC_DIR/"

echo "[sync-frontend] done. 'weightslab start' will now serve the updated UI."
