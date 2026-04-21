#!/bin/bash
set -euo pipefail

# Default environment file path; can be overridden by first argument.
ENV_FILE_PATH="${1:-./../.env}"

if [ ! -f "$ENV_FILE_PATH" ]; then
	echo "Error: Environment file '$ENV_FILE_PATH' not found." >&2
	exit 1
fi

echo "Loading environment variables from $ENV_FILE_PATH..."

set -a
# shellcheck disable=SC1090
source "$ENV_FILE_PATH"
set +a

echo "Environment loading completed."
