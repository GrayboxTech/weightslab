#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../../.." && pwd)"
venv_path="${VIT_EDIT_VENV:-${repo_root}/.venv-vit-edit}"
python_bin="${PYTHON_BIN:-python3}"

"${python_bin}" -m venv "${venv_path}"
"${venv_path}/bin/python" -m pip install --upgrade pip
"${venv_path}/bin/python" -m pip install --editable "${repo_root}"

echo "Environment ready: ${venv_path}"
echo "Run: ${venv_path}/bin/python ${script_dir}/run_head_experiment.py"
