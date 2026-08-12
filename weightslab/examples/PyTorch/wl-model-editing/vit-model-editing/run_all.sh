#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../../.." && pwd)"
venv_path="${VIT_EDIT_VENV:-${repo_root}/.venv-vit-edit}"
python_bin="${venv_path}/bin/python"
output_dir="${VIT_EDIT_OUTPUT:-${repo_root}/outputs/vit_model_editing}"

export WL_NO_TELEMETRY="${WL_NO_TELEMETRY:-1}"
export WEIGHTSLAB_LOG_LEVEL="${WEIGHTSLAB_LOG_LEVEL:-INFO}"
export WEIGHTSLAB_DISABLE_WATCHDOGS="${WEIGHTSLAB_DISABLE_WATCHDOGS:-1}"

"${python_bin}" "${script_dir}/probe_full_vit.py" \
  --output-dir "${output_dir}" \
  --allow-unsupported

"${python_bin}" "${script_dir}/run_head_experiment.py" \
  --output-dir "${output_dir}"

echo "Experiment reports: ${output_dir}"
