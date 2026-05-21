#!/usr/bin/env bash
set -euo pipefail

# Run every du_iibtd_based/*/train.py entrypoint sequentially with the
# residual-Sr DU-IIBTD backend.
#
# Optional environment overrides shared by every entrypoint:
#   PYTHON_BIN="python"                 # or e.g. "conda run -n myenv python"
#   DEVICE="cuda:0"
#   IIBTD_DEVICE="cuda:0"
#   SEED="42"
#   RADIOSEER_SAMPLE_INDEX="255"
#   TOTAL_TIMESTEPS="120000"          # optional learned-policy training length
#   BATCH_LOG_DIR="/path/to/batch/logs"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

read -r -a PYTHON_CMD <<< "${PYTHON_BIN:-python}"

RES_SR_CHECKPOINTS=(
  "DU_IIBTD_res_Sr/runs_t3_h04_res_balance_bw/checkpoints/best_nmse.pth"
  "DU_IIBTD_res_Sr/runs_t3_h05_res_balance_bw/checkpoints/best_nmse.pth"
  "DU_IIBTD_res_Sr/runs_t3_h06_res_balance_bw/checkpoints/best_nmse.pth"
)
RES_SR_CHECKPOINT_CSV="$(IFS=,; printf '%s' "${RES_SR_CHECKPOINTS[*]}")"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
BATCH_LOG_DIR="${BATCH_LOG_DIR:-${SCRIPT_DIR}/batch_logs/${RUN_ID}}"

COMMON_ARGS=()
if [[ -n "${DEVICE:-}" ]]; then
  COMMON_ARGS+=(--device "${DEVICE}")
fi
if [[ -n "${IIBTD_DEVICE:-}" ]]; then
  COMMON_ARGS+=(--iibtd_device "${IIBTD_DEVICE}")
fi
if [[ -n "${SEED:-}" ]]; then
  COMMON_ARGS+=(--seed "${SEED}")
fi
if [[ -n "${RADIOSEER_SAMPLE_INDEX:-}" ]]; then
  COMMON_ARGS+=(--radioseer_sample_index "${RADIOSEER_SAMPLE_INDEX}")
fi

cd "${ROOT_DIR}"

for checkpoint in "${RES_SR_CHECKPOINTS[@]}"; do
  if [[ ! -f "${checkpoint}" ]]; then
    echo "[ERROR] Missing residual-Sr checkpoint: ${checkpoint}" >&2
    exit 1
  fi
done

mkdir -p "${BATCH_LOG_DIR}"

mapfile -t TRAIN_SCRIPTS < <(
  find "${SCRIPT_DIR}" \
    -mindepth 2 \
    -maxdepth 2 \
    -type f \
    -name train.py \
    | sort
)

if [[ "${#TRAIN_SCRIPTS[@]}" -eq 0 ]]; then
  echo "[ERROR] No train.py files found under ${SCRIPT_DIR}" >&2
  exit 1
fi

echo "[INFO] Found ${#TRAIN_SCRIPTS[@]} train.py entrypoints."
echo "[INFO] Residual-Sr backend forced via --iibtd_backend du_iibtd_res_sr."
echo "[INFO] Python stdout/stderr unbuffered."
echo "[INFO] Batch logs: ${BATCH_LOG_DIR}"

export PYTHONUNBUFFERED=1

cleanup_after_train() {
  local suite_name="$1"
  echo "[INFO] ${suite_name} finished. Waiting 3s for OS/CUDA resources to settle..."
  wait || true
  sleep 3
}

for train_py in "${TRAIN_SCRIPTS[@]}"; do
  suite_dir="$(dirname "${train_py}")"
  suite_name="$(basename "${suite_dir}")"
  run_log="${BATCH_LOG_DIR}/${suite_name}.log"

  args=()

  if grep -q -- "--iibtd_du_checkpoint_path" "${train_py}"; then
    # Newer shareMember branches use one comma-separated checkpoint option.
    args+=(
      --iibtd_backend du_iibtd_res_sr
      --iibtd_du_checkpoint_path "${RES_SR_CHECKPOINT_CSV}"
    )
  elif grep -q -- "--du_iibtd_checkpoints" "${train_py}"; then
    # Legacy branches use nargs='+' for the same residual-Sr ensemble.
    args+=(
      --iibtd_backend du_iibtd_res_sr
      --du_iibtd_checkpoints "${RES_SR_CHECKPOINTS[@]}"
    )
  else
    echo "[ERROR] ${train_py} has no recognized DU-IIBTD checkpoint argument." >&2
    exit 1
  fi

  if grep -q -- "--log_dir" "${train_py}"; then
    args+=(--log_dir "${suite_dir}/logs")
  fi
  if grep -q -- "--model_dir" "${train_py}"; then
    args+=(--model_dir "${suite_dir}/checkpoints")
  fi
  if [[ -n "${TOTAL_TIMESTEPS:-}" ]] && grep -q -- "--total_timesteps" "${train_py}"; then
    args+=(--total_timesteps "${TOTAL_TIMESTEPS}")
  fi
  args+=("${COMMON_ARGS[@]}")

  echo
  echo "================================================================"
  echo "[RUN] ${suite_name}"
  printf '[CMD]'
  printf ' %q' "${PYTHON_CMD[@]}" "${train_py}" "${args[@]}"
  printf '\n'
  echo "================================================================"

  set +e
  "${PYTHON_CMD[@]}" "${train_py}" "${args[@]}" 2>&1 | tee "${run_log}"
  pipeline_status=("${PIPESTATUS[@]}")
  set -e

  cleanup_after_train "${suite_name}"

  train_status="${pipeline_status[0]}"
  tee_status="${pipeline_status[1]}"
  if [[ "${train_status}" -ne 0 ]]; then
    echo "[ERROR] ${suite_name} failed with exit status ${train_status}. See ${run_log}" >&2
    exit "${train_status}"
  fi
  if [[ "${tee_status}" -ne 0 ]]; then
    echo "[ERROR] tee failed for ${suite_name} with exit status ${tee_status}." >&2
    exit "${tee_status}"
  fi
done

echo
echo "[DONE] All train.py entrypoints finished."
echo "[DONE] Batch logs saved under: ${BATCH_LOG_DIR}"
