#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
RUN_MODE="${RUN_MODE:-smoke}"
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/models/Qwen2.5-7B}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

case "$RUN_MODE" in
  smoke)
    CONFIG="$ROOT_DIR/configs/train_sft32k_instruct_v2_smoke.json"
    DEFAULT_OUTPUT_DIR="$ROOT_DIR/runs/train_sft32k_instruct_v2_smoke"
    ;;
  formal)
    CONFIG="$ROOT_DIR/configs/train_sft32k_instruct_v2.json"
    DEFAULT_OUTPUT_DIR="$ROOT_DIR/runs/touchagent_qwen2_5_7b_lora_sft32k_instruct_v2_seed42"
    ;;
  *)
    echo "RUN_MODE must be smoke or formal" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}.log}"

IFS=',' read -r -a DEVICE_LIST <<< "$CUDA_DEVICES"
if [[ "${#DEVICE_LIST[@]}" -ne 4 ]]; then
  echo "CUDA_DEVICES must contain exactly four distinct GPU IDs" >&2
  exit 2
fi
declare -A SEEN_DEVICES=()
for device in "${DEVICE_LIST[@]}"; do
  if [[ ! "$device" =~ ^[0-9]+$ ]] || [[ -n "${SEEN_DEVICES[$device]:-}" ]]; then
    echo "CUDA_DEVICES must contain exactly four distinct non-negative integers" >&2
    exit 2
  fi
  SEEN_DEVICES["$device"]=1
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable is unavailable: $PYTHON_BIN" >&2
  exit 2
fi
if ! command -v "$TORCHRUN_BIN" >/dev/null 2>&1; then
  echo "torchrun executable is unavailable: $TORCHRUN_BIN" >&2
  exit 2
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Local model directory is missing: $MODEL_PATH" >&2
  exit 2
fi
if [[ ! -f "$ROOT_DIR/reports/tokenizer_audit.json" ]]; then
  echo "Frozen tokenizer audit is missing: $ROOT_DIR/reports/tokenizer_audit.json" >&2
  exit 2
fi
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  if [[ ! -d "$RESUME_FROM_CHECKPOINT" ]]; then
    echo "Resume checkpoint does not exist: $RESUME_FROM_CHECKPOINT" >&2
    exit 2
  fi
elif [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]]; then
  echo "Training output is not empty: $OUTPUT_DIR" >&2
  exit 2
fi

COMMON_ARGS=(
  --config "$CONFIG"
  --model-path "$MODEL_PATH"
  --output-dir "$OUTPUT_DIR"
)
TRAIN_ARGS=("${COMMON_ARGS[@]}")
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  TRAIN_ARGS+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi

cd "$ROOT_DIR"
mkdir -p "$(dirname "$OUTPUT_DIR")" "$(dirname "$LOG_FILE")"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "[$(date -Is)] TouchAgent-Instruct-v2 SFT $RUN_MODE; devices=$CUDA_DEVICES"
"$PYTHON_BIN" -m touchagent_train.cli preflight "${COMMON_ARGS[@]}"
"$PYTHON_BIN" -m touchagent_train.cli verify-manifest "${COMMON_ARGS[@]}"
"$TORCHRUN_BIN" --standalone --nproc_per_node=4 \
  -m touchagent_train.cli train "${TRAIN_ARGS[@]}" \
  2>&1 | tee -a "$LOG_FILE"
