#!/bin/bash
set -e

# Configuration
MODEL="google/gemma-3-27b-it"
PORT=8001
BASE_URL="http://localhost:${PORT}/v1"
MAX_TOKENS=512
NUM_EPISODES=${NUM_EPISODES:-200}
MAX_STEPS=50

# Optional: set START_SERVER=1 to launch vLLM from this script.
START_SERVER=${START_SERVER:-0}
VLLM_ENV=${VLLM_ENV:-vllm_clean}
CLIENT_ENV=${CLIENT_ENV:-rl4vlm-alf}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${SCRIPT_DIR}/debug_outputs/model_debug"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/../VLM_PPO_ALF/alf-config.yaml}"

cd "$SCRIPT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh

if [[ "$START_SERVER" == "1" ]]; then
  echo "Activating conda env for vLLM server: ${VLLM_ENV}"
  conda activate "$VLLM_ENV"

  echo "Starting vLLM server for ${MODEL} on port ${PORT}"
  # export CUDA_VISIBLE_DEVICES=7
  vllm serve "$MODEL" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.75 \
    --max-model-len 4096 \
    --host 0.0.0.0 \
    --port "$PORT" \
    > gemma_vllm.log 2>&1 &

  VLLM_PID=$!
fi

echo "Waiting for endpoint: ${BASE_URL}/models"
until curl -s "${BASE_URL}/models" > /dev/null 2>&1; do
  sleep 2
done
echo "Endpoint is ready."

echo "Activating client env: ${CLIENT_ENV}"
conda activate "$CLIENT_ENV"

mkdir -p "$OUTPUT_ROOT"
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="$OUTPUT_ROOT/gemma_3_27b_it_alf_debug_${timestamp}.log"

echo "========================================"
echo "Running ALFWorld debugger with endpoint model"
echo "Model     : ${MODEL}"
echo "Base URL  : ${BASE_URL}"
echo "Config    : ${CONFIG_PATH}"
echo "Output dir: ${OUTPUT_ROOT}"
echo "Log file  : ${log_file}"
echo "========================================"

python evaluation_single_model.py \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --config "$CONFIG_PATH" \
  --num-episodes "$NUM_EPISODES" \
  --max-steps "$MAX_STEPS" \
  --max-tokens "$MAX_TOKENS" \
  --output-dir "$OUTPUT_ROOT" \
  --log-json \
  2>&1 | tee "$log_file"

if [[ -n "${VLLM_PID:-}" ]]; then
  echo "Stopping vLLM server (pid=${VLLM_PID})"
  kill "$VLLM_PID" || true
fi