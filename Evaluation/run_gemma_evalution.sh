#!/bin/bash
set -e

MODEL="google/gemma-3-27b-it"
PORT=8001
BASE_URL="http://localhost:${PORT}/v1"

# Evaluation targets
PICK_EPISODES=${PICK_EPISODES:-200}
LOOK_EPISODES=${LOOK_EPISODES:-200}
MAX_STEPS=${MAX_STEPS:-80}
MAX_TOKENS=${MAX_TOKENS:-128}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-1.0}
SEED=${SEED:-42}

# Optional: start local vLLM server from this script.
START_SERVER=${START_SERVER:-0}
VLLM_ENV=${VLLM_ENV:-vllm_clean}
CLIENT_ENV=${CLIENT_ENV:-rl4vlm-alf}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/../VLM_PPO_ALF/alf-config.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/debug_outputs/model_debug}"
IMAGE_DIR="${IMAGE_DIR:-${SCRIPT_DIR}/images}"

cd "$SCRIPT_DIR"
source ~/miniconda3/etc/profile.d/conda.sh

if [[ "$START_SERVER" == "1" ]]; then
  echo "Activating ${VLLM_ENV} and launching vLLM server..."
  conda activate "$VLLM_ENV"

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
echo "Endpoint is ready"

echo "Activating client env: ${CLIENT_ENV}"
conda activate "$CLIENT_ENV"

mkdir -p "$OUTPUT_DIR" "$IMAGE_DIR"

echo "========================================"
echo "Gemma Pick/Look Evaluation"
echo "Model        : ${MODEL}"
echo "Base URL     : ${BASE_URL}"
echo "Pick episodes: ${PICK_EPISODES}"
echo "Look episodes: ${LOOK_EPISODES}"
echo "Max steps    : ${MAX_STEPS}"
echo "Image dir    : ${IMAGE_DIR}"
echo "Output dir   : ${OUTPUT_DIR}"
echo "========================================"

xvfb-run -a python Gemma_evalution.py \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --config "$CONFIG_PATH" \
  --pick-episodes "$PICK_EPISODES" \
  --look-episodes "$LOOK_EPISODES" \
  --max-steps "$MAX_STEPS" \
  --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --seed "$SEED" \
  --output-dir "$OUTPUT_DIR" \
  --image-dir "$IMAGE_DIR"

if [[ -n "${VLLM_PID:-}" ]]; then
  echo "Stopping vLLM server (pid=${VLLM_PID})"
  kill "$VLLM_PID" || true
fi
