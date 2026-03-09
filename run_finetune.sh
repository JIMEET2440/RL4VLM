#!/bin/bash
# ============================================================================
# run_finetune.sh - SFT finetuning for LLaVA-v1.6-mistral-7b on ALFWorld data
# ============================================================================
# Based on: RL4VLM/finetune.sh (template)
# Data:     LEVI-Project/sft-data (HuggingFace)
# Model:    liuhaotian/llava-v1.6-mistral-7b
# ============================================================================

set -e

# ---- Paths ----
ROOT_DIR="/mnt/raid/rl_gaming/RL4VLM_ALF/RL4VLM"
LLAVA_DIR="${ROOT_DIR}/LLaVA"
VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"
DATA_JSON="${ROOT_DIR}/sft_data/alfworld-gpt4-45k.json"
IMAGE_FOLDER="${ROOT_DIR}/sft_data/alf_data_folder"
OUTPUT_DIR="${ROOT_DIR}/checkpoints/llava-v1.6-mistral-7b-alf-sft"
DEEPSPEED_CONFIG="${LLAVA_DIR}/scripts/zero2.json"

# ---- GPU Config ----
# Using physical GPU 4 only (free GPU on the server).
# GPU 6 is in Exclusive_Process mode (owned by co-worker).
# GPU 4 is in Default mode with 58 MiB used (essentially free).
export CUDA_VISIBLE_DEVICES=4

# ---- Activate venv ----
source "${ROOT_DIR}/.venv/bin/activate"

# ---- Run from LLaVA directory ----
cd "${LLAVA_DIR}"

echo "============================================"
echo "Starting LLaVA SFT Finetuning"
echo "  Model:   liuhaotian/llava-v1.6-mistral-7b"
echo "  Data:    ${DATA_JSON}"
echo "  Images:  ${IMAGE_FOLDER}"
echo "  Output:  ${OUTPUT_DIR}"
echo "  GPU:     Physical GPU 4 (CUDA_VISIBLE_DEVICES=4)"
echo "============================================"

# For single-GPU training on H200 (143 GB VRAM), DeepSpeed is not needed.
# The 7B model (~14 GB in bf16) + optimizer states (~28 GB) + activations
# fits comfortably without ZeRO sharding. Using the deepspeed launcher on a
# single GPU causes NCCL hangs, and without the launcher, HF Trainer creates
# DummyOptim placeholders that crash. So we skip DeepSpeed entirely.
"${VENV_PYTHON}" -u llava/train/train_mem.py \
    --model_name_or_path liuhaotian/llava-v1.6-mistral-7b \
    --version v1 \
    --data_path "${DATA_JSON}" \
    --image_folder "${IMAGE_FOLDER}" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none
