#!/usr/bin/env bash
# Usage: bash scripts/paired_image_text_retrieval/openclip.sh 0,1,2,3 (or export CUDA_VISIBLE_DEVICES beforehand)
export CUDA_VISIBLE_DEVICES=${1:-${CUDA_VISIBLE_DEVICES:-0}}
export PATH=~/.conda/envs/retrieval/bin:$PATH

# Huggingface settings

TASK_CONFIG=PairedVLRetrieval
MODEL_CONFIG=OpenCLIP
EXP_NAME=main_exp

# model (OpenCLIP by default)
MODEL_NAMES=(
  # ViT-L-14-quickgelu
  # ViT-L-16-SigLIP-384
  # ViT-L-16-SigLIP2-512
  PE-Core-L-14-336
)
PRETRAINEDS=(
  # dfn2b
  # hf-hub:timm/ViT-L-16-SigLIP-384
  # hf-hub:timm/ViT-L-16-SigLIP2-512
  hf-hub:timm/PE-Core-L-14-336
)
IMAGE_ENCODER_TYPE=openclip
TEXT_ENCODER_TYPE=openclip

for IDX in "${!MODEL_NAMES[@]}"; do
  MODEL_NAME="${MODEL_NAMES[$IDX]}"
  PRETRAINED="${PRETRAINEDS[$IDX]}"

  python main.py \
    --task_config "${TASK_CONFIG}" \
    --model_config "${MODEL_CONFIG}" \
    --exp_name "${EXP_NAME}" \
    --query_mode "text" \
    --model_name "${MODEL_NAME}" \
    --pretrained "${PRETRAINED}" \
    --image_encoder_type "${IMAGE_ENCODER_TYPE}" \
    --text_encoder_type "${TEXT_ENCODER_TYPE}"
done
