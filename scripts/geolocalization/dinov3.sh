#!/usr/bin/env bash
# Usage: bash scripts/dinov3.sh 0,1,2,3  (or export CUDA_VISIBLE_DEVICES beforehand)
export CUDA_VISIBLE_DEVICES=${1:-${CUDA_VISIBLE_DEVICES:-0}}

# Huggingface settings

TASK_CONFIG=GlobalGeoLocalization
MODEL_CONFIG=DinoV3
EXP_NAME=main_exp

# query settings
QUERY_MODE=image   # image (DINOv3 only supports image)
QUERY_TEXTS=(alluvial_fans glacier-like_form landslides pitted_cones yardangs)

# model / encoder settings (DINOv3)
IMAGE_ENCODER_TYPE=dinov3
TEXT_ENCODER_TYPE=none
MODEL_NAME=facebook/dinov3-vitl16-pretrain-lvd1689m  # HF model ID
PRETRAINED=hf  # Only used for logging/naming

# optional distributed DB build (skips if DB already exists)
NPROC=$(( $(echo "${CUDA_VISIBLE_DEVICES:-}" | tr -cd ',' | wc -c) + 1 ))
if [[ "${NPROC}" -gt 1 ]]; then
  echo "Building DB with torchrun on ${NPROC} GPUs: ${CUDA_VISIBLE_DEVICES}"
  torchrun --nproc_per_node=${NPROC} build_geo_db.py \
    --task_config "${TASK_CONFIG}" \
    --model_config "${MODEL_CONFIG}" \
    --exp_name "${EXP_NAME}" \
    --image_encoder_type "${IMAGE_ENCODER_TYPE}" \
    --text_encoder_type "none" \
    --model_name "${MODEL_NAME}" \
    --pretrained "${PRETRAINED}"
fi

# run retrieval

for QUERY_TEXT in "${QUERY_TEXTS[@]}"; do
  python main.py \
    --task_config "${TASK_CONFIG}" \
      --model_config "${MODEL_CONFIG}" \
    --exp_name "${EXP_NAME}" \
    --query_mode "${QUERY_MODE}" \
    --query_text "${QUERY_TEXT}" \
    --image_encoder_type "${IMAGE_ENCODER_TYPE}" \
    --text_encoder_type "${TEXT_ENCODER_TYPE}" \
    --model_name "${MODEL_NAME}" \
    --pretrained "${PRETRAINED}"

done
