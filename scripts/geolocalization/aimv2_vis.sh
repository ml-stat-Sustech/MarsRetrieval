#!/usr/bin/env bash
# Usage: bash scripts/aimv2_vis.sh 0,1  (or export CUDA_VISIBLE_DEVICES beforehand)
export CUDA_VISIBLE_DEVICES=${1:-${CUDA_VISIBLE_DEVICES:-0}}
export PATH=~/.conda/envs/retrieval/bin:$PATH

# Huggingface mirrors/caches

TASK_CONFIG=GlobalGeoLocalization
MODEL_CONFIG=AimV2Vis
EXP_NAME=main_exp

# Query settings
QUERY_MODE=image   # image-only retrieval
QUERY_TEXTS=(alluvial_fans glacier-like_form landslides pitted_cones yardangs)

# AIMv2 vision-only model
MODEL_NAME=apple/aimv2-large-patch14-448
PRETRAINED=cefb13f21003bdadba65bfbee956c82b976cd23d
IMAGE_ENCODER_TYPE=aimv2_vis
TEXT_ENCODER_TYPE=none

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
