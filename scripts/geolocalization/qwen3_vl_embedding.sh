#!/usr/bin/env bash
# Usage: bash scripts/opsmm.sh 0,1  (or export CUDA_VISIBLE_DEVICES beforehand)
export TORCH_DISTRIBUTED_TIMEOUT=86400
export TRANSFORMERS_VERBOSITY=error HF_HUB_VERBOSITY=error
export CUDA_VISIBLE_DEVICES=${1:-${CUDA_VISIBLE_DEVICES:-0}}
export PATH=~/.conda/envs/retrieval/bin:$PATH

# Huggingface mirrors/caches

TASK_CONFIG=GlobalGeoLocalization
MODEL_CONFIG=Qwen3VLEmbedding
EXP_NAME=main_exp

# Query settings
QUERY_MODES=(image text)
QUERY_TEXTS=(alluvial_fans glacier-like_form landslides pitted_cones yardangs)

# Qwen3-VL embedding
MODEL_NAME=Qwen/Qwen3-VL-Embedding-2B
PRETRAINED=""
IMAGE_ENCODER_TYPE=qwen3_vl_embedding
TEXT_ENCODER_TYPE=qwen3_vl_embedding

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


for QUERY_MODE in "${QUERY_MODES[@]}"; do

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

done
