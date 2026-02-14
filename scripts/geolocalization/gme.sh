#!/usr/bin/env bash
# Usage: bash scripts/gme.sh 0,1  (or export CUDA_VISIBLE_DEVICES beforehand)
export TORCH_DISTRIBUTED_TIMEOUT=86400
export CUDA_VISIBLE_DEVICES=${1:-${CUDA_VISIBLE_DEVICES:-0}}
export PATH=~/.conda/envs/retrieval/bin:$PATH


TASK_CONFIG=GlobalGeoLocalization
MODEL_CONFIG=GME
EXP_NAME=main_exp

# Query settings
QUERY_MODES=(image text)
QUERY_TEXTS=(alluvial_fans glacier-like_form landslides pitted_cones yardangs)

# GME model (Qwen2-VL)
MODEL_NAME=Alibaba-NLP/gme-Qwen2-VL-2B-Instruct
PRETRAINED=gme
IMAGE_ENCODER_TYPE=gme
TEXT_ENCODER_TYPE=gme

# Optional distributed DB build (skips if DB already exists)
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

# Run retrieval

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

    # Note: GME requires transformers<4.52.0 (e.g., 4.51.3) because of remote code constraints.

  done

done
