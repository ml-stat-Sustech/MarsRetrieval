#!/usr/bin/env bash
# Usage: bash scripts/landform_retrieval/qwen3_vl_embedding.sh 0,1 (or export CUDA_VISIBLE_DEVICES beforehand)
export CUDA_VISIBLE_DEVICES=${1:-${CUDA_VISIBLE_DEVICES:-0}}
export PATH=~/.conda/envs/retrieval/bin:$PATH


TASK_CONFIG=LandformRetrieval
MODEL_CONFIG=Qwen3VLEmbedding
EXP_NAME=main_exp

QUERY_MODE=text

MODEL_NAME=Qwen/Qwen3-VL-Embedding-2B
PRETRAINED=qwen3_vl_embedding
IMAGE_ENCODER_TYPE=qwen3_vl_embedding
TEXT_ENCODER_TYPE=qwen3_vl_embedding

python main.py \
  --task_config "${TASK_CONFIG}" \
  --model_config "${MODEL_CONFIG}" \
  --exp_name "${EXP_NAME}" \
  --query_mode "${QUERY_MODE}" \
  --model_name "${MODEL_NAME}" \
  --pretrained "${PRETRAINED}" \
  --image_encoder_type "${IMAGE_ENCODER_TYPE}" \
  --text_encoder_type "${TEXT_ENCODER_TYPE}"
