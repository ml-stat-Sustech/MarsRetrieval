#!/usr/bin/env bash
# Usage: bash scripts/landform_retrieval/e5v.sh 0,1 (or export CUDA_VISIBLE_DEVICES beforehand)
export CUDA_VISIBLE_DEVICES=${1:-${CUDA_VISIBLE_DEVICES:-0}}
export PATH=~/.conda/envs/retrieval/bin:$PATH


TASK_CONFIG=LandformRetrieval
MODEL_CONFIG=E5V
EXP_NAME=main_exp

QUERY_MODE=text

MODEL_NAME=royokong/e5-v
PRETRAINED=e5-v
IMAGE_ENCODER_TYPE=e5-v
TEXT_ENCODER_TYPE=e5-v

python main.py \
  --task_config "${TASK_CONFIG}" \
  --model_config "${MODEL_CONFIG}" \
  --exp_name "${EXP_NAME}" \
  --query_mode "${QUERY_MODE}" \
  --model_name "${MODEL_NAME}" \
  --pretrained "${PRETRAINED}" \
  --image_encoder_type "${IMAGE_ENCODER_TYPE}" \
  --text_encoder_type "${TEXT_ENCODER_TYPE}"
