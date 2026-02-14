#!/usr/bin/env bash
# Usage: bash scripts/landform_retrieval/gme.sh 0,1 (or export CUDA_VISIBLE_DEVICES beforehand)
export CUDA_VISIBLE_DEVICES=${1:-${CUDA_VISIBLE_DEVICES:-0}}
export PATH=~/.conda/envs/retrieval/bin:$PATH


TASK_CONFIG=LandformRetrieval
MODEL_CONFIG=GME
EXP_NAME=main_exp

QUERY_MODE=text

MODEL_NAME=Alibaba-NLP/gme-Qwen2-VL-2B-Instruct
PRETRAINED=gme
IMAGE_ENCODER_TYPE=gme
TEXT_ENCODER_TYPE=gme

python main.py \
  --task_config "${TASK_CONFIG}" \
  --model_config "${MODEL_CONFIG}" \
  --exp_name "${EXP_NAME}" \
  --query_mode "${QUERY_MODE}" \
  --model_name "${MODEL_NAME}" \
  --pretrained "${PRETRAINED}" \
  --image_encoder_type "${IMAGE_ENCODER_TYPE}" \
  --text_encoder_type "${TEXT_ENCODER_TYPE}"
