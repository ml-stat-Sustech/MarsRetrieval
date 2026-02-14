#!/usr/bin/env bash
# Usage: bash scripts/landform_retrieval/aimv2_vl.sh 0,1,2,3 (or export CUDA_VISIBLE_DEVICES beforehand)
export CUDA_VISIBLE_DEVICES=${1:-${CUDA_VISIBLE_DEVICES:-0}}
export PATH=~/.conda/envs/retrieval/bin:$PATH


TASK_CONFIG=LandformRetrieval
MODEL_CONFIG=AimV2VL
EXP_NAME=main_exp

QUERY_MODE=text

MODEL_NAME=apple/aimv2-large-patch14-224-lit
PRETRAINED=c2cd59a786c4c06f39d199c50d08cc2eab9f8605
IMAGE_ENCODER_TYPE=aimv2_vl
TEXT_ENCODER_TYPE=aimv2_vl

python main.py \
  --task_config "${TASK_CONFIG}" \
  --model_config "${MODEL_CONFIG}" \
  --exp_name "${EXP_NAME}" \
  --query_mode "${QUERY_MODE}" \
  --model_name "${MODEL_NAME}" \
  --pretrained "${PRETRAINED}" \
  --image_encoder_type "${IMAGE_ENCODER_TYPE}" \
  --text_encoder_type "${TEXT_ENCODER_TYPE}"
